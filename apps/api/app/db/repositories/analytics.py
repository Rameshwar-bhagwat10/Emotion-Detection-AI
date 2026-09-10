"""Repository for database-level analytics aggregations, statistics, and history querying."""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Float, Integer, case, cast, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.analysis_session import AnalysisSession
from app.db.models.detected_face import DetectedFaceRecord
from app.db.models.prediction import PredictionRecord

SUPPORTED_EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]


def _calc_duration_seconds(start: datetime, end: datetime | None) -> float:
    """Safely calculate non-negative elapsed seconds between two timestamps handling offset-naive vs aware."""
    if start is None:
        return 0.0
    if end is None:
        end = datetime.now(UTC) if start.tzinfo is not None else datetime.utcnow()

    # Normalize tzinfo
    if start.tzinfo is not None and end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    elif start.tzinfo is None and end.tzinfo is not None:
        start = start.replace(tzinfo=UTC)

    diff = (end - start).total_seconds()
    return max(0.0, diff)


class AnalyticsRepository:
    """Performs SQL-level aggregations for session and global analytics."""

    async def get_global_overview(
        self,
        session: AsyncSession,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        model_version: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate global counts, expression distribution, confidence stats, and duration."""
        # 1. Total sessions
        sess_stmt = select(func.count(AnalysisSession.id))
        if start_date:
            sess_stmt = sess_stmt.where(AnalysisSession.started_at >= start_date)
        if end_date:
            sess_stmt = sess_stmt.where(AnalysisSession.started_at <= end_date)
        sess_result = await session.execute(sess_stmt)
        total_sessions = sess_result.scalar() or 0

        # Total duration in seconds across sessions
        dur_stmt = select(AnalysisSession.started_at, AnalysisSession.ended_at).where(
            AnalysisSession.ended_at.is_not(None)
        )
        if start_date:
            dur_stmt = dur_stmt.where(AnalysisSession.started_at >= start_date)
        if end_date:
            dur_stmt = dur_stmt.where(AnalysisSession.started_at <= end_date)
        dur_res = await session.execute(dur_stmt)
        total_duration_seconds = 0.0
        for s_start, s_end in dur_res.all():
            if s_start and s_end:
                total_duration_seconds += _calc_duration_seconds(s_start, s_end)

        # 2. Total predictions & faces count
        pred_filter = []
        if start_date:
            pred_filter.append(PredictionRecord.created_at >= start_date)
        if end_date:
            pred_filter.append(PredictionRecord.created_at <= end_date)
        if model_version:
            pred_filter.append(PredictionRecord.model_version == model_version)

        pred_stmt = select(func.count(PredictionRecord.id))
        if pred_filter:
            pred_stmt = pred_stmt.where(*pred_filter)
        pred_result = await session.execute(pred_stmt)
        total_predictions = pred_result.scalar() or 0

        # 3. Face predictions aggregate (counts, confidence stats, emotion distribution)
        face_filter = []
        if pred_filter:
            face_filter = [
                DetectedFaceRecord.prediction_id == PredictionRecord.id,
                *pred_filter,
            ]
            face_join = select(
                func.count(DetectedFaceRecord.id),
                func.avg(DetectedFaceRecord.confidence),
                func.min(DetectedFaceRecord.confidence),
                func.max(DetectedFaceRecord.confidence),
            ).join(PredictionRecord, DetectedFaceRecord.prediction_id == PredictionRecord.id)
            if pred_filter:
                face_join = face_join.where(*pred_filter)
            face_res = await session.execute(face_join)
        else:
            face_stmt = select(
                func.count(DetectedFaceRecord.id),
                func.avg(DetectedFaceRecord.confidence),
                func.min(DetectedFaceRecord.confidence),
                func.max(DetectedFaceRecord.confidence),
            )
            face_res = await session.execute(face_stmt)

        total_faces, avg_conf, min_conf, max_conf = face_res.one()
        total_faces = total_faces or 0
        avg_confidence = round(float(avg_conf or 0.0), 4)
        min_confidence = round(float(min_conf or 0.0), 4)
        max_confidence = round(float(max_conf or 0.0), 4)

        # 4. Expression counts & distribution
        if pred_filter:
            dist_stmt = (
                select(DetectedFaceRecord.emotion, func.count(DetectedFaceRecord.id))
                .join(PredictionRecord, DetectedFaceRecord.prediction_id == PredictionRecord.id)
                .where(*pred_filter)
                .group_by(DetectedFaceRecord.emotion)
            )
        else:
            dist_stmt = select(
                DetectedFaceRecord.emotion, func.count(DetectedFaceRecord.id)
            ).group_by(DetectedFaceRecord.emotion)

        dist_res = await session.execute(dist_stmt)
        raw_counts = {row[0].lower(): row[1] for row in dist_res.all()}

        # Ensure all supported emotions (and any database classes like 'uncertain') exist
        expression_items = []
        dominant_expression: str | None = None
        highest_count = -1

        all_emotions = sorted(list(set(SUPPORTED_EMOTIONS) | set(raw_counts.keys())))
        for emotion in all_emotions:
            cnt = raw_counts.get(emotion, 0)
            pct = round((cnt / total_faces * 100.0), 2) if total_faces > 0 else 0.0
            expression_items.append(
                {
                    "emotion": emotion,
                    "count": cnt,
                    "percentage": pct,
                }
            )
            # Deterministic tie-breaking: highest count, then alphabetical
            if cnt > highest_count:
                highest_count = cnt
                dominant_expression = emotion
            elif cnt == highest_count and dominant_expression and emotion < dominant_expression:
                dominant_expression = emotion

        if total_faces == 0 or highest_count == 0:
            dominant_expression = None

        # 5. Confidence bins histogram
        # 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
        bin_cases = [
            func.count(case((DetectedFaceRecord.confidence < 0.20, 1))),
            func.count(
                case(
                    (
                        (DetectedFaceRecord.confidence >= 0.20)
                        & (DetectedFaceRecord.confidence < 0.40),
                        1,
                    )
                )
            ),
            func.count(
                case(
                    (
                        (DetectedFaceRecord.confidence >= 0.40)
                        & (DetectedFaceRecord.confidence < 0.60),
                        1,
                    )
                )
            ),
            func.count(
                case(
                    (
                        (DetectedFaceRecord.confidence >= 0.60)
                        & (DetectedFaceRecord.confidence < 0.80),
                        1,
                    )
                )
            ),
            func.count(case((DetectedFaceRecord.confidence >= 0.80, 1))),
            func.count(
                case(
                    (
                        (DetectedFaceRecord.confidence < 0.60)
                        | (DetectedFaceRecord.is_uncertain == True),  # noqa: E712
                        1,
                    )
                )
            ),
            func.count(case((DetectedFaceRecord.confidence >= 0.80, 1))),
        ]

        if pred_filter:
            bin_stmt = (
                select(*bin_cases)
                .join(PredictionRecord, DetectedFaceRecord.prediction_id == PredictionRecord.id)
                .where(*pred_filter)
            )
        else:
            bin_stmt = select(*bin_cases)

        bin_res = await session.execute(bin_stmt)
        b0, b1, b2, b3, b4, low_conf_cnt, high_conf_cnt = bin_res.one()

        confidence_distribution = [
            {
                "range": "0-20%",
                "min_val": 0.0,
                "max_val": 0.20,
                "count": b0 or 0,
                "percentage": round(((b0 or 0) / total_faces * 100.0), 2)
                if total_faces > 0
                else 0.0,
            },
            {
                "range": "20-40%",
                "min_val": 0.20,
                "max_val": 0.40,
                "count": b1 or 0,
                "percentage": round(((b1 or 0) / total_faces * 100.0), 2)
                if total_faces > 0
                else 0.0,
            },
            {
                "range": "40-60%",
                "min_val": 0.40,
                "max_val": 0.60,
                "count": b2 or 0,
                "percentage": round(((b2 or 0) / total_faces * 100.0), 2)
                if total_faces > 0
                else 0.0,
            },
            {
                "range": "60-80%",
                "min_val": 0.60,
                "max_val": 0.80,
                "count": b3 or 0,
                "percentage": round(((b3 or 0) / total_faces * 100.0), 2)
                if total_faces > 0
                else 0.0,
            },
            {
                "range": "80-100%",
                "min_val": 0.80,
                "max_val": 1.00,
                "count": b4 or 0,
                "percentage": round(((b4 or 0) / total_faces * 100.0), 2)
                if total_faces > 0
                else 0.0,
            },
        ]

        # 6. Class-specific average confidences
        if pred_filter:
            cls_conf_stmt = (
                select(DetectedFaceRecord.emotion, func.avg(DetectedFaceRecord.confidence))
                .join(PredictionRecord, DetectedFaceRecord.prediction_id == PredictionRecord.id)
                .where(*pred_filter)
                .group_by(DetectedFaceRecord.emotion)
            )
        else:
            cls_conf_stmt = select(
                DetectedFaceRecord.emotion, func.avg(DetectedFaceRecord.confidence)
            ).group_by(DetectedFaceRecord.emotion)

        cls_res = await session.execute(cls_conf_stmt)
        class_confidences = {
            row[0].lower(): round(float(row[1]), 4) for row in cls_res.all() if row[1] is not None
        }
        for emo in SUPPORTED_EMOTIONS:
            if emo not in class_confidences:
                class_confidences[emo] = 0.0

        # 7. Model version distribution
        mv_stmt = select(
            PredictionRecord.model_version, func.count(PredictionRecord.id)
        ).group_by(PredictionRecord.model_version)
        if pred_filter:
            mv_stmt = mv_stmt.where(*pred_filter)
        mv_res = await session.execute(mv_stmt)
        model_version_distribution = {row[0]: row[1] for row in mv_res.all()}

        # 8. Historical daily trends
        trend_stmt = (
            select(
                func.date(PredictionRecord.created_at).label("date_str"),
                func.count(PredictionRecord.id).label("pred_count"),
                func.count(func.distinct(PredictionRecord.session_id)).label("sess_count"),
            )
            .group_by("date_str")
            .order_by("date_str")
            .limit(30)
        )
        trend_res = await session.execute(trend_stmt)
        recent_trends = [
            {
                "date": str(row[0]),
                "predictions_count": row[1],
                "sessions_count": row[2] or 0,
            }
            for row in trend_res.all()
        ]

        return {
            "total_sessions": total_sessions,
            "total_predictions": total_predictions,
            "total_faces": total_faces,
            "total_duration_seconds": round(total_duration_seconds, 1),
            "dominant_expression": dominant_expression,
            "average_confidence": avg_confidence,
            "expression_distribution": {
                "items": expression_items,
                "dominant_emotion": dominant_expression,
                "total_predictions": total_faces,
            },
            "confidence_analytics": {
                "average_confidence": avg_confidence,
                "min_confidence": min_confidence,
                "max_confidence": max_confidence,
                "distribution": confidence_distribution,
                "class_confidences": class_confidences,
                "low_confidence_count": low_conf_cnt or 0,
                "high_confidence_count": high_conf_cnt or 0,
            },
            "model_version_distribution": model_version_distribution,
            "recent_trends": recent_trends,
        }

    async def get_session_analytics(
        self,
        session: AsyncSession,
        session_id: uuid.UUID,
    ) -> dict[str, Any] | None:
        """Calculate detailed analytics strictly isolated to a single session."""
        # Check session exists
        sess_stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        sess_res = await session.execute(sess_stmt)
        analysis_session = sess_res.scalar_one_or_none()
        if analysis_session is None:
            return None

        # Duration
        duration_seconds = _calc_duration_seconds(
            analysis_session.started_at, analysis_session.ended_at
        )

        # Prediction count
        pred_cnt_stmt = select(func.count(PredictionRecord.id)).where(
            PredictionRecord.session_id == session_id
        )
        pred_cnt_res = await session.execute(pred_cnt_stmt)
        total_predictions = pred_cnt_res.scalar() or 0

        # Model versions used
        mv_stmt = (
            select(PredictionRecord.model_version)
            .where(PredictionRecord.session_id == session_id)
            .distinct()
        )
        mv_res = await session.execute(mv_stmt)
        model_versions = [row[0] for row in mv_res.all()]

        # Face predictions metrics
        face_agg_stmt = (
            select(
                func.count(DetectedFaceRecord.id),
                func.avg(DetectedFaceRecord.confidence),
                func.min(DetectedFaceRecord.confidence),
                func.max(DetectedFaceRecord.confidence),
            )
            .join(PredictionRecord, DetectedFaceRecord.prediction_id == PredictionRecord.id)
            .where(PredictionRecord.session_id == session_id)
        )
        face_res = await session.execute(face_agg_stmt)
        total_faces, avg_conf, min_conf, max_conf = face_res.one()
        total_faces = total_faces or 0
        avg_confidence = round(float(avg_conf or 0.0), 4)
        min_confidence = round(float(min_conf or 0.0), 4)
        max_confidence = round(float(max_conf or 0.0), 4)

        # Prediction rate (predictions per minute)
        duration_minutes = duration_seconds / 60.0 if duration_seconds > 0 else 0.0
        prediction_rate_per_minute = (
            round(total_predictions / duration_minutes, 1) if duration_minutes > 0 else 0.0
        )

        # Expression distribution
        dist_stmt = (
            select(DetectedFaceRecord.emotion, func.count(DetectedFaceRecord.id))
            .join(PredictionRecord, DetectedFaceRecord.prediction_id == PredictionRecord.id)
            .where(PredictionRecord.session_id == session_id)
            .group_by(DetectedFaceRecord.emotion)
        )
        dist_res = await session.execute(dist_stmt)
        raw_counts = {row[0].lower(): row[1] for row in dist_res.all()}

        expression_items = []
        dominant_expression: str | None = None
        highest_count = -1

        all_session_emotions = sorted(list(set(SUPPORTED_EMOTIONS) | set(raw_counts.keys())))
        for emotion in all_session_emotions:
            cnt = raw_counts.get(emotion, 0)
            pct = round((cnt / total_faces * 100.0), 2) if total_faces > 0 else 0.0
            expression_items.append(
                {
                    "emotion": emotion,
                    "count": cnt,
                    "percentage": pct,
                }
            )
            if cnt > highest_count:
                highest_count = cnt
                dominant_expression = emotion
            elif cnt == highest_count and dominant_expression and emotion < dominant_expression:
                dominant_expression = emotion

        if total_faces == 0 or highest_count == 0:
            dominant_expression = None

        # Confidence distribution bins
        bin_stmt = (
            select(
                func.count(case((DetectedFaceRecord.confidence < 0.20, 1))),
                func.count(
                    case(
                        (
                            (DetectedFaceRecord.confidence >= 0.20)
                            & (DetectedFaceRecord.confidence < 0.40),
                            1,
                        )
                    )
                ),
                func.count(
                    case(
                        (
                            (DetectedFaceRecord.confidence >= 0.40)
                            & (DetectedFaceRecord.confidence < 0.60),
                            1,
                        )
                    )
                ),
                func.count(
                    case(
                        (
                            (DetectedFaceRecord.confidence >= 0.60)
                            & (DetectedFaceRecord.confidence < 0.80),
                            1,
                        )
                    )
                ),
                func.count(case((DetectedFaceRecord.confidence >= 0.80, 1))),
                func.count(
                    case(
                        (
                            (DetectedFaceRecord.confidence < 0.60)
                            | (DetectedFaceRecord.is_uncertain == True),  # noqa: E712
                            1,
                        )
                    )
                ),
                func.count(case((DetectedFaceRecord.confidence >= 0.80, 1))),
            )
            .join(PredictionRecord, DetectedFaceRecord.prediction_id == PredictionRecord.id)
            .where(PredictionRecord.session_id == session_id)
        )
        bin_res = await session.execute(bin_stmt)
        b0, b1, b2, b3, b4, low_conf_cnt, high_conf_cnt = bin_res.one()

        confidence_distribution = [
            {
                "range": "0-20%",
                "min_val": 0.0,
                "max_val": 0.20,
                "count": b0 or 0,
                "percentage": round(((b0 or 0) / total_faces * 100.0), 2)
                if total_faces > 0
                else 0.0,
            },
            {
                "range": "20-40%",
                "min_val": 0.20,
                "max_val": 0.40,
                "count": b1 or 0,
                "percentage": round(((b1 or 0) / total_faces * 100.0), 2)
                if total_faces > 0
                else 0.0,
            },
            {
                "range": "40-60%",
                "min_val": 0.40,
                "max_val": 0.60,
                "count": b2 or 0,
                "percentage": round(((b2 or 0) / total_faces * 100.0), 2)
                if total_faces > 0
                else 0.0,
            },
            {
                "range": "60-80%",
                "min_val": 0.60,
                "max_val": 0.80,
                "count": b3 or 0,
                "percentage": round(((b3 or 0) / total_faces * 100.0), 2)
                if total_faces > 0
                else 0.0,
            },
            {
                "range": "80-100%",
                "min_val": 0.80,
                "max_val": 1.00,
                "count": b4 or 0,
                "percentage": round(((b4 or 0) / total_faces * 100.0), 2)
                if total_faces > 0
                else 0.0,
            },
        ]

        # Class confidences
        cls_conf_stmt = (
            select(DetectedFaceRecord.emotion, func.avg(DetectedFaceRecord.confidence))
            .join(PredictionRecord, DetectedFaceRecord.prediction_id == PredictionRecord.id)
            .where(PredictionRecord.session_id == session_id)
            .group_by(DetectedFaceRecord.emotion)
        )
        cls_res = await session.execute(cls_conf_stmt)
        class_confidences = {
            row[0].lower(): round(float(row[1]), 4) for row in cls_res.all() if row[1] is not None
        }
        for emo in SUPPORTED_EMOTIONS:
            if emo not in class_confidences:
                class_confidences[emo] = 0.0

        return {
            "session_id": analysis_session.id,
            "name": analysis_session.name,
            "status": analysis_session.status,
            "started_at": analysis_session.started_at,
            "ended_at": analysis_session.ended_at,
            "duration_seconds": round(duration_seconds, 1),
            "total_predictions": total_predictions,
            "total_faces": total_faces,
            "prediction_rate_per_minute": prediction_rate_per_minute,
            "dominant_expression": dominant_expression,
            "expression_distribution": {
                "items": expression_items,
                "dominant_emotion": dominant_expression,
                "total_predictions": total_faces,
            },
            "confidence_analytics": {
                "average_confidence": avg_confidence,
                "min_confidence": min_confidence,
                "max_confidence": max_confidence,
                "distribution": confidence_distribution,
                "class_confidences": class_confidences,
                "low_confidence_count": low_conf_cnt or 0,
                "high_confidence_count": high_conf_cnt or 0,
            },
            "model_versions": model_versions,
        }

    async def get_session_timeline(
        self,
        session: AsyncSession,
        session_id: uuid.UUID,
        bucket_seconds: int | None = None,
    ) -> dict[str, Any] | None:
        """Aggregate session predictions into time buckets for timeline visualization."""
        # 1. Check session
        sess_stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        sess_res = await session.execute(sess_stmt)
        analysis_session = sess_res.scalar_one_or_none()
        if analysis_session is None:
            return None

        # Fetch chronological predictions with their faces
        pred_stmt = (
            select(
                PredictionRecord.created_at,
                DetectedFaceRecord.emotion,
                DetectedFaceRecord.confidence,
                DetectedFaceRecord.is_uncertain,
            )
            .join(DetectedFaceRecord, DetectedFaceRecord.prediction_id == PredictionRecord.id)
            .where(PredictionRecord.session_id == session_id)
            .order_by(PredictionRecord.created_at.asc())
        )
        pred_res = await session.execute(pred_stmt)
        records = pred_res.all()

        if not records:
            return {
                "session_id": session_id,
                "bucket_seconds": bucket_seconds or 5,
                "total_buckets": 0,
                "buckets": [],
            }

        first_time = records[0][0]
        last_time = records[-1][0]
        span_seconds = max(1.0, (last_time - first_time).total_seconds())

        # Determine optimal bucket size if not provided
        if not bucket_seconds or bucket_seconds <= 0:
            if span_seconds <= 30:
                bucket_seconds = 1
            elif span_seconds <= 120:
                bucket_seconds = 2
            elif span_seconds <= 300:
                bucket_seconds = 5
            elif span_seconds <= 900:
                bucket_seconds = 10
            elif span_seconds <= 3600:
                bucket_seconds = 30
            else:
                bucket_seconds = 60

        total_buckets = max(1, math.ceil(span_seconds / bucket_seconds))
        # Limit buckets to max 120 for chart responsiveness
        if total_buckets > 120:
            bucket_seconds = math.ceil(span_seconds / 120)
            total_buckets = math.ceil(span_seconds / bucket_seconds)

        all_timeline_emotions = sorted(
            list(set(SUPPORTED_EMOTIONS) | {r[1].lower() for r in records})
        )
        buckets_map: dict[int, dict[str, Any]] = {}
        for b_idx in range(total_buckets):
            buckets_map[b_idx] = {
                "bucket_index": b_idx,
                "relative_seconds": b_idx * bucket_seconds,
                "timestamp": first_time.timestamp() + (b_idx * bucket_seconds),
                "prediction_count": 0,
                "emotion_counts": {emo: 0 for emo in all_timeline_emotions},
                "confidence_sum": 0.0,
                "confidence_count": 0,
            }

        for dt, emotion, confidence, is_uncertain in records:
            offset = max(0.0, (dt - first_time).total_seconds())
            b_idx = min(total_buckets - 1, int(offset // bucket_seconds))
            b = buckets_map[b_idx]
            b["prediction_count"] += 1
            emo_clean = emotion.lower()
            if emo_clean in b["emotion_counts"]:
                b["emotion_counts"][emo_clean] += 1
            b["confidence_sum"] += confidence
            b["confidence_count"] += 1

        # Format output
        output_buckets = []
        for b_idx in range(total_buckets):
            b = buckets_map[b_idx]
            cnt = b["prediction_count"]
            avg_conf = (
                round(b["confidence_sum"] / b["confidence_count"], 4)
                if b["confidence_count"] > 0
                else 0.0
            )

            # Dominant in this bucket
            dom_emo = None
            max_c = 0
            for emo, c in b["emotion_counts"].items():
                if c > max_c:
                    max_c = c
                    dom_emo = emo
                elif c == max_c and dom_emo and emo < dom_emo:
                    dom_emo = emo

            output_buckets.append(
                {
                    "bucket_index": b["bucket_index"],
                    "timestamp": datetime.fromtimestamp(b["timestamp"], tz=UTC).isoformat(),
                    "relative_seconds": b["relative_seconds"],
                    "prediction_count": cnt,
                    "emotion_counts": b["emotion_counts"],
                    "dominant_emotion": dom_emo,
                    "average_confidence": avg_conf,
                }
            )

        return {
            "session_id": session_id,
            "bucket_seconds": bucket_seconds,
            "total_buckets": len(output_buckets),
            "buckets": output_buckets,
        }

    async def list_prediction_history(
        self,
        session: AsyncSession,
        session_id: uuid.UUID | None = None,
        emotion: str | None = None,
        is_uncertain: bool | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        model_version: str | None = None,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "created_at",
        order: str = "desc",
    ) -> tuple[list[dict[str, Any]], int]:
        """Query historical prediction items with server-side filters and pagination."""
        filters = []
        if session_id:
            filters.append(PredictionRecord.session_id == session_id)
        if emotion:
            filters.append(func.lower(DetectedFaceRecord.emotion) == emotion.lower())
        if is_uncertain is not None:
            filters.append(DetectedFaceRecord.is_uncertain == is_uncertain)
        if min_confidence is not None:
            filters.append(DetectedFaceRecord.confidence >= min_confidence)
        if max_confidence is not None:
            filters.append(DetectedFaceRecord.confidence <= max_confidence)
        if start_date:
            filters.append(PredictionRecord.created_at >= start_date)
        if end_date:
            filters.append(PredictionRecord.created_at <= end_date)
        if model_version:
            filters.append(PredictionRecord.model_version == model_version)

        # Count matching
        count_stmt = (
            select(func.count(DetectedFaceRecord.id))
            .join(PredictionRecord, DetectedFaceRecord.prediction_id == PredictionRecord.id)
            .where(*filters)
        )
        total_res = await session.execute(count_stmt)
        total_count = total_res.scalar() or 0

        # Query records
        query = (
            select(
                PredictionRecord.id.label("pred_id"),
                PredictionRecord.session_id,
                PredictionRecord.request_id,
                PredictionRecord.created_at,
                PredictionRecord.model_version,
                PredictionRecord.processing_time_ms,
                DetectedFaceRecord.face_id,
                DetectedFaceRecord.emotion,
                DetectedFaceRecord.confidence,
                DetectedFaceRecord.is_uncertain,
                DetectedFaceRecord.bbox_x,
                DetectedFaceRecord.bbox_y,
                DetectedFaceRecord.bbox_width,
                DetectedFaceRecord.bbox_height,
                DetectedFaceRecord.probabilities,
            )
            .join(PredictionRecord, DetectedFaceRecord.prediction_id == PredictionRecord.id)
            .where(*filters)
        )

        # Sorting
        if sort_by == "confidence":
            sort_col = DetectedFaceRecord.confidence
        else:
            sort_col = PredictionRecord.created_at

        if order.lower() == "asc":
            query = query.order_by(sort_col.asc())
        else:
            query = query.order_by(sort_col.desc())

        query = query.limit(limit).offset(offset)
        rows = await session.execute(query)

        items = []
        for r in rows.all():
            items.append(
                {
                    "prediction_id": r.pred_id,
                    "session_id": r.session_id,
                    "request_id": r.request_id,
                    "timestamp": r.created_at,
                    "model_version": r.model_version,
                    "face_id": r.face_id,
                    "emotion": r.emotion,
                    "confidence": round(float(r.confidence), 4),
                    "is_uncertain": r.is_uncertain,
                    "bbox": {
                        "x": r.bbox_x,
                        "y": r.bbox_y,
                        "width": r.bbox_width,
                        "height": r.bbox_height,
                    },
                    "probabilities": r.probabilities
                    if isinstance(r.probabilities, dict)
                    else {},
                    "processing_time_ms": round(float(r.processing_time_ms), 2),
                }
            )

        return items, total_count

    async def list_sessions_with_analytics(
        self,
        session: AsyncSession,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """List sessions enriched with aggregated predictions count, dominant emotion, and avg confidence."""
        filters = []
        if status:
            filters.append(AnalysisSession.status == status)
        if start_date:
            filters.append(AnalysisSession.started_at >= start_date)
        if end_date:
            filters.append(AnalysisSession.started_at <= end_date)

        # Total count
        cnt_stmt = select(func.count(AnalysisSession.id)).where(*filters)
        total_res = await session.execute(cnt_stmt)
        total_sessions = total_res.scalar() or 0

        # Query sessions
        sess_stmt = (
            select(AnalysisSession)
            .where(*filters)
            .order_by(AnalysisSession.started_at.desc())
            .limit(limit)
            .offset(offset)
        )
        res = await session.execute(sess_stmt)
        sessions_list = res.scalars().all()

        results = []
        for s in sessions_list:
            # Aggregate for this session
            p_stmt = (
                select(
                    func.count(DetectedFaceRecord.id),
                    func.avg(DetectedFaceRecord.confidence),
                )
                .join(PredictionRecord, DetectedFaceRecord.prediction_id == PredictionRecord.id)
                .where(PredictionRecord.session_id == s.id)
            )
            p_res = await session.execute(p_stmt)
            p_cnt, p_avg = p_res.one()

            # Dominant expression
            dom_stmt = (
                select(DetectedFaceRecord.emotion, func.count(DetectedFaceRecord.id).label("cnt"))
                .join(PredictionRecord, DetectedFaceRecord.prediction_id == PredictionRecord.id)
                .where(PredictionRecord.session_id == s.id)
                .group_by(DetectedFaceRecord.emotion)
                .order_by(desc("cnt"), DetectedFaceRecord.emotion.asc())
                .limit(1)
            )
            dom_res = await session.execute(dom_stmt)
            dom_row = dom_res.first()
            dom_emotion = dom_row[0] if dom_row else None

            # Duration
            dur = _calc_duration_seconds(s.started_at, s.ended_at)

            results.append(
                {
                    "id": s.id,
                    "name": s.name or f"Session {str(s.id)[:8]}",
                    "status": s.status,
                    "started_at": s.started_at,
                    "ended_at": s.ended_at,
                    "duration_seconds": round(dur, 1),
                    "prediction_count": p_cnt or 0,
                    "dominant_expression": dom_emotion,
                    "average_confidence": round(float(p_avg or 0.0), 4),
                }
            )

        return results, total_sessions
