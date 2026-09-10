"""Repository for Video Analysis persistence, queries, and analytical aggregations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.video_analysis import (
    ExpressionSegmentRecord,
    VideoAnalysisRecord,
    VideoPredictionRecord,
    VideoTrackRecord,
)
from app.schemas.prediction import BoundingBoxSchema
from app.schemas.video_analysis import (
    ExpressionEventSchema,
    ExpressionSegmentSchema,
    VideoAnalyticsSummarySchema,
    VideoDistributionItemSchema,
    VideoMetadataSchema,
    VideoPredictionItemSchema,
    VideoTrackSchema,
)


class VideoAnalysisRepository:
    """Handles persistence and analytical queries for video analysis entities."""

    async def create(
        self,
        session: AsyncSession,
        video_id: uuid.UUID,
        filename: str,
        file_path: str,
        duration_seconds: float = 0.0,
        source_fps: float = 0.0,
        analysis_fps: float = 5.0,
        width: int = 0,
        height: int = 0,
        total_frames: int = 0,
        session_id: uuid.UUID | None = None,
    ) -> VideoAnalysisRecord:
        """Create and persist a new VideoAnalysisRecord."""
        record = VideoAnalysisRecord(
            id=video_id,
            session_id=session_id,
            filename=filename,
            file_path=file_path,
            status="QUEUED",
            current_stage="queued",
            progress_percent=0.0,
            duration_seconds=duration_seconds,
            source_fps=source_fps,
            analysis_fps=analysis_fps,
            width=width,
            height=height,
            total_frames=total_frames,
            frames_analyzed=0,
            processing_time_seconds=0.0,
        )
        session.add(record)
        await session.flush()
        return record

    async def get_by_id(
        self,
        session: AsyncSession,
        video_id: uuid.UUID,
    ) -> VideoAnalysisRecord | None:
        """Retrieve VideoAnalysisRecord by UUID."""
        stmt = (
            select(VideoAnalysisRecord)
            .options(
                selectinload(VideoAnalysisRecord.tracks),
            )
            .where(VideoAnalysisRecord.id == video_id)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_recent(
        self,
        session: AsyncSession,
        limit: int = 20,
    ) -> list[VideoAnalysisRecord]:
        """Retrieve recent video analysis records ordered by creation date."""
        stmt = (
            select(VideoAnalysisRecord)
            .order_by(desc(VideoAnalysisRecord.created_at))
            .limit(limit)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())


    async def update_progress(
        self,
        session: AsyncSession,
        video_id: uuid.UUID,
        progress_percent: float,
        frames_analyzed: int,
        current_stage: str | None = None,
    ) -> None:
        """Update progress metrics in database."""
        record = await session.get(VideoAnalysisRecord, video_id)
        if record:
            record.progress_percent = round(min(100.0, max(0.0, progress_percent)), 1)
            record.frames_analyzed = frames_analyzed
            if current_stage:
                record.current_stage = current_stage
            await session.flush()

    async def update_status(
        self,
        session: AsyncSession,
        video_id: uuid.UUID,
        status: str,
        current_stage: str,
        error_message: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        processing_time_seconds: float | None = None,
    ) -> None:
        """Update lifecycle status and execution metrics."""
        record = await session.get(VideoAnalysisRecord, video_id)
        if record:
            record.status = status
            record.current_stage = current_stage
            if error_message is not None:
                record.error_message = error_message
            if started_at is not None:
                record.started_at = started_at
            if completed_at is not None:
                record.completed_at = completed_at
            if processing_time_seconds is not None:
                record.processing_time_seconds = processing_time_seconds
            await session.flush()

    async def save_results(
        self,
        session: AsyncSession,
        video_id: uuid.UUID,
        tracks: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> None:
        """Batch save tracks, expression segments, and sampled frame predictions."""
        # 1. Tracks
        for t in tracks:
            track_rec = VideoTrackRecord(
                id=uuid.uuid4(),
                video_analysis_id=video_id,
                track_id=t["track_id"],
                first_seen_sec=t["first_seen_sec"],
                last_seen_sec=t["last_seen_sec"],
                total_detections=t.get("total_detections", 0),
            )
            session.add(track_rec)

        # 2. Segments
        for s in segments:
            seg_rec = ExpressionSegmentRecord(
                id=uuid.uuid4(),
                video_analysis_id=video_id,
                track_id=s["track_id"],
                emotion=s["emotion"],
                start_time=s["start_time"],
                end_time=s["end_time"],
                duration=s["duration"],
                average_confidence=s["average_confidence"],
                min_confidence=s["min_confidence"],
                max_confidence=s["max_confidence"],
                prediction_count=s["prediction_count"],
            )
            session.add(seg_rec)

        # 3. Predictions
        for p in predictions:
            pred_rec = VideoPredictionRecord(
                id=uuid.uuid4(),
                video_analysis_id=video_id,
                track_id=p["track_id"],
                timestamp=p["timestamp"],
                frame_index=p["frame_index"],
                bbox_x=p["bbox_x"],
                bbox_y=p["bbox_y"],
                bbox_width=p["bbox_width"],
                bbox_height=p["bbox_height"],
                detection_confidence=p["detection_confidence"],
                raw_emotion=p["raw_emotion"],
                raw_confidence=p["raw_confidence"],
                smoothed_emotion=p["smoothed_emotion"],
                smoothed_confidence=p["smoothed_confidence"],
                is_uncertain=p["is_uncertain"],
                probabilities=p["probabilities"],
                raw_probabilities=p["raw_probabilities"],
            )
            session.add(pred_rec)

        await session.flush()

    async def get_timeline(
        self,
        session: AsyncSession,
        video_id: uuid.UUID,
        track_id: int | None = None,
    ) -> tuple[list[ExpressionSegmentSchema], list[ExpressionEventSchema]]:
        """Retrieve chronological expression segments and compute transition events."""
        stmt = (
            select(ExpressionSegmentRecord)
            .where(ExpressionSegmentRecord.video_analysis_id == video_id)
            .order_by(ExpressionSegmentRecord.track_id, ExpressionSegmentRecord.start_time)
        )
        if track_id is not None:
            stmt = stmt.where(ExpressionSegmentRecord.track_id == track_id)

        result = await session.execute(stmt)
        records = result.scalars().all()

        segments: list[ExpressionSegmentSchema] = []
        events: list[ExpressionEventSchema] = []

        # Group by track to identify transitions
        by_track: dict[int, list[ExpressionSegmentRecord]] = {}
        for r in records:
            segments.append(
                ExpressionSegmentSchema(
                    id=str(r.id),
                    track_id=r.track_id,
                    emotion=r.emotion,
                    start_time=r.start_time,
                    end_time=r.end_time,
                    duration=r.duration,
                    average_confidence=r.average_confidence,
                    min_confidence=r.min_confidence,
                    max_confidence=r.max_confidence,
                    prediction_count=r.prediction_count,
                )
            )
            by_track.setdefault(r.track_id, []).append(r)

        # Build chronological transition events between adjacent segments
        for t_id, segs in by_track.items():
            for i in range(len(segs) - 1):
                prev_seg = segs[i]
                next_seg = segs[i + 1]
                if prev_seg.emotion != next_seg.emotion:
                    events.append(
                        ExpressionEventSchema(
                            track_id=t_id,
                            timestamp=next_seg.start_time,
                            from_emotion=prev_seg.emotion,
                            to_emotion=next_seg.emotion,
                            confidence=next_seg.average_confidence,
                        )
                    )

        # Sort all events chronologically
        events.sort(key=lambda e: e.timestamp)
        return segments, events

    async def get_predictions(
        self,
        session: AsyncSession,
        video_id: uuid.UUID,
        track_id: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[VideoPredictionItemSchema], int]:
        """Query paginated frame-level predictions."""
        base_query = select(VideoPredictionRecord).where(
            VideoPredictionRecord.video_analysis_id == video_id
        )
        count_query = select(func.count(VideoPredictionRecord.id)).where(
            VideoPredictionRecord.video_analysis_id == video_id
        )

        if track_id is not None:
            base_query = base_query.where(VideoPredictionRecord.track_id == track_id)
            count_query = count_query.where(VideoPredictionRecord.track_id == track_id)

        if start_time is not None:
            base_query = base_query.where(VideoPredictionRecord.timestamp >= start_time)
            count_query = count_query.where(VideoPredictionRecord.timestamp >= start_time)

        if end_time is not None:
            base_query = base_query.where(VideoPredictionRecord.timestamp <= end_time)
            count_query = count_query.where(VideoPredictionRecord.timestamp <= end_time)

        total_res = await session.execute(count_query)
        total = total_res.scalar() or 0

        stmt = (
            base_query.order_by(VideoPredictionRecord.timestamp)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await session.execute(stmt)
        records = result.scalars().all()

        items: list[VideoPredictionItemSchema] = []
        for r in records:
            items.append(
                VideoPredictionItemSchema(
                    id=str(r.id),
                    track_id=r.track_id,
                    timestamp=r.timestamp,
                    frame_index=r.frame_index,
                    bbox=BoundingBoxSchema(
                        x=r.bbox_x,
                        y=r.bbox_y,
                        width=r.bbox_width,
                        height=r.bbox_height,
                    ),
                    detection_confidence=r.detection_confidence,
                    raw_emotion=r.raw_emotion,
                    raw_confidence=r.raw_confidence,
                    smoothed_emotion=r.smoothed_emotion,
                    smoothed_confidence=r.smoothed_confidence,
                    is_uncertain=r.is_uncertain,
                    probabilities=r.probabilities or {},
                    raw_probabilities=r.raw_probabilities or {},
                )
            )

        return items, total

    async def get_analytics_summary(
        self,
        session: AsyncSession,
        video_id: uuid.UUID,
    ) -> VideoAnalyticsSummarySchema | None:
        """Aggregate video-level analytics from segments, predictions, and tracks."""
        record = await session.get(VideoAnalysisRecord, video_id)
        if not record or record.status != "COMPLETED":
            return None

        # 1. Total tracks
        tracks_res = await session.execute(
            select(func.count(VideoTrackRecord.id)).where(
                VideoTrackRecord.video_analysis_id == video_id
            )
        )
        tracked_faces_count = tracks_res.scalar() or 0

        # 2. Total predictions count & avg confidence
        preds_stmt = select(
            func.count(VideoPredictionRecord.id),
            func.avg(VideoPredictionRecord.smoothed_confidence),
        ).where(VideoPredictionRecord.video_analysis_id == video_id)
        preds_res = await session.execute(preds_stmt)
        total_preds, avg_conf = preds_res.one()
        total_preds = total_preds or 0
        avg_conf = float(avg_conf or 0.0)

        # 3. Segments aggregation for Time Share
        segs_stmt = select(
            ExpressionSegmentRecord.emotion,
            func.count(ExpressionSegmentRecord.id),
            func.sum(ExpressionSegmentRecord.duration),
            func.sum(ExpressionSegmentRecord.prediction_count),
        ).where(ExpressionSegmentRecord.video_analysis_id == video_id).group_by(
            ExpressionSegmentRecord.emotion
        )
        segs_res = await session.execute(segs_stmt)
        seg_rows = segs_res.all()

        total_tracked_time = sum(float(r[2] or 0.0) for r in seg_rows)

        # 4. Expression count from predictions
        counts_stmt = select(
            VideoPredictionRecord.smoothed_emotion,
            func.count(VideoPredictionRecord.id),
        ).where(VideoPredictionRecord.video_analysis_id == video_id).group_by(
            VideoPredictionRecord.smoothed_emotion
        )
        counts_res = await session.execute(counts_stmt)
        pred_counts = dict(counts_res.all())

        distribution: list[VideoDistributionItemSchema] = []
        emotions_seen = set(pred_counts.keys()) | {r[0] for r in seg_rows}
        # Standard emotion ordering
        standard_order = ["happy", "neutral", "sad", "surprise", "fear", "angry", "disgust", "uncertain"]
        all_ordered = [e for e in standard_order if e in emotions_seen] + [
            e for e in sorted(emotions_seen) if e not in standard_order
        ]

        seg_duration_map = {r[0]: float(r[2] or 0.0) for r in seg_rows}

        dominant_emotion = "neutral"
        max_time_share = -1.0

        for em in all_ordered:
            c = pred_counts.get(em, 0)
            p = round((c / total_preds * 100.0), 2) if total_preds > 0 else 0.0
            dur = round(seg_duration_map.get(em, 0.0), 2)
            ts = round((dur / total_tracked_time * 100.0), 2) if total_tracked_time > 0 else 0.0

            if ts > max_time_share and em != "uncertain":
                max_time_share = ts
                dominant_emotion = em

            distribution.append(
                VideoDistributionItemSchema(
                    emotion=em,
                    count=c,
                    percentage=p,
                    time_seconds=dur,
                    time_share_percent=ts,
                )
            )

        # If all were uncertain, fallback
        if max_time_share < 0 and distribution:
            dominant_emotion = distribution[0].emotion
            max_time_share = distribution[0].time_share_percent

        # 5. Transitions count & matrix
        _, events = await self.get_timeline(session, video_id)
        transition_counts: dict[str, int] = {}
        for ev in events:
            key = f"{ev.from_emotion}->{ev.to_emotion}"
            transition_counts[key] = transition_counts.get(key, 0) + 1

        proc_time = record.processing_time_seconds or 0.0
        dur_sec = record.duration_seconds or 1.0
        proc_ratio = round(proc_time / dur_sec, 2) if dur_sec > 0 else 0.0

        return VideoAnalyticsSummarySchema(
            duration_seconds=round(record.duration_seconds, 2),
            tracked_faces_count=tracked_faces_count,
            total_predictions_count=total_preds,
            total_transitions_count=len(events),
            dominant_expression=dominant_emotion,
            dominant_expression_time_share=max(0.0, max_time_share),
            average_confidence=round(avg_conf, 4),
            processing_time_seconds=round(proc_time, 2),
            processing_ratio=proc_ratio,
            analysis_fps=round(record.analysis_fps, 1),
            expression_distribution=distribution,
            transition_counts=transition_counts,
        )
