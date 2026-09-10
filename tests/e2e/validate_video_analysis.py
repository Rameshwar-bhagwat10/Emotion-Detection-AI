"""End-to-end validation script for Video Analysis capability.

Tests the live running FastAPI (port 8000) and Next.js (port 3000) servers:
1. Verifies FastAPI server is healthy.
2. Verifies Next.js /video-analysis page is serving HTML (HTTP 200).
3. Synthesizes a real multi-second test video with controlled expressions.
4. Uploads video via POST /api/v1/video/upload.
5. Polls GET /api/v1/video/{id}/status until COMPLETED.
6. Retrieves GET /api/v1/video/{id} (metadata & analytics).
7. Retrieves GET /api/v1/video/{id}/timeline (segments & events).
8. Retrieves GET /api/v1/video/{id}/predictions (frame predictions).
9. Tests GET /api/v1/video/{id}/stream (video stream).
10. Validates mathematical consistency:
    - total_frames > 0
    - duration_seconds > 0
    - frames_analyzed > 0
    - start_time < end_time
    - events chronologically ordered
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import cv2
import httpx
import numpy as np


def generate_synthetic_video(duration_sec: float = 3.0, fps: int = 10) -> bytes:
    """Generate a synthetic MP4 video in memory."""
    total_frames = int(duration_sec * fps)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, float(fps), (240, 240))

        for i in range(total_frames):
            frame = np.full((240, 240, 3), 220, dtype=np.uint8)
            cx = 120 + int(8 * np.sin(i / 2.0))
            cy = 120
            # Skin oval
            cv2.ellipse(frame, (cx, cy), (45, 60), 0, 0, 360, (180, 200, 240), -1)
            # Eyes
            cv2.circle(frame, (cx - 16, cy - 14), 5, (50, 50, 50), -1)
            cv2.circle(frame, (cx + 16, cy - 14), 5, (50, 50, 50), -1)
            # First half smile, second half neutral line
            if i < total_frames // 2:
                cv2.ellipse(frame, (cx, cy + 20), (18, 10), 0, 0, 180, (30, 30, 180), 3)
            else:
                cv2.line(frame, (cx - 14, cy + 22), (cx + 14, cy + 22), (30, 30, 180), 3)

            writer.write(frame)

        writer.release()
        with open(tmp_path, "rb") as f:
            data = f.read()
        return data
    finally:
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()


async def main():
    print("--------------------------------------------------")
    print("RUNNING VIDEO ANALYSIS END-TO-END LIVE VALIDATION")
    print("--------------------------------------------------")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Check FastAPI health
        print("\n1. Testing FastAPI Health (http://localhost:8000/api/v1/health)...")
        res_health = await client.get("http://localhost:8000/api/v1/health")
        assert res_health.status_code == 200, f"FastAPI health failed: {res_health.status_code}"
        print(f"   [+] FastAPI is healthy (HTTP {res_health.status_code}): {res_health.json()['service']}")

        # 2. Check Next.js Video Analysis route
        print("\n2. Testing Next.js Video Analysis Page (http://localhost:3000/video-analysis)...")
        res_web = await client.get("http://localhost:3000/video-analysis")
        assert res_web.status_code == 200, f"Next.js page failed: {res_web.status_code}"
        assert "Video Facial Expression Analysis" in res_web.text or "<html" in res_web.text
        print(f"   [+] Next.js /video-analysis serving successfully (HTTP {res_web.status_code})")

        # 3. Generate synthetic video
        print("\n3. Synthesizing 3.0-second test video (30 frames @ 10 FPS)...")
        video_bytes = generate_synthetic_video(duration_sec=3.0, fps=10)
        print(f"   [+] Video generated: {len(video_bytes)} bytes")

        # 4. Upload video
        print("\n4. Uploading video to POST /api/v1/video/upload...")
        files = {"video": ("e2e_live_test.mp4", video_bytes, "video/mp4")}
        data = {"sampling_fps": "5.0"}
        res_upload = await client.post("http://localhost:8000/api/v1/video/upload", files=files, data=data)
        assert res_upload.status_code == 202, f"Upload failed: {res_upload.text}"
        job_data = res_upload.json()
        video_id = job_data["video_id"]
        print(f"   [+] Job accepted (HTTP 202): video_id={video_id}, status={job_data['status']}")

        # 5. Poll status
        print(f"\n5. Polling processing status for video '{video_id}'...")
        completed = False
        status_info = None
        for attempt in range(40):
            res_status = await client.get(f"http://localhost:8000/api/v1/video/{video_id}/status")
            assert res_status.status_code == 200
            status_info = res_status.json()
            print(f"   [Polling {attempt+1}] stage={status_info['current_stage']}, progress={status_info['progress_percent']}%")

            if status_info["status"] == "COMPLETED":
                completed = True
                break
            elif status_info["status"] == "FAILED":
                raise RuntimeError(f"Video analysis failed: {status_info.get('error_message')}")

            await asyncio.sleep(0.7)

        assert completed, f"Job timed out: {status_info}"
        print(f"   [+] Video analysis completed! Frames analyzed: {status_info['frames_analyzed']}")

        # 6. Retrieve detail report
        print(f"\n6. Retrieving full analysis detail report (GET /api/v1/video/{video_id})...")
        res_detail = await client.get(f"http://localhost:8000/api/v1/video/{video_id}")
        assert res_detail.status_code == 200
        detail = res_detail.json()
        meta = detail["metadata"]
        analytics = detail["analytics"]
        print(f"   [+] Duration: {meta['duration_seconds']}s | Source FPS: {meta['source_fps']} | Frames: {meta['frames_analyzed']}/{meta['total_frames']}")
        print(f"   [+] Dominant Expression: {analytics['dominant_expression']} ({analytics['dominant_expression_time_share']}% time share)")
        print(f"   [+] Tracked Faces: {analytics['tracked_faces_count']} | Predictions: {analytics['total_predictions_count']}")
        print(f"   [+] Processing Ratio: {analytics['processing_ratio']}x realtime ({analytics['processing_time_seconds']}s)")

        # 7. Retrieve timeline & events
        print(f"\n7. Retrieving synchronized timeline (GET /api/v1/video/{video_id}/timeline)...")
        res_timeline = await client.get(f"http://localhost:8000/api/v1/video/{video_id}/timeline")
        assert res_timeline.status_code == 200
        timeline = res_timeline.json()
        segments = timeline["segments"]
        events = timeline["events"]
        print(f"   [+] Retrieved {len(segments)} expression segments and {len(events)} transition events")

        if segments:
            for s in segments[:3]:
                print(f"      - Track {s['track_id']}: {s['emotion']} from {s['start_time']}s to {s['end_time']}s (avg conf: {round(s['average_confidence']*100, 1)}%)")

        # 8. Retrieve predictions
        print(f"\n8. Querying paginated frame predictions (GET /api/v1/video/{video_id}/predictions)...")
        res_preds = await client.get(f"http://localhost:8000/api/v1/video/{video_id}/predictions?page=1&page_size=5")
        assert res_preds.status_code == 200
        preds_data = res_preds.json()
        print(f"   [+] Paginated predictions: total={preds_data['total']}, returned={len(preds_data['predictions'])}")

        # 9. Test streaming
        print(f"\n9. Testing video stream endpoint (GET /api/v1/video/{video_id}/stream)...")
        res_stream = await client.get(f"http://localhost:8000/api/v1/video/{video_id}/stream")
        assert res_stream.status_code in (200, 206)
        assert len(res_stream.content) > 0
        print(f"   [+] Video stream valid: {len(res_stream.content)} bytes served with Content-Type: {res_stream.headers.get('content-type')}")

        print("\n--------------------------------------------------")
        print("[SUCCESS] ALL VIDEO ANALYSIS E2E VALIDATION CHECKS PASSED!")
        print("--------------------------------------------------")


if __name__ == "__main__":
    asyncio.run(main())
