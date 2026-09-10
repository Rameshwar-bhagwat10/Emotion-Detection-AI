"""End-to-End validation specifically for the user video:
WIN_20260910_23_49_54_Pro.mp4
"""

import json
import urllib.request
import os
from pathlib import Path

USER_VIDEO_PATH = r"C:\Users\RAMESHWAR\OneDrive\Videos\Screen Recordings\WIN_20260910_23_49_54_Pro.mp4"
BASE_API = "http://localhost:8000"
BASE_WEB = "http://localhost:3000"


def test_user_video_e2e():
    # 1. Verify file exists
    assert os.path.exists(USER_VIDEO_PATH), f"Video file not found at {USER_VIDEO_PATH}"
    size_mb = os.path.getsize(USER_VIDEO_PATH) / (1024 * 1024)
    print(f"[1/6] User video confirmed: {size_mb:.2f} MB")

    # 2. Check Next.js video-analysis page
    req_web = urllib.request.Request(f"{BASE_WEB}/video-analysis")
    with urllib.request.urlopen(req_web) as res:
        assert res.status == 200
        html = res.read().decode()
        assert "Video Facial Expression Analysis" in html
        print("[2/6] Next.js /video-analysis page is serving HTTP 200")

    # 3. Check /api/v1/video/recent
    req_recent = urllib.request.Request(f"{BASE_API}/api/v1/video/recent")
    with urllib.request.urlopen(req_recent) as res:
        assert res.status == 200
        recent = json.loads(res.read().decode())
        print(f"[3/6] Retrieved {len(recent)} recent video records")
        user_records = [r for r in recent if "WIN_20260910_23_49_54_Pro" in r.get("filename", "")]
        assert len(user_records) > 0, "No records found for WIN_20260910_23_49_54_Pro.mp4"
        latest = user_records[0]
        video_id = latest["video_id"]
        assert latest["status"] == "COMPLETED"
        print(f"      Matched user video record: id={video_id}, status={latest['status']}, frames={latest['frames_analyzed']}")

    # 4. Fetch detail and timeline for this video
    with urllib.request.urlopen(f"{BASE_API}/api/v1/video/{video_id}") as res:
        assert res.status == 200
        detail = json.loads(res.read().decode())
        assert detail["status"] == "COMPLETED"
        analytics = detail["analytics"]
        assert analytics is not None
        print(f"[4/6] Video Analysis Detail: dominant={analytics['dominant_expression']} ({analytics['dominant_expression_time_share']:.1f}%), tracks={analytics['tracked_faces_count']}, avg_conf={analytics['average_confidence']:.2f}")

    with urllib.request.urlopen(f"{BASE_API}/api/v1/video/{video_id}/timeline") as res:
        assert res.status == 200
        timeline = json.loads(res.read().decode())
        segments = timeline["segments"]
        events = timeline["events"]
        print(f"      Timeline: {len(segments)} segments, {len(events)} transitions")

    # 5. Fetch predictions
    with urllib.request.urlopen(f"{BASE_API}/api/v1/video/{video_id}/predictions?page=1&page_size=500") as res:
        assert res.status == 200
        preds = json.loads(res.read().decode())
        print(f"[5/6] Sampled predictions loaded: {len(preds['predictions'])} frame evaluations")

    # 6. Test Stream with Range request
    req_stream = urllib.request.Request(
        f"{BASE_API}/api/v1/video/{video_id}/stream",
        headers={"Range": "bytes=0-1048576"},
    )
    with urllib.request.urlopen(req_stream) as res:
        assert res.status in (200, 206)
        content_type = res.headers.get("Content-Type")
        assert "video" in content_type
        content_range = res.headers.get("Content-Range")
        print(f"[6/6] Stream verified: HTTP {res.status}, Content-Type: {content_type}, Content-Range: {content_range}")

    print("\nSUCCESS: All user video analysis checks passed perfectly!")


if __name__ == "__main__":
    test_user_video_e2e()
