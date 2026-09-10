"""Complete End-to-End Validation Script for Phase 13 Analytics, History & Explainability."""

import io
import json
import urllib.request
import urllib.parse
import uuid
import numpy as np
from PIL import Image

API_BASE = "http://localhost:8000/api/v1"
WEB_BASE = "http://localhost:3000"

def create_synthetic_face() -> bytes:
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = (200, 200, 200)
    for y in range(300):
        for x in range(300):
            if ((x - 150) / 70) ** 2 + ((y - 150) / 90) ** 2 <= 1.0:
                img[y, x] = (160, 190, 230)
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def http_json(url: str, method: str = "GET", data: dict = None) -> dict:
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", "application/json")
    body = json.dumps(data).encode("utf-8") if data else None
    with urllib.request.urlopen(req, data=body) as res:
        return json.loads(res.read().decode("utf-8"))

def main():
    print("=" * 60)
    print("PHASE 13 END-TO-END VALIDATION SUITE")
    print("=" * 60)

    # 1. Health & Readiness
    health = http_json(f"{API_BASE}/health")
    print(f"[OK] Health Check: status={health['status']}, service={health['service']}")
    ready = http_json(f"{API_BASE}/health/ready")
    print(f"[OK] Readiness Check: status={ready['status']}, model={ready['model']}, database={ready['database']}")

    # 2. Create Analysis Session
    sess = http_json(f"{API_BASE}/sessions", method="POST", data={"name": "E2E Phase 13 Final Verification Session"})
    session_id = sess["id"]
    print(f"[OK] Created Analysis Session: {session_id} ('{sess['name']}')")

    # 3. Post Predictions into this Session
    from pathlib import Path
    sample_path = Path(__file__).resolve().parent.parent.parent / "reports" / "random_and_internet_eval" / "internet_1_happy_smile_orig.jpg"
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            img_bytes = f.read()
    else:
        img_bytes = create_synthetic_face()

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body_parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"session_id\"\r\n\r\n{session_id}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"face.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode(),
        img_bytes,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    req = urllib.request.Request(f"{API_BASE}/predictions", method="POST", data=b"".join(body_parts))
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req) as res:
        pred_res = json.loads(res.read().decode("utf-8"))
    
    print(f"[OK] Ingested Image Prediction: status={pred_res['status']}, faces={pred_res['faces_detected']}, latency={pred_res['timing']['total_ms']}ms")
    prediction_id = pred_res["prediction_id"]

    # 4. Finalize Session
    end_sess = http_json(f"{API_BASE}/sessions/{session_id}/end", method="POST")
    print(f"[OK] Finalized Session: status={end_sess['status']}, duration={end_sess.get('duration_seconds', 'N/A')}")

    # 5. Verify Global Analytics
    global_analytics = http_json(f"{API_BASE}/analytics/overview")
    print(f"[OK] Global Analytics Overview:")
    print(f"   - Total Sessions: {global_analytics['total_sessions']}")
    print(f"   - Total Predictions: {global_analytics['total_predictions']}")
    print(f"   - Total Faces: {global_analytics['total_faces']}")
    print(f"   - Dominant Expression: {global_analytics['dominant_expression']}")
    print(f"   - Average Confidence: {global_analytics['average_confidence'] * 100:.1f}%")
    print(f"   - Expression Classes: {[it['emotion'] for it in global_analytics['expression_distribution']['items']]}")

    # Mathematical consistency assertion
    dist_items = global_analytics["expression_distribution"]["items"]
    sum_counts = sum(it["count"] for it in dist_items)
    assert sum_counts == global_analytics["total_faces"], f"Sum of counts ({sum_counts}) != total faces ({global_analytics['total_faces']})"
    print(f"[OK] Mathematical Consistency Verified: sum(expression counts) == total faces ({sum_counts})")

    # 6. Verify Session Analytics
    session_analytics = http_json(f"{API_BASE}/analytics/sessions/{session_id}")
    print(f"[OK] Session Analytics (#{session_id[:8]}):")
    print(f"   - Status: {session_analytics['status']}")
    print(f"   - Duration: {session_analytics['duration_seconds']}s")
    print(f"   - Predictions: {session_analytics['total_predictions']}")
    print(f"   - Dominant Emotion: {session_analytics['dominant_expression']}")
    print(f"   - Average Confidence: {session_analytics['confidence_analytics']['average_confidence'] * 100:.1f}%")
    print(f"   - Model Versions: {session_analytics['model_versions']}")

    # 7. Verify Timeline Bucketing
    timeline = http_json(f"{API_BASE}/analytics/sessions/{session_id}/timeline")
    print(f"[OK] Session Timeline:")
    print(f"   - Bucket Interval: {timeline['bucket_seconds']}s")
    print(f"   - Total Buckets: {timeline['total_buckets']}")

    # 8. Verify Prediction History with Filters
    history = http_json(f"{API_BASE}/history/predictions?session_id={session_id}&limit=10")
    print(f"[OK] Prediction History for Session: {history['total']} records found")
    first_pred = history["items"][0]
    print(f"   - Record ID: {first_pred['prediction_id']}")
    print(f"   - Predicted Emotion: {first_pred['emotion']} ({first_pred['confidence']*100:.1f}%)")
    print(f"   - Bounding Box: {first_pred['bbox']}")
    print(f"   - Top-K Probabilities: {list(first_pred['probabilities'].keys())}")

    # 9. Verify Next.js Web Pages
    with urllib.request.urlopen(f"{WEB_BASE}/analytics") as res:
        print(f"[OK] Next.js /analytics Page Reachable: HTTP {res.status}")
    with urllib.request.urlopen(f"{WEB_BASE}/history") as res:
        print(f"[OK] Next.js /history Page Reachable: HTTP {res.status}")
    with urllib.request.urlopen(f"{WEB_BASE}/sessions/{session_id}") as res:
        print(f"[OK] Next.js /sessions/{session_id} Page Reachable: HTTP {res.status}")

    print("=" * 60)
    print("ALL PHASE 13 WORKFLOW CHECKS PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
