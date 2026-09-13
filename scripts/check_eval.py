import json
from pathlib import Path
from ml.inference.engine import EmotionInferenceEngine

engine = EmotionInferenceEngine(auto_load=True)
sample_dir = Path("data/Sample Images")
img_paths = sorted(sample_dir.glob("*.jpg"))
results = []
for p in img_paths:
    res = engine.predict_image(p)
    preds = [{"face_id": f.face_id, "emotion": f.emotion, "conf": round(f.confidence, 3)} for f in res.faces]
    results.append({"file": p.name, "faces": preds})

emotions = {}
total_faces = 0
for r in results:
    for f in r["faces"]:
        total_faces += 1
        emotions[f["emotion"]] = emotions.get(f["emotion"], 0) + 1

print(f"Total faces: {total_faces}")
print(f"Emotion distribution with bias fix alone: {emotions}")
for r in results:
    sads = [f for f in r["faces"] if f["emotion"] == "sad"]
    if sads:
        print(f"  {r['file']}: sad faces = {sads}")
