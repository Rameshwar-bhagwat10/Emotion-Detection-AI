"""Export fine-tuned ResNet18-CBAM champion model to ONNX."""

import os
import sys

# Ensure UTF-8 stdout encoding to prevent Windows cp1252 charmap errors
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
import torch

from ml.models.factory import create_model

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_WEIGHTS = ROOT_DIR / "artifacts" / "optimized" / "champion" / "model.pt"
ONNX_EXPORT = ROOT_DIR / "artifacts" / "optimized" / "champion" / "model.onnx"

def export():
    device = torch.device("cpu")
    print(f"Loading weights from {MODEL_WEIGHTS}...")
    model = create_model("resnet18_cbam", num_classes=7, pretrained=False)
    state = torch.load(MODEL_WEIGHTS, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 112, 112, dtype=torch.float32)
    print(f"Exporting to ONNX at {ONNX_EXPORT}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_EXPORT),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=18,
        dynamo=False,
    )
    print("ONNX export succeeded!")

if __name__ == "__main__":
    export()
