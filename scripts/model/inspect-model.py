"""Inspect Baseline CNN model architecture, parameter counts, and tensor dimensions.

CLI utility for inspecting model layers, weight shapes, and executing a dummy forward pass.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add root directory to pythonpath
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch  # noqa: E402

from ml.models.factory import create_model  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inspect-model")


def inspect_model(
    model_name: str = "baseline_cnn", config_path: str = "ml/configs/models.yaml"
) -> None:
    """Inspect model architecture, parameters, shape flow, and dummy forward pass."""
    logger.info("Instantiating model '%s' from '%s'...", model_name, config_path)
    model = create_model(model_name=model_name, config_path=config_path)
    model.eval()

    summary = getattr(model, "get_model_summary", lambda: {})()
    param_counts = getattr(model, "get_parameter_count", lambda: {})()

    print("\n" + "=" * 60)
    print(f"MODEL INSPECTION REPORT: {model_name.upper()}")
    print("=" * 60)
    print(f"Model Name:              {summary.get('model_name', model_name)}")
    print(f"Model Version:           {summary.get('model_version', 'N/A')}")
    print(f"Input Contract:          {summary.get('input_shape', '[B, 1, 48, 48]')}")
    print(f"Output Contract:         {summary.get('output_shape', '[B, 7]')} (Raw Logits)")
    print(
        f"Emotion Classes ({summary.get('num_classes', 7)}):      {', '.join(summary.get('class_names', []))}"
    )
    print("-" * 60)
    print("PARAMETER SUMMARY:")
    print(f"  Total Parameters:      {param_counts.get('total', 0):,}")
    print(f"  Trainable Parameters:  {param_counts.get('trainable', 0):,}")
    print(f"  Non-Trainable:         {param_counts.get('non_trainable', 0):,}")
    print("-" * 60)
    print("LAYER-BY-LAYER SHAPE TRACING (Input: [1, 1, 48, 48]):")

    if hasattr(model, "trace_shapes"):
        traces = model.trace_shapes((1, 1, 48, 48))
        for layer_name, shape in traces:
            print(f"  -> {layer_name:<30} {str(shape):<20}")

    print("-" * 60)
    print("DUMMY FORWARD PASS CHECK:")
    dummy_input = torch.randn(4, 1, 48, 48, dtype=torch.float32)
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"  Input Batch Shape:     {tuple(dummy_input.shape)} (dtype: {dummy_input.dtype})")
    print(f"  Output Logits Shape:   {tuple(logits.shape)} (dtype: {logits.dtype})")
    print(f"  Logits Min / Max:      {logits.min().item():.4f} / {logits.max().item():.4f}")
    print(f"  Logits Mean / Std:     {logits.mean().item():.4f} / {logits.std().item():.4f}")
    print(
        f"  NaNs / Infs:           {int(torch.isnan(logits).sum().item())} / {int(torch.isinf(logits).sum().item())}"
    )
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect model architecture and parameter counts.")
    parser.add_argument("--model", type=str, default="baseline_cnn", help="Model name in registry")
    parser.add_argument(
        "--config", type=str, default="ml/configs/models.yaml", help="Path to models.yaml"
    )
    args = parser.parse_args()

    inspect_model(model_name=args.model, config_path=args.config)


if __name__ == "__main__":
    main()
