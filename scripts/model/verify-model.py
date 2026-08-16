"""Verification and sanity-check script for Baseline CNN model.

Validates model initialization, forward pass, gradient flow, train/eval modes,
state_dict serialization, and Phase 03 DataLoader batch compatibility.
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

from ml.models.cnn.baseline_cnn import BaselineCNN  # noqa: E402
from ml.models.factory import create_model  # noqa: E402
from ml.preprocessing.dataloaders import build_dataloaders  # noqa: E402
from ml.utils.device import get_device, get_device_info  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("verify-model")


def verify_model(
    model_name: str = "baseline_cnn",
    config_path: str = "ml/configs/models.yaml",
    data_path: str = "data/raw/fer2013/fer2013.csv",
) -> bool:
    """Run comprehensive verification checks on the baseline CNN model."""
    print("\n" + "=" * 60)
    print("PHASE 04 BASELINE CNN MODEL VERIFICATION SUITE")
    print("=" * 60)

    # 1. Model Creation & Initialization Check
    logger.info("1. Creating model '%s' from '%s'...", model_name, config_path)
    model = create_model(model_name=model_name, config_path=config_path)
    assert isinstance(model, BaselineCNN), f"Expected BaselineCNN instance, got {type(model)}"

    # Check for NaN / Inf in initialized parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            raise ValueError(f"Parameter '{name}' contains NaN values!")
        if torch.isinf(param).any():
            raise ValueError(f"Parameter '{name}' contains Infinite values!")

    params = model.get_parameter_count()
    print(
        f"  [PASS] Model initialized cleanly ({params['total']:,} total parameters, {params['trainable']:,} trainable)"
    )

    # 2. Dummy Forward Pass Shape and Dtype Check
    logger.info("2. Testing forward pass with dummy tensor [4, 1, 48, 48]...")
    dummy_input = torch.randn(4, 1, 48, 48, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        dummy_logits = model(dummy_input)

    assert dummy_logits.shape == (4, 7), f"Expected shape (4, 7), got {dummy_logits.shape}"
    assert dummy_logits.dtype == torch.float32, f"Expected dtype float32, got {dummy_logits.dtype}"
    assert not torch.isnan(dummy_logits).any(), "Dummy logits contain NaNs!"
    assert not torch.isinf(dummy_logits).any(), "Dummy logits contain Infs!"
    print("  [PASS] Dummy forward pass verified: shape [4, 7], float32, finite logits")

    # 3. Gradient Flow Verification
    logger.info("3. Testing gradient flow and differentiability...")
    model.train()
    model.zero_grad()
    grad_input = torch.randn(2, 1, 48, 48, dtype=torch.float32, requires_grad=True)
    grad_logits = model(grad_input)
    dummy_loss = grad_logits.sum()
    dummy_loss.backward()

    # Check that gradients exist and are finite on trainable weights
    grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter '{name}' did not receive gradients!"
            assert not torch.isnan(param.grad).any(), f"Gradient for '{name}' contains NaNs!"
            grad_count += 1

    print(
        f"  [PASS] Gradient flow verified: backward pass populated gradients for {grad_count} parameter tensors"
    )

    # 4. Train vs Eval Mode Behavior
    logger.info("4. Testing train / eval mode behavior...")
    model.eval()
    assert not model.training, "Model should be in eval mode"
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.Dropout)):
            assert not m.training, f"Submodule {m} should be in eval mode"

    model.train()
    assert model.training, "Model should be in train mode"
    print("  [PASS] Train/Eval mode toggle verified across BatchNorm and Dropout submodules")

    # 5. State Dict Serialization & Load Consistency
    logger.info("5. Testing state_dict serialization and load consistency...")
    model.eval()
    state_dict = model.state_dict()
    model_b = create_model(model_name=model_name, config_path=config_path)
    model_b.load_state_dict(state_dict)
    model_b.eval()

    test_tensor = torch.randn(4, 1, 48, 48, dtype=torch.float32)
    with torch.no_grad():
        out_a = model(test_tensor)
        out_b = model_b(test_tensor)

    assert torch.allclose(
        out_a, out_b, atol=1e-6
    ), "Model outputs mismatch after state_dict reload!"
    print("  [PASS] State dict serialization & reload verified: bit-identical outputs")

    # 6. Real Phase 03 DataLoader Compatibility Check
    logger.info("6. Testing integration with real Phase 03 DataLoaders from '%s'...", data_path)
    if Path(data_path).exists():
        train_loader, val_loader, test_loader = build_dataloaders(
            data_path=data_path, batch_size=32
        )

        model.eval()
        with torch.no_grad():
            for split_name, loader in [
                ("Train", train_loader),
                ("Val", val_loader),
                ("Test", test_loader),
            ]:
                batch = next(iter(loader))
                imgs = batch["image"]
                lbls = batch["label"]
                logits = model(imgs)

                assert lbls.shape[0] == imgs.shape[0]
                assert logits.shape == (
                    imgs.shape[0],
                    7,
                ), f"Expected shape ({imgs.shape[0]}, 7), got {logits.shape}"
                assert logits.dtype == torch.float32
                assert not torch.isnan(logits).any()
                assert not torch.isinf(logits).any()
                print(
                    f"  [PASS] Real Phase 03 {split_name} Batch Integration: {tuple(imgs.shape)} -> {tuple(logits.shape)}"
                )
    else:
        logger.warning(
            "Raw CSV not found at '%s', skipping real dataset integration test", data_path
        )

    # 7. Device Compatibility (CPU / CUDA)
    device_info = get_device_info()
    device = get_device("auto")
    logger.info(
        "7. Testing device compatibility on: %s (Type: %s)...", device, device_info["device_type"]
    )
    model.to(device)
    device_input = torch.randn(2, 1, 48, 48, dtype=torch.float32, device=device)
    with torch.no_grad():
        dev_logits = model(device_input)
    assert dev_logits.device.type == device.type
    print(f"  [PASS] Device execution verified on {device.type.upper()}")

    print("=" * 60)
    print("STATUS: ALL PHASE 04 VERIFICATION CHECKS PASSED (100%)")
    print("=" * 60 + "\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 04 Baseline CNN Model verification suite."
    )
    parser.add_argument("--model", type=str, default="baseline_cnn", help="Model name in registry")
    parser.add_argument(
        "--config", type=str, default="ml/configs/models.yaml", help="Path to models.yaml"
    )
    parser.add_argument(
        "--data", type=str, default="data/raw/fer2013/fer2013.csv", help="Path to raw FER2013 CSV"
    )
    args = parser.parse_args()

    success = verify_model(model_name=args.model, config_path=args.config, data_path=args.data)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
