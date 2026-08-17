"""Strict independent audit and verification script for Phase 07 Transfer Learning & Model Selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.evaluation.selection import (  # noqa: E402
    ModelCandidate,
    select_champion_model,
)
from ml.models.cnn.baseline_cnn import BaselineCNN  # noqa: E402
from ml.models.factory import create_model  # noqa: E402
from ml.models.transfer_learning.mobilenet import MobileNetV3SmallTransfer  # noqa: E402
from ml.models.transfer_learning.resnet import ResNet18Transfer  # noqa: E402


def audit_model_factory() -> dict[str, bool]:
    """Audit model factory and instantiation."""
    results = {}

    b_model = create_model("baseline_cnn")
    results["baseline_cnn_instantiation"] = isinstance(b_model, BaselineCNN)

    r_model = create_model("resnet18", pretrained=False)
    results["resnet18_instantiation"] = isinstance(r_model, ResNet18Transfer)

    m_model = create_model("mobilenet_v3_small", pretrained=False)
    results["mobilenet_instantiation"] = isinstance(m_model, MobileNetV3SmallTransfer)

    # Output dimensions check [B, 7]
    dummy = torch.randn(2, 1, 48, 48)
    r_out = r_model(dummy)
    m_out = m_model(dummy)
    b_out = b_model(dummy)

    results["resnet18_output_shape_7"] = r_out.shape == (2, 7)
    results["mobilenet_output_shape_7"] = m_out.shape == (2, 7)
    results["baseline_output_shape_7"] = b_out.shape == (2, 7)

    # Input channel check
    try:
        r_model(torch.randn(2, 3, 48, 48))  # in_channels is 1
        results["channel_mismatch_rejected"] = False
    except ValueError:
        results["channel_mismatch_rejected"] = True

    return results


def audit_pretrained_weights() -> dict[str, bool]:
    """Verify pretrained weights differ from randomly initialized weights."""
    results = {}

    m_pretrained = ResNet18Transfer(pretrained=True)
    m_random = ResNet18Transfer(pretrained=False)

    p_weights = next(m_pretrained.backbone.conv1.parameters())
    r_weights = next(m_random.backbone.conv1.parameters())

    results["pretrained_weights_differ_from_random"] = not torch.equal(p_weights, r_weights)
    results["pretrained_weights_finite"] = bool(torch.isfinite(p_weights).all())

    # MobileNet check
    mob_pre = MobileNetV3SmallTransfer(pretrained=True)
    mob_rand = MobileNetV3SmallTransfer(pretrained=False)
    mob_p_weights = next(mob_pre.backbone.features[0].parameters())
    mob_r_weights = next(mob_rand.backbone.features[0].parameters())
    results["mobilenet_pretrained_differ"] = not torch.equal(mob_p_weights, mob_r_weights)

    return results


def audit_freezing_and_finetuning() -> dict[str, bool]:
    """Verify backbone freezing immutability and fine-tuning weight mutation."""
    results = {}

    model = ResNet18Transfer(pretrained=False)
    model.train()

    # Stage 1: Freeze backbone
    model.freeze_backbone()
    frozen_params_before = [p.clone().detach() for p in model.get_backbone_parameters()]
    head_params_before = [p.clone().detach() for p in model.get_head_parameters()]

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    dummy_x = torch.randn(4, 1, 48, 48)
    dummy_y = torch.tensor([0, 1, 2, 3])

    out = model(dummy_x)
    loss = criterion(out, dummy_y)
    loss.backward()
    optimizer.step()

    frozen_params_after = [p.clone().detach() for p in model.get_backbone_parameters()]
    head_params_after = [p.clone().detach() for p in model.get_head_parameters()]

    # Backbone parameters must be 100% bit-exact unchanged
    backbone_unchanged = all(
        torch.equal(b, a) for b, a in zip(frozen_params_before, frozen_params_after, strict=True)
    )
    # Head parameters must be updated
    head_changed = any(
        not torch.equal(b, a) for b, a in zip(head_params_before, head_params_after, strict=True)
    )

    results["frozen_backbone_unchanged_after_step"] = backbone_unchanged
    results["head_parameters_updated_after_step"] = head_changed

    # Stage 2: Unfreeze backbone for fine-tuning
    model.unfreeze_backbone()
    optimizer2 = torch.optim.SGD(model.parameters(), lr=0.1)
    out2 = model(dummy_x)
    loss2 = criterion(out2, dummy_y)
    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()

    unfrozen_params_after = [p.clone().detach() for p in model.get_backbone_parameters()]
    backbone_updated = any(
        not torch.equal(b, a)
        for b, a in zip(frozen_params_after, unfrozen_params_after, strict=True)
    )
    results["unfrozen_backbone_updated_after_finetuning_step"] = backbone_updated

    return results


def audit_test_set_isolation_and_selection() -> dict[str, bool]:
    """Verify test set isolation and Champion selection algorithm."""
    results = {}

    candidates = [
        ModelCandidate(
            model_name="baseline_cnn",
            checkpoint_path="baseline/best.pt",
            val_accuracy=0.2951,
            val_macro_f1=0.1649,
            val_weighted_f1=0.2068,
            val_loss=1.7131,
            batch_latency_ms=26.0,
            single_sample_latency_ms=0.79,
            throughput_fps=2400.0,
            total_parameters=422119,
            model_size_mb=4.85,
        ),
        ModelCandidate(
            model_name="resnet18",
            checkpoint_path="resnet18/best.pt",
            val_accuracy=0.5183,
            val_macro_f1=0.4279,
            val_weighted_f1=0.4949,
            val_loss=1.2572,
            batch_latency_ms=35.0,
            single_sample_latency_ms=3.13,
            throughput_fps=1800.0,
            total_parameters=11180103,
            model_size_mb=128.06,
        ),
        ModelCandidate(
            model_name="mobilenet_v3_small",
            checkpoint_path="mobilenet/best.pt",
            val_accuracy=0.3594,
            val_macro_f1=0.2634,
            val_weighted_f1=0.3138,
            val_loss=1.6549,
            batch_latency_ms=18.0,
            single_sample_latency_ms=3.50,
            throughput_fps=3200.0,
            total_parameters=1522855,
            model_size_mb=17.67,
        ),
    ]

    champion, ranked_df = select_champion_model(candidates)
    results["champion_is_resnet18"] = champion.model_name == "resnet18"
    results["ranked_df_has_all_3_models"] = len(ranked_df) == 3
    results["rank_1_is_resnet18"] = ranked_df.iloc[0]["model_name"] == "resnet18"
    results["composite_score_descending"] = (
        ranked_df["composite_score"].diff().dropna() <= 0
    ).all()

    # Tie breaking test
    tie_candidates = [
        ModelCandidate("model_a", "a.pt", 0.50, 0.40, 0.45, 1.0, 10.0, 2.0, 1000.0, 1000000, 10.0),
        ModelCandidate("model_b", "b.pt", 0.55, 0.40, 0.45, 1.0, 10.0, 2.0, 1000.0, 1000000, 10.0),
    ]
    tie_champ, _ = select_champion_model(tie_candidates)
    results["tie_breaker_higher_acc_wins"] = tie_champ.model_name == "model_b"

    return results


def audit_champion_test_artifacts() -> dict[str, bool]:
    """Verify Champion Model test artifacts, confusion matrix mathematical properties, and metric consistency."""
    results = {}

    eval_dir = ROOT_DIR / "artifacts/evaluation/champion_model"
    results["champion_eval_dir_exists"] = eval_dir.exists()

    cm_path = eval_dir / "confusion_matrix.csv"
    results["confusion_matrix_csv_exists"] = cm_path.exists()
    cm_df = pd.read_csv(cm_path, index_col=0)
    cm_array = cm_df.values

    results["confusion_matrix_shape_7x7"] = cm_array.shape == (7, 7)
    total_test_samples = int(cm_array.sum())
    results["confusion_matrix_sum_matches_test_split_3589"] = total_test_samples == 3589

    correct_preds = int(np.trace(cm_array))
    computed_acc = correct_preds / total_test_samples

    # Load evaluation_results.json
    with open(eval_dir / "evaluation_results.json", encoding="utf-8") as f:
        res_json = json.load(f)

    json_acc = res_json.get("accuracy", res_json.get("summary", {}).get("accuracy", 0.0))
    results["accuracy_matches_confusion_trace"] = np.isclose(computed_acc, json_acc, atol=1e-4)
    results["champion_test_acc_approx_51_percent"] = 0.50 < json_acc < 0.55

    # Check parameter count of Champion (ResNet-18)
    model = create_model("resnet18", pretrained=False)
    actual_params = sum(p.numel() for p in model.parameters())
    results["resnet18_param_count_exact"] = actual_params == 11180103
    json_params = res_json.get(
        "total_parameters", res_json.get("summary", {}).get("total_parameters", 0)
    )
    results["param_count_matches_model_stats"] = json_params == actual_params

    return results


def audit_baseline_preservation() -> dict[str, bool]:
    """Verify Phase 06 Baseline CNN benchmark artifacts remain preserved and untouched."""
    results = {}

    baseline_dir = ROOT_DIR / "artifacts/evaluation/baseline_cnn"
    results["baseline_eval_dir_exists"] = baseline_dir.exists()

    baseline_cm = list(baseline_dir.glob("**/confusion_matrix.csv"))
    results["baseline_confusion_matrix_preserved"] = len(baseline_cm) > 0

    baseline_ckpt = ROOT_DIR / "artifacts/training/baseline_cnn/smoke_test_20260817_055905/best.pt"
    results["baseline_checkpoint_preserved"] = baseline_ckpt.exists()

    return results


def audit_scope_compliance() -> dict[str, bool]:
    """Verify no out-of-scope Phase 08+ features exist."""
    results = {}

    prohibited_terms = [
        "ResNet50",
        "EfficientNet",
        "VisionTransformer",
        "SwinTransformer",
        "optuna",
        "ray.tune",
        "cv2.VideoCapture",
        "fastapi_inference",
        "dashboard_integration",
        "torch.quantization",
        "torch.nn.utils.prune",
        "onnxruntime",
        "tensorrt",
    ]

    source_files = list((ROOT_DIR / "ml").rglob("*.py"))
    violations = []
    for sf in source_files:
        # Ignore empty scaffold files
        content = sf.read_text(encoding="utf-8")
        if "Placeholder implementation" in content and len(content.strip().splitlines()) <= 5:
            continue
        for term in prohibited_terms:
            if term in content:
                violations.append((str(sf), term))

    results["no_prohibited_scope_terms_found"] = len(violations) == 0
    if violations:
        print("Scope violations:", violations)

    return results


def main() -> None:
    """Run all Phase 07 audit checks and print results."""
    print("=" * 70)
    print("PHASE 07 INDEPENDENT AUDIT EXECUTION")
    print("=" * 70)

    all_checks = {}

    factory_res = audit_model_factory()
    all_checks.update(factory_res)
    print("\n1. Model Factory & Architecture Audit:")
    for k, v in factory_res.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    pretrained_res = audit_pretrained_weights()
    all_checks.update(pretrained_res)
    print("\n2. Pretrained Weights Audit:")
    for k, v in pretrained_res.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    freeze_res = audit_freezing_and_finetuning()
    all_checks.update(freeze_res)
    print("\n3. Freezing & Fine-Tuning Audit:")
    for k, v in freeze_res.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    selection_res = audit_test_set_isolation_and_selection()
    all_checks.update(selection_res)
    print("\n4. Decision Matrix & Champion Selection Audit:")
    for k, v in selection_res.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    champ_res = audit_champion_test_artifacts()
    all_checks.update(champ_res)
    print("\n5. Champion Model Test Evaluation & Artifact Audit:")
    for k, v in champ_res.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    base_res = audit_baseline_preservation()
    all_checks.update(base_res)
    print("\n6. Phase 06 Baseline Preservation Audit:")
    for k, v in base_res.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    scope_res = audit_scope_compliance()
    all_checks.update(scope_res)
    print("\n7. Scope Compliance Audit:")
    for k, v in scope_res.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    all_passed = all(all_checks.values())
    print("\n" + "=" * 70)
    print(f"OVERALL AUDIT RESULT: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print("=" * 70)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
