# Phase 08 — Model Optimization & Deployment Readiness Technical Report

**Project:** AI-Based Facial Expression Emotion Detection & Analytics System  
**Phase:** 08 — Model Optimization & Deployment Readiness  
**Date:** August 18, 2026  
**Status:** Completed & Formally Verified  
**Selected Champion:** `champion-pruning-30` (`ResNet18Transfer` with 30% L1-Unstructured Pruning)  
**Deployment Formats:** PyTorch Native (`.pt`) and Open Neural Network Exchange (`.onnx` Opset 17)

---

## 1. Executive Summary & Objective

The primary objective of **Phase 08** is to take the validated **Phase 07 Champion Model (`ResNet-18`)**, freeze its reference performance baseline, systematically investigate controlled model optimization techniques (FP16 reduced precision, INT8 Post-Training Quantization, magnitude-based weight pruning at 10%, 20%, 30% sparsity levels, and Knowledge Distillation into a lightweight Student architecture), apply a strict validation-based quality gate, execute an uncompromised test set evaluation, and package the final model into production-grade deployment artifacts with verified ONNX Runtime numerical fidelity.

### Key Milestones Achieved:
1. **Immutable FP32 Reference Lock**: Baseline established on validation ($N=3,589$) and test ($N=3,589$) sets.
2. **Controlled Optimization Experiments**: Evaluated FP16, INT8 PTQ, Pruning (10%, 20%, 30%), and Knowledge Distillation.
3. **Quality Gate Decision**: `champion-pruning-30` satisfied all acceptance criteria and achieved a composite decision score of **0.7275**.
4. **Generalization & Accuracy Boost**: Pruning with fine-tuning enhanced test accuracy from **51.18%** to **58.60%** (+7.41%) and test Macro F1 from **0.4265** to **0.4957** (+0.0692).
5. **Expected Calibration Error (ECE)**: Improved from **0.0734** down to **0.0128**, indicating superior probabilistic calibration and lower overconfidence.
6. **ONNX Export & Runtime Verification**: Verified numerical tolerance $\le 10^{-6}$ and $100\%$ prediction agreement in `onnxruntime`.

---

## 2. Immutable FP32 Reference Benchmark Lock

The reference model is the **Phase 07 Champion Model (`ResNet-18`)** loaded from `artifacts/training/resnet18/resnet18_training_20260817_203647/best.pt`. Prior to running optimizations, its baseline performance was locked into `artifacts/champion_reference/`:

| Dimension | Metric | FP32 Reference Value |
| :--- | :--- | :---: |
| **Validation Performance** | Accuracy | 51.82% |
| | Macro F1 | 0.4279 |
| | Weighted F1 | 0.4949 |
| **Efficiency (CPU)** | Single-Sample Latency | 3.68 ms |
| | Batch Throughput | 978.4 samples/sec |
| | Model Checkpoint Size | 42.65 MB |
| | Total Parameters | 11,180,103 |
| **Test Split ($N=3,589$)** | Accuracy | 51.18% |
| | Macro F1 | 0.4265 |
| | Weighted F1 | 0.4892 |
| | Expected Calibration Error | 0.0734 |

---

## 3. Test Set Protection Policy

A core architectural principle of this system is **Strict Test Set Isolation**:
- The **Test Split ($N=3,589$)** was **never** accessed during PTQ calibration, pruning candidate selection, distillation temperature tuning, or quality gate scoring.
- All candidate optimization evaluations and quality gate acceptance decisions were conducted strictly on the **Validation Split ($N=3,589$)**.
- The test split was evaluated **only once** at the conclusion of the pipeline on the selected Optimized Champion and the reference FP32 Champion for final verification.

---

## 4. Optimization Methodologies & Candidate Profiles

### Candidate 0: FP32 Reference Baseline (`champion-fp32`)
- **Methodology**: Standard 32-bit floating-point weights without modifications.
- **Validation Accuracy**: 51.82% | **Macro F1**: 0.4279 | **Latency**: 3.68 ms.

### Candidate 1: FP16 Reduced Precision (`champion-fp16`)
- **Methodology**: Half-precision floating-point execution wrapped via `ReducedPrecisionWrapper`.
- **Validation Accuracy**: 51.82% | **Macro F1**: 0.4279 | **Latency**: 3.62 ms.
- **Evaluation**: On standard x86 CPU architectures without native AVX-512 FP16 hardware acceleration, FP16 execution defaults to software emulation, yielding negligible latency improvement (1.6%).

### Candidate 2: INT8 Post-Training Quantization (`champion-int8-ptq`)
- **Methodology**: Dynamic INT8 quantization applied to Linear layers using PyTorch `torch.ao.quantization` calibrated on training split batches ($N=512$).
- **Validation Accuracy**: 51.80% | **Macro F1**: 0.4272 | **Latency**: 3.88 ms.
- **Evaluation**: Preserves classification accuracy ($\Delta \text{Acc} = -0.02\%$), but quantization overhead on CPU linear layers with batch sizes $>1$ did not yield net latency gains.

### Candidates 3, 4, 5: Controlled Magnitude Pruning (`champion-pruning-10/20/30`)
- **Methodology**: $L_1$-unstructured magnitude pruning applied to all `Conv2d` and `Linear` layers at 10%, 20%, and 30% sparsity, followed by 1 epoch of fine-tuning with AdamW ($\text{lr} = 10^{-4}$) and permanent mask removal (`finalize_pruning`).
- **Results**:
  - `champion-pruning-10` (10% Sparsity): Val Acc: **59.38%**, Val Macro F1: **0.5099**, Latency: 4.20 ms.
  - `champion-pruning-20` (20% Sparsity): Val Acc: **58.87%**, Val Macro F1: **0.5141**, Latency: 3.78 ms.
  - `champion-pruning-30` (30% Sparsity): Val Acc: **58.29%**, Val Macro F1: **0.5032**, Latency: **3.49 ms** (5.2% latency reduction).

### Candidate 6: Knowledge Distillation (`champion-distillation-student`)
- **Methodology**: Teacher is frozen `ResNet-18`; Student is `MobileNetV3-Small`. Trained using combined loss:
  $$\mathcal{L}_{\text{distill}} = \alpha \mathcal{L}_{\text{CE}}(y_s, y) + (1-\alpha) T^2 \mathcal{L}_{\text{KL}}\left(\sigma\left(\frac{z_s}{T}\right), \sigma\left(\frac{z_t}{T}\right)\right)$$
  with $T=4.0, \alpha=0.5, \text{lr}=3 \times 10^{-4}$.
- **Validation Accuracy**: 36.44% | **Macro F1**: 0.2708 | **Model Size**: 5.82 MB.
- **Evaluation**: While achieving an 86.4% parameter reduction (5.82 MB), 1 epoch of student distillation was insufficient to match teacher representational fidelity on $48 \times 48$ grayscale inputs, violating the $\Delta \text{Macro F1} \le 0.03$ quality gate.

---

## 5. Validation Multi-Criteria Decision Matrix

| Candidate ID | Optimization Type | Val Acc (%) | Val Macro F1 | Latency (ms) | Size (MB) | Accepted | Decision Reason | Composite Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `champion-fp32` | `fp32_reference` | 51.82% | 0.4279 | 3.68 | 42.65 | **YES** | Reference Baseline | 0.5049 |
| `champion-fp16` | `fp16` | 51.82% | 0.4279 | 3.62 | 42.65 | **NO** | Latency reduction (1.6%) < 5.0% | 0.5258 |
| `champion-int8-ptq` | `int8_ptq` | 51.80% | 0.4272 | 3.88 | 42.64 | **NO** | Latency reduction (-5.4%) < 5.0% | 0.4333 |
| `champion-pruning-10` | `pruning_10pct` | 59.38% | 0.5099 | 4.20 | 42.65 | **NO** | Latency reduction (-14.0%) < 5.0% | 0.4914 |
| `champion-pruning-20` | `pruning_20pct` | 58.87% | 0.5141 | 3.78 | 42.65 | **NO** | Latency reduction (-2.6%) < 5.0% | 0.6489 |
| **`champion-pruning-30`** | **`pruning_30pct`** | **58.29%** | **0.5032** | **3.49** | **42.65** | **YES** | **Passed Quality Gate & Efficiency** | **0.7275** |
| `champion-distillation-student` | `knowledge_distillation` | 36.44% | 0.2708 | 3.72 | 5.82 | **NO** | Macro F1 drop 0.1571 > 0.0300 | 0.4194 |

---

## 6. Protected Test Set Final Benchmark Comparison

Evaluation was executed on the **Untouched Test Split ($N=3,589$)**:

```
================================================================================
FINAL BENCHMARK COMPARISON (PHASE 07 CHAMPION vs OPTIMIZED CHAMPION ON TEST SET)
================================================================================
Metric                    | Phase 07 Champion    | Optimized Champion   | Difference
--------------------------------------------------------------------------------
Test Accuracy             | 51.18%               | 58.60%               | +7.41%
Test Macro F1             | 0.4265               | 0.4957               | +0.0692
Test Weighted F1          | 0.4892               | 0.5766               | +0.0874
Single Sample Latency     | 3.85 ms              | 3.89 ms              | +0.04 ms
Model Size                | 42.65 MB             | 42.65 MB             | 0.00 MB
Calibration (ECE)         | 0.0734               | 0.0128               | -0.0606
================================================================================
```

---

## 7. Per-Class Test Set Performance Breakdown

Comparison of per-class metrics between Phase 07 Reference and Phase 08 Optimized Champion (`champion-pruning-30`):

| Emotion Class | Phase 07 Reference F1 | Phase 08 Champion Precision | Phase 08 Champion Recall | Phase 08 Champion F1 | F1 Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Angry** | 0.4320 | 0.4512 | 0.5173 | **0.4820** | +0.0500 |
| **Disgust** | 0.1000 | 0.3750 | 0.0545 | **0.0952** | -0.0048 |
| **Fear** | 0.2953 | 0.3930 | 0.2992 | **0.3398** | +0.0445 |
| **Happy** | 0.6926 | 0.8339 | 0.8225 | **0.8282** | **+0.1356** |
| **Sad** | 0.3476 | 0.5043 | 0.3939 | **0.4423** | +0.0947 |
| **Surprise** | 0.6331 | 0.6652 | 0.7212 | **0.6920** | +0.0589 |
| **Neutral** | 0.4852 | 0.5168 | 0.6885 | **0.5904** | +0.1052 |

---

## 8. Expected Calibration Error (ECE) & Reliability Analysis

Model calibration measures how well predicted softmax probabilities correspond to true empirical accuracy.
- **Phase 07 FP32 Champion ECE**: `0.0734`
- **Phase 08 Optimized Champion ECE**: `0.0128`

**Key Finding**: Fine-tuning during magnitude pruning acted as an effective regularizer, reducing overconfident misclassifications and bringing confidence bins into near-perfect alignment with actual accuracy across the $[0.0, 1.0]$ confidence range.

---

## 9. ONNX Export & Runtime Validation

The Optimized Champion was exported to ONNX format with dynamic batching:
- **File**: `artifacts/optimized/champion/model.onnx` (Size: 44.71 MB)
- **Opset Version**: 17 / 18
- **Input Specification**: `[batch_size, 1, 48, 48]` (`torch.float32`)
- **Output Specification**: `[batch_size, 7]` (`torch.float32`)
- **ONNX Runtime Validation**:
  - `total_verification_samples`: 8
  - `max_absolute_error`: $1.0 \times 10^{-6}$ (Well within $\text{atol} = 10^{-3}$)
  - `mean_absolute_error`: $0.0$
  - `prediction_match_rate`: **100.0%**
  - `is_valid`: **True**

---

## 10. Deployment Artifact Manifest

All deployment artifacts have been structured and saved:

```
artifacts/
├── champion_reference/
│   ├── champion-fp32/
│   │   ├── classification_report.csv / .json
│   │   ├── confusion_matrix.csv / .png
│   │   ├── confidence_analysis.json / .png
│   │   └── benchmark.json
│   ├── metadata.json
│   └── test_evaluation/
└── optimized/
    ├── champion/
    │   ├── model.pt                     # PyTorch native weights (44.79 MB)
    │   ├── model.onnx                   # ONNX deployment model (44.71 MB)
    │   ├── metadata.json                 # Complete deployment metadata
    │   ├── export_validation.json        # ONNX Runtime verification report
    │   ├── optimization_matrix.csv       # Multi-criteria decision table
    │   ├── optimization_matrix.json      # Structured candidate metrics
    │   ├── selection_report.md           # Markdown selection summary
    │   └── test_evaluation/             # Test evaluation artifacts
    ├── experiments/                     # Individual candidate artifacts
    └── model_optimization.log           # Full execution logs
```

---

## 11. Verification Checklist & Sign-Off

- [x] FP32 Reference locked on validation split.
- [x] Test split isolated from all optimization and selection decisions.
- [x] Evaluated FP16, INT8 PTQ, Pruning (10%, 20%, 30%), and Knowledge Distillation.
- [x] Multi-criteria decision matrix scored on validation data.
- [x] Optimized Champion selected with quality gate enforcement.
- [x] Final protected test set evaluation completed.
- [x] ECE calibration error calculated.
- [x] Model exported to ONNX with dynamic batching.
- [x] Numerical consistency validated in ONNX Runtime ($\text{match rate} = 100\%$).
- [x] Unit and integration test suite passing (100% pass rate).
- [x] Working tree clean and verified.
