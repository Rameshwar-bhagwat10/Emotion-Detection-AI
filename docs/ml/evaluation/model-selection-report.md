# Phase 07 — Transfer Learning Candidate Comparison & Model Selection Report

## 1. Executive Summary

In **Phase 07**, we evaluated three candidate architectures under a strict validation-only decision matrix to determine whether a stronger pretrained architecture should replace the Phase 06 `BaselineCNN`.

**ResNet-18 was decisively selected as the Champion Model** with a multi-criteria score of **0.7203** (versus 0.3664 for MobileNetV3-Small and 0.3000 for Baseline CNN).

On the single, uncompromised test set benchmark ($N = 3,589$), **ResNet-18 achieved 51.18% Accuracy** and **0.4265 Macro F1**, delivering a **$+21.98\%$ absolute accuracy gain** and **$+163\%$ macro F1 improvement** over the Phase 06 baseline while maintaining an ultra-fast **3.10 ms CPU inference latency** ($\sim 322$ FPS).

---

## 2. Multi-Criteria Candidate Comparison (Validation Split)

All candidate models were evaluated on the **Validation Split** ($N = 3,589$) without touching the Test Split.

### Decision Matrix Table:
| Rank | Model Candidate | Validation Accuracy | Validation Macro F1 | Weighted F1 | Single Sample Latency (CPU) | Model Size | Composite Score | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **ResNet-18** | **51.83%** | **0.4279** | **0.4949** | **3.13 ms** | **128.06 MB** | **0.7203** | **SELECTED CHAMPION** |
| **2** | MobileNetV3-Small | 35.94% | 0.2634 | 0.3138 | 3.50 ms | 17.67 MB | 0.3664 | Eliminated |
| **3** | Baseline CNN | 29.51% | 0.1649 | 0.2068 | 0.79 ms | 4.85 MB | 0.3000 | Reference Baseline |

### Scoring Weights:
- **Validation Macro F1:** 35% weight
- **Validation Accuracy:** 35% weight
- **Inference Latency (lower is better):** 15% weight
- **Model Storage Size (lower is better):** 15% weight

---

## 3. Final Champion Benchmark vs Baseline Benchmark (Test Split)

The selected **Champion Model (ResNet-18)** was evaluated on the **untouched FER2013 Test Split** ($N = 3,589$):

| Metric | Phase 06 Baseline CNN | Phase 07 Champion (ResNet-18) | Absolute Improvement | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Test Accuracy** | 29.20% | **51.18%** | **+21.98%** | **+75.3%** |
| **Macro Precision** | 19.91% | **52.35%** | **+32.44%** | **+162.9%** |
| **Macro Recall** | 21.42% | **42.04%** | **+20.62%** | **+96.3%** |
| **Macro F1-Score** | 0.1619 | **0.4265** | **+0.2646** | **+163.4%** |
| **Weighted Precision** | 23.04% | **50.64%** | **+27.60%** | **+119.8%** |
| **Weighted Recall** | 29.20% | **51.18%** | **+21.98%** | **+75.3%** |
| **Weighted F1-Score** | 0.2031 | **0.4892** | **+0.2861** | **+140.9%** |
| **CPU Latency (Single Sample)** | 0.86 ms | **3.10 ms** | +2.24 ms | Real-Time Capable ($\sim 322$ FPS) |
| **Total Parameters** | 422,119 | 11,180,103 | +10.76M | Deep Residual Architecture |

---

## 4. Per-Class Detailed Performance (Champion ResNet-18 vs Baseline)

| Emotion Class | Baseline Test F1 | Champion Test F1 | Baseline Recall | Champion Recall | Champion Precision | Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Angry** | 0.0229 | **0.4320** | 1.22% | **40.12%** | 46.79% | 491 |
| **Disgust** | 0.0000 | **0.1000** | 0.00% | **5.45%** | 60.00% | 55 |
| **Fear** | 0.0074 | **0.2953** | 0.38% | **25.76%** | 34.61% | 528 |
| **Happy** | 0.4379 | **0.6926** | 82.25% | **89.19%** | 56.61% | 879 |
| **Sad** | 0.0738 | **0.3476** | 4.55% | **28.62%** | 44.27% | 594 |
| **Surprise** | 0.4065 | **0.6331** | 45.43% | **52.88%** | 78.85% | 416 |
| **Neutral** | 0.1850 | **0.4852** | 16.13% | **52.24%** | 45.29% | 626 |

---

## 5. Artifact Locations

- **Candidate Selection Matrix**: `artifacts/evaluation/model_selection/model_selection_matrix.csv`
- **Candidate Selection Summary**: `artifacts/evaluation/model_selection/model_selection_summary.md`
- **Champion Evaluation Results**: `artifacts/evaluation/champion_model/evaluation_results.json`
- **Champion Classification Report**: `artifacts/evaluation/champion_model/classification_report.csv`
- **Champion Confusion Matrix**: `artifacts/evaluation/champion_model/confusion_matrix.png`
- **Champion Confidence Distribution**: `artifacts/evaluation/champion_model/confidence_distribution.png`
