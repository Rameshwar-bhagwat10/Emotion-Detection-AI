# Model Evaluation & Verification

This directory contains the empirical evaluation reports, benchmark metrics, and generalization analyses for the facial expression emotion recognition models.

## Documentation Index

- [Comprehensive Model Performance Evaluation Report (Phase 11/12 Baseline)](file:///d:/projects/emotion-detection-ai/docs/ml/evaluation/model-performance-evaluation-report.md): In-depth 36-section audit analyzing accuracy, calibration, per-class performance, generalization gap, error distributions, and real-time inference consistency for the Champion ResNet-18 model (`champion-pruning-30`).
- [Model Selection & Champion Tournament Report](file:///d:/projects/emotion-detection-ai/docs/ml/evaluation/model-selection-report.md): Multi-model tournament comparing Baseline CNN, MobileNetV3-Small, and ResNet-18 on accuracy, F1, throughput, and parameter efficiency.
- [Baseline CNN Evaluation Report](file:///d:/projects/emotion-detection-ai/docs/ml/evaluation/baseline-evaluation.md): Initial evaluation of the baseline 4-layer custom convolutional network.

## Key Performance Summary (Champion ResNet-18)

| Metric | Test Set (`PrivateTest`, N=3,589) | Validation Set (`PublicTest`, N=3,589) | Training Set (N=28,709) |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | **58.60%** | 58.29% | 60.54% |
| **Balanced Accuracy** | **49.96%** | 50.43% | 52.49% |
| **Macro F1-Score** | **0.4957** | 0.5032 | 0.5258 |
| **Weighted F1-Score**| **0.5766** | 0.5734 | 0.5969 |
| **Macro ROC-AUC** | **0.8831** | 0.8850 | 0.8912 |
| **Expected Calibration Error (ECE)** | **0.0128** | 0.0186 | 0.0261 |
| **Inference Latency (Single Sample, CPU)** | **5.58 ms** | — | — |
| **Throughput (Single Sample, CPU)** | **179.3 FPS** | — | — |
| **Model Size** | **42.72 MB** (11.18M params, 30% pruned) | — | — |
