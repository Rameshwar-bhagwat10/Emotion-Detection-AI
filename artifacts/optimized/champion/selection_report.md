# Model Optimization & Candidate Selection Report

**Selected Optimized Champion:** `champion-pruning-30` (`pruning_30pct`)
- **Validation Accuracy:** 58.29%
- **Validation Macro F1:** 0.5032
- **Inference Latency:** 3.49 ms
- **Model Size:** 42.65 MB
- **Acceptance Decision:** `ACCEPTED` (Passed all quality and efficiency thresholds)

## Optimization Candidate Trade-off Matrix

| Candidate ID | Optimization | Val Acc (%) | Val Macro F1 | Latency (ms) | Size (MB) | Accepted | Decision Reason |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `champion-fp32` | `fp32_reference` | 51.82% | 0.4279 | 3.68 | 42.65 | **YES** | FP32 Champion Reference Baseline |
| `champion-fp16` | `fp16` | 51.82% | 0.4279 | 3.62 | 42.65 | **NO** | Efficiency gain insufficient (latency reduction: 1.6%, size reduction: 0.0%) |
| `champion-int8-ptq` | `int8_ptq` | 51.80% | 0.4272 | 3.88 | 42.64 | **NO** | Efficiency gain insufficient (latency reduction: -5.4%, size reduction: 0.0%) |
| `champion-pruning-10` | `pruning_10pct` | 59.38% | 0.5099 | 4.20 | 42.65 | **NO** | Efficiency gain insufficient (latency reduction: -14.0%, size reduction: 0.0%) |
| `champion-pruning-20` | `pruning_20pct` | 58.87% | 0.5141 | 3.78 | 42.65 | **NO** | Efficiency gain insufficient (latency reduction: -2.6%, size reduction: 0.0%) |
| `champion-pruning-30` | `pruning_30pct` | 58.29% | 0.5032 | 3.49 | 42.65 | **YES** | Passed all quality and efficiency thresholds |
| `champion-distillation-student` | `knowledge_distillation` | 36.44% | 0.2708 | 3.72 | 5.82 | **NO** | Macro F1 drop 0.1571 exceeds max allowed threshold 0.0300 |
