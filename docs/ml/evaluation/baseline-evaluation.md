# Baseline CNN Model Evaluation & Benchmarking Report

## 1. Executive Summary

Phase 06 establishes the **official baseline benchmark** for the **AI-Based Facial Expression Emotion Detection & Analytics System**. The evaluation was performed on the untouched **FER2013 test dataset** ($N = 3,589$ single-channel grayscale $48 \times 48$ images) using the trained Phase 05 `BaselineCNN` (`baseline_cnn_v1`) loaded from the `best.pt` checkpoint.

### Key Benchmark Metrics Summary:
- **Overall Test Accuracy:** **29.20%** ($1,048 / 3,589$ correct)
- **Macro F1-Score:** **0.1619** (Macro Precision: 0.1991, Macro Recall: 0.2142)
- **Weighted F1-Score:** **0.2031** (Weighted Precision: 0.2304, Weighted Recall: 0.2920)
- **Mean Single-Sample Latency (CPU):** **0.86 ms** ($\sim 1,164$ inferences/sec)
- **Mean Batch Latency (Batch=64, CPU):** **26.17 ms** ($\sim 2,445$ samples/sec)
- **Total Model Parameters:** **422,119** (100% trainable)
- **Checkpoint Filesystem Size:** **4.85 MB**

---

## 2. Evaluation Setup & Methodology

- **Model Architecture:** `BaselineCNN` (4 convolutional blocks with BatchNorm, ReLU, MaxPool, Dropout, and 2 Dense layers).
- **Evaluated Checkpoint:** `artifacts/training/baseline_cnn/smoke_test_20260817_055905/best.pt`.
- **Dataset Split:** FER2013 Test Split (`data/raw/fer2013/fer2013.csv`, `split='test'`, $N = 3,589$).
- **Preprocessing Contract:** `[B, 1, 48, 48]` float32 tensors, normalized with training split statistics ($\mu = 0.507743, \sigma = 0.255009$).
- **Evaluation Mode:** Strictly evaluated with `model.eval()` inside `torch.no_grad()`; model weights are 100% immutable.
- **Evaluation Environment:** CPU (`x86_64`, PyTorch 2.6.0+cpu).

---

## 3. Comprehensive Classification Metrics

### Aggregate Metrics Table:
| Metric | Value |
| :--- | :--- |
| **Total Test Samples** | 3,589 |
| **Correct Predictions** | 1,048 (29.20%) |
| **Incorrect Predictions** | 2,541 (70.80%) |
| **Overall Accuracy** | **29.20%** |
| **Macro Precision** | **19.91%** |
| **Macro Recall** | **21.42%** |
| **Macro F1-Score** | **0.1619** |
| **Weighted Precision** | **23.04%** |
| **Weighted Recall** | **29.20%** |
| **Weighted F1-Score** | **0.2031** |

### Per-Class Detailed Performance:
| Emotion Class | Precision | Recall | F1-Score | Support | Correct | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Angry** | 0.1818 | 0.0122 | 0.0229 | 491 | 6 | 1.22% |
| **Disgust** | 0.0000 | 0.0000 | 0.0000 | 55 | 0 | 0.00% |
| **Fear** | 0.1333 | 0.0038 | 0.0074 | 528 | 2 | 0.38% |
| **Happy** | 0.2984 | 0.8225 | **0.4379** | 879 | 723 | 82.25% |
| **Sad** | 0.1957 | 0.0455 | 0.0738 | 594 | 27 | 4.55% |
| **Surprise** | 0.3677 | 0.4543 | **0.4065** | 416 | 189 | 45.43% |
| **Neutral** | 0.2167 | 0.1613 | 0.1850 | 626 | 101 | 16.13% |

---

## 4. Confusion Matrix & Misclassification Patterns

### Raw Confusion Matrix ($7 \times 7$ Sample Counts):
```text
Actual \ Pred    angry   disgust      fear     happy       sad  surprise   neutral
----------------------------------------------------------------------------------
angry                6         0         1       306         2        80        96
disgust              0         0         0        36         0         9        10
fear                 1         0         2       328         4       112        81
happy               14         0         6       723        17        41        78
sad                  5         0         3       400        27        33       126
surprise             3         0         1       176         4       189        43
neutral              4         0         2       448        10        61       101
```

### Top 10 Most Confused Emotion Pairs (Off-Diagonal):
| Rank | Actual Emotion | Predicted Emotion | Misclassified Sample Count |
| :---: | :--- | :--- | :---: |
| **1** | Neutral | Happy | **448** |
| **2** | Sad | Happy | **400** |
| **3** | Fear | Happy | **328** |
| **4** | Angry | Happy | **306** |
| **5** | Surprise | Happy | **176** |
| **6** | Sad | Neutral | **126** |
| **7** | Fear | Surprise | **112** |
| **8** | Neutral | Happy | **101** |
| **9** | Angry | Neutral | **96** |
| **10** | Fear | Neutral | **81** |

---

## 5. Prediction Confidence & Error Analysis

### Confidence Statistics:
- **Overall Mean Confidence:** `0.2995` ($\pm 0.1077$)
- **Correct Predictions Mean Confidence:** `0.3473` ($\pm 0.1311$)
- **Incorrect Predictions Mean Confidence:** `0.2798` ($\pm 0.0892$)
- **High-Confidence Errors ($\ge 0.90$):** `0` samples (0.00% of errors).
- **Confidence Calibration Insight:** The early baseline model produces low to moderate softmax probabilities ($\mu \approx 0.30$), avoiding severe overconfidence on ambiguous samples.

### Confidence Distribution Across Samples:
| Confidence Bin | Total Count | Correct Count | Incorrect Count |
| :---: | :---: | :---: | :---: |
| `0.10 – 0.20` | 236 | 29 | 207 |
| `0.20 – 0.30` | 2,055 | 454 | 1,601 |
| `0.30 – 0.40` | 764 | 284 | 480 |
| `0.40 – 0.50` | 295 | 136 | 159 |
| `0.50 – 0.60` | 137 | 68 | 69 |
| `0.60 – 0.70` | 65 | 45 | 20 |
| `0.70 – 0.80` | 26 | 23 | 3 |
| `0.80 – 0.90` | 11 | 9 | 2 |
| `0.90 – 1.00` | 0 | 0 | 0 |

---

## 6. Inference Latency & Throughput Benchmark

Benchmarked on host CPU with $10$ warmup iterations and $50$ timed iterations:

### Batch Inference (Batch Size = 64):
- **Mean Latency:** **26.17 ms**
- **Median Latency:** **26.14 ms**
- **95th Percentile (p95) Latency:** **29.58 ms**
- **Throughput:** **2,445.19 samples / second**

### Single-Sample Inference (Batch Size = 1):
- **Mean Latency:** **0.86 ms**
- **Median Latency:** **0.79 ms**
- **95th Percentile (p95) Latency:** **1.11 ms**
- **Throughput:** **1,164.28 frames / second**

---

## 7. Model Statistics & Storage Efficiency

- **Total Parameter Count:** `422,119`
- **Trainable Parameters:** `422,119` (100%)
- **Non-Trainable Parameters:** `0`
- **Checkpoint File Size:** `4.85 MB` (`5,090,697` bytes)

---

## 8. Artifact Locations

All evaluation outputs are persisted under `artifacts/evaluation/baseline_cnn/baseline_cnn_eval_20260817_063335/`:

- `evaluation_results.json`: High-level summary of all benchmark metrics.
- `classification_report.json` & `classification_report.csv`: Complete per-class and aggregate metric breakdown.
- `confusion_matrix.csv`: Raw 7x7 sample confusion matrix.
- `confusion_matrix_normalized.csv`: Support-normalized confusion matrix.
- `confusion_pairs.json`: Ranked list of all off-diagonal misclassification pairs.
- `confidence_analysis.json`: Confidence distributions and statistics.
- `incorrect_predictions.csv`: Sample-by-sample record of all 2,541 misclassified test items.
- `benchmark.json`: Inference latency and throughput measurements.
- `model_stats.json`: Architecture parameter counts and filesystem footprint.
- `confusion_matrix.png` & `confusion_matrix_normalized.png`: Heatmap visualizations.
- `confidence_distribution.png`: Correct vs. incorrect confidence distribution plot.

---

## 9. Baseline Findings & Recommendations for Future Phases

1. **Dominant Majority Class Bias:** The early baseline model predominantly predicts the most frequent classes (`happy` and `surprise`), achieving 82.25% recall on happy but near-zero recall on minority classes like `disgust` (0.0%), `fear` (0.38%), and `angry` (1.22%).
2. **Subtle Facial Emotion Overlap:** Negative subtle emotions (`sad`, `fear`, `neutral`, `angry`) are heavily confused with each other and collapsed into `happy` and `neutral`.
3. **Capacity & Regularization:** The 4-block baseline CNN demonstrates ultra-fast CPU inference (0.86 ms per frame), establishing an ideal speed benchmark. Advanced architectures (ResNet/Attention backbones in Phase 07+), focal loss, and class rebalancing will address minority class recall without compromising inference throughput.
