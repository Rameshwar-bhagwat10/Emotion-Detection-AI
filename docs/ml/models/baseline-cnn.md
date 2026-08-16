# Baseline CNN Model Architecture Specification

## 1. Overview & Purpose

The **Baseline CNN (`baseline_cnn_v1`)** is a clean, 4-stage Convolutional Neural Network designed specifically for $48 \times 48$ grayscale facial expression emotion classification on the FER2013 dataset. It establishes the benchmark neural architecture for subsequent model training (Phase 05) and advanced backbone experimentation.

---

## 2. Input & Output Contracts

| Attribute | Specification | Notes |
| :--- | :--- | :--- |
| **Input Shape** | `[B, 1, 48, 48]` | Batch size $B$, Grayscale $C=1$, Height $H=48$, Width $W=48$ |
| **Input Data Type** | `torch.float32` | Normalized tensors ($\mu=0.507743, \sigma=0.255009$) |
| **Output Shape** | `[B, 7]` | 7 Emotion Classes |
| **Output Values** | **Raw Logits** | **No Softmax inside model**. Suitable for `nn.CrossEntropyLoss` |
| **Emotion Classes** | `[0..6]` | `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral` |

---

## 3. Architecture & Shape Tracing

```text
                                  INPUT TENSOR
                             [B, 1, 48, 48] (float32)
                                        │
                                        ▼
                                 STAGE 1 CONV BLOCK
                             Conv2d(1 -> 32, k=3, s=1, p=1)
                             BatchNorm2d(32)
                             ReLU(inplace=True)
                             MaxPool2d(k=2, s=2)
                             Output: [B, 32, 24, 24]
                                        │
                                        ▼
                                 STAGE 2 CONV BLOCK
                             Conv2d(32 -> 64, k=3, s=1, p=1)
                             BatchNorm2d(64)
                             ReLU(inplace=True)
                             MaxPool2d(k=2, s=2)
                             Output: [B, 64, 12, 12]
                                        │
                                        ▼
                                 STAGE 3 CONV BLOCK
                             Conv2d(64 -> 128, k=3, s=1, p=1)
                             BatchNorm2d(128)
                             ReLU(inplace=True)
                             MaxPool2d(k=2, s=2)
                             Output: [B, 128, 6, 6]
                                        │
                                        ▼
                                 STAGE 4 CONV BLOCK
                             Conv2d(128 -> 256, k=3, s=1, p=1)
                             BatchNorm2d(256)
                             ReLU(inplace=True)
                             MaxPool2d(k=2, s=2)
                             Output: [B, 256, 3, 3]
                                        │
                                        ▼
                              ADAPTIVE AVERAGE POOLING
                             AdaptiveAvgPool2d((1, 1))
                             Output: [B, 256, 1, 1]
                                        │
                                        ▼
                                     FLATTEN
                             Flatten() -> [B, 256]
                                        │
                                        ▼
                               CLASSIFICATION HEAD
                             Linear(256 -> 128)
                             ReLU(inplace=True)
                             Dropout(p=0.3)
                             Linear(128 -> 7)
                                        │
                                        ▼
                                   RAW LOGITS
                                    [B, 7]
```

### Layer-by-Layer Dimension Tracing:

| Layer / Stage | Output Tensor Dimension | Receptive Field / Downsampling |
| :--- | :---: | :--- |
| **Input** | `[B, 1, 48, 48]` | Raw preprocessed tensor |
| **Stage 1 (Conv + BN + ReLU + MaxPool)** | `[B, 32, 24, 24]` | Downsampled $2\times$ |
| **Stage 2 (Conv + BN + ReLU + MaxPool)** | `[B, 64, 12, 12]` | Downsampled $4\times$ |
| **Stage 3 (Conv + BN + ReLU + MaxPool)** | `[B, 128, 6, 6]` | Downsampled $8\times$ |
| **Stage 4 (Conv + BN + ReLU + MaxPool)** | `[B, 256, 3, 3]` | Downsampled $16\times$ |
| **Adaptive Average Pooling** | `[B, 256, 1, 1]` | Spatially aggregated feature maps |
| **Flatten** | `[B, 256]` | 1D feature vector per image |
| **Linear + ReLU + Dropout** | `[B, 128]` | Dense hidden representation |
| **Final Linear Classifier** | `[B, 7]` | Raw unnormalized emotion logits |

---

## 4. Architectural Design Rationale

1. **$3 \times 3$ Small Convolutions:** Standard VGG-style stacked $3 \times 3$ filters with padding 1 capture localized facial edge and texture patterns efficiently without spatial shrinking prior to pooling.
2. **Batch Normalization:** Placed immediately following convolutions to stabilize gradient propagation and accelerate convergence.
3. **Adaptive Average Pooling:** Aggregates variable spatial feature maps into a fixed $1 \times 1$ dimension, decoupling the classification head from rigid spatial dimensions.
4. **Dropout Regularization ($p=0.3$):** Moderate dropout in the classification head mitigates overfitting on the FER2013 training split.
5. **Kaiming / He Normal Weight Initialization:** Initializes Conv and Linear weights with $\mathcal{N}(0, \sqrt{2/\text{fan\_out}})$ to prevent vanishing/exploding gradients in deep ReLU networks.

---

## 5. Parameter Count Summary

| Component | Total Parameters | Trainable Parameters | Non-Trainable Parameters |
| :--- | :---: | :---: | :---: |
| **Feature Extractor (Stages 1–4)** | $388,320$ | $388,320$ | $0$ |
| **Classification Head** | $33,799$ | $33,799$ | $0$ |
| **Total Model Parameters** | **$422,119$** | **$422,119$** | **$0$** |

*(Parameter counts calculated via `get_parameter_count()` API).*

---

## 6. Model Factory & Registry Usage

```python
from ml.models.factory import create_model

# 1. Instantiate baseline model from default configuration
model = create_model("baseline_cnn")

# 2. Inspect model metadata
summary = model.get_model_summary()
print(summary)

# 3. Forward pass
import torch
dummy_input = torch.randn(4, 1, 48, 48, dtype=torch.float32)
logits = model(dummy_input)  # Shape: [4, 7]
```

---

## 7. Train vs. Eval Mode Behavior

- **`model.train()`:** Activates `Dropout(p=0.3)` and updates running statistics in `BatchNorm2d`.
- **`model.eval()`:** Disables `Dropout` (identity mapping) and freezes `BatchNorm2d` running mean/variance for deterministic validation/testing.

---

## 8. State Dict Serialization & Invariance

The model supports PyTorch `state_dict` serialization:

```python
# Save weights
torch.save(model.state_dict(), "models/baseline_cnn.pt")

# Reload weights into a fresh instance
new_model = create_model("baseline_cnn")
new_model.load_state_dict(torch.load("models/baseline_cnn.pt"))
new_model.eval()
```

---

## 9. CLI Tools

### Inspect Model Architecture & Parameters
```bash
python scripts/model/inspect-model.py
```

### Run Model Verification Suite
```bash
python scripts/model/verify-model.py
```

---

## 10. Phase 04 Boundaries & Limitations

> [!NOTE]
> **Phase 04 Scope Contract:**
> - The model returns **raw unnormalized logits**. Softmax is applied in inference/API phases, and CrossEntropyLoss is applied in the training phase.
> - Phase 04 implements model architecture, blocks, registry, and testing. It does **not** train weights, instantiate optimizers, or run inference loops.
