# Transfer Learning & Advanced Vision Architectures

## 1. Overview

Phase 07 introduces transfer learning for the **AI-Based Facial Expression Emotion Detection & Analytics System**. By leveraging pretrained convolutional neural networks trained on the ImageNet dataset, we transfer rich visual feature representations (edge detectors, texture filters, facial geometry extractors) to facial emotion classification on the FER2013 dataset.

---

## 2. Architectures Implemented

### 2.1 ResNet-18 Transfer (`ResNet18Transfer`)
- **Backbone:** Deep residual network with 18 weighted layers and identity shortcut connections.
- **Grayscale Adaptation:** Converts 1-channel `[B, 1, 48, 48]` tensors to 3-channel representations via channel repetition (`x.repeat(1, 3, 1, 1)`), allowing full reuse of ImageNet `conv1` pretrained convolutional kernels without weight reinitialization.
- **Head Adaptation:** Replaces the standard 1,000-class linear projection with `nn.Sequential(nn.Dropout(p=0.2), nn.Linear(512, 7))`.
- **Parameter Count:** `11,180,103` total parameters (all trainable).

### 2.2 MobileNetV3-Small Transfer (`MobileNetV3SmallTransfer`)
- **Backbone:** Hardware-aware neural architecture designed for high-efficiency mobile and edge inference using depthwise separable convolutions and Squeeze-and-Excitation (SE) attention modules.
- **Grayscale Adaptation:** Converts 1-channel input to 3 channels via channel repetition.
- **Head Adaptation:** Replaces the final classifier projection with `nn.Linear(1024, 7)`.
- **Parameter Count:** `1,522,855` total parameters.

---

## 3. Fine-Tuning & Training Strategies

- **Optimizer:** `AdamW` ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 10^{-4}$).
- **Learning Rate:** $\text{lr} = 3 \times 10^{-4}$ (ResNet-18) and $\text{lr} = 5 \times 10^{-4}$ (MobileNetV3-Small).
- **Learning Rate Schedule:** `ReduceLROnPlateau` (factor $= 0.5$, patience $= 2$ epochs).
- **Gradient Regularization:** Gradient norm clipping at $\text{max\_norm} = 1.0$.
- **Loss Function:** Categorical Cross-Entropy with optional label smoothing ($\epsilon = 0.05$).
