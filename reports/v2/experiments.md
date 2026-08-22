# Model V2 Experimentation Log

This document details all controlled experiments performed during the Model V2 improvement program.

## EXP_V1_BASELINE
- **Model**: `resnet18_v1`
- **Input**: `48x48` (channels=1)
- **Loss / Sampler**: `cross_entropy` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 58.29%
- **Validation Balanced Accuracy**: 50.43%
- **Validation Macro F1**: 50.32%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 9.52% / 33.98% / 44.23% / 48.20%
- **Notes**: Original V1 baseline (pruned 30% champion)

## EXP_A1_SAMPLER_BALANCED
- **Model**: `resnet18`
- **Input**: `48x48` (channels=1)
- **Loss / Sampler**: `cross_entropy` / `balanced_sampler`
- **Augmentation**: `none`
- **Validation Accuracy**: 38.50%
- **Validation Balanced Accuracy**: 40.75%
- **Validation Macro F1**: 36.05%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 15.71% / 22.49% / 28.83% / 32.69%
- **Notes**: Full inverse class frequency sampling

## EXP_A2_SAMPLER_SMOOTHED
- **Model**: `resnet18`
- **Input**: `48x48` (channels=1)
- **Loss / Sampler**: `cross_entropy` / `smoothed_sampler`
- **Augmentation**: `none`
- **Validation Accuracy**: 42.08%
- **Validation Balanced Accuracy**: 37.33%
- **Validation Macro F1**: 37.88%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 16.67% / 29.61% / 30.67% / 31.71%
- **Notes**: Square-root smoothed class frequency sampling

## EXP_B1_WEIGHTED_CE_INVERSE
- **Model**: `resnet18`
- **Input**: `48x48` (channels=1)
- **Loss / Sampler**: `cross_entropy` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 38.25%
- **Validation Balanced Accuracy**: 37.11%
- **Validation Macro F1**: 35.05%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 18.92% / 28.57% / 33.33% / 25.39%
- **Notes**: Inverse class frequency weighted CrossEntropy

## EXP_B2_WEIGHTED_CE_EFFECTIVE
- **Model**: `resnet18`
- **Input**: `48x48` (channels=1)
- **Loss / Sampler**: `cross_entropy` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 38.42%
- **Validation Balanced Accuracy**: 37.85%
- **Validation Macro F1**: 33.51%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 13.11% / 23.45% / 34.83% / 22.30%
- **Notes**: Cui et al. effective number of samples weighting

## EXP_C1_FOCAL_GAMMA_1
- **Model**: `resnet18`
- **Input**: `48x48` (channels=1)
- **Loss / Sampler**: `focal` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 38.67%
- **Validation Balanced Accuracy**: 38.90%
- **Validation Macro F1**: 34.34%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 23.53% / 22.96% / 36.97% / 22.90%
- **Notes**: Focal Loss gamma=1.0 with effective sample weights

## EXP_C2_FOCAL_GAMMA_2
- **Model**: `resnet18`
- **Input**: `48x48` (channels=1)
- **Loss / Sampler**: `focal` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 37.92%
- **Validation Balanced Accuracy**: 36.66%
- **Validation Macro F1**: 32.93%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 14.74% / 24.14% / 31.98% / 24.59%
- **Notes**: Focal Loss gamma=2.0 with effective sample weights

## EXP_C3_FOCAL_GAMMA_3
- **Model**: `resnet18`
- **Input**: `48x48` (channels=1)
- **Loss / Sampler**: `focal` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 37.75%
- **Validation Balanced Accuracy**: 34.73%
- **Validation Macro F1**: 33.15%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 15.62% / 26.79% / 32.81% / 23.35%
- **Notes**: Focal Loss gamma=3.0

## EXP_D1_LABEL_SMOOTHING_005
- **Model**: `resnet18`
- **Input**: `48x48` (channels=1)
- **Loss / Sampler**: `label_smoothing` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 43.92%
- **Validation Balanced Accuracy**: 36.33%
- **Validation Macro F1**: 34.97%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 0.00% / 26.58% / 39.80% / 24.29%
- **Notes**: Label smoothing eps=0.05

## EXP_D2_LABEL_SMOOTHING_010
- **Model**: `resnet18`
- **Input**: `48x48` (channels=1)
- **Loss / Sampler**: `label_smoothing` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 41.25%
- **Validation Balanced Accuracy**: 34.64%
- **Validation Macro F1**: 33.33%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 0.00% / 28.57% / 39.77% / 23.70%
- **Notes**: Label smoothing eps=0.10

## EXP_E1_RESOLUTION_112
- **Model**: `resnet18`
- **Input**: `112x112` (channels=1)
- **Loss / Sampler**: `cross_entropy` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 49.25%
- **Validation Balanced Accuracy**: 41.05%
- **Validation Macro F1**: 39.05%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 0.00% / 30.40% / 42.25% / 31.28%
- **Notes**: 112x112 grayscale input resolution

## EXP_F1_RGB_112
- **Model**: `resnet18`
- **Input**: `112x112` (channels=3)
- **Loss / Sampler**: `cross_entropy` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 50.58%
- **Validation Balanced Accuracy**: 41.71%
- **Validation Macro F1**: 39.66%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 0.00% / 25.85% / 47.14% / 28.82%
- **Notes**: 112x112 RGB with ImageNet normalization

## EXP_G1_CONSERVATIVE_AUG
- **Model**: `resnet18`
- **Input**: `112x112` (channels=3)
- **Loss / Sampler**: `cross_entropy` / `uniform`
- **Augmentation**: `conservative`
- **Validation Accuracy**: 48.50%
- **Validation Balanced Accuracy**: 39.95%
- **Validation Macro F1**: 38.49%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 0.00% / 30.68% / 29.07% / 30.20%
- **Notes**: Conservative facial expression augmentation

## EXP_G2_MODERATE_AUG
- **Model**: `resnet18`
- **Input**: `112x112` (channels=3)
- **Loss / Sampler**: `cross_entropy` / `uniform`
- **Augmentation**: `moderate`
- **Validation Accuracy**: 51.42%
- **Validation Balanced Accuracy**: 42.39%
- **Validation Macro F1**: 41.57%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 0.00% / 28.39% / 42.83% / 28.80%
- **Notes**: Moderate augmentation with ColorJitter

## EXP_H1_MIXUP
- **Model**: `resnet18`
- **Input**: `112x112` (channels=3)
- **Loss / Sampler**: `cross_entropy` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 50.75%
- **Validation Balanced Accuracy**: 42.22%
- **Validation Macro F1**: 39.42%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 0.00% / 33.55% / 43.08% / 14.13%
- **Notes**: MixUp alpha=0.2

## EXP_I1_CUTMIX
- **Model**: `resnet18`
- **Input**: `112x112` (channels=3)
- **Loss / Sampler**: `cross_entropy` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 48.42%
- **Validation Balanced Accuracy**: 41.03%
- **Validation Macro F1**: 39.21%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 0.00% / 26.35% / 38.74% / 27.97%
- **Notes**: CutMix alpha=0.5

## EXP_K1_BACKBONE_RESNET50
- **Model**: `resnet50`
- **Input**: `112x112` (channels=3)
- **Loss / Sampler**: `cross_entropy` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 24.42%
- **Validation Balanced Accuracy**: 16.54%
- **Validation Macro F1**: 9.83%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 0.00% / 0.00% / 1.92% / 1.15%
- **Notes**: ResNet-50 112x112 RGB

## EXP_K2_BACKBONE_EFFICIENTNET_B0
- **Model**: `efficientnet_b0`
- **Input**: `112x112` (channels=3)
- **Loss / Sampler**: `cross_entropy` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 24.33%
- **Validation Balanced Accuracy**: 16.28%
- **Validation Macro F1**: 10.73%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 0.00% / 10.07% / 2.80% / 1.14%
- **Notes**: EfficientNet-B0 112x112 RGB

## EXP_K3_BACKBONE_MOBILENET_V3
- **Model**: `mobilenet_v3_small`
- **Input**: `112x112` (channels=3)
- **Loss / Sampler**: `cross_entropy` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 21.58%
- **Validation Balanced Accuracy**: 17.86%
- **Validation Macro F1**: 16.15%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 0.00% / 15.75% / 10.76% / 1.19%
- **Notes**: MobileNetV3-Small 112x112 RGB

## EXP_K4_BACKBONE_CONVNEXT_TINY
- **Model**: `convnext_tiny`
- **Input**: `112x112` (channels=3)
- **Loss / Sampler**: `cross_entropy` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 28.92%
- **Validation Balanced Accuracy**: 20.73%
- **Validation Macro F1**: 15.85%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 0.00% / 5.94% / 5.61% / 20.00%
- **Notes**: ConvNeXt-Tiny 112x112 RGB

## EXP_L1_ATTENTION_SE
- **Model**: `resnet18_se`
- **Input**: `112x112` (channels=3)
- **Loss / Sampler**: `cross_entropy` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 38.58%
- **Validation Balanced Accuracy**: 31.44%
- **Validation Macro F1**: 28.03%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 0.00% / 17.29% / 28.75% / 9.76%
- **Notes**: ResNet-18 + Squeeze-and-Excitation

## EXP_M1_ATTENTION_CBAM
- **Model**: `resnet18_cbam`
- **Input**: `112x112` (channels=3)
- **Loss / Sampler**: `cross_entropy` / `uniform`
- **Augmentation**: `none`
- **Validation Accuracy**: 32.50%
- **Validation Balanced Accuracy**: 24.76%
- **Validation Macro F1**: 19.78%
- **Minority F1 (Disgust / Fear / Sad / Angry)**: 0.00% / 4.10% / 8.77% / 2.19%
- **Notes**: ResNet-18 + CBAM (Channel & Spatial Attention)

