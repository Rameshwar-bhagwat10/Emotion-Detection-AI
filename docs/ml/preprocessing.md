# FER2013 Data Preprocessing & Augmentation Pipeline

## 1. Overview & Dataset Input Contract

The **Phase 03 Preprocessing Pipeline** transforms the raw, validated FER2013 facial expression dataset into standardized, model-ready PyTorch tensors and batched DataLoaders for deep learning models (Phase 04).

| Attribute | Specification |
| :--- | :--- |
| **Input Format** | Canonical FER2013 records ($48 \times 48$ 8-bit unsigned integer array $[0, 255]$) |
| **Output Tensor Shape** | Mini-batch: $[B, 1, 48, 48]$ \| Single item: $[1, 48, 48]$ |
| **Output Data Type** | `torch.float32` |
| **Color Channels** | Single-channel Grayscale ($C = 1$) |
| **Emotion Classes** | 7 Discrete Categories ($0\text{..}6$): `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral` |
| **Raw Dataset Immutability** | 100% Immutable — All augmentations and scalings are applied on-the-fly dynamically |

---

## 2. Preprocessing & Transform Pipelines

```text
                        FER2013 RAW DATASET (data/raw/fer2013/fer2013.csv)
                                                │
                                                ▼
                                    Original Benchmark Splits
                                    /           |           \
                                   /            |            \
                        TRAIN (28,709)      VAL (3,589)     TEST (3,589)
                               │                │                │
                               ▼                │                │
                    TRAIN-ONLY AUGMENTATION      │                │
                    • Random HFlip (p=0.5)      │                │
                    • Random Rotation (±12°)    │                │
                    • Random Affine (±5%)       │                │
                               │                │                │
                               └────────┬───────┴────────────────┘
                                        ▼
                                Tensor Conversion
                                        ▼
                                  Pixel Scaling
                                 x_scaled = x / 255.0  (in [0.0, 1.0])
                                        ▼
                               TRAIN-Only Normalization
                               x_norm = (x_scaled - 0.507743) / 0.255009
                                        ▼
                                 PyTorch DataLoader
                                 Images: [B, 1, 48, 48] float32
                                 Labels: [B] int64
```

### 2.1 Training Pipeline (`build_train_transform`)
1. **ToPILImage:** Decodes raw numpy array into PIL Image representation.
2. **On-the-Fly Augmentation:** Applies conservative random horizontal flip, rotation, and affine jitter.
3. **ToTensor:** Converts PIL Image $[0, 255]$ to `torch.FloatTensor` of shape $[1, 48, 48]$ scaled to $[0.0, 1.0]$.
4. **Normalize:** Subtracts training mean ($\mu = 0.507743$) and divides by training standard deviation ($\sigma = 0.255009$).

### 2.2 Validation & Test Pipeline (`build_val_transform` / `build_test_transform`)
1. **ToPILImage:** Converts raw array into PIL Image.
2. **ToTensor:** Converts to `torch.FloatTensor` of shape $[1, 48, 48]$ scaled to $[0.0, 1.0]$.
3. **Normalize:** Uses the exact same **TRAIN** statistics ($\mu = 0.507743, \sigma = 0.255009$).
4. **Zero Random Augmentation:** 100% deterministic evaluation.

---

## 3. Training-Only Normalization Statistics

To mathematically guarantee **zero data leakage**, normalization statistics are calculated **exclusively from the 28,709 training samples**. Validation and test pixels are strictly excluded from statistical calculation.

| Domain Range | Metric | Value (Calculated on TRAIN Only) | Formula |
| :--- | :--- | :---: | :--- |
| **Normalized Range** $[0.0, 1.0]$ | **Mean ($\mu$)** | **`0.507743`** | $\mu = \frac{1}{N \cdot H \cdot W}\sum \frac{x_{train}}{255.0}$ |
| **Normalized Range** $[0.0, 1.0]$ | **Std Dev ($\sigma$)** | **`0.255009`** | $\sigma = \sqrt{\frac{1}{N \cdot H \cdot W}\sum (\frac{x_{train}}{255.0} - \mu)^2}$ |
| **Raw Pixel Range** $[0, 255]$ | **Raw Mean** | `129.474340` | $\mu_{raw} = \mu \times 255.0$ |
| **Raw Pixel Range** $[0, 255]$ | **Raw Std Dev** | `65.027273` | $\sigma_{raw} = \sigma \times 255.0$ |

> [!IMPORTANT]
> **Data Leakage Comparison:**
> - Full Dataset Mean ($35,887$ samples): $129.39$ ($0.507412$)
> - Train Split Mean ($28,709$ samples): $129.47$ ($0.507743$)
>
> Normalizing with the global dataset mean would incorporate validation and test distribution information into the training pipeline. Our pipeline strictly isolates training statistics.

---

## 4. Facial Expression Augmentation Policy

Augmentations are designed to improve model generalization across facial poses, lighting, and camera alignments while strictly preserving expression semantics.

| Augmentation Technique | Hyperparameter Setting | Facial Feature Justification |
| :--- | :---: | :--- |
| **Random Horizontal Flip** | Probability = $0.5$ | Facial expressions are bilaterally symmetric across the sagittal plane (e.g. smiles, eye widening). |
| **Random Rotation** | Degrees = $\pm 12^\circ$ (Bilinear) | Simulates slight head tilt and natural pose variation without altering upright facial geometry. |
| **Random Affine** | Translation = $\pm 5\%$, Scale = $[0.95, 1.05]$ | Simulates slight camera zoom, distance, and face crop centering errors. |
| **Color Jitter** | Disabled (Baseline) | Contrast/brightness jitter is configurable but kept off in baseline to preserve low-contrast micro-expressions. |

### Augmentations Prohibited in Baseline:
- **Vertical Flip:** Faces are never vertically inverted in realistic deployment.
- **90° / 180° Rotations:** Destroys orientation landmarks.
- **Aggressive Random Erasing / Cropping:** Occludes critical facial action units (mouth corners, eyebrows).

---

## 5. Dataset Cardinality & Split Preservation

Cardinalities match the Phase 02 verified benchmarks:

| Split | Sample Count | Percentage | Shuffling | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | **28,709** | 80.0% | `shuffle=True` | Optimization and weight updates |
| **Validation** | **3,589** | 10.0% | `shuffle=False` | Hyperparameter tuning and early stopping |
| **Test** | **3,589** | 10.0% | `shuffle=False` | Final model benchmark evaluation |
| **Total** | **35,887** | 100.0% | — | Full FER2013 canonical dataset |

---

## 6. PyTorch Module Architecture

- [`ml/configs/preprocessing.yaml`](file:///d:/projects/emotion-detection-ai/ml/configs/preprocessing.yaml): Single source of truth for resolutions, normalization parameters, augmentation bounds, and dataloader settings.
- [`ml/preprocessing/normalization.py`](file:///d:/projects/emotion-detection-ai/ml/preprocessing/normalization.py): `calculate_train_normalization_stats`, `normalize`, `denormalize`.
- [`ml/preprocessing/augmentation.py`](file:///d:/projects/emotion-detection-ai/ml/preprocessing/augmentation.py): `AugmentationConfig`, `build_augmentation_pipeline`.
- [`ml/preprocessing/transforms.py`](file:///d:/projects/emotion-detection-ai/ml/preprocessing/transforms.py): `build_train_transform`, `build_val_transform`, `build_test_transform`.
- [`ml/preprocessing/datasets.py`](file:///d:/projects/emotion-detection-ai/ml/preprocessing/datasets.py): `FER2013Dataset` (PyTorch `Dataset`).
- [`ml/preprocessing/dataloaders.py`](file:///d:/projects/emotion-detection-ai/ml/preprocessing/dataloaders.py): `create_train_loader`, `create_val_loader`, `create_test_loader`, `build_dataloaders`.

---

## 7. Generated Visual Artifacts

The following visual and statistical artifacts are saved under `data/interim/fer2013/preprocessing/`:
1. `preprocessing_comparison.png`: Side-by-side progression showing `Raw Image` $\to$ `Horizontal Flip` $\to$ `Rotation` $\to$ `Affine` $\to$ `Final Model-Ready Sample`.
2. `augmentation_samples.png`: $4 \times 4$ variation grid demonstrating realistic stochastic facial transformations.
3. `normalization_distribution.png`: Three-stage histogram showing the progression from raw $[0, 255]$ pixels to scaled $[0.0, 1.0]$ floats and normalized $\mathcal{N}(0, 1)$ distributions.
4. `normalization_report.json`: Machine-readable metadata verifying TRAIN-only sample counts and exact floats.

---

## 8. CLI Execution Commands

### Calculate Training Normalization Statistics
```bash
python scripts/data/calculate-normalization.py
```

### Generate Augmentation & Preprocessing Visualizations
```bash
python scripts/data/visualize-augmentation.py
```

### Run Sanity Check & Verification
```bash
python scripts/data/verify-preprocessing.py
```

### Run Complete Test Suite
```bash
pytest -v
```
