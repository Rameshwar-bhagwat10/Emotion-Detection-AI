# Baseline CNN Model Training Pipeline

## 1. Overview & Purpose

The **Phase 05 Baseline CNN Model Training Pipeline** establishes a fully reproducible, modular, configurable training infrastructure for the 7-class facial emotion classification task on FER2013 grayscale images.

The training pipeline integrates the verified components from previous phases:
- **Phase 02:** Authoritative 7-class emotion mapping (`angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`).
- **Phase 03:** Deterministic DataLoaders streaming preprocessed `[B, 1, 48, 48]` float32 tensors with zero leakage.
- **Phase 04:** `BaselineCNN` (`baseline_cnn_v1`) producing raw unnormalized class logits `[B, 7]`.

---

## 2. Complete Training Flow & Architecture

```text
Phase 03 Train DataLoader [B, 1, 48, 48]
        │
        ├─► Model Forward (BaselineCNN)
        │       ↓
        │   Raw Logits [B, 7] (No Softmax)
        │       ↓
        ├─► CrossEntropyLoss(logits, targets)
        │       ↓
        ├─► Loss.backward()
        │       ↓
        ├─► [Optional Gradient Clipping]
        │       ↓
        ├─► AdamW Optimizer Step
        │
Phase 03 Validation DataLoader [B, 1, 48, 48]
        │
        ├─► torch.no_grad() Validation
        │       ↓
        │   Sample-Weighted Loss & Accuracy
        │       ↓
        ├─► ReduceLROnPlateau Step(val_loss)
        │       ↓
        ├─► EarlyStopping Check (patience=5, min_delta=0.001)
        │       ↓
        ├─► CheckpointManager (Atomic Write: best.pt, last.pt)
        │       ↓
        └─► Artifact Persistence (history.json, config.yaml, training.log)
```

---

## 3. Input & Output Data Contracts

- **Input Image Batch:** `[B, 1, 48, 48]` (`torch.float32`), single-channel grayscale, normalized with training-set statistics ($\mu = 0.507743, \sigma = 0.255009$).
- **Ground Truth Labels:** `[B]` (`torch.long`), integer indices $\in [0, 6]$ matching canonical class mapping.
- **Model Output Logits:** `[B, 7]` (`torch.float32`), unnormalized logits fed directly to `nn.CrossEntropyLoss`.
- **Softmax Rule:** Softmax is strictly **excluded** from the training pipeline and model architecture; it is reserved exclusively for downstream inference in Phase 08.
- **Test Set Protection:** The `test_loader` is strictly **untouched** during training, validation, early stopping, checkpoint selection, and scheduler adjustments. Evaluation on the test set is reserved exclusively for Phase 06.

---

## 4. Training Hyperparameter Configuration

Centralized in [`ml/configs/training/baseline.yaml`](file:///d:/projects/emotion-detection-ai/ml/configs/training/baseline.yaml):

```yaml
experiment:
  name: baseline_cnn
  version: v1
  description: Baseline CNN training on FER2013 grayscale facial emotion dataset

model:
  name: baseline_cnn

data:
  raw_path: data/raw/fer2013/fer2013.csv
  batch_size: 64
  num_workers: 0
  pin_memory: false

training:
  epochs: 30
  batch_size: 64

  optimizer:
    name: adamw
    learning_rate: 0.001
    weight_decay: 0.0001
    betas: [0.9, 0.999]
    eps: 1.0e-8

  loss:
    name: cross_entropy
    class_weighted: false

  scheduler:
    name: reduce_on_plateau
    mode: min
    factor: 0.5
    patience: 2
    min_lr: 1.0e-6

  early_stopping:
    enabled: true
    monitor: val_loss
    mode: min
    patience: 5
    min_delta: 0.001

  checkpoint:
    save_best: true
    save_last: true
    monitor: val_loss
    mode: min
    save_dir: artifacts/training/baseline_cnn

  mixed_precision:
    enabled: false

  gradient_clipping:
    enabled: false
    max_norm: 1.0

reproducibility:
  seed: 42
  deterministic: true
```

---

## 5. Checkpointing & Resume Mechanics

### Saved Artifacts per Run:
```text
artifacts/training/baseline_cnn/<run_id>/
├── config.yaml          # Exact snapshot of YAML run parameters
├── history.json         # Structured epoch-by-epoch loss, accuracy, lr, duration
├── training.log         # Complete execution log with timestamps
├── best.pt              # Checkpoint containing weights for best observed val_loss
└── last.pt              # Checkpoint containing state for the most recent completed epoch
```

### Checkpoint Contents:
Each `.pt` checkpoint is saved atomically (temporary write $\to$ rename) and encapsulates:
- `epoch`: Last completed epoch index.
- `model_state_dict`: Full model weights (16 parameter tensors + BatchNorm running statistics).
- `optimizer_state_dict`: AdamW momentum buffers and second-moment statistics.
- `scheduler_state_dict`: ReduceLROnPlateau internal state counters and learning rates.
- `scaler_state_dict`: AMP GradScaler state (if mixed precision is enabled).
- `history`: Cumulative training history list up to this epoch.
- `best_metric`: Best recorded validation metric value.
- `best_epoch`: Epoch index at which best metric was achieved.
- `config`: Snapshot of training configuration dictionary.

### Resuming Training:
```bash
python scripts/training/train.py --config ml/configs/training/baseline.yaml --resume artifacts/training/baseline_cnn/<run_id>/last.pt
```
When resumed, training seamlessly starts at `checkpoint["epoch"] + 1`, restores optimizer and scheduler learning rates, preserves history, and continues monitoring from `best_metric`.

---

## 6. CLI Commands & Execution

### Single-Epoch Smoke Test:
```bash
python scripts/training/train.py --config ml/configs/training/baseline.yaml --smoke-test
```

### Full Baseline Training Run:
```bash
python scripts/training/train.py --config ml/configs/training/baseline.yaml
```

### Explicit Device Override:
```bash
python scripts/training/train.py --config ml/configs/training/baseline.yaml --device cpu
```
