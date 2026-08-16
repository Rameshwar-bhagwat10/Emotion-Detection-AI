# FER2013 Dataset Documentation & Exploratory Data Analysis (EDA)

## 1. Dataset Overview

| Attribute | Value |
| :--- | :--- |
| **Dataset Name** | Facial Expression Recognition 2013 (FER2013) |
| **Primary Task** | 7-Class Facial Expression Classification |
| **Original Origin** | ICML 2013 Challenges in Representation Learning / Kaggle Challenge |
| **Acquisition Source** | Canonical FER2013 Dataset Mirror (`AutumnQiu/fer2013`) |
| **Storage Format** | Canonical CSV (`data/raw/fer2013/fer2013.csv`) & Metadata (`metadata.json`) |
| **Total Samples** | **35,887** |
| **Image Resolution** | **48 × 48 pixels** (100% uniform) |
| **Color Mode** | Grayscale (1 Channel, 8-bit unsigned integer $[0, 255]$) |
| **License** | Open Database License / Research and Educational Use |

---

## 2. Emotion Classes & Label Mapping

The dataset encodes seven discrete basic facial expression categories mapped to integer identifiers $[0..6]$:

| Label ID | Emotion Class | Description & Key Facial Action Units (FAUs) |
| :---: | :---: | :--- |
| **0** | **Angry** | Furrowed brow (AU4), narrowed eyelids (AU7), compressed lip margin (AU23/24). |
| **1** | **Disgust** | Nose wrinkler (AU9), upper lip raiser (AU10), lowered brow. |
| **2** | **Fear** | Raised eyebrows (AU1+2), wide open eyes (AU5), parted lips (AU25). |
| **3** | **Happy** | Cheek raiser (AU6), lip corner puller / Duchenne smile (AU12). |
| **4** | **Sad** | Inner brow raiser (AU1), brow lowerer (AU4), lip corner depressor (AU15). |
| **5** | **Surprise** | High inner/outer brow raiser (AU1+2), upper lid raiser (AU5), jaw drop (AU26). |
| **6** | **Neutral** | Relaxed facial musculature without prominent emotional Action Units. |

---

## 3. Verified Dataset Statistics

All values below have been derived directly from the verified raw dataset (`35,887` records):

### 3.1 Split Distribution

| Split | Usage Tag in Raw CSV | Sample Count | Percentage |
| :--- | :--- | :---: | :---: |
| **Train** | `Training` | 28,709 | 80.00% |
| **Validation** | `PublicTest` | 3,589 | 10.00% |
| **Test** | `PrivateTest` | 3,589 | 10.00% |
| **Total** | — | **35,887** | **100.00%** |

### 3.2 Class Distribution per Split

| Emotion Class | Train Count | Val Count | Test Count | Overall Count | Overall Share |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Happy** | 7,215 | 895 | 879 | **8,989** | **25.05%** |
| **Neutral** | 4,965 | 607 | 626 | **6,198** | **17.27%** |
| **Sad** | 4,830 | 653 | 594 | **6,077** | **16.93%** |
| **Fear** | 4,097 | 496 | 528 | **5,121** | **14.27%** |
| **Angry** | 3,995 | 467 | 491 | **4,953** | **13.80%** |
| **Surprise** | 3,171 | 415 | 416 | **4,002** | **11.15%** |
| **Disgust** | 436 | 56 | 55 | **547** | **1.52%** |
| **Total** | **28,709** | **3,589** | **3,589** | **35,887** | **100.00%** |

---

## 4. Class Imbalance Analysis

```text
Most Represented Class:   Happy (8,989 samples | 25.05%)
Least Represented Class:  Disgust (547 samples | 1.52%)
Class Imbalance Ratio:    16.43 : 1
```

### Observations & Downstream Implications:
1. **Severe Minority Class (`disgust`):** At only 547 total instances (1.52%), unweighted cross-entropy loss risks ignoring the disgust class in favor of majority classes.
2. **Mitigation Strategy (Phase 03 / Phase 05):**
   - Class-weighted loss functions (e.g. Focal Loss, inverse frequency class weights).
   - Targeted minority class data augmentation in Phase 03.
   - Stratified mini-batch sampling during DataLoader configuration.

---

## 5. Image Characteristics & Pixel Distribution

- **Resolution:** 48 × 48 pixels ($H=48, W=48$).
- **Color Channels:** 1 (Single channel grayscale).
- **Pixel Intensity Range:** $[0, 255]$ ($8\text{-bit}$ integer).
- **Mean Pixel Intensity ($\mu$):** **$129.39$**
- **Pixel Standard Deviation ($\sigma$):** **$65.05$**
- **Min / Max Pixel Values:** $0.0 / 255.0$

---

## 6. Data Quality & Integrity Audit

| Check Category | Inspected Items | Result | Notes |
| :--- | :---: | :---: | :--- |
| **Schema Completeness** | 35,887 records | **PASS** | `emotion`, `pixels`, `Usage` headers present and valid. |
| **Missing Records / NaNs** | 35,887 records | **PASS** | 0 nulls, NaNs, or empty strings. |
| **Corrupt Image Records** | 35,887 records | **PASS** | 0 decoding failures; all 2,304 pixel integers parse cleanly. |
| **Dimension Integrity** | 35,887 records | **PASS** | 100% of images conform to (48, 48) shape. |
| **Label Range Verification** | 35,887 records | **PASS** | 100% of labels strictly fall in integer range $[0..6]$. |

---

## 7. Duplicate & Data-Leakage Audit

An exact pixel hash analysis (MD5 fingerprinting) was conducted across the 35,887 samples:

| Metric | Measured Value |
| :--- | :---: |
| **Total Samples** | 35,887 |
| **Unique Samples** | 34,034 |
| **Exact Duplicate Samples** | 1,853 |
| **Duplicate Groups** | 1,516 |
| **Duplicates Within Train** | 1,236 |
| **Duplicates Within Val** | 26 |
| **Duplicates Within Test** | 17 |
| **Cross-Split Overlap: Train $\leftrightarrow$ Val** | **270** |
| **Cross-Split Overlap: Train $\leftrightarrow$ Test** | **278** |
| **Cross-Split Overlap: Val $\leftrightarrow$ Test** | **43** |

### Data Leakage Findings & Phase 03 Guidance:
> [!WARNING]
> The raw FER2013 benchmark contains **591 cross-split exact duplicates** between `train`, `val`, and `test` splits.
> While preserving the canonical benchmark splits is required for reproducible literature comparison, training pipelines in Phase 03 will include an option to deduplicate or quarantine evaluation contamination.

---

## 8. Visual & Morphological Observations

1. **Tight Cropping & Face Alignment:** Faces in FER2013 are closely cropped around facial bounding boxes, with varying quality in centering.
2. **Resolution Artifacts:** At 48 × 48 resolution, fine micro-expressions and subtle eye gaze movements are low-frequency, making global facial geometry (mouth curves, brow furrows) primary discriminative features.
3. **Expression Ambiguity:**
   - **Fear vs. Surprise:** Characterized by overlapping brow raising ($AU1+2$) and open mouths ($AU25/26$).
   - **Sad vs. Neutral:** Subtle down-turned lip corners can visually mimic neutral resting face under flat lighting.
   - **Angry vs. Disgust:** Shared brow furrowing ($AU4$) and upper lip tension.
4. **Lighting & Occlusion:** Moderate presence of glasses, hand-to-face occlusions, watermark artifacts, and strong directional lighting.

---

## 9. Dataset Limitations & Ethical AI Boundaries

> [!IMPORTANT]
> **Scientific & Ethical Distinction: Visible Facial Expression vs. Psychological Emotion**
> 
> 1. **Classification Boundary:** This system trains machine learning models to detect **visible, surface-level facial expressions** based on observed morphological patterns (Facial Action Coding System).
> 2. **Not a Psychological Diagnostic Tool:** The predictions generated by this system **do not diagnose, determine, or infer internal psychological or emotional states**, cognitive intent, or mental health conditions.
> 3. **Contextual Diversity:** Facial expressions vary significantly across cultures, neurotypes, contexts, and personal communicative styles. The model should never be used as a lie detector, employment assessment barrier, or psychiatric diagnostic instrument.

---

## 10. EDA Execution Commands

To reproduce the dataset acquisition, validation, and exploratory data analysis:

```bash
# 1. Acquire FER2013 dataset
python scripts/data/download-dataset.py

# 2. Validate dataset integrity
python scripts/data/validate-dataset.py

# 3. Run complete Exploratory Data Analysis & generate reports/charts
python scripts/data/run-eda.py
```

Generated outputs will be saved in:
- `data/interim/fer2013/eda/charts/` (High-res bar charts, split graphs, pixel distributions)
- `data/interim/fer2013/eda/samples/` (Class-wise and random sample grids)
- `data/interim/fer2013/eda/reports/` (JSON statistical and leakage reports)
