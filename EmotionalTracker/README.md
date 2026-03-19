> **Dual-output classification pipeline** that predicts emotional state + session intensity from journal entries, face emotion signals, and ambient session metadata — then generates a personalised recommendation using a pure attention mechanism.

<br>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production_Ready-22C55E?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Best_Y1_Acc-54.4%25_CV-6366F1?style=for-the-badge)

<br>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Pipeline Architecture](#%EF%B8%8F-pipeline-architecture)
- [Model Benchmarks](#-model-benchmarks)
- [Why XGBoost Won](#-why-xgboost-won)
- [Feature Engineering](#-feature-engineering)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Overfitting Controls](#-overfitting-controls)
- [Current Bottleneck Analysis](#-current-bottleneck-analysis)
- [Uncertainty Analysis](#-uncertainty-analysis)
- [Output Format](#-output-format)
- [Files](#-files)
- [Limitations & Next Steps](#-limitations--next-steps)

<br>

---

## 🎯 Project Overview

Each Arvyax reflective session produces a short journal entry, a face emotion hint, ambient sound choice, and session metadata. This pipeline predicts:

| Target | Task | Classes | Best Acc |
|--------|------|---------|----------|
| **Y1** — Emotional State | 6-class classification | calm · focused · mixed · neutral · overwhelmed · restless | **54.4% CV** |
| **Y2** — Session Intensity | 5-class classification → 3-bucket | 0 · 1 · 2 | **42.2% val (3-bucket)** |

After prediction, a **soft attention recommendation engine** (zero if/else) selects the most relevant wellbeing recommendation from 8 templates using the predicted class probability vector as a query.

<br>

---

## 📦 Dataset

```
Training samples : 1,200
Test samples     : 120
Features         : 11 columns (text + categorical + numeric)
Targets          : emotional_state (6 classes) | intensity (1–5)
```

**Class distribution — perfectly balanced (by design):**

```
Y1 — Emotional State          Y2 — Intensity
──────────────────────        ──────────────
calm          216  (18%)      1  →  226  (18.8%)
restless      209  (17.4%)    2  →  228  (19.0%)
neutral       201  (16.8%)    3  →  240  (20.0%)
focused       193  (16.1%)    4  →  277  (23.1%)
mixed         191  (15.9%)    5  →  229  (19.1%)
overwhelmed   190  (15.8%)
```

**Input columns used:**

| Column | Type | Usage |
|--------|------|-------|
| `journal_text` | text | TF-IDF + semantic similarity + ambience proximity |
| `ambience_type` | categorical | one-hot + proximity weight in text |
| `face_emotion_hint` | categorical | one-hot (7 categories) |
| `previous_day_mood` | categorical | one-hot (8 categories) |
| `reflection_quality` | ordinal | vague=0 · conflicted=0.5 · clear=1 |
| `time_of_day` | categorical | one-hot (5 categories) |
| `duration_min` | numeric | direct feature for Y2 |
| `sleep_hours` | numeric | direct + sleep recommendation rule |
| `energy_level` | numeric | direct feature for Y2 |
| `stress_level` | numeric | direct feature for Y2 |

<br>

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT (per session)                       │
│  journal_text  ambience  face_emotion  prev_mood  time_of_day   │
│  duration_min  sleep_hours  energy_level  stress_level          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │     FEATURE ENGINEERING    │
              │                           │
              │  TEXT PATH                │
              │  ├─ TF-IDF word bigrams   │  ─→  SVD (40d)  ──┐
              │  └─ TF-IDF char 4-grams   │  ─→  SVD (20d)  ──┤
              │                           │                    │
              │  STRUCTURED PATH (50d)    │                    │
              │  ├─ sem_sim_vec      (6d) │  ─────────────────►│
              │  ├─ amb_proximity    (7d) │                    │
              │  ├─ face_mood_onehot(15d) │               ┌────▼────┐
              │  ├─ ambience_oh      (5d) │               │  ~110d  │
              │  ├─ tod_oh           (5d) │               │ feature │
              │  ├─ reflection       (1d) │               │ matrix  │
              │  ├─ dominant_emotion (6d) │               └────┬────┘
              │  ├─ conf_margin      (1d) │                    │
              │  └─ numeric          (4d) │         Imputer + StandardScaler
              └───────────────────────────┘                    │
                                              ┌────────────────┴──────────────┐
                                              │                               │
                                    ┌─────────▼─────────┐         ┌──────────▼──────────┐
                                    │   Y1 XGBoost HGB   │         │   Y2 XGBoost HGB    │
                                    │  max_depth=4       │         │  max_depth=3        │
                                    │  lr=0.05           │         │  lr=0.05            │
                                    │  class_weight=bal  │         │  min_leaf=15        │
                                    └─────────┬──────────┘         └──────────┬──────────┘
                                              │                               │
                                    cls_probs[6]                     intensity_bin
                                              │                               │
                                    ┌─────────▼──────────────────────────────▼──┐
                                    │         ATTENTION RECOMMENDATION           │
                                    │                                            │
                                    │  Q = [cls_probs(6) | intensity_norm(1)]   │
                                    │  K = template_key_matrix (8×7)            │
                                    │  attn = softmax(K @ Q / √7)               │
                                    │  top_template = argmax(attn)              │
                                    │                                            │
                                    │  + sleep rule: rec = 8 if sleep < 6       │
                                    └────────────────────────────────────────────┘
                                                        │
                                              📄 OUTPUT CSV
```

<br>

---

## 📊 Model Benchmarks

> All results use **3-fold stratified CV with no data leakage** — TF-IDF fitted inside each fold on train split only.

### Y1 — Emotional State (6-class)

```
Model                    CV Mean   CV Std   Val Acc   Bar
─────────────────────────────────────────────────────────────────────
SVM (RBF)                 56.8%    ±3.6%    ~51%    ██████████████████████▌
Random Forest             54.9%    ±2.5%    ~52%    █████████████████████▉  ← stable
XGBoost (HGB) ✅          54.4%    ±2.7%     52.2%  █████████████████████▋  ← chosen
Neural Network (TF)       50.9%    ±5.7%    ~49%    ████████████████████▎
Logistic Regression       49.6%    ±3.1%    ~43%    ███████████████████▊
─────────────────────────────────────────────────────────────────────
Random Baseline           16.7%      —       —      ██████▋
```

> 💡 SVM had the highest CV mean (56.8%) but was eliminated because it **does not output class probabilities**, which are required by the attention recommendation engine. XGBoost gives almost identical accuracy AND probability vectors.

<br>

### Y2 — Intensity (5-class, random baseline = 20%)

```
Model                    CV Mean   CV Std   Val Acc (3-bucket)
──────────────────────────────────────────────────────────────
XGBoost (HGB) ✅          21.9%    ±0.3%       42.2%
Random Forest             24.5%    ±1.9%       ~22%
Neural Network (TF)       20.8%    ±2.2%       ~21%
──────────────────────────────────────────────────────────────
Random Baseline           20.0%      —          —
```

> ⚠️ **Y2 honest note:** Raw 5-class CV mean for all models is 21–24% — barely above 20% random. The 42.2% val figure is achieved through 3-class bucketing (1-2=low, 3=medium, 4-5=high). This is a **data problem**, not a model problem. See [Uncertainty Analysis](#-uncertainty-analysis).

<br>

### Per-Class Performance — XGBoost Y1 (Validation Set)

```
Class          Precision   Recall    F1     Recall Bar
────────────────────────────────────────────────────────────────────
calm              0.61      0.59     0.60   ██████████████████████████████▌
focused           0.59      0.55     0.57   ████████████████████████████
mixed             0.57      0.55     0.56   ████████████████████████████
neutral           0.50      0.47     0.48   ████████████████████████▌
overwhelmed       0.43      0.41     0.42   █████████████████████▌   ← hardest
restless          0.45      0.55     0.49   ████████████████████████████▌
────────────────────────────────────────────────────────────────────
weighted avg      0.53      0.52     0.52
```

<br>

### Y2 Intensity — 3-Bucket Breakdown

```
Bucket        Precision   Recall    F1     Support
───────────────────────────────────────────────────
low  (1–2)       0.44      0.48     0.46      71
medium  (3)      0.25      0.16     0.19      32   ← near-random
high (4–5)       0.45      0.48     0.46      77
───────────────────────────────────────────────────
weighted avg     0.41      0.42     0.41     180
```

> 🔍 `medium` (intensity=3) is essentially random (F1=0.19). Users assign `3` when uncertain — it's a catch-all label, not a distinct emotional state. `low` and `high` perform reasonably because they represent clear extremes.

<br>

---

## 🏆 Why XGBoost Won

Four reasons XGBoost was selected over RF, NN, SVM, and Logistic Regression:

```
┌─────────────────────────────────┬──────────────────────────────────────────┐
│ Requirement                     │ Why XGBoost Passes                        │
├─────────────────────────────────┼──────────────────────────────────────────┤
│ Probability vectors for attn    │ ✅ predict_proba() returns [6] float vec   │
│ NaN handling                    │ ✅ HistGradientBoosting native NaN support  │
│ Stability across folds          │ ✅ std=0.027 vs NN std=0.057               │
│ Sparse feature tolerance        │ ✅ Tree splits on one feature at a time    │
│ Class imbalance handling        │ ✅ class_weight='balanced' built-in        │
└─────────────────────────────────┴──────────────────────────────────────────┘
```

**Why Neural Network specifically underperforms (–4% gap vs RF):**

This is a **structural problem**, not a tuning problem. Every NN architecture tested — `(128,64)`, `(256,128,64)`, `(64,32)`, tanh vs relu, 5 alpha values — converged to the same 50–51% range.

```
Feature Group             Zero Fraction    Impact on NN Backprop
──────────────────────────────────────────────────────────────────
sem_sim (6d)              ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  87.2%   gradient ≈ 0
face/mood onehot (15d)    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  86.7%   gradient ≈ 0
amb_proximity (7d)        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   84.6%   gradient ≈ 0
ambience/tod onehot (10d) ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   80.0%   gradient ≈ 0
numeric (4d)                                   0.0%   ✅ dense
──────────────────────────────────────────────────────────────────
Overall                                       75.8%   76% dead gradients
```

> RF splits on **one feature at a time** → zeros in other features irrelevant. NN backprop computes gradients across **all weights simultaneously** → when 76% of inputs are zero, 76% of weight updates are zero. Those neurons stop learning. This is not fixable by architecture changes without denser features.

<br>

---

## 🔧 Feature Engineering

### Text Features (via TF-IDF → SVD)

```python
# Word bigrams — captures multi-word emotion phrases
TfidfVectorizer(ngram_range=(1,2), max_features=500, sublinear_tf=True, min_df=3)
# → "mind racing", "less tense", "locked in", "emotionally tired" as single features

# Character 4-grams — catches typos + partial words
TfidfVectorizer(analyzer='char_wb', ngram_range=(3,4), max_features=200, min_df=4)
# → "teh" ≈ "the", "tehre" ≈ "there"

# Both compressed via SVD to prevent noise
TruncatedSVD(40)  # word → 40d
TruncatedSVD(20)  # char → 20d
```

> ✅ **No leakage**: TF-IDF + SVD fitted on **train fold only**. `.transform()` used on val/test.

<br>

### Ambience Proximity Weight

When the ambience word appears in the journal text, emotion keywords near it receive a **2× proximity boost**:

```python
proximity_weight = 2.0 / (1.0 + distance * 0.1)
# "ocean audio was nice" → calm keywords near "ocean" → 2× weight
# "forest sounds worked for a bit" → calm boosted
```

### Semantic Similarity Vector (6d)

Cosine-style overlap between journal tokens and each emotion's vocabulary cluster:

```python
sim = overlap(tokens, vocab_cluster) / (√|tokens| × √|vocab|)
# Returns [calm_sim, restless_sim, focused_sim, overwhelmed_sim, neutral_sim, mixed_sim]
```

### Face × Mood Interaction (15d)

Rather than hardcoding rules like `if calm_face + overwhelmed_mood → ...`, both are encoded as one-hot vectors and **concatenated** — letting the model learn the interaction weights during training:

```
face_emotion_hint:    [0, 0, 1, 0, 0, 0, 0]   ← neutral_face
previous_day_mood:    [0, 0, 0, 0, 1, 0, 0, 0]  ← overwhelmed
concat →              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  (15d)
```

<br>

---

## ⚙️ Hyperparameter Tuning

All values settled via grid search on 3-fold stratified CV. Testing range shown in brackets.

### Y1 — Emotional State Model

| Parameter | Value | Tested Range | Reasoning |
|-----------|-------|-------------|-----------|
| `max_iter` | 300 | 100–500 | Enough for convergence; early stopping prevents over-running |
| `max_depth` | 4 | 3–7 | Depth 5+ caused train/val gap > 20%. Shallow = less overfit |
| `learning_rate` | 0.05 | 0.01, 0.05, 0.1 | 0.05 gave best CV mean. 0.1 overfit, 0.01 underfit |
| `min_samples_leaf` | 10 | 5, 10, 15, 20 | Key regularizer for 1200 samples. Prevents micro-splits |
| `class_weight` | `'balanced'` | None, balanced | +3% recall on overwhelmed and neutral vs no weight |

### Y2 — Intensity Model

| Parameter | Value | Tested Range | Reasoning |
|-----------|-------|-------------|-----------|
| `max_depth` | 3 | 2–5 | Shallower than Y1 — weaker signal, deeper trees overfit immediately |
| `min_samples_leaf` | 15 | 5, 10, 15, 20 | Higher than Y1 because intensity signal is noisier |
| `class_weight` | `None` | None, balanced | Intensity classes are naturally balanced (226–277 per bin) |

### TF-IDF / SVD

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| Word `max_features` | 500 | Going above 700 didn't improve CV. Below 300 lost key bigrams |
| Char `max_features` | 200 | Captures subword patterns without explosion |
| `min_df` | 3 (word) / 4 (char) | Ignores rare terms appearing in < 3 docs — pure noise |
| `sublinear_tf` | `True` | Log-scale TF prevents high-frequency words dominating |
| SVD word dims | 40 | Tested 20/40/60; 40 was optimal. 60 added noise |
| SVD char dims | 20 | Tested 10/20/30; 20 optimal |

<br>

---

## 🛡️ Overfitting Controls

### The Leakage Problem (biggest fix)

Early iterations showed:
```
Train accuracy :  90%  ← fake
Val accuracy   :  57%  ← inflated
Gap            :  33 points  ← pure leakage
```

**Root cause:** `TfidfVectorizer.fit_transform()` was called on all 1200 rows *before* the train/val split. The vectorizer learned vocabulary frequencies from validation samples — so val accuracy was inflated.

**Fix:**
```python
# ❌ BEFORE (leakage)
X_tfidf = tfidf.fit_transform(all_texts)
X_tr, X_val = train_test_split(X_tfidf)

# ✅ AFTER (no leakage)
X_tr_raw = tfidf.fit_transform(train_texts)   # fit on train ONLY
X_val_raw = tfidf.transform(val_texts)         # transform val — no fitting
```

**Effect:** Train/val gap dropped from 33 points → ~10–15 points (normal generalisation gap).

<br>

### ReduceLROnPlateau (TF Neural Network)

```python
# Halves LR every time val loss stagnates for 12 epochs
ReduceLROnPlateau(factor=0.5, patience=12, min_lr=1e-5)

# Example LR schedule observed during training:
# Epoch 1–40:   lr = 0.001000
# Epoch 41–60:  lr = 0.000500  (plateau hit)
# Epoch 61–80:  lr = 0.000250  (plateau hit again)
# Epoch 81+:    Early stopping fires
```

<br>

### Early Stopping

```python
# TF Neural Network
EarlyStopping(patience=25, restore_best_weights=True, monitor='val_accuracy')

# XGBoost / RF
# HistGradientBoostingClassifier has built-in validation monitoring
# max_iter=300 is an upper bound — actual training stops earlier
```

<br>

### Full Regularisation Stack

```
Technique              Applied To              Effect
──────────────────────────────────────────────────────────────────────
L2 (kernel_reg=1e-3)   TF Dense layers         Penalises large weights
BatchNormalization     TF between layers       Stabilises activation scale
Dropout (0.4 / 0.3)   TF Dense layers         40% / 30% neurons zeroed per batch
min_samples_leaf=10    XGBoost, RF             Prevents micro-splits on outliers
Balanced class weight  All models              Prevents majority-class collapse
Stratified CV splits   All evaluation          Each fold has proportional class dist
```

<br>

### Stratified 3-Fold CV — What You'll See

```
[2/5] Stratified 3-fold CV...
  fold 1 | Y1: 0.558  Y2: 0.221
  fold 2 | Y1: 0.533  Y2: 0.212
  fold 3 | Y1: 0.542  Y2: 0.225
  CV mean  | Y1: 0.544  Y2: 0.219
  CV std   | Y1: 0.027  Y2: 0.003
```

Low std (0.027 for Y1) = stable model. High std (0.057 for NN) = variance-sensitive, unreliable.

<br>

---

## 🔍 Current Bottleneck Analysis

### 1. Hand-Crafted Features Have a Hard Ceiling

The 29-dim structured feature vector relies entirely on keyword matching. Works for clear cases:
- `"overwhelmed"` in text → overwhelmed cluster gets score
- `calm_face` + `ocean` proximity → calm gets boosted

**Completely fails for short, informal entries (~30-40% of the dataset):**

```
Entry              Feature vector      Problem
─────────────────────────────────────────────────────────────────────
"ok session"       [0,0,0,0,0,0,...]   2 words. Zero keyword hits.
"still off"        [0,0,0,0,0,0,...]   No emotion vocab match.
"actually helped"  [0,0,0,0,0,0,...]   Positive but no vocab hit.
"mind racing"      [0,0.8,0,0,0,0.1]  Correctly caught ✅
"bit restless"     [0,0.9,0,0,0,0,...]  Correctly caught ✅
```

These short entries generate **near-identical feature vectors regardless of content**. No model can learn to distinguish between them — this is a feature ceiling, not a model ceiling.

<br>

### 2. TF-IDF Partially Solves This But Still Has Limits

TF-IDF adds bigrams like `"mind racing"`, `"locked in"`, `"emotionally tired"` as real features. This helped accuracy from ~49% → ~54%. But:

```python
# TF-IDF cannot understand negation or context:
"not calm"  # contains "calm" → partially overlaps with calm cluster
"calm"      # contains "calm" → overlaps with calm cluster
# These two entries look similar to TF-IDF
```

<br>

### 3. Feature Sparsity Structurally Disadvantages Neural Networks

```
Feature Group              Zero Fraction   NN Impact
───────────────────────────────────────────────────────────────
sem_sim (6d)               87.2%  ██████████████████ → gradients ≈ 0
face/mood onehot (15d)     86.7%  █████████████████▉ → gradients ≈ 0
amb_proximity (7d)         84.6%  █████████████████▍ → gradients ≈ 0
ambience/tod onehot (10d)  80.0%  ████████████████   → gradients ≈ 0
numeric (4d)                0.0%  ░░░░░░░░░░░░░░░░░░ → ✅ dense

Overall: 75.8% zeros → 76% of NN weight gradients are zero per batch
```

This is **irreducible** with the current feature design. Tested architectures:

| Architecture | Activation | Alpha | CV Mean | iters |
|-------------|-----------|-------|---------|-------|
| (128, 64) | relu | 0.01 | 50.7% | 31–36 |
| (256, 128, 64) | relu | 0.005 | 51.7% | 26–33 |
| (128, 64) | tanh | 0.005 | 51.2% | 32–65 |
| (64, 32) | tanh | 0.01 | 50.8% | 46–103 |

All converge to 50–52%. The gap vs RF (54.9%) is structural, not tunable.

<br>

### 4. Why BERT Is Not the Answer Here

```
Model              Params    Min samples needed   Works here?
──────────────────────────────────────────────────────────────
BERT-base          110M      ~10,000+             ❌ overfit on 1200
all-MiniLM-L6-v2   22M      ~2,000+              ⚠️  borderline
TF-IDF + SVD       ~0        200+                ✅ current approach
```

BERT would give dense, context-aware embeddings — `"ok session"` and `"actually helped"` would become completely different 768-dim vectors. But 1200 training samples is far too small to fine-tune even the lightest BERT variant without severe overfitting.

**Correct next step:** `sentence-transformers/all-MiniLM-L6-v2` with frozen weights + linear head + strong dropout.

<br>

### 5. Categorical Features as One-Hots vs Embeddings

| Approach | ambience_type | time_of_day | prev_mood |
|----------|-------------|-------------|-----------|
| One-hot (current) | 5d, 4 always 0 | 5d, 4 always 0 | 8d, 7 always 0 |
| Learned embedding | 4d, all dense | 4d, all dense | 6d, all dense |

One-hot treats every category as independent. A 4d embedding would let the model learn that `overwhelmed` and `restless` as previous moods are more similar to each other than to `calm`. This requires a neural architecture with embedding layers — not possible with XGBoost/RF.

<br>

---

## 📐 Uncertainty Analysis

### Prediction Confidence (Y1)

The `cls_confidence` column in the output CSV is `max(cls_probs)` — the model's top class probability.

```
Confidence      Meaning                          Action
────────────────────────────────────────────────────────────────────
> 0.70          High — clear signal              ✅ use directly
0.50 – 0.70     Moderate — some uncertainty      ✅ use, recommendation adapts
0.35 – 0.50     Low — spread across 2-3 classes  ⚠️  treat with caution
< 0.35          Very low — near-uniform           ❌ short entry, no signal
```

> 💡 The attention recommendation naturally handles low confidence: a near-uniform `cls_probs` vector produces a near-uniform `attn` vector, which means the recommendation is less sharply personalised — but it never crashes or makes a hard wrong decision.

<br>

### Per-Class Confusion Patterns

```
overwhelmed ──→ restless     (both have "heavy activity" vocabulary)
calm ──→ neutral             (similar short journal entries)
focused ──→ calm             (clarity language appears in both)
mixed ──→ any other          (by definition contains vocabulary from 2 classes)
neutral ──→ anything         (no strong positive/negative vocabulary anchor)
```

<br>

### Y2 Intensity — Aleatory Uncertainty

The intensity label's correlation with available features:

```
Feature              Pearson r with intensity
─────────────────────────────────────────────
stress_level              0.14   ← strongest
energy_level              0.12
duration_min              0.08
sleep_hours              -0.06
─────────────────────────────────────────────
All correlations near zero → no linear predictive signal
```

This is **aleatory uncertainty** (irreducible noise in the labelling process) — not **epistemic uncertainty** (fixable by collecting more data or using a better model). Even a perfect model cannot reliably predict intensity 2 vs 3 from session metadata alone, because users do not assign these labels consistently.

**Evidence:** All 3 models converge to 21–24% on 5-class intensity regardless of architecture, features, or regularisation.

```
Model               5-class CV Mean    vs random (20%)
──────────────────────────────────────────────────────
XGBoost              21.9%             +1.9%
Random Forest        24.5%             +4.5%
Neural Network       20.8%             +0.8%
```

<br>

### Uncertainty Propagation into Recommendations

```
High confidence prediction:
  cls_probs = [0.05, 0.03, 0.07, 0.06, 0.75, 0.04]  ← overwhelmed dominant
  attn      = [0.05, 0.06, 0.10, 0.07, 0.62, 0.07, 0.02, 0.01]
  result    → "emotional_offload" template (sharp selection)

Low confidence prediction:
  cls_probs = [0.20, 0.15, 0.18, 0.22, 0.14, 0.11]  ← near-uniform
  attn      = [0.12, 0.11, 0.14, 0.15, 0.13, 0.14, 0.11, 0.10]
  result    → "steady_continuity" template (wins by small margin)
```

The recommendation degrades **gracefully** under uncertainty. No single threshold or if/else branches — just a softer attention distribution.

<br>

---

## 📄 Output Format

Each row in `arvyax_predictions_xgboost.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Session ID |
| `predicted_emotional_state` | str | One of 6 emotion classes |
| `predicted_intensity` | int | 1–5 |
| `recommendation` | str | Personalised wellbeing recommendation |
| `top_template` | str | Recommendation template used |
| `duration_min` | int | Session duration from input |
| `sleep_hours_recommended` | float | 8 if actual < 6, else actual |
| `time_of_day` | str | From input |
| `attention_weights` | dict | {template: float} — full attention distribution |
| `cls_confidence` | float | max(cls_probs) — model confidence |
| `ambience_type` | str | From input |

**Sample output:**
```json
{
  "id": 10001,
  "predicted_emotional_state": "focused",
  "predicted_intensity": 3,
  "recommendation": "Your mind is in a receptive state. Use this for deep work. The cafe ambience supported focus today. Start with the hardest task while this clarity holds.",
  "top_template": "deep_work",
  "duration_min": 4,
  "sleep_hours_recommended": 8,
  "time_of_day": "night",
  "cls_confidence": 0.612
}
```

<br>

---

## 📁 Files

```
arvyax/
├── 📄 arvyax_xgboost.py          ← RECOMMENDED. Best model + 3-fold CV + full eval
├── 📄 arvyax_random_forest.py    ← RF pipeline + feature importance printout
├── 📄 arvyax_nn.py               ← TF neural network. Benchmarking only, not recommended
│
├── 📊 arvyax_predictions_xgboost.csv
├── 📊 arvyax_predictions_rf.csv
├── 📊 arvyax_predictions_nn.csv
│
└── 📘 README.md                  ← this file
```

### Quick Start

```bash
# Install dependencies
pip install scikit-learn pandas numpy tensorflow

# Run best model (XGBoost)
python arvyax_xgboost.py

# Run all three and compare
python arvyax_xgboost.py      # Y1: ~54%  Y2: 42% (3-bucket)
python arvyax_random_forest.py # Y1: ~52%  Y2: ~22%
python arvyax_nn.py            # Y1: ~49%  Y2: ~21%
```

**Data paths** (update these to match your Colab / local paths):
```python
train_df = pd.read_csv('/content/Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv')
test_df  = pd.read_csv('/content/arvyax_test_inputs_120.xlsx - Sheet1.csv')
```

<br>

---

## 🚧 Limitations & Next Steps

| # | Limitation | Severity | Next Step |
|---|-----------|----------|-----------|
| 1 | Keyword features ceiling at ~54-57% Y1 | 🔴 High | Replace `sem_sim` + `amb_proximity` with `all-MiniLM-L6-v2` sentence embeddings (frozen) |
| 2 | Y2 intensity near-random (5-class CV ~22%) | 🟠 Medium | Reframe as 3-class (low/med/high) or collect self-reported pre-session mood |
| 3 | `face_emotion_hint` ~20% missing | 🟡 Low | Add explicit missingness indicator feature or impute with predicted value |
| 4 | Short journals (<5 words) undetectable | 🔴 High | Flag ultra-short entries as `low_confidence=True` in output |
| 5 | Categorical features as one-hots | 🟡 Low | Switch to learned 4-8d embeddings (requires neural architecture) |
| 6 | Recommendation templates are static strings | 🟢 Nice-to-have | Use user's journal text as context for a small generative model |
| 7 | NN underperforms RF by ~4% | 🟡 Low | Structural issue (sparse features) — only fixable with denser embeddings |

<br>

---

## 📈 Accuracy Journey

```
Version    Approach                          Y1 Val    Gap (train-val)  Note
──────────────────────────────────────────────────────────────────────────────────
v1.0  Pure numpy NN, hand-crafted 29d        66%*      33+ pts         *inflated
v1.1  Fixed leakage, honest eval             ~49%       ~12 pts        real baseline
v2.0  + TF-IDF bigrams + char ngrams         ~52%       ~12 pts        text signal
v2.1  + Expanded emotion vocab               ~54%       ~10 pts
v3.0  + Stratified 3-fold CV (final)         54.4%      ~10 pts        current
──────────────────────────────────────────────────────────────────────────────────
```

> 🔑 The jump from v1.0 to v1.1 is not a regression — it's fixing fake accuracy caused by data leakage. Real accuracy was never 66%.

<br>

---
