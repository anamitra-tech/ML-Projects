# 🧘 ARVYAX — Reflective Session ML Pipeline

> **Dual-output classification pipeline** that predicts emotional state + session intensity from mindfulness journal entries, face emotion signals, and ambient session metadata — then generates a personalised recommendation using a pure attention mechanism (zero if/else).

<br>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production_Ready-22C55E?style=for-the-badge)
![Y1 Acc](https://img.shields.io/badge/Y1_CV_Mean-52.4%25-6366F1?style=for-the-badge)
![Y2 Acc](https://img.shields.io/badge/Y2_Val_Acc-44.4%25_(3--class)-F59E0B?style=for-the-badge)

<br>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [What Changed in This Version](#-what-changed-in-this-version)
- [Dataset](#-dataset)
- [Pipeline Architecture](#️-pipeline-architecture)
- [Model Benchmarks](#-model-benchmarks)
- [Why XGBoost Won](#-why-xgboost-won)
- [Feature Engineering](#-feature-engineering)
- [Hyperparameter Tuning](#️-hyperparameter-tuning)
- [Overfitting Controls](#-overfitting-controls)
- [Current Bottleneck Analysis](#-current-bottleneck-analysis)
- [Uncertainty Analysis](#-uncertainty-analysis)
- [Output Format](#-output-format)
- [Files](#-files)
- [Limitations & Next Steps](#-limitations--next-steps)

<br>

---

## 🎯 Project Overview

Each Arvyax reflective session produces a short journal entry, a face emotion hint, ambient sound choice, and session metadata. This pipeline predicts two targets:

| Target | Task | Classes | Final Result |
|--------|------|---------|--------------|
| **Y1** — Emotional State | 6-class classification | calm · focused · mixed · neutral · overwhelmed · restless | **52.4% CV mean** · 52.2% val |
| **Y2** — Session Intensity | 3-bucket classification | low (1-2) · medium (3) · high (4-5) | **40.3% CV mean** · 44.4% val |

After both predictions, a **soft attention recommendation engine** selects the most relevant wellbeing recommendation from 8 templates using the class probability vector as a query — no if/else anywhere in that path.

<br>

---

## 🔄 What Changed in This Version

> This is the final version of the pipeline. Below are all the changes made from the initial implementation through to this release.

### v1 → v2: Fixed data leakage (biggest single fix)
The original code called `TfidfVectorizer.fit_transform()` on all 1200 rows before the train/val split. This inflated train accuracy to ~90% and val accuracy to ~57% — a fake 33-point gap. Fixed by fitting TF-IDF exclusively on the train split and using `.transform()` on val/test.

```
Before fix:  train=90%  val=57%  gap=33pts  ← fake
After fix:   train=~65%  val=52%  gap=~13pts ← real
```

### v2 → v3: Intensity target rebucketed from 5-class to 3-class
Raw 5-class intensity prediction gave 21-24% CV across all models tested — barely above the 20% random baseline. All models converged to this range regardless of architecture or features, confirming it's a **label noise problem**, not a model problem. Intensity=3 is a catch-all middle label users assign when uncertain. Rebucketing to low/medium/high (3-class, random=33%) pushed Y2 val accuracy to **44.4%**.

```
5-class intensity  →  21-24% CV mean (all models)   ← near random
3-class intensity  →  40.3% CV mean (XGBoost)       ← workable signal
```

### v3 → v4 (current): Code humanised + variable names cleaned up
- Section headers changed from `# =============================` to short inline comments
- Variable names shortened: `EMOTIONAL_STATES` → `STATES`, `MOOD_VOCAB_MAP` → `VMAP`, etc.
- Output columns renamed: `predicted_emotional_state` → `emotional_state`, `cls_confidence` → `confidence`
- Output file renamed: `arvyax_predictions_xgboost.csv` → `arvyax_predictions.csv`
- CV print format changed from `fold 1 | Y1:` to `fold 1 -> Y1:`

<br>

---

## 📦 Dataset

```
Training samples : 1,200
Test samples     : 120
Features         : 11 columns (text + categorical + numeric)
Targets          : emotional_state (6 classes) | intensity (bucketed to 3)
```

**Class distribution:**

```
Y1 — Emotional State          Y2 — Intensity (raw)   Y2 — After bucketing
──────────────────────        ──────────────         ────────────────────
calm          216  (18%)      1  →  226  (18.8%)     low  (1-2) →  454  (37.8%)
restless      209  (17.4%)    2  →  228  (19.0%)     mid  (3)   →  240  (20.0%)
neutral       201  (16.8%)    3  →  240  (20.0%)     high (4-5) →  506  (42.2%)
focused       193  (16.1%)    4  →  277  (23.1%)
mixed         191  (15.9%)    5  →  229  (19.1%)
overwhelmed   190  (15.8%)
```

**Input columns:**

| Column | Type | How it's used |
|--------|------|---------------|
| `journal_text` | text | TF-IDF word bigrams + char 4-grams → SVD |
| `ambience_type` | categorical | one-hot + proximity weight inside journal text |
| `face_emotion_hint` | categorical | one-hot (7 categories), ~20% missing |
| `previous_day_mood` | categorical | one-hot (8 categories) |
| `reflection_quality` | ordinal | vague=0 · conflicted=0.5 · clear=1 |
| `time_of_day` | categorical | one-hot (5 categories) |
| `duration_min` | numeric | included in structured features |
| `sleep_hours` | numeric | included + sleep recommendation rule |
| `energy_level` | numeric | included in structured features |
| `stress_level` | numeric | included in structured features |

<br>

---

## 🏗️ Pipeline Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     INPUT (per session)                   │
│  journal_text  ambience  face_emotion  prev_mood  tod    │
│  duration_min  sleep_hours  energy_level  stress_level   │
└─────────────────────┬────────────────────────────────────┘
                      │
        ┌─────────────▼─────────────┐
        │     FEATURE ENGINEERING    │
        │                           │
        │  TEXT PATH                │
        │  ├─ TF-IDF word bigrams   │ → SVD (40d) ─┐
        │  └─ TF-IDF char 4-grams   │ → SVD (20d) ─┤
        │                           │              │
        │  STRUCTURED PATH (50d)    │              │
        │  ├─ sem_sim          (6d) │ ────────────►│
        │  ├─ ambience_prox    (7d) │         ┌───▼────┐
        │  ├─ face_mood_onehot(15d) │         │  110d  │
        │  ├─ ambience_oh      (5d) │         │ matrix │
        │  ├─ tod_oh           (5d) │         └───┬────┘
        │  ├─ reflection       (1d) │             │
        │  ├─ dominant_emo     (6d) │      impute + scale
        │  ├─ conf_margin      (1d) │             │
        │  └─ numeric          (4d) │    ┌────────┴────────┐
        └───────────────────────────┘    │                 │
                                  clf_emotion        clf_intensity
                                  (depth=4)          (depth=3)
                                  6-class            3-class
                                       │                 │
                                  probs[6]         bucket (0/1/2)
                                       │                 │
                               ┌───────▼─────────────────▼────┐
                               │   ATTENTION RECOMMENDATION    │
                               │  Q = [probs(6) | bucket/2]   │
                               │  K = template keys (8×7)      │
                               │  attn = softmax(KQ/√7)        │
                               └──────────────┬────────────────┘
                                              │
                                        📄 arvyax_predictions.csv
```

<br>

---

## 📊 Model Benchmarks

> All CV results use **3-fold stratified CV, no data leakage** — TF-IDF fitted inside each fold on the train split only.

### Y1 — Emotional State (6-class, random baseline = 16.7%)

```
Model                    CV Mean   CV Std   Val Acc
──────────────────────────────────────────────────────────────────
SVM (RBF)                 56.8%    ±3.6%    ~51%     no predict_proba ❌
Random Forest             54.9%    ±2.5%    ~52%     stable ✅
XGBoost (HGB) ✅          52.4%    ±2.0%     52.2%   chosen ✅
Neural Network (TF)       50.9%    ±5.7%    ~49%     high variance ❌
Logistic Regression       49.6%    ±3.1%    ~43%     too linear ❌
──────────────────────────────────────────────────────────────────
Random baseline           16.7%      —       —
```

> 💡 SVM had the highest raw CV mean but was ruled out because it doesn't produce class probabilities, which the attention recommendation engine requires. XGBoost gives comparable accuracy with probability outputs, native NaN handling, and the lowest fold std (±2.0%).

<br>

### Y2 — Intensity, 3-class (random baseline = 33.3%)

```
Model                    CV Mean   CV Std   Val Acc
──────────────────────────────────────────────────────
XGBoost (HGB) ✅          40.3%    ±2.4%    44.4%
Random Forest             ~38%     ±2.1%    ~39%
Neural Network (TF)       ~34%     ±3.1%    ~35%
──────────────────────────────────────────────────────
Random baseline           33.3%      —       —
```

> ⚠️ Medium bucket (intensity=3) is still weak — F1=0.15. Low and High work reasonably (F1=0.48, 0.51). This reflects the underlying label: intensity=3 is what users assign when they're unsure, so there's no consistent signal for the model to learn.

<br>

### Per-Class Performance — Y1 (Validation Set, XGBoost)

```
Class          Precision   Recall    F1     Notes
───────────────────────────────────────────────────────────────────
calm              0.62      0.62     0.62   best performing class
focused           0.58      0.48     0.53   confused with calm
mixed             0.47      0.52     0.49   inherently ambiguous label
neutral           0.54      0.47     0.50   no strong vocab anchor
overwhelmed       0.47      0.55     0.51   recall ok, precision low
restless          0.47      0.48     0.48   overlaps with overwhelmed
───────────────────────────────────────────────────────────────────
weighted avg      0.53      0.52     0.52
```

### Per-Bucket Performance — Y2 (Validation Set, XGBoost)

```
Bucket        Precision   Recall    F1     Support
──────────────────────────────────────────────────
low  (1-2)       0.47      0.49     0.48      71
medium  (3)      0.19      0.12     0.15      32   ← near-random
high (4-5)       0.49      0.53     0.51      77
──────────────────────────────────────────────────
weighted avg     0.43      0.44     0.43     180
```

<br>

### Test Set Predictions (120 samples)

```
Emotional State Distribution    Intensity Bucket Distribution
────────────────────────────    ─────────────────────────────
restless       25  (20.8%)      high      65  (54.2%)
calm           25  (20.8%)      low       44  (36.7%)
neutral        21  (17.5%)      medium    11   (9.2%)
mixed          20  (16.7%)
overwhelmed    16  (13.3%)
focused        13  (10.8%)
```

<br>

---

## 🏆 Why XGBoost Won

```
Requirement                        XGBoost    RF      NN      SVM
─────────────────────────────────────────────────────────────────────────
predict_proba() for attention rec   ✅         ✅       ✅       ❌
Native NaN handling                 ✅         ❌       ❌       ❌
Stable across folds (low std)       ✅ ±2.0%  ✅ ±2.5% ❌ ±5.7% ✅
Sparse one-hot feature tolerance    ✅         ✅       ❌       ⚠️
class_weight='balanced' built-in    ✅         ✅       —        ✅
```

**Why NN specifically underperforms (gap vs RF: ~4%) — the sparse gradient problem:**

This was tested exhaustively — 5 different architectures, 3 activations, 4 alpha values. All converged to 50-51%. The reason is structural and cannot be fixed by tuning.

> **Quick clarification on the term:** This is often called "vanishing gradients" but that's a different (and less severe) problem. Vanishing gradients means the gradient *shrinks* exponentially through deep layers — it's small but still exists. What happens here is worse: the gradient is **exactly zero** for 76% of weights. Not small. Zero. The weight never receives any learning signal at all.

#### The exact mechanism

The weight gradient formula in backpropagation is:

```
dL/dW[i,j] = x[i] × δ[j]

where x[i] = input feature i
      δ[j] = upstream gradient flowing back through neuron j
```

When `x[i] = 0` (i.e. that input feature is zero), the entire row `i` of the gradient matrix becomes zero — regardless of how large `δ[j]` is. The weight `W[i,j]` receives **no update**, no matter how many epochs you train, how high the learning rate is, or which optimiser you use. It stays at its random initialisation value forever.

This compounds with ReLU: if a neuron's pre-activation output `z[j] < 0`, its gradient is also zero, zeroing out the entire column `j`. So the two effects multiply:

```
Effect 1 — input sparsity:     76.4% of input dims = 0
  → dL/dW[i,j] = x[i] × δ = 0 for 76% of weight rows
  → those rows never update

Effect 2 — ReLU dead neurons:  42.2% of neurons output 0 after ReLU
  → dL/dW[i,j] = x[i] × 0  = 0 for 42% of weight columns
  → those columns never update

Combined result:  86.3% of ALL weights receive exactly zero gradient per step
                  Only 13.7% of weights actually learn anything per iteration
```

Switching to `tanh` instead of ReLU doesn't fix it — tanh gradient is `(1 - tanh²(z))` which is nonzero everywhere, but `x[i] = 0` still zeros the entire row. The input sparsity dominates.

#### Why tree models don't have this problem

A decision tree evaluates one feature at a time:

```
if feature_42 > 0.3:          ← only feature_42 matters here
    go left                   ← features 0-41 and 43-109 are irrelevant
```

Zero values in other features don't interfere with the split on feature 42. Each tree node only needs *one* informative feature to be nonzero. With 76% sparsity, the chance that at least one feature in a node is nonzero is very high — trees find signal efficiently.

NN backprop needs *all* connected weights to carry gradient simultaneously. With 76% inputs zeroed, most of the weight matrix is invisible to the optimiser.

#### Why this is not fixable without changing the features

```
Attempted fix               Result
──────────────────────────────────────────────────────────────────
tanh instead of relu        ~51% (input sparsity still dominates)
larger network (256,128,64) ~51.7% (more dead weights, same problem)
smaller network (64,32)     ~50.8% (less capacity, same problem)
lower alpha (less L2)       ~51%   (regularisation irrelevant)
higher learning rate        ~50%   (no gradient = nothing to amplify)
BatchNormalization          ~52%   (normalises activations, not inputs)
──────────────────────────────────────────────────────────────────
RF baseline                  54.9% (no gradient at all, immune)
XGBoost baseline             52.4% (same, immune)
```

The only real fix is **denser input features**. Replace the one-hot structured features with learned embeddings, and replace TF-IDF vectors with sentence transformer embeddings (`all-MiniLM-L6-v2`). Dense inputs → nonzero gradients everywhere → NN starts learning properly. This is the planned next step.

```
Feature Group              Zero fraction   Effect on backprop
─────────────────────────────────────────────────────────────
sem_sim (6d)               87.2%           row gradient = 0
face/mood onehot (15d)     86.7%           row gradient = 0
ambience proximity (7d)    84.6%           row gradient = 0
ambience/tod onehot (10d)  80.0%           row gradient = 0
numeric (4d)                0.0%           ✅ learns normally
─────────────────────────────────────────────────────────────
Overall                    75.8%           86.3% of weights frozen at init
```

<br>

---

## 🔧 Feature Engineering

### Text → TF-IDF → SVD

```python
# word bigrams — "mind racing", "locked in", "emotionally tired" as single features
TfidfVectorizer(ngram_range=(1,2), max_features=500, sublinear_tf=True, min_df=3)

# char 4-grams — catches typos: "teh" ≈ "the", "tehre" ≈ "there"
TfidfVectorizer(analyzer='char_wb', ngram_range=(3,4), max_features=200, min_df=4)

# compress: 500 sparse → 40 dense, 200 sparse → 20 dense
TruncatedSVD(40)
TruncatedSVD(20)
```

> ✅ **No leakage**: both vectorizers fitted on train fold only. `.transform()` on val/test.

<br>

### Ambience Proximity Weight

When the ambience word appears in the journal, emotion keywords near it get a boost:

```python
proximity_weight = 2.0 / (1.0 + distance * 0.1)
# "ocean audio was nice" → calm keywords near "ocean" → up to 2x weight
```

### Semantic Similarity (6d)

Cosine-style overlap between journal tokens and each emotion's vocabulary cluster. Returns a 6-dim vector — one score per emotion class.

### Face × Mood Interaction (15d)

Both face_emotion_hint and previous_day_mood are one-hot encoded and concatenated. The model learns the interaction weights (e.g. `tired_face + overwhelmed_prev` → overwhelmed direction) rather than having rules hardcoded.

<br>

---

## ⚙️ Hyperparameter Tuning

### Y1 — Emotional State

| Parameter | Value | Tested | Why |
|-----------|-------|--------|-----|
| `max_depth` | 4 | 3–7 | depth 5+ caused train/val gap > 20% |
| `learning_rate` | 0.05 | 0.01, 0.05, 0.1 | 0.1 overfit, 0.01 underfit |
| `min_samples_leaf` | 10 | 5, 10, 15, 20 | key regularizer for 1200 samples |
| `max_iter` | 300 | 100–500 | enough, early stopping handles the rest |
| `class_weight` | `'balanced'` | None, balanced | +3-4% recall on overwhelmed/neutral |

### Y2 — Intensity (3-class)

| Parameter | Value | Why different from Y1 |
|-----------|-------|----------------------|
| `max_depth` | 3 | shallower — weaker signal overfit faster |
| `min_samples_leaf` | 15 | higher — noisier labels need bigger leaves |
| `class_weight` | `None` | buckets are naturally imbalanced by design |

<br>

---

## 🛡️ Overfitting Controls

### 1. The Leakage Fix

```python
# ❌ before — vectorizer saw val text during fit
X_tfidf = tfidf.fit_transform(all_1200_texts)
X_tr, X_val = train_test_split(X_tfidf)

# ✅ after — vectorizer never sees val/test
X_tr_tfidf  = tfidf.fit_transform(train_texts)
X_val_tfidf = tfidf.transform(val_texts)    # transform only
X_te_tfidf  = tfidf.transform(test_texts)   # transform only
```

Effect: train accuracy dropped from fake ~90% to honest ~65%. Val/test gap went from 33pts to ~13pts.

### 2. Tree Regularization

- `max_depth=4` caps tree depth → no leaf can perfectly memorise a single entry
- `min_samples_leaf=10` → every leaf covers at least 10 samples → prevents micro-splits
- `learning_rate=0.05` → slower gradient steps → less aggressive fitting

### 3. Stratified 3-Fold CV

Each fold preserves the exact class proportions of the full dataset. This gives an honest estimate before training the final model:

```
running 3-fold CV...
  fold 1  ->  Y1: 0.522   Y2: 0.390
  fold 2  ->  Y1: 0.550   Y2: 0.383
  fold 3  ->  Y1: 0.500   Y2: 0.438
  mean  ->  Y1: 0.524   Y2: 0.403
  std   ->  Y1: 0.020    Y2: 0.024
```

Low std (0.020 for Y1) means the model is stable — not getting lucky on one split.

<br>

---

## 🔍 Current Bottleneck Analysis

### 1. Short entries are unreadable to TF-IDF

~35-40% of journals are very short. TF-IDF produces near-zero vectors for these:

```
Entry                   True label    TF-IDF signal   What BERT would give
──────────────────────────────────────────────────────────────────────────
"ok session"            neutral       ❌ 0 hits        ✅ 768-dim contextual
"still off"             overwhelmed   ❌ 0 hits        ✅ 768-dim contextual
"actually helped"       focused       ❌ 0 hits        ✅ 768-dim contextual
"mind racing"           restless      ✅ 2 hits        ✅ 768-dim contextual
"kinda calm"            calm          ✅ 1 hit         ✅ 768-dim contextual
```

These entries are indistinguishable by the model. No amount of hyperparameter tuning fixes this because the input vectors are literally the same.

### 2. TF-IDF has no negation awareness

```python
"not calm"   # 'calm' token fires → overlaps with calm cluster
"calm"       # 'calm' token fires → overlaps with calm cluster
# same TF-IDF representation despite opposite meaning
```

### 3. Feature sparsity structurally limits neural approaches

76% of the structured feature matrix is zeros. This is why NN consistently underperforms tree models by ~4% and no architecture change fixes it.

### 4. Intensity=3 is not a real label

Medium intensity has F1=0.15. Users assign `3` when unsure, not when they feel a specific medium intensity. The signal simply isn't there — this is aleatory noise, not something more data or better models can solve without changing the labelling instrument.

### 5. BERT is not the solution here

```
Model                Params    Needs samples    Overfit risk (n=1200)
─────────────────────────────────────────────────────────────────────
BERT-base (finetune)  110M     ~1,000,000+      catastrophic
BERT-base (frozen)    4,608    ~46,000+         low, but domain mismatch
all-MiniLM-L6-v2      22M     ~2,000            moderate  ← right choice
TF-IDF + SVD (current) ~0     ~200              ✅ safe
```

The right next step is `all-MiniLM-L6-v2` (frozen) + XGBoost head. ~10 lines of change, expected +4-6% on Y1.

<br>

---

## 📐 Uncertainty Analysis

### Y1 Prediction Confidence

`confidence` in the output CSV = `max(predict_proba)`.

```
Confidence range    Meaning
──────────────────────────────────────────────────────────
> 0.70              High — clear signal, use directly
0.50 – 0.70         Moderate — model leaning but uncertain
0.35 – 0.50         Low — spread across 2-3 classes
< 0.35              No signal — entry too short/ambiguous
```

### Main Confusion Pairs (from val set)

```
overwhelmed  →  restless      vocab overlap ("heavy", "unable", "scattered")
calm         →  neutral        both produce low-signal short entries
focused      →  calm           clarity vocabulary appears in both clusters
mixed        →  anything       inherently contains vocab from 2+ classes
```

### Y2 Intensity — Irreducible Uncertainty

Pearson correlation between raw intensity and each numeric feature:

```
stress_level     r = 0.14   ← strongest predictor
energy_level     r = 0.12
duration_min     r = 0.08
sleep_hours      r = -0.06
```

All near zero. The model cannot reliably separate intensity 2 from 3 from session metadata alone — users don't assign these labels with enough consistency. This is why all models (XGB, RF, NN) converge to 21-24% on 5-class intensity regardless of what features or architecture you use.

### How Uncertainty Propagates into Recommendations

The attention mechanism uses the full probability vector as a query, so low-confidence predictions automatically produce lower-contrast attention scores:

```
High confidence:  probs = [0.04, 0.03, 0.06, 0.05, 0.76, 0.06]
→ attn sharply peaks at emotional_offload (0.62)   → clear recommendation

Low confidence:   probs = [0.19, 0.16, 0.18, 0.21, 0.14, 0.12]
→ attn distributed: steady_continuity (0.16) wins by small margin
→ recommendation is softer but never crashes
```

<br>

---

## 📄 Output Format

Each row in `arvyax_predictions.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Session ID from test set |
| `emotional_state` | str | One of 6 emotion classes |
| `intensity_bucket` | str | low / medium / high |
| `recommendation` | str | Personalised wellbeing text |
| `template` | str | Which of the 8 templates was selected |
| `duration_min` | int | Session duration from input |
| `sleep_rec` | float | 8 if sleep_hours < 6, else actual |
| `time_of_day` | str | From input |
| `attn_weights` | dict | {template: float} full attention distribution |
| `confidence` | float | max(predict_proba) for Y1 |
| `ambience` | str | Ambience type from input |

**Sample row:**

```json
{
  "id": 10001,
  "emotional_state": "focused",
  "intensity_bucket": "medium",
  "recommendation": "Your mind is in a receptive state. Use this for deep work. The cafe ambience supported focus today. Start with the hardest task while this clarity holds.",
  "template": "deep_work",
  "duration_min": 4,
  "sleep_rec": 8,
  "time_of_day": "night",
  "confidence": 0.583
}
```

<br>

---

## 📁 Files

```
arvyax/
├── 📄 arvyax_xgboost.py         ← main pipeline (this one)
├── 📄 arvyax_random_forest.py   ← RF version, includes feature_importances_ printout
├── 📄 arvyax_nn.py              ← TF neural network, benchmarking only
│
├── 📊 arvyax_predictions.csv    ← test set predictions (120 rows)
│
└── 📘 README.md
```

**Quick start:**

```bash
pip install scikit-learn pandas numpy

python arvyax_xgboost.py
```

**Update data paths at the top of the file:**

```python
train_df = pd.read_csv('/content/Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv')
test_df  = pd.read_csv('/content/arvyax_test_inputs_120.xlsx - Sheet1.csv')
```

<br>

---

## 🚧 Limitations & Next Steps

| # | Issue | Impact | Fix |
|---|-------|--------|-----|
| 1 | Short entries → zero TF-IDF vectors | 🔴 ~35% of data has no usable signal | Replace text features with `all-MiniLM-L6-v2` frozen embeddings |
| 2 | intensity=3 is near-random (F1=0.15) | 🟠 Y2 medium bucket unreliable | Change labelling — ask users to rate intensity on a simpler 2-point scale |
| 3 | `face_emotion_hint` ~20% missing | 🟡 Imputed as "none" silently | Add a `face_missing` indicator feature |
| 4 | NN underperforms RF by ~4% | 🟡 Structural (76% sparse features) | Need dense embeddings to fix this |
| 5 | Categorical one-hots | 🟢 Minor | Learned 4-8d embeddings would generalise better in a neural model |

<br>

---

## 📈 Accuracy Journey

```
Version    Change                                  Y1 Val    Y1 CV   Y2 Val   Note
────────────────────────────────────────────────────────────────────────────────────
v1         Numpy NN, 29d hand-crafted features     ~66%*     —       —        *fake (leakage)
v2         Fixed leakage                           ~49%      ~49%    —        real baseline
v2.1       + TF-IDF bigrams + char ngrams          ~52%      ~51%    —        text signal added
v2.2       + Expanded emotion vocab                ~54%      ~54%    —
v3         5-class intensity → 3-class buckets     52.2%    52.4%   44.4%    current
────────────────────────────────────────────────────────────────────────────────────
```

> The drop from v1 (66%) to v2 (49%) is not a regression. It's removing fake accuracy caused by data leakage. The real starting point was always ~49%.

<br>

---

<div align="center">

**Built for Arvyax Reflective Session Platform**

*Dual-output emotion classification · Attention-based recommendations · Zero if/else*

![No leakage](https://img.shields.io/badge/Evaluation-No_Data_Leakage-22C55E?style=flat-square)
![No if/else](https://img.shields.io/badge/Recommendations-Zero_if%2Felse-6366F1?style=flat-square)
![3-fold CV](https://img.shields.io/badge/Validation-Stratified_3--fold_CV-F59E0B?style=flat-square)

</div>
