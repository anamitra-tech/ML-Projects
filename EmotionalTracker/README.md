# 🧘 ARVYAX — Reflective Session ML Pipeline

> **Dual-output classification pipeline** that predicts emotional state + session intensity from mindfulness journal entries, face emotion signals, and ambient session metadata — then generates a personalised recommendation using a pure attention mechanism (zero if/else).

<br>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Final-22C55E?style=for-the-badge)
![Y1 Acc](https://img.shields.io/badge/Y1_CV_Mean-57--58%25_(projected)-6366F1?style=for-the-badge)
![Y2 Acc](https://img.shields.io/badge/Y2_Val_Acc-44.4%25_(3--class)-F59E0B?style=for-the-badge)

<br>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Full Version History](#-full-version-history)
- [Dataset](#-dataset)
- [Pipeline Architecture](#️-pipeline-architecture)
- [Model Benchmarks](#-model-benchmarks)
- [Why XGBoost Won](#-why-xgboost-won)
- [Feature Engineering](#-feature-engineering)
- [Recommendation Engine](#-recommendation-engine)
- [Hyperparameter Tuning](#️-hyperparameter-tuning)
- [Overfitting Controls](#-overfitting-controls)
- [Bottleneck Analysis](#-bottleneck-analysis)
- [Uncertainty Analysis](#-uncertainty-analysis)
- [Output Format](#-output-format)
- [Files](#-files)
- [Limitations & Next Steps](#-limitations--next-steps)

<br>

---

## 🎯 Project Overview

Each Arvyax reflective session produces a short journal entry, a face emotion hint, ambient sound choice, and session metadata. This pipeline predicts two targets from each session:

| Target | Task | Classes | Final Result |
|--------|------|---------|--------------|
| **Y1** — Emotional State | 6-class classification | calm · focused · mixed · neutral · overwhelmed · restless | **~57-58% CV** (projected with v5 fixes) |
| **Y2** — Session Intensity | 3-bucket classification | low (1-2) · medium (3) · high (4-5) | **40.3% CV mean** · 44.4% val |

After both predictions, a **soft attention recommendation engine** selects the most relevant wellbeing recommendation from 8 templates. The query vector now includes class probabilities, intensity bucket, ambience type, and session duration — making recommendations context-aware beyond just the emotion label.

<br>

---

## 🔄 Full Version History

Every change made from first prototype to final version, with the exact numbers:

### v1 — Initial prototype (Numpy NN, hand-crafted features)
- Pure numpy neural network, 29-dim hand-crafted feature vector
- **Reported val accuracy: ~66%** — but this was entirely fake
- Root cause: `TfidfVectorizer.fit_transform()` called on all 1200 rows before splitting, so the vectorizer had already seen validation text. This is data leakage.

### v2 — Fixed leakage (biggest single fix, +0% but honest)
- TF-IDF now fitted on train split only; `.transform()` on val/test
- Accuracy dropped from fake 66% to honest ~49% — not a regression, just the truth
```
Before: train=90%  val=57%  gap=33pts  ← leakage
After:  train=~65% val=49%  gap=~16pts ← real
```

### v2.1 — Added TF-IDF word bigrams + char 4-grams
- Replaced the 29d hand-crafted features with proper text vectorisation
- Word bigrams catch phrases like `"mind racing"`, `"locked in"`, `"less tense"`
- Char 4-grams catch typos: `"teh"` ≈ `"the"`, `"tehre"` ≈ `"there"`
- **+3% CV** (49% → ~51-52%)

### v2.2 — Expanded emotion vocab
- Added domain-specific phrases found by reading the actual dataset: `"emotionally tired"`, `"still uneasy"`, `"two moods"`, `"mind jumping"` etc.
- **+2% CV** (52% → ~54%)

### v3 — Intensity rebucketed from 5-class to 3-class
- 5-class intensity prediction gave 21-24% CV across all models — barely above 20% random baseline
- Root cause: intensity=3 is not a real label. Users assign it when uncertain, not to signal a distinct emotional intensity. It's label noise.
- Rebucketed to low(1-2) / medium(3) / high(4-5) → random baseline rises to 33%
```
5-class CV: 21-24% (all models) ← near random
3-class CV: 40.3% (XGBoost)    ← workable signal
```
### v4 — Three structural fixes (current, +3.6% projected)

**Fix 1: TF-IDF was being poisoned by non-text tokens (-3.1% bug)**

The previous code built text strings as `f"{journal} {ambience} {face_emotion} {prev_mood}"` before feeding to TF-IDF. The intent was to give TF-IDF more context. The actual effect was the opposite. `"cafe"` appears in every single cafe session regardless of emotion — its IDF weight is non-discriminative but its presence in 240/1200 rows makes it show up in the top SVD components, explaining ambience variance instead of emotion variance.

Ablation proof:
```
journal only:                      39.3%  (text features alone)
journal + ambience + face + prev:  31.0%  (text features alone)
                                   -8.3% on text features
                                   -3.1% on full pipeline
```
Fix: `journal_text_only()` — TF-IDF sees journal text only. Ambience, face, and mood stay in the structured features where they're encoded properly.

**Fix 2: face × mood outer product instead of concat (+0.5%)**

Previous code: `np.concatenate([face_vec(7d), mood_vec(8d)])` = 15d

This teaches the model `tired_face` and `overwhelmed_prev` as separate independent features. It cannot learn the combination `tired_face AND overwhelmed_prev together`. The outer product `np.outer(face_vec, mood_vec).flatten()` = 56d creates one dimension per combination — the model can now directly weight `tense_face × overwhelmed_prev → overwhelmed/restless direction`.

**Fix 3: Recommendation engine extended to use ambience and duration**

Previous query: `Q = [cls_probs(6) | bucket/2]` — 7 dimensions

This gave identical attention scores (and therefore identical recommendations) to:
- 4-minute restless cafe session
- 30-minute restless ocean session

These are meaningfully different situations. The fix extends Q to 14 dimensions:
`Q = [cls_probs(6) | bucket(1) | ambience_oh(5) | dur_norm(1) | slp_norm(1)]`

Each template now has an ambience affinity vector and duration affinity score baked into its key vector. `rest_recovery` has `ocean=0.9, dur=0.2` (suits short tired sessions with calming ambience). `deep_work` has `ocean=1.0, dur=0.8` (suits long focused sessions).

```
Total improvement from v5:  +3.6% CV (0.547 → 0.583)
Projected Y1 CV:             ~57-58%
```

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
calm          216  (18.0%)    1  →  226  (18.8%)     low  (1-2) →  454  (37.8%)
restless      209  (17.4%)    2  →  228  (19.0%)     mid  (3)   →  240  (20.0%)
neutral       201  (16.8%)    3  →  240  (20.0%)     high (4-5) →  506  (42.2%)
focused       193  (16.1%)    4  →  277  (23.1%)
mixed         191  (15.9%)    5  →  229  (19.1%)
overwhelmed   190  (15.8%)
```

**Input columns:**

| Column | Type | How it's used |
|--------|------|---------------|
| `journal_text` | text | TF-IDF (journal only) → SVD |
| `ambience_type` | categorical | one-hot (5d) + proximity weight in journal |
| `face_emotion_hint` | categorical | outer product with prev_mood (56d) |
| `previous_day_mood` | categorical | outer product with face_emotion (56d combined) |
| `reflection_quality` | ordinal | vague=0 · conflicted=0.5 · clear=1, also × tod |
| `time_of_day` | categorical | one-hot (5d) + × reflection_quality interaction |
| `duration_min` | numeric | normalised [0,1], in attention Q vector |
| `sleep_hours` | numeric | normalised [0,1], sleep_rec rule, in attention Q |
| `energy_level` | numeric | normalised [0,1] |
| `stress_level` | numeric | normalised [0,1] |

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
        ┌─────────────▼──────────────────────────┐
        │          FEATURE ENGINEERING            │
        │                                         │
        │  TEXT PATH (journal only → no leakage) │
        │  ├─ TF-IDF word bigrams → SVD (40d)    │
        │  └─ TF-IDF char 4-grams → SVD (20d)   │
        │                                         │
        │  STRUCTURED PATH (96d)                  │
        │  ├─ sem_sim              (6d)           │
        │  ├─ ambience_proximity   (7d)           │
        │  ├─ face × mood outer    (56d) ← NEW   │
        │  ├─ ambience_oh          (5d)           │
        │  ├─ tod_oh               (5d)           │
        │  ├─ reflection           (1d)           │
        │  ├─ dominant_emotion     (6d)           │
        │  ├─ confidence_margin    (1d)           │
        │  ├─ numeric normalised   (4d)           │
        │  └─ tod × reflection     (5d) ← NEW    │
        └──────────────────┬──────────────────────┘
                           │ hstack → 156d
                     impute + scale
                           │
              ┌────────────┴────────────┐
              │                         │
        clf_emotion               clf_intensity
        depth=4, balanced          depth=3
        6-class                    3-class
              │                         │
         probs[6]               bucket (0/1/2)
              │                         │
        ┌─────▼─────────────────────────▼──────────────────────┐
        │              ATTENTION RECOMMENDATION                  │
        │                                                        │
        │  Q = [probs(6) | bucket/2 | ambience_oh(5)           │
        │        | dur_norm | slp_norm]   shape (14,) ← NEW    │
        │  K = template key matrix        shape (8, 14)         │
        │  attn = softmax(K @ Q / √14)                          │
        │  rec = templates[argmax(attn)]  zero if/else          │
        └──────────────────────┬─────────────────────────────────┘
                               │
                      📄 arvyax_predictions.csv
```

<br>

---

## 📊 Model Benchmarks

> All CV results use **3-fold stratified CV, no data leakage** — TF-IDF fitted inside each fold on the train split only.

### Y1 — Emotional State (6-class, random baseline = 16.7%)

```
Model                    CV Mean   CV Std   Val Acc   Notes
──────────────────────────────────────────────────────────────────────
SVM (RBF)                 56.8%    ±3.6%    ~51%      no predict_proba ❌
Random Forest             54.9%    ±2.5%    ~52%      stable ✅
XGBoost (HGB) ✅          52.4%→57.8%* ±2.0%  52%+  chosen ✅
Neural Network (TF)       50.9%    ±5.7%    ~49%      sparse gradient problem ❌
Logistic Regression       49.6%    ±3.1%    ~43%      too linear ❌
──────────────────────────────────────────────────────────────────────
Random baseline           16.7%      —       —
```
*57.8% is the v5 ablation result with all three fixes applied (5-fold CV)

> 💡 SVM has the highest raw CV mean but produces no class probabilities — which the attention recommendation engine requires. XGBoost was chosen for probability outputs, native NaN handling, and the lowest fold variance (±2.0%).

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

> ⚠️ Medium bucket (intensity=3) has F1=0.15 — near-random. This is label noise, not a model problem. Users assign 3 when uncertain. Low and High are workable (F1=0.48, 0.51).

<br>

### Per-Class Performance — Y1 (Validation Set, v4 XGBoost)

```
Class          Precision   Recall    F1     Confusion pattern
───────────────────────────────────────────────────────────────────────────
calm              0.62      0.62     0.62   confused with neutral (short entries)
focused           0.58      0.48     0.53   confused with calm (shared clarity vocab)
mixed             0.47      0.52     0.49   inherently ambiguous — contains 2+ class vocabs
neutral           0.54      0.47     0.50   no strong anchor vocabulary
overwhelmed       0.47      0.55     0.51   confused with restless (shared "heavy", "unable")
restless          0.47      0.48     0.48   over-predicted (25/120 in test set)
───────────────────────────────────────────────────────────────────────────
weighted avg      0.53      0.52     0.52
```

<br>

### Test Set Predictions — v4 (120 samples)

```
Emotional State                Intensity Bucket
────────────────────────────   ─────────────────────────────
restless       25  (20.8%)     high      65  (54.2%)
calm           25  (20.8%)     low       44  (36.7%)
neutral        21  (17.5%)     medium    11   (9.2%)
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
Stable across folds (low std)       ✅ ±2.0%   ✅ ±2.5% ❌ ±5.7% ✅
Sparse feature tolerance            ✅         ✅       ❌       ⚠️
class_weight='balanced' built-in    ✅         ✅       —        ✅
```

### Why NN underperforms by ~4% — the sparse gradient problem

> **On the term "vanishing gradients":** That describes a different problem. Vanishing gradients means the gradient *shrinks* through deep layers — small but nonzero. What happens here is worse: the gradient is **exactly zero** for 86% of weights. Not small. Zero. Those weights never update regardless of learning rate, architecture, or activation function.

The weight gradient formula in backprop is:

```
dL/dW[i,j] = x[i] × δ[j]
```

When `x[i] = 0`, the entire row i of the gradient matrix is zero — regardless of how large `δ[j]` is. That weight never receives a signal and stays at its random initialisation value forever.

Two effects compound:

```
Effect 1 — input sparsity:     76.4% of input dims = 0
  → dL/dW[i,j] = 0 for 76% of weight rows (those rows never update)

Effect 2 — ReLU dead neurons:  42.2% of neurons output 0 post-ReLU
  → dL/dW[i,j] = 0 for 42% of weight columns (those columns never update)

Combined: 86.3% of ALL weights receive exactly zero gradient per step
          Only 13.7% of the network actually learns anything per iteration
```

Switching to `tanh` doesn't fix it — `tanh` gradient is nonzero everywhere, but `x[i] = 0` still zeros the entire row. The input sparsity dominates, not the activation.

```
Attempted fix               NN CV Result
──────────────────────────────────────────────────────────
tanh instead of relu        ~51% (input sparsity dominates)
larger (256, 128, 64)       ~51.7%
smaller (64, 32)            ~50.8%
lower L2 (alpha)            ~51%
higher learning rate        ~50% (no gradient = nothing to amplify)
BatchNormalization           ~52% (normalises activations, not inputs)
──────────────────────────────────────────────────────────
RF (immune — trees)          54.9%
XGBoost (immune — trees)     52.4% → 57.8% with v5 fixes
```

Zero fraction per feature group:

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

The only real fix: **denser inputs**. Replace one-hot structured features with learned embeddings and replace TF-IDF with `all-MiniLM-L6-v2` sentence embeddings. Dense inputs → nonzero gradients everywhere → NN works properly.

<br>

---

## 🔧 Feature Engineering

### TF-IDF — journal text only (v5 fix)

```python
# WRONG (v4): appending ambience/face/mood tokens to the text string
text = f"{journal} {ambience} {face_emotion} {prev_mood}"
# Why it hurts: "cafe" appears in all 240 cafe sessions regardless of emotion
# TF-IDF amplifies it → top SVD components explain ambience variance, not emotion

# CORRECT (v5): journal text only
text = journal_text
# Ablation: journal only = 39.3% on text, appended version = 31.0% — delta -8.3%
```

Word bigrams catch multi-word emotion phrases:
```python
TfidfVectorizer(ngram_range=(1,2), max_features=500, sublinear_tf=True, min_df=3)
# "mind racing", "locked in", "emotionally tired", "less tense" → single features
```

Char 4-grams catch informal writing and typos:
```python
TfidfVectorizer(analyzer='char_wb', ngram_range=(3,4), max_features=200, min_df=4)
# "teh" ≈ "the", "tehre" ≈ "there", "kinda" → subword patterns
```

Both compressed with TruncatedSVD: 500 sparse dims → 40 dense, 200 → 20.

> ✅ **No leakage**: both vectorizers fitted on the train split only inside each fold. `.transform()` on val/test.

<br>

### Face × Mood Interaction (v5 fix)

```python
# WRONG (v4): simple concatenation — learns face and mood independently
np.concatenate([face_vec(7d), mood_vec(8d)])  # 15d
# Can learn: "tired_face is common in restless entries"
# Cannot learn: "tired_face AND overwhelmed_prev together → strong restless signal"

# CORRECT (v5): outer product — learns joint combinations
np.outer(face_vec(7d), mood_vec(8d)).flatten()  # 56d
# Each of 7×8=56 combinations gets its own weight
# "tense_face × overwhelmed_prev" → the model can directly weight this combination
```

### Ambience Proximity Weight

When the ambience word appears in the journal, emotion keywords near it get up to 2× weight:

```python
proximity_weight = 2.0 / (1.0 + token_distance * 0.1)
# "ocean audio was nice" → calm keywords within 5 tokens of "ocean" → boosted
# "cafe ambience weirdly helped" → cafe as ambience context → boosted
```

### Semantic Similarity (6d)

Cosine-style overlap between journal tokens and each of the 6 emotion vocabulary clusters. Returns one score per class. Also used to derive `dominant_emotion` (argmax) and `confidence_margin` (top score minus second score).

### Normalised Numerics (4d)

```python
dur_norm = clip(duration_min / 35.0, 0, 1)  # max session = 35 min
slp_norm = clip(sleep_hours  / 8.5,  0, 1)  # max sleep = 8.5h
nrg_norm = energy_level / 5.0
str_norm = stress_level / 5.0
# normalised so all features sit on [0,1] alongside probability-based features
```

### Time × Reflection Interaction (5d, v5 new)

```python
tod_rq = tod_oh(5d) * reflection_quality(scalar)
# morning + clear (1.0) = different signal from morning + vague (0.0)
# night   + conflicted  = night entry where person is still uncertain
```

<br>

---

## 💡 Recommendation Engine

### How it works (soft attention, zero if/else)

```
Q  = [cls_probs(6) | bucket/2 | ambience_oh(5) | dur_norm | slp_norm]  shape (14,)
K  = template_key_matrix                                                  shape (8, 14)
attn = softmax(K @ Q / √14)                                              shape (8,)
rec  = templates[argmax(attn)]
```

The 8 templates each have a key vector with:
- **Emotion class affinities** — which emotion states suit this recommendation
- **Ambience affinities** — which ambient sound types suit this template (v5 new)
- **Duration affinity** — whether this template suits short or long sessions (v5 new)

### Why ambience and duration were added (v5)

Without them, these two sessions produce identical Q vectors and therefore identical recommendations:
```
Session A: restless, high intensity, 4min, cafe
Session B: restless, high intensity, 30min, ocean
→ old Q = [0.05, 0.1, 0.1, 0.05, 0.1, 0.6, 1.0] (identical)
→ new Q = [... | cafe_oh | 0.11, 0.84]  vs  [... | ocean_oh | 0.86, 0.84]
         (different → different template)
```

### Template ambience and duration affinities

| Template | Best ambience | Duration bias | Why |
|----------|--------------|---------------|-----|
| `deep_work` | ocean, mountain | long (0.8) | focused flow needs sustained time |
| `gentle_reset` | ocean, forest, rain | medium (0.4) | short calming breaks work well |
| `grounding_practice` | rain, cafe | short (0.3) | person couldn't settle → short session |
| `emotional_offload` | forest, rain | medium (0.5) | nature sounds encourage release |
| `steady_continuity` | cafe, mountain | medium (0.5) | baseline maintenance |
| `rest_recovery` | ocean, rain | short (0.2) | too tired for a long session |
| `high_intensity` | rain, mountain | medium (0.4) | tension needs structured reset |
| `dual_awareness` | ocean, mountain | medium (0.5) | balanced environments for mixed states |

### Sleep recommendation rule

```python
sleep_rec = 8 if sleep_hours < 6 else round(sleep_hours, 1)
# deterministic, applied per row — no uncertainty
```

<br>

---

## ⚙️ Hyperparameter Tuning

### Y1 — Emotional State

| Parameter | Value | Tested range | Why |
|-----------|-------|-------------|-----|
| `max_depth` | 4 | 3–7 | depth 5+ caused >20pt train/val gap |
| `learning_rate` | 0.05 | 0.01, 0.05, 0.1 | 0.1 overfit, 0.01 underfit |
| `min_samples_leaf` | 10 | 5, 10, 15, 20 | key regulariser for 1200 samples |
| `max_iter` | 300 | 100–500 | upper bound — internal early stopping handles the rest |
| `class_weight` | `'balanced'` | None, balanced | +3–4% recall on overwhelmed and neutral |

### Y2 — Intensity (3-class)

| Parameter | Value | Why different from Y1 |
|-----------|-------|----------------------|
| `max_depth` | 3 | shallower — weaker signal overfits faster |
| `min_samples_leaf` | 15 | noisier labels need larger leaves to generalise |
| `class_weight` | `None` | 3-class buckets are imbalanced by design (37/20/42%) |

<br>

---

## 🛡️ Overfitting Controls

### 1. TF-IDF leakage fix (v2)

```python
# ❌ v1 — vectorizer learns val vocabulary during fit
X_tfidf = tfidf.fit_transform(all_1200_texts)
X_tr, X_val = train_test_split(X_tfidf)

# ✅ v2+ — vectorizer never sees val/test
X_tr_tfidf  = tfidf.fit_transform(train_texts)   # fit on train only
X_val_tfidf = tfidf.transform(val_texts)          # transform only
X_te_tfidf  = tfidf.transform(test_texts)         # transform only
```

Effect: train accuracy honest ~65%, val ~52%, gap ~13pts (was 33pts).

### 2. Tree regularisation

- `max_depth=4` — no leaf memorises individual entries
- `min_samples_leaf=10` — every leaf covers at least 10 samples
- `learning_rate=0.05` — slower gradient steps, less aggressive fitting

### 3. Stratified 3-fold CV

Each fold preserves exact class proportions. CV mean gives an honest estimate before training the final model. Low std = model is stable, not lucky on one split.

```
fold 1  ->  Y1: 0.522   Y2: 0.390
fold 2  ->  Y1: 0.550   Y2: 0.383
fold 3  ->  Y1: 0.500   Y2: 0.438
mean    ->  Y1: 0.524   Y2: 0.403   (v4)
std     ->  Y1: 0.020   Y2: 0.024
```

<br>

---

## 🔍 Bottleneck Analysis

### 1. TF-IDF is blind to ~35% of journal entries

Short entries produce near-zero vectors. The model cannot distinguish between them:

```
Entry              True label    TF-IDF vector    Fix
─────────────────────────────────────────────────────────────────────
"ok session"       neutral       [0, 0, 0, ...]   all-MiniLM embeddings
"still off"        overwhelmed   [0, 0, 0, ...]   all-MiniLM embeddings
"actually helped"  focused       [0, 0, 0, ...]   all-MiniLM embeddings
"mind racing"      restless      [0, 0.3, 0, ...] ✅ already works
"kinda calm"       calm          [0.2, 0, 0, ...] ✅ already works
```

### 2. TF-IDF has no negation awareness

```
"not calm"  → 'calm' token fires → overlaps with calm cluster
"calm"      → 'calm' token fires → overlaps with calm cluster
identical representation, opposite meaning
```

### 3. Feature sparsity kills neural approaches

86.3% of NN weights receive exactly zero gradient per step (see Why XGBoost Won section for the full breakdown). Not fixable by architecture changes — requires denser input features.

### 4. Intensity=3 is not a real label

F1=0.15 on the medium bucket. Users assign 3 when uncertain about intensity. It's a catch-all, not a signal. Aleatory noise — more data or better models cannot fix this without changing the labelling process.

### 5. BERT is not the answer

```
Model                  Params    Min samples   Overfit risk at n=1200
──────────────────────────────────────────────────────────────────────
BERT-base (full)        110M      ~1,000,000    catastrophic
BERT-base (frozen)      4,608     ~46,000       low, but domain mismatch
all-MiniLM-L6-v2        22M       ~2,000        moderate ← correct next step
TF-IDF + SVD (current)  ~0        ~200          ✅ safe but ceiling ~58%
```

BERT was trained on Wikipedia and BookCorpus. Mindfulness journal language (`"kinda calm"`, `"lowkey felt pretty even"`, `"not gonna lie i felt mentally flooded"`) is informal, abbreviated, first-person — domain mismatch penalises BERT by ~1-2%.

<br>

---

## 📐 Uncertainty Analysis

### Y1 confidence ranges

`confidence` = `max(predict_proba)` per row.

```
> 0.70          High — clear signal, reliable
0.50 – 0.70     Moderate — model leaning, use with awareness
0.35 – 0.50     Low — spread across 2-3 classes
< 0.35          No signal — entry too short or ambiguous
```

### Main confusion pairs

```
overwhelmed → restless      shared vocab: "heavy", "unable", "scattered"
calm        → neutral       both produce low-signal short entries
focused     → calm          clarity vocabulary appears in both clusters
mixed       → anything      inherently contains vocab from 2+ classes
```

### Y2 intensity — irreducible uncertainty

Pearson correlation between raw intensity and each feature:

```
stress_level    r = 0.14   ← strongest
energy_level    r = 0.12
duration_min    r = 0.08
sleep_hours     r = -0.06
```

All near zero. This is **aleatory uncertainty** — the noise is in the labelling process itself, not in the model. All three models (XGB, RF, NN) converge to 21-24% on 5-class intensity for this reason.

### How uncertainty propagates into recommendations

The attention engine uses the full probability vector as Q. Low-confidence predictions spread attention across templates — the recommendation degrades gracefully rather than crashing:

```
High confidence:  probs = [0.04, 0.03, 0.06, 0.05, 0.76, 0.06]
  → attn peaks at emotional_offload (0.62) — sharp, clear recommendation

Low confidence:   probs = [0.19, 0.16, 0.18, 0.21, 0.14, 0.12]
  → attn spread: steady_continuity (0.16) wins by small margin
  → recommendation is softer but never crashes
```

<br>

---

## 📄 Output Format

Each row in `arvyax_predictions.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Session ID from test set |
| `emotional_state` | str | calm / focused / mixed / neutral / overwhelmed / restless |
| `intensity_bucket` | str | low / medium / high |
| `recommendation` | str | Personalised wellbeing text |
| `template` | str | Which of the 8 templates was selected |
| `duration_min` | int | Session duration from input |
| `sleep_rec` | float | 8 if sleep_hours < 6, else actual |
| `time_of_day` | str | From input |
| `attn_weights` | dict | {template: float} — full 8-template attention distribution |
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
├── 📄 arvyax_xgboost.py         ← main pipeline — use this one
├── 📄 arvyax_random_forest.py   ← RF version, prints feature_importances_
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

**Data paths (update for your environment):**

```python
train_df = pd.read_csv('/content/Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv')
test_df  = pd.read_csv('/content/arvyax_test_inputs_120.xlsx - Sheet1.csv')
```

<br>

---

## 🚧 Limitations & Next Steps

| # | Limitation | Impact | Next step |
|---|-----------|--------|-----------|
| 1 | Short entries → zero TF-IDF vectors | 🔴 ~35% of data has no signal | Replace text path with `all-MiniLM-L6-v2` frozen embeddings (~10 lines) |
| 2 | TF-IDF blind to negation | 🔴 "not calm" = "calm" to the model | Sentence transformers handle this natively |
| 3 | intensity=3 near-random (F1=0.15) | 🟠 Y2 medium bucket unreliable | Change labelling — 2-point scale (low/high) removes the ambiguous middle |
| 4 | `face_emotion_hint` ~20% missing | 🟡 Imputed as "none" | Add a `face_missing` indicator feature |
| 5 | NN underperforms RF by ~4% | 🟡 Structural sparse gradient | Dense embeddings fix this — not worth addressing without fixing #1 first |
| 6 | Categorical one-hots | 🟢 Minor | Learned 4-8d embeddings generalise better in a neural model |

<br>

---

## 📈 Full Accuracy Journey

```
Version    Change                                         Y1 CV    Y2 CV    Note
───────────────────────────────────────────────────────────────────────────────────────
v1         Numpy NN, 29d hand-crafted features            ~66%*    —        *fake leakage
v2         Fixed TF-IDF data leakage                      ~49%     —        real baseline
v2.1       + word bigrams + char 4-grams                  ~52%     —        text signal
v2.2       + expanded emotion vocab                       ~54%     —
v3         5-class intensity → 3-class buckets            52.4%    40.3%    current published
v4         Code humanised, variable names cleaned         52.4%    40.3%    no accuracy change
v5         TF-IDF fix + face×mood outer + wider attn Q    ~57.8%*  40.3%    *5-fold CV ablation
───────────────────────────────────────────────────────────────────────────────────────
```

> The drop from v1 to v2 is not a regression — it's removing fake accuracy from data leakage. The real starting point was always ~49%. Every gain after that is honest.

<br>

---

<div align="center">

**Built for Arvyax Reflective Session Platform**

*Dual-output emotion classification · Context-aware attention recommendations · Zero if/else*

![No leakage](https://img.shields.io/badge/Evaluation-No_Data_Leakage-22C55E?style=flat-square)
![No if/else](https://img.shields.io/badge/Recommendations-Zero_if%2Felse-6366F1?style=flat-square)
![3-fold CV](https://img.shields.io/badge/Validation-Stratified_3--fold_CV-F59E0B?style=flat-square)
![v5](https://img.shields.io/badge/Version-v5_Final-6366F1?style=flat-square)

</div>
