# 📊 Churn Prediction Model Comparison – README

## 📌 Problem Statement
The objective is to **predict customer churn** in a **highly imbalanced dataset**, where the number of churned customers is much smaller than non-churned customers.

In this business context, the **primary goal is to identify as many churned customers as possible**, even if that means allowing some false alarms.

➡️ **Recall is more important than accuracy or precision**.

---

## ⚖️ Dataset Challenge: Class Imbalance
- Majority class: Non-churn
- Minority class: Churn

**Why imbalance matters:**
A model can achieve high accuracy by always predicting “No Churn” — but that would be useless for business.

Therefore:
- Accuracy ❌ misleading
- Recall ✅ critical
- F1-score ✅ secondary (balance metric)

---

## 🧠 Models Evaluated
All models were evaluated using **Stratified Cross-Validation** to preserve churn distribution across folds:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  

Each model was:
- Class-weighted / imbalance-aware
- Evaluated using **Recall & F1-score**
- Tested using a recall-oriented probability threshold

---

## 📈 Results Summary (Stratified CV)

| Model                | Recall | F1 Score |
|---------------------|--------|----------|
| **Random Forest**   | **0.86** | 0.36 |
| Decision Tree       | 0.81 | 0.37 |
| Logistic Regression | 0.73 | 0.25 |
| XGBoost             | 0.56 | 0.35 |

---

## 🏆 Model Selected: **Random Forest**

### Why Random Forest performed best

**Fundamental reasoning:**

- Random Forest uses **bagging** (bootstrap aggregation)
- Each tree learns from a slightly different sample
- Noise is *averaged out*, not amplified
- Leads to **broader decision boundaries**
- Better at capturing **minority-class patterns**

➡️ This directly improves **recall**, which is the business priority.

---

## 🤔 Why XGBoost did not win here (important insight)

XGBoost is a **boosting-based** model.

Fundamentally:
- Boosting focuses on correcting previous mistakes
- It aggressively fits hard-to-classify points
- This creates **tight, confident decision boundaries**
- Noise is suppressed

**Impact on churn prediction:**
- Fewer false positives ✅
- More missed churners ❌

➡️ Lower recall is an expected outcome, not a failure.

---

## 💼 Business Interpretation of “Noise”

In churn prediction:
- False Positive = Customer flagged as churn risk but stays
- False Negative = Customer churns without intervention

**Cost comparison:**
- False Positive → email / discount / call
- False Negative → lost customer + lifetime value

➡️ Accepting noise is cheaper than missing churn.

---

## 🧪 Why This Is the Right Engineering Decision

There is **no universally best model**.

> The best model is the one that aligns with the **cost structure of the problem**.

- XGBoost → precision-oriented, high confidence
- Random Forest → recall-oriented, robust to imbalance

For churn prevention:
- Recall dominates
- Random Forest aligns better


## 🚀 What This Demonstrates
- Understanding of **model fundamentals**
- Correct metric selection for imbalanced data
- Business-aligned ML decision making
- Ability to challenge “model hype” with evidence

📌 *Model choice driven by reasoning, not reputation.*

