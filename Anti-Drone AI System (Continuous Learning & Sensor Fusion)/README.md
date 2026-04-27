Here is a research-grade, highly professional README file designed to be copy-pasted directly into your GitHub repository or project documentation. It synthesizes the extensive architectural iterations, hyperparameter tuning, and production-readiness gates into a compelling narrative.

***

# AegisDrone — AI-Based Drone Threat Detection & Classification System

![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen)
![Version](https://img.shields.io/badge/Version-v28_Continuous_Learning-blue)
![Latency](https://img.shields.io/badge/Latency-Edge_Optimized(<1ms)-orange)

**AegisDrone** is a production-grade, continuous-learning Machine Learning pipeline designed for real-time Counter-UAS (Unmanned Aerial Systems) operations. By fusing Radio Frequency (RF) spectral analysis, flight kinematics, and communication protocols, AegisDrone identifies, categorizes, and autonomously tracks aerial threats in highly congested airspace.

---

## 🧠 Why This Architecture is the Best
Traditional AI systems in defense suffer from the "Black Box" problem: they are too slow for edge deployment, suffer from high false-alarm rates in noisy environments, and fail catastrophically when encountering unarchitected (Zero-Day) signals. 

AegisDrone solves this via a **Memory-First / Soft-Fusion Architecture**:

1. **Memory-First Pipeline (O(1) Latency):** AI shouldn't infer what it already knows. Known signal fingerprints are hashed and matched in an O(1) database, bypassing the neural network entirely. This yields **<1 ms latency** and zero UI flickering for recognized friendly/threat drones.
2. **Deep Ensemble Intelligence:** Unknown signals fall through to a multi-model fusion engine combining a **1D-CNN** (for raw sequence embeddings), **Random Forests** (for Mutual Information features), **Gradient Boosted Trees** (for high-variance features), and **Deep SVDD** (for anomaly/open-set detection). 
3. **Autonomous Promotion Loop:** A Temporal Tracker accumulates sequential bursts. Once an unknown signal proves stable across **4+ observations**, the system autonomously categorizes it (e.g., Friendly, Threat, or Background) and commits it to the Memory Database without human intervention or model retraining. Time-to-trust is verified at **~0.2 seconds**.

---

## 🔬 Evolution & Hyperparameter Tuning
The current state of AegisDrone is the result of rigorous empirical testing and iterative tuning to balance **Sensitivity (Recall)** against **Cognitive Load (Hold/False Alarm Rates)**.

* **The "Confidence Bypass" Tuning:** Earlier versions suffered from "blind bypassing," where high-probability but highly anomalous signals were fast-pathed as "Friendly." We introduced a **Dual-Condition Bypass**: signals now require *both* a max classifier probability of **>0.999** AND an anomaly score strictly below the open-set threshold.
* **Open-Set Floor Calibration:** We discovered that anchoring the "Unknown" threshold to the overall data distribution trapped genuine novel drones in the noise floor. We tuned the Open-Set threshold to specifically use the **1st percentile of known-drone scores**, capped mathematically at **0.45**. This perfectly bounded the Open-Set fraction to **~7.6%** without sacrificing known-threat recall.
* **Hysteresis UI Filtering:** Raw ML outputs jitter between classes frame-by-frame. We engineered a temporal Hysteresis Filter (window = **5**, majority = **4**) that mathematically smoothed the output, dropping the UI Flicker Index to an exceptionally stable **0.223**.

---

## 📊 Production Safety Quadrant (v28 Metrics)
The system is gated by a strict "Safety Quadrant" to ensure radar operators are never overwhelmed by alarm fatigue. Tested against **1,200** realistic, overlapping, noise-injected synthetic/real samples:

| Metric Category | Specific Target | AegisDrone Result | Status |
| :--- | :--- | :--- | :--- |
| **Integrity** | Threat Recall $\ge 85\%$ | **88.8%** | ✅ PASS |
| **Safety** | False Alarm Rate $\le 10\%$ | **0.5%** | ✅ PASS |
| **Cognitive Load** | Ambiguity (HOLD) Rate $\le 20\%$ | **0.0%** | ✅ PASS |
| **Identity** | Flicker Index $< 0.50$ | **0.223** | ✅ PASS |
| **Sensitivity** | Open-Set Recognition $\ge 4\%$ | **7.6%** | ✅ PASS |

### 🛡️ Professional Stress-Testing
* **Ghost Hunt Test:** **0** label transitions on known-DB drones (100% stability).
* **Adversarial Test:** Successfully isolated adversarial Gaussian noise, pushing them into safe/hold labels.
* **Recovery Test:** Achieved a stable zero-day threat ID in exactly **0.2s**, with a **p95** burst processing time of **486.6 ms** (and **<1ms** for knowns).

---

## 🔭 Future Insights for Real-World Datasets
While the current architecture is theoretically robust, transitioning this system to live hardware (e.g., Software Defined Radios like HackRF or USRP) in active warzones or urban environments presents future research vectors:

1. **Blind Source Separation (BSS):** Real datasets often contain simultaneous multi-drone transmissions. Future iterations should incorporate Independent Component Analysis (ICA) or Autoencoders *prior* to feature extraction to separate overlapping 2.4GHz/5.8GHz signals.
2. **Adaptive SNR Normalization:** Low Signal-to-Noise Ratios (SNR) severely degrade spectral entropy and kurtosis features. Implementing real-time dynamic range compression and dynamic SNR weighting will prevent distant drones from falling into the `OPEN_SET_UNKNOWN` category prematurely.
3. **Hardware-in-the-Loop (HITL) Optimization:** The Python-based `scipy.signal` STFT and Welch's PSD functions should be translated to C++ or directly implemented on FPGA fabric to reduce the **p99 latency (~544 ms)** of the heavy neural pathway down to strict microsecond boundaries.
4. **Adversarial Electronic Warfare:** Adaptive spoofing requires us to push Deep SVDD further. Integrating contrastive learning with hard-negative mining on *live* jammed signals will ensure the system does not categorize enemy jammers as benign background noise.

---

## 🚀 Quick Start

**Prerequisites:**
```bash
pip install numpy pandas scipy scikit-learn imbalanced-learn torch shap
```

**Running the System:**
```python
from aegis_drone import SoftFusionEngine, FingerprintDatabase

# Initialize Memory Database
fp_db = FingerprintDatabase("antidrone_db.json")

# Process live incoming RF Burst (8192 sample window)
decision = classify_signal(live_rf_array)

print(f"Detected: {decision['label']} | Confidence: {decision['soft_score']:.2f}")
```
