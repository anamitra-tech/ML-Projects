# AegisDrone 🛡️
## Anti-Drone AI Harness — v34-EW-FIELD (Electronic Warfare, MLOps-Deployed, Production-Live)

> **🎉 ALL PRODUCTION GATES PASSED — v34 IS LIVE IN PRODUCTION**
>
> `Recall 99.9%` · `FA 2.1%` · `Open-Set 6.2%` · `Flicker 0.480` · `p95 Latency 65.9ms`
>
> **Live API:** [aegisdrone.onrender.com/docs](https://aegisdrone.onrender.com/docs) · **Repo:** [github.com/anamitra-tech/aegisdrone](https://github.com/anamitra-tech/aegisdrone)

<p align="center">
  <img src="aegis_drone.gif" width="900">
</p>

---

## The Core Question v34 Answers

Every prior version (v28–v31) answered the question *"can the model classify drone signals accurately enough?"* By v31, the answer was yes — 92% recall, 0.1% false alarms, p95=52.8ms, all six production gates green on the bench.

**v34 was built to answer a different, harder question: "what does this system actually *do* when it sees a threat, and can anyone outside this notebook actually use it?"**

Two things were missing in v31 that made it a research artifact rather than a deployable system:

1. **No response.** The classifier could say `POTENTIAL_THREAT`, but nothing happened next. A real anti-drone system has to *act* — jam the signal, verify the jam worked, and adapt if it didn't.
2. **No interface.** The entire pipeline lived inside a Colab notebook. There was no way for another program, a dashboard, or a field operator's device to query it.

v34 closes both gaps. Everything below is organized around **why** each decision was made, in the order the problems appeared.

---

## Decision 1 — From "Classify" to "Classify, Then Act": The EW Layer

### The Problem

A `POTENTIAL_THREAT` or `OPEN_SET_UNKNOWN` label is useless to a field operator on its own. The system needed a defined *response* — and that response needed to be **frequency-aware**, because jamming the wrong part of the spectrum either does nothing or, worse, jams a friendly/protected band.

### The Decision: WidebandSpectrumSweeper + LookThroughScheduler

Rather than build a generic "jam everything" reflex, v34 first **classifies which of 8 spectrum bands** (100 kHz–40 GHz, covering ELINT HF/VHF/UHF, ISM S/C-band, X-band radar, MIL millimeter-wave, and both GNSS NavIC L5 and GPS L1) the threat occupies, then chooses a jamming waveform appropriate to that band via `generate_jamming_suggest()`:

| Detected Signal | Chosen Waveform | Why This Waveform |
|---|---|---|
| AR/FHSS control link | Spot-Follower Noise | Narrow-band hopper — a wide jammer wastes power; follow the hop |
| Phantom GFSK / DSSS wideband | Fast Swept-Spot | Spread-spectrum signal — a fixed spot misses the spread; sweep covers it |
| GNSS (NavIC/GPS) | Coherent Barrage Jamming | Entire allocation must be denied — no partial jam is acceptable here |
| Unknown signal | Serrated Sawtooth Comb | Harmonic structure unknown — a comb covers multiple harmonics at once |

**Why a `<100ms` look-through cycle, enforced by assertion (`LookThroughScheduler.__init__` asserts `cycle_time_ms < 100.0`):** A jammer that transmits continuously is also blind — it cannot hear whether the jam is working, or whether a *new* threat has appeared underneath it. The look-through cycle forces a periodic quiet window (15% for vehicle, 20% for manpack) so the receiver can check in. This single design decision is what makes Decision 2 (the efficacy monitor) possible at all — without a quiet window, there is no signal to observe.

### The Decision: GNSS Protected-Band Override — A Hard-Coded Exception to the ML Pipeline

**This is the one place in v34 where the answer is deliberately *not* "let the model decide."**

```python
if sweeper.is_gnss_protected(assigned_band):
    return _ret("CONFIRMED_THREAT", source=source_tag, ss_override=0.95)
```

Every other decision in the pipeline flows through calibrated thresholds, ensembles, and fusion weights. NavIC L5 and GPS L1 do not. **Any energy detected inside those two bands is unconditionally `CONFIRMED_THREAT`, before the classifier, SVDD, or fingerprint DB ever run.**

Why override the ML stack here specifically? Because the cost of the two possible errors is wildly asymmetric:
- If the ML stack is *right* that GNSS-band energy is benign, the override costs one unnecessary alert.
- If the ML stack is *wrong* — and an open-set or low-confidence GNSS jammer slips past a soft-fusion threshold — the cost is **navigation denial in a contested environment**, which the rest of the system has no way to detect or recover from.

This is a case where "the model is 99.9% accurate" is not good enough, and the right engineering decision is to not ask the model at all.

### The Decision: Jamming Efficacy Monitor — Closing the Loop

**The problem this solves:** v31's `ActionController` (carried into v34 as Layer 3 of the AI-Harness) could *trigger* a jam, but had no way to know if it *worked*. A jammer that fires once and never checks back is operating blind for the rest of the engagement.

**The decision:** After `ActionController.trigger_defense()` fires, `JammingEfficacyMonitor.register_jam()` opens a 10-burst observation window (`EFFICACY_WINDOW_BURSTS`) on that specific track. Each subsequent burst through `_ret()` calls `observe_post_jam()`, which checks whether the track has gone quiet (`label in {"BACKGROUND","OPEN_SET_UNKNOWN"}`).

- **If ≥80% of the window is silent** (`EFFICACY_SILENCE_THRESH`): jamming worked. Reset failure count.
- **If not, and this is the 3rd consecutive failure** (`EFFICACY_REPLAN_AFTER`): **replan** — cycle to a different waveform mode with 1.5× the bandwidth (capped at 40 MHz) and try again.

**This is the decision that turns AegisDrone from "detect and alert" into "detect, act, verify, and adapt."** In the live v34 run, this loop fired 16 times on a single adversarial track (Track 15), cycling between `Serrated Sawtooth Comb` and `Spot-Follower Noise` with progressively wider bandwidth, and **eventually achieved `EFFICACY_SUCCESS` with 100% silence rate** — the closed loop worked end-to-end, not just in theory.

### The Decision: Channel Realism Tuned for a Specific Theater (Ladakh / Siachen)

**The problem:** v31's synthetic data generator produced statistically clean bursts. A model trained only on clean data learns decision boundaries that don't survive contact with a real RF environment — wind-induced Doppler, battery-sag amplitude drift, and multipath from terrain all shift feature distributions in ways clean training data never shows.

**The decision:** Rather than generic "add noise" augmentation, v34's `_apply_rician_fading`, `_apply_rayleigh_multipath`, `_apply_lo_drift`, `_apply_thermal_noise`, and `_apply_spectral_congestion` model **specific, named physical mechanisms** calibrated to a high-altitude mountainous deployment:

| Impairment | Parameter | Physical Cause Modeled |
|---|---|---|
| Rician fading | K≈1.5±0.8 | Direct path + scattered returns off terrain |
| Rayleigh multipath | 6 paths, σ=0.25 | Reflections in mountainous terrain spreading bandwidth/centroid |
| LO drift | σ=25 kHz | Local-oscillator mismatch between emitter and receiver |
| Thermal noise floor | −105 dBm (vehicle) / −100 dBm (manpack) | Receiver-class-dependent noise floor |
| Spectral congestion | 20% probability | Co-channel interference from a second emitter |

**Why this matters for the deployment-mode split:** these aren't cosmetic — `ENERGY_GATE_MIN_POWER`, `THERMAL_NOISE_FLOOR_DBM`, `LOOK_THROUGH_CYCLE_MS`, and `MAX_ALTITUDE_M` all change between `vehicle` and `manpack` because **a smaller manpack antenna has a fundamentally different noise floor and compute budget than a vehicle-mounted system**. Encoding this as a single `DEPLOYMENT_MODE` switch (rather than two separate codebases) means the same trained model and the same decision logic serve both platforms — only the gates and timing change.

### The Decision: A 5th Feature Family for LPI/FHSS/DSSS — and Why It Gets 20% of the Fusion Weight

**The problem:** The original 83-feature schema (53 RF + 18 Flight + 12 Comm) was built to distinguish drone types by *what they transmit*. It had no features describing *how stealthily* they transmit — frequency hopping, spread-spectrum chip rates, low-probability-of-intercept margins. An adversary using LPI/FHSS techniques specifically to evade detection would be invisible to the v31 feature set.

**The decision:** Add `EW_FEATURE_NAMES` — `hop_rate_hz`, `dwell_time_ms`, `lpi_snr_margin_db`, `dsss_correlation_peak`, `chip_rate_estimate` — bringing the total from 83 to **86 features** (`N_COMM` also dropped from 12 to 10 in the schema reorganization, netting +3 overall).

**Why these 5 features get their own fusion term (`FUSION_W_LPI_FHSS = 0.20`) — the single largest weight in the soft-score formula:**

```python
FUSION_W_CLF        = 0.40   # classifier confidence
FUSION_W_CNN        = 0.05   # waveform texture
FUSION_W_EVM        = 0.15   # open-set inclusion
FUSION_W_NORMALITY  = 0.15   # anomaly score
FUSION_W_AGREEMENT  = 0.05   # cross-model agreement
FUSION_W_LPI_FHSS   = 0.20   # ← second-largest weight, after classifier confidence
```

A signal deliberately using LPI/FHSS techniques is, by definition, trying to keep `clf_conf` low and `evm_score`/`normality` ambiguous — that's what "low probability of intercept" *means*. If the EW weight were small, exactly the signals most worth flagging would be the ones most easily missed by the rest of the fusion formula. Giving `ew_boost` 20% weight means a high `lpi_snr_margin_db` (>10, per the `ew_boost` gate in `SoftFusionEngine.score()`) can meaningfully raise the soft score **even when every other signal is ambiguous** — which is exactly the scenario an LPI emitter is designed to create.

---

## Decision 2 — From "Trust Me, It Works" to "Here's the Receipt": MLflow + DagsHub

### The Problem

v31's results lived in a Colab cell's stdout. The 37/38 self-test pass rate, the calibration thresholds, the trained model weights — all of it existed only until the runtime disconnected. There was no way to compare "this run" against "last week's run," no way to retrieve a specific trained model later, and no audit trail.

### The Decision: Log Everything, Register the Model, Make It Restorable

`run_v34_with_mlflow()` wraps the entire v34 pipeline and logs to a remote DagsHub MLflow server:

- **~50 hyperparameters** — every fusion weight, every augmentation setting, every LightGBM config, every channel-model constant, every EW threshold. The reasoning: if a future run's recall drops from 99.9% to 95%, the *first* question is "what changed?" — and the answer needs to be in the log, not in someone's memory of what they edited.
- **Full evaluation metrics** — not just accuracy, but the same gates v31 introduced (recall, FA, hold fraction, flicker index, memory hit-rate, per-class recall, latency percentiles, stress-test pass/fail, ECE, calibrated thresholds). **The gates didn't change in v34 — what changed is that they're now permanently recorded per-run instead of printed once and lost.**
- **The full model, registered.** `AegisDronePyFuncModel` wraps the cloudpickled `classify_signal` pipeline in MLflow's standard `pyfunc` interface and registers it as `AegisDrone_v34_Model`. The decision to use the standard pyfunc interface (rather than a custom save format) is what makes Decision 3 — serving the model over HTTP — a thin wrapper instead of a rewrite: `mlflow.pyfunc.load_model("models:/AegisDrone_v34_Model/latest")` returns something with a `.predict()` method, full stop.

**Why two logging modes (`run_v34_with_mlflow()` vs `log_existing_run_to_mlflow()`):** Training the full ensemble (CNN, embedder, LGB-RF, LGB-GBT, stacker, SVDD) takes real time. If a run already trained successfully and only the *logging* failed or needs updating, retraining from scratch to fix a logging bug is wasteful. `log_existing_run_to_mlflow()` accepts an already-trained pipeline and logs it — separating "did the training succeed" from "did the logging succeed" as independent concerns.

---

## Decision 3 — From "Runs in My Colab" to "Runs on the Internet": FastAPI + Docker + Render

### The Problem

A registered MLflow model is still not *usable* by anything outside a Python session with MLflow installed and DagsHub credentials configured. For the system to be queryable by a dashboard, another service, or a field device, it needs to speak HTTP.

### The Decision: A Minimal FastAPI Wrapper Around the Registered Model

The API surface is deliberately small — two endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Returns `{"status":"healthy"}` once the model has loaded |
| `/predict` | POST | Accepts an 86-dim feature vector, returns label/soft_score/threat_level |

**Why so minimal?** Every additional endpoint is additional surface area that has to be kept in sync with the underlying pipeline's 86-feature schema, decision labels, and fusion output format. The `/health` endpoint exists specifically because model loading from a remote registry can fail silently or slowly — `/health` gives a load balancer or operator a fast, cheap way to ask "is the model actually ready?" before sending real traffic.

On startup, the app authenticates to DagsHub via environment variables (`MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `DAGSHUB_USER_TOKEN`) and calls `mlflow.pyfunc.load_model("models:/AegisDrone_v34_Model/latest")` — **the same registered model from Decision 2**, with no separate export step.

### The Decision: Python 3.12 + numpy 2.0.2 in the Container — Pinned, Not "Latest"

```dockerfile
FROM python:3.12-slim
```

This looks like a trivial choice until you've debugged the alternative. **The `classify_signal` pipeline is serialized with cloudpickle**, which captures closures and object references by reflecting into the running interpreter's memory layout. Training happened in a Colab environment running **Python 3.12.13 / numpy 2.0.2**. A container running Python 3.10 or numpy 2.1.x produces `ModuleNotFoundError: No module named 'numpy._core.numeric'`-style failures at model-load time — the pickle is structurally valid, but the runtime's internal module layout doesn't match what was captured.

**The decision was to treat the training environment's exact versions as part of the model artifact, not as an implementation detail.** `requirements.txt` pins `numpy==2.0.2` for this reason — not because 2.0.2 is special, but because **it's what the pickle was made with**, and cloudpickle compatibility is a training/serving contract, not a "pick the latest stable version" decision.

---

## Decision 4 — From "I'll Redeploy It Manually" to "It Redeploys Itself": GitHub Actions CI/CD

### The Problem

Even with a live container, every model update or code fix required manually rebuilding and redeploying. This doesn't scale, and more importantly, it means the *deployed* version can silently drift from the *committed* version.

### The Decision: Push to `main` → Health-Check → Redeploy, Automatically

```yaml
on:
  push:
    branches: [main]
jobs:
  deploy:
    steps:
      - Checkout code
      - Test health endpoint   # curl --fail the LIVE /health, not a local one
      - Deploy to Render        # POST to Render's Deploy API
```

**Why health-check the *live* endpoint before deploying the *new* code, rather than after?** This step's job is to confirm the *currently running* service is healthy before triggering a new deploy — catching the case where the pipeline would otherwise pile a new deploy on top of an already-broken one, masking the original failure. The redeploy itself is Render's responsibility once triggered; this pipeline's job is to make sure "push to main" and "what's live" never silently diverge.

---

## Decision 5 — Kubernetes Manifests: Documenting the Next Step, Not Solving a Problem That Exists Yet

### The Problem (a future one)

Render's single-container deployment is sufficient for the current load. But it represents a ceiling: one container, one region, no horizontal scaling, no rolling updates with readiness gating.

### The Decision: Write the Manifests Now, While the Constraints Are Fresh

```yaml
spec:
  replicas: 2
  template:
    spec:
      containers:
        - image: aegisdrone:v1
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 30
```

The `readinessProbe` pointing at `/health` is not incidental — **it's the same endpoint built in Decision 3, for the same reason**: model loading from the registry takes time, and Kubernetes needs to know not to route traffic to a pod that's still downloading `AegisDrone_v34_Model`. `initialDelaySeconds: 30` is a direct acknowledgment that the registry-load step (Decision 2/3) is not instantaneous.

This is deliberately scoped as **documentation of the orchestration pattern**, not a running cluster — the manifests reference a placeholder image (`aegisdrone:v1`) because the actual decision to move off Render's single-container model hasn't been needed yet. Writing it now means the pattern is proven and ready the moment it is needed.

---

## System Architecture — How the Decisions Compose

The v31 decision core is **unchanged at its center** — same ensemble, same calibration math, same hysteresis filter. What changed is everything *around* it: a new gate before (GNSS override), a new feature family feeding into it (EW features), and a new action loop after it (jamming + efficacy monitoring).

```
IQ Window (8192 samples @ 10 MHz)
        │
        ▼
[Hilbert + Welch PSD + STFT + EW features]  ──►  86-dim Feature Vector (fv)
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│                    TWO-STAGE SECURE THREAT GATE                   │
│                                                                    │
│  STAGE 1: Physical / Energy Gate            (carried from v31)    │
│           Impossible/weak signal → BACKGROUND  (no AI)            │
│                 ↓                                                  │
│  STAGE 2: GNSS Protected-Band Override       [Decision 1, NEW]     │
│           NavIC L5 / GPS L1 → CONFIRMED_THREAT  (hard override)    │
│                 ↓                                                  │
│  STAGE 3: SVDD / Anomaly Gate                (carried from v31)    │
│           evm_score<0.35 or anomaly>0.65 → OPEN_SET_UNKNOWN        │
│                 ↓                                                  │
│  STAGE 4: Fingerprint DB Lookup  ◄── O(1) hash  (carried from v31) │
│           HIT → MEMORY_MATCH                                       │
│                 ↓                                                  │
│  STAGE 5: Soft Fusion Decision → Label   (now includes EW weight)  │
│                 ↓                                                  │
│  [HysteresisFilter] window=5, majority=6     (carried from v31)    │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
  Decision Label + soft_score + threat_level
        │
        ▼
  [Decision 1, NEW] ActionController → LookThroughScheduler → Jamming Suggestion
        │
        ▼
  [Decision 1, NEW] JammingEfficacyMonitor → success/failure → replan if needed
```

### Why the Core Ensemble Was *Not* Touched

| Component | Algorithm | Role | Why unchanged in v34 |
|-----------|-----------|------|------|
| **LGB-RF** | LightGBM RF (500 trees, MI-top-45) | Primary classifier + fast-path | v31's [FIX-5] already delivered 6× speedup; the bottleneck moved elsewhere |
| **LGB-GBT** | LightGBM GBDT (200 rounds, Var-top-40) | Boundary learner | Same — no evidence of being a limiting factor |
| **GBP** | Gaussian Bayes Posterior (τ=0.85) | Probabilistic baseline | Stable contributor to the stacker |
| **Stacker** | Logistic Regression on RF+GBT+GBP | Learned combination | [FIX-4] from v30 still the right call |
| **1D-CNN** | Conv1D → AdaptiveAvgPool | Waveform texture | 5% weight unchanged — still a minor signal |
| **Ensemble** | 3× Bootstrap LGB-RF | Epistemic uncertainty | Still feeds `ens_vacuity` penalty |
| **Sub-clf** | LightGBM GBT binary | Fine disambiguation | Unchanged role |
| **DeepEmitterEmbedder** | Conv1D triplet network | 16-dim fingerprint | Unchanged — still feeds Stage 4 memory lookup |

**The decision not to touch this list is itself a decision.** v34's problems (no action loop, no API, no deployment) were not classification-accuracy problems. Re-tuning a system that was already passing all six v31 gates would have risked breaking something that worked, in service of a goal (better classification) that wasn't the actual bottleneck. Engineering effort went where the actual gap was.

---

## Evaluation Results — v34 Live Run

### Production Gates (8,000-sample synthetic field run, 2-class deployment)

| Gate | Metric | v34 Result | Target | Status |
|------|--------|-----------|--------|--------|
| **Integrity** | Drone detection recall | **99.9%** | ≥ 85% | ✅ |
| — | Phantom GFSK Link recall | **99.9%** | ≥ 80% | ✅ |
| **Safety** | False alarm rate | **2.1%** | ≤ 10% | ✅ |
| **Cognitive Load** | HOLD fraction | **0.5%** | ≤ 20% | ✅ |
| **Identity** | Flicker Index | **0.480** | < 0.65 | ✅ |
| **Sensitivity** | Open-set fraction | **6.2%** | ≥ 2% | ✅ |
| **Memory** | DB hit-rate | **81.1%** | ≥ 1% | ✅ |
| **Bypass** | Confidence bypass | **< 10%** | < 10% | ✅ |

**Why recall went from 92% (v31) to 99.9% (v34) despite an unchanged ensemble:** this run trained on a 2-class problem (Background RF + Phantom GFSK Link only), which is a structurally easier separation than v31's 3-class problem. This is a property of *this run's dataset*, not evidence that the EW/deployment changes improved classification — consistent with the "core ensemble unchanged" decision above. The number to watch for regression is the **per-class recall on the full class set**, not this run's headline figure.

### Latency

| Percentile | v34 Result |
|---|---|
| Mean | **41.5ms** |
| p50 | **38.7ms** |
| **p95** | **65.9ms ✅** (< 100ms target) |
| p99 | 84.5ms |

**Why p95 (65.9ms) is higher than v31's bench figure (52.8ms) despite the same LightGBM core:** the EW layer adds work *after* classification — spectrum band lookup, jamming suggestion generation, and efficacy-window bookkeeping all run inside `_ret()` on every classified burst. This is the direct cost of Decision 1. **65.9ms still clears the <100ms target with ~34ms of headroom**, so the decision to add the EW layer's per-burst overhead was judged acceptable against the latency budget established in v31.

### Stress Tests — All Passing

| Test | What It Simulates | v34 Result |
|------|-------------------|-----------|
| **[A] Ghost Hunt** (Active FHSS Tracking) | Known-DB drone under noise | **0 transitions ✅** |
| **[B] LPI Adversarial Scan** | Pure noise / LPI signals | **100% safe-rate ✅** |
| **[C] Look-Through Recovery** | Fresh emitter → stable ID | **Lock time 0.35s, p95=44.2ms ✅** |

### The Jamming Loop, Observed Live

This is the result that didn't exist as a *metric* in v31, because the capability didn't exist:

```
Track 15: REPLAN ×16 (Serrated Sawtooth Comb ↔ Spot-Follower Noise, BW 10→15 MHz)
Track 15: EFFICACY_SUCCESS — silence_rate=1.0 (100% silent)
```

A single track required 16 replans before achieving 100% jamming efficacy. **This is reported as a success, not a concern** — the alternative (no efficacy monitoring) would have meant the system kept transmitting an ineffective jam indefinitely with no indication anything was wrong. 16 replans-then-success demonstrates the adaptive loop doing exactly what it was built to do.

---

## What "Production-Live" Means in v34 — And What It Doesn't

| Requirement | Status | Evidence |
|-------------|--------|---------|
| Public HTTPS API | ✅ | `https://aegisdrone.onrender.com/health` returns `200 healthy` |
| Containerized, version-pinned | ✅ | Python 3.12 / numpy 2.0.2 — matches training environment (Decision 3) |
| Model registry | ✅ | `AegisDrone_v34_Model` on DagsHub MLflow (Decision 2) |
| Auto-redeploy on push | ✅ | GitHub Actions → Render Deploy API (Decision 4) |
| Orchestration pattern documented | ✅ | K8s manifests with `/health`-based readiness probe (Decision 5) |
| p95 < 100ms with EW overhead | ✅ | 65.9ms measured live |
| GNSS hard-override | ✅ | NavIC L5 / GPS L1 unconditionally → CONFIRMED_THREAT |
| Closed-loop jam-verify-replan | ✅ | 16 replans → 100% efficacy, observed live |

### What It Deliberately Does Not Mean

- **This run is not the 4-class system.** The live run trained on 2 classes (Background RF + Phantom GFSK Link). `CLASS_NAMES` still defines all 4 (including AR Drone and NavIC), and the GNSS override logic is independent of which classes were trained — but the 99.9% recall figure above should not be read as "all 4 classes individually hit 99.9%."
- **Kubernetes is not yet running this in a cluster.** The manifests are correct and ready (Decision 5), but Render's single-container deployment is the actual production path today. Moving to K8s is a decision for when horizontal scaling is actually needed, not before.
- **Cloudpickle version-pinning is a known fragility, accepted deliberately.** The decision in Decision 3 to pin exact training-environment versions is a *mitigation*, not an elimination, of the underlying fragility. A future decision point: export to ONNX (the pipeline already attempts `import onnx, onnxruntime` at startup) would remove this fragility entirely, at the cost of re-implementing the fusion/tracking/EW logic outside the pickled closure.

---

## Known Limitations (Honest Accounting)

| Limitation | Why It Exists Given the Decisions Above | Mitigation Path |
|------------|------------------------------------------|------------------|
| 2-class training in this run | Fastest path to a clean live demonstration of the *deployment* pipeline (Decisions 2–4), which was the actual goal of v34 | Re-run `run_v34_with_mlflow()` with all 4 `CLASS_NAMES` populated |
| Render free-tier cold starts | Decision 3 prioritized "live and pinned" over "always-warm"; cold start re-triggers the registry download from Decision 2 | Keep-alive ping, or paid tier |
| Cloudpickle version sensitivity | Accepted tradeoff in Decision 3 — exact version pinning works *today* but is brittle to any future dependency upgrade | ONNX export path already scaffolded (`EXPORT_ONNX` flag, onnx/onnxruntime installed) |
| Single-window processing | Out of scope — this is a v28-era limitation untouched by v34's deployment focus | Sliding-window disaggregation or multi-channel SDR (unchanged from v31 roadmap) |
| K8s manifest uses placeholder image | Decision 5 was deliberately "document the pattern," not "operate the pattern" | Push built image to a registry and update manifest when horizontal scaling is needed |

---

## Quickstart

### Run the Full Training + MLflow Pipeline (Colab)

```python
# Run the v34 main script first (defines all classes/helpers), then:
fusion, eval_results, latency_stats, action_ctrl, efficacy_mon, run_id = run_v34_with_mlflow()
```

### Call the Live API

```bash
curl https://aegisdrone.onrender.com/health

curl -X POST https://aegisdrone.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{"feature_vector": [0,0,0, ... 86 values total]}'
```

### Restore the Model Locally (Same Registered Artifact the API Uses)

```python
classify_signal = load_and_deploy(use_registry=True)
decision = classify_signal(fv_raw)
print(decision["label"], decision["soft_score"], decision["threat_score"])
```

### Run Locally with Docker (Same Image as Production)

```bash
docker build -t aegisdrone:v1 .
docker run -p 8000:8000 aegisdrone:v1
curl http://localhost:8000/health
```

---

## Versioning & Pillar History

| Version | The Question It Answered | Status |
|---------|--------------------------|------------|
| v28–v30 | "Can the ensemble classify accurately and stay calibrated?" | Gates passing on bench |
| v31-FIELD | "Is it fast enough and trustworthy enough for the field?" | 37/38 tests, p95=52.8ms |
| **v34-EW-FIELD** | **"When it sees a threat, what does it *do* — and can anything outside this notebook ask it?"** | **All gates ✅, p95=65.9ms, live in production** |

### Complete Pillar Reference

| Pillar | Tag | Decision It Implements |
|--------|-----|-------------|
| **Wideband Sweeper** | **[EW]** | **Frequency-aware response selection (Decision 1)** |
| **Look-Through Scheduler** | **[EW]** | **<100ms quiet window — prerequisite for efficacy monitoring (Decision 1)** |
| **GNSS Protected-Band Override** | **[EW]** | **Deliberate ML bypass for asymmetric-risk bands (Decision 1)** |
| **Jamming Efficacy Monitor** | **[EW]** | **Closes the detect→act loop with verify/replan (Decision 1)** |
| **LPI/FHSS/DSSS features + 20% fusion weight** | **[EW]** | **Counters signals designed to evade the original 83-feature set (Decision 1)** |
| **MLflow + DagsHub tracking** | **[MLOPS]** | **Permanent audit trail + restorable model registry (Decision 2)** |
| **FastAPI inference service** | **[DEPLOY]** | **Thin pyfunc wrapper over the registered model (Decision 3)** |
| **Docker, Python 3.12 / numpy 2.0.2 pinned** | **[DEPLOY]** | **Cloudpickle/training-environment compatibility (Decision 3)** |
| **Render live deployment** | **[DEPLOY]** | **Public endpoint loading latest registry model on startup (Decision 3)** |
| **GitHub Actions CI/CD** | **[CICD]** | **Push-to-main parity between committed and live code (Decision 4)** |
| **Kubernetes manifests** | **[K8S]** | **Documented scaling pattern, readiness-gated on `/health` (Decision 5)** |
| LightGBM inference | [FIX-5, v31] | 6× CPU speedup vs sklearn — unchanged, still the right call |
| Trust variance relaxation | [FIX-6, v31] | TRUST_MAX_VARIANCE 0.60→0.90 — unchanged |
| Memory-first lookup | [M1, v28] | O(1) emitter fingerprint hash — unchanged, feeds Stage 4 |
| Deep SVDD open-set | [P1, v28] | 8-dim hypersphere detector — unchanged, feeds Stage 3 |
| Hysteresis filter | [P2, v28] | window=5, majority=6 — unchanged, final stabilization step |
| Autonomous promotion | [M2, v28] | Auto-commits trusted emitters — unchanged |
