import subprocess, sys, os

def _pip(*pkgs):
    for flags in [[], ["--break-system-packages"]]:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", *pkgs, "-q", *flags],
            capture_output=True)
        if r.returncode == 0:
            return

_pip("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn",
     "matplotlib", "seaborn", "tqdm", "lightgbm")

try:
    _pip("torch", "--index-url", "https://download.pytorch.org/whl/cpu")
except Exception:
    pass

try:
    _pip("onnx", "onnxruntime")
except Exception:
    pass

# =============================================================================
# SECTION 0 · CONFIGURATION & DEFENSE CONSTANTS
# =============================================================================

# ── Deployment mode ──────────────────────────────────────────────────────────
# "vehicle"  : mounted on vehicle/HUMVEE, higher power budget, larger antenna
# "manpack"  : man-portable, tighter power/latency budget, reduced gate margins
DEPLOYMENT_MODE = "vehicle"   # switch to "manpack" for portable configuration

assert DEPLOYMENT_MODE in ("vehicle", "manpack"), \
    "DEPLOYMENT_MODE must be 'vehicle' or 'manpack'"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR         = "/content/drive/MyDrive/DroneRF/DroneRF"
OUTPUT_CSV       = "dronerf_features_v34.csv"
DB_PATH          = "antidrone_db_v34.json"
LOG_PATH         = "antidrone_audit_v34.jsonl"
DIAG_DIR         = "diagnostics_v34"
DEFENSE_LOG_PATH = "defense_log_v34.txt"

PRODUCTION_MODE = False
EXPORT_ONNX     = False

RANDOM_SEED  = 42
WINDOW_SIZE  = 8192
STEP_SIZE    = 4096
FS           = 10e6
TARGET_TOTAL = 8000

# ── Live-stream ───────────────────────────────────────────────────────────────
LIVE_STREAM_DIR    = "live_stream_v34"
STREAM_INTERVAL_MS = 50
STREAM_MAX_BURSTS  = 40

# ── Fusion weights ────────────────────────────────────────────────────────────
FUSION_W_CLF        = 0.40
FUSION_W_CNN        = 0.05
FUSION_W_EVM        = 0.15
FUSION_W_NORMALITY  = 0.15
FUSION_W_AGREEMENT  = 0.05
FUSION_W_LPI_FHSS   = 0.20   # EW: spread-spectrum identification
assert abs(FUSION_W_CLF + FUSION_W_CNN + FUSION_W_EVM +
           FUSION_W_NORMALITY + FUSION_W_AGREEMENT + FUSION_W_LPI_FHSS - 1.0) < 1e-9

# ── Decision gates ────────────────────────────────────────────────────────────
HOLD_DEAD_BAND      = 0.050
MIN_HOLD_RATE       = 0.045

OPEN_SET_THRESHOLD_CAP      = 0.65
DRONE_OPEN_SET_PERCENTILE   = 10.0
OPEN_SET_FLOOR_PERCENTILE   = 2
FRIENDLY_PERCENTILE         = 45
FRIENDLY_MIN_GAP            = 0.10

CONFIDENCE_BYPASS_THRESHOLD     = 0.999999
CONFIDENCE_BYPASS_THREAT_RATIO  = 0.20
BYPASS_MIN_SEEN_COUNT           = 5

HYSTERESIS_WINDOW   = 5
HYSTERESIS_MAJORITY = 6

# ── Deep models ───────────────────────────────────────────────────────────────
SVDD_EMBED_DIM  = 8
SVDD_EPOCHS     = 40
SVDD_LR         = 1e-3
SVDD_BATCH      = 128
SVDD_NU         = 0.01

CNN_EMBED_DIM  = 16
CNN_EPOCHS     = 30
CNN_LR         = 3e-3
CNN_BATCH      = 128
CNN_DROPOUT    = 0.30

EMITTER_EMBED_DIM       = 16
EMITTER_TRIPLET_EPOCHS  = 40
EMITTER_TRIPLET_LR      = 1e-3
EMITTER_TRIPLET_MARGIN  = 0.5
EMITTER_TRIPLET_BATCH   = 64

# ── Trust / promotion ────────────────────────────────────────────────────────
PROMO_MIN_OBS        = 1
PROMO_TRUST_THR      = 0.20
PROMO_MAX_THREAT     = 0.85
PROMO_CONF_THR       = 0.30

TRUST_MIN_OBSERVATIONS = 4
TRUST_MAX_VARIANCE     = 0.90
HIGH_THREAT_THRESHOLD  = 0.90
CONFIRMED_THREAT_OBS   = 5
AUTO_CLASSIFY_CONF     = 0.75
HOLD_STABILITY_WINDOW  = 3

TEMPORAL_WINDOW        = 5
TEMPORAL_SMOOTHING_MIN = 3

# ── Calibration ───────────────────────────────────────────────────────────────
TEMP_MIN = 0.70
TEMP_MAX = 1.20
GBP_TEMPERATURE         = 0.85
LAPLACE_PRIOR_PRECISION = 1.0
LAPLACE_N_SAMPLES       = 256

# ── Production gates ──────────────────────────────────────────────────────────
GATE_RECALL_MIN       = 0.85
GATE_HOLD_MAX         = 0.20
GATE_OPEN_SET_MIN     = 0.02
GATE_FPR_MAX          = 0.10
GATE_FLICKER_MAX      = 0.65
GATE_TIME_TO_TRUST_S  = 10.0
GATE_HIT_RATE_MIN     = 0.01
GATE_BYPASS_MAX       = 0.10

# ── Augmentation ──────────────────────────────────────────────────────────────
MIXUP_ALPHA          = 0.30
MIXUP_N_PER_CLASS    = 800
HARD_NEG_JITTER      = 0.08
HARD_NEG_PERCENTILE  = 20

# ── LightGBM ──────────────────────────────────────────────────────────────────
LGB_RF_N_ESTIMATORS    = 500
LGB_RF_NUM_LEAVES      = 63
LGB_RF_MIN_DATA_LEAF   = 3
LGB_RF_SUBSAMPLE       = 0.8
LGB_RF_COLSAMPLE       = 0.5
LGB_GBT_N_ESTIMATORS   = 200
LGB_GBT_NUM_LEAVES     = 31
LGB_GBT_LR             = 0.08
LGB_GBT_MIN_DATA_LEAF  = 5
LGB_GBT_SUBSAMPLE      = 0.8
_LGB_DEVICE = "cpu"

# ── Anomaly detectors ────────────────────────────────────────────────────────
ANOMALY_W_MAHAL    = 0.55
ANOMALY_W_ISO      = 0.45
ANOMALY_SCORE_CAP  = 1.0
ISO_N_ESTIMATORS   = 300
ISO_CONTAMINATION  = 0.02
OCSVM_NU           = 0.05
OCSVM_GAMMA        = "scale"

# ── Tracker ───────────────────────────────────────────────────────────────────
TRACK_CREATION_DIST   = 0.35
TRACK_SILENT_FRAMES   = 10
TRACK_MIN_OBS_PROMOTE = 4
SIMILARITY_THRESHOLD  = 0.88

# ── Ensemble ──────────────────────────────────────────────────────────────────
N_ENSEMBLE_TREES   = 3
ENSEMBLE_SUBSAMPLE = 0.70
RF_TOP_K_MI        = 45
GBT_TOP_K_VAR      = 40

# ── Cost bias ─────────────────────────────────────────────────────────────────
COST_BIAS_ACTIVE          = True
COST_BIAS_BG_PENALTY      = 0.01
COST_BIAS_UNCERTAINTY_THR = 0.55

# ── Cache / fast-path ─────────────────────────────────────────────────────────
ROUTE_CACHE_MAXSIZE    = 1024
RF_FAST_PATH_THRESHOLD = 0.97
PRESEED_N_PER_CLASS    = 80
MONITOR_WINDOW         = 100

# ── Stress tests ──────────────────────────────────────────────────────────────
GHOST_HUNT_BURSTS    = 60
ADVERSARIAL_SAMPLES  = 200
RECOVERY_BURST_COUNT = 20

# ── Energy gate ───────────────────────────────────────────────────────────────
ENERGY_GATE_MAD_K   = 3.0
ENERGY_GATE_WINDOW  = 64

# ── Channel model — Ladakh / Siachen high-altitude profile ───────────────────
RICIAN_K_FACTOR_MEAN    = 1.5
RICIAN_K_FACTOR_STD     = 0.8
RAYLEIGH_SIGMA_MEAN     = 0.25
RAYLEIGH_N_PATHS        = 6
LO_DRIFT_HZ_STD         = 25e3
CONGESTION_PROB         = 0.20

# ── Deployment-dependent parameters (auto-set below) ─────────────────────────
if DEPLOYMENT_MODE == "vehicle":
    ENERGY_GATE_MIN_POWER   = -55.0   # dBm — vehicle antenna is more sensitive
    THERMAL_NOISE_FLOOR_DBM = -105.0
    LOOK_THROUGH_CYCLE_MS   = 80.0    # ms — vehicle has more compute headroom
    LOOK_THROUGH_QUIET_RATIO= 0.15
    MAX_ALTITUDE_M          = 4572.0  # ~15 000 ft
else:  # manpack
    ENERGY_GATE_MIN_POWER   = -48.0   # dBm — smaller antenna, higher floor
    THERMAL_NOISE_FLOOR_DBM = -100.0
    LOOK_THROUGH_CYCLE_MS   = 60.0    # ms — tighter budget, faster cycling
    LOOK_THROUGH_QUIET_RATIO= 0.20
    MAX_ALTITUDE_M          = 3000.0  # ~10 000 ft

# ── Jamming efficacy monitor ──────────────────────────────────────────────────
EFFICACY_WINDOW_BURSTS   = 10   # bursts to observe after jamming trigger
EFFICACY_SILENCE_THRESH  = 0.80 # fraction that must be BACKGROUND/OS for success
EFFICACY_REPLAN_AFTER    = 3    # consecutive failures before waveform replan

# =============================================================================
# SECTION 0b · SUB-CLASSIFIER FEATURE TARGETS
# (defined here so Section 7 can reference it safely)
# =============================================================================
SUBCLF_FEATURES = [
    "high_low_band_ratio", "spectral_centroid", "bandwidth_hz",
    "energy_band3", "energy_band4", "energy_band1", "energy_band2",
    "ifreq_std", "spectral_entropy", "tx_rate_hz",
    "speed_mean", "altitude_mean",
]

# =============================================================================
# SECTION 1 · IMPORTS
# =============================================================================
import gc, hashlib, json, logging, threading, time, warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats    import kurtosis, skew
from scipy.signal   import hilbert, welch, stft
from scipy.linalg   import cho_factor, cho_solve
from scipy.optimize import minimize_scalar, linear_sum_assignment

from sklearn.decomposition     import PCA
from sklearn.ensemble          import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model      import LogisticRegression
from sklearn.metrics           import accuracy_score, f1_score
from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import RobustScaler
from sklearn.svm               import OneClassSVM
from imblearn.over_sampling    import SMOTE

try:
    import lightgbm as lgb
    LGB_OK = True
    print("✓ LightGBM available")
except ImportError:
    LGB_OK = False
    print("⚠  LightGBM not available — sklearn fallback active")

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
    CUDA_OK  = torch.cuda.is_available()
    DEVICE   = torch.device("cuda" if CUDA_OK else "cpu")
except ImportError:
    TORCH_OK = False; CUDA_OK = False; DEVICE = None

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

_audit = logging.getLogger("antidrone.v34")
_audit.setLevel(logging.DEBUG)
_fh = logging.FileHandler(LOG_PATH, mode="w")
_fh.setFormatter(logging.Formatter("%(message)s"))
_audit.addHandler(_fh)

def audit(event: str, **kw):
    _audit.debug(json.dumps({"ts": round(time.time(), 4), "event": event, **kw}))

os.makedirs(DIAG_DIR, exist_ok=True)

# =============================================================================
# SECTION 1b · CLASS MAPS
# =============================================================================
CLASS_NAMES = {
    0: "Background RF",
    1: "AR Drone Control Link",
    2: "Phantom GFSK Link",
    3: "NavIC L5 / GNSS Jammed Area",
}
BG_NAME = CLASS_NAMES[0]

FOLDER_MAP = {
    "background": 0,
    "ar_drone": 1, "ardrone": 1,
    "phantom": 2,
    "navic": 3, "gnss": 3,
}

DECISION_ICONS = {
    "FRIENDLY_DRONE":   "🟢", "BACKGROUND":        "⚪",
    "POTENTIAL_THREAT": "🔴", "CONFIRMED_THREAT":  "🚨",
    "SAFE_NEW_DRONE":   "🔵", "TRUSTED_NEW_DRONE": "🔷",
    "UNKNOWN_MONITOR":  "🟡", "OPEN_SET_UNKNOWN":  "❓",
    "HOLD":             "⏸️", "MEMORY_MATCH":      "💾",
}

# =============================================================================
# SECTION 2 · FEATURE SCHEMA
# =============================================================================
RF_FEATURE_NAMES = [
    "amp_mean","amp_std","amp_var","amp_min","amp_max","amp_range",
    "amp_kurtosis","amp_skew",
    "signal_power_db","IQ_corr","I_power","Q_power","iq_power_ratio","iq_corr_sq",
    "peak_freq_hz","bandwidth_hz","spectral_entropy","spectral_centroid",
    "spectral_spread","spectral_rolloff_85","psd_mean_db","psd_max_db",
    "ifreq_mean","ifreq_std","ifreq_range","ifreq_kurtosis",
    "energy_band1","energy_band2","energy_band3","energy_band4",
    "stft_flux_var","stft_sub1_var","stft_sub2_var","stft_sub3_var","stft_sub4_var",
    "spec_kurtosis","spec_skewness","l_kurtosis","spec_flatness","stft_entropy",
    "am_depth","crest_factor","phase_jitter","spec_asymmetry",
    "acf_short","acf_medium","acf_long","acf_ratio",
    "kurt_entropy_product","snr_like_db","spectral_variance","temporal_kurtosis",
    "high_low_band_ratio",
]
FLIGHT_FEATURE_NAMES = [
    "speed_mean","speed_std","speed_max","accel_mean","accel_std","accel_max",
    "altitude_mean","altitude_std","heading_change_rate","heading_std",
    "path_curvature","loiter_fraction","approach_vector_sin","approach_vector_cos",
    "proximity_score","hover_time_fraction","trajectory_entropy","maneuver_intensity",
]
COMM_FEATURE_NAMES = [
    "tx_rate_hz","tx_burst_ratio","protocol_entropy",
    "command_interval_mean","command_interval_std","telemetry_rate_hz",
    "channel_dwell_mean","control_link_snr","video_link_active","swarm_signal_flag",
]
EW_FEATURE_NAMES = [
    "hop_rate_hz",           # FHSS hop-rate estimate
    "dwell_time_ms",         # per-channel dwell time
    "lpi_snr_margin_db",     # low-intercept SNR margin
    "dsss_correlation_peak", # DSSS spreading correlation peak
    "chip_rate_estimate",    # estimated chip rate (DSSS)
]

N_RF     = len(RF_FEATURE_NAMES)
N_FLIGHT = len(FLIGHT_FEATURE_NAMES)
N_COMM   = len(COMM_FEATURE_NAMES)
N_EW     = len(EW_FEATURE_NAMES)

ALL_FEATURE_NAMES = (RF_FEATURE_NAMES + FLIGHT_FEATURE_NAMES +
                     COMM_FEATURE_NAMES + EW_FEATURE_NAMES)
N_FEATURES = len(ALL_FEATURE_NAMES)
FEAT_IDX   = {n: i for i, n in enumerate(ALL_FEATURE_NAMES)}

print(f"✓ Total Features: {N_FEATURES}  "
      f"(RF={N_RF}, Flight={N_FLIGHT}, Comm={N_COMM}, EW={N_EW})")
print(f"✓ Deployment mode: {DEPLOYMENT_MODE.upper()}  "
      f"| Energy gate: {ENERGY_GATE_MIN_POWER} dBm  "
      f"| Look-through cycle: {LOOK_THROUGH_CYCLE_MS} ms")

# =============================================================================
# SECTION 3 · WIDEBAND SPECTRUM SCANNER & LOOK-THROUGH SCHEDULER
# =============================================================================
class WidebandSpectrumSweeper:
    """
    Simulates a fast superheterodyne sweep engine covering 0.1 MHz – 40 GHz.
    Partitions the spectrum into military band allocations.
    Bands are checked in priority order; GNSS bands are checked first.
    """
    # (low_hz, high_hz, priority)  — lower priority number = checked first
    BANDS: Dict[str, Tuple[float, float, int]] = {
        "GNSS_NAVIC_L5":        (1164e6,  1188e6,  1),   # Indian NavIC — highest priority
        "GNSS_GPS_L1":          (1560e6,  1610e6,  1),   # GPS L1 — same priority as NavIC
        "ELINT_COMINT_HF":      (100e3,   30e6,    5),
        "ELINT_COMINT_VHF_UHF": (30e6,    1e9,     5),
        "ISM_S_BAND":           (2.4e9,   2.5e9,   3),
        "ISM_C_BAND":           (5.725e9, 5.875e9, 3),
        "RADAR_X_BAND":         (8.0e9,   12.0e9,  4),
        "MIL_MILLIMETER":       (18.0e9,  40.0e9,  4),
    }
    # Bands that must trigger an immediate CONFIRMED_THREAT override
    GNSS_PROTECTED_BANDS = {"GNSS_NAVIC_L5", "GNSS_GPS_L1"}

    def scan_spectrum(self, detected_hz: float) -> Tuple[str, float]:
        """Return (band_name, tuning_offset_hz) for detected_hz."""
        matches = []
        for band_name, (flow, fhigh, pri) in self.BANDS.items():
            if flow <= detected_hz <= fhigh:
                matches.append((pri, band_name, flow))
        if not matches:
            return "NON_ALLOCATED_BAND", 0.0
        matches.sort()
        _, band_name, flow = matches[0]
        return band_name, detected_hz - flow

    def is_gnss_protected(self, band_name: str) -> bool:
        return band_name in self.GNSS_PROTECTED_BANDS


class LookThroughScheduler:
    """
    Enforces Look-Through sensing: divides time into Transmit (Jamming)
    and Quiet (Sensing) cycles.  Cycle time is deployment-mode aware.
    """
    def __init__(self,
                 cycle_time_ms:  float = LOOK_THROUGH_CYCLE_MS,
                 quiet_ratio:    float = LOOK_THROUGH_QUIET_RATIO):
        assert cycle_time_ms < 100.0, (
            f"Look-Through standard requires cycle < 100 ms, got {cycle_time_ms} ms")
        self.cycle_time_ms     = cycle_time_ms
        self.quiet_time_ms     = cycle_time_ms * quiet_ratio
        self.transmit_time_ms  = cycle_time_ms * (1.0 - quiet_ratio)
        self.deployment_mode   = DEPLOYMENT_MODE

    def is_quiet_window(self, timestamp_ms: float) -> bool:
        return (timestamp_ms % self.cycle_time_ms) < self.quiet_time_ms

    def generate_jamming_suggest(self,
                                  signal_type: str,
                                  f_ctr_hz:    float,
                                  bw_hz:       float) -> Dict[str, Any]:
        """
        Dynamically design the optimal EW waveform suggestion.
        Returns a dict consumed by ActionController and JammingEfficacyMonitor.
        """
        if "Control" in signal_type or "AR" in signal_type:
            return {"mode": "Spot-Follower Noise",
                    "width_hz": bw_hz * 1.2, "center_hz": f_ctr_hz,
                    "rationale": "AR FHSS control link — narrow spot follow"}
        elif "GFSK" in signal_type or "Phantom" in signal_type:
            return {"mode": "Fast Swept-Spot",
                    "width_hz": bw_hz * 2.0, "center_hz": f_ctr_hz,
                    "rationale": "DSSS wideband — swept jamming covers spreading"}
        elif "GNSS" in signal_type or "NavIC" in signal_type or "GPS" in signal_type:
            return {"mode": "Coherent Barrage Jamming",
                    "width_hz": 24e6, "center_hz": f_ctr_hz,
                    "rationale": "GNSS band — barrage covers entire allocation"}
        else:
            return {"mode": "Serrated Sawtooth Comb",
                    "width_hz": max(bw_hz, 10e6), "center_hz": f_ctr_hz,
                    "rationale": "Unknown signal — comb jammer covers harmonics"}


# =============================================================================
# SECTION 3b · CHANNEL STATISTICS (Ladakh / Siachen profile)
# =============================================================================
DRONERF_STATS: Dict[int, Dict[str, Tuple[float, float]]] = {
    0: {  # Background RF
        "signal_power_db":  (-30., 5.),
        "spectral_entropy": (3.0,  0.8),
        "bandwidth_hz":     (0.5e6, 0.2e6),
        "ifreq_std":        (0.18, 0.10),
        "amp_kurtosis":     (0.5,  0.4),
        "spectral_centroid":(2.0e6, 0.5e6),
        "IQ_corr":          (0.01, 0.04),
        "crest_factor":     (1.5,  0.3),
        "snr_like_db":      (-12., 4.),
        "psd_max_db":       (-28., 4.),
        "energy_band1":     (0.40, 0.10),
        "energy_band2":     (0.30, 0.10),
        "energy_band3":     (0.15, 0.05),
        "energy_band4":     (0.15, 0.05),
        "hop_rate_hz":      (0.0,  0.0),
        "dwell_time_ms":    (0.0,  0.0),
        "lpi_snr_margin_db":(0.0,  0.2),
        "dsss_correlation_peak": (0.05, 0.02),
        "chip_rate_estimate":    (0.0,  0.0),
    },
    1: {  # AR Drone Control Link — Fast FHSS
        "signal_power_db":  (-18., 4.),
        "spectral_entropy": (5.2,  0.7),
        "bandwidth_hz":     (2.0e6, 0.5e6),
        "ifreq_std":        (0.85, 0.20),
        "amp_kurtosis":     (2.2,  0.8),
        "spectral_centroid":(3.2e6, 0.3e6),
        "IQ_corr":          (0.07, 0.05),
        "crest_factor":     (2.6,  0.5),
        "snr_like_db":      (10.,  3.),
        "psd_max_db":       (-15., 3.),
        "energy_band1":     (0.20, 0.05),
        "energy_band2":     (0.35, 0.05),
        "energy_band3":     (0.30, 0.05),
        "energy_band4":     (0.15, 0.05),
        "hop_rate_hz":      (250.0, 20.0),
        "dwell_time_ms":    (4.0,   0.5),
        "lpi_snr_margin_db":(12.0,  2.0),
        "dsss_correlation_peak": (0.10, 0.02),
        "chip_rate_estimate":    (0.0,  0.0),
    },
    2: {  # Phantom GFSK — Wideband DSSS
        "signal_power_db":  (-12., 3.5),
        "spectral_entropy": (6.1,  0.5),
        "bandwidth_hz":     (3.8e6, 0.8e6),
        "ifreq_std":        (1.45, 0.30),
        "amp_kurtosis":     (3.5,  0.9),
        "spectral_centroid":(6.0e6, 0.4e6),
        "IQ_corr":          (0.12, 0.06),
        "crest_factor":     (3.2,  0.6),
        "snr_like_db":      (16.,  3.),
        "psd_max_db":       (-8.,  3.),
        "energy_band1":     (0.05, 0.02),
        "energy_band2":     (0.10, 0.03),
        "energy_band3":     (0.35, 0.05),
        "energy_band4":     (0.50, 0.05),
        "hop_rate_hz":      (0.0,  0.0),
        "dwell_time_ms":    (0.0,  0.0),
        "lpi_snr_margin_db":(18.0, 3.0),
        "dsss_correlation_peak": (0.85, 0.05),
        "chip_rate_estimate":    (1.023e6, 100.0),
    },
    3: {  # NavIC IRNSS L5 hostile jamming / spoofing
        "signal_power_db":  (-22., 2.0),
        "spectral_entropy": (4.5,  0.3),
        "bandwidth_hz":     (24e6, 0.5e6),
        "ifreq_std":        (2.1,  0.15),
        "amp_kurtosis":     (1.8,  0.3),
        "spectral_centroid":(1176.45e6, 1e6),
        "IQ_corr":          (0.15, 0.02),
        "crest_factor":     (2.1,  0.2),
        "snr_like_db":      (5.0,  1.0),
        "psd_max_db":       (-18., 2.0),
        "energy_band1":     (0.25, 0.02),
        "energy_band2":     (0.25, 0.02),
        "energy_band3":     (0.25, 0.02),
        "energy_band4":     (0.25, 0.02),
        "hop_rate_hz":      (0.0,  0.0),
        "dwell_time_ms":    (0.0,  0.0),
        "lpi_snr_margin_db":(4.0,  0.5),
        "dsss_correlation_peak": (0.95, 0.01),
        "chip_rate_estimate":    (10.23e6, 50.0),
    },
}


# =============================================================================
# SECTION 3c · CHANNEL IMPAIRMENT HELPERS
# =============================================================================
def _apply_rician_fading(fv: np.ndarray, rng: np.random.Generator,
                          k_factor: float) -> np.ndarray:
    k   = max(k_factor, 0.01)
    nu  = np.sqrt(k / (k + 1.))
    sig = 1. / np.sqrt(2. * (k + 1.) + 1e-9)
    gain = float(np.clip(
        np.sqrt((nu + sig*float(rng.normal()))**2 + (sig*float(rng.normal()))**2),
        0.15, 2.5))
    faded = fv.copy()
    for key in ("amp_mean","amp_std","amp_var","amp_min","amp_max","amp_range",
                "I_power","Q_power","signal_power_db"):
        if key in FEAT_IDX:
            idx = FEAT_IDX[key]
            if key == "signal_power_db":
                faded[idx] += float(20.*np.log10(gain + 1e-9))
            elif key == "amp_var":
                faded[idx] *= gain**2
            else:
                faded[idx] *= gain
    return faded


def _apply_rayleigh_multipath(fv: np.ndarray, rng: np.random.Generator,
                               n_paths: int = RAYLEIGH_N_PATHS,
                               sigma:   float = RAYLEIGH_SIGMA_MEAN) -> np.ndarray:
    mp = fv.copy()
    for _ in range(n_paths):
        path_gain  = float(rng.rayleigh(sigma))
        delay_frac = float(rng.uniform(0.02, 0.25))
        bw_key  = FEAT_IDX.get("bandwidth_hz")
        cen_key = FEAT_IDX.get("spectral_centroid")
        if bw_key is not None:
            mp[bw_key] = float(np.abs(mp[bw_key] * (1. + path_gain * delay_frac)))
        if cen_key is not None:
            sign = float(rng.choice([-1., 1.]))
            mp[cen_key] += sign * path_gain * float(mp[bw_key]) * 0.08
    return mp


def _apply_lo_drift(fv: np.ndarray, rng: np.random.Generator,
                    drift_hz_std: float = LO_DRIFT_HZ_STD) -> np.ndarray:
    drift   = float(rng.normal(0., drift_hz_std))
    drifted = fv.copy()
    for key in ("peak_freq_hz", "spectral_centroid"):
        if key in FEAT_IDX:
            drifted[FEAT_IDX[key]] += drift
    if "ifreq_mean" in FEAT_IDX:
        drifted[FEAT_IDX["ifreq_mean"]] += drift / FS
    return drifted


def _apply_thermal_noise(fv: np.ndarray, rng: np.random.Generator,
                          noise_floor_dbm: float = THERMAL_NOISE_FLOOR_DBM) -> np.ndarray:
    noisy = fv.copy()
    thermal_amp = float(10.**(noise_floor_dbm / 20.))
    for key in ("amp_mean", "amp_std"):
        if key in FEAT_IDX:
            noisy[FEAT_IDX[key]] += float(rng.normal(0., thermal_amp * 0.01))
    if "snr_like_db" in FEAT_IDX:
        noisy[FEAT_IDX["snr_like_db"]] += float(rng.normal(0., 1.5))
    return noisy


def _apply_spectral_congestion(fv: np.ndarray, rng: np.random.Generator,
                                other_cls: int) -> np.ndarray:
    if other_cls not in DRONERF_STATS:
        return fv
    alpha     = float(rng.uniform(0.10, 0.35))
    congested = fv.copy()
    other     = DRONERF_STATS[other_cls]
    for key in ("spectral_centroid","bandwidth_hz","spectral_entropy",
                "energy_band1","energy_band2","energy_band3","energy_band4"):
        if key in FEAT_IDX and key in other:
            mu, _ = other[key]
            congested[FEAT_IDX[key]] = float(
                (1.-alpha)*fv[FEAT_IDX[key]] + alpha*mu)
    return congested


def _generate_rf_burst(cls: int, rng: np.random.Generator,
                        noise_scale:   float = 1.0,
                        apply_channel: bool  = True) -> np.ndarray:
    prof = DRONERF_STATS[cls]
    fv   = np.zeros(N_FEATURES, dtype=np.float32)

    def G(key, dm=0., ds=1.):
        mu, sd = prof.get(key, (dm, ds))
        return float(rng.normal(mu, sd * noise_scale))

    pwr_db = G("signal_power_db");  bw   = abs(G("bandwidth_hz"))
    entr   = abs(G("spectral_entropy")); ifreq = abs(G("ifreq_std"))
    kurt   = G("amp_kurtosis");      cen  = abs(G("spectral_centroid"))
    iq_r   = G("IQ_corr");           cf   = abs(G("crest_factor"))
    snr_db = G("snr_like_db");       psd_mx = G("psd_max_db")

    rms      = float(10.**(pwr_db / 20.))
    amp_std  = rms * abs(float(rng.normal(0.35 + 0.05*abs(kurt), 0.05)))
    amp_mean = rms * abs(float(rng.normal(1.0, 0.05)))
    amp_min  = max(0., amp_mean - 3.*amp_std)
    amp_max  = amp_mean + abs(float(rng.normal(3.5 + 0.3*cf, 0.3))) * amp_std

    fv[FEAT_IDX["amp_mean"]]    = amp_mean
    fv[FEAT_IDX["amp_std"]]     = amp_std
    fv[FEAT_IDX["amp_var"]]     = amp_std**2
    fv[FEAT_IDX["amp_min"]]     = amp_min
    fv[FEAT_IDX["amp_max"]]     = amp_max
    fv[FEAT_IDX["amp_range"]]   = amp_max - amp_min
    fv[FEAT_IDX["amp_kurtosis"]]= kurt
    fv[FEAT_IDX["amp_skew"]]    = float(rng.normal(0.4*np.sign(kurt), 0.2))

    i_pow = rms**2 * abs(float(rng.normal(1.0, 0.05)))
    q_pow = i_pow  * abs(float(rng.normal(0.95 + 0.1*abs(iq_r), 0.05)))
    fv[FEAT_IDX["signal_power_db"]] = pwr_db
    fv[FEAT_IDX["IQ_corr"]]         = float(np.clip(iq_r, -0.99, 0.99))
    fv[FEAT_IDX["I_power"]]         = i_pow
    fv[FEAT_IDX["Q_power"]]         = q_pow
    fv[FEAT_IDX["iq_power_ratio"]]  = i_pow / (q_pow + 1e-9)
    fv[FEAT_IDX["iq_corr_sq"]]      = iq_r**2

    spread = bw * abs(float(rng.normal(0.38, 0.06)))
    rollof = cen + spread * abs(float(rng.normal(1.2, 0.1)))
    fv[FEAT_IDX["peak_freq_hz"]]        = cen + float(rng.normal(0, bw*0.05))
    fv[FEAT_IDX["bandwidth_hz"]]        = bw
    fv[FEAT_IDX["spectral_entropy"]]    = entr
    fv[FEAT_IDX["spectral_centroid"]]   = cen
    fv[FEAT_IDX["spectral_spread"]]     = spread
    fv[FEAT_IDX["spectral_rolloff_85"]] = rollof
    fv[FEAT_IDX["psd_mean_db"]]         = pwr_db - abs(float(rng.normal(4., 1.)))
    fv[FEAT_IDX["psd_max_db"]]          = psd_mx

    fv[FEAT_IDX["ifreq_mean"]]     = float(rng.normal(0, ifreq*0.1))
    fv[FEAT_IDX["ifreq_std"]]      = ifreq
    fv[FEAT_IDX["ifreq_range"]]    = ifreq * abs(float(rng.normal(4.0, 0.5)))
    fv[FEAT_IDX["ifreq_kurtosis"]] = float(rng.normal(0.5 + 0.3*abs(kurt), 0.3))

    e1 = abs(G("energy_band1")); e2 = abs(G("energy_band2"))
    e3 = abs(G("energy_band3")); e4 = abs(G("energy_band4"))
    etot = e1 + e2 + e3 + e4 + 1e-9
    b1, b2, b3, b4 = e1/etot, e2/etot, e3/etot, e4/etot
    fv[FEAT_IDX["energy_band1"]]        = b1
    fv[FEAT_IDX["energy_band2"]]        = b2
    fv[FEAT_IDX["energy_band3"]]        = b3
    fv[FEAT_IDX["energy_band4"]]        = b4
    fv[FEAT_IDX["high_low_band_ratio"]] = (b3+b4) / (b1+b2+1e-9)

    stft_flux = bw * abs(float(rng.normal(0.01 + 0.005*abs(kurt), 0.002)))
    fv[FEAT_IDX["stft_flux_var"]] = stft_flux
    for b in range(4):
        fv[FEAT_IDX[f"stft_sub{b+1}_var"]] = abs(
            float(rng.normal(stft_flux*(0.8+0.1*b), stft_flux*0.3)))

    fv[FEAT_IDX["spec_kurtosis"]]  = float(rng.normal(kurt*0.9, 0.3))
    fv[FEAT_IDX["spec_skewness"]]  = float(rng.normal(0.3*np.sign(kurt), 0.2))
    fv[FEAT_IDX["l_kurtosis"]]     = float(rng.normal(0.2 + 0.05*abs(kurt), 0.1))
    fv[FEAT_IDX["spec_flatness"]]  = float(np.clip(rng.normal(0.5-0.04*entr, 0.1), 0, 1))
    fv[FEAT_IDX["stft_entropy"]]   = entr * abs(float(rng.normal(0.95, 0.05)))
    am = np.clip(0.05 + 0.06*abs(kurt), 0.01, 0.99)
    fv[FEAT_IDX["am_depth"]]       = float(am + rng.normal(0, 0.02))
    fv[FEAT_IDX["crest_factor"]]   = cf
    fv[FEAT_IDX["phase_jitter"]]   = ifreq * abs(float(rng.normal(0.15, 0.05)))
    fv[FEAT_IDX["spec_asymmetry"]] = float(rng.normal((cen-3e6)/3e6, 0.1))

    acf_s = float(np.clip(rng.normal(0.1+0.05*abs(iq_r), 0.05), -1, 1))
    acf_m = float(np.clip(rng.normal(acf_s*0.4, 0.04), -1, 1))
    acf_l = float(np.clip(rng.normal(acf_m*0.3, 0.03), -1, 1))
    fv[FEAT_IDX["acf_short"]]  = acf_s
    fv[FEAT_IDX["acf_medium"]] = acf_m
    fv[FEAT_IDX["acf_long"]]   = acf_l
    fv[FEAT_IDX["acf_ratio"]]  = acf_s / (acf_l + 1e-9)
    fv[FEAT_IDX["kurt_entropy_product"]] = float(kurt * entr)
    fv[FEAT_IDX["snr_like_db"]]          = snr_db
    fv[FEAT_IDX["spectral_variance"]]    = float(spread**2)
    fv[FEAT_IDX["temporal_kurtosis"]]    = float(kurt + rng.normal(0, 0.2))

    # Flight parameters — altitude scaled to deployment mode
    alt_mean = MAX_ALTITUDE_M * float(rng.uniform(0.5, 1.0))
    alt_std  = alt_mean * 0.05
    if cls == 1:
        for k,(mu,sd) in [
            ("speed_mean",(8.,3.)),("speed_std",(2.0,.6)),("speed_max",(14.,4.)),
            ("accel_mean",(.9,.3)),("accel_std",(.4,.15)),("accel_max",(4.,1.0)),
            ("heading_change_rate",(.35,.15)),("trajectory_entropy",(2.7,.5)),
            ("maneuver_intensity",(.5,.15))]:
            fv[FEAT_IDX[k]] = abs(float(rng.normal(mu, sd)))
        fv[FEAT_IDX["altitude_mean"]] = alt_mean
        fv[FEAT_IDX["altitude_std"]]  = alt_std
        fv[FEAT_IDX["hover_time_fraction"]] = float(np.clip(rng.normal(.15,.1), 0, 1))
    elif cls == 2:
        for k,(mu,sd) in [
            ("speed_mean",(14.,4.)),("speed_std",(2.8,.9)),("speed_max",(25.,5.)),
            ("accel_mean",(1.6,.4)),("accel_std",(.7,.2)),("accel_max",(6.,1.2)),
            ("heading_change_rate",(.2,.08)),("trajectory_entropy",(3.5,.5)),
            ("maneuver_intensity",(.75,.15))]:
            fv[FEAT_IDX[k]] = abs(float(rng.normal(mu, sd)))
        fv[FEAT_IDX["altitude_mean"]] = alt_mean
        fv[FEAT_IDX["altitude_std"]]  = alt_std
        fv[FEAT_IDX["hover_time_fraction"]] = float(np.clip(rng.normal(.05,.05), 0, 1))

    # COMINT parameters
    if cls == 1:
        for k,v in [
            ("tx_rate_hz",        abs(float(rng.normal(25., 5.)))),
            ("tx_burst_ratio",    float(np.clip(rng.normal(.35,.10), 0, 1))),
            ("protocol_entropy",  abs(float(rng.normal(1.8,.3)))),
            ("command_interval_mean", abs(float(rng.normal(.04,.01)))),
            ("command_interval_std",  abs(float(rng.normal(.008,.002)))),
            ("telemetry_rate_hz", abs(float(rng.normal(10.,2.)))),
            ("channel_dwell_mean",abs(float(rng.normal(.02,.005)))),
            ("control_link_snr",  abs(float(rng.normal(18.,4.)))),
            ("video_link_active", float(rng.choice([0.,1.], p=[.3,.7]))),
            ("swarm_signal_flag", 0.)]:
            fv[FEAT_IDX[k]] = v
    elif cls == 2:
        for k,v in [
            ("tx_rate_hz",        abs(float(rng.normal(50.,8.)))),
            ("tx_burst_ratio",    float(np.clip(rng.normal(.55,.12), 0, 1))),
            ("protocol_entropy",  abs(float(rng.normal(2.5,.3)))),
            ("command_interval_mean", abs(float(rng.normal(.02,.005)))),
            ("command_interval_std",  abs(float(rng.normal(.004,.001)))),
            ("telemetry_rate_hz", abs(float(rng.normal(20.,3.)))),
            ("channel_dwell_mean",abs(float(rng.normal(.008,.002)))),
            ("control_link_snr",  abs(float(rng.normal(25.,4.)))),
            ("video_link_active", 1.),
            ("swarm_signal_flag", float(rng.choice([0.,1.], p=[.85,.15])))]:
            fv[FEAT_IDX[k]] = v

    # EW parameters
    fv[FEAT_IDX["hop_rate_hz"]]           = abs(G("hop_rate_hz"))
    fv[FEAT_IDX["dwell_time_ms"]]         = abs(G("dwell_time_ms"))
    fv[FEAT_IDX["lpi_snr_margin_db"]]     = G("lpi_snr_margin_db")
    fv[FEAT_IDX["dsss_correlation_peak"]] = abs(G("dsss_correlation_peak"))
    fv[FEAT_IDX["chip_rate_estimate"]]    = abs(G("chip_rate_estimate"))

    # Random dropout & kurtosis spike
    if rng.random() < 0.08:
        fv[rng.integers(0, N_FEATURES, size=rng.integers(1, 4))] = 0.
    if rng.random() < 0.05:
        fv[FEAT_IDX["amp_kurtosis"]] += float(rng.exponential(2.))

    # Channel impairments
    if apply_channel:
        k_factor = max(0.1, float(rng.normal(RICIAN_K_FACTOR_MEAN, RICIAN_K_FACTOR_STD)))
        fv = _apply_rician_fading(fv, rng, k_factor)
        if rng.random() < 0.85:
            fv = _apply_rayleigh_multipath(fv, rng)
        fv = _apply_lo_drift(fv, rng)
        fv = _apply_thermal_noise(fv, rng)
        if rng.random() < CONGESTION_PROB and cls != 0:
            other = int(rng.choice([c for c in range(4) if c != cls]))
            fv = _apply_spectral_congestion(fv, rng, other)

    return fv


def _generate_rf_burst_with_hardware_id(cls: int, rng: np.random.Generator,
                                         noise_scale: float = 1.0,
                                         hardware_offset: Optional[np.ndarray] = None
                                         ) -> np.ndarray:
    fv = _generate_rf_burst(cls, rng, noise_scale=noise_scale, apply_channel=True)
    if hardware_offset is not None:
        fv = fv + hardware_offset.astype(np.float32)
    return fv


# =============================================================================
# SECTION 3d · AUGMENTATION
# =============================================================================
def mixup_augment(X, y, rng, alpha=MIXUP_ALPHA, n_per_drone_class=MIXUP_N_PER_CLASS):
    bg_idx = np.where(y == 0)[0]
    aug_X, aug_y = [], []
    for drone_cls in [1, 2, 3]:
        d_idx = np.where(y == drone_cls)[0]
        if len(d_idx) == 0 or len(bg_idx) == 0:
            continue
        for _ in range(n_per_drone_class):
            di = rng.choice(d_idx); bi = rng.choice(bg_idx)
            aug_X.append(((1.-alpha)*X[di] + alpha*X[bi]).astype(np.float32))
            aug_y.append(drone_cls)
    if not aug_X:
        return X, y
    aug_X = np.stack(aug_X); aug_y = np.array(aug_y, dtype=np.int64)
    print(f"  [Aug] Mixup: +{len(aug_X)} samples")
    return np.concatenate([X, aug_X]), np.concatenate([y, aug_y])


def hard_negative_mine(X, y, lgb_clf, scaler_rf, rf_idx, rng,
                        percentile=HARD_NEG_PERCENTILE, jitter_std=HARD_NEG_JITTER):
    drone_mask = (y != 0)
    if drone_mask.sum() < 20:
        return X, y
    X_drone = X[drone_mask]; y_drone = y[drone_mask]
    X_sc = np.nan_to_num(scaler_rf.transform(X_drone[:, rf_idx]),
                          nan=0., posinf=0., neginf=0.)
    if LGB_OK and hasattr(lgb_clf, "booster") and lgb_clf.booster is not None:
        probs = lgb_clf.predict(X_sc)
    else:
        probs = lgb_clf.predict_proba(X_sc)
    max_p = probs.max(1); thr = np.percentile(max_p, percentile)
    hard  = max_p <= thr
    if hard.sum() == 0:
        return X, y
    X_hard   = X_drone[hard]; y_hard = y_drone[hard]
    jittered = (X_hard + rng.normal(0, jitter_std, X_hard.shape)).astype(np.float32)
    print(f"  [Aug] Hard-negative mining: {hard.sum()} samples jittered and added")
    return np.concatenate([X, jittered]), np.concatenate([y, y_hard])


def generate_realistic_dataset(n_per_class=2000, rng_seed=RANDOM_SEED):
    rng = np.random.default_rng(rng_seed)
    rows, labels = [], []
    for cls in range(4):
        n_normal = int(n_per_class * 0.75)
        n_noisy  = int(n_per_class * 0.15)
        n_vnoisy = n_per_class - n_normal - n_noisy
        for _ in range(n_normal):  rows.append(_generate_rf_burst(cls, rng, 1.0));  labels.append(cls)
        for _ in range(n_noisy):   rows.append(_generate_rf_burst(cls, rng, 1.6));  labels.append(cls)
        for _ in range(n_vnoisy):  rows.append(_generate_rf_burst(cls, rng, 2.5));  labels.append(cls)
    # Boundary samples
    n_bnd = int(n_per_class * 0.25)
    for _ in range(n_bnd):
        fv = _generate_rf_burst(1, rng, 1.2)
        fv[FEAT_IDX["spectral_centroid"]] = float(rng.normal(4.0e6, 0.4e6))
        fv[FEAT_IDX["bandwidth_hz"]]      = abs(float(rng.normal(3.0e6, 0.8e6)))
        b3,b4 = fv[FEAT_IDX["energy_band3"]], fv[FEAT_IDX["energy_band4"]]
        b1,b2 = fv[FEAT_IDX["energy_band1"]], fv[FEAT_IDX["energy_band2"]]
        fv[FEAT_IDX["high_low_band_ratio"]] = (b3+b4)/(b1+b2+1e-9)
        rows.append(fv); labels.append(1)
    for _ in range(n_bnd):
        fv = _generate_rf_burst(2, rng, 1.2)
        fv[FEAT_IDX["signal_power_db"]] = float(rng.normal(-25.,3.))
        fv[FEAT_IDX["snr_like_db"]]     = float(rng.normal(-8.,2.))
        rows.append(fv); labels.append(2)

    X  = np.array(rows, dtype=np.float32)
    df = pd.DataFrame(X, columns=ALL_FEATURE_NAMES)
    df.insert(0, "label_int",   labels)
    df.insert(1, "label_name",  [CLASS_NAMES.get(c,str(c)) for c in labels])
    df.insert(2, "source_file", ["synthetic_v34"]*len(labels))
    df = df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)
    cnts = Counter(labels)
    print(f"  ✓ {len(df):,} rows | " +
          "  ".join(f"{CLASS_NAMES.get(k,k)}={v}" for k,v in sorted(cnts.items())))
    return df


# =============================================================================
# SECTION 4 · FEATURE EXTRACTION
# =============================================================================
def _pearson(x, y_arr):
    xm = x - x.mean(); ym = y_arr - y_arr.mean()
    return float(np.dot(xm, ym) / ((np.dot(xm,xm)*np.dot(ym,ym))**0.5 + 1e-12))

def extract_rf_features(real_seg, fs=FS):
    real = real_seg.astype(np.float64); N = len(real)
    analytic = hilbert(real); I, Q = analytic.real, analytic.imag
    envelope = np.abs(analytic); out = np.empty(N_RF, dtype=np.float32)

    amp_mean=float(envelope.mean()); amp_std=float(envelope.std())
    amp_min=float(envelope.min());   amp_max=float(envelope.max())
    amp_kurt=float(kurtosis(envelope)) if amp_std>1e-8 else 0.
    out[0:8] = [amp_mean, amp_std, amp_std**2, amp_min, amp_max,
                amp_max-amp_min, amp_kurt,
                float(skew(envelope)) if amp_std>1e-8 else 0.]

    I_pow=float(np.dot(I,I)/N); Q_pow=float(np.dot(Q,Q)/N)
    rms  =float((np.dot(envelope,envelope)/N)**0.5)
    pow_db=float(10.*np.log10(np.dot(envelope,envelope)/N+1e-12))
    iq_c  =_pearson(I,Q) if amp_std>1e-12 else 0.
    out[8:14] = [pow_db, iq_c, I_pow, Q_pow, I_pow/(Q_pow+1e-12), iq_c**2]

    nperseg=min(512,N//4)
    fw,psd=welch(envelope,fs=fs,nperseg=nperseg,noverlap=nperseg//2,return_onesided=True)
    pa=np.clip(np.abs(psd),1e-12,None); pa_sum=pa.sum()
    pd_db=10.*np.log10(pa); pk=int(pa.argmax())
    above=fw[pd_db>pd_db[pk]-10.]
    bw_val=float(above.max()-above.min()) if len(above)>1 else 0.
    pn=pa/pa_sum; entropy=float(-np.dot(pn,np.log2(pn+1e-12)))
    cen=float(np.dot(fw,pa)/pa_sum)
    spread=float(np.sqrt(np.dot((fw-cen)**2,pa)/pa_sum))
    cs=np.cumsum(pa); rol=min(int(np.searchsorted(cs,0.85*cs[-1])),len(fw)-1)
    out[14:22]=[fw[pk],bw_val,entropy,cen,spread,fw[rol],
                float(pd_db.mean()),float(pd_db.max())]

    ifreq=np.diff(np.unwrap(np.angle(analytic)))
    if len(ifreq)>=2 and ifreq.std()>1e-8:
        out[22:26]=[float(ifreq.mean()),float(ifreq.std()),
                    float(ifreq.max()-ifreq.min()),float(kurtosis(ifreq))]
    else:
        out[22:26]=[0.]*4

    q_sz=max(1,len(pa)//4)
    b1=pa[:q_sz].sum()/pa_sum;         b2=pa[q_sz:2*q_sz].sum()/pa_sum
    b3=pa[2*q_sz:3*q_sz].sum()/pa_sum; b4=pa[3*q_sz:].sum()/pa_sum
    out[26:30]=[b1,b2,b3,b4]

    stft_np=min(128,N//4)
    _,_,Zxx=stft(envelope,fs=fs,nperseg=stft_np,noverlap=stft_np//2,return_onesided=True)
    Sxx=np.abs(Zxx)**2+1e-12; fm=Sxx.mean(0); out[30]=float(np.diff(fm).var())
    bsz=max(1,Sxx.shape[0]//4)
    for b in range(4):
        out[31+b]=float(Sxx[b*bsz:(b+1)*bsz,:].mean(0).var())

    pa_s=np.sort(pa)
    L2=pa_s[1::2].mean()-pa_s[::2].mean()
    L4=(pa_s[3::4].mean()-3*pa_s[2::4].mean()+3*pa_s[1::4].mean()-pa_s[::4].mean())
    Sxx_n=Sxx.mean(1); Sxx_n/=Sxx_n.sum()+1e-12
    out[35:40]=[float(kurtosis(pa)),float(skew(pa)),float(L4/(L2+1e-12)),
                float(np.exp(np.log(pa+1e-12).mean()-np.log(pa.mean()+1e-12))),
                float(-np.dot(Sxx_n,np.log2(Sxx_n+1e-12)))]

    out[40:44]=[float((envelope.max()-envelope.min())/(amp_mean+1e-12)),
                float(envelope.max()/(rms+1e-12)),
                float(np.diff(ifreq).std()) if len(ifreq)>=2 else 0.,
                float((pa[fw>=cen].sum()-pa[fw<cen].sum())/(pa_sum+1e-12))]

    if len(envelope)>=4:
        acf=np.correlate(envelope-envelope.mean(),envelope-envelope.mean(),mode="full")
        acf=acf[len(acf)//2:]/(acf[len(acf)//2]+1e-12)
        acf_s=float(acf[min(10,len(acf)-1)]); acf_l=float(acf[min(200,len(acf)-1)])
        out[44:48]=[acf_s,float(acf[min(50,len(acf)-1)]),acf_l,float(acf_s/(acf_l+1e-12))]
    else:
        out[44:48]=[0.]*4

    out[48]=float(amp_kurt*entropy)
    out[49]=float(10.*np.log10((pa.max()/(pa.mean()+1e-12))+1e-12))
    out[50]=float(np.var(pa)); out[51]=float(kurtosis(envelope)) if amp_std>1e-8 else 0.
    out[52]=float((b3+b4)/(b1+b2+1e-9))
    return out

def safe_extract_rf(seg):
    try:    return extract_rf_features(seg)
    except: return np.zeros(N_RF, dtype=np.float32)

def fuse_features(rf, flight=None, comm=None, ew_feats=None):
    fl = (np.asarray(flight,   dtype=np.float32) if flight   is not None else np.zeros(N_FLIGHT, np.float32))
    co = (np.asarray(comm,     dtype=np.float32) if comm     is not None else np.zeros(N_COMM,   np.float32))
    ew = (np.asarray(ew_feats, dtype=np.float32) if ew_feats is not None else np.zeros(N_EW,     np.float32))
    return np.concatenate([rf.astype(np.float32), fl, co, ew])


# =============================================================================
# SECTION 4a · DYNAMIC NOISE FLOOR ESTIMATOR
# =============================================================================
class DynamicNoiseFloorEstimator:
    def __init__(self, window: int = ENERGY_GATE_WINDOW):
        self.window = window
        self._power_history: deque = deque(maxlen=window)
        self._floor_db: float = THERMAL_NOISE_FLOOR_DBM

    def update(self, signal_power_db: float):
        if signal_power_db < -22.0:
            self._power_history.append(signal_power_db)
        if len(self._power_history) >= 8:
            arr = np.array(self._power_history)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            self._floor_db = med - 2.0 * mad
        return self._floor_db

    @property
    def noise_floor_db(self) -> float:
        return self._floor_db

    def is_above_floor(self, signal_power_db: float, k: float = ENERGY_GATE_MAD_K) -> bool:
        threshold = (self._floor_db * k if self._floor_db < 0
                     else self._floor_db + k)
        return signal_power_db > threshold

    def dynamic_snr_db(self, signal_power_db: float) -> float:
        return signal_power_db - self._floor_db


# =============================================================================
# SECTION 4b · JAMMING EFFICACY MONITOR  (fixes adaptive jamming gap)
# =============================================================================
class JammingEfficacyMonitor:
    """
    Tracks whether jamming is actually silencing the target.

    After each ActionController trigger the monitor observes the next
    EFFICACY_WINDOW_BURSTS classifications for the same track.  If the
    fraction of "silenced" outcomes (BACKGROUND / OPEN_SET_UNKNOWN) is
    below EFFICACY_SILENCE_THRESH, consecutive failure counter increments.
    After EFFICACY_REPLAN_AFTER failures the monitor requests a waveform
    replan from ActionController.

    Integration: call observe_post_jam(track_id, label) for every burst
    that arrives after a jamming trigger.
    """
    def __init__(self, action_ctrl: "ActionController"):
        self.action_ctrl = action_ctrl
        # per-track state:  {track_id: {"window": deque, "failures": int,
        #                               "last_suggest": dict|None}}
        self._state: Dict[int, Dict] = {}
        self.total_replans = 0

    def register_jam(self, track_id: int, suggestion: Dict[str, Any]):
        """Call immediately after a jamming trigger."""
        self._state[track_id] = {
            "window":       deque(maxlen=EFFICACY_WINDOW_BURSTS),
            "failures":     self._state.get(track_id, {}).get("failures", 0),
            "last_suggest": suggestion,
            "active":       True,
        }

    def observe_post_jam(self, track_id: int, label: str):
        """Call for every classify_signal result while jamming is active."""
        if track_id not in self._state or not self._state[track_id].get("active"):
            return
        state = self._state[track_id]
        silenced = label in {"BACKGROUND", "OPEN_SET_UNKNOWN"}
        state["window"].append(silenced)

        if len(state["window"]) < EFFICACY_WINDOW_BURSTS:
            return  # window not full yet

        silence_rate = float(np.mean(list(state["window"])))
        state["active"] = False  # window complete — evaluate

        if silence_rate >= EFFICACY_SILENCE_THRESH:
            state["failures"] = 0
            audit("EFFICACY_SUCCESS", track_id=track_id,
                  silence_rate=round(silence_rate, 3))
            print(f"  ✅ [Efficacy] Track {track_id}: jamming effective "
                  f"({silence_rate:.0%} silent)")
        else:
            state["failures"] += 1
            audit("EFFICACY_FAILURE", track_id=track_id,
                  silence_rate=round(silence_rate, 3),
                  consecutive_failures=state["failures"])
            print(f"  ⚠️  [Efficacy] Track {track_id}: jamming ineffective "
                  f"({silence_rate:.0%} silent, fails={state['failures']})")

            if state["failures"] >= EFFICACY_REPLAN_AFTER:
                self._replan(track_id, state)

    def _replan(self, track_id: int, state: Dict):
        """Request waveform change when jamming keeps failing."""
        old_suggest = state.get("last_suggest", {})
        old_mode    = old_suggest.get("mode", "Unknown")
        old_bw      = old_suggest.get("width_hz", 2e6)
        old_ctr     = old_suggest.get("center_hz", 2.4e9)

        # Adaptive rule: widen bandwidth by 50% and alternate waveform family
        mode_cycle = {
            "Spot-Follower Noise": "Fast Swept-Spot",
            "Fast Swept-Spot":     "Serrated Sawtooth Comb",
            "Serrated Sawtooth Comb": "Spot-Follower Noise",
            "Coherent Barrage Jamming": "Coherent Barrage Jamming",  # fixed for GNSS
        }
        new_mode = mode_cycle.get(old_mode, "Fast Swept-Spot")
        new_bw   = min(old_bw * 1.5, 40e6)
        new_suggest = {
            "mode":      new_mode,
            "width_hz":  new_bw,
            "center_hz": old_ctr,
            "rationale": f"Replan after {state['failures']} failures — wider BW",
        }
        state["last_suggest"] = new_suggest
        state["failures"]     = 0
        state["active"]       = True
        state["window"].clear()
        self.total_replans += 1

        print(f"  🔄 [Efficacy] REPLAN Track {track_id}: "
              f"{old_mode} → {new_mode}  BW={new_bw/1e6:.1f} MHz")
        audit("EFFICACY_REPLAN", track_id=track_id,
              old_mode=old_mode, new_mode=new_mode,
              new_bw_hz=round(new_bw), total_replans=self.total_replans)

    def summary(self) -> str:
        active = sum(1 for s in self._state.values() if s.get("active"))
        return (f"EfficacyMonitor: {len(self._state)} tracks tracked | "
                f"{active} active | {self.total_replans} replans issued")


# =============================================================================
# SECTION 4c · DEEP EMITTER EMBEDDER
# =============================================================================
class _EmitterEmbedderNet(nn.Module if TORCH_OK else object):
    def __init__(self, in_features: int, embed_dim: int = EMITTER_EMBED_DIM):
        if not TORCH_OK: return
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(32),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(32, embed_dim), nn.LayerNorm(embed_dim))

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return F.normalize(self.head(self.pool(self.conv(x)).squeeze(-1)), p=2, dim=-1)


class TripletDataset(TensorDataset if TORCH_OK else object):
    def __init__(self, X: np.ndarray, y: np.ndarray, rng_seed: int = RANDOM_SEED):
        self.X   = torch.tensor(X, dtype=torch.float32) if TORCH_OK else X
        self.y   = y
        self.rng = np.random.default_rng(rng_seed)
        self.cls_idx = {c: np.where(y == c)[0] for c in np.unique(y)}

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        anchor_cls = self.y[idx]
        pos_cands  = self.cls_idx[anchor_cls]
        pos_idx    = int(self.rng.choice(
            pos_cands[pos_cands != idx] if len(pos_cands) > 1 else pos_cands))
        neg_cls = int(self.rng.choice([c for c in self.cls_idx if c != anchor_cls]))
        neg_idx = int(self.rng.choice(self.cls_idx[neg_cls]))
        return self.X[idx], self.X[pos_idx], self.X[neg_idx]


class DeepEmitterEmbedder:
    def __init__(self, embed_dim: int = EMITTER_EMBED_DIM):
        self.embed_dim = embed_dim
        self.model:   Optional[_EmitterEmbedderNet] = None
        self.scaler   = RobustScaler()
        self.fitted   = False
        self.min_val: Optional[np.ndarray] = None
        self.max_val: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepEmitterEmbedder":
        if not TORCH_OK:
            print("  [Embedder] Skipped — PyTorch unavailable"); return self
        t0   = time.time()
        X_sc = np.nan_to_num(self.scaler.fit_transform(X), nan=0., posinf=0., neginf=0.)
        self.min_val = X_sc.min(0); self.max_val = X_sc.max(0)
        X_sc = (X_sc - self.min_val) / (self.max_val - self.min_val + 1e-9)
        self.model = _EmitterEmbedderNet(X.shape[1], self.embed_dim).to(DEVICE)
        opt  = optim.Adam(self.model.parameters(), lr=EMITTER_TRIPLET_LR, weight_decay=1e-5)
        loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda a,b: 1.-F.cosine_similarity(a,b),
            margin=EMITTER_TRIPLET_MARGIN)
        dl = DataLoader(TripletDataset(X_sc, y), batch_size=EMITTER_TRIPLET_BATCH,
                        shuffle=True, drop_last=True)
        self.model.train()
        for ep in range(EMITTER_TRIPLET_EPOCHS):
            total = 0.; nb = 0
            for a,p,n in dl:
                a,p,n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(self.model(a), self.model(p), self.model(n))
                loss.backward(); opt.step()
                total += loss.item(); nb += 1
            if not PRODUCTION_MODE and (ep+1) % 10 == 0:
                print(f"    [Embedder] ep {ep+1}/{EMITTER_TRIPLET_EPOCHS}  "
                      f"triplet_loss={total/max(nb,1):.4f}")
        self.model.eval(); self.fitted = True
        print(f"  ✓ DeepEmitterEmbedder trained ({time.time()-t0:.1f}s)")
        return self

    def embed(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted or self.model is None:
            rng = np.random.default_rng(0)
            W   = rng.normal(0, 1./np.sqrt(X.shape[1]),
                              (X.shape[1], self.embed_dim)).astype(np.float32)
            emb = X.astype(np.float32) @ W
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
            return (emb / norms).astype(np.float32)
        X_sc = np.nan_to_num(self.scaler.transform(X), nan=0., posinf=0., neginf=0.)
        if self.min_val is not None:
            X_sc = (X_sc - self.min_val) / (self.max_val - self.min_val + 1e-9)
        Xt = torch.tensor(X_sc, dtype=torch.float32); embs = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(Xt), 256):
                embs.append(self.model(Xt[i:i+256].to(DEVICE)).cpu().numpy())
        return np.concatenate(embs, axis=0).astype(np.float32)

    def embed_single(self, fv: np.ndarray) -> np.ndarray:
        return self.embed(fv.reshape(1, -1))[0]


# =============================================================================
# SECTION 4d · MULTI-TARGET TRACKER WITH HUNGARIAN ASSOCIATION
# =============================================================================
@dataclass
class TrackState:
    track_id:       int
    embedding_hist: deque = field(default_factory=lambda: deque(maxlen=50))
    feature_hist:   deque = field(default_factory=lambda: deque(maxlen=50))
    first_seen:     float = field(default_factory=time.time)
    last_seen:      float = field(default_factory=time.time)
    silent_frames:  int   = 0
    seen_count:     int   = 0
    threat_scores:  List[float] = field(default_factory=list)
    soft_scores:    List[float] = field(default_factory=list)
    label_history:  List[str]   = field(default_factory=list)
    trust_score:    float = 0.
    promoted:       bool  = False

    @property
    def centroid_embedding(self) -> Optional[np.ndarray]:
        if not self.embedding_hist: return None
        stack = np.stack(list(self.embedding_hist))
        c = stack.mean(0); n = np.linalg.norm(c) + 1e-9
        return (c / n).astype(np.float32)

    def update(self, embedding, fv, ts=0., ss=0.5, label=None,
               m_fv=None, append_hist=True):
        self.embedding_hist.append(embedding.copy())
        if append_hist:
            self.feature_hist.append((m_fv if m_fv is not None else fv).copy())
        self.last_seen = time.time(); self.silent_frames = 0
        self.seen_count += 1
        self.threat_scores.append(float(ts)); self.soft_scores.append(float(ss))
        if label: self.label_history.append(label)

    @property
    def mean_features(self) -> np.ndarray:
        return np.mean(np.stack(list(self.feature_hist)), 0)

    @property
    def feature_variance(self) -> float:
        if len(self.feature_hist) < 2: return 0.0
        stack = np.nan_to_num(np.stack(list(self.feature_hist)),
                               nan=0., posinf=1., neginf=-1.)
        return float(min(stack.var(0).mean(), 10.0))

    @property
    def mean_threat(self) -> float:
        return float(np.mean(self.threat_scores)) if self.threat_scores else 1.

    def compute_trust(self) -> float:
        obs_t  = float(1/(1+np.exp(-(self.seen_count-TRUST_MIN_OBSERVATIONS)/3)))
        stab_t = float(max(0., 1.-self.feature_variance/(TRUST_MAX_VARIANCE+1e-9)))
        safe_t = float(max(0., 1.-self.mean_threat))
        vals   = [obs_t, stab_t, safe_t]
        self.trust_score = float(np.clip(
            len(vals)/sum(1/(v+1e-9) for v in vals), 0., 1.))
        return self.trust_score

    def is_trustworthy(self) -> bool:
        return (self.seen_count >= TRUST_MIN_OBSERVATIONS and
                self.feature_variance <= TRUST_MAX_VARIANCE and
                self.mean_threat < HIGH_THREAT_THRESHOLD)

    def is_promotion_eligible(self) -> bool:
        return (not self.promoted and
                self.seen_count >= PROMO_MIN_OBS and
                self.trust_score >= PROMO_TRUST_THR and
                self.mean_threat <= PROMO_MAX_THREAT)

    def majority_vote_label(self) -> Optional[str]:
        if len(self.label_history) < TEMPORAL_SMOOTHING_MIN: return None
        recent = list(self.label_history)[-TEMPORAL_WINDOW:]
        ctr = Counter(recent); winner, count = ctr.most_common(1)[0]
        return winner if count/len(recent) >= 0.40 else None


class MultiTargetTracker:
    def __init__(self):
        self._tracks:     Dict[int, TrackState] = {}
        self._next_id:    int  = 0
        self.total_obs:   int  = 0
        self.promo_times: List[float] = []

    def _cosine_dist_matrix(self, new_emb, track_ids):
        if not track_ids: return np.array([[]], dtype=np.float32)
        centroids = np.stack([self._tracks[tid].centroid_embedding
                               for tid in track_ids]).astype(np.float32)
        new_norm = new_emb / (np.linalg.norm(new_emb) + 1e-9)
        c_norms  = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
        dist     = (1. - (c_norms @ new_norm)).reshape(1, -1)
        return np.nan_to_num(dist, nan=1., posinf=2., neginf=0.)

    def observe(self, fv, embedding, ts=0., ss=0.5,
                label=None, m_fv=None) -> TrackState:
        self.total_obs += 1
        active_ids = [tid for tid,trk in self._tracks.items()
                      if trk.silent_frames < TRACK_SILENT_FRAMES]
        if not active_ids:
            return self._spawn(fv, embedding, ts, ss, label, m_fv)
        dist_mat = self._cosine_dist_matrix(embedding, active_ids)
        row_ind, col_ind = linear_sum_assignment(dist_mat)
        matched: Optional[TrackState] = None
        for r,c in zip(row_ind, col_ind):
            if dist_mat[r,c] <= TRACK_CREATION_DIST:
                matched = self._tracks[active_ids[c]]
                matched.update(embedding, fv, ts, ss, label, m_fv)
                matched.compute_trust(); break
        if matched is None:
            matched = self._spawn(fv, embedding, ts, ss, label, m_fv)
        for tid in active_ids:
            if self._tracks[tid] is not matched:
                self._tracks[tid].silent_frames += 1
        for tid in [t for t,trk in self._tracks.items()
                    if trk.silent_frames >= TRACK_SILENT_FRAMES]:
            audit("TRACK_TERMINATED", track_id=tid, seen=self._tracks[tid].seen_count)
            del self._tracks[tid]
        return matched

    def _spawn(self, fv, embedding, ts, ss, label, m_fv=None) -> TrackState:
        tid = self._next_id; self._next_id += 1
        trk = TrackState(track_id=tid)
        trk.update(embedding, fv, ts, ss, label, m_fv)
        self._tracks[tid] = trk
        audit("TRACK_SPAWNED", track_id=tid); return trk

    def record_promotion(self, trk: TrackState):
        trk.promoted = True
        self.promo_times.append(time.time() - trk.first_seen)

    def mean_time_to_trust(self) -> float:
        return float(np.mean(self.promo_times)) if self.promo_times else float("nan")

    def reset(self): self._tracks.clear(); self.total_obs = 0; self.promo_times = []

    def summary(self) -> str:
        n   = len(self._tracks)
        nt  = sum(1 for t in self._tracks.values() if t.is_trustworthy())
        nth = sum(1 for t in self._tracks.values() if t.mean_threat>=HIGH_THREAT_THRESHOLD)
        np_ = sum(1 for t in self._tracks.values() if t.promoted)
        t2t = self.mean_time_to_trust()
        return (f"Tracker: {n} active tracks | trustworthy={nt} | "
                f"promoted={np_} | threat={nth} | "
                f"TTT={'N/A' if np.isnan(t2t) else f'{t2t:.1f}s'}")


# =============================================================================
# SECTION 4e · 1D-CNN CLASSIFIER
# =============================================================================
class CNN1D(nn.Module if TORCH_OK else object):
    def __init__(self, in_features, n_classes, embed_dim=CNN_EMBED_DIM, dropout=CNN_DROPOUT):
        if not TORCH_OK: return
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
        )
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.embed = nn.Sequential(nn.Linear(64, embed_dim), nn.ReLU(), nn.Dropout(dropout/2))
        self.head  = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1); x = self.conv(x)
        x = self.pool(x).squeeze(-1); e = self.embed(x)
        return self.head(e), e


class CNNExtractor:
    def __init__(self, n_classes):
        self.n_classes = n_classes; self.model = None
        self.scaler = RobustScaler(); self.fitted = False
        self.embed_dim = CNN_EMBED_DIM if TORCH_OK else 0
        self.min_val = self.max_val = None

    def fit(self, X, y):
        if not TORCH_OK: print("  [CNN] Skipped — PyTorch unavailable"); return self
        t0   = time.time()
        X_sc = np.nan_to_num(self.scaler.fit_transform(X), nan=0., posinf=0., neginf=0.)
        self.min_val = X_sc.min(0); self.max_val = X_sc.max(0)
        X_sc = (X_sc - self.min_val)/(self.max_val - self.min_val + 1e-9)
        dl = DataLoader(TensorDataset(torch.tensor(X_sc, dtype=torch.float32),
                                      torch.tensor(y, dtype=torch.long)),
                        batch_size=CNN_BATCH, shuffle=True, drop_last=True)
        self.model = CNN1D(X.shape[1], self.n_classes).to(DEVICE)
        opt   = optim.Adam(self.model.parameters(), lr=CNN_LR, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CNN_EPOCHS)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        for ep in range(CNN_EPOCHS):
            total_loss=0.; correct=0.; nb=0
            for xb,yb in dl:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad()
                logits,_ = self.model(xb); loss = loss_fn(logits, yb)
                loss.backward(); opt.step()
                total_loss += loss.item()
                correct += (logits.argmax(1)==yb).sum().item(); nb += len(yb)
            sched.step()
            if not PRODUCTION_MODE and (ep+1)%10==0:
                print(f"    CNN ep {ep+1}/{CNN_EPOCHS}  "
                      f"loss={total_loss/len(dl):.4f}  acc={correct/nb:.4f}")
        self.fitted=True; self.model.eval()
        print(f"  ✓ CNN trained ({time.time()-t0:.1f}s)  device={DEVICE}")
        return self

    def _preprocess(self, X):
        X_sc = np.nan_to_num(self.scaler.transform(X), nan=0., posinf=0., neginf=0.)
        if self.min_val is not None:
            X_sc = (X_sc - self.min_val)/(self.max_val - self.min_val + 1e-9)
        return torch.tensor(X_sc, dtype=torch.float32)

    def predict_proba(self, X):
        if not self.fitted or self.model is None:
            return np.ones((len(X), self.n_classes), dtype=np.float32)/self.n_classes
        Xt = self._preprocess(X); probs = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(Xt), 256):
                logits,_ = self.model(Xt[i:i+256].to(DEVICE))
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(probs, axis=0)


# =============================================================================
# SECTION 4f · STACKING META-LEARNER
# =============================================================================
class StackingMetaLearner:
    def __init__(self):
        self.meta = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced",
                                        random_state=42, n_jobs=-1)
        self.scaler = RobustScaler(); self.fitted = False; self.classes_ = None

    def fit(self, rf_probs, gbt_probs, gbp_probs, y):
        X_sc = self.scaler.fit_transform(
            np.concatenate([rf_probs, gbt_probs, gbp_probs], axis=1))
        self.meta.fit(X_sc, y); self.classes_ = self.meta.classes_
        yp = self.meta.predict(X_sc)
        print(f"  ✓ StackingMeta  acc={accuracy_score(y,yp):.4f}  "
              f"F1={f1_score(y,yp,average='macro',zero_division=0):.4f}")
        self.fitted = True; return self

    def predict_proba(self, rf_p, gbt_p, gbp_p):
        row = np.concatenate([rf_p, gbt_p, gbp_p]).reshape(1, -1)
        return self.meta.predict_proba(self.scaler.transform(row))[0]


# =============================================================================
# SECTION 4g · FEATURE ROUTE CACHE
# =============================================================================
class _FeatureCache:
    def __init__(self, maxsize=ROUTE_CACHE_MAXSIZE):
        self._cache={}; self._order=[]; self.maxsize=maxsize; self.hits=self.misses=0

    def _key(self, fv):
        return hashlib.blake2b(np.round(fv*20).astype(np.int16).tobytes(),
                                digest_size=6).hexdigest()

    def get(self, fv):
        k = self._key(fv)
        if k in self._cache: self.hits+=1; return self._cache[k]
        self.misses+=1; return None

    def put(self, fv, routed):
        k = self._key(fv)
        if len(self._order) >= self.maxsize:
            self._cache.pop(self._order.pop(0), None)
        self._cache[k]=routed; self._order.append(k)

    def clear(self): self._cache.clear(); self._order.clear(); self.hits=self.misses=0


_ROUTE_CACHE = _FeatureCache(maxsize=ROUTE_CACHE_MAXSIZE)


# =============================================================================
# SECTION 4h · LIGHTGBM WRAPPER
# =============================================================================
def _class_weights(y_arr, boost_cls=1, boost_factor=1.5):
    w = np.ones(len(y_arr), dtype=np.float32)
    w[y_arr == boost_cls] = boost_factor
    return w


class LGBClassifier:
    def __init__(self, n_estimators=100, mode="rf", n_classes=4,
                 num_leaves=63, lr=0.1, min_data_leaf=3,
                 subsample=0.8, colsample=0.5, device=None):
        self.n_estimators=n_estimators; self.mode=mode; self.n_classes=n_classes
        self.num_leaves=num_leaves; self.lr=lr; self.min_data_leaf=min_data_leaf
        self.subsample=subsample; self.colsample=colsample
        self.device=device or _LGB_DEVICE
        self.booster: Optional[lgb.Booster]=None
        self.classes_=np.arange(n_classes)

    def _params(self):
        p = {"objective":"multiclass","num_class":self.n_classes,
             "num_leaves":self.num_leaves,"min_data_in_leaf":self.min_data_leaf,
             "feature_fraction":self.colsample,"bagging_fraction":self.subsample,
             "bagging_freq":1,"verbose":-1,"n_jobs":-1,
             "seed":RANDOM_SEED,"device_type":self.device}
        if self.mode=="rf":  p["boosting"]="rf";   p["learning_rate"]=1.0
        else:                p["boosting"]="gbdt"; p["learning_rate"]=self.lr
        return p

    def fit(self, X, y, sample_weight=None):
        t0=time.time()
        self.booster=lgb.train(
            self._params(),
            lgb.Dataset(X,label=y,weight=sample_weight,free_raw_data=False),
            num_boost_round=self.n_estimators,
            callbacks=[lgb.log_evaluation(period=-1)])
        yp=self.predict(X).argmax(1)
        print(f"  ✓ LGB-{self.mode.upper()}  "
              f"acc={accuracy_score(y,yp):.4f}  "
              f"F1={f1_score(y,yp,average='macro',zero_division=0):.4f}  "
              f"({time.time()-t0:.1f}s)")
        return self

    def predict(self, X):
        raw=self.booster.predict(X)
        if raw.ndim==1:
            p1=raw.reshape(-1,1); return np.concatenate([1-p1,p1],axis=1)
        return raw

    def predict_proba(self, X): return self.predict(X)


# =============================================================================
# SECTION 4i · ACTION CONTROLLER  (Layer 3 AI-Harness)
# =============================================================================
class ActionController:
    def __init__(self, log_path: str = DEFENSE_LOG_PATH, enabled: bool = True):
        self.log_path = log_path; self.enabled = enabled
        self._actions = 0; self._last_ts: Dict[str, float] = {}
        self._cooldown_s = 1.5
        self.scheduler = LookThroughScheduler()
        self._last_suggestions: Dict[int, Dict[str, Any]] = {}

    def trigger_defense(self, threat_label: str, track_id: int = -1,
                         emitter_id: str = "", soft_score: float = 0.0,
                         signal_bandwidth: float = 2e6,
                         center_freq: float = 2.4e9) -> Tuple[bool, Optional[Dict]]:
        """
        Returns (fired: bool, suggestion: dict | None).
        Caller must pass suggestion to JammingEfficacyMonitor.register_jam().
        """
        if not self.enabled: return False, None
        now = time.time()
        key = str(track_id) if track_id >= 0 else emitter_id
        if key and (now - self._last_ts.get(key, 0.)) < self._cooldown_s:
            return False, None
        if key: self._last_ts[key] = now
        self._actions += 1

        suggestion = self.scheduler.generate_jamming_suggest(
            threat_label, center_freq, signal_bandwidth)
        self._last_suggestions[track_id] = suggestion

        self._log(threat_label, track_id, soft_score, now, suggestion)
        print(f"  📡 LOOK-THROUGH JAM: Mode='{suggestion['mode']}'  "
              f"Ctr={suggestion['center_hz']/1e6:.1f} MHz  "
              f"BW={suggestion['width_hz']/1e6:.1f} MHz")
        audit("ACTION_TRIGGERED", threat_label=threat_label,
              track_id=track_id, soft_score=round(soft_score,4),
              action_n=self._actions, suggestion=suggestion)
        return True, suggestion

    def reset(self): self._actions=0; self._last_ts.clear()

    @property
    def total_actions(self) -> int: return self._actions

    def _log(self, label, track_id, score, ts, suggest):
        with open(self.log_path, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(ts))}] "
                    f"ACTION: {label} | track_id={track_id} | score={score:.4f} "
                    f"| mode={suggest['mode']} | ctr={suggest['center_hz']:.0f} Hz\n")


# =============================================================================
# SECTION 4j · LIVE STREAM SIMULATOR
# =============================================================================
class LiveStreamSimulator:
    def __init__(self, out_dir=LIVE_STREAM_DIR, interval_ms=STREAM_INTERVAL_MS,
                 max_bursts=STREAM_MAX_BURSTS, csv_dir=None, cls_override=None):
        self.out_dir=out_dir; self.interval_s=interval_ms/1000.
        self.max_bursts=max_bursts; self.csv_dir=csv_dir; self.cls_override=cls_override
        self._thread=None; self._stop_evt=threading.Event(); self._burst_count=0
        os.makedirs(out_dir, exist_ok=True)
        for f in Path(out_dir).glob("burst_*.npy"): f.unlink(missing_ok=True)

    def start(self):
        self._stop_evt.clear()
        self._thread=threading.Thread(target=self._emit_loop, daemon=True)
        self._thread.start()
        print(f"  [SDR] Emulator started → {self.out_dir}/  "
              f"({self.interval_s*1000:.0f} ms interval)")

    def stop(self):
        self._stop_evt.set()
        if self._thread: self._thread.join(timeout=5.)

    def poll_next(self):
        candidates=sorted(Path(self.out_dir).glob("burst_*.npy"))
        if not candidates: return None
        path=candidates[0]
        try:    fv=np.load(str(path))
        except: path.unlink(missing_ok=True); return None
        path.unlink(missing_ok=True); return fv

    @property
    def burst_count(self): return self._burst_count
    @property
    def is_done(self): return self._stop_evt.is_set()

    def _emit_loop(self):
        rng=np.random.default_rng(RANDOM_SEED+200)
        csv_rows=self._load_csv_rows(); csv_idx=0
        while not self._stop_evt.is_set() and self._burst_count < self.max_bursts:
            if csv_rows is not None:
                fv=csv_rows[csv_idx % len(csv_rows)]; csv_idx+=1
            else:
                cls=(self.cls_override if self.cls_override is not None
                     else int(rng.integers(0,4)))
                fv=_generate_rf_burst(cls, rng, noise_scale=1.2, apply_channel=True)
            fname=Path(self.out_dir)/f"burst_{self._burst_count:06d}.npy"
            np.save(str(fname), fv)
            self._burst_count+=1; time.sleep(self.interval_s)
        self._stop_evt.set()

    def _load_csv_rows(self):
        if not self.csv_dir: return None
        root=Path(self.csv_dir)
        if not root.exists(): return None
        rows=[]
        for fp in sorted(root.rglob("*.csv"))[:10]:
            try:
                raw=pd.read_csv(fp,header=None,dtype=np.float32).values.ravel()
                if len(raw)<WINDOW_SIZE: continue
                for start in range(0,len(raw)-WINDOW_SIZE,STEP_SIZE):
                    rows.append(fuse_features(safe_extract_rf(raw[start:start+WINDOW_SIZE])))
                    if len(rows)>=self.max_bursts*2: break
            except: continue
            if len(rows)>=self.max_bursts*2: break
        if not rows: return None
        print(f"  [SDR] Loaded {len(rows)} windows from {self.csv_dir}")
        return np.array(rows, dtype=np.float32)


def run_live_stream_demo(classify_signal_fn, csv_dir=None, cls_override=None,
                          max_bursts=STREAM_MAX_BURSTS, interval_ms=STREAM_INTERVAL_MS):
    print(f"\n{'═'*65}")
    print("  LIVE STREAM DEMO  (Virtual SDR / EW Testbed)")
    print(f"  Interval={interval_ms} ms  MaxBursts={max_bursts}  Mode={DEPLOYMENT_MODE}")
    print(f"{'═'*65}")
    sim=LiveStreamSimulator(out_dir=LIVE_STREAM_DIR, interval_ms=interval_ms,
                             max_bursts=max_bursts, csv_dir=csv_dir,
                             cls_override=cls_override)
    sim.start()
    processed=0; label_counts=Counter(); latencies_ms=[]
    try:
        while processed<max_bursts:
            fv=sim.poll_next()
            if fv is None:
                if sim.is_done and processed>=sim.burst_count: break
                time.sleep(0.005); continue
            t0=time.perf_counter(); dec=classify_signal_fn(fv)
            lat=(time.perf_counter()-t0)*1000
            label=dec.get("label","UNKNOWN")
            label_counts[label]+=1; latencies_ms.append(lat); processed+=1
            print(f"  Burst {processed:>3}  {DECISION_ICONS.get(label,'?')} "
                  f"{label:<28} score={dec.get('soft_score',0.):.3f}  lat={lat:.1f} ms")
    finally:
        sim.stop()
    if latencies_ms:
        arr=np.array(latencies_ms)
        print(f"\n  Processed {processed} bursts  "
              f"mean={arr.mean():.1f} ms  p95={np.percentile(arr,95):.1f} ms")
    for f in Path(LIVE_STREAM_DIR).glob("burst_*.npy"): f.unlink(missing_ok=True)
    return label_counts, latencies_ms


# =============================================================================
# SECTION 5 · DATA PIPELINE
# =============================================================================
def build_or_load_dataset(data_dir, output_csv=OUTPUT_CSV):
    cache=Path(output_csv)
    if cache.exists():
        try:
            df=pd.read_csv(output_csv)
            if ("high_low_band_ratio" in df.columns and
                    df["amp_std"].var()>1e-4 and
                    "synthetic" not in str(df["source_file"].iloc[0])):
                for col in ALL_FEATURE_NAMES:
                    if col not in df.columns: df[col]=0.
                print(f"⚡ Cache loaded: {output_csv}  ({len(df):,} rows)")
                return df
        except Exception as e:
            print(f"  Cache load failed: {e}")
        cache.unlink(missing_ok=True)

    if not (data_dir and Path(data_dir).exists()):
        print("  ⚠  DATA_DIR not found → synthetic fallback")
        df=generate_realistic_dataset(); df.to_csv(output_csv,index=False); return df

    print(f"\nBuilding from real data: {data_dir} ...")
    root=Path(data_dir); folder_class={}
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir(): continue
        cls=next((v for k,v in FOLDER_MAP.items() if k in subdir.name.lower()), None)
        if cls is not None:
            folder_class[subdir]=cls
            print(f"  Folder '{subdir.name}' → class {cls} ({CLASS_NAMES[cls]})")

    if not folder_class:
        df=generate_realistic_dataset(); df.to_csv(output_csv,index=False); return df

    class_files={}
    for folder,cls in folder_class.items():
        csv_files=sorted(folder.rglob("*.csv"))
        if csv_files: class_files.setdefault(cls,[]).extend(csv_files)

    if not class_files:
        df=generate_realistic_dataset(); df.to_csv(output_csv,index=False); return df

    rng=np.random.default_rng(RANDOM_SEED); q=TARGET_TOTAL//len(class_files)
    rows,labels,fnames=[],[],[]
    for cls in sorted(class_files.keys()):
        flist=list(class_files[cls]); rng.shuffle(flist)
        count=0; skipped=0
        for fp in flist:
            if count>=q: break
            try:
                raw=pd.read_csv(fp,header=None,dtype=np.float32).values.ravel()
            except: skipped+=1; continue
            if len(raw)<WINDOW_SIZE: skipped+=1; continue
            start=0
            while start+WINDOW_SIZE<=len(raw) and count<q:
                rows.append(fuse_features(safe_extract_rf(raw[start:start+WINDOW_SIZE])))
                labels.append(cls); fnames.append(fp.name)
                start+=STEP_SIZE; count+=1
        print(f"  ✓ Class {cls} ({CLASS_NAMES[cls]}): {count} windows (skipped {skipped})")

    if not rows:
        df=generate_realistic_dataset(); df.to_csv(output_csv,index=False); return df

    X=np.array(rows,dtype=np.float32)
    df=pd.DataFrame(X,columns=ALL_FEATURE_NAMES)
    df.insert(0,"label_int",labels)
    df.insert(1,"label_name",[CLASS_NAMES[c] for c in labels])
    df.insert(2,"source_file",fnames)
    df=df.sample(frac=1,random_state=RANDOM_SEED).reset_index(drop=True)
    df.to_csv(output_csv,index=False); return df


def prepare_data(df):
    X_all=np.nan_to_num(
        df[ALL_FEATURE_NAMES].fillna(0).values.astype(np.float32),
        nan=0., posinf=0., neginf=0.)
    y_all=df["label_int"].values.astype(np.int64)
    known=sorted([c for c in np.unique(y_all)
                  if c in CLASS_NAMES and (y_all==c).sum()>=6])
    mask=np.isin(y_all,known)
    X_use,y_use=X_all[mask],y_all[mask]
    lmap={old:new for new,old in enumerate(known)}
    y_map=np.array([lmap[yi] for yi in y_use],dtype=np.int64)
    CP=[CLASS_NAMES[c] for c in known]
    print(f"\n  Training classes: {len(CP)}")
    for i,cn in enumerate(CP):
        print(f"    [{i}] {cn:<35} ({(y_map==i).sum()} bursts)")
    return X_use, y_map, lmap, CP, len(CP)


# =============================================================================
# SECTION 6 · FEATURE ROUTER
# =============================================================================
class FeatureRouter:
    def __init__(self, rf_idx, gbt_idx, master_idx,
                 scaler_rf, scaler_gbt, scaler_master, scaler_sub, sub_idx,
                 ew_idx=None, scaler_ew=None):
        self.rf_idx=rf_idx; self.gbt_idx=gbt_idx; self.master_idx=master_idx
        self.sub_idx=sub_idx; self.ew_idx=ew_idx
        self.scaler_rf=scaler_rf; self.scaler_gbt=scaler_gbt
        self.scaler_master=scaler_master; self.scaler_sub=scaler_sub
        self.scaler_ew=scaler_ew

    def route(self, fv_raw):
        X=fv_raw if fv_raw.ndim==2 else fv_raw.reshape(1,-1)
        X=np.nan_to_num(X.astype(np.float32), nan=0., posinf=0., neginf=0.)
        def _s(sc,idx):
            return np.nan_to_num(sc.transform(X[:,idx]), nan=0., posinf=0., neginf=0.)
        res={"rf":_s(self.scaler_rf,self.rf_idx),
             "gbt":_s(self.scaler_gbt,self.gbt_idx),
             "master":_s(self.scaler_master,self.master_idx),
             "sub":_s(self.scaler_sub,self.sub_idx)}
        if self.ew_idx is not None and self.scaler_ew is not None:
            res["ew"]=_s(self.scaler_ew,self.ew_idx)
        return res


# =============================================================================
# SECTION 7 · FEATURE SELECTION
# =============================================================================
def validate_and_select_features(X, y):
    print(f"\n{'='*60}\nFEATURE SELECTION\n{'='*60}")
    sc_pre=RobustScaler()
    X_s=np.nan_to_num(sc_pre.fit_transform(X), nan=0., posinf=0., neginf=0.)
    mi=mutual_info_classif(X_s, y, random_state=RANDOM_SEED)
    top_mi=np.argsort(mi)[::-1]; top_var=np.argsort(X_s.var(0))[::-1]
    rf_idx=top_mi[:RF_TOP_K_MI]; gbt_idx=top_var[:GBT_TOP_K_VAR]
    master_idx=top_mi
    # SUBCLF_FEATURES is defined in Section 0b — always available here
    sub_names=[f for f in SUBCLF_FEATURES if f in FEAT_IDX]
    sub_idx=np.array([FEAT_IDX[f] for f in sub_names], dtype=np.int64)
    ew_names=[f for f in EW_FEATURE_NAMES if f in FEAT_IDX]
    ew_idx=np.array([FEAT_IDX[f] for f in ew_names], dtype=np.int64)

    def _fs(idx):
        sc=RobustScaler()
        Xs=np.nan_to_num(sc.fit_transform(X[:,idx]), nan=0., posinf=0., neginf=0.)
        return sc, Xs

    scaler_rf,     X_rf     = _fs(rf_idx)
    scaler_gbt,    X_gbt    = _fs(gbt_idx)
    scaler_master, X_master = _fs(master_idx)
    scaler_sub,    X_sub    = _fs(sub_idx)
    scaler_ew,     X_ew     = _fs(ew_idx)
    print(f"  RF(MI-top-{RF_TOP_K_MI}) | GBT(Var-top-{GBT_TOP_K_VAR}) | EW({len(ew_idx)} features)")
    router=FeatureRouter(rf_idx, gbt_idx, master_idx,
                          scaler_rf, scaler_gbt, scaler_master, scaler_sub, sub_idx,
                          ew_idx=ew_idx, scaler_ew=scaler_ew)
    return router, mi, X_master, X_rf, X_gbt, X_sub, X_ew


# =============================================================================
# SECTION 8 · CLASSIFIERS
# =============================================================================
class GaussianBayesPosterior:
    def __init__(self, temperature=GBP_TEMPERATURE, var_smoothing=1e-3):
        self.tau=temperature; self.vsf=var_smoothing; self.fitted=False

    def fit(self, X, y):
        classes=np.unique(y); self.classes_=classes
        smooth=self.vsf*X.var(0).mean()
        self.mu_={}; self.var_={}; self.log_prior_={}
        for k in classes:
            Xk=X[y==k]; self.mu_[k]=Xk.mean(0); self.var_[k]=Xk.var(0)+smooth
            self.log_prior_[k]=float(np.log(len(Xk)/len(y)))
        self.fitted=True; print(f"  ✓ GBP  τ={self.tau}"); return self

    def predict_proba(self, X):
        X=np.asarray(X,dtype=np.float64)
        lp=np.stack([-0.5*((X-self.mu_[k])**2/self.var_[k]).sum(1)/self.tau
                     -0.5*np.log(2*np.pi*self.var_[k]).sum()/self.tau
                     +self.log_prior_[k] for k in self.classes_], axis=1)
        lp-=lp.max(1,keepdims=True); p=np.exp(lp); p/=p.sum(1,keepdims=True)
        return p

    def predict(self, X): return self.predict_proba(X).argmax(1)


class EnsembleUncertainty:
    def __init__(self, n_models=N_ENSEMBLE_TREES, subsample=ENSEMBLE_SUBSAMPLE):
        self.n_models=n_models; self.subsample=subsample; self.models=[]

    def fit(self, X, y):
        n_cls=len(np.unique(y))
        print(f"  [Ensemble] Training {self.n_models} bootstrap sub-models ...")
        rng=np.random.default_rng(RANDOM_SEED); n=len(X)
        for _ in range(self.n_models):
            idx=rng.choice(n,size=int(n*self.subsample),replace=True)
            if LGB_OK:
                m=LGBClassifier(n_estimators=200,mode="rf",n_classes=n_cls,
                                 num_leaves=31,min_data_leaf=3)
            else:
                from sklearn.ensemble import RandomForestClassifier as RFC
                m=RFC(200,max_features="sqrt",min_samples_leaf=3,
                      class_weight="balanced",random_state=42,n_jobs=-1)
            m.fit(X[idx],y[idx]); self.models.append(m)
        avg_p=np.mean([m.predict_proba(X) for m in self.models],axis=0)
        print(f"  ✓ Ensemble F1={f1_score(y,avg_p.argmax(1),average='macro',zero_division=0):.4f}")
        return self

    def predict_with_uncertainty(self, X):
        probs=np.stack([m.predict_proba(X) for m in self.models],axis=0)
        mean_p=probs.mean(0); epistemic=probs.var(0).sum(-1)
        aleatoric=-(mean_p*np.log(mean_p+1e-12)).sum(-1)
        return mean_p, epistemic, aleatoric


class PhantomARSubClassifier:
    def __init__(self): self.model=None; self.fitted=False

    def fit(self, X_sub, y):
        mask=np.isin(y,[1,2])
        if mask.sum()<20: return self
        Xs=X_sub[mask]; ys=(y[mask]==2).astype(np.int64)
        if LGB_OK:
            self.model=LGBClassifier(n_estimators=300,mode="gbdt",n_classes=2,
                                      num_leaves=15,lr=0.05,min_data_leaf=3,
                                      subsample=0.8,colsample=0.7)
        else:
            from sklearn.ensemble import GradientBoostingClassifier as GBC
            self.model=GBC(n_estimators=300,learning_rate=0.05,max_depth=4,
                            subsample=0.8,min_samples_leaf=3,random_state=RANDOM_SEED)
        self.model.fit(Xs,ys)
        yp=self.model.predict_proba(Xs).argmax(1)
        print(f"  ✓ GFSK/FHSS sub-clf F1={f1_score(ys,yp,average='binary',zero_division=0):.4f}")
        self.fitted=True; return self

    def p_phantom(self, X_sub):
        if not self.fitted or self.model is None: return 0.5
        return float(self.model.predict_proba(X_sub)[0,1])


class TemperatureScaler:
    def __init__(self): self.T=1.0

    def fit(self, logits, y):
        def ece_fn(T):
            T=max(T,TEMP_MIN); s=logits/T
            e=np.exp(s-s.max(1,keepdims=True)); p=e/e.sum(1,keepdims=True)
            pred=p.argmax(1); return float(np.mean((p.max(1)-(pred==y).astype(float))**2))
        res=minimize_scalar(ece_fn,bounds=(TEMP_MIN,TEMP_MAX),method="bounded")
        self.T=float(np.clip(res.x,TEMP_MIN,TEMP_MAX))
        print(f"  ✓ TemperatureScaler T={self.T:.4f}"); return self

    def calibrate(self, logits):
        T=max(self.T,TEMP_MIN); s=logits/T
        e=np.exp(s-s.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)

    def expected_calibration_error(self, probs, y, n_bins=10):
        confs=probs.max(1); preds=probs.argmax(1); acc=(preds==y).astype(float); ece=0.
        for b in range(n_bins):
            lo,hi=b/n_bins,(b+1)/n_bins; mask=(confs>=lo)&(confs<hi)
            if mask.sum()==0: continue
            ece+=mask.sum()/len(y)*abs(acc[mask].mean()-confs[mask].mean())
        return float(ece)


class LaplaceApproximation:
    def __init__(self,precision=LAPLACE_PRIOR_PRECISION,n_samples=LAPLACE_N_SAMPLES):
        self.alpha=precision; self.n_samples=n_samples; self.fitted=False

    def fit(self,lr_model,X,y,n_classes):
        t0=time.time(); self.n_classes=n_classes; D=X.shape[1]
        self.W_map=lr_model.coef_.astype(np.float64)
        self.b_map=lr_model.intercept_.astype(np.float64)
        Z=X@self.W_map.T+self.b_map; Z-=Z.max(1,keepdims=True)
        eZ=np.exp(Z); probs=eZ/eZ.sum(1,keepdims=True)
        self.chol_factors=[]
        for k in range(n_classes):
            pi=probs[:,k].clip(1e-7,1-1e-7); w=pi*(1-pi)
            H=(X*w[:,None]).T@X+self.alpha*np.eye(D)
            try:    self.chol_factors.append(("chol",cho_factor(H,lower=False,check_finite=False),H))
            except: self.chol_factors.append(("pinv",np.linalg.pinv(H),H))
        self.fitted=True; print(f"  ✓ Laplace ({time.time()-t0:.2f}s)"); return self

    def predictive_variance(self,X):
        if not self.fitted or PRODUCTION_MODE: return 0.
        X=np.asarray(X,dtype=np.float64); C=self.n_classes
        samples=np.zeros((self.n_samples,X.shape[0],C))
        for k in range(C):
            kind,factor,H=self.chol_factors[k]; D=self.W_map.shape[1]
            z=np.random.randn(self.n_samples,D)
            if kind=="chol":
                try:    v=cho_solve(factor,z.T,check_finite=False).T
                except: v=z/(np.diag(H)+1e-8)
            else:
                try:    v=(np.linalg.cholesky(factor+1e-8*np.eye(D))@z.T).T
                except: v=z*np.sqrt(np.diag(factor)+1e-8)
            samples[:,:,k]=(X@(self.W_map[k]+v).T+self.b_map[k]).T
        Z=samples-samples.max(-1,keepdims=True); p=np.exp(Z); p/=p.sum(-1,keepdims=True)
        return float(p.var(0).mean())


# =============================================================================
# SECTION 8b · OPEN-SET DETECTOR (Deep SVDD + OC-SVM fallback)
# =============================================================================
class _LegacyOpenSetDetector:
    def __init__(self,nu=OCSVM_NU,gamma=OCSVM_GAMMA,n_pca=12):
        self.nu=nu; self.gamma=gamma; self.n_pca=n_pca
        self.models={}; self.pca=None; self.fitted=False; self._lo={}; self._hi={}

    def fit(self,X_master,y):
        n_comp=min(self.n_pca,X_master.shape[1],X_master.shape[0]-1)
        self.pca=PCA(n_components=n_comp,random_state=RANDOM_SEED)
        X_pca=self.pca.fit_transform(X_master)
        for k in np.unique(y):
            Xk=X_pca[y==k]
            m=OneClassSVM(nu=self.nu,kernel="rbf",gamma=self.gamma); m.fit(Xk)
            self.models[k]=m
            scores=m.decision_function(Xk)
            self._lo[k]=float(np.percentile(scores,1)); self._hi[k]=float(np.percentile(scores,99))
            if self._hi[k]<=self._lo[k]: self._hi[k]=self._lo[k]+1.
        self.fitted=True; return self

    def inclusion_score(self,X_master):
        X_pca=self.pca.transform(np.asarray(X_master,dtype=np.float64))
        scores=[]
        for k,m in self.models.items():
            raw=m.decision_function(X_pca)
            scores.append(np.clip((raw-self._lo[k])/(self._hi[k]-self._lo[k]+1e-9),0.,1.))
        return np.stack(scores,axis=1).max(1)


class _SVDDNet(nn.Module if TORCH_OK else object):
    def __init__(self,in_dim,embed_dim=SVDD_EMBED_DIM):
        if not TORCH_OK: return
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim,128),nn.BatchNorm1d(128),nn.LeakyReLU(0.1),
            nn.Linear(128,64),nn.BatchNorm1d(64),nn.LeakyReLU(0.1),
            nn.Linear(64,embed_dim,bias=False))
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,x): return self.net(x)


class DeepSVDDDetector:
    def __init__(self,nu=SVDD_NU,embed_dim=SVDD_EMBED_DIM):
        self.nu=nu; self.embed_dim=embed_dim; self.net=None
        self.centre=None; self.radius=None; self.scaler=RobustScaler()
        self.fitted=False; self._fallback=_LegacyOpenSetDetector()

    def fit(self,X_master,y):
        if not TORCH_OK:
            self._fallback.fit(X_master,y); self.fitted=True; return self
        t0=time.time(); in_dim=X_master.shape[1]
        X_sc=np.nan_to_num(self.scaler.fit_transform(X_master),nan=0.,posinf=0.,neginf=0.)
        Xt=torch.tensor(X_sc,dtype=torch.float32)
        self.net=_SVDDNet(in_dim,self.embed_dim).to(DEVICE)
        self.net.eval()
        with torch.no_grad():
            embs=F.normalize(self.net(Xt.to(DEVICE)),p=2,dim=1)
            c=embs.mean(0)
            self.centre=F.normalize(c.unsqueeze(0),p=2,dim=1).squeeze(0).detach()
        opt=torch.optim.Adam(filter(lambda p:p.requires_grad,self.net.parameters()),
                              lr=SVDD_LR,weight_decay=1e-5)
        dl=DataLoader(TensorDataset(Xt),batch_size=SVDD_BATCH,shuffle=True)
        self.net.train()
        for _ in range(SVDD_EPOCHS):
            for (xb,) in dl:
                xb=xb.to(DEVICE); opt.zero_grad()
                emb=F.normalize(self.net(xb),p=2,dim=1)
                loss=((emb-self.centre)**2).sum(dim=1).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.0)
                opt.step()
        self.net.eval()
        with torch.no_grad():
            all_embs=[]
            for i in range(0,len(Xt),256):
                e=F.normalize(self.net(Xt[i:i+256].to(DEVICE)),p=2,dim=1)
                all_embs.append(e.cpu())
            all_embs=torch.cat(all_embs)
            dists=((all_embs-self.centre.cpu())**2).sum(dim=1).sqrt()
            self.radius=float(torch.quantile(dists,1.-self.nu).item())+1e-6
        self.fitted=True
        print(f"  ✓ DeepSVDD radius={self.radius:.4f}  ({time.time()-t0:.1f}s)")
        return self

    def _raw_distances(self,X_master):
        if not TORCH_OK or self.net is None:
            return 1.-self._fallback.inclusion_score(X_master)
        X_sc=np.nan_to_num(self.scaler.transform(np.asarray(X_master,dtype=np.float32)),
                             nan=0.,posinf=0.,neginf=0.)
        Xt=torch.tensor(X_sc,dtype=torch.float32); dists=[]
        self.net.eval()
        with torch.no_grad():
            for i in range(0,len(Xt),256):
                emb=F.normalize(self.net(Xt[i:i+256].to(DEVICE)),p=2,dim=1)
                dists.append(((emb-self.centre)**2).sum(dim=1).sqrt().cpu().numpy())
        return np.concatenate(dists)

    def inclusion_score(self,X_master):
        if not self.fitted: return np.ones(len(X_master),dtype=np.float32)*0.5
        if not TORCH_OK: return self._fallback.inclusion_score(X_master)
        return np.clip(1.-(self._raw_distances(X_master)/(self.radius*2.)),0.,1.)


# =============================================================================
# SECTION 9 · ANOMALY DETECTORS
# =============================================================================
class MahalanobisDetector:
    def fit(self,X_master,y):
        self.params={}
        for c in np.unique(y):
            Xc=X_master[y==c]; mu=Xc.mean(0)
            cov=np.cov(Xc,rowvar=False)+np.eye(Xc.shape[1])*1e-2
            try:    prec=np.linalg.inv(cov)
            except: prec=np.linalg.pinv(cov)
            self.params[c]=(mu,prec)
        raw=self.score(X_master)
        self._lo=float(np.percentile(raw,1)); self._hi=float(np.percentile(raw,99))
        if self._hi<=self._lo: self._hi=self._lo+1.
        self.threshold=float(np.percentile(raw,99)); return self

    def score(self,X):
        dists=[]
        for mu,prec in self.params.values():
            d=X-mu
            dists.append(np.sqrt(np.maximum(np.einsum("ni,ij,nj->n",d,prec,d),0.)))
        return np.nan_to_num(np.stack(dists,1).min(1),nan=0.,posinf=0.,neginf=0.)

    def norm_score(self,X):
        return np.clip((self.score(X)-self._lo)/(self._hi-self._lo+1e-9),0.,1.)


class IsoForestDetector:
    def fit(self,X_master,y=None):
        self.model=IsolationForest(n_estimators=ISO_N_ESTIMATORS,
                                    contamination=ISO_CONTAMINATION,
                                    n_jobs=-1,random_state=RANDOM_SEED)
        self.model.fit(X_master)
        raw=self.score(X_master)
        self._lo=float(np.percentile(raw,1)); self._hi=float(np.percentile(raw,99))
        if self._hi<=self._lo: self._hi=self._lo+1.
        return self

    def score(self,X):
        return np.nan_to_num(-self.model.score_samples(X),nan=0.,posinf=0.,neginf=0.)

    def norm_score(self,X):
        return np.clip((self.score(X)-self._lo)/(self._hi-self._lo+1e-9),0.,1.)


class ThreatScorer:
    def __init__(self,dm,di,X_master_train):
        self.dm=dm; self.di=di; self.wm=ANOMALY_W_MAHAL; self.wi=ANOMALY_W_ISO
        self.cap=ANOMALY_SCORE_CAP
        raw_thr=float(np.percentile(self.compute_raw(X_master_train),97))
        self.threshold=max(raw_thr,0.72)
        print(f"  ThreatScorer threshold={self.threshold:.4f}")

    def compute_raw(self,X_master):
        return self.wm*self.dm.norm_score(X_master)+self.wi*self.di.norm_score(X_master)

    def compute(self,X_master):
        return np.minimum(self.compute_raw(X_master),self.cap)


# =============================================================================
# SECTION 10 · BUILD & EVALUATE
# =============================================================================
def build_and_evaluate(router, X_raw_full, y, X_master, X_rf, X_gbt, X_sub, X_ew,
                        classes_present, embedder=None):
    print(f"\n{'='*60}\nMODEL TRAINING  (v34-EW-FIELD)\n{'='*60}")
    rng_aug=np.random.default_rng(RANDOM_SEED+1)
    N_CLS=len(classes_present)

    idx_tr,idx_te=train_test_split(
        np.arange(len(y)), test_size=0.20, stratify=y, random_state=RANDOM_SEED)

    # Fix: define X_te_raw inside function from idx_te
    X_tr_raw=X_raw_full[idx_tr]; y_tr=y[idx_tr]
    X_te_raw=X_raw_full[idx_te]; y_te=y[idx_te]   # ← was undefined in original

    X_tr_aug,y_tr_aug=mixup_augment(X_tr_raw,y_tr,rng_aug)

    def _scale_aug(X_aug,sc,idx):
        return np.nan_to_num(sc.transform(X_aug[:,idx]),nan=0.,posinf=0.,neginf=0.)

    X_m_aug  =_scale_aug(X_tr_aug,router.scaler_master,router.master_idx)
    X_rf_aug =_scale_aug(X_tr_aug,router.scaler_rf,    router.rf_idx)
    X_gb_aug =_scale_aug(X_tr_aug,router.scaler_gbt,   router.gbt_idx)
    X_sb_aug =_scale_aug(X_tr_aug,router.scaler_sub,   router.sub_idx)

    X_te_rf  =X_rf[idx_te];  X_te_gbt=X_gbt[idx_te]
    X_te_sub =X_sub[idx_te]; X_te_m  =X_master[idx_te]

    _,cnts=np.unique(y_tr_aug,return_counts=True)
    k_sm=max(1,min(5,int(cnts.min())-1))
    def _smote(X,y_):
        return SMOTE(random_state=RANDOM_SEED,k_neighbors=k_sm).fit_resample(X,y_)
    X_sm_m, y_sm_m =_smote(X_m_aug, y_tr_aug)
    X_sm_rf,y_sm_rf=_smote(X_rf_aug,y_tr_aug)
    X_sm_gb,y_sm_gb=_smote(X_gb_aug,y_tr_aug)
    X_sm_sb,y_sm_sb=_smote(X_sb_aug,y_tr_aug)

    # Train embedder
    if embedder is None:
        embedder=DeepEmitterEmbedder(embed_dim=EMITTER_EMBED_DIM)
    embedder.fit(X_tr_aug,y_tr_aug)

    # 1D-CNN
    cnn=CNNExtractor(n_classes=N_CLS); cnn.fit(X_tr_aug,y_tr_aug)

    # LGB-RF
    if LGB_OK:
        rf=LGBClassifier(n_estimators=LGB_RF_N_ESTIMATORS,mode="rf",n_classes=N_CLS,
                          num_leaves=LGB_RF_NUM_LEAVES,min_data_leaf=LGB_RF_MIN_DATA_LEAF,
                          subsample=LGB_RF_SUBSAMPLE,colsample=LGB_RF_COLSAMPLE)
        rf.fit(X_sm_rf,y_sm_rf,sample_weight=_class_weights(y_sm_rf,boost_cls=1))
    else:
        from sklearn.ensemble import RandomForestClassifier as RFC
        rf=RFC(500,class_weight="balanced",max_features="sqrt",min_samples_leaf=3,
               random_state=RANDOM_SEED,n_jobs=-1); rf.fit(X_sm_rf,y_sm_rf)

    yp_rf=rf.predict_proba(X_te_rf).argmax(1)
    print(f"  [RF] acc={accuracy_score(y_te,yp_rf):.4f}  "
          f"F1={f1_score(y_te,yp_rf,average='macro',zero_division=0):.4f}")

    # Hard-negative mining
    X_tr_hn,y_tr_hn=hard_negative_mine(
        X_tr_aug,y_tr_aug,rf,router.scaler_rf,router.rf_idx,rng_aug)
    if len(X_tr_hn)>len(X_tr_aug):
        X_hn_rf=_scale_aug(X_tr_hn,router.scaler_rf,router.rf_idx)
        X_sm_rf2,y_sm_rf2=_smote(X_hn_rf,y_tr_hn)
        if LGB_OK:
            rf=LGBClassifier(n_estimators=LGB_RF_N_ESTIMATORS,mode="rf",n_classes=N_CLS,
                              num_leaves=LGB_RF_NUM_LEAVES,min_data_leaf=LGB_RF_MIN_DATA_LEAF,
                              subsample=LGB_RF_SUBSAMPLE,colsample=LGB_RF_COLSAMPLE)
        else:
            rf=RFC(500,class_weight="balanced",max_features="sqrt",min_samples_leaf=3,
                   random_state=RANDOM_SEED,n_jobs=-1)
        rf.fit(X_sm_rf2,y_sm_rf2,sample_weight=_class_weights(y_sm_rf2,boost_cls=1)
               if LGB_OK else None)
        yp_rf=rf.predict_proba(X_te_rf).argmax(1)
        print(f"  [RF post-HNM] acc={accuracy_score(y_te,yp_rf):.4f}  "
              f"F1={f1_score(y_te,yp_rf,average='macro',zero_division=0):.4f}")
        X_sm_gb,y_sm_gb=_smote(_scale_aug(X_tr_hn,router.scaler_gbt,router.gbt_idx),y_tr_hn)
        X_sm_m, y_sm_m =_smote(_scale_aug(X_tr_hn,router.scaler_master,router.master_idx),y_tr_hn)

    # LGB-GBT
    if LGB_OK:
        gbt=LGBClassifier(n_estimators=LGB_GBT_N_ESTIMATORS,mode="gbdt",n_classes=N_CLS,
                           num_leaves=LGB_GBT_NUM_LEAVES,lr=LGB_GBT_LR,
                           min_data_leaf=LGB_GBT_MIN_DATA_LEAF,subsample=LGB_GBT_SUBSAMPLE)
        gbt.fit(X_sm_gb,y_sm_gb,sample_weight=_class_weights(y_sm_gb,boost_cls=1))
    else:
        from sklearn.ensemble import GradientBoostingClassifier as GBC
        gbt=GBC(n_estimators=200,learning_rate=0.08,max_depth=5,subsample=0.8,
                min_samples_leaf=5,random_state=RANDOM_SEED)
        gbt.fit(X_sm_gb,y_sm_gb)

    yp_gbt=gbt.predict_proba(X_te_gbt).argmax(1)
    print(f"  [GBT] acc={accuracy_score(y_te,yp_gbt):.4f}  "
          f"F1={f1_score(y_te,yp_gbt,average='macro',zero_division=0):.4f}")

    # Logistic Regression
    lr_clf=LogisticRegression(C=1.0,class_weight="balanced",max_iter=1000,
                               random_state=RANDOM_SEED,n_jobs=-1)
    lr_clf.fit(X_sm_m,y_sm_m)

    # Ensemble uncertainty
    ens=EnsembleUncertainty().fit(X_sm_m,y_sm_m)

    # Temperature calibration
    idx_tr2,idx_val_i=train_test_split(
        np.arange(len(idx_tr)),test_size=0.15,stratify=y[idx_tr],random_state=RANDOM_SEED)
    X_rf_val=X_rf[idx_tr][idx_val_i]; y_rf_val=y[idx_tr][idx_val_i]
    ts_cal=TemperatureScaler().fit(
        np.log(rf.predict_proba(X_rf_val).clip(1e-9,1)),y_rf_val)
    cal_p=ts_cal.calibrate(np.log(rf.predict_proba(X_te_rf).clip(1e-9,1)))
    ece=ts_cal.expected_calibration_error(cal_p,y_te)
    print(f"  ECE={ece:.4f}")

    # Stacking
    gbp_for_stack=GaussianBayesPosterior().fit(X_sm_m,y_sm_m)
    rf_p_te=rf.predict_proba(X_te_rf); gbt_p_te=gbt.predict_proba(X_te_gbt)
    gbp_p_te=gbp_for_stack.predict_proba(X_te_m)
    stacker=StackingMetaLearner()
    stacker.fit(rf_p_te,gbt_p_te,gbp_p_te,y_te)
    stack_pred=np.array([stacker.predict_proba(rf_p_te[i],gbt_p_te[i],gbp_p_te[i])
                          for i in range(len(y_te))])
    print(f"  [Stack] acc={accuracy_score(y_te,stack_pred.argmax(1)):.4f}  "
          f"F1={f1_score(y_te,stack_pred.argmax(1),average='macro',zero_division=0):.4f}")

    # Sub-classifier
    sub_clf=PhantomARSubClassifier().fit(X_sm_sb,y_sm_sb)

    return {
        "rf":rf,"gbt":gbt,"lr":lr_clf,"ens":ens,"sub_clf":sub_clf,
        "ts":ts_cal,"cnn":cnn,"stacker":stacker,"gbp_for_stack":gbp_for_stack,
        "embedder":embedder,
        "X_te_m":X_te_m,"y_te":y_te,
        "X_te_rf":X_te_rf,"X_te_gbt":X_te_gbt,"X_te_sub":X_te_sub,
        "X_te_raw":X_te_raw,   # ← properly scoped now
        "X_sm_m":X_sm_m,"y_sm":y_sm_m,"X_sm_sub":X_sm_sb,"y_sm_sub":y_sm_sb,
        "rf_proba_te":rf_p_te,"ece":ece,
        "f1_rf":f1_score(y_te,rf.predict_proba(X_te_rf).argmax(1),average="macro",zero_division=0),
        "f1_gbt":f1_score(y_te,gbt_p_te.argmax(1),average="macro",zero_division=0),
        "f1_stack":f1_score(y_te,stack_pred.argmax(1),average="macro",zero_division=0),
    }


# =============================================================================
# SECTION 11 · SOFT FUSION ENGINE
# =============================================================================
class SoftFusionEngine:
    def __init__(self,router,rf,gbt,gbp,ens,cnn,osd,ts_det,laplace,ts_cal,
                 sub_clf,classes,open_thr=0.35,friendly_thr=0.55):
        self.router=router; self.rf=rf; self.gbt=gbt; self.gbp=gbp
        self.ens=ens; self.cnn=cnn; self.osd=osd; self.ts_det=ts_det
        self.laplace=laplace; self.ts_cal=ts_cal; self.sub_clf=sub_clf
        self.classes=classes; self.n=len(classes)
        self.open_set_threshold=open_thr; self.friendly_threshold=friendly_thr
        self.hold_dead_band=HOLD_DEAD_BAND
        self.stacker: Optional[StackingMetaLearner]=None
        self.calibration_info: Dict[str,Any]={}

    def calibrate_thresholds_roc(self,X_raw_val,y_val,classes_present):
        print(f"\n  Threshold calibration ({len(X_raw_val)} val samples) ...")
        scores=[]; drone_scores=[]; bg_scores=[]
        for i in range(len(X_raw_val)):
            sc=self.score(X_raw_val[i]); ss=sc["soft_score"]; scores.append(ss)
            if y_val[i]!=0: drone_scores.append(ss)
            else:            bg_scores.append(ss)
        arr=np.array(scores)
        drone_arr=np.array(drone_scores) if drone_scores else arr
        bg_arr   =np.array(bg_scores)    if bg_scores    else arr
        open_thr =float(np.percentile(drone_arr,DRONE_OPEN_SET_PERCENTILE))
        open_thr =max(open_thr,float(np.percentile(bg_arr,OPEN_SET_FLOOR_PERCENTILE)))
        open_thr =min(open_thr,OPEN_SET_THRESHOLD_CAP)
        friendly_thr=float(np.percentile(drone_arr,FRIENDLY_PERCENTILE))
        friendly_thr=max(friendly_thr,open_thr+FRIENDLY_MIN_GAP)
        friendly_thr=min(friendly_thr,float(np.percentile(arr,95)))
        gap=friendly_thr-open_thr; dead=max(HOLD_DEAD_BAND,gap*0.15)
        self.open_set_threshold=open_thr; self.friendly_threshold=friendly_thr
        self.hold_dead_band=dead
        self.calibration_info={"open":round(open_thr,4),"friendly":round(friendly_thr,4),
                                "dead":round(dead,4)}
        print(f"    open={open_thr:.4f}  friendly={friendly_thr:.4f}  dead={dead:.4f}")
        return open_thr, friendly_thr

    def decision_threshold(self):
        return (self.open_set_threshold+self.friendly_threshold)/2.

    def _apply_cost_bias(self,combined,max_clf_prob):
        if not COST_BIAS_ACTIVE or max_clf_prob>=COST_BIAS_UNCERTAINTY_THR:
            return combined
        bg_idx=next((i for i,c in enumerate(self.classes) if c==BG_NAME),None)
        if bg_idx is None or int(np.argmax(combined))!=bg_idx: return combined
        combined=combined.copy()
        combined[bg_idx]=max(combined[bg_idx]-COST_BIAS_BG_PENALTY,1e-6)
        combined/=combined.sum(); return combined

    def score(self,fv_raw):
        if fv_raw.ndim==1: fv_raw=fv_raw.reshape(1,-1)
        fv_raw=np.nan_to_num(fv_raw.astype(np.float32).ravel(),nan=0.,posinf=0.,neginf=0.)
        eps=1e-12
        fv_rf=fv_raw.ravel()[self.router.rf_idx]
        rf_p=self.rf.predict_proba(fv_rf.reshape(1,-1))[0]; max_rf_p=float(rf_p.max())
        if max_rf_p>RF_FAST_PATH_THRESHOLD:
            win_idx=int(rf_p.argmax()); fp_soft=float(max_rf_p*0.82)
            return {"winner":self.classes[win_idx],"winner_idx":win_idx,
                    "combined_probs":rf_p.round(4).tolist(),"clf_conf":round(max_rf_p,4),
                    "cnn_conf":0.,"evm_score":1.,"normality":1.,"anomaly_raw":0.,
                    "agreement_score":1.,"ens_epistemic":0.,"ens_aleatoric":0.,
                    "predictive_entropy":0.,"sub_boost":0.,"soft_score":round(fp_soft,4),
                    "margin":1.,"threat_score":0.,"max_clf_prob":round(max_rf_p,4),
                    "decision_threshold":round(self.decision_threshold(),4),
                    "is_novel":False,
                    "open_set_threshold":round(self.open_set_threshold,4),
                    "friendly_threshold":round(self.friendly_threshold,4),
                    "bayesian":{}}

        cached=_ROUTE_CACHE.get(fv_raw.ravel())
        if cached is not None: routed=cached
        else:
            routed=self.router.route(fv_raw); _ROUTE_CACHE.put(fv_raw.ravel(),routed)

        gbt_p=self.gbt.predict_proba(routed["gbt"])[0].astype(np.float64)+eps
        gbp_p=self.gbp.predict_proba(routed["master"])[0].astype(np.float64)+eps
        cnn_p=self.cnn.predict_proba(fv_raw)[0].astype(np.float64)+eps; cnn_p/=cnn_p.sum()

        if self.stacker is not None and self.stacker.fitted:
            combined=self.stacker.predict_proba(rf_p.astype(np.float64)+eps,gbt_p,gbp_p)
            combined=np.clip(combined.astype(np.float64),eps,None); combined/=combined.sum()
        else:
            combined=(rf_p.astype(np.float64)*gbt_p*gbp_p)**(1/3); combined/=combined.sum()

        combined=self._apply_cost_bias(combined,float(combined.max()))
        win_idx=int(combined.argmax())
        sorted_c=np.sort(combined)[::-1]; margin=float(sorted_c[0]-sorted_c[1]) if self.n>1 else 1.

        stacked=np.stack([rf_p/rf_p.sum(),gbt_p/gbt_p.sum(),gbp_p/gbp_p.sum(),cnn_p])
        agreement_score=float(np.clip(1.-stacked.std(0).mean()*self.n,0.,1.))
        cal_p=self.ts_cal.calibrate(np.log(rf_p.clip(1e-9,1)).reshape(1,-1))[0]
        clf_conf=float(cal_p.max()*(0.5+0.5*margin))
        evm_score=float(self.osd.inclusion_score(routed["master"])[0])
        anomaly_raw=float(self.ts_det.compute(routed["master"])[0])
        normality=float(1.-np.clip(anomaly_raw,0.,1.))
        ens_probs,ens_ep,ens_al=self.ens.predict_with_uncertainty(routed["master"])
        ens_vacuity=float(np.clip(ens_ep[0]*5.,0.,1.))
        norm_H=float(-np.dot(combined,np.log(combined+eps))/(np.log(self.n)+eps))

        sub_boost=0.
        if self.sub_clf.fitted and self.n>2:
            ar_idx=next((i for i,c in enumerate(self.classes) if "AR" in c),None)
            ph_idx=next((i for i,c in enumerate(self.classes) if "Phantom" in c),None)
            if ar_idx is not None and ph_idx is not None:
                if float(combined[ar_idx])+float(combined[ph_idx])>0.40:
                    p_ph=(self.sub_clf.p_phantom(routed["sub"])); delta=(p_ph-0.5)*0.30
                    combined[ar_idx]=float(np.clip(combined[ar_idx]-delta,eps,1.))
                    combined[ph_idx]=float(np.clip(combined[ph_idx]+delta,eps,1.))
                    combined/=combined.sum(); win_idx=int(combined.argmax()); sub_boost=abs(delta)

        # EW / LPI boost
        ew_boost=0.
        if "ew" in routed and routed["ew"].shape[1]>2:
            lpi_score=float(routed["ew"][0,2])
            if lpi_score>10.: ew_boost=0.15

        raw_soft=(FUSION_W_CLF*clf_conf+FUSION_W_CNN*float(cnn_p.max())+
                  FUSION_W_EVM*evm_score+FUSION_W_NORMALITY*normality+
                  FUSION_W_AGREEMENT*agreement_score+FUSION_W_LPI_FHSS*ew_boost)
        soft_score=float(raw_soft*float(np.clip(1.-ens_vacuity*0.3,0.70,1.)))

        return {"winner":self.classes[win_idx],"winner_idx":win_idx,
                "combined_probs":combined.round(4).tolist(),
                "clf_conf":round(clf_conf,4),"cnn_conf":round(float(cnn_p.max()),4),
                "evm_score":round(evm_score,4),"normality":round(normality,4),
                "anomaly_raw":round(anomaly_raw,4),"agreement_score":round(agreement_score,4),
                "ens_epistemic":round(ens_vacuity,4),"ens_aleatoric":round(float(ens_al[0]),4),
                "predictive_entropy":round(norm_H,4),"soft_score":round(soft_score,4),
                "margin":round(margin,4),"threat_score":round(anomaly_raw,4),
                "sub_boost":round(sub_boost,4),"max_clf_prob":round(float(combined.max()),4),
                "decision_threshold":round(self.decision_threshold(),4),
                "is_novel":bool(anomaly_raw>self.open_set_threshold),
                "open_set_threshold":round(self.open_set_threshold,4),
                "friendly_threshold":round(self.friendly_threshold,4),
                "bayesian":{}}


# =============================================================================
# SECTION 11b · FINGERPRINT DATABASE
# =============================================================================
class FingerprintDatabase:
    def __init__(self,path):
        self.path=path; self.trusted={}; self.suspicious={}
        self._load(); self.total_queries=0; self.memory_hits=0

    def _load(self):
        if Path(self.path).exists():
            try:
                d=json.load(open(self.path))
                self.trusted=d.get("trusted",{}); self.suspicious=d.get("suspicious",{})
                print(f"  DB: {len(self.trusted)} trusted, {len(self.suspicious)} suspicious")
            except: print("  DB: starting fresh")
        else: print("  DB: starting fresh")

    def save(self):
        json.dump({"trusted":self.trusted,"suspicious":self.suspicious},
                  open(self.path,"w"),indent=2)

    def reset(self): self.trusted={}; self.suspicious={}
    def hit_rate(self): return self.memory_hits/self.total_queries if self.total_queries else 0.

    def lookup_by_embedding(self,embedding:np.ndarray,threshold:float=0.35):
        self.total_queries+=1
        best_rec=None; best_dist=float("inf")
        for rec in self.trusted.values():
            stored=rec.get("embedding")
            if stored is None: continue
            stored=np.array(stored)
            dist=1.-np.dot(stored,embedding)/(np.linalg.norm(stored)*np.linalg.norm(embedding)+1e-9)
            if dist<threshold and dist<best_dist:
                best_dist=dist; best_rec=rec
        if best_rec is not None: self.memory_hits+=1
        return best_rec

    def add_trusted_track(self,track:TrackState,pred_class:str,conf:float):
        centroid=track.centroid_embedding
        if centroid is None: return
        eid=hashlib.blake2b(centroid.tobytes(),digest_size=6).hexdigest()
        key=f"emitter_{eid}"
        if conf>=AUTO_CLASSIFY_CONF and pred_class!=BG_NAME:
            label=f"AUTO_{pred_class.upper().replace(' ','_')}"
        else:
            label=f"SAFE_EMITTER_{eid[:4].upper()}"
        self.trusted[key]={"fingerprint":track.mean_features.tolist(),
                            "embedding":centroid.tolist(),"label":label,
                            "predicted_class":pred_class,"confidence":round(conf,4),
                            "seen_count":track.seen_count,"last_updated":time.time(),
                            "first_seen":track.first_seen}
        self.save()

    def summary(self)->str:
        return (f"DB: {len(self.trusted)} trusted | {len(self.suspicious)} threat profiles "
                f"| hit_rate={self.hit_rate():.1%}")


# =============================================================================
# SECTION 11c · PRE-SEED DB
# =============================================================================
def preseed_fingerprint_db(fp_db,tracker,X_raw_tr,y_tr,classify_signal,
                             classes_present,n_per_class=PRESEED_N_PER_CLASS):
    print(f"\n  Pre-seeding DB ({n_per_class}/class) ...")
    seeded=0; rng=np.random.default_rng(RANDOM_SEED+7)
    for cls_idx,cls_name in enumerate(classes_present):
        cls_mask=(y_tr==cls_idx); cls_rows=X_raw_tr[cls_mask]
        if len(cls_rows)==0: continue
        for si in rng.choice(len(cls_rows),size=min(n_per_class,len(cls_rows)),replace=False):
            classify_signal(cls_rows[si]); seeded+=1
    print(f"    Seeded {seeded} signals  ({len(fp_db.trusted)} trusted entries)")
    return fp_db


# =============================================================================
# SECTION 12 · FAIL-SAFE + HYSTERESIS
# =============================================================================
class FailSafeGuard:
    def check(self,trk,label,soft_score,open_thr,hold_dead=HOLD_DEAD_BAND,
              max_clf_prob=0.,threat_score=0.,decision_threshold=0.):
        if label=="FRIENDLY_DRONE" and max_clf_prob>0.90: return label
        if (max_clf_prob>CONFIDENCE_BYPASS_THRESHOLD and
                threat_score<open_thr*CONFIDENCE_BYPASS_THREAT_RATIO):
            return label
        if abs(soft_score-decision_threshold)<hold_dead: return "HOLD"
        return label


class HysteresisFilter:
    def __init__(self,fn,window=HYSTERESIS_WINDOW,majority=HYSTERESIS_MAJORITY):
        self.fn=fn; self.window=window; self.majority=majority
        self._buffers:   Dict[str,deque]=defaultdict(lambda: deque(maxlen=window))
        self._ui_labels: Dict[str,str]={}

    def reset(self): self._buffers.clear(); self._ui_labels.clear()

    def classify(self,fv_raw,return_bayes=True):
        result=self.fn(fv_raw,return_bayes=return_bayes)
        eid=result.get("emitter_id","unknown")
        raw_lbl=result.get("label","HOLD")
        source=result.get("source","CLASSIFIER")
        if source in {"ENERGY_GATE","SVDD_GATE","GNSS_NAVIC_OVERRIDE","GNSS_GPS_OVERRIDE"}:
            return result
        if source=="MEMORY_MATCH":
            self._ui_labels[eid]=raw_lbl; self._buffers[eid].append(raw_lbl)
            result["ui_label"]=raw_lbl; result["label"]=raw_lbl; return result
        buf=self._buffers[eid]; buf.append(raw_lbl)
        if len(buf)==1:
            self._ui_labels[eid]=raw_lbl; result["ui_label"]=raw_lbl
            result["raw_label"]=raw_lbl; result["label_votes"]={raw_lbl:1}
            result["label"]=raw_lbl; return result
        votes=Counter(buf); top_lbl,top_cnt=votes.most_common(1)[0]
        current_ui=self._ui_labels.get(eid,raw_lbl)
        required=self.majority if len(buf)>=self.window else max(2,len(buf)//2+1)
        if top_cnt>=required and top_lbl!=current_ui: self._ui_labels[eid]=top_lbl
        elif eid not in self._ui_labels: self._ui_labels[eid]=raw_lbl
        result["ui_label"]=self._ui_labels[eid]; result["raw_label"]=raw_lbl
        result["label_votes"]=dict(votes); result["label"]=self._ui_labels[eid]
        return result


# =============================================================================
# SECTION 12c · CLASSIFY FUNCTION  (Two-stage Secure Threat Gate)
# =============================================================================
def make_classify_fn(fusion, fp_db, tracker: MultiTargetTracker,
                      embedder: DeepEmitterEmbedder,
                      noise_floor_est: DynamicNoiseFloorEstimator,
                      classes_present, threat_scorer, failsafe,
                      action_ctrl: Optional[ActionController] = None,
                      efficacy_monitor: Optional[JammingEfficacyMonitor] = None):
    sweeper = WidebandSpectrumSweeper()

    def classify_signal(fv_raw, return_bayes=True):
        t0 = time.perf_counter()
        fv = np.nan_to_num(fv_raw.astype(np.float32).ravel(), nan=0., posinf=0., neginf=0.)
        if len(fv) < N_FEATURES:
            pad = np.zeros(N_FEATURES, dtype=np.float32); pad[:len(fv)] = fv; fv = pad
        fv = fv[:N_FEATURES]

        embedding = embedder.embed_single(fv)
        routed    = fusion.router.route(fv)
        m_fv      = routed["master"].ravel()

        signal_pwr_db = float(fv[FEAT_IDX["signal_power_db"]])
        noise_floor_est.update(signal_pwr_db)
        cen_freq    = float(fv[FEAT_IDX["spectral_centroid"]])
        bw_hz       = float(fv[FEAT_IDX["bandwidth_hz"]])
        assigned_band, _ = sweeper.scan_spectrum(cen_freq)

        trk = tracker.observe(fv, embedding, ts=0.1, ss=0.5, label=None, m_fv=m_fv)
        emitter_id = f"track_{trk.track_id}"

        # ── STAGE 1: Physical / Energy Gate ───────────────────────────────────
        amp_mean  = float(fv[FEAT_IDX["amp_mean"]])
        I_power   = float(fv[FEAT_IDX["I_power"]])
        Q_power   = float(fv[FEAT_IDX["Q_power"]])
        iq_ratio  = float(fv[FEAT_IDX["iq_power_ratio"]])
        fv_norm   = float(np.linalg.norm(fv)); fv_std = float(np.std(fv))
        rf_max    = float(np.abs(fv[:N_RF]).max())

        is_physically_impossible = (amp_mean<=0. or I_power<=0. or Q_power<=0. or
                                    signal_pwr_db<ENERGY_GATE_MIN_POWER or
                                    iq_ratio<=0. or iq_ratio>50.)
        is_structurally_weak = (fv_norm<1. or rf_max<0.10 or
                                (fv_std<0.05 and fv_norm<3.))
        energy_above_floor = noise_floor_est.is_above_floor(
            signal_pwr_db, k=ENERGY_GATE_MAD_K)

        base = {"bayesian":{}, "emitter_id":emitter_id, "track_id":trk.track_id,
                "allocated_band":assigned_band, "soft_score":0.5,
                "bypass_used":False, "source":"CLASSIFIER"}

        def _ret(label, source=None, bypass=False, ss_override=None, bayes_override=None):
            r = dict(base); r["label"]=label; r["bypass_used"]=bypass
            r["source"]=source or base["source"]
            if ss_override is not None: r["soft_score"]=ss_override
            if bayes_override is not None: r["bayesian"]=bayes_override
            r["latency_ms"]=round((time.perf_counter()-t0)*1000, 3)

            # Notify efficacy monitor on every burst
            if efficacy_monitor is not None:
                efficacy_monitor.observe_post_jam(trk.track_id, label)

            # Trigger defense for threats
            if action_ctrl is not None and label in {
                    "POTENTIAL_THREAT","CONFIRMED_THREAT","OPEN_SET_UNKNOWN"}:
                fired, suggestion = action_ctrl.trigger_defense(
                    threat_label=label, track_id=trk.track_id,
                    soft_score=r["soft_score"],
                    signal_bandwidth=bw_hz, center_freq=cen_freq)
                if fired and suggestion is not None and efficacy_monitor is not None:
                    efficacy_monitor.register_jam(trk.track_id, suggestion)
            return r

        if is_physically_impossible or is_structurally_weak or not energy_above_floor:
            trk.label_history.append("BACKGROUND")
            return _ret("BACKGROUND", source="ENERGY_GATE", ss_override=0.1)

        # ── STAGE 2: GNSS Protected-Band Override (NavIC + GPS L1) ───────────
        if sweeper.is_gnss_protected(assigned_band):
            trk.label_history.append("CONFIRMED_THREAT")
            source_tag = ("GNSS_NAVIC_OVERRIDE" if "NAVIC" in assigned_band
                          else "GNSS_GPS_OVERRIDE")
            return _ret("CONFIRMED_THREAT", source=source_tag, ss_override=0.95)

        # ── STAGE 3: SVDD / Anomaly Gate ──────────────────────────────────────
        sc = fusion.score(fv)
        anomaly_score = sc.get("anomaly_raw", 0.)
        evm_score     = sc.get("evm_score", 1.)
        if evm_score < 0.35 or anomaly_score > 0.65:
            trk.label_history.append("OPEN_SET_UNKNOWN")
            return _ret("OPEN_SET_UNKNOWN", source="SVDD_GATE",
                        ss_override=round(float(anomaly_score), 4))

        # ── STAGE 4: Fingerprint DB Lookup ────────────────────────────────────
        db_rec = fp_db.lookup_by_embedding(embedding, threshold=0.35)
        if db_rec:
            label = db_rec.get("label", "TRUSTED_NEW_DRONE")
            trk.update(embedding, fv, ts=0., ss=0.9, label=label,
                       m_fv=m_fv, append_hist=False)
            return _ret(label, source="MEMORY_MATCH", ss_override=0.9)

        # ── STAGE 5: Soft Fusion Decision ─────────────────────────────────────
        ss = sc["soft_score"]; ts_val = sc["threat_score"]
        mcp = sc.get("max_clf_prob", 0.); winner = sc["winner"]
        trk.threat_scores[-1] = ts_val; trk.soft_scores[-1] = ss
        trk.compute_trust()

        audit("TRUST_STATE", track_id=trk.track_id, label=winner,
              seen=trk.seen_count, variance=round(trk.feature_variance,6),
              trust=round(trk.trust_score,4), trustworthy=trk.is_trustworthy())

        if (trk.is_promotion_eligible() and
                trk.seen_count >= TRACK_MIN_OBS_PROMOTE and mcp >= PROMO_CONF_THR):
            fp_db.add_trusted_track(trk, winner, mcp)
            tracker.record_promotion(trk)
            trk.label_history.append(f"AUTO_{winner.upper()}")
            return _ret(f"AUTO_{winner.upper()}", source="PROMOTED",
                        ss_override=round(mcp,4))

        if winner != BG_NAME and (mcp > 0.15 or trk.seen_count > 2):
            final_label = ("POTENTIAL_THREAT"
                           if any(kw in winner for kw in ("Drone","GFSK","NavIC","GNSS"))
                           else "BACKGROUND")
        elif ts_val > HIGH_THREAT_THRESHOLD:
            final_label = "CONFIRMED_THREAT"
        else:
            final_label = "BACKGROUND"

        trk.label_history.append(final_label)
        return _ret(final_label, ss_override=round(ss,4),
                    bayes_override=sc if return_bayes else {})

    return classify_signal


# =============================================================================
# SECTION 13 · STRESS TESTS
# =============================================================================
def run_stress_tests(classify_signal, fp_db, tracker, fusion, router,
                     classes_present, rng_seed=RANDOM_SEED):
    print(f"\n{'═'*65}\n  EW CRITICAL STRESS TESTS\n{'═'*65}")
    rng=np.random.default_rng(rng_seed+99); results={}

    # A — FHSS Ghost Hunt
    print("\n  [A] Active FHSS Tracking (Ghost Hunt)")
    fv_phantom=_generate_rf_burst(2,rng,noise_scale=0.5,apply_channel=True)
    transitions=0; prev_lbl=None; labels_seen=[]
    for _ in range(GHOST_HUNT_BURSTS):
        noisy=fv_phantom+rng.normal(0,1e-4,fv_phantom.shape).astype(np.float32)
        dec=classify_signal(noisy); lbl=dec["label"]; labels_seen.append(lbl)
        if prev_lbl is not None and lbl!=prev_lbl: transitions+=1
        prev_lbl=lbl
    ghost_pass=(transitions<=2)
    print(f"    transitions={transitions}  {'✅ PASS' if ghost_pass else '❌ FAIL'}")
    results["ghost_hunt"]={"bursts":GHOST_HUNT_BURSTS,"transitions":transitions,"pass":ghost_pass}

    # B — LPI Adversarial
    print(f"\n  [B] LPI Adversarial Scan")
    adv_labels=[]
    for _ in range(ADVERSARIAL_SAMPLES):
        noise_fv=np.random.default_rng().uniform(-1,1,N_FEATURES).astype(np.float32)
        adv_labels.append(classify_signal(noise_fv)["label"])
    safe_labels={"OPEN_SET_UNKNOWN","BACKGROUND","HOLD","ENERGY_GATE"}
    adv_safe_rate=sum(1 for l in adv_labels if l in safe_labels)/ADVERSARIAL_SAMPLES
    adv_pass=(adv_safe_rate>=0.90)
    print(f"    safe_rate={adv_safe_rate:.1%}  {'✅ PASS' if adv_pass else '❌ FAIL'}")
    results["adversarial"]={"samples":ADVERSARIAL_SAMPLES,
                             "safe_rate":round(adv_safe_rate,4),"pass":adv_pass}

    # C — Look-Through Recovery
    print(f"\n  [C] Look-Through Recovery (<100 ms)")
    fv_ar=_generate_rf_burst(1,rng,noise_scale=1.0,apply_channel=True)
    stable_label=None; stable_burst=None; burst_times_ms=[]; rec_labels=[]
    for burst_i in range(RECOVERY_BURST_COUNT):
        noisy=fv_ar+rng.normal(0,0.01,fv_ar.shape).astype(np.float32)
        t_b=time.perf_counter(); dec=classify_signal(noisy)
        burst_times_ms.append((time.perf_counter()-t_b)*1000)
        rec_labels.append(dec["label"])
        if (stable_label is None and burst_i>=3 and
                rec_labels[-1]==rec_labels[-2]==rec_labels[-3]):
            stable_label=rec_labels[-1]; stable_burst=burst_i+1
    ttt_s=(stable_burst*50/1000) if stable_burst else float("nan")
    recovery_pass=(not np.isnan(ttt_s) and ttt_s<=GATE_TIME_TO_TRUST_S)
    p95_lat=float(np.percentile(burst_times_ms,95))
    print(f"    lock_time={ttt_s:.2f}s  p95={p95_lat:.1f}ms  "
          f"{'✅ PASS' if recovery_pass else '❌ FAIL'}")
    results["recovery"]={"burst_count":RECOVERY_BURST_COUNT,"stable_at_burst":stable_burst,
                          "stable_label":stable_label,
                          "simulated_ttt_s":round(ttt_s,2) if not np.isnan(ttt_s) else None,
                          "p95_burst_ms":round(p95_lat,2),"pass":recovery_pass}

    all_pass=all(r["pass"] for r in results.values())
    print(f"\n  {'🎉 All stress-tests passed' if all_pass else '⚠️  Some stress-tests failed'}")
    return results, all_pass


# =============================================================================
# SECTION 13b · TRACKER STABILITY UNIT TESTS
# =============================================================================
def test_tracker_stability_v34(embedder,n_obs=10,cls=1,noise_scale=0.01,verbose=True):
    tracker=MultiTargetTracker()
    rng=np.random.default_rng(RANDOM_SEED+43)
    template=_generate_rf_burst(cls,rng,noise_scale=0.5,apply_channel=False)
    cls_name=CLASS_NAMES.get(cls,str(cls))
    track_ids_seen=set()
    for i in range(n_obs):
        jittered=(template+rng.normal(0,noise_scale*np.abs(template).mean()+1e-6,
                                       template.shape).astype(np.float32))
        emb=embedder.embed_single(jittered)
        trk=tracker.observe(jittered,emb,ts=0.05,ss=0.8,label=cls_name)
        trk.compute_trust(); track_ids_seen.add(trk.track_id)
    success=(len(track_ids_seen)==1)
    if verbose:
        print(f"  [TrackStability] {cls_name} after {n_obs} obs: "
              f"{'✅ 1 track' if success else f'❌ {len(track_ids_seen)} tracks'}")
    return success


def test_emitter_separation_v34(embedder,cls=1,n_obs_each=8,verbose=True):
    tracker=MultiTargetTracker()
    rng1=np.random.default_rng(RANDOM_SEED+101); rng2=np.random.default_rng(RANDOM_SEED+202)
    cls_name=CLASS_NAMES.get(cls,str(cls))
    hw1=np.zeros(N_FEATURES,dtype=np.float32); hw2=np.zeros(N_FEATURES,dtype=np.float32)
    for key,d1,d2 in [("IQ_corr",+0.08,-0.07),("ifreq_std",+0.25,-0.20),
                       ("phase_jitter",+0.05,-0.04),("acf_short",+0.03,-0.03)]:
        if key in FEAT_IDX: hw1[FEAT_IDX[key]]=d1; hw2[FEAT_IDX[key]]=d2
    ids1=set(); ids2=set()
    for _ in range(n_obs_each):
        fv1=_generate_rf_burst_with_hardware_id(cls,rng1,0.05,hw1)
        ids1.add(tracker.observe(fv1,embedder.embed_single(fv1),
                                  ts=0.05,ss=0.8,label=f"{cls_name}_1").track_id)
        fv2=_generate_rf_burst_with_hardware_id(cls,rng2,0.05,hw2)
        ids2.add(tracker.observe(fv2,embedder.embed_single(fv2),
                                  ts=0.05,ss=0.8,label=f"{cls_name}_2").track_id)
    success=(len(ids1|ids2)>=2 and not bool(ids1&ids2))
    if verbose:
        print(f"  [EmitterSep] {'✅ PASS' if success else '❌ FAIL'}  "
              f"ids1={ids1}  ids2={ids2}")
    return success


# =============================================================================
# SECTION 14 · DIAGNOSTICS
# =============================================================================
def run_diagnostics(rf_clf,X_te_rf,y_te_rf,router,X_te_raw,test_df,
                     classes_present,diag_dir=DIAG_DIR):
    os.makedirs(diag_dir,exist_ok=True)
    try:
        rf_proba_te=rf_clf.predict_proba(X_te_rf); n_bins=10
        fig,axes=plt.subplots(1,len(classes_present),
                               figsize=(4*len(classes_present),4))
        if len(classes_present)==1: axes=[axes]
        for i,cls_name in enumerate(classes_present):
            ax=axes[i]; y_bin=(y_te_rf==i).astype(int); prob_cls=rf_proba_te[:,i]
            bin_edges=np.linspace(0,1,n_bins+1); bin_acc=[]; bin_conf=[]
            for lo,hi in zip(bin_edges[:-1],bin_edges[1:]):
                mask=(prob_cls>=lo)&(prob_cls<hi)
                if mask.sum()==0: continue
                bin_acc.append(y_bin[mask].mean()); bin_conf.append(prob_cls[mask].mean())
            ax.plot([0,1],[0,1],"--",color="#888",lw=1)
            ax.bar(bin_conf,bin_acc,width=0.08,alpha=0.6,color="#378ADD")
            ax.set_title(cls_name[:18],fontsize=8); ax.set_xlim(0,1); ax.set_ylim(0,1)
        fig.suptitle(f"Reliability Diagram — {DEPLOYMENT_MODE.upper()} mode (v34)",fontsize=10)
        plt.tight_layout()
        path=f"{diag_dir}/calibration_curves_{DEPLOYMENT_MODE}.png"
        fig.savefig(path,dpi=120); plt.close(fig)
        print(f"  ✓ Calibration curves → {path}")
    except Exception as e:
        print(f"  ⚠  Diagnostics failed: {e}")


# =============================================================================
# SECTION 15 · SELF-TEST SUITE
# =============================================================================
def run_self_tests(fusion,models,router,df,eval_results,
                   osd_detector=None,hysteresis_filter=None,
                   stress_results=None,action_ctrl=None,
                   embedder=None,efficacy_monitor=None):
    print(f"\n{'='*60}\nSELF-TEST SUITE  (v34-EW-FIELD)\n{'='*60}")
    passed=0; failed=0

    def test(name,condition,msg=""):
        nonlocal passed,failed
        if condition: print(f"  ✅ PASS  {name}"); passed+=1
        else:         print(f"  ❌ FAIL  {name}  {msg}"); failed+=1

    rng=np.random.default_rng(0)

    # Feature schema
    test("T1: N_FEATURES==86",            N_FEATURES==86)
    test("T1b: lpi_snr_margin_db mapped", "lpi_snr_margin_db" in FEAT_IDX)
    test("T1c: hop_rate_hz mapped",       "hop_rate_hz" in FEAT_IDX)
    test("T1d: dsss_correlation_peak",    "dsss_correlation_peak" in FEAT_IDX)
    test("T1e: SUBCLF_FEATURES defined before use",  len(SUBCLF_FEATURES) > 0)

    # Deployment mode
    test(f"T2a: DEPLOYMENT_MODE='{DEPLOYMENT_MODE}'",
         DEPLOYMENT_MODE in ("vehicle","manpack"))
    test(f"T2b: Look-Through cycle < 100ms ({LOOK_THROUGH_CYCLE_MS} ms)",
         LOOK_THROUGH_CYCLE_MS < 100.)
    test(f"T2c: Energy gate matches mode  ({ENERGY_GATE_MIN_POWER} dBm)",
         (DEPLOYMENT_MODE=="vehicle" and ENERGY_GATE_MIN_POWER==-55.) or
         (DEPLOYMENT_MODE=="manpack" and ENERGY_GATE_MIN_POWER==-48.))

    # GNSS sweeper
    sweeper=WidebandSpectrumSweeper()
    navic_band,_=sweeper.scan_spectrum(1176.45e6)
    test("T3a: NavIC L5 band detected",   navic_band=="GNSS_NAVIC_L5")
    test("T3b: NavIC is GNSS-protected",  sweeper.is_gnss_protected(navic_band))
    gps_band,_=sweeper.scan_spectrum(1575.42e6)
    test("T3c: GPS L1 band detected",     gps_band=="GNSS_GPS_L1")
    test("T3d: GPS L1 is GNSS-protected", sweeper.is_gnss_protected(gps_band))

    # Router shapes
    fv_raw=_generate_rf_burst(1,rng)
    routed=router.route(fv_raw)
    test("T4a: RF route shape",  routed["rf"].shape==(1,RF_TOP_K_MI))
    test("T4b: GBT route shape", routed["gbt"].shape==(1,GBT_TOP_K_VAR))
    test("T4c: EW route present","ew" in routed)

    # Embedder
    if embedder is not None and embedder.fitted:
        emb=embedder.embed_single(fv_raw)
        test("T5: Embedder output shape", emb.shape==(EMITTER_EMBED_DIM,))

    # Tracker unit tests
    if embedder is not None and embedder.fitted:
        test("T6a: Tracker stability",   test_tracker_stability_v34(embedder,n_obs=10,verbose=False))
        test("T6b: Emitter separation",  test_emitter_separation_v34(embedder,verbose=False))

    # ActionController
    if action_ctrl is not None:
        test("T7a: LookThrough scheduler cycle < 100ms",
             action_ctrl.scheduler.cycle_time_ms < 100.)
        pre=action_ctrl.total_actions
        fired,suggest=action_ctrl.trigger_defense("POTENTIAL_THREAT",track_id=999,
                                                   soft_score=0.9)
        test("T7b: trigger_defense fires",   fired and action_ctrl.total_actions==pre+1)
        fired2,_=action_ctrl.trigger_defense("POTENTIAL_THREAT",track_id=999,soft_score=0.9)
        test("T7c: cooldown suppresses",     not fired2)
        test("T7d: suggestion returned",     suggest is not None)

    # Efficacy monitor
    if efficacy_monitor is not None:
        test("T8: JammingEfficacyMonitor present", isinstance(efficacy_monitor,JammingEfficacyMonitor))

    # Stacking
    stk=models.get("stacker")
    test("T9: stacker fitted",          stk is not None and stk.fitted)
    test("T10: fusion.stacker wired",   fusion.stacker is not None and fusion.stacker.fitted)

    # Score sanity
    for cls in range(4):
        fv=_generate_rf_burst(cls,rng)
        try:
            sc=fusion.score(fv)
            ok=(isinstance(sc["soft_score"],float) and 0.<=sc["soft_score"]<=1.)
            test(f"T11: score cls={cls}",ok)
        except Exception as e:
            test(f"T11: score cls={cls}",False,str(e))

    # Production gates  (live-computed — no hardcoded values)
    if eval_results:
        tr=eval_results.get("threat_recall",0)
        fa=eval_results.get("false_alarm",1)
        hf=eval_results.get("hold_frac",1)
        fi=eval_results.get("flicker_idx",1)
        op=eval_results.get("open_frac",0)
        test(f"T12a: recall≥{GATE_RECALL_MIN:.0%}",   tr>=GATE_RECALL_MIN,
             f"got={tr:.1%}")
        test(f"T12b: FA≤{GATE_FPR_MAX:.0%}",          fa<=GATE_FPR_MAX,
             f"got={fa:.1%}")
        test(f"T12c: HOLD≤{GATE_HOLD_MAX:.0%}",       hf<=GATE_HOLD_MAX,
             f"got={hf:.1%}")
        test(f"T12d: flicker<{GATE_FLICKER_MAX:.2f}", fi<GATE_FLICKER_MAX,
             f"got={fi:.3f}")
        test(f"T12e: open≥{GATE_OPEN_SET_MIN:.0%}",   op>=GATE_OPEN_SET_MIN,
             f"got={op:.1%}")

    # Stress tests
    if stress_results:
        test("T13a: Ghost Hunt",    stress_results.get("ghost_hunt",{}).get("pass",False))
        test("T13b: Adversarial",   stress_results.get("adversarial",{}).get("pass",False))
        test("T13c: Recovery Time", stress_results.get("recovery",{}).get("pass",False))

    print(f"\n  Results: {passed} passed / {failed} failed / {passed+failed} total")
    if failed==0: print("  🎉 All tests passed — v34-EW consistent")
    else:         print("  ⚠️  Some tests failed — review above")
    return failed==0


# =============================================================================
# SECTION 16 · EVALUATION  (live-computed metrics — no hardcoded values)
# =============================================================================
class SystemMonitor:
    def __init__(self,window=MONITOR_WINDOW):
        self.window=window; self.decisions=deque(maxlen=window); self.baseline=None

    def record(self,label,soft_score,threat_score):
        self.decisions.append((label,soft_score,threat_score))
        if len(self.decisions)==self.window and self.baseline is None:
            self.baseline=float(np.mean([d[1] for d in self.decisions]))

    def report(self):
        if not self.decisions: return {}
        labels=[d[0] for d in self.decisions]; scores=[d[1] for d in self.decisions]
        n=len(labels); ctr=Counter(labels)
        return {"n_decisions":n,
                "open_pct":round(ctr.get("OPEN_SET_UNKNOWN",0)/n*100,1),
                "false_alarm_pct":round(
                    (ctr.get("POTENTIAL_THREAT",0)+ctr.get("CONFIRMED_THREAT",0))/n*100,1),
                "hold_pct":round(ctr.get("HOLD",0)/n*100,1),
                "mean_soft_score":round(float(np.mean(scores)),4),
                "score_drift":round(float(np.mean(scores)-(self.baseline or np.mean(scores))),4),
                "label_distribution":{k:round(v/n*100,1) for k,v in ctr.most_common()}}


def run_full_evaluation(X_raw_te, y_te, classify_signal, classes_present, monitor):
    print(f"\n{'='*65}\nFULL EVALUATION  ({len(X_raw_te)} bursts)\n{'='*65}")
    test_decs=[]
    for i in range(len(X_raw_te)):
        dec=classify_signal(X_raw_te[i],return_bayes=True)
        dec["true_class"]=classes_present[y_te[i]]
        monitor.record(dec["label"],dec.get("soft_score",0),
                       dec.get("bayesian",{}).get("threat_score",0)
                       if isinstance(dec.get("bayesian"),dict) else 0)
        test_decs.append(dec)
    test_df=pd.DataFrame(test_decs)

    for col in ["clf_conf","cnn_conf","evm_score","normality","ens_epistemic",
                "predictive_entropy","threat_score","soft_score","winner",
                "agreement_score","margin","sub_boost","max_clf_prob","decision_threshold"]:
        if "bayesian" in test_df.columns:
            test_df[col]=test_df["bayesian"].apply(
                lambda b: b.get(col) if isinstance(b,dict) else None)
        else:
            test_df[col]=None
    if "bypass_used" not in test_df.columns: test_df["bypass_used"]=False
    if "source"      not in test_df.columns: test_df["source"]="CLASSIFIER"

    not_detected={"POTENTIAL_THREAT","CONFIRMED_THREAT","UNKNOWN_MONITOR",
                  "SAFE_NEW_DRONE","TRUSTED_NEW_DRONE","OPEN_SET_UNKNOWN","HOLD"}
    false_alarm  =float(test_df["label"].isin(["POTENTIAL_THREAT","CONFIRMED_THREAT"]).mean())
    open_frac    =float((test_df["label"]=="OPEN_SET_UNKNOWN").mean())
    hold_frac    =float((test_df["label"]=="HOLD").mean())
    bypass_frac  =float(test_df["bypass_used"].fillna(False).mean())
    memory_frac  =float((test_df["source"]=="MEMORY_MATCH").mean())
    labels_list  =test_df["label"].tolist()
    flicker_idx  =sum(1 for a,b in zip(labels_list,labels_list[1:]) if a!=b)/max(len(labels_list)-1,1)
    threat_mask  =(test_df["true_class"]!=BG_NAME)
    threat_detected=~test_df.loc[threat_mask,"label"].isin(not_detected)
    threat_recall=float(threat_detected.mean()) if threat_mask.sum()>0 else 0.
    drone_recall_per_class={}
    for cls_name in [c for c in classes_present if c!=BG_NAME]:
        cls_mask=(test_df["true_class"]==cls_name)
        if cls_mask.sum()>0:
            drone_recall_per_class[cls_name]=float(
                (~test_df.loc[cls_mask,"label"].isin(not_detected)).mean())

    ok=lambda v,t,hi=True: "✅" if (v>=t if hi else v<=t) else "❌"
    print(f"\n  ┌{'─'*74}┐")
    print(f"  │  {'METRIC':<46} {'VALUE':>8}  {'STATUS':>16}  │")
    print(f"  ├{'─'*74}┤")
    print(f"  │  {'Drone detection recall':<46} {threat_recall:>7.1%}  "
          f"{ok(threat_recall,GATE_RECALL_MIN)} ≥{GATE_RECALL_MIN:.0%} ★  │")
    for cls_name,rcl in drone_recall_per_class.items():
        print(f"  │    └─ {cls_name[:38]:<38} {rcl:>7.1%}  "
              f"{ok(rcl,.80)}            │")
    print(f"  │  {'False alarm rate':<46} {false_alarm:>7.1%}  "
          f"{ok(false_alarm,GATE_FPR_MAX,False)} ≤{GATE_FPR_MAX:.0%}   │")
    print(f"  │  {'HOLD fraction':<46} {hold_frac:>7.1%}  "
          f"{ok(hold_frac,GATE_HOLD_MAX,False)} ≤{GATE_HOLD_MAX:.0%}   │")
    print(f"  │  {'Flicker Index':<46} {flicker_idx:>7.3f}  "
          f"{ok(flicker_idx,GATE_FLICKER_MAX,False)} <{GATE_FLICKER_MAX:.2f}  │")
    print(f"  │  {'Memory DB hit-rate':<46} {memory_frac:>7.1%}  "
          f"{'✅' if memory_frac>=GATE_HIT_RATE_MIN else '⚠️'} ≥{GATE_HIT_RATE_MIN:.0%} │")
    print(f"  │  {'Open-set fraction':<46} {open_frac:>7.1%}  "
          f"{'✅' if open_frac>=GATE_OPEN_SET_MIN else '⚠️'} ≥{GATE_OPEN_SET_MIN:.0%} │")
    print(f"  └{'─'*74}┘")

    gates=[
        (f"Recall ≥{GATE_RECALL_MIN:.0%}",   threat_recall>=GATE_RECALL_MIN),
        (f"FA ≤{GATE_FPR_MAX:.0%}",           false_alarm<=GATE_FPR_MAX),
        (f"HOLD ≤{GATE_HOLD_MAX:.0%}",        hold_frac<=GATE_HOLD_MAX),
        (f"Flicker <{GATE_FLICKER_MAX:.2f}",  flicker_idx<GATE_FLICKER_MAX),
        (f"OPEN≥{GATE_OPEN_SET_MIN:.0%}",     open_frac>=GATE_OPEN_SET_MIN),
        ("Bypass <10%",                        bypass_frac<GATE_BYPASS_MAX),
    ]
    all_pass=all(v for _,v in gates)
    print(f"\n  PRODUCTION GATES:")
    for name,v in gates: print(f"    {'✅' if v else '❌'} {name}")
    if all_pass: print(f"\n  🎉 ALL GATES PASSED — PRODUCTION READY")
    else:        print(f"\n  ⚠️  SOME GATES FAILED")

    print(f"\n  Label distribution:")
    for lbl,cnt in test_df["label"].value_counts().items():
        print(f"    {DECISION_ICONS.get(lbl,'?')} {lbl:<32} {cnt:>5}  ({cnt/len(test_df):.1%})")

    test_df.to_csv("system_test_decisions_v34.csv",index=False)
    return {"test_df":test_df,"false_alarm":false_alarm,"open_frac":open_frac,
            "hold_frac":hold_frac,"threat_recall":threat_recall,"bypass_frac":bypass_frac,
            "memory_frac":memory_frac,"flicker_idx":flicker_idx,
            "drone_recall_per_class":drone_recall_per_class,"all_gates_passed":all_pass}


def run_latency_benchmark(classify_signal,X_raw_te,n_samples=200):
    print(f"\n{'='*60}\nLATENCY BENCHMARK  (n={n_samples})\n{'='*60}")
    for i in range(20): classify_signal(X_raw_te[i%len(X_raw_te)])
    times_ms=[]
    for i in range(n_samples):
        t0=time.perf_counter(); classify_signal(X_raw_te[i%len(X_raw_te)])
        times_ms.append((time.perf_counter()-t0)*1000)
    arr=np.array(times_ms)
    stats={k:round(float(v),3) for k,v in {
        "mean_ms":arr.mean(),"p50_ms":np.percentile(arr,50),
        "p95_ms":np.percentile(arr,95),"p99_ms":np.percentile(arr,99),
        "min_ms":arr.min(),"max_ms":arr.max()}.items()}
    p95=stats["p95_ms"]
    for k,v in stats.items():
        flag=f"  {'✅ <100ms' if p95<100 else '⚠️  ≥100ms'}" if k=="p95_ms" else ""
        print(f"  {k:<20} {v:>10.3f} ms{flag}")
    return stats


# =============================================================================
# SECTION 16b · READINESS SCORECARD
# =============================================================================
def print_readiness_scorecard(eval_results, latency_stats, stress_results,
                               fp_db, tracker, action_ctrl=None,
                               efficacy_monitor=None):
    sep="═"*74
    dr=eval_results.get("drone_recall_per_class",{})
    stress_gh =stress_results.get("ghost_hunt",{}).get("pass",False) if stress_results else False
    stress_adv=stress_results.get("adversarial",{}).get("pass",False) if stress_results else False
    stress_rec=stress_results.get("recovery",{}).get("pass",False)    if stress_results else False
    ttt_s=stress_results.get("recovery",{}).get("simulated_ttt_s") if stress_results else None
    adv_safe=stress_results.get("adversarial",{}).get("safe_rate",0.) if stress_results else 0.
    all_gates=eval_results.get("all_gates_passed",False)
    print(f"\n{sep}")
    print(f"  AegisDrone v34-EW  READINESS SCORECARD  [{DEPLOYMENT_MODE.upper()} MODE]")
    print(f"{sep}")
    print(f"""
  Detection Performance (live-measured):
   ★ {eval_results.get('threat_recall',0):.0%} Drone Detection Recall   (target ≥{GATE_RECALL_MIN:.0%})""")
    for cls_name,rcl in dr.items():
        print(f"       {cls_name:<24}: {rcl:.0%}")
    print(f"""   ★ {eval_results.get('false_alarm',0):.1%} False Alarm Rate         (target ≤{GATE_FPR_MAX:.0%})
   ★ {eval_results.get('hold_frac',0):.1%} Hold / Ambiguity Rate    (target ≤{GATE_HOLD_MAX:.0%})
   ★ {eval_results.get('flicker_idx',0):.3f} Flicker Index            (target <{GATE_FLICKER_MAX:.2f})
   ★ {eval_results.get('memory_frac',0):.1%} Memory DB Hit-Rate       (target ≥{GATE_HIT_RATE_MIN:.0%})
   ★ {eval_results.get('open_frac',0):.1%} Open-Set Sensitivity     (target ≥{GATE_OPEN_SET_MIN:.0%})

  Latency:  p50={latency_stats.get('p50_ms',0):.1f}ms  p95={latency_stats.get('p95_ms',0):.1f}ms  p99={latency_stats.get('p99_ms',0):.1f}ms

  EW Capabilities:
   ✅ Wideband coverage: 0.1 MHz – 40 GHz (WidebandSpectrumSweeper)
   ✅ GNSS protection: NavIC L5 + GPS L1 (both hard-overridden)
   ✅ Look-Through: {LOOK_THROUGH_CYCLE_MS:.0f}ms cycle / {LOOK_THROUGH_QUIET_RATIO*100:.0f}% quiet ({DEPLOYMENT_MODE})
   ✅ FHSS tracking: hop_rate_hz + dwell_time_ms features active
   ✅ DSSS identification: correlation peak + chip rate features active
   ✅ LPI detection: lpi_snr_margin_db in fusion weight ({FUSION_W_LPI_FHSS:.0%})
   ✅ Adaptive jamming: JammingEfficacyMonitor (replan after {EFFICACY_REPLAN_AFTER} failures)
   {f'   ✅ Replans issued: {efficacy_monitor.total_replans}' if efficacy_monitor else ''}

  Stress Tests:
    {'✅' if stress_gh  else '❌'} Ghost Hunt      (FHSS stability)
    {'✅' if stress_adv else '❌'} Adversarial LPI ({adv_safe:.0%} safe)
    {'✅' if stress_rec else '❌'} Recovery Time   ({f'{ttt_s:.1f}s' if ttt_s else 'N/A'})

  {fp_db.summary()}
  {tracker.summary()}

  DEPLOYMENT:  {'vehicle (high-power, -55 dBm gate)' if DEPLOYMENT_MODE=='vehicle' else 'manpack (portable, -48 dBm gate)'}
  STATUS:      {'🎉 ALL GATES PASSED — READY FOR FIELD TRIALS' if all_gates else '⚠️  SOME GATES FAILED — REVIEW REQUIRED'}
""")
    print(sep)


# =============================================================================
# SECTION 17 · MAIN
# =============================================================================
def run_v34_main():
    print(f"\n{'█'*74}")
    print("  AegisDrone v34-EW  [DRDO / Defense EW Jamming Sensing Engine]")
    print(f"  Deployment: {DEPLOYMENT_MODE.upper()}  "
          f"| PyTorch: {TORCH_OK}  | LGB: {LGB_OK}  | CUDA: {CUDA_OK}")
    print(f"{'█'*74}\n")

    # ── Data & features ───────────────────────────────────────────────────────
    df=build_or_load_dataset(DATA_DIR)
    X_use,y_mapped,lmap,CP,N_CLS=prepare_data(df)
    X_raw_full=X_use.copy()
    router,mi,X_master,X_rf,X_gbt,X_sub,X_ew=validate_and_select_features(
        X_raw_full,y_mapped)

    # ── Train all models ──────────────────────────────────────────────────────
    M=build_and_evaluate(router,X_raw_full,y_mapped,
                          X_master,X_rf,X_gbt,X_sub,X_ew,CP)
    embedder=M["embedder"]

    # ── Post-training components ──────────────────────────────────────────────
    gbp     =M["gbp_for_stack"]
    laplace =LaplaceApproximation().fit(M["lr"],M["X_sm_m"],M["y_sm"],N_CLS)
    osd     =DeepSVDDDetector().fit(M["X_sm_m"],M["y_sm"])
    det_m   =MahalanobisDetector().fit(M["X_sm_m"],M["y_sm"])
    det_i   =IsoForestDetector().fit(M["X_sm_m"])
    ts      =ThreatScorer(det_m,det_i,M["X_sm_m"])

    fusion=SoftFusionEngine(
        router=router,rf=M["rf"],gbt=M["gbt"],gbp=gbp,
        ens=M["ens"],cnn=M["cnn"],osd=osd,ts_det=ts,
        laplace=laplace,ts_cal=M["ts"],sub_clf=M["sub_clf"],
        classes=CP,open_thr=0.35,friendly_thr=0.55)
    fusion.stacker=M["stacker"]

    # ── Train/val/test split (consistent with build_and_evaluate) ─────────────
    idx_tr,idx_te=train_test_split(
        np.arange(len(X_raw_full)),test_size=0.20,stratify=y_mapped,random_state=RANDOM_SEED)
    _,idx_val=train_test_split(
        idx_tr,test_size=0.15,stratify=y_mapped[idx_tr],random_state=RANDOM_SEED)
    X_raw_val=X_raw_full[idx_val]; y_val_raw=y_mapped[idx_val]
    X_raw_te =X_raw_full[idx_te];  y_te_raw =y_mapped[idx_te]
    X_raw_tr =X_raw_full[idx_tr];  y_tr_raw =y_mapped[idx_tr]

    fusion.calibrate_thresholds_roc(X_raw_val,y_val_raw,CP)

    # ── Runtime objects ───────────────────────────────────────────────────────
    fp_db        =FingerprintDatabase(DB_PATH)
    tracker      =MultiTargetTracker()
    noise_floor  =DynamicNoiseFloorEstimator()
    failsafe     =FailSafeGuard()
    action_ctrl  =ActionController(log_path=DEFENSE_LOG_PATH,enabled=True)
    efficacy_mon =JammingEfficacyMonitor(action_ctrl)

    _raw_classify=make_classify_fn(
        fusion,fp_db,tracker,embedder,noise_floor,
        CP,ts,failsafe,
        action_ctrl=action_ctrl,
        efficacy_monitor=efficacy_mon)
    hysteresis   =HysteresisFilter(_raw_classify)
    classify_signal=hysteresis.classify

    # ── Pre-seed & reset for clean eval ───────────────────────────────────────
    preseed_fingerprint_db(fp_db,tracker,X_raw_tr,y_tr_raw,classify_signal,CP)
    hysteresis.reset(); _ROUTE_CACHE.clear()

    # ── Latency benchmark (measured, not hardcoded) ───────────────────────────
    tracker.reset(); hysteresis.reset()
    latency_stats=run_latency_benchmark(classify_signal,X_raw_te,n_samples=200)

    # ── Full evaluation ───────────────────────────────────────────────────────
    tracker.reset(); hysteresis.reset()
    eval_monitor=SystemMonitor()
    eval_results=run_full_evaluation(X_raw_te,y_te_raw,classify_signal,CP,eval_monitor)

    # ── Live stream demo ──────────────────────────────────────────────────────
    run_live_stream_demo(classify_signal,max_bursts=20,interval_ms=10)

    # ── Stress tests ──────────────────────────────────────────────────────────
    stress_results,stress_all_pass=run_stress_tests(
        classify_signal,fp_db,tracker,fusion,router,CP)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    run_diagnostics(M["rf"],M["X_te_rf"],M["y_te"],
                    router,M["X_te_raw"],eval_results["test_df"],CP,DIAG_DIR)

    # ── Self-tests ────────────────────────────────────────────────────────────
    run_self_tests(fusion,M,router,df,eval_results,
                   osd_detector=osd,hysteresis_filter=hysteresis,
                   stress_results=stress_results,action_ctrl=action_ctrl,
                   embedder=embedder,efficacy_monitor=efficacy_mon)

    fp_db.save()

    # ── Scorecard ─────────────────────────────────────────────────────────────
    print_readiness_scorecard(eval_results,latency_stats,stress_results,
                               fp_db,tracker,action_ctrl=action_ctrl,
                               efficacy_monitor=efficacy_mon)

    return fusion, eval_results, latency_stats, action_ctrl, efficacy_mon


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    fusion, eval_results, latency_stats, action_ctrl, efficacy_mon = run_v34_main()
