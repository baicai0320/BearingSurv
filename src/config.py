"""Runtime configuration for the public BearingSurv XJTU-SY release.

The public pipeline replays fixed FPT scores and runs eight-model
cross-validation. Feature preparation happens inside the training coordinator.
All values below are fixed experiment settings.
"""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "XJTU-SY"
RAW_DATA_PATH_XJTU = (
    DATA_DIR / "35Hz12kN",
    DATA_DIR / "37.5Hz11kN",
    DATA_DIR / "40Hz10kN",
)
OUTPUT_DIR = ROOT_DIR / "outputs"
FPT_LABELS_DIR = OUTPUT_DIR / "fpt_v2_a_coverage"
FPT_LABELS_PATH = FPT_LABELS_DIR / "fpt_v2_fold_labels.csv"
FPT_MANIFEST_PATH = FPT_LABELS_DIR / "fpt_v2_a_coverage_manifest.json"

DATASET_NAME = "xjtu"
CONDITIONS = (0, 1, 2)
CONDITION_NAMES = ("35Hz12kN", "37.5Hz11kN", "40Hz10kN")
BEARING_IDS = (1, 2, 3, 4, 5)
CENSORING_LEVELS = (0.25, 0.50, 0.75)
RANDOM_SEED = 0
FPT_RANDOM_SEED = 42
MOVING_AVERAGE_WINDOW_BY_CONDITION = {0: 2, 1: 4, 2: 6}

# Fixed FPTv2-A profiles selected before this public release. A fold uses one
# profile for all three conditions. The coverage rule only rescues an
# ``insufficient_groups`` primary miss with a stable three-band consensus.
FPT_V2_A_COVERAGE_PROFILES = {
    1: {
        "shared": {"hidden_dim": 16, "latent_dim": 4, "epochs": 200, "learning_rate": 1e-3, "batch_size": 32, "dropout_rate": 0.1, "mc_samples": 30, "score_batch_size": 256},
        "detector": {"healthy_window": 15, "smoothing_window": 3, "drift_sigma": 2.5, "threshold_sigma": 10.0, "persistence": 1, "score_mode": "composite", "reconstruction_weight": 0.5, "epistemic_weight": 0.5, "log_transform": True, "z_clip": 25.0, "fusion": "median", "min_triggered_groups": 2, "consensus_mode": "onset_cluster", "consensus_absolute_spread": 5.0, "consensus_relative_spread": 0.1, "max_relative_mad": 0.1},
    },
    2: {
        "shared": {"hidden_dim": 16, "latent_dim": 4, "epochs": 200, "learning_rate": 1e-3, "batch_size": 32, "dropout_rate": 0.3, "mc_samples": 50, "score_batch_size": 256},
        "detector": {"healthy_window": 20, "smoothing_window": 5, "drift_sigma": 3.5, "threshold_sigma": 15.0, "persistence": 1, "score_mode": "composite", "reconstruction_weight": 0.75, "epistemic_weight": 0.25, "log_transform": True, "z_clip": 25.0, "fusion": "median", "min_triggered_groups": 2, "consensus_mode": "onset_cluster", "consensus_absolute_spread": 10.0, "consensus_relative_spread": 0.2, "max_relative_mad": 0.1},
    },
    3: {
        "shared": {"hidden_dim": 16, "latent_dim": 4, "epochs": 200, "learning_rate": 1e-3, "batch_size": 32, "dropout_rate": 0.1, "mc_samples": 15, "score_batch_size": 256},
        "detector": {"healthy_window": 20, "smoothing_window": 5, "drift_sigma": 3.0, "threshold_sigma": 10.0, "persistence": 1, "score_mode": "composite", "reconstruction_weight": 0.5, "epistemic_weight": 0.5, "log_transform": True, "z_clip": 25.0, "fusion": "median", "min_triggered_groups": 2, "consensus_mode": "onset_cluster", "consensus_absolute_spread": 5.0, "consensus_relative_spread": 0.15, "max_relative_mad": 0.1},
    },
    4: {
        "shared": {"hidden_dim": 16, "latent_dim": 4, "epochs": 200, "learning_rate": 1e-3, "batch_size": 32, "dropout_rate": 0.3, "mc_samples": 15, "score_batch_size": 256},
        "detector": {"healthy_window": 20, "smoothing_window": 7, "drift_sigma": 4.0, "threshold_sigma": 15.0, "persistence": 2, "score_mode": "composite", "reconstruction_weight": 0.25, "epistemic_weight": 0.75, "log_transform": True, "z_clip": 25.0, "fusion": "median", "min_triggered_groups": 2, "consensus_mode": "onset_cluster", "consensus_absolute_spread": 5.0, "consensus_relative_spread": 0.1, "max_relative_mad": 0.1},
    },
    5: {
        "shared": {"hidden_dim": 16, "latent_dim": 4, "epochs": 200, "learning_rate": 1e-3, "batch_size": 32, "dropout_rate": 0.3, "mc_samples": 30, "score_batch_size": 256},
        "detector": {"healthy_window": 20, "smoothing_window": 7, "drift_sigma": 4.0, "threshold_sigma": 10.0, "persistence": 1, "score_mode": "composite", "reconstruction_weight": 0.5, "epistemic_weight": 0.5, "log_transform": True, "z_clip": 25.0, "fusion": "median", "min_triggered_groups": 2, "consensus_mode": "onset_cluster", "consensus_absolute_spread": 10.0, "consensus_relative_spread": 0.2, "max_relative_mad": 0.1},
    },
}

FPT_V2_A_COVERAGE_RESCUE = {
    "reconstruction_weight": 0.75,
    "epistemic_weight": 0.25,
    "smoothing_window": 7,
    "drift_sigma": 4.0,
    "threshold_sigma": 10.0,
    "persistence": 1,
    "fusion": "trimmed_mean",
    "min_triggered_groups": 3,
    "consensus_absolute_spread": 5.0,
    "consensus_relative_spread": 0.15,
    "primary_miss_status": "insufficient_groups",
}

# These values define the label-generation contract.  Keep this payload
# separate from the complete runtime configuration: survival-model changes
# must not silently invalidate an otherwise identical FPT artifact.
FPT_FEATURE_GROUPS = ("FTF", "Fr", "BSF", "BPFO", "BPFI")
FPT_FEATURE_DIRECTIONS = ("H", "V")
FPT_FEATURE_TYPES = ("amplitude_hilbert",)
FPT_SIGNAL_LENGTH = 32768


def fpt_configuration_payload() -> dict:
    """Return the canonical, JSON-serializable FPT label contract."""
    return {
        "dataset": DATASET_NAME,
        "conditions": list(CONDITIONS),
        "condition_names": list(CONDITION_NAMES),
        "bearing_ids": list(BEARING_IDS),
        "random_seed": int(FPT_RANDOM_SEED),
        "signal_length": int(FPT_SIGNAL_LENGTH),
        "feature_groups": list(FPT_FEATURE_GROUPS),
        "feature_directions": list(FPT_FEATURE_DIRECTIONS),
        "feature_types": list(FPT_FEATURE_TYPES),
        "profiles": FPT_V2_A_COVERAGE_PROFILES,
        "coverage_rescue": FPT_V2_A_COVERAGE_RESCUE,
    }

# One canonical numerical profile, extracted from the paper's main revision.
# Compatibility aliases do not create a second set of training parameters.
from paper.config import (
    PARAMS_R14F_OTSURV as PARAMS_R14F,
    REPORT_EPOCHS as OTSURV_REPORT_EPOCHS,
    TRAIN_EPOCHS as OTSURV_TRAIN_EPOCHS,
)

PUBLIC_MODELS = (
    "otsurv", "rawcnn", "coxph", "coxboost", "rsf", "mtlr", "dsm", "bnnsurv",
)
