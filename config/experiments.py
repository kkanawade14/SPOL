"""
experiments.py
--------------
All experiment configs in one place.  Plain dicts — nothing fancy.

To add a new experiment:
    1. Copy one of the blocks below.
    2. Change the values.
    3. Register it in EXPERIMENTS at the bottom.

To add a new variable to an existing experiment:
    Just add an entry to its "variables" dict.
"""

import numpy as np

# ──────────────────────────────────────────
#  Boussinesq (3D, box mesh)
# ──────────────────────────────────────────

BOUSSINESQ = {
    "folder":       "boussinesq",       # folder name in DATASETS/ and RESULTS/
    "dim":          8,
    "meshname":     "box3d.xml",
    "seed":         15,
    "pmax":         40,
    "max_iter":     2500,
    "tol":          1e-8,
    "m_schedule":   [100, 500, 1000],   # must be ascending
    "total_trials": 3,
    "valid_keys":   ["logKL", "aff_S3", "aff_F9"],

    # Which variables can we recover, and how?
    #   coeff_key / norm_key  : suffixes in the .mat files
    #   norm_type             : "l4" (FEniCS CPU) or "l2" (GPU via mass matrix)
    #   space                 : FEniCS space used for norm computation
    "variables": {
        "u": {
            "coeff_key": "coeff_u",
            "norm_key":  "norm_u",
            "norm_type": "l4",
            "space":     "vector_dg1",
        },
        "p": {
            "coeff_key": "coeff_p",
            "norm_key":  "norm_p",
            "norm_type": "l2",
            "space":     "cg1",
        },
        "phi": {
            "coeff_key": "coeff_phi",
            "norm_key":  "norm_phi",
            "norm_type": "l4",
            "space":     "cg1",
        },
    },
}

# ──────────────────────────────────────────
#  Navier–Stokes–Brinkman (2D, Poisson mesh)
# ──────────────────────────────────────────

NSB = {
    "folder":       "nsb",
    "dim":          11,
    "meshname":     "meshes/poisson.xml",
    "seed":         15,
    "pmax":         13,
    "max_iter":     2000,
    "tol":          1e-10,
    "m_schedule":   [500],
    "total_trials": 1,
    "valid_keys":   ["aff_S3"],

    "variables": {
        "u": {
            "coeff_key": "coeff_u",
            "norm_key":  "norm_u",
            "norm_type": "l4",
            "space":     "vector_dg1",
        },
        "p": {
            "coeff_key": "coeff_p",
            "norm_key":  "norm_p",
            "norm_type": "l2",
            "space":     "dg0",
        },
    },
}

# ──────────────────────────────────────────
#  Registry — maps CLI name → config dict
# ──────────────────────────────────────────

EXPERIMENTS = {
    "bsnq": BOUSSINESQ,
    "nsb":  NSB,
}


def get_experiment(name):
    """Look up an experiment config by CLI name. Validates m_schedule."""
    name = name.lower()
    if name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{name}'. Choose from: {list(EXPERIMENTS.keys())}"
        )
    cfg = EXPERIMENTS[name]

    # Sanity check: schedule must be strictly ascending
    ms = cfg["m_schedule"]
    if any(ms[i] >= ms[i + 1] for i in range(len(ms) - 1)) and len(ms) > 1:
        raise ValueError(
            f"m_schedule must be strictly ascending, got {ms}"
        )
    return cfg


def build_data_dir(folder, key, dim, level):
    """
    Construct the dataset path.
    Example: DATASETS/boussinesq/logKL_d8_level3/
    """
    import os
    return os.path.join("DATASETS", folder, f"{key}_d{dim}_level{level}")


def build_results_dir(folder, variable, key, dim):
    """
    Construct the results path.
    Example: RESULTS/boussinesq/u/logKL_d8/20260422_143012/
    """
    import os
    from datetime import datetime
    return os.path.join(
        "RESULTS", folder, variable,
        f"{key}_d{dim}", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
