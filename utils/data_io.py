"""
data_io.py
----------
Load training / test data from HDF5 files produced by the generators.
Works for both Boussinesq and NSB — same key names.

HDF5 layout (written by both generators):
    coeff_u, coeff_p, [coeff_phi]   — solution coefficients
    params                          — parameter samples
    norm_u, norm_p, [norm_phi]      — precomputed norms
    weights                         — quadrature weights (test only)

All arrays are (n_samples, n_dofs) — batch-first.
"""

import os
import h5py
import numpy as np


def load_data(data_dir, split, coeff_key, norm_key):
    """
    Load train or test data from HDF5.

    Parameters
    ----------
    data_dir  : str — folder, e.g. "DATASETS/boussinesq/logKL_d8_level3"
    split     : str — "train" or "test"
    coeff_key : str — e.g. "coeff_u", "coeff_p", "coeff_phi"
    norm_key  : str — e.g. "norm_u", "norm_p", "norm_phi"

    Returns
    -------
    dict with keys:
        "solutions" : (n_samples, n_dofs)
        "params"    : (n_samples, dim)
        "norms"     : (n_samples,)
        "weights"   : (n_test,)  — test only
    """
    path = os.path.join(data_dir, f"{split}.h5")

    if not os.path.exists(path):
        raise FileNotFoundError(f"No {split}.h5 found in {data_dir}")

    print(f"  Loading {path}")
    with h5py.File(path, "r") as f:
        out = {
            "solutions": f[coeff_key][:],
            "params":    f["params"][:],
            "norms":     f[norm_key][:],
        }
        if split == "test" and "weights" in f:
            out["weights"] = f["weights"][:]

    print(f"  {split}: solutions {out['solutions'].shape}, "
          f"params {out['params'].shape}, norms {out['norms'].shape}")
    return out


def list_datasets(data_dir):
    """Print what's inside an HDF5 file for debugging."""
    for split in ["train", "test"]:
        path = os.path.join(data_dir, f"{split}.h5")
        if not os.path.exists(path):
            print(f"  {split}.h5 — not found")
            continue
        with h5py.File(path, "r") as f:
            n = f["params"].shape[0]
            keys = list(f.keys())
            print(f"  {split}.h5 — {n} samples, keys: {keys}")