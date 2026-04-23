"""
generate.py — NSB (Navier-Stokes-Brinkman) dataset generation

Solves the NSB PDE for random parameter samples.
Each solve is appended to HDF5 immediately — crash-safe.

Usage:
    python -m pdr_recovery.data_generation.nsb.generate \
        --dim 11 --key aff_S3 --level 3 --nb_train 2000 --seed 42

Output:
    DATASETS/nsb/{key}_d{dim}_level{level}/
        train.h5        — training data (incremental, crash-safe)
        test.h5         — test data (sparse grid)
        failures.log    — failed solves (if any)
"""

import os, gc, argparse, time
os.environ["DIJITSO_CACHE_DIR"] = os.path.expanduser("~/.cache/dijitso")
os.environ["CPLUS_INCLUDE_PATH"] = "/usr/common/sc-conda-envs/fenics-tf-gpu/include/eigen3:" + os.environ.get("CPLUS_INCLUDE_PATH", "")
import numpy as np
import h5py
import Tasmanian
from dolfin import *
from .PDE_data_NSB import gen_dirichlet_data_NSB


# ──────────────────────────────────────────
#  HDF5 helpers (u and p for NSB)
# ──────────────────────────────────────────

def create_h5(path, dim, n_dofs_u, n_dofs_p):
    """Create an empty HDF5 file with resizable datasets."""
    with h5py.File(path, "w") as f:
        f.create_dataset("coeff_u", shape=(0, n_dofs_u), maxshape=(None, n_dofs_u), dtype="f8")
        f.create_dataset("coeff_p", shape=(0, n_dofs_p), maxshape=(None, n_dofs_p), dtype="f8")
        f.create_dataset("params",  shape=(0, dim),       maxshape=(None, dim),      dtype="f8")
        f.create_dataset("norm_u",  shape=(0,),           maxshape=(None,),          dtype="f8")
        f.create_dataset("norm_p",  shape=(0,),           maxshape=(None,),          dtype="f8")


def append_h5(path, coeff_u, coeff_p, params, norm_u, norm_p):
    """Append one sample to the HDF5 file."""
    with h5py.File(path, "a") as f:
        for name, val in [("coeff_u", coeff_u), ("coeff_p", coeff_p), ("params", params)]:
            ds = f[name]
            ds.resize(ds.shape[0] + 1, axis=0)
            ds[-1] = val

        for name, val in [("norm_u", norm_u), ("norm_p", norm_p)]:
            ds = f[name]
            ds.resize(ds.shape[0] + 1, axis=0)
            ds[-1] = val


def log_failure(log_path, index, z, error_msg):
    """Append a failure line to a plain text log."""
    with open(log_path, "a") as f:
        f.write(f"sample {index} | error: {error_msg} | z: {z.tolist()}\n")


# ──────────────────────────────────────────
#  Solve loop (used for both train and test)
# ──────────────────────────────────────────

def generate_samples(y_points, h5_path, fail_log, start_idx,
                     mesh, Hh, example, dim, label=""):
    """
    Solve PDE for each row of y_points, append results to h5_path.
    Returns number of successes and failures.
    """
    n_total = y_points.shape[0]
    n_fail  = 0

    for i in range(start_idx, n_total):
        z = y_points[i]
        t0 = time.time()

        try:
            _, p_c, u_c, n_u, n_p, _ = gen_dirichlet_data_NSB(z, mesh, Hh, example, dim)
            append_h5(h5_path, u_c, p_c, z, n_u, n_p)

            if (i + 1) % 100 == 0:
                print(f"  {label} [{i+1}/{n_total}] OK  ({time.time()-t0:.1f}s)")

        except Exception as e:
            print(f"  {label} [{i+1}/{n_total}] FAILED: {e}")
            log_failure(fail_log, i, z, str(e))
            n_fail += 1
            continue

        if (i + 1) % 200 == 0:
            gc.collect()

    n_success = n_total - start_idx - n_fail
    return n_success, n_fail


# ──────────────────────────────────────────
#  Main
# ──────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate NSB PDE dataset")
    parser.add_argument("--dim",      type=int, default=11)
    parser.add_argument("--level",    type=int, default=3)
    parser.add_argument("--key",      type=str, default="aff_S3",
                        help="Coefficient type: aff_S3")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--nb_train", type=int, default=2000)
    args = parser.parse_args()

    np.random.seed(args.seed)
    dim      = args.dim
    level    = args.level
    example  = args.key
    meshname = "meshes/poisson.xml"

    # Output folder
    outdir = os.path.join("DATASETS", "nsb", f"{example}_d{dim}_level{level}")
    os.makedirs(outdir, exist_ok=True)
    fail_log = os.path.join(outdir, "failures.log")

    # ── FEniCS setup ──
    mesh = Mesh(meshname)
    deg  = 1
    Ht   = VectorElement("DG", mesh.ufl_cell(), deg + 1, dim=3)
    Hsig = FiniteElement("BDM", mesh.ufl_cell(), deg + 1)
    Hu   = VectorElement("DG", mesh.ufl_cell(), deg)
    Hgam = FiniteElement("DG", mesh.ufl_cell(), deg)
    Hh   = FunctionSpace(mesh, MixedElement([Hu, Ht, Hsig, Hsig, Hgam]))

    Hu_space = FunctionSpace(mesh, Hu)
    Ph       = FunctionSpace(mesh, 'DG', 0)

    n_dofs_u = Hu_space.dim()
    n_dofs_p = Ph.dim()
    print(f"DOFs — u: {n_dofs_u}, p: {n_dofs_p}")

    # ══════════════════════════════════════
    #  Training data
    # ══════════════════════════════════════

    m_train    = args.nb_train
    y_train    = -1.0 + 2.0 * np.random.rand(m_train, dim)
    train_path = os.path.join(outdir, "train.h5")

    # Resume support
    start_idx = 0
    if os.path.exists(train_path):
        with h5py.File(train_path, "r") as f:
            start_idx = f["coeff_u"].shape[0]
        print(f"Resuming training from sample {start_idx}")
    else:
        create_h5(train_path, dim, n_dofs_u, n_dofs_p)

    print(f"\nGenerating {m_train - start_idx} training samples...")
    ok, fail = generate_samples(
        y_train, train_path, fail_log, start_idx,
        mesh, Hh, example, dim, label="Train"
    )
    print(f"Training done: {ok} succeeded, {fail} failed")

    # ══════════════════════════════════════
    #  Test data — Clenshaw-Curtis sparse grid
    # ══════════════════════════════════════

    grid = Tasmanian.SparseGrid()
    grid.makeGlobalGrid(dim, 0, level, "level", "clenshaw-curtis")
    y_test  = grid.getPoints()
    weights = grid.getQuadratureWeights()
    m_test  = y_test.shape[0]

    test_path = os.path.join(outdir, "test.h5")
    create_h5(test_path, dim, n_dofs_u, n_dofs_p)

    # Store quadrature weights
    with h5py.File(test_path, "a") as f:
        f.create_dataset("weights", data=weights)

    print(f"\nGenerating {m_test} test samples (sparse grid)...")
    ok, fail = generate_samples(
        y_test, test_path, fail_log, 0,
        mesh, Hh, example, dim, label="Test"
    )
    print(f"Test done: {ok} succeeded, {fail} failed")

    print("\nDone.")