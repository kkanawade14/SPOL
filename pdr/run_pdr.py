"""
run_pdr.py
----------
Single entry point for all PDR experiments.

    python -m SPOL.pdr.run_pdr --experiment bsnq --variable u --m_schedule 100 500 1000
    python -m SPOL.pdr.run_pdr --experiment nsb  --variable p --m_schedule 500
    python -m SPOL.pdr.run_pdr --experiment bsnq --variable phi --m_schedule 100 500 1000

Replaces:  BSNQ_PDR_U.py, BSNQ_PDR_P.py, BSNQ_PDR_phi.py, NSB_PDR_U.py, NSB_PDR_P.py
"""

import os, math, gc, argparse, yaml
os.environ["DIJITSO_CACHE_DIR"] = os.path.expanduser("~/.cache/dijitso")
from datetime import datetime
import numpy as np
import cupy as cp
import scipy.io as sio

from config.experiments import get_experiment, build_data_dir, build_results_dir
from utils.data_io import load_data
from utils.fenics_setup import load_mesh, build_norm_function, build_mass_diagonal
from pdr.legendre import multiidx_gen, build_design_matrix, hyperbolic_cross_rule
from pdr.norms import compute_l4_norms, compute_l2_norms
from pdr.solvers import PDR_gpu


# ──────────────────────────────────────────
#  Norm-function builders
# ──────────────────────────────────────────

def make_l4_norm_fn(fn):
    """Build a norm_fn closure for L4 (CPU roundtrip via FEniCS)."""
    def norm_fn(x_gpu):
        return cp.asarray(compute_l4_norms(cp.asnumpy(x_gpu), fn))
    return norm_fn


def make_l2_norm_fn(V):
    """Build a norm_fn closure for L2 (GPU, mass-matrix diagonal)."""
    diag = build_mass_diagonal(V)
    W_gpu = cp.asarray(np.diag(diag))  # sparse would be better for large problems
    def norm_fn(x_gpu):
        Mx = W_gpu @ x_gpu.T
        return cp.sqrt(cp.sum(x_gpu.T * Mx, axis=0))
    return norm_fn


# ──────────────────────────────────────────
#  Main
# ──────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PDR recovery experiment")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name: bsnq, nsb")
    parser.add_argument("--variable",   type=str, required=True,
                        help="Variable to recover: u, p, phi")
    parser.add_argument("--key",        type=str, required=True,
                        help="Coefficient key: logKL, aff_S3, aff_F9")
    parser.add_argument("--level",      type=int, default=3,
                        help=" sparse grid level")
    parser.add_argument("--m_schedule", type=int, nargs="+", default=None,
                        help="Training sample schedule (ascending)")
    parser.add_argument("--seed",       type=int, default=None)
    parser.add_argument("--total_trials", type=int, default=None)
    args = parser.parse_args()

    # ── Load config ──
    cfg = get_experiment(args.experiment)
    var_cfg = cfg["variables"][args.variable]

    # Validate key
    if args.key not in cfg["valid_keys"]:
        raise ValueError(
            f"Key '{args.key}' not valid for {args.experiment}. "
            f"Options: {cfg['valid_keys']}"
        )

    dim        = cfg["dim"]
    meshname   = cfg["meshname"]
    seed       = args.seed or cfg["seed"]
    pmax       = cfg["pmax"]
    max_iter   = cfg["max_iter"]
    tol        = cfg["tol"]
    m_schedule = args.m_schedule or cfg["m_schedule"]
    trials     = args.total_trials or cfg["total_trials"]
    data_dir   = build_data_dir(cfg["folder"], args.key, dim, args.level)
    norm_type  = var_cfg["norm_type"]
    m_train = sum(m_schedule)

    # Validate schedule
    if len(m_schedule) > 1:
        for i in range(len(m_schedule) - 1):
            if m_schedule[i] >= m_schedule[i + 1]:
                raise ValueError(f"m_schedule must be strictly ascending: {m_schedule}")

    # Validate data exists
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. "
            f"Run generate.py with --key {args.key} --dim {dim} --level {args.level} first."
        )

    # ── Output directory ──
    outdir = build_results_dir(cfg["folder"], args.variable, args.key, dim)
    os.makedirs(outdir)

    # Save config for reproducibility
    with open(os.path.join(outdir, "config.yaml"), "w") as f:
        yaml.safe_dump({
            "experiment": args.experiment, "variable": args.variable,
            "key": args.key, "level": args.level,
            "dim": dim, "seed": seed, "pmax": pmax, "max_iter": max_iter,
            "tol": tol, "m_schedule": m_schedule, "trials": trials,
            "norm_type": norm_type, "meshname": meshname,
            "data_dir": data_dir,
    }, f, sort_keys=False)

    # ── Load data ──
    # load_data auto-detects HDF5 vs legacy .mat
    coeff_key = var_cfg["coeff_key"]   # e.g. "coeff_u"
    norm_key  = var_cfg["norm_key"]    # e.g. "norm_u"

    print("Loading training data...")
    train = load_data(data_dir, "train", coeff_key, norm_key)
    print("Loading test data...")
    test = load_data(data_dir, "test", coeff_key, norm_key)

    train_sol = train["solutions"]   # (m_pool, K)
    train_X   = train["params"]      # (m_pool, dim)
    test_sol  = test["solutions"]    # (m_test, K)
    test_X    = test["params"]       # (m_test, dim)
    test_W    = test["weights"]      # (m_test,)
    test_norms_true = test["norms"]  # (m_test,)

    K       = train_sol.shape[1]

    print(f"Train: {train_sol.shape} | Test: {test_sol.shape} | K={K}")

    # ── FEniCS setup ──
    mesh = load_mesh(meshname)
    V, fn = build_norm_function(mesh, var_cfg["space"])
    print(f"FEniCS space ({var_cfg['space']}): {V.dim()} DOFs")

    if norm_type == "l4":
        norm_fn = make_l4_norm_fn(fn)
        test_error_fn = lambda err: compute_l4_norms(err, fn)
    elif norm_type == "l2":
        norm_fn = make_l2_norm_fn(V)
        test_error_fn = lambda err: compute_l2_norms(err, fn)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

    # ── Legendre basis ──
    Lambda = multiidx_gen(dim, HCfunc, pmax).astype(int)
    N = len(Lambda)
    print('Generated multi index set Lambda of shape', Lambda.shape)
    print('Using', N, 'basis functions with HC index set')
    d_pow = 2.0 ** dim

    for trial in range(trials):
        current_seed = seed + trial
    
        print(f"\n{'='*60}")
        print(f"Trial {trial + 1}/{trials} | seed={current_seed}")
        print(f"{'='*60}")

        rng = np.random.default_rng(current_seed)
    
        pool = np.arange(m_train)
        rng.shuffle(pool)
    
        prev_m = 0
        for m in m_schedule:
            
            new_picks = pool[prev_m:m] 
            all_picked = pool[:m]      
            print(f"\n--- m={m} ({bsize} new samples) ---")

            # Build design matrix
            A, weights = build_design_matrix(train_X[all_picked, :], Lambda)
            A = A / np.sqrt(m)
            b = train_sol[all_picked, :] / np.sqrt(m)

            # Move to GPU
            A_gpu = cp.asarray(A)
            b_gpu = cp.asarray(b)
            w_gpu = cp.asarray(weights)

            del A, b
            gc.collect()

            A_norm = float(cp.linalg.norm(A_gpu, 2).item())
            b_norm = float(cp.linalg.norm(b_gpu, 2).item())

            # Solver parameters
            lamb1 = 1.0 / np.sqrt(25 * m)
            step = 1.0 / A_norm
            r = math.e ** (-1)
            T = int(np.ceil(2 * A_norm / r))
            s = T / (2 * A_norm)
            R = int(np.ceil(max_iter / T))

            print(f"  A_norm={A_norm:.2f} | T={T} | R={R}")

            
            cbar, rel_errors = PDR_gpu(
                A_gpu, b_gpu, w_gpu, lamb1, step, step,
                T, R, tol, r, s, b_norm, m, K, N, norm_fn
            )

            # Free GPU memory
            del A_gpu, b_gpu, w_gpu
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

            # ── Evaluate on test set ──
            Psi_test, _ = build_design_matrix(test_X, Lambda)
            Y_pred = Psi_test @ cbar
            del Psi_test
            gc.collect()

            err = Y_pred - test_sol
            err_norms = test_error_fn(err)

            # Weighted relative error
            numerator = np.sqrt(np.abs(np.dot(err_norms ** 2, test_W)) / d_pow)
            denom     = np.sqrt(np.abs(np.dot(test_norms_true ** 2, test_W)) / d_pow)
            rel_err   = numerator / denom

            print(f"  Relative {norm_type.upper()} error: {rel_err:.6e}")

            # ── Save ──
            sid = min(42, test_sol.shape[0] - 1)  
            sio.savemat(os.path.join(outdir, f"trial_{trial+1}_m{m}.mat"), {
                "cbar": cbar, "Lambda": Lambda,
                "m": m, "dim": dim, "N_basis": N, "pmax": pmax,
                "rel_err": rel_err, "norm_type": norm_type,
                "true_sample": test_sol[sid],
                "pred_sample": Y_pred[sid],
                "sample_idx": sid,
                "rel_errors_history": np.array(rel_errors),
            })

            prev_m = m
            del Y_pred, cbar
            gc.collect()

        seed += 1  # different seed per trial

    print("\nDone.")


