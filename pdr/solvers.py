"""
solvers.py
----------
Primal-dual SR-LASSO with restarts.

Two functions only:
    PD_srlasso_gpu  — inner primal-dual loop
    PDR_gpu         — restart wrapper

The norm type (L4 via FEniCS, L2 via mass matrix) is injected via `norm_fn`.
This makes the solver completely experiment-agnostic.

Usage:
    # L4 norm (velocity, temperature)
    from pdr_recovery.pdr.norms import compute_l4_norms
    norm_fn = lambda x_gpu: cp.asarray(compute_l4_norms(cp.asnumpy(x_gpu), uh))

    # L2 norm (pressure, GPU-only)
    def norm_fn(x_gpu):
        Mx = P_WEIGHTS_gpu @ x_gpu.T
        return cp.sqrt(cp.sum(x_gpu.T * Mx, axis=0))

    cbar, errors = PDR_gpu(A, b, weights, ..., norm_fn=norm_fn)
"""

import cupy as cp
import numpy as np
from . norms import compute_l4_norms, compute_l2_norms

def PD_srlasso_gpu(A, b, weights, lamb1, tau, sigma, T, c, xi, N, norm_fn):
    """
    Inner primal-dual SR-LASSO loop.

    Parameters
    ----------
    A, b      : cupy arrays — (m, N) measurement matrix and (m, K) data
    weights   : cupy array, shape (N,) — sparsity weights
    lamb1     : float — regularisation parameter
    tau, sigma: float — primal / dual step sizes
    T         : int   — number of inner iterations
    c, xi     : cupy arrays — initial primal (N, K) / dual (m, K)
    N         : int   — number of basis functions
    norm_fn   : callable — maps cupy (n, K) → cupy (n,) norms
    

    Returns
    -------
    cbar       : cupy array (N, K) — ergodic average
    rel_errors : list[float]
    """
    cbar = c.copy()
    bnorm = float(cp.linalg.norm(b))
    rel_errors = []
    

    for n in range(T):
        # ── Primal step ──
        p = c - tau * (A.T @ xi)
        p_norm = norm_fn(p)

        ST = cp.zeros(N)
        mask = p_norm > 0
        ST[mask] = cp.maximum(
            p_norm[mask] - tau * lamb1 * weights[mask], 0
        ) / p_norm[mask]
        c_new = ST[:, cp.newaxis] * p

        # ── Dual step ──
        q = xi + sigma * (A @ (2 * c_new - c)) - sigma * b
        q_norm = norm_fn(q)

        ST_q = cp.ones_like(q_norm)
        mask_q = q_norm > 1.0
        ST_q[mask_q] = 1.0 / q_norm[mask_q]
        xi_new = ST_q[:, cp.newaxis] * q

        # ── Ergodic average ──
        cbar = (n / (n + 1)) * cbar + (1.0 / (n + 1)) * c_new

        # ── Track residual ──
        rel_errors.append(float(cp.linalg.norm(A @ cbar - b) / bnorm))

        c = c_new
        xi = xi_new

    return cbar, rel_errors


def PDR_gpu(A, b, weights, lamb1, tau, sigma, T, R,
            tol, r, s, eps0, m, K, N, norm_fn):
    """
    Primal-dual with restarts.

    Parameters
    ----------
    A, b      : cupy arrays
    weights   : cupy array, shape (N,)
    lamb1     : float
    tau, sigma: float
    T         : int   — inner iterations per restart
    R         : int   — max restarts
    tol       : float — convergence tolerance
    r, s      : float — restart scaling parameters
    eps0      : float — initial epsilon (typically ||b||)
    m, K, N   : int   — dimensions
    norm_fn   : callable

    Returns
    -------
    cbar_cpu   : np.ndarray (N, K)
    rel_errors : list[float] — concatenated across all restarts
    """
    cbar = cp.zeros((N, K))
    xi = cp.zeros((m, K))
    rel_errors = []
    eps_old = eps0

    for restart in range(R):
        eps_new = r * (eps_old + tol)
        a_old = s * eps_new

        cbar_new, errs = PD_srlasso_gpu(
            A, b / a_old, weights, lamb1, tau, sigma, T,
            cbar / a_old, xi * 0, N, norm_fn
        )
        rel_errors.extend(errs)

        cbar_prev = cbar.copy()
        cbar = cbar_new * a_old
        eps_old = eps_new

        # Early stopping
        change = float(cp.linalg.norm(cbar - cbar_prev))
        if change <= 5 * tol:
            print(f"  Early stop at restart {restart + 1}/{R} "
                  f"(change={change:.2e})")
            break

    return cp.asnumpy(cbar), rel_errors
