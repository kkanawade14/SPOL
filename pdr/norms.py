"""
norms.py
--------
FEniCS-based norm computations.
    compute_l4_norms  — for velocity / temperature (CPU, via FEniCS assembly)
    compute_l2_norms  — for pressure (CPU, via FEniCS assembly)

These are used:
    1. Inside the PDR solver (called every iteration for L4; L2 uses GPU path instead)
    2. For evaluating test-set errors
"""

import numpy as np
from dolfin import sqrt, assemble, dx
from dolfin import *

def compute_l4_norms(U, fn):
    """
    Compute L4 norm for each row of U:  ||u||_L4 = (∫ |u|⁴ dx)^{1/4}

    Parameters
    ----------
    U  : np.ndarray, shape (n, n_dofs)
    fn : dolfin.Function — standalone Function in the appropriate space

    Returns
    -------
    np.ndarray, shape (n,)
    """
    n = len(U)
    norms = np.zeros(n)
    for i in range(n):
        fn.vector().set_local(U[i].flatten())
        norms[i] = sqrt(sqrt(assemble(((fn) ** 2) ** 2 * dx)))
    return norms


def compute_l2_norms(P, fn):
    """
    Compute L2 norm for each row of P:  ||p||_L2 = (∫ p² dx)^{1/2}

    Parameters
    ----------
    P  : np.ndarray, shape (n, n_dofs)
    fn : dolfin.Function — standalone Function in the appropriate space

    Returns
    -------
    np.ndarray, shape (n,)
    """
    n = len(P)
    norms = np.zeros(n)
    for i in range(n):
        fn.vector().set_local(P[i].flatten())
        norms[i] = sqrt(assemble((fn) ** 2 * dx))
    return norms
