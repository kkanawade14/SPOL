"""
fenics_setup.py
---------------
Build FEniCS function spaces and Functions for norm computation.
Keeps FEniCS imports isolated from the rest of the codebase.
"""
import os
os.environ["DIJITSO_CACHE_DIR"] = os.path.expanduser("~/.cache/dijitso")
os.environ["CPLUS_INCLUDE_PATH"] = "/usr/common/sc-conda-envs/fenics-tf-gpu/include/eigen3:" + os.environ.get("CPLUS_INCLUDE_PATH", "")
from dolfin import Mesh, FunctionSpace, VectorFunctionSpace, Function
from dolfin import TrialFunction, TestFunction, assemble, dx
import numpy as np


def load_mesh(meshname):
    """Load a FEniCS mesh from an XML file."""
    return Mesh(meshname)


def build_norm_function(mesh, space_type):
    """
    Create a standalone FEniCS Function for norm evaluation.

    Parameters
    ----------
    mesh       : dolfin.Mesh
    space_type : str — one of "vector_dg1", "cg1", "dg0"

    Returns
    -------
    (V, fn) where V is the FunctionSpace and fn is a Function(V).
    fn is standalone (not a mixed-space view), safe for set_local().
    """
    builders = {
        "vector_dg1": lambda m: VectorFunctionSpace(m, "DG", 1),
        "cg1":        lambda m: FunctionSpace(m, "CG", 1),
        "dg0":        lambda m: FunctionSpace(m, "DG", 0),
        "dg1":        lambda m: FunctionSpace(m, "DG", 1),
    }
    if space_type not in builders:
        raise ValueError(
            f"Unknown space type '{space_type}'. Options: {list(builders.keys())}"
        )
    V = builders[space_type](mesh)
    return Function(V)


def build_mass_diagonal(V):
    """
    Assemble the lumped (diagonal) mass matrix for a scalar FunctionSpace.
    Used to compute L2 norms on GPU without FEniCS.

    Parameters
    ----------
    V : dolfin.FunctionSpace  (scalar, e.g. CG1 or DG0)

    Returns
    -------
    np.ndarray, shape (n_dofs,) — diagonal of the mass matrix.
    """
    u = TrialFunction(V)
    v = TestFunction(V)
    M = assemble(u * v * dx)
    return np.array(M.array().diagonal())
