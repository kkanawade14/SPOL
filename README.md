# SPOL — Sparse Polynomial Learning for Parametric PDEs

**Primal-Dual Restart Square-Root LASSO Recovery of Parametric PDE Solutions in Legendre Polynomial Bases**

---

## Overview

This repository implements a computational framework for the **sparse polynomial approximation** of solutions to parametric partial differential equations (PDEs). Given a PDE whose coefficients depend on a high-dimensional random parameter $\mathbf{y} \in [-1,1]^d$, we seek to recover the map $\mathbf{y} \mapsto u(\mathbf{y})$ from a small number of pointwise solution snapshots.

The core methodology combines:

1. **Legendre polynomial expansions** over hyperbolic cross multi-index sets, yielding structured sparsity in the coefficient representation;
2. **Square-Root LASSO (SR-LASSO)** minimisation with weighted $\ell_1$ regularisation, solved via a **primal-dual algorithm with restarts (PDR)** on GPU;
3. **FEniCS-based PDE solvers** for generating high-fidelity training and test data from two benchmark problems.

The implementation targets high-performance computing environments equipped with NVIDIA GPUs and leverages [CuPy](https://cupy.dev/) for GPU-accelerated linear algebra within the optimisation loop, while [FEniCS/DOLFIN](https://fenicsproject.org/) handles all finite-element assembly on CPU.

---

## Problem Formulation

Consider a parametric PDE of the form

$$\mathcal{L}(\mathbf{y})\, u(\mathbf{y}) = f, \qquad \mathbf{y} \in [-1,1]^d,$$

where $\mathcal{L}(\mathbf{y})$ is a differential operator whose coefficients depend on the parameter vector $\mathbf{y}$. We approximate the parameter-to-solution map via a **truncated Legendre expansion**:

$$u(\mathbf{y}) \approx \sum_{\boldsymbol{\nu} \in \Lambda} c_{\boldsymbol{\nu}}\, L_{\boldsymbol{\nu}}(\mathbf{y}),$$

where $\Lambda \subset \mathbb{N}_0^d$ is a **hyperbolic cross** multi-index set of maximum order $p_{\max}$, $L_{\boldsymbol{\nu}}(\mathbf{y}) = \prod_{k=1}^d \sqrt{2\nu_k + 1}\, P_{\nu_k}(y_k)$ are normalised tensorised Legendre polynomials, and $c_{\boldsymbol{\nu}}$ are the unknown coefficients (each a vector of FE DoFs).

Given $m$ i.i.d. samples $\{\mathbf{y}^{(i)}\}_{i=1}^m$ drawn uniformly from $[-1,1]^d$ and the corresponding PDE solutions $\{u(\mathbf{y}^{(i)})\}_{i=1}^m$, we solve the **weighted SR-LASSO** problem:

$$\min_{\mathbf{c}} \frac{1}{\sqrt{m}} \left\| A\mathbf{c} - \mathbf{b} \right\|_2 + \lambda \sum_{\boldsymbol{\nu} \in \Lambda} w_{\boldsymbol{\nu}} \left\| c_{\boldsymbol{\nu}} \right\|_{L^p(\mathcal{D})},$$

where $A \in \mathbb{R}^{m \times N}$ is the (rescaled) design matrix with entries $A_{i,\boldsymbol{\nu}} = L_{\boldsymbol{\nu}}(\mathbf{y}^{(i)}) / \sqrt{m}$, and $\|\cdot\|_{L^p(\mathcal{D})}$ denotes either the $L^4$ or $L^2$ norm over the spatial domain $\mathcal{D}$, evaluated via FEniCS.

---

## Benchmark Experiments

### 1. Boussinesq Equations (`bsnq`)

A **3D steady Boussinesq system** coupling natural convection (velocity $\mathbf{u}$, pressure $p$) with heat transfer (temperature $\phi$) on the unit cube $\mathcal{D} = [0,1]^3$. The parametric diffusion coefficient $\kappa(\mathbf{y})$ admits three expansion types (see below). The mixed finite-element formulation employs DG1 elements for velocity, RT2 for the stress, and CG1 for pressure and temperature.

| Variable | FE Space | Recovery Norm |
|----------|----------|---------------|
| Velocity $\mathbf{u}$ | Vector DG1 | $L^4(\mathcal{D})$ |
| Pressure $p$ | CG1 | $L^2(\mathcal{D})$ |
| Temperature $\phi$ | CG1 | $L^4(\mathcal{D})$ |

### 2. Navier–Stokes–Brinkman Equations (`nsb`)

A **2D steady Navier–Stokes–Brinkman system** modelling flow through a porous medium with an obstacle on a Poisson-type mesh . Uses DG1 elements for velocity and DG0 for pressure.

| Variable | FE Space | Recovery Norm |
|----------|----------|---------------|
| Velocity $\mathbf{u}$ | Vector DG1 | $L^4(\mathcal{D})$ |
| Pressure $p$ | DG0 | $L^2(\mathcal{D})$ |

### Parametric Coefficient Types

Three diffusion coefficient families are supported, each indexed by $\mathbf{y} \in [-1,1]^d$ with $d = 8$:

| Key | Description | Decay |
|-----|-------------|-------|
| `logKL` | Log-normal Karhunen–Loève expansion | Exponential |
| `aff_S3` | Affine expansion with sinusoidal modes | $\mathcal{O}(j^{-3/2})$ |
| `aff_F9` | Affine expansion with sinusoidal modes | $\mathcal{O}(j^{-9/5})$ |

---

## Data Generation

Each generator solves the parametric PDE for a collection of parameter samples and stores the resulting FE coefficient vectors, pre-computed norms, and parameters in HDF5 files.

### Training Data

$m_{\text{train}}$ parameter samples are drawn i.i.d. from the uniform distribution on $[-1,1]^d$. For each sample $\mathbf{y}^{(i)}$, the PDE is solved via FEniCS and the solution's DoF vector, along with its $L^4$ and $L^2$ norms, are appended to `train.h5`. Writes are incremental (one sample per append), providing **crash-safe resumption**: if the process is interrupted, it detects the number of existing samples and continues from where it stopped.

### Test Data

Test samples are placed at the nodes of a **Clenshaw–Curtis sparse grid** of level $\ell$ in $d$ dimensions, constructed via [Tasmanian](https://tasmanian.ornl.gov/). The associated **quadrature weights** are stored alongside the solution data in `test.h5`, enabling accurate numerical integration for error evaluation.


## Dependencies

| Package | Purpose |
|---------|---------|
| [FEniCS/DOLFIN](https://fenicsproject.org/) | Finite element assembly and PDE solves |
| [CuPy](https://cupy.dev/) | GPU-accelerated linear algebra |
| [Tasmanian](https://tasmanian.ornl.gov/) | Sparse grid construction (Clenshaw–Curtis) |
| [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) | Numerical routines, Legendre polynomials, I/O |
| [h5py](https://www.h5py.org/) | HDF5 data storage |
| [PyYAML](https://pyyaml.org/) | Configuration serialisation |

---

## Citation

If you use this code, please cite:

> **[1]** B. Adcock, S. Brugiapaglia, N. Dexter, and S. Moraga. *On Efficient Algorithms for Computing Near-Best Polynomial Approximations to High-Dimensional, Hilbert-Valued Functions from Limited Samples.* Memoirs of the European Mathematical Society, **13**, 1–112, 2024. EMS Press.

```bibtex
@article{adcock2024efficient,
  title     = {On Efficient Algorithms for Computing Near-Best Polynomial
               Approximations to High-Dimensional, Hilbert-Valued Functions
               from Limited Samples},
  author    = {Adcock, Ben and Brugiapaglia, Simone and Dexter, Nick and Moraga, Sebastian},
  journal   = {Memoirs of the European Mathematical Society},
  volume    = {13},
  pages     = {1--112},
  year      = {2024},
  publisher = {EMS Press}
}
```
