# Summary of `legendre.py`

This module provides the mathematical building blocks for constructing **polynomial approximations** of solutions to partial differential equations (PDEs) using **Legendre polynomials** on sparse grids. It is designed to be experiment-agnostic — it works for any PDE in any number of dimensions.

---

## Overview

The file contains three core components:

| Function | Purpose |
|---|---|
| `multiidx_gen` | Generates a set of multi-indices that define which polynomial terms to include |
| `build_design_matrix` | Constructs the Legendre polynomial design matrix and associated sparsity weights |
| `hyperbolic_cross_rule` | Defines a selection rule that favours low-interaction polynomial terms |

---

## Function-by-Function Breakdown

### 1. `multiidx_gen(N, rule, w, base=0, ...)`

**What it does:**  
Recursively enumerates all multi-indices of length `N` (one entry per dimension) whose combined "cost" — measured by a user-supplied `rule` function — does not exceed a budget `w`.

**How it works:**  
- A *multi-index* is a tuple like `(p₁, p₂, …, pₙ)` where each `pₖ` is the polynomial degree in dimension `k`.
- The function explores a tree of all possible multi-indices in a depth-first manner.
- At each branch, it increments the current dimension's degree starting from `base` (typically 0) and recurses deeper only if `rule(current_index) ≤ w`.
- Once a complete multi-index of length `N` is formed and satisfies the rule, it is stored.

**Result:**  
A matrix where each row is one admissible multi-index — collectively defining the polynomial basis to be used.

> This code is a Python translation of the *Sparse Grid Matlab Kit* by L. Tamellini and F. Nobile (2009–2018).

---

### 2. `build_design_matrix(X, Lambda)`

**What it does:**  
Given sample points `X` (shape `m × d`, each row a point in `[-1, 1]^d`) and a multi-index set `Lambda` (shape `N × d`), it constructs:

- **Design matrix `A`** (shape `m × N`): Each column corresponds to one multi-dimensional Legendre basis function evaluated at all sample points.
- **Sparsity weights** (shape `N`): A weight for each basis function, used to promote sparsity in downstream compressed-sensing or optimisation solvers.

**How it works:**  
For every multi-index `λ = (λ₁, λ₂, …, λ_d)` and every sample point `x = (x₁, x₂, …, x_d)`:

```
A[m, n] = ∏ₖ √(2λₖ + 1) · Pλₖ(xₖ)
```

where `Pₗ` is the Legendre polynomial of degree `ℓ`. The factor `√(2ℓ + 1)` normalises each univariate Legendre polynomial to unit L²-norm on `[-1, 1]`, making the resulting basis **orthonormal**.

The sparsity weight for each basis function is simply the product of these normalisation factors across all dimensions.

---

### 3. `hyperbolic_cross_rule(x)`

**What it does:**  
Implements the *hyperbolic cross* selection criterion:

```
rule(x) = ∏ₖ (xₖ + 1) − 1
```

**Why it matters:**  
The hyperbolic cross favours multi-indices whose entries are not simultaneously large. Compared to a full tensor-product basis (which grows exponentially with dimension), the hyperbolic cross keeps the basis size manageable while still capturing the most important cross-dimensional interactions. This is a cornerstone of **sparse polynomial approximation** in high-dimensional uncertainty quantification.

---

## How the Pieces Fit Together

1. **Choose a budget `w`** and call `multiidx_gen` with the `hyperbolic_cross_rule` to obtain the set of polynomial multi-indices `Lambda`.
2. **Draw or choose sample points** `X` in the parameter domain `[-1, 1]^d`.
3. **Call `build_design_matrix(X, Lambda)`** to get the measurement matrix `A` and weights.
4. **Solve a (weighted) compressed-sensing or least-squares problem** `A c ≈ b` to recover the polynomial coefficients `c`, where `b` contains the PDE solution evaluations at the sample points.

This pipeline enables efficient, sparse recovery of polynomial surrogate models for expensive PDE simulations — a key technique in **uncertainty quantification** and **surrogate modelling**.
