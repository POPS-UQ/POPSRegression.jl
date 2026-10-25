```@meta
CurrentModule = POPSRegression
```

# Constrained fitting

Prior knowledge about a model often takes the form of linear constraints on its coefficients $\theta$: a quantity that must be
non-negative or monotonic over some range, a known value at a given point, a sum rule, or a linearized stability condition.
`fit` accepts constraints of the form

```math
l \le G\theta \le u,
```

where the `K × P` matrix $G$ (dense or sparse) is passed as `constraint_matrix`, and the tuple of length-`K` vectors `(l, u)` as `constraint_bounds`.
An equality is obtained by setting `l[k] == u[k]`, and a one-sided constraint by setting the other bound to `-Inf` or `Inf`.
When `fit_intercept=true`, the intercept is the first entry of $\theta$, so that $G$ has `P + 1` columns.
Constrained fitting is currently only supported for univariate responses.

## How it works

Write $C = X^\top W X + \Sigma_0/N$ for the regularized Gram matrix and $b = X^\top W y$.
Without constraints, the mean model is the ridge solution $\theta^\star = C^{-1}b$, and the POPS minimizer of a retained training point $(x_i, y_i)$
is the minimizer of the same loss under the constraint of interpolating this point exactly, $x_i^\top\theta = y_i$. The corresponding POPS correction has the closed form

```math
\delta_i = \frac{y_i - x_i^\top\theta^\star}{h_i} C^{-1}x_i, \qquad h_i = x_i^\top C^{-1}x_i.
```

With constraints, both steps become quadratic programs:

1. The mean model (`coef`) is the constrained ridge solution
```math
\theta^\star = \arg\min_\theta \frac12\theta^\top C\theta - b^\top\theta \quad\text{subject to}\quad l \le G\theta \le u.
```
2. For each retained point $i$, the POPS minimizer $\theta_i$ solves the same problem with the additional equality constraint $x_i^\top\theta = y_i$, and the POPS correction is $\delta_i = \theta_i - \theta^\star$.

The hypercube is then built from the corrections exactly as in the unconstrained case.
With an empty constraint set (e.g. `constraint_matrix=zeros(0, P)`), the quadratic programs reduce to the closed-form expressions above, which is checked in the test suite.

Both problems are solved with [OSQP](https://osqp.org). The per-point problems only differ in the first row of the constraint matrix and in its bounds,
so the solver is set up once and updated in place for each point, which also warm-starts each solve from the previous solution.
Each retained point still requires its own solve: increasing `leverage_percentile` is the simplest way to reduce the cost of a constrained fit.

## Solver settings

OSQP settings for the two problems are passed as `NamedTuple`s, which are merged over the following defaults.

| Keyword | Defaults |
|:--------|:---------|
| `mean_solver_settings` | `max_iter=500_000, check_termination=10, eps_abs=5e-6, eps_rel=5e-6, verbose=false` |
| `correction_solver_settings` | `max_iter=500_000, check_termination=10, eps_abs=5e-4, eps_rel=5e-4, verbose=false` |

Any [OSQP setting](https://osqp.org/docs/interfaces/solver_settings.html) can be passed, for instance

```julia
m = fit(POPSModel, X, y; constraint_matrix=G, constraint_bounds=(l, u),
    mean_solver_settings=(eps_abs=1e-9, eps_rel=1e-9),
    correction_solver_settings=(eps_abs=1e-6, eps_rel=1e-6, verbose=true))
```

The tolerances `eps_abs` and `eps_rel` apply to the residuals of the optimality conditions, not to the coefficients themselves:
constraints are satisfied, and training points interpolated, up to about `eps_abs + eps_rel * ‖Gθ‖∞`, and the coefficients can be further
from the exact optimum when $C$ is ill-conditioned. They should be tightened if the POPS corrections are not large compared to this accuracy.
A warning is issued if OSQP does not converge, and an error is thrown if the constraints are infeasible, either on their own or together with
the interpolation of one of the training points.

## Constraints and the hypercube ensemble

Every POPS minimizer $\theta_i$ satisfies the constraints and, the feasible set being convex, so does any convex combination of them.
Equality constraints are inherited by all samples of the ensemble: every correction lies in the null space of the equality rows of $G$, hence so does the hypercube.
However, the hypercube is a bounding box of the corrections, which is larger than their convex hull, so that samples drawn from it (by `sample` or `predict`)
may violate inequality constraints. When strict feasibility matters, samples can simply be filtered, as in the example below.

## Example: a non-negative fit

We fit a degree-4 polynomial to the decaying function $\mathrm{e}^{-8x}$ on $[0, 1]$. The polynomial cannot represent this function exactly,
and the unconstrained fit dips below zero, although the target quantity is non-negative. We enforce non-negativity on a fine grid of $[0, 1]$.

```julia
using POPSRegression, Random, Statistics

x = range(0.0, 1.0; length=30)
y = exp.(-8 .* x)

deg = 4
features(z) = hcat((z .^ k for k in 0:deg)...) # columns 1, z, …, z⁴
X = features(x)

m_free = fit(POPSModel, X, y)

# f̂(z) ≥ 0 on a grid of [0, 1]
x_grid = range(0.0, 1.0; length=50)
G = features(x_grid)
l, u = zeros(length(x_grid)), fill(Inf, length(x_grid))

m = fit(POPSModel, X, y; constraint_matrix=G, constraint_bounds=(l, u))

minimum(G * vec(coef(m_free))) # negative: the unconstrained fit dips below zero
minimum(G * vec(coef(m)))      # non-negative, up to solver tolerance
```

Predictions and uncertainties are then obtained as usual.

```julia
pred = predict(m, features(x_grid); return_bounds=true, return_std=true)
```

Finally, we draw coefficient samples from both hypercube ensembles, and check which of them satisfy the constraints,
up to a tolerance accounting for the accuracy of the solver.

```julia
rng = Xoshiro(1)
tol = 1e-3
is_feasible(S) = vec(all(l .- tol .<= G * S .<= u .+ tol; dims=1))

S_free = sample(rng, m_free, 2000) # P × 2000 coefficient samples
S = sample(rng, m, 2000)

mean(is_feasible(S_free)) # only a few percent of the unconstrained ensemble is non-negative
mean(is_feasible(S))      # most of the constrained ensemble is

S_feasible = S[:, is_feasible(S)] # rejection step: keep only the feasible samples
```
