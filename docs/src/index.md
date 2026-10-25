```@meta
CurrentModule = POPSRegression
```

# POPSRegression.jl

A Julia library of probabilistic surrogate models adapted to misspecified functional
forms in the small observation noise regime.

The package takes its name from the POPS regression algorithm
([Perez & Swinburne, 2025](https://doi.org/10.1088/2632-2153/ad9fce)).
Implemented models follow the
[StatsAPI.jl](https://github.com/JuliaStats/StatsAPI.jl) conventions.

## Implemented features
  - Univariate and multivariate POPS hypercube regression for linear models                                                                                                                                           
  - Leverage-based filtering of training points for efficient fitting                                                                              
  - Predictive uncertainty quantification: min–max bounds, standard deviations, and differential entropy estimates                                                
  - Compliance with the StatsAPI.jl interface                                                                                                                                            
  - Linear equality and inequality constraints on the model coefficients (see [Constrained fitting](constraints.md))
  - Unit testing, including checks of the constrained fit against closed-form solutions
  - Simple examples, including uncertainty quantification for a linear MLIP predictions, and structural quantities from subsequent molecular dynamics simulations.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/POPS-UQ/POPS.jl")
```

## Running tests

From the package root directory, run `julia --project =.`, then from the REPL:

```julia
]
test
```

## Running the examples

Two worked examples are given in the `examples/` directory, each
with their own project environment. To run e.g. the molecular dynamics example,
first instantiate the corresponding environment.

```bash
julia --project=examples/md -e "using Pkg; Pkg.instantiate()"
```
This will install the required dependencies, and may take a few minutes. You can then run the example script (here using all available threads):

```bash
julia --project=examples/md -t $(nproc) examples/md/pops_rdf.jl
```

The same applies to the ACE example under `examples/ace/`.

## Quick start

```julia
using POPSRegression, Random

rng = Xoshiro(0)
N, P, D = 200, 5, 2
X = randn(rng, N, P)
W_true = randn(rng, P, D)
Y = X * W_true .+ 0.01 .* randn(rng, N, D)

model = fit(POPSModel, X, Y; prior_covariance=1e-3, leverage_percentile=0.5)

X_test = randn(rng, 10, P)
pred = predict(model, X_test; return_bounds=true, return_std=true, level=0.95)

pred.mean   # 10 × 2 mean predictions
pred.lower  # 10 × 2 lower 2.5% quantile
pred.upper  # 10 × 2 upper 97.5% quantile
pred.std    # 10 × 2 empirical std
```

## Contents

```@contents
Pages = ["./api.md", "./constraints.md", "./ace.md", "./md.md"]
Depth = 2
```
