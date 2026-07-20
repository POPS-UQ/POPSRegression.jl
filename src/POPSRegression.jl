module POPSRegression

import StatsAPI
import StatsAPI: fit, predict, coef, nobs, dof, islinear, isfitted,
    rss, vcov, residuals, leverage, dof_residual

using LinearAlgebra, Statistics
using Random
using Sobol
using OSQP
using SparseArrays

export POPSModel, sample
export fit, predict, coef, nobs, dof, islinear, isfitted,
    rss, vcov, residuals, leverage, dof_residual

# Helper methods


"""
    POPSModel{R,T,S} <: StatsAPI.RegressionModel

A fitted POPS hypercube regression model.

Represents the POPS-covering hypercube ansatz from Swinburne & Perez (2025),
providing misspecification-aware parameter uncertainties for linear surrogate models
in the low-noise regime.

# Type parameters
- `R`: effective rank of the POPS correction matrix (dimensionality of the hypercube)
- `T`: numerical type of the data (e.g. `Float64`)
- `S`: type of prior covariance

# Fitting hyperparameters
- `prior_covariance`: prior covariance used during fitting
- `leverage_percentile`: observation blocks with leverage below this threshold are not used to fit the parameter distribution
- `rank_threshold`: relative threshold used to determine the effective rank R

# Fit results
- `coef`: ridge solution, `P × D` matrix
- `pops_corrections`: per-point corrections, `M × P × D` tensor (`M` = retained points)
- `residuals`: `N × D` matrix of training residuals
- `leverage_scores`: length-`N` vector
- `C`: Cholesky factor of the regularized Gram matrix `X'X + Σ₀/N`

# Ensemble
- `rotation`: `(P·D) × R` orthonormal basis spanning the hypercube directions in vec-space
- `lower_bounds`, `upper_bounds`: `R`-tuples defining the hypercube extents
"""
@kwdef struct POPSModel{R,T<:AbstractFloat,S} <: StatsAPI.RegressionModel
    # Hyperparameters
    prior_covariance::S
    leverage_percentile::T
    rank_threshold::T
    fit_intercept::Bool
    is_univariate::Bool

    # Fit results
    coef::Matrix{T}
    pops_corrections::Array{T,3}
    residuals::Matrix{T}
    leverage_scores::Vector{T}
    C::Cholesky{T,Matrix{T}}

    # Ensemble
    rotation::Matrix{T}
    lower_bounds::NTuple{R,T}
    upper_bounds::NTuple{R,T}
end


# StatsAPI: StatisticalModel methods

StatsAPI.coef(m::POPSModel) = m.coef
StatsAPI.nobs(m::POPSModel) = size(m.residuals, 1)
StatsAPI.dof(m::POPSModel) = length(m.coef)
StatsAPI.islinear(::POPSModel) = true
StatsAPI.isfitted(::POPSModel) = true
StatsAPI.rss(m::POPSModel) = sum(abs2, m.residuals)

# StatsAPI: RegressionModel methods

StatsAPI.residuals(m::POPSModel) = m.residuals
StatsAPI.leverage(m::POPSModel) = m.leverage_scores
StatsAPI.dof_residual(m::POPSModel) = StatsAPI.nobs(m) - StatsAPI.dof(m)

# TODO: implement other StatsAPI methods


include("fit.jl")
include("predict.jl")

end