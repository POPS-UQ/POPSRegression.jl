_resolve_prior(::Nothing, P::Int, ::Type{T}) where {T} = zeros(T, P, P) # No regularization
_resolve_prior(alpha::Real, P::Int, ::Type{T}) where {T} = Matrix{T}(T(alpha) * I, P, P)
_resolve_prior(d::AbstractVector, P::Int, ::Type{T}) where {T} = diagm(T.(d))
_resolve_prior(Sigma::AbstractMatrix, ::Int, ::Type{T}) where {T} = Matrix{T}(Sigma)

# Default OSQP settings for the constrained fit, overridden by `mean_solver_settings` / `correction_solver_settings`
const _MEAN_SOLVER_DEFAULTS = (max_iter=500_000, check_termination=10, eps_abs=5e-6, eps_rel=5e-6, verbose=false)
const _CORRECTION_SOLVER_DEFAULTS = (max_iter=500_000, check_termination=10, eps_abs=5e-4, eps_rel=5e-4, verbose=false)

_is_infeasible(status::Symbol) = status in (:Primal_infeasible, :Primal_infeasible_inaccurate)


"""
    StatsAPI.fit(::Type{POPSModel}, X, y; prior_covariance=nothing,
                leverage_percentile=0.5, rank_threshold=nothing, fit_intercept=false,
                weights=nothing, verbose=false,
                constraint_matrix=nothing, constraint_bounds=nothing,
                mean_solver_settings=(;), correction_solver_settings=(;))

Fit a POPS hypercube regression model, optionally subject to linear constraints on the coefficients.

# Arguments
- `X::AbstractMatrix`: feature matrix, `N × P`
- `y::AbstractVecOrMat`: response, length-`N` vector or `N × D` matrix

# Keyword arguments
- `prior_covariance`: prior covariance `Σ₀` (P × P). Accepts `nothing` (Σ₀=0),
a scalar (`αI`), a length-`P` vector (diagonal), or a `P × P` matrix.
For multivariate regression, we assume a separable prior `Γ = I_D ⊗ Σ₀`.
- `leverage_percentile`: only points with leverage at or above this quantile
contribute to the hypercube fit (default `0.5`).
- `rank_threshold`: relative singular-value threshold for computing effective rank.
Default `eps(T) * max(M, P·D)`, where M is the number of samples used for fitting
- `fit_intercept` : whether to add a constant colums to the feature matrix (default false)
- `weights`: optional length-`N` vector of non-negative sample weights. Default to `nothing` (unit weights).
- `verbose`: warn when the effective rank exceeds the output dimension (default `false`). OSQP output is
controlled separately, through the `verbose` field of the solver settings below.

# Linear constraints
- `constraint_matrix`: optional `K × P` matrix `G` (dense or sparse) of linear constraints on the coefficients `θ`.
Its columns follow those of `X`, preceded by the intercept column if `fit_intercept=true` (`G` then has `P + 1` columns).
Only univariate `y` is supported.
- `constraint_bounds`: tuple `(l, u)` of length-`K` vectors imposing `l ≤ G θ ≤ u`, required with `constraint_matrix`.
Use `l[k] == u[k]` for an equality and `±Inf` for a one-sided constraint.
- `mean_solver_settings`: `NamedTuple` of [OSQP settings](https://osqp.org/docs/interfaces/solver_settings.html)
for the constrained mean problem, merged over the defaults
`(max_iter=500_000, check_termination=10, eps_abs=5e-6, eps_rel=5e-6, verbose=false)`.
- `correction_solver_settings`: same, for the per-point correction problems, merged over the defaults
`(max_iter=500_000, check_termination=10, eps_abs=5e-4, eps_rel=5e-4, verbose=false)`.

With constraints, the mean model `coef` is the constrained ridge solution, and the POPS correction of each retained
point is the solution of the same problem with the additional equality of exactly interpolating that point, taken
relative to `coef`. Both are quadratic programs solved with OSQP; without constraints, they reduce to the closed-form
fit. See the *Constrained fitting* page of the documentation.
"""
function StatsAPI.fit(::Type{POPSModel}, X::AbstractMatrix, y::AbstractVecOrMat;
    prior_covariance=nothing,
    leverage_percentile=0.5,
    rank_threshold=nothing,
    fit_intercept=false,
    weights=nothing,
    verbose=false,
    constraint_matrix=nothing,
    constraint_bounds=nothing,
    mean_solver_settings=(;),
    correction_solver_settings=(;))

    @assert size(X, 1) == size(y, 1) "Number of rows in X and y must match"

    N, P = size(X)
    FT = float(promote_type(eltype(X), eltype(y)))
    lp = FT(leverage_percentile)

    X_ = Matrix{FT}(X)

    is_univariate = ndims(y) == 1

    y_ = is_univariate ? reshape(Vector{FT}(y), :, 1) : Matrix{FT}(y)
    D = size(y_, 2)

    if fit_intercept
        X_ = [ones(FT, N) X_]
        P += 1
    end

    if isnothing(weights)
        WX_ = X_
        Wy_ = y_
    else
        @assert length(weights) == N "length(weights) must equal the number of rows in X"
        @assert all(>=(0), weights) "weights must be non-negative"
        w_row = FT.(weights)
        WX_ = w_row .* X_
        Wy_ = w_row .* y_
    end

    Sigma0 = _resolve_prior(prior_covariance, P, FT)
    C_mat = Symmetric(X_' * WX_ + Sigma0 / N) # P × P, regularized covariance matrix
    C_fact = cholesky(C_mat)

    A = C_fact \ X_'                       # P × N, influence matrix
    h = vec(sum(X_ .* A'; dims=2))         # N, leverage scores diag(X C⁻¹ X')

    w = A * Wy_                            # P × D ridge solution (global loss minimizer)
    residuals = y_ - X_ * w                # N × D

    # keep high-leverage points
    mask = lp > 0 ? (h .>= quantile(h, lp)) : trues(N)
    M = sum(mask)

    A_m = @view A[:, mask]                       # P × M
    residuals_m = @view residuals[mask, :]       # M × D
    h_m = @view h[mask]                          # M


    if !isnothing(constraint_matrix)
        @assert D == 1 "constrained POPS fitting only supports univariate y (D = 1)"
        @assert !isnothing(constraint_bounds) "constraint_bounds = (l, u) must be given with constraint_matrix"
        @assert size(constraint_matrix, 2) == P "constraint_matrix must have P columns (P + 1 if fit_intercept=true)"

        # OSQP.jl only accepts CSC matrices and Float64 vectors, and only reads the upper triangle of P
        G = sparse(Float64.(constraint_matrix))
        l_c = Vector{Float64}(constraint_bounds[1])
        u_c = Vector{Float64}(constraint_bounds[2])
        @assert length(l_c) == length(u_c) == size(G, 1) "constraint_bounds must have one entry per row of constraint_matrix"
        P_qp = sparse(triu(Matrix{Float64}(C_mat)))
        q_qp = Vector{Float64}(-vec(X_' * Wy_))

        # constrained mean model: global constrained ridge solution (no per-point equality)
        mean_qp = OSQP.Model()
        OSQP.setup!(mean_qp; P=P_qp, q=q_qp, A=G, l=l_c, u=u_c,
                    merge(_MEAN_SOLVER_DEFAULTS, mean_solver_settings)...)
        mean_results = OSQP.solve!(mean_qp)
        status = mean_results.info.status
        _is_infeasible(status) && error("constraints are infeasible (OSQP status $status)")
        status == :Solved || @warn "OSQP did not converge on the constrained mean model (status $status)"
        w_final = reshape(FT.(mean_results.x), P, D)

        # per-point POPS corrections: each solves the same constraints plus an
        # equality forcing exact interpolation of that point; correction is
        # measured relative to the constrained mean model above.
        # Only the first row of A and its bounds change between points, so the solver is set up once,
        # with a dense placeholder first row fixing the sparsity pattern, and updated in place
        # (which also warm-starts each solve from the previous solution).
        A_qp = [sparse(ones(1, P)); G]
        Ax_idx = A_qp.colptr[1:P]          # positions of A_qp[1, :] in A_qp.nzval
        l_qp = [0.0; l_c]
        u_qp = [0.0; u_c]

        corr_qp = OSQP.Model()
        OSQP.setup!(corr_qp; P=P_qp, q=q_qp, A=A_qp, l=l_qp, u=u_qp,
                    merge(_CORRECTION_SOLVER_DEFAULTS, correction_solver_settings)...)

        T_corr_mat = zeros(FT, M, P)
        n_unsolved = 0
        for (j, i) in enumerate(findall(mask))
            l_qp[1] = u_qp[1] = y_[i, 1]
            OSQP.update!(corr_qp; l=l_qp, u=u_qp, Ax=Vector{Float64}(X_[i, :]), Ax_idx=Ax_idx)

            results = OSQP.solve!(corr_qp)
            status = results.info.status
            _is_infeasible(status) && error("constraints are infeasible when interpolating training point $i (OSQP status $status)")
            n_unsolved += status != :Solved
            T_corr_mat[j, :] = results.x .- vec(w_final)
        end
        n_unsolved > 0 && @warn "OSQP did not converge on $n_unsolved of $M POPS correction problems"
        T_corr = reshape(T_corr_mat, M, P, D)

    else
        w_final = w
        # compute POPS corrections
        # i-th POPS-constrained minimizer = ridge_solution + (1/leverage[i]) A[i] residual[i]ᵀ  (P × D), stacked as (M, P, D)
        scaled = (A_m ./ h_m')'                # M × P, row i = A[i]ᵀ / leverage[i]
        T_corr = reshape(scaled, M, P, 1) .* reshape(residuals_m, M, 1, D)
        T_corr_mat = reshape(T_corr, M, P * D) # M × (P·D)
    end
    # Hypercube bounds  (SVD + rank truncation)
    rt = isnothing(rank_threshold) ? FT(eps(FT) * max(M, P * D)) : FT(rank_threshold)
    F = svd(T_corr_mat)
    R = max(count(>(rt * F.S[1]), F.S), 1)

    (R > D) && verbose && @warn "effective rank ($R) > output dimension ($D). Predictive entropies will use Gaussian approximation."

    V_R = F.V[:, 1:R]                      # (P·D) × R
    T_tilde = T_corr_mat * V_R             # M × R
    lower = ntuple(r -> minimum(@view T_tilde[:, r]), R)
    upper = ntuple(r -> maximum(@view T_tilde[:, r]), R)

    return POPSModel{R,FT,typeof(prior_covariance)}(;
        prior_covariance,
        leverage_percentile=lp,
        rank_threshold=rt,
        coef=w_final,
        fit_intercept=fit_intercept,
        is_univariate=is_univariate,
        pops_corrections=T_corr,
        residuals=y_ - X_ * w_final,
        leverage_scores=h,
        C=C_fact,
        rotation=V_R,
        lower_bounds=lower,
        upper_bounds=upper,
    )
end