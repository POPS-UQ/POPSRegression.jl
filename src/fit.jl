_resolve_prior(::Nothing, P::Int, ::Type{T}) where {T} = zeros(T, P, P) # No regularization
_resolve_prior(alpha::Real, P::Int, ::Type{T}) where {T} = Matrix{T}(T(alpha) * I, P, P)
_resolve_prior(d::AbstractVector, P::Int, ::Type{T}) where {T} = diagm(T.(d))
_resolve_prior(Sigma::AbstractMatrix, ::Int, ::Type{T}) where {T} = Matrix{T}(Sigma)


"""
    StatsAPI.fit(::Type{POPSModel}, X, y; prior_covariance=nothing,
                leverage_percentile=0.5, rank_threshold=nothing)

Fit a POPS hypercube regression model.

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
"""
function StatsAPI.fit(::Type{POPSModel}, X::AbstractMatrix, y::AbstractVecOrMat;
    prior_covariance=nothing,
    leverage_percentile=0.5,
    rank_threshold=nothing,
    fit_intercept=false,
    weights=nothing,
    verbose=false,
    constraint_matrix=nothing,
    constraint_bounds=nothing)

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


    if (constraint_matrix !== nothing)
        @assert D == 1 "constrained POPS fitting only supports univariate y (D = 1)"
        b = -vec(X_' * Wy_)

        # constrained mean model: global constrained ridge solution (no per-point equality)
        mean_qp = OSQP.Model()
        OSQP.setup!(mean_qp; P=sparse(C_mat), q=b, A=sparse(constraint_matrix),
                    l=constraint_bounds[1], u=constraint_bounds[2],
                    max_iter=500_000, check_termination=10, verbose=verbose, eps_abs=5e-6, eps_rel=5e-6)
        mean_results = OSQP.solve!(mean_qp)
        w_final = reshape(mean_results.x, P, D)

        # per-point POPS corrections: each solves the same constraints plus an
        # equality forcing exact interpolation of that point; correction is
        # measured relative to the constrained mean model above.
        model = OSQP.Model()
        T_corr_mat = zeros(FT, M, P)
        j = 0
        for (idx, mask_bool) in enumerate(mask)
            mask_bool || continue
            j += 1
            A_full = vcat(WX_[idx, :]', constraint_matrix)
            l_full = vcat([Wy_[idx, 1]], constraint_bounds[1])
            u_full = vcat([Wy_[idx, 1]], constraint_bounds[2])
            A_sparse = sparse(A_full)

            OSQP.setup!(model; P=sparse(C_mat), q=b, A=A_sparse, l=l_full, u=u_full,
                        max_iter=500_000, check_termination=10, verbose=verbose, eps_abs=5e-4, eps_rel=5e-4)

            results = OSQP.solve!(model)
            T_corr_mat[j, :] = results.x .- vec(w_final)
        end
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