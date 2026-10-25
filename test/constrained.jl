## Tests of the OSQP-based constrained fit (src/fit.jl, `constraint_matrix`/`constraint_bounds` kwargs).
##
## Sanity check: with an empty constraint set, the OSQP-based constrained fit
## should reproduce the closed-form leverage-based POPS fit, since both are then solving the
## exact same per-point equality-constrained ridge QP. Also plots the two
## predictive distributions on top of each other so the agreement is visible.
##
## With an equality constraint, the mean and every POPS minimizer must match the exact
## solution of the corresponding KKT system. With inequality constraints, every POPS
## minimizer must be feasible and interpolate its training point. Both use dense constraint
## rows, so that the first-row positions in the CSC storage of the per-point constraint
## matrix differ from 1:P (an index error there would go unnoticed with no constraints).

using Test
using LinearAlgebra
using Random
using POPSRegression
using CairoMakie
using SparseArrays

@testset "analytical vs OSQP agree (no extra constraints)" begin

    # Degree-5 polynomial fit to a wiggly, noise-free function that a
    # degree-5 polynomial cannot represent exactly, so residuals (and
    # hence POPS corrections) are non-trivial.
    deg = 5
    P = deg + 1
    x = range(-1.5, 1.5; length=25)
    f(x) = exp(x) * sin(4x)
    y = f.(x)

    vander(z) = hcat((z .^ k for k in 0:deg)...)
    X = vander(x)   # N × P Vandermonde matrix

    m_analytical = fit(POPSModel, X, y; leverage_percentile=0.0)

    m_osqp = fit(POPSModel, X, y; leverage_percentile=0.0,
        constraint_matrix=zeros(0, P), constraint_bounds=(Float64[], Float64[]))

    coef_diff = maximum(abs.(coef(m_analytical) .- coef(m_osqp)))
    corr_diff = maximum(abs.(m_analytical.pops_corrections .- m_osqp.pops_corrections))

    @info "max |Δcoef|" coef_diff
    @info "max |Δpops_corrections|" corr_diff

    @test coef(m_analytical) ≈ coef(m_osqp) atol = 1e-3
    @test m_analytical.pops_corrections ≈ m_osqp.pops_corrections atol = 1e-3

    # both should still exactly interpolate every retained training point
    w_a, w_o = coef(m_analytical), coef(m_osqp)
    for i in 1:size(X, 1)
        @test X[i, :]' * (w_a .+ m_analytical.pops_corrections[i, :, :]) ≈ [y[i]] atol = 1e-8
        @test X[i, :]' * (w_o .+ m_osqp.pops_corrections[i, :, :]) ≈ [y[i]] atol = 1e-3
    end

    # ── Plot: predictive mean + bounds from both methods, overlaid ──────────
    xg = range(-1.5, 1.5; length=300)
    Xg = vander(xg)

    rng = Xoshiro(42)
    pred_a = predict(m_analytical, Xg; return_bounds=true, sampling_method=:sobol, rng=rng)
    pred_o = predict(m_osqp, Xg; return_bounds=true, sampling_method=:sobol, rng=rng)

    @info "max |Δmean prediction|" maximum(abs.(pred_a.mean .- pred_o.mean))

    # Filled bands from both methods would just alpha-blend into one blob once
    # they overlap almost exactly, so draw the bound curves as distinct lines
    # instead (solid vs dashed) — any daylight between them is then visible —
    # and add a difference panel underneath to make the (tiny) gap explicit.
    fig = Figure(size=(800, 700))

    ax = Axis(fig[1, 1], xlabel="x", ylabel="y",
        title="Analytical vs OSQP-constrained POPS fit (degree-$deg polynomial)")

    lines!(ax, xg, f.(xg); color=:black, linestyle=:dash, label="true function")
    scatter!(ax, x, y; color=:black, markersize=8, label="training data")

    lines!(ax, xg, pred_a.mean; color=:dodgerblue, linewidth=3, label="analytical mean")
    lines!(ax, xg, pred_a.lower; color=:dodgerblue, linewidth=1.5, label="analytical bounds")
    lines!(ax, xg, pred_a.upper; color=:dodgerblue, linewidth=1.5)

    lines!(ax, xg, pred_o.mean; color=:orangered, linewidth=2, linestyle=:dot, label="OSQP mean")
    lines!(ax, xg, pred_o.lower; color=:orangered, linewidth=1.5, linestyle=:dash, label="OSQP bounds")
    lines!(ax, xg, pred_o.upper; color=:orangered, linewidth=1.5, linestyle=:dash)

    axislegend(ax; position=:lb, merge=true)

    ax2 = Axis(fig[2, 1], xlabel="x", ylabel="analytical − OSQP",
        title="Pointwise differences (should hover near zero)")
    hlines!(ax2, [0.0]; color=:gray, linestyle=:dash)
    lines!(ax2, xg, pred_a.mean .- pred_o.mean; color=:black, label="mean")
    lines!(ax2, xg, pred_a.lower .- pred_o.lower; color=:dodgerblue, label="lower bound")
    lines!(ax2, xg, pred_a.upper .- pred_o.upper; color=:orangered, label="upper bound")
    axislegend(ax2; position=:rt)

    rowsize!(fig.layout, 1, Relative(0.65))

    plot_path = joinpath(@__DIR__, "analytical_vs_osqp.png")
    save(plot_path, fig)
    @info "saved plot" plot_path
end

@testset "equality constraint matches KKT solution" begin

    deg = 5
    P = deg + 1
    x = range(-1.5, 1.5; length=25)
    y = @. exp(x) * sin(4x)

    vander(z) = hcat((z .^ k for k in 0:deg)...)
    X = vander(x)

    # pin the fitted polynomial between two training points: f̂(0.3) = 0.5
    E = vander([0.3])
    e = [0.5]

    # tight tolerances, also checks that the solver settings are passed through
    # (the default correction tolerances only give ~1e-4 agreement here)
    tight = (eps_abs=1e-9, eps_rel=1e-9)
    m = fit(POPSModel, X, y; leverage_percentile=0.0,
        constraint_matrix=E, constraint_bounds=(e, e),
        mean_solver_settings=tight, correction_solver_settings=tight)

    # exact solution of min ½θᵀCθ - (Xᵀy)ᵀθ  s.t.  Bθ = d
    C = X' * X
    kkt(B, d) = ([C B'; B zeros(size(B, 1), size(B, 1))] \ [X' * y; d])[1:P]

    θ = vec(coef(m))
    @test θ ≈ kkt(E, e) atol = 1e-7
    for i in 1:size(X, 1)
        @test θ .+ m.pops_corrections[i, :, 1] ≈ kkt([X[i:i, :]; E], [y[i]; e]) atol = 1e-7
    end
end

@testset "inequality constraints hold for every POPS minimizer" begin

    deg = 5
    P = deg + 1
    x = range(-1.5, 1.5; length=25)
    y = @. exp(x) * sin(4x)

    vander(z) = hcat((z .^ k for k in 0:deg)...)
    X = vander(x)

    # bound the extrapolation just outside the data, |f̂(±1.8)| ≤ c, with c half the
    # unconstrained excursion so that the bound is active (passed as a sparse matrix)
    G = vander([-1.8, 1.8])
    c = 0.5 * maximum(abs.(G * vec(coef(fit(POPSModel, X, y)))))
    l, u = fill(-c, 2), fill(c, 2)

    m = fit(POPSModel, X, y; leverage_percentile=0.0,
        constraint_matrix=sparse(G), constraint_bounds=(l, u))

    # OSQP residual tolerance eps_abs + eps_rel * ‖Ax‖∞ with the default settings
    tol_mean = 1e-4
    tol = 3e-3

    θ = vec(coef(m))
    @test all(l .- tol_mean .<= G * θ .<= u .+ tol_mean)
    @test maximum(abs.(G * θ)) ≈ c atol = tol_mean   # the bound is active
    for i in 1:size(X, 1)
        θ_i = θ .+ m.pops_corrections[i, :, 1]
        @test all(l .- tol .<= G * θ_i .<= u .+ tol)
        @test X[i, :]' * θ_i ≈ y[i] atol = tol
    end
end
