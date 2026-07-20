## Sanity check: with an empty constraint set, the OSQP-based constrained fit
## (src/fit.jl, `constraint_matrix`/`constraint_bounds` kwargs) should reproduce
## the closed-form leverage-based POPS fit, since both are then solving the
## exact same per-point equality-constrained ridge QP. Also plots the two
## predictive distributions on top of each other so the agreement is visible.

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

    save("test/analytical_vs_osqp.png", fig)
    @info "saved plot" "analytical_vs_osqp.png"
end
