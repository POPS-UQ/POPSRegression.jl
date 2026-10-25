using Documenter
using POPSRegression

DocMeta.setdocmeta!(POPSRegression, :DocTestSetup, :(using POPSRegression); recursive=true)

makedocs(;
    modules=[POPSRegression],
    authors="POPS-UQ organization",
    sitename="POPSRegression.jl",
    format=Documenter.HTML(;
        canonical="https://POPS-UQ.github.io/POPSRegression.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API reference" => "api.md",
        "Constrained fitting" => "constraints.md",
        "Fitting an ACE model" => "ace.md",
        "Quantifying uncertainties in MD" => "md.md",
    ],
)

deploydocs(;
    repo="github.com/POPS-UQ/POPSRegression.jl.git",
    devbranch="main",
)