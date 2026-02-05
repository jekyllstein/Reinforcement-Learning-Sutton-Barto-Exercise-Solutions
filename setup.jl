using Pkg

manifest_paths = [
    "TabularRL.jl",
    "ApproximationUtils.jl/",
    "NonTabularRL.jl",
]

for path in manifest_paths
    base_path = joinpath(@__DIR__, "ReinforcementLearning.jl", "src", path)
    Pkg.activate(base_path)
    Pkg.resolve()
    Pkg.precompile()
end

top_path = "ReinforcementLearning.jl"

Pkg.activate(top_path)
Pkg.resolve()
Pkg.precompile()

Pkg.activate("PlutoStartup")
Pkg.resolve()
Pkg.precompile()

mkpath(joinpath(@__DIR__, "setup_complete"))
