using Pkg

manifest_paths = [
    "TabularRL.jl",
    "ApproximationUtils.jl/"
]

for path in manifest_paths
    base_path = joinpath(@__DIR__, "ReinforcementLearning.jl", "src", path)
    Pkg.activate(base_path)
    Pkg.update()
    Pkg.precompile()
end

top_path = "ReinforcementLearning.jl"
Pkg.activate(top_path)
Pkg.instantiate()
Pkg.precompile()

Pkg.activate("PlutoStartup")
Pkg.update()
Pkg.precompile()