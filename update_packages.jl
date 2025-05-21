using Pkg

manifest_paths = [
    "TabularRL.jl",
    "ApproximationUtils.jl/",
    "NonTabularRL.jl",
    "PlutoStartup.jl"
]

for path in manifest_paths
    base_path = joinpath(@__DIR__, path)
    Pkg.activate(base_path)
    Pkg.update()
    Pkg.precompile()
end