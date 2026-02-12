module ApproximationUtils

using Reexport 

include(joinpath(@__DIR__, "..", "..", "TabularRL.jl", "src", "TabularRL.jl"))

@reexport using .TabularRL

@reexport using NVIDIALibraries

@reexport using TailRec, FCANN, Transducers, SparseArrays, LinearAlgebra, Statistics, Random, StatsBase, StaticArrays

end # module ApproximationUtils
