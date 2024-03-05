module PlutoStartup

using Reexport, PrecompileTools, Base.Threads

@reexport using Pluto, PlutoSliderServer, Statistics, StatsBase

function __init__()
    Pluto.run(launch_browser=false)
end
# @setup_workload begin
#     using PlotlyJS
#     @compile_workload begin
#         t = scatter(x = rand(100), y = rand(100); mode = "markers")
#         p = plot(t)
#         savefig(p, "test.html")
#     end
# end
end # module PlutoStartup
