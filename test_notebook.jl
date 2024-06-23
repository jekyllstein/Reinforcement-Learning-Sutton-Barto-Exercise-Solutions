### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 2e75e7a8-66e7-4228-a85f-6b32ba933018
using Pkg

# ╔═╡ 0d7f4d51-110a-4276-b719-693c1ddd83ec
Pkg.activate(mktempdir())

# ╔═╡ ce59a070-5412-4830-8e4d-0b2cbf7403a4
# ╠═╡ show_logs = false
Pkg.develop(path = "./TabularRL.jl")

# ╔═╡ 72a342b4-dd84-4abd-ae36-139eb1ffb11c
# ╠═╡ show_logs = false
Pkg.add(["StatsBase", "DataFrames"])

# ╔═╡ dd616985-1a9d-4c35-86f2-66e4d2e55eb2
using TabularRL, Statistics, StatsBase, DataFrames

# ╔═╡ 597af002-3c40-4dcc-859e-149c34adb0d1
make_gridworld()

# ╔═╡ aa5ce283-909c-4752-8e1b-71b09720ae1b
value_iteration_v(make_gridworld().mdp, 0.9f0)

# ╔═╡ Cell order:
# ╠═2e75e7a8-66e7-4228-a85f-6b32ba933018
# ╠═0d7f4d51-110a-4276-b719-693c1ddd83ec
# ╠═ce59a070-5412-4830-8e4d-0b2cbf7403a4
# ╠═72a342b4-dd84-4abd-ae36-139eb1ffb11c
# ╠═dd616985-1a9d-4c35-86f2-66e4d2e55eb2
# ╠═597af002-3c40-4dcc-859e-149c34adb0d1
# ╠═aa5ce283-909c-4752-8e1b-71b09720ae1b
