### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 6825b8c9-4fd1-4859-ab67-c140a5ee191e
using Random

# ╔═╡ b06b42ed-1a73-4ad6-a3b9-44dbf9d6ad7b
using PlutoPlotly, Statistics

# ╔═╡ 516234a8-2748-11ed-35df-432eebaa5162
md"""
# Chapter 8 - Planning and Learning with Tabular Methods
## 8.1 Models and Planning
## 8.2 Dyna: Integrated Planning, Acting, and Learning
"""

# ╔═╡ 4c8de0e2-aee2-4659-9a91-46e2774f542f
function ϵ_greedy(s, Q; ϵ = 0.1)
	qvec = Q[s]
	l = length(qvec)
	newinds = shuffle(1:l)
	idxgreed = newinds[argmax(qvec[newinds])]
	if rand() > ϵ
		idxgreed
	else
		rand(1:l)
	end
end

# ╔═╡ 5afad18b-1d87-450e-a0ff-8c1249d663ed
function tabular_dynaQ(env, sinit, sterm, states::AbstractVector{S}, actions::AbstractVector{A}, α, γ, n, maxepisodes; qinit=0.0, rinit = 0.0) where {S, A}
	history = Dict(s => Vector{A}() for s in states) #save a record of all visited states and actions taken
	Q = Dict(s => fill(qinit, length(actions)) for s in states)
	model = Dict((s, a) => (s, rinit) for s in states for a in actions)
	aidx_lookup = Dict(zip(actions, 1:length(actions)))
	steps = zeros(maxepisodes)
	steprewards = Vector{Float64}()

	for i in 1:maxepisodes
		s = sinit
		while s != sterm
			aidx = ϵ_greedy(s, Q)
			a = actions[aidx]
			push!(history[s], a)
			(s′, r) = env(s, a)
			push!(steprewards, r)
			Q[s][aidx] += α*(r + γ*maximum(Q[s′]) - Q[s][aidx])
			model[(s, a)] = (s′, r)
			s = s′
			for j in 1:n
				validstates = filter(s -> !isempty(history[s]), keys(history))
				s_sim = rand(validstates)
				a_sim = rand(history[s_sim])
				aidx_sim = aidx_lookup[a_sim]
				(s_sim′, r_sim) = model[(s_sim, a_sim)]
				Q[s_sim][aidx_sim] += α*(r_sim + γ*maximum(Q[s_sim′]) - Q[s_sim][aidx_sim])
			end
			steps[i] += 1
		end	
	end
	return Q, model, steps, steprewards
end

# ╔═╡ 65818e67-c146-4686-a9aa-d0859ef662fb
md"""
### Example 8.1: Dyna Maze
"""

# ╔═╡ 4b597372-a1a3-4d00-be58-046ca2e5dcc6
begin
	abstract type GridAction end
	struct Up <: GridAction end
	struct Down <: GridAction end
	struct Right <: GridAction end
	struct Left <: GridAction end

	struct GridPoint
		x::Int64
		y::Int64
	end

	move(p::GridPoint, ::Up) = GridPoint(p.x, p.y+1)
	move(p::GridPoint, ::Down) = GridPoint(p.x, p.y-1)
	move(p::GridPoint, ::Left) = GridPoint(p.x-1, p.y)
	move(p::GridPoint, ::Right) = GridPoint(p.x+1, p.y)
end


# ╔═╡ 3c556510-d71f-44b3-a765-484d2060fe55
function create_maze(xmax, ymax, start::GridPoint, goal::GridPoint; obstacles=Set{GridPoint}())
	states = [GridPoint(x, y) for x in 1:xmax for y in 1:ymax if !in((x, y), obstacles)]
	actions = [Up(), Down(), Left(), Right()]
	sterm = goal

	function badpoint(s)
		s.x < 1 && return true
		s.y < 1 && return true
		s.x > xmax && return true
		s.y > ymax && return true
		in(s, obstacles) && return true
		return false
	end
		
	function env(s, a)
		s′ = move(s, a)
		badpoint(s′) && return (s, 0.0)
		(s′ == goal) && return (goal, 1.0)
		return (s′, 0.0)
	end
	(states = states, actions = actions, start = start, goal = goal, env = env)
end

# ╔═╡ 773ac9c5-c126-4e7d-b280-299adffcd840
function create_dynamic_maze(xmax, ymax, start::GridPoint, goal::GridPoint; dynamicobstacles=Dict(typemax(Int64) => Set{GridPoint}()))
	states = [GridPoint(x, y) for x in 1:xmax for y in 1:ymax if !in((x, y), obstacles)]
	actions = [Up(), Down(), Left(), Right()]
	sterm = goal

	function badpoint(s, obstacles)
		s.x < 1 && return true
		s.y < 1 && return true
		s.x > xmax && return true
		s.y > ymax && return true
		in(s, obstacles) && return true
		return false
	end

	function get_obstacles(τ)
		k = findfirst(k -> k <= τ, keys(dynamicobstacles))
		dynamicobstacles[k]
	end
		
	function env(s, a, τ)
		obstacles = get_obstacles(τ)
		s′ = move(s, a)
		badpoint(s′, obstacles) && return (s, 0.0)
		(s′ == goal) && return (goal, 1.0)
		return (s′, 0.0)
	end
	(states = states, actions = actions, start = start, goal = goal, env = env)
end

# ╔═╡ c71ae2cb-8492-4bbb-8fff-3f7c6d096563
maze8_1 = create_maze(9, 6, GridPoint(1, 4), GridPoint(9, 6), obstacles = Set([GridPoint(p...) for p in [(3, 3), (3, 4), (3, 5), (6, 2), (8, 4), (8, 5), (8, 6)]]))

# ╔═╡ cd139745-1877-43a2-97a0-3333e544cbd8
function figure8_2()
	results = [[begin
		Random.seed!(seed)
		tabular_dynaQ(maze8_1.env, maze8_1.start, maze8_1.goal, maze8_1.states, maze8_1.actions, 0.1, 0.95, n, 50; qinit=0.0, rinit = 0.0)
	end
	for n in [0, 5, 50]] for seed in 1:30]
	t1 = scatter(x = 2:50, y = mean(r[1][3] for r in results)[2:end], name = "no planning steps")
	t2 = scatter(x = 2:50, y = mean(r[2][3] for r in results)[2:end], name = "5 planning steps")
	t3 = scatter(x = 2:50, y = mean(r[3][3] for r in results)[2:end], name = "50 planning steps")
	plot([t1, t2, t3], Layout(legend_orientation="h"))
end

# ╔═╡ 8dbc76fd-ac73-47ca-983e-0e90023390e3
figure8_2()

# ╔═╡ e0cc1ca1-595d-44e2-8612-261df9e2d327
md"""
> *Exercise 8.1* The nonplanning method looks particularly poor in Figure 8.3 because it is a one-step method; a method using multi-step bootstrapping would do better. Do you think one of the multi-step bootstrapping methods from Chapter 7 could do as well as the Dyna method? Explain why or why not.

For the n = 50 agent, it can learn a policy that covers nearly the entire maze during the second episode.  In the extreme case of a multistep method we would attempt to solve the maze using monte carlo sampling in which case after a single episode we would have action/value updates for every state visited along the random trajectory.  However, these action/value estimates would only be accurate for the random policy as only one step of optimization has been performed.  In contrast, the Dyna method after one randomly wandering episode has presumably captured most of the properties of the maze so during the long planning phase it can do a series of purely bootstrapped Q updates.  As long as something is sampled close to the goal that information will propagate through to the rest of the states and each update is simultaneously improving the ϵ-greedy policy.  With multi-step bootstrapping we can extend the updates back along the trajectory a certain distance, but in the extreme case we just sample values from the random policy without any bootstrapping or we bootstrap to a limited degree close to the goal and still have the lack of information further away from it.  The arbitrarily high number of planning steps can nearly fully optimize the Q function after just one wandering episode of data similar to the benefits of off policy learning in which we are learning a Q function from data gathered from a random policy but instead of actually sampling more trajectories we are just using the model to simulate those transitions with a continually improving Q function.
"""

# ╔═╡ 4f4551fe-54a9-4186-ab8f-3535dc2bf4c5
md"""
## 8.3 When the Model Is Wrong

### Example 8.2: Blocking Maze
"""

# ╔═╡ 870b7e41-7d1f-4af3-a145-8952a7fc8d78
maze8_2 = create_maze(9, 6, GridPoint(1, 4), GridPoint(9, 6), obstacles = Set([GridPoint(x, 3) for x in 1:8]))

# ╔═╡ f3f05eb3-db68-44c8-806e-d09127276f4d
function figure8_4()
	dynaq_results = tabular_dynaQ(maze8_2.env, maze8_2.start, maze8_2.goal, maze8_2.states, maze8_2.actions, 0.1, 0.95, 5, 250; qinit=0.0, rinit = 0.0)

	# t1 = scatter(y = cumsum(results[1][4]), name = "no planning steps")
	# t2 = scatter(y = cumsum(results[2][4]), name = "5 planning steps")
	# t3 = scatter(y = cumsum(results[3][4]), name = "50 planning steps")
	# plot([t1, t2, t3], Layout(legend_orientation="h"))
	plot(cumsum(dynaq_results[4]), Layout(title="5 Planning Steps DynaQ vs DynaQ+ on Blocking Maze"))
end

# ╔═╡ 79069fa0-56bc-4dca-9ccd-873c370bf9f8
figure8_4()

# ╔═╡ 24efe9b4-9308-4ad1-8ef0-69f6f93407c0
md"""
> *Exercise 8.2* Why did the Dyna agent with exploration bonus, Dyna-Q+, perform better in the first phase as well as in the second phase of the blocking and shortcut experiments?

For the second phase, the maze changed in both cases so the exploration reward bonus in Dyna-Q+ encourages the policy to attempt different actions that have not been visited recently which would result in model updates that reflect the new environment.  For the first phase where the model is accurate, this task may benefit from larger initial exploration than the ϵ of 0.1 provides.  In that case the Dyna-Q+ reward simply acts like if we had a larger ϵ in the first place which may result in faster learning.
"""

# ╔═╡ 26fe0c28-8f0f-4cff-87fb-76f04fce1be1
md"""
> *Exercise 8.3* Careful inspection of Figure 8.5 reveals that the difference between Dyna-Q+ and Dyna-Q narrowed slightly over the first part of the experiment. What is the reason for this?

As long as the environment isn't changing, no matter how small ϵ is both algorithms will converge to the optimal policy.  After more steps the difficiencies of each algorithm will diminish and they will converge to similar performance.
	"""

# ╔═╡ 340ba72b-172a-4d92-99b2-17687ab511c7
md"""
> *Exercise 8.4 (programming)* The exploration bonus described above actually changes the estimated values of states and actions. Is this necessary? Suppose the bonus $\kappa \sqrt{\tau}$ was used not in updates, but solely in action selection. That is, suppose the action selected was always that for which $Q(S_t,a) + \kappa \sqrt{\tau(S_t, a)}$ was maximal. Carry out a gridworld experiment that tests and illustrates the strengths and weaknesses of this alternate approach.
"""

# ╔═╡ 01f4268e-e947-4057-94a4-19d757be266d
# need to make a maze environment which has an internal state which tracks how many steps have been simulated and then can alter the maze based on that.  Also need to implement Dyna-Q+ which involves augmenting the history with the time since last visited.  Curious about connection with the benefits of the "simulation" steps here with the generalized policy iteration method where you wait until the action/value function has converged with interative updates without actually changing the policy.  Those updates should apply to all states and benefit from fully using the existing experience.

# ╔═╡ d00014fa-1539-4f42-ba63-15c7c9fecfde
function tabular_dynaQplus(env, sinit, sterm, states::AbstractVector{S}, actions::AbstractVector{A}, α, γ, κ, n, maxepisodes; qinit=0.0, rinit = 0.0) where {S, A}
	history = Dict(s => Vector{A}() for s in states) #save a record of all visited states and actions taken
	history_times = Dict{Tuple{S, A}, Int64}()
	Q = Dict(s => fill(qinit, length(actions)) for s in states)
	model = Dict((s, a) => (s, rinit) for s in states for a in actions)
	aidx_lookup = Dict(zip(actions, 1:length(actions)))
	steps = zeros(maxepisodes)
	τ = 1
	
	steprewards = Vector{Float64}()
	for i in 1:maxepisodes
		s = sinit
		while s != sterm
			aidx = ϵ_greedy(s, Q)
			a = actions[aidx]
			push!(history[s], a)
			history_times[(s, a)] = τ
			(s′, r) = env(s, a, τ)
			push!(steprewards, r)
			Q[s][aidx] += α*(r + γ*maximum(Q[s′]) - Q[s][aidx])
			model[(s, a)] = (s′, r)
			s = s′
			#planning updates
			for j in 1:n
				validstates = filter(s -> !isempty(history[s]), keys(history))
				s_sim = rand(validstates)
				a_sim = rand(history[s_sim])
				aidx_sim = aidx_lookup[a_sim]
				(s_sim′, r_sim) = model[(s_sim, a_sim)]
				Q[s_sim][aidx_sim] += α*(r_sim + κ*sqrt(τ - history_times[(s_sim, a_sim)]) + γ*maximum(Q[s_sim′]) - Q[s_sim][aidx_sim])
			end
			steps[i] += 1
			τ += 1
		end	
	end
	return Q, model, steps, steprewards
end

# ╔═╡ a2dbb1e2-2038-40ee-a2d9-8f5a594dd7a8
function test_dynaQplus()
	results = [[begin
		Random.seed!(seed)
		tabular_dynaQplus(maze8_1.env, maze8_1.start, maze8_1.goal, maze8_1.states, maze8_1.actions, 0.1, 0.95, 0.0001, n, 50; qinit=0.0, rinit = 0.0)
	end
	for n in [0, 5, 50]] for seed in 1:30]
	t1 = scatter(x = 2:50, y = mean(r[1][3] for r in results)[2:end], name = "no planning steps")
	t2 = scatter(x = 2:50, y = mean(r[2][3] for r in results)[2:end], name = "5 planning steps")
	t3 = scatter(x = 2:50, y = mean(r[3][3] for r in results)[2:end], name = "50 planning steps")
	plot([t1, t2, t3], Layout(legend_orientation="h"))
end

# ╔═╡ 95d18377-aef7-468c-8b2c-57f82bc7fe91
test_dynaQplus()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
PlutoPlotly = "~0.3.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "7504e1c35d64e2674406ee5a0b3e63da6ee8eca9"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "80ca332f6dcb2508adba68f22f551adb2d00a624"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.3"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "5856d3031cdb1f3b2b6340dfdc66b6d9a149a374"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.2.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "180d744848ba316a3d0fdf4dbd34b77c7242963a"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.18"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Dates", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "PlotlyBase", "PlutoUI", "Reexport"]
git-tree-sha1 = "b470931aa2a8112c8b08e66ea096c6c62c60571e"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.3.4"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─516234a8-2748-11ed-35df-432eebaa5162
# ╠═6825b8c9-4fd1-4859-ab67-c140a5ee191e
# ╠═4c8de0e2-aee2-4659-9a91-46e2774f542f
# ╠═5afad18b-1d87-450e-a0ff-8c1249d663ed
# ╟─65818e67-c146-4686-a9aa-d0859ef662fb
# ╠═4b597372-a1a3-4d00-be58-046ca2e5dcc6
# ╠═3c556510-d71f-44b3-a765-484d2060fe55
# ╠═773ac9c5-c126-4e7d-b280-299adffcd840
# ╠═c71ae2cb-8492-4bbb-8fff-3f7c6d096563
# ╠═cd139745-1877-43a2-97a0-3333e544cbd8
# ╠═b06b42ed-1a73-4ad6-a3b9-44dbf9d6ad7b
# ╠═8dbc76fd-ac73-47ca-983e-0e90023390e3
# ╟─e0cc1ca1-595d-44e2-8612-261df9e2d327
# ╟─4f4551fe-54a9-4186-ab8f-3535dc2bf4c5
# ╠═870b7e41-7d1f-4af3-a145-8952a7fc8d78
# ╠═f3f05eb3-db68-44c8-806e-d09127276f4d
# ╠═79069fa0-56bc-4dca-9ccd-873c370bf9f8
# ╟─24efe9b4-9308-4ad1-8ef0-69f6f93407c0
# ╟─26fe0c28-8f0f-4cff-87fb-76f04fce1be1
# ╟─340ba72b-172a-4d92-99b2-17687ab511c7
# ╠═01f4268e-e947-4057-94a4-19d757be266d
# ╠═d00014fa-1539-4f42-ba63-15c7c9fecfde
# ╠═a2dbb1e2-2038-40ee-a2d9-8f5a594dd7a8
# ╠═95d18377-aef7-468c-8b2c-57f82bc7fe91
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
