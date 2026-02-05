### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# ╔═╡ a82504ad-a837-49cc-9f2b-460c1fa68348
using Serialization

# ╔═╡ f173569d-182f-4873-97ed-1b9cad2b4309
using DataFrames

# ╔═╡ 9a4d0c70-ca15-4201-8a2e-56af95a60290
using PlutoDevMacros, Random, Statistics, LinearAlgebra, Transducers, Base.Threads, Random, Distributions, Statistics, StatsBase, StaticArrays

# ╔═╡ acfabeef-f268-4c7f-a07c-4c05d1333305
begin
	PlutoDevMacros.@frompackage @raw_str(joinpath(@__DIR__, "..", "NonTabularRL.jl")) using NonTabularRL

	switch_device(3)
end

# ╔═╡ c77a29da-a2a4-4956-9795-56ce63337495
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI, PlutoPlotly, LaTeXStrings, PlutoProfile, HypertextLiteral, ProgressLogging, BenchmarkTools
	TableOfContents(;depth = 4)
end
  ╠═╡ =#

# ╔═╡ e7c4aad1-562f-4027-b64b-39859ac8abbb
md"""
# Environment Description

Consider the task of driving an underpowered care up a steep mountain road.  The car begins at the bottom of the hill and the goal is to drive forward to the top.  Behind the car is another hill.  Even at full throttle the car cannot fully reach top of either side accelerating in one direction.  The only way to reach the top is to build up momentum by changing direction and slowly accumulating enough velocity to crest the hill.  The shape of the hill is fixed and the forces on the car automatically take this into account so we can reduce the variables to the single dimension of the horizontal position.  To fully describe the state we need both the horizontal position and velocity of the car.  The environment dynamics determine the position and velocity at the next time step from this information.
"""

# ╔═╡ 00e79eb9-93aa-4afc-b665-aa410a1d3a9b
md"""
## Dynamics/Rules

### States
- Horizontal position and velocity: ``(x, \dot x)``
- Bounds: ``-1.2 \leq x \leq 0.5``, ``-0.07 \leq \dot x \leq 0.07``
- Initialization: ``x_0 \in [-0.6, -0.4)``, ``\dot x_0 = 0``
### Actions
Full throttle forward (+1), zero throttle (0), and full throttle reverse (-1)

### Transitions
- Deterministic update to both position and velocity
-  ``x_{t+1} \doteq bound [ x_t + \dot x_{t+1} ]``
-  ``\dot x_{t+1} \doteq bound [ \dot x_t + 0.001 A_t - 0.0025 \cos (3 x_t)]``
- Bounds are enforced after each step and if ``x_{t+1} \lt -1.2`` then velocity is also reset to 0

### Goals and Rewards
- Reach ``x = 0.5`` as quickly as possible
- Episodic task: Episode ends with ``x = 0.5`` and resets position as stated above
- Continuing task: After ``x = 0.5`` receive goal reward and reset position as stated above 
- Episodic task reward options
  - -1 per step with no discounting ``\gamma = 1``
  - 0 per step, +1 for reaching goal, ``\gamma \lt 1``
- Continuing task reward: 0 per step with +1 every time goal is reached (average reward is maximized when goal is reached as quickly as possible from initial state)
"""

# ╔═╡ d52f66cd-c4c2-4750-b182-42e08a9a27f4
md"""
## MDP Definitions
"""

# ╔═╡ 92b57e64-973c-4601-8789-7864b7fd07de
function create_mountaincar_mdps()
	episodic_transition1 = MountainCarTask.transition_deterministic
	function step2(s, i_a::Integer)
		s′ = MountainCarTask.step(s, MountainCarTask.actions[i_a])
		r = Float32(MountainCarTask.isterm(s′))
		(r, s′)
	end

	function continuing_step(s, i_a::Integer)
		s′ = MountainCarTask.step(s, MountainCarTask.actions[i_a])
		terminated = MountainCarTask.isterm(s′)
		r = Float32(terminated)
		s′ = terminated ? MountainCarTask.initialize_state() : s′
		(r, s′)
	end

	episodic_transition2 = StateMDPTransitionDeterministic(step2, MountainCarTask.initialize_state())
	continuing_transition = StateMDPTransitionDeterministic(continuing_step, MountainCarTask.initialize_state())

	mdp_episodic1 = StateMDP(MountainCarTask.actions, episodic_transition1, MountainCarTask.initialize_state, MountainCarTask.isterm)
	mdp_episodic2 = StateMDP(MountainCarTask.actions, episodic_transition2, MountainCarTask.initialize_state, MountainCarTask.isterm)
	mdp_continuing = StateMDP(MountainCarTask.actions, continuing_transition, MountainCarTask.initialize_state, Returns(false))
	(mdp_episodic1, mdp_episodic2, mdp_continuing)
end

# ╔═╡ d05722e6-f0d2-44fd-bd07-20b5ae1ffc11
const mountaincar_mdps = create_mountaincar_mdps()

# ╔═╡ a6d0b563-d561-445c-a0b8-98b84c48fb36
md"""
### Visualizing Random Policy Trajectories
"""

# ╔═╡ 873fec07-6816-447a-8033-1aeb21d5bae8
md"""
For the episodic task the reward is just -1 times the number of steps.  I've cut off the episode at 1000 steps so this is an incomplete episode
"""

# ╔═╡ 9d6c7dd3-e3e7-4571-bab1-b988488807c2
md"""
For the continuing task the reward is only non-zero for reaching the goal, so for this limited number of steps there is no reward signal yet.
"""

# ╔═╡ cb12f8b6-cec1-40fe-9b29-0acae1d3291b
md"""
# Approximation Techniques
"""

# ╔═╡ c8ebb9fe-bc0b-488c-bf68-761e57624a87
md"""
## Feature Vectors

For any approximation technique we must define a feature vector mapping: $s \rightarrow \mathbf{x}(s)$ and select the dimensions of this feature vector.  The mountaincar task has two real numbers which define a state.  The simplest possible feature vector would be $\mathbf{x}(s) = [x, \dot x]$ but with linear approximation this would only result in two parameters and likely not be complex enough to solve the problem.  An alternative is to construct features from these two values in such a way that we expand the number of paramters.  Tile coding is a natural choice for this problem where we segment the state space into overlapping buckets in the two dimensions.  Alternatively we could use non-linear approximation with the small feature vector and rely on the hidden layers to discover relevant features.
"""

# ╔═╡ 8155a38b-913e-410a-9ac6-aa8d642a4c12
md"""
### Normalized Values

For states constructed of a list of values, it is often useful for the feature vector itself to have normalized values which lie within a reasonable range such as -1 to 1.  If we know the bounds on the true state values, then we can construct a function which normalizes the values for us.
"""

# ╔═╡ be46a64e-7fc1-4d5e-96a9-8474fef04f9f
const mountaincar_simple_feature_setup = normalized_feature_setup(mountaincar_mdps[1], identity, MountainCarTask.min_vals, MountainCarTask.max_vals)

# ╔═╡ 4f10e409-bda0-4765-ae00-a20761f1f85d
#=╠═╡
@bind test_mountaincar_state PlutoUI.combine() do Child
	md"""
	position: $(Child(Slider(-1.2f0:0.1f0:0.5f0, default = 0f0, show_value=true)))
	
	velocity: $(Child(Slider(-0.07f0:0.01f0:0.07f0, default = 0f0, show_value=true)))
	"""
end
  ╠═╡ =#

# ╔═╡ ed7ced42-cacf-434a-a572-2b1d255b938f
#=╠═╡
vtest = mountaincar_simple_feature_setup.update_feature_vector!(zeros(Float32, 2), Tuple(test_mountaincar_state))
  ╠═╡ =#

# ╔═╡ 388b8009-8220-4c21-8812-2194a9ccc5da
#=╠═╡
function plot_state_feature(s::Tuple, v::Vector)
	t1 = scatter(x = [s[1]], y = [s[2]])
	t2 = scatter(x = [0, v[1]], y = [0, v[2]])
	p1 = plot(t1, Layout(yaxis_range = [-0.07, 0.07], xaxis_range = [-1.2, 0.5], title = "Mountaincar State"))
	p2 = plot(t2, Layout(xaxis_range = [-1, 1], yaxis = attr(range = [-1, 1], scaleanchor = "x", scaleratio = 1), title = "Normalized Feature Vector"))
	[p1 p2]
end
  ╠═╡ =#

# ╔═╡ 99a47b16-98e3-4c7e-ae30-bdde7a8a6f8e
md"""
### Tile Coding

When constructing tiles we must select how many tiles are in each dimension and also how many total tilings to use.  The number of tilings determine how many non-zero features exist for any particular state.  Which features are active will move depending on the state.
"""

# ╔═╡ e55f16b0-3242-4185-b23c-37f3d8df7fc4
const test_mountaincar_tiles = tile_coding_feature_setup(mountaincar_mdps[1], MountainCarTask.min_vals, MountainCarTask.max_vals, (5, 5), 5)

# ╔═╡ d8ee80bd-da37-4187-93db-cdab8f1a827a
#=╠═╡
@bind test_mountaincar_state2 PlutoUI.combine() do Child
	md"""
	position: $(Child(Slider(-1.2f0:0.1f0:0.5f0, default = 0f0, show_value=true)))
	
	velocity: $(Child(Slider(-0.07f0:0.01f0:0.07f0, default = 0f0, show_value=true)))
	"""
end
  ╠═╡ =#

# ╔═╡ b52f900d-abd7-4800-a4b2-cbefd770c002
#=╠═╡
test_mountaincar_tiles.update_feature_vector!(test_mountaincar_tiles.feature_vector, Tuple(test_mountaincar_state2))
  ╠═╡ =#

# ╔═╡ 30cfb931-b3af-4360-a432-9a5c4b22ec6d
#=╠═╡
function plot_state_feature(s::Tuple, v::BinaryFeatureVector{Int64, N}) where N
	features = zeros(N)
	for i in v.active_features
		features[i] = 1
	end
	t1 = scatter(x = [s[1]], y = [s[2]])
	t2 = bar(x = 1:N, y = features)
	p1 = plot(t1, Layout(yaxis_range = [-0.07, 0.07], xaxis_range = [-1.2, 0.5], title = "Mountaincar State"))
	p2 = plot(t2, Layout(title = "Tile Coding Active Features"))
	[p1 p2]
end
  ╠═╡ =#

# ╔═╡ 23eb5078-4d37-43e8-960a-9f0970d00a83
#=╠═╡
plot_state_feature(Tuple(test_mountaincar_state), vtest)
  ╠═╡ =#

# ╔═╡ cdb5a7e8-8478-4841-b14e-e1baf5acf6d9
#=╠═╡
plot_state_feature(Tuple(test_mountaincar_state2), test_mountaincar_tiles.feature_vector)
  ╠═╡ =#

# ╔═╡ 0bab50cf-60f9-469d-81cf-9740c0c69b2c
md"""
## Value Function Methods
"""

# ╔═╡ 72250ee1-8f9d-4ec1-a7e4-4d3bcd1255cf
md"""
### Sarsa λ

We approximate the state-action value function using environment samples.  SARSA is an acronym that refers to the samples we need to compute the gradient update, namely the current state action pair, the next reward, the next state, and the action selected at the future state.  If we combine these samples with eligibility traces to track the gradient we can effectively use a n-step target value that ranges from 1 step (λ = 0) to nearly Monte Carlo sampling (λ → 1).

### DP λ

We approximate the state value function using environment samples.  We use the deterministic transition to determine the action values and the greedy policy.  This policy is used to sample states but each state update can be performed prior to sampling the transition state or selecting an action there.
"""

# ╔═╡ 0f55b5cf-c5aa-4c54-8df3-17291ad47894
md"""
### Linear Methods

We will consider the simple linear feature vector of the normalized state values as well as tile-coding feature construction.  The former will only allow for two linear parameters while tile-coding can be as complicated as desired
"""

# ╔═╡ 470495da-76b7-43d0-91a1-6f08a378e95c
begin
	function run_mountaincar_λ_linear(mdp::StateMDP, γ::T, α::T, λ::T, feature_vector, update_feature_vector!::Function; use_dp = false, num_steps = 50_000, kwargs...) where T<:Real 
		if iszero(λ)
			algo = use_dp ? semi_gradient_dp_linear : semi_gradient_sarsa_linear
			algo(mdp, γ, typemax(Int64), num_steps, feature_vector, update_feature_vector!; α = α, kwargs...)
		else
			algo = use_dp ? dp_λ_linear : sarsa_λ_linear
			algo(mdp, γ, λ, typemax(Int64), num_steps, feature_vector, update_feature_vector!; α = α, kwargs...)
		end
	end
	
	function run_mountaincar_λ_linear(mdp::StateMDP, α::T, λ::T, feature_vector, update_feature_vector!::Function; use_dp = false, num_steps = 50_000, kwargs...) where T<:Real 
		# if iszero(λ)
		# 	algo = use_dp ? semi_gradient_differential_dp_linear : semi_gradient_differential_sarsa_linear
		# 	algo(mdp, typemax(Int64), num_steps, feature_vector, update_feature_vector!; α = α, kwargs...)
		# else
			algo = use_dp ? dp_λ_linear : sarsa_λ_linear
			algo(mdp, λ, num_steps, feature_vector, update_feature_vector!; α = α, kwargs...)
		# end
	end
end

# ╔═╡ 2ffa9945-f1cd-46ba-9755-942a2b2ea6fe
function setup_mountaincar_simple()
	x = mountaincar_simple_feature_setup.feature_vector
	f! = mountaincar_simple_feature_setup.update_feature_vector!
	train1(α, λ; kwargs...) = run_mountaincar_λ_linear(mountaincar_mdps[1], 1f0, α, λ, copy(x), f!; kwargs...)
	train2(α, λ; kwargs...) = run_mountaincar_λ_linear(mountaincar_mdps[3], α, λ, copy(x), f!; kwargs...)
	return (train_ep = train1, train_cont = train2)
end

# ╔═╡ 2811ce8e-8d0f-44ea-a41c-24b2319413f1
function setup_mountaincar_tilecoding()
	function train1(α, λ; num_tiles::Integer = 10, num_tilings::Integer = 10, kwargs...) 
		setup = tile_coding_feature_setup(mountaincar_mdps[1], MountainCarTask.min_vals, MountainCarTask.max_vals, (num_tiles, num_tiles), num_tilings)
		run_mountaincar_λ_linear(mountaincar_mdps[1], 1f0, α, λ, setup.feature_vector, setup.update_feature_vector!; kwargs...)
	end

	function train2(α, λ; num_tiles::Integer = 10, num_tilings::Integer = 10, kwargs...) 
		setup = tile_coding_feature_setup(mountaincar_mdps[3], MountainCarTask.min_vals, MountainCarTask.max_vals, (num_tiles, num_tiles), num_tilings)
		run_mountaincar_λ_linear(mountaincar_mdps[3], α, λ, setup.feature_vector, setup.update_feature_vector!; kwargs...)
	end
	return (train_ep = train1, train_cont = train2)
end

# ╔═╡ 18f06e6c-4e4a-4a56-8822-390da61294d8
const mountaincar_simple = setup_mountaincar_simple()

# ╔═╡ 014d8f0f-7367-47c3-94a2-e7e9949d56be
const mountaincar_tilecoding = setup_mountaincar_tilecoding()

# ╔═╡ fcfbbce6-43d4-4494-924d-a36e2832dafa
#=╠═╡
begin
	simple_ep_trial(α, λ; kwargs...) = mountaincar_simple.train_ep(α, λ; kwargs...).episode_rewards |> mean
	simple_cont_trial(α, λ; kwargs...) = mountaincar_simple.train_cont(α, λ; kwargs...).reward_history |> mean

	simple_ep_study = setup_parameter_study(simple_ep_trial, (:α, :λ), (use_dp = false, num_steps = 100_000, ϵ = 0.01f0))
	if isfile("simple_ep_study.bin")
		let 
			d = deserialize("simple_ep_study.bin")
			for k in keys(d)
				simple_ep_study.results[k] = d[k]
			end
		end
	end
	simple_cont_study = setup_parameter_study(simple_cont_trial, (:α, :λ), (use_dp = false, num_steps = 100_000, α_r̄ = 0.01f0, ϵ = 0.01f0))
	if isfile("simple_cont_study.bin")
		let
			d = deserialize("simple_cont_study.bin")
			for k in keys(d)
				simple_cont_study.results[k] = d[k]
			end
		end
	end
	function run_simple_ep_study(α_list, λ_list; kwargs...)
		for α in α_list for λ in λ_list
			simple_ep_study.update_results!(α, λ; kwargs...)
		end
		end
		return simple_ep_study.results
	end
end

  ╠═╡ =#

# ╔═╡ 98774381-55a9-4a93-af33-1e00c81aedd9
#=╠═╡
function run_simple_cont_study(α_list, λ_list; kwargs...)
	for α in α_list for λ in λ_list
		simple_cont_study.update_results!(α, λ; kwargs...)
	end
	end
	return simple_cont_study.results
end
  ╠═╡ =#

# ╔═╡ 224153cd-0588-4f4e-a1d8-81c92b42b868
#=╠═╡
@bind save_studies CounterButton("Save Parameter Studies")
  ╠═╡ =#

# ╔═╡ fa5ad7ab-0a57-43ff-a6e1-a9bd73ed8566
#=╠═╡
function plot_simple_ep_algo_results(study; use_dp = false, num_steps = 100_000, num_trials = Base.Threads.nthreads())
	function valid_key(k)
		k.use_dp == use_dp &&
		k.num_steps == num_steps &&
		k.num_trials == num_trials
	end
	
	ks = filter(valid_key, keys(study.results))

	results = Dict(begin
		(α = k.α, λ = k.λ) => -study.results[k]
	 end
	 for k in ks)

	λs = unique(r.λ for r in keys(results)) |> sort
	αs = unique(r.α for r in keys(results)) |> sort

	traces = [begin
			 y = [haskey(results, (α = α, λ = λ)) ? results[(α = α, λ = λ)] : NaN32 for α in αs]
			 x = αs
			 scatter(x = x, y = y, mode = "markers", name = "λ = $λ")
			end
			for λ in λs]
	plot(traces, Layout(xaxis_type = "log"))
end
  ╠═╡ =#

# ╔═╡ 6eb8894e-5101-49a7-a760-6c2289a62cd2
#=╠═╡
function plot_simple_cont_algo_results(study; use_dp = false, num_steps = 100_000, α_r̄ = 0.01f0, num_trials = Base.Threads.nthreads())
	function valid_key(k)
		k.use_dp == use_dp &&
		k.num_steps == num_steps &&
		k.num_trials == num_trials &&
		k.α_r̄ == α_r̄
	end
	
	ks = filter(valid_key, keys(study.results))

	results = Dict(begin
		(α = k.α, λ = k.λ) => study.results[k] |> inv
	 end
	 for k in ks)

	λs = unique(r.λ for r in keys(results)) |> sort
	αs = unique(r.α for r in keys(results)) |> sort

	traces = [begin
			 y = [haskey(results, (α = α, λ = λ)) ? results[(α = α, λ = λ)] : NaN32 for α in αs]
			 x = αs
			 scatter(x = x, y = y, mode = "markers", name = "λ = $λ")
			end
			for λ in λs]
	plot(traces, Layout(xaxis_type = "log"))
end
  ╠═╡ =#

# ╔═╡ efe9d6c3-c8ff-4891-b0f4-5eca72bee048
md"""
#### Episodic Simple Linear Results
"""

# ╔═╡ aa030517-7db9-4b23-94aa-3d96966f5347
md"""
##### Sarsa λ Parameter Study
"""

# ╔═╡ 6276d0e8-fc97-4b04-97af-a1985d181d7c
#=╠═╡
begin 
	run_simple_ep_study(2f0 .^ (-10:-1), vcat(0f0:0.1f0:0.9f0, 0.99f0))
	plot_simple_ep_algo_results(simple_ep_study)
end
  ╠═╡ =#

# ╔═╡ 691cb412-8340-41d9-b775-0b838a5aee62
md"""
##### DP λ Parameter Study
"""

# ╔═╡ a7503137-cbec-41b5-b644-f950665ec934
#=╠═╡
begin 
	run_simple_ep_study(2f0 .^ (-5:2), vcat(0f0:0.1f0:0.9f0, 0.99f0); use_dp = true)
	plot_simple_ep_algo_results(simple_ep_study; use_dp = true)
end
  ╠═╡ =#

# ╔═╡ ea4ee76e-ee31-442c-b7f2-1da96da5cb78
md"""
##### Sarsa λ Best Result
"""

# ╔═╡ ac93e260-0792-434d-93a3-877955795103
md"""
##### DP λ Best Result
"""

# ╔═╡ 8ae34f97-28b5-4ccd-b980-67e38525d203
md"""
#### Continuing Simple Linear Results
"""

# ╔═╡ 5c080c53-9768-4bbb-b34d-0df7e40634bd
md"""
##### Sarsa λ Parameter Study
"""

# ╔═╡ 376d0588-6c13-4b7f-99dd-9355b484f594
#=╠═╡
begin 
	run_simple_cont_study(2f0 .^ (-5:0), [0f0, 0.2f0, 0.5f0, 0.7f0, 0.9f0, 0.99f0])
	plot_simple_cont_algo_results(simple_cont_study)
end
  ╠═╡ =#

# ╔═╡ da080be5-c2ed-42e6-b361-5542d58c95c7
md"""
##### DP λ Parameter Study
"""

# ╔═╡ f8766c91-825d-41a9-9fcf-276c1ef6c708
#=╠═╡
begin 
	run_simple_cont_study(2f0 .^ (-1:2), [0.9f0, 0.95f0, 0.99f0]; use_dp = true)
	plot_simple_cont_algo_results(simple_cont_study; use_dp = true)
end
  ╠═╡ =#

# ╔═╡ 40aa5ac9-e63c-43d1-9301-c8e899ffe2b3
md"""
#### Episodic Tilecoding Results
"""

# ╔═╡ 4ef9f073-2cc8-4e5c-bbb2-df77bb4f1eaa
#=╠═╡
begin
	tilecoding_ep_trial(α, λ; kwargs...) = mountaincar_tilecoding.train_ep(α, λ; kwargs...).episode_rewards |> mean
	tilecoding_cont_trial(α, λ; kwargs...) = mountaincar_tilecoding.train_cont(α, λ; kwargs...).reward_history |> mean

	tilecoding_ep_study = setup_parameter_study(tilecoding_ep_trial, (:α, :λ), (use_dp = false, num_steps = 100_000, num_tiles = 5, num_tilings = 5, ϵ = 0.01f0))
	tilecoding_cont_study = setup_parameter_study(tilecoding_cont_trial, (:α, :λ), (use_dp = false, num_steps = 100_000, α_r̄ = 0.01f0, num_tiles = 5, num_tilings = 5, ϵ = 0.01f0))

	if isfile("tilecoding_ep_study.bin")
		let 
			d = deserialize("tilecoding_ep_study.bin")
			for k in keys(d)
				tilecoding_ep_study.results[k] = d[k]
			end
		end
	end

	if isfile("tilecoding_cont_study.bin")
		let 
			d = deserialize("tilecoding_cont_study.bin")
			for k in keys(d)
				tilecoding_cont_study.results[k] = d[k]
			end
		end
	end

	function run_tilecoding_ep_study(α_list, λ_list; kwargs...)
		for α in α_list for λ in λ_list
			tilecoding_ep_study.update_results!(α, λ; kwargs...)
		end
		end
		return tilecoding_ep_study.results
	end

	function run_tilecoding_cont_study(α_list, λ_list; kwargs...)
		for α in α_list for λ in λ_list
			tilecoding_cont_study.update_results!(α, λ; kwargs...)
		end
		end
		return tilecoding_cont_study.results
	end
end
  ╠═╡ =#

# ╔═╡ c5981ebc-48d2-4a0d-9e3e-042e4e9fbc27
#=╠═╡
function plot_tilecoding_ep_algo_results(study; use_dp = false, num_steps = 100_000, num_trials = Base.Threads.nthreads(), num_tiles = 5, num_tilings = 5)
	function valid_key(k)
		k.use_dp == use_dp &&
		k.num_steps == num_steps &&
		k.num_trials == num_trials &&
		k.num_tiles == num_tiles &&
		k.num_tilings == num_tilings
	end
	
	ks = filter(valid_key, keys(study.results))

	results = Dict(begin
		(α = k.α, λ = k.λ) => -study.results[k]
	 end
	 for k in ks)

	λs = unique(r.λ for r in keys(results)) |> sort
	αs = unique(r.α for r in keys(results)) |> sort

	traces = [begin
			 y = [haskey(results, (α = α, λ = λ)) ? results[(α = α, λ = λ)] : NaN32 for α in αs]
			 x = αs
			 scatter(x = x, y = y, mode = "markers", name = "λ = $λ")
			end
			for λ in λs]
	plot(traces, Layout(xaxis_type = "log"))
end
  ╠═╡ =#

# ╔═╡ 53e58944-bb49-49e1-8d91-d9b485dfe140
#=╠═╡
function plot_tilecoding_cont_algo_results(study; use_dp = false, num_steps = 100_000, α_r̄ = 0.01f0, num_trials = Base.Threads.nthreads(), num_tiles = 5, num_tilings = 5)
	function valid_key(k)
		k.use_dp == use_dp &&
		k.num_steps == num_steps &&
		k.num_trials == num_trials &&
		k.α_r̄ == α_r̄ &&
		k.num_tiles == num_tiles &&
		k.num_tilings == num_tilings
	end
	
	ks = filter(valid_key, keys(study.results))

	results = Dict(begin
		(α = k.α, λ = k.λ) => study.results[k] |> inv
	 end
	 for k in ks)

	λs = unique(r.λ for r in keys(results)) |> sort
	αs = unique(r.α for r in keys(results)) |> sort

	traces = [begin
			 y = [haskey(results, (α = α, λ = λ)) ? results[(α = α, λ = λ)] : NaN32 for α in αs]
			 x = αs
			 scatter(x = x, y = y, mode = "markers", name = "λ = $λ")
			end
			for λ in λs]
	plot(traces, Layout(xaxis_type = "log"))
end
  ╠═╡ =#

# ╔═╡ 013db3d5-34e9-4447-8f4b-59c6f908e6e8
md"""
##### Sarsa λ Parameter Study
"""

# ╔═╡ dda86adb-96fb-4d46-8b5d-4f9713066dd0
#=╠═╡
@bind sarsa_ep_tiles PlutoUI.combine() do Child
	md"""
	Num Tiles: $(Child(:num_tiles, NumberField(1:32, default = 8)))
	Num Tilings: $(Child(:num_tilings, NumberField(1:32, default = 8)))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ d80daf76-0f55-491b-a760-048b43ae3d74
#=╠═╡
begin
	run_tilecoding_ep_study(2f0 .^ (-12:-4), [0.5f0, 0.8f0, 0.9f0, 0.99f0]; sarsa_ep_tiles...)
	plot_tilecoding_ep_algo_results(tilecoding_ep_study; sarsa_ep_tiles...)
end
  ╠═╡ =#

# ╔═╡ 616cd58d-bd13-4ee8-a08d-2ff0ba2ebad2
md"""
##### DP λ Parameter Study
"""

# ╔═╡ db84dc50-7a04-4475-8ea7-412307654b0d
#=╠═╡
@bind dp_ep_tiles PlutoUI.combine() do Child
	md"""
	Num Tiles: $(Child(:num_tiles, NumberField(1:32, default = 16)))
	Num Tilings: $(Child(:num_tilings, NumberField(1:32, default = 8)))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ 0d735858-c628-4a41-91b4-e86fe6854ea7
#=╠═╡
begin
	run_tilecoding_ep_study(2f0 .^ (-14:-6), [0.5f0, 0.9f0, 0.95f0, 0.99f0]; use_dp = true, dp_ep_tiles...)
	plot_tilecoding_ep_algo_results(tilecoding_ep_study; use_dp = true, dp_ep_tiles...)
end
  ╠═╡ =#

# ╔═╡ 26ea259e-2cad-4e42-b6dc-c38befd2e3cb
md"""
##### Best Tilecoding Episodic Results
"""

# ╔═╡ be8e6e0d-04d0-4a1c-9d76-1ca6fb688fcb
tilecoding_ep_best = mountaincar_tilecoding.train_ep(1f-4, 0.99f0; num_steps = 1_000_000, use_dp = true, num_tiles = 16, num_tilings = 32, ϵ = 0.01f0)

# ╔═╡ de86e2da-908e-44a9-998b-761c93297b66
md"""
#### Continuing Tilecoding Results
"""

# ╔═╡ bbfe7723-c811-4389-bc8e-9a9e145ac872
md"""
##### Sarsa λ Parameter Study
"""

# ╔═╡ ccbeb628-6e90-4b00-abce-40aa4549c23a
#=╠═╡
@bind sarsa_cont_tiles PlutoUI.combine() do Child
	md"""
	Num Tiles: $(Child(:num_tiles, NumberField(1:32, default = 8)))
	Num Tilings: $(Child(:num_tilings, NumberField(1:32, default = 4)))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ 4c7efa25-a062-42bc-9685-5212fd00f398
#=╠═╡
begin
	run_tilecoding_cont_study(2f0 .^ (-10:-6), [0.5f0, 0.8f0, 0.7f0, 0.9f0, 0.99f0]; sarsa_cont_tiles...)
	plot_tilecoding_cont_algo_results(tilecoding_cont_study; sarsa_cont_tiles...)
end
  ╠═╡ =#

# ╔═╡ d4867d9f-5fd3-44d6-8d46-e383e99124dd
md"""
##### DP λ Parameter Study
"""

# ╔═╡ 8b941289-939f-4565-91dc-29756a19d0ea
#=╠═╡
@bind dp_cont_tiles PlutoUI.combine() do Child
	md"""
	Num Tiles: $(Child(:num_tiles, NumberField(1:32, default = 8)))
	Num Tilings: $(Child(:num_tilings, NumberField(1:32, default = 4)))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ feef74be-642e-4f4b-8bbf-7cff678ced8b
#=╠═╡
begin
	run_tilecoding_cont_study(2f0 .^ (-14:-8), [0f0, 0.2f0, 0.5f0, 0.6f0, 0.9f0]; use_dp = true, dp_cont_tiles...)
	plot_tilecoding_cont_algo_results(tilecoding_cont_study; use_dp = true, dp_cont_tiles...)
end
  ╠═╡ =#

# ╔═╡ 50c1669f-fe46-4729-b867-f8bb2784de47
const tilecoding_cont_best = mountaincar_tilecoding.train_cont(1f-4, 0.5f0; α_r̄ = 0.01f0, num_steps = 100_000, use_dp = true, num_tiles = 8, num_tilings = 16)

# ╔═╡ fdfd5f7c-504b-492a-aca3-4690ed17f56f
md"""
#### Tilecoding Solution Study
"""

# ╔═╡ 4de6f442-370b-47ff-a59a-06bac7e8fbad
function train_tile_value_grid(α, λ; num_steps = 200_000, tile_min = 1, tile_max = 4, tilings_min = 0, tilings_max = 4, kwargs...)
	f(num_tiles, num_tilings) = mountaincar_tilecoding.train_ep(α, λ; num_steps = num_steps, num_tiles = num_tiles, num_tilings = num_tilings, kwargs...)
	
	tiles = 2 .^ (tile_min:tile_max)
	tilings = 2 .^ (tilings_min:tilings_max)
	args = [(n1, n2) for n1 in tiles for n2 in tilings ]
	grid = args |> Map(t -> t => f(t...)) |> tcollect |> Dict
	(grid = grid, tiles = tiles, tilings = tilings, args = args)
end

# ╔═╡ 1af31a32-399f-4568-8748-42224fafd6ed
const tile_value_grid = train_tile_value_grid(1f-4, 0.99f0; tilings_max = 5, tile_max = 5, use_dp = true, num_steps = 1_000_000, ϵ = 0.01f0)

# ╔═╡ dd310782-7f49-463f-800c-db8f206b49a5
md"""
### Non-linear Methods
"""

# ╔═╡ 04e267e7-a994-4b8f-b25f-dc845a93d909
#=╠═╡
@bind save_fcann_value_params CounterButton("Save Non-linear Value Parameters")
  ╠═╡ =#

# ╔═╡ 5caad033-32c0-4502-bbaa-746da59278e2
begin
	const mountaincar_fcann_value_episodic_results = if isfile("mountaincar_fcann_value_episodic_results.bin")
		deserialize("mountaincar_fcann_value_episodic_results.bin")
	else
		Dict{NamedTuple, NamedTuple}()
	end
end

# ╔═╡ a7ee78f2-bb3c-4536-8940-450e4c921fc7
begin
	const mountaincar_fcann_value_continuing_results = if isfile("mountaincar_fcann_value_continuing_results.bin")
		deserialize("mountaincar_fcann_value_continuing_results.bin")
	else
		Dict{NamedTuple, NamedTuple}()
	end
end

# ╔═╡ 81363dfd-868a-432f-9b9d-0a730d7ec745
#=╠═╡
if save_fcann_value_params > 0
	serialize("mountaincar_fcann_value_episodic_results.bin", mountaincar_fcann_value_episodic_results)
	serialize("mountaincar_fcann_value_continuing_results.bin", mountaincar_fcann_value_continuing_results)
end
  ╠═╡ =#

# ╔═╡ 38d91348-c574-46a3-829a-2f14766a717d
begin
	function run_mountaincar_λ_fcann(mdp::StateMDP, γ::T, α::T, λ::T, hidden_layers::Vector{Int64}; use_dp = false, reslayers = 0, num_steps = 50_000, newparams::Bool = true, kwargs...) where T<:Real 
		key = (use_dp = use_dp, hidden_layers = hidden_layers, reslayers = reslayers)
		params = if !newparams && haskey(mountaincar_fcann_value_episodic_results, key)
			mountaincar_fcann_value_episodic_results[key].final_parameters
		else
			output_size = use_dp ? 1 : 3
			initialize_fcann_params(2, hidden_layers, output_size, reslayers, true)
		end
		if iszero(λ)
			algo = use_dp ? semi_gradient_dp_fcann : semi_gradient_sarsa_fcann
			output = algo(mdp, γ, typemax(Int64), num_steps, copy(mountaincar_simple_feature_setup.feature_vector), mountaincar_simple_feature_setup.update_feature_vector!, hidden_layers; α = α, reslayers = reslayers, parameters = params, kwargs...)
		else
			algo = use_dp ? dp_λ_fcann : sarsa_λ_fcann
			output = algo(mdp, γ, λ, typemax(Int64), num_steps, copy(mountaincar_simple_feature_setup.feature_vector), mountaincar_simple_feature_setup.update_feature_vector!, hidden_layers; α = α, reslayers = reslayers, parameters = params, kwargs...)
		end
		mountaincar_fcann_value_episodic_results[key] = output
	end
	
	function run_mountaincar_λ_fcann(mdp::StateMDP, α::T, λ::T, hidden_layers::Vector{Int64}; use_dp = false, reslayers = 0, num_steps = 50_000, newparams::Bool = true, kwargs...) where T<:Real 
		key = (use_dp = use_dp, hidden_layers = hidden_layers, reslayers = reslayers)
		params = if !newparams && haskey(mountaincar_fcann_value_continuing_results, key)
			mountaincar_fcann_value_continuing_results[key].final_parameters
		else
			output_size = use_dp ? 1 : 3
			initialize_fcann_params(2, hidden_layers, output_size, reslayers, true)
		end
		# if iszero(λ)
		# 	algo = use_dp ? semi_gradient_differential_dp_fcann : semi_gradient_differential_sarsa_fcann
		# 	output = algo(mdp, num_steps, copy(mountaincar_simple_feature_setup.feature_vector), mountaincar_simple_feature_setup.update_feature_vector!, hidden_layers; α = α, reslayers = reslayers, parameters = params, kwargs...)
		# else
			algo = use_dp ? dp_λ_fcann : sarsa_λ_fcann
			output = algo(mdp, λ, num_steps, copy(mountaincar_simple_feature_setup.feature_vector), mountaincar_simple_feature_setup.update_feature_vector!, hidden_layers; α = α, reslayers = reslayers, parameters = params, kwargs...)
		# end
		mountaincar_fcann_value_continuing_results[key] = output
	end
end

# ╔═╡ a11ab2ab-54d0-4d5a-846a-c16af29c0d51
function Base.copy!(dst::FCANNParams{T}, src::FCANNParams{T}) where T<:Real
	for i in eachindex(src.weights[1])
		for j in 1:2
			dst.weights[j][i] .= src.weights[j][i]
		end
	end
end

# ╔═╡ a6ca67c7-2fb2-4034-b761-b593177d9dce
#=╠═╡
function setup_mountaincar_fcann()
	function train1(α, λ; layer_size::Integer = 4, num_layers::Integer = 2, kwargs...) 
		hidden_layers = fill(layer_size, num_layers)
		run_mountaincar_λ_fcann(mountaincar_mdps[1], 1f0, α, λ, hidden_layers; kwargs...)
	end

	function train2(α, λ; layer_size::Integer = 4, num_layers::Integer = 2, kwargs...) 
		hidden_layers = fill(layer_size, num_layers)
		run_mountaincar_λ_fcann(mountaincar_mdps[3], α, λ, hidden_layers; kwargs...)
	end

	function calculate_episodic_reward_metric(output::NamedTuple)
		rewards = output.episode_rewards
		isempty(rewards) && return -Inf32
		l = length(rewards)
		l2 = round(Int64, l / 2)
		mean(view(rewards, l2:l))
	end

	function calculate_continuing_reward_metric(output::NamedTuple)
		l = length(output.reward_history)
		l2 = round(Int64, l / 2)
		mean(view(output.reward_history, l2:l))
	end

	function train_exhaustive(isepisodic::Bool, α, λ; kwargs...)
		f = isepisodic ? train1 : train2
		f2 = isepisodic ? calculate_episodic_reward_metric : calculate_continuing_reward_metric
		@info "Training with learning rate $α until results fail to improve"
		results1 = f(0f0, λ; num_steps = 1, kwargs..., newparams = false)
		π(s) = results1.value_function(s).maximizing_action
		mean_step_reward = runepisode(mountaincar_mdps[3]; π = π, max_steps = 1_000_000)[3] |> mean
		avg_reward1 = isepisodic ? -inv(mean_step_reward) : mean_step_reward
		@info "Reference reward is $avg_reward1"

		params = copy(results1.final_parameters)
		results2 = f(α, λ; kwargs..., newparams = false)
		avg_reward2 = f2(results2)
		trial = 2
		while avg_reward2 > avg_reward1
			copy!(params, results2.final_parameters)
			@info "On trial $trial, reward improved from $avg_reward1 to $avg_reward2"
			avg_reward1 = avg_reward2
			results1 = results2
			results2 = f(α, λ; kwargs..., newparams = false)
			trial += 1
			avg_reward2 = f2(results2)
		end
		@info "Concluded training with learning rate of $α after $(trial - 1) trials with an average reward of $avg_reward1"
		copy!(results2.final_parameters, params)
		return (output = results1, performance = avg_reward1)
	end

	function train_rate_decay(isepisodic::Bool, α_init, λ; kwargs...)
		@info "Training with an initial learning rate of $α_init and decaying by 50% until failure to improve"
		α = α_init
		results1, performance1 = train_exhaustive(isepisodic, α, λ; kwargs...)
		params = copy(results1.final_parameters)

		α /= 2
		results2, performance2 = train_exhaustive(isepisodic, α, λ; kwargs...)

		trial = 2
		while performance2 > performance1
			copy!(params, results2.final_parameters)
			@info "After $trial rounds of learning rate decay, performance improved from $performance1 to $performance2"
			@info "Reducing learning rate to $(α/2)"
			α /= 2
			results1 = results2
			performance1 = performance2
			results2, performance2 = train_exhaustive(isepisodic, α, λ; kwargs...)
			trial += 1
		end
		@info "Concluded after $(trial - 1) rounds of learning rate decay with a learning rate of $(α*2) and a performance of $performance1"
		copy!(results2.final_parameters, params)
		return results1
	end

	train_exhaustive1(α, λ; kwargs...) = train_exhaustive(true, α, λ; kwargs...)
	train_exhaustive2(α, λ; kwargs...) = train_exhaustive(false, α, λ; kwargs...)
	train_rate_decay1(α, λ; kwargs...) = train_rate_decay(true, α, λ; kwargs...)
	train_rate_decay2(α, λ; kwargs...) = train_rate_decay(false, α, λ; kwargs...)

	return (train_ep = train1, train_cont = train2, train_ep_exhaustive = train_exhaustive1, train_cont_exhaustive = train_exhaustive2, train_ep_rate_decay = train_rate_decay1, train_cont_rate_decay = train_rate_decay2)
end
  ╠═╡ =#

# ╔═╡ b7c60bbb-a599-4d20-9f50-0d80b3a2649f
#=╠═╡
const mountaincar_fcann = setup_mountaincar_fcann()
  ╠═╡ =#

# ╔═╡ c4c1569f-5a33-4ea2-a41b-aef32c9b9cce
#=╠═╡
begin
	fcann_ep_trial(α, λ; kwargs...) = mountaincar_fcann.train_ep(α, λ; kwargs...).episode_rewards |> mean
	fcann_cont_trial(α, λ; kwargs...) = mountaincar_fcann.train_cont(α, λ; kwargs...).reward_history |> mean

	const fcann_ep_study = setup_parameter_study(fcann_ep_trial, (:α, :λ), (use_dp = false, num_steps = 100_000, layer_size = 8, num_layers = 2, reslayers = 1, ϵ = 0.01f0))
	const fcann_cont_study = setup_parameter_study(fcann_cont_trial, (:α, :λ), (use_dp = false, num_steps = 100_000, α_r̄ = 0.01f0, layer_size = 8, num_layers = 2, reslayers = 1, ϵ = 0.01f0))

	if isfile("fcann_ep_study.bin")
		let 
			d = deserialize("fcann_ep_study.bin")
			for k in keys(d)
				fcann_ep_study.results[k] = d[k]
			end
		end
	end

	if isfile("fcann_cont_study.bin")
		let 
			d = deserialize("fcann_cont_study.bin")
			for k in keys(d)
				fcann_cont_study.results[k] = d[k]
			end
		end
	end

	function run_fcann_ep_study(α_list, λ_list; kwargs...)
		for α in α_list for λ in λ_list
			fcann_ep_study.update_results!(α, λ; kwargs...)
		end
		end
		return fcann_ep_study.results
	end

	function run_fcann_cont_study(α_list, λ_list; kwargs...)
		for α in α_list for λ in λ_list
			fcann_cont_study.update_results!(α, λ; kwargs...)
		end
		end
		return fcann_cont_study.results
	end
end
  ╠═╡ =#

# ╔═╡ 2b45a044-3b15-4e67-b63b-2b06094e66c3
#=╠═╡
function plot_fcann_ep_algo_results(study; use_dp = false, num_steps = 100_000, num_trials = Base.Threads.nthreads(), layer_size = 8, num_layers = 2)
	function valid_key(k)
		k.use_dp == use_dp &&
		k.num_steps == num_steps &&
		k.num_trials == num_trials &&
		k.layer_size == layer_size &&
		k.num_layers == num_layers
	end
	
	ks = filter(valid_key, keys(study.results))

	results = Dict(begin
		(α = k.α, λ = k.λ) => -study.results[k]
	 end
	 for k in ks)

	λs = unique(r.λ for r in keys(results)) |> sort
	αs = unique(r.α for r in keys(results)) |> sort

	traces = [begin
			 y = [haskey(results, (α = α, λ = λ)) ? results[(α = α, λ = λ)] : NaN32 for α in αs]
			 x = αs
			 scatter(x = x, y = y, mode = "markers", name = "λ = $λ")
			end
			for λ in λs]
	plot(traces, Layout(xaxis_type = "log"))
end
  ╠═╡ =#

# ╔═╡ 7d0678f1-f5ef-43f8-981c-f0f5f5e63293
#=╠═╡
function plot_fcann_cont_algo_results(study; use_dp = false, num_steps = 100_000, α_r̄ = 0.01f0, num_trials = Base.Threads.nthreads(), layer_size = 8, num_layers = 2, ymin = nothing, ymax = nothing)
	function valid_key(k)
		k.use_dp == use_dp &&
		k.num_steps == num_steps &&
		k.num_trials == num_trials &&
		k.α_r̄ == α_r̄ &&
		k.layer_size == layer_size &&
		k.num_layers == num_layers
	end
	
	ks = filter(valid_key, keys(study.results))

	results = Dict(begin
		(α = k.α, λ = k.λ) => study.results[k] |> inv
	 end
	 for k in ks)

	λs = unique(r.λ for r in keys(results)) |> sort
	αs = unique(r.α for r in keys(results)) |> sort

	traces = [begin
			 y = [haskey(results, (α = α, λ = λ)) ? results[(α = α, λ = λ)] : NaN32 for α in αs]
			 x = αs
			 scatter(x = x, y = y, mode = "markers", name = "λ = $λ")
			end
			for λ in λs]
	plot(traces, Layout(xaxis_type = "log", yaxis_range = [ymin, ymax]))
end
  ╠═╡ =#

# ╔═╡ 4b8d413e-72da-4232-ac6b-2125f79c96cd
md"""
#### Parameter Study
"""

# ╔═╡ c671702c-ba40-4cc0-b5b6-adf674fe9825
#=╠═╡
@bind sarsa_ep_layers PlutoUI.combine() do Child
	md"""
	##### Episodic Training
	
	Layer Size: $(Child(:layer_size, NumberField(2:64, default = 16)))
	Num Layers: $(Child(:num_layers, NumberField(2:32, default = 4)))
	Use DP: $(Child(:use_dp, CheckBox()))
	Num Steps: $(Child(:num_steps, NumberField(10_000:10_000_000, default = 100_000)))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ 2a722083-270f-4c56-bd86-b336fd4a2883
#=╠═╡
begin
	run_fcann_ep_study(2f0 .^ (-18:-10), [0.0f0, 0.1f0, 0.2f0, 0.5f0, 0.6f0]; sarsa_ep_layers...)
	plot_fcann_ep_algo_results(fcann_ep_study; sarsa_ep_layers...)
end
  ╠═╡ =#

# ╔═╡ d633893f-1abb-4dde-9e1a-19ffbbc0cd98
#=╠═╡
@bind sarsa_cont_layers PlutoUI.combine() do Child
	md"""
	##### Continuing Training
	
	Layer Size: $(Child(:layer_size, NumberField(2:64, default = 16)))
	Num Layers: $(Child(:num_layers, NumberField(2:32, default = 4)))
	``\alpha_{\bar{r}}`` : $(Child(:α_r̄, NumberField(0.001f0:0.001f0:0.1f0, default = 0.001f0)))
	Use DP: $(Child(:use_dp, CheckBox()))
	Num Steps: $(Child(:num_steps, NumberField(10_000:10_000_000, default = 100_000)))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ 16bb7a14-0d99-4878-8f0d-075b342a524a
#=╠═╡
begin
	run_fcann_cont_study(2f0 .^ (-8:-1), [0.9f0, 0.95f0, 0.99f0]; sarsa_cont_layers...)
	plot_fcann_cont_algo_results(fcann_cont_study; sarsa_cont_layers..., ymin = 0, ymax = 1000)
end
  ╠═╡ =#

# ╔═╡ 0ea1cfb0-09cc-4848-9d0d-558085c63cc6
md"""
#### Trained Example
"""

# ╔═╡ 74150cad-3e4f-4e4d-b819-bb15769fe6d0
#=╠═╡
const fcann_value_best = mountaincar_fcann.train_cont_rate_decay(0.02f0, 0.99f0; num_steps = 1_000_000, layer_size = 64, num_layers = 8, α_r̄ = 0.001f0, reslayers = 1, ϵ = 0.01f0, use_dp = true)
  ╠═╡ =#

# ╔═╡ effaa34e-24f4-48b6-9169-274e92aacdc9
md"""
#### Performance Profiling
"""

# ╔═╡ b0056c84-b76e-4898-ad70-753e5083f965
md"""
##### Sarsa λ
"""

# ╔═╡ 2a48428d-fd0b-4d8f-899f-93377de393e3
#=╠═╡
@plutoprofview mountaincar_fcann.train_cont(0.0f0, 0.99f0; num_steps = 10_000, layer_size = 64, num_layers = 2, α_r̄ = 0.001f0, reslayers = 1, ϵ = 0.01f0, newparams = false)
  ╠═╡ =#

# ╔═╡ 24a2050e-750c-4e77-85e9-c7d1859f5b3f
#=╠═╡
@plutoprofview mountaincar_fcann.train_cont(0.0f0, 0.99f0; num_steps = 1_000, layer_size = 512, num_layers = 2, α_r̄ = 0.001f0, reslayers = 1, ϵ = 0.01f0, newparams = false)
  ╠═╡ =#

# ╔═╡ ac1a4222-46e0-4242-bd7e-f1aa1ae15341
#=╠═╡
@plutoprofview mountaincar_fcann.train_ep(0.0f0, 0.99f0; num_steps = 1_000, layer_size = 64, num_layers = 2, reslayers = 1, ϵ = 0.01f0, newparams = false, use_gpu=true)
  ╠═╡ =#

# ╔═╡ c73139ce-1090-4522-b4bb-de3b553dd468
#=╠═╡
@plutoprofview mountaincar_fcann.train_ep(0.0f0, 0.99f0; num_steps = 1_000, layer_size = 4096, num_layers = 2, reslayers = 1, ϵ = 0.01f0, newparams = false, use_gpu=true)
  ╠═╡ =#

# ╔═╡ 1dc43eb9-74f6-4b40-8976-604f728777f0
#=╠═╡
@btime mountaincar_fcann.train_ep(0.0f0, 0.99f0; num_steps = 100, layer_size = 2048, num_layers = 2, reslayers = 1, ϵ = 0.01f0, newparams = false, use_gpu=true)
  ╠═╡ =#

# ╔═╡ 47bc9026-130e-4cbb-ad7c-ed02047ac036
#=╠═╡
@btime mountaincar_fcann.train_ep(0.0f0, 0.99f0; num_steps = 100, layer_size = 2048, num_layers = 2, reslayers = 1, ϵ = 0.01f0, newparams = false, use_gpu=false)
  ╠═╡ =#

# ╔═╡ 5e4e5fd1-dcea-4a9d-8698-880c4a110840
#=╠═╡
@btime mountaincar_fcann.train_ep(0.0f0, 0.99f0; num_steps = 100, layer_size = 1024, num_layers = 2, reslayers = 1, ϵ = 0.01f0, newparams = false, use_gpu=true)
  ╠═╡ =#

# ╔═╡ 5789ab7c-5062-4ad7-bb9f-34a50e4cf0fe
#=╠═╡
@btime mountaincar_fcann.train_ep(0.0f0, 0.99f0; num_steps = 100, layer_size = 1024, num_layers = 2, reslayers = 1, ϵ = 0.01f0, newparams = false, use_gpu=false)
  ╠═╡ =#

# ╔═╡ 2c3f066c-744c-4861-abf8-4cab5b99d9ff
#=╠═╡
@btime mountaincar_fcann.train_ep(0.0f0, 0.99f0; num_steps = 100, layer_size = 512, num_layers = 8, reslayers = 1, ϵ = 0.01f0, newparams = false, use_gpu=true)
  ╠═╡ =#

# ╔═╡ 8dd794be-9145-4cf7-9df0-40d75b541783
#=╠═╡
@btime mountaincar_fcann.train_ep(0.0f0, 0.99f0; num_steps = 100, layer_size = 512, num_layers = 8, reslayers = 1, ϵ = 0.01f0, newparams = false, use_gpu=false)
  ╠═╡ =#

# ╔═╡ 721026dc-ac4a-4fdd-8c63-56c05244272e
#=╠═╡
@btime mountaincar_fcann.train_ep(0.0f0, 0.99f0; num_steps = 100, layer_size = 256, num_layers = 128, reslayers = 1, ϵ = 0.01f0, newparams = false, use_gpu=true)
  ╠═╡ =#

# ╔═╡ 7ff78994-872a-46e4-85ce-363a9cbf4071
#=╠═╡
@btime mountaincar_fcann.train_ep(0.0f0, 0.99f0; num_steps = 100, layer_size = 256, num_layers = 128, reslayers = 1, ϵ = 0.01f0, newparams = false, use_gpu=false)
  ╠═╡ =#

# ╔═╡ 3d9404cc-9491-4d48-a56b-88174e91507a
md"""
##### DP λ
"""

# ╔═╡ 07116240-fe0c-499f-87cf-02d8d316f546
#=╠═╡
@plutoprofview mountaincar_fcann.train_cont(0.0f0, 0.99f0; num_steps = 10_000, layer_size = 64, num_layers = 2, α_r̄ = 0.001f0, reslayers = 1, ϵ = 0.01f0, use_dp = true, newparams = false)
  ╠═╡ =#

# ╔═╡ 41ead506-92a0-4620-ab15-22678898e169
#=╠═╡
@plutoprofview mountaincar_fcann.train_cont(0.0f0, 0.99f0; num_steps = 1_000, layer_size = 512, num_layers = 2, α_r̄ = 0.001f0, reslayers = 1, ϵ = 0.01f0, use_dp = true, newparams = false)
  ╠═╡ =#

# ╔═╡ 81337bb1-d7d5-46a3-8b2f-ac0626326f24
#=╠═╡
@plutoprofview mountaincar_fcann.train_cont(0.0f0, 0.99f0; num_steps = 1_000, layer_size = 64, num_layers = 2, α_r̄ = 0.001f0, reslayers = 1, ϵ = 0.01f0, use_dp = true, newparams = false, use_gpu=true)
  ╠═╡ =#

# ╔═╡ e7b02c90-b394-413b-918c-6076edb334e1
#=╠═╡
@plutoprofview mountaincar_fcann.train_cont(0.0f0, 0.99f0; num_steps = 1_000, layer_size = 512, num_layers = 2, α_r̄ = 0.001f0, reslayers = 1, ϵ = 0.01f0, use_dp = true, newparams = false, use_gpu=true)
  ╠═╡ =#

# ╔═╡ 2584ce25-545c-4190-aa8f-2a0adb94cd5a
#=╠═╡
@plutoprofview mountaincar_fcann.train_cont(0.0f0, 0.99f0; num_steps = 1_000, layer_size = 2048, num_layers = 4, α_r̄ = 0.001f0, reslayers = 1, ϵ = 0.01f0, use_dp = true, newparams = false, use_gpu=true)
  ╠═╡ =#

# ╔═╡ 9249a644-b6e9-4386-8d85-d9ce11d80519
#=╠═╡
@btime mountaincar_fcann.train_ep(0.0f0, 0.99f0; num_steps = 100, layer_size = 2048, num_layers = 2, reslayers = 1, ϵ = 0.01f0, use_dp = true, newparams = false, use_gpu=true)
  ╠═╡ =#

# ╔═╡ b31e6203-c3ad-465b-98e7-5606bd468401
#=╠═╡
@btime mountaincar_fcann.train_ep(0.0f0, 0.99f0; num_steps = 100, layer_size = 2048, num_layers = 2, reslayers = 1, ϵ = 0.01f0, use_dp = true, newparams = false, use_gpu=false)
  ╠═╡ =#

# ╔═╡ e123d347-2ed0-4db2-910c-73d11850c7f9
#=╠═╡
@btime mountaincar_fcann.train_ep(0.0f0, 0.99f0; num_steps = 100, layer_size = 1024, num_layers = 2, reslayers = 1, ϵ = 0.01f0, use_dp = true, newparams = false, use_gpu=true)
  ╠═╡ =#

# ╔═╡ ddd08669-d1eb-427c-a76f-8027bf5e9875
#=╠═╡
@btime mountaincar_fcann.train_ep(0.0f0, 0.99f0; num_steps = 100, layer_size = 1024, num_layers = 2, reslayers = 1, ϵ = 0.01f0, use_dp = true, newparams = false, use_gpu=false)
  ╠═╡ =#

# ╔═╡ 22a27fde-8f2a-4301-afa6-4e08388f04e4
#=╠═╡
@btime mountaincar_fcann.train_ep(0.0f0, 0.99f0; num_steps = 100, layer_size = 512, num_layers = 8, reslayers = 1, ϵ = 0.01f0, use_dp = true, newparams = false, use_gpu=true)
  ╠═╡ =#

# ╔═╡ 7b1f5872-f0de-4b41-ab6b-aa2f8b1598c2
#=╠═╡
@btime mountaincar_fcann.train_ep(0.0f0, 0.99f0; num_steps = 100, layer_size = 512, num_layers = 8, reslayers = 1, ϵ = 0.01f0, use_dp = true, newparams = false, use_gpu=false)
  ╠═╡ =#

# ╔═╡ 27322c95-f130-4669-b9da-8195cdafa460
#=╠═╡
function train_fcann_value_grid(α, λ; num_steps = 1_000_000, n_min = 2, n_max = 6, layers_min = 1, layers_max = 3, reslayers = 1, kwargs...)
	function f(layer_size, num_layers) 
		mountaincar_fcann.train_cont(α, λ; num_steps = num_steps, layer_size = layer_size, num_layers = num_layers, reslayers = reslayers, ϵ = 0.8f0, kwargs..., newparams = true)
		mountaincar_fcann.train_cont_rate_decay(α, λ; num_steps = num_steps, layer_size = layer_size, num_layers = num_layers, reslayers = reslayers, ϵ = 0.8f0, kwargs...)
		mountaincar_fcann.train_cont_rate_decay(α, λ; num_steps = num_steps, layer_size = layer_size, num_layers = num_layers, reslayers = reslayers, ϵ = 0.01f0, kwargs...)
	end
	
	n = 2 .^ (n_min:n_max)
	layers = 2 .^ (layers_min:layers_max)
	args = [(n1, n2) for n1 in n for n2 in layers]
	grid = args |> Map(t -> t => f(t...)) |> tcollect |> Dict
	(grid = grid, layer_size = n, num_layers = layers, args = args)
end
  ╠═╡ =#

# ╔═╡ 8ca72314-7ea9-4864-8698-594f57e69f31
# ╠═╡ show_logs = false
#=╠═╡
const fcann_sarsa_value_grid = train_fcann_value_grid(0.04f0, 0.9f0)
  ╠═╡ =#

# ╔═╡ 8eb01d9b-836b-42c3-850a-e20a5875d2e1
# ╠═╡ show_logs = false
#=╠═╡
const fcann_dp_value_grid = train_fcann_value_grid(0.04f0, 0.9f0; use_dp = true)
  ╠═╡ =#

# ╔═╡ 1b9078af-d7d1-4322-897e-89452ff8a4de
md"""
## Policy Gradient Methods
"""

# ╔═╡ 3cd7f197-a86d-4210-b932-6bb1c8e5b9ec
md"""
### Actor-Critic with Eligibility Traces
"""

# ╔═╡ 3163a090-ca95-4df4-9a0e-33505ee6de0e
begin
	run_mountaincar_ac_linear(mdp::StateMDP, γ::T, α_θ::T, α_w::T, λ_θ::T, λ_w::T, feature_vector, update_feature_vector!::Function; num_steps = 50_000, kwargs...) where T<:Real = actor_critic_with_eligibility_traces_linear(mdp, γ, λ_θ, λ_w, typemax(Int64), num_steps, feature_vector, update_feature_vector!; α_θ = α_θ, α_w = α_w, kwargs...)
	
	run_mountaincar_ac_linear(mdp::StateMDP, α_θ::T, α_w::T, λ_θ::T, λ_w::T, feature_vector, update_feature_vector!::Function; num_steps = 50_000, kwargs...) where T<:Real = actor_critic_with_eligibility_traces_linear(mdp, λ_θ, λ_w, num_steps, feature_vector, update_feature_vector!; α_θ = α_θ, α_w = α_w, kwargs...)
end

# ╔═╡ ba397f1d-c8fc-4521-a56e-2c71817ab1f8
function setup_mountaincar_ac_simple()
	x = mountaincar_simple_feature_setup.feature_vector
	f! = mountaincar_simple_feature_setup.update_feature_vector!
	train1(α_θ, α_w, λ_θ, λ_w; kwargs...) = run_mountaincar_ac_linear(mountaincar_mdps[1], 1f0, α_θ, α_w, λ_θ, λ_w, copy(x), f!; kwargs...)
	train2(α_θ, α_w, λ_θ, λ_w; kwargs...) = run_mountaincar_ac_linear(mountaincar_mdps[3], α_θ, α_w, λ_θ, λ_w, copy(x), f!; kwargs...)
	return (train_ep = train1, train_cont = train2)
end

# ╔═╡ b20dafb1-282e-4fe7-ae19-5c6101a59965
function setup_mountaincar_ac_tilecoding()
	function train1(α_θ, α_w, λ_θ, λ_w; num_tiles::Integer = 10, num_tilings::Integer = 10, kwargs...) 
		setup = tile_coding_feature_setup(mountaincar_mdps[1], MountainCarTask.min_vals, MountainCarTask.max_vals, (num_tiles, num_tiles), num_tilings)
		run_mountaincar_ac_linear(mountaincar_mdps[1], 1f0, α_θ, α_w, λ_θ, λ_w, setup.feature_vector, setup.update_feature_vector!; kwargs...)
	end

	function train2(α_θ, α_w, λ_θ, λ_w; num_tiles::Integer = 10, num_tilings::Integer = 10, kwargs...) 
		setup = tile_coding_feature_setup(mountaincar_mdps[3], MountainCarTask.min_vals, MountainCarTask.max_vals, (num_tiles, num_tiles), num_tilings)
		run_mountaincar_ac_linear(mountaincar_mdps[3], α_θ, α_w, λ_θ, λ_w, setup.feature_vector, setup.update_feature_vector!; kwargs...)
	end
	return (train_ep = train1, train_cont = train2)
end

# ╔═╡ f95db313-e4af-4c2b-af24-3c2ad7459add
const mountaincar_ac_simple = setup_mountaincar_ac_simple()

# ╔═╡ 43966b13-bc9c-46d8-a57e-dad0cfbb690f
const mountaincar_ac_tilecoding = setup_mountaincar_ac_tilecoding()

# ╔═╡ 2b605f0f-f19d-4ca0-b145-4af6fa0ab346
#=╠═╡
simple_ep_ac_trial(α_θ, α_w, λ_θ, λ_w; kwargs...) = mountaincar_ac_simple.train_ep(α_θ, α_w, λ_θ, λ_w; kwargs...).episode_rewards |> mean
  ╠═╡ =#

# ╔═╡ 8e9874ba-c8c0-4882-993c-382f15f5c5ea
#=╠═╡
simple_cont_ac_trial(α_θ, α_w, λ_θ, λ_w; kwargs...) = mountaincar_ac_simple.train_cont(α_θ, α_w, λ_θ, λ_w; kwargs...).reward_history |> mean
  ╠═╡ =#

# ╔═╡ b5d37679-eac3-4acd-b16e-3bc7c7a2a15b
#=╠═╡
begin
	simple_ep_ac_study = setup_parameter_study(simple_ep_ac_trial, (:α_θ, :α_w, :λ_θ, :λ_w), (num_steps = 100_000,))
	if isfile("simple_ep_ac_study.bin")
		let 
			d = deserialize("simple_ep_ac_study.bin")
			for k in keys(d)
				simple_ep_ac_study.results[k] = d[k]
			end
		end
	end
	simple_cont_ac_study = setup_parameter_study(simple_cont_ac_trial, (:α_θ, :α_w, :λ_θ, :λ_w), (num_steps = 100_000, α_r̄ = 0.01f0))
		if isfile("simple_cont_ac_study.bin")
		let 
			d = deserialize("simple_cont_ac_study.bin")
			for k in keys(d)
				simple_cont_ac_study.results[k] = d[k]
			end
		end
	end
end
  ╠═╡ =#

# ╔═╡ 416eb8cd-33a5-4e4d-ad78-dfa4a127a8b4
#=╠═╡
tilecoding_ep_ac_trial(α_θ, α_w, λ_θ, λ_w; kwargs...) = mountaincar_ac_tilecoding.train_ep(α_θ, α_w, λ_θ, λ_w; kwargs...).episode_rewards |> mean
  ╠═╡ =#

# ╔═╡ b8e17616-7b71-494a-a90e-999fc5c21989
#=╠═╡
tilecoding_cont_ac_trial(α_θ, α_w, λ_θ, λ_w; kwargs...) = mountaincar_ac_tilecoding.train_cont(α_θ, α_w, λ_θ, λ_w; kwargs...).reward_history |> mean
  ╠═╡ =#

# ╔═╡ be700714-8168-4640-980c-64cd70107fb6
#=╠═╡
begin
	const tilecoding_ep_ac_study = setup_parameter_study(tilecoding_ep_ac_trial, (:α_θ, :α_w, :λ_θ, :λ_w), (num_steps = 100_000, num_tiles = 5, num_tilings = 5))


	if isfile("tilecoding_ep_ac_study.bin")
		let 
			d = deserialize("tilecoding_ep_ac_study.bin")
			for k in keys(d)
				tilecoding_ep_ac_study.results[k] = d[k]
			end
		end
	end

	const tilecoding_cont_ac_study = setup_parameter_study(tilecoding_cont_ac_trial, (:α_θ, :α_w, :λ_θ, :λ_w), (num_steps = 100_000, α_r̄ = 0.01f0, num_tiles = 5, num_tilings = 5))

	if isfile("tilecoding_cont_ac_study.bin")
		let 
			d = deserialize("tilecoding_cont_ac_study.bin")
			for k in keys(d)
				tilecoding_cont_ac_study.results[k] = d[k]
			end
		end
	end
end
  ╠═╡ =#

# ╔═╡ 15dd5412-2fc6-4243-8098-4ab74b8b9838
function run_ac_study(study, α_θ_list, α_w_list, λ_θ_list, λ_w_list; kwargs...)
	for α_w in α_w_list for λ_w in λ_w_list for α_θ in α_θ_list for λ_θ in λ_θ_list
		study.update_results!(α_θ, α_w, λ_θ, λ_w; kwargs...)
	end end end end
	
	return study.results
end

# ╔═╡ 2cd6da7f-1a12-40c6-9d91-2646673f559a
md"""
#### Simple Linear Results
"""

# ╔═╡ 295b92f3-6bbc-4ff3-b39a-589c05c3a07f
md"""
##### Episodic Training
"""

# ╔═╡ 6fea8dfb-9c00-4864-8a17-96d2bbb5bb59
#=╠═╡
begin 
	run_ac_study(simple_ep_ac_study, 2f0 .^ (-6:0), 2f0 .^ (-6:-1), [0f0, 0.1f0, 0.2f0, 0.3f0, 0.5f0, 0.9f0, 0.95f0, 0.99f0], [0f0, 0.5f0, 0.8f0, 0.9f0, 0.95f0, 0.99f0])
	DataFrame((;a[1]..., value = -a[2]) for a in simple_ep_ac_study.results) |> df -> filter(a -> !isnan(a.value), df) |> df -> sort(df, :value)
end
  ╠═╡ =#

# ╔═╡ 6b1daf65-fc46-42aa-9c99-91c1fec3bdda
md"""
##### Continuing Training
"""

# ╔═╡ da88bca7-644b-401a-8fb6-d5b96d609755
#=╠═╡
begin 
	run_ac_study(simple_cont_ac_study, 2f0 .^ (-7:-1), 2f0 .^ (-7:-1), [0.9f0, 0.99f0], [0f0, 0.1f0, 0.2f0, 0.3f0]; α_r̄ = 0.005f0)
	DataFrame((;a[1]..., value = inv(a[2])) for a in simple_cont_ac_study.results) |> df -> filter(a -> !isnan(a.value), df) |> df -> sort(df, :value)
end
  ╠═╡ =#

# ╔═╡ 452efa64-3595-4388-aa9b-98ce5a0fc404
const simple_ac_best = mountaincar_ac_simple.train_ep(0.25f0, 0.25f0, 0.2f0, 0.9f0; num_steps = 1_000_000)

# ╔═╡ 77314512-e87a-4d39-a2c4-2bb6027aa658
md"""
#### Tilecoding Linear Results
"""

# ╔═╡ 8fcd2433-619c-4202-a71c-826007f50749
md"""
##### Episodic Training
"""

# ╔═╡ 7723dc4f-43a6-4ece-80d5-88107f2fbf46
#=╠═╡
begin 
	run_ac_study(tilecoding_ep_ac_study, 2f0 .^ (-11:-6), 2f0 .^ (-11:-6), [0f0, 0.1f0, 0.2f0, 0.9f0, 0.99f0], [0.8f0, 0.9f0, 0.99f0]; num_tilings = 8, num_tiles = 8, num_steps = 1_000_000)
	DataFrame((;a[1]..., value = -a[2]) for a in tilecoding_ep_ac_study.results) |> df -> filter(a -> !isnan(a.value), df) |> df -> sort(df, :value)
end
  ╠═╡ =#

# ╔═╡ 6d251521-e502-4f96-bc65-a172cea7f224
md"""
##### Continuing Training
"""

# ╔═╡ 3c9ed6e9-b15e-4520-97ea-bdaa724a6e98
#=╠═╡
begin 
	run_ac_study(tilecoding_cont_ac_study, 2f0 .^ (-4:-2), 2f0 .^ (-7:-5), [0.1f0, 0.2f0], [0.95f0, 0.99f0]; num_tilings = 10, num_tiles = 10, α_r̄ = 0.01f0)
	DataFrame((;a[1]..., value = inv(a[2])) for a in tilecoding_cont_ac_study.results) |> df -> filter(a -> !isnan(a.value), df) |> df -> sort(df, :value)
end
  ╠═╡ =#

# ╔═╡ 6a547a22-1bf4-4b42-9d62-f1e14a35da47
const tile_ac_best = mountaincar_ac_tilecoding.train_ep(0.004f0, 0.008f0, 0.9f0, 0.99f0; num_steps = 1_000_000, num_tiles = 16, num_tilings = 8)

# ╔═╡ 715ab50e-b136-41ca-b7d6-e169ef457a00
md"""
#### Tilecoding Solution Study
"""

# ╔═╡ 75428084-5ff2-4996-8fce-81b09e83f287
function train_tile_grid(α_θ, α_w, λ_θ, λ_w; num_steps = 1_000_000, tile_min = 1, tile_max = 4, tilings_min = 0, tilings_max = 4, kwargs...)
	f(num_tiles, num_tilings) = mountaincar_ac_tilecoding.train_ep(α_θ * min(1f0, 8f0 / num_tilings), α_w * min(1f0, 8f0 / num_tilings), λ_θ, λ_w; num_steps = num_steps, num_tiles = num_tiles, num_tilings = num_tilings, kwargs...)
	
	tiles = 2 .^ (tile_min:tile_max)
	tilings = 2 .^ (tilings_min:tilings_max)
	args = [(n1, n2) for n1 in tiles for n2 in tilings ]
	grid = args |> Map(t -> t => f(t...)) |> tcollect |> Dict
	(grid = grid, tiles = tiles, tilings = tilings, args = args)
end

# ╔═╡ d66f814a-a4fd-41c3-8aec-f99969355e98
const tile_grid = train_tile_grid(0.002f0, 0.006f0, 0.9f0, 0.99f0; tilings_max = 5, tile_max = 5, num_steps = 1_000_000)

# ╔═╡ 32a3159a-0f5c-49e7-af9c-1d24addbcee0
md"""
##### Tilecoding Policy Grid
"""

# ╔═╡ b5207681-0976-4ca3-808c-442010dd67aa
#=╠═╡
md"""
Show policy distribution for action: $(@bind tile_grid_display_action Select(1:3, default = 3))
"""
  ╠═╡ =#

# ╔═╡ ffda6edf-42e1-4748-bae1-abcba9d85be3
md"""
##### Tilecoding Value Grid
"""

# ╔═╡ 998c9e14-8e8e-4a9c-b03a-9b5bfd38055f
md"""
#### Non-linear Results
"""

# ╔═╡ 8432affe-4371-4528-bb79-87d6b6374871
begin
	run_mountaincar_ac_fcann(mdp::StateMDP, γ::T, α_θ::T, α_w::T, λ_θ::T, λ_w::T, feature_vector, update_feature_vector!::Function; num_steps = 50_000, layer_size::Integer = 4, num_layers::Integer = 2, kwargs...) where T<:Real = actor_critic_with_eligibility_traces_fcann(mdp, γ, λ_θ, λ_w, typemax(Int64), num_steps, feature_vector, update_feature_vector!, fill(layer_size, num_layers); α_θ = α_θ, α_w = α_w, use_μP = true, kwargs...)
	
	run_mountaincar_ac_fcann(mdp::StateMDP, α_θ::T, α_w::T, λ_θ::T, λ_w::T, feature_vector, update_feature_vector!::Function; num_steps = 50_000, layer_size::Integer = 4, num_layers::Integer = 2, kwargs...) where T<:Real = actor_critic_with_eligibility_traces_fcann(mdp, λ_θ, λ_w, num_steps, feature_vector, update_feature_vector!, fill(layer_size, num_layers); α_θ = α_θ, α_w = α_w, kwargs...)
end

# ╔═╡ 7d00197a-e50e-4c9c-819e-f909287bc6bc
function setup_mountaincar_ac_fcann()
	x = mountaincar_simple_feature_setup.feature_vector
	f! = mountaincar_simple_feature_setup.update_feature_vector!
	train1(α_θ, α_w, λ_θ, λ_w; kwargs...) = run_mountaincar_ac_fcann(mountaincar_mdps[1], 1f0, α_θ, α_w, λ_θ, λ_w, copy(x), f!; kwargs...)
	train2(α_θ, α_w, λ_θ, λ_w; kwargs...) = run_mountaincar_ac_fcann(mountaincar_mdps[3], α_θ, α_w, λ_θ, λ_w, copy(x), f!; kwargs...)
	return (train_ep = train1, train_cont = train2)
end

# ╔═╡ 28e735b1-0590-4f4a-b066-deacab0f3dca
const mountaincar_ac_fcann = setup_mountaincar_ac_fcann()

# ╔═╡ 6020c4d7-572f-41a5-bb14-3ae248ab8219
#=╠═╡
fcann_ep_ac_trial(α_θ, α_w, λ_θ, λ_w; kwargs...) = mountaincar_ac_fcann.train_ep(α_θ, α_w, λ_θ, λ_w; kwargs...).episode_rewards |> mean
  ╠═╡ =#

# ╔═╡ 6deb3dad-050b-4d5f-b8d5-df29f41e62b0
#=╠═╡
fcann_cont_ac_trial(α_θ, α_w, λ_θ, λ_w; kwargs...) = mountaincar_ac_fcann.train_cont(α_θ, α_w, λ_θ, λ_w; kwargs...).reward_history |> mean
  ╠═╡ =#

# ╔═╡ 7ee82745-a862-4892-a06e-63420a2d7c03
#=╠═╡
begin
	fcann_ep_ac_study = setup_parameter_study(fcann_ep_ac_trial, (:α_θ, :α_w, :λ_θ, :λ_w), (num_steps = 100_000, num_layers = 2, layer_size = 4, reslayers = 1))
	if isfile("fcann_ep_ac_study.bin")
		let 
			d = deserialize("fcann_ep_ac_study.bin")
			for k in keys(d)
				fcann_ep_ac_study.results[k] = d[k]
			end
		end
	end
	fcann_cont_ac_study = setup_parameter_study(fcann_cont_ac_trial, (:α_θ, :α_w, :λ_θ, :λ_w), (num_steps = 100_000, α_r̄ = 0.01f0, num_layers = 2, layer_size = 4, reslayers = 1))
	if isfile("fcann_cont_ac_study.bin")
		let 
			d = deserialize("fcann_cont_ac_study.bin")
			for k in keys(d)
				fcann_cont_ac_study.results[k] = d[k]
			end
		end
	end
end
  ╠═╡ =#

# ╔═╡ b84a0ca4-a931-4955-a095-b5018db6f40c
#=╠═╡
if save_studies > 0
	serialize("simple_cont_study.bin", simple_cont_study.results)
	serialize("simple_ep_study.bin", simple_ep_study.results)
	serialize("tilecoding_cont_study.bin", tilecoding_cont_study.results)
	serialize("tilecoding_ep_study.bin", tilecoding_ep_study.results)
	serialize("fcann_cont_study.bin", fcann_cont_study.results)
	serialize("fcann_ep_study.bin", fcann_ep_study.results)

	serialize("simple_cont_ac_study.bin", simple_cont_ac_study.results)
	serialize("simple_ep_ac_study.bin", simple_ep_ac_study.results)
	serialize("tilecoding_cont_ac_study.bin", tilecoding_cont_ac_study.results)
	serialize("tilecoding_ep_ac_study.bin", tilecoding_ep_ac_study.results)
	serialize("fcann_cont_ac_study.bin", fcann_cont_ac_study.results)
	serialize("fcann_ep_ac_study.bin", fcann_ep_ac_study.results)
end
  ╠═╡ =#

# ╔═╡ dfa43b77-7cd7-4ae9-80c0-b473a08c7ed4
#=╠═╡
begin 
	run_ac_study(fcann_ep_ac_study, 2f0 .^ (-7:-5), 2f0 .^ (-7:-5), [0.1f0], [0.1f0]; num_steps = 10_000_000, layer_size = 8, num_layers = 4)
	DataFrame((;a[1]..., value = -a[2]) for a in fcann_ep_ac_study.results) |> df -> filter(a -> !isnan(a.value), df) |> df -> sort(df, :value)
end
  ╠═╡ =#

# ╔═╡ 10a482b1-fb19-4d5d-95ea-55b4900887b5
#=╠═╡
begin 
	run_ac_study(fcann_cont_ac_study, 2f0 .^ (-7:-5), 2f0 .^ (-7:-5), [0.8f0, 0.9f0, 0.99f0], [0.8f0, 0.9f0, 0.99f0]; num_steps = 10_000_000, layer_size = 8, num_layers = 4, α_r̄ = 0.005f0)
	DataFrame((;a[1]..., value = inv(a[2])) for a in fcann_cont_ac_study.results) |> df -> filter(a -> !isnan(a.value), df) |> df -> sort(df, :value)
end
  ╠═╡ =#

# ╔═╡ b032b2b1-5e07-44c4-9bfb-3fb84528c123
md"""
#### Non-linear Example
"""

# ╔═╡ 4e03ab81-a0bc-42a1-8753-fc6f866019d6
const fcann_ac_test = mountaincar_ac_fcann.train_cont(0.008f0, 0.01f0, 0.99f0, 0.9f0; num_steps = 10_000_000, layer_size = 32, num_layers = 8, α_r̄ = 0.005f0, reslayers=1)

# ╔═╡ af9b4662-1553-4a9e-b916-375cdbe4176a
#visualization of policy during hte training process to see how it evolves.

# ╔═╡ 28656452-ba55-4d46-be56-1c11c1928c23
md"""
#### Non-linear Performance Profiling
"""

# ╔═╡ 4d8416e0-67d4-436b-98bd-58d917aa84b3
#=╠═╡
@plutoprofview mountaincar_ac_fcann.train_cont(0.01f0, 0.015f0, 0.99f0, 0.9f0; num_steps = 10_000, layer_size = 64, num_layers = 2, α_r̄ = 0.005f0, reslayers = 1)
  ╠═╡ =#

# ╔═╡ df47a841-db47-4002-bc37-75778e7211d7
#=╠═╡
@plutoprofview mountaincar_ac_fcann.train_cont(0.01f0, 0.015f0, 0.99f0, 0.9f0; num_steps = 1_000, layer_size = 512, num_layers = 4, α_r̄ = 0.005f0, reslayers = 1)
  ╠═╡ =#

# ╔═╡ 5738326e-6817-424c-96ba-c9dffe02e590
#=╠═╡
@plutoprofview mountaincar_ac_fcann.train_cont(0.01f0, 0.015f0, 0.99f0, 0.9f0; num_steps = 1_000, layer_size = 512, num_layers = 4, α_r̄ = 0.005f0, reslayers = 1, use_gpu = true)
  ╠═╡ =#

# ╔═╡ 2a35f221-4c6a-4b79-b300-e87e8fc770cd
#=╠═╡
@plutoprofview mountaincar_ac_fcann.train_cont(0.01f0, 0.015f0, 0.99f0, 0.9f0; num_steps = 1_000, layer_size = 1024, num_layers = 2, α_r̄ = 0.005f0, reslayers = 1)
  ╠═╡ =#

# ╔═╡ 63ed4500-a199-4a31-ace1-cec54f152380
#=╠═╡
@plutoprofview mountaincar_ac_fcann.train_cont(0.01f0, 0.015f0, 0.99f0, 0.9f0; num_steps = 1_000, layer_size = 1024, num_layers = 2, α_r̄ = 0.005f0, reslayers = 1, use_gpu = true)
  ╠═╡ =#

# ╔═╡ 33848c9b-fe3c-4767-abfc-c40a7a68cb56
md"""
#### Non-linear Solution Study
"""

# ╔═╡ 84445f5a-cd73-4098-b3a4-861abf42061d
function train_fcann_grid(α_θ, α_w, λ_θ, λ_w; num_steps = 200_000, n_min = 0, n_max = 4, layers_min = 0, layers_max = 4, kwargs...)
	f(layer_size, num_layers) = mountaincar_ac_fcann.train_cont(α_θ, α_w, λ_θ, λ_w; num_steps = num_steps, layer_size = layer_size, num_layers = num_layers, kwargs...)
	
	n = 2 .^ (n_min:n_max)
	layers = 2 .^ (layers_min:layers_max)
	args = [(n1, n2) for n1 in n for n2 in layers]
	grid = args |> Map(t -> t => f(t...)) |> tcollect |> Dict
	(grid = grid, layer_size = n, num_layers = layers, args = args)
end

# ╔═╡ ec352be3-b742-423d-8454-f8a7c44b3543
const fcann_grid = train_fcann_grid(0.01f0, 0.015f0, 0.99f0, 0.9f0; n_min = 2, n_max = 6, layers_min = 1, layers_max = 3, num_steps = 10_000_000, α_r̄ = 0.005f0, reslayers = 1)

# ╔═╡ 4f16565e-09bb-11f0-3729-7ffc5462cdc8
md"""
# Dependencies
"""

# ╔═╡ 43999574-da93-475c-9a67-e5024fb08202
# ╠═╡ skip_as_script = true
#=╠═╡
import HypertextLiteral.@htl
  ╠═╡ =#

# ╔═╡ 1b84943c-c8f5-4ad4-b95a-66dc818fa609
# ╠═╡ skip_as_script = true
#=╠═╡
html"""
	<style>
		main {
			margin: 0 auto;
			max-width: min(1600px, 90%);
			padding-left: max(10px, 5%);
			padding-right: max(10px, 5%);
			font-size: max(10px, min(24px, 2vw));
		}
	</style>
	"""
  ╠═╡ =#

# ╔═╡ afb3f1df-8aa7-4e57-bf03-9d901c9c2946
md"""
## Visualization Tools
"""

# ╔═╡ 6130d10b-b23c-4e15-97b3-ec7a1134e732
#=╠═╡
function plot_mountaincar_values(v̂_mountain_car, π; n1 = 100, n2 = 100)
	xvals = LinRange(-1.2f0, 0.5f0, n1)
	vvals = LinRange(-0.07f0, 0.07f0, n2)
	values = zeros(Float32, n1, n2)
	actions = zeros(Float32, n1, n2)
	for (i, x) in enumerate(xvals)
		for (j, v) in enumerate(vvals)
			v̂ = v̂_mountain_car((x, v))
			values[j, i] = v̂
			actions[j, i] = π((x, v))
		end
	end
	p1 = plot(heatmap(x = xvals, y = vvals, z = values), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Learned Value Function", height = 400))
	p2 = plot(heatmap(x = xvals, y = vvals, z = actions, colorscale = "rb", showscale = false), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Policy (blue = accelerate left, <br>red = accelerate right, gray = no acceleration)", height = 400))
	@htl("""
	<div style = "display:flex;">
	$p1 
	$p2
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ b76a1c2b-f5df-41c1-8dae-4be15ae274a5
function mountaincar_policy_action_dist(policy_function::Function, i_a::Integer; n1 = 100, n2 = 100)
	xvals = LinRange(-1.2f0, 0.5f0, n1)
	vvals = LinRange(-0.07f0, 0.07f0, n2)
	action_dist = zeros(Float32, n1, n2)
	for (i, x) in enumerate(xvals)
		for (j, v) in enumerate(vvals)
			π = policy_function((x, v))
			action_dist[j, i] = π[i_a]
		end
	end
	return action_dist
end

# ╔═╡ 67da98d7-c525-47bf-bee3-61efaa3231b4
#=╠═╡
function plot_tile_value_grid(grid_output::NamedTuple, i_a::Integer; kwargs...)
	(grid, tiles, tilings, args) = grid_output
	plots = [begin
		π(s) = grid[k].value_function(s).action_values |> make_greedy_policy! 
		grid_matrix = mountaincar_policy_action_dist(π, i_a; kwargs...)
		yaxis_text = if k[2] == 1
			"$(k[1]) tiles"
		else
			""
		end

		title_text = if k[1] == 2
			"$(k[2]) tilings"
		else
			""
		end

		step_avg = grid[k].episode_rewards[max(1, end-1_000):end] |> v -> round(-mean(v); sigdigits = 4)

		xaxis_text = "$step_avg steps"
		p = plot(heatmap(z = grid_matrix, showscale = false, colorscale = "rb"), Layout(title = title_text, yaxis_title = yaxis_text, paper_bgcolor = "rgb(30, 30, 30", font_color = "white", xaxis_title = xaxis_text, xaxis_tickvals = [], yaxis_tickvals = [], margin_l = 0, margin_r = 0, margin_b = 0, margin_t = 30))
		
		@htl("""
			 <div style = "width: $(inv(length(tilings))*100)%; aspect-ratio: 1 / 1; background-color: rgbt(0, 0, 0, 0);">
			 $p
			 </div>
			""")
	end
	for k in args]

	@htl("""
		 <div style = "display: flex; flex-wrap: wrap; ">
		 $plots
		 </div>
		 """)
end
  ╠═╡ =#

# ╔═╡ 23fd97ba-5b99-4d53-b564-e6cc1f0140e3
#=╠═╡
function plot_fcann_value_grid(grid_output::NamedTuple, i_a::Integer; kwargs...)
	(grid, layer_size, num_layers, args) = grid_output
	plots = [begin
		π(s) = grid[k].value_function(s).action_values |> make_greedy_policy! 
		grid_matrix = mountaincar_policy_action_dist(π, i_a; kwargs...)
		yaxis_text = if k[2] == first(num_layers)
			"$(k[1]) layer size"
		else
			""
		end

		title_text = if k[1] == first(layer_size)
			"$(k[2]) num layers"
		else
			""
		end

		step_avg = grid[k].reward_history[max(1, end-100_000):end] |> v -> round(inv(mean(v)); sigdigits = 4)

		xaxis_text = "$step_avg steps"
		p = plot(heatmap(z = grid_matrix, showscale = false, colorscale = "rb"), Layout(title = title_text, yaxis_title = yaxis_text, paper_bgcolor = "rgb(30, 30, 30", font_color = "white", xaxis_title = xaxis_text, xaxis_tickvals = [], yaxis_tickvals = [], margin_l = 0, margin_r = 0, margin_b = 0, margin_t = 30))
		
		@htl("""
			 <div style = "width: $(inv(length(num_layers))*100)%; aspect-ratio: 1 / 1; background-color: rgbt(0, 0, 0, 0);">
			 $p
			 </div>
			""")
	end
	for k in args]

	@htl("""
		 <div style = "display: flex; flex-wrap: wrap; ">
		 $plots
		 </div>
		 """)
end
  ╠═╡ =#

# ╔═╡ 1c12179c-13c5-4f64-a1fe-cbe2d9d219bb
#=╠═╡
function plot_tile_grid(grid_output::NamedTuple, i_a::Integer; kwargs...)
	(grid, tiles, tilings, args) = grid_output
	plots = [begin
		grid_matrix = mountaincar_policy_action_dist(grid[k].policy_function, i_a; kwargs...)
		yaxis_text = if k[2] == 1
			"$(k[1]) tiles"
		else
			""
		end

		title_text = if k[1] == 2
			"$(k[2]) tilings"
		else
			""
		end

		step_avg = grid[k].episode_rewards[max(1, end-10_000):end] |> v -> round(-mean(v); sigdigits = 4)

		xaxis_text = "$step_avg steps"
		p = plot(heatmap(z = grid_matrix, showscale = false, colorscale = "rb"), Layout(title = title_text, yaxis_title = yaxis_text, paper_bgcolor = "rgb(30, 30, 30", font_color = "white", xaxis_title = xaxis_text, xaxis_tickvals = [], yaxis_tickvals = [], margin_l = 0, margin_r = 0, margin_b = 0, margin_t = 30))
		
		@htl("""
			 <div style = "width: $(inv(length(tilings))*100)%; aspect-ratio: 1 / 1; background-color: rgbt(0, 0, 0, 0);">
			 $p
			 </div>
			""")
	end
	for k in args]

	@htl("""
		 <div style = "display: flex; flex-wrap: wrap; ">
		 $plots
		 </div>
		 """)
end
  ╠═╡ =#

# ╔═╡ 90c7aeca-aafc-4b48-b6e2-82ccd249fd24
#=╠═╡
function plot_fcann_grid(grid_output::NamedTuple, i_a::Integer; kwargs...)
	(grid, layer_size, num_layers, args) = grid_output
	plots = [begin
		grid_matrix = mountaincar_policy_action_dist(grid[k].policy_function, i_a; kwargs...)
		yaxis_text = if k[2] == first(num_layers)
			"$(k[1]) Units Per Layer"
		else
			""
		end

		title_text = if k[1] == first(layer_size)
			"$(k[2]) Layers"
		else
			""
		end

		step_avg = grid[k].reward_history[max(1, end-100_000):end] |> v -> round(inv(mean(v)); sigdigits = 4)

		xaxis_text = "$step_avg steps"
		p = plot(heatmap(z = grid_matrix, showscale = false, colorscale = "rb"), Layout(title = title_text, yaxis_title = yaxis_text, paper_bgcolor = "rgb(30, 30, 30", font_color = "white", xaxis_title = xaxis_text, xaxis_tickvals = [], yaxis_tickvals = [], margin_l = 0, margin_r = 0, margin_b = 0, margin_t = 30))
		
		@htl("""
			 <div style = "width: $(inv(length(num_layers))*100)%; aspect-ratio: 1 / 1; background-color: rgbt(0, 0, 0, 0);">
			 $p
			 </div>
			""")
	end
	for k in args]

	@htl("""
		 <div style = "display: flex; flex-wrap: wrap; ">
		 $plots
		 </div>
		 """)
end
  ╠═╡ =#

# ╔═╡ 66394600-ab20-4cc8-9319-1bcbfc5191dc
function mountaincar_value_grid(value_function::Function; n1 = 100, n2 = 100)
	xvals = LinRange(-1.2f0, 0.5f0, n1)
	vvals = LinRange(-0.07f0, 0.07f0, n2)
	value_grid = zeros(Float32, n1, n2)
	for (i, x) in enumerate(xvals)
		for (j, v) in enumerate(vvals)
			v = value_function((x, v))
			value_grid[j, i] = v
		end
	end
	return value_grid
end

# ╔═╡ a8c60935-3497-4019-9fa4-8323e1642f02
#=╠═╡
function plot_tile_value_grid(grid_output::NamedTuple; kwargs...)
	(grid, tiles, tilings, args) = grid_output
	plots = [begin
		v̂(s) = grid[k].value_function(s).maximizing_value
		grid_matrix = mountaincar_value_grid(v̂; kwargs...)
		yaxis_text = if k[2] == 1
			"$(k[1]) tiles"
		else
			""
		end

		title_text = if k[1] == 2
			"$(k[2]) tilings"
		else
			""
		end

		step_avg = grid[k].episode_rewards[max(1, end-1_000):end] |> v -> round(-mean(v); sigdigits = 4)

		xaxis_text = "$step_avg steps"
		p = plot(heatmap(z = grid_matrix, showscale = false, colorscale = "rb"), Layout(title = title_text, yaxis_title = yaxis_text, paper_bgcolor = "rgb(30, 30, 30", font_color = "white", xaxis_title = xaxis_text, xaxis_tickvals = [], yaxis_tickvals = [], margin_l = 0, margin_r = 0, margin_b = 0, margin_t = 30))
		
		@htl("""
			 <div style = "width: $(inv(length(tilings))*100)%; aspect-ratio: 1 / 1; background-color: rgbt(0, 0, 0, 0);">
			 $p
			 </div>
			""")
	end
	for k in args]

	@htl("""
		 <div style = "display: flex; flex-wrap: wrap; ">
		 $plots
		 </div>
		 """)
end
  ╠═╡ =#

# ╔═╡ 7b988e80-02a3-4584-867b-38315cd30a97
#=╠═╡
plot_tile_value_grid(tile_value_grid)
  ╠═╡ =#

# ╔═╡ 00fefa0f-c6c3-4887-a0aa-7cfdff438812
#=╠═╡
plot_tile_value_grid(tile_value_grid, 3)
  ╠═╡ =#

# ╔═╡ c8368d88-8284-4054-83c1-2efab8d86678
#=╠═╡
function plot_fcann_value_grid(grid_output::NamedTuple; kwargs...)
	(grid, layer_size, num_layers, args) = grid_output
	plots = [begin
		v̂(s) = grid[k].value_function(s).maximizing_value
		grid_matrix = mountaincar_value_grid(v̂; kwargs...)
		yaxis_text = if k[2] == first(num_layers)
			"$(k[1]) layer size"
		else
			""
		end

		title_text = if k[1] == first(layer_size)
			"$(k[2]) num layers"
		else
			""
		end

		step_avg = grid[k].reward_history[max(1, end-100_000):end] |> v -> round(inv(mean(v)); sigdigits = 4)

		xaxis_text = "$step_avg steps"
		p = plot(heatmap(z = grid_matrix, showscale = false, colorscale = "rb"), Layout(title = title_text, yaxis_title = yaxis_text, paper_bgcolor = "rgb(30, 30, 30", font_color = "white", xaxis_title = xaxis_text, xaxis_tickvals = [], yaxis_tickvals = [], margin_l = 0, margin_r = 0, margin_b = 0, margin_t = 30))
		
		@htl("""
			 <div style = "width: $(inv(length(num_layers))*100)%; aspect-ratio: 1 / 1; background-color: rgbt(0, 0, 0, 0);">
			 $p
			 </div>
			""")
	end
	for k in args]

	@htl("""
		 <div style = "display: flex; flex-wrap: wrap; ">
		 $plots
		 </div>
		 """)
end
  ╠═╡ =#

# ╔═╡ 8d20c827-11e8-44ae-9ed3-e4f2899be007
#=╠═╡
@htl("""
<h4>Sarsa Solution Study</h4>
<hr>
<div style = "display: flex;">
	 <div style = "width: 50%;">
	 <h5>Value Function</h5>
	$(plot_fcann_value_grid(fcann_sarsa_value_grid))
	 </div>
	 <div style = "width: 50%;">
	 <h5>Policy Function Distribution for Action 3</h5>
	$(plot_fcann_value_grid(fcann_sarsa_value_grid, 3; n1 = 300, n2 = 300))
	 </div>
</div>
""")
  ╠═╡ =#

# ╔═╡ e807f411-38c3-4f35-99a3-68b09ae1b3a7
#=╠═╡
@htl("""
<h4>DP Solution Study</h4>
<hr>
<div style = "display: flex;">
	 <div style = "width: 50%;">
	 <h5>Value Function</h5>
	 <hr>
	$(plot_fcann_value_grid(fcann_dp_value_grid))
	 </div>
	 <div style = "width: 50%;">
	 <h5>Policy Function Distribution for Action 3</h5>
	 <hr>
	$(plot_fcann_value_grid(fcann_dp_value_grid, 3; n1 = 300, n2 = 300))
	 </div>
</div>
""")
  ╠═╡ =#

# ╔═╡ 47b8c456-c809-4fdb-9f05-9c461bc8e167
#=╠═╡
function plot_tile_grid(grid_output::NamedTuple; kwargs...)
	(grid, tiles, tilings, args) = grid_output
	plots = [begin
		grid_matrix = mountaincar_value_grid(grid[k].value_function; kwargs...)
		yaxis_text = if k[2] == 1
			"$(k[1]) tiles"
		else
			""
		end

		title_text = if k[1] == 2
			"$(k[2]) tilings"
		else
			""
		end

		step_avg = grid[k].episode_rewards[max(1, end-10_000):end] |> v -> round(-mean(v); sigdigits = 4)

		xaxis_text = "$step_avg steps"
		p = plot(heatmap(z = grid_matrix, showscale = false, colorscale = "rb"), Layout(title = title_text, yaxis_title = yaxis_text, paper_bgcolor = "rgb(30, 30, 30", font_color = "white", xaxis_title = xaxis_text, xaxis_tickvals = [], yaxis_tickvals = [], margin_l = 0, margin_r = 0, margin_b = 0, margin_t = 30))
		
		@htl("""
			 <div style = "width: $(inv(length(tilings))*100)%; aspect-ratio: 1 / 1; background-color: rgbt(0, 0, 0, 0);">
			 $p
			 </div>
			""")
	end
	for k in args]

	@htl("""
		 <div style = "display: flex; flex-wrap: wrap; ">
		 $plots
		 </div>
		 """)
end
  ╠═╡ =#

# ╔═╡ ca378efe-514d-4ba0-9414-7614ba84eaa5
#=╠═╡
plot_tile_grid(tile_grid, tile_grid_display_action)
  ╠═╡ =#

# ╔═╡ c8d689cd-a9c7-4cfa-b0ea-1a9ffd71010e
#=╠═╡
plot_tile_grid(tile_grid)
  ╠═╡ =#

# ╔═╡ 62b5acb7-3ec7-442d-afe5-5d76c62b5582
#=╠═╡
function plot_fcann_grid(grid_output::NamedTuple; kwargs...)
	(grid, layer_size, num_layers, args) = grid_output
	plots = [begin
		grid_matrix = mountaincar_value_grid(grid[k].value_function; kwargs...)
		yaxis_text = if k[2] == first(num_layers)
			"$(k[1]) Units Per Layer"
		else
			""
		end

		title_text = if k[1] == first(layer_size)
			"$(k[2]) Layers"
		else
			""
		end

		step_avg = grid[k].reward_history[max(1, end-100_000):end] |> v -> round(inv(mean(v)); sigdigits = 4)

		xaxis_text = "$step_avg steps"
		p = plot(heatmap(z = grid_matrix, showscale = false, colorscale = "rb"), Layout(title = title_text, yaxis_title = yaxis_text, paper_bgcolor = "rgb(30, 30, 30", font_color = "white", xaxis_title = xaxis_text, xaxis_tickvals = [], yaxis_tickvals = [], margin_l = 0, margin_r = 0, margin_b = 0, margin_t = 30))
		
		@htl("""
			 <div style = "width: $(inv(length(num_layers))*100)%; aspect-ratio: 1 / 1; background-color: rgbt(0, 0, 0, 0);">
			 $p
			 </div>
			""")
	end
	for k in args]

	@htl("""
		 <div style = "display: flex; flex-wrap: wrap; ">
		 $plots
		 </div>
		 """)
end
  ╠═╡ =#

# ╔═╡ d9fdbf41-5be5-411b-9e72-8c0cdee88761
#=╠═╡
@htl("""
<h4>Actor Critic Solution Study</h4>
<hr>
<div style = "display: flex;">
	 <div style = "width: 50%;">
	 <h5>Policy Function Distribution for Action 3</h5>
	$(plot_fcann_grid(fcann_grid, 3; n1 = 300, n2 = 300))
	 </div>
	 <div style = "width: 50%;">
	 <h5>State Value Function</h5>
	$(plot_fcann_grid(fcann_grid; n1 = 300, n2 = 300))
	 </div>
</div>
""")
  ╠═╡ =#

# ╔═╡ 72f9972e-9f83-4bf7-b459-764c309552b4
#=╠═╡
function plot_mountaincar_policy(policy_function::Function; n1 = 100, n2 = 100)
	xvals = LinRange(-1.2f0, 0.5f0, n1)
	vvals = LinRange(-0.07f0, 0.07f0, n2)
	action_dists = [zeros(Float32, n1, n2) for i in 1:3]
	for (i, x) in enumerate(xvals)
		for (j, v) in enumerate(vvals)
			π = policy_function((x, v))
			for k in 1:3
				action_dists[k][j, i] = π[k]
			end
		end
	end
	p2 = [plot(heatmap(x = xvals, y = vvals, z = action_dists[k], colorscale = "rb"), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Policy Probability for Action $k", height = 400, width = 400)) for k in 1:3]
	@htl("""
	<div style = "display:flex; flex-wrap: wrap;">
	$p2
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ acf47599-4813-412d-8dfe-a6d4967f710c
#=╠═╡
function plot_mountaincar_policy_values(policy_and_value::Function; n1 = 100, n2 = 100)
	xvals = LinRange(-1.2f0, 0.5f0, n1)
	vvals = LinRange(-0.07f0, 0.07f0, n2)
	values = zeros(Float32, n1, n2)
	action_dists = [zeros(Float32, n1, n2) for i in 1:3]
	for (i, x) in enumerate(xvals)
		for (j, v) in enumerate(vvals)
			v̂, π = policy_and_value((x, v))
			values[j, i] = v̂
			for k in 1:3
				action_dists[k][j, i] = π[k]
			end
		end
	end
	p1 = plot(heatmap(x = xvals, y = vvals, z = values), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Learned Value Function", height = 300, width = 350))
	p2 = [plot(heatmap(x = xvals, y = vvals, z = action_dists[k], colorscale = "rb", showscale= (k == 3)), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Policy Probability for Action $k", height = 300, width = 350)) for k in 1:3]
	@htl("""
	<div style = "display:flex;">
	$(vcat(p1, p2))
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ 3ab40052-d656-4e14-a6fd-3c5b09664cf4
#=╠═╡
function plot_mountaincar_policy_values(policy_function::Function, value_function::Function; n1 = 100, n2 = 100)
	xvals = LinRange(-1.2f0, 0.5f0, n1)
	vvals = LinRange(-0.07f0, 0.07f0, n2)
	values = zeros(Float32, n1, n2)
	action_dists = [zeros(Float32, n1, n2) for i in 1:3]
	for (i, x) in enumerate(xvals)
		for (j, v) in enumerate(vvals)
			π = policy_function((x, v))
			v̂ = value_function((x, v))
			values[j, i] = v̂
			for k in 1:3
				action_dists[k][j, i] = π[k]
			end
		end
	end
	p1 = plot(heatmap(x = xvals, y = vvals, z = values), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Learned Value Function", height = 300, width = 350))
	p2 = [plot(heatmap(x = xvals, y = vvals, z = action_dists[k], colorscale = "rb", showscale= (k == 3)), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Policy Probability for Action $k", height = 300, width = 350)) for k in 1:3]
	@htl("""
	<div style = "display:flex;">
	$(vcat(p1, p2))
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ 880a7ec3-9d3a-420d-a721-7a5616deb01b
#=╠═╡
function show_mountaincar_trajectory(π::Function, max_steps::Integer; mdp = first(mountaincar_mdps))
	states, actions, rewards, sterm, nsteps = runepisode(mdp; π = π, max_steps = max_steps)
	positions = [s[1] for s in states]
	velocities = [s[2] for s in states]
	tr1 = scatter(x = positions, y = velocities, mode = "markers", showlegend = false)
	tr2 = scatter(y = positions, showlegend = false)
	tr3 = scatter(y = [mdp.actions[i] for i in actions], showlegend = false)
	p1 = plot(tr1, Layout(xaxis_title = "position", yaxis_title = "velocity", xaxis_range = [-1.2, 0.5], yaxis_range = [-0.07, 0.07], height = 400))
	p2 = plot(tr2, Layout(xaxis_title = "time", yaxis_title = "position", height = 400))
	p3 = plot(tr3, Layout(xaxis_title = "time", yaxis_title = "action", height = 400))
	@htl("""
	Total Reward: $(sum(rewards))
	<div style = "display: flex;">
	$([p1 p2 p3])
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ 3ab1b709-f17d-4773-b6ff-e43909d537f6
#=╠═╡
show_mountaincar_trajectory(make_random_policy(mountaincar_mdps[1]), 1_000)
  ╠═╡ =#

# ╔═╡ 88cdec08-ef35-4474-968d-3c30761ee7dd
#=╠═╡
show_mountaincar_trajectory(make_random_policy(mountaincar_mdps[3]), 1_000; mdp = mountaincar_mdps[3])
  ╠═╡ =#

# ╔═╡ 127b258b-6df0-4347-ab6d-1e9f12e130d3
#=╠═╡
function plot_mountaincar_action_values(q̂_mountain_car, n1, n2)
	xvals = LinRange(-1.2f0, 0.5f0, n1)
	vvals = LinRange(-0.07f0, 0.07f0, n2)
	values = zeros(Float32, n1, n2)
	actions = zeros(Float32, n1, n2)
	for (i, x) in enumerate(xvals)
		for (j, v) in enumerate(vvals)
			q̂ = q̂_mountain_car((x, v))
			values[j, i] = q̂.maximizing_value
			actions[j, i] = MountainCarTask.actions[q̂.maximizing_action]
		end
	end
	p1 = plot(heatmap(x = xvals, y = vvals, z = values), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Learned Value Function"))
	p2 = plot(heatmap(x = xvals, y = vvals, z = actions, colorscale = "rb", showscale = false), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Policy (blue = accelerate left, <br>red = accelerate right, gray = no acceleration)"))
	[p1 p2]
end
  ╠═╡ =#

# ╔═╡ 45cc5cf5-ccc7-4e2f-9027-d3335ef7f041
#=╠═╡
function smooth_error(error_history, n)
	l = length(error_history)
	n+1:l |> Map(i -> mean(view(error_history, i-n:i))) |> tcollect
end
  ╠═╡ =#

# ╔═╡ a14ebbce-3590-4759-81b3-253a13e73fc6
#=╠═╡
function display_mountaincar_ac_results(output::NamedTuple; nsmooth = 100, npoints = 1000, max_steps = 2000, n1 = 100, n2 = 100)
	policy_kwargs = output.form_policy_kwargs()
	π_sample(s) = output.policy_sample_action(s; policy_kwargs...)
	p1 = show_mountaincar_trajectory(π_sample, max_steps)

	value_kwargs = output.form_value_kwargs()
	v̂(s) = output.value_function(s; value_kwargs...)
	π_dist(s) = output.policy_function(s; policy_kwargs...)
	p2 = plot_mountaincar_policy_values(π_dist, v̂; n1 = n1, n2 = n2)

	try 
		rewards = output.episode_rewards
	catch
		rewards = output.reward_history
		nsmooth = 100*nsmooth
	end
	
	if isempty(rewards)
		p3 = nothing
	elseif length(rewards) ≤ npoints
		p3 = plot(rewards)
	else
		rewards = smooth_error(rewards, nsmooth)
		l = length(rewards)
		sample_inds = round.(Int64, LinRange(1, l, npoints))
		p3 = plot(view(rewards, sample_inds))
	end
 
	@htl("""
		 $p1
		
		 $p2
		
		 $p3
		 """)
end
  ╠═╡ =#

# ╔═╡ 25aa07bb-6c2b-4195-bbb2-01b031a1b91e
#=╠═╡
display_mountaincar_ac_results(simple_ac_best)
  ╠═╡ =#

# ╔═╡ ddd8a237-0d77-4ef2-ac8f-b9d76c2448a6
#=╠═╡
display_mountaincar_ac_results(tile_ac_best)
  ╠═╡ =#

# ╔═╡ 92288560-f4e9-4e6a-b182-cecc07b0b457
#=╠═╡
display_mountaincar_ac_results(fcann_ac_test; max_steps = 300, n1 = 300, n2 = 300)
  ╠═╡ =#

# ╔═╡ 283fef98-fc27-42c6-b8c7-579f29dd2881
#=╠═╡
function display_mountaincar_results(output::NamedTuple; nsmooth = 100, npoints = 1000, ϵ = 0.1f0)
	p1 = show_mountaincar_trajectory(s -> rand() < ϵ ? rand(1:3) : output.value_function(s).maximizing_action, 2000)

	kwargs = output.form_kwargs()
	v̂(s) = output.value_function(s; kwargs...)
	p2 = plot_mountaincar_action_values(v̂, 200, 200)

	try 
		rewards = output.episode_rewards
	catch
		rewards = output.reward_history
		nsmooth = 100*nsmooth
	end
	
	if isempty(rewards)
		p3 = nothing
	elseif length(rewards) ≤ npoints
		p3 = plot(rewards)
	else
		rewards = smooth_error(rewards, nsmooth)
		l = length(rewards)
		sample_inds = round.(Int64, LinRange(1, l, npoints))
		p3 = plot(rewards[sample_inds])
	end
 
	@htl("""
		 $p1
		 <div style = "display: flex">
		 $p2
		 </div>
		 $p3
		 """)
end
  ╠═╡ =#

# ╔═╡ a7a9d3ce-43eb-42ea-98eb-f10fc1c2c0f8
#=╠═╡
mountaincar_simple.train_ep(0.002f0, 0.8f0; num_steps = 100_000, ϵ = 0.01f0) |> display_mountaincar_results
  ╠═╡ =#

# ╔═╡ fc0d0fac-5b35-4626-b304-2c51f1f8a898
#=╠═╡
mountaincar_simple.train_ep(.5f0, 0.9f0; num_steps = 100_000, algo = dp_λ_linear, ϵ = 0.01f0) |> display_mountaincar_results
  ╠═╡ =#

# ╔═╡ fa23401b-ef86-418c-ad53-132bdb384b05
#=╠═╡
mountaincar_simple.train_cont(2f0, 0.9f0; num_steps = 100_000, use_dp = true) |> display_mountaincar_results
  ╠═╡ =#

# ╔═╡ 98f0d534-9d86-4eef-9afb-78914e02d4f8
#=╠═╡
display_mountaincar_results(tilecoding_ep_best)
  ╠═╡ =#

# ╔═╡ 10e2c439-f214-4470-8eb5-64d87d55289f
#=╠═╡
display_mountaincar_results(tilecoding_cont_best)
  ╠═╡ =#

# ╔═╡ 25b1dd0b-3826-40f9-a7a8-324653c1ec3f
#=╠═╡
display_mountaincar_results(fcann_value_best)
  ╠═╡ =#

# ╔═╡ 38efb2be-2f3b-4325-9b60-8fec2d87c087
#=╠═╡
begin
	plot_rewards(rewards::AbstractVector{T}, nsmooth::Integer, npoints::Integer) where T<:Real = plot(smooth_error(rewards, nsmooth)[round.(Int64, LinRange(1, length(rewards) - nsmooth, npoints))])

	function plot_rewards(rewards::AbstractVector{A}, nsmooth::Integer, npoints::Integer) where A <: Union{Missing, T} where T<:Real 	
		newrewards = [!ismissing(a) for a in rewards]
		plot_rewards(newrewards, nsmooth, npoints)
	end
end
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Transducers = "28d57a85-8fef-5791-bfe6-a80928e7c999"

[compat]
BenchmarkTools = "~1.6.0"
DataFrames = "~1.8.0"
Distributions = "~0.25.122"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
PlutoDevMacros = "~0.9.1"
PlutoPlotly = "~0.6.5"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.71"
ProgressLogging = "~0.1.5"
StaticArrays = "~1.9.15"
StatsBase = "~0.34.6"
Transducers = "~0.4.85"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.7"
manifest_format = "2.0"
project_hash = "ea61808d53f7b96f544c6650b0f34d7f86b26aca"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "26f41e1df02c330c4fa1e98d4aa2168fdafc9b1f"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.4"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "e38fbc49a620f5d0b660d7f543db1009fe0f8336"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "980f01d6d3283b3dbdfd7ed89405f96b7256ad57"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "2.0.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "c967271c27a95160e30432e011b58f42cd7501b5"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6c72198e6a101cccdd4c9731d3985e904ba26037"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.1"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3bc002af51045ca3b47d2e1787d6ce02e68b943a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.122"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "d60eb76f37d7e5a40cc2e7c36974d864b82dc802"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.1"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "173e4d8f14230a7523ae11b9a3fa9edb3e0efd78"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.14.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.FlameGraphs]]
deps = ["AbstractTrees", "Colors", "FileIO", "FixedPointNumbers", "IndirectArrays", "LeftChildRightSiblingTrees", "Profile"]
git-tree-sha1 = "d9eee53657f6a13ee51120337f98684c9c702264"
uuid = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"
version = "0.2.10"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "d8337622fe53c05d16f031df24daf0270e53bc64"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.5"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "b864cb409e8e445688bc478ef87c0afe4f6d1f8d"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f07c06228a1c670ae4c87d1276b92c7c597fdda0"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.35"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "28278bb0053da0fd73537be94afd1682cc5a0a83"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.21"

    [deps.PlotlyBase.extensions]
    DataFramesExt = "DataFrames"
    DistributionsExt = "Distributions"
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyBase.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlutoDevMacros]]
deps = ["JuliaInterpreter", "Logging", "MacroTools", "Pkg", "TOML"]
git-tree-sha1 = "1cb861c9295d79dc6e23170d4b33bce013f69643"
uuid = "a0499f29-c39b-4c5c-807c-88074221b949"
version = "0.9.1"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "8acd04abc9a636ef57004f4c2e6f3f6ed4611099"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.5"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoProfile]]
deps = ["AbstractTrees", "FlameGraphs", "Profile", "ProfileCanvas"]
git-tree-sha1 = "154819e606ac4205dd1c7f247d7bda0bf4f215c4"
uuid = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
version = "0.4.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "8329a3a4f75e178c11c1ce2342778bcbbbfa7e3c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.71"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "6b8e2f0bae3f678811678065c09571c1619da219"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.1.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProfileCanvas]]
deps = ["FlameGraphs", "JSON", "Pkg", "Profile", "REPL"]
git-tree-sha1 = "41fd9086187b8643feda56b996eef7a3cc7f4699"
uuid = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
version = "0.1.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "d95ed0324b0799843ac6f7a6a85e65fe4e5173f0"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.5"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2c962245732371acd51700dbb268af311bddd719"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.6"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "8e45cecc66f3b42633b8ce14d431e8e57a3e242e"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

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
version = "1.11.0"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "SplittablesBase", "Tables"]
git-tree-sha1 = "4aa1fdf6c1da74661f6f5d3edfd96648321dade9"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.85"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─e7c4aad1-562f-4027-b64b-39859ac8abbb
# ╟─00e79eb9-93aa-4afc-b665-aa410a1d3a9b
# ╟─d52f66cd-c4c2-4750-b182-42e08a9a27f4
# ╠═92b57e64-973c-4601-8789-7864b7fd07de
# ╠═d05722e6-f0d2-44fd-bd07-20b5ae1ffc11
# ╟─a6d0b563-d561-445c-a0b8-98b84c48fb36
# ╟─873fec07-6816-447a-8033-1aeb21d5bae8
# ╠═3ab1b709-f17d-4773-b6ff-e43909d537f6
# ╟─9d6c7dd3-e3e7-4571-bab1-b988488807c2
# ╟─88cdec08-ef35-4474-968d-3c30761ee7dd
# ╟─cb12f8b6-cec1-40fe-9b29-0acae1d3291b
# ╟─c8ebb9fe-bc0b-488c-bf68-761e57624a87
# ╟─8155a38b-913e-410a-9ac6-aa8d642a4c12
# ╠═be46a64e-7fc1-4d5e-96a9-8474fef04f9f
# ╟─4f10e409-bda0-4765-ae00-a20761f1f85d
# ╠═ed7ced42-cacf-434a-a572-2b1d255b938f
# ╠═23eb5078-4d37-43e8-960a-9f0970d00a83
# ╠═388b8009-8220-4c21-8812-2194a9ccc5da
# ╟─99a47b16-98e3-4c7e-ae30-bdde7a8a6f8e
# ╠═e55f16b0-3242-4185-b23c-37f3d8df7fc4
# ╟─d8ee80bd-da37-4187-93db-cdab8f1a827a
# ╠═cdb5a7e8-8478-4841-b14e-e1baf5acf6d9
# ╠═b52f900d-abd7-4800-a4b2-cbefd770c002
# ╠═30cfb931-b3af-4360-a432-9a5c4b22ec6d
# ╟─0bab50cf-60f9-469d-81cf-9740c0c69b2c
# ╟─72250ee1-8f9d-4ec1-a7e4-4d3bcd1255cf
# ╟─0f55b5cf-c5aa-4c54-8df3-17291ad47894
# ╠═470495da-76b7-43d0-91a1-6f08a378e95c
# ╠═2ffa9945-f1cd-46ba-9755-942a2b2ea6fe
# ╠═2811ce8e-8d0f-44ea-a41c-24b2319413f1
# ╠═18f06e6c-4e4a-4a56-8822-390da61294d8
# ╠═014d8f0f-7367-47c3-94a2-e7e9949d56be
# ╠═fcfbbce6-43d4-4494-924d-a36e2832dafa
# ╠═98774381-55a9-4a93-af33-1e00c81aedd9
# ╠═a82504ad-a837-49cc-9f2b-460c1fa68348
# ╟─224153cd-0588-4f4e-a1d8-81c92b42b868
# ╠═b84a0ca4-a931-4955-a095-b5018db6f40c
# ╠═fa5ad7ab-0a57-43ff-a6e1-a9bd73ed8566
# ╠═6eb8894e-5101-49a7-a760-6c2289a62cd2
# ╟─efe9d6c3-c8ff-4891-b0f4-5eca72bee048
# ╟─aa030517-7db9-4b23-94aa-3d96966f5347
# ╟─6276d0e8-fc97-4b04-97af-a1985d181d7c
# ╟─691cb412-8340-41d9-b775-0b838a5aee62
# ╟─a7503137-cbec-41b5-b644-f950665ec934
# ╟─ea4ee76e-ee31-442c-b7f2-1da96da5cb78
# ╠═a7a9d3ce-43eb-42ea-98eb-f10fc1c2c0f8
# ╟─ac93e260-0792-434d-93a3-877955795103
# ╠═fc0d0fac-5b35-4626-b304-2c51f1f8a898
# ╟─8ae34f97-28b5-4ccd-b980-67e38525d203
# ╟─5c080c53-9768-4bbb-b34d-0df7e40634bd
# ╟─376d0588-6c13-4b7f-99dd-9355b484f594
# ╟─da080be5-c2ed-42e6-b361-5542d58c95c7
# ╟─f8766c91-825d-41a9-9fcf-276c1ef6c708
# ╠═fa23401b-ef86-418c-ad53-132bdb384b05
# ╟─40aa5ac9-e63c-43d1-9301-c8e899ffe2b3
# ╠═4ef9f073-2cc8-4e5c-bbb2-df77bb4f1eaa
# ╠═c5981ebc-48d2-4a0d-9e3e-042e4e9fbc27
# ╠═53e58944-bb49-49e1-8d91-d9b485dfe140
# ╟─013db3d5-34e9-4447-8f4b-59c6f908e6e8
# ╟─dda86adb-96fb-4d46-8b5d-4f9713066dd0
# ╟─d80daf76-0f55-491b-a760-048b43ae3d74
# ╟─616cd58d-bd13-4ee8-a08d-2ff0ba2ebad2
# ╟─db84dc50-7a04-4475-8ea7-412307654b0d
# ╟─0d735858-c628-4a41-91b4-e86fe6854ea7
# ╟─26ea259e-2cad-4e42-b6dc-c38befd2e3cb
# ╠═98f0d534-9d86-4eef-9afb-78914e02d4f8
# ╠═be8e6e0d-04d0-4a1c-9d76-1ca6fb688fcb
# ╟─de86e2da-908e-44a9-998b-761c93297b66
# ╟─bbfe7723-c811-4389-bc8e-9a9e145ac872
# ╟─ccbeb628-6e90-4b00-abce-40aa4549c23a
# ╠═4c7efa25-a062-42bc-9685-5212fd00f398
# ╟─d4867d9f-5fd3-44d6-8d46-e383e99124dd
# ╟─8b941289-939f-4565-91dc-29756a19d0ea
# ╠═feef74be-642e-4f4b-8bbf-7cff678ced8b
# ╠═10e2c439-f214-4470-8eb5-64d87d55289f
# ╠═50c1669f-fe46-4729-b867-f8bb2784de47
# ╟─fdfd5f7c-504b-492a-aca3-4690ed17f56f
# ╠═4de6f442-370b-47ff-a59a-06bac7e8fbad
# ╠═67da98d7-c525-47bf-bee3-61efaa3231b4
# ╠═a8c60935-3497-4019-9fa4-8323e1642f02
# ╠═1af31a32-399f-4568-8748-42224fafd6ed
# ╟─7b988e80-02a3-4584-867b-38315cd30a97
# ╟─00fefa0f-c6c3-4887-a0aa-7cfdff438812
# ╟─dd310782-7f49-463f-800c-db8f206b49a5
# ╟─04e267e7-a994-4b8f-b25f-dc845a93d909
# ╠═81363dfd-868a-432f-9b9d-0a730d7ec745
# ╠═5caad033-32c0-4502-bbaa-746da59278e2
# ╠═a7ee78f2-bb3c-4536-8940-450e4c921fc7
# ╠═38d91348-c574-46a3-829a-2f14766a717d
# ╠═a11ab2ab-54d0-4d5a-846a-c16af29c0d51
# ╠═a6ca67c7-2fb2-4034-b761-b593177d9dce
# ╠═b7c60bbb-a599-4d20-9f50-0d80b3a2649f
# ╠═c4c1569f-5a33-4ea2-a41b-aef32c9b9cce
# ╠═2b45a044-3b15-4e67-b63b-2b06094e66c3
# ╠═7d0678f1-f5ef-43f8-981c-f0f5f5e63293
# ╟─4b8d413e-72da-4232-ac6b-2125f79c96cd
# ╟─c671702c-ba40-4cc0-b5b6-adf674fe9825
# ╠═2a722083-270f-4c56-bd86-b336fd4a2883
# ╟─d633893f-1abb-4dde-9e1a-19ffbbc0cd98
# ╠═16bb7a14-0d99-4878-8f0d-075b342a524a
# ╟─0ea1cfb0-09cc-4848-9d0d-558085c63cc6
# ╠═25b1dd0b-3826-40f9-a7a8-324653c1ec3f
# ╠═74150cad-3e4f-4e4d-b819-bb15769fe6d0
# ╟─effaa34e-24f4-48b6-9169-274e92aacdc9
# ╟─b0056c84-b76e-4898-ad70-753e5083f965
# ╠═2a48428d-fd0b-4d8f-899f-93377de393e3
# ╠═24a2050e-750c-4e77-85e9-c7d1859f5b3f
# ╠═ac1a4222-46e0-4242-bd7e-f1aa1ae15341
# ╠═c73139ce-1090-4522-b4bb-de3b553dd468
# ╠═1dc43eb9-74f6-4b40-8976-604f728777f0
# ╠═47bc9026-130e-4cbb-ad7c-ed02047ac036
# ╠═5e4e5fd1-dcea-4a9d-8698-880c4a110840
# ╠═5789ab7c-5062-4ad7-bb9f-34a50e4cf0fe
# ╠═2c3f066c-744c-4861-abf8-4cab5b99d9ff
# ╠═8dd794be-9145-4cf7-9df0-40d75b541783
# ╠═721026dc-ac4a-4fdd-8c63-56c05244272e
# ╠═7ff78994-872a-46e4-85ce-363a9cbf4071
# ╟─3d9404cc-9491-4d48-a56b-88174e91507a
# ╠═07116240-fe0c-499f-87cf-02d8d316f546
# ╠═41ead506-92a0-4620-ab15-22678898e169
# ╠═81337bb1-d7d5-46a3-8b2f-ac0626326f24
# ╠═e7b02c90-b394-413b-918c-6076edb334e1
# ╠═2584ce25-545c-4190-aa8f-2a0adb94cd5a
# ╠═9249a644-b6e9-4386-8d85-d9ce11d80519
# ╠═b31e6203-c3ad-465b-98e7-5606bd468401
# ╠═e123d347-2ed0-4db2-910c-73d11850c7f9
# ╠═ddd08669-d1eb-427c-a76f-8027bf5e9875
# ╠═22a27fde-8f2a-4301-afa6-4e08388f04e4
# ╠═7b1f5872-f0de-4b41-ab6b-aa2f8b1598c2
# ╠═27322c95-f130-4669-b9da-8195cdafa460
# ╠═c8368d88-8284-4054-83c1-2efab8d86678
# ╠═23fd97ba-5b99-4d53-b564-e6cc1f0140e3
# ╠═8ca72314-7ea9-4864-8698-594f57e69f31
# ╟─8d20c827-11e8-44ae-9ed3-e4f2899be007
# ╠═8eb01d9b-836b-42c3-850a-e20a5875d2e1
# ╟─e807f411-38c3-4f35-99a3-68b09ae1b3a7
# ╟─1b9078af-d7d1-4322-897e-89452ff8a4de
# ╟─3cd7f197-a86d-4210-b932-6bb1c8e5b9ec
# ╠═3163a090-ca95-4df4-9a0e-33505ee6de0e
# ╠═ba397f1d-c8fc-4521-a56e-2c71817ab1f8
# ╠═b20dafb1-282e-4fe7-ae19-5c6101a59965
# ╠═f95db313-e4af-4c2b-af24-3c2ad7459add
# ╠═43966b13-bc9c-46d8-a57e-dad0cfbb690f
# ╠═2b605f0f-f19d-4ca0-b145-4af6fa0ab346
# ╠═8e9874ba-c8c0-4882-993c-382f15f5c5ea
# ╠═b5d37679-eac3-4acd-b16e-3bc7c7a2a15b
# ╠═416eb8cd-33a5-4e4d-ad78-dfa4a127a8b4
# ╠═b8e17616-7b71-494a-a90e-999fc5c21989
# ╠═be700714-8168-4640-980c-64cd70107fb6
# ╠═15dd5412-2fc6-4243-8098-4ab74b8b9838
# ╟─2cd6da7f-1a12-40c6-9d91-2646673f559a
# ╟─295b92f3-6bbc-4ff3-b39a-589c05c3a07f
# ╟─6fea8dfb-9c00-4864-8a17-96d2bbb5bb59
# ╟─6b1daf65-fc46-42aa-9c99-91c1fec3bdda
# ╟─da88bca7-644b-401a-8fb6-d5b96d609755
# ╠═25aa07bb-6c2b-4195-bbb2-01b031a1b91e
# ╠═452efa64-3595-4388-aa9b-98ce5a0fc404
# ╟─77314512-e87a-4d39-a2c4-2bb6027aa658
# ╟─8fcd2433-619c-4202-a71c-826007f50749
# ╠═7723dc4f-43a6-4ece-80d5-88107f2fbf46
# ╟─6d251521-e502-4f96-bc65-a172cea7f224
# ╟─3c9ed6e9-b15e-4520-97ea-bdaa724a6e98
# ╠═6a547a22-1bf4-4b42-9d62-f1e14a35da47
# ╠═ddd8a237-0d77-4ef2-ac8f-b9d76c2448a6
# ╟─715ab50e-b136-41ca-b7d6-e169ef457a00
# ╠═75428084-5ff2-4996-8fce-81b09e83f287
# ╠═1c12179c-13c5-4f64-a1fe-cbe2d9d219bb
# ╠═47b8c456-c809-4fdb-9f05-9c461bc8e167
# ╠═d66f814a-a4fd-41c3-8aec-f99969355e98
# ╟─32a3159a-0f5c-49e7-af9c-1d24addbcee0
# ╟─b5207681-0976-4ca3-808c-442010dd67aa
# ╟─ca378efe-514d-4ba0-9414-7614ba84eaa5
# ╟─ffda6edf-42e1-4748-bae1-abcba9d85be3
# ╟─c8d689cd-a9c7-4cfa-b0ea-1a9ffd71010e
# ╟─998c9e14-8e8e-4a9c-b03a-9b5bfd38055f
# ╠═8432affe-4371-4528-bb79-87d6b6374871
# ╠═7d00197a-e50e-4c9c-819e-f909287bc6bc
# ╠═28e735b1-0590-4f4a-b066-deacab0f3dca
# ╠═6020c4d7-572f-41a5-bb14-3ae248ab8219
# ╠═6deb3dad-050b-4d5f-b8d5-df29f41e62b0
# ╠═7ee82745-a862-4892-a06e-63420a2d7c03
# ╟─dfa43b77-7cd7-4ae9-80c0-b473a08c7ed4
# ╟─10a482b1-fb19-4d5d-95ea-55b4900887b5
# ╟─b032b2b1-5e07-44c4-9bfb-3fb84528c123
# ╠═4e03ab81-a0bc-42a1-8753-fc6f866019d6
# ╠═af9b4662-1553-4a9e-b916-375cdbe4176a
# ╠═92288560-f4e9-4e6a-b182-cecc07b0b457
# ╟─28656452-ba55-4d46-be56-1c11c1928c23
# ╠═4d8416e0-67d4-436b-98bd-58d917aa84b3
# ╠═df47a841-db47-4002-bc37-75778e7211d7
# ╠═5738326e-6817-424c-96ba-c9dffe02e590
# ╠═2a35f221-4c6a-4b79-b300-e87e8fc770cd
# ╠═63ed4500-a199-4a31-ace1-cec54f152380
# ╟─33848c9b-fe3c-4767-abfc-c40a7a68cb56
# ╠═84445f5a-cd73-4098-b3a4-861abf42061d
# ╠═90c7aeca-aafc-4b48-b6e2-82ccd249fd24
# ╠═62b5acb7-3ec7-442d-afe5-5d76c62b5582
# ╠═ec352be3-b742-423d-8454-f8a7c44b3543
# ╟─d9fdbf41-5be5-411b-9e72-8c0cdee88761
# ╟─4f16565e-09bb-11f0-3729-7ffc5462cdc8
# ╠═f173569d-182f-4873-97ed-1b9cad2b4309
# ╠═9a4d0c70-ca15-4201-8a2e-56af95a60290
# ╠═acfabeef-f268-4c7f-a07c-4c05d1333305
# ╠═c77a29da-a2a4-4956-9795-56ce63337495
# ╠═43999574-da93-475c-9a67-e5024fb08202
# ╠═1b84943c-c8f5-4ad4-b95a-66dc818fa609
# ╟─afb3f1df-8aa7-4e57-bf03-9d901c9c2946
# ╠═6130d10b-b23c-4e15-97b3-ec7a1134e732
# ╠═b76a1c2b-f5df-41c1-8dae-4be15ae274a5
# ╠═66394600-ab20-4cc8-9319-1bcbfc5191dc
# ╠═72f9972e-9f83-4bf7-b459-764c309552b4
# ╠═acf47599-4813-412d-8dfe-a6d4967f710c
# ╠═3ab40052-d656-4e14-a6fd-3c5b09664cf4
# ╠═880a7ec3-9d3a-420d-a721-7a5616deb01b
# ╠═a14ebbce-3590-4759-81b3-253a13e73fc6
# ╠═283fef98-fc27-42c6-b8c7-579f29dd2881
# ╠═127b258b-6df0-4347-ab6d-1e9f12e130d3
# ╠═45cc5cf5-ccc7-4e2f-9027-d3335ef7f041
# ╠═38efb2be-2f3b-4325-9b60-8fec2d87c087
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
