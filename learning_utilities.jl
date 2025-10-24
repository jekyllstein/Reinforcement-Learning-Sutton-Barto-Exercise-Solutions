### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# ╔═╡ ddc38332-503c-4732-9432-8b998dfca6e5
using PlutoDevMacros, Random, Statistics, LinearAlgebra, Transducers, Base.Threads, Random, Distributions, StatsBase, StaticArrays, DataFrames, SpecialFunctions

# ╔═╡ 2648f295-b04d-4e2d-9d81-7d2f868f9051
@only_in_nb PlutoDevMacros.@frompackage @raw_str(joinpath(@__DIR__, "ApproximationUtils.jl")) using ApproximationUtils

# ╔═╡ 3a2fa0dd-1da7-41cf-bc4a-d9dbc774dc09
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI, PlutoPlotly, LaTeXStrings, PlutoProfile, HypertextLiteral, ProgressLogging, BenchmarkTools
	TableOfContents(;depth = 4)
end
  ╠═╡ =#

# ╔═╡ c9e47c3f-333e-49e8-be88-dda128cc8418
@only_in_nb begin
	include(joinpath(@__DIR__, "Chapter-09", "Chapter_09_On-policy_Prediction_with_Approximation.jl"))
	include(joinpath(@__DIR__, "Chapter-10", "Chapter_10_On_policy_Control_with_Approximation.jl"))
	include(joinpath(@__DIR__, "Chapter-11", "Chapter_11_Off_policy_Methods_with_Approximation.jl"))
	include(joinpath(@__DIR__, "Chapter-12", "Chapter_12_Eligibility_Traces.jl"))
	include(joinpath(@__DIR__, "Chapter-13", "Chapter_13_Policy_Gradient_Methods.jl"))
end

# ╔═╡ 6c56e883-d727-4caf-9dac-6dc2aeab102f
md"""
# Training Utilities

## Introduction

Often when we face a reinforcement learning problem, a lot of experimentation is required to solve it successfully, especially in the realm of approximate solution methods.  When dealing with function approximation, we can choose from a variety of value function and policy gradient techniques all with their own set of hyperparameters and implications for the function parameters themselves.  In order to help determine the most effective technique it is often necessary to perform parameter studies and have the ability to repeat learning procedures multiple times with different hyper parameters.

In the previous chapters, we have implemented all of the basic techniques of reinforcement learning which usually includes sampling from an environment for some number of steps and updating parameters or estimates.  These algorithms run for some number of steps with one or more learning rates and return a variety of results such as the learned functions themselves and performance metrics collected during training.  We can layer atop these algorithms utilities to repeat this training until some convergence metric is reached, save parameters more permanently to disk, and repeat training on the same learned parameters.  We have already seen one version of this type of utility in the form of parameter study setups.  Here we extend this utility by creating default constructors for the most common algorithms we repurpose over and over.
"""

# ╔═╡ 005259da-6f82-4bd6-ae52-a20f8c07ef00
md"""
## Parameter Studies
We already have utilities to create general parameter studies, but we often have a particular set of hyperparameters that are common to a style of learning technique.  We can broadly categorize these techniques into value methods and policy gradient methods.  Within value methods, we can use SARSA techniques that learn action values or DP techniques that derive action values from state values as long as we have enough information about the environment.  These techniques have a single learning rate but may have different ways to calculate the target value as well as and eligibility trace parameter ``\lambda``.  Also, all value techniques will have an exploration paramter ``\epsilon`` which controls how often random actions are taken during the learning process.

Policy gradient methods generally have two learning rates and two ``\lambda`` parameters and no ``\epsilon`` parameter.  We have average reward versions of both families of algorithms too that require the use of MDP environments without terminal states.

For all parameter studies we judge the results based on a numerical score.  For episodic problems this is usually the average reward gained per episode of training.  For continuing problems this is the average reward per step.  
"""

# ╔═╡ 8cf192df-e554-41b6-b9d0-71a142766021
function make_episodic_trial(algo::Function, minvalue::T) where T<:Real
	function trial(args...; kwargs...)
		output = algo(args...; kwargs...)
		rewards = output.episode_rewards
		isempty(rewards) && return minvalue
		return Statistics.mean(rewards)
	end
	return trial
end

# ╔═╡ c20fad45-4033-4c60-b7db-3d8c9148026f
function make_continuing_trial(algo::Function)
	function trial(args...; kwargs...)
		output = algo(args...; kwargs...)
		rewards = output.reward_history
		Statistics.mean(rewards)
	end
	return trial
end

# ╔═╡ f43d6616-d28e-4799-a677-b00b5811b2c1
md"""
### Value Function Methods

When we construct a parameter study for value function methods, usually we care about the learning rate and the training time.  The other parameters can have default settings.
"""

# ╔═╡ 42537afd-7655-45a2-b18b-0759982b124c
md"""
#### Episodic Studies
"""

# ╔═╡ e932d0fd-5832-41eb-a2a3-13a89a1e8751
function setup_episodic_value_parameter_studies(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3}
	function sarsa_train_linear(γ::T, α::T, λ::T, max_steps::Integer; max_episodes::Integer = typemax(Int64), trace_type = AccumulatingTrace(), kwargs...)
		if iszero(λ)
			semi_gradient_sarsa_linear(mdp, γ, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!; α = α, kwargs...)
		else
			sarsa_λ_linear(mdp, γ, λ, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!; α = α, trace_type = trace_type, kwargs...)
		end
	end
	sarsa_linear_study = setup_parameter_study(make_episodic_trial(sarsa_train_linear, typemin(T)), (:γ, :α, :λ, :max_steps), (max_episodes = typemax(Int64), compute_value = compute_sarsa_value, ϵ = one(T) / 10, trace_type = AccumulatingTrace()))

	function sarsa_train_nonlinear(γ::T, α::T, λ::T, max_steps::Integer, layer_size::Integer, num_layers::Integer, reslayers::Integer; max_episodes::Integer = typemax(Int64), trace_type = AccumulatingTrace(), kwargs...)
		hidden_layers = fill(layer_size, num_layers)
		if iszero(λ)
			semi_gradient_sarsa_fcann(mdp, γ, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; reslayers = reslayers, α = α, kwargs...)
		else
			sarsa_λ_fcann(mdp, γ, λ, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; reslayers = reslayers, α = α, trace_type = trace_type, kwargs...)
		end
	end
	
	sarsa_nonlinear_study = setup_parameter_study(make_episodic_trial(sarsa_train_nonlinear, typemin(T)), (:γ, :α, :λ, :max_steps, :layer_size, :num_layers, :reslayers), (max_episodes = typemax(Int64), compute_value = compute_sarsa_value, ϵ = one(T) / 10, trace_type = AccumulatingTrace()))

	function monte_carlo_linear(γ::T, α::T, num_episodoes::Integer; kwargs...) 
		output = gradient_monte_carlo_control_linear(mdp, γ, num_episodes, deepcopy(feature_vector), update_feature_vector!; α = α, kwargs...)
		rewards = output.reward_history
		isempty(rewards) && return typemin(T)
		Statistics.mean(rewards)
	end

	monte_carlo_linear_study = setup_parameter_study(monte_carlo_linear, (:γ, :α, :num_episodes), (compute_value = compute_sarsa_value, ϵ = one(T) / 10, max_steps = typemax(Int64), use_unfinished_episodes = true))

	function monte_carlo_nonlinear(γ::T, α::T, num_episodoes::Integer, layer_size::Integer, num_layers::Integer, reslayers::Integer; kwargs...) 
		hidden_layers = fill(layer_size, num_layers)
		output = gradient_monte_carlo_control_linear(mdp, γ, num_episodes, deepcopy(feature_vector), update_feature_vector!; reslayers = reslayers, α = α, kwargs...)
		rewards = output.reward_history
		isempty(rewards) && return typemin(T)
		Statistics.mean(rewards)
	end

	monte_carlo_nonlinear_study = setup_parameter_study(monte_carlo_nonlinear, (:γ, :α, :num_episodes, :layer_size, :num_layers, :reslayers), (compute_value = compute_sarsa_value, ϵ = one(T) / 10, max_steps = typemax(Int64), use_unfinished_episodes = true))

	(sarsa_linear_study = sarsa_linear_study, sarsa_nonlinear_study = sarsa_nonlinear_study, monte_carlo_linear_study = monte_carlo_linear_study, monte_carlo_nonlinear_study = monte_carlo_nonlinear_study)
end

# ╔═╡ 742c8135-9aac-49f3-ac9a-8430aa4c2b41
function setup_episodic_value_parameter_studies(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function, use_dp::Bool) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1, F2, F3}
	sarsa_studies = setup_episodic_value_parameter_studies(mdp, feature_vector, update_feature_vector!)
	!use_dp && return sarsa_studies
	
	function dp_train_linear(γ::T, α::T, λ::T, max_steps::Integer; max_episodes::Integer = typemax(Int64), trace_type = AccumulatingTrace(), kwargs...)
		if iszero(λ)
			semi_gradient_dp_linear(mdp, γ, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!; α = α, kwargs...)
		else
			dp_λ_linear(mdp, γ, λ, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!; α = α, trace_type = trace_type, kwargs...)
		end
	end
	dp_linear_study = setup_parameter_study(make_episodic_trial(dp_train_linear, typemin(T)), (:γ, :α, :λ, :max_steps), (max_episodes = typemax(Int64), ϵ = one(T) / 10, trace_type = AccumulatingTrace()))

	function dp_train_nonlinear(γ::T, α::T, λ::T, max_steps::Integer, layer_size::Integer, num_layers::Integer, reslayers::Integer; max_episodes::Integer = typemax(Int64), trace_type = AccumulatingTrace(), kwargs...)
		hidden_layers = fill(layer_size, num_layers)
		if iszero(λ)
			semi_gradient_dp_fcann(mdp, γ, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; reslayers = reslayers, α = α, kwargs...)
		else
			dp_λ_fcann(mdp, γ, λ, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; reslayers = reslayers, α = α, trace_type = trace_type, kwargs...)
		end
	end
	
	dp_nonlinear_study = setup_parameter_study(make_episodic_trial(dp_train_nonlinear, typemin(T)), (:γ, :α, :λ, :max_steps, :layer_size, :num_layers, :reslayers), (max_episodes = typemax(Int64), ϵ = one(T)/10, trace_type = AccumulatingTrace()))
	
	(;sarsa_studies..., dp_linear_study = dp_linear_study, dp_nonlinear_study = dp_nonlinear_study)
end

# ╔═╡ e4b0a6f5-8e6d-4b97-8d70-4fc301473be0
md"""
#### Episodic Tests
"""

# ╔═╡ 5ffaab88-af31-4b70-87c0-b047354c65ba
# ╠═╡ skip_as_script = true
#=╠═╡
const episodic_mdp = MountainCarTask.deterministic_mdp
  ╠═╡ =#

# ╔═╡ 69a45749-bed3-4527-afb4-c869c5ca4dd6
#=╠═╡
const episodic_setup = normalized_feature_setup(episodic_mdp, identity, MountainCarTask.min_vals, MountainCarTask.max_vals)
  ╠═╡ =#

# ╔═╡ 1e88e11d-67da-4dbd-8092-6e636bebf91f
#=╠═╡
episodic_value_studies = setup_episodic_value_parameter_studies(episodic_mdp, episodic_setup.feature_vector, episodic_setup.update_feature_vector!, true)
  ╠═╡ =#

# ╔═╡ 7c3d01db-4aca-4f95-91e5-ec7b3b2b6eb6
#=╠═╡
episodic_value_studies.sarsa_linear_study.update_results!(1f0, 1f-2, 0.5f0, 100_000)
  ╠═╡ =#

# ╔═╡ 2f44bb34-816a-42a0-b1d2-74493b9994a4
#=╠═╡
episodic_value_studies.sarsa_linear_study.update_results!(1f0, 1f-2, 0f0, 100_000)
  ╠═╡ =#

# ╔═╡ 6d07d39f-80e5-4a06-8ba9-7d71fea29d69
#=╠═╡
episodic_value_studies.sarsa_linear_study.update_results!(1f0, 1f-2, 0.5f0, 100_000; compute_value = compute_q_learning_value)
  ╠═╡ =#

# ╔═╡ 2800e5b9-5e3e-4d34-b71b-34edebe06a3f
#=╠═╡
episodic_value_studies.sarsa_linear_study.update_results!(1f0, 1f-2, 0.5f0, 100_000; compute_value = compute_expected_sarsa_value)
  ╠═╡ =#

# ╔═╡ 41da78e2-a8a9-4cb7-8c05-68b78991ab0b
#=╠═╡
episodic_value_studies.dp_linear_study.update_results!(1f0, 1f-2, 0.5f0, 100_000)
  ╠═╡ =#

# ╔═╡ c7e12dd0-de44-4179-9002-fcc4dcee6fbc
#=╠═╡
episodic_value_studies.dp_linear_study.update_results!(1f0, 1f-2, 0.0f0, 100_000)
  ╠═╡ =#

# ╔═╡ a72d3703-8578-454a-b293-69bd103cd186
#=╠═╡
episodic_value_studies.sarsa_nonlinear_study.update_results!(1f0, 1f-2, 0.5f0, 100_000, 16, 4, 1)
  ╠═╡ =#

# ╔═╡ d4ea0c96-4c23-485e-964b-fd52efc399b5
#=╠═╡
episodic_value_studies.sarsa_nonlinear_study.update_results!(1f0, 1f-2, 0.5f0, 100_000, 16, 4, 1; compute_value = compute_q_learning_value)
  ╠═╡ =#

# ╔═╡ 09a1958b-2990-47c9-8ce7-134d7bcde5ce
#=╠═╡
episodic_value_studies.sarsa_nonlinear_study.update_results!(1f0, 1f-2, 0.5f0, 100_000, 16, 4, 1; compute_value = compute_expected_sarsa_value)
  ╠═╡ =#

# ╔═╡ c812e9c9-b6ab-4788-a5a7-d391d270f387
#=╠═╡
episodic_value_studies.sarsa_nonlinear_study.update_results!(1f0, 1f-2, 0.0f0, 100_000, 16, 4, 1)
  ╠═╡ =#

# ╔═╡ 21344e9a-e0fa-4356-8947-b5310e6aa4e1
#=╠═╡
episodic_value_studies.dp_nonlinear_study.update_results!(1f0, 1f-2, 0.5f0, 100_000, 16, 4, 1)
  ╠═╡ =#

# ╔═╡ 30425ebb-2fe5-4624-a3c9-f5f332beca4d
#=╠═╡
episodic_value_studies.dp_nonlinear_study.update_results!(1f0, 1f-2, 0.0f0, 100_000, 16, 4, 1)
  ╠═╡ =#

# ╔═╡ 75b4c6da-80b7-4a34-bb15-f63ada17d5b4
md"""
#### Continuing Studies
"""

# ╔═╡ 5f9b180e-9201-45d5-92dd-5c0eb9de01e7
function setup_continuing_value_parameter_studies(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3}
	function sarsa_train_linear(α::T, λ::T, num_steps::Integer; trace_type = AccumulatingTrace(), kwargs...)
		if iszero(λ)
			semi_gradient_differential_sarsa_linear(mdp, num_steps, deepcopy(feature_vector), update_feature_vector!; α = α, kwargs...)
		else
			sarsa_λ_linear(mdp, λ, num_steps, deepcopy(feature_vector), update_feature_vector!; α = α, trace_type = trace_type, kwargs...)
		end
	end
	sarsa_linear_study = setup_parameter_study(make_continuing_trial(sarsa_train_linear), (:α, :λ, :num_steps), (compute_value = compute_sarsa_value, ϵ = one(T) / 10, α_r̄ = one(T) / 100, trace_type = AccumulatingTrace(),))

	function sarsa_train_nonlinear(α::T, λ::T, num_steps::Integer, layer_size::Integer, num_layers::Integer, reslayers::Integer; trace_type = AccumulatingTrace(), kwargs...)
		hidden_layers = fill(layer_size, num_layers)
		if iszero(λ)
			semi_gradient_differential_sarsa_fcann(mdp, num_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; reslayers = reslayers, α = α, kwargs...)
		else
			sarsa_λ_fcann(mdp, λ, num_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; reslayers = reslayers, α = α, trace_type = trace_type, kwargs...)
		end
	end
	
	sarsa_nonlinear_study = setup_parameter_study(make_continuing_trial(sarsa_train_nonlinear), (:α, :λ, :num_steps, :layer_size, :num_layers, :reslayers), (compute_value = compute_sarsa_value, ϵ = 0.1f0, α_r̄ = one(T) / 100, trace_type = AccumulatingTrace()))

	(sarsa_linear_study = sarsa_linear_study, sarsa_nonlinear_study = sarsa_nonlinear_study)
end

# ╔═╡ f578ab23-12cb-4f38-b76f-fef4189d31cc
function setup_continuing_value_parameter_studies(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function, use_dp::Bool) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1, F2, F3}
	sarsa_studies = setup_continuing_value_parameter_studies(mdp, feature_vector, update_feature_vector!)
	!use_dp && return sarsa_studies
	
	function dp_train_linear(α::T, λ::T, num_steps::Integer; trace_type = AccumulatingTrace(), kwargs...)
		if iszero(λ)
			semi_gradient_differential_dp_linear(mdp, num_steps, deepcopy(feature_vector), update_feature_vector!; α = α, kwargs...)
		else
			dp_λ_linear(mdp, λ, num_steps, deepcopy(feature_vector), update_feature_vector!; α = α, trace_type = trace_type, kwargs...)
		end
	end
	dp_linear_study = setup_parameter_study(make_continuing_trial(dp_train_linear), (:α, :λ, :num_steps), (ϵ = 0.1f0, α_r̄ = one(T) / 100, trace_type = AccumulatingTrace()))

	function dp_train_nonlinear(α::T, λ::T, num_steps::Integer, layer_size::Integer, num_layers::Integer, reslayers::Integer; kwargs...)
		hidden_layers = fill(layer_size, num_layers)
		if iszero(λ)
			semi_gradient_differential_dp_fcann(mdp, num_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; reslayers = reslayers, α = α, kwargs...)
		else
			dp_λ_fcann(mdp, λ, num_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; reslayers = reslayers, α = α, kwargs...)
		end
	end
	
	dp_nonlinear_study = setup_parameter_study(make_continuing_trial(dp_train_nonlinear), (:α, :λ, :num_steps, :layer_size, :num_layers, :reslayers), (ϵ = one(T) / 10, α_r̄ = one(T) / 100, trace_type = AccumulatingTrace()))
	
	(;sarsa_studies..., dp_linear_study = dp_linear_study, dp_nonlinear_study = dp_nonlinear_study)
end

# ╔═╡ 54807cf1-6efe-4625-a966-db6a57246c95
md"""
#### Continuing Tests
"""

# ╔═╡ 5967f3f3-48cb-40ee-b886-2a86d1a3b666
# ╠═╡ skip_as_script = true
#=╠═╡
const continuing_mdp, continuing_setup = create_access_control_task(10, [1f0, 2f0, 4f0, 8f0])
  ╠═╡ =#

# ╔═╡ 5626bdac-537b-4ff1-85dc-471881a6e496
#=╠═╡
continuing_value_studies = setup_continuing_value_parameter_studies(continuing_mdp, continuing_setup...)
  ╠═╡ =#

# ╔═╡ 9ee432ea-888e-4df0-902b-58741d3f8e57
#=╠═╡
continuing_value_studies.sarsa_linear_study.update_results!(1f-2, 0.5f0, 1_000)
  ╠═╡ =#

# ╔═╡ a185fcda-eeab-4fac-97b0-d479b4cce4fd
#=╠═╡
continuing_value_studies.sarsa_linear_study.update_results!(1f-2, 0.5f0, 1_000; compute_value = compute_q_learning_value)
  ╠═╡ =#

# ╔═╡ 739db9d4-4b4a-46b7-b1f4-6176dfd467e3
#=╠═╡
continuing_value_studies.sarsa_linear_study.update_results!(1f-2, 0.5f0, 1_000; compute_value = compute_expected_sarsa_value)
  ╠═╡ =#

# ╔═╡ 86cbffeb-cf4a-4d81-9b4e-2c3cbbdbaac5
#=╠═╡
continuing_value_studies.sarsa_linear_study.update_results!(1f-2, 0.0f0, 1_000)
  ╠═╡ =#

# ╔═╡ f9e6eed9-4fd4-47f7-b99a-05691fa901b3
#=╠═╡
continuing_value_studies.sarsa_nonlinear_study.update_results!(1f-2, 0.5f0, 1_000, 16, 4, 1)
  ╠═╡ =#

# ╔═╡ 9b02733b-f769-4616-8baa-c5518812aa43
#=╠═╡
continuing_value_studies.sarsa_nonlinear_study.update_results!(1f-2, 0.5f0, 1_000, 16, 4, 1; compute_value = compute_q_learning_value)
  ╠═╡ =#

# ╔═╡ cb475da0-614f-48e6-bc28-7898d4450ff1
#=╠═╡
continuing_value_studies.sarsa_nonlinear_study.update_results!(1f-2, 0.5f0, 1_000, 16, 4, 1; compute_value = compute_expected_sarsa_value)
  ╠═╡ =#

# ╔═╡ d07826fd-02a1-4cb4-991d-534e01c2a27f
#=╠═╡
continuing_value_studies.sarsa_nonlinear_study.update_results!(1f-2, 0.0f0, 1_000, 16, 4, 1)
  ╠═╡ =#

# ╔═╡ 72353588-a6d5-4d18-870c-cfeacbaac669
md"""
### Policy Gradient Methods

When we construct a parameter study for policy gradient methods, usually we care about the two learning rates (if applicable) and the training time.  The other parameters can have default settings.
"""

# ╔═╡ 4e2f55f2-4639-4017-a060-03a7487eebb8
md"""
#### Episodic Studies
"""

# ╔═╡ d2387c9d-aa6e-4eda-904a-101e5fdd3cae
function setup_episodic_policy_parameter_studies(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3}
	function ac_train_linear(γ::T, α_θ::T, α_w::T, λ_θ::T, λ_w::T, max_steps::Integer; max_episodes::Integer = typemax(Int64), trace_type = AccumulatingTrace(), kwargs...)
		if all(iszero, (λ_θ, λ_w))
			one_step_actor_critic_linear(mdp, γ, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!; α_θ = α_θ, α_w = α_w, kwargs...)
		else
			actor_critic_with_eligibility_traces_linear(mdp, γ, λ_θ, λ_w, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!; α_θ = α_θ, α_w = α_w, trace_type = trace_type, kwargs...)
		end
	end
	ac_linear_study = setup_parameter_study(make_episodic_trial(ac_train_linear, typemin(T)), (:γ, :α_θ, :α_w, :λ_θ, :λ_w, :max_steps), (max_episodes = typemax(Int64), trace_type = AccumulatingTrace()))

	function ac_train_nonlinear(γ::T, α_θ::T, α_w::T, λ_θ::T, λ_w::T, max_steps::Integer, layer_size::Integer, num_layers::Integer, reslayers::Integer; max_episodes::Integer = typemax(Int64), trace_type = AccumulatingTrace(), kwargs...)
		hidden_layers = fill(layer_size, num_layers)
		if all(iszero, (λ_θ, λ_w))
			one_step_actor_critic_fcann(mdp, γ, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; α_θ = α_θ, α_w = α_w, reslayers = reslayers, kwargs...)
		else
			actor_critic_with_eligibility_traces_fcann(mdp, γ, λ_θ, λ_w, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; α_θ = α_θ, α_w = α_w, reslayers = reslayers, trace_type = trace_type, kwargs...)
		end
	end
	ac_nonlinear_study = setup_parameter_study(make_episodic_trial(ac_train_nonlinear, typemin(T)), (:γ, :α_θ, :α_w, :λ_θ, :λ_w, :max_steps, :layer_size, :num_layers, :reslayers), (max_episodes = typemax(Int64), trace_type = AccumulatingTrace()))

	function reinforce_linear(γ::T, α_θ::T, α_w::T, num_episodes::Integer; kwargs...) 
		if iszero(α_w)
			reinforce_monte_carlo_control_linear(mdp, γ, num_episodes, deepcopy(feature_vector), update_feature_vector!; α_θ = α_θ, kwargs...)
		else
			reinforce_with_baseline_monte_carlo_control_linear(mdp, γ, num_episodes, deepcopy(feature_vector), update_feature_vector!; α_θ = α_θ, α_w = α_w, kwargs...)
		end
	end

	reinforce_linear_study = setup_parameter_study(make_episodic_trial(reinforce_linear, typemin(T)), (:γ, :α_θ, :α_w, :num_episodes), (max_steps = typemax(Int64), use_unfinished_episodes = true))

	function reinforce_nonlinear(γ::T, α_θ::T, α_w::T, num_episodes::Integer, layer_size::Integer, num_layers::Integer, reslayers::Integer; kwargs...) 
		hidden_layers = fill(layer_size, num_layers)
		if iszero(α_w)
			reinforce_monte_carlo_control_fcann(mdp, γ, num_episodes, deepcopy(feature_vector), update_feature_vector!, hidden_layers; α_θ = α_θ, reslayers = reslayers, kwargs...)
		else
			reinforce_with_baseline_monte_carlo_control_fcann(mdp, γ, num_episodes, deepcopy(feature_vector), update_feature_vector!, hidden_layers; α_θ = α_θ, α_w = α_w, reslayers = reslayers, kwargs...)
		end
	end

	reinforce_nonlinear_study = setup_parameter_study(make_episodic_trial(reinforce_nonlinear, typemin(T)), (:γ, :α_θ, :α_w, :num_episodes, :layer_size, :num_layers, :reslayers), (max_steps = typemax(Int64), use_unfinished_episodes = true))

	(ac_linear_study = ac_linear_study, ac_nonlinear_study = ac_nonlinear_study, reinforce_linear_study = reinforce_linear_study, reinforce_nonlinear_study = reinforce_nonlinear_study)
end

# ╔═╡ d87259e7-be0d-4a55-9902-b74567e16750
md"""
#### Episodic Tests
"""

# ╔═╡ b430a02b-d4a2-4167-b4b3-28cd604f1f08
#=╠═╡
episodic_policy_studies = setup_episodic_policy_parameter_studies(episodic_mdp, episodic_setup.feature_vector, episodic_setup.update_feature_vector!)
  ╠═╡ =#

# ╔═╡ abb0b7f7-db96-4804-852a-52b5e25049ea
#=╠═╡
episodic_policy_studies.ac_linear_study.update_results!(1f0, 1f-2, 1f-2, 0.5f0, 0.5f0, 100_000)
  ╠═╡ =#

# ╔═╡ 2a51bca3-0425-450b-83d4-86408197c3e8
#=╠═╡
episodic_policy_studies.ac_linear_study.update_results!(1f0, 1f-2, 1f-2, 0.0f0, 0.0f0, 100_000)
  ╠═╡ =#

# ╔═╡ 77a8c332-ec8c-4770-af12-c25b56e3e87f
#=╠═╡
episodic_policy_studies.reinforce_linear_study.update_results!(1f0, 1f-2, 1f-2, 100; max_steps = 10_000)
  ╠═╡ =#

# ╔═╡ 9bd00356-36b4-4424-b298-d78255b2642d
#=╠═╡
episodic_policy_studies.reinforce_linear_study.update_results!(1f0, 1f-2, 0f0, 100; max_steps = 10_000)
  ╠═╡ =#

# ╔═╡ ee241d93-653e-4254-a4e5-a03f98e2dd9e
#=╠═╡
episodic_policy_studies.ac_nonlinear_study.update_results!(1f0, 1f-2, 1f-2, 0.5f0, 0.5f0, 100_000, 16, 4, 1)
  ╠═╡ =#

# ╔═╡ 0613b3e4-62d2-46bd-8d42-d782bece25ca
#=╠═╡
episodic_policy_studies.ac_nonlinear_study.update_results!(1f0, 1f-2, 1f-2, 0.0f0, 0.0f0, 100_000, 16, 4, 1)
  ╠═╡ =#

# ╔═╡ 1f4ae3a7-1877-46e7-b0e5-06df88086153
#=╠═╡
episodic_policy_studies.reinforce_nonlinear_study.update_results!(1f0, 1f-2, 1f-2, 100, 16, 4, 1; max_steps = 10_000)
  ╠═╡ =#

# ╔═╡ d3fbd37e-ea13-4b8f-abe8-dfb5efa19415
md"""
#### Continuing Studies
"""

# ╔═╡ 2405bd06-0061-4856-a977-0302d911c760
function setup_continuing_policy_parameter_studies(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3}
	function ac_train_linear(α_θ::T, α_w::T, λ_θ::T, λ_w::T, num_steps::Integer; trace_type = AccumulatingTrace(), kwargs...)
		if all(iszero, (λ_θ, λ_w))
			one_step_actor_critic_linear(mdp, num_steps, deepcopy(feature_vector), update_feature_vector!; α_θ = α_θ, α_w = α_w, kwargs...)
		else
			actor_critic_with_eligibility_traces_linear(mdp, λ_θ, λ_w, num_steps, deepcopy(feature_vector), update_feature_vector!; α_θ = α_θ, α_w = α_w, trace_type = trace_type, kwargs...)
		end
	end
	ac_linear_study = setup_parameter_study(make_continuing_trial(ac_train_linear), (:α_θ, :α_w, :λ_θ, :λ_w, :num_steps), (α_r̄ = one(T)/100, trace_type = AccumulatingTrace()))

	function ac_train_nonlinear(α_θ::T, α_w::T, λ_θ::T, λ_w::T, num_steps::Integer, layer_size::Integer, num_layers::Integer, reslayers::Integer; trace_type = AccumulatingTrace(), kwargs...)
		hidden_layers = fill(layer_size, num_layers)
		if all(iszero, (λ_θ, λ_w))
			one_step_actor_critic_fcann(mdp, num_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; α_θ = α_θ, α_w = α_w, reslayers = reslayers, kwargs...)
		else
			actor_critic_with_eligibility_traces_fcann(mdp, λ_θ, λ_w, num_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; α_θ = α_θ, α_w = α_w, reslayers = reslayers, trace_type = trace_type, kwargs...)
		end
	end
	ac_nonlinear_study = setup_parameter_study(make_continuing_trial(ac_train_nonlinear), (:α_θ, :α_w, :λ_θ, :λ_w, :num_steps, :layer_size, :num_layers, :reslayers), (α_r̄ = one(T)/100, trace_type = AccumulatingTrace()))

	(ac_linear_study = ac_linear_study, ac_nonlinear_study = ac_nonlinear_study)
end

# ╔═╡ afdcbf2d-cd4b-42f6-9b33-7cb5c9700c61
md"""
#### Continuing Tests
"""

# ╔═╡ 6ddbe04d-1bbc-4350-aac6-860212f870dc
#=╠═╡
continuing_policy_studies = setup_continuing_policy_parameter_studies(continuing_mdp, continuing_setup.feature_vector, continuing_setup.update_feature_vector!)
  ╠═╡ =#

# ╔═╡ 67f858f5-af3e-446c-bc29-07f0ecc08bce
#=╠═╡
continuing_policy_studies.ac_linear_study.update_results!(1f-2, 1f-2, 0.5f0, 0.5f0, 10_000)
  ╠═╡ =#

# ╔═╡ e2fcc077-9dbe-49fd-b1e0-d5cb43b20671
#=╠═╡
continuing_policy_studies.ac_linear_study.update_results!(1f-2, 1f-2, 0.0f0, 0.0f0, 10_000)
  ╠═╡ =#

# ╔═╡ 352a4649-f395-4cf3-9266-94bf393d8a7a
#=╠═╡
continuing_policy_studies.ac_nonlinear_study.update_results!(1f-2, 1f-2, 0.5f0, 0.5f0, 10_000, 16, 4, 1)
  ╠═╡ =#

# ╔═╡ d00cdab9-e0b2-41a9-9cd1-080d844f8a57
#=╠═╡
continuing_policy_studies.ac_nonlinear_study.update_results!(1f-2, 1f-2, 0.0f0, 0.0f0, 10_000, 16, 4, 1)
  ╠═╡ =#

# ╔═╡ 4c51ea7d-0ba3-45be-bb15-31103953e2b1
md"""
## Exhaustive Training

The basic algorithms run for a fixed number of steps or episodes, but often we want to train until some convergence criteria has been reached and adjust hyperparameters to achieve better results over time.  Doing so requires caching function parameters, saving results, and resuming training from a checkpoint.  The first step to acheiving all these goals is to train repeatedly with a given set of hyper parameters until results fail to improve and save those learned parameters for later use.
"""

# ╔═╡ bbced5a1-fe8b-402b-9dc5-6b37ebe07767
md"""
### Episodic Training
"""

# ╔═╡ 3a938f27-b6fe-4411-8c41-5d1efaa8189c
function evaluate_episodic_policy_performance(mdp::StateMDP{T, S, A, P, F1, F2, F3}, π::Function, eval_steps::Integer) where {T<:Real, S, A, P, F1, F2, F3}
	(states, actions, rewards, sterm, nsteps) = runepisode(mdp; π = π, max_steps = eval_steps)
	!mdp.isterm(sterm) && return typemin(T)
	reward_sum = sum(rewards)
	episode_count = 1
	remaining_steps = eval_steps - nsteps
	while remaining_steps > 0
		(states, actions, rewards, sterm, nsteps) = runepisode(mdp; π = π, max_steps = remaining_steps)
		if mdp.isterm(sterm) 
			reward_sum += sum(rewards)
			episode_count += 1
		end
		remaining_steps -= nsteps
	end
	return reward_sum / episode_count
end

# ╔═╡ a5b13027-c3b6-488e-82a1-2ee3be6c63be
function check_reward_progress(episode_rewards::Vector{T}) where T<:Real 
	isempty(episode_rewards) && return typemin(T)
	l = length(episode_rewards)
	episode_check = ceil(Int64, l/2)
	Statistics.mean(view(episode_rewards, episode_check:l))
end

# ╔═╡ 0d583c27-134f-4651-89d9-63b599aa8c4f
function setup_episodic_value_linear_training(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function; linear_sarsa_params::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, zero(T)), linear_dp_params::Vector{T} = initialize_linear_parameters(feature_vector, zero(T))) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3}

	function reset_params(use_dp::Bool)
		if use_dp
			params = linear_dp_params
			args = (feature_vector, zero(T)) 
		else
			params = linear_sarsa_params
			args = (feature_vector, mdp, zero(T))
		end
		new_params = initialize_linear_parameters(args...)
		params .= new_params
	end

	function td_train_linear(γ::T, α::T, λ::T, max_steps::Integer; max_episodes::Integer = typemax(Int64), trace_type = AccumulatingTrace(), new_params::Bool = true, use_dp::Bool = false, kwargs...)
		new_params && reset_params(use_dp)
		params = use_dp ? linear_dp_params : linear_sarsa_params
		if iszero(λ)
			f = use_dp ? semi_gradient_dp_linear : semi_gradient_sarsa_linear
			f(mdp, γ, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!; α = α, parameters = params, kwargs...)
		else
			f = use_dp ? dp_λ_linear : sarsa_λ_linear
			f(mdp, γ, λ, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!; α = α, trace_type = trace_type, parameters = params, kwargs...)
		end
	end

	function td_train_exhaustive(γ::T, α::T, λ::T, trial_steps::Integer; use_dp::Bool = false, new_params::Bool = false, ϵ = one(T) / 10, kwargs...)
		params = use_dp ? linear_dp_params : linear_sarsa_params
		
		@info "Starting exhaustive training with γ = $γ, α = $α, and λ = $λ with $trial_steps steps per trial"
		output1 = td_train_linear(γ, zero(T), zero(T), 0; new_params = new_params, use_dp = use_dp, kwargs...)
		π(s) = rand(T) < ϵ ? rand(eachindex(mdp.actions)) : output1.value_function(s).maximizing_action
		baseline_reward = evaluate_episodic_policy_performance(mdp, π, trial_steps)
		reward1 = baseline_reward
		trial = 0
		
		@info "Baseline episode reward is $reward1, beginning first trial"
		backup_params = copy(params)
		output2 = td_train_linear(γ, α, λ, trial_steps; new_params = false, use_dp = use_dp, ϵ = ϵ, kwargs...)
		reward2 = check_reward_progress(output2.episode_rewards)

		if reward2 ≤ reward1
			@info "First trial performance of $reward2 failed to improve reward"
			params .= backup_params
			return (;output1..., performance = reward1)
		end

		episode_rewards = output2.episode_rewards
		while reward2 > reward1
			trial += 1
			@info "On trial $trial, reward improved from $reward1 to $reward2"
			output1 = output2
			reward1 = reward2
			backup_params .= params
			episode_rewards = vcat(episode_rewards, output1.episode_rewards)
			
			output2 = td_train_linear(γ, α, λ, trial_steps; new_params = false, use_dp = use_dp, ϵ = ϵ, kwargs...)
			reward2 = check_reward_progress(output2.episode_rewards)
		end

		@info "Final trial performance of $reward2 failed to improve reward.  Performance after $trial trials improved from $baseline_reward to $reward1"

		params .= backup_params
		return (;output1..., episode_rewards = episode_rewards, performance = reward1)
	end

	function td_train_rate_decay(γ::T, α_init::T, λ::T, trial_steps::Integer; new_params = false, kwargs...)
		@info "Beginning exhaustive trials with learning rate $α_init"
		output1 = td_train_exhaustive(γ, α_init, λ, trial_steps; new_params = new_params, kwargs...)
		episode_rewards = output1.episode_rewards

		α = α_init / 2
		@info "Reducing learning rate to $α for next set of trials"
		output2 = td_train_exhaustive(γ, α, λ, trial_steps; kwargs...)

		if output2.performance ≤ output1.performance
			@info "Second round performance of $(output2.performance) failed to improve reward"
			@info "Completed rate decay training after 1 round with performance $(output1.performance)"
			return output1
		end

		round = 2
		while output2.performance > output1.performance
			round += 1
			α /= 2
			output1 = output2
			episode_rewards = vcat(episode_rewards, output1.episode_rewards)
			@info "On round $round reducing learning rate to $α"
			output2 = td_train_exhaustive(γ, α, λ, trial_steps; kwargs...)
		end
		@info "Completed rate decay training after $round rounds with performance $(output1.performance)"
		return (;output1..., episode_rewards = episode_rewards)
	end

	(train = td_train_linear, train_exhaustive = td_train_exhaustive, train_rate_decay = td_train_rate_decay, sarsa_params = linear_sarsa_params, dp_params = linear_dp_params)	
end

# ╔═╡ c68eab1e-b4f1-4fb5-8b3e-f23ad0df0be0
function initialize_fcann_value_params(mdp::StateMDP, feature_vector, hidden_layers::Vector{Int64}, reslayers::Integer, use_dp::Bool)
	if use_dp
		args = (feature_vector, hidden_layers, 1, reslayers, true) 
	else
		args = (feature_vector, hidden_layers, length(mdp.actions), reslayers, true)
	end
	initialize_fcann_params(args...)
end

# ╔═╡ 98d94e3b-4ca5-4ff0-8409-9d748799931f
function setup_episodic_value_nonlinear_training(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function; fcann_parameters::Dict = Dict{NamedTuple, FCANNParams{T}}()) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3}

	function initialize_params(hidden_layers::Vector{Int64}, reslayers::Integer, use_dp::Bool; reset_params::Bool = false)
		key = (hidden_layers = hidden_layers, reslayers = reslayers, use_dp = use_dp)
		if (!haskey(fcann_parameters, key) || reset_params) 
			@info "Initializing new parameters with hidden layers = $hidden_layers and reslayers = $reslayers"
			fcann_parameters[key] = initialize_fcann_value_params(mdp, feature_vector, hidden_layers, reslayers, use_dp)
		else
			fcann_parameters[key]
		end
	end

	function td_train_nonlinear(hidden_layers::Vector{Int64}, reslayers::Integer, γ::T, α::T, λ::T, max_steps::Integer; max_episodes::Integer = typemax(Int64), trace_type = AccumulatingTrace(), new_params::Bool = true, use_dp::Bool = false, kwargs...)
		params = initialize_params(hidden_layers, reslayers, use_dp; reset_params = new_params)
		if iszero(λ)
			f = use_dp ? semi_gradient_dp_fcann : semi_gradient_sarsa_fcann
			f(mdp, γ, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; reslayers = reslayers, α = α, parameters = params, kwargs...)
		else
			f = use_dp ? dp_λ_fcann : sarsa_λ_fcann
			f(mdp, γ, λ, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; reslayers = reslayers, α = α, trace_type = trace_type, parameters = params, kwargs...)
		end
	end

	function td_train_exhaustive(hidden_layers::Vector{Int64}, reslayers::Integer, γ::T, α::T, λ::T, trial_steps::Integer; use_dp::Bool = false, new_params::Bool = false, ϵ = one(T) / 10, kwargs...)
		params = initialize_params(hidden_layers, reslayers, use_dp; reset_params = new_params)
		
		@info "Starting exhaustive training with γ = $γ, α = $α, and λ = $λ with $trial_steps steps per trial"
		output1 = td_train_nonlinear(hidden_layers, reslayers, γ, zero(T), zero(T), 0; new_params = false, use_dp = use_dp, kwargs...)
		π(s) = rand(T) < ϵ ? rand(eachindex(mdp.actions)) : output1.value_function(s).maximizing_action
		baseline_reward = evaluate_episodic_policy_performance(mdp, π, trial_steps)
		reward1 = baseline_reward
		trial = 0
		
		@info "Baseline episode reward is $reward1, beginning first trial"
		backup_params = copy(params)
		output2 = td_train_nonlinear(hidden_layers, reslayers, γ, α, λ, trial_steps; new_params = false, use_dp = use_dp, ϵ = ϵ, kwargs...)
		reward2 = check_reward_progress(output2.episode_rewards)

		if reward2 ≤ reward1
			@info "First trial performance of $reward2 failed to improve reward"
			copy!(params, backup_params)
			return (;output1..., performance = reward1)
		end

		episode_rewards = output2.episode_rewards
		while reward2 > reward1
			trial += 1
			@info "On trial $trial, reward improved from $reward1 to $reward2"
			output1 = output2
			reward1 = reward2
			copy!(backup_params, params)
			episode_rewards = vcat(episode_rewards, output1.episode_rewards)
			
			output2 = td_train_nonlinear(hidden_layers, reslayers, γ, α, λ, trial_steps; new_params = false, use_dp = use_dp, ϵ = ϵ, kwargs...)
			reward2 = check_reward_progress(output2.episode_rewards)
		end

		@info "Final trial performance of $reward2 failed to improve reward.  Performance after $trial trials improved from $baseline_reward to $reward1"

		copy!(params, backup_params)
		return (;output1..., episode_rewards = episode_rewards, performance = reward1)
	end

	function td_train_rate_decay(hidden_layers::Vector{Int64}, reslayers::Integer, γ::T, α_init::T, λ::T, trial_steps::Integer; new_params = false, use_dp = false, kwargs...)
		params = initialize_params(hidden_layers, reslayers, use_dp; reset_params = new_params)
		
		@info "Beginning exhaustive trials with learning rate $α_init"
		output1 = td_train_exhaustive(hidden_layers, reslayers, γ, α_init, λ, trial_steps; kwargs...)
		episode_rewards = output1.episode_rewards

		α = α_init / 2
		@info "Reducing learning rate to $α for next set of trials"
		output2 = td_train_exhaustive(hidden_layers, reslayers, γ, α, λ, trial_steps; kwargs...)

		if output2.performance ≤ output1.performance
			@info "Second round performance of $(output2.performance) failed to improve reward"
			@info "Completed rate decay training after 1 round with performance $(output1.performance)"
			return output1
		end

		round = 2
		while output2.performance > output1.performance
			round += 1
			α /= 2
			output1 = output2
			episode_rewards = vcat(episode_rewards, output1.episode_rewards)
			@info "On round $round reducing learning rate to $α"
			output2 = td_train_exhaustive(hidden_layers, reslayers, γ, α, λ, trial_steps; kwargs...)
		end
		@info "Completed rate decay training after $round rounds with performance $(output1.performance)"
		return (;output1..., episode_rewards = episode_rewards)
	end

	(train = td_train_nonlinear, train_exhaustive = td_train_exhaustive, train_rate_decay = td_train_rate_decay, parameters = fcann_parameters)	
end

# ╔═╡ 33aa329f-7a8b-4264-837e-19130773315f
function setup_episodic_policy_linear_training(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function; linear_policy_params::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, zero(T)), linear_value_params::Vector{T} = initialize_linear_parameters(feature_vector, zero(T))) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3}
	
	function reset_params()
		linear_policy_params .= initialize_linear_parameters(feature_vector, mdp, zero(T))
		linear_value_params .= initialize_linear_parameters(feature_vector, zero(T))
	end
	
	function ac_train_linear(γ::T, α_θ::T, α_w::T, λ_θ::T, λ_w::T, max_steps::Integer; max_episodes = typemax(Int64), trace_type = AccumulatingTrace(), new_params::Bool = true, kwargs...)
		new_params && reset_params()
		if all(iszero, (λ_θ, λ_w))
			one_step_actor_critic_linear(mdp, γ, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!; α_θ = α_θ, α_w = α_w, policy_params = linear_policy_params, value_params = linear_value_params, kwargs...)
		else
			actor_critic_with_eligibility_traces_linear(mdp, γ, λ_θ, λ_w, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!; α_θ = α_θ, α_w = α_w, trace_type = trace_type, policy_params = linear_policy_params, value_params = linear_value_params, kwargs...)
		end
	end

	function ac_train_exhaustive(γ::T, α_θ::T, α_w::T, λ_θ::T, λ_w::T, trial_steps::Integer; new_params = false, kwargs...)
		@info "Starting exhaustive training with α_θ = $(α_θ), α_w = $(α_w), λ_θ = $(λ_θ), and λ_w = $(λ_w) with $trial_steps steps per trial"
		output1 = ac_train_linear(γ, zero(T), zero(T), zero(T), zero(T), 0; new_params = new_params, kwargs...)
		π(s) = output1.policy_sample_action(s)
		baseline_reward = evaluate_episodic_policy_performance(mdp, π, trial_steps)
		reward1 = baseline_reward
		trial = 0
		
		@info "Baseline episode reward is $reward1, beginning first trial"
		backup_policy_params = copy(linear_policy_params)
		backup_value_params = copy(linear_value_params)
		output2 = ac_train_linear(γ, α_θ, α_w, λ_θ, λ_w, trial_steps; kwargs..., new_params = false)
		reward2 = check_reward_progress(output2.episode_rewards)

		if reward2 ≤ reward1
			@info "First trial performance of $reward2 failed to improve reward"
			linear_policy_params .= backup_policy_params
			linear_value_params .= backup_value_params
			return (;output1..., performance = reward1)
		end

		episode_rewards = output2.episode_rewards
		while reward2 > reward1
			trial += 1
			@info "On trial $trial, episode reward improved from $reward1 to $reward2"
			output1 = output2
			reward1 = reward2
			backup_policy_params .= linear_policy_params
			backup_value_params .= linear_value_params
			episode_rewards = vcat(episode_rewards, output1.episode_rewards)
			
			output2 = ac_train_linear(γ, α_θ, α_w, λ_θ, λ_w, trial_steps; kwargs..., new_params = false)
			reward2 = check_reward_progress(output2.episode_rewards)
		end

		@info "Final trial performance of $reward2 failed to improve reward.  Performance after $trial trials improved from $baseline_reward to $reward1"

		linear_policy_params .= backup_policy_params
		linear_value_params .= backup_value_params
		return (;output1..., episode_rewards = episode_rewards, performance = reward1)
	end

	function ac_train_rate_decay(γ, α_θ_init, α_w_init, λ_θ, λ_w, trial_steps::Integer; new_params = false, kwargs...)
		@info "Beginning exhaustive trials with learning rates $(α_θ_init) and $(α_w_init)"
		output1 = ac_train_exhaustive(γ, α_θ_init, α_w_init, λ_θ, λ_w, trial_steps; new_params = new_params, kwargs...)
		episode_rewards = output1.episode_rewards

		α_θ = α_θ_init / 2
		α_w = α_w_init / 2
		@info "Reducing learning rates to $α_θ and $α_w for next set of trials"
		output2 = ac_train_exhaustive(γ, α_θ, α_w, λ_θ, λ_w, trial_steps; kwargs...)

		round = 2
		while output2.performance > output1.performance
			episode_rewards = vcat(episode_rewards, output2.episode_rewards)
			round += 1
			α_θ = α_θ / 2
			α_w = α_w / 2
			output1 = output2
			@info "On round $round reducing learning rates to $α_θ and $α_w"
			output2 = ac_train_exhaustive(γ, α_θ, α_w, λ_θ, λ_w, trial_steps; kwargs...)
		end
		@info "Completed rate decay training after $round rounds with performance $(output1.performance)"
		return (;output1..., episode_rewards = episode_rewards)
	end

	(train = ac_train_linear, train_exhaustive = ac_train_exhaustive, train_rate_decay = ac_train_rate_decay, policy_params = linear_policy_params, value_params = linear_value_params)	
end

# ╔═╡ 8e91e2c2-a5e6-4cce-8d62-d1568bae7e08
function initialize_fcann_policy_params(mdp::StateMDP, feature_vector, hidden_layers::Vector{Int64}, reslayers::Integer)
	policy_parameters = initialize_fcann_params(feature_vector, hidden_layers, length(mdp.actions), reslayers, true)
	value_parameters = initialize_fcann_value_params(policy_parameters, true)
	(policy_parameters, value_parameters)
end

# ╔═╡ ad63e185-0618-476c-931e-f69b5f24d2a1
function setup_episodic_policy_nonlinear_training(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function; fcann_policy_parameters::Dict = Dict{NamedTuple, FCANNParams{T}}(), fcann_value_parameters::Dict = Dict{NamedTuple, FCANNParams{T}}()) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3}

	function initialize_params(hidden_layers::Vector{Int64}, reslayers::Integer; reset_params::Bool = false)
		key = (hidden_layers = hidden_layers, reslayers = reslayers)
		if (!haskey(fcann_policy_parameters, key) || reset_params) 
			@info "Initializing new parameters with hidden layers = $hidden_layers and reslayers = $reslayers"
			(policy_params, value_params) = initialize_fcann_policy_params(mdp, feature_vector, hidden_layers, reslayers)
			fcann_policy_parameters[key] = policy_params
			fcann_value_parameters[key] = value_params
		end
		(fcann_policy_parameters[key], fcann_value_parameters[key])
	end

	function ac_train_nonlinear(hidden_layers, reslayers, γ::T, α_θ::T, α_w::T, λ_θ::T, λ_w::T, max_steps::Integer; max_episodes = typemax(Int64), trace_type = AccumulatingTrace(), new_params::Bool = true, kwargs...)
		(policy_params, value_params) = initialize_params(hidden_layers, reslayers; reset_params = new_params)
		if all(iszero, (λ_θ, λ_w))
			one_step_actor_critic_fcann(mdp, γ, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; α_θ = α_θ, α_w = α_w, policy_params = policy_params, value_params = value_params, reslayers=reslayers, kwargs...)
		else
			actor_critic_with_eligibility_traces_fcann(mdp, γ, λ_θ, λ_w, max_episodes, max_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; α_θ = α_θ, α_w = α_w, trace_type = trace_type, policy_params = policy_params, value_params = value_params, reslayers=reslayers, kwargs...)
		end
	end

	function ac_train_exhaustive(hidden_layers, reslayers, γ::T, α_θ::T, α_w::T, λ_θ::T, λ_w::T, trial_steps::Integer; new_params = false, kwargs...)
		(policy_params, value_params) = initialize_params(hidden_layers, reslayers; reset_params = new_params)
		@info "Starting exhaustive training with α_θ = $(α_θ), α_w = $(α_w), λ_θ = $(λ_θ), and λ_w = $(λ_w) with $trial_steps steps per trial"
		output1 = ac_train_nonlinear(hidden_layers, reslayers, γ, zero(T), zero(T), zero(T), zero(T), 0; new_params = false, kwargs...)
		π(s) = output1.policy_sample_action(s)
		baseline_reward = evaluate_episodic_policy_performance(mdp, π, trial_steps)
		reward1 = baseline_reward
		trial = 0
		
		@info "Baseline episode reward is $reward1, beginning first trial"
		backup_policy_params = copy(policy_params)
		backup_value_params = copy(value_params)
		output2 = ac_train_nonlinear(hidden_layers, reslayers, γ, α_θ, α_w, λ_θ, λ_w, trial_steps; kwargs..., new_params = false)
		reward2 = check_reward_progress(output2.episode_rewards)

		if reward2 ≤ reward1
			@info "First trial performance of $reward2 failed to improve reward"
			copy!(policy_params, backup_policy_params)
			copy!(value_params, backup_value_params)
			return (;output1..., performance = reward1)
		end

		episode_rewards = output2.episode_rewards
		while reward2 > reward1
			trial += 1
			@info "On trial $trial, episode reward improved from $reward1 to $reward2"
			output1 = output2
			reward1 = reward2
			copy!(backup_policy_params, policy_params)
			copy!(backup_value_params, value_params)
			episode_rewards = vcat(episode_rewards, output1.episode_rewards)
			
			output2 = ac_train_nonlinear(hidden_layers, reslayers, γ, α_θ, α_w, λ_θ, λ_w, trial_steps; kwargs..., new_params = false)
			reward2 = check_reward_progress(output2.episode_rewards)
		end

		@info "Final trial performance of $reward2 failed to improve reward.  Performance after $trial trials improved from $baseline_reward to $reward1"

		copy!(policy_params, backup_policy_params)
		copy!(value_params, backup_value_params)
		return (;output1..., episode_rewards = episode_rewards, performance = reward1)
	end

	function ac_train_rate_decay(hidden_layers, reslayers, γ, α_θ_init, α_w_init, λ_θ, λ_w, trial_steps::Integer; new_params = false, kwargs...)
		(policy_params, value_params) = initialize_params(hidden_layers, reslayers; reset_params = new_params)
		@info "Beginning exhaustive trials with learning rates $(α_θ_init) and $(α_w_init)"
		output1 = ac_train_exhaustive(hidden_layers, reslayers, γ, α_θ_init, α_w_init, λ_θ, λ_w, trial_steps; kwargs...)
		episode_rewards = output1.episode_rewards

		α_θ = α_θ_init / 2
		α_w = α_w_init / 2
		@info "Reducing learning rates to $α_θ and $α_w for next set of trials"
		output2 = ac_train_exhaustive(hidden_layers, reslayers, γ, α_θ, α_w, λ_θ, λ_w, trial_steps; kwargs...)

		round = 2
		while output2.performance > output1.performance
			episode_rewards = vcat(episode_rewards, output2.episode_rewards)
			round += 1
			α_θ = α_θ / 2
			α_w = α_w / 2
			output1 = output2
			@info "On round $round reducing learning rates to $α_θ and $α_w"
			output2 = ac_train_exhaustive(hidden_layers, reslayers, γ, α_θ, α_w, λ_θ, λ_w, trial_steps; kwargs...)
		end
		@info "Completed rate decay training after $(round-1) rounds with performance $(output1.performance)"
		return (;output1..., episode_rewards = episode_rewards)
	end

	(train = ac_train_nonlinear, train_exhaustive = ac_train_exhaustive, train_rate_decay = ac_train_rate_decay, policy_params = fcann_policy_parameters, value_params = fcann_value_parameters)	
end

# ╔═╡ 857d4ddd-2b8c-4a45-ac72-81f5467d0e4c
md"""
#### Value Function Linear Example
"""

# ╔═╡ aa1b2b58-66c8-4c43-b2d1-f2de6ff982ed
#=╠═╡
episodic_linear_value_test = setup_episodic_value_linear_training(episodic_mdp, episodic_setup.feature_vector, episodic_setup.update_feature_vector!)
  ╠═╡ =#

# ╔═╡ 97bfe437-8cfb-4070-88e6-690647709b62
#=╠═╡
episodic_linear_value_result = episodic_linear_value_test.train_rate_decay(1f0, 1f-3, 0.99f0, 1_000_000; new_params = true, ϵ = 0.1f0, use_dp = false)
  ╠═╡ =#

# ╔═╡ e7653fea-304b-4958-b9e6-9ebe86b91d6f
#=╠═╡
plot(-episodic_linear_value_result.episode_rewards, Layout(yaxis_type = "log"))
  ╠═╡ =#

# ╔═╡ d9deac3f-208b-489d-b964-6d44c7e6379d
md"""
#### Value Function Non-linear Example
"""

# ╔═╡ 5443cd2e-78c8-4723-9093-df2840f59a33
#=╠═╡
episodic_nonlinear_value_test = setup_episodic_value_nonlinear_training(episodic_mdp, episodic_setup.feature_vector, episodic_setup.update_feature_vector!)
  ╠═╡ =#

# ╔═╡ eabd4d6b-ce35-41ad-845d-aa1498003814
#=╠═╡
episodic_nonlinear_value_result = episodic_nonlinear_value_test.train_rate_decay(fill(16, 2), 1, 1f0, 1f-3, 0.99f0, 1_000_000; new_params = false, ϵ = 0.05f0, use_dp = false)
  ╠═╡ =#

# ╔═╡ e9be7dd3-3b76-4043-99b1-cad431310d35
#=╠═╡
episodic_nonlinear_value_test.parameters
  ╠═╡ =#

# ╔═╡ f9c9ccb4-0291-461d-8016-8f13a9dc1c5d
md"""
#### Policy Gradient Linear Example
"""

# ╔═╡ aeed95c6-1f66-4087-a491-faf928fd8f4c
#=╠═╡
episodic_linear_policy_test = setup_episodic_policy_linear_training(episodic_mdp, episodic_setup.feature_vector, episodic_setup.update_feature_vector!)
  ╠═╡ =#

# ╔═╡ 2fb66afd-5889-4866-8a93-e8903881de9d
#=╠═╡
episodic_linear_policy_result = episodic_linear_policy_test.train_rate_decay(1f0, 4f-3, 4f-3, 0.99f0, 0.99f0, 1_000_000; new_params = true)
  ╠═╡ =#

# ╔═╡ d73cd76a-61eb-47b9-abd8-769f00601743
md"""
#### Policy Gradient Non-linear Example
"""

# ╔═╡ d7d58cdd-920e-47b4-8ef9-6b5623b85e7d
#=╠═╡
episodic_nonlinear_policy_test = setup_episodic_policy_nonlinear_training(episodic_mdp, episodic_setup.feature_vector, episodic_setup.update_feature_vector!)
  ╠═╡ =#

# ╔═╡ 984158d0-7fb1-4eb1-b904-3bc6011501ad
#=╠═╡
episodic_nonlinear_policy_result = episodic_nonlinear_policy_test.train_rate_decay(fill(2, 2), 1, 1f0, 1f-3, 1f-3, 0.5f0, 0.5f0, 100_000; new_params = false)
  ╠═╡ =#

# ╔═╡ 001c295b-9fe6-4036-9fb6-337cff79687c
#=╠═╡
plot(episodic_nonlinear_policy_result.episode_rewards)
  ╠═╡ =#

# ╔═╡ cebdb010-7e8d-4fb8-bf49-418181061ad4
md"""
### Continuing Training
"""

# ╔═╡ 64c23666-9e34-4f95-9787-2d1593725bff
function evaluate_continuing_policy_performance(mdp::StateMDP{T, S, A, P, F1, F2, F3}, π::Function, eval_steps::Integer) where {T<:Real, S, A, P, F1, F2, F3}
	(states, actions, rewards, sterm, nsteps) = runepisode(mdp; π = π, max_steps = eval_steps)
	Statistics.mean(rewards)
end

# ╔═╡ 9d244394-8523-4975-af85-f70cd0cfa430
function setup_continuing_value_linear_training(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function; linear_sarsa_params::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, zero(T)), linear_dp_params::Vector{T} = initialize_linear_parameters(feature_vector, zero(T))) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3}

	function reset_params(use_dp::Bool)
		if use_dp
			params = linear_dp_params
			args = (feature_vector, zero(T)) 
		else
			params = linear_sarsa_params
			args = (feature_vector, mdp, zero(T))
		end
		new_params = initialize_linear_parameters(args...)
		params .= new_params
	end
	
	function td_train_linear(α::T, λ::T, num_steps::Integer; new_params = true, trace_type = AccumulatingTrace(), use_dp::Bool = false, kwargs...)
		new_params && reset_params(use_dp)
		params = use_dp ? linear_dp_params : linear_sarsa_params
		if iszero(λ)
			f = use_dp ? semi_gradient_differential_dp_linear : semi_gradient_differential_sarsa_linear
			f(mdp, num_steps, deepcopy(feature_vector), update_feature_vector!; α = α, parameters = params, kwargs...)
		else
			f = use_dp ? dp_λ_linear : sarsa_λ_linear
			f(mdp, λ, num_steps, deepcopy(feature_vector), update_feature_vector!; α = α, parameters = params, trace_type = trace_type, kwargs...)
		end
	end

	function td_train_exhaustive(α::T, λ::T, trial_steps::Integer; new_params = false, use_dp = false, ϵ = one(T) / 10, kwargs...)
		params = use_dp ? linear_dp_params : linear_sarsa_params
		@info "Starting exhaustive training with α = $α and λ = $λ with $trial_steps steps per trial"
		output1 = td_train_linear(zero(T), zero(T), 0; use_dp = use_dp, new_params = new_params, kwargs...)
		π(s) = rand(T) < ϵ ? rand(eachindex(mdp.actions)) : output1.value_function(s).maximizing_action
		baseline_reward = evaluate_continuing_policy_performance(mdp, π, trial_steps)
		reward1 = baseline_reward
		trial = 0
		
		@info "Baseline average reward is $reward1, beginning first trial"
		backup_params = copy(params)
		output2 = td_train_linear(α, λ, trial_steps; use_dp = use_dp, new_params = false, ϵ = ϵ, kwargs...)
		reward2 = check_reward_progress(output2.reward_history)

		if reward2 ≤ reward1
			@info "First trial performance of $reward2 failed to improve reward"
			params .= backup_params
			return (;output1..., performance = reward1)
		end

		reward_history = output2.reward_history
		while reward2 > reward1
			trial += 1
			@info "On trial $trial, average reward improved from $reward1 to $reward2"
			output1 = output2
			reward1 = reward2
			backup_params .= params
			reward_history = vcat(reward_history, output1.reward_history)
			
			output2 = td_train_linear(α, λ, trial_steps; use_dp = use_dp, new_params = false, ϵ = ϵ, kwargs...)
			reward2 = check_reward_progress(output2.reward_history)
		end

		@info "Final trial performance of $reward2 failed to improve reward.  Performance after $trial trials improved from $baseline_reward to $reward1"

		params .= backup_params
		return (;output1..., reward_history = reward_history, performance = reward1)
	end

	function td_train_rate_decay(α_init::T, λ::T, trial_steps::Integer; new_params = false, kwargs...)
		@info "Beginning exhaustive trials with learning rate $α_init"
		output1 = td_train_exhaustive(α_init, λ, trial_steps; new_params = new_params, kwargs...)
		reward_history = output1.reward_history

		α = α_init / 2
		@info "Reducing learning rate to $α for next set of trials"
		output2 = td_train_exhaustive(α, λ, trial_steps; kwargs...)

		round = 2
		while output2.performance > output1.performance
			round += 1
			α /= 2
			output1 = output2
			reward_history = vcat(reward_history, output1.reward_history)
			@info "On round $round reducing learning rate to $α"
			output2 = td_train_exhaustive(α, λ, trial_steps; kwargs...)
		end
		@info "Completed rate decay training after $round rounds with performance $(output1.performance)"
		return (;output1..., reward_history = reward_history)
	end

	(train = td_train_linear, train_exhaustive = td_train_exhaustive, train_rate_decay = td_train_rate_decay, sarsa_params = linear_sarsa_params, dp_params = linear_dp_params)	
end

# ╔═╡ d1440c54-faaf-4bf5-a11d-f7c3afb3437f
function setup_continuing_value_nonlinear_training(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function; fcann_parameters::Dict = Dict{NamedTuple, FCANNParams{T}}()) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3}

	function initialize_params(hidden_layers::Vector{Int64}, reslayers::Integer, use_dp::Bool; reset_params::Bool = false)
		key = (hidden_layers = hidden_layers, reslayers = reslayers, use_dp = use_dp)
		if (!haskey(fcann_parameters, key) || reset_params) 
			@info "Initializing new parameters with hidden layers = $hidden_layers and reslayers = $reslayers"
			fcann_parameters[key] = initialize_fcann_value_params(mdp, feature_vector, hidden_layers, reslayers, use_dp)
		else
			fcann_parameters[key]
		end
	end

	function td_train_nonlinear(hidden_layers::Vector{Int64}, reslayers::Integer, α::T, λ::T, num_steps::Integer; trace_type = AccumulatingTrace(), new_params::Bool = true, use_dp::Bool = false, kwargs...)
		params = initialize_params(hidden_layers, reslayers, use_dp; reset_params = new_params)
		if iszero(λ)
			f = use_dp ? semi_gradient_differential_dp_fcann : semi_gradient_differential_sarsa_fcann
			f(mdp, num_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; reslayers = reslayers, α = α, parameters = params, kwargs...)
		else
			f = use_dp ? dp_λ_fcann : sarsa_λ_fcann
			f(mdp, λ, num_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; reslayers = reslayers, α = α, trace_type = trace_type, parameters = params, kwargs...)
		end
	end

	function td_train_exhaustive(hidden_layers::Vector{Int64}, reslayers::Integer, α::T, λ::T, trial_steps::Integer; use_dp::Bool = false, new_params::Bool = false, ϵ = one(T) / 10, kwargs...)
		params = initialize_params(hidden_layers, reslayers, use_dp; reset_params = new_params)
		
		@info "Starting exhaustive training with $α = $α, and λ = $λ with $trial_steps steps per trial"
		output1 = td_train_nonlinear(hidden_layers, reslayers, zero(T), zero(T), 0; new_params = false, use_dp = use_dp, kwargs...)
		π(s) = rand(T) < ϵ ? rand(eachindex(mdp.actions)) : output1.value_function(s).maximizing_action
		baseline_reward = evaluate_continuing_policy_performance(mdp, π, trial_steps)
		reward1 = baseline_reward
		trial = 0
		
		@info "Baseline average reward is $reward1, beginning first trial"
		backup_params = copy(params)
		output2 = td_train_nonlinear(hidden_layers, reslayers, α, λ, trial_steps; new_params = false, use_dp = use_dp, ϵ = ϵ, kwargs...)
		reward2 = check_reward_progress(output2.reward_history)

		if reward2 ≤ reward1
			@info "First trial performance of $reward2 failed to improve reward"
			copy!(params, backup_params)
			return (;output1..., performance = reward1)
		end

		reward_history = output2.reward_history
		while reward2 > reward1
			trial += 1
			@info "On trial $trial, reward improved from $reward1 to $reward2"
			output1 = output2
			reward1 = reward2
			copy!(backup_params, params)
			reward_history = vcat(reward_history, output1.reward_history)
			
			output2 = td_train_nonlinear(hidden_layers, reslayers, α, λ, trial_steps; new_params = false, use_dp = use_dp, ϵ = ϵ, kwargs...)
			reward2 = check_reward_progress(output2.reward_history)
		end

		@info "Final trial performance of $reward2 failed to improve reward.  Performance after $trial trials improved from $baseline_reward to $reward1"

		copy!(params, backup_params)
		return (;output1..., reward_history = reward_history, performance = reward1)
	end

	function td_train_rate_decay(hidden_layers::Vector{Int64}, reslayers::Integer, α_init::T, λ::T, trial_steps::Integer; new_params = false, use_dp = false, kwargs...)
		params = initialize_params(hidden_layers, reslayers, use_dp; reset_params = new_params)
		
		@info "Beginning exhaustive trials with learning rate $α_init"
		output1 = td_train_exhaustive(hidden_layers, reslayers, α_init, λ, trial_steps; kwargs...)
		reward_history = output1.reward_history

		α = α_init / 2
		@info "Reducing learning rate to $α for next set of trials"
		output2 = td_train_exhaustive(hidden_layers, reslayers, α, λ, trial_steps; kwargs...)

		if output2.performance ≤ output1.performance
			@info "Second round performance of $(output2.performance) failed to improve reward"
			@info "Completed rate decay training after 1 round with performance $(output1.performance)"
			return output1
		end

		round = 2
		while output2.performance > output1.performance
			round += 1
			α /= 2
			output1 = output2
			reward_history = vcat(reward_history, output1.reward_history)
			@info "On round $round reducing learning rate to $α"
			output2 = td_train_exhaustive(hidden_layers, reslayers, α, λ, trial_steps; kwargs...)
		end
		@info "Completed rate decay training after $round rounds with performance $(output1.performance)"
		return (;output1..., reward_history = reward_history)
	end

	(train = td_train_nonlinear, train_exhaustive = td_train_exhaustive, train_rate_decay = td_train_rate_decay, parameters = fcann_parameters)	
end

# ╔═╡ 93e197d7-3b7d-41a0-ae6e-2dad6c327f51
function setup_continuing_policy_linear_training(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function; linear_policy_params::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, zero(T)),	linear_value_params::Vector{T} = initialize_linear_parameters(feature_vector, zero(T))) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3}
	
	function reset_params()
		linear_policy_params .= initialize_linear_parameters(feature_vector, mdp, zero(T))
		linear_value_params .= initialize_linear_parameters(feature_vector, zero(T))
	end
	
	function ac_train_linear(α_θ::T, α_w::T, λ_θ::T, λ_w::T, num_steps::Integer; trace_type = AccumulatingTrace(), new_params::Bool = true, kwargs...)
		new_params && reset_params()
		if all(iszero, (λ_θ, λ_w))
			one_step_actor_critic_linear(mdp, num_steps, deepcopy(feature_vector), update_feature_vector!; α_θ = α_θ, α_w = α_w, policy_params = linear_policy_params, value_params = linear_value_params, kwargs...)
		else
			actor_critic_with_eligibility_traces_linear(mdp, λ_θ, λ_w, num_steps, deepcopy(feature_vector), update_feature_vector!; α_θ = α_θ, α_w = α_w, trace_type = trace_type, policy_params = linear_policy_params, value_params = linear_value_params, kwargs...)
		end
	end

	function ac_train_exhaustive(α_θ::T, α_w::T, λ_θ::T, λ_w::T, trial_steps::Integer; new_params = false, kwargs...)
		@info "Starting exhaustive training with α_θ = $(α_θ), α_w = $(α_w), λ_θ = $(λ_θ), and λ_w = $(λ_w) with $trial_steps steps per trial"
		output1 = ac_train_linear(zero(T), zero(T), zero(T), zero(T), 0; new_params = new_params, kwargs...)
		π(s) = output1.policy_sample_action(s)
		baseline_reward = evaluate_continuing_policy_performance(mdp, π, trial_steps)
		reward1 = baseline_reward
		trial = 0
		
		@info "Baseline average reward is $reward1, beginning first trial"
		backup_policy_params = copy(linear_policy_params)
		backup_value_params = copy(linear_value_params)
		output2 = ac_train_linear(α_θ, α_w, λ_θ, λ_w, trial_steps; kwargs..., new_params = false)
		reward2 = check_reward_progress(output2.reward_history)

		if reward2 ≤ reward1
			@info "First trial performance of $reward2 failed to improve reward"
			linear_policy_params .= backup_policy_params
			linear_value_params .= backup_value_params
			return (;output1..., performance = reward1)
		end

		reward_history = output2.reward_history
		while reward2 > reward1
			trial += 1
			@info "On trial $trial, average reward improved from $reward1 to $reward2"
			output1 = output2
			reward1 = reward2
			backup_policy_params .= linear_policy_params
			backup_value_params .= linear_value_params
			reward_history = vcat(reward_history, output1.reward_history)
			
			output2 = ac_train_linear(α_θ, α_w, λ_θ, λ_w, trial_steps; kwargs..., new_params = false)
			reward2 = check_reward_progress(output2.reward_history)
		end

		@info "Final trial performance of $reward2 failed to improve reward.  Performance after $trial trials improved from $baseline_reward to $reward1"

		linear_policy_params .= backup_policy_params
		linear_value_params .= backup_value_params
		return (;output1..., reward_history = reward_history, performance = reward1)
	end

	function ac_train_rate_decay(α_θ_init, α_w_init, λ_θ, λ_w, trial_steps::Integer; new_params = false, kwargs...)
		@info "Beginning exhaustive trials with learning rates $(α_θ_init) and $(α_w_init)"
		output1 = ac_train_exhaustive(α_θ_init, α_w_init, λ_θ, λ_w, trial_steps; new_params = new_params, kwargs...)
		reward_history = output1.reward_history

		α_θ = α_θ_init / 2
		α_w = α_w_init / 2
		@info "Reducing learning rates to $α_θ and $α_w for next set of trials"
		output2 = ac_train_exhaustive(α_θ, α_w, λ_θ, λ_w, trial_steps; kwargs...)

		round = 2
		while output2.performance > output1.performance
			reward_history = vcat(reward_history, output2.reward_history)
			round += 1
			α_θ = α_θ / 2
			α_w = α_w / 2
			output1 = output2
			@info "On round $round reducing learning rates to $α_θ and $α_w"
			output2 = ac_train_exhaustive(α_θ, α_w, λ_θ, λ_w, trial_steps; kwargs...)
		end
		@info "Completed rate decay training after $round rounds with performance $(output1.performance)"
		return (;output1..., reward_history = reward_history)
	end

	(train = ac_train_linear, train_exhaustive = ac_train_exhaustive, train_rate_decay = ac_train_rate_decay, policy_params = linear_policy_params, value_params = linear_value_params)	
end

# ╔═╡ 5054ef58-74fd-4fd3-aaaa-099cc00492e2
function setup_continuing_policy_nonlinear_training(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function; fcann_policy_parameters::Dict = Dict{NamedTuple, FCANNParams{T}}(), fcann_value_parameters::Dict = Dict{NamedTuple, FCANNParams{T}}()) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3}
	fcann_policy_parameters = Dict{NamedTuple, FCANNParams{T}}()
	fcann_value_parameters = Dict{NamedTuple, FCANNParams{T}}()

	function initialize_params(hidden_layers::Vector{Int64}, reslayers::Integer; reset_params::Bool = false)
		key = (hidden_layers = hidden_layers, reslayers = reslayers)
		if (!haskey(fcann_policy_parameters, key) || reset_params) 
			@info "Initializing new parameters with hidden layers = $hidden_layers and reslayers = $reslayers"
			(policy_params, value_params) = initialize_fcann_policy_params(mdp, feature_vector, hidden_layers, reslayers)
			fcann_policy_parameters[key] = policy_params
			fcann_value_parameters[key] = value_params
		end
		(fcann_policy_parameters[key], fcann_value_parameters[key])
	end

	function ac_train_nonlinear(hidden_layers, reslayers, α_θ::T, α_w::T, λ_θ::T, λ_w::T, num_steps::Integer; trace_type = AccumulatingTrace(), new_params::Bool = true, kwargs...)
		(policy_params, value_params) = initialize_params(hidden_layers, reslayers; reset_params = new_params)
		if all(iszero, (λ_θ, λ_w))
			one_step_actor_critic_fcann(mdp, num_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; α_θ = α_θ, α_w = α_w, policy_params = policy_params, value_params = value_params, reslayers=reslayers, kwargs...)
		else
			actor_critic_with_eligibility_traces_fcann(mdp, λ_θ, λ_w, num_steps, deepcopy(feature_vector), update_feature_vector!, hidden_layers; α_θ = α_θ, α_w = α_w, trace_type = trace_type, policy_params = policy_params, value_params = value_params, reslayers=reslayers, kwargs...)
		end
	end

	function ac_train_exhaustive(hidden_layers, reslayers, α_θ::T, α_w::T, λ_θ::T, λ_w::T, trial_steps::Integer; new_params = false, kwargs...)
		(policy_params, value_params) = initialize_params(hidden_layers, reslayers; reset_params = new_params)
		@info "Starting exhaustive training with α_θ = $(α_θ), α_w = $(α_w), λ_θ = $(λ_θ), and λ_w = $(λ_w) with $trial_steps steps per trial"
		output1 = ac_train_nonlinear(hidden_layers, reslayers, zero(T), zero(T), zero(T), zero(T), 0; new_params = false, kwargs...)
		π(s) = output1.policy_sample_action(s)
		baseline_reward = evaluate_continuing_policy_performance(mdp, π, trial_steps)
		reward1 = baseline_reward
		trial = 0
		
		@info "Baseline average reward is $reward1, beginning first trial"
		backup_policy_params = copy(policy_params)
		backup_value_params = copy(value_params)
		output2 = ac_train_nonlinear(hidden_layers, reslayers, α_θ, α_w, λ_θ, λ_w, trial_steps; kwargs..., new_params = false)
		reward2 = check_reward_progress(output2.reward_history)

		if reward2 ≤ reward1
			@info "First trial performance of $reward2 failed to improve reward"
			copy!(policy_params, backup_policy_params)
			copy!(value_params, backup_value_params)
			return (;output1..., performance = reward1)
		end

		reward_history = output2.reward_history
		while reward2 > reward1
			trial += 1
			@info "On trial $trial, average reward improved from $reward1 to $reward2"
			output1 = output2
			reward1 = reward2
			copy!(backup_policy_params, policy_params)
			copy!(backup_value_params, value_params)
			reward_history = vcat(reward_history, output1.reward_history)
			
			output2 = ac_train_nonlinear(hidden_layers, reslayers, α_θ, α_w, λ_θ, λ_w, trial_steps; kwargs..., new_params = false)
			reward2 = check_reward_progress(output2.reward_history)
		end

		@info "Final trial performance of $reward2 failed to improve reward.  Performance after $trial trials improved from $baseline_reward to $reward1"

		copy!(policy_params, backup_policy_params)
		copy!(value_params, backup_value_params)
		return (;output1..., reward_history = reward_history, performance = reward1)
	end

	function ac_train_rate_decay(hidden_layers, reslayers, α_θ_init, α_w_init, λ_θ, λ_w, trial_steps::Integer; new_params = false, kwargs...)
		(policy_params, value_params) = initialize_params(hidden_layers, reslayers; reset_params = new_params)
		@info "Beginning exhaustive trials with learning rates $(α_θ_init) and $(α_w_init)"
		output1 = ac_train_exhaustive(hidden_layers, reslayers, α_θ_init, α_w_init, λ_θ, λ_w, trial_steps; kwargs...)
		reward_history = output1.reward_history

		α_θ = α_θ_init / 2
		α_w = α_w_init / 2
		@info "Reducing learning rates to $α_θ and $α_w for next set of trials"
		output2 = ac_train_exhaustive(hidden_layers, reslayers, α_θ, α_w, λ_θ, λ_w, trial_steps; kwargs...)

		round = 2
		while output2.performance > output1.performance
			reward_history = vcat(reward_history, output2.reward_history)
			round += 1
			α_θ = α_θ / 2
			α_w = α_w / 2
			output1 = output2
			@info "On round $round reducing learning rates to $α_θ and $α_w"
			output2 = ac_train_exhaustive(hidden_layers, reslayers, α_θ, α_w, λ_θ, λ_w, trial_steps; kwargs...)
		end
		@info "Completed rate decay training after $(round-1) rounds with performance $(output1.performance)"
		return (;output1..., reward_history = reward_history)
	end

	(train = ac_train_nonlinear, train_exhaustive = ac_train_exhaustive, train_rate_decay = ac_train_rate_decay, policy_params = fcann_policy_parameters, value_params = fcann_value_parameters)	
end

# ╔═╡ cd7afe0e-486c-43ed-874a-8ce20a01a8bb
md"""
#### Value Function Linear Example
"""

# ╔═╡ 1568bed2-f17e-4a28-8b26-6d5cca22d1ea
#=╠═╡
continuing_linear_value_test = setup_continuing_value_linear_training(continuing_mdp, continuing_setup.feature_vector, continuing_setup.update_feature_vector!)
  ╠═╡ =#

# ╔═╡ 87365f50-017a-4773-8168-e94c6ebc0c04
#=╠═╡
continuing_linear_value_result = continuing_linear_value_test.train_rate_decay(1f-2, 0.5f0, 1_000_000; ϵ = 0.01f0, α_r̄ = 0.01f0)
  ╠═╡ =#

# ╔═╡ f9c4402f-da4e-48e8-a125-d6e5db026ae8
md"""
#### Value Function Non-linear Example
"""

# ╔═╡ ed63609b-1b7d-4075-b71f-62f1205bb122
#=╠═╡
continuing_nonlinear_value_test = setup_continuing_value_nonlinear_training(continuing_mdp, continuing_setup.feature_vector, continuing_setup.update_feature_vector!)
  ╠═╡ =#

# ╔═╡ 77466131-49bb-4ea5-9c87-423d29842b98
#=╠═╡
continuing_nonlinear_value_result = continuing_nonlinear_value_test.train_rate_decay(fill(4, 2), 1, 1f-2, 0.5f0, 100_000; ϵ = 0.01f0, α_r̄ = 0.01f0, new_params = true)
  ╠═╡ =#

# ╔═╡ 1766314b-b9ee-4be0-bdb7-7aa714cc7e6d
md"""
#### Policy Gradient Linear Example
"""

# ╔═╡ 5638255a-7d2a-481d-b724-9c72c830ca7a
#=╠═╡
continuing_linear_policy_test = setup_continuing_policy_linear_training(continuing_mdp, continuing_setup.feature_vector, continuing_setup.update_feature_vector!)
  ╠═╡ =#

# ╔═╡ 79784c4b-2ccf-4c83-8864-6376091a5c9a
#=╠═╡
continuing_linear_policy_result = continuing_linear_policy_test.train_rate_decay(1f-2, 1f-2, 0.75f0, 0.75f0, 100_000; α_r̄ = 0.01f0, new_params = true)
  ╠═╡ =#

# ╔═╡ 474df763-cfed-469a-9cde-832e9f52a1b1
md"""
#### Policy Gradient Non-linear Example
"""

# ╔═╡ 560473d7-19d5-44b4-a666-7007807a8288
#=╠═╡
continuing_nonlinear_policy_test = setup_continuing_policy_nonlinear_training(continuing_mdp, continuing_setup.feature_vector, continuing_setup.update_feature_vector!)
  ╠═╡ =#

# ╔═╡ 62a03117-f939-4e8a-9b17-dce78804641e
#=╠═╡
continuing_nonlinear_policy_result = continuing_nonlinear_policy_test.train_rate_decay(fill(2, 2), 1, 1f-3, 1f-3, 0.9f0, 0.9f0, 100_000; α_r̄ = 0.1f0, new_params = true)
  ╠═╡ =#

# ╔═╡ 95e267e2-2c3f-4ab2-bc1b-40147a3cb94a
#=╠═╡
continuing_nonlinear_policy_test.policy_params
  ╠═╡ =#

# ╔═╡ 8f21188f-3118-4933-a8a5-83d1c9ffd503
#=╠═╡
continuing_nonlinear_policy_test.value_params
  ╠═╡ =#

# ╔═╡ 487ab8b6-d9a8-4f78-a0d0-1f655450857f
md"""
## Saving and Loading Parameters from Disk

Often we want to save results from one session and load them in the future or transfer them to another machine.  We can efficiently store parameters to disk in a binary representation and use naming convensions to match parameters with their appropriate problem and algorithm combination.  In the case of non-linear parameters we also need to store the residual layer count in the same since that is necessary to specify the network architecture.  Also in the case of policy gradient methods with non-linear parameters, we will store the policy and value networks separately, but when it comes time to resuming training with them, we must link the non-output layers again since that relationship will not be saved to disk.
"""

# ╔═╡ 8abce157-f051-4637-bc37-f661eff08146
md"""
### Linear Parameters
"""

# ╔═╡ 8d535a89-8eab-4ec3-a144-cd54d3abdfee
function save_linear_value_parameters(base_name::AbstractString, mdp::StateMDP, feature_vector, params::Vector{T}) where T<:Float32
	l = length(feature_vector)
	@assert l == length(params) "Feature vector length of $l does not match parameter vector length"
	m = length(mdp.actions)
	filename = string(base_name, "_dp_linear_value_parameters_$(l)_input_$(m)_actions.bin")
	input = reshape(params, length(params), 1)
	FCANN.writeArray(input, filename)
	@info "Saved parameters to $filename"
	return filename
end

# ╔═╡ 696015af-9017-4afd-a6ab-e82e2e2a5a04
function save_linear_value_parameters(base_name::AbstractString, mdp::StateMDP, feature_vector, params::Matrix{T}) where T<:Float32
	l = length(feature_vector)
	@assert l == size(params, 1) "Feature vector length of $l does not match parameter dimension"

	num_actions = length(mdp.actions)
	@assert num_actions == size(params, 2) "MDP action count of $num_actions does not match parameter dimension"
	
	filename = string(base_name, "_sarsa_linear_value_parameters_$(l)_input_$(num_actions)_actions.bin")
	FCANN.writeArray(params, filename)
	@info "Saved parameters to $filename"
	return filename
end

# ╔═╡ c1c283f3-97b1-435c-b2c2-81415d86679e
function load_linear_value_parameters(base_name::AbstractString, mdp::StateMDP, feature_vector, use_dp::Bool)
	l = length(feature_vector)
	num_actions = length(mdp.actions)
	
	label1 = use_dp ? "dp" : "sarsa"
	label2 = "$(l)_input"
	label3 = "_$(num_actions)_actions"
	filename = string(base_name, "_$label1", "_linear_value_parameters_$label2$label3.bin")
	raw_params = FCANN.readBinArray(filename)
	!use_dp && return raw_params

	return raw_params[:]
end

# ╔═╡ f7908105-cdb4-4182-b7b1-10d1cf2ce534
function linear_value_parameters_save_check(base_name::AbstractString, mdp::StateMDP, feature_vector, use_dp::Bool)
	l = length(feature_vector)
	num_actions = length(mdp.actions)
	
	label1 = use_dp ? "dp" : "sarsa"
	label2 = "$(l)_input"
	label3 = "_$(num_actions)_actions"
	filename = string(base_name, "_$label1", "_linear_value_parameters_$label2$label3.bin")
	check = isfile(filename)
	return (check, filename)
end

# ╔═╡ 2d24fea7-915d-43df-b290-a30fdf203eb5
#=╠═╡
function linear_value_disk_test()
	dp_params = rand(Float32, 2)
	fname = save_linear_value_parameters("test1", episodic_mdp, episodic_setup.feature_vector, dp_params)
	loaded_dp_params = load_linear_value_parameters("test1", episodic_mdp, episodic_setup.feature_vector, true)
	rm(fname)
	@info "Removed $fname from disk"
	@assert (dp_params == loaded_dp_params) "Loaded dp parameters do not match originals"
	@info "Successfully tested linear dp parameter saving/loading"

	sarsa_params = rand(Float32, 2, 3)
	fname = save_linear_value_parameters("test1", episodic_mdp, episodic_setup.feature_vector, sarsa_params)
	loaded_sarsa_params = load_linear_value_parameters("test1", episodic_mdp, episodic_setup.feature_vector, false)
	rm(fname)
	@info "Removed $fname from disk"
	@assert (sarsa_params == loaded_sarsa_params) "Loaded sarsa parameters do not match originals"
	@info "Successfully tested linear sarsa parameter saving/loading"
end
  ╠═╡ =#

# ╔═╡ b30d8e53-a6d7-4e74-9687-89c4431a37bb
# ╠═╡ skip_as_script = true
#=╠═╡
linear_value_disk_test()
  ╠═╡ =#

# ╔═╡ 460fb9d2-4f40-47ec-9637-c6c6ec0a1b17
function save_linear_policy_parameters(base_name::AbstractString, mdp::StateMDP, feature_vector, policy_params::Matrix{T}, value_params::Vector{T}) where T<:Float32
	l = length(feature_vector)
	m = size(policy_params, 2)
	
	@assert length(feature_vector) == length(value_params) "Parameter dimensions do not match feature vector"
	@assert length(mdp.actions) == size(policy_params, 2) "Policy parameter dimensions do not match MDP action space"
	@assert size(policy_params, 1) == length(value_params) "Policy and value parameter dimensions do not match"
	
	filename1 = string(base_name, "_linear_policy_parameters_$(l)_input_$(m)_actions.bin")
	filename2 = string(base_name, "_linear_value_parameters_$(l)_input_$(m)_actions.bin")
	value_input = reshape(value_params, length(value_params), 1)
	FCANN.writeArray(policy_params, filename1)
	FCANN.writeArray(value_input, filename2)
	@info "Saved policy parameters to $filename1 and value parameters to $filename2"
	return (filename1, filename2)
end

# ╔═╡ 9cab6940-85df-47f0-a6c6-c5f7ef2d2f10
function linear_policy_parameters_save_check(base_name::AbstractString, mdp::StateMDP, feature_vector)
	l = length(feature_vector)
	m = length(mdp.actions)
	filename1 = string(base_name, "_linear_policy_parameters_$(l)_input_$(m)_actions.bin")
	filename2 = string(base_name, "_linear_value_parameters_$(l)_input_$(m)_actions.bin")

	check = (isfile(filename1) && isfile(filename2))
	return (check, filename1, filename2)
end

# ╔═╡ 78589481-3163-48aa-a8d9-51d258f6a930
function load_linear_policy_parameters(base_name::AbstractString, mdp::StateMDP, feature_vector)
	l = length(feature_vector)
	m = length(mdp.actions)
	filename1 = string(base_name, "_linear_policy_parameters_$(l)_input_$(m)_actions.bin")
	filename2 = string(base_name, "_linear_value_parameters_$(l)_input_$(m)_actions.bin")
	
	policy_params = FCANN.readBinArray(filename1)
	value_params = FCANN.readBinArray(filename2)[:]
	return (policy_params, value_params)
end

# ╔═╡ 69b62157-1af5-4aed-959c-b0eefebf7389
#=╠═╡
function linear_policy_disk_test()
	policy_params = rand(Float32, 2, 3)
	value_params = rand(Float32, 2)
	fname1, fname2 = save_linear_policy_parameters("test1", episodic_mdp, episodic_setup.feature_vector, policy_params, value_params)
	loaded_linear_params = load_linear_policy_parameters("test1", episodic_mdp, episodic_setup.feature_vector)
	rm(fname1)
	rm(fname2)
	@info "Removed $fname1 and $fname2 from disk"
	@assert (policy_params == loaded_linear_params[1]) "Loaded policy parameters do not match originals"
	@assert (value_params == loaded_linear_params[2]) "Loaded value parameters do not match originals"
	@info "Successfully tested linear policy parameter saving/loading"
end
  ╠═╡ =#

# ╔═╡ 40e18712-6715-4074-89f7-40d4751e8d20
# ╠═╡ skip_as_script = true
#=╠═╡
linear_policy_disk_test()
  ╠═╡ =#

# ╔═╡ 7d454b42-050b-4c2a-a9b2-2c445fd9fec1
function setup_value_linear_training(basename::AbstractString, isepisodic::Bool, mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function; linear_sarsa_params::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, zero(T)), linear_dp_params::Vector{T} = initialize_linear_parameters(feature_vector, zero(T))) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3}
	(check1, fname1) = linear_value_parameters_save_check(basename, mdp, feature_vector, true)
	(check2, fname2) = linear_value_parameters_save_check(basename, mdp, feature_vector, false)
	if check1 && check2
		linear_dp_params = load_linear_value_parameters(basename, mdp, feature_vector, true)
		linear_sarsa_params = load_linear_value_parameters(basename, mdp, feature_vector, false)
	end

	setup = isepisodic ? setup_episodic_value_linear_training : setup_continuing_value_linear_training
	output = setup(mdp, feature_vector, update_feature_vector!; linear_sarsa_params = linear_sarsa_params, linear_dp_params = linear_dp_params)

	function save_params()
		save_linear_value_parameters(basename, mdp, feature_vector, linear_dp_params)
		save_linear_value_parameters(basename, mdp, feature_vector, linear_sarsa_params)
	end

	function erase_params()
		rm(fname1)
		rm(fname2)
		@info "Erased parameters at $fname1 and $fname2"
	end

	(;output..., save_params = save_params, erase_params = erase_params)
end

# ╔═╡ a131a509-3b5d-484e-a892-098f3518092c
#=╠═╡
begin
	value_disk_linear_test = setup_value_linear_training("test_episodic", true, episodic_mdp, episodic_setup.feature_vector, episodic_setup.update_feature_vector!)
	value_disk_linear_test.train_rate_decay(1f0, 1f-2, 0.5f0, 10_000; use_dp = true)
	value_disk_linear_test.train_rate_decay(1f0, 1f-2, 0.5f0, 10_000; use_dp = false)
	value_disk_linear_test.save_params()
	value_disk_linear_load_test = setup_value_linear_training("test_episodic", true, episodic_mdp, episodic_setup.feature_vector, episodic_setup.update_feature_vector!)
	@assert !iszero(sum(value_disk_linear_load_test.dp_params)) "Load test failed"
	value_disk_linear_test.erase_params()
end
  ╠═╡ =#

# ╔═╡ 5e5051fb-e0fe-4108-b581-e7ac9b7d2198
function setup_policy_linear_training(basename::AbstractString, isepisodic::Bool, mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function; linear_policy_params::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, zero(T)), linear_value_params::Vector{T} = initialize_linear_parameters(feature_vector, zero(T))) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3}
	(check, fname1, fname2) = linear_policy_parameters_save_check(basename, mdp, feature_vector)
	if check
		(linear_policy_params, linear_value_params) = load_linear_policy_parameters(basename, mdp, feature_vector)
	end

	setup = isepisodic ? setup_episodic_policy_linear_training : setup_continuing_policy_linear_training
	output = setup(mdp, feature_vector, update_feature_vector!; linear_policy_params = linear_policy_params, linear_value_params = linear_value_params)

	save_params() = save_linear_policy_parameters(basename, mdp, feature_vector, linear_policy_params, linear_value_params)
	function erase_params()
		rm(fname1)
		rm(fname2)
		@info "Erased parameters from disk at $fname1 and $fname2"
	end

	(;output..., save_params = save_params, erase_params = erase_params)
end

# ╔═╡ 94f9d831-2fbc-43b7-b199-d9c547932e49
#=╠═╡
begin
	policy_disk_linear_test = setup_policy_linear_training("test_episodic", true, episodic_mdp, episodic_setup.feature_vector, episodic_setup.update_feature_vector!)
	policy_disk_linear_test.train_rate_decay(1f0, 1f-2, 1f-2, 0.5f0, 0.5f0, 10_000)
	policy_disk_linear_test.save_params()
	policy_disk_linear_load_test = setup_policy_linear_training("test_episodic", true, episodic_mdp, episodic_setup.feature_vector, episodic_setup.update_feature_vector!)
	@assert !iszero(sum(policy_disk_linear_load_test.value_params)) "Load test failed"
	policy_disk_linear_test.erase_params()
end
  ╠═╡ =#

# ╔═╡ 10d1e2fe-605f-49f7-b06a-e8ce97dfba95
md"""
### Non-linear Parameters
"""

# ╔═╡ 673a2a41-5df6-4b45-93b3-33251c39e953
function save_nonlinear_value_parameters(base_name::AbstractString, mdp, feature_vector, params::FCANNParams{T}) where T<:Float32
	(input_size, hidden_layers, num_layers) = get_network_dimensions(params)
	output_size = params.weights[2][end] |> length
	@assert input_size == length(feature_vector) "Parameter dimensions do not match feature vector"
	@assert output_size == 1 || output_size == length(mdp.actions) "Parameter dimensions do not match MDP action space"
	label = (output_size == 1) ? "dp" : "sarsa"
	reslayers = params.reslayers
	filename = string(base_name, "_$(label)_nonlinear_value_parameters_$(input_size)_input_$(hidden_layers)_hidden_$(reslayers)_reslayers_$(length(mdp.actions))_actions.bin")
	FCANN.writeParams([params.weights], filename)
	@info "Saved parameters to $filename"
	return filename
end

# ╔═╡ 379ae227-a73a-4693-8941-16f3a725737a
function load_nonlinear_value_parameters(base_name::AbstractString, mdp::StateMDP, feature_vector, hidden_layers::Vector{Int64}, reslayers::Integer, use_dp::Bool)
	l = length(feature_vector)
	num_actions = length(mdp.actions)
	
	label1 = use_dp ? "dp" : "sarsa"
	label2 = "$(l)_input"
	label3 = "$(num_actions)_actions"
	filename = string(base_name, "_$label1", "_nonlinear_value_parameters_$(label2)_$(hidden_layers)_hidden_$(reslayers)_reslayers_$label3.bin")
	raw_params = FCANN.readBinParams(filename)
	(weights = raw_params[1], reslayers = reslayers)
end

# ╔═╡ c3595d58-9c2d-4953-a760-e05ac4b5e6b6
function nonlinear_value_parameters_save_check(base_name::AbstractString, mdp::StateMDP, feature_vector, hidden_layers::Vector{Int64}, reslayers::Integer, use_dp::Bool)
	l = length(feature_vector)
	num_actions = length(mdp.actions)
	
	label1 = use_dp ? "dp" : "sarsa"
	label2 = "$(l)_input"
	label3 = "$(num_actions)_output"
	filename = string(base_name, "_$label1", "_nonlinear_value_parameters_$(label2)_$(hidden_layers)_hidden_$(reslayers)_reslayers_$label3.bin")

	check = isfile(filename)
	return (check, filename)
end

# ╔═╡ 5dabe906-e44c-47c5-b12c-22f829bbc2c9
function save_nonlinear_value_parameters(base_name::AbstractString, mdp, feature_vector, params_dict::Dict{N, F}) where {N<:NamedTuple, F<:FCANNParams}
	for k in keys(params_dict)
		save_nonlinear_value_parameters(base_name, mdp, feature_vector, params_dict[k])
	end
end

# ╔═╡ aee066a0-e458-475e-ac15-7fa878d8ce87
function save_nonlinear_policy_parameters(base_name::AbstractString, mdp, feature_vector, policy_params::FCANNParams{T}, value_params::FCANNParams{T}) where T<:Float32
	(input_size, hidden_layers, num_layers) = get_network_dimensions(policy_params)
	output_size = policy_params.weights[2][end] |> length
	value_output_size = value_params.weights[2][end] |> length
	@assert input_size == length(feature_vector) "Parameter dimensions do not match feature vector"
	@assert output_size == length(mdp.actions) "Policy parameter dimensions do not match MDP action space"
	@assert value_output_size == 1 "Value parameter output size is not 1"
	
	reslayers = policy_params.reslayers
	filename1 = string(base_name, "_nonlinear_policy_parameters_$(input_size)_input_$(hidden_layers)_hidden_$(reslayers)_reslayers_$(length(mdp.actions))_actions.bin")
	filename2 = string(base_name, "_nonlinear_value_parameters_$(input_size)_input_$(hidden_layers)_hidden_$(reslayers)_reslayers_$(length(mdp.actions))_actions.bin")

	FCANN.writeParams([policy_params.weights], filename1)
	FCANN.writeParams([value_params.weights], filename2)
	@info "Saved parameters to $filename1 and $filename2"
	return (filename1, filename2)
end

# ╔═╡ b596752c-6c9e-451e-963d-07682db396d9
function load_nonlinear_policy_parameters(base_name::AbstractString, mdp::StateMDP, feature_vector, hidden_layers::Vector{Int64}, reslayers::Integer)
	l = length(feature_vector)
	num_actions = length(mdp.actions)
	label2 = "$(l)_input"
	label3 = "$(num_actions)_actions"
	
	filename1 = string(base_name, "_nonlinear_policy_parameters_$(label2)_$(hidden_layers)_hidden_$(reslayers)_reslayers_$label3.bin")
	filename2 = string(base_name, "_nonlinear_value_parameters_$(label2)_$(hidden_layers)_hidden_$(reslayers)_reslayers_$label3.bin")

	raw_policy_params = FCANN.readBinParams(filename1)
	raw_value_params = FCANN.readBinParams(filename2)
	
	policy_params = (weights = raw_policy_params[1], reslayers = reslayers)
	value_params = initialize_fcann_value_params(policy_params, true)
	value_params.weights[1][end] .= raw_value_params[1][1][end]
	value_params.weights[2][end] .= raw_value_params[1][2][end]

	return (policy_params, value_params)
end

# ╔═╡ 81a61620-3cfa-4a70-b934-e190fed83284
function nonlinear_policy_parameters_save_check(base_name::AbstractString, mdp::StateMDP, feature_vector, hidden_layers::Vector{Int64}, reslayers::Integer)
	l = length(feature_vector)
	num_actions = length(mdp.actions)
	label2 = "$(l)_input"
	label3 = "$(num_actions)_actions"
	
	filename1 = string(base_name, "_nonlinear_policy_parameters_$(label2)_$(hidden_layers)_hidden_$(reslayers)_reslayers_$label3.bin")
	filename2 = string(base_name, "_nonlinear_value_parameters_$(label2)_$(hidden_layers)_hidden_$(reslayers)_reslayers_$label3.bin")

	check = (isfile(filename1) && isfile(filename2))
	return (check, filename1, filename2)
end

# ╔═╡ 75247e5b-f1bb-4409-aaed-16cc8c0e0538
function save_nonlinear_policy_parameters(base_name::AbstractString, mdp, feature_vector, policy_params_dict::Dict{N, F}, value_params_dict::Dict{N, F}) where {N<:NamedTuple, F<:FCANNParams}
	for k in keys(policy_params_dict)
		save_nonlinear_policy_parameters(base_name, mdp, feature_vector, policy_params_dict[k], value_params_dict[k])
	end
end

# ╔═╡ ef178187-d3b9-4cf2-82c8-585d5e89ac01
function load_nonlinear_value_parameters(base_name::AbstractString, mdp::StateMDP, feature_vector)
	l = length(feature_vector)
	num_actions = length(mdp.actions)
	parameters = Dict{NamedTuple, FCANNParams}()
	
	for label1 in ["dp", "sarsa"]
		str1 = 
		str = Regex("^\\Q$base_name\\E_\\Q$label1\\E_nonlinear_value_parameters_\\Q$l\\E_input_\\[([\\d,\\s]+)\\]_hidden_(\\d+)_reslayers_\\Q$num_actions\\E_actions\\.bin\$")
		use_dp = (label1 == "dp")
		for f in readdir()
		 	m = match(str, f)
		 	if !isnothing(m)
				try
					@info "Loading parameters from disk with filename $f"
					hidden_layers = parse.(Int64, split(m.captures[1], ','))
					reslayers = parse(Int64, m.captures[2])
					params = load_nonlinear_value_parameters(base_name, mdp, feature_vector, hidden_layers, reslayers, use_dp)
					parameters[(hidden_layers = hidden_layers, reslayers = reslayers, use_dp = use_dp)] = params
				catch
					@warn "Could not load parameters from $m" 
				end
		 	end
		end
	end
	return parameters
end

# ╔═╡ 6bdf813e-cd76-46df-afe7-5210435df5e7
#=╠═╡
function nonlinear_value_disk_test()
	dp_params = initialize_fcann_params(2, [2, 2], 1, 1, true)
	fname = save_nonlinear_value_parameters("test1", episodic_mdp, episodic_setup.feature_vector, dp_params)
	loaded_dp_params = load_nonlinear_value_parameters("test1", episodic_mdp, episodic_setup.feature_vector, [2, 2], 1, true)
	rm(fname)
	@info "Removed $fname from disk"
	@assert (dp_params == loaded_dp_params) "Loaded dp parameters do not match originals"
	@info "Successfully tested nonlinear dp parameter saving/loading"

	sarsa_params = initialize_fcann_params(2, [2, 2], 3, 1, true)
	fname = save_nonlinear_value_parameters("test1", episodic_mdp, episodic_setup.feature_vector, sarsa_params)
	loaded_sarsa_params = load_nonlinear_value_parameters("test1", episodic_mdp, episodic_setup.feature_vector, [2, 2], 1, false)
	rm(fname)
	@info "Removed $fname from disk"
	@assert (sarsa_params == loaded_sarsa_params) "Loaded sarsa parameters do not match originals"
	@info "Successfully tested nonlinear sarsa parameter saving/loading"
end
  ╠═╡ =#

# ╔═╡ 90976177-23f6-465f-8646-48c77ce5c5a3
#=╠═╡
nonlinear_value_disk_test()
  ╠═╡ =#

# ╔═╡ ed6bd002-a0a1-49d8-a4a5-f76d3443576d
function erase_nonlinear_value_parameters(base_name::AbstractString, mdp::StateMDP, feature_vector)
	l = length(feature_vector)
	num_actions = length(mdp.actions)
	for label1 in ["dp", "sarsa"]
		str1 = 
		str = Regex("^\\Q$base_name\\E_\\Q$label1\\E_nonlinear_value_parameters_\\Q$l\\E_input_\\[([\\d,\\s]+)\\]_hidden_(\\d+)_reslayers_\\Q$num_actions\\E_actions\\.bin\$")
		use_dp = (label1 == "dp")
		for f in readdir()
		 	m = match(str, f)
		 	if !isnothing(m)
				@info "Deleting parameters from disk with filename $f"
				rm(f)
		 	end
		end
	end
end

# ╔═╡ 707b1a25-0d56-486d-a187-b17b922d49c9
function setup_value_nonlinear_training(base_name::AbstractString, isepisodic::Bool, mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function; fcann_parameters::Dict = Dict{NamedTuple, FCANNParams{T}}()) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3}
	loaded_parameters = load_nonlinear_value_parameters(base_name, mdp, feature_vector)
	for k in keys(loaded_parameters)
		fcann_parameters[k] = loaded_parameters[k]
	end

	setup = isepisodic ? setup_episodic_value_nonlinear_training : setup_continuing_value_nonlinear_training

	output = setup(mdp, feature_vector, update_feature_vector!; fcann_parameters = fcann_parameters)

	save_params() = save_nonlinear_value_parameters(base_name, mdp, feature_vector, fcann_parameters)
	erase_params() = erase_nonlinear_value_parameters(base_name, mdp, feature_vector)
	(;output..., save_params = save_params, erase_params = erase_params)
end

# ╔═╡ 102759b5-f467-4768-8e5e-78806a5b9ad6
#=╠═╡
begin
	value_disk_nonlinear_test = setup_value_nonlinear_training("nonlinear_value_test_episodic", true, episodic_mdp, episodic_setup.feature_vector, episodic_setup.update_feature_vector!)

	for layer_size = [4, 8, 16] for use_dp = [true, false]
		value_disk_nonlinear_test.train_rate_decay(fill(layer_size, 4), 1, 1f0, 1f-2, 0.5f0, 1_000; use_dp = use_dp)
	end end
	value_disk_nonlinear_test.save_params()
		
	value_disk_nonlinear_load_test = setup_value_nonlinear_training("nonlinear_value_test_episodic", true, episodic_mdp, episodic_setup.feature_vector, episodic_setup.update_feature_vector!)

	@assert !isempty(value_disk_nonlinear_load_test.parameters) "Load test failed"

	value_disk_nonlinear_test.erase_params()
end
  ╠═╡ =#

# ╔═╡ 3624fa9e-a3bb-437f-a8db-1b1662e3ba31
function load_nonlinear_policy_parameters(base_name::AbstractString, mdp::StateMDP, feature_vector)
	l = length(feature_vector)
	num_actions = length(mdp.actions)
	value_parameters = Dict{NamedTuple, FCANNParams}()
	policy_parameters = Dict{NamedTuple, FCANNParams}()

	str = Regex("^\\Q$base_name\\E_nonlinear_policy_parameters_\\Q$l\\E_input_\\[([\\d,\\s]+)\\]_hidden_(\\d+)_reslayers_\\Q$num_actions\\E_actions\\.bin\$")	
	for f in readdir()
	 	m = match(str, f)
		if !isnothing(m)
			try
				@info "Loading parameters from disk with filename $f"
				hidden_layers = parse.(Int64, split(m.captures[1], ','))
				reslayers = parse(Int64, m.captures[2])
				policy_params, value_params = load_nonlinear_policy_parameters(base_name, mdp, feature_vector, hidden_layers, reslayers)
				policy_parameters[(hidden_layers = hidden_layers, reslayers = reslayers)] = policy_params
				value_parameters[(hidden_layers = hidden_layers, reslayers = reslayers)] = value_params
			catch
				@warn "Could not load parameters from $m" 
			end
		end
	end
	return policy_parameters, value_parameters
end

# ╔═╡ 0a1d6fb2-f99f-469f-8946-b68841d46171
#=╠═╡
function nonlinear_policy_disk_test()
	policy_params = initialize_fcann_params(2, [2, 2], 3, 1, true)
	value_params = initialize_fcann_value_params(policy_params, true)
	fname1, fname2 = save_nonlinear_policy_parameters("test1", episodic_mdp, episodic_setup.feature_vector, policy_params, value_params)
	loaded_policy_params, loaded_value_params = load_nonlinear_policy_parameters("test1", episodic_mdp, episodic_setup.feature_vector, [2, 2], 1)
	rm(fname1)
	rm(fname2)
	@info "Removed $fname1 from disk"
	@info "Removed $fname2 from disk"
	@assert (policy_params == loaded_policy_params) "Loaded policy parameters do not match originals"
	@assert (value_params == loaded_value_params) "Loaded value parameters do not match originals"
	@info "Successfully tested nonlinear policy and value parameter saving/loading"
end
  ╠═╡ =#

# ╔═╡ 7042bc0c-ad15-4e73-89e1-aa8028335ce0
#=╠═╡
nonlinear_policy_disk_test()
  ╠═╡ =#

# ╔═╡ 9394e249-1d18-47eb-8b3d-15c71586af53
function erase_nonlinear_policy_parameters(base_name::AbstractString, mdp::StateMDP, feature_vector)
	l = length(feature_vector)
	num_actions = length(mdp.actions)

	
	str1 = Regex("^\\Q$base_name\\E_nonlinear_policy_parameters_\\Q$l\\E_input_\\[([\\d,\\s]+)\\]_hidden_(\\d+)_reslayers_\\Q$num_actions\\E_actions\\.bin\$")	
	str2 = Regex("^\\Q$base_name\\E_nonlinear_value_parameters_\\Q$l\\E_input_\\[([\\d,\\s]+)\\]_hidden_(\\d+)_reslayers_\\Q$num_actions\\E_actions\\.bin\$")	

	for f in readdir()
		for str in [str1, str2]
		 	m = match(str, f)
			if !isnothing(m)
				@info "Deleting parameters from disk with filename $f"
				rm(f)
			end
		end
	end
end

# ╔═╡ c182475b-c1b0-4835-8347-f0f4a831909b
function setup_policy_nonlinear_training(base_name::AbstractString, isepisodic::Bool, mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, update_feature_vector!::Function; fcann_policy_parameters::Dict = Dict{NamedTuple, FCANNParams{T}}(), fcann_value_parameters::Dict = Dict{NamedTuple, FCANNParams{T}}()) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3}
	loaded_policy_parameters, loaded_value_parameters = load_nonlinear_policy_parameters(base_name, mdp, feature_vector)
	for k in keys(loaded_policy_parameters)
		fcann_policy_parameters[k] = loaded_policy_parameters[k]
		fcann_value_parameters[k] = loaded_value_parameters[k]
	end

	setup = isepisodic ? setup_episodic_policy_nonlinear_training : setup_continuing_policy_nonlinear_training

	output = setup(mdp, feature_vector, update_feature_vector!; fcann_policy_parameters = fcann_policy_parameters, fcann_value_parameters = fcann_value_parameters)

	save_params() = save_nonlinear_policy_parameters(base_name, mdp, feature_vector, fcann_policy_parameters, fcann_value_parameters)
	erase_params() = erase_nonlinear_policy_parameters(base_name, mdp, feature_vector)
	(;output..., save_params = save_params, erase_params = erase_params)
end

# ╔═╡ b5c0be58-c2cd-440e-9b8b-254e3990ff44
#=╠═╡
begin
	policy_disk_nonlinear_test = setup_policy_nonlinear_training("nonlinear_policy_test_episodic", true, episodic_mdp, episodic_setup.feature_vector, episodic_setup.update_feature_vector!)

	for layer_size = [4, 8, 16]
		policy_disk_nonlinear_test.train_rate_decay(fill(layer_size, 4), 1, 1f0, 1f-2, 1f-2, 0.5f0, 0.5f0, 1_000)
	end
	policy_disk_nonlinear_test.save_params()
		
	policy_disk_nonlinear_load_test = setup_policy_nonlinear_training("nonlinear_policy_test_episodic", true, episodic_mdp, episodic_setup.feature_vector, episodic_setup.update_feature_vector!)

	@assert !isempty(policy_disk_nonlinear_load_test.policy_params) "Load test failed"

	policy_disk_nonlinear_test.erase_params()
end
  ╠═╡ =#

# ╔═╡ 6245ffaa-acb4-11f0-3a8d-47ce889cb225
md"""
# Dependencies
"""

# ╔═╡ 5575d394-0178-4935-b3aa-949c3ec38b45
# ╠═╡ skip_as_script = true
#=╠═╡
import HypertextLiteral.@htl
  ╠═╡ =#

# ╔═╡ 73e5b3d9-a675-4a1e-83fd-6c6e69ef0e9d
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

# ╔═╡ 455f956d-6c92-46e8-90d4-d62167d455cb
#=╠═╡
function smooth_error(error_history, n)
	l = length(error_history)
	[mean(view(error_history, i-n:i)) for i in n+1:l]
end
  ╠═╡ =#

# ╔═╡ 65068069-1374-4344-83e7-950a894957b9
#=╠═╡
begin
	plot_rewards(rewards::AbstractVector{T}, nsmooth::Integer, npoints::Integer) where T<:Real = plot(smooth_error(rewards, nsmooth)[round.(Int64, LinRange(1, length(rewards) - nsmooth, npoints))])

	function plot_rewards(rewards::AbstractVector{A}, nsmooth::Integer, npoints::Integer) where A <: Union{Missing, T} where T<:Real 	
		newrewards = [!ismissing(a) for a in rewards]
		plot_rewards(newrewards, nsmooth, npoints)
	end
end
  ╠═╡ =#

# ╔═╡ ddd87cf8-b424-469d-900e-5c46057aa05f
#=╠═╡
plot_rewards(-episodic_nonlinear_value_result.episode_rewards, 100, 1000)
  ╠═╡ =#

# ╔═╡ e2161522-ec87-47ff-89a5-d683c64f75a1
#=╠═╡
plot_rewards(episodic_linear_policy_result.episode_rewards, 100, 1000)
  ╠═╡ =#

# ╔═╡ 2211b988-2a0b-4bb3-947d-efcf72473626
#=╠═╡
plot_rewards(continuing_linear_value_result.reward_history, 1000, 1000)
  ╠═╡ =#

# ╔═╡ c3467768-4c72-4a33-a9d3-12d94f755ee9
#=╠═╡
plot_rewards(continuing_nonlinear_value_result.reward_history, 1000, 1000)
  ╠═╡ =#

# ╔═╡ f63d3830-c236-4068-bc64-f4e9bda950fc
#=╠═╡
plot_rewards(continuing_linear_policy_result.reward_history, 100, 1000)
  ╠═╡ =#

# ╔═╡ 996286c3-6766-4ab8-9da3-c0f20cd1cb58
#=╠═╡
plot_rewards(continuing_nonlinear_policy_result.reward_history, 100, 1000)
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
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
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
SpecialFunctions = "~2.6.1"
StaticArrays = "~1.9.15"
StatsBase = "~0.34.6"
Transducers = "~0.4.85"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.7"
manifest_format = "2.0"
project_hash = "dacc54eeb0e9ec145b235f924764cfdd23758c51"

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
# ╟─6c56e883-d727-4caf-9dac-6dc2aeab102f
# ╟─005259da-6f82-4bd6-ae52-a20f8c07ef00
# ╠═8cf192df-e554-41b6-b9d0-71a142766021
# ╠═c20fad45-4033-4c60-b7db-3d8c9148026f
# ╟─f43d6616-d28e-4799-a677-b00b5811b2c1
# ╟─42537afd-7655-45a2-b18b-0759982b124c
# ╠═e932d0fd-5832-41eb-a2a3-13a89a1e8751
# ╠═742c8135-9aac-49f3-ac9a-8430aa4c2b41
# ╟─e4b0a6f5-8e6d-4b97-8d70-4fc301473be0
# ╠═5ffaab88-af31-4b70-87c0-b047354c65ba
# ╠═69a45749-bed3-4527-afb4-c869c5ca4dd6
# ╠═1e88e11d-67da-4dbd-8092-6e636bebf91f
# ╠═7c3d01db-4aca-4f95-91e5-ec7b3b2b6eb6
# ╠═2f44bb34-816a-42a0-b1d2-74493b9994a4
# ╠═6d07d39f-80e5-4a06-8ba9-7d71fea29d69
# ╠═2800e5b9-5e3e-4d34-b71b-34edebe06a3f
# ╠═41da78e2-a8a9-4cb7-8c05-68b78991ab0b
# ╠═c7e12dd0-de44-4179-9002-fcc4dcee6fbc
# ╠═a72d3703-8578-454a-b293-69bd103cd186
# ╠═d4ea0c96-4c23-485e-964b-fd52efc399b5
# ╠═09a1958b-2990-47c9-8ce7-134d7bcde5ce
# ╠═c812e9c9-b6ab-4788-a5a7-d391d270f387
# ╠═21344e9a-e0fa-4356-8947-b5310e6aa4e1
# ╠═30425ebb-2fe5-4624-a3c9-f5f332beca4d
# ╟─75b4c6da-80b7-4a34-bb15-f63ada17d5b4
# ╠═5f9b180e-9201-45d5-92dd-5c0eb9de01e7
# ╠═f578ab23-12cb-4f38-b76f-fef4189d31cc
# ╟─54807cf1-6efe-4625-a966-db6a57246c95
# ╠═5967f3f3-48cb-40ee-b886-2a86d1a3b666
# ╠═5626bdac-537b-4ff1-85dc-471881a6e496
# ╠═9ee432ea-888e-4df0-902b-58741d3f8e57
# ╠═a185fcda-eeab-4fac-97b0-d479b4cce4fd
# ╠═739db9d4-4b4a-46b7-b1f4-6176dfd467e3
# ╠═86cbffeb-cf4a-4d81-9b4e-2c3cbbdbaac5
# ╠═f9e6eed9-4fd4-47f7-b99a-05691fa901b3
# ╠═9b02733b-f769-4616-8baa-c5518812aa43
# ╠═cb475da0-614f-48e6-bc28-7898d4450ff1
# ╠═d07826fd-02a1-4cb4-991d-534e01c2a27f
# ╟─72353588-a6d5-4d18-870c-cfeacbaac669
# ╟─4e2f55f2-4639-4017-a060-03a7487eebb8
# ╠═d2387c9d-aa6e-4eda-904a-101e5fdd3cae
# ╟─d87259e7-be0d-4a55-9902-b74567e16750
# ╠═b430a02b-d4a2-4167-b4b3-28cd604f1f08
# ╠═abb0b7f7-db96-4804-852a-52b5e25049ea
# ╠═2a51bca3-0425-450b-83d4-86408197c3e8
# ╠═77a8c332-ec8c-4770-af12-c25b56e3e87f
# ╠═9bd00356-36b4-4424-b298-d78255b2642d
# ╠═ee241d93-653e-4254-a4e5-a03f98e2dd9e
# ╠═0613b3e4-62d2-46bd-8d42-d782bece25ca
# ╠═1f4ae3a7-1877-46e7-b0e5-06df88086153
# ╟─d3fbd37e-ea13-4b8f-abe8-dfb5efa19415
# ╠═2405bd06-0061-4856-a977-0302d911c760
# ╟─afdcbf2d-cd4b-42f6-9b33-7cb5c9700c61
# ╠═6ddbe04d-1bbc-4350-aac6-860212f870dc
# ╠═67f858f5-af3e-446c-bc29-07f0ecc08bce
# ╠═e2fcc077-9dbe-49fd-b1e0-d5cb43b20671
# ╠═352a4649-f395-4cf3-9266-94bf393d8a7a
# ╠═d00cdab9-e0b2-41a9-9cd1-080d844f8a57
# ╟─4c51ea7d-0ba3-45be-bb15-31103953e2b1
# ╟─bbced5a1-fe8b-402b-9dc5-6b37ebe07767
# ╠═3a938f27-b6fe-4411-8c41-5d1efaa8189c
# ╠═a5b13027-c3b6-488e-82a1-2ee3be6c63be
# ╠═0d583c27-134f-4651-89d9-63b599aa8c4f
# ╠═c68eab1e-b4f1-4fb5-8b3e-f23ad0df0be0
# ╠═98d94e3b-4ca5-4ff0-8409-9d748799931f
# ╠═33aa329f-7a8b-4264-837e-19130773315f
# ╠═8e91e2c2-a5e6-4cce-8d62-d1568bae7e08
# ╠═ad63e185-0618-476c-931e-f69b5f24d2a1
# ╟─857d4ddd-2b8c-4a45-ac72-81f5467d0e4c
# ╠═aa1b2b58-66c8-4c43-b2d1-f2de6ff982ed
# ╠═97bfe437-8cfb-4070-88e6-690647709b62
# ╠═e7653fea-304b-4958-b9e6-9ebe86b91d6f
# ╟─d9deac3f-208b-489d-b964-6d44c7e6379d
# ╠═5443cd2e-78c8-4723-9093-df2840f59a33
# ╠═eabd4d6b-ce35-41ad-845d-aa1498003814
# ╠═ddd87cf8-b424-469d-900e-5c46057aa05f
# ╠═e9be7dd3-3b76-4043-99b1-cad431310d35
# ╟─f9c9ccb4-0291-461d-8016-8f13a9dc1c5d
# ╠═aeed95c6-1f66-4087-a491-faf928fd8f4c
# ╠═2fb66afd-5889-4866-8a93-e8903881de9d
# ╠═e2161522-ec87-47ff-89a5-d683c64f75a1
# ╟─d73cd76a-61eb-47b9-abd8-769f00601743
# ╠═d7d58cdd-920e-47b4-8ef9-6b5623b85e7d
# ╠═984158d0-7fb1-4eb1-b904-3bc6011501ad
# ╠═001c295b-9fe6-4036-9fb6-337cff79687c
# ╟─cebdb010-7e8d-4fb8-bf49-418181061ad4
# ╠═64c23666-9e34-4f95-9787-2d1593725bff
# ╠═9d244394-8523-4975-af85-f70cd0cfa430
# ╠═d1440c54-faaf-4bf5-a11d-f7c3afb3437f
# ╠═93e197d7-3b7d-41a0-ae6e-2dad6c327f51
# ╠═5054ef58-74fd-4fd3-aaaa-099cc00492e2
# ╟─cd7afe0e-486c-43ed-874a-8ce20a01a8bb
# ╠═1568bed2-f17e-4a28-8b26-6d5cca22d1ea
# ╠═87365f50-017a-4773-8168-e94c6ebc0c04
# ╠═2211b988-2a0b-4bb3-947d-efcf72473626
# ╟─f9c4402f-da4e-48e8-a125-d6e5db026ae8
# ╠═ed63609b-1b7d-4075-b71f-62f1205bb122
# ╠═77466131-49bb-4ea5-9c87-423d29842b98
# ╠═c3467768-4c72-4a33-a9d3-12d94f755ee9
# ╟─1766314b-b9ee-4be0-bdb7-7aa714cc7e6d
# ╠═5638255a-7d2a-481d-b724-9c72c830ca7a
# ╠═79784c4b-2ccf-4c83-8864-6376091a5c9a
# ╠═f63d3830-c236-4068-bc64-f4e9bda950fc
# ╟─474df763-cfed-469a-9cde-832e9f52a1b1
# ╠═560473d7-19d5-44b4-a666-7007807a8288
# ╠═62a03117-f939-4e8a-9b17-dce78804641e
# ╠═996286c3-6766-4ab8-9da3-c0f20cd1cb58
# ╠═95e267e2-2c3f-4ab2-bc1b-40147a3cb94a
# ╠═8f21188f-3118-4933-a8a5-83d1c9ffd503
# ╟─487ab8b6-d9a8-4f78-a0d0-1f655450857f
# ╟─8abce157-f051-4637-bc37-f661eff08146
# ╠═8d535a89-8eab-4ec3-a144-cd54d3abdfee
# ╠═696015af-9017-4afd-a6ab-e82e2e2a5a04
# ╠═c1c283f3-97b1-435c-b2c2-81415d86679e
# ╠═f7908105-cdb4-4182-b7b1-10d1cf2ce534
# ╠═2d24fea7-915d-43df-b290-a30fdf203eb5
# ╠═b30d8e53-a6d7-4e74-9687-89c4431a37bb
# ╠═460fb9d2-4f40-47ec-9637-c6c6ec0a1b17
# ╠═9cab6940-85df-47f0-a6c6-c5f7ef2d2f10
# ╠═78589481-3163-48aa-a8d9-51d258f6a930
# ╠═69b62157-1af5-4aed-959c-b0eefebf7389
# ╠═40e18712-6715-4074-89f7-40d4751e8d20
# ╠═7d454b42-050b-4c2a-a9b2-2c445fd9fec1
# ╠═a131a509-3b5d-484e-a892-098f3518092c
# ╠═5e5051fb-e0fe-4108-b581-e7ac9b7d2198
# ╠═94f9d831-2fbc-43b7-b199-d9c547932e49
# ╟─10d1e2fe-605f-49f7-b06a-e8ce97dfba95
# ╠═673a2a41-5df6-4b45-93b3-33251c39e953
# ╠═379ae227-a73a-4693-8941-16f3a725737a
# ╠═c3595d58-9c2d-4953-a760-e05ac4b5e6b6
# ╠═5dabe906-e44c-47c5-b12c-22f829bbc2c9
# ╠═6bdf813e-cd76-46df-afe7-5210435df5e7
# ╠═90976177-23f6-465f-8646-48c77ce5c5a3
# ╠═aee066a0-e458-475e-ac15-7fa878d8ce87
# ╠═b596752c-6c9e-451e-963d-07682db396d9
# ╠═81a61620-3cfa-4a70-b934-e190fed83284
# ╠═75247e5b-f1bb-4409-aaed-16cc8c0e0538
# ╠═0a1d6fb2-f99f-469f-8946-b68841d46171
# ╠═7042bc0c-ad15-4e73-89e1-aa8028335ce0
# ╠═ef178187-d3b9-4cf2-82c8-585d5e89ac01
# ╠═ed6bd002-a0a1-49d8-a4a5-f76d3443576d
# ╠═707b1a25-0d56-486d-a187-b17b922d49c9
# ╠═102759b5-f467-4768-8e5e-78806a5b9ad6
# ╠═3624fa9e-a3bb-437f-a8db-1b1662e3ba31
# ╠═9394e249-1d18-47eb-8b3d-15c71586af53
# ╠═c182475b-c1b0-4835-8347-f0f4a831909b
# ╠═b5c0be58-c2cd-440e-9b8b-254e3990ff44
# ╟─6245ffaa-acb4-11f0-3a8d-47ce889cb225
# ╠═ddc38332-503c-4732-9432-8b998dfca6e5
# ╠═2648f295-b04d-4e2d-9d81-7d2f868f9051
# ╠═c9e47c3f-333e-49e8-be88-dda128cc8418
# ╠═3a2fa0dd-1da7-41cf-bc4a-d9dbc774dc09
# ╠═5575d394-0178-4935-b3aa-949c3ec38b45
# ╠═73e5b3d9-a675-4a1e-83fd-6c6e69ef0e9d
# ╠═65068069-1374-4344-83e7-950a894957b9
# ╠═455f956d-6c92-46e8-90d4-d62167d455cb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
