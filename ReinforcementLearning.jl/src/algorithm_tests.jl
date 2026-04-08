### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ cecc1d61-1d2e-4abc-968a-3d2acd8bff92
using PlutoDevMacros

# ╔═╡ 85566904-09a8-4375-8968-fb0738db02db
begin
	using PlutoUI, BenchmarkTools, PlutoPlotly, PlutoProfile
	TableOfContents(depth = 4)
end

# ╔═╡ 988823f4-b092-4957-b179-a11ed125f1dc
md"""
# Introduction

The purpose of this notebook is to test all of the RL algorithms in trivial environments in order to verify correctness.  In the case of gradient-based algorithms that rely on approximation, a trivial environment is one with a feature vector that reduces the problem to a tabular one.  Regarless of the type of approximation function, every state and state/action pair can be isolated and the gradient based solution can exactly match a tabular solution.  By constructing feature vectors in this manner, we can test the correctness of algorithms without any concern for whether the approximation function itself is suitable to solve the problem.
"""

# ╔═╡ b0563c95-b55f-41b4-a62e-ff2863a5c5c8
md"""
# Test Environment

We can use a gridworld example for this purpose and create versions of it for discounted episodic learning, undiscounted episodic learning, and continuing learning.  We will use a tabular environment and then convert it into a StateMDP for testing approximation algorithms.  The goal for training an agent in such an environment is for it to find the "exit" as quickly as possible.  Note also that the environment is stochastic due to wind values which change the vertical position by a random amount for non-zero values.  Continuing solutions techniques sometimes require stochastic environments to ensure an interative solution for finding the state probability distribution of a policy.
"""

# ╔═╡ accb4677-9953-4fb1-bc63-63b507140e52
md"""
## Tabular Gridworld Environments
"""

# ╔═╡ d389f899-5e8c-42f3-8df4-8555d99abb68
const wind_values = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# ╔═╡ 0f9b0dd8-1a81-4883-8408-97649a5a3b41
md"""
### Episodic with Discounting

Reward for reaching terminal state is 1 while step rewards are 0.  Discounting is required to incentivize and agent to reach the end quickly.
"""

# ╔═╡ ee5ca870-3cb3-4baf-92e3-33ba7ec823c9
md"""
### Episodic without Discounting

Step rewards are -1 with no reward for reaching the terminal state.  Because the step rewards are negative, an agent is incentivized to end the episode quickly even without discounting.
"""

# ╔═╡ f460f943-dd40-4f38-a9ba-5c87075d59b9
md"""
### Continuing

Step rewards are 0 and reaching the "goal" provides a reward of 1.  After reaching the goal the agent is reset to the start position.  As a continuing problem, the agent's incentive is to increase the average reward per step which can only occur by reaching the goal as quickly as possible from the start on a repeated basis.
"""

# ╔═╡ e403d805-26f5-4675-b3eb-e0d5c6fd7518
md"""
### Exact Solutions
"""

# ╔═╡ b7cec404-6e9e-4097-aa2b-1d11e3b94d08
md"""
## State Gridworld Environments
"""

# ╔═╡ 3c0db40d-c18d-4446-9d29-b6627fe29a26
md"""
### Random Performance

Each environment should yield the same performance as calculated by the average number of steps needed to reach the goal.  The value is around 3800.  With an ideal solution, that value is between 19 and 20 steps although the stochastic nature of the environments means this performance cna only be measured as an average since any individual episode or finite set of steps may yield a different value.
"""

# ╔═╡ b3cd6b31-dc2f-45d4-af37-c695fad2e623
md"""
### Feature Vector Setup

State aggregation is a technique of grouping states that share a value.  For our trivial feature vector construction, the groups will consist of individual states thus making the approximation solution equivalent to a tabular problem.  For testing purposes, we will represent this vector as a custom sparse type that only stores the group index for a state as well as a normal vector that explicitely stores all of the values which are 0 except for the "active" group which corresponds to the state.
"""

# ╔═╡ 6546e509-7702-4a9e-a244-8654762ca556
md"""
### Parameter Studies
"""

# ╔═╡ ff003bcf-e806-4ac7-8437-e449829be97d
md"""
#### Value Study Setup

Each parameter study contains an `update_results!` function with required and optional arguments.  Those arguments are shown below for the different types of studies:

```julia
sarsa_linear_study.update_results!(γ::T, α::T, λ::T, max_steps::Integer; max_episodes::Integer = typemax(Int64), compute_value::Function = compute_sarsa_value, ϵ::T = one(T)/10, trace_type::AbstractEligibilityTrace = AccumulatingTrace(), kwargs...) where T<:Real

sarsa_nonlinear_study.update_results!(γ::T, α::T, λ::T, max_steps::Integer, layer_size::Integer, num_layers::Integer, reslayers::Integer; max_episodes::Integer = typemax(Int64), compute_value::Function = compute_sarsa_value, ϵ::T = one(T)/10, trace_type::AbstractEligibilityTrace = AccumulatingTrace(), kwargs...) where T<:Real

monte_carlo_linear_study.update_results!(γ::T, α::T, num_episodes::Integer; compute_value::Function = compute_sarsa_value, ϵ::T = one(T)/10, max_steps::Integer = typemax(Int64), use_unfinished_episodes::Bool = true, kwargs...) where T<:Real

monte_carlo_nonlinear_study.update_results!(γ::T, α::T, num_episodes::Integer, layer_size::Integer, num_layers::Integer, reslayers::Integer; compute_value::Function = compute_sarsa_value, ϵ::T = one(T)/10, max_steps::Integer = typemax(Int64), use_unfinished_episodes::Bool = true, kwargs...) where T<:Real
```

Additional keyword arguments may be passed to the function, but the results will only be indexed by the arguments specified above.  Therefore one must ensure that the added arguments are the same accross different update calls.

"""

# ╔═╡ bae07dc9-2175-4377-bb2c-6365f0b25ecc
md"""
##### Episodic Discounted
"""

# ╔═╡ 2b4361e9-5b6a-4578-af62-46b073087b92
md"""
##### Episodic Undiscounted
"""

# ╔═╡ bb43ad9b-0133-4e2f-a667-87cbc01d4f2c
md"""
##### Continuing
"""

# ╔═╡ ac21f09b-0e5b-43fb-8f67-5eac6b46057e
md"""
#### Linear Sparse Value Study Results
"""

# ╔═╡ ae5d83e7-cebd-493b-9aed-da5cb262f926
md"""
##### Sarsa Discounted
"""

# ╔═╡ c67027a6-b6d6-4a50-9bc3-5372e61b228a
md"""
##### Sarsa Undiscounted
"""

# ╔═╡ a48eedc4-eafa-4566-abcb-9560e13077a5
md"""
##### DP Undiscounted
"""

# ╔═╡ b20f2bd9-67ea-4978-a0ee-a3ce9cf18a9f
md"""
##### DQN Undiscounted
"""

# ╔═╡ b45670e2-cf4f-49bd-94f2-9c85d2b47984
md"""
##### Sarsa Continuing
"""

# ╔═╡ cfc16980-b945-4748-a16c-985ae2266229
md"""
##### DP Continuing
"""

# ╔═╡ cd55b233-be76-49b3-9b84-2ab210697fcd
md"""
#### Linear Dense Value Study Results
"""

# ╔═╡ 37c7cb89-623a-45c4-9f5a-259cf05cf76b
md"""
##### Discounted Episodic
"""

# ╔═╡ 8aec3474-6fdd-4a78-a804-8142f0c848e0
md"""
##### Undiscounted Episodic
"""

# ╔═╡ 4aad2e81-d601-477a-9fff-db08b9ffdfbb
md"""
##### Continuing
"""

# ╔═╡ b220c9a0-5526-4166-aeb8-1c5f8b65b06a
md"""
#### Policy Study Setup
"""

# ╔═╡ 7694d5fc-1bfd-4f01-86b7-c8ac63345e2d
md"""
#### Linear Sparse Policy Study Results
"""

# ╔═╡ 28eb83ac-8e92-4e2d-95da-c5658511ea1d
md"""
##### Undiscounted Episodic
"""

# ╔═╡ 4b057115-d90b-42ad-bc6d-f74be65e7c61
md"""
##### Discounted Episodic
"""

# ╔═╡ 71074ccd-5781-486e-87d4-76174ed68e78
md"""
##### Continuing
"""

# ╔═╡ 5a4d22bf-d468-440d-9408-9a9d865bffc4
md"""
#### Linear Dense Policy Study Results
"""

# ╔═╡ db6b05cd-ccdc-42b8-bf74-e943011a8663
md"""
#### Nonlinear Sparse Policy Study Results
"""

# ╔═╡ ea79816b-5356-4c84-9efb-d4c761cf192f
# ╠═╡ disabled = true
#=╠═╡
begin
	for α_θ = 2f0 .^(-3:-1)
		for α_w in 2f0 .^(-3:-1)
			for λ_θ = 0f0:0.1f0:0.9f0
				for λ_w in 0f0:0.1f0:0.9f0
					for trace_type in [AccumulatingTrace(), DutchTrace()]
						ep_policy_sparse_studies.ac_nonlinear_study.update_results!(1f0, α_θ, α_w, λ_θ, λ_w, 10_000, 4, 2, 1; trace_type = trace_type)
					end
				end
			end
		end
	end
	display_study(ep_policy_sparse_studies.ac_nonlinear_study) |> df -> sort(df, :value; rev = true)
end
  ╠═╡ =#

# ╔═╡ f898f611-1b9d-4225-9c53-764e7f046e38
# ╠═╡ disabled = true
#=╠═╡
begin
	for α_θ = 2f0 .^(-3:-1)
		for α_w = 2f0 .^(-3:-1)
			for N = [0, 1, 2, 4, 8]
				for num_env = [1, 2, 4, 8]
					ep_policy_sparse_studies.ac_sync_nonlinear_study.update_results!(1f0, α_θ, α_w, 10_000, 4, 2, 1; num_env = num_env, nstep = N)
				end
			end
		end
	end
	display_study(ep_policy_sparse_studies.ac_sync_nonlinear_study) |> df -> sort(df, :value; rev = true)
end
  ╠═╡ =#

# ╔═╡ 15388b25-eddc-43a4-a7de-5652db3bb8c2
# ╠═╡ disabled = true
#=╠═╡
begin
	for α_θ = 2f0 .^(-3:-1)
		for α_w in 2f0 .^(-3:-1)
			for λ_θ = 0f0:0.1f0:0.9f0
				for λ_w in 0f0:0.1f0:0.9f0
					for trace_type in [AccumulatingTrace(), DutchTrace()]
						cont_policy_sparse_studies.ac_nonlinear_study.update_results!(α_θ, α_w, λ_θ, λ_w, 10_000, 4, 2, 1; trace_type = trace_type)
					end
				end
			end
		end
	end
	display_study(cont_policy_sparse_studies.ac_nonlinear_study) |> df -> sort(df, :value; rev = true)
end
  ╠═╡ =#

# ╔═╡ 83d91144-b280-46a4-b315-30e34ffa7b53
md"""
#### Nonlinear Dense Policy Study Results
"""

# ╔═╡ 1d206b10-2126-4078-8e00-613fe8e683fa
# ╠═╡ disabled = true
#=╠═╡
begin
	for α_θ = 2f0 .^(-3:-1)
		for α_w in 2f0 .^(-3:-1)
			for λ_θ = 0f0:0.1f0:0.9f0
				for λ_w in 0f0:0.1f0:0.9f0
					for trace_type in [AccumulatingTrace(), DutchTrace()]
						ep_policy_dense_studies.ac_nonlinear_study.update_results!(1f0, α_θ, α_w, λ_θ, λ_w, 10_000, 4, 2, 1; trace_type = trace_type)
					end
				end
			end
		end
	end
	display_study(ep_policy_dense_studies.ac_nonlinear_study) |> df -> sort(df, :value; rev = true)
end
  ╠═╡ =#

# ╔═╡ 4927fb9f-b8f3-45bc-a455-601ae3d15f5d
# ╠═╡ disabled = true
#=╠═╡
begin
	for α_θ = 2f0 .^(-3:-1)
		for α_w = 2f0 .^(-3:-1)
			for N = [0, 1, 2, 4, 8]
				for num_env = [1, 2, 4, 8]
					ep_policy_dense_studies.ac_sync_nonlinear_study.update_results!(1f0, α_θ, α_w, 10_000, 4, 2, 1; num_env = num_env, nstep = N)
				end
			end
		end
	end
	display_study(ep_policy_dense_studies.ac_sync_nonlinear_study) |> df -> sort(df, :value; rev = true)
end
  ╠═╡ =#

# ╔═╡ c301903e-c24f-46c9-b830-46d4a444352b
md"""
## Exhaustive Training
"""

# ╔═╡ 602aab90-5e7f-40ab-9f7d-1293916131ae
md"""
### Undiscounted Value
"""

# ╔═╡ f27efbe1-2e5e-4100-92be-b4f54b68c590
md"""
#### Linear
"""

# ╔═╡ a201adc4-70cb-4074-ab02-a6c3a8db0d98
md"""
#### Nonlinear
"""

# ╔═╡ 58d5c28a-d6f6-4f1b-9722-69d7f94e2ec1
md"""
### Discounted Value
"""

# ╔═╡ 613685e8-8a3d-4be3-b578-a6b328dd0c42
md"""
#### Linear
"""

# ╔═╡ 53b59c19-8b4f-4ce8-bbdc-0c57825954b6
md"""
#### Nonlinear
"""

# ╔═╡ 7c1536dc-b0f6-45e3-9c27-f8c54a2742b1
md"""
### Discounted Policy
"""

# ╔═╡ d4e96791-1a02-4968-a547-54b44c912bce
md"""
#### Linear
"""

# ╔═╡ d246a37e-cd59-43ca-9c3f-94b7a8f12146
md"""
#### Nonlinear
"""

# ╔═╡ 25abe591-c78b-4752-9ab1-54f6aa82d0e1
md"""
### Continuing Value
"""

# ╔═╡ 8938c848-a841-4aa3-8e7d-2a701aba0059
md"""
#### Linear
"""

# ╔═╡ 9c4cb883-8ad4-43ff-8b89-51398dbb5b65
md"""
#### Nonlinear
"""

# ╔═╡ 910d1e85-dac2-4521-bf7e-888f8b496adc
md"""
### Continuing Policy
"""

# ╔═╡ 7d4bbbbc-61cc-4d68-ab7c-4bf335f46951
md"""
#### Linear
"""

# ╔═╡ c857fbad-4ac6-4084-aeaf-09d15fe5e4fc
md"""
#### Nonlinear
"""

# ╔═╡ 3eae7c86-2a25-11f1-9289-910bb1befa22
md"""
# Dependencies
"""

# ╔═╡ 18530ced-7b01-4633-814e-b009fef375cc
@only_in_nb @fromparent import *

# ╔═╡ 879c91b7-5874-49d3-8b1e-2d3900f50540
const mdp_tab_ep_γ = make_stochastic_gridworld(;wind = wind_values)

# ╔═╡ 504a89f7-1c04-41c7-8ee3-4c62577a6b8d
const mdp_tab_ep = make_stochastic_gridworld(; termreward = 0f0, stepreward = -1f0, wind = wind_values)

# ╔═╡ 92c54927-5bed-4e94-b994-181bf2967493
const mdp_tab_cont = make_stochastic_gridworld(; continuing=true, wind = wind_values)

# ╔═╡ de403321-7c0c-49f4-b7fa-dcca37c01c13
# ╠═╡ show_logs = false
const ep_γ_exact = value_iteration_v(mdp_tab_ep_γ, 0.99f0)

# ╔═╡ 2d84c38f-62d9-48de-a641-83d3dd1a717c
log(ep_γ_exact.final_value[mdp_tab_ep_γ.initialize_state_index()]) / log(0.99f0)

# ╔═╡ 4de9aef9-c06e-4611-91e3-8ceea7ed385d
# ╠═╡ show_logs = false
const ep_exact = value_iteration_v(mdp_tab_ep, 1f0)

# ╔═╡ 8c09e264-156b-4596-bccd-3c679ddb219b
ep_exact.final_value[mdp_tab_ep.initialize_state_index()] * -1

# ╔═╡ 828b0549-7f84-42d7-a63e-7c90f9ba6099
# ╠═╡ show_logs = false
const cont_exact = value_iteration_v(mdp_tab_cont)

# ╔═╡ 79e9cc83-6a61-42b1-82ef-6ac615259fbe
(cont_exact.reward_estimates[end] |> inv) - 2

# ╔═╡ 4807bb46-c63e-4c01-973c-926d13968e48
const mdp_ep_γ = StateMDP(mdp_tab_ep_γ)

# ╔═╡ 499885eb-d829-4398-b39b-1813ca909377
const mdp_ep = StateMDP(mdp_tab_ep)

# ╔═╡ c0db1024-174e-4157-a9a1-4d26e4ad1b88
const mdp_cont = StateMDP(mdp_tab_cont)

# ╔═╡ a9ad9ee3-880a-4f37-8b72-c375926562c0
evaluate_episodic_policy_performance(mdp_ep_γ, make_random_policy(mdp_ep_γ), 10_000_000; use_steps = true) |> inv

# ╔═╡ df6f6076-ce15-4660-ae4a-f6d45360422e
evaluate_episodic_policy_performance(mdp_ep, make_random_policy(mdp_ep), 10_000_000) * -1

# ╔═╡ f53ce896-8b00-40bc-b5bc-4e0cf78e8abe
evaluate_continuing_policy_performance(mdp_cont, make_random_policy(mdp_cont), 10_000_000) |> inv

# ╔═╡ f40ed5e6-87f3-43f8-8631-d9caaf936259
const sparse_feature_setup = state_aggregation_feature_setup(mdp_ep_γ.initialize_state(), length(mdp_tab_ep_γ.states), s -> mdp_tab_ep_γ.state_index[s])

# ╔═╡ 9f19744e-f435-4003-90cb-7ab2b0840b43
sparse_feature_setup.update_feature_vector!(sparse_feature_setup.feature_vector, mdp_ep_γ.initialize_state())

# ╔═╡ acbfb5cb-1657-49fb-8bfe-4d9f999611fc
function update_dense_gridworld_feature_vector!(x::Vector{T}, s::GridworldState) where T<:Real
	i_s = mdp_tab_ep_γ.state_index[s]
	x .= zeros(T)
	x[i_s] = one(T)
end

# ╔═╡ 4a6eb093-957c-483d-a4f5-60e2c9cf5759
const dense_feature_setup = (feature_vector = zeros(Float32, length(mdp_tab_ep_γ.states)), update_feature_vector! = update_dense_gridworld_feature_vector!)

# ╔═╡ e18b5a05-f86e-45de-a2a9-e8358c01fc80
begin
	dense_feature_setup.update_feature_vector!(dense_feature_setup.feature_vector, mdp_ep_γ.initialize_state())
	dense_feature_setup.feature_vector
end

# ╔═╡ 1b54f130-0742-40e8-8750-f99b1c47afce
const ep_γ_value_sparse_studies = setup_episodic_value_parameter_studies(mdp_ep_γ, sparse_feature_setup..., true; use_steps = true, min_reward = 0f0)

# ╔═╡ 7080cda4-dfa9-4e60-9f46-54f5da7209c7
const ep_γ_value_dense_studies = setup_episodic_value_parameter_studies(mdp_ep_γ, dense_feature_setup..., true; use_steps = true, min_reward = 0f0)

# ╔═╡ 4ca3f5ba-f122-4dd4-90e5-59b3d7b2f190
const ep_value_sparse_studies = setup_episodic_value_parameter_studies(mdp_ep, sparse_feature_setup..., true)

# ╔═╡ ffe5745a-504c-4fef-accc-23e258ce9349
const ep_value_dense_studies = setup_episodic_value_parameter_studies(mdp_ep, dense_feature_setup..., true)

# ╔═╡ 6213e05f-fdcd-433d-a85d-f040561f2de0
const cont_value_sparse_studies = setup_continuing_value_parameter_studies(mdp_cont, sparse_feature_setup..., true)

# ╔═╡ 66ac57e8-f981-4483-8125-7bfde253aa13
const cont_value_dense_studies = setup_continuing_value_parameter_studies(mdp_cont, dense_feature_setup..., true)

# ╔═╡ 27765ebf-2f0a-461a-9919-c1096405e461
begin
	for α = 2f0 .^(-5:-1)
		for λ = 0f0:0.1f0:0.9f0
			for compute_value in [compute_sarsa_value, compute_expected_sarsa_value, compute_q_learning_value]
				for trace_type in [AccumulatingTrace(), DutchTrace(), ReplacingTrace()]
					ep_γ_value_sparse_studies.sarsa_linear_study.update_results!(0.99f0, α, λ, 10_000; compute_value = compute_value, trace_type = trace_type)
				end
			end
		end
	end
	display_study(ep_γ_value_sparse_studies.sarsa_linear_study) |> df -> sort(df, :value; rev = true)
end

# ╔═╡ d7fb9a39-1d0a-4111-99c5-023a7b012093
begin
	for α = 2f0 .^(-5:-1)
		for λ = 0f0:0.1f0:0.9f0
			for compute_value in [compute_sarsa_value, compute_expected_sarsa_value, compute_q_learning_value]
				for trace_type in [AccumulatingTrace(), DutchTrace(), ReplacingTrace()]
					ep_value_sparse_studies.sarsa_linear_study.update_results!(1f0, α, λ, 10_000; compute_value = compute_value, trace_type = trace_type)
				end
			end
		end
	end
	display_study(ep_value_sparse_studies.sarsa_linear_study) |> df -> sort(df, :value; rev = true)
end

# ╔═╡ 0762eab4-da0e-4bb0-ac9c-39a98ebd8e24
begin
	for α = 2f0 .^(-5:-1)
		for λ = 0f0:0.1f0:0.9f0
			for trace_type in [AccumulatingTrace(), DutchTrace(), ReplacingTrace()]
				ep_value_sparse_studies.dp_linear_study.update_results!(1f0, α, λ, 10_000; trace_type = trace_type)
			end
		end
	end
	display_study(ep_value_sparse_studies.dp_linear_study) |> df -> sort(df, :value; rev = true)
end

# ╔═╡ 0becec40-acbf-45b8-b66e-020cf04a0162
begin
	for α = 2f0 .^(-5:-1)
		for N in [0, 1, 2, 4, 8]
			for batch_size in [1, 2, 4, 8]
				for use_double_q = [true, false]
					ep_value_sparse_studies.dqn_linear_study.update_results!(1f0, α, 10_000; batch_size = batch_size, use_double_q = use_double_q, N = N)
				end
			end
		end
	end
	display_study(ep_value_sparse_studies.dqn_linear_study) |> df -> sort(df, :value; rev = true)
end

# ╔═╡ 243dfd87-d52e-423d-8ca2-da52666becdb
begin
	for α = 2f0 .^(-4:-1)
		for λ = 0f0:0.1f0:0.9f0
			for compute_value in [compute_sarsa_value, compute_expected_sarsa_value, compute_q_learning_value]
				for trace_type in [AccumulatingTrace(), DutchTrace(), ReplacingTrace()]
					for α_r̄ in [0.01f0, 0.02f0, 0.04f0]
						cont_value_sparse_studies.sarsa_linear_study.update_results!(α, λ, 10_000; compute_value = compute_value, trace_type = trace_type, α_r̄ = α_r̄)
					end
				end
			end
		end
	end
	display_study(cont_value_sparse_studies.sarsa_linear_study) |> df -> sort(df, :value; rev = true)
end

# ╔═╡ 25e7fcaa-7f68-445b-b65d-b473b550524d
begin
	for α = 2f0 .^(-4:-1)
		for λ = 0f0:0.1f0:0.9f0
			for trace_type in [AccumulatingTrace(), DutchTrace(), ReplacingTrace()]
				for α_r̄ in [0.01f0, 0.02f0, 0.04f0]
					cont_value_sparse_studies.dp_linear_study.update_results!(α, λ, 10_000; trace_type = trace_type, α_r̄ = α_r̄)
				end
			end
		end
	end
	display_study(cont_value_sparse_studies.dp_linear_study) |> df -> sort(df, :value; rev = true)
end

# ╔═╡ c1969714-2c59-45e5-ad47-44798ead30af
begin
	for α = 2f0 .^(-5:-1)
		for λ = 0f0:0.1f0:0.9f0
			for compute_value in [compute_sarsa_value, compute_expected_sarsa_value, compute_q_learning_value]
				for trace_type in [AccumulatingTrace(), DutchTrace()]
					ep_γ_value_dense_studies.sarsa_linear_study.update_results!(0.99f0, α, λ, 10_000; compute_value = compute_value, trace_type = trace_type)
				end
			end
		end
	end
	display_study(ep_γ_value_dense_studies.sarsa_linear_study) |> df -> sort(df, :value; rev = true)
end

# ╔═╡ f96b02ba-0371-4e0b-af96-824c19c070ef
begin
	for α = 2f0 .^(-5:-1)
		for λ = 0f0:0.1f0:0.9f0
			for compute_value in [compute_sarsa_value, compute_expected_sarsa_value, compute_q_learning_value]
				for trace_type in [AccumulatingTrace(), DutchTrace()]
					ep_value_dense_studies.sarsa_linear_study.update_results!(1f0, α, λ, 10_000; compute_value = compute_value, trace_type = trace_type)
				end
			end
		end
	end
	display_study(ep_value_dense_studies.sarsa_linear_study) |> df -> sort(df, :value; rev = true)
end

# ╔═╡ ef39ebd7-31cd-418a-93a6-3cbc9f09f0d0
begin
	for α = 2f0 .^(-4:-1)
		for λ = 0f0:0.1f0:0.9f0
			for trace_type in [AccumulatingTrace(), DutchTrace()]
				for α_r̄ in [0.01f0, 0.02f0, 0.04f0]
					cont_value_dense_studies.dp_linear_study.update_results!(α, λ, 10_000; trace_type = trace_type, α_r̄ = α_r̄)
				end
			end
		end
	end
	display_study(cont_value_dense_studies.dp_linear_study) |> df -> sort(df, :value; rev = true)
end

# ╔═╡ a2eb15c2-c0e8-48e8-90db-d759034d401d
const ep_policy_sparse_studies = setup_episodic_policy_parameter_studies(mdp_ep, sparse_feature_setup...)

# ╔═╡ 917c06f1-9616-4588-9429-f6e37aecca8d
const ep_policy_dense_studies = setup_episodic_policy_parameter_studies(mdp_ep, dense_feature_setup...)

# ╔═╡ d1d4f6fd-35c1-44e3-a7d9-21bf96a4a39c
const ep_γ_policy_sparse_studies = setup_episodic_policy_parameter_studies(mdp_ep_γ, sparse_feature_setup...; use_steps = true, min_reward = 0f0)

# ╔═╡ b248a5bd-9f5a-4e5d-a202-5951ba6d3d88
const ep_γ_policy_dense_studies = setup_episodic_policy_parameter_studies(mdp_ep_γ, dense_feature_setup...; use_steps = true, min_reward = 0f0)

# ╔═╡ 96d63afe-635d-4442-a041-a65f6e90de9f
const cont_policy_sparse_studies = setup_continuing_policy_parameter_studies(mdp_cont, sparse_feature_setup...)

# ╔═╡ ba016f7e-5bb6-4cde-98c3-ca5fe75f3fd1
const cont_policy_dense_studies = setup_continuing_policy_parameter_studies(mdp_cont, dense_feature_setup...)

# ╔═╡ 6b3352aa-7f2a-4a54-8e74-afae546e1caf
begin
	for α_θ = 2f0 .^(-3:-1)
		for α_w in 2f0 .^(-3:-1)
			for λ_θ = 0f0:0.1f0:0.9f0
				for λ_w in 0f0:0.1f0:0.9f0
					for trace_type in [AccumulatingTrace(), DutchTrace(), ReplacingTrace()]
						ep_policy_sparse_studies.ac_linear_study.update_results!(1f0, α_θ, α_w, λ_θ, λ_w, 10_000; trace_type = trace_type)
					end
				end
			end
		end
	end
	display_study(ep_policy_sparse_studies.ac_linear_study) |> df -> sort(df, :value; rev = true)
end

# ╔═╡ e6a745af-f45e-4fa8-b0fd-55e5ec18f3d2
begin
	for α_θ = 2f0 .^(-3:-1)
		for α_w = 2f0 .^(-3:-1)
			for N = [0, 1, 2, 4, 8]
				for num_env = [1, 2, 4, 8]
					ep_policy_sparse_studies.ac_sync_linear_study.update_results!(1f0, α_θ, α_w, 10_000; num_env = num_env, nstep = N)
				end
			end
		end
	end
	display_study(ep_policy_sparse_studies.ac_sync_linear_study) |> df -> sort(df, :value; rev = true)
end

# ╔═╡ b8622cde-3621-460c-9d9b-8b43b430c9c8
begin
	for α_θ = 2f0 .^(-3:-1)
		for α_w in 2f0 .^(-3:-1)
			for λ_θ = 0f0:0.1f0:0.9f0
				for λ_w in 0f0:0.1f0:0.9f0
					for trace_type in [AccumulatingTrace(), DutchTrace(), ReplacingTrace()]
						ep_γ_policy_sparse_studies.ac_linear_study.update_results!(0.99f0, α_θ, α_w, λ_θ, λ_w, 10_000; trace_type = trace_type)
					end
				end
			end
		end
	end
	display_study(ep_γ_policy_sparse_studies.ac_linear_study) |> df -> sort(df, :value; rev = true)
end

# ╔═╡ f0969b11-afee-4f4f-8826-016e04aaeda1
begin
	for α_θ = 2f0 .^(-3:-1)
		for α_w = 2f0 .^(-4:-1)
			for N = [0, 1, 2, 4, 8, 16]
				for num_env = [1, 2, 4, 8, 16]
					ep_γ_policy_sparse_studies.ac_sync_linear_study.update_results!(0.99f0, α_θ, α_w, 10_000; num_env = num_env, nstep = N)
				end
			end
		end
	end
	display_study(ep_γ_policy_sparse_studies.ac_sync_linear_study) |> df -> sort(df, :value; rev = true)
end

# ╔═╡ 2bf3fdb5-e7d9-4201-9643-9a745a2dd80a
begin
	for α_θ = 2f0 .^(-3:-1)
		for α_w in 2f0 .^(-3:-1)
			for λ_θ = 0f0:0.1f0:0.9f0
				for λ_w in 0f0:0.1f0:0.9f0
					for trace_type in [AccumulatingTrace(), DutchTrace(), ReplacingTrace()]
						cont_policy_sparse_studies.ac_linear_study.update_results!(α_θ, α_w, λ_θ, λ_w, 10_000; trace_type = trace_type)
					end
				end
			end
		end
	end
	display_study(cont_policy_sparse_studies.ac_linear_study) |> df -> sort(df, :value; rev = true)
end

# ╔═╡ 071ca401-c497-4540-971e-c9da093bacbb
begin
	for α_θ = 2f0 .^(-3:-1)
		for α_w in 2f0 .^(-3:-1)
			for λ_θ = 0f0:0.1f0:0.9f0
				for λ_w in 0f0:0.1f0:0.9f0
					for trace_type in [AccumulatingTrace(), DutchTrace()]
						ep_policy_dense_studies.ac_linear_study.update_results!(1f0, α_θ, α_w, λ_θ, λ_w, 10_000; trace_type = trace_type)
					end
				end
			end
		end
	end
	display_study(ep_policy_dense_studies.ac_linear_study) |> df -> sort(df, :value; rev = true)
end

# ╔═╡ b54d7da6-c32b-4029-834a-961443f50e1f
begin
	for α_θ = 2f0 .^(-3:-1)
		for α_w in 2f0 .^(-3:-1)
			for λ_θ = 0f0:0.1f0:0.9f0
				for λ_w in 0f0:0.1f0:0.9f0
					for trace_type in [AccumulatingTrace(), DutchTrace()]
						cont_policy_dense_studies.ac_linear_study.update_results!(α_θ, α_w, λ_θ, λ_w, 10_000; trace_type = trace_type)
					end
				end
			end
		end
	end
	display_study(cont_policy_dense_studies.ac_linear_study) |> df -> sort(df, :value; rev = true)
end

# ╔═╡ 6b0136f8-3fed-49a2-a84c-81edd7c6abf4
begin
	for α_θ = 2f0 .^(-3:-1)
		for α_w = 2f0 .^(-3:-1)
			for N = [0, 1, 2, 4, 8]
				for num_env = [1, 2, 4, 8]
					ep_policy_dense_studies.ac_sync_linear_study.update_results!(1f0, α_θ, α_w, 10_000; num_env = num_env, nstep = N)
				end
			end
		end
	end
	display_study(ep_policy_dense_studies.ac_sync_linear_study) |> df -> sort(df, :value; rev = true)
end

# ╔═╡ 7cee4469-57a7-4b34-a495-f3f9b13b4fef
const setup_sparse_ep = setup_episodic_value_linear_training(mdp_ep, sparse_feature_setup...)

# ╔═╡ 70f8f7d2-29ca-4a66-8757-ad48b81ceb0d
setup_sparse_ep.train_rate_decay(1f0, 0.5f0, 0.7f0, 10_000)

# ╔═╡ ed8cf3f5-b5bd-48ac-96c7-6d81d3c46ff0
setup_sparse_ep.train_rate_decay(1f0, 0.5f0, 0.7f0, 10_000; use_dp = true)

# ╔═╡ 13c18d3e-0e1b-461f-be47-1e44bec1c570
setup_sparse_ep.train_dqn_rate_decay(1f0, 0.5f0, 10_000; batch_size = 128, N = 2)

# ╔═╡ 39a1de79-e797-4d4f-994b-c4a23471fd69
const setup_dense_ep = setup_episodic_value_linear_training(mdp_ep, dense_feature_setup...)

# ╔═╡ 9c918a8d-ba9c-4c9a-a0b0-76ae4bba50ef
setup_dense_ep.train_rate_decay(1f0, 0.5f0, 0.7f0, 10_000)

# ╔═╡ a5e9aab9-bd15-43d4-a4ac-6b3adf621aea
setup_dense_ep.train_rate_decay(1f0, 0.5f0, 0.7f0, 10_000; use_dp = true)

# ╔═╡ 5241bf09-4861-466c-8fd6-415199689b3d
setup_dense_ep.train_dqn_rate_decay(1f0, 0.5f0, 10_000; batch_size = 128, N = 2)

# ╔═╡ e0045b54-ba5e-4ab3-9914-a4c297fb91e0
const setup_sparse_ep_nonlinear = setup_episodic_value_nonlinear_training(mdp_ep, sparse_feature_setup...)

# ╔═╡ 3c31bc74-9a7a-426d-8b58-fcfecbe506be
setup_sparse_ep_nonlinear.train_rate_decay([4, 4], 1, 1f0, 0.001f0, 0.9f0, 100_000; new_params = true)

# ╔═╡ 79e9a002-49bc-4df8-b730-00e287f1e6b3
setup_sparse_ep_nonlinear.train_rate_decay([4, 4], 1, 1f0, 0.01f0, 0.5f0, 100_000; new_params = true, use_dp = true)

# ╔═╡ 64289b91-4e2f-44be-9d86-21fb3c1e153f
setup_sparse_ep_nonlinear.train_dqn_rate_decay([4, 4], 1, 1f0, 0.1f0, 100_000; new_params = true, N = 10, batch_size = 16)

# ╔═╡ 9070a393-3405-4c5f-b78a-aab8040ac897
const setup_dense_ep_nonlinear = setup_episodic_value_nonlinear_training(mdp_ep, dense_feature_setup...)

# ╔═╡ 1885ff61-ba55-45b6-a4fe-1d00fd8b5486
setup_dense_ep_nonlinear.train_rate_decay([4, 4], 1, 1f0, 0.001f0, 0.9f0, 100_000; new_params = true)

# ╔═╡ b3acf3da-efd1-439c-8fdb-c1eacffbc737
setup_dense_ep_nonlinear.train_rate_decay([4, 4], 1, 1f0, 0.1f0, 0.25f0, 100_000; new_params = true, use_dp = true)

# ╔═╡ 400f65b7-4e69-4186-8a93-eaeb13fce3a9
setup_dense_ep_nonlinear.train_dqn_rate_decay([4, 4], 1, 1f0, 0.1f0, 100_000; new_params = true, N = 10, batch_size = 16)

# ╔═╡ e98ef437-71b7-4c84-b802-f09af460a6f2
const setup_sparse_ep_γ = setup_episodic_value_linear_training(mdp_ep_γ, sparse_feature_setup...; min_reward = 0f0)

# ╔═╡ 7d8e323b-3eb8-4f76-be41-10064da72919
setup_sparse_ep_γ.train_rate_decay(0.99f0, 0.5f0, 0.7f0, 10_000; new_params = true, use_steps = true, min_value = 0f0)

# ╔═╡ 62550dc1-1fe9-4488-8d08-f780e707855d
setup_sparse_ep_γ.train_rate_decay(0.99f0, 0.25f0, 0.7f0, 10_000; use_dp = true, new_params = true, use_steps = true)

# ╔═╡ 0845b182-9568-4368-bd6d-c3c6440768cd
setup_sparse_ep_γ.train_dqn_rate_decay(0.99f0, 0.5f0, 10_000; new_params = true, use_steps = true, batch_size = 16, N = 1)

# ╔═╡ f67bdbb9-1a5c-4fe6-a556-3f54120721bc
const setup_sparse_nonlinear_ep_γ = setup_episodic_value_nonlinear_training(mdp_ep_γ, sparse_feature_setup...; min_reward = 0f0)

# ╔═╡ 12c70fc7-0ed4-489e-badc-5fdae0100678
setup_sparse_nonlinear_ep_γ.train_rate_decay([4, 4], 1, 0.99f0, 0.5f0, 0.7f0, 100_000; new_params = true, use_steps = true, min_value = 0f0)

# ╔═╡ 5321300d-518e-46c9-a517-ae6e94ca52b0
setup_sparse_nonlinear_ep_γ.train_rate_decay([4, 4], 1, 0.99f0, 0.5f0, 0.7f0, 100_000; new_params = true, use_dp=true, use_steps = true, min_value = 0f0)

# ╔═╡ 6bf176c3-da2e-40ba-9bf4-0f6b0a3ec777
setup_sparse_nonlinear_ep_γ.train_dqn_rate_decay([4, 4], 1, 0.99f0, 0.05f0, 100_000; new_params = true, batch_size = 16, N = 10, use_steps = true, min_value = 0f0)

# ╔═╡ 44d7e0b0-2336-4a3c-be4c-f0be9c27b3a9
setup_sparse_nonlinear_ep_γ.train_dqn_rate_decay([4, 4], 1, 0.99f0, 0.1f0, 100_000; new_params = true, batch_size = 16, N = 10, use_steps = true, min_value = 0f0, use_double_q = true)

# ╔═╡ fdecf521-e735-4c23-a232-f90e24f3b88c
const setup_policy_linear_sparse_ep_γ = setup_episodic_policy_linear_training(mdp_ep_γ, sparse_feature_setup...; min_reward = 0f0)

# ╔═╡ edb4a3e8-20a7-43ca-b919-a01d8b76ccab
setup_policy_linear_sparse_ep_γ.train_rate_decay(0.99f0, 0.5f0, 0.5f0, 0.5f0, 0.5f0, 10_000; use_steps = true)

# ╔═╡ 2593328b-717e-40a7-a641-4f1670bc8783
setup_policy_linear_sparse_ep_γ.sync_train_rate_decay(0.99f0, 0.1f0, 0.1f0, 100_000; use_steps = true, num_env = 8, new_params = true, N = 10)

# ╔═╡ e38bb29f-8dc8-41e5-b5b1-ea820ef0d8b9
const setup_policy_nonlinear_sparse_ep_γ = setup_episodic_policy_nonlinear_training(mdp_ep_γ, sparse_feature_setup...; min_reward = 0f0)

# ╔═╡ b5cdc14e-c112-443e-a327-18d4e38ca169
setup_policy_nonlinear_sparse_ep_γ.train_rate_decay([4, 4], 1, 0.99f0, 0.1f0, 0.1f0, 0.99f0, 0.25f0, 100_000; new_params = true, use_steps = true)

# ╔═╡ 6196ff81-a052-4870-8019-fdadf6282baa
setup_policy_nonlinear_sparse_ep_γ.sync_train_rate_decay([4, 4], 1, 0.99f0, 0.2f0, 0.05f0, 100_000; use_steps = true, num_env = 16, new_params = true, N = 10)

# ╔═╡ fa566c0c-5639-4131-96b3-b16481e5a6d9
const setup_value_linear_sparse_cont = setup_continuing_value_linear_training(mdp_cont, sparse_feature_setup...)

# ╔═╡ e627266a-4c30-44da-bcdb-5a83a8758f16
setup_value_linear_sparse_cont.train_rate_decay(0.01f0, 0.5f0, 100_000; new_params = true)

# ╔═╡ 90dd105d-3bc4-4040-a90d-dd7cd72c4698
setup_value_linear_sparse_cont.train_rate_decay(0.01f0, 0.5f0, 100_000; new_params = true, use_dp = true)

# ╔═╡ 2da971a8-bf39-41e4-90f6-498c7adecd4f
const setup_value_nonlinear_sparse_cont = setup_continuing_value_nonlinear_training(mdp_cont, sparse_feature_setup...)

# ╔═╡ 4fea1551-6fe2-44c0-a58b-fc6f5cc7fb96
setup_value_nonlinear_sparse_cont.train_rate_decay([4, 4], 1, 0.2f0, 0.8f0, 100_000; new_params = true, use_dp = true)

# ╔═╡ 96317044-89d5-47c6-9e77-e8d24971db8e
setup_value_nonlinear_sparse_cont.train_rate_decay([4, 4], 1, 0.4f0, 0.5f0, 100_000; new_params = true, compute_value = compute_expected_sarsa_value)

# ╔═╡ 45fb51fb-ff12-4926-88f1-010c9d8ca907
const setup_policy_linear_sparse_cont = setup_continuing_policy_linear_training(mdp_cont, sparse_feature_setup...)

# ╔═╡ 52e00038-5fc4-4b89-8ed0-151da1d5dcc8
setup_policy_linear_sparse_cont.train_rate_decay(0.1f0, 0.1f0, 0.5f0, 0.5f0, 100_000; new_params = true)

# ╔═╡ 4da26810-e075-4a78-a01c-2d919044845d
const setup_policy_linear_dense_cont = setup_continuing_policy_linear_training(mdp_cont, dense_feature_setup...)

# ╔═╡ 367925ac-e192-4092-8ad0-1d68e2c3a8e9
setup_policy_linear_dense_cont.train_rate_decay(0.1f0, 0.1f0, 0.5f0, 0.5f0, 100_000; new_params = true)

# ╔═╡ 256449e7-e84c-4476-aab4-128211955ce3
const setup_policy_nonlinear_sparse_cont = setup_continuing_policy_nonlinear_training(mdp_cont, sparse_feature_setup...)

# ╔═╡ b08b052f-0fea-42bb-bc52-be1ca61cb2a8
setup_policy_nonlinear_sparse_cont.train_rate_decay([4, 4], 1, 0.1f0, 0.1f0, 0.95f0, 0.5f0, 100_000; new_params = true)

# ╔═╡ e021457d-4b5e-4e20-b90e-cc884297586d
const setup_policy_nonlinear_dense_cont = setup_continuing_policy_nonlinear_training(mdp_cont, dense_feature_setup...)

# ╔═╡ 217b1ef8-1fc0-4471-8840-fc5ac4cc90c3
setup_policy_nonlinear_dense_cont.train_rate_decay([4, 4], 1, 0.1f0, 0.1f0, 0.95f0, 0.5f0, 100_000; new_params = true)

# ╔═╡ a87bdad2-723d-4dee-a146-30eaffc3d201
html"""
<style>
	main {
		margin: 0 auto;
		max-width: min(1600px, 90%);
		padding-left: max(10px, 5%);
		padding-right: max(200px, 5%);
		font-size: max(10px, min(24px, 2vw));
	}
</style>
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
BenchmarkTools = "~1.6.3"
PlutoDevMacros = "~0.9.2"
PlutoPlotly = "~0.6.5"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.80"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "805f66821b001a6450e0a647a50cf24b833d682d"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "7fecfb1123b8d0232218e2da0c213004ff15358d"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.3"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "b7231a755812695b8046e8471ddc34c8268cbad5"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.0"

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

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

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
version = "1.3.0+1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6522cfb3b8fe97bec632252263057996cbd3de20"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.18.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

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

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

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
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "3d3b79166e2a0afcf875df20db110af91ad3ab61"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.11"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

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
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

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
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

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
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "6256ab3ee24ef079b3afa310593817e069925eeb"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.23"

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
git-tree-sha1 = "709c36a806ec0af91840184f3052bb3c6cc60915"
uuid = "a0499f29-c39b-4c5c-807c-88074221b949"
version = "0.9.2"

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
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "fbc875044d82c113a9dee6fc14e16cf01fd48872"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.80"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProfileCanvas]]
deps = ["FlameGraphs", "JSON", "Pkg", "Profile", "REPL"]
git-tree-sha1 = "41fd9086187b8643feda56b996eef7a3cc7f4699"
uuid = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
version = "0.1.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "ac4b837d89a58c848e85e698e2a2514e9d59d8f6"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

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

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

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
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╟─988823f4-b092-4957-b179-a11ed125f1dc
# ╟─b0563c95-b55f-41b4-a62e-ff2863a5c5c8
# ╟─accb4677-9953-4fb1-bc63-63b507140e52
# ╠═d389f899-5e8c-42f3-8df4-8555d99abb68
# ╟─0f9b0dd8-1a81-4883-8408-97649a5a3b41
# ╠═879c91b7-5874-49d3-8b1e-2d3900f50540
# ╟─ee5ca870-3cb3-4baf-92e3-33ba7ec823c9
# ╠═504a89f7-1c04-41c7-8ee3-4c62577a6b8d
# ╟─f460f943-dd40-4f38-a9ba-5c87075d59b9
# ╠═92c54927-5bed-4e94-b994-181bf2967493
# ╟─e403d805-26f5-4675-b3eb-e0d5c6fd7518
# ╠═de403321-7c0c-49f4-b7fa-dcca37c01c13
# ╠═2d84c38f-62d9-48de-a641-83d3dd1a717c
# ╠═4de9aef9-c06e-4611-91e3-8ceea7ed385d
# ╠═8c09e264-156b-4596-bccd-3c679ddb219b
# ╠═828b0549-7f84-42d7-a63e-7c90f9ba6099
# ╠═79e9cc83-6a61-42b1-82ef-6ac615259fbe
# ╟─b7cec404-6e9e-4097-aa2b-1d11e3b94d08
# ╠═4807bb46-c63e-4c01-973c-926d13968e48
# ╠═499885eb-d829-4398-b39b-1813ca909377
# ╠═c0db1024-174e-4157-a9a1-4d26e4ad1b88
# ╟─3c0db40d-c18d-4446-9d29-b6627fe29a26
# ╠═a9ad9ee3-880a-4f37-8b72-c375926562c0
# ╠═df6f6076-ce15-4660-ae4a-f6d45360422e
# ╠═f53ce896-8b00-40bc-b5bc-4e0cf78e8abe
# ╟─b3cd6b31-dc2f-45d4-af37-c695fad2e623
# ╠═f40ed5e6-87f3-43f8-8631-d9caaf936259
# ╠═9f19744e-f435-4003-90cb-7ab2b0840b43
# ╠═acbfb5cb-1657-49fb-8bfe-4d9f999611fc
# ╠═4a6eb093-957c-483d-a4f5-60e2c9cf5759
# ╠═e18b5a05-f86e-45de-a2a9-e8358c01fc80
# ╟─6546e509-7702-4a9e-a244-8654762ca556
# ╟─ff003bcf-e806-4ac7-8437-e449829be97d
# ╟─bae07dc9-2175-4377-bb2c-6365f0b25ecc
# ╠═1b54f130-0742-40e8-8750-f99b1c47afce
# ╠═7080cda4-dfa9-4e60-9f46-54f5da7209c7
# ╟─2b4361e9-5b6a-4578-af62-46b073087b92
# ╠═4ca3f5ba-f122-4dd4-90e5-59b3d7b2f190
# ╠═ffe5745a-504c-4fef-accc-23e258ce9349
# ╟─bb43ad9b-0133-4e2f-a667-87cbc01d4f2c
# ╠═6213e05f-fdcd-433d-a85d-f040561f2de0
# ╠═66ac57e8-f981-4483-8125-7bfde253aa13
# ╟─ac21f09b-0e5b-43fb-8f67-5eac6b46057e
# ╟─ae5d83e7-cebd-493b-9aed-da5cb262f926
# ╠═27765ebf-2f0a-461a-9919-c1096405e461
# ╟─c67027a6-b6d6-4a50-9bc3-5372e61b228a
# ╠═d7fb9a39-1d0a-4111-99c5-023a7b012093
# ╟─a48eedc4-eafa-4566-abcb-9560e13077a5
# ╠═0762eab4-da0e-4bb0-ac9c-39a98ebd8e24
# ╟─b20f2bd9-67ea-4978-a0ee-a3ce9cf18a9f
# ╠═0becec40-acbf-45b8-b66e-020cf04a0162
# ╟─b45670e2-cf4f-49bd-94f2-9c85d2b47984
# ╠═243dfd87-d52e-423d-8ca2-da52666becdb
# ╟─cfc16980-b945-4748-a16c-985ae2266229
# ╠═25e7fcaa-7f68-445b-b65d-b473b550524d
# ╟─cd55b233-be76-49b3-9b84-2ab210697fcd
# ╟─37c7cb89-623a-45c4-9f5a-259cf05cf76b
# ╠═c1969714-2c59-45e5-ad47-44798ead30af
# ╟─8aec3474-6fdd-4a78-a804-8142f0c848e0
# ╠═f96b02ba-0371-4e0b-af96-824c19c070ef
# ╟─4aad2e81-d601-477a-9fff-db08b9ffdfbb
# ╠═ef39ebd7-31cd-418a-93a6-3cbc9f09f0d0
# ╟─b220c9a0-5526-4166-aeb8-1c5f8b65b06a
# ╠═a2eb15c2-c0e8-48e8-90db-d759034d401d
# ╠═917c06f1-9616-4588-9429-f6e37aecca8d
# ╠═d1d4f6fd-35c1-44e3-a7d9-21bf96a4a39c
# ╠═b248a5bd-9f5a-4e5d-a202-5951ba6d3d88
# ╠═96d63afe-635d-4442-a041-a65f6e90de9f
# ╠═ba016f7e-5bb6-4cde-98c3-ca5fe75f3fd1
# ╟─7694d5fc-1bfd-4f01-86b7-c8ac63345e2d
# ╟─28eb83ac-8e92-4e2d-95da-c5658511ea1d
# ╠═6b3352aa-7f2a-4a54-8e74-afae546e1caf
# ╠═e6a745af-f45e-4fa8-b0fd-55e5ec18f3d2
# ╟─4b057115-d90b-42ad-bc6d-f74be65e7c61
# ╠═b8622cde-3621-460c-9d9b-8b43b430c9c8
# ╠═f0969b11-afee-4f4f-8826-016e04aaeda1
# ╟─71074ccd-5781-486e-87d4-76174ed68e78
# ╠═2bf3fdb5-e7d9-4201-9643-9a745a2dd80a
# ╟─5a4d22bf-d468-440d-9408-9a9d865bffc4
# ╠═071ca401-c497-4540-971e-c9da093bacbb
# ╠═b54d7da6-c32b-4029-834a-961443f50e1f
# ╠═6b0136f8-3fed-49a2-a84c-81edd7c6abf4
# ╟─db6b05cd-ccdc-42b8-bf74-e943011a8663
# ╠═ea79816b-5356-4c84-9efb-d4c761cf192f
# ╠═f898f611-1b9d-4225-9c53-764e7f046e38
# ╠═15388b25-eddc-43a4-a7de-5652db3bb8c2
# ╟─83d91144-b280-46a4-b315-30e34ffa7b53
# ╠═1d206b10-2126-4078-8e00-613fe8e683fa
# ╠═4927fb9f-b8f3-45bc-a455-601ae3d15f5d
# ╟─c301903e-c24f-46c9-b830-46d4a444352b
# ╟─602aab90-5e7f-40ab-9f7d-1293916131ae
# ╟─f27efbe1-2e5e-4100-92be-b4f54b68c590
# ╠═7cee4469-57a7-4b34-a495-f3f9b13b4fef
# ╠═70f8f7d2-29ca-4a66-8757-ad48b81ceb0d
# ╠═ed8cf3f5-b5bd-48ac-96c7-6d81d3c46ff0
# ╠═13c18d3e-0e1b-461f-be47-1e44bec1c570
# ╠═39a1de79-e797-4d4f-994b-c4a23471fd69
# ╠═9c918a8d-ba9c-4c9a-a0b0-76ae4bba50ef
# ╠═a5e9aab9-bd15-43d4-a4ac-6b3adf621aea
# ╠═5241bf09-4861-466c-8fd6-415199689b3d
# ╟─a201adc4-70cb-4074-ab02-a6c3a8db0d98
# ╠═e0045b54-ba5e-4ab3-9914-a4c297fb91e0
# ╠═3c31bc74-9a7a-426d-8b58-fcfecbe506be
# ╠═79e9a002-49bc-4df8-b730-00e287f1e6b3
# ╠═64289b91-4e2f-44be-9d86-21fb3c1e153f
# ╠═9070a393-3405-4c5f-b78a-aab8040ac897
# ╠═1885ff61-ba55-45b6-a4fe-1d00fd8b5486
# ╠═b3acf3da-efd1-439c-8fdb-c1eacffbc737
# ╠═400f65b7-4e69-4186-8a93-eaeb13fce3a9
# ╟─58d5c28a-d6f6-4f1b-9722-69d7f94e2ec1
# ╟─613685e8-8a3d-4be3-b578-a6b328dd0c42
# ╠═e98ef437-71b7-4c84-b802-f09af460a6f2
# ╠═7d8e323b-3eb8-4f76-be41-10064da72919
# ╠═62550dc1-1fe9-4488-8d08-f780e707855d
# ╠═0845b182-9568-4368-bd6d-c3c6440768cd
# ╟─53b59c19-8b4f-4ce8-bbdc-0c57825954b6
# ╠═f67bdbb9-1a5c-4fe6-a556-3f54120721bc
# ╠═12c70fc7-0ed4-489e-badc-5fdae0100678
# ╠═5321300d-518e-46c9-a517-ae6e94ca52b0
# ╠═6bf176c3-da2e-40ba-9bf4-0f6b0a3ec777
# ╠═44d7e0b0-2336-4a3c-be4c-f0be9c27b3a9
# ╟─7c1536dc-b0f6-45e3-9c27-f8c54a2742b1
# ╟─d4e96791-1a02-4968-a547-54b44c912bce
# ╠═fdecf521-e735-4c23-a232-f90e24f3b88c
# ╠═edb4a3e8-20a7-43ca-b919-a01d8b76ccab
# ╠═2593328b-717e-40a7-a641-4f1670bc8783
# ╟─d246a37e-cd59-43ca-9c3f-94b7a8f12146
# ╠═e38bb29f-8dc8-41e5-b5b1-ea820ef0d8b9
# ╠═b5cdc14e-c112-443e-a327-18d4e38ca169
# ╠═6196ff81-a052-4870-8019-fdadf6282baa
# ╟─25abe591-c78b-4752-9ab1-54f6aa82d0e1
# ╟─8938c848-a841-4aa3-8e7d-2a701aba0059
# ╠═fa566c0c-5639-4131-96b3-b16481e5a6d9
# ╠═e627266a-4c30-44da-bcdb-5a83a8758f16
# ╠═90dd105d-3bc4-4040-a90d-dd7cd72c4698
# ╟─9c4cb883-8ad4-43ff-8b89-51398dbb5b65
# ╠═2da971a8-bf39-41e4-90f6-498c7adecd4f
# ╠═96317044-89d5-47c6-9e77-e8d24971db8e
# ╠═4fea1551-6fe2-44c0-a58b-fc6f5cc7fb96
# ╟─910d1e85-dac2-4521-bf7e-888f8b496adc
# ╟─7d4bbbbc-61cc-4d68-ab7c-4bf335f46951
# ╠═45fb51fb-ff12-4926-88f1-010c9d8ca907
# ╠═52e00038-5fc4-4b89-8ed0-151da1d5dcc8
# ╠═4da26810-e075-4a78-a01c-2d919044845d
# ╠═367925ac-e192-4092-8ad0-1d68e2c3a8e9
# ╟─c857fbad-4ac6-4084-aeaf-09d15fe5e4fc
# ╠═256449e7-e84c-4476-aab4-128211955ce3
# ╠═b08b052f-0fea-42bb-bc52-be1ca61cb2a8
# ╠═e021457d-4b5e-4e20-b90e-cc884297586d
# ╠═217b1ef8-1fc0-4471-8840-fc5ac4cc90c3
# ╟─3eae7c86-2a25-11f1-9289-910bb1befa22
# ╠═cecc1d61-1d2e-4abc-968a-3d2acd8bff92
# ╠═18530ced-7b01-4633-814e-b009fef375cc
# ╠═85566904-09a8-4375-8968-fb0738db02db
# ╠═a87bdad2-723d-4dee-a146-30eaffc3d201
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
