### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ ba851c42-8bb0-11f1-94f1-f50977498160
using PlutoDevMacros

# ╔═╡ db92cc68-04d0-4957-8f1e-bd3dc21d69fa
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI, PlutoPlotly, ProfileCanvas, BenchmarkTools, LaTeXStrings, HypertextLiteral
	TableOfContents()
end
  ╠═╡ =#

# ╔═╡ bf7a594f-7aef-4f8c-ad93-8f254c2dad3f
md"""
# Introduction

The goal of this notebook is to check for performance bottlenecks such as runtime dispatch and garbage collection that is unecessary in the `ReinforcementLearning.jl` package.  I will use the profile view and benchmark tools timing to check the main functionality of the package.
"""

# ╔═╡ 27b818f8-b6cb-4d26-b0df-a9a7d277351c
md"""
# Profiling Utilities
"""

# ╔═╡ 7255a15c-9e00-48a5-a83c-ef88f127d52e
function repeated_run(f::Function, n::Integer, args::Vararg{Any, M}; kwargs...) where M
	for _ in 1:n
		f(args...; kwargs...)
	end
	return nothing
end

# ╔═╡ 4041b1d0-7765-438f-b5ce-e7a5bcce3045
md"""
# Tabular Methods
"""

# ╔═╡ 69264750-07fd-4fb6-bb10-7f968a56f772
md"""
## Test Environments
"""

# ╔═╡ 0e68d19c-c077-47e9-b396-e8395b4cebc6
const wind_values = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# ╔═╡ 704f0549-d227-4a3a-b78e-d56c6d16e613
md"""
## Exact Solution Methods
"""

# ╔═╡ 2be43ef3-c3f5-4d02-8e18-9a5f55d05bee
md"""
### Policy Evaluation
"""

# ╔═╡ a2815932-6ce4-479a-a152-74ed6d38f2f2
md"""
### Policy Iteration
"""

# ╔═╡ 1f1aa1e5-b10a-4831-8c5e-699a10171617
md"""
### Value Iteration
"""

# ╔═╡ 8200bb1c-787c-473c-a095-5a52d0bd0b11
md"""
## Monte Carlo Solution Methods
"""

# ╔═╡ 3bfaa827-923a-405e-8f12-ca37ceab590a
md"""
### Policy Prediction
"""

# ╔═╡ dc0d2f22-2646-4249-99f8-ae5d00506527
md"""
### Control
"""

# ╔═╡ 24975ce1-40db-4fa2-94e3-ab09ff7fe86e
md"""
## Temporal Difference Solution Methods
"""

# ╔═╡ f4c7fc77-0900-44b1-aad8-b1c78bcb56f0
md"""
### Policy Prediction
"""

# ╔═╡ f5a2747a-29f1-41e5-8d99-c5c51050721d
md"""
### Sarsa
"""

# ╔═╡ d97c9506-aea1-43b4-a282-81418df894ef
md"""
## Planning Methods
"""

# ╔═╡ d7052ba5-f0ef-4350-befc-0b2458af8c60
md"""
# Non-Tabular Methods
"""

# ╔═╡ 04b6f612-c5aa-49a0-ac83-702f21c17774
md"""
## Test Environments
"""

# ╔═╡ 3076c786-db6d-4573-99fe-35540ee171c5
md"""
## Feature Vector Setup
"""

# ╔═╡ b3da9525-bf8e-4761-a518-ce0e7292aaee
md"""
## On-policy Prediction with Approximation
"""

# ╔═╡ 740834f5-5fa7-4a8f-9c8b-77e1bece4f0e
md"""
## On-policy Control with Approximation
"""

# ╔═╡ c63c0a29-5604-4d5f-a9e8-2733ea8c5865
md"""
### Semi-gradient Sarsa
"""

# ╔═╡ 9a84b6e7-aaf4-4c84-ba03-c6e9d1fecf7f
md"""
### Semi-gradient Double Sarsa
"""

# ╔═╡ 807e0eef-2811-4288-8a62-65119c35f093
md"""
### Semi-gradient DP
"""

# ╔═╡ b79ca0ac-6386-4f8f-b74a-811e53229eea
md"""
### Semi-gradient Differential Sarsa
"""

# ╔═╡ b509a680-e2d2-4a37-91ca-4cf7d4b43970
md"""
## Eligibility Traces
"""

# ╔═╡ dfe5108d-1488-4e4e-9a8f-902951af993a
md"""
### Semi-gradient TD-λ
"""

# ╔═╡ d74c0f55-023c-4939-aecc-4d6467001dce
md"""
#### Binary Episodic
"""

# ╔═╡ ebc7fafa-9e31-4f06-8a56-762472f878ee
md"""
#### State Aggregation Episodic
"""

# ╔═╡ 01ecf555-af75-4c56-b667-4754ab588b9f
md"""
#### Dense Features Episodic
"""

# ╔═╡ 239f8a7a-1cf6-42ad-b8fe-3f77896c5240
md"""
#### Sparse Features Continuing
"""

# ╔═╡ 5f4307fe-0c50-4488-a879-89b8d77aa846
md"""
#### Dense Features Continuing
"""

# ╔═╡ a685c2b2-9cf5-42aa-8cf8-ed7c306c500a
md"""
### True-online TDλ
"""

# ╔═╡ 38741ee4-928b-45e3-b845-964148a5448c
md"""
### Sarsa λ
"""

# ╔═╡ edebeb0e-e10a-43cf-932e-b304438583e3
md"""
#### Binary Episodic
"""

# ╔═╡ 30ed68c0-dd81-4b0f-87ef-398b4c7a160d
md"""
#### State Aggregation Episodic
"""

# ╔═╡ 9c5e2c3c-3c9a-47c5-86e5-daea279cc3d9
md"""
#### Dense Features Episodic
"""

# ╔═╡ aa2dd41f-2915-4f02-9fc6-f974a5b0cb5a
md"""
#### Binary Features Continuing
"""

# ╔═╡ 83fbc478-a212-4a94-86a9-3a2d77798e09
md"""
### DP λ
"""

# ╔═╡ 46dbcd43-8edc-45fe-8924-407926078316
md"""
#### Binary Features Episodic
"""

# ╔═╡ 5a52b405-6308-4d4f-afcb-10fa9a7bdd84
md"""
#### Dense Features Episodic
"""

# ╔═╡ 7b9c5a5a-7db3-4265-ba00-b608f00ae969
md"""
#### Binary Features Continuing
"""

# ╔═╡ d79e92bb-3282-4912-a7cb-f7ef72bada4e
md"""
### True Online Sarsa λ
"""

# ╔═╡ 7fd6d59a-8768-4f45-a58b-681490fbf111
md"""
### True Online DP λ
"""

# ╔═╡ 6f453370-146c-4e3f-be21-90ca1d925976
md"""
## Policy Gradient Methods
"""

# ╔═╡ 1bb636c9-ae1f-4c39-9403-75e954103ed9
md"""
### REINFORCE
"""

# ╔═╡ 7d1e97e3-7e21-4dd6-b6bf-211de7933a7e
md"""
### One-step Actor Critic
"""

# ╔═╡ ddb83cc4-81f2-4890-b6c6-1d26b2faa350
md"""
#### Binary Features Episodic
"""

# ╔═╡ 83b63252-48bd-45e9-86f9-672b43d2202e
md"""
#### Dense Features Episodic
"""

# ╔═╡ 7fbf7a3f-775e-47a1-a9dd-4b41090c229a
md"""
#### Binary Features Continuing
"""

# ╔═╡ b0fdf1a0-340e-440d-9859-c6d32531e4ac
md"""
#### Dense Features Continuing
"""

# ╔═╡ ff0d89e2-3a3c-41d4-8c60-d27af690313d
md"""
### Actor Critic with Eligibility Traces
"""

# ╔═╡ e5887db2-8af2-4e39-99a1-e6944b229f18
md"""
#### Binary Features Episodic
"""

# ╔═╡ 450a9920-def5-4bf5-89f9-df302a7ab1c3
md"""
#### Binary Features Continuing
"""

# ╔═╡ a964d032-aa9b-4559-830b-40d2096860be
md"""
#### Dense Features Episodic
"""

# ╔═╡ e3a33387-ae77-4e59-8f68-bfdedd1f86cf
md"""
#### Dense Features Continuing
"""

# ╔═╡ e79916f5-5f11-4e5f-b76c-3bc8bc8eebc4
md"""
## Batch and Parallel Learning Methods
"""

# ╔═╡ 3f268a8f-d67b-4412-a071-094247dd54c8
md"""
### DQN
"""

# ╔═╡ 2b67c1d2-7642-4191-813e-203fecbbda47
md"""
#### Binary Feature Vectors
"""

# ╔═╡ e3edbd2e-92cc-4e77-8cc5-b546823b18d8
md"""
#### State Aggregation Feature Vector
"""

# ╔═╡ ba31075c-945b-4963-91b0-a5ad223c327f
md"""
#### Dense Feature Vector
"""

# ╔═╡ d53302f5-8f63-4e3c-b3de-5ade72c4b2b9
md"""
### Synchronous Actor Critic
"""

# ╔═╡ 721ded12-9bc1-43e8-82ec-c689d899d13c
md"""
#### Binary Features
"""

# ╔═╡ e616a1fe-c5b3-408e-9263-aadf18cbaf84
md"""
#### Dense Features
"""

# ╔═╡ 28c9bd5d-2a46-4df4-9b80-3d6e4c6f2530
md"""
# Dependencies
"""

# ╔═╡ 9f51bb2a-574f-4a89-bc8c-426ac2961f7c
@fromparent import *

# ╔═╡ 15ffc73e-f883-48b7-bb95-ae2428c7839e
const mdp_deterministic = make_deterministic_gridworld(;wind = wind_values)

# ╔═╡ d6f3f0d7-c58e-4101-a70f-7977c60c1821
const dense_feature_setup = let
	feature_vector = zeros(Float32, length(mdp_deterministic.states))
	function update_feature_vector!(v::Vector{T}, s) where T<:Real
		v .= zero(T)
		idx = mdp_deterministic.state_index[s]
		v[idx] = one(T)
		return v
	end
	(;feature_vector, update_feature_vector!)
end

# ╔═╡ 88cb461d-d332-4f8a-ac39-6c00defb17a2
const mdp_stochastic = make_stochastic_gridworld(;wind = wind_values)

# ╔═╡ dc87916c-e528-4095-980c-be82b4ffff43
const mdp_continuing = make_stochastic_gridworld(;wind = wind_values, continuing=true)

# ╔═╡ 8f1f7c38-7789-4104-af6f-798173aa0d3f
const mdp_π = make_random_policy(mdp_deterministic)

# ╔═╡ 27d4015b-817d-4724-b712-1dd33990cc80
# ╠═╡ skip_as_script = true
#=╠═╡
repeated_run(policy_evaluation_v, 100, mdp_deterministic, mdp_π, 0.99f0)
  ╠═╡ =#

# ╔═╡ 731987c4-4335-4e39-827d-74d1b17351a1
# ╠═╡ skip_as_script = true
#=╠═╡
repeated_run(policy_evaluation_v, 1, mdp_deterministic, mdp_π, 0.99f0; usethreads=true)
  ╠═╡ =#

# ╔═╡ ac3d9473-3734-49a0-9e01-360eb423a6f0
repeated_run(policy_evaluation_v, 100, mdp_stochastic, mdp_π, 0.99f0)

# ╔═╡ a741f0b6-a472-4f81-aeaa-1000251bf448
repeated_run(policy_evaluation_v, 1, mdp_stochastic, mdp_π, 0.99f0; usethreads=true)

# ╔═╡ cf43f9bf-4e01-4eb7-8c71-d7be657ad746
#=╠═╡
begin
	@profview_allocs repeated_run(policy_evaluation_v, 1, mdp_deterministic, mdp_π, 0.99f0; usethreads=false)
	@profview_allocs repeated_run(policy_evaluation_v, 100, mdp_deterministic, mdp_π, 0.99f0; usethreads=false)
end
  ╠═╡ =#

# ╔═╡ 86bb7340-2df3-40b5-b5a8-ee6a0881cc98
#=╠═╡
begin
	@profview repeated_run(policy_evaluation_v, 1, mdp_deterministic, mdp_π, 0.99f0)
	@profview repeated_run(policy_evaluation_v, 100, mdp_deterministic, mdp_π, 0.99f0)
end
  ╠═╡ =#

# ╔═╡ a7569534-e573-45c9-b129-0b45940d9409
#=╠═╡
begin
	@profview repeated_run(policy_evaluation_v, 1, mdp_stochastic, mdp_π, 0.99f0)
	@profview repeated_run(policy_evaluation_v, 100, mdp_stochastic, mdp_π, 0.99f0)
end
  ╠═╡ =#

# ╔═╡ 3630121c-ef7e-4f70-812b-62e28ed24f05
#=╠═╡
begin
	@profview repeated_run(policy_evaluation_v, 1, mdp_deterministic, mdp_π, 0.99f0; usethreads=true)
	@profview repeated_run(policy_evaluation_v, 1, mdp_deterministic, mdp_π, 0.99f0; usethreads=true)
end
  ╠═╡ =#

# ╔═╡ fb2b1149-dd76-4ff5-a25c-c1aa435e6f09
#=╠═╡
begin
	@profview repeated_run(policy_evaluation_v, 1, mdp_stochastic, mdp_π, 0.99f0; usethreads=true)
	@profview repeated_run(policy_evaluation_v, 1, mdp_stochastic, mdp_π, 0.99f0; usethreads=true)
end
  ╠═╡ =#

# ╔═╡ 87c0880f-87d5-4566-8b4c-694b25ef864c
# ╠═╡ skip_as_script = true
#=╠═╡
repeated_run(policy_evaluation_q, 100, mdp_deterministic, mdp_π, 0.99f0)
  ╠═╡ =#

# ╔═╡ 62b42256-95f8-4e19-b7a9-1e25e7ad30c4
# ╠═╡ skip_as_script = true
#=╠═╡
repeated_run(policy_evaluation_q, 100, mdp_stochastic, mdp_π, 0.99f0)
  ╠═╡ =#

# ╔═╡ 082f4f1a-6302-4959-aad2-9a561d91362f
# ╠═╡ skip_as_script = true
#=╠═╡
@code_warntype repeated_run(policy_evaluation_q, 1, mdp_deterministic, mdp_π, 0.99f0)
  ╠═╡ =#

# ╔═╡ 27d555ce-3b4b-453f-8f7c-65644fe42c8c
#=╠═╡
begin
	@profview repeated_run(policy_evaluation_q, 1, mdp_deterministic, mdp_π, 0.99f0)
	@profview repeated_run(policy_evaluation_q, 100, mdp_deterministic, mdp_π, 0.99f0)
end
  ╠═╡ =#

# ╔═╡ 90b9f7b4-032d-40a5-b0c5-fb3aab6e8231
#=╠═╡
begin
	@profview repeated_run(policy_evaluation_q, 1, mdp_stochastic, mdp_π, 0.99f0)
	@profview repeated_run(policy_evaluation_q, 100, mdp_stochastic, mdp_π, 0.99f0)
end
  ╠═╡ =#

# ╔═╡ f35b4530-4099-4016-8b73-4aa615dce58e
# ╠═╡ skip_as_script = true
#=╠═╡
repeated_run(policy_iteration_v, 100, mdp_deterministic, 0.99f0)
  ╠═╡ =#

# ╔═╡ 2d4b5fa3-5404-4901-a6bf-69ce0a511994
# ╠═╡ skip_as_script = true
#=╠═╡
repeated_run(policy_iteration_v, 100, mdp_stochastic, 0.99f0)
  ╠═╡ =#

# ╔═╡ 40df5a8d-0480-42a8-8415-0f48228cb8f7
#=╠═╡
begin
	@profview repeated_run(policy_iteration_v, 1, mdp_deterministic, 0.99f0)
	@profview repeated_run(policy_iteration_v, 100, mdp_deterministic, 0.99f0)
end
  ╠═╡ =#

# ╔═╡ 8111ca1e-2c57-497a-9476-c91359759c9a
#=╠═╡
begin
	@profview repeated_run(policy_iteration_v, 1, mdp_stochastic, 0.99f0)
	@profview repeated_run(policy_iteration_v, 100, mdp_stochastic, 0.99f0)
end
  ╠═╡ =#

# ╔═╡ 27f441f9-7fcb-4a83-893f-c403aa9a7868
# ╠═╡ skip_as_script = true
#=╠═╡
repeated_run(value_iteration_v, 10_000, mdp_deterministic, 0.99f0; show_message = false)
  ╠═╡ =#

# ╔═╡ 65b427b6-fed6-4cba-acad-aa8cad443d28
# ╠═╡ skip_as_script = true
#=╠═╡
repeated_run(value_iteration_v, 1_000, mdp_stochastic, 0.99f0; show_message = false)
  ╠═╡ =#

# ╔═╡ e6c21fbe-cd94-4562-99ef-933651af002f
#=╠═╡
begin
	@profview repeated_run(value_iteration_v, 1, mdp_deterministic, 0.99f0; show_message = false)
	@profview repeated_run(value_iteration_v, 10_000, mdp_deterministic, 0.99f0; show_message = false)
end
  ╠═╡ =#

# ╔═╡ ed1675a9-7788-4ca8-864f-131d9acb3ce7
#=╠═╡
begin
	@profview repeated_run(value_iteration_v, 1, mdp_stochastic, 0.99f0; show_message = false)
	@profview repeated_run(value_iteration_v, 1_000, mdp_stochastic, 0.99f0; show_message = false)
end
  ╠═╡ =#

# ╔═╡ 54cd9005-5330-4e24-963c-767e39309d9c
# ╠═╡ skip_as_script = true
#=╠═╡
monte_carlo_policy_prediction_v(mdp_deterministic, mdp_π, 0.99f0, 100)
  ╠═╡ =#

# ╔═╡ f31270cf-038d-454d-8c39-79b56cda992d
monte_carlo_policy_prediction_v(mdp_stochastic, mdp_π, 0.99f0, 100)

# ╔═╡ 4a93e565-56e2-424d-a614-c5aa8c78f2f1
#=╠═╡
begin
	@profview monte_carlo_policy_prediction_v(mdp_deterministic, mdp_π, 0.99f0, 1)
	@profview monte_carlo_policy_prediction_v(mdp_deterministic, mdp_π, 0.99f0, 100)
end
  ╠═╡ =#

# ╔═╡ 87355b4b-0868-4e07-a13e-de253cc69a95
#=╠═╡
begin
	@profview monte_carlo_policy_prediction_v(mdp_stochastic, mdp_π, 0.99f0, 1)
	@profview monte_carlo_policy_prediction_v(mdp_stochastic, mdp_π, 0.99f0, 100)
end
  ╠═╡ =#

# ╔═╡ 3bd283c9-f413-4a08-a030-103eb7d186a4
monte_carlo_policy_prediction_q(mdp_deterministic, mdp_π, 0.99f0, 100)

# ╔═╡ ac52d6f5-9db0-4095-8301-0bf8d6116cae
monte_carlo_policy_prediction_q(mdp_stochastic, mdp_π, 0.99f0, 100)

# ╔═╡ b5271864-a42e-4ecf-b5b2-dfa45e0fd7a6
#=╠═╡
begin
	@profview monte_carlo_policy_prediction_q(mdp_deterministic, mdp_π, 0.99f0, 1)
	@profview monte_carlo_policy_prediction_q(mdp_deterministic, mdp_π, 0.99f0, 100)
end
  ╠═╡ =#

# ╔═╡ 5789ae84-8cb0-4bf2-8de5-9302e321cb07
#=╠═╡
begin
	@profview monte_carlo_policy_prediction_q(mdp_stochastic, mdp_π, 0.99f0, 1)
	@profview monte_carlo_policy_prediction_q(mdp_stochastic, mdp_π, 0.99f0, 100)
end
  ╠═╡ =#

# ╔═╡ 0702027f-49dd-465c-9f40-b0ce52f6d8c4
monte_carlo_control_ϵ_soft(mdp_deterministic, 0.99f0, 10_000)

# ╔═╡ 8561fce9-c455-4253-a76c-a8c9845b53d5
monte_carlo_control_ϵ_soft(mdp_stochastic, 0.99f0, 10_000)

# ╔═╡ 84b4c5be-62ac-4c4b-9a7d-a61a8f7b7d87
#=╠═╡
begin
	@profview monte_carlo_control_ϵ_soft(mdp_deterministic, 0.99f0, 1)
	@profview monte_carlo_control_ϵ_soft(mdp_deterministic, 0.99f0, 10_000)
end
  ╠═╡ =#

# ╔═╡ fb665ee2-51f7-4942-a5b1-39b1e73d81a5
#=╠═╡
begin
	@profview monte_carlo_control_ϵ_soft(mdp_stochastic, 0.99f0, 1)
	@profview monte_carlo_control_ϵ_soft(mdp_stochastic, 0.99f0, 10_000)
end
  ╠═╡ =#

# ╔═╡ 42790259-bdc5-4962-b13f-e781095671ae
td0_policy_prediction_v(mdp_deterministic, mdp_π, 0.99f0; max_steps = 100_000)

# ╔═╡ 63c8aeab-9d63-472b-a475-d158dba6f126
td0_policy_prediction_v(mdp_stochastic, mdp_π, 0.99f0; max_steps = 100_000)

# ╔═╡ ed114798-c20e-48ec-8a7a-d69a6327a047
#=╠═╡
begin
	@profview td0_policy_prediction_v(mdp_deterministic, mdp_π, 0.99f0; max_steps = 10)
	@profview td0_policy_prediction_v(mdp_deterministic, mdp_π, 0.99f0; max_steps = 100_000)
end
  ╠═╡ =#

# ╔═╡ db52630a-9c96-4974-afc8-c3f717cca604
#=╠═╡
begin
	@profview td0_policy_prediction_v(mdp_stochastic, mdp_π, 0.99f0; max_steps = 10)
	@profview td0_policy_prediction_v(mdp_stochastic, mdp_π, 0.99f0; max_steps = 100_000)
end
  ╠═╡ =#

# ╔═╡ beeab34c-339b-4cb3-9998-713a139ad843
sarsa(mdp_deterministic, 0.99f0; max_steps = 100_000)

# ╔═╡ 3a09fd58-bee6-4f89-8c65-cd0e9248e8cb
sarsa(mdp_stochastic, 0.99f0; max_steps = 100_000)

# ╔═╡ 9e0f227a-b3a6-410a-ab28-92a174e12337
#=╠═╡
begin
	@profview sarsa(mdp_deterministic, 0.99f0; max_steps = 10)
	@profview sarsa(mdp_deterministic, 0.99f0; max_steps = 100_000)
end
  ╠═╡ =#

# ╔═╡ 832351b7-f515-4f66-991f-09e1c214462b
#=╠═╡
begin
	@profview sarsa(mdp_stochastic, 0.99f0; max_steps = 10)
	@profview sarsa(mdp_stochastic, 0.99f0; max_steps = 100_000)
end
  ╠═╡ =#

# ╔═╡ 78317a12-091c-41ee-b529-238ddd7b7361
expected_sarsa(mdp_deterministic, 0.99f0; max_steps = 100_000)

# ╔═╡ 91f5ac83-78e7-4015-bffe-7cee8cb1f8d6
expected_sarsa(mdp_stochastic, 0.99f0; max_steps = 100_000)

# ╔═╡ 12d0471a-9f92-44a0-8af3-0e0a6f4c9ef1
#=╠═╡
begin
	@profview expected_sarsa(mdp_deterministic, 0.99f0; max_steps = 10)
	@profview expected_sarsa(mdp_deterministic, 0.99f0; max_steps = 100_000)
end
  ╠═╡ =#

# ╔═╡ 4267be90-7bce-4709-883c-7bbff85e6471
#=╠═╡
begin
	@profview expected_sarsa(mdp_stochastic, 0.99f0; max_steps = 10)
	@profview expected_sarsa(mdp_stochastic, 0.99f0; max_steps = 100_000)
end
  ╠═╡ =#

# ╔═╡ 2cad4cba-7081-4e87-a66e-db072ff9eb41
double_expected_sarsa(mdp_deterministic, 0.99f0; max_steps = 100_000)

# ╔═╡ 7ec8c174-f08c-464b-adc8-11b98883e44a
double_expected_sarsa(mdp_stochastic, 0.99f0; max_steps = 100_000)

# ╔═╡ 66388b8e-f206-4222-bd96-52c1b9a10127
#=╠═╡
begin
	@profview double_expected_sarsa(mdp_deterministic, 0.99f0; max_steps = 10)
	@profview double_expected_sarsa(mdp_deterministic, 0.99f0; max_steps = 100_000)
end
  ╠═╡ =#

# ╔═╡ b185599c-005e-472a-8a9e-1c06964d79aa
#=╠═╡
begin
	@profview double_expected_sarsa(mdp_stochastic, 0.99f0; max_steps = 10)
	@profview double_expected_sarsa(mdp_stochastic, 0.99f0; max_steps = 100_000)
end
  ╠═╡ =#

# ╔═╡ 96271cbc-1ed7-4db4-a2e9-4d2cf629c414
#=╠═╡
begin
	@profview q_learning(mdp_stochastic, 0.99f0; max_steps = 10)
	@profview q_learning(mdp_stochastic, 0.99f0; max_steps = 100_000)
end
  ╠═╡ =#

# ╔═╡ 55062ba3-870e-497d-a7ff-fec73abc3b20
const state_mdp_deterministic = StateMDP(mdp_deterministic)

# ╔═╡ 01e9cb83-ce4e-476d-8791-5ce451b24cd1
const state_mdp_stochastic = StateMDP(mdp_stochastic)

# ╔═╡ a66048e9-cbfc-403c-b09e-f653fd93c4cd
const state_mdp_continuing = StateMDP(mdp_continuing)

# ╔═╡ 3cab6d8c-7aeb-412d-9acf-b8bc6097d160
state_mdp_π = make_random_policy(state_mdp_deterministic)

# ╔═╡ bbb4a031-2f56-43b5-9c71-afe327e95f27
sample_rollout(mdp_deterministic, mdp_π, 0.99f0)

# ╔═╡ 0dfae082-d8f2-4b7f-8fcc-57e6ce9bdc28
sample_rollout(mdp_deterministic, 0.99f0)

# ╔═╡ 1edfdab0-df23-4e1c-acb5-9f22e35642a1
sample_rollout(state_mdp_deterministic, state_mdp_π, 0.99f0)

# ╔═╡ 75294323-3a3a-4579-8657-c074ea608cfb
#=╠═╡
begin
	@profview repeated_run(sample_rollout, 1, mdp_deterministic, 0.99f0)
	@profview repeated_run(sample_rollout, 100, mdp_deterministic, 0.99f0)
end
  ╠═╡ =#

# ╔═╡ 3e198fe0-3552-4a2b-8758-c00b9dcc5516
average_stochastic_rollout(1000, state_mdp_deterministic, state_mdp_π, 0.99f0)

# ╔═╡ d72b933f-8ced-4aaf-9014-3ee64e27fe2a
repeated_run(sample_rollout, 1000, state_mdp_deterministic, state_mdp_π, 0.99f0)

# ╔═╡ 43f7b3cb-e071-4185-aef8-369e423d9dc6
#=╠═╡
begin
	@profview repeated_run(sample_rollout, 1, state_mdp_deterministic, state_mdp_π, 0.99f0)
	@profview repeated_run(sample_rollout, 1000, state_mdp_deterministic, state_mdp_π, 0.99f0)
end
  ╠═╡ =#

# ╔═╡ 84d72414-f1a4-4701-be6d-4342e22461ae
#=╠═╡
begin
	@profview average_stochastic_rollout(1, state_mdp_deterministic, state_mdp_π, 0.99f0)
	@profview average_stochastic_rollout(1000, state_mdp_deterministic, state_mdp_π, 0.99f0)
end
  ╠═╡ =#

# ╔═╡ 1de02aeb-e36d-4357-adba-3e069c61ebfa
monte_carlo_tree_search(state_mdp_deterministic, 0.99f0, state_mdp_deterministic.initialize_state(); nsims = 1_000)

# ╔═╡ 003928c2-99d6-40df-9067-79bc51ec80b5
#=╠═╡
begin
	@profview monte_carlo_tree_search(state_mdp_deterministic, 0.99f0, state_mdp_deterministic.initialize_state(); nsims = 10)
	@profview monte_carlo_tree_search(state_mdp_deterministic, 0.99f0, state_mdp_deterministic.initialize_state(); nsims = 1_000)
end
  ╠═╡ =#

# ╔═╡ 889a5212-5336-4ccf-ab3b-693759e1afb3
const mountaincar_mdp = MountainCarTask.deterministic_mdp

# ╔═╡ 57a307b5-2aed-48e9-90e2-d3d7550231d9
mountaincar_π = make_random_policy(mountaincar_mdp)

# ╔═╡ 67903e58-e7b2-4556-b551-40eefd9ec203
const mountaincar_continuing = create_mountaincar_continuing_mdp()

# ╔═╡ e20df14f-8eff-43f3-a4c2-8793ec788b82
const access_control = create_access_control_task(10, [1f0, 2f0, 4f0, 8f0])

# ╔═╡ 4f3909ba-96ae-43b9-bd6c-599e54e48b75
const access_control_dense_setup = let
	v_sparse = copy(access_control.setup.feature_vector)
	v = zeros(Float32, length(access_control.setup.feature_vector))
	function f!(v::Vector{T}, s) where T<:Real
		access_control.setup.update_feature_vector!(v_sparse, s)
		v .= zeros(T)
		i = v_sparse.group_index
		v[i] = one(T)
		return v
	end
	(feature_vector = v, update_feature_vector! = f!)
end

# ╔═╡ b337d2d4-d3dc-4ec4-b9d8-96a28cbb5863
access_control_π = make_random_policy(access_control.mdp)

# ╔═╡ 54a0a47f-7dd8-47f8-8f55-86540bb05d3d
const sparse_feature_setup = state_aggregation_feature_setup(state_mdp_deterministic.initialize_state(), length(mdp_deterministic.states), s -> mdp_deterministic.state_index[s])

# ╔═╡ 51e3970d-87da-4be9-a452-faaff294465b
const mountaincar_features = let
	setup = setup_mountaincar_tiles(10, 5)
	(;feature_vector = setup.feature_vector, update_feature_vector! = setup.update_feature_vector!)
end

# ╔═╡ fbb60ce0-6fc3-4278-bf46-9dca5a30a6dd
#=╠═╡
begin
	@profview gradient_monte_carlo_policy_estimation_linear(state_mdp_deterministic, state_mdp_π, 0.99f0, 1, sparse_feature_setup...)
	@profview gradient_monte_carlo_policy_estimation_linear(state_mdp_deterministic, state_mdp_π, 0.99f0, 100, sparse_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ f7acbbcd-134a-47e4-b17f-39c402bb601d
#=╠═╡
begin
	@profview gradient_monte_carlo_policy_estimation_linear(mountaincar_mdp, mountaincar_π, 0.99f0, 1, mountaincar_features...)
	@profview gradient_monte_carlo_policy_estimation_linear(mountaincar_mdp, mountaincar_π, 0.99f0, 10, mountaincar_features...)
end
  ╠═╡ =#

# ╔═╡ 357190d3-e84e-4e1b-8ff6-e6f303528584
#=╠═╡
begin
	@profview gradient_monte_carlo_policy_estimation_fcann(mountaincar_mdp, mountaincar_π, 0.99f0, 1, mountaincar_features..., [64, 64])
	@profview gradient_monte_carlo_policy_estimation_fcann(mountaincar_mdp, mountaincar_π, 0.99f0, 1, mountaincar_features..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 0867de49-1960-4cf1-9f1d-409ad0e05793
semi_gradient_td0_policy_estimation_linear(state_mdp_deterministic, state_mdp_π, 0.99f0, 1000, 1_000_000, sparse_feature_setup...)

# ╔═╡ 2f3ebcbb-e82b-4ab2-88a4-24ed6fb2a448
semi_gradient_td0_policy_estimation_linear(state_mdp_deterministic, state_mdp_π, 0.99f0, 1000, 500_000, dense_feature_setup...)

# ╔═╡ c953683e-fc9b-4eea-bf17-f24566ee99f9
#=╠═╡
begin
	@profview semi_gradient_td0_policy_estimation_linear(state_mdp_deterministic, state_mdp_π, 0.99f0, 1000, 1, sparse_feature_setup...)
	@profview semi_gradient_td0_policy_estimation_linear(state_mdp_deterministic, state_mdp_π, 0.99f0, 1000, 1_000_000, sparse_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ a4abe5ca-8f5f-473a-a7f4-8710e3ce7ec1
#=╠═╡
begin
	@profview semi_gradient_td0_policy_estimation_linear(state_mdp_deterministic, state_mdp_π, 0.99f0, 1000, 1, dense_feature_setup...)
	@profview semi_gradient_td0_policy_estimation_linear(state_mdp_deterministic, state_mdp_π, 0.99f0, 1000, 500_000, dense_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ e5be5928-e5b8-4690-acb3-fe5855634e5d
#=╠═╡
begin
	@profview semi_gradient_td0_policy_estimation_fcann(state_mdp_deterministic, state_mdp_π, 0.99f0, 1000, 1, dense_feature_setup..., [64, 64])
	@profview semi_gradient_td0_policy_estimation_fcann(state_mdp_deterministic, state_mdp_π, 0.99f0, 1000, 100_000, dense_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 91c8ed60-9c3a-4a17-9883-d31bf6314c60
#=╠═╡
begin
	@profview semi_gradient_td0_policy_estimation_fcann(state_mdp_deterministic, state_mdp_π, 0.99f0, 1000, 1, sparse_feature_setup..., [64, 64])
	@profview semi_gradient_td0_policy_estimation_fcann(state_mdp_deterministic, state_mdp_π, 0.99f0, 1000, 100_000, sparse_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 8eb6c31a-8500-4d69-907a-5daa741fbb29
#=╠═╡
begin
	@profview semi_gradient_td0_policy_estimation_fcann(mountaincar_mdp, mountaincar_π, 0.99f0, 1000, 1, mountaincar_features..., [64, 64])
	@profview semi_gradient_td0_policy_estimation_fcann(mountaincar_mdp, mountaincar_π, 0.99f0, 1000, 100_000, mountaincar_features..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ fc1175c7-9275-4b12-8718-c05d5d4634b4
semi_gradient_sarsa_linear(mountaincar_mdp, 1f0, 1000, 1_000_000, mountaincar_features...)

# ╔═╡ 678cda5d-66fa-4a1a-a115-41e16d5555d4
semi_gradient_sarsa_linear(state_mdp_stochastic, 1f0, 1000, 1_000_000, sparse_feature_setup...)

# ╔═╡ 147e3bd2-29f2-4a78-a3fd-4f3531afc3d5
semi_gradient_sarsa_linear(state_mdp_stochastic, 1f0, 1000, 1_000_000, dense_feature_setup...)

# ╔═╡ 21b5e82d-fdda-4caf-a1aa-222152b69f8b
#=╠═╡
begin
	@profview semi_gradient_sarsa_linear(mountaincar_mdp, 1f0, 1000, 1, mountaincar_features...)
	@profview semi_gradient_sarsa_linear(mountaincar_mdp, 1f0, 1000, 1_000_000, mountaincar_features...)
end
  ╠═╡ =#

# ╔═╡ ac1e6eeb-fcd1-4bad-a46a-edc9a9d17c1f
#=╠═╡
begin
	@profview semi_gradient_sarsa_linear(state_mdp_stochastic, 1f0, 1000, 1, sparse_feature_setup...)
	@profview semi_gradient_sarsa_linear(state_mdp_stochastic, 1f0, typemax(Int64), 100_000, sparse_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ beafd867-c4c3-4b97-99a8-c8dc3d591341
#=╠═╡
begin
	@profview semi_gradient_sarsa_linear(state_mdp_stochastic, 1f0, 1000, 1, dense_feature_setup...)
	@profview semi_gradient_sarsa_linear(state_mdp_stochastic, 1f0, typemax(Int64), 100_000, dense_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ affba263-c895-4d8f-af32-2c561a884d49
semi_gradient_sarsa_fcann(mountaincar_mdp, 1f0, 1000, 10_000, mountaincar_features..., [64, 64])

# ╔═╡ 16ad1491-85db-4b1a-a451-421202d079b2
semi_gradient_sarsa_fcann(state_mdp_stochastic, 1f0, 1000, 10_000, sparse_feature_setup..., [64, 64])

# ╔═╡ 1a45053e-1732-4dbd-9682-d34f60579e74
semi_gradient_sarsa_fcann(state_mdp_stochastic, 1f0, 1000, 10_000, dense_feature_setup..., [64, 64])

# ╔═╡ 3b5d759c-8085-4912-9646-fe45087d3d9f
#=╠═╡
begin
	@profview semi_gradient_sarsa_fcann(mountaincar_mdp, 1f0, 1000, 1, mountaincar_features..., [64, 64])
	@profview semi_gradient_sarsa_fcann(mountaincar_mdp, 1f0, 1000, 10_000, mountaincar_features..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 4a290372-be64-4e45-a96a-01968b57aacf
#=╠═╡
begin
	@profview semi_gradient_sarsa_fcann(state_mdp_stochastic, 1f0, 1000, 1, sparse_feature_setup..., [64, 64])
	@profview semi_gradient_sarsa_fcann(state_mdp_stochastic, 1f0, 1000, 10_000, sparse_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ c10aba9e-daf4-4767-9e14-51cd06ad5495
#=╠═╡
begin
	@profview semi_gradient_sarsa_fcann(state_mdp_stochastic, 1f0, 1000, 1, dense_feature_setup..., [64, 64])
	@profview semi_gradient_sarsa_fcann(state_mdp_stochastic, 1f0, typemax(Int64), 10_000, dense_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ b00e0fe1-cdc2-4d28-946a-a7d256ab7f49
semi_gradient_double_sarsa_linear(mountaincar_mdp, 1f0, 1000, 1_000_000, mountaincar_features...)

# ╔═╡ 46b49f4a-1375-4286-a1a9-e86b4b8e2c63
#=╠═╡
begin
	@profview semi_gradient_double_sarsa_linear(mountaincar_mdp, 1f0, 1000, 1, mountaincar_features...)
	@profview semi_gradient_double_sarsa_linear(mountaincar_mdp, 1f0, 1000, 100_000, mountaincar_features...)
end
  ╠═╡ =#

# ╔═╡ f7259f58-f9c2-43c7-8173-e01a2a2683b0
semi_gradient_double_sarsa_fcann(mountaincar_mdp, 1f0, 1000, 10_000, mountaincar_features..., [64, 64])

# ╔═╡ 185ec9b4-5c0f-4400-8ca6-623a24a975f5
#=╠═╡
begin
	@profview semi_gradient_double_sarsa_fcann(mountaincar_mdp, 1f0, 1000, 1, mountaincar_features..., [64, 64])
	@profview semi_gradient_double_sarsa_fcann(mountaincar_mdp, 1f0, 1000, 10_000, mountaincar_features..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ a7127d77-ca72-448d-aa8f-8e25eeee547a
semi_gradient_double_sarsa_linear(state_mdp_stochastic, 0.9f0, typemax(Int64), 100_000, sparse_feature_setup...)

# ╔═╡ 340ef2f9-889b-4e97-8bd5-66e6f1d174cd
#=╠═╡
begin
	@profview semi_gradient_double_sarsa_linear(state_mdp_stochastic, 0.9f0, typemax(Int64), 1, sparse_feature_setup...)
	@profview semi_gradient_double_sarsa_linear(state_mdp_stochastic, 0.9f0, typemax(Int64), 100_000, sparse_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ 205a88eb-ff68-46f0-9f3c-650d9e83a401
semi_gradient_double_sarsa_fcann(state_mdp_stochastic, 0.9f0, typemax(Int64), 10_000, sparse_feature_setup..., [64, 64])

# ╔═╡ 109d913a-de49-4b2b-ab38-fa44b80ec178
#=╠═╡
begin
	@profview semi_gradient_double_sarsa_fcann(state_mdp_stochastic, 0.9f0, typemax(Int64), 1, sparse_feature_setup..., [64, 64])
	@profview semi_gradient_double_sarsa_fcann(state_mdp_stochastic, 0.9f0, typemax(Int64), 10_000, sparse_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ a08cca6f-71ba-4c27-bae3-7c361d6a7684
semi_gradient_double_sarsa_linear(state_mdp_stochastic, 0.9f0, typemax(Int64), 100_000, dense_feature_setup...)

# ╔═╡ 0ed2e466-10ec-4cb4-9c95-329e72da55aa
#=╠═╡
begin
	@profview semi_gradient_double_sarsa_linear(state_mdp_stochastic, 0.9f0, typemax(Int64), 1, dense_feature_setup...)
	@profview semi_gradient_double_sarsa_linear(state_mdp_stochastic, 0.9f0, typemax(Int64), 100_000, dense_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ e162133d-3c89-45fe-bd9f-c7f04df1038f
semi_gradient_double_sarsa_fcann(state_mdp_stochastic, 0.9f0, typemax(Int64), 10_000, dense_feature_setup..., [64, 64])

# ╔═╡ 81acfb57-9863-4f26-bc70-2181640fbc77
#=╠═╡
begin
	@profview semi_gradient_double_sarsa_fcann(state_mdp_stochastic, 0.9f0, typemax(Int64), 1, dense_feature_setup..., [64, 64])
	@profview semi_gradient_double_sarsa_fcann(state_mdp_stochastic, 0.9f0, typemax(Int64), 10_000, dense_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 17d49653-118b-4865-a9cc-197981889bee
semi_gradient_dp_linear(mountaincar_mdp, 1f0, 1000, 1_000_000, mountaincar_features...)

# ╔═╡ e6ec3fdf-e3a0-4d38-8791-0d99de9e23eb
#=╠═╡
begin
	@profview semi_gradient_dp_linear(mountaincar_mdp, 1f0, 1000, 1, mountaincar_features...)
	@profview semi_gradient_dp_linear(mountaincar_mdp, 1f0, 1000, 100_000, mountaincar_features...)
end
  ╠═╡ =#

# ╔═╡ 0c1eefae-b73b-4d2b-83f2-0a605072f363
semi_gradient_dp_linear(state_mdp_stochastic, 0.9f0, 1000, 1_000_000, sparse_feature_setup...)

# ╔═╡ cdb3b561-afaf-4a8e-8ee4-6b8ae8798de8
#=╠═╡
begin
	@profview semi_gradient_dp_linear(state_mdp_stochastic, 0.9f0, 1000, 1, sparse_feature_setup...)
	@profview semi_gradient_dp_linear(state_mdp_stochastic, 0.9f0, 1000, 1_000_000, sparse_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ e7c43fa3-af1b-46cf-abab-e34ff3637c8b
semi_gradient_dp_linear(state_mdp_stochastic, 0.9f0, typemax(Int64), 100_000, dense_feature_setup...)

# ╔═╡ 9eabf565-ba4d-4d7e-be25-626d7a7894c8
#=╠═╡
begin
	@profview semi_gradient_dp_linear(state_mdp_stochastic, 0.9f0, typemax(Int64), 1, dense_feature_setup...)
	@profview semi_gradient_dp_linear(state_mdp_stochastic, 0.9f0, typemax(Int64), 100_000, dense_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ a855e694-f802-42e8-82d4-4a9b48e1e65b
semi_gradient_dp_fcann(mountaincar_mdp, 1f0, 1000, 10_000, mountaincar_features..., [64, 64])

# ╔═╡ a4462943-4187-4556-81a4-18a90e3373a7
#=╠═╡
begin
	@profview semi_gradient_dp_fcann(mountaincar_mdp, 1f0, 1000, 1, mountaincar_features..., [64, 64])
	@profview semi_gradient_dp_fcann(mountaincar_mdp, 1f0, 1000, 10_000, mountaincar_features..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 08408cc9-c964-49f3-9811-3399c463ca30
semi_gradient_dp_fcann(state_mdp_stochastic, 0.9f0, 1000, 10_000, sparse_feature_setup..., [64, 64])

# ╔═╡ 488bed89-cfd6-465f-9317-d9bb9a2123ba
#=╠═╡
begin
	@profview semi_gradient_dp_fcann(state_mdp_stochastic, 0.9f0, 1000, 1, sparse_feature_setup..., [64, 64])
	@profview semi_gradient_dp_fcann(state_mdp_stochastic, 0.9f0, 1000, 10_000, sparse_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 2281fc10-27b0-4cd4-bd1b-11545e96d1e6
semi_gradient_dp_fcann(state_mdp_stochastic, 0.9f0, 1000, 10_000, dense_feature_setup..., [64, 64])

# ╔═╡ d925e225-ce8b-46f0-97b3-2c267968cb34
#=╠═╡
begin
	@profview semi_gradient_dp_fcann(state_mdp_stochastic, 0.9f0, 1000, 1, dense_feature_setup..., [64, 64])
	@profview semi_gradient_dp_fcann(state_mdp_stochastic, 0.9f0, 1000, 10_000, dense_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 9e9c511a-b07a-4973-a0ff-147c5a5da73d
semi_gradient_differential_sarsa_linear(access_control.mdp, 1_000_000, access_control.setup...)

# ╔═╡ f9948d01-ce46-4911-bc6a-12a1728f769e
#=╠═╡
begin
	@profview semi_gradient_differential_sarsa_linear(access_control.mdp, 1, access_control.setup...)
	@profview semi_gradient_differential_sarsa_linear(access_control.mdp, 1_000_000, access_control.setup...)
end
  ╠═╡ =#

# ╔═╡ 2bc3a54d-e1da-45ef-b724-2fa0c97b9823
semi_gradient_differential_sarsa_fcann(access_control.mdp, 100_000, access_control.setup..., [64, 64])

# ╔═╡ 2dd2d7b0-11a4-439c-823e-dcd59cd6badc
#=╠═╡
begin
	@profview semi_gradient_differential_sarsa_fcann(access_control.mdp, 1, access_control.setup..., [64, 64])
	@profview semi_gradient_differential_sarsa_fcann(access_control.mdp, 100_000, access_control.setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 9722bbc8-108b-4fc3-990c-84e1eeedcf24
semi_gradient_differential_sarsa_linear(access_control.mdp, 1_000_000, access_control_dense_setup...)

# ╔═╡ 4ba4e546-441b-4632-9d63-7a9010dc4ae6
#=╠═╡
begin
	@profview semi_gradient_differential_sarsa_linear(access_control.mdp, 1, access_control_dense_setup...)
	@profview semi_gradient_differential_sarsa_linear(access_control.mdp, 1_000_000, access_control_dense_setup...)
end
  ╠═╡ =#

# ╔═╡ 87f35c6f-a9c4-4288-a5f4-f7a169c9df84
semi_gradient_differential_sarsa_fcann(access_control.mdp, 100_000, access_control_dense_setup..., [64, 64])

# ╔═╡ aefe689f-c043-4bf0-91f6-b5f5c23f11fa
#=╠═╡
begin
	@profview semi_gradient_differential_sarsa_fcann(access_control.mdp, 1, access_control_dense_setup..., [64, 64])
	@profview semi_gradient_differential_sarsa_fcann(access_control.mdp, 100_000, access_control_dense_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 5e6c8504-875a-497f-b82e-e09e7f0eba2e
semi_gradient_TDλ_linear(mountaincar_mdp, mountaincar_π, 1f0, 0.5f0, typemax(Int64), 1_000_000, mountaincar_features...)

# ╔═╡ 716e0e85-53ee-4415-8c5e-ec5f5581a93f
#=╠═╡
begin
	@profview semi_gradient_TDλ_linear(mountaincar_mdp, mountaincar_π, 1f0, 0.5f0, typemax(Int64), 1, mountaincar_features...)
	@profview semi_gradient_TDλ_linear(mountaincar_mdp, mountaincar_π, 1f0, 0.5f0, typemax(Int64), 1_000_000, mountaincar_features...)
end
  ╠═╡ =#

# ╔═╡ 53305264-bb8b-4061-873c-9af7f79491f4
semi_gradient_TDλ_fcann(mountaincar_mdp, mountaincar_π, 1f0, 0.5f0, typemax(Int64), 10_000, mountaincar_features..., [64, 64])

# ╔═╡ 803720bd-72b2-4f89-9c97-100cc3d4aa98
#=╠═╡
begin
	@profview semi_gradient_TDλ_fcann(mountaincar_mdp, mountaincar_π, 1f0, 0.5f0, typemax(Int64), 1, mountaincar_features..., [64, 64])
	@profview semi_gradient_TDλ_fcann(mountaincar_mdp, mountaincar_π, 1f0, 0.5f0, typemax(Int64), 10_000, mountaincar_features..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ b13cd00e-0330-4eca-8970-52ddaa2847a8
semi_gradient_TDλ_linear(state_mdp_stochastic, state_mdp_π, 1f0, 0.5f0, typemax(Int64), 1_000_000, sparse_feature_setup...)

# ╔═╡ 5ca8cc42-7ab5-42d0-b74a-c1933fad3a36
#=╠═╡
begin
	@profview semi_gradient_TDλ_linear(state_mdp_stochastic, state_mdp_π, 1f0, 0.5f0, typemax(Int64), 1, sparse_feature_setup...)
	@profview semi_gradient_TDλ_linear(state_mdp_stochastic, state_mdp_π, 1f0, 0.5f0, typemax(Int64), 1_000_000, sparse_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ 0261cf11-25bd-430e-8b3b-f3295ff88921
semi_gradient_TDλ_fcann(state_mdp_stochastic, state_mdp_π, 1f0, 0.5f0, typemax(Int64), 100_000, sparse_feature_setup..., [64, 64])

# ╔═╡ 32c6b199-e334-43f9-a0ef-daf77964ec0e
#=╠═╡
begin
	@profview semi_gradient_TDλ_fcann(state_mdp_stochastic, state_mdp_π, 1f0, 0.5f0, typemax(Int64), 1, sparse_feature_setup..., [64, 64])
	@profview semi_gradient_TDλ_fcann(state_mdp_stochastic, state_mdp_π, 1f0, 0.5f0, typemax(Int64), 100_000, sparse_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ a07e5b9b-a36a-42d0-ba68-21a83444de82
semi_gradient_TDλ_linear(state_mdp_stochastic, state_mdp_π, 1f0, 0.5f0, typemax(Int64), 1_000_000, dense_feature_setup...)

# ╔═╡ cd5d18d5-e69e-4b82-9c5d-fceab0906e93
#=╠═╡
begin
	@profview semi_gradient_TDλ_linear(state_mdp_stochastic, state_mdp_π, 1f0, 0.5f0, typemax(Int64), 1, dense_feature_setup...)
	@profview semi_gradient_TDλ_linear(state_mdp_stochastic, state_mdp_π, 1f0, 0.5f0, typemax(Int64), 1_000_000, dense_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ 25527ecb-0e92-4535-9261-6edeb4ea28cf
semi_gradient_TDλ_fcann(state_mdp_stochastic, state_mdp_π, 1f0, 0.5f0, typemax(Int64), 10_000, dense_feature_setup..., [64, 64])

# ╔═╡ 8b3f9720-46b9-49a9-81a7-7fa97351a3db
#=╠═╡
begin
	@profview semi_gradient_TDλ_fcann(state_mdp_stochastic, state_mdp_π, 1f0, 0.5f0, typemax(Int64), 1, dense_feature_setup..., [64, 64])
	@profview semi_gradient_TDλ_fcann(state_mdp_stochastic, state_mdp_π, 1f0, 0.5f0, typemax(Int64), 10_000, dense_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 080a9c27-20f0-4458-8156-8f5724f921a7
semi_gradient_TDλ_linear(access_control.mdp, access_control_π, 0.5f0, 1_000_000, access_control.setup...)

# ╔═╡ 0989bd35-34b7-43ec-8e5a-954a3a31968c
#=╠═╡
begin
	@profview semi_gradient_TDλ_linear(access_control.mdp, access_control_π, 0.5f0, 1, access_control.setup...)
	@profview semi_gradient_TDλ_linear(access_control.mdp, access_control_π, 0.5f0, 1_000_000, access_control.setup...)
end
  ╠═╡ =#

# ╔═╡ 5c095c88-3337-414f-9c9b-cf5cbd73fff6
semi_gradient_TDλ_fcann(access_control.mdp, access_control_π, 0.5f0, 100_000, access_control.setup..., [64, 64])

# ╔═╡ 198cb57d-e466-46e9-bb0a-4bbf91cc7ff3
#=╠═╡
begin
	@profview semi_gradient_TDλ_fcann(access_control.mdp, access_control_π, 0.5f0, 1, access_control.setup..., [64, 64])
	@profview semi_gradient_TDλ_fcann(access_control.mdp, access_control_π, 0.5f0, 100_000, access_control.setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 0c9cbd7c-0d84-4e26-81e4-54a430d31825
semi_gradient_TDλ_linear(access_control.mdp, access_control_π, 0.5f0, 1_000_000, access_control_dense_setup...)

# ╔═╡ e298d5a6-1ba7-4b28-ba76-f99e87f30462
#=╠═╡
begin
	@profview semi_gradient_TDλ_linear(access_control.mdp, access_control_π, 0.5f0, 1, access_control_dense_setup...)
	@profview semi_gradient_TDλ_linear(access_control.mdp, access_control_π, 0.5f0, 1_000_000, access_control_dense_setup...)
end
  ╠═╡ =#

# ╔═╡ 606d1f9e-b875-45d8-a9c0-2e754201901d
semi_gradient_TDλ_fcann(access_control.mdp, access_control_π, 0.5f0, 100_000, access_control_dense_setup..., [64, 64])

# ╔═╡ 7d6c0197-51f1-441d-8ed8-58d11dd83201
#=╠═╡
begin
	@profview semi_gradient_TDλ_fcann(access_control.mdp, access_control_π, 0.5f0, 1, access_control_dense_setup..., [64, 64])
	@profview semi_gradient_TDλ_fcann(access_control.mdp, access_control_π, 0.5f0, 100_000, access_control_dense_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 0ba6dc03-0a00-4668-940b-dd940d78068b
true_online_TDλ(state_mdp_stochastic, state_mdp_π, 0.99f0, 0.5f0, typemax(Int64), 1_000_000, sparse_feature_setup...)

# ╔═╡ bb837440-8cde-4a74-88ff-b80e5ebe0f41
#=╠═╡
begin
	@profview true_online_TDλ(state_mdp_stochastic, state_mdp_π, 0.99f0, 0.5f0, typemax(Int64), 1, sparse_feature_setup...)
	@profview true_online_TDλ(state_mdp_stochastic, state_mdp_π, 0.99f0, 0.5f0, typemax(Int64), 1_000_000, sparse_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ 890990a9-0f1f-4f33-9c5b-079ec565a4f2
true_online_TDλ(state_mdp_stochastic, state_mdp_π, 0.99f0, 0.5f0, typemax(Int64), 100_000, dense_feature_setup...)

# ╔═╡ 7a3b441c-84ee-4dec-a000-27adc7d2fc5e
#=╠═╡
begin
	@profview true_online_TDλ(state_mdp_stochastic, state_mdp_π, 0.99f0, 0.5f0, typemax(Int64), 1, dense_feature_setup...)
	@profview true_online_TDλ(state_mdp_stochastic, state_mdp_π, 0.99f0, 0.5f0, typemax(Int64), 1_000_000, dense_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ 9a7c1d5e-0ecd-4f1a-b2ec-0f263d90895a
sarsa_λ_linear(mountaincar_mdp, 1f0, 0.5f0, typemax(Int64), 100_000, mountaincar_features...)

# ╔═╡ 50add8c3-83b9-48a9-861a-aba64f15b130
#=╠═╡
begin
	@profview sarsa_λ_linear(mountaincar_mdp, 1f0, 0.5f0, typemax(Int64), 1, mountaincar_features...)
	@profview sarsa_λ_linear(mountaincar_mdp, 1f0, 0.5f0, typemax(Int64), 100_000, mountaincar_features...)
end
  ╠═╡ =#

# ╔═╡ d983b0b0-58a3-48c9-8ad2-02c70aa56ed4
sarsa_λ_fcann(mountaincar_mdp, 1f0, 0.5f0, typemax(Int64), 10_000, mountaincar_features..., [64, 64])

# ╔═╡ ef9ee88d-ec94-4827-aef1-f219e2e937af
#=╠═╡
begin
	@profview sarsa_λ_fcann(mountaincar_mdp, 1f0, 0.5f0, typemax(Int64), 1, mountaincar_features..., [64, 64])
	@profview sarsa_λ_fcann(mountaincar_mdp, 1f0, 0.5f0, typemax(Int64), 10_000, mountaincar_features..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 73ffa88a-702c-44f7-a8c4-a22f5e18738f
sarsa_λ_linear(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 100_000, sparse_feature_setup...)

# ╔═╡ 00355a66-641b-4d85-95ef-daec220a2f07
#=╠═╡
begin
	@profview sarsa_λ_linear(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 1, sparse_feature_setup...)
	@profview sarsa_λ_linear(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 100_000, sparse_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ c1389485-f630-4182-940b-3aba622a1aea
sarsa_λ_fcann(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 10_000, sparse_feature_setup..., [64, 64])

# ╔═╡ fe31a5ca-dcec-41f6-8ee3-87bf606a13cc
#=╠═╡
begin
	@profview sarsa_λ_fcann(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 1, sparse_feature_setup..., [64, 64])
	@profview sarsa_λ_fcann(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 10_000, sparse_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 377dfd87-9166-47fd-88af-541e17f59504
#=╠═╡
begin
	@profview sarsa_λ_linear(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 1, dense_feature_setup...)
	@profview sarsa_λ_linear(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 100_000, dense_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ 183a4cea-71ec-410e-b321-cd97aaf4cecb
#=╠═╡
begin
	@profview sarsa_λ_fcann(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 1, dense_feature_setup..., [64, 64])
	@profview sarsa_λ_fcann(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 10_000, dense_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ fea5b8ce-d36f-4e4a-9193-20cf2dc8a0df
sarsa_λ_linear(access_control.mdp, 0.5f0, 1_000_000, access_control.setup...)

# ╔═╡ de93b46d-48c8-4dd3-93a5-1b07e69764cb
#=╠═╡
begin
	@profview sarsa_λ_linear(access_control.mdp, 0.5f0, 1, access_control.setup...)
	@profview sarsa_λ_linear(access_control.mdp, 0.5f0, 1_000_000, access_control.setup...)
end
  ╠═╡ =#

# ╔═╡ 03a7ef3d-599f-457a-8ec1-26aef821c1e6
#=╠═╡
begin
	@profview sarsa_λ_fcann(access_control.mdp, 0.5f0, 1, access_control.setup..., [64, 64])
	@profview sarsa_λ_fcann(access_control.mdp, 0.5f0, 10_000, access_control.setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 1a0463af-078e-4360-807e-a234e1bef07f
sarsa_λ_linear(mountaincar_continuing, 0.5f0, 100_000, mountaincar_features...)

# ╔═╡ 37fe1f34-e18c-4197-a209-4c899b82ef21
#=╠═╡
begin
	@profview sarsa_λ_linear(mountaincar_continuing, 0.5f0, 1, mountaincar_features...)
	@profview sarsa_λ_linear(mountaincar_continuing, 0.5f0, 100_000, mountaincar_features...)
end
  ╠═╡ =#

# ╔═╡ 66917f9d-e58e-4e55-b779-7988fa5acfef
#=╠═╡
begin
	@profview sarsa_λ_fcann(mountaincar_continuing, 0.5f0, 1, mountaincar_features..., [64, 64])
	@profview sarsa_λ_fcann(mountaincar_continuing, 0.5f0, 10_000, mountaincar_features..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ cdf997ef-1ea5-4bd5-ade9-64e7382cec40
dp_λ_linear(mountaincar_mdp, 1f0, 0.5f0, typemax(Int64), 1_000_000, mountaincar_features...)

# ╔═╡ a78caabf-6cae-46c9-b8e9-f599905d1341
#=╠═╡
begin
	@profview dp_λ_linear(mountaincar_mdp, 1f0, 0.5f0, typemax(Int64), 1, mountaincar_features...)
	@profview dp_λ_linear(mountaincar_mdp, 1f0, 0.5f0, typemax(Int64), 1_000_000, mountaincar_features...)
end
  ╠═╡ =#

# ╔═╡ 73b5862f-fb82-4599-be8c-0c76481f728d
dp_λ_fcann(mountaincar_mdp, 1f0, 0.5f0, typemax(Int64), 10_000, mountaincar_features..., [64, 64])

# ╔═╡ da8985f7-4ba3-45cb-bc71-0d1a70739cd5
#=╠═╡
begin
	@profview dp_λ_fcann(mountaincar_mdp, 1f0, 0.5f0, typemax(Int64), 1, mountaincar_features..., [64, 64])
	@profview dp_λ_fcann(mountaincar_mdp, 1f0, 0.5f0, typemax(Int64), 10_000, mountaincar_features..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 47bfd881-aa39-479e-8dba-5367411cf5ca
#=╠═╡
begin
	@profview dp_λ_linear(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 1, dense_feature_setup...)
	@profview dp_λ_linear(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 100_000, dense_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ 23530928-97f4-4401-b03c-d7f965d00584
#=╠═╡
begin
	@profview dp_λ_fcann(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 1, dense_feature_setup..., [64, 64])
	@profview dp_λ_fcann(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 10_000, dense_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 14ae22db-d08b-4dff-a7ef-9a8ed6bed3ec
#=╠═╡
begin
	@profview dp_λ_linear(mountaincar_continuing, 0.5f0, 1, mountaincar_features...)
	@profview dp_λ_linear(mountaincar_continuing, 0.5f0, 1_000_000, mountaincar_features...)
end
  ╠═╡ =#

# ╔═╡ dddf0de5-71b8-4cd9-bf5f-f28ea877cb43
#=╠═╡
begin
	@profview dp_λ_fcann(mountaincar_continuing, 0.5f0, 1, mountaincar_features..., [64, 64])
	@profview dp_λ_fcann(mountaincar_continuing, 0.5f0, 10_000, mountaincar_features..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 10e9add3-f59c-4428-8528-46be159c1087
true_online_sarsa_λ(mountaincar_mdp, 1f0, 0.5f0, typemax(Int64), 100_000, mountaincar_features...)

# ╔═╡ 2b2e2b52-7d2b-41a2-a42a-b261cf2fad63
#=╠═╡
begin
	@profview true_online_sarsa_λ(mountaincar_mdp, 1f0, 0.5f0, typemax(Int64), 1, mountaincar_features...)
	@profview true_online_sarsa_λ(mountaincar_mdp, 1f0, 0.5f0, typemax(Int64), 100_000, mountaincar_features...)
end
  ╠═╡ =#

# ╔═╡ 2c74b5f0-ea2d-432c-b78d-f6f054796b59
#=╠═╡
begin
	@profview true_online_sarsa_λ(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 1, sparse_feature_setup...)
	@profview true_online_sarsa_λ(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 100_000, sparse_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ 682a0e87-b80a-4395-8aaf-f104e3ef11ad
#=╠═╡
begin
	@profview true_online_sarsa_λ(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 1, dense_feature_setup...)
	@profview true_online_sarsa_λ(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 100_000, dense_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ 21a4ba91-b2c9-492b-acab-58072572d5a5
#=╠═╡
begin
	@profview true_online_dp_λ(mountaincar_mdp, 1f0, 0.5f0, typemax(Int64), 1, mountaincar_features...)
	@profview true_online_dp_λ(mountaincar_mdp, 1f0, 0.5f0, typemax(Int64), 100_000, mountaincar_features...)
end
  ╠═╡ =#

# ╔═╡ 43d327de-84e6-4c01-9cb0-b7956b2c2002
#=╠═╡
begin
	@profview true_online_dp_λ(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 1, sparse_feature_setup...)
	@profview true_online_dp_λ(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 100_000, sparse_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ 7db780e4-bd31-47da-abb4-f1bc954a5282
#=╠═╡
begin
	@profview true_online_dp_λ(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 1, dense_feature_setup...)
	@profview true_online_dp_λ(state_mdp_stochastic, 1f0, 0.5f0, typemax(Int64), 100_000, dense_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ 8e00e381-f20d-426a-86bf-aee76b4673eb
reinforce_monte_carlo_control_linear(state_mdp_stochastic, 0.99f0, 100, sparse_feature_setup...)

# ╔═╡ 4da33e35-5268-4b4e-ab98-1787e1b10ecc
#=╠═╡
begin
	@profview reinforce_monte_carlo_control_linear(state_mdp_stochastic, 0.99f0, 1, sparse_feature_setup...)
	@profview reinforce_monte_carlo_control_linear(state_mdp_stochastic, 0.99f0, 100, sparse_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ b9ea0197-1f4d-4414-b47c-d8ae099132ef
reinforce_monte_carlo_control_fcann(state_mdp_stochastic, 0.99f0, 10, sparse_feature_setup..., [64, 64])

# ╔═╡ e52570ee-5f3f-417c-80f4-36af114a142b
#=╠═╡
begin
	@profview reinforce_monte_carlo_control_fcann(state_mdp_stochastic, 0.99f0, 1, sparse_feature_setup..., [64, 64])
	@profview reinforce_monte_carlo_control_fcann(state_mdp_stochastic, 0.99f0, 10, sparse_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ c4b6b3e9-1688-43b0-ba29-62efac1b53d1
#=╠═╡
begin
	@profview reinforce_monte_carlo_control_linear(state_mdp_stochastic, 0.99f0, 1, dense_feature_setup...)
	@profview reinforce_monte_carlo_control_linear(state_mdp_stochastic, 0.99f0, 100, dense_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ 0e873fac-670d-4e56-933e-358edc81f8d9
#=╠═╡
begin
	@profview reinforce_monte_carlo_control_fcann(state_mdp_stochastic, 0.99f0, 1, dense_feature_setup..., [64, 64])
	@profview reinforce_monte_carlo_control_fcann(state_mdp_stochastic, 0.99f0, 10, dense_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 52317f1f-4fd8-4454-ab79-aa527cd61187
#=╠═╡
begin
	@profview reinforce_with_baseline_monte_carlo_control_linear(state_mdp_stochastic, 0.99f0, 1, dense_feature_setup...)
	@profview reinforce_with_baseline_monte_carlo_control_linear(state_mdp_stochastic, 0.99f0, 100, dense_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ 002935ac-aaa0-4207-ac28-af8ca9522f8e
#=╠═╡
begin
	@profview reinforce_with_baseline_monte_carlo_control_linear(state_mdp_stochastic, 0.99f0, 1, sparse_feature_setup...)
	@profview reinforce_with_baseline_monte_carlo_control_linear(state_mdp_stochastic, 0.99f0, 100, sparse_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ 86789763-03d5-49db-9061-8729930c2af4
#=╠═╡
begin
	@profview reinforce_with_baseline_monte_carlo_control_fcann(state_mdp_stochastic, 0.99f0, 1, dense_feature_setup..., [64, 64])
	@profview reinforce_with_baseline_monte_carlo_control_fcann(state_mdp_stochastic, 0.99f0, 10, dense_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ a17d557d-285c-4cba-820a-ec84ea2724b3
#=╠═╡
begin
	@profview reinforce_with_baseline_monte_carlo_control_fcann(state_mdp_stochastic, 0.99f0, 1, sparse_feature_setup..., [64, 64])
	@profview reinforce_with_baseline_monte_carlo_control_fcann(state_mdp_stochastic, 0.99f0, 10, sparse_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 40f001a0-a7f7-4006-bbf8-35e5c263f890
one_step_actor_critic_linear(mountaincar_mdp, 1f0, typemax(Int64), 100_000, mountaincar_features...; α_θ = 0.0001f0, α_w = 0.0001f0)

# ╔═╡ b113a69f-495c-47c5-9ce9-6fd0455050cc
#=╠═╡
begin
	@profview one_step_actor_critic_linear(mountaincar_mdp, 1f0, typemax(Int64), 1, mountaincar_features...; α_θ = 0.001f0, α_w = 0.001f0)
	@profview one_step_actor_critic_linear(mountaincar_mdp, 1f0, typemax(Int64), 1_000_000, mountaincar_features...; α_θ = 0.00001f0, α_w = 0.00001f0)
end
  ╠═╡ =#

# ╔═╡ d129a4ae-0708-40c8-87c3-e8868551ceb6
one_step_actor_critic_fcann(mountaincar_mdp, 1f0, typemax(Int64), 10_000, mountaincar_features..., [64, 64]; α_θ = 0.001f0, α_w = 0.001f0)

# ╔═╡ 001c7e75-0ddc-4f87-beaf-4b0cdc754019
#=╠═╡
begin
	@profview one_step_actor_critic_fcann(mountaincar_mdp, 1f0, typemax(Int64), 1, mountaincar_features..., [64, 64]; α_θ = 0.001f0, α_w = 0.001f0)
	@profview one_step_actor_critic_fcann(mountaincar_mdp, 1f0, typemax(Int64), 10_000, mountaincar_features..., [64, 64]; α_θ = 0.001f0, α_w = 0.001f0)
end
  ╠═╡ =#

# ╔═╡ 3affafae-e095-4688-b709-ea8a98b7d4d9
one_step_actor_critic_linear(state_mdp_stochastic, 1f0, typemax(Int64), 100_000, dense_feature_setup...; α_θ = 0.001f0, α_w = 0.001f0)

# ╔═╡ 58721604-e4a4-492e-87e7-bc0bc078db39
#=╠═╡
begin
	@profview one_step_actor_critic_linear(state_mdp_stochastic, 1f0, typemax(Int64), 1, dense_feature_setup...; α_θ = 0.001f0, α_w = 0.001f0)
	@profview one_step_actor_critic_linear(state_mdp_stochastic, 1f0, typemax(Int64), 100_000, dense_feature_setup...; α_θ = 0.001f0, α_w = 0.001f0)
end
  ╠═╡ =#

# ╔═╡ 124007c6-f01e-4183-b916-3b00d24428bf
one_step_actor_critic_fcann(state_mdp_stochastic, 1f0, typemax(Int64), 10_000, dense_feature_setup..., [64, 64]; α_θ = 0.001f0, α_w = 0.001f0)

# ╔═╡ 5ee5f47c-8b4b-4f03-a715-b60a0e8ff9fd
#=╠═╡
begin
	@profview one_step_actor_critic_fcann(state_mdp_stochastic, 1f0, typemax(Int64), 1, dense_feature_setup..., [64, 64]; α_θ = 0.001f0, α_w = 0.001f0)
	@profview one_step_actor_critic_fcann(state_mdp_stochastic, 1f0, typemax(Int64), 10_000, dense_feature_setup..., [64, 64]; α_θ = 0.001f0, α_w = 0.001f0)
end
  ╠═╡ =#

# ╔═╡ 10e48c8f-f11a-4291-adf4-4eee3d29d5ec
one_step_actor_critic_linear(mountaincar_continuing, 1_000_000, mountaincar_features...; α_θ = 0.0001f0, α_w = 0.0001f0)

# ╔═╡ 2a6fe0c4-9362-4e4c-8c3a-d92c4b93ecb4
#=╠═╡
begin
	@profview one_step_actor_critic_linear(mountaincar_continuing, 1, mountaincar_features...; α_θ = 0.0001f0, α_w = 0.0001f0)
	@profview one_step_actor_critic_linear(mountaincar_continuing, 1_000_000, mountaincar_features...; α_θ = 0.0001f0, α_w = 0.0001f0)
end
  ╠═╡ =#

# ╔═╡ b2da6f92-b4df-42f7-acb9-cd40b4b3eb9f
#=╠═╡
begin
	@profview one_step_actor_critic_fcann(mountaincar_continuing, 1, mountaincar_features..., [64, 64]; α_θ = 0.0001f0, α_w = 0.0001f0)
	@profview one_step_actor_critic_fcann(mountaincar_continuing, 10_000, mountaincar_features..., [64, 64]; α_θ = 0.0001f0, α_w = 0.0001f0)
end
  ╠═╡ =#

# ╔═╡ 1f928153-7e87-4e6d-887d-2110dff64bd6
one_step_actor_critic_linear(state_mdp_continuing, 100_000, dense_feature_setup...; α_θ = 0.001f0, α_w = 0.001f0)

# ╔═╡ f1e2c1a4-1cc8-4ecf-b32f-2e82150ae23a
#=╠═╡
begin
	@profview one_step_actor_critic_linear(state_mdp_continuing, 1, dense_feature_setup...; α_θ = 0.001f0, α_w = 0.001f0)
	@profview one_step_actor_critic_linear(state_mdp_continuing, 100_000, dense_feature_setup...; α_θ = 0.001f0, α_w = 0.001f0)
end
  ╠═╡ =#

# ╔═╡ ef67dd87-020e-4ecf-8343-e3a69cbb8996
#=╠═╡
begin
	@profview one_step_actor_critic_fcann(state_mdp_continuing, 1, dense_feature_setup..., [64, 64]; α_θ = 0.001f0, α_w = 0.001f0)
	@profview one_step_actor_critic_fcann(state_mdp_continuing, 10_000, dense_feature_setup..., [64, 64]; α_θ = 0.001f0, α_w = 0.001f0)
end
  ╠═╡ =#

# ╔═╡ 6b2aa205-efd9-4442-8319-d0155f91f7a6
actor_critic_with_eligibility_traces_linear(mountaincar_mdp, 1f0, 0.5f0, 0.5f0, typemax(Int64), 100_000, mountaincar_features...; α_θ = 0.0001f0, α_w = 0.0001f0)

# ╔═╡ b93d3a77-da87-4f01-8112-01182492e9c6
#=╠═╡
begin
	@profview actor_critic_with_eligibility_traces_linear(mountaincar_mdp, 1f0, 0.5f0, 0.5f0, typemax(Int64), 1, mountaincar_features...; α_θ = 0.0001f0, α_w = 0.0001f0)
	@profview actor_critic_with_eligibility_traces_linear(mountaincar_mdp, 1f0, 0.5f0, 0.5f0, typemax(Int64), 100_000, mountaincar_features...; α_θ = 0.0001f0, α_w = 0.0001f0)
end
  ╠═╡ =#

# ╔═╡ c8d856f6-398b-4af5-840a-a75e7b1fdccf
actor_critic_with_eligibility_traces_fcann(mountaincar_mdp, 1f0, 0.5f0, 0.5f0, typemax(Int64), 10_000, mountaincar_features..., [64, 64]; α_θ = 0.0001f0, α_w = 0.0001f0)

# ╔═╡ b9233288-6d6c-4513-bccc-67eead92aa8b
#=╠═╡
begin
	@profview actor_critic_with_eligibility_traces_fcann(mountaincar_mdp, 1f0, 0.5f0, 0.5f0, typemax(Int64), 1, mountaincar_features..., [64, 64]; α_θ = 0.0001f0, α_w = 0.0001f0)
	@profview actor_critic_with_eligibility_traces_fcann(mountaincar_mdp, 1f0, 0.5f0, 0.5f0, typemax(Int64), 10_000, mountaincar_features..., [64, 64]; α_θ = 0.0001f0, α_w = 0.0001f0)
end
  ╠═╡ =#

# ╔═╡ 8dbaf437-d86a-453a-abe8-88c3443be7b4
actor_critic_with_eligibility_traces_linear(mountaincar_continuing, 0.5f0, 0.5f0, 100_000, mountaincar_features...)

# ╔═╡ 71054dbc-cc2d-4a46-b72a-8f3f2ceae0c6
#=╠═╡
begin
	@profview actor_critic_with_eligibility_traces_linear(mountaincar_continuing, 0.5f0, 0.5f0, 1, mountaincar_features...)
	@profview actor_critic_with_eligibility_traces_linear(mountaincar_continuing, 0.5f0, 0.5f0, 100_000, mountaincar_features...)
end
  ╠═╡ =#

# ╔═╡ 85caf656-437b-4446-a44b-58b659d9ee23
#=╠═╡
begin
	@profview actor_critic_with_eligibility_traces_fcann(mountaincar_continuing, 0.5f0, 0.5f0, 1, mountaincar_features..., [64, 64])
	@profview actor_critic_with_eligibility_traces_fcann(mountaincar_continuing, 0.5f0, 0.5f0, 10_000, mountaincar_features..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ 4c28a58f-66df-4798-9327-d2ee11d3ee0f
actor_critic_with_eligibility_traces_linear(state_mdp_stochastic, 1f0, 0.5f0, 0.5f0, typemax(Int64), 100_000, dense_feature_setup...; α_θ = 0.001f0, α_w = 0.001f0)

# ╔═╡ a939a648-2fec-47a4-bb74-158149e7437a
#=╠═╡
begin
	@profview actor_critic_with_eligibility_traces_linear(state_mdp_stochastic, 1f0, 0.5f0, 0.5f0, typemax(Int64), 1, dense_feature_setup...; α_θ = 0.001f0, α_w = 0.001f0)
	@profview actor_critic_with_eligibility_traces_linear(state_mdp_stochastic, 1f0, 0.5f0, 0.5f0, typemax(Int64), 100_000, dense_feature_setup...; α_θ = 0.001f0, α_w = 0.001f0)
end
  ╠═╡ =#

# ╔═╡ efa52f3f-bd34-4aed-a011-4ce0166ba6ca
actor_critic_with_eligibility_traces_fcann(state_mdp_stochastic, 1f0, 0.5f0, 0.5f0, typemax(Int64), 1, dense_feature_setup..., [64, 64]; α_θ = 0.001f0, α_w = 0.001f0, use_gpu = true)

# ╔═╡ cb66981e-73a9-4185-8af2-2967bdece45f
#=╠═╡
begin
	@profview actor_critic_with_eligibility_traces_fcann(state_mdp_stochastic, 1f0, 0.5f0, 0.5f0, typemax(Int64), 1, dense_feature_setup..., [64, 64]; α_θ = 0.001f0, α_w = 0.001f0)
	@profview actor_critic_with_eligibility_traces_fcann(state_mdp_stochastic, 1f0, 0.5f0, 0.5f0, typemax(Int64), 10_000, dense_feature_setup..., [64, 64]; α_θ = 0.001f0, α_w = 0.001f0)
end
  ╠═╡ =#

# ╔═╡ 6c0329b1-a7cc-4865-94a8-4381189a16d0
#=╠═╡
begin
	@profview actor_critic_with_eligibility_traces_linear(state_mdp_continuing, 0.5f0, 0.5f0, 1, dense_feature_setup...)
	@profview actor_critic_with_eligibility_traces_linear(state_mdp_continuing, 0.5f0, 0.5f0, 100_000, dense_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ f1343320-1954-4303-92bf-277dea840e91
#=╠═╡
begin
	@profview actor_critic_with_eligibility_traces_fcann(state_mdp_continuing, 0.5f0, 0.5f0, 1, dense_feature_setup..., [64, 64]; α_θ = 0.001f0, α_w = 0.001f0)
	@profview actor_critic_with_eligibility_traces_fcann(state_mdp_continuing, 0.5f0, 0.5f0, 10_000, dense_feature_setup..., [64, 64]; α_θ = 0.001f0, α_w = 0.001f0)
end
  ╠═╡ =#

# ╔═╡ 085a977e-1e16-49d7-a9a3-46ec12e586fd
dqn_linear(mountaincar_mdp, 1f0, typemax(Int64), 100_000, mountaincar_features...; batch_size = 64)

# ╔═╡ 3038316c-5d5a-42e6-aa19-561f618956df
#=╠═╡
begin
	@profview dqn_linear(mountaincar_mdp, 1f0, typemax(Int64), 1, mountaincar_features...; batch_size = 64)
	@profview dqn_linear(mountaincar_mdp, 1f0, typemax(Int64), 40_000, mountaincar_features...; batch_size = 64)
end
  ╠═╡ =#

# ╔═╡ 17138683-cd80-46f9-9f60-3a5bd70da7d9
dqn_fcann(mountaincar_mdp, 1f0, typemax(Int64), 100, mountaincar_features..., [64, 64]; batch_size = 64)

# ╔═╡ 6fad8123-8a3e-4cd2-87af-d2462e3f495d
#=╠═╡
begin
	@profview dqn_fcann(mountaincar_mdp, 1f0, typemax(Int64), 1, mountaincar_features..., [64, 64]; batch_size = 64)
	@profview dqn_fcann(mountaincar_mdp, 1f0, typemax(Int64), 100, mountaincar_features..., [64, 64]; batch_size = 64)
end
  ╠═╡ =#

# ╔═╡ 33969258-23fa-4f13-a4c6-8126759d0e69
dqn_linear(state_mdp_stochastic, 1f0, typemax(Int64), 10_000, sparse_feature_setup...; batch_size = 64)

# ╔═╡ 10c2fc1c-6dc3-4087-8cfa-6377ed126998
#=╠═╡
begin
	@profview dqn_linear(state_mdp_stochastic, 1f0, typemax(Int64), 1, sparse_feature_setup...; batch_size = 64)
	@profview dqn_linear(state_mdp_stochastic, 1f0, typemax(Int64), 10_000, sparse_feature_setup...; batch_size = 64)
end
  ╠═╡ =#

# ╔═╡ f7eaf74c-716c-4f1f-977f-548a30532512
dqn_fcann(state_mdp_stochastic, 1f0, typemax(Int64), 1_000, sparse_feature_setup..., [64, 64]; batch_size = 64)

# ╔═╡ 98e1e398-04cc-4de5-b47e-09b4e8951438
#=╠═╡
begin
	@profview dqn_fcann(state_mdp_stochastic, 1f0, typemax(Int64), 1, sparse_feature_setup..., [64, 64]; batch_size = 64)
	@profview dqn_fcann(state_mdp_stochastic, 1f0, typemax(Int64), 1_000, sparse_feature_setup..., [64, 64]; batch_size = 64)
end
  ╠═╡ =#

# ╔═╡ 8465aaa3-4fb3-4654-b237-27f744907fd5
#=╠═╡
begin
	@profview dqn_linear(state_mdp_stochastic, 1f0, typemax(Int64), 100, dense_feature_setup...; batch_size = 64)
	@profview dqn_linear(state_mdp_stochastic, 1f0, typemax(Int64), 10_000, dense_feature_setup...; batch_size = 64)
end
  ╠═╡ =#

# ╔═╡ 4a60fb05-473a-4800-bf89-304e0203bf78
#=╠═╡
begin
	@profview dqn_fcann(state_mdp_stochastic, 1f0, typemax(Int64), 1, dense_feature_setup..., [64, 64]; batch_size = 64)
	@profview dqn_fcann(state_mdp_stochastic, 1f0, typemax(Int64), 10_000, dense_feature_setup..., [64, 64]; batch_size = 64)
end
  ╠═╡ =#

# ╔═╡ bfe2883a-1566-419a-a360-be370d583608
synchronous_actor_critic_linear(mountaincar_mdp, 1f0, 100_000, 8, mountaincar_features...)

# ╔═╡ 47353eb3-6993-4cb1-9f39-0e4d26678a74
synchronous_nstep_actor_critic_linear(mountaincar_mdp, 1f0, 100_000, 8, mountaincar_features...; N = 10)

# ╔═╡ 2f996cfe-a8dc-4eb2-8fd6-697ad55feabf
#=╠═╡
begin
	@profview synchronous_actor_critic_linear(mountaincar_mdp, 1f0, 100, 8, mountaincar_features...)
	@profview synchronous_actor_critic_linear(mountaincar_mdp, 1f0, 100_000, 8, mountaincar_features...)
end
  ╠═╡ =#

# ╔═╡ 38c62051-9027-4157-a544-06c968115324
#=╠═╡
begin
	@profview synchronous_nstep_actor_critic_linear(mountaincar_mdp, 1f0, 100, 8, mountaincar_features...; N = 10)
	@profview synchronous_nstep_actor_critic_linear(mountaincar_mdp, 1f0, 100_000, 8, mountaincar_features...; N = 10)
end
  ╠═╡ =#

# ╔═╡ 1662c3fd-343b-4b63-b9ca-3f4cc9df061b
synchronous_actor_critic_fcann(mountaincar_mdp, 1f0, 10_000, 8, mountaincar_features..., [64, 64]; α_θ = 0.0001f0, α_w = 0.0001f0)

# ╔═╡ 3186e8a7-021b-47f4-bb6c-1084a0ca2271
synchronous_nstep_actor_critic_fcann(mountaincar_mdp, 1f0, 1_000, 8, mountaincar_features..., [64, 64]; α_θ = 0.0001f0, α_w = 0.0001f0, N = 10)

# ╔═╡ dc5b027d-d476-4c4f-8a72-a24251e42eb6
#=╠═╡
begin
	@profview synchronous_actor_critic_fcann(mountaincar_mdp, 1f0, 1, 8, mountaincar_features..., [64, 64]; α_θ = 0.0001f0, α_w = 0.0001f0)
	@profview synchronous_actor_critic_fcann(mountaincar_mdp, 1f0, 10_000, 8, mountaincar_features..., [64, 64]; α_θ = 0.0001f0, α_w = 0.0001f0)
end
  ╠═╡ =#

# ╔═╡ 728d6b33-5fbb-48f5-a610-23085e90c909
#=╠═╡
begin
	@profview synchronous_nstep_actor_critic_fcann(mountaincar_mdp, 1f0, 1, 8, mountaincar_features..., [64, 64]; α_θ = 0.0001f0, α_w = 0.0001f0, N = 10)
	@profview synchronous_nstep_actor_critic_fcann(mountaincar_mdp, 1f0, 1_000, 8, mountaincar_features..., [64, 64]; α_θ = 0.0001f0, α_w = 0.0001f0, N = 10)
end
  ╠═╡ =#

# ╔═╡ 39a8069f-6bbe-4d52-9cba-1d94d940c6f5
synchronous_actor_critic_linear(state_mdp_stochastic, 1f0, 10_000, 8, dense_feature_setup...)

# ╔═╡ 387884df-cfda-4217-bbd6-e0a76434489e
#=╠═╡
begin
	@profview synchronous_actor_critic_linear(state_mdp_stochastic, 1f0, 1, 8, dense_feature_setup...)
	@profview synchronous_actor_critic_linear(state_mdp_stochastic, 1f0, 10_000, 8, dense_feature_setup...)
end
  ╠═╡ =#

# ╔═╡ cd8a8732-e579-47a3-875b-42d62be46b29
#=╠═╡
begin
	@profview synchronous_nstep_actor_critic_linear(state_mdp_stochastic, 1f0, 1, 8, dense_feature_setup...; N = 10)
	@profview synchronous_nstep_actor_critic_linear(state_mdp_stochastic, 1f0, 10_000, 8, dense_feature_setup...; N = 10)
end
  ╠═╡ =#

# ╔═╡ 3a674ea7-0d01-4950-992f-968e57eb5bd8
synchronous_actor_critic_fcann(state_mdp_stochastic, 1f0, 10_000, 8, dense_feature_setup..., [64, 64])

# ╔═╡ ca366c95-098f-4d08-9929-9da48f84feca
synchronous_actor_critic_fcann(state_mdp_stochastic, 1f0, 1_000, 8, dense_feature_setup..., [64, 64]; use_gpu = true)

# ╔═╡ a836140b-378c-4a8a-8d4d-9cb0757ab3f3
#=╠═╡
begin
	@profview synchronous_actor_critic_fcann(state_mdp_stochastic, 1f0, 1, 8, dense_feature_setup..., [64, 64])
	@profview synchronous_actor_critic_fcann(state_mdp_stochastic, 1f0, 10_000, 8, dense_feature_setup..., [64, 64])
end
  ╠═╡ =#

# ╔═╡ c3877f0d-5171-4ca4-befc-4f3263213e06
#=╠═╡
begin
	@profview synchronous_nstep_actor_critic_fcann(state_mdp_stochastic, 1f0, 1, 8, dense_feature_setup..., [64, 64]; N = 10)
	@profview synchronous_nstep_actor_critic_fcann(state_mdp_stochastic, 1f0, 10_000, 8, dense_feature_setup..., [64, 64]; N = 10)
end
  ╠═╡ =#

# ╔═╡ 33751e9d-9134-4ba4-a249-89dd63cf8078
# ╠═╡ skip_as_script = true
#=╠═╡
html"""
<style>
	main {
		margin: 0 auto;
		max-width: min(1600px, 90%);
		padding-left: max(50px, 5%);
		padding-right: max(200px, 5%);
		font-size: max(10px, min(24px, 2vw));
	}
</style>
"""
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProfileCanvas = "efd6af41-a80b-495e-886c-e51b0c7d77a3"

[compat]
BenchmarkTools = "~1.8.0"
HypertextLiteral = "~1.0.0"
LaTeXStrings = "~1.4.1"
PlutoDevMacros = "~0.9.2"
PlutoPlotly = "~0.6.6"
PlutoUI = "~0.7.83"
ProfileCanvas = "~0.1.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.0"
manifest_format = "2.1"
project_hash = "0541ab5aafb5779fe45de2a6c88a720c1b5da587"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
registries = "General"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

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
deps = ["Compat", "JSON", "Logging", "PrecompileTools", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "9670d3febc2b6da60a0ae57846ba74670290653f"
registries = "General"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.8.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "REPL", "UUIDs"]
git-tree-sha1 = "cfb7a2e89e245a9d5016b70323db412b3a7438d5"
registries = "General"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
registries = "General"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
registries = "General"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
registries = "General"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
registries = "General"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
registries = "General"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
registries = "General"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
registries = "General"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
registries = "General"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
registries = "General"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "58927c485919bf17ea308d9d82156de1adf4b006"
registries = "General"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.12"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
registries = "General"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "1.0.0"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "Zstd_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.18.0+1"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.1+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "OpenSSL_jll", "Zlib_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.103+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.13.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
registries = "General"
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
version = "2026.8.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.46.0+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
registries = "General"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "ba0dc8a8a67cacac4842631f960c046e4e563675"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.8"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "Zstd_jll", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.13.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "6256ab3ee24ef079b3afa310593817e069925eeb"
registries = "General"
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
registries = "General"
uuid = "a0499f29-c39b-4c5c-807c-88074221b949"
version = "0.9.2"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "2b9e3d771adfe535a4fdda855f4741fdaacd3f7f"
registries = "General"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.6"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
registries = "General"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
registries = "General"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
registries = "General"
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
deps = ["Base64", "JSON", "Pkg", "Profile", "REPL"]
git-tree-sha1 = "990016fb1508b0726a70039f39569720d054c78d"
registries = "General"
uuid = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
version = "0.1.7"

[[deps.REPL]]
deps = ["Base64", "Dates", "FileWatching", "InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
registries = "General"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
registries = "General"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
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
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
registries = "General"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"

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
registries = "General"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
registries = "General"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "908fec9df6c5de98548ead82a468c95ccf6cd263"
registries = "General"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
registries = "General"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl"]
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.67.1+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.8.2+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╟─bf7a594f-7aef-4f8c-ad93-8f254c2dad3f
# ╟─27b818f8-b6cb-4d26-b0df-a9a7d277351c
# ╠═7255a15c-9e00-48a5-a83c-ef88f127d52e
# ╟─4041b1d0-7765-438f-b5ce-e7a5bcce3045
# ╟─69264750-07fd-4fb6-bb10-7f968a56f772
# ╠═0e68d19c-c077-47e9-b396-e8395b4cebc6
# ╠═15ffc73e-f883-48b7-bb95-ae2428c7839e
# ╠═88cb461d-d332-4f8a-ac39-6c00defb17a2
# ╠═dc87916c-e528-4095-980c-be82b4ffff43
# ╠═8f1f7c38-7789-4104-af6f-798173aa0d3f
# ╟─704f0549-d227-4a3a-b78e-d56c6d16e613
# ╟─2be43ef3-c3f5-4d02-8e18-9a5f55d05bee
# ╠═27d4015b-817d-4724-b712-1dd33990cc80
# ╠═731987c4-4335-4e39-827d-74d1b17351a1
# ╠═ac3d9473-3734-49a0-9e01-360eb423a6f0
# ╠═a741f0b6-a472-4f81-aeaa-1000251bf448
# ╠═cf43f9bf-4e01-4eb7-8c71-d7be657ad746
# ╠═86bb7340-2df3-40b5-b5a8-ee6a0881cc98
# ╠═a7569534-e573-45c9-b129-0b45940d9409
# ╠═3630121c-ef7e-4f70-812b-62e28ed24f05
# ╠═fb2b1149-dd76-4ff5-a25c-c1aa435e6f09
# ╠═87c0880f-87d5-4566-8b4c-694b25ef864c
# ╠═62b42256-95f8-4e19-b7a9-1e25e7ad30c4
# ╠═082f4f1a-6302-4959-aad2-9a561d91362f
# ╠═27d555ce-3b4b-453f-8f7c-65644fe42c8c
# ╠═90b9f7b4-032d-40a5-b0c5-fb3aab6e8231
# ╟─a2815932-6ce4-479a-a152-74ed6d38f2f2
# ╠═f35b4530-4099-4016-8b73-4aa615dce58e
# ╠═2d4b5fa3-5404-4901-a6bf-69ce0a511994
# ╠═40df5a8d-0480-42a8-8415-0f48228cb8f7
# ╠═8111ca1e-2c57-497a-9476-c91359759c9a
# ╟─1f1aa1e5-b10a-4831-8c5e-699a10171617
# ╠═27f441f9-7fcb-4a83-893f-c403aa9a7868
# ╠═65b427b6-fed6-4cba-acad-aa8cad443d28
# ╠═e6c21fbe-cd94-4562-99ef-933651af002f
# ╠═ed1675a9-7788-4ca8-864f-131d9acb3ce7
# ╟─8200bb1c-787c-473c-a095-5a52d0bd0b11
# ╟─3bfaa827-923a-405e-8f12-ca37ceab590a
# ╠═54cd9005-5330-4e24-963c-767e39309d9c
# ╠═f31270cf-038d-454d-8c39-79b56cda992d
# ╠═4a93e565-56e2-424d-a614-c5aa8c78f2f1
# ╠═87355b4b-0868-4e07-a13e-de253cc69a95
# ╠═3bd283c9-f413-4a08-a030-103eb7d186a4
# ╠═ac52d6f5-9db0-4095-8301-0bf8d6116cae
# ╠═b5271864-a42e-4ecf-b5b2-dfa45e0fd7a6
# ╠═5789ae84-8cb0-4bf2-8de5-9302e321cb07
# ╟─dc0d2f22-2646-4249-99f8-ae5d00506527
# ╠═0702027f-49dd-465c-9f40-b0ce52f6d8c4
# ╠═8561fce9-c455-4253-a76c-a8c9845b53d5
# ╠═84b4c5be-62ac-4c4b-9a7d-a61a8f7b7d87
# ╠═fb665ee2-51f7-4942-a5b1-39b1e73d81a5
# ╟─24975ce1-40db-4fa2-94e3-ab09ff7fe86e
# ╟─f4c7fc77-0900-44b1-aad8-b1c78bcb56f0
# ╠═42790259-bdc5-4962-b13f-e781095671ae
# ╠═63c8aeab-9d63-472b-a475-d158dba6f126
# ╠═ed114798-c20e-48ec-8a7a-d69a6327a047
# ╠═db52630a-9c96-4974-afc8-c3f717cca604
# ╟─f5a2747a-29f1-41e5-8d99-c5c51050721d
# ╠═beeab34c-339b-4cb3-9998-713a139ad843
# ╠═3a09fd58-bee6-4f89-8c65-cd0e9248e8cb
# ╠═9e0f227a-b3a6-410a-ab28-92a174e12337
# ╠═832351b7-f515-4f66-991f-09e1c214462b
# ╠═78317a12-091c-41ee-b529-238ddd7b7361
# ╠═91f5ac83-78e7-4015-bffe-7cee8cb1f8d6
# ╠═12d0471a-9f92-44a0-8af3-0e0a6f4c9ef1
# ╠═4267be90-7bce-4709-883c-7bbff85e6471
# ╠═2cad4cba-7081-4e87-a66e-db072ff9eb41
# ╠═7ec8c174-f08c-464b-adc8-11b98883e44a
# ╠═66388b8e-f206-4222-bd96-52c1b9a10127
# ╠═b185599c-005e-472a-8a9e-1c06964d79aa
# ╠═96271cbc-1ed7-4db4-a2e9-4d2cf629c414
# ╟─d97c9506-aea1-43b4-a282-81418df894ef
# ╠═55062ba3-870e-497d-a7ff-fec73abc3b20
# ╠═01e9cb83-ce4e-476d-8791-5ce451b24cd1
# ╠═a66048e9-cbfc-403c-b09e-f653fd93c4cd
# ╠═3cab6d8c-7aeb-412d-9acf-b8bc6097d160
# ╠═bbb4a031-2f56-43b5-9c71-afe327e95f27
# ╠═0dfae082-d8f2-4b7f-8fcc-57e6ce9bdc28
# ╠═1edfdab0-df23-4e1c-acb5-9f22e35642a1
# ╠═75294323-3a3a-4579-8657-c074ea608cfb
# ╠═3e198fe0-3552-4a2b-8758-c00b9dcc5516
# ╠═d72b933f-8ced-4aaf-9014-3ee64e27fe2a
# ╠═43f7b3cb-e071-4185-aef8-369e423d9dc6
# ╠═84d72414-f1a4-4701-be6d-4342e22461ae
# ╠═1de02aeb-e36d-4357-adba-3e069c61ebfa
# ╠═003928c2-99d6-40df-9067-79bc51ec80b5
# ╟─d7052ba5-f0ef-4350-befc-0b2458af8c60
# ╟─04b6f612-c5aa-49a0-ac83-702f21c17774
# ╠═889a5212-5336-4ccf-ab3b-693759e1afb3
# ╠═57a307b5-2aed-48e9-90e2-d3d7550231d9
# ╠═67903e58-e7b2-4556-b551-40eefd9ec203
# ╠═e20df14f-8eff-43f3-a4c2-8793ec788b82
# ╠═b337d2d4-d3dc-4ec4-b9d8-96a28cbb5863
# ╠═4f3909ba-96ae-43b9-bd6c-599e54e48b75
# ╟─3076c786-db6d-4573-99fe-35540ee171c5
# ╠═54a0a47f-7dd8-47f8-8f55-86540bb05d3d
# ╠═d6f3f0d7-c58e-4101-a70f-7977c60c1821
# ╠═51e3970d-87da-4be9-a452-faaff294465b
# ╟─b3da9525-bf8e-4761-a518-ce0e7292aaee
# ╠═fbb60ce0-6fc3-4278-bf46-9dca5a30a6dd
# ╠═f7acbbcd-134a-47e4-b17f-39c402bb601d
# ╠═357190d3-e84e-4e1b-8ff6-e6f303528584
# ╠═0867de49-1960-4cf1-9f1d-409ad0e05793
# ╠═2f3ebcbb-e82b-4ab2-88a4-24ed6fb2a448
# ╠═c953683e-fc9b-4eea-bf17-f24566ee99f9
# ╠═a4abe5ca-8f5f-473a-a7f4-8710e3ce7ec1
# ╠═e5be5928-e5b8-4690-acb3-fe5855634e5d
# ╠═91c8ed60-9c3a-4a17-9883-d31bf6314c60
# ╠═8eb6c31a-8500-4d69-907a-5daa741fbb29
# ╟─740834f5-5fa7-4a8f-9c8b-77e1bece4f0e
# ╟─c63c0a29-5604-4d5f-a9e8-2733ea8c5865
# ╠═fc1175c7-9275-4b12-8718-c05d5d4634b4
# ╠═678cda5d-66fa-4a1a-a115-41e16d5555d4
# ╠═147e3bd2-29f2-4a78-a3fd-4f3531afc3d5
# ╠═21b5e82d-fdda-4caf-a1aa-222152b69f8b
# ╠═ac1e6eeb-fcd1-4bad-a46a-edc9a9d17c1f
# ╠═beafd867-c4c3-4b97-99a8-c8dc3d591341
# ╠═affba263-c895-4d8f-af32-2c561a884d49
# ╠═16ad1491-85db-4b1a-a451-421202d079b2
# ╠═1a45053e-1732-4dbd-9682-d34f60579e74
# ╠═3b5d759c-8085-4912-9646-fe45087d3d9f
# ╠═4a290372-be64-4e45-a96a-01968b57aacf
# ╠═c10aba9e-daf4-4767-9e14-51cd06ad5495
# ╟─9a84b6e7-aaf4-4c84-ba03-c6e9d1fecf7f
# ╠═b00e0fe1-cdc2-4d28-946a-a7d256ab7f49
# ╠═46b49f4a-1375-4286-a1a9-e86b4b8e2c63
# ╠═f7259f58-f9c2-43c7-8173-e01a2a2683b0
# ╠═185ec9b4-5c0f-4400-8ca6-623a24a975f5
# ╠═a7127d77-ca72-448d-aa8f-8e25eeee547a
# ╠═340ef2f9-889b-4e97-8bd5-66e6f1d174cd
# ╠═205a88eb-ff68-46f0-9f3c-650d9e83a401
# ╠═109d913a-de49-4b2b-ab38-fa44b80ec178
# ╠═a08cca6f-71ba-4c27-bae3-7c361d6a7684
# ╠═0ed2e466-10ec-4cb4-9c95-329e72da55aa
# ╠═e162133d-3c89-45fe-bd9f-c7f04df1038f
# ╠═81acfb57-9863-4f26-bc70-2181640fbc77
# ╟─807e0eef-2811-4288-8a62-65119c35f093
# ╠═17d49653-118b-4865-a9cc-197981889bee
# ╠═e6ec3fdf-e3a0-4d38-8791-0d99de9e23eb
# ╠═0c1eefae-b73b-4d2b-83f2-0a605072f363
# ╠═cdb3b561-afaf-4a8e-8ee4-6b8ae8798de8
# ╠═e7c43fa3-af1b-46cf-abab-e34ff3637c8b
# ╠═9eabf565-ba4d-4d7e-be25-626d7a7894c8
# ╠═a855e694-f802-42e8-82d4-4a9b48e1e65b
# ╠═a4462943-4187-4556-81a4-18a90e3373a7
# ╠═08408cc9-c964-49f3-9811-3399c463ca30
# ╠═488bed89-cfd6-465f-9317-d9bb9a2123ba
# ╠═2281fc10-27b0-4cd4-bd1b-11545e96d1e6
# ╠═d925e225-ce8b-46f0-97b3-2c267968cb34
# ╟─b79ca0ac-6386-4f8f-b74a-811e53229eea
# ╠═9e9c511a-b07a-4973-a0ff-147c5a5da73d
# ╠═f9948d01-ce46-4911-bc6a-12a1728f769e
# ╠═2bc3a54d-e1da-45ef-b724-2fa0c97b9823
# ╠═2dd2d7b0-11a4-439c-823e-dcd59cd6badc
# ╠═9722bbc8-108b-4fc3-990c-84e1eeedcf24
# ╠═4ba4e546-441b-4632-9d63-7a9010dc4ae6
# ╠═87f35c6f-a9c4-4288-a5f4-f7a169c9df84
# ╠═aefe689f-c043-4bf0-91f6-b5f5c23f11fa
# ╟─b509a680-e2d2-4a37-91ca-4cf7d4b43970
# ╟─dfe5108d-1488-4e4e-9a8f-902951af993a
# ╟─d74c0f55-023c-4939-aecc-4d6467001dce
# ╠═5e6c8504-875a-497f-b82e-e09e7f0eba2e
# ╠═716e0e85-53ee-4415-8c5e-ec5f5581a93f
# ╠═53305264-bb8b-4061-873c-9af7f79491f4
# ╠═803720bd-72b2-4f89-9c97-100cc3d4aa98
# ╟─ebc7fafa-9e31-4f06-8a56-762472f878ee
# ╠═b13cd00e-0330-4eca-8970-52ddaa2847a8
# ╠═5ca8cc42-7ab5-42d0-b74a-c1933fad3a36
# ╠═0261cf11-25bd-430e-8b3b-f3295ff88921
# ╠═32c6b199-e334-43f9-a0ef-daf77964ec0e
# ╟─01ecf555-af75-4c56-b667-4754ab588b9f
# ╠═a07e5b9b-a36a-42d0-ba68-21a83444de82
# ╠═cd5d18d5-e69e-4b82-9c5d-fceab0906e93
# ╠═25527ecb-0e92-4535-9261-6edeb4ea28cf
# ╠═8b3f9720-46b9-49a9-81a7-7fa97351a3db
# ╟─239f8a7a-1cf6-42ad-b8fe-3f77896c5240
# ╠═080a9c27-20f0-4458-8156-8f5724f921a7
# ╠═0989bd35-34b7-43ec-8e5a-954a3a31968c
# ╠═5c095c88-3337-414f-9c9b-cf5cbd73fff6
# ╠═198cb57d-e466-46e9-bb0a-4bbf91cc7ff3
# ╟─5f4307fe-0c50-4488-a879-89b8d77aa846
# ╠═0c9cbd7c-0d84-4e26-81e4-54a430d31825
# ╠═e298d5a6-1ba7-4b28-ba76-f99e87f30462
# ╠═606d1f9e-b875-45d8-a9c0-2e754201901d
# ╠═7d6c0197-51f1-441d-8ed8-58d11dd83201
# ╟─a685c2b2-9cf5-42aa-8cf8-ed7c306c500a
# ╠═0ba6dc03-0a00-4668-940b-dd940d78068b
# ╠═bb837440-8cde-4a74-88ff-b80e5ebe0f41
# ╠═890990a9-0f1f-4f33-9c5b-079ec565a4f2
# ╠═7a3b441c-84ee-4dec-a000-27adc7d2fc5e
# ╟─38741ee4-928b-45e3-b845-964148a5448c
# ╟─edebeb0e-e10a-43cf-932e-b304438583e3
# ╠═9a7c1d5e-0ecd-4f1a-b2ec-0f263d90895a
# ╠═50add8c3-83b9-48a9-861a-aba64f15b130
# ╠═d983b0b0-58a3-48c9-8ad2-02c70aa56ed4
# ╠═ef9ee88d-ec94-4827-aef1-f219e2e937af
# ╟─30ed68c0-dd81-4b0f-87ef-398b4c7a160d
# ╠═73ffa88a-702c-44f7-a8c4-a22f5e18738f
# ╠═00355a66-641b-4d85-95ef-daec220a2f07
# ╠═c1389485-f630-4182-940b-3aba622a1aea
# ╠═fe31a5ca-dcec-41f6-8ee3-87bf606a13cc
# ╟─9c5e2c3c-3c9a-47c5-86e5-daea279cc3d9
# ╠═377dfd87-9166-47fd-88af-541e17f59504
# ╠═183a4cea-71ec-410e-b321-cd97aaf4cecb
# ╟─aa2dd41f-2915-4f02-9fc6-f974a5b0cb5a
# ╠═fea5b8ce-d36f-4e4a-9193-20cf2dc8a0df
# ╠═de93b46d-48c8-4dd3-93a5-1b07e69764cb
# ╠═03a7ef3d-599f-457a-8ec1-26aef821c1e6
# ╠═1a0463af-078e-4360-807e-a234e1bef07f
# ╠═37fe1f34-e18c-4197-a209-4c899b82ef21
# ╠═66917f9d-e58e-4e55-b779-7988fa5acfef
# ╟─83fbc478-a212-4a94-86a9-3a2d77798e09
# ╟─46dbcd43-8edc-45fe-8924-407926078316
# ╠═cdf997ef-1ea5-4bd5-ade9-64e7382cec40
# ╠═a78caabf-6cae-46c9-b8e9-f599905d1341
# ╠═73b5862f-fb82-4599-be8c-0c76481f728d
# ╠═da8985f7-4ba3-45cb-bc71-0d1a70739cd5
# ╟─5a52b405-6308-4d4f-afcb-10fa9a7bdd84
# ╠═47bfd881-aa39-479e-8dba-5367411cf5ca
# ╠═23530928-97f4-4401-b03c-d7f965d00584
# ╟─7b9c5a5a-7db3-4265-ba00-b608f00ae969
# ╠═14ae22db-d08b-4dff-a7ef-9a8ed6bed3ec
# ╠═dddf0de5-71b8-4cd9-bf5f-f28ea877cb43
# ╟─d79e92bb-3282-4912-a7cb-f7ef72bada4e
# ╠═10e9add3-f59c-4428-8528-46be159c1087
# ╠═2b2e2b52-7d2b-41a2-a42a-b261cf2fad63
# ╠═2c74b5f0-ea2d-432c-b78d-f6f054796b59
# ╠═682a0e87-b80a-4395-8aaf-f104e3ef11ad
# ╟─7fd6d59a-8768-4f45-a58b-681490fbf111
# ╠═21a4ba91-b2c9-492b-acab-58072572d5a5
# ╠═43d327de-84e6-4c01-9cb0-b7956b2c2002
# ╠═7db780e4-bd31-47da-abb4-f1bc954a5282
# ╟─6f453370-146c-4e3f-be21-90ca1d925976
# ╟─1bb636c9-ae1f-4c39-9403-75e954103ed9
# ╠═8e00e381-f20d-426a-86bf-aee76b4673eb
# ╠═4da33e35-5268-4b4e-ab98-1787e1b10ecc
# ╠═b9ea0197-1f4d-4414-b47c-d8ae099132ef
# ╠═e52570ee-5f3f-417c-80f4-36af114a142b
# ╠═c4b6b3e9-1688-43b0-ba29-62efac1b53d1
# ╠═0e873fac-670d-4e56-933e-358edc81f8d9
# ╠═52317f1f-4fd8-4454-ab79-aa527cd61187
# ╠═002935ac-aaa0-4207-ac28-af8ca9522f8e
# ╠═86789763-03d5-49db-9061-8729930c2af4
# ╠═a17d557d-285c-4cba-820a-ec84ea2724b3
# ╟─7d1e97e3-7e21-4dd6-b6bf-211de7933a7e
# ╟─ddb83cc4-81f2-4890-b6c6-1d26b2faa350
# ╠═40f001a0-a7f7-4006-bbf8-35e5c263f890
# ╠═b113a69f-495c-47c5-9ce9-6fd0455050cc
# ╠═d129a4ae-0708-40c8-87c3-e8868551ceb6
# ╠═001c7e75-0ddc-4f87-beaf-4b0cdc754019
# ╟─83b63252-48bd-45e9-86f9-672b43d2202e
# ╠═3affafae-e095-4688-b709-ea8a98b7d4d9
# ╠═58721604-e4a4-492e-87e7-bc0bc078db39
# ╠═124007c6-f01e-4183-b916-3b00d24428bf
# ╠═5ee5f47c-8b4b-4f03-a715-b60a0e8ff9fd
# ╟─7fbf7a3f-775e-47a1-a9dd-4b41090c229a
# ╠═10e48c8f-f11a-4291-adf4-4eee3d29d5ec
# ╠═2a6fe0c4-9362-4e4c-8c3a-d92c4b93ecb4
# ╠═b2da6f92-b4df-42f7-acb9-cd40b4b3eb9f
# ╟─b0fdf1a0-340e-440d-9859-c6d32531e4ac
# ╠═1f928153-7e87-4e6d-887d-2110dff64bd6
# ╠═f1e2c1a4-1cc8-4ecf-b32f-2e82150ae23a
# ╠═ef67dd87-020e-4ecf-8343-e3a69cbb8996
# ╟─ff0d89e2-3a3c-41d4-8c60-d27af690313d
# ╟─e5887db2-8af2-4e39-99a1-e6944b229f18
# ╠═6b2aa205-efd9-4442-8319-d0155f91f7a6
# ╠═b93d3a77-da87-4f01-8112-01182492e9c6
# ╠═c8d856f6-398b-4af5-840a-a75e7b1fdccf
# ╠═b9233288-6d6c-4513-bccc-67eead92aa8b
# ╟─450a9920-def5-4bf5-89f9-df302a7ab1c3
# ╠═8dbaf437-d86a-453a-abe8-88c3443be7b4
# ╠═71054dbc-cc2d-4a46-b72a-8f3f2ceae0c6
# ╠═85caf656-437b-4446-a44b-58b659d9ee23
# ╟─a964d032-aa9b-4559-830b-40d2096860be
# ╠═4c28a58f-66df-4798-9327-d2ee11d3ee0f
# ╠═a939a648-2fec-47a4-bb74-158149e7437a
# ╠═efa52f3f-bd34-4aed-a011-4ce0166ba6ca
# ╠═cb66981e-73a9-4185-8af2-2967bdece45f
# ╟─e3a33387-ae77-4e59-8f68-bfdedd1f86cf
# ╠═6c0329b1-a7cc-4865-94a8-4381189a16d0
# ╠═f1343320-1954-4303-92bf-277dea840e91
# ╟─e79916f5-5f11-4e5f-b76c-3bc8bc8eebc4
# ╟─3f268a8f-d67b-4412-a071-094247dd54c8
# ╟─2b67c1d2-7642-4191-813e-203fecbbda47
# ╠═085a977e-1e16-49d7-a9a3-46ec12e586fd
# ╠═3038316c-5d5a-42e6-aa19-561f618956df
# ╠═17138683-cd80-46f9-9f60-3a5bd70da7d9
# ╠═6fad8123-8a3e-4cd2-87af-d2462e3f495d
# ╟─e3edbd2e-92cc-4e77-8cc5-b546823b18d8
# ╠═33969258-23fa-4f13-a4c6-8126759d0e69
# ╠═10c2fc1c-6dc3-4087-8cfa-6377ed126998
# ╠═f7eaf74c-716c-4f1f-977f-548a30532512
# ╠═98e1e398-04cc-4de5-b47e-09b4e8951438
# ╟─ba31075c-945b-4963-91b0-a5ad223c327f
# ╠═8465aaa3-4fb3-4654-b237-27f744907fd5
# ╠═4a60fb05-473a-4800-bf89-304e0203bf78
# ╟─d53302f5-8f63-4e3c-b3de-5ade72c4b2b9
# ╟─721ded12-9bc1-43e8-82ec-c689d899d13c
# ╠═bfe2883a-1566-419a-a360-be370d583608
# ╠═47353eb3-6993-4cb1-9f39-0e4d26678a74
# ╠═2f996cfe-a8dc-4eb2-8fd6-697ad55feabf
# ╠═38c62051-9027-4157-a544-06c968115324
# ╠═1662c3fd-343b-4b63-b9ca-3f4cc9df061b
# ╠═3186e8a7-021b-47f4-bb6c-1084a0ca2271
# ╠═dc5b027d-d476-4c4f-8a72-a24251e42eb6
# ╠═728d6b33-5fbb-48f5-a610-23085e90c909
# ╟─e616a1fe-c5b3-408e-9263-aadf18cbaf84
# ╠═39a8069f-6bbe-4d52-9cba-1d94d940c6f5
# ╠═387884df-cfda-4217-bbd6-e0a76434489e
# ╠═cd8a8732-e579-47a3-875b-42d62be46b29
# ╠═3a674ea7-0d01-4950-992f-968e57eb5bd8
# ╠═ca366c95-098f-4d08-9929-9da48f84feca
# ╠═a836140b-378c-4a8a-8d4d-9cb0757ab3f3
# ╠═c3877f0d-5171-4ca4-befc-4f3263213e06
# ╟─28c9bd5d-2a46-4df4-9b80-3d6e4c6f2530
# ╠═ba851c42-8bb0-11f1-94f1-f50977498160
# ╠═9f51bb2a-574f-4a89-bc8c-426ac2961f7c
# ╠═db92cc68-04d0-4957-8f1e-bd3dc21d69fa
# ╠═33751e9d-9134-4ba4-a249-89dd63cf8078
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
