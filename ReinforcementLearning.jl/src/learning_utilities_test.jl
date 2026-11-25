### A Pluto.jl notebook ###
# v1.1.0

using Markdown
using InteractiveUtils

# ╔═╡ 5106ddc1-b3bc-4688-b6fb-75441f3c67d5
using PlutoDevMacros

# ╔═╡ 8845bf98-b287-48c2-9af5-67a992e513dd
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI, PlutoPlotly, ProfileCanvas, BenchmarkTools, LaTeXStrings, HypertextLiteral
	TableOfContents()
end
  ╠═╡ =#

# ╔═╡ efef7b11-9ea4-4632-b696-9b524e1e7c67
md"""
# Test Environments
"""

# ╔═╡ bf71e7e5-09e9-4e38-810f-6c9e09c5f8f3
const wind_values = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# ╔═╡ 51ee1eed-6f63-4d2d-9eb5-d53414dc2935
md"""
# Exact Solutions
"""

# ╔═╡ 85cb7aba-2783-45ab-b0f3-1a30217e5650
const gridworld_γ = 0.99f0

# ╔═╡ df2f0afc-8048-445c-bcdc-994bd9961306
md"""
# Feature Vector Setup
"""

# ╔═╡ a560cc4d-3dab-4e28-94c8-9f1a584ed993
md"""
# Episodic Training
"""

# ╔═╡ 95d771a3-f4e7-4193-ba24-c03b310ef975
md"""
## Value Training
"""

# ╔═╡ 41928240-0422-4a65-860b-7d5388da9b30
md"""
### Dense Linear Training
"""

# ╔═╡ baaa8768-e01d-4aec-9824-48098a2c5e96
md"""
### Sparse Linear Training
"""

# ╔═╡ f107a9f6-e871-4fa7-8245-2c478cb01210
md"""
### Dense Nonlinear Training
"""

# ╔═╡ fb9c24b7-a414-4919-a40e-f4336baea615
md"""
### Sparse Nonlinear Training
"""

# ╔═╡ 26b6a884-e26d-4963-a611-9de373c73294
md"""
## Policy Training
"""

# ╔═╡ f7997044-51c5-4cd3-b3c9-316365061633
md"""
### Dense Linear Training
"""

# ╔═╡ b94857be-c9ee-4bd2-ac6a-89b7360aad20
md"""
### Sparse Linear Training
"""

# ╔═╡ da50ae43-506b-4428-b0ee-aa56d11bb4f0
md"""
### Dense Nonlinear Training
"""

# ╔═╡ d2627e4f-f02d-4496-a07a-fd998fddacee
md"""
### Sparse Nonlinear Training
"""

# ╔═╡ bf5d0a2c-5f7a-4172-8ff4-a1b9c0f29a91
md"""
# Continuing Training
"""

# ╔═╡ 4ff2e16f-3cca-4749-8253-12b66e373b6d
md"""
## Value Training
"""

# ╔═╡ 582a58f5-2ca7-4eb1-904f-acb3c1f232f7
md"""
### Dense Linear Training
"""

# ╔═╡ 731b19ad-c56a-4be3-bc73-89d964e2c755
md"""
### Sparse Linear Training
"""

# ╔═╡ 537649b8-21f7-477f-876f-5bbeda17b241
md"""
### Dense Nonlinear Training
"""

# ╔═╡ 50386fb2-6d35-421c-b57c-47184da1e0e1
md"""
### Sparse Nonlinear Training
"""

# ╔═╡ d5a26b86-c534-4d3c-a2d7-8c45983b7538
md"""
## Policy Training
"""

# ╔═╡ 4fa2de8d-816c-408b-a0ff-84e17233dbe0
md"""
### Dense Linear Training
"""

# ╔═╡ f6806ed0-a4af-4e80-9ae2-cae025463a64
md"""
### Sparse Linear Training
"""

# ╔═╡ 3f7368f5-5be4-4bfc-bdd6-6d0faaacd25f
md"""
### Dense Nonlinear Training
"""

# ╔═╡ fb853499-f6c7-423f-a7ed-6949946dba22
md"""
### Sparse Nonlinear Training
"""

# ╔═╡ 674ed54f-f4d0-4d01-85e9-9f6eca2f7ecc
md"""
### Binary Features
"""

# ╔═╡ 89db1701-8975-48ac-a210-21c7e1faee55
# add testing criteria to progress with statistical significance and variance

# ╔═╡ d0eb2a8e-b000-11f1-b0d1-918e6553d269
md"""
# Dependencies
"""

# ╔═╡ 32d1e371-bf66-4b84-bc53-de595cc9080e
@fromparent import *

# ╔═╡ a920905a-9780-4fa9-9c79-bc9868681f6f
const mdp_stochastic = make_stochastic_gridworld(;wind = wind_values)

# ╔═╡ d5fa6e23-f041-4d6c-a633-b989b68ebea6
const dense_feature_setup = let
	feature_vector = zeros(Float32, length(mdp_stochastic.states))
	function update_feature_vector!(v::Vector{T}, s) where T<:Real
		v .= zero(T)
		idx = mdp_stochastic.state_index[s]
		v[idx] = one(T)
		return v
	end
	(;feature_vector, update_feature_vector!)
end

# ╔═╡ cf22debd-f4e0-4149-bc3c-40c36b962f62
const mdp_continuing = make_stochastic_gridworld(;wind = wind_values, continuing=true)

# ╔═╡ 016c1788-9d4d-4417-a5cd-dd8e6848903c
const state_mdp_stochastic = StateMDP(mdp_stochastic)

# ╔═╡ 87d1a03b-184d-4b4c-8a2b-5119d4e0d7c1
const state_mdp_continuing = StateMDP(mdp_continuing)

# ╔═╡ e042687c-e711-4b9f-b608-c6b3b8acd45d
const mountaincar_mdp = MountainCarTask.deterministic_mdp

# ╔═╡ 19065fe3-8b50-44d7-9ac2-92d65bf77e6e
const mountaincar_continuing = create_mountaincar_continuing_mdp()

# ╔═╡ fa354347-df5d-4229-a8be-d37620189e0f
# true value for average reward per step
value_iteration_v(mdp_stochastic, gridworld_γ).final_value[mdp_stochastic.initialize_state_index] |> x -> log(gridworld_γ) / log(x)

# ╔═╡ 49535902-0416-4da7-8ab2-9a104a1dfa57
value_iteration_v(mdp_continuing).reward_estimates[end]

# ╔═╡ 1182a8c9-940f-4da7-ae0c-e96d12be07d0
const sparse_feature_setup = state_aggregation_feature_setup(state_mdp_stochastic.initialize_state, length(mdp_stochastic.states), s -> mdp_stochastic.state_index[s])

# ╔═╡ 7e8b08c8-86d6-4d77-8db2-d22dbed52a58
const mountaincar_features = let
	setup = setup_mountaincar_tiles(10, 5)
	(;feature_vector = setup.feature_vector, update_feature_vector! = setup.update_feature_vector!)
end

# ╔═╡ c6601ce1-a5af-4498-94a8-c7b4673a3252
const dense_setup = setup_all_training(state_mdp_stochastic, dense_feature_setup...)

# ╔═╡ 72cf6e8e-a44d-42c9-a065-1df05f5e37d0
dense_setup.value_train_ϵ_decay(100_000, gridworld_γ, typemax(Int64); use_steps = true, show_messages = true, show_inner_messages = true)

# ╔═╡ 1b9809c5-fbe8-40f8-9fea-67a6619fedf4
#=╠═╡
@profview dense_setup.value_train_ϵ_decay(100_000, gridworld_γ, 0.5f0, typemax(Int64); use_steps = true, show_inner_messages = true)
  ╠═╡ =#

# ╔═╡ 159c371d-5005-4e01-b511-cb279defa821
#=╠═╡
@profview dense_setup.value_train_ϵ_decay(100_000, gridworld_γ, typemax(Int64); use_steps = true, show_messages = false, use_dp=true)
  ╠═╡ =#

# ╔═╡ db4e84e4-42c4-4012-8529-f2b0f4d469d1
#=╠═╡
@profview dense_setup.value_train_ϵ_decay(100_000, gridworld_γ, 0.5f0, typemax(Int64); use_steps = true, show_messages = false, use_dp=true)
  ╠═╡ =#

# ╔═╡ 51f7b0ef-692b-4e46-91ce-15eb14f148f1
#=╠═╡
@profview dense_setup.value_train_ϵ_decay(100_000, gridworld_γ, 10, typemax(Int64); use_steps = true, show_messages = false, batch_size=64)
  ╠═╡ =#

# ╔═╡ 54ac99fd-1bbb-43f6-8bce-5eaf1d89a585
#=╠═╡
@profview dense_setup.value_train_ϵ_decay(100_000, [4, 4], 1, gridworld_γ, typemax(Int64); use_steps = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ 7b354ebe-a3db-462a-a09f-5b346a019e50
#=╠═╡
@profview dense_setup.value_train_ϵ_decay(100_000, [4, 4], 1, gridworld_γ, 0.5f0, typemax(Int64); use_steps = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ 36d528c5-e6c5-45e0-9bb9-2163f71d7fc8
#=╠═╡
@profview dense_setup.value_train_ϵ_decay(100_000, [4, 4], 1, gridworld_γ, typemax(Int64); use_steps = true, show_messages = false, show_inner_messages = false, use_dp=true)
  ╠═╡ =#

# ╔═╡ bcbf7017-e196-4ea1-8a9b-a5aab279f96d
#=╠═╡
@profview dense_setup.value_train_ϵ_decay(100_000, [4, 4], 1, gridworld_γ, 0.5f0, typemax(Int64); use_steps = true, show_messages = false, use_dp=true)
  ╠═╡ =#

# ╔═╡ 1b572db3-f442-4df8-bfc7-45314ff851af
#=╠═╡
@profview dense_setup.value_train_ϵ_decay(100_000, [4, 4], 1, gridworld_γ, 10, typemax(Int64); use_steps = true, show_messages = false, batch_size=64)
  ╠═╡ =#

# ╔═╡ 0cb5288d-71ea-4e47-ab2a-d5a568c9c15c
#=╠═╡
@profview dense_setup.policy_train_β_decay(100_000, gridworld_γ, typemax(Int64); use_steps = true, show_messages = false, new_params = true)
  ╠═╡ =#

# ╔═╡ c2c18335-7939-44ca-bf89-c0915af902bd
#=╠═╡
@profview dense_setup.policy_train_β_decay(100_000, gridworld_γ, 0.9f0, 0.25f0, typemax(Int64); use_steps = true, show_messages = false, new_params = true)
  ╠═╡ =#

# ╔═╡ f661b99d-79b8-41b6-b05c-65fb72694f0b
#=╠═╡
@profview dense_setup.policy_train_β_decay(100_000, gridworld_γ; use_steps = true, show_messages = false, α_w_init = 0.01f0, α_θ_init = 0.1f0, N = 10, new_params = true)
  ╠═╡ =#

# ╔═╡ c2c58203-4dc9-4185-8427-e558ac7905c0
#=╠═╡
@profview dense_setup.policy_train_β_decay(100_000, [4, 4], 1, gridworld_γ, typemax(Int64); use_steps = true, show_messages = false, new_params = true)
  ╠═╡ =#

# ╔═╡ b4f19abe-d639-4796-85b9-aab48c4e4e65
#=╠═╡
@profview dense_setup.policy_train_β_decay(100_000, [4, 4], 1, gridworld_γ, 0.9f0, 0.5f0, typemax(Int64); use_steps = true, show_messages = false, α_w_init = 0.0001f0, new_params = true)
  ╠═╡ =#

# ╔═╡ 6510b77e-7860-4542-87f4-6e5a0293e6d1
#=╠═╡
@profview dense_setup.policy_train_β_decay(100_000, [4, 4], 1, gridworld_γ; use_steps = true, show_messages = false, α_w_init = 0.01f0, N = 100, new_params = true)
  ╠═╡ =#

# ╔═╡ 9e7af7ec-257a-483f-8955-81b663fbca58
const sparse_setup = setup_all_training(state_mdp_stochastic, sparse_feature_setup...)

# ╔═╡ eb11b82f-5d09-457a-aec9-b507bb2d9797
#=╠═╡
@profview sparse_setup.value_train_ϵ_decay(10_000, gridworld_γ, typemax(Int64); use_steps = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ 60f58042-62c0-4dd7-80c8-7368db62b23f
#=╠═╡
@profview sparse_setup.value_train_ϵ_decay(10_000, gridworld_γ, typemax(Int64); use_steps = true, show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ d6b79ea5-ecf2-48ca-8b9a-f50d9ee1632e
#=╠═╡
@profview sparse_setup.value_train_ϵ_decay(10_000, gridworld_γ, 0.5f0, typemax(Int64); use_steps = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ e7a86921-e040-4a29-bdf7-5f78e0ea82de
#=╠═╡
@profview sparse_setup.value_train_ϵ_decay(10_000, gridworld_γ, 0.5f0, typemax(Int64); use_steps = true, show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ b9318fff-906b-4fb3-92c6-6d93eb60c95d
#=╠═╡
@profview sparse_setup.value_train_ϵ_decay(10_000, gridworld_γ, 10, typemax(Int64); use_steps = true, show_messages = false, batch_size = 64)
  ╠═╡ =#

# ╔═╡ 803d4da9-2e3b-4041-93b1-8f5abde93097
#=╠═╡
@profview sparse_setup.value_train_ϵ_decay(10_000, [4, 4], 1, gridworld_γ, typemax(Int64); use_steps = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ 167530e4-26d8-40d1-9ccf-bbd72e994b68
#=╠═╡
@profview sparse_setup.value_train_ϵ_decay(10_000, [4, 4], 1, gridworld_γ, typemax(Int64); use_steps = true, show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ de05c778-93f3-4b1b-a8e1-2ddef0b2c468
#=╠═╡
@profview sparse_setup.value_train_ϵ_decay(10_000, [4, 4], 1, gridworld_γ, 0.5f0, typemax(Int64); use_steps = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ 0903ec42-6e80-49db-b5f1-fb32cb72799f
#=╠═╡
@profview sparse_setup.value_train_ϵ_decay(10_000, [4, 4], 1, gridworld_γ, 0.5f0, typemax(Int64); use_steps = true, show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ 81ad7c51-f3c8-4b63-a5a6-38980ccb5e69
#=╠═╡
@profview sparse_setup.value_train_ϵ_decay(10_000, [4, 4], 1, gridworld_γ, 10, typemax(Int64); use_steps = true, show_messages = false, batch_size = 64)
  ╠═╡ =#

# ╔═╡ 279270be-15d5-44fb-8bd4-da4e868ea9a7
#=╠═╡
@profview sparse_setup.policy_train_β_decay(10_000, gridworld_γ, typemax(Int64); use_steps = true, show_messages = false, new_params = true)
  ╠═╡ =#

# ╔═╡ 6b2b8a01-4b72-426f-944b-d2d909e4f2a5
#=╠═╡
@profview sparse_setup.policy_train_β_decay(10_000, gridworld_γ, 0.5f0, 0.5f0, typemax(Int64); use_steps = true, show_messages = false, new_params = true)
  ╠═╡ =#

# ╔═╡ 2bfc214d-ec16-4ec7-94a5-eb6d0b809b86
#=╠═╡
@profview sparse_setup.policy_train_β_decay(10_000, gridworld_γ; use_steps = true, show_messages = false, new_params = true, N = 10)
  ╠═╡ =#

# ╔═╡ 5954dbe4-a797-4bb9-8233-bff59cb83507
#=╠═╡
@profview sparse_setup.policy_train_β_decay(10_000, [4, 4], 1, gridworld_γ, typemax(Int64); use_steps = true, show_messages = false, new_params = true)
  ╠═╡ =#

# ╔═╡ 5c45eee2-cd4c-4d81-a92f-563642f0b934
#=╠═╡
@profview sparse_setup.policy_train_β_decay(10_000, [4, 4], 1, gridworld_γ, 0.5f0, 0.5f0, typemax(Int64); use_steps = true, show_messages = false, new_params = true)
  ╠═╡ =#

# ╔═╡ 8c5477f4-39ec-4caa-ad1a-a9fd3192a664
#=╠═╡
@profview sparse_setup.policy_train_β_decay(10_000, [4, 4], 1, gridworld_γ; use_steps = true, show_messages = false, new_params = true, N = 100, α_θ_init = 0.001f0, α_w_init = 0.0001f0)
  ╠═╡ =#

# ╔═╡ 11629c59-6e63-481f-80e1-d1553f495f4f
const dense_cont_setup = setup_all_training(state_mdp_continuing, dense_feature_setup...)

# ╔═╡ f5f380db-995c-4eac-a37d-7ad643cc94d3
#=╠═╡
@profview dense_cont_setup.value_train_ϵ_decay(10_000; use_steps = true, new_params = true)
  ╠═╡ =#

# ╔═╡ 1ab6e038-02d7-486d-aaec-175a53aa112b
#=╠═╡
@profview dense_cont_setup.value_train_ϵ_decay(10_000; use_steps = true, show_messages = false, new_params = true, use_dp = true)
  ╠═╡ =#

# ╔═╡ 2644fef7-f22e-4a3c-a8b8-e88b2e327532
#=╠═╡
@profview dense_cont_setup.value_train_ϵ_decay(10_000, 0.5f0; show_messages = false, use_steps = true, new_params = true)
  ╠═╡ =#

# ╔═╡ d87574c2-a4c6-4337-9943-7fc63c9f44a5
#=╠═╡
@profview dense_cont_setup.value_train_ϵ_decay(10_000, 0.5f0; show_messages = false, use_steps = true, new_params = true, use_dp = true)
  ╠═╡ =#

# ╔═╡ 2b3b6fd6-a093-422a-b943-8fa4a237153b
#=╠═╡
@profview dense_cont_setup.value_train_ϵ_decay(10_000, [4, 4], 1; use_steps = true, new_params = true)
  ╠═╡ =#

# ╔═╡ 1a83dcf2-3f9e-4b9b-9120-90a25bb82f00
#=╠═╡
@profview dense_cont_setup.value_train_ϵ_decay(10_000, [4, 4], 1; use_steps = true, new_params = true, use_dp = true)
  ╠═╡ =#

# ╔═╡ 2f8c4b5e-b500-454a-9d39-23d3bcb7ca54
#=╠═╡
@profview dense_cont_setup.value_train_ϵ_decay(10_000, [4, 4], 1, 0.9f0; use_steps = true, new_params = true)
  ╠═╡ =#

# ╔═╡ 6d3714ec-f5e5-42bf-a86d-45b82bb08892
#=╠═╡
@profview dense_cont_setup.value_train_ϵ_decay(10_000, [4, 4], 1, 0.9f0; use_steps = true, new_params = true, use_dp=true)
  ╠═╡ =#

# ╔═╡ 8f327a09-ebc9-430f-9e78-dddea105d06c
#=╠═╡
@profview dense_cont_setup.policy_train_β_decay(10_000; use_steps = true, new_params = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ 9217149c-ad62-4159-bea1-0b71dc560868
#=╠═╡
@profview dense_cont_setup.policy_train_β_decay(10_000, 0.5f0, 0.5f0; use_steps = true, new_params = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ 2f71426b-09fa-43d4-8434-23a010c21dbc
#=╠═╡
@profview dense_cont_setup.policy_train_β_decay(10_000, [4, 4], 1; use_steps = true, new_params = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ 07cc6013-56c3-4289-ab60-89303c2b3106
#=╠═╡
@profview dense_cont_setup.policy_train_β_decay(10_000, [4, 4], 1, 0.5f0, 0.5f0; use_steps = true, new_params = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ 36aa072c-b225-4b0c-9cac-4ff94ffab429
const sparse_cont_setup = setup_all_training(state_mdp_continuing, sparse_feature_setup...)

# ╔═╡ 00f129cb-8b25-4cda-b237-94d301eaf38d
#=╠═╡
@profview sparse_cont_setup.value_train_ϵ_decay(10_000; use_steps = true, new_params = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ 7d6b5cca-584e-4e2f-8542-42bd4967406f
#=╠═╡
@profview sparse_cont_setup.value_train_ϵ_decay(10_000; use_steps = true, new_params = true, show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ 5cbec763-2401-4457-986e-853feb697fa1
#=╠═╡
@profview sparse_cont_setup.value_train_ϵ_decay(10_000, 0.5f0; use_steps = true, new_params = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ 3b9f7420-8d82-4492-b6f7-d0b257ec2112
#=╠═╡
@profview sparse_cont_setup.value_train_ϵ_decay(10_000, 0.5f0; use_steps = true, new_params = true, show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ 6c212dd4-cdbb-4af6-aefa-6b67b0817039
#=╠═╡
@profview sparse_cont_setup.value_train_ϵ_decay(10_000, [4, 4], 1; use_steps = true, new_params = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ ee40ecc8-a363-4b2e-85ed-38921a6864f0
#=╠═╡
@profview sparse_cont_setup.value_train_ϵ_decay(10_000, [4, 4], 1; use_steps = true, new_params = true, show_messages = false, use_dp=true)
  ╠═╡ =#

# ╔═╡ 2ed11f49-438a-4130-a501-d3430a8ca410
#=╠═╡
@profview sparse_cont_setup.value_train_ϵ_decay(10_000, [4, 4], 1, 0.5f0; use_steps = true, new_params = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ 3550aac2-31ff-4f3c-98e3-b4d93e60a9e7
#=╠═╡
@profview sparse_cont_setup.value_train_ϵ_decay(10_000, [4, 4], 1, 0.5f0; use_steps = true, new_params = true, show_messages = false, use_dp=true)
  ╠═╡ =#

# ╔═╡ ea072419-cfb5-465e-b779-bfca36cd028b
#=╠═╡
@profview sparse_cont_setup.policy_train_β_decay(10_000; use_steps = true, new_params = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ 6b13d189-6056-4a0a-a9da-9885da799652
#=╠═╡
@profview sparse_cont_setup.policy_train_β_decay(10_000, 0.5f0, 0.5f0; use_steps = true, new_params = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ 4282fd95-f5db-4ada-8328-934162f88692
#=╠═╡
@profview sparse_cont_setup.policy_train_β_decay(10_000, [4, 4], 1; use_steps = true, new_params = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ 21862b14-50bd-4c99-91ba-8393872fb2a8
#=╠═╡
@profview sparse_cont_setup.policy_train_β_decay(10_000, [4, 4], 1, 0.9f0, 0.9f0; use_steps = true, new_params = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ c2a4aa42-0d6d-4b56-9fe9-11036ab6b844
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

julia_version = "1.13.4"
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
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
registries = "General"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

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
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "88352712893ec50bee3680605891eaf0e9ed6368"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.8.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

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
version = "8.22.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.7+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "OpenSSL_jll", "Zlib_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.104+0"

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
version = "2026.11.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.8+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.47.0+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
registries = "General"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "663e8b48b789916221e0765393b289ca6c88f24e"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.0"

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

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "2d0fc55c61321ba245c47be599570d11bac50303"
registries = "General"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.5"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

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
version = "1.3.2+0"

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
version = "1.70.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.8.5+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╟─efef7b11-9ea4-4632-b696-9b524e1e7c67
# ╠═bf71e7e5-09e9-4e38-810f-6c9e09c5f8f3
# ╠═a920905a-9780-4fa9-9c79-bc9868681f6f
# ╠═cf22debd-f4e0-4149-bc3c-40c36b962f62
# ╠═016c1788-9d4d-4417-a5cd-dd8e6848903c
# ╠═87d1a03b-184d-4b4c-8a2b-5119d4e0d7c1
# ╠═e042687c-e711-4b9f-b608-c6b3b8acd45d
# ╠═19065fe3-8b50-44d7-9ac2-92d65bf77e6e
# ╟─51ee1eed-6f63-4d2d-9eb5-d53414dc2935
# ╠═85cb7aba-2783-45ab-b0f3-1a30217e5650
# ╠═fa354347-df5d-4229-a8be-d37620189e0f
# ╠═49535902-0416-4da7-8ab2-9a104a1dfa57
# ╟─df2f0afc-8048-445c-bcdc-994bd9961306
# ╠═1182a8c9-940f-4da7-ae0c-e96d12be07d0
# ╠═d5fa6e23-f041-4d6c-a633-b989b68ebea6
# ╠═7e8b08c8-86d6-4d77-8db2-d22dbed52a58
# ╟─a560cc4d-3dab-4e28-94c8-9f1a584ed993
# ╠═c6601ce1-a5af-4498-94a8-c7b4673a3252
# ╠═9e7af7ec-257a-483f-8955-81b663fbca58
# ╟─95d771a3-f4e7-4193-ba24-c03b310ef975
# ╟─41928240-0422-4a65-860b-7d5388da9b30
# ╠═72cf6e8e-a44d-42c9-a065-1df05f5e37d0
# ╠═1b9809c5-fbe8-40f8-9fea-67a6619fedf4
# ╠═159c371d-5005-4e01-b511-cb279defa821
# ╠═db4e84e4-42c4-4012-8529-f2b0f4d469d1
# ╠═51f7b0ef-692b-4e46-91ce-15eb14f148f1
# ╟─baaa8768-e01d-4aec-9824-48098a2c5e96
# ╠═eb11b82f-5d09-457a-aec9-b507bb2d9797
# ╠═60f58042-62c0-4dd7-80c8-7368db62b23f
# ╠═d6b79ea5-ecf2-48ca-8b9a-f50d9ee1632e
# ╠═e7a86921-e040-4a29-bdf7-5f78e0ea82de
# ╠═b9318fff-906b-4fb3-92c6-6d93eb60c95d
# ╟─f107a9f6-e871-4fa7-8245-2c478cb01210
# ╠═54ac99fd-1bbb-43f6-8bce-5eaf1d89a585
# ╠═7b354ebe-a3db-462a-a09f-5b346a019e50
# ╠═36d528c5-e6c5-45e0-9bb9-2163f71d7fc8
# ╠═bcbf7017-e196-4ea1-8a9b-a5aab279f96d
# ╠═1b572db3-f442-4df8-bfc7-45314ff851af
# ╟─fb9c24b7-a414-4919-a40e-f4336baea615
# ╠═803d4da9-2e3b-4041-93b1-8f5abde93097
# ╠═167530e4-26d8-40d1-9ccf-bbd72e994b68
# ╠═de05c778-93f3-4b1b-a8e1-2ddef0b2c468
# ╠═0903ec42-6e80-49db-b5f1-fb32cb72799f
# ╠═81ad7c51-f3c8-4b63-a5a6-38980ccb5e69
# ╟─26b6a884-e26d-4963-a611-9de373c73294
# ╟─f7997044-51c5-4cd3-b3c9-316365061633
# ╠═0cb5288d-71ea-4e47-ab2a-d5a568c9c15c
# ╠═c2c18335-7939-44ca-bf89-c0915af902bd
# ╠═f661b99d-79b8-41b6-b05c-65fb72694f0b
# ╟─b94857be-c9ee-4bd2-ac6a-89b7360aad20
# ╠═279270be-15d5-44fb-8bd4-da4e868ea9a7
# ╠═6b2b8a01-4b72-426f-944b-d2d909e4f2a5
# ╠═2bfc214d-ec16-4ec7-94a5-eb6d0b809b86
# ╟─da50ae43-506b-4428-b0ee-aa56d11bb4f0
# ╠═c2c58203-4dc9-4185-8427-e558ac7905c0
# ╠═b4f19abe-d639-4796-85b9-aab48c4e4e65
# ╠═6510b77e-7860-4542-87f4-6e5a0293e6d1
# ╟─d2627e4f-f02d-4496-a07a-fd998fddacee
# ╠═5954dbe4-a797-4bb9-8233-bff59cb83507
# ╠═5c45eee2-cd4c-4d81-a92f-563642f0b934
# ╠═8c5477f4-39ec-4caa-ad1a-a9fd3192a664
# ╟─bf5d0a2c-5f7a-4172-8ff4-a1b9c0f29a91
# ╠═11629c59-6e63-481f-80e1-d1553f495f4f
# ╠═36aa072c-b225-4b0c-9cac-4ff94ffab429
# ╟─4ff2e16f-3cca-4749-8253-12b66e373b6d
# ╟─582a58f5-2ca7-4eb1-904f-acb3c1f232f7
# ╠═f5f380db-995c-4eac-a37d-7ad643cc94d3
# ╠═1ab6e038-02d7-486d-aaec-175a53aa112b
# ╠═2644fef7-f22e-4a3c-a8b8-e88b2e327532
# ╠═d87574c2-a4c6-4337-9943-7fc63c9f44a5
# ╟─731b19ad-c56a-4be3-bc73-89d964e2c755
# ╠═00f129cb-8b25-4cda-b237-94d301eaf38d
# ╠═7d6b5cca-584e-4e2f-8542-42bd4967406f
# ╠═5cbec763-2401-4457-986e-853feb697fa1
# ╠═3b9f7420-8d82-4492-b6f7-d0b257ec2112
# ╟─537649b8-21f7-477f-876f-5bbeda17b241
# ╠═2b3b6fd6-a093-422a-b943-8fa4a237153b
# ╠═1a83dcf2-3f9e-4b9b-9120-90a25bb82f00
# ╠═2f8c4b5e-b500-454a-9d39-23d3bcb7ca54
# ╠═6d3714ec-f5e5-42bf-a86d-45b82bb08892
# ╟─50386fb2-6d35-421c-b57c-47184da1e0e1
# ╠═6c212dd4-cdbb-4af6-aefa-6b67b0817039
# ╠═ee40ecc8-a363-4b2e-85ed-38921a6864f0
# ╠═2ed11f49-438a-4130-a501-d3430a8ca410
# ╠═3550aac2-31ff-4f3c-98e3-b4d93e60a9e7
# ╟─d5a26b86-c534-4d3c-a2d7-8c45983b7538
# ╟─4fa2de8d-816c-408b-a0ff-84e17233dbe0
# ╠═8f327a09-ebc9-430f-9e78-dddea105d06c
# ╠═9217149c-ad62-4159-bea1-0b71dc560868
# ╟─f6806ed0-a4af-4e80-9ae2-cae025463a64
# ╠═ea072419-cfb5-465e-b779-bfca36cd028b
# ╠═6b13d189-6056-4a0a-a9da-9885da799652
# ╟─3f7368f5-5be4-4bfc-bdd6-6d0faaacd25f
# ╠═2f71426b-09fa-43d4-8434-23a010c21dbc
# ╠═07cc6013-56c3-4289-ab60-89303c2b3106
# ╟─fb853499-f6c7-423f-a7ed-6949946dba22
# ╠═4282fd95-f5db-4ada-8328-934162f88692
# ╠═21862b14-50bd-4c99-91ba-8393872fb2a8
# ╟─674ed54f-f4d0-4d01-85e9-9f6eca2f7ecc
# ╠═89db1701-8975-48ac-a210-21c7e1faee55
# ╟─d0eb2a8e-b000-11f1-b0d1-918e6553d269
# ╠═5106ddc1-b3bc-4688-b6fb-75441f3c67d5
# ╠═32d1e371-bf66-4b84-bc53-de595cc9080e
# ╠═8845bf98-b287-48c2-9af5-67a992e513dd
# ╠═c2a4aa42-0d6d-4b56-9fe9-11036ab6b844
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
