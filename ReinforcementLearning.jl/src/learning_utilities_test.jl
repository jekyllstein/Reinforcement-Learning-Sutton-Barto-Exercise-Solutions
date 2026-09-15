### A Pluto.jl notebook ###
# v1.0.3

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

# ╔═╡ 43fa1c8f-e9db-461e-82cd-a1e844707fa9


# ╔═╡ df2f0afc-8048-445c-bcdc-994bd9961306
md"""
# Feature Vector Setup
"""

# ╔═╡ a560cc4d-3dab-4e28-94c8-9f1a584ed993
md"""
# Episodic Value Training
"""

# ╔═╡ 41928240-0422-4a65-860b-7d5388da9b30
md"""
## Linear Training
"""

# ╔═╡ f107a9f6-e871-4fa7-8245-2c478cb01210
md"""
## Nonlinear Training
"""

# ╔═╡ 26b6a884-e26d-4963-a611-9de373c73294
md"""
# Episodic Policy Training
"""

# ╔═╡ f7997044-51c5-4cd3-b3c9-316365061633
md"""
### Linear Training
"""

# ╔═╡ da50ae43-506b-4428-b0ee-aa56d11bb4f0
md"""
### Nonlinear Training
"""

# ╔═╡ bf5d0a2c-5f7a-4172-8ff4-a1b9c0f29a91
md"""
# Continuing Value Training
"""

# ╔═╡ 4ff2e16f-3cca-4749-8253-12b66e373b6d
md"""
## Linear Training
"""

# ╔═╡ d70867bf-2ccb-4877-96b9-8bccc1dbc7b3
md"""
### Setups
"""

# ╔═╡ db838dd8-7a59-434f-b473-86320277d08b
md"""
### Dense Features
"""

# ╔═╡ 11a977b5-b2c6-4999-a7f3-5b4cc98ba1e2
md"""
### Sparse Features
"""

# ╔═╡ 674ed54f-f4d0-4d01-85e9-9f6eca2f7ecc
md"""
### Binary Features
"""

# ╔═╡ cd20f90c-b98e-4427-bb99-5cb8721771db
md"""
## Nonlinear Training
"""

# ╔═╡ d83dddf4-abf5-4aa2-b979-9c04578638a5
md"""
### Dense Features
"""

# ╔═╡ 9c2fe4f9-3a76-4175-a1a7-dd24d2dc3477
md"""
### Sparse Features
"""

# ╔═╡ 310fb90a-52fe-45b8-bd10-564577300a6d
md"""
### Binary Features
"""

# ╔═╡ ac2e5ff3-cb46-472a-9115-e231846db5d2
md"""
# Continuing Policy Training
"""

# ╔═╡ 7597f4be-83c5-4b5f-a427-c38e3d09fbeb
md"""
## Linear Training
"""

# ╔═╡ f7d2463c-86d6-44ad-8366-c725161381e7
md"""
### Dense Features
"""

# ╔═╡ 7a6151d7-5055-4a4e-810a-1d169a6cc0c9
md"""
### Sparse Features
"""

# ╔═╡ bb01eff3-d860-4f93-be88-847233cf5081
md"""
### Binary Features
"""

# ╔═╡ e3a19a90-cf7b-43a2-8bc6-fa3bcb0ae6d1
md"""
## Nonlinear Training
"""

# ╔═╡ 69253b86-be71-4ab2-b83d-9dd4fa0e6f7e
md"""
### Dense Features
"""

# ╔═╡ 3d01ed1a-14b4-4a00-aae5-1f69d99fde36
md"""
### Sparse Features
"""

# ╔═╡ a9c15b2a-37b0-4aaf-8fac-3ccf8dd43f34
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

# ╔═╡ 1182a8c9-940f-4da7-ae0c-e96d12be07d0
const sparse_feature_setup = state_aggregation_feature_setup(state_mdp_stochastic.initialize_state(), length(mdp_stochastic.states), s -> mdp_stochastic.state_index[s])

# ╔═╡ 7e8b08c8-86d6-4d77-8db2-d22dbed52a58
const mountaincar_features = let
	setup = setup_mountaincar_tiles(10, 5)
	(;feature_vector = setup.feature_vector, update_feature_vector! = setup.update_feature_vector!)
end

# ╔═╡ ff72a08d-d980-4e2e-9b43-1ae45daee73b
const linear_value_dense_setup = setup_episodic_value_linear_training(state_mdp_stochastic, dense_feature_setup...)

# ╔═╡ 1b9809c5-fbe8-40f8-9fea-67a6619fedf4
#=╠═╡
@profview linear_value_dense_setup.train_ϵ_decay(0.9f0, 0.01f0, 0.5f0, 100_000; use_steps = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ 5af68de4-46b6-4375-808d-0875b286070c
#=╠═╡
@profview linear_value_dense_setup.train_ϵ_decay(0.9f0, 0.01f0, 0.5f0, 100_000; use_steps = true, show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ 51f7b0ef-692b-4e46-91ce-15eb14f148f1
#=╠═╡
@profview linear_value_dense_setup.train_dqn_ϵ_decay(0.9f0, 0.01f0, 10_000; use_steps = true, batch_size = 64, show_messages = false)
  ╠═╡ =#

# ╔═╡ 34f8c6c7-17fe-4bb3-9908-9536336f3a62
const nonlinear_value_dense_setup = setup_episodic_value_nonlinear_training(state_mdp_stochastic, dense_feature_setup...)

# ╔═╡ 54ac99fd-1bbb-43f6-8bce-5eaf1d89a585
#=╠═╡
@profview nonlinear_value_dense_setup.train_ϵ_decay([64, 64], 1, 0.9f0, 0.01f0, 0.5f0, 10_000; use_steps = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ 36d528c5-e6c5-45e0-9bb9-2163f71d7fc8
#=╠═╡
@profview nonlinear_value_dense_setup.train_ϵ_decay([64, 64], 1, 0.9f0, 0.01f0, 0.5f0, 10_000; use_steps = true, show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ bcbf7017-e196-4ea1-8a9b-a5aab279f96d
#=╠═╡
@profview nonlinear_value_dense_setup.train_dqn_ϵ_decay([64, 64], 1, 0.9f0, 0.01f0, 10_000; batch_size = 64, use_steps = true, show_messages = false, N = 10)
  ╠═╡ =#

# ╔═╡ 1b572db3-f442-4df8-bfc7-45314ff851af
#=╠═╡
@profview nonlinear_value_dense_setup.train_dqn_ϵ_decay([64, 64], 1, 0.9f0, 0.01f0, 100; α = 0.0001f0, batch_size = 64, use_steps = true, show_messages = false, N = 10, use_gpu = true)
  ╠═╡ =#

# ╔═╡ 0cb5288d-71ea-4e47-ab2a-d5a568c9c15c
const linear_policy_dense_setup = setup_episodic_policy_linear_training(state_mdp_stochastic, dense_feature_setup...)

# ╔═╡ 229cddb9-f38d-4e12-ac2b-3b593f46e283
#=╠═╡
@profview linear_policy_dense_setup.train_rate_decay(0.99f0, 0.1f0, 0.1f0, 0.5f0, 0.5f0, 100_000; use_steps = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ ff91023e-7f58-433c-ab6d-65f3ee49549f
#=╠═╡
@profview linear_policy_dense_setup.sync_train_rate_decay(0.99f0, 0.1f0, 0.1f0, 10_000; use_steps = true, show_messages = false, N = 10)
  ╠═╡ =#

# ╔═╡ 3b5782ee-b607-4959-bb9b-107a83f34639
const nonlinear_policy_dense_setup = setup_episodic_policy_nonlinear_training(state_mdp_stochastic, dense_feature_setup...)

# ╔═╡ 5ebac3c4-6be7-4a77-8e64-503629b02e04
nonlinear_policy_dense_setup.train([64, 64], 1, 0.99f0, 0.1f0, 0.1f0, 0.5f0, 0.5f0, 100)

# ╔═╡ 5728c42d-751c-4c41-b192-0f6323c6eb73
nonlinear_policy_dense_setup.train([64, 64], 1, 0.99f0, 0.1f0, 0.1f0, 0.5f0, 0.5f0, 100; use_gpu = true)

# ╔═╡ 1ee13ee1-9d9c-422c-9206-85dd7ae2a572
nonlinear_policy_dense_setup.train_exhaustive([64, 64], 1, 0.99f0, 0.1f0, 0.1f0, 0.5f0, 0.5f0, 10_000; use_steps = true, show_messages = false)

# ╔═╡ 744948ab-cebf-4091-bfb9-66ba74d563ce
nonlinear_policy_dense_setup.train_exhaustive([64, 64], 1, 0.99f0, 0.1f0, 0.1f0, 0.5f0, 0.5f0, 10_000; use_steps = true, show_messages = false, use_gpu = true)

# ╔═╡ aaf3c52e-356b-415d-a3ff-db43c235b322
#=╠═╡
@profview nonlinear_policy_dense_setup.train_rate_decay([64, 64], 1, 0.99f0, 0.1f0, 0.1f0, 0.0f0, 0.0f0, 10_000; use_steps = true, show_messages = false)
  ╠═╡ =#

# ╔═╡ f451a55d-e204-4f56-b356-2115ba83829b
#=╠═╡
@profview nonlinear_policy_dense_setup.train_rate_decay([64, 64], 1, 0.99f0, 0.1f0, 0.1f0, 0.5f0, 0.5f0, 1_000; use_steps = true, show_messages = false, use_gpu = true)
  ╠═╡ =#

# ╔═╡ 330c8c03-3371-4a49-8b88-bc924ca5f76f
#=╠═╡
@profview nonlinear_policy_dense_setup.sync_train_rate_decay([64, 64], 1, 0.99f0, 0.1f0, 0.1f0, 10_000; use_steps = true, N = 0, show_messages = false)
  ╠═╡ =#

# ╔═╡ 8893550d-31b2-4df6-af66-17f97609041a
#=╠═╡
@profview nonlinear_policy_dense_setup.sync_train_rate_decay([64, 64], 1, 0.99f0, 0.1f0, 0.1f0, 1_000; use_steps = true, N = 0, show_messages = false, use_gpu = true)
  ╠═╡ =#

# ╔═╡ 70f6f0cf-8ee0-424b-a14e-f746bcb4e810
const linear_value_cont_dense_setup = setup_continuing_value_linear_training(state_mdp_continuing, dense_feature_setup...)

# ╔═╡ 1f65e88b-2060-4cbb-9cae-a5af73094495
#=╠═╡
@profview linear_value_cont_dense_setup.train_ϵ_decay(0.01f0, 0.5f0, 100_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ 63c69867-1060-4c50-8dfa-a25278ddbcda
#=╠═╡
@profview linear_value_cont_dense_setup.train_ϵ_decay(0.01f0, 0.0f0, 100_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ f34e5443-7f7a-4461-a8e2-f61a30fcf3e3
#=╠═╡
@profview linear_value_cont_dense_setup.train_ϵ_decay(0.01f0, 0.5f0, 100_000; show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ 49037ef7-1273-44a7-881d-b1910015fd14
#=╠═╡
@profview linear_value_cont_dense_setup.train_ϵ_decay(0.01f0, 0.0f0, 100_000; show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ 8f2dd588-aac5-40d3-8e06-7040611fe867
const linear_value_cont_sparse_setup = setup_continuing_value_linear_training(state_mdp_continuing, sparse_feature_setup...)

# ╔═╡ 1d677cef-e362-4bd1-84ee-5aefa224feb7
#=╠═╡
@profview linear_value_cont_sparse_setup.train_ϵ_decay(0.01f0, 0.5f0, 100_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ 307302ec-4f13-4037-b26a-a34a11c1f8bb
#=╠═╡
@profview linear_value_cont_sparse_setup.train_ϵ_decay(0.01f0, 0.0f0, 100_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ 0ab5e09a-d6cc-46d5-ab9b-7c30f1ba102c
#=╠═╡
@profview linear_value_cont_sparse_setup.train_ϵ_decay(0.01f0, 0.5f0, 100_000; show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ 3ad842ea-fcb9-4f71-a8ea-78b03ad23a17
#=╠═╡
@profview linear_value_cont_sparse_setup.train_ϵ_decay(0.01f0, 0.0f0, 100_000; show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ c1ace692-752a-477e-aa7d-badce258d8f8
const linear_value_cont_mountaincar_setup = setup_continuing_value_linear_training(mountaincar_continuing, mountaincar_features...)

# ╔═╡ 248b5206-13d0-4d3c-a78b-fbaeb7cc9bb9
#=╠═╡
@profview linear_value_cont_mountaincar_setup.train_ϵ_decay(0.01f0, 0.5f0, 100_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ 13ef47af-ce86-441e-a35d-85406ee50c7c
#=╠═╡
@profview linear_value_cont_mountaincar_setup.train_ϵ_decay(0.01f0, 0.0f0, 100_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ 271b3bf8-78d1-4247-b8de-3240e5f56561
#=╠═╡
@profview linear_value_cont_mountaincar_setup.train_ϵ_decay(0.01f0, 0.5f0, 100_000; show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ f9dbdd26-db8a-4f6b-a99c-1ff74a82b8a4
#=╠═╡
@profview linear_value_cont_mountaincar_setup.train_ϵ_decay(0.01f0, 0.0f0, 100_000; show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ af20c893-2849-4fb2-a1a7-ce52d77f118e
const nonlinear_value_cont_dense_setup = setup_continuing_value_nonlinear_training(state_mdp_continuing, dense_feature_setup...)

# ╔═╡ 0bf447f3-29b2-430e-bdc4-d371e2f96dbf
#=╠═╡
@profview nonlinear_value_cont_dense_setup.train_ϵ_decay([64, 64], 1, 0.01f0, 0.5f0, 10_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ 4dd66844-ece8-42b4-9661-58cfeb4249e1
#=╠═╡
@profview nonlinear_value_cont_dense_setup.train_ϵ_decay([64, 64], 1, 0.01f0, 0.5f0, 1_000; show_messages = false, use_gpu = true)
  ╠═╡ =#

# ╔═╡ 8ce8b762-a0d8-43fc-8dcb-e2adf73e3954
#=╠═╡
@profview nonlinear_value_cont_dense_setup.train_ϵ_decay([64, 64], 1, 0.01f0, 0.0f0, 10_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ d9c95396-dc56-4edb-8f9b-dd98465708f4
#=╠═╡
@profview nonlinear_value_cont_dense_setup.train_ϵ_decay([64, 64], 1, 0.01f0, 0.0f0, 1_000; show_messages = false, use_gpu = true)
  ╠═╡ =#

# ╔═╡ c4b7255e-164e-4b5b-b96b-ac10cf8c92a3
#=╠═╡
@profview nonlinear_value_cont_dense_setup.train_ϵ_decay([64, 64], 1, 0.01f0, 0.5f0, 10_000; show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ 6d8378e5-8353-4ebb-9996-017e606ea904
#=╠═╡
@profview nonlinear_value_cont_dense_setup.train_ϵ_decay([64, 64], 1, 0.01f0, 0.5f0, 1_000; show_messages = false, use_dp = true, use_gpu = true)
  ╠═╡ =#

# ╔═╡ 0db68fb5-5d05-49b8-bdce-5a5c2d485cad
#=╠═╡
@profview nonlinear_value_cont_dense_setup.train_ϵ_decay([64, 64], 1, 0.01f0, 0.0f0, 10_000; show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ 100b2ed7-1959-4127-b98d-be0ef673080a
#=╠═╡
@profview nonlinear_value_cont_dense_setup.train_ϵ_decay([64, 64], 1, 0.01f0, 0.0f0, 1_000; show_messages = false, use_dp = true, use_gpu = true)
  ╠═╡ =#

# ╔═╡ 40570db1-b5a6-4755-b113-1f895a1aa1e9
const nonlinear_value_cont_sparse_setup = setup_continuing_value_nonlinear_training(state_mdp_continuing, sparse_feature_setup...)

# ╔═╡ 6ebb572b-779d-4ca7-b413-9ad277dc984f
#=╠═╡
@profview nonlinear_value_cont_sparse_setup.train_ϵ_decay([64, 64], 1, 0.01f0, 0.5f0, 10_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ d5227717-3ee5-45ce-8d69-7db16b1adebd
#=╠═╡
@profview nonlinear_value_cont_sparse_setup.train_ϵ_decay([64, 64], 1, 0.01f0, 0.0f0, 10_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ 52e51a05-7bc8-4913-a286-625da19e883b
#=╠═╡
@profview nonlinear_value_cont_sparse_setup.train_ϵ_decay([64, 64], 1, 0.01f0, 0.5f0, 10_000; show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ 41233c64-e5f6-4f78-9935-bbfa88fd5316
#=╠═╡
@profview nonlinear_value_cont_sparse_setup.train_ϵ_decay([64, 64], 1, 0.01f0, 0.0f0, 10_000; show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ 8c993685-49d9-4044-b856-98bc7bb8864d
const nonlinear_value_cont_mountaincar_setup = setup_continuing_value_nonlinear_training(mountaincar_continuing, mountaincar_features...)

# ╔═╡ 9f57da50-d75c-47b1-8924-7580cda1c683
#=╠═╡
@profview nonlinear_value_cont_mountaincar_setup.train_ϵ_decay([64, 64], 1, 0.1f0, 0.5f0, 10_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ e9eda3fa-6b0b-4944-9100-45b08b76d4c4
#=╠═╡
@profview nonlinear_value_cont_mountaincar_setup.train_ϵ_decay([64, 64], 1, 0.1f0, 0.0f0, 10_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ f6d9781b-0c27-4583-88db-d816a16ae947
#=╠═╡
@profview nonlinear_value_cont_mountaincar_setup.train_ϵ_decay([64, 64], 1, 0.1f0, 0.5f0, 10_000; show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ 58037d00-0135-40a1-89b0-68e2c037994c
#=╠═╡
@profview nonlinear_value_cont_mountaincar_setup.train_ϵ_decay([64, 64], 1, 0.1f0, 0.0f0, 10_000; show_messages = false, use_dp = true)
  ╠═╡ =#

# ╔═╡ bf35cb9a-1dc9-4ce3-8e8b-335fb4718fb5
const linear_policy_cont_dense_setup = setup_continuing_policy_linear_training(state_mdp_continuing, dense_feature_setup...)

# ╔═╡ f4343e65-1cfb-4480-80ec-4f9721460671
#=╠═╡
@profview linear_policy_cont_dense_setup.train_rate_decay(0.1f0, 0.1f0, 0.5f0, 0.5f0, 10_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ 334770f9-02fb-4598-8040-c268b055342b
#=╠═╡
@profview linear_policy_cont_dense_setup.train_rate_decay(0.1f0, 0.1f0, 0.0f0, 0.0f0, 10_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ c9bf84ab-cc7b-4271-a51f-e9e32481d4b4
const linear_policy_cont_sparse_setup = setup_continuing_policy_linear_training(state_mdp_continuing, sparse_feature_setup...)

# ╔═╡ dabae945-a332-49de-9740-5b815208f03d
#=╠═╡
@profview linear_policy_cont_sparse_setup.train_rate_decay(0.1f0, 0.1f0, 0.5f0, 0.5f0, 100_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ 6c7ef221-e69a-4b02-ad37-5e59712e28ca
#=╠═╡
@profview linear_policy_cont_sparse_setup.train_rate_decay(0.1f0, 0.1f0, 0.0f0, 0.0f0, 100_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ 20855619-ee21-42ad-9ddb-6f5e78c58733
const linear_policy_cont_mountaincar_setup = setup_continuing_policy_linear_training(mountaincar_continuing, mountaincar_features...)

# ╔═╡ 41933441-41af-4d5d-8ee8-6209ed7e32f1
#=╠═╡
@profview linear_policy_cont_mountaincar_setup.train_rate_decay(0.001f0, 0.001f0, 0.5f0, 0.5f0, 10_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ 633948df-412b-47eb-8ba3-09f630d3b809
#=╠═╡
@profview linear_policy_cont_mountaincar_setup.train_rate_decay(0.001f0, 0.001f0, 0.0f0, 0.0f0, 100_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ 4b97a139-8ccb-4e29-9413-6406df018cc5
const nonlinear_policy_cont_dense_setup = setup_continuing_policy_nonlinear_training(state_mdp_continuing, dense_feature_setup...)

# ╔═╡ 3d5cf378-935e-4c44-af73-20af5c904ae2
#=╠═╡
@profview nonlinear_policy_cont_dense_setup.train_rate_decay([64, 64], 1, 0.1f0, 0.1f0, 0.5f0, 0.5f0, 10_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ 7255f430-45e2-4a5d-acb7-20c6df10435a
#=╠═╡
@profview nonlinear_policy_cont_dense_setup.train_rate_decay([64, 64], 1, 0.1f0, 0.1f0, 0.5f0, 0.5f0, 10_000; show_messages = false, use_gpu = true)
  ╠═╡ =#

# ╔═╡ e08a2ffe-1770-4939-a53f-2ebc08d778fe
#=╠═╡
@profview nonlinear_policy_cont_dense_setup.train_rate_decay([64, 64], 1, 0.1f0, 0.1f0, 0.0f0, 0.0f0, 10_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ 9ded9f86-4de8-48d0-891f-0d08a11fcdd4
#=╠═╡
@profview nonlinear_policy_cont_dense_setup.train_rate_decay([64, 64], 1, 0.1f0, 0.1f0, 0.0f0, 0.0f0, 10_000; show_messages = false, use_gpu = true)
  ╠═╡ =#

# ╔═╡ a8713ec6-f6ca-4be8-ae83-c8b795a3248f
const nonlinear_policy_cont_sparse_setup = setup_continuing_policy_nonlinear_training(state_mdp_continuing, sparse_feature_setup...)

# ╔═╡ a9a960af-85c3-4c8b-a172-9d35ced9c68f
#=╠═╡
@profview nonlinear_policy_cont_sparse_setup.train_rate_decay([64, 64], 1, 0.1f0, 0.1f0, 0.5f0, 0.5f0, 10_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ 6369c8a3-f773-47f5-bcf9-6e541b725beb
#=╠═╡
@profview nonlinear_policy_cont_sparse_setup.train_rate_decay([64, 64], 1, 0.1f0, 0.1f0, 0.0f0, 0.0f0, 10_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ ad97387f-6579-46d8-9f3b-e4424bcd34db
const nonlinear_policy_cont_mountaincar_setup = setup_continuing_policy_nonlinear_training(mountaincar_continuing, mountaincar_features...)

# ╔═╡ e7bdf50e-19be-4cd2-95c5-2ad98a7f860d
#=╠═╡
@profview nonlinear_policy_cont_mountaincar_setup.train_rate_decay([64, 64], 1, 0.001f0, 0.001f0, 0.5f0, 0.5f0, 10_000; show_messages = false)
  ╠═╡ =#

# ╔═╡ 85fd7e7f-3f20-4a96-84fb-4f7f135571c2
#=╠═╡
@profview nonlinear_policy_cont_mountaincar_setup.train_rate_decay([64, 64], 1, 0.001f0, 0.001f0, 0.0f0, 0.0f0, 10_000; show_messages = false)
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
# ╟─efef7b11-9ea4-4632-b696-9b524e1e7c67
# ╠═bf71e7e5-09e9-4e38-810f-6c9e09c5f8f3
# ╠═a920905a-9780-4fa9-9c79-bc9868681f6f
# ╠═cf22debd-f4e0-4149-bc3c-40c36b962f62
# ╠═016c1788-9d4d-4417-a5cd-dd8e6848903c
# ╠═87d1a03b-184d-4b4c-8a2b-5119d4e0d7c1
# ╠═e042687c-e711-4b9f-b608-c6b3b8acd45d
# ╠═19065fe3-8b50-44d7-9ac2-92d65bf77e6e
# ╠═43fa1c8f-e9db-461e-82cd-a1e844707fa9
# ╠═df2f0afc-8048-445c-bcdc-994bd9961306
# ╠═1182a8c9-940f-4da7-ae0c-e96d12be07d0
# ╠═d5fa6e23-f041-4d6c-a633-b989b68ebea6
# ╠═7e8b08c8-86d6-4d77-8db2-d22dbed52a58
# ╟─a560cc4d-3dab-4e28-94c8-9f1a584ed993
# ╟─41928240-0422-4a65-860b-7d5388da9b30
# ╠═ff72a08d-d980-4e2e-9b43-1ae45daee73b
# ╠═1b9809c5-fbe8-40f8-9fea-67a6619fedf4
# ╠═5af68de4-46b6-4375-808d-0875b286070c
# ╠═51f7b0ef-692b-4e46-91ce-15eb14f148f1
# ╟─f107a9f6-e871-4fa7-8245-2c478cb01210
# ╠═34f8c6c7-17fe-4bb3-9908-9536336f3a62
# ╠═54ac99fd-1bbb-43f6-8bce-5eaf1d89a585
# ╠═36d528c5-e6c5-45e0-9bb9-2163f71d7fc8
# ╠═bcbf7017-e196-4ea1-8a9b-a5aab279f96d
# ╠═1b572db3-f442-4df8-bfc7-45314ff851af
# ╟─26b6a884-e26d-4963-a611-9de373c73294
# ╟─f7997044-51c5-4cd3-b3c9-316365061633
# ╠═0cb5288d-71ea-4e47-ab2a-d5a568c9c15c
# ╠═229cddb9-f38d-4e12-ac2b-3b593f46e283
# ╠═ff91023e-7f58-433c-ab6d-65f3ee49549f
# ╟─da50ae43-506b-4428-b0ee-aa56d11bb4f0
# ╠═3b5782ee-b607-4959-bb9b-107a83f34639
# ╠═5ebac3c4-6be7-4a77-8e64-503629b02e04
# ╠═5728c42d-751c-4c41-b192-0f6323c6eb73
# ╠═1ee13ee1-9d9c-422c-9206-85dd7ae2a572
# ╠═744948ab-cebf-4091-bfb9-66ba74d563ce
# ╠═aaf3c52e-356b-415d-a3ff-db43c235b322
# ╠═f451a55d-e204-4f56-b356-2115ba83829b
# ╠═330c8c03-3371-4a49-8b88-bc924ca5f76f
# ╠═8893550d-31b2-4df6-af66-17f97609041a
# ╟─bf5d0a2c-5f7a-4172-8ff4-a1b9c0f29a91
# ╟─4ff2e16f-3cca-4749-8253-12b66e373b6d
# ╟─d70867bf-2ccb-4877-96b9-8bccc1dbc7b3
# ╠═70f6f0cf-8ee0-424b-a14e-f746bcb4e810
# ╠═8f2dd588-aac5-40d3-8e06-7040611fe867
# ╠═c1ace692-752a-477e-aa7d-badce258d8f8
# ╟─db838dd8-7a59-434f-b473-86320277d08b
# ╠═1f65e88b-2060-4cbb-9cae-a5af73094495
# ╠═63c69867-1060-4c50-8dfa-a25278ddbcda
# ╠═f34e5443-7f7a-4461-a8e2-f61a30fcf3e3
# ╠═49037ef7-1273-44a7-881d-b1910015fd14
# ╠═11a977b5-b2c6-4999-a7f3-5b4cc98ba1e2
# ╠═1d677cef-e362-4bd1-84ee-5aefa224feb7
# ╠═307302ec-4f13-4037-b26a-a34a11c1f8bb
# ╠═0ab5e09a-d6cc-46d5-ab9b-7c30f1ba102c
# ╠═3ad842ea-fcb9-4f71-a8ea-78b03ad23a17
# ╟─674ed54f-f4d0-4d01-85e9-9f6eca2f7ecc
# ╠═248b5206-13d0-4d3c-a78b-fbaeb7cc9bb9
# ╠═13ef47af-ce86-441e-a35d-85406ee50c7c
# ╠═271b3bf8-78d1-4247-b8de-3240e5f56561
# ╠═f9dbdd26-db8a-4f6b-a99c-1ff74a82b8a4
# ╟─cd20f90c-b98e-4427-bb99-5cb8721771db
# ╟─d83dddf4-abf5-4aa2-b979-9c04578638a5
# ╠═af20c893-2849-4fb2-a1a7-ce52d77f118e
# ╠═0bf447f3-29b2-430e-bdc4-d371e2f96dbf
# ╠═4dd66844-ece8-42b4-9661-58cfeb4249e1
# ╠═8ce8b762-a0d8-43fc-8dcb-e2adf73e3954
# ╠═d9c95396-dc56-4edb-8f9b-dd98465708f4
# ╠═c4b7255e-164e-4b5b-b96b-ac10cf8c92a3
# ╠═6d8378e5-8353-4ebb-9996-017e606ea904
# ╠═0db68fb5-5d05-49b8-bdce-5a5c2d485cad
# ╠═100b2ed7-1959-4127-b98d-be0ef673080a
# ╟─9c2fe4f9-3a76-4175-a1a7-dd24d2dc3477
# ╠═40570db1-b5a6-4755-b113-1f895a1aa1e9
# ╠═6ebb572b-779d-4ca7-b413-9ad277dc984f
# ╠═d5227717-3ee5-45ce-8d69-7db16b1adebd
# ╠═52e51a05-7bc8-4913-a286-625da19e883b
# ╠═41233c64-e5f6-4f78-9935-bbfa88fd5316
# ╟─310fb90a-52fe-45b8-bd10-564577300a6d
# ╠═8c993685-49d9-4044-b856-98bc7bb8864d
# ╠═9f57da50-d75c-47b1-8924-7580cda1c683
# ╠═e9eda3fa-6b0b-4944-9100-45b08b76d4c4
# ╠═f6d9781b-0c27-4583-88db-d816a16ae947
# ╠═58037d00-0135-40a1-89b0-68e2c037994c
# ╟─ac2e5ff3-cb46-472a-9115-e231846db5d2
# ╟─7597f4be-83c5-4b5f-a427-c38e3d09fbeb
# ╟─f7d2463c-86d6-44ad-8366-c725161381e7
# ╠═bf35cb9a-1dc9-4ce3-8e8b-335fb4718fb5
# ╠═f4343e65-1cfb-4480-80ec-4f9721460671
# ╠═334770f9-02fb-4598-8040-c268b055342b
# ╟─7a6151d7-5055-4a4e-810a-1d169a6cc0c9
# ╠═c9bf84ab-cc7b-4271-a51f-e9e32481d4b4
# ╠═dabae945-a332-49de-9740-5b815208f03d
# ╠═6c7ef221-e69a-4b02-ad37-5e59712e28ca
# ╟─bb01eff3-d860-4f93-be88-847233cf5081
# ╠═20855619-ee21-42ad-9ddb-6f5e78c58733
# ╠═41933441-41af-4d5d-8ee8-6209ed7e32f1
# ╠═633948df-412b-47eb-8ba3-09f630d3b809
# ╟─e3a19a90-cf7b-43a2-8bc6-fa3bcb0ae6d1
# ╟─69253b86-be71-4ab2-b83d-9dd4fa0e6f7e
# ╠═4b97a139-8ccb-4e29-9413-6406df018cc5
# ╠═3d5cf378-935e-4c44-af73-20af5c904ae2
# ╠═7255f430-45e2-4a5d-acb7-20c6df10435a
# ╠═e08a2ffe-1770-4939-a53f-2ebc08d778fe
# ╠═9ded9f86-4de8-48d0-891f-0d08a11fcdd4
# ╟─3d01ed1a-14b4-4a00-aae5-1f69d99fde36
# ╠═a8713ec6-f6ca-4be8-ae83-c8b795a3248f
# ╠═a9a960af-85c3-4c8b-a172-9d35ced9c68f
# ╠═6369c8a3-f773-47f5-bcf9-6e541b725beb
# ╟─a9c15b2a-37b0-4aaf-8fac-3ccf8dd43f34
# ╠═ad97387f-6579-46d8-9f3b-e4424bcd34db
# ╠═e7bdf50e-19be-4cd2-95c5-2ad98a7f860d
# ╠═85fd7e7f-3f20-4a96-84fb-4f7f135571c2
# ╠═89db1701-8975-48ac-a210-21c7e1faee55
# ╟─d0eb2a8e-b000-11f1-b0d1-918e6553d269
# ╠═5106ddc1-b3bc-4688-b6fb-75441f3c67d5
# ╠═32d1e371-bf66-4b84-bc53-de595cc9080e
# ╠═8845bf98-b287-48c2-9af5-67a992e513dd
# ╠═c2a4aa42-0d6d-4b56-9fe9-11036ab6b844
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
