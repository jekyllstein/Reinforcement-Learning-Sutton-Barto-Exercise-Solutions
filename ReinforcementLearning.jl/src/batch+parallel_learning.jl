### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ a966b2b2-b3d9-4f28-9042-66167400f2cb
using PlutoDevMacros

# ╔═╡ 8b4b8bfa-9dfd-45c4-9ce0-8f4af97f9721
using DataStructures

# ╔═╡ e6aefa92-94d5-487d-91ba-b9a4d1b3277c
# ╠═╡ skip_as_script = true
#=╠═╡
using PlutoUI, BenchmarkTools, PlutoPlotly, PlutoProfile
  ╠═╡ =#

# ╔═╡ 78e8285e-5b98-420c-9fdc-cb943053a206
md"""
# Deep Q-Networks

To address some of the problems created by combining Q-learning with approximation techniques, DQN attempts to use a target network and a replay buffer to mitigate the moving target value problem and break correlations between consecutive samples.  
"""

# ╔═╡ 05535cef-05f4-42a1-925c-ceb85bb6dfba
md"""
## Linear Approximation
"""

# ╔═╡ 78a997cb-146c-4cb5-b2ec-4bd1188909e6
md"""
## Non-linear Approximation
"""

# ╔═╡ 0fd4da20-cf83-4071-b8a1-4259cbba7d8c
md"""
## Tabular Problem

For a tabular problem we can form feature vectors that are state aggregation features which have a unique feature for every state.
"""

# ╔═╡ c7f4374c-3151-443b-9d6a-85581c3f2438
md"""
## Gridworld Example
"""

# ╔═╡ 62bc710e-cd7e-43a5-bf9e-da024802358c
md"""
### Exact Solution
"""

# ╔═╡ 76bb3fb6-836b-42a7-8051-48e0856bedb3
md"""
### Approximation Methods

With a problem this small, we can simply use parameters that isolate every state action pair; however, to test the effectiveness of algorithms that deal with non-linear functions and overfitting, we can also initialize a neural network that uses the state input as an (X, Y) coordinate.  In this case, we would normalize the X, Y position to a range such as -1 to 1 and use the pair of values as a dense input rather than the sparse input of length 70 where we one-hot encode the states.
"""

# ╔═╡ e4fd6d59-be28-4852-a2d2-6ddc1a40116d
const gridworld_feature = zeros(Float32, 2)

# ╔═╡ e8c3f789-7ede-4f6d-83ee-3050c9ef5840
md"""
### Q-learning Example
"""

# ╔═╡ 3c11234c-5ea5-4709-b84b-ae477fb8dc55
md"""
### DQN
"""

# ╔═╡ 33b59f50-07e1-11f1-9748-31081ab2ceaf
md"""
# Dependencies
"""

# ╔═╡ ad872f3c-7be0-4427-bd1d-5afe25b6e9fa
@only_in_nb @fromparent import *

# ╔═╡ 0c0a7330-29bf-4326-8939-78b7e8b58d55
#fill in batch_inds with a uniform sample across the buffer_size or the current step, whichever is smaller
function update_batch_inds!(batch_inds::Vector{Int64}, step::Integer, buffer_size::Integer)
	l = min(buffer_size, step)
	sample!(1:l, batch_inds; replace = false)
end

# ╔═╡ cf40f4b3-4495-4f26-a007-18c6589ed4cf
begin
	form_batch_action_value_args(mdp::StateMDP, feature_vector, parameters, batch_size::Integer) = ()

	function form_batch_action_value_args(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, parameters::FCANNParams{T}, batch_size::Integer) where {T<:Real, S, A, P, F1, F2, F3}
		num_actions = length(mdp.actions)
		activations = FCANN.form_activations(parameters.weights[1], batch_size)
		(activations,)
	end

	function form_batch_action_value_args(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector::Vector{T}, parameters::FCANNParamsGPU, batch_size::Integer) where {T<:Real, S, A, P, F1, F2, F3}
		num_actions = length(mdp.actions)
		output_matrix = zeros(T, batch_size, num_actions)
		gpu_output = FCANN.cuda_allocate(output_matrix)
		gpu_input = FCANN.cuda_allocate(feature_matrix)
		activations = FCANN.form_activations(parameters.weights[1], batch_size)
		(gpu_input, gpu_output, activations)
	end
end

# ╔═╡ b3d7c539-d5a0-47fc-85bc-a62aafca8fa0
begin
	get_input_orientation(feature_matrix::Matrix{T}) where {T<:Real} = 'T'
	get_input_orientation(feature_matrix::Vector{V}) where V<:AbstractBinaryFeatures = 'N'
end

# ╔═╡ 576ff132-27d0-4a91-a955-e797fe6637c1
#update target values using parameters, action_value computation function and batch_args which will vary depending on the type of network
begin
	#linear function approximation with a dense feature vector
	function update_targets!(targets::Vector{T}, γ::T, replay_buffer::CircularBuffer, batch_inds::Vector{Int64}, target_params::Matrix{T}, feature_matrix::Matrix{T}, action_values::Vector{T}, output_matrix::Matrix{T}) where {T<:Real}
		#update feature matrix with replay buffer
		for i in eachindex(batch_inds)
			(x, i_a, r, x′, terminated) = replay_buffer[batch_inds[i]]
			update_feature_matrix!(feature_matrix, x′, i)
			#populate target values with the reward 
			targets[i] = r
		end

		#perform forward pass to fill in target values with function output times the discount rate plus the reward
		LinearAlgebra.BLAS.gemm!('T', 'N', γ, feature_matrix, target_params, zero(T), output_matrix)

		#for non terminal states add to target discounted future function value
		for i in eachindex(batch_inds)
			(s, i_a, r, s′, terminated) = replay_buffer[batch_inds[i]]
			if !terminated
				targets[i] += γ * maximum(view(output_matrix, i, :))
			end
		end
	end

	#linear function approximation with a binary feature vector
	function update_targets!(targets::Vector{T}, γ::T, replay_buffer::CircularBuffer, batch_inds::Vector{Int64}, target_params::Matrix{T}, feature_matrix::Vector{V}, action_values::Vector{T}, output_matrix::Matrix{T}) where {T<:Real, V<:AbstractBinaryFeatures}
		#update feature matrix with replay buffer
		for i in eachindex(batch_inds)
			(x, i_a, r, x′, terminated) = replay_buffer[batch_inds[i]]
			targets[i] = r
			if !terminated
				update_linear_action_values!(action_values, x′, target_params)
				targets[i] += γ * maximum(action_values)
			end
		end
	end

	#linear function approximation with a dense feature vector
	function update_targets!(targets::Vector{T}, γ::T, replay_buffer::CircularBuffer, batch_inds::Vector{Int64}, target_params::FCANNParams{T}, feature_matrix, action_values::Vector{T}, output_matrix::Matrix{T}, activations::FCANNActivations{T}) where {T<:Real}
		#update feature matrix with replay buffer
		for i in eachindex(batch_inds)
			(x, i_a, r, x′, terminated) = replay_buffer[batch_inds[i]]
			update_feature_matrix!(feature_matrix, x′, i)
			#populate target values with the reward 
			targets[i] = r
		end

		input_orientation = get_input_orientation(feature_matrix)

		#perform forward pass to fill in target values with function output
		FCANN.forwardNOGRAD_base!(activations, target_params.weights..., feature_matrix, target_params.reslayers; input_orientation = input_orientation)
		output_matrix .= activations[end]

		#for non terminal states add to target discounted future function value
		for i in eachindex(batch_inds)
			(s, i_a, r, s′, terminated) = replay_buffer[batch_inds[i]]
			if !terminated
				targets[i] += γ * maximum(view(output_matrix, i, :))
			end
		end
	end
end

# ╔═╡ a743f767-1ff6-4e1c-8c6f-88622d07c175
begin
	function accumulate_linear_gradient!(∇q̂::Matrix{T}, c::T, i::Integer, i_a::Integer, feature_matrix::Matrix{T}) where {T<:Real}
		#the feature matrix contains the feature vector of each example in a column, the i argument is the index of the example being used and i_a is the action index for that example
		@inbounds @simd for j in 1:size(feature_matrix, 1)
			∇q̂[j, i_a] += c * feature_matrix[j, i]
		end
	end

	function accumulate_linear_gradient!(∇q̂::Matrix{T}, c::T, i::Integer, i_a::Integer, feature_vectors::Vector{V}) where {T<:Real, V<:StateAggregationFeatureVector}
		#the i argument is the index of the example being used and i_a is the action index for that example, only need to update term for the active feature in that action
		i_s = feature_vectors[i].group_index
		∇q̂[i_s, i_a] += c
	end

	function accumulate_linear_gradient!(∇q̂::Matrix{T}, c::T, i::Integer, i_a::Integer, feature_vectors::Vector{V}) where {T<:Real, V<:BinaryFeatureVector}
		#the i argument is the index of the example being used and i_a is the action index for that example, only need to update terms for active features in that action
		x = feature_vectors[i]
		@inbounds @simd for ind in 1:x.num_features
			j = x.active_features[ind]
			∇q̂[j, i_a] += c
		end
	end
end

# ╔═╡ f76f33f3-7cb3-4715-800c-9a0b2561f05b
function ReinforcementLearning.update_linear_value_gradient!(∇q̂::Matrix{T}, value_params::Matrix{T}, targets::Vector{T}, output_indices::Vector{I}, feature_matrix, output_matrix::Matrix{T}) where {T<:Real, I<:Integer}
	#reset gradient to 0
	∇q̂ .= zero(T)

	#initialize batch size in order to calculate constant for average
	batch_size = length(targets)
	c = T(2 / batch_size)

	#get reference value using value params
	LinearAlgebra.BLAS.gemm!('T', 'N', one(T), feature_matrix, value_params, zero(T), output_matrix)

	#accumulate gradient of loss function per example
	for i in eachindex(targets)
		q̂ = output_matrix[i, output_indices[i]]
		δ = targets[i] - q̂
		accumulate_linear_gradient!(∇q̂, c*δ, i, output_indices[i], feature_matrix)
	end
end

# ╔═╡ f7a94436-b905-48b5-a0f3-ae26d0ecab5e
#=╠═╡
begin
	function ReinforcementLearning.update_fcann_value_gradient!(∇q̂::FCANNParams{T}, value_params::FCANNParams{T}, targets::Vector{T}, output_indices::Vector{I}, feature_matrix, output_matrix::Matrix{T}, hidden_layers::Vector{Int64}, l2::T, tanh_grad_z::FCANNActivations{T}, activations::FCANNActivations{T}, deltas::FCANNActivations{T}, dropout::T, activation_list::AbstractVector{B}) where {T<:Float32, B<:Bool, I<:Integer}
		FCANN.nnCostFunction(params.weights..., hidden_layers, x, targets, output_indices, l2, ∇v̂.weights..., tanh_grad_z, activations, deltas, dropout; resLayers = params.reslayers, loss_type = "sqErr", activation_list = activation_list, input_orientation = get_input_orientation(feature_matrix))
	end
end
  ╠═╡ =#

# ╔═╡ 2dd9b971-fa6e-4a55-8a82-b16739199fab
begin
	form_feature_matrix(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector::Vector{T}, batch_size::Integer) where {T<:Real, S, A, P, F1, F2, F3} = zeros(T, length(feature_vector), batch_size)

	function form_feature_matrix(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector::V, batch_size::Integer)  where {T<:Real, S, A, P, F1, F2, F3, V<:AbstractBinaryFeatures}
		output = Vector{V}(undef, batch_size)
		for i in 1:batch_size
			output[i] = deepcopy(feature_vector)
		end
		return output
	end
end

# ╔═╡ 3a4510e6-054b-40fe-989d-7ac8c86db757
function dqn!(value_params::Q, target_params::Q, mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector::V, update_feature_vector!::Function, update_action_values!::Function, ∇q̂, update_value_gradient!::Function; α = one(T)/10, ϵ = one(T) / 10, buffer_size::Integer = 10_000, batch_size::Integer = 512, target_update_interval::Integer = 100, α_decay = one(T), decay_step = typemax(Int64), save_step_rewards::Bool = false, kwargs...) where {Q, T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1<:Function, F2<:Function, F3<:Function, V}

	#initialize memory
	action_values = zeros(T, length(mdp.actions))
	policy = copy(action_values)
	replay_buffer = CircularBuffer{Tuple{V, Int64, T, V, Bool}}(buffer_size)
	targets = Vector{T}(undef, batch_size)
	batch_inds = Vector{Int64}(undef, batch_size)
	feature_matrix = form_feature_matrix(mdp, feature_vector, batch_size)
	output_matrix = zeros(T, batch_size, length(mdp.actions))
	batch_args = form_batch_action_value_args(mdp, feature_vector, value_params, batch_size)
	output_inds = Vector{Int64}(undef, batch_size)
	feature_vector2 = deepcopy(feature_vector)
	
	s = mdp.initialize_state()
	update_feature_vector!(feature_vector, s)
	update_action_values!(action_values, feature_vector, value_params)
	policy .= action_values
	make_ϵ_greedy_policy!(policy; ϵ = ϵ)
	i_a = sample_action(policy)
	
	ep = 1
	step = 1
	epreward = zero(T)
	episode_rewards = Vector{T}()
	episode_steps = Vector{Int64}()
	step_rewards = Vector{T}()
	decay = one(T)
	
	while (ep <= max_episodes) && (step <= max_steps)
		# update_value_gradient!(∇q̂, action_values, feature_vector, i_a, parameters)
		# q̂ = action_values[i_a]

		#get next reward and state from transition and add it to the replay buffer, note that the buffer also stores whether the transition state s′ is terminal
		(r, s′) = mdp.ptf(s, i_a)
		update_feature_vector!(feature_vector2, s′)
		terminated = mdp.isterm(s′)
		
		push!(replay_buffer, (deepcopy(feature_vector), i_a, r, deepcopy(feature_vector2), terminated))

		save_step_rewards && push!(step_rewards, r)

		epreward += r

		#if an episode terminates, initialize a new starting state and add information about the episode to the history
		if terminated
			# @info "episode terminated on step $step with reward $r"
			s′ = mdp.initialize_state()
			update_feature_vector!(feature_vector2, s′)
			push!(episode_rewards, epreward)
			push!(episode_steps, step)
			epreward = zero(T)
			ep += 1
		end

		#prepare next action selection from s′
		# update_feature_vector!(feature_vector, s′)
		update_action_values!(action_values, feature_vector2, value_params)
		policy .= action_values
		make_ϵ_greedy_policy!(policy; ϵ = ϵ)
		i_a′ = sample_action(policy)
		#@info "action choice is $i_a′"

		decay *= (step > decay_step)*α_decay + (step <= decay_step)

		#only perform gradient parameter update once the replay buffer is large enough to fill up an entire batch
		if step ≥ batch_size
			update_batch_inds!(batch_inds, step, buffer_size)
			# @info "batch inds are $batch_inds"
			
			update_targets!(targets, γ, replay_buffer, batch_inds, target_params, feature_matrix, action_values, output_matrix, batch_args...)
			# @info "target values are $targets"

			#update feature matrix
			for i in eachindex(batch_inds)
				(x_k, i_a_k, _, _, _) = replay_buffer[batch_inds[i]]
				update_feature_matrix!(feature_matrix, x_k, i)
				output_inds[i] = i_a_k
			end
	
			update_value_gradient!(∇q̂, value_params, targets, output_inds, feature_matrix, output_matrix, batch_args...)
	
			update_params_with_gradient!(value_params, α*decay, ∇q̂)

			#on set interval update target params with current value params
			if iszero(step % target_update_interval)
				# @info "batch stuff: target = $targets, inds = $output_inds"
				copy!(target_params, value_params)
			end
		end
		
		s = s′
		feature_vector = deepcopy(feature_vector2)
		i_a = i_a′
		step += 1
	end

	q̂, form_kwargs = form_value_function(mdp, update_feature_vector!, update_action_values!, feature_vector, value_params)
	return (value_function = q̂, episode_rewards = episode_rewards, episode_steps = episode_steps, final_parameters = deepcopy(value_params), form_kwargs = form_kwargs)
end

# ╔═╡ 830ba410-377a-423b-9e75-6884c8cbbbea
dqn_linear(mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), value_params::Matrix{T} = initialize_linear_parameters(feature_vector,mdp, init_value), target_params::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, init_value), kwargs...) where T<:Real = dqn!(value_params, target_params, mdp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, update_linear_action_values!, copy(value_params), update_linear_value_gradient!; kwargs...) 

# ╔═╡ 05fa2e5c-8633-4c43-89ba-09dd0c83cdfa
function dqn(mdp::TabularMDP, γ::T, max_episodes::Integer, max_steps::Integer; kwargs...) where T<:Real 
	state_mdp = StateMDP(mdp)
	setup = state_aggregation_feature_setup(first(mdp.states), length(mdp.states), s -> mdp.state_index[s])
	dqn_linear(state_mdp, γ, max_episodes, max_steps, setup...; kwargs...)
end

# ╔═╡ a3d54085-dbdc-4889-9313-efd1ece2150a
const gridworld_mdp = make_stochastic_gridworld(;wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0])

# ╔═╡ 8b141bcf-db3e-44d2-96de-c916f4018740
#=╠═╡
function plot_gridworld_value_function(v::Vector{T}) where T<:Real
	zs = zeros(T, 7, 10)
	xs = 1:10
	ys = 1:7
	for (i, s) in enumerate(gridworld_mdp.states)
		if !iszero(v[i])
			zs[s.y, s.x] = v[i]
		else
			zs[s.y, s.x] = NaN32
		end
	end
	tr = heatmap(x = xs, y = ys, z = zs)
	plot(tr, Layout(yaxis_scaleanchor = "x", xaxis_ticknames = 1:10, xaxis_tickvals = 1:10, width = 560))
end
  ╠═╡ =#

# ╔═╡ 37a4303b-a4b9-423a-a5fa-3282c323f04b
#=╠═╡
function plot_gridworld_value_function(q̂::Function)
	zs = zeros(Float32, 7, 10)
	xs = 1:10
	ys = 1:7
	for (i, s) in enumerate(gridworld_mdp.states)
		out = q̂(s)
		zs[s.y, s.x] = out.maximizing_value
	end
	tr = heatmap(x = xs, y = ys, z = zs)
	plot(tr, Layout(yaxis_scaleanchor = "x", xaxis_ticknames = 1:10, xaxis_tickvals = 1:10, width = 560))
end
  ╠═╡ =#

# ╔═╡ de6aca47-7d2c-4c8c-8df9-c3e16e6dc2bd
const gridworld_dqn = dqn(gridworld_mdp, 0.99f0, typemax(Int64), 40_000; α = 3f-4, ϵ = 0.01f0, buffer_size = 10_000, batch_size = 512, target_update_interval = 100)

# ╔═╡ 696650a0-9e0f-45d0-a768-679b02688f06
const gridworld_state_mdp = StateMDP(gridworld_mdp)

# ╔═╡ 01697aad-517e-4e48-b543-92d2974990ef
# ╠═╡ show_logs = false
const gridworld_exact = value_iteration_v(gridworld_mdp, 0.99f0; save_history = false)

# ╔═╡ 293ee629-3fb9-442d-be27-11e9c0545fae
#=╠═╡
plot_gridworld_value_function(gridworld_exact.final_value)
  ╠═╡ =#

# ╔═╡ 8705bc78-b419-4b30-8a1e-02398fd456b5
begin
	function eval_gridworld_final_policy(π::Matrix{T}; samples = 10_000) where T<:Real
		out = [runepisode(gridworld_mdp; π = π)[5] for _ in 1:samples]
		summarystats(out)
	end

	function eval_gridworld_final_policy(π::Function; samples = 10_000)
		state_mdp = StateMDP(gridworld_mdp)
		out = [runepisode(state_mdp; π = π)[5] for _ in 1:samples]
		summarystats(out)
	end
end

# ╔═╡ 2753435d-8ea2-4bab-9b06-cb0e36bcab99
eval_gridworld_final_policy(gridworld_exact.optimal_policy)

# ╔═╡ d1058fb3-cb03-4031-b089-ce5f077036a0
eval_gridworld_final_policy(s -> gridworld_dqn.value_function(s).maximizing_action)

# ╔═╡ 9a70941b-bf9e-4b3c-aef6-915f3e2019fe
function update_gridworld_feature!(v::Vector{T}, s::GridworldState) where T<:Real
	xmin = 1
	xmax = 10
	ymin = 1
	ymax = 7
	v[1] = 2*(((s.x - xmin) / (xmax - xmin)) - one(T)/2)
	v[2] = 2*(((s.y - ymin) / (ymax - ymin)) - one(T)/2)
	return v
end

# ╔═╡ 81058b8e-c80c-4a39-b33e-15b42d1225b8
const gridworld_q = sarsa_λ(gridworld_mdp, 0.99f0, 0f0, typemax(Int64), 100_000; α = 3f-4, ϵ = 0.1f0)

# ╔═╡ 4f222b09-00e0-48b9-bd3e-6b6ebfba5727
eval_gridworld_final_policy(s -> gridworld_q.value_function(s).maximizing_action)

# ╔═╡ ab6d36c1-5674-49f4-b291-f367840c9335
const gridworld_q2 = sarsa_λ_linear(gridworld_state_mdp, 0.99f0, 0f0, typemax(Int64), 100_000, gridworld_feature, update_gridworld_feature!; α = 0.04f0, ϵ = 0.01f0, compute_value = compute_q_learning_value)

# ╔═╡ 809132ad-fad8-4bc7-b8f4-1f98dbc8503c
eval_gridworld_final_policy(s -> gridworld_q2.value_function(s).maximizing_action)

# ╔═╡ b2096a04-74f8-4287-aad5-dc27752a21f7
#=╠═╡
plot_gridworld_value_function(gridworld_q2.value_function)
  ╠═╡ =#

# ╔═╡ b067424c-2b40-4d99-9dba-419af6fb2209
#=╠═╡
function eval_gridworld_returns(output::NamedTuple; total_steps = 100_000, interval = 100)
	step_rewards = zeros(Float32, total_steps)
	step_rewards[output.episode_steps] .= 1f0
	[mean(view(step_rewards, i-interval+1:i)) for i in interval:interval:total_steps]
end
  ╠═╡ =#

# ╔═╡ 281718b9-3123-479f-8847-37e48b6298f7
#=╠═╡
eval_gridworld_returns(gridworld_q2; interval = 1000) |> plot
  ╠═╡ =#

# ╔═╡ b1880149-4d45-4cfb-91cd-4c094ac5a1eb
#=╠═╡
function evaluate_gridworld_q_learning(γ, steps, α, ϵ; nruns = 100, kwargs...)
	f(x) = sarsa_λ(gridworld_mdp, γ, 0f0, typemax(Int64), steps; α = α, ϵ = ϵ) |> v -> eval_gridworld_returns(v; total_steps = steps, kwargs...)
	1:nruns |> Map(f) |> foldxt((a, b) -> a .+ b) |> v -> v ./ nruns
end
  ╠═╡ =#

# ╔═╡ daa89863-cb36-4a12-a699-646d8ba55904
#=╠═╡
plot([scatter(y = evaluate_gridworld_q_learning(0.99f0, 20_000, α, 0.01f0; nruns = 100, interval = 100), name = "α = $α") for α in [1f-2, 1f-3, 1f-4]], Layout(title = "Q-learning Gridworld Rewards"))
  ╠═╡ =#

# ╔═╡ c5e0bbaf-a04d-4940-b7ab-3317f9d513a9
#=╠═╡
function evaluate_gridworld_q_learning2(γ, steps, α, ϵ; nruns = 100, kwargs...)
	f(x) = sarsa_λ_linear(gridworld_state_mdp, γ, 0f0, typemax(Int64), steps, gridworld_feature, update_gridworld_feature!; α = α, ϵ = ϵ) |> v -> eval_gridworld_returns(v; total_steps = steps, kwargs...)
	1:nruns |> Map(f) |> foldxt((a, b) -> a .+ b) |> v -> v ./ nruns
end
  ╠═╡ =#

# ╔═╡ 7edf78dd-9dad-4726-8173-c95f3e4ed6ab
#=╠═╡
plot([scatter(y = evaluate_gridworld_q_learning2(0.99f0, 100_000, α, 0.1f0; nruns = 100, interval = 100), name = "α = $α") for α in [1f-2, 1f-3, 1f-4]], Layout(title = "Q-learning Gridworld Rewards"))
  ╠═╡ =#

# ╔═╡ d4a59bc4-2c2a-4966-8e1a-e336e26d4d40
const gridworld_q3 = sarsa_λ_fcann(gridworld_state_mdp, 0.99f0, 0.5f0, typemax(Int64), 100_000, gridworld_feature, update_gridworld_feature!, [32, 32]; α = .15f0, ϵ = 0.05f0, reslayers = 1)

# ╔═╡ 16bdaa3d-77b2-433b-8343-54af3a121e85
eval_gridworld_final_policy(s -> gridworld_q3.value_function(s).maximizing_action)

# ╔═╡ d437cfc2-6dfe-40a9-bb22-c90d25bebd25
#=╠═╡
plot_gridworld_value_function(gridworld_q3.value_function)
  ╠═╡ =#

# ╔═╡ 22744a27-614a-4317-a0ed-833bb0ef659c
const gridworld_value_studies = setup_episodic_value_parameter_studies(gridworld_state_mdp, gridworld_feature, update_gridworld_feature!; use_steps = true, min_reward = 0f0)

# ╔═╡ 95098580-9d74-41af-af28-c08c7dede5f4
function display_study(study::NamedTuple)
	results = study.results
	DataFrame(begin
		(;k..., value = results[k])
	end
	for k in keys(results))
end

# ╔═╡ 1a3493ac-966b-4260-8c06-0e60033ba41f
begin
	for α in 2f0 .^ (-15:-1)
		gridworld_value_studies.sarsa_linear_study.update_results!(0.99f0, α, 0.0f0, 100_000; ϵ = 0.05f0, compute_value = compute_q_learning_value)
	end
	display_study(gridworld_value_studies.sarsa_linear_study) |> df -> sort(df, :value; rev=true)
end

# ╔═╡ f1130bab-babd-41e1-891c-00a2d846b39f
begin
	for α in 2f0 .^ (-9:-2)
		gridworld_value_studies.sarsa_nonlinear_study.update_results!(0.99f0, α, 0.0f0, 100_000, 16, 4, 1; ϵ = 0.01f0, compute_value = compute_sarsa_value)
	end
	display_study(gridworld_value_studies.sarsa_nonlinear_study) |> df -> sort(df, :value; rev=true)
end

# ╔═╡ cd104c87-dd11-45a7-9ef5-ccecfdea3abd
#=╠═╡
function evaluate_gridworld_q_learning3(γ, steps, α, ϵ; nruns = 100, hidden_layers = [8, 8], λ = 0f0, kwargs...)
	f(x) = sarsa_λ_fcann(gridworld_state_mdp, γ, λ, typemax(Int64), steps, gridworld_feature, update_gridworld_feature!, hidden_layers; α = α, ϵ = ϵ, compute_value = compute_q_learning_value) |> v -> eval_gridworld_returns(v; total_steps = steps, kwargs...)
	1:nruns |> Map(f) |> foldxt((a, b) -> a .+ b) |> v -> v ./ nruns
end
  ╠═╡ =#

# ╔═╡ 55f2b6a8-52e7-47bd-82a1-40fa0ff11d98
#=╠═╡
plot([scatter(y = evaluate_gridworld_q_learning3(0.99f0, 100_000, α, 0.05f0; nruns = 40, interval = 100, λ = 0.0f0, hidden_layers = [32, 32]), name = "α = $α") for α in [.125f0]], Layout(title = "Q-learning Gridworld Rewards"))
  ╠═╡ =#

# ╔═╡ 1109fcd4-1706-4022-9422-f4e947f275af
#=╠═╡
function evaluate_gridworld_dqn(γ, steps, α, ϵ, buffer_size, batch_size, update_interval; nruns = 100, kwargs...)
	f(x) = dqn(gridworld_mdp, γ, typemax(Int64), steps; α = α, ϵ = ϵ, buffer_size = buffer_size, batch_size = batch_size, target_update_interval = update_interval) |> v -> eval_gridworld_returns(v; total_steps = steps, kwargs...)
	1:nruns |> Map(f) |> foldxt((a, b) -> a .+ b) |> v -> v ./ nruns
end
  ╠═╡ =#

# ╔═╡ b045eb8f-5911-4afe-b9aa-6bd291a2652c
#=╠═╡
plot([scatter(y = evaluate_gridworld_dqn(0.99f0, 20_000, 1f-4, 0.01f0, buffer_size, 512, 100; nruns = 100, interval = 100), name = "buffer size: $buffer_size") for buffer_size in [1000, 2000, 10_000]])
  ╠═╡ =#

# ╔═╡ 8621e3e0-f145-46a9-a08c-2f2327525e85
#=╠═╡
TableOfContents()
  ╠═╡ =#

# ╔═╡ ea607cf6-a263-4d44-9d72-dc9a5de055ad
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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
BenchmarkTools = "~1.6.3"
DataStructures = "~0.19.3"
PlutoDevMacros = "~0.9.2"
PlutoPlotly = "~0.6.5"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.79"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "c7d214e40b00e5489998fe95c088b0dc2a5411f0"

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

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

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
git-tree-sha1 = "80580012d4ed5a3e8b18c7cd86cebe4b816d17a6"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.9"

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
git-tree-sha1 = "3ac7038a98ef6977d44adeadc73cc6f596c08109"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.79"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

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
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

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
# ╟─78e8285e-5b98-420c-9fdc-cb943053a206
# ╠═0c0a7330-29bf-4326-8939-78b7e8b58d55
# ╠═cf40f4b3-4495-4f26-a007-18c6589ed4cf
# ╠═b3d7c539-d5a0-47fc-85bc-a62aafca8fa0
# ╠═576ff132-27d0-4a91-a955-e797fe6637c1
# ╠═a743f767-1ff6-4e1c-8c6f-88622d07c175
# ╠═f76f33f3-7cb3-4715-800c-9a0b2561f05b
# ╠═f7a94436-b905-48b5-a0f3-ae26d0ecab5e
# ╠═2dd9b971-fa6e-4a55-8a82-b16739199fab
# ╠═3a4510e6-054b-40fe-989d-7ac8c86db757
# ╟─05535cef-05f4-42a1-925c-ceb85bb6dfba
# ╠═830ba410-377a-423b-9e75-6884c8cbbbea
# ╟─78a997cb-146c-4cb5-b2ec-4bd1188909e6
# ╟─0fd4da20-cf83-4071-b8a1-4259cbba7d8c
# ╠═05fa2e5c-8633-4c43-89ba-09dd0c83cdfa
# ╟─c7f4374c-3151-443b-9d6a-85581c3f2438
# ╠═a3d54085-dbdc-4889-9313-efd1ece2150a
# ╠═696650a0-9e0f-45d0-a768-679b02688f06
# ╟─62bc710e-cd7e-43a5-bf9e-da024802358c
# ╠═01697aad-517e-4e48-b543-92d2974990ef
# ╠═293ee629-3fb9-442d-be27-11e9c0545fae
# ╠═8b141bcf-db3e-44d2-96de-c916f4018740
# ╠═8705bc78-b419-4b30-8a1e-02398fd456b5
# ╠═2753435d-8ea2-4bab-9b06-cb0e36bcab99
# ╟─76bb3fb6-836b-42a7-8051-48e0856bedb3
# ╠═9a70941b-bf9e-4b3c-aef6-915f3e2019fe
# ╠═e4fd6d59-be28-4852-a2d2-6ddc1a40116d
# ╟─e8c3f789-7ede-4f6d-83ee-3050c9ef5840
# ╠═81058b8e-c80c-4a39-b33e-15b42d1225b8
# ╠═4f222b09-00e0-48b9-bd3e-6b6ebfba5727
# ╠═ab6d36c1-5674-49f4-b291-f367840c9335
# ╠═809132ad-fad8-4bc7-b8f4-1f98dbc8503c
# ╠═b2096a04-74f8-4287-aad5-dc27752a21f7
# ╠═281718b9-3123-479f-8847-37e48b6298f7
# ╠═37a4303b-a4b9-423a-a5fa-3282c323f04b
# ╠═b1880149-4d45-4cfb-91cd-4c094ac5a1eb
# ╠═c5e0bbaf-a04d-4940-b7ab-3317f9d513a9
# ╠═daa89863-cb36-4a12-a699-646d8ba55904
# ╠═b067424c-2b40-4d99-9dba-419af6fb2209
# ╠═7edf78dd-9dad-4726-8173-c95f3e4ed6ab
# ╠═d4a59bc4-2c2a-4966-8e1a-e336e26d4d40
# ╠═16bdaa3d-77b2-433b-8343-54af3a121e85
# ╠═d437cfc2-6dfe-40a9-bb22-c90d25bebd25
# ╠═22744a27-614a-4317-a0ed-833bb0ef659c
# ╠═1a3493ac-966b-4260-8c06-0e60033ba41f
# ╠═f1130bab-babd-41e1-891c-00a2d846b39f
# ╠═95098580-9d74-41af-af28-c08c7dede5f4
# ╠═cd104c87-dd11-45a7-9ef5-ccecfdea3abd
# ╠═55f2b6a8-52e7-47bd-82a1-40fa0ff11d98
# ╟─3c11234c-5ea5-4709-b84b-ae477fb8dc55
# ╠═de6aca47-7d2c-4c8c-8df9-c3e16e6dc2bd
# ╠═d1058fb3-cb03-4031-b089-ce5f077036a0
# ╠═b045eb8f-5911-4afe-b9aa-6bd291a2652c
# ╠═1109fcd4-1706-4022-9422-f4e947f275af
# ╟─33b59f50-07e1-11f1-9748-31081ab2ceaf
# ╠═a966b2b2-b3d9-4f28-9042-66167400f2cb
# ╠═ad872f3c-7be0-4427-bd1d-5afe25b6e9fa
# ╠═8b4b8bfa-9dfd-45c4-9ce0-8f4af97f9721
# ╠═e6aefa92-94d5-487d-91ba-b9a4d1b3277c
# ╠═8621e3e0-f145-46a9-a08c-2f2327525e85
# ╠═ea607cf6-a263-4d44-9d72-dc9a5de055ad
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
