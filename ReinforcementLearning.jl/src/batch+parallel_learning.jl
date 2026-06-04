### A Pluto.jl notebook ###
# v0.20.25

using Markdown
using InteractiveUtils

# ╔═╡ c3953fda-dc68-48ca-b033-7a04dc2beae0
# ╠═╡ skip_as_script = true
#=╠═╡
using HypertextLiteral
  ╠═╡ =#

# ╔═╡ a966b2b2-b3d9-4f28-9042-66167400f2cb
using PlutoDevMacros

# ╔═╡ 8b4b8bfa-9dfd-45c4-9ce0-8f4af97f9721
using DataStructures

# ╔═╡ e6aefa92-94d5-487d-91ba-b9a4d1b3277c
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI, BenchmarkTools, PlutoPlotly, PlutoProfile
	TableOfContents(depth = 4)
end
  ╠═╡ =#

# ╔═╡ 78e8285e-5b98-420c-9fdc-cb943053a206
md"""
# Deep Q-Networks

To address some of the problems created by combining Q-learning with approximation techniques, DQN attempts to use a target network and a replay buffer to mitigate the moving target value problem and break correlations between consecutive samples.  
"""

# ╔═╡ 789ac927-e8dd-424a-8a28-b3240d295523
md"""
## Utility Functions
"""

# ╔═╡ 1b4c3482-165e-4c09-bbc2-c705e5ceb2fe
begin
	#idea is to fill the first column of the output matrix with the maximum values in a single pass efficiently
	function maximize_output_matrix!(output_matrix::Matrix{T}) where T<:Real
		for j in 2:size(output_matrix, 2)
			@inbounds @simd for i in 1:size(output_matrix, 1)
				output_matrix[i, 1] = max(output_matrix[i, 1], output_matrix[i, j])
			end
		end
	end
	
	#fill in the first column of target_output with the value of the index for the maximum value in the row of value output
	function maximize_output_matrix!(value_output::Matrix{T}, target_output::Matrix{T}) where T<:Real
		for j in 2:size(value_output, 2)
			@inbounds @simd for i in 1:size(value_output, 1)
				new_max = (value_output[i, j] > value_output[i, 1])
				value_output[i, 1] = max(value_output[i, 1], value_output[i, j])
				target_output[i, 1] = target_output[i, 1] * !new_max + target_output[i, j] * new_max
			end
		end
	end
end

# ╔═╡ f9d3ee23-f39d-46e4-834e-86b8eee1ce50
const FCANNActivationsBatch{T} = Vector{Matrix{T}} where T<:Float32

# ╔═╡ ce90a48f-4111-4f33-9448-a04af33e6231
md"""
## Algorithm
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

# ╔═╡ e5091bad-761c-49ef-a84a-0fec4ce2fbd1
md"""
### Random Performance
"""

# ╔═╡ 62bc710e-cd7e-43a5-bf9e-da024802358c
md"""
### Exact Solution
"""

# ╔═╡ be0e9e8f-e95c-42c6-99ac-c4291eed0f66
md"""
Note that the value function implies an average number of steps to completion of about 20 which is and average reward of 0.05 per step.


|Name|Num Steps|Avg Reward per Step|
|---|---|---|
|Random|3800|0.0003|
|Optimal|21|0.048|
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

# ╔═╡ fb94bc10-a580-4c32-b307-c551e4113bd7
md"""
#### Tabular Q-Learning
"""

# ╔═╡ afa30291-a919-44da-83c9-97cd2a43c168
md"""
#### Q-Learning with Linear Approximation
"""

# ╔═╡ 74e2001a-12a4-4a96-bb1c-567e238cf6a9
md"""
#### Q-Learning with Non-Linear Approximation
"""

# ╔═╡ 1a3493ac-966b-4260-8c06-0e60033ba41f
# ╠═╡ disabled = true
#=╠═╡
begin
	for α in 2f0 .^ (-15:-1)
		gridworld_value_studies.sarsa_linear_study.update_results!(0.99f0, α, 0.0f0, 100_000; ϵ = 0.05f0, compute_value = compute_q_learning_value)
	end
	display_study(gridworld_value_studies.sarsa_linear_study) |> df -> sort(df, :value; rev=true)
end
  ╠═╡ =#

# ╔═╡ f1130bab-babd-41e1-891c-00a2d846b39f
# ╠═╡ disabled = true
#=╠═╡
begin
	for α in 2f0 .^ (-9:-2)
		gridworld_value_studies.sarsa_nonlinear_study.update_results!(0.99f0, α, 0.0f0, 100_000, 16, 4, 1; ϵ = 0.01f0, compute_value = compute_sarsa_value)
	end
	display_study(gridworld_value_studies.sarsa_nonlinear_study) |> df -> sort(df, :value; rev=true)
end
  ╠═╡ =#

# ╔═╡ 3c11234c-5ea5-4709-b84b-ae477fb8dc55
md"""
### DQN
"""

# ╔═╡ 354a4a98-528f-46e2-93bd-c04f5a9ccad3
md"""
#### Tabular Approximation
"""

# ╔═╡ 710073d0-1af3-427a-9273-4044b252377b
md"""
#### Linear Approximation
"""

# ╔═╡ 14203d22-a8da-4e32-b2e8-d90936b83875
md"""
#### Non-Linear Approximation
"""

# ╔═╡ 7b5a691c-69be-4c8b-a2af-fc80b6597086
md"""
#### GPU Benchmark Evaluation
"""

# ╔═╡ 5d2836bc-9dde-4bb7-a45d-309de2292671
#4 threads is the best with 256x256 network with 512 batch size

# ╔═╡ 1d56e95a-fb19-4f28-a581-f27bc45b1149
md"""
# Policy Gradient Methods
"""

# ╔═╡ e9cf4424-6a73-4f22-9a89-91d99f5e92b7
md"""
## Reinforce
"""

# ╔═╡ 5df8574a-251b-49f4-8fc0-60988a1263d2
md"""
### Gridworld Example
"""

# ╔═╡ 16698d73-c9db-4680-966b-a246c7137e1e
md"""
#### Reinforce with Linear Approximation
"""

# ╔═╡ ddccecac-cc53-402b-b891-875ed332da6a
md"""
#### Reinforce with Non-Linear Approximation
"""

# ╔═╡ 52811707-b5e9-4b24-b7ca-c7786d1ad0b6
md"""
## Actor Critic
"""

# ╔═╡ 1bea29d7-7268-40b5-88c1-0d56b0d0c89d
md"""
### Gridworld Example
"""

# ╔═╡ 99190158-126a-4149-9d32-dffda0259cab
md"""
## Actor Critic with Synchronous Environments
"""

# ╔═╡ c113e64f-7463-4714-8253-40d196496b1d
md"""
### Utility Functions
"""

# ╔═╡ e6f4574b-f28f-43b1-b8f9-7080ecacfb39
#sample a batch of actions from a matrix of probability distributions.  each row represents a separate environment with its own action distribution.  Fill the respective action selections in the vector `actions`
function sample_batch_actions!(actions::Vector{I}, πs::Matrix{T}) where {T<:Real, I<:Integer} 
	num_env, num_actions = size(πs)
	actions .= one(I)
	maxvs = fill(T(-Inf), num_env)
	@inbounds @fastmath for i in 1:num_actions
		@simd for k in 1:num_env
			x = πs[k, i] 
			g = log(x) - log(-log(rand(T)))
			newmax = (g > maxvs[k])
			maxvs[k] = max(g, maxvs[k])
			actions[k] += newmax*(i - actions[k])
		end
	end
	return actions
end

# ╔═╡ 90d81200-8b3e-4b14-b913-7c8eb0a965b0
function update_row_extrema!(row_mins, row_maxes, πs)
	#populate both extrema with the values in the first column
	@inbounds @simd for i in eachindex(row_mins)
		row_mins[i] = πs[i, 1]
		row_maxes[i] = πs[i, 1]
	end
	
	for j in 2:size(πs, 2)
		@inbounds @simd for i in eachindex(row_mins)
			row_mins[i] = min(row_mins[i], πs[i, j])
			row_maxes[i] = max(row_maxes[i], πs[i, j])
		end
	end
	return row_mins, row_maxes
end

# ╔═╡ 948a1e0b-83ae-4989-812e-83be6df4c86b
md"""
### Algorithm
"""

# ╔═╡ e0bd880a-b962-4b15-873a-428fea0624ee
md"""
### Linear Approximation
"""

# ╔═╡ ff1d3dea-9aa9-4832-9474-0095924b747d
md"""
### Non-linear Approximation
"""

# ╔═╡ 034109fe-6b46-4d77-bbc8-f1399c02bdac
md"""
### Gridworld Example
"""

# ╔═╡ ed81e0f8-8b92-484f-b66a-0a7fda1b8dd2
#this environment does not do well with 1 step returns and actor critic methods, you can see reinforce performing much better in comparison.  Need to add n-step return option

# ╔═╡ ee0405a4-a03c-4373-a270-475cda8de910
md"""
### N-step Method
"""

# ╔═╡ be9639f8-e987-4448-a961-0cafaaaf4980
md"""
The issue with doing n-step returns in batches is how to handle the end of an episode.  So normally we would start collecting the step data up to N times before updating the state from step 1 and then the actual gradient updates would lag the data collection.  Now let's say an episode in one of the environments terminates.  Then you would still do the updates for the returns up to the termination point where now you don't need to do bootstrap estimates and the other episodes that have not terminated just continue on with the lagged state updates.  But when you reach the end of environment that did terminate, now you need to initiate a new episode and you would be able to do updates yet since you first need to collect enough of the data.  So the step sequence will be out of sync.  I think the best way to handle this is to allow each environment to run for multiple steps if needed to produce a viable gradient update state to batch together with the others.  To perform these updates I need to save a buffer for each environment with the feature vector of the state to be updated as well as the corresponding nstep return values and the feature vector of the bootstrap state if necessary.  I still need to perform all the updates as a batch so for each environment and each gradient update step I need to accumulate two sets of feature vectors and the list of return values in a buffer with a way to track if the episode has terminated.  Also for the total number of steps, I need to modify it to total batch gradient update steps since some of the environments will need to take extra steps to catch up with the others when a new episode starts.

Other information to track is the current state for every environment so I can do the steps but the buffers is where I will store the feature vectors.  So I should maintain two feature vector lists.  One for the current state being updated and the other for the bootstrap values and another vector for the discount factor being multiplied by the bootstrap values.  For episodes that have terminated, that factor will be zero.  In order to compute the discounted return values as well I will need to maintain a reward buffer for each environment of at least length n.

For tracking the problem I need to have the feature matrix I use for the current time step to do an action prediction.  But there are steps when only a subset of the environments need to do an update.  So initially I need to do an action prediction for every state.  Now let's say one of the environments has terminated.  So then that one is ready for a gradient update, but the other environments are not ready and will need to perform action prediction until either they reach N + 1 reward values or they also hit a terminal state.  So I can just leave the feature vector for those ready environments unchanged and only update the feature vectors for the other environments that still need to accumulate more steps.
"""

# ╔═╡ 4ba0a4d3-d04b-4f8d-94be-0c953b9d8719
md"""
### GPU Benchmark Evaluation
"""

# ╔═╡ 33b59f50-07e1-11f1-9748-31081ab2ceaf
md"""
# Dependencies
"""

# ╔═╡ ad872f3c-7be0-4427-bd1d-5afe25b6e9fa
@only_in_nb @fromparent import *

# ╔═╡ 0c0a7330-29bf-4326-8939-78b7e8b58d55
#fill in batch_inds with a uniform sample across the buffer_size or the current step, whichever is smaller
function update_batch_inds!(batch_inds::Vector{Int64}, step::Integer, buffer_size::Integer, N::Integer)
	l = min(buffer_size, step)
	sample!(1:(l - N), batch_inds; replace = false) #for N step returns make sure no index is too close to the end of the buffer
end

# ╔═╡ cf40f4b3-4495-4f26-a007-18c6589ed4cf
begin
	form_batch_action_value_args(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector, parameters, batch_size::Integer) where {T<:Real, S, A, P, F1, F2, F3} = ()

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

# ╔═╡ cb972f94-d22c-4d00-8c70-50daff8f697e
#note that the first four arguments are modified inside this function
function update_nstep_returns!(targets::Vector{T}, target_const::Vector{T}, feature_matrix, state_list::Vector{S}, γ::T, replay_buffer::CircularBuffer, batch_inds::Vector{Int64}, N::Integer) where {T<:Real, S}
	for i in eachindex(batch_inds)
		j = batch_inds[i]
		(x, i_a, r, x′, terminated, s′) = replay_buffer[j]
		g = r
		k = j+1
		while !terminated && (k <= j+N)
			(x, i_a, r, x′, terminated, s′) = replay_buffer[k]
			g += r * γ^(k - j)
			k += 1
		end
		update_feature_matrix!(feature_matrix, x′, i)
		state_list[i] = s′
		#populate target values with the reward 
		targets[i] = g
		target_const[i] = terminated ? zero(T) : γ^(k-j) #update constant to be used to multiply the target values.  Depending on the number of future steps, the discount rate is used but if the N-step window ends in termination then the output value is ignored
	end
end

# ╔═╡ 576ff132-27d0-4a91-a955-e797fe6637c1
#update target values using parameters, action_value computation function and batch_args which will vary depending on the type of network
begin
	#-------------------Single Q maximization
	#linear function approximation with a dense feature vector
	function update_targets!(targets::Vector{T}, state_list::Vector{S}, is_valid_action::Function, γ::T, replay_buffer, batch_inds::Vector{Int64}, N::Integer, target_const::Vector{T}, target_params::Matrix{T}, feature_matrix::Matrix{T}, action_values::Vector{T}, output_matrix::Matrix{T})  where {T<:Real, S}
		#update feature matrix with replay buffer
		update_nstep_returns!(targets, target_const, feature_matrix, state_list, γ, replay_buffer, batch_inds, N)

		#perform forward pass to fill in target values with function output
		LinearAlgebra.BLAS.gemm!('T', 'N', one(T), feature_matrix, target_params, zero(T), output_matrix)

		mask_invalid_actions!(output_matrix, state_list, is_valid_action)

		maximize_output_matrix!(output_matrix)

		#for non terminal states add to target discounted future function value
		@inbounds @simd for i in eachindex(batch_inds)
			# (_, _, _, _, terminated) = replay_buffer[batch_inds[i]]
			# if !terminated
				# targets[i] += γ * output_matrix[i, 1] # maximum(view(output_matrix, i, :))
			# end
			targets[i] += target_const[i] * output_matrix[i, 1]
		end
	end

	#linear function approximation with a binary feature vector
	function update_targets!(targets::Vector{T}, state_list::Vector{S}, is_valid_action::Function, γ::T, replay_buffer::CircularBuffer, batch_inds::Vector{Int64}, N::Integer, target_const::Vector{T}, target_params::Matrix{T}, feature_matrix::Vector{V}, action_values::Vector{T}, output_matrix::Matrix{T}) where {T<:Real, V<:AbstractBinaryFeatures, S}
		#update feature matrix with replay buffer
		for i in eachindex(batch_inds)
			j = batch_inds[i]
			(x, i_a, r, x′, terminated, s′) = replay_buffer[j]
			g = r
			k = j+1
			while !terminated && (k <= j+N)
				(x, i_a, r, x′, terminated, s′) = replay_buffer[k]
				g += r * γ^(k-j)
				k += 1
			end
			targets[i] = g
			if !terminated
				update_linear_action_values!(action_values, x′, target_params; is_valid_action = i_a -> is_valid_action(s′, i_a))
				targets[i] += γ^(k-j) * maximum(action_values)
			end
		end
	end

	#nonlinear gpu function approximation with a dense feature vector
	function update_targets!(targets::Vector{T}, state_list::Vector{S}, is_valid_action::Function, γ::T, replay_buffer, batch_inds::Vector{Int64}, N::Integer, target_const::Vector{T}, target_params::FCANNParamsGPU, feature_matrix, action_values::Vector{T}, output_matrix::Matrix{T}, activations::FCANNActivationsGPU, gpu_input::FCANN.CUDAArray) where {T<:Real, S}
		#update feature matrix with replay buffer
		update_nstep_returns!(targets, target_const, feature_matrix, state_list, γ, replay_buffer, batch_inds, N)
		input_orientation = get_input_orientation(feature_matrix)
		FCANN.memcpy!(gpu_input, feature_matrix)

		#perform forward pass to fill in target values with function output
		FCANN.forwardNOGRAD_base!(activations, target_params.weights..., gpu_input, target_params.reslayers; input_orientation = input_orientation)
		FCANN.memcpy!(output_matrix, activations[end])
		mask_invalid_actions!(output_matrix, state_list, is_valid_action)
		maximize_output_matrix!(output_matrix)

		#for non terminal states add to target discounted future function value
		@inbounds @simd for i in eachindex(batch_inds)
			# (_, _, _, _, terminated) = replay_buffer[batch_inds[i]]
			# if !terminated
				# targets[i] += γ * activations[end][i, 1] # maximum(view(output_matrix, i, :))
			# end
			targets[i] += target_const[i] * output_matrix[i, 1]
		end
	end

	#nonlinear function approximation with a dense feature vector
	function update_targets!(targets::Vector{T}, state_list::Vector{S}, is_valid_action::Function, γ::T, replay_buffer, batch_inds::Vector{Int64}, N::Integer, target_const::Vector{T}, target_params::FCANNParams{T}, feature_matrix, action_values::Vector{T}, output_matrix::Matrix{T}, activations::FCANNActivationsBatch{T}) where {T<:Real, S}
		#update feature matrix with replay buffer
		update_nstep_returns!(targets, target_const, feature_matrix, state_list, γ, replay_buffer, batch_inds, N)

		input_orientation = get_input_orientation(feature_matrix)

		#perform forward pass to fill in target values with function output
		FCANN.forwardNOGRAD_base!(activations, target_params.weights..., feature_matrix, target_params.reslayers; input_orientation = input_orientation)
		output_matrix .= activations[end]
		mask_invalid_actions!(output_matrix, state_list, is_valid_action)
		maximize_output_matrix!(activations[end])

		#for non terminal states add to target discounted future function value
		@inbounds @simd for i in eachindex(batch_inds)
			# (_, _, _, _, terminated) = replay_buffer[batch_inds[i]]
			# if !terminated
				# targets[i] += γ * activations[end][i, 1] # maximum(view(output_matrix, i, :))
			# end
			targets[i] += target_const[i] * activations[end][i, 1]
		end
	end

	#-------------- Double Q Maximization
	#linear function approximation with a dense feature vector
	function update_targets!(targets::Vector{T}, state_list::Vector{S}, is_valid_action::Function, γ::T, replay_buffer, batch_inds::Vector{Int64}, N::Integer, target_const::Vector{T}, target_params::Matrix{T}, value_params::Matrix{T}, feature_matrix::Matrix{T}, action_values::Vector{T}, target_output::Matrix{T}, value_output::Matrix{T}) where {T<:Real, S}
		#update feature matrix with replay buffer
		update_nstep_returns!(targets, target_const, feature_matrix, state_list, γ, replay_buffer, batch_inds, N)

		#perform forward pass to fill in target values with function output times the discount rate plus the reward
		LinearAlgebra.BLAS.gemm!('T', 'N', γ, feature_matrix, target_params, zero(T), target_output)
		LinearAlgebra.BLAS.gemm!('T', 'N', γ, feature_matrix, value_params, zero(T), value_output)

		mask_invalid_actions!(target_output, state_list, is_valid_action)
		mask_invalid_actions!(value_output, state_list, is_valid_action)

		maximize_output_matrix!(value_output, target_output)

		#for non terminal states add to target discounted future function value
		@inbounds @simd for i in eachindex(batch_inds)
			# (_, _, _, _, terminated) = replay_buffer[batch_inds[i]]
			# if !terminated
			# 	targets[i] += γ * target_output[i, 1] # maximum(view(output_matrix, i, :))
			# end
			targets[i] += target_const[i] * target_output[i, 1]
		end
	end

	#linear function approximation with a binary feature vector
	function update_targets!(targets::Vector{T}, state_list::Vector{S}, is_valid_action::Function, γ::T, replay_buffer::CircularBuffer, batch_inds::Vector{Int64}, N::Integer, target_const::Vector{T}, target_params::Matrix{T}, value_params::Matrix{T}, feature_matrix::Vector{V}, action_values::Vector{T}, target_output::Matrix{T}, value_output::Matrix{T}) where {T<:Real, S, V<:AbstractBinaryFeatures}
		#update feature matrix with replay buffer
		for i in eachindex(batch_inds)
			j = batch_inds[i]
			(x, i_a, r, x′, terminated, s′) = replay_buffer[j]
			g = r
			k = j+1
			while !terminated && (k <= j+N)
				(x, i_a, r, x′, terminated, s′) = replay_buffer[k]
				g += r * γ^(k-j)
				k += 1
			end
			targets[i] = g
			if !terminated
				update_linear_action_values!(action_values, x′, value_params; is_valid_action = i_a -> is_valid_action(s′, i_a))
				i_a_max = argmax(action_values)
				update_linear_action_values!(action_values, x′, target_params; is_valid_action = i_a -> is_valid_action(s′, i_a))
				targets[i] += γ^(k-j) * action_values[i_a_max]
			end
		end
	end

	#nonlinear function approximation with a dense feature vector
	function update_targets!(targets::Vector{T}, state_list::Vector{S}, is_valid_action::Function, γ::T, replay_buffer, batch_inds::Vector{Int64}, N::Integer, target_const::Vector{T}, target_params::FCANNParams{T}, value_params::FCANNParams{T}, feature_matrix, action_values::Vector{T}, target_output::Matrix{T}, value_output::Matrix{T}, activations::FCANNActivationsBatch{T}) where {T<:Real, S}
		#update feature matrix with replay buffer
		update_nstep_returns!(targets, target_const, feature_matrix, state_list, γ, replay_buffer, batch_inds, N)

		input_orientation = get_input_orientation(feature_matrix)

		#perform forward pass to fill in target values with function output
		FCANN.forwardNOGRAD_base!(activations, target_params.weights..., feature_matrix, target_params.reslayers; input_orientation = input_orientation)
		target_output .= activations[end]
		FCANN.forwardNOGRAD_base!(activations, value_params.weights..., feature_matrix, value_params.reslayers; input_orientation = input_orientation)
		value_output .= activations[end]
		mask_invalid_actions!(target_output, state_list, is_valid_action)
		maximize_output_matrix!(value_output, target_output)

		#for non terminal states add to target discounted future function value
		@inbounds @simd for i in eachindex(batch_inds)
			# (_, _, _, _, terminated) = replay_buffer[batch_inds[i]]
			# if !terminated
			# 	targets[i] += γ * target_output[i, 1] # maximum(view(output_matrix, i, :))
			# end
			targets[i] += target_const[i] * target_output[i, 1]
		end
	end

	#nonlinear gpu function approximation with a dense feature vector
	function update_targets!(targets::Vector{T}, state_list::Vector{S}, is_valid_action::Function, γ::T, replay_buffer, batch_inds::Vector{Int64}, N::Integer, target_const::Vector{T}, target_params::FCANNParamsGPU, value_params::FCANNParamsGPU, feature_matrix, action_values::Vector{T}, target_output::Matrix{T}, value_output::Matrix{T}, activations::FCANNActivationsGPU, gpu_input::FCANN.CUDAArray) where {T<:Real, S}
		#update feature matrix with replay buffer
		update_nstep_returns!(targets, target_const, feature_matrix, state_list, γ, replay_buffer, batch_inds, N)
		input_orientation = get_input_orientation(feature_matrix)
		FCANN.memcpy!(gpu_input, feature_matrix)

		#perform forward pass to fill in target values with function output
		FCANN.forwardNOGRAD_base!(activations, target_params.weights..., gpu_input, target_params.reslayers; input_orientation = input_orientation)
		FCANN.memcpy!(target_output, activations[end])
		FCANN.forwardNOGRAD_base!(activations, value_params.weights..., gpu_input, value_params.reslayers; input_orientation = input_orientation)
		FCANN.memcpy!(value_output, activations[end])
		mask_invalid_actions!(target_output, state_list, is_valid_action)
		mask_invalid_actions!(value_output, state_list, is_valid_action)
		maximize_output_matrix!(value_output, target_output)

		#for non terminal states add to target discounted future function value
		@inbounds @simd for i in eachindex(batch_inds)
			# (_, _, _, _, terminated) = replay_buffer[batch_inds[i]]
			# if !terminated
			# 	targets[i] += γ * target_output[i, 1] # maximum(view(output_matrix, i, :))
			# end
			targets[i] += target_const[i] * target_output[i, 1]
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

# ╔═╡ 44f59eb0-a7e5-43ff-b4ca-657cc505220f
begin
	function get_input_dimension(X::Matrix{T}) where T<:Real
		(m, l) = size(X)
		return l, m
	end

	function get_input_dimension(X::Vector{V}) where V
		m = length(X)
		l = length(first(X))
		return l, m
	end

	function get_input_dimension(X::FCANN.CUDAArray)
		m = X.size[1]
		l = X.size[2]
		return l, m
	end
end

# ╔═╡ f7a94436-b905-48b5-a0f3-ae26d0ecab5e
begin
	function ReinforcementLearning.update_fcann_value_gradient!(∇q̂::FCANNParams{T}, value_params::FCANNParams{T}, feature_matrix, targets::Vector{T}, output_indices::Vector{I}, hidden_layers::Vector{Int64}, l2::T, tanh_grad_z::FCANNActivationsBatch{T}, activations::FCANNActivationsBatch{T}, deltas::FCANNActivationsBatch{T}, onesvec::Vector{T}, dropout::T, activation_list::AbstractVector{B}) where {T<:Float32, B<:Bool, I<:Integer}
		FCANN.nnCostFunction(value_params.weights..., hidden_layers, feature_matrix, targets, output_indices, l2, ∇q̂.weights..., tanh_grad_z, activations, deltas, onesvec, dropout; resLayers = value_params.reslayers, costFunc = "sqErr", activation_list = activation_list, input_orientation = get_input_orientation(feature_matrix))
	end

	function ReinforcementLearning.update_fcann_value_gradient!(∇q̂::FCANNParamsGPU, value_params::FCANNParamsGPU, feature_matrix::FCANN.CUDAArray, targets::FCANN.CUDAArray, output_indices::FCANN.CUDAArray, hidden_layers::Vector{Int64}, l2::T, tanh_grad_z::FCANNActivationsGPU, activations::FCANNActivationsGPU, deltas::FCANNActivationsGPU, onesvec::FCANN.CUDAArray, dropout::T, activation_list::AbstractVector{B}) where {T<:Float32, B<:Bool}
		input_layer_size, m = get_input_dimension(feature_matrix)
		output_layer_size = activations[end].size[2]
		FCANN.nnCostFunction(value_params.weights..., input_layer_size, output_layer_size, hidden_layers, m, onesvec, activations, tanh_grad_z, deltas, ∇q̂.weights..., feature_matrix, targets, output_indices, l2, dropout; resLayers = value_params.reslayers, costFunc = "sqErrIndex", activation_list = activation_list, input_orientation = 'T')
	end
end

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

# ╔═╡ 1805574f-a668-477f-a6a9-e7ee29ce08bf
function Base.copy!(dst::FCANNParamsGPU, src::FCANNParamsGPU)
	for i in eachindex(src.weights[1])
		for j in 1:2
			FCANN.memcpy!(dst.weights[j][i], src.weights[j][i])
		end
	end
end

# ╔═╡ 75ba6587-ebe0-4f54-89f6-a65ec26abd63
begin
	#since the gradient is created inside the dqn! function it needs to handle any cleanup if necessary.  In this case it only applies to a gpu array where we want the memory explicitly freed
	cleanup_gradient!(∇q̂) = return nothing
	function cleanup_gradient!(∇q̂::FCANNParamsGPU) 
		FCANN.clear_gpu_data(∇q̂.weights[1])
		FCANN.clear_gpu_data(∇q̂.weights[2])
	end
end

# ╔═╡ 6fe10947-cdc4-48c5-8fbe-942714640dca
function ReinforcementLearning.setup_fcann_action_value_arguments(value_params::FCANNParams{T}, target_params::FCANNParams{T}, batch_size::Integer, l2::T, dropout::T, use_μP::Bool, activation_list; use_gpu = false) where {T<:Real}
	input_length, hidden_layers, num_hidden = get_network_dimensions(value_params)
	input_length2, hidden_layers2, num_hidden2 = get_network_dimensions(target_params)
	@assert input_length == input_length2 "Value and target networks don't share the same input dimension"
	@assert hidden_layers == hidden_layers2 "Value and target networks don't share the same hidden layers"
	@assert value_params.reslayers == target_params.reslayers "Value and target networks don't share the same skip connections"
	
	#form activations for network
	activations_batch = FCANN.form_activations(value_params.weights[1], batch_size)
	activations = FCANN.form_activations(value_params.weights[1])
	tanh_grad_z = deepcopy(activations_batch)
	deltas = deepcopy(activations_batch)
	onesvec = ones(T, batch_size)

	#note that the scales are multiplied by -1 to minimize loss in gradient update
	scales = fill(-one(T), length(value_params.weights[1]))
	if use_μP
		for i in eachindex(hidden_layers)
			i′ = i + 1
			scales[i′] /= size(value_params.weights[1][i′], 2)
		end
	end

	function update_action_values!(action_values::Vector{T}, x, params; activations::FCANNActivations{T} = activations, is_valid_action::Function = i_a -> true, kwargs...) 
		fcann_value_function!(activations, x, params)
		action_values .= activations[end]
		mask_invalid_actions!(action_values, is_valid_action)
		val, index = findmax(action_values)
		isnan(val) && error("Got NaN action value inside $action_values")
		isinf(val) && error("Got Inf action value inside $action_values")
		return (val, index)
	end
	
	function update_value_gradient!(∇q̂::FCANNParams{T}, params::FCANNParams{T}, targets::Vector{T}, output_inds::Vector{I}, feature_matrix, output_matrix::Matrix{T}) where I<:Integer
		update_fcann_value_gradient!(∇q̂, params, feature_matrix, targets, output_inds, hidden_layers, l2, tanh_grad_z, activations_batch, deltas, onesvec, dropout, activation_list)
		scale_fcann_params!(∇q̂, scales) #note that this also multiplies the gradient by -1 to account for minimization
		return ∇q̂
	end

	output = (update_action_values! = update_action_values!, update_value_gradient! = update_value_gradient!, target_args = (activations_batch,))

	!use_gpu && return output

	if in(:GPU, backendList)
		d_activations = FCANN.device_allocate(activations)
		d_activations_batch = FCANN.device_allocate(activations_batch)
		d_tanh_grad_z = FCANN.device_allocate(tanh_grad_z)
		d_deltas = FCANN.device_allocate(deltas)
		d_value_params = initialize_gpu_params(value_params)
		d_target_params = initialize_gpu_params(target_params)
		d_x = FCANN.cuda_allocate(zeros(T, input_length))
		d_feature_matrix = FCANN.cuda_allocate(zeros(T, input_length, batch_size))
		d_targets = FCANN.cuda_allocate(zeros(T, batch_size))
		d_output_inds = FCANN.cuda_allocate(zeros(Cint, batch_size))
		output_inds2 = zeros(Cint, batch_size)
		d_onesvec = FCANN.cuda_allocate(onesvec)

		gpu_feature_update! = setup_gpu_feature(zeros(T, input_length), update_feature_vector!)

		#x is always going to come from the replay buffer and hence will be an ordinary vector
		function update_action_values!(action_values::Vector{T}, x::Vector{T}, params::FCANNParamsGPU; d_x::FCANN.CUDAArray = d_x, d_activations::FCANNActivationsGPU = d_activations, is_valid_action::Function = i_a -> true, kwargs...)		
			FCANN.memcpy!(d_x, x)
			fcann_value_function!(d_activations, d_x, params)
			FCANN.memcpy!(action_values, d_activations[end])
			mask_invalid_actions!(action_values, is_valid_action)
			val, index = findmax(action_values)
			isnan(val) && error("Got NaN action value inside $action_values")
			isinf(val) && error("Got Inf action value inside $action_values")
			return (val, index)
		end

		function update_value_gradient!(∇q̂::FCANNParamsGPU, params::FCANNParamsGPU, targets::Vector{T}, output_inds::Vector{I}, feature_matrix::Matrix{T}, output_matrix::Matrix{T}) where {T<:Real, I<:Integer}
			FCANN.memcpy!(d_feature_matrix, feature_matrix)
			FCANN.memcpy!(d_targets, targets)
			output_inds2 .= Cint.(output_inds .- 1) #note that the GPU uses zero indexing and 32 bit integers
			FCANN.memcpy!(d_output_inds, output_inds2)
			update_fcann_value_gradient!(∇q̂, params, d_feature_matrix, d_targets, d_output_inds, hidden_layers, l2, d_tanh_grad_z, d_activations_batch, d_deltas, d_onesvec, dropout, activation_list)
			scale_fcann_params!(∇q̂, scales)
			return ∇q̂
		end

		function cleanup_vars()
			FCANN.clear_gpu_data(d_value_params.weights[1])
			FCANN.clear_gpu_data(d_value_params.weights[2])
			FCANN.clear_gpu_data(d_target_params.weights[1])
			FCANN.clear_gpu_data(d_target_params.weights[2])
			FCANN.clear_gpu_data(d_deltas)
			FCANN.clear_gpu_data(d_tanh_grad_z)
			FCANN.clear_gpu_data([d_x])
			FCANN.clear_gpu_data([d_feature_matrix])
			FCANN.clear_gpu_data([d_targets])
			FCANN.clear_gpu_data([d_output_inds])
			FCANN.clear_gpu_data(d_activations)
			FCANN.clear_gpu_data(d_activations_batch)
			FCANN.clear_gpu_data([d_onesvec])
		end

		gpu_args = (value_params = d_value_params, target_params = d_target_params, target_args = (d_activations_batch, d_feature_matrix), cleanup_vars = cleanup_vars)
	else
		gpu_args = ()
	end

	return (;output..., gpu_args = gpu_args)
end

# ╔═╡ a3d54085-dbdc-4889-9313-efd1ece2150a
# ╠═╡ skip_as_script = true
#=╠═╡
const gridworld_mdp = make_stochastic_gridworld(;wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
  ╠═╡ =#

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

# ╔═╡ 018a26c6-680e-42ed-b1ff-96b377808e11
#=╠═╡
function plot_gridworld_state_value_function(v̂::Function)
	zs = zeros(Float32, 7, 10)
	xs = 1:10
	ys = 1:7
	for (i, s) in enumerate(gridworld_mdp.states)
		out = v̂(s)
		zs[s.y, s.x] = out
	end
	tr = heatmap(x = xs, y = ys, z = zs)
	plot(tr, Layout(yaxis_scaleanchor = "x", xaxis_ticknames = 1:10, xaxis_tickvals = 1:10, width = 560))
end
  ╠═╡ =#

# ╔═╡ 2bfcda4a-e61a-4f85-bce2-b788e9372de2
#=╠═╡
function plot_gridworld_policy_function(π::Function)
	zs = [zeros(Float32, 7, 10) for _ in 1:4]
	xs = 1:10
	ys = 1:7
	for (i, s) in enumerate(gridworld_mdp.states)
		out = π(s)
		for i in eachindex(out)
			zs[i][s.y, s.x] = out[i]
		end
	end
	action_names = ["Up", "Down", "Left", "Right"]
	trs = [heatmap(x = xs, y = ys, z = zs[i]) for i in 1:4]
	plots = [plot(trs[i], Layout(title = "$(action_names[i]) Action Probabilities", yaxis_scaleanchor = "x", xaxis_ticknames = 1:10, xaxis_tickvals = 1:10, width = 560)) for i in 1:4]
	@htl("""
		 <div style = "display: flex;">
		 $(plots[1:2])
		 </div>

		<div style = "display: flex;">
		 $(plots[3:4])
		 </div>
		 """)
end
  ╠═╡ =#

# ╔═╡ 696650a0-9e0f-45d0-a768-679b02688f06
#=╠═╡
const gridworld_state_mdp = StateMDP(gridworld_mdp)
  ╠═╡ =#

# ╔═╡ 01697aad-517e-4e48-b543-92d2974990ef
# ╠═╡ show_logs = false
#=╠═╡
const gridworld_exact = value_iteration_v(gridworld_mdp, 0.99f0; save_history = false)
  ╠═╡ =#

# ╔═╡ 293ee629-3fb9-442d-be27-11e9c0545fae
#=╠═╡
plot_gridworld_value_function(gridworld_exact.final_value)
  ╠═╡ =#

# ╔═╡ 8705bc78-b419-4b30-8a1e-02398fd456b5
#=╠═╡
begin
	function eval_gridworld_final_policy(π::Matrix{T}; samples = 10_000, max_steps = 1_000) where T<:Real
		1:samples |> Map(a -> runepisode(gridworld_mdp; π = π, max_steps = max_steps)[5]) |> tcollect |> summarystats
	end

	function eval_gridworld_final_policy(π::Function; samples = 10_000, max_steps = 1_000)
		state_mdp = StateMDP(gridworld_mdp)
		1:samples |> Map(a -> runepisode(state_mdp; π = π, max_steps = max_steps)[5]) |> tcollect |> summarystats
	end
end
  ╠═╡ =#

# ╔═╡ 57c4027a-0209-4e22-917a-6b1e4364186c
#=╠═╡
eval_gridworld_final_policy(make_random_policy(gridworld_mdp), max_steps = 100_000)
  ╠═╡ =#

# ╔═╡ 2753435d-8ea2-4bab-9b06-cb0e36bcab99
#=╠═╡
eval_gridworld_final_policy(gridworld_exact.optimal_policy)
  ╠═╡ =#

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

# ╔═╡ 22744a27-614a-4317-a0ed-833bb0ef659c
#=╠═╡
const gridworld_value_studies = setup_episodic_value_parameter_studies(gridworld_state_mdp, gridworld_feature, update_gridworld_feature!; use_steps = true, min_reward = 0f0)
  ╠═╡ =#

# ╔═╡ 81058b8e-c80c-4a39-b33e-15b42d1225b8
#=╠═╡
const gridworld_q = sarsa_λ(gridworld_mdp, 0.99f0, 0f0, typemax(Int64), 100_000; α = 3f-4, ϵ = 0.1f0)
  ╠═╡ =#

# ╔═╡ 4f222b09-00e0-48b9-bd3e-6b6ebfba5727
#=╠═╡
eval_gridworld_final_policy(s -> gridworld_q.value_function(s).maximizing_action)
  ╠═╡ =#

# ╔═╡ b067424c-2b40-4d99-9dba-419af6fb2209
#=╠═╡
function eval_gridworld_returns(output::NamedTuple; total_steps = 100_000, interval = 100, kwargs...)
	step_rewards = zeros(Float32, total_steps+1)
	step_rewards[output.episode_steps] .= 1f0
	[mean(view(step_rewards, i-interval+1:i)) for i in interval:interval:total_steps+1]
end
  ╠═╡ =#

# ╔═╡ b1880149-4d45-4cfb-91cd-4c094ac5a1eb
#=╠═╡
function evaluate_gridworld_q_learning(γ, steps, α, ϵ; λ=0f0, nruns = 100, kwargs...)
	f(x) = sarsa_λ(gridworld_mdp, γ, λ, typemax(Int64), steps; α = α, ϵ = ϵ) |> v -> eval_gridworld_returns(v; total_steps = steps, kwargs...)
	1:nruns |> Map(f) |> foldxt((a, b) -> a .+ b) |> v -> v ./ nruns
end
  ╠═╡ =#

# ╔═╡ daa89863-cb36-4a12-a699-646d8ba55904
#=╠═╡
plot([scatter(y = evaluate_gridworld_q_learning(0.99f0, 20_000, α, 0.01f0; nruns = 100, interval = 100), name = "α = $α") for α in [1f-2, 1f-3, 1f-4]], Layout(title = "Q-learning Gridworld Rewards"))
  ╠═╡ =#

# ╔═╡ e3fa7c27-706c-445b-a2e2-837c91fcb9a7
#=╠═╡
plot([scatter(y = evaluate_gridworld_q_learning(0.99f0, 20_000, 1f-2, 0.01f0; λ=λ, nruns = 100, interval = 100), name = "λ = $λ") for λ in [0f0, 0.5f0, 0.9f0]], Layout(title = "Q-learning Gridworld Rewards"))
  ╠═╡ =#

# ╔═╡ c5e0bbaf-a04d-4940-b7ab-3317f9d513a9
#=╠═╡
function evaluate_gridworld_q_learning2(γ, steps, α, ϵ; λ=0f0, nruns = 100, kwargs...)
	f(x) = sarsa_λ_linear(gridworld_state_mdp, γ, λ, typemax(Int64), steps, copy(gridworld_feature), update_gridworld_feature!; α = α, ϵ = ϵ) |> v -> eval_gridworld_returns(v; total_steps = steps, kwargs...)
	1:nruns |> Map(f) |> foldxt((a, b) -> a .+ b) |> v -> v ./ nruns
end
  ╠═╡ =#

# ╔═╡ 7edf78dd-9dad-4726-8173-c95f3e4ed6ab
#=╠═╡
plot([scatter(y = evaluate_gridworld_q_learning2(0.99f0, 100_000, α, 0.05f0; nruns = 100, interval = 100), name = "α = $α") for α in [8f-2, 1f-2, 5f-3]], Layout(title = "Q-learning Gridworld Rewards"))
  ╠═╡ =#

# ╔═╡ 239e7e55-e68f-40f0-a666-c4c4c3ff3d35
#=╠═╡
plot([scatter(y = evaluate_gridworld_q_learning2(0.99f0, 100_000, 1f-2, 0.05f0; λ = λ, nruns = 100, interval = 100), name = "λ = $λ") for λ in [0f0, 0.2f0, 0.5f0]], Layout(title = "Q-learning Gridworld Rewards"))
  ╠═╡ =#

# ╔═╡ ab6d36c1-5674-49f4-b291-f367840c9335
#=╠═╡
const gridworld_q2 = sarsa_λ_linear(gridworld_state_mdp, 0.99f0, .5f0, typemax(Int64), 100_000, gridworld_feature, update_gridworld_feature!; α = 0.01f0, ϵ = 0.01f0, compute_value = compute_q_learning_value)
  ╠═╡ =#

# ╔═╡ 809132ad-fad8-4bc7-b8f4-1f98dbc8503c
#=╠═╡
eval_gridworld_final_policy(s -> gridworld_q2.value_function(s).maximizing_action)
  ╠═╡ =#

# ╔═╡ b2096a04-74f8-4287-aad5-dc27752a21f7
#=╠═╡
[plot_gridworld_value_function(gridworld_q2.value_function), plot_gridworld_value_function(gridworld_exact.final_value)]
  ╠═╡ =#

# ╔═╡ d4a59bc4-2c2a-4966-8e1a-e336e26d4d40
#=╠═╡
const gridworld_q3 = sarsa_λ_fcann(gridworld_state_mdp, 0.99f0, 0.5f0, typemax(Int64), 100_000, gridworld_feature, update_gridworld_feature!, [32, 32]; α = .15f0, ϵ = 0.05f0, reslayers = 1)
  ╠═╡ =#

# ╔═╡ 16bdaa3d-77b2-433b-8343-54af3a121e85
#=╠═╡
eval_gridworld_final_policy(s -> gridworld_q3.value_function(s).maximizing_action)
  ╠═╡ =#

# ╔═╡ d437cfc2-6dfe-40a9-bb22-c90d25bebd25
#=╠═╡
plot_gridworld_value_function(gridworld_q3.value_function)
  ╠═╡ =#

# ╔═╡ 95098580-9d74-41af-af28-c08c7dede5f4
function display_study(study::NamedTuple)
	results = study.results
	DataFrame(begin
		(;k..., value = results[k])
	end
	for k in keys(results))
end

# ╔═╡ cd104c87-dd11-45a7-9ef5-ccecfdea3abd
#=╠═╡
function evaluate_gridworld_q_learning3(γ, steps, α, ϵ; nruns = 100, hidden_layers = [8, 8], λ = 0f0, kwargs...)
	f(x) = sarsa_λ_fcann(gridworld_state_mdp, γ, λ, typemax(Int64), steps, copy(gridworld_feature), update_gridworld_feature!, hidden_layers; α = α, ϵ = ϵ, compute_value = compute_q_learning_value) |> v -> eval_gridworld_returns(v; total_steps = steps, kwargs...)
	1:nruns |> Map(f) |> foldxt((a, b) -> a .+ b) |> v -> v ./ nruns
end
  ╠═╡ =#

# ╔═╡ 55f2b6a8-52e7-47bd-82a1-40fa0ff11d98
#=╠═╡
plot([scatter(y = evaluate_gridworld_q_learning3(0.99f0, 100_000, 4f-3, 0.05f0; nruns = 40, λ = λ, hidden_layers = [32, 32]), name = "λ = $λ") for λ in [0f0, 0.1f0, 0.2f0, 0.5f0]], Layout(title = "Q-learning Gridworld Rewards"))
  ╠═╡ =#

# ╔═╡ 98c778fb-aa6d-4372-80cc-e6a6751bee96
LinearAlgebra.BLAS.set_num_threads(4)

# ╔═╡ 5d4dfd0c-440e-4187-a9fd-05ef01ec8eab
# ╠═╡ skip_as_script = true
#=╠═╡
function convert_tabular_gridworld_policy(π::Matrix{T}) where T<:Real
	function policy(s::GridworldState)
		i_s = gridworld_mdp.state_index[s]
		return π[:, i_s]
	end
end
  ╠═╡ =#

# ╔═╡ 0f3874fb-134f-4727-adc3-3eb1b3e1fd7d
#=╠═╡
plot_gridworld_policy_function(π::Matrix{T}) where T<:Real = plot_gridworld_policy_function(convert_tabular_gridworld_policy(π))
  ╠═╡ =#

# ╔═╡ 7e4a2949-6ec5-4761-9be0-777b989256dd
#=╠═╡
plot_gridworld_policy_function(gridworld_exact.optimal_policy)
  ╠═╡ =#

# ╔═╡ 7604fb1e-e6ed-4314-be7b-a786c6422e10
#=╠═╡
const gridworld_reinforce = reinforce_monte_carlo_control_linear(gridworld_state_mdp, 0.99f0, 10_000, gridworld_feature, update_gridworld_feature!; α = 1f-2, max_steps = 100)
  ╠═╡ =#

# ╔═╡ 7603c756-4132-4166-b385-71e99fa69f40
#=╠═╡
eval_gridworld_final_policy(gridworld_reinforce.policy_sample_action)
  ╠═╡ =#

# ╔═╡ fbd74aff-5107-40cd-a02c-a0c5e0b19464
#=╠═╡
function evaluate_gridworld_reinforce_linear(num_episodes::Integer, α::T; num_trials = Base.Threads.nthreads(), max_steps = 1_000) where T<:Real
	1:num_trials |> Map() do _
		output = reinforce_monte_carlo_control_linear(gridworld_state_mdp, 0.99f0, num_episodes, copy(gridworld_feature), update_gridworld_feature!; α = α, max_steps = max_steps)
		steps = output.episode_steps
		[ismissing(a) ? max_steps : a for a in steps]
	end |> foldxt((a, b) -> a .+ b) |> v -> v ./ num_trials
end	
  ╠═╡ =#

# ╔═╡ 3a7a2d32-23e1-41da-b7d0-52bb29b31def
#=╠═╡
function evaluate_gridworld_reinforce_linear(num_episodes::Integer, α_θ::T, α_w::T; num_trials = Base.Threads.nthreads(), max_steps = 1_000) where T<:Real
	1:num_trials |> Map() do _
		output = reinforce_with_baseline_monte_carlo_control_linear(gridworld_state_mdp, 0.99f0, num_episodes, copy(gridworld_feature), update_gridworld_feature!; α_θ = α_θ, α_w = α_w, max_steps = max_steps)
		steps = output.episode_steps
		[ismissing(a) ? max_steps : a for a in steps]
	end |> foldxt((a, b) -> a .+ b) |> v -> v ./ num_trials
end	
  ╠═╡ =#

# ╔═╡ 02253513-2802-4ecb-b14c-61e5d4e7fd86
#=╠═╡
[scatter(y = inv.(evaluate_gridworld_reinforce_linear(1_000, α)), name = "α: $α") for α in [1f-9, 1f-8, 1f-7]] |> plot
  ╠═╡ =#

# ╔═╡ b52c50ea-cd77-4560-b8f2-863962ff2546
#=╠═╡
[scatter(y = inv.(evaluate_gridworld_reinforce_linear(1_000, α_θ, α_w)), name = "α_θ: $α_θ, α_w: $α_w") for α_θ in [1f-2, 2f-2, 4f-2, 8f-2, 16f-2] for α_w in [2f-2, 4f-2, 8f-2]] |> plot
  ╠═╡ =#

# ╔═╡ 4b510e94-4ec2-4e82-ae8f-523ac90f34d9
#=╠═╡
const gridworld_reinforce2 = reinforce_with_baseline_monte_carlo_control_linear(gridworld_state_mdp, 0.99f0, 1_000, gridworld_feature, update_gridworld_feature!; α_θ = 8f-2, α_w = 2f-2, max_steps = 1000)
  ╠═╡ =#

# ╔═╡ 465ee26d-cf29-4dde-9ce6-c2030699996d
#=╠═╡
eval_gridworld_final_policy(gridworld_reinforce2.policy_sample_action)
  ╠═╡ =#

# ╔═╡ 5eca8610-aa4b-4ed8-a97c-43dbc2bae53c
#=╠═╡
plot_gridworld_policy_function(gridworld_reinforce2.policy_function)
  ╠═╡ =#

# ╔═╡ 9caedc2f-94d7-451f-85c8-28fdd387fb08
#=╠═╡
function evaluate_gridworld_reinforce_fcann(num_episodes::Integer, α_θ::T, α_w::T; hidden_layers = [32, 32], num_trials = Base.Threads.nthreads(), max_steps = 1_000) where T<:Real
	1:num_trials |> Map() do _
		output = reinforce_with_baseline_monte_carlo_control_fcann(gridworld_state_mdp, 0.99f0, num_episodes, copy(gridworld_feature), update_gridworld_feature!, hidden_layers; reslayers=1, α_θ = α_θ, α_w = α_w, max_steps = max_steps)
		steps = output.episode_steps
		[ismissing(a) ? max_steps : a for a in steps]
	end |> foldxt((a, b) -> a .+ b) |> v -> v ./ num_trials
end	
  ╠═╡ =#

# ╔═╡ f6e1e30c-0bb0-4a7f-8d89-11dc29b918b4
#=╠═╡
[scatter(y = inv.(evaluate_gridworld_reinforce_fcann(1_000, α_θ, α_w)), name = "α_θ: $α_θ, α_w: $α_w") for α_θ in [8f-2, 16f-2] for α_w in [2f-2]] |> plot
  ╠═╡ =#

# ╔═╡ 82c69936-1e15-4c23-a869-0aa47edb1796
#=╠═╡
actor_critic_with_eligibility_traces_linear(gridworld_state_mdp, 0.99f0, .95f0, .5f0, typemax(Int64), 100_000, gridworld_feature, update_gridworld_feature!; α_θ = 5f-1, α_w = 1f-2) |> eval_gridworld_returns |> plot
  ╠═╡ =#

# ╔═╡ 0b17138b-9bcc-41cd-8b9d-3a079445109c
#=╠═╡
function evaluate_gridworld_ac_linear(γ, steps, λ_θ, λ_w, α_θ, α_w; nruns = 100, kwargs...)
	f(x) = actor_critic_with_eligibility_traces_linear(gridworld_state_mdp, γ, λ_θ, λ_w, typemax(Int64), steps, copy(gridworld_feature), update_gridworld_feature!; α_θ = α_θ, α_w = α_w, kwargs...) |> v -> eval_gridworld_returns(v; total_steps = steps, kwargs...)
	1:nruns |> Map(f) |> foldxt((a, b) -> a .+ b) |> v -> v ./ nruns
end
  ╠═╡ =#

# ╔═╡ 47ddf8d8-54cf-4d4c-a429-d489804434b0
#=╠═╡
evaluate_gridworld_ac_linear(0.99f0, 100_000, 0.95f0, 0.5f0, 1f-2, 1f-2) |> plot
  ╠═╡ =#

# ╔═╡ 4c19375a-ffb9-4b49-bf42-3d9557b4e8e5
#=╠═╡
[scatter(y = evaluate_gridworld_ac_linear(0.99f0, 40_000, 0.9f0, 0.5f0, α_θ, α_w), name = "α_θ: $α_θ, α_w: $α_w") for α_θ in [5f-1, 1f-1] for α_w in [1f-2]] |> plot
  ╠═╡ =#

# ╔═╡ d570cc59-7263-46f9-99c8-8faa99e854ab
#=╠═╡
[scatter(y = evaluate_gridworld_ac_linear(0.99f0, 100_000, λ, λ, 32f-2, 8f-4), name = "λ: $λ") for λ in [0f0, 0.1f0, 0.5f0, 0.9f0, 0.99f0]] |> plot
  ╠═╡ =#

# ╔═╡ e7a6798e-9c2d-4f9d-abeb-f8c28ebe542f
#=╠═╡
const gridworld_ac = actor_critic_with_eligibility_traces_linear(gridworld_state_mdp, 0.99f0, .95f0, .5f0, typemax(Int64), 100_000, gridworld_feature, update_gridworld_feature!; α_θ = 5f-1, α_w = 1f-2)
  ╠═╡ =#

# ╔═╡ 55168857-d8d7-487e-a86b-f0bb6c7a1467
#=╠═╡
eval_gridworld_returns(gridworld_ac) |> plot
  ╠═╡ =#

# ╔═╡ 4bb49599-941e-49b9-9c3a-079ee7f789e2
#=╠═╡
eval_gridworld_final_policy(gridworld_ac.policy_sample_action)
  ╠═╡ =#

# ╔═╡ b5583c7e-f11b-4b4c-a3b7-32a8bb95e3a2
#=╠═╡
plot_gridworld_state_value_function(gridworld_ac.value_function)
  ╠═╡ =#

# ╔═╡ c3a15808-071a-433b-a26e-d0bfe879bcba
#=╠═╡
plot_gridworld_policy_function(gridworld_ac.policy_function)
  ╠═╡ =#

# ╔═╡ 3437d6ec-a373-4114-a144-29b4e642924e
#=╠═╡
const gridworld_ac2 = actor_critic_with_eligibility_traces_fcann(gridworld_state_mdp, 0.99f0, .95f0, .95f0, typemax(Int64), 100_000, copy(gridworld_feature), update_gridworld_feature!, [32, 32]; reslayers = 1, α_θ = 2f-1, α_w = 1f-3)
  ╠═╡ =#

# ╔═╡ 18bad2a5-5c90-4e83-b29a-415730b1d5bf
#=╠═╡
plot_gridworld_state_value_function(gridworld_ac2.value_function)
  ╠═╡ =#

# ╔═╡ b3afb879-c2ed-4408-b77d-fcc535934b17
#=╠═╡
plot_gridworld_policy_function(gridworld_ac2.policy_function)
  ╠═╡ =#

# ╔═╡ e366c806-9f44-4bdc-b2b0-8754d99c8342
#=╠═╡
eval_gridworld_final_policy(gridworld_ac2.policy_sample_action)
  ╠═╡ =#

# ╔═╡ aeb7304c-0f5f-4db9-b3c1-d19048fd4176
begin
	function update_batch_policy_gradient!(∇lnπ::Matrix{T}, θ::Matrix{T}, δs::Vector{T}, π_dists::Matrix{T}, batch_actions::Vector{I}, feature_vectors::Vector{V}) where {T<:Real, V<:StateAggregationFeatureVector, I<:Integer}
		K, num_actions = size(π_dists)
		π_dists .*= δs
		∇lnπ .= zero(T)
		c = one(T) / K
		@inbounds for i_a in 1:num_actions
			@simd for k in 1:K
				group_index = feature_vectors[k].group_index
				∇lnπ[group_index, i_a] -= c*π_dists[k, i_a] #note that π_dists row k is already multiplied by δs[k]
			end	
		end

		@inbounds @simd for k in 1:K
			group_index = feature_vectors[k].group_index
			∇lnπ[group_index, batch_actions[k]] += c*δs[k]
		end
		return ∇lnπ
	end

	function update_batch_policy_gradient!(∇lnπ::Matrix{T}, θ::Matrix{T}, δs::Vector{T}, π_dists::Matrix{T}, batch_actions::Vector{I}, feature_vectors::Vector{V}) where {T<:Real, V<:BinaryFeatureVector, I<:Integer}
		K, num_actions = size(π_dists)
		π_dists .*= δs
		∇lnπ .= zero(T)
		c = one(T) / K
		@inbounds for k in 1:K
			v = feature_vectors[k]
			num_features = v.num_features
			i_a = batch_actions[k]
			@simd for i in 1:num_features 
				j = v.active_features[i]
				∇lnπ[j, i_a] += c*δs[k]
			end
		end
		
		@inbounds for i_a in 1:num_actions
			for k in 1:K
				v = feature_vectors[k]
				num_features = v.num_features
				@simd for i in 1:num_features
					j = v.active_features[i]
					∇lnπ[j, i_a] -= c*π_dists[k, i_a] #note that π_dists row k is already multiplied by δs[k]
				end
			end
		end
		return ∇lnπ
	end

	function update_batch_policy_gradient!(∇lnπ::Matrix{T}, θ::Matrix{T}, δs::Vector{T}, π_dists::Matrix{T}, batch_actions::Vector{I}, X::Matrix{T}) where {T<:Real, I<:Integer}
		num_features = size(X, 1)
		K, num_actions = size(π_dists)
		π_dists .*= δs
		∇lnπ .= zero(T)
		c = one(T) / K
		for k in 1:K
			@inbounds @simd for i in 1:num_features
				∇lnπ[i, batch_actions[k]] += c*δs[k]*X[i, k]
			end
		end

		LinearAlgebra.BLAS.gemm!('N', 'N', -c, X, π_dists, one(T), ∇lnπ) #note that π_dists row k is already multiplied by δs[k]
		
		return ∇lnπ
	end

	function update_batch_policy_gradient!(∇lnπ::FCANNParams{T}, θ::FCANNParams{T}, δs::Vector{T}, π_dists::Matrix{T}, batch_actions::Vector{I}, X, hidden_layers, l2::T, tanh_grad_z::FCANNActivationsBatch{T}, activations::FCANNActivationsBatch{T}, deltas::FCANNActivationsBatch{T}, onesvec::Vector{T}, dropout::T, activation_list, scales::Vector{T}) where {T <: Real, I<:Integer}
		FCANN.nnCostFunction(θ.weights..., hidden_layers, X, batch_actions, δs, l2, ∇lnπ.weights..., tanh_grad_z, activations, deltas, onesvec, dropout; resLayers = θ.reslayers, activation_list = activation_list, loss_type = CrossEntropyLoss(), input_orientation = 'T')
		scale_fcann_params!(∇lnπ, scales) #note that this also multiplies the gradient by -1 to convert cross entropy loss to policy gradient
	end

	function update_batch_policy_gradient!(∇lnπ::FCANNParamsGPU, θ::FCANNParamsGPU, δs::Vector{T}, π_dists::Matrix{T}, batch_actions::Vector{I}, X::Matrix{T}, hidden_layers, l2::T, tanh_grad_z::FCANNActivationsGPU, activations::FCANNActivationsGPU, deltas::FCANNActivationsGPU, onesvec::FCANN.CUDAArray, dropout::T, activation_list, scales::Vector{T}, gpu_input::FCANN.CUDAArray, reindexed_output_indices::Vector{Cint},  gpu_output_indices::FCANN.CUDAArray, gpu_output_values::FCANN.CUDAArray) where {T <: Real, I<:Integer}
		reindexed_output_indices .= batch_actions .- 1 #note that the gpu uses zero indexing by default
		FCANN.memcpy!(gpu_output_indices, reindexed_output_indices)
		FCANN.memcpy!(gpu_input, X)
		FCANN.memcpy!(gpu_output_values, δs)
		FCANN.nnCostFunction(θ.weights..., size(X, 1), size(π_dists, 2), hidden_layers, length(batch_actions), onesvec, activations, tanh_grad_z, deltas, ∇lnπ.weights..., gpu_input, gpu_output_indices, gpu_output_values; lambda = l2, D = dropout, resLayers = θ.reslayers, activation_list = activation_list, input_orientation = 'T')
		scale_fcann_params!(∇lnπ, scales) #note that this also multiplies the gradient by -1 to convert cross entropy loss to policy gradient
	end
end

# ╔═╡ c0b907ca-9959-45d9-8cc1-343d71ef5dd8
begin
	function initialize_synchronous_features(x::Vector{T}, num_env::Integer) where T<:Real
		l = length(x)
		zeros(T, l, num_env)
	end

	function initialize_synchronous_features(x::AbstractBinaryFeatures{T}, num_env) where T<:Real
		[deepcopy(x) for _ in 1:num_env]
	end
end

# ╔═╡ 7f848bd1-e325-427d-8d68-8f7c2d1e7039
function BLAS.gemm!(::Char, ::Char, α::T, X::Vector{V}, v::Vector{T}, β::T, output::Array{T, N}) where {T<:Real, N, V<:AbstractBinaryFeatures}
	@assert length(X) == length(output)
	for i in eachindex(X)
		output[i] = β*output[i] + α*linear_value_function(X[i], v)
	end
end

# ╔═╡ 7335116e-670c-45ea-ae23-015b69964e5b
begin
	function accumulate_linear_gradient!(∇v̂::Vector{T}, c::T, δs::Vector{T}, feature_matrix::Matrix{T}) where {T<:Real}
		# for i in eachindex(δs)
		# 	#the feature matrix contains the feature vector of each example in a column, the i argument is the index of the example being used
		# 	@inbounds @simd for j in 1:size(feature_matrix, 1)
		# 		∇v̂[j] += c * δs[i] * feature_matrix[j, i]
		# 	end
		# end
		LinearAlgebra.BLAS.gemv!('N', c, feature_matrix, δs, zero(T), ∇v̂)
	end

	function accumulate_linear_gradient!(∇v̂::Vector{T}, c::T, δs::Vector{T}, feature_vectors::Vector{V}) where {T<:Real, V<:StateAggregationFeatureVector}
		∇v̂ .= zero(T)
		for i in eachindex(δs)
			#the i argument is the index of the example being used. only need to update term for the active features in that state
			i_s = feature_vectors[i].group_index
			∇v̂[i_s] += c*δs[i]
		end
	end

	function accumulate_linear_gradient!(∇v̂::Vector{T}, c::T, δs::Vector{T}, feature_vectors::Vector{V}) where {T<:Real, V<:BinaryFeatureVector}
		∇v̂ .= zero(T)
		for i in eachindex(δs)
			#the i argument is the index of the example being used. only need to update term for the active features in that state
			x = feature_vectors[i]
			@inbounds @simd for ind in 1:x.num_features
				j = x.active_features[ind]
				∇v̂[j] += c*δs[i]
			end
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

# ╔═╡ be468ef6-6e8f-4b9a-aa20-993102168ca6
function update_batch_value_gradient!(∇v̂::Vector{T}, δs::Vector{T}, value_params::Vector{T}, targets::Vector{T}, feature_matrix) where {T<:Real}
	#reset gradient to 0
	∇v̂ .= zero(T)

	#initialize batch size in order to calculate constant for average
	batch_size = length(targets)
	c = T(2 / batch_size)

	#get reference value using value params
	LinearAlgebra.BLAS.gemm!('T', 'N', one(T), feature_matrix, value_params, zero(T), δs)

	δs .= targets .- δs

	accumulate_linear_gradient!(∇v̂, c, δs, feature_matrix)
end

# ╔═╡ a92a325e-4e3f-4c9f-8299-33b1a54cef10
begin
	function update_batch_value_gradient!(∇v̂::FCANNParams{T}, δs::Vector{T}, value_params::FCANNParams{T}, targets::Vector{T}, feature_matrix, input_size::Integer, output::Matrix{T}, hidden_layers::Vector{Int64}, l2::T, tanh_grad_z::FCANNActivationsBatch{T}, activations::FCANNActivationsBatch{T}, deltas::FCANNActivationsBatch{T}, onesvec::Vector{T}, dropout::T, activation_list::AbstractVector{B}, scales::Vector{T}) where {T<:Float32, B<:Bool}
		output .= targets
		FCANN.nnCostFunction(value_params.weights..., input_size, hidden_layers, feature_matrix, output, l2, ∇v̂.weights..., tanh_grad_z, activations, deltas, onesvec, dropout; resLayers = value_params.reslayers, costFunc = "sqErr", activation_list = activation_list, input_orientation = get_input_orientation(feature_matrix))
		scale_fcann_params!(∇v̂, scales)
		δs .= activations[end] #note that this is just the state value output
		δs .= targets .- δs
	end

	function update_batch_value_gradient!(∇v̂::FCANNParamsGPU, δs::Vector{T}, value_params::FCANNParamsGPU, targets::Vector{T}, feature_matrix::Matrix{T}, input_size::Integer, gpu_output::FCANN.CUDAArray, cpu_output::Matrix{T}, hidden_layers::Vector{Int64}, l2::T, tanh_grad_z::FCANNActivationsGPU, activations::FCANNActivationsGPU, deltas::FCANNActivationsGPU, onesvec::FCANN.CUDAArray, dropout::T, activation_list::AbstractVector{B}, scales::Vector{T}, gpu_input::FCANN.CUDAArray) where {T<:Float32, B<:Bool}
		cpu_output .= targets
		FCANN.memcpy!(gpu_output, cpu_output)
		FCANN.memcpy!(gpu_input, feature_matrix)
		FCANN.nnCostFunction(value_params.weights..., input_size, 1, hidden_layers, length(targets), onesvec, activations, tanh_grad_z, deltas, ∇v̂.weights..., gpu_input, gpu_output, dropout; costFunc = "sqErr", activation_list = activation_list, input_orientation = 'T')
		scale_fcann_params!(∇v̂, scales)
		FCANN.memcpy!(cpu_output, activations[end]) #note that this is just the state value output
		δs .= targets .- cpu_output
	end
end

# ╔═╡ 6e337a0f-b69c-4f18-9e33-bc792d588fbe
#take a matrix of policy preferences where each row represents a different distribution and apply a softmax on a per row basis storing the result in the original matrix
function ReinforcementLearning.soft_max!(πs::AbstractMatrix{T}, row_sums::Vector{T}, row_mins::Vector{T}, row_maxes::Vector{T}) where T<:Real
	batch_size, num_actions = size(πs)
	c = one(T) / num_actions
	update_row_extrema!(row_mins, row_maxes, πs)
	
	for col in 1:num_actions
		for row in 1:batch_size
			minx = row_mins[row]
			maxx = row_maxes[row]
			if minx == maxx
				πs[row, col] = c
			else
				h = exp(πs[row, col] - maxx)
				row_sums[row] += h
				πs[row, col] = h
			end
		end
	end

	for col in 1:num_actions 
		for row in 1:batch_size
			minx = row_mins[row]
			maxx = row_maxes[row]
			if minx != maxx
				πs[row, col] /= row_sums[row]
			end
		end
	end
end

# ╔═╡ 05e4d2f6-bfc4-426e-866e-834d39666eb7
begin
	function update_batch_policy_dist!(policy_matrix::Matrix{T}, X, θ::Matrix{T}, row_sums::Vector{T}, row_mins::Vector{T}, row_maxes::Vector{T}, state_list::Vector{S}, is_valid_action::Function) where {T<:Real, S}
		LinearAlgebra.BLAS.gemm!('T', 'N', one(T), X, θ, zero(T), policy_matrix)
		mask_invalid_actions!(policy_matrix, state_list, is_valid_action)
		soft_max!(policy_matrix, row_sums, row_mins, row_maxes)
	end

	function update_batch_policy_dist!(policy_matrix::Matrix{T}, X, θ::FCANNParams{T}, row_sums::Vector{T}, row_mins::Vector{T}, row_maxes::Vector{T}, state_list::Vector{S}, is_valid_action::Function, activations::FCANNActivationsBatch{T}) where {T<:Real, S}
		FCANN.forwardNOGRAD_base!(activations, θ.weights..., X, θ.reslayers; input_orientation = 'T')
		# update_state_values!(policy_matrix, X, θ, activations)
		policy_matrix .= last(activations)
		mask_invalid_actions!(policy_matrix, state_list, is_valid_action)
		soft_max!(policy_matrix, row_sums, row_mins, row_maxes)
	end

	function update_batch_policy_dist!(policy_matrix::Matrix{T}, X::Matrix{T}, θ::FCANNParamsGPU, row_sums::Vector{T}, row_mins::Vector{T}, row_maxes::Vector{T}, state_list::Vector{S}, is_valid_action::Function, activations::FCANNActivationsGPU, gpu_input::FCANN.CUDAArray) where {T<:Real, S}
		FCANN.memcpy!(gpu_input, X)
		FCANN.forwardNOGRAD_base!(activations, θ.weights..., gpu_input, θ.reslayers; input_orientation = 'T')
		# update_state_values!(policy_matrix, X, θ, activations)
		FCANN.memcpy!(policy_matrix, last(activations))
		mask_invalid_actions!(policy_matrix, state_list, is_valid_action)
		soft_max!(policy_matrix, row_sums, row_mins, row_maxes)
	end
end

# ╔═╡ 4622a796-1ef7-40b1-87da-a1799369f758
begin
	update_batch_state_values!(state_values::Vector{T}, X, w::Vector{T}) where T<:Real = update_state_values!(state_values, X, w, nothing)
	update_batch_state_values!(state_values::Vector{T}, X, w::FCANNParams{T}, activations::FCANNActivationsBatch{T}) where T<:Real = update_state_values!(state_values, X, w, activations)

	function update_batch_state_values!(state_values::Vector{T}, X::Matrix{T}, w::FCANNParamsGPU, activations::FCANNActivationsGPU, gpu_input::FCANN.CUDAArray, output::Matrix{T}) where T<:Real
		FCANN.memcpy!(gpu_input, X)
		FCANN.forwardNOGRAD_base!(activations, w.weights..., gpu_input, w.reslayers; input_orientation = 'T')
		FCANN.memcpy!(output, last(activations))
		state_values .= output
	end
end

# ╔═╡ be6065fa-83ba-41be-9e7c-8fff2646425b
function Base.copy!(x1::Vector{V}, x2::Vector{V}) where V<:AbstractBinaryFeatures
	for i in 1:length(x1)
		update_feature_vector!(x1[i], x2[i])
	end
end

# ╔═╡ 3a4510e6-054b-40fe-989d-7ac8c86db757
function dqn!(value_params::Q, target_params::Q, mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector::V, update_feature_vector!::Function, update_action_values!::Function, update_value_gradient!::Function; target_args::Tuple = (), α = one(T)/10, ϵ = one(T) / 10, buffer_size::Integer = 10_000, batch_size::Integer = 512, target_update_interval::Integer = 100, α_decay = one(T), decay_step = typemax(Int64), save_step_rewards::Bool = false, use_double_q::Bool = false, N::Integer = 0, ∇q̂::Q = copy(value_params), kwargs...) where {Q, T<:Real, S, A, P<:AbstractStateTransition, F1<:Function, F2<:Function, F3<:Function, V}

	#initialize memory
	action_values = zeros(T, length(mdp.actions))
	policy = copy(action_values)
	replay_buffer = CircularBuffer{Tuple{V, Int64, T, V, Bool, S}}(buffer_size)
	targets = Vector{T}(undef, batch_size)
	target_const = Vector{T}(undef, batch_size)
	batch_inds = Vector{Int64}(undef, batch_size)
	feature_matrix = form_feature_matrix(mdp, feature_vector, batch_size)
	output_matrix = zeros(T, batch_size, length(mdp.actions))
	output_args = !use_double_q ? (output_matrix,) : (output_matrix, copy(output_matrix))
	param_args = !use_double_q ? (target_params,) : (target_params, value_params)
	output_inds = Vector{Int64}(undef, batch_size)
	feature_vector2 = deepcopy(feature_vector)
	state_list = Vector{S}(undef, batch_size)
	
	s = mdp.initialize_state()
	update_feature_vector!(feature_vector, s)
	update_action_values!(action_values, feature_vector, value_params; is_valid_action = i_a -> mdp.is_valid_action(s, i_a))
	policy .= action_values
	make_ϵ_greedy_policy!(policy; ϵ = ϵ, is_valid_action = i_a -> mdp.is_valid_action(s, i_a))
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
		
		push!(replay_buffer, (deepcopy(feature_vector), i_a, r, deepcopy(feature_vector2), terminated, s′))

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
		update_action_values!(action_values, feature_vector2, value_params; is_valid_action = i_a -> mdp.is_valid_action(s′, i_a))
		policy .= action_values
		make_ϵ_greedy_policy!(policy; ϵ = ϵ, is_valid_action = i_a -> mdp.is_valid_action(s′, i_a))
		i_a′ = sample_action(policy)
		#@info "action choice is $i_a′"

		decay *= (step > decay_step)*α_decay + (step <= decay_step)

		#only perform gradient parameter update once the replay buffer is large enough to fill up an entire batch
		if step ≥ (batch_size + N)
			update_batch_inds!(batch_inds, step, buffer_size, N)
			# @info "batch inds are $batch_inds"
			
			update_targets!(targets, state_list, mdp.is_valid_action, γ, replay_buffer, batch_inds, N, target_const, param_args..., feature_matrix, action_values, output_args..., target_args...)
			# @info "target values are $targets"

			#update feature matrix
			for i in eachindex(batch_inds)
				(x_k, i_a_k, _, _, _, _) = replay_buffer[batch_inds[i]]
				update_feature_matrix!(feature_matrix, x_k, i)
				output_inds[i] = i_a_k
			end
	
			update_value_gradient!(∇q̂, value_params, targets, output_inds, feature_matrix, output_matrix)
	
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

	cleanup_gradient!(∇q̂)

	q̂, form_kwargs = form_value_function(mdp, update_feature_vector!, update_action_values!, feature_vector, value_params)
	return (value_function = q̂, episode_rewards = episode_rewards, episode_steps = episode_steps, final_parameters = deepcopy(value_params), form_kwargs = form_kwargs)
end

# ╔═╡ 830ba410-377a-423b-9e75-6884c8cbbbea
dqn_linear(mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), value_params::Matrix{T} = initialize_linear_parameters(feature_vector,mdp, init_value), target_params::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, init_value), kwargs...) where T<:Real = dqn!(value_params, target_params, mdp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, update_linear_action_values!, update_linear_value_gradient!; kwargs...) 

# ╔═╡ 05fa2e5c-8633-4c43-89ba-09dd0c83cdfa
function dqn(mdp::TabularMDP, γ::T, max_episodes::Integer, max_steps::Integer; kwargs...) where T<:Real 
	state_mdp = StateMDP(mdp)
	setup = state_aggregation_feature_setup(first(mdp.states), length(mdp.states), s -> mdp.state_index[s])
	dqn_linear(state_mdp, γ, max_episodes, max_steps, setup...; kwargs...)
end

# ╔═╡ de6aca47-7d2c-4c8c-8df9-c3e16e6dc2bd
#=╠═╡
const gridworld_dqn = dqn(gridworld_mdp, 0.99f0, typemax(Int64), 40_000; α = 3f-4, ϵ = 0.01f0, buffer_size = 10_000, batch_size = 512, target_update_interval = 100)
  ╠═╡ =#

# ╔═╡ d1058fb3-cb03-4031-b089-ce5f077036a0
#=╠═╡
eval_gridworld_final_policy(s -> gridworld_dqn.value_function(s).maximizing_action)
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

# ╔═╡ bbc470e8-0e6b-41a6-a547-22c339a87d63
#=╠═╡
plot([scatter(y = evaluate_gridworld_dqn(0.99f0, 20_000, α, 0.01f0, 1000, 128, 100; nruns = 100, interval = 100), name = "learning rate: $α") for α in [1f-3, 2f-3, 4f-3, 8f-3]])
  ╠═╡ =#

# ╔═╡ 26146b44-e9ee-4da4-8431-862b8b17f68a
#=╠═╡
plot([scatter(y = evaluate_gridworld_dqn(0.99f0, 20_000, 1f-3, 0.01f0, 1000, 128, 100; nruns = 100, interval = 100, N = N), name = "N: $N") for N in [0, 1, 2, 4]])
  ╠═╡ =#

# ╔═╡ 51a27e0e-14e5-4822-a255-bd63aac2a00e
#=╠═╡
function evaluate_gridworld_dqn_linear(γ, steps, α, ϵ, buffer_size, batch_size, update_interval; nruns = 100, kwargs...)
	f(x) = dqn_linear(gridworld_state_mdp, γ, typemax(Int64), steps, copy(gridworld_feature), update_gridworld_feature!; α = α, ϵ = ϵ, buffer_size = buffer_size, batch_size = batch_size, target_update_interval = update_interval, kwargs...) |> v -> eval_gridworld_returns(v; total_steps = steps, kwargs...)
	1:nruns |> Map(f) |> foldxt((a, b) -> a .+ b) |> v -> v ./ nruns
end
  ╠═╡ =#

# ╔═╡ 305e3122-76f2-4f90-9120-db20d2e7255a
#=╠═╡
plot([scatter(y = evaluate_gridworld_dqn_linear(0.99f0, 100_000, 5f-3, 0.01f0, buffer_size, 512, 100; nruns = 100, interval = 100), name = "buffer size: $buffer_size") for buffer_size in [1000, 2000]])
  ╠═╡ =#

# ╔═╡ 3eb1e4a2-bd61-4364-baa8-b9ef0ae68418
#=╠═╡
plot([scatter(y = evaluate_gridworld_dqn_linear(0.99f0, 100_000, α, 0.01f0, 1000, 512, 100; nruns = 100, interval = 100), name = "learning rate: $α") for α in [4f-3, 8f-3, 16f-3]])
  ╠═╡ =#

# ╔═╡ 4dad43fe-11e9-4949-82c1-c87503c2162a
#=╠═╡
plot([scatter(y = evaluate_gridworld_dqn_linear(0.99f0, 100_000, α, 0.01f0, 1000, 512, 100; nruns = 100, interval = 100, use_double_q = true), name = "learning rate: $α") for α in [4f-3, 8f-3, 2f-2]])
  ╠═╡ =#

# ╔═╡ 462af35e-dcbc-4f01-b0de-2e9193890b7c
#=╠═╡
plot([scatter(y = evaluate_gridworld_dqn_linear(0.99f0, 100_000, 8f-3, 0.01f0, 1000, 512, 100; nruns = 100, interval = 100, use_double_q = true, N = N), name = "N: $N") for N in [0, 1, 2, 4]])
  ╠═╡ =#

# ╔═╡ cbcb5270-2661-4a30-9787-6aa0d8915ae1
#=╠═╡
plot([scatter(y = evaluate_gridworld_dqn_linear(0.99f0, 100_000, 8f-3, 0.01f0, 1000, 128, 100; nruns = 100, interval = 100, use_double_q = false, N = N), name = "N: $N") for N in [0, 1, 2, 4, 8, 16]])
  ╠═╡ =#

# ╔═╡ 6d87928f-0ed3-410e-bbe3-e2e338919806
#=╠═╡
plot([scatter(y = evaluate_gridworld_dqn_linear(0.99f0, 100_000, α, 0.01f0, 1000, 128, 100; nruns = 100, interval = 100, use_double_q = false, N = 1), name = "Learning Rate: $α") for α in [2f-2, 1f-2, 8f-3]])
  ╠═╡ =#

# ╔═╡ 31eb8e01-380b-4553-90dc-22ffaea7aaac
#=╠═╡
const gridworld_dqn2 = dqn_linear(gridworld_state_mdp, 0.99f0, typemax(Int64), 100_000, gridworld_feature, update_gridworld_feature!; α = 1f-2, ϵ = 0.05f0, buffer_size = 1_000, batch_size = 128, target_update_interval = 100, N = 1)
  ╠═╡ =#

# ╔═╡ 4f88dd4e-4f18-4770-a6eb-1ad1094b92c5
#=╠═╡
eval_gridworld_final_policy(s -> gridworld_dqn2.value_function(s).maximizing_action)
  ╠═╡ =#

# ╔═╡ 70093ded-4ebc-48aa-bcfd-48d492aa1c5e
#=╠═╡
plot_gridworld_value_function(gridworld_dqn2.value_function)
  ╠═╡ =#

# ╔═╡ d9437c04-d197-4b8c-b1ad-0029b3d77144
function dqn_fcann(mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; batch_size::Integer = 512, reslayers::Int64 = 0, use_μP::Bool = true, value_params::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, length(mdp.actions), reslayers, use_μP), target_params::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, length(mdp.actions), reslayers, use_μP), dropout = zero(T), activation_list = fill(true, length(hidden_layers)), l2 = zero(T), use_gpu::Bool = false, kwargs...) where T<:Real 
	setup = setup_fcann_action_value_arguments(value_params, target_params, batch_size, l2, dropout, use_μP, activation_list; use_gpu = use_gpu)
	
	!use_gpu && return dqn!(value_params, target_params, mdp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, setup.update_action_values!, setup.update_value_gradient!; target_args = setup.target_args, batch_size = batch_size, kwargs...)
	
	isempty(setup.gpu_args) && error("GPU backend is not available")
	
	output = dqn!(setup.gpu_args.value_params, setup.gpu_args.target_params, mdp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, setup.update_action_values!, setup.update_value_gradient!; target_args = setup.gpu_args.target_args, batch_size = batch_size, kwargs...)
	FCANN.GPU2Host(value_params.weights, setup.gpu_args.value_params.weights)
	FCANN.GPU2Host(target_params.weights, setup.gpu_args.target_params.weights)
	setup.gpu_args.cleanup_vars()
	return (;output..., final_parameters = deepcopy(value_params)) #note that dqn! will copy the gpu params which have been cleaned up so we need to replace this output in the named tuple with the parameters we transfered back to the host before cleaning up
end

# ╔═╡ f0582c1f-7f6d-4f38-9051-fb5ef158612f
#=╠═╡
function evaluate_gridworld_dqn_fcann(hidden_layers, γ, steps, α, ϵ, buffer_size, batch_size, update_interval; nruns = 100, kwargs...)
	f(x) = dqn_fcann(gridworld_state_mdp, γ, typemax(Int64), steps, copy(gridworld_feature), update_gridworld_feature!, hidden_layers; α = α, ϵ = ϵ, buffer_size = buffer_size, batch_size = batch_size, target_update_interval = update_interval, kwargs...) |> v -> eval_gridworld_returns(v; total_steps = steps, kwargs...)
	1:nruns |> Map(f) |> foldxt((a, b) -> a .+ b) |> v -> v ./ nruns
end
  ╠═╡ =#

# ╔═╡ 383c9892-b603-4d04-8af6-6f7f1308b1d7
#=╠═╡
plot([scatter(y = evaluate_gridworld_dqn_fcann([32, 32], 0.99f0, 100_000, 4f-1, 0.05f0, 1_000, batch_size, 100; nruns = 40, interval = 100), name = "batch size: $batch_size") for batch_size in [2, 4, 8, 16, 32, 64, 128, 256, 512]])
  ╠═╡ =#

# ╔═╡ 1c2765ce-72b8-4e51-bab1-f646536a5979
#=╠═╡
plot([scatter(y = evaluate_gridworld_dqn_fcann([32, 32], 0.99f0, 100_000, 8f-2, 0.05f0, 1_000, 64, 100; nruns = 40, interval = 100, use_double_q = false, N = N), name = "N: $N") for N in [0, 1, 2, 4, 8, 16]])
  ╠═╡ =#

# ╔═╡ ad924e38-1ced-42e8-8824-51c75506229e
#=╠═╡
plot([scatter(y = evaluate_gridworld_dqn_fcann([32, 32], 0.99f0, 100_000, α, 0.05f0, 1_000, 64, 100; nruns = 40, interval = 100, use_double_q = false, N = 16), name = "α: $α") for α in [1f-2, 4f-2, 8f-2]])
  ╠═╡ =#

# ╔═╡ b3c6aad2-027c-4bb2-93a2-e3175c7ee66a
#=╠═╡
const gridworld_dqn3 = dqn_fcann(gridworld_state_mdp, 0.99f0, typemax(Int64), 100_000, gridworld_feature, update_gridworld_feature!, [32, 32]; reslayers = 1, α = 1f-3, ϵ = 0.05f0, buffer_size = 1000, batch_size = 128, target_update_interval = 100, N = 8)
  ╠═╡ =#

# ╔═╡ dac1ec53-d528-4f78-8306-294b0725a183
#=╠═╡
plot_gridworld_value_function(gridworld_dqn3.value_function)
  ╠═╡ =#

# ╔═╡ ffea4162-4cc7-4ee0-b962-d2d8aa5660c3
#=╠═╡
eval_gridworld_final_policy(s -> gridworld_dqn3.value_function(s).maximizing_action)
  ╠═╡ =#

# ╔═╡ 3ec9a9b2-4bd2-4228-94f5-d2f80f5831ea
#=╠═╡
dqn_fcann(gridworld_state_mdp, 0.99f0, typemax(Int64), 100_000, gridworld_feature, update_gridworld_feature!, [32, 32]; reslayers = 1, α = 1f-3, ϵ = 0.05f0, buffer_size = 1000, batch_size = 128, target_update_interval = 100, N = 8)
  ╠═╡ =#

# ╔═╡ f8cb49ae-10b5-4a94-a5ab-24c6b70d597c
#=╠═╡
@plutoprofview dqn_fcann(gridworld_state_mdp, 0.99f0, typemax(Int64), 1_000, gridworld_feature, update_gridworld_feature!, [256, 256]; reslayers = 1, α = 1f-20, ϵ = 0.05f0, buffer_size = 1_000, batch_size = 64, target_update_interval = 100, N = 2, use_gpu = true)
  ╠═╡ =#

# ╔═╡ 109954db-b333-4899-91e1-18f067742372
#=╠═╡
@plutoprofview dqn_fcann(gridworld_state_mdp, 0.99f0, typemax(Int64), 1_000, gridworld_feature, update_gridworld_feature!, [1024, 1024]; reslayers = 1, α = 1f-3, ϵ = 0.05f0, buffer_size = 1_000, batch_size = 64, target_update_interval = 100, N = 8, use_gpu = false)
  ╠═╡ =#

# ╔═╡ c7994abc-e49d-4ccb-b09b-e68c14bb7d6f
function synchronous_actor_critic!(policy_params::PP, value_params::VP, mdp::StateMDP{T, S, A, PTF, F1, F2, F3}, γ::T, max_steps::Integer, num_env::Integer, feature_vector, update_feature_vector!::Function, value_args::Tuple, value_gradient_args::Tuple, policy_args::Tuple, policy_gradient_args::Tuple; α_w::T = one(T)/10, α_θ::T = one(T)/10, ∇v̂::VP = deepcopy(value_params), ∇lnπ::PP = deepcopy(policy_params)) where {T<:Real, S, A, PTF, F1, F2, F3, VP, PP}
	episode_steps = Vector{Int64}()
	episode_rewards = Vector{T}()
	avg_step_rewards = zeros(T, max_steps)

	#initialize variables
	batch_episodes = ones(Int64, num_env)
	batch_episode_steps = [Vector{Int64}() for _ in 1:num_env]
	batch_episode_rewards = [Vector{T}() for _ in 1:num_env]
	rtots = zeros(T, num_env)
	batch_rewards = zeros(T, num_env)
	cs = ones(T, num_env)
	batch_states = [mdp.initialize_state() for _ in 1:num_env]
	batch_term_check = [false for _ in 1:num_env]
	feature_vectors = initialize_synchronous_features(feature_vector, num_env)
	feature_vectors2 = initialize_synchronous_features(feature_vector, num_env)
	policy_matrix = zeros(T, num_env, length(mdp.actions))
	batch_actions = ones(Int64, num_env)
	batch_state_values = zeros(T, num_env)
	batch_targets = zeros(T, num_env)
	δs = zeros(T, num_env)
	row_sums = zeros(T, num_env)
	row_mins = zeros(T, num_env)
	row_maxes = zeros(T, num_env)
	state_list = Vector{S}(undef, num_env)

	for (i, s) in enumerate(batch_states)
		update_feature_vector!(feature_vector, s)
		update_feature_matrix!(feature_vectors, feature_vector, i)
		state_list[i] = s
	end
	
	for step in 1:max_steps
		#for each environment update the policy distribution on a per row basis and then sample an action from each environment
		update_batch_policy_dist!(policy_matrix, feature_vectors, policy_params, row_sums, row_mins, row_maxes, state_list, mdp.is_valid_action, policy_args...)
		sample_batch_actions!(batch_actions, policy_matrix)

		# @info "Current batch states: $batch_states"
		# @info "Using a policy matrix of $policy_matrix sampled the following actions: $batch_actions"

		r_avg = zero(T) 
		#perform transitions for entire batch
		for k in 1:num_env
			(r, s′) = mdp.ptf(batch_states[k], batch_actions[k])
			terminal = mdp.isterm(s′)
			batch_term_check[k] = terminal
			rtots[k] += r
			r_avg += r
			if terminal
				s′ = mdp.initialize_state()
				update_feature_vector!(feature_vector, s′)
				batch_episodes[k] += 1
				push!(batch_episode_steps[k], step)
				push!(batch_episode_rewards[k], rtots[k])
				rtots[k] = zero(T)
				cs[k] = one(T)
			else
				update_feature_vector!(feature_vector, s′)
				cs[k] *= γ
			end
			batch_states[k] = s′
			update_feature_matrix!(feature_vectors2, feature_vector, k)
			state_list[k] = s′
			batch_rewards[k] = r
		end

		avg_step_rewards[step] = r_avg / num_env

		#calculate state values for transition states
		update_batch_state_values!(batch_state_values, feature_vectors2, value_params, value_args...)

		#zero out prediction values for terminal states and add discounted value to reward
		batch_targets .= batch_rewards .+ γ .* batch_state_values .* .!batch_term_check

		#updates value gradient with the loss function and updates δs with the states values minus the target values for use later in the policy gradient calculation
		update_batch_value_gradient!(∇v̂, δs, value_params, batch_targets, feature_vectors, value_gradient_args...)	

		#updates batch advantage values to use in policy gradient by multiplying by γ^n where n is the number of steps since the episode started
		δs .*= cs

		#decay c by γ but reset if that environment reached a terminal state
		cs .= .!batch_term_check .* cs .*γ .+ batch_term_check
		
		#update value parameters using the value gradient
		update_params_with_gradient!(value_params, α_w, ∇v̂)

		# @info "Updating policy_params with the following information: δs = $δs, policy_matrix = $policy_matrix"
		#update policy parameters using the policy distribution, batch actions, and advantage values
		update_batch_policy_gradient!(∇lnπ, policy_params, δs, policy_matrix, batch_actions, feature_vectors, policy_gradient_args...)
		
		update_params_with_gradient!(policy_params, α_θ, ∇lnπ)

		#prepare for next step
		copy!(feature_vectors, feature_vectors2)
		# feature_vectors .= feature_vectors2
		# for k in 1:num_env
		# 	batch_states[k] = batch_transition_states[k]
		# end
	end

	#note that this step is a noop unless the gradients are gpu objects in which case they get deallocated
	cleanup_gradient!(∇v̂)
	cleanup_gradient!(∇lnπ)

	policy_and_value_components = form_policy_and_value_function(mdp, feature_vector, update_feature_vector!, policy_params, value_params)

	return (;avg_step_rewards = avg_step_rewards, batch_episodes = batch_episodes, batch_episode_steps = batch_episode_steps, batch_episode_rewards = batch_episode_rewards, policy_parameters = deepcopy(policy_params), value_parameters = deepcopy(value_params), policy_and_value_components...)
end

# ╔═╡ 738241ce-0315-41d8-a67f-64c91479fe57
synchronous_actor_critic_linear(mdp::StateMDP{T, S, A, PTF, F1, F2, F3}, γ::T, max_steps::Integer, num_env::Integer, feature_vector, update_feature_vector!::Function; policy_params::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, zero(T)), value_params::Vector{T} = initialize_linear_parameters(feature_vector, zero(T)), kwargs...) where {T<:Real, S, A, PTF, F1, F2, F3} = synchronous_actor_critic!(policy_params, value_params, mdp, γ, max_steps, num_env, feature_vector, update_feature_vector!, (), (), (), (); kwargs...)

# ╔═╡ 85a24ea7-411e-4d11-a83b-bddad52772df
#=╠═╡
const gridworld_sync_ac = synchronous_actor_critic_linear(gridworld_state_mdp, 0.99f0, 100_000, 1, gridworld_feature, update_gridworld_feature!; α_θ = 64f-2, α_w = 32f-2)
  ╠═╡ =#

# ╔═╡ 0d045fdd-67e4-45d4-957e-c58b266dfe5e
#=╠═╡
plot(cumsum(gridworld_sync_ac.avg_step_rewards) ./ (1:length(gridworld_sync_ac.avg_step_rewards)))
  ╠═╡ =#

# ╔═╡ c106a277-cf6a-4a19-b17f-6370c42c43c8
#=╠═╡
eval_gridworld_final_policy(gridworld_sync_ac.policy_sample_action)
  ╠═╡ =#

# ╔═╡ 39ce5032-3a13-49c9-ab7d-d74d1aff883d
#=╠═╡
plot_gridworld_state_value_function(gridworld_sync_ac.value_function)
  ╠═╡ =#

# ╔═╡ 2fa46312-ec98-4680-be6e-a5ddd08baca2
function setup_fcann_batch_policy_arguments(params::FCANNParams{T}, batch_size::Integer, l2::T, dropout::T, use_μP::Bool, activation_list; use_gpu = false) where {T<:Real}
	input_length, hidden_layers, num_hidden = get_network_dimensions(params)
	
	#form activations for network
	activations = FCANN.form_activations(params.weights[1], batch_size)
	tanh_grad_z = deepcopy(activations)
	deltas = deepcopy(activations)
	onesvec = ones(T, batch_size)
	output = zeros(T, batch_size)

	#note that the policy gradient will be multiplied by -1 to acheive maximization instead of minimization
	scales = fill(-one(T), length(params.weights[1]))
	if use_μP
		for i in eachindex(hidden_layers)
			i′ = i + 1
			scales[i′] /= size(params.weights[1][i′], 2)
		end
	end

	policy_args = (activations,)
	policy_gradient_args = (hidden_layers, l2, tanh_grad_z, activations, deltas, onesvec, dropout, activation_list, scales)

	!use_gpu && return (policy_args = policy_args, policy_gradient_args = policy_gradient_args)

	if in(:GPU, backendList)
		d_activations = FCANN.device_allocate(activations)
		d_tanh_grad_z = FCANN.device_allocate(tanh_grad_z)
		d_deltas = FCANN.device_allocate(deltas)
		d_params = initialize_gpu_params(params)
		d_x = FCANN.cuda_allocate(zeros(T, input_length, batch_size))
		d_onesvec = FCANN.cuda_allocate(onesvec)
		gpu_output_inds = FCANN.cuda_allocate(zeros(Cint, batch_size))
		gpu_output_vals = FCANN.cuda_allocate(output)
		reindexed_output = zeros(Cint, batch_size)

		function cleanup_vars()
			FCANN.clear_gpu_data(d_deltas)
			FCANN.clear_gpu_data(d_tanh_grad_z)
			FCANN.clear_gpu_data([d_x])
			FCANN.clear_gpu_data(d_activations)
			FCANN.clear_gpu_data(d_params.weights[1])
			FCANN.clear_gpu_data(d_params.weights[2])
			FCANN.clear_gpu_data([d_onesvec])
			FCANN.clear_gpu_data([gpu_output_inds])
			FCANN.clear_gpu_data([gpu_output_vals])
		end

		gpu_policy_args = (d_activations, d_x)
		gpu_policy_gradient_args = (hidden_layers, l2, d_tanh_grad_z, d_activations, d_deltas, d_onesvec, dropout, activation_list, scales, d_x, reindexed_output, gpu_output_inds, gpu_output_vals)
		gpu_args = (params = d_params, policy_args = gpu_policy_args, policy_gradient_args = gpu_policy_gradient_args, cleanup_vars = cleanup_vars)
	else
		gpu_args = ()
	end

	return (policy_args = policy_args, policy_gradient_args = policy_gradient_args, gpu_args = gpu_args)
end

# ╔═╡ d5364951-0bfd-424a-8e8c-fecc05ce124b
function setup_fcann_batch_value_arguments(policy_setup::NamedTuple, params::FCANNParams{T}, batch_size::Integer, l2::T, dropout::T, use_μP::Bool, activation_list; use_gpu = false) where {T<:Real}
	input_length, hidden_layers, num_hidden = get_network_dimensions(params)
	
	#form activations for network
	activations = FCANN.form_activations(params.weights[1], batch_size)
	tanh_grad_z = deepcopy(activations)
	deltas = deepcopy(activations)
	onesvec = ones(T, batch_size)
	output = zeros(T, batch_size, 1)

	scales = fill(-one(T), length(params.weights[1]))
	if use_μP
		for i in eachindex(hidden_layers)
			i′ = i + 1
			scales[i′] /= size(params.weights[1][i′], 2)
		end
	end

	value_args = (activations,)
	value_gradient_args = (input_length, output, hidden_layers, l2, tanh_grad_z, activations, deltas, onesvec, dropout, activation_list, scales)

	!use_gpu && return (value_args = value_args, value_gradient_args = value_gradient_args)

	if in(:GPU, backendList)
		d_activations = FCANN.device_allocate(activations)
		d_tanh_grad_z = FCANN.device_allocate(tanh_grad_z)
		d_deltas = FCANN.device_allocate(deltas)
		d_onesvec = FCANN.cuda_allocate(onesvec)
		d_params = initialize_fcann_value_params(policy_setup.gpu_args.params, use_μP) #we need the policy setup here to share the layers between the policy and value params except the output layer
		d_x = FCANN.cuda_allocate(zeros(T, input_length, batch_size))
		d_output = FCANN.cuda_allocate(output)
		function cleanup_vars()
			FCANN.clear_gpu_data(d_deltas)
			FCANN.clear_gpu_data(d_tanh_grad_z)
			FCANN.clear_gpu_data([d_x])
			FCANN.clear_gpu_data(d_activations)
			FCANN.clear_gpu_data(d_params.weights[1])
			FCANN.clear_gpu_data(d_params.weights[2])
			FCANN.clear_gpu_data([d_output])
			FCANN.clear_gpu_data([d_onesvec])
		end

		gpu_value_args = (d_activations, d_x, output)
		gpu_gradient_args = (input_length, d_output, output, hidden_layers, l2, d_tanh_grad_z, d_activations, d_deltas, d_onesvec, dropout, activation_list, scales, d_x)

		gpu_args = (params = d_params, value_args = gpu_value_args, value_gradient_args = gpu_gradient_args, cleanup_vars = cleanup_vars)
	else
		gpu_args = ()
	end

	return (value_args = value_args, value_gradient_args = value_gradient_args, gpu_args = gpu_args)
end

# ╔═╡ b65512a7-1a98-4d99-835f-eca70ced2404
function synchronous_actor_critic_fcann(mdp::StateMDP{T, S, A, PTF, F1, F2, F3}, γ::T, max_steps::Integer, num_env::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; reslayers = 0, use_μP::Bool = true, policy_params::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, length(mdp.actions), reslayers, use_μP), value_params::FCANNParams = initialize_fcann_value_params(policy_params, use_μP), l2::T = zero(T), dropout::T = zero(T), activation_list::Vector{Bool} = fill(true, length(hidden_layers)), use_gpu::Bool = false, kwargs...) where {T<:Real, S, A, PTF, F1, F2, F3}

	policy_setup = setup_fcann_batch_policy_arguments(policy_params, num_env, l2, dropout, use_μP, activation_list)
	value_setup = setup_fcann_batch_value_arguments(policy_setup, value_params, num_env, l2, dropout, use_μP, activation_list)
	

	!use_gpu && return synchronous_actor_critic!(policy_params, value_params, mdp, γ, max_steps, num_env, feature_vector, update_feature_vector!, value_setup..., policy_setup...; kwargs...)

	isempty(value_setup.gpu_args) && error("GPU backend is not available")
	isempty(policy_setup.gpu_args) && error("GPU backend is not available")

	output = synchronous_actor_critic!(policy_setup.gpu_args.params, value_setup.gpu_args.params, mdp, γ, max_steps, num_env, feature_vector, update_feature_vector!, value_setup.gpu_args.value_args, value_setup.gpu_args.value_gradient_args, policy_setup.gpu_args.policy_args, policy_setup.gpu_args.policy_gradient_args; kwargs...)


	FCANN.GPU2Host(value_params.weights, value_setup.gpu_args.params.weights)
	FCANN.GPU2Host(policy_params.weights, policy_setup.gpu_args.params.weights)

	value_setup.gpu_args.cleanup_vars()
	policy_setup.gpu_args.cleanup_vars()
	return (;output..., policy_parameters = deepcopy(policy_params), value_parameters = deepcopy(value_params))	#note that synchronous_actor_critic! will copy the gpu params which have been cleaned up so we need to replace this output in the named tuple with the parameters we transfered back to the host before cleaning up
end

# ╔═╡ 89317d85-e18d-4940-9f1b-bf4f9fbf9880
#=╠═╡
const gridworld_sync_ac2 = synchronous_actor_critic_fcann(gridworld_state_mdp, 0.99f0, 1_000_000, 4, gridworld_feature, update_gridworld_feature!, [32, 32]; reslayers = 1, α_θ = 64f-2, α_w = 64f-2)
  ╠═╡ =#

# ╔═╡ f411dc93-77df-49c6-a222-5985e8aea544
#=╠═╡
plot(cumsum(gridworld_sync_ac2.avg_step_rewards) ./ (1:length(gridworld_sync_ac2.avg_step_rewards)))
  ╠═╡ =#

# ╔═╡ ae0a282c-76ba-49a6-8a2b-02012c8dcda9
#=╠═╡
plot_gridworld_state_value_function(gridworld_sync_ac2.value_function)
  ╠═╡ =#

# ╔═╡ 04bc310f-bbaf-4d0f-9d5b-6c79b47a74c7
#=╠═╡
plot_gridworld_policy_function(gridworld_sync_ac2.policy_function)
  ╠═╡ =#

# ╔═╡ 1d5be8fa-e03e-4906-b772-25a1899275a6
#=╠═╡
eval_gridworld_final_policy(gridworld_sync_ac2.policy_sample_action)
  ╠═╡ =#

# ╔═╡ 65ab73ac-0073-47dd-bf68-d565e6028d9a
#=╠═╡
const gridworld_ac_td = one_step_actor_critic_linear(gridworld_state_mdp, 0.99f0, typemax(Int64), 100_000, gridworld_feature, update_gridworld_feature!; α_θ = 1f-3, α_w = 1f-3)
  ╠═╡ =#

# ╔═╡ b4f95b92-127f-4ab4-92e8-c405fbf91e67
#=╠═╡
eval_gridworld_final_policy(gridworld_ac_td.policy_sample_action)
  ╠═╡ =#

# ╔═╡ 4a206490-b150-4004-93b3-04d187d04619
#=╠═╡
const gridworld_ac_td2 = one_step_actor_critic_fcann(gridworld_state_mdp, 0.99f0, typemax(Int64), 100_000, gridworld_feature, update_gridworld_feature!, [32, 32]; reslayers = 1, α_θ = 64f-2, α_w = 32f-2)
  ╠═╡ =#

# ╔═╡ 1a462642-08be-452a-8157-e6ac83640db9
#=╠═╡
plot_gridworld_policy_function(gridworld_ac_td2.policy_function)
  ╠═╡ =#

# ╔═╡ eebbc332-1020-4279-b82c-ccda4b3fe0cc
#=╠═╡
eval_gridworld_final_policy(gridworld_ac_td2.policy_sample_action)
  ╠═╡ =#

# ╔═╡ 44c9104e-8586-4202-8edd-eaea0073842a
function synchronous_nstep_actor_critic!(policy_params::PP, value_params::VP, mdp::StateMDP{T, S, A, PTF, F1, F2, F3}, γ::T, max_steps::Integer, num_env::Integer, feature_vector, update_feature_vector!::Function, value_args::Tuple, value_gradient_args::Tuple, policy_args::Tuple, policy_gradient_args::Tuple; α_w::T = one(T)/10, α_θ::T = one(T)/10, N::Integer = 0, ∇v̂::VP = deepcopy(value_params), ∇lnπ::PP = deepcopy(policy_params)) where {T<:Real, S, A, PTF, F1, F2, F3, VP, PP}

	iszero(N) && return synchronous_actor_critic!(policy_params, value_params, mdp, γ, max_steps, num_env, feature_vector, update_feature_vector!, value_args, value_gradient_args, policy_args, policy_gradient_args; α_w = α_w, α_θ = α_θ, ∇v̂ = ∇v̂, ∇lnπ = ∇lnπ)
	
	episode_steps = Vector{Int64}()
	episode_rewards = Vector{T}()
	avg_step_rewards = Vector{T}()

	#initialize variables
	batch_episodes = ones(Int64, num_env)
	batch_episode_steps = [Vector{Int64}() for _ in 1:num_env]
	batch_episode_rewards = [Vector{T}() for _ in 1:num_env]
	rtots = zeros(T, num_env)
	cs = ones(T, num_env)
	batch_states = [mdp.initialize_state() for _ in 1:num_env]
	current_feature_vectors = initialize_synchronous_features(feature_vector, num_env) #should store the feature vectors of the current time state for that environment
	state_list = Vector{S}(undef, num_env)
	update_feature_vectors = initialize_synchronous_features(feature_vector, num_env) #should store the feature vectors of the state being updated
	policy_matrix = zeros(T, num_env, length(mdp.actions))
	batch_actions = ones(Int64, num_env)
	batch_state_values = zeros(T, num_env)
	batch_targets = zeros(T, num_env)
	δs = zeros(T, num_env)
	row_sums = zeros(T, num_env)
	row_mins = zeros(T, num_env)
	row_maxes = zeros(T, num_env)

	batch_nstep_rewards = [CircularBuffer{T}(N+1) for _ in 1:num_env]
	batch_nstep_states = [CircularBuffer{S}(N+1) for _ in 1:num_env]
	batch_nstep_actions = [CircularBuffer{Int64}(N+1) for _ in 1:num_env]
	batch_bootstrap_discount = ones(T, num_env)
	batch_ready = fill(false, num_env) #tracks for each environment if it is ready for a batch update.  initially this will not be true for any before not enough n-step data has been accumulated yet
	batch_terminal_check = fill(false, num_env) #tracks for each environment if the current episode has terminated or not
	update_actions = fill(0, num_env)

	for (i, s) in enumerate(batch_states)
		update_feature_vector!(feature_vector, s)
		update_feature_matrix!(current_feature_vectors, feature_vector, i)
		state_list[i] = s
	end

	num_updates = 0
	batch_steps = fill(0, num_env)
	
	while num_updates < max_steps
		# @info "Current batch states: $batch_states"
		# @info "Using a policy matrix of $policy_matrix sampled the following actions: $batch_actions"

		#for each environment update the policy distribution on a per row basis and then sample an action from each environment
		if !all(batch_ready) && !all(batch_terminal_check) #only envs that are NOT ready perform a step update so if all are ready we can just proceed straight to gradient updates
			update_batch_policy_dist!(policy_matrix, current_feature_vectors, policy_params, row_sums, row_mins, row_maxes, state_list, mdp.is_valid_action, policy_args...)
			sample_batch_actions!(batch_actions, policy_matrix)
		end
		
		r_avg = zero(T) 
		#perform transitions for entire batch
		for k in 1:num_env
			if !batch_ready[k] #only update if the batch is not ready from the previous step
				if !batch_terminal_check[k] #only take a new step if the environment hasn't terminated yet
					(r, s′) = mdp.ptf(batch_states[k], batch_actions[k])
					batch_steps[k] += 1
					push!(batch_nstep_actions[k], batch_actions[k])
					push!(batch_nstep_states[k], batch_states[k])
					push!(batch_nstep_rewards[k], r)
					batch_states[k] = s′
					terminal = mdp.isterm(s′)
					rtots[k] += r
					r_avg += r
					batch_terminal_check[k] = terminal
					
					if !terminal
						update_feature_vector!(feature_vector, s′)
						update_feature_matrix!(current_feature_vectors, feature_vector, k)
						state_list[k] = s′
						batch_bootstrap_discount[k] = γ^(length(batch_nstep_rewards[k]))
					else
						batch_ready[k] = true
						batch_bootstrap_discount[k] = zero(T)
					end
						
					#if the current buffer is full or the current state is terminal then this environment is ready for a batch update
					if (length(batch_nstep_rewards[k]) == N + 1) || terminal
						batch_ready[k] = true
					end
				elseif length(batch_nstep_rewards[k]) > 1 # the environment is needed for the next gradient update, but it has already terminated, so we need to remove items from the buffer and update the feature vector for the gradient update 
					popfirst!(batch_nstep_rewards[k])
					popfirst!(batch_nstep_states[k])
					popfirst!(batch_nstep_actions[k])
					batch_ready[k] = true
					batch_bootstrap_discount[k] = zero(T)
				else #the environment is needed for the next gradient update but the buffers are empty so we need to initialize a new episode
					popfirst!(batch_nstep_rewards[k])
					popfirst!(batch_nstep_states[k])
					popfirst!(batch_nstep_actions[k])
					s′ = mdp.initialize_state()
					update_feature_vector!(feature_vector, s′)
					update_feature_matrix!(current_feature_vectors, feature_vector, k)
					state_list[k] = s′
					batch_episodes[k] += 1
					push!(batch_episode_steps[k], batch_steps[k])
					push!(batch_episode_rewards[k], rtots[k])
					rtots[k] = zero(T)
					cs[k] = one(T)
					batch_bootstrap_discount[k] = zero(T)
					batch_ready[k] = false
					batch_terminal_check[k] = false
					batch_states[k] = s′
				end
			end
		end

		#only update targets and gradient when the entire batch is ready
		if all(batch_ready)
			r_avg = zero(T)
			for k in 1:num_env
				update_feature_vector!(feature_vector, first(batch_nstep_states[k]))
				update_feature_matrix!(update_feature_vectors, feature_vector, k)
				update_actions[k] = first(batch_nstep_actions[k])
				batch_targets[k] = sum(batch_nstep_rewards[k][t]*γ^(t-1) for t in eachindex(batch_nstep_rewards[k]); init = zero(T)) #update batch_targets with discounted reward sum for up to the previous N+1 rewards
				r_avg += first(batch_nstep_rewards[k])
			end
			push!(avg_step_rewards, r_avg / num_env)
			
			#calculate state values for current states
			update_batch_state_values!(batch_state_values, current_feature_vectors, value_params, value_args...)
	
			#zero out prediction values for terminal states and add discounted value to reward
			batch_targets .+= batch_bootstrap_discount .* batch_state_values
	
			#updates value gradient with the loss function and updates δs with the states values minus the target values for use later in the policy gradient calculation
			update_batch_value_gradient!(∇v̂, δs, value_params, batch_targets, update_feature_vectors, value_gradient_args...)	
	
			#updates batch advantage values to use in policy gradient by multiplying by γ^n where n is the number of steps since the episode started
			δs .*= cs
			
			#update value parameters using the value gradient
			update_params_with_gradient!(value_params, α_w, ∇v̂)
	
			# @info "Updating policy_params with the following information: δs = $δs, policy_matrix = $policy_matrix"
			#update policy parameters using the policy distribution, batch actions, and advantage values
			update_batch_policy_gradient!(∇lnπ, policy_params, δs, policy_matrix, update_actions, update_feature_vectors, policy_gradient_args...)
			
			update_params_with_gradient!(policy_params, α_θ, ∇lnπ)
	
			batch_ready .= false #once a gradient update has occured, inform all environments they need to perform a new step
			cs .*= γ
			num_updates += 1
		end

		
	end

	policy_and_value_components = form_policy_and_value_function(mdp, feature_vector, update_feature_vector!, policy_params, value_params)

	#note that this step is a noop unless the gradients are gpu objects in which case they get deallocated
	cleanup_gradient!(∇v̂)
	cleanup_gradient!(∇lnπ)

	return (;avg_step_rewards = avg_step_rewards, batch_episodes = batch_episodes, batch_episode_steps = batch_episode_steps, batch_episode_rewards = batch_episode_rewards, policy_parameters = policy_params, value_parameters = value_params, policy_and_value_components...)
end

# ╔═╡ b9792720-4b34-426e-a244-732b7ebce7a0
synchronous_nstep_actor_critic_linear(mdp::StateMDP{T, S, A, PTF, F1, F2, F3}, γ::T, max_steps::Integer, num_env::Integer, feature_vector, update_feature_vector!::Function; policy_params::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, zero(T)), value_params::Vector{T} = initialize_linear_parameters(feature_vector, zero(T)), kwargs...) where {T<:Real, S, A, PTF, F1, F2, F3} = synchronous_nstep_actor_critic!(policy_params, value_params, mdp, γ, max_steps, num_env, feature_vector, update_feature_vector!, (), (), (), (); kwargs...)

# ╔═╡ 0c7af119-461e-43ef-b899-0708ff088e45
#=╠═╡

@plutoprofview synchronous_nstep_actor_critic_linear(gridworld_state_mdp, 0.99f0, 10_000, 128, gridworld_feature, update_gridworld_feature!; α_θ = 16f-2, α_w =8f-2, N = 10)
  ╠═╡ =#

# ╔═╡ 04102d7f-b139-4923-8928-fe73d124e055
#=╠═╡
const gridworld_sync_nstep_ac = synchronous_nstep_actor_critic_linear(gridworld_state_mdp, 0.99f0, 1_000_000, 8, gridworld_feature, update_gridworld_feature!; α_θ = 2f-2, α_w = 6f-3, N = 20)
  ╠═╡ =#

# ╔═╡ 670641ea-8600-41c0-af5c-48e3d6bc7a0a
#=╠═╡
plot(cumsum(gridworld_sync_nstep_ac.avg_step_rewards) ./ (1:length(gridworld_sync_nstep_ac.avg_step_rewards)))
  ╠═╡ =#

# ╔═╡ 449125ea-7eb6-4bc3-b994-ea85bd7a68aa
#=╠═╡
eval_gridworld_final_policy(gridworld_sync_nstep_ac.policy_sample_action)
  ╠═╡ =#

# ╔═╡ d6228bca-6fe4-4d9c-a486-e13bac1b7c99
#=╠═╡
plot_gridworld_state_value_function(gridworld_sync_nstep_ac.value_function)
  ╠═╡ =#

# ╔═╡ ce65c62a-c646-4208-8a8d-8c99e20bee17
#=╠═╡
plot_gridworld_policy_function(gridworld_sync_nstep_ac.policy_function)
  ╠═╡ =#

# ╔═╡ 4e8a8484-8e2c-4d02-9f1b-2fdff77fde7c
function synchronous_nstep_actor_critic_fcann(mdp::StateMDP{T, S, A, PTF, F1, F2, F3}, γ::T, max_steps::Integer, num_env::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; reslayers = 0, use_μP::Bool = true, policy_params::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, length(mdp.actions), reslayers, use_μP), value_params::FCANNParams = initialize_fcann_value_params(policy_params, use_μP), l2::T = zero(T), dropout::T = zero(T), activation_list::Vector{Bool} = fill(true, length(hidden_layers)), use_gpu::Bool = false, kwargs...) where {T<:Real, S, A, PTF, F1, F2, F3}

	policy_setup = setup_fcann_batch_policy_arguments(policy_params, num_env, l2, dropout, use_μP, activation_list; use_gpu = use_gpu)
	value_setup = setup_fcann_batch_value_arguments(policy_setup, value_params, num_env, l2, dropout, use_μP, activation_list; use_gpu = use_gpu)

	!use_gpu && return synchronous_nstep_actor_critic!(policy_params, value_params, mdp, γ, max_steps, num_env, feature_vector, update_feature_vector!, value_setup..., policy_setup...; kwargs...)

	isempty(value_setup.gpu_args) && error("GPU backend is not available")
	isempty(policy_setup.gpu_args) && error("GPU backend is not available")

	output = synchronous_nstep_actor_critic!(policy_setup.gpu_args.params, value_setup.gpu_args.params, mdp, γ, max_steps, num_env, feature_vector, update_feature_vector!, value_setup.gpu_args.value_args, value_setup.gpu_args.value_gradient_args, policy_setup.gpu_args.policy_args, policy_setup.gpu_args.policy_gradient_args; kwargs...)


	FCANN.GPU2Host(value_params.weights, value_setup.gpu_args.params.weights)
	FCANN.GPU2Host(policy_params.weights, policy_setup.gpu_args.params.weights)

	value_setup.gpu_args.cleanup_vars()
	policy_setup.gpu_args.cleanup_vars()
	return (;output..., policy_parameters = deepcopy(policy_params), value_parameters = deepcopy(value_params))	#note that synchronous_actor_critic! will copy the gpu params which have been cleaned up so we need to replace this output in the named tuple with the parameters we transfered back to the host before cleaning up
end

# ╔═╡ 631ead41-0bf5-4bc0-bbe6-d98ceb32ca20
#=╠═╡
const gridworld_sync_nstep_ac2 = synchronous_nstep_actor_critic_fcann(gridworld_state_mdp, 0.99f0, 100_000, 16, gridworld_feature, update_gridworld_feature!, [32, 32]; reslayers = 1, α_θ = 64f-2, α_w = 32f-2, N = 5)
  ╠═╡ =#

# ╔═╡ 6c60168b-b5d6-4146-a530-c343adba4f87
#=╠═╡
plot(cumsum(gridworld_sync_nstep_ac2.avg_step_rewards) ./ (1:length(gridworld_sync_nstep_ac2.avg_step_rewards)))
  ╠═╡ =#

# ╔═╡ e716de3d-9673-4bd6-bed9-8ab1f65dcfa5
#=╠═╡
eval_gridworld_final_policy(gridworld_sync_nstep_ac2.policy_sample_action)
  ╠═╡ =#

# ╔═╡ ac9c3b9d-6eca-46cf-ba88-8ddcc32dea76
#=╠═╡
plot_gridworld_state_value_function(gridworld_sync_nstep_ac2.value_function)
  ╠═╡ =#

# ╔═╡ 18359ce1-43c0-4179-a1ca-6d0e131072a2
#=╠═╡
plot_gridworld_policy_function(gridworld_sync_nstep_ac2.policy_function)
  ╠═╡ =#

# ╔═╡ ccdb85e1-dfef-42b8-9e39-6e6062b93b19
#=╠═╡
synchronous_nstep_actor_critic_fcann(gridworld_state_mdp, 0.99f0, 10_000, 16, gridworld_feature, update_gridworld_feature!, [32, 32]; reslayers = 1, α_θ = 64f-2, α_w = 32f-2, N = 5)
  ╠═╡ =#

# ╔═╡ 82c3f031-0fe0-442a-8d14-91046aad4760
#=╠═╡
synchronous_nstep_actor_critic_fcann(gridworld_state_mdp, 0.99f0, 1_000, 16, gridworld_feature, update_gridworld_feature!, [32, 32]; reslayers = 1, α_θ = 64f-2, α_w = 32f-2, N = 5, use_gpu = true)
  ╠═╡ =#

# ╔═╡ ea607cf6-a263-4d44-9d72-dc9a5de055ad
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
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
BenchmarkTools = "~1.6.3"
DataStructures = "~0.19.3"
HypertextLiteral = "~0.9.5"
PlutoDevMacros = "~0.9.2"
PlutoPlotly = "~0.6.5"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.79"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "c12e28897d0bd84055b96a173f0d8d7a134b96ec"

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
# ╟─789ac927-e8dd-424a-8a28-b3240d295523
# ╠═0c0a7330-29bf-4326-8939-78b7e8b58d55
# ╠═cf40f4b3-4495-4f26-a007-18c6589ed4cf
# ╠═b3d7c539-d5a0-47fc-85bc-a62aafca8fa0
# ╠═1b4c3482-165e-4c09-bbc2-c705e5ceb2fe
# ╠═f9d3ee23-f39d-46e4-834e-86b8eee1ce50
# ╠═cb972f94-d22c-4d00-8c70-50daff8f697e
# ╠═576ff132-27d0-4a91-a955-e797fe6637c1
# ╠═a743f767-1ff6-4e1c-8c6f-88622d07c175
# ╠═f76f33f3-7cb3-4715-800c-9a0b2561f05b
# ╠═44f59eb0-a7e5-43ff-b4ca-657cc505220f
# ╠═f7a94436-b905-48b5-a0f3-ae26d0ecab5e
# ╠═2dd9b971-fa6e-4a55-8a82-b16739199fab
# ╠═1805574f-a668-477f-a6a9-e7ee29ce08bf
# ╠═75ba6587-ebe0-4f54-89f6-a65ec26abd63
# ╟─ce90a48f-4111-4f33-9448-a04af33e6231
# ╠═3a4510e6-054b-40fe-989d-7ac8c86db757
# ╟─05535cef-05f4-42a1-925c-ceb85bb6dfba
# ╠═830ba410-377a-423b-9e75-6884c8cbbbea
# ╟─78a997cb-146c-4cb5-b2ec-4bd1188909e6
# ╠═6fe10947-cdc4-48c5-8fbe-942714640dca
# ╠═d9437c04-d197-4b8c-b1ad-0029b3d77144
# ╟─0fd4da20-cf83-4071-b8a1-4259cbba7d8c
# ╠═05fa2e5c-8633-4c43-89ba-09dd0c83cdfa
# ╟─c7f4374c-3151-443b-9d6a-85581c3f2438
# ╠═a3d54085-dbdc-4889-9313-efd1ece2150a
# ╠═696650a0-9e0f-45d0-a768-679b02688f06
# ╟─e5091bad-761c-49ef-a84a-0fec4ce2fbd1
# ╠═57c4027a-0209-4e22-917a-6b1e4364186c
# ╟─62bc710e-cd7e-43a5-bf9e-da024802358c
# ╠═01697aad-517e-4e48-b543-92d2974990ef
# ╠═293ee629-3fb9-442d-be27-11e9c0545fae
# ╠═8b141bcf-db3e-44d2-96de-c916f4018740
# ╠═8705bc78-b419-4b30-8a1e-02398fd456b5
# ╠═2753435d-8ea2-4bab-9b06-cb0e36bcab99
# ╟─be0e9e8f-e95c-42c6-99ac-c4291eed0f66
# ╟─76bb3fb6-836b-42a7-8051-48e0856bedb3
# ╠═9a70941b-bf9e-4b3c-aef6-915f3e2019fe
# ╠═e4fd6d59-be28-4852-a2d2-6ddc1a40116d
# ╟─e8c3f789-7ede-4f6d-83ee-3050c9ef5840
# ╠═81058b8e-c80c-4a39-b33e-15b42d1225b8
# ╠═4f222b09-00e0-48b9-bd3e-6b6ebfba5727
# ╠═37a4303b-a4b9-423a-a5fa-3282c323f04b
# ╠═018a26c6-680e-42ed-b1ff-96b377808e11
# ╠═b1880149-4d45-4cfb-91cd-4c094ac5a1eb
# ╠═c5e0bbaf-a04d-4940-b7ab-3317f9d513a9
# ╟─fb94bc10-a580-4c32-b307-c551e4113bd7
# ╠═daa89863-cb36-4a12-a699-646d8ba55904
# ╠═e3fa7c27-706c-445b-a2e2-837c91fcb9a7
# ╠═b067424c-2b40-4d99-9dba-419af6fb2209
# ╠═ab6d36c1-5674-49f4-b291-f367840c9335
# ╠═809132ad-fad8-4bc7-b8f4-1f98dbc8503c
# ╟─afa30291-a919-44da-83c9-97cd2a43c168
# ╠═7edf78dd-9dad-4726-8173-c95f3e4ed6ab
# ╠═239e7e55-e68f-40f0-a666-c4c4c3ff3d35
# ╠═b2096a04-74f8-4287-aad5-dc27752a21f7
# ╟─74e2001a-12a4-4a96-bb1c-567e238cf6a9
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
# ╟─354a4a98-528f-46e2-93bd-c04f5a9ccad3
# ╠═de6aca47-7d2c-4c8c-8df9-c3e16e6dc2bd
# ╠═d1058fb3-cb03-4031-b089-ce5f077036a0
# ╠═b045eb8f-5911-4afe-b9aa-6bd291a2652c
# ╠═bbc470e8-0e6b-41a6-a547-22c339a87d63
# ╠═26146b44-e9ee-4da4-8431-862b8b17f68a
# ╠═1109fcd4-1706-4022-9422-f4e947f275af
# ╠═51a27e0e-14e5-4822-a255-bd63aac2a00e
# ╠═f0582c1f-7f6d-4f38-9051-fb5ef158612f
# ╟─710073d0-1af3-427a-9273-4044b252377b
# ╠═305e3122-76f2-4f90-9120-db20d2e7255a
# ╠═3eb1e4a2-bd61-4364-baa8-b9ef0ae68418
# ╠═4dad43fe-11e9-4949-82c1-c87503c2162a
# ╠═462af35e-dcbc-4f01-b0de-2e9193890b7c
# ╠═cbcb5270-2661-4a30-9787-6aa0d8915ae1
# ╠═6d87928f-0ed3-410e-bbe3-e2e338919806
# ╠═31eb8e01-380b-4553-90dc-22ffaea7aaac
# ╠═4f88dd4e-4f18-4770-a6eb-1ad1094b92c5
# ╠═70093ded-4ebc-48aa-bcfd-48d492aa1c5e
# ╟─14203d22-a8da-4e32-b2e8-d90936b83875
# ╠═b3c6aad2-027c-4bb2-93a2-e3175c7ee66a
# ╠═dac1ec53-d528-4f78-8306-294b0725a183
# ╠═ffea4162-4cc7-4ee0-b962-d2d8aa5660c3
# ╠═383c9892-b603-4d04-8af6-6f7f1308b1d7
# ╠═1c2765ce-72b8-4e51-bab1-f646536a5979
# ╠═ad924e38-1ced-42e8-8824-51c75506229e
# ╟─7b5a691c-69be-4c8b-a2af-fc80b6597086
# ╠═3ec9a9b2-4bd2-4228-94f5-d2f80f5831ea
# ╠═f8cb49ae-10b5-4a94-a5ab-24c6b70d597c
# ╠═98c778fb-aa6d-4372-80cc-e6a6751bee96
# ╠═5d2836bc-9dde-4bb7-a45d-309de2292671
# ╠═109954db-b333-4899-91e1-18f067742372
# ╟─1d56e95a-fb19-4f28-a581-f27bc45b1149
# ╠═c3953fda-dc68-48ca-b033-7a04dc2beae0
# ╠═2bfcda4a-e61a-4f85-bce2-b788e9372de2
# ╠═5d4dfd0c-440e-4187-a9fd-05ef01ec8eab
# ╠═0f3874fb-134f-4727-adc3-3eb1b3e1fd7d
# ╠═7e4a2949-6ec5-4761-9be0-777b989256dd
# ╟─e9cf4424-6a73-4f22-9a89-91d99f5e92b7
# ╟─5df8574a-251b-49f4-8fc0-60988a1263d2
# ╠═7604fb1e-e6ed-4314-be7b-a786c6422e10
# ╠═fbd74aff-5107-40cd-a02c-a0c5e0b19464
# ╟─16698d73-c9db-4680-966b-a246c7137e1e
# ╠═02253513-2802-4ecb-b14c-61e5d4e7fd86
# ╠═7603c756-4132-4166-b385-71e99fa69f40
# ╠═3a7a2d32-23e1-41da-b7d0-52bb29b31def
# ╠═b52c50ea-cd77-4560-b8f2-863962ff2546
# ╠═4b510e94-4ec2-4e82-ae8f-523ac90f34d9
# ╠═465ee26d-cf29-4dde-9ce6-c2030699996d
# ╠═5eca8610-aa4b-4ed8-a97c-43dbc2bae53c
# ╟─ddccecac-cc53-402b-b891-875ed332da6a
# ╠═9caedc2f-94d7-451f-85c8-28fdd387fb08
# ╠═f6e1e30c-0bb0-4a7f-8d89-11dc29b918b4
# ╟─52811707-b5e9-4b24-b7ca-c7786d1ad0b6
# ╟─1bea29d7-7268-40b5-88c1-0d56b0d0c89d
# ╠═55168857-d8d7-487e-a86b-f0bb6c7a1467
# ╠═82c69936-1e15-4c23-a869-0aa47edb1796
# ╠═47ddf8d8-54cf-4d4c-a429-d489804434b0
# ╠═0b17138b-9bcc-41cd-8b9d-3a079445109c
# ╠═4c19375a-ffb9-4b49-bf42-3d9557b4e8e5
# ╠═d570cc59-7263-46f9-99c8-8faa99e854ab
# ╠═e7a6798e-9c2d-4f9d-abeb-f8c28ebe542f
# ╠═4bb49599-941e-49b9-9c3a-079ee7f789e2
# ╠═b5583c7e-f11b-4b4c-a3b7-32a8bb95e3a2
# ╠═c3a15808-071a-433b-a26e-d0bfe879bcba
# ╠═3437d6ec-a373-4114-a144-29b4e642924e
# ╠═18bad2a5-5c90-4e83-b29a-415730b1d5bf
# ╠═b3afb879-c2ed-4408-b77d-fcc535934b17
# ╠═e366c806-9f44-4bdc-b2b0-8754d99c8342
# ╟─99190158-126a-4149-9d32-dffda0259cab
# ╟─c113e64f-7463-4714-8253-40d196496b1d
# ╠═aeb7304c-0f5f-4db9-b3c1-d19048fd4176
# ╠═c0b907ca-9959-45d9-8cc1-343d71ef5dd8
# ╠═e6f4574b-f28f-43b1-b8f9-7080ecacfb39
# ╠═7f848bd1-e325-427d-8d68-8f7c2d1e7039
# ╠═be468ef6-6e8f-4b9a-aa20-993102168ca6
# ╠═7335116e-670c-45ea-ae23-015b69964e5b
# ╠═a92a325e-4e3f-4c9f-8299-33b1a54cef10
# ╠═6e337a0f-b69c-4f18-9e33-bc792d588fbe
# ╠═90d81200-8b3e-4b14-b913-7c8eb0a965b0
# ╠═05e4d2f6-bfc4-426e-866e-834d39666eb7
# ╠═4622a796-1ef7-40b1-87da-a1799369f758
# ╠═be6065fa-83ba-41be-9e7c-8fff2646425b
# ╟─948a1e0b-83ae-4989-812e-83be6df4c86b
# ╠═c7994abc-e49d-4ccb-b09b-e68c14bb7d6f
# ╟─e0bd880a-b962-4b15-873a-428fea0624ee
# ╠═738241ce-0315-41d8-a67f-64c91479fe57
# ╟─ff1d3dea-9aa9-4832-9474-0095924b747d
# ╠═2fa46312-ec98-4680-be6e-a5ddd08baca2
# ╠═d5364951-0bfd-424a-8e8c-fecc05ce124b
# ╠═b65512a7-1a98-4d99-835f-eca70ced2404
# ╟─034109fe-6b46-4d77-bbc8-f1399c02bdac
# ╠═ed81e0f8-8b92-484f-b66a-0a7fda1b8dd2
# ╠═65ab73ac-0073-47dd-bf68-d565e6028d9a
# ╠═b4f95b92-127f-4ab4-92e8-c405fbf91e67
# ╠═4a206490-b150-4004-93b3-04d187d04619
# ╠═1a462642-08be-452a-8157-e6ac83640db9
# ╠═eebbc332-1020-4279-b82c-ccda4b3fe0cc
# ╠═85a24ea7-411e-4d11-a83b-bddad52772df
# ╠═0d045fdd-67e4-45d4-957e-c58b266dfe5e
# ╠═c106a277-cf6a-4a19-b17f-6370c42c43c8
# ╠═39ce5032-3a13-49c9-ab7d-d74d1aff883d
# ╠═89317d85-e18d-4940-9f1b-bf4f9fbf9880
# ╠═f411dc93-77df-49c6-a222-5985e8aea544
# ╠═ae0a282c-76ba-49a6-8a2b-02012c8dcda9
# ╠═04bc310f-bbaf-4d0f-9d5b-6c79b47a74c7
# ╠═1d5be8fa-e03e-4906-b772-25a1899275a6
# ╟─ee0405a4-a03c-4373-a270-475cda8de910
# ╟─be9639f8-e987-4448-a961-0cafaaaf4980
# ╠═44c9104e-8586-4202-8edd-eaea0073842a
# ╠═b9792720-4b34-426e-a244-732b7ebce7a0
# ╠═4e8a8484-8e2c-4d02-9f1b-2fdff77fde7c
# ╠═0c7af119-461e-43ef-b899-0708ff088e45
# ╠═04102d7f-b139-4923-8928-fe73d124e055
# ╠═670641ea-8600-41c0-af5c-48e3d6bc7a0a
# ╠═449125ea-7eb6-4bc3-b994-ea85bd7a68aa
# ╠═d6228bca-6fe4-4d9c-a486-e13bac1b7c99
# ╠═ce65c62a-c646-4208-8a8d-8c99e20bee17
# ╠═631ead41-0bf5-4bc0-bbe6-d98ceb32ca20
# ╠═6c60168b-b5d6-4146-a530-c343adba4f87
# ╠═e716de3d-9673-4bd6-bed9-8ab1f65dcfa5
# ╠═ac9c3b9d-6eca-46cf-ba88-8ddcc32dea76
# ╠═18359ce1-43c0-4179-a1ca-6d0e131072a2
# ╟─4ba0a4d3-d04b-4f8d-94be-0c953b9d8719
# ╠═ccdb85e1-dfef-42b8-9e39-6e6062b93b19
# ╠═82c3f031-0fe0-442a-8d14-91046aad4760
# ╟─33b59f50-07e1-11f1-9748-31081ab2ceaf
# ╠═a966b2b2-b3d9-4f28-9042-66167400f2cb
# ╠═ad872f3c-7be0-4427-bd1d-5afe25b6e9fa
# ╠═8b4b8bfa-9dfd-45c4-9ce0-8f4af97f9721
# ╠═e6aefa92-94d5-487d-91ba-b9a4d1b3277c
# ╠═ea607cf6-a263-4d44-9d72-dc9a5de055ad
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
