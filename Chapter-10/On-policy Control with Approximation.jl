### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 7cda4e0e-ed30-4389-bad7-c2552427e94a
using PlutoDevMacros

# ╔═╡ 1ba08cec-c8dc-4d04-8465-9f5bb6f4c79e
# ╠═╡ show_logs = false
PlutoDevMacros.@frompackage @raw_str(joinpath(@__DIR__, "..", "NonTabularRL.jl")) begin
	using NonTabularRL
	using >.Random, >.Statistics, >.LinearAlgebra, >.Transducers
end

# ╔═╡ 1bd08b27-634f-45b0-89be-d2f2ce7c0343
# ╠═╡ skip_as_script = true
#=╠═╡
using PlutoPlotly, PlutoUI, PlutoProfile, BenchmarkTools, LaTeXStrings, HypertextLiteral
  ╠═╡ =#

# ╔═╡ 7bae6cbe-b392-4b6c-a838-b93091712133
md"""
In this chapter we turn to the control problem, and like before we seek to approximate the state-action value function $\hat q(s, a, \boldsymbol{w})$ with the goal of applying policy improvement to find $q_* (s, a)$.
"""

# ╔═╡ b4c83bb2-b1ab-4458-9dfb-b319b1bd52a3
md"""
# 10.1 Episodic Semi-gradient Control

It is straightforward to extend the semi-gradient prediction methods in Chapter 9 to action values.  We simply consider examples of the form $S_t, A_t \rightarrow U_t$ where $U_t$ is any of the previously described update targets such as the Monte Carlo Return ($G_t$).  The new gradient-decent update for action-value prediction is:

$\boldsymbol{w}_{t+1} \doteq \alpha \left [ U_t - \hat q(S_t, A_t, \boldsymbol{w}_t) \right ] \nabla \hat q(S_t, A_t, \boldsymbol{w}_t)$

For example, the one-step Sarsa update is:

$\boldsymbol{w}_{t+1} \doteq \alpha \left [ R_{t+1} + \gamma \hat q(S_{t+1}, A_{t+1}, \boldsymbol{w}_t) - \hat q(S_t, A_t, \boldsymbol{w}_t) \right ] \nabla \hat q(S_t, A_t, \boldsymbol{w}_t)$

If the action set is discrete, then at the next state $S_{t+1}$ we can compute $\hat q(S_{t+1}, a, \boldsymbol{w}_t)$ for every action and then find the greedy action $A^*_{t+1} = \text{argmax}_a\hat q(S_{t+1}, a, \boldsymbol{w}_t)$.  Policy improvement is then done by changing the estimation policy ot a soft approximation of the greedy policy such as the $\epsilon$-greedy policy.  Actions are selected according to this same policy.
"""

# ╔═╡ 98e0f34a-d05c-4ac5-a892-4f5d6ae4e3c2
function semi_gradient_sarsa!(parameters::P, mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, estimate_value::Function, update_parameters!::Function, state_representation::AbstractVector{T}; α = one(T)/10, ϵ = one(T) / 10, gradients::P = deepcopy(parameters), kwargs...) where {P, T<:Real}
	s = mdp.initialize_state()
	i_a = rand(eachindex(mdp.actions))
	ep = 1
	step = 1
	epreward = zero(T)
	episode_rewards = zeros(T, max_episodes)
	episode_steps = zeros(Int64, max_episodes)
	action_values = zeros(T, length(mdp.actions))
	policy = zeros(T, length(mdp.actions))
	while (ep <= max_episodes) && (step <= max_steps)
		(r, s′) = mdp.ptf(s, i_a)
		epreward += r
		if mdp.isterm(s′)
			U_t = r
			s′ = mdp.initialize_state()
			i_a′ = rand(eachindex(mdp.actions))
			episode_rewards[ep] = epreward
			episode_steps[ep] = step
			epreward = zero(T)
			ep += 1
		else
			estimate_value(s′, parameters, state_representation, action_values)
			policy .= action_values
			make_ϵ_greedy_policy!(policy; ϵ = ϵ)
			i_a′ = sample_action(policy)
			q̂ = action_values[i_a′]
			U_t = r + γ*q̂
		end
		update_parameters!(parameters, gradients, state_representation, s, i_a, U_t, α)
		s = s′
		i_a = i_a′
		step += 1
	end
	return episode_rewards, episode_steps
end

# ╔═╡ 44f28dd6-f0ef-4b67-a92f-817b27ea0f0b
function semi_gradient_expected_sarsa!(parameters::P, mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, estimate_value::Function, update_parameters!::Function, state_representation::AbstractVector{T}; α = one(T)/10, ϵ = one(T) / 10, gradients::P = deepcopy(parameters), kwargs...) where {P, T<:Real}
	s = mdp.initialize_state()
	i_a = rand(eachindex(mdp.actions))
	ep = 1
	step = 1
	epreward = zero(T)
	episode_rewards = zeros(T, max_episodes)
	episode_steps = zeros(Int64, max_episodes)
	action_values = zeros(T, length(mdp.actions))
	while (ep <= max_episodes) && (step <= max_steps)
		(r, s′) = mdp.ptf(s, i_a)
		epreward += r
		if mdp.isterm(s′)
			U_t = r
			s′ = mdp.initialize_state()
			i_a′ = rand(eachindex(mdp.actions))
			episode_rewards[ep] = epreward
			episode_steps[ep] = step
			epreward = zero(T)
			ep += 1
		else
			q̂ = estimate_value(s′, parameters, state_representation, action_values)
			make_ϵ_greedy_policy!(action_values; ϵ = ϵ)
			i_a′ = sample_action(action_values)
			U_t = r + γ*q̂
		end
		update_parameters!(parameters, gradients, state_representation, s, i_a, U_t, α)
		s = s′
		i_a = i_a′
		step += 1
	end
	return episode_rewards, episode_steps
end

# ╔═╡ 6710b24b-9ef4-4330-8ed8-f52d7fbe1ed7
function linear_features_action_gradient_setup(problem::Union{StateMDP{T, S, A, P, F1, F2, F3}, StateMRP{T, S, P, F1, F2}}, state_representation::AbstractVector{T}, update_feature_vector!::Function) where {T<:Real, N, S <: Union{T, NTuple{N, T}}, A, P, F1<:Function, F2<:Function, F3<:Function}
	s0 = problem.initialize_state()
	update_feature_vector!(state_representation, s0) #verify that feature vector update is compatible with provided state representation

	function update_params!(parameters::Vector{Vector{T}}, gradients::Vector{Vector{T}}, state_representation::AbstractVector{T}, s::S, i_a::Integer, g::T, α::T)
		update_feature_vector!(state_representation, s)
		NonTabularRL.update_parameters!(parameters[i_a], gradients[i_a], state_representation, g, α)
	end
	
	
	function v̂(s::S, i_a::Integer, w::Vector{Vector{T}}, state_representation::AbstractVector{T}) where {T<:Real} 
		update_feature_vector!(state_representation, s)
		dot(state_representation, w[i_a])
	end

	function v̂(s::S, w::Vector{Vector{T}}, state_representation::AbstractVector{T}, action_values::Vector{T}) where {T<:Real} 
		update_feature_vector!(state_representation, s)
		best_value = typemin(T)
		for i_a in eachindex(w)
			q = dot(state_representation, w[i_a])
			best_value = max(q, best_value)
			action_values[i_a] = q
		end
		return best_value
	end
	
	return (value_function = v̂, parameter_update = update_params!, feature_vector = state_representation)
end

# ╔═╡ cb0a43ff-11fc-40c4-a601-daf5ad04e2e0
md"""
## Example 10.1: Mountain Car Task
"""

# ╔═╡ edf014bb-3fd6-446b-bbef-736b684519a9
function initialize_car_state()
	a = rand(Float32) * 0.2f0
	x = a - 0.6f0
	ẋ = 0f0
	(x, ẋ)
end

# ╔═╡ 7e473e00-0ea0-4f79-ae7a-e5d7eb139959
initialize_car_state()

# ╔═╡ 061ab5b7-7edb-4757-84e1-224c93375714
const mountain_car_actions = [-1f0, 0f0, 1f0]

# ╔═╡ 5fdea69c-00c3-42bc-88fd-56ab6b0ba72b
function mountain_car_step(s::Tuple{Float32, Float32}, i_a::Int64)
	a = mountain_car_actions[i_a]
	ẋ′ = clamp(s[2] + 0.001f0*a - 0.0025f0*cos(3*s[1]), -0.07f0, 0.07f0)
	x′ = clamp(s[1] + ẋ′, -1.2f0, 0.5f0)
	x′ == -1.2f0 && return (-1f0, (x′, 0f0))
	return (-1f0, (x′, ẋ′))
end

# ╔═╡ 1681538a-81ca-48df-9fcb-2b2dc83acd5d
const mountain_car_transition = StateMDPTransitionSampler(mountain_car_step, initialize_car_state())

# ╔═╡ 4d5f43aa-2b0f-4a4a-910f-bf3425244192
const mountain_car_mdp = StateMDP(mountain_car_actions, mountain_car_transition, initialize_car_state, s -> s[1] == 0.5f0)

# ╔═╡ 4d6d3d2c-ae76-485a-8f7e-d073a307b2c9
#=╠═╡
function show_mountaincar_trajectory(π::Function, max_steps::Integer, name)
	states, actions, rewards, sterm, nsteps = runepisode(mountain_car_mdp; π = π, max_steps = max_steps)
	positions = [s[1] for s in states]
	velocities = [s[2] for s in states]
	tr1 = scatter(x = positions, y = velocities, mode = "markers", showlegend = false)
	tr2 = scatter(y = positions, showlegend = false)
	tr3 = scatter(y = [mountain_car_actions[i] for i in actions], showlegend = false)
	p1 = plot(tr1, Layout(xaxis_title = "position", yaxis_title = "velocity"))
	p2 = plot(tr2, Layout(xaxis_title = "time", yaxis_title = "position"))
	p3 = plot(tr3, Layout(xaxis_title = "time", yaxis_title = "action"))
	mdname = Markdown.parse(name)
	md"""
	$mdname
	Total Reward: $(sum(rewards))
	$([p1 p2 p3])
	"""
end
  ╠═╡ =#

# ╔═╡ 6f6f21c3-88e3-4259-8184-6066490ac815
#=╠═╡
show_mountaincar_trajectory(s -> 3, 200, "Mountain Car Trajectory for Acceleration only Policy")
  ╠═╡ =#

# ╔═╡ aa68518b-82c4-488f-8ba0-8fd1d6866507
mountain_car_setup = tile_coding_gradient_setup(mountain_car_mdp, (-1.2f0, 0.5f0), (-0.07f0, 0.07f0), (1f0/8, 1f0/8), 8, (1, 3); linear_setup = linear_features_action_gradient_setup)

# ╔═╡ e6766bbe-2705-4ae7-b341-6b8715137c90
function mountaincar_test(max_episodes::Integer, α::Float32, ϵ::Float32; method = semi_gradient_sarsa!)
	setup = tile_coding_gradient_setup(mountain_car_mdp, (-1.2f0, 0.5f0), (-0.07f0, 0.07f0), (1f0/8, 1f0/8), 8, (1, 3); linear_setup = linear_features_action_gradient_setup)
	params = [zeros(Float32, length(setup.feature_vector)) for i_a in eachindex(mountain_car_mdp.actions)]
	(rewards, steps) = semi_gradient_sarsa!(params, mountain_car_mdp, 1f0, max_episodes, typemax(Int64), setup.value_function, setup.parameter_update, setup.get_feature_vector(mountain_car_mdp.initialize_state()); α = α, ϵ = ϵ)
	feature_vector = setup.get_feature_vector(mountain_car_mdp.initialize_state())
	q̂(s, i_a) = setup.value_function(s, i_a, params, feature_vector)
	return (value_function = q̂, episode_rewards = rewards, episode_steps = steps)
end

# ╔═╡ a7474f60-0a16-4dc0-a82d-aab9911354ab
(q̂_mountain_car, episode_rewards, episode_steps) = mountaincar_test(500, 0.00008f0, 0.01f0; method = semi_gradient_expected_sarsa!)

# ╔═╡ c669957e-70d8-4fef-be9b-7e16d900dc62
#=╠═╡
function plot_mountaincar_action_values()
	n = 100
	xvals = LinRange(-1.2f0, 0.5f0, n)
	vvals = LinRange(-0.07f0, 0.07f0, n)
	values = zeros(Float32, n, n)
	actions = zeros(Float32, n, n)
	for (i, x) in enumerate(xvals)
		for (j, v) in enumerate(vvals)
			(q̂, i_a) = findmax(i -> q̂_mountain_car((x, v), i), 1:3)
			values[i, j] = q̂
			actions[i, j] = mountain_car_actions[i_a]
		end
	end
	p1 = plot(heatmap(x = xvals, y = vvals, z = values), Layout(xaxis_title = "position", yaxis_title = "velocity"))
	p2 = plot(heatmap(x = xvals, y = vvals, z = actions), Layout(xaxis_title = "position", yaxis_title = "velocity"))
	[p1 p2]
end
  ╠═╡ =#

# ╔═╡ b4a6d133-b045-4413-b33e-59d887df459b
#=╠═╡
plot_mountaincar_action_values()
  ╠═╡ =#

# ╔═╡ 512387d4-4b0f-4016-8a94-c0ee722182da
π_mountain_car(s) = argmax(i_a -> q̂_mountain_car(s, i_a), eachindex(mountain_car_actions))

# ╔═╡ 34e9cb53-d914-4b1e-8dad-34ee6515b8d9
#=╠═╡
show_mountaincar_trajectory(π_mountain_car, 500, "Sarsa Learned Policy")
  ╠═╡ =#

# ╔═╡ 65934e92-57f6-4e01-ac61-7274ef9a941c
#=╠═╡
plot(scatter(y = -episode_rewards), Layout(yaxis_type = "log"))
  ╠═╡ =#

# ╔═╡ 72822da1-edb9-4d70-bd4e-58b64f3fa01a
# ╠═╡ disabled = true
#=╠═╡
test_rewards = 1:100 |> Map(_ -> mountaincar_test(500, 0.005f0, 0.01f0; method = semi_gradient_expected_sarsa!).episode_rewards) |> foldxt((a, b) -> a .+ b) |> v -> -v ./ 100
  ╠═╡ =#

# ╔═╡ f9d1ce79-7e33-46d1-859f-d19345b0f0ae
#=╠═╡
plot(scatter(y = test_rewards), Layout(yaxis_type = "log"))
  ╠═╡ =#

# ╔═╡ 0639007a-6881-449c-92a7-ed1c0681d2eb
md"""
# 10.2 Semi-gradient *n*-step Sarsa
"""

# ╔═╡ 37d3812c-2710-4f97-b2f8-4dfd6f9b8390
md"""
> ### *Exercise 10.1* 
> We have not explicitely considered or given pseudocode for any Monte Carlo methods in this chapter.  What would they be like?  Why is it reasonable not to give pseudocode for them?  How would they perform on the Mountain Car task?

Monte Carlo methods require an episode to terminate prior to updating any action value estimates.  After the final reward is retrieved then all the action value pairs visited along the trajectory can be updated and the policy can be updated prior to starting the next episode.  For tasks such as the Mountain Car task where a random policy will likely never terminate, such a method will never be able to complete a single episode worth of updates.  We saw in earlier chapters with the racetrack and gridworld examples that for some environments a bootstrap method is the only suitable one given this possibility of an episode never terminating.
"""

# ╔═╡ 6289fd48-a2ea-43d3-bcf8-bcc29447d425
md"""
> ### *Exercise 10.2* 
> Give pseudocode for semi-gradient one-step *Expected* Sarsa for control.

Use the same pseudocode given for semi-gradient one-step Sarsa but with the following change to the weight update step in the non-terminal case:

$\mathbf{w} \leftarrow \mathbf{w} + \alpha[R + \gamma \sum_a \pi(a|S^\prime)\hat q(S^\prime, a, \mathbf{w}) - \hat q(S, A, \mathbf{w}) ] \nabla \hat q(S, A, \mathbf{w})$

where π is the currently used policy which is ϵ greedy with respect to q̂
"""

# ╔═╡ 15fe88ba-43a3-42cd-ba55-45f1586276e3
md"""
> ### *Exercise 10.3* 
> Why do the results shown in Figure 10.4 have higher standard errors at large *n* than at small *n*?

At large n more of the reward function comes from the actual trajectory observed during a run.  Since random actions are taken initially there will be more spread in the observed reward estimates than with 1 step bootstrapping which is more dependent on the initialization of the action value function.  If ties are broken randomly then you would select random actions for the first n-steps of bootstrapping thus experience more spread in the early trajectories for higher n.
"""

# ╔═╡ 87b277b6-5c79-45fd-b6f3-e2e4ccf18f61
md"""
# 10.3 Average Reward: A New Problem Setting for Continuing Tasks
"""

# ╔═╡ 60901786-2f6f-451d-971d-27e684d079fa
md"""
> ### *Exercise 10.4* 
> Give pseudocode for a differential version of semi-gradient Q-learning.

Given the pseudocode for semi-gradient Sarsa, make the following changes:

$\vdots$

Initialize S

Loop for each step of episode:

Choose A from S using ϵ-greedy policy
Take action A, observe R, S'

$\delta \leftarrow R - \bar R + \max_a \hat q(S^\prime, a, \mathbf{w}) - \hat q(S, A, \mathbf{w})$

$\vdots$
$S \leftarrow S^\prime$

"""

# ╔═╡ d06375b3-f377-45a6-be16-01b22c5a2b3f
md"""
> ### *Exercise 10.5* 
> What equations are needed (beyond 10.10) to specify the differential version of TD(0)?

10.10 includes a reward estimate at time t, $\bar R_t$, which also needs to be updated.  The TD error represents the newly observed reward the was experienced in excess of the estimated average to the update equation should move $\bar R$ in the direction of the TD error.
"""

# ╔═╡ 2c6951f9-33cb-400e-a83a-1a16f2ee0870
md"""
> ### *Exercise 10.6* 
> Suppose there is an MDP that under any policy produces the deterministic sequence of rewards +1, 0, +1, 0, +1, 0, . . . going on forever. Technically, this violates ergodicity; there is no stationary limiting distribution $μ_\pi$ and the limit (10.7) does not exist. Nevertheless, the average reward (10.6) is well defined. What is it? Now consider two states in this MDP. From A, the reward sequence is exactly as described above, starting with a +1, whereas, from B, the reward sequence starts with a 0 and then continues with +1, 0, +1, 0, . . .. We would like to compute the di↵erential values of A and B. Unfortunately, the differential return (10.9) is not well defined when starting from these states as the implicit limit does not exist. To repair this, one could alternatively define the differential value of a state as $v_\pi (s) \dot = \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} \sum_{t=0}^h \gamma^t \left ( \mathbb{E_\pi} [R_{t+1}|S_0=s]-r(\pi)  \right )$.  Under this definition what are the differential values of states A and B?

The average reward is 0.5 per step.

$v_\pi (A) = \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} 0.5 - 0.5\gamma + 0.5 \gamma^2 - 0.5\gamma^3 + \cdots =0.5\lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty}\sum_{t=0}^h (-\gamma)^t$
$=0.5\lim_{\gamma \rightarrow 1}\frac{1}{\gamma +1 } = 0.25$

$v_\pi (B) = \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} -0.5 + 0.5\gamma - 0.5 \gamma^2 + 0.5\gamma^3 + \cdots =-0.5\lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty}\sum_{t=0}^h (-\gamma)^t$
$=-0.5\lim_{\gamma \rightarrow 1}\frac{1}{\gamma +1 } = -0.25$
"""

# ╔═╡ 4a67aeba-dfaf-480d-84eb-7b8bcda549cb
md"""
> ### *Exercise 10.7* 
> Consider a Markov reward process consisting of a ring of three states A, B, and C, with state transitions going deterministically around the ring.  A reward of +1 is received upon arrival in A and otherwise the reward is 0.  What are the differential values of the three states, using (10.13)

From 10.13 we have 

$v_\pi (s) \dot = \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} \sum_{t=0}^h \gamma^t \left ( \mathbb{E_\pi} [R_{t+1}|S_0=s]-r(\pi)  \right )$

The average reward per step is $\frac{1}{3}$ so we can apply the same method used in exercise 10.6 where the elements inside the parentheses of the sum are: $\frac{2}{3}$ for $C \rightarrow A$ and $-\frac{1}{3}$ for the other two.  Starting in state A we transition twice and then on the third arrive in state A leading to the following mean corrected values of $-\frac{1}{3}$, $-\frac{1}{3}$, and $\frac{2}{3}$.  The other states will have these values cyclically permuted leading to the following infinite sums:

For state A:
$-\frac{1}{3} - \frac{1}{3}\gamma + \frac{2}{3}\gamma^2 - \frac{1}{3}\gamma^3 - \frac{1}{3}\gamma^4 + \cdots$

For state B:
$-\frac{1}{3} + \frac{2}{3}\gamma - \frac{1}{3}\gamma^2 - \frac{1}{3}\gamma^3 + \frac{2}{3}\gamma^4 + \cdots$

For state C:
$\frac{2}{3} - \frac{1}{3}\gamma - \frac{1}{3} \gamma^2 + \frac{2}{3}\gamma^3 + \cdots = 3 \times (2 - \gamma - \gamma^2 + 2\gamma^3 + \cdots)$

Comparing these sequences we have:

$\gamma \times v(A) = v(C) - \frac{2}{3} \implies v(A) = \frac{v(C) - \frac{2}{3}}{\gamma}$
$\gamma \times v(B) = v(A) + \frac{1}{3} \implies v(A) = \gamma \times v(B) - \frac{1}{3}$

so

$\frac{v(C) - \frac{2}{3}}{\gamma} = \gamma \times v(B) - \frac{1}{3} \implies v(C) = \gamma \left ( \gamma \times v(B) - \frac{1}{3} \right ) + \frac{2}{3}$

also 

$\gamma \times v(C) = v(B) + \frac{1}{3} \implies v(C) = \frac{v(B) + \frac{1}{3}}{\gamma}$

Equation the two sides for $v(C)$ that only contain $v(B)$ terms we have:

$\frac{v(B) + \frac{1}{3}}{\gamma} = \gamma \left ( \gamma \times v(B) - \frac{1}{3} \right ) + \frac{2}{3}$

$v(B) = \gamma \left ( \gamma \left ( \gamma \times v(B) - \frac{1}{3} \right ) + \frac{2}{3} \right ) - \frac{1}{3} = \gamma^3 v(B) - \gamma^2 \frac{1}{3} + \gamma\frac{2}{3} - \frac{1}{3}$

$v(B) \left ( 1 - \gamma^3 \right ) = - \gamma^2 \frac{1}{3} + \gamma\frac{2}{3} - \frac{1}{3} \implies v(B) = \frac{- \gamma^2 \frac{1}{3} + \gamma\frac{2}{3} - \frac{1}{3}}{1 - \gamma^3}$

$v(B) = -\frac{1}{3} \frac{\gamma^2 - 2\gamma + 1}{1 - \gamma^3} = -\frac{1}{3} \frac{(\gamma - 1)^2}{-(\gamma - 1)(\gamma^2 + \gamma + 1)} = \frac{1}{3} \frac{\gamma - 1}{\gamma^2 + \gamma + 1}$

Therefore, 

$\lim_{\gamma \rightarrow 1} v(B) = \frac{1}{3} \frac{1 - 1}{3} = 0$
$\lim_{\gamma \rightarrow 1} v(A) = \gamma \times 0 - \frac{1}{3} = -\frac{1}{3}$
$\lim_{\gamma \rightarrow 1} v(C) =  \frac{0 + \frac{1}{3}}{\gamma} = \frac{1}{3}$
"""

# ╔═╡ 9aeacb77-5c2b-4244-878f-eb5d52af49e0
md"""
> ### *Exercise 10.8* 
> The pseudocode in the box on page 251 updates $\overline{R}_t$ using $\delta_t$ as an error rather than simply $R_{t+1} - \overline{R}_t$.  Both errors work, but using $\delta_t$ is better.  To see why, consider the ring MRP of three states from Exercise 10.7.  The estimate of the average reward should tend towards its true value of $\frac{1}{3}$.  Suppose it was already there and was held stuck there.  What would the sequence of $R_{t+1} - \overline{R}_t$ errors be?  What would the sequence of $\delta_t$ errors be (using Equation 10.10)?  Which error sequence would produce a more stable estimate of the average reward if the estimate were allowed to change in response to the errors? Why?

The sequence of $R_{t+1} - \bar R_t$ would be given by the cyclical sequence of rewards.  Let's assume we start the sequence at state A.  Then our reward sequence will be 0, 0, 1, 0, 0, 1... so the error sequence will be $-\frac{1}{3}$, $-\frac{1}{3}$, $\frac{2}{3}$,...  If we update the average error estimate using these corrections it would remain centered at the correct value but fluctuate up and down with each correction.

In order to calculate $\delta_t$ we must use the definition given by 10.10:

$\delta_t = R_{t+1} - \bar R_t + \hat v(S_{t+1}, \mathbf{w}_t) - \hat v(S_t, \mathbf{w}_t)$

This equation requires us to have value estimates for each state which we can assume have converged to the true values as we have for the average reward estimate: $\hat v(A) = -\frac{1}{3}$, $\hat v(B) = 0$, and $\hat v(C) = \frac{1}{3}$.  Starting at state A, $\delta_t = 0 - \frac{1}{3} + 0 - -\frac{1}{3} = 0$.  For the following state we have $0 - \frac{1}{3} + \frac{1}{3} - 0$.  Finally we have $1 - \frac{1}{3} + -\frac{1}{3} - \frac{1}{3} = 0$.  So if we use the TD error to update our average reward estimate, at equilibrium all the values will remain unchanged.

"""

# ╔═╡ 38f9069b-1675-4012-b3e7-74ddbdfd73cb
md"""
# 10.4 Deprecating the Discounted Setting

In a special case of indistinguishable states, we can only use the actions and reward sequences to analyze a continuing task.  For a policy $\pi$, the average of the discounted returns with discount factor $\gamma$ is always $\frac{r(\pi)}{1-\gamma}$.  Therefore the *ordering* of all policies is independent of the discount rate and would match the ordering we get in the average reward setting.  This derivation however depends on states being indistinguishable allowing us to match up the weights on reward sequences from different policies.

We can use discounting in approximate solution methods regardless but then $\gamma$ changes from a problem parameter to a solution method parameter.  Unfortunately, discounting algorithms with function approximation do not optimize discounted value over the on-policy distribution, and thus are not guaranteed to optimze average reward.

The root cause of the problem applying discounting with function approximation is that we have lost the policy improvement theorem which states that a policy $\pi^\prime$ is better than policy $\pi$ if $v_{\pi^\prime}(s) \geq v_\pi(s) \forall s\in \mathcal{S}$.  Under this theorem we could take a deterministic policy, choose a specific state, and find a new action at that state with a higher expected reward than the current policy.  If the policy is an approximation function that uses states represented by feature vectors, then adjusting the parameters can in general affect the actions at many states including ones that have not been encountered yet.  In fact, with approximate solution methods we cannot guarantee  policy improvement in any setting.  Later we will introduce a theoretical guarantee called the "policy-gradient theorem" but for an alternative class of algorithms based on parametrized policies.
"""

# ╔═╡ c0318318-5ca4-4dea-86da-9092cd774656
md"""
Applying the derivation of discount independence to the MDP in exercise 3.22 who's optimal policy depends on $\gamma$

$J(\pi) = \sum_s \mu_\pi(s)v_\pi^\gamma(s)$

Consider $\pi_{left}$: $J(\pi_{left})=0.5 \times (1 + 0 + \gamma^2 + 0 + \gamma^4 + 0 + \cdots) + 0.5 \times(0 + \gamma + 0 + \gamma^3 + 0 + \gamma^5 + \cdots)$
$J(\pi_{left}) = 0.5 \times (1 + \gamma + \gamma^2 + \gamma^3 + \gamma^4 + \gamma^5 + \cdots)$

Consider $\pi_{right}$: $J(\pi_{right})=0.5 \times (0 + 2\gamma + 0 + 2\gamma^3 + 0 + \cdots) + 0.5 \times(2 + 0 + 2\gamma^2 + 0 + 2\gamma^4 + \cdots)$
$J(\pi_{right}) = 0.5 \times 2 \times (1 + \gamma + \gamma^2 + \gamma^3 + \gamma^4 + \gamma^5 + \cdots)$

So both average reward values have the same factor for the discount rate and thus the right policy appears better since the average reward value is higher.  Previously, we had calculated that a discount rate less than 0.5 made the left policy favorable since the reward was obtained sooner going left vs right.  In the original problem we can consider the value of the top state for both left and right policies:
$v_{\pi_{left}} (top) = 1 + 0 + \gamma^2 + 0 + \gamma^4 + \cdots = 1 + \gamma^2 + \gamma^4 + \cdots$
$v_{\pi_{right}} (top) = 0 + 2\gamma + 0 + 2\gamma^3 + \cdots = 2 \times (\gamma + \gamma^3 + \cdots) = 2\gamma(v_{\pi_{left}}(top))$

Clearly for $\gamma > 0.5$ the right policy is better.

Similarly, we can consider the value of the left state for both left and right policies:
$v_{\pi_{left}} (left) = 0 + \gamma + 0 + \gamma^3 + \cdots = \gamma + \gamma^3 + \cdots$
$v_{\pi_{right}} (left) = 0 + 0 + 2\gamma^2 + 0  + 2\gamma^4 + \cdots = 2 \times (\gamma^2 + \gamma^4 + \cdots) = 2\gamma(v_{\pi_{left}}(left))$

Again, for $\gamma > 0.5$ the right policy is better.

And finally for the right state:
$v_{\pi_{left}} (right) = 2 + \gamma + 0 + \gamma^3 + 0 + \gamma^5 \cdots = 2+\gamma(1 + \gamma^2 + \gamma^4 + \cdots)=2 + \frac{\gamma}{1-\gamma^2}$ 
$= \frac{2(1-\gamma^2) + \gamma}{1-\gamma^2} = \frac{2 - 2\gamma^2 + \gamma}{1-\gamma^2}$
$v_{\pi_{right}} (right) = 2 + 0 + 2\gamma^2 + 0 + 2\gamma^4 +  \cdots = 2 \times (1+\gamma^2 + \gamma^4 + \cdots) = \frac{2}{1-\gamma^2}$

$\frac{v_{\pi_{left}} (right)}{v_{\pi_{right}} (right)}=\frac{2 - 2\gamma^2 + \gamma}{2}$

For $\gamma=0$ this quantity is 1 meaning the policies are equal and for $\gamma=1$ this quantity is 0.5 meaning that the right policy is better.  At $\gamma=0.5$ the quantity is $\frac{2 - 0.5 + 0.5}{2}=\frac{2}{2}=1$ meaning they are equal.  The maximum value occurs at $2\gamma = 0.5 \implies \gamma = 0.25$ with a ratio value of $\frac{2 - 0.125 + 0.25}{2}=\frac{2.125}{2}=1.0625$ meaning that the left policy is slightly better or equal from $0 \leq \gamma \leq 0.5$ and worse at $\gamma > 0.5$ which matches the earlier states.
"""

# ╔═╡ b1319fd7-5043-41d9-8971-ad88725f2d3c
md"""
The reason why the left policy can be better if $\gamma < 0.5$ in the original example is because it has a higher value in each state considered.  Consider $\gamma = 0.25$.  The left policy has the following approximate discounted value estimates for top, left, right: 

1.0667, 0.2667, 2.2667. 

Meanwhile the right policy has the corresponding values of: 

0.533, 0.133, 2.133.

Each value is smaller for the right policy.  However when we calculate the average value calculated over the long term distribution of states, the left policy averages the first two values while the right policy averages the first and third values because in the long run we expect the left policy to only exist in the top and left state while the right policy will exist in the top and right state.  Because the right state has such a high value for both policies but only the right policy includes it in the average it makes its entire objective estimate higher.  However, we can see that in the event of being in the right state, it is still a higher value expectation following the left policy in this case.  The decision to average based on the final distribution results in a policy ordering that doesn't match with what we know to be the optimal policy from the policy improvement theorem over finite states.
"""

# ╔═╡ e1e21ba6-07a6-4c35-ba71-0eaf6ccf74d6
md"""
# 10.5 Differential Semi-gradient *n*-step Sarsa
"""

# ╔═╡ a649e52b-e428-4f13-8628-7373b1163a4e
md"""
> ### *Exercise 10.9* 
> In the differential semi-gradient n-step Sarsa algorithm, the step-size parameter on the average reward, $\beta$, needs to be quite small so that $\bar R$ becomes a good long-term estimate of the average reward. Unfortunately, $\bar R$ will then be biased by its initial value for many steps, which may make learning inefficient. Alternatively, one could use a sample average of the observed rewards for $\bar R$. That would initially adapt rapidly but in the long run would also adapt slowly. As the policy slowly changed, $\bar R$ would also change; the potential for such long-term nonstationarity makes sample-average methods ill-suited. In fact, the step-size parameter on the average reward is a perfect place to use the unbiased constant-step-size trick from Exercise 2.7. Describe the specific changes needed to the boxed algorithm for differential semi-gradient n-step Sarsa to use this trick.

At the start initialize $\bar o = 0$ and select $\lambda > 0$ small instead of $\beta$. 

Within the loop under the $\tau \geq 0$ line, add two lines; one to update $\bar o$ and one to calculate the update rate for the average reward: 

Line 1: $\bar o \leftarrow \bar o + \lambda (1 - \bar o)$

Line 2: $\beta = \lambda / \bar o$

As steps progress $\beta$ will approach $\lambda$ but early on will take on much larger values as $\bar o$ starts close to 0 and approaches 1.
"""

# ╔═╡ e40c2294-7283-4af6-b593-8e348b5f29d9
md"""
# Dependencies
"""

# ╔═╡ 0742b6d4-3792-4681-acf1-608fb39afd1e
#=╠═╡
TableOfContents()
  ╠═╡ =#

# ╔═╡ b06f651a-20b0-4dbc-92b0-38a4819d5165
html"""
	<style>
		main {
			margin: 0 auto;
			max-width: min(1200px, 90%);
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
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
BenchmarkTools = "~1.5.0"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.3.1"
PlutoDevMacros = "~0.9.0"
PlutoPlotly = "~0.5.0"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.48"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.5"
manifest_format = "2.0"
project_hash = "c720785703774427b6bdd8e4fa682c0cb44ad8d4"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BaseDirs]]
git-tree-sha1 = "cb25e4b105cc927052c2314f8291854ea59bf70a"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.2.4"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b5278586822443594ff615963b0c09755771b3e0"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.26.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

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

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "82d8afa92ecf4b52d78d869f038ebfb881267322"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.FlameGraphs]]
deps = ["AbstractTrees", "Colors", "FileIO", "FixedPointNumbers", "IndirectArrays", "LeftChildRightSiblingTrees", "Profile"]
git-tree-sha1 = "d9eee53657f6a13ee51120337f98684c9c702264"
uuid = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"
version = "0.2.10"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "4b415b6cccb9ab61fec78a621572c82ac7fa5776"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.35"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

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
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "cceb0257b662528ecdf0b4b4302eb00e767b38e7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlutoDevMacros]]
deps = ["JuliaInterpreter", "Logging", "MacroTools", "Pkg", "TOML"]
git-tree-sha1 = "72f65885168722413c7b9a9debc504c7e7df7709"
uuid = "a0499f29-c39b-4c5c-807c-88074221b949"
version = "0.9.0"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "BaseDirs", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "Reexport", "TOML"]
git-tree-sha1 = "653b48f9c4170343c43c2ea0267e451b68d69051"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.5.0"

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
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "efc140104e6d0ae3e7e30d56c98c4a927154d684"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.48"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProfileCanvas]]
deps = ["FlameGraphs", "JSON", "Pkg", "Profile", "REPL"]
git-tree-sha1 = "41fd9086187b8643feda56b996eef7a3cc7f4699"
uuid = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
version = "0.1.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
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

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

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

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

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
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─7bae6cbe-b392-4b6c-a838-b93091712133
# ╟─b4c83bb2-b1ab-4458-9dfb-b319b1bd52a3
# ╠═98e0f34a-d05c-4ac5-a892-4f5d6ae4e3c2
# ╠═44f28dd6-f0ef-4b67-a92f-817b27ea0f0b
# ╠═6710b24b-9ef4-4330-8ed8-f52d7fbe1ed7
# ╠═cb0a43ff-11fc-40c4-a601-daf5ad04e2e0
# ╠═edf014bb-3fd6-446b-bbef-736b684519a9
# ╠═7e473e00-0ea0-4f79-ae7a-e5d7eb139959
# ╠═5fdea69c-00c3-42bc-88fd-56ab6b0ba72b
# ╠═1681538a-81ca-48df-9fcb-2b2dc83acd5d
# ╠═061ab5b7-7edb-4757-84e1-224c93375714
# ╠═4d5f43aa-2b0f-4a4a-910f-bf3425244192
# ╠═6f6f21c3-88e3-4259-8184-6066490ac815
# ╠═4d6d3d2c-ae76-485a-8f7e-d073a307b2c9
# ╠═aa68518b-82c4-488f-8ba0-8fd1d6866507
# ╠═e6766bbe-2705-4ae7-b341-6b8715137c90
# ╠═b4a6d133-b045-4413-b33e-59d887df459b
# ╠═c669957e-70d8-4fef-be9b-7e16d900dc62
# ╠═a7474f60-0a16-4dc0-a82d-aab9911354ab
# ╠═512387d4-4b0f-4016-8a94-c0ee722182da
# ╠═34e9cb53-d914-4b1e-8dad-34ee6515b8d9
# ╠═65934e92-57f6-4e01-ac61-7274ef9a941c
# ╠═72822da1-edb9-4d70-bd4e-58b64f3fa01a
# ╠═f9d1ce79-7e33-46d1-859f-d19345b0f0ae
# ╠═0639007a-6881-449c-92a7-ed1c0681d2eb
# ╟─37d3812c-2710-4f97-b2f8-4dfd6f9b8390
# ╟─6289fd48-a2ea-43d3-bcf8-bcc29447d425
# ╟─15fe88ba-43a3-42cd-ba55-45f1586276e3
# ╟─87b277b6-5c79-45fd-b6f3-e2e4ccf18f61
# ╟─60901786-2f6f-451d-971d-27e684d079fa
# ╟─d06375b3-f377-45a6-be16-01b22c5a2b3f
# ╟─2c6951f9-33cb-400e-a83a-1a16f2ee0870
# ╟─4a67aeba-dfaf-480d-84eb-7b8bcda549cb
# ╟─9aeacb77-5c2b-4244-878f-eb5d52af49e0
# ╟─38f9069b-1675-4012-b3e7-74ddbdfd73cb
# ╟─c0318318-5ca4-4dea-86da-9092cd774656
# ╟─b1319fd7-5043-41d9-8971-ad88725f2d3c
# ╟─e1e21ba6-07a6-4c35-ba71-0eaf6ccf74d6
# ╟─a649e52b-e428-4f13-8628-7373b1163a4e
# ╟─e40c2294-7283-4af6-b593-8e348b5f29d9
# ╠═7cda4e0e-ed30-4389-bad7-c2552427e94a
# ╠═1ba08cec-c8dc-4d04-8465-9f5bb6f4c79e
# ╠═1bd08b27-634f-45b0-89be-d2f2ce7c0343
# ╠═0742b6d4-3792-4681-acf1-608fb39afd1e
# ╠═b06f651a-20b0-4dbc-92b0-38a4819d5165
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
