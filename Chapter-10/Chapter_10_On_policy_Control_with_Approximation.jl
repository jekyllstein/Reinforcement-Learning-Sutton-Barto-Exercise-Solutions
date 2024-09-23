### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 69fb26ed-763e-44ad-9b70-193e5a1a09b9
using PlutoDevMacros

# ╔═╡ 3f7484b3-272d-410d-92b1-ca13e5d7a8b7
# ╠═╡ show_logs = false
PlutoDevMacros.@frompackage @raw_str(joinpath(@__DIR__, "..", "NonTabularRL.jl")) begin
	using NonTabularRL
	using >.Random, >.Statistics, >.LinearAlgebra, >.Transducers
end

# ╔═╡ 9fb5dace-a799-4424-bcb3-8542e508dd4b
using PlutoUI,PlutoPlotly, PlutoProfile, BenchmarkTools, LaTeXStrings, HypertextLiteral

# ╔═╡ 35d59eae-77fd-11ef-2790-35dd5a834060
md"""
# Chapter 10: On-policy Control with Approximation
"""

# ╔═╡ 823c640d-b026-4690-a41b-3667206d23ac
md"""
In this chapter we turn to the control problem, and like before we seek to approximate the state-action value function $\hat q(s, a, \mathbf{w})$ with the goal of applying policy improvement to find $q_* (s, a)$.
"""

# ╔═╡ 6351304f-50ac-4755-86e1-cd4680f2d803
md"""
## 10.1 Episodic Semi-gradient Control

It is straightforward to extend the semi-gradient prediction methods in Chapter 9 to action values.  We simply consider examples of the form $S_t, A_t \rightarrow U_t$ where $U_t$ is any of the previously described update targets such as the Monte Carlo Return ($G_t$).  The new gradient-decent update for action-value prediction is:

$\mathbf{w}_{t+1} \doteq \alpha \left [ U_t - \hat q(S_t, A_t, \mathbf{w}_t) \right ] \nabla \hat q(S_t, A_t, \mathbf{w}_t)$

For example, the one-step Sarsa update is:

$\mathbf{w}_{t+1} \doteq \alpha \left [ R_{t+1} + \gamma \hat q(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat q(S_t, A_t, \mathbf{w}_t) \right ] \nabla \hat q(S_t, A_t, \mathbf{w}_t)$

If the action set is discrete, then at the next state $S_{t+1}$ we can compute $\hat q(S_{t+1}, a, \mathbf{w}_t)$ for every action and then find the greedy action $A^*_{t+1} = \text{argmax}_a\hat q(S_{t+1}, a, \mathbf{w}_t)$.  Policy improvement is then done by changing the estimation policy to a soft approximation of the greedy policy such as the $\epsilon$-greedy policy.  Actions are selected according to this same policy.
"""

# ╔═╡ e7bf61d7-c362-433d-9b83-6537d308c255
md"""
### *Semi-gradient Sarsa Implementation*

Below is an implementation of Semi-gradient Sarsa in a similar style to the algorithms in Chapter 9.  This function updates the provided parameters using the `update_parameters!` function and also requires an `estimate_value` function.  The linear function approximation version of this simplifies the required arguments greatly, needing only a state representation update function.  
"""

# ╔═╡ fc0b88f3-fbf9-450d-b770-b34357ffad49
compute_sarsa_value(action_values::Vector{T}, policy::Vector{T}, i_a::Integer) where T<:Real = action_values[i_a]

# ╔═╡ 991492f4-7dfc-43aa-ab6c-a6b1f3e38225
function semi_gradient_sarsa!(parameters::P, mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, estimate_value::Function, estimate_args::Tuple, update_parameters!::Function, update_args::Tuple; α = one(T)/10, ϵ = one(T) / 10, compute_value = compute_sarsa_value, kwargs...) where {P, T<:Real}
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
			estimate_value(action_values, s′, parameters, estimate_args...)
			policy .= action_values
			make_ϵ_greedy_policy!(policy; ϵ = ϵ)
			i_a′ = sample_action(policy)
			q̂ = compute_value(action_values, policy, i_a′)
			U_t = r + γ*q̂
		end
		update_parameters!(parameters, s, i_a, U_t, α, update_args...)
		s = s′
		i_a = i_a′
		step += 1
	end
	return episode_rewards, episode_steps
end

# ╔═╡ 54b92594-04b8-4a8a-82c2-773b4a24680d
md"""
### *Action-Value Implementation of Linear Approximation*

If we update the techniques from Chapter 9 to accomodate action-value estimates, then all of the linear techniques explored there can be used for these control algorihtms.  Previously, we only considered state value estimates, but we can adapt all of those techniques to action values in the linear case quite easily.  The main difference is that the action index needs to be included as an argument to the parameter update function.
"""

# ╔═╡ 278a26ac-c48f-4e18-93bb-706a4634c8c0
md"""
Two options to represent parameters for use with action-values is to simply have a unique set of parameters for each action each of which matches the size of the state representation vector.  Using this technique means we do not need to encode the action space into the representation.  The parameters could be represented by a vector of vectors or a matrix in which each column is assigned to an action.  Below are the update rules for both of these cases.
"""

# ╔═╡ 5f8b0254-88f4-4d19-ade1-8e7c40941b43
begin
	function update_parameters!(parameters::Vector{Vector{T}}, state_representation::SparseVector{T, Int64}, i_a::Integer, α::T, δ::T) where T<:Real
		x = α*δ
		for i in eachindex(state_representation.nzind)
			parameters[i_a][state_representation.nzind[i]] += x .* state_representation.nzval[i]
		end
	end

	function update_parameters!(parameters::Matrix{T}, state_representation::SparseVector{T, Int64}, i_a::Integer, α::T, δ::T) where T<:Real
		x = α*δ
		for i in eachindex(state_representation.nzind)
			parameters[state_representation.nzind[i], i_a] += x .* state_representation.nzval[i]
		end
	end

	update_parameters!(parameters::Vector{Vector{T}}, state_representation::AbstractVector{T}, i_a::Integer, α::T, δ::T) where T<:Real = (parameters[i_a] .+= α .* δ .* state_representation)

	function update_parameters!(parameters::Matrix{T}, state_representation::AbstractVector{T}, i_a::Integer, α::T, δ::T) where T<:Real 
		@inbounds @simd for i in eachindex(state_representation)
			parameters[i, i_a] += α .* δ .* state_representation[i]
		end
	end
end

# ╔═╡ c5c839f7-1806-463d-b63a-bd7e1384f203
md"""
The action value calculation also depends on how parameters are represented.  Either a dot product is used by extracting the appropriate parameter vector, or the parameter matrix needs to be iterated over the appropriate column
"""

# ╔═╡ f11787a1-57f8-4077-8d60-bc760ece7cc6
begin
	calculate_action_value(state_representation::AbstractVector{T}, i_a::Integer, parameters::Vector{Vector{T}}) where T<:Real = dot(state_representation, parameters[i_a])

	function calculate_action_value(state_representation::AbstractVector{T}, i_a::Integer, parameters::Matrix{T}) where T<:Real 
		q = zero(T)
		@inbounds @simd for i in eachindex(state_representation)
			q += state_representation[i]*parameters[i, i_a]
		end
		return q
	end

	function calculate_action_value(state_representation::SparseVector{T, Int64}, i_a::Integer, parameters::Matrix{T}) where T<:Real 
		q = zero(T)
		@inbounds @simd for i in state_representation.nzind
			q += state_representation[i]*parameters[i, i_a]
		end
		return q
	end
end	

# ╔═╡ 1478745a-634d-4f31-8a70-b74f0e536201
md"""
Part of the Sarsa algorithm requires us to identify the maximizing action.  These functions update a vector of action-values using the parameters and state representation.  By computing all of the action values at once, the maximum can be identified, and these functions compute all the action values more efficiently than using the above function for each action individually.
"""

# ╔═╡ c697e0b6-d3e4-4f5f-96e9-b9486c9d7efc
begin
	fill_action_values!(action_values::Vector{T}, state_representation::AbstractVector{T}, parameters::Matrix{T}) where T<:Real = mul!(action_values, parameters', state_representation)

	function fill_action_values!(action_values::Vector{T}, state_representation::AbstractVector{T}, parameters::Vector{Vector{T}}) where T<:Real 
		@inbounds for i in eachindex(action_values)
			action_values[i] = dot(state_representation, parameters[i])
		end
	end
end

# ╔═╡ dc2cffeb-9adf-4956-afa3-ac82af377c59
md"""
Finally we can create the action-value function and parameter update for the generic linear case.  If a vector of action values is provided as the first argument to the value function, that vector will be updated with all of the action values for a given state.
"""

# ╔═╡ a22e5d34-4b8d-479c-985c-d6abd41a6c80
md"""
### Example 10.1: Mountain Car Task
"""

# ╔═╡ cafb20b4-a2bd-46a9-9660-b0ace84d6e4c
function initialize_car_state()
	a = rand(Float32) * 0.2f0
	x = a - 0.6f0
	ẋ = 0f0
	(x, ẋ)
end

# ╔═╡ d577b393-4b40-4c90-9993-4ffbcbd9df6d
const mountain_car_actions = [-1f0, 0f0, 1f0]

# ╔═╡ b07460f1-0461-4f63-b145-c4e1818a497e
function mountain_car_step(s::Tuple{Float32, Float32}, i_a::Int64)
	a = mountain_car_actions[i_a]
	ẋ′ = clamp(s[2] + 0.001f0*a - 0.0025f0*cos(3*s[1]), -0.07f0, 0.07f0)
	x′ = clamp(s[1] + ẋ′, -1.2f0, 0.5f0)
	x′ == -1.2f0 && return (-1f0, (x′, 0f0))
	return (-1f0, (x′, ẋ′))
end

# ╔═╡ ac80958a-73ec-4342-b553-b33df6612a50
const mountain_car_transition = StateMDPTransitionSampler(mountain_car_step, initialize_car_state())

# ╔═╡ 1e9c537a-a731-4b81-8f6a-cb658b52c5be
const mountain_car_mdp = StateMDP(mountain_car_actions, mountain_car_transition, initialize_car_state, s -> s[1] == 0.5f0)

# ╔═╡ cc9197e0-f5bd-4742-bea3-b54e0b8e3b93
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

# ╔═╡ d42bb733-07e2-4932-aab4-09229ff67492
show_mountaincar_trajectory(s -> 3, 200, "Mountain Car Trajectory for Acceleration only Policy")

# ╔═╡ 742100ba-c38e-4840-8988-40990039b527
setup_mountain_car_tiles(tile_size::NTuple{2, Float32}, num_tilings::Integer) = NonTabularRL.tile_coding_setup(mountain_car_mdp, (-1.2f0, 0.5f0), (-0.07f0, 0.07f0), tile_size, num_tilings, (1, 3))

# ╔═╡ 59ec5223-f23f-4f32-9e5f-8a08e450da85
md"""
## 10.2 Semi-gradient *n*-step Sarsa

We can obtain an $n$-step version of semi-gradient Sarsa by using an $n$-step return as the update target for the semi-gradient Sarsa update equation (10.1).  The $n$-step return immediately generalizes from its tabular form (7.4) to a function approximation form: 

$G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n \hat q(S_{t+n}, A_{t+n}, \mathbf{w}_{t+n-1}), \quad t+n \lt T \tag{10.4}$

with $G_{t:t+n} \doteq G_t$ if $t+n \geq T$, as usual.  The $n$-step update equation is

$\mathbf{w}_{t+n} \doteq \mathbf{w}_{t+n-1} + \alpha \left [ G_{t:t+n} - \hat q(S_t, A_t, \mathbf{w}_{t+n-1}) \right ] \nabla \hat q(S_t, A_t, \mathbf{w}_{t+n-1}), \quad 0 \leq t \lt T \tag{10.5}$

As we have seen before, performance is often best with an $n$ that is some intermediate value between the 1-step sarsa method and Monte Carlo; however, we will not create a full implementation of this algorithm here as it will be replaced by semi-gradient Sarsa($\lambda$) in Chapter 12 which is a much more efficient version of the same concept.
"""

# ╔═╡ 49249ac1-8964-4afc-89f2-3cd4d4322cc2
md"""
> ### *Exercise 10.1* 
> We have not explicitely considered or given pseudocode for any Monte Carlo methods in this chapter.  What would they be like?  Why is it reasonable not to give pseudocode for them?  How would they perform on the Mountain Car task?

Monte Carlo methods require an episode to terminate prior to updating any action value estimates.  After the final reward is retrieved then all the action value pairs visited along the trajectory can be updated and the policy can be updated prior to starting the next episode.  For tasks such as the Mountain Car task where a random policy will likely never terminate, such a method will never be able to complete a single episode worth of updates.  We saw in earlier chapters with the racetrack and gridworld examples that for some environments a bootstrap method is the only suitable one given this possibility of an episode never terminating.
"""

# ╔═╡ e1abf8c7-06b8-4cd5-b557-1d187004bdf1
md"""
> ### *Exercise 10.2* 
> Give pseudocode for semi-gradient one-step *Expected* Sarsa for control.

Use the same pseudocode given for semi-gradient one-step Sarsa but with the following change to the weight update step in the non-terminal case:

$\mathbf{w} \leftarrow \mathbf{w} + \alpha[R + \gamma \sum_a \pi(a|S^\prime)\hat q(S^\prime, a, \mathbf{w}) - \hat q(S, A, \mathbf{w}) ] \nabla \hat q(S, A, \mathbf{w})$

where $\pi$ is the currently used policy which is $\epsilon$ greedy with respect to $\hat q$.  See complete implementation below. 
"""

# ╔═╡ 98a5d65e-4253-4523-a74e-99d03be03b89
md"""
### *Semi-gradient Expected Sarsa Implementation*

Since we already update the policy and action values in the sarsa algorithm, the only difference in expected sarsa is to compute the action-value using the entire policy distribution instead of just the sampled action.  Similarly for Q-learning we would only select the maximum value.
"""

# ╔═╡ 8ed6f8fd-8574-4d5a-9964-ce8a32629c6f
compute_expected_sarsa_value(action_values::Vector{T}, policy::Vector{T}, i_a::Integer) where T<:Real = dot(action_values, policy)

# ╔═╡ 1410db13-4b73-4a87-af34-30a5232af4ba
compute_q_learning_value(action_values::Vector{T}, policy::Vector{T}, i_a::Integer) where T<:Real = maximum(action_values)

# ╔═╡ 7812e801-70fd-4331-ad0a-fad02c1a399f
semi_gradient_expected_sarsa!(args...; kwargs...) = semi_gradient_sarsa!(args...; kwargs..., compute_value=compute_expected_sarsa_value)

# ╔═╡ 5be866c3-0fb2-4d1f-9c31-b85aba332905
semi_gradient_q_learning!(args...; kwargs...) = semi_gradient_sarsa!(args...; kwargs..., compute_value=compute_q_learning_value)

# ╔═╡ f7410fe7-e3d8-4047-8fa7-f076476e9d3a
md"""
### Example: Semi-gradient Q-learning on Mountain Car Task
"""

# ╔═╡ d6ad1ff1-8fbf-4799-8b1b-ae1e3ce88c5b
md"""
> ### *Exercise 10.3* 
> Why do the results shown in Figure 10.4 have higher standard errors at large *n* than at small *n*?

At large n more of the reward function comes from the actual trajectory observed during a run.  Since random actions are taken initially there will be more spread in the observed reward estimates than with 1 step bootstrapping which is more dependent on the initialization of the action value function.  If ties are broken randomly then you would select random actions for the first n-steps of bootstrapping thus experience more spread in the early trajectories for higher n.
"""

# ╔═╡ b8c031ca-7995-4501-a1e3-df3f34e5f0da
md"""
## 10.3 Average Reward: A New Problem Setting for Continuing Tasks

We now introduce an alternative to the discount setting for solving continuing problems (MDPs without a terminal state).  The average-reward setting is more commonly used in the classical theory of dynamic programming.  The purpose of introducing the average-reward is because discounting is problematic with function approximation in a way it was not problematic for tabular problems.  

In the average-reward setting the quality of a policy $\pi$ is defined as the average rate of reward, or simply *average reward*, while following that policy, which we denote as $r(\pi)$:

$\begin{flalign}
r(\pi) &\doteq \lim_{h \rightarrow \infty} \frac{1}{h}\sum_{t=1}^h \mathbb{E}[R_t \mid S_0,A_{0:t-1} \sim \pi] \tag{10.6}\\
&= \lim_{h \rightarrow \infty} \mathbb{E} [R_t \mid S_0,A_{0:t-1} \sim \pi] \tag{10.7}\\
&= \sum_s \mu_\pi(s)\sum_a\pi(a \vert s) \sum_{s^\prime,r} p(s^\prime,r \vert s, a)r
\end{flalign}$

where the expectations are conditioned on the initial state, $S_0$, and on the subsequent actions, $A_0, A_1, \dots,A_{t-1}$, being taken according to $\pi$. The second and third equations hold if the state-state distribution $\mu_\pi(s) \doteq \lim_{t\rightarrow \infty} \Pr \{S_t = s \mid A_{0:t-1} \sim \pi \}$, exists and is independent of $S_0$, in other words, if the MDP is *ergodic*. In an ergodic MDP, the starting state and any early decision made by the agent can only have a temporary effect; in the long run the expectation of being in a state depends on the policy and the MDP transition probabilities.  Ergodicity is sufficient but not necessary to guarantee the existence of the limit in (10.6).

In this setting, we consider all policies that obtain the maximum value of $r(\pi)$ or the *reward rate* to be optimal.  Note that the steady state distribution $\mu_\pi$ is the special distribution under which, if you select actions according to $\pi$, you remain in the same distribution.  That is, for which 

$\sum_s \mu_\pi(s) \sum_a \pi(a\vert s)p(s^\prime \vert s, a) = \mu_\pi(s^\prime) \tag{10.8}$

In the average-reward setting, returns are defined in terms of differences between rewards and the average reward: 

$G_t \doteq R_{t+1} - r(\pi) + R_{t+2} - r(\pi) + R_{t+3} - r(\pi) + \cdots \tag{10.9}$

This is known as the *differential* return, and th corresponding value functions are known as *differential* value functions.  Differential value functions are defined in terms of the new return just as conventional value functions were defined in terms of the discounted return; thus we will use the same notation, $v_\pi (s) \doteq \mathbb{E}_\pi[G_t \vert S_t = s]$ and $q_\pi (s, a) \doteq \mathbb{E}_\pi[G_t \vert S_t = s, A_t = a]$ (similarly for $v_*$ and $q_*$), for differential value functions.  Differential value functions also have Bellman equations, just slightly different from those we have seen earlier.  We simply remove all $\gamma$s and replace all rewards by the difference between the reward and the true average reward:

$\begin{flalign}
&v_\pi(s) = \sum_a \pi(a\vert s) \sum_{r, s^\prime}p(s^\prime, r \vert s, a) \left [ r - r(\pi) + v_\pi(s^\prime) \right ] \\
&q_\pi(s, a) = \sum_{r, s^\prime}p(s^\prime, r \vert s, a) \left [ r - r(\pi) + \sum_{a^\prime} \pi(a^\prime \vert s^\prime) q_\pi(s^\prime, a^\prime) \right ] \\
&v_* = \max_a \sum_{r, s^\prime}p(s^\prime, r \vert s, a) \left [ r - r(\pi) + v_*(s^\prime) \right ] \\
&q_* = \sum_{r, s^\prime}p(s^\prime, r \vert s, a) \left [ r - \max_{\pi}r(\pi) + \max_a q_\pi(s^\prime, a^\prime) \right ] \\
\end{flalign}$

There is also a differential form of the two TD errors:

$\delta_t \doteq R_{t+1} - \bar{R}_t+ \hat v (S_{t+1}, \mathbf{w}_t) - \hat v(S_t, \mathbf{w}_t) \tag{10.10}$

and

$\delta_t \doteq R_{t+1} - \bar{R}_t+ \hat q (S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat q(S_t, A_t, \mathbf{w}_t) \tag{10.11}$

where $\bar{R}_t$ is an estimate at time $t$ of the average reward $r(\pi)$.  With these alternate definitions, most of our algorithms and many theoretical results carry through to the average_reward setting without any change.  

For example, an average reward version of semi-gradient Sarsa could be defined just as in (10.2) except with the differential version of the TD error.  That is by

$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t \nabla \hat q(S_t, A_t, \mathbf{w}_t)$

with $\delta_t$ given by (10.11).  See a full implementation below.  One limitation of this algorithm is that it does not converge to the differential values but to the differential values plut an arbitrary offset.  Notice that the Bellman equations and TD errors given above are unaffected if all the values are shifted by the same amount.  Thus, the offset may not matter in practice.
"""

# ╔═╡ 69a06405-57cd-42e5-96b1-5cc77d74aa03
md"""
### *Differential Semi-gradient Sarsa Implementation*
"""

# ╔═╡ a9fdb1fd-3f62-4e1c-9157-c4eee6215261
function differential_semi_gradient_sarsa!(parameters, mdp::StateMDP{T, S, A, P, F1, F2, F3}, max_episodes::Integer, max_steps::Integer, estimate_value::Function, estimate_args::Tuple, update_parameters!::Function, update_args::Tuple; α = one(T)/10, β = one(T)/100, ϵ = one(T) / 10, compute_value = compute_sarsa_value, kwargs...) where {T<:Real, S, A, P, F1, F2, F3}
	s = mdp.initialize_state()
	i_a = rand(eachindex(mdp.actions))
	ep = 1
	step = 1
	R̄ = zero(T)
	ō = zero(T)
	epreward = zero(T)
	episode_rewards = zeros(T, max_episodes)
	episode_steps = zeros(Int64, max_episodes)
	action_values = zeros(T, length(mdp.actions))
	average_step_reward = Vector{T}()
	policy = zeros(T, length(mdp.actions))
	while (ep <= max_episodes) && (step <= max_steps)
		(r, s′) = mdp.ptf(s, i_a)
		estimate_value(action_values, s, parameters, estimate_args...)
		q̂ = action_values[i_a]
		U_t = r - R̄
		epreward += r
		if mdp.isterm(s′)
			s′ = mdp.initialize_state()
			i_a′ = rand(eachindex(mdp.actions))
			episode_rewards[ep] = epreward
			episode_steps[ep] = step
			epreward = zero(T)
			ep += 1
		else
			estimate_value(action_values, s′, parameters, estimate_args...)
			policy .= action_values
			make_ϵ_greedy_policy!(policy; ϵ = ϵ)
			i_a′ = sample_action(policy)
			q̂′ = compute_value(action_values, policy, i_a′)
			U_t += q̂′
		end
		δ = U_t - q̂
		ō += β * (one(T) - ō)
		R̄ += (β/ō)*δ
		push!(average_step_reward, R̄)
		update_parameters!(parameters, s, i_a, U_t, α, update_args...)
		s = s′
		i_a = i_a′
		step += 1
	end
	return episode_rewards, episode_steps, average_step_reward
end

# ╔═╡ 565c53ee-7ad5-44e2-bce5-4ff1f5f162c0
differential_semi_gradient_expected_sarsa!(args...; kwargs...) = differential_semi_gradient_sarsa!(args...; kwargs..., compute_value=compute_expected_sarsa_value)

# ╔═╡ c9759bd9-ec9b-47a1-9080-a7fc332be565
differential_semi_gradient_q_learning!(args...; kwargs...) = differential_semi_gradient_sarsa!(args...; kwargs..., compute_value=compute_q_learning_value)

# ╔═╡ 1a7ba296-52ca-4069-85fa-792d08d77b0e
md"""
### Example: Differential Sarsa and Q-learning with Mountain Car Task

In order to apply differential learning to the mountain car task, we need to change the rewards per step.  Previously, the rewards were assigned in a manner appropriate for learning with a discount rate of 1.  The reward of -1 per episode step ensures that policies that finish the task faster have a higher reward.  In the average reward setting, every policy would have an average reward per step of -1 making the task ill posed.  Instead, we can assign a reward of 1 for finishing to the right and 0 at all other steps.  These rewards would produce an ill posed task for $\gamma = 1$ but are perfectly fine for the average reward setting.  Now our learning procedure should find a policy that produces the highest average reward $\frac{1}{\text{num steps}}$ which is maximized when the number of steps to finish an episode is minimized.
"""

# ╔═╡ eb28458f-b222-4f8e-9a5b-8203d3997f7b
function mountain_car_differential_step(s::Tuple{Float32, Float32}, i_a::Int64)
	a = mountain_car_actions[i_a]
	ẋ′ = clamp(s[2] + 0.001f0*a - 0.0025f0*cos(3*s[1]), -0.07f0, 0.07f0)
	x′ = clamp(s[1] + ẋ′, -1.2f0, 0.5f0)
	x′ == -1.2f0 && return (0f0, (x′, 0f0))
	s′ = (x′, ẋ′)
	r = Float32(x′ == 0.5f0)
	return (r, s′)
end

# ╔═╡ e5ad765a-341f-4f11-9ae8-37d81cb349d2
const mountain_car_differential_transition = StateMDPTransitionSampler(mountain_car_differential_step, initialize_car_state())

# ╔═╡ bc1d7cce-c0f4-47a8-b674-8acb82491c7f
const mountain_car_differential_mdp = StateMDP(mountain_car_actions, mountain_car_differential_transition, initialize_car_state, s -> s[1] == 0.5f0)

# ╔═╡ 9df1a18d-137c-4ea5-8d15-05697f7bbf07
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

See implementation above
"""

# ╔═╡ 0c7f5742-6c51-4c6a-b67f-217163935ba5
md"""
> ### *Exercise 10.5* 
> What equations are needed (beyond 10.10) to specify the differential version of TD(0)?

10.10 includes a reward estimate at time t, $\bar R_t$, which also needs to be updated.  The TD error represents the newly observed reward the was experienced in excess of the estimated average so the update equation should move $\bar R$ in the direction of the TD error.  After each step, the following updates should occur.

$\begin{flalign}
\delta &\leftarrow R - \bar R + \hat v(S^\prime, \mathbf{w}) - \hat v(S, \mathbf{w}) \\
\bar R &\leftarrow \bar R + \beta \delta \\
\mathbf{w} &\leftarrow \mathbf{w} + \alpha \delta \nabla \hat v(S, \mathbf{w}) \\
S & \leftarrow S^\prime \\
\end{flalign}$
"""

# ╔═╡ a6c5ec28-b2d5-4893-a118-95c1318d1f7f
md"""
> ### *Exercise 10.6* 
> Suppose there is an MDP that under any policy produces the deterministic sequence of rewards +1, 0, +1, 0, +1, 0, . . . going on forever. Technically, this violates ergodicity; there is no stationary limiting distribution $μ_\pi$ and the limit (10.7) does not exist. Nevertheless, the average reward (10.6) is well defined. What is it? Now consider two states in this MDP. From A, the reward sequence is exactly as described above, starting with a +1, whereas, from B, the reward sequence starts with a 0 and then continues with +1, 0, +1, 0, . . .. We would like to compute the differential values of A and B. Unfortunately, the differential return (10.9) is not well defined when starting from these states as the implicit limit does not exist. To repair this, one could alternatively define the differential value of a state as $v_\pi (s) \doteq \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} \sum_{t=0}^h \gamma^t \left ( \mathbb{E_\pi} [R_{t+1}|S_0=s]-r(\pi)  \right )$.  Under this definition what are the differential values of states A and B?
"""

# ╔═╡ 44d43dbf-fe32-438e-b89d-c677bbc35893
md"""
In order to use (10.6): $r(\pi) \doteq \lim_{h \rightarrow \infty} \frac{1}{h} \sum_{t = 1}^h \mathbb{E} [R_t \mid S_0, A_{0:t-1} \sim \pi]$ we need to compute $\mathbb{E} [R_t \mid S_0, A_{0:t-1} \sim \pi]$.  In this case, we are told that regardless of the policy, the reward sequence will be +1, 0, +1, 0, ....  We can therefore replace the expected values in the equation with this sequence since the rewards at each time step are known with 100% probability.

the average reward can be computed as $r(\pi) = \lim_{h \rightarrow \infty} \frac{1}{2h}\sum_{t=1}^h (-1)^{t+1} + 1 = \lim_{h \rightarrow \infty} \frac{h}{2h} + \frac{1}{2h} \sum_{t=1}^h (-1)^{t+1}$

The sum left in the expression is $1 - 1 + 1 - 1 \cdots$ which is 1 for even h and 0 for odd h.  Either way, when divided by $2h$ that term goes to 0 leaving only the term $\frac{h}{2h} = \frac{1}{2}$ so $r(\pi) = \frac{1}{2}$

To compute the differential value function for state A and B, consider the alternative definition above using the fact that $r(\pi) = 0.5$.  

For state A, each parenthetical term in the sum will be: $1 - 0.5, 0 - 0.5, 1 - 0.5, 0 - 0.5, \dots = 0.5, -0.5, 0.5, -0.5, \dots$

For state B, each parenthetical term in the sum will be: $0 - 0.5, 1 - 0.5, 0 - 0.5, 1 - 0.5, \dots = -0.5, 0.5, -0.5, 0.5, \dots$

$\begin{flalign}
v_\pi (A) &= \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} 0.5 - 0.5\gamma + 0.5 \gamma^2 - 0.5\gamma^3 + \cdots \\
&=0.5\lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty}\sum_{t=0}^h (-\gamma)^t \\
&=0.5\lim_{\gamma \rightarrow 1}\frac{1}{\gamma +1 } = 0.25
\end{flalign}$

$\begin{flalign}
v_\pi (B) &= \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} -0.5 + 0.5\gamma - 0.5 \gamma^2 + 0.5\gamma^3 + \cdots \\
&=-0.5\lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty}\sum_{t=0}^h (-\gamma)^t \\
&=-0.5\lim_{\gamma \rightarrow 1}\frac{1}{\gamma +1 } = -0.25
\end{flalign}$
"""

# ╔═╡ f1edb500-fbd1-4c03-b033-53860dfa452d
md"""
> ### *Exercise 10.7* 
> Consider a Markov reward process consisting of a ring of three states A, B, and C, with state transitions going deterministically around the ring.  A reward of +1 is received upon arrival in A and otherwise the reward is 0.  What are the differential values of the three states, using (10.13)

From 10.13 we have 

$v_\pi (s) \doteq \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} \sum_{t=0}^h \gamma^t \left ( \mathbb{E_\pi} [R_{t+1}|S_0=s]-r(\pi)  \right )$

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

$\begin{flalign}
\lim_{\gamma \rightarrow 1} v(B) &= \frac{1}{3} \frac{1 - 1}{3} = 0 \\
\lim_{\gamma \rightarrow 1} v(A) &= \gamma \times 0 - \frac{1}{3} = -\frac{1}{3} \\
$\lim_{\gamma \rightarrow 1} v(C) &=  \frac{0 + \frac{1}{3}}{\gamma} = \frac{1}{3}
\end{flalign}$
"""

# ╔═╡ 2d7679ad-a9b3-448b-a4bc-7e5b9bce6adb
md"""
> ### *Exercise 10.8* 
> The pseudocode in the box on page 251 updates $\bar R_t$ using $\delta_t$ as an error rather than simply $R_{t+1} - \bar R_t$.  Both errors work, but using $\delta_t$ is better.  To see why, consider the ring MRP of three states from Exercise 10.7.  The estimate of the average reward should tend towards its true value of $\frac{1}{3}$.  Suppose it was already there and was held stuck there.  What would the sequence of $R_{t+1} - \bar R_t$ errors be?  What would the sequence of $\delta_t$ errors be (using Equation 10.10)?  Which error sequence would produce a more stable estimate of the average reward if the estimate were allowed to change in response to the errors? Why?

The sequence of $R_{t+1} - \bar R_t$ would be given by the cyclical sequence of rewards.  Let's assume we start the sequence at state A.  Then our reward sequence will be 0, 0, 1, 0, 0, 1... so the error sequence will be $-\frac{1}{3}$, $-\frac{1}{3}$, $\frac{2}{3}$,...  If we update the average error estimate using these corrections it would remain centered at the correct value but fluctuate up and down with each correction.

In order to calculate $\delta_t$ we must use the definition given by 10.10:

$\delta_t = R_{t+1} - \bar R_t + \hat v(S_{t+1}, \mathbf{w}_t) - \hat v(S_t, \mathbf{w}_t)$

This equation requires us to have value estimates for each state which we can assume have converged to the true values as we have for the average reward estimate: $\hat v(A) = -\frac{1}{3}$, $\hat v(B) = 0$, and $\hat v(C) = \frac{1}{3}$.  Starting at state A, $\delta_t = 0 - \frac{1}{3} + 0 - -\frac{1}{3} = 0$.  For the following state we have $0 - \frac{1}{3} + \frac{1}{3} = 0$.  Finally we have $1 - \frac{1}{3} + -\frac{1}{3} - \frac{1}{3} = 0$.  So if we use the TD error to update our average reward estimate, at equilibrium all the values will remain unchanged.

"""

# ╔═╡ a9b74949-9392-4048-bcb6-5fd48c1d9b98
md"""
### Example 10.2: An Access-Control Queuing Task
"""

# ╔═╡ fbf1c64f-1979-4384-a8c6-dc7875174d1f
begin
	abstract type AccessControlAction end
	struct Accept <: AccessControlAction end
	struct Reject <: AccessControlAction end
end

# ╔═╡ e7372e2b-a2db-4a93-9efc-f75aa74c197b
struct AccessControlState
	num_free_servers::Int64
	top_priority::Float32
end

# ╔═╡ 014339eb-5b23-4ac5-a551-8eeb2238366f
begin 
	function access_control_step(s::AccessControlState, ::Reject, num_servers::Integer, priority_payments::Vector{Float32})
		occupied_servers = num_servers - s.num_free_servers
		freed_servers = sum(_ -> Float32(rand() < 0.06), 1:occupied_servers; init = 0f0)
		new_occupied_servers = occupied_servers - freed_servers
		new_free_servers = num_servers - new_occupied_servers
		new_priority = rand(priority_payments)
		(0f0, AccessControlState(new_free_servers, new_priority))
	end

	function access_control_step(s::AccessControlState, ::Accept, num_servers::Integer, priority_payments::Vector{Float32})
		occupied_servers = num_servers - s.num_free_servers
		(r_reject, s′) = access_control_step(s, Reject(), num_servers, priority_payments)
		s.num_free_servers == 0 && return (r_reject, s′)
		(s.top_priority, AccessControlState(s′.num_free_servers - 1, s′.top_priority))
	end
end

# ╔═╡ 96548352-cd4d-4448-8312-ed10057f4359
begin
	function update_parameters!(parameters::Vector{Vector{T}}, i_s::Integer, i_a::Integer, g::T, α::T) where {T<:Real}
		q̂ = parameters[i_a][i_s]
		δ = (g - q̂)
		parameters[i_a][i_s] += α*δ
		return nothing
	end
	function update_parameters!(parameters::Matrix{T}, i_s::Integer, i_a::Integer, g::T, α::T) where {T<:Real}
		q̂ = parameters[i_s, i_a]
		δ = (g - q̂)
		parameters[i_s, i_a] += α*δ
		return nothing
	end
end

# ╔═╡ b9ebd6bb-90a1-4945-85ed-023206e2420a
function linear_features_action_gradient_setup(problem::Union{StateMDP{T, S, A, P, F1, F2, F3}, StateMRP{T, S, P, F1, F2}}, state_representation::AbstractVector{T}, update_feature_vector!::Function) where {T<:Real, S, A, P, F1<:Function, F2<:Function, F3<:Function}
	s0 = problem.initialize_state()
	update_feature_vector!(state_representation, s0) #verify that feature vector update is compatible with provided state representation

	function update_params!(parameters::AbstractArray, s::S, i_a::Integer, g::T, α::T, state_representation::AbstractVector{T}) where T<:Real
		update_feature_vector!(state_representation, s)
		q̂ = calculate_action_value(state_representation, i_a, parameters)
		δ = (g - q̂)
		iszero(δ) && return nothing
		update_parameters!(parameters, state_representation, i_a, α, δ)
		return nothing
	end
	
	function q̂(s::S, i_a::Integer, parameters::AbstractArray, state_representation::AbstractVector{T}) where {T<:Real} 
		update_feature_vector!(state_representation, s)
		calculate_action_value(state_representation, i_a, parameters)
	end

	function q̂(action_values::Vector{T}, s::S, parameters::AbstractArray, state_representation::AbstractVector{T}) where {T<:Real} 
		update_feature_vector!(state_representation, s)
		fill_action_values!(action_values, state_representation, parameters)
		return action_values
	end
	
	return (value_function = q̂, value_args = (state_representation,), parameter_update = update_params!, update_args = (copy(state_representation),))
end

# ╔═╡ 2c620fe4-2f62-40f8-a666-8dced1e0b84a
function run_linear_semi_gradient_sarsa(mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, state_representation::AbstractVector{T}, update_state_representation!::Function; setup_kwargs = NamedTuple(), kwargs...) where T<:Real
	setup = linear_features_action_gradient_setup(mdp, state_representation, update_state_representation!; setup_kwargs...)
	l = length(state_representation)
	num_actions = length(mdp.actions)
	# parameters = zeros(T, l, num_actions)
	parameters = [zeros(T, l) for _ in 1:num_actions]
	episode_rewards, episode_steps = semi_gradient_sarsa!(parameters, mdp, γ, max_episodes, max_steps, setup.value_function, setup.value_args, setup.parameter_update, setup.update_args; kwargs...)
	q̂(s, i_a) = setup.value_function(s, i_a, parameters, setup.value_args...)
	function q̂(s)
		action_values = zeros(T, num_actions)
		setup.value_function(action_values, s, parameters, setup.value_args...)
		findmax(action_values)
	end
	return (value_function = q̂, reward_history = episode_rewards, step_history = episode_steps)
end

# ╔═╡ 7c5fb569-81f0-4b70-ae95-1fce0c51b6f4
function mountaincar_test(max_episodes::Integer, α::Float32, ϵ::Float32; num_tiles = 12, num_tilings = 8, kwargs...)
	setup = setup_mountain_car_tiles((1f0/num_tiles, 1f0/num_tiles), num_tilings)
	v = setup.args.feature_vector
	run_linear_semi_gradient_sarsa(mountain_car_mdp, 1f0, max_episodes, typemax(Int64), zeros(Float32, length(v)), setup.args.feature_vector_update; α = α, ϵ = ϵ, kwargs...)
end

# ╔═╡ 30ab21ba-3f5b-46a8-8b8c-753f2755d419
(q̂_mountain_car, episode_rewards, episode_steps) = mountaincar_test(5000, 0.0008f0/8, 0.01f0)

# ╔═╡ 5d23a0ba-5882-4ef6-ad56-596a3d66d3e8
π_mountain_car(s) = argmax(i_a -> q̂_mountain_car(s, i_a), eachindex(mountain_car_actions))

# ╔═╡ f2201afe-8952-4dde-9e39-02beeb920f6f
show_mountaincar_trajectory(π_mountain_car, 10_000, "Sarsa Learned Policy")

# ╔═╡ c1388562-0708-4a6a-acfe-927413dab5d2
plot(scatter(y = -episode_rewards), Layout(yaxis_type = "log"))

# ╔═╡ 5db29488-a150-42ee-aedb-380a3a4fd548
function plot_mountaincar_action_values()
	n = 100
	xvals = LinRange(-1.2f0, 0.5f0, n)
	vvals = LinRange(-0.07f0, 0.07f0, n)
	values = zeros(Float32, n, n)
	actions = zeros(Float32, n, n)
	for (i, x) in enumerate(xvals)
		for (j, v) in enumerate(vvals)
			(q̂, i_a) = q̂_mountain_car((x, v))
			values[i, j] = q̂
			actions[i, j] = mountain_car_actions[i_a]
		end
	end
	p1 = plot(heatmap(x = xvals, y = vvals, z = values), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Learned Value Function"))
	p2 = plot(heatmap(x = xvals, y = vvals, z = actions, colorscale = "rb", showscale = false), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Policy (blue = accelerate left, <br>red = accelerate right, gray = no acceleration)"))
	[p1 p2]
end

# ╔═╡ 4afbb723-340b-4d85-9115-027a0ff8dfad
plot_mountaincar_action_values()

# ╔═╡ 1a82ae95-3c3e-4281-bc1d-9eb19bf50286
function figure_10_2(;α_list = [0.1f0, 0.2f0, 0.5f0], num_episodes = 50, ϵ = 0.05f0)
	traces = map(α_list) do α
		scatter(y = 1:100 |> Map(_ -> mountaincar_test(num_episodes, α/8, ϵ; num_tiles = 12, num_tilings = 8).reward_history) |> foldxt((a, b) -> a .+ b) |> v -> -v ./ 100, name = "α = $α/8")
	end
	plot(traces, Layout(xaxis_title = "Episode", yaxis_title = "Steps per episode<br>averaged over 100 runs", yaxis_type = "log"))
end

# ╔═╡ ddcb50be-5287-47f8-89f9-58c026a6b151
figure_10_2()

# ╔═╡ cbac1927-b087-4c4c-98ae-6aa5f0b824ad
(q̂_mountain_car_q, episode_rewards_q, episode_steps_q) = mountaincar_test(5000, 0.0002f0/8, 0.05f0; compute_value = compute_q_learning_value)

# ╔═╡ 5515db1c-b3d1-4af5-8613-030a4b0faf09
π_mountain_car_q(s) = argmax(i_a -> q̂_mountain_car_q(s, i_a), eachindex(mountain_car_actions))

# ╔═╡ b5409b69-a254-4355-b2b9-99394eceb2f7
show_mountaincar_trajectory(π_mountain_car_q, 10_000, "Q-Learning Learned Policy")

# ╔═╡ 065b2626-01f1-443f-8be4-3036003a2772
function run_linear_differential_semi_gradient_sarsa(mdp::StateMDP, max_episodes::Integer, max_steps::Integer, state_representation::AbstractVector{T}, update_state_representation!::Function; setup_kwargs = NamedTuple(), kwargs...) where T<:Real
	setup = linear_features_action_gradient_setup(mdp, state_representation, update_state_representation!; setup_kwargs...)
	l = length(state_representation)
	num_actions = length(mdp.actions)
	# parameters = zeros(T, l, num_actions)
	parameters = [zeros(T, l) for _ in 1:num_actions]
	episode_rewards, episode_steps, average_step_reward = differential_semi_gradient_sarsa!(parameters, mdp, max_episodes, max_steps, setup.value_function, setup.value_args, setup.parameter_update, setup.update_args; kwargs...)
	q̂(s, i_a) = setup.value_function(s, i_a, parameters, setup.value_args...)
	function q̂(s)
		action_values = zeros(T, num_actions)
		setup.value_function(action_values, s, parameters, setup.value_args...)
		findmax(action_values)
	end
	return (value_function = q̂, reward_history = episode_rewards, step_history = episode_steps, average_step_reward = average_step_reward)
end

# ╔═╡ 49e43d51-05d6-415b-a685-76e50904c5bc
function mountaincar_differential_test(max_episodes::Integer, α::Float32, β::Float32, ϵ::Float32; num_tiles = 12, num_tilings = 8, kwargs...)
	setup = setup_mountain_car_tiles((1f0/num_tiles, 1f0/num_tiles), num_tilings)
	v = setup.args.feature_vector
	run_linear_differential_semi_gradient_sarsa(mountain_car_differential_mdp, max_episodes, typemax(Int64), zeros(Float32, length(v)), setup.args.feature_vector_update; α = α, β = β, ϵ = ϵ, kwargs...)
end

# ╔═╡ db189316-e880-4cc8-9070-ccfe2b4fc545
(q̂_mountain_car2, episode_rewards2, episode_steps2, average_step_reward) = mountaincar_differential_test(100, 0.01f0/8, 0.01f0, 0.3f0; compute_value = compute_q_learning_value)

# ╔═╡ 7bc49107-9de5-4985-8750-979f36b3aa81
π_mountain_car2(s) = argmax(i_a -> q̂_mountain_car2(s, i_a), eachindex(mountain_car_actions))

# ╔═╡ ab4cb3db-3a2d-4145-826b-b1001114eeff
show_mountaincar_trajectory(π_mountain_car2, 1_000, "Differential Q-learning Learned Policy")

# ╔═╡ 4775126e-4374-49be-b25f-4597401f3642
plot(scatter(y = inv.(average_step_reward)), Layout(yaxis_type = "log"))

# ╔═╡ 4271151d-d5a6-4a29-96c3-f2102b142b95
begin
	get_action_value(i_s::Integer, i_a::Integer, parameters::Vector{Vector{T}}) where T<:Real = parameters[i_a][i_s]
	get_action_value(i_s::Integer, i_a::Integer, parameters::Matrix{T}) where T<:Real = parameters[i_s, i_a]
end

# ╔═╡ 5534526a-d790-4379-98b0-8e4ee981fd9f
begin
	function update_action_values!(action_values::Vector{T}, i_s::Integer, parameters::Vector{Vector{T}}) where T<:Real
		maxvalue = typemin(T)
		maxindex = 1
		@inbounds @simd for i_a in eachindex(action_values)
			q = parameters[i_a][i_s]
			action_values[i_a] = q
			newmax = q > maxvalue
			maxvalue = newmax*q + !newmax*maxvalue
			maxindex = newmax*i_a + !newmax*maxindex
		end
		return maxvalue, maxindex
	end
	function update_action_values!(action_values::Vector{T}, i_s::Integer, parameters::Matrix{T}) where T<:Real
		maxvalue = typemin(T)
		maxindex = 1
		@inbounds @simd for i_a in eachindex(action_values)
			q = parameters[i_s, i_a]
			action_values[i_a] = q
			newmax = q > maxvalue
			maxvalue = newmax*q + !newmax*maxvalue
			maxindex = newmax*i_a + !newmax*maxindex
		end
		return maxvalue, maxindex
	end
end

# ╔═╡ f68952bd-4c0b-4331-a09c-5b118c8fa5a9
function state_aggregation_action_gradient_setup(assign_state_group::Function)
	function update_params!(parameters, s, i_a, g::T, α::T) where {T<:Real}
		i_s = assign_state_group(s)
		update_parameters!(parameters, i_s, i_a, g, α)
	end

	q̂(s, i_a, parameters) = get_action_value(assign_state_group(s), i_a, parameters)
	q̂(action_values::Vector{T}, s, parameters) where T<:Real = update_action_values!(action_values, assign_state_group(s), parameters)
		
	return (value_function = q̂, value_args = (), parameter_update = update_params!, update_args = ())
end

# ╔═╡ 62839b2a-398a-4445-87d1-b15ff2acc1d1
function create_access_control_task(num_servers::Integer, priority_payments::Vector{Float32})
	actions = [Accept(), Reject()]

	initialize_state() = AccessControlState(num_servers, rand(priority_payments))

	transition = StateMDPTransitionSampler((s, i_a) -> access_control_step(s, actions[i_a], num_servers, priority_payments), initialize_state())
	mdp = StateMDP(actions, transition, initialize_state, s -> false)
	states =  [AccessControlState(n, p) for n in 0:num_servers for p in priority_payments]
	assign_group(s::AccessControlState) = s.num_free_servers + 1 + (num_servers+1)*Int64(log2(s.top_priority))
	(mdp = mdp, gradient_setup = state_aggregation_action_gradient_setup(assign_group), num_groups = (num_servers+1) * length(priority_payments))
end

# ╔═╡ b4af8d87-a6e5-4e09-92b4-b07757f58f7f
function run_access_control_differential_sarsa(max_steps::Int64; num_servers = 10, priority_payments = [1f0, 2f0, 4f0, 8f0], kwargs...)
	(mdp, gradient_setup, num_groups) = create_access_control_task(num_servers, priority_payments)
	parameters = [zeros(Float32, num_groups) for _ in eachindex(mdp.actions)]
	state_representation = zeros(Float32, num_groups)
	(_, _, steprewards) = differential_semi_gradient_sarsa!(parameters, mdp, 1, max_steps, gradient_setup...; kwargs...)
	action_values = zeros(Float32, length(mdp.actions))
	v̂(num_free_servers::Int64, priority::Real) = gradient_setup.value_function(action_values, AccessControlState(num_free_servers, Float32(priority)), parameters)

	(value_function = v̂, mdp = mdp, parameters = parameters, steprewards = steprewards)
end

# ╔═╡ 546a775e-d3c9-4693-9f64-d4c47a84fb9f
function figure_10_5(;numsteps = 2_000_000, α = 0.01f0, β = 0.01f0, ϵ = 0.1f0)
	access_control_output = run_access_control_differential_sarsa(numsteps; β = β, α = α, ϵ = ϵ)
	policy_output = BitArray(undef, (4, 10))
	priorities = [8, 4, 2, 1]
	actions = [true, false]
	value_function_outputs = [zeros(Float32, 11) for _ in 1:4]
	for num_free_servers in 0:10
		for priority in 1:4
			v, i_a = access_control_output.value_function(num_free_servers, priorities[priority])
			value_function_outputs[priority][num_free_servers+1] = v
			if num_free_servers > 0
				policy_output[priority, num_free_servers] = actions[i_a]
			end
		end
	end
	policy_trace = heatmap(x = 1:10, y = 1:4, z = Float32.(policy_output), colorscale="Greys", showscale = false)
	value_traces = [scatter(x = 0:10, y = value_function_outputs[i], name = "priority $(priorities[i])") for i in 1:4]
	p1 = plot(policy_trace, Layout(yaxis_tickvals = 1:4, yaxis_ticktext = priorities, xaxis_ticktext = 1:10, xaxis_tickvals = 1:10, xaxis_title = "Number of free servers", yaxis_title = "Priority", title = "Policy (black=reject, white=accept)"))
	p2 = plot(value_traces, Layout(xaxis_title = "Number of free servers", yaxis_title = "Differential value of best action", title = "Value Function"))

	md"""
	Figure 10.5

	The policy and value function found by differential semi-gradient one-step Sarsa on the access-control queuing task after 2 million steps.  The value learned for $\bar R$ was about $(access_control_output.steprewards[end-10000:end] |> mean |> Float64 |> x -> round(x, sigdigits = 3))
	$([p1 p2])
	"""
end

# ╔═╡ 41c626c7-908d-4ff6-9730-4ad0b8c3cc25
figure_10_5()

# ╔═╡ 662759be-282c-460b-adc3-8595475b53c2
md"""
## 10.4 Deprecating the Discounted Setting

In a special case of indistinguishable states, we can only use the actions and reward sequences to analyze a continuing task.  For a policy $\pi$, the average of the discounted returns with discount factor $\gamma$ is always $\frac{r(\pi)}{1-\gamma}$.  Therefore the *ordering* of all policies is independent of the discount rate and would match the ordering we get in the average reward setting.  This derivation however depends on states being indistinguishable allowing us to match up the weights on reward sequences from different policies.

We can use discounting in approximate solution methods regardless but then $\gamma$ changes from a problem parameter to a solution method parameter.  Unfortunately, discounting algorithms with function approximation do not optimize discounted value over the on-policy distribution, and thus are not guaranteed to optimze average reward.

The root cause of the problem applying discounting with function approximation is that we have lost the policy improvement theorem which states that a policy $\pi^\prime$ is better than policy $\pi$ if $v_{\pi^\prime}(s) \geq v_\pi(s) \forall s\in \mathcal{S}$.  Under this theorem we could take a deterministic policy, choose a specific state, and find a new action at that state with a higher expected reward than the current policy.  If the policy is an approximation function that uses states represented by feature vectors, then adjusting the parameters can in general affect the actions at many states including ones that have not been encountered yet.  In fact, with approximate solution methods we cannot guarantee  policy improvement in any setting.  Later we will introduce a theoretical guarantee called the "policy-gradient theorem" but for an alternative class of algorithms based on parametrized policies.
"""

# ╔═╡ 0e66a941-1ec1-4d3b-b064-e5f25cc93baf
md"""
### Connection to Chapter 3
"""

# ╔═╡ c316c5d3-f484-4e8e-bd56-be1e236d96bc
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

# ╔═╡ bc220d14-97fd-486d-9880-6908135fe036
md"""
The reason why the left policy can be better if $\gamma < 0.5$ in the original example is because it has a higher value in each state considered.  Consider $\gamma = 0.25$.  The left policy has the following approximate discounted value estimates for top, left, right: 

1.0667, 0.2667, 2.2667. 

Meanwhile the right policy has the corresponding values of: 

0.533, 0.133, 2.133.

Each value is smaller for the right policy.  However when we calculate the average value calculated over the long term distribution of states, the left policy averages the first two values while the right policy averages the first and third values because in the long run we expect the left policy to only exist in the top and left state while the right policy will exist in the top and right state.  Because the right state has such a high value for both policies but only the right policy includes it in the average it makes its entire objective estimate higher.  However, we can see that in the event of being in the right state, it is still a higher value expectation following the left policy in this case.  The decision to average based on the final distribution results in a policy ordering that doesn't match with what we know to be the optimal policy from the policy improvement theorem over finite states.
"""

# ╔═╡ 39eada35-8c3e-4ddc-8df9-7cf9f120928d
md"""
## 10.5 Differential Semi-gradient *n*-step Sarsa
"""

# ╔═╡ 8752c98d-fac1-4b3b-b20b-70acc0677fcb
md"""
> ### *Exercise 10.9* 
> In the differential semi-gradient n-step Sarsa algorithm, the step-size parameter on the average reward, $\beta$, needs to be quite small so that $\bar R$ becomes a good long-term estimate of the average reward. Unfortunately, $\bar R$ will then be biased by its initial value for many steps, which may make learning inefficient. Alternatively, one could use a sample average of the observed rewards for $\bar R$. That would initially adapt rapidly but in the long run would also adapt slowly. As the policy slowly changed, $\bar R$ would also change; the potential for such long-term nonstationarity makes sample-average methods ill-suited. In fact, the step-size parameter on the average reward is a perfect place to use the unbiased constant-step-size trick from Exercise 2.7. Describe the specific changes needed to the boxed algorithm for differential semi-gradient n-step Sarsa to use this trick.

At the start initialize $\bar o = 0$ and select $\lambda > 0$ small instead of $\beta$. 

Within the loop under the $\tau \geq 0$ line, add two lines; one to update $\bar o$ and one to calculate the update rate for the average reward: 

Line 1: $\bar o \leftarrow \bar o + \lambda (1 - \bar o)$

Line 2: $\beta = \lambda / \bar o$

As steps progress $\beta$ will approach $\lambda$ but early on will take on much larger values as $\bar o$ starts close to 0 and approaches 1.
"""

# ╔═╡ 6cea9e69-bf8c-4079-9884-663a728d7b08
md"""
# Dependencies
"""

# ╔═╡ f5e32900-6eb6-4b61-916d-893c0bcaf214
TableOfContents()

# ╔═╡ ed1bd92c-8cc7-457f-9692-a10a9487c953
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
PlutoUI = "~0.7.60"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.5"
manifest_format = "2.0"
project_hash = "be9838d8582d0b0ad1f9af1a75a9e540a766619c"

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
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.FlameGraphs]]
deps = ["AbstractTrees", "Colors", "FileIO", "FixedPointNumbers", "IndirectArrays", "LeftChildRightSiblingTrees", "Profile"]
git-tree-sha1 = "d9eee53657f6a13ee51120337f98684c9c702264"
uuid = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"
version = "0.2.10"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "2984284a8abcfcc4784d95a9e2ea4e352dd8ede7"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.36"

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
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

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
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

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
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

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
# ╟─35d59eae-77fd-11ef-2790-35dd5a834060
# ╟─823c640d-b026-4690-a41b-3667206d23ac
# ╟─6351304f-50ac-4755-86e1-cd4680f2d803
# ╟─e7bf61d7-c362-433d-9b83-6537d308c255
# ╠═fc0b88f3-fbf9-450d-b770-b34357ffad49
# ╠═991492f4-7dfc-43aa-ab6c-a6b1f3e38225
# ╟─54b92594-04b8-4a8a-82c2-773b4a24680d
# ╟─278a26ac-c48f-4e18-93bb-706a4634c8c0
# ╠═5f8b0254-88f4-4d19-ade1-8e7c40941b43
# ╟─c5c839f7-1806-463d-b63a-bd7e1384f203
# ╠═f11787a1-57f8-4077-8d60-bc760ece7cc6
# ╟─1478745a-634d-4f31-8a70-b74f0e536201
# ╠═c697e0b6-d3e4-4f5f-96e9-b9486c9d7efc
# ╟─dc2cffeb-9adf-4956-afa3-ac82af377c59
# ╠═b9ebd6bb-90a1-4945-85ed-023206e2420a
# ╠═2c620fe4-2f62-40f8-a666-8dced1e0b84a
# ╟─a22e5d34-4b8d-479c-985c-d6abd41a6c80
# ╠═cafb20b4-a2bd-46a9-9660-b0ace84d6e4c
# ╠═b07460f1-0461-4f63-b145-c4e1818a497e
# ╠═d577b393-4b40-4c90-9993-4ffbcbd9df6d
# ╠═ac80958a-73ec-4342-b553-b33df6612a50
# ╠═1e9c537a-a731-4b81-8f6a-cb658b52c5be
# ╠═cc9197e0-f5bd-4742-bea3-b54e0b8e3b93
# ╠═d42bb733-07e2-4932-aab4-09229ff67492
# ╠═742100ba-c38e-4840-8988-40990039b527
# ╠═7c5fb569-81f0-4b70-ae95-1fce0c51b6f4
# ╠═30ab21ba-3f5b-46a8-8b8c-753f2755d419
# ╠═5d23a0ba-5882-4ef6-ad56-596a3d66d3e8
# ╠═4afbb723-340b-4d85-9115-027a0ff8dfad
# ╠═f2201afe-8952-4dde-9e39-02beeb920f6f
# ╠═c1388562-0708-4a6a-acfe-927413dab5d2
# ╠═ddcb50be-5287-47f8-89f9-58c026a6b151
# ╠═1a82ae95-3c3e-4281-bc1d-9eb19bf50286
# ╠═5db29488-a150-42ee-aedb-380a3a4fd548
# ╟─59ec5223-f23f-4f32-9e5f-8a08e450da85
# ╟─49249ac1-8964-4afc-89f2-3cd4d4322cc2
# ╟─e1abf8c7-06b8-4cd5-b557-1d187004bdf1
# ╟─98a5d65e-4253-4523-a74e-99d03be03b89
# ╠═8ed6f8fd-8574-4d5a-9964-ce8a32629c6f
# ╠═1410db13-4b73-4a87-af34-30a5232af4ba
# ╠═7812e801-70fd-4331-ad0a-fad02c1a399f
# ╠═5be866c3-0fb2-4d1f-9c31-b85aba332905
# ╟─f7410fe7-e3d8-4047-8fa7-f076476e9d3a
# ╠═cbac1927-b087-4c4c-98ae-6aa5f0b824ad
# ╠═5515db1c-b3d1-4af5-8613-030a4b0faf09
# ╠═b5409b69-a254-4355-b2b9-99394eceb2f7
# ╟─d6ad1ff1-8fbf-4799-8b1b-ae1e3ce88c5b
# ╟─b8c031ca-7995-4501-a1e3-df3f34e5f0da
# ╟─69a06405-57cd-42e5-96b1-5cc77d74aa03
# ╠═a9fdb1fd-3f62-4e1c-9157-c4eee6215261
# ╠═565c53ee-7ad5-44e2-bce5-4ff1f5f162c0
# ╠═c9759bd9-ec9b-47a1-9080-a7fc332be565
# ╠═065b2626-01f1-443f-8be4-3036003a2772
# ╟─1a7ba296-52ca-4069-85fa-792d08d77b0e
# ╠═eb28458f-b222-4f8e-9a5b-8203d3997f7b
# ╠═e5ad765a-341f-4f11-9ae8-37d81cb349d2
# ╠═bc1d7cce-c0f4-47a8-b674-8acb82491c7f
# ╠═49e43d51-05d6-415b-a685-76e50904c5bc
# ╠═db189316-e880-4cc8-9070-ccfe2b4fc545
# ╠═7bc49107-9de5-4985-8750-979f36b3aa81
# ╠═ab4cb3db-3a2d-4145-826b-b1001114eeff
# ╠═4775126e-4374-49be-b25f-4597401f3642
# ╟─9df1a18d-137c-4ea5-8d15-05697f7bbf07
# ╟─0c7f5742-6c51-4c6a-b67f-217163935ba5
# ╟─a6c5ec28-b2d5-4893-a118-95c1318d1f7f
# ╟─44d43dbf-fe32-438e-b89d-c677bbc35893
# ╟─f1edb500-fbd1-4c03-b033-53860dfa452d
# ╟─2d7679ad-a9b3-448b-a4bc-7e5b9bce6adb
# ╠═a9b74949-9392-4048-bcb6-5fd48c1d9b98
# ╠═fbf1c64f-1979-4384-a8c6-dc7875174d1f
# ╠═e7372e2b-a2db-4a93-9efc-f75aa74c197b
# ╠═014339eb-5b23-4ac5-a551-8eeb2238366f
# ╠═62839b2a-398a-4445-87d1-b15ff2acc1d1
# ╠═96548352-cd4d-4448-8312-ed10057f4359
# ╠═4271151d-d5a6-4a29-96c3-f2102b142b95
# ╠═5534526a-d790-4379-98b0-8e4ee981fd9f
# ╠═f68952bd-4c0b-4331-a09c-5b118c8fa5a9
# ╠═b4af8d87-a6e5-4e09-92b4-b07757f58f7f
# ╠═41c626c7-908d-4ff6-9730-4ad0b8c3cc25
# ╠═546a775e-d3c9-4693-9f64-d4c47a84fb9f
# ╠═662759be-282c-460b-adc3-8595475b53c2
# ╟─0e66a941-1ec1-4d3b-b064-e5f25cc93baf
# ╟─c316c5d3-f484-4e8e-bd56-be1e236d96bc
# ╟─bc220d14-97fd-486d-9880-6908135fe036
# ╟─39eada35-8c3e-4ddc-8df9-7cf9f120928d
# ╟─8752c98d-fac1-4b3b-b20b-70acc0677fcb
# ╟─6cea9e69-bf8c-4079-9884-663a728d7b08
# ╠═69fb26ed-763e-44ad-9b70-193e5a1a09b9
# ╠═3f7484b3-272d-410d-92b1-ca13e5d7a8b7
# ╠═9fb5dace-a799-4424-bcb3-8542e508dd4b
# ╠═f5e32900-6eb6-4b61-916d-893c0bcaf214
# ╠═ed1bd92c-8cc7-457f-9692-a10a9487c953
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
