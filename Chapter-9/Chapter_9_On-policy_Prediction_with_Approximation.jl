### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 6da69e64-743f-4ea9-9670-fd023c7ffab7
using PlutoDevMacros

# ╔═╡ 808fcb4f-f113-4623-9131-c709320130df
PlutoDevMacros.@frompackage joinpath(@__DIR__, "..", "NonTabularRL.jl") begin
	using NonTabularRL
	using >.SparseArrays, >.Random, >.Statistics
end

# ╔═╡ 5169f3c9-926b-46c2-9d1c-35cc6919f46a
using LinearAlgebra

# ╔═╡ db8dd224-abf1-4a65-b8bb-e2da6ab43f7e
# ╠═╡ skip_as_script = true
#=╠═╡
using PlutoPlotly, PlutoUI, PlutoProfile, BenchmarkTools, LaTeXStrings, HypertextLiteral
  ╠═╡ =#

# ╔═╡ 19d23ef5-27db-44a8-99fe-a7343a5db2b8
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
# Chapter 9 On-policy Prediction with Approximation
## 9.1 Value-function Approximation
The method we use to approximate the true value function must be able to learn efficiently from incrementally acquired data.  Also the target values of training the function may be non stationary.  We will designate some approximation function for our value function as $\hat v(S, w)$ which is parametrized by some weights  that in general will be much smaller than in size to the true state space.
"""
  ╠═╡ =#

# ╔═╡ c4c71ace-c3a4-412b-b08b-31d246f8db5f
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
## 9.2 The Prediction Objective ($\overline {VE}$)
In tabular methods, the learned value can exactly equal the true objective and each state approximation is independent.  Neither of these are true for parametrized approximation.  We must specificy a state distribution $\mu(s) \geq 0, \sum_s{\mu(s)}=1$ that represents how much we care about the error in each state.  One natural objective function is the mean squared error weighted over this distribution.

$\overline{VE}(\boldsymbol{w}) \doteq \sum_{s \in S} \mu(s)[v_\pi(s) - \hat v(s, \boldsymbol{w})]^2 \tag{9.1}$

Often $\mu(s)$ is taken to be the fraction of time spent in $s$.  In contiunuing tasks the on-policy distribution is the stationary distribution under $\pi$.  In episodic tasks one must account for the probability of starting an episode in a particular state and the probability of transitioning to that state during an episode.  The state distribution will need to depend on that function typically denoted $\eta(s)$.

An ideal goal for optimizing $\overline {VE}$ is to find a *global optimum* for the weight vector such that $\overline {VE}(\boldsymbol{w}^*) \leq \overline {VE}(\boldsymbol{w})$ for all posible weights.  Typically this isn't possible but we can find a *local optimum* but even this objective is not guaranteed for many approximation methods.  In this chapter we will focus on approximation methods based on linear gradient-descent methods to we have easily find an optimum.
"""
  ╠═╡ =#

# ╔═╡ cb5e302b-a14b-4135-b6ff-bee300f9dee6
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
## 9.3 Stochastic-gradient and Semi-gradient Methods
We will assume a weight vector with a fixed number of components $\boldsymbol{w} \doteq (w_1, w_2, \dots, w_d)$ and a differentiable value function $\hat v(s, \boldsymbol{w})$ that exists for all states.  We will update weights at each of a series of discrete time steps so we can denote $\boldsymbol{w}_t$ as the weight vector at each step.  Assume at each step we observe a state and its true value under the policy.  We assume that states appear in the same distribution $\mu$ over which we are trying to optimize the prediction objective.  Under these assumptions we can try to minimize the error observed on each example using *Stochastic gradient-descent* (SGD) by adjusting the weight vector a small amount after each observation:

$$\begin{flalign}
\boldsymbol{w}_{t+1} & \doteq \boldsymbol{w}_t - \frac{1}{2} \alpha \nabla [v_\pi(S_t) - \hat v(S_t, \boldsymbol{w}_t)]^2 \\
& = \boldsymbol{w}_t + \alpha[v_\pi(S_t) - \hat v(S_t, \boldsymbol{w}_t)]\nabla\hat v(S_t, \boldsymbol{w}_t) \tag{9.5}
\end{flalign}$$

where $\alpha$ is a learning rate.  In general this method will only converge to the weight vector that minimizes the error objective if $\alpha$ is sufficiently small and decreases over time.  The gradient is defined as follows:

$\nabla f(\boldsymbol{w}) \doteq \left ( \frac{\partial{f(\boldsymbol{w})}}{\partial{w_1}} , \frac{\partial{f(\boldsymbol{w})}}{\partial{w_2}}, \cdots, \frac{\partial{f(\boldsymbol{w})}}{\partial{w_d}} \right ) ^ \top \tag{9.6}$

If we do not receive the true value function at each example but rather a bootstrap approxmiation or a noise corrupted version, we can use the same formula and simply replace $v_\pi(S_t)$ with $U_t$.  As long as $U_t$ is an *unbiased* estimate for each example then the weights are still guaranteed to converge to a local optimum stochastically.  One example of an unbiased estimate would be a monte carlo sample of the discounted future return.

If we use a bootstrapped estimate of the value, then the estimate depends on the current weight vector and will no longer be *unbiased* which requires that the update target be independent of $\boldsymbol{w}_t$.  A method using bootstrapping with function approximation would be considered a *semi-gradient method* because it violates part of the convergence assumptions.  In the case of a linear function, however, they can still converge reliably.  One typical example of this is semi-gradient TD(0) learning which uses the value estimate target of $U_t \doteq R_{t+1} + \gamma \hat v(S_{t+1}, \boldsymbol{w})$.  In this case the update step for the weight vector is as follows:

$\boldsymbol{w}_{t+1} = \boldsymbol{w}_t + \alpha[R_t + \gamma \hat v(S_{t+1}, \boldsymbol{w}_t) - \hat v(S_t, \boldsymbol{w}_t)] \nabla \hat v(S_t, \boldsymbol{w}_t) \tag{9.7}$

*State aggregation* is a simple form of generalizing function approximation in which states are grouped together, with one estimated value (one component of the weight vector **w**) for each group.  The value of a state is estimated as its group's component, and when the state is updated, that component alone is updated.  State aggregation is a special case of SGD in which the gradient, $\nabla \hat v(S_t, \boldsymbol{w}_t)$, is 1 for the observed state's component and 0 for others.
"""
  ╠═╡ =#

# ╔═╡ 865ed63a-a7ee-403f-9004-b3ec659d756f
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
### *Gradient Monte Carlo Algorithm for Estimating $$\hat v \approx v_\pi$$*

Monte Carlo sampling to estiamate $G_t$ can be used as a true gradient approximation method because $G_t$ is an unbiased estimate of $v_\pi (S_t)$ that does not depend on the parameters of the estimator.
"""
  ╠═╡ =#

# ╔═╡ be546bdb-77a9-48c4-9a98-1205d73fc8c6
"""
	gradient_monte_carlo_episode_update!(parameters::Vector{T}, gradients::Vector{T}, ▽v̂!::Function, states::AbstractVector{S}, actions::AbstractVector{A}, rewards::AbstractVector{T}, γ::T, α::T) where {T<:Real, S, A}

Interate through an episode and update the `parameters` given in the first argument as a vector of some numerical type.  The second input, `gradients` must be a vector of the same length as `parameters` and will also be filled with the correct gradient values using `▽v̂!` which must be callable as follows:  `v̂::T = ▽v̂!(gradients::Vector{T}, s::S, parameters::Vector{T})`.  It produces the value of the current estimator at state `s` but also updates the gradients for the parameter update.  

# Arguments
- `parameters::Vector{T}`: Vector of parameters used by the estimation function
- `gradients::Vector{T}`: Vector of the same length as `parameters` which stores the gradient updates
- `▽v̂!`: Function which performs the gradient update and the function value.  See above for call signature
- `states::AbstractVector{S}`: List of states encountered during an episode
- `actions::AbstractVector{A}`: List of actions encountered during an episode
- `rewards::AbstractVector{T}`: List of rewards encountered during an episode
- `γ::T`: Discount rate for computing discounted future reward
- `α::T`: Step size parameter for gradient update
"""
function gradient_monte_carlo_episode_update!(parameters::P, gradients::P, state_representation::AbstractVector{T}, update_parameters!::Function, states::AbstractVector{S}, rewards::AbstractVector{T}, γ::T, α::T; save_error = false) where {T<:Real, S, P}
	g = zero(T)
	l = length(states)
	error = zero(T)
	for i in l:-1:1
		s = states[i]
		g = γ * g + rewards[i]
		sqerr = update_parameters!(parameters, gradients, state_representation, s, g, α)
		error += sqerr
	end
	return error / l
end

# ╔═╡ ae19496f-7d6c-4b91-8456-d7a1eacbe3d3
function gradient_monte_carlo_policy_estimation!(parameters, mdp::StateMDP, π::Function, γ::T, num_episodes::Integer, update_parameters!::Function, state_representation::AbstractVector{T}; α = one(T)/10, gradients = deepcopy(parameters), epkwargs...) where {T<:Real}
	(states, actions, rewards, _) = runepisode(mdp; π = π, epkwargs...)
	sqerr = gradient_monte_carlo_episode_update!(parameters, gradients, state_representation, update_parameters!, states, rewards, γ, α)
	rmse_history = zeros(T, num_episodes)
	rmse_history[1] = sqrt(sqerr)
	for ep in 2:num_episodes
		(states, actions, rewards, _, n_steps) = runepisode!((states, actions, rewards), mdp; π = π, epkwargs...)
		sqerr = gradient_monte_carlo_episode_update!(parameters, gradients, state_representation, update_parameters!, view(states, 1:n_steps), view(rewards, 1:n_steps), γ, α)
		rmse_history[ep] = sqrt(sqerr)
	end
	return rmse_history
end


# ╔═╡ 7542ff9c-c6a1-4d41-8863-05388fea8ce2
function gradient_monte_carlo_estimation!(parameters, mrp::StateMRP, γ::T, num_episodes::Integer, update_parameters!::Function, state_representation::AbstractVector{T}; α = one(T)/10, gradients = deepcopy(parameters), epkwargs...) where {T<:Real}
	(states, rewards, _) = runepisode(mrp;epkwargs...)
	sqerr = gradient_monte_carlo_episode_update!(parameters, gradients, state_representation, update_parameters!, states, rewards, γ, α)
	rmse_history = zeros(T, num_episodes)
	rmse_history[1] = sqrt(sqerr)
	for ep in 2:num_episodes
		(states, rewards, _, n_steps) = runepisode!((states, rewards), mrp; epkwargs...)
		sqerr = gradient_monte_carlo_episode_update!(parameters, gradients, state_representation, update_parameters!, view(states, 1:n_steps), view(rewards, 1:n_steps), γ, α)
		rmse_history[ep] = sqrt(sqerr)
	end
	return rmse_history
end

# ╔═╡ df56b803-0aa5-4946-8338-601195e57a3e
md"""
### *Semi-gradient TD(0) for estimating $$\hat v \approx v_\pi$$*

When $U_t \doteq R_{t+1} + \gamma \hat v(S_{t+1}, \boldsymbol{w})$ the target value is the same as for temporal difference learning.  Now that the target uses parameter estimates, our gradient update is no longer correct since the target also depends on the parameters.  Thus this method is called `semi` gradient and has good convergence properties in the linear case.
"""

# ╔═╡ 160d1b6f-3340-4326-bfd3-c7d3f2328488
function semi_gradient_td0_policy_estimation!(parameters::Vector{T}, mdp::StateMDP, π::Function, γ::T, max_episodes::Integer, max_steps::Integer, estimate_value::Function, update_parameters!::Function, state_representation::AbstractVector{T}; α = one(T)/10, gradients = similar(parameters), scratch = similar(parameters), epkwargs...) where {T<:Real}
	s = mdp.initialize_state()
	a = π(s)
	(r, s′) = mdp.ptf(s, a)
	ep = 1
	step = 1
	while (ep <= max_episodes) && (step <= max_steps)
		v̂′ = mdp.isterm(s′) ? 0f0 : estimate_value(s′, parameters, state_representation)
		v̂ = r + γ*v̂′
		update_parameters!(parameters, gradients, state_representation, s, v̂, α)
		if mdp.isterm(s′)
			s = mdp.initialize_state()
			ep += 1
		else
			s = s′
		end
		a = π(s)
		(r, s′) = mdp.ptf(s, a)
		step += 1
	end
	return parameters
end

# ╔═╡ 5d90b840-4979-4e8b-bad1-68a3dc7aa392
function semi_gradient_td0_estimation!(parameters::Vector{T}, mrp::StateMRP, γ::T, max_episodes::Integer, max_steps::Integer, estimate_value::Function, update_parameters!::Function, state_representation::AbstractVector{T}; α = one(T)/10, gradients = similar(parameters), scratch = similar(parameters), epkwargs...) where {T<:Real}
	s = mrp.initialize_state()
	(r, s′) = mrp.ptf(s)
	ep = 1
	step = 1
	while (ep <= max_episodes) && (step <= max_steps)
		v̂′ = mrp.isterm(s′) ? 0f0 : estimate_value(s′, parameters, state_representation)
		v̂ = r + γ*v̂′
		update_parameters!(parameters, gradients, state_representation, s, v̂, α)
		if mrp.isterm(s′)
			s = mrp.initialize_state()
			ep += 1
		else
			s = s′
		end
		(r, s′) = mrp.ptf(s)
		step += 1
	end
	return parameters
end

# ╔═╡ 512f1358-0536-4d60-9ba6-173138ee6e14
semi_gradient_td0_policy_estimation(mdp::StateMDP, π::Function, γ::T, num_params::Integer, estimate_value::Function, update_parameters!::Function, state_representation::AbstractVector{T}; max_steps = 100_000, max_episodes = typemax(Int64), w_init = zero(T), parameters = fill(w_init, num_params), kwargs...) where {T<:Real} = semi_gradient_td0_policy_estimation!(parameters, mdp, π, γ, max_episodes, max_steps, estimate_value, update_parameters!, state_representation; kwargs...)

# ╔═╡ 8f4c82ee-d45a-41d8-b668-234de6d85f4d
semi_gradient_td0_estimation(mrp::StateMRP, γ::T, num_params::Integer, estimate_value::Function, update_parameters!::Function, state_representation::AbstractVector{T}; max_steps = 100_000, max_episodes = typemax(Int64), w_init = zero(T), parameters = fill(w_init, num_params), kwargs...) where {T<:Real} = semi_gradient_td0_estimation!(parameters, mrp, γ, max_episodes, max_steps, estimate_value, update_parameters!, state_representation; kwargs...)

# ╔═╡ cb2005fd-d3e0-4f37-908c-77e4bbac45b8
#=╠═╡
md"""
### Example 9.1: State Aggregation on the $(@bind num_states NumberField(100:100_000, default = 1000)) State Random Walk
"""
  ╠═╡ =#

# ╔═╡ de9bea60-c91d-4253-bdd8-a3c1fde8941c
"""
	make_random_walk_mdp(num_states::Integer) 

Construct a random walk task as a tabular MDP in which any action has an equal chance of transitioning to one of the 200 neighboring states on the left or right of the current state.  If any of those states exceed the limit of the random walk, then that probability becomes the probability of terminating on that edge and receiving a reward of -1 for left termination or -1 for right termination.

Return type is a `TabularMDP`

See also: [`TabularMDP`](@ref TabularRL.TabularMDP)

"""
function make_random_walk_mrp(num_states::Integer)
	states = collect(0:num_states+1)
	state_index = TabularRL.makelookup(states)
	initial_state = ceil(Int64, num_states / 2)
	initialize_state_index() = initial_state + 1
	state_transition_map = Vector{SparseVector{Float32, Int64}}(undef, num_states+2)
	reward_transition_map = Vector{Vector{Float32}}(undef, num_states+2)
	for s in states
		if (s == 0) || (s == num_states+1)
			v = zeros(Float32, num_states+2)
			v[s+1] = 1f0
			state_transition_map[s+1] = SparseVector(v)
			reward_transition_map[s+1] = [0f0]
		else
			
			state_transitions = SparseVector(zeros(Float32, num_states+2))
			reward_transitions = Vector{Float32}()
			minleft = s-100
			maxright = s+100
			ptermleft = if minleft > 0
				0f0
			else
				Float32((-minleft + 1)/100)
			end
	
			pnontermleft = 1f0 - ptermleft
			nontermleftstates = max(1, s - 100):s-1
			for s′ in nontermleftstates
				state_transitions[s′+1] = (0.5f0 * pnontermleft) / length(nontermleftstates)
			end
			state_transitions[1] = ptermleft/2
	
			ptermright = if maxright <= num_states
				0f0
			else
				Float32((maxright - num_states) / 100)
			end
	
			pnontermright = 1f0 - ptermright
			nontermrightstates = s+1:min(num_states, maxright)
			for s′ in nontermrightstates
				state_transitions[s′+1] = (0.5f0 * pnontermright) / length(nontermrightstates)
			end
			state_transitions[num_states+2] = ptermright/2
			
			state_transition_map[s+1] = state_transitions
	
			for i_s′ in state_transitions.nzind
				r = if i_s′ == 1
					-1f0
				elseif i_s′ == num_states+2
					1f0
				else
					0f0
				end
				push!(reward_transitions, r)
			end
			reward_transition_map[s+1] = reward_transitions
		end
	end
	
	TabularMRP(states, TabularStochasticTransition(state_transition_map, reward_transition_map), initialize_state_index)
end

# ╔═╡ 7814bda0-4306-4060-8f9a-2bcf1cf8e132
#=╠═╡
const random_walk_tabular_mrp = make_random_walk_mrp(num_states)
  ╠═╡ =#

# ╔═╡ 69223862-4d74-46c9-8c78-b24d659151ac
#=╠═╡
const random_walk_v = mrp_evaluation(random_walk_tabular_mrp, 1f0)
  ╠═╡ =#

# ╔═╡ f4459b0d-ee3e-47c7-9c82-981af622edfa
#=╠═╡
const initial_state::Int64 = ceil(Int64, num_states / 2)
  ╠═╡ =#

# ╔═╡ 90e5fc0e-2e97-424b-a5dd-9deb38293121
#=╠═╡
md"""
Consider a $num_states-state version of the random walk task in which the states are numbered from 1 to $num_states, left to right and all episodes begin near the center, in state $initial_state.  State transitions are from the current state to one of the 100 neighboring states to its left, or to one of the 100 neighboring states to its right, all with equal probability.  Of course, if the current state is near an edge, then there may be fewer than 100 neighbors on that side of it.  In this case, all the probability that would have gone into those missing neighbors goes into the probability of terminating on that side (thus, state 1 has a 0.5 chance of terminating on the left, and state $(num_states - 50) has a 0.25 chance of terminating on the right).  Left termination produces a reward of -1 and right +1.

The following function constructs this random walk as a tabular problem with a stochastic distribution function like we'd see in part 1 of the book.  From this representation of the problem, we can perform methods like value iteration to calculate the correct state values and then compare to approximation methods later.
"""
  ╠═╡ =#

# ╔═╡ 68a4151a-52ee-4ed0-b988-3fecc34d8d32
#=╠═╡
md"""
#### Transition Probabilities Visualized for $num_states State Random Walk

Using the tabular MDP, we can visualize the transition probabilities for any state.  Notice that at the edges, more probability is shifted to a terminal state.
"""
  ╠═╡ =#

# ╔═╡ 24e8b391-00ec-4ed5-85dc-0796eb85bf4f
#=╠═╡
md"""Select State to View Transition Probabilities: $(@bind smap Slider(1:num_states; default = ceil(Int64, num_states/2), show_value=true))"""
  ╠═╡ =#

# ╔═╡ 736b7667-904d-4a9c-bb10-a6b0b831bfb6
#=╠═╡
random_walk_tabular_mrp.ptf.state_transition_map[smap+1] |> v -> plot(bar(x = 0:num_states+1, y = v), Layout(xaxis_title = "State", yaxis_title = "Transition Probability"))
  ╠═╡ =#

# ╔═╡ 9c3f07b1-61eb-4d70-9dde-986c032a0840
md"""
#### Non-tabular Version of Random Walk Example
Since our goal is to compare estimation methods, we need to create a version of this problem that is non-tabular.  That way our state assignment function can be used properly to map a state to a particular parameter, effectively grouping them together instead of treading them each individually as in the tabular case.  The transition function for this case will operate on states and produce states that can then be mapped to the appropriate parameters.  Rather than converting the tabular MDP into a non-tabular one, this construction uses a faster step function.  By default, the conversion would create a step that produces the full distribution of transition states rather than just efficiently randomly sampling from them which is achieved here by the `randomwalk_step` method.
"""

# ╔═╡ 3f2ce7e0-b623-4ce3-90cf-949f3a6b0633
function randomwalk_step(s::Float32, num_states::Int64)
	x = Float32(ceil(rand() * 100))
	s′ = s + x * rand((-1f0, 1f0))

	r = Float32(-(s′ < 1) + (s′ > num_states))
	(r, s′)
end

# ╔═╡ 39c6ec4d-306e-4dee-9d5a-130925341a6c
#=╠═╡
const randomwalk_state_ptf = StateMRPTransitionSampler((s) -> randomwalk_step(s, num_states), 1f0)
  ╠═╡ =#

# ╔═╡ 60d68f9b-d18d-4d23-9adb-27fcb205e54b
randomwalk_isterm(s::Float32, num_states::Int64) = (s < 1) || (s > num_states)

# ╔═╡ c79db82f-289e-4523-bf07-57cfdc38c073
#=╠═╡
randomwalk_state_init() = Float32(initial_state)
  ╠═╡ =#

# ╔═╡ 2720329c-4c80-47cb-a3e3-d24fcec6ef43
#=╠═╡
const random_walk_state_mrp = StateMRP(randomwalk_state_ptf, randomwalk_state_init, s -> randomwalk_isterm(s, num_states))
  ╠═╡ =#

# ╔═╡ 2c6809f9-50ed-44b8-8f27-0a62e88d118c
#=╠═╡
md"""
#### State Aggregation

The simplest form of function approximation in which each state is assigned to a unique group.  Each group is represented by a parameter that estimates the value of every state in that group.  The gradient for this technique has the simple form: $\nabla \hat v (S_t, \boldsymbol{w}_t) = 1$ if $S_t$ is in the group represented by $\boldsymbol{w}_t$ and 0 otherwise.  For the random walk example, state aggregation can simply assign states to groups as: {1 to 100}, {101 to 200}, ..., {$(num_states - 100) to $num_states}.
"""
  ╠═╡ =#

# ╔═╡ 91e4e5da-4e0f-48b2-98bd-1e9f1330b0a8
#=╠═╡
md"""Number of State Aggregation Groups: $(@bind num_groups NumberField(1:num_states, default = 10))"""
  ╠═╡ =#

# ╔═╡ 5ebafa8b-c316-4f95-8adc-581f2eb40e1f
function make_random_walk_group_assign(num_states::Integer, num_groups::Integer)
	groupsize = num_states / num_groups
	assign_group(s::Real) = ceil(Int64, s / groupsize)
end

# ╔═╡ 24b99200-053a-41bf-a628-0b14b807fb86
#=╠═╡
#this function will assign a state to a group
random_walk_group_assign = make_random_walk_group_assign(num_states, num_groups)
  ╠═╡ =#

# ╔═╡ d68c0147-a66f-4542-a395-5f9b43e16b09
#=╠═╡
md"""
#### Group Aggregation Visualization for $num_states State Random Walk
"""
  ╠═╡ =#

# ╔═╡ 1adf0786-0897-4119-9336-09de869463b4
#=╠═╡
random_walk_group_assign.(random_walk_tabular_mrp.states) |> v -> plot(scatter(x = random_walk_tabular_mrp.states, y = v), Layout(xaxis_title = "State", yaxis_title = "Aggregation Group", title = "$num_states Random Walk States Partitioned into $num_groups Groups"))
  ╠═╡ =#

# ╔═╡ b361815f-d5b0-4c71-b331-c3b48ce53e73
md"""
Using the simple gradient for state aggregation, we can construct a function that computes the state value estimate and gradient per parameter component.  In order to implement state aggregation, one must have a fixed number of groups and a function to map states to a group index.  There will be one parameter value per group, so the gradient function needs to provide a component for every group.  Once a state is assigned into a unique group index, the gradient values will all be zero except for at the group index.  The value estimate is just the parameter value at that index.  In the case of the random walk example, assigning states to groups is simply a matter of dividing the state value by the group size and finding the next highest integer value.
"""

# ╔═╡ c46c36f6-42da-4767-9e25-fa0ebe43998f
function state_aggregation_gradient_setup(assign_state_group::Function)
	function update_parameters!(parameters::Vector{T}, gradients::Vector{T}, x, s, g::T, α::T) where {T<:Real}
		i = assign_state_group(s)
		((i < 1) || (i > length(parameters))) && return nothing
		v̂ = parameters[i]
		δ = (g - v̂)
		parameters[i] += α*δ
		return δ^2
	end

	v̂(s, w::Vector{T}, x) where {T<:Real} = w[assign_state_group(s)]
	
	return (value_function = v̂, parameter_update = update_parameters!)
end

# ╔═╡ 47116ee6-53db-47fe-bfc9-a322f85b3e4e
function run_state_aggregation_monte_carlo_policy_estimation(mdp::StateMDP{T, S, A, P, F1, F2, F3}, π::Function, γ::T, num_episodes::Integer, num_groups::Integer, assign_state_group::Function; w0::T = zero(T), kwargs...) where {T<:Real, S, A, P<:AbstractStateTransition{T}, F1<:Function, F2<:Function, F3<:Function}
	setup = state_aggregation_gradient_setup(assign_state_group)
	params = fill(w0, num_groups)
	err_history = gradient_monte_carlo_estimation!(params, mdp, π, γ, num_episodes, setup.parameter_update, [one(T)]; kwargs...)
	v̂(s) = setup.value_function(params, s, [one(T)])
	return (v̂ = v̂, error_history = err_history)
end

# ╔═╡ 2aadb2bf-942b-436e-8b93-111a90b3ea2b
function run_state_aggregation_monte_carlo_estimation(mrp::StateMRP{T, S, P, F1, F2}, γ::T, num_episodes::Integer, num_groups::Integer, assign_state_group::Function; w0::T = zero(T), kwargs...) where {T<:Real, S, P<:AbstractStateTransition{T}, F1<:Function, F2<:Function}
	setup = state_aggregation_gradient_setup(assign_state_group)
	params = fill(w0, num_groups)
	err_history = gradient_monte_carlo_estimation!(params, mrp, γ, num_episodes, setup.parameter_update, [one(T)]; kwargs...)
	v̂(s) = setup.value_function(s, params, [one(T)])
	return (v̂ = v̂, error_history = err_history)
end

# ╔═╡ ace0693b-b4ce-43df-966e-0330d4399638
#=╠═╡
md"""
### *Figure 9.1*

Function approximation by state aggregation on the $num_states-state random walk task.  The blue line shows the true state values computed using value iteration.  The stepped orange line shows the group values as calculated using Gradient Monte Carlo estimation using the state aggregation parameters.  The distribution of visited states during an episode is also shown as a history.
"""
  ╠═╡ =#

# ╔═╡ bc479ae0-78ea-4255-863f-dcd126ae9b96
md"""
Our prediction objective will favor lower error on highly visited states than less requently visited ones.  Since the distribution of visited states as weighted towards the center, the error between the parameter estimated state value and the true state value is lower for states close to the center in a group.  That can be seen very clearly for group 1 where the right edge is far close to the blue line than the leftmost edge.  The leftmost state is the least likely to be visited and thus matters to least for minimizing prediction error.
"""

# ╔═╡ 214714a5-ad1e-4439-8567-9095d10411a6
#=╠═╡
function figure_9_1()
	v = random_walk_v.value_function[2:end-1]
	(random_walk_v̂, error_history) = run_state_aggregation_monte_carlo_estimation(random_walk_state_mrp, 1f0, 100_000, num_groups, random_walk_group_assign, α = 2f-5)
	v̂ = random_walk_v̂.(1:num_states)
	x = 1:num_states
	n1 = L"v_\pi"
	n2 = L"\hat v"
	tr1 = scatter(x = x, y = v, name = "True value $n1")
	tr2 = scatter(x = x, y = v̂, name = "Approximate MC value $n2")
	

	state_counts = zeros(Int64, num_states)
	function update_state_counts!(state_counts, states)
		for s in states
			state_counts[Integer(s)] += 1
		end
	end
	
	(states, rewards, sterm, numsteps) = runepisode(random_walk_state_mrp)
	update_state_counts!(state_counts, view(states, 1:numsteps))
	for _ in 1:100_000
		(states, rewards, sterm, num_steps) = runepisode!((states, rewards), random_walk_state_mrp)
		update_state_counts!(state_counts, view(states, 1:num_steps))
	end
	state_distribution = state_counts ./ sum(state_counts)
	n3 = L"\mu"
	tr3 = bar(x = x, y = state_distribution, yaxis = "y2", name = "State distribution $n3", marker_color = "gray")

	state_mean = sum(state_distribution[i]*i for i in eachindex(state_distribution))
	state_variance = sum(state_distribution[i]*((i - state_mean)^2) for i in eachindex(state_distribution))
	p1 = plot([tr1, tr2, tr3], Layout(xaxis_title = "State", yaxis_title = "Value scale", yaxis2 = attr(title = "Distribution scale", overlaying = "y", side = "right"), title = "State Mean Value: $state_mean, State Value Variance: $state_variance"))

	p2 = plot(scatter(x = 1001:length(error_history), y = [mean(error_history[i-1000:i]) for i in 1001:length(error_history)]), Layout(xaxis_title = "Episode", yaxis_title = "RMS Error Over Previous 1000 Episodes"))
	md"""
	$p1
	$p2
	"""
end
  ╠═╡ =#

# ╔═╡ c0e9ea1f-8cbe-4bc1-990f-ffd3ab1989cc
#=╠═╡
figure_9_1()
  ╠═╡ =#

# ╔═╡ 49320a88-206e-4283-b3fc-a5d1ac41ddc4
#=╠═╡
function smooth_error(error_history, n)
	l = length(error_history)
	[mean(error_history[i-n:i]) for i in n+1:l]
end
  ╠═╡ =#

# ╔═╡ 3160e3ec-d1b9-47ea-ad10-3d6ea40cc0b5
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
## 9.4 Linear Methods
"""
  ╠═╡ =#

# ╔═╡ 6c6c0ef4-0e68-4f50-8c3a-76ed3afb2d20
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
Linear methods represent the value function as an inner product between *feature vectors* and *weight vectors*.

$\hat v(s, \mathbf{w})\doteq \mathbf{w}^\top \mathbf{x}(x) \doteq \sum_{i=1}^d w_i x_i(s)$ 

The vector $\mathbf{x}(s)$ is called a *feature vector* representing state x which is the same length as the number of parameters contained in $\mathbf{w}$.  For linear methods, features are *basis functions* because they form a linear basis for the set of approximate functions.

The gradient of linear value functions takes on a particularly simple form: $\nabla \hat v (s, \boldsymbol{w}) = \boldsymbol{x}(s)$.  Thus the general SGD update (9.7) reduces to:

$\boldsymbol{w}_{t+1} \doteq \boldsymbol{w}_t + \alpha \left [ U_t - \hat v (S_t, \boldsymbol{w}_t) \right ] \boldsymbol{x}(S_t)$

In the linera case there is only one optimum (or set of equally good optima), so any method that is guaranteed to converge to a local optimum is automatically guaranteed to converge to or near the global optimum.  For example, gradient Monte Carlo converges to the global optimum of the $\overline{VE}$ under linear function approximation if $\alpha$ is reduced over time according to the usual conditions.

The semi-gradient TD(0) algorithm presented in the previous section also converges under linear function approximation, but this does not follow from general results on SGD; a separate theorem is necessary.  The weight vector converged to is also not the global optimum, but rather a point near the local optimum.  It is useful to consider this important case in more default, specifically for the continuing case.  The update at each time step $t$ is 

$\begin{flalign}
\boldsymbol{w}_{t+1} &\doteq \boldsymbol{w}_t +\alpha \left (R_{t+1} + \gamma \boldsymbol{w}_t ^ \top \boldsymbol{x}_{t+1} - \boldsymbol{w}_t ^ \top \boldsymbol{x}_t \right ) \boldsymbol{x}_t \tag{9.9}\\
&= \boldsymbol{w}_t + \alpha \left ( R_{t+1} \boldsymbol{x}_t - \boldsymbol{x}_t (\boldsymbol{x}_t - \gamma \boldsymbol{x}_{t+1} ) ^ \top \right ) \boldsymbol{w}_t
\end{flalign}$

where here we have used the notational shorthand $\boldsymbol{x}_t = \boldsymbol{x}(S_t)$.  Once the system has reached steady state, for any given $\boldsymbol{w}_t$, the expected next weight vector can be written

$\mathbb{E}[\boldsymbol{w}_{t+1} /vert \boldsymbol{w}_t] = \boldsymbol{w}_t + \alpha(\boldsymbol{b} - \boldsymbol{A} \boldsymbol{w}_t \tag{9.10}$

where

$\boldsymbol{b} \doteq \mathbb{E}[R_{t+1} \boldsymbol{x}_t] \in \mathbb{R}^d \text{             and           } \boldsymbol{a} \doteq \mathbb{E} \left [ \boldsymbol{x}_t (\boldsymbol{x}_t - \gamma \boldsymbol{x}_{t+1}) ^\top \right] \in \mathbb{R}^{d \times d} \tag{9.11}$

From (9.10) it is clear that, if the system converges, it must converge to the weight vector $\boldsymbol{w}_{\text{TD}}$ at which

$\begin{flalign}
\boldsymbol{b} - \boldsymbol{A} \boldsymbol{w}_\text{TD} &= \boldsymbol{0} \\
\implies \boldsymbol{b} = \boldsymbol{A} \boldsymbol{w}_\text{TD} \\
\implies \boldsymbol{w}_\text{TD} \doteq \boldsymbol{A}^{-1} \boldsymbol{b} \tag{9.12}
\end{flalign}$

This quantity is called the *TD fixed point*.  In fact, linear semi-gradient TD(0) converges to this point.  See details below:
"""
  ╠═╡ =#

# ╔═╡ b6737cef-b6f9-4e40-82d8-bf887e17eb7c
md"""
### Proof of Convergence of Linear TD(0)
"""

# ╔═╡ 3db9f60e-a823-4d78-bd16-e73cedffa755
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
At the TD fixed point, it has also been proven (in the continuing case) that the $\overline{VE}$ is within a bounded expansion of the lowest possible error: 

$\overline{VE}(\boldsymbol{w}_\text{TD}) \leq \frac{1}{1-\gamma} \min_{\boldsymbol{w}} \overline{VE} (\boldsymbol{w}) \tag{9.14}$

That is, the asymptotic error of the TD method is no more than $\frac{1}{1-\gamma}$ times the smallest possible error, that attained in the limit by the Monte Carlo method.  Because $\gamma$ is often near one, this expansion factor can be quite large, so there is substantial potential loss in asymptotic performance with the TD method.  On the otehr hand, recall that the TD methods are often of vastly reduced variance compared to Monte Carlo methods, and thus faster, as we saw in Chapters 6 and 7.

A bound analogous to (9.14) applies to other on-policy bootstrapping methods as well.  For example, linear semi-gradient DP $\left ( U_t \doteq \sum_a \pi(a \vert S_t) \sum_{s^\prime, r} p(s\prime, r \mid S_t, a)[r+\gamma \hat v(s^\prime, \boldsymbol{w}_t)] \right )$ with updates according to the on-policy distribution will also converge to the TD fixed point.  One-step semi-gradient *action-value* methods, such as semi-gradient Sarsa(0) convered in the next chapter converge to an analogous fixed point and an analogous bound.  Critical to these convergence results is that states are updated according to the on-policy distribution.  For other update distributions, bootstrapping methods using function approximation may actually diverge to infinity.
"""
  ╠═╡ =#

# ╔═╡ 8bd63a96-fcbe-47a8-a710-0c276586c3d6
begin
	function update_parameters!(parameters::Vector{T}, gradients::Vector{T}, state_representation::AbstractVector{T}, g::T, α::T) where {T<:Real}
		v̂ = dot(state_representation, parameters)
		δ = (g - v̂)
		iszero(δ) && return nothing
		parameters .+= α .* δ .* state_representation
		return δ^2
	end

	function update_parameters!(parameters::Vector{T}, gradients::Vector{T}, state_representation::SparseVector{T, Int64}, g::T, α::T) where {T<:Real}
		v̂ = dot(state_representation, parameters)
		δ = (g - v̂)
		iszero(δ) && return zero(T)
		x = α*δ
		for i in eachindex(state_representation.nzind)
			parameters[state_representation.nzind[i]] += x .* state_representation.nzval[i]
		end
		return δ^2
	end
end

# ╔═╡ 8ed8530f-4569-4429-92fc-3c3b1752475b
function linear_features_gradient_setup(problem::Union{StateMDP{T, S, A, P, F1, F2, F3}, StateMRP{T, S, P, F1, F2}}, state_representation::AbstractVector{T}, update_feature_vector!::Function) where {T<:Real, N, S <: Union{T, NTuple{N, T}}, A, P, F1<:Function, F2<:Function, F3<:Function}
	s0 = problem.initialize_state()
	update_feature_vector!(state_representation, s0) #verify that feature vector update is compatible with provided state representation

	function update_params!(parameters, gradients, state_representation, s, g, α)
		update_feature_vector!(state_representation, s)
		update_parameters!(parameters, gradients, state_representation, g, α)
	end
	
	
	function v̂(s::S, w::Vector{T}, state_representation::AbstractVector{T}) where {T<:Real} 
		update_feature_vector!(state_representation, s)
		dot(state_representation, w)
	end
	
	return (value_function = v̂, parameter_update = update_params!, feature_vector = state_representation)
end

# ╔═╡ 645ba5fc-8575-4b8f-8982-f8bd20ac27ff
#=╠═╡
md"""
### Example 9.2: Bootstrapping on the $num_states-state Random Walk

State aggregation is a special case of linear function approximation, so we can use the previous example to illustrate the convergence properties of semi-gradient TD(0) vs gradient Monte Carlo.  
"""
  ╠═╡ =#

# ╔═╡ 6046143f-a2c3-4569-a04a-c1ad4f3daf9d
function run_state_aggregation_semi_gradient_policy_estimation(mdp::StateMDP{T, S, A, P, F1, F2, F3}, π::Function, γ::T, num_groups::Integer, assign_state_group::Function; kwargs...) where {T<:Real, S, A, P<:AbstractStateTransition{T}, F1<:Function, F2<:Function, F3<:Function}
	setup = state_aggregation_gradient_setup(assign_state_group)
	params = semi_gradient_td0_policy_estimation(mdp, π, γ, num_groups, setup.value_function, setup.parameter_update, [one(T)]; kwargs...)
	v̂(s) = setup.value_function(params, s, [one(T)])
	return v̂
end

# ╔═╡ 023f0a8c-fa3c-4335-8301-6f358380fb76
function run_state_aggregation_semi_gradient_estimation(mrp::StateMRP{T, S, P, F1, F2}, γ::T, num_groups::Integer, assign_state_group::Function; kwargs...) where {T<:Real, S, P<:AbstractStateTransition{T}, F1<:Function, F2<:Function}
	setup = state_aggregation_gradient_setup(assign_state_group)
	params = semi_gradient_td0_estimation(mrp, γ, num_groups, setup.value_function, setup.parameter_update, [one(T)]; kwargs...)
	v̂(s) = setup.value_function(s, params, [one(T)])
	return v̂
end

# ╔═╡ cf9d7c7d-4519-410a-8a05-af90312e291c
#=╠═╡
md"""
### Figure 9.2
Bootstrapping with state aggregation on the $num_states-state random walk task.  The asymptotic values of semi-gradient TD are worse than the asymptotic Monte Carlo values which matches with the expectation from the TD-fixed point convergence.
"""
  ╠═╡ =#

# ╔═╡ bfb1858b-5e05-4239-bcae-a3b718074630
#=╠═╡
function figure_9_2()
	v = random_walk_v.value_function[2:end-1]
	
	v̂_mc, err_history = run_state_aggregation_monte_carlo_estimation(random_walk_state_mrp, 1f0, 100_000, num_groups, random_walk_group_assign, α = 2f-5)

	#this function will produce the learned value estimate given a random walk state
	v̂_td = run_state_aggregation_semi_gradient_estimation(random_walk_state_mrp, 1f0, num_groups, random_walk_group_assign; max_episodes = 100_000, max_steps = typemax(Int64), α = 1f-3)
	
	
	x = 1:num_states

	v̂_mc = v̂_mc.(x)
	v̂_td = v̂_td.(x)
	
	n1 = L"v_\pi"
	n2 = L"\hat v"
	tr1 = scatter(x = x, y = v, name = "True value $n1")
	tr2 = scatter(x = x, y = v̂_mc, name = "Approximate MC value $n2")
	tr3 = scatter(x = x, y = v̂_td, name = "Approximate TD value $n2")

	plot([tr1, tr2, tr3], Layout(xaxis_title = "State", yaxis_title = "Value"))
end
  ╠═╡ =#

# ╔═╡ c05ea239-2eea-4f41-b4e3-993db0fe2de5
#=╠═╡
figure_9_2()
  ╠═╡ =#

# ╔═╡ f5203959-29ef-406c-abac-4f01fa9630a3
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
> ### *Exercise 9.1* 
> Show that tabular methods such as presented in Part I of this book are a special case of linear function approximation.  What would the feature vectors be?

The simplest form of function approximation presented so far is state-aggregation which is a special case of linear function approximation.  Consider a case of state-aggregation where every state is in its own unique group and there is a parameter vector $\boldsymbol{w}$ such that $w_i$ is the approximation value for $s_i$.  Following the rules of state aggregation, the feature vectors would be orthanormal basis vectors of dimension matching the number of states, thus state 1 would be represented by the feature vector [1, 0, 0, ...], state 2 by [0, 1, 0, 0, ...] and so on.  The gradient Monte Carlo update rule for these feature vectors would be $w_i = w_i + \alpha [G_t - w_i]$ for an episode step encountering state $s_i$.  The TD(0) update rule would be $w_i = w_i + \alpha [R_t + \gamma w_j - w_i]$ where the next state encountered is $s_j$.  Both of these rules are exactly the same as tabular Monte Carlo policy prediction (with constant step size averaging) and tabular TD(0) policy prediction where $v_i = w_i$.  So the value function from the tabular setting is still a list of $\vert \mathcal{S} \vert$ values, one for each state and every state value update has no effect on the value estimates of other states.
"""
  ╠═╡ =#

# ╔═╡ 53924a3a-8fab-45c5-b6fa-90882138fac9
#once you do state aggregation you have effectively reduced it to a tabular problem, so why not just solve with DP methods like value iteration if you can get the probability distribution from the environment like we could with this random walk task?  Given the state groups I could construct an actual distribution model for this using the groups and then it should converge to the VE error I think.  the problem is even though I can get the distribution into the new groups from a given state, I have to add up all of those weighted equally by each state in the beginning group.

# ╔═╡ c3da96b0-d584-4a43-acdb-16516e2d0452
md"""
## 9.5 Feature Construction for Linear Methods

Linear methods can only make approximations that additively combine the effects of multiple features.  In order to account for interactions between state properties such as the position and velocity of an object, features must be constructed that explicitely combine those state values.  The purpose of feature construction is to inject into the problem domain knowledge related to what type of information from the states will be useful to solving the problem.
"""

# ╔═╡ 0ee3afe9-9c33-45c8-b304-26062675e1b8
md"""
### 9.5.1 Polynomials

Consider a state with two numerical features $s_1, s_2$.  We could construct a feature vector that simply uses each value $(s_1, s_2)$ but this would restrict our value estimator to outputs of the form $as_1 + bs_2$.  This functional form would make it impossible for an estimated value to be non-zero if both state values are zero which may not be true in the environment.  In order to lift this restriction it is common to add a bias feature that is always 1.  Another desired feature may be one that combines both state values together multiplicatively.  Additional features of this nature are called polynomial features and take the form:

$x_i(s) = \prod_{j=1}^k s_j^{c_{i,j}} \tag{9.17}$

where each $c_{i,j}$ is an integer in the set $\{0, 1, \dots, n \}$ for an integer $n \geq 0$.  An example of such a feature vector for $n=2$ and $k=2$ state values is shown below:

$\boldsymbol{x}(s) = (1, s_1, s_2, s_1 s_2, s_1^2, s_2^2, s_1 s_2^2, s_1^2 s_2, s_1^2 s_2^2)$

This combination yields $(2+1)^2 = 9$ features since each of the two state values can be raised to 3 different exponents and then combined.
"""

# ╔═╡ d65a0ca9-5577-4df8-af77-44ecfbcc0a07
md"""
> ### *Exercise 9.2* 
> Why does (9.17) define $(n+1)^k$ distinct features for dimension $k$?
n represents the highest power to take for each individual dimension of the state and we consider powers from 0 up to n for each dimension.  If we list the exponent per dimension as a tuple, we have for n = 1, k = 2: (0, 0), (0, 1), (1, 0), (1, 1).
For n = 1, k = 3: (0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1).  This pattern consists of tuples of length k which can be formed by selecting from n + 1 choices of exponent.  The number of resulting tuples is $(n+1)^k$
"""

# ╔═╡ c5adf2d7-0b6b-4a87-974b-a90824d0323b
md"""
> ### *Exercise 9.3* 
> What $n$ and $c_{i, j}$ produce the feature vectors $\mathbf{x}(s)=(1, s_1, s_2, s_1s_2, s_1^2, s_2^2, s_1s_2^2, s_1^2s_2, s_1^2s_2^2)^\top$

Since the highest exponent considered is 2, $n=2$.  For the exponents we can visualize $c_{i, j}$ as the following matrix where rows correspond to $i$ and columns to $j$


$\begin{matrix}
0 & 0\\
1 & 0\\
0 & 1\\
1 & 1\\
2 & 0\\
0 & 2\\
1 & 2\\
2 & 1\\
2 & 2\\
\end{matrix}$
"""

# ╔═╡ 38f09914-e128-4336-8e70-9906675971f2
function get_order_coefficients(k, n; coefs = ())
	k == 0 && return coefs
	reduce(vcat, get_order_coefficients(k-1, n; coefs = (coefs..., e)) for e in 0:n)
end

# ╔═╡ f5dea7d5-4597-430c-9020-b74cdf8f3055
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
Notice that these 9 exponents match the ones for the feature vector in exercise 9.3
"""
  ╠═╡ =#

# ╔═╡ 9d7ca70c-0e60-4029-8ea0-26192ccea849
function order_features_gradient_setup(problem::Union{StateMDP{T, S, A, P, F1, F2, F3}, StateMRP{T, S, P, F1, F2}}, n::Integer, min_values::S, max_values::S, feature_calculation::Function) where {T<:Real, N, S <: Union{T, NTuple{N, T}}, A, P, F1<:Function, F2<:Function, F3<:Function}
	#states must be tuples with k elements or some number value
	k = S == T ? 1 : N
	coefs = get_order_coefficients(k, n)

	l = length(coefs)

	state_representation = zeros(T, l)

	function update_feature_vector!(x::Vector{T}, s::T)
		@inbounds @simd for i in eachindex(x)
			feature = feature_calculation(s, min_values, max_values, coefs[i])
			x[i] = feature
		end
	end

	linear_features_gradient_setup(problem, state_representation, update_feature_vector!)
end

# ╔═╡ a0c4df88-ba30-463b-81ac-6f5511683730
function run_order_features_monte_carlo_policy_estimation(mdp::StateMDP{T, S, A, P, F1, F2, F3}, π::Function, γ::T, num_episodes::Integer, n::Integer, feature_calculation::Function, min_values::S, max_values::S; w0::T = zero(T), kwargs...) where {T<:Real, S, A, P<:AbstractStateTransition, F1<:Function, F2<:Function, F3<:Function}
	setup = order_features_gradient_setup(mdp, n, min_values, max_values, feature_calculation)
	params = fill(w0, setup.num_features)
	rmse_history = gradient_monte_carlo_estimation!(params, mdp, π, γ, num_episodes, setup.parameter_update; kwargs...)
	v̂(s) = setup.value_function(s, params, setup.feature_vector)
	return (v̂ = v̂, rmse_history = rmse_history)
end

# ╔═╡ 8c140c5e-af37-49cc-980e-96b146ebeb3c
function run_order_features_monte_carlo_policy_estimation(mrp::StateMRP{T, S, P, F1, F2}, γ::T, num_episodes::Integer, n::Integer, feature_calculation::Function, min_values::S, max_values::S; w0::T = zero(T), kwargs...) where {T<:Real, S, P<:AbstractStateTransition, F1<:Function, F2<:Function}
	setup = order_features_gradient_setup(mrp, n, min_values, max_values, feature_calculation)
	params = fill(w0, length(setup.feature_vector))
	rmse_history = gradient_monte_carlo_estimation!(params, mrp, γ, num_episodes, setup.parameter_update, setup.feature_vector; kwargs...)
	v̂(s) = setup.value_function(s, params, setup.feature_vector)
	return (v̂ = v̂, rmse_history = rmse_history)
end

# ╔═╡ bc2e52ff-7f47-4141-aff1-e752fe217f6a
begin
	calc_poly_feature(s::NTuple{N, T}, min_values::NTuple{N, T}, max_values::NTuple{N, T}, e::NTuple{N, Int64}) where {T<:Real, N} = prod(((s[i] - min_values[i]) / (max_values[i] - min_values[i]))^e[i] for i in 1:N)
	calc_poly_feature(s::T, min_value::T, max_value::T, e::NTuple{1, Int64}) where {T<:Real} = ((s - min_value) / (max_value - min_value))^e[1]
end

# ╔═╡ c609ee03-7217-4068-9da2-c91fb02623a9
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
Note that a scaling factor of 1/num_states means that all states will be mapped to the range of 0 to 1 for the purpose of computing the polynomial features.  This helps with numerical stability when we are taking the state integers to large powers of n"""
  ╠═╡ =#

# ╔═╡ eb8b26ed-8429-47b5-ab82-c6d79dd053e4
#=╠═╡
polynomial_random_walk_mc_v̂, poly_random_walk_rmse = run_order_features_monte_carlo_policy_estimation(random_walk_state_mrp, 1f0, 5_000, 5, calc_poly_feature, 1f0, Float32(num_states); α = 0.0005f0)
  ╠═╡ =#

# ╔═╡ 55ce3135-44b9-4a8d-b0e6-a8a5ec972432
#=╠═╡
plot([scatter(y = polynomial_random_walk_mc_v̂.(Float32.(1:num_states)), name = "polynomial basis"), scatter(y = random_walk_v.value_function[2:end-1], name = "true value")])
  ╠═╡ =#

# ╔═╡ 9164f6e1-5988-45f6-ac1c-2c48b303c3cd
#=╠═╡
plot(scatter(x = 1000:5000, y = smooth_error(poly_random_walk_rmse, 1000)), Layout(xaxis_title = "Episode", yaxis_title = "RMS Error Averaged over Previous 1000 Episodes"))
  ╠═╡ =#

# ╔═╡ ed00f1b2-79b0-406a-aabc-8c8c7ad61c31
md"""
### 9.5.2 Fourier Basis

With fourier features we generate the same integer vectors that we had for the polynomial basis so $(n+1)^k$ different vectors which define the different features.  The difference is that instead of exponents, these coefficients are now used to create an argument for a cosine function: $x_i(s) = \cos(\pi \boldsymbol{s}^\top \boldsymbol{c}^i)$.  For $k = 2$ and $n = 2$, the first few of these $\boldsymbol{c}$ vectors would look like: $[0, 0], [0, 1], [1, 0], \dots$.  Also, it is important for the numerical features that are the elements of $s$ be scaled between 0 and 1, so this method only works well if the numerical values of the state space fall within a known range.
"""

# ╔═╡ f1b7b56e-7701-4954-8217-1b2c7d01e309
begin
	calc_fourier_feature(s::NTuple{N, T}, min_values::NTuple{N, T}, max_values::NTuple{N, T}, c::NTuple{N, Int64}) where {T<:Real, N} = cos(Float32(π) * sum((s[i] - min_values[i])*e[i] / (max_values[i] - min_values[i]) for i in 1:N))
	calc_fourier_feature(s::T, min_value::T, max_value::T, c::NTuple{1, Int64}) where {T<:Real} = cos(Float32(π)*(s - min_value)*c[1] / (max_value - min_value))
end

# ╔═╡ 483c9b4e-bb4f-4909-aaa1-ddd00b9158dd
#=╠═╡
fourier_random_walk_mc_v̂, fourier_rmse = run_order_features_monte_carlo_policy_estimation(random_walk_state_mrp, 1f0, 5_000, 5, calc_fourier_feature, 1f0, Float32(num_states); α = 0.00005f0)
  ╠═╡ =#

# ╔═╡ 705aef3d-69dd-4ef2-ba79-9c4233bf3d73
#=╠═╡
plot([scatter(y = fourier_random_walk_mc_v̂.(Float32.(1:num_states)), name = "fourier basis"), scatter(y = random_walk_v.value_function[2:end-1], name = "true value")])
  ╠═╡ =#

# ╔═╡ 2e83b6e1-bec3-4bf7-b64e-1060d63d109c
#=╠═╡
plot(scatter(x = 1000:5000, y = smooth_error(fourier_rmse, 1000)), Layout(xaxis_title = "Episode", yaxis_title = "RMS Error Averaged over Previous 1000 Episodes"))
  ╠═╡ =#

# ╔═╡ a99ef185-0360-4005-9a8c-f10ca58babda
md"""
### 9.5.3 Coarse Coding

Coarse coding also operates in a state space where we can clearly define one or more numerical dimensions that scale over a known range of values.  Consider a number of overlapping regions in this space.  If we have N regions then that defines N binary features.  Each feature just indicates whether a state is present in that region.  Since the regions are overlapping most states will activate more than one feature.  If the regions are defined in a consistent way with a set shape and displacement vector, then each state will always activate the same number of features.  If the regions do not overlap and fully cover the state space, then this is equivalent to state aggregation where each state activates a single feature.
"""

# ╔═╡ 168e84f6-429e-45d6-bdbd-f47552fce8b5
#=╠═╡
@bind coarse_linear_display PlutoUI.combine() do Child
	md"""
	State Value: $(Child(:x, Slider(0:0.01:3; show_value=true, default = 1.5)))
	
	Zone Offset: $(Child(:offset, NumberField(0:0.1:1, default = 0.5)))
	"""
end
  ╠═╡ =#

# ╔═╡ 40f0fd57-a4ea-47a0-b883-3b038a6612c4
#=╠═╡
function show_coarse_coding_regions(x, offset_percentage)
	make_zone(offsetx, offsety) = scatter(x = [offsetx, 1+offsetx], y = [offsety, offsety], showlegend = false)
	region_starts = 0:offset_percentage:2.5
	traces = [make_zone(offsetx, offsety) for (offsetx, offsety) in zip(region_starts, region_starts)]
	state_trace = scatter(x = [x, x], y = [-1, 4], line_color = "black", mode = "lines", name = "state")

	feature_vector = Int64.([(x > a) && (x < a + 1) for a in region_starts])

	vector_string = reduce((a, b) -> "$a, $b", feature_vector)

	test = Markdown.parse(L"[%$vector_string]")
	
	md"""
	Feature Vector
	$test

	$(plot([traces; state_trace]))
	"""
end
  ╠═╡ =#

# ╔═╡ 529e262c-c94c-407b-8f13-be3b0f737e61
#=╠═╡
show_coarse_coding_regions(coarse_linear_display.x, coarse_linear_display.offset)
  ╠═╡ =#

# ╔═╡ e565c041-17bd-40c8-9240-e86931c83010
md"""
### 9.5.4 Tile Coding

Tile coding is a form of coarse coding where each state will be present in one distinct *tile* for each tiling.  A tiling is a segmentation of the state space that covers the entire space with non-overlapping regions that have no gaps.  Each tiling is thus a single instance of state aggregation.  To create multiple tilings, each tiling is shifted a set amount in each dimension of the state space to create a new set of regions shifted in position from the originals.  The shape of the tiles and the amount of offset could be different in each dimension and sometimes this asymmetry is desireable to avoid approximation artifacts such as prefered directions in the state space caused by uniform offsets.
"""

# ╔═╡ d215b917-c43d-4c14-aa97-2310f922d71a
begin
	scale_state(s::T, min_value::T, range::T) where T<:Real = (s - min_value) / range
	scale_state(s::NTuple{N, T}, min_value::NTuple{N, T}, range::NTuple{N, T}) where {N, T<:Real} = Tuple((s[i] - min_value[i])/range[i] for i in 1:k)
end

# ╔═╡ 09fb1fcd-55f9-4e04-bdb5-e5cdc649370b
begin
	#calculates which tile a state is in for the tiling represented by one offset
	function update_active_features!(feature_vector::AbstractVector{T}, scaled_state::T, offset::T, d::Int64, num_tilings::Integer, tile_size::T, num_tiles::Int64) where T<:Real
		feature_vector .= zero(T)
		for tiling in 1:num_tilings
			i = max(1, ceil(Int64, (scaled_state + offset*d*(tiling-1)) / tile_size))
			feature_vector[min(i + (tiling - 1)*num_tiles, length(feature_vector))] = one(T)
		end
		return feature_vector
	end

	function update_active_features!(feature_vector::AbstractVector{T}, scaled_state::NTuple{N, T}, offset::NTuple{N, Int64}, displacement::NTuple{N, Int64}, num_tilings::Integer, tile_size::NTuple{N, T}, num_tiles::NTuple{N, Int64}) where {N, T<:Real}
		feature_vector .= zero(T)
		total_tiles = prod(num_tiles)
		for tiling in 1:num_tilings
			index = 0
			for d in 1:N
				i = max(1, ceil(Int64, (scaled_state[i] + offset[i]*displacement[i]*(tiling - 1)) / tile_size[i]))
				index += i * (d - 1)*(num_tiles[d])
			end
			feature_vector[min(index + (tiling - 1)*total_tiles, length(feature_vector))] = one(T)
		end
		return feature_vector
	end
end

# ╔═╡ bb81db16-7c4d-4e08-bf17-45147be2b0db
function tile_coding_gradient_setup(problem::Union{StateMDP{T, S, A, P, F1, F2, F3}, StateMRP{T, S, P, F1, F2}}, min_value::S, max_value::S, tile_size::S, num_tilings::Integer, displacement_vector::Union{Int64, NTuple{N, Int64}}) where {T<:Real, N, S <: Union{T, NTuple{N, T}}, A, P, F1<:Function, F2<:Function, F3<:Function}
	#states must be tuples with k elements or some number value
	k = S == T ? 1 : N

	#ensure that all tile sizes are some percentage of the total state space
	@assert all(0 < l < 1 for l in tile_size)

	max_d = k == 1 ? displacement_vector : maximum(displacement_vector)

	s_range = if k == 1
		max_value - min_value
	else
		Tuple(max_value[i] - min_value[i] for i in 1:k)
	end

	#number of tiles in each direction of the state space
	num_tiles = if k == 1
		x = inv(tile_size)
		if isinteger(x)
			Int64(x) + 1
		else
			ceil(Int64, x)
		end
	else
		Tuple(begin
			x = inv(l)
			if isinteger(x)
				Int64(x) + 1
			else
				ceil(Int64, x)
			end
		end
		for l in tile_size)
	end

	features_per_tiling = prod(num_tiles)


	num_features = features_per_tiling*num_tilings

	feature_vector = SparseVector(zeros(T, num_features))

	#the vector representing how much each offset is shifted from the base for single unit shifts
	offset = k == 1 ? tile_size/num_tilings/max_d : Tuple(l/num_tilings/max_d for l in tile_size)

	function update_feature_vector!(x::SparseVector{T, Int64}, s::S)
		scaled_state = scale_state(s, min_value, s_range)
		update_active_features!(x, scaled_state, offset, displacement_vector, num_tilings, tile_size, num_tiles)
	end

	function get_feature_vector(s::S)
		feature_vector = SparseVector(zeros(T, num_features))
		update_feature_vector!(feature_vector, s)
		return feature_vector
	end

	output = linear_features_gradient_setup(problem, feature_vector, update_feature_vector!)
	(;output..., get_feature_vector = get_feature_vector, num_tiles = num_tiles)
end

# ╔═╡ e6514762-31e0-4916-aa21-c280674c2fc1
md"""
### *Example: 1-Dimensional Tile Coding*
"""

# ╔═╡ 84d9aac5-cf3b-402b-b222-9e8985a80b5b
#=╠═╡
@bind tile_coding_params PlutoUI.combine() do Child
	md"""
	Tile Size (% of $s_{max}$): $(Child(:tile_size, NumberField(0.01:0.01:.99, default = 0.3)))

	Number of Tilings: $(Child(:num_tilings, NumberField(1:10, default = 2)))
	
	"""
end
  ╠═╡ =#

# ╔═╡ dda74c94-3574-4e7b-bab1-d106111d36d4
#=╠═╡
tile_coding_test = tile_coding_gradient_setup(random_walk_state_mrp, 0f0, 1000f0, Float32(tile_coding_params.tile_size), tile_coding_params.num_tilings, 1)
  ╠═╡ =#

# ╔═╡ d17926d5-bcfa-4789-9609-59a69d87d194
#=╠═╡
md"""
The following shows which feature is active for each tiling in the 1 dimensional space used for the random walk example.  The tile size as a percent of the size of the state space determines how many tiles there are for each tiling.  In this case, a tile size of $(tile_coding_params.tile_size) translates into $(tile_coding_test.num_tiles) tiles.  Each of the $(tile_coding_params.num_tilings) tilings will have one of $(tile_coding_test.num_tiles) features active corresponding to which tile the state falls into.  Note that in order to cover the entire state space for each tiling, the number of tiles must overshoot the state space.  By convention the tilings will move in the negative direction of each dimension so the edge tiles must extend beyond the state space enough to still cover the space even after the shifting.
"""
  ╠═╡ =#

# ╔═╡ 71e7eef0-0304-4e26-8991-fa20da83df9a
#=╠═╡
plot(heatmap(z = reduce(hcat, tile_coding_test.get_feature_vector.(Float32.(1:num_states))), colorscale = false))
  ╠═╡ =#

# ╔═╡ 0179a9bb-0778-4220-8b13-a5297c00b763
function run_tile_coding_monte_carlo_policy_estimation(mdp::StateMDP{T, S, A, P, F1, F2, F3}, π::Function, γ::T, num_episodes::Integer, min_value::S, max_value::S, tile_size::S, num_tilings::Integer, displacement_vector::Union{Int64, NTuple{N, Int64}}; w0::T = zero(T), kwargs...) where {T<:Real, N, S<:Union{T, NTuple{N, T}}, A, P<:AbstractStateTransition, F1<:Function, F2<:Function, F3<:Function}
	setup = tile_coding_gradient_setup(mdp, min_value, max_value, tile_size, num_tilings, displacement_vector)
	params = fill(w0, length(setup.feature_vector))
	gradient_monte_carlo_estimation!(params, mdp, π, γ, num_episodes, setup.parameter_update, setup.feature_vector; kwargs...)
	v̂(s) = setup.value_function(s, params, setup.feature_vector)
	return v̂
end

# ╔═╡ fe0140fa-aba3-4338-9d19-6a591e7a95c7
function run_tile_coding_monte_carlo_estimation(mrp::StateMRP{T, S, P, F1, F2}, γ::T, num_episodes::Integer, min_value::S, max_value::S, tile_size::S, num_tilings::Integer, displacement_vector::Union{Int64, NTuple{N, Int64}}; w0::T = zero(T), kwargs...) where {T<:Real, N, S<:Union{T, NTuple{N, T}}, P<:AbstractStateTransition, F1<:Function, F2<:Function}
	setup = tile_coding_gradient_setup(mrp, min_value, max_value, tile_size, num_tilings, displacement_vector)
	params = fill(w0, length(setup.feature_vector))
	rmse_history = gradient_monte_carlo_estimation!(params, mrp, γ, num_episodes, setup.parameter_update, setup.feature_vector; kwargs...)
	v̂(s) = setup.value_function(s, params, setup.feature_vector)
	return v̂, rmse_history
end

# ╔═╡ acc3c44b-2740-4ff8-9a5d-41e4bd1d6e3e
#=╠═╡
tile_coding_mc_v̂, tile_coding_rmse = run_tile_coding_monte_carlo_estimation(random_walk_state_mrp, 1f0, 5_000, 1f0, 1_000f0, 0.2f0, 50, 1; α = 1f-4 / 50)
  ╠═╡ =#

# ╔═╡ d5d83bb4-fdbd-42f6-bc9a-14741f2786e0
#=╠═╡
plot([scatter(y = tile_coding_mc_v̂.(Float32.(1:num_states)), name = "tile coding"), scatter(y = random_walk_v.value_function[2:end-1], name = "true value")])
  ╠═╡ =#

# ╔═╡ 605a6ab5-b42a-4278-b61a-05a76bb312e3
#=╠═╡
plot(scatter(x = 1000:5000, y = smooth_error(tile_coding_rmse, 1000)), Layout(xaxis_title = "Episode", yaxis_title = "RMS Error Averaged over Previous 1000 Episodes"))
  ╠═╡ =#

# ╔═╡ a4d9efaf-1e1e-4115-973f-570014c1fd06
md"""
> ### *Exercise 9.4* 
> Suppose we believe that one of two state dimensions is more likely to have an effect on the value function than is the other, that generalization should be primarily across this dimension rather than along it.  What kind of tilings could be used to take advantage of this prior knowledge?

We could use striped tilings such that the narrow width of the tile is in the direction of the important dimension and the elongated height of the tile is in the other direction.  That way states that have the same value of the important dimension would be treated similarly regardless of their value in the other dimension.  The most rapid changes in value would occur in the direction of the important dimension.
"""

# ╔═╡ 22f6f2b1-745d-4ee5-8dfa-0fe2a61c2c54
#=╠═╡
plot([scatter(x = [a, a+2, a+2, a, a], y = [b, b, b+5, b+5, b], line_color = "blue", name = "", showlegend = false) for a in 0:2:8 for b in [0, 5]], Layout(width = 300, height = 300, margin = attr(t = 0, l = 0, r = 0, b = 0), xaxis_title = "Important Dimension", yaxis_title = "Unimportant Dimenson"))
  ╠═╡ =#

# ╔═╡ dfeead7c-65ab-4cb3-ac1c-a28a78e8448e
md"""
### 9.5.5 Radial Basis Functions
Requires much more computational complexity to tile coding without much advantage.  Also more fine tuning is required.
"""

# ╔═╡ 6beee5a8-c262-469e-9b1b-00b91e3b1b55
md"""
## 9.6 Selecting Step-Size Parameters Manually

Consider the tabular case with constant step size averaging to compute state values.  If $\alpha = 1$ (zero weight is placed on the previous estimate), then the error for that state is reduced to zero for the sampled value of that state every step.  Similarly, $\alpha = \frac{1}{10}$ implies that about ten experiences are neeed to converge approximately to their mean value.  In general tabular estimation of a state with $\alpha = \frac{1}{\tau}$ will approach the mean of its targets about $\tau$ experiences with that state.

With general function approximation there is not such a clear notion of *number* of experiences with a state; however a similar rule can be derived using feature vectors instead of states.  Suppose you wanted to learn in about $\tau$ experiences with substantially the same feature vector.  A good rule of thumb for the step-size parameter is then

$\alpha \doteq \left ( \tau \mathbb{E} \left [\boldsymbol{x}^\top \boldsymbol{x} \right ] \right ) ^{-1}$

where $\boldsymbol{x}$ is a random feature vector chosen from the same distribution as input vectors will be in the SGD.  This method words best if $\boldsymbol{x}^\top \boldsymbol{x}$ is a constant so the expected value plays no role.  Here the expected total weight on parameters that will be affected by an update replaces the value of one that was implied in the tabular case since in that case only values for individual states are updated.  In the approximation case, each feature vector represents a region of states and thus this update rule accounts for the other states that will all be affected by the update.  In the extreme case of state aggregation where each state gets its own group, then this update rule reduces to the same one from the tabular case since only one feature will be activated at a time.
"""

# ╔═╡ 858a6d4f-2241-43c3-9db0-ff9cec00c2c1
md"""
> ### *Exercise 9.5* 
> Suppose you are using tile coding to transform a seven-dimensional continuous state space into binary feature vectors to estimate a state value function $\hat v(s,\mathbf{w}) \approx v_\pi(s)$.  You believe that the dimensions do not interact strongly, so you decide to use eight tilings of each dimension separately (stripe tilings), for $7 \times 8 = 56$ tilings. In addition, in case there are some pairwise interactions between the dimensions, you also take all ${7\choose2} = 21$ pairs of dimensions and tile each pair conjunctively with rectangular tiles. You make two tilings for each pair of dimensions, making a grand total of $21 \times 2 + 56 = 98$ tilings.  Given these feature vectors, you suspect that you still have to average out some noise, so you decide that you want learning to be gradual, taking about 10 presentations with the same feature vector before learning nears its asymptote. What step-size parameter should you use? Why?

Each tiling will contribute one non-zero element to the feature vector.  With 98 tilings, we have 98 one values in each feature vector so the inner product in equation (9.19) would be $\mathbb{E}\left[\sum_{i=1}^{98} x_i^2 \right]=98$ so $\alpha=\frac{1}{10 \times 98}=\frac{1}{980} \approx 0.001$ 
	"""

# ╔═╡ be019186-33ad-4eb7-a218-9124ff40b6fb
md"""
> ### *Exercise 9.6* 
> If $\tau=1$ and $\mathbf{x}(S_t)^\top \mathbf{x}(S_t) = \mathbb{E} [\mathbf{x}^\top \mathbf{x}]$, prove that (9.19) together with (9.7) and linear function approximation results in the error being reduced to zero in one update.
"""

# ╔═╡ b447a3a9-fe35-4457-886b-05c5862ad8e0
md"""
$$\alpha \doteq \left ( \tau \mathbb{E}\left [ \boldsymbol{x}^\top \boldsymbol{x} \right ] \right ) ^{-1} \tag{9.19}$$
$$\boldsymbol{w}_{t+1} \doteq \boldsymbol{w}_t + \alpha \left [ U_t - \hat v(S_t, \boldsymbol{w}_t) \right] \nabla \hat v(S_t, \boldsymbol{w}_t) \tag{9.7}$$

Note that in the case of linear function approximation $\nabla \hat v(S_t, \boldsymbol{w}_t) = \boldsymbol{x}_t$ and $\hat v(S_t, \boldsymbol{w}_t) = \boldsymbol{x}_t^\top \boldsymbol{w}_t$ so (9.7) reduces to $\boldsymbol{w}_{t+1} \doteq \boldsymbol{w}_t + \alpha \left [ U_t - \boldsymbol{x}_t^\top \boldsymbol{w}_t \right] \boldsymbol{x}_t = \boldsymbol{w}_t(\mathbb{1} - \boldsymbol{x}_t ^\top \boldsymbol{x}_t) + \alpha U_t\boldsymbol{x}_t$ 

For the error at the state $S_t$ to be zero after this update, $\boldsymbol{x}_t^\top \boldsymbol{w}_t = U_t$

For a given time, the only parameter values that contribute to the value estimate are those for which $\boldsymbol{x}_t$ are 1.  For these indices, the contribution from the original weight vector is 0.  So $\boldsymbol{w}_{t+1} = \alpha U_t \boldsymbol{x}_t$ for indices that are updated, otherwise the values are unchanged from before.  So $\boldsymbol{x}_t^\top \boldsymbol{w}_{t+1} = \alpha U_t \boldsymbol{x}_t^\top \boldsymbol{x}_t$.  Using (9.19) with $\tau = 1$, the expected update is $\mathbb{E} [ \boldsymbol{x}_t \boldsymbol{w}_{t+1} ]  = \mathbb{E} [ \hat v(S_t, \boldsymbol{w}_{t+1})]= \mathbb{E} [U_t]$.  So the expected approximation value of the state at step t will be updated to equal the true expected value at that state.
"""

# ╔═╡ d7c1810a-8f20-4178-83ca-017d53e3e7e9
md"""
## 9.7 Nonlinear Function Approxmation: Artificial Neural Networks
"""

# ╔═╡ eca42c3b-fa09-4999-b260-c5de95c2987c
function update_nn_parameters!(θs::Vector{Matrix{Float32}}, βs::Vector{Vector{Float32}}, layers::Vector{Int64}, ∇θ::Vector{Matrix{Float32}}, ∇β::Vector{Vector{Float32}}, input::Matrix{Float32}, output::Matrix{Float32}, ∇tanh_z::Vector{Matrix{Float32}}, activations::Vector{Matrix{Float32}}, δs::Vector{Matrix{Float32}}, onesvec::Vector{Float32}, α::Float32, scales::Vector{Float32})
	input_layer_size = size(input, 2)
	FCANN.nnCostFunction(θs, βs, input_layer_size, layers, input, output, 0f0, ∇θ, ∇β, ∇tanh_z, activations, δs, onesvec; costFunc = "sqErr", resLayers = 1)
	FCANN.updateParams!(α, θs, βs, ∇θ, ∇β, scales)
	return (δs[end][1]/2)^2
end

# ╔═╡ ed115628-b644-4c5d-9bbe-0cf20bd6b5ed
function fcann_gradient_setup(problem::Union{StateMDP{T, S, A, P, F1, F2, F3}, StateMRP{T, S, P, F1, F2}}, layers::Vector{Int64}, feature_vector::Vector{Float32}, update_feature_vector!::Function) where {T<:Real, S, A, P, F1<:Function, F2<:Function, F3<:Function}
	s0 = problem.initialize_state()
	update_feature_vector!(feature_vector, s0)
	θ, β = initializeParams(length(feature_vector), layers, 1, 1, true)

	∇θ = deepcopy(θ)
	∇β = deepcopy(β)
	∇tanh_z = deepcopy(θ)
	activations = [zeros(Float32, 1, l) for l in [layers; 1]]
	δs = deepcopy(activations)
	onesvec = [0f0]

	input_layer_size = length(feature_vector)

	input = zeros(Float32, 1, input_layer_size)
	output = zeros(Float32, 1, 1)
	scales = ones(Float32, length(layers)+1)

	function update_input!(input::Matrix{Float32}, feature_vector::Vector{Float32}, num::Integer)
		for i in eachindex(feature_vector)
			input[num, i] = feature_vector[i]
		end
	end
	
	function update_parameters!(parameters, gradients, state_representation, s, g, α)
		update_feature_vector!(state_representation, s)
		update_input!(input, state_representation, 1)
		output[1, 1] = g
		update_nn_parameters!(parameters[1], parameters[2], layers, gradients[1], gradients[2], input, output, ∇tanh_z, activations, δs, onesvec, α, scales)
	end

	function v̂(parameters, s::S, state_representation) 
		update_feature_vector!(state_representation, s)
		update_input!(input, state_representation, 1)
		predict(parameters[1], parameters[2], input, 1)
	end

	function v̂(parameters, states::AbstractVector{S}, state_representation)
		input = zeros(Float32, length(states), length(state_representation))
		for i in eachindex(states)
			update_feature_vector!(state_representation, states[i])
			update_input!(input, state_representation, i)
		end
		predict(parameters[1], parameters[2], input, 1)
	end
	
	return (value_function = v̂, parameter_update = update_parameters!, parameters = (θ, β), gradients = (∇θ, ∇β), state_representation = feature_vector)
end

# ╔═╡ 0c7d2eb3-02ce-47b0-955c-fc62d5c86994
md"""
### *Nonlinear Function Approximation with Random Walk Example*
"""

# ╔═╡ 15b93928-98fb-47ed-ba46-e6ee785d46e5
function update_random_walk_vector!(feature_vector::Vector{Float32}, s::Float32)
	x1 = (s - 500f0) / sqrt(46295f0)
	feature_vector[1] = x1
end

# ╔═╡ cfc5964b-3a23-48d9-b320-861fd4a43364
function run_random_walk_fcann_monte_carlo_policy_estimation(mrp::StateMRP{T, S, P, F1, F2}, γ::T, num_episodes::Integer, layers::Vector{Int64}; kwargs...) where {T<:Real, S, P<:AbstractStateTransition{T}, F1<:Function, F2<:Function}
	setup = fcann_gradient_setup(mrp, layers, [zero(T)], update_random_walk_vector!)
	rmse = gradient_monte_carlo_estimation!(setup.parameters, mrp, γ, num_episodes, setup.parameter_update, setup.state_representation; gradients = setup.gradients, kwargs...)
	v̂(s) = setup.value_function(setup.parameters, s, setup.state_representation)
	return (v̂ = v̂, parameters = setup.parameters, error_history = rmse)
end

# ╔═╡ 3ab43d46-f171-4f3b-b788-91ebbff4420c
const nn_layers = [2, 2, 2]

# ╔═╡ e15dc0eb-9e83-4994-b953-b28c74e58030
#=╠═╡
fcann_random_walk_output = run_random_walk_fcann_monte_carlo_policy_estimation(random_walk_state_mrp, 1f0, 5_000, nn_layers; α = 5f-5)
  ╠═╡ =#

# ╔═╡ 7d78f83f-edb0-44f9-922f-ab0bc4ed668a
#=╠═╡
fcann_random_walk_output.parameters
  ╠═╡ =#

# ╔═╡ c8334c7c-7a0e-4cf4-a837-cb0404f2fe1b
#=╠═╡
plot([scatter(y = fcann_random_walk_output.v̂(Float32.(1:1000))[:], name = "neural network approximation"), scatter(y = random_walk_v.value_function[2:end-1], name = "true value")], Layout(title = "Neural Network Approximation with Layers: $nn_layers"))
  ╠═╡ =#

# ╔═╡ 6b30d3c2-0dd0-4630-ace3-1571dda25bab
#=╠═╡
plot(scatter(x = 1000:5000, y = smooth_error(fcann_random_walk_output.error_history, 1000)), Layout(xaxis_title = "Episode", yaxis_title = "RMS Error Averaged over Previous 1000 Episodes"))
  ╠═╡ =#

# ╔═╡ b227bf76-4c34-4e07-91ab-ee07ab9c5b77
md"""
In this case we have the true value reference, but what do we do in a situation where we don't have that?  We can at least measure the error history per episode through time.
"""

# ╔═╡ b22ef023-4e6a-4114-b3c2-bf91e16e9a43
md"""
## 9.8 Least-Squares TD
"""

# ╔═╡ 290200a3-7523-4e0f-bd3a-288626adaf29
md"""
## 9.9 Memory-based Function Approximation
"""

# ╔═╡ 34b78988-40f9-47e9-9c5a-7823de866b12
md"""
## 9.10 Kernel-based Function Approximation
"""

# ╔═╡ 905b032d-5fa0-4a3c-9055-fec92fd5879e
md"""
## 9.11 Looking Deeper at On-policy Learning: Interest and Emphasis
"""

# ╔═╡ 1636120f-9065-45a8-a849-731842374d60
md"""
## 9.12 Summary
"""

# ╔═╡ 022bb60c-6af7-4dd6-8410-69c7974707e8
md"""
> ### *Exercise 9.7*
> One of the simplest artificial neural networks consists of a single semi-linear unit with a logistic nonlinearity.  The need to handle approximate value functions of this form is common in games that end with either a win or a loss, in which case the value of a state can be interpreted as the probability of winning.  Derive the learning algorithm for this case, from (9.7), such that no gradient notation appears.
"""

# ╔═╡ 272c7e61-8e16-421e-9c5b-b8ee32814e6b
md"""
The logistic function is: 
$f(x) = 1 / (1 + e^{-x})$

(9.7) is:

$\boldsymbol{w}_{t+1} \doteq \boldsymbol{w}_t + \alpha [U_t - \hat v(S_t, \boldsymbol{w}_t)] \nabla \hat v(S_t, \boldsymbol{w}_t)$

For a single semi-linear unit, $\hat v(S_t, \boldsymbol{w}_t) = f(\boldsymbol{w}_t ^\top \boldsymbol{x}_t)$ where $f$ is the logistic function and $\boldsymbol{x}_t$ is the feature vector of state $S_t$ with the same length as $\boldsymbol{w}_t$.  

Also, using the finition of the logistic function:

$\begin{flalign}
f(x) &\doteq (1 + e^{-x})^{-1} \tag{1}\\
f(x)^{-1} &= 1 + e^{-x} \\
e^{-x} &= f(x)^{-1} - 1 \tag{2}\\
\end{flalign}$

Therefore, we can derive an expression for $f^\prime$ purely in terms of $f$:

$\begin{flalign}
f^\prime(x) &= -(1+e^{-x})^{-2}(-e^{-x}) \tag{chain rule} \\
&= e^{-x}(1 + e^{-x})^{-2} \\
&= f(x)^2 (f(x)^{-1} - 1) \tag{1 and 2}\\
&= f(x) (1 - f(x)) \\
\end{flalign}$

Applying to (9.7) with the chain rule and using the fact that $\nabla\boldsymbol{w}_t ^\top \boldsymbol{x}_t = \boldsymbol{x}_t$ :

$\begin{flalign}
	\boldsymbol{w}_{t+1} &\doteq \boldsymbol{w}_t + \alpha [U_t - \hat v(S_t, \boldsymbol{w}_t)] \nabla \hat v(S_t, \boldsymbol{w}_t) \\

	&= \boldsymbol{w}_t + \alpha [U_t - f(\boldsymbol{w}_t ^\top \boldsymbol{x}_t)] f(\boldsymbol{w}_t ^\top \boldsymbol{x}_t)(1-f(\boldsymbol{w}_t ^\top \boldsymbol{x}_t)) \boldsymbol{x}_t \\

\end{flalign}$
"""

# ╔═╡ 76de6624-6be3-450e-85a8-83e91af53272
md"""
> ### *Exercise 9.8*
> Arguably, the squared error used to derive (9.7) is inappropriate for the case treated in the preceding exercise, and the right error measure is the *cross-entropy loss*.  Repeat the derivation in Section 9.3, using the cross-entropy loss instead of the squared error in (9.4), all the way to an explicit form with no gradient or logarithm notation in it.  Is your final form more complex, or simpler, than you obtained in the preceding exercise?
"""

# ╔═╡ 1d3f3e29-22cc-415d-be87-2e491d0ecdf8
md"""
For a single output, the cross-entropy loss is 

$$-y \log{\hat y} - (1 - y)\log(1 - \hat y)$$ where $\hat y = f(\boldsymbol{w}_t^{\top} \boldsymbol{x}_t)$ is the approximation and $y = U_t$.  

The error for each example is then: $-U_t \log(f(\boldsymbol{w}_t^{\top} \boldsymbol{x}_t)) - (1 - U_t) \log(1 - f(\boldsymbol{w}_t^{\top} \boldsymbol{x}_t))$

where $f(x) = 1/(1 + e^{-x})$ is the logistic function

Our goal is to minimize this error over $\mu(s)$ using stochastic gradient descent, so the parameter update will be:

$\boldsymbol{w}_{t+1} \doteq \boldsymbol{w}_t - \alpha \nabla \left [-U_t \log(f(\boldsymbol{w}_t^{\top} \boldsymbol{x}_t)) - (1 - U_t) \log(1 - f(\boldsymbol{w}_t^{\top} \boldsymbol{x}_t)) \right ]$

From the previous exercise we know that $f^\prime(x) = f(x)(1-f(x))$, so applying the chain rule to the gradient gives: 

$\nabla \log(f(x)) = \nabla(x)f^\prime(x)/f(x) = (1 - f(x))\nabla(x)$

$\nabla \log(1 - f(x)) = -\nabla(x)f(x)^\prime/(1 - f(x)) = -f(x)\nabla(x)$

Using the fact that $\nabla(\boldsymbol{w}_t^{\top} \boldsymbol{x}_t) = \boldsymbol{x}_t$ So the parameter update rule can be simplified to:

$\begin{flalign}
\boldsymbol{w}_{t+1} &= \boldsymbol{w}_t - \alpha \nabla \left [-U_t \log(f(\boldsymbol{w}_t^{\top} \boldsymbol{x}_t)) - (1 - U_t) \log(1 - f(\boldsymbol{w}_t^{\top} \boldsymbol{x}_t)) \right ] \\
&= \boldsymbol{w}_t - \alpha \left [ -U_t(1-f(\boldsymbol{w}_t^{\top} \boldsymbol{x}_t)))\nabla(\boldsymbol{w}_t^{\top} \boldsymbol{x}_t)) + (1 - U_t)f(\boldsymbol{w}_t^{\top} \boldsymbol{x}_t)) \nabla(\boldsymbol{w}_t^{\top} \boldsymbol{x}_t)) \right ] \\
&= \boldsymbol{w}_t - \alpha  \left [-U_t + U_tf(\boldsymbol{w}_t^{\top} \boldsymbol{x}_t) + f(\boldsymbol{w}_t^{\top} \boldsymbol{x}_t) - U_t f(\boldsymbol{w}_t^{\top} \boldsymbol{x}_t)  \right ] \boldsymbol{x}_t \\
&= \boldsymbol{w}_t + \alpha  \left [U_t - f(\boldsymbol{w}_t^{\top} \boldsymbol{x}_t) \right ] \boldsymbol{x}_t \\
\end{flalign}$

This update rule is much simpler than the one in exercise 9.8 and is identical to the linear update rule with $\hat v = f(\boldsymbol{w}_t^{\top} \boldsymbol{x}_t)$ instead of $\hat v = \boldsymbol{w}_t^{\top} \boldsymbol{x}_t$
"""

# ╔═╡ 1a69bf65-7fa5-4ebd-b8e2-543a8e0dbf4f
cross_entropy_loss(y, ŷ) = -y*log(ŷ) - (1-y)*log(1-ŷ)

# ╔═╡ b4327edc-0677-4daf-a86d-1bcc908f2337
#=╠═╡
plot([scatter(x = LinRange(0, 1, 1000), y = cross_entropy_loss.(0, LinRange(0, 1, 1000)), name = "y is false"), scatter(x = LinRange(0, 1, 1000), y = cross_entropy_loss.(1, LinRange(0, 1, 1000)), name = "y is true")], Layout(yaxis_title = "Error", xaxis_title = "ŷ", title = "Cross Entropy Loss for a Single Output"))
  ╠═╡ =#

# ╔═╡ 5464338c-904a-4a1b-8d47-6c79da550c71
md"""
# Dependencies
"""

# ╔═╡ 507bcfda-cd09-4873-94a7-a51fefb3c25d
#=╠═╡
TableOfContents()
  ╠═╡ =#

# ╔═╡ c1488837-602d-4fbf-9d18-fba4a7fc8140
# ╠═╡ skip_as_script = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
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
PlutoUI = "~0.7.59"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.5"
manifest_format = "2.0"
project_hash = "0d18ca12e753a64defc73e431ab24249be1b6b66"

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
git-tree-sha1 = "7ae67d8567853d367e3463719356b8989e236069"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.34"

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
git-tree-sha1 = "ab55ee1510ad2af0ff674dbcced5e94921f867a9"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.59"

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
# ╟─19d23ef5-27db-44a8-99fe-a7343a5db2b8
# ╟─c4c71ace-c3a4-412b-b08b-31d246f8db5f
# ╟─cb5e302b-a14b-4135-b6ff-bee300f9dee6
# ╟─865ed63a-a7ee-403f-9004-b3ec659d756f
# ╠═be546bdb-77a9-48c4-9a98-1205d73fc8c6
# ╠═ae19496f-7d6c-4b91-8456-d7a1eacbe3d3
# ╠═7542ff9c-c6a1-4d41-8863-05388fea8ce2
# ╟─df56b803-0aa5-4946-8338-601195e57a3e
# ╠═160d1b6f-3340-4326-bfd3-c7d3f2328488
# ╠═5d90b840-4979-4e8b-bad1-68a3dc7aa392
# ╠═512f1358-0536-4d60-9ba6-173138ee6e14
# ╠═8f4c82ee-d45a-41d8-b668-234de6d85f4d
# ╟─cb2005fd-d3e0-4f37-908c-77e4bbac45b8
# ╟─90e5fc0e-2e97-424b-a5dd-9deb38293121
# ╠═de9bea60-c91d-4253-bdd8-a3c1fde8941c
# ╠═7814bda0-4306-4060-8f9a-2bcf1cf8e132
# ╠═69223862-4d74-46c9-8c78-b24d659151ac
# ╠═f4459b0d-ee3e-47c7-9c82-981af622edfa
# ╟─68a4151a-52ee-4ed0-b988-3fecc34d8d32
# ╟─24e8b391-00ec-4ed5-85dc-0796eb85bf4f
# ╟─736b7667-904d-4a9c-bb10-a6b0b831bfb6
# ╟─9c3f07b1-61eb-4d70-9dde-986c032a0840
# ╠═3f2ce7e0-b623-4ce3-90cf-949f3a6b0633
# ╠═39c6ec4d-306e-4dee-9d5a-130925341a6c
# ╠═60d68f9b-d18d-4d23-9adb-27fcb205e54b
# ╠═c79db82f-289e-4523-bf07-57cfdc38c073
# ╠═2720329c-4c80-47cb-a3e3-d24fcec6ef43
# ╟─2c6809f9-50ed-44b8-8f27-0a62e88d118c
# ╟─91e4e5da-4e0f-48b2-98bd-1e9f1330b0a8
# ╠═5ebafa8b-c316-4f95-8adc-581f2eb40e1f
# ╠═24b99200-053a-41bf-a628-0b14b807fb86
# ╟─d68c0147-a66f-4542-a395-5f9b43e16b09
# ╟─1adf0786-0897-4119-9336-09de869463b4
# ╟─b361815f-d5b0-4c71-b331-c3b48ce53e73
# ╠═c46c36f6-42da-4767-9e25-fa0ebe43998f
# ╠═47116ee6-53db-47fe-bfc9-a322f85b3e4e
# ╠═2aadb2bf-942b-436e-8b93-111a90b3ea2b
# ╟─ace0693b-b4ce-43df-966e-0330d4399638
# ╟─c0e9ea1f-8cbe-4bc1-990f-ffd3ab1989cc
# ╟─bc479ae0-78ea-4255-863f-dcd126ae9b96
# ╠═214714a5-ad1e-4439-8567-9095d10411a6
# ╠═49320a88-206e-4283-b3fc-a5d1ac41ddc4
# ╟─3160e3ec-d1b9-47ea-ad10-3d6ea40cc0b5
# ╟─6c6c0ef4-0e68-4f50-8c3a-76ed3afb2d20
# ╟─b6737cef-b6f9-4e40-82d8-bf887e17eb7c
# ╟─3db9f60e-a823-4d78-bd16-e73cedffa755
# ╠═8bd63a96-fcbe-47a8-a710-0c276586c3d6
# ╠═8ed8530f-4569-4429-92fc-3c3b1752475b
# ╟─645ba5fc-8575-4b8f-8982-f8bd20ac27ff
# ╠═6046143f-a2c3-4569-a04a-c1ad4f3daf9d
# ╠═023f0a8c-fa3c-4335-8301-6f358380fb76
# ╟─cf9d7c7d-4519-410a-8a05-af90312e291c
# ╠═c05ea239-2eea-4f41-b4e3-993db0fe2de5
# ╠═bfb1858b-5e05-4239-bcae-a3b718074630
# ╟─f5203959-29ef-406c-abac-4f01fa9630a3
# ╠═53924a3a-8fab-45c5-b6fa-90882138fac9
# ╟─c3da96b0-d584-4a43-acdb-16516e2d0452
# ╟─0ee3afe9-9c33-45c8-b304-26062675e1b8
# ╟─d65a0ca9-5577-4df8-af77-44ecfbcc0a07
# ╟─c5adf2d7-0b6b-4a87-974b-a90824d0323b
# ╠═38f09914-e128-4336-8e70-9906675971f2
# ╟─f5dea7d5-4597-430c-9020-b74cdf8f3055
# ╠═9d7ca70c-0e60-4029-8ea0-26192ccea849
# ╠═a0c4df88-ba30-463b-81ac-6f5511683730
# ╠═8c140c5e-af37-49cc-980e-96b146ebeb3c
# ╠═bc2e52ff-7f47-4141-aff1-e752fe217f6a
# ╟─c609ee03-7217-4068-9da2-c91fb02623a9
# ╠═eb8b26ed-8429-47b5-ab82-c6d79dd053e4
# ╟─55ce3135-44b9-4a8d-b0e6-a8a5ec972432
# ╟─9164f6e1-5988-45f6-ac1c-2c48b303c3cd
# ╟─ed00f1b2-79b0-406a-aabc-8c8c7ad61c31
# ╠═f1b7b56e-7701-4954-8217-1b2c7d01e309
# ╠═483c9b4e-bb4f-4909-aaa1-ddd00b9158dd
# ╟─705aef3d-69dd-4ef2-ba79-9c4233bf3d73
# ╟─2e83b6e1-bec3-4bf7-b64e-1060d63d109c
# ╟─a99ef185-0360-4005-9a8c-f10ca58babda
# ╟─168e84f6-429e-45d6-bdbd-f47552fce8b5
# ╟─529e262c-c94c-407b-8f13-be3b0f737e61
# ╠═40f0fd57-a4ea-47a0-b883-3b038a6612c4
# ╟─e565c041-17bd-40c8-9240-e86931c83010
# ╠═d215b917-c43d-4c14-aa97-2310f922d71a
# ╠═09fb1fcd-55f9-4e04-bdb5-e5cdc649370b
# ╠═bb81db16-7c4d-4e08-bf17-45147be2b0db
# ╟─e6514762-31e0-4916-aa21-c280674c2fc1
# ╟─84d9aac5-cf3b-402b-b222-9e8985a80b5b
# ╠═dda74c94-3574-4e7b-bab1-d106111d36d4
# ╟─d17926d5-bcfa-4789-9609-59a69d87d194
# ╠═71e7eef0-0304-4e26-8991-fa20da83df9a
# ╠═0179a9bb-0778-4220-8b13-a5297c00b763
# ╠═fe0140fa-aba3-4338-9d19-6a591e7a95c7
# ╠═acc3c44b-2740-4ff8-9a5d-41e4bd1d6e3e
# ╟─d5d83bb4-fdbd-42f6-bc9a-14741f2786e0
# ╠═605a6ab5-b42a-4278-b61a-05a76bb312e3
# ╟─a4d9efaf-1e1e-4115-973f-570014c1fd06
# ╟─22f6f2b1-745d-4ee5-8dfa-0fe2a61c2c54
# ╟─dfeead7c-65ab-4cb3-ac1c-a28a78e8448e
# ╟─6beee5a8-c262-469e-9b1b-00b91e3b1b55
# ╟─858a6d4f-2241-43c3-9db0-ff9cec00c2c1
# ╟─be019186-33ad-4eb7-a218-9124ff40b6fb
# ╟─b447a3a9-fe35-4457-886b-05c5862ad8e0
# ╟─d7c1810a-8f20-4178-83ca-017d53e3e7e9
# ╠═eca42c3b-fa09-4999-b260-c5de95c2987c
# ╠═ed115628-b644-4c5d-9bbe-0cf20bd6b5ed
# ╟─0c7d2eb3-02ce-47b0-955c-fc62d5c86994
# ╠═15b93928-98fb-47ed-ba46-e6ee785d46e5
# ╠═cfc5964b-3a23-48d9-b320-861fd4a43364
# ╠═3ab43d46-f171-4f3b-b788-91ebbff4420c
# ╠═7d78f83f-edb0-44f9-922f-ab0bc4ed668a
# ╠═e15dc0eb-9e83-4994-b953-b28c74e58030
# ╟─c8334c7c-7a0e-4cf4-a837-cb0404f2fe1b
# ╟─6b30d3c2-0dd0-4630-ace3-1571dda25bab
# ╟─b227bf76-4c34-4e07-91ab-ee07ab9c5b77
# ╟─b22ef023-4e6a-4114-b3c2-bf91e16e9a43
# ╟─290200a3-7523-4e0f-bd3a-288626adaf29
# ╟─34b78988-40f9-47e9-9c5a-7823de866b12
# ╟─905b032d-5fa0-4a3c-9055-fec92fd5879e
# ╟─1636120f-9065-45a8-a849-731842374d60
# ╟─022bb60c-6af7-4dd6-8410-69c7974707e8
# ╟─272c7e61-8e16-421e-9c5b-b8ee32814e6b
# ╟─76de6624-6be3-450e-85a8-83e91af53272
# ╟─1d3f3e29-22cc-415d-be87-2e491d0ecdf8
# ╟─b4327edc-0677-4daf-a86d-1bcc908f2337
# ╠═1a69bf65-7fa5-4ebd-b8e2-543a8e0dbf4f
# ╟─5464338c-904a-4a1b-8d47-6c79da550c71
# ╠═6da69e64-743f-4ea9-9670-fd023c7ffab7
# ╠═808fcb4f-f113-4623-9131-c709320130df
# ╠═5169f3c9-926b-46c2-9d1c-35cc6919f46a
# ╠═db8dd224-abf1-4a65-b8bb-e2da6ab43f7e
# ╠═507bcfda-cd09-4873-94a7-a51fefb3c25d
# ╠═c1488837-602d-4fbf-9d18-fba4a7fc8140
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
