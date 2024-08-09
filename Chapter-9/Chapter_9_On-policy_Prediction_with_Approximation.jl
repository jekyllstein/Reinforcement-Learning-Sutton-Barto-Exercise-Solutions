### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 808fcb4f-f113-4623-9131-c709320130df
using PlutoDevMacros

# ╔═╡ 5c99214c-977f-49b4-aa09-4e2ed31b15e3
@frompackage joinpath(@__DIR__, "..", "TabularRL.jl") begin
	using TabularRL
	using >.SparseArrays, >.Random, >.Statistics
end

# ╔═╡ db8dd224-abf1-4a65-b8bb-e2da6ab43f7e
# ╠═╡ skip_as_script = true
#=╠═╡
using PlutoPlotly, PlutoUI, PlutoProfile, BenchmarkTools, LaTeXStrings
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

$\overline{VE}(\boldsymbol{w}) \doteq \sum_{s \in S} \mu(s)[v_\pi(s) - \hat v(s, \boldsymbol{w})]^2$

Often $\mu(s)$ is taken to be the fraction of time spent in $s$.  In contiunuing tasks the on-policy distribution is the stationary distribution under $\pi$.  In episodic tasks one must account for the probability of starting an episode in a particular state and the probability of transitioning to that state during an episode.  The state distribution will need to depend on that function typically denoted $\eta(s)$.

An ideal goal for optimizing $\overline {VE}$ is to find a *global optimum* for the weight vector such that $\overline {VE}(\boldsymbol{w}^*) \leq \overline {VE}(\boldsymbol{w})$ for all posible weights.  Typically this isn't possible but we can find a *local optimum* but even this objective is not guaranteed for many approximation methods.  In this chapter we will focus on approximation methods based on linear gradient-descent methods to we have easily find an optimum.
"""
  ╠═╡ =#

# ╔═╡ cb5e302b-a14b-4135-b6ff-bee300f9dee6
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
## 9.3 Stochastic-gradient and Semi-gradient Methods
We will assume a weight vector with a fixed number of components $\boldsymbol{w} \dot = (w_1, w_2, \dots, w_d)$ and a differentiable value function $\hat v(s, \boldsymbol{w})$ that exists for all states.  We will update weights at each of a series of discrete time steps so we can denote $\boldsymbol{w}_t$ as the weight vector at each step.  Assume at each step we observe a state and its true value under the policy.  We assume that states appear in the same distribution $\mu$ over which we are trying to optimize the prediction objective.  Under these assumptions we can try to minimize the error observed on each example using *Stochastic gradient-descent* (SGD) by adjusting the weight vector a small amount after each observation:

$\boldsymbol{w}_{t+1} \dot = \boldsymbol{w}_t - \frac{1}{2} \alpha \nabla [v_\pi(S_t) - \hat v(S_t, \boldsymbol{w}_t)]^2$
$= \boldsymbol{w}_t + \alpha[v_\pi(S_t) - \hat v(S_t, \boldsymbol{w}_t)]\nabla\hat v(S_t, \boldsymbol{w}_t)$

where $\alpha$ is a learning rate.  In general this method will only converge to the weight vector that minimizes the error objective if $\alpha$ is sufficiently small and decreases over time.

If we do not receive the true value function at each example but rather a bootstrap approxmiation or a noise corrupted version, we can use the same formula and simply replace $v_\pi(S_t)$ with $U_t$.  As long as $U_t$ is an *unbiased* estimate for each example then the weights are still guaranteed to converge to a local optimum stochastically.  One example of an unbiased estimate would be a monte carlo sample of the discounted future return.

If we use a bootstrapped estimate of the value, then the estimate depends on the current weight vector and will no longer be *unbiased* which requires that the update target be independent of $\boldsymbol{w}_t$.  A method using bootstrapping with function approximation would be considered a *semi-gradient method* because it violates part of the convergence assumptions.  In the case of a linear function, however, they can still converge reliably.  One typical example of this is semi-gradient TD(0) learning which uses the value estimate target of $U_t \dot = R_{t+1} + \gamma \hat v(S_{t+1}, \boldsymbol{w})$.  In this case the update step for the weight vector is as follows:

$\boldsymbol{w}_{t+1} = \boldsymbol{w}_t + \alpha[R_t + \gamma \hat v(S_{t+1}, \boldsymbol{w}_t) - \hat v(S, \boldsymbol{w}_t)] \nabla \hat v(S_t, \boldsymbol{w}_t)$

*State aggregation* is a simple form of generalizing function approximation in which states are grouped together, with one estimated value (one component of the weight vector **w**) for each group.  The value of a state is estimated as its group's component, and when the state is updated, that component alone is updated.  State aggregation is a special case of SGD in which the gradient, $\nabla \hat v(S_t, \boldsymbol{w}_t)$, is 1 for the observed state's component and 0 for others.
"""
  ╠═╡ =#

# ╔═╡ 865ed63a-a7ee-403f-9004-b3ec659d756f
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
### *Gradient Monte Carlo Algorithm for Estimating $$\hat v \approx v_\pi$$*
"""
  ╠═╡ =#

# ╔═╡ be546bdb-77a9-48c4-9a98-1205d73fc8c6
function gradient_monte_carlo_episode_update!(parameters::Vector{T}, gradients::Vector{T}, ▽v̂!::Function, states::AbstractVector{S}, actions::AbstractVector{A}, rewards::AbstractVector{T}, γ::T, α::T) where {T<:Real, S, A}
	g = zero(T)
	l = length(states)
	for i in l:-1:1
		s = states[i]
		v̂ = ▽v̂!(gradients, s, parameters)
		g = γ * g + rewards[i]
		δ = g - v̂
		
		parameters .+= α .* δ .* gradients
	end
end

# ╔═╡ ae19496f-7d6c-4b91-8456-d7a1eacbe3d3
function gradient_monte_carlo_policy_estimation!(parameters::Vector{T}, mdp::StateMDP, π::Function, γ::T, num_episodes::Integer, ▽v̂!::Function; α = one(T)/10, gradients = similar(parameters), epkwargs...) where {T<:Real}
	(states, actions, rewards, _) = runepisode(mdp; π = π, epkwargs...)
	gradient_monte_carlo_episode_update!(parameters, gradients, ▽v̂!, states, actions, rewards, γ, α)
	for ep in 2:num_episodes
		(states, actions, rewards, _, n_steps) = runepisode!((states, actions, rewards), mdp; π = π, epkwargs...)
		gradient_monte_carlo_episode_update!(parameters, gradients, ▽v̂!, view(states, 1:n_steps), view(actions, 1:n_steps), view(rewards, 1:n_steps), γ, α)
	end
	return parameters
end

# ╔═╡ c46c36f6-42da-4767-9e25-fa0ebe43998f
function state_aggregation_gradient_setup(mdp::StateMDP{T, S, A, P, F1, F2, F3}, assign_state_group::Function) where {T<:Real, S, A, P<:AbstractStateMDPTransition{T}, F1<:Function, F2<:Function, F3<:Function}
	function ▽v̂!(gradients::Vector{T}, s::S, params::Vector{T})
		i = assign_state_group(s)
		v = params[i]
		gradients .= zero(T)
		gradients[i] = one(T)
		return v
	end

	v̂(w::Vector{T}, s::S) = w[assign_state_group(s)]
	
	return (value_function = v̂, gradient_update = ▽v̂!)
end

# ╔═╡ 47116ee6-53db-47fe-bfc9-a322f85b3e4e
function run_state_aggregation_monte_carlo_policy_estimation(mdp::StateMDP{T, S, A, P, F1, F2, F3}, π::Function, γ::T, num_episodes::Integer, num_groups::Integer, assign_state_group::Function; w0::T = zero(T), kwargs...) where {T<:Real, S, A, P<:AbstractStateMDPTransition{T}, F1<:Function, F2<:Function, F3<:Function}
	setup = state_aggregation_gradient_setup(mdp, assign_state_group)
	params = fill(w0, num_groups)
	gradient_monte_carlo_policy_estimation!(params, mdp, π, γ, num_episodes, setup.gradient_update; kwargs...)
	v̂(s) = setup.value_function(params, s)
	return v̂
end

# ╔═╡ cb2005fd-d3e0-4f37-908c-77e4bbac45b8
#=╠═╡
md"""
### Example 9.1: State Aggregation on the $(@bind num_states NumberField(100:100_000, default = 1000)) State Random Walk
"""
  ╠═╡ =#

# ╔═╡ 91e4e5da-4e0f-48b2-98bd-1e9f1330b0a8
#=╠═╡
md"""Number of State Aggregation Groups: $(@bind num_groups NumberField(1:num_states, default = 10))"""
  ╠═╡ =#

# ╔═╡ de9bea60-c91d-4253-bdd8-a3c1fde8941c
function make_random_walk_mdp(num_states::Integer)
	states = collect(0:num_states+1)
	state_index = TabularRL.makelookup(states)
	actions = [1]
	initial_state = ceil(Int64, num_states / 2)
	initialize_state_index() = initial_state + 1
	state_transition_map = Matrix{SparseVector{Float32, Int64}}(undef, 1, num_states+2)
	reward_transition_map = Matrix{Vector{Float32}}(undef, 1, num_states+2)
	for s in states
		if (s == 0) || (s == num_states+1)
			v = zeros(Float32, num_states+2)
			v[s+1] = 1f0
			state_transition_map[1, s+1] = SparseVector(v)
			reward_transition_map[1, s+1] = [0f0]
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
			
			state_transition_map[1, s+1] = state_transitions
	
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
			reward_transition_map[1, s+1] = reward_transitions
		end
	end
	
	TabularMDP(states, actions, TabularStochasticTransition(state_transition_map, reward_transition_map), initialize_state_index)
end

# ╔═╡ 7814bda0-4306-4060-8f9a-2bcf1cf8e132
#=╠═╡
const random_walk_tabular_mdp = make_random_walk_mdp(num_states)
  ╠═╡ =#

# ╔═╡ 68a4151a-52ee-4ed0-b988-3fecc34d8d32
#=╠═╡
md"""
#### Transition Probabilities Visualized for $num_states State Random Walk
"""
  ╠═╡ =#

# ╔═╡ 24e8b391-00ec-4ed5-85dc-0796eb85bf4f
#=╠═╡
md"""Select State to View Transition Probabilities: $(@bind smap Slider(1:num_states; default = ceil(Int64, num_states/2), show_value=true))"""
  ╠═╡ =#

# ╔═╡ 736b7667-904d-4a9c-bb10-a6b0b831bfb6
#=╠═╡
random_walk_tabular_mdp.ptf.state_transition_map[1, smap+1] |> v -> plot(bar(x = 0:num_states+1, y = v), Layout(xaxis_title = "State", yaxis_title = "Transition Probability"))
  ╠═╡ =#

# ╔═╡ 5ebafa8b-c316-4f95-8adc-581f2eb40e1f
function make_random_walk_group_assign(num_states::Integer, num_groups::Integer)
	groupsize = num_states / num_groups
	assign_group(s::Integer) = ceil(Int64, s / groupsize)
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
random_walk_group_assign.(random_walk_tabular_mdp.states) |> v -> plot(scatter(x = random_walk_tabular_mdp.states, y = v), Layout(xaxis_title = "State", yaxis_title = "Aggregation Group", title = "$num_states Random Walk States Partitioned into $num_groups Groups"))
  ╠═╡ =#

# ╔═╡ f4459b0d-ee3e-47c7-9c82-981af622edfa
#=╠═╡
const initial_state::Int64 = ceil(Int64, num_states / 2)
  ╠═╡ =#

# ╔═╡ 9c3f07b1-61eb-4d70-9dde-986c032a0840
md"""
#### Non-tabular Version of Random Walk Example
Rather than converting the tabular MDP into a non-tabular one, this construction uses a faster step function.  By default, the conversion would create a step that produces the full distribution of transition states rather than just efficiently randomly sampling from them which is achieved here by the `randomwalk_step` method.
"""

# ╔═╡ c79db82f-289e-4523-bf07-57cfdc38c073
#=╠═╡
randomwalk_state_init() = initial_state
  ╠═╡ =#

# ╔═╡ 60d68f9b-d18d-4d23-9adb-27fcb205e54b
randomwalk_isterm(s::Int64, num_states::Int64) = (s < 1) || (s > num_states)

# ╔═╡ 3f2ce7e0-b623-4ce3-90cf-949f3a6b0633
function randomwalk_step(s::Int64, i_a::Int64, num_states::Int64)
	x = ceil(Int64, rand() * 100)
	s′ = s + x * rand((-1, 1))

	r = Float32(-(s′ < 1) + (s′ > num_states))
	(r, s′)
end

# ╔═╡ 39c6ec4d-306e-4dee-9d5a-130925341a6c
#=╠═╡
const randomwalk_state_ptf = StateTransitionSampler((s, i_a) -> randomwalk_step(s, i_a, num_states), 1)
  ╠═╡ =#

# ╔═╡ 2720329c-4c80-47cb-a3e3-d24fcec6ef43
#=╠═╡
const random_walk_state_mdp = StateMDP([1], randomwalk_state_ptf, randomwalk_state_init, s -> randomwalk_isterm(s, num_states))
  ╠═╡ =#

# ╔═╡ 214714a5-ad1e-4439-8567-9095d10411a6
#=╠═╡
function figure_9_1()
	v_π = value_iteration_v(random_walk_tabular_mdp, 1f0; save_history = false).final_value[2:end-1]
	random_walk_v̂ = run_state_aggregation_monte_carlo_policy_estimation(random_walk_state_mdp, s -> 1, 1f0, 100_000, num_groups, random_walk_group_assign, α = 2f-5)
	v̂_π = random_walk_v̂.(1:num_states)
	x = 1:num_states
	n1 = L"v_\pi"
	n2 = L"\hat v"
	tr1 = scatter(x = x, y = v_π, name = "True value $n1")
	tr2 = scatter(x = x, y = v̂_π, name = "Approximate MC value $n2")
	

	state_counts = zeros(Int64, num_states)
	function update_state_counts!(state_counts, states)
		for s in states
			state_counts[s] += 1
		end
	end
	
	(states, actions, rewards, sterm, numsteps) = runepisode(random_walk_state_mdp)
	update_state_counts!(state_counts, view(states, 1:numsteps))
	for _ in 1:100_000
		(states, actions, rewards, sterm, num_steps) = runepisode!((states, actions, rewards), random_walk_state_mdp)
		update_state_counts!(state_counts, view(states, 1:num_steps))
	end
	state_distribution = state_counts ./ sum(state_counts)
	n3 = L"\mu"
	tr3 = bar(x = x, y = state_distribution, yaxis = "y2", name = "State distribution $n3", marker_color = "gray")

	plot([tr1, tr2, tr3], Layout(xaxis_title = "State", yaxis_title = "Value scale", yaxis2 = attr(title = "Distribution scale", overlaying = "y", side = "right")))
end
  ╠═╡ =#

# ╔═╡ c0e9ea1f-8cbe-4bc1-990f-ffd3ab1989cc
#=╠═╡
figure_9_1()
  ╠═╡ =#

# ╔═╡ 1e225e35-2171-4628-a2f3-e263db622511
#=╠═╡
state_aggregation_gradient_setup(random_walk_state_mdp, random_walk_group_assign)
  ╠═╡ =#

# ╔═╡ 07ec7fa3-6062-4d46-aca7-230c451eae65
#=╠═╡
const π_rand_tabular = make_random_policy(random_walk_tabular_mdp)
  ╠═╡ =#

# ╔═╡ 3f47928b-897a-4a96-9284-ddebf8f4ec76
#=╠═╡
runepisode(random_walk_tabular_mdp; π = π_rand_tabular)[1] |> plot
  ╠═╡ =#

# ╔═╡ 6726105a-15ca-4474-aa93-fa04f9a46e2d
#=╠═╡
runepisode(random_walk_state_mdp)[1] |> plot
  ╠═╡ =#

# ╔═╡ 3160e3ec-d1b9-47ea-ad10-3d6ea40cc0b5
md"""
## 9.4 Linear Methods
"""

# ╔═╡ 6c6c0ef4-0e68-4f50-8c3a-76ed3afb2d20
md"""
$\hat v(s, \mathbf{w})\dot = \mathbf{w}^\top \mathbf{x}(x) \dot =\sum_{i=1}^d w_i x_i(s)$ 

The vector $\mathbf{x}(s)$ is called a *feature vector* representing state x which is the same length as the number of parameters contained in $\mathbf{w}$.  For linear methods, features are *basis functions* because they form a linear basis for the set of approximate functions.
"""

# ╔═╡ f5203959-29ef-406c-abac-4f01fa9630a3
md"""
> *Exercise 9.1* Show that tabular methods such as presented in Part I of this book are a special case of linear function approximation.  What would the feature vectors be?

The tabular methods in Part I store a single value estimate for each state.  Such a lookup can be thought of as a function that is simply a map from each state to a value.  The feature vectors for this function would be orthanormal basis vectors of dimension matching the number of states, thus state 1 would be represented by the feature vector [1, 0, 0, ...], state 2 by [0, 1, 0, 0, ...] and so on.  The parameters would simply be the value estimate for each state.
"""

# ╔═╡ c3da96b0-d584-4a43-acdb-16516e2d0452
md"""
## 9.5 Feature Construction for Linear Methods
"""

# ╔═╡ 0ee3afe9-9c33-45c8-b304-26062675e1b8
md"""
### 9.5.1 Polynomials
"""

# ╔═╡ d65a0ca9-5577-4df8-af77-44ecfbcc0a07
md"""
> *Exercise 9.2* Why does (9.17) define $(n+1)^k$ distinct features for dimension $k$?
n represents the highest power to take for each individual dimension of the state and we consider powers from 0 up to n for each dimension.  If we list the exponent per dimension as a tuple, we have for n = 1, k = 2: (0, 0), (0, 1), (1, 0), (1, 1).
For n = 1, k = 3: (0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1).  This pattern consists of tuples of length k which can be formed by selecting from n + 1 choices of exponent.  The number of resulting tuples is $(n+1)^k$
"""

# ╔═╡ c5adf2d7-0b6b-4a87-974b-a90824d0323b
md"""
>*Exercise 9.3* What $n$ and $c_{i, j}$ produce the feature vectors $\mathbf{x}(s)=(1, s_1, s_2, s_1s_2, s_1^2, s_2^2, s_1s_2^2, s_1^2s_2, s_1^2s_2^2)^\top$

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

# ╔═╡ f5501489-46b8-4e5e-aa4f-427d8bc7a9b9
md"""
### 9.5.2 Fourier Basis
### 9.5.3 Coarse Coding
### 9.5.4 Tile Coding

> *Exercise 9.4* Suppose we believe that one of two state dimensions is more likely to have an effect on the value function than is the other, that generalization should be primarily across this dimension rather than along it.  What kind of tilings could be used to take advantage of this prior knowledge?

We could use striped tilings such that each stripe is the width of several of the important dimension but completely covers the entire space of the other dimension.  That way states that have the same value of the important dimension would be treated similarly regardless of their value in the other dimension and the overlap in the direction of the first dimension would allow some generalization if those states are close to each other along that dimension.
"""

# ╔═╡ dfeead7c-65ab-4cb3-ac1c-a28a78e8448e
md"""
### 9.5.5 Radial Basis Functions
Requires much more computational complexity to tile coding without much advantage.  Also more fine tuning is required.
"""

# ╔═╡ 6beee5a8-c262-469e-9b1b-00b91e3b1b55
md"""
## 9.6 Selecting Step-Size Parameters Manually
"""

# ╔═╡ 858a6d4f-2241-43c3-9db0-ff9cec00c2c1
md"""
> *Exercise 9.5* Suppose you are using tile coding to transform a seven-dimensional continuous state space into binary feature vectors to estimate a state value function $\hat v(s,\mathbf{w}) \approx v_\pi(s)$.  You believe that the dimensions do not interact strongly, so you decide to use eight tilings of each dimension separately (stripe tilings), for $7 \times 8 = 56$ tilings. In addition, in case there are some pairwise interactions between the dimensions, you also take all ${7\choose2} = 21$ pairs of dimensions and tile each pair conjunctively with rectangular tiles. You make two tilings for each pair of dimensions, making a grand total of $21 \times 2 + 56 = 98$ tilings.  Given these feature vectors, you suspect that you still have to average out some noise, so you decide that you want learning to be gradual, taking about 10 presentations with the same feature vector before learning nears its asymptote. What step-size parameter should you use? Why?

Each tiling will contribute one non-zero element to the feature vector.  With 98 tilings, we have 98 one values in each feature vector so the inner product in equation (9.19) would be $\mathbb{E}\left[\sum_{i=1}^{98} x_i^2 \right]=98$ so $\alpha=\frac{1}{10 \times 98}=\frac{1}{980} \approx 0.001$ 
	"""

# ╔═╡ be019186-33ad-4eb7-a218-9124ff40b6fb
md"""
> *Exercise 9.6* If $\tau=1$ and $\mathbf{x}(S_t)^\top \mathbf{x}(S_t) = \mathbb{E} [\mathbf{x}^\top \mathbf{x}]$, prove that (9.19) together with (9.7) and linear function approximation results in the error being reduced to zero in one update.
"""

# ╔═╡ 5464338c-904a-4a1b-8d47-6c79da550c71
md"""
# Dependencies
"""

# ╔═╡ 507bcfda-cd09-4873-94a7-a51fefb3c25d
#=╠═╡
TableOfContents()
  ╠═╡ =#

# ╔═╡ c1488837-602d-4fbf-9d18-fba4a7fc8140
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
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
BenchmarkTools = "~1.5.0"
LaTeXStrings = "~1.3.1"
PlutoDevMacros = "~0.9.0"
PlutoPlotly = "~0.5.0"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.59"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "77355d038513de5efcd9374a5fad72c1da9cf2f0"

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
git-tree-sha1 = "5d3a5a206297af3868151bb4a2cf27ebce46f16d"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.33"

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
version = "5.8.0+1"

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
# ╠═c46c36f6-42da-4767-9e25-fa0ebe43998f
# ╠═47116ee6-53db-47fe-bfc9-a322f85b3e4e
# ╠═c0e9ea1f-8cbe-4bc1-990f-ffd3ab1989cc
# ╠═214714a5-ad1e-4439-8567-9095d10411a6
# ╠═1e225e35-2171-4628-a2f3-e263db622511
# ╟─cb2005fd-d3e0-4f37-908c-77e4bbac45b8
# ╟─91e4e5da-4e0f-48b2-98bd-1e9f1330b0a8
# ╠═de9bea60-c91d-4253-bdd8-a3c1fde8941c
# ╠═7814bda0-4306-4060-8f9a-2bcf1cf8e132
# ╟─68a4151a-52ee-4ed0-b988-3fecc34d8d32
# ╟─24e8b391-00ec-4ed5-85dc-0796eb85bf4f
# ╠═736b7667-904d-4a9c-bb10-a6b0b831bfb6
# ╠═5ebafa8b-c316-4f95-8adc-581f2eb40e1f
# ╠═24b99200-053a-41bf-a628-0b14b807fb86
# ╟─d68c0147-a66f-4542-a395-5f9b43e16b09
# ╟─1adf0786-0897-4119-9336-09de869463b4
# ╠═f4459b0d-ee3e-47c7-9c82-981af622edfa
# ╟─9c3f07b1-61eb-4d70-9dde-986c032a0840
# ╠═2720329c-4c80-47cb-a3e3-d24fcec6ef43
# ╠═39c6ec4d-306e-4dee-9d5a-130925341a6c
# ╠═c79db82f-289e-4523-bf07-57cfdc38c073
# ╠═60d68f9b-d18d-4d23-9adb-27fcb205e54b
# ╠═3f2ce7e0-b623-4ce3-90cf-949f3a6b0633
# ╠═07ec7fa3-6062-4d46-aca7-230c451eae65
# ╠═3f47928b-897a-4a96-9284-ddebf8f4ec76
# ╠═6726105a-15ca-4474-aa93-fa04f9a46e2d
# ╟─3160e3ec-d1b9-47ea-ad10-3d6ea40cc0b5
# ╟─6c6c0ef4-0e68-4f50-8c3a-76ed3afb2d20
# ╟─f5203959-29ef-406c-abac-4f01fa9630a3
# ╟─c3da96b0-d584-4a43-acdb-16516e2d0452
# ╟─0ee3afe9-9c33-45c8-b304-26062675e1b8
# ╟─d65a0ca9-5577-4df8-af77-44ecfbcc0a07
# ╟─c5adf2d7-0b6b-4a87-974b-a90824d0323b
# ╟─f5501489-46b8-4e5e-aa4f-427d8bc7a9b9
# ╟─dfeead7c-65ab-4cb3-ac1c-a28a78e8448e
# ╟─6beee5a8-c262-469e-9b1b-00b91e3b1b55
# ╟─858a6d4f-2241-43c3-9db0-ff9cec00c2c1
# ╟─be019186-33ad-4eb7-a218-9124ff40b6fb
# ╟─5464338c-904a-4a1b-8d47-6c79da550c71
# ╠═808fcb4f-f113-4623-9131-c709320130df
# ╠═5c99214c-977f-49b4-aa09-4e2ed31b15e3
# ╠═db8dd224-abf1-4a65-b8bb-e2da6ab43f7e
# ╠═507bcfda-cd09-4873-94a7-a51fefb3c25d
# ╠═c1488837-602d-4fbf-9d18-fba4a7fc8140
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
