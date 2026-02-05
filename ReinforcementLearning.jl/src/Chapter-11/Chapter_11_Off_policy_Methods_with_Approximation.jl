### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ d992126b-9a1e-4806-b146-7c88c2ba17a2
using PlutoDevMacros

# ╔═╡ 983931c6-c60c-4884-8818-0050acc544fc
using StatsBase

# ╔═╡ 702e5559-55b0-4392-af55-846886aa1244
# ╠═╡ show_logs = false
@only_in_nb PlutoDevMacros.@frompackage @raw_str(joinpath(@__DIR__, "..", "ApproximationUtils.jl")) using ApproximationUtils

# ╔═╡ 9b35e3ae-95c4-4fe6-a84e-df4e22ab85e2
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoPlotly, PlutoUI, PlutoProfile, BenchmarkTools, LaTeXStrings, HypertextLiteral
	TableOfContents()
end
  ╠═╡ =#

# ╔═╡ c8bae838-0549-48e3-b858-0c071334c0b7
@only_in_nb begin
	include(joinpath(@__DIR__, "..", "Chapter-09", "Chapter_09_On-policy_Prediction_with_Approximation.jl"))
	include(joinpath(@__DIR__, "..", "Chapter-10", "Chapter_10_On_policy_Control_with_Approximation.jl"))
end

# ╔═╡ 46076214-2d52-4289-98e6-8b74c337f7d7
md"""
# Chapter 11: Off-policy Methods with Approximation

As we saw earlier with value function approximation, the on-policy distribution affects the results.  When we do off-policy learning, it is important to alter the sampled values so they reflect the desired distribution.  Previously we only used the importance sampling ratio to alter the values since we updated all values an arbitrarily large number of times.  Now we may need to consider also doing a transformation on the sampled states as well as the values.  However, we will also consider approaches that use true gradients with bootstrapping so we consider the effect of the parameters on the transition state value.
"""

# ╔═╡ a23b5ab9-8963-426d-9672-cf99a71d8884
md"""
## 11.1 Semi-gradient Methods

The importance sampling ratio weights samples from one distribution so sample statistics can match the target distribution.

$\rho_t \doteq \rho_{t:t} = \frac{\pi(A_t \vert S_t)}{b(A_t \vert S_t)} \tag{11.1}$

We can use the importance sampling ratio in the semi-gradient weight updates from before to try to implement an off-policy parameter update:

$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \rho_t \delta_t \nabla \hat v(S_t \vert \mathbf{w}_t) \tag{11.2}$

where $\delta_t$ is the error term depending on the target value such as the TD(0) discounted reward: 

$\delta_t \doteq R_{t+1} + \gamma \hat v (S_{t+1}, \mathbf{w}_t) - \hat v(S_t, \mathbf{w}_t) \tag{11.3}$

or the average reward for a continuing task: 

$\delta_t \doteq R_{t+1} - \bar R_t + \hat v(S_{t+1}, \mathbf{w}_t) - \hat v (S_t, \mathbf{w}_t) \tag{11.4}$

For action-values and expected updates, we do not need to use the importance sampling ratio (as we don't in Q-learning) since the bootstrap update does not depend on the actual action taken.  However, in the tabular case each state estimation was independent so this method converged even in the off policy case.  In the case of function approximation, the target value is actually the value-error which depends on the on-policy distribution.  So if the samples appear according to the off-policy distriution this method may still not converge to the correct values as we will see in some later examples.

$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t \nabla \hat q(S_t, A_t, \mathbf{w}_t) \tag{11.5}$

$\delta_t \doteq R_{t+1} + \gamma \sum_a \pi(a \vert S_{t + 1})\hat q(S_{t+1}, a, \mathbf{w}_t) - \hat q (S_t, A_t, \mathbf{w}_t) \tag{discounted expected sarsa}$

$\delta_t \doteq R_{t+1} - \bar R_t +  \sum_a \pi(a \vert S_{t + 1})\hat q(S_{t+1}, a, \mathbf{w}_t) - \hat q (S_t, A_t, \mathbf{w}_t) \tag{average reward expected sarsa}$
"""

# ╔═╡ 434045f4-865e-4993-913e-938b6cdf7a3f
md"""
> ### *Exercise 11.1* 
> Convert the equation of *n*-step off-policy TD (7.9) to the semi-gradient form.  Give accompanying definitions of the return for both the episodic and continuing cases.

$\begin{flalign}
V_{t+n}(S_t) \doteq V_{t+n-1}(S_t)+\alpha\rho_{t:t+n-1}[G_{t:t+n}-V_{t+n-1}(S_t)], \hspace{1cm}  0 \leq t < T \tag{7.9}
\end{flalign}$
where $\rho_{t:t+n-1}$, called the *importance sampling ratio*, is the relative probability under the two policies of taking the *n* actions from $A_t$ to $A_{t+n-1}$

$\rho_{t:h} \doteq \prod_{k=t}^{\min(h, T-1)}\frac{\pi(A_k | S_k)}{b(A_k|S_k)} \tag{7.10}$

To convert this to a semi-gradient method we need to provide update equations for the weight vector that defines the value function approximation.

$\begin{flalign}
\mathbf{w}_{t+n} &\doteq \mathbf{w}_{t+n-1} + \alpha \rho_{t} \cdots \rho_{t+n-1} 
[G_{t:t+n} - \hat v(S_{t}, \mathbf{w}_{t+n-1})]\nabla \hat v (S_t, \mathbf{w}_{t+n-1})\\
G_{t:t+n} &\doteq R_{t+1} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n \hat v(S_{t+n}, \mathbf{w}_{t+n-1})	\tag{episodic}\\ 
G_{t:t+n} &\doteq \sum_{k = 1}^{n-1} R_{t+k} - (n-1)\bar R_t + \hat v(S_{t+n}, \mathbf{w}_{t+n-1})\tag{continuing}
\end{flalign}$

The tablular value at a particular time step is replaced with the weight parameter at that time step with the gradient also being added next to the error term.
"""

# ╔═╡ 2c668d98-453d-482b-8980-bfbccf82dd86
md"""
> ### *Exercise 11.2* 
> Convert the equations of $n\text{-step} \; Q(\sigma)$ (7.11 and 7.17) to semi-gradient form.  Give definitions that cover both the episodic and continuing cases.

$\begin{flalign}
Q_{t+n} & \doteq Q_{t+n-1}(S_t, A_t) + \alpha \rho_{t+1:t+n}[G_{t:t+n} - Q_{t+n-1}(S_t, A_t)] \tag{7.11}\\
G_{t:h} & \doteq R_{t+1} + \gamma \left ( \sigma_{t+1}\rho_{t+1} + (1 - \sigma_{t+1} \pi(A_{t+1}|S_{t+1}) \right ) \left ( G_{t+1:h} - Q_{h-1}(S_{t+1}, A_{t+1}) \right )\\ & + \gamma \bar V_{h-1}(S_{t+1}) \tag{7.17}\\
\bar V_t(s) & \doteq \sum_a \pi(a|s)Q_t(s, a)
\end{flalign}$

To convert this to a semi-gradient method we need to provide update equations for the weight vector that defines the value function approximation.

$\begin{flalign}
\mathbf{w}_{t+n} &\doteq \mathbf{w}_{t+n-1} + \alpha \rho_{t+1} \cdots \rho_{t+n} 
[G_{t:t+n} - \hat q(S_{t}, A_t, \mathbf{w}_{t+n-1})]\nabla \hat q (S_t, A_t, \mathbf{w}_{t+n-1})\\
G_{t:h} &\doteq R_{t+1} + \gamma \left ( \sigma_{t+1}\rho_{t+1} + (1 - \sigma_{t+1} \pi(A_{t+1}|S_{t+1}) \right ) \left ( G_{t+1:h} - \hat q(S_{t+1}, A_{t+1}, \mathbf{w}_{h-1}) \right )\\ & + \gamma \overline{V}_{h-1}(S_{t+1}), \text{\; for } t < h \leq T	\tag{episodic}\\ 
G_{t:h} &\doteq R_{t+1} - \bar R_{t+1} + \gamma \left ( \sigma_{t+1}\rho_{t+1} + (1 - \sigma_{t+1} \pi(A_{t+1}|S_{t+1}) \right ) \left ( G_{t+1:h} - \hat q(S_{t+1}, A_{t+1}, \mathbf{w}_{h-1}) \right )\\ & + \gamma \bar V_{h-1}(S_{t+1}), \text{\; for } t < h \leq T \tag{continuing}\\
\overline{V}_t(s) & \doteq \sum_a \pi(a|s)\hat q(s, a, \mathbf{w}_t)
\end{flalign}$

"""

# ╔═╡ 676ea0b7-b27c-4c62-88fd-8d892b57c6b2
md"""
### *Semi-gradient Dynamic Programming Policy Evaluation*

This version of semi-gradient DP uses a function `π!` which can update the probability distribution over actions for a given policy and state.  Policy evaluation is done through trajectory sampling where the policy is used to generate the distribution of states which are updated.  All of the potential transition states are used in the bootstrap update, so enough samples need to be collected in order to visit those states.
"""

# ╔═╡ 255bb3cc-5a26-4817-b515-3b760c351f2e
function semi_gradient_dp_policy_estimation!(parameters::Q, mdp::StateMDP{T, S, A, P, F1, F2, F3}, π!::Function, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, value_function::Function, ∇v̂, update_value_gradient!::Function; α = one(T)/10, ϵ = one(T) / 10, α_decay = one(T), decay_step = typemax(Int64), save_parameter_history = false, kwargs...) where {T<:Real, S, A, P<:StateMDPTransitionDistribution, F1<:Function, F2<:Function, F3<:Function, Q}
	action_values = zeros(T, length(mdp.actions))
	policy = copy(action_values)
	
	s = mdp.initialize_state()
	
	ep = 1
	step = 1
	epreward = zero(T)
	episode_rewards = Vector{T}()
	episode_steps = Vector{Int64}()
	decay = one(T)
	parameter_history = Vector{Q}()
	save_parameter_history && push!(parameter_history, deepcopy(parameters))
	
	while (ep <= max_episodes) && (step <= max_steps)
		update_feature_vector!(feature_vector, s)
		v̂ = update_value_gradient!(∇v̂, feature_vector, parameters)
		
		#computes all of the action values for a particular MDP and an existing value estimation
		update_action_values!(action_values, s, feature_vector, update_feature_vector!, value_function, parameters, mdp, γ)
		
		#updates the policy distribution with the current state
		π!(policy, s)

		#compute the expected value target according to the distribution of transition states 
		v_target = dot(policy, action_values)

		δ = v_target - v̂

		
		update_params_with_gradient!(parameters, α*decay*δ, ∇v̂)
		save_parameter_history && push!(parameter_history, deepcopy(parameters))

		i_a = sample_action(policy)
		(r, s) = mdp.ptf(s, i_a)
		epreward += r
		
		if mdp.isterm(s)
			s = mdp.initialize_state()
			push!(episode_rewards, epreward)
			push!(episode_steps, step)
			epreward = zero(T)
			ep += 1
		end
		
		decay *= (step > decay_step)*α_decay + (step <= decay_step)
		step += 1
	end

	q̂, form_kwargs = form_value_function(mdp, γ, update_feature_vector!, value_function, feature_vector, parameters)

	return (value_function = q̂, episode_rewards = episode_rewards, episode_steps = episode_steps, parameter_history = parameter_history, final_parameters = deepcopy(parameters), form_kwargs = form_kwargs)
end

# ╔═╡ bea94375-277b-4f38-ad9d-4fa7fc646364
semi_gradient_dp_policy_estimation_linear(mdp::StateMDP, π_dist!::Function, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), parameters::Vector{T} = initialize_linear_parameters(feature_vector, init_value), kwargs...) where T<:Real = semi_gradient_dp_policy_estimation!(parameters, mdp, π_dist!, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, linear_value_function, deepcopy(feature_vector), update_linear_value_gradient!; kwargs...)

# ╔═╡ e6e606c4-39d7-4b87-bd1a-b5799281f033
md"""
## 11.2 Examples of Off-policy Divergence

With approximation, state values affect eachother which necessitated the value error objective $\overline{\text{VE}} = \sum_{s} \mu_\pi(s) \left ( \hat v_\pi(s) - v_\pi(s) \right )^2$.  Since we do not know $v_\pi$, it must be approximated somehow and the only method that generates unbiased samples is the Monte Carlo return $G_t$ since $\mathbb{E}_\pi[G_t \mid S_t = s] = v_\pi(s)$.  In continuing tasks, however, this objective is not available, so we must rely on bootstrapping in which some number of rewards are observed, but then the approximation function itself is used to complete the target value.  In Chapter 9, we saw that if we use the semi-gradient parameter update with this objective, it converges to the TD fixed point which may not equal the point of minimum value error.  As we will see later, at the TD fixed point, the projected Bellman error is 0.  All of this convergence, however, depended upon the semi-gradient updates actually using the value error objective which uses $\mu_\pi$.  If we instead try to do the same updates but sample the states differently, then the convergence guarantees from before do not apply.

Since the root of the problem with function approximation is the connection between state values, we can illustrate some pathological cases by forcing a relationship between state values that causes problems.  Consider a portion of an MDP with a single parameter $\mathbf{w}$ and two states whose approximations are $w$ and $2w$.  Furthermore, consider that from the first state, there is only one transition deterministically into the second state with a reward of 0.  If we try to estimate the state values with sampling the TD(0) error, then the target value will be $R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w}_t) = 0 + \gamma2w_t = \gamma2w_t$ and the error term $\delta_t = \gamma2w_t - w_t = w_t(2\gamma - 1)$.  So the paramter update will be:

$w_{t+1} = w_t + \alpha \rho_t \delta_t \nabla \hat v(S_t, w_t) = w_t + \alpha w_t (2\gamma - 1) w_t = w_t(1 + \alpha(2\gamma - 1))$

So the parameter update involves multiplying the value by some number: $w_{t+1} = cw_t$ and if $c>1$ for all updates then this process is unstable and the parameter will grow in magnitude indefinitely.  In this case $c = 1 + \alpha (2\gamma - 1)$ so the process is unstable when $\alpha(2\gamma - 1) \gt 0$ where $0 \leq \gamma \lt 1$.  Since $\alpha > 0$ the only condition that matters is $2\gamma - 1 \gt 0 \implies \gamma \gt 0.5$.  This assumes that $w$ will never be updated from a transition out of state 2 since that transition could occur with 0 probability under the target policy so the importance sampling ratio would be 0.  Later on we will construct an example where this exact scenario happens.  While divergence may only occur in the off-policy case, this can still result in different value estimates depending on which objective is used.
"""

# ╔═╡ 8d463e53-12ee-441c-bd14-e8b377fcdced
md"""
### Baird's Counter Example
"""

# ╔═╡ 29364905-2458-426a-999c-210cd3c60263
md"""
#### Baird Setup Functions
"""

# ╔═╡ 7b913193-0bcc-43b3-b9b2-908a9c29524e
function make_baird_ptf(n::Integer)
	state_transition_map = Matrix{SparseVector{Float32, Int64}}(undef, 2, n)
	reward_transition_map = Matrix{Vector{Float32}}(undef, 2, n)
	for i_s in 1:n
		state_transition_map[1, i_s] = sparse([fill(1f0/(n-1), n-1); 0f0])
		reward_transition_map[1, i_s] = zeros(Float32, n-1)
		state_transition_map[2, i_s] = sparse(vcat(zeros(Float32, n-1), 1f0))
		reward_transition_map[2, i_s] = [0f0]
	end

	function step(s, i_a)
		rewards = reward_transition_map[i_a, s]
		states = state_transition_map[i_a, s].nzind
		probabilities = state_transition_map[i_a, s].nzval
		(rewards, states, probabilities)
	end

	ptf1 = StateMDPTransitionDistribution(step, 1)
	ptf2 = TabularStochasticTransition(state_transition_map, reward_transition_map)
	return (tabular_ptf = ptf2, state_ptf = ptf1)
end

# ╔═╡ 822bc5a4-8aa4-454f-8da0-a1d081b93c55
function make_baird_mdps(n::Integer)
	ptfs = make_baird_ptf(n)
	states = collect(1:n)
	actions = [1, 2]
	ptfs = make_baird_ptf(n)
	tabular_mdp = TabularMDP(states, actions, ptfs.tabular_ptf)
	state_mdp =  StateMDP(actions, ptfs.state_ptf, () -> rand(1:n), s -> false)
	return (tabular_mdp = tabular_mdp, state_mdp = state_mdp)
end

# ╔═╡ 3e2afc15-0e7b-4a5d-8e38-91e70cfa87e5
# ╠═╡ skip_as_script = true
#=╠═╡
const baird_tabular_mdp, baird_state_mdp = make_baird_mdps(7)
  ╠═╡ =#

# ╔═╡ e7abb675-6697-4f23-a8b7-01eb2231f6d1
function baird_update_state_vector!(x::Vector{Float32}, s::Integer)
	x .= 0f0
	if s < 7
		x[end] = 1f0
		x[s] = 2f0
	else
		x[s] = 1f0
		x[end] = 2f0
	end
end

# ╔═╡ 4996bfd5-137c-4b31-9f80-e463ca5d2b8a
md"""
##### Baird Target Policy

This policy always takes the "solid" action which is the second action using the convention above.
"""

# ╔═╡ ddda80bc-fd6b-4110-83b3-aaf995ce8a71
function π_baird!(x::Vector{Float32}, s)
	x[1] = 0f0
	x[2] = 1f0
end

# ╔═╡ 6ae6a0c3-6ba6-4512-8ce7-ad98758f835f
md"""
##### Baird Behavior Policy

This policy takes the "dashed" action $\frac{6}{7}$ of the time in order to equalize the probability of visiting each state.
"""

# ╔═╡ 05a77bfa-2573-4b31-b108-ad4351902d11
function b_baird!(x::Vector{Float32}, s)
	x[1] = 6f0/7f0
	x[2] = inv(7f0)
end

# ╔═╡ 25572d20-89a3-4e08-948b-d678bc978b70
md"""
##### Dynamic Programming Solution to Baird Using Trajectory Sampling
"""

# ╔═╡ aca67da3-b936-4233-88ea-77987e31b90c
#=╠═╡
@bind on_policy_baird_params PlutoUI.combine() do Child
	md"""
	Select policy: $(Child(:π, Select([π_baird!, b_baird!])))
	
	Select initial value: $(Child(:w, Slider(-10:.1f0:10; show_value=true)))
	"""
end
  ╠═╡ =#

# ╔═╡ c15bedae-1231-4421-8abe-ab2fa3cb34ec
#=╠═╡
const baird_dp_result = semi_gradient_dp_policy_estimation_linear(baird_state_mdp, on_policy_baird_params.π, 0.99f0, 100, 10000, zeros(Float32, 8), baird_update_state_vector!, save_parameter_history = true, parameters = on_policy_baird_params.w .* ones(Float32, 8))
  ╠═╡ =#

# ╔═╡ ac53f74b-3909-44b7-acf2-d2dd5f2e57cc
#=╠═╡
plot([scatter(y = [x[i] for x in baird_dp_result.parameter_history], name = "weight $i") for i in 1:8], Layout(xaxis_title = "Evaluation Steps", yaxis_title = "Weight Value", title = "Baird Weights During Semi-Gradient DP Learning"))
  ╠═╡ =#

# ╔═╡ 21174c39-7b4d-48ce-80b1-a2c72be9239a
#=╠═╡
plot([scatter(y = [x[7] + 2*x[8] for x in baird_dp_result.parameter_history], name = "State value 7"), scatter(y = [2*x[1] + x[8] for x in baird_dp_result.parameter_history], name = "State value 1")], Layout(xaxis_title = "Evaluation Steps", yaxis_title = "State Value", title = "Value of Baird States 1 and 7 During Semi-gradient DP Learning"))
  ╠═╡ =#

# ╔═╡ 3c2bb063-db4f-4680-bf4a-2f3ac10a174d
md"""
Note that for the target policy, all of the time is spent in state 7, so that is the only value that is properly updated.  For the behavior policy, although the weights do not go to zero, they do find a stable solution setting all of the state values to 0.  Since this algorithm uses trajectory sampling, it is an on-policy method.  If instead we swept through the state space then we see the instability.
"""

# ╔═╡ c044414e-77d5-4a54-865e-dca4a879cd30
# ╠═╡ skip_as_script = true
#=╠═╡
function make_baird_dynamics(;nstates = 7)
	states = 1:nstates
	actions = 1:2

	n2 = nstates - 1
	#dashed action takes system to one of six upper states with equal probability 
	dash = [s′ <= n2 ? 1/n2 : 0.0 for s′ in states][:, [1]] #turn into matrix
	#solid action takes system to the 7th state
	solid = [s′ == nstates ? 1.0 : 0.0 for s′ in states][:, [1]]
	
	Dict((s, a) => a == 1 ? dash : solid for s in states for a in actions)
end
  ╠═╡ =#

# ╔═╡ d2033a7d-3d9d-4983-8fd1-b4e6ee015080
# ╠═╡ skip_as_script = true
#=╠═╡
function bairdtransition(s::Int64, a::Int64)
	if a == 1 #dashed action takes system to one of the six upper states with equal probability
		s′ = rand(1:6)
	elseif a == 2 #solid action takes the system to the 7th state
		s′ = 7
	end
	(s′, 0.0)
end
  ╠═╡ =#

# ╔═╡ 1ba56556-2ac7-4d23-98c3-0d3fb54ec3d6
# ╠═╡ skip_as_script = true
#=╠═╡
bairdbehavior(s::Int64) = [6/7, 1/7]
  ╠═╡ =#

# ╔═╡ 1be8182a-c183-486c-9991-bcc325e75449
# ╠═╡ skip_as_script = true
#=╠═╡
bairdπ(s::Int64) = [0.0, 1.0]
  ╠═╡ =#

# ╔═╡ 2feb4657-3377-434f-bf8a-400cfcfe9fef
#=╠═╡
#run the baird example with a given policy for a set number of steps and keep track of visit statistics
@tailrec function runbaird(s0::Int64, π, nsteps::Int64, counts::Vector{Int64})
	counts[s0] += 1
	nsteps == 0 && return counts ./ sum(counts)
	a = sample(1:7, pweights(π(s0)))
	(r, s) = baird_state_mdp.ptf(s0, a)
	runbaird(s, π, nsteps-1, counts)
end
  ╠═╡ =#

# ╔═╡ 3238aaa1-92aa-4d80-af22-4e237be9f0fc
#=╠═╡
function startbaird(π, nsteps)
	runbaird(1, π, nsteps, zeros(Int64, 7))
end
  ╠═╡ =#

# ╔═╡ 1e010e8e-2dde-4228-b914-fdc120fa91ca
md"""
### Long-run State Distribution
"""

# ╔═╡ 4b7b7bb6-8484-42ac-983f-ec33dbf2c73e
#=╠═╡
#confirm that the distribution of visited states is uniform for the behavior policy
plot(bar(x = 1:7, y = startbaird(bairdbehavior, 1000000)), Layout(title = "Baird Behavior Policy State Distribution"))
  ╠═╡ =#

# ╔═╡ 1074eb62-a5ee-43cb-a1a6-fe2bbc196f72
#=╠═╡
plot(bar(x = 1:7, y =startbaird(bairdπ, 1000000)), Layout(title = "Baird Target Policy State Distribution"))
  ╠═╡ =#

# ╔═╡ cad27ba6-aa01-41e2-902b-ff411037cf0f
md"""
### Semi-gradient Estimation Functions
"""

# ╔═╡ 5d85cf97-3e46-4ace-8246-2fc73a93cc2f
# ╠═╡ skip_as_script = true
#=╠═╡
abstract type MDP_Environment end
  ╠═╡ =#

# ╔═╡ 12ff2e46-fa3e-4fe8-9a1f-58afc2a43c25
#=╠═╡
#the step function must be called as follows (s′, r) = step(s, a::Int64) where s and a are the starting state and selected action while the return values are the subsequent state and reward.  a is an integer which represents which action is taken from some list of actions
struct Episodic_MDP{S} <: MDP_Environment
	states::Vector{S}
	actions
	step::Function
	sterm::S
	γ::Float64
end
  ╠═╡ =#

# ╔═╡ 84997a09-960f-4116-9045-74cb2e0e9d03
# ╠═╡ skip_as_script = true
#=╠═╡
struct Continuing_MDP{S}
	states::Vector{S}
	actions
	step::Function
end
  ╠═╡ =#

# ╔═╡ 77ca116d-675d-4db5-8a68-53d1085528f4
#=╠═╡
#step is a dictionary that maps state/action pairs to a matrix describing the distribution over possible subsequent state/reward pairs.  The value in each position is the probability of that transition occuring
struct Episodic_Full_Finite_MDP{S} <: MDP_Environment
	states::Vector{S}
	actions
	rewards::Vector{Float64}
	step::Dict{Tuple{S, Int64}, Matrix{Float64}}
	sterm::S
	γ::Float64
end
  ╠═╡ =#

# ╔═╡ 8efa076f-d14d-44ab-bc03-e7ff964bc3b3
#=╠═╡
#On Policy Episodic Semi-gradient TD0 Value Estimation
#update weight vector that act as parameters for a value function estimate and its gradient.  Weight updates will occur to optimize value function according to the policy π.  The function will modify the initially provided weight vector but also keep track of the weight vector history for the purpose of tracking progress of the value function over time
function semi_gradient_TD0_v̂!(π::Function, mdp::Episodic_MDP, v̂::Function, ∇v̂::Function, w::Vector, maxsteps::Int64; α = 0.01, maxeplength = Inf)	
	s0 = rand(mdp.states)
	w_history = [copy(w)]
	@tailrec function step!(s, nmax, eplength = 0)
		nmax == 0 && return nothing
		s == mdp.sterm && return step!(rand(mdp.states), nmax-1)
		eplength > maxeplength && return step!(rand(mdp.states), nmax)
		a = sample(mdp.states, pweights(π(s)))
		(s′, r) = mdp.step(s, a)
		δ =  r .+ (mdp.γ .* v̂(s′, w)) .- v̂(s, w) 
		w .+= α .* δ .* ∇v̂(s, w)
		push!(w_history, copy(w))
		step!(s′, nmax-1, eplength + 1)
	end
	step!(s0, maxsteps)
	return w_history
end
  ╠═╡ =#

# ╔═╡ 0b146651-a99f-489b-92f5-b5bd74d275fe
#=╠═╡
#On Policy Continuing Semi-gradient TD0 Value Estimation
#update weight vector that act as parameters for a value function estimate and its gradient.  Weight updates will occur to optimize value function according to the policy π.  The function will modify the initially provided weight vector but also keep track of the weight vector history for the purpose of tracking progress of the value function over time
function semi_gradient_TD0_v̂!(π::Function, mdp::Continuing_MDP, v̂::Function, ∇v̂::Function, w::Vector, maxsteps::Int64; alpha = 0.01, β = 0.01, r̄ = 0.0)
	s0 = rand(mdp.states)
	w_history = [copy(w)]
	@tailrec function step!(s, nmax, r̄)
		(nmax == 0) && return r̄
		a = sample(mdp.states, pweights(π(s)))
		(s′, r) = mdp.step(s, a)
		δ = r .- r̄ .+ v̂(s′, w) .- v̂(s, w)
		r̄ += β * δ
		w .+= α .* δ .* ∇v̂(s, w)
		push!(w_history, copy(w))
		step!(s′, nmax-1, r̄)
	end
	r̄ = step!(s0, maxsteps, r̄)
	return w_history, r̄
end
  ╠═╡ =#

# ╔═╡ 1853cb36-a97d-4922-92c2-02261843c761
#=╠═╡
#Off Policy Episodic Semi-gradient TD0 Value Estimation
#update weight vector that act as parameters for a value function estimate and its gradient.  Weight updates will occur to optimize value function according to the target policy π with samples drawn from the behavior policy b.  The function will modify the initially provided weight vector but also keep track of the weight vector history for the purpose of tracking progress of the value function over time
function semi_gradient_TD0_v̂!(π::Function, b::Function, mdp::Episodic_MDP, v̂::Function, ∇v̂::Function, w::Vector, maxsteps::Int64; α = 0.01)	
	s0 = rand(mdp.states)
	w_history = [copy(w)]
	@tailrec function step!(s, nmax)
		nmax == 0 && return nothing
		s == mdp.sterm && step!(rand(mdp.states), nmax-1)
		a = sample(mdp.states, pweights(b(s)))
		ρ = π(s)[a] / b(s)[a]
		(s′, r) = mdp.step(s, a)
		δ =  r .+ (mdp.γ .* v̂(s′, w)) .- v̂(s, w) 
		w .+= α .* ρ .* δ .* ∇v̂(s, w)
		push!(w_history, copy(w))
		step!(s′, nmax-1)
	end
	step!(s0, maxsteps)
	return w_history
end
  ╠═╡ =#

# ╔═╡ d1cedda0-1ebf-42a6-b2f8-7df665252c08
#=╠═╡
#On Policy Episodic Semi-gradient DP Value Estimation
#update weight vector that act as parameters for a value function estimate and its gradient.  Weight updates will occur to optimize value function according to the target policy π.  The function will modify the initially provided weight vector but also keep track of the weight vector history for the purpose of tracking progress of the value function over time
function semi_gradient_DP_v̂!(π::Function, mdp::Episodic_MDP, v̂::Function, ∇v̂::Function, w::Vector, maxsteps::Int64; α = 0.01)	
	w_history = [copy(w)]
	nactions = length(π(mdp.states[1]))
	@tailrec function step!(nmax)
		nmax == 0 && return nothing
		δ = sum(begin
			ρ = π(s)
			ℰ = sum(begin
				(s′, r) = mdp.step(s, a)
				δ = r + (mdp.γ * v̂(s′, w))
				δ * ρ[a]
			end
			for a in 1:nactions)
			#calculate expected value of delta by multiplying the discounted reward expectations by the target policy distribution and dividing by the sum in case the provided policy distribution is not normalized
			((ℰ / sum(ρ)) - v̂(s, w)) .* ∇v̂(s, w)
		end
		#note that this uniformly samples over states which effectively is doing a behavior policy with a uniform distribution rather than using μ(s).  This is fine in the non-approximate case because each state is updated independently but convergence will be worse if state visits for the policy in question doesn't match uniform.
		for s in mdp.states)
		w .+= α .* δ ./ length(mdp.states)
		push!(w_history, copy(w))
		step!(nmax-1)
	end
	step!(maxsteps)
	return w_history
end
  ╠═╡ =#

# ╔═╡ c3ad2cdc-6e85-48a7-a746-c7599f80a126
#=╠═╡
#On Policy Episodic Semi-gradient DP Value Estimation
#update weight vector that act as parameters for a value function estimate and its gradient.  Weight updates will occur to optimize value function according to the target policy π.  The function will modify the initially provided weight vector but also keep track of the weight vector history for the purpose of tracking progress of the value function over time
function semi_gradient_DP_v̂!(π::Function, mdp::Episodic_Full_Finite_MDP, v̂::Function, ∇v̂::Function, w::Vector, maxsteps::Int64; α = 0.01, μ = Dict(s => 1.0 /length(mdp.states) for s in mdp.states))	
	w_history = [copy(w)]
	nactions = length(π(mdp.states[1]))
	@tailrec function step!(nmax)
		nmax == 0 && return nothing
		v̂s = [v̂(s′, w) for s′ in mdp.states]
		δ = sum(begin
			ρ = π(s)
			ℰ = sum(begin
				δ = sum(mdp.step[(s, a)]' * (mdp.rewards' .+ v̂s .* mdp.γ))
				δ * ρ[a]
			end
			for a in 1:nactions)
			#calculate expected value of delta by multiplying the discounted reward expectations by the target policy distribution and dividing by the sum in case the provided policy distribution is not normalized
			μ[s] .* (((ℰ / sum(ρ)) - v̂(s, w)) .* ∇v̂(s, w))
		end
		#note that this uniformly samples over states which effectively is doing a behavior policy with a uniform distribution rather than using μ(s).  This is fine in the non-approximate case because each state is updated independently but convergence will be worse if state visits for the policy in question doesn't match uniform.
		for s in mdp.states)
		w .+= α .*  δ ./ sum(μ[s] for s in mdp.states)
		push!(w_history, copy(w))
		step!(nmax-1)
	end
	step!(maxsteps)
	return w_history
end
  ╠═╡ =#

# ╔═╡ ad6c8986-8fb0-4682-ade8-ebb76b4c829a
#=╠═╡
function figure11_2(;initializeweights = () -> [1., 1., 1., 1., 1., 1., 10., 1.], γ = 0.99)
	epmax = 1000
	
	bairdfeatures = [
		[2, 0, 0, 0, 0, 0, 0, 1],
		[0, 2, 0, 0, 0, 0, 0, 1],
		[0, 0, 2, 0, 0, 0, 0, 1],
		[0, 0, 0, 2, 0, 0, 0, 1],
		[0, 0, 0, 0, 2, 0, 0, 1],
		[0, 0, 0, 0, 0, 2, 0, 1],
		[0, 0, 0, 0, 0, 0, 1, 2]
	]
	
	#define value function estimator and its gradient with respect to parameters
	v̂(s, w) = w' * bairdfeatures[s]
	∇v̂(s, w) = bairdfeatures[s]

	mdp = Episodic_MDP(collect(1:7), [1, 2], bairdtransition, 0, γ)
	fullmdp = Episodic_Full_Finite_MDP(collect(1:7), [1, 2], [0.0], make_baird_dynamics(), 0, γ)

	#change the on policy state visit distribution to match the target policy.  setting x close to 1 will better match the policy which only stays in state 7
	μ_π(x) = Dict(s => s == 7 ? x : (1.0 - x) / 6 for s in 1:7)

	w_history_onpolicy = semi_gradient_TD0_v̂!(bairdπ, mdp, v̂, ∇v̂, initializeweights(), 10000, maxeplength = 1000)
	w_history_offpolicy = semi_gradient_TD0_v̂!(bairdπ, bairdbehavior, mdp, v̂, ∇v̂, initializeweights(), epmax)
	# w_history_DP = semi_gradient_DP_v̂!(bairdπ, mdp, v̂, ∇v̂, initializeweights(), epmax)
	w_history_DP = semi_gradient_DP_v̂!(bairdπ, fullmdp, v̂, ∇v̂, initializeweights(), epmax)
	w_history_DP2 = semi_gradient_DP_v̂!(bairdπ, fullmdp, v̂, ∇v̂, initializeweights(), 10000, μ = μ_π(1.0))
	
	function plot_weights(w_history, title; legend = true)
		l = length(w_history)
		traces = [scatter(x = 1:l, y = [w[i] for w in w_history], name = "w_$i") for i in 1:length(initializeweights())]
		Plot(traces, Layout(showlegend=legend, title=title, legend_orientation="h"))
	end

	calc_v̂s(w) = [v̂(s, w) for s in mdp.states]
	plotvalue(w, name) = scatter(x = 1:7, y = calc_v̂s(w), name = name)
	v_onpolicy = [v̂(s, w_history_onpolicy[end]) for s in mdp.states]
	p1 = plot_weights(w_history_onpolicy, "On Policy TD")
	p2 = plot_weights(w_history_offpolicy, "Off Policy TD")
	p3 = plot_weights(w_history_DP, "Semi-gradient DP")
	p4 = plot_weights(w_history_DP2, "Semi-gradient DP On-policy Distribution")
	valuetraces = [plotvalue(a...) for a in [(w_history_onpolicy[end], "On Policy TD"), (w_history_offpolicy[end], "Off Policy TD"), (w_history_DP[end], "Semi-gradient DP"), (w_history_DP2[end], "Semi-gradient DP On-policy Distribution")]]
	p5 = plot(valuetraces, Layout(title = "Value Estimates", xaxis_title = "State", legend_orientation="h"))
	md"""
	$(plot([p1 p2; p3 p4]))
	$p5

	Note that if we correct the dynamic programming method for the on policy distribution we recover the convergence properties of on policy TD.  However because of the target policy repeatedly visiting state 7, only the parameters for that state have a chance of being updated.  So we can expect an accurate value estimate for state 7 based on updates to weights 7 and 8 but not for the other states since weights 1 through 6 won't be affected by updates
	"""
end
  ╠═╡ =#

# ╔═╡ fcef571c-9656-42e4-9a85-e13c3ed51edb
#=╠═╡
md"""
### Figure 11.2
$(figure11_2())
"""
  ╠═╡ =#

# ╔═╡ 6965a4d3-5422-4a3e-8eba-fa101cb1b16d
md"""
### Example 11.1: Tsitsiklis and Van Roy's Counterexample
"""

# ╔═╡ d9f38410-a1e4-4e10-a16b-ee933da553d2
md"""
#### Exact Solution
Since all of the rewards are 0, this MRP is solved exactly when both state values are 0:

$v_1 = 0$
$v_2 = 0$

#### Minimum Value Error Solution
The value error is just the squared error between the approximate solution and the exact solution weighted by the on-policy distribution.  Since both state values are just multiples of $w$, the exact solution is trivial to find:  $w = 0 \implies \hat v_1 = 0, \hat v_2 = 0$

#### TD Fixed-Point Evaluation

At the TD Fixed-Point the expected TD update is 0 for both states which is given by:

State 1: 

$w_{k+1} = w_k + \alpha [ \gamma 2 w_k - w_k ] = w_k(1 + \alpha(2 \gamma - 1))$

State 2:

$w_{k+1} = w_k + \alpha 2[(1 - \epsilon) \gamma 2 w_k - 2 w_k] = w_k(1 + 4\alpha((1-\epsilon)\gamma - 1))$

Since both updates involve the learning rate times a multiple of $w_k$ then this update is only 0 when $w_k = 0$ which is the correct value.  Since this problem is solved exactly with $w = 0$, the solution that minimizes the value error matches the TD Fixed-Point.
"""

# ╔═╡ 17307c42-3175-4cfc-b9b7-e5d21e02d64a
md"""
#### Ignoring the on-policy distribution

The following weight updates are calculated to minimize the average estimation error for each transition weighted by the probability of experiencing that transition. (Note that vs equation (9.1) this is missing the on policy distribution over states).

$\begin{flalign}
w_{k+1} &= \text{argmin}_{w \in \mathbb{R}} \enspace \sum_{s \in \mathcal{S}} \left ( \hat v(s, w) - \mathbb{E}_\pi[R_{t+1} + \gamma \hat v(S_{t+1}, w_k) | S_t = s] \right )^2\\ 
&= \text{argmin}_{w \in \mathbb{R}} \enspace (w - \gamma2w_k)^2 + (2w - (1-\epsilon)\gamma2w_k)^2\\
\therefore\\
\frac{\partial{w_{k+1}}}{\partial w} &= 2(w - \gamma2w_k) + 4(2w - (1-\epsilon)\gamma2w_k) = 10w - 4\gamma w_k - 8(1-\epsilon)\gamma w_k\\
&\text{setting this equal to 0 results in }\\
10w &= 4\gamma w_k + 8(1-\epsilon)\gamma w_k = 4 \gamma w_k (3 - 2\epsilon)\\
w &= \gamma w_k \frac{4(3 - 2\epsilon)}{10} = \gamma w_k \frac{6 - 4\epsilon}{5}
\end{flalign}$

What if $$\gamma > \frac{5}{6-4\epsilon}$$?  In this case the factor multiplying $w_k$ on each update is greater than 1, thus the weight will diverge under any condition except where the initial value is 0.




We are still safe if the threshold exceeds 1 since for this problem $$\gamma \le 1$$ and we know that $0 \lt \epsilon \lt 1$ so we will never see divergence when $$5 \gt 6 - 4\epsilon \implies 4\epsilon \gt 6 - 5 \implies \epsilon \gt \frac{1}{4}$$

Indeed, in the plot below, when $\epsilon \ge \frac{1}{4}$ the $\gamma$ threshold is greater than 1 and we are guaranteed convergence.  Note that the larger the value of $\epsilon$ the closer the expected state visit counts get to each other so we approach the on policy case again.
"""

# ╔═╡ e39098da-a3df-47a0-867d-ccaf1a5a54f3
#=╠═╡
plot(scatter(x = 0:0.01:1, y = 5 ./(6 .- 4 .* (0:0.01:1))), Layout(xaxis_title = "ϵ", yaxis_title = "γ threshold", title = "γ above the blue line results in diverging weights for a given ϵ"))
  ╠═╡ =#

# ╔═╡ bd2abdf1-725a-491d-b6f3-5a15ae51762c
md"""
#### Considering the On-policy Distribution

In the first equation we didn't correctly account for the on policy distribution over states.  To calculate this we need to first get the expected value of state visits.  For simplicity assume that episodes always begin in state 1.  The number of visits to state 1 will always be 1 since there is only one transition that leaves the state permanently.  The expected number of visits to state 2 is:

$\begin{flalign}
\mathbb{E}[\eta(2)] &= \sum_{n = 1}^\infty n \Pr \{ \eta(2) = n \} \\
&= 1\epsilon + 2\epsilon(1-\epsilon) + 3\epsilon(1-\epsilon)^2 + 4\epsilon(1-\epsilon)^3 + \cdots \\
&=\epsilon\sum_{n=1}^\infty (1-\epsilon)^{n-1}n = \epsilon \frac{1}{\epsilon^2} = \frac{1}{\epsilon}
\end{flalign}$

If $\epsilon = 1$ then we only spend 1 visit in state 2 and as $\epsilon \rightarrow 0$ the system spends increasingly more time in state 2.  To get the on-policy distribution we must normalize these visit counts to get the probabilities

$\begin{flalign}
\eta(1) &= 1\\
\eta(2) &= \frac{1}{\epsilon}\\
\sum_{s}\eta(s) &= 1 + \frac{1}{\epsilon} = \frac{1+\epsilon}{\epsilon}\\
\mu(1) &= \frac{\epsilon}{1+\epsilon}\\
\mu(2) &= \frac{1}{\epsilon}\frac{\epsilon}{1+\epsilon} = \frac{1}{1+\epsilon}
\end{flalign}$
"""

# ╔═╡ 256efb33-1b85-4fbb-be51-e43384fd149c
md"""
Note that as $\epsilon \rightarrow 1$, the on policy distribution approaches the case of equal visit time.  That explains why the convergence is no longer unstable if $\epsilon$ is larger than a certain value which is $\frac{1}{4}$ in this case.
"""

# ╔═╡ c63b1e1c-db89-4d47-af39-e353dda0e50b
#=╠═╡
function plot_μ_11_1()
	μ1(ϵ) = ϵ / (1 + ϵ)
	μ2(ϵ) = 1 / (1 + ϵ)
	ϵs = 0:0.01:1
	tr1 = scatter(x = ϵs, y = μ1.(ϵs), name = "State 1")
	tr2 = scatter(x = ϵs, y = μ2.(ϵs), name = "State 2")
	plot([tr1, tr2], Layout(xaxis_title = "ϵ", yaxis_title = "Probability", title = "On-policy Distribution"))
end;
  ╠═╡ =#

# ╔═╡ 40a966dd-d8c1-486e-bed7-5a0094778f31
#=╠═╡
plot_μ_11_1()
  ╠═╡ =#

# ╔═╡ f82090ed-8b6b-4b2e-89c9-26cc0ef4b30a
#=╠═╡
md"""
Returning to the previous expression but including the on-policy distribution results in:

$\begin{flalign}
w_{k+1} &= \text{argmin}_{w \in \mathbb{R}} \enspace \sum_{s \in \mathcal{S}} \mu(s) \left ( \hat v(s, w) - \mathbb{E}_\pi[R_{t+1} + \gamma \hat v(S_{t+1}, w_k) | S_t = s] \right )^2\\ 
&= \text{argmin}_{w \in \mathbb{R}} \enspace \frac{\epsilon}{1+\epsilon} (w - \gamma2w_k)^2 + \frac{1}{1+\epsilon} (2w - (1-\epsilon)\gamma2w_k)^2\\
\therefore\\
\frac{\partial{w_{k+1}}}{\partial w} &= \frac{2}{1+\epsilon} \left [ \epsilon (w - \gamma2w_k) + 4(w - (1-\epsilon)\gamma w_k) \right ]\\
&=\frac{2}{1+\epsilon} \left [ w(\epsilon + 4) - 2\gamma w_k ( \epsilon + 2(1-\epsilon) \right ]\\
&=\frac{2}{1+\epsilon} \left [ w(\epsilon + 4) - 2\gamma w_k ( 2 - \epsilon ) \right ]\\
&\text{setting this equal to 0 and solving for w yields}\\
w &= 2 \gamma w_k \frac{2 - \epsilon}{\epsilon + 4}\\
&\text{therefore weight updates will diverge when}\\
1 &\lt 2 \gamma \frac{2 - \epsilon}{\epsilon + 4} \\
\gamma &> \frac{1}{2} \frac{\epsilon + 4}{2 - \epsilon}\\
\end{flalign}$

Since $0 \le \epsilon \le 1$ we know that $\epsilon + 4 \ge 4$ and $1 \le 2 - \epsilon \le 2$ so the fraction is greater than $\frac{4}{2} = 2$.  Therefore, this expression states that $\gamma \gt 1$ is the divergence condition which is never true.

$(plot(scatter(x = collect(0.0:0.01:1.0), y = [0.5 * (x + 4) / (2 - x) for x in 0.0:0.01:1.0]), Layout(xaxis_title = "ϵ", yaxis_title = "γ threshold")))
"""
  ╠═╡ =#

# ╔═╡ 3dade251-ddf7-463e-8d55-1c37e6d8ac9a
md"""
What if we consider the weight updates using the TD0 semi-gradient on-policy learning?

$\begin{flalign}
w_{t+1} &= w_t + \alpha(R_{t+1} + \gamma \hat v(S_{t+1}, w) - \hat v(S_t, w_t)) \nabla \hat v(S_t, w_t)\\
&\text{there are 3 different possible updates depending on the transition observed}\\
&= w_t + \alpha(0 + \gamma 2w_t - w_t) = w_t(1 + \alpha (2\gamma - 1))\\
&= w_t + 2\alpha(0 + \gamma 2w_t - 2w_t) = w_t(1 + 4\alpha (\gamma - 1))\\
&= w_t + 2\alpha(0 + \gamma 0 - 2w_t) = w_t(1 - 4\alpha)
\end{flalign}$

In this case we can see that the only update in which the weight will grow is the first one for the transition from state 1 to state 2.  So it seems that while for this counterexample dynamic programming and direct minimization fail, semi-gradient TD0 in fact can converge regardless of the value of γ and ϵ?  

Using the dynamic programming semi-gradient update yields:

$\begin{flalign}
w_{k+1} &= w_k + \alpha \sum_s \left( \mathbb{E}[R_{t+1} + \gamma \hat v(S_{t+1}, w_k) | S_t = s] - \hat v(s, w_k) \right) \nabla \hat v(s, w_k)\\ 
&= w_k + \alpha \left [ \left ( \gamma 2w_k - w_k \right) + (((1-\epsilon)(\gamma 2 w_k) + \epsilon(0)) - 2w_k)2 \right ]\\
&= w_k + \alpha [w_k(2\gamma - 1) + 2((1 - \epsilon)(2 \gamma w_k) - 2w_k)]\\
&= w_k + \alpha [w_k(2\gamma - 1) + 4w_k(\gamma - \gamma\epsilon - 1)]\\
&= w_k(1 + \alpha [2\gamma - 1 + 4\gamma - 4\gamma\epsilon - 4])\\
&= w_k(1 + \alpha [6\gamma - 5 - 4\gamma\epsilon])
\end{flalign}$

In this case we can see that if $\alpha(6\gamma - 5 - 4\gamma\epsilon)>0$ then the weights will grow indefinitely.  What does this imply about the relationship between γ and ϵ?

$\begin{flalign}
6\gamma - 5 - 4\gamma\epsilon &> 0\\
\gamma(6 - 4\epsilon) &> 5\\
\gamma &> \frac{5}{6 - 4\epsilon}
\end{flalign}$

This is the same stability condition we had before with the explicit minimization calculation.
"""

# ╔═╡ 3280e9dc-e0e4-4a18-88a5-0a4ac188e71c
#=╠═╡
function tsitsiklis_counterexample(ϵ, γ, w_0; maxsteps = 1000, α = 0.01)
	thresh = 5 / (6 - 4*ϵ)
	if γ > thresh
		println("Weights for value function approxmation will diverge with dynamic programming and direct minimization since γ > 5/(6-4ϵ)): $γ > $thresh")
		if w_0[1] == 0
			println("Since the weight is initialized at 0 it is already at the value for perfect approximation the updates will not diverge.  Any starting value other than this will have a problem though.")
		end
	else
		println("Weights for value function approxmation will NOT diverge under any method since γ < 5/(6-4ϵ)): $γ < $thresh")
	end

	states = [1, 2, 3]
	actions = [1]
	rewards = [0.0]
	ptr = Dict((s, a) => (s == 1) ? [0.0, 1.0, 0.0][:, [1]] : [0.0, 1.0 - ϵ, ϵ][:, [1]] for s in states for a in actions)
	
	function transition(s, a)
		if s == 1
			(2, 0.)
		elseif s == 2
			if rand() < ϵ
				(3, 0.)
			else
				(2, 0.)
			end
		end
	end
	
	features = [[1.], [2.], [0.0]] 
	
	#define value function estimator and its gradient with respect to parameters
	v̂(s, w) = w' * features[s]
	∇v̂(s, w) = features[s]

	mdp = Episodic_MDP(states, actions, transition, 3, γ)
	fullmdp = Episodic_Full_Finite_MDP(states, actions, rewards, ptr, 3, γ)

	#there is no meaningful action here
	π(s) = [1.]

	make_input(mdp) = (π, mdp, v̂, ∇v̂, copy(w_0), maxsteps)

	function η(s)
		if s == 1
			0.5
		elseif s == 2
			1.0 + (1. - ϵ)/ϵ
		else
			1.0
		end
	end

	μ = [η(s) for s in states] ./ sum(η(s) for s in states)

	w_history_onpolicy = semi_gradient_TD0_v̂!(make_input(mdp)..., α = α)
	w_history_DP = semi_gradient_DP_v̂!(make_input(fullmdp)..., α = α)
	w_history_DP_fixed = semi_gradient_DP_v̂!(make_input(fullmdp)..., μ = μ, α = α)

	function plot_weights(w_history, title; legend = true)
		l = length(w_history)
		traces = [scatter(x = 1:l, y = [w[i] for w in w_history], name = "w_$i") for i in 1:1]
		Plot(traces, Layout(showlegend=legend, title=title, legend_orientation="h"))
	end

	v_onpolicy = [v̂(s, w_history_onpolicy[end]) for s in mdp.states]
	p1 = plot_weights(w_history_onpolicy, "On Policy TD0", legend=false)
	p2 = plot_weights(w_history_DP, "Semi-gradient DP", legend=false)
	p3 = plot_weights(w_history_DP_fixed, "Semi-gradient DP On-policy Distribution", legend=false)
	plot([p1 p2; p3])
	# w_history_onpolicy
end
  ╠═╡ =#

# ╔═╡ 5960d4a9-5493-41d8-a98f-e9d91e34fa79
#=╠═╡
tsitsiklis_counterexample(0.001, 0.9, [0.])
  ╠═╡ =#

# ╔═╡ 14fe90c3-50a7-4098-8626-b2d2a4b617ca
#=╠═╡
tsitsiklis_counterexample(0.01, 0.5, [1.])
  ╠═╡ =#

# ╔═╡ e2751f9f-1554-4cb2-934e-0e032ad9a244
#=╠═╡
tsitsiklis_counterexample(0.01, 0.83, [1.])
  ╠═╡ =#

# ╔═╡ e28a8728-bf1d-4a94-89f3-24d15d81425a
#=╠═╡
tsitsiklis_counterexample(0.01, 0.839, [1.])
  ╠═╡ =#

# ╔═╡ fab9d8f8-8dbc-450e-8a40-7b83b5a236d0
#=╠═╡
tsitsiklis_counterexample(0.01, 0.99, [1.], maxsteps = 1000)
  ╠═╡ =#

# ╔═╡ 4965afd6-b7b9-4fa9-ad1c-9744d5b9727d
md"""
> ### *Exercise 11.3 (programming)* 
> Apply one-step semi-gradient Q-learning to Baird's counterexample and show empirically that its weights diverge.
"""

# ╔═╡ b68b6bf1-78ad-4339-a872-993e9d9fdfc2
#=╠═╡
function exercise_11_3_2(;winit = Float32.([1, 1, 1, 1, 1, 1, 10, 1]), maxsteps = 10_000, γ = 0.99f0, ϵ = 0.99f0, α = 0.01f0, state_index = 1)
	winit = Float32.([1, 1, 1, 1, 1, 1, 10, 1])
	sarsa_output = semi_gradient_sarsa_linear(baird_state_mdp, γ, 1000, maxsteps, zeros(Float32, 8), baird_update_state_vector!; ϵ = ϵ, α = α, save_parameter_history = true, parameters = [winit winit])

	q_learning_output = semi_gradient_sarsa_linear(baird_state_mdp, γ, 1000, maxsteps, zeros(Float32, 8), baird_update_state_vector!; ϵ = ϵ, α = α, save_parameter_history = true, parameters = [winit winit], compute_value = compute_q_learning_value)
	
	p1 = plot([scatter(y = [a[i, 1] for a in sarsa_output.parameter_history], name = "Parameter $i", showlegend=false) for i in 1:8], Layout(xaxis_title = "Step", yaxis_title = "Parameter Value"))
	p2 = plot([scatter(y = [a[i, 2] for a in q_learning_output.parameter_history], name = "Parameter $i") for i in 1:8])

	baird_values(w::Vector{T}) where T<:Real = [2*w[1] + w[8], 2*w[2] + w[8], 2*w[3]+w[8], 2*w[4] + w[8], 2*w[5] + w[8], 2*w[6] + w[8], w[7] + 2*w[8]]
	
	q_value_history1 = [baird_values(w[:, 1][:]) for w in q_learning_output.parameter_history]
	q_value_history2 = [baird_values(w[:, 2][:]) for w in q_learning_output.parameter_history]

	sarsa_value_history1 = [baird_values(w[:, 1][:]) for w in sarsa_output.parameter_history]
	sarsa_value_history2 = [baird_values(w[:, 2][:]) for w in sarsa_output.parameter_history]

	p3 = plot([scatter(y = [a[i] for a in q_value_history1], name = "State $i") for i in 1:7])
	p4 = plot([scatter(y = [a[i] for a in q_value_history2], name = "State $i") for i in 1:7])
	p5 = plot(scatter(y = [Float32(q_value_history1[i][state_index] - q_value_history2[i][state_index]) for i in 1:length(q_learning_output.parameter_history)]))
	p6 = plot(scatter(y = [Float32(sarsa_value_history1[i][state_index] - sarsa_value_history2[i][state_index]) for i in 1:length(sarsa_output.parameter_history)]))
	
	md"""
	$([p1 p2; p3 p4; p5 p6])
	"""
end
  ╠═╡ =#

# ╔═╡ 86c51ac5-10d7-4652-9219-d514cfe07bb6
#=╠═╡
exercise_11_3_2(;state_index = 1)
  ╠═╡ =#

# ╔═╡ 6a654e0e-2809-4e46-989f-815de38c8bf6
md"""
I applied one-step semi-gradient Q-learning to Baird's counterexample extending the feature vectors by 2 elements to represent the two actions.  After checking different intial weight vectors and ϵ values, both sarsa and q-learning seem to converge to show no preference for actions and value estimates of 0.  While the weights may diverge momentarily, after enough time steps it converges over a range of parameter values.  In the section describing the counter example it mentions that with the ϵ greedy behavior policy in Q-learning it has not been found to diverge, so I'm not sure why the weights would be expected to diverge here.  Notice that the weights do diverge at first but then begin to converge to the correct values even with $\epsilon = 0.99$.  The convergence shift seems to occur around the same time of the optimal policy changing to favor action 1 instead of action 2.    
"""

# ╔═╡ b62b78f5-4721-4fb6-b056-cc4dae9eae9f
md"""
## 11.3 The Deadly Triad
Instability and divergence arise when we combine the following three elements in solving an RL problem:

**Function approximation**
Necessary to scale up techniques to large problems where the state/action space is too large to store.

**Bootstrapping**
Important for data efficiency.  If we cannot use any bootstrapping we may need to wait and store results for very long episodes and sometimes they aren't even guaranteed to terminate.

**Off-policy training**
Often we could use Sarsa instead of Q-learning to remedy this, so avoiding off-policy training might be the best way to guarantee stability for now.  However there will be cases in the future where off-policy training might be necessary such as estimating multiple policies at once.
"""

# ╔═╡ c79e0f4d-6858-4f9c-960c-08f3c247566d
md"""
## 11.4 Linear Value-function Geometry
"""

# ╔═╡ 3bd92abe-cb9d-4e71-af82-096e6fce17a5
md"""
Consider a MRP with just two states each of which have two equally probable transitions.  The first state can transition into itself or to the second state, both with zero reward.  The second state can transition to itself or to the first state, both with a reward of 2.  We can define the MDP as a tabular problem as follows:
"""

# ╔═╡ 2dff23c5-0641-4377-8d7a-a4e2d3459b2f
# ╠═╡ skip_as_script = true
#=╠═╡
const mrp_tabular_transition = TabularStochasticTransition([sparse([0.5f0, 0.5f0]), sparse([0.5f0, 0.5f0])], [[0f0, 0f0], [2f0, 2f0]])
  ╠═╡ =#

# ╔═╡ f2bc7752-d263-4f11-afec-40f82d5188ec
# ╠═╡ skip_as_script = true
#=╠═╡
function mrp_step(s::Integer)
	if s == 1
		([0f0, 0f0], [1, 2], [0.5f0, 0.5f0])
	else
		([2f0, 2f0], [1, 2], [0.5f0, 0.5f0])
	end
end
  ╠═╡ =#

# ╔═╡ ee4ba290-9a25-44ba-abe0-44f4e39a1099
#=╠═╡
const mrp_state_transition = StateMRPTransitionDistribution(mrp_step, 1)
  ╠═╡ =#

# ╔═╡ 672c91f9-6df1-4834-a2ae-ead92d245cda
# ╠═╡ skip_as_script = true
#=╠═╡
const mrp_tabular_example = TabularMRP([1, 2], mrp_tabular_transition, () -> 1)
  ╠═╡ =#

# ╔═╡ e91cf338-fc0b-4d05-828d-6302c6acc924
# ╠═╡ skip_as_script = true
#=╠═╡
const mrp_state_example = StateMRP(mrp_state_transition, () -> 1, s -> false)
  ╠═╡ =#

# ╔═╡ b3a62cf0-f1f2-42f3-978c-14a09b20eb75
md"""
The values of this MRP can be solved exactly with the Bellman equation given a discount rate $\gamma$: $v(s) = \mathbb{E}[R_t + \gamma v(S_{t+1}) \mid S_t = s]$

$\begin{flalign}
v_1 &= \frac{1}{2} [(0 + \gamma v_2) + (0 + \gamma v_1)] = \frac{\gamma}{2}(v_1 + v_2) \\
v_2 &= \frac{1}{2} [(2 + \gamma v_1) + (2 + \gamma v_2)] \\
&= 2 + \frac{\gamma}{2}(v_1 + v_2) \\ 
&= 2 + v_1 \\
&\therefore \\
v_1 &= \frac{\gamma}{2}(v_1 + 2 + v_1) \\
&= \gamma(1 + v_1) \\
&\therefore \\
\gamma &= v_1(1 - \gamma) \implies v_1 = \frac{\gamma}{1 - \gamma}, v_2 = 2 + \frac{\gamma}{1 - \gamma} = \frac{2-\gamma}{1-\gamma} 
\end{flalign}$
"""

# ╔═╡ 677532a9-82a7-439b-b05a-013c92dd2f60
md"""
Select discount rate for MRP evaluation
"""

# ╔═╡ 5a083034-5075-46fe-a988-4dab0011c9a4
#=╠═╡
@bind γ_mrp Slider(0:0.01f0:1; default = 0.5f0, show_value = true)
  ╠═╡ =#

# ╔═╡ 3560cece-1420-41e5-8590-54041d210996
#=╠═╡
function plot_mrp_values(γ, dp_values)
	γs = 0:0.01:1
	v1(γ) = γ / (1 - γ)
	v2(γ) = 2 + v1(γ)
	tr1 = scatter(x = γs, y = v1.(γs), name = "State 1 True Value")
	tr2 = scatter(x = γs, y = v2.(γs), name = "State 2 True Value")
	tr3 = scatter(x = [γ], y = [dp_values[1]], name = "State 1 DP Value")
	tr4 = scatter(x = [γ], y = [dp_values[2]], name = "State 2 DP Value")
	plot([tr1, tr2, tr3, tr4], Layout(xaxis_title = "Discount Rate", yaxis_title = "Value", yaxis_range = [-1, 40], xaxis_range = [0, 1]))
end
  ╠═╡ =#

# ╔═╡ bfc7e5e3-2f2a-49f8-b1e8-c86e6d16b160
#=╠═╡
const mrp_dp_values = mrp_evaluation(mrp_tabular_example, γ_mrp)
  ╠═╡ =#

# ╔═╡ 94adc5c2-9ade-4b66-8814-705c2cc23534
#=╠═╡
plot_mrp_values(γ_mrp, mrp_dp_values.value_function)
  ╠═╡ =#

# ╔═╡ 88a415dd-0fe8-493e-9446-75b909b3f68c
md"""
The steady-state distribution for this MRP is also equal for both states, so each state value error should be weighted equally.  Consider now an approximate solution using only a single parameter $w$.  In general, the approximation function will depend on the feature vectors for each state but in this case that reduces to a single value.  Consider an approximation function of the following form:

$\begin{flalign}
\hat v_1 &= w \\
\hat v_2 &= cw \\
\end{flalign}$

where $c$ is some constant.  Depending on the value of $c$ and $w$, there is an infinite family of approximation functions we can visualize in the 2D plane.
"""

# ╔═╡ 88aa7985-dab6-4bd0-8685-321e1499f830
#=╠═╡
@bind mrp_test_params PlutoUI.combine() do Child
	md"""
	Discount Rate: $(Child(:γ, Slider(0:0.01:1, default = 0.8, show_value=true)))
	
	Linear Constant: $(Child(:c, Slider(0:0.01:2, default = 0.5, show_value=true)))

	Evaluation Weight: $(Child(:w, Slider(0:0.1:10, default = 5, show_value=true)))
	"""
end
  ╠═╡ =#

# ╔═╡ 9145a0ff-bfa8-47d4-91c6-118ef17da5b8
md"""
Notice there is a point where the Bellman operator produces the true value function.  At this point the Bellman Error equals the value error though neither of them are minimized.  It is unclear if other problems have a point in general for which this occurs.  The point approximation point is just something for which if the values were initialized there then one sweep of dynamic programming would produce the correct value.  Note that the projected Bellman error vector also suggests a direction of change for the parameter which is used when we do semi-gradient TD(0).  You could also project the value error and the point for which this is 0 is also the point of minimum VE.  The problem with this objective is that we cannot observe the true value.  The Bellman operator on the approximation however can always be evaluated from the environment  or if not that then at least a sample of it.  There is also a point for which the mean squared TD error is equal to the mean squared Bellman error.  We can see the two vectors that make up the TD error together are equivalent to the Bellman Operator.  With the TD error we minimize the sum of the lengths of these vectors whereas minimizing the Bellman error is minimizing the length of the sum of the vectors.  When these equal, the Bellman operator points along the line of approximation functions.  So at this particular w, the Bellman operator produces another value function which is in the approximation space.
"""

# ╔═╡ c401d8fc-704b-42e7-bbb2-0322329341fe
#=╠═╡
@bind mrp_value_params PlutoUI.combine() do Child
	md"""
	Discount Rate: $(Child(:γ, Slider(0:0.01:1, default = 0.5, show_value=true)))
	
	Linear Constant: $(Child(:c, Slider(0:0.01:10, default = 1, show_value=true)))
	"""
end
  ╠═╡ =#

# ╔═╡ 5d84f27f-4d72-46e9-99e8-2e5aa361c5f9
#could also show some kind of animation with this of a point moving towards the true value function where the value error is 0 and then how the projection works, I need to standardize the plot so it can just put all the relevant vectors for a single evaluation on like the bellman operator and the projection back.  I should maybe also show the actual gradient step that would be taken and how that's equivalent to the bellman operator projected back

#should show TDE, BE, PBE, and VE for each estimate to see when each gets minimized, also should show the magnitude of all the vectors

# ╔═╡ 860de14e-751c-483c-b570-3b1ae938a1b3
#=╠═╡
@bind mrp_bellman_iteration_params PlutoUI.combine() do Child
	md"""
	Discount Rate: $(Child(:γ, Slider(0:0.01:1, default = 0.5)))
	
	Initial Value 1: $(Child(:v1, Slider(0:0.1:10, default = 1, show_value=true)))

	Initial Value 2: $(Child(:v2, Slider(0:0.1:5, default = 1, show_value=true)))
	"""
end
  ╠═╡ =#

# ╔═╡ 6def32ca-da3d-438c-a677-40247afd2119
#add to this the TD(0) iteration with dynamic programming and approximation and see if it acts like the projected bellman error

# ╔═╡ 8a8f8290-e71d-415e-b36f-bf509163a6a6
md"""
Consider the value error under this approximation:

$\begin{flalign}
\overline{\text{VE}} &= \frac{1}{2} \left [ (\hat v_1 - v_1)^2  + (\hat v_2 - v_2)^2\right ] \\
&= \frac{1}{2}\left [\left ( w - \frac{\gamma}{1 - \gamma} \right )^2 + \left (cw - \frac{2-\gamma}{1-\gamma} \right )^2 \right ] \\
\end{flalign}$

Our goal is to find the value of $w$ that minimizes the value error and we can use gradient updates to do this with the derivative:

$\frac{\partial \overline{\text{VE}}}{\partial w} = w - \frac{\gamma}{1 - \gamma} + c^2w - c\frac{2-\gamma}{1-\gamma}$

Setting this equal to 0 implies:

$\begin{flalign}
w(1 + c^2) &= \frac{\gamma + 2c - c \gamma}{1 - \gamma} \\
&\therefore \\
w &= \frac{\gamma + 2c - c \gamma}{(1 - \gamma)(1 + c^2)} \\
&= \frac{\gamma(1 - c) + 2c}{(1 - \gamma)(1 + c^2)} \\
\end{flalign}$

"""

# ╔═╡ ac92068b-5fc7-456c-9ab1-f7a6acbe0089
# ╠═╡ skip_as_script = true
#=╠═╡
mrp_ve_min(γ, c) = (γ + 2*c - c*γ) / ((1 - γ)*(1 + c^2))
  ╠═╡ =#

# ╔═╡ d5b1580a-154d-43e7-9eb3-86ac2504e6b1
# ╠═╡ skip_as_script = true
#=╠═╡
mrp_bellman_operator(w, γ, c) = (γ*w*(1+c)/2, 2 + γ*w*(1+c)/2)
  ╠═╡ =#

# ╔═╡ 2bf91a55-b327-422d-b331-630c898dcb64
# ╠═╡ skip_as_script = true
#=╠═╡
mrp_bellman_operator2(v1, v2, γ) = (γ*(v1 + v2)/2, 2 + γ*(v1 + v2)/2)
  ╠═╡ =#

# ╔═╡ 620d9b39-cf3f-4937-b3f1-332558aef6fb
#=╠═╡
function plot_mrp_bellman_iteration(γ, v0::Tuple; maxiter = 50)
	v1s = 0:0.01:10
	v2s = 0:0.01:10
	v1 = γ / (1 - γ)
	v2 = 2 + γ / (1-γ)
	vtrue = scatter(x = [v1], y = [v2]; name = "True Value Function", mode = "markers", marker_size = 10.)

	v1_path = [v0[1]]
	v2_path = [v0[2]]
	v′ = mrp_bellman_operator2(v0..., γ)
	iter = 1
	while !isapprox(v′[1], v1) && !isapprox(v′[2], v2) && (iter < maxiter)
		push!(v1_path, v′[1])
		push!(v2_path, v′[2])
		v′ = mrp_bellman_operator2(v′..., γ)
		iter += 1
	end

	vpath = scatter(x = v1_path, y = v2_path, name = "Bellman Iteration", mode = "lines+markers", marker_size = 4.)
	
	plot([vtrue, vpath], Layout(xaxis_title = "State 1 Value", yaxis_title = "State 2 Value", title = "MRP Values for γ = $γ", xaxis_range = [0, 10], yaxis_range = [0, 5], xaxis_constrain = "domain", xaxis_scaleanchor = "y"))
end
  ╠═╡ =#

# ╔═╡ c5c814ce-7524-4742-880a-9827153b0cd2
#=╠═╡
plot_mrp_bellman_iteration(mrp_bellman_iteration_params.γ, (mrp_bellman_iteration_params.v1, mrp_bellman_iteration_params.v2))
  ╠═╡ =#

# ╔═╡ 95d37731-401e-456f-97a3-1e965cbe8b9e
# ╠═╡ skip_as_script = true
#=╠═╡
mrp_be_vector(w, w_k, γ, c) = (w - (1+c)*γ*w_k / 2, (c*w - 2 - (1+c)*γ*w_k/2))
  ╠═╡ =#

# ╔═╡ a9e39487-19d7-49ad-8536-29f79f3a80d8
# ╠═╡ skip_as_script = true
#=╠═╡
mrp_be(w, γ, c) = ((w - (1+c)*γ*w/2)^2 + (c*w - 2 - γ*w*(1+c)/2)^2)/2
  ╠═╡ =#

# ╔═╡ 3b4be6a5-e4bc-4a48-91ad-99705700b81f
md"""
We can perform the same calculation for the Bellman Error:

$\begin{flalign}
\overline{\text{BE}} &= \frac{1}{2} \left [ \left (w - \frac{\gamma w}{2} (1 + c) \right )^2 + \left (cw - 2 - \frac{\gamma w}{2}(1 + c) \right )^2 \right ]
\end{flalign}$

$\begin{flalign}
\frac{\partial{\overline{\text{BE}}}}{\partial w} &= \left [ \left (w - \frac{\gamma w(1+c)}{2} \right ) \left (1 - \frac{\gamma(1+c)}{2} \right) + \left (cw - 2 - \frac{\gamma w(1+c)}{2} \right ) \left ( c - \frac{\gamma(1+c)}{2} \right ) \right ] \\
&= \left [\frac{w(2 - \gamma(1 + c))}{2} \frac{2 - \gamma(1+c)}{2} + \frac{2cw - 4  - \gamma w (1+c)}{2} \frac{2c - \gamma(1+c)}{2}\right ] \\
&= \frac{1}{2} \left [w(2 - \gamma(1 + c))^2 + (2cw - 4  - \gamma w (1+c))(2c - \gamma(1+c)) \right ] \\
&= \frac{1}{2} \left [w((2 - \gamma(1 + c))^2 + (2c - \gamma(1+c))^2) - 4(2c - \gamma(1+c)) \right ] \\
\end{flalign}$

Setting this equal to 0 and solving for $w$ yields:

$\begin{flalign}
w &= \frac{4(2c - \gamma(1+c))}{(2 - \gamma(1 + c))^2 + (2c - \gamma(1+c))^2} \\
&= \frac{4(2c - \gamma(1+c))}{4 - 4\gamma(1+c) + \gamma^2 (1+c)^2 + 4c^2 - 4c\gamma(1+c) + \gamma^2 (1+c)^2} \\
&= \frac{4(2c - \gamma(1+c))}{4 - 4\gamma(1+c)(1 + c) + 2\gamma^2 (1+c)^2 + 4c^2} \\
&= \frac{4(2c - \gamma(1+c))}{4 + 2\gamma(1+c)^2 (\gamma - 2) + 4c^2} \\
&= \frac{2(2c - \gamma(1+c))}{2 + \gamma(1+c)^2 (\gamma - 2) + 2c^2} \\
\end{flalign}$

"""

# ╔═╡ f715250f-291b-4fae-a40e-149f88a01bfe
min_be(γ, c) = 2*(2*c - γ*(1+c)) / (2 + γ*(γ - 2)*(1+c)^2 + 2*c^2)

# ╔═╡ bcace027-418c-4d2e-beb2-cb40a5f16c22
#=╠═╡
plot(min_be.(0.5, 0:0.01:2))
  ╠═╡ =#

# ╔═╡ 39ac140a-5e2b-41fe-93e1-3612b6dd0604
md"""
In this 2D geometry with a fixed $c$, we can calculate the point on the line defining the space of possible $\hat v$ functions that is closest to any true value function vector defined by $\mathbf{v} = (v_1, v_2)$.  The squared distance from this vector to the line of possible approximation functions is $((v_1 - w)^2 + (v_2 - cw)^2)$.  The $w$ that minimizes this distance can be found by the usual process of setting the derivative with respect to $w$ to 0:

$-2(v_1 - w) - 2c(v_2 - cw) = 0 \implies 2w + 2c^2 w - 2 v_1 - 2cv_2 = 0 \implies w(1 + c^2) = v_1 + c v_2 \implies w = \frac{v_1 + cv_2}{1+c^2}$

So for any value function in the space, we can find the value $w$ that minimizes the distance to the approximation line with this formula.
"""

# ╔═╡ 5c5331ae-675a-4e07-a14f-fed84250829e
# ╠═╡ skip_as_script = true
#=╠═╡
mrp_wmin(v1, v2, c) = (v1 + c*v2) / (1 + c^2)
  ╠═╡ =#

# ╔═╡ 99f42969-f9a0-4c02-8eaf-2ae395d55147
md"""
Given two points on the approximation line defined by $w_1$ and $w_2$, the squared distance between them is just:

$(w_1 - w_2)^2 + (c w_1 - c w_2)^2 = (w_1 - w_2)^2 (1 + c)$

Using this formula we can also write down explicitely the projected Bellman Error:

$\begin{flalign}
B_\pi \hat v &= \left (\frac{\gamma w (1 + c)}{2}, \frac{4 + \gamma w (1+c)}{2} \right ) \\
w_{\text{min}} &= \frac{\gamma w (1+c) + c(4 + \gamma w (1+c))}{2(1+c^2)} \\
\vert \text{PBE} \vert &= \left ( w - \frac{\gamma w (1+c) + c(4 + \gamma w (1+c))}{2(1+c^2)} \right )^2 (1+c) \\
&= \left ( \frac{2w(1+c^2) - \gamma w (1+c) - c(4 + \gamma w (1+c))}{2(1+c^2)} \right )^2 (1+c) \\
&= \left ( 2w(1+c^2) - \gamma w (1+c) - c(4 + \gamma w (1+c)) \right )^2 \frac{(1+c)}{4(1+c^2)^2} \\
&= \left ( w(2(1+c^2) - \gamma (1+c) - c\gamma(1+c)) -4c \right )^2 \frac{(1+c)}{4(1+c^2)^2} \\
&\therefore \\
\frac{\partial \vert \text{PBE} \vert }{\partial w} &= \left ( w(2(1+c^2) - \gamma (1+c) - c\gamma(1+c)) -4c \right ) \frac{(1+c)(2(1+c^2) - \gamma (1+c) - c\gamma(1+c))}{2(1+c^2)^2} \\
\end{flalign}$

Setting this equal to 0 implies the $w$ which minimies the projected Bellman Error is:

$\begin{flalign}
w_{\text{min}} = \frac{4c}{2(1+c^2) - \gamma (1+c)^2}
\end{flalign}$
"""

# ╔═╡ f6141748-a3fd-4cc3-8296-6c311a8060cc
# ╠═╡ skip_as_script = true
#=╠═╡
mrp_pbe_min(γ, c) = 4*c / (2*(1+c^2) - γ*(1+c)^2)
  ╠═╡ =#

# ╔═╡ 4befb480-593c-4c29-adcf-3775cc3e736f
md"""
Finally, consider the *mean square TD error*

$\frac{1}{4} \left [ (\gamma w - w)^2 + (\gamma c w - w)^2 + (2 + \gamma w - cw)^2 + (2 + \gamma c w - c w)^2 \right ]$

$\frac{1}{4} \left [ w^2((\gamma -1)^2 + (\gamma c  - 1)^2) + (2 + \gamma w - cw)^2 + (2 + \gamma c w - c w)^2 \right ]$

$\frac{1}{4} \left [ w^2((\gamma -1)^2 + (\gamma c  - 1)^2) + (2 + w (\gamma - c))^2 + (2 + w c(\gamma   - 1)^2 \right ]$

$\frac{1}{4} \left [ w^2((\gamma -1)^2 + (\gamma c  - 1)^2) + 4 + 4w(\gamma - c) + w^2 (\gamma - c)^2 + 4 + 4wc(\gamma - 1) + w^2 c^2 (\gamma - 1)^2 \right ]$

$\frac{1}{4} \left [ w^2((\gamma -1)^2 + (\gamma c  - 1)^2 + (\gamma - c)^2 + c^2(\gamma - 1)^2) + 4w(\gamma - 2c + c\gamma)) + 8\right ]$

The difference between this and the Bellman error is that we take the expectation of the TD squared difference rather than the square of the expected TDE

Taking the derivative of this wrt $w$ yields

$\frac{1}{4} \left [ 2w((\gamma -1)^2 + (\gamma c  - 1)^2 + (\gamma - c)^2 + c^2(\gamma - 1)^2) + 4(\gamma - 2c + c\gamma))\right ]$

$\frac{1}{2}w((\gamma -1)^2 + (\gamma c  - 1)^2 + (\gamma - c)^2 + c^2(\gamma - 1)^2) + \gamma - 2c + c\gamma$

setting to 0 and solving for $w$

$w = \frac{2(2c - \gamma (1 + c))}{(\gamma -1)^2 + (\gamma c  - 1)^2 + (\gamma - c)^2 + c^2(\gamma - 1)^2}$
"""

# ╔═╡ 376fe140-bd40-447a-992c-97b52ffc4c2b
# ╠═╡ skip_as_script = true
#=╠═╡
mrp_tde(γ, c, w) = (w^2*((γ-1)^2 + (γ*c - 1)^2 + (γ - c)^2 + c^2 * (γ-1)^2) + 4*w*(γ-2*c + c*γ) + 8)/4
  ╠═╡ =#

# ╔═╡ 65c4f33b-0c7d-4003-9a69-9e4d1147641e
#=╠═╡
function plot_mrp_errors(γ, c, wtest)
	v1s = 0:0.01:10
	v2s = 0:0.01:10
	v1 = γ / (1 - γ)
	v2 = 2 + γ / (1-γ)
	vtrue = scatter(x = [v1], y = [v2]; name = "True Value Function", mode = "markers")
	v̂ = scatter(x = [wtest], y = [c*wtest], name = "Approximate Value Function", mode = "markers")
	ve = scatter(x = [wtest, v1], y = [c*wtest, v2], name = "Value Error", mode = "lines")

	tde1s = ((γ*wtest - wtest) / 2, (γ*c*wtest - wtest)/2)
	tde2s = ((2 + γ*c*wtest - c*wtest) / 2, (2 + γ*wtest - c*wtest)/2)
	
	
	bo_td = mrp_bellman_operator(wtest, γ, c)
	bπv̂ = scatter(x = [bo_td[1]], y = [bo_td[2]], name = "Bellman Operator on Approximation", mode = "markers")
	be = scatter(x = [wtest, bo_td[1]], y = [c*wtest, bo_td[2]], name = "Bellman Error Vector", mode = "lines")
	tde1 = scatter(x = [wtest + tde1s[1]], y = [c*wtest + tde2s[1]] , name = "TDE Target 1", mode = "markers")
	tde1_error = scatter(x = [wtest, wtest+tde1s[1]], y = [c*wtest, c*wtest + tde2s[1]], showlegend = false, mode = "lines", line_color = "gray", line_dash = "dot")
	tde2 = scatter(x = [wtest+tde1s[1]+tde1s[2]], y = [c*wtest+tde2s[1]+tde2s[2]], name = "TDE Target 2", mode = "markers")
	tde2_error = scatter(x = [wtest+tde1s[1], wtest+tde1s[1]+tde1s[2]], y = [c*wtest + tde2s[1], c*wtest+tde2s[1]+tde2s[2]], showlegend = false, mode = "lines", line_color = "gray", line_dash = "dot")
	
	w_pbe = mrp_wmin(bo_td[1], bo_td[2], c)
	pbe = scatter(x = [wtest, w_pbe], y = [c*wtest, c*w_pbe], name = "Projected Bellman Error", mode = "lines", line_color = "black")

	pbe_line = scatter(x = [w_pbe, bo_td[1]], y = [c*w_pbe, bo_td[2]], mode = "lines", line_dash = "dash", line_color = "black", showlegend = false)

	w_minve = mrp_ve_min(γ, c)

	wmin = -3
	ws = wmin:0.01:10
	v̂s = scatter(x = ws, y = c .* ws; name = "Approximate Value Functions")

	p1 = plot([vtrue, ve, v̂, bπv̂, be, v̂s, pbe, pbe_line, tde1, tde2, tde1_error, tde2_error], Layout(xaxis_title = "State 1 Value", yaxis_title = "State 2 Value", title = "MRP Values for γ = $γ and c = $c, <br> w that minimizes value error = $(round(w_minve, sigdigits = 4))", xaxis_range = [0, 8], yaxis_range = [0, 10], xaxis_constrain = "domain", yaxis_scaleanchor = "x", height = 800, legend_orientation = "h"))

	calc_ve(w) = ((w - v1)^2 + (c*w - v2)^2)/2
	calc_error(v1, v2) = ((v1[1] - v2[1])^2 + (v1[2] - v2[2])^2)/2
	function calc_pbe(w)
		bo = mrp_bellman_operator(w, γ, c)
		w_pbe = mrp_wmin(bo[1], bo[2], c)
		calc_error([w, c*w], [w_pbe, c*w_pbe])
	end
	ves = calc_ve.(ws)
	bes = [calc_error(mrp_bellman_operator(w, γ, c), [w, c*w]) for w in ws]
	pbes = [calc_pbe(w) for w in ws]
	tdes = mrp_tde.(γ, c, ws)
	ve_tr = scatter(x = ws, y = sqrt.(ves), name = "RMS Value Errors")
	be_tr = scatter(x = ws, y = sqrt.(bes), name = "RMS Bellman Errors")
	pbe_tr = scatter(x = ws, y = sqrt.(pbes), name = "RMS Projected Bellman Errors")
	tde_tr = scatter(x = ws, y = sqrt.(tdes), name = "RMS TD Errors")
	ve_point = scatter(x = [wtest], y = [sqrt(calc_ve(wtest))], mode = "markers", name = "Value Error at w = $wtest")
	be_point = scatter(x = [wtest], y = [sqrt(calc_error(mrp_bellman_operator(wtest, γ, c), [wtest, c*wtest]))], mode = "markers", name = "Bellman Error at w = $wtest")
	pbe_point = scatter(x = [wtest], y = [sqrt(calc_pbe(wtest))], mode = "markers", name = "Projected Bellman Error at w = $wtest")
	p2 = plot([ve_tr, ve_point, be_tr, be_point, pbe_tr, pbe_point, tde_tr], Layout(xaxis_range = [wmin, 10], yaxis_range = [0, 10], legend_orientation = "h", height = 800))

	@htl("""
	<div style = "display: flex;">
	$p2 
	$p1
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ d90dcfef-325c-4227-84a9-671f01b7383a
#=╠═╡
plot_mrp_errors(mrp_test_params...)
  ╠═╡ =#

# ╔═╡ 9be0a35f-bf46-4edd-be72-cd92a76822da
# ╠═╡ skip_as_script = true
#=╠═╡
mrp_wmin_tde(γ, c) = 2(2c - γ*(1+c)) / ((γ-1)^2 + (γ*c - 1)^2 + (γ-c)^2 + c^2 * (γ-1)^2)
  ╠═╡ =#

# ╔═╡ d16d7e17-3662-4f5e-a98c-1c423399feed
#=╠═╡
function plot_mrp_value_functions(γ, c)
	v1s = 0:0.01:10
	v2s = 0:0.01:10
	v1 = γ / (1 - γ)
	v2 = 2 + γ / (1-γ)
	vtrue = scatter(x = [v1], y = [v2]; name = "True Value Function", mode = "markers")
	w_minve = mrp_ve_min(γ, c)
	w_test = min_be(γ, c)
	ve_magnitude = ((w_minve - v1)^2 + (c*w_minve - v2)^2)/2
	v̂_ve = scatter(x = [w_minve], y = [c*w_minve]; name = "Minimum VE", mode = "markers")
	v̂_test = scatter(x = [w_test], y = [c*w_test]; name = "Minimum BE", mode = "markers")
	
	w_pbe_min = mrp_pbe_min(γ, c)
	bo_td = mrp_bellman_operator(w_pbe_min, γ, c)
	bo_td_tr = scatter(x = [bo_td[1]], y = [bo_td[2]]; name = "Bellman Operator on TD Fixed Point", mode = "markers")

	be_td_tr = scatter(x = [w_pbe_min, bo_td[1]], y = [c*w_pbe_min, bo_td[2]]; name = "Bellman Error on TD Fixed Point", mode = "lines")

	wmin_tde = mrp_wmin_tde(γ, c)
	min_tde_tr = scatter(x = [wmin_tde], y = [c*wmin_tde], name = "Minimum TDE", mode = "markers")
	
	ve = scatter(x = [w_minve, v1], y = [c*w_minve, v2], name = "Value Error Vector, magnitude = $(round(ve_magnitude, sigdigits = 3))", mode = "lines")
	bo_v̂ = mrp_bellman_operator(w_minve, γ, c)
	bo_test = mrp_bellman_operator(w_test, γ, c)
	be_test = ((bo_test[1] - w_test)^2 + (bo_test[2] - c*w_test)^2)/2
	bo_tr = scatter(x = [bo_v̂[1]], y = [bo_v̂[2]]; name = "Bellman Operator on v̂", mode = "markers")
	bo_test_tr = scatter(x = [bo_test[1]], y = [bo_test[2]]; name = "Bellman Operator on v̂ test", mode = "markers")
	be_test_tr = scatter(x = [w_test, bo_test[1]], y = [c*w_test, bo_test[2]]; name = "Test Bellman Error Vector, magnitude = $(round(be_test; sigdigits = 3))", mode = "lines")
	be_tr = scatter(x = [w_minve, bo_v̂[1]], y = [c*w_minve, bo_v̂[2]]; name = "Bellman Error Vector", mode = "lines")
	pbe_w = mrp_wmin(bo_test[1], bo_test[2], c)
	pbe_tr = scatter(x = [pbe_w], y = [c*pbe_w], name = "PBE", mode = "markers")

	pbe_min_tr = scatter(x = [w_pbe_min], y = [c*w_pbe_min], name = "TD fixed point", mode = "markers")
	ws = 0:0.01:10

	bos = [mrp_bellman_operator(w, γ, c) for w in ws]
	bos_tr = scatter(x = [x[1] for x in bos], y = [x[2] for x in bos], name = "Bellman Operator on Approximations", mode = "lines")
	v̂ = scatter(x = ws, y = c .* ws; name = "Approximate Value Functions")
	plot([vtrue, v̂, v̂_ve, v̂_test, ve, bo_tr, be_tr, bo_test_tr, be_test_tr, pbe_tr, pbe_min_tr, bo_td_tr, be_td_tr, bos_tr, min_tde_tr], Layout(xaxis_title = "State 1 Value", yaxis_title = "State 2 Value", title = "MRP Values for γ = $γ and c = $c, w that minimizes value error = $w_minve", xaxis_range = [0, 10], yaxis_range = [0, 7], xaxis_constrain = "domain", yaxis_scaleanchor = "x", height = 700))
end
  ╠═╡ =#

# ╔═╡ 874003a9-40f0-4d73-8070-085143487d12
#=╠═╡
plot_mrp_value_functions(mrp_value_params...)
  ╠═╡ =#

# ╔═╡ fbf4401f-fb57-4d9c-a8a6-439ad19fd5bb
md"""
#### True Values vs Minimizing Solutions for All Errors
"""

# ╔═╡ c0e58f98-a52e-4742-a850-661faac4bbed
#=╠═╡
md"""
Adjust Discount Rate γ: $(@bind wcompare_γ Slider(0.:0.01:.99999; default = 0.5, show_value=true))
"""
  ╠═╡ =#

# ╔═╡ e37d1246-ccd6-481a-af2b-7d2d6acb8bbf
md"""
Comparing all of the potential errors: 

$\begin{flalign}
w_{ve} &= \frac{\gamma(1 - c) + 2c}{(1 - \gamma)(1 + c^2)} \\

w_{be} &= \frac{2(2c - \gamma(1+c))}{2 + \gamma(1+c)^2 (\gamma - 2) + 2c^2} \\

w_{pbe} &= \frac{4c}{2(1+c^2) - \gamma (1+c)^2} \\

w_{tde} &=  \frac{2(2c - \gamma (1 + c))}{(\gamma -1)^2 + (\gamma c  - 1)^2 + (\gamma - c)^2 + c^2(\gamma - 1)^2}
\end{flalign}$
"""

# ╔═╡ 12d724c0-a40b-4f7b-922e-9f8738bf01f4
#=╠═╡
function compare_optimal_w(γ::T; c_range = LinRange(zero(T), one(T)*3, 1000)) where T<:Real
	ve = mrp_ve_min.(γ, c_range)
	be = min_be.(γ, c_range)
	td0 = mrp_pbe_min.(γ, c_range)
	tde = mrp_wmin_tde.(γ, c_range)
	traces1 = [scatter(x = c_range, y = y, name = name) for (y, name) in zip([ve, be, td0, tde], ["Value Error", "Bellman Error", "Projected Bellman Error", "Mean Square TD Error"])]
	traces2 = [scatter(x = c_range, y = c_range .* y, name = name) for (y, name) in zip([ve, be, td0, tde], ["Value Error 2", "Bellman Error 2", "Projected Bellman Error 2", "Mean Square TD Error 2"])]
	v1_true = γ / (1 - γ)
	value1_trace = scatter(x = c_range, y = fill(v1_true, 1000), name = "True Value 1", line_color = "black", line_dash = "dash")
	value2_trace = scatter(x = c_range, y = fill(2 + v1_true, 1000), name = "True Value 2", line_color = "black", line_dash = "dash")
	plot([traces1; traces2; value1_trace; value2_trace], Layout(xaxis_title = "Value of c", yaxis_title = "State Value Estimates"))
end
  ╠═╡ =#

# ╔═╡ 56672e64-6834-4639-921b-0e87cede4d7a
#=╠═╡
compare_optimal_w(wcompare_γ)
  ╠═╡ =#

# ╔═╡ 3ddf0432-99e5-4ce3-ac63-86f43b2d1a1c
md"""
The 3D space contains vectors that represent all value function of 3 states: $\{s1, s2, s3\}$ where $\overline{v} = [v1, v2, v3]$.  Let's say we approximate these value functions with a parameter vector $\mathbf{w} = \{w1, w2\}$ such that $\hat v(s) = \mathbf{w} \cdot \mathbf{x}(s)$ where $\mathbf{x}(s)$ is the feature vector representation of a given state.  There are three states so three feature vectors that must be defined: $\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3$.  Each feature vector has two components, one for each parameter, so three of them cannot be linearly independent.  All value function approximations $\hat v(s)$ lie in a plane within the 3D space expressing the constraints placed between the 3 state values.  For example let's say the feature vectors are $\{0, 1 \}, \left \{\frac{\sqrt{3}}{2}, -\frac{1}{2}\right \},  \left \{-\frac{\sqrt{3}}{2}, -\frac{1}{2}\right \}$.  Then the values function approximation would be $\hat v = \left \{ w2, \frac{w1 \sqrt{3} - w2}{2}, -\frac{w1 \sqrt{3} + w2}{2} \right \}$

Another option is let's say to simplify the problem, we just group two of the states together so the feature vector is the same for two of the states.  In this scenario we could have $\hat v = \left \{ x1_1 w1 + x1_2 w2, x2_1 w1 + x2_2 w2, x3_1 w1 + x3_2 w2 \right \}$
"""

# ╔═╡ 6e6b9d64-2d90-40a4-abde-2fd0d6ab7d7a
#=╠═╡
plot([scatter(x = [0, 0], y = [0, 1], name = "x1"), scatter(x = [0, sqrt(3)/2], y = [0, -0.5], name = "x2"), scatter(x = [0, -sqrt(3)/2], y = [0, -0.5], name = "x3")], Layout(xaxis_range = [-1, 1], yaxis_range = [-1, 1], width = 500, height = 500, legend_orientation = "r", margin = attr(t = 60, l = 60, r = 0, b = 60)))
  ╠═╡ =#

# ╔═╡ d577b03d-bc68-4b32-9c6d-d92e0c4d7c99
#=╠═╡
@bind feature_angles PlutoUI.combine() do Child
	md"""
	 $$\theta_2$$ : $(Child(:θ2, Slider(0:360, default = 90, show_value=true))) °

	 $$\theta_3$$ : $(Child(:θ3, Slider(0:360, show_value=true))) °
	"""
end
  ╠═╡ =#

# ╔═╡ d45813bd-8aa2-4454-bc12-ae8dce0a4590
function test_step_dist(;num_steps = 1000)
	state_visits = zeros(3)
	s = 1
	for i in 1:num_steps
		s += rand((-1, 1))
		s = clamp(s, 1, 3)
		state_visits[s] += 1
	end
	return state_visits
end

# ╔═╡ a36695bb-599d-4f83-a747-9e3d0668e7d8
test_step_dist(; num_steps = 100_000)

# ╔═╡ 5047d396-af48-49fa-bf68-702fbe42c18e
#=╠═╡
const feature_vectors = [[cos(2*π*θ / 360), sin(2*π*θ/360)] for θ in feature_angles]
  ╠═╡ =#

# ╔═╡ c4916313-d4f0-443c-a81e-05d2b765acf0
#=╠═╡
plot([scatter(x = [0, feature_vectors[i][1]], y = [0, feature_vectors[i][2]], name = "x$i") for i in 1:2], Layout(xaxis_range = [-1, 1], yaxis_range = [-1, 1], width = 500, height = 500, legend_orientation = "r", margin = attr(t = 60, l = 60, r = 0, b = 60)))
  ╠═╡ =#

# ╔═╡ 9bc2895e-ab70-49f2-be7c-61f19054cf50
#=╠═╡
@bind weight_select PlutoUI.combine() do Child
	md"""
	w1 : $(Child(Slider(-1:0.1:1, show_value=true)))

	w2 : $(Child(Slider(-1:0.1:1, show_value=true)))
	"""
end
  ╠═╡ =#

# ╔═╡ aeca907a-ee07-4045-b98f-0c67b1734008
#=╠═╡
function plot_value_approximation(x1, x2, x3; n = 100, w = [1, 1])
	v(w) = (x = dot(x1, w), y = dot(x2, w), z = dot(x3, w))
	xs = []
	ys = []
	zs = []
	for w1 in LinRange(-1, 1, n)
		for w2 in LinRange(-1, 1, n)
			push!(xs, dot(x1, [w1, w2]))
			push!(ys, dot(x2, [w1, w2]))
			push!(zs, dot(x3, [w1, w2]))
		end
	end
	tr = scatter3d(x = xs, y = ys, z = zs, mode = "markers", marker_size = 1)
	v̂ = v(w)
	vtr = scatter3d(x = [0, v̂.x], y = [0, v̂.y], z = [0, v̂.z], name = "v̂($w)", mode = "lines+markers")
	plot([tr, vtr], Layout(scene = attr(xaxis_range = [-1, 1], yaxis_range = [-1, 1], zaxis_range = [-1, 1])))
end
  ╠═╡ =#

# ╔═╡ a780e90c-c6d1-44c8-9b55-d52cf4c20db4
#=╠═╡
plot_value_approximation([1, 0], feature_vectors[1], feature_vectors[2]; w = [weight_select[1], weight_select[2]])
  ╠═╡ =#

# ╔═╡ 666a27b0-6fa3-4f5c-b85e-c3f66f0b5790
md"""
## 11.5 Gradient Descent in the Bellman Error

True gradient methods update the parameters of an objective function in the direction of the negative gradient of some objective function.  Under the right conditions, these techniques have robust convergence properties and can find the true minimum value of the objective.  Previously, the only true gradient methods developed were Monte Carlo methods, but now we seek to find true gradient methods in the off-policy case in terms of some other objective.  If we forgo Monte Carlo methods, then the value error objective is no longer available to us so we must consider alternatives such as the Bellman Error.  It also turns out that in the on-policy case, semi-gradient TD in the linear case does behave like a true gradient method that reaches the TD fixed point, however this point minimizes the projected Bellman Error rather than the value error.  The TD fixed point is a fixed point because unlike the other error metrics, it can be reduced to 0 for any MDP and any linear approximation function.  At this point, the semi-gradient update is expected to be 0.
"""

# ╔═╡ 16b45f02-8bdf-4721-ab70-ab2146c139bb
md"""
### Bellman Error Definition

With a true SGD method for off-policy training, we would not have to worry about divergence even under non-linear approximation.  One of the most popular proposed objectives for SGD is the *Bellman error*.  Consider first the effect of applying the Bellman operator on a given state value function approximation $\hat v(s)$

$(B_\pi \hat v)(s) \doteq \sum_a \pi(a\vert s)\sum_{s^\prime, r} p(s^\prime, r \vert s, a)[r + \gamma \hat v (s^\prime)$

The *Bellman error* at state $s$ is the difference between the value at state s and the Bellman operator value:

$\overline{\delta}_\mathbf{w}(s) \doteq (B_\pi \hat v)(s) - \hat v (s)$

We know that for the true value function $v_\pi$, the effect of applying the Bellman operator is to leave the value unchanged:

$(B_\pi v_\pi)(s) = v_\pi(s) \; \forall s$

So for the true value function the Bellman error is 0 at every state.  In the tabular methods for part 1 of the book, we sought to find this unique value function for which the Bellman error is 0 at every state and the value error is also 0 at every state.  When we deal with approximation, however, it is impossible to guarantee 0 Bellman error, so we must define a different objective like we did for the value error, one that weights the errors at each state by the on-policy distribution.  This distribution is irrelevant for tabular methods since the final result will have 0 error at every state.
"""

# ╔═╡ 3368ae68-3ce7-4426-a088-a24e5574abde
md"""
### Mean Square Bellman Error

One natural choice for objective function is to take the squared sum of the Bellman error at each state weighted by the probability of visiting that state while following the target policy.  We call this objective the mean square Bellman error and it is defined as:

$\begin{flalign}
\overline{\text{BE}} & \doteq \sum_s \mu_\pi(s) \overline{\delta}_\mathbf{w}(s)^2 \\
&= \sum_s \mu_\pi(s) \left ( (B_\pi \hat v)(s) - \hat v (s) \right )^2 \\
&= \sum_s \mu_\pi(s) \left [ \sum_a \pi(a\vert s)\sum_{s^\prime, r} p(s^\prime, r \vert s, a)[r + \gamma \hat v (s^\prime)] - \hat v (s) \right ]^2 \\
\end{flalign}$

In the diagram in the previous section this error is represented by the length of a vector which points from the initial approximation point to the Bellman operator applied to that point.  Since that diagram represents a two state MDP, the point's position on the graph represents both values.
"""

# ╔═╡ 68539414-2775-4f1e-bc03-016a7736c583
md"""
### Mean Square TD Error

Prior to analyzing $\overline{\text{BE}}$, consider the one-step TD error that we often use as a starting point for bootstrap approximation methods:

$\delta_t = R_{t+1} + \gamma \hat v (S_{t+1}) - \hat v(S_t)$

This error term is always available by sampling from the environment, so let's consider an objective that directly uses this to as a minimization objective.  For example, we could square the TD error to ensure it is always positive and weight it accoring to the target policy distribution.  Such an objective is called the *mean square TD error*:

$\begin{flalign}
\overline{\text{TDE}}(\mathbf{w}) &= \sum_{s \in \mathcal{S}} \mu(s) \mathbb{E}[\delta_t^2 \mid S_t = s, A_t \sim \pi] \\
&= \sum_{s \in \mathcal{S}} \mu(s) \mathbb{E}[\rho_t \delta_t^2 \mid S_t = s, A_t \sim b] \\
&= \mathbb{E}_b [\rho_t \delta_t^2] \tag{if μ is the distribution encountered under b}
\end{flalign}$

The importance sampling ratio is first used to adjust the values from actions taken by the behavior policy.  Notice that in the last step, however, if we sample an entire trajectory by the behavior policy, we will not weight updates according to the on-policy distribution $\mu_\pi(s)$ but rather according to $\mu_b (s)$.  This can be sampled from the environment but represents a different objective.  Although incorrect, a true SGD method will converge to this objective which eliminates one of the problems we previously had for off-policy training.  We can derive the following update rule by taking the gradient of this objective assuming that our approximation function is parametrized by some weight vector $\hat v(s, \mathbf{w})$:

$\begin{flalign}
\mathbf{w}_{t+1} &= \mathbf{w}_t - \frac{1}{2} \alpha \nabla (\rho_t \delta_t^2) \\
&= \mathbf{w}_t - \alpha \rho_t \delta_t \nabla \delta_t \\
&= \mathbf{w}_t + \alpha \rho_t \delta_t \left ( \nabla \hat v(S_t, \mathbf{w}_t) - \gamma \nabla \hat v(S_{t+1}, \mathbf{w}_t) \right ) \tag{11.23} \\
\end{flalign}$

This update rule is a true gradient method and converges to the minimum $\overline{\text{TDE}}$ as defined above.  The algorithm which uses (11.23) to update weights is known as the *naive residual-gradient* algorithm.  It appears very similar to the semi-gradient TD update with importance sampling:

$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \rho_t \delta_t \nabla \hat v (S_t, \mathbf{w}_t)$

The only difference is the final term which turns this into a true SGD method.  However, this method does not converge to the TD fixed point or the point of minimum value error in general which is not surprising considering the objective is different.  As seen above in the linear value function diagram, this point may be undesireable since it is further away from the point of minimum value error.  The subsequent example highlights another problem with this objective even in the on-policy case where the distribution of states is no longer a problem.
"""

# ╔═╡ 2344c8c6-6cff-4047-bf43-5e081a3067ef
md"""
### Example 11.2: A-split example

Consider a stochastic episodic environment with three states: A, B, C.  Episodes always begin in state A and have an equal chance of transitioning into B or C with 0 reward.  Episodes always terminate from states B and C with a reward of 1 and 0 respectively.  Since this problem is an MRP, there are no policies and thus no consideration of off-policy learning or importance sampling ratios.  Since the problem is episodic, we can consider the undiscounted case $\gamma = 1$.  Given these constraints, the true state values are easy to determine.

#### True Values

$\begin{flalign}
v_B &= 1 \\
v_C &= 0 \\
v_A &= \frac{1}{2}(v_B + v_C) = \frac{1}{2}
\end{flalign}$

#### Approximation Function

Consider the case of approximation but with three parameters $w_A, w_B, w_C$ so that an exact solution is possible with $w_A = \frac{1}{2}, w_B = 1, w_C = 0$ where each parameter matches the corresponding state value:

$\begin{flalign}
\hat v_A &= w_A \\
\hat v_B &= w_B \\
\hat v_C &= w_C \\
\end{flalign}$

Approximation methods must also take into account $\mu(s)$.  Episodes always begin in state A and split evenly into B and C before terminating, so $\mu_A = 2 \mu_B = 2 \mu_C$ and $\mu_B = \mu_C$.  From these relationships we can solve for the state distribution:

$\begin{flalign}
\mu_A &= \frac{1}{2} \\
\mu_B &= \frac{1}{4} \\
\mu_C &= \frac{1}{4} \\
\end{flalign}$

With three parameters this is a tabular problem, and we have already established that Monte Carlo gradient methods will converge to the exact solution in this case.  The semi-gradient TD(0) method should also converge to the exact solution since the minimum value-error is 0, but we can also verify this by studying the update rule:
"""

# ╔═╡ bb099078-5995-48aa-b634-82c5c8f5f396
md"""
#### Semi-gradient TD(0)

The update for each state depends upon the TD error.  State A has two equally likely TD errors given by:

$\begin{flalign} 
\delta_A &= 0 + w_B - w_A = w_B - w_A \\
\delta_A &= 0 + w_C - w_A = w_C - w_A\\
\end{flalign}$

States B and C both transition into terminal states deterministically, so their TD errors are given by:

$\begin{flalign} 
\delta_B &= 1 - w_B \\
\delta_C &= 0 - w_C = w_C\\
\end{flalign}$

At the TD-fixed point, the expected parameter update after visiting states according to the on-policy distribution should be 0.  Let's consider all 4 updates that occur:

$\begin{flalign} 
w_A &\leftarrow w_A + \alpha \left ( w_B - w_A \right ) \\
w_A &\leftarrow w_A + \alpha \left ( w_C - w_A \right ) \\
w_B &\leftarrow w_B + \alpha (1 - w_B) \\
w_C &\leftarrow w_C + \alpha w_C
\end{flalign}$

The convergence condition for $w_B$ and $w_C$ is obvious from the last two equations: $w_B = 1$ and $w_C = 0$.  For state A, the two updates occur with equal probability, so the sum of the term multiplying $\alpha$ for each should be 0.  We can substitude the values we already know for $w_B$ and $w_C$ to get: $1 - w_A - w_A = 0 \implies w_A = \frac{1}{2}$, confirming that the TD fixed point equals the exact solution:  

$\begin{flalign}
w_A &= \frac{1}{2} = v_A \\
w_B &= 1 = v_B \\
w_C &= 0 = v_C
\end{flalign}$
"""

# ╔═╡ 1b9d8e13-7872-4bfe-9269-aee9a31d5ae6
md"""
#### Minimum $\overline{\text{TDE}}$ Solution

Now let's calculate the parameters that minimize the mean square TD error.  Instead of applying the parameter update rule, we can solve directly for the minimizing parameters using the definition of the objective:

$\overline{\text{TDE}} = \mathbb{E}_b [\rho_t \delta_t^2]$

There are four distinct TD errors as shown above.  The two for state A have equal probability and $\mu_A = \frac{1}{2}$ so each TD error will occur with $\frac{1}{4}$ probability:

$\begin{flalign}
\overline{\text{TDE}} &= \mathbb{E}_b [\rho_t \delta_t^2] \\
&= \frac{1}{4}\left ( (w_B - w_A)^2 + (w_C - w_A)^2 + (1 - w_B)^2 + w_C^2 \right ) \\
&= \frac{1}{4} \left ( 2w_B^2 - 2 w_B w_A + 2w_A^2 + 2w_C^2 - 2 w_A w_C+ 1 - 2 w_B \right )
\end{flalign}$

Next consider the gradient with respect to each parameter which will be 0 at convergence to the minimum

$\begin{flalign}
\frac{\partial \overline{\text{TDE}}}{\partial w_A} &= \frac{1}{4} \left ( -2 w_B + 4w_A - 2 w_C \right ) = \frac{1}{2}(2w_A - w_B - w_C)\\
\frac{\partial \overline{\text{TDE}}}{\partial w_B} &= \frac{1}{4} \left (4w_B - 2w_A - 2 \right ) = \frac{1}{2}(2w_B - w_A - 1)\\
\frac{\partial \overline{\text{TDE}}}{\partial w_C} &= \frac{1}{4} \left ( 4w_C - 2w_A \right ) = \frac{1}{2}(2w_C - w_A)\\
\end{flalign}$

Setting the three components to 0 implies the following

$\begin{flalign}
0 &=  2w_A - w_B - w_C \implies 2w_A = w_B + w_C \\
0 &= 2w_B - w_A - 1 \implies 2w_B = 1 + w_A\\
0 &= 2w_C - w_A \implies 2w_C = w_A\\
\end{flalign}$

Using the last two expressions we can change the first expression into one just in terms of $w_A$: $4w_A = 1 + w_A + w_A \implies w_A = \frac{1}{2}$

Then the other weights follow: $w_B = \frac{1}{2} (1 + \frac{1}{2}) = \frac{3}{4}$ and $w_C = \frac{1}{4}$

$\begin{flalign}
w_A &= \frac{1}{2} = v_A \\
w_B &= \frac{3}{4} \ne v_B \\
w_C &= \frac{1}{4} \ne v_C
\end{flalign}$

So even though an exact solution is possible, minimizing the $\overline{\text{TDE}}$ does not find it.  For Monte Carlo gradient and semi-gradient TD methods developed earlier, the tabular approximation case always reduced to the tabular methods from part 1.  This simple example shows that is not the case for the mean square TD objective.

"""

# ╔═╡ 0babc5a1-c404-4ce8-bf30-74db15790c72
md"""
### Residual Gradient Algorithm

Let's now return to the mean square Bellman error and try to derive an update rule that can be achieved through sampling:

$\begin{flalign}
\overline{\text{BE}} &= \sum_s \mu_\pi(s) \left [ \sum_a \pi(a\vert s)\sum_{s^\prime, r} p(s^\prime, r \vert s, a)[r + \gamma \hat v (s^\prime)] - \hat v (s) \right ]^2 \\
&= \sum_s \mu_\pi(s) \left [ \sum_a \pi(a\vert s)\sum_{s^\prime, r} p(s^\prime, r \vert s, a)[r + \gamma \hat v (s^\prime) - \hat v (s)] \right ]^2 \\
&= \sum_s \mu_\pi(s) \mathbb{E} \left [ \delta_t \vert S_t = s, A_t \sim \pi(s) \right ]^2 \\
&= \sum_s \mu_\pi(s) \mathbb{E}_\pi \left [ \delta_t \vert S_t = s \right ]^2 \\
&= \sum_s \mu_\pi(s) \mathbb{E}_b \left [\rho_t \delta_t \vert S_t = s \right ]^2 \\
\end{flalign}$

We can see that the mean square Bellman error is just expected value of the TD error for each state squared.  The only difference between this and the $\overline{\text{TDE}}$ is that the squaring is done outside of the expected value.

We can derive an SGD parameter update rule by taking the gradient with respect to the parameters.  If the visited states matches the behavior policy instead of the target policy then we are really optimizing something different again which is the mean square Bellman error weighted by the behavior policy, but it will still converge to that value as well.  The example we will consider does not have an off-policy distribution, and the problems with this approach will appear ignoring that problem.

$\begin{flalign}
\mathbf{w}_{t+1} &= \mathbf{w}_t - \frac{1}{2} \alpha \nabla (\mathbb{E}_\pi [\delta_t]^2) \\
&= \mathbf{w}_t - \frac{1}{2} \alpha \nabla (\mathbb{E}_b [\rho_t \delta_t]^2) \\
&= \mathbf{w}_t - \alpha \mathbb{E}_b[\rho_t \delta_t] \nabla \mathbb{E}_b [\rho_t \delta_t] \tag{chain rule}\\
&= \mathbf{w}_t - \alpha \mathbb{E}_b [\rho_t (R_{t+1} + \gamma \hat v (S_{t+1}, \mathbf{w}) - \hat v(S_t, \mathbf{w}))] \mathbb{E}_b[\rho_t \nabla \delta_t] \tag{linearity of expected value}\\
&= \mathbf{w}_t + \alpha \left [ \mathbb{E}_b [\rho_t (R_{t+1} + \gamma \hat v (S_{t+1}, \mathbf{w}) ] - \hat v(S_t, \mathbf{w}) \right ] \left [ \nabla \hat v(S_t, \mathbb{w}) - \gamma \mathbb{E}_b[\rho_t \nabla \hat v(S_{t+1}, \mathbf{w})] \right ] \\
\end{flalign}$

Note that the two expected values in this expression cannot in general use the same sample of the transition state since the product would reflect the correlation between the samples.  Instead we need two independent samples or, if the environment is deterministic, this is unecessary.  For example 11.2, this algorithm would find the correct values since it is guaranteed to minimize the Bellman error and the exact solution will always have 0 Bellman error at every state.  We can however, modify the example to one in which the minimum Bellman error solution is also problematic.
"""

# ╔═╡ 6edbe471-361c-463a-b942-3ecbd879c427
md"""
### Example 11.3: A-presplit example, a counterexample for the $\overline{\text{BE}}$

Consider the A-split example from before but converted into a deterministic problem by separating state A into two parts each of which does a deterministic transition either into state B or C.  The exact solution is trivial for the four state values:

#### Exact Solution

$\begin{flalign}
v_B &= 1 \\
v_C &= 0 \\
v_{A1} &= 0 + v_B = 1 \\
v_{A2} &= 0 + v_C = 0 \\
\end{flalign}$

There are four deterministic states in this MRP, but we will consider approximate solutions in which both A states share a single parameter.

$\begin{flalign}
\hat v_{A1} = w_A \\
\hat v_{A2} = w_A \\
\hat v_B = w_B \\
\hat v_C = w_C \\
\end{flalign}$

Equal time is spent in all four states since episodes start in either A1 or A2 with equal probability.  From the perspective of the approximation parameter, $w_A$ will appear $\frac{1}{2}$ of the time while $w_B$ and $w_C$ will appear $\frac{1}{4}$ of the time, so this problem appears the same as the A-split example with respect of $\mathbf{w}$
"""

# ╔═╡ 77566379-e31c-41a7-bb31-ccb3f2d69723
md"""
#### Minimum $\overline{\text{VE}}$ Solution

Using the true values, we can write down the value error objective and directly calculate the parameters which minimize it by taking the gradient:

$\overline{\text{VE}} = \frac{1}{4} \left ( (w_A - 1)^2 + (w_A - 0)^2 + (w_B - 1)^2 + (w_C - 0)^2 \right ) = \frac{1}{4} \left ( 2(w_A^2 - w_A) + 1 + (w_B - 1)^2 + w_C^2 \right )$

$\begin{flalign}
\frac{\partial \overline{\text{VE}}}{\partial w_A} &= \frac{1}{2} \left ( 2 w_A - 1 \right ) \\
\frac{\partial \overline{\text{VE}}}{\partial w_B} &= \frac{1}{2} \left (w_B - 1 \right )\\
\frac{\partial \overline{\text{VE}}}{\partial w_C} &= \frac{1}{2} w_C\\
\end{flalign}$



Setting each of these to zero reveals the unique solution which minimizes the value error:

$\begin{flalign}
w_A &= \frac{1}{2} \\
w_B &= 1 \\
w_C &= 0 \\
\end{flalign}$

which is the same solution as before, expect now the value error is not zero but rather $$\frac{1}{4}(2(\frac{1}{4} - \frac{1}{2}) + 1) = \frac{1}{8}$$.  An exact solution is not possible since we have one too few parameters, but the whole purpose of approximation is to define what we mean by a good solution that cannot be exact.  The data with respect to the parameters is the same as would be for the A-split example, and if this data did come from that problem then the value error would be 0.  The fact that two distinct problems can appear the same but have different true value error might seem to be a problem, but the difference is value error is always a variance term which does not depend on the parameters but only on the stochastic nature of the problem.  From that perspective, the A-split example has zero value error for these parameters because the true value from state A is always $$\frac{1}{2}$$ while the the A-presplit example the true value of A1 and A2 is not $$\frac{1}{2}$$ but that information is hidden from the approximation.
"""

# ╔═╡ 358daffe-cf39-440d-a9b4-d0a3497b41ad
md"""
#### Minimum $\overline{\text{BE}}$ Solution

What if we instead try to minimize the Bellman error?  We fully know the dynamics of the problem, so we can directly find a solution without resorting to gradient methods. 

$\begin{flalign}
\overline{\text{BE}} &= \sum_{s \in \mathcal{S}} \mu_\pi(s)\mathbb{E}_\pi [\delta_t \mid S_t = s]^2 \\
&= \frac{1}{4} \left [(w_B - w_A)^2 + (w_C - w_A)^2 + (1 - w_B)^2 + w_C^2 \right ]\\
\end{flalign}$

But this is the same expression we had for the $\overline{\text{TDE}}$ for example 11.2, so the previous minimizing parameters will also apply here:

$\begin{flalign}
w_A &= \frac{1}{2} \\
w_B &= \frac{3}{4} \\
w_C &= \frac{1}{4}
\end{flalign}$

Obviously this differs from the minimum value error solution.  What about the TD fixed point though?  We know in general that the semi-gradient algorithm does not converge to the minimum value error, but would it perform better than the Bellman error in this case?  Let's consider the TD(0) update rule and when it converges.
"""

# ╔═╡ c368eab7-9ce2-4b9a-bd91-3a8810472839
md"""
#### TD Fixed Point Solution

$\begin{flalign} 

w_A &\leftarrow w_A + \alpha \left ( w_B - w_A \right ) \\
w_A &\leftarrow w_A + \alpha \left ( w_C - w_A \right ) \\
w_B &\leftarrow w_B + \alpha (1 - w_B) \\
w_C &\leftarrow w_C + \alpha (0 - w_C)
\end{flalign}$

These updates each occur with equal probability and are exactly the same as the expected updates for the A-spit example.  Therefore, the solution will be the same and match the exactly solution for A-split and the solution which minimizes the value error for A-presplit:

$\begin{flalign}
w_A &= \frac{1}{2} \\
w_B &= 1 \\
w_C &= 0 \\
\end{flalign}$

We did not attempt to show any bound for the Bellman error to ensure that its solution is close to the value error, but just from this example it is clear that there are examples where semi-gradient TD methods find the same solution as the minimum value error but the Bellman error minimum solution is different.

"""

# ╔═╡ f12ad623-59f9-4efe-8fb5-14b1bf6904bc
md"""
## 11.6 The Bellman Error is Not Learnable

In the context of this chapter, learnability means that a quantity can be estimated from data alone.  In the strictest approximation case, the only data available is the rewards observed as well as the feature vector of whatever state we are in.  Sometimes we also have access to the state information in addition to the feature vector alone, and in this case there are more quantities we can calculate.

Consider the original objective we defined for approximation, the value error: $\overline{\text{VE}} = \sum_{s \in \mathcal{S}} \mu_\pi(s)(v_\pi(s) - \hat v (s))^2$.  This calculation depends on knowing the true state value which is only possible if we have full information about the state.  Consider an alternative error though in which we replace this unknown quantity with something that can be observed: $\overline{\text{RE}}(\mathbf{w}) = \mathbb{E}\left [ \left ( G_t - \hat v(S_t, \mathbf{w}) \right )^2 \right]$.  It turns out that this objective is equal to the value error objective plus a variance term that does not depend on the paramters $\mathbf{w}$.  So if we optimize $\overline{\text{RE}}$ the parameters we find will match those we would have found optimizing $\overline{\text{VE}}$.  Note that we can only sample this in the case of Monte Carlo estimation and that is not possible for continuing problems.  For those, we must consider the bootstrapping objectives such as the TD error.

Luckily the TD error is also always observable $\delta_t \doteq R_{t+1} + \gamma \hat v (S_{t+1}, \mathbf{w}_t) - \hat v(S_t, \mathbf{w}_t)$ since it only depends on the observed rewards and feature vectors.  The semi-gradient methods we previously developed only depend on the TD error and thus are observable and in the case of linear approximation are bounded to some region around the minimum value error solution.  

The Bellman error, on the other hand is $\mathbb{E}_\pi [\delta_t]$ is not observable because it requires knowledge of the state in order to correctly assign samples to the state.  We can clearly sample this error on each step, but if two states appear to have the same feature vector, and that is the only information we have, then we cannot correctly separate those samples to the assigned states and apply the probability weighting to the error.  If we had access to the state, then we can in theory observe the Bellman error, but we will see that minimizing it does not produce desireable results.
"""

# ╔═╡ e49849c5-d9b1-426b-b471-3acd32dcf07d
md"""
> ### * *Exercise 11.4* 
> Prove (11.24). Hint: Write the $\overline{\text{RE}}$ as an expectation over possible states $s$ of the expectation of the squared error given that $S_t = s$.  Then add and subtract the true value of state $s$ from the error (before squaring), grouping the subtracted true value with the return and the added true value with the estimated value.  Then if you expand the square, the most complex term will end up being zero, leaving you with (11.24).

$\overline{\text{RE}}(\mathbf{w}) = \overline{\text{VE}}(\mathbf{w}) + \mathbb{E}_\pi \left [ \left ( G_t - v_\pi(S_t) \right )^2 \right ] \tag{11.24}$

To start out we have the definition of the *mean square return error*

$\overline{\text{RE}}(\mathbf{w}) = \mathbb{E}_\pi \left [ (G_t - \hat v(S_t, \mathbf{w}))^2 \right ]$

Also we can note from Chapter 3 that 

$v_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] \tag{1}$ 

and from Chapter 9 that 

$\overline{\text{VE}}(\mathbf{w}) \doteq \sum_{s \in \mathcal{S}} \mu(s) \left [ v_\pi(s) - \hat v(s, \mathbf{w}) \right ]^2 \tag{2}$

Rewriting expectation results in:

$\begin{flalign}
\overline{\text{RE}}(\mathbf{w}) &= \sum_s \mu_\pi(s) \mathbb{E_\pi}\left [ (G_t - \hat v(S_t, \mathbf{w}))^2 | S_t = s \right ]\\
&= \sum_s \mu_\pi(s) \mathbb{E_\pi}\left [ (G_t - \hat v(S_t, \mathbf{w}) + v_\pi(S_t) - v_\pi(S_t))^2 | S_t = s \right ]\\
&= \sum_s \mu_\pi(s) \mathbb{E_\pi}\left [ ((G_t - v_\pi(S_t)) + (v_\pi(S_t) - \hat v(S_t, \mathbf{w})))^2 | S_t = s\right ]\\
&= \sum_s \mu_\pi(s) \mathbb{E_\pi} \left [ (G_t - v_\pi(S_t))^2 + (v_\pi(S_t) - \hat v(S_t, \mathbf{w}))^2 + 2((G_t - v_\pi(S_t))(v_\pi(S_t) - \hat v(S_t, \mathbf{w}))) | S_t = s \right ]\\
&= \mathbb{E}\left [ (G_t - v_\pi(S_t))^2 \right ] + \sum_s  \mu_\pi(s) \left [v_\pi(s) - \hat v(s, \mathbf{w}) \right ]^2 + \sum_s 2\mu_\pi(s) \left [ \mathbb{E_\pi}[G_t | S_t = s] (v_\pi(s) - \hat v(s, \mathbf{w})) - v_\pi(s)^2 + v_\pi(s) \hat v(s, \mathbf{w}) \right]\\
&= \mathbb{E}\left [ ((G_t - v(S_t))^2 \right ] + \overline{\text{VE}} + \sum_s 2\mu_\pi(s) \left [ v_\pi(s)^2 -  \hat v(s, \mathbf{w}) v_\pi(s) - v_\pi(s)^2 + v_\pi(s) \hat v(s, \mathbf{w}) \right] \tag{by (1) and (2)}\\
&= \mathbb{E}\left [ ((G_t - v(S_t))^2 \right ] + \overline{\text{VE}} + \sum_s 2\mu_\pi(s) \times 0\\
&\therefore\\
\overline{\text{RE}}(\mathbf{w}) &= \mathbb{E}\left [ ((G_t - v(S_t))^2 \right ] + \overline{\text{VE}}
\end{flalign}$
"""

# ╔═╡ b180997e-fa2b-44de-936f-eb42bef4b6ad
md"""
### Example 11.4: Counterexample to the learnability of the Bellman error

#### True Values
We can use the Bellman equations to calculate the true state values for both MDPs

##### Left MDP
$\begin{flalign}
v_A &= 0 + \gamma v_B \implies v_A = \gamma v_B \\
v_B &= \frac{1}{2} \left (-1 + \gamma v_B + 1 + \gamma v_A \right ) = \frac{\gamma}{2}(v_B + v_A) \\
&\text{combining the two expressions we can get an equation just for the value of state B} \\
v_B &= \frac{\gamma}{2}(v_B + \gamma v_B) \implies v_B = \frac{\gamma v_B}{2}(1 + \gamma)
\end{flalign}$

Given that $0 \le \gamma \lt 1$, this equality is only possible when $v_B = 0 \implies v_A = 0$ and these values are independent of $\gamma$

##### Right MDP
$\begin{flalign}
v_A &= \frac{1}{2} \left ( 0 + \gamma v_B + 0 + \gamma v_{B^\prime} \right ) = \frac{\gamma}{2} \left ( v_B + v_{B^\prime} \right ) \\
v_B &= 1 + \gamma v_A \\
v_B^\prime &= \frac{1}{2} \left ( -1 + \gamma v_{B^\prime} - 1 + \gamma v_B \right ) = -1 + \frac{\gamma}{2} \left (v_{B^\prime} + v_B) \right ) = -1 + v_A \\
&\text{Using the last two expressions, we can derive an equation involving only the value of state A} \\
v_A &= \frac{\gamma}{2}(1 + \gamma v_A - 1 + v_A) = \frac{\gamma v_A}{2} (\gamma + 1) \\
\end{flalign}$

Given that $0 \le \gamma \lt 1$, this equality is only possible when $v_A = 0$.  The other two values follow immediately giving a complete value function of:

$\begin{flalign}
v_A &= 0 \\
v_B &= 1 \\
v_B^\prime &= -1 \\
\end{flalign}$

which like the first is independent of $\gamma$.
"""

# ╔═╡ f10a08f2-eba8-4e5e-87f8-313bb7494a49
md"""
#### Value Error
Consider an approximate solution with only two parameters $w_1, w_2$.  For the first problem, this is still a tabular case with an optimal solution of $w_1 = v_A$ and $w_2 = v_B$.  For the second problem, we impose the following: 

$\hat v_A = w_1, \; \hat v_B = \hat v_{B^\prime} = w_2$
Now consider the original objective for approximation for which we need access to $\mu(s)$.  For the first problem $\mu(A) = \frac{1}{3}, \; \mu(B) = \frac{2}{3}$ and for the second $\mu(A) = \mu(B) = \mu(B^\prime) = \frac{1}{3}$.  Note that this also implies that the observed reward sequence for each problem will look exactly the same with respect to the feature vectors.

##### Left MDP

Since the problem is tabular, the minimum value error is 0 with the following solution:

$\begin{flalign}
\overline{\text{VE}} &= 0 \\
w_1 &= w_2 = 0 \\
\hat v_A &= \hat v_B = \hat v_{B^\prime} = 0
\end{flalign}$

##### Right MDP

This problem cannot have 0 value error due to the limited number of parameters, so we can only hope to minimize it.  We can calculate this minimum explicitely using the known true values:

$\begin{flalign}
\overline{\text{VE}} &= \frac{1}{3} \left [ w_1^2 + (w_2 - 1)^2 + (w_2 + 1)^ 2 \right ] \\
&= \frac{1}{3} \left [ w_1^2 + w_2^2 - 2w_2 + 1 + w_2^2 + 2 w_2 + 1 \right ] \\
&= \frac{1}{3} \left [ w_1^2 + 2 w_2^2 + 2 \right ] 
\end{flalign}$

We can minimize this by setting the gradient to 0 for each term:

$\begin{flalign}
\frac{\partial \overline{\text{VE}}}{\partial w_1} &= \frac{2 w_1 }{3}\\
\frac{\partial \overline{\text{VE}}}{\partial w_2} &= \frac{4 w_2 }{3}\\
\therefore w_1 &= w_2 = 0
\end{flalign}$

So the final solution is the same as that for the first problem but with a different value error:

$\begin{flalign}
\overline{\text{VE}} &= \frac{2}{3} \\
w_1 &= w_2 = 0 \\
\hat v_A &= \hat v_B = \hat v_{B^\prime} = 0
\end{flalign}$

"""

# ╔═╡ a4e11eb7-d314-4d78-b9b6-df8fc2838149
md"""
#### Bellman Error

$\overline{\text{BE}} = \sum_{s} \mu_s \mathbb{E}_\pi \left [ R_{t+1} + \gamma \hat v(S_{t+1}) - \hat v(S_t) \mid S_t = s \right ] ^2$ 

##### Left MDP

Since the problem is tabular, we'd expect the Bellman error to be 0 with the exact solution.  We can verify the solution using the above expression for the Bellman error:

$\begin{flalign}
\overline{\text{BE}} &= \frac{1}{3}(0 + \gamma w_2 - w_1)^2 + \frac{2}{3}\left [ \frac{1}{2}(-1 + \gamma w_2 - w_2 + 1 + \gamma w_1 - w_2) \right ]^2 \\
&= \frac{1}{3}(\gamma w_2 - w_1)^2 + \frac{2}{12}\left [w_2 (\gamma - 2) + \gamma w_1 \right ]^2 \\
\end{flalign}$

We can already see that the minimum value of 0 is achieved when $w_1 = w_2 = 0$, but we can also verify this by setting each term of the gradient to 0:

$\begin{flalign}
\frac{\partial \overline{\text{BE}}}{\partial w_1} &= \frac{2}{3}(\gamma w_2 - w_1) \times -1 + \frac{4}{12}(w_2(\gamma - 2) + \gamma w_1) \times \gamma\\
&= \frac{2}{3}(\gamma w_2 - w_1) \times -1 + \frac{4}{12}(w_2(\gamma - 2) + \gamma w_1) \times \gamma\\
&= \frac{\gamma}{3} (w_2 \left (\gamma - 4 \right ) + w_1 \left ( 2 + \gamma \right )) \\

&\therefore \\
w_1 &= w_2 \frac{\gamma - 4}{2 + \gamma} \\
\frac{\partial \overline{\text{BE}}}{\partial w_2} &= \frac{2}{3}(\gamma w_2 - w_1) \times \gamma + \frac{4}{12}(w_2(\gamma - 2) + \gamma w_1) (\gamma - 2)\\
&= \frac{1}{3} \left [ 2\gamma(\gamma w_2 - w_1) + (\gamma - 2)^2 w_2 + \gamma (\gamma - 2) w_1 \right ]\\
&= \frac{1}{3} \left [ \gamma w_1(\gamma - 4) + w_2(3\gamma^2 - 4\gamma + 4) \right ]\\
& \therefore \\
w_1 &= w_2 \frac{3\gamma^2 - 4\gamma + 4}{\gamma(\gamma - 4)}
\end{flalign}$

Since two fractions are not equal, this equality is only satisfied when $w_1 = w_2 = 0$ which is the same exact solution we had earlier with a Bellman error of 0.

##### Right MDP

Even though the problem appears the same with respect to the state representation, the Bellman error expression will differ for the second problem since we must separate the terms by state.

$\begin{flalign}
\overline{\text{BE}} &= \frac{1}{3} \left [ (0 + \gamma w_2 - w_1)^2 + (1 + \gamma w_1 - w_2)^2 + (-1 + \gamma w_2 - w_2)^2 \right ]\\
&\therefore \\
\frac{\partial \overline{\text{BE}}}{\partial w_1} &= \frac{2}{3} \left [ (\gamma w_2 - w_1)\times -1 + (1 + \gamma w_1 - w_2)\times \gamma \right ] \\
&= \frac{2}{3} \left [ w_1(1 + \gamma^2) - 2\gamma w_2 + \gamma \right ] \\
&\therefore \\ 
w_1 &= \frac{2\gamma w_2 - \gamma}{1 + \gamma^2} = w_2 \frac{2 \gamma}{1+\gamma^2} - \frac{\gamma}{1 + \gamma^2} \\

\frac{\partial \overline{\text{BE}}}{\partial w_2} &= \frac{2}{3} \left [ (\gamma w_2 - w_1)\times \gamma - (1 + \gamma w_1 - w_2) + (w_2(\gamma - 1) - 1)(\gamma - 1) \right ] \\
&= \frac{2}{3} \left [ -2\gamma w_1 + w_2(\gamma ^2 + 1 + (\gamma - 1)^2) - 1 - \gamma + 1\right ] \\
&= \frac{2}{3} \left [ w_2(2\gamma ^2 - 2\gamma + 2) -2\gamma w_1 - \gamma \right ] \\
&\therefore \\ 
w_1 &= \frac{2w_2(\gamma^2 - \gamma + 1) - \gamma}{2\gamma} = w_2 \frac{\gamma^2 - \gamma + 1}{\gamma} - \frac{1}{2}\\
&\therefore
w_2 \frac{2 \gamma}{1+\gamma^2} - \frac{\gamma}{1 + \gamma^2} = w_2 \frac{\gamma^2 - \gamma + 1}{\gamma} - \frac{1}{2} \\
&w_2 \left [ \frac{2 \gamma}{1+\gamma^2} -\frac{\gamma^2 - \gamma + 1}{\gamma} \right ] = \frac{\gamma}{1 + \gamma^2} - \frac{1}{2} \\
\end{flalign}$

Clearly this answer depends on $\gamma$ unlike the previous solutions. In the limit of $\gamma \rightarrow 1$, we can calculate the optimal weights according to the Bellman error:

$w_2 = 0 - (1 - 1) = 0 \implies w_1 = w_2 - \frac{1}{2} = -\frac{1}{2}$

which does not match the solution we had for the minimum value error.

"""

# ╔═╡ b5f926d8-3ea2-47a8-9093-ea366afe374f
# ╠═╡ skip_as_script = true
#=╠═╡
function min_be_weights(γ)
	w2 = (γ / (1 + γ^2)) - 0.5
	w1 = w2 * (γ^2 - γ + 1)/γ - 0.5
	(w1, w2)
end
  ╠═╡ =#

# ╔═╡ 7f6c554b-3423-4bb5-bf07-853afa4e76fb
#=╠═╡
function plot_11_4_be()
	γs = LinRange(0.2, 1, 1000)
	ws = min_be_weights.(γs)
	
	tr1 = scatter(x = γs, y = [w[1] for w in ws], name = "w1")
	tr2 = scatter(x = γs, y = [w[2] for w in ws], name = "w2")
	plot([tr1, tr2], Layout(xaxis_title = "Discount Rate", yaxis_title = "Weight Value", title = "Weights that Minimize Bellman Error"))
end
  ╠═╡ =#

# ╔═╡ 1d1afe4f-8b7f-4d81-aa97-1448b47befac
#=╠═╡
plot_11_4_be()
  ╠═╡ =#

# ╔═╡ bce9cdca-de1d-432c-a2f2-cbb6e414dfcb
#=╠═╡
@bind params_11_4 PlutoUI.combine() do Child
	md"""
	γ: $(Child(:γ, Slider(0:0.1:1, default = 1, show_value=true)))
	
	w1: $(Child(:w1, Slider(-1:0.1:1, default = 0, show_value=true)))
	
	w2: $(Child(:w2, Slider(-1:0.1:1, default = 0, show_value=true)))
	"""
end
  ╠═╡ =#

# ╔═╡ 05647633-14d2-4d5b-8c60-e236fbfeb334
#=╠═╡
function plot_11_4_value(;γ = 1, w1 = 0, w2 = 0, n = 100)
	v_true = scatter3d(x = [0], y = [1], z = [-1], mode = "markers", name = "True Value Function")
	v(w1, w2) = (x = [w1], y = [w2], z = [w2])
	ve(w1, w2) = (1/3)*(w1^2 + 2*w2^2 + 2)
	bo(w1, w2) = (x = [γ*w2], y = [1 + γ*w1], z = [-1 + γ*w2])
	be(w1, w2) = (1/3)*((γ+w2 - w1)^2 + (1 + γ*w1 - w2)^2 + (-1 + γ*w2 - w2)^2)
	xs = []
	ys = []
	zs = []
	ves = []
	bes = []
	for w1 in LinRange(-3, 3, n)
		for w2 in LinRange(-3, 3, n)
			push!(xs, w1)
			push!(ys, w2)
			push!(zs, w2)
			push!(ves, ve(w1, w2))
			push!(bes, be(w1, w2))
		end
	end

	w_minbe = min_be_weights(γ)

	param1_tr = scatter3d(x = [-3, 3], y = [0, 0], z = [0, 0], line_color = "black", name = "weight 1 axis", mode = "lines", line_width = 5)
	param2_tr = scatter3d(x = [0, 0], y = [-3, 3], z = [-3, 3], line_color = "gray", name = "weight 2 axis", mode = "lines", line_width = 5)

	min_be_tr = scatter3d(;v(w_minbe...)..., mode = "markers", name = "Minimum BE Solution")
	
	ves_tr = scatter3d(x = xs, y = ys, z = ves, mode = "markers", marker_size = 1, name = "Value Error")
	bes_tr = scatter3d(x = xs, y = ys, z = bes, mode = "markers", marker_size = 1, name = "Mean Square Bellman Error")
	
	v̂_tr = scatter3d(x = xs, y = ys, z = zs, mode = "markers", marker_size = 1, color = "blue", name = "Possible Approximate Value Functions")
	v̂_ex_tr = scatter3d(x = [w1], y = [w2], z = [w2], mode = "markers", name = "Value Function for w = [$w1, $w2])")
	ve_tr = scatter3d(x = [w1, 0], y = [w2, 1], z = [w2, -1], mode = "lines", name = "Value Error Vector for w = [$w1, $w2])", line_dash = "dot", line_color = "black", line_width = 8)
	bo_tr = scatter3d(;bo(w1, w2)..., mode = "markers", name = "Bellman Operator on Approximation")
	p1 = plot([v̂_tr, v_true, v̂_ex_tr, ve_tr, bo_tr, param1_tr, param2_tr, min_be_tr], Layout(scene = attr(xaxis_range = [-2, 2], yaxis_range = [-2, 2], zaxis_range = [-2, 2], xaxis_title = "Value A", yaxis_title = "Value B", zaxis_title = "Value C", camera = attr(eye = attr(x = 2, y = 1, z = .5), up = attr(x = 1.4, y = .95, z = 0.1))), height = 700, legend_orientation = "h"))
	# p2 = plot([ves_tr, bes_tr], Layout(scene = attr(xaxis_title = "w1", yaxis_title = "w2", zaxis_title = "value error", zaxis_range = [0, 6]), legend_orientation = "h"))
	# @htl("""
	# <div style = "display: flex;">
	# $p1 
	
	# $p2
	# </div>
	# """)
end
  ╠═╡ =#

# ╔═╡ 6b70f463-7d50-4b03-8210-b151575f98db
#=╠═╡
plot_11_4_value(;params_11_4...)
  ╠═╡ =#

# ╔═╡ 6c2fcfc8-158e-4165-9375-638d9444f70b
md"""
#### Projection Operator

For this form of approximation we can directly compute the parameters which minimize the distance from an arbitrary value function to an approximate one.  All three states have equal visit probability so we can simply sum the squared distance components and use the fact that states 2 and 3 share the same approximation:

$\begin{flalign}
d(v_1, v_2, v_3) &= (v_1 - w_1)^2 + (v_2 - w_2)^2 + (v_3 - w_2)^2 \\
&\therefore \\
\frac{\partial d}{\partial w_1} &= -2(v_1 - w_1) \implies w_1 = v_1 \\
\frac{\partial d}{\partial w_2} &= -2(v_2 - w_2) - 2(v_3 - w_2) \\
&= -2(v_2 - 2w_2 + v_3)\implies w_2 = \frac{v_2 + v_3}{2} \\
\end{flalign}$

#### Projected Bellman Error

If we apply the Bellman operator to an approximation, we get the following three state values: 

$B_\pi(\hat v(w_1, w_2)) = \{\gamma w_2, 1 + \gamma w_1, -1 + \gamma w_2 \}$

Applying the projection operator to these results in the following parameters:

$\begin{flalign}
w_1 &= \gamma w_2 \\
w_2 &= \frac{\gamma}{2} (w_1 + w_2)
\end{flalign}$

The Projected Bellman Error is the distance between the initial state values and the new ones implied by the updated parameters:

$\begin{flalign}
\overline{\text{PBE}} &= \frac{1}{3} \left [ (w_1 - \gamma w_2)^2 + (w_2 - \frac{\gamma}{2}(w_1 + w_2))^2 + (w_2 - \frac{\gamma}{2}(w_1 + w_2))^2 \right ] \\
&= \frac{1}{3} \left [ (\gamma w_2 - w_1)^2 + 2(\frac{\gamma}{2}(w_1 + w_2) - w_2)^2 \right ]
\end{flalign}$

Normally, we'd have to minimize this, but since we know the best solution will always have exactly 0 Projected Bellman Error, we can notice that when $w_1 = w_2 = 0$ this entire expression is 0.  That is the TD fixed point which also matches the minimizing value error solution.
"""

# ╔═╡ bfbe7c40-3f60-49b4-9690-ad0ee0d7db99
md"""
## 11.7 Gradient-TD Methods

We now consider SGD methods for minimizing the $\overline{\text{PBE}}$.  Remember that in the linear case, there is always an exact solution, the TD fixed point $\mathbf{w}_{\text{TD}}$, at which the $\overline{\text{PBE}}$ is zero.  This could be found by least-squares methods (Section 9.8), but only by methods of quadratic $O(d^2)$ complexity in the number of parameters.  We seek instead an SGD method, which should be $O(d)$ and have robust convergence properties.  Gradient-TD methods come close to achieving these goals, at the cost of a rough doubling of computational complexity.

 $\mathbf{D}$ is the $\vert \mathcal{S} \vert \times \vert \mathcal{S} \vert$ diagonal matrix with the $\mu(s)$ on the diagonal
 
 $\mathbf{X}$ is the $\vert \mathcal{S} \vert \times d$ matrix whose rows are the feature vectors $\mathbf{x}(s)^\top$, one for each state $s$

To derive an SGD method for the $\overline{\text{PBE}}$ (assuming linear function approximation) we begin by expanding and rewriting the objective (11.22) in matrix terms:

$\begin{flalign}
\overline{\text{PBE}}(\mathbf{w}) &= \left \vert \left \vert \Pi \overline{\delta}_\mathbf{w} \right \vert  \right \vert^2_\mu \\
&= (\Pi \overline{\delta}_{\mathbf{w}})^{\top} \mathbf{D} \Pi \overline{\delta}_\mathbf{w} \\
\end{flalign}$
"""

# ╔═╡ f3915dbd-6266-48cd-9bc0-a40b39b8dd22
md"""
### TDC: TD(0) with gradient correction

$\mathbf{v}_{t+1} \doteq \mathbf{v}_t + \beta \rho_t \left ( \delta_t - \mathbf{v}_t^\top\mathbf{x}_t \right ) \mathbf{x}_t$

$\begin{flalign}
\mathbf{w}_{t+1} &= \mathbf{w}_t + \alpha \left ( \mathbb{E} [\rho_t \delta_t \mathbf{x}_t ] - \gamma \mathbb{E} \left [ \rho_t \mathbf{x}_{t+1} \mathbf{x}_t ^\top \right ] \mathbf{v}_t \right ) \\
&\approx  \mathbf{w}_t + \alpha \rho_t \left ( \delta_t \mathbf{x}_t - \gamma \mathbf{x}_{t+1} \mathbf{x}_t ^ \top \mathbf{v}_t \right )\tag{sampling}
\end{flalign}$
"""

# ╔═╡ 0ac1a38b-92bd-4000-8da6-295fcd1f028f
#since we're sampling from the behavior policy, this projected Bellman error is not the true one we care about for the on-policy distribution.  however, at convergence, the PBE is uniquely 0 which means that the projected Bellman error is zero for every state. That means that the fixed point won't change even if the weighting of states changes so the different visit distribution won't actually change the convergence to the TD fixed point.  So we don't end up correcting for the difference in state visit distribution.  Instead we use an objective that is 0 regardless of it.

# ╔═╡ 5b69d857-d3c1-4a1a-9295-1691bbc73cf6
#this update rule won't match semi-gradient TD(0) even though it converges to the same point in the on policy case.  the difference is that the semi-gradient method only converges when the states are visited according to the on policy distribution and this converges regardless of how the updates are distributed.  You could for example visit the same state over and over which would affect the value of v and prevent divergence.  Actually if the semi-gradient method converges then it will converge to the TD fixed point even under the off-policy visits.  The problem is there are times when it diverges but it doesn't converge to something that's the wrong value.  This correction term only serves to prevent divergence, not change the convergence value.  This stems again from the unique 0 value of the PBE

# ╔═╡ 4fd5b88f-eb4b-415b-9f8e-781dd4e194d0
md"""
### *TDC Implementation*
"""

# ╔═╡ 5f7635d8-42a3-4b74-b027-6a870d6e7d47
function tdc_estimation(mdp::StateMDP, γ::T, π!::Function, b!::Function, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_state_representation!::Function; s0::S = mdp.initialize_state(), calculate_error::Function = (v̂, s) -> zero(T), α = one(T)/10, β = one(T)/10, init_value::T = zero(T), parameters = initialize_linear_parameters(feature_vector, init_value), save_parameter_history = false) where {T<:Real, S}
	s = mdp.initialize_state()
	ep = 1
	step = 1
	d = length(feature_vector)

	state_representation1 = deepcopy(feature_vector)
	state_representation2 = deepcopy(feature_vector)

	parameter_history = Vector{Vector{T}}()

	save_parameter_history && push!(parameter_history, copy(parameters))

	π_dist = zeros(T, length(mdp.actions))
	b_dist = zeros(T, length(mdp.actions))
	v = zeros(T, d)
	
	update_state_representation!(state_representation1, s)
	episode_errors = Vector{T}()
	err = zero(T)
	epstep = 1
	
	while (ep <= max_episodes) && (step <= max_steps)
		π!(π_dist, state_representation1)
		b!(b_dist, state_representation1)
		i_a = sample_action(b_dist)
		(r, s′) = mdp.ptf(s, i_a)
		ρ = π_dist[i_a] / b_dist[i_a]
		if mdp.isterm(s′)
			v̂2 = zero(T)
		else
			update_state_representation!(state_representation2, s′)
			v̂2 = linear_value_function(state_representation2, parameters)
		end

		if !iszero(ρ)
			δ = r + γ*v̂2 - linear_value_function(state_representation1, parameters) 
			c1 = α*ρ*δ
			update_params_with_gradient!(parameters, c1, state_representation1)
			v̂1_v = linear_value_function(state_representation1, v)
			if !mdp.isterm(s′)
				c2 = -α*ρ*γ*v̂1_v
				update_params_with_gradient!(parameters, c2, state_representation2)
			end
			# parameters .+= α .* ρ .* (δ .* state_representation1 .- γ .* state_representation2 .* linear_value_function(state_representation1, v))
			c3 = ((β*ρ) * (δ - v̂1_v))
			update_params_with_gradient!(v, c3, state_representation1)
			# v .+= ((β*ρ) * (δ - v̂1_v)) .* state_representation1
		end

		save_parameter_history && push!(parameter_history, copy(parameters))
		s = s′
		epstep += 1
		if mdp.isterm(s′)
			s = mdp.initialize_state()
			ep += 1
			ep_step = 1
			update_state_representation!(state_representation1, s)
		else
			s = s′
			state_representation1 .= state_representation2
		end
		step += 1
	end

	function v(s::S)
		x = deepcopy(feature_vector)
		update_state_representation!(x, s)
		linear_value_function(x, parameters)
	end

	# function v(states::AbstractVector{S})
	# 	x = zeros(T, d)
	# 	input = zeros(T, length(states), d)
	# 	for i in eachindex(states)
	# 		update_state_representation!(x, states[i])
	# 		for j in 1:d
	# 			input[i, j] = x[j]
	# 		end
	# 	end
	# 	input*parameters
	# end
	
	return (parameters = parameters, value_estimate = v, episode_errors = episode_errors, parameter_history = parameter_history)
end

# ╔═╡ 6fd223aa-3d28-47e9-ba4f-391be5362521
md"""
### Revisiting Baird's Counter Example

There are 7 states, 2 actions, and 8 parameters.  We can analytically write down the different errors in terms of the 8 parameter values for the target policy which always takes the solid action.  Also, the on-policy distribution for the target policy is $\mu(7) = 1$ so we only need to consider there error for that state.

#### Value Error

The true values are 0 for every state, so the value error is simply:

$\overline{\text{VE}} = (w_7 + 2w_8)^2$ 

which is minimized when $w_7 = -2w_8$

#### Mean Squared Bellman Error

$(0 + \gamma (w_7 + 2 w_8) - w_7 - 2 w_8)^2 = ((w_7 + 2w_8)(\gamma - 1))^2$

If we minimize this with respect to $w_7$ and $w_8$

$0 = 2(\gamma - 1)(w_7 + 2w_8) \implies w_7 = -2w_8$

$0 = 4(\gamma - 1)(w_7 + 2w_8) \implies w_7 = -2w_8$

#### Projected Bellman Error

Normally, there are fewer parameters than states, so there is a single projection point which minimizes the distance from any true state value function to the approximation function.  Here we have one more parameter, so there is a whole family of projections that are equally good and reproduce the exact value function.  Again, the only state value error that matters is for state 7 whose value is given as

$\begin{flalign}
v_7 &= w_7 + 2w_8 \\
\end{flalign}$

This distance is 0 for any parameters that satisfy this equality.  To calculate the mean squared projected Bellman Error, we first take state 7 and apply the Bellman operator to it:

$B_\pi(\hat v_7) = \gamma(w_7 + 2 w_8)$

Given this new value, which parameters satisfy the projection?  We can pick any new value which satisfies $\gamma (w_7 + 2 w_8) = w_7^\prime + 2w_8^\prime$.  One easy option that satisfies this relationship is just $w_7^\prime = \gamma w_7$ and $w_8^\prime = \gamma w_8$ so each parameter can just be multiplied by $\gamma$ to find the projection point.  Now the projected Bellman error is the distance between these two values:

$\overline{\text{PBE}} = ((w_7 + 2w_8) - \gamma (w_7 + 2 w_8))^2 = ((w_7 + 2 w_8)(1-\gamma))^2 = \overline{\text{VE}} (1-\gamma)^2$

which is the same as the Bellman Error since the projection in this case can just match any value from the Bellman operator, and we already know this is minimized when $w_7 = -2w_8$ which is the same value that minimizes the value error.
"""

# ╔═╡ 773c82f4-ba00-4907-953a-c1d7d6eb3478
#=╠═╡
function figure_11_5(;steps = 2_000, γ::Float32 = 0.99f0, α::Float32 = 0.0005f0, β::Float32 = α*10)

	out = tdc_estimation(baird_state_mdp, γ, π_baird!, b_baird!, 1, steps, zeros(Float32, 8), baird_update_state_vector!; parameters = Float32.([1, 1, 1, 1, 1, 1, 10, 1]), α = α, β = β, save_parameter_history = true)

	ve(params) = (params[7] + 2*params[8])^2

	param_traces = [scatter(y = [a[i] for a in out.parameter_history], name = latexstring("w_$i")) for i in 1:8]
	ve_trace = scatter(y = [sqrt(ve(a)) for a in out.parameter_history], name = L"\sqrt{\overline{\text{VE}}}")
	pbe_trace = scatter(y = [sqrt(ve(a))*(1-γ) for a in out.parameter_history], name = L"\sqrt{\overline{\text{PBE}}}")
	baseline_tr = scatter(x = [0, steps], y = [0, 0], mode = "lines", line_dash = "dash", line_color = "black", showlegend=false)
	traces = [param_traces; ve_trace; pbe_trace; baseline_tr]
	@htl("""
	<div style = "width: min(90%, 800px);">
	$(plot(traces, Layout(yaxis_range = [-3, 10], title = "TDC", xaxis_title = "Steps")))
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ aee362e3-b1a0-4378-9af0-1ea1ed6580fe
#=╠═╡
figure_11_5(;α = 0.005f0, γ = 0.99f0)
  ╠═╡ =#

# ╔═╡ 9a21ebe8-186f-4ab4-b0ae-8a0c668c3f92
md"""
The convergence is very slow due to $\gamma$ being close to 1.  Even though the value error remains high, the fact that the observed projected Bellman error is shrunk by a factor of $(1 - \gamma)^2$ makes it nearly 0 even while the value error is close to 4.  If we shrink the value of $\gamma$ towards 0, then both errors converge to 0 much faster.
"""

# ╔═╡ e523bc1f-f2ad-49a0-ae3e-aa79db6e8043
md"""
### TDC Generalized Policy Iteration

We can use the TDC algorithm to try to do a better job of Q-learning with linear approximation.  Previously, we had used some value function approximation to learn action values and then try to follow the greedy policy while updating the values.  With sarsa this was an on-policy method, but in the case of Q-learning we used the parameter update that takes the maximum action value while still following the $\epsilon$-greedy policy.  Usually if $\epsilon$ isn't too large, this does not cause diverging weights, but there is always a risk of that happening.  We can try to use the TDC method instead learn the greedy value function while still following the $\epsilon$-greedy one.  We can update the policy after one or more update steps while tracking the approximation vectors used in TDC.  Unlike in TDC estimation, we will not have an explicit target policy.  The behavior policy will usually just be the $\epsilon$-greedy policy and change throughout training, but this method also means we can have a static behavior policy such as one that visits all states with equal probability.
"""

# ╔═╡ 9fe4b328-6487-48e0-9e28-5cdae655204e
begin
	function update_feature_vector!(v1::V, v2::V) where {T<:Real, V <: Vector{T}}
		v1 .= v2
	end

	function update_feature_vector!(v1::V, v2::V) where {V <: BinaryFeatureVector}
		update_binary_feature_vector!(v1, v2)
	end

	function update_feature_vector!(v1::V, v2::V) where {V <: StateAggregationFeatureVector}
		v1.group_index = v2.group_index
		return v1
	end
end

# ╔═╡ 12068dea-798d-4cc3-86f0-07b7315caa91
function tdc_control(mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_state_representation!::Function; s0::S = mdp.initialize_state(), calculate_error::Function = (v̂, s)->zero(T), α = one(T)/10, β = one(T)/10, ϵ = one(T)/10, init_value = zero(T), parameters = initialize_linear_parameters(feature_vector, mdp, init_value), save_parameter_history = false) where {T<:Real, S}
	s = mdp.initialize_state()
	ep = 1
	step = 1
	d = length(feature_vector)

	state_representation1 = deepcopy(feature_vector)
	state_representation2 = deepcopy(feature_vector)

	parameter_history = Vector{Matrix{T}}()

	save_parameter_history && push!(parameter_history, deepcopy(parameters))

	action_values = zeros(T, length(mdp.actions))
	policy = zeros(T, length(mdp.actions))
	v = zeros(T, d)
	
	update_state_representation!(state_representation1, s)
	episode_errors = Vector{T}()
	err = zero(T)
	epstep = 1
	
	while (ep <= max_episodes) && (step <= max_steps)
		(qmax, i_a_max) = update_linear_action_values!(action_values, state_representation1, parameters)
		make_ϵ_greedy_policy!(policy; ϵ =  ϵ)
		i_a = sample_action(policy)
		v̂1 = action_values[i_a]
		(r, s′) = mdp.ptf(s, i_a)
		ρ = (i_a_max == i_a) / action_values[i_a]
		q′ = if mdp.isterm(s′)
			r
		else
			update_state_representation!(state_representation2, s′)
			r + γ*update_linear_action_values!(action_values, state_representation2, parameters)[1]
		end

		if !iszero(ρ)
			δ = q′ - v̂1 
			c1 = α*ρ*δ
			update_params_with_gradient!(parameters, c1, state_representation1, i_a)
			v̂′ = linear_value_function(state_representation1, v)
			if !mdp.isterm(s′)
				c2 = -α*ρ*γ*v̂′
				update_params_with_gradient!(parameters, c2, state_representation2, i_a)
			end
			c3 = β*ρ*(δ - v̂′)
			update_params_with_gradient!(v, c3, state_representation1)
			# parameters[i_a] .+= α .* ρ .* (δ .* state_representation1 .- γ .* state_representation2 .* dot(state_representation1, v))
			# v .+= ((β*ρ) * (δ - dot(v, state_representation1))) .* state_representation1
		end

		save_parameter_history && push!(parameter_history, deepcopy(parameters))
		s = s′
		epstep += 1
		if mdp.isterm(s′)
			s = mdp.initialize_state()
			ep += 1
			ep_step = 1
			update_state_representation!(state_representation1, s)
		else
			s = s′
			update_feature_vector!(state_representation1, state_representation2)
		end
		step += 1
	end

	q̂, form_kwargs = form_value_function(mdp, update_state_representation!, update_linear_action_values!, feature_vector, parameters)
	
	return (value_function = q̂, parameters = parameters, episode_errors = episode_errors, parameter_history = parameter_history, form_kwargs = form_kwargs)
end

# ╔═╡ 85bf8c44-348b-4825-b89a-33ec7614bb25
function tdc_dp_control(mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_state_representation!::Function; s0::S = mdp.initialize_state(), calculate_error::Function = (v̂, s)->zero(T), α = one(T)/10, β = one(T)/10, ϵ = one(T)/10, init_value = zero(T), parameters = initialize_linear_parameters(feature_vector, init_value), save_parameter_history = false) where {T<:Real, S, A, P <: StateMDPTransitionDistribution, F1, F2, F3}
	s = mdp.initialize_state()
	ep = 1
	step = 1
	d = length(feature_vector)
	
	state_representation1 = deepcopy(feature_vector)
	state_representation2 = deepcopy(feature_vector)
	state_representation3 = deepcopy(feature_vector)

	parameter_history = Vector{Vector{T}}()

	save_parameter_history && push!(parameter_history, deepcopy(parameters))

	action_values = zeros(T, length(mdp.actions))
	v = zeros(T, d)
	
	update_state_representation!(state_representation1, s)
	episode_errors = Vector{T}()
	err = zero(T)
	epstep = 1
	
	while (ep <= max_episodes) && (step <= max_steps)
		(qmax, i_a_max) = update_action_values!(action_values, s, state_representation3, update_state_representation!, linear_value_function, parameters, mdp, γ)
		make_ϵ_greedy_policy!(action_values; ϵ =  ϵ)
		i_a = sample_action(action_values)
		(r, s′) = mdp.ptf(s, i_a)
		if !mdp.isterm(s′)
			update_state_representation!(state_representation2, s′)
		end
		ρ = (i_a_max == i_a) / action_values[i_a]

		if !iszero(ρ)
			v̂1 = linear_value_function(state_representation1, parameters)
			δ = qmax - v̂1
			update_params_with_gradient!(parameters, α*ρ*δ, state_representation1)
			v̂_v = linear_value_function(state_representation1, v)
			if !mdp.isterm(s′)
				update_params_with_gradient!(parameters, -α*ρ*γ*v̂_v, state_representation2)
			end
			update_params_with_gradient!(v, β*ρ*(δ - v̂_v), state_representation1)
			# parameters .+= α .* ρ .* (δ .* state_representation1 .- γ .* state_representation2 .* dot(state_representation1, v))
			# v .+= ((β*ρ) * (δ - dot(v, state_representation1))) .* state_representation1
		end

		save_parameter_history && push!(parameter_history, deepcopy(parameters))
		s = s′
		epstep += 1
		if mdp.isterm(s′)
			s = mdp.initialize_state()
			ep += 1
			ep_step = 1
			update_state_representation!(state_representation1, s)
		else
			s = s′
			update_feature_vector!(state_representation1, state_representation2)
		end
		step += 1
	end

	q̂, form_kwargs = form_value_function(mdp, γ, update_state_representation!, linear_value_function, feature_vector, parameters)
		
	return (value_function = q̂, parameters = parameters, episode_errors = episode_errors, parameter_history = parameter_history, form_kwargs = form_kwargs)
end

# ╔═╡ cfe7ed5a-514a-4753-b029-9118813fa0ed
md"""
### *TDC Control on Baird Counter Example*
"""

# ╔═╡ 344a94a8-58ad-4cb5-ad1c-dcf779a6ea76
#=╠═╡
function tdc_control_baird(;winit = Float32.([1, 1, 1, 1, 1, 1, 10, 1]), maxsteps = 2_000, γ = 0.99f0, ϵ = 0.5f0, α = 0.01f0, state_index = 1)
	winit = Float32.([1, 1, 1, 1, 1, 1, 10, 1])
	sarsa_output = semi_gradient_sarsa_linear(baird_state_mdp, γ, 1000, maxsteps, zeros(Float32, 8), baird_update_state_vector!; ϵ = ϵ, α = α, save_parameter_history = true, parameters = [copy(winit) copy(winit)])

	tdc_control_output = tdc_control(baird_state_mdp, γ, 1000, maxsteps, zeros(Float32, 8), baird_update_state_vector!; ϵ = ϵ, α = α, save_parameter_history = true, parameters = [copy(winit) copy(winit)])

	tdc_dp_control_output = tdc_dp_control(baird_state_mdp, γ, 1000, maxsteps, zeros(Float32, 8), baird_update_state_vector!; ϵ = ϵ, α = α, save_parameter_history = true, parameters = copy(winit))
	
	tdc_parameter_traces = [scatter(y = [tdc_control_output.parameter_history[i][i_w, 2] for i in 1:maxsteps], name = latexstring("w_$i_w")) for i_w in 1:8]


	tdc_value_traces = [scatter(y = [tdc_control_output.value_function(i_s; parameters = tdc_control_output.parameter_history[i]).maximizing_value for i in 1:maxsteps], name = "State $(i_s) value") for i_s in 1:7]

	tdc_dp_value_traces = [scatter(y = [tdc_dp_control_output.value_function(i_s; parameters = tdc_dp_control_output.parameter_history[i]).maximizing_value for i in 1:maxsteps], name = "State $(i_s) value") for i_s in 1:7]

	tdc_dp_parameter_traces = [scatter(y = [tdc_dp_control_output.parameter_history[i][i_w] for i in 1:maxsteps], name = latexstring("w_$i_w")) for i_w in 1:8]

	tdc_value_plot = plot([tdc_value_traces; tdc_parameter_traces])

	tdc_dp_value_plot = plot([tdc_dp_value_traces; tdc_dp_parameter_traces])

	md"""
	$tdc_value_plot $tdc_dp_value_plot
	"""

	# tdc_dp_control_output.parameter_history
	
	# p1 = plot([scatter(y = [a[1][i] for a in sarsa_output.parameter_history], name = "Parameter $i", showlegend=false) for i in 1:8], Layout(xaxis_title = "Step", yaxis_title = "Parameter Value"))
	# p2 = plot([scatter(y = [a[1][i] for a in q_learning_output.parameter_history], name = "Parameter $i") for i in 1:8])

	# baird_values(w::Vector{T}) where T<:Real = [2*w[1] + w[8], 2*w[2] + w[8], 2*w[3]+w[8], 2*w[4] + w[8], 2*w[5] + w[8], 2*w[6] + w[8], w[7] + 2*w[8]]
	
	# q_value_history1 = [baird_values(w[1]) for w in q_learning_output.parameter_history]
	# q_value_history2 = [baird_values(w[2]) for w in q_learning_output.parameter_history]

	# sarsa_value_history1 = [baird_values(w[1]) for w in sarsa_output.parameter_history]
	# sarsa_value_history2 = [baird_values(w[2]) for w in sarsa_output.parameter_history]

	# p3 = plot([scatter(y = [a[i] for a in q_value_history1], name = "State $i") for i in 1:7])
	# p4 = plot([scatter(y = [a[i] for a in q_value_history2], name = "State $i") for i in 1:7])
	# p5 = plot(scatter(y = [Float32(q_value_history1[i][state_index] - q_value_history2[i][state_index]) for i in 1:length(q_learning_output.parameter_history)]))
	# p6 = plot(scatter(y = [Float32(sarsa_value_history1[i][state_index] - sarsa_value_history2[i][state_index]) for i in 1:length(sarsa_output.parameter_history)]))
	
	# md"""
	# $([p1 p2; p3 p4; p5 p6])
	# """
end
  ╠═╡ =#

# ╔═╡ 4ffc878c-d856-4183-96ea-ef77447b8a5c
#=╠═╡
tdc_control_baird()
  ╠═╡ =#

# ╔═╡ 4b932772-b69f-427a-b8f4-ee52fba4696a
#add parameter study comparing TDC and semi-gradient q learning

# ╔═╡ d88a5c6b-8bcb-44d9-a52a-994e945aecad
# ╠═╡ skip_as_script = true
#=╠═╡
const mountain_car_mdp = MountainCarTask.mdp
  ╠═╡ =#

# ╔═╡ 321474b9-364d-4b91-bf1c-601a9b8ad470
# ╠═╡ skip_as_script = true
#=╠═╡
const mountain_car_dist_mdp = MountainCarTask.dist_mdp
  ╠═╡ =#

# ╔═╡ 9f85ad2d-f417-4463-9894-53f0eead4d83
#=╠═╡
function mountaincar_dist_test(max_episodes::Integer, α::Float32, ϵ::Float32; num_tiles = 24, num_tilings = 32, max_steps = typemax(Int64), kwargs...)
	setup = setup_mountain_car_tiles((1f0/num_tiles, 1f0/num_tiles), num_tilings)
	v = setup.feature_vector
	semi_gradient_dp_linear(mountain_car_dist_mdp, 1f0, max_episodes, max_steps, v, setup.update_feature_vector!; α = α, ϵ = ϵ, kwargs...)
end
  ╠═╡ =#

# ╔═╡ c25fa03c-0464-4e07-a777-9ee5f732b0a2
#=╠═╡
(v̂_mountain_car_dp, episode_rewards_dp, episode_steps_dp) = mountaincar_dist_test(100, 0.001f0/32, 0.9f0)
  ╠═╡ =#

# ╔═╡ bf5d1782-5109-4cfd-8744-82ac80b5bc45
#=╠═╡
function mountaincar_tdc_test(max_episodes::Integer, α::Float32, ϵ::Float32; num_tiles = 24, num_tilings = 32, max_steps = typemax(Int64), kwargs...)
	setup = setup_mountain_car_tiles((1f0/num_tiles, 1f0/num_tiles), num_tilings)
	v = setup.feature_vector
	tdc_dp_control(mountain_car_dist_mdp, 1f0, max_episodes, max_steps, v, setup.update_feature_vector!; α = α, ϵ = ϵ, kwargs...)
end
  ╠═╡ =#

# ╔═╡ 41bd4480-7672-4adf-924a-5bc1e9d4b45e
#=╠═╡
tdc_out = mountaincar_tdc_test(10000, 0.000002f0, 0.9f0; max_steps = 500_000, β = 1f-8)
  ╠═╡ =#

# ╔═╡ a5ec3b99-af5a-42e5-8dff-533b45e50af5
#=╠═╡
function show_mountaincar_trajectory(π::Function, max_steps::Integer, name)
	states, actions, rewards, sterm, nsteps = runepisode(mountain_car_mdp; π = π, max_steps = max_steps)
	positions = [s[1] for s in states]
	velocities = [s[2] for s in states]
	tr1 = scatter(x = positions, y = velocities, mode = "markers", showlegend = false)
	tr2 = scatter(y = positions, showlegend = false)
	tr3 = scatter(y = [MountainCarTask.actions[i] for i in actions], showlegend = false)
	p1 = plot(tr1, Layout(xaxis_title = "position", yaxis_title = "velocity", xaxis_range = [-1.2, 0.5], yaxis_range = [-0.07, 0.07]))
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

# ╔═╡ a9443d53-1eae-4eab-b001-904be1523ca4
#=╠═╡
show_mountaincar_trajectory(s -> tdc_out.value_function(s).maximizing_action, 500, "TDC Learned Policy")
  ╠═╡ =#

# ╔═╡ 5c87ea86-aec5-42d3-9423-4cd9d14dbc97
#=╠═╡
show_mountaincar_trajectory(s -> v̂_mountain_car_dp(s).maximizing_action, 1_000, "DP Learned Policy")
  ╠═╡ =#

# ╔═╡ 3d48d0e2-353c-44cf-a51b-1fad1b0002d2
#=╠═╡
function plot_mountaincar_values(v̂_mountain_car, π; n1 = 100, n2 = 100)
	xvals = LinRange(-1.2f0, 0.5f0, n1)
	vvals = LinRange(-0.07f0, 0.07f0, n2)
	values = zeros(Float32, n1, n2)
	actions = zeros(Float32, n1, n2)
	for (i, x) in enumerate(xvals)
		for (j, v) in enumerate(vvals)
			v̂ = v̂_mountain_car((x, v))
			values[j, i] = v̂
			actions[j, i] = π((x, v))
		end
	end
	p1 = plot(heatmap(x = xvals, y = vvals, z = values), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Learned Value Function"))
	p2 = plot(heatmap(x = xvals, y = vvals, z = actions, colorscale = "rb", showscale = false), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Policy (blue = accelerate left, <br>red = accelerate right, gray = no acceleration)"))
	[p1 p2]
end
  ╠═╡ =#

# ╔═╡ 38e6ef4c-63c1-4df5-9451-f40df4fe57e7
#=╠═╡
plot_mountaincar_values(s -> tdc_out.value_function(s).maximizing_value, s -> tdc_out.value_function(s).maximizing_action)
  ╠═╡ =#

# ╔═╡ 45e8699f-18ca-47a6-97eb-f855950b326d
md"""
# Dependencies
"""

# ╔═╡ edd27759-c2c5-4b5a-92b2-590f8673461a
# ╠═╡ skip_as_script = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 4c505f66-0c2c-4f59-858d-bd16c59f3397
#=╠═╡
@htl("""
<div style="display: flex; align-items: center; background-color: lightgray; color: black; height: 70px">
<div>0</div>
<div class="loop" style = "transform:rotate(-135deg);"></div>
<div class="backup" style="transform: scale(130%)">
	<div class="circlestate"></div>
</div>
<div class="loop"></div>
<div>2</div>
</div>
<style>
	.loop {
		display: flex;
		border-width: 2px 0px 0px 2px;
		border-style: solid;
		border-color: black;
		width: 38px;
		height: 28px;
		border-radius: 50% 50% 50% 15%;
		transform: translateY(0px) rotate(45deg);
	}
	.loop::before {
		content: '';
		position: relative;
		width: 5px;
		height: 5px;
		border-width: 0px 0px 3px 3px;
		border-style: solid;
		border-color: black;
		transform: translateX(-4px) translateY(17px) rotate(-45deg)
	}
	.loop::after {
		content: '';
		border-width: 0px 2px 2px 0px;
		border-style: solid;
		border-color: black;
		width: 38px;
		height: 28px;
		border-radius: 50% 50% 50% 0%;
	}
</style>
""")
  ╠═╡ =#

# ╔═╡ faba7178-bc20-4d93-87e3-26541851b1ad
HTML("""
<style>

	.backup {
		margin: 5px;
	}
	.backup, .backup * {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		color: black;
	}
	.circlestate, .circleaction {
		margin: 0;
	}
	.circlestate::before {
		content: 'w';
		display: inline-block;
		border: 1px solid black;
		border-radius: 50%;
		height: 20px;
		width: 20px;
		background-color: white;
	}
	.circleaction::before {
		content: '';
		display: inline-block;
		border: 1px solid black;
		border-radius: 50%;
		height: 10px;
		width: 10px;
		background-color: black;
	}
	.arrow {
		display: flex;
		justify-content: center;
		align-items: center;
	}
	.arrow::before {
		content: '';
		display: inline-block;
		width: 2px;
		height: 30px;
		background-color: black;
		margin-bottom: 0px;
	}
	.arrow::after {
		content: '';
		display: inline-block;
		width: 4px;
		height: 4px;
		border-bottom: 3px solid black;
		border-right: 3px solid black;
		transform: translateY(-5px) rotate(45deg);
		position: relative;
	}
	.term::before {
		content: '';
		display: inline-block;
		width: 20px;
		height: 20px;
		border: 2px solid black;
		background-color: rgb(50, 50, 50);
	}
</style>
""")


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
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
BenchmarkTools = "~1.6.3"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
PlutoDevMacros = "~0.9.1"
PlutoPlotly = "~0.6.5"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.73"
StatsBase = "~0.34.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "ef643e1538d2ce5994d4d92f93191f1b21219d89"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

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
git-tree-sha1 = "980f01d6d3283b3dbdfd7ed89405f96b7256ad57"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "2.0.1"

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

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6c72198e6a101cccdd4c9731d3985e904ba26037"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.1"

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
version = "1.6.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "d60eb76f37d7e5a40cc2e7c36974d864b82dc802"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.1"

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

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "277779adfedf4a30d66b64edc75dc6bb6d52a16e"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.6"

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
version = "8.11.1+1"

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

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

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

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.5.20"

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
version = "3.5.1+0"

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
version = "1.12.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "28278bb0053da0fd73537be94afd1682cc5a0a83"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.21"

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
git-tree-sha1 = "1cb861c9295d79dc6e23170d4b33bce013f69643"
uuid = "a0499f29-c39b-4c5c-807c-88074221b949"
version = "0.9.1"

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
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "3faff84e6f97a7f18e0dd24373daa229fd358db5"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.73"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

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

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

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

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "a136f98cefaf3e2924a66bd75173d1c891ab7453"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.7"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

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
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.5.0+2"
"""

# ╔═╡ Cell order:
# ╟─46076214-2d52-4289-98e6-8b74c337f7d7
# ╟─a23b5ab9-8963-426d-9672-cf99a71d8884
# ╟─434045f4-865e-4993-913e-938b6cdf7a3f
# ╟─2c668d98-453d-482b-8980-bfbccf82dd86
# ╟─676ea0b7-b27c-4c62-88fd-8d892b57c6b2
# ╠═255bb3cc-5a26-4817-b515-3b760c351f2e
# ╠═bea94375-277b-4f38-ad9d-4fa7fc646364
# ╟─e6e606c4-39d7-4b87-bd1a-b5799281f033
# ╟─8d463e53-12ee-441c-bd14-e8b377fcdced
# ╟─29364905-2458-426a-999c-210cd3c60263
# ╠═7b913193-0bcc-43b3-b9b2-908a9c29524e
# ╠═822bc5a4-8aa4-454f-8da0-a1d081b93c55
# ╠═3e2afc15-0e7b-4a5d-8e38-91e70cfa87e5
# ╠═e7abb675-6697-4f23-a8b7-01eb2231f6d1
# ╟─4996bfd5-137c-4b31-9f80-e463ca5d2b8a
# ╠═ddda80bc-fd6b-4110-83b3-aaf995ce8a71
# ╟─6ae6a0c3-6ba6-4512-8ce7-ad98758f835f
# ╠═05a77bfa-2573-4b31-b108-ad4351902d11
# ╟─25572d20-89a3-4e08-948b-d678bc978b70
# ╠═c15bedae-1231-4421-8abe-ab2fa3cb34ec
# ╟─aca67da3-b936-4233-88ea-77987e31b90c
# ╟─ac53f74b-3909-44b7-acf2-d2dd5f2e57cc
# ╟─21174c39-7b4d-48ce-80b1-a2c72be9239a
# ╟─3c2bb063-db4f-4680-bf4a-2f3ac10a174d
# ╠═c044414e-77d5-4a54-865e-dca4a879cd30
# ╠═d2033a7d-3d9d-4983-8fd1-b4e6ee015080
# ╠═1ba56556-2ac7-4d23-98c3-0d3fb54ec3d6
# ╠═1be8182a-c183-486c-9991-bcc325e75449
# ╠═2feb4657-3377-434f-bf8a-400cfcfe9fef
# ╠═3238aaa1-92aa-4d80-af22-4e237be9f0fc
# ╟─1e010e8e-2dde-4228-b914-fdc120fa91ca
# ╟─4b7b7bb6-8484-42ac-983f-ec33dbf2c73e
# ╟─1074eb62-a5ee-43cb-a1a6-fe2bbc196f72
# ╟─cad27ba6-aa01-41e2-902b-ff411037cf0f
# ╠═5d85cf97-3e46-4ace-8246-2fc73a93cc2f
# ╠═12ff2e46-fa3e-4fe8-9a1f-58afc2a43c25
# ╠═84997a09-960f-4116-9045-74cb2e0e9d03
# ╠═77ca116d-675d-4db5-8a68-53d1085528f4
# ╠═8efa076f-d14d-44ab-bc03-e7ff964bc3b3
# ╠═0b146651-a99f-489b-92f5-b5bd74d275fe
# ╠═1853cb36-a97d-4922-92c2-02261843c761
# ╠═d1cedda0-1ebf-42a6-b2f8-7df665252c08
# ╠═c3ad2cdc-6e85-48a7-a746-c7599f80a126
# ╠═ad6c8986-8fb0-4682-ade8-ebb76b4c829a
# ╟─fcef571c-9656-42e4-9a85-e13c3ed51edb
# ╟─6965a4d3-5422-4a3e-8eba-fa101cb1b16d
# ╟─d9f38410-a1e4-4e10-a16b-ee933da553d2
# ╟─17307c42-3175-4cfc-b9b7-e5d21e02d64a
# ╟─e39098da-a3df-47a0-867d-ccaf1a5a54f3
# ╟─bd2abdf1-725a-491d-b6f3-5a15ae51762c
# ╟─40a966dd-d8c1-486e-bed7-5a0094778f31
# ╟─256efb33-1b85-4fbb-be51-e43384fd149c
# ╟─c63b1e1c-db89-4d47-af39-e353dda0e50b
# ╟─f82090ed-8b6b-4b2e-89c9-26cc0ef4b30a
# ╟─3dade251-ddf7-463e-8d55-1c37e6d8ac9a
# ╠═3280e9dc-e0e4-4a18-88a5-0a4ac188e71c
# ╠═5960d4a9-5493-41d8-a98f-e9d91e34fa79
# ╠═14fe90c3-50a7-4098-8626-b2d2a4b617ca
# ╠═e2751f9f-1554-4cb2-934e-0e032ad9a244
# ╠═e28a8728-bf1d-4a94-89f3-24d15d81425a
# ╠═fab9d8f8-8dbc-450e-8a40-7b83b5a236d0
# ╟─4965afd6-b7b9-4fa9-ad1c-9744d5b9727d
# ╠═b68b6bf1-78ad-4339-a872-993e9d9fdfc2
# ╠═86c51ac5-10d7-4652-9219-d514cfe07bb6
# ╟─6a654e0e-2809-4e46-989f-815de38c8bf6
# ╟─b62b78f5-4721-4fb6-b056-cc4dae9eae9f
# ╟─c79e0f4d-6858-4f9c-960c-08f3c247566d
# ╟─3bd92abe-cb9d-4e71-af82-096e6fce17a5
# ╠═2dff23c5-0641-4377-8d7a-a4e2d3459b2f
# ╠═f2bc7752-d263-4f11-afec-40f82d5188ec
# ╠═ee4ba290-9a25-44ba-abe0-44f4e39a1099
# ╠═672c91f9-6df1-4834-a2ae-ead92d245cda
# ╠═e91cf338-fc0b-4d05-828d-6302c6acc924
# ╟─b3a62cf0-f1f2-42f3-978c-14a09b20eb75
# ╟─677532a9-82a7-439b-b05a-013c92dd2f60
# ╟─5a083034-5075-46fe-a988-4dab0011c9a4
# ╟─94adc5c2-9ade-4b66-8814-705c2cc23534
# ╠═3560cece-1420-41e5-8590-54041d210996
# ╠═bfc7e5e3-2f2a-49f8-b1e8-c86e6d16b160
# ╟─88a415dd-0fe8-493e-9446-75b909b3f68c
# ╟─88aa7985-dab6-4bd0-8685-321e1499f830
# ╠═d90dcfef-325c-4227-84a9-671f01b7383a
# ╟─9145a0ff-bfa8-47d4-91c6-118ef17da5b8
# ╠═65c4f33b-0c7d-4003-9a69-9e4d1147641e
# ╟─c401d8fc-704b-42e7-bbb2-0322329341fe
# ╟─874003a9-40f0-4d73-8070-085143487d12
# ╠═5d84f27f-4d72-46e9-99e8-2e5aa361c5f9
# ╠═d16d7e17-3662-4f5e-a98c-1c423399feed
# ╟─860de14e-751c-483c-b570-3b1ae938a1b3
# ╠═c5c814ce-7524-4742-880a-9827153b0cd2
# ╠═6def32ca-da3d-438c-a677-40247afd2119
# ╠═620d9b39-cf3f-4937-b3f1-332558aef6fb
# ╟─8a8f8290-e71d-415e-b36f-bf509163a6a6
# ╠═ac92068b-5fc7-456c-9ab1-f7a6acbe0089
# ╠═d5b1580a-154d-43e7-9eb3-86ac2504e6b1
# ╠═2bf91a55-b327-422d-b331-630c898dcb64
# ╠═95d37731-401e-456f-97a3-1e965cbe8b9e
# ╠═a9e39487-19d7-49ad-8536-29f79f3a80d8
# ╟─3b4be6a5-e4bc-4a48-91ad-99705700b81f
# ╠═f715250f-291b-4fae-a40e-149f88a01bfe
# ╠═bcace027-418c-4d2e-beb2-cb40a5f16c22
# ╟─39ac140a-5e2b-41fe-93e1-3612b6dd0604
# ╠═5c5331ae-675a-4e07-a14f-fed84250829e
# ╟─99f42969-f9a0-4c02-8eaf-2ae395d55147
# ╠═f6141748-a3fd-4cc3-8296-6c311a8060cc
# ╟─4befb480-593c-4c29-adcf-3775cc3e736f
# ╠═376fe140-bd40-447a-992c-97b52ffc4c2b
# ╠═9be0a35f-bf46-4edd-be72-cd92a76822da
# ╟─fbf4401f-fb57-4d9c-a8a6-439ad19fd5bb
# ╟─c0e58f98-a52e-4742-a850-661faac4bbed
# ╟─56672e64-6834-4639-921b-0e87cede4d7a
# ╟─e37d1246-ccd6-481a-af2b-7d2d6acb8bbf
# ╠═12d724c0-a40b-4f7b-922e-9f8738bf01f4
# ╟─3ddf0432-99e5-4ce3-ac63-86f43b2d1a1c
# ╠═6e6b9d64-2d90-40a4-abde-2fd0d6ab7d7a
# ╟─d577b03d-bc68-4b32-9c6d-d92e0c4d7c99
# ╠═c4916313-d4f0-443c-a81e-05d2b765acf0
# ╠═a36695bb-599d-4f83-a747-9e3d0668e7d8
# ╠═d45813bd-8aa2-4454-bc12-ae8dce0a4590
# ╠═5047d396-af48-49fa-bf68-702fbe42c18e
# ╠═9bc2895e-ab70-49f2-be7c-61f19054cf50
# ╠═a780e90c-c6d1-44c8-9b55-d52cf4c20db4
# ╠═aeca907a-ee07-4045-b98f-0c67b1734008
# ╟─666a27b0-6fa3-4f5c-b85e-c3f66f0b5790
# ╟─16b45f02-8bdf-4721-ab70-ab2146c139bb
# ╟─3368ae68-3ce7-4426-a088-a24e5574abde
# ╟─68539414-2775-4f1e-bc03-016a7736c583
# ╟─2344c8c6-6cff-4047-bf43-5e081a3067ef
# ╟─bb099078-5995-48aa-b634-82c5c8f5f396
# ╟─1b9d8e13-7872-4bfe-9269-aee9a31d5ae6
# ╟─0babc5a1-c404-4ce8-bf30-74db15790c72
# ╟─6edbe471-361c-463a-b942-3ecbd879c427
# ╟─77566379-e31c-41a7-bb31-ccb3f2d69723
# ╟─358daffe-cf39-440d-a9b4-d0a3497b41ad
# ╟─c368eab7-9ce2-4b9a-bd91-3a8810472839
# ╟─f12ad623-59f9-4efe-8fb5-14b1bf6904bc
# ╟─e49849c5-d9b1-426b-b471-3acd32dcf07d
# ╟─b180997e-fa2b-44de-936f-eb42bef4b6ad
# ╟─f10a08f2-eba8-4e5e-87f8-313bb7494a49
# ╟─a4e11eb7-d314-4d78-b9b6-df8fc2838149
# ╠═b5f926d8-3ea2-47a8-9093-ea366afe374f
# ╟─1d1afe4f-8b7f-4d81-aa97-1448b47befac
# ╠═7f6c554b-3423-4bb5-bf07-853afa4e76fb
# ╟─bce9cdca-de1d-432c-a2f2-cbb6e414dfcb
# ╟─6b70f463-7d50-4b03-8210-b151575f98db
# ╠═05647633-14d2-4d5b-8c60-e236fbfeb334
# ╟─6c2fcfc8-158e-4165-9375-638d9444f70b
# ╟─bfbe7c40-3f60-49b4-9690-ad0ee0d7db99
# ╟─f3915dbd-6266-48cd-9bc0-a40b39b8dd22
# ╠═0ac1a38b-92bd-4000-8da6-295fcd1f028f
# ╠═5b69d857-d3c1-4a1a-9295-1691bbc73cf6
# ╟─4fd5b88f-eb4b-415b-9f8e-781dd4e194d0
# ╠═5f7635d8-42a3-4b74-b027-6a870d6e7d47
# ╟─6fd223aa-3d28-47e9-ba4f-391be5362521
# ╠═773c82f4-ba00-4907-953a-c1d7d6eb3478
# ╠═aee362e3-b1a0-4378-9af0-1ea1ed6580fe
# ╟─9a21ebe8-186f-4ab4-b0ae-8a0c668c3f92
# ╟─e523bc1f-f2ad-49a0-ae3e-aa79db6e8043
# ╠═9fe4b328-6487-48e0-9e28-5cdae655204e
# ╠═12068dea-798d-4cc3-86f0-07b7315caa91
# ╠═85bf8c44-348b-4825-b89a-33ec7614bb25
# ╟─cfe7ed5a-514a-4753-b029-9118813fa0ed
# ╠═344a94a8-58ad-4cb5-ad1c-dcf779a6ea76
# ╠═4ffc878c-d856-4183-96ea-ef77447b8a5c
# ╠═41bd4480-7672-4adf-924a-5bc1e9d4b45e
# ╠═a9443d53-1eae-4eab-b001-904be1523ca4
# ╠═4b932772-b69f-427a-b8f4-ee52fba4696a
# ╠═38e6ef4c-63c1-4df5-9451-f40df4fe57e7
# ╠═c25fa03c-0464-4e07-a777-9ee5f732b0a2
# ╠═5c87ea86-aec5-42d3-9423-4cd9d14dbc97
# ╠═d88a5c6b-8bcb-44d9-a52a-994e945aecad
# ╠═321474b9-364d-4b91-bf1c-601a9b8ad470
# ╠═9f85ad2d-f417-4463-9894-53f0eead4d83
# ╠═bf5d1782-5109-4cfd-8744-82ac80b5bc45
# ╠═a5ec3b99-af5a-42e5-8dff-533b45e50af5
# ╠═3d48d0e2-353c-44cf-a51b-1fad1b0002d2
# ╟─45e8699f-18ca-47a6-97eb-f855950b326d
# ╠═d992126b-9a1e-4806-b146-7c88c2ba17a2
# ╠═983931c6-c60c-4884-8818-0050acc544fc
# ╠═702e5559-55b0-4392-af55-846886aa1244
# ╠═c8bae838-0549-48e3-b858-0c071334c0b7
# ╠═9b35e3ae-95c4-4fe6-a84e-df4e22ab85e2
# ╠═edd27759-c2c5-4b5a-92b2-590f8673461a
# ╟─4c505f66-0c2c-4f59-858d-bd16c59f3397
# ╟─faba7178-bc20-4d93-87e3-26541851b1ad
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
