### A Pluto.jl notebook ###
# v0.19.16

using Markdown
using InteractiveUtils

# ╔═╡ 27ca1027-d60c-4045-99b5-b08d50254f22
begin
	using Pkg
	Pkg.activate(mktempdir())
	Pkg.add(url = "https://github.com/TakekazuKATO/TailRec.jl")
	pkglist = ["Random", "Statistics", "StatsBase", "PlutoPlotly", "PlutoUI"]
	map(Pkg.add, pkglist)
	using Random, Statistics, StatsBase, TailRec, PlutoUI, PlutoPlotly
end

# ╔═╡ a23b5ab9-8963-426d-9672-cf99a71d8884
md"""
# 11.1 Semi-gradient Methods
"""

# ╔═╡ 434045f4-865e-4993-913e-938b6cdf7a3f
md"""
> *Exercise 11.1* Convert the equation of *n*-step off-policy TD (7.9) to the semi-gradient form.  Give accompanying definitions of the return for both the episodic and continuing cases.

$\begin{flalign}
V_{t+n}(S_t) \dot = V_{t+n-1}(S_t)+\alpha\rho_{t:t+n-1}[G_{t:t+n}-V_{t+n-1}(S_t)], \hspace{1cm}  0 \leq t < T \tag{7.9}
\end{flalign}$
where $\rho_{t:t+n-1}$, called the *importance sampling ratio*, is the relative probability under the two policies of taking the *n* actions from $A_t$ to $A_{t+n-1}$

$\rho_{t:h} \dot = \prod_{k=t}^{\min(h, T-1)}\frac{\pi(A_k | S_k)}{b(A_k|S_k)} \tag{7.10}$

To convert this to a semi-gradient method we need to provide update equations for the weight vector that defines the value function approximation.

$\begin{flalign}
\mathbf{w}_{t+n} &\dot = \mathbf{w}_{t+n-1} + \alpha \rho_{t} \cdots \rho_{t+n-1} 
[G_{t:t+n} - \hat v(S_{t}, \mathbf{w}_{t+n-1})]\nabla \hat v (S_t, \mathbf{w}_{t+n-1})\\
G_{t:t+n} &\dot = R_{t+1} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n \hat v(S_{t+n}, \mathbf{w}_{t+n-1})	\tag{episodic}\\ 
G_{t:t+n} &\dot = R_{t+1} - \bar R_t + \cdots + R_{t+n} - \bar R_{t+n-1} + \hat v(S_{t+n}, \mathbf{w}_{t+n-1})\tag{continuing}
\end{flalign}$

The tablular value at a particular time step is replaced with the weight parameter at that time step with the gradient also being added next to the error term.
"""

# ╔═╡ 2c668d98-453d-482b-8980-bfbccf82dd86
md"""
> *Exercise 11.2* Convert the equations of $n\text{-step} \; Q(\sigma)$ (7.11 and 7.17) to semi-gradient form.  Give definitions that cover both the episodic and continuing cases.

$\begin{flalign}
Q_{t+n} & \dot = Q_{t+n-1}(S_t, A_t) + \alpha \rho_{t+1:t+n}[G_{t:t+n} - Q_{t+n-1}(S_t, A_t)] \tag{7.11}\\
G_{t:h} & \dot = R_{t+1} + \gamma \left ( \sigma_{t+1}\rho_{t+1} + (1 - \sigma_{t+1} \pi(A_{t+1}|S_{t+1}) \right ) \left ( G_{t+1:h} - Q_{h-1}(S_{t+1}, A_{t+1}) \right )\\ & + \gamma \bar V_{h-1}(S_{t+1}) \tag{7.17}\\
\bar V_t(s) & \dot = \sum_a \pi(a|s)Q_t(s, a)
\end{flalign}$

To convert this to a semi-gradient method we need to provide update equations for the weight vector that defines the value function approximation.

$\begin{flalign}
\mathbf{w}_{t+n} &\dot = \mathbf{w}_{t+n-1} + \alpha \rho_{t+1} \cdots \rho_{t+n} 
[G_{t:t+n} - \hat q(S_{t}, A_t, \mathbf{w}_{t+n-1})]\nabla \hat q (S_t, A_t, \mathbf{w}_{t+n-1})\\
G_{t:h} &\dot = R_{t+1} + \gamma \left ( \sigma_{t+1}\rho_{t+1} + (1 - \sigma_{t+1} \pi(A_{t+1}|S_{t+1}) \right ) \left ( G_{t+1:h} - \hat q(S_{t+1}, A_{t+1}, \mathbf{w}_{h-1}) \right )\\ & + \gamma \bar V_{h-1}(S_{t+1}), \text{\; for } t < h \leq T	\tag{episodic}\\ 
G_{t:h} &\dot = R_{t+1} - \bar R_{t+1} + \gamma \left ( \sigma_{t+1}\rho_{t+1} + (1 - \sigma_{t+1} \pi(A_{t+1}|S_{t+1}) \right ) \left ( G_{t+1:h} - \hat q(S_{t+1}, A_{t+1}, \mathbf{w}_{h-1}) \right )\\ & + \gamma \bar V_{h-1}(S_{t+1}), \text{\; for } t < h \leq T \tag{continuing}\\
\bar V_t(s) & \dot = \sum_a \pi(a|s)\hat q(s, a, \mathbf{w}_t)
\end{flalign}$

"""

# ╔═╡ e6e606c4-39d7-4b87-bd1a-b5799281f033
md"""
# 11.2 Examples of Off-policy Divergence
"""

# ╔═╡ 8d463e53-12ee-441c-bd14-e8b377fcdced
md"""
## Baird's Counter Example
"""

# ╔═╡ 29364905-2458-426a-999c-210cd3c60263
md"""
### Baird Setup Functions
"""

# ╔═╡ d2033a7d-3d9d-4983-8fd1-b4e6ee015080
function bairdtransition(s::Int64, a::Int64)
	if a == 1 #dashed action takes system to one of the six upper states with equal probability
		s′ = rand(1:6)
	elseif a == 2 #solid action takes the system to the 7th state
		s′ = 7
	end
	(s′, 0.0)
end

# ╔═╡ 1ba56556-2ac7-4d23-98c3-0d3fb54ec3d6
bairdbehavior(s::Int64) = [6/7, 1/7]

# ╔═╡ 1be8182a-c183-486c-9991-bcc325e75449
bairdπ(s::Int64) = [0.0, 1.0]

# ╔═╡ 2feb4657-3377-434f-bf8a-400cfcfe9fef
#run the baird example with a given policy for a set number of steps and keep track of visit statistics
@tailrec function runbaird(s0::Int64, π, nsteps::Int64, counts::Vector{Int64})
	counts[s0] += 1
	nsteps == 0 && return counts ./ sum(counts)
	a = sample(1:7, pweights(π(s0)))
	(s, r) = bairdtransition(s0, a)
	runbaird(s, π, nsteps-1, counts)
end

# ╔═╡ 3238aaa1-92aa-4d80-af22-4e237be9f0fc
function startbaird(π, nsteps)
	runbaird(1, π, nsteps, zeros(Int64, 7))
end

# ╔═╡ 1e010e8e-2dde-4228-b914-fdc120fa91ca
md"""
### Long-run State Distribution
"""

# ╔═╡ 4b7b7bb6-8484-42ac-983f-ec33dbf2c73e
#confirm that the distribution of visited states is uniform for the behavior policy
plot(bar(x = 1:7, y = startbaird(bairdbehavior, 1000000)), Layout(title = "Baird Behavior Policy State Distribution"))

# ╔═╡ 1074eb62-a5ee-43cb-a1a6-fe2bbc196f72
plot(bar(x = 1:7, y =startbaird(bairdπ, 1000000)), Layout(title = "Baird Target Policy State Distribution"))

# ╔═╡ cad27ba6-aa01-41e2-902b-ff411037cf0f
md"""
### Semi-gradient Estimation Functions
"""

# ╔═╡ 5d85cf97-3e46-4ace-8246-2fc73a93cc2f
abstract type MDP_Environment end

# ╔═╡ 12ff2e46-fa3e-4fe8-9a1f-58afc2a43c25
#the step function must be called as follows (s′, r) = step(s, a::Int64) where s and a are the starting state and selected action while the return values are the subsequent state and reward.  a is an integer which represents which action is taken from some list of actions
struct Episodic_MDP{S} <: MDP_Environment
	states::Vector{S}
	actions
	step::Function
	sterm::S
	γ::Float64
end

# ╔═╡ 84997a09-960f-4116-9045-74cb2e0e9d03
struct Continuing_MDP{S}
	states::Vector{S}
	actions
	step::Function
end

# ╔═╡ 8efa076f-d14d-44ab-bc03-e7ff964bc3b3
#On Policy Episodic Semi-gradient TD0 Value Estimation
#update weight vector that act as parameters for a value function estimate and its gradient.  Weight updates will occur to optimize value function according to the policy π.  The function will modify the initially provided weight vector but also keep track of the weight vector history for the purpose of tracking progress of the value function over time
function semi_gradient_TD0_v̂!(π::Function, mdp::Episodic_MDP, v̂::Function, ∇v̂::Function, w::Vector, maxsteps::Int64; α = 0.01)	
	s0 = rand(mdp.states)
	w_history = [copy(w)]
	@tailrec function step!(s, nmax)
		nmax == 0 && return nothing
		s == mdp.sterm && return step!(rand(mdp.states), nmax-1)
		a = sample(mdp.states, pweights(π(s)))
		(s′, r) = mdp.step(s, a)
		δ =  r .+ (mdp.γ .* v̂(s′, w)) .- v̂(s, w) 
		w .+= α .* δ .* ∇v̂(s, w)
		push!(w_history, copy(w))
		step!(s′, nmax-1)
	end
	step!(s0, maxsteps)
	return w_history
end

# ╔═╡ 0b146651-a99f-489b-92f5-b5bd74d275fe
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

# ╔═╡ 1853cb36-a97d-4922-92c2-02261843c761
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

# ╔═╡ d1cedda0-1ebf-42a6-b2f8-7df665252c08
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
		for s in mdp.states)
		w .+= α .* δ ./ length(mdp.states)
		push!(w_history, copy(w))
		step!(nmax-1)
	end
	step!(maxsteps)
	return w_history
end

# ╔═╡ ad6c8986-8fb0-4682-ade8-ebb76b4c829a
function figure11_2(;initializeweights = () -> [1., 1., 1., 1., 1., 1., 10., 1.])
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

	mdp = Episodic_MDP(collect(1:7), [1, 2], bairdtransition, 0, 0.99)

	w_history_onpolicy = semi_gradient_TD0_v̂!(bairdπ, mdp, v̂, ∇v̂, initializeweights(), 20000)
	w_history_offpolicy = semi_gradient_TD0_v̂!(bairdπ, bairdbehavior, mdp, v̂, ∇v̂, initializeweights(), epmax)
	w_history_DP = semi_gradient_DP_v̂!(bairdπ, mdp, v̂, ∇v̂, initializeweights(), epmax)

	function plot_weights(w_history, title; legend = true)
		l = length(w_history)
		traces = [scatter(x = 1:l, y = [w[i] for w in w_history], name = "w_$i") for i in 1:8]
		Plot(traces, Layout(showlegend=legend, title=title, legend_orientation="h"))
	end

	v_onpolicy = [v̂(s, w_history_onpolicy[end]) for s in mdp.states]
	p1 = plot_weights(w_history_onpolicy, "On Policy TD")
	p2 = plot_weights(w_history_offpolicy, "Off Policy TD")
	p3 = plot_weights(w_history_DP, "Semi-gradient DP")
	(v_onpolicy, p1, p2)
	plot([p1 p2; p3])
end

# ╔═╡ fcef571c-9656-42e4-9a85-e13c3ed51edb
md"""
### Figure 11.2
$(figure11_2())
"""

# ╔═╡ 6965a4d3-5422-4a3e-8eba-fa101cb1b16d
md"""
## Example 11.1: Tsitsiklis and Van Roy's Counterexample
"""

# ╔═╡ a9264500-167f-4883-8514-d3fb962ef143
md"""
The following weight updates are calculated to minimize the average estimation error for each transition weighted by the probability of experiencing that transition.
$\begin{flalign}
w_{k+1} &= \text{argmin}_{w \in \mathbb{R}} \enspace \sum_{s \in \mathcal{S}} \left ( \hat v(s, w) - \mathbb{E}_\pi[R_{t+1} + \gamma \hat v(S_{t+1}, w_k) | S_t = s] \right )^2\\ 
&= \text{argmin}_{w \in \mathbb{R}} \enspace (w - \gamma2w_k)^2 + (2w - (1-\epsilon)\gamma2w_k)^2\\
\therefore\\
\frac{\partial{w_{k+1}}}{\partial w} &= 2(w - \gamma2w_k) + 4(2w - (1-\epsilon)\gamma2w_k) = 10w - 4\gamma w_k - 8(1-\epsilon)\gamma w_k\\
&\text{setting this equal to 0 results in }\\
10w &= 4\gamma w_k + 8(1-\epsilon)\gamma w_k = 4 \gamma w_k (3 - 2\epsilon)\\
w &= \gamma w_k \frac{4(3 - 2\epsilon)}{10} = \gamma w_k \frac{6 - 4\epsilon}{5}
\end{flalign}$

What if $\gamma > \frac{5}{6-4\epsilon}$?  In this case the factor multiplying $w_k$ on each update is greater than 1, thus the weight will diverge under any condition except where the initial value is 0.
"""

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
function tsitsiklis_counterexample(ϵ, γ, w_0)
	thresh = 5 / (6 - 4*ϵ)
	if γ > thresh
		println("Weights for value function approxmation will diverge dynamic programming and direct minimization since γ > 5/(6-4ϵ)): $γ > $thresh")
		if w_0[1] == 0
			println("Since the weight is initialized at 0 it is already at the value for perfect approximation the updates will not diverge.  Any starting value other than this will have a problem though.")
		end
	else
		println("Weights for value function approxmation will NOT diverge under any method since γ < 5/(6-4ϵ)): $γ < $thresh")
	end
	
	function transition(s, a)
		if s == 1
			(2, 0.)
		elseif s == 2
			if rand() < ϵ
				(0, 0.)
			else
				(2, 0.)
			end
		end
	end
	
	maxsteps = 1000
	
	features = [[1.], [2.]] 
	
	#define value function estimator and its gradient with respect to parameters
	v̂(s, w) = s == 0 ? 0. : w' * features[s]
	∇v̂(s, w) = s == 0 ? 0. : features[s]

	mdp = Episodic_MDP(collect(1:2), [1], transition, 0, γ)

	#there is no meaningful action here
	π(s) = [1.]

	sg_input = (π, mdp, v̂, ∇v̂, w_0, maxsteps)

	w_history_onpolicy = semi_gradient_TD0_v̂!(sg_input...)
	w_history_DP = semi_gradient_DP_v̂!(sg_input...)

	function plot_weights(w_history, title; legend = true)
		l = length(w_history)
		traces = [scatter(x = 1:l, y = [w[i] for w in w_history], name = "w_$i") for i in 1:1]
		Plot(traces, Layout(showlegend=legend, title=title, legend_orientation="h"))
	end

	v_onpolicy = [v̂(s, w_history_onpolicy[end]) for s in mdp.states]
	p1 = plot_weights(w_history_onpolicy, "On Policy TD0")
	p2 = plot_weights(w_history_DP, "Semi-gradient DP")
	plot([p1 p2])
	# w_history_onpolicy
end

# ╔═╡ 5960d4a9-5493-41d8-a98f-e9d91e34fa79
tsitsiklis_counterexample(0.001, 0.9, [0.])

# ╔═╡ 14fe90c3-50a7-4098-8626-b2d2a4b617ca
tsitsiklis_counterexample(0.01, 0.5, [1.])

# ╔═╡ e2751f9f-1554-4cb2-934e-0e032ad9a244
tsitsiklis_counterexample(0.01, 0.83, [1.])

# ╔═╡ e28a8728-bf1d-4a94-89f3-24d15d81425a
tsitsiklis_counterexample(0.01, 0.85, [1.])

# ╔═╡ fab9d8f8-8dbc-450e-8a40-7b83b5a236d0
tsitsiklis_counterexample(0.01, 0.99, [1.])

# ╔═╡ 4965afd6-b7b9-4fa9-ad1c-9744d5b9727d
md"""
> *Exercise 11.3 (programming)* Apply one-step semi-gradient Q-learning to Baird's counterexample and show empirically that its weights diverge.
"""

# ╔═╡ 96e4f976-27ec-48c5-b902-b53af8a8d802
#One step Semi-gradient Sarsa using ϵ-greedy policy
function semi_gradient_sarsa(mdp::Episodic_MDP, q̂::Function, ∇q̂::Function, w::Vector, maxsteps::Int64; α = 0.01, ϵ = 0.01)	
	#q̂ should be a function that takes a feature vector representing a state/action pair and produces a value estimate as a single value
	s0 = rand(mdp.states)
	a0 = rand(mdp.actions)
	w_history = [copy(w)]
	greedyaction(s) = argmax(q̂(s, a, w) for a in mdp.actions)
	function ϵgreedy_action(s)
		if rand() < ϵ
			rand(mdp.actions)
		else
			greedyaction(s)
		end
	end
	
	@tailrec function step!(s, a, nmax)
		nmax == 0 && return nothing
		(s′, r) = mdp.step(s, a)
		if s′ == mdp.sterm 
			w .+= α .* (r .- q̂(s, a, w)) .* ∇q̂(s, a, w)
			return step!(rand(mdp.states), rand(mdp.actions), nmax-1)
		end
		a′ = ϵgreedy_action(s′)
		δ =  r .+ (mdp.γ .* q̂(s′, a′, w)) .- q̂(s, a, w) 
		w .+= α .* δ .* ∇q̂(s, a, w)
		push!(w_history, copy(w))
		step!(s′, a′, nmax-1)
	end
	step!(s0, a0, maxsteps)
	return w_history
end

# ╔═╡ 00e447c7-1ec8-4b51-80b9-784020bd5071
#One step Semi-gradient Q-learning using ϵ-greedy policy
function semi_gradient_qlearning(mdp::Episodic_MDP, q̂::Function, ∇q̂::Function, w::Vector, maxsteps::Int64; α = 0.01, ϵ = 0.01)	
	#q̂ should be a function that takes a feature vector representing a state/action pair and produces a value estimate as a single value
	s0 = rand(mdp.states)
	w_history = [copy(w)]
	#find the greedy action at state s based on the value estimate q̂
	greedyaction(s) = argmax(q̂(s, a, w) for a in mdp.actions)
	#find the value estimate for the action that produces the maximum value at state s
	maxq(s) = maximum(q̂(s, a, w) for a in mdp.actions)
	function ϵgreedy_action(s)
		if rand() < ϵ
			rand(mdp.actions)
		else
			greedyaction(s)
		end
	end
	
	@tailrec function step!(s, nmax)
		nmax == 0 && return nothing
		s == mdp.sterm && return step!(rand(mdp.states), nmax-1)
		a = ϵgreedy_action(s)
		(s′, r) = mdp.step(s, a)
		δ =  r .+ (mdp.γ .* maxq(s′)) .- q̂(s, a, w) 
		w .+= α .* δ .* ∇q̂(s, a, w)
		push!(w_history, copy(w))
		step!(s′, nmax-1)
	end
	step!(s0, maxsteps)
	return w_history
end

# ╔═╡ c537aeb0-963c-4cf9-88fd-cf94859b1964
function exercise_11_3(;initializeweights = () -> [1., 1., 1., 1., 1., 1., 10., 1., 1., 10.], maxsteps = 1000, γ = 0.99, ϵ = 0.01, α = 0.01)
	
	statefeatures = [
		[2, 0, 0, 0, 0, 0, 0, 1],
		[0, 2, 0, 0, 0, 0, 0, 1],
		[0, 0, 2, 0, 0, 0, 0, 1],
		[0, 0, 0, 2, 0, 0, 0, 1],
		[0, 0, 0, 0, 2, 0, 0, 1],
		[0, 0, 0, 0, 0, 2, 0, 1],
		[0, 0, 0, 0, 0, 0, 1, 2]
	]

	actionfeatures = [
		[2., 1.],
		[1., 2.]
	]

	#form state/action features by appending state feature to action feature
	x(s, a) = [statefeatures[s]; actionfeatures[a]]
	
	#define value function estimator and its gradient with respect to parameters
	q̂(s, a, w) = w' * x(s, a)
	∇q̂(s, a, w) = x(s, a)

	mdp = Episodic_MDP(collect(1:7), [1, 2], bairdtransition, 0, γ)

	w_history_sarsa = semi_gradient_sarsa(mdp, q̂, ∇q̂, initializeweights(), maxsteps, ϵ = ϵ, α = α)	
	w_history_qlearn = semi_gradient_qlearning(mdp, q̂, ∇q̂, initializeweights(), maxsteps, ϵ = ϵ, α = α)	

	qstar_sarsa = mapreduce(a -> [q̂(s, a, w_history_sarsa[end]) for s in mdp.states], hcat, mdp.actions)
	qstar_qlearn = mapreduce(a -> [q̂(s, a, w_history_qlearn[end]) for s in mdp.states], hcat, mdp.actions)
	
	function plot_weights(w_history, title; legend = true)
		l = length(w_history)
		traces = [scatter(x = 1:l, y = [w[i] for w in w_history], name = "w_$i") for i in 1:length(initializeweights())]
		Plot(traces, Layout(showlegend=legend, title=title, legend_orientation="h"))
	end

	p1 = plot_weights(w_history_sarsa, "One Step Sarsa")
	h1 = heatmap(x = mdp.actions, y = mdp.states, z = qstar_sarsa)
	p2 = plot_weights(w_history_qlearn, "One Step Q-Learning")
	h2 = heatmap(x = mdp.actions, y = mdp.states, z = qstar_qlearn)
	# p3 = plot_weights(w_history_DP, "Semi-gradient DP")
	# plot([p1 p2; p3])
	# plot(p1)
	md"""
	$(plot([p1 p2]))
	$(plot(h1, Layout(xaxis_title = "Action", yaxis_title="State", title = "Optimal Policy Action/Value Estimates Sarsa")))
	$(plot(h2, Layout(xaxis_title = "Action", yaxis_title="State", title = "Optimal Policy Action/Value Estimates Q-Learning")))
	"""
end

# ╔═╡ 1b68a25e-9f12-4894-a3a7-3fdd6df34316
exercise_11_3(maxsteps = 100_000, ϵ = 0.25, α = 0.01)

# ╔═╡ 6a654e0e-2809-4e46-989f-815de38c8bf6
md"""
I applied one-step semi-gradient Q-learning to Baird's counterexample extending the feature vectors by 2 elements to represent the two actions.  After checking different intial weight vectors and ϵ values, both sarsa and q-learning seem to converge to show no preference for actions and value estimates of 0.  While the weights may diverge momentarily, after enough time steps it converges over a range of parameter values.  In the section describing the counter example it mentions that with the ϵ greedy behavior policy in Q-learning it has not been found to diverge, so I'm not sure why the weights would be expected to diverge here.
"""

# ╔═╡ b62b78f5-4721-4fb6-b056-cc4dae9eae9f
md"""
# 11.3 The Deadly Triad
Instability and divergence arise when we combine the following three elements in solving an RL problem:

**Function approximation**
Necessary to scale up techniques to large problems where the state/action space is too large to store.

**Bootstrapping**
Important for data efficiency.  If we cannot use any bootstrapping we may need to wait and store results for very long episodes and sometimes they aren't even guaranteed to terminate.

**Off-policy training**
Often we could use Sarsa instead of Q-learning to remedy this, so avoiding off-policy training might be the best way to guarantee stability for now.  However there will be cases in the future where off-policy training might be necessary such as estimating multiple policies at once.
"""

# ╔═╡ e4f1211b-d880-4a24-8a76-bb5018199791
md"""
# 11.4 Linear Value-function Geometry
# 11.5 Gradient Descent in the Bellman Error
# 11.6 The Bellman Error is Not Learnable
"""

# ╔═╡ e49849c5-d9b1-426b-b471-3acd32dcf07d
md"""
> *Exercise 11.4* Prove (11.24). Hint: Write the $\overline{\text{RE}}$ as an expectation over possible states $s$ of the expectation of the squared error given that $S_t = s$.  Then add and subtract the true value of state $s$ from the error (before squaring), grouping the subtracted true value with the return and the added true value with the estimated value.  Then if you expand the square, the most complex term will end up being zero, leaving you with (11.24).

To start out we have the definition of the *mean square return error*

$\overline{\text{RE}}(\mathbf{w}) = \mathbb{E} \left [ (G_t - \hat v(S_t, \mathbf{w}))^2 \right ]$

Also we can note from Chapter 3 that $v_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$ and from Chapter 9 that $\overline{\text{VE}}(\mathbf{w}) \dot = \sum_{s \in \mathcal{S}} \mu(s) \left [ v_\pi(s) - \hat v(s, \mathbf{w}) \right ]^2$.

Rewriting expectation results in:

$\begin{flalign}
\overline{\text{RE}}(\mathbf{w}) &= \sum_s \mu_\pi(s) \mathbb{E_\pi}\left [ (G_t - \hat v(S_t, \mathbf{w}))^2 | S_t = s \right ]\\
&= \sum_s \mu_\pi(s) \mathbb{E_\pi}\left [ (G_t - \hat v(S_t, \mathbf{w}) + v_\pi(S_t) - v_\pi(S_t))^2 | S_t = s \right ]\\
&= \sum_s \mu_\pi(s) \mathbb{E_\pi}\left [ ((G_t - v_\pi(S_t)) + (v_\pi(S_t) - \hat v(S_t, \mathbf{w})))^2 | S_t = s\right ]\\
&= \sum_s \mu_\pi(s) \mathbb{E_\pi} \left [ (G_t - v_\pi(S_t))^2 + (v_\pi(S_t) - \hat v(S_t, \mathbf{w}))^2 + 2((G_t - v_\pi(S_t))(v_\pi(S_t) - \hat v(S_t, \mathbf{w}))) | S_t = s \right ]\\
&= \mathbb{E}\left [ (G_t - v_\pi(S_t))^2 \right ] + \sum_s  \mu_\pi(s) \left [v_\pi(s) - \hat v(s, \mathbf{w}) \right ]^2 +\\ 
&\sum_s 2\mu_\pi(s) \left [ v_\pi(s) \mathbb{E_\pi}[G_t | S_t = s] -  \hat v(s, \mathbf{w}) \mathbb{E_\pi}[G_t | S_t = s] - v_\pi(s)^2 + v_\pi(s) \hat v(s, \mathbf{w}) \right]\\
&= \mathbb{E}\left [ ((G_t - v(S_t))^2 \right ] + \overline{\text{VE}} + \sum_s 2\mu_\pi(s) \left [ v_\pi(s)^2 -  \hat v(s, \mathbf{w}) v_\pi(s) - v_\pi(s)^2 + v_\pi(s) \hat v(s, \mathbf{w}) \right]\\
&= \mathbb{E}\left [ ((G_t - v(S_t))^2 \right ] + \overline{\text{VE}} + \sum_s 2\mu_\pi(s) \times 0\\
&\therefore\\
\overline{\text{RE}}(\mathbf{w}) &= \mathbb{E}\left [ ((G_t - v(S_t))^2 \right ] + \overline{\text{VE}}
\end{flalign}$
"""

# ╔═╡ 16dd27c3-3cbb-4654-8931-db0defed4c29
md"""
# Table of Conents Settings
"""

# ╔═╡ 75a492fc-5441-4fb6-a752-2719bd5c929b
TableOfContents()

# ╔═╡ Cell order:
# ╟─a23b5ab9-8963-426d-9672-cf99a71d8884
# ╟─434045f4-865e-4993-913e-938b6cdf7a3f
# ╟─2c668d98-453d-482b-8980-bfbccf82dd86
# ╟─e6e606c4-39d7-4b87-bd1a-b5799281f033
# ╟─8d463e53-12ee-441c-bd14-e8b377fcdced
# ╠═27ca1027-d60c-4045-99b5-b08d50254f22
# ╟─29364905-2458-426a-999c-210cd3c60263
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
# ╠═8efa076f-d14d-44ab-bc03-e7ff964bc3b3
# ╠═0b146651-a99f-489b-92f5-b5bd74d275fe
# ╠═1853cb36-a97d-4922-92c2-02261843c761
# ╠═d1cedda0-1ebf-42a6-b2f8-7df665252c08
# ╠═ad6c8986-8fb0-4682-ade8-ebb76b4c829a
# ╟─fcef571c-9656-42e4-9a85-e13c3ed51edb
# ╟─6965a4d3-5422-4a3e-8eba-fa101cb1b16d
# ╟─a9264500-167f-4883-8514-d3fb962ef143
# ╟─3dade251-ddf7-463e-8d55-1c37e6d8ac9a
# ╠═3280e9dc-e0e4-4a18-88a5-0a4ac188e71c
# ╠═5960d4a9-5493-41d8-a98f-e9d91e34fa79
# ╠═14fe90c3-50a7-4098-8626-b2d2a4b617ca
# ╠═e2751f9f-1554-4cb2-934e-0e032ad9a244
# ╠═e28a8728-bf1d-4a94-89f3-24d15d81425a
# ╠═fab9d8f8-8dbc-450e-8a40-7b83b5a236d0
# ╟─4965afd6-b7b9-4fa9-ad1c-9744d5b9727d
# ╠═96e4f976-27ec-48c5-b902-b53af8a8d802
# ╠═00e447c7-1ec8-4b51-80b9-784020bd5071
# ╠═c537aeb0-963c-4cf9-88fd-cf94859b1964
# ╠═1b68a25e-9f12-4894-a3a7-3fdd6df34316
# ╟─6a654e0e-2809-4e46-989f-815de38c8bf6
# ╟─b62b78f5-4721-4fb6-b056-cc4dae9eae9f
# ╟─e4f1211b-d880-4a24-8a76-bb5018199791
# ╟─e49849c5-d9b1-426b-b471-3acd32dcf07d
# ╟─16dd27c3-3cbb-4654-8931-db0defed4c29
# ╠═75a492fc-5441-4fb6-a752-2719bd5c929b
