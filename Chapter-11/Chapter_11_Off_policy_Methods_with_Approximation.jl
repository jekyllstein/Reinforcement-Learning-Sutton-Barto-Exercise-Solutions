### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 31333ae3-615e-4587-80cf-d2716669af9e
using PlutoDevMacros

# ╔═╡ 702e5559-55b0-4392-af55-846886aa1244
PlutoDevMacros.@frompackage @raw_str(joinpath(@__DIR__, "..", "NonTabularRL.jl")) begin
	using NonTabularRL
	using >.Random, >.Statistics, >.LinearAlgebra, >.TailRec
end

# ╔═╡ 9b35e3ae-95c4-4fe6-a84e-df4e22ab85e2
begin
	using StatsBase, PlutoPlotly, PlutoUI, PlutoProfile, BenchmarkTools, LaTeXStrings, HypertextLiteral
	
	TableOfContents()
end

# ╔═╡ 46076214-2d52-4289-98e6-8b74c337f7d7
md"""
# Chapter 11: Off-policy Methods with Approximation
"""

# ╔═╡ a23b5ab9-8963-426d-9672-cf99a71d8884
md"""
## 11.1 Semi-gradient Methods
"""

# ╔═╡ 434045f4-865e-4993-913e-938b6cdf7a3f
md"""
> ### *Exercise 11.1* 
> Convert the equation of *n*-step off-policy TD (7.9) to the semi-gradient form.  Give accompanying definitions of the return for both the episodic and continuing cases.

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
> ### *Exercise 11.2* 
> Convert the equations of $n\text{-step} \; Q(\sigma)$ (7.11 and 7.17) to semi-gradient form.  Give definitions that cover both the episodic and continuing cases.

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
## 11.2 Examples of Off-policy Divergence
"""

# ╔═╡ 8d463e53-12ee-441c-bd14-e8b377fcdced
md"""
### Baird's Counter Example
"""

# ╔═╡ 29364905-2458-426a-999c-210cd3c60263
md"""
#### Baird Setup Functions
"""

# ╔═╡ c044414e-77d5-4a54-865e-dca4a879cd30
function make_baird_dynamics()
	states = 1:7
	actions = 1:2
	
	#dashed action takes system to one of six upper states with equal probability 
	dash = [s′ <= 6 ? 1/6 : 0.0 for s′ in states][:, [1]] #turn into matrix
	#solid action takes system to the 7th state
	solid = [s′ == 7 ? 1.0 : 0.0 for s′ in states][:, [1]]
	
	Dict((s, a) => a == 1 ? dash : solid for s in states for a in actions)
end

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

# ╔═╡ 77ca116d-675d-4db5-8a68-53d1085528f4
#step is a dictionary that maps state/action pairs to a matrix describing the distribution over possible subsequent state/reward pairs.  The value in each position is the probability of that transition occuring
struct Episodic_Full_Finite_MDP{S} <: MDP_Environment
	states::Vector{S}
	actions
	rewards::Vector{Float64}
	step::Dict{Tuple{S, Int64}, Matrix{Float64}}
	sterm::S
	γ::Float64
end

# ╔═╡ 8efa076f-d14d-44ab-bc03-e7ff964bc3b3
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
		#note that this uniformly samples over states which effectively is doing a behavior policy with a uniform distribution rather than using μ(s).  This is fine in the non-approximate case because each state is updated independently but convergence will be worse if state visits for the policy in question doesn't match uniform.
		for s in mdp.states)
		w .+= α .* δ ./ length(mdp.states)
		push!(w_history, copy(w))
		step!(nmax-1)
	end
	step!(maxsteps)
	return w_history
end

# ╔═╡ c3ad2cdc-6e85-48a7-a746-c7599f80a126
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

# ╔═╡ ad6c8986-8fb0-4682-ade8-ebb76b4c829a
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

# ╔═╡ fcef571c-9656-42e4-9a85-e13c3ed51edb
md"""
### Figure 11.2
$(figure11_2())
"""

# ╔═╡ 6965a4d3-5422-4a3e-8eba-fa101cb1b16d
md"""
### Example 11.1: Tsitsiklis and Van Roy's Counterexample
"""

# ╔═╡ a9264500-167f-4883-8514-d3fb962ef143
md"""
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

What if $\gamma > \frac{5}{6-4\epsilon}$?  In this case the factor multiplying $w_k$ on each update is greater than 1, thus the weight will diverge under any condition except where the initial value is 0.

In the first equation we didn't correctly account for the on policy distribution over states.  To calculate this we need to first get the expected value of state visits.  For simplicity assume that episodes always begin in state 1:

$\begin{flalign}
\eta(1) &= 1\\
\eta(2) &= 2 + \frac{1 - \epsilon}{\epsilon} = \frac{1+\epsilon}{\epsilon}\\
\sum_{s}\eta(s) &= 1 + \frac{1+\epsilon}{\epsilon} = \frac{1+2\epsilon}{\epsilon}\\
\mu(1) &= \frac{\epsilon}{1+2\epsilon}\\
\mu(2) &= \frac{1+\epsilon}{\epsilon} \frac{\epsilon}{1+2\epsilon} = \frac{1+\epsilon}{1+2\epsilon}
\end{flalign}$

Returning to the previous expression but including the on-policy distribution results in:

$\begin{flalign}
w_{k+1} &= \text{argmin}_{w \in \mathbb{R}} \enspace \sum_{s \in \mathcal{S}} \mu(s) \left ( \hat v(s, w) - \mathbb{E}_\pi[R_{t+1} + \gamma \hat v(S_{t+1}, w_k) | S_t = s] \right )^2\\ 
&= \text{argmin}_{w \in \mathbb{R}} \enspace \frac{\epsilon}{1+2\epsilon} (w - \gamma2w_k)^2 + \frac{1+\epsilon}{1+2\epsilon} (2w - (1-\epsilon)\gamma2w_k)^2\\
\therefore\\
\frac{\partial{w_{k+1}}}{\partial w} &= \frac{\epsilon}{1+2\epsilon} 2(w - \gamma2w_k) + \frac{1+\epsilon}{1+2\epsilon} 4(2w - (1-\epsilon)\gamma2w_k)\\
&\text{setting this equal to 0 and solving for w yields}\\
w &= 2 \gamma w_k \frac{\epsilon + 2 - 2\epsilon ^2}{5 \epsilon + 4}\\
&\text{therefore weight updates will diverge when}\\
\gamma &> 0.5 \frac{5 \epsilon + 4}{\epsilon + 2 - 2\epsilon ^2}\\
\end{flalign}$

Since γ never exceeds 1, this condition will never diverge for values of ϵ that result in a threshold that is greater than 1.  Under which conditions then will we always converge?

$\begin{flalign}
\frac{5 \epsilon + 4}{\epsilon + 2 - 2\epsilon ^2} &> 2\\
5 \epsilon + 4 &> 2\epsilon + 4 - 4\epsilon ^2\\
0 &> -3\epsilon - 4\epsilon^2\\
4\epsilon &> -3\\
\epsilon &> \frac{-3}{4}\\
\end{flalign}$

Since ϵ is always between 0 and 1 this condition will always hold.  This can be verified with a plot of the factor γ must exceed for divergence which ends up being greater than 1.
$(plot(scatter(x = collect(0.0:0.01:1.0), y = [0.5 * (5x + 4) / (x + 2 - 2x^2) for x in 0.0:0.01:1.0]), Layout(xaxis_title = "ϵ", yaxis_title = "γ threshold")))
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

# ╔═╡ 5960d4a9-5493-41d8-a98f-e9d91e34fa79
tsitsiklis_counterexample(0.001, 0.9, [0.])

# ╔═╡ 14fe90c3-50a7-4098-8626-b2d2a4b617ca
tsitsiklis_counterexample(0.01, 0.5, [1.])

# ╔═╡ e2751f9f-1554-4cb2-934e-0e032ad9a244
tsitsiklis_counterexample(0.01, 0.83, [1.])

# ╔═╡ e28a8728-bf1d-4a94-89f3-24d15d81425a
tsitsiklis_counterexample(0.01, 0.839, [1.])

# ╔═╡ fab9d8f8-8dbc-450e-8a40-7b83b5a236d0
tsitsiklis_counterexample(0.01, 0.99, [1.], maxsteps = 1000)

# ╔═╡ 4965afd6-b7b9-4fa9-ad1c-9744d5b9727d
md"""
> ### *Exercise 11.3 (programming)* 
> Apply one-step semi-gradient Q-learning to Baird's counterexample and show empirically that its weights diverge.
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
exercise_11_3(maxsteps = 100_000, ϵ = 0.1, α = 0.01)

# ╔═╡ 6a654e0e-2809-4e46-989f-815de38c8bf6
md"""
I applied one-step semi-gradient Q-learning to Baird's counterexample extending the feature vectors by 2 elements to represent the two actions.  After checking different intial weight vectors and ϵ values, both sarsa and q-learning seem to converge to show no preference for actions and value estimates of 0.  While the weights may diverge momentarily, after enough time steps it converges over a range of parameter values.  In the section describing the counter example it mentions that with the ϵ greedy behavior policy in Q-learning it has not been found to diverge, so I'm not sure why the weights would be expected to diverge here.
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

# ╔═╡ e4f1211b-d880-4a24-8a76-bb5018199791
md"""
## 11.4 Linear Value-function Geometry
## 11.5 Gradient Descent in the Bellman Error
## 11.6 The Bellman Error is Not Learnable
"""

# ╔═╡ e49849c5-d9b1-426b-b471-3acd32dcf07d
md"""
> ### *Exercise 11.4* 
> Prove (11.24). Hint: Write the $\overline{\text{RE}}$ as an expectation over possible states $s$ of the expectation of the squared error given that $S_t = s$.  Then add and subtract the true value of state $s$ from the error (before squaring), grouping the subtracted true value with the return and the added true value with the estimated value.  Then if you expand the square, the most complex term will end up being zero, leaving you with (11.24).

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

# ╔═╡ 45e8699f-18ca-47a6-97eb-f855950b326d
md"""
# Dependencies
"""

# ╔═╡ edd27759-c2c5-4b5a-92b2-590f8673461a
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
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
BenchmarkTools = "~1.5.0"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.3.1"
PlutoDevMacros = "~0.9.0"
PlutoPlotly = "~0.5.0"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.60"
StatsBase = "~0.34.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.5"
manifest_format = "2.0"
project_hash = "bae601692c90be741d826e3b2da651b6634caf1e"

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

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

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

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

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

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

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

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

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

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

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
# ╟─46076214-2d52-4289-98e6-8b74c337f7d7
# ╟─a23b5ab9-8963-426d-9672-cf99a71d8884
# ╟─434045f4-865e-4993-913e-938b6cdf7a3f
# ╟─2c668d98-453d-482b-8980-bfbccf82dd86
# ╟─e6e606c4-39d7-4b87-bd1a-b5799281f033
# ╟─8d463e53-12ee-441c-bd14-e8b377fcdced
# ╟─29364905-2458-426a-999c-210cd3c60263
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
# ╠═6965a4d3-5422-4a3e-8eba-fa101cb1b16d
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
# ╟─45e8699f-18ca-47a6-97eb-f855950b326d
# ╠═31333ae3-615e-4587-80cf-d2716669af9e
# ╠═702e5559-55b0-4392-af55-846886aa1244
# ╠═9b35e3ae-95c4-4fe6-a84e-df4e22ab85e2
# ╠═edd27759-c2c5-4b5a-92b2-590f8673461a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
