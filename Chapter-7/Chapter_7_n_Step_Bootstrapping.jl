### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 0321b9d1-7d4e-4bf8-ac61-9c16ab6bc461
begin
	using Statistics, PlutoPlotly, Random, StatsBase, PlutoUI
	TableOfContents()
end

# ╔═╡ b8e46d42-44f1-4f34-a501-7a4943e60a83
md"""
# Chapter 7 n-step Bootstrapping

In this chapter we unify the Monte Carlo (MC) methods and the one-step temporal difference (TD) methods presented in the previous two chapters.  Neither MC methods nor one-step TD methods are always the best.  In this chapter we present *n-step TD methods* that generalize both methods so that one can shift from one to the other smoothly as needed to meet the demands of a particular task.  *n*-step methods span a spectrum with MC methods at one end and one-step TD methods at the other.  The best methods are often intermediate between the two extremes.

Another way of looking at the benefits of *n*-step methods is that they free you from the tyranny of the time step.  With one-step TD methods the same time step determines how often the action can be changed and the time interval over which bootstrapping is done.  In many applications one wants to be able to update the action very fast to take into account anything that has changed, but bootstrapping works best if it is over a length of time in which a significant and recognizable state change has occured (e.g. in a game where the reward is only assigned at the end, bootstrapping may be less effective).  With one-step TD methods, these time intervals are the same, and so a compromise must be made.  *n*-step methods enable bootstrapping to occur over multiple steps, freeing us from the tyranny of the single time step.

As usual, we first consider the prediction problem and then the control problem.  That is, we first consider how *n*-step methods can help in predicting returns as a function of state for a fixed policy (i.e. estimating $v_\pi$).  Then we extend the ideas to action values and control methods.
"""

# ╔═╡ 4e4ee05c-c585-4847-b2ab-0e5c4c7f6ca4
md"""
## 7.1 n-step TD Prediction

Consider estimating $v_\pi$ from sample episodes generated using $\pi$.  Monte Carlo methods perform an update for each state based on the entire sequence of observed rewards from that state until the end of the episode.  The update of one-step TD methods, on the other hand, is based on just the one next reward, bootstrapping from the value of the state one step later as a proxy for the remaining rewards.  One kind of intermediate method, then, would perform an update based on an intermediate number of rewards: more than one, but less than all of them until termination.  For example, a two-step method update would be based on the first two rewards and the estimated value of the state two steps later.  Similarly, we could have three-step updates, four-step updates, and so on.  Figure 7.1 shows the backup diagrams of the spectrum of *n-step updates* for $v_\pi$, with the one-step TD update on the left and the up-until-termination Monte Carlo update on the right.
"""

# ╔═╡ 28e10379-a475-4b29-bfc6-f9c52201358a
HTML("""
<div class = "backup-diagram">
	1-step TD <br> and TD(0)
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
</div>
""")

# ╔═╡ 88e27f19-2cfb-4e17-8a62-5eadafbda85e
HTML("""
<style>
.backup-diagram {
	display: flex;
	flex-direction: column;
	align-items: center;
	justify-content: center;
	background-color: white;
	color: black;
}

.down-arrow {
	display: flex;
	flex-direction: column;
	align-items: center;
	justify-content: flex-end;
	width: 2px;
	height: 35px;
	background-color: black;
	padding-bottom: 0px;
	margin-bottom: 1px;
}

.down-arrow::before {
	content: '';
	width: 0;
	height: 0;
	border-left: 4px solid transparent;
	border-right: 4px solid transparent;
	border-top: 8px solid black;
	transform: translateY(1px);
}

.state {
	width: 30px;
	height: 30px;
	border: 2px solid black;
	background-color: white;
	border-radius: 50%;
}
.action {
	width: 20px;
	height: 20px;
	background-color: black;
	border-radius: 50%;
}
</style>
""")

# ╔═╡ 4f7135bf-911d-4d57-a845-e6f86ff7f9a4
md"""

$V_{t+n}(S_t) \dot= V_{t+n-1}(S_t) + \alpha [G_{t:t+n} - V_{t+n-1}(S_t)], 0\leq t <T$
"""

# ╔═╡ 28d9394a-6ce0-4212-941e-2a699e08b7df
md"""
> Exercise 7.1 In Chapter 6 we noted that Monte Carlo error can be written as the sum of TD errors (6.6) if the value estimates don't change from step to step.  Show that the n-step error used in (7.2) can also be written as a sum of TD errors (again if the value estimates don't change) generalizing the earlier result

The n-step update used in (7.2) can be written as:

$V_{t+n}(S_t) \dot= V_{t+n-1}(S_t) + \alpha [G_{t:t+n} - V_{t+n-1}(S_t)], 0\leq t <T$

while the TD errors  are defined as:

$\delta_t \dot = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$

Per the conventions used in the earlier derivation, the n-step error is the term being multipled by α:

$G_{t:t+n} - V(S_t) = \sum_{i=1}^n \gamma^{i-1}R_{t+i} + \gamma^n V(S_{t+n}) - V(S_t)$

$=R_{t+1} + \gamma V(S_{t+1}) - V(S_t) + \sum_{i=2}^n \gamma^{i-1}R_{t+i} - \gamma V(S_{t+1}) + \gamma^n V(S_{t+n})$ 

$=\delta_t + \gamma R_{t+2} + \sum_{i=3}^n \gamma^{i-1}R_{t+i} - \gamma V(S_{t+1}) + \gamma^n V(S_{t+n})$ 

$=\delta_t + \gamma \left [ R_{t+2} - V(S_{t+1}) + \gamma V(S_{t+2}) \right ]  + \sum_{i=3}^n \gamma^{i-1}R_{t+i} + \gamma^n V(S_{t+n}) - \gamma^2 V(S_{t+2})$ 

$=\delta_t + \gamma \delta_{t+1} + \sum_{i=3}^n \gamma^{i-1}R_{t+i} + \gamma^n V(S_{t+n}) - \gamma^2 V(S_{t+2})$

Extending this procedure out to i = n yields:

$=\sum_{i=0}^{n-2} \gamma^i \delta_{t+i} - \gamma^{n-1}V(S_{t+n-1}) + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$

$=\sum_{i=0}^{n-2} \gamma^i \delta_{t+i} + \gamma^{n-1} \left [\gamma V(S_{t+n}) - V(S_{t+n-1}) + R_{t+n} \right ]$

$=\sum_{i=0}^{n-2} \gamma^i \delta_{t+i} + \gamma^{n-1} \delta_{t+n-1} = \sum_{i=0}^{n-1} \gamma^i \delta_{t+i}$
"""

# ╔═╡ 5954a783-14a6-4d2b-b342-9fee0d9ef564
md"""
> Exercise 7.2 (programming) With an n-step method, the value estimates *do* change from step to step, so an algorithm that used the sum of TD errors (see previous exercise) in place of the error in (7.2) would actually be a slightly different algorithm.  Would it be a better algorithm or a worse one?  Devise and program a small experiment to answer this question empirically.

The pseudo-code for the typical algorithm is given already.  The algorithm to compare this to is one that calculates all the updated values for an episode using the n-step method but does not update the values until the end of the episode.  This can be tested by having two versions of V so that the values from the end of the previous episode can be preserved all while the new values are being updated.
"""

# ╔═╡ 6445e66b-1c07-4474-81ff-5c4cbba88ca6
#based on pseudocode described in book for n-step TD value estimation
function n_step_TD_Vest(π, α, n, states, sterm, sim, γ; v0 = 0.0, numep = 1000, Vtrue = Dict(s => v0 for s in states))
	V = Dict(s => v0 for s in states)
	V[sterm] = 0.0
	Svec = Vector{eltype(states)}(undef, n+1)
	Rvec = Vector{Float64}(undef, n+1)
	rmserr() = sqrt(mean((V[s] - Vtrue[s])^2 for s in states))
	rmserrs = Vector{Float64}(undef, numep)
	for ep in 1:numep
		#for each episode save a record of states and rewards
		s0 = rand(states)
		Svec[1] = s0
		s = s0
		T = typemax(Int64)
		τ = 0
		t = 0
		while τ != T - 1
			if t < T
				a = π(s)
				(s, r) = sim(Svec[mod(t, n+1)+1], a)
				storeind = mod(t+1, n+1) + 1
				Svec[storeind] = s
				Rvec[storeind] = r
				(s == sterm) && (T = t + 1)
			end
			τ = t - n + 1

			if τ >= 0
				G = sum(γ^(i - τ - 1) * Rvec[mod(i, n+1)+1] for i in (τ + 1):min(τ+n, T))
				if τ+n < T
					G += γ^n * V[Svec[mod(τ+n, n+1)+1]]
				end
				if τ == 0
					V[s0] += α*(G - V[s0])
				else
					V[Svec[mod(τ, n+1)+1]] += α*(G - V[Svec[mod(τ, n+1)+1]])
				end
			end
			t += 1
		end
		rmserrs[ep] = rmserr()
	end
	return V, rmserrs
end

# ╔═╡ a0092f58-8f2d-4d98-895a-5cc622fb8f4f
#modified from pseudocode to not used updated value estimates until episode is completed
function n_step_TD_Vest_static(π, α, n, states, sterm, sim, γ; v0 = 0.0, numep = 1000, Vtrue = Dict(s => v0 for s in states))
	Vold = Dict(s => v0 for s in states)
	Vnew = Dict(s => v0 for s in states)
	Vold[sterm] = 0.0
	Vnew[sterm] = 0.0
	Svec = Vector{eltype(states)}(undef, n+1)
	Rvec = Vector{Float64}(undef, n+1)
	rmserr() = sqrt(mean((Vnew[s] - Vtrue[s])^2 for s in states))
	rmserrs = Vector{Float64}(undef, numep)
	for ep in 1:numep
		#for each episode save a record of states and rewards
		s0 = rand(states)
		Svec[1] = s0
		s = s0
		T = typemax(Int64)
		τ = 0
		t = 0
		while τ != T - 1
			if t < T
				a = π(s)
				(s, r) = sim(Svec[mod(t, n+1)+1], a)
				storeind = mod(t+1, n+1) + 1
				Svec[storeind] = s
				Rvec[storeind] = r
				(s == sterm) && (T = t + 1)
			end
			τ = t - n + 1

			if τ >= 0
				G = sum(γ^(i - τ - 1) * Rvec[mod(i, n+1)+1] for i in (τ + 1):min(τ+n, T))
				if τ+n < T
					G += γ^n * Vold[Svec[mod(τ+n, n+1)+1]]
				end
				if τ == 0
					Vnew[s0] += α*(G - Vold[s0])
				else
					Vnew[Svec[mod(τ, n+1)+1]] += α*(G - Vold[Svec[mod(τ, n+1)+1]])
				end
			end
			t += 1
		end
		rmserrs[ep] = rmserr()
		#at the end of the episode set Vold to equal Vnew
		for s in states
			Vold[s] = Vnew[s]
		end
	end
	return Vnew, rmserrs
end

# ╔═╡ 7f32af81-580b-4f7b-8a71-0aa67928afa4
begin
	abstract type LinearMoves end
	struct Left <: LinearMoves end
	struct Right <: LinearMoves end
end

# ╔═╡ 1e199d68-4bc1-4f56-bbce-f4299e649dee
#create a random walk mdp of length n where the left terminal state produces a reward of -1 and the right a reward of 1
function create_random_walk(n)
	states = Tuple(1:n)
	sterm = 0
	actions = (:left, :right)
	move(s, ::Left) = s - 1
	move(s, ::Right) = s + 1
	function step(s0, action)
		s = move(s0, action)
		(s == 0) && return (sterm, -1.0)
		(s > n) && return (sterm, 1.0)
		return (s, 0.0)
	end
	(states, sterm, step)
end

# ╔═╡ 7f6dee5f-1cc5-4dab-86c2-a68c92bc135a


# ╔═╡ d3d39d13-e711-4289-a893-28c2c1af50f6
function value_estimate_random_walk(nstates, α, n, numep = 1000)
	#estimate random policy
	π(s) = rand([Left(), Right()])
	(states, sterm, step) = create_random_walk(nstates)
	Vest, rmserrs = n_step_TD_Vest(π, α, n, states, sterm, step, 1.0, numep = numep)
	c = (nstates + 1)/2
	Vtrue = [(s-c)/c for s in 1:nstates]
	t1 = scatter(x = states, y = [Vest[s] for s in states], name = "Value Estimate")
	t2 = scatter(x=states, y = Vtrue, name = "True Value")
	plot([t1, t2], Layout(xaxis_title = "State", title = "$nstates State Random Walk State Values"))
end

# ╔═╡ 3b1ee21f-6147-40bb-a534-26dd255ce26e
value_estimate_random_walk(5, 0.001, 1, 100_000)

# ╔═╡ e6ad9fb0-9efe-4a38-8160-43f1b9c7ee40
function nsteptd_error_random_walk(nstates, estimator; v0=0.0)
	#estimate random policy
	π(s) = rand([Left(), Right()])

	c = (nstates + 1)/2
	Vtrue = [(s-c)/c for s in 1:nstates]

	maxerr = sqrt(mean(Vtrue .^2))

	(states, sterm, step) = create_random_walk(nstates)
	
	function get_nstep_error(α, n)
		Vest, rmserrs = estimator(π, α, n, states, sterm, step, 1.0, numep = 10, v0=v0, Vtrue = Vtrue)
		mean(rmserrs)
	end

	α_vec = 0.0:0.1:1.0
	n_vec = [2 .^ (0:8); 1_000]
	rmsvecs = [[mean(get_nstep_error(α, n) for _ in 1:100) for α in 0.0:0.1:1.0] for n in n_vec]
	(rmsvecs, α_vec, n_vec, ymax = maxerr)
end

# ╔═╡ 63e005e5-ca51-4c18-8dcf-50db1ec831c8
function optimize_n_randomwalk(nstates; estimator = n_step_TD_Vest, v0=0.0)
	(rmsvecs, α_vec, n_vec, ymax) = nsteptd_error_random_walk(nstates, estimator, v0=v0)
	traces = [scatter(x = α_vec, y = rmsvecs[i], name = "n=$(n_vec[i])") for i in eachindex(rmsvecs)]
	ymin = minimum(minimum(v) for v in rmsvecs) * 0.9
	plot(traces, Layout(title="RMS Error for $nstates State Chain with Random Policy", xaxis_title = "α", yaxis_range = [ymin, ymax]))
end

# ╔═╡ e79f15d7-c31c-4fc3-917d-8f7c7bf88d59
md"""
These results match the figure in the book.  Using this same procedure but with the modified estimator that keeps the value estimates static during episode, we can compare the performance over a variety of n and α values.  Then we can observe whether allowing V to update during an episode leads to better or worse results on this particular random walk task.
"""

# ╔═╡ b736204d-bb74-479f-abc2-b59af511227c
walk19_plot1 = optimize_n_randomwalk(19)

# ╔═╡ da4a8d1b-83e6-416b-bdb1-49a3ea120cbe
walk5_plot1 = optimize_n_randomwalk(5, v0=0.0)

# ╔═╡ 14d8454c-76ef-4922-9bcd-d62c91ae7518
md"""
Using the modified value estimator we can perform the same plots as above for 19 and 5 state random walks
"""

# ╔═╡ 8e6fd9da-c8d2-4d29-bae2-d82315103693
begin
	walk19_plot2 = optimize_n_randomwalk(19, estimator = n_step_TD_Vest_static)
	[walk19_plot1, walk19_plot2]
end

# ╔═╡ 6376fefa-480f-4e3f-93ee-5425388a4f27
begin
	walk5_plot2 = optimize_n_randomwalk(5, estimator = n_step_TD_Vest_static)
	[walk5_plot1, walk5_plot2]
end

# ╔═╡ 81e53593-3a5c-4949-aa8d-9d92abe2ccb1
md"""
For each of these random walk sizes, the modified estimator performs worse overall and is less robust to changes in α.  The lowest error after 10 episodes for any value of α is better for the original algorithm. 
	"""

# ╔═╡ 0e318ccf-f4ad-4715-8415-125539e02690
md"""
> *Exercise 7.3* Why do you think a larger random walk task (19 states instead of 5) was used in the examples of this chapter?  Would a smaller walk have shifted the advantage to a different value of n?  How about the change in hte left-side outcome from 0 to -1 made in the larger walk?  Do you think that made any difference in the best value of n?

For the shorter chain, an n of 1 seems optimal over the first 10 episodes whereas the longer chain estimate performs best at n = 4.  Because each state is much closer to a terminal state in the shorter chain, there is less need to use estimates from states further away from the current state.  The information from the terminal state does not have to diffuse very far to reach any given state compared to the 19 state chain.  Changing the left side outcome to -1 means that the initial value of 0 is now the true value for the central state.  Keeping the true mean value equal to the initial value will affect performance a lot.  If there is a mismatch in this value then it might shift the optimal value of n.
"""

# ╔═╡ e10b6467-05f4-40b8-b899-2a31b5067030
md"""
## 7.2 *n*-step Sarsa

We redefine *n*-step returns (update targets) in terms of estimated action values:

$G_{t:t+n} \dot = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n Q_{t+n-1}(S_{t+n}, A_{t+n}), n \geq 1, 0 \leq t \lt T-n$ with $G_{t:t+n} \dot = G_t$ if $t+n \geq T$

The natural algorithm is then

$Q_{t+n}(S_t, A_t) \dot = Q_{t+n-1}(S_t, A_t) + \alpha [G_{t:t+n} - Q_{t+n-1}(S_t, A_t)], 0 \leq t \lt T$

while the values of all others states remain unchanged.  This is the algorithm we call *n-step Sarsa*.
"""

# ╔═╡ f4d7917d-f773-46d4-8605-87195c293d11
#based on pseudocode described in book for n-step Sarsa for estimating Q
function n_step_sarsa(ϵ, α, n, states::Vector{S}, sterm, actions::Vector{A}, sim, γ; q0 = 0.0, numep = 1000) where S where A
	#mapping of actions to indicies in action list
	actiondict = Dict(actions[i] => i for i in eachindex(actions))
	numactions = length(actions)

	#initialize action values as a vector of values for each state
	Q = Dict(s => fill(q0, numactions) for s in states)
	Q[sterm] = fill(0.0, numactions)

	#initialize policy to be random at each state
	π = Dict(s => ones(numactions) ./ numactions for s in states)

	#define a function to select actions from a policy
	sample_action(s) = sample(actions, weights(π[s])) 

	#with a probability ϵ a random action will be selected
	baseval = ϵ / n

	#with a probability 1-ϵ the greedy action will be selected
	bonusval = 1.0 - ϵ
	
	#define a function to update π to be ϵ-greedy wrt Q
	function update_π!()
		for s in states
			qvec = Q[s]
			i = argmax(qvec)
			π[s] .= baseval
			π[s][i] += bonusval
		end
	end

	#initialize vectors to store a history up to length n+1 of states, actions, and rewards
	svec = Vector{S}(undef, n+1)
	avec = Vector{A}(undef, n+1)
	rvec = Vector{Float64}(undef, n+1)

	#define index calculator
	getind(i) = mod(i, n+1) + 1
	
	for ep in 1:numep
		#initialize state
		s0 = rand(states)
		#initialize action
		a0 = sample_action(s0)
		
		svec[1] = s0
		avec[1] = a0
		s = s0
		a = a0
		T = typemax(Int64)
		τ = 0
		t = 0
		while τ != T - 1
			if t < T
				(s, r) = sim(svec[getind(t)], a)
				storeind = getind(t+1)
				svec[storeind] = s
				rvec[storeind] = r
				if s == sterm 
					T = t + 1
				else
					a = sample_action(s)
					avec[storeind] = a
				end
			end
			τ = t - n + 1
			if τ >= 0
				G = sum(γ^(i - τ - 1) * rvec[getind(i)] for i in (τ + 1):min(τ+n, T))
				if τ+n < T
					G += γ^n * Q[svec[getind(τ+n)]][actiondict[avec[getind(τ+n)]]]
				end
				ind = getind(τ)
				Q[svec[ind]][actiondict[avec[ind]]] += α*(G-Q[svec[ind]][actiondict[avec[ind]]])
				update_π!()
			end
			t += 1
		end
	end

	greedy_π = Dict(s => actions[argmax(π[s])] for s in states)
	
	return Q, greedy_π
end

# ╔═╡ 9d12661a-e830-4205-8504-a1da35a84521
begin
	abstract type GridworldAction end
	struct North <: GridworldAction end
	struct South <: GridworldAction end
	struct East <: GridworldAction end
	struct West <: GridworldAction end
end

# ╔═╡ 703e9324-e4cf-494a-9f3f-8bd258938bd7
function make_gridworld(n, m, goal, reward, γ)
	states = [(x, y) for x in 1:n for y in 1:m]
	actions = [North(), South(), East(), West()]
	
	move((x, y), ::West) = (max(1, x-1), y)
	move((x, y), ::East) = (min(n, x+1), y)
	move((x, y), ::North) = (x, min(m, y+1))
	move((x, y), ::South) = (x, max(1, y-1))

	function step(s0, a::GridworldAction)
		s = move(s0, a)
		(s == goal) && return (s, reward)
		(s, 0.0)
	end

	return (states, goal, actions, step)
end

# ╔═╡ 0c64afd6-ef23-4582-9250-2c1d4ae3cc43
function test_n_step_sarsa(n)
	(states, sterm, actions, sim) = make_gridworld(10, 8, (7, 4), 1.0, 1.0)
	(Q, π) = n_step_sarsa(0.1, 0.1, n, states, sterm, actions, sim, 1.0, numep = 1000)
	s = (1, 1)
	while s != sterm
		println(s, π[s])
		(s, r) = sim(s, π[s])
	end
end

# ╔═╡ 8ac0f5e2-d2a2-4bc7-a4a0-910068947f90
test_n_step_sarsa(10)

# ╔═╡ 1b12b915-4576-4c3d-8360-50eb9ad2392d
md"""
> *Exercise 7.4* Prove that the *n*-step return of Sarsa (7.4) can be written exactly in terms of a novel TD error, as 
>$G_{t:t+n}=Q_{t-1}(S_t, A_t) + \sum_{k=t}^{\text{min}(t+n,T)-1} \gamma^{k-t}[R_{k+1}+\gamma Q_k (S_{k+1}, A_{k+1}) - Q_{k-1}(S_k, A_k)]$

*n*-step return for Sarsa is:

$G_{t:t+n} \dot = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n Q_{t+n-1}(S_{t+n}, A_{t+n}), n \geq 1, 0 \leq t \lt T-n$

So we can see that:

$G_{k:k+1} \dot = R_{k+1} + \gamma Q_k(S_{k+1}, A_{k+1})$

which we can use to rewrite the novel expression as:

$G_{t:t+n}=Q_{t-1}(S_t, A_t) + \sum_{k=t}^{\text{min}(t+n,T)-1} \gamma^{k-t}[G_{k:k+1} - Q_{k-1}(S_k, A_k)]$
"""

# ╔═╡ 6819a060-7e26-46cb-9c9d-5c4e3364b66a
md"""
## 7.3 *n*-step Off-policy Learning
"""

# ╔═╡ 8e3415b2-0464-43ee-a16f-39c17364e0be
#based on pseudocode described in book for off-policy n-step Sarsa for estimating Q
function n_step_sarsa_offpolicy(b, ϵ, α, n, states::Vector{S}, sterm, actions::Vector{A}, sim, γ; q0 = 0.0, numep = 1000) where S where A
	#mapping of actions to indicies in action list
	actiondict = Dict(actions[i] => i for i in eachindex(actions))
	numactions = length(actions)

	#initialize action values as a vector of values for each state
	Q = Dict(s => fill(q0, numactions) for s in states)
	Q[sterm] = fill(0.0, numactions)

	#initialize policy to be random at each state
	π = Dict(s => ones(numactions) ./ numactions for s in states)

	#define a function to select actions from behavior policy
	sample_action(s) = sample(actions, weights(b[s])) 

	#with a probability ϵ a random action will be selected
	baseval = ϵ / n

	#with a probability 1-ϵ the greedy action will be selected
	bonusval = 1.0 - ϵ
	
	#define a function to update π to be ϵ-greedy wrt Q
	function update_π!()
		for s in states
			qvec = Q[s]
			i = argmax(qvec)
			π[s] .= baseval
			π[s][i] += bonusval
		end
	end

	#initialize vectors to store a history up to length n+1 of states, actions, and rewards
	svec = Vector{S}(undef, n+1)
	avec = Vector{A}(undef, n+1)
	rvec = Vector{Float64}(undef, n+1)

	#define index calculator
	getind(i) = mod(i, n+1) + 1

	#sample action value dictionaries given storage index
	function getvalue(d::Dict, i)
		ind = getind(i)
		d[svec[ind]][actiondict[avec[ind]]]
	end
	
	for ep in 1:numep
		#initialize state
		s0 = rand(states)
		#initialize action
		a0 = sample_action(s0)
		
		svec[1] = s0
		avec[1] = a0
		s = s0
		a = a0
		T = typemax(Int64)
		τ = 0
		t = 0
		while τ != T - 1
			if t < T
				(s, r) = sim(svec[getind(t)], a)
				storeind = getind(t+1)
				svec[storeind] = s
				rvec[storeind] = r
				if s == sterm 
					T = t + 1
				else
					a = sample_action(s)
					avec[storeind] = a
				end
			end
			τ = t - n + 1
			if τ >= 0
				ρ = prod(getvalue(π, i)/getvalue(b, i) for i in τ+1:min(τ+n, T-1))
				G = sum(γ^(i - τ - 1) * rvec[getind(i)] for i in (τ + 1):min(τ+n, T))
				if τ+n < T
					G += γ^n * getvalue(Q, τ+n)
				end
				ind = getind(τ)
				Q[svec[ind]][actiondict[avec[ind]]] += α*ρ*(G-getvalue(Q, τ))
				update_π!()
			end
			t += 1
		end
	end

	greedy_π = Dict(s => actions[argmax(π[s])] for s in states)
	
	return Q, greedy_π
end

# ╔═╡ 17c52a18-33ae-47dd-aa43-07440c586b6c
md"""
## 7.4 *Per-decision Methods with Control Variates

For the *n* steps ending at horizon *h*, the *n*-step return can be written

$G_{t:h} = R_{t+1} + \gamma G_{t+1:h}, t<h<T$

where $G_{h:h} \dot = V_{h-1}(S_h)$ (Recall that this return is used at time *h*, previously denoted $t+n$).  Now consider the effect of following a behavior policy *b* that is not the same as the target policy π.  All of the resulting experience, including the first reward $R_{t+1}$ and the next state $S_{t+1}$ must be weighted by the importance sampling ratio for time $t$, $\rho_t = \frac{\pi(A_t|S_t)}{b(A_t,S_t)}$.  One might be tempted to simply weight the righthand side of the above equation, but one can do better.  Suppose the action at time $t$ would never be selected by $\pi$, so that $\rho_t$ is zero.  Then a simple weighting owuld result in the *n*-step return being zero, which oculd result in high variance when it was used as a target.  Instead, in this more sophisticated approach, one uses an alternate, *off-policy* definition of the *n*-step return ending at horizon *h*, as 

$G_{t:h} \dot = \rho_t (R_{t+1}+\gamma G_{t+1:h}) + (1-\rho_t)V_{h-1}(S_t), t<h<T$

In this approach, if $\rho_t$ is zero, then instead of the target being zero and causing the estimate to shrink, the target is the same as the estimate and cuases no change.  The importance sampling ratio being zero means we should ignore the sample, so leaving the estimate unchanged seemed appropriate.
"""

# ╔═╡ baab474f-e491-4b73-8d08-afc4a3bacde5
md"""
> *Exercise 7.5* Write the pseudocode for the off-policy state-value prediction algorithm described above.

Going off the new definition of the *n*-step return, we can expand it out per step:

$G_{t:h} \dot = \rho_t (R_{t+1}+\gamma G_{t+1:h}) + (1-\rho_t)V_{h-1}(S_t), t<h<T$

$= \rho_t R_{t+1} + (1-\rho_t)V_{h-1}(S_t) + \rho_t \gamma G_{t+1:h}$

$= \rho_t R_{t+1} + (1-\rho_t)V_{h-1}(S_t) + \rho_t \gamma [\rho_{t+1} R_{t+2} + (1-\rho_{t+1})V_{h-1}(S_{t+1}) + \rho_{t+1} \gamma G_{t+2:h}]$

$= \rho_t [R_{t+1} + \gamma \rho_{t+1} R_{t+2}] + (1-\rho_t)V_{h-1}(S_t) + \rho_t \gamma (1-\rho_{t+1})V_{h-1}(S_{t+1}) + \rho_t \rho_{t+1} \gamma^2 G_{t+2:h}$

$=\sum_{i=t}^{h-1} \left [ \rho_{t:i} \left ( \gamma^{i-t} R_{t+1} + \left ( \frac{1}{\rho_i}-1 \right )V_{h-1}(S_i) \right ) \right ] + \rho_{t:h-1} \gamma^h V_{h-1}(S_h)$

where $\rho_{t:h} \dot = \prod_{i=t}^{h} \rho_i$

See implementation below
	"""

# ╔═╡ 1ea5aa34-38e9-40d5-a5b5-a30991b23413
function n_step_TD_Vest_offpolicy(b, π, α, n, states, sterm, sim, γ; v0 = 0.0, numep = 1000, Vtrue = Dict(s => v0 for s in states))
	V = Dict(s => v0 for s in states)
	V[sterm] = 0.0
	Svec = Vector{eltype(states)}(undef, n+1)
	Rvec = Vector{Float64}(undef, n+1)
	ρvec = Vector{Float64}(undef, n+1)

	#define index calculator
	getind(i) = mod(i, n+1) + 1

	#get value at modded index
	getvalue(v, i) = v[getind(i)]

	#define a function to select actions from behavior policy
	sample_action(s) = sample(actions, weights(b[s])) 

	for ep in 1:numep
		#for each episode save a record of states and rewards
		s0 = rand(states)
		Svec[1] = s0
		s = s0
		T = typemax(Int64)
		τ = 0
		t = 0
		while τ != T - 1
			if t < T
				storeind = getind(t)
				a = sample_action(s)
				ρvec[storind] = π[s][a] / b[s][a]
				(s, r) = sim(getvalue(Svec, t), a)
				Svec[storeind] = s
				Rvec[storeind] = r
				(s == sterm) && (T = t + 1)
			end
			τ = t - n + 1

			if τ >= 0
				ρ_prod = cumprod(getvalue(ρvec, i) for i in (τ + 1):min(τ+n, T))
				G = sum(ρ_prod[i] * (γ^(i - τ - 1) * getvalue(Rvec, i) + (1.0/getvalue(ρvec, i) - 1.0)*V[getvalue(svec, i)]) for i in (τ + 1):min(τ+n, T))
				if τ+n < T
					G += γ^n * ρ_prod[end] * V[getvalue(Svec, τ+n)]
				end
				if τ == 0
					V[s0] += α*(G - V[s0])
				else
					V[getvalue(Svec, τ)] += α*(G - V[getvalue(Svec, τ)])
				end
			end
			t += 1
		end
	end
	return V
end

# ╔═╡ fd9b3f70-bd00-4e30-8231-6ada24529585
md"""
For action value estimates using off-policy control variates, we can write it recursively as:

$G_{t:h} \dot = R_{t+1} + \gamma \left ( \rho_{t+1}G_{t+1:h} + \bar V_{h+1}(S_{t+1}) - \rho_{t+1}Q_{h-1}(S_{t+1}, A_{t+1}) \right )$

$= R_{t+1} + \gamma \rho_{t+1} \left ( G_{t+1:h} - Q_{h-1}(S_{t+1}, A_{t+1}) \right ) + \gamma \bar V_{h-1}(S_{t+1}), t<h \leq T$

If $h<T$, then the recursion ends with $G_{h:h} \dot = Q_{h-1}(S_h, A_h)$, whereas if $h \geq T$, the recursion ends with $G_{T-1:h} \dot = R_T$.  Thre resultant prediction algorithm (after combining with (7.5)) is analogous to Expected Sarsa.
"""

# ╔═╡ b8ae179f-3ab9-4d8d-a49d-5d4035af63fd
md"""
> *Exercise 7.6* Prove that the control variate in the above equations does not change the expected value of the return

From above we have:

$G_{t:h} = R_{t+1} + \gamma \rho_{t+1} \left ( G_{t+1:h} - Q_{h-1}(S_{t+1}, A_{t+1}) \right ) + \gamma \bar V_{h-1}(S_{t+1}), t<h \leq T$

If we unroll this recursively we get

$G_{t:h} = R_{t+1} + \gamma \rho_{t+1} \left ( R_{t+2} + \gamma \rho_{t+2} \left ( G_{t+2:h} - Q_{h-1}(S_{t+2}, A_{t+2}) \right ) + \gamma \bar V_{h-1}(S_{t+2}) - Q_{h-1}(S_{t+1}, A_{t+1}) \right ) + \gamma \bar V_{h-1}(S_{t+1})$

$G_{t:h} = R_{t+1} + \gamma \rho_{t+1} \left ( R_{t+2} + \gamma \rho_{t+2} \left ( R_{t+3} + \gamma \rho_{t+3} \left ( G_{t+4:h} - Q_{h-1}(S_{t+3}, A_{t+3} \right ) + \gamma \bar V_{h-1}(S_{t+3}) - Q_{h-1}(S_{t+2}, A_{t+2}) \right ) + \gamma \bar V_{h-1}(S_{t+2}) - Q_{h-1}(S_{t+1}, A_{t+1}) \right ) + \gamma \bar V_{h-1}(S_{t+1})$

$\vdots$

$G_{t:h} = R_{t+1} + \gamma \bar V_{h-1}(S_{t+1}) + \gamma^{h-1} \rho_{t+1:h} G_{h:h} + \sum_{i=t+1}^{h-1} \gamma^{i-1}\rho_{t+1:i} \left (R_{i+1} - Q_{h-1}(S_i, A_i) + \gamma \bar V_{h-1}(S_{i+1}) \right )$

In the case of $h<T$ this becomes

$G_{t:h} = R_{t+1} + \gamma \bar V_{h-1}(S_{t+1}) + \gamma^{h} \rho_{t+1:h} Q_{h-1}(S_h, A_h) + \sum_{i=t+1}^{h-1} \gamma^{i-1}\rho_{t+1:i} \left (R_{i+1} - Q_{h-1}(S_i, A_i) + \gamma \bar V_{h-1}(S_{i+1}) \right )$

and in the case of $h \geq T$ this becomes

$G_{t:h} = R_{t+1} + \gamma \bar V_{h-1}(S_{t+1}) + \gamma^{h} \rho_{t+1:h} R_T + \sum_{i=t+1}^{h-1} \gamma^{i-1}\rho_{t+1:i} \left (R_{i+1} - Q_{h-1}(S_i, A_i) + \gamma \bar V_{h-1}(S_{i+1}) \right )$

Now applying expected values to the return expression where $\mathbb{E}$ means the expectation under sampling by the behavior policy unless specified otherwise with an underscore:

$R_{t+1} + \gamma \mathbb{E}[\bar V_{h-1}(S_{t+1})] + \gamma^{h} \mathbb{E}[\rho_{t+1:h} R_T] + \mathbb{E} \left [ \sum_{i=t+1}^{h-1} \gamma^{i-1}\rho_{t+1:i} \left (R_{i+1} - Q_{h-1}(S_i, A_i) + \gamma \bar V_{h-1}(S_{i+1}) \right ) \right ]$

$R_{t+1} + \gamma \mathbb{E_\pi}[G_{t+1:T}] + \sum_{i=t+1}^{h} \gamma^{i-1} \mathbb{E} [\rho_{t+1:i} R_{i+1} ] + \sum_{i=t+1}^{h-1} \gamma^{i-1} \mathbb{E} \left [ \rho_{t+1:i} \left ( - Q_{h-1}(S_i, A_i) + \gamma \bar V_{h-1}(S_{i+1}) \right ) \right ]$

$\mathbb{E_\pi}[G_{t:h}] + \gamma \mathbb{E_\pi}[G_{t+1:T}] + \sum_{i=t+1}^{h-1} \gamma^{i-1} \mathbb{E} \left [ \rho_{t+1:i} \left ( - Q_{h-1}(S_i, A_i) + \gamma \bar V_{h-1}(S_{i+1}) \right ) \right ]$
"""

# ╔═╡ ec706721-a414-47c9-910e-9d58e77664ea
md"""
# Dependencies
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
PlutoPlotly = "~0.4.5"
PlutoUI = "~0.7.58"
StatsBase = "~0.34.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "ebf0638d19b8d1d937688ae1dd56a5a0febafe3c"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0f748c81756f2e5e6854298f11ad8b2dfae6911a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BaseDirs]]
git-tree-sha1 = "3e93fcd95fe8db4704e98dbda14453a0bfc6f6c3"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.2.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

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
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "c955881e3c981181362ae4088b35995446298b80"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.14.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1fb174f0d48fe7d142e1109a10636bc1d14f5ac2"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.17"

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

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

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
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

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

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

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
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

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

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

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

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "BaseDirs", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "Reexport", "TOML"]
git-tree-sha1 = "fbf637823ec24c5669b1a66f3771c2306f60857c"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.4.5"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "71a22244e352aa8c5f0f2adde4150f62368a3f2e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.58"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

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
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

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
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

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
# ╟─b8e46d42-44f1-4f34-a501-7a4943e60a83
# ╟─4e4ee05c-c585-4847-b2ab-0e5c4c7f6ca4
# ╠═28e10379-a475-4b29-bfc6-f9c52201358a
# ╠═88e27f19-2cfb-4e17-8a62-5eadafbda85e
# ╠═4f7135bf-911d-4d57-a845-e6f86ff7f9a4
# ╟─28d9394a-6ce0-4212-941e-2a699e08b7df
# ╟─5954a783-14a6-4d2b-b342-9fee0d9ef564
# ╠═6445e66b-1c07-4474-81ff-5c4cbba88ca6
# ╠═a0092f58-8f2d-4d98-895a-5cc622fb8f4f
# ╠═7f32af81-580b-4f7b-8a71-0aa67928afa4
# ╠═1e199d68-4bc1-4f56-bbce-f4299e649dee
# ╠═7f6dee5f-1cc5-4dab-86c2-a68c92bc135a
# ╠═d3d39d13-e711-4289-a893-28c2c1af50f6
# ╠═3b1ee21f-6147-40bb-a534-26dd255ce26e
# ╠═e6ad9fb0-9efe-4a38-8160-43f1b9c7ee40
# ╠═63e005e5-ca51-4c18-8dcf-50db1ec831c8
# ╟─e79f15d7-c31c-4fc3-917d-8f7c7bf88d59
# ╟─b736204d-bb74-479f-abc2-b59af511227c
# ╠═da4a8d1b-83e6-416b-bdb1-49a3ea120cbe
# ╟─14d8454c-76ef-4922-9bcd-d62c91ae7518
# ╟─8e6fd9da-c8d2-4d29-bae2-d82315103693
# ╟─6376fefa-480f-4e3f-93ee-5425388a4f27
# ╟─81e53593-3a5c-4949-aa8d-9d92abe2ccb1
# ╟─0e318ccf-f4ad-4715-8415-125539e02690
# ╟─e10b6467-05f4-40b8-b899-2a31b5067030
# ╠═f4d7917d-f773-46d4-8605-87195c293d11
# ╠═9d12661a-e830-4205-8504-a1da35a84521
# ╠═703e9324-e4cf-494a-9f3f-8bd258938bd7
# ╠═0c64afd6-ef23-4582-9250-2c1d4ae3cc43
# ╠═8ac0f5e2-d2a2-4bc7-a4a0-910068947f90
# ╟─1b12b915-4576-4c3d-8360-50eb9ad2392d
# ╟─6819a060-7e26-46cb-9c9d-5c4e3364b66a
# ╠═8e3415b2-0464-43ee-a16f-39c17364e0be
# ╟─17c52a18-33ae-47dd-aa43-07440c586b6c
# ╟─baab474f-e491-4b73-8d08-afc4a3bacde5
# ╠═1ea5aa34-38e9-40d5-a5b5-a30991b23413
# ╟─fd9b3f70-bd00-4e30-8231-6ada24529585
# ╠═b8ae179f-3ab9-4d8d-a49d-5d4035af63fd
# ╟─ec706721-a414-47c9-910e-9d58e77664ea
# ╠═0321b9d1-7d4e-4bf8-ac61-9c16ab6bc461
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
