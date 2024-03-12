### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 0321b9d1-7d4e-4bf8-ac61-9c16ab6bc461
begin
	using Statistics, PlutoPlotly, Random, StatsBase, PlutoUI, HypertextLiteral, DataStructures, StaticArrays, Transducers
	TableOfContents()
end

# ╔═╡ b8e46d42-44f1-4f34-a501-7a4943e60a83
md"""
# Chapter 7 $n$-step Bootstrapping

In this chapter we unify the Monte Carlo (MC) methods and the one-step temporal difference (TD) methods presented in the previous two chapters.  Neither MC methods nor one-step TD methods are always the best.  In this chapter we present *n-step TD methods* that generalize both methods so that one can shift from one to the other smoothly as needed to meet the demands of a particular task.  *n*-step methods span a spectrum with MC methods at one end and one-step TD methods at the other.  The best methods are often intermediate between the two extremes.

Another way of looking at the benefits of *n*-step methods is that they free you from the tyranny of the time step.  With one-step TD methods the same time step determines how often the action can be changed and the time interval over which bootstrapping is done.  In many applications one wants to be able to update the action very fast to take into account anything that has changed, but bootstrapping works best if it is over a length of time in which a significant and recognizable state change has occured (e.g. in a game where the reward is only assigned at the end, bootstrapping may be less effective).  With one-step TD methods, these time intervals are the same, and so a compromise must be made.  *n*-step methods enable bootstrapping to occur over multiple steps, freeing us from the tyranny of the single time step.

As usual, we first consider the prediction problem and then the control problem.  That is, we first consider how *n*-step methods can help in predicting returns as a function of state for a fixed policy (i.e. estimating $v_\pi$).  Then we extend the ideas to action values and control methods.
"""

# ╔═╡ 4e4ee05c-c585-4847-b2ab-0e5c4c7f6ca4
md"""
## 7.1 n-step TD Prediction

Consider estimating $v_\pi$ from sample episodes generated using $\pi$.  Monte Carlo methods perform an update for each state based on the entire sequence of observed rewards from that state until the end of the episode.  The update of one-step TD methods, on the other hand, is based on just the one next reward, bootstrapping from the value of the state one step later as a proxy for the remaining rewards.  One kind of intermediate method, then, would perform an update based on an intermediate number of rewards: more than one, but less than all of them until termination.  For example, a two-step method update would be based on the first two rewards and the estimated value of the state two steps later.  Similarly, we could have three-step updates, four-step updates, and so on.  Figure 7.1 shows the backup diagrams of the spectrum of *n-step updates* for $v_\pi$, with the one-step TD update on the left and the up-until-termination Monte Carlo update on the right.
"""

# ╔═╡ 01f45d99-7cce-48ef-8fc1-eccb35dac6ea
md"""
### Figure 7.1

The backup diagrams of *n*-step methods.  These methods form a spectrum ranging from one-step TD methods to Monte Carlo methods.
"""

# ╔═╡ 28e10379-a475-4b29-bfc6-f9c52201358a
HTML("""
<div style = "display: flex; align-items: flex-end; background-color: white; color: black;">
	<div style = "padding-left: 3vw; padding-right: 3vw;">1-step TD <br> and TD(0)</div>
	<div style = "padding-left: 3vw; padding-right: 3vw;">2-step TD</div>
	<div style = "padding-left: 3vw; padding-right: 3vw;">3-step TD</div>
	<div style = "padding-left: 3vw; padding-right: 3vw;">n-step TD</div>
	<div style = "padding-left: 3vw; padding-right: 3vw;">&infin;-step TD <br> and Monte Carlo</div>
</div>
<div style = "display: flex; align-items: flex-start; background-color: white; padding: 5px;">
<div class = "backup-diagram">
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
</div>
<div class = "backup-diagram">
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
</div>
<div class = "backup-diagram">
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
</div>
<div style = "font-size: 5vw; color: black; transform: translateY(150px);">&hellip;</div>
<div class = "backup-diagram">
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div style = "font-size: 3vw; padding: 10px; color: black;">&vellip;</div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
</div>
<div style = "font-size: 5vw; color: black; transform: translateY(150px);">&hellip;</div>
<div class = "backup-diagram">
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div style = "font-size: 3vw; padding: 10px; color: black;">&vellip;</div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "term"></div>
</div>
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
	width: max(100px, 10vw); 
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
.term {
	width: 30px;
	height: 30px;
	background-color: gray;
	border: 2px solid black;
}
</style>
""")

# ╔═╡ 4f7135bf-911d-4d57-a845-e6f86ff7f9a4
md"""
The methods that use *n*-step updates are still TD methods because they still change an earlier estimate based on how it differs from a later estimate.  Now the later estimate is not one step later, but *n* steps later.  Methods in which temporal difference extends over *n* steps are called *n-step TD methods*.  The TD methods introduced in the previous chapter all used one-step updates, which is why we call them one-step TD methods.  

More formally consider the update of the estimated value of state $S_t$ as a result of the state-reward sequence, $S_t, R_{t+1}, R_{t+2}, \dots, R_T, S_T$ (omitting the actions).  We know that in Monte Carlo updates the estimate of $v_\pi(S_t)$ is updated in the direction of the complete return: 

$G_t \doteq R_{t+1}+\gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1}R_T$

where $T$ is the last time step of the episode.  Let us call this quantity the *target* of the update.  Whereas in Monte Carlo updates the target is the return, in one-step updates the target is the first reward plus the discounted estimated value of the next state, which we call the *one-step return*:

$G_{t:t+1} \doteq R_{t+1} + \gamma V_t(S_{t+1})$

where $V_t : \mathcal{S} \rightarrow \mathbb{R}$ here is the estimate at time $t$ of $v_\pi$.  The subscripts on $G_{t:t+1}$ indicate that this is a truncated return for time $t$ using rewards up until time $t+1$, with the discounted estimate $\gamma V_t(S_{t+1})$ taking the place of the other terms $\gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1} R_T$ of the full return, as discussed in the previous chapter.  Our point now is that this idea makes just as much sense after two steps as it does after one.  The target for a two step update is the *two-step return*:

$G_{t:t+2} \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 V_{t+1}(S_{t+2})$

where now $\gamma^2V_{t+1}(S_{t+2})$ corrects for the absense of the terms $\gamma^2R_{t+3} + \gamma ^3 R_{t+4} + \cdots + \gamma^{T-t-1}R_T$.  Similarly, the target for an arbitrary *n*-step update is the *n-ste return*:

$G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n V_{t+n-1}(S_{t+n}) \tag{7.1}$

for all $n,t$ such that $n \geq 1$ and $0 \leq t \leq T-n$.  All *n*-step returns can be considered approximations to the full return, truncated after *n* steps and then corrected for the remaining missing terms by $V_{t+n-1}(S_{t+n})$.  If $t+n \geq T$ (if the *n*-step return extends to or beyond termination), then all the missing terms are taken as zero, and the *n*-step return defined to be equal to the ordinary full return $(G_{t:t+n} \doteq G_t \;\; \text{if} \;\; t+n \geq T)$.

Note that *n*-step returns for $n > 1$ involve future rewards and states that are not available at the time of transition from $t$ to $t+1$.  No real algorithm can use the *n*-step return until after it has seen $R_{t+n}$ and computed $V_{t+n-1}$.  The first time these are available is $t+n$.  The natural state-value learning algorithm for using *n*-step returns is thus: 

$V_{t+n}(S_t) \doteq V_{t+n-1}(S_t) + \alpha \left [ G_{t:t+n} - V_{t+n-1}(S_t) \right ], \;\;\;\; 0\leq t <T, \tag{7.2}$

while the values of all other states remain unchanged: $V_{t+n}(s) = V_{t+n-1}(s)$, for all $s \neq S_t$.  We call this algorithm *n-step TD*.  Note that no changes at all are made during the first $n-1$ steps of each episode.  to make up for that, an equal number of additional updates are made at the end of the episode, after termination and before starting the next episode.  Complete code implementing *n-step TD* can be found below in *Exercise 7.2*.
"""

# ╔═╡ 28d9394a-6ce0-4212-941e-2a699e08b7df
md"""
> ### *Exercise 7.1* 
> In Chapter 6 we noted that Monte Carlo error can be written as the sum of TD errors (6.6) if the value estimates don't change from step to step.  Show that the n-step error used in (7.2) can also be written as a sum of TD errors (again if the value estimates don't change) generalizing the earlier result

The n-step update used in (7.2) can be written as:

$V_{t+n}(S_t) \doteq V_{t+n-1}(S_t) + \alpha [G_{t:t+n} - V_{t+n-1}(S_t)], \; 0\leq t <T$

while the TD errors  are defined as:

$\delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$

Per the conventions used in the earlier derivation, the n-step error is the term being multipled by α:

$\begin{flalign}
G_{t:t+n} - V(S_t) &= \sum_{i=1}^n \gamma^{i-1}R_{t+i} + \gamma^n V(S_{t+n}) - V(S_t) \\
&=R_{t+1} + \gamma V(S_{t+1}) - V(S_t) + \sum_{i=2}^n \gamma^{i-1}R_{t+i} - \gamma V(S_{t+1}) + \gamma^n V(S_{t+n}) \\
&=\delta_t + \gamma R_{t+2} + \sum_{i=3}^n \gamma^{i-1}R_{t+i} - \gamma V(S_{t+1}) + \gamma^n V(S_{t+n})\\
&=\delta_t + \gamma \left [ R_{t+2} - V(S_{t+1}) + \gamma V(S_{t+2}) \right ]  + \sum_{i=3}^n \gamma^{i-1}R_{t+i} + \gamma^n V(S_{t+n}) - \gamma^2 V(S_{t+2}) \\
&=\delta_t + \gamma \delta_{t+1} + \sum_{i=3}^n \gamma^{i-1}R_{t+i} + \gamma^n V(S_{t+n}) - \gamma^2 V(S_{t+2})
\end{flalign}$

Extending this procedure out to i = n yields:

$\begin{flalign}
&=\left ( \sum_{i=0}^{n-2} \gamma^i \delta_{t+i} \right ) - \gamma^{n-1}V(S_{t+n-1}) + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n}) \\
&= \left ( \sum_{i=0}^{n-2} \gamma^i \delta_{t+i} \right ) + \gamma^{n-1} \left [\gamma V(S_{t+n}) - V(S_{t+n-1}) + R_{t+n} \right ] \\
&= \left ( \sum_{i=0}^{n-2} \gamma^i \delta_{t+i} \right ) + \gamma^{n-1} \delta_{t+n-1} \\
&=\sum_{i=0}^{n-1} \gamma^i \delta_{t+i}
\end{flalign}$
"""

# ╔═╡ 5954a783-14a6-4d2b-b342-9fee0d9ef564
md"""
> ### *Exercise 7.2 (programming)* 
> With an n-step method, the value estimates *do* change from step to step, so an algorithm that used the sum of TD errors (see previous exercise) in place of the error in (7.2) would actually be a slightly different algorithm.  Would it be a better algorithm or a worse one?  Devise and program a small experiment to answer this question empirically.

The pseudo-code for the typical algorithm is given already.  The algorithm to compare this to is one that calculates all the updated values for an episode using the n-step method but does not update the values until the end of the episode.  This can be tested by having two versions of V so that the values from the end of the previous episode can be preserved all while the new values are being updated.
"""

# ╔═╡ 4517fbb5-b608-4b6b-8130-c7d99cb149cd
md"""
#### *n*-step TD for estimating $V \approx v_\pi$
The following algorithm is based on the pseudocode found in the book
"""

# ╔═╡ 4bc8a25e-2681-44e7-b4c3-9f1314f3c268
@bind ex7_2_params PlutoUI.combine() do Child
	md"""
	Length of Chain: $(Child(:nstates, NumberField(2:50, default = 19)))
	n-step Learning Parameter: $(Child(:nstep, NumberField(1:50, default = 4)))
	"""
end

# ╔═╡ 2d8178db-7247-4ecf-a3fc-802536d82292
md"""
RMS error on $(ex7_2_params.nstates) state chain with $(ex7_2_params.nstep)-step TD Learning.  The two curves represent learning with and without the state values updating during an episode.  Over a wide range of chain lengths and n-step selections, the static method has worse performance.  For n-step parameters that approach pure Monte-Carlo learning, the static method has an advantage for longer chains.
"""

# ╔═╡ 496ad31c-fd21-409c-9425-d86a242c766e
md"""
The *n*-step return uses the value function $V_{t+n-1}$ to correct for the missing rewards beyond $R_{t+n}$.  An important property of *n*-step returns is that their expectation is guaranteed to be a better estimate of $v_\pi$ than $V_{t+n-1}$ is, in a worst-state sense.  That is, the worst error of the expected *n*-step return is guaranteed to be less than or equal to $\gamma^n$ times the worst error under $V_{t+n-1}$:

$\max_s \bigg\vert \mathbb{E}_\pi [G_{t:t+n} \vert S_t=s]-v_\pi(s) \bigg\vert \leq \gamma^n \max_s \bigg\vert V_{t+n-1}(s) - v_\pi(s) \bigg\vert \tag{7.3}$

for all $n \geq 1$.  Tis is called the *error reduction property* of *n*-step returns.  Because of the error reeuction property, one can show formally that all *n*-step TD methods converge to the correct predictions under appropriate technical conditions.  The *n*-step TD methods thus form a family of sound methods, with one-step TD methods and Monte Carlo methods as extreme members.
"""

# ╔═╡ 61568d33-a45f-492b-8deb-740483191238
md"""
### Example 7.1: *n*-step TD Methods on the Random Walk
Consider using *n*-step TD methods on the 5-state random walk task described in Example 6.2.  Suppose the first episode progressed directly from the center state, C, to the right, through D and E, and then terminated on the right with a return of 1.  Recall that the estimated values of all the states started at an intermediate value, $V(s) = 0.5$.  As a result of this experience, a one-step method would change only the estimate for the last state, $V(\text{E})$, which would be incremented toward 1, the observed return.  A two-step method, on the other hand, would increment the values of the two states precding termination: $V(\text{D})$ and $V(\text{E})$ both would be incremented toward 1.  A three-step method, or any *n*-step method for $n > 2$ would increment the values of all three of the visited states toward 1, all by the same amount.

Which value of *n* is better?  Figure 7.2 shows the results of a simple empirical test for a larger random walk process, with 19 states instead of 5 (and with a -1 outcome on the left, all values initialized to 0), which we use as a running example in this chapter.  Results are shown for *n*-step TD methods with a range of values for $n$ and $\alpha$.  The performance measure for each parameter setting, shown on the vertical axis, is the square-root of the average squared error between the predictions at the end of the episode for the 19 states and their true values, then averaged over the first 10 episodes and 100 repetitions of the whole experiment.  Note that methods with an intermediate value of *n* worked best.  This illustratees how the generalization of TD and Monte Carlo methods to *n*-step methods can potentially perform better than either of the two extreme methods.
"""

# ╔═╡ b88b0b6f-ca6e-4550-ae72-3850e71d99af
md"""
Number of States: $(@bind eg_7_2 NumberField(3:50, default = 19))
Initial State Value: $(@bind eg_7_2_init NumberField(-1:0.1:1, default = 0.0))
"""

# ╔═╡ aebdea2d-1cf2-449d-887a-de9f1b160eba
md"""
#### Figure 7.2: Performance of *n*-step TD methods as a function of $\alpha$, for various values of $n$ on a $eg_7_2-state random walk task
"""

# ╔═╡ 0e318ccf-f4ad-4715-8415-125539e02690
md"""
> ### *Exercise 7.3* 
> Why do you think a larger random walk task (19 states instead of 5) was used in the examples of this chapter?  Would a smaller walk have shifted the advantage to a different value of n?  How about the change in the left-side outcome from 0 to -1 made in the larger walk?  Do you think that made any difference in the best value of n?

For the shorter chain, an n of 1 seems optimal over the first 10 episodes whereas the longer chain estimate performs best at n = 4.  Because each state is much closer to a terminal state in the shorter chain, there is less need to use estimates from states further away from the current state.  The information from the terminal state does not have to diffuse very far to reach any given state compared to the 19 state chain.  Changing the left side outcome to -1 means that the initial value of 0 is now the true value for the central state and this is also the initialization that minimizes the starting error given the constraint that all state values are equal.  As can be observed by modifying the initial value in Figure 7.2, as the value is changed from 0, then the optimal *n* gets larger, favoring a method closer to Monte-Carlo estimation.  TD methods rely on the information already contained in the state-value estiamtes, so if we choose a worse initialization, then methods that rely less on these values and more on the observed returns will become more favored. 
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

# ╔═╡ 0968a366-d843-49cf-8931-94e5b5b04cd6
md"""
## MDP Types and Functions
"""

# ╔═╡ 0f62c007-aff7-4927-ade3-b315b48d4d18
function sample_action(π::Matrix{T}, i_s::Integer) where T<:AbstractFloat
	(n, m) = size(π)
	sample(1:n, weights(π[:, i_s]))
end

# ╔═╡ 4fa8f201-c78a-4d51-9129-eb549e6e9a9b
makelookup(v::Vector) = Dict(x => i for (i, x) in enumerate(v))

# ╔═╡ 63ef17d5-8aee-4ea1-872a-af4fa0ac2e3f
struct MDP_TD{S, A, F<:Function, G<:Function, H<:Function}
	states::Vector{S}
	statelookup::Dict{S, Int64}
	actions::Vector{A}
	actionlookup::Dict{A, Int64}
	state_init::G #function that produces an initial state for an episode
	step::F #function that produces reward and updated state given a state action pair
	isterm::H #function that returns true if the state input is terminal
	function MDP_TD(states::Vector{S}, actions::Vector{A}, state_init::G, step::F, isterm::H) where {S, A, F<:Function, G<:Function, H<:Function}
		statelookup = makelookup(states)
		actionlookup = makelookup(actions)
		new{S, A, F, G, H}(states, statelookup, actions, actionlookup, state_init, step, isterm)
	end
end

# ╔═╡ 8dc47055-933e-4319-9d0a-963e783615d4
function make_random_policy(mdp::MDP_TD; init::T = 1.0f0) where T <: AbstractFloat
	ones(T, length(mdp.actions), length(mdp.states)) ./ length(mdp.actions)
end

# ╔═╡ 6cef8b9e-00ed-4eae-96f3-9f8fab2f6fd2
initialize_state_value(mdp::MDP_TD; vinit::T = 0.0f0) where T<:AbstractFloat = ones(T, length(mdp.states)) .* vinit

# ╔═╡ 392b039e-5cef-4ca7-a340-bbfe2e0efaf7
initialize_state_action_value(mdp::MDP_TD; qinit::T = 0.0f0) where T<:AbstractFloat = ones(T, length(mdp.actions), length(mdp.states)) .* qinit

# ╔═╡ 96a3e618-9d94-4954-9431-059d2621ebbd
function check_policy(π::Matrix{T}, mdp::MDP_TD) where {T <: AbstractFloat}
#checks to make sure that a policy is defined over the same space as an MDP
	(n, m) = size(π)
	num_actions = length(mdp.actions)
	num_states = length(mdp.states)
	@assert n == num_actions "The policy distribution length $n does not match the number of actions in the mdp of $(num_actions)"
	@assert m == num_states "The policy is defined over $m states which does not match the mdp state count of $num_states"
	return nothing
end

# ╔═╡ 2e85c71b-c630-4d46-b789-9af014f69a92
#take a step in the environment from state s using policy π
function takestep(mdp::MDP_TD{S, A, F, G, H}, π::Matrix{T}, s::S) where {S, A, F<:Function, G<:Function, H<:Function, T<:Real}
	i_s = mdp.statelookup[s]
	i_a = sample_action(π, i_s)
	a = mdp.actions[i_a]
	(r, s′) = mdp.step(s, a)
	i_s′ = mdp.statelookup[s′]
	return (i_s, i_s′, r, s′, a, i_a)
end

# ╔═╡ 6445e66b-1c07-4474-81ff-5c4cbba88ca6
function n_step_TD_Vest(π::Matrix{X}, mdp::MDP_TD{S, A, F, E, H}, n::Integer, α::X, γ::X; num_episodes::Integer = 1000, vinit::X = zero(X), V::Vector{X} = initialize_state_value(mdp; vinit = vinit), save_states::Vector{S} = Vector{S}(), static_values = false) where {X <: AbstractFloat, S, A, F, E, H}
	check_policy(π, mdp)
	terminds = findall(mdp.isterm(s) for s in mdp.states)
	
	#initialize
	stateindexbuffer = MVector{n+1, Int64}(zeros(Int64, n+1))
	rewardbuffer = MVector{n+1, X}(zeros(X, n+1))
	get_state_index(i) = stateindexbuffer[mod(i, n+1) + 1]
	get_reward(i) = rewardbuffer[mod(i, n+1) + 1]
	get_value(buffer, i) = buffer[mod(i, n+1)+1]
	V[terminds] .= zero(X) #terminal state must always have 0 value
	if static_values
		V_copy = copy(V)
	end
	v_saves = zeros(X, length(save_states), num_episodes+1)
	function updatesaves!(ep)
		for (i, s) in enumerate(save_states)
			i_s = mdp.statelookup[s]
			v_saves[i, ep] = V[i_s]
		end
	end
	updatesaves!(1)

	#simulate and episode and update the value function every step
	function runepisode!(V, j)
		s = mdp.state_init()
		T = typemax(Int64)
		t = 0
		τ = 0
		stateindexbuffer[1] = mdp.statelookup[s]
		while τ != T - 1
			if t < T
				(i_s, i_s′, r, s′, a, i_a) = takestep(mdp, π, s)
				s = s′
				i = mod(t+1, n+1) + 1
				stateindexbuffer[i] = i_s′
				rewardbuffer[i] = r
				if mdp.isterm(s′)
					T = t + 1
				end
			end
			τ = t - n + 1
			if τ >= 0
				G = zero(X)
				for i in τ+1:min(τ+n, T)
					G += (γ^(i - τ - 1))*get_reward(i)
				end
				if τ+n < T
					G += γ^n * V[get_state_index(τ+n)]
				end
				i_τ = get_value(stateindexbuffer, τ)
				update_value = V[i_τ] + α*(G-V[i_τ])
				if static_values
					V_copy[i_τ] = update_value
				else
					V[i_τ] = update_value
				end
			end
			t += 1
		end
		updatesaves!(j+1)
		if static_values
			V .= V_copy
		end
		return V
	end
		
	for i = 1:num_episodes;	runepisode!(V, i); end
	
	return V, v_saves
end

# ╔═╡ 5fef24aa-55f2-4ece-9722-6c1a132d793c
function runepisode(mdp::MDP_TD{S, A, F, G, H}, π::Matrix{T}; max_steps = Inf) where {S, A, F, G, H, T<:Real}
	states = Vector{S}()
	actions = Vector{A}()
	rewards = Vector{T}()
	s = mdp.state_init()
	step = 1

	#note that the terminal state will not be added to the state list
	while !mdp.isterm(s) && (step <= max_steps)
		push!(states, s)
		(i_s, i_s′, r, s′, a, i_a) = takestep(mdp, π, s)
		push!(actions, a)
		push!(rewards, r)
		s = s′
		step += 1
	end
	return states, actions, rewards
end

# ╔═╡ 36b394dd-1956-40ab-9600-624e6900f665
const mrp_moves = [-1, 1]

# ╔═╡ 2e3b1699-0e35-43f6-800b-4788b5e3bc7d
function make_mrp(;l = (5), r_left = -1f0, r_right = 1f0)
	function step(s, a)
		x = s + rand(mrp_moves)
		r = r_right*(x == l+1) + r_left*(x == 0)
		(r, mod(x, l+1)) #if a transition is terminal will return 0
	end
	MDP_TD(collect(0:l), [1], () -> ceil(Int64, l/2), step, s -> s == 0)
end

# ╔═╡ d3d39d13-e711-4289-a893-28c2c1af50f6
function value_estimate_random_walk(nstates, α, n; kwargs...)
	mdp = make_mrp(l = nstates)
	π = make_random_policy(mdp)
	Vest,v_saves = n_step_TD_Vest(π, mdp, n, α, 1f0; save_states = collect(1:nstates), kwargs...)
	c = (nstates + 1)/2
	Vtrue = [(s-c)/c for s in 1:nstates]
	sqrt.(mean((v_saves .- Vtrue) .^2, dims = 1)) |> mean
	# (v_saves, Vest)
end

# ╔═╡ ab12f509-ae50-4360-bf0b-874912c00852
function random_walk_method_compare(nstates, n; kwargs...)
	α_vec = Float32.(0.0:0.1:1.0)
	get_α_line(n, static_values) = [mean(value_estimate_random_walk(nstates, α, n; num_episodes = 10, static_values = static_values, kwargs...) for _ in 1:100) for α in α_vec]
	line1 = get_α_line(n, false)
	line2 = get_α_line(n, true)
	trace1 = scatter(x = α_vec, y = line1, name = "Non-Static State Values", mode = "lines", line_shape = "spline")
	trace2 = scatter(x = α_vec, y = line2, name = "Static State Values", mode = "lines", line_shape = "spline")
	plot([trace1, trace2], Layout(xaxis_title = "α", yaxis_title = "Average RMS error over $nstates <br> states and first 10 episodes", yaxis_maxallowed = first(line1), title = "$n-step TD Learning"))
end

# ╔═╡ 05bc2b77-ae63-46c1-9480-5a72c17b7510
random_walk_method_compare(ex7_2_params...)

# ╔═╡ e6ad9fb0-9efe-4a38-8160-43f1b9c7ee40
function nsteptd_error_random_walk(nstates; kwargs...)
	α_vec = Float32.(0.0:0.1:1.0)
	n_vec = 2 .^ (0:9)
	get_α_line(n) = α_vec |> Map(α -> (1:100 |> Map(_ -> value_estimate_random_walk(nstates, α, n; num_episodes = 10, kwargs...)) |> foldxt(+)) / 100) |> collect
	lines = n_vec |> Map(n -> get_α_line(n)) |> collect
	traces = [scatter(x = α_vec, y = lines[i], name = "n = $n", mode = "lines", line_shape = "spline") for (i, n) in enumerate(n_vec)]
	plot(traces, Layout(xaxis_title = "α", yaxis_title = "Average RMS error over $nstates <br> states and first 10 episodes", yaxis_maxallowed = first(first(lines))))
end

# ╔═╡ 953a3fdc-81d9-4693-88bb-0ecace0bf219
nsteptd_error_random_walk(eg_7_2; vinit = Float32(eg_7_2_init))

# ╔═╡ cd7cbdfb-ae59-4e8a-a4bf-c9dad96062c3
runepisode(mdp::MDP_TD; kwargs...) = runepisode(mdp, make_random_policy(mdp); kwargs...)

# ╔═╡ 993110fc-e3bc-46a8-a35b-32b15dac87d5
begin
	abstract type CompleteMDP{T<:Real} end
	struct FiniteMDP{T<:Real, S, A} <: CompleteMDP{T} 
		states::Vector{S}
		actions::Vector{A}
		rewards::Vector{T}
		# ptf::Dict{Tuple{S, A}, Matrix{T}}
		ptf::Array{T, 4}
		action_scratch::Vector{T}
		state_scratch::Vector{T}
		reward_scratch::Vector{T}
		state_index::Dict{S, Int64}
		action_index::Dict{A, Int64}
		function FiniteMDP{T, S, A}(states::Vector{S}, actions::Vector{A}, rewards::Vector{T}, ptf::Array{T, 4}) where {T <: Real, S, A}
			new(states, actions, rewards, ptf, Vector{T}(undef, length(actions)), Vector{T}(undef, length(states)+1), Vector{T}(undef, length(rewards)), Dict(zip(states, eachindex(states))), Dict(zip(actions, eachindex(actions))))
		end	
	end
	FiniteMDP(states::Vector{S}, actions::Vector{A}, rewards::Vector{T}, ptf::Array{T, 4}) where {T <: Real, S, A} = FiniteMDP{T, S, A}(states, actions, rewards, ptf)
end

# ╔═╡ 751e8763-3274-4cf7-80ef-b544b8c46f4b
#forms a random policy for a generic finite state mdp.  The policy is a matrix where the rows represent actions and the columns represent states.  Each column is a probability distribution of actions over that state.
form_random_policy(mdp::CompleteMDP{T}) where T = ones(T, length(mdp.actions), length(mdp.states)) ./ length(mdp.actions)

# ╔═╡ 52eb8101-6dd2-443e-9aff-979f2d2bb532
function make_greedy_policy!(π::Matrix{T}, mdp::FiniteMDP{T, S, A}, V::Vector{T}, γ::T) where {T<:Real,S,A}
	for i_s in eachindex(mdp.states)
		maxv = -Inf
		for i_a in eachindex(mdp.actions)
			x = zero(T)
			for i_r in eachindex(mdp.rewards)
				for i_s′ in eachindex(V)
					x += mdp.ptf[i_s′, i_r, i_a, i_s] * (mdp.rewards[i_r] + γ * V[i_s′])
				end
			end
			maxv = max(maxv, x)
			π[i_a, i_s] = x
		end
		π[:, i_s] .= (π[:, i_s] .≈ maxv)
		π[:, i_s] ./= sum(π[:, i_s])
	end
	return π
end

# ╔═╡ 8f77dd8e-7689-4f85-a990-58550c723920
function create_gridworld_mdp(mdp::MDP_TD, step_reward)
	#this only works when the mdp is deterministic.  add a version for the stochastic wind example
	ptf = zeros(Float32, length(mdp.states), 2, length(mdp.actions), length(mdp.states))
	for s in mdp.states
		i_s = mdp.statelookup[s]
		if mdp.isterm(s)
			ptf[i_s, 1, :, i_s] .= 1.0f0
		else
			for a in mdp.actions
				(r, s′) = mdp.step(s, a)
				i_a = mdp.actionlookup[a]
				i_s′ = mdp.statelookup[s′]
				i_s = mdp.statelookup[s]
				ptf[i_s′, 2, i_a, i_s] = 1.0f0
			end
		end
	end
			
	FiniteMDP(mdp.states, mdp.actions, [0.0f0, step_reward], ptf)	
end

# ╔═╡ 631b2d23-584d-47bb-be24-7fa58be53dfa
function bellman_optimal_value!(V::Vector{T}, mdp::FiniteMDP{T, S, A}, γ::T) where {T <: Real, S, A}
	delt = zero(T)
	@inbounds @fastmath @simd for i_s in eachindex(mdp.states)
		maxvalue = typemin(T)
		@inbounds @fastmath @simd for i_a in eachindex(mdp.actions)
			x = zero(T)
			for (i_r, r) in enumerate(mdp.rewards)
				@inbounds @fastmath @simd for i_s′ in eachindex(V)
					x += mdp.ptf[i_s′, i_r, i_a, i_s] * (r + γ * V[i_s′])
				end
			end
			maxvalue = max(maxvalue, x)
		end
		delt = max(delt, abs(maxvalue - V[i_s]) / (eps(abs(V[i_s])) + abs(V[i_s])))
		V[i_s] = maxvalue
	end
	return delt
end

# ╔═╡ c5e01eea-20af-47b3-8dc4-681d3b01df8f
function value_iteration_v!(V, θ, mdp, γ, nmax, valuelist)
	nmax <= 0 && return valuelist
	
	#update value function
	delt = bellman_optimal_value!(V, mdp, γ)
	
	#add copy of value function to results list
	push!(valuelist, copy(V))

	#halt when value function is no longer changing
	delt <= θ && return valuelist
	
	value_iteration_v!(V, θ, mdp, γ, nmax - 1, valuelist)	
end

# ╔═╡ 459539ca-387a-4e60-894b-94eb2906db42
begin_value_iteration_v(mdp::FiniteMDP{T,S,A}, γ::T; Vinit::T = zero(T), kwargs...) where {T<:Real,S,A} = begin_value_iteration_v(mdp, γ, Vinit .* ones(T, size(mdp.ptf, 1)); kwargs...)

# ╔═╡ ea0b8273-e77c-482e-bc6c-f3e7bc7c7d46
function make_ϵ_greedy_policy!(v::AbstractVector{T}, ϵ::T; valid_inds = eachindex(v)) where T <: Real
	vmax = maximum(view(v, valid_inds))
	v .= T.(isapprox.(v, vmax))
	s = sum(v)
	c = s * ϵ / length(valid_inds)
	d = one(T)/s - ϵ #value to add to actions that are maximizing
	for i in valid_inds
		if v[i] == 1
			v[i] = d + c
		else
			v[i] = c
		end
	end
	return v
end

# ╔═╡ cf3418ed-af8b-4d86-8057-d1b1d22581c7
function make_greedy_policy!(v::AbstractVector{T}; c = 1000) where T<:Real
	(vmin, vmax) = extrema(v)
	if vmin == vmax
		v .= zero(T)
		v .= one(T) / length(v)
	else
		v .= (v .- vmax) ./ abs(vmin - vmax)
		v .= exp.(c .* v)
		v .= v ./ sum(v)
	end
	return v
end

# ╔═╡ 795db9a8-71f4-4664-8aa1-dad112b2da12
function begin_value_iteration_v(mdp::M, γ::T, V::Vector{T}; θ = eps(zero(T)), nmax=typemax(Int64)) where {T<:Real, M <: CompleteMDP{T}}
	valuelist = [copy(V)]
	value_iteration_v!(V, θ, mdp, γ, nmax, valuelist)

	π = form_random_policy(mdp)
	make_greedy_policy!(π, mdp, V, γ)
	return (valuelist, π)
end

# ╔═╡ 9b2e64df-0341-4bd5-8484-3b7c1ef2c828
function create_greedy_policy(Q::Matrix{T}; c = 1000, π = copy(Q)) where T<:Real
	vhold = zeros(T, size(Q, 1))
	for j in 1:size(Q, 2)
		vhold .= Q[:, j]
		make_greedy_policy!(vhold; c = c)
		π[:, j] .= vhold
	end
	return π
end

# ╔═╡ c17dc4eb-5a12-4313-b2e3-defa2be85295
begin
	abstract type GridworldAction end
	struct Up <: GridworldAction end
	struct Down <: GridworldAction end
	struct Left <: GridworldAction end
	struct Right <: GridworldAction end
	struct UpRight <: GridworldAction end
	struct DownRight <: GridworldAction end
	struct UpLeft <: GridworldAction end
	struct DownLeft <: GridworldAction end
	struct Stay <: GridworldAction end
	
	struct GridworldState
		x::Int64
		y::Int64
	end

	rook_actions = [Up(), Down(), Left(), Right()]
	
	move(::Up, x, y) = (x, y+1)
	move(::Down, x, y) = (x, y-1)
	move(::Left, x, y) = (x-1, y)
	move(::Right, x, y) = (x+1, y)
	move(::UpRight, x, y) = (x+1, y+1)
	move(::UpLeft, x, y) = (x-1, y+1)
	move(::DownRight, x, y) = (x+1, y-1)
	move(::DownLeft, x, y) = (x-1, y-1)
	move(::Stay, x, y) = (x, y)
	apply_wind(w, x, y) = (x, y+w)
	const wind_vals = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
end

# ╔═╡ 294d55aa-5de7-4cb2-adf0-85af09fb2464
function make_windy_gridworld(;actions = rook_actions, apply_wind = apply_wind, sterm = GridworldState(8, 4), start = GridworldState(1, 4), xmax = 10, ymax = 7, winds = wind_vals, get_step_reward = () -> -1f0)
	
	states = [GridworldState(x, y) for x in 1:xmax for y in 1:ymax]
	
	boundstate(x::Int64, y::Int64) = (clamp(x, 1, xmax), clamp(y, 1, ymax))
	
	function step(s::GridworldState, a::GridworldAction)
		w = winds[s.x]
		(x1, y1) = move(a, s.x, s.y)
		(x2, y2) = apply_wind(w, x1, y1)
		GridworldState(boundstate(x2, y2)...)
	end

	tr(s0::GridworldState, a0::GridworldAction) = (get_step_reward(), step(s0, a0))
	isterm(s::GridworldState) = s == sterm

			
	MDP_TD(states, actions, () -> start, tr, isterm)
end	

# ╔═╡ eb4a7088-e7b3-4b18-a007-349694a49278
function create_gridworld_mdp(width, height, start, goal, wind, actions, step_reward)
	mdp = make_windy_gridworld(;actions = actions, apply_wind = apply_wind, sterm = goal, start = start, xmax = width, ymax = height, winds = wind_vals, get_step_reward = () -> step_reward)
	ptf = zeros(Float32, length(mdp.states), 2, length(mdp.actions), length(mdp.states))
	for s in mdp.states
		i_s = mdp.statelookup[s]
		if mdp.isterm(s)
			ptf[i_s, 1, :, i_s] .= 1.0f0
		else
			for a in mdp.actions
				w = wind[s.x]
				(r, s′) = mdp.step(s, a)
				i_a = mdp.actionlookup[a]
				i_s = mdp.statelookup[s]
				i_s′ = mdp.statelookup[s′]
				ptf[i_s′, 2, i_a, i_s] = 1.0f0
			end
		end
	end
			
	FiniteMDP(mdp.states, mdp.actions, [0.0f0, step_reward], ptf)	
end

# ╔═╡ 14b269e1-fa81-4ed8-957e-119bff365d0e
function show_grid_value(mdp, Q::Matrix, wind::Vector, name; action_display = king_action_display, scale = 1.0)
	width = maximum(s.x for s in mdp.states)
	height = maximum(s.y for s in mdp.states)
	start = mdp.state_init()
	termind = findfirst(mdp.isterm, mdp.states)
	sterm = mdp.states[termind]
	ngrid = width*height
	@htl("""
		<div style = "display: flex; transform: scale($scale); background-color: white;">
			<div>
				<div class = "gridworld $name value">
					$(HTML(mapreduce(i -> """<div class = "gridcell $name value" x = "$(mdp.states[i].x)" y = "$(mdp.states[i].y)" style = "grid-row: $(height - mdp.states[i].y + 1); grid-column: $(mdp.states[i].x); font-size: 20px; color: black;">$(round(maximum(Q[:, i]), sigdigits = 2))</div>""", *, eachindex(mdp.states))))
				</div>
				<div class = "windrow" style = "display: grid; grid-template-columns: repeat($width, 40px)">
					$(HTML(mapreduce(i -> """<div class="windbox downarrow" w = "$(wind[i])"><div style = "transform: rotate(180deg); color: black;">$(wind[i])</div></div>""", *, 1:width)))
				</div>
			</div>
			<div style = "display: flex; flex-direction: column; align-items: flex-start; justify-content: flex-end; color: black; font-size: 18px; width: 5em; margin-left: 1em;">
				$(action_display)
				<div>Wind Values</div>
			</div>
		</div>
	
		<style>
			.$name.value.gridworld {
				display: grid;
				grid-template-columns: repeat($width, 40px);
				grid-template-rows: repeat($height, 40px);
				background-color: white;
			}

			.$name.value[x="$(start.x)"][y="$(start.y)"] {
				content: '';
				background-color: rgba(0, 255, 0, 0.5);
				
			}

			.$name.value[x="$(sterm.x)"][y="$(sterm.y)"] {
				content: '';
				background-color: rgba(255, 0, 0, 0.5);
				
			}

		</style>
	""")
end

# ╔═╡ 2d1be15f-b0a8-49bd-9636-7f7348a2aae0
function show_grid_value(mdp, V::Vector, wind::Vector, name; action_display = king_action_display, scale = 1.0)
	width = maximum(s.x for s in mdp.states)
	height = maximum(s.y for s in mdp.states)
	start = mdp.state_init()
	termind = findfirst(mdp.isterm, mdp.states)
	sterm = mdp.states[termind]
	ngrid = width*height
	@htl("""
		<div style = "display: flex; transform: scale($scale); background-color: white; margin: 0; padding: 0;">
			<div>
				<div class = "gridworld $name value">
					$(HTML(mapreduce(i -> """<div class = "gridcell $name value" x = "$(mdp.states[i].x)" y = "$(mdp.states[i].y)" style = "grid-row: $(height - mdp.states[i].y + 1); grid-column: $(mdp.states[i].x); font-size: 20px; color: black;">$(round(V[i], sigdigits = 2))</div>""", *, eachindex(mdp.states))))
				</div>
				<div class = "windrow" style = "display: grid; grid-template-columns: repeat($width, 40px)">
					$(HTML(mapreduce(i -> """<div class="windbox downarrow" w = "$(wind[i])"><div style = "transform: rotate(180deg); color: black;">$(wind[i])</div></div>""", *, 1:width)))
				</div>
			</div>
			<div style = "display: flex; flex-direction: column; align-items: flex-start; justify-content: flex-end; color: black; font-size: 18px; width: 5em; margin-left: 1em;">
				$(action_display)
				<div>Wind Values</div>
			</div>
		</div>
	
		<style>
			.$name.value.gridworld {
				display: grid;
				grid-template-columns: repeat($width, 40px);
				grid-template-rows: repeat($height, 40px);
				background-color: white;

			.$name.value[x="$(start.x)"][y="$(start.y)"] {
				content: '';
				background-color: rgba(0, 255, 0, 0.5);
				
			}

			.$name.value[x="$(sterm.x)"][y="$(sterm.y)"] {
				content: '';
				background-color: rgba(255, 0, 0, 0.5);
				
			}

		</style>
	""")
end

# ╔═╡ 87ad13fa-604f-48f7-8232-a2c021b0fefd
function show_grid_policy(mdp, π, wind::Vector, display_function, name; action_display = king_action_display, scale = 1.0)
	width = maximum(s.x for s in mdp.states)
	height = maximum(s.y for s in mdp.states)
	start = mdp.state_init()
	termind = findfirst(mdp.isterm, mdp.states)
	sterm = mdp.states[termind]
	ngrid = width*height
	@htl("""
		<div style = "display: flex; transform: scale($scale); background-color: white;">
			<div>
				<div class = "gridworld $name">
					$(HTML(mapreduce(i -> """<div class = "gridcell $name" x = "$(mdp.states[i].x)" y = "$(mdp.states[i].y)" style = "grid-row: $(height - mdp.states[i].y + 1); grid-column: $(mdp.states[i].x);">$(display_function(π[:, i], scale =0.8))</div>""", *, eachindex(mdp.states))))
				</div>
				<div class = "windrow" style = "display: grid; grid-template-columns: repeat($width, 40px)">
					$(HTML(mapreduce(i -> """<div class="windbox downarrow" w = "$(wind[i])"><div style = "transform: rotate(180deg); color: black;">$(wind[i])</div></div>""", *, 1:width)))
				</div>
			</div>
			<div style = "display: flex; flex-direction: column; align-items: flex-start; justify-content: flex-end; color: black; font-size: 18px; width: 5em; margin-left: 1em;">
				$(action_display)
				<div>Wind Values</div>
			</div>
		</div>
	
		<style>
			.$name.gridworld {
				display: grid;
				grid-template-columns: repeat($width, 40px);
				grid-template-rows: repeat($height, 40px);
				background-color: white;

			.$name[x="$(start.x)"][y="$(start.y)"]::before {
				content: 'S';
				position: absolute;
				color: green;
				opacity: 1.0;
			}

			.$name[x="$(sterm.x)"][y="$(sterm.y)"]::before {
				content: 'G';
				position: absolute;
				color: red;
				opacity: 1.0;
			}

		</style>
	""")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Transducers = "28d57a85-8fef-5791-bfe6-a80928e7c999"

[compat]
DataStructures = "~0.18.17"
HypertextLiteral = "~0.9.5"
PlutoPlotly = "~0.4.5"
PlutoUI = "~0.7.58"
StaticArrays = "~1.9.3"
StatsBase = "~0.34.2"
Transducers = "~0.4.81"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "7a528d9116edbdb7448d61269f824c07bc9f11f8"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0f748c81756f2e5e6854298f11ad8b2dfae6911a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.0"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown", "Test"]
git-tree-sha1 = "c0d491ef0b135fd7d63cbc6404286bc633329425"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.36"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "0fb305e0253fd4e833d486914367a2ee2c2e78d0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BangBang]]
deps = ["Accessors", "Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires"]
git-tree-sha1 = "490e739172eb18f762e68dc3b928cad2a077983a"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.1"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BaseDirs]]
git-tree-sha1 = "3e93fcd95fe8db4704e98dbda14453a0bfc6f6c3"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.2.3"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

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

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1fb174f0d48fe7d142e1109a10636bc1d14f5ac2"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.17"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

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

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "68772f49f54b479fa88ace904f6127f0a3bb2e46"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.12"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

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

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

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

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

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

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "bf074c045d3d5ffd956fa0a461da38a44685d6b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.3"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

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

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

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

[[deps.Transducers]]
deps = ["Accessors", "Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "47e516e2eabd0cf1304cd67839d9a85d52dd659d"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.81"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

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
# ╟─01f45d99-7cce-48ef-8fc1-eccb35dac6ea
# ╟─28e10379-a475-4b29-bfc6-f9c52201358a
# ╟─88e27f19-2cfb-4e17-8a62-5eadafbda85e
# ╟─4f7135bf-911d-4d57-a845-e6f86ff7f9a4
# ╟─28d9394a-6ce0-4212-941e-2a699e08b7df
# ╟─5954a783-14a6-4d2b-b342-9fee0d9ef564
# ╟─4517fbb5-b608-4b6b-8130-c7d99cb149cd
# ╠═6445e66b-1c07-4474-81ff-5c4cbba88ca6
# ╠═d3d39d13-e711-4289-a893-28c2c1af50f6
# ╠═ab12f509-ae50-4360-bf0b-874912c00852
# ╟─4bc8a25e-2681-44e7-b4c3-9f1314f3c268
# ╟─2d8178db-7247-4ecf-a3fc-802536d82292
# ╟─05bc2b77-ae63-46c1-9480-5a72c17b7510
# ╟─496ad31c-fd21-409c-9425-d86a242c766e
# ╟─61568d33-a45f-492b-8deb-740483191238
# ╟─b88b0b6f-ca6e-4550-ae72-3850e71d99af
# ╟─aebdea2d-1cf2-449d-887a-de9f1b160eba
# ╟─953a3fdc-81d9-4693-88bb-0ecace0bf219
# ╠═e6ad9fb0-9efe-4a38-8160-43f1b9c7ee40
# ╟─0e318ccf-f4ad-4715-8415-125539e02690
# ╟─e10b6467-05f4-40b8-b899-2a31b5067030
# ╠═f4d7917d-f773-46d4-8605-87195c293d11
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
# ╟─0968a366-d843-49cf-8931-94e5b5b04cd6
# ╠═0f62c007-aff7-4927-ade3-b315b48d4d18
# ╠═4fa8f201-c78a-4d51-9129-eb549e6e9a9b
# ╠═63ef17d5-8aee-4ea1-872a-af4fa0ac2e3f
# ╠═8dc47055-933e-4319-9d0a-963e783615d4
# ╠═6cef8b9e-00ed-4eae-96f3-9f8fab2f6fd2
# ╠═392b039e-5cef-4ca7-a340-bbfe2e0efaf7
# ╠═96a3e618-9d94-4954-9431-059d2621ebbd
# ╠═2e85c71b-c630-4d46-b789-9af014f69a92
# ╠═5fef24aa-55f2-4ece-9722-6c1a132d793c
# ╠═36b394dd-1956-40ab-9600-624e6900f665
# ╠═2e3b1699-0e35-43f6-800b-4788b5e3bc7d
# ╠═cd7cbdfb-ae59-4e8a-a4bf-c9dad96062c3
# ╠═993110fc-e3bc-46a8-a35b-32b15dac87d5
# ╠═751e8763-3274-4cf7-80ef-b544b8c46f4b
# ╠═52eb8101-6dd2-443e-9aff-979f2d2bb532
# ╠═eb4a7088-e7b3-4b18-a007-349694a49278
# ╠═8f77dd8e-7689-4f85-a990-58550c723920
# ╠═631b2d23-584d-47bb-be24-7fa58be53dfa
# ╠═c5e01eea-20af-47b3-8dc4-681d3b01df8f
# ╠═795db9a8-71f4-4664-8aa1-dad112b2da12
# ╠═459539ca-387a-4e60-894b-94eb2906db42
# ╠═ea0b8273-e77c-482e-bc6c-f3e7bc7c7d46
# ╠═cf3418ed-af8b-4d86-8057-d1b1d22581c7
# ╠═9b2e64df-0341-4bd5-8484-3b7c1ef2c828
# ╠═c17dc4eb-5a12-4313-b2e3-defa2be85295
# ╠═294d55aa-5de7-4cb2-adf0-85af09fb2464
# ╠═14b269e1-fa81-4ed8-957e-119bff365d0e
# ╠═2d1be15f-b0a8-49bd-9636-7f7348a2aae0
# ╠═87ad13fa-604f-48f7-8232-a2c021b0fefd
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
