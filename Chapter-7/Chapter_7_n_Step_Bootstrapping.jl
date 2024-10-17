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
<div style = "display: flex; align-items: flex-end; background-color: white; color: black; font-size: max(20px, 1.5vw);">
	<div style = "padding-left: 4vw; padding-right: 0vw;">1-step TD <br> and TD(0)</div>
	<div style = "padding-left: 4vw; padding-right: 0vw;">2-step TD</div>
	<div style = "padding-left: 4vw; padding-right: 0vw;">3-step TD</div>
	<div style = "padding-left: 4vw; padding-right: 0vw;">n-step TD</div>
	<div style = "padding-left: 4vw; padding-right: 0vw;">&infin;-step TD <br> and Monte Carlo</div>
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

To use $n$-step methods for control, we simply switch states for state-action pairs and then use an $\epsilon$-greedy policy with respect to the state-action value function.  Only now the mechanism for the value updates will be based on the $n$-step return rather than either TD(0) or Monte Carlo returns.

We redefine $n$-step returns (update targets) in terms of estimated action values:

$G_{t:t+n} \dot = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n Q_{t+n-1}(S_{t+n}, A_{t+n}), \; n \geq 1, 0 \leq t \lt T-n$ with $G_{t:t+n} \doteq G_t$ if $t+n \geq T \tag{7.4}$

The natural algorithm is then

$Q_{t+n}(S_t, A_t) \doteq Q_{t+n-1}(S_t, A_t) + \alpha [G_{t:t+n} - Q_{t+n-1}(S_t, A_t)], \;\; 0 \leq t \lt T \tag{7.5}$

while the values of all others states remain unchanged.  This is the algorithm we call *n-step Sarsa*.  Implementation of this algorithm can be found in the next section
"""

# ╔═╡ 71370f7c-4af7-43ae-aba3-da2d412c248a
md"""
### Figure 7.3: 
The backup diagrams for the spectrum of $n$-step methods for state-action values.  They range from the one-step update of Sarsa(0) to the up-until-termination update of the Monte Carlo method.  In between are the $n$-step updates based on $n$ steps of real rewards and the estimated value of the $n$th next state-action pair, all appropriately discounted.
"""

# ╔═╡ e528af0a-1af3-429a-b781-de2e8421b4e8
HTML("""
<div style = "display: flex; align-items: flex-end; background-color: white; color: black; font-size: max(20px, 1.5vw);">
	<div style = "padding-left: 3vw; padding-right: 0vw;">1-step Sarsa <br> aka Sarsa(0)</div>
	<div style = "padding-left: 3vw; padding-right: 0vw;">2-step Sarsa</div>
	<div style = "padding-left: 3vw; padding-right: 0vw;">3-step Sarsa</div>
	<div style = "padding-left: 3vw; padding-right: 0vw;">n-step Sarsa</div>
	<div style = "padding-left: 3vw; padding-right: 0vw;">&infin;-step Sarsa <br> aka Monte Carlo</div>
</div>
<div style = "display: flex; align-items: flex-start; background-color: white; padding: 5px;">
<div class = "backup-diagram">
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
</div>
<div class = "backup-diagram">
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
</div>
<div class = "backup-diagram">
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
</div>
<div style = "font-size: 5vw; color: black; transform: translateY(100px);">&hellip;</div>
<div class = "backup-diagram">
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
	<div class = "down-arrow"></div>
	<div class = "action"></div>
</div>
<div class = "backup-diagram">
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

# ╔═╡ cf9fdc5d-a3f5-4c0d-8bf9-8d43fe152d5a
md"""
### $n$-step Sarsa Implementation
"""

# ╔═╡ aa9b62f2-4dc7-48bf-93e6-31a5b19b01f3
md"""
### Figure 7.4: 
Gridworld example of learning using $n$-step methods.  The left plot shows the path taken by the agent during the first episode of training.  All of the values are initially zero and there is a single reward of 1 for the terminal step only.  The light blue squares show values that were changed from zero for each of the three $n$-step methods.  The right is equivalent to Monte Carlo learning because the episode is shorter than the $n$ selected, and in this case every state along the path is updated.  For one-step Sarsa only the value of the cell adjacent to the goal is increased.
"""

# ╔═╡ 9b3d9429-dbe1-4363-9909-15106481b3d8
@bind run7_4 Button("New Path")

# ╔═╡ 1b12b915-4576-4c3d-8360-50eb9ad2392d
md"""
> ### *Exercise 7.4* 
> Prove that the *n*-step return of Sarsa (7.4) can be written exactly in terms of a novel TD error, as $G_{t:t+n}=Q_{t-1}(S_t, A_t) + \sum_{k=t}^{\text{min}(t+n,T)-1} \gamma^{k-t}[R_{k+1}+\gamma Q_k (S_{k+1}, A_{k+1}) - Q_{k-1}(S_k, A_k)]$

*n*-step return for Sarsa is:

$G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n Q_{t+n-1}(S_{t+n}, A_{t+n}), \; n \geq 1, 0 \leq t \lt T-n$

So we can see that:

$G_{k:k+1} \doteq R_{k+1} + \gamma Q_k(S_{k+1}, A_{k+1})$

which we can use to rewrite the novel expression as:

$\begin{flalign}
G_{t:t+n}&=Q_{t-1}(S_t, A_t) + \sum_{k=t}^{\text{min}(t+n,T)-1} \gamma^{k-t}[G_{k:k+1} - Q_{k-1}(S_k, A_k)] \\
&=Q_{t-1}(S_t, A_t) + G_{t:t+1} - Q_{t-1}(S_t, A_t) + \sum_{k=t+1}^{\text{min}(t+n,T)-1} \gamma^{k-t}[G_{k:k+1} - Q_{k-1}(S_k, A_k)] \\
&=G_{t:t+1} + \sum_{k=t+1}^{\text{min}(t+n,T)-1} \gamma^{k-t}[G_{k:k+1} - Q_{k-1}(S_k, A_k)] \\
&=R_{t+1} + \gamma Q_t(S_{t+1}, A_{t+1})+\sum_{k=t+1}^{\text{min}(t+n,T)-1} \gamma^{k-t}[G_{k:k+1} - Q_{k-1}(S_k, A_k)] \\
\end{flalign}$
"""

# ╔═╡ b0b56d97-3802-41f2-814d-8a3eb617a442
md"""
We can also define an $n$-step version of Expected Sarsa by writing the $n$-step return as 

$G_{t:t+n} \doteq R_{t+1} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n \overline{V}_{t_n-1}(S_{t+n}), \;\; t+n \lt T \tag{7.7}$

(with $G_{t:t+n} \doteq T_t \; for \; t+n \geq T$) where $\overline{V}_t(s)$ is the *expected approximate value* of state $s$, using the estimated action values at time $t$, under the target policy:

$\overline{V}_t(s) \doteq \sum_a \pi(a \vert s) Q_t(s, a), \;\; \forall s \in \mathcal{S} \tag{7.8}$

Expected approximate values are used in developing many of the action-value methods explored in the rest of the book.  If $s$ is terminal, then its expected approximate value is defined to be 0.
"""

# ╔═╡ 2180a3a3-08aa-4b0a-8734-52377966f661
md"""
### $n$-step Expected Sarsa for estimating $Q \approx q_*$ or $q_\pi$
"""

# ╔═╡ d1cdf977-abc0-4df3-bc15-287ce4b94fc3
md"""
Step Size $\alpha$: $(@bind ex7_4_α NumberField(0.1f0:0.1f0:1.0f0, default = 0.4))
"""

# ╔═╡ 6819a060-7e26-46cb-9c9d-5c4e3364b66a
md"""
## 7.3 $n$-step Off-policy Learning

Recall that off-policy learning is learning the value function for one policy, $\pi$, while following another policy $b$.  Often, $\pi$ is the greedy policy for the current action-value function estimate, and $b$ is a more exploratory policy, perhaps $\epsilon$-greedy.  In order to use data from $b$ we must take into account the difference between the two policies, using their relative probability of taking the actions that were taken.  In $n$-step methods, returns are constructed over $n$ steps, so we are interested in the relative probability of just those $n$ actions.  For example, to make a simple off-policy version of $n$-step TD, the update for time $t$ (actually made at time $t+n$) can simply be weighted by $\rho_{t:t+n-1}$:

$V_{t+n}(S_t) \doteq V_{t+n-1}(S_t) + \alpha \rho_{t:t+n-1} [G_{t:t+n} - V_{t+n-1}(S_t)], \;\; 0 \leq t \lt T \tag{7.9}$

where $\rho_{t:t+n-1}$, called the *importance sampling ratio*, is the relative probability under the two policies of taking the $n$ actions from $A_t$ to $A_{t+n-1}$:

$\rho_{t:h} \doteq \prod_{k=t}^{\min(h, T-1)} \frac{\pi(A_k \vert S_k)}{b(A_k \vert S_k)} \tag{7.10}$

For example, if any one of the actions owuld never be taken by $\pi$ then the $n$-step return should be given zero weight and be totally ignored.  On the other hand if by chance an action is taken that $\pi$ would take with much greater probability than $b$ does, then this will increase the weight that would otherwise be givevn to the return.  If the two policies are the same, then the importance sampling ratio is always 1.  Thus our new update (7.9) generalizes and can completely replace our earlier $n$-step TD update.  Similarly, our previous $n$-step Sarsa update can be completely replaced by a simple off-policy form: 

$Q_{t+n}(S_t,A_t) \doteq Q_{t+n-1}(S_t, A_t) + \alpha \rho_{t+1:t+n}[G_{t:t+n}-Q_{t+n-1}(S_t, A_t)] \tag{7.11}$

for $0 \leq t \lt T$.  Note that the importance sampling ratio here starts and ends one step later than for $n$-step TD (7.9).  This is because here we are updating a state-action pair.  We do not have to care how likely we were to select the action; now that we have selected it we want to learn fully from what happens, with importance sampling only for the subsequent actions.  A full implementation for state value estimation and sarsa based on off-policy learning is in the following sections.
"""

# ╔═╡ c9f3ae61-3ce5-4d24-9910-b31d18b14a7e
md"""
### Off-policy $n$-step TD for estimating $V \approx v_\pi$
"""

# ╔═╡ ba2d9b8b-573b-4771-9741-ec77272a37c2
md"""
### Example 7.2: Off-policy $n$-step TD Methods on Cliff Gridworld

Consider the same simple gridworld as in figure 7.4 where we seek to estimate the value function of a target policy using a behavior policy.  In this case the target policy simply steps to the right from the start while the behavior policy is random.  If we estimate the value of a policy that only steps right, we'd expect every state along the path to be values at 1 in the undiscounted scenario.  Below is an example of the path taken by the behavior policy as well as the value estimates from off-policy estimation.
"""

# ╔═╡ 71595857-a23d-45c1-b69c-eec1a22ceb25
md"""
Number of steps for n-step prediction
"""

# ╔═╡ 913f2c01-5964-4fc6-9a2b-ff35b5ae6bab
@bind n_eg_7_2 NumberField(1:10)

# ╔═╡ 5b1977be-2610-4eba-a953-364813504505
md"""
### Off-policy $n$-step Sarsa for estimating $Q \approx q_*$ or $q_\pi$
"""

# ╔═╡ 08e8be9a-e94d-4d44-ba5e-74aa3d69dc9c
md"""
With the plain gridworld, both on and off-policy sarsa can learn the optimal path for various values of n, although the learning rate typically must be lower for off-policy learning
"""

# ╔═╡ fe6c2450-8bfe-41ab-aa92-1d13c08973e0
md"""
*n*-step method: $(@bind eg_7_3_n NumberField(1:100, default = 5)) 
Training Episodes: $(@bind eg_7_3_ep NumberField(10:1000, default = 100))
"""

# ╔═╡ cc48d63a-93fb-4db8-9f82-21dc90c9ebfe
md"""
If we instead consider a cliffworld where the direct path to the goal is partially surrounded, then our ϵ-greedy policy from sarsa may need to avoid the direct path. Off-policy learning however should learn to take the direct path regardless of how dangerous it is because the optimal policy need not be stochastic.
"""

# ╔═╡ 17c52a18-33ae-47dd-aa43-07440c586b6c
md"""
## 7.4 *Per-decision Methods with Control Variates

For the *n* steps ending at horizon *h*, the *n*-step return can be written

$G_{t:h} = R_{t+1} + \gamma G_{t+1:h}, \;\; t<h<T \tag{7.12}$

where $G_{h:h} \doteq V_{h-1}(S_h)$ (Recall that this return is used at time *h*, previously denoted $t+n$).  Now consider the effect of following a behavior policy $b$ that is not the same as the target policy $\pi$.  All of the resulting experience, including the first reward $R_{t+1}$ and the next state $S_{t+1}$ must be weighted by the importance sampling ratio for time $t$, $\rho_t = \frac{\pi(A_t|S_t)}{b(A_t,S_t)}$.  One might be tempted to simply weight the righthand side of the above equation, but one can do better.  Suppose the action at time $t$ would never be selected by $\pi$, so that $\rho_t$ is zero.  Then a simple weighting would result in the $n$-step return being zero, which could result in high variance when it was used as a target.  Instead, in this more sophisticated approach, one uses an alternate, *off-policy* definition of the $n$-step return ending at horizon $h$, as 

$G_{t:h} \doteq \rho_t (R_{t+1}+\gamma G_{t+1:h}) + (1-\rho_t)V_{h-1}(S_t), \;\; t<h<T \tag{7.13}$

In this approach, if $\rho_t$ is zero, then instead of the target being zero and causing the estimate to shrink, the target is the same as the estimate and cuases no change.  The importance sampling ratio being zero means we should ignore the sample, so leaving the estimate unchanged seemed appropriate.  The second, additional term in (7.13) is called a *control variate* (for obscure reasons).  Notice that the control variate does not change the expected update; the importance sampling ratio has expected value one and is uncorrelated with the estimate, so the expected value of the control variate is zero.  Also note that the off-policy definition (7.13) is a strict generalization of the earlier on-policy definition of the $n$-step return (7.1), as the two are identical in the on-policy case, in which $\rho_t$ is always 1.

For a conventional $n$-step method, the learning rule to use in conjunction with (7.13) is the $n$-step TD update (7.2), which has no explicit importance ampling ratios other than those embedded in the return.
"""

# ╔═╡ baab474f-e491-4b73-8d08-afc4a3bacde5
md"""
> ### *Exercise 7.5* 
> Write the pseudocode for the off-policy state-value prediction algorithm described above.

Going off the new definition of the $n$-step return, we can expand it out per step:

$\begin{flalign}
G_{t:h} &\doteq \rho_t (R_{t+1}+\gamma G_{t+1:h}) + (1-\rho_t)V_{h-1}(S_t), \;\; t<h<T \\
&= \rho_t R_{t+1} + (1-\rho_t)V_{h-1}(S_t) + \rho_t \gamma G_{t+1:h}\\
&= \rho_t R_{t+1} + (1-\rho_t)V_{h-1}(S_t) + \rho_t \gamma [\rho_{t+1} R_{t+2} + (1-\rho_{t+1})V_{h-1}(S_{t+1}) + \rho_{t+1} \gamma G_{t+2:h}]\\
&= \rho_t [R_{t+1} + \gamma \rho_{t+1} R_{t+2}] + (1-\rho_t)V_{h-1}(S_t) + \rho_t \gamma (1-\rho_{t+1})V_{h-1}(S_{t+1}) + \rho_t \rho_{t+1} \gamma^2 G_{t+2:h}\\
&=\sum_{i=t}^{h-1} \left [ \rho_{t:i} \left ( \gamma^{i-t} R_{t+1} + \left ( \frac{1}{\rho_i}-1 \right )V_{h-1}(S_i) \right ) \right ] + \rho_{t:h-1} \gamma^h V_{h-1}(S_h)\\
\end{flalign}$

where $\rho_{t:h} \doteq \prod_{i=t}^{h} \rho_i$

See implementation below
	"""

# ╔═╡ fd9b3f70-bd00-4e30-8231-6ada24529585
md"""
For action value estimates using off-policy control variates, we can write it recursively as:

$\begin{flalign}
G_{t:h} &\doteq R_{t+1} + \gamma \left ( \rho_{t+1}G_{t+1:h} + \bar V_{h+1}(S_{t+1}) - \rho_{t+1}Q_{h-1}(S_{t+1}, A_{t+1}) \right ) \\
&= R_{t+1} + \gamma \rho_{t+1} \left ( G_{t+1:h} - Q_{h-1}(S_{t+1}, A_{t+1}) \right ) + \gamma \bar V_{h-1}(S_{t+1}), \;\; t<h \leq T \tag{7.14}
\end{flalign}$

If $h<T$, then the recursion ends with $G_{h:h} \doteq Q_{h-1}(S_h, A_h)$, whereas if $h \geq T$, the recursion ends with $G_{T-1:h} \doteq R_T$.  Thre resultant prediction algorithm (after combining with (7.5)) is analogous to Expected Sarsa.
"""

# ╔═╡ b8ae179f-3ab9-4d8d-a49d-5d4035af63fd
md"""
> ### *Exercise 7.6* 
> Prove that the control variate in the above equations does not change the expected value of the return

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

# ╔═╡ b70147e8-1053-4791-b885-cd3fa4dcb216
md"""
> ### * *Exercise 7.7*
> Write the pseudocode for the off-policy action-value prediction algorithm described immediately above.  Pay particular attention to the termination conditions for the recursion upon hitting the horizon or the end of episode.
"""

# ╔═╡ f710c428-ef84-4ae4-9fcc-7882e2a13acb
md"""
> ### *Exercise 7.8*
> Show that the general (off-policy) version of the $n$-step return (7.13) can still be written exactly and compactly as the sum of state-based TD errors (6.5) if the approximate state value function does not change.
"""

# ╔═╡ 3b655b4a-709e-40cd-ab5d-bab2ce7030e2
md"""
> ### *Exercise 7.9*
> Repeat the above exercise for the action version of the off-policy $n$-step return (7.14) and the Expected Sarsa TD error (the quantity in brackets in Equation 6.9).
"""

# ╔═╡ a23108c9-2710-41b1-9a0d-cbb343fe5fda
md"""
> ### *Exercise 7.10 (programming)*
> Devise a small off-policy prediction problem and use it to show that the off-policy learning algorithm using (7.13) and (7.2) is more data efficient than the simpler algorithm using (7.1) and (7.9).
"""

# ╔═╡ 36e56ddd-32ae-4948-b8b1-0d43b5ba2b75
md"""
Consider the simple gridworld from figure 7.4 and the prediction problem for the policy that always moves right.  In the undiscounted case the values should be one along the straight line from the start to the goal.  If we attempt to estimate this value function using a random behavior policy, we can compare the error during training between the method that uses control variates and the method that does not.  For the n = 1 case, there is no difference between the methods as expected, but as n gets larger, the method that does not use control variates needs to use a very small learning rate to avoid diverging.  The control variate method can regain a stable estimate even if the learning rate is too large.  On problems where the optimal n is greater than 1, this method should have an advantage.
"""

# ╔═╡ 5df8059d-3b24-4ccc-870d-c477467d5719
@bind ex_7_10_params confirm(PlutoUI.combine() do Child
	md"""
	Learning Rate $\alpha$: $(Child(:α, NumberField(0.01f0:0.01f0:1.0f0, default = 0.1f0)))
	n: $(Child(:n, NumberField(1:100, default = 2)))
	"""
end)

# ╔═╡ b7f932f7-f889-45e2-804d-fe6874296b65
md"""
The importance sampling that we have used in this section, the previous section, and in Chapter 5, enables sound off-policy learning, but also results in high variance updates, forcing the use of a small step-size parameter and thereby causing learning to be slow.  It is probably inevitable that off-policy training is slower than on-policy training -- after all, the data is less relevant to what is being learned.  However, it is probably also true that these methods can be improved on.  The control variates are one way of reducing the variance.  In the next section we consider an off-policy learning method that does not use importance sampling."""

# ╔═╡ 2a09d4e1-15aa-4acc-b8cf-763e5654baf9
md"""
## 7.5 Off-policy Learning Without Importance Sampling: The $n$-step Tree Backup Algorithm

In the previous Chapter, Q-learning was an example of off-policy learning without importance sampling.  To extend this idea to an $n$-step method we introduce the *tree-backup algorithm*.

Along a trajectory, the n steps of sample states, rewards, and actions are used, but we also consider the actions that were not selected at each state.  For the final state we use the expected update since there is no action selection.  The difference between this algorithm and the usual $n$-step algorithm is that the contribution for each step is weighted by the target policy probability of selecting that action.  Thus each first-level action $a$ contributes with a weight of $\pi(a\vert S_{t+1})$, except that the action actually taken, $A_{t+1}$, does not contribute at all.  Its probability, $\pi(A_{t+1}\vert S_{t+1})$ is used to weight all the second-level action values.  Thus each non-selected second-level action $a^\prime$ contributes with weight $\pi(A_{t+1}\vert S_{t+1})\pi(a^\prime\vert S_{t+2})$.  Each third-level action contributes with a weight $\pi(A_{t+1}\vert S_{t+1})\pi(A_{t+2}\vert S_{t+2})\pi(a^{\prime\prime}\vert S_{t+3})$, and so on.  It is as if each arrow to an action node in the diagram is weighted by the action's probability of being selected under the target policy and, if there is a tree below the action, then that weight applies to all the leaf nodes in the tree.

We can think of a $n$-step tree-backup update as consisting of $2n$ half-steps, alternating between sample half-steps from an action to a subsequent state, and expected half-steps considering from that state all possible actions with their probabilities of occurring under the policy.

What follows are detailed equations for the $n$-step tree-backup algorithm.  The one-step return (target) is the same as that of Expected Sarsa,

$G_{t:t+1} \doteq R_{t+1}+\gamma \sum_a \pi(a\vert S_{t+1})Q_t(S_{t+1}, a) \tag{7.15}$

for $t \lt T-1$, and the two-step tree-backup return is 

$\begin{flalign}
G_{t:t+2} &\doteq R_{t+1} + \gamma \sum_{a \neq A_{t+1}} \pi(a\vert S_{t+1})Q_t(S_{t+1}, a) \\ &+ \gamma \pi(A_{t+1}\vert S_{t+1})\left ( R_{t+2} + \gamma \sum_a \pi(a\vert S_{t+2})Q_{t+1}(S_{t+2}, a) \right ) \\
&= R_{t+1} + \gamma \sum_{a \neq A_{t+1}}\pi(a \vert S_{t+1}) Q_{t+1}(S_{t+1}, a) + \gamma \pi(A_{t+1}\vert S_{t+1})G_{t+1:t+2}
\end{flalign}$

for $t \lt T-2$.  The latter form suggests the general recursive definition of the tree-backup $n$-step return:

$G_{t:t+n} \doteq R_{t+1} + \gamma \sum_{a \neq A_{t+1}} \pi(a \vert S_{t+1})Q_{t+n-1}S_{t+1}, a) + \gamma \pi(A_{t+1}\vert S_{t+1})G_{t+1:t+n} \tag{7.16}$

for $t \lt T-1, n \geq 2$, with the $n=1$ case handled by (7.15) except for $G_{T-1:t+n} \doteq R_T$.  This target is then used with the usual action-value update rule from $n$-step Sarsa:

$Q_{t+n}(S_t, A_t) \doteq Q_{t+n-1}(S_t, A_t) + \alpha [G_{t:t+n} - Q_{t+n-1}(S_t, A_t)],$

for $0 \leq t \lt T$, while the values of all other state-action pairs remain unchanged:  $Q_{t+n}(s, a) = Q_{t+n-1}(s, a)$, for all $s, a$ such that $s \neq S_t$ or $a \neq A_t$.  Code implementing this algorithm is in the next section
"""

# ╔═╡ 602213a5-73dc-4e9d-af2e-541da7425f2f
md"""
> ### *Exercise 7.11*
> Show that if the approximate action values are unchanging, then the tree-backup return (7.16) can be written as a sum of expectation-based TD errors:

> $G_{t:t+n} = Q(S_t, A_t) + \sum_{k=t}^{\min{t+n-1, T-1}} \delta_k \prod_{i = t+1}^k \gamma \pi(A_i \vert S_i)$

> where $\delta_t \doteq R_{t+1} + \gamma \overline{V}_t(S_{t+1}) - Q(S_t, A_t)$ and $\overline{V}_t$ is given by (7.8).
"""

# ╔═╡ e35d1b11-f985-4593-8aab-5fdbaf23c316
md"""
### $n$-step Tree Backup for estimating $Q \approx q_{*}$ or $q_\pi$
"""

# ╔═╡ d0859458-777b-4756-a541-9ed31c8632a2
md"""
## 7.6 *A Unifying Algorithm: $n$-step $Q(\sigma)$

So far we have considered algorithms that use sampling to calculate updates from the trajectory generated by a behavior policy as well as using all of the transitions from a given state with the target policy to calculate an expected value.  One idea to unify these algorithms is to decide on a step-by-step basis whether one wants to take the action as a sample or to consider the expectation over actions instead.  We can consider a continuous variation between sampling and expectation specified by a parameter $\sigma_t \in [0, 1]$ with $\sigma=1$ denoting full sampling and $\sigma=0$ denoting pure expectation with no sampling.  The random variable $\sigma_t$ might be set as a function of the state, action or state-action pair at the time $t$.  We call this proposed new algorithm $n$-step $Q(\sigma)$.

To develop the equations for $n$-step $Q(\sigma)$, first write the $n$-step return (7.16) in terms of the horizon $h=t+n$ and then in terms of the expected approximate value $\overline{V}$ (7.8):

$\begin{flalign}
G_{t:h} &= R_{t+1} + \gamma \sum_{a \neq A_{t+1}} \pi(a \vert S_{t+1})Q_{h-1}(S_{t+1}, a) + \gamma \pi(A_{t+1} \vert S_{t+1})G_{t+1:h}\\
&=R_{t+1} + \gamma \overline{V}_{h-1} (S_{t+1}) - \gamma \pi(A_{t+1} \vert S_{t+1}) Q_{h-1}(S_{t+1}, A_{t+1}) + \gamma \pi(A_{t+1}\vert S_{t+1}) G_{t+1:h} \\
&=R_{t+1} + \gamma \pi(A_{t+1}\vert S_{t+1})(G_{t+1:h}  - Q_{h-1}(S_{t+1}, A_{t+1})) -  \gamma \overline{V}_{h-1} (S_{t+1}) \\
\end{flalign}$

after which it is exactly like the $n$-step return for Sarsa with control variates (7.14) except with the action probability $\pi(A_{t+1} \vert S_{t+1})$ substituted for the importance-sampling ratio $\rho_{t+1}$.  For $Q(\sigma)$, we slide linearly between these two cases:

$\begin{flalign}
G_{t:h} \doteq R_{t+1} &+ \gamma \left ( \sigma_{t+1}\rho_{t+1} + (1-\sigma_{t+1}) \pi(A_{t+1}\vert S_{t+1}) \right ) \left ( G_{t+1:h}  - Q_{h-1}(S_{t+1}, A_{t+1}) \right ) \\ &+ \gamma \overline{V}_{h-1} (S_{t+1}) \tag{7.17}
\end{flalign}$

for $t \lt h \leq T$.  The recursion ends with $G_{h:h} \doteq Q_{h-1}(S_h, A_h)$ if $h \lt T$, or with $G_{T-1:T} \doteq R_T$ if $h = T$.  Then we use the earlier update for $n$-step Sarsa without importance-sampling ratios (7.5) instead of (7.11), because now the ratios are incorporated in the $n$-step return.  Code implementing this algorithm is in the following section.
"""

# ╔═╡ a6e782f1-beaa-4673-99f9-0b8e6181e12c
md"""
### Off-policy $n$-step $Q(\sigma)$ for estimating $Q \approx q_*$ or $q_\pi$
"""

# ╔═╡ 2c8d25e2-30b5-41c6-aad6-55a46d83538f
iscliff_path(s) = s.x >= 4 && (s.y == 3 || s.y == 5)

# ╔═╡ fb39b4cc-c4fa-4824-b584-753832eca4d8
md"""
In this example, these is no safe path to the goal around the cliffs.  There is also a secondary terminal state directly up from the start that exits with a reward of 0.  The optimal policy is still to step to the goal in the straight line, but with any sarsa method, that won't be possible to learn due to the nature of the ϵ-greedy policy.  With all of the off policy methods, the true optimal policy can still be learned.

If the secondary goal has a negative reward instead of 0, then the n-step sarsa algorithm can fail to converge because the penalty for exiting at the secondary goal is similar to the penalty for failing to reach the primary goal.
"""

# ╔═╡ ec706721-a414-47c9-910e-9d58e77664ea
md"""
# Dependencies
"""

# ╔═╡ 5838ecbf-8982-4ab3-aa56-423b0e3d9563
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

# ╔═╡ 7b07b6ec-0428-495b-a99d-05290a968e06
function n_step_off_policy_TD_Vest(π::Matrix{X}, b::Matrix{X}, mdp::MDP_TD{S, A, F, E, H}, n::Integer, α::X, γ::X; num_episodes::Integer = 1000, vinit::X = zero(X), V::Vector{X} = initialize_state_value(mdp; vinit = vinit), save_states::Vector{S} = Vector{S}(), static_values = false) where {X <: AbstractFloat, S, A, F, E, H}
	check_policy(π, mdp)
	terminds = findall(mdp.isterm(s) for s in mdp.states)
	
	#initialize
	stateindexbuffer = MVector{n+1, Int64}(zeros(Int64, n+1))
	rewardbuffer = MVector{n+1, X}(zeros(X, n+1))
	ρbuffer = MVector{n+1, X}(zeros(X, n+1))
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
				(i_s, i_s′, r, s′, a, i_a) = takestep(mdp, b, s)
				s = s′
				i = mod(t+1, n+1) + 1
				stateindexbuffer[i] = i_s′
				rewardbuffer[i] = r
				ρbuffer[mod(t, n+1) + 1] = π[i_a, i_s] / b[i_a, i_s]
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
				# isinf(G) && @info "G is inf"
				ρ = one(X)
				for k = τ:min(τ+n-1, T-1)
					ρ *= get_value(ρbuffer, k)
				end
				# isnan(ρ) && @info "ρ is nan"
				i_τ = get_value(stateindexbuffer, τ)
				update_value = V[i_τ] + α*ρ*(G-V[i_τ])
				# isnan(update_value) && @info "update_value is nan"
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

# ╔═╡ a3254d1e-5bcd-49a6-a604-223ab221e419
function n_step_off_policy_TD_Vest_control_variate(π::Matrix{X}, b::Matrix{X}, mdp::MDP_TD{S, A, F, E, H}, n::Integer, α::X, γ::X; num_episodes::Integer = 1000, vinit::X = zero(X), V::Vector{X} = initialize_state_value(mdp; vinit = vinit), save_states::Vector{S} = Vector{S}(), static_values = false) where {X <: AbstractFloat, S, A, F, E, H}
	check_policy(π, mdp)
	terminds = findall(mdp.isterm(s) for s in mdp.states)
	
	#initialize
	stateindexbuffer = MVector{n+1, Int64}(zeros(Int64, n+1))
	rewardbuffer = MVector{n+1, X}(zeros(X, n+1))
	ρbuffer = MVector{n+1, X}(zeros(X, n+1))
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
				(i_s, i_s′, r, s′, a, i_a) = takestep(mdp, b, s)
				s = s′
				i = mod(t+1, n+1) + 1
				stateindexbuffer[i] = i_s′
				rewardbuffer[i] = r
				ρbuffer[mod(t, n+1) + 1] = π[i_a, i_s] / b[i_a, i_s]
				if mdp.isterm(s′)
					T = t + 1
				end
			end
			τ = t - n + 1
			if τ >= 0
				v_h = if τ+n < T
					V[get_state_index(τ+n)]
				else
					zero(X)
				end
				G = v_h
				for i in min(τ+n, T)-1:-1:τ
					ρ = get_value(ρbuffer, i)
					G = ρ*(get_reward(i+1) + γ*G) + (1 - ρ)*V[get_state_index(i)]
				end
				# isinf(G) && @info "G is inf"

				i_τ = get_value(stateindexbuffer, τ)
				update_value = V[i_τ] + α*(G-V[i_τ])
				# isnan(update_value) && @info "update_value is nan"
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
	sterm = s

	#note that the terminal state will not be added to the state list
	while !mdp.isterm(s) && (step <= max_steps)
		push!(states, s)
		(i_s, i_s′, r, s′, a, i_a) = takestep(mdp, π, s)
		push!(actions, a)
		push!(rewards, r)
		s = s′
		step += 1
		if mdp.isterm(s′)
			sterm = s′
		end
	end
	return states, actions, rewards, sterm
end

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

# ╔═╡ d7892cb7-c744-4dbe-89a4-c1878f275f47
function n_step_tree_backup(mdp::MDP_TD{S, A, F, E, H}, n::Integer, α::X, γ::X; num_episodes::Integer = 1000, qinit::X = zero(X), Qinit = initialize_state_action_value(mdp; qinit = qinit), πinit = create_greedy_policy(Qinit), history_state::S = first(mdp.states), update_policy! = (v, s) -> make_greedy_policy!(v), save_path = false, select_action_index = s -> rand(eachindex(mdp.actions))) where {X <: AbstractFloat, S, A, F, E, H}
	
	terminds = findall(mdp.isterm(s) for s in mdp.states)
	Q = copy(Qinit)
	Q[:, terminds] .= zero(X)
	π = copy(πinit)
	vhold = zeros(X, length(mdp.actions))
	rewards = zeros(X, num_episodes)
	steps = zeros(Int64, num_episodes)

	if save_path
		path = Vector{S}()
	end
	
	#initialize
	stateindexbuffer = MVector{n+1, Int64}(zeros(Int64, n+1))
	actionindexbuffer = MVector{n+1, Int64}(zeros(Int64, n+1))
	rewardbuffer = MVector{n+1, X}(zeros(X, n+1))
	get_state_index(i) = stateindexbuffer[mod(i, n+1) + 1]
	get_action_index(i) = actionindexbuffer[mod(i, n+1) + 1]
	get_reward(i) = rewardbuffer[mod(i, n+1) + 1]
	get_value(buffer, i) = buffer[mod(i, n+1) + 1]

	#simulate and episode and update the value function every step
	function runepisode!(Q, j)
		s = mdp.state_init()
		i_s = mdp.statelookup[s]
		T = typemax(Int64)
		t = 0
		τ = 0
		stateindexbuffer[1] = i_s
		i_a = select_action_index(s)
		actionindexbuffer[1] = i_a
		rtot = zero(X)
		while τ != T - 1
			if t < T
				i_s = get_state_index(t)
				s = mdp.states[i_s]
				if save_path && (j == num_episodes)
					push!(path, s)
				end
				i = mod(t+1, n+1) + 1
				i_a = get_action_index(t)
				a = mdp.actions[i_a]
				(r, s′) = mdp.step(s, a)
				rtot += r
				rewardbuffer[i] = r
				i_s′ = mdp.statelookup[s′]
				stateindexbuffer[i] = i_s′
				if mdp.isterm(s′)
					T = t + 1
				else
					i_a′ = select_action_index(s′)
					actionindexbuffer[i] = i_a′
				end
			end
			τ = t - n + 1
			if τ >= 0
				if t+1 >= T
					G = get_reward(T)
				else
					G = get_reward(t+1) + γ * sum(π[i, get_state_index(t+1)] * Q[i, get_state_index(t+1)] for i in eachindex(mdp.actions))
				end
					
				for i in min(t, T+1):-1:τ+1
					i_a = get_action_index(i)
					i_s = get_state_index(i)
					G += get_reward(i) + γ*π[i_a, i_s]*G
					G += γ*sum(π[a, i_s]*Q[a, i_s] for a in 1:i_a-1; init = zero(X))
					G += γ*sum(π[a, i_s]*Q[a, i_s] for a in i_a+1:length(mdp.actions); init = zero(X))
				end
			
				i_s_τ = get_value(stateindexbuffer, τ)
				i_a_τ = get_value(actionindexbuffer, τ)
				
				Q[i_a_τ, i_s_τ] += α*(G-Q[i_a_τ, i_s_τ])
				vhold .= Q[:, i_s_τ]
				update_policy!(vhold, mdp.states[i_s_τ])
				π[:, i_s_τ] .= vhold
			end
			t += 1
		end
		steps[j] = t
		rewards[j] = rtot
		return Q
	end
	for i = 1:num_episodes; runepisode!(Q, i); end
	default_return = Q, π, steps, rewards
	save_path && return (default_return..., path)
	return default_return
end

# ╔═╡ d26b08ff-fef7-4aca-a51c-58c91fa1555a
function n_step_off_policy_Qσ(mdp::MDP_TD{S, A, F, E, H}, n::Integer, α::X, γ::X; num_episodes::Integer = 1000, qinit::X = zero(X), Qinit = initialize_state_action_value(mdp; qinit = qinit), πinit = create_greedy_policy(Qinit), history_state::S = first(mdp.states), update_target_policy! = (v, s) -> make_greedy_policy!(v), binit = make_random_policy(mdp), update_behavior_policy! = (v, s) -> make_ϵ_greedy_policy!(v, one(X)/10), select_σ = (s, a, t) -> one(X)/2, save_path = false) where {X <: AbstractFloat, S, A, F, E, H}
	
	terminds = findall(mdp.isterm(s) for s in mdp.states)
	Q = copy(Qinit)
	Q[:, terminds] .= zero(X)
	π = copy(πinit)
	b = copy(binit)
	vhold = zeros(X, length(mdp.actions))
	rewards = zeros(X, num_episodes)
	steps = zeros(Int64, num_episodes)

	if save_path
		path = Vector{S}()
	end
	
	#initialize
	stateindexbuffer = MVector{n+1, Int64}(zeros(Int64, n+1))
	actionindexbuffer = MVector{n+1, Int64}(zeros(Int64, n+1))
	rewardbuffer = MVector{n+1, X}(zeros(X, n+1))
	σbuffer = MVector{n+1, X}(zeros(X, n+1))
	ρbuffer = MVector{n+1, X}(zeros(X, n+1))
	get_state_index(i) = stateindexbuffer[mod(i, n+1) + 1]
	get_action_index(i) = actionindexbuffer[mod(i, n+1) + 1]
	get_reward(i) = rewardbuffer[mod(i, n+1) + 1]
	get_σ(i) = get_value(σbuffer, i)
	get_ρ(i) = get_value(ρbuffer, i)
	get_value(buffer, i) = buffer[mod(i, n+1) + 1]

	#simulate and episode and update the value function every step
	function runepisode!(Q, j)
		s = mdp.state_init()
		i_s = mdp.statelookup[s]
		T = typemax(Int64)
		t = 0
		τ = 0
		stateindexbuffer[1] = i_s
		i_a = sample_action(b, i_s)
		actionindexbuffer[1] = i_a
		rtot = zero(X)
		while τ != T - 1
			if t < T
				i_s = get_state_index(t)
				s = mdp.states[i_s]
				if save_path && (j == num_episodes)
					push!(path, s)
				end
				i = mod(t+1, n+1) + 1
				i_a = get_action_index(t)
				a = mdp.actions[i_a]
				(r, s′) = mdp.step(s, a)
				rtot += r
				rewardbuffer[i] = r
				i_s′ = mdp.statelookup[s′]
				stateindexbuffer[i] = i_s′
				if mdp.isterm(s′)
					T = t + 1
				else
					i_a′ = sample_action(b, i_s′)
					σbuffer[i] = select_σ(s′, mdp.actions[i_a′], t+1)
					actionindexbuffer[i] = i_a′
					ρbuffer[i] = π[i_a′, i_s′] / b[i_a′, i_s′]
				end
			end
			τ = t - n + 1
			if τ >= 0
				if t+1 < T
					G = Q[get_action_index(t+1), get_state_index(t+1)]
				else
					G = zero(X)
				end
				for k = min(t+1, T):-1:τ+1
					if k == T
						G = get_reward(T)
					else
						k_s = get_state_index(k)
						k_a = get_action_index(k)
						v̄ = sum(π[a, k_s]*Q[a, k_s] for a in eachindex(mdp.actions))
						G = get_reward(k) + γ*(get_σ(k)*get_ρ(k) + (1 - get_σ(k))*π[k_a, k_s])*(G - Q[k_a, k_s]) + γ*v̄
					end
				end
				i_s_τ = get_value(stateindexbuffer, τ)
				i_a_τ = get_value(actionindexbuffer, τ)
				Q[i_a_τ, i_s_τ] += α*(G-Q[i_a_τ, i_s_τ])
				vhold .= Q[:, i_s_τ]
				update_target_policy!(vhold, mdp.states[i_s_τ])
				π[:, i_s_τ] .= vhold
				vhold .= Q[:, i_s_τ]
				update_behavior_policy!(vhold, mdp.states[i_s_τ])
				b[:, i_s_τ] .= vhold
			end
			t += 1
		end
		steps[j] = t
		rewards[j] = rtot
		return Q
	end
	for i = 1:num_episodes; runepisode!(Q, i); end
	default_return = Q, π, steps, rewards
	save_path && return (default_return..., path)
	return default_return
end

# ╔═╡ a9f377a9-f76d-42a9-be95-a6e3e802c79b
md"""
### Random Walk Environment
"""

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

# ╔═╡ ace391b4-ec40-4314-a624-a6b6d2c038db
function create_ϵ_greedy_policy(Q::Matrix{T}, ϵ::T; π = copy(Q), get_valid_inds = j -> 1:size(Q, 1)) where T<:Real
	vhold = zeros(T, size(Q, 1))
	for j in 1:size(Q, 2)
		vhold .= Q[:, j]
		make_ϵ_greedy_policy!(vhold, ϵ; valid_inds = get_valid_inds(j))
		π[:, j] .= vhold
	end
	return π
end

# ╔═╡ 39a181d4-dbde-4e1b-9ca0-389c11852465
function n_step_sarsa(mdp::MDP_TD{S, A, F, E, H}, n::Integer, α::X, γ::X; num_episodes::Integer = 1000, qinit::X = zero(X), ϵinit = one(X)/10, Qinit = initialize_state_action_value(mdp; qinit = qinit), πinit = create_ϵ_greedy_policy(Qinit, ϵinit), history_state::S = first(mdp.states), update_policy! = (v, ϵ, s) -> make_ϵ_greedy_policy!(v, ϵ), decay_ϵ = false, save_path = false) where {X <: AbstractFloat, S, A, F, E, H}
	
	terminds = findall(mdp.isterm(s) for s in mdp.states)
	Q = copy(Qinit)
	Q[:, terminds] .= zero(X)
	π = copy(πinit)
	vhold = zeros(X, length(mdp.actions))
	rewards = zeros(X, num_episodes)
	steps = zeros(Int64, num_episodes)

	if save_path
		path = Vector{S}()
	end
	
	#initialize
	stateindexbuffer = MVector{n+1, Int64}(zeros(Int64, n+1))
	actionindexbuffer = MVector{n+1, Int64}(zeros(Int64, n+1))
	rewardbuffer = MVector{n+1, X}(zeros(X, n+1))
	get_state_index(i) = stateindexbuffer[mod(i, n+1) + 1]
	get_action_index(i) = actionindexbuffer[mod(i, n+1) + 1]
	get_reward(i) = rewardbuffer[mod(i, n+1) + 1]
	get_value(buffer, i) = buffer[mod(i, n+1) + 1]

	#simulate and episode and update the value function every step
	function runepisode!(Q, j)
		ϵ = ϵinit / (1 + j*decay_ϵ)
		s = mdp.state_init()
		i_s = mdp.statelookup[s]
		T = typemax(Int64)
		t = 0
		τ = 0
		stateindexbuffer[1] = i_s
		i_a = sample_action(π, i_s)
		actionindexbuffer[1] = i_a
		rtot = zero(X)
		while τ != T - 1
			if t < T
				i_s = get_state_index(t)
				s = mdp.states[i_s]
				if save_path && (j == num_episodes)
					push!(path, s)
				end
				i = mod(t+1, n+1) + 1
				i_a = get_action_index(t)
				a = mdp.actions[i_a]
				(r, s′) = mdp.step(s, a)
				rtot += r
				rewardbuffer[i] = r
				i_s′ = mdp.statelookup[s′]
				stateindexbuffer[i] = i_s′
				if mdp.isterm(s′)
					T = t + 1
				else
					i_a′ = sample_action(π, i_s′)
					actionindexbuffer[i] = i_a′
				end
			end
			τ = t - n + 1
			if τ >= 0
				G = zero(X)
				for i in τ+1:min(τ+n, T)
					G += (γ^(i - τ - 1))*get_reward(i)
				end
				if τ+n < T
					G += γ^n * Q[get_action_index(τ+n), get_state_index(τ+n)]
				end
				i_s_τ = get_value(stateindexbuffer, τ)
				i_a_τ = get_value(actionindexbuffer, τ)
				Q[i_a_τ, i_s_τ] += α*(G-Q[i_a_τ, i_s_τ])
				vhold .= Q[:, i_s_τ]
				update_policy!(vhold, ϵ, mdp.states[i_s_τ])
				π[:, i_s_τ] .= vhold
			end
			t += 1
		end
		steps[j] = t
		rewards[j] = rtot
		return Q
	end
	for i = 1:num_episodes; runepisode!(Q, i); end
	default_return = Q, π, steps, rewards
	save_path && return (default_return..., path, last(path))
	return default_return
end

# ╔═╡ 8732e67e-3944-4a9f-bc69-1d8e2e0c2fa3
function n_step_expected_sarsa(mdp::MDP_TD{S, A, F, E, H}, n::Integer, α::X, γ::X; num_episodes::Integer = 1000, qinit::X = zero(X), ϵinit = one(X)/10, Qinit = initialize_state_action_value(mdp; qinit = qinit), πinit = create_ϵ_greedy_policy(Qinit, ϵinit), history_state::S = first(mdp.states), update_policy! = (v, ϵ, s) -> make_ϵ_greedy_policy!(v, ϵ), decay_ϵ = false, save_path = false) where {X <: AbstractFloat, S, A, F, E, H}
	
	terminds = findall(mdp.isterm(s) for s in mdp.states)
	Q = copy(Qinit)
	Q[:, terminds] .= zero(X)
	π = copy(πinit)
	vhold = zeros(X, length(mdp.actions))
	rewards = zeros(X, num_episodes)
	steps = zeros(Int64, num_episodes)

	if save_path
		path = Vector{S}()
	end
	
	#initialize
	stateindexbuffer = MVector{n+1, Int64}(zeros(Int64, n+1))
	actionindexbuffer = MVector{n+1, Int64}(zeros(Int64, n+1))
	rewardbuffer = MVector{n+1, X}(zeros(X, n+1))
	get_state_index(i) = stateindexbuffer[mod(i, n+1) + 1]
	get_action_index(i) = actionindexbuffer[mod(i, n+1) + 1]
	get_reward(i) = rewardbuffer[mod(i, n+1) + 1]
	get_value(buffer, i) = buffer[mod(i, n+1) + 1]

	#simulate and episode and update the value function every step
	function runepisode!(Q, j)
		ϵ = ϵinit / (1 + j*decay_ϵ)
		s = mdp.state_init()
		i_s = mdp.statelookup[s]
		T = typemax(Int64)
		t = 0
		τ = 0
		stateindexbuffer[1] = i_s
		i_a = sample_action(π, i_s)
		actionindexbuffer[1] = i_a
		rtot = zero(X)
		while τ != T - 1
			if t < T
				i_s = get_state_index(t)
				s = mdp.states[i_s]
				if save_path && (j == num_episodes)
					push!(path, s)
				end
				i = mod(t+1, n+1) + 1
				i_a = get_action_index(t)
				a = mdp.actions[i_a]
				(r, s′) = mdp.step(s, a)
				rtot += r
				rewardbuffer[i] = r
				i_s′ = mdp.statelookup[s′]
				stateindexbuffer[i] = i_s′
				if mdp.isterm(s′)
					T = t + 1
				else
					i_a′ = sample_action(π, i_s′)
					actionindexbuffer[i] = i_a′
				end
			end
			τ = t - n + 1
			if τ >= 0
				G = zero(X)
				for i in τ+1:min(τ+n, T)
					G += (γ^(i - τ - 1))*get_reward(i)
				end
				
				if τ+n < T
					i_s_n = get_state_index(τ+n)
					v̄ = sum(π[i, i_s_n]*Q[i, i_s_n] for i in eachindex(mdp.actions))
					G += γ^n * v̄
				end
				i_s_τ = get_value(stateindexbuffer, τ)
				i_a_τ = get_value(actionindexbuffer, τ)
				
				Q[i_a_τ, i_s_τ] += α*(G-Q[i_a_τ, i_s_τ])
				vhold .= Q[:, i_s_τ]
				update_policy!(vhold, ϵ, mdp.states[i_s_τ])
				π[:, i_s_τ] .= vhold
			end
			t += 1
		end
		steps[j] = t
		rewards[j] = rtot
		return Q
	end
	for i = 1:num_episodes; runepisode!(Q, i); end
	default_return = Q, π, steps, rewards
	save_path && return (default_return..., path)
	return default_return
end

# ╔═╡ ef9261a3-6bfd-4811-836c-a25ee781a756
function n_step_sarsa_off_policy(mdp::MDP_TD{S, A, F, E, H}, n::Integer, α::X, γ::X; num_episodes::Integer = 1000, qinit::X = zero(X), ϵinit = one(X)/10, Qinit = initialize_state_action_value(mdp; qinit = qinit), πinit = create_ϵ_greedy_policy(Qinit, ϵinit), history_state::S = first(mdp.states), update_target_policy! = (v, ϵ, s) -> make_greedy_policy!(v), decay_ϵ = false, save_path = false) where {X <: AbstractFloat, S, A, F, E, H}
	
	terminds = findall(mdp.isterm(s) for s in mdp.states)
	Q = copy(Qinit)
	Q[:, terminds] .= zero(X)
	π = copy(πinit)
	b = make_random_policy(mdp)
	vhold = zeros(X, length(mdp.actions))
	rewards = zeros(X, num_episodes)
	steps = zeros(Int64, num_episodes)

	if save_path
		path = Vector{S}()
	end
	
	#initialize
	stateindexbuffer = MVector{n+1, Int64}(zeros(Int64, n+1))
	actionindexbuffer = MVector{n+1, Int64}(zeros(Int64, n+1))
	rewardbuffer = MVector{n+1, X}(zeros(X, n+1))
	get_state_index(i) = stateindexbuffer[mod(i, n+1) + 1]
	get_action_index(i) = actionindexbuffer[mod(i, n+1) + 1]
	get_reward(i) = rewardbuffer[mod(i, n+1) + 1]
	get_value(buffer, i) = buffer[mod(i, n+1) + 1]

	#simulate and episode and update the value function every step
	function runepisode!(Q, j)
		ϵ = ϵinit / (1 + j*decay_ϵ)
		s = mdp.state_init()
		i_s = mdp.statelookup[s]
		T = typemax(Int64)
		t = 0
		τ = 0
		stateindexbuffer[1] = i_s
		i_a = sample_action(b, i_s)
		actionindexbuffer[1] = i_a
		rtot = zero(X)
		while τ != T - 1
			if t < T
				i_s = get_state_index(t)
				s = mdp.states[i_s]
				if save_path && (j == num_episodes)
					push!(path, s)
				end
				i = mod(t+1, n+1) + 1
				i_a = get_action_index(t)
				a = mdp.actions[i_a]
				(r, s′) = mdp.step(s, a)
				rtot += r
				rewardbuffer[i] = r
				i_s′ = mdp.statelookup[s′]
				stateindexbuffer[i] = i_s′
				if mdp.isterm(s′)
					T = t + 1
				else
					i_a′ = sample_action(b, i_s′)
					actionindexbuffer[i] = i_a′
				end
			end
			τ = t - n + 1
			if τ >= 0
				ρ = one(X)
				G = zero(X)
				t_end = min(τ+n, T-1)
				for i in τ+1:min(τ+n, T)
					if i <= t_end
						i_a = get_action_index(i)
						i_s = get_state_index(i)
						ρ *= π[i_a, i_s] / b[i_a, i_s]
					end
					G += (γ^(i - τ - 1))*get_reward(i)
				end
				if τ+n < T
					G += γ^n * Q[get_action_index(τ+n), get_state_index(τ+n)]
				end
				i_s_τ = get_value(stateindexbuffer, τ)
				i_a_τ = get_value(actionindexbuffer, τ)
				Q[i_a_τ, i_s_τ] += α*ρ*(G-Q[i_a_τ, i_s_τ])
				vhold .= Q[:, i_s_τ]
				update_target_policy!(vhold, ϵ, mdp.states[i_s_τ])
				π[:, i_s_τ] .= vhold
			end
			t += 1
		end
		steps[j] = t
		rewards[j] = rtot
		return Q
	end
	for i = 1:num_episodes; runepisode!(Q, i); end
	default_return = Q, π, steps, rewards
	save_path && return (default_return..., path)
	return default_return
end

# ╔═╡ 51834c9a-da99-41a5-9fcf-20790e741d53
md"""
### Gridworld Environment
"""

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

# ╔═╡ 53cbf192-e558-4dd7-83b2-79a4674240b5
function make_gridworld(;actions = rook_actions, sterm = GridworldState(8, 4), start = GridworldState(1, 4), xmax = 10, ymax = 7, stepreward = 0.0f0, termreward = 1.0f0, iscliff = s -> false, cliffreward = -100f0, goal2 = GridworldState(start.x, ymax), goal2reward = 0.0f0, usegoal2 = false)
	
	states = [GridworldState(x, y) for x in 1:xmax for y in 1:ymax]
	
	boundstate(x::Int64, y::Int64) = (clamp(x, 1, xmax), clamp(y, 1, ymax))
	
	function step(s::GridworldState, a::GridworldAction)
		(x, y) = move(a, s.x, s.y)
		GridworldState(boundstate(x, y)...)
	end
		
	function isterm(s::GridworldState) 
		s == sterm && return true
		usegoal2 && (s == goal2) && return true
		return false
	end

	function tr(s::GridworldState, a::GridworldAction) 
		isterm(s) && return (0f0, s)
		s′ = step(s, a)
		iscliff(s′) && return (cliffreward, start)
		x = Float32(isterm(s′))
		usegoal2 && (s′ == goal2) && return (goal2reward, goal2)
		r = (1f0 - x)*stepreward + x*termreward
		(r, s′)
	end	
	MDP_TD(states, actions, () -> start, tr, isterm)
end	

# ╔═╡ 57e9260d-f628-433b-92fe-f27e8d5294f5
const plain_gridworld = make_gridworld(sterm = GridworldState(7, 4), start = GridworldState(2, 4), xmax = 10, ymax = 8)

# ╔═╡ e016e721-62a1-4b58-b904-c95e8e898e0e
function compare_sarsa(n::Integer, α::AbstractFloat, γ::AbstractFloat, num_episodes::Integer)
	args = (plain_gridworld, n, α, γ)
	kwargs = (num_episodes = num_episodes, save_path = true)
	results = n_step_sarsa(args...; kwargs...)
	exp_results = n_step_expected_sarsa(args...; kwargs...)
	(results[3], exp_results[3])
end

# ╔═╡ 78888166-1b70-4a5f-829e-dddf7bed67ad
function compare_sarsa_trials(args...; trials = 100)
	1:trials |> Map(_ -> compare_sarsa(args...)) |> foldxt((a, b) -> (first(a) .+ first(b), last(a) .+ last(b))) |> a -> (mean(first(a) ./ trials), mean(last(a) ./ trials))
end

# ╔═╡ b99db42e-543b-42c1-8f08-f6ba1c81bff1
function compare_sarsa(nlist::AbstractVector{Int64}; α = 0.4f0, trials = 100)
	trials = [compare_sarsa_trials(n, 0.4f0, 0.999f0, 100) for n in nlist]
	sarsa_trace = scatter(x = nlist, y = [first(a) for a in trials], name = "Sarsa")
	expected_sarsa_trace = scatter(x = nlist, y = [last(a) for a in trials], name = "Expected Sarsa")
	plot([sarsa_trace, expected_sarsa_trace], Layout(xaxis_title = "n for n-step method", yaxis_title = "Average Number of Steps <br> Per Episode after <br> 100 Training Episodes", title = "Sarsa and Expected Sarsa Performance on Simple Gridworld with α = $α"))
end

# ╔═╡ dcae4bbe-263d-4c6e-8d10-8c30c97174af
compare_sarsa(1:5; α = ex7_4_α)

# ╔═╡ 4790d50d-fab8-43cf-bd0e-2f57f1fb5aef
tree_results = n_step_tree_backup(plain_gridworld, 5, 0.1f0, 0.9f0; num_episodes = 10)

# ╔═╡ 981376ba-3791-4ec4-b286-e7e5ec3637a4
qσ_results = n_step_off_policy_Qσ(plain_gridworld, 1, 0.2f0, 0.9f0; num_episodes = 100)

# ╔═╡ 992ef08d-6ba7-4267-8d97-b226fa241f34
const dangerous_cliffworld = make_gridworld(sterm = GridworldState(7, 4), start = GridworldState(2, 4), xmax = 10, ymax = 8, iscliff = s -> s.x >= 4 && s.x <= 6 && (s.y == 3 || s.y == 5))

# ╔═╡ eb6107b7-4914-4c20-9306-e0fe56119dbc
tree_results2 = n_step_tree_backup(dangerous_cliffworld, 1, 0.1f0, 0.9f0; num_episodes = 100)

# ╔═╡ c99b759f-42c4-4521-ba64-c6b85b735bde
const pathological_cliffworld = make_gridworld(;actions = [rook_actions; Stay()], sterm = GridworldState(7, 4), start = GridworldState(2, 4), xmax = 10, ymax = 8, iscliff = iscliff_path, usegoal2 = true, goal2reward = -1f0)

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

# ╔═╡ e2186e6f-f8d0-407a-b70b-19ef27741516
md"""
### Gridworld Visualization
"""

# ╔═╡ d7215445-fcb2-4828-a27d-a5a6b18fb068
function plot_path(mdp, states::Vector, sterm; title = "Optimal policy <br> path example", iscliff = s -> false)
	xmax = maximum([s.x for s in mdp.states])
	ymax = maximum([s.y for s in mdp.states])
	start = mdp.state_init()
	goal = mdp.states[findlast(mdp.isterm(s) for s in mdp.states)]
	start_trace = scatter(x = [start.x + 0.5], y = [start.y + 0.5], mode = "text", text = ["S"], textposition = "left", showlegend=false)
	finish_trace = scatter(x = [goal.x + .5], y = [goal.y + .5], mode = "text", text = ["G"], textposition = "left", showlegend=false)

	cliff_squares = filter(iscliff, mdp.states)
	if !isempty(cliff_squares)
		cliff_traces = [scatter(x = [s.x + 0.5], y = [s.y+0.5], mode = "text", text = ["C"], textposition = "left", showlegend = false) for s in cliff_squares]
	end
	
	path_traces = [scatter(x = [states[i].x + 0.5, states[i+1].x + 0.5], y = [states[i].y + 0.5, states[i+1].y + 0.5], line_color = "blue", mode = "lines", showlegend=false, name = "Optimal Path") for i in 1:length(states)-1]
	finalpath = scatter(x = [states[end].x + 0.5, sterm.x + .5], y = [states[end].y + 0.5, sterm.y + 0.5], line_color = "blue", mode = "lines", showlegend=false, name = "Optimal Path")

	h1 = 30*ymax
	traces = [start_trace; finish_trace; path_traces; finalpath]
	if !isempty(cliff_squares)
		cliff_traces = [scatter(x = [s.x + 0.5], y = [s.y+0.5], mode = "text", text = ["C"], textposition = "left", showlegend = false) for s in cliff_squares]
		traces = [traces; cliff_traces]
	end
	plot(traces, Layout(xaxis = attr(showgrid = true, showline = true, gridwith = 1, gridcolor = "black", zeroline = true, linecolor = "black", mirror=true, tickvals = 1:xmax, ticktext = fill("", 10), range = [1, xmax+1]), yaxis = attr(linecolor="black", mirror = true, gridcolor = "black", showgrid = true, gridwidth = 1, showline = true, tickvals = 1:ymax, ticktext = fill("", ymax), range = [1, ymax+1]), width = max(30*xmax, 200), height = max(h1, 200), autosize = false, padding=0, paper_bgcolor = "rgba(0, 0, 0, 0)", title = attr(text = title, font_size = 14, x = 0.5)))
end

# ╔═╡ db618d5c-63e2-4719-829f-f71405d2ced0
function plot_path(mdp, π::Matrix; kwargs...)
	(states, actions, rewards, sterm) = runepisode(mdp, π; max_steps = 100)
	plot_path(mdp, states, sterm; kwargs...)
end

# ╔═╡ 00e4e3c7-1e1f-4ab0-b670-b7f2fff8c056
plot_path(mdp; title = "Random policy <br> path example", kwargs...) = plot_path(mdp, make_random_policy(mdp); title = title, kwargs...)

# ╔═╡ e1a06901-55df-4687-9993-d4afa4321c36
function test_off_policy_n_step_sarsa(n; mdp = plain_gridworld, iscliff = s -> false, kwargs...)
	results1 = n_step_sarsa(mdp, n, 0.05f0, 0.9f0; kwargs...)
	results2 = n_step_sarsa_off_policy(mdp, n, 0.01f0, 0.9f0; kwargs...)
	π1 = create_greedy_policy(first(results1))
	π2 = create_greedy_policy(first(results2))
	[plot_path(mdp, π1, title = "$n-step Sarsa Path", iscliff = iscliff) plot_path(mdp, π2, title = "$n-step Off Policy Sarsa Path", iscliff = iscliff)]
end

# ╔═╡ 9ca471a9-f634-4988-9c8a-5fe97f66f3b5
test_off_policy_n_step_sarsa(eg_7_3_n;num_episodes = eg_7_3_ep)

# ╔═╡ 6e99c015-b4d4-4d64-ab6a-d00cd34de3ce
test_off_policy_n_step_sarsa(eg_7_3_n; mdp = dangerous_cliffworld, num_episodes = eg_7_3_ep, iscliff = s -> s.x >= 4 && s.x <= 6 && (s.y == 3 || s.y == 5))

# ╔═╡ f37d2508-9267-4a63-83de-7c6b3707a588
function ex_7_10(;max_episodes = 30, n = 1, α = 0.01f0, ntrials = 100)
	mdp = plain_gridworld
	π = mapreduce(s -> [0f0, 0f0, 0f0, 1f0], hcat, mdp.states)
	b = make_random_policy(mdp)
	π_path = plot_path(mdp, π)
	b_path = plot_path(mdp, b, title = "Random policy path example")

	save_states = [GridworldState(x, 4) for x = 1:6]
	# est_states = [plain_gridworld.statelookup[GridworldState(x, 4)] for x = 1:6]

	calcerr(out) = mean((last(out) .- 1) .^2, dims = 1)
	# αlist = LinRange(α_min, α_max, 3)
	# nlist = n_min:n_max
	
	est_err(f, α, n) = (1:ntrials |> Map(_ -> f(π, b, mdp, n, α, 1.0f0, num_episodes = max_episodes, save_states = save_states) |> calcerr) |> foldxt(+))[:] ./ ntrials

	err1 = est_err(n_step_off_policy_TD_Vest, α, n)
	err2 = est_err(n_step_off_policy_TD_Vest_control_variate, α, n)

	t1 = scatter(x = 1:max_episodes, y = err1, name = "Simple Off Policy Prediction")
	t2 = scatter(x = 1:max_episodes, y = err2, name = "Off Policy Prediction with Control Variates")
	plot([t1, t2], Layout(yaxis_range = [0, 2], xaxis_title = "Number of Episodes", yaxis_title = "Mean Squared Error<br> Averaged over $ntrials Trials", title = "α = $α, n = $n"))
	
	# traces1 = [scatter(x = nlist, y = [est_err(n_step_off_policy_TD_Vest, α, n) for n in nlist], name = "α = $α") for α in αlist]
	# traces2 = [scatter(x = nlist, y = [est_err(n_step_off_policy_TD_Vest_control_variate, α, n) for n in nlist], name = "α = $α") for α in αlist]

	# p1 = plot(traces1, Layout(yaxis_range = [0, 2], xaxis_title = "n", yaxis_title = "Mean Squared Error<br> After $max_episodes Episodes <br> Averaged over $ntrials Trials", title = "Simple Off Policy Prediction"))
	# p2 = plot(traces2, Layout(yaxis_range = [0, 2], xaxis_title = "n", title = "Off Policy Prediction with Control Variates"))
	# [p1 p2]
end

# ╔═╡ 952c1503-5933-4420-94f8-7556bde281ae
ex_7_10(;α = 0.2f0, n = 1)

# ╔═╡ 66b4a730-175d-45a2-a1d3-3870206d9e48
ex_7_10(; ex_7_10_params...)

# ╔═╡ e7be8cc5-c4b1-459f-8bc1-49c0a2633969
plot_path(plain_gridworld, tree_results[2])

# ╔═╡ cc086d13-87e8-496e-afe3-94004e879a39
plot_path(dangerous_cliffworld, tree_results2[2])

# ╔═╡ 5f52aa16-33c2-438f-8b5c-5d5a1ae2edd5
plot_path(plain_gridworld, qσ_results[2])

# ╔═╡ 08562519-6fe0-4be8-8e39-ce890b2ba410
function test_algorithm(mdp, algo, n, α, γ; iscliff = s -> false, kwargs...)
	results = algo(mdp, n, α, γ; kwargs...)
	plot_path(mdp, results[2]; iscliff = iscliff)
end

# ╔═╡ 3d55de86-e619-4ede-bc4b-61a572577872
test_algorithm(dangerous_cliffworld, n_step_off_policy_Qσ, 3, 0.1f0, 0.9f0; num_episodes = 100, select_σ = (a, b, c) -> 1f0, update_behavior_policy! = (v, s) -> make_ϵ_greedy_policy!(v, 0.9f0), iscliff = s -> s.x >= 4 && s.x <= 6 && (s.y == 3 || s.y == 5))

# ╔═╡ 935cd4f3-0ba2-49fc-a6e9-eb1e32cda84a
test_algorithm(dangerous_cliffworld, n_step_sarsa, 2, 0.3f0, 0.9f0; num_episodes = 1000, iscliff = s -> s.x >= 4 && s.x <= 6 && (s.y == 3 || s.y == 5))

# ╔═╡ bcf62294-020d-4bcf-8e74-6a259051480a
test_algorithm(pathological_cliffworld, n_step_off_policy_Qσ, 5, 0.01f0, 0.9f0; num_episodes = 1000, iscliff = iscliff_path, update_behavior_policy! = (v, s) -> make_ϵ_greedy_policy!(v, 0.5f0))

# ╔═╡ b4a2a4d8-417c-475b-909c-33767f8f3a1d
test_algorithm(pathological_cliffworld, n_step_sarsa, 5, 0.1f0, 0.9f0; num_episodes = 1000, iscliff = iscliff_path)

# ╔═╡ 57710048-d814-43ac-8396-cad6135279d8
function addelements(e1, e2)
	@htl("""
	$e1
	$e2
	""")
end

# ╔═╡ 14b269e1-fa81-4ed8-957e-119bff365d0e
function show_grid_value(mdp, Q, name; scale = 1.0, title = "", sigdigits = 2)
	width = maximum(s.x for s in mdp.states)
	height = maximum(s.y for s in mdp.states)
	start = mdp.state_init()
	termind = findfirst(mdp.isterm, mdp.states)
	sterm = mdp.states[termind]
	ngrid = width*height

	displayvalue(Q::Matrix, i) = round(maximum(Q[:, i]), sigdigits = sigdigits)
	displayvalue(V::Vector, i) = round(V[i], sigdigits = sigdigits)
	@htl("""
		<div style = "display: flex; transform: scale($scale); background-color: white; color: black; font-size: 16px;">
			<div>
				$title
				<div class = "gridworld $name value">
					$(HTML(mapreduce(i -> """<div class = "gridcell $name value" x = "$(mdp.states[i].x)" y = "$(mdp.states[i].y)" style = "grid-row: $(height - mdp.states[i].y + 1); grid-column: $(mdp.states[i].x); font-size: 12px; color: black; $(displayvalue(Q, i) != 0 ? "background-color: lightblue;" : "")">$(displayvalue(Q, i))</div>""", *, eachindex(mdp.states))))
				</div>
			</div>
		</div>
	
		<style>
			.$name.value.gridworld {
				display: grid;
				grid-template-columns: repeat($width, 20px);
				grid-template-rows: repeat($height, 20px);
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

# ╔═╡ 0c64afd6-ef23-4582-9250-2c1d4ae3cc43
function test_n_step_sarsa(;nlist = [1, 10, 1000])
	seed = rand(UInt64)
	sarsa_results = [begin
		Random.seed!(seed)
		n_step_sarsa(plain_gridworld, n, 0.5f0, 0.9999f0; num_episodes = 1, save_path = true)
	end
	for n in nlist]
		

	path = plot_path(plain_gridworld, first(sarsa_results)[end-1], first(sarsa_results)[end]; title = "Path Taken")
	valuegrids = [show_grid_value(plain_gridworld, first(a), "fig7_4"; title = @htl("Action Values Changed <br> By $(nlist[i])-step Sarsa"), sigdigits = 1, scale = 0.9) for (i, a) in enumerate(sarsa_results)]
	@htl("""
	<div style = "display: flex; justify-content: space-around; align-items: center; background-color: white; flex-wrap: wrap;">
	<div>$path</div>
	$(reduce(addelements, valuegrids))
	</div>
	""")
end

# ╔═╡ 8ac0f5e2-d2a2-4bc7-a4a0-910068947f90
begin
	run7_4
	test_n_step_sarsa()
end

# ╔═╡ 9239dfc8-206c-4651-97d9-2ec28bd1cd6a
function example_7_2(;num_episodes = 1000, n = 5)
	π = mapreduce(s -> [0f0, 0f0, 0f0, 1f0], hcat, plain_gridworld.states)
	b = make_random_policy(plain_gridworld)
	π_path = plot_path(plain_gridworld, π)
	b_path = plot_path(plain_gridworld, b, title = "Random policy path example")

	vπ_est = n_step_off_policy_TD_Vest(π, b, plain_gridworld, n, 0.01f0, 1.0f0, num_episodes = num_episodes)
	vπ_grid = show_grid_value(plain_gridworld, vπ_est[1], "ex7_2_a", sigdigits = 1, scale = 1.5)
	pathcompare = [π_path b_path]
	@htl("""
	<div style = "display: flex; flex-direction: column; align-items: center;">
	<div>$pathcompare</div>
	<div style = "margin: 50px;">$vπ_grid</div>
	</div>
	""")
end

# ╔═╡ b5b36dd8-e817-4377-af43-e077de659673
example_7_2(;n = n_eg_7_2)

# ╔═╡ 5220ffcf-1035-4f0f-b593-b4a0930908a6
function ex_7_5(;num_episodes = 1000, n = 5)
	π = mapreduce(s -> [0f0, 0f0, 0f0, 1f0], hcat, plain_gridworld.states)
	b = make_random_policy(plain_gridworld)
	π_path = plot_path(plain_gridworld, π)
	b_path = plot_path(plain_gridworld, b, title = "Random policy path example")

	vπ_est = n_step_off_policy_TD_Vest_control_variate(π, b, plain_gridworld, n, 0.1f0, 1.0f0, num_episodes = num_episodes)
	vπ_grid = show_grid_value(plain_gridworld, vπ_est[1], "ex7_2_a", sigdigits = 1, scale = 1.5)
	pathcompare = [π_path b_path]
	@htl("""
	<div style = "display: flex; flex-direction: column; align-items: center;">
	<div>$pathcompare</div>
	<div style = "margin: 50px;">$vπ_grid</div>
	</div>
	""")
end

# ╔═╡ 7ef516ce-ae52-4a90-b9b3-8282e035f59a
ex_7_5(;n = 7, num_episodes = 100)

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

# ╔═╡ 2023cd4e-6b1f-430e-b81a-da97a23b5ed5
const rook_action_display = @htl("""
<div style = "display: flex; flex-direction: column; align-items: center; justify-content: center; color: black; background-color: rgba(100, 100, 100, 0.1);">
	<div style = "display: flex; align-items: center; justify-content: center;">
	<div class = "downarrow" style = "transform: rotate(90deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(180deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(270deg);"></div>
	<div class = "downarrow" style = "position: absolute;"></div>
	</div>
	<div>Actions</div>
</div>
""")

# ╔═╡ df8a0a95-71e2-4b37-ae2c-dd8826a91369
HTML("""
<style>
	.downarrow {
		display: flex;
		justify-content: center;
		align-items: center;
		flex-direction: column;
	}

	.downarrow::before {
		content: '';
		width: 2px;
		height: 40px;
		background-color: black;
	}
	.downarrow::after {
		content: '';
		width: 0px;
		height: 0px;
		border-left: 5px solid transparent;
		border-right: 5px solid transparent;
		border-top: 10px solid black;
	}

	.gridcell {
			display: flex;
			justify-content: center;
			align-items: center;
			border: 1px solid black;
		}

	.windbox {
		height: 40px;
		width: 40px;
		display: flex;
		justify-content: center;
		align-items: center;
		transform: rotate(180deg);
		background-color: green;
	}

	.windbox * {
		background-color: green;
		color: green;
	}

	.windbox[w="0"] {
		opacity: 0.0; 
	}

	.windbox[w="1"] {
		opacity: 0.5;
	}

	.windbox[w="2"] {
		opacity: 1.0;
	}
</style>
""")

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
# ╟─71370f7c-4af7-43ae-aba3-da2d412c248a
# ╟─e528af0a-1af3-429a-b781-de2e8421b4e8
# ╟─cf9fdc5d-a3f5-4c0d-8bf9-8d43fe152d5a
# ╠═39a181d4-dbde-4e1b-9ca0-389c11852465
# ╠═57e9260d-f628-433b-92fe-f27e8d5294f5
# ╟─aa9b62f2-4dc7-48bf-93e6-31a5b19b01f3
# ╟─9b3d9429-dbe1-4363-9909-15106481b3d8
# ╟─8ac0f5e2-d2a2-4bc7-a4a0-910068947f90
# ╠═0c64afd6-ef23-4582-9250-2c1d4ae3cc43
# ╟─1b12b915-4576-4c3d-8360-50eb9ad2392d
# ╟─b0b56d97-3802-41f2-814d-8a3eb617a442
# ╟─2180a3a3-08aa-4b0a-8734-52377966f661
# ╠═8732e67e-3944-4a9f-bc69-1d8e2e0c2fa3
# ╠═e016e721-62a1-4b58-b904-c95e8e898e0e
# ╠═78888166-1b70-4a5f-829e-dddf7bed67ad
# ╠═b99db42e-543b-42c1-8f08-f6ba1c81bff1
# ╟─d1cdf977-abc0-4df3-bc15-287ce4b94fc3
# ╠═dcae4bbe-263d-4c6e-8d10-8c30c97174af
# ╟─6819a060-7e26-46cb-9c9d-5c4e3364b66a
# ╟─c9f3ae61-3ce5-4d24-9910-b31d18b14a7e
# ╠═7b07b6ec-0428-495b-a99d-05290a968e06
# ╟─ba2d9b8b-573b-4771-9741-ec77272a37c2
# ╟─71595857-a23d-45c1-b69c-eec1a22ceb25
# ╟─913f2c01-5964-4fc6-9a2b-ff35b5ae6bab
# ╟─b5b36dd8-e817-4377-af43-e077de659673
# ╠═9239dfc8-206c-4651-97d9-2ec28bd1cd6a
# ╟─5b1977be-2610-4eba-a953-364813504505
# ╠═ef9261a3-6bfd-4811-836c-a25ee781a756
# ╠═e1a06901-55df-4687-9993-d4afa4321c36
# ╟─08e8be9a-e94d-4d44-ba5e-74aa3d69dc9c
# ╟─fe6c2450-8bfe-41ab-aa92-1d13c08973e0
# ╟─9ca471a9-f634-4988-9c8a-5fe97f66f3b5
# ╟─cc48d63a-93fb-4db8-9f82-21dc90c9ebfe
# ╠═992ef08d-6ba7-4267-8d97-b226fa241f34
# ╟─6e99c015-b4d4-4d64-ab6a-d00cd34de3ce
# ╟─17c52a18-33ae-47dd-aa43-07440c586b6c
# ╟─baab474f-e491-4b73-8d08-afc4a3bacde5
# ╠═a3254d1e-5bcd-49a6-a604-223ab221e419
# ╠═5220ffcf-1035-4f0f-b593-b4a0930908a6
# ╠═7ef516ce-ae52-4a90-b9b3-8282e035f59a
# ╟─fd9b3f70-bd00-4e30-8231-6ada24529585
# ╟─b8ae179f-3ab9-4d8d-a49d-5d4035af63fd
# ╠═b70147e8-1053-4791-b885-cd3fa4dcb216
# ╠═f710c428-ef84-4ae4-9fcc-7882e2a13acb
# ╠═3b655b4a-709e-40cd-ab5d-bab2ce7030e2
# ╟─a23108c9-2710-41b1-9a0d-cbb343fe5fda
# ╟─36e56ddd-32ae-4948-b8b1-0d43b5ba2b75
# ╠═f37d2508-9267-4a63-83de-7c6b3707a588
# ╟─952c1503-5933-4420-94f8-7556bde281ae
# ╟─5df8059d-3b24-4ccc-870d-c477467d5719
# ╟─66b4a730-175d-45a2-a1d3-3870206d9e48
# ╟─b7f932f7-f889-45e2-804d-fe6874296b65
# ╟─2a09d4e1-15aa-4acc-b8cf-763e5654baf9
# ╟─602213a5-73dc-4e9d-af2e-541da7425f2f
# ╟─e35d1b11-f985-4593-8aab-5fdbaf23c316
# ╠═d7892cb7-c744-4dbe-89a4-c1878f275f47
# ╠═4790d50d-fab8-43cf-bd0e-2f57f1fb5aef
# ╠═e7be8cc5-c4b1-459f-8bc1-49c0a2633969
# ╠═eb6107b7-4914-4c20-9306-e0fe56119dbc
# ╠═cc086d13-87e8-496e-afe3-94004e879a39
# ╟─d0859458-777b-4756-a541-9ed31c8632a2
# ╟─a6e782f1-beaa-4673-99f9-0b8e6181e12c
# ╠═d26b08ff-fef7-4aca-a51c-58c91fa1555a
# ╠═981376ba-3791-4ec4-b286-e7e5ec3637a4
# ╠═5f52aa16-33c2-438f-8b5c-5d5a1ae2edd5
# ╠═08562519-6fe0-4be8-8e39-ce890b2ba410
# ╠═3d55de86-e619-4ede-bc4b-61a572577872
# ╠═935cd4f3-0ba2-49fc-a6e9-eb1e32cda84a
# ╠═2c8d25e2-30b5-41c6-aad6-55a46d83538f
# ╠═c99b759f-42c4-4521-ba64-c6b85b735bde
# ╟─fb39b4cc-c4fa-4824-b584-753832eca4d8
# ╠═bcf62294-020d-4bcf-8e74-6a259051480a
# ╠═b4a2a4d8-417c-475b-909c-33767f8f3a1d
# ╠═ec706721-a414-47c9-910e-9d58e77664ea
# ╠═0321b9d1-7d4e-4bf8-ac61-9c16ab6bc461
# ╠═5838ecbf-8982-4ab3-aa56-423b0e3d9563
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
# ╠═cd7cbdfb-ae59-4e8a-a4bf-c9dad96062c3
# ╠═993110fc-e3bc-46a8-a35b-32b15dac87d5
# ╠═751e8763-3274-4cf7-80ef-b544b8c46f4b
# ╠═52eb8101-6dd2-443e-9aff-979f2d2bb532
# ╠═631b2d23-584d-47bb-be24-7fa58be53dfa
# ╠═c5e01eea-20af-47b3-8dc4-681d3b01df8f
# ╠═795db9a8-71f4-4664-8aa1-dad112b2da12
# ╠═459539ca-387a-4e60-894b-94eb2906db42
# ╠═ea0b8273-e77c-482e-bc6c-f3e7bc7c7d46
# ╠═cf3418ed-af8b-4d86-8057-d1b1d22581c7
# ╠═9b2e64df-0341-4bd5-8484-3b7c1ef2c828
# ╟─a9f377a9-f76d-42a9-be95-a6e3e802c79b
# ╠═36b394dd-1956-40ab-9600-624e6900f665
# ╠═2e3b1699-0e35-43f6-800b-4788b5e3bc7d
# ╠═ace391b4-ec40-4314-a624-a6b6d2c038db
# ╠═51834c9a-da99-41a5-9fcf-20790e741d53
# ╠═eb4a7088-e7b3-4b18-a007-349694a49278
# ╠═8f77dd8e-7689-4f85-a990-58550c723920
# ╠═c17dc4eb-5a12-4313-b2e3-defa2be85295
# ╠═53cbf192-e558-4dd7-83b2-79a4674240b5
# ╠═294d55aa-5de7-4cb2-adf0-85af09fb2464
# ╠═e2186e6f-f8d0-407a-b70b-19ef27741516
# ╠═d7215445-fcb2-4828-a27d-a5a6b18fb068
# ╠═db618d5c-63e2-4719-829f-f71405d2ced0
# ╠═00e4e3c7-1e1f-4ab0-b670-b7f2fff8c056
# ╠═57710048-d814-43ac-8396-cad6135279d8
# ╠═14b269e1-fa81-4ed8-957e-119bff365d0e
# ╠═87ad13fa-604f-48f7-8232-a2c021b0fefd
# ╠═2023cd4e-6b1f-430e-b81a-da97a23b5ed5
# ╠═df8a0a95-71e2-4b37-ae2c-dd8826a91369
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
