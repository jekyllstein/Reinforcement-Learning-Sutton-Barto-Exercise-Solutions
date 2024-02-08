### A Pluto.jl notebook ###
# v0.19.36

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

# ╔═╡ 4a8d17a2-348b-4077-8071-708017daaf05
using BenchmarkTools

# ╔═╡ c6715072-a5a7-433f-90e1-7abbb221eb25
using PlutoProfile

# ╔═╡ 639840dc-976a-4e5c-987f-a92afb2d99d8
begin
	using StatsBase, Statistics, PlutoUI, HypertextLiteral, LaTeXStrings, PlutoPlotly, Base.Threads, LinearAlgebra
	TableOfContents()
end

# ╔═╡ 814d89be-cfdf-11ec-3295-49a8f302bbcf
md"""
# Chapter 6 Temporal-Difference Learning
TD methods combine the Monte Carlo concept of learning from experience with the self-consistency ideas from dynamic programming.  Unlike the pure Monte Carlo methods of Chapter 5, TD methods do not require waiting for the final outcome of an episode to start learning.  In other words they bootstrap learning by exploiting what is known about the properties of the value function.  Eventually we will see that different degrees of bootstrapping can be used that bridge the gap between the techniques in Chapter 5 and 6.
## 6.1 TD Prediction
"""

# ╔═╡ 495f5606-0567-47ad-a266-d21320eecfc6
md"""
Monte Carlo nonstationary update rule for value function

$V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)] \tag{6.1}$

where $G_t$ is the actual return following time $t$, and $\alpha$ is a constant step-size parameter.  Call this method *constant-α MC*.  The use of a constant step size α instead of the usual sample average is what makes this estiamtion method suitable for non-stationary problems.  Because the value $G_t$ is required, this method requires waiting for the final results from the end of an episode.

In contrast, TD methods need only wait for results from the following timestep to perform an update.  The following is the simplest TD method update rule:

$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)] \tag{6.2}$

where the update can be made immediately on transition to $S_{t+1}$ after receiving $R_{t+1}$.  This TD method is called $TD(0)$, or *one-step TD*.  See below for code implementing this.
"""

# ╔═╡ 410abe1d-04a6-4434-9abf-0d29dd6498e6
md"""
### Tabular TD(0) Implementation
"""

# ╔═╡ 7a5ff8f7-70d4-46f1-a4a7-bbfcec4f6e3f
function sample_action(π::Matrix{T}, i_s::Integer) where T<:AbstractFloat
	(n, m) = size(π)
	sample(1:n, weights(π[:, i_s]))
end

# ╔═╡ 834e5810-77ea-4dfd-9f37-9d9dbf6585a4
makelookup(v::Vector) = Dict(x => i for (i, x) in enumerate(v))

# ╔═╡ 3e767962-7339-4f35-a039-b5521a098ed5
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

# ╔═╡ 8e34202a-f841-4464-9017-cd50194f7987
function make_random_policy(mdp::MDP_TD; init::T = 1.0f0) where T <: AbstractFloat
	ones(T, length(mdp.actions), length(mdp.states)) ./ length(mdp.actions)
end

# ╔═╡ 401831c3-3925-465c-a093-28686f0dad2e
initialize_state_value(mdp::MDP_TD; vinit::T = 0.0f0) where T<:AbstractFloat = ones(T, length(mdp.states)) .* vinit

# ╔═╡ c5d32889-634b-4b00-8ba7-0d1ecaf94f05
initialize_state_action_value(mdp::MDP_TD; qinit::T = 0.0f0) where T<:AbstractFloat = ones(T, length(mdp.actions), length(mdp.states)) .* qinit

# ╔═╡ 24a441c8-7aaf-4642-b245-5e1201456d67
function check_policy(π::Matrix{T}, mdp::MDP_TD) where {T <: AbstractFloat}
#checks to make sure that a policy is defined over the same space as an MDP
	(n, m) = size(π)
	num_actions = length(mdp.actions)
	num_states = length(mdp.states)
	@assert n == num_actions "The policy distribution length $n does not match the number of actions in the mdp of $(num_actions)"
	@assert m == num_states "The policy is defined over $m states which does not match the mdp state count of $num_states"
	return nothing
end

# ╔═╡ d5abd922-a8c2-4f5c-9a6e-d2490a8ad7dc
#take a step in the environment from state s using policy π
function takestep(mdp::MDP_TD{S, A, F, G, H}, π::Matrix{T}, s::S) where {S, A, F<:Function, G<:Function, H<:Function, T<:Real}
	i_s = mdp.statelookup[s]
	i_a = sample_action(π, i_s)
	a = mdp.actions[i_a]
	(r, s′) = mdp.step(s, a)
	i_s′ = mdp.statelookup[s′]
	return (i_s, i_s′, r, s′, a, i_a)
end

# ╔═╡ bfe71b40-3157-47df-8494-67f8eb8e4e93
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

# ╔═╡ 7035c082-6e50-4df5-919f-5f09d2011b4a
runepisode(mdp::MDP_TD; kwargs...) = runepisode(mdp, make_random_policy(mdp); kwargs...)

# ╔═╡ eb735ead-978b-409c-8990-b5fa7a027ebf
function tabular_TD0_pred_V(π::Matrix{T}, mdp::MDP_TD{S, A, F, G, H}, α::T, γ::T; num_episodes::Integer = 1000, vinit::T = zero(T), V::Vector{T} = initialize_state_value(mdp; vinit = vinit), save_states::Vector{S} = Vector{S}()) where {T <: AbstractFloat, S, A, F, G, H}
	check_policy(π, mdp)
	terminds = findall(mdp.isterm(s) for s in mdp.states)
	
	#initialize
	counts = zeros(Integer, length(mdp.states))
	V[terminds] .= zero(T) #terminal state must always have 0 value
	v_saves = zeros(T, length(save_states), num_episodes+1)
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
		while !mdp.isterm(s)
			(i_s, i_s′, r, s′, a, i_a) = takestep(mdp, π, s)
			V[i_s] += α * (r + γ*V[i_s′] - V[i_s])
			s = s′
		end
		updatesaves!(j+1)
		return V
	end

	for i = 1:num_episodes;	runepisode!(V, i); end
	
	return V, v_saves
end

# ╔═╡ 415ea466-2038-48fe-9d24-39a90182f1eb
function monte_carlo_pred_V(π::Matrix{T}, mdp::MDP_TD{S, A, F, G, H}, α::T, γ::T; num_episodes::Integer = 1000, vinit::T = zero(T), V::Vector{T} = initialize_state_value(mdp; vinit=vinit), save_states = Vector{S}()) where {T <: AbstractFloat, S, A, F, G, H}
	
	check_policy(π, mdp)

	terminds = findall(mdp.isterm(s) for s in mdp.states)
	
	V[terminds] .= zero(T) #terminal state must always have 0 value
	v_saves = zeros(T, length(save_states), num_episodes+1)
	function updatesaves!(ep)
		for (i, s) in enumerate(save_states)
			i_s = mdp.statelookup[s]
			v_saves[i, ep] = V[i_s]
		end
	end
	updatesaves!(1)

	#there's no check here so this is equivalent to every-visit estimation
	function updateV!(states, actions, rewards; t = length(states), g = zero(T))		
		t = length(states)
		g = zero(T)
		for t = length(states):-1:1
			#accumulate future discounted returns
			g = γ*g + rewards[t]
			i_s = mdp.statelookup[states[t]]
			i_a = mdp.actionlookup[actions[t]]
			V[i_s] += α*(g - V[i_s]) #update running average of V
		end
	end

	for j in 1:num_episodes
		(states, actions, rewards) = runepisode(mdp, π)
	
		#update value function for each trajectory
		updateV!(states, actions, rewards)
		updatesaves!(j+1)
	end
	return V, v_saves
end

# ╔═╡ a0d2333f-e87b-4981-bb52-d436ec6481c1
md"""
Because TD(0) bases its update in part on an existing estimate, we say that it is a *bootstrapping* method, like DP.  We know from Chapter 3 that

$\begin{flalign}
	v_\pi & \doteq \mathbb{E}_\pi[G_t \mid S_t = s] \tag{6.3}\\
	&= \mathbb{E}[R_{t+1} + \gamma G_{t+1} \mid S_t = s] \tag{from (3.9)}\\
	&=\mathbb{E}[R_{t+1} + \gamma v_\pi (S_{t+1}) \mid S_t = s] \tag{6.4}
\end{flalign}$

Roughly speaking, Monte Carlo methods use an estimate of (6.3) as a target whereas DP methods use an estiamte of (6.4) as a target.  The Monte Carlo target is an estimate because the exepcted value in (6.3) is not known; a sample return is used in place of the real expected return.  The DP target is an estimate not because of the expected values, which are assumed to be completely provided by a model of the environment, but because $v_\pi(S_{t+1})$ is not known and the current estimate, $V(S_{t+1})$, is used isntead.  The TD target is an estimate for both reasons; it samples the expected values in (6.4) *and* it uses the current estimate $V$ instead of the true $v_\pi$.  Thus, TD methods combine the sampling of Monte Carlo with the bootstrapping of DP.

TD and Monte Carlo updates are both refered to as *sample updates* because they involve looking ahead to a sample successsor state (or state-action pair).  *Expected updates* used in DP methods use the complete distribution of all possible successor states rather than a single sample.

Note that the quantity in the brakets in (6.2) is a sort of error, measuring the difference between the estimated value of $S_t$ and the better estimate $R_{t+1} + \gamma V(S_{t+1})$.  This quantity is called the *TD error*:

$\delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \tag{6.5}$

The TD error depends on the subsequent state so it is not available until one step later.  That is to say $\delta_t$ is not known until time $t+1$.  Also note that if we do not update $V$ during an episode (as we do not in Monte Carlo methods), then the Monte Carlo error can be written as the sum of TD errors:

$\begin{flalign}
G_t - V(S_t) &= R_{t+1} + \gamma G_{t+1} - V(S_t) + \gamma V(S_{t+1}) - \gamma V(S_{t+1}) \tag{from (3.9)} \\
&=\delta_t + \gamma(G_{t+1} - V(S_{t+1})) \tag{a}\\
&=\delta_t + \gamma \left ( \delta_{t+1} + \gamma(G_{t+2} - V(S_{t+2})) \right ) \tag{using (a)}\\
&=\delta_t + \gamma \delta_{t+1} + \gamma^2 \left ( G_{t+2} - V(S_{t+2}) \right ) \\
&=\delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2} + \cdots + \gamma^{T-t-1}\delta_{T-1} + \gamma^{T-t}(G_T - V(S_T)) \tag{applying (a) until terination}\\
&=\delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2} + \cdots + \gamma^{T-t-1}\delta_{T-1} + \gamma^{T-t}(0-0) \tag{definition of terminal state}\\
&=\sum_{k=t}^{T-1} \gamma^{k-t} \delta_k \tag{6.6}
\end{flalign}$

This identity is not exact if $V$ is updated during the episode (as it is in TD(0)), but if the step size is small then it may still hold approximately.
"""

# ╔═╡ 3b16cbb7-f859-4871-9a63-8b40eb4191be
md"""
> ### *Exercise 6.1* 
> If $V$ changes during the episode, then (6.6) only holds approximately; what would the difference be between the two sides?  Let $V_t$ denote the array of state values used at time $t$ in the TD error (6.5) and in the TD update (6.2).  Redo the derivation above to determine the additional amount that must be added to the sum of TD errors in order to equal the Monte Carlo error.
"""

# ╔═╡ d4e39164-9833-4deb-84ca-22f49a1c33d8
md"""
Reference equations:

$\begin{flalign}
V(S_t) &\leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)] \tag{6.2} \\
\delta_t &\doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \tag{6.5}
\end{flalign}$

Re-write equation (6.5) using the values known at time t.  $V_t$ means the value function estimate at time $t$.

$\delta_t \doteq R_{t+1} + \gamma V_t(S_{t+1}) - V_t(S_t)$

Now equation (6.2) becomes

$V_{t+1}(S_t) = V_t(S_t) + \alpha \delta_t$
"""

# ╔═╡ c93ed1f2-3c38-4f68-8bf8-2cdf4e7bee34
md"""
Now we can rewrite the Monte Carlo error using (3.9) again and proceed with the derivation keeping track of the time index of the value estiamtes:

$\begin{flalign}
G_t - V_t(S_t) &= R_{t+1} + \gamma G_{t+1} - V_t(S_t) + \gamma V_{t}(S_{t+1}) - \gamma V_{t}(S_{t+1}) \tag{from (3.9)}\\
&= \delta_t + \gamma \left [ G_{t+1} - V_t(S_{t+1}) \right ] \\
&= \delta_t + \gamma \left [ G_{t+1} -  V_{t+1}(S_{t+1}) + V_{t+1}(S_{t+1}) - V_t(S_{t+1}) \right ] \\
\end{flalign}$

Define the following

$\eta_{t} \doteq V_{t+1}(S_{t+1}) - V_t(S_{t+1})$ 

which let's us re-write the equation

$G_t - V_t(S_t) = \delta_t + \gamma \eta_{t} + \gamma \left [ G_{t+1} - V_{t+1}(S_{t+1})\right ]$

Notice that the term in the brakets is equivalent to the left hand side but shifted forward one time step.  That implies the equation can be expanded recursively as we did with the original derivation.
"""

# ╔═╡ 1e3b3234-3fe1-46c9-82b7-f729c656eb25
md"""
$\begin{flalign}
G_t - V_t(S_t) &= \delta_t + \gamma \eta_{t} + \gamma \left [\delta_{t+1} + \gamma \eta_{t+1} +  \gamma (G_{t+2} - V_{t+2}(S_{t+2}) ) \right ] \\
&= \delta_t + \gamma \eta_{t} + \gamma \delta_{t+1}  + \gamma^2 \eta_{t+1} + \gamma^2 \left [G_{t+2} - V_{t+2}(S_{t+2}) \right ] \\
&= (\delta_t + \gamma \eta_t) + \gamma (\delta_{t+1} + \gamma \eta_{t+1}) + \cdots + \gamma^{T-t-1}(\delta_{T-1} + \gamma \eta_{T-1}) + \gamma^{T-t} \left [G_T - V_T(S_T) \right ]\\
&= (\delta_t + \gamma \eta_t) + \gamma (\delta_{t+1} + \gamma \eta_{t+1}) + \cdots + \gamma^{T-t-1}(\delta_{T-1} + \gamma \eta_{T-1})\\
&=\sum_{k=t}^{T-1} \gamma^{k-t} (\delta_k + \gamma \eta_k)\\
\end{flalign}$
"""

# ╔═╡ c09530bc-f37e-4d57-a267-14d4027147da
md"""
Returning to the definition of $\eta_t$, we can simplify further:

$\eta_{t} \doteq V_{t+1}(S_{t+1}) - V_t(S_{t+1})$

This quantity is the change in value estimate at a state between two time steps.  Note that at time $t+1$ we have only performed an update for the value at state $S_t$ using the equation:

$V_{t+1}(S_t) = V_t(S_t) + \alpha \delta_t$

If $S_{t+1} \neq S_t$, then the value estimate at this state will not occur on either time step $t$ or $t+1$, so $V_{t+1}(S_{t+1}) = V_t(S_{t+1}) \implies \eta_{t} = 0$

The only case in which $V_{t+1}(S_{t+1}) \neq V_t(S_{t+1})$ is when $S_t = S_{t+1} = S$.  In this case, $V_{t+1}(S) = V_t(S) + \alpha \delta_t \implies V_{t+1}(S) - V_t(S) = \alpha \delta_t$ 

So we can rewrite $\eta_{t} = \alpha \delta_t \mathbb{1}_{t}$ where $\mathbb{1}_{t} = \begin{cases} 1 & \text{if } S_{t+1} = S_t \\ 0 & \text{otherwise} \end{cases}$ 

So the original equation can be written as:

$\begin{flalign}
G_t - V_t(S_t) &= \sum_{k=t}^{T-1} \gamma^{k-t} (\delta_k + \gamma \alpha \delta_k \mathbb{1}_k) \\
&= \sum_{k=t}^{T-1} \gamma^{k-t} \delta_k (1 + \gamma \alpha \mathbb{1}_k) \\
\end{flalign}$

Where the first term is the value from the original derivation and the second term is only non-zero when a state appears twice concecutively in an episode.
"""

# ╔═╡ b5187232-d808-49b6-9f7e-a4cbeb6c2b3e
md"""
### Example 6.1: Driving Home
"""

# ╔═╡ 5f32fed0-c921-4cbb-85fe-ade54d4c6c95
md"""
At each state or checkpoint you try to predict how much longer it will take to get home using any information that is relevant.  Notice that regardless of how inaccurate we were on previous steps, we can still make an accurate prediction for the time to go.

|State|Elapsed Time (minutes)|Predicted Time to Go|Predicted Total Time|
|---|---|---|---|
|leaving office, friday at 6|0|30|30|
|reach car, raining|5|35|40|
|exiting highway|20|15|35|
|2ndary road, behind truck|30|10|40|
|entering home street|40|3|43|
|arriving home|43|0|43|

The rewards in this example are the elapsed times on each leg of the journey and there is no discounting, thus the return for each state is the actual time to go from that state.  The value of each state is the *expected* time to go.  The second column of numbers gives the current estimated value for the state encountered.

A simple way to view the operation of Mone Carlo methods is to plot hte predicted total time (the last column) over the sequence.  For each state we would compare that value with the actual elapsed time which was 43 minutes.
"""

# ╔═╡ bc8bad61-a49a-47d6-8fa6-7dcf6c221910
function example_6_1(;elapsed = [0, 5, 20, 30, 40, 43], predicted_ttg = [30, 35, 15, 10, 3, 0])
	states = [:leaving, :reach_car, :exit_highway, :snd_rd, :home_st, :arrive]
	tt = last(elapsed)
	predicted_tt = predicted_ttg .+ elapsed
	actual_tt = fill(tt, 6)

	t1 = scatter(x = states, y = predicted_tt, line_color = "black", name = "actual outcome")
	t1′ = scatter(x = states, y = predicted_tt, line_color = "black", name = "actual outcome", showlegend=false)
	t2 = scatter(x = states, y = actual_tt, mode = "lines", line = attr(dash = "dash", color = "black"), name = "Monte Carlo Prediction")
	errortraces = [scatter(x = [s, s], y = [e, tt], line = attr(color = "red"), marker = attr(symbol = "arrow-bar-up", angleref = "previous"), showlegend = false, name = "Mone Carlo Error") for (s, e) in zip(states, predicted_tt)]
	p1 = plot([t1; t2; errortraces], Layout(xaxis_title = "State", yaxis_title = "Predicted total <br> travel time", xaxis_ticktext = ["leaving office", "reach car", "exiting highway", "2ndary road", "home street", "arrive home"], xaxis_tickvals = states, width = 600, legend_orientation = "h", legend_y = 1.1))

	td_prediction = [predicted_tt[2:end]; tt]
	t3 = scatter(x = states, y = td_prediction, name = "TD(0) Prediction", mode = "lines", line = attr(dash = "dash", color = "black", shape = "hv"))
	tderrors = [scatter(x = [states[i], states[i]], y = [predicted_tt[i], td_prediction[i]], line = attr(color = "red"), marker = attr(symbol = "arrow-bar-up", angleref = "previous"), showlegend = false, name = "TD(0) Error") for i in eachindex(states)]
	p2 = plot([t1′; t3; tderrors], Layout(xaxis_title = "State", xaxis_ticktext = ["leaving office", "reach car", "exiting highway", "2ndary road", "home street", "arrive home"], xaxis_tickvals = states, width = 600, showlegend = false))

	[p1 p2]
	# plot(predicted_tt, xticks = (1:6, String.(states)), ylabel = "Minutes", lab = "Preicted Outcome", size = (680, 400))
	# plot!(fill(43, 6), line = :dot, lab = "actual outcome")
end

# ╔═╡ 6edb550d-5c9f-4ea6-8746-6632806df11e
example_6_1()

# ╔═╡ 0f22e85f-ed31-49df-a7c7-0579298f05fe
md"""
For Monte Carlo learning each state estimate is updated with the error shown by the red arrows only after the episode is finished.  For TD(0) learning, as soon as the feedback from the subsequent state is received, the error can be calculated and it is only based on the new information from one state into the future.  
"""

# ╔═╡ 9017093c-a9c3-40ea-a9c6-881ee62fc379
md"""
> ### *Exercise 6.2* 
> This is an exercise to help develop your intuition about why TD methods are often more efficient than Monte Carlo methods.  Consider the driving home example and how it is addressed by TD and Monte Carlo methods.  Can you imagine a scenario in which a TD update would be better on average than a Monte Carlo update?  Give an example scenario - a description of past experience and a current state - in which you would expect the TD update to be better.  Here's a hint: Suppose you have lots of experience driving home from work.  Then you move to a new building and a new parking lot (but you still enter the highway at the same place).  Now you are starting to learn predictions for the new building.  Can you see why TD updates are likely to be much better, at least initially, in this case?  Might the same sort of thing happen in the original scenario?

Originally, from the starting state, the expected total time to reach home is 30 minutes.  Now if we change the route so that it now takes on average 5 more minutes to reach the car, but the expected elapsed time for every other leg of the journey is unchanged.  Now our total time estimate should be 35 minutes from the starting state on average.  Let's say we reach the car and nothing out of the ordinary is happening.  The predicted time to go will be 25 minutes and the predicted total time will be 35 minutes.  If nothing further out of the ordinary occurs, then only the first state will be corrected.  For the Monte Carlo method, the only state with an estimate error will be the first state, but this update will not occur until after we've arrived at our destination.  Either way, the next time we drive we will have a new, more accurate estimate reflecting the longer time required to reach the car.

$(example_6_1(;elapsed = [0, 10, 20, 25, 32, 35], predicted_ttg = [30, 25, 15, 10, 3, 0]))

In the example, during the drive several events occur during the journey that change the predicted and actual time from the average.  For simplicity let's assume that when we enter our home street there is a garbage truck blocking our path.  Normally it only takes 3 minutes to arrive at home, but with the truck present we estimate it will take 5 minutes (2 minutes longer).  Now the total predicted time will be increased from 35 minutes to 37 minutes.  In the case of Monte Carlo learning, this additional 2 minutes will propagate backwards to all of the previous states because we experienced a true travel time of 37 minutes rather than the 35 minutes predicted after the 2nd state and the 30 minutes predicted after the first state.  For TD(0) learning, however, this delay will only impact the previous state after a single update.  Effectively it will increase the predicted time spent on the final leg of the journey only.  The prediction from the starting state will only be increased by the 5 minute increase from the walk to the car, not the delay from the garbage truck.  Since we are actually starting from a new point, that feedback will be consistent and does reflect a true change in the expected time from the starting state.  The garbage truck, however, may be a rare occurence.  By the time this change propagates backwards through the states to the starting state, a lot more experience will be accummulated at all the other states and if α is some reasonable value, this delay will not be counted nearly as much as the updates from the first leg of the journey.  Since TD(0) only uses feedback from one step into the future immediately, if changes are made to the environment, those changes will only affect the most closely related states immediately.  In this example, all of the accurate predictions we still have about the later legs of the journey will be used to keep the predictions more stable.

$(example_6_1(;elapsed = [0, 10, 20, 25, 32, 37], predicted_ttg = [30, 25, 15, 10, 5, 0]))

The opposite extreme though could create a situation where the Monte Carlo updates were better.  Imagine instead that you moved houses in the same neighborhood such that once you enter the home street, it takes 5 minutes to reach your home instead of 3 minutes.  In this case, the Monte Carlo updates would move all of the state predictions up towards the 2 minute increase since all of the predictions would be too short.  The TD(0) update though would initially only increase the prediction for the final leg of the journey and we would have to wait for this change to propagate backwards to all the other states.  So the efficiency of updates for each method depends on where in the episode environmental changes occur.

Actual environment change at the end of the route
$(example_6_1(;elapsed = [0, 5, 15, 20, 27, 32], predicted_ttg = [30, 25, 15, 10, 3, 0]))

Now there is a randomly experienced shorter leg at the start of the journey which won't affect most of the Monte Carlo updates.
$(example_6_1(;elapsed = [0, 3, 13, 18, 25, 30], predicted_ttg = [30, 25, 15, 10, 3, 0]))
"""

# ╔═╡ 5290ae65-6f56-4849-a842-fe347315c6dc
md"""
## 6.2 Advantages of TD Prediction Methods
TD methods can learn before an episode terminates, so this is an advantage in environments that have very long episodes.  Also, in continuing problems, Monte Carlo methods may not be suitable at all because there is no termination condition.  Furthermore, if we consider off-policy learning, Monte Carlo methods must ignore returns if exploratory actions (ones never taken by the target policy) are taken later in the episode whereas TD methods could learn from individual steps that are not exploratory regardless of what happens later on.  

For any fixed policy $v_\pi$ TD(0) has been proved to converge to $v_\pi$ in the mean for a constant step-size parameter if it is sufficiently small, and with probability 1 if the step-size parameter decreases according to the usual stochastic approximation conditions (2.7).  Since both TD and Monte Carlo methods converge, one natural question is which converges faster, which makes more efficient use of limited data?  There is no mathematical proof to this question, nor is it clear how to even pose it formally; however, TD methods have usually been found to converge faster than constant-α MC methods on stochastic tasks, as illustrated in Example 6.2.
"""

# ╔═╡ 47c2cbdd-f6db-4ce5-bae2-8141f30aacbc
md"""
### Example 6.2 Random Walk

In this example we empirically compare the prediction abilities of TD(0) and constant-α MC when applied to the following Markov reward process:

In this MRP the agent's actions are irrelevant as each step the state transition occurs either to the left or the right with equal probability.  An episode ends when the transition terminates at the left or right side of the chain.  If the agent exits to the right, it receives a reward of 1.  Otherwise, all other transitions receive a reward of 0.  Below is an animation of the agent randomly moving through an episode.  Longer chains will have longer episode times on average growing roughly quadratically with the length of the chain.  Underneath the visualizations is the code.
"""

# ╔═╡ 5455fc97-55cb-4b0e-a3be-9433ccc96fc0
md"""
Number of States: $(@bind nstates Slider(3:10, default = 5, show_value=true))

Animation Interval (s): $(@bind delay Slider(0.1:0.1:1.0, default = 0.2, show_value=true))

$(@bind start_mrp Button("New Random Walk"))
"""

# ╔═╡ a9dda9b5-f568-481c-9e8f-9bb887468775
md"""
#### Random Walk MDP Setup
"""

# ╔═╡ 846720cc-550a-4a3c-a80e-40b99671f4e2
const mrp_moves = [-1, 1]

# ╔═╡ 4ddcd409-c31c-444c-8fcf-7cc45b68d93b
function make_mrp(;l = (5))
	function step(s, a)
		x = s + rand(mrp_moves)
		r = Float32(floor(x / (l+1)))
		(r, mod(x, l+1)) #if a transition is terminal will return 0
	end
	MDP_TD(collect(0:l), [1], () -> ceil(Int64, l/2), step, s -> s == 0)
end

# ╔═╡ 4b0d96d0-25d1-4fed-b105-c65fa2883a61
const mrp_6_2 = make_mrp(l = nstates)

# ╔═╡ 64fe8336-d1c2-41fe-a522-1b6f63260fc9
const π_mrp = make_random_policy(mrp_6_2)

# ╔═╡ 12c5efe4-d64d-4b82-877c-29b0e537fee6
begin
	start_mrp
	mrp_trajectory = runepisode(mrp_6_2, π_mrp)
end

# ╔═╡ 53145cc2-784c-468b-8e91-9bb7866db218
@bind t PlutoUI.Clock(interval = delay, max_value = length(mrp_trajectory[1])+5, repeat=true, start_running=true)

# ╔═╡ 54d97122-2d01-46ec-aafe-00bfc9f2d6d1
md"""
Step: $(min(length(first(mrp_trajectory)), t)) / $(length(first(mrp_trajectory)))
"""

# ╔═╡ a5009785-64b4-489b-a967-f7840b4a9463
md"""
#### Random Walk Visualization
"""

# ╔═╡ de50f95f-984e-4387-958c-64e0265f5953
function render_walk(id; l = 5)
	l > 26 && error("Cannot render more than 26 states")
	names = Iterators.take('A':'Z', l) |> collect
	startstate = names[ceil(Int64, l/2)]

	makestate(s) = """<div class = "circlestate $s"></div>"""

	function combinestates(s1, s2)
		"""
		$s1
		<div class = "arrow left right"><div>0</div></div>
		$s2
		"""
	end
	@htl("""
	<div id = "$id" style="display: flex; flex-direction: column; align-items: space-around; font-size: 18px;">
		<div class="randomwalk">
			<div class = "term left"></div>
			<div class="arrow left"><div>0</div></div>
			$(HTML(mapreduce(makestate, combinestates, names)))
			<div class="arrow right"><div>1</div></div>
			<div class="term right"></div>
		</div>
	</div>
	<style>
		.circlestate.$startstate::after {
			content: 'start';
			transform: translateX(-12%) translateY(-15%);
			position: absolute;
		}
	</style>
	""")
end

# ╔═╡ e4c6456c-867d-4ade-a3c8-310c1e065f14
render_walk("eg1", l = nstates)

# ╔═╡ f841c4d8-5176-4007-b472-9e01a799d85c
function addelements(e1, e2)
	"""
	$e1
	$e2
	"""
end

# ╔═╡ 889611fb-7dac-4769-9251-9a90e3a1422f
function statestyle(s)
	"""
	.circlestate.$s::before {
		content: '$s';
	}
	"""
end

# ╔═╡ 902738c3-2f7b-49cb-8580-29359c857027
@htl("""
<style>
$(mapreduce(statestyle, addelements, 'A':'Z'))
</style>
""")

# ╔═╡ 510761f6-66c7-4faf-937b-e1422ec829a6
HTML("""
<style>
	.randomwalk {
		margin: 5px;
		background-color: gray;
		height: 50px;
	}
	.randomwalk, .backup * {
		display: flex;
		flex-direction: row;
		align-items: center;
		justify-content: center;
		color: black;
	}
	.circlestate, .circleaction {
		margin: 2px;
	}

	.circlestate * {
		position: absolute;
		top: 18px;
		transform: translateX(5px);
	}

	.arrow * {
		transform: translateY(-11px);
	}

	.circlestate::before {
		content: '';
		display: flex;
		align-items:center;
		justify-content: center;
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
		width: 50px;
		height: 2px;
		background-color: black;
	}

	.arrow.left::before {
		content: '';
		display: inline-block;
		width: 4px;
		height: 4px;
		border-bottom: 2px solid black;
		border-right: 2px solid black;
		transform: translateX(-16px) rotate(135deg);
	}

	.arrow.right::after {
		content: '';
		display: inline-block;
		width: 4px;
		height: 4px;
		border-bottom: 2px solid black;
		border-right: 2px solid black;
		transform: translateX(16px) rotate(-45deg);
	}

	.arrow::after {
	
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

# ╔═╡ 87fadfc0-2cdb-4be2-81ad-e8fdeffb690c
function show_mrp_state(id, states, rewards, index)
	reward = rewards[min(index, length(states))]
	state = states[min(index, length(states))]
	dir = reward== 0 ? "left" : "right"

	termcolor = if index >= length(states)
		"""
		#$id .term.$dir::before {
			background-color: red;
		}
		"""
	else
		""""""
	end
	activestate = collect('A':'Z')[state]
	HTML("""
	<style>
		#$id .circlestate.$activestate::before {
			background-color: green;
		}
		$termcolor
	</style>
	"""
	)
end

# ╔═╡ 1dd1ba55-548a-41f6-903e-70742fd60e3d
show_mrp_state("eg1", mrp_trajectory[1], mrp_trajectory[3], t)

# ╔═╡ 2786101e-d365-4d6a-8de7-b9794499efb4
function example_6_2(;l = 5, max_episodes = 100, nruns = 100, vinit = 0.5f0)
	mrp = make_mrp(l = l)
	π = make_random_policy(mrp)
	true_values = collect(1:l) ./ (l+1)
	get_rw_names(l) = string.(Iterators.take('A':'Z', l) |> collect)
	(_, td0_est) = tabular_TD0_pred_V(π, mrp, 0.1f0, 1.0f0; num_episodes = 100, vinit = 0.5f0, save_states = collect(1:l))
	traces = [scatter(x = get_rw_names(l), y = td0_est[:, n], name = "$(n-1) episodes") for n in [1, 2, 11, 101]]
	tv_trace = scatter(x = get_rw_names(l), y = true_values, name = "True values", line_color="black")
	p1 = plot([tv_trace; traces], Layout(title = "Estimated Value with TD(0)", xaxis_title = "State"))

	calc_rms(v_saves) = [sqrt(mean((v .- true_values) .^2)) for v in eachcol(v_saves)]

	run_estimate(f, α, n) = f(π, mrp, α, 1.0f0; num_episodes = n, vinit = vinit, save_states = collect(1:l))

	td_αs = [0.05f0, 0.1f0, 0.15f0]
	mc_αs = 0.01f0:0.01f0:0.04f0 |> collect
	td_est = [mean([calc_rms(last(run_estimate(tabular_TD0_pred_V, α, max_episodes))) for _ in 1:nruns]) for α in td_αs]
	mc_est = [mean([calc_rms(last(run_estimate(monte_carlo_pred_V, α, max_episodes))) for _ in 1:nruns]) for α in mc_αs]
	td_traces = [scatter(x = collect(1:max_episodes), y = td_est[i], name = "$(i == 1 ? "TD" : "") α = $(td_αs[i])", line_color = "rgba(0, 0, 255, $(i/3))") for i in eachindex(td_est)]
	mc_traces = [scatter(x = collect(1:max_episodes), y = mc_est[i], name = "$(i == 1 ? "MC" : "") α = $(mc_αs[i])", line_color = "rgba(255, 0, 0, $(i/5))") for i in eachindex(mc_est)]

	p2 = plot([td_traces; mc_traces], Layout(xaxis_title = "Walks / Episodes", title = "Empirical RMS error, averaged over states"))
	@htl("""<div style = "display: flex;">
	$p1
	$p2
	</div>
	The right graph shows learning curves for the two methods for various values of α.  The performance measure shown is the root mean square (RMS) error between the vlue function learned and the true value function, averaged over the $l states, then averaged over $nruns runs.  In all cases the approximate value function was initialized to the intermediate value 0.5.  The TD method was consistently better than the MC method on this task.""")
end		

# ╔═╡ 9db7a268-1e6d-4366-a0ec-ebf54916d3b0
example_6_2(l = nstates)

# ╔═╡ 0b9c6dbd-4eb3-4167-886e-64db9ec7ff04
md"""
> ### *Exercise 6.3* 
> From the results shown in the left graph of the random walk example it appears that the first episode results in a change only in $V(A)$.  What does this tell you about what happened on the first episode?  Why was only the estimate for this one state changed?  By exactly how much was it changed?

The update rule with TD(0) learning is given by 

$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$

All states, A, B, C, D, E are initialized at 0.5 with the terminal state initialized at 0.  During the first episode for all transitions before the end, the reward is 0 and the difference between adjacent states would be 0 resulting in no change to the value function.  Since the value estimate for state A decreases from the initial value, this means that the first episode terminated to the left.  For this final transition we have the following update.

$V(A) \leftarrow V(A) + \alpha[0 + \gamma V(\text{Term}) - V(A)]$

We know that prior to the update $V(A) = 0.5$, $V(\text{Term}) = 0$ and $\gamma=1$ so the update is

$V(A) \leftarrow 0.5 + \alpha[0 - 0.5]$

For this plot, $\alpha=0.1$, so the updated value for $V(A)$ is $0.5+0.1(-0.5)=0.5-0.05=0.45$
"""

# ╔═╡ 52aebb7b-c2a9-443f-bc03-24cd25793b32
md"""
> ### *Exercise 6.4* 
> The specific results shown in the right graph of the random walk example are dependent on the value of the step-size parameter $\alpha$. Do you think the conclusions about which algorithm is better would be affected if a wider range of values were used? Is there a different, fixed value of $\alpha$ at which either algorithm would have performed significantly better than shown? Why or why not?

Both algorithms should theoretically converge to the true values with a sufficiently small $\alpha$ and a large enough number of samples.  Over this limited window of 100 episodes, an $\alpha$ that is too small might result in convergence so slow that it does not reach error as low as a larger $\alpha$.  For the MC method, $\alpha=0.01$ is the smallest value and it has the slowest convergence over this range.  $\alpha=0.04$ is the largest value tested, and it results in approximately the same error after 100 episodes.  The intermediate values show better performance over this number of episodes indicating that the best possible performance is already captured in this interval. 

For the TD method, the best results shown are for $\alpha=0.05$ which is already the smallest value with the slowest convergence rate.  An even smaller value might result in a better outcome over 100 episodes, but this performance is already better than anything observed for the MC method.
"""

# ╔═╡ e6672866-c0a0-46f2-bb52-25fcc3352645
md"""
> ### *Exercise 6.5* 
> In the right graph of the random walk example, the RMS error of the TD method seems to go down and then up again, particularly at high $\alpha$’s. What could have caused this? Do you think this always occurs, or might it be a function of how the approximate value function was initialized?

Since the value function was initialized at the correct value for the center state, all of the values to the right must be increased and the values to the left must be decreased to reach the true values.  Episodes that terminate to the right will receive a reward of 1 and push up the rightmost estimate while episodes that terminate to the left will receive a reward of 0 and decrease the leftmost estimate.  The correct value for each of these estimates is $\frac{1}{6}$ and $\frac{5}{6}$ respectively.  Since there is an equal probability of exiting the walk on the right or the left, both ends of the value estimates will be updated at roughly the same rate.  That means that both ends of the chain will move towards the correct value at about the same time and if those updates stay someone synchronized, all of the states will move through correct values at a similar time.  At the time when the values are roughly accurate, what happens if $\alpha=0.15$?  In this case, consider an update for state E assuming the estimate is already the correct value.  $V(E) \leftarrow \frac{5}{6} + 0.15[1 - \frac{5}{6}] \approx 0.858 \gt \frac{5}{6}$.  A similar effect happens with state A pushing it below the correct value.  The larger $\alpha$ is, the more over-correction we have on future transitions and the feedback from the other neighboring states won't be enough to bring it back to the correct value.  Since we pass through or very close to the correct value on the way, we pass through a minimum error value before over or undershooting the value estimate.  

If we had instead initialized the state values at 0, then the estimate at A would already be too low and would not get corrected until information from the right side propagated through.  State E, however, will receive large updates for each episode that exits to the right, but the values for the states to its left will be too low.  Since the state value estimates are not moving symmetrically, we won't have the same synchronized pass through the minimum error, since at the time the E estimate is correct, A will still be high error.  In this case, we are more likely to see error continue to fall as more updates occur.  Below is a visualization of the state estimates at different stages in the training with the original initialization and a 0 initialization.  In the 0 case, you can see the left-size estimates take a long time to reach the correct value, but in the original initialization, all the estimate approach the correct values roughly together. 
"""

# ╔═╡ ddf3bb61-16c9-48c4-95d4-263260309762
function exercise_6_5(;l = 5, max_episodes = 100, nruns = 100, α = 0.3f0, vinit = 0.5f0)
	mrp = make_mrp(l = l)
	π = make_random_policy(mrp)
	true_values = collect(1:l) ./ (l+1)
	get_rw_names(l) = string.(Iterators.take('A':'Z', l) |> collect)
	(_, td0_est) = tabular_TD0_pred_V(π, mrp, α, 1.0f0; num_episodes = 100, vinit = vinit, save_states = collect(1:l))
	traces = [scatter(x = get_rw_names(l), y = td0_est[:, n], name = "$(n-1) episodes") for n in [1, 2, 8, 16, 100]]
	tv_trace = scatter(x = get_rw_names(l), y = true_values, name = "True values", line_color="black")
	plot([tv_trace; traces], Layout(title = "Estimated Value with TD(0)", xaxis_title = "State"))
end		

# ╔═╡ e8f94345-9ad5-48d4-8709-d796fb55db3f
exercise_6_5(α = 0.2f0)

# ╔═╡ a72d07bf-e337-4bd4-af5c-44d74d163b6b
exercise_6_5(α = 0.2f0, vinit = 0.0f0)

# ╔═╡ 105c5c23-270d-437e-89dd-12297814c6e0
md"""
> ### *Exercise 6.6* 
> In Example 6.2 we stated that the true values for the random walk example are 1/6 , 2/6 , 3/6 , 4/6 , and 5/6 , for states A through E. Describe at least two different ways that these could have been computed. Which would you guess we actually used? Why?

###### Method 1: Set up the following system of equations that represent the relationship between state values
$\begin{flalign}
V(A) &= \frac{0+V(B)}{2} \implies 2V(A)=V(B) \\
V(B) &= \frac{V(A)+V(C)}{2} \implies 2V(B) = V(A)+V(C)\\
V(C) &= \frac{V(B)+V(D)}{2} \implies 2V(C)=V(B)+V(D)\\
V(D) &= \frac{V(C)+V(E)}{2} \implies 2V(D)=V(C)+V(E)\\
V(E) &= \frac{V(D)+1}{2} \implies 2V(E)=V(D)+1\\
\end{flalign}$

We can work down from the top equation expressing everything in terms of A.  For shorter expressions $V(A)$ will be written below as $A$ and likewise for other states:

$\begin{flalign}
B&=2A \\
2B&=A+C \implies C = 3A \\
2C&=B+D \implies D = 6A-2A=4A \\
2D&=C+E \implies E = 8A-3A = 5A \\
2E &= D + 1 \implies 10A = 4A + 1 \implies A = \frac{1}{6}
\end{flalign}$

Now that we have the value for A, all the others are trivial multiplications of it from 2 to 5.

###### Method 2: Calculate each value from probability of each trajectory
With this method to get $V(A)$ we would write down every possible trajectory to a terminal state with the associated probability of each.  Since trajectories terminating to the left have a value of 0, we only need to add up the trajectories that terminate to the right.  Below are some examples for state A.

$V(A) = 0.5^5 + 4 \times 0.5^7 + \cdots$

This equation represents the single trajectory that takes 5 steps to the right each with probability one half and the 4 possible trajectories that turn around once on the way right resulting in 7 steps.  This sum will end up being infintely long to account for all of the trajectories that bounce back and forth arbitrarily large amounts of time.  This method is significantly harder to calculate for each state compared to the first method and is more in line with how estimates are calculated with MC sampling.  The first method is more analogous to TD sampling using the bootstrapped form of the Bellman equation.
"""

# ╔═╡ 48b557e3-e239-45e9-ab15-105bcca96492
md"""
## 6.3 Optimality of TD(0)

Suppose there is available only a finite amount of experience, say 10 episodes or 100 time steps.  In this case, a common approach with incremental learning methods is to present the experience repeatedly until the method converges upon an answer.  Given an approximate value function $V$, the increments specified by (6.1) or (6.2) are computed for every time step $t$ at which a nonterminal state is visited, but the value function is changed only once, by the sum of all the increments.  Then all the available experience is processed again with the new value function to produce a new overall increment, and so on, until the value function converged.  We call this *batch updating* because updates are made only after processing each complete *batch* of training data.  

Under batch updating, TD(0) converges deterministically to a single answer independent of the step-size parameter, $\alpha$, as long as $\alpha$ is chosen to be sufficiently small.  The constant $\alpha$ MC method also converges deterministically under the same conditions, but to a difference answer.  Understanding these two answers will help us understand the difference between the two methods.  Under normal updating the methods do not move all the way to their respective batch answers, but in some sense they take steps in these directions.  Before trying to understand the two answers in general, for all possible tasks, we first look at a few examples.

### Example 6.3: Random walk under batch updating

Batch-updating versions of TD(0) and constant-$\alpha$ MC were applied as follows to the random walk prediction example (Example 6.2).  After each new episode, all episodes seen so far were treated as a batch.  They were repeatedly presented to the algorithm, either TD(0) or constant-$\alpha$ MC, with $\alpha$ sufficiently small that the value function converged.  The resulting value function was then compared with $v_\pi$, and the average root mean square error across the five states (and accross 100 independent repetitions of the whole experiment) was plotted to obtain the learning curves shown in Figure 6.2.  Note that the batch TD method was consistently better than the batch Monte Caro method.  

Under batch training, constant-$\alpha$ MC converges to the values, $V(s)$, that are sample averages of the actual returns experienced after visiting each state $s$.  These are optimal estimates in the sense that they minimize the mean square error from the actual returns in the training set.  In this sense it is surprising that the batch TD method was able to perform better according to the root mean square error measure shown in figure 6.2.  How is it that batch TD was able to perform better than this optimal method?  The answer is that the Monte Carlo method is optimal only in a limited way, and that TD is optimal in a way that is more relevant to predicting returns.

Below is code implementing both batch methods in general for arbitrary MDPs.

"""

# ╔═╡ 0a4ed8c7-27ca-45cb-af15-70ddd86240fb
md"""
#### Batch Method Estimation Implementation
"""

# ╔═╡ 620a6426-cb29-4010-997b-aa4f9d5f8fb0
begin
	abstract type BatchMethod end
	struct TD0 <: BatchMethod end
	struct MC <: BatchMethod end
end

# ╔═╡ 3d8b1ccd-9bb3-42f2-a77a-6afdb72c1ff8
#calculate the percentage error for a value update handling cases of zero values
function calc_error(v_old::T, v_new::T) where T<:AbstractFloat
	d = v_new - v_old
	return abs(d)
	f(x) = x <= eps(one(T))
	f(d) && f(v_old) && return zero(T)
	f(v_old) && return typemax(T)
	abs(d) / abs(v_old)
end

# ╔═╡ 209881b3-3ac8-490e-97bd-fa5ae24a39f5
#update the value function with the TD0 method using a single episode
function update_value!(V::Vector{T}, ::TD0, α::T, γ::T, mdp::MDP_TD{S, A, F, G, H}, states::Vector{S}, actions::Vector{A}, rewards::Vector{T}) where {T<:AbstractFloat, S, A, F<:Function, G<:Function, H<:Function}
	l = length(states)
	err = zero(T)
	for i in 1:l-1
		s = states[i]
		s′ = states[i+1]
		i_s = mdp.statelookup[s]
		v_old = V[i_s]
		i_s′ = mdp.statelookup[s′]
		v_new = v_old + α*(rewards[i] + γ*V[i_s′] - v_old)
		err = max(err, calc_error(v_old, v_new))
		V[i_s] = v_new
	end
	#perform update for terminal state
	s = last(states)
	i_s = mdp.statelookup[s]
	v_old = V[i_s]
	v_new = v_old + α*(rewards[l] - v_old)
	err = max(err, calc_error(v_old, v_new))
	V[i_s] = v_new
	return err
end

# ╔═╡ 72b4d8d5-464c-4561-8c69-28ef3f59630b
#update the value function with the MC method using a single episode
function update_value!(V::Vector{T}, ::MC, α::T, γ::T, mdp::MDP_TD{S, A, F, G, H}, states::Vector{S}, actions::Vector{A}, rewards::Vector{T}) where {T<:AbstractFloat, S, A, F<:Function, G<:Function, H<:Function}
	l = length(states)
	g = zero(T)
	err = zero(T)
	for i in l:-1:1
		g = γ*g + rewards[i]
		s = states[i]
		i_s = mdp.statelookup[s]
		v_old = V[i_s]
		v_new = v_old + α*(g-v_old)
		err = max(err, calc_error(v_old, v_new))
		V[i_s] = v_new
	end
	return err
end

# ╔═╡ 3f3ebc9b-b070-4d73-8be9-823b399c664c
#compute the value function for a policy π on an mdp with a constant step size parameter α and a discount rate of γ.  Must provide a tolerance ϵ which is the maximum difference observed when updating the value function that can be tollerated to consider the value function to be converged.
function batch_value_est(π::Matrix{T}, mdp::MDP_TD{S, A, F, G, H}, α::T, γ::T, ϵ::T; num_episodes::Integer = 1000, vinit::T = zero(T), save_states::Vector{S} = Vector{S}(), V::Vector{T} = initialize_state_value(mdp; vinit = vinit), estimation_method::BatchMethod = TD0(), maxcount = typemax(T)) where {T<:AbstractFloat, S, A, F, G, H} 
	check_policy(π, mdp)
	terminds = findall(mdp.isterm(s) for s in mdp.states)
	V[terminds] .= zero(T)
	v_saves = zeros(T, length(save_states), num_episodes+1)
	errors = zeros(T, num_episodes)
	function update_saves!(v_saves, ep)
		for (i, s) in enumerate(save_states)
			i_s = mdp.statelookup[s]
			v_saves[i, ep] = V[i_s]
		end
	end
	
	update_saves!(v_saves, 1)

	#each tuple in this vector matches an output from the runepisode function
	saved_episodes = Vector{Tuple{Vector{S}, Vector{A}, Vector{T}}}()

	for n in 1:num_episodes
		push!(saved_episodes, runepisode(mdp, π))
		err = typemax(T)
		#wait until the error has converged
		count = zero(T)
		while (count < maxcount) && (err > ϵ)
			worst_error = zero(T)
			#update values for entire batch of episodes
			for ep in saved_episodes
				#update values for each episode in a batch and update the worst error
				worst_error = max(worst_error, update_value!(V, estimation_method, α, γ, mdp, ep...))
			end
			err = worst_error
			count += 1
		end
		errors[n] = err
		#only update saves after the value function has converged for this batch
		update_saves!(v_saves, n+1)
	end
	return V, v_saves, errors
end

# ╔═╡ 1e3d231a-4065-48ce-a74e-018066fb232a
function example_6_3(;l = 5, max_episodes = 100, nruns = 100, vinit = 0.5f0, α = 0.05f0, ϵ = α, kwargs...)
	#note that for this task the error tolerance is set to the step size because the only reward experienced is 1, so the smallest possible maximum value update is α anyway
	mrp = make_mrp(l = l)
	π = make_random_policy(mrp)
	true_values = collect(1:l) ./ (l+1)

	function get_errors(method)
		(v, v_saves, errors) = batch_value_est(π, mrp, α, 1.0f0, ϵ; num_episodes = max_episodes, vinit=vinit, save_states = collect(1:l), estimation_method = method, kwargs...)
		sqrt.(mean((v_saves .- true_values) .^2, dims = 1))
	end

	mc_errors = mean([get_errors(MC()) for _ in 1:nruns])[:]
	td0_errors = mean([get_errors(TD0()) for _ in 1:nruns])[:]

	t1 = scatter(x = 0:max_episodes, y = mc_errors, name = "MC")
	t2 = scatter(x = 0:max_episodes, y = td0_errors, name = "TD")

	p = plot([t1, t2], Layout(xaxis_title = "Walks / Episodes", yaxis_title = "RMS error, averaged over states", title = "Batch Training"))

	md"""
	#### Figure 6.2
	
	$p

	Performance of TD(0) and constant-α MC under batch training on the random walk task with $l states
	"""
	
end		

# ╔═╡ 187fc682-2282-46ca-b988-c9de438f36fd
@bind params_6_2 confirm(PlutoUI.combine() do Child
	md"""
	Batch Training of Random Walk Task
	
	|||
	|:-:|:-:|
	|$\alpha$| $(Child(:α, Slider(0.001:0.001:0.1, default = 0.01, show_value=true)))|
	|Number of States | $(Child(:l, Slider(3:10, default = 5, show_value=true)))|
	|Maximum Episodes | $(Child(:ep, Slider(100:1000, default = 100, show_value=true)))|
		"""
end)

# ╔═╡ 22c2213e-5b9b-410f-a0ef-8f1e3db3c532
example_6_3(;l = params_6_2.l, max_episodes = params_6_2.ep, α = Float32(params_6_2.α), vinit=0.5f0)

# ╔═╡ 0e59e813-3d48-4a24-b5b3-9a9de7c500c2
md"""
> ### *Exercise 6.7* 
> Design an off-policy version of the TD(0) update that can be used with arbitrary target policy $\pi$ and convering behavior policy $b$, using each step $t$ the importance sampling ratio $\rho_{t:t}$ (5.3).

Recall that equation 5.3 defines:

$\rho_{t:T-1} = \prod_{k=t}^{T-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}$

with the property that:

$\mathbb{E}[\rho_{t:T-1}G_t \mid S_t = s] = v_\pi(s)$ when $G_t$ is generated by the behavior policy.

The TD(0) update rule is given by:

$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$

based on the following form of the Bellman equation:

$v_\pi (s)=\text{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s]$

In the off-policy case, the reward $R_{t+1}$ and the subsequent state $S_{t+1}$ would be generated from the behavior policy, but the subsequent value would still be based on the target policy value function.  Consider instead the quantity: $q_\pi(s, a) = \mathbb{E} [R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s, A_t = a]$ where we have removed the policy from the expectation since nothing in the bracket depends on sampling from the policy.  Even if we chose actions a based on a behavior policy that differs from the target policy, these estimates will be correct because we are directly calculating the value for choosing that action, regardless of what the probability is.  Consier we are following some behavior policy $b$ and  recall that: 

$\begin{flalign}
v_\pi(s) &= \sum_a \pi(a \vert s) q_\pi (s, a) \\
&= \sum_a \pi(a \vert s) \mathbb{E} [R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s, A_t = a]\\ 
&= \mathbb{E}_\pi [R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s]\\ 
v_b(s) &= \sum_a b(a \vert s) q_\pi (s, a) \\
&= \sum_a b(a \vert s) \mathbb{E} [R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s, A_t = a] \\
&= \mathbb{E}_b [R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s]\\ 
\end{flalign}$

In the TD(0) update we do not calculate this expected value directly but instead average samples together that are drawn from the target policy.  This sampling will produce samples weighted by the target policy probabilities thus mimicking the expected value sum.  If instead, our samples are drawn from the behavior policy, then the samples will mimic the behavior policy probability weights instead of the target policy.  So in order to correctly calculate the expected value we must multiply each behavior policy sample by $\frac{\pi(a \vert s)}{b(a \vert s)} = \frac{\pi(A_t \vert S_t)}{b(A_t \vert S_t)} = \rho_{t:t}$ resulting in the following update rule:

$V(S_t) \leftarrow V(S_t) + \alpha [\rho_{t:t} \left ( R_{t+1} + \gamma V(S_{t+1}) \right ) - V(S_t)]$
"""

# ╔═╡ 0d6a11af-b146-4bbc-997e-a11b897269a7
md"""
## 6.4 Sarsa: On-policy TD Control
"""

# ╔═╡ a925534e-f9b8-471a-9d86-c9212129b630
md"""
The following represents a trajectory taken by a policy in an environment.  We week to estimate $q_\pi(s, a)$ for the current behavior policy $\pi$ using the same TD method we introduced above.  The update rule now, however, estimates the value of state action pairs rather than the states themselves.
"""

# ╔═╡ 62a9a36a-bedb-4f5a-80a4-2d4111a65c12
@htl("""
<div style = "display: flex; justify-content: center; align-items: center; background-color: gray; height: 100px; font-size: .75em; color: black;">
<div>$(md"""$\cdots \:$""")</div>
<div class = "link1"></div>
<div class = "state"><div>$(md"""$S_t$""")</div></div>
<div class = "link1"></div>
<div class = "action">$(md"""$A_t$""")</div>
<div class = "link2"><div>$(md"""$R_{t+1}$""")</div></div>
<div class = "state">$(md"""$S_{t+1}$""")</div>
<div class = "link1"></div>
<div class = "action">$(md"""$A_{t+1}$""")</div>
<div class = "link2"><div>$(md"""$R_{t+2}$""")</div></div>
<div class = "state">$(md"""$S_{t+2}$""")</div>
<div class = "link1"></div>
<div class = "action">$(md"""$A_{t+2}$""")</div>
<div class = "link2"><div>$(md"""$R_{t+3}$""")</div></div>
<div class = "state">$(md"""$S_{t+3}$""")</div>
<div class = "link1"></div>
<div>$(md"""$\:\cdots$""")</div>
</div>

<style>
	.state {
		color: black;
		background-color: white;
		width: 40px;
		height: 40px;
		border: 2px solid black;
		border-radius: 50%;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.link1 {
		width: 40px;
		height: 2px;
		background-color: black;
	}

	.link2 {
		width: 40px;
		height: 2px;
		background-color: black;
	}

	.link2 * {
		position: relative;
		color: black;
		top: -9px;
	}

	.action {
		color: black;
		width: 10px;
		height: 10px;
		background-color: black;
		border: 2px solid black;
		border-radius: 50%;
	}

	.action * {
		color: black;
		position: relative;
		top: -3px;
	}
</style>

""")

# ╔═╡ b35264b0-ac5b-40ce-95e4-9b2bc4cb106f
md"""
TD(0) update rule for action values:

$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})-Q(S_t, A_t)]$

This update is done after every transition from a nonterminal state $S_t$.  If $S_{t+1}$ is terminal, then $Q(S_{t+1}, A_{t+1})$ is defined as zero.  This rule uses every element of the quintuple of events, $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$, that make up a transition from one state-action pair to the next.  This quintuple gives rise to the name *Sarsa* for the algorithm.  Each update only uses the immediate reward and the value of the state-action pair in the subsequent state as illustrated in the backup diagram shown below.
"""

# ╔═╡ 4d7619ee-933f-452a-9202-e95a8f3da20f
@htl("""
Sarsa backup diagram.  Black circles represent actions and white circles represent states.
<div style="display: flex; align-items: center; justify-content: center; background-color: gray;">
<div class = "action"></div>
<div class="arrow right"></div>
<div class = "state"></div>
<div class = "arrow right"></div>
<div class = "action"></div>
</div>
""")

# ╔═╡ fe2ebf39-4ab3-4aa8-abbd-23389eaf400e
md"""
Sarsa converges with probability 1 to an optimal policy and action-value function, under the usual conditions on step sizes (2.7), as long as all state-action pairs are visited an infinite number of times and the policy converges in the limit to the greedy policy (which can be arranged, for example, with $\epsilon$-greedy policies by setting $\epsilon = 1/t$).  Below is code that implements Sarsa using the $\epsilon$-greedy method for exploration.
"""

# ╔═╡ 1ae30f5d-b25b-4dcb-800f-45c463641ec5
md"""
> ### *Exercise 6.8* 
> Show that an action-value version of (6.6) holds for the action-value form of the TD error $\delta_t=R_{t+1}+\gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$, again assuming that the values don't change from step to step.

The derivation in (6.6) starts with the definition in (3.9):

$G_t = R_{t+1} + \gamma G_{t+1}$

and derives the following:

$\delta_t \doteq R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$
$G_t - V(S_t) = \sum_{k=t}^{T-1} \gamma^{k-t} \delta_k$

Now we have the action-value form of the TD error:

$\delta_t \doteq R_{t+1}+\gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$

Let us transform (3.9) in a similar manner to derive the rule:

$\begin{flalign}
G_t - Q(S_t, A_t) &= R_{t+1} + \gamma G_{t+1} - Q(S_t, A_t) + \gamma Q(S_{t+1}, A_{t+1}) - \gamma Q(S_{t+1}, A_{t+1}) \\
&= \delta_t + \gamma (G_{t+1} - Q(S_{t+1}, A_{t+1})) \\
&= \delta_t + \gamma \delta_{t+1} + \gamma^2 (G_{t+2} - Q(S_{t+2}, A_{t+2})) \tag{using recursion} \\
&= \delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+1} + \cdots + \gamma^{T-t-1} \delta_{T-1} + \gamma^{T-t}(G_T - Q(S_T, A_T)) \\
&= \delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+1} + \cdots + \gamma^{T-t-1} \delta_{T-1} + \gamma^{T-t}(0-0) \tag{terminal value} \\
&= \sum_{k=t}^{T-1}\gamma^{k-t}\delta_k
\end{flalign}$
"""

# ╔═╡ 6a1503c6-c77b-4e3a-9f07-74b2af1a5ff7
md"""
### Sarsa Implementation
"""

# ╔═╡ 6b496582-cc0e-4195-87ef-94792b0fff54
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

# ╔═╡ cb07a6a5-c50a-4900-9e5b-a17dc7ee5710
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

# ╔═╡ 4d4577b5-3753-450d-a247-ebd8c3e8f799
function create_ϵ_greedy_policy(Q::Matrix{T}, ϵ::T; π = copy(Q), get_valid_inds = j -> 1:size(Q, 1)) where T<:Real
	vhold = zeros(T, size(Q, 1))
	for j in 1:size(Q, 2)
		vhold .= Q[:, j]
		make_ϵ_greedy_policy!(vhold, ϵ; valid_inds = get_valid_inds(j))
		π[:, j] .= vhold
	end
	return π
end

# ╔═╡ 12aac612-758b-4655-8ede-daddd4af6d3e
#take a step in the environment from state s using policy π and generate the subsequent action selection as well
function sarsa_step(mdp::MDP_TD{S, A, F, G, H}, π::Matrix{T}, s::S, a::A) where {S, A, F<:Function, G<:Function, H<:Function, T<:Real}
	(r, s′) = mdp.step(s, a)
	i_s′ = mdp.statelookup[s′]
	i_a′ = sample_action(π, i_s′)
	a′ = mdp.actions[i_a′]
	return (s′, i_s′, r, a′, i_a′)
end

# ╔═╡ 3ed12c33-ab0a-49b1-b9e7-c4305ba35767
#take a step in the environment from state s using policy π and generate the subsequent action selection as well
function init_step(mdp::MDP_TD{S, A, F, G, H}, π::Matrix{T}, s::S) where {S, A, F<:Function, G<:Function, H<:Function, T<:Real}
	i_s = mdp.statelookup[s]
	i_a = sample_action(π, i_s)
	a = mdp.actions[i_a]
	return (i_s, i_a, a)
end

# ╔═╡ 61bbf9db-49a0-4709-83f4-44f228be09c0
function sarsa(mdp::MDP_TD{S, A, F, G, H}, α::T, γ::T; num_episodes = 1000, qinit = zero(T), ϵinit = one(T)/10, Qinit = initialize_state_action_value(mdp; qinit=qinit), history_state::S = first(mdp.states), save_history = false, decay_ϵ = false) where {S, A, F, G, H, T<:AbstractFloat}
	terminds = findall(mdp.isterm(s) for s in mdp.states)
	Q = copy(Qinit)
	Q[:, terminds] .= zero(T)
	π = create_ϵ_greedy_policy(Q, ϵinit)
	vhold = zeros(T, length(mdp.actions))
	#keep track of rewards and steps per episode as a proxy for training speed
	rewards = zeros(T, num_episodes)
	steps = zeros(Int64, num_episodes)

	if save_history
		action_history = Vector{A}(undef, num_episodes)
	end

	for ep in 1:num_episodes
		ϵ = decay_ϵ ? ϵinit/ep : ϵinit
		s = mdp.state_init()
		(i_s, i_a, a) = init_step(mdp, π, s)
		rtot = zero(T)
		l = 0
		while !mdp.isterm(s)
			(s′, i_s′, r, a′, i_a′) = sarsa_step(mdp, π, s, a)
			if save_history && (s == history_state)
				action_history[ep] = a
			end
			Q[i_a, i_s] += α * (r + γ*Q[i_a′, i_s′] - Q[i_a, i_s])
			
			#update terms for next step
			vhold .= Q[:, i_s]
			make_ϵ_greedy_policy!(vhold, ϵ)
			π[:, i_s] .= vhold
			s = s′
			a = a′
			i_s = i_s′
			i_a = i_a′
			
			l+=1
			rtot += r
		end
		steps[ep] = l
		rewards[ep] = rtot
	end

	default_return =  Q, π, steps, rewards
	save_history && return (default_return..., action_history)
	return default_return
end

# ╔═╡ 8d05403a-adeb-40ac-a98a-87586d5a5170
md"""
### Example 6.5: Windy Gridworld
"""

# ╔═╡ e19db54c-4b3c-42d1-b016-9620daf89bfb
begin
	abstract type GridworldAction end
	struct Up <: GridworldAction end
	struct Down <: GridworldAction end
	struct Left <: GridworldAction end
	struct Right <: GridworldAction end

	struct GridworldState
		x::Int64
		y::Int64
	end

	rook_actions = [Up(), Down(), Left(), Right()]
	
	move(::Up, x, y) = (x, y+1)
	move(::Down, x, y) = (x, y-1)
	move(::Left, x, y) = (x-1, y)
	move(::Right, x, y) = (x+1, y)

	apply_wind(w, x, y) = (x, y+w)
	const wind_vals = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
end

# ╔═╡ 500d8dd4-fc53-4021-b797-114224ca4deb
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

# ╔═╡ 136d1d96-b590-4f03-9e42-2337efc560cc
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
</style>
""")

# ╔═╡ 4556cf44-4a1c-4ca4-bfb8-4841301a2ce6
function display_rook_policy(v::Vector{T}; scale = 1.0) where T<:AbstractFloat
	@htl("""
		<div style = "display: flex; align-items: center; justify-content: center; transform: scale($scale);">
		<div class = "downarrow" style = "position: absolute; transform: rotate(180deg); opacity: $(v[1]);"></div>	
		<div class = "downarrow" style = "position: absolute; opacity: $(v[2])"></div>
		<div class = "downarrow" style = "position: absolute; transform: rotate(90deg); opacity: $(v[3])"></div>
		<div class = "downarrow" style = "transform: rotate(-90deg); opacity: $(v[4])"></div>
		</div>
	""")
end

# ╔═╡ 9f28772c-9afe-4253-ab3b-055b0f48be6e
function plot_path(mdp, π; title = "Optimal policy <br> path example", windtext =  [0, 0, 0, 1, 1, 1, 2, 2, 1, 0], xtitle = "Wind Values")
	eg = runepisode(mdp, π; max_steps = 100)
	xmax = maximum([s.x for s in mdp.states])
	ymax = maximum([s.y for s in mdp.states])
	start = mdp.state_init()
	goal = mdp.states[findfirst(mdp.isterm(s) for s in mdp.states)]
	start_trace = scatter(x = [start.x + 0.5], y = [start.y + 0.5], mode = "text", text = ["S"], textposition = "left", showlegend=false)
	finish_trace = scatter(x = [goal.x + .5], y = [goal.y + .5], mode = "text", text = ["G"], textposition = "left", showlegend=false)
	path_traces = [scatter(x = [eg[1][i].x + 0.5, eg[1][i+1].x + 0.5], y = [eg[1][i].y + 0.5, eg[1][i+1].y + 0.5], line_color = "blue", mode = "lines", showlegend=false, name = "Optimal Path") for i in 1:length(eg[1])-1]
	finalpath = scatter(x = [eg[1][end].x + 0.5, goal.x + .5], y = [eg[1][end].y + 0.5, goal.y + 0.5], line_color = "blue", mode = "lines", showlegend=false, name = "Optimal Path")

	h1 = 30*ymax
	plot([start_trace; finish_trace; path_traces; finalpath], Layout(xaxis = attr(showgrid = true, showline = true, gridwith = 1, gridcolor = "black", zeroline = true, linecolor = "black", mirror=true, tickvals = 1:xmax, ticktext = windtext, range = [1, xmax+1], title = xtitle), yaxis = attr(linecolor="black", mirror = true, gridcolor = "black", showgrid = true, gridwidth = 1, showline = true, tickvals = 1:ymax, ticktext = fill("", ymax), range = [1, ymax+1]), width = 30*xmax, height = max(h1, 200), autosize = false, padding=0, paper_bgcolor = "rgba(0, 0, 0, 0)", title = attr(text = title, font_size = 14, x = 0.5)))
end

# ╔═╡ 0ad739c9-8aca-4b82-bf20-c73584d29535
md"""
> ### *Exercise 6.9 Windy Gridworld with King's Moves (programming)* 
> Re-solve the windy gridworld assuming eight possible actions, including the diagonal moves, rather than four.  How much better can you do with the extra actions?  Can you do even better by including a ninth action that causes no movement at all other than that caused by the wind?
"""

# ╔═╡ 031e1106-7408-4c7e-b78e-b713c19123d1
begin
	struct UpRight <: GridworldAction end
	struct DownRight <: GridworldAction end
	struct UpLeft <: GridworldAction end
	struct DownLeft <: GridworldAction end

	const diagonal_actions = [UpRight(), UpLeft(), DownRight(), DownLeft()]
	const king_actions = [rook_actions; diagonal_actions]
	
	move(::UpRight, x, y) = (x+1, y+1)
	move(::UpLeft, x, y) = (x-1, y+1)
	move(::DownRight, x, y) = (x+1, y-1)
	move(::DownLeft, x, y) = (x-1, y-1)
end

# ╔═╡ cdedd35e-52b8-40a5-938d-2d36f6f93217
const king_action_display = @htl("""
<div style = "display: flex; flex-direction: column; align-items: center; justify-content: center; color: black; background-color: rgba(100, 100, 100, 0.1);">
	<div style = "display: flex; align-items: center; justify-content: center;">
	<div class = "downarrow" style = "transform: rotate(90deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(180deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(45deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(-45deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(135deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(-135deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(270deg);"></div>
	<div class = "downarrow" style = "position: absolute;"></div>
	</div>
	<div>Actions</div>
</div>
""")

# ╔═╡ 9651f823-e1cd-4e6e-9ce0-be9ea1c3f0a4
function display_king_policy(v::Vector{T}; scale = 1.0) where T<:AbstractFloat
	@htl("""
		<div style = "display: flex; align-items: center; justify-content: center; transform: scale($scale);">
		<div class = "downarrow" style = "position: absolute; transform: rotate(180deg); opacity: $(v[1]);"></div>	
		<div class = "downarrow" style = "position: absolute; opacity: $(v[2])"></div>
		<div class = "downarrow" style = "position: absolute; transform: rotate(90deg); opacity: $(v[3])"></div>
		<div class = "downarrow" style = "transform: rotate(-90deg); opacity: $(v[4])"></div>
		<div class = "downarrow" style = "position: absolute; transform: rotate(-135deg); opacity: $(v[5])"></div>
		<div class = "downarrow" style = "position: absolute; transform: rotate(135deg); opacity: $(v[6])"></div>
		<div class = "downarrow" style = "position: absolute; transform: rotate(-45deg); opacity: $(v[7])"></div>
		<div class = "downarrow" style = "position: absolute; transform: rotate(45deg); opacity: $(v[8])"></div>
		</div>
	""")
end

# ╔═╡ 2155adfa-7a93-4960-950e-1b123da9eea4
king_actions

# ╔═╡ d259ecca-0249-4b28-a4d7-6880d4d84495
const action3_display = @htl("""
<div style = "display: flex; flex-direction: column; align-items: center; justify-content: center; color: black; background-color: rgba(100, 100, 100, 0.1);">
	<div style = "display: flex; align-items: center; justify-content: center;">
	<div class = "downarrow" style = "transform: rotate(90deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(180deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(45deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(-45deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(135deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(-135deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(270deg);"></div>
	<div class = "downarrow" style = "position: absolute;"></div>
	<div style = "width: 15px; height: 15px; background-color: black; position: absolute; border-radius: 50%;"></div>
	</div>
	<div>Actions</div>
</div>
""")

# ╔═╡ 39470c74-e554-4f6c-919d-97bec1eec0f3
md"""
Adding king's move actions, the optimal policy can finish in 7 steps vs 15 for the original actions.  What happens after adding a 9th action that causes no movement?
"""

# ╔═╡ e9359ca3-4d11-4365-bc6e-7babc6fcc7de
begin
	struct Stay <: GridworldAction end
	move(::Stay, x, y) = (x, y)
end

# ╔═╡ ec285c96-4a75-4af6-8898-ec3176fa34c6
function make_windy_gridworld(;actions = rook_actions, apply_wind = apply_wind, sterm = GridworldState(8, 4), start = GridworldState(1, 4), xmax = 10, ymax = 7, winds = wind_vals, step_reward = -1f0)
	
	states = [GridworldState(x, y) for x in 1:xmax for y in 1:ymax]
	
	boundstate(x::Int64, y::Int64) = (clamp(x, 1, xmax), clamp(y, 1, ymax))
	
	function step(s::GridworldState, a::GridworldAction)
		w = winds[s.x]
		(x1, y1) = move(a, s.x, s.y)
		(x2, y2) = apply_wind(w, x1, y1)
		GridworldState(boundstate(x2, y2)...)
	end

	tr(s0::GridworldState, a0::GridworldAction) = (step_reward, step(s0, a0))
	isterm(s::GridworldState) = s == sterm

			
	MDP_TD(states, actions, () -> start, tr, isterm)
end	

# ╔═╡ ab331778-f892-4690-8bb3-26464e3fc05f
const windy_gridworld = make_windy_gridworld()

# ╔═╡ dda222ef-8178-40bb-bf20-d242924c4fab
const king_gridworld = make_windy_gridworld(;actions=king_actions)

# ╔═╡ db31579e-3e56-4271-8fc3-eb13bc95ac27
md"""
Adding the no-movement action doesn't seem to change the shortest path of 7 steps
"""

# ╔═╡ b59eacf8-7f78-4015-bf2c-66f89bf0e24e
md"""
> ### *Exercise 6.10: Stochastic Wind (programming)* 
> Re-solve the windy gridworld task with King's moves, assuming the effect of the wind, if there is any, is stochastic, sometimes varying by 1 from the mean values given for each column.  That is, a third of the time you move exactly according to these values, as in the previous exercise, but also a third of the time you move one cell above that, and another third of the time you move one cell below that.  For example, if you are one cell to the right of the goal and you move left, then one-third of the time you move one cell above the goal, one-third of the time you move two cells above the goal, and one-third of the time you move to the goal.
"""

# ╔═╡ 02f34da1-551f-4ce5-a588-7f3a14afd716
const wind_var = [-1, 0, 1]

# ╔═╡ aa0791a5-8cf1-499b-9900-4d0c59be808c
function stochastic_wind(w, x, y)
	w == 0 && return (x, y)
	
	v = rand(wind_var)
	(x, y+w+v)
end

# ╔═╡ 4ddc7d99-0b79-4689-bd93-8798b105c0a2
const stochastic_gridworld = make_windy_gridworld(actions = king_actions, apply_wind = stochastic_wind)

# ╔═╡ 2d881aa9-1da3-4d1e-8d05-245956dbaf33
HTML("""
	<style>
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

# ╔═╡ 8bc54c94-9c92-4904-b3a6-13ff3f0110bb
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

# ╔═╡ 678cad7a-1abb-4fcc-91ba-b5abcbb914cb
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

# ╔═╡ 9da5fd84-800d-4b3e-8627-e90ce8f20297
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

# ╔═╡ 44c49006-e210-4f97-916e-fe62f36c593f
md"""
## 6.5 Q-learning: Off-policy TD Control

One of the early breakthroughs in reinforcement learning was the development of an off-policy TD control algorithm known as *Q-learning* (Watkins, 1989), defined by

$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \text{max}_a Q(S_{t+1}, a) - Q(S_t, A_t)]$
"""

# ╔═╡ 2034fd1e-5171-4eda-85d5-2de62d7a1e8b
function q_learning(mdp::MDP_TD{S, A, F, G, H}, α::T, γ::T; num_episodes = 1000, qinit = zero(T), ϵinit = one(T)/10, Qinit = initialize_state_action_value(mdp; qinit=qinit), πinit = create_ϵ_greedy_policy(Qinit, ϵinit), decay_ϵ = false, history_state::S = first(mdp.states), save_history = false, update_policy! = (v, ϵ, s) -> make_ϵ_greedy_policy!(v, ϵ)) where {S, A, F, G, H, T<:AbstractFloat}
	terminds = findall(mdp.isterm(s) for s in mdp.states)
	Q = copy(Qinit)
	Q[:, terminds] .= zero(T)
	π = copy(πinit)
	vhold = zeros(T, length(mdp.actions))
	#keep track of rewards and steps per episode as a proxy for training speed
	rewards = zeros(T, num_episodes)
	steps = zeros(Int64, num_episodes)
	
	if save_history
		history_actions = Vector{A}(undef, num_episodes)
	end
	
	for ep in 1:num_episodes
		ϵ = decay_ϵ ? ϵinit/ep : ϵinit
		s = mdp.state_init()
		rtot = zero(T)
		l = 0
		while !mdp.isterm(s)
			(i_s, i_s′, r, s′, a, i_a) = takestep(mdp, π, s)
			if save_history && (s == history_state)
				history_actions[ep] = a
			end
			qmax = maximum(Q[i, i_s′] for i in eachindex(mdp.actions))
			Q[i_a, i_s] += α*(r + γ*qmax - Q[i_a, i_s])
			
			#update terms for next step
			vhold .= Q[:, i_s]
			update_policy!(vhold, ϵ, s)
			π[:, i_s] .= vhold
			s = s′
			
			l+=1
			rtot += r
		end
		steps[ep] = l
		rewards[ep] = rtot
	end

	save_history && return Q, π, steps, rewards, history_actions
	return Q, π, steps, rewards
end

# ╔═╡ c34678f6-53bb-4f2a-96f0-a7b16f894ddd
function show_gridworld_policy_value(mdp, results; action_display = rook_action_display, policy_display = display_rook_policy)
	Q, π, steps, rewards = results
	policy_display = show_grid_policy(mdp, π, wind_vals, policy_display, String(rand('A':'Z', 10)); action_display = action_display, scale = 0.9)
	value_display = show_grid_value(mdp, Q, wind_vals, String(rand('A':'Z', 10)); action_display = action_display, scale = 0.9)
	path = plot_path(mdp, π)
	@htl("""
	<div style="display:flex; justify-content: center;">	
		$policy_display 
		$value_display
	</div>
	$path
	""")
end

# ╔═╡ 9d01c0ef-6313-4091-b444-3e9765aba90c
md"""
### Windy Gridworld Solutions with Q-Learning
"""

# ╔═╡ 4b1a4c14-3c2b-40c0-995c-cd0334ed8b3a
md"""
#### Normal Actions
"""

# ╔═╡ 897fde24-9a4a-465e-96f2-dd9e8baab294
show_gridworld_policy_value(windy_gridworld, q_learning(windy_gridworld, 0.5f0, 1.0f0; num_episodes = 400))

# ╔═╡ f2776908-d06a-4073-b2ce-ecbf109c9cc7
md"""
#### King Actions
"""

# ╔═╡ 1115f3ec-f4b2-4fba-bd5e-321a63b10a6d
show_gridworld_policy_value(king_gridworld, q_learning(king_gridworld, 0.1f0, 1.0f0; num_episodes = 2000); action_display = king_action_display, policy_display = display_king_policy)

# ╔═╡ c4719c42-87aa-482a-95aa-a1492d42835d
md"""
#### Stochastic Gridworld
"""

# ╔═╡ 1e45a661-c2e1-40c2-b27b-5f80f95efdab
show_gridworld_policy_value(stochastic_gridworld, q_learning(stochastic_gridworld, 0.1f0, 1.0f0; num_episodes = 2000); action_display = king_action_display, policy_display = display_king_policy)

# ╔═╡ 8224b808-5778-458b-b683-ea2603c82117
md"""
### Example 6.6: Cliff Walking
"""

# ╔═╡ 6556dafb-04fa-434c-868a-8d7bb7b5b196
function make_cliffworld(;actions = rook_actions, xmax = 12, ymax = 4, cliff_penalty::T = -100f0, step_reward::T = -1f0) where T<:AbstractFloat
	start = GridworldState(1, 1)
	sinit() = start
	isterm(s) = s == GridworldState(xmax, 1)
	states = [GridworldState(x, y) for x in 1:xmax for y in 1:ymax]

	boundstate(x::Int64, y::Int64) = (clamp(x, 1, xmax), clamp(y, 1, ymax))

	function cliffcheck(s)
		safereturn = (step_reward, s)
		unsafereturn = (cliff_penalty, start)
		s.y > 1 && return safereturn
		(s.x == 1) && return safereturn
		(s.x == xmax) && return safereturn
		unsafereturn
	end
	
	function step(s::GridworldState, a::GridworldAction)
		(x1, y1) = move(a, s.x, s.y)
		(x2, y2) = boundstate(x1, y1)
		cliffcheck(GridworldState(x2, y2))
	end
		
	MDP_TD(states, actions, sinit, step, isterm)
end	

# ╔═╡ 6faa3015-3ac4-44af-a78c-10b175822441
const cliffworld = make_cliffworld()

# ╔═╡ 05664aaf-575b-4249-974c-d8a2e63f380a
md"""
> ### *Exercise 6.11* 
> Why is Q-learning considered an *off-policy* control method?

If we compare to the on-policy update rule, the expected value being calculated at each state action pair should be:

$Q_\pi(S_t, A_t) = \text{E}_\pi [R_{t+1} + \gamma Q_\pi(S_{t+1}, A_{t+1})]$

which we estimate with sampling.  In Q-learning, the expected value being estimated is instead:

$Q_\pi(S_t, A_t) = \text{E}_\pi [R_{t+1} + \gamma \text{max}_a Q_\pi(S_{t+1}, a)]$

Since the behavior policy being used to select the subsequent action taken from state $S_{t+1}$ is $\epsilon$-greedy, there is a probability that the next action will not match the maximizing action.  So the Q-Learning update is computing the optimal greedy state-action value function rather than the optimal $\epsilon$-greedy value function of the behavior policy.  Sarsa, in contrast follows the same policy and computes the value function which matches this policy, thus making it a true on-policy method.
"""

# ╔═╡ 2a3e4617-efbb-4bbc-9c61-8535628e439c
md"""
> ### *Exercise 6.12* 
> Supposed action selection is greedy.  Is Q-learning then exactly the same algorithm as Sarsa?  Will they make exactly the same action selections and weight updates?

Consider both updates when the greedy policy is followed during training.

Sarsa Update:

$Q_\pi(S_t, A_t) = \text{E}_\pi [R_{t+1} + \gamma Q_\pi(S_{t+1}, A_{t+1})]$ with $A_{t+1}$ chosen by the greedy policy accoring to $\text{max}_a Q_\pi(S_{t+1})$ for the estimates prior to this update.

Q-Learning Update:

$Q_\pi(S_t, A_t) = \text{E}_\pi [R_{t+1} + \gamma \text{max}_a Q_\pi(S_{t+1}, a)]$

The value updates are identical since the Q estimate used in both cases will be based on the maximizing action at state $S_{t+1}$.  In the case of Sarsa, $A_{t+1}$ has already been selected prior to this update occurring, so this value update will properly reflect the next step in the trajectory.  In Q-learning, the action selection at $S_{t+1}$ will occur after the update step.  Notice that we only updated $Q_\pi(S_t, A_t)$ and did not touch $Q_\pi(S_{t+1}, A_{t+1})$, so our next action selection should be unaffected by this update.  However, there in one exception for the case where the state is identical through the transition: $S_t = S_{t+1}$.  In this case, the update could actually affect the next action selection, for example, let's say a very low reward was received during the update.  That would lower the estimate for this action selected on step t and it may no longer be maximizing on step t+1.  Then Sarsa would have chosen the same action ahead of the update but Q-learning would chose a different action on the next step even though the state is unchanged.  Despite this difference, both methods are still computing the state-action value function for the optimal policy, but neither is guaranteed to converge to this function due to the violation of the assumption that all state-action pairs are visited during training.
"""

# ╔═╡ 6e06bd39-486f-425a-bbca-bf363b58988c
md"""
## 6.6 Expected Sarsa
Consider the learning algorithm that is just like Q-learning except that intsead of the maximization over next state-action pairs it uses the expected value, taking into account how likely each action is under the current policy.  That is consider the algorithm with the update rule

$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left [ R_{t+1} + \gamma \text{E}_\pi  [Q(S_{t+1}, A_{t+1})|S_{t+1}] - Q(S_t, A_t) \right ]$
$= Q(S_t, A_t) + \alpha \left [ R_{t+1} + \gamma \sum_a \pi(a|S_{t+1})Q(S_{t+1}, a) - Q(S_t, A_t) \right ]$

but that otherwise follows the scheme of Q-learning.  Given the next state, $S_{t+1}$, this algorithm moves *deterministically* in the same direction as Sarsa moves *in expectation*, and accordingly it is called *Expected Sarsa*.  Although more computationally complex than Sarsa, it eliminates the variance due to the random selection of $A_{t+1}$

In general Expected Sarsa might use a policy different from the target policy π to generate behavior in which case it becomes an off-policy algorithm.  For example, supppose π is the greedy policy while behavior is more exploratory; then Expected Sarsa is exactly Q-learning.  In this sense Expected Sarsa subsumes and generalizes Q-learning while reliably improving over Sarsa.

"""

# ╔═╡ 292d9018-b550-4278-a8e0-78dd6a6853f1
function expected_sarsa(mdp::MDP_TD, α::T, γ::T; num_episodes = 1000, qinit = zero(T), ϵinit = one(T)/10, Qinit = initialize_state_action_value(mdp; qinit=qinit), decay_ϵ = false) where T<:AbstractFloat
	terminds = findall(mdp.isterm(s) for s in mdp.states)
	Q = copy(Qinit)
	Q[:, terminds] .= zero(T)
	π = create_ϵ_greedy_policy(Q, ϵinit)
	vhold = zeros(T, length(mdp.actions))
	#keep track of rewards and steps per episode as a proxy for training speed
	rewards = zeros(T, num_episodes)
	steps = zeros(Int64, num_episodes)
	
	for ep in 1:num_episodes
		ϵ = decay_ϵ ? ϵinit/ep : ϵinit
		s = mdp.state_init()
		rtot = zero(T)
		l = 0
		while !mdp.isterm(s)
			(i_s, i_s′, r, s′, a, i_a) = takestep(mdp, π, s)
			q_expected = sum(π[i, i_s′]*Q[i, i_s′] for i in eachindex(mdp.actions))
			Q[i_a, i_s] += α*(r + γ*q_expected - Q[i_a, i_s])
			
			#update terms for next step
			vhold .= Q[:, i_s]
			make_ϵ_greedy_policy!(vhold, ϵ)
			π[:, i_s] .= vhold
			s = s′
			
			l+=1
			rtot += r
		end
		steps[ep] = l
		rewards[ep] = rtot
	end

	return Q, π, steps, rewards
end

# ╔═╡ 047a8881-c2ec-4dd1-8778-e3acf9beba2e
md"""
#### Sarsa vs Q-learning vs Expected Sarsa Performance on Cliff Walking Example
"""

# ╔═╡ c8500b89-644d-407f-881a-bcbd7da23502
md"""
**Figure 6.3** Interim and aymptotic performance shown for TD control methods on cliff-walking task as a function of α.  Dashed lines represent interim performance and solid lines are asymptotic.
"""

# ╔═╡ 6d9ae541-cf8c-4687-9f0a-f008944657e3
function figure_6_3(mdp)
	αlist = 0.1f0:0.05f0:1.0f0
	function generate_data(estimator, nep, nruns)
		out = zeros(length(αlist))
		@threads for i in eachindex(αlist)
			rmean = mean(begin
				α = αlist[i]	
				(Qstar, πstar, steps, rsum) = estimator(mdp, α, 1.0f0; num_episodes = nep, ϵinit = 0.1f0)
				mean(rsum)
				end
			for _ in 1:nruns)
			out[i] = rmean
		end
		return out
	end

	interim_data(estimator) = generate_data(estimator, 100, 100)
	asymp_data(estimator) = generate_data(estimator, 100_000, 1)

	estimators = [expected_sarsa, sarsa, q_learning]
	names = ["Expected Sarsa", "Sarsa", "Q-learning"]

	interim_traces = [scatter(x = αlist, y = interim_data(estimator), name = "Intermim $name", mode = "lines+markers", line = attr(dash = "dash")) for (estimator, name) in zip(estimators, names)]
	asymp_traces = [scatter(x = αlist, y = asymp_data(estimator), name = "Asymptotic $name", mode = "lines+markers", line = attr(dash = "dot")) for (estimator, name) in zip(estimators, names)]
	plot([interim_traces; asymp_traces], Layout(axis_title = "α", yaxis_title = "Sum of rewards per episode", yaxis_range = [-150, 0]))
end

# ╔═╡ cafedde8-be94-4697-a511-510a5fea0155
# ╠═╡ disabled = true
#=╠═╡
figure_6_3(cliffworld)
  ╠═╡ =#

# ╔═╡ 29b0a2d5-9629-46cd-b57c-6f3ef797de66
md"""
## 6.7 Maximization Bias and Double Learning
All the control algorithms that we have discussed so far involve maximization in the construction of the target policies.  For example, in Q-learning the target policy is the greedy policy given the current action values, which is defined with a max, and in Sarsa the policy is often $\epsilon$-greedy, which also involves a maximization operation.  In these algorithms, a maximum over estimated values is used implicitely as an estimate of the maximum value, which can lead to significant positive bias.  To see why, consider a isngle state $s$ where there are many actions $a$ whose true values $q(s, a)$, are all zero, but whose estimated values, $Q(s, a)$, are uncertain and thus distributed above and some below zero.  The maximum of the true values is zero, but the maximum of the estimates is positive, a positive bias.  We call this *maximization bias*.

To elaborate on the bias, consider just two random variables $X \sim \mathcal{N}(\theta_1, 1)$ and $Y \sim \mathcal{N}(\theta_2, 1)$.  We would like to estimate $\text{max}(\theta_1, \theta_2)$ and using the approach analogous to our learning algorithms we would calculate $\text{max} \left ( \mathbb{E}[X], \mathbb{E}[Y] \right )$ which we could calculate with sample averages as follows: $\text{max} \left ( \sum_{i=1}^N \frac{x_i}{N}, \sum_{i=1}^M \frac{y_i}{M} \right )$.  The problem with this approach is that for small numbers of samples, the variance of our estimator is high and we are using this estimator both to select which random variable has the higher expected value and what that value is.  Empirically, this results in a positive bias which gets worse the more variables we are considering as illustrated in the plot below.
"""

# ╔═╡ 4862942b-d1e2-4ac8-8e88-65205e91a070
@bind max_visual_params PlutoUI.combine() do Child
	md"""
	Number of Variables: $(Child(:nvars, NumberField(2:100, default = 2)))
	"""
end |> confirm

# ╔═╡ 15dc58b8-8532-4b5a-b161-591852979ae5
md"""
To understand the origin of the bias, consider a case where we only have a single sample from each variable which follows a standard normal distribution.  In this case our expected value estimate is just this value and it has variance 1.  What is the expected value of our maximum estimate in this case?  

$\mathbb{E} \left [ \text{max}\left( \mathbb{E}[X], \mathbb{E}[Y] \right ) \right ] = \mathbb{E} \left [ \text{max}\left( x, y \right ) \right ]$ where $x,y$ are single samples from the distributions.  This expected value can be calculated using the distribution of the maximum of two standard normal random variables:  

$\mathbb{E}\left [ \text{max}(\mathcal{N}(0, 1), \mathcal{N}(0, 1)) \right ] = \frac{1}{\sqrt{\pi}} \approx 0.564$

So apparantly our estimate has a positive bias.  If we had more samples for each variable then we would use the distribution of the sample average rather than a single sample and that distribution has a variance proportional to the inverse of the number of smaples.  So the bias will converge to zero in the limit of infinite samples, and in the graph the bias does in fact converge to zero over more samples.  Could we instead use an estimator that avoids some or all of this initial bias?

Consider a multivariate random variable $\boldsymbol{X} = [X_1, X_2, \cdots, X_N]$ which has some true expected value $\mathbb{E}[\boldsymbol{X}] = [\theta_1, \theta_2, \cdots, \theta_N]$.  We would like to know the true value for $\text{maximum}(\mathbb{E}[\boldsymbol{X}]) = \theta_{i_{max}} \geq \theta_i \forall i$.  If we take the maximum of the sample average, this is not an unbiased estimator $\text{maximum} ( \overline{\boldsymbol{X}})$ since the sample averages themselves have non-zero variance.  $\mathbb{E}[\boldsymbol{X}_{i_{max}}] = \sum_{i=1}^N \left [ P\{i = i_{max} \} \times \overline{\boldsymbol{X}}_i \right ]$ where $i_{max}$ is defined above but not known to us at the time of estimation.  We do not know apriori though what these probabilities are.  So an alternative solution is to lower the bias of our estimators some other way.  One suggestion from the book is to use one set of samples to estimate which variable is maximizing and then another set of samples to estimate the average of that variable.  On the plot is also shown a method where the samples are separated into two halves.  One half is used to estimate which variable is maximizing and the other half is used to calculate the maximum.  The process is reversed as well and the two estimates are averaged for the final answer.  This method completely removes the bias and all of the estiamtes have an expected value of 0 regardless of how few samples there are.    
"""

# ╔═╡ 3f4f078a-9fc4-4b02-b499-a805fd5f1071
function max_bias_visualization(;nvars = 2, nmax = 100, nruns = 10_000)
	nlist = collect(2:2:nmax)
	vars = [randn(nmax, nruns) for _ in 1:nvars]
	max_estimate = [begin
	mapreduce(j -> begin
	means1 = [mean(view(x, 1:2:n, j)) for x in vars]
	means2 = [mean(view(x, 2:2:n, j)) for x in vars]
	max1 = maximum(means1 .+ means2) / 2
	max2 = (means2[argmax(means1)] + means1[argmax(means2)]) / 2
	return (max1, max2)
	end, (a, b) -> (a[1]+b[1], a[2]+b[2]), 1:nruns) 
	end
	for n in nlist]
	estimate1 = [a[1] for a in max_estimate] ./ (nruns .* nlist)
	estimate2 = [a[2] for a in max_estimate] ./ (nruns .* nlist)
	t1 = scatter(x = 2:2:nmax, y = estimate1, name = "Max of Means Estimate")
	t2 = scatter(x = 2:2:nmax, y = estimate2, name = "Double Max Estimate")
	plot([t1, t2], Layout(xaxis_title = "Number of Samples Per Variable", yaxis_title = "Estimate of Maximum Mean", title = "Maximization Bias for $nvars Variables with Zero Mean"))
end

# ╔═╡ ff5d051e-5de1-48a9-9578-5dbafd71afd1
max_bias_visualization(;nvars = max_visual_params.nvars, nmax = 30)

# ╔═╡ e039a5be-4b59-4023-be97-2d1de970be27
md"""
### Double Learning Implementation
"""

# ╔═╡ 223055df-7d5c-4d99-bc8d-fbc9702f906f
md"""
### Example 6.7: Maximization Bias Example

Consider an MDP with two non-terminal states A and B.  Episodes always start in state A and there are two actions, left and right.  Choosing right will always result in a reward of 0 and the episode terminating.  Choosing left will transition into state B from which there are many actions, all of which result in a terminal transition with random rewards.  The distribution of rewards for each of these actions is $\mathcal{N}(-0.1, 1)$.  The estimated value of (A, right) will always be 0 since that is the only possible sample to be collected.  The estimated value of (A, left) however will have higher variance but an expected value of -0.1.  The problem with Q-learning is that, due to the maximization bias, (A, left) will have a higher value estimate when few samples have been collected since it is very likely that one of the state-action pairs from B will produce a reward greater than 0.  The more of these actions exist, the worse the bias and the more samples needed to be collected to remove it.  If we employ Double Q-learning instead, however, we can eliminate the bias completely.
"""

# ╔═╡ 4382928c-6325-4ecd-b7cf-282525a270ab
begin
	abstract type MaxBiasStates end
	struct A <: MaxBiasStates end
	struct B <: MaxBiasStates end
	struct Term <: MaxBiasStates end
end

# ╔═╡ 42799973-9884-4a0e-b29a-039890e92d21
md"""
> ### *Exercise 6.13* 
> What are the update equations for Double Expected Sarsa with an ϵ-greedy target policy?

For Q-learning the action-value update equation is:

$Q(S_t, A_t) = Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \text{max}_a Q(S_{t+1}, a) - Q(S_t, A_t)]$

For expected Sarsa the action-value update equation is:

$Q(S_t, A_t) = Q(S_t, A_t) + \alpha [ R_{t+1} + \gamma \sum_a \pi(a|S_{t+1})Q(S_{t+1}, a) - Q(S_t, A_t)]$

For double Q-learning, the twin action-value update equations are:

$Q_1(S_t, A_t) = Q_1(S_t, A_t) + \alpha [ R_{t+1} + \gamma Q_2(S_{t+1}, \text{argmax}_a Q_1(S_{t+1}, a)) - Q_1(S_t, A_t)]$

$Q_2(S_t, A_t) = Q_2(S_t, A_t) + \alpha [ R_{t+1} + \gamma Q_1(S_{t+1}, \text{argmax}_a Q_2(S_{t+1}, a)) - Q_2(S_t, A_t)]$

For double expected sarsa, we have two action-value estimates like in Double Q-learining, but the bootstrap calculation is an expected value calculation using each value function's target policy.  In this case that target is the $\epsilon$-greedy policy rather than the greedy policy in Q-learning.  The expected value uses the probabilities from the matching value function but the values from the other one:

With 50% probability:

$Q_1(S_t, A_t) = Q_1(S_t, A_t) + \alpha [ R_{t+1} + \gamma \sum_a \pi_1(a|S_{t+1}) Q_2(S_{t+1}, a) - Q_1(S_t, A_t)]$ 

and make $\pi_1$ $\epsilon$-greedy with respect to $Q_1$

With 50% probability:

$Q_2(S_t, A_t) = Q_2(S_t, A_t) + \alpha [ R_{t+1} + \gamma \sum_a \pi_2(a|S_{t+1}) Q_1(S_{t+1}, a) - Q_2(S_t, A_t)]$

and make $\pi_2$ $\epsilon$-greedy with respect to $Q_2$

"""

# ╔═╡ 35dc0d94-145a-4292-b0df-9e84a286c036
md"""
## 6.8 Games, Afterstates, and Other Special Cases 
"""

# ╔═╡ f95ceb98-f12e-4650-9ad3-0609b7ecd0f3
md"""
> *Exercise 6.14* Describe how the task of Jack's Car Rental (Example 4.2) could be reformulated in terms of afterstates.  Why, in terms of this specific task, would such a reformulation be likely to speed convergence?

In the original problem the state is the number of cars at each location at the end of the day.  The actions are the net numbers of cars moved between the two locations overnight.  With an afterstate approach, the value function would only consider the number of cars after the movement is performed.  This would be equivalent to valuing the state the following morning when customers begin to return and rent new cars.

The random processes that occur the following day will have a good/bad outcome based on the cars available at each location at the start of the day.  This approach would likely converge faster because we are only modeling the value of the state that is directly related to whether or not cars will be available.  Similar to the tic-tac-toe example, many actions will result in the same afterstate, but equivalent afterstates should have the same value.
"""

# ╔═╡ f36822d7-9ea8-4f5c-9925-dc2a466a68ba
md"""
# Dependencies and Settings
"""

# ╔═╡ 14b456f9-5fd1-4340-a3c7-ab9b91b4e3e0
html"""
	<style>
		main {
			margin: 0 auto;
			max-width: min(2000px, 90%);
	    	padding-left: max(10px, 5%);
	    	padding-right: max(10px, 5%);
			font-size: max(10px, min(24px, 2vw));
		}
	</style>
	"""

# ╔═╡ 22c4ce8c-bd82-4eb3-8af5-55342018edff
md"""
# Dynamic Programming Code
"""

# ╔═╡ 393cd9d2-dd97-496e-b260-ec6e8b1c13b5
begin
	struct FiniteMDP{T<:Real, S, A} 
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

# ╔═╡ 0748902c-ffc0-4634-9a1b-e642b3dfb77b
#forms a random policy for a generic finite state mdp.  The policy is a matrix where the rows represent actions and the columns represent states.  Each column is a probability distribution of actions over that state.
form_random_policy(mdp::FiniteMDP{T, S, A}) where {T, S, A} = ones(T, length(mdp.actions), length(mdp.states)) ./ length(mdp.actions)

# ╔═╡ c4919d14-8cba-43e6-9369-efc52bcb9b23
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

# ╔═╡ 84a71bf8-0d66-42cd-ac7b-589d63a16eda
function create_greedy_policy(Q::Matrix{T}; c = 1000, π = copy(Q)) where T<:Real
	vhold = zeros(T, size(Q, 1))
	for j in 1:size(Q, 2)
		vhold .= Q[:, j]
		make_greedy_policy!(vhold; c = c)
		π[:, j] .= vhold
	end
	return π
end

# ╔═╡ 6bffb08c-704a-4b7c-bfce-b3d099cf35c0
function gridworld_Q_vs_sarsa_solve(mdp; α=0.5f0, ϵ=0.1f0, num_episodes = 500, nruns = 100)
	function addtuple(t1, t2)
		Tuple(t1[i] .+ t2[i] for i in eachindex(t1))
	end
	
	sarsa_results = mapreduce(addtuple, 1:nruns) do _
		sarsa(mdp, α, 1.0f0; num_episodes = num_episodes, ϵinit = ϵ)
	end

	qlearning_results = mapreduce(addtuple, 1:nruns) do _
		q_learning(mdp, α, 1.0f0; num_episodes = num_episodes, ϵinit = ϵ)
	end
	
	# qlearning_results = [q_learning(mdp, α, 1.0f0; num_episodes = num_episodes, ϵinit = ϵ) for _ in 1:nruns]
	p1 = plot_path(mdp, create_greedy_policy(sarsa_results[1] ./ nruns); windtext = fill("", 12), xtitle = "", title = "Cliff Walking Sarsa Path")
	p2 = plot_path(mdp, qlearning_results[2] ./ nruns; windtext = fill("", 12), xtitle = "", title = "Cliff Walking Q Learning Path")

	traces = [scatter(x = 1:num_episodes, y = results[4] ./ nruns, name = name) for (results, name) in zip([sarsa_results, qlearning_results], ["Sarsa", "Q-learning"])]
	p3 = plot(traces, Layout(xaxis_title = "Episodes", yaxis = attr(title = "Sum of rewards during episode", range = [-100, -15])))

	p3 = plot(traces, Layout(xaxis_title = "Episodes", yaxis = attr(title = "Sum of rewards during episode", range = [-100, -15])))

	steptraces = [scatter(x = 1:num_episodes, y = results[3] ./ nruns, name = name) for (results, name) in zip([sarsa_results, qlearning_results], ["Sarsa", "Q-learning"])]
	p4 = plot(steptraces, Layout(xaxis_title = "Episodes", yaxis = attr(title = "Average steps per episode <br> during training", range = [0, 100])))
	
	@htl("""
	<div style = "display: flex; background-color: white;">
	$p1
	$p2
	</div>
	$p3
	$p4
	""")
end

# ╔═╡ a4c4d5f2-d76d-425e-b8c9-9047fe53c4f0
gridworld_Q_vs_sarsa_solve(cliffworld)

# ╔═╡ 84584793-8274-4aa1-854f-b167c7434548
function gridworld_Q_vs_sarsa_vs_expected_sarsa_solve(mdp; α=0.5f0, ϵ=0.1f0, num_episodes = 500, nruns = 100)
	function addtuple(t1, t2)
		Tuple(t1[i] .+ t2[i] for i in eachindex(t1))
	end
	
	sarsa_results = mapreduce(addtuple, 1:nruns) do _
		sarsa(mdp, α, 1.0f0; num_episodes = num_episodes, ϵinit = ϵ)
	end

	qlearning_results = mapreduce(addtuple, 1:nruns) do _
		q_learning(mdp, α, 1.0f0; num_episodes = num_episodes, ϵinit = ϵ)
	end

	expected_sarsa_results = mapreduce(addtuple, 1:nruns) do _
		expected_sarsa(mdp, α, 1.0f0; num_episodes = num_episodes, ϵinit = ϵ)
	end
	
	# qlearning_results = [q_learning(mdp, α, 1.0f0; num_episodes = num_episodes, ϵinit = ϵ) for _ in 1:nruns]
	p1 = plot_path(mdp, create_greedy_policy(sarsa_results[1] ./ nruns); windtext = fill("", 12), xtitle = "", title = "Cliff Walking Sarsa Path")
	p2 = plot_path(mdp, qlearning_results[2] ./ nruns; windtext = fill("", 12), xtitle = "", title = "Cliff Walking Q Learning Path")
	expected_sarsa_path = plot_path(mdp, create_greedy_policy(expected_sarsa_results[1] ./ nruns); windtext = fill("", 12), xtitle = "", title = "Cliff Walking Expected Sarsa Path")

	traces = [scatter(x = 1:num_episodes, y = results[4] ./ nruns, name = name) for (results, name) in zip([sarsa_results, qlearning_results, expected_sarsa_results], ["Sarsa", "Q-learning", "Expected Sarsa"])]
	p3 = plot(traces, Layout(xaxis_title = "Episodes", yaxis = attr(title = "Sum of rewards during episode", range = [-100, -15])))

	p3 = plot(traces, Layout(xaxis_title = "Episodes", yaxis = attr(title = "Sum of rewards during episode", range = [-100, -15])))

	steptraces = [scatter(x = 1:num_episodes, y = results[3] ./ nruns, name = name) for (results, name) in zip([sarsa_results, qlearning_results, expected_sarsa_results], ["Sarsa", "Q-learning", "Expected Sarsa"])]
	p4 = plot(steptraces, Layout(xaxis_title = "Episodes", yaxis = attr(title = "Average steps per episode <br> during training", range = [0, 100])))
	
	@htl("""
	<div style = "display: flex; background-color: white; flex-wrap: wrap;">
	<div>$p1</div>
	<div>$p2</div>
	<div>$expected_sarsa_path</div>
	</div>
	$p3
	$p4
	"""
	)
end

# ╔═╡ 667666b9-3ab6-4836-953d-9878208103c9
gridworld_Q_vs_sarsa_vs_expected_sarsa_solve(cliffworld)

# ╔═╡ 3756a3f8-18e8-4d62-afa1-cfeb4183820c
function double_expected_sarsa(mdp::MDP_TD{S, A, F, G, H}, α::T, γ::T; num_episodes = 1000, qinit = zero(T), ϵinit = one(T)/10, Qinit::Matrix{T} = initialize_state_action_value(mdp; qinit=qinit), decay_ϵ = false, target_policy_function! = (v, ϵ, s) -> make_ϵ_greedy_policy!(v, ϵ), behavior_policy_function! = (v, ϵ, s) -> make_ϵ_greedy_policy!(v, ϵ), πinit_target::Matrix{T} = create_ϵ_greedy_policy(Qinit, ϵinit), πinit_behavior::Matrix{T} = create_ϵ_greedy_policy(Qinit, ϵinit), save_state::S = first(mdp.states), save_history = false) where {S, A, F, G, H, T<:AbstractFloat}
	terminds = findall(mdp.isterm(s) for s in mdp.states)
	
	Q1 = copy(Qinit)
	Q2 = copy(Qinit) 
	Q1[:, terminds] .= zero(T)
	Q2[:, terminds] .= zero(T)
	π_target1 = copy(πinit_target)
	π_target2 = copy(πinit_target)
	π_behavior = copy(πinit_behavior)
	vhold1 = zeros(T, length(mdp.actions))
	vhold2 = zeros(T, length(mdp.actions))
	vhold3 = zeros(T, length(mdp.actions))
	#keep track of rewards and steps per episode as a proxy for training speed
	rewards = zeros(T, num_episodes)
	steps = zeros(Int64, num_episodes)

	if save_history
		action_history = Vector{A}(undef, num_episodes)
	end
	
	for ep in 1:num_episodes
		ϵ = decay_ϵ ? ϵinit/ep : ϵinit
		s = mdp.state_init()
		rtot = zero(T)
		l = 0
		while !mdp.isterm(s)
			
			(i_s, i_s′, r, s′, a, i_a) = takestep(mdp, π_behavior, s)
			if save_history && (s == save_state)
				action_history[ep] = a
			end
			
			# q_expected = sum(π_target[i, i_s′]*(Q1[i, i_s′]*toggle + Q2[i, i_s′]*(1-toggle)) for i in eachindex(mdp.actions))
			toggle = rand() < 0.5
			q_expected = if toggle 
				sum(π_target2[i, i_s′]*Q1[i, i_s′] for i in eachindex(mdp.actions))
			else
				sum(π_target1[i, i_s′]*Q2[i, i_s′] for i in eachindex(mdp.actions))
			end

			if toggle
				Q2[i_a, i_s] += α*(r + γ*q_expected - Q2[i_a, i_s])
			else
				Q1[i_a, i_s] += α*(r + γ*q_expected - Q1[i_a, i_s])
			end
			
			#update terms for next step
			if toggle
				vhold2 .= Q2[:, i_s]
				target_policy_function!(vhold2, ϵ, s)
				π_target2[:, i_s] .= vhold2
			else
				vhold1 .= Q1[:, i_s]
				target_policy_function!(vhold1, ϵ, s)
				π_target1[:, i_s] .= vhold1
			end
			
			vhold3 .= vhold1 .+ vhold2
			behavior_policy_function!(vhold3, ϵ, s)
			π_behavior[:, i_s] .= vhold3
			
			s = s′

			l+=1
			rtot += r
		end
		steps[ep] = l
		rewards[ep] = rtot
	end

	Q1 .+= Q2
	Q1 ./= 2
	plain_return = Q1, create_greedy_policy(Q1), steps, rewards

	save_history && return (plain_return..., action_history)
	return plain_return
end

# ╔═╡ d526a3a4-63cc-4f94-8f55-98c9a4a9d134
function double_q_learning(mdp::MDP_TD{S, A, F, G, H}, α::T, γ::T; 
	num_episodes = 1000, 
	qinit = zero(T), 
	ϵinit = one(T)/10, 
	Qinit::Matrix{T} = initialize_state_action_value(mdp; qinit=qinit), 
	decay_ϵ = false, 
	target_policy_function! = (v, ϵ, s) -> make_greedy_policy!(v), 
	behavior_policy_function! = (v, ϵ, s) -> make_ϵ_greedy_policy!(v, ϵ), 
	πinit_target::Matrix{T} = create_greedy_policy(Qinit), 
	πinit_behavior::Matrix{T} = create_ϵ_greedy_policy(Qinit, ϵinit), 
	save_state::S = first(mdp.states), 
	save_history = false) where {S, A, F, G, H, T<:AbstractFloat} 
	
	double_expected_sarsa(mdp, α, γ; num_episodes = num_episodes, qinit = qinit, ϵinit = ϵinit, Qinit = Qinit, decay_ϵ = decay_ϵ, target_policy_function! = target_policy_function!, behavior_policy_function! = behavior_policy_function!, πinit_target = πinit_target, πinit_behavior = πinit_behavior, save_state = save_state, save_history = save_history)
end

# ╔═╡ 69eedbfd-396f-4461-b7a1-c36abc094581
function example_6_7_mdp(;num_actions::Integer = 10, num_episodes = 300, nruns = 10_000, α = 0.1f0, ϵ = 0.1f0)
	states = [A(), B(), Term()]
	actions = collect(1:num_actions)
	function step(::A, a)
		a == 1 && return (0.0f0, B())
		a == 2 && return (0.0f0, Term())
		return (-100f0, Term())
	end

	step(::B, a) = (randn(Float32) - 0.1f0, Term())
	state_init() = A()
	
	isterm(::Term) = true
	isterm(s) = false

	mdp = MDP_TD(states, actions, state_init, step, isterm)

	function get_valid_inds(i_s)
		i_s == 1 && return 1:2
		return 1:num_actions
	end

	update_behavior!(v, ϵ, ::A) = make_ϵ_greedy_policy!(v, ϵ; valid_inds = 1:2)
	update_behavior!(v, ϵ, s) = make_ϵ_greedy_policy!(v, ϵ)

	Qinit = [[[0.0f0, 0.0f0]; fill(-100f0, num_actions-2)] zeros(Float32, num_actions) zeros(Float32, num_actions)]
	πinit = create_ϵ_greedy_policy(Qinit, ϵ; get_valid_inds = get_valid_inds)
	
	
	
	# sarsa_results = mean(last(sarsa(mdp, 0.1f0, 1.0f0; num_episodes = num_episodes, save_history = true, ϵinit = ϵ, Qinit = Qinit, πinit = πinit, update_policy! = update_behavior!)) .== 1 for _ in 1:nruns)
	
	q_learning_results = mean(last(q_learning(mdp, 0.1f0, 1.0f0; num_episodes = num_episodes, save_history = true, ϵinit = ϵ, Qinit = Qinit, πinit = πinit, update_policy! = update_behavior!)) .== 1 for _ in 1:nruns)
	
	
	double_q_learning_results = mean(last(double_q_learning(mdp, 0.1f0, 1.0f0; num_episodes = num_episodes, save_history = true, ϵinit = ϵ, Qinit = Qinit, πinit_behavior = πinit, behavior_policy_function! = update_behavior!)) .== 1 for _ in 1:nruns)
	
	# double_expected_sarsa_results = mean(last(double_expected_sarsa(mdp, 0.1f0, 1.0f0; ϵinit = ϵ, num_episodes = num_episodes, save_history = true, Qinit = Qinit, πinit_behavior = πinit, behavior_policy_function! = update_behavior!)) .== 1 for _ in 1:nruns)

	optimal_trace = scatter(x = 1:num_episodes, y = fill(ϵ / 2, num_episodes), name = "optimal", line_dash = "dash")

	# t0 = scatter(x = 1:num_episodes, y = sarsa_results, name = "Sarsa")
	t1 = scatter(x = 1:num_episodes, y = q_learning_results, name = "Q-learning")
	t2 = scatter(x = 1:num_episodes, y = double_q_learning_results, name = "Double Q-learning")
	# t3 = scatter(x = 1:num_episodes, y = double_expected_sarsa_results, name = "Double Expected Sarsa")
	# plot([t0, t1, t2, t3])
	traces = [t1, t2, optimal_trace]
	plot(traces, Layout(xaxis_title = "Episodes", yaxis_title = "% left actions from A"))
end

# ╔═╡ 00d67a93-437c-4cda-899a-9daa1102e1f2
example_6_7_mdp(;num_episodes = 300, nruns = 10_000, num_actions = 10)

# ╔═╡ 95245673-2c29-401e-bb4b-a39dc8172297
function create_gridworld_mdp(width, height, start, goal, wind, actions, step_reward)
	mdp = make_windy_gridworld(;actions = actions, apply_wind = apply_wind, sterm = goal, start = start, xmax = width, ymax = height, winds = wind_vals, step_reward = step_reward)
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

# ╔═╡ 07c57f37-22be-4c39-8279-d80addcea0c5
function create_stochastic_gridworld_mdp(width, height, start, goal, wind, actions, step_reward)
	mdp = make_windy_gridworld(;actions = actions, apply_wind = apply_wind, sterm = goal, start = start, xmax = width, ymax = height, winds = wind_vals, step_reward = step_reward)
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
				if w == 0
					ptf[i_s′, 2, i_a, i_s] = 1.0f0
				else #with stochastic wind split the probabilities between the possible outcomes
					ptf[i_s′, 2, i_a, i_s] += Float32(1/3)
					s′2 = GridworldState(s′.x, min(height, s′.y + 1))
					i_s′2 = mdp.statelookup[s′2]
					ptf[i_s′2, 2, i_a, i_s] += Float32(1/3)
					s′3 = GridworldState(s′.x, max(1, s′.y - 1))
					i_s′3 = mdp.statelookup[s′3]
					ptf[i_s′3, 2, i_a, i_s] += Float32(1/3)
				end
			end
		end
	end
			
	FiniteMDP(mdp.states, mdp.actions, [0.0f0, step_reward], ptf)	
end

# ╔═╡ 7ac99619-5232-4db8-8553-d79ea5415d29
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

# ╔═╡ 71774d5f-7841-403f-bc6b-1a0cbbb72d6d
const windy_gridworld_mdp_dp = create_gridworld_mdp(10, 7, GridworldState(1, 4), GridworldState(8, 4), wind_vals, rook_actions, -1.0f0)

# ╔═╡ 2f4e2da2-b1a1-41b1-8904-39b59f426da4
const king_gridworld_mdp_dp = create_gridworld_mdp(10, 7, GridworldState(1, 4), GridworldState(8, 4), wind_vals, king_actions, -1.0f0)

# ╔═╡ 0e488135-49e5-4e71-83b1-05d8e61f0510
const kingplus_gridworld_mdp_dp = create_gridworld_mdp(10, 7, GridworldState(1, 4), GridworldState(8, 4), wind_vals, [king_actions; Stay()], -1.0f0)

# ╔═╡ 8e15f4b5-0dc7-47a5-9477-9f4d8807b331
const stochastic_gridworld_mdp_dp = create_stochastic_gridworld_mdp(10, 7, GridworldState(1, 4), GridworldState(8, 4), wind_vals, king_actions, -1.0f0)

# ╔═╡ dea61907-d4fb-492d-b2bb-c037c7f785cb
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

# ╔═╡ 45213165-11c8-40fe-acee-99331852c2a7
function value_iteration_v(θ::Real, mdp::NamedTuple, γ::Real, V::Dict, delt::Real, nmax::Real, valuelist)
	(p, sa_keys) = mdp
	if nmax <= 0 || delt <= θ
		(πstar, πraw) = calculatepolicy(mdp, γ, V)
		return (valuelist, πstar, πraw)
	else 
		newV = deepcopy(V)
		delt = bellman_optimal_value!(newV, p, sa_keys, γ)
		value_iteration_v(θ, mdp, γ, newV, delt, nmax - 1, vcat(valuelist, newV))	
	end
end

# ╔═╡ 8787a5fd-d0ab-46b5-a7df-e7bc103a7378
function value_iteration_v!(V::Vector{T}, θ::Real, mdp::FiniteMDP{T, S, A}, γ::T, nmax::Real, valuelist) where {T<:Real, S, A}
	nmax <= 0 && return valuelist
	
	#update value function
	delt = bellman_optimal_value!(V, mdp, γ)
	
	#add copy of value function to results list
	push!(valuelist, copy(V))

	#halt when value function is no longer changing
	delt <= θ && return valuelist
	
	value_iteration_v!(V, θ, mdp, γ, nmax - 1, valuelist)	
end

# ╔═╡ 4019c974-dcaa-46c8-ac90-e6566a376ea1
function begin_value_iteration_v(mdp::FiniteMDP{T,S,A}, γ::T, V::Vector{T}; θ = eps(0.0), nmax=typemax(Int64)) where {T<:Real,S,A}
	valuelist = [copy(V)]
	value_iteration_v!(V, θ, mdp, γ, nmax, valuelist)

	π = form_random_policy(mdp)
	make_greedy_policy!(π, mdp, V, γ)
	return (valuelist, π)
end

# ╔═╡ 3134e913-1e86-495d-a558-c3ec4828bf7b
begin_value_iteration_v(mdp::FiniteMDP{T,S,A}, γ::T; Vinit::T = zero(T), kwargs...) where {T<:Real,S,A} = begin_value_iteration_v(mdp, γ, Vinit .* ones(T, size(mdp.ptf, 1)); kwargs...)

# ╔═╡ d299d800-a64e-4ba2-9603-efa833343405
function example_6_5(;mdp = windy_gridworld, num_episodes = 170, action_display = rook_action_display, policy_display = display_rook_policy, use_stochastic_dp=false)
	(Qstar, πstar, steps, rewards) = sarsa(mdp, 0.5f0, 1.0f0; ϵinit = 0.1f0, num_episodes = num_episodes, decay_ϵ = false)
	# eg = runepisode(mdp, create_greedy_policy(Qstar))
	eg = runepisode(mdp, πstar; max_steps = 100_000)

	mdp_dp = use_stochastic_dp ? stochastic_gridworld_mdp_dp : create_gridworld_mdp(mdp, -1.0f0)
	v_dp, π_dp = begin_value_iteration_v(mdp_dp, 1.0f0)
	path_dp = plot_path(mdp, π_dp; title = "Value Iteration Policy <br> Path Example")

	policy_display_dp = show_grid_policy(mdp, π_dp, wind_vals, policy_display, String(rand('A':'Z', 10)); action_display = action_display, scale = 1.0)
	value_display_dp = show_grid_value(mdp, v_dp[end], wind_vals, String(rand('A':'Z', 10)); action_display = action_display, scale = 1.0)
	
	start_trace = scatter(x = [1.5], y = [4.5], mode = "text", text = ["S"], textposition = "left", showlegend=false)
	finish_trace = scatter(x = [8.5], y = [4.5], mode = "text", text = ["G"], textposition = "left", showlegend=false)
	path_traces = [scatter(x = [eg[1][i].x + 0.5, eg[1][i+1].x + 0.5], y = [eg[1][i].y + 0.5, eg[1][i+1].y + 0.5], line_color = "blue", mode = "lines", showlegend=false, name = "Optimal Path") for i in 1:length(eg[1])-1]
	finalpath = scatter(x = [eg[1][end].x + 0.5, 8.5], y = [eg[1][end].y + 0.5, 4.5], line_color = "blue", mode = "lines", showlegend=false, name = "Optimal Path")
	p1 = plot(scatter(x = cumsum(steps), y = 1:num_episodes, line_color = "red"), Layout(xaxis_title = "Time steps", yaxis_title = "Episodes"))
	
	p2 = plot([start_trace; finish_trace; path_traces; finalpath], Layout(xaxis = attr(showgrid = true, showline = true, gridwith = 1, gridcolor = "black", zeroline = true, linecolor = "black", mirror=true, tickvals = 1:10, ticktext = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0], range = [1, 11], title = "Wind Values"), yaxis = attr(linecolor="black", mirror = true, gridcolor = "black", showgrid = true, gridwidth = 1, showline = true, tickvals = 1:7, ticktext = fill("", 7), range = [1, 8]), width = 300, height = 210, autosize = false, padding=0, paper_bgcolor = "rgba(0, 0, 0, 0)", title = attr(text = "Sarsa policy <br> Path Example", font_size = 14, x = 0.5)))
	
	p3 = plot(scatter(x = 1:num_episodes, y = steps), Layout(xaxis_title = "Time steps", yaxis_title = "Steps Per Episode", yaxis_type = "log"))

	policy_display = show_grid_policy(mdp, πstar, wind_vals, policy_display, String(rand('A':'Z', 10)); action_display = action_display, scale = 1.0)
	value_display = show_grid_value(mdp, Qstar, wind_vals, String(rand('A':'Z', 10)); action_display = action_display, scale = 1.0)

	
	return @htl("""
	<div style="background-color: white; color: black;">
	<div>
	$p1
	<div style = "position: absolute; top: 0px; left: 5%;">$p2</div>
	<div style = "position: absolute; top: 0px; left: 30%;">$path_dp</div>
	</div>
	$p3

	Sarsa Solution
	<div style = "display: flex; background-color: white; justify-content: space-around; transform: scale(0.85);">
	$policy_display
	$value_display
	</div>
	Value Iteration Solution
	<div style = "display: flex; background-color: white; justify-content: space-around; transform: scale(0.85);">
	$policy_display_dp
	$value_display_dp
	</div>
	</div>
	""")
end

# ╔═╡ 04a0be81-ee5f-4eeb-963a-ad930392d50b
example_6_5()

# ╔═╡ f0f9d3d5-e76a-4472-bfb1-da29d73a7916
example_6_5(;mdp = king_gridworld, num_episodes = 400, action_display = king_action_display, policy_display = display_king_policy)

# ╔═╡ dee6b500-0ba1-4bbc-b217-cbb9ad47ad06
example_6_5(;mdp = make_windy_gridworld(actions = [king_actions; Stay()]), num_episodes = 400, action_display = action3_display, policy_display = display_king_policy)

# ╔═╡ ed4e863b-22dd-4d2b-88d0-b3a56d6713b7
example_6_5(;mdp = stochastic_gridworld, num_episodes = 400, action_display = king_action_display, policy_display = display_king_policy, use_stochastic_dp=true)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
BenchmarkTools = "~1.4.0"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.3.1"
PlutoPlotly = "~0.4.4"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.55"
StatsBase = "~0.34.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0"
manifest_format = "2.0"
project_hash = "fe2252dfe7eee84e810c7f5f83aff791b8218574"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "c278dfab760520b8bb7e9511b968bf4ba38b7acc"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.3"

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
git-tree-sha1 = "4b41ad09c2307d5f24e36cd6f92eb41b218af22c"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.2.1"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1f03a9fa24271160ed7e73051fba3c1a759b53f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.4.0"

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
git-tree-sha1 = "75bd5b6fc5089df449b5d35fa501c846c9b6549b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.12.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "ac67408d9ddf207de5cfa9a97e114352430f01ed"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.16"

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
git-tree-sha1 = "c5c28c245101bd59154f649e19b038d15901b5dc"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.2"

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
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

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
version = "0.3.23+2"

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
git-tree-sha1 = "58dcb661ba1e58a13c7adce77435c3c6ac530ef9"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.4.4"

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
git-tree-sha1 = "68723afdb616445c6caaef6255067a8339f91325"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.55"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

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
# ╟─814d89be-cfdf-11ec-3295-49a8f302bbcf
# ╟─495f5606-0567-47ad-a266-d21320eecfc6
# ╟─410abe1d-04a6-4434-9abf-0d29dd6498e6
# ╠═7a5ff8f7-70d4-46f1-a4a7-bbfcec4f6e3f
# ╠═834e5810-77ea-4dfd-9f37-9d9dbf6585a4
# ╠═3e767962-7339-4f35-a039-b5521a098ed5
# ╠═8e34202a-f841-4464-9017-cd50194f7987
# ╠═401831c3-3925-465c-a093-28686f0dad2e
# ╠═c5d32889-634b-4b00-8ba7-0d1ecaf94f05
# ╠═24a441c8-7aaf-4642-b245-5e1201456d67
# ╠═d5abd922-a8c2-4f5c-9a6e-d2490a8ad7dc
# ╠═bfe71b40-3157-47df-8494-67f8eb8e4e93
# ╠═7035c082-6e50-4df5-919f-5f09d2011b4a
# ╠═eb735ead-978b-409c-8990-b5fa7a027ebf
# ╠═415ea466-2038-48fe-9d24-39a90182f1eb
# ╟─a0d2333f-e87b-4981-bb52-d436ec6481c1
# ╟─3b16cbb7-f859-4871-9a63-8b40eb4191be
# ╟─d4e39164-9833-4deb-84ca-22f49a1c33d8
# ╟─c93ed1f2-3c38-4f68-8bf8-2cdf4e7bee34
# ╟─1e3b3234-3fe1-46c9-82b7-f729c656eb25
# ╟─c09530bc-f37e-4d57-a267-14d4027147da
# ╟─b5187232-d808-49b6-9f7e-a4cbeb6c2b3e
# ╟─5f32fed0-c921-4cbb-85fe-ade54d4c6c95
# ╠═bc8bad61-a49a-47d6-8fa6-7dcf6c221910
# ╟─6edb550d-5c9f-4ea6-8746-6632806df11e
# ╟─0f22e85f-ed31-49df-a7c7-0579298f05fe
# ╟─9017093c-a9c3-40ea-a9c6-881ee62fc379
# ╟─5290ae65-6f56-4849-a842-fe347315c6dc
# ╟─47c2cbdd-f6db-4ce5-bae2-8141f30aacbc
# ╟─5455fc97-55cb-4b0e-a3be-9433ccc96fc0
# ╟─12c5efe4-d64d-4b82-877c-29b0e537fee6
# ╟─53145cc2-784c-468b-8e91-9bb7866db218
# ╟─54d97122-2d01-46ec-aafe-00bfc9f2d6d1
# ╟─e4c6456c-867d-4ade-a3c8-310c1e065f14
# ╟─9db7a268-1e6d-4366-a0ec-ebf54916d3b0
# ╟─a9dda9b5-f568-481c-9e8f-9bb887468775
# ╠═846720cc-550a-4a3c-a80e-40b99671f4e2
# ╠═4ddcd409-c31c-444c-8fcf-7cc45b68d93b
# ╠═4b0d96d0-25d1-4fed-b105-c65fa2883a61
# ╠═64fe8336-d1c2-41fe-a522-1b6f63260fc9
# ╟─a5009785-64b4-489b-a967-f7840b4a9463
# ╠═de50f95f-984e-4387-958c-64e0265f5953
# ╠═f841c4d8-5176-4007-b472-9e01a799d85c
# ╠═902738c3-2f7b-49cb-8580-29359c857027
# ╠═889611fb-7dac-4769-9251-9a90e3a1422f
# ╟─510761f6-66c7-4faf-937b-e1422ec829a6
# ╠═87fadfc0-2cdb-4be2-81ad-e8fdeffb690c
# ╠═4a8d17a2-348b-4077-8071-708017daaf05
# ╠═1dd1ba55-548a-41f6-903e-70742fd60e3d
# ╠═c6715072-a5a7-433f-90e1-7abbb221eb25
# ╠═2786101e-d365-4d6a-8de7-b9794499efb4
# ╟─0b9c6dbd-4eb3-4167-886e-64db9ec7ff04
# ╟─52aebb7b-c2a9-443f-bc03-24cd25793b32
# ╟─e6672866-c0a0-46f2-bb52-25fcc3352645
# ╟─e8f94345-9ad5-48d4-8709-d796fb55db3f
# ╟─a72d07bf-e337-4bd4-af5c-44d74d163b6b
# ╠═ddf3bb61-16c9-48c4-95d4-263260309762
# ╟─105c5c23-270d-437e-89dd-12297814c6e0
# ╟─48b557e3-e239-45e9-ab15-105bcca96492
# ╟─0a4ed8c7-27ca-45cb-af15-70ddd86240fb
# ╠═620a6426-cb29-4010-997b-aa4f9d5f8fb0
# ╠═3d8b1ccd-9bb3-42f2-a77a-6afdb72c1ff8
# ╠═209881b3-3ac8-490e-97bd-fa5ae24a39f5
# ╠═72b4d8d5-464c-4561-8c69-28ef3f59630b
# ╠═3f3ebc9b-b070-4d73-8be9-823b399c664c
# ╠═1e3d231a-4065-48ce-a74e-018066fb232a
# ╟─187fc682-2282-46ca-b988-c9de438f36fd
# ╟─22c2213e-5b9b-410f-a0ef-8f1e3db3c532
# ╟─0e59e813-3d48-4a24-b5b3-9a9de7c500c2
# ╟─0d6a11af-b146-4bbc-997e-a11b897269a7
# ╟─a925534e-f9b8-471a-9d86-c9212129b630
# ╟─62a9a36a-bedb-4f5a-80a4-2d4111a65c12
# ╟─b35264b0-ac5b-40ce-95e4-9b2bc4cb106f
# ╟─4d7619ee-933f-452a-9202-e95a8f3da20f
# ╟─fe2ebf39-4ab3-4aa8-abbd-23389eaf400e
# ╟─1ae30f5d-b25b-4dcb-800f-45c463641ec5
# ╟─6a1503c6-c77b-4e3a-9f07-74b2af1a5ff7
# ╠═6b496582-cc0e-4195-87ef-94792b0fff54
# ╠═cb07a6a5-c50a-4900-9e5b-a17dc7ee5710
# ╠═84a71bf8-0d66-42cd-ac7b-589d63a16eda
# ╠═4d4577b5-3753-450d-a247-ebd8c3e8f799
# ╠═12aac612-758b-4655-8ede-daddd4af6d3e
# ╠═3ed12c33-ab0a-49b1-b9e7-c4305ba35767
# ╠═61bbf9db-49a0-4709-83f4-44f228be09c0
# ╟─8d05403a-adeb-40ac-a98a-87586d5a5170
# ╠═e19db54c-4b3c-42d1-b016-9620daf89bfb
# ╠═ec285c96-4a75-4af6-8898-ec3176fa34c6
# ╠═ab331778-f892-4690-8bb3-26464e3fc05f
# ╟─500d8dd4-fc53-4021-b797-114224ca4deb
# ╟─136d1d96-b590-4f03-9e42-2337efc560cc
# ╠═4556cf44-4a1c-4ca4-bfb8-4841301a2ce6
# ╠═9f28772c-9afe-4253-ab3b-055b0f48be6e
# ╠═d299d800-a64e-4ba2-9603-efa833343405
# ╟─04a0be81-ee5f-4eeb-963a-ad930392d50b
# ╟─0ad739c9-8aca-4b82-bf20-c73584d29535
# ╠═031e1106-7408-4c7e-b78e-b713c19123d1
# ╟─cdedd35e-52b8-40a5-938d-2d36f6f93217
# ╠═9651f823-e1cd-4e6e-9ce0-be9ea1c3f0a4
# ╠═2155adfa-7a93-4960-950e-1b123da9eea4
# ╟─d259ecca-0249-4b28-a4d7-6880d4d84495
# ╠═dda222ef-8178-40bb-bf20-d242924c4fab
# ╟─f0f9d3d5-e76a-4472-bfb1-da29d73a7916
# ╟─39470c74-e554-4f6c-919d-97bec1eec0f3
# ╠═e9359ca3-4d11-4365-bc6e-7babc6fcc7de
# ╟─dee6b500-0ba1-4bbc-b217-cbb9ad47ad06
# ╟─db31579e-3e56-4271-8fc3-eb13bc95ac27
# ╟─b59eacf8-7f78-4015-bf2c-66f89bf0e24e
# ╠═02f34da1-551f-4ce5-a588-7f3a14afd716
# ╠═aa0791a5-8cf1-499b-9900-4d0c59be808c
# ╠═4ddc7d99-0b79-4689-bd93-8798b105c0a2
# ╟─ed4e863b-22dd-4d2b-88d0-b3a56d6713b7
# ╟─2d881aa9-1da3-4d1e-8d05-245956dbaf33
# ╠═8bc54c94-9c92-4904-b3a6-13ff3f0110bb
# ╠═678cad7a-1abb-4fcc-91ba-b5abcbb914cb
# ╠═9da5fd84-800d-4b3e-8627-e90ce8f20297
# ╟─44c49006-e210-4f97-916e-fe62f36c593f
# ╠═2034fd1e-5171-4eda-85d5-2de62d7a1e8b
# ╠═c34678f6-53bb-4f2a-96f0-a7b16f894ddd
# ╟─9d01c0ef-6313-4091-b444-3e9765aba90c
# ╟─4b1a4c14-3c2b-40c0-995c-cd0334ed8b3a
# ╟─897fde24-9a4a-465e-96f2-dd9e8baab294
# ╟─f2776908-d06a-4073-b2ce-ecbf109c9cc7
# ╟─1115f3ec-f4b2-4fba-bd5e-321a63b10a6d
# ╟─c4719c42-87aa-482a-95aa-a1492d42835d
# ╟─1e45a661-c2e1-40c2-b27b-5f80f95efdab
# ╟─8224b808-5778-458b-b683-ea2603c82117
# ╠═6556dafb-04fa-434c-868a-8d7bb7b5b196
# ╠═6faa3015-3ac4-44af-a78c-10b175822441
# ╠═6bffb08c-704a-4b7c-bfce-b3d099cf35c0
# ╠═a4c4d5f2-d76d-425e-b8c9-9047fe53c4f0
# ╟─05664aaf-575b-4249-974c-d8a2e63f380a
# ╟─2a3e4617-efbb-4bbc-9c61-8535628e439c
# ╟─6e06bd39-486f-425a-bbca-bf363b58988c
# ╠═292d9018-b550-4278-a8e0-78dd6a6853f1
# ╟─047a8881-c2ec-4dd1-8778-e3acf9beba2e
# ╟─667666b9-3ab6-4836-953d-9878208103c9
# ╟─cafedde8-be94-4697-a511-510a5fea0155
# ╟─c8500b89-644d-407f-881a-bcbd7da23502
# ╠═84584793-8274-4aa1-854f-b167c7434548
# ╠═6d9ae541-cf8c-4687-9f0a-f008944657e3
# ╟─29b0a2d5-9629-46cd-b57c-6f3ef797de66
# ╟─4862942b-d1e2-4ac8-8e88-65205e91a070
# ╟─ff5d051e-5de1-48a9-9578-5dbafd71afd1
# ╟─15dc58b8-8532-4b5a-b161-591852979ae5
# ╠═3f4f078a-9fc4-4b02-b499-a805fd5f1071
# ╟─e039a5be-4b59-4023-be97-2d1de970be27
# ╠═3756a3f8-18e8-4d62-afa1-cfeb4183820c
# ╠═d526a3a4-63cc-4f94-8f55-98c9a4a9d134
# ╟─223055df-7d5c-4d99-bc8d-fbc9702f906f
# ╠═4382928c-6325-4ecd-b7cf-282525a270ab
# ╠═69eedbfd-396f-4461-b7a1-c36abc094581
# ╟─00d67a93-437c-4cda-899a-9daa1102e1f2
# ╟─42799973-9884-4a0e-b29a-039890e92d21
# ╟─35dc0d94-145a-4292-b0df-9e84a286c036
# ╟─f95ceb98-f12e-4650-9ad3-0609b7ecd0f3
# ╟─f36822d7-9ea8-4f5c-9925-dc2a466a68ba
# ╠═639840dc-976a-4e5c-987f-a92afb2d99d8
# ╠═14b456f9-5fd1-4340-a3c7-ab9b91b4e3e0
# ╟─22c4ce8c-bd82-4eb3-8af5-55342018edff
# ╠═393cd9d2-dd97-496e-b260-ec6e8b1c13b5
# ╠═0748902c-ffc0-4634-9a1b-e642b3dfb77b
# ╠═c4919d14-8cba-43e6-9369-efc52bcb9b23
# ╠═95245673-2c29-401e-bb4b-a39dc8172297
# ╠═07c57f37-22be-4c39-8279-d80addcea0c5
# ╠═7ac99619-5232-4db8-8553-d79ea5415d29
# ╠═71774d5f-7841-403f-bc6b-1a0cbbb72d6d
# ╠═2f4e2da2-b1a1-41b1-8904-39b59f426da4
# ╠═0e488135-49e5-4e71-83b1-05d8e61f0510
# ╠═8e15f4b5-0dc7-47a5-9477-9f4d8807b331
# ╠═dea61907-d4fb-492d-b2bb-c037c7f785cb
# ╠═45213165-11c8-40fe-acee-99331852c2a7
# ╠═8787a5fd-d0ab-46b5-a7df-e7bc103a7378
# ╠═4019c974-dcaa-46c8-ac90-e6566a376ea1
# ╠═3134e913-1e86-495d-a558-c3ec4828bf7b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
