### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ 77cf7fee-0ad8-4d22-b376-75833307db93
begin
	using StatsBase, Statistics, PlutoUI, HypertextLiteral, LaTeXStrings, PlutoPlotly

	html"""
	<style>
		main {
			margin: 0 auto;
			max-width: 2000px;
	    	padding-left: max(160px, 10%);
	    	padding-right: max(160px, 10%);
		}
	</style>
	"""
end

# ╔═╡ 826139cc-b52e-11ec-0d47-25ab689851fd
md"""
# Chapter 5 Monte Carlo Methods

In this chapter we will use a sampling technique that does not require complete knowledge of the environment, only the ability to interact with it.  We will focus on episodic tasks so that the sampled returns are always well defined.  Unlike the dynamic programming methods from before, these techniques will estimate the value functions rather than computing them directly.  With enough samples, these techniques will also converge to the correct solution.

## 5.1 Monte Carlo Prediction

First we consider using Monte Carlo methods to learn the state-value function for a given policy.  The most obvious way to estimate this is to simply average the returns observed after vising a particular state.  If we save the trajectories of an agent interacting with an MDP, we can keep track of how many times an agent visits a state.  In the technique called *first visit MC* we look at the returns accumulated after the first visit to a state in that episode only making a single update. It is also possible to update the state value for every visit that occurs in an episode, but the theoretical properties for *first visit MC* are more widely studied.  Both techniques, however, can be shown to converge as the number of visits goes to infinity.  Below is code implementing Monte Carlo prediction using *every visit MC* and an in place update of the average
"""

# ╔═╡ a85e7dd3-ea5e-4c77-a18e-f1e190658ae3
function sample_action(π::Matrix{T}, i_s::Integer) where T<:AbstractFloat
	(n, m) = size(π)
	sample(1:n, weights(π[:, i_s]))
end

# ╔═╡ 72e9721f-ee17-4557-b090-71728e6c22ce
makelookup(v::Vector) = Dict(x => i for (i, x) in enumerate(v))

# ╔═╡ 21d248f5-edad-4614-8c1c-ae330f9e5a18
struct MDP_Opaque{S, A, F<:Function, G<:Function}
	states::Vector{S}
	statelookup::Dict{S, Int64}
	actions::Vector{A}
	actionlookup::Dict{A, Int64}
	state_init::G #function that produces an initial state for an episode
	simulator::F #function that takes as input an initial state action pair as well as the policy to be simulated.  Should return a trajectory as a vector of state action pairs and a vector of rewards
	function MDP_Opaque(states::Vector{S}, actions::Vector{A}, state_init::G, simulator::F) where {S, A, F<:Function, G<:Function}
		statelookup = makelookup(states)
		actionlookup = makelookup(actions)
		new{S, A, F, G}(states, statelookup, actions, actionlookup, state_init, simulator)
	end
end

# ╔═╡ b2115f33-f8e2-452e-9b0e-90610f0f09b1
function make_random_policy(mdp::MDP_Opaque; init::T = 1.0f0) where T <: AbstractFloat
	ones(T, length(mdp.actions), length(mdp.states)) ./ length(mdp.actions)
end

# ╔═╡ 202eb066-877d-4537-85b2-31b41db8eca0
initialize_state_value(mdp::MDP_Opaque; vinit::T = 0.0f0) where T<:AbstractFloat = ones(length(mdp.states)) .* vinit

# ╔═╡ d55860d1-e4c1-4a79-adbe-b40a6d6283a7
initialize_state_action_value(mdp::MDP_Opaque; qinit::T = 0.0f0) where T<:AbstractFloat = ones(T, length(mdp.actions), length(mdp.states)) .* qinit

# ╔═╡ 11723075-d1db-4512-abf2-3fe494a71a3b
function check_policy(π::Matrix{T}, mdp::MDP_Opaque{S, A, F}) where {T <: AbstractFloat, S, A, F}
#checks to make sure that a policy is defined over the same space as an MDP
	(n, m) = size(π)
	num_actions = length(mdp.actions)
	num_states = length(mdp.states)
	@assert n == num_actions "The policy distribution length $n does not match the number of actions in the mdp of $(num_actions)"
	@assert m == num_states "The policy is defined over $m states which does not match the mdp state count of $num_states"
	return nothing
end

# ╔═╡ 760c5361-02d4-46b7-a05c-fc2d10d93de6
function monte_carlo_pred(π::Matrix{T}, mdp::MDP_Opaque{S, A, F, G}, γ::T; num_episodes::Integer = 1000, vinit::T = zero(T), V::Vector{T} = ones(T, length(mdp.states)) .* vinit) where {T <: AbstractFloat, S, A, F, G}
	
	check_policy(π, mdp)
	
	#initialize
	counts = zeros(Integer, length(mdp.states))

	#there's no check here so this is equivalent to every-visit estimation
	function updateV!(traj, rewards; t = length(traj), g = zero(T))		
		#terminate at the end of a trajectory
		t == 0 && return nothing
		#accumulate future discounted returns
		g = γ*g + rewards[t]
		(i_s,i_a) = traj[t]
		#increment count by 1
		counts[i_s] += 1
		V[i_s] += (g - V[i_s])/counts[i_s] #update running average of V
		updateV!(traj, rewards; t = t-1, g = g)
	end

	
	for i in 1:num_episodes
		s0 = mdp.state_init()
		i_a0 = sample_action(π, mdp.statelookup[s0])
		a0 = mdp.actions[i_a0]
		# (s0, a0) = initialize_episode()
		(traj, rewards) = mdp.simulator(s0, a0, π)
	
		#update value function for each trajectory
		updateV!(traj, rewards)
	end
	return V
end

# ╔═╡ 6a11daf7-2859-41fa-9c3d-d1f3580dbb5f
md"""
### Example 5.1: Blackjack
"""

# ╔═╡ e87932d9-fbc0-4ea5-a8ee-28eac5eed84f
begin
	abstract type BlackjackAction end
	struct Hit <: BlackjackAction end
	struct Stick <: BlackjackAction end
	const cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, :A]
	const unique_cards = [:A, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	const cardlookup = Dict(unique_cards[i] => i for i in eachindex(unique_cards))
	const blackjack_actions = [Hit(), Stick()]
	const blackjackstates = [(sum=s, upcard=c, ua = ua) for s in 12:21 for c in unique_cards for ua in (true, false)]
	const π_blackjack_random = ones(Float32, 2, length(blackjackstates)) /2
end

# ╔═╡ 481ad435-cc80-4164-882d-8310b010ca91
function make_simple_blackjack_policy(stick_sum::Integer)
	mapreduce(hcat, blackjackstates) do state
		state.sum < stick_sum && return [1.0f0, 0.0f0]
		return [0.0f0, 1.0f0]
	end
end

# ╔═╡ 7fb4244c-828a-49d6-a15b-178fa3a42e00
#policy defined in Example 5.1 which sticks on sums of 20 or 21
const π_blackjack1 = make_simple_blackjack_policy(20)

# ╔═╡ c23475f0-c5f0-49ce-a665-f93c8bda0474
function make_blackjack_mdp()
	actionlookup = Dict([Hit() => 1, Stick() => 2])
	#deal a card from an infinite deck and return either the value of that card or an ace
	deal() = rand(cards)
	blackjackstates = [(sum=s, upcard=c, ua = ua) for s in 12:21 for c in unique_cards for ua in (true, false)]
	statelookup = Dict(blackjackstates[i] => i for i in eachindex(blackjackstates))

	#takes a previous sum, usable ace indicator, and a card to be added to the sum.  Returns the updated sum and whether an ace is still usable
	function addsum(s::Int64, ua::Bool, c::Symbol)
		if !ua
			s >= 11 ? (s+1, false) : (s+11, true)
		else
			(s+1, true)
		end
	end

	function addsum(s::Int64, ua::Bool, c::Int64)
		if !ua
			(s + c, false)
		else
			if (s + c) > 21
				(s + c - 10, false)
			else
				(s + c, true)
			end
		end
	end

	playersim(state, ::Stick, π, traj = [(statelookup[state], 2)]) = (state.sum, traj)
	function playersim(state, action::Hit, π, traj = [(statelookup[state], 1)])
		(s, ua) = addsum(state.sum, state.ua, deal())
		(s >= 21) && return (s, traj)
		newstate = (sum=s, upcard=state.upcard, ua=ua)
		i_s = statelookup[newstate]
		i_a = sample_action(π, i_s)
		push!(traj, (i_s, i_a))
		playersim(newstate, blackjack_actions[i_a], π, traj)
	end

	function dealer_sim(s::Int64, ua::Bool)
		(s >= 17) && return s
		(s, ua) = addsum(s, ua, deal())
		dealer_sim(s, ua)
	end

	#score a game in which the player didn't go bust
	function scoregame(playersum, dealersum)
		#if the dealer goes bust, the player wins
		dealersum > 21 && return 1.0f0

		#if the player is closer to 21 the player wins
		playersum > dealersum && return 1.0f0

		#if the dealer sum is closer to 21 the player loses
		playersum < dealersum && return -1.0f0

		#otherwise the outcome is a draw
		return 0.0f0
	end

	state_init() = (sum = rand(12:21), upcard = rand(unique_cards), ua = rand([true, false]))
		

	#starting with an initial state, action, and policy, generate a trajectory for blackjack returning that and the reward
	function blackjackepisode(s0::NamedTuple{(:sum, :upcard, :ua)}, a0::BlackjackAction, π::Matrix{Float32})
		(s, c, ua) = s0
		playernatural = (s == 21)
		splayer, traj = playersim(s0, a0, π)
		rewardbase = zeros(Float32, length(traj) - 1)
		finalr = if splayer > 21 
			#if the player goes bust, the game is lost regardless of the dealers actions
			-1.0f0
		else
			#generate hidden dealer card and final state
			hc = deal()
			(ds, dua) = if c == :A
				addsum(11, true, hc)
			else 
				addsum(c, false, hc)
			end
	
			dealernatural = ds == 21
	
			if playernatural
				Float32(!dealernatural)
			elseif dealernatural #not stated in book but used by authors in their code and matches actual blackjack rules
				-1.0f0
			else
				sdealer = dealer_sim(ds, dua)
				scoregame(splayer, sdealer)
			end
		end
		return (traj, [rewardbase; finalr])
	end

	
	MDP_Opaque(blackjackstates, blackjack_actions, state_init, blackjackepisode)
end

# ╔═╡ 6d765229-2816-4abb-a868-6be919a96530
const blackjack_mdp = make_blackjack_mdp()

# ╔═╡ a35de859-4046-4f6d-9ea9-b523d21cee5d
function make_value_grid(v_π)
	vgridua = zeros(Float32, 10, 10)
	vgridnua = zeros(Float32, 10, 10)
	for i in eachindex(blackjackstates)
		(s, c, ua) = blackjackstates[i]
		if ua
			vgridua[s-11, cardlookup[c]] = v_π[i]
		else
			vgridnua[s-11, cardlookup[c]] = v_π[i]
		end
	end
	return vgridua, vgridnua
end

# ╔═╡ 01683e11-a368-45bc-abbc-bbd5c94d7b22
plot_value(grid; title = "", xtitle = "Dealer showing", ytitle = "Player sum", xticktext = ["A", "10"], yticktext = [12, 21], showscale = false, width = 350, height = 300) = plot(heatmap(x = 1:10, y = 12:21, z = grid, showscale=showscale, colorscale = "Bluered", zmin = -1.0, zmax = 1.0), Layout(title = title, yaxis_title = ytitle, yaxis_tickvals = [12, 21], xaxis_title = xtitle, yaxis_ticktext = yticktext, xaxis_tickvals = [1, 10], xaxis_ticktext = xticktext, width = width, height = height, autosize = false, margin = attr(l = 0, b = 0, r = 0)))

# ╔═╡ d766d44e-3684-497c-814e-8f71740cb232
md"""
### **Figure 5.1**:  
Approximate state-value functions for the blackjack policy that sticks only on 20 or 21, computed by Monte Carlo policy evaluation
"""

# ╔═╡ e5384dd0-fad1-4a24-b011-73b062fcfb1b
md"""
> ### *Exercise 5.1* 
> Consider the diagroms on the right in Figure 5.1.  Why does the estimated value function jump for the last two rows in the rear?  Why does it drop off for the whole last row on the left?  Why are the frontmost values higher in the upper diagrams than in the lower?

The last two rows in the rear are for a player sum equal to 20 or 21.  For a sum of 19 and lower, this policy will hit which is suboptimal for these high sums.  For the sums of 20 or 21 though, sticking is optimal so the value jumps compared to the suboptimal hit at 19.

The far left row represents cases where the dealer is showing an Ace.  Since an Ace is a flexible card, the dealer policy will have more options that result in a win including the possibility of having another face card already (dealer natural).  It is always worse for the player if the dealer is known to have an Ace.

The frontmost values represent cases where the player sum is 12.  If there is a usable Ace this means that means that the player has two Aces which results in a sum of 12 when the first Ace is counted as 1 and the second is *usable* and counted as 11.  If there is no usable Ace than a sum of 12 would have to result from some other combination of cards such as 10/2, 9/3, etc...  Since the first case has two Aces, it means that potentially both could count as 1 if needed to avoid a bust.  In the case without a usable Ace, the sum is the same, but there are more opportunities to bust if we draw a card worth 10, so having a sum of 12 with a usable Ace is strictly better.
"""

# ╔═╡ 30809344-b4ab-468b-b4b7-5ef3dca5ffc7
md"""
> ### *Exercise 5.2* 
> Suppose every-visit MC was used instead of first-visit MC on the blackjack task.  Would you expect the results to be very different?  Why or why not?

As an episode proceeds in blackjack the states will not repeat since every time a card is dealt the player sum changes, or the usable Ace flag changes.  Thus the check ensuring that only the first visit to a state is counted in the return average will have no effect on the MC evaluation.
"""

# ╔═╡ f406be9e-3e3f-4b55-99b0-4858c774ed96
md"""
## 5.2 Monte Carlo Estimation of Action Values
"""

# ╔═╡ be7c096c-cc8c-407b-8287-8fb2ee7150a7
md"""
> ### *Exercise 5.3* 
> What is the backup diagram for Monte Carlo estimation of $q_\pi$

Similar to the $v_\pi$ diagram except the root is the s,a pair under consideration followed by the new state and the action taken along the trajectory.  The rewards are still accumulated to the end, just the start of the trajectory is a solid filled in circle that would contain the value for that s,a pair.
"""

# ╔═╡ 2572e35c-e6a3-4562-aa0f-6a5ab32d39ea
@htl("""
<div style="display: flex; flex-direction: row; align-items: flex-start; justify-content: center; background-color: rgb(100, 100, 100)">
	
	<div class="backup">
		<div>State Value Function</div>
		<div class="circlestate"></div>
		<div class="arrow"></div>
		<div class="circleaction"></div>
		<div class="arrow"></div>
		<div class="circlestate"></div>
		<div class="arrow"></div>
		<div class="circleaction"></div>
		<div style = "color: black; font-size: 30px;">&#8942;</div>
		<div class="circleaction"></div>
		<div class="arrow"></div>
		<div class="term"></div>
	</div>
	<div class="backup">
		<div>State Action Value Function</div>
		<div class="circleaction"></div>
		<div class="arrow"></div>
		<div class="circlestate"></div>
		<div class="arrow"></div>
		<div class="circleaction"></div>
		<div class="arrow"></div>
		<div class="circlestate"></div>
		<div style = "color: black; font-size: 30px;">&#8942;</div>
		<div class="circlestate"></div>
		<div class="arrow"></div>
		<div class="circleaction"></div>
		<div class="arrow"></div>
		<div class="term"></div>
	</div>
	<div>
		<div class="q_backup"></div>
	</div>
</div>

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
		content: '';
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

# ╔═╡ 47daf83b-8fe9-4491-b9ae-84bd269d5546
md"""
## 5.3 Monte Carlo Control
"""

# ╔═╡ 2d10281a-a4af-4ea8-b63b-e11f2d0893ed
function make_greedy_policy!(v::AbstractVector{T}; c = 1000) where T<:Real
	(vmin, vmax) = extrema(v)
	if vmin == vmax
		v .= one(T) / length(v)
	else
		v .= (v .- vmax) ./ abs(vmin - vmax)
		v .= exp.(c .* v)
		v .= v ./ sum(v)
	end
	return v
end

# ╔═╡ 13cc524c-d983-44f4-8731-0595249fb888
function monte_carlo_ES(mdp::MDP_Opaque, π_init::Matrix{T}, Q_init::Matrix{T}, γ, num_episodes) where T <: Real
	#initialize
	π = copy(π_init)
	Q = copy(Q_init)
	counts = zeros(Int64, length(mdp.actions), length(mdp.states))
	vhold = zeros(T, length(mdp.actions))
	for i in 1:num_episodes
		s0 = mdp.state_init()
		a0 = rand(mdp.actions)
		(traj, rewards) = mdp.simulator(s0, a0, π)

		#there's no check here so this is equivalent to every-visit estimation
		function updateQ!(traj, rewards; t = length(traj), g_old = zero(T))		
			#terminate at the end of a trajectory
			t == 0 && return nothing
			#accumulate future discounted returns
			g = γ*g_old + rewards[t]
			(i_s,i_a) = traj[t]
			#increment count by 1
			counts[i_a, i_s] += 1
			Q[i_a, i_s] += (g - Q[i_a, i_s])/counts[i_a, i_s] #update running average of Q
			vhold .= Q[:, i_s]
			make_greedy_policy!(vhold)
			π[:, i_s] .= vhold
			updateQ!(traj, rewards; t = t-1, g_old = g)
		end
		updateQ!(traj, rewards)
	end
	return π, Q
end

# ╔═╡ 1b78b25d-3942-4a6b-a2bd-7d97242da9fe
monte_carlo_ES(mdp::MDP_Opaque, γ::T, num_episodes; Q_init::Matrix{T} = initialize_state_action_value(mdp; qinit = zero(T)), π_init::Matrix{T} = make_random_policy(mdp; init = one(T))) where T <: Real = monte_carlo_ES(mdp, π_init, Q_init, γ, num_episodes) 

# ╔═╡ 9618a093-cdb7-4589-a783-de8e9021b705
md"""
### Example 5.3: Solving Blackjack
"""

# ╔═╡ 5c57e4b7-51d6-492d-9fc9-bcdab1dd46f4
function plot_blackjack_policy(π)
	πstargridua = zeros(Float64, 10, 10)
	πstargridnua = zeros(Float64, 10, 10)
	for state in blackjackstates
		(s, c, ua) = state
		i_s = blackjack_mdp.statelookup[state]
		v = π[1, i_s] - π[2, i_s]

		y = c == :A ? 1 : c
		if ua
			πstargridua[s-11, y] = v
		else
			πstargridnua[s-11, y] = v
		end
	end

	function plot_policy_grid(grid)
		plot(heatmap(y = 12:21, z = grid, showscale=false, colorscale = "Greys", zmin = -1, zmax = 1), Layout(xaxis_title = "Dealer showing", yaxis_title = "Player sum", margin = attr(t = 30, b = 0, l = 0, r = 10), width = 300, height = 300, xaxis_tickvals = 1:10, yaxis_tickvals = 12:21, yaxis_ticktext = 12:21, xaxis_ticktext = ["A"; 2:10]))
	end

	p1 = plot_policy_grid(πstargridua)
	p2 = plot_policy_grid(πstargridnua)
	# vstar = eval_blackjack_policy(Dict(s => π[s] == :hit ? [1.0, 0.0] : [0.0, 1.0] for s in blackjackstates), 500_000)
	# p1 = heatmap(πstargridua, xticks = (1:10, ["A", 2, 3, 4, 5, 6, 7, 8, 9, 10]), yticks = (1:10, 12:21), legend = false, title = "Usable Ace Policy, Black=Stick, White = Hit")
	# p2 = heatmap(πstargridnua, xticks = (1:10, ["A", 2, 3, 4, 5, 6, 7, 8, 9, 10]), yticks = (1:10, 12:21), legend = false, title = "No usable Ace Policy", xlabel = "Dealer Showing", ylabel = "Player sum")

	# p3 = heatmap(vstar[1], legend = false, yticks = (1:10, 12:21), title = "V*")
	# p4 = heatmap(vstar[2], yticks = (1:10, 12:21))
	# plot(p1, p3, p2, p4, layout = (2,2))
	(p1, p2)
end

# ╔═╡ 06960937-a87a-4015-bdeb-5d17aa41fe6b
function plot_blackjack_value(v)
	uagrid, nuagrid = make_value_grid(v)
	p1 = plot_value(uagrid, showscale=true, width = 400)
	p2 = plot_value(nuagrid, showscale=true, width = 400)
	
	(p1, p2)
end

# ╔═╡ ec29865f-3ba3-4bb3-84df-c2b472e03ff2
(πstar_blackjack, Qstar_blackjack) = monte_carlo_ES(blackjack_mdp, 1.0f0, 2_000_000; π_init = π_blackjack1)

# ╔═╡ 627d5aa3-974f-4949-8b65-9500eba1d7cc
#recreation of figure 5.2
function figure_5_2()
	policyplot = plot_blackjack_policy(πstar_blackjack)
	vstar = sum(Qstar_blackjack .* πstar_blackjack, dims = 1)[:]
	# vhit = Qstar_blackjack[1, :][:]
	# vstick = Qstar_blackjack[2, :][:]
	# vdiff = (vhit .- vstick)
	valueplot = plot_blackjack_value(vstar)
	# policyplot = plot_blackjack_value(vdiff)
	vtitle = md"""$v_*$"""
	ptitle = md"""$π_*$"""

	md"""
	| |$\pi_*$ (Black = Stick, White = Hit)|$v_*$|
	|---|:---:|:---:|
	|Usable ace|$(policyplot[1])|$(valueplot[1])|
	|No usable ace|$(policyplot[2])|$(valueplot[2])|
	"""
end

# ╔═╡ 9fe42679-dce3-4ee3-b565-eed4ff7d8e4d
md"""
### Figure 5.2:

The optimal policy and state-value function for blackjack found by Monte Carlo ES.  The state value function shown was computed from the action-value function found by Monte Carlo ES
"""

# ╔═╡ f9bc84ff-c2f9-4ac2-b2ce-34dfcf837a73
figure_5_2()

# ╔═╡ c883748c-76b9-4086-8698-b40df51390da
md"""
> ### *Exercise 5.4* 
> The pseudocode for Monte Carlo ES is inefficient because, for each state-action pair, it maintains a list of all returns and repeatedly calculates their mean.  It would be more dfficient to use techniques similar to those explained in Section 2.4 to maintain just the mean and a count (for each state-action pair) and update them incrementally.  Describe how the pseudocode would be altered to achieve this.

Returns(s,a) will not maintain a list but instead be a list of single values for each state-action pair.  Additionally, another list Counts(s,a) should be initialized at 0 for each pair.  When new G values are obtained for state-action pairs, the Count(s,a) value should be incremented by 1.  Then Returns(s,a) can be updated with the following formula: 

$\text{Returns}(s,a) = \frac{\text{Returns}(s,a) \times (\text{Count}(s,a) - 1) + G(s,a)}{\text{Count}(s,a)} = \text{Returns}(s,a) + \frac{G(s,a) - \text{Returns}(s,a)}{\text{Count}(s,a)}$
"""

# ╔═╡ e2a720b0-a8c4-43a2-bf34-750ff3323004
md"""
## 5.4 Monte Carlo Control without Exploring Starts
"""

# ╔═╡ 38957d8d-e1d0-44c3-bac9-1417925ac882
function monte_carlo_ϵsoft(states, actions, simulator, γ, ϵ, nmax = 1000; gets0 = () -> rand(states))
	#initialize
	nact = length(actions)
	avec = collect(actions)
	adict = Dict(a => i for (i, a) in enumerate(actions))
	π = Dict(s => ones(nact)./nact for s in states)
	Q = Dict((s, a) => 0.0 for s in states for a in actions)
	counts = Dict((s, a) => 0 for s in states for a in actions)
	sampleπ(s) = sample(avec, weights(π[s]))
	for i in 1:nmax
		s0 = gets0()
		a0 = sampleπ(s0)
		(traj, rewards) = simulator(s0, a0, sampleπ)
		
		#there's no check here so this is equivalent to every-visit estimation
		t = length(traj)
		g = 0.0
		while t != 0
			g = γ*g + rewards[t]
			(s,a) = traj[t]
			counts[(s,a)] += 1
			Q[(s,a)] += (g - Q[(s,a)])/counts[(s,a)]
			astar = argmax(a -> Q[(s,a)], actions)
			istar = adict[astar]
			π[s] .= ϵ/nact
			π[s][istar] += 1 - ϵ
			t -= 1
		end
	end
	π_det = Dict(s => actions[argmax(π[s])] for s in states)
	return π_det, Q
end

# ╔═╡ bf19e6cf-1fb5-49c9-974e-1613d90ef4cf
(πstar_blackjack2, Qstar_blackjack2) = monte_carlo_ϵsoft(blackjackstates, blackjackactions, blackjackepisode, 1.0, 0.05, 10_000_000)

# ╔═╡ b040f245-b2d6-4ec6-aa7f-511c54aabd0d
#recreation of figure 5.2 using ϵ-soft method
plot_blackjack_policy(πstar_blackjack2)

# ╔═╡ 12c0cd0b-eb4f-45a0-836b-53b3c5cdafd9
md"""
## 5.5 Off-policy Prediction via Importance Sampling
"""

# ╔═╡ 9793b5c9-d4ec-492d-a72d-8737bb65c8a5
md"""
Given a starting state $S_t$, the probability of the subsequent state-action trajectory, $A_t, S_{t+1}, A_{t+1}, \ldots ,S_T$, occuring under any policy $\pi$ is:

$Pr_{\pi}\{\mathcal{traj}\} = \prod_{k=t}^{T-1} \pi(A_k|S_k)p(S_{k+1}|S_k, A_k)$

where $p$ here is the state-transition probability function defined by (3.4).  Thus, the relative probability of the trajectory under the target and behavior policies (the importance-sampling ratio) is

$\rho_{t:T-1} \dot= \prod_{k=t}^{T-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}$
To estimate $v_\pi(s)$, we simply scale the returns by the ratios and average the results:

$V(s) \dot= \frac{\sum_{t \in \mathscr{T}(s)} \rho_{t:T(t)-1}G_t}{|\mathscr{T}(s)|}$ 

When importance sampling is done as a simple average in this way it is called *ordinary importance sampling*.

An important alternative is *weighted importance sampling*, which uses a *weighted* average, defined as

$V(s) \dot= \frac{\sum_{t \in \mathscr{T}(s)} \rho_{t:T(t)-1}G_t}{\sum_{t \in \mathscr{T}(s)} \rho_{t:T(t)-1}}$,
or zero is the denominator is zero.
"""

# ╔═╡ 15925cc6-9605-4357-9c2a-cdfe54070989
md"""
Consider an implementation or ordinary importance sampling that updates $V(s)$ incrementally every time a $G$ value is observed for that state.  The equations should be similar to the incremental update rule previously derived for $V(s)$ without importance sampling.

Consider a sequence of returns $G_1, G_2, \dots , G_{n-1}$, all starting in the same state and each with a corresponding weight $W_i = \rho_{t_i:T(t_i)-1}$.  We wish to form the estimate

$V_n \dot= \frac{\sum_{k=1}^{n-1}W_k G_k}{n-1}, n \geq 2$

and keep it up-to-date as we obtain a single additional return $G_n$.  Observe that we can increment n by 1 to get an espression for V in terms of itself.

$V_{n+1} = \frac{\sum_{k=1}^n W_k G_k}{n} = \frac{W_n G_n + \sum_{k=1}^{n-1} W_k G_k}{n}$

Using the original formula for $V_n$, we can make the following substitution:

$\sum_{k=1}^{n-1} W_k G_k = (n-1)V_n$

which results in 

$V_{n+1} = \frac{W_n G_n + V_n (n-1)}{n} = V_n + \frac{W_n G_n - V_n}{n}$

So, to calculate the value function, we can simply apply the following update rule after obtaining new values for W and G:

$C \leftarrow C + 1$
$V \leftarrow V + \frac{WG-V}{C}$

which looks very similar to the ordinary average update rule but with the weight multiplied by G.  C just keeps a running total of the times the state was observed.  Note that C needs to be updated even in the case where W is 0 which is not the case for weighted importance sampling.  A similar inremental update rule is derived later for the weighted case as well as an algorithm for updating the action-value estimate using this method.  Below are code examples for calculating the value estimates for both weighted and normal importance sampling using the incremental implementation.
"""

# ╔═╡ d97f693d-27f2-49be-a549-07a290c95b53
#types allow dispatch for each sampling method based on different incremental update rules
abstract type ImportanceMethod end

# ╔═╡ 660ef59c-205c-44c2-9c46-5a74e09497ab
struct Weighted <: ImportanceMethod end

# ╔═╡ 4197cc28-b24c-48cb-bd8d-ef998983ad77
struct Ordinary <: ImportanceMethod end

# ╔═╡ 6be7ff29-5845-4df4-ba18-bafea79ace71
function monte_carlo_pred(π_target, π_behavior, states, actions, simulator, γ, nmax = 1000; gets0 = () -> rand(states), historystate = states[1], samplemethod::ImportanceMethod = Ordinary(), V0 = 0.0)
	#initialize values and counts at 0
	V = Dict(s => V0 for s in states)
	Vhistory = zeros(nmax)
	counts = Dict(s => 0.0 for s in states)
	
	#maps actions to the index for the probability lookup
	adict = Dict(a => i for (i, a) in enumerate(actions))
	
	avec = collect(actions) #in case actions aren't a vector
	sample_b(s) = sample(avec, weights(π_behavior[s])) #samples probabilities defined in policy to generate actions 

	#updates the denominator used in the value update.  For ordinary sampling, this is just the count of visits to that state.  For weighted sampling, this is the sum of all importance-sampling ratios at that state
	updatecounts!(::Ordinary, s, w) = counts[s] += 1.0
	updatecounts!(::Weighted, s, w) = counts[s] += w
	

	#updates the value estimates at a given state using the future discounted return and the importance-sampling ratio
	updatevalue!(::Ordinary, s, g, w) = V[s] += (w*g - V[s])/counts[s]
	updatevalue!(::Weighted, s, g, w) = V[s] += (g - V[s])*w/counts[s]
		
	for i in 1:nmax
		s0 = gets0()
		a0 = sample_b(s0)
		(traj, rewards) = simulator(s0, a0, sample_b)
		
		#there's no check here so this is equivalent to every-visit estimation
		function updateV!(t = length(traj); g = 0.0, w = 1.0)
			#terminate at the end of a trajectory
			t == 0 && return nothing
			(s,a) = traj[t]
			
			#since this is the value estimate, every action must update the importance-sampling weight before the update to that state is calculated. In contrast, for the action-value estimate the weight is only the actions made after the current step are relevant to the weight
			w *= π_target[s][adict[a]] / π_behavior[s][adict[a]]
			
			updatecounts!(samplemethod, s, w)
			
			#terminate when w = 0 if the weighted sample method is being used.  under ordinary sampling, the updates to the count and value will still occur because the denominator will still increment
			(w == 0 && isa(samplemethod, Weighted)) && return nothing
			
			#update discounted future return from the current step
			g = γ*g + rewards[t]
			updatevalue!(samplemethod, s, g, w)

			#continue back through trajectory one step
			updateV!(t-1, g = g, w = w)
		end
		updateV!()
		Vhistory[i] = V[historystate] #save the value after iteration i for the specified state
	end
	return V, Vhistory
end

# ╔═╡ 94be5289-bba7-4490-bdcd-0d217a31c665
#calculate value function for blackjack policy π and save results in plot-ready grid form
function eval_blackjack_policy(π, episodes; γ=1f0)
	v_π = monte_carlo_pred(π, blackjack_mdp, γ; num_episodes = episodes)
	make_value_grid(v_π)
end

# ╔═╡ a68b4965-c392-47e5-9b29-93e7ada9990a
function plot_fig5_1()
	(uagrid10k, nuagrid10k) = eval_blackjack_policy(π_blackjack1, 10_000)
	(uagrid500k, nuagrid500k) = eval_blackjack_policy(π_blackjack1, 500_000)
	p1 = plot_value(uagrid10k; title = "After 10,000 episodes", width = 320, ytitle = "", xtitle = "") 
	p2 = plot_value(nuagrid10k, width = 320, ytitle = "")
	# p2 = plot(heatmap(z = nuagrid10k, showscale = false, colorscale = "Bluered", zmin = -1.0, zmax = 1.0), Layout(width = 300, height = 300, margin = attr(b = 0, r = 0, t=0, l = 0), autosize = false, yaxis_title = "No usable ace"))
	p3 = plot_value(uagrid500k, title = "After 500,000 episodes", showscale=true, width = 400, xtitle = "")
	# p3 = plot(heatmap(z = uagrid500k, showscale = false, colorscale = "Bluered", zmin = -1.0, zmax = 1.0), Layout(margin = attr(b = 0, r = 0, t=0, l = 0), autosize = false, width = 350, height = 300, title = "After 500,000 episodes"))
	p4 = plot_value(nuagrid500k; showscale = true, width = 400)
	# p4 = plot(heatmap(z = nuagrid500k, colorscale = "Bluered", zmin = -1.0, zmax = 1.0), Layout(width = 300, height = 300, yaxis_ticks = (1:10, 12:21), xaxis_ticks = (1:10, ["A", "", "", "", "", "", "", "", "", "10"]), legend = false, xaxis_title = "Dealer Showing"))

	# plot(p1, p3, p2, p4, layout = (2, 2))
	@htl("""
	<div style = "display: flex; flex-wrap: wrap; flex-direction: row; width: 760; background-color: white; color: black; align-items: center; justify-content: center;">
		<div style = "width: 50px;">Usable ace</div>
		<div>$p1</div> 
		<div>$p3</div>
		<div style = "width: 50px;">No usable ace</div>
		<div>$p2</div> 
		<div>$p4</div>
	</div>
	""")
	# [p1 p3; p2 p4]
end

# ╔═╡ 82284c63-2306-4469-9b1a-a5ec87037e79
plot_fig5_1()

# ╔═╡ cede8090-5b1a-436b-a184-fca5c4d3de48
md"""
> *Exercise 5.5* Consider an MDP with a single nonterminal state and a single action that transitions back to the nonterminal state with probability $p$ and transitions to the terminal state with probability $1-p$.  Let the reward be +1 on all transitions, and let $\gamma=1$.  Suppose you observe one episode that lasts 10 steps, with a return of 10.  What are the first-visit and every-visit estimators of the value of the nonterminal state?

For the first-visit estimator, we only consider the single future reward from the starting state which would be 10.  There is nothing to average since we just have the single value of 10 for the episode.

For the every-visit estimator, we need to average together all 10 visits to the non-terminal state.  For the first visit, the future reward is 10.  For the second visit it is 9, third 8, and so forth.  The final visit has a reward of 1, so the value estimate is the average of 10, 9, ..., 1 which is $\frac{(1+10) \times 5}{10}=\frac{55}{10} = 5.5$
"""

# ╔═╡ b39d1ea0-86a2-4215-ae73-e4492f3f2f00
md"""
### Example 5.4: Off-policy Estimation of a Blackjack State Value
"""

# ╔═╡ a2b7d85e-c9ad-4104-92d6-bac2ce362f1d
# ╠═╡ disabled = true
#=╠═╡
blackjackepisode((13, 2, true), :hit, s -> sample(collect(blackjackactions), weights(π_blackjack1[s])))
  ╠═╡ =#

# ╔═╡ 1d2e91d0-a1a0-4630-ac06-37684a9104f3
# ╠═╡ disabled = true
#=╠═╡
blackjackepisode((13, 2, true), :hit, s -> rand(blackjackactions))
  ╠═╡ =#

# ╔═╡ acb08e38-fc87-43f4-ac2d-6c6bf0f2e9e6
function estimate_blackjack_state(n, π)
	avec = collect(blackjackactions)
	rewards = zeros(n)
	sampleπ(s) = sample(avec, weights(π[s]))
	s0 = (13, 2, true)
	a0 = sampleπ(s0)
	for i in 1:n
		ep = blackjackepisode(s0, a0, sampleπ)
		rewards[i] = ep[2][end]
	end

	return mean(rewards), var(rewards)
end

# ╔═╡ 2b9131c1-4d79-4ea3-b51f-7f3380aeb629
# ╠═╡ disabled = true
#=╠═╡
#target policy state value estimate and variance, why is the mean squared error after 1 episode for weighted importance sampling less than the variance of the state values?  The reason is after 1 episode of weighted importance sampling, there is a greater than 50% chance of a 0% decision being made resulting in the value estimate not updating from its initial value.  Since that initial value is 0 and episodes are likely to terminate with rewards of -1 or 1, the variance is moved closer to 0 than for truly sampling the target policy rewards
estimate_blackjack_state(100_000_000, π_blackjack1)
  ╠═╡ =#

# ╔═╡ 70d9d39f-020d-4f25-810c-82a143a3335b
const π_rand_blackjack = Dict(s => [0.5, 0.5] for s in blackjackstates)

# ╔═╡ 8faca500-b80d-4b50-88b6-683d18a1286b
#behavior policy state value estimate and variance
estimate_blackjack_state(10_000_000, π_rand_blackjack)

# ╔═╡ d6863551-a254-44b6-b6fe-551d134cdf01
# ╠═╡ disabled = true
#=╠═╡
v_offpol = monte_carlo_pred(π_blackjack1, Dict(s => [0.5, 0.5] for s in blackjackstates), blackjackstates, blackjackactions, blackjackepisode, 1.0, 10_000_000, gets0 = () -> (13, 2, true))
  ╠═╡ =#

# ╔═╡ c5482c11-1635-4016-bf6a-4c5f01ae66b9
#=╠═╡
#confirms that the off policy value estimate is accurate
v_offpol[1][(13, 2, true)]
  ╠═╡ =#

# ╔═╡ 303c852d-177c-4ddc-aa53-b72e6e82cc55
function figure5_3_extra(n = 100; ep = 10^4)
	v0 = 0.0
	s0 = (13, 2, true)
	gets0() = s0

	π_rand = Dict(s => [0.5, 0.5] for s in blackjackstates)

	π_b = π_rand
	# π_b = π_blackjack1
	
	vhist_ordinary = reduce(hcat, [monte_carlo_pred(π_blackjack1, π_b, blackjackstates, blackjackactions, blackjackepisode, 1.0, ep, gets0 = gets0, historystate = s0, V0 = v0)[2] for _ in 1:n])
	vhist_weighted = reduce(hcat, [monte_carlo_pred(π_blackjack1, π_b, blackjackstates, blackjackactions, blackjackepisode, 1.0, ep, gets0 = gets0, historystate = s0, samplemethod = Weighted(), V0 = v0)[2] for _ in 1:n])
	ord_var = var(vhist_ordinary, dims = 2)
	ord_mean = mean(vhist_ordinary, dims = 2)
	weighted_var = var(vhist_weighted, dims = 2)
	weighted_mean = mean(vhist_weighted, dims = 2)
	plot(ord_var, xaxis = :log, lab = "ordinary variance", yaxis = [-0.5, 2.0])
	plot!(ord_mean, lab = "ordinary mean")
	plot!(weighted_var, lab = "weighted variance")
	plot!(weighted_mean, lab = "weighted mean")
end

# ╔═╡ 74fc1b42-9784-4968-8c85-f3d0b778fa2f
figure5_3_extra(1000)

# ╔═╡ 00cd2194-af13-415a-b725-bb34832e5d9a
function figure5_3(n = 100)
	s0 = (13, 2, true)
	gets0() = s0
	π_rand = Dict(s => [0.5, 0.5] for s in blackjackstates)
	vhist_ordinary = [(monte_carlo_pred(π_blackjack1, π_rand, blackjackstates, blackjackactions, blackjackepisode, 1.0, 10_000, gets0 = gets0, historystate = s0)[2] .+ 0.27726) .^2 for _ in 1:n]
	vhist_weighted = [(monte_carlo_pred(π_blackjack1, π_rand, blackjackstates, blackjackactions, blackjackepisode, 1.0, 10_000, gets0 = gets0, historystate = s0, samplemethod = Weighted())[2] .+ 0.27726) .^2 for _ in 1:n]
	plot(reduce((a, b) -> a .+ b, vhist_ordinary) ./ n, xaxis = :log, lab = "ordinary")
	plot!(reduce((a, b) -> a .+ b, vhist_weighted) ./ n, xaxis = :log, lab = "weighted", yaxis = [0, 5])
end

# ╔═╡ 1a97b3de-1aaa-4f51-b358-5c5f2b1d0851
figure5_3(1000)

# ╔═╡ e10378eb-12b3-4468-9c22-1838107da450
md"""
### Example 5.5: Infinite Variance
"""

# ╔═╡ f071b7e2-3f78-4132-8b84-f810f178c89d
const one_state_actions = (:left, :right)

# ╔═╡ ab2c27c8-122e-4e38-8aab-73291077b640
function one_state_simulator(s0, a0, π::Function)
	traj = [(s0,a0)]
	rewards = Vector{Float64}()
	function runsim(s, a)
		if a == :right
			push!(rewards, 0.0)
			return traj, rewards
		else
			t = rand()
			if t <= 0.1
				push!(rewards, 1.0)
				return traj, rewards
			else
				push!(rewards, 0.0)
				anew = π(s)
				push!(traj, (s, anew))
				runsim(s, anew)
			end
		end
	end
	runsim(s0, a0)
end

# ╔═╡ 61eb74eb-88e0-42e6-a14b-3730a800694d
const onestate_π_target = Dict(0 => [1.0, 0.0])

# ╔═╡ 82a1f11e-b1b3-4c7d-ab9d-9f6136ae8195
const onestate_π_b = Dict(0 => [0.5, 0.5])

# ╔═╡ a7c1e2ae-54ec-407f-8496-4f8a799f8759
monte_carlo_pred(onestate_π_target, onestate_π_b, [0], one_state_actions, one_state_simulator, 1.0, 1000, gets0 = () -> 0, historystate = 0, samplemethod = Ordinary())

# ╔═╡ 76877978-9a2b-48ba-aa4e-785622ac8c3d
monte_carlo_pred(onestate_π_target, onestate_π_b, [0], one_state_actions, one_state_simulator, 1.0, 1000, gets0 = () -> 0, historystate = 0, samplemethod = Weighted())

# ╔═╡ 5948f670-1203-4b75-8517-f8470f5d01aa
function figure_5_4(expmax, nsims)
	nmax = 10^expmax
	function makeplotinds(expmax)
		plotinds = mapreduce(i -> i:min(i, 1000):i*9, vcat, 10 .^(0:expmax-1))
		vcat(plotinds, 10^(expmax))
	end

	plotinds = makeplotinds(expmax)
	
	vhistnormal = [monte_carlo_pred(onestate_π_target, onestate_π_b, [0], one_state_actions, one_state_simulator, 1.0, nmax, gets0 = () -> 0, historystate = 0)[2][plotinds] for _ in 1:nsims]

	vhistweighted = monte_carlo_pred(onestate_π_target, onestate_π_b, [0], one_state_actions, one_state_simulator, 1.0, nmax, gets0 = () -> 0, historystate = 0, samplemethod = Weighted())[2][plotinds]

	
	plot(plotinds, vhistnormal, lab = "normal")
	plot!(plotinds, vhistweighted, lab = "weighted", yaxis = [0, 3])
end

# ╔═╡ 830d6d61-7259-43e7-8d6d-09bc81818dd9
# ╠═╡ disabled = true
#=╠═╡
figure_5_4(7, 3)
  ╠═╡ =#

# ╔═╡ 881ba6c3-0e6a-4bb8-bc14-2e0face560f2
md"""
> *Exercise 5.6* What is the equation analogous to (5.6) for *action* values $Q(s,a)$ instead of state values $V(s)$, again given returns generated using $b$?

Equation (5.6):

$V(s) = \frac{\sum_{t \in \mathscr{T}(s)}\rho_{t:T(t)-1}G_t}{\sum_{t \in \mathscr{T}(s)} \rho_{t:T(t)-1}}$

For $Q(s,a)$, there is no need to calculate the sampling ratio for the first action selected.  This also assumes that the trajectory used for $G$ and $\rho$ has the first action being the one specified by $Q(s,a)$.

$Q(s,a) = \frac{\sum_{t \in \mathscr{T}(s)}\rho_{t+1:T(t)-1}G_t}{\sum_{t \in \mathscr{T}(s)} \rho_{t+1:T(t)-1}}$
"""

# ╔═╡ 6979db32-670d-466a-9a6e-97c2d8527f3d
md"""
> *Exercise 5.7* In learning curves such as those shown in Figure 5.3 error generally decreases with training, as indeed happened for the ordinary importance-sampling method.  But for the weighted importance-sampling method error first increased and then decreased.  Why do you think this happened?

If the initial trajectories sampled are similar to ones we'd expect from the target policy, then the error will start low.  Since the weighted method has bias which only converged to 0 with large episodes, we might expect to see the error rise as we sample trajectories which are less probable with the target policy.  This bias will only dissappear as we add samples.  In Figure 5.3, if the generator policy produces trajectories which are greater than 50% probable by the target policy, then early on our biased estimate will be low.  But as we add more episodes the average will include more unlikely trajectories and push up the bias until the large numbers of samples has it convering back to 0 again.
"""

# ╔═╡ bf33230e-342b-4486-89ac-9667438de503
md"""
> *Exercise 5.8* The results with Example 5.5 and shown in Figure 5.4 used a first-visit MC method.  Suppose that instead an every-visit MC method was used on the same problem.  Would the variance of the estimator still be infinite?  Why or why not?

Terms for each episode length are as follows:

Length 1 episode

$\frac{1}{2} \cdot 0.1 \cdot 2^2$

Length 2 episode, the term representing X is now an average of every visit along the trajectory.  The probability of the trajectory is unchanged from before.

$\frac{1}{2} \cdot 0.9 \cdot \frac{1}{2} \cdot 0.1 \left ( \frac{2^2 + 2}{2} \right)^2$

Length 3 episode

$\frac{1}{2} \cdot 0.9 \cdot \frac{1}{2} \cdot 0.9 \cdot \frac{1}{2} \cdot 0.1 \left ( \frac{2^3 + 2^2 + 2}{3} \right)^2$

Length N episode

$=0.1 \left ( \frac{1}{2} \right ) ^N 0.9^{N-1} \left ( \frac{\sum_{i=1}^N 2^i}{N} \right ) ^2$

So the expected value is the sum of these terms for every possible episode length

$=0.1 \sum_{k=1}^\infty \left ( \frac{1}{2} \right ) ^k 0.9^{k-1} \left ( \frac{\sum_{i=1}^k 2^i}{k} \right ) ^2$

$=0.05 \sum_{k=1}^\infty .9^{k-1} \frac{1}{k^2}  2^{1-k} \left( \sum_{i=1}^k 2^i \right ) ^2$

$>0.05 \sum_{k=1}^\infty .9^{k-1} \frac{1}{k^2}  2^{1-k} 2^{2k}$

$=0.05 \sum_{k=1}^\infty .9^{k-1} \frac{1}{k^2}  2^{k+1}$

$=0.2 \sum_{k=1}^\infty 1.8^{k-1} \frac{1}{k^2}$

The expected value in question is greater than this expression, but as k approaches infinity, each term diverges so the expected value still diverges with every-visit MC.
"""

# ╔═╡ fca388bd-eb8b-41e2-8da0-87b9386629c1
md"""
## 5.6 Incremental Implementation
"""

# ╔═╡ b57462b6-8f9c-4553-9c05-134ff043b00d
md"""
> *Exercise 5.9* Modify the algorithm for first-visit MC policy evaluation (section 5.1) to use the incremental implementation for sample averages described in Section 2.4

Returns(s) will not maintain a list but instead be a list of single values for each state.  Additionally, another list Counts(s) should be initialized at 0 for each state.  When new G values are obtained for state, the Count(s) value should be incremented by 1.  Then Returns(s) can be updated with the following formula: $\text{Returns}(s) = \left [ \text{Returns}(s) \times (\text{Count}(s) - 1) + G \right ] / \text{Count}(s)$
"""

# ╔═╡ fa49c253-e016-46a7-ba94-0e7448a7e0ae
md"""
> *Exercise 5.10* Derive the weighted-average update rule (5.8) from (5.7).  Follow the pattern of the derivation of the unweighted rule (2.3).

Equation (5.7)

$V_{n} = \frac{\sum_{k=1}^{n-1} W_k G_k}{\sum_{k=1}^{n-1}W_k}$

or

$V_{n+1} = \frac{\sum_{k=1}^{n} W_k G_k}{\sum_{k=1}^{n}W_k}$

now we can expand the expresion for $V_{n+1}$ to get an incremental rule

$V_{n+1} = \frac{W_n G_n + \sum_{k=1}^{n-1} W_k G_k}{\sum_{k=1}^{n}W_k}$
$V_{n+1} = \frac{W_n G_n + V_n \sum_{k=1}^{n-1}W_k}{\sum_{k=1}^{n}W_k}$
$V_{n+1} = \frac{W_n G_n + V_n \sum_{k=1}^{n}W_k - V_n W_n}{\sum_{k=1}^{n}W_k}$
$V_{n+1} = V_n + W_n\frac{G_n - V_n}{\sum_{k=1}^{n}W_k}$

For a fully incremental rule we also have to replace the sum over $W_k$ which can simply be a running total.

$C_n = \sum_{k=1}^n W_k$

the following update rule will produce an equivalent $C_n$ assuming we take $C_0=0$

$C_n = C_{n-1} + W_n$

Now we can rewrite our last expression for $V_{n+1}$

$V_{n+1} = V_n + \frac{W_n}{C_n}(G_n - V_n)$
"""

# ╔═╡ e67d9deb-a074-48ba-84db-2e6938ea01f8
function monte_carlo_Q_pred(π_target, π_behavior, states, actions, simulator, γ, nmax = 1000; gets0 = () -> rand(states))
	#initialize
	Q = Dict((s, a) => 0.0 for s in states for a in actions)
	counts = Dict((s, a) => 0.0 for s in states for a in actions)
	adict = Dict(a => i for (i, a) in enumerate(actions))
	avec = collect(actions)
	sample_b(s) = sample(avec, weights(π_behavior[s]))
	for i in 1:nmax
		s0 = gets0()
		a0 = sample_b(s0)
		(traj, rewards) = simulator(s0, a0, sample_b)
		
		#there's no check here so this is equivalent to every-visit estimation
		function updateQ!(t = length(traj); g = 0.0, w = 1.0)
			#terminate at the end of a trajectory or when w = 0
			((t == 0) || (w == 0)) && return nothing
			#accumulate future discounted returns
			g = γ*g + rewards[t]
			(s,a) = traj[t]
			counts[(s, a)] += w
			Q[(s, a)] += (g - Q[(s, a)])*w/counts[(s, a)] #update running average of V
			w *= π_target[s][adict[a]] / π_behavior[s][adict[a]]
			updateQ!(t-1, g = g, w = w)
		end
		#update value function for each trajectory
		updateQ!()
	end
	return Q
end

# ╔═╡ a47474b0-f262-4453-a116-addc3a09119e
# ╠═╡ disabled = true
#=╠═╡
q_offpol = monte_carlo_Q_pred(π_blackjack1, Dict(s => [0.5, 0.5] for s in blackjackstates), blackjackstates, blackjackactions, blackjackepisode, 1.0, 10_000_000, gets0 = () -> (13, 2, true))
  ╠═╡ =#

# ╔═╡ 2ed0d4bb-bb6e-4adf-ad7c-8ddeb8c20b84
#=╠═╡
q_offpol[((13, 2, true), :hit)] #should converge to -0.27726 same as the value function for the policy that hits on this state
  ╠═╡ =#

# ╔═╡ 3558aaea-2594-4042-99ea-c373ed304850
#=╠═╡
q_offpol[((13, 2, true), :stick)] #should be a lower value estimate because sticking is a worse action than hitting
  ╠═╡ =#

# ╔═╡ 35914757-6af1-4056-bba4-a9996f65f7f7
monte_carlo_Q_pred(onestate_π_target, onestate_π_b, [0], one_state_actions, one_state_simulator, 1.0, 10000, gets0 = () -> 0)

# ╔═╡ 7c9486cd-1916-4e13-b415-fb113bd9e04b
md"""
## 5.7 Off-policy Monte Carlo Control
"""

# ╔═╡ 924f4e74-e3a0-42eb-89da-1f9836275588
function off_policy_MC_control(states, actions, simulator, γ, nmax = 1000; gets0 = () -> rand(states))
	#initialize
	nact = length(actions)
	avec = collect(actions)
	π_b = Dict(s => ones(nact)./nact for s in states)
	Q = Dict((s, a) => 0.0 for s in states for a in actions)
	counts = Dict((s, a) => 0.0 for s in states for a in actions)
	adict = Dict(a => i for (i, a) in enumerate(actions))
	sample_b(s) = sample(avec, weights(π_b[s]))
	π_star = Dict(s => rand(actions) for s in states)
	for i in 1:nmax
		s0 = gets0()
		a0 = sample_b(s0)
		(traj, rewards) = simulator(s0, a0, sample_b)
		
		#there's no check here so this is equivalent to every-visit estimation
		function updatedicts!(t = length(traj); g = 0.0, w = 1.0)
			t == 0 && return nothing
			g = γ*g + rewards[t]
			(s,a) = traj[t]
			counts[(s,a)] += w
			Q[(s,a)] += (g - Q[(s,a)])*w/counts[(s,a)]
			astar = argmax(a -> Q[(s,a)], actions)
			π_star[s] = astar 
			a != astar && return nothing
			w /= π_b[s][adict[a]]
			updatedicts!(t-1, g=g, w=w)
		end
		updatedicts!()
	end
	return π_star, Q
end

# ╔═╡ 39da4cf9-5265-41bc-83d4-311e86675db7
# ╠═╡ disabled = true
#=╠═╡
(πstar_blackjack3, Qstar_blackjack3) = off_policy_MC_control(blackjackstates, blackjackactions, blackjackepisode, 1.0, 10_000_000)
  ╠═╡ =#

# ╔═╡ 53487950-f5c4-4715-a4e4-1bf2fd91b213
#=╠═╡
#recreation of figure 5.2 using off-policy method
plot_blackjack_policy(πstar_blackjack3)
  ╠═╡ =#

# ╔═╡ ab10ccd7-75ba-475c-af26-f8b36daaf880
md"""
> *Exercise 5.11* In the boxed algorithm for off-policy MC control, you may have been expecting the $W$ update to have involved the importance-sampling ratio $\frac{\pi(A_t|S_t)}{b(A_t|S_T)}$, but instead it involves $\frac{1}{b(A_t|S_t)}$.  Why is this nevertheless correct?

The target policy $\pi(s)$ is always deterministic, only selecting a single action according to $\pi(s)=\text{argmax}_a Q(s,a)$.  Therefore the numerator in importance-sampling ratio will either be 1 when the trajectory action matches the one given by $\pi(s)$ or it will be 0.  The inner loop will exit if such as action is selected as it will result in zero values of W for the rest of the trajectory and thus no further updates to $Q(s,a)$ or $\pi(s)$.  The only value of $\pi(s)$ that would be encountered in the equation is therefore 1 which is why the numerator is a constant.
"""

# ╔═╡ cebb79b7-c9d6-4b79-ba0e-b2f3c7587724
md"""
> *Exercise 5.12: Racetrack (programming)*  Consider driving a race car around a turn like those shown in Figure 5.5.  You want to go as fast as possible, but not so fast as to run off the track.  In our simplified racetrack, the car is at one of a discrete set of grid positions, the cells in the diagram.  The velocity is also discrete, a number of grid cells moved horizontally and vertically per time step.  The actions are increments to the velocity components.  Each may be changed by +1, -1, or 0 in each step, for a total of nine (3x3) actions.  Both velocity components are restricted to be nonnegative and less than 5, and they cannot both be zero except at the starting line.  Each episode begins in one of the randomly selected start states with both velocity components zero and ends when the car crosses the finish line.  The rewards are -1 for each step until the car crosses the finish line.  If the car hits the track boundry, it is moved back to a random position on the starting line, both velocity components are reduced to zero, and the episode continues.  Before updating the car's location at each time step, check to see if the projected path of the car intersects the track boundary.  If it intersects the finish line, the episode ends; if it intersects anywhere else, the car is considered to have hit the track boundary and is sent back to the starting line.  To make the task more challenging, with probality 0.1 at each time step the velocity increments are both zero, independently of the intended increments.  Apply a Monte Carlo control method to this task to compute the optimal policy from each starting state.  Exhibit several trajectories following the optimal policy (but turn the noise off for these trajectories).
"""

# ╔═╡ 67e535c7-437a-49ba-b5ab-9dfe63fe4aaa
md"""
See code below to create racetrack environment
"""

# ╔═╡ 63814164-f305-49b3-ab51-675c822d7b18
const racetrack_velocities = [(vx, vy) for vx in 0:4 for vy in 0:4]

# ╔═╡ bb617e58-2700-4f0d-b8c8-3a266142fb70
const racetrack_actions = [(dx, dy) for dx in -1:1 for dy in -1:1]

# ╔═╡ 6dc0de46-2164-4580-b186-a73cb5b5167d
#given a position, velocity, and action takes a forward step in time and returns the new position, new velocity, and a set of points that represent the space covered in between
function project_path(p, v, a)
    (vx, vy) = v
    (dx, dy) = a

    vxnew = clamp(vx + dx, 0, 4)
    vynew = clamp(vy + dy, 0, 4)

    #ensure that the updated velocities are not 0
    if vxnew + vynew == 0
        if iseven(p[1] + p[2])
            vxnew += 1
        else
            vynew += 1
        end
    end

    #position the car ends up at
    pnew = (p[1] + vxnew, p[2] + vynew)

    #how to check if the path intersects the finish line or the boundary?  Form a square from vxnew and vynew and see if the off-track area or finish line is contained in that square
    pathsquares = Set((x, y) for x in p[1]:pnew[1] for y in p[2]:pnew[2])

    (pnew, (vxnew, vynew), pathsquares)
end

# ╔═╡ df3c4a33-45f1-4cc2-8c06-d500a0eecc8f
#track is defined as a set of points for each of the start, body, and finish
const track1 = (  start = Set((x, 0) for x in 0:5), 
            finish = Set((13, y) for y in 26:31), 
            body = union(   Set((x, y) for x in 0:5 for y in 1:2),
                            Set((x, y) for x in -1:5 for y in 3:9),
                            Set((x, y) for x in -2:5 for y in 10:17),
                            Set((x, y) for x in -3:5 for y in 18:24),
                            Set((x, 25) for x in -3:6),
                            Set((x, y) for x in -3:12 for y in 26:27),
                            Set((x, 28) for x in -2:12),
                            Set((x, y) for x in -1:12 for y in 29:30),
                            Set((x, 31) for x in 0:12))
        )

# ╔═╡ b9ad0bd3-236c-4ca0-b057-b5f1e53f3e48
#convert a track into a grid for plotting purposes
function get_track_square(track)
    trackpoints = union(track...)
    xmin, xmax = extrema(p -> p[1], trackpoints)
    ymin, ymax = extrema(p -> p[2], trackpoints)

    w = xmax - xmin + 1
    l = ymax - ymin + 1

    trackgrid = Matrix{Int64}(undef, w, l)
    for x in 1:w for y in 1:l
            p = (x - 1 + xmin, y - 1 + ymin)
            val = if in(p, track.start)
                0
            elseif in(p, track.finish)
                2
            elseif in(p, track.body)
                1
            else
                -1
            end
            trackgrid[x, y] = val
    end end

    return trackgrid
end 

# ╔═╡ 82d8e3c7-a3ec-4293-864a-5226428a1803
const track1grid = get_track_square(track1)

# ╔═╡ 524cf31e-b08a-4d9d-b74b-788bc955bfba
#visualization of first track in book with the starting line and finish line in purple and yellow respectively.
heatmap(track1grid', legend = false, size = 20 .* (size(track1grid)))

# ╔═╡ 658ceeaa-1b45-47bd-a364-eaa1759d17ac
#starting in state s0 and with policy π, complete a single episode on given track returning the trajectory and rewards
function race_track_episode(s0, a0, π, track; maxsteps = Inf, failchance = 0.1)
    # @assert in(s0.position, track.start)
    # @assert s0.velocity == (0, 0)

    #take a forward step from current state returning new state and whether or not the episode is over
    function step(s, a)
        pnew, vnew, psquare = project_path(s.position, s.velocity, a)
        fsquares = intersect(psquare, track.finish)
        outsquares = setdiff(psquare, track.body, track.start)
        if !isempty(fsquares) #car finished race
            ((position = first(fsquares), velocity = (0, 0)), true)
        elseif !isempty(outsquares) #car path went outside of track
            ((position = rand(track1.start), velocity = (0, 0)), false)
        else
            ((position = pnew, velocity = vnew), false)
        end
    end
	
    traj = [(s0, a0)]
    rewards = Vector{Float64}()

    function get_traj(s, a, nstep = 1)
        (snew, isdone) = step(s, a)
		push!(rewards, -1.0)
		while !isdone && (nstep < maxsteps)
        	anew = π(snew)
        	push!(traj, (snew, anew))
			(snew, isdone) = step(snew, rand() > failchance ? anew : (0, 0))
			push!(rewards, -1.0)
			nstep += 1
		end
    end

    isdone = get_traj(s0, a0)

    return traj, rewards
end

# ╔═╡ 0dfd7afb-127c-4afd-8374-3c9a20a9ee76
π_racetrack_rand(s) = rand(racetrack_actions)

# ╔═╡ 3955d71a-3105-445a-868d-66ba0b3dc515
race_episode = race_track_episode((position = rand(track1.start), velocity = (0, 0)), rand(racetrack_actions), π_racetrack_rand, track1)

# ╔═╡ 5e2420fb-b2cc-49fc-91e3-3de80fba698b
#run a single race episode from a valid starting position with a given policy and track
function runrace(π, track = track1)
	s0 = (position = rand(track.start), velocity = (0, 0))
	a0 = π(s0)
	race_track_episode(s0, a0, π, track, maxsteps = 100000, failchance = 0.0)
end

# ╔═╡ 8f38a3e9-1872-4ea6-9a03-87112af4bf07
#run n episodes of a race and measure the statistics of the time required to finish
function sampleracepolicy(π, n = 1000)
	trajs = [runrace(π)[1] for _ in 1:n]
	ls = length.(trajs)
	extrema(ls), mean(ls), var(ls)
end

# ╔═╡ 7a2e6009-4370-4cb9-a4cd-9ee5aba8c09b
# ╠═╡ disabled = true
#=╠═╡
#using a random policy, the mean time to finish on track 1 is ~2800 steps.  The best possible time when we get "lucky" with random decisions is ~12 steps with worst times ~15-30k steps
sampleracepolicy(s -> rand(racetrack_actions), 10_000)
  ╠═╡ =#

# ╔═╡ 5b1d5d03-b7cf-42e0-bd5b-f3b4fff12df2
const track1states = [(position = p, velocity = v) for p in union(track1.start, track1.body) for v in racetrack_velocities]

# ╔═╡ dfc2d648-ec08-49cd-a55f-72a766cad728
# ╠═╡ disabled = true
#=╠═╡
(πstar_racetrack1, Qstar_racetrack1) = off_policy_MC_control(track1states, racetrack_actions, (s, a, π) -> race_track_episode(s, a, π, track1), 1.0, 10_000)
  ╠═╡ =#

# ╔═╡ 9524a3f5-fca3-4ce9-b04d-0cb6c0cd0c90
#=╠═╡
#off policy control doesn't produce a policy that can finish the race.  A cutoff of 100k steps is used to ensure the system doesn't run forever
runrace(s -> πstar_racetrack1[s])
  ╠═╡ =#

# ╔═╡ 68f9e0d9-5c9d-4e89-b5ae-f24fd7544a09
# ╠═╡ disabled = true
#=╠═╡
(πstar_racetrack2, Qstar_racetrack2) = monte_carlo_ES(track1states, racetrack_actions, (s, a, π) -> race_track_episode(s, a, π, track1), 1.0, 100_000)
  ╠═╡ =#

# ╔═╡ 3d10f89f-876e-4d25-b8d6-34ce5c99eb8c
#=╠═╡
#exploring starts on policy training also doesn't produce a policy that can finish the race
runrace(s -> πstar_racetrack2[s])
  ╠═╡ =#

# ╔═╡ 1d6eccf0-2731-47fc-9a41-ea8649e290ef
# ╠═╡ disabled = true
#=╠═╡
(πstar_racetrack3, Qstar_racetrack3) = monte_carlo_ϵsoft(track1states, racetrack_actions, (s, a, π) -> race_track_episode(s, a, π, track1), 1.0, 0.25, 10_000_000, gets0 = () -> (position = rand(track1.start), velocity = (0, 0)))
  ╠═╡ =#

# ╔═╡ 7212f887-5347-4ee1-90b8-43d282f0fa6e
#=╠═╡
runrace(s -> πstar_racetrack3[s])
  ╠═╡ =#

# ╔═╡ 985f0537-2bbe-4dbb-a113-8ac98d2e0a5f
#=╠═╡
sampleracepolicy(s -> πstar_racetrack3[s], 10_000)
  ╠═╡ =#

# ╔═╡ 84c28558-9d21-4393-8d5d-65ad9234aadd
function plotpolicy(π)
	x = [a[1] for a in union(track1...)]
	y = [a[2] for a in union(track1...)]
	dv = Dict(a => (0.0, 0.0) for a in union(track1...))
	v = Dict(a => (0.0, 0.0) for a in union(track1...))
	cv = Dict(a => 0 for a in union(track1...))
	for a in keys(π)
		(dx, dy) = π[a]
		(vx, vy) = a.velocity
		p = a.position
		cv[p] += 1
		dv[p] = ((dv[p][1] * (cv[p] - 1) + dx) / cv[p],  (dv[p][2] * (cv[p] - 1) + dy) / cv[p])
		v[p] = ((v[p][1] * (cv[p] - 1) + vx) / cv[p],  (v[p][2] * (cv[p] - 1) + vy) / cv[p])
	end
	dx = [dv[a][1] for a in zip(x, y)]
	dy = [dv[a][2] for a in zip(x, y)]
	vx = [v[a][1] for a in zip(x,y)]
	vy = [v[a][2] for a in zip(x,y)]
	quiver(x, y, quiver = (dx, dy))
end

# ╔═╡ 666beaf1-366b-4338-acb9-750e203b2185
function plotpolicy2(π)
	positions = union(track1.start, track1.body)
	x = [a[1] for a in positions]
	y = [a[2] for a in positions]
	dv = Dict(a => (0.0, 0.0) for a in positions)
	cv = Dict(a => 0 for a in positions)
	for p in positions
		s = (position = p, velocity = (1, 0))
		(dx, dy) = π[s]
		dv[p] = (dx, dy)
	end
	dx = [dv[a][1] for a in zip(x, y)]
	dy = [dv[a][2] for a in zip(x, y)]
	quiver(x, y, quiver = (dx, dy))
end

# ╔═╡ cc49451a-a3de-4341-ac4f-3897cbc321d6
# ╠═╡ disabled = true
#=╠═╡
plotpolicy2(πstar_racetrack3)
  ╠═╡ =#

# ╔═╡ 3558d0be-f51b-472d-8fd7-6213a3f0c4af
function visualize_policy_traj(π)
	fig = heatmap(track1grid', legend = false, size = 20 .* (size(track1grid)))
	for i in 0:4 #cycle through starting positions
		s0 = (position = (i, 0), velocity = (0, 0))
		a0 = π[s0]
		race_episode_star = race_track_episode(s0, a0, s -> π[s], track1, maxsteps = 10000, failchance = 0.0)
		plot!([t[1].position .+ (4, 1) for t in race_episode_star[1]], color = :green)
	end
	plot(fig)
end

# ╔═╡ 0a1ee6fb-b86d-4e82-a662-e31b4f55e3a9
function visualize_policy_traj2(π)
	fig = heatmap(track1grid', legend = false, size = 20 .* (size(track1grid)))
	s0 = (position = (2, 0), velocity = (0, 0))
	a0 = π[s0]
	race_episode_star = race_track_episode(s0, a0, s -> π[s], track1, maxsteps = 100, failchance = 0.0)
	x = [t[1].position[1] + 4 for t in race_episode_star[1]]
	y = [t[1].position[2] + 1 for t in race_episode_star[1]]
	vx = [t[1].velocity[1] for t in race_episode_star[1]]
	vy = [t[1].velocity[2] for t in race_episode_star[1]]
	dx = [t[2][1] for t in race_episode_star[1]]
	dy = [t[2][2] for t in race_episode_star[1]]
	quiver!(x, y, quiver = (vx, vy))
	quiver!(x, y, quiver = (dx, dy), linecolor = :green)
	plot(fig)
end

# ╔═╡ 7c81169f-5ac3-4660-8b0c-8ac737d86271
#=╠═╡
visualize_policy_traj2(πstar_racetrack1)
  ╠═╡ =#

# ╔═╡ 8645934b-a141-41e4-ae5b-6a0a86a94156
#=╠═╡
visualize_policy_traj2(πstar_racetrack2)
  ╠═╡ =#

# ╔═╡ 2a13fdd4-97b5-4b65-8cb5-a0124ecf3dac
#=╠═╡
visualize_policy_traj(πstar_racetrack3)
  ╠═╡ =#

# ╔═╡ 85205698-76a3-4901-8caf-e6e6fc5524ee
#=╠═╡
#trajectory of a successful race policy, black arrows indicate velocity, green arrows indicate action.  Note that negative velocities are forbidden so any arrow pointing left on a vertical trajectory will have no impact.
visualize_policy_traj2(πstar_racetrack3)
  ╠═╡ =#

# ╔═╡ 859354fe-7f40-4658-bf12-b5ee20a815a7
md"""
## 5.8 Discounting-aware Importance Sampling
"""

# ╔═╡ 88335fca-fd87-487b-9de2-ea7c779b54cf
md"""
## 5.9 Per-decision Importance Sampling
"""

# ╔═╡ abe70666-39f8-4f1d-a285-a3a99f696d10
md"""
> *Exercise 5.13* Show the steps to derive (5.14) from (5.12)

Starting at (5.12)

$\rho_{t:T-1}R_{t+1}=\frac{\pi(A_t|S_t)}{b(A_t|S_t)}\frac{\pi(A_{t+1}|S_{t+1})}{b(A_{t+1}|S_{t+1})}\frac{\pi(A_{t+2}|S_{t+2})}{b(A_{t+2}|S_{t+2})}\cdots\frac{\pi(A_{T-1}|S_{T-1})}{b(A_{T-1}|S_{T-1})}R_{t+1}$

For (5.14) we need to turn this into an expected value

$\mathbb{E}[\rho_{t:T-1}R_{t+1}]$

Now we know that the reward at time step t+1 is only dependent on the action and state at time t.  Moreover, the later parts of the trajectory are also independent of each other.  So we can separate some of these terms into a product of expected values rather than an expected value of products:

$\mathbb{E}[\rho_{t:T-1}R_{t+1}]=\mathbb{E} \left [ \frac{\pi(A_t|S_t)}{b(A_t|S_t)}\frac{\pi(A_{t+1}|S_{t+1})}{b(A_{t+1}|S_{t+1})}\frac{\pi(A_{t+2}|S_{t+2})}{b(A_{t+2}|S_{t+2})}\cdots\frac{\pi(A_{T-1}|S_{T-1})}{b(A_{T-1}|S_{T-1})}R_{t+1} \right ]$

$=\mathbb{E} \left [ \frac{\pi(A_t|S_t)}{b(A_t|S_t)}R_{t+1} \right ] \prod_{k=t+1}^{T-1}\mathbb{E} \left [ \frac{\pi(A_k|S_k)}{b(A_k|S_k)} \right ]$

We know from (5.13) that $\mathbb{E} \left [ \frac{\pi(A_k|S_k)}{b(A_k|S_k)} \right ] = 1$ so the above expression simplifies to: $\mathbb{E} \left [ \frac{\pi(A_t|S_t)}{b(A_t|S_t)}R_{t+1} \right ]$.  Using the original shorthand with ρ:

$\mathbb{E}[\rho_{t:T-1}R_{t+1}]=\mathbb{E} \left [ \frac{\pi(A_t|S_t)}{b(A_t|S_t)}R_{t+1} \right ]=\mathbb{E}[\rho_{t:t}R_{t+1}]$
"""

# ╔═╡ dc57a71f-e44c-4385-ad2a-e6c14d5e5201
md"""
> *Exercise 5.14* Modify the algorithm for off-policy Monte Carlo control (page 111) to use the idea of the truncated weighted-average estimator (5.10).  Note that you will first need to convert this equation to action values.

Equation (5.10)

$V(s) = \frac{\sum_{t \in \mathscr{T}(s)}\left((1-\gamma)\sum_{h=t+1}^{T(t)-1}\gamma^{h-t-1} \rho_{t:h-1} \bar{G}_{t:h} + \gamma^{T(t)-t-1} \rho_{t:T(t)-1} \bar{G}_{t:T(t)} \right)}{\sum_{t \in \mathscr{T}(s)} \left( (1-\gamma) \sum_{h=t+1}^{T(t)-1} \gamma^{h-t-1} \rho_{t:h-1} + \gamma^{T(t)-t-1} \rho_{t:T(t)-1} \right)}$

Converting this to action-value estimates:

$Q(s,a) = \frac{\sum_{t \in \mathscr{T}(s,a)}\left( R_{t+1} + (1-\gamma)\sum_{h=t+2}^{T(t)-1}\gamma^{h-t-1} \rho_{t+1:h-1} \bar{G}_{t+1:h} + \gamma^{T(t)-t-1} \rho_{t+1:T(t)-1} \bar{G}_{t+1:T(t)} \right)}{\sum_{t \in \mathscr{T}(s,a)} \left( 1 + (1-\gamma) \sum_{h=t+2}^{T(t)-1} \gamma^{h-t-1} \rho_{t+1:h-1} + \gamma^{T(t)-t-1} \rho_{t+1:T(t)-1} \right)}$

For the algorithm on page 111, need to add a variable in the loop to keep track of Ḡ both from the start of the episode forwards.  The inner loop should also start from the beginning of each episode and go forwards rather than starting at the end going backwards.  The term added to the numerator and denominator will be ready including Ḡ and ρ once the end of the episode is reached.  A γ accumulator can be initiazed at 1 and kept track of in the inner loop by repeatedly multiplying by γ each iteration.
"""

# ╔═╡ 2e57e4b5-d228-4ce3-b9d8-cf4375bb7c50
md"""
# Dependencies and Settings
"""

# ╔═╡ 1f2a62c0-b78d-4f9b-8c89-d610d168cb88
TableOfContents()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.3.1"
PlutoPlotly = "~0.4.4"
PlutoUI = "~0.7.50"
StatsBase = "~0.33.21"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0"
manifest_format = "2.0"
project_hash = "0db33129a411493ff9d6d371968e3a831122631a"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

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
deps = ["UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+1"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

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
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

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
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

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
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "478ac6c952fddd4399e71d4779797c538d0ff2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.8"

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

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "5bb5129fdd62a2bbbe17c2756932259acf467386"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.50"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

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

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

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
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

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
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

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
# ╟─826139cc-b52e-11ec-0d47-25ab689851fd
# ╠═a85e7dd3-ea5e-4c77-a18e-f1e190658ae3
# ╠═72e9721f-ee17-4557-b090-71728e6c22ce
# ╠═21d248f5-edad-4614-8c1c-ae330f9e5a18
# ╠═b2115f33-f8e2-452e-9b0e-90610f0f09b1
# ╠═202eb066-877d-4537-85b2-31b41db8eca0
# ╠═d55860d1-e4c1-4a79-adbe-b40a6d6283a7
# ╠═11723075-d1db-4512-abf2-3fe494a71a3b
# ╠═760c5361-02d4-46b7-a05c-fc2d10d93de6
# ╟─6a11daf7-2859-41fa-9c3d-d1f3580dbb5f
# ╠═e87932d9-fbc0-4ea5-a8ee-28eac5eed84f
# ╠═481ad435-cc80-4164-882d-8310b010ca91
# ╠═7fb4244c-828a-49d6-a15b-178fa3a42e00
# ╠═c23475f0-c5f0-49ce-a665-f93c8bda0474
# ╠═6d765229-2816-4abb-a868-6be919a96530
# ╠═a35de859-4046-4f6d-9ea9-b523d21cee5d
# ╠═94be5289-bba7-4490-bdcd-0d217a31c665
# ╠═01683e11-a368-45bc-abbc-bbd5c94d7b22
# ╠═a68b4965-c392-47e5-9b29-93e7ada9990a
# ╟─d766d44e-3684-497c-814e-8f71740cb232
# ╟─82284c63-2306-4469-9b1a-a5ec87037e79
# ╟─e5384dd0-fad1-4a24-b011-73b062fcfb1b
# ╟─30809344-b4ab-468b-b4b7-5ef3dca5ffc7
# ╟─f406be9e-3e3f-4b55-99b0-4858c774ed96
# ╟─be7c096c-cc8c-407b-8287-8fb2ee7150a7
# ╟─2572e35c-e6a3-4562-aa0f-6a5ab32d39ea
# ╟─47daf83b-8fe9-4491-b9ae-84bd269d5546
# ╠═2d10281a-a4af-4ea8-b63b-e11f2d0893ed
# ╠═13cc524c-d983-44f4-8731-0595249fb888
# ╠═1b78b25d-3942-4a6b-a2bd-7d97242da9fe
# ╟─9618a093-cdb7-4589-a783-de8e9021b705
# ╠═5c57e4b7-51d6-492d-9fc9-bcdab1dd46f4
# ╠═06960937-a87a-4015-bdeb-5d17aa41fe6b
# ╠═627d5aa3-974f-4949-8b65-9500eba1d7cc
# ╠═ec29865f-3ba3-4bb3-84df-c2b472e03ff2
# ╟─9fe42679-dce3-4ee3-b565-eed4ff7d8e4d
# ╟─f9bc84ff-c2f9-4ac2-b2ce-34dfcf837a73
# ╟─c883748c-76b9-4086-8698-b40df51390da
# ╟─e2a720b0-a8c4-43a2-bf34-750ff3323004
# ╠═38957d8d-e1d0-44c3-bac9-1417925ac882
# ╠═bf19e6cf-1fb5-49c9-974e-1613d90ef4cf
# ╠═b040f245-b2d6-4ec6-aa7f-511c54aabd0d
# ╟─12c0cd0b-eb4f-45a0-836b-53b3c5cdafd9
# ╟─9793b5c9-d4ec-492d-a72d-8737bb65c8a5
# ╟─15925cc6-9605-4357-9c2a-cdfe54070989
# ╠═d97f693d-27f2-49be-a549-07a290c95b53
# ╠═660ef59c-205c-44c2-9c46-5a74e09497ab
# ╠═4197cc28-b24c-48cb-bd8d-ef998983ad77
# ╠═6be7ff29-5845-4df4-ba18-bafea79ace71
# ╟─cede8090-5b1a-436b-a184-fca5c4d3de48
# ╟─b39d1ea0-86a2-4215-ae73-e4492f3f2f00
# ╠═a2b7d85e-c9ad-4104-92d6-bac2ce362f1d
# ╠═1d2e91d0-a1a0-4630-ac06-37684a9104f3
# ╠═acb08e38-fc87-43f4-ac2d-6c6bf0f2e9e6
# ╠═2b9131c1-4d79-4ea3-b51f-7f3380aeb629
# ╠═70d9d39f-020d-4f25-810c-82a143a3335b
# ╠═8faca500-b80d-4b50-88b6-683d18a1286b
# ╠═d6863551-a254-44b6-b6fe-551d134cdf01
# ╠═c5482c11-1635-4016-bf6a-4c5f01ae66b9
# ╠═303c852d-177c-4ddc-aa53-b72e6e82cc55
# ╠═74fc1b42-9784-4968-8c85-f3d0b778fa2f
# ╠═00cd2194-af13-415a-b725-bb34832e5d9a
# ╠═1a97b3de-1aaa-4f51-b358-5c5f2b1d0851
# ╟─e10378eb-12b3-4468-9c22-1838107da450
# ╠═f071b7e2-3f78-4132-8b84-f810f178c89d
# ╠═ab2c27c8-122e-4e38-8aab-73291077b640
# ╠═61eb74eb-88e0-42e6-a14b-3730a800694d
# ╠═82a1f11e-b1b3-4c7d-ab9d-9f6136ae8195
# ╠═a7c1e2ae-54ec-407f-8496-4f8a799f8759
# ╠═76877978-9a2b-48ba-aa4e-785622ac8c3d
# ╠═5948f670-1203-4b75-8517-f8470f5d01aa
# ╠═830d6d61-7259-43e7-8d6d-09bc81818dd9
# ╟─881ba6c3-0e6a-4bb8-bc14-2e0face560f2
# ╟─6979db32-670d-466a-9a6e-97c2d8527f3d
# ╟─bf33230e-342b-4486-89ac-9667438de503
# ╟─fca388bd-eb8b-41e2-8da0-87b9386629c1
# ╟─b57462b6-8f9c-4553-9c05-134ff043b00d
# ╟─fa49c253-e016-46a7-ba94-0e7448a7e0ae
# ╠═e67d9deb-a074-48ba-84db-2e6938ea01f8
# ╠═a47474b0-f262-4453-a116-addc3a09119e
# ╠═2ed0d4bb-bb6e-4adf-ad7c-8ddeb8c20b84
# ╠═3558aaea-2594-4042-99ea-c373ed304850
# ╠═35914757-6af1-4056-bba4-a9996f65f7f7
# ╟─7c9486cd-1916-4e13-b415-fb113bd9e04b
# ╠═924f4e74-e3a0-42eb-89da-1f9836275588
# ╠═39da4cf9-5265-41bc-83d4-311e86675db7
# ╠═53487950-f5c4-4715-a4e4-1bf2fd91b213
# ╟─ab10ccd7-75ba-475c-af26-f8b36daaf880
# ╟─cebb79b7-c9d6-4b79-ba0e-b2f3c7587724
# ╟─67e535c7-437a-49ba-b5ab-9dfe63fe4aaa
# ╠═63814164-f305-49b3-ab51-675c822d7b18
# ╠═bb617e58-2700-4f0d-b8c8-3a266142fb70
# ╠═6dc0de46-2164-4580-b186-a73cb5b5167d
# ╠═df3c4a33-45f1-4cc2-8c06-d500a0eecc8f
# ╠═b9ad0bd3-236c-4ca0-b057-b5f1e53f3e48
# ╠═82d8e3c7-a3ec-4293-864a-5226428a1803
# ╠═524cf31e-b08a-4d9d-b74b-788bc955bfba
# ╠═658ceeaa-1b45-47bd-a364-eaa1759d17ac
# ╠═0dfd7afb-127c-4afd-8374-3c9a20a9ee76
# ╠═3955d71a-3105-445a-868d-66ba0b3dc515
# ╠═5e2420fb-b2cc-49fc-91e3-3de80fba698b
# ╠═8f38a3e9-1872-4ea6-9a03-87112af4bf07
# ╠═7a2e6009-4370-4cb9-a4cd-9ee5aba8c09b
# ╠═5b1d5d03-b7cf-42e0-bd5b-f3b4fff12df2
# ╠═dfc2d648-ec08-49cd-a55f-72a766cad728
# ╠═9524a3f5-fca3-4ce9-b04d-0cb6c0cd0c90
# ╠═68f9e0d9-5c9d-4e89-b5ae-f24fd7544a09
# ╠═3d10f89f-876e-4d25-b8d6-34ce5c99eb8c
# ╠═1d6eccf0-2731-47fc-9a41-ea8649e290ef
# ╠═7212f887-5347-4ee1-90b8-43d282f0fa6e
# ╠═985f0537-2bbe-4dbb-a113-8ac98d2e0a5f
# ╠═84c28558-9d21-4393-8d5d-65ad9234aadd
# ╠═666beaf1-366b-4338-acb9-750e203b2185
# ╠═cc49451a-a3de-4341-ac4f-3897cbc321d6
# ╠═3558d0be-f51b-472d-8fd7-6213a3f0c4af
# ╠═0a1ee6fb-b86d-4e82-a662-e31b4f55e3a9
# ╠═7c81169f-5ac3-4660-8b0c-8ac737d86271
# ╠═8645934b-a141-41e4-ae5b-6a0a86a94156
# ╠═2a13fdd4-97b5-4b65-8cb5-a0124ecf3dac
# ╠═85205698-76a3-4901-8caf-e6e6fc5524ee
# ╟─859354fe-7f40-4658-bf12-b5ee20a815a7
# ╟─88335fca-fd87-487b-9de2-ea7c779b54cf
# ╟─abe70666-39f8-4f1d-a285-a3a99f696d10
# ╟─dc57a71f-e44c-4385-ad2a-e6c14d5e5201
# ╟─2e57e4b5-d228-4ce3-b9d8-cf4375bb7c50
# ╠═77cf7fee-0ad8-4d22-b376-75833307db93
# ╠═1f2a62c0-b78d-4f9b-8c89-d610d168cb88
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
