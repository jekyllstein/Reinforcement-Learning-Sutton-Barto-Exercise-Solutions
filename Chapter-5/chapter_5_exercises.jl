### A Pluto.jl notebook ###
# v0.19.3

using Markdown
using InteractiveUtils

# ╔═╡ c044c399-1f07-4312-b18b-4ba2a91d1d71
using StatsBase

# ╔═╡ bd63d4b2-423d-4860-8dd3-3587a124ced5
begin
	using Plots
	plotly()
end

# ╔═╡ c54dba6b-35bc-4537-928c-f25bff2a4a18
using BenchmarkTools

# ╔═╡ 826139cc-b52e-11ec-0d47-25ab689851fd
md"""
# Chapter 5 Monte Carlo Methods
## 5.1 Monte Carlo Prediction
"""

# ╔═╡ 760c5361-02d4-46b7-a05c-fc2d10d93de6
function monte_carlo_pred(π::Dict{T, Vector{Float64}}, states::Vector{T}, actions, simulator::Function, γ, nmax = 1000) where T
	avec = collect(actions)
	sample_π(s) = sample(avec, weights(π[s]))
	
	#initialize
	V = Dict(s => 0.0 for s in states)
	counts = Dict(s => 0 for s in states)
	for i in 1:nmax
		s0 = rand(states)
		a0 = sample_π(s0)
		(traj, rewards) = simulator(s0, a0, sample_π)
		
		#there's no check here so this is equivalent to every-visit estimation
		function updateV!(t = length(traj); g = 0.0)
			#terminate at the end of a trajectory
			t == 0 && return nothing
			#accumulate future discounted returns
			g = γ*g + rewards[t]
			(s,a) = traj[t]
			#increment count by 1
			counts[s] += 1
			V[s] += (g - V[s])/counts[s] #update running average of V
			updateV!(t-1, g = g)
		end

		#update value function for each trajectory
		updateV!()
	end
	return V
end

# ╔═╡ 6a11daf7-2859-41fa-9c3d-d1f3580dbb5f
md"""
### Example 5.1: Blackjack
"""

# ╔═╡ 72e29874-f7cb-4089-bcc4-8e45336cdc23
const cards = (2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, :A)

# ╔═╡ fa0fbe9a-c826-4a9a-b3a2-0af483114055
const blackjackactions = (:hit, :stick)

# ╔═╡ 1049a6e6-7446-49ba-97d5-43b02ace5e39
#deal a card from an infinite deck and return either the value of that card or an ace
deal() = rand(cards)

# ╔═╡ 278ff1d0-9983-4e6d-8a67-2bbe584415bf
const blackjackstates = [(s, c, ua) for s in 12:21 for c in 1:10 for ua in (true, false)]

# ╔═╡ 887c9749-d23d-46cd-9386-8473e3d603a9
#takes a previous sum, usable ace indicator, and a card to be added to the sum.  Returns the updated sum and whether an ace is still usable
function addsum(s::Int64, ua::Bool, c::Symbol)
	if !ua
		s >= 11 ? (s+1, false) : (s+11, true)
	else
		(s+1, true)
	end
end

# ╔═╡ b5c96cca-7f05-45a3-8641-535b41969540
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

# ╔═╡ 1df4cae0-71a9-4598-b941-58a659ba1381
function playersim(state, a, π::Function, traj = [(state, a)])
	(s, c, ua) = state
	a == :stick && return (s, traj)
	(s, ua) = addsum(s, ua, deal())
	(s >= 21) && return (s, traj)
	newstate = (s, c, ua)
	a = π(newstate)
	push!(traj, (newstate, a))
	playersim(newstate, a, π, traj)
end

# ╔═╡ 0e3217d8-1a5e-4387-b4df-3121355d3464
function dealer_sim(s::Int64, ua::Bool)
	(s >= 17) && return s
	(s, ua) = addsum(s, ua, deal())
	dealer_sim(s, ua)
end

# ╔═╡ 2076b266-ec17-4492-814d-8bf8942221ae
#starting with an initial state, action, and policy, generate a trajectory for blackjack returning that and the reward
function blackjackepisode(s0, a0, π::Function)
	#score a game in which the player didn't go bust
	function scoregame(playersum, dealersum)
		#if the dealer goes bust, the player wins
		dealersum > 21 && return 1.0

		#if the player is closer to 21 the player wins
		playersum > dealersum && return 1.0

		#if the dealer sum is closer to 21 the player loses
		playersum < dealersum && return -1.0

		#otherwise the outcome is a draw
		return 0.0
	end
	
	(s, c, ua) = s0
	splayer, traj = playersim(s0, a0, π)
	rewardbase = zeros(length(traj) - 1)
	finalr = if splayer > 21 
		#if the player goes bust, the game is lost regardless of the dealers actions
		-1.0
	else
		#generate hidden dealer card and final state
		hc = deal()
		(ds, dua) = if c == 1
			addsum(11, true, hc)
		else 
			addsum(c, false, hc)
		end

		playernatural = (splayer == 21) && (length(traj) == 1)
		dealernatural = ds == 21

		if playernatural
			Float64(!dealernatural)
		else
			sdealer = dealer_sim(ds, dua)
			scoregame(splayer, sdealer)
		end
	end
	return (traj, [rewardbase; finalr])
end

# ╔═╡ 8f5d1487-0070-422d-a1f8-1b14a312db6c
#policy defined in Example 5.1
const π_blackjack1 = Dict((s, c, ua) => (s >= 20) ? [0.0, 1.0] : [1.0, 0.0] for (s, c, ua) in blackjackstates)

# ╔═╡ e5384dd0-fad1-4a24-b011-73b062fcfb1b
md"""
> *Exercise 5.1* Consider the diagroms on the right in Figure 5.1.  Why does the estimated value function jumpm for the last two rows in the rear?  Why does it drop off for the whole last row on the left?  Why are the frontmots values higher in the upper diagrams than in the lower?

The last two rows in the rear are for a player sum equal to 20 or 21.  Per player policy, any sum less than this will result in a hit.  Sticking on these sums is a good strategy and will likely result in a win, but the policy at 19 and lower is suboptimal.

The far left row represents cases where the dealer is showing an Ace.  Since an Ace is a flexible card, the dealer policy will have more options that result in a win including the possibility of having another face card already.  It is always a bad outcome for the player if the dealer is known to have an Ace.

The frontmost values represent cases where the player sum is 12.  If there is a usable Ace this means that means that the player has two Aces which results in a sum of 12 when the first Ace is counted as 1 and the second is *usable* and counted as 11.  If there is no usable Ace than a sum of 12 would have to result from some other combination of cards such as 10/2, 9/3, etc...  Since the first case has two Aces, it means that potentially both could count as 1 if needed to avoid a bust.  In the case without a usable Ace, the sum is the same, but there are more opportunities to bust if we draw a card worth 10, so having a sum of 12 with a usable Ace is strictly better.
"""

# ╔═╡ 30809344-b4ab-468b-b4b7-5ef3dca5ffc7
md"""
> *Exercise 5.2* Suppose every-visit MC was used instead of first-visit MC on the blackjack task.  Would you expect the results to be very different?  Why or why not?

As an episode proceeds in blackjack the states will not repeat since every time a card is dealt the player sum changes or the usable Ace flag changes.  Thus the check ensuring that only the first visit to a state is counted in the return average will have no effect on the MC evaluation.
"""

# ╔═╡ f406be9e-3e3f-4b55-99b0-4858c774ed96
md"""
## 5.2 Monte Carlo Estimation of Action Values
"""

# ╔═╡ be7c096c-cc8c-407b-8287-8fb2ee7150a7
md"""
> *Exercise 5.3* What is the backup diagram for Monte Carlo estimation of $q_\pi$

Similar to the $v_\pi$ diagram except the root is the s,a pair under consideration followed by the new state and the action taken along the trajectory.  The rewards are still accumulated to the end, just the start of the trajectory is a solid filled in circle that would contain the value for that s,a pair.
"""

# ╔═╡ 47daf83b-8fe9-4491-b9ae-84bd269d5546
md"""
## 5.3 Monte Carlo Control
"""

# ╔═╡ 13cc524c-d983-44f4-8731-0595249fb888
function monte_carlo_ES(states, actions, simulator, γ, nmax = 1000)
	#initialize
	π = Dict(s => rand(actions) for s in states)
	Q = Dict((s, a) => 0.0 for s in states for a in actions)
	counts = Dict((s, a) => 0 for s in states for a in actions)
	for i in 1:nmax
		s0 = rand(states)
		a0 = rand(actions)
		(traj, rewards) = simulator(s0, a0, s -> π[s])

		#there's no check here so this is equivalent to every-visit estimation
		t = length(traj)
		g = 0.0
		while t != 0
			g = γ*g + rewards[t]
			(s,a) = traj[t]
			counts[(s,a)] += 1
			Q[(s,a)] += (g - Q[(s,a)])/counts[(s,a)]
			π[s] = argmax(a -> Q[(s,a)], actions)
			t -= 1
		end
	end
	return π, Q
end

# ╔═╡ 9618a093-cdb7-4589-a783-de8e9021b705
md"""
### Example 5.3: Solving Blackjack
"""

# ╔═╡ ec29865f-3ba3-4bb3-84df-c2b472e03ff2
(πstar_blackjack, Qstar_blackjack) = monte_carlo_ES(blackjackstates, blackjackactions, blackjackepisode, 1.0, 10_000_000)

# ╔═╡ c883748c-76b9-4086-8698-b40df51390da
md"""
> *Exercise 5.4* The pseudocode for Monte Carlo ES is inefficient because, for each state-action pair, it maintains a list of all returns and repeatedly calculates their mean.  It would be more dfficient to use techniques similar to those explained in Section 2.4 to maintain just the mean and a count (for each state-action pair) and update them incrementally.  Describe how the pseudocode would be altered to achieve this.

Returns(s,a) will not maintain a list but instead be a list of single values for each state-action pair.  Additionally, another list Counts(s,a) should be initialized at 0 for each pair.  When new G values are obtained for state-action pairs, the Count(s,a) value should be incremented by 1.  Then Returns(s,a) can be updated with the following formula: $\text{Returns}(s,a) = \left [ \text{Returns}(s,a) \times (\text{Count}(s,a) - 1) + G(s,a) \right ] / \text{Count}(s,a)$

Alternatively, this can be written as:

$\text{Returns}(s,a) = \text{Returns}(s,a) + \frac{G(s,a) - \text{Returns}(s,a)}{\text{Count}(s,a)}$
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
function monte_carlo_pred(π_target, π_behavior, states, actions, simulator, γ, nmax = 1000; gets0 = () -> rand(states), historystate = states[1], samplemethod::ImportanceMethod = Ordinary())
	#initialize values and counts at 0
	V = Dict(s => 0.0 for s in states)
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
function eval_blackjack_policy(π, episodes; γ=1.)
	v_π = monte_carlo_pred(π, blackjackstates, blackjackactions, blackjackepisode, γ, episodes)
	vgridua = zeros(10, 10)
	vgridnua = zeros(10, 10)
	for state in blackjackstates
		(s, c, ua) = state
		if ua
			vgridua[s-11, c] = v_π[state]
		else
			vgridnua[s-11, c] = v_π[state]
		end
	end
	return vgridua, vgridnua
end

# ╔═╡ a68b4965-c392-47e5-9b29-93e7ada9990a
function plot_fig5_1()
	(uagrid10k, nuagrid10k) = eval_blackjack_policy(π_blackjack1, 10_000)
	(uagrid500k, nuagrid500k) = eval_blackjack_policy(π_blackjack1, 500_000)

	p1 = heatmap(uagrid10k, title = "After 10,000 episodes", ylabel = "Usable ace", yticks = false, xticks = false)
	p2 = heatmap(nuagrid10k, ylabel = "No usable ace", yaxis = false, xaxis = false, legend = false)
	p3 = heatmap(uagrid500k, title = "After 500,000 episodes", yaxis = false, xaxis = false, legend = false)
	p4 = heatmap(nuagrid500k, yticks = (1:10, 12:21), xticks = (1:10, ["A", "", "", "", "", "", "", "", "", "10"]), legend = false, xlabel = "Dealer Showing")

	plot(p1, p3, p2, p4, layout = (2, 2))
end

# ╔═╡ 82284c63-2306-4469-9b1a-a5ec87037e79
plot_fig5_1()

# ╔═╡ 5c57e4b7-51d6-492d-9fc9-bcdab1dd46f4
function plot_blackjack_policy(π)
	πstargridua = zeros(Int64, 10, 10)
	πstargridnua = zeros(Int64, 10, 10)
	for state in blackjackstates
		(s, c, ua) = state
		a = π[state]
		if ua
			(a == :hit) && (πstargridua[s-11, c] = 1)
		else
			(a == :hit) && (πstargridnua[s-11, c] = 1)
		end
	end

	vstar = eval_blackjack_policy(Dict(s => π[s] == :hit ? [1.0, 0.0] : [0.0, 1.0] for s in blackjackstates), 500_000)
	p1 = heatmap(πstargridua, xticks = (1:10, ["A", 2, 3, 4, 5, 6, 7, 8, 9, 10]), yticks = (1:10, 12:21), legend = false, title = "Usable Ace Policy, Black=Stick, White = Hit")
	p2 = heatmap(πstargridnua, xticks = (1:10, ["A", 2, 3, 4, 5, 6, 7, 8, 9, 10]), yticks = (1:10, 12:21), legend = false, title = "No usable Ace Policy", xlabel = "Dealer Showing", ylabel = "Player sum")

	p3 = heatmap(vstar[1], legend = false, yticks = (1:10, 12:21), title = "V*")
	p4 = heatmap(vstar[2], yticks = (1:10, 12:21))
	plot(p1, p3, p2, p4, layout = (2,2))
end

# ╔═╡ 627d5aa3-974f-4949-8b65-9500eba1d7cc
#recreation of figure 5.2
plot_blackjack_policy(πstar_blackjack)

# ╔═╡ b040f245-b2d6-4ec6-aa7f-511c54aabd0d
#recreation of figure 5.2 using ϵ-soft method
plot_blackjack_policy(πstar_blackjack2)

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
blackjackepisode((13, 2, true), :hit, s -> sample(collect(blackjackactions), weights(π_blackjack1[s])))

# ╔═╡ 1d2e91d0-a1a0-4630-ac06-37684a9104f3
blackjackepisode((13, 2, true), :hit, s -> rand(blackjackactions))

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
#target policy state value estimate and variance, why is the mean squared error after 1 episode for weighted importance sampling less than the variance of the state values?  Also this value estimate does not match what it says in the book of -0.27726 so there might be something subtlely wrong with my simulator
estimate_blackjack_state(10_000_000, π_blackjack1)

# ╔═╡ 70d9d39f-020d-4f25-810c-82a143a3335b
const π_rand_blackjack = Dict(s => [0.5, 0.5] for s in blackjackstates)

# ╔═╡ 8faca500-b80d-4b50-88b6-683d18a1286b
#behavior policy state value estimate and variance
estimate_blackjack_state(10_000_000, π_rand_blackjack)

# ╔═╡ d6863551-a254-44b6-b6fe-551d134cdf01
v_offpol = monte_carlo_pred(π_blackjack1, Dict(s => [0.5, 0.5] for s in blackjackstates), blackjackstates, blackjackactions, blackjackepisode, 1.0, 1_000_000, gets0 = () -> (13, 2, true))

# ╔═╡ c5482c11-1635-4016-bf6a-4c5f01ae66b9
v_offpol[1][(13, 2, true)]

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
figure_5_4(7, 3)

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
q_offpol = monte_carlo_Q_pred(π_blackjack1, Dict(s => [0.5, 0.5] for s in blackjackstates), blackjackstates, blackjackactions, blackjackepisode, 1.0, 10_000_000, gets0 = () -> (13, 2, true))

# ╔═╡ 2ed0d4bb-bb6e-4adf-ad7c-8ddeb8c20b84
q_offpol[((13, 2, true), :hit)] #should converge to -0.27726 same as the value function for the policy that hits on this state

# ╔═╡ 3558aaea-2594-4042-99ea-c373ed304850
q_offpol[((13, 2, true), :stick)] #should be a lower value estimate because sticking is a worse action than hitting

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
(πstar_blackjack3, Qstar_blackjack3) = off_policy_MC_control(blackjackstates, blackjackactions, blackjackepisode, 1.0, 10_000_000)

# ╔═╡ 53487950-f5c4-4715-a4e4-1bf2fd91b213
#recreation of figure 5.2 using off-policy method
plot_blackjack_policy(πstar_blackjack3)

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
#using a random policy, the mean time to finish on track 1 is ~2800 steps.  The best possible time when we get "lucky" with random decisions is ~12 steps with worst times ~15-30k steps
sampleracepolicy(s -> rand(racetrack_actions), 10_000)

# ╔═╡ 5b1d5d03-b7cf-42e0-bd5b-f3b4fff12df2
const track1states = [(position = p, velocity = v) for p in union(track1.start, track1.body) for v in racetrack_velocities]

# ╔═╡ dfc2d648-ec08-49cd-a55f-72a766cad728
(πstar_racetrack1, Qstar_racetrack1) = off_policy_MC_control(track1states, racetrack_actions, (s, a, π) -> race_track_episode(s, a, π, track1), 1.0, 10_000)

# ╔═╡ 9524a3f5-fca3-4ce9-b04d-0cb6c0cd0c90
#off policy control doesn't produce a policy that can finish the race.  A cutoff of 100k steps is used to ensure the system doesn't run forever
runrace(s -> πstar_racetrack1[s])

# ╔═╡ 68f9e0d9-5c9d-4e89-b5ae-f24fd7544a09
(πstar_racetrack2, Qstar_racetrack2) = monte_carlo_ES(track1states, racetrack_actions, (s, a, π) -> race_track_episode(s, a, π, track1), 1.0, 100_000)

# ╔═╡ 3d10f89f-876e-4d25-b8d6-34ce5c99eb8c
#exploring starts on policy training also doesn't produce a policy that can finish the race
runrace(s -> πstar_racetrack2[s])

# ╔═╡ 1d6eccf0-2731-47fc-9a41-ea8649e290ef
(πstar_racetrack3, Qstar_racetrack3) = monte_carlo_ϵsoft(track1states, racetrack_actions, (s, a, π) -> race_track_episode(s, a, π, track1), 1.0, 0.25, 10_000_000, gets0 = () -> (position = rand(track1.start), velocity = (0, 0)))

# ╔═╡ 7212f887-5347-4ee1-90b8-43d282f0fa6e
runrace(s -> πstar_racetrack3[s])

# ╔═╡ 985f0537-2bbe-4dbb-a113-8ac98d2e0a5f
sampleracepolicy(s -> πstar_racetrack3[s], 10_000)

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
plotpolicy2(πstar_racetrack3)

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
	race_episode_star = race_track_episode(s0, a0, s -> π[s], track1, maxsteps = 10000, failchance = 0.0)
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
visualize_policy_traj(πstar_racetrack1)

# ╔═╡ 8645934b-a141-41e4-ae5b-6a0a86a94156
visualize_policy_traj(πstar_racetrack2)

# ╔═╡ 2a13fdd4-97b5-4b65-8cb5-a0124ecf3dac
visualize_policy_traj(πstar_racetrack3)

# ╔═╡ 85205698-76a3-4901-8caf-e6e6fc5524ee
#trajectory of a successful race policy, black arrows indicate velocity, green arrows indicate action.  Note that negative velocities are forbidden so any arrow pointing left on a vertical trajectory will have no impact.
visualize_policy_traj2(πstar_racetrack3)

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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
BenchmarkTools = "~1.3.1"
Plots = "~1.28.1"
StatsBase = "~0.33.16"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0"
manifest_format = "2.0"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "af237c08bda486b74318c8070adb96efa6952530"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "cd6efcf9dc746b06709df14e462f0a3fe0786b1e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.2+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "46a39b9c58749eefb5f2dc1178cb8fab5332b1ab"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.15"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "76c987446e8d555677f064aaac1145c4c17662f8"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.14"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "d05baca9ec540de3d8b12ef660c7353aae9f9477"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.28.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "cd56bf18ed715e8b09f06ef8c6b781e6cdc49911"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c82aaa13b44ea00134f8c9c89819477bd3986ecd"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.3.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "8f705dd141733d79aa2932143af6c6e0b6cea8df"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─826139cc-b52e-11ec-0d47-25ab689851fd
# ╠═c044c399-1f07-4312-b18b-4ba2a91d1d71
# ╠═760c5361-02d4-46b7-a05c-fc2d10d93de6
# ╟─6a11daf7-2859-41fa-9c3d-d1f3580dbb5f
# ╠═72e29874-f7cb-4089-bcc4-8e45336cdc23
# ╠═fa0fbe9a-c826-4a9a-b3a2-0af483114055
# ╠═1049a6e6-7446-49ba-97d5-43b02ace5e39
# ╠═278ff1d0-9983-4e6d-8a67-2bbe584415bf
# ╠═887c9749-d23d-46cd-9386-8473e3d603a9
# ╠═b5c96cca-7f05-45a3-8641-535b41969540
# ╠═1df4cae0-71a9-4598-b941-58a659ba1381
# ╠═0e3217d8-1a5e-4387-b4df-3121355d3464
# ╠═2076b266-ec17-4492-814d-8bf8942221ae
# ╠═8f5d1487-0070-422d-a1f8-1b14a312db6c
# ╠═94be5289-bba7-4490-bdcd-0d217a31c665
# ╠═bd63d4b2-423d-4860-8dd3-3587a124ced5
# ╠═a68b4965-c392-47e5-9b29-93e7ada9990a
# ╠═82284c63-2306-4469-9b1a-a5ec87037e79
# ╟─e5384dd0-fad1-4a24-b011-73b062fcfb1b
# ╟─30809344-b4ab-468b-b4b7-5ef3dca5ffc7
# ╟─f406be9e-3e3f-4b55-99b0-4858c774ed96
# ╟─be7c096c-cc8c-407b-8287-8fb2ee7150a7
# ╟─47daf83b-8fe9-4491-b9ae-84bd269d5546
# ╠═13cc524c-d983-44f4-8731-0595249fb888
# ╟─9618a093-cdb7-4589-a783-de8e9021b705
# ╠═ec29865f-3ba3-4bb3-84df-c2b472e03ff2
# ╟─5c57e4b7-51d6-492d-9fc9-bcdab1dd46f4
# ╠═627d5aa3-974f-4949-8b65-9500eba1d7cc
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
# ╠═c54dba6b-35bc-4537-928c-f25bff2a4a18
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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
