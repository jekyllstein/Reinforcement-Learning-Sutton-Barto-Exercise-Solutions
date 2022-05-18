### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 9b9ee4f2-f5f9-444b-aa23-85f145d8f9ca
using Plots

# ╔═╡ 659cd7f9-38c7-4dde-818e-be5e26bed09f
using Statistics

# ╔═╡ 705fc9b0-6372-4c5f-8696-78c7cfaa3a76
using Random, StatsBase

# ╔═╡ 814d89be-cfdf-11ec-3295-49a8f302bbcf
md"""
# Chapter 6 Temporal-Difference Learning
## 6.1 TD Prediction
"""

# ╔═╡ 495f5606-0567-47ad-a266-d21320eecfc6
md"""
Monte Carlo nonstationary update rule for value function

$V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$

where $G_t$ is the actual return following time $t$, and $\alpha$ is a constant step-size parameter.  Call this method *constant-α MC*.

In contrast, the following is the simplest TD method update rule

$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$

where the update can be made immediately on transition to $S_{t+1}$ after receiving $R_{t+1}$.  This TD method is called $TD(0)$, or *one-step TD*.  See below for code implementing this.
"""

# ╔═╡ c3c43440-d8f5-4d16-8735-8d83981b9f15
function tabular_TD0_value_est(π::Function, α, γ, states, sterm, actions, tr::Function, n = 1000; gets0 = () -> rand(states), v0 = 0.0)
	V = Dict(s => v0 for s in states)
	V[sterm] = 0.0
	for i in 1:n
		s0 = gets0()
		a0 = π(s0)
		function run_episode!(s0, a0)
			(s, r, isterm) = tr(s0, a0)
			V[s0] += α*(r + γ*V[s] - V[s0])
			isterm && return nothing
			a = π(s)
			run_episode!(s, a)
		end
		run_episode!(s0, a0)
	end
	return V
end		

# ╔═╡ 7cd8e898-05c8-4fa0-b12a-60c5c1110cf8
md"""
> *Exercise 6.1* If $V$ changes during the episode, then (6.6) only holds approximately; what would the difference be between the two sides?  Let $V_t$ denote the array of state values used at time $t$ in the TD error (6.5) and in the TD update (6.2).  Redo the derivation above to determine the additional amount that must be added to the sum of TD errors in order to equal the Monte Carlo error.

Re-write equation (6.5) using the values known at time t

$\delta_t \dot= R_{t+1} + \gamma V_t(S_{t+1}) - V_t(S_t)$

Now equation (6.2) becomes

$V_{t+1}(S_t) = V_t(S_t) + \alpha \delta_t$

and $V$ is updated throughout the episode.  So if we extend this to later points in the episode we have

$V_{t+2}(S_{t+1}) = V_{t+1}(S_{t+1}) + \alpha \delta_{t+1}$

$\delta_{t+1} \dot= R_{t+2} + \gamma V_{t+1}(S_{t+2}) - V_{t+1}(S_{t+1})$

Starting the derivation again using (3.9)

$G_t - V_t(S_t) = R_{t+1} + \gamma G_{t+1} - V_t(S_t) + \gamma V_{t}(S_{t+1}) - \gamma V_{t}(S_{t+1})$

$\cdots = \delta_t + \gamma \left [ G_{t+1} - V_t(S_{t+1}) \right ]$
$\cdots = \delta_t + \gamma \left [ G_{t+1} - V_t(S_{t+1}) + V_{t+1}(S_{t+1}) - V_{t+1}(S_{t+1}) \right ]$
$\cdots = \delta_t + \gamma \left [ G_{t+1} - V_{t+1}(S_{t+1}) + V_{t+1}(S_{t+1}) - V_t(S_{t+1}) \right ]$

Define the following

$\eta _t \dot = V_{t+1}(S_{t+1}) - V_t(S_{t+1})$ 

which let's us re-write the equation

$G_t - V_t(S_t) = \delta_t + \gamma \eta_t + \gamma \left [ G_{t+1} - V_{t+1}(S_{t+1})\right ]$

This can now be expanded recursively

$\cdots = \delta_t + \gamma \eta_t + \gamma \left [\delta_{t+1} + \gamma \eta_{t+1} +  \gamma (G_{t+2} - V_{t+2}(S_{t+2}) ) \right ]$

$\cdots = \delta_t + \gamma \eta_t + \gamma \delta_{t+1}  + \gamma^2 \eta_{t+1} + \gamma^2 \left [G_{t+2} - V_{t+2}(S_{t+2}) \right ]$

$= \delta_t + \gamma \delta_{t+1} + \cdots + \gamma^{T-t-1}\delta_{T-1} + \gamma \eta_t + \gamma^2 \eta_{t+1} + \cdots + \gamma^{T-t} \eta_{T-t} + \gamma^{T-t} \left [G_T - V_T(S_T) \right ]$

$= \delta_t + \gamma \delta_{t+1} + \cdots + \gamma^{T-t-1}\delta_{T-1} + \gamma \eta_t + \gamma^2 \eta_{t+1} + \cdots + \gamma^{T-t} \eta_{T-t} + \gamma^{T-t} \left [0 - 0 \right ]$

$= \sum_{k=t}^{T-1} \gamma^{k-t} (\delta_k + \gamma \eta_k)$
"""

# ╔═╡ 8b2a09f5-37c5-4b91-af5c-4596c44b96ea
md"""
$V_{t+1}(S_t) = V_t(S_t) + \alpha \delta_t$
$\eta _t \dot = V_{t+1}(S_{t+1}) - V_t(S_{t+1})$ 

if $S_t = S_{t+1}$

$\eta _t \dot = (V_t(S_t) + \alpha \delta_t) - V_t(S_{t}) = \alpha \delta_t$ 


"""

# ╔═╡ b5187232-d808-49b6-9f7e-a4cbeb6c2b3e
md"""
### Example 6.1: Driving Home
"""

# ╔═╡ 7b3e55f4-72b8-48a5-a62a-7ce7ffadae35
plotly()

# ╔═╡ bc8bad61-a49a-47d6-8fa6-7dcf6c221910
function example_6_1()
	states = [:leaving, :reach_car, :exit_highway, :snd_rd, :home_st, :arrive]
	elapsed = [0, 5, 20, 30, 40, 43]
	predicted_ttg = [30, 35, 15, 10, 3, 0]
	predicted_tt = predicted_ttg .+ elapsed

	plot(predicted_tt, xticks = (1:6, String.(states)), ylabel = "Minutes", lab = "Preicted Outcome", size = (680, 400))
	plot!(fill(43, 6), line = :dot, lab = "actual outcome")
end

# ╔═╡ 6edb550d-5c9f-4ea6-8746-6632806df11e
example_6_1()

# ╔═╡ 9017093c-a9c3-40ea-a9c6-881ee62fc379
md"""
> *Exercise 6.2* This is an exercise to help develop your intuition about why TD methods are often more efficient than Monte Carlo methods.  Consider the driving home example and how it is addressed by TD and Monte Carlo methods.  Can you imagine a scenario in which a TD update would be better on average than a Monte Carlo update?  Give an example scenario - a description of past experiene and a current state - in which you would expect the TD update to be better.  Here's a hint: Suppose you have lots of experience driving home from work.  Then you move to a new building and a new parking lot (but you still enter the highway at the same place).  Now you are starting to learn predictions for the new building.  Can you see why TD updates are likely to be much better, at least initially, in this case?  Might the same sort of thing happen in the original scenario?

Originally, from the starting state, the expected total time to reach home is 30 minutes.  Now if we change the route so that it now takes on average 5 more minutes to reach the car and another 5 more minutes to exit the highway, the total time should really be 40 minutes.  However, all of the estimates after exiting the highway will still be accurate.  Using a Monte Carlo method, we would wait until the end of the journey and then update the first two predictions, increasing them by 10 and 5 minutes respectively (if α=1).  Using a TD method, once we get our updated prediction from state 2, we would update our starting prediction by increasing it 5 minutes and the update for the second state by 5 minutes upon exiting the highway.  So halfway through our journey, we would have a accurate estimate from reaching the car and an improved estimate from leaving the office.  Also, suppose that unusual events occur after exiting the highway that change the outcome from the mean value.  Using the Monte Carlo updates, those fluctuations would impact the estimate for the first two states as well.  With the TD method, only the observed longer time arriving at states 2 and 3 would change the first 2 estimates.  Assuming we already have good estimates for the following states, we would never want to update our new 2 starting state estimates using the later data, and with TD learning we only use the observed differences arriving at states 2 and 3 which are the states that have been affected by the new starting location.  
"""

# ╔═╡ 5290ae65-6f56-4849-a842-fe347315c6dc
md"""
## 6.2 Advantages of TD Prediction Methods
"""

# ╔═╡ 47c2cbdd-f6db-4ce5-bae2-8141f30aacbc
md"""
### Example 6.2 Random Walk
"""

# ╔═╡ d501f74b-c3a3-4591-b01f-5df5b71a85f2
begin
	abstract type MRP_State end
	struct A <: MRP_State end
	struct B <: MRP_State end
	struct C <: MRP_State end
	struct D <: MRP_State end
	struct E <: MRP_State end
	struct Term <: MRP_State end
end

# ╔═╡ ae182e8c-6fc0-4043-ac9f-cb6726d173e7
function tabular_MC_value_est(π::Function, α, γ, states, sterm, actions, tr::Function, n = 1000; gets0 = () -> rand(states), v0 = 0.0)
	V = Dict(s => v0 for s in states)
	V[sterm] = 0.0
	for i in 1:n
		s0 = gets0()
		a0 = π(s0)
		traj = Vector{Tuple{T, Int64} where T <: MRP_State}()
		push!(traj, (s0, a0))
		rewards = Vector{Float64}()
		function run_episode!(s0, a0)
			(s, r, isterm) = tr(s0, a0)
			push!(rewards, r)
			isterm && return nothing
			a = π(s)
			push!(traj, (s, a))
			run_episode!(s, a)
		end
		run_episode!(s0, a0)

		function updateV!(t = length(traj), g = 0.0)
			t == 0 && return nothing
			(s, a) = traj[t]
			r = rewards[t]
			g = γ*g + r
			V[s] += α * (g - V[s])
			updateV!(t - 1, g)
		end
		updateV!()
	end
	return V
end		

# ╔═╡ a116f557-ca8f-4e28-bf8c-84e7e19b30da
function random_walk_6_2()
	left(::A) = (Term(), 0, true)
	left(::B) = (A(), 0, false)
	left(::C) = (B(), 0, false)
	left(::D) = (C(), 0, false)
	left(::E) = (D(), 0, false)

	right(::A) = (B(), 0, false)
	right(::B) = (C(), 0, false)
	right(::C) = (D(), 0, false)
	right(::D) = (E(), 0, false)
	right(::E) = (Term(), 1, true)

	tr(s0 , a0) = rand() < 0.5 ? left(s0) : right(s0)

	states = [A(), B(), C(), D(), E(), Term()]
	actions = [1]
	(states = states, actions = actions, tr = tr)
end

# ╔═╡ 2786101e-d365-4d6a-8de7-b9794499efb4
function example_6_2()
	(states, actions, tr) = random_walk_6_2()
	π(s) = 1
	nlist = [1, 10, 100]

	TD0_est(α, n) = tabular_TD0_value_est(π, α, 1.0, states, Term(), actions, tr, n, gets0 = () -> C(), v0 = 0.5)

	MC_est(α, n) = tabular_MC_value_est(π, α, 1.0, states, Term(), actions, tr, n, gets0 = () -> C(), v0 = 0.5)

	V_ests = [TD0_est(0.1, n) for n in nlist]

	true_values = collect(1:5) ./ 6

	x1 = ["A", "B", "C", "D", "E"]
	y1 = [[V_ests[i][s] for s in states[1:end-1]] for i in 1:3]
	p1 = plot(vcat([true_values], y1), xticks = (1:5, x1), lab = hcat("True values", ["$n ep est" for n in nlist']), xlabel="State")

	MC_est(0.1, 1)

	function rms_err(Vest)
		sqrt(mean(([Vest[s] for s in states[1:end-1]] .- true_values) .^2))
	end
	
	samples = 100
	rms_TD0(n, α) = mean(rms_err(TD0_est(α, n)) for _ in 1:samples)
	rms_MC(n, α) = mean(rms_err(MC_est(α, n)) for _ in 1:samples)
	
	maxepisodes = 100
	
	αlist1 = [0.05, 0.1, 0.15, 1.0]
	αlist2 = [0.01, 0.02, 0.03, 0.04]
	y2 = [[rms_TD0(n, α) for n in 1:maxepisodes] for α in αlist1]
	y3 = [[rms_MC(n, α) for n in 1:maxepisodes] for α in αlist2]
	p2 = plot(y2, lab = ["TD α = $α" for α in αlist1'], xlabel = "Episodes", title = "Empirical RMS error, averaged over states")
	plot!(y3, lab = ["MC α = $α" for α in αlist2'])

	plot(p1, p2, layout = (2, 1), size = (680, 700))
end		

# ╔═╡ 9db7a268-1e6d-4366-a0ec-ebf54916d3b0
example_6_2()

# ╔═╡ b5d0def2-9b65-4e28-a910-d261a25e31f1
sqrt(sum([(n/6)^2 for n in 1:5]))

# ╔═╡ 0b9c6dbd-4eb3-4167-886e-64db9ec7ff04
md"""
> *Exercise 6.3* From the results shown in the left graph of the random walk example it appears that the first episode reuslts in a change only in $V(A)$.  What does this tell you about what happened on the first episode?  Why was only the estimate for this one state changed?  By exactly how much was it changed?

The update rule with TD0 learning is given by 

$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$

All states, A, B, C, D, E are initialized at 0.5 with the terminal state initialized at 0.  During the first episode for all transitions before the end, the reward is 0 and the difference between adjacent states would be 0 resulting in no change to the value function.  Since the value estimate for state A decreases from the initial value, this teams that the first episode terminated to the far left.  For this final transition we have the following update.

$V(A) \leftarrow V(A) + \alpha[0 + \gamma V(\text{Term}) - V(A)]$

We know that prior to the update $V(A) = 0.5$, $V(\text{Term}) = 0$ and $\gamma=1$ so the update is

$V(A) \leftarrow 0.5 + \alpha[0 - 0.5]$

For this plot, $\alpha=0.1$, so the updated value for $V(A)$ is $0.5+0.1(-0.5)=0.5-0.05=0.45$
"""

# ╔═╡ 52aebb7b-c2a9-443f-bc03-24cd25793b32
md"""
> *Exercise 6.4* The specific results shown in the right graph of the random walk example are dependent on the value of the step-size parameter α. Do you think the conclusions about which algorithm is better would be affected if a wider range of values were used? Is there a different, fixed value of α at which either algorithm would have performed significantly better than shown? Why or why not?

Both algorithms should theoretically converge to the true values with a sufficiently small α and a large enough number of samples.  Over this limited window of 100 episodes, an α that is too small might result in convergence so slow that it does not reach error as low as a larger α.  For the MC method, the range of α's already show the case of too slow convergence and too large α with the best outcome at α=0.02.  For the TD method, the best results shown are for α=0.05, so a smaller α might result in even better performance OR it could result in a curve that converges too slowly to beat α=0.05 in 100 episodes.  Either way, there is no alternative combination of step sizes for either method that would result in TD learning appearing worse than MC learning over this number of episodes.
"""

# ╔═╡ e6672866-c0a0-46f2-bb52-25fcc3352645
md"""
> *Exercise 6.5* In the right graph of the random walk example, the RMS error of the TD method seems to go down and then up again, particularly at high α’s. What could have caused this? Do you think this always occurs, or might it be a function of how the approximate value function was initialized?


"""

# ╔═╡ 105c5c23-270d-437e-89dd-12297814c6e0
md"""
> *Exercise 6.6* In Example 6.2 we stated that the true values for the random walk example are 1/6 , 2/6 , 3/6 , 4/6 , and 5/6 , for states A through E. Describe at least two different ways that these could have been computed. Which would you guess we actually used? Why?

###### Method 1: Set up the following system of equations
$V(A) = \frac{0+V(B)}{2} \implies 2V(A)=V(B)$
$V(B) = \frac{V(A)+V(C)}{2} \implies 2V(B) = V(A)+V(C)$
$V(C) = \frac{V(B)+V(D)}{2} \implies 2V(C)=V(B)+V(D)$
$V(D) = \frac{V(C)+V(E)}{2} \implies 2V(D)=V(C)+V(E)$
$V(E) = \frac{V(D)+1}{2} \implies 2V(E)=V(D)+1$

We can work down from the top equation expressing everything in terms of A.  For shorter expressions $V(A)$ will be written below as $A$:

$B=2A$

$2B=A+C \implies C = 3A$

$2C=B+D \implies D = 6A-2A=4A$

$2D=C+E \implies E = 8A-3A = 5A$

$2E = D + 1 \implies 10A = 4A + 1 \implies A = \frac{1}{6}$

Now that we have the value for A, all the others are trivial multiplications of it from 2 to 5.

###### Method 2: Calculate each value from probabilities
With this method to get $V(A)$ we would write down every possible trajectory to a terminal state with the associated probability of each.  Since trajectories terminating to the left have a value of 0, we only need to add up the trajectories that terminate to the right.  Below are some examples for state A.

$V(A) = 0.5^5 + 4 \times 0.5^7 + \cdots$

This equation represents the single trajectory that takes 5 steps to the right each with probability one half and the 4 possible trajectories that turn around once on the way right resulting in 7 steps.  This sum will end up being infintely long to account for all of the trajectories that bounce back and forth arbitrarily large amounts of time.  This method is significantly harder to calculate for each state compared to the first method and is more in line with how estimates are calculated with MC sampling.  The first method is more analogous to TD sampling using the bootstrapped form of the Bellman equation.
"""

# ╔═╡ 48b557e3-e239-45e9-ab15-105bcca96492
md"""
## 6.3 Optimality of TD(0)

### Example 6.3: Random walk under batch updating

"""

# ╔═╡ 620a6426-cb29-4010-997b-aa4f9d5f8fb0
begin
	abstract type BatchMethod end
	struct TD0 <: BatchMethod end
	struct MC <: BatchMethod end
end

# ╔═╡ 6f185046-dfdb-41ca-bf3f-e2f90e2e4bc0
function batch_value_est(π::Function, α, γ, states::Vector{S}, sterm, actions::Vector{A}, tr::Function, n = 1000; gets0 = () -> rand(states), v0 = 0.0, ϵ = 0.01, Vref = Dict(s => v0 for s in states), estmethod = TD0()) where {S, A}
	V = Dict(s => v0 for s in states)
	V[sterm] = 0.0
	errs = Vector{Float64}()

	#store all episodes
	episodes = Vector{Vector{Tuple{S, A, S, Float64}}}()

	#add a new episode to the list
	function add_episode!()
		s0 = gets0()
		a0 = π(s0)
		traj = Vector{Tuple{S, A, S, Float64}}()
		function run_episode!(s0, a0)
			(s, r, isterm) = tr(s0, a0)
			push!(traj, (s0, a0, s, r))
			isterm && return nothing
			a = π(s)
			run_episode!(s, a)
		end
		run_episode!(s0, a0)
		push!(episodes, traj)
	end

	#use existing episode list to update value estimates
	function update_value!(::TD0, episodes)
		maxpctdelt = 0.0
		for traj in episodes
			for (s0, a0, s, r) in traj
				v0 = V[s0]
				delt = α*(r + γ*V[s] - V[s0])
				V[s0] += delt
				pctdelt = if v0 == 0
					if delt == 0
						0.0
					else
						Inf
					end
				else
					abs(delt/v0)
				end
				maxpctdelt = max(maxpctdelt, pctdelt)
			end
		end
		return maxpctdelt
	end
	
	function update_value!(::MC, episodes)
		maxpctdelt = 0.0
		for traj in episodes
			g = 0.0
			for (s0, a0, s, r) in reverse(traj)
				g = γ*g + r
				v0 = V[s0]
				delt = α*(g - V[s0])
				V[s0] += delt
				pctdelt = if v0 == 0
					if delt == 0
						0.0
					else
						Inf
					end
				else
					abs(delt/v0)
				end
				maxpctdelt = max(maxpctdelt, pctdelt)
			end
		end
		return maxpctdelt
	end

	#use existing episode list to update value estimates until convergence
	function update_value!()
		i = 1
		maxpctdelt = Inf
		while (maxpctdelt > ϵ) && (i < 100)
			maxpctdelt = update_value!(estmethod, episodes)
			i += 1
		end
		return maxpctdelt
	end

	function rms_err()
		mean(sqrt.(([V[s] for s in states[1:end-1]] .- [Vref[s] for s in states[1:end-1]]) .^2))
	end
	
	for i in 1:n
		add_episode!()
		pctdelt = update_value!()
		# println("On episode $i, value function converged with a maximum percent change of $pctdelt")
		push!(errs, rms_err())
	end
	return V, errs
end	

# ╔═╡ 0ad7b475-6394-4780-908e-849c0684a966
function example_6_3()
	(states, actions, tr) = random_walk_6_2()
	π(s) = 1

	true_values = collect(1:5) ./ 6
	Vref = Dict(zip(states[1:end-1], true_values))

	TD0_est(α, n) = batch_value_est(π, α, 1.0, states, Term(), actions, tr, n, gets0 = () -> C(), v0 = 0.5, Vref = Vref, estmethod = TD0())

	MC_est(α, n) = batch_value_est(π, α, 1.0, states, Term(), actions, tr, n, gets0 = () -> C(), v0 = 0.5, Vref = Vref, estmethod = MC())

	# x1 = ["A", "B", "C", "D", "E"]
	# y1 = [[V_ests[i][s] for s in states[1:end-1]] for i in 1:3]
	# p1 = plot(vcat([true_values], y1), xticks = (1:5, x1), lab = hcat("True values", ["$n ep est" for n in nlist']), xlabel="State")

	# MC_est(0.1, 1)

	# function rms_err(Vest)
	# 	sqrt(mean(([Vest[s] for s in states[1:end-1]] .- true_values) .^2))
	# end
	
	samples = 100
	rms_TD0(n, α) = mean(reduce(hcat, (TD0_est(α, n)[2] for _ in 1:samples)), dims = 2)
	rms_MC(n, α) =mean(reduce(hcat, (MC_est(α, n)[2] for _ in 1:samples)), dims = 2)
	
	maxepisodes = 25
	
	y2 = rms_TD0(maxepisodes, 0.05)
	y3 = rms_MC(maxepisodes, 0.02)
	p2 = plot(y2, xlabel = "Episodes", title = "Empirical RMS error, averaged over states", lab = "TD0")
	plot!(y3, lab = "MC")

	# plot(p1, p2, layout = (2, 1), size = (680, 700))
end		

# ╔═╡ 22c2213e-5b9b-410f-a0ef-8f1e3db3c532
example_6_3()

# ╔═╡ 0e59e813-3d48-4a24-b5b3-9a9de7c500c2
md"""
> *Exercise 6.7* Design an off-policy version of the TD(0) update that can be used with arbitrary target policy $\pi$ and convering behavior policy $b$, using each step $t$ the importance sampling ratio $\rho_{t:t}$ (5.3).

Recall that equation 5.3 defines:

$\rho_{t:T-1} = \prod_{k=t}^{T-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}$

The TD(0) update rule is given by:

$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$

based on the following form of the Bellman equation:

$v_\pi=\text{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s]$

In the off-policy case, the only data we have to work with is from the behavior policy, yet we still want the expectation over the target policy.  However, the value estimates being updated will already represent the target policy, so the only random sampling value that needs to be adjusted is the reward signal.  We can therefore use the importance sampling ratio on the reward and ensure that our expected value is correct.

$V(S_t) \leftarrow V(S_t) + \alpha [\rho_{t:t}R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$
"""

# ╔═╡ 0d6a11af-b146-4bbc-997e-a11b897269a7
md"""
## 6.4 Sarsa: On-policy TD Control

TD(0) update rule for action values:

$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})-Q(S_t, A_t)]$
"""

# ╔═╡ 1ae30f5d-b25b-4dcb-800f-45c463641ec5
md"""
> *Exercise 6.8* Show that an action-value version of (6.6) holds for the action-value form of the TD error $\delta_t=R_{t+1}+\gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$, again assuming that the values don't change from step to step.

The derivation in (6.6) starts with the definition in (3.9):

$G_t = R_{t+1} + \gamma G_{t+1}$

and derives the following:

$\delta_t \dot = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$
$G_t - V(S_t) = \sum_{k=t}^{T-1} \gamma^{k-t} \delta_k$

Now we have the action-value form of the TD error:

$\delta_t=R_{t+1}+\gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$

Let us transform (3.9) in a similar manner to derive the rule:

$G_t - Q(S_t, A_t) = R_{t+1} + \gamma G_{t+1} - Q(S_t, A_t) + \gamma Q(S_{t+1}, A_{t+1}) - \gamma Q(S_{t+1}, A_{t+1})$
$= \delta_t + \gamma (G_{t+1} - Q(S_{t+1}, A_{t+1}))$
$= \delta_t + \gamma \delta_{t+1} + \gamma^2 (G_{t+2} - Q(S_{t+2}, A_{t+2}))$
$= \delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+1} + \cdots + \gamma^{T-t-1} \delta_{T-1} + \gamma^{T-t}(G_T - Q(S_T, A_T))$

The action value is defined to be 0 whenever the state is terminal

$= \delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+1} + \cdots + \gamma^{T-t-1} \delta_{T-1} + \gamma^{T-t}(0-0)$
$=\sum_{k=t}^{T-1}\gamma^{k-t}\delta_k$
"""

# ╔═╡ 61bbf9db-49a0-4709-83f4-44f228be09c0
function sarsa_onpolicy(α, γ, states, sterm, actions, tr::Function, n = 1000; gets0 = () -> rand(states), q0 = 0.0, ϵ = 0.1)
	Q = Dict((s, a) => q0 for s in states for a in actions)
	nact = length(actions)
	π = Dict(s => fill(1.0 / nact, nact) for s in states) #create policy sampling
	sampleπ(s) = sample(actions, weights(π[s]))
	otherv = ϵ / nact #probability weight given to random actions
	topv = 1.0 - ϵ + otherv #probability weight given to the top action in the distribution
	
	for a in actions
		Q[(sterm, a)] = 0.0
	end
	steps = zeros(Int64, n)
	rewardsums = zeros(n)
	#update policy with epsilon greedy strategy following Q
	function updateπ!()
		for s in states
			aind = argmax(Q[(s, a)] for a in actions) #index of selected action
			π[s] .= otherv
			π[s][aind] = topv
		end
	end

	for i in 1:n
		updateπ!()
		s0 = gets0()
		a0 = sampleπ(s0)

		function updateq!(s0, a0, l = 1; rsum = 0.0)
			(s, r, isterm) = tr(s0, a0)
			rsum += r
			a = sampleπ(s)
			Q[(s0, a0)] += α*(r + γ*Q[(s, a)] - Q[(s0, a0)])
			updateπ!()
			isterm && return (l, rsum)
			updateq!(s, a, l+1, rsum = rsum)
		end
		l, rsum = updateq!(s0, a0)
		steps[i] = l
		rewardsums[i] = rsum
	end

	π_det = Dict(s => actions[argmax(π[s])] for s in states)
	return Q, π_det, steps, rewardsums
end

# ╔═╡ e19db54c-4b3c-42d1-b016-9620daf89bfb
begin
	abstract type GridworldAction end
	struct Up <: GridworldAction end
	struct Down <: GridworldAction end
	struct Left <: GridworldAction end
	struct Right <: GridworldAction end

	wind_actions1 = [Up(), Down(), Left(), Right()]
	
	move(::Up, x, y) = (x, y+1)
	move(::Down, x, y) = (x, y-1)
	move(::Left, x, y) = (x-1, y)
	move(::Right, x, y) = (x+1, y)

	applywind(w, x, y) = (x, y+w)
end

# ╔═╡ 56f794c3-8e37-48b4-b953-7ad0a45aadd6
function gridworld_sarsa_solve(gridworld; α=0.5, ϵ=0.1)
	states, sterm, actions, tr, episode, gets0 = gridworld
	tr(gets0(), Up())
	(Qstar, πstar, steps, rsum) = sarsa_onpolicy(α, 1.0, states, sterm, actions, tr, 250, gets0 = gets0, q0 = 0.0, ϵ = ϵ)
	path = episode(s -> πstar[s])
	p1 = plot(path, legend = false, title = "Finished in $(length(path)-1) steps")
	p2 = plot(steps, legend = false, xlabel = "episodes", ylabel = "Steps")
	p3 = plot([minimum(steps[1:i]) for i in eachindex(steps)], yaxis = :log, legend = false, ylabel = "Min Path So Far")
	p4 = plot(cumsum(steps), 1:length(steps), legend = false)
	l = @layout [
    a{0.4w} [grid(3,1)]
]
	plot(p1, p2, p3, p4, layout = l, size = (680, 400))
end

# ╔═╡ 0ad739c9-8aca-4b82-bf20-c73584d29535
md"""
> *Exercise 6.9 Windy Gridworld with King's Moves (programming)* Re-solve the windy gridworld assuming eight possible actions, including the diagonal moves, rather than four.  How much better can you do with the extra actions?  Can you do even better by including a ninth action that causes no movement at all other than that caused by the wind?
"""

# ╔═╡ 031e1106-7408-4c7e-b78e-b713c19123d1
begin
	struct UpRight <: GridworldAction end
	struct DownRight <: GridworldAction end
	struct UpLeft <: GridworldAction end
	struct DownLeft <: GridworldAction end

	wind_actions2 = [UpRight(), UpLeft(), DownRight(), DownLeft()]
	
	move(::UpRight, x, y) = (x+1, y+1)
	move(::UpLeft, x, y) = (x-1, y+1)
	move(::DownRight, x, y) = (x+1, y-1)
	move(::DownLeft, x, y) = (x-1, y-1)
end

# ╔═╡ 39470c74-e554-4f6c-919d-97bec1eec0f3
md"""
Adding king's move actions, the optimal policy can finish in 7 steps vs 15 for the original actions.  What happens after adding a 9th action that causes no movement?
"""

# ╔═╡ e9359ca3-4d11-4365-bc6e-7babc6fcc7de
begin
	struct Stay <: GridworldAction end
	move(::Stay, x, y) = (x, y)
	wind_actions3 = [Stay()]
end

# ╔═╡ ec285c96-4a75-4af6-8898-ec3176fa34c6
function windy_gridworld(actions, applywind = applywind)
	xmax = 10
	ymax = 7
	states = [(x, y) for x in 1:xmax for y in 1:ymax]
	sterm = (8, 4)
	gets0 = () -> (1, 4)

	#wind values at each x value
	winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

	# applywind(w, x, y) = (x, y+w)

	boundstate(x::Int64, y::Int64) = (clamp(x, 1, xmax), clamp(y, 1, ymax))
	
	function step(s::Tuple{Int64, Int64}, a::GridworldAction)
		w = winds[s[1]]
		(x, y) = s
		(x1, y1) = move(a, x, y)
		(x2, y2) = applywind(w, x1, y1)
		boundstate(x2, y2)
	end

	function tr(s0::Tuple{Int64, Int64}, a0::GridworldAction)
		snew = step(s0, a0)
		r = -1
		isterm = (snew == sterm)
		(snew, r, isterm)
	end

	function episode(π, lmax = 1000)
		s = gets0()
		path = [s]
		a = π(s)
		isterm = false
		l = 1
		while !isterm && (l < lmax)
			(s, r, isterm) = tr(s, a)
			push!(path, s)
			a = π(s)
			l += 1
		end
		return path
	end
			
	return states, sterm, actions, tr, episode, gets0
end	

# ╔═╡ 331d0b67-c00d-46fd-a175-b8412f6a93c5
gridworld_sarsa_solve(windy_gridworld(wind_actions1))

# ╔═╡ 1abefc8c-5be0-42b4-892e-14c0c47c16f0
gridworld_sarsa_solve(windy_gridworld(vcat(wind_actions1, wind_actions2)))

# ╔═╡ dee6b500-0ba1-4bbc-b217-cbb9ad47ad06
gridworld_sarsa_solve(windy_gridworld(vcat(wind_actions1, wind_actions2, wind_actions3)))

# ╔═╡ db31579e-3e56-4271-8fc3-eb13bc95ac27
md"""
Adding the no-movement action doesn't seem to change the shortest path of 7 steps
"""

# ╔═╡ b59eacf8-7f78-4015-bf2c-66f89bf0e24e
md"""
> *Exercise 6.10: Stochastic Wind (programming)* Re-solve the windy gridworld task with King's moves, assuming the effect of the wind, if there is any, is stochastic, sometimes varying by 1 from the mean values given for each column.  That is, a third of the time you move exactly according to these values, as in the previous exercise, but also a third of the time you move one cell above that, and another third of the time you move one cell below that.  For example, if you are one cell to the right of the goal and you move left, then one-third of the time you move one cell above the goal, one-third of the time you move two cells above the goal, and one-third of the time you move to the goal.
"""

# ╔═╡ aa0791a5-8cf1-499b-9900-4d0c59be808c
function stochastic_wind(w, x, y)
	w == 0 && return (x, y)
	
	v = rand([-1, 0, 1])
	(x, y+w+v)
end

# ╔═╡ ced61b99-9073-4dee-afbf-82531e59c7d8
gridworld_sarsa_solve(windy_gridworld(vcat(wind_actions1, wind_actions2), stochastic_wind))

# ╔═╡ 44c49006-e210-4f97-916e-fe62f36c593f
md"""
## 6.5 Q-learning: Off-policy TD Control

One of the early breakthroughs in reinforcement learning was the development of an off-policy TD control algorithm known as *Q-learning* (Watkins, 1989), defined by

$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \text{max}_a Q(S_{t+1}, a) - Q(S_t, A_t)]$
"""

# ╔═╡ f90263de-1053-48cb-8240-56112d6dc67f
function Q_learning(α, γ, states, sterm, actions, tr::Function, n = 1000; gets0 = () -> rand(states), q0 = 0.0, ϵ = 0.1)
	#initialize Q(s,a) and set Q(terminal, a) = 0 for all actions
	Q = Dict((s, a) => q0 for s in states for a in actions)
	for a in actions
		Q[(sterm, a)] = 0.0
	end
	
	nact = length(actions)
	π = Dict(s => fill(1.0 / nact, nact) for s in states) #create policy sampling
	sampleπ(s) = sample(actions, weights(π[s]))
	otherv = ϵ / nact #probability given to a random action
	topv = 1.0 - ϵ + otherv #probability weight given to the top action in the distribution
	
	steps = zeros(Int64, n) #keep track of length of each episode
	rewardsums = zeros(n)
	
	#update policy with epsilon greedy strategy following Q
	function updateπ!()
		for s in states
			aind = argmax(Q[(s, a)] for a in actions) #index of selected action
			π[s] .= otherv
			π[s][aind] = topv
		end
	end

	function updateq!(s0, l = 1; rsum = 0.0)
		a0 = sampleπ(s0)
		(s, r, isterm) = tr(s0, a0)
		rsum += r
		Q[(s0, a0)] += α*(r + γ*maximum(Q[(s, a)] for a in actions) - Q[(s0, a0)])
		updateπ!()
		isterm && return (l, rsum)
		updateq!(s, l+1, rsum=rsum)
	end

	for i in 1:n
		s0 = gets0()
		l, rsum = updateq!(s0)
		steps[i] = l
		rewardsums[i] = rsum
	end

	π_det = Dict(s => actions[argmax(π[s])] for s in states)
	return Q, π_det, steps, rewardsums
end

# ╔═╡ 8224b808-5778-458b-b683-ea2603c82117
md"""
### Example 6.6: Cliff Walking
"""

# ╔═╡ 6556dafb-04fa-434c-868a-8d7bb7b5b196
function cliffworld(actions)
	xmax = 12
	ymax = 4
	states = [(x, y) for x in 1:xmax for y in 1:ymax]
	sterm = (xmax, 1)
	gets0 = () -> (1, 1)

	cliffstates = [(x, 1) for x in 2:11]

	# applywind(w, x, y) = (x, y+w)

	boundstate(x::Int64, y::Int64) = (clamp(x, 1, xmax), clamp(y, 1, ymax))

	function cliffcheck(s)
		(x, y) = s
		safereturn = (s, false)
		unsafereturn = ((1, 1), true)
		y > 1 && return safereturn
		(x == 1) && return safereturn
		(x == xmax) && return safereturn
		unsafereturn
	end
	
	function step(s::Tuple{Int64, Int64}, a::GridworldAction)
		(x, y) = s
		(x1, y1) = move(a, x, y)
		s2 = boundstate(x1, y1)
		(s3, hitcliff) = cliffcheck(s2)
	end

	function tr(s0::Tuple{Int64, Int64}, a0::GridworldAction)
		(snew, hitcliff) = step(s0, a0)
		r = hitcliff ? -100 : -1
		isterm = (snew == sterm)
		(snew, r, isterm)
	end

	function episode(π, lmax = 1000)
		s = gets0()
		path = [s]
		a = π(s)
		isterm = false
		l = 1
		while !isterm && (l < lmax)
			(s, r, isterm) = tr(s, a)
			push!(path, s)
			a = π(s)
			l += 1
		end
		return path
	end
			
	return states, sterm, actions, tr, episode, gets0
end	

# ╔═╡ 6bffb08c-704a-4b7c-bfce-b3d099cf35c0
function gridworld_Q_vs_sarsa_solve(gridworld; α=0.5, ϵ=0.1)
	states, sterm, actions, tr, episode, gets0 = gridworld
	tr(gets0(), Up())
	(Qstar1, πstar1, steps1, rsum1) = sarsa_onpolicy(α, 1.0, states, sterm, actions, tr, 250, gets0 = gets0, q0 = 0.0, ϵ = ϵ)
	(Qstar2, πstar2, steps2, rsum2) = Q_learning(α, 1.0, states, sterm, actions, tr, 250, gets0 = gets0, q0 = 0.0, ϵ = ϵ)
	path1 = episode(s -> πstar1[s])
	path2 = episode(s -> πstar2[s])
	p1 = plot(path1, lab = "Sarsa finished in $(length(path1)-1) steps")
	plot!(path2, lab = "Q-learning finished in $(length(path2)-1) steps")
	p2 = plot(rsum1, lab = "Sarsa", xlabel = "episodes", ylabel = "Reward Sum", yaxis = [-100, 0])
	plot!(rsum2, lab = "Q-learning")
	plot(p1, p2, layout = (2, 1), size = (680, 400))
end

# ╔═╡ a4c4d5f2-d76d-425e-b8c9-9047fe53c4f0
gridworld_Q_vs_sarsa_solve(cliffworld(wind_actions1), α=0.5)

# ╔═╡ 05664aaf-575b-4249-974c-d8a2e63f380a
md"""
> *Exercise 6.11* Why is Q-learning considered an *off-policy* control method?

If we compare to the on-policy update rule, the expected value being calculated at each state action pair should be:

$Q_\pi(S_t, A_t) = \text{E}_\pi [R_{t+1} + \gamma Q_\pi(S_{t+1}, A_{t+1})]$

which we estimate with sampling.  In Q-learning, the expected value being estimated is instead:

$Q_\pi(S_t, A_t) = \text{E}_\pi [R_{t+1} + \gamma \text{max}_a Q_\pi(S_{t+1}, a)]$

Also in Sarsa, we select the next action to take and update Q according to that action, so the estimate is reflecting the policy at the time.  In Q-learning, we update the Q-function prior to selecting the action based on the greedy policy with respect to the Q-function at that moment.  However, that assumes that the greedy action is always taken whereas the policy actually being sampled in ϵ-greedy.  The actual action taken at the next step should match the Sarsa algorithm because that next state has not yet been updated in both cases, but in Q-learning that action may not match the assumed greedy action in the update step.
"""

# ╔═╡ 2a3e4617-efbb-4bbc-9c61-8535628e439c
md"""
> *Exercise 6.12* Supposed action selection is greedy.  Is Q-learning then exactly the same algorithm as Sarsa?  Will they make exactly the same action selections and weight updates?

Generally yes, because the term in the Sarsa update that uses the Q value of the subsequent state-action pair will always equal the maximization value in Q-learning.  Both will select the action at the next state that is greedy with respect to the Q-function.  There in one exception for the case where the state is identical through the transition.  In this case, Sarsa will chose the next action from that state prior to updating the Q value for that state.  In the case of Q-learning, the Q update will be identical, but then the subsequent action selection might be different because it occurs after the Q value for that state-action pair was updated.  If the value of that action is decreased for example then an alternative action may be selected on the next step whereas in Sarsa the same action would always be selected the next step.
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

# ╔═╡ 1583b122-3570-4f93-92c8-4dd6bfa0944d
function expected_sarsa(α, γ, states, sterm, actions, tr::Function, n = 1000; gets0 = () -> rand(states), q0 = 0.0, ϵ = 0.1)
	#initialize Q(s,a) and set Q(terminal, a) = 0 for all actions
	Q = Dict((s, a) => q0 for s in states for a in actions)
	for a in actions
		Q[(sterm, a)] = 0.0
	end
	
	nact = length(actions)
	π = Dict(s => fill(1.0 / nact, nact) for s in states) #create policy sampling
	sampleπ(s) = sample(actions, weights(π[s]))
	otherv = ϵ / nact #probability given to a random action
	topv = 1.0 - ϵ + otherv #probability weight given to the top action in the distribution
	
	steps = zeros(Int64, n) #keep track of length of each episode
	rewardsums = zeros(n)
	
	#update policy with epsilon greedy strategy following Q
	function updateπ!()
		for s in states
			aind = argmax(Q[(s, a)] for a in actions) #index of selected action
			π[s] .= otherv
			π[s][aind] = topv
		end
	end

	function updateq!(s0, l = 1; rsum = 0.0)
		a0 = sampleπ(s0)
		(s, r, isterm) = tr(s0, a0)
		rsum += r
		Q[(s0, a0)] += α*(r + γ*sum(π[s][i]*Q[(s, a)] for (i, a) in enumerate(actions)) - Q[(s0, a0)])
		updateπ!()
		isterm && return (l, rsum)
		updateq!(s, l+1, rsum=rsum)
	end

	function run_episode!(i)
		s0 = gets0()
		l, rsum = updateq!(s0)
		steps[i] = l
		rewardsums[i] = rsum
	end

	for i in 1:n
		run_episode!(i)
	end

	π_det = Dict(s => actions[argmax(π[s])] for s in states)
	return Q, π_det, steps, rewardsums
end

# ╔═╡ 84584793-8274-4aa1-854f-b167c7434548
function gridworld_Q_vs_sarsa_vs_expectedsarsa_solve(gridworld; α=0.5, ϵ=0.1)
	states, sterm, actions, tr, episode, gets0 = gridworld
	tr(gets0(), Up())
	(Qstar1, πstar1, steps1, rsum1) = sarsa_onpolicy(α, 1.0, states, sterm, actions, tr, 250, gets0 = gets0, q0 = 0.0, ϵ = ϵ)
	(Qstar2, πstar2, steps2, rsum2) = Q_learning(α, 1.0, states, sterm, actions, tr, 250, gets0 = gets0, q0 = 0.0, ϵ = ϵ)
	(Qstar3, πstar3, steps3, rsum3) = expected_sarsa(α, 1.0, states, sterm, actions, tr, 250, gets0 = gets0, q0 = 0.0, ϵ = ϵ)
	path1 = episode(s -> πstar1[s])
	path2 = episode(s -> πstar2[s])
	path3 = episode(s -> πstar3[s])
	p1 = plot(path1, lab = "Sarsa finished in $(length(path1)-1) steps")
	plot!(path2, lab = "Q-learning finished in $(length(path2)-1) steps")
	plot!(path3, lab = "Expected Sarsa finished in $(length(path3)-1) steps")
	p2 = plot(rsum1, lab = "Sarsa", xlabel = "episodes", ylabel = "Reward Sum", yaxis = [-100, 0])
	plot!(rsum2, lab = "Q-learning")
	plot!(rsum3, lab = "Expected Sarsa")
	plot(p1, p2, layout = (2, 1), size = (680, 400))
end

# ╔═╡ 667666b9-3ab6-4836-953d-9878208103c9
gridworld_Q_vs_sarsa_vs_expectedsarsa_solve(cliffworld(wind_actions1))

# ╔═╡ 6d9ae541-cf8c-4687-9f0a-f008944657e3
function figure_6_3()

end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
Plots = "~1.29.0"
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
git-tree-sha1 = "1e315e3f4b0b7ce40feded39c73049692126cf53"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "7297381ccb5df764549818d9a7d57e45f1057d30"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.18.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

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
git-tree-sha1 = "cc1a8e22627f33c789ab60b36a9132ac050bbf75"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.12"

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
git-tree-sha1 = "336cc738f03e069ef2cac55a104eb823455dca75"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.4"

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

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

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
git-tree-sha1 = "d457f881ea56bbfa18222642de51e0abf67b9027"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.29.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

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

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

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
git-tree-sha1 = "e75d82493681dfd884a357952bbd7ab0608e1dc3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.7"

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

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

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
# ╟─814d89be-cfdf-11ec-3295-49a8f302bbcf
# ╟─495f5606-0567-47ad-a266-d21320eecfc6
# ╠═c3c43440-d8f5-4d16-8735-8d83981b9f15
# ╠═7cd8e898-05c8-4fa0-b12a-60c5c1110cf8
# ╠═8b2a09f5-37c5-4b91-af5c-4596c44b96ea
# ╟─b5187232-d808-49b6-9f7e-a4cbeb6c2b3e
# ╠═9b9ee4f2-f5f9-444b-aa23-85f145d8f9ca
# ╠═7b3e55f4-72b8-48a5-a62a-7ce7ffadae35
# ╠═bc8bad61-a49a-47d6-8fa6-7dcf6c221910
# ╠═6edb550d-5c9f-4ea6-8746-6632806df11e
# ╟─9017093c-a9c3-40ea-a9c6-881ee62fc379
# ╟─5290ae65-6f56-4849-a842-fe347315c6dc
# ╟─47c2cbdd-f6db-4ce5-bae2-8141f30aacbc
# ╠═d501f74b-c3a3-4591-b01f-5df5b71a85f2
# ╠═ae182e8c-6fc0-4043-ac9f-cb6726d173e7
# ╠═659cd7f9-38c7-4dde-818e-be5e26bed09f
# ╠═a116f557-ca8f-4e28-bf8c-84e7e19b30da
# ╠═2786101e-d365-4d6a-8de7-b9794499efb4
# ╠═9db7a268-1e6d-4366-a0ec-ebf54916d3b0
# ╠═b5d0def2-9b65-4e28-a910-d261a25e31f1
# ╟─0b9c6dbd-4eb3-4167-886e-64db9ec7ff04
# ╟─52aebb7b-c2a9-443f-bc03-24cd25793b32
# ╟─e6672866-c0a0-46f2-bb52-25fcc3352645
# ╟─105c5c23-270d-437e-89dd-12297814c6e0
# ╟─48b557e3-e239-45e9-ab15-105bcca96492
# ╠═620a6426-cb29-4010-997b-aa4f9d5f8fb0
# ╠═6f185046-dfdb-41ca-bf3f-e2f90e2e4bc0
# ╠═0ad7b475-6394-4780-908e-849c0684a966
# ╠═22c2213e-5b9b-410f-a0ef-8f1e3db3c532
# ╟─0e59e813-3d48-4a24-b5b3-9a9de7c500c2
# ╟─0d6a11af-b146-4bbc-997e-a11b897269a7
# ╟─1ae30f5d-b25b-4dcb-800f-45c463641ec5
# ╠═705fc9b0-6372-4c5f-8696-78c7cfaa3a76
# ╠═61bbf9db-49a0-4709-83f4-44f228be09c0
# ╠═e19db54c-4b3c-42d1-b016-9620daf89bfb
# ╠═ec285c96-4a75-4af6-8898-ec3176fa34c6
# ╠═56f794c3-8e37-48b4-b953-7ad0a45aadd6
# ╠═331d0b67-c00d-46fd-a175-b8412f6a93c5
# ╟─0ad739c9-8aca-4b82-bf20-c73584d29535
# ╠═031e1106-7408-4c7e-b78e-b713c19123d1
# ╠═1abefc8c-5be0-42b4-892e-14c0c47c16f0
# ╟─39470c74-e554-4f6c-919d-97bec1eec0f3
# ╠═e9359ca3-4d11-4365-bc6e-7babc6fcc7de
# ╠═dee6b500-0ba1-4bbc-b217-cbb9ad47ad06
# ╟─db31579e-3e56-4271-8fc3-eb13bc95ac27
# ╟─b59eacf8-7f78-4015-bf2c-66f89bf0e24e
# ╠═aa0791a5-8cf1-499b-9900-4d0c59be808c
# ╠═ced61b99-9073-4dee-afbf-82531e59c7d8
# ╟─44c49006-e210-4f97-916e-fe62f36c593f
# ╠═f90263de-1053-48cb-8240-56112d6dc67f
# ╟─8224b808-5778-458b-b683-ea2603c82117
# ╠═6556dafb-04fa-434c-868a-8d7bb7b5b196
# ╠═6bffb08c-704a-4b7c-bfce-b3d099cf35c0
# ╠═a4c4d5f2-d76d-425e-b8c9-9047fe53c4f0
# ╟─05664aaf-575b-4249-974c-d8a2e63f380a
# ╟─2a3e4617-efbb-4bbc-9c61-8535628e439c
# ╟─6e06bd39-486f-425a-bbca-bf363b58988c
# ╠═1583b122-3570-4f93-92c8-4dd6bfa0944d
# ╠═84584793-8274-4aa1-854f-b167c7434548
# ╠═667666b9-3ab6-4836-953d-9878208103c9
# ╠═6d9ae541-cf8c-4687-9f0a-f008944657e3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
