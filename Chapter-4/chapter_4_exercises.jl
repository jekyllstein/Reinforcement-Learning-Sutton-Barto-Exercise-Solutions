### A Pluto.jl notebook ###
# v0.18.0

using Markdown
using InteractiveUtils

# ╔═╡ 2de434b0-92f5-4ceb-9d8c-3315cdd7ec5a
using BenchmarkTools, Plots

# ╔═╡ 4017d910-3635-4ffa-ac1b-919a7bff1e6e
md"""
## Chapter 4
# Dynamic Programming
"""

# ╔═╡ 55276004-877e-47c0-b5b5-49dbe29aa6f7
md"""
### 4.1 Policy Evaluation (Prediction)
"""

# ╔═╡ d46254d0-5ad5-4796-aaa4-a7c135c56da8
plotly()

# ╔═╡ 5f574281-88e0-4d82-bf18-1cfe4c1990fc
@enum GridworldAction up down left right

# ╔═╡ df9593ca-6d27-45de-9d46-79bddc7a3862
#p is the state transition function for an mdp which maps the 4 arguments to a probability.  This function uses p to generate two dictionaries.  The first maps each state to a set of possible actions in that state.  The second maps each state/action pair to a set of possible transition/reward pairs
function get_sa_keys(p::Dict{Tuple{A, B, A, C}, T}) where {T <: Real, A, B, C}
	#map from states to a list of possible actions
	state_actions = Dict{A, Set{C}}()

	#map from state action pairs to a list of possible newstate/reward pairs
	sa_s′rewards = Dict{Tuple{A, C}, Set{Tuple{A, B}}}()
	for k in keys(p)
		(s′, r, s, a) = k
		haskey(state_actions, s) ? push!(state_actions[s], a) : state_actions[s] = Set([a])
		haskey(sa_s′rewards, (s,a)) ? push!(sa_s′rewards[(s,a)], (s′, r)) : sa_s′rewards[(s,a)] = Set([(s′,r)])
	end
	return state_actions, sa_s′rewards
end	

# ╔═╡ 59b91c65-3f8a-4015-bb08-d7455623101c
function bellman_value!(V::Dict, p::Dict, sa_keys::Tuple, π::Dict, γ::Real)
	delt = 0.0
	for s in intersect(keys(sa_keys[1]), keys(π))
		v = V[s]
		actions = intersect(sa_keys[1][s], keys(π[s]))
		# if !isempty(actions)
			V[s] = 	sum(π[s][a] * 
						sum(p[(s′,r,s,a)] * (r + γ*V[s′]) 
							for (s′,r) in sa_keys[2][(s,a)])
					for a in actions)
		# end
		delt = max(delt, abs(v - V[s]))
	end
	return delt
end

# ╔═╡ 5b912508-aa15-470e-be4e-430e88d8a68d
function iterative_policy_eval_v(π::Dict, θ::Real, mdp::NamedTuple, γ::Real, V::Dict, delt::Real, nmax::Real)
	(p, sa_keys) = mdp
	if nmax <= 0 || delt <= θ
		return V
	else 
		delt = bellman_value!(V, p, sa_keys, π, γ)
		iterative_policy_eval_v(π, θ, mdp, γ, V, delt, nmax - 1)	
	end
end

# ╔═╡ e6cdb2be-697d-4191-bd5a-9c129b32246d
function iterative_policy_eval_v(π::Dict, θ::Real, mdp::NamedTuple, γ::Real, Vinit = 0.0; nmax = Inf)
	(p, sa_keys) = mdp
	V = Dict(s => Vinit for s in keys(sa_keys[1]))
	delt = bellman_value!(V, p, sa_keys, π, γ)
	iterative_policy_eval_v(π, θ, mdp, γ, V, delt, nmax - 1)	
end

# ╔═╡ 9253064c-7dfe-445f-b377-fc1acbb6886e
function iterative_policy_eval_v(π::Dict, θ::Real, mdp::NamedTuple, γ::Real, Vinit::Dict; nmax=Inf)
	(p, sa_keys) = mdp
	V = deepcopy(Vinit)
	delt = bellman_value!(V, p, sa_keys, π, γ)
	iterative_policy_eval_v(π, θ, mdp, γ, V, delt, nmax - 1)
end

# ╔═╡ 7c8c99fd-1b55-494c-898f-194c61e36724
md"""
## Example 4.1
"""

# ╔═╡ c7bdf32a-2f89-4bf8-916b-7558ceedb628
function gridworld4x4_mdp()
	S = collect(1:14)
	s_term = 0
	A = [up, down, left, right]
	#define p by iterating over all possible states and transitions
	p = Dict{Tuple{Int64, Int64, Int64, GridworldAction}, Float64}()

	#there is 0 reward and a probability of 1 staying in the terminal state for all 	actions taken from the terminal state
	for a in A
		push!(p, (0, 0, 0, a) => 1.0)
	end

	#add cases where end up in the terminal state
	push!(p, (s_term, -1, 14, right) => 1.0)
	push!(p, (s_term, -1, 11, down) => 1.0)
	push!(p, (s_term, -1, 1, left) => 1.0)
	push!(p, (s_term, -1, 4, up) => 1.0)
	
	for s in S
		for a in A
			for s′ in S
				check = if a == right
					if (s == 3) || (s == 7) || (s == 11) 
						s′ == s
					else 
						s′ == s+1
					end
				elseif a == left
					if (s == 4) || (s == 8) || (s == 12)
						s′ == s
					else 
						s′ == s-1
					end
				elseif a == up
					if (s == 1) || (s == 2) || (s == 3)
						s′ == s
					else
						s′ == s - 4
					end
				elseif a == down
					if (s == 12) || (s == 13) || (s == 14)
						s′ == s
					else
						s′ == s + 4
					end
				end
				check && push!(p, (s′,-1,s,a) => 1.0)
			end
		end
	end
	sa_keys = get_sa_keys(p)
	return (p = p, sa_keys = sa_keys)
end

# ╔═╡ 6048b106-458e-4e3b-bba9-5f3578458c7c
#forms a random policy for a generic finite state mdp.  The policy is a dictionary that maps each state to a dictionary of action/probability pairs.
function form_random_policy(sa_keys)
	Dict([begin
		s = k[1]
		actions = k[2]
		l = length(actions)
		p = inv(l)
		s => Dict(a => p for a in actions)
	end
	for k in sa_keys[1]])
end

# ╔═╡ 0d0e82e4-b3a4-4528-9288-285fdc5aa8af
function makefig4_1(nmax=Inf)
	gridworldmdp = gridworld4x4_mdp()
	π_rand = form_random_policy(gridworldmdp[2])
	V = iterative_policy_eval_v(π_rand, eps(0.0), gridworldmdp, 1.0, nmax = nmax)
	[(s, V[s]) for s in 0:14]
end

# ╔═╡ 84068701-40d7-4a5e-93f2-af2a751ab2ec
makefig4_1(Inf)

# ╔═╡ f80580b3-f370-4a02-a9e2-ed791f380521
md"""
> *Exercise 4.1* In Example 4.1, if $\pi$ is the equiprobable random policy, what is $q_{\pi}(11,\text{down})$?  What is $q_{\pi}(7,\text{down})$?
$q_{\pi}(11, \text{down}) = -1$ because this will transition into the terminal state and terminate the episode receiving the single reward of -1.

$q_{\pi}(7,\text{down})=-15$ because we are gauranteed to end up in state 11 and receive a reward of -1 from the first action.  Once we are in state 11, we can add $v_{\pi_{random}}(11)=-14$ to this value since the rewards are not discounted.

"""

# ╔═╡ 0bebb164-4347-4cff-8169-9f4da4553ae6
md"""
> *Exercise 4.2* In Example 4.1, supposed a new state 15 is added to the gridworld just below state 13, and its actions, $\text{left}$, $\text{up}$, $\text{right}$, and $\text{down}$, take the agent to states 12, 13, 14, and 15 respectively.  Assume that the transitions *from* the original states are unchanged.  What, then is $v_{\pi}(15)$ for the equiprobable random policy?  Now supposed the dynamics of state 13 are also changed, such that action $\text{down}$ from state 13 takes the agent to the new state 15.  What is $v_{\pi}(15)$ for the equiprobable random policy in this case?

In the first case, we can never re-enter state 15 from any other state, so we can use the average of the value function in the states it transitions into.  

$v_{\pi}(15) = 0.25 \times \left ( v_{\pi}(12) + v_{\pi}(13) + v_{\pi}(14)+ v_{\pi}(15) \right )$ 
$v_\pi(15) = 0.25 \times (-22 + -20 + -14 + v_\pi(15))$

Solving for the value at 15 yields:

$v_\pi(15) = \frac{0.25 \times -56}{0.75}=-18.666 \dots$

In the second case, the value function at 13 and every other state will be different because state 15 can be entered from 13 and thus any other state eventually.  Additional steps of policy iteration will need to happen to update the values.  Carrying out this calculation below using the same method used to generate Figure 4.1, we see a value of -20 which is equal to the original value of state 13.  If we compare state 15 and 13, we see that it shares the same transition dynamics as the original state 13 asside from the up transition.  The original 13 however had a state immediately above it that shared the same value.  Noticing this symmetry we could infer that the added state 15 would have the same value as the original state 13.

Try writing down the bellman equations for state 13 and 15 and try to reason that the value for 13 is unchanged.  Is there a rigorous way to identify that the value functions are unchanged even in the second case?
"""

# ╔═╡ 10c9b166-3a88-460e-82e8-a16c020c1378
#Exercise 4.2 part 2
function gridworld_modified_mdp()
	S = collect(1:15)
	s_term = 0
	A = [up, down, left, right]

	#no discounting in this episodic task
	γ = 1.0
	
	#define p by iterating over all possible states and transitions
	p = Dict{Tuple{Int64, Int64, Int64, GridworldAction}, Float64}()

	#there is 0 reward and a probability of 1 staying in the terminal state for all 	actions taken from the terminal state
	for a in A
		push!(p, (0, 0, 0, a) => 1.0)
	end

	#add cases where end up in the terminal state
	push!(p, (s_term, -1, 14, right) => 1.0)
	push!(p, (s_term, -1, 11, down) => 1.0)
	push!(p, (s_term, -1, 1, left) => 1.0)
	push!(p, (s_term, -1, 4, up) => 1.0)
	
	for s in S
		for a in A
			for s′ in S
				check = if a == right
					if (s == 3) || (s == 7) || (s == 11) 
						s′ == s
					elseif s == 15
						s′ == 14
					else
						(s != 14) && (s′ == s+1)
					end
				elseif a == left
					if (s == 4) || (s == 8) || (s == 12)
						s′ == s
					elseif (s == 15)
						s′ == 12
					else
						s′ == s-1
					end
				elseif a == up
					if (s == 1) || (s == 2) || (s == 3)
						s′ == s
					elseif (s == 15)
						s′ == 13
					else
						s′ == s - 4
					end
				elseif a == down
					if (s == 12) || (s == 14) || (s == 15)
						s′ == s
					elseif (s == 13)
						s′ == 15
					else
						(s != 11) && (s′ == s + 4)
					end
				end
				check && push!(p, (s′,-1,s,a) => 1.0)
			end
		end
	end
	sa_keys = get_sa_keys(p)
	return (p = p, sa_keys = sa_keys)
end		

# ╔═╡ 1618ec46-6e13-42bf-a7f2-68f8dbe3714c
function exercise4_2(nmax=Inf)
	gridworldmdp = gridworld_modified_mdp()
	π_rand = form_random_policy(gridworldmdp[2])
	V = iterative_policy_eval_v(π_rand, eps(0.0), gridworldmdp, 1.0, nmax = nmax)
	[(s, V[s]) for s in 0:15]
end

# ╔═╡ e4bfdaca-3f3d-43bb-b8aa-7536adbff662
#calculates value function for gridworld example in part 2 of exercise 4.2 with an added state 15
exercise4_2()

# ╔═╡ 35e1ffe5-d36a-449d-aa73-c618e2855042
md"""
> *Exercise 4.3* What are the equations analogous to (4.3), (4.4), and (4.5), but for *action*-value functions instead of state-value functions?

Equation (4.3)

$v_\pi(s)=\mathbb{E}_\pi [R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s]$

action-value equivalent

$q_\pi(s,a)=\mathbb{E}_\pi [R_{t+1} + \gamma q_\pi(S_{t+1},A_{t+1}) | S_t = s, A_t=a]$

Equation (4.4)

$v_\pi(s)=\sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma v_\pi(s')]$

action-value equivalent

$q_\pi(s,a)=\sum_{s',r} p(s',r|s,a)[r + \gamma \sum_{a'} \pi(s',a') q_\pi(s',a')]$

Equation (4.5)

$v_{k+1}(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma v_k(s')]$

action-value equivalent

$q_{k+1}(s,a) = \sum_{s',r} p(s',r|s,a)[r + \gamma \sum_{a'} \pi(a'|s') q_k(s',a')]$
"""

# ╔═╡ 67b06f3b-13df-4b27-ad80-d112432e8f42
md"""
### 4.3 Policy Iteration
"""

# ╔═╡ f0e5d2e6-3d00-4ffc-962e-e98d4bb28e4e
function policy_improvement_v(π::Dict, mdp::NamedTuple, γ::Real, V::Dict)
	(p, sa_keys) = mdp
	π_new = Dict(begin
		actions = sa_keys[1][s]
		newdist = Dict(a => 
				sum(p[(s′,r,s,a)] * (r + γ*V[s′]) for (s′,r) in sa_keys[2][(s,a)])
				for a in actions)
		new_action = argmax(newdist)
		s => Dict(new_action => 1.0)
	end
	for s in keys(sa_keys[1]))

	policy_stable = mapreduce((a,b) -> a && b, keys(sa_keys[1])) do s
		argmax(π[s]) == argmax(π_new[s])
	end

	return (policy_stable, π_new)
end 

# ╔═╡ 4d15118f-f1ab-4115-bcc9-7f98246eca1c
function policy_iteration_v(mdp::NamedTuple, π::Dict, γ::Real, Vold::Dict, iters, θ, evaln, policy_stable, resultlist)
	policy_stable && return (true, resultlist)
	V = iterative_policy_eval_v(π, θ, mdp, γ, Vold, nmax = evaln)
	(V == resultlist[end][1]) && return (true, resultlist)
	newresultlist = vcat(resultlist, (V, π))
	(iters <= 0) &&	return (false, newresultlist)
	(new_policy_stable, π_new) = policy_improvement_v(π, mdp, γ, V)
	policy_iteration_v(mdp, π_new, γ, V, iters-1, θ, evaln, new_policy_stable, newresultlist)
end

# ╔═╡ 77250f6b-60d1-426f-85b2-497186b86c50
function begin_policy_iteration_v(mdp::NamedTuple, π::Dict, γ::Real; iters=Inf, θ=eps(0.0), evaln = Inf, V = iterative_policy_eval_v(π, θ, mdp, γ, nmax = evaln))
	resultlist = [(V, π)]
	(policy_stable, π_new) = policy_improvement_v(π, mdp, γ, V)
	policy_iteration_v(mdp, π_new, γ, V, iters-1, θ, evaln, policy_stable, resultlist)
end

# ╔═╡ c47882c7-ded1-440c-a9a3-0b89a0e7a011
function gridworld_policy_iteration(nmax=10; θ=eps(0.0), γ=1.0)
	gridworldmdp = gridworld4x4_mdp()
	π_rand = form_random_policy(gridworldmdp[2])
	(policy_stable, resultlist) = begin_policy_iteration_v(gridworldmdp, π_rand, γ, iters = nmax)
	(Vstar, πstar) = resultlist[end]
	(policy_stable, [(s, first(keys(πstar[s]))) for s in 0:14])
end

# ╔═╡ 0079b02d-8895-4dd4-9557-5f08ac341404
#seems to match optimal policy from figure 4.1
gridworld_policy_iteration()

# ╔═╡ 0d6936d1-38af-45f1-b496-da49b60f11f8
md"""
> *Exercise 4.4* The policy iteration algorithm on page 80 has a subtle bug in that it may never terminate if the policy continually switches between two or more policies that are equally good.  This is okay for pedagogy, but not for actual use.  Modify the pseudocode so that convergence is guaranteed.

Initialize $V_{best}$ at the start randomly and replace it with the first value function calculated.  After each policy improvement, replace $V_{best}$ with the new value function, however add a check after step 2. that if the value function is the same as $V_{best}$ then stop.  This would ensure that no matter how many equivalent policies are optimal, they would all share the same value function and thus trigger the termination condition.
"""

# ╔═╡ ad1bf1d2-211d-44ca-a258-fc6e112785da
md"""
> *Exercise 4.5* How would policy iteration be defined for action values?  Give a complete algorithm for computer $q_*$, analogous to that on page 80 for computing $v_*$.  Please pay special attention to this exercise, because the ideas involved will be used throughout the rest of the book.

**Policy Iteration (using iterative policy evaluation) for estimating $\pi \approx \pi_*$ using action-values**
1. Initialization
$Q(s,a) \in \mathbb{R} \text{ and } \pi(s) \in \mathcal{A}(s) \text{ arbitrarily for all } s \in \mathcal{S}; Q(terminal,a) \doteq 0 \space \forall \space a \in \mathcal{A}$

2. Policy Evaluation
   
   Loop:

$\Delta \leftarrow 0$

Loop for each $s \in \mathcal{S}$:

Loop for each $a \in \mathcal{A}(s)$:

$q \leftarrow Q(s,a)$
$Q(s,a) \leftarrow \ \sum_{s',r} p(s',r|s,a)[r + \gamma \sum_{a'} \pi(s',a') Q(s',a')]$
$\Delta \leftarrow \max{(\Delta, |q - Q(s,a)|)}$

until $\Delta < \theta$ (a small positive number determining the accuracy of estimation)

3. Policy Improvement
*policy-stable* $\leftarrow$ *true*

For each $s \in \mathcal{S}$:

$old-action \leftarrow \pi(s)$

$\pi(s) \leftarrow \text{argmax}_a Q(s,a)$

If *old-action* $\neq \pi(s)$, then *policy-stable* $\leftarrow false$

If *policy-stable*, then stop and return $Q \approx q_*$ and $\pi \approx \pi_*$; else go to 2
"""

# ╔═╡ e316f59a-8070-4510-96f3-15498897347c
md"""
>*Exercise 4.6* Suppose you are restricted to considering only policies that are *ϵ-soft*, meaning that the probability of selecting each action in each state, s, is at least $\epsilon / |\mathcal{A}(s)|$.  Describe qualitatively the changes that would be required in each of the steps 3,2,and 1, in that order, of the policy iteration algorithm for $v_*$ on page 80.

For step 3: 
To get the old-action take the argmax over possible actions of the policy distribution for state s.  Rewrite π as π(a|s).
Instead of having a probability of 1.0 for the argmax of the expression, we must adjust the value to be $1.0 - \epsilon$.  Similarly the *old-action* and *new-action* should be the argmax of the policy distribution at state s rather than the single value.

For step 2:

The expression for updating the value function should have a sum over possible actions weighted by the policy distribution for each action.  The inner sum can remain the same except the policy argument for p should be replaced with the variable summing over actions.

For step 1:

The initialization of the policy function should be a uniform distribution over all possible actions for each state rather than a single action value.
"""

# ╔═╡ 5dbfb100-49a8-4f9d-a752-bda4da54699e
md"""
> *Exercise 4.7 (programming)* Write a program for policy iteration and re-solve Jack's car rental problem with the following changes.  One of Jack's employees at the first location rides a bus home each night and lives near the second location. She is happy to shuttle one car to the second location for free. Each additional car still costs $2, as do all cars moved in the other direction. In addition, Jack has limited parking space at each location.  If more than 10 cars are kept overnight at a location (after any moving of cars), then an additional cost of $4 must be incurred to use a second parking lot (independent of how many cars are kept there). These sorts of nonlinearities and arbitrary dynamics often occur in real problems and cannot easily be handled by optimization methods other than dynamic programming. To check your program, first replicate the results given for the original problem.
"""

# ╔═╡ e722b7e0-63a3-4195-b13e-0449abb3cc39
poisson(n, λ) = exp(-λ) * (λ^n) / factorial(n)

# ╔═╡ 66b7746a-5dd2-4d3b-8026-0c9c10fb5ba3
function car_rental_mdp(;nmax=20, λs = (3,4,3,2), movecost = 2, rentcredit = 10, movemax=5)
	#enumerate all possible states from which to transition
	S = ((x, y) for x in 0:nmax for y in 0:nmax)

	#check that new states are valid
	function checkstate(S)
		@assert (S[1] >= 0) && (S[1] <= nmax)
		@assert (S[2] >= 0) && (S[2] <= nmax)
	end
	
	#before proceeding, it will be useful to have a lookup table of probabilities for all of the possible rental and return requests at each location.  Since we can never rent more cars than are available, 0 to nmax-1 is the only range that needs to be considered.  For returns, we can receive any arbitrary number but any that exceed nmax will be returned.  Thus we may have a situation where receiving as a return any number greater than or equal to a given value will result in the same state.  To calculate such a probability we need to sum up all of the probilities for return values less than that and subtract it from 1.  If we have 0 cars at a given location prior to returns, then the maximum return value we would need to calculate is up to nmax-1.  That way the probability leading to nmax cars would be 1 minus the sum of every other probility calculated from 0 to nmax-1.
	rentprobs = Dict((loc, rent) => poisson(rent, λs[loc]) for rent in 0:nmax-1 for loc in 1:2)
	retprobs = Dict((loc, ret) => poisson(ret, λs[loc+2]) for ret in 0:nmax-1 for loc in 1:2)

	#define p by iterating over all possible states and transitions
	ptf = Dict{Tuple{Tuple{Int64, Int64}, Int64, Tuple{Int64, Int64}, Int64}, Float64}()
	
	for s in S
		#for actions a negative number indicates moving cars from 2 to 1
		#a positive number indicates moving cars from 1 to 2
		for a in -min(movemax, s[2]):min(movemax, s[1])
			#after taking action a, we have our first intermediate state for the next morning which cannot exceed nmax at each location
			sint1 = (min(s[1]-a, nmax), min(s[2]+a, nmax))
			checkstate(sint1)
			
			#the next day we can only rent cars from each location that are available
			for (rent1, rent2) in ((x,y) for x in 0:sint1[1] for y in 0:sint1[2])
				#after specifing the number of cars rented we have our final reward value
				r = rentcredit*(rent1+rent2) - movecost*abs(a)
				
				#if we n cars from a given location, we could have received rental requests for that number or higher.  So the probability of such a rental is 1 minus the sum of the probability of receiving every request less than that number
				function calcrentprob(loc, nrent)
					ncars = sint1[loc]
					@assert nrent <= ncars
					if ncars == 0
						1.0
					elseif nrent < ncars
						rentprobs[(loc, nrent)]
					else
						1.0 - sum(rentprobs[(loc, r)] for r in 0:nrent-1)
					end
				end

				#calculate the probability of renting these cars at these locations
				prent = calcrentprob(1, rent1)*calcrentprob(2, rent2)

				#new intermediate state after renting cars
				sint2 = (sint1[1]-rent1, sint1[2]-rent2)
				checkstate(sint2)

				#after receiving returns, we can only increase the number of cars at each loaction, so the possible final transition states we can end up with are as follows
				for s′ in ((x,y) for x in sint2[1]:nmax for y in sint2[2]:nmax)
					checkstate(s′)
					
					#change in cars from returns
					delt1 = s′[1] - sint2[1]
					delt2 = s′[2] - sint2[2]
					
					function pdelt(loc, delt)
						if sint2[loc] == nmax
							#in this case the location already had the maximum number of cars so any return value is possible
							1.0
						elseif s′[loc] < nmax
							#in this the requested returns match delta
							retprobs[(loc,delt)]
						else
							1.0 - sum(retprobs[(loc, r)] for r in 0:delt-1)
						end
					end

					pret = pdelt(1, delt1)*pdelt(2, delt2)

					totalprob = prent*pret
					
					#finally we can assign the probability of the entire transition, if keys appear more than once, we need to add the probabilities since there are multiple ways to observe the same transition
					newkey = (s′, r, s, a)
					basevalue = haskey(ptf, newkey) ? ptf[newkey] : 0.0
					ptf[newkey] = basevalue + totalprob
				end
			end
		end
	end
	sa_keys = get_sa_keys(ptf)
	return (p = ptf, sa_keys = sa_keys)
end

# ╔═╡ b775fac4-8da9-4a27-a830-934df4b86dc2
jacks_car_mdp = car_rental_mdp()

# ╔═╡ f0552339-b9ae-4036-8c6a-c232faee2b42
function convertcarpolicy(V, π, nmax=20)
	vmat = zeros(nmax+1, nmax+1)
	pmat = zeros(nmax+1, nmax+1)
	A = -nmax:nmax
	for i = 0:nmax
		for j = 0:nmax
			vmat[i+1,j+1] = V[(i,j)]
			a = argmax(π[(i,j)])
			pmat[i+1,j+1] = a
		end
	end
	return (value=vmat, policy=pmat)
end

# ╔═╡ b9998e51-f7e5-46c2-80cd-e327911db01b
#first test that the policy evaluation works on the mdp
function car_rental_policy_eval(mdp, nmax=Inf; θ = eps(0.0), γ=0.9)
	states = keys(mdp.sa_keys[1])
	π_0 = Dict(s => Dict(0 => 1.0) for s in states)
	V0 = iterative_policy_eval_v(π_0, θ, mdp, γ, nmax = nmax)
	nullpolicymats = convertcarpolicy(V0, π_0)
	(V0, π_0, nullpolicymats)
end

# ╔═╡ 3c874757-2f48-4ba0-93ce-38c019fb1f1b
V0_car_rental_eval = car_rental_policy_eval(jacks_car_mdp, Inf)	

# ╔═╡ a2e0108a-4bf7-40a9-8c06-dc8403042988
heatmap(V0_car_rental_eval[3][1], title="Value Function No Movement Car Rental Policy")

# ╔═╡ 1bf3eba7-1a02-4035-962c-73c3fda304bd
#now try policy iteration
function car_rental_policy_iteration(mdp, nmax=10; θ=eps(0.0), γ=0.9, null_policy_eval = car_rental_policy_eval(mdp))
	(V0, π_0, mats) = null_policy_eval
	(converged, resultlist) = begin_policy_iteration_v(mdp, π_0, γ, V = V0, iters = nmax, θ = θ)
	(converged, [(Vstar, πstar, convertcarpolicy(Vstar, πstar)) for (Vstar, πstar) in resultlist])
end

# ╔═╡ 4d00ffa4-40d5-4dbc-a8e2-5a57cbbcbacd
example4_2_results = car_rental_policy_iteration(jacks_car_mdp, θ=0.01, null_policy_eval=V0_car_rental_eval)

# ╔═╡ 2c4d1304-fa80-4d9f-98f8-5f8d3107110d
function plotcarpolicy(results)
	πheatmaps = [a[3][2] for a in results]
	finalvaluemap = results[end][3][1]
	plist = [heatmap(h, title="π_$(i-1)") for (i,h) in enumerate(πheatmaps)]
	pvalue = surface(finalvaluemap, title="Value Function after $(length(results)-1) Iterations", legend = false)
	πplots = plot(Tuple(plist)...)
	plot(πplots, pvalue, layout = (2,1), size=(700, 900))
end

# ╔═╡ 7c2bd3b2-8388-4e80-b423-41cf2a4c95ef
plotcarpolicy(example4_2_results[2])

# ╔═╡ ad5e013f-7938-4e3d-acec-bfce21b63b61
function car_rental_modified_mdp(;nmax=20, λs = (3,4,3,2), movecost = 2, rentcredit = 10, movemax=5)
	#enumerate all possible states from which to transition
	S = ((x, y) for x in 0:nmax for y in 0:nmax)

	#lookup tables for rental and return request probabilities
	rentprobs = Dict((loc, rent) => poisson(rent, λs[loc]) for rent in 0:nmax-1 for loc in 1:2)
	retprobs = Dict((loc, ret) => poisson(ret, λs[loc+2]) for ret in 0:nmax-1 for loc in 1:2)

	#define p by iterating over all possible states and transitions
	ptf = Dict{Tuple{Tuple{Int64, Int64}, Int64, Tuple{Int64, Int64}, Int64}, Float64}()
	
	for s in S
		#for actions a negative number indicates moving cars from 2 to 1
		#a positive number indicates moving cars from 1 to 2
		for a in -min(movemax, s[2]):min(movemax, s[1])
			#after taking action a, we have our first intermediate state for the next morning which cannot exceed nmax at each location
			sint1 = (min(s[1]-a, nmax), min(s[2]+a, nmax))

			move_expense = movecost * ((a > 0) ? (a-1) : -a)
			
			#the next day we can only rent cars from each location that are available
			for (rent1, rent2) in ((x,y) for x in 0:sint1[1] for y in 0:sint1[2])
				#if we n cars from a given location, we could have received rental requests for that number or higher.  So the probability of such a rental is 1 minus the sum of the probability of receiving every request less than that number
				function calcrentprob(loc, nrent)
					ncars = sint1[loc]
					if ncars == 0
						1.0
					elseif nrent < ncars
						rentprobs[(loc, nrent)]
					else
						1.0 - sum(rentprobs[(loc, r)] for r in 0:nrent-1)
					end
				end

				#calculate the probability of renting these cars at these locations
				prent = calcrentprob(1, rent1)*calcrentprob(2, rent2)

				#new intermediate state after renting cars
				sint2 = (sint1[1]-rent1, sint1[2]-rent2)

				#after receiving returns, we can only increase the number of cars at each loaction, so the possible final transition states we can end up with are as follows
				for s′ in ((x,y) for x in sint2[1]:nmax for y in sint2[2]:nmax)
		
					#change in cars from returns
					delt1 = s′[1] - sint2[1]
					delt2 = s′[2] - sint2[2]
					
					function pdelt(loc, delt)
						if sint2[loc] == nmax
							#in this case the location already had the maximum number of cars so any return value is possible
							1.0
						elseif s′[loc] < nmax
							#in this the requested returns match delta
							retprobs[(loc,delt)]
						else
							1.0 - sum(retprobs[(loc, r)] for r in 0:delt-1)
						end
					end

					pret = pdelt(1, delt1)*pdelt(2, delt2)

					#after specifing the number of cars rented, moved, and at each location we can calculate our total reward
					secondlotcost = 4 * ((s′[1] > 10) + (s′[2] > 10))
					r = rentcredit*(rent1+rent2) - move_expense - secondlotcost
					
					totalprob = prent*pret
					
					#finally we can assign the probability of the entire transition, if keys appear more than once, we need to add the probabilities since there are multiple ways to observe the same transition
					newkey = (s′, r, s, a)
					basevalue = haskey(ptf, newkey) ? ptf[newkey] : 0.0
					ptf[newkey] = basevalue + totalprob
				end
			end
		end
	end
	sa_keys = get_sa_keys(ptf)
	return (p = ptf, sa_keys = sa_keys)
end

# ╔═╡ 5e18e5f1-3d19-4ba7-a7e4-613e6d70a806
modified_jacks_car_mdp = car_rental_modified_mdp()

# ╔═╡ 0a60d83e-3ef6-449f-9131-0c0d0777d413
V0_modified_car_rental_eval = car_rental_policy_eval(modified_jacks_car_mdp)	

# ╔═╡ 65408c28-eee7-4f60-b71a-cbf31a3c43aa
heatmap(V0_modified_car_rental_eval[3][1], title="Value Function No Movement Modified Car Rental Policy")

# ╔═╡ c973a895-07bb-4f8f-8498-4b4773bdd02d
exercise4_7_results = car_rental_policy_iteration(modified_jacks_car_mdp, θ=0.01, null_policy_eval=V0_modified_car_rental_eval)

# ╔═╡ 48ff63c9-97f4-4162-a009-d05517b8f06f
plotcarpolicy(exercise4_7_results[2])

# ╔═╡ 2bedc22e-9615-4fb4-94bf-6a0e7114c417
md"""
## 4.4 Value Iteration
"""

# ╔═╡ 7140970d-d4a7-45bc-9626-26cf1a2f945b
function bellman_optimal_value!(V::Dict, p::Dict, sa_keys::Tuple, γ::Real)
	delt = 0.0
	for s in keys(sa_keys[1])
		v = V[s]
		actions = sa_keys[1][s]
		V[s] = maximum(sum(p[(s′,r,s,a)] * (r + γ*V[s′]) for (s′,r) in sa_keys[2][(s,a)]) for a in actions)
		delt = max(delt, abs(v - V[s]))
	end
	return delt
end

# ╔═╡ 39e0e313-79e7-4343-8e02-526f30a66aad
function calculatepolicy(mdp::NamedTuple, γ::Real, V::Dict)
	(p, sa_keys) = mdp
	πraw = Dict(begin
		actions = sa_keys[1][s]
		newdist = Dict(a => 
				sum(p[(s′,r,s,a)] * (r + γ*V[s′]) for (s′,r) in sa_keys[2][(s,a)])
				for a in actions)
		s => newdist
	end
	for s in keys(sa_keys[1]))
	πstar = Dict(s => Dict(argmax(πraw[s]) => 1.0) for s in keys(πraw))
	πstar, πraw
end 

# ╔═╡ f9b3f359-0c24-4a70-9ba8-be185cc01a62
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

# ╔═╡ a5f1a71f-3c28-49ed-8f29-ff164c4ea02c
function begin_value_iteration_v(mdp::NamedTuple, γ::Real; θ = eps(0.0), nmax=Inf, Vinit = 0.0)
	(p, sa_keys) = mdp
	V = Dict(s => Vinit for s in keys(sa_keys[1]))
	newV = deepcopy(V)
	delt = bellman_optimal_value!(newV, p, sa_keys, γ)
	value_iteration_v(θ, mdp, γ, newV, delt, nmax-1, [V, newV])
end

# ╔═╡ 3d7f4a19-316e-4874-8c28-8e8fe96a9002
function begin_value_iteration_v(mdp::NamedTuple, γ::Real, V; θ = eps(0.0), nmax=Inf)
	(p, sa_keys) = mdp
	newV = deepcopy(V)
	delt = bellman_optimal_value!(newV, p, sa_keys, γ)
	value_iteration_v(θ, mdp, γ, newV, delt, nmax-1, [V, newV])
end

# ╔═╡ c2fca344-c1db-4416-8ce4-39eae9e972af
begin_value_iteration_v(gridworld4x4_mdp(), 1.0)

# ╔═╡ 315562d0-2bf6-431a-be3f-fb7d2af248b5
md"""
### Example 4.3: Gambler's Problem
"""

# ╔═╡ c456fbdc-0d52-41a9-8e18-dee3c2f4e258
function make_gambler_mdp(p::Real)
	ptf = Dict{Tuple{Int64, Int64, Int64, Int64}, Float64}()
	stermwin = 100
	stermlose = 0
	for s in 1:99
		for a in 0:min(s,100-s)
			swin = s+a
			slose = s-a
			if swin == stermwin
				ptf[(swin, 1, s, a)] = p
			else
				ptf[(swin, 0, s, a)] = p
			end

			ptf[(slose, 0, s, a)] = 1.0-p
		end
	end
	sa_keys = get_sa_keys(ptf)
	V = Dict(s => 0.0 for s in keys(sa_keys[1]))
	V[stermwin] = 0.0
	V[stermlose] = 0.0
	return (p = ptf, sa_keys = sa_keys, Vinit = V)
end		

# ╔═╡ 8bc5e53f-632d-4ea4-98ca-b3740d98c297
gambler_mdp = make_gambler_mdp(0.4)

# ╔═╡ ed8c454a-817b-4736-a70d-e2ca4b946ed9
function multiargmax(π_s::Dict{A, B}) where {A, B}
#takes a distribution over actions and returns a set of actions that share the same maximum value.  If there is a unique maximum then only one element will be in the set
	a_max = argmax(π_s)
	p_max = π_s[a_max]
	a_set = Set([a_max])
	for a in keys(π_s)
		(π_s[a] ≈ p_max) && push!(a_set, a)
	end
	return a_set
end

# ╔═╡ 75cc90c0-ad26-4a4a-8aa2-96b81a6cf369
function create_action_grid(action_sets, statelist)
#converts action_sets at each state into a square matrix where optimal actions are marked 1 and others are 0
	l = length(statelist)
	output = zeros(l+1, l)
	for i in 1:l
		for j in action_sets[i]
			output[j+1, i] = 1
		end
	end
	return output
end

# ╔═╡ 9a97fd66-1534-41d3-ac85-13542558ffbe
#for plotting purposes take a long list of lines and sample them coarsely 
function formindrange(l, maxind::Int64 = 5, inds::AbstractVector = 1:5)	
	if l < maxind
			return vcat(filter(a -> a < l, inds), l)
		else
			newmax = 5*maxind
			newrange = maxind*2:maxind:newmax
			formindrange(l, newmax, vcat(inds, newrange))
		end
	end

# ╔═╡ 93eada3b-d961-4a27-974a-d81125069c20
function plot_gambler_results(p, θ=eps(0.0))
	mdp = make_gambler_mdp(p)
	statelist = sort(collect(keys(mdp.sa_keys[1])))
	(valuelist, πstar, πraw) = begin_value_iteration_v(mdp, 1.0, mdp.Vinit, θ=θ)
	l = length(valuelist)
	indlist = formindrange(l)	

	value_estimates = mapreduce(hcat, view(valuelist, indlist)) do v
		[v[s] for s in statelist]
	end
	
	p1 = plot(statelist, value_estimates, ylabel = "Value estimates", title = "Gamber's Problem Solution for p = $p", lab = reshape(["sweep $i" for i in indlist], 1, length(indlist)))
	optimal_actions = [argmax(πstar[s]) for s in statelist]
	optimal_action_sets = [multiargmax(πstar[s]) for s in statelist]
	p2 = bar(statelist, optimal_actions, ylabel = "Final policy (stake)")
	p3 = heatmap(create_action_grid(optimal_action_sets, statelist), xlabel = "Capital", ylabel = "Equally Optimal Actions", legend=false, yaxis = [1, 51], yticks = 0:10:100)
	plot(p1, p2, p3, layout=(3, 1), size = (670, 1100), legend = false)
end

# ╔═╡ cb8942ba-2072-4960-ae9a-a2994f9f4ad5
md"""
#### Figure 4.3
"""

# ╔═╡ 1798e5f1-9f4b-4320-8987-e35e946a9bcf
plot_gambler_results(0.4)

# ╔═╡ 04e6f567-31c5-4f05-b5e2-8b46d22dffbc
md"""
> *Exercise 4.8* Why does the optimal policy for the gambler's problem have such a curious form?  In particular, for capital of 50 it bets it all on one flip, but for capital of 51 it does not.  Why is this a good policy?

At capital of 50, it is possible to reach the terminal winning state with a 100% stake.  In the value function estimate we see that this state is valued, as expected, at the probability of receiving a winning flip.  Every capital state larger than 50 has a higher value estimate than this presumably because if we lose a flip we can always try again from the 50 state and otherwise we can more slowly advance up the capital states.  Then again at 75, there is a potentially winning stake of 25.  However, if we lose at the 75 state, we drop to 50 and have another chance to win.  That is why the 75 state will always be valued higher than the 50 state.  Since $p_h$ is less than 50%, if we chose to play it safe and bet less than a winning amount at 50, it is actually most likely that we lose capital progressively and never again reach the 50 state.  Therefore, it makes sense that the moment we reach the 50 state (one flip away from a win), we take the oppotunity to win immediately.  The situation is completely different in a game where the probability of a winning flip is greater than half.  In that case, it would never make sense to risk enough capital to lose in one turn, because we would expect in the long run to accumulate capital slowly.

"""

# ╔═╡ 2f2f6821-8459-4bf9-b0d8-62deffbe5c6b
md"""
> *Exercise 4.9 (programming)* Implement value iteration for the gamber's problem and solve it for $p_h=0.25$ and $p_h = 0.55$.  In programming, you may find it convenient to introduce two dummy states corresponding to termination with capital of 0 and 100, giving them values of 0 and 1 respectively.  Show your results graphically as in Figure 4.3  Are you results stable as $\theta \rightarrow 0$?

See code in the section for Example 4.3, below are plots for the desired p values.  In both cases, as the tolerance is made arbitrarily low the value estimates converge to a stable curve.  For $p_h>0.5$ the curves are smoother as the policy and solution are more predictable.
	"""

# ╔═╡ 64144caf-7b21-41b5-a002-6a86e5119f8b
plot_gambler_results(0.25)

# ╔═╡ d79c93ff-7945-435c-8db1-dfdd6518e34e
plot_gambler_results(0.55)

# ╔═╡ 42e4a3d6-26ef-48bb-9164-118186ec118b
md"""
> *Exercise 4.10* What is the analog of the value iteration update (4.10) for action values, $q_{k+1}(s,a)$?

Copying equation 4.10 we have

$v_{k+1}(s) = \max_a \sum_{s',r} p(s',r|s,a)[r+\gamma v_k(s')]$

To create the equivalent for action values, we need to use the Bellman Optimality Equation for q rather than v

$q_{k+1}(s,a) = \sum_{s',r}p(s',r|s,a)[r + \gamma \max_{a'} q_k(s',a')]$
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[compat]
BenchmarkTools = "~1.3.0"
Plots = "~1.25.8"
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
git-tree-sha1 = "be0cff14ad0059c1da5a017d66f763e6a637de6a"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f9982ef575e19b0e5c7a98c6e75ee496c0f73a93"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.12.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Luxor", "Random"]
git-tree-sha1 = "5b7d2a8b53c68dfdbce545e957a3b5cc4da80b01"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.0"

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
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

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
git-tree-sha1 = "ae13fcbc7ab8f16b0856729b050ef0c446aa3492"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.4+0"

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

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "80ced645013a5dbdc52cf70329399c35ce007fae"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.13.0"

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
git-tree-sha1 = "4a740db447aae0fbeb3ee730de1afbb14ac798a1"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.63.1"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "aa22e1ee9e722f1da183eb33370df4c1aeb6c2cd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.63.1+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

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

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "1c5a84319923bea76fa145d49e93aa4394c73fc2"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.1"

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
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

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
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

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
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

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

[[deps.Librsvg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pango_jll", "Pkg", "gdk_pixbuf_jll"]
git-tree-sha1 = "25d5e6b4eb3558613ace1c67d6a871420bfca527"
uuid = "925c91fb-5dd6-59dd-8e8c-345e74382d89"
version = "2.52.4+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

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
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Luxor]]
deps = ["Base64", "Cairo", "Colors", "Dates", "FFMPEG", "FileIO", "Juno", "LaTeXStrings", "Random", "Requires", "Rsvg"]
git-tree-sha1 = "81a4fd2c618ba952feec85e4236f36c7a5660393"
uuid = "ae8d54c2-7ccd-5906-9d76-62fc9837b5bc"
version = "3.0.0"

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

[[deps.Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

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
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

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
git-tree-sha1 = "648107615c15d4e09f7eca16307bc821c1f718d8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.13+0"

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

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a121dfbba67c94a5bec9dde613c3d0cbcf3a12b"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.50.3+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0b5cfbb704034b5b4c1869e36634438a047df065"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "6f1b25e8ea06279b5689263cc538f51331d7ca17"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.1.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "eb1432ec2b781f70ce2126c277d120554605669a"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.25.8"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

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
git-tree-sha1 = "37c1631cb3cc36a535105e6d5557864c82cd8c2b"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.0"

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

[[deps.Rsvg]]
deps = ["Cairo", "Glib_jll", "Librsvg_jll"]
git-tree-sha1 = "3d3dc66eb46568fb3a5259034bfc752a0eb0c686"
uuid = "c4c386cf-5103-5370-be45-f3a111cca3b8"
version = "1.0.0"

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
git-tree-sha1 = "a635a9333989a094bddc9f940c04c549cd66afcf"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.3.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "d88665adc9bcf45903013af0982e2fd05ae3d0a6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "51383f2d367eb3b444c961d485c565e4c0cf4ba0"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.14"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "d21f2c564b21a202f4677c0fba5b5ee431058544"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.4"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

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
git-tree-sha1 = "66d72dc6fcc86352f01676e8f0f698562e60510f"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.23.0+0"

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

[[deps.gdk_pixbuf_jll]]
deps = ["Artifacts", "Glib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Xorg_libX11_jll", "libpng_jll"]
git-tree-sha1 = "c23323cd30d60941f8c68419a70905d9bdd92808"
uuid = "da03df04-f53b-5353-a52f-6a8b0620ced0"
version = "2.42.6+1"

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
# ╟─4017d910-3635-4ffa-ac1b-919a7bff1e6e
# ╟─55276004-877e-47c0-b5b5-49dbe29aa6f7
# ╠═2de434b0-92f5-4ceb-9d8c-3315cdd7ec5a
# ╠═d46254d0-5ad5-4796-aaa4-a7c135c56da8
# ╠═5f574281-88e0-4d82-bf18-1cfe4c1990fc
# ╠═df9593ca-6d27-45de-9d46-79bddc7a3862
# ╠═59b91c65-3f8a-4015-bb08-d7455623101c
# ╠═5b912508-aa15-470e-be4e-430e88d8a68d
# ╠═e6cdb2be-697d-4191-bd5a-9c129b32246d
# ╠═9253064c-7dfe-445f-b377-fc1acbb6886e
# ╟─7c8c99fd-1b55-494c-898f-194c61e36724
# ╠═c7bdf32a-2f89-4bf8-916b-7558ceedb628
# ╠═6048b106-458e-4e3b-bba9-5f3578458c7c
# ╠═0d0e82e4-b3a4-4528-9288-285fdc5aa8af
# ╠═84068701-40d7-4a5e-93f2-af2a751ab2ec
# ╟─f80580b3-f370-4a02-a9e2-ed791f380521
# ╟─0bebb164-4347-4cff-8169-9f4da4553ae6
# ╠═10c9b166-3a88-460e-82e8-a16c020c1378
# ╠═1618ec46-6e13-42bf-a7f2-68f8dbe3714c
# ╠═e4bfdaca-3f3d-43bb-b8aa-7536adbff662
# ╟─35e1ffe5-d36a-449d-aa73-c618e2855042
# ╟─67b06f3b-13df-4b27-ad80-d112432e8f42
# ╠═f0e5d2e6-3d00-4ffc-962e-e98d4bb28e4e
# ╠═4d15118f-f1ab-4115-bcc9-7f98246eca1c
# ╠═77250f6b-60d1-426f-85b2-497186b86c50
# ╠═c47882c7-ded1-440c-a9a3-0b89a0e7a011
# ╠═0079b02d-8895-4dd4-9557-5f08ac341404
# ╟─0d6936d1-38af-45f1-b496-da49b60f11f8
# ╟─ad1bf1d2-211d-44ca-a258-fc6e112785da
# ╟─e316f59a-8070-4510-96f3-15498897347c
# ╟─5dbfb100-49a8-4f9d-a752-bda4da54699e
# ╠═e722b7e0-63a3-4195-b13e-0449abb3cc39
# ╠═66b7746a-5dd2-4d3b-8026-0c9c10fb5ba3
# ╠═b775fac4-8da9-4a27-a830-934df4b86dc2
# ╠═f0552339-b9ae-4036-8c6a-c232faee2b42
# ╠═b9998e51-f7e5-46c2-80cd-e327911db01b
# ╠═3c874757-2f48-4ba0-93ce-38c019fb1f1b
# ╠═a2e0108a-4bf7-40a9-8c06-dc8403042988
# ╠═1bf3eba7-1a02-4035-962c-73c3fda304bd
# ╠═4d00ffa4-40d5-4dbc-a8e2-5a57cbbcbacd
# ╠═2c4d1304-fa80-4d9f-98f8-5f8d3107110d
# ╠═7c2bd3b2-8388-4e80-b423-41cf2a4c95ef
# ╠═ad5e013f-7938-4e3d-acec-bfce21b63b61
# ╠═5e18e5f1-3d19-4ba7-a7e4-613e6d70a806
# ╠═0a60d83e-3ef6-449f-9131-0c0d0777d413
# ╠═65408c28-eee7-4f60-b71a-cbf31a3c43aa
# ╠═c973a895-07bb-4f8f-8498-4b4773bdd02d
# ╠═48ff63c9-97f4-4162-a009-d05517b8f06f
# ╟─2bedc22e-9615-4fb4-94bf-6a0e7114c417
# ╠═7140970d-d4a7-45bc-9626-26cf1a2f945b
# ╠═f9b3f359-0c24-4a70-9ba8-be185cc01a62
# ╠═39e0e313-79e7-4343-8e02-526f30a66aad
# ╠═a5f1a71f-3c28-49ed-8f29-ff164c4ea02c
# ╠═3d7f4a19-316e-4874-8c28-8e8fe96a9002
# ╠═c2fca344-c1db-4416-8ce4-39eae9e972af
# ╟─315562d0-2bf6-431a-be3f-fb7d2af248b5
# ╠═c456fbdc-0d52-41a9-8e18-dee3c2f4e258
# ╠═8bc5e53f-632d-4ea4-98ca-b3740d98c297
# ╠═ed8c454a-817b-4736-a70d-e2ca4b946ed9
# ╠═75cc90c0-ad26-4a4a-8aa2-96b81a6cf369
# ╠═9a97fd66-1534-41d3-ac85-13542558ffbe
# ╠═93eada3b-d961-4a27-974a-d81125069c20
# ╟─cb8942ba-2072-4960-ae9a-a2994f9f4ad5
# ╠═1798e5f1-9f4b-4320-8987-e35e946a9bcf
# ╟─04e6f567-31c5-4f05-b5e2-8b46d22dffbc
# ╟─2f2f6821-8459-4bf9-b0d8-62deffbe5c6b
# ╠═64144caf-7b21-41b5-a002-6a86e5119f8b
# ╠═d79c93ff-7945-435c-8db1-dfdd6518e34e
# ╟─42e4a3d6-26ef-48bb-9164-118186ec118b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
