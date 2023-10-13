### A Pluto.jl notebook ###
# v0.19.29

using Markdown
using InteractiveUtils

# ╔═╡ f5809bd3-64d4-47ee-9e41-e491f8c09719
begin
	using PlutoPlotly, PlutoUI
	TableOfContents()
end

# ╔═╡ 4017d910-3635-4ffa-ac1b-919a7bff1e6e
md"""
# Chapter 4 

# Dynamic Programming
"""

# ╔═╡ fb497aa8-9d34-48e6-ae85-72df30c1adf3
md"""
Throughout this chapter we explore methods to solve the *Bellman optimality equations*.  Below are the equations for the *state-value function* as well as the *state-action value funtion*:

$\begin{flalign}
	v_*(s) &= \max_a \mathbb{E}[R_{t+1}+\gamma v_*(S_{t+1}) \vert S_t = s, A_t = a] \\
	&= \max_a \sum_{s^\prime,r}p(s^\prime,r\vert s,a)\left [ r + \gamma v_*(s^\prime) \right ] \tag{4.1}
\end{flalign}$

or

$\begin{flalign}
	q_*(s,a) &= \mathbb{E} \left [ R_{t+1}+\gamma \max_{a^\prime} q_*(S_{t+1}, a^\prime) \bigg | S_t = s, A_t = a \right ] \\
	&= \max_a \sum_{s^\prime,r}p(s^\prime,r\vert s,a)\left [ r + \gamma \max_{a^\prime} q_*(s^\prime, a^\prime) \right ] \tag{4.2}
\end{flalign}$

for all $s \in \mathcal{S}, \: a \in \mathcal{A}(s)$, and $s^\prime \in \mathcal{S}^+$
"""

# ╔═╡ 55276004-877e-47c0-b5b5-49dbe29aa6f7
md"""
## 4.1 Policy Evaluation (Prediction)
"""

# ╔═╡ 772d17b0-6fbc-4309-b55b-d17f9b4d3ddf
md"""
The state-value function for an arbitrary policy $\pi$ is defined for all $s \in \mathcal{S}$:

$\begin{flalign}
v_\pi(s) &\doteq \mathbb{E}_\pi [G_t \mid S_t = s] \\
&=\mathbb{E}_\pi [R_{t+1} + \gamma G_{t+1} \mid S_t = s] \tag{from (3.9)} \\ 
&=\mathbb{E}_\pi [R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s] \tag{4.3} \\
&=\sum_a \pi(a\vert s) \sum_{s^\prime,r}p(s^\prime,r \vert s, a) \left [ r + \gamma v_\pi(s^\prime) \right ] \tag{4.4}
\end{flalign}$

where $\pi(a \vert s)$ is the probability of taking action $a$ in state $s$ under policy $\pi$.  The existence and uniqueness of $v_\pi$ are guaranteed as long as either $\gamma < 1$ or eventual termination is guaranteed from all states under the policy $\pi$.

If we have the probability transition function $p(s^\prime, r \vert s, a)$, then (4.4) is a system of $\left | \mathcal{S} \right |$ simultaneous linear equations which can be solved tediously with linear algebra techniques.  However, there is an alternative iterative solution method that makes use of the Bellman equation for $v_\pi$ (4.4), turning it into an update rule.

$\begin{flalign}
v_{k+1}(s) &\doteq \mathbb{E}_\pi [R_{t+1} + \gamma v_k(S_{t+1}) \mid S_t = s] \\
&=\sum_a \pi(a\vert s) \sum_{s^\prime,r}p(s^\prime,r \vert s, a) \left [ r + \gamma v_k(s^\prime) \right ] \tag{4.5}
\end{flalign}$

for all $s \in \mathcal{S}$.  $v_k = v_\pi$ is a fixed point of this iteration because the Bellman equation assures us of equality in this case.  The sequence $\{ v_k \}$ in general converges to $v_\pi$ as $k \rightarrow \infty$ under the same conditions that guarantee the existence of $v_\pi$.  This algorithm is called *iterative policy evaluation*.

Each round of policy evaluation updates every state once.  These are called *expected* updates because they involve directly calculating the expected value using the true probabilities.  To follow this algorithm precisely, we must keep all the values of $v_k$ fixed while we compute $v_{k+1}$ but in practice we can update in place for each state.  As we sweep through the state space then the updates are computed and use new values as soon as they are available.  This method can converge faster than the strict version and requires keeping track of one less item.  Usually when implemented this in-place version of the algorithm is prefered.  Below are examples of code that implements iterative policy evaluation.
"""

# ╔═╡ 4665aa5c-87d1-4359-8cfd-7502d8c5d2e2
md"""
### Iterative Policy Evaluation Implementation
"""

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
### Example 4.1
"""

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

In the second case, the value function at 13 and 15 become coupled because transitions back and forth are allowed.  We can write down new Bellman equations for the equiprobably policy π of these states:

$v_{\pi}(13) = -1 + \frac{1}{4}(v_{\pi}(9) + v_{\pi}(14) + v_{\pi}(12) + v_{\pi}(15))$
$v_{\pi}(15) = -1 + \frac{1}{4}(v_{\pi}(13) + v_{\pi}(14) + v_{\pi}(12) + v_{\pi}(15))$

In the second equation we can simplify to get an equation for state 15 in terms of just 3 others.

$v_{\pi}(15) \times \frac{3}{4} = -1 + \frac{1}{4}(v_{\pi}(13) + v_{\pi}(14) + v_{\pi}(12))$
$v_{\pi}(15) = \frac{1}{3}(-4 + v_{\pi}(13) + v_{\pi}(14) + v_{\pi}(12))$

Let's try to approximate the new value at state 15 by substituting in the known values of the unmodified states.

$v_{\pi}(15) \approx \frac{1}{3}(-4 - 20 - 14 - 22) = \frac{1}{3}(-60) = -20$

Now let's get an implied updated value at state 13 by substituting in the approximate value at 15.

$v_{\pi_{new}}(13) = -1 + \frac{1}{4}(-20 - 14 - 22 - 20) = -1 - \frac{76}{4} = -1 - 19 = -20=v_{\pi_{old}}(13)$

So we assumed that the value at state 13 was unchanged to get the approximation for state 15.  Then using the self consistency equation for state 13 we confirmed that the original value is consistent with the approximate solution.  This step of approximating the value at state 15 with a previous value function is analogous to what we would do in policy evaluation.  However, when checking the value of state 13 we see that it remains unchanged after using state 15.  If we were to carry this out for the other states that depend on 15, we would find that no futher changes are needed since 13 is the only state with a transition to 15 and states 12, 13, and 9 all now have new trasitions to 15 which would have been transitions to 13 previously.  But the value estimate at 15 is identical to the original value at 13.  This is the stopping condition for policy evaluation.  Indeed if we carry out the full policy evaluation calculation below using the same method used to generate Figure 4.1, we see a value of -20 which is equal to the original value of state 13.
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
	(policy_stable, Vstar, [(s, first(keys(πstar[s]))) for s in 0:14])
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

# ╔═╡ 64f22fbb-5c6c-4e8f-bd79-3c9eb19bedca
md"""
# Dependencies and Settings
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoPlotly = "~0.4.1"
PlutoUI = "~0.7.52"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0-beta3"
manifest_format = "2.0"
project_hash = "ea5b8488697e5bd401f7286591a24de33ba7f884"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BaseDirs]]
git-tree-sha1 = "1c9b6f39f40dba0ef22244a175e2d4e42c8f6ee7"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.2.0"

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

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+1"

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
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.0.1+1"

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
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

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
git-tree-sha1 = "9fefc3bfea24f08474e86e86743ee7f8f1bf12a0"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.4.1"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

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

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.0+1"

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
git-tree-sha1 = "b7a5e99f24892b6824a954199a45e9ffcc1c70f0"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.0"

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
# ╟─4017d910-3635-4ffa-ac1b-919a7bff1e6e
# ╟─fb497aa8-9d34-48e6-ae85-72df30c1adf3
# ╟─55276004-877e-47c0-b5b5-49dbe29aa6f7
# ╟─772d17b0-6fbc-4309-b55b-d17f9b4d3ddf
# ╟─4665aa5c-87d1-4359-8cfd-7502d8c5d2e2
# ╠═59b91c65-3f8a-4015-bb08-d7455623101c
# ╠═5b912508-aa15-470e-be4e-430e88d8a68d
# ╠═e6cdb2be-697d-4191-bd5a-9c129b32246d
# ╠═9253064c-7dfe-445f-b377-fc1acbb6886e
# ╟─7c8c99fd-1b55-494c-898f-194c61e36724
# ╠═5f574281-88e0-4d82-bf18-1cfe4c1990fc
# ╠═df9593ca-6d27-45de-9d46-79bddc7a3862
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
# ╠═64f22fbb-5c6c-4e8f-bd79-3c9eb19bedca
# ╠═f5809bd3-64d4-47ee-9e41-e491f8c09719
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
