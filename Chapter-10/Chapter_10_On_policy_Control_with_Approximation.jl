### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 69fb26ed-763e-44ad-9b70-193e5a1a09b9
using PlutoDevMacros, Random, Statistics, LinearAlgebra, Transducers

# ╔═╡ 3f7484b3-272d-410d-92b1-ca13e5d7a8b7
# ╠═╡ show_logs = false
PlutoDevMacros.@frompackage @raw_str(joinpath(@__DIR__, "..", "ApproximationUtils.jl")) using ApproximationUtils

# ╔═╡ 9fb5dace-a799-4424-bcb3-8542e508dd4b
# ╠═╡ skip_as_script = true
#=╠═╡
using PlutoUI,PlutoPlotly, PlutoProfile, BenchmarkTools, LaTeXStrings, HypertextLiteral
  ╠═╡ =#

# ╔═╡ 318b398a-d8f2-4f39-a45d-fd9023961bf8
# ╠═╡ skip_as_script = true
#=╠═╡
include(joinpath(@__DIR__, "..", "Chapter-9", "Chapter_9_On-policy_Prediction_with_Approximation.jl"))
  ╠═╡ =#

# ╔═╡ 35d59eae-77fd-11ef-2790-35dd5a834060
md"""
# Chapter 10: On-policy Control with Approximation
"""

# ╔═╡ b0265b93-ae5f-48f2-a9fd-44fd6115164b
md"""
## Solving the Control Problem

### Goal: Maximizing $G_t$
For a given MDP problem with a state space $\mathcal{S}$ and action space $\mathcal{A}$, there exists some optimal policy $\pi_*$ for which taking actions under that policy will result in a higher expected discounted reward sum than any other policy.  In short:

$\mathbb{E}_{\pi_*}[G_t \vert S_t = s] \ge \mathbb{E}_{\pi}[G_t \vert S_t = s] \quad \forall s,\pi$ where $G_t \doteq \sum_{k=0}^\infty \gamma^k R_{t+k+1} \text{ or } \sum_{k = t+1} ^ T \gamma^{k-t-1}R_k$
"""

# ╔═╡ 47710ddd-79d9-464d-b5dd-27180f2d6b31
md"""
### Defining the Value Function

Since we are interested in maximizing the expected value of $G_t$, it is useful to define functions which calculate this expected value given a state or state action pair
Below is the definition of the two types of value functions as well as derived expressions that are used in solution or approximation techniques.  The Bellman equation as well as other definitions are used to derive all of the useful expressions for the value function.

#### State Value Function
$\begin{flalign}
v_\pi(s) &\doteq \mathbb{E}_\pi [G_t \mid S_t = s] \tag{Used in Monte Carlo Estimation}\\
&= \sum_a \pi(a \vert s) \mathbb{E}_\pi [G_{t} \mid S_t = s, A_t = a] \tag{exp value def} \\
&= \sum_a \pi(a \vert s) q_\pi(s, a) \tag{by definition of q} \\
&= \sum_a \pi(a \vert s) \sum_{s^\prime, r} p(s^\prime, r \vert s, a) [r + \gamma v_\pi(s^\prime)] \tag{Used in Dynamic Programming when p is available} \\
&= \mathbb{E}_\pi [R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s] \tag{Used in TD(0) when p is not available}\\
\end{flalign}$

#### State-Action Value Function
$\begin{flalign}
q_\pi(s, a) &\doteq \mathbb{E}_\pi[G_t \mid S_t=s,A_t=a] \tag{Used in Monte Carlo Estimation} \\
& = \mathbb{E}_\pi \left [ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a \right ] \tag{by (3.9)} \\
& = \sum_{s^\prime, r} p(s^\prime, r \vert s, a) \mathbb{E}_\pi \left [ r + \gamma G_{t+1} \mid S_{t+1} = s^\prime \right ] \tag{exp value def}\\
& = \sum_{s^\prime, r} p(s^\prime, r \vert s, a) [r + \gamma v_\pi(s^\prime)] \tag{by definition of v (4.6)} \\
& = \sum_{s^\prime, r} p(s^\prime, r \vert s, a) [r + \gamma\sum_{a^\prime} \pi(a^\prime \vert s^\prime) q_\pi(s^\prime, a^\prime)] \tag{Used in Dynamic Programming when p is available} \\
& = \mathbb{E} [R_{t+1} + \gamma\sum_{a^\prime} \pi(a^\prime \vert s^\prime) q_\pi(s^\prime, a^\prime) \mid S_t = s, A_t = a] \tag{Used in Expected Sarsa when p is not available} \\
\end{flalign}$

The optimal policy $\pi_*$ will have a value function $v_{\pi_*}(s)$ $q_{\pi_*}(s, a)$ whose values are greather than or equal to any other value function at every state or state action pair.  In short

$v_{\pi_*}(s) \geq v_\pi(s) \forall s \in \mathcal{S} \quad \text{and} \quad q_{\pi_*}(s, a) \geq q_\pi(s, a) \forall s \in \mathcal{S}, \forall a \in \mathcal{A}$ 
"""

# ╔═╡ 4d392303-4681-4ea1-8dcc-e002a78ea0a1
md"""
### Policy Improvement
The purpose of the value function lies in the *Policy Improvement Theorem* which provides a way to iteratively improve a given policy towards the optimal policy.  Consider a policy $\pi(s)$ and its associated value functions $v_\pi(s)$ and $q_\pi(s, a)$.  Now consider a new policy $\pi^\prime(s)$ that would select a different action $a^\prime$ at state $s$ than the original policy $\pi$ such that $q_\pi(s, a^\prime) \geq v_\pi(s)$.  The theorem proves that if the former is true, then $v_{\pi^\prime}(s) \geq v_\pi(s) \forall s \in \mathcal{S}$.  In other words, the new policy is superior to the old one since it has a higher value value at every state.  There is also an easy way to construct a policy which meets the necessary criteria of the policy improvement theorem:

$\begin{flalign}
\pi^\prime(s) &\doteq \mathrm{argmax}_a q_\pi(s, a) \\
& = \mathrm{argmax}_a \mathbb{E} [R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t = a] \\
& = \mathrm{argmax}_a \sum_{s^\prime, r} p(s^\prime, r \vert s, a) [r + \gamma v_\pi(s^\prime)] \\
\end{flalign}$

If we have the state-action value function $q_\pi(s, a)$, then the construction is trivial.  If we only have $v_\pi(s)$, then we must also have the probability transition function for the environment in order to reconstruct the state-action values.  In the absence of $p$ we must rely on $q_\pi$.  Consider some future policy such that $\pi(s) = \mathrm{argmax}_a q_\pi(s, a) \: \forall s$.  In this case, the updated policy will be identical to the original policy and the process will have converged.  The policy at convergence will also be the optimal policy $\pi_*$ whose value functions fulfill the following properties:

$\begin{flalign}
v_*(s) &\doteq \max_\pi v_\pi(s) \: \forall \: s \in \mathcal{S} \tag{3.15} \\
&= \max_{a \in \mathcal{A}(s)} q_{*}(s, a) \: \forall \: s \in \mathcal{S} \tag{meaning of optimal}\\
&= \max_{a \in \mathcal{A}(s)} \sum_{s^\prime, r} p(s^\prime, r \vert s, a) \left [ r + γ v_* (s^\prime) \right ] \quad \forall s \in \mathcal{S} \tag{Used by Value Iteration when p is available}\\
q_*(s, a) &\doteq \max_\pi q_\pi(s, a) \: \forall \: s \in \mathcal{S} \text{ and } a \in \mathcal{A}(s) \tag{3.16} \\
&=\mathbb{E} \left [ R_{t+1} + \gamma v_* (S_{t+1}) \mid S_t = s, A_t = a \right ] \tag{3.17} \\
&= \sum_{s^\prime, r} p(s^\prime, r \vert s, a) \left [ r + γ v_* (s^\prime) \right ] \tag{exp value def (3.21)} \\
&= \sum_{s^\prime, r} p(s^\prime, r \vert s, a) \left [ r + γ \max_{a^\prime} q_*(s^\prime, a^\prime) \right ] \tag{Used by Q Value Iteration when p is available} \\
&= \mathbb{E} \left [ R_t + γ \max_{a^\prime} q_*(s^\prime, a^\prime) \mid S_t = s, A_t = a \right ] \tag{Used by Q Learning when p is not available} \\
\end{flalign}$

The policy improvement theorem suggests an approach where we initialize a random policy, find its value function, and then derive an improved policy until the process has converged.  If we have the probability transition function, we can use dynamic programming to calculate the value function for a policy as well as the improved policy.  Repeating this process until convergence is called *Policy Iteration*.  Alternatively, we can use the properties of the optimal value function without explicitely considering a sequence of policies.  That is only possible when the probability transition function is known and is called *Value Iteration*.
"""

# ╔═╡ a3cf270b-b309-44f0-9972-bd84228bcf17
md"""
### Finding the Optimal Policy
Depending on the nature of the problem and the available information, different techniques are available.  Below is a summary of the conditions and available techniques for different scenarios starting with the ideal case where all information is available.

#### Idealized Tabular Case
##### Problem conditions
- All states and actions can be enumerated: $\mathcal{S} = \{s_1, s_2, \cdots, s_n\}$ and $\mathcal{A} = \{a_1, a_2, \cdots, a_m\}$
-  $p(s^\prime, r \vert s, a)$ is available for all states and actions

In order to verify that we have the correct value function or the optimal one, we need to confirm the conditions for every state or state-action pair.  In short we require $p(s^\prime, r \vert s, a) \: \forall s, a$ which also implies the ability to check all the state action pairs.  If this is possible, then the problem is *tabular* since we can tabulate all of the necessary values.  It is only in this best case scenario that we can definitively verify the correct solution.

##### Solution Techniques
- Value Iteration
  - initialize a list of state values: $[v_1, v_2, \cdots, v_n]$, one for each state
  - perform the following update accross all states until the values converge: $v_i = \max_{a \in \mathcal{A}} \sum_{s_j, r} p(s_j, r \vert s_i, a) \left [ r + γ v_j \right ]$
- Policy Iteration
  - initialize a random policy $\pi$ (probability distribution over actions for each state, could be a matrix)
  - initialize a list of state values: $[v_1, v_2, \cdots, v_n]$, one for each state
  - repeat the following until the values converge
    - use dynamic programming policy evaulation to update the state values
    - update the policy to be greedy with respect to the value function
"""

# ╔═╡ 2f685ee2-6ad8-4bb1-b326-e5de7c15eb18
md"""
#### Sampling Tabular Case
##### Problem conditions
- All states and actions can be enumerated: $\mathcal{S} = \{s_1, s_2, \cdots, s_n\}$ and $\mathcal{A} = \{a_1, a_2, \cdots, a_m\}$
-  $p(s^\prime, r \vert s, a)$ is not available
- Observations from the environment can be collected: $S_t, A_t \implies R_{t+1}, S_{t+1}$

##### Solution Techniques
In the absence of $p$, we cannot use value iteration which is the only technique that circumvents the need for a policy.  So we must use some form of policy iteration.  It turns out that as long as we continue to update both the policy and the value function, neither need to converge to the correct values at any intermediate step.  The idea behind *Generalized Policy Iteration* is to maintain a value estimate for a given policy and update the policy periodically prior to knowing whether or not the value function has converged.  This interval could be as short as a single time step or as it takes to converge.  For episodic tasks, using a single episode of samples to update the value function followed by the policy is a natural choice.  For continuing tasks, one or more steps can be used in place of an episode.  All of the techniques will have the following in common:
- A list of state-action value estimates: $q = [q_{1, 1}, q_{1, 2}, \cdots, q_{1, m}, \cdots q_{n, m}]$, one for each state-action pair initialized to some value
- An initial random policy $\pi$ (probability distribution over actions for each state, could be a matrix)
- State-action values will be updated by averaging together samples which are unbiased estimates of $q_\pi(s, a)$

The primary difference between the techniques will be which equation is used to calculate the sample values.

- Monte Carlo Control: uses samples of the expected value in the definition of $q_\pi(s, a) = \mathbb{E}_\pi [G_t \vert S_t = s, A_t = a]$
  - repeat the following for a set number of episodes
    - Collect a trajectory under the policy $\pi$: $S_0 \overset{\pi}{\rightarrow} A_0 \rightarrow R_1, S_1 \overset{\pi}{\rightarrow} A_1 \rightarrow R_2, S_2 \overset{\pi}{\rightarrow} A_2 \rightarrow \cdots\rightarrow R_T, S_T$
    - Use the reward sequence to compute an unbiased estimate of $\mathbb{E}_\pi[G_t] = \mathbb{E}[q_\pi(S_t, A_t)]$
    - Update the existing value for each state-action pair $S_t, A_t$ observed in the episode using the estimate (can use any number of averaging techniques)
    - Update $\pi$ to select greedy actions with respect to the state-action value estimates and random actions occassionally (this is required to ensure visits to all state-action pairs and convergence of the expected values)

- Sarsa/Expected Sarsa: uses samples of the expected value in the Bellman equation for $q_\pi(s, a) = \mathbb{E} [R_{t+1} + \gamma\sum_{a^\prime} \pi(a^\prime \vert s^\prime) q_\pi(s^\prime, a^\prime) \mid S_t = s, A_t = a]$
  - initialize a state $S_0$
  - repeat the following for a set number of steps
    - Use $\pi$ to select an action $A_t$
    - Sample from the environment $R_{t+1}, S_{t+1}$
    - Calculate an unbiased estimate for $q_\pi(S_t, A_t)$ using $R_{t+1} + \gamma\sum_{a^\prime} \pi(a^\prime \vert s^\prime) q(S_{t+1}, a^\prime)$ OR use $\pi$ to select the next action $A_{t+1}$ and use $R_{t+1} + \gamma q(S_{t+1}, A_{t+1})$
    - Use some averaging method to update the estimated state-action value for $S_t, A_t$
    - Update $\pi$ to select greedy actions with respect to the state-action value estimates and random actions occassionally (this is required to ensure visits to all state-action pairs and convergence of the expected values)

- Q-learning: uses samples of the expected value in the Bellman optimality equation for $q_*(s, a) = \mathbb{E} \left [ R_t + γ \max_{a^\prime} q_*(s^\prime, a^\prime) \mid S_t = s, A_t = a \right ]$
  - initialize a state $S_0$
  - repeat the following for a set number of steps
    - Use $\pi$ to select an action $A_t$
    - Sample from the environment $R_{t+1}, S_{t+1}$
    - Calculate an unbiased estimate for $q_*(S_t, A_t)$ using $R_{t+1} + \gamma \max_{a^\prime} q(S_{t+1}, a^\prime)$
    - Use some averaging method ot update the state-action value for $S_t, A_t$
    - Update $\pi$ to select greedy actions with respect to the state-action value estimates and random actions occassionally (this is required to ensure visits to all state-action pairs and convergence of the expected values)

Note that technically, Monte Carlo Control and Sarsa find the optimal $\epsilon$-greedy policy and value function; however in practice $\epsilon$ can be reduced over time to arbitrarily approach the true optimal policy.  Q-learning, on the other hand can work even if the policy is never updated since the value update does not depend at all on the policy.
"""

# ╔═╡ 14fe2253-cf2c-4159-a360-1e65f1c82b09
md"""
#### Distributional Non-Tabular Case

##### Problem conditions
- All actions can be enumerated: $\mathcal{A} = \{a_1, a_2, \cdots, a_m\}$ but the state space is either infinite or too large to practically count
-  $p(s^\prime, r \vert s, a)$ is available

Previously we considered value estimates in the form of a list.  Since that list is now uncountably large, a different approach, that of approximation, is needed to estimate the value function.  The approximation function must rely on a countable, limited set of information such as a list of parameters $\mathbf{w}$ or a memory of past observations.  Either way, that limited set of information must be used to generalize value estimates accross any state that is encountered.  By construction, this function cannot guarantee that a change to one state value does not affect another, thus the previous goal of optimizing all of the state values is no longer possible.  In order to even define success in this case, a new objective is needed.  Consider the caes of a parameterized function $\hat v(s, \mathbf{w})$ whose goal is to estimate the true value function of a policy $v_\pi(s)$.  One natural objective is to find the parameters that minimize the squared error this function has with the true value function under the distribution of states visited under that policy.  This objective is called the *value error*:

$\overline{\text{VE}}(\mathbf{w}) \doteq \sum_{s \in S} \mu(s)[v_\pi(s) - \hat v(s, \mathbf{w})]^2$

In Chapter 9, we used stochastic gradient descent to derive an update rule for the parameters in the case of knowning the true value function:

$$\begin{flalign}
\mathbf{w}_{t+1} & \doteq \mathbf{w}_t - \frac{1}{2} \alpha \nabla [v_\pi(S_t) - \hat v(S_t, \mathbf{w}_t)]^2 \\
& = \mathbf{w}_t + \alpha[v_\pi(S_t) - \hat v(S_t, \mathbf{w}_t)]\nabla\hat v(S_t, \mathbf{w}_t)
\end{flalign}$$

Since we do not know the true value function, we can sample unbiased estimates of it using the methods already described above.  Then the solution techniques would mirror those in the tabular case with the value updates simply replaced with parameter updates using the same sample estimate.

##### Solution Techniques

Since $p$ is available, we can use state value estimates to compute state-action value estimates.  Consider the estimate Bellman optimality equation for state values:

$v_* =  \max_{a \in \mathcal{A}(s)} \sum_{s^\prime, r} p(s^\prime, r \vert s, a) \left [ r + γ v_* (s^\prime) \right ] \: \forall s \in \mathcal{S}$

This target value can be used in the gradient update as follows:

$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left [ \max_{a \in \mathcal{A}(s)} \left [ \sum_{s^\prime, r} p(s^\prime, r \vert S_t, a) \left [ r + γ \hat v(s^\prime, \mathbf{w}_t) \right ] \right ]- \hat v(S_t, \mathbf{w}_t) \right ]\nabla\hat v(S_t, \mathbf{w}_t)$

while the optimal policy can be derived as:

$\pi(s) = \mathrm{argmax}_a\sum_{s^\prime, r} p(s^\prime, r \vert s, a)( r + \gamma \hat v(s^\prime, \mathbf{w}))$

The derivation of the parameter update rule assumed that the sample estimate for $v_\pi$ did not depend on the parameters.  This assumption is violated here as it was in the semi-gradient TD methods from Chapter 9.  Nevertheless, in the linear case, this technique can converge to some bounded region around the true minimum value error.  Updating the parameters in this way is similar to value iteration which used the same target value and swept across the entire state space.  Using this method here highlights the problem with approximation and its connection to the minimum value error objective.  That objective is only defined in terms of the on policy distribution, so in order for this to converge, the states sampled must match the on-policy distribution which would be the greedy policy in this case.  So while the update rule does not explicitely reference the policy (seemingly implying like in Q-learning that we could perform these updates with any policy), the samples do need to be drawn from the on policy distribution for this to work properly.

#### Sample Non-Tabular Case
##### Problem conditions
- All actions can be enumerated: $\mathcal{A} = \{a_1, a_2, \cdots, a_m\}$ but the state space is either infinite or too large to practically count
-  $p(s^\prime, r \vert s, a)$ is not available

##### Solution Techniques
Finally, we arrive at the case considered at the beginning of Chapter 10 where we must rely on sampling from the environment in the non-tabular case.  Since $p$ is not available, we must instead estimate $\hat q_\pi(s, a)$ for a given policy and proceed with generalized policy iteration as before either with episodic or per step updates to the parameters.  The value error objective can easily be modified to consider all actions as well as states.  Since the actions are enumerable in this case, there is no need to consider the on policy distribution as the error can be minimized across all actions:

$\overline{\text{VE}}(\mathbf{w}) \doteq \sum_a \sum_{s \in S} \mu(s)[q_\pi(s, a) - \hat q(s, a, \mathbf{w})]^2$

The parameter update will look identical to that for state values with the update target replaced by the true state-action value.  The control algorithms will mirror the tabular case with the gradient update replacing the averaging update.  The techniques are defined by which target value is used:

- Semi-gradient Sarsa: $q_\pi(s, a) = \mathbb{E} [R_{t+1} + \gamma\sum_{a^\prime} \pi(a^\prime \vert s^\prime) q_\pi(s^\prime, a^\prime) \mid S_t = s, A_t = a]$
- Gradient Monte Carlo Control: $q_\pi(s, a) = \mathbb{E}_\pi[G_t \vert S_t = s, A_t = a]$
- Semi-gradient Q-learning: $q_*(s, a) = \mathbb{E} \left [ R_t + γ \max_{a^\prime} q_*(s^\prime, a^\prime) \mid S_t = s, A_t = a \right ]$

Note that genearlized policy iteration only works if the policy improvement theorem applies.  In the case of approximation, that is not the case since a policy change cannot be said to only apply to one state.  While many of these techniques can work emphirically, there is no theoretical guarantee that iterating in this manner will produce the optimal policy, even if the approximation is linear.  The only guarantee we can make in that case is that the value function will converge to one that minimizes the value error for the policy at the time.
"""

# ╔═╡ 6351304f-50ac-4755-86e1-cd4680f2d803
md"""
## 10.1 Episodic Semi-gradient Control

It is straightforward to extend the semi-gradient prediction methods in Chapter 9 to action values.  We simply consider examples of the form $S_t, A_t \rightarrow U_t$ where $U_t$ is any of the previously described update targets such as the Monte Carlo Return ($G_t$).  The new gradient-decent update for action-value prediction is:

$\mathbf{w}_{t+1} \doteq \alpha \left [ U_t - \hat q(S_t, A_t, \mathbf{w}_t) \right ] \nabla \hat q(S_t, A_t, \mathbf{w}_t)$

For example, the one-step Sarsa update is:

$\mathbf{w}_{t+1} \doteq \alpha \left [ R_{t+1} + \gamma \hat q(S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat q(S_t, A_t, \mathbf{w}_t) \right ] \nabla \hat q(S_t, A_t, \mathbf{w}_t)$

If the action set is discrete, then at the next state $S_{t+1}$ we can compute $\hat q(S_{t+1}, a, \mathbf{w}_t)$ for every action and then find the greedy action $A^*_{t+1} = \text{argmax}_a\hat q(S_{t+1}, a, \mathbf{w}_t)$.  Policy improvement is then done by changing the estimation policy to a soft approximation of the greedy policy such as the $\epsilon$-greedy policy.  Actions are selected according to this same policy.
"""

# ╔═╡ e7bf61d7-c362-433d-9b83-6537d308c255
md"""
### *Semi-gradient Sarsa Implementation*

Below is an implementation of Semi-gradient Sarsa in a similar style to the algorithms in Chapter 9.  This function updates the provided parameters using the `update_parameters!` function and also requires an `estimate_value` function.  The linear function approximation version of this simplifies the required arguments greatly, needing only a state representation update function.  
"""

# ╔═╡ fc0b88f3-fbf9-450d-b770-b34357ffad49
compute_sarsa_value(action_values::Vector{T}, policy::Vector{T}, i_a::Integer) where T<:Real = action_values[i_a]

# ╔═╡ 991492f4-7dfc-43aa-ab6c-a6b1f3e38225
function semi_gradient_sarsa!(parameters::P, mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, estimate_value::Function, estimate_args::Tuple, update_parameters!::Function, update_args::Tuple; α = one(T)/10, ϵ = one(T) / 10, compute_value = compute_sarsa_value, nn_momentum = false, α_decay = one(T), decay_step = typemax(Int64), kwargs...) where {P, T<:Real}
	s = mdp.initialize_state()
	i_a = rand(eachindex(mdp.actions))
	ep = 1
	step = 1
	epreward = zero(T)
	episode_rewards = Vector{T}()
	episode_steps = Vector{Int64}()
	action_values = zeros(T, length(mdp.actions))
	policy = zeros(T, length(mdp.actions))
	decay = one(T)
	while (ep <= max_episodes) && (step <= max_steps)
		(r, s′) = mdp.ptf(s, i_a)
		epreward += r
		if mdp.isterm(s′)
			U_t = r
			s′ = mdp.initialize_state()
			i_a′ = rand(eachindex(mdp.actions))
			push!(episode_rewards, epreward)
			push!(episode_steps, step)
			epreward = zero(T)
			ep += 1
		else
			estimate_value(action_values, s′, parameters, estimate_args...)
			policy .= action_values
			make_ϵ_greedy_policy!(policy; ϵ = ϵ)
			i_a′ = sample_action(policy)
			q̂ = compute_value(action_values, policy, i_a′)
			U_t = r + γ*q̂
		end
		learning_rate = nn_momentum ? T(1 - 0.999^step) : one(T)
		if step > decay_step
			decay *= α_decay
		end
		update_parameters!(parameters, s, i_a, U_t, α * learning_rate * decay, update_args...)
		s = s′
		i_a = i_a′
		step += 1
	end
	return episode_rewards, episode_steps
end

# ╔═╡ 8b7e1031-9864-439c-86eb-11aa08f53b90
function semi_gradient_double_sarsa!(parameters1::P, parameters2::P, mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, estimate_value::Function, estimate_args::Tuple, update_parameters!::Function, update_args::Tuple; α = one(T)/10, ϵ = one(T) / 10, compute_value = compute_sarsa_value, nn_decay = false, kwargs...) where {P, T<:Real}
	s = mdp.initialize_state()
	i_a = rand(eachindex(mdp.actions))
	ep = 1
	step = 1
	epreward = zero(T)
	episode_rewards = Vector{T}()
	episode_steps = Vector{Int64}()
	action_values = zeros(T, length(mdp.actions))
	policy = zeros(T, length(mdp.actions))
	while (ep <= max_episodes) && (step <= max_steps)
		(r, s′) = mdp.ptf(s, i_a)
		epreward += r
		if rand() < 0.5
			update_params = parameters1
			select_params = parameters2
		end
		if mdp.isterm(s′)
			U_t = r
			s′ = mdp.initialize_state()
			i_a′ = rand(eachindex(mdp.actions))
			push!(episode_rewards, epreward)
			push!(episode_steps, step)
			epreward = zero(T)
			ep += 1
		else
			estimate_value(action_values, s′, select_params, estimate_args...)
			policy .= action_values
			make_ϵ_greedy_policy!(policy; ϵ = ϵ)
			i_a′ = sample_action(policy)
			estimate_value(action_values, s′, update_params, estimate_args...)
			q̂ = compute_value(action_values, policy, i_a′)
			U_t = r + γ*q̂
		end
		learning_rate = nn_decay ? T(1 - 0.999^step) : one(T)
		update_parameters!(update_params, s, i_a, U_t, α * learning_rate, update_args...)
		s = s′
		i_a = i_a′
		step += 1
	end
	return episode_rewards, episode_steps
end

# ╔═╡ 05e2fff5-4871-4468-a00e-9c1b7ba0ffc6
md"""
### Semi-gradient Dynamic Programming

Typically, to solve the control problem we require action-value estimates.  Even in the non-tabular case; however, we may have access to the transition distribution of an MDP.  A simple example of this is a deterministic problem in which there may be an uncountable number of states, but the transition dynamics are known exactly.  In this case, one can reconstruct the action values from the state values as follows:

$\hat q(s, a) = \sum_{s^\prime, r}p(s^\prime, r \vert s, a)\left (r + \gamma \hat v(s^\prime) \right ) = \sum_{s^\prime}p(s^\prime \vert s, a) \left[ r(s^\prime, s, a) + \gamma \hat v(s^\prime) \right ]$

In the case of a deterministic problem there is only one transition state $s^\prime = t(s, a)$ where $t$ is the deterministic mapping function.  Then the formula simplifies to $\hat q(s, a) = r(s, a) + \hat v(t(s, a))$.  We can update the parameters $\mathbf{w}$ for some value function $\hat v(s, \mathbf{w})$ using the techniques in Chapter 9.  Then, to derive the greedy policy, we can use 

$\pi(s) = \text{argmax}_a \left [ r(s, a) + \gamma \sum_{s^\prime}p(s^\prime \vert s, a) \hat v(s^\prime) \right ]$

From tabular dynamic programming, we have the following update rule for the optimal state value function:

$v_*(s) = \max_a \sum_{s^\prime}p(s^\prime \vert s, a) \left[ r(s^\prime, s, a) + \gamma v_*(s^\prime) \right ]$

If the right side expression uses the approximate value function, then it is available to use as an update target instead of the usual Sarsa one.
"""

# ╔═╡ 57a6510f-bd42-4d1d-a550-d1442f79569f
md"""
### *Semi-gradient Dynamic Programming Implementation*
"""

# ╔═╡ b0761704-5447-4e64-8270-708d9dccef60
function semi_gradient_dp!(parameters, mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, max_episodes::Integer, max_steps::Integer, estimate_value::Function, estimate_args::Tuple, update_parameters!::Function, update_args::Tuple; α = one(T)/10, ϵ = one(T) / 10, nn_momentum = false, α_decay = one(T), decay_step = typemax(Int64), kwargs...) where {T<:Real, S, A, P<:StateMDPTransitionDistribution, F1<:Function, F2<:Function, F3<:Function}
	s = mdp.initialize_state()
	i_a = rand(eachindex(mdp.actions))
	ep = 1
	step = 1
	epreward = zero(T)
	episode_rewards = Vector{T}()
	episode_steps = Vector{Int64}()
	action_values = zeros(T, length(mdp.actions))
	policy = zeros(T, length(mdp.actions))
	decay = one(T)
	while (ep <= max_episodes) && (step <= max_steps)
		maxq = typemin(T)
		#computes q and finds maximizing action value, this is effectively trajectory sampling in the case of approximation where we stay close to the optimal policy
		for i_a in eachindex(action_values)
			(rewards, states, probabilities) = mdp.ptf.step(s, i_a)
			v′ = zero(T) 
			r_avg = zero(T)
			for i in eachindex(probabilities)
				v′ += probabilities[i]*(!mdp.isterm(states[i])*estimate_value(states[i], parameters, estimate_args...))
				r_avg += probabilities[i]*rewards[i]
			end
			q = r_avg + γ*v′
			action_values[i_a] = q
			if q > maxq
				maxq = q
			end
		end
		
		learning_rate = nn_momentum ? T(1 - 0.999^step) : one(T)
		update_parameters!(parameters, s, maxq, α * learning_rate * decay, update_args...)

		make_ϵ_greedy_policy!(action_values; ϵ = ϵ)
		i_a = sample_action(action_values)
		(r, s) = mdp.ptf(s, i_a)
		epreward += r
		
		if mdp.isterm(s)
			s = mdp.initialize_state()
			push!(episode_rewards, epreward)
			push!(episode_steps, step)
			epreward = zero(T)
			ep += 1
		end
		
		if step > decay_step
			decay *= α_decay
		end
		step += 1
	end
	return episode_rewards, episode_steps
end

# ╔═╡ 54b92594-04b8-4a8a-82c2-773b4a24680d
md"""
### *Action-Value Implementation of Linear Approximation*

If we update the techniques from Chapter 9 to accomodate action-value estimates, then all of the linear techniques explored there can be used for these control algorihtms.  Previously, we only considered state value estimates, but we can adapt all of those techniques to action values in the linear case quite easily.  The main difference is that the action index needs to be included as an argument to the parameter update function.
"""

# ╔═╡ 278a26ac-c48f-4e18-93bb-706a4634c8c0
md"""
Two options to represent parameters for use with action-values is to simply have a unique set of parameters for each action each of which matches the size of the state representation vector.  Using this technique means we do not need to encode the action space into the representation.  The parameters could be represented by a vector of vectors or a matrix in which each column is assigned to an action.  Below are the update rules for both of these cases.
"""

# ╔═╡ 5f8b0254-88f4-4d19-ade1-8e7c40941b43
begin
	function update_parameters!(parameters::Vector{Vector{T}}, state_representation::SparseVector{T, Int64}, i_a::Integer, α::T, δ::T) where T<:Real
		x = α*δ
		for i in eachindex(state_representation.nzind)
			parameters[i_a][state_representation.nzind[i]] += x .* state_representation.nzval[i]
		end
	end

	function update_parameters!(parameters::Matrix{T}, state_representation::SparseVector{T, Int64}, i_a::Integer, α::T, δ::T) where T<:Real
		x = α*δ
		for i in eachindex(state_representation.nzind)
			parameters[state_representation.nzind[i], i_a] += x .* state_representation.nzval[i]
		end
	end

	update_parameters!(parameters::Vector{Vector{T}}, state_representation::AbstractVector{T}, i_a::Integer, α::T, δ::T) where T<:Real = (parameters[i_a] .+= α .* δ .* state_representation)

	function update_parameters!(parameters::Matrix{T}, state_representation::AbstractVector{T}, i_a::Integer, α::T, δ::T) where T<:Real 
		@inbounds @simd for i in eachindex(state_representation)
			parameters[i, i_a] += α .* δ .* state_representation[i]
		end
	end
end

# ╔═╡ c5c839f7-1806-463d-b63a-bd7e1384f203
md"""
The action value calculation also depends on how parameters are represented.  Either a dot product is used by extracting the appropriate parameter vector, or the parameter matrix needs to be iterated over the appropriate column
"""

# ╔═╡ f11787a1-57f8-4077-8d60-bc760ece7cc6
begin
	calculate_action_value(state_representation::AbstractVector{T}, i_a::Integer, parameters::Vector{Vector{T}}) where T<:Real = dot(state_representation, parameters[i_a])

	function calculate_action_value(state_representation::AbstractVector{T}, i_a::Integer, parameters::Matrix{T}) where T<:Real 
		q = zero(T)
		@inbounds @simd for i in eachindex(state_representation)
			q += state_representation[i]*parameters[i, i_a]
		end
		return q
	end

	function calculate_action_value(state_representation::SparseVector{T, Int64}, i_a::Integer, parameters::Matrix{T}) where T<:Real 
		q = zero(T)
		@inbounds @simd for i in state_representation.nzind
			q += state_representation[i]*parameters[i, i_a]
		end
		return q
	end
end	

# ╔═╡ 1478745a-634d-4f31-8a70-b74f0e536201
md"""
Part of the Sarsa algorithm requires us to identify the maximizing action.  These functions update a vector of action-values using the parameters and state representation.  By computing all of the action values at once, the maximum can be identified, and these functions compute all the action values more efficiently than using the above function for each action individually.
"""

# ╔═╡ c697e0b6-d3e4-4f5f-96e9-b9486c9d7efc
begin
	fill_action_values!(action_values::Vector{T}, state_representation::AbstractVector{T}, parameters::Matrix{T}) where T<:Real = mul!(action_values, parameters', state_representation)

	function fill_action_values!(action_values::Vector{T}, state_representation::AbstractVector{T}, parameters::Vector{Vector{T}}) where T<:Real 
		@inbounds for i in eachindex(action_values)
			action_values[i] = dot(state_representation, parameters[i])
		end
	end
end

# ╔═╡ dc2cffeb-9adf-4956-afa3-ac82af377c59
md"""
Finally we can create the action-value function and parameter update for the generic linear case.  If a vector of action values is provided as the first argument to the value function, that vector will be updated with all of the action values for a given state.
"""

# ╔═╡ 56b0d69b-b7c3-4365-9b02-e0d5e8a85f94
function run_linear_semi_gradient_dp(mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, state_representation::AbstractVector{T}, update_state_representation!::Function; setup_kwargs = NamedTuple(), kwargs...) where T<:Real
	setup = linear_features_gradient_setup(mdp, state_representation, update_state_representation!; setup_kwargs...)
	l = length(state_representation)
	num_actions = length(mdp.actions)
	parameters = zeros(T, l)
	episode_rewards, episode_steps = semi_gradient_dp!(parameters, mdp, γ, max_episodes, max_steps, setup.value_function, setup.value_args, setup.parameter_update, setup.update_args; kwargs...)
	v̂(s) = setup.value_function(s, parameters, setup.value_args...)
	function π_greedy(s)
		action_values = zeros(T, num_actions)
		for i_a in eachindex(action_values)
			(rewards, states, probabilities) = mdp.ptf.step(s, i_a)
			q = zero(T) 
			for i in eachindex(probabilities)
				v̂′ = !mdp.isterm(states[i])*v̂(states[i])
				q += probabilities[i]*(rewards[i] + γ*v̂′)
			end
			action_values[i_a] = q
		end
		make_greedy_policy!(action_values)
		i_a = sample_action(action_values)
	end
	return (value_function = v̂, π_greedy = π_greedy, reward_history = episode_rewards, step_history = episode_steps)
end

# ╔═╡ 8d096d0d-8fea-421a-aa33-82269d3fe7e2
md"""
### *Action-Value Implementation of Non-Linear Approximation*
"""

# ╔═╡ f58cd0a2-8c82-46b2-bb8f-00f6aa1d867f
function update_input!(input::Matrix{Float32}, feature_vector::Vector{Float32}, i_a::Integer, num::Integer)
	l1 = length(feature_vector)
	l2 = size(input, 2)
	for i in eachindex(feature_vector)
		input[num, i] = feature_vector[i]
	end
	for i in l1+1:l2
		input[num, i] = 0f0
	end
	input[num, l1+i_a] = 1f0
end

# ╔═╡ c678846c-aaff-4266-8b2c-07a3a92445ef
function fcann_action_gradient_setup(mdp::StateMDP, layers::Vector{Int64}, feature_vector::Vector{Float32}, update_feature_vector!::Function; kwargs...)
	s0 = mdp.initialize_state()
	num_actions = length(mdp.actions)
	input_layer_size = length(feature_vector)
	update_feature_vector!(feature_vector, s0)
	θ, β = FCANN.initializeparams_saxe(input_layer_size, layers, num_actions, 1; use_μP = true)

	∇θ = deepcopy(θ)
	∇β = deepcopy(β)
	∇tanh_z = FCANN.form_tanh_grads(layers, 1)

	function setup_training(batch_size::Integer)
		activations = [zeros(Float32, batch_size, l) for l in [layers; num_actions]]
		δs = deepcopy(activations)
		onesvec = zeros(Float32, batch_size)
		return (activations, δs, onesvec)
	end

	(activations, δs, onesvec) = setup_training(1)

	input = zeros(Float32, 1, input_layer_size)
	output = zeros(Float32, 1, num_actions)
	scales = ones(Float32, length(layers)+1)
	for i in 2:length(scales)
		scales[i] /= size(θ[i], 2)
	end
	mT = deepcopy(θ)
	vT = deepcopy(θ)
	# θ_est = deepcopy(θ)
	# θ_avg = deepcopy(θ)
	mB = deepcopy(β)
	vB = deepcopy(β)
	# β_est = deepcopy(β)
	# β_avg = deepcopy(β)
	FCANN.zeroParams!(mT, vT)
	FCANN.zeroParams!(vT, vB)
	# function update_parameters!(parameters, s, i_a::Integer, g::Float32, α::Float32, gradients, state_representation::Vector{Float32}, input, output, ∇tanh_z, activations, δs, onesvec, scales)
	# 	update_feature_vector!(state_representation, s)
	# 	NonTabularRL.update_input!(input, state_representation, 1)
	# 	FCANN.predict!(parameters[1], parameters[2], input, activations, 1)
	# 	for i in 1:num_actions
	# 		output[1, i] = activations[end][1, i]
	# 	end
	# 	output[1, i_a] = g
	# 	NonTabularRL.update_nn_parameters!(parameters[1], parameters[2], layers, gradients[1], gradients[2], input, output, ∇tanh_z, activations, δs, onesvec, α, scales; kwargs...)
	# end

	function update_parameters!(parameters, s, i_a::Integer, g::Float32, α::Float32, gradients, state_representation::Vector{Float32}, input, output, ∇tanh_z, activations, δs, onesvec, scales, mT, mB, vT, vB)
		update_feature_vector!(state_representation, s)
		update_input!(input, state_representation, 1)
		for i in 1:num_actions
			output[1, i] = activations[end][1, i]
		end
		output[1, i_a] = g
		update_nn_parameters!(parameters[1], parameters[2], layers, gradients[1], gradients[2], input, output, ∇tanh_z, activations, δs, onesvec, α, scales, mT, mB, vT, vB, 10000)
	end

	function q̂(s, i_a::Integer, parameters, state_representation, input, activations) 
		update_feature_vector!(state_representation, s)
		update_input!(input, state_representation, 1)
		FCANN.predict!(parameters[1], parameters[2], input, activations, 1)
		return activations[end][1, i_a]
	end

	function q̂(action_values::Vector{Float32}, s, parameters, state_representation, input, activations) 
		update_feature_vector!(state_representation, s)
		update_input!(input, state_representation, 1)
		FCANN.predict!(parameters[1], parameters[2], input, activations, 1)
		best_action = 1
		best_value = typemin(Float32)
		for i_a in eachindex(action_values)
			q = activations[end][1, i_a]
			newmax = q > best_value
			best_action = !newmax*best_action + newmax*i_a
			best_value = !newmax*best_value + newmax*q
			action_values[i_a] = q
		end
		return (best_value, best_action)
	end

	# update_args = ((∇θ, ∇β), feature_vector, input, output, ∇tanh_z, activations, δs, onesvec, scales)
	update_args = ((∇θ, ∇β), feature_vector, input, output, ∇tanh_z, activations, δs, onesvec, scales, mT, mB, vT, vB)
	
	return (value_function = q̂, value_args = (feature_vector, input, activations), parameter_update = update_parameters!, update_args = update_args, parameters = (θ, β))
end

# ╔═╡ 39b0f100-a7e7-4633-b296-ccc87a5a35ab
function run_fcann_semi_gradient_sarsa(mdp::StateMDP, γ::Float32, max_episodes::Integer, max_steps::Integer, state_representation::Vector{Float32}, update_state_representation!::Function, layers::Vector{T}; λ = 0f0, c = Inf, dropout = 0f0, kwargs...) where T<:Integer
	setup = fcann_action_gradient_setup(mdp, layers, state_representation, update_state_representation!; λ = λ, c = c, dropout = dropout)
	num_actions = length(mdp.actions)
	episode_rewards, episode_steps = semi_gradient_sarsa!(setup.parameters, mdp, γ, max_episodes, max_steps, setup.value_function, setup.value_args, setup.parameter_update, setup.update_args; nn_momentum = true, kwargs...)
	q̂(s, i_a) = setup.value_function(s, i_a, setup.parameters, setup.value_args...)
	action_values = zeros(Float32, num_actions)
	q̂(s) = setup.value_function(action_values, s, setup.parameters, setup.value_args...)
	π_greedy(s) = q̂(s)[2]
	return (value_function = q̂, π_greedy = π_greedy, reward_history = episode_rewards, step_history = episode_steps, parameters = setup.parameters)
end

# ╔═╡ 00e7783f-7f17-4944-a085-ea87509cd75a
function run_fcann_semi_gradient_dp(mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, state_representation::AbstractVector{T}, update_state_representation!::Function, layers::Vector{Int64}; λ = 0f0, c = Inf, dropout = 0f0, kwargs...) where T<:Real
	setup = fcann_gradient_setup(mdp, layers, state_representation, update_state_representation!; λ = λ, c = c, dropout = dropout)
	l = length(state_representation)
	num_actions = length(mdp.actions)
	episode_rewards, episode_steps = semi_gradient_dp!(setup.parameters, mdp, γ, max_episodes, max_steps, setup.value_function, setup.value_args, setup.parameter_update, setup.update_args; kwargs...)
	v̂(s) = setup.value_function(s, setup.parameters, setup.value_args...)
	function π_greedy(s)
		action_values = zeros(T, num_actions)
		for i_a in eachindex(action_values)
			(rewards, states, probabilities) = mdp.ptf.step(s, i_a)
			q = zero(T) 
			for i in eachindex(probabilities)
				v̂′ = !mdp.isterm(states[i])*v̂(states[i])
				q += probabilities[i]*(rewards[i] + γ*v̂′)
			end
			action_values[i_a] = q
		end
		make_greedy_policy!(action_values)
		i_a = sample_action(action_values)
	end
	return (value_function = v̂, π_greedy = π_greedy, reward_history = episode_rewards, step_history = episode_steps)
end

# ╔═╡ a22e5d34-4b8d-479c-985c-d6abd41a6c80
md"""
### Example 10.1: Mountain Car Task
"""

# ╔═╡ b990ba67-42c8-4ab9-943d-085392204fdd
md"""
#### Defining Car State and Dynamics
"""

# ╔═╡ cafb20b4-a2bd-46a9-9660-b0ace84d6e4c
function initialize_car_state()
	a = rand(Float32) * 0.2f0
	x = a - 0.6f0
	ẋ = 0f0
	(x, ẋ)
end

# ╔═╡ d577b393-4b40-4c90-9993-4ffbcbd9df6d
const mountain_car_actions = [-1f0, 0f0, 1f0]

# ╔═╡ b07460f1-0461-4f63-b145-c4e1818a497e
function mountain_car_step(s::Tuple{Float32, Float32}, i_a::Int64)
	a = mountain_car_actions[i_a]
	ẋ′ = clamp(s[2] + 0.001f0*a - 0.0025f0*cos(3*s[1]), -0.07f0, 0.07f0)
	x′ = clamp(s[1] + ẋ′, -1.2f0, 0.5f0)
	x′ == -1.2f0 && return (-1f0, (x′, 0f0))
	return (-1f0, (x′, ẋ′))
end

# ╔═╡ df07524f-b3fe-4a66-98ac-8f80df66bcff
function mountain_car_dist_step(s::Tuple{Float32, Float32}, i_a::Int64)
	(r, s′) = mountain_car_step(s, i_a)
	([r], [s′], [1f0])
end

# ╔═╡ 28e0d632-0df3-4a5b-85c4-571c845ff827
const mountain_car_action_names = ["Decelerate", "Nothing", "Accelerate"]

# ╔═╡ 8befede5-378a-447a-96bd-edcd9d2ce98b
md"""
We can use these to create a sampling transition function, although it will be deterministic.  The positions and velocities are still defined by two real numbers so the state space is unbounded and we cannot use a tabular method.
"""

# ╔═╡ ac80958a-73ec-4342-b553-b33df6612a50
const mountain_car_transition = StateMDPTransitionSampler(mountain_car_step, initialize_car_state())

# ╔═╡ f9abf433-8ce3-4520-9b47-f17f8f07b4cb
const mountain_car_transition_distribution = StateMDPTransitionDistribution(mountain_car_dist_step, initialize_car_state())

# ╔═╡ 1e9c537a-a731-4b81-8f6a-cb658b52c5be
# ╠═╡ skip_as_script = true
#=╠═╡
const mountain_car_mdp = StateMDP(mountain_car_actions, mountain_car_transition, initialize_car_state, s -> s[1] == 0.5f0)
  ╠═╡ =#

# ╔═╡ 5b2ffd90-ead0-42ce-999a-584ed8995910
# ╠═╡ skip_as_script = true
#=╠═╡
const mountain_car_dist_mdp = StateMDP(mountain_car_actions, mountain_car_transition_distribution, initialize_car_state, s -> s[1] == 0.5f0)
  ╠═╡ =#

# ╔═╡ f6e08689-040f-4565-9dfb-e9a65d1c1f18
md"""
#### Visualizing Trajectories
"""

# ╔═╡ 528533f7-68f1-4d19-9a37-6d4d0d7c38e2
# ╠═╡ skip_as_script = true
#=╠═╡
@bind constant_params PlutoUI.combine() do Child
	md"""
	Number of Steps: $(Child(:nsteps, Slider(1:1000, default = 200, show_value=true)))
	
	Select Constant Action: $(Child(:action, Select([1 => "Decelerate", 2 => "Nothing", 3 => "Accelerate"])))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ afee7bc9-aff0-4c71-a227-9845cb23d4e9
# ╠═╡ skip_as_script = true
#=╠═╡
md"""Number of Steps: $(@bind rand_nsteps confirm(Slider(1:1_000, default = 200, show_value=true)))"""
  ╠═╡ =#

# ╔═╡ cc9197e0-f5bd-4742-bea3-b54e0b8e3b93
#=╠═╡
function show_mountaincar_trajectory(π::Function, max_steps::Integer, name)
	states, actions, rewards, sterm, nsteps = runepisode(mountain_car_mdp; π = π, max_steps = max_steps)
	positions = [s[1] for s in states]
	velocities = [s[2] for s in states]
	tr1 = scatter(x = positions, y = velocities, mode = "markers", showlegend = false)
	tr2 = scatter(y = positions, showlegend = false)
	tr3 = scatter(y = [mountain_car_actions[i] for i in actions], showlegend = false)
	p1 = plot(tr1, Layout(xaxis_title = "position", yaxis_title = "velocity", xaxis_range = [-1.2, 0.5], yaxis_range = [-0.07, 0.07]))
	p2 = plot(tr2, Layout(xaxis_title = "time", yaxis_title = "position"))
	p3 = plot(tr3, Layout(xaxis_title = "time", yaxis_title = "action"))
	mdname = Markdown.parse(name)
	md"""
	$mdname
	Total Reward: $(sum(rewards))
	$([p1 p2 p3])
	"""
end
  ╠═╡ =#

# ╔═╡ ca970333-fa08-412c-b89d-491e70f0ac79
md"""
#### Typical Episode Length under Random Policy
"""

# ╔═╡ e86bc86f-9909-458d-b86d-0a4ac4b9d43d
# ╠═╡ skip_as_script = true
#=╠═╡
@bind nsamples NumberField(1:100_000, default = 1000) |> confirm
  ╠═╡ =#

# ╔═╡ b5273dfa-2262-487a-856b-441f007bd163
# ╠═╡ skip_as_script = true
#=╠═╡
(1:nsamples |> Map(i -> runepisode(mountain_car_mdp; max_steps = 100_000)[5] == 100_000) |> foldxt(+)) / nsamples
  ╠═╡ =#

# ╔═╡ dae59fd9-0397-4307-afd8-bafb6f0bfa52
# ╠═╡ skip_as_script = true
#=╠═╡
(1:nsamples |> Map(i -> runepisode(mountain_car_mdp; max_steps = 100_000)[5]) |> foldxt(+)) / nsamples
  ╠═╡ =#

# ╔═╡ d291541d-ddba-4b71-a4eb-37fef758b71b
md"""
#### Tabular Version of Mountain Car

If we discretize the positions and velocities then we can transform this into a tabular problem.  The number of states will be NxM where N and M are the number of distinct values for position and velocity respectively.  As N and M approach infinity this problem will approach the original MDP, so we can study the limiting behavior of the optimal policy and value function using tabular methods that are guaranteed to converge.
"""

# ╔═╡ 12f5065b-5bed-4d03-a0f0-72a942492394
function make_tabular_mountaincar(N, M)
	x_range = (-1.2f0, 0.5f0)
	v_range = (-0.07f0, 0.07f0)
	x_vals = LinRange(x_range..., N) 
	v_vals = LinRange(v_range..., M) 
	states = [(x, v) for x in x_vals for v in v_vals]
	state_transition_map = zeros(Int64, length(mountain_car_actions), length(states))
	reward_transition_map = zeros(Float32, length(mountain_car_actions), length(states))

	#assign a state to the closest state in the list by euclidean distance
	d(x1, x2) = (x1 - x2)^2
	function bucket_state(s1)
		i_x = searchsortedfirst(x_vals, s1[1])
		i_v = searchsortedfirst(v_vals, s1[2])
		M*(i_x-1) + i_v
	end
	
	for (i_s, s) in enumerate(states)
		if s[1] == 0.5f0
			state_transition_map[:, i_s] .= i_s
			reward_transition_map[:, i_s] .= 0f0
		else
			for (i_a, a) in enumerate(mountain_car_actions)
				(r, s′) = mountain_car_step(s, i_a)
				i_s′ = bucket_state(s′)
				state_transition_map[i_a, i_s] = i_s′
				reward_transition_map[i_a, i_s] = r
			end
		end
	end

	init_state_index() = bucket_state(initialize_car_state())
	ptf = TabularDeterministicTransition(state_transition_map, reward_transition_map)
	(mdp = TabularMDP(states, mountain_car_actions, ptf, init_state_index), assign_state_index = bucket_state)
end

# ╔═╡ 39c63495-36c3-4e62-b8fb-36865f2c6243
md"""
##### Visualizing Policies in Tabular Mountain Car
"""

# ╔═╡ 33ea5f09-3a1f-476d-875a-1f3635a40295
#=╠═╡
@bind tabular_mountaincar_args PlutoUI.combine() do Child
	md"""
	Policy View Selection: $(Child(:policy, Select([1 => "Random", 2 => "Accelerate Only"])))
	Number of Positions: $(Child(:num_positions, NumberField(1:10000, default = 500)))
	Number of Velocities: $(Child(:num_velocities, NumberField(1:10000, default = 500)))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ 1d417a66-205f-4883-b49c-a6fc900af4ce
#=╠═╡
const mountaincar_positions = LinRange(-1.2f0, 0.5f0, tabular_mountaincar_args.num_positions) 
  ╠═╡ =#

# ╔═╡ 7e8c89aa-8a5e-4ff4-afd2-df8f5c77b5b2
#=╠═╡
const mountaincar_velocities = LinRange(-0.07f0, 0.07f0, tabular_mountaincar_args.num_velocities) 
  ╠═╡ =#

# ╔═╡ 9d65285f-d49e-40ce-acea-1f565bcd4108
#=╠═╡
const (tabular_mountaincar_mdp, assign_state_index_tabular_mountaincar) = make_tabular_mountaincar(tabular_mountaincar_args.num_positions, tabular_mountaincar_args.num_velocities)
  ╠═╡ =#

# ╔═╡ 1b15efa9-c331-46bf-93db-f96dee026fe2
#=╠═╡
const tabular_mountaincar_πrand = make_random_policy(tabular_mountaincar_mdp)
  ╠═╡ =#

# ╔═╡ e338be2b-05f1-43f4-a194-45ffd710777e
#=╠═╡
const accelerate_mountaincar_π = begin
	out = zeros(Float32, 3, length(tabular_mountaincar_mdp.states))
	out[3, :] .= 1f0
	out
end
  ╠═╡ =#

# ╔═╡ e48af9f4-0b47-4a45-b0ad-8f53b094e712
#=╠═╡
const tabular_policies = [tabular_mountaincar_πrand, accelerate_mountaincar_π]
  ╠═╡ =#

# ╔═╡ 72f575ee-d656-4af6-bf78-aab42bf1debd
md"""
##### Solving Mountain Car Tabular Problem with Value Iteration
"""

# ╔═╡ 57ea3538-33be-4673-b914-8191d35426a9
#=╠═╡
const mountaincar_value_iteration = value_iteration_v(tabular_mountaincar_mdp, 1f0; save_history = false)
  ╠═╡ =#

# ╔═╡ 57659c52-de1b-46e6-a863-8eeec0cee601
#=╠═╡
π_optimal_value_iteration(s) = sample_action(view(mountaincar_value_iteration.optimal_policy, :, assign_state_index_tabular_mountaincar(s)))
  ╠═╡ =#

# ╔═╡ be77b538-d106-4ca0-a974-289415588c47
md"""
##### Solving Mountain Car Tabular Problem with Policy Iteration
"""

# ╔═╡ 3c300a2b-4139-4df0-906b-4cae3592cc2b
# ╠═╡ skip_as_script = true
#=╠═╡
const mountaincar_policy_iteration = policy_iteration_v(tabular_mountaincar_mdp, 1f0; θ = 0.001f0, max_iterations = 2)
  ╠═╡ =#

# ╔═╡ e2cd69c5-eda7-4897-9e64-0adf940d4d96
#=╠═╡
@bind policy_num Select(eachindex(mountaincar_policy_iteration[1]))
  ╠═╡ =#

# ╔═╡ 5bc2eda5-5f4c-4165-9afb-16920f30b0c5
#=╠═╡
π_optimal_policy_iteration(s) = sample_action(view(mountaincar_policy_iteration[1][end], :, assign_state_index_tabular_mountaincar(s)))
  ╠═╡ =#

# ╔═╡ 8a5d9e3d-e8ef-4cea-8cd8-6975f797d7bd
# ╠═╡ skip_as_script = true
#=╠═╡
function plot_mountaincar_action_values(tabular_mountaincar_mdp, value_function::Matrix{Float32}, π::Matrix{Float32}, mountaincar_positions, mountaincar_velocities)
	n = 100
	num_positions = length(mountaincar_positions)
	num_velocities = length(mountaincar_velocities)
	values = [zeros(Float32, num_positions, num_velocities) for _ in 1:size(π, 1)]
	actions = zeros(Float32, num_positions, num_velocities)
	for (i_x, x) in enumerate(mountaincar_positions)
		for (i_v, v) in enumerate(mountaincar_velocities)
			value_index = tabular_mountaincar_mdp.state_index[(x, v)]
			(pmax, i_amax) = findmax(value_function[i, value_index] for i in 1:3)
			actions[i_v, i_x] = i_amax
			# pmax ≈ 1f0/3 ? actions[i_x, i_v] = 2 : actions[i_x, i_v] = i_amax
			for i_a in 1:size(value_function, 1)
				values[i_a][i_v, i_x] = value_function[i_a, value_index]
			end
		end
	end
			
	p1 = [plot(heatmap(x = mountaincar_positions, y = mountaincar_velocities, z = values[i]) , Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Learned Value Function")) for i in eachindex(values)]
	p2 = plot(heatmap(x = mountaincar_positions, y = mountaincar_velocities, z = actions, colorscale = "rb", showscale = false), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Greedy Policy (blue = accelerate left, <br>red = accelerate right, gray = no acceleration)"))
	# [p1; p2]
	# $(HTML(reduce(add_elements, p1)))
	@htl("""
	<div style = "display: flex;">
	
	$(reduce(hcat, p1))
	</div>
	$p2
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ d8d5db17-d89c-47db-b258-6ad1635478b7
# ╠═╡ skip_as_script = true
#=╠═╡
function plot_mountaincar_action_values(tabular_mountaincar_mdp, value_function::Vector{Float32}, π::Matrix{Float32}, mountaincar_positions, mountaincar_velocities)
	n = 100
	num_positions = length(mountaincar_positions)
	num_velocities = length(mountaincar_velocities)
	values = zeros(Float32, num_positions, num_velocities)
	actions = zeros(Float32, num_positions, num_velocities)
	for (i_x, x) in enumerate(mountaincar_positions)
		for (i_v, v) in enumerate(mountaincar_velocities)
			value_index = tabular_mountaincar_mdp.state_index[(x, v)]
			values[i_v, i_x] = value_function[value_index]
			(pmax, i_amax) = findmax(π[i, value_index] for i in 1:3)
			pmax ≈ 1f0/3 ? actions[i_v, i_x] = 2 : actions[i_v, i_x] = i_amax
		end
	end
			
	p1 = plot(heatmap(x = mountaincar_positions, y = mountaincar_velocities, z = values), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Learned Value Function"))
	p2 = plot(heatmap(x = mountaincar_positions, y = mountaincar_velocities, z = actions, colorscale = "rb", showscale = false), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Policy (blue = accelerate left, <br>red = accelerate right, gray = no acceleration)"))
	[p1 p2]
end
  ╠═╡ =#

# ╔═╡ 1054cfa3-9f58-4a93-a318-c2d21cf23220
# ╠═╡ skip_as_script = true
#=╠═╡
function show_mountaincar_trajectory(tabular_mountaincar_mdp, π::Matrix{T}, max_steps::Integer, name) where T<:Real
	state_indices, actions, rewards, sterm, nsteps = runepisode(tabular_mountaincar_mdp; π = π, max_steps = max_steps)
	states = [tabular_mountaincar_mdp.states[i_s] for i_s in state_indices]
	positions = [s[1] for s in states]
	velocities = [s[2] for s in states]
	tr1 = scatter(x = positions, y = velocities, mode = "markers", showlegend = false)
	tr2 = scatter(y = positions, showlegend = false)
	tr3 = scatter(y = [mountain_car_actions[i] for i in actions], showlegend = false)
	p1 = plot(tr1, Layout(xaxis_title = "position", yaxis_title = "velocity", xaxis_range = [-1.2, 0.5], yaxis_range = [-0.07, 0.07]))
	p2 = plot(tr2, Layout(xaxis_title = "time", yaxis_title = "position"))
	p3 = plot(tr3, Layout(xaxis_title = "time", yaxis_title = "action"))
	mdname = Markdown.parse(name)
	md"""
	$mdname
	Total Reward: $(sum(rewards))
	$([p1 p2 p3])
	"""
end
  ╠═╡ =#

# ╔═╡ d42bb733-07e2-4932-aab4-09229ff67492
#=╠═╡
show_mountaincar_trajectory(s -> constant_params.action, constant_params.nsteps, "Mountain Car Trajectory for $(mountain_car_action_names[constant_params.action]) only Policy")
  ╠═╡ =#

# ╔═╡ 864450b9-1319-4426-961f-ee6df93463d8
#=╠═╡
show_mountaincar_trajectory(s -> rand(1:3), rand_nsteps, "Mountain Car Trajectory for Random Policy")
  ╠═╡ =#

# ╔═╡ 99e3ec39-24f0-43d6-b6fd-9910b738ce2c
#=╠═╡
show_mountaincar_trajectory(tabular_mountaincar_mdp, tabular_policies[tabular_mountaincar_args.policy], 1000, "Tabular Mountain Car Trajectory")
  ╠═╡ =#

# ╔═╡ a97e3b12-b7a5-4f88-bdb9-c3158203e0ff
#=╠═╡
show_mountaincar_trajectory(tabular_mountaincar_mdp, mountaincar_value_iteration.optimal_policy, 1000, "Tabular Mountain Car Value Iteration Policy")
  ╠═╡ =#

# ╔═╡ cbf1e5ed-8308-486e-a9b7-6cf7fb441fe3
#=╠═╡
show_mountaincar_trajectory(π_optimal_value_iteration, 1000, "Tabular Mountain Car Value Iteration Policy on True MDP")
  ╠═╡ =#

# ╔═╡ 66d6a4b0-ddf8-4781-b3b4-20f02b25199a
#=╠═╡
show_mountaincar_trajectory(tabular_mountaincar_mdp, mountaincar_policy_iteration[1][policy_num], 500, "Policy Iteration Number $policy_num")
  ╠═╡ =#

# ╔═╡ 78087a57-33a0-4581-81de-926476090931
#=╠═╡
show_mountaincar_trajectory(s -> sample_action(view(mountaincar_policy_iteration[1][policy_num], :, assign_state_index_tabular_mountaincar(s))), 500, "Policy Iteration Number $policy_num on True MDP")
  ╠═╡ =#

# ╔═╡ 1a5acfb0-3b35-41b1-98f8-ffce941c587f
md"""
#### Mountain Car Linear Approximation with Tile Coding
"""

# ╔═╡ 742100ba-c38e-4840-8988-40990039b527
# ╠═╡ skip_as_script = true
#=╠═╡
setup_mountain_car_tiles(tile_size::NTuple{2, Float32}, num_tilings::Integer) = tile_coding_setup(mountain_car_mdp, (-1.2f0, 0.5f0), (-0.07f0, 0.07f0), tile_size, num_tilings, (1, 3))
  ╠═╡ =#

# ╔═╡ af97f222-08d1-4200-a10b-8da178182175
md"""
#### Dynamic Programming
"""

# ╔═╡ 224b4bec-9ec5-434d-a950-f5974cd786d0
md"""
##### Linear Tile Coding

Since we are only learning the value function, the same tiling setup will have fewer parameters than the action value techniques.  Empirically, more tilings are necessary to learn a state value function that can approach the optimal policy.
"""

# ╔═╡ b0cc6ff8-7296-461c-9db7-e52fa518e2e2
#=╠═╡
function mountaincar_dist_test(max_episodes::Integer, α::Float32, ϵ::Float32; num_tiles = 24, num_tilings = 32, max_steps = typemax(Int64), kwargs...)
	setup = setup_mountain_car_tiles((1f0/num_tiles, 1f0/num_tiles), num_tilings)
	v = setup.args.feature_vector
	run_linear_semi_gradient_dp(mountain_car_dist_mdp, 1f0, max_episodes, max_steps, zeros(Float32, length(v)), setup.args.feature_vector_update; α = α, ϵ = ϵ, kwargs...)
end
  ╠═╡ =#

# ╔═╡ d0cf3806-05c6-4a50-94c8-55c9042d51b7
# ╠═╡ skip_as_script = true
#=╠═╡
(v̂_mountain_car, π_greedy_dp, episode_rewards_dp, episode_steps_dp) = mountaincar_dist_test(100, 0.001f0/32, 0.01f0)
  ╠═╡ =#

# ╔═╡ 7d21c4cd-ab79-4f40-9b8b-f637b3efcab0
# ╠═╡ skip_as_script = true
#=╠═╡
show_mountaincar_trajectory(π_greedy_dp, 1_000, "DP Learned Policy")
  ╠═╡ =#

# ╔═╡ 31fb07d2-1c34-44ec-b932-a598e78ec8dc
md"""
#### Non-linear Neural Network

Compared to tile coding, the feature vector for non-linear learning will simply be two values: one for the position and another for the velocity.  Both values will be scaled so the mean value is 0 and the variance is approximately 1.
"""

# ╔═╡ c12070a9-df63-4b25-99e6-26ff876af1b4
function update_mountaincar_feature_vector!(v::Vector{Float32}, s::NTuple{2, Float32})
	x1 = 3.45f0*(((s[1] - 1.2f0) / 1.7f0) - 0.5f0)
	x2 = 1.725f0*s[2] / 0.07f0
	v[1] = x1
	v[2] = x2
end

# ╔═╡ 0f958535-6b18-46de-a1ba-81f64c217ee0
#=╠═╡
function mountaincar_fcann_dp(max_episodes::Integer, α::Float32, ϵ::Float32; layers = [4, 4], max_steps = typemax(Int64), kwargs...)
	run_fcann_semi_gradient_dp(mountain_car_dist_mdp, 1f0, max_episodes, max_steps, zeros(Float32, 2), update_mountaincar_feature_vector!, layers; α = α, ϵ = ϵ, kwargs...)
end
  ╠═╡ =#

# ╔═╡ ee59176e-24b6-4213-8f8e-759a70bc1d5e
# ╠═╡ skip_as_script = true
#=╠═╡
mountaincar_fcann_dp_results = mountaincar_fcann_dp(100, 1f-5, 0.1f0; layers = [16, 16, 16], c = 10f0)
  ╠═╡ =#

# ╔═╡ 1e224a46-91ef-4a5f-ae35-ef4062147f2d
# ╠═╡ skip_as_script = true
#=╠═╡
show_mountaincar_trajectory(mountaincar_fcann_dp_results.π_greedy, 1_000, "DP Learned Policy")
  ╠═╡ =#

# ╔═╡ 00399548-b21c-43b5-90e2-30656ab1541e
# ╠═╡ skip_as_script = true
#=╠═╡
plot(scatter(y = -mountaincar_fcann_dp_results.reward_history), Layout(yaxis_type = "log"))
  ╠═╡ =#

# ╔═╡ 5db29488-a150-42ee-aedb-380a3a4fd548
# ╠═╡ skip_as_script = true
#=╠═╡
function plot_mountaincar_action_values(q̂_mountain_car, n1, n2)
	xvals = LinRange(-1.2f0, 0.5f0, n1)
	vvals = LinRange(-0.07f0, 0.07f0, n2)
	values = zeros(Float32, n1, n2)
	actions = zeros(Float32, n1, n2)
	for (i, x) in enumerate(xvals)
		for (j, v) in enumerate(vvals)
			(q̂, i_a) = q̂_mountain_car((x, v))
			values[j, i] = q̂
			actions[j, i] = mountain_car_actions[i_a]
		end
	end
	p1 = plot(heatmap(x = xvals, y = vvals, z = values), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Learned Value Function"))
	p2 = plot(heatmap(x = xvals, y = vvals, z = actions, colorscale = "rb", showscale = false), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Policy (blue = accelerate left, <br>red = accelerate right, gray = no acceleration)"))
	[p1 p2]
end
  ╠═╡ =#

# ╔═╡ c799ffe4-f4af-487d-b557-8b50d13632b7
#=╠═╡
plot_mountaincar_action_values(tabular_mountaincar_mdp, mountaincar_value_iteration.final_value, mountaincar_value_iteration.optimal_policy, mountaincar_positions, mountaincar_velocities)
  ╠═╡ =#

# ╔═╡ 58a0b622-1b51-4b42-a416-24109ae41a90
# ╠═╡ skip_as_script = true
#=╠═╡
plot_mountaincar_action_values(tabular_mountaincar_mdp, mountaincar_policy_iteration[2][policy_num], mountaincar_policy_iteration[1][policy_num], mountaincar_positions, mountaincar_velocities)
  ╠═╡ =#

# ╔═╡ 2f0d0a71-c65b-4aa0-a493-e7cdccd901eb
# ╠═╡ skip_as_script = true
#=╠═╡
function plot_mountaincar_values(v̂_mountain_car, π; n1 = 100, n2 = 100)
	xvals = LinRange(-1.2f0, 0.5f0, n1)
	vvals = LinRange(-0.07f0, 0.07f0, n2)
	values = zeros(Float32, n1, n2)
	actions = zeros(Float32, n1, n2)
	for (i, x) in enumerate(xvals)
		for (j, v) in enumerate(vvals)
			v̂ = v̂_mountain_car((x, v))
			values[j, i] = v̂
			actions[j, i] = π((x, v))
		end
	end
	p1 = plot(heatmap(x = xvals, y = vvals, z = values), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Learned Value Function"))
	p2 = plot(heatmap(x = xvals, y = vvals, z = actions, colorscale = "rb", showscale = false), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Policy (blue = accelerate left, <br>red = accelerate right, gray = no acceleration)"))
	[p1 p2]
end
  ╠═╡ =#

# ╔═╡ bd1f42e5-94cc-4aef-b82a-9bffd1c951d8
# ╠═╡ skip_as_script = true
#=╠═╡
plot_mountaincar_values(v̂_mountain_car, π_greedy_dp)
  ╠═╡ =#

# ╔═╡ b3658e4d-ee8e-45cd-906a-06dd512a6921
# ╠═╡ skip_as_script = true
#=╠═╡
plot_mountaincar_values(mountaincar_fcann_dp_results.value_function, mountaincar_fcann_dp_results.π_greedy; n1 = 100, n2 = 100)
  ╠═╡ =#

# ╔═╡ 7a47a518-dfc7-4310-a0b5-6f0d151c8263
md"""
### Neural Network Mountain Car Semi-gradient Sarsa
"""

# ╔═╡ c11aa069-93c2-435a-8f0e-353ced9633b6
# ╠═╡ skip_as_script = true
#=╠═╡
function mountaincar_fcann_test(max_steps::Integer, α::Float32, ϵ::Float32; num_layers = 3, layer_size = 2, kwargs...)
	feature_vector = zeros(Float32, 2)
	function update_feature_vector!(v::Vector{Float32}, s::NTuple{2, Float32})
		x1 = 3.45f0*(((s[1] - 1.2f0) / 1.7f0) - 0.5f0)
		x2 = 1.725f0*s[2] / 0.07f0
		v[1] = x1
		v[2] = x2
	end
	layers = fill(layer_size, num_layers)
	run_fcann_semi_gradient_sarsa(mountain_car_mdp, 1f0, 100, max_steps, feature_vector, update_feature_vector!, layers; α = α, ϵ = ϵ, kwargs...)
end
  ╠═╡ =#

# ╔═╡ 5fdbce61-ca25-45e0-b07d-94adf7138446
# ╠═╡ skip_as_script = true
#=╠═╡
mountain_car_fcann = mountaincar_fcann_test(1_000_000, 8f-7, 0.05f0; num_layers = 3, layer_size = 16, compute_value = compute_sarsa_value, λ = 0f-4, c = 10f0, dropout = 0.0f0)
  ╠═╡ =#

# ╔═╡ b9125c5b-01d6-451e-84b5-a419e38425b5
# ╠═╡ skip_as_script = true
#=╠═╡
π_mountain_car_fcann(s) = argmax(i_a -> mountain_car_fcann.value_function(s, i_a), eachindex(mountain_car_actions))
  ╠═╡ =#

# ╔═╡ fc3e0577-45aa-4bba-a275-fa7a352fc5cc
# ╠═╡ skip_as_script = true
#=╠═╡
plot_mountaincar_action_values(mountain_car_fcann.value_function, 200, 200)
  ╠═╡ =#

# ╔═╡ 5cbaeb8e-bc02-47c9-87b4-57df554cea9d
# ╠═╡ skip_as_script = true
#=╠═╡
show_mountaincar_trajectory(π_mountain_car_fcann, 1_000, "Sarsa Learned Policy")
  ╠═╡ =#

# ╔═╡ 59ec5223-f23f-4f32-9e5f-8a08e450da85
md"""
## 10.2 Semi-gradient *n*-step Sarsa

We can obtain an $n$-step version of semi-gradient Sarsa by using an $n$-step return as the update target for the semi-gradient Sarsa update equation (10.1).  The $n$-step return immediately generalizes from its tabular form (7.4) to a function approximation form: 

$G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n \hat q(S_{t+n}, A_{t+n}, \mathbf{w}_{t+n-1}), \quad t+n \lt T \tag{10.4}$

with $G_{t:t+n} \doteq G_t$ if $t+n \geq T$, as usual.  The $n$-step update equation is

$\mathbf{w}_{t+n} \doteq \mathbf{w}_{t+n-1} + \alpha \left [ G_{t:t+n} - \hat q(S_t, A_t, \mathbf{w}_{t+n-1}) \right ] \nabla \hat q(S_t, A_t, \mathbf{w}_{t+n-1}), \quad 0 \leq t \lt T \tag{10.5}$

As we have seen before, performance is often best with an $n$ that is some intermediate value between the 1-step sarsa method and Monte Carlo; however, we will not create a full implementation of this algorithm here as it will be replaced by semi-gradient Sarsa($\lambda$) in Chapter 12 which is a much more efficient version of the same concept.
"""

# ╔═╡ 49249ac1-8964-4afc-89f2-3cd4d4322cc2
md"""
> ### *Exercise 10.1* 
> We have not explicitely considered or given pseudocode for any Monte Carlo methods in this chapter.  What would they be like?  Why is it reasonable not to give pseudocode for them?  How would they perform on the Mountain Car task?

Monte Carlo methods require an episode to terminate prior to updating any action value estimates.  After the final reward is retrieved then all the action value pairs visited along the trajectory can be updated and the policy can be updated prior to starting the next episode.  For tasks such as the Mountain Car task where a random policy will likely never terminate, such a method will never be able to complete a single episode worth of updates.  We saw in earlier chapters with the racetrack and gridworld examples that for some environments a bootstrap method is the only suitable one given this possibility of an episode never terminating.
"""

# ╔═╡ e1abf8c7-06b8-4cd5-b557-1d187004bdf1
md"""
> ### *Exercise 10.2* 
> Give pseudocode for semi-gradient one-step *Expected* Sarsa for control.

Use the same pseudocode given for semi-gradient one-step Sarsa but with the following change to the weight update step in the non-terminal case:

$\mathbf{w} \leftarrow \mathbf{w} + \alpha[R + \gamma \sum_a \pi(a|S^\prime)\hat q(S^\prime, a, \mathbf{w}) - \hat q(S, A, \mathbf{w}) ] \nabla \hat q(S, A, \mathbf{w})$

where $\pi$ is the currently used policy which is $\epsilon$ greedy with respect to $\hat q$.  See complete implementation below. 
"""

# ╔═╡ 98a5d65e-4253-4523-a74e-99d03be03b89
md"""
### *Semi-gradient Expected Sarsa Implementation*

Since we already update the policy and action values in the sarsa algorithm, the only difference in expected sarsa is to compute the action-value using the entire policy distribution instead of just the sampled action.  Similarly for Q-learning we would only select the maximum value.
"""

# ╔═╡ 8ed6f8fd-8574-4d5a-9964-ce8a32629c6f
compute_expected_sarsa_value(action_values::Vector{T}, policy::Vector{T}, i_a::Integer) where T<:Real = dot(action_values, policy)

# ╔═╡ 1410db13-4b73-4a87-af34-30a5232af4ba
compute_q_learning_value(action_values::Vector{T}, policy::Vector{T}, i_a::Integer) where T<:Real = maximum(action_values)

# ╔═╡ 7812e801-70fd-4331-ad0a-fad02c1a399f
semi_gradient_expected_sarsa!(args...; kwargs...) = semi_gradient_sarsa!(args...; kwargs..., compute_value=compute_expected_sarsa_value)

# ╔═╡ 5be866c3-0fb2-4d1f-9c31-b85aba332905
semi_gradient_q_learning!(args...; kwargs...) = semi_gradient_sarsa!(args...; kwargs..., compute_value=compute_q_learning_value)

# ╔═╡ f7410fe7-e3d8-4047-8fa7-f076476e9d3a
md"""
### Example: Semi-gradient Q-learning on Mountain Car Task
"""

# ╔═╡ d6ad1ff1-8fbf-4799-8b1b-ae1e3ce88c5b
md"""
> ### *Exercise 10.3* 
> Why do the results shown in Figure 10.4 have higher standard errors at large *n* than at small *n*?

At large n more of the reward function comes from the actual trajectory observed during a run.  Since random actions are taken initially there will be more spread in the observed reward estimates than with 1 step bootstrapping which is more dependent on the initialization of the action value function.  If ties are broken randomly then you would select random actions for the first n-steps of bootstrapping thus experience more spread in the early trajectories for higher n.
"""

# ╔═╡ b8c031ca-7995-4501-a1e3-df3f34e5f0da
md"""
## 10.3 Average Reward: A New Problem Setting for Continuing Tasks

We now introduce an alternative to the discount setting for solving continuing problems (MDPs without a terminal state).  The average-reward setting is more commonly used in the classical theory of dynamic programming.  The purpose of introducing the average-reward is because discounting is problematic with function approximation in a way it was not problematic for tabular problems.  

In the average-reward setting the quality of a policy $\pi$ is defined as the average rate of reward, or simply *average reward*, while following that policy, which we denote as $r(\pi)$:

$\begin{flalign}
r(\pi) &\doteq \lim_{h \rightarrow \infty} \frac{1}{h}\sum_{t=1}^h \mathbb{E}[R_t \mid S_0,A_{0:t-1} \sim \pi] \tag{10.6}\\
&= \lim_{h \rightarrow \infty} \mathbb{E} [R_t \mid S_0,A_{0:t-1} \sim \pi] \tag{10.7}\\
&= \sum_s \mu_\pi(s)\sum_a\pi(a \vert s) \sum_{s^\prime,r} p(s^\prime,r \vert s, a)r
\end{flalign}$

where the expectations are conditioned on the initial state, $S_0$, and on the subsequent actions, $A_0, A_1, \dots,A_{t-1}$, being taken according to $\pi$. The second and third equations hold if the state-state distribution $\mu_\pi(s) \doteq \lim_{t\rightarrow \infty} \Pr \{S_t = s \mid A_{0:t-1} \sim \pi \}$, exists and is independent of $S_0$, in other words, if the MDP is *ergodic*. In an ergodic MDP, the starting state and any early decision made by the agent can only have a temporary effect; in the long run the expectation of being in a state depends on the policy and the MDP transition probabilities.  Ergodicity is sufficient but not necessary to guarantee the existence of the limit in (10.6).

In this setting, we consider all policies that obtain the maximum value of $r(\pi)$ or the *reward rate* to be optimal.  Note that the steady state distribution $\mu_\pi$ is the special distribution under which, if you select actions according to $\pi$, you remain in the same distribution.  That is, for which 

$\sum_s \mu_\pi(s) \sum_a \pi(a\vert s)p(s^\prime \vert s, a) = \mu_\pi(s^\prime) \tag{10.8}$

In the average-reward setting, returns are defined in terms of differences between rewards and the average reward: 

$G_t \doteq R_{t+1} - r(\pi) + R_{t+2} - r(\pi) + R_{t+3} - r(\pi) + \cdots \tag{10.9}$

This is known as the *differential* return, and th corresponding value functions are known as *differential* value functions.  Differential value functions are defined in terms of the new return just as conventional value functions were defined in terms of the discounted return; thus we will use the same notation, $v_\pi (s) \doteq \mathbb{E}_\pi[G_t \vert S_t = s]$ and $q_\pi (s, a) \doteq \mathbb{E}_\pi[G_t \vert S_t = s, A_t = a]$ (similarly for $v_*$ and $q_*$), for differential value functions.  Differential value functions also have Bellman equations, just slightly different from those we have seen earlier.  We simply remove all $\gamma$s and replace all rewards by the difference between the reward and the true average reward:

$\begin{flalign}
&v_\pi(s) = \sum_a \pi(a\vert s) \sum_{r, s^\prime}p(s^\prime, r \vert s, a) \left [ r - r(\pi) + v_\pi(s^\prime) \right ] \\
&q_\pi(s, a) = \sum_{r, s^\prime}p(s^\prime, r \vert s, a) \left [ r - r(\pi) + \sum_{a^\prime} \pi(a^\prime \vert s^\prime) q_\pi(s^\prime, a^\prime) \right ] \\
&v_* = \max_a \sum_{r, s^\prime}p(s^\prime, r \vert s, a) \left [ r - r(\pi) + v_*(s^\prime) \right ] \\
&q_* = \sum_{r, s^\prime}p(s^\prime, r \vert s, a) \left [ r - \max_{\pi}r(\pi) + \max_a q_\pi(s^\prime, a^\prime) \right ] \\
\end{flalign}$

There is also a differential form of the two TD errors:

$\delta_t \doteq R_{t+1} - \bar{R}_t+ \hat v (S_{t+1}, \mathbf{w}_t) - \hat v(S_t, \mathbf{w}_t) \tag{10.10}$

and

$\delta_t \doteq R_{t+1} - \bar{R}_t+ \hat q (S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat q(S_t, A_t, \mathbf{w}_t) \tag{10.11}$

where $\bar{R}_t$ is an estimate at time $t$ of the average reward $r(\pi)$.  With these alternate definitions, most of our algorithms and many theoretical results carry through to the average_reward setting without any change.  

For example, an average reward version of semi-gradient Sarsa could be defined just as in (10.2) except with the differential version of the TD error.  That is by

$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t \nabla \hat q(S_t, A_t, \mathbf{w}_t)$

with $\delta_t$ given by (10.11).  See a full implementation below.  One limitation of this algorithm is that it does not converge to the differential values but to the differential values plut an arbitrary offset.  Notice that the Bellman equations and TD errors given above are unaffected if all the values are shifted by the same amount.  Thus, the offset may not matter in practice.
"""

# ╔═╡ 69a06405-57cd-42e5-96b1-5cc77d74aa03
md"""
### *Differential Semi-gradient Sarsa Implementation*
"""

# ╔═╡ a9fdb1fd-3f62-4e1c-9157-c4eee6215261
function differential_semi_gradient_sarsa!(parameters, mdp::StateMDP{T, S, A, P, F1, F2, F3}, max_episodes::Integer, max_steps::Integer, estimate_value::Function, estimate_args::Tuple, update_parameters!::Function, update_args::Tuple; α = one(T)/10, β = one(T)/100, ϵ = one(T) / 10, compute_value = compute_sarsa_value, kwargs...) where {T<:Real, S, A, P, F1, F2, F3}
	s = mdp.initialize_state()
	i_a = rand(eachindex(mdp.actions))
	ep = 1
	step = 1
	R̄ = zero(T)
	ō = zero(T)
	epreward = zero(T)
	episode_rewards = zeros(T, max_episodes)
	episode_steps = zeros(Int64, max_episodes)
	action_values = zeros(T, length(mdp.actions))
	average_step_reward = Vector{T}()
	policy = zeros(T, length(mdp.actions))
	while (ep <= max_episodes) && (step <= max_steps)
		(r, s′) = mdp.ptf(s, i_a)
		estimate_value(action_values, s, parameters, estimate_args...)
		q̂ = action_values[i_a]
		U_t = r - R̄
		epreward += r
		if mdp.isterm(s′)
			s′ = mdp.initialize_state()
			episode_rewards[ep] = epreward
			episode_steps[ep] = step
			epreward = zero(T)
			ep += 1
		end
		estimate_value(action_values, s′, parameters, estimate_args...)
		policy .= action_values
		make_ϵ_greedy_policy!(policy; ϵ = ϵ)
		i_a′ = sample_action(policy)
		q̂′ = compute_value(action_values, policy, i_a′)
		U_t += q̂′
		δ = U_t - q̂
		ō += β * (one(T) - ō)
		R̄ += (β/ō)*δ
		# R̄ += β*δ
		push!(average_step_reward, R̄)
		update_parameters!(parameters, s, i_a, U_t, α, update_args...)
		s = s′
		i_a = i_a′
		step += 1
	end
	return episode_rewards, episode_steps, average_step_reward
end

# ╔═╡ 565c53ee-7ad5-44e2-bce5-4ff1f5f162c0
differential_semi_gradient_expected_sarsa!(args...; kwargs...) = differential_semi_gradient_sarsa!(args...; kwargs..., compute_value=compute_expected_sarsa_value)

# ╔═╡ c9759bd9-ec9b-47a1-9080-a7fc332be565
differential_semi_gradient_q_learning!(args...; kwargs...) = differential_semi_gradient_sarsa!(args...; kwargs..., compute_value=compute_q_learning_value)

# ╔═╡ b5d2776f-4b93-4eaa-8873-c1c4e610e6b0
function run_fcann_differential_semi_gradient_sarsa(mdp::StateMDP, max_episodes::Integer, max_steps::Integer, state_representation::Vector{Float32}, update_state_representation!::Function, layers::Vector{T}; kwargs...) where T<:Integer
	setup = fcann_action_gradient_setup(mdp, layers, state_representation, update_state_representation!)
	num_actions = length(mdp.actions)
	episode_rewards, episode_steps, average_step_reward = differential_semi_gradient_sarsa!(setup.parameters, mdp, max_episodes, max_steps, setup.value_function, setup.value_args, setup.parameter_update, setup.update_args; kwargs...)
	q̂(s, i_a) = setup.value_function(s, i_a, setup.parameters, setup.value_args...)
	action_values = zeros(Float32, num_actions)
	q̂(s) = setup.value_function(action_values, s, setup.parameters, setup.value_args...)
	return (value_function = q̂, reward_history = episode_rewards, step_history = episode_steps, average_step_reward = average_step_reward, parameters = setup.parameters)
end

# ╔═╡ 1a7ba296-52ca-4069-85fa-792d08d77b0e
md"""
### Example: Differential Sarsa and Q-learning with Mountain Car Task

In order to apply differential learning to the mountain car task, we need to change the rewards per step.  Previously, the rewards were assigned in a manner appropriate for learning with a discount rate of 1.  The reward of -1 per episode step ensures that policies that finish the task faster have a higher reward.  In the average reward setting, every policy would have an average reward per step of -1 making the task ill posed.  Instead, we can assign a reward of 1 for finishing to the right and 0 at all other steps.  These rewards would produce an ill posed task for $\gamma = 1$ but are perfectly fine for the average reward setting.  Now our learning procedure should find a policy that produces the highest average reward $\frac{1}{\text{num steps}}$ which is maximized when the number of steps to finish an episode is minimized.
"""

# ╔═╡ eb28458f-b222-4f8e-9a5b-8203d3997f7b
function mountain_car_differential_step(s::Tuple{Float32, Float32}, i_a::Int64)
	a = mountain_car_actions[i_a]
	ẋ′ = clamp(s[2] + 0.001f0*a - 0.0025f0*cos(3*s[1]), -0.07f0, 0.07f0)
	x′ = clamp(s[1] + ẋ′, -1.2f0, 0.5f0)
	x′ == -1.2f0 && return (0f0, (x′, 0f0))
	s′ = (x′, ẋ′)
	r = Float32(x′ == 0.5f0)
	return (r, s′)
end

# ╔═╡ e5ad765a-341f-4f11-9ae8-37d81cb349d2
# ╠═╡ skip_as_script = true
#=╠═╡
const mountain_car_differential_transition = StateMDPTransitionSampler(mountain_car_differential_step, initialize_car_state())
  ╠═╡ =#

# ╔═╡ bc1d7cce-c0f4-47a8-b674-8acb82491c7f
# ╠═╡ skip_as_script = true
#=╠═╡
const mountain_car_differential_mdp = StateMDP(mountain_car_actions, mountain_car_differential_transition, initialize_car_state, s -> s[1] == 0.5f0)
  ╠═╡ =#

# ╔═╡ d3ba78fa-f032-4bb9-9359-ef3bcff2252d
# ╠═╡ skip_as_script = true
#=╠═╡
function mountaincar_fcann_differential_test(max_steps::Integer, α::Float32, β::Float32, ϵ::Float32; num_layers = 3, layer_size = 2, kwargs...)
	feature_vector = zeros(Float32, 2)
	function update_feature_vector!(v::Vector{Float32}, s::NTuple{2, Float32})
		x1 = 3.45f0*(((s[1] - 1.2f0) / 1.7f0) - 0.5f0)
		x2 = 1.725f0*s[2] / 0.07f0
		v[1] = x1
		v[2] = x2
	end
	layers = fill(layer_size, num_layers)
	run_fcann_differential_semi_gradient_sarsa(mountain_car_differential_mdp, 1000, max_steps, feature_vector, update_feature_vector!, layers; α = α, β = β, ϵ = ϵ, kwargs...)
end
  ╠═╡ =#

# ╔═╡ ae5c5377-8b44-4c82-a63c-d2cb8a0d6667
#=╠═╡
(q̂_mountain_car2_fcann, episode_rewards2_fcann, episode_steps2_fcann, average_step_reward_fcann) = mountaincar_fcann_differential_test(1_000_000, 1f-4, 0.001f0, 0.5f0; num_layers = 3, layer_size = 8, compute_value = compute_q_learning_value)
  ╠═╡ =#

# ╔═╡ b094bf9f-bb97-4f23-acdc-f39411a07fb9
#=╠═╡
π_mountain_car2_fcann(s) = rand() < 0.05 ? rand(1:3) : argmax(i_a -> q̂_mountain_car2_fcann(s, i_a), eachindex(mountain_car_actions))
  ╠═╡ =#

# ╔═╡ 425fe768-c7bb-4d3e-87e6-47fa052ba612
#=╠═╡
plot(scatter(y = (episode_steps2_fcann[2:findlast(!iszero, episode_steps2_fcann)] .- episode_steps2_fcann[1:findlast(!iszero, episode_steps2_fcann) - 1])), Layout(yaxis_type = "log"))
  ╠═╡ =#

# ╔═╡ b191d3f9-cf25-4fb4-8f5a-8da86e96e125
#=╠═╡
show_mountaincar_trajectory(π_mountain_car2_fcann, 1_000, "Differential Q-learning Learned Policy")
  ╠═╡ =#

# ╔═╡ c44dd6c6-8213-49fb-8d33-ba8f2c766b2e
#=╠═╡
plot_mountaincar_action_values(q̂_mountain_car2_fcann, 500, 500)
  ╠═╡ =#

# ╔═╡ 9df1a18d-137c-4ea5-8d15-05697f7bbf07
md"""
> ### *Exercise 10.4* 
> Give pseudocode for a differential version of semi-gradient Q-learning.

Given the pseudocode for semi-gradient Sarsa, make the following changes:

$\vdots$

Initialize S

Loop for each step of episode:

Choose A from S using ϵ-greedy policy
Take action A, observe R, S'

$\delta \leftarrow R - \bar R + \max_a \hat q(S^\prime, a, \mathbf{w}) - \hat q(S, A, \mathbf{w})$

$\vdots$
$S \leftarrow S^\prime$

See implementation above
"""

# ╔═╡ 0c7f5742-6c51-4c6a-b67f-217163935ba5
md"""
> ### *Exercise 10.5* 
> What equations are needed (beyond 10.10) to specify the differential version of TD(0)?

10.10 includes a reward estimate at time t, $\bar R_t$, which also needs to be updated.  The TD error represents the newly observed reward the was experienced in excess of the estimated average so the update equation should move $\bar R$ in the direction of the TD error.  After each step, the following updates should occur.

$\begin{flalign}
\delta &\leftarrow R - \bar R + \hat v(S^\prime, \mathbf{w}) - \hat v(S, \mathbf{w}) \\
\bar R &\leftarrow \bar R + \beta \delta \\
\mathbf{w} &\leftarrow \mathbf{w} + \alpha \delta \nabla \hat v(S, \mathbf{w}) \\
S & \leftarrow S^\prime \\
\end{flalign}$
"""

# ╔═╡ a6c5ec28-b2d5-4893-a118-95c1318d1f7f
md"""
> ### *Exercise 10.6* 
> Suppose there is an MDP that under any policy produces the deterministic sequence of rewards +1, 0, +1, 0, +1, 0, . . . going on forever. Technically, this violates ergodicity; there is no stationary limiting distribution $μ_\pi$ and the limit (10.7) does not exist. Nevertheless, the average reward (10.6) is well defined. What is it? Now consider two states in this MDP. From A, the reward sequence is exactly as described above, starting with a +1, whereas, from B, the reward sequence starts with a 0 and then continues with +1, 0, +1, 0, . . .. We would like to compute the differential values of A and B. Unfortunately, the differential return (10.9) is not well defined when starting from these states as the implicit limit does not exist. To repair this, one could alternatively define the differential value of a state as $v_\pi (s) \doteq \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} \sum_{t=0}^h \gamma^t \left ( \mathbb{E_\pi} [R_{t+1}|S_0=s]-r(\pi)  \right )$.  Under this definition what are the differential values of states A and B?
"""

# ╔═╡ 44d43dbf-fe32-438e-b89d-c677bbc35893
md"""
In order to use (10.6): $r(\pi) \doteq \lim_{h \rightarrow \infty} \frac{1}{h} \sum_{t = 1}^h \mathbb{E} [R_t \mid S_0, A_{0:t-1} \sim \pi]$ we need to compute $\mathbb{E} [R_t \mid S_0, A_{0:t-1} \sim \pi]$.  In this case, we are told that regardless of the policy, the reward sequence will be +1, 0, +1, 0, ....  We can therefore replace the expected values in the equation with this sequence since the rewards at each time step are known with 100% probability.

the average reward can be computed as $r(\pi) = \lim_{h \rightarrow \infty} \frac{1}{2h}\sum_{t=1}^h (-1)^{t+1} + 1 = \lim_{h \rightarrow \infty} \frac{h}{2h} + \frac{1}{2h} \sum_{t=1}^h (-1)^{t+1}$

The sum left in the expression is $1 - 1 + 1 - 1 \cdots$ which is 1 for even h and 0 for odd h.  Either way, when divided by $2h$ that term goes to 0 leaving only the term $\frac{h}{2h} = \frac{1}{2}$ so $r(\pi) = \frac{1}{2}$

To compute the differential value function for state A and B, consider the alternative definition above using the fact that $r(\pi) = 0.5$.  

For state A, each parenthetical term in the sum will be: $1 - 0.5, 0 - 0.5, 1 - 0.5, 0 - 0.5, \dots = 0.5, -0.5, 0.5, -0.5, \dots$

For state B, each parenthetical term in the sum will be: $0 - 0.5, 1 - 0.5, 0 - 0.5, 1 - 0.5, \dots = -0.5, 0.5, -0.5, 0.5, \dots$

$\begin{flalign}
v_\pi (A) &= \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} 0.5 - 0.5\gamma + 0.5 \gamma^2 - 0.5\gamma^3 + \cdots \\
&=0.5\lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty}\sum_{t=0}^h (-\gamma)^t \\
&=0.5\lim_{\gamma \rightarrow 1}\frac{1}{\gamma +1 } = 0.25
\end{flalign}$

$\begin{flalign}
v_\pi (B) &= \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} -0.5 + 0.5\gamma - 0.5 \gamma^2 + 0.5\gamma^3 + \cdots \\
&=-0.5\lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty}\sum_{t=0}^h (-\gamma)^t \\
&=-0.5\lim_{\gamma \rightarrow 1}\frac{1}{\gamma +1 } = -0.25
\end{flalign}$
"""

# ╔═╡ f1edb500-fbd1-4c03-b033-53860dfa452d
md"""
> ### *Exercise 10.7* 
> Consider a Markov reward process consisting of a ring of three states A, B, and C, with state transitions going deterministically around the ring.  A reward of +1 is received upon arrival in A and otherwise the reward is 0.  What are the differential values of the three states, using (10.13)

From 10.13 we have 

$v_\pi (s) \doteq \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} \sum_{t=0}^h \gamma^t \left ( \mathbb{E_\pi} [R_{t+1}|S_0=s]-r(\pi)  \right )$

The average reward per step is $\frac{1}{3}$ so we can apply the same method used in exercise 10.6 where the elements inside the parentheses of the sum are: $\frac{2}{3}$ for $C \rightarrow A$ and $-\frac{1}{3}$ for the other two.  Starting in state A we transition twice and then on the third arrive in state A leading to the following mean corrected values of $-\frac{1}{3}$, $-\frac{1}{3}$, and $\frac{2}{3}$.  The other states will have these values cyclically permuted leading to the following infinite sums:

For state A:
$-\frac{1}{3} - \frac{1}{3}\gamma + \frac{2}{3}\gamma^2 - \frac{1}{3}\gamma^3 - \frac{1}{3}\gamma^4 + \cdots$

For state B:
$-\frac{1}{3} + \frac{2}{3}\gamma - \frac{1}{3}\gamma^2 - \frac{1}{3}\gamma^3 + \frac{2}{3}\gamma^4 + \cdots$

For state C:
$\frac{2}{3} - \frac{1}{3}\gamma - \frac{1}{3} \gamma^2 + \frac{2}{3}\gamma^3 + \cdots = 3 \times (2 - \gamma - \gamma^2 + 2\gamma^3 + \cdots)$

Comparing these sequences we have:

$\gamma \times v(A) = v(C) - \frac{2}{3} \implies v(A) = \frac{v(C) - \frac{2}{3}}{\gamma}$
$\gamma \times v(B) = v(A) + \frac{1}{3} \implies v(A) = \gamma \times v(B) - \frac{1}{3}$

so

$\frac{v(C) - \frac{2}{3}}{\gamma} = \gamma \times v(B) - \frac{1}{3} \implies v(C) = \gamma \left ( \gamma \times v(B) - \frac{1}{3} \right ) + \frac{2}{3}$

also 

$\gamma \times v(C) = v(B) + \frac{1}{3} \implies v(C) = \frac{v(B) + \frac{1}{3}}{\gamma}$

Equation the two sides for $v(C)$ that only contain $v(B)$ terms we have:

$\frac{v(B) + \frac{1}{3}}{\gamma} = \gamma \left ( \gamma \times v(B) - \frac{1}{3} \right ) + \frac{2}{3}$

$v(B) = \gamma \left ( \gamma \left ( \gamma \times v(B) - \frac{1}{3} \right ) + \frac{2}{3} \right ) - \frac{1}{3} = \gamma^3 v(B) - \gamma^2 \frac{1}{3} + \gamma\frac{2}{3} - \frac{1}{3}$

$v(B) \left ( 1 - \gamma^3 \right ) = - \gamma^2 \frac{1}{3} + \gamma\frac{2}{3} - \frac{1}{3} \implies v(B) = \frac{- \gamma^2 \frac{1}{3} + \gamma\frac{2}{3} - \frac{1}{3}}{1 - \gamma^3}$

$v(B) = -\frac{1}{3} \frac{\gamma^2 - 2\gamma + 1}{1 - \gamma^3} = -\frac{1}{3} \frac{(\gamma - 1)^2}{-(\gamma - 1)(\gamma^2 + \gamma + 1)} = \frac{1}{3} \frac{\gamma - 1}{\gamma^2 + \gamma + 1}$

Therefore, 

$\begin{flalign}
\lim_{\gamma \rightarrow 1} v(B) &= \frac{1}{3} \frac{1 - 1}{3} = 0 \\
\lim_{\gamma \rightarrow 1} v(A) &= \gamma \times 0 - \frac{1}{3} = -\frac{1}{3} \\
$\lim_{\gamma \rightarrow 1} v(C) &=  \frac{0 + \frac{1}{3}}{\gamma} = \frac{1}{3}
\end{flalign}$
"""

# ╔═╡ 2d7679ad-a9b3-448b-a4bc-7e5b9bce6adb
md"""
> ### *Exercise 10.8* 
> The pseudocode in the box on page 251 updates $\bar R_t$ using $\delta_t$ as an error rather than simply $R_{t+1} - \bar R_t$.  Both errors work, but using $\delta_t$ is better.  To see why, consider the ring MRP of three states from Exercise 10.7.  The estimate of the average reward should tend towards its true value of $\frac{1}{3}$.  Suppose it was already there and was held stuck there.  What would the sequence of $R_{t+1} - \bar R_t$ errors be?  What would the sequence of $\delta_t$ errors be (using Equation 10.10)?  Which error sequence would produce a more stable estimate of the average reward if the estimate were allowed to change in response to the errors? Why?

The sequence of $R_{t+1} - \bar R_t$ would be given by the cyclical sequence of rewards.  Let's assume we start the sequence at state A.  Then our reward sequence will be 0, 0, 1, 0, 0, 1... so the error sequence will be $-\frac{1}{3}$, $-\frac{1}{3}$, $\frac{2}{3}$,...  If we update the average error estimate using these corrections it would remain centered at the correct value but fluctuate up and down with each correction.

In order to calculate $\delta_t$ we must use the definition given by 10.10:

$\delta_t = R_{t+1} - \bar R_t + \hat v(S_{t+1}, \mathbf{w}_t) - \hat v(S_t, \mathbf{w}_t)$

This equation requires us to have value estimates for each state which we can assume have converged to the true values as we have for the average reward estimate: $\hat v(A) = -\frac{1}{3}$, $\hat v(B) = 0$, and $\hat v(C) = \frac{1}{3}$.  Starting at state A, $\delta_t = 0 - \frac{1}{3} + 0 - -\frac{1}{3} = 0$.  For the following state we have $0 - \frac{1}{3} + \frac{1}{3} = 0$.  Finally we have $1 - \frac{1}{3} + -\frac{1}{3} - \frac{1}{3} = 0$.  So if we use the TD error to update our average reward estimate, at equilibrium all the values will remain unchanged.  Using $\delta$ provides a lower variance estimator than simply using the reward.

"""

# ╔═╡ a9b74949-9392-4048-bcb6-5fd48c1d9b98
md"""
### Example 10.2: An Access-Control Queuing Task
"""

# ╔═╡ fbf1c64f-1979-4384-a8c6-dc7875174d1f
begin
	abstract type AccessControlAction end
	struct Accept <: AccessControlAction end
	struct Reject <: AccessControlAction end
end

# ╔═╡ e7372e2b-a2db-4a93-9efc-f75aa74c197b
struct AccessControlState
	num_free_servers::Int64
	top_priority::Float32
end

# ╔═╡ 014339eb-5b23-4ac5-a551-8eeb2238366f
begin 
	function access_control_step(s::AccessControlState, ::Reject, num_servers::Integer, priority_payments::Vector{Float32})
		occupied_servers = num_servers - s.num_free_servers
		freed_servers = sum(_ -> Float32(rand() < 0.06), 1:occupied_servers; init = 0f0)
		new_occupied_servers = occupied_servers - freed_servers
		new_free_servers = num_servers - new_occupied_servers
		new_priority = rand(priority_payments)
		(0f0, AccessControlState(new_free_servers, new_priority))
	end

	function access_control_step(s::AccessControlState, ::Accept, num_servers::Integer, priority_payments::Vector{Float32})
		occupied_servers = num_servers - s.num_free_servers
		(r_reject, s′) = access_control_step(s, Reject(), num_servers, priority_payments)
		s.num_free_servers == 0 && return (r_reject, s′)
		(s.top_priority, AccessControlState(s′.num_free_servers - 1, s′.top_priority))
	end
end

# ╔═╡ 96548352-cd4d-4448-8312-ed10057f4359
begin
	function update_parameters!(parameters::Vector{Vector{T}}, i_s::Integer, i_a::Integer, g::T, α::T) where {T<:Real}
		q̂ = parameters[i_a][i_s]
		δ = (g - q̂)
		parameters[i_a][i_s] += α*δ
		return nothing
	end
	function update_parameters!(parameters::Matrix{T}, i_s::Integer, i_a::Integer, g::T, α::T) where {T<:Real}
		q̂ = parameters[i_s, i_a]
		δ = (g - q̂)
		parameters[i_s, i_a] += α*δ
		return nothing
	end
end

# ╔═╡ b9ebd6bb-90a1-4945-85ed-023206e2420a
function linear_features_action_gradient_setup(problem::Union{StateMDP{T, S, A, P, F1, F2, F3}, StateMRP{T, S, P, F1, F2}}, state_representation::AbstractVector{T}, update_feature_vector!::Function) where {T<:Real, S, A, P, F1<:Function, F2<:Function, F3<:Function}
	s0 = problem.initialize_state()
	update_feature_vector!(state_representation, s0) #verify that feature vector update is compatible with provided state representation

	function update_params!(parameters::AbstractArray, s::S, i_a::Integer, g::T, α::T, state_representation::AbstractVector{T}) where T<:Real
		update_feature_vector!(state_representation, s)
		q̂ = calculate_action_value(state_representation, i_a, parameters)
		δ = (g - q̂)
		iszero(δ) && return nothing
		update_parameters!(parameters, state_representation, i_a, α, δ)
		return nothing
	end
	
	function q̂(s::S, i_a::Integer, parameters::AbstractArray, state_representation::AbstractVector{T}) where {T<:Real} 
		update_feature_vector!(state_representation, s)
		calculate_action_value(state_representation, i_a, parameters)
	end

	function q̂(action_values::Vector{T}, s::S, parameters::AbstractArray, state_representation::AbstractVector{T}) where {T<:Real} 
		update_feature_vector!(state_representation, s)
		fill_action_values!(action_values, state_representation, parameters)
		findmax(action_values)
	end
	
	return (value_function = q̂, value_args = (state_representation,), parameter_update = update_params!, update_args = (copy(state_representation),))
end

# ╔═╡ 2c620fe4-2f62-40f8-a666-8dced1e0b84a
function run_linear_semi_gradient_sarsa(mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, state_representation::AbstractVector{T}, update_state_representation!::Function; setup_kwargs = NamedTuple(), kwargs...) where T<:Real
	setup = linear_features_action_gradient_setup(mdp, state_representation, update_state_representation!; setup_kwargs...)
	l = length(state_representation)
	num_actions = length(mdp.actions)
	# parameters = zeros(T, l, num_actions)
	parameters = [zeros(T, l) for _ in 1:num_actions]
	episode_rewards, episode_steps = semi_gradient_sarsa!(parameters, mdp, γ, max_episodes, max_steps, setup.value_function, setup.value_args, setup.parameter_update, setup.update_args; kwargs...)
	q̂(s, i_a) = setup.value_function(s, i_a, parameters, setup.value_args...)
	function q̂(s)
		action_values = zeros(T, num_actions)
		setup.value_function(action_values, s, parameters, setup.value_args...)
		findmax(action_values)
	end

	π_greedy(s) = q̂(s)[2]
	
	return (value_function = q̂, π_greedy = π_greedy, reward_history = episode_rewards, step_history = episode_steps)
end

# ╔═╡ 7c5fb569-81f0-4b70-ae95-1fce0c51b6f4
# ╠═╡ skip_as_script = true
#=╠═╡
function mountaincar_test(max_episodes::Integer, α::Float32, ϵ::Float32; num_tiles = 12, num_tilings = 8, kwargs...)
	setup = setup_mountain_car_tiles((1f0/num_tiles, 1f0/num_tiles), num_tilings)
	v = setup.args.feature_vector
	run_linear_semi_gradient_sarsa(mountain_car_mdp, 1f0, max_episodes, typemax(Int64), zeros(Float32, length(v)), setup.args.feature_vector_update; α = α, ϵ = ϵ, kwargs...)
end
  ╠═╡ =#

# ╔═╡ 30ab21ba-3f5b-46a8-8b8c-753f2755d419
# ╠═╡ skip_as_script = true
#=╠═╡
(q̂_mountain_car, π_mountain_car, episode_rewards, episode_steps) = mountaincar_test(5000, 0.0001f0/8, 0.01f0)
  ╠═╡ =#

# ╔═╡ 4afbb723-340b-4d85-9115-027a0ff8dfad
# ╠═╡ skip_as_script = true
#=╠═╡
plot_mountaincar_action_values(q̂_mountain_car, 500, 500)
  ╠═╡ =#

# ╔═╡ f2201afe-8952-4dde-9e39-02beeb920f6f
# ╠═╡ skip_as_script = true
#=╠═╡
show_mountaincar_trajectory(π_mountain_car, 1_000, "Sarsa Learned Policy")
  ╠═╡ =#

# ╔═╡ c1388562-0708-4a6a-acfe-927413dab5d2
# ╠═╡ skip_as_script = true
#=╠═╡
plot(scatter(y = -episode_rewards), Layout(yaxis_type = "log"))
  ╠═╡ =#

# ╔═╡ 1a82ae95-3c3e-4281-bc1d-9eb19bf50286
# ╠═╡ skip_as_script = true
#=╠═╡
function figure_10_2(;α_list = [0.1f0, 0.2f0, 0.5f0], num_episodes = 50, ϵ = 0.05f0)
	traces = map(α_list) do α
		scatter(y = 1:100 |> Map(_ -> mountaincar_test(num_episodes, α/8, ϵ; num_tiles = 12, num_tilings = 8).reward_history) |> foldxt((a, b) -> a .+ b) |> v -> -v ./ 100, name = "α = $α/8")
	end
	plot(traces, Layout(xaxis_title = "Episode", yaxis_title = "Steps per episode<br>averaged over 100 runs", yaxis_type = "log"))
end
  ╠═╡ =#

# ╔═╡ ddcb50be-5287-47f8-89f9-58c026a6b151
# ╠═╡ skip_as_script = true
#=╠═╡
figure_10_2()
  ╠═╡ =#

# ╔═╡ cbac1927-b087-4c4c-98ae-6aa5f0b824ad
# ╠═╡ skip_as_script = true
#=╠═╡
(q̂_mountain_car_q, π_mountain_car_q, episode_rewards_q, episode_steps_q) = mountaincar_test(1_000, 0.002f0/8, 0.25f0; compute_value = compute_q_learning_value)
  ╠═╡ =#

# ╔═╡ b5409b69-a254-4355-b2b9-99394eceb2f7
# ╠═╡ skip_as_script = true
#=╠═╡
show_mountaincar_trajectory(π_mountain_car_q, 1_000, "Q-Learning Learned Policy")
  ╠═╡ =#

# ╔═╡ f9ee13e8-7406-4fba-9a30-1e2714bd7cfc
# ╠═╡ skip_as_script = true
#=╠═╡
plot_mountaincar_action_values(q̂_mountain_car_q, 500, 500)
  ╠═╡ =#

# ╔═╡ 065b2626-01f1-443f-8be4-3036003a2772
function run_linear_differential_semi_gradient_sarsa(mdp::StateMDP, max_episodes::Integer, max_steps::Integer, state_representation::AbstractVector{T}, update_state_representation!::Function; setup_kwargs = NamedTuple(), kwargs...) where T<:Real
	setup = linear_features_action_gradient_setup(mdp, state_representation, update_state_representation!; setup_kwargs...)
	l = length(state_representation)
	num_actions = length(mdp.actions)
	# parameters = zeros(T, l, num_actions)
	parameters = [zeros(T, l) for _ in 1:num_actions]
	episode_rewards, episode_steps, average_step_reward = differential_semi_gradient_sarsa!(parameters, mdp, max_episodes, max_steps, setup.value_function, setup.value_args, setup.parameter_update, setup.update_args; kwargs...)
	q̂(s, i_a) = setup.value_function(s, i_a, parameters, setup.value_args...)
	action_values = zeros(T, num_actions)
	q̂(s) = setup.value_function(action_values, s, parameters, setup.value_args...)
	return (value_function = q̂, reward_history = episode_rewards, step_history = episode_steps, average_step_reward = average_step_reward)
end

# ╔═╡ 49e43d51-05d6-415b-a685-76e50904c5bc
# ╠═╡ skip_as_script = true
#=╠═╡
function mountaincar_differential_test(max_episodes::Integer, α::Float32, β::Float32, ϵ::Float32; num_tiles = 12, num_tilings = 8, kwargs...)
	setup = setup_mountain_car_tiles((1f0/num_tiles, 1f0/num_tiles), num_tilings)
	v = setup.args.feature_vector
	run_linear_differential_semi_gradient_sarsa(mountain_car_differential_mdp, max_episodes, typemax(Int64), zeros(Float32, length(v)), setup.args.feature_vector_update; α = α, β = β, ϵ = ϵ, kwargs...)
end
  ╠═╡ =#

# ╔═╡ db189316-e880-4cc8-9070-ccfe2b4fc545
# ╠═╡ skip_as_script = true
#=╠═╡
(q̂_mountain_car2, episode_rewards2, episode_steps2, average_step_reward) = mountaincar_differential_test(100, 0.0008f0/8, 0.00001f0, 0.33f0; compute_value = compute_q_learning_value)
  ╠═╡ =#

# ╔═╡ 7bc49107-9de5-4985-8750-979f36b3aa81
#=╠═╡
π_mountain_car2(s) = argmax(i_a -> q̂_mountain_car2(s, i_a), eachindex(mountain_car_actions))
  ╠═╡ =#

# ╔═╡ ab4cb3db-3a2d-4145-826b-b1001114eeff
#=╠═╡
show_mountaincar_trajectory(π_mountain_car2, 1_000, "Differential Q-learning Learned Policy")
  ╠═╡ =#

# ╔═╡ 0e3e506d-1959-47fd-8da9-b3dfd294be67
#=╠═╡
plot_mountaincar_action_values(q̂_mountain_car2, 500, 500)
  ╠═╡ =#

# ╔═╡ 4775126e-4374-49be-b25f-4597401f3642
#=╠═╡
plot([scatter(y = view(average_step_reward, episode_steps2[2:end]), name = "Average Steps Reward"), scatter(y = (episode_steps2[2:end] .- episode_steps2[1:end-1])[:], name = "Steps Per Episode", yaxis = "y2")], Layout(yaxis2 = attr(overlaying = "y", side = "right")))
  ╠═╡ =#

# ╔═╡ 4271151d-d5a6-4a29-96c3-f2102b142b95
begin
	get_action_value(i_s::Integer, i_a::Integer, parameters::Vector{Vector{T}}) where T<:Real = parameters[i_a][i_s]
	get_action_value(i_s::Integer, i_a::Integer, parameters::Matrix{T}) where T<:Real = parameters[i_s, i_a]
end

# ╔═╡ 5534526a-d790-4379-98b0-8e4ee981fd9f
begin
	function update_action_values!(action_values::Vector{T}, i_s::Integer, parameters::Vector{Vector{T}}) where T<:Real
		maxvalue = typemin(T)
		maxindex = 1
		@inbounds @simd for i_a in eachindex(action_values)
			q = parameters[i_a][i_s]
			action_values[i_a] = q
			newmax = q > maxvalue
			maxvalue = newmax*q + !newmax*maxvalue
			maxindex = newmax*i_a + !newmax*maxindex
		end
		return maxvalue, maxindex
	end
	function update_action_values!(action_values::Vector{T}, i_s::Integer, parameters::Matrix{T}) where T<:Real
		maxvalue = typemin(T)
		maxindex = 1
		@inbounds @simd for i_a in eachindex(action_values)
			q = parameters[i_s, i_a]
			action_values[i_a] = q
			newmax = q > maxvalue
			maxvalue = newmax*q + !newmax*maxvalue
			maxindex = newmax*i_a + !newmax*maxindex
		end
		return maxvalue, maxindex
	end
end

# ╔═╡ f68952bd-4c0b-4331-a09c-5b118c8fa5a9
function state_aggregation_action_gradient_setup(assign_state_group::Function)
	function update_params!(parameters, s, i_a, g::T, α::T) where {T<:Real}
		i_s = assign_state_group(s)
		update_parameters!(parameters, i_s, i_a, g, α)
	end

	q̂(s, i_a, parameters) = get_action_value(assign_state_group(s), i_a, parameters)
	q̂(action_values::Vector{T}, s, parameters) where T<:Real = update_action_values!(action_values, assign_state_group(s), parameters)
		
	return (value_function = q̂, value_args = (), parameter_update = update_params!, update_args = ())
end

# ╔═╡ 62839b2a-398a-4445-87d1-b15ff2acc1d1
function create_access_control_task(num_servers::Integer, priority_payments::Vector{Float32})
	actions = [Accept(), Reject()]

	initialize_state() = AccessControlState(num_servers, rand(priority_payments))

	transition = StateMDPTransitionSampler((s, i_a) -> access_control_step(s, actions[i_a], num_servers, priority_payments), initialize_state())
	mdp = StateMDP(actions, transition, initialize_state, s -> false)
	states =  [AccessControlState(n, p) for n in 0:num_servers for p in priority_payments]
	assign_group(s::AccessControlState) = s.num_free_servers + 1 + (num_servers+1)*Int64(log2(s.top_priority))
	(mdp = mdp, gradient_setup = state_aggregation_action_gradient_setup(assign_group), num_groups = (num_servers+1) * length(priority_payments))
end

# ╔═╡ b4af8d87-a6e5-4e09-92b4-b07757f58f7f
function run_access_control_differential_sarsa(max_steps::Int64; num_servers = 10, priority_payments = [1f0, 2f0, 4f0, 8f0], kwargs...)
	(mdp, gradient_setup, num_groups) = create_access_control_task(num_servers, priority_payments)
	parameters = [zeros(Float32, num_groups) for _ in eachindex(mdp.actions)]
	state_representation = zeros(Float32, num_groups)
	(_, _, steprewards) = differential_semi_gradient_sarsa!(parameters, mdp, 1, max_steps, gradient_setup...; kwargs...)
	action_values = zeros(Float32, length(mdp.actions))
	v̂(num_free_servers::Int64, priority::Real) = gradient_setup.value_function(action_values, AccessControlState(num_free_servers, Float32(priority)), parameters)

	(value_function = v̂, mdp = mdp, parameters = parameters, steprewards = steprewards)
end

# ╔═╡ 546a775e-d3c9-4693-9f64-d4c47a84fb9f
# ╠═╡ skip_as_script = true
#=╠═╡
function figure_10_5(;numsteps = 2_000_000, α = 0.01f0, β = 0.01f0, ϵ = 0.1f0)
	access_control_output = run_access_control_differential_sarsa(numsteps; β = β, α = α, ϵ = ϵ)
	policy_output = BitArray(undef, (4, 10))
	priorities = [8, 4, 2, 1]
	actions = [true, false]
	value_function_outputs = [zeros(Float32, 11) for _ in 1:4]
	for num_free_servers in 0:10
		for priority in 1:4
			v, i_a = access_control_output.value_function(num_free_servers, priorities[priority])
			value_function_outputs[priority][num_free_servers+1] = v
			if num_free_servers > 0
				policy_output[priority, num_free_servers] = actions[i_a]
			end
		end
	end
	policy_trace = heatmap(x = 1:10, y = 1:4, z = Float32.(policy_output), colorscale="Greys", showscale = false)
	value_traces = [scatter(x = 0:10, y = value_function_outputs[i], name = "priority $(priorities[i])") for i in 1:4]
	p1 = plot(policy_trace, Layout(yaxis_tickvals = 1:4, yaxis_ticktext = priorities, xaxis_ticktext = 1:10, xaxis_tickvals = 1:10, xaxis_title = "Number of free servers", yaxis_title = "Priority", title = "Policy (black=reject, white=accept)"))
	p2 = plot(value_traces, Layout(xaxis_title = "Number of free servers", yaxis_title = "Differential value of best action", title = "Value Function"))

	md"""
	Figure 10.5

	The policy and value function found by differential semi-gradient one-step Sarsa on the access-control queuing task after 2 million steps.  The value learned for $\bar R$ was about $(access_control_output.steprewards[end-10000:end] |> mean |> Float64 |> x -> round(x, sigdigits = 3))
	$([p1 p2])
	"""
end
  ╠═╡ =#

# ╔═╡ 41c626c7-908d-4ff6-9730-4ad0b8c3cc25
#=╠═╡
figure_10_5()
  ╠═╡ =#

# ╔═╡ 662759be-282c-460b-adc3-8595475b53c2
md"""
## 10.4 Deprecating the Discounted Setting

In a special case of indistinguishable states, we can only use the actions and reward sequences to analyze a continuing task.  For a policy $\pi$, the average of the discounted returns with discount factor $\gamma$ is always $\frac{r(\pi)}{1-\gamma}$.  Therefore the *ordering* of all policies is independent of the discount rate and would match the ordering we get in the average reward setting.  This derivation however depends on states being indistinguishable allowing us to match up the weights on reward sequences from different policies.

We can use discounting in approximate solution methods regardless but then $\gamma$ changes from a problem parameter to a solution method parameter.  Unfortunately, discounting algorithms with function approximation do not optimize discounted value over the on-policy distribution, and thus are not guaranteed to optimze average reward.

The root cause of the problem applying discounting with function approximation is that we have lost the policy improvement theorem which states that a policy $\pi^\prime$ is better than policy $\pi$ if $v_{\pi^\prime}(s) \geq v_\pi(s) \forall s\in \mathcal{S}$.  Under this theorem we could take a deterministic policy, choose a specific state, and find a new action at that state with a higher expected reward than the current policy.  If the policy is an approximation function that uses states represented by feature vectors, then adjusting the parameters can in general affect the actions at many states including ones that have not been encountered yet.  In fact, with approximate solution methods we cannot guarantee  policy improvement in any setting.  Later we will introduce a theoretical guarantee called the "policy-gradient theorem" but for an alternative class of algorithms based on parametrized policies.
"""

# ╔═╡ 0e66a941-1ec1-4d3b-b064-e5f25cc93baf
md"""
### Connection to Chapter 3
"""

# ╔═╡ c316c5d3-f484-4e8e-bd56-be1e236d96bc
md"""
Applying the derivation of discount independence to the MDP in exercise 3.22 who's optimal policy depends on $\gamma$

$J(\pi) = \sum_s \mu_\pi(s)v_\pi^\gamma(s)$

Consider $\pi_{left}$: $J(\pi_{left})=0.5 \times (1 + 0 + \gamma^2 + 0 + \gamma^4 + 0 + \cdots) + 0.5 \times(0 + \gamma + 0 + \gamma^3 + 0 + \gamma^5 + \cdots)$
$J(\pi_{left}) = 0.5 \times (1 + \gamma + \gamma^2 + \gamma^3 + \gamma^4 + \gamma^5 + \cdots)$

Consider $\pi_{right}$: $J(\pi_{right})=0.5 \times (0 + 2\gamma + 0 + 2\gamma^3 + 0 + \cdots) + 0.5 \times(2 + 0 + 2\gamma^2 + 0 + 2\gamma^4 + \cdots)$
$J(\pi_{right}) = 0.5 \times 2 \times (1 + \gamma + \gamma^2 + \gamma^3 + \gamma^4 + \gamma^5 + \cdots)$

So both average reward values have the same factor for the discount rate and thus the right policy appears better since the average reward value is higher.  Previously, we had calculated that a discount rate less than 0.5 made the left policy favorable since the reward was obtained sooner going left vs right.  In the original problem we can consider the value of the top state for both left and right policies:
$v_{\pi_{left}} (top) = 1 + 0 + \gamma^2 + 0 + \gamma^4 + \cdots = 1 + \gamma^2 + \gamma^4 + \cdots$
$v_{\pi_{right}} (top) = 0 + 2\gamma + 0 + 2\gamma^3 + \cdots = 2 \times (\gamma + \gamma^3 + \cdots) = 2\gamma(v_{\pi_{left}}(top))$

Clearly for $\gamma > 0.5$ the right policy is better.

Similarly, we can consider the value of the left state for both left and right policies:
$v_{\pi_{left}} (left) = 0 + \gamma + 0 + \gamma^3 + \cdots = \gamma + \gamma^3 + \cdots$
$v_{\pi_{right}} (left) = 0 + 0 + 2\gamma^2 + 0  + 2\gamma^4 + \cdots = 2 \times (\gamma^2 + \gamma^4 + \cdots) = 2\gamma(v_{\pi_{left}}(left))$

Again, for $\gamma > 0.5$ the right policy is better.

And finally for the right state:
$v_{\pi_{left}} (right) = 2 + \gamma + 0 + \gamma^3 + 0 + \gamma^5 \cdots = 2+\gamma(1 + \gamma^2 + \gamma^4 + \cdots)=2 + \frac{\gamma}{1-\gamma^2}$ 
$= \frac{2(1-\gamma^2) + \gamma}{1-\gamma^2} = \frac{2 - 2\gamma^2 + \gamma}{1-\gamma^2}$
$v_{\pi_{right}} (right) = 2 + 0 + 2\gamma^2 + 0 + 2\gamma^4 +  \cdots = 2 \times (1+\gamma^2 + \gamma^4 + \cdots) = \frac{2}{1-\gamma^2}$

$\frac{v_{\pi_{left}} (right)}{v_{\pi_{right}} (right)}=\frac{2 - 2\gamma^2 + \gamma}{2}$

For $\gamma=0$ this quantity is 1 meaning the policies are equal and for $\gamma=1$ this quantity is 0.5 meaning that the right policy is better.  At $\gamma=0.5$ the quantity is $\frac{2 - 0.5 + 0.5}{2}=\frac{2}{2}=1$ meaning they are equal.  The maximum value occurs at $2\gamma = 0.5 \implies \gamma = 0.25$ with a ratio value of $\frac{2 - 0.125 + 0.25}{2}=\frac{2.125}{2}=1.0625$ meaning that the left policy is slightly better or equal from $0 \leq \gamma \leq 0.5$ and worse at $\gamma > 0.5$ which matches the earlier states.
"""

# ╔═╡ bc220d14-97fd-486d-9880-6908135fe036
md"""
The reason why the left policy can be better if $\gamma < 0.5$ in the original example is because it has a higher value in each state considered.  Consider $\gamma = 0.25$.  The left policy has the following approximate discounted value estimates for top, left, right: 

1.0667, 0.2667, 2.2667. 

Meanwhile the right policy has the corresponding values of: 

0.533, 0.133, 2.133.

Each value is smaller for the right policy.  However when we calculate the average value calculated over the long term distribution of states, the left policy averages the first two values while the right policy averages the first and third values because in the long run we expect the left policy to only exist in the top and left state while the right policy will exist in the top and right state.  Because the right state has such a high value for both policies but only the right policy includes it in the average it makes its entire objective estimate higher.  However, we can see that in the event of being in the right state, it is still a higher value expectation following the left policy in this case.  The decision to average based on the final distribution results in a policy ordering that doesn't match with what we know to be the optimal policy from the policy improvement theorem over finite states.
"""

# ╔═╡ 39eada35-8c3e-4ddc-8df9-7cf9f120928d
md"""
## 10.5 Differential Semi-gradient *n*-step Sarsa
"""

# ╔═╡ 8752c98d-fac1-4b3b-b20b-70acc0677fcb
md"""
> ### *Exercise 10.9* 
> In the differential semi-gradient n-step Sarsa algorithm, the step-size parameter on the average reward, $\beta$, needs to be quite small so that $\bar R$ becomes a good long-term estimate of the average reward. Unfortunately, $\bar R$ will then be biased by its initial value for many steps, which may make learning inefficient. Alternatively, one could use a sample average of the observed rewards for $\bar R$. That would initially adapt rapidly but in the long run would also adapt slowly. As the policy slowly changed, $\bar R$ would also change; the potential for such long-term nonstationarity makes sample-average methods ill-suited. In fact, the step-size parameter on the average reward is a perfect place to use the unbiased constant-step-size trick from Exercise 2.7. Describe the specific changes needed to the boxed algorithm for differential semi-gradient n-step Sarsa to use this trick.

At the start initialize $\bar o = 0$ and select $\lambda > 0$ small instead of $\beta$. 

Within the loop under the $\tau \geq 0$ line, add two lines; one to update $\bar o$ and one to calculate the update rate for the average reward: 

Line 1: $\bar o \leftarrow \bar o + \lambda (1 - \bar o)$

Line 2: $\beta = \lambda / \bar o$

As steps progress $\beta$ will approach $\lambda$ but early on will take on much larger values as $\bar o$ starts close to 0 and approaches 1.
"""

# ╔═╡ 50f6ff51-d81b-4e97-9f8a-0daf03af7192
md"""
## Monte Carlo Gradient Control
"""

# ╔═╡ 953907cd-9926-4478-99b3-da7068118c22
function gradient_monte_carlo_action_episode_update!(parameters, update_parameters!::Function, states::AbstractVector{S}, actions::AbstractVector{A}, rewards::AbstractVector{T}, γ::T, α::T, update_args...) where {T<:Real, S, A}
	g = zero(T)
	l = length(states)
	for i in l:-1:1
		s = states[i]
		i_a = actions[i]
		g = γ * g + rewards[i]
		update_parameters!(parameters, s, i_a, g, α, update_args...)
	end
end

# ╔═╡ 13e477f0-dc15-46cb-9691-c04a1b4c83c8
function gradient_monte_carlo_action_policy_estimation!(parameters, mdp::StateMDP, π::Function, γ::T, num_episodes::Integer, update_parameters!::Function, update_args::Tuple; α = one(T)/10, epkwargs...) where {T<:Real}
	(states, actions, rewards, sterm, nsteps) = runepisode(mdp; π = π, epkwargs...)
	gradient_monte_carlo_action_episode_update!(parameters, update_parameters!, states, actions, rewards, γ, α, update_args...)
	step_history = [nsteps]
	for ep in 2:num_episodes
		(states, actions, rewards, sterm, nsteps) = runepisode!((states, actions, rewards), mdp; π = π, epkwargs...)
		gradient_monte_carlo_episode_update!(parameters, update_parameters!, view(states, 1:nsteps), view(actions:1:nsteps), view(rewards, 1:nsteps), γ, α, update_args...)
		push!(step_history, nsteps)
	end
	return step_history, reward_history
end

# ╔═╡ 06834750-cc3a-468a-b0c2-81349c288a33
function gradient_monte_carlo_control!(parameters, mdp::StateMDP, γ::T, num_episodes::Integer, value_function::Function, value_args::Tuple, update_parameters!::Function, update_args::Tuple; α = one(T)/10, ϵ = one(T)/10, suppress_warning = false, epkwargs...) where {T<:Real}
	action_values = zeros(T, length(mdp.actions))
	function π_ϵ_greedy(s)
		rand() < ϵ && return rand(eachindex(mdp.actions)) 
		value_function(action_values, s, parameters, value_args...)[2]
	end
	(states, actions, rewards, sterm, nsteps) = runepisode(mdp; epkwargs...)
	gradient_monte_carlo_action_episode_update!(parameters, update_parameters!, states, actions, rewards, γ, α, update_args...)
	if !suppress_warning && !mdp.isterm(sterm) 
		@info "Warning: Episode 1 did not conclude in $nsteps steps"
	end
	step_history = [nsteps]
	reward_history = [sum(rewards)]
	for ep in 2:num_episodes
		(states, actions, rewards, sterm, nsteps) = runepisode!((states, actions, rewards), mdp; π = π_ϵ_greedy, epkwargs...)
		gradient_monte_carlo_action_episode_update!(parameters, update_parameters!, view(states, 1:nsteps), view(actions, 1:nsteps), view(rewards, 1:nsteps), γ, α, update_args...)
		if !suppress_warning && !mdp.isterm(sterm) 
			@info "Warning: Episode $ep did not conclude in $nsteps steps"
		end
		push!(step_history, nsteps)
		push!(reward_history, sum(rewards[i] for i in 1:nsteps))
	end
	return step_history, reward_history
end

# ╔═╡ c85033e1-3ee6-42ad-9ef0-144ce6238ce4
#=╠═╡
function smooth_error(error_history, n)
	l = length(error_history)
	[mean(error_history[i-n:i]) for i in n+1:l]
end
  ╠═╡ =#

# ╔═╡ 5c920177-8e46-49c9-9b95-1a657fdcae4e
# ╠═╡ skip_as_script = true
#=╠═╡
plot(scatter(y = -smooth_error(episode_rewards_dp, 1)), Layout(yaxis_type = "log"))
  ╠═╡ =#

# ╔═╡ 4ccb8a52-c6af-445d-a39e-d4d9b10c0d6a
# ╠═╡ skip_as_script = true
#=╠═╡
plot(scatter(y = -smooth_error(mountain_car_fcann.reward_history, 1)), Layout(yaxis_type = "log"))
  ╠═╡ =#

# ╔═╡ 09088eee-4cb3-40ac-b127-658ce1332fba
#=╠═╡
plot(scatter(y = -smooth_error(episode_rewards_q, 10)), Layout(yaxis_type = "log", xaxis_title = "Episode", yaxis_title = "Number of Steps"))
  ╠═╡ =#

# ╔═╡ 17d11fea-883b-4ddb-bec2-c4ad491b39dd
function run_linear_gradient_monte_carlo_control(mdp::StateMDP, γ::T, num_episodes::Integer, state_representation::AbstractVector{T}, update_state_representation!::Function; kwargs...) where T<:Real
	setup = linear_features_action_gradient_setup(mdp, state_representation, update_state_representation!)
	parameters = [zeros(T, length(state_representation)) for _ in 1:length(mdp.actions)]
	step_history, reward_history = gradient_monte_carlo_control!(parameters, mdp, γ, num_episodes, setup...; kwargs...)
	q̂(s) = setup.value_function(action_values, s, parameters, setup.value_args...)
	action_values = zeros(T, length(mdp.actions))
	π_greedy(s) = setup.value_function(action_values, s, parameters, setup.value_args...)[2]
	π_ϵ_greedy(s, ϵ) = rand() < ϵ ? rand(eachindex(mdp.actions)) : setup.value_function(action_values, s, parameters, setup.value_args...)[2]
	return (value_function = q̂, π_greedy = π_greedy, π_ϵ_greedy = π_ϵ_greedy, step_history = step_history, reward_history = reward_history)
end

# ╔═╡ 0714a1cf-9288-4f1e-ba72-d82608704d69
# ╠═╡ skip_as_script = true
#=╠═╡
mc_test = run_linear_gradient_monte_carlo_control(mountain_car_mdp, 1f0, 5_000, setup_mountain_car_tiles((1/12f0, 1/12f0), 9).args...; α = 1f-7, ϵ = 0.25f0, max_steps = 100_000)
  ╠═╡ =#

# ╔═╡ b76551e0-c027-4682-b5ae-bba7ea2b987a
# ╠═╡ skip_as_script = true
#=╠═╡
plot(smooth_error(mc_test.step_history, 10), Layout(yaxis_type = "log"))
  ╠═╡ =#

# ╔═╡ 954848db-6dcc-4666-90f8-b5a900203242
#=╠═╡
show_mountaincar_trajectory(s -> mc_test.π_ϵ_greedy(s, 0.25f0), 10000, "MC Learned Policy")
  ╠═╡ =#

# ╔═╡ b55d50a4-b039-4240-b434-42f7b724d24d
# ╠═╡ skip_as_script = true
#=╠═╡
plot_mountaincar_action_values(mc_test.value_function, 100, 100)
  ╠═╡ =#

# ╔═╡ c75dc51c-cbff-48b1-b0fd-108828929b51
function run_fcann_gradient_monte_carlo_control(mdp::StateMDP, γ::T, num_episodes::Integer, state_representation::AbstractVector{T}, update_state_representation!::Function, layers::Vector{Int64}; λ = 0f0, c = Inf, dropout = 0f0, kwargs...) where T<:Real
	setup = fcann_action_gradient_setup(mdp, layers, state_representation, update_state_representation!; λ = λ, c = c, dropout = dropout)
	num_actions = length(mdp.actions)
	step_history, reward_history = gradient_monte_carlo_control!(setup.parameters, mdp, γ, num_episodes, setup.value_function, setup.value_args, setup.parameter_update, setup.update_args; kwargs...)
	q̂(s, i_a) = setup.value_function(s, i_a, setup.parameters, setup.value_args...)
	action_values = zeros(Float32, num_actions)
	q̂(s) = setup.value_function(action_values, s, setup.parameters, setup.value_args...)
	π_greedy(s) = q̂(s)[2]
	π_ϵ_greedy(s, ϵ) = rand() < ϵ ? rand(eachindex(mdp.actions)) : π_greedy(s)
	return (value_function = q̂, π_greedy = π_greedy, π_ϵ_greedy = π_ϵ_greedy, step_history = step_history, reward_history = reward_history)
end

# ╔═╡ 6cea9e69-bf8c-4079-9884-663a728d7b08
md"""
# Dependencies
"""

# ╔═╡ f5e32900-6eb6-4b61-916d-893c0bcaf214
# ╠═╡ skip_as_script = true
#=╠═╡
TableOfContents()
  ╠═╡ =#

# ╔═╡ ed1bd92c-8cc7-457f-9692-a10a9487c953
# ╠═╡ skip_as_script = true
#=╠═╡
html"""
<style>
	main {
		margin: 0 auto;
		max-width: min(1600px, 90%);
		padding-left: max(10px, 5%);
		padding-right: max(10px, 5%);
		font-size: max(10px, min(24px, 2vw));
	}
</style>
"""
  ╠═╡ =#

# ╔═╡ dd472c0f-7b43-4abe-ada9-9dc8004a18cb
# ╠═╡ skip_as_script = true
#=╠═╡
begin
function add_elements(a, b)
	@htl("""
	$a
	$b
	""")
end
add_elements(a::HTML, b::HTML) = add_elements(a.content, b.content)
add_elements(a::HTML, b::AbstractString) = add_elements(a.content, b)
add_elements(a::AbstractString, b::HTML) = add_elements(a, b.content)
end
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Transducers = "28d57a85-8fef-5791-bfe6-a80928e7c999"

[compat]
BenchmarkTools = "~1.5.0"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.3.1"
PlutoDevMacros = "~0.9.0"
PlutoPlotly = "~0.5.0"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.60"
Statistics = "~1.11.1"
Transducers = "~0.4.84"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "f96742aa2407e40995be21bd65f67a020afa7644"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown"]
git-tree-sha1 = "b392ede862e506d451fc1616e79aa6f4c673dab8"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.38"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsDatesExt = "Dates"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsTestExt = "Test"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires"]
git-tree-sha1 = "e2144b631226d9eeab2d746ca8880b7ccff504ae"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.3"

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
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "cb25e4b105cc927052c2314f8291854ea59bf70a"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.2.4"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

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

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

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
version = "1.11.0"

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
git-tree-sha1 = "62ca0547a14c57e98154423419d8a342dca75ca9"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.4"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

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

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "2984284a8abcfcc4784d95a9e2ea4e352dd8ede7"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.36"

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
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

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
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

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
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

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
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProfileCanvas]]
deps = ["FlameGraphs", "JSON", "Pkg", "Profile", "REPL"]
git-tree-sha1 = "41fd9086187b8643feda56b996eef7a3cc7f4699"
uuid = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
version = "0.1.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

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
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

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
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

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
version = "1.11.0"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

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
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

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
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─35d59eae-77fd-11ef-2790-35dd5a834060
# ╟─b0265b93-ae5f-48f2-a9fd-44fd6115164b
# ╟─47710ddd-79d9-464d-b5dd-27180f2d6b31
# ╟─4d392303-4681-4ea1-8dcc-e002a78ea0a1
# ╟─a3cf270b-b309-44f0-9972-bd84228bcf17
# ╟─2f685ee2-6ad8-4bb1-b326-e5de7c15eb18
# ╟─14fe2253-cf2c-4159-a360-1e65f1c82b09
# ╟─6351304f-50ac-4755-86e1-cd4680f2d803
# ╟─e7bf61d7-c362-433d-9b83-6537d308c255
# ╠═fc0b88f3-fbf9-450d-b770-b34357ffad49
# ╠═991492f4-7dfc-43aa-ab6c-a6b1f3e38225
# ╠═8b7e1031-9864-439c-86eb-11aa08f53b90
# ╟─05e2fff5-4871-4468-a00e-9c1b7ba0ffc6
# ╟─57a6510f-bd42-4d1d-a550-d1442f79569f
# ╠═b0761704-5447-4e64-8270-708d9dccef60
# ╟─54b92594-04b8-4a8a-82c2-773b4a24680d
# ╟─278a26ac-c48f-4e18-93bb-706a4634c8c0
# ╠═5f8b0254-88f4-4d19-ade1-8e7c40941b43
# ╟─c5c839f7-1806-463d-b63a-bd7e1384f203
# ╠═f11787a1-57f8-4077-8d60-bc760ece7cc6
# ╟─1478745a-634d-4f31-8a70-b74f0e536201
# ╠═c697e0b6-d3e4-4f5f-96e9-b9486c9d7efc
# ╟─dc2cffeb-9adf-4956-afa3-ac82af377c59
# ╠═b9ebd6bb-90a1-4945-85ed-023206e2420a
# ╠═2c620fe4-2f62-40f8-a666-8dced1e0b84a
# ╠═56b0d69b-b7c3-4365-9b02-e0d5e8a85f94
# ╟─8d096d0d-8fea-421a-aa33-82269d3fe7e2
# ╠═f58cd0a2-8c82-46b2-bb8f-00f6aa1d867f
# ╠═c678846c-aaff-4266-8b2c-07a3a92445ef
# ╠═39b0f100-a7e7-4633-b296-ccc87a5a35ab
# ╠═00e7783f-7f17-4944-a085-ea87509cd75a
# ╟─a22e5d34-4b8d-479c-985c-d6abd41a6c80
# ╟─b990ba67-42c8-4ab9-943d-085392204fdd
# ╠═cafb20b4-a2bd-46a9-9660-b0ace84d6e4c
# ╠═b07460f1-0461-4f63-b145-c4e1818a497e
# ╠═df07524f-b3fe-4a66-98ac-8f80df66bcff
# ╠═d577b393-4b40-4c90-9993-4ffbcbd9df6d
# ╠═28e0d632-0df3-4a5b-85c4-571c845ff827
# ╟─8befede5-378a-447a-96bd-edcd9d2ce98b
# ╠═ac80958a-73ec-4342-b553-b33df6612a50
# ╠═f9abf433-8ce3-4520-9b47-f17f8f07b4cb
# ╠═1e9c537a-a731-4b81-8f6a-cb658b52c5be
# ╠═5b2ffd90-ead0-42ce-999a-584ed8995910
# ╟─f6e08689-040f-4565-9dfb-e9a65d1c1f18
# ╟─528533f7-68f1-4d19-9a37-6d4d0d7c38e2
# ╟─d42bb733-07e2-4932-aab4-09229ff67492
# ╟─afee7bc9-aff0-4c71-a227-9845cb23d4e9
# ╟─864450b9-1319-4426-961f-ee6df93463d8
# ╠═cc9197e0-f5bd-4742-bea3-b54e0b8e3b93
# ╟─ca970333-fa08-412c-b89d-491e70f0ac79
# ╟─e86bc86f-9909-458d-b86d-0a4ac4b9d43d
# ╠═b5273dfa-2262-487a-856b-441f007bd163
# ╠═dae59fd9-0397-4307-afd8-bafb6f0bfa52
# ╟─d291541d-ddba-4b71-a4eb-37fef758b71b
# ╠═12f5065b-5bed-4d03-a0f0-72a942492394
# ╠═1d417a66-205f-4883-b49c-a6fc900af4ce
# ╠═7e8c89aa-8a5e-4ff4-afd2-df8f5c77b5b2
# ╠═9d65285f-d49e-40ce-acea-1f565bcd4108
# ╠═1b15efa9-c331-46bf-93db-f96dee026fe2
# ╠═e338be2b-05f1-43f4-a194-45ffd710777e
# ╠═e48af9f4-0b47-4a45-b0ad-8f53b094e712
# ╟─39c63495-36c3-4e62-b8fb-36865f2c6243
# ╟─33ea5f09-3a1f-476d-875a-1f3635a40295
# ╠═99e3ec39-24f0-43d6-b6fd-9910b738ce2c
# ╟─72f575ee-d656-4af6-bf78-aab42bf1debd
# ╠═57ea3538-33be-4673-b914-8191d35426a9
# ╠═57659c52-de1b-46e6-a863-8eeec0cee601
# ╟─a97e3b12-b7a5-4f88-bdb9-c3158203e0ff
# ╟─cbf1e5ed-8308-486e-a9b7-6cf7fb441fe3
# ╟─c799ffe4-f4af-487d-b557-8b50d13632b7
# ╟─be77b538-d106-4ca0-a974-289415588c47
# ╠═3c300a2b-4139-4df0-906b-4cae3592cc2b
# ╟─e2cd69c5-eda7-4897-9e64-0adf940d4d96
# ╟─66d6a4b0-ddf8-4781-b3b4-20f02b25199a
# ╟─78087a57-33a0-4581-81de-926476090931
# ╠═58a0b622-1b51-4b42-a416-24109ae41a90
# ╠═5bc2eda5-5f4c-4165-9afb-16920f30b0c5
# ╠═8a5d9e3d-e8ef-4cea-8cd8-6975f797d7bd
# ╠═d8d5db17-d89c-47db-b258-6ad1635478b7
# ╠═1054cfa3-9f58-4a93-a318-c2d21cf23220
# ╟─1a5acfb0-3b35-41b1-98f8-ffce941c587f
# ╠═742100ba-c38e-4840-8988-40990039b527
# ╠═7c5fb569-81f0-4b70-ae95-1fce0c51b6f4
# ╠═30ab21ba-3f5b-46a8-8b8c-753f2755d419
# ╠═4afbb723-340b-4d85-9115-027a0ff8dfad
# ╠═f2201afe-8952-4dde-9e39-02beeb920f6f
# ╠═c1388562-0708-4a6a-acfe-927413dab5d2
# ╠═ddcb50be-5287-47f8-89f9-58c026a6b151
# ╟─af97f222-08d1-4200-a10b-8da178182175
# ╟─224b4bec-9ec5-434d-a950-f5974cd786d0
# ╠═b0cc6ff8-7296-461c-9db7-e52fa518e2e2
# ╠═d0cf3806-05c6-4a50-94c8-55c9042d51b7
# ╠═bd1f42e5-94cc-4aef-b82a-9bffd1c951d8
# ╠═7d21c4cd-ab79-4f40-9b8b-f637b3efcab0
# ╠═5c920177-8e46-49c9-9b95-1a657fdcae4e
# ╟─31fb07d2-1c34-44ec-b932-a598e78ec8dc
# ╠═c12070a9-df63-4b25-99e6-26ff876af1b4
# ╠═0f958535-6b18-46de-a1ba-81f64c217ee0
# ╠═ee59176e-24b6-4213-8f8e-759a70bc1d5e
# ╠═b3658e4d-ee8e-45cd-906a-06dd512a6921
# ╠═1e224a46-91ef-4a5f-ae35-ef4062147f2d
# ╠═00399548-b21c-43b5-90e2-30656ab1541e
# ╠═1a82ae95-3c3e-4281-bc1d-9eb19bf50286
# ╠═5db29488-a150-42ee-aedb-380a3a4fd548
# ╠═2f0d0a71-c65b-4aa0-a493-e7cdccd901eb
# ╟─7a47a518-dfc7-4310-a0b5-6f0d151c8263
# ╠═c11aa069-93c2-435a-8f0e-353ced9633b6
# ╠═5fdbce61-ca25-45e0-b07d-94adf7138446
# ╠═b9125c5b-01d6-451e-84b5-a419e38425b5
# ╠═4ccb8a52-c6af-445d-a39e-d4d9b10c0d6a
# ╠═fc3e0577-45aa-4bba-a275-fa7a352fc5cc
# ╠═5cbaeb8e-bc02-47c9-87b4-57df554cea9d
# ╟─59ec5223-f23f-4f32-9e5f-8a08e450da85
# ╟─49249ac1-8964-4afc-89f2-3cd4d4322cc2
# ╟─e1abf8c7-06b8-4cd5-b557-1d187004bdf1
# ╟─98a5d65e-4253-4523-a74e-99d03be03b89
# ╠═8ed6f8fd-8574-4d5a-9964-ce8a32629c6f
# ╠═1410db13-4b73-4a87-af34-30a5232af4ba
# ╠═7812e801-70fd-4331-ad0a-fad02c1a399f
# ╠═5be866c3-0fb2-4d1f-9c31-b85aba332905
# ╟─f7410fe7-e3d8-4047-8fa7-f076476e9d3a
# ╠═cbac1927-b087-4c4c-98ae-6aa5f0b824ad
# ╠═b5409b69-a254-4355-b2b9-99394eceb2f7
# ╠═f9ee13e8-7406-4fba-9a30-1e2714bd7cfc
# ╟─09088eee-4cb3-40ac-b127-658ce1332fba
# ╟─d6ad1ff1-8fbf-4799-8b1b-ae1e3ce88c5b
# ╟─b8c031ca-7995-4501-a1e3-df3f34e5f0da
# ╟─69a06405-57cd-42e5-96b1-5cc77d74aa03
# ╠═a9fdb1fd-3f62-4e1c-9157-c4eee6215261
# ╠═565c53ee-7ad5-44e2-bce5-4ff1f5f162c0
# ╠═c9759bd9-ec9b-47a1-9080-a7fc332be565
# ╠═065b2626-01f1-443f-8be4-3036003a2772
# ╠═b5d2776f-4b93-4eaa-8873-c1c4e610e6b0
# ╟─1a7ba296-52ca-4069-85fa-792d08d77b0e
# ╠═eb28458f-b222-4f8e-9a5b-8203d3997f7b
# ╠═e5ad765a-341f-4f11-9ae8-37d81cb349d2
# ╠═bc1d7cce-c0f4-47a8-b674-8acb82491c7f
# ╠═49e43d51-05d6-415b-a685-76e50904c5bc
# ╠═db189316-e880-4cc8-9070-ccfe2b4fc545
# ╠═7bc49107-9de5-4985-8750-979f36b3aa81
# ╠═ab4cb3db-3a2d-4145-826b-b1001114eeff
# ╠═0e3e506d-1959-47fd-8da9-b3dfd294be67
# ╠═4775126e-4374-49be-b25f-4597401f3642
# ╠═b094bf9f-bb97-4f23-acdc-f39411a07fb9
# ╠═d3ba78fa-f032-4bb9-9359-ef3bcff2252d
# ╠═ae5c5377-8b44-4c82-a63c-d2cb8a0d6667
# ╠═425fe768-c7bb-4d3e-87e6-47fa052ba612
# ╠═b191d3f9-cf25-4fb4-8f5a-8da86e96e125
# ╠═c44dd6c6-8213-49fb-8d33-ba8f2c766b2e
# ╟─9df1a18d-137c-4ea5-8d15-05697f7bbf07
# ╟─0c7f5742-6c51-4c6a-b67f-217163935ba5
# ╟─a6c5ec28-b2d5-4893-a118-95c1318d1f7f
# ╟─44d43dbf-fe32-438e-b89d-c677bbc35893
# ╟─f1edb500-fbd1-4c03-b033-53860dfa452d
# ╟─2d7679ad-a9b3-448b-a4bc-7e5b9bce6adb
# ╟─a9b74949-9392-4048-bcb6-5fd48c1d9b98
# ╠═fbf1c64f-1979-4384-a8c6-dc7875174d1f
# ╠═e7372e2b-a2db-4a93-9efc-f75aa74c197b
# ╠═014339eb-5b23-4ac5-a551-8eeb2238366f
# ╠═62839b2a-398a-4445-87d1-b15ff2acc1d1
# ╠═96548352-cd4d-4448-8312-ed10057f4359
# ╠═4271151d-d5a6-4a29-96c3-f2102b142b95
# ╠═5534526a-d790-4379-98b0-8e4ee981fd9f
# ╠═f68952bd-4c0b-4331-a09c-5b118c8fa5a9
# ╠═b4af8d87-a6e5-4e09-92b4-b07757f58f7f
# ╠═41c626c7-908d-4ff6-9730-4ad0b8c3cc25
# ╠═546a775e-d3c9-4693-9f64-d4c47a84fb9f
# ╟─662759be-282c-460b-adc3-8595475b53c2
# ╟─0e66a941-1ec1-4d3b-b064-e5f25cc93baf
# ╟─c316c5d3-f484-4e8e-bd56-be1e236d96bc
# ╟─bc220d14-97fd-486d-9880-6908135fe036
# ╟─39eada35-8c3e-4ddc-8df9-7cf9f120928d
# ╟─8752c98d-fac1-4b3b-b20b-70acc0677fcb
# ╟─50f6ff51-d81b-4e97-9f8a-0daf03af7192
# ╠═953907cd-9926-4478-99b3-da7068118c22
# ╠═13e477f0-dc15-46cb-9691-c04a1b4c83c8
# ╠═06834750-cc3a-468a-b0c2-81349c288a33
# ╠═0714a1cf-9288-4f1e-ba72-d82608704d69
# ╠═c85033e1-3ee6-42ad-9ef0-144ce6238ce4
# ╠═b76551e0-c027-4682-b5ae-bba7ea2b987a
# ╠═954848db-6dcc-4666-90f8-b5a900203242
# ╠═b55d50a4-b039-4240-b434-42f7b724d24d
# ╠═17d11fea-883b-4ddb-bec2-c4ad491b39dd
# ╠═c75dc51c-cbff-48b1-b0fd-108828929b51
# ╟─6cea9e69-bf8c-4079-9884-663a728d7b08
# ╠═69fb26ed-763e-44ad-9b70-193e5a1a09b9
# ╠═3f7484b3-272d-410d-92b1-ca13e5d7a8b7
# ╠═318b398a-d8f2-4f39-a45d-fd9023961bf8
# ╠═9fb5dace-a799-4424-bcb3-8542e508dd4b
# ╠═f5e32900-6eb6-4b61-916d-893c0bcaf214
# ╠═ed1bd92c-8cc7-457f-9692-a10a9487c953
# ╠═dd472c0f-7b43-4abe-ada9-9dc8004a18cb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
