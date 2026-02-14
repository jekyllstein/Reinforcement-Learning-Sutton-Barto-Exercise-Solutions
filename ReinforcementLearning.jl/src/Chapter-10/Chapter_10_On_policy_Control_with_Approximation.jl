### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ bee124fd-5605-4512-833b-945ef77c056e
using PlutoDevMacros

# ╔═╡ 9fb5dace-a799-4424-bcb3-8542e508dd4b
# ╠═╡ show_logs = false
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI,PlutoPlotly, PlutoProfile, BenchmarkTools, LaTeXStrings, HypertextLiteral
	TableOfContents(depth=4)
end
  ╠═╡ =#

# ╔═╡ fa5fecfd-c039-4063-9acb-365a046e06f2
@only_in_nb begin
	PlutoDevMacros.@frompackage @raw_str(joinpath(@__DIR__, "..", "ApproximationUtils.jl")) import *
	include(joinpath(@__DIR__, "..", "Chapter-09", "Chapter_09_On-policy_Prediction_with_Approximation.jl"))
end

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

Below is an implementation of Semi-gradient Sarsa building from the algorithms in Chapter 9.  In addition to the state representation and a function to update it given a state, Sarsa also requires a gradient update for action values.  The linear function approximation version of this simplifies the required arguments greatly, needing only a state representation vector and update function.  
"""

# ╔═╡ d88ebdb9-47bc-478c-b471-804a02ad2acf
md"""
#### *Action-Value Utility Functions*

In Chapter 9, we established functions for value and gradient computations given all of the feature vector types.  Now we must extend those functions to apply to action value estimates where the parameter space is larger and so are the gradients.  For linear approximation I will rely on the state representation only and simply multiply the parameter space by the number of actions.  Since the gradient update only applies to the features corresponding to the selected action, I only ever need to store a vector of gradients which will only update the column in the parameter matrix for that index.
"""

# ╔═╡ 9043a684-6f16-48d0-83d4-2e00f9b7dbc2
"""
    LinearActionValueGradient{I <: Integer, V <: LinearFeatureVector}

Mutable storage for action-value function gradients in linear function approximation.

Stores the gradient with respect to parameters for a specific action, enabling efficient 
updates in action-value learning algorithms like SARSA and Q-learning.

# Type Parameters
- `I <: Integer`: Index type for action identification
- `V <: LinearFeatureVector`: Feature vector type for gradient storage (typically [`LinearFeatureVector`](@ref) or [`BinaryFeatureVector`](@ref))

# Fields
- `action_gradient::V`: Gradient vector ∇q̂(s,a) with respect to action-value parameters
- `action_index::I`: Index identifying which action this gradient corresponds to

# See Also
[`LinearFeatureVector`](@ref), [`BinaryFeatureVector`](@ref)
"""
mutable struct LinearActionValueGradient{I <: Integer, V <: LinearFeatureVector}
	action_gradient::V
	action_index::I
end

# ╔═╡ 3273ed4a-6787-4635-8399-65ddf65b31ea
begin
	#for the parameter matrix, each column corresponds to a different action and is the same length as the feature vector.  Each function will return the maximum q value and its action index as well since that will be needed later for some functions
	"""
	    update_linear_action_values!(action_values, x, w) -> (max_value, max_action_index)
	
	Computes all action-values q̂(s,a) for linear function approximation and returns the maximum.
	
	Updates the action-values vector in-place by computing q̂(s,a) = x'w[:,a] for each action a,
	where w is a parameter matrix with columns corresponding to actions. Returns both the maximum
	action-value and its index for use in control algorithms.
	
	# Type Parameters
	- `T <: Real`: Numeric type for action-values and parameters
	
	# Arguments
	- `action_values::Vector{T}`: Action-value storage to update (modified in-place)
	- `x`: Feature representation of current state
	- `w::Matrix{T}`: Parameter matrix (features × actions)
	
	# Returns
	- `(max_value, max_action_index)`: Tuple containing maximum q̂(s,a) and corresponding action index
	
	# See Also
	[`update_linear_value_gradient!`](@ref), [`update_params_with_gradient!`](@ref)
	
	# Methods
	
	## Dense Features
	```julia
	update_linear_action_values!(action_values::Vector{T}, x::Vector{T}, w::Matrix{T}) where T<:Real
	```
	Computes action-values using optimized BLAS matrix-vector multiplication.
	Uses `BLAS.gemv!` for efficient computation of all action-values simultaneously.
	
	- `action_values::Vector{T}`: Action-value vector to update
	- `x::Vector{T}`: Dense feature vector
	- `w::Matrix{T}`: Parameter matrix (features × actions)
	
	## Binary Features
	```julia
	update_linear_action_values!(action_values::Vector{T}, x::BinaryFeatureVector, w::Matrix{T}) where T<:Real
	```
	Computes action-values using sparse feature representation.
	Only sums parameters corresponding to active features, tracking maximum during computation.
	
	- `action_values::Vector{T}`: Action-value vector to update
	- `x::`[`BinaryFeatureVector`](@ref): Sparse binary feature representation
	- `w::Matrix{T}`: Parameter matrix (features × actions)
	
	## State Aggregation Features
	```julia
	update_linear_action_values!(action_values::Vector{T}, x::StateAggregationFeatureVector, w::Matrix{T}) where T<:Real
	```
	Computes action-values for state aggregation features.
	Directly accesses parameter matrix row corresponding to the active group.
	
	- `action_values::Vector{T}`: Action-value vector to update  
	- `x::`[`StateAggregationFeatureVector`](@ref): State aggregation feature representation
	- `w::Matrix{T}`: Parameter matrix (groups × actions)
	
	# Performance Notes
	- Dense method uses BLAS for optimal performance on large feature vectors
	- Binary and state aggregation methods avoid branching with branchless max computation
	- All methods update action-values in-place to minimize allocations
	- Matrix layout optimized for column-wise access patterns (features × actions)
	"""
	function update_linear_action_values!(action_values::Vector{T}, x::Vector{T}, w::Matrix{T}) where T<:Real
		BLAS.gemv!('T', one(T), w, x, zero(T), action_values)
		return findmax(action_values)
	end

	function update_linear_action_values!(action_values::Vector{T}, x::BinaryFeatureVector, w::Matrix{T}) where T<:Real
		maxq = typemin(T)
		i_a_max = 0
		for i_a in eachindex(action_values)
			q = zero(T)
			@inbounds @simd for i in 1:x.num_features
				j = x.active_features[i]
				q += w[j, i_a]
			end
			action_values[i_a] = q
			newmax = q > maxq
			maxq = maxq*!newmax + newmax*q
			i_a_max = i_a_max*!newmax + newmax*i_a
		end
		return (maxq, i_a_max)
	end

	function update_linear_action_values!(action_values::Vector{T}, x::StateAggregationFeatureVector, w::Matrix{T}) where T<:Real
		maxq = typemin(T)
		i_a_max = 0
		i = x.group_index
		for i_a in eachindex(action_values)
			q = w[i, i_a]
			action_values[i_a] = q
			newmax = q > maxq
			maxq = maxq*!newmax + newmax*q
			i_a_max = i_a_max*!newmax + newmax*i_a
		end
		return (maxq, i_a_max)
	end
end

# ╔═╡ 0226d8a3-bb22-4a32-9700-e234abf518a6
#for a linear function the gradient is just the feature vector and I can reuse all of the gradient update functions from chapter 9 with the added information of storing the action index

"""
    update_linear_value_gradient!(∇q̂::LinearActionValueGradient, x, i_a, value_params) -> LinearActionValueGradient

Updates action-value gradient storage with feature vector and action index.

Extends the existing linear gradient system to handle action-value functions by delegating
gradient computation to [`update_linear_value_gradient!`](@ref) and storing the action index.

# Arguments
- `∇q̂::LinearActionValueGradient`: Action-value gradient storage to update in-place
- `x::LinearFeatureVector`: Feature vector representing ∇q̂(s,a)
- `i_a::Integer`: Action index for this gradient
- `value_params`: Value function parameters (unused, maintains API consistency)

# Returns
- `LinearActionValueGradient`: The updated gradient storage (same as input `∇q̂`)

See [`update_linear_value_gradient!`](@ref) for complete documentation of the gradient system.
"""
function update_linear_value_gradient!(∇q̂::LinearActionValueGradient{I, V}, x::V, i_a::Integer, value_params) where {I <: Integer, V <: LinearFeatureVector}
	update_linear_value_gradient!(∇q̂.action_gradient, x, value_params)
	∇q̂.action_index = i_a
	return ∇q̂
end

# ╔═╡ 08c74b7d-7aa6-4085-a09b-b6191f8d098e
function update_linear_value_gradient!(∇q̂::LinearActionValueGradient{I, V}, action_values::Vector{T}, x::V, i_a::Integer, value_params) where {I <: Integer, V <: LinearFeatureVector, T<:Real}
	update_linear_value_gradient!(∇q̂, x, i_a, value_params)
	update_linear_action_values!(action_values, x, value_params)
end

# ╔═╡ 1393f7a6-05c7-48a3-96a9-130eb6d45937
#for the gradient update I need to use the action index to only update the columns of the parameter matrix that apply for linear approximation.  all of these implementations are for the case of linear approximation where I only store the gradient of the action values plus the action index and the linear approximation parameters are stored in a matrix
begin
	function update_params_with_gradient!(w::Matrix{T}, α::T, ∇w::Vector{T}, i_a::Integer) where {T<:Real}
		@inbounds @simd for i in eachindex(∇w)
			w[i, i_a] += α * ∇w[i]
		end
		return w
	end
	
	function update_params_with_gradient!(w::Matrix{T}, α::T, ∇w::BinaryFeatureVector, i_a::Integer) where {T<:Real}
		@inbounds @simd for i in 1:∇w.num_features
			j = ∇w.active_features[i]
			w[j, i_a] += α
		end
		return w
	end

	function update_params_with_gradient!(w::Matrix{T}, α::T, ∇w::StateAggregationFeatureVector, i_a::Integer) where {T<:Real}
		w[∇w.group_index, i_a] += α
		return w
	end

	update_params_with_gradient!(w::Matrix{T}, α::T, ∇w::LinearActionValueGradient) where T<:Real = update_params_with_gradient!(w, α, ∇w.action_gradient, ∇w.action_index)
end

# ╔═╡ 7956c5af-4f26-4790-88c6-9e4d2a7b37f7
#add gpu option toggle for action value functions too

# ╔═╡ 585140d8-4c17-4adb-999c-ef4a72ab07b7
begin
	function form_action_value_kwargs(mdp, feature_vector, parameters::Array{T, N}) where {T<:Real, N} 
		(action_values = zeros(T, length(mdp.actions)), feature_vector = deepcopy(feature_vector), parameters = parameters)
	end

	function form_action_value_kwargs(mdp, feature_vector, parameters1::Array{T, N}, parameters2::Array{T, N}) where {T<:Real, N} 
		(action_values1 = zeros(T, length(mdp.actions)), action_values2 = zeros(T, length(mdp.actions)), feature_vector = deepcopy(feature_vector), parameters1 = parameters1, parameters2 = parameters2)
	end

	function form_action_value_kwargs(mdp, feature_vector, parameters::FCANNParams{T}) where {T<:Real} 
		(action_values = zeros(T, length(mdp.actions)), feature_vector = deepcopy(feature_vector), parameters = parameters, activations = FCANN.form_activations(parameters.weights[1]))
	end

	function form_action_value_kwargs(mdp, feature_vector::Vector{T}, cpu_params::FCANNParams{T}, gpu_params::FCANNParamsGPU) where {T<:Real} 
		activations = FCANN.form_activations(gpu_params.weights[1])
		d_x = FCANN.cuda_allocate(feature_vector)
		function cleanup_vars()
			FCANN.clear_gpu_data(activations)
			FCANN.clear_gpu_data([d_x])
		end
		gpu_kwargs = (activations = activations, d_x = d_x, cleanup_vars = cleanup_vars)
		(action_values = zeros(T, length(mdp.actions)), feature_vector = deepcopy(feature_vector), parameters = cpu_params, activations = FCANN.form_activations(cpu_params.weights[1]), gpu_kwargs = gpu_kwargs)
	end

	function form_action_value_kwargs(mdp, feature_vector, parameters1::FCANNParams{T}, parameters2::FCANNParams{T}) where {T<:Real} 
		(action_values1 = zeros(T, length(mdp.actions)), action_values2 = zeros(T, length(mdp.actions)), feature_vector = deepcopy(feature_vector), parameters1 = parameters1, parameters2 = parameters2, activations = FCANN.form_activations(parameters1.weights[1]))
	end
end

# ╔═╡ fc0b88f3-fbf9-450d-b770-b34357ffad49
#in normal sarsa, we use the action value of the action actually taken, later on with methods like expected sarsa we would actually use the policy vector compute a weighted average of all action values
"""
    compute_sarsa_value(action_values, policy, i_a) -> Real

Computes the SARSA target value using the action-value of the selected action.

For standard SARSA, the target value is simply the action-value q̂(s',a') of the action
actually taken. This function can be passed to control algorithms like 
[`semi_gradient_sarsa!`](@ref) to specify the target computation method.

# Arguments
- `action_values::Vector{T}`: Action-values q̂(s',a) for next state
- `policy::Vector{T}`: Policy probabilities (unused in standard SARSA)
- `i_a::Integer`: Index of action actually taken

# Returns
- `T<:Real`: Action-value q̂(s',a') for the selected action

# Algorithm Details
Returns `action_values[i_a]` for standard SARSA behavior. Alternative methods like
Expected SARSA would compute weighted averages using the policy probabilities.
"""
compute_sarsa_value(action_values::Vector{T}, policy::Vector{T}, i_a::Integer) where T<:Real = action_values[i_a]

# ╔═╡ f9057d17-00fe-4cc9-83a1-fef34c116b25
md"""
#### *Vanilla Implementation*
"""

# ╔═╡ 05e2fff5-4871-4468-a00e-9c1b7ba0ffc6
md"""
### Semi-gradient Dynamic Programming

Typically, to solve the control problem we require action-value estimates.  Even in the non-tabular case; however, we may have access to the transition distribution of an MDP.  A simple example of this is a deterministic problem in which there may be an uncountable number of states, but the transition dynamics are known exactly.  In this case, one can reconstruct the action values from the state values as follows:

$\hat q(s, a) = \sum_{s^\prime, r}p(s^\prime, r \vert s, a)\left (r + \gamma \hat v(s^\prime) \right ) = r(s, a) + \gamma \sum_{s^\prime}p(s^\prime \vert s, a) \hat v(s^\prime)$

In the case of a deterministic problem there is only one transition state $s^\prime = t(s, a)$ where $t$ is the deterministic mapping function.  Then the formula simplifies to $\hat q(s, a) = r(s, a) + \hat v(t(s, a))$.  We can update the parameters $\mathbf{w}$ for some value function $\hat v(s, \mathbf{w})$ using the techniques in Chapter 9.  Then, to derive the greedy policy, we can use 

$\pi(s) = \text{argmax}_a \left [ r(s, a) + \gamma \sum_{s^\prime}p(s^\prime \vert s, a) \hat v(s^\prime) \right ]$

From tabular dynamic programming, we have the following update rule for the optimal state value function:

$v_*(s) = \max_a \left [ r(s, a) + \gamma \sum_{s^\prime}p(s^\prime \vert s, a) v_*(s^\prime) \right ]$

If the right side expression uses the approximate value function, then it is available to use as an update target instead of the usual Sarsa one.
"""

# ╔═╡ 57a6510f-bd42-4d1d-a550-d1442f79569f
md"""
### *Semi-gradient Dynamic Programming Implementation*
"""

# ╔═╡ bf980179-911b-4dd6-8abe-16f6e497a0bc
begin
	#by default fill in the columns of the feature matrix with the feature vector
	function update_feature_matrix!(feature_matrix::Matrix{T}, feature_vector::Vector{T}, i_a::Integer) where T<:Real
		@inbounds @simd for i in eachindex(feature_vector)
			feature_matrix[i, i_a] = feature_vector[i]
		end
		return feature_matrix
	end

	#for anything other than dense vectors just make a vector of the features
	function update_feature_matrix!(feature_matrix::Vector{V}, feature_vector::V, i_a::Integer) where V <: BinaryFeatureVector
		v = feature_matrix[i_a]
		update_binary_feature_vector!(v, feature_vector)
		return feature_matrix
	end
	#for anything other than dense vectors just make a vector of the features
	function update_feature_matrix!(feature_matrix::Vector{V}, feature_vector::V, i_a::Integer) where V <: StateAggregationFeatureVector
		v = feature_matrix[i_a]
		update_state_aggregation_feature_vector!(v, feature_vector)
		return feature_matrix
	end
end

# ╔═╡ 2ef47fe1-e082-406b-b131-5e2ae1bcb08b
begin
	#for linear approximation just compute all of the state values with a matrix-vector multiplication
	function update_state_values!(state_values::Array{T, N}, feature_matrix::Matrix{T}, parameters::Vector{T}, activations) where {N, T<:Real}
		LinearAlgebra.BLAS.gemm!('T', 'N', one(T), feature_matrix, parameters, zero(T), state_values)
		return state_values
	end

	#for non-linear approximation, use the forward pass function
	function update_state_values!(state_values::Array{T, N}, feature_matrix::Matrix{T}, parameters::FCANNParams{T}, activations) where {N, T<:Real}
		FCANN.forwardNOGRAD_base!(activations, parameters.weights..., feature_matrix, parameters.reslayers; input_orientation = 'T')
		state_values .= activations[end]
	end

	#for non-linear approximation, use the forward pass function
	function update_state_values!(state_values::Array{T, N}, feature_matrix::FCANN.CUDAArray, parameters::FCANNParamsGPU, activations::FCANNActivationsGPU) where {N, T<:Real}
		FCANN.forwardNOGRAD_base!(activations, parameters.weights..., feature_matrix, parameters.reslayers; input_orientation = 'T')
		FCANN.memcpy!(state_values, activations[end])
	end

	#for non-linear approximation, use the forward pass function
	function update_state_values!(state_values::Array{T, N}, feature_matrix::Vector{V}, parameters::FCANNParams{T}, activations) where {N, V<:AbstractBinaryFeatures, T<:Real}
		FCANN.forwardNOGRAD_base!(activations, parameters.weights..., feature_matrix, parameters.reslayers)
		state_values .= activations[end]
	end
end

# ╔═╡ a4c6a5c0-29c5-440c-bf86-20d0f881ee06
begin
	"""
	    update_action_values!(action_values, s, feature_vector, update_feature_vector!, value_function, parameters, mdp, γ; kwargs...) -> (max_value, max_action_index)
	
	Computes action-values from state-value function using Bellman equation with transition distributions.
	
	Updates the action-values vector by computing q̂(s,a) = Σ p(s',r|s,a)[r + γv̂(s')] for each action,
	using the MDP's transition distribution. Returns the maximum action-value and its index for 
	use in dynamic programming control algorithms.
	
	# Type Parameters
	- `T <: Real`: Numeric type for action-values and computations
	
	# Arguments
	- `action_values::Vector{T}`: Action-value storage to update (modified in-place)
	- `s`: Current state
	- `feature_vector`: Serves as input to value function
	- `update_feature_vector!::Function`: Updates a feature vector in place with a state - `update_feature_vector!(feature_vector, s) -> feature_vector` 
	- `value_function::Function`: State-value function that operates on feature vector `value_function(feature_vector, params) -> v̂
	- `mdp::StateMDP`: Markov Decision Process with transition distributions
	- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
	- `kwargs::NamedTuple`: Additional keyword arguments to pass into value function
	
	# Returns
	- `(max_value, max_action_index)`: Tuple containing maximum q̂(s,a) and corresponding action index
	
	# See Also
	[`update_linear_action_values!`](@ref), [`semi_gradient_dp!`](@ref), [`StateMDP`](@ref)
	
	# Algorithm Details
	1. For each action a:
	   - Query transition distribution to get (rewards, next_states, probabilities)
	   - Compute expected immediate reward: r̄ = Σ p(s',r|s,a) · r
	   - Compute expected future value: v̄' = Σ p(s'|s,a) · v̂(s') (excluding terminal states)
	   - Set q̂(s,a) = r̄ + γ · v̄'
	2. Track maximum action-value during computation using branchless operations
	3. Return maximum value and corresponding action index
	
	# Performance Notes
	- Uses branchless maximum tracking to avoid conditional branching
	- Handles terminal states by excluding them from value computation
	- Compatible with any state-value function that accepts individual states
	"""
	function update_action_values!(action_values::Array{T, N}, s, feature_vector, update_feature_vector!::Function, value_function::Function, parameters, mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T; kwargs...) where {T<:Real, S, A, P<:StateMDPTransitionDistribution, F1<:Function, F2<:Function, F3<:Function, N}
		maxq = typemin(T)
		i_a_max = 0
		for i_a in eachindex(action_values)
			(rewards, states, probabilities) = mdp.ptf.step(s, i_a)
			v′ = zero(T) 
			r_avg = zero(T)
			for i in eachindex(probabilities)
				s′ = states[i]
				if !mdp.isterm(s′)
					update_feature_vector!(feature_vector, s′)
					v̂ = value_function(feature_vector, parameters; kwargs...)
					v′ += probabilities[i] * v̂
				end
				r_avg += probabilities[i]*rewards[i]
			end
			q = r_avg + γ*v′
			action_values[i_a] = q
			newmax = q > maxq
			maxq = newmax*q + !newmax*maxq
			i_a_max = newmax*i_a + !newmax*i_a_max
		end
		return maxq, i_a_max
	end

	function update_action_values!(action_values::Array{T, N}, s, feature_vector::V, update_feature_vector!::Function, value_function::Function, parameters::Vector{T}, mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T; kwargs...) where {T<:Real, S, A, P<:StateMDPTransitionDeterministic, F1<:Function, F2<:Function, F3<:Function, V<:Union{BinaryFeatureVector, StateAggregationFeatureVector}, N}
		maxq = typemin(T)
		i_a_max = 0
		for i_a in eachindex(action_values)
			r, s′ = mdp.ptf(s, i_a)
			update_feature_vector!(feature_vector, s′)
			v̂ = value_function(feature_vector, parameters; kwargs...)
			q = r + γ*v̂
			action_values[i_a] = q
			newmax = q > maxq
			maxq = newmax*q + !newmax*maxq
			i_a_max = newmax*i_a + !newmax*i_a_max
		end
		return maxq, i_a_max
	end

	function update_action_values!(action_values::Array{T, N}, s, feature_vector, update_feature_vector!::Function, value_function::Function, parameters, mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, reward_values::Vector{T}, feature_matrix, activations; kwargs...) where {T<:Real, S, A, P<:StateMDPTransitionDeterministic, F1<:Function, F2<:Function, F3<:Function, N}
		for i_a in eachindex(action_values)
			r, s′ = mdp.ptf.step(s, i_a)
			update_feature_vector!(feature_vector, s′)
			update_feature_matrix!(feature_matrix, feature_vector, i_a)
			reward_values[i_a] = r #populate action value vector with reward, will be added to the future state value later
		end
		update_state_values!(action_values, feature_matrix, parameters, activations)
		action_values .= reward_values .+ γ .* action_values
		maxq, imax = findmax(action_values)
		isinf(maxq) && @warn "Infinite action value found in state $s out of $action_values"
		isnan(maxq) && @warn "NaN action value found in state $s out of $action_values"
		return maxq, prod(Tuple(imax)) 
	end

	function update_action_values!(action_values::Array{T, N}, s, feature_vector::Vector{T}, update_feature_vector!::Function, value_function::Function, parameters::FCANNParamsGPU, mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, reward_values::Vector{T}, feature_matrix::Matrix{T}, gpu_matrix::FCANN.CUDAArray, activations; kwargs...) where {T<:Real, S, A, P<:StateMDPTransitionDeterministic, F1<:Function, F2<:Function, F3<:Function, N}
		for i_a in eachindex(action_values)
			r, s′ = mdp.ptf.step(s, i_a)
			update_feature_vector!(feature_vector, s′)
			update_feature_matrix!(feature_matrix, feature_vector, i_a)
			reward_values[i_a] = r #populate action value vector with reward, will be added to the future state value later
		end
		FCANN.memcpy!(gpu_matrix, feature_matrix)
		update_state_values!(action_values, gpu_matrix, parameters, activations)
		action_values .= reward_values .+ γ .* action_values
		maxq, imax = findmax(action_values)
		isinf(maxq) && @warn "Infinite action value found in state $s out of $action_values"
		isnan(maxq) && @warn "NaN action value found in state $s out of $action_values"
		return maxq, prod(Tuple(imax))
	end
end

# ╔═╡ 97e56e3f-1ef7-45a5-8261-c8fa103b9747
begin
	form_action_value_args(mdp, feature_vector, parameters) = ()

	function form_action_value_args(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector::Vector{T}, parameters) where {T<:Real, S, A, P <: StateMDPTransitionDeterministic, F1, F2, F3}
		num_actions = length(mdp.actions)
		reward_values = zeros(T, num_actions)
		feature_matrix = zeros(T, length(feature_vector), num_actions)
		(reward_values, feature_matrix, nothing)
	end

	function form_action_value_args(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector::Vector{T}, parameters::FCANNParams{T}) where {T<:Real, S, A, P <: StateMDPTransitionDeterministic, F1, F2, F3}
		num_actions = length(mdp.actions)
		reward_values = zeros(T, num_actions)
		feature_matrix = zeros(T, length(feature_vector), num_actions)
		activations = FCANN.form_activations(parameters.weights[1], num_actions)
		(reward_values, feature_matrix, activations)
	end

	function form_action_value_args(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector::Vector{T}, parameters::FCANNParamsGPU) where {T<:Real, S, A, P <: StateMDPTransitionDeterministic, F1, F2, F3}
		num_actions = length(mdp.actions)
		reward_values = zeros(T, num_actions)
		feature_matrix = zeros(T, length(feature_vector), num_actions)
		gpu_matrix = FCANN.cuda_allocate(feature_matrix)
		activations = FCANN.form_activations(parameters.weights[1], num_actions)
		(reward_values, feature_matrix, gpu_matrix, activations)
	end

	function form_action_value_args(mdp::StateMDP{T, S, A, P, F1, F2, F3}, feature_vector::V, parameters::FCANNParams) where {T<:Real, S, A, P <: StateMDPTransitionDeterministic, F1, F2, F3, V<:AbstractBinaryFeatures}
		num_actions = length(mdp.actions)
		reward_values = zeros(T, num_actions)
		feature_matrix = Vector{V}(undef, num_actions)
		activations = FCANN.form_activations(parameters.weights[1], num_actions)
		for i in 1:num_actions
			feature_matrix[i] = deepcopy(feature_vector)
		end
		(reward_values, feature_matrix, activations)
	end
end	

# ╔═╡ 94fa7f7d-c77c-4df5-a7b9-b3c931cb3bce
begin
	"""
	    form_value_function(mdp, update_feature_vector!, update_action_values!, feature_vector, parameters) -> Function
	
	Creates action-value function q̂(s) for control algorithms with direct parametric computation.
	
	Returns a closure that computes action-values for any state, finds the maximizing action, and provides
	structured output for use in control algorithms like SARSA and Q-learning.
	
	# Type Parameters
	- `T <: Real`: Numeric type for action-values and parameters
	- `S`: State type from MDP
	- `A`: Action type from MDP
	- `V`: Feature vector type
	- `W`: Parameter type (Vector or Matrix)
	
	# Arguments
	- `mdp::`[`StateMDP`](@ref): Markov Decision Process structure
	- `update_feature_vector!::Function`: Function to extract features from states
	- `update_action_values!::Function`: Function to compute action-values from features and parameters
	- `feature_vector::V`: Template feature vector for allocation
	- `parameters::W`: Value function parameters
	
	# Returns
	- `Function`: Action-value function q̂(s; kwargs...) with signature:
	  ```julia
	  q̂(s; action_values=zeros(T, length(mdp.actions)), x=deepcopy(feature_vector), parameters=parameters, action_value_kwargs...)
	  ```
	  Returns: `(action_values = action_values, maximizing_action = i_a_max, maximizing_value = maxq)`
	
	# See Also
	[`StateMDP`](@ref), [`form_state_value_function`](@ref)
	
	# Methods
	
	## State-Value Based Computation
	```julia
	form_value_function(mdp, γ, update_feature_vector!, value_function, feature_vector, parameters)
	```
	Creates action-value function by computing from state-value function using Bellman equation.
	Computes action-values via expected returns q̂(s,a) = Σ p(s',r|s,a)[r + γv̂(s')].
	
	- `γ::T`: Discount factor for expected return computation
	- `value_function::Function`: State-value function (features, params) -> value
	- Other arguments: See main method documentation above
	
	## Double Parameter Set (Double SARSA)
	```julia
	form_value_function(mdp, update_feature_vector!, update_action_values!, feature_vector, parameters1, parameters2)
	```
	Creates action-value function for double SARSA algorithm that averages two parameter sets.
	Computes action-values from both parameter sets and averages them.
	
	- `parameters1::W`: First set of value function parameters
	- `parameters2::W`: Second set of value function parameters
	- Other arguments: See main method documentation above
	"""
	function form_value_function(mdp::StateMDP{T, S, A, P, F1, F2, F3}, update_feature_vector!::Function, update_action_values!::Function, feature_vector::V, parameters::W) where {T<:Real, S, A, P<:AbstractStateTransition, F1<:Function, F2<:Function, F3<:Function, V, W}
		function q̂(s::S; action_values::Vector{T} = zeros(T, length(mdp.actions)), feature_vector::V = deepcopy(feature_vector), parameters::W = parameters, kwargs...)
			update_feature_vector!(feature_vector, s)
			maxq, i_a_max = update_action_values!(action_values, feature_vector, parameters; kwargs...)
			(action_values = action_values, maximizing_action = i_a_max, maximizing_value = maxq)
		end

		form_kwargs() = form_action_value_kwargs(mdp, feature_vector, parameters)
	
		return q̂, form_kwargs
	end

	function form_value_function(mdp::StateMDP{T, S, A, P, F1, F2, F3}, update_feature_vector!::Function, update_action_values!::Function, feature_vector::Vector{T}, parameters::W) where {T<:Real, S, A, P<:AbstractStateTransition, F1<:Function, F2<:Function, F3<:Function, W<:FCANNParamsGPU}
		cpu_params = initialize_cpu_params(parameters)
		gpu_params = initialize_gpu_params(cpu_params)
		function q̂(s::S; feature_vector::Vector{T} = copy(feature_vector), parameters::FCANNParams{T} = cpu_params, gpu_params::FCANNParamsGPU = gpu_params, gpu_kwargs::NamedTuple = NamedTuple(), use_gpu::Bool = false, kwargs...)
			update_feature_vector!(feature_vector, s)
			if !use_gpu
				q̂(feature_vector, parameters; kwargs...)
			else
				q̂(feature_vector, gpu_params; gpu_kwargs...)
			end
		end

		function q̂(x::Vector{T}, parameters::FCANNParams{T}; action_values::Vector{T} = zeros(T, length(mdp.actions)), activations = FCANN.form_activations(cpu_params.weights[1]), kwargs...)
			maxq, i_a_max = update_action_values!(action_values, x, cpu_params; activations = activations, kwargs...)
			(action_values = action_values, maximizing_action = i_a_max, maximizing_value = maxq)
		end

		function q̂(x::Vector{T}, parameters::FCANNParamsGPU; action_values::Vector{T} = zeros(T, length(mdp.actions)), activations = FCANN.form_activations(gpu_params.weights[1]), d_x::FCANN.CUDAArray = FCANN.cuda_allocate(x), kwargs...)
			FCANN.memcpy!(d_x, x)
			maxq, i_a_max = update_action_values!(action_values, d_x, parameters; activations = activations, kwargs...)
			(action_values = action_values, maximizing_action = i_a_max, maximizing_value = maxq)
		end
	
		form_kwargs() = form_action_value_kwargs(mdp, feature_vector, cpu_params, gpu_params)
	
		return q̂, form_kwargs
	end

	function form_value_function(mdp::StateMDP{T, S, A, P, F1, F2, F3}, update_feature_vector!::Function, update_action_values!::Function, feature_vector::FCANN.CUDAArray, parameters::W) where {T<:Real, S, A, P<:AbstractStateTransition, F1<:Function, F2<:Function, F3<:Function, W<:FCANNParamsGPU}
		cpu_feature = FCANN.host_allocate(feature_vector)
		form_value_function(mdp, update_feature_vector!, update_action_values!, cpu_feature, parameters)
	end

	#form value function when training two sets of parameters with double sarsa
	function form_value_function(mdp::StateMDP{T, S, A, P, F1, F2, F3}, update_feature_vector!::Function, update_action_values!::Function, feature_vector::V, parameters1::W, parameters2::W) where {T<:Real, S, A, P<:AbstractStateTransition, F1<:Function, F2<:Function, F3<:Function, V, W}
		function q̂(s::S; action_values1::Vector{T} = zeros(T, length(mdp.actions)), action_values2::Vector{T} = zeros(T, length(mdp.actions)), feature_vector::V = deepcopy(feature_vector), parameters1::W = parameters1, parameters2::W = parameters2, action_value_kwargs...)
			update_feature_vector!(feature_vector, s)
			update_action_values!(action_values1, feature_vector, parameters1; action_value_kwargs...)
			update_action_values!(action_values2, feature_vector, parameters2; action_value_kwargs...)
			action_values1 .+= action_values2
			action_values1 ./= 2
			(maxq, i_a_max) = findmax(action_values1)
				
			(action_values = action_values1, maximizing_action = i_a_max, maximizing_value = maxq)
		end

		form_kwargs() = form_action_value_kwargs(mdp, feature_vector, parameters1, parameters2)

		return q̂, form_kwargs
	end	

	function form_value_function(mdp::StateMDP{T, S, A, P, F1, F2, F3}, update_feature_vector!::Function, update_action_values!::Function, feature_vector::Vector{T}, parameters1::W, parameters2::W) where {T<:Real, S, A, P<:AbstractStateTransition, F1<:Function, F2<:Function, F3<:Function, W <: FCANNParamsGPU}
		cpu_params1 = initialize_cpu_params(parameters1)
		cpu_params2 = initialize_cpu_params(parameters2)
		form_value_function(mdp, update_feature_vector!, update_action_values!, feature_vector, cpu_params1, cpu_params2)
	end

	function form_value_function(mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, update_feature_vector!::Function, value_function::Function, feature_vector::V, parameters::W) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1<:Function, F2<:Function, F3<:Function, V, W}
		function q̂(s::S; action_values::Matrix{T} = zeros(T, length(mdp.actions), 1), parameters::W = parameters, feature_vector::V = deepcopy(feature_vector), action_value_args = form_action_value_args(mdp, feature_vector, parameters), kwargs...)
			maxq, i_a_max = update_action_values!(action_values, s, feature_vector, update_feature_vector!, value_function, parameters, mdp, γ, action_value_args...; kwargs...)
			(action_values = action_values, maximizing_action = i_a_max |> Tuple |> prod, maximizing_value = maxq)
		end #since the action values here are a matrix, findmax will produce a cartesian index, this step transforms it back into an integer

		form_kwargs() = (action_values = zeros(T, length(mdp.actions), 1), parameters = parameters, feature_vector = deepcopy(feature_vector), action_value_args = form_action_value_args(mdp, feature_vector, parameters))
		return q̂, form_kwargs
	end

	function form_value_function(mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, update_feature_vector!::Function, value_function::Function, feature_vector::Vector{T}, parameters::FCANNParamsGPU) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1<:Function, F2<:Function, F3<:Function}
		cpu_params = initialize_cpu_params(parameters)
		gpu_params = initialize_gpu_params(cpu_params)

		function q̂(s::S; use_gpu::Bool = false, parameters = cpu_params, gpu_params = gpu_params, kwargs...)
			if !use_gpu
				q̂(s, parameters; kwargs...)
			else
				q̂(s, gpu_params; kwargs...)
			end
		end

		function q̂(s::S, parameters::FCANNParams{T}; action_values::Matrix{T} = zeros(T, length(mdp.actions), 1), feature_vector::Vector{T} = copy(feature_vector), action_value_args = form_action_value_args(mdp, feature_vector, parameters), kwargs...)
			maxq, i_a_max = update_action_values!(action_values, s, feature_vector, update_feature_vector!, value_function, parameters, mdp, γ, action_value_args...; kwargs...)
			(action_values = action_values, maximizing_action = i_a_max |> Tuple |> prod, maximizing_value = maxq)
		end

		function q̂(s::S, parameters::FCANNParamsGPU; action_values::Matrix{T} = zeros(T, length(mdp.actions), 1), feature_vector::Vector{T} = copy(feature_vector), gpu_action_value_args = form_action_value_args(mdp, feature_vector, parameters), kwargs...)
			maxq, i_a_max = update_action_values!(action_values, s, feature_vector, update_feature_vector!, value_function, parameters, mdp, γ, gpu_action_value_args...; kwargs...)
			(action_values = action_values, maximizing_action = i_a_max |> Tuple |> prod, maximizing_value = maxq)
		end

		form_kwargs() = (action_values = zeros(T, length(mdp.actions), 1), parameters = cpu_params, feature_vector = copy(feature_vector), action_value_args = form_action_value_args(mdp, feature_vector, cpu_params), gpu_action_value_args = form_action_value_args(mdp, feature_vector, gpu_params))
		return q̂, form_kwargs
	end

	function form_value_function(mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, update_feature_vector!::Function, value_function::Function, feature_vector::FCANN.CUDAArray, parameters::FCANNParamsGPU) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1<:Function, F2<:Function, F3<:Function}
		cpu_feature = FCANN.cuda_allocate(feature_vector)
		form_value_function(mdp, γ, update_feature_vector!, value_function, cpu_feature, parameters)
	end
end

# ╔═╡ 991492f4-7dfc-43aa-ab6c-a6b1f3e38225
"""
    semi_gradient_sarsa!(parameters, mdp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, update_action_values!, ∇q̂, update_value_gradient!; kwargs...) -> NamedTuple

Semi-gradient SARSA algorithm for control with function approximation.

Performs on-policy temporal difference control using SARSA updates with ε-greedy policy improvement.
Updates action-value function parameters using semi-gradient methods and function approximation.

# Type Parameters
- `P`: Parameter type (Vector or Matrix)
- `T <: Real`: Numeric type for computations

# Arguments
- `parameters::P`: Action-value function parameters (modified in-place)
- `mdp::`[`StateMDP`](@ref): Markov Decision Process structure
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `max_episodes::Integer`: Maximum number of episodes to run
- `max_steps::Integer`: Maximum total steps across all episodes
- `feature_vector`: Feature vector storage for state representations
- `update_feature_vector!::Function`: Function to extract features from states
- `update_action_values!::Function`: Function to compute action-values from features and parameters
- `∇q̂`: Gradient storage for action-value function gradients
- `update_value_gradient!::Function`: Function to compute action-value function gradient

# Keyword Arguments
- `α::T = 0.1`: Learning rate
- `ϵ::T = 0.1`: Exploration probability for ε-greedy policy
- `compute_value::Function = compute_sarsa_value`: Function to compute target values from action-values and policy
- `α_decay::T = 1.0`: Learning rate decay factor
- `decay_step::Integer = typemax(Int64)`: Step at which to begin learning rate decay
- `save_parameter_history::Bool = false`: Whether to save parameter history at each step
- `kwargs...`: Additional arguments passed to update functions

# Returns
- `NamedTuple`: Results containing:
  - `value_function`: Final action-value function q̂(s) created by [`form_value_function`](@ref)
  - `episode_rewards::Vector{T}`: Total reward per episode
  - `episode_steps::Vector{Int64}`: Step count per episode
  - `parameter_history::Vector{P}`: Parameter history (if `save_parameter_history=true`)
  - `final_parameters::P`: Copy of final parameters

# See Also
[`compute_sarsa_value`](@ref), [`form_value_function`](@ref), [`StateMDP`](@ref)

# Algorithm Details
1. Initialize state and select initial action using ε-greedy policy
2. For each step:
   - Compute current action-value q̂(s,a) and gradient ∇q̂(s,a)
   - Take action, observe reward r and next state s'
   - Select next action a' using ε-greedy policy on updated parameters
   - Compute target using `compute_value(action_values, policy, a')`
   - Update parameters: θ ← θ + α·δ·∇q̂(s,a) where δ = r + γ·q̂(s',a') - q̂(s,a)
   - Continue with s ← s', a ← a'
3. Return final action-value function and training statistics

# Performance Notes
- Updates parameters in-place to minimize allocations
- Supports learning rate decay after specified step count
- Optional parameter history tracking for analysis
- Compatible with various feature representations and approximation methods
"""
function semi_gradient_sarsa!(parameters::P, mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, update_action_values!::Function, ∇q̂, update_value_gradient!::Function; α = one(T)/10, ϵ = one(T) / 10, compute_value::Function = compute_sarsa_value, α_decay = one(T), decay_step = typemax(Int64), save_parameter_history = false, kwargs...) where {P, T<:Real}
	action_values = zeros(T, length(mdp.actions))
	policy = copy(action_values)
	
	s = mdp.initialize_state()
	update_feature_vector!(feature_vector, s)
	update_action_values!(action_values, feature_vector, parameters)
	policy .= action_values
	make_ϵ_greedy_policy!(policy; ϵ = ϵ)
	i_a = sample_action(policy)
	
	ep = 1
	step = 1
	epreward = zero(T)
	episode_rewards = Vector{T}()
	episode_steps = Vector{Int64}()
	action_values = zeros(T, length(mdp.actions))
	policy = zeros(T, length(mdp.actions))
	decay = one(T)
	parameter_history = Vector{P}()
	save_parameter_history && push!(parameter_history, deepcopy(parameters))
	
	while (ep <= max_episodes) && (step <= max_steps)
		update_value_gradient!(∇q̂, action_values, feature_vector, i_a, parameters)
		q̂ = action_values[i_a]
		
		(r, s′) = mdp.ptf(s, i_a)
		epreward += r

		terminated = mdp.isterm(s′)
		if terminated
			s′ = mdp.initialize_state()
			push!(episode_rewards, epreward)
			push!(episode_steps, step)
			epreward = zero(T)
			ep += 1
		end
		
		update_feature_vector!(feature_vector, s′)
		update_action_values!(action_values, feature_vector, parameters)
		policy .= action_values
		make_ϵ_greedy_policy!(policy; ϵ = ϵ)
		i_a′ = sample_action(policy)

		q̂′ = if terminated
			zero(T)
		else
			compute_value(action_values, policy, i_a′)
		end

		target = r + γ*q̂′

		δ = target - q̂
		
		decay *= (step > decay_step)*α_decay + (step <= decay_step)
		
		update_params_with_gradient!(parameters, α*decay*δ, ∇q̂)
		
		save_parameter_history && push!(parameter_history, deepcopy(parameters))
		s = s′
		i_a = i_a′
		step += 1
	end

	q̂, form_kwargs = form_value_function(mdp, update_feature_vector!, update_action_values!, feature_vector, parameters)
	return (value_function = q̂, episode_rewards = episode_rewards, episode_steps = episode_steps, parameter_history = parameter_history, final_parameters = deepcopy(parameters), form_kwargs = form_kwargs)
end;

# ╔═╡ b0761704-5447-4e64-8270-708d9dccef60
"""
    semi_gradient_dp!(parameters, mdp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, value_function, ∇v̂, update_value_gradient!; kwargs...) -> NamedTuple

Semi-gradient dynamic programming algorithm for control with function approximation.

Combines dynamic programming value updates with trajectory sampling for control. Uses state-value
function approximation with Bellman equation-based action-value computation and ε-greedy policy
improvement. Effectively performs trajectory sampling while staying close to the optimal policy.

# Type Parameters
- `T <: Real`: Numeric type for computations
- `S`: State type from MDP
- `A`: Action type from MDP  
- `PR`: Parameter type (Vector or Matrix)

# Arguments
- `parameters::PR`: State-value function parameters (modified in-place)
- `mdp::StateMDP`: Markov Decision Process with transition distributions
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `max_episodes::Integer`: Maximum number of episodes to run
- `max_steps::Integer`: Maximum total steps across all episodes
- `feature_vector`: Feature vector storage for state representations
- `update_feature_vector!::Function`: Function to extract features from states
- `value_function::Function`: State-value function (features, params) -> value
- `∇v̂`: Gradient storage for state-value function gradients
- `update_value_gradient!::Function`: Function to compute state-value function gradient

# Keyword Arguments
- `α::T = 0.1`: Learning rate
- `ϵ::T = 0.1`: Exploration probability for ε-greedy policy
- `α_decay::T = 1.0`: Learning rate decay factor
- `decay_step::Integer = typemax(Int64)`: Step at which to begin learning rate decay
- `save_parameter_history::Bool = false`: Whether to save parameter history at each step
- `kwargs...`: Additional arguments passed to update functions

# Returns
- `NamedTuple`: Results containing:
  - `value_function`: Final action-value function q̂(s) created by [`form_value_function`](@ref)
  - `episode_rewards::Vector{T}`: Total reward per episode
  - `episode_steps::Vector{Int64}`: Step count per episode
  - `parameter_history::Vector{PR}`: Parameter history (if `save_parameter_history=true`)
  - `final_parameters::PR`: Copy of final parameters

# See Also
[`update_action_values!`](@ref), [`form_value_function`](@ref), [`StateMDP`](@ref)

# Algorithm Details
1. For each step:
   - Compute current state-value v̂(s) and gradient ∇v̂(s)
   - Use [`update_action_values!`](@ref) to compute all action-values via Bellman equation
   - Update state-value parameters: θ ← θ + α·δ·∇v̂(s) where δ = max_a q̂(s,a) - v̂(s)
   - Select action using ε-greedy policy on computed action-values
   - Take action and transition to next state
2. Return final action-value function derived from learned state-value function

# Performance Notes
- Combines exact dynamic programming updates with approximate trajectory sampling
- Uses transition distributions for exact action-value computation at each step
- Creates internal closure for state-value function evaluation during action-value computation
- Supports learning rate decay and parameter history tracking
"""
function semi_gradient_dp!(parameters::PR, mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, value_function::Function, ∇v̂, update_value_gradient!::Function; α = one(T)/10, ϵ = one(T) / 10, α_decay = one(T), decay_step = typemax(Int64), save_parameter_history = false, kwargs...) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1<:Function, F2<:Function, F3<:Function, PR}
	action_values = zeros(T, length(mdp.actions), 1)
	policy = zeros(T, length(mdp.actions))

	action_value_args = form_action_value_args(mdp, feature_vector, parameters)
	
	s = mdp.initialize_state()
	
	ep = 1
	step = 1
	epreward = zero(T)
	episode_rewards = Vector{T}()
	episode_steps = Vector{Int64}()
	decay = one(T)
	parameter_history = Vector{PR}()
	save_parameter_history && push!(parameter_history, deepcopy(parameters))

	while (ep <= max_episodes) && (step <= max_steps)
		update_feature_vector!(feature_vector, s)
		v̂ = update_value_gradient!(∇v̂, feature_vector, parameters)
		
		#computes q and finds maximizing action value, this is effectively trajectory sampling in the case of approximation where we stay close to the optimal policy
		target, i_a_max = update_action_values!(action_values, s, feature_vector, update_feature_vector!, value_function, parameters, mdp, γ, action_value_args...; kwargs...)
	
		δ = target - v̂

		decay *= (step > decay_step)*α_decay + (step <= decay_step)
		update_params_with_gradient!(parameters, α*decay*δ, ∇v̂)

		policy .= action_values
		make_ϵ_greedy_policy!(policy; ϵ = ϵ)
		i_a = sample_action(policy)

		(r, s′) = mdp.ptf(s, i_a)
		epreward += r

		if mdp.isterm(s′)
			s′ = mdp.initialize_state()
			push!(episode_rewards, epreward)
			push!(episode_steps, step)
			epreward = zero(T)
			ep += 1
		end
		
		save_parameter_history && push!(parameter_history, deepcopy(parameters))
		s = s′
		step += 1
	end

	q̂, form_kwargs = form_value_function(mdp, γ, update_feature_vector!, value_function, feature_vector, parameters)

	
	return (value_function = q̂, episode_rewards = episode_rewards, episode_steps = episode_steps, parameter_history = parameter_history, final_parameters = deepcopy(parameters), form_kwargs = form_kwargs)
end;

# ╔═╡ de3e4afe-f935-4b33-9218-08d403743c60
begin
	"""
	    get_num_actions(num_actions) -> Integer
	    get_num_actions(mdp) -> Integer
	
	Extracts the number of actions from various input types.
	
	Provides a uniform interface for determining action space size from either
	direct specification or MDP structures.
	
	# Arguments
	- `num_actions::Integer`: Direct specification of number of actions
	- `mdp::`[`StateMDP`](@ref): MDP structure containing action space
	
	# Returns
	- `Integer`: Number of actions in the action space
	
	# See Also
	[`StateMDP`](@ref), [`initialize_linear_parameters`](@ref)
	"""
	get_num_actions(num_actions::Integer) = num_actions
	get_num_actions(mdp::StateMDP) = length(mdp.actions)
end

# ╔═╡ d82faf3b-c975-4b23-ad62-473bd943c4e2
begin
	function initialize_linear_parameters(feature_vector_length::Integer, num_actions::Integer, init_value::T) where T<:Real
		params = ones(T, feature_vector_length, num_actions)
		params .*= init_value
		return params
	end
	initialize_linear_parameters(x, y, init_value) = initialize_linear_parameters(length(x), get_num_actions(y), init_value)
end

# ╔═╡ 8513264e-6a14-41ab-8cfd-a335682a06aa
md"""
#### *Linear Approximation*
"""

# ╔═╡ b697c5ba-4647-4998-a153-1e97dd91cb23
"""
    semi_gradient_sarsa_linear(mdp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!; kwargs...) -> NamedTuple

Semi-gradient SARSA algorithm with linear function approximation.

Convenience method that automatically sets up linear approximation components and delegates
to [`semi_gradient_sarsa!`](@ref) with appropriate linear functions and gradient storage.

# Type Parameters
- `T <: Real`: Numeric type for computations

# Arguments
- `mdp::`[`StateMDP`](@ref): Markov Decision Process structure
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `max_episodes::Integer`: Maximum number of episodes to run
- `max_steps::Integer`: Maximum total steps across all episodes
- `feature_vector::`[`LinearFeatureVector`](@ref): Template feature vector for linear approximation
- `update_feature_vector!::Function`: Function to extract features from states

# Keyword Arguments
- `init_value::T = zero(T)`: Initial value for all parameters
- `parameters::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, init_value)`: Pre-initialized parameter matrix
- `kwargs...`: Additional arguments passed to [`semi_gradient_sarsa!`](@ref)

# Returns
- `NamedTuple`: Same as [`semi_gradient_sarsa!`](@ref) - see that function for details

# See Also
[`semi_gradient_sarsa!`](@ref), [`LinearFeatureVector`](@ref), [`update_linear_action_values!`](@ref), [`LinearActionValueGradient`](@ref)

# Algorithm Details
1. Creates parameter matrix using [`initialize_linear_parameters`](@ref) if not provided
2. Sets up [`LinearActionValueGradient`](@ref) for gradient storage
3. Delegates to [`semi_gradient_sarsa!`](@ref) with:
   - [`update_linear_action_values!`](@ref) for action-value computation
   - [`update_linear_value_gradient!`](@ref) for gradient updates
4. Returns results from core algorithm

# Examples
```julia-repl
julia> # Basic usage with tile coding features
julia> result = semi_gradient_sarsa_linear(mdp, 0.9f0, 1000, 50000, 
                                          feature_vector, update_tile_features!)

julia> # With custom parameters and learning rate
julia> result = semi_gradient_sarsa_linear(mdp, 0.9f0, 1000, 50000,
                                          feature_vector, update_tile_features!;
                                          α=0.05f0, ϵ=0.05f0)
```

# Performance Notes
- Automatically handles linear approximation setup to minimize user setup code
- Reuses parameter matrix if provided to avoid reinitialization
- Compatible with all linear feature representations
"""
semi_gradient_sarsa_linear(mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), parameters::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, init_value), kwargs...) where T<:Real = semi_gradient_sarsa!(parameters, mdp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, update_linear_action_values!, LinearActionValueGradient(deepcopy(feature_vector), 0), update_linear_value_gradient!; kwargs...)

# ╔═╡ 526689e2-85ea-47d5-9791-5aa730f8b1ab
"""
    semi_gradient_dp_linear(mdp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!; kwargs...) -> NamedTuple

Semi-gradient Dynamic Programming algorithm with linear function approximation.

Convenience method that automatically sets up linear approximation components and delegates
to [`semi_gradient_dp!`](@ref) with appropriate linear functions and gradient storage.
Performs value function estimation using dynamic programming principles with function approximation.

# Type Parameters
- `T <: Real`: Numeric type for computations

# Arguments
- `mdp::`[`StateMDP`](@ref): Markov Decision Process structure
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `max_episodes::Integer`: Maximum number of episodes to run
- `max_steps::Integer`: Maximum total steps across all episodes
- `feature_vector::`[`LinearFeatureVector`](@ref): Template feature vector for linear approximation
- `update_feature_vector!::Function`: Function to extract features from states

# Keyword Arguments
- `init_value::T = zero(T)`: Initial value for all parameters
- `parameters::Vector{T} = initialize_linear_parameters(feature_vector, init_value)`: Pre-initialized parameter vector
- `kwargs...`: Additional arguments passed to [`semi_gradient_dp!`](@ref)

# Returns
- `NamedTuple`: Same as [`semi_gradient_dp!`](@ref) - see that function for details

# See Also
[`semi_gradient_dp!`](@ref), [`linear_value_function`](@ref), [`LinearFeatureVector`](@ref), [`update_linear_value_gradient!`](@ref)

# Algorithm Details
1. Creates parameter vector using [`initialize_linear_parameters`](@ref) if not provided
2. Sets up gradient storage with feature vector copy
3. Delegates to [`semi_gradient_dp!`](@ref) with:
   - [`linear_value_function`](@ref) for value computation
   - [`update_linear_value_gradient!`](@ref) for gradient updates
4. Returns results from core algorithm
"""
semi_gradient_dp_linear(mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), parameters::Vector{T} = initialize_linear_parameters(feature_vector, init_value), kwargs...) where T<:Real = semi_gradient_dp!(parameters, mdp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, linear_value_function, deepcopy(feature_vector), update_linear_value_gradient!; kwargs...)

# ╔═╡ 8d096d0d-8fea-421a-aa33-82269d3fe7e2
md"""
#### *Non-linear Approximation*
"""

# ╔═╡ be1ad356-de4b-469c-bb65-81d630f07674
"""
    setup_fcann_action_value_arguments(params, input_length, hidden_layers, reslayers, l2, dropout, use_μP, activation_list) -> NamedTuple

Set up neural network components for action-value function approximation.

Creates feature vectors, gradients, and specialized functions for FCANN-based action-value estimation.
Handles μP scaling, activation management, and gradient computation setup for multi-action problems.

# Type Parameters
- `T <: Real`: Numeric type for computations

# Arguments
- `params::`[`FCANNParams`](@ref)`{T}`: Pre-initialized network parameters
- `input_length::Integer`: Dimension of input feature vectors
- `hidden_layers::Vector{Int64}`: Number of units in each hidden layer
- `reslayers::Integer`: Number of residual layers in network
- `l2::T`: L2 regularization strength
- `dropout::T`: Dropout rate for training
- `use_μP::Bool`: Whether to apply μP scaling for network initialization
- `activation_list`: Activation function configuration per layer

# Returns
- `NamedTuple` with fields:
  - `feature_vector`: Input feature vector storage
  - `gradient`: Gradient storage matching parameter structure
  - `update_action_values!::Function`: Function to compute action values and return maximum
  - `update_value_gradient!::Function`: Function to compute gradients for specific actions
  - `activations`: Pre-allocated activation storage for network forward pass

# See Also
[`setup_fcann_value_arguments`](@ref), [`FCANNParams`](@ref), [`update_fcann_value_gradient!`](@ref), [`fcann_value_function!`](@ref)

# Algorithm Details
1. Creates input vector and activation storage using [`FCANN.form_activations`](@ref)
2. Sets up gradient computation storage (tanh_grad_z, deltas)
3. Configures μP scaling factors if enabled
4. Returns specialized functions for action-value computation and gradient updates
"""
function setup_fcann_action_value_arguments(params::FCANNParams{T}, l2::T, dropout::T, use_μP::Bool, activation_list; use_gpu = false) where {T<:Real}
	input_length, hidden_layers, num_hidden = get_network_dimensions(params)
	
	#form activations for network
	activations = FCANN.form_activations(params.weights[1])
	tanh_grad_z = deepcopy(activations)
	deltas = deepcopy(activations)

	scales = fill(one(T), length(params.weights[1]))
	if use_μP
		for i in eachindex(hidden_layers)
			i′ = i + 1
			scales[i′] /= size(params.weights[1][i′], 2)
		end
	end

	function update_action_values!(action_values::Vector{T}, x, params; activations = activations, kwargs...) 
		fcann_value_function!(activations, x, params)
		action_values .= activations[end]
		val, index = findmax(action_values)
		isnan(val) && @warn "Got NaN action value inside $action_values"
		isinf(val) && @warn "Got Inf action value inside $action_values"
		return (val, index)
	end
	
	function update_value_gradient!(∇q̂::FCANNParams, x, i_a::Integer, params::FCANNParams) 
		update_fcann_value_gradient!(∇q̂, x, i_a, params, hidden_layers, l2, tanh_grad_z, activations, deltas, dropout, activation_list)
		use_μP && scale_fcann_params!(∇q̂, scales)
		return ∇q̂
	end

	function update_value_gradient!(∇q̂::FCANNParams, action_values::Vector{T}, x, i_a::Integer, params::FCANNParams) 
		update_value_gradient!(∇q̂, x, i_a, params)
		action_values .= activations[end]
		val, index = findmax(action_values)
		isnan(val) && @warn "Got NaN action value inside $action_values"
		isinf(val) && @warn "Got Inf action value inside $action_values"
		return (val, index)
	end

	if use_gpu && in(:GPU, backendList)
		d_activations = FCANN.device_allocate(activations)
		d_tanh_grad_z = FCANN.device_allocate(tanh_grad_z)
		d_deltas = FCANN.device_allocate(deltas)
		d_params = initialize_gpu_params(params)
		d_gradient = initialize_gpu_params(params)
		d_x = FCANN.cuda_allocate(zeros(T, input_length))

		function update_action_values!(action_values::Vector{T}, d_x::FCANN.CUDAArray, params::FCANNParamsGPU; activations::FCANNActivationsGPU = d_activations, kwargs...) 			
			fcann_value_function!(activations, d_x, params)
			FCANN.memcpy!(action_values, activations[end])
			val, index = findmax(action_values)
			isnan(val) && @warn "Got NaN action value inside $action_values"
			isinf(val) && @warn "Got Inf action value inside $action_values"
			return (val, index)
		end

		function update_action_values!(action_values::Vector{T}, x::Vector{T}, params::FCANNParamsGPU; gpu_feature_vector::FCANN.CUDAArray = d_x, kwargs...) 			
			FCANN.memcpy!(gpu_feature_vector, x)
			update_action_values!(action_values, gpu_feature_vector, params; kwargs...)
		end

		function update_value_gradient!(∇q̂::FCANNParamsGPU, d_x::FCANN.CUDAArray, i_a::Integer, params::FCANNParamsGPU) 
			update_fcann_value_gradient!(∇q̂, d_x, i_a, params, hidden_layers, l2, d_tanh_grad_z, d_activations, d_deltas, dropout, activation_list)
			use_μP && scale_fcann_params!(∇q̂, scales)
			return ∇q̂
		end

		function update_value_gradient!(∇q̂::FCANNParamsGPU, x::Vector{T}, i_a::Integer, params::FCANNParamsGPU)
			FCANN.memcpy!(gpu_feature_vector, d_x)
			update_value_gradient!(∇q̂, d_x, i_a, params)
		end

		function update_value_gradient!(∇q̂::FCANNParamsGPU, action_values::Vector{T}, x, i_a::Integer, params::FCANNParamsGPU)
			update_value_gradient!(∇q̂, x, i_a, params)
			FCANN.memcpy!(action_values, d_activations[end])
			val, index = findmax(action_values)
			isnan(val) && @warn "Got NaN action value inside $action_values"
			isinf(val) && @warn "Got Inf action value inside $action_values"
			return (val, index)
		end

		function cleanup_vars()
			FCANN.clear_gpu_data(d_gradient.weights[1])
			FCANN.clear_gpu_data(d_gradient.weights[2])
			FCANN.clear_gpu_data(d_params.weights[1])
			FCANN.clear_gpu_data(d_params.weights[2])
			FCANN.clear_gpu_data(d_deltas)
			FCANN.clear_gpu_data(d_tanh_grad_z)
			FCANN.clear_gpu_data([d_x])
			FCANN.clear_gpu_data(d_activations)
		end

		gpu_args = (activations = d_activations, gradient = d_gradient, params = d_params, feature_vector = d_x, cleanup_vars = cleanup_vars)
	else
		gpu_args = ()
	end

	return (gradient = deepcopy(params), update_action_values! = update_action_values!, update_value_gradient! = update_value_gradient!, activations = activations, gpu_args = gpu_args)
end;

# ╔═╡ 7e87f2ec-c96f-4897-bb61-c27913f6944f
"""
    semi_gradient_sarsa_fcann(mdp, γ, max_episodes, max_steps, update_feature_vector!, num_features, hidden_layers; kwargs...) -> NamedTuple

Semi-gradient SARSA algorithm with fully-connected neural network approximation.

Convenience method that automatically sets up FCANN approximation components and delegates
to [`semi_gradient_sarsa!`](@ref) with appropriate neural network functions and gradient storage.

# Type Parameters
- `T <: Real`: Numeric type for computations

# Arguments
- `mdp::`[`StateMDP`](@ref): Markov Decision Process structure
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `max_episodes::Integer`: Maximum number of episodes to run
- `max_steps::Integer`: Maximum total steps across all episodes
- `update_feature_vector!::Function`: Function to extract features from states
- `num_features::Integer`: Dimension of input feature vectors
- `hidden_layers::Vector{Int64}`: Number of units in each hidden layer

# Keyword Arguments
- `reslayers::Integer = 0`: Number of residual layers
- `use_μP::Bool = true`: Whether to apply μP scaling
- `parameters::`[`FCANNParams`](@ref)`{T} = FCANN.initializeparams_saxe(...)`: Pre-initialized network parameters
- `dropout::T = zero(T)`: Dropout rate for training
- `activation_list = fill(true, length(hidden_layers))`: Activation configuration per layer
- `l2::T = zero(T)`: L2 regularization strength
- `kwargs...`: Additional arguments passed to [`semi_gradient_sarsa!`](@ref)

# Returns
- `NamedTuple` with fields:
  - `value_function`: Closure for action-value function evaluation
  - `episode_rewards`: Reward history per episode
  - `episode_steps`: Step counts per episode
  - `parameter_history`: Training history of network parameters
  - `final_parameters`: Final trained network parameters

# See Also
[`semi_gradient_sarsa!`](@ref), [`setup_fcann_action_value_arguments`](@ref), [`FCANNParams`](@ref), [`semi_gradient_dp_fcann`](@ref)

# Algorithm Details
1. Sets up FCANN components using [`setup_fcann_action_value_arguments`](@ref)
2. Initializes network parameters with [`FCANN.initializeparams_saxe`](@ref) if not provided
3. Delegates to [`semi_gradient_sarsa!`](@ref) with neural network functions
4. Returns wrapped value function with activation storage management
"""
function semi_gradient_sarsa_fcann(mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; reslayers::Integer = 0, use_μP::Bool = true, parameters::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, length(mdp.actions), reslayers, use_μP), dropout = zero(T), activation_list = fill(true, length(hidden_layers)), l2 = zero(T), use_gpu::Bool = false, kwargs...) where T<:Real 
	setup = setup_fcann_action_value_arguments(parameters, l2, dropout, use_μP, activation_list; use_gpu = use_gpu)
	!use_gpu && return semi_gradient_sarsa!(parameters, mdp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, setup.update_action_values!, setup.gradient, setup.update_value_gradient!; kwargs...)

	isempty(setup.gpu_args) && error("GPU backend is not available")
	gpu_feature_update! = setup_gpu_feature(feature_vector, update_feature_vector!)
	output = semi_gradient_sarsa!(setup.gpu_args.params, mdp, γ, max_episodes, max_steps, setup.gpu_args.feature_vector, gpu_feature_update!, setup.update_action_values!, setup.gpu_args.gradient, setup.update_value_gradient!; kwargs...)
	FCANN.GPU2Host(parameters.weights, setup.gpu_args.params.weights)
	setup.gpu_args.cleanup_vars()
	(;output..., final_parameters = parameters)
end

# ╔═╡ 6d4b513d-2744-4f9c-8bee-e51fe9d0bade
begin
	make_value_activations(params::FCANNParams, mdp::StateMDP{T, S, A, P, F1, F2, F3}) where {T<:Real, S, A, P<:StateMDPTransitionDistribution, F1, F2, F3} = FCANN.form_activations(params.weights[1])
	make_value_activations(params::FCANNParams, mdp::StateMDP{T, S, A, P, F1, F2, F3}) where {T<:Real, S, A, P<:StateMDPTransitionDeterministic, F1, F2, F3}  = FCANN.form_activations(params.weights[1], length(mdp.actions))
end

# ╔═╡ 4c94be37-dcd7-4b32-8e7f-3371ddaa254a
"""
    semi_gradient_dp_fcann(mdp, γ, max_episodes, max_steps, update_feature_vector!, num_features, hidden_layers; kwargs...) -> NamedTuple

Semi-gradient Dynamic Programming algorithm with fully-connected neural network approximation.

Convenience method that automatically sets up FCANN approximation components and delegates
to [`semi_gradient_dp!`](@ref) with appropriate neural network functions and gradient storage.
Uses single-output network for state value function approximation.

# Type Parameters
- `T <: Real`: Numeric type for computations

# Arguments
- `mdp::`[`StateMDP`](@ref): Markov Decision Process structure
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `max_episodes::Integer`: Maximum number of episodes to run
- `max_steps::Integer`: Maximum total steps across all episodes
- `update_feature_vector!::Function`: Function to extract features from states
- `num_features::Integer`: Dimension of input feature vectors
- `hidden_layers::Vector{Int64}`: Number of units in each hidden layer

# Keyword Arguments
- `reslayers::Integer = 0`: Number of residual layers
- `use_μP::Bool = true`: Whether to apply μP scaling
- `parameters::`[`FCANNParams`](@ref)`{T} = FCANN.initializeparams_saxe(...)`: Pre-initialized network parameters for single output
- `dropout::T = zero(T)`: Dropout rate for training
- `activation_list = fill(true, length(hidden_layers))`: Activation configuration per layer
- `l2::T = zero(T)`: L2 regularization strength
- `kwargs...`: Additional arguments passed to [`semi_gradient_dp!`](@ref)

# Returns
- `NamedTuple` with fields:
  - `value_function`: Closure for state value function evaluation
  - `episode_rewards`: Reward history per episode
  - `episode_steps`: Step counts per episode
  - `parameter_history`: Training history of network parameters
  - `final_parameters`: Final trained network parameters

# See Also
[`semi_gradient_dp!`](@ref), [`setup_fcann_value_arguments`](@ref), [`FCANNParams`](@ref), [`semi_gradient_sarsa_fcann`](@ref)

# Algorithm Details
1. Sets up FCANN components using [`setup_fcann_value_arguments`](@ref)
2. Initializes single-output network parameters with [`FCANN.initializeparams_saxe`](@ref) if not provided
3. Delegates to [`semi_gradient_dp!`](@ref) with neural network functions
4. Returns wrapped value function with activation storage management
"""
function semi_gradient_dp_fcann(mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; reslayers::Integer = 0, use_μP::Bool = true, parameters::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, 1, reslayers, use_μP), dropout = zero(T), activation_list = fill(true, length(hidden_layers)), l2 = zero(T), use_gpu::Bool = false, kwargs...) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1, F2, F3} 
	setup = setup_fcann_value_arguments(parameters, l2, dropout, use_μP, activation_list; use_gpu = use_gpu)
	!use_gpu && return semi_gradient_dp!(parameters, mdp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, setup.value_function, setup.gradient, setup.update_gradient!; kwargs...)

	isempty(setup.gpu_args) && error("GPU backend is not available")
	output = semi_gradient_dp!(setup.gpu_args.params, mdp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, setup.value_function, setup.gpu_args.gradient, setup.update_gradient!; kwargs...)
	FCANN.GPU2Host(parameters.weights, setup.gpu_args.params.weights)
	setup.gpu_args.cleanup_vars()
	(;output..., final_parameters = parameters)
end

# ╔═╡ a22e5d34-4b8d-479c-985c-d6abd41a6c80
md"""
### Example 10.1: Mountain Car Task
"""

# ╔═╡ b990ba67-42c8-4ab9-943d-085392204fdd
md"""
#### Defining Car State and Dynamics
"""

# ╔═╡ f221fb13-4ef2-4ebe-b71b-fe6adbddb1e4
module MountainCarTask
	import ..TabularRL
	const actions = [-1f0, 0f0, 1f0]
	const action_names = ["Decelerate", "Nothing", "Accelerate"]

	function initialize_state()
		a = rand(Float32) * 0.2f0
		x = a - 0.6f0
		ẋ = 0f0
		(x, ẋ)
	end

	min_vals = (-1.2f0, -0.07f0)
	max_vals = (0.5f0, 0.07f0)

	function step(s::Tuple{Float32, Float32}, a::Float32)
		ẋ′ = clamp(s[2] + 0.001f0*a - 0.0025f0*cos(3*s[1]), -0.07f0, 0.07f0)
		x′ = clamp(s[1] + ẋ′, -1.2f0, 0.5f0)
		x′ == -1.2f0 && return (x′, 0f0)
		return (x′, ẋ′)
	end

	function step(s::Tuple{Float32, Float32}, i_a::Int64)
		a = actions[i_a]
		s′ = step(s, a)
		return (-1f0, s′)
	end

	#We can use these to create a sampling transition function, although it will be deterministic.  The positions and velocities are still defined by two real numbers so the state space is unbounded and we cannot use a tabular method.

	const rlist = ones(Float32, 1)
	const slist = [initialize_state()]
	const plist = ones(Float32, 1)
	function dist_step(s::Tuple{Float32, Float32}, i_a::Int64)
		(r, s′) = step(s, i_a)
		rlist[1] = r
		slist[1] = s′
		plist[1] = 1f0
		(rlist, slist, plist)
		# ([r], [s′], [1f0])
	end

	transition = TabularRL.StateMDPTransitionSampler(step, initialize_state())
	transition_distribution = TabularRL.StateMDPTransitionDistribution(dist_step, initialize_state())
	transition_deterministic = TabularRL.StateMDPTransitionDeterministic(step, initialize_state())

	isterm(s::Tuple{Float32, Float32}) = s[1] == 0.5f0

	mdp = TabularRL.StateMDP(actions, transition, initialize_state, isterm)
	dist_mdp = TabularRL.StateMDP(actions, transition_distribution, initialize_state, isterm)
	deterministic_mdp = TabularRL.StateMDP(actions, transition_deterministic, initialize_state, isterm)
end

# ╔═╡ 1e9c537a-a731-4b81-8f6a-cb658b52c5be
# ╠═╡ skip_as_script = true
#=╠═╡
const mountain_car_mdp = MountainCarTask.mdp
  ╠═╡ =#

# ╔═╡ 5b2ffd90-ead0-42ce-999a-584ed8995910
# ╠═╡ skip_as_script = true
#=╠═╡
const mountain_car_dist_mdp = MountainCarTask.dist_mdp
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
	tr3 = scatter(y = [MountainCarTask.actions[i] for i in actions], showlegend = false)
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
	state_transition_map = zeros(Int64, 3, length(states))
	reward_transition_map = zeros(Float32, 3, length(states))

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
			for (i_a, a) in enumerate(MountainCarTask.actions)
				(r, s′) = MountainCarTask.step(s, i_a)
				i_s′ = bucket_state(s′)
				state_transition_map[i_a, i_s] = i_s′
				reward_transition_map[i_a, i_s] = r
			end
		end
	end

	init_state_index() = bucket_state(MountainCarTask.initialize_state())
	ptf = TabularDeterministicTransition(state_transition_map, reward_transition_map)
	(mdp = TabularMDP(states, MountainCarTask.actions, ptf, init_state_index), assign_state_index = bucket_state)
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
mountaincar_value_iteration = value_iteration_v(tabular_mountaincar_mdp, 1f0; save_history = false, show_message = false, make_final_policy = TabularRL.make_greedy_bit_policy)
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
function plot_mountaincar_action_values(tabular_mountaincar_mdp, value_function::Matrix{Float32}, π::AbstractMatrix{T}, mountaincar_positions, mountaincar_velocities) where T<:Real
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
function plot_mountaincar_action_values(tabular_mountaincar_mdp, value_function::Vector{Float32}, π::AbstractMatrix{T}, mountaincar_positions, mountaincar_velocities) where T<:Real
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
function show_mountaincar_trajectory(tabular_mountaincar_mdp, π::AbstractMatrix{T}, max_steps::Integer, name) where T<:Real
	state_indices, actions, rewards, sterm, nsteps = runepisode(tabular_mountaincar_mdp; π = π, max_steps = max_steps)
	states = [tabular_mountaincar_mdp.states[i_s] for i_s in state_indices]
	positions = [s[1] for s in states]
	velocities = [s[2] for s in states]
	tr1 = scatter(x = positions, y = velocities, mode = "markers", showlegend = false)
	tr2 = scatter(y = positions, showlegend = false)
	tr3 = scatter(y = [MountainCarTask.actions[i] for i in actions], showlegend = false)
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
show_mountaincar_trajectory(s -> constant_params.action, constant_params.nsteps, "Mountain Car Trajectory for $(MountainCarTask.action_names[constant_params.action]) only Policy")
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
#### Linear Approximation with Tile Coding
"""

# ╔═╡ 742100ba-c38e-4840-8988-40990039b527
"""
    setup_mountain_car_tiles(tile_size, num_tilings) -> NamedTuple

Set up tile coding features for the Mountain Car environment.

Convenience function that creates tile coding feature representation specifically configured
for Mountain Car state space with position range [-1.2, 0.5] and velocity range [-0.07, 0.07].

# Arguments
- `tile_size::NTuple{2, Float32}`: Size of tiles in (position, velocity) dimensions
- `num_tilings::Integer`: Number of overlapping tilings for feature coverage

# Returns
- `NamedTuple`: Same as [`tile_coding_feature_setup`](@ref) with Mountain Car-specific configuration

# See Also
[`tile_coding_feature_setup`](@ref), [`MountainCarTask.mdp`](@ref)

# Algorithm Details
1. Delegates to [`tile_coding_feature_setup`](@ref) with Mountain Car parameters:
   - MDP structure from [`MountainCarTask.mdp`](@ref)
   - State bounds: position ∈ [-1.2, 0.5], velocity ∈ [-0.07, 0.07]
   - Displacement vector (1, 3) for tiling offset
2. Returns complete tile coding setup for immediate use
"""
setup_mountain_car_tiles(tile_size::NTuple{2, Float32}, num_tilings::Integer) = tile_coding_feature_setup(MountainCarTask.mdp, (-1.2f0, -0.07f0), (0.5f0, 0.07f0), tile_size, num_tilings)

# ╔═╡ e5c0b558-4902-455f-a370-cddb9b291c15
setup_mountain_car_tiles((1f0/12, 1f0/12), 8)

# ╔═╡ 9ffad966-a568-437a-b9ab-522c08ba681c
md"""
##### Sarsa Solution
"""

# ╔═╡ 7c5fb569-81f0-4b70-ae95-1fce0c51b6f4
# ╠═╡ skip_as_script = true
#=╠═╡
function mountaincar_test(max_episodes::Integer, α::Float32, ϵ::Float32; num_tiles = 12, num_tilings = 8, algo = semi_gradient_sarsa_linear, kwargs...)
	setup = setup_mountain_car_tiles((1f0/num_tiles, 1f0/num_tiles), num_tilings)
	algo(mountain_car_mdp, 1f0, max_episodes, typemax(Int64), setup.feature_vector, setup.update_feature_vector!; α = α, ϵ = ϵ, kwargs...)
end
  ╠═╡ =#

# ╔═╡ 30ab21ba-3f5b-46a8-8b8c-753f2755d419
# ╠═╡ skip_as_script = true
#=╠═╡
const (q̂_mountain_car, episode_rewards, episode_steps) = mountaincar_test(5000, 0.1f0/8, 0.01f0)
  ╠═╡ =#

# ╔═╡ ae1adf97-1a2d-44ff-98ab-422899afd096
#=╠═╡
q̂_mountain_car(mountain_car_mdp.initialize_state())
  ╠═╡ =#

# ╔═╡ f2201afe-8952-4dde-9e39-02beeb920f6f
# ╠═╡ skip_as_script = true
#=╠═╡
show_mountaincar_trajectory(s -> q̂_mountain_car(s).maximizing_action, 1_000, "Sarsa Learned Policy")
  ╠═╡ =#

# ╔═╡ af97f222-08d1-4200-a10b-8da178182175
md"""
##### Dynamic Programming Solution
"""

# ╔═╡ 224b4bec-9ec5-434d-a950-f5974cd786d0
md"""
##### Linear Tile Coding

Since we are only learning the value function, the same tiling setup will have fewer parameters than the action value techniques.  Empirically, more tilings are necessary to learn a state value function that can approach the optimal policy.
"""

# ╔═╡ b0cc6ff8-7296-461c-9db7-e52fa518e2e2
#=╠═╡
function mountaincar_dist_test(max_episodes::Integer, α::Float32, ϵ::Float32; num_tiles = 20, num_tilings = 10, max_steps = typemax(Int64), mdp = mountain_car_dist_mdp, kwargs...)
	setup = setup_mountain_car_tiles((1f0/num_tiles, 1f0/num_tiles), num_tilings)
	semi_gradient_dp_linear(mdp, 1f0, max_episodes, max_steps, setup.feature_vector, setup.update_feature_vector!; α = α, ϵ = ϵ, kwargs...)
end
  ╠═╡ =#

# ╔═╡ d0cf3806-05c6-4a50-94c8-55c9042d51b7
# ╠═╡ skip_as_script = true
#=╠═╡
const (q̂_dp_mountain_car, episode_rewards_dp, episode_steps_dp, param_history_dp, final_params_dp) = mountaincar_dist_test(5_000, 0.06f0/10, 0.01f0; mdp = MountainCarTask.deterministic_mdp)
  ╠═╡ =#

# ╔═╡ 7d21c4cd-ab79-4f40-9b8b-f637b3efcab0
# ╠═╡ skip_as_script = true
#=╠═╡
show_mountaincar_trajectory(s -> q̂_dp_mountain_car(s).maximizing_action, 1_000, "DP Learned Policy")
  ╠═╡ =#

# ╔═╡ 31fb07d2-1c34-44ec-b932-a598e78ec8dc
md"""
#### Non-linear Approximation with Neural Network

Compared to tile coding, the feature vector for non-linear learning will simply be two values: one for the position and another for the velocity.  Both values will be scaled so the mean value is 0 and the variance is approximately 1.
"""

# ╔═╡ c12070a9-df63-4b25-99e6-26ff876af1b4
"""
    update_mountaincar_feature_vector!(v, s) -> Vector{Float32}

Update feature vector with normalized Mountain Car state features.

In-place function that transforms Mountain Car state (position, velocity) into normalized
feature representation suitable for function approximation. Applies scaling and
centering to map state bounds to appropriate feature ranges.

# Arguments
- `v::Vector{Float32}`: Feature vector storage (modified in-place)
- `s::NTuple{2, Float32}`: Mountain Car state as (position, velocity) tuple

# Returns
- `Vector{Float32}`: The modified feature vector (same as input `v`)

# See Also
[`setup_mountain_car_tiles`](@ref), [`scale_state`](@ref)

# Algorithm Details
1. Normalizes position: maps [-1.2, 0.5] to centered and scaled range
2. Normalizes velocity: maps [-0.07, 0.07] to scaled range
3. Stores transformed features in first two elements of feature vector
4. Returns the modified vector for chaining operations
"""
function update_mountaincar_feature_vector!(v::Vector{Float32}, s::NTuple{2, Float32})
	x1 = 3.45f0*(((s[1] - 1.2f0) / 1.7f0) - 0.5f0)
	x2 = 1.725f0*s[2] / 0.07f0
	v[1] = x1
	v[2] = x2
	return v
end;

# ╔═╡ 680561af-db37-440c-9c48-2969e8fd99fc
md"""
##### Sarsa Solution
"""

# ╔═╡ c11aa069-93c2-435a-8f0e-353ced9633b6
# ╠═╡ skip_as_script = true
#=╠═╡
function mountaincar_fcann_sarsa_test(max_steps::Integer, α::Float32, ϵ::Float32; usetiles = false, num_layers = 3, layer_size = 2, algo = semi_gradient_sarsa_fcann, kwargs...)
	function update_feature_vector!(v::Vector{Float32}, s::NTuple{2, Float32})
		x1 = 3.45f0*(((s[1] - 1.2f0) / 1.7f0) - 0.5f0)
		x2 = 1.725f0*s[2] / 0.07f0
		v[1] = x1
		v[2] = x2
	end
	layers = fill(layer_size, num_layers)

	x, f! = if usetiles
		setup = setup_mountain_car_tiles((1/12f0, 1/12f0), 12)
		setup.feature_vector, setup.update_feature_vector!
	else
		zeros(Float32, 2), update_feature_vector!
	end
	
	algo(mountain_car_mdp, 1f0, 100, max_steps, x, f!, layers; α = α, ϵ = ϵ, kwargs...)
end
  ╠═╡ =#

# ╔═╡ 5fdbce61-ca25-45e0-b07d-94adf7138446
# ╠═╡ skip_as_script = true
#=╠═╡
const mountain_car_fcann_sarsa = mountaincar_fcann_sarsa_test(100_000, 1f-6, 0.01f0; num_layers = 4, layer_size = 64, compute_value = compute_sarsa_value, reslayers=1, use_gpu=false, usetiles=false)
  ╠═╡ =#

# ╔═╡ 7cef3dab-7091-4293-a2fb-edddb15a8af8
#=╠═╡
show_mountaincar_trajectory(s -> rand() < 0.05 ? rand(1:3) : mountain_car_fcann_sarsa.value_function(s; mountain_car_fcann_sarsa.form_kwargs()...).maximizing_action, 1_000, "Sarsa Learned Policy")
  ╠═╡ =#

# ╔═╡ 6bcd0ce5-f059-4adc-9cec-c51d0b98ce19
md"""
##### Dynamic Programming Solution
"""

# ╔═╡ 0f958535-6b18-46de-a1ba-81f64c217ee0
function mountaincar_fcann_dp(max_steps::Integer, α::Float32, ϵ::Float32, layers::Vector{Int64}, mdp::StateMDP = mountaincar_dist_mdp; usetiles = false, kwargs...)
	x, f! = if usetiles
		setup = setup_mountain_car_tiles((1f0/10, 1f0/10), 10)
		setup.feature_vector, setup.update_feature_vector!
	else
		zeros(Float32, 2), update_mountaincar_feature_vector!
	end
	semi_gradient_dp_fcann(mdp, 1f0, 100, max_steps, x, f!, layers; α = α, ϵ = ϵ, kwargs...)
end

# ╔═╡ ee59176e-24b6-4213-8f8e-759a70bc1d5e
# ╠═╡ skip_as_script = true
#=╠═╡
const mountaincar_fcann_dp_results = mountaincar_fcann_dp(100_000, 1f-4, 0.01f0, fill(64, 4), MountainCarTask.deterministic_mdp; reslayers = 1, use_gpu = false)
  ╠═╡ =#

# ╔═╡ 1e224a46-91ef-4a5f-ae35-ef4062147f2d
# ╠═╡ skip_as_script = true
#=╠═╡
show_mountaincar_trajectory(s -> rand() < 0.05 ? rand(1:3) : mountaincar_fcann_dp_results.value_function(s).maximizing_action, 1_000, "DP Learned Policy")
  ╠═╡ =#

# ╔═╡ 00399548-b21c-43b5-90e2-30656ab1541e
# ╠═╡ skip_as_script = true
#=╠═╡
plot(scatter(y = -mountaincar_fcann_dp_results.episode_rewards), Layout(yaxis_type = "log"))
  ╠═╡ =#

# ╔═╡ 1a82ae95-3c3e-4281-bc1d-9eb19bf50286
# ╠═╡ skip_as_script = true
#=╠═╡
function figure_10_2(;α_list = [0.1f0, 0.2f0, 0.5f0], num_episodes = 50, ϵ = 0.01f0, num_trials = 100)
	traces = map(α_list) do α
		scatter(y = 1:num_trials |> Map(_ -> mountaincar_test(num_episodes, α/8, ϵ; num_tiles = 12, num_tilings = 8).episode_rewards) |> foldxt((a, b) -> a .+ b) |> v -> -v ./ 100, name = "α = $α/8")
	end
	plot(traces, Layout(xaxis_title = "Episode", yaxis_title = "Steps per episode<br>averaged over 100 runs", yaxis_type = "log"))
end
  ╠═╡ =#

# ╔═╡ ddcb50be-5287-47f8-89f9-58c026a6b151
# ╠═╡ skip_as_script = true
#=╠═╡
figure_10_2(;num_episodes = 500)
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
			q̂ = q̂_mountain_car((x, v))
			values[j, i] = q̂.maximizing_value
			actions[j, i] = MountainCarTask.actions[q̂.maximizing_action]
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

# ╔═╡ 4afbb723-340b-4d85-9115-027a0ff8dfad
# ╠═╡ skip_as_script = true
#=╠═╡
plot_mountaincar_action_values(q̂_mountain_car, 500, 500)
  ╠═╡ =#

# ╔═╡ bd1f42e5-94cc-4aef-b82a-9bffd1c951d8
# ╠═╡ skip_as_script = true
#=╠═╡
plot_mountaincar_action_values(q̂_dp_mountain_car, 500, 500)
  ╠═╡ =#

# ╔═╡ fc3e0577-45aa-4bba-a275-fa7a352fc5cc
# ╠═╡ skip_as_script = true
#=╠═╡
plot_mountaincar_action_values(mountain_car_fcann_sarsa.value_function, 200, 200)
  ╠═╡ =#

# ╔═╡ b3658e4d-ee8e-45cd-906a-06dd512a6921
# ╠═╡ skip_as_script = true
#=╠═╡
plot_mountaincar_action_values(mountaincar_fcann_dp_results.value_function, 200, 200)
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
"""
    compute_expected_sarsa_value(action_values, policy, i_a) -> Real

Compute expected value for Expected SARSA algorithm.

Calculates the expected action value under the given policy by taking the dot product
of action values and policy probabilities. Used as the target value computation in
Expected SARSA updates.

# Type Parameters
- `T <: Real`: Numeric type for computations

# Arguments
- `action_values::Vector{T}`: Current action values for all actions
- `policy::Vector{T}`: Policy probabilities for all actions
- `i_a::Integer`: Index of selected action (unused in expected value computation)

# Returns
- `T`: Expected value under the policy

# See Also
[`compute_q_learning_value`](@ref), [`semi_gradient_expected_sarsa_linear`](@ref)
"""
compute_expected_sarsa_value(action_values::Vector{T}, policy::Vector{T}, i_a::Integer) where T<:Real = dot(action_values, policy);

# ╔═╡ 8b7e1031-9864-439c-86eb-11aa08f53b90
"""
    semi_gradient_double_sarsa!(parameters1, parameters2, mdp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, update_action_values!, ∇q̂, update_value_gradient!; kwargs...) -> NamedTuple

Semi-gradient Double SARSA algorithm for control with function approximation.

Performs on-policy temporal difference control using two sets of parameters to reduce maximization bias.
Randomly selects one parameter set for updates while using the other for target computation, with 
policy improvement based on averaged action-values.

# Type Parameters
- `P`: Parameter type (Vector or Matrix)
- `T <: Real`: Numeric type for computations

# Arguments
- `parameters1::P`: First set of action-value function parameters (modified in-place)
- `parameters2::P`: Second set of action-value function parameters (modified in-place)
- `mdp::`[`StateMDP`](@ref): Markov Decision Process structure
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `max_episodes::Integer`: Maximum number of episodes to run
- `max_steps::Integer`: Maximum total steps across all episodes
- `feature_vector`: Feature vector storage for state representations
- `update_feature_vector!::Function`: Function to extract features from states
- `update_action_values!::Function`: Function to compute action-values from features and parameters
- `∇q̂`: Gradient storage for action-value function gradients
- `update_value_gradient!::Function`: Function to compute action-value function gradient

# Keyword Arguments
- `α::T = 0.1`: Learning rate
- `ϵ::T = 0.1`: Exploration probability for ε-greedy policy
- `compute_value::Function = compute_expected_sarsa_value`: Function to compute target values from action-values and policy
- `α_decay::T = 1.0`: Learning rate decay factor
- `decay_step::Integer = typemax(Int64)`: Step at which to begin learning rate decay
- `save_parameter_history::Bool = false`: Whether to save parameter history for both sets
- `kwargs...`: Additional arguments passed to update functions

# Returns
- `NamedTuple`: Results containing:
  - `value_function`: Final action-value function q̂(s) created by [`form_value_function`](@ref) using both parameter sets
  - `episode_rewards::Vector{T}`: Total reward per episode
  - `episode_steps::Vector{Int64}`: Step count per episode
  - `parameter_history::Tuple{Vector{P}, Vector{P}}`: History for both parameter sets (if `save_parameter_history=true`)
  - `final_parameters::Tuple{P, P}`: Copies of final parameters for both sets

# See Also
[`semi_gradient_sarsa!`](@ref), [`form_value_function`](@ref), [`StateMDP`](@ref)

# Algorithm Details
1. Initialize state and select initial action using ε-greedy policy on averaged action-values
2. For each step:
   - Randomly select which parameter set to update (50% probability each)
   - Compute current action-value and gradient using selected parameter set
   - Take action, observe reward r and next state s'
   - Compute action-values with both parameter sets
   - Use non-updated parameter set for target computation with policy from updated set
   - Update selected parameters: θ ← θ + α·δ·∇q̂(s,a)
   - Select next action using ε-greedy policy on sum of both action-value sets
3. Return final averaged action-value function and training statistics

# Performance Notes
- Reduces maximization bias by decoupling action selection from value estimation
- Updates one parameter set per step while using the other for target computation
- Policy improvement uses averaged action-values from both parameter sets
- Compatible with various target computation methods via `compute_value` parameter
"""
function semi_gradient_double_sarsa!(parameters1::P, parameters2::P, mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, update_action_values!::Function, ∇q̂, update_value_gradient!::Function; α = one(T)/10, ϵ = one(T) / 10, compute_value = compute_expected_sarsa_value, α_decay = one(T), decay_step = typemax(Int64), save_parameter_history = false, kwargs...) where {P, T<:Real}
	action_values1 = zeros(T, length(mdp.actions))
	action_values2 = zeros(T, length(mdp.actions))
	policy = copy(action_values1)

	s = mdp.initialize_state()
	update_feature_vector!(feature_vector, s)
	update_action_values!(action_values1, feature_vector, parameters1)
	update_action_values!(action_values2, feature_vector, parameters2)
	policy .= action_values1 .+ action_values2
	make_ϵ_greedy_policy!(policy; ϵ = ϵ)
	i_a = sample_action(policy)
	
	ep = 1
	step = 1
	epreward = zero(T)
	episode_rewards = Vector{T}()
	episode_steps = Vector{Int64}()
	action_values = zeros(T, length(mdp.actions))
	policy = zeros(T, length(mdp.actions))
	decay = one(T)
	parameter_history1 = Vector{P}()
	parameter_history2 = Vector{P}()
	save_parameter_history && push!(parameter_history1, deepcopy(parameters1))
	save_parameter_history && push!(parameter_history2, deepcopy(parameters2))
	
	while (ep <= max_episodes) && (step <= max_steps)
		case1 = rand() < 0.5
		(action_values, parameters) = if case1
			action_values1, parameters1
		else
			action_values2, parameters2
		end
		
		q̂ = action_values[i_a]
		update_value_gradient!(∇q̂, feature_vector, i_a, parameters)

		
		(r, s′) = mdp.ptf(s, i_a)
		epreward += r

		terminated = mdp.isterm(s′)
		if terminated
			s′ = mdp.initialize_state()
			push!(episode_rewards, epreward)
			push!(episode_steps, step)
			epreward = zero(T)
			ep += 1
		end

		update_feature_vector!(feature_vector, s′)
		(max_q1, i_a_max1) = update_action_values!(action_values1, feature_vector, parameters1)
		(max_q2, i_a_max2) = update_action_values!(action_values2, feature_vector, parameters2)
		

		#use the action-values from the parameters not being updated and the hypothetical policy from the parameters being updated to compute the target value
		action_values, i_a′ = if case1
			policy .= action_values1
			action_values2, i_a_max1
		else
			policy .= action_values2
			action_values1, i_a_max2
		end
		make_ϵ_greedy_policy!(policy; ϵ = ϵ)

		q̂′ = if terminated
			zero(T)
		else
			compute_value(action_values, policy, i_a′)
		end

		target = r + γ*q̂′

		δ = target - q̂
		
		decay *= (step > decay_step)*α_decay + (step <= decay_step)
		
		update_params_with_gradient!(parameters, α*decay*δ, ∇q̂)

		#these action values will be used to compute the state-action value for the next state using the updated parameters
		update_action_values!(action_values1, feature_vector, parameters1)
		update_action_values!(action_values2, feature_vector, parameters2)

		#select next action using both sets of updated parameters
		policy .= action_values1 .+ action_values2
		make_ϵ_greedy_policy!(policy; ϵ = ϵ)
		i_a = sample_action(policy)
		
		save_parameter_history && push!(parameter_history1, deepcopy(parameters1))
		save_parameter_history && push!(parameter_history2, deepcopy(parameters2))
		s = s′
		
		step += 1
	end

	q̂, form_kwargs = form_value_function(mdp, update_feature_vector!, update_action_values!, feature_vector, parameters1, parameters2)
	
	return (value_function = q̂, episode_rewards = episode_rewards, episode_steps = episode_steps, parameter_history = (parameter_history1, parameter_history2), final_parameters = (deepcopy(parameters1), deepcopy(parameters2)), form_kwargs = form_kwargs)
end;

# ╔═╡ b8cd582e-26fc-4f21-85cc-950bac60bee0
"""
    semi_gradient_double_sarsa_linear(mdp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!; kwargs...) -> NamedTuple

Semi-gradient Double SARSA algorithm with linear function approximation.

Convenience method that automatically sets up linear approximation components and delegates
to [`semi_gradient_double_sarsa!`](@ref) with appropriate linear functions and gradient storage.
Double SARSA maintains two separate value function approximations to reduce maximization bias
in action selection.

# Type Parameters
- `T <: Real`: Numeric type for computations

# Arguments
- `mdp::`[`StateMDP`](@ref): Markov Decision Process structure
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `max_episodes::Integer`: Maximum number of episodes to run
- `max_steps::Integer`: Maximum total steps across all episodes
- `feature_vector::`[`LinearFeatureVector`](@ref): Template feature vector for linear approximation
- `update_feature_vector!::Function`: Function to extract features from states

# Keyword Arguments
- `init_value::T = zero(T)`: Initial value for all parameters in both approximators
- `parameters1::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, init_value)`: Parameter matrix for first approximator
- `parameters2::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, init_value)`: Parameter matrix for second approximator
- `kwargs...`: Additional arguments passed to [`semi_gradient_double_sarsa!`](@ref)

# Returns
- `NamedTuple`: Same as [`semi_gradient_double_sarsa!`](@ref) - see that function for details

# See Also
[`semi_gradient_double_sarsa!`](@ref), [`semi_gradient_sarsa_linear`](@ref), [`LinearFeatureVector`](@ref), [`update_linear_action_values!`](@ref), [`LinearActionValueGradient`](@ref)

# Algorithm Details
1. Creates two parameter matrices using [`initialize_linear_parameters`](@ref) if not provided
2. Sets up [`LinearActionValueGradient`](@ref) for gradient storage
3. Delegates to [`semi_gradient_double_sarsa!`](@ref) with:
   - [`update_linear_action_values!`](@ref) for action-value computation
   - [`update_linear_value_gradient!`](@ref) for gradient updates
4. Returns results from core algorithm with both approximators
"""
semi_gradient_double_sarsa_linear(mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), parameters1::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, init_value), parameters2::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, init_value), kwargs...) where T<:Real = semi_gradient_double_sarsa!(parameters1, parameters2, mdp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, update_linear_action_values!, LinearActionValueGradient(deepcopy(feature_vector), 0), update_linear_value_gradient!; kwargs...)

# ╔═╡ 8cdf042f-2214-48e0-afc2-c6a7d385ee4e
function semi_gradient_double_sarsa_fcann(mdp::StateMDP, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; reslayers::Integer = 0, use_μP::Bool = true, parameters1::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, length(mdp.actions), reslayers, use_μP), parameters2::FCANNParams{T} = deepcopy(parameters1), dropout = zero(T), activation_list = fill(true, length(hidden_layers)), l2 = zero(T), kwargs...) where T<:Real 
	setup = setup_fcann_action_value_arguments(parameters1, l2, dropout, use_μP, activation_list)
	
	semi_gradient_double_sarsa!(parameters1, parameters2, mdp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, setup.update_action_values!, setup.gradient, setup.update_value_gradient!; kwargs...)
end

# ╔═╡ 1410db13-4b73-4a87-af34-30a5232af4ba
"""
    compute_q_learning_value(action_values, policy, i_a) -> Real

Compute maximum value for Q-learning algorithm.

Calculates the maximum action value for Q-learning updates. Policy argument is ignored
as Q-learning uses the maximum over all actions regardless of the current policy.

# Type Parameters
- `T <: Real`: Numeric type for computations

# Arguments
- `action_values::Vector{T}`: Current action values for all actions
- `policy::Vector{T}`: Policy probabilities (unused in max computation)
- `i_a::Integer`: Index of selected action (unused in max computation)

# Returns
- `T`: Maximum action value

# See Also
[`compute_expected_sarsa_value`](@ref), [`semi_gradient_q_learning_linear`](@ref)
"""
compute_q_learning_value(action_values::Vector{T}, policy::Vector{T}, i_a::Integer) where T<:Real = maximum(action_values);

# ╔═╡ f7410fe7-e3d8-4047-8fa7-f076476e9d3a
md"""
### Example: Semi-gradient Q-learning on Mountain Car Task
"""

# ╔═╡ cbac1927-b087-4c4c-98ae-6aa5f0b824ad
# ╠═╡ skip_as_script = true
#=╠═╡
(q̂_mountain_car_q, episode_rewards_q, episode_steps_q) = mountaincar_test(2_000, .8f0/20, 0.01f0; compute_value = compute_q_learning_value, algo = semi_gradient_double_sarsa_linear)
  ╠═╡ =#

# ╔═╡ b5409b69-a254-4355-b2b9-99394eceb2f7
# ╠═╡ skip_as_script = true
#=╠═╡
show_mountaincar_trajectory(s -> q̂_mountain_car_q(s).maximizing_action, 1_000, "Q-Learning Learned Policy")
  ╠═╡ =#

# ╔═╡ f9ee13e8-7406-4fba-9a30-1e2714bd7cfc
# ╠═╡ skip_as_script = true
#=╠═╡
plot_mountaincar_action_values(q̂_mountain_car_q, 500, 500)
  ╠═╡ =#

# ╔═╡ 7c160da9-d546-42f8-ad99-7e74c96cabe5
#=╠═╡
const mountaincar_fcann_results2 = mountaincar_fcann_sarsa_test(100_000, 1f-6, 0.01f0; layers = fill(64, 4), reslayers=1, compute_value = compute_q_learning_value, algo = semi_gradient_double_sarsa_fcann)
  ╠═╡ =#

# ╔═╡ 00bd6bfc-2ea6-4fcc-8c51-cb7aabb5ce25
#=╠═╡
show_mountaincar_trajectory(s -> mountaincar_fcann_results2.value_function(s).maximizing_action, 1_000, "Q-Learning Learned Policy")
  ╠═╡ =#

# ╔═╡ 26712d1f-d5d1-4784-967c-f1682c3e07aa
#=╠═╡
plot_mountaincar_action_values(mountaincar_fcann_results2.value_function, 200, 200)
  ╠═╡ =#

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

where the expectations are conditioned on the initial state, $S_0$, and on the subsequent actions, $A_0, A_1, \dots,A_{t-1}$, being taken according to $\pi$. The second and third equations hold if the steady-state distribution $\mu_\pi(s) \doteq \lim_{t\rightarrow \infty} \Pr \{S_t = s \mid A_{0:t-1} \sim \pi \}$, exists and is independent of $S_0$, in other words, if the MDP is *ergodic*. In an ergodic MDP, the starting state and any early decision made by the agent can only have a temporary effect; in the long run the expectation of being in a state depends on the policy and the MDP transition probabilities.  Ergodicity is sufficient but not necessary to guarantee the existence of the limit in (10.6).

In this setting, we consider all policies that obtain the maximum value of $r(\pi)$ or the *reward rate* to be optimal.  Note that the steady state distribution $\mu_\pi$ is the special distribution under which, if you select actions according to $\pi$, you remain in the same distribution.  That is, for which 

$\sum_s \mu_\pi(s) \sum_a \pi(a\vert s)p(s^\prime \vert s, a) = \mu_\pi(s^\prime) \tag{10.8}$

In the average-reward setting, returns are defined in terms of differences between rewards and the average reward: 

$G_t \doteq R_{t+1} - r(\pi) + R_{t+2} - r(\pi) + R_{t+3} - r(\pi) + \cdots \tag{10.9}$

This is known as the *differential* return, and the corresponding value functions are known as *differential* value functions.  Differential value functions are defined in terms of the new return just as conventional value functions were defined in terms of the discounted return; thus we will use the same notation, $v_\pi (s) \doteq \mathbb{E}_\pi[G_t \vert S_t = s]$ and $q_\pi (s, a) \doteq \mathbb{E}_\pi[G_t \vert S_t = s, A_t = a]$ (similarly for $v_*$ and $q_*$), for differential value functions.  Differential value functions also have Bellman equations, just slightly different from those we have seen earlier.  We simply remove all $\gamma$s and replace all rewards by the difference between the reward and the true average reward:

$\begin{flalign}
&v_\pi(s) = \sum_a \pi(a\vert s) \sum_{r, s^\prime}p(s^\prime, r \vert s, a) \left [ r - r(\pi) + v_\pi(s^\prime) \right ] \\
&q_\pi(s, a) = \sum_{r, s^\prime}p(s^\prime, r \vert s, a) \left [ r - r(\pi) + \sum_{a^\prime} \pi(a^\prime \vert s^\prime) q_\pi(s^\prime, a^\prime) \right ] \\
&v_* = \max_a \sum_{r, s^\prime}p(s^\prime, r \vert s, a) \left [ r - \max_{\pi}r(\pi) + v_*(s^\prime) \right ] \\
&q_* = \sum_{r, s^\prime}p(s^\prime, r \vert s, a) \left [ r - \max_{\pi}r(\pi) + \max_a q_\pi(s^\prime, a^\prime) \right ] \\
\end{flalign}$

There is also a differential form of the two TD errors:

$\delta_t \doteq R_{t+1} - \bar{R}_t+ \hat v (S_{t+1}, \mathbf{w}_t) - \hat v(S_t, \mathbf{w}_t) \tag{10.10}$

and

$\delta_t \doteq R_{t+1} - \bar{R}_t+ \hat q (S_{t+1}, A_{t+1}, \mathbf{w}_t) - \hat q(S_t, A_t, \mathbf{w}_t) \tag{10.11}$

where $\bar{R}_t$ is an estimate at time $t$ of the average reward $r(\pi)$.  With these alternate definitions, most of our algorithms and many theoretical results carry through to the average reward setting without any change.  

For example, an average reward version of semi-gradient Sarsa could be defined just as in (10.2) except with the differential version of the TD error.  That is by

$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t \nabla \hat q(S_t, A_t, \mathbf{w}_t)$

with $\delta_t$ given by (10.11).  See a full implementation below.  One limitation of this algorithm is that it does not converge to the differential values but to the differential values plut an arbitrary offset.  Notice that the Bellman equations and TD errors given above are unaffected if all the values are shifted by the same amount.  Thus, the offset may not matter in practice.
"""

# ╔═╡ 5f9a2231-8c4a-4519-adbe-a0dd92838ba4
md"""
### Necessary Conditions for Average Reward Setting

- Continuing task (i.e. no terminal state)
- The average reward exists: $\lim_{h \rightarrow \infty} \frac{1}{h} \sum_{t=1}^h \mathbb{E}[R_t \mid S_0, A_{0:t-1} \sim \pi]$
  - To be useful, this limit should depend on the policy
  - If the reward is -1 per step for example, the average reward will always be -1 regardless of the policy
- The long term limit of expected value of the reward exists: $\lim_{t \rightarrow \infty} \mathbb{E}[R_t \mid S_0, A_{0:t-1} \sim \pi]$
  - This limit only exists if the expected reward approaches a single value in the limit of infinite time away from some initial state $S_0$
  - If the limit exists, then in the limit, the sum above will be dominated by this value as more terms are added, wiping out any impact from the early rewards in the sequence
  - A repeating reward sequence would violate this assumption since the expected value of $R_t$ would be a specific value depending on the time step
- The steady-state distribution exists and is independent of $S_0$: $\mu_\pi(s) \doteq \lim_{t \rightarrow \infty} \Pr \{ S_t = s \mid A_{0:t-1} \sim \pi \}$
  - In this case we can express $r(\pi)$ without reference to a starting state and simply use the policy, transition function, and steady-state distribution
  - The expected value of the reward must also not depend on $S_0$ as well
  - If either distribution or expected value exist, but depend on the starting state, then we cannot use the Bellman expressions with the average reward
"""

# ╔═╡ 4154e827-6d0b-4b94-9f14-64baa85739af
md"""
### Discounted Setting Connection to Average Reward

Consider an environment with a reset state and a goal state, like a gridworld.  As a continuing problem, this can be formulated with a reward of 0 for every step and some positive reward r at the goal.  If this problem is solved in the average reward setting, then the policy would minimize the expected number of steps to reach the goal from any state including the reset state.  

What if instead it is treated as an episodic problem with a reward of -1 per step and a termination condition at the goal.  In the discounted setting, we can define the value function of a state $s$ and a policy $\pi$ as $v_\pi(s) = -\mathbb{E}_\pi \left [ \sum_{k=0}^{N_s - 1} \gamma^k \right ]$ where $N_s$ is the number of steps until terminationfrom state $s$.  For an arbitrarily policy and environment, there could be some distribution over $N_s$.  In the undiscounted case where $\gamma = 1$, the value at each state is simply: $v_\pi(s) = -\mathbb{E}_\pi [N_s]$ which is also minus the average number of steps until termination.  So regardless of the distribution, maximizing the value is equivalent to minimizing the expected number of steps until termination.  This policy will be equivalent to that found in the average reward setting.

Now consider the discounted case with some $0 \le \gamma \lt 1$.  $v_\pi(s) = -\mathbb{E}_\pi \left [ \sum_{k=0}^{N_s - 1} \gamma^k \right ] = - \mathbb{E}_\pi \left [ \frac{\gamma^{N_s} - 1}{\gamma - 1} \right ] = \frac{1 - \mathbb{E}_\pi [ \gamma ^{N_s}]}{\gamma - 1}$ where now the value depends on the average of $\gamma^{N_s}$ rather than $N_s$.  One notable difference is that in the undiscounted case, larger values of $N_s$ contribute negatively to the value in an unbounded fashion.  In the discounted case, $0 \le \gamma \lt 1, N_s \rightarrow \infty \implies \gamma^{N_s} \rightarrow 0 \implies v_\pi(s) \rightarrow \frac{1}{1-\gamma}$.  In the plot below, we can see this limiting value come into play especially for smaller $\gamma$ so that the negative contribution from very large $N_s$ is capped.  Therefore, it is always beneficial to symetrically add probabilities of finishing in a shorter time and the correspondingly longer time on the other side of it.
"""

# ╔═╡ e90a591d-0bd0-46a9-8327-f61bfb155a31
md"""
In the plot below, the distribution $p(N_s) = \frac{1}{N_s+2d}; N_s - d \le N_s \le N_s + d$ and 0 otherwise.  It is a uniform distribution around the mean value. Note that for a given discount rate and average number of steps, the discounted value will increase with $d$ while the expected value of $N_s$ is unchanged.
"""

# ╔═╡ 7cb7ca66-3130-4f06-a0dc-a3335ef85fdb
#=╠═╡
@bind graphparams2 PlutoUI.combine() do Child
md"""
Discount Rate: $(Child(:γ, Slider(0:0.00001:1; default = 0.9, show_value=true)))

Average Steps: $(Child(:N, Slider(1:100; default = 100, show_value=true)))

Half-Width Spread: $(Child(:d, Slider(0:100; default = 50, show_value=true)))
"""
end
  ╠═╡ =#

# ╔═╡ 37db9d03-1978-4842-a016-f416c33ba1d7
md"""
For the expected value calculation, each of these values is added with equal probability.  That is why the discounted value favors higher variance because the longer episodes are not counted as much as in the undiscounted case where γ = 1.  The expected discounted value does converge to the undiscounted one as γ approaches 1.  One consequence of favoring higher variance is that a policy could be favored by producing episodes with a longer average number of steps, as long as the variance increases enough.  

Consider the same distribution above where the distribution of $N_s$ is uniform from $N_s - d$ to $N_s + d$.  Below is a derivation of the value function in this case and a graph that uses this value to find how much of an increase in $d$ is required to favor a higher $N$.  Different lines are shown from different discount rates.  In the limit of $\gamma = 1$, there is no value for $d$ that would result in higher values since the lowest average step is always favored.
"""

# ╔═╡ 81a0a342-f92a-4f5a-a173-fd555188895f
md"""
$\begin{flalign}
\mathbb{E} \left [ \gamma ^ N \right ] &= \sum_N p(N) \gamma^N \\
&=\frac{1}{2d+1} \sum_{k = N-d}^{N+d} \gamma^k \\
&= \frac{\gamma^{N-d} + \gamma^{N-d+1} + \cdots + \gamma^{N+d}}{2d+1} \\
&= \frac{\gamma^{N-d} (1 + \gamma + \gamma^2 + \cdots + \gamma^{2d})}{2d+1} \\
&= \frac{\gamma^{N-d} (\gamma^{2d+1} - 1)}{(2d+1)(\gamma - 1)} \\
\end{flalign}$

Therefore:

$\begin{flalign}
v_\pi(s) &= \frac{1 - \mathbb{E}[\gamma^{N_s}]}{\gamma - 1}\\
&= \frac{1}{\gamma - 1} - \frac{\gamma^{N-d} (\gamma^{2d+1} - 1)}{(2d+1)(\gamma - 1)^2}
\end{flalign}$
"""

# ╔═╡ a92db2a7-3a6c-4328-a2a8-cb74d2e671e9
#=╠═╡
@bind graphparams PlutoUI.combine() do Child
md"""
Average Steps: $(Child(:N, Slider(1:100; default = 50, show_value=true)))

Half-Width Spread: $(Child(:d, Slider(0:100; default = 0, show_value=true)))
"""
end
  ╠═╡ =#

# ╔═╡ 25159f84-a120-4a20-aab8-010c110571a4
# ╠═╡ skip_as_script = true
#=╠═╡
function uniform_value(γ::Real, d::Integer, N::Integer)
	(1- γ^(2*d + 1))*(γ^(N-d))/((γ-1)^2 * (2*d + 1)) + inv(γ-1)
end
  ╠═╡ =#

# ╔═╡ 5a9bcf45-a04b-4a81-b825-9891021c8a15
# ╠═╡ skip_as_script = true
#=╠═╡
function get_equivalent_values(N::Integer, d::Integer, γ::Real; nmax::Integer = 10)
	v0 = uniform_value(γ, d, N)
	δout = []
	for i in 1:nmax
		v = uniform_value(γ, d, N+i)
		δ = 0
		while v <= v0
			δ += 1
			v = uniform_value(γ, d + δ, N+i)
		end
		if δ + d > N
			push!(δout, nothing)
		else
			push!(δout, δ)
		end
	end
	return δout
	# out = zeros(dmax, nmax)
	# for i in 1:dmax
	# 	for j in 1:nmax
	# 		out[i, j] = uniform_value(graphparams.γ, d+i, N + j) > v0
	# 	end
	# end
	# plot(heatmap(x = 1:dmax, y = 1:nmax, z = out), Layout(xaxis_title = "n steps", yaxis_title = "δ Steps"))
end
  ╠═╡ =#

# ╔═╡ 44b7a560-d03b-4636-ad24-b30c8965ab8f
#=╠═╡
plot([scatter(x = 1:graphparams.N, y = get_equivalent_values(graphparams.N, graphparams.d, γ; nmax = graphparams.N); mode = "markers+lines", name = "γ = $γ") for γ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.995, 0.996, 0.997, 0.998]], Layout(xaxis_title = "Increase in Average Steps", yaxis_title = "Required Increase in Spread"))
# plot(scatter(y = get_equivalent_values(graphparams.N, graphparams.d, graphparams.γ); name = "γ = $(graphparams.γ)"))
  ╠═╡ =#

# ╔═╡ e6bf5b6e-75cd-49b3-bf36-7ed6dee11aaf
#=╠═╡
function show_distributions(N::Integer, d::Integer, γ::Real)
	nvals = N-d:N+d
	range = 2*d + 1
	uniform_p = inv(range)
	yvals = (γ .^ nvals .- 1) ./ (γ- 1)
	n̄ = sum(uniform_p .* nvals)
	γn̄ = sum(yvals .* uniform_p)
	tr1 = bar(x = nvals, y = -nvals, name = "Undiscounted Values")
	tr2 = bar(x = nvals, y = -yvals, name = "Discounted Values")
	plot([tr1, tr2], Layout(title = "Expected Value $(-n̄), Expected Discounted Value = $(-γn̄)"))
end
  ╠═╡ =#

# ╔═╡ 4a5805f0-0cff-4e30-8305-304340734232
#=╠═╡
show_distributions(graphparams2.N, graphparams2.d, graphparams2.γ)
  ╠═╡ =#

# ╔═╡ 69a06405-57cd-42e5-96b1-5cc77d74aa03
md"""
### *Differential Semi-gradient Sarsa Implementation*
"""

# ╔═╡ a9fdb1fd-3f62-4e1c-9157-c4eee6215261

"""
    semi_gradient_differential_sarsa!(parameters, mdp, max_episodes, max_steps, feature_vector, update_feature_vector!, update_action_values!, ∇q̂, update_value_gradient!; kwargs...) -> NamedTuple

Semi-gradient Differential SARSA algorithm for continuing tasks.

Core implementation of differential SARSA with function approximation for average reward
continuing tasks. Maintains an estimate of the average reward and uses differential returns
(rewards minus average) for value updates without discounting.

# Type Parameters
- `T <: Real`: Numeric type for computations
- `S`: State type
- `A`: Action type  
- `P`: Transition probability type
- `F1, F2, F3`: Function types for MDP structure
- `PR`: Parameter type for function approximation

# Arguments
- `parameters::PR`: Function approximation parameters (modified in-place)
- `mdp::`[`StateMDP`](@ref): Markov Decision Process structure
- `max_episodes::Integer`: Maximum number of episodes to run
- `max_steps::Integer`: Maximum total steps across all episodes
- `feature_vector`: Feature vector storage for state representations
- `update_feature_vector!::Function`: Function to extract features from states
- `update_action_values!::Function`: Function to compute action values from features
- `∇q̂`: Gradient storage for action-value gradients
- `update_value_gradient!::Function`: Function to compute action-value gradients

# Keyword Arguments
- `α::T = one(T)/10`: Learning rate for value function parameters
- `β::T = one(T)/100`: Learning rate for average reward estimation
- `ϵ::T = one(T)/10`: Exploration parameter for ε-greedy policy
- `compute_value::Function = compute_sarsa_value`: Function to compute target values
- `max_only_update::Bool = false`: Whether to update average reward only for greedy actions
- `save_parameter_history::Bool = false`: Whether to save parameter evolution
- `kwargs...`: Additional arguments

# Returns
- `NamedTuple` with fields:
  - `value_function`: Closure for action-value function evaluation
  - `episode_rewards`: Cumulative rewards per episode
  - `episode_steps`: Step counts per episode  
  - `average_step_reward`: Evolution of average reward estimate
  - `parameter_history`: Parameter evolution (if saved)
  - `final_parameters`: Final trained parameters

# See Also
[`semi_gradient_differential_sarsa_linear`](@ref), [`semi_gradient_differential_sarsa_fcann`](@ref), [`make_ϵ_greedy_policy!`](@ref)

# Algorithm Details
1. Initializes average reward estimate R̄ = 0 and bias correction factor ō = 0
2. For each step:
   - Computes differential return U_t = r - R̄ + q̂'
   - Updates parameters using temporal difference δ = U_t - q̂
   - Updates average reward estimate using bias-corrected update
   - Continues without episode termination for average reward criterion
3. Returns trained value function and learning statistics
"""
function semi_gradient_differential_sarsa!(parameters::PR, mdp::StateMDP{T, S, A, P, F1, F2, F3}, num_steps::Integer, feature_vector, update_feature_vector!::Function, update_action_values!::Function, ∇q̂, update_value_gradient!::Function; α = one(T)/10, α_r̄ = one(T)/100, ϵ = one(T) / 10, compute_value = compute_sarsa_value, max_only_update = false, save_parameter_history = false, kwargs...) where {T<:Real, S, A, P, F1, F2, F3, PR}
	action_values = zeros(T, length(mdp.actions))
	policy = zeros(T, length(mdp.actions))
	
	s = mdp.initialize_state()
	update_feature_vector!(feature_vector, s)
	update_action_values!(action_values, feature_vector, parameters)
	policy .= action_values
	make_ϵ_greedy_policy!(policy; ϵ = ϵ)
	i_a = sample_action(policy)

	R̄ = zero(T)
	ō = zero(T)
	epreward = zero(T)
	
	#initialize records
	reward_history = zeros(T, num_steps)
	average_reward_history = zeros(T, num_steps)
	parameter_history = Vector{PR}(undef, num_steps)
	
	for step in 1:num_steps
		update_value_gradient!(∇q̂, action_values, feature_vector, i_a, parameters)
		q̂ = action_values[i_a]
		
		(r, s′) = mdp.ptf(s, i_a)
		U_t = r - R̄
		reward_history[step] = r
		average_reward_history[step] = R̄

		mdp.isterm(s′) && error("$s′ is a terminal state and this method only applies to continuing tasks")

		update_feature_vector!(feature_vector, s′)
		q_max, i_a_max = update_action_values!(action_values, feature_vector, parameters)
		
		policy .= action_values
		make_ϵ_greedy_policy!(policy; ϵ = ϵ)
		i_a′ = sample_action(policy)
		
		q̂′ = compute_value(action_values, policy, i_a′)
		
		U_t += q̂′
		δ = U_t - q̂

		update_params_with_gradient!(parameters, α*δ, ∇q̂)
		
		if !max_only_update || (q_max == action_values[i_a′])
			ō += α_r̄ * (one(T) - ō)
			R̄ += (α_r̄/ō)*δ
		end
		
		save_parameter_history && (parameter_history[step] = copy(parameters))
		s = s′
		i_a = i_a′
	end

	q̂, form_kwargs = form_value_function(mdp, update_feature_vector!, update_action_values!, feature_vector, parameters)
	
	return (value_function = q̂, reward_history = reward_history, average_reward_history = average_reward_history, parameter_history = parameter_history, final_parameters = deepcopy(parameters), form_kwargs = form_kwargs) 
end;

# ╔═╡ efee131c-318a-40d6-be83-ce24edbbe11c
md"""
#### *Linear Approximation*
"""

# ╔═╡ aceeb425-cd5f-4c4c-903e-d4359d2de88d
"""
    semi_gradient_differential_sarsa_linear(mdp, max_episodes, max_steps, feature_vector, update_feature_vector!; kwargs...) -> NamedTuple

Semi-gradient Differential SARSA algorithm with linear function approximation.

Convenience method that automatically sets up linear approximation components and delegates
to [`semi_gradient_differential_sarsa!`](@ref) with appropriate linear functions and gradient storage.
Designed for continuing tasks with average reward criterion.

# Type Parameters
- `T <: Real`: Numeric type for computations
- `S`: State type
- `A`: Action type
- `P <: AbstractStateTransition`: Transition type
- `F1, F2, F3`: Function types for MDP structure

# Arguments
- `mdp::`[`StateMDP`](@ref): Markov Decision Process structure
- `max_episodes::Integer`: Maximum number of episodes to run
- `max_steps::Integer`: Maximum total steps across all episodes
- `feature_vector::`[`LinearFeatureVector`](@ref): Template feature vector for linear approximation
- `update_feature_vector!::Function`: Function to extract features from states

# Keyword Arguments
- `init_value::T = zero(T)`: Initial value for all parameters
- `parameters::Matrix{T} = initialize_linear_parameters(...)`: Pre-initialized parameter matrix
- `kwargs...`: Additional arguments passed to [`semi_gradient_differential_sarsa!`](@ref)

# Returns
- `NamedTuple`: Same as [`semi_gradient_differential_sarsa!`](@ref)

# See Also
[`semi_gradient_differential_sarsa!`](@ref), [`semi_gradient_differential_sarsa_fcann`](@ref), [`LinearFeatureVector`](@ref), [`update_linear_action_values!`](@ref)

# Algorithm Details
1. Creates parameter matrix using [`initialize_linear_parameters`](@ref) if not provided
2. Sets up [`LinearActionValueGradient`](@ref) for gradient storage
3. Delegates to [`semi_gradient_differential_sarsa!`](@ref) with linear approximation functions
4. Returns results from core differential SARSA algorithm
"""
semi_gradient_differential_sarsa_linear(mdp::StateMDP{T, S, A, P, F1, F2, F3}, num_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), parameters::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, init_value), kwargs...) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3} = semi_gradient_differential_sarsa!(parameters, mdp, num_steps, feature_vector, update_feature_vector!, update_linear_action_values!, LinearActionValueGradient(deepcopy(feature_vector), 0), update_linear_value_gradient!; kwargs...)

# ╔═╡ 9b629126-0b8f-4592-8727-cbe710bd4a24
md"""
#### *Non-linear Approximation*
"""

# ╔═╡ db778942-1bed-4c42-a2f0-a176a0364772
"""
    semi_gradient_differential_sarsa_fcann(mdp, max_episodes, max_steps, update_feature_vector!, num_features, hidden_layers; kwargs...) -> NamedTuple

Semi-gradient Differential SARSA algorithm with fully-connected neural network approximation.

Convenience method that automatically sets up FCANN approximation components and delegates
to [`semi_gradient_differential_sarsa!`](@ref) with appropriate neural network functions and gradient storage.
Designed for continuing tasks with average reward criterion.

# Type Parameters
- `T <: Real`: Numeric type for computations
- `S`: State type
- `A`: Action type
- `P <: AbstractStateTransition`: Transition type
- `F1, F2, F3`: Function types for MDP structure

# Arguments
- `mdp::`[`StateMDP`](@ref): Markov Decision Process structure
- `max_episodes::Integer`: Maximum number of episodes to run
- `max_steps::Integer`: Maximum total steps across all episodes
- `update_feature_vector!::Function`: Function to extract features from states
- `num_features::Integer`: Dimension of input feature vectors
- `hidden_layers::Vector{Int64}`: Number of units in each hidden layer

# Keyword Arguments
- `reslayers::Integer = 0`: Number of residual layers
- `use_μP::Bool = true`: Whether to apply μP scaling
- `parameters::`[`FCANNParams`](@ref)`{T} = FCANN.initializeparams_saxe(...)`: Pre-initialized network parameters
- `dropout::T = zero(T)`: Dropout rate for training
- `activation_list = fill(true, length(hidden_layers))`: Activation configuration per layer
- `l2::T = zero(T)`: L2 regularization strength
- `kwargs...`: Additional arguments passed to [`semi_gradient_differential_sarsa!`](@ref)

# Returns
- `NamedTuple` with fields:
  - `value_function`: Closure for action-value function evaluation with activation management
  - `episode_rewards`: Cumulative rewards per episode
  - `episode_steps`: Step counts per episode
  - `average_step_reward`: Evolution of average reward estimate
  - `parameter_history`: Parameter evolution (if saved)
  - `final_parameters`: Final trained network parameters

# See Also
[`semi_gradient_differential_sarsa!`](@ref), [`semi_gradient_differential_sarsa_linear`](@ref), [`setup_fcann_action_value_arguments`](@ref), [`FCANNParams`](@ref)

# Algorithm Details
1. Sets up FCANN components using [`setup_fcann_action_value_arguments`](@ref)
2. Initializes network parameters with [`FCANN.initializeparams_saxe`](@ref) if not provided
3. Delegates to [`semi_gradient_differential_sarsa!`](@ref) with neural network functions
4. Returns wrapped value function with activation storage management
"""
function semi_gradient_differential_sarsa_fcann(mdp::StateMDP{T, S, A, P, F1, F2, F3}, num_steps::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; reslayers::Integer = 0, use_μP::Bool = true, parameters::FCANNParams{T} = initialize_fcann_params(length(feature_vector), hidden_layers, length(mdp.actions), reslayers, use_μP), dropout = zero(T), activation_list = fill(true, length(hidden_layers)), l2 = zero(T), use_gpu::Bool = false, kwargs...) where {T<:Real, S, A, P<:AbstractStateTransition, F1, F2, F3}
	setup = setup_fcann_action_value_arguments(parameters, l2, dropout, use_μP, activation_list; use_gpu = use_gpu)
	!use_gpu && return semi_gradient_differential_sarsa!(parameters, mdp, num_steps, feature_vector, update_feature_vector!, setup.update_action_values!, setup.gradient, setup.update_value_gradient!; kwargs...)

	isempty(setup.gpu_args) && error("GPU backend is not available")
	gpu_feature_update! = setup_gpu_feature(feature_vector, update_feature_vector!)
	output = semi_gradient_differential_sarsa!(setup.gpu_args.params, mdp, num_steps, setup.gpu_args.feature_vector, gpu_feature_update!, setup.update_action_values!, setup.gpu_args.gradient, setup.update_value_gradient!; kwargs...)
	FCANN.GPU2Host(parameters.weights, setup.gpu_args.params.weights)
	setup.gpu_args.cleanup_vars()
	(;output..., final_parameters = parameters)
end

# ╔═╡ 063e6f33-8b65-463c-a96f-5411f0ba0326
md"""
### *Differential Semi-gradient Dynamic Programming Implementation*
"""

# ╔═╡ 91447aff-5598-4f02-acd5-6a90c563f4f6
begin
"""
    update_differential_action_values!(action_values, s, v̂, mdp, R̄) -> Tuple{Real, Integer}

Update action values for differential dynamic programming.

Computes action values using differential returns (rewards minus average reward) plus
expected next state values. Used in differential DP where the value function represents
differential values relative to the average reward baseline.

# Type Parameters
- `T <: Real`: Numeric type for computations
- `S`: State type
- `A`: Action type
- `P <: StateMDPTransitionDistribution`: Transition distribution type
- `F1, F2, F3 <: Function`: MDP function types

# Arguments
- `action_values::Vector{T}`: Action value storage (modified in-place)
- `s`: Current state
- `v̂::Function`: State value function for differential values
- `mdp::`[`StateMDP`](@ref): MDP with transition distributions
- `R̄::T`: Current average reward estimate

# Returns
- `Tuple{T, Integer}`: Maximum action value and corresponding action index

# See Also
[`form_differential_value_function`](@ref), [`semi_gradient_differential_dp!`](@ref)

# Algorithm Details
1. For each action, computes expected immediate reward and next state value
2. Forms differential action value: q = r_avg - R̄ + E[v̂(s')]
3. Tracks maximum value and corresponding action during computation
4. Updates action_values vector in-place with differential values
"""
function update_differential_action_values!(action_values::Array{T, N}, s, feature_vector, update_feature_vector!::Function, value_function::Function, parameters, mdp::StateMDP{T, S, A, P, F1, F2, F3}, R̄::T; kwargs...) where {T<:Real, S, A, P<:StateMDPTransitionDistribution, F1<:Function, F2<:Function, F3<:Function, N}
	maxq = typemin(T)
	i_a_max = 0
	for i_a in eachindex(action_values)
		(rewards, states, probabilities) = mdp.ptf.step(s, i_a)
		v′ = zero(T) 
		r_avg = zero(T)
		for i in eachindex(probabilities)
			s′ = states[i]
			if !mdp.isterm(s′)
				update_feature_vector!(feature_vector, s′)
				v′ += probabilities[i] * value_function(feature_vector, parameters; kwargs...)
			end
			r_avg += probabilities[i]*rewards[i]
		end
		q = r_avg - R̄ + v′
		action_values[i_a] = q
		newmax = q > maxq
		maxq = newmax*q + !newmax*maxq
		i_a_max = newmax*i_a + !newmax*i_a_max
	end
	return maxq, i_a_max
end

	function update_differential_action_values!(action_values::Array{T, N}, s, feature_vector::V, update_feature_vector!::Function, value_function::Function, parameters::Vector{T}, mdp::StateMDP{T, S, A, P, F1, F2, F3}, R̄::T; kwargs...) where {T<:Real, S, A, P<:StateMDPTransitionDeterministic, F1<:Function, F2<:Function, F3<:Function, V<:Union{BinaryFeatureVector, StateAggregationFeatureVector}, N}
		maxq = typemin(T)
		i_a_max = 0
		for i_a in eachindex(action_values)
			r, s′ = mdp.ptf(s, i_a)
			update_feature_vector!(feature_vector, s′)
			v̂ = value_function(feature_vector, parameters; kwargs...)
			q = r - R̄ + v̂
			action_values[i_a] = q
			newmax = q > maxq
			maxq = newmax*q + !newmax*maxq
			i_a_max = newmax*i_a + !newmax*i_a_max
		end
		return maxq, i_a_max
	end

	function update_differential_action_values!(action_values::Array{T, N}, s, feature_vector, update_feature_vector!::Function, value_function::Function, parameters, mdp::StateMDP{T, S, A, P, F1, F2, F3}, R̄::T, reward_values::Vector{T}, feature_matrix, activations; kwargs...) where {T<:Real, S, A, P<:StateMDPTransitionDeterministic, F1<:Function, F2<:Function, F3<:Function, N}
		for i_a in eachindex(action_values)
			r, s′ = mdp.ptf.step(s, i_a)
			update_feature_vector!(feature_vector, s′)
			update_feature_matrix!(feature_matrix, feature_vector, i_a)
			reward_values[i_a] = r #populate action value vector with reward, will be added to the future state value later
		end
		update_state_values!(action_values, feature_matrix, parameters, activations)
		action_values .= reward_values .- R̄ .+ action_values
		vmax, imax = findmax(action_values)
		isinf(vmax) && @warn "Infinite action value found in state $s out of $action_values"
		isnan(vmax) && @warn "NaN action value found in state $s out of $action_values"
		return (vmax, prod(Tuple(imax)))
	end

	function update_differential_action_values!(action_values::Array{T, N}, s, feature_vector::Vector{T}, update_feature_vector!::Function, value_function::Function, parameters::FCANNParamsGPU, mdp::StateMDP{T, S, A, P, F1, F2, F3}, R̄::T, reward_values::Vector{T}, feature_matrix::Matrix{T}, gpu_matrix::FCANN.CUDAArray, activations; kwargs...) where {T<:Real, S, A, P<:StateMDPTransitionDeterministic, F1<:Function, F2<:Function, F3<:Function, N}
		for i_a in eachindex(action_values)
			r, s′ = mdp.ptf.step(s, i_a)
			update_feature_vector!(feature_vector, s′)
			update_feature_matrix!(feature_matrix, feature_vector, i_a)
			reward_values[i_a] = r #populate action value vector with reward, will be added to the future state value later
		end
		FCANN.memcpy!(gpu_matrix, feature_matrix)
		update_state_values!(action_values, gpu_matrix, parameters, activations)
		action_values .= reward_values .- R̄ .+ action_values
		vmax, imax = findmax(action_values)
		isinf(vmax) && @warn "Infinite action value found in state $s out of $action_values"
		isnan(vmax) && @warn "NaN action value found in state $s out of $action_values"
		return (vmax, prod(Tuple(imax)))
	end
end

# ╔═╡ 4e955391-ac29-412e-8ed2-bad3b46961b0
begin
	"""
	    form_differential_value_function(mdp, R̄, update_feature_vector!, value_function, feature_vector, parameters) -> Function
	
	Create action-value function for differential dynamic programming.
	
	Forms a closure that computes differential action values using the trained state value function
	and current average reward estimate. Returns both action values and greedy action information.
	
	# Type Parameters
	- `T <: Real`: Numeric type for computations
	- `S`: State type  
	- `A`: Action type
	- `P <: StateMDPTransitionDistribution`: Transition distribution type
	- `F1, F2, F3 <: Function`: MDP function types
	- `V`: Feature vector type
	- `W`: Parameter type
	
	# Arguments
	- `mdp::`[`StateMDP`](@ref): MDP with transition distributions
	- `R̄::T`: Average reward estimate from training
	- `update_feature_vector!::Function`: Feature extraction function
	- `value_function::Function`: Trained state value function
	- `feature_vector::V`: Template feature vector
	- `parameters::W`: Trained parameters
	
	# Returns
	- `Function`: Action-value function q̂(s) returning NamedTuple with action_values, maximizing_action, maximizing_value
	
	# See Also
	[`update_differential_action_values!`](@ref), [`form_state_value_function`](@ref), [`semi_gradient_differential_dp!`](@ref)
	
	# Algorithm Details
	1. Creates state value function closure using [`form_state_value_function`](@ref)
	2. Returns action-value function that uses [`update_differential_action_values!`](@ref)
	3. Manages feature vector and parameter storage for efficient evaluation
	4. Provides both action values and greedy policy information
	"""
	function form_differential_value_function(mdp::StateMDP{T, S, A, P, F1, F2, F3}, R̄::T, update_feature_vector!::Function, value_function::Function, feature_vector::V, parameters::W) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1<:Function, F2<:Function, F3<:Function, V, W}
		function q̂(s::S; action_values::Matrix{T} = zeros(T, length(mdp.actions), 1), feature_vector::V = deepcopy(feature_vector), parameters::W = parameters, action_value_args = form_action_value_args(mdp, feature_vector, parameters), kwargs...)
			maxq, i_a_max = update_differential_action_values!(action_values, s, feature_vector, update_feature_vector!, value_function, parameters, mdp, R̄, action_value_args...; kwargs...)
			(action_values = action_values, maximizing_action = i_a_max, maximizing_value = maxq)
		end
	
		form_kwargs() = (action_values = zeros(T, length(mdp.actions), 1), feature_vector = deepcopy(feature_vector), parameters = parameters, action_value_args = form_action_value_args(mdp, feature_vector, parameters))
		return q̂, form_kwargs
	end

	function form_differential_value_function(mdp::StateMDP{T, S, A, P, F1, F2, F3}, R̄::T, update_feature_vector!::Function, value_function::Function, feature_vector::V, parameters::W) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1<:Function, F2<:Function, F3<:Function, V <: Vector{T}, W <: FCANNParamsGPU}
		cpu_params = initialize_cpu_params(parameters)
		gpu_params = initialize_gpu_params(cpu_params)
		
		function q̂(s::S; use_gpu::Bool = false, parameters = cpu_params, gpu_params = gpu_params, kwargs...)
			if !use_gpu
				q̂(s, parameters; kwargs...)
			else
				q̂(s, gpu_params; kwargs...)
			end
		end

		function q̂(s::S, parameters::FCANNParams{T}; action_values::Matrix{T} = zeros(T, length(mdp.actions), 1), feature_vector::Vector{T} = copy(feature_vector), action_value_args = form_action_value_args(mdp, feature_vector, parameters), kwargs...)
			maxq, i_a_max = update_differential_action_values!(action_values, s, feature_vector, update_feature_vector!, value_function, parameters, mdp, R̄, action_value_args...; kwargs...)
			(action_values = action_values, maximizing_action = i_a_max, maximizing_value = maxq)
		end

		function q̂(s::S, parameters::FCANNParamsGPU; action_values::Matrix{T} = zeros(T, length(mdp.actions), 1), feature_vector::Vector{T} = copy(feature_vector), gpu_action_value_args = form_action_value_args(mdp, feature_vector, parameters), kwargs...)
			maxq, i_a_max = update_differential_action_values!(action_values, s, feature_vector, update_feature_vector!, value_function, parameters, mdp, R̄, gpu_action_value_args...; kwargs...)
			(action_values = action_values, maximizing_action = i_a_max, maximizing_value = maxq)
		end

		form_kwargs() = (action_values = zeros(T, length(mdp.actions), 1), parameters = cpu_params, feature_vector = copy(feature_vector), action_value_args = form_action_value_args(mdp, feature_vector, cpu_params), gpu_action_value_args = form_action_value_args(mdp, feature_vector, gpu_params))
		return q̂, form_kwargs
	end

	function form_differential_value_function(mdp::StateMDP{T, S, A, P, F1, F2, F3}, R̄::T, update_feature_vector!::Function, value_function::Function, feature_vector::V, parameters::W) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1<:Function, F2<:Function, F3<:Function, V <: FCANN.CUDAArray, W <: FCANNParamsGPU}
		cpu_feature = FCANN.host_allocate(feature_vector)
		form_differential_value_function(mdp, R̄, update_feature_vector!, value_function, cpu_feature, parameters)
	end
end

# ╔═╡ 12fa7b75-d13f-4a16-8562-1142002f3f3f
"""
    semi_gradient_differential_dp!(parameters, mdp, max_episodes, max_steps, feature_vector, update_feature_vector!, value_function, ∇v̂, update_value_gradient!; kwargs...) -> NamedTuple

Semi-gradient Differential Dynamic Programming algorithm for continuing tasks.

Core implementation of differential DP using state value function approximation for average
reward continuing tasks. Updates the state value function using differential Bellman targets
computed from the full action-value backup.

# Type Parameters
- `T <: Real`: Numeric type for computations
- `S`: State type
- `A`: Action type
- `P <: StateMDPTransitionDistribution`: Transition distribution type
- `F1, F2, F3 <: Function`: MDP function types
- `PR`: Parameter type

# Arguments
- `parameters::PR`: State value function parameters (modified in-place)
- `mdp::`[`StateMDP`](@ref): MDP with transition distributions
- `max_episodes::Integer`: Maximum number of episodes to run
- `max_steps::Integer`: Maximum total steps across all episodes
- `feature_vector`: Feature vector storage for state representations
- `update_feature_vector!::Function`: Function to extract features from states
- `value_function::Function`: State value function (features, params) -> value
- `∇v̂`: Gradient storage for state value gradients
- `update_value_gradient!::Function`: Function to compute state value gradients

# Keyword Arguments
- `α::T = one(T)/10`: Learning rate for value function parameters
- `β::T = one(T)/100`: Learning rate for average reward estimation
- `ϵ::T = one(T)/10`: Exploration parameter for ε-greedy policy
- `α_decay::T = one(T)`: Decay factor for learning rate
- `decay_step::Integer = typemax(Int64)`: Step at which to start learning rate decay
- `save_parameter_history::Bool = false`: Whether to save parameter evolution
- `kwargs...`: Additional arguments

# Returns
- `NamedTuple` with fields:
  - `value_function`: Function q̂(s) that returns NamedTuple with fields (action_values, maximizing_action, maximizing_value)
  - `episode_rewards`: Cumulative rewards per episode
  - `episode_steps`: Step counts per episode
  - `average_step_reward`: Evolution of average reward estimate
  - `parameter_history`: Parameter evolution (if saved)
  - `final_parameters`: Final trained parameters

# See Also
[`semi_gradient_differential_dp_linear`](@ref), [`semi_gradient_differential_dp_fcann`](@ref), [`update_differential_action_values!`](@ref), [`form_differential_value_function`](@ref)

# Algorithm Details
1. Trains state value function V(s) using differential DP targets
2. For each step:
   - Computes all action values using [`update_differential_action_values!`](@ref)
   - Uses maximum action value as target for state value update
   - Updates average reward estimate R̄ only for greedy actions
   - Applies ε-greedy policy for action selection
3. Returns differential action-value function using trained state value function
"""
function semi_gradient_differential_dp!(parameters::PR, mdp::StateMDP{T, S, A, P, F1, F2, F3}, num_steps::Integer, feature_vector, update_feature_vector!::Function, value_function::Function, ∇v̂, update_value_gradient!::Function; α = one(T)/10, α_r̄ = one(T)/100, ϵ = one(T) / 10, α_decay = one(T), decay_step = typemax(Int64), save_parameter_history = false, kwargs...) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1<:Function, F2<:Function, F3<:Function, PR}
	action_values = zeros(T, length(mdp.actions), 1)
	policy = zeros(T, length(mdp.actions))

	#initialize records
	reward_history = zeros(T, num_steps)
	average_reward_history = zeros(T, num_steps)
	parameter_history = Vector{PR}(undef, num_steps)
	
	s = mdp.initialize_state()

	action_value_args = form_action_value_args(mdp, feature_vector, parameters)
	
	
	decay = one(T)
	R̄ = zero(T)
	ō = zero(T)
	for step in 1:num_steps
		save_parameter_history && (parameter_history[step] = copy(parameters))
		update_feature_vector!(feature_vector, s)
		v̂ = update_value_gradient!(∇v̂, feature_vector, parameters)
		
		#computes q and finds maximizing action value, this is effectively trajectory sampling in the case of approximation where we stay close to the optimal policy
		target, i_a_max = update_differential_action_values!(action_values, s, feature_vector, update_feature_vector!, value_function, parameters, mdp, R̄, action_value_args...)
		
		δ = target - v̂

		decay *= (step > decay_step)*α_decay + (step <= decay_step)
		update_params_with_gradient!(parameters, α*decay*δ, ∇v̂)

		policy .= action_values
		make_ϵ_greedy_policy!(policy; ϵ = ϵ)
		i_a = sample_action(policy)

		(r, s′) = mdp.ptf(s, i_a)
		reward_history[step] = r
		average_reward_history[step] = R̄

		mdp.isterm(s′) && error("$s′ is a terminal state and this method only applies to continuing tasks")
		
		#only update average reward for actions that match the greedy policy
		if action_values[i_a] == target
			ō += α_r̄ * (one(T) - ō)
			R̄ += (α_r̄/ō)*(target - R̄ - v̂)
		end

		# if step <= 3
		# 	@info "v̂ = $v̂, target = $target, δ = $δ, R̄ = $R̄, action values are $action_values"
		# end

		s = s′
	end

	q̂, form_kwargs = form_differential_value_function(mdp, R̄, update_feature_vector!, value_function, feature_vector, parameters)
	
	return (value_function = q̂, reward_history = reward_history, average_reward_history = average_reward_history, parameter_history = parameter_history, final_parameters = deepcopy(parameters), form_kwargs = form_kwargs)
end;

# ╔═╡ 9b56eac4-10be-42c3-b3a9-a0c4852b7cce
md"""
#### *Linear Approximation*
"""

# ╔═╡ 7c22d050-bd56-4b84-8a01-e575475db099
"""
    semi_gradient_differential_dp_linear(mdp, max_episodes, max_steps, feature_vector, update_feature_vector!; kwargs...) -> NamedTuple

Semi-gradient Differential Dynamic Programming algorithm with linear function approximation.

Convenience method that automatically sets up linear approximation components and delegates
to [`semi_gradient_differential_dp!`](@ref) with appropriate linear functions and gradient storage.
Uses state value function approximation for continuing tasks.

# Type Parameters
- `T <: Real`: Numeric type for computations
- `S`: State type
- `A`: Action type
- `P <: StateMDPTransitionDistribution`: Transition distribution type
- `F1, F2, F3`: Function types for MDP structure

# Arguments
- `mdp::`[`StateMDP`](@ref): MDP with transition distributions
- `max_episodes::Integer`: Maximum number of episodes to run
- `max_steps::Integer`: Maximum total steps across all episodes
- `feature_vector::`[`LinearFeatureVector`](@ref): Template feature vector for linear approximation
- `update_feature_vector!::Function`: Function to extract features from states

# Keyword Arguments
- `init_value::T = zero(T)`: Initial value for all parameters
- `parameters::Vector{T} = initialize_linear_parameters(...)`: Pre-initialized parameter vector
- `kwargs...`: Additional arguments passed to [`semi_gradient_differential_dp!`](@ref)

# Returns
- `NamedTuple`: Same as [`semi_gradient_differential_dp!`](@ref)

# See Also
[`semi_gradient_differential_dp!`](@ref), [`semi_gradient_differential_dp_fcann`](@ref), [`linear_value_function`](@ref), [`update_linear_value_gradient!`](@ref)

# Algorithm Details
1. Creates parameter vector using [`initialize_linear_parameters`](@ref) if not provided
2. Sets up gradient storage with feature vector copy
3. Delegates to [`semi_gradient_differential_dp!`](@ref) with linear approximation functions
4. Returns results from core differential DP algorithm
"""
semi_gradient_differential_dp_linear(mdp::StateMDP{T, S, A, P, F1, F2, F3}, num_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(0f0), parameters::Vector{T} = initialize_linear_parameters(feature_vector, init_value), kwargs...) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1, F2, F3} = semi_gradient_differential_dp!(parameters, mdp, num_steps, feature_vector, update_feature_vector!, linear_value_function, deepcopy(feature_vector), update_linear_value_gradient!; kwargs...)

# ╔═╡ 571fad6e-ca32-4661-bc48-62f3f49d124b
md"""
#### *Non-linear Approximation*
"""

# ╔═╡ e04b9ac4-7e7f-4f6a-b068-d62b319a23fa
"""
	semi_gradient_differential_dp_fcann(mdp, max_episodes, max_steps, update_feature_vector!, num_features, hidden_layers; kwargs...) -> NamedTuple

Semi-gradient Differential Dynamic Programming algorithm with fully-connected neural network approximation.

Convenience method that automatically sets up FCANN approximation components and delegates
to [`semi_gradient_differential_dp!`](@ref) with appropriate neural network functions and gradient storage.
Uses single-output network for state value function approximation in continuing tasks.

# Type Parameters
- `T <: Real`: Numeric type for computations
- `S`: State type
- `A`: Action type
- `P <: StateMDPTransitionDistribution`: Transition distribution type
- `F1, F2, F3`: Function types for MDP structure

# Arguments
- `mdp::`[`StateMDP`](@ref): MDP with transition distributions
- `max_episodes::Integer`: Maximum number of episodes to run
- `max_steps::Integer`: Maximum total steps across all episodes
- `update_feature_vector!::Function`: Function to extract features from states
- `num_features::Integer`: Dimension of input feature vectors
- `hidden_layers::Vector{Int64}`: Number of units in each hidden layer

# Keyword Arguments
- `reslayers::Integer = 0`: Number of residual layers
- `use_μP::Bool = true`: Whether to apply μP scaling
- `parameters::`[`FCANNParams`](@ref)`{T} = FCANN.initializeparams_saxe(...)`: Pre-initialized network parameters for single output
- `dropout::T = zero(T)`: Dropout rate for training
- `activation_list = fill(true, length(hidden_layers))`: Activation configuration per layer
- `l2::T = zero(T)`: L2 regularization strength
- `kwargs...`: Additional arguments passed to [`semi_gradient_differential_dp!`](@ref)

# Returns
- `NamedTuple` with fields:
  - `value_function`: Function q̂(s) that returns NamedTuple with fields (action_values, maximizing_action, maximizing_value)
  - `episode_rewards`: Cumulative rewards per episode
  - `episode_steps`: Step counts per episode
  - `average_step_reward`: Evolution of average reward estimate
  - `parameter_history`: Parameter evolution (if saved)
  - `final_parameters`: Final trained network parameters

# See Also
[`semi_gradient_differential_dp!`](@ref), [`semi_gradient_differential_dp_linear`](@ref), [`setup_fcann_value_arguments`](@ref), [`FCANNParams`](@ref)

# Algorithm Details
1. Sets up FCANN components using [`setup_fcann_value_arguments`](@ref)
2. Initializes single-output network parameters with [`FCANN.initializeparams_saxe`](@ref) if not provided
3. Delegates to [`semi_gradient_differential_dp!`](@ref) with neural network functions
4. Returns wrapped value function with activation storage management
"""
function semi_gradient_differential_dp_fcann(mdp::StateMDP{T, S, A, P, F1, F2, F3}, num_steps::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; reslayers::Integer = 0, use_μP::Bool = true, parameters::FCANNParams{T} = initialize_fcann_params(length(feature_vector), hidden_layers, 1, reslayers, use_μP), dropout = zero(T), activation_list = fill(true, length(hidden_layers)), l2 = zero(T), use_gpu::Bool = false, kwargs...) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1, F2, F3}
	setup = setup_fcann_value_arguments(parameters, l2, dropout, use_μP, activation_list; use_gpu = use_gpu)
	!use_gpu && return semi_gradient_differential_dp!(parameters, mdp, num_steps, feature_vector, update_feature_vector!, setup.value_function, setup.gradient, setup.update_gradient!; kwargs...)

	isempty(setup.gpu_args) && error("GPU backend is not available")
	output = semi_gradient_differential_dp!(setup.gpu_args.params, mdp, num_steps, feature_vector, update_feature_vector!, setup.value_function, setup.gpu_args.gradient, setup.update_gradient!; kwargs...)
	FCANN.GPU2Host(parameters.weights, setup.gpu_args.params.weights)
	setup.gpu_args.cleanup_vars()
	(;output..., final_parameters = parameters)
end	

# ╔═╡ 1a7ba296-52ca-4069-85fa-792d08d77b0e
md"""
### Example: Differential Sarsa and Q-learning with Mountain Car Task

In order to apply differential learning to the mountain car task, we need to change the rewards per step.  Previously, the rewards were assigned in a manner appropriate for learning with a discount rate of 1.  The reward of -1 per episode step ensures that policies that finish the task faster have a higher reward.  In the average reward setting, every policy would have an average reward per step of -1 making the task ill posed.  Instead, we can assign a reward of 1 for finishing to the right and 0 at all other steps.  These rewards would produce an ill posed task for $\gamma = 1$ but are perfectly fine for the average reward setting.  Now our learning procedure should find a policy that produces the highest average reward $\frac{1}{\text{num steps}}$ which is maximized when the number of steps to finish an episode is minimized.
"""

# ╔═╡ eb28458f-b222-4f8e-9a5b-8203d3997f7b
"""
    mountain_car_differential_step(s, i_a) -> Tuple{Float32, Tuple{Float32, Float32}}

Mountain Car transition function for differential reinforcement learning.

Performs one step of Mountain Car dynamics with modified reward structure for continuing
tasks. Returns sparse positive reward only when reaching the goal position (0.5), with
zero reward elsewhere to focus on average reward optimization.

# Arguments
- `s::Tuple{Float32, Float32}`: Current state as (position, velocity) tuple
- `i_a::Int64`: Action index into [`MountainCarTask.actions`](@ref)

# Returns
- `Tuple{Float32, Tuple{Float32, Float32}}`: (reward, next_state) where reward is 1.0 at goal, 0.0 elsewhere

# See Also
[`MountainCarTask.step`](@ref), [`MountainCarTask.actions`](@ref)

# Algorithm Details
1. Maps action index to actual action using [`MountainCarTask.actions`](@ref)
2. Computes next state using [`MountainCarTask.step`](@ref)
3. Returns reward of 1.0 if goal position (0.5) reached, 0.0 otherwise
4. Designed for continuing tasks where average reward rate is the optimization criterion
"""
function mountain_car_differential_step(s::Tuple{Float32, Float32}, i_a::Int64)
	a = MountainCarTask.actions[i_a]
	s′ = MountainCarTask.step(s, a)
	isterm = (s′[1] == 0.5f0)
	r = Float32(isterm)
	!isterm && return (r, s′)
	return (r, MountainCarTask.initialize_state())
end;

# ╔═╡ d66cd124-7111-401a-a3e8-1059b31c6db7
"""
    create_differential_mountaincar_mdp() -> StateMDP

Create Mountain Car MDP configured for differential reinforcement learning.

Constructs a StateMDP with Mountain Car dynamics using modified reward structure suitable
for continuing tasks and average reward optimization. Uses sparse positive rewards only
at the goal state rather than negative step penalties.

# Returns
- [`StateMDP`](@ref): Mountain Car MDP with differential reward structure

# See Also
[`mountain_car_differential_step`](@ref), [`StateMDPTransitionSampler`](@ref), [`MountainCarTask.initialize_state`](@ref), [`MountainCarTask.isterm`](@ref)

# Algorithm Details
1. Creates transition sampler using [`mountain_car_differential_step`](@ref) for reward modification
2. Constructs [`StateMDP`](@ref) with:
   - [`MountainCarTask.actions`](@ref) for action space
   - [`StateMDPTransitionSampler`](@ref) for stochastic transitions
   - [`MountainCarTask.initialize_state`](@ref) for state initialization
   - [`MountainCarTask.isterm`](@ref) for termination checking
3. Returns MDP suitable for differential/average reward algorithms
"""
function create_differential_mountaincar_mdp()
	ptf1 = StateMDPTransitionSampler(mountain_car_differential_step, MountainCarTask.initialize_state())
	ptf2 = StateMDPTransitionDeterministic(mountain_car_differential_step, MountainCarTask.initialize_state())
	mdp1 = StateMDP(MountainCarTask.actions, ptf1, MountainCarTask.initialize_state, Returns(false))
	mdp2 = StateMDP(MountainCarTask.actions, ptf2, MountainCarTask.initialize_state, Returns(false))
	(mdp = mdp1, deterministic_mdp = mdp2)
end;

# ╔═╡ bc1d7cce-c0f4-47a8-b674-8acb82491c7f
# ╠═╡ skip_as_script = true
#=╠═╡
const mountain_car_differential_mdps = create_differential_mountaincar_mdp()
  ╠═╡ =#

# ╔═╡ 0a494e3e-0af5-4497-b80e-e471acc1fabc
md"""
#### Action Value Methods

##### Tile Coding Linear Approximation
"""

# ╔═╡ 49e43d51-05d6-415b-a685-76e50904c5bc
# ╠═╡ skip_as_script = true
#=╠═╡
function mountaincar_differential_test(num_steps::Integer, α::Float32, β::Float32, ϵ::Float32; num_tiles = 16, num_tilings = 10, kwargs...)
	setup = setup_mountain_car_tiles((1f0/num_tiles, 1f0/num_tiles), num_tilings)
	semi_gradient_differential_sarsa_linear(mountain_car_differential_mdps.mdp, num_steps, setup.feature_vector, setup.update_feature_vector!; α = α, β = β, ϵ = ϵ, kwargs...)
end
  ╠═╡ =#

# ╔═╡ db189316-e880-4cc8-9070-ccfe2b4fc545
# ╠═╡ skip_as_script = true
#=╠═╡
(q̂_mountain_car2, reward_history, average_step_reward) = mountaincar_differential_test(1_000_000, 4f-2, 1f-2, 0.01f0; compute_value = compute_sarsa_value)
  ╠═╡ =#

# ╔═╡ 7bc49107-9de5-4985-8750-979f36b3aa81
#=╠═╡
π_mountain_car2(s) = q̂_mountain_car2(s).maximizing_action
  ╠═╡ =#

# ╔═╡ ab4cb3db-3a2d-4145-826b-b1001114eeff
#=╠═╡
show_mountaincar_trajectory(π_mountain_car2, 1_000, "Differential Q-learning Learned Policy")
  ╠═╡ =#

# ╔═╡ 0e3e506d-1959-47fd-8da9-b3dfd294be67
#=╠═╡
plot_mountaincar_action_values(q̂_mountain_car2, 200, 200)
  ╠═╡ =#

# ╔═╡ 53c5558b-e713-4c72-bdf8-e162c3892e6f
#=╠═╡
plot(average_step_reward)
  ╠═╡ =#

# ╔═╡ 86cd431e-7b05-410a-b943-ba03b286f3f0
md"""
##### Non-linear Approximation
"""

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
	semi_gradient_differential_sarsa_fcann(mountain_car_differential_mdps.mdp, max_steps, feature_vector, update_feature_vector!, layers; α = α, β = β, ϵ = ϵ, kwargs...)
end
  ╠═╡ =#

# ╔═╡ ae5c5377-8b44-4c82-a63c-d2cb8a0d6667
#=╠═╡
(q̂_mountain_car2_fcann, fcann_rewards2, average_step_reward_fcann) = mountaincar_fcann_differential_test(100_000, 1f-5, 1f-3, 0.01f0; num_layers = 4, layer_size = 64, reslayers = 1, compute_value = compute_q_learning_value, use_gpu=false)
  ╠═╡ =#

# ╔═╡ 6d4016c6-8edd-4466-9cc4-015452c669ba
#=╠═╡
const cont_sarsa_fcann_test = mountaincar_fcann_differential_test(100, 1f-5, 1f-3, 0.01f0; num_layers = 4, layer_size = 64, reslayers = 1, compute_value = compute_q_learning_value, use_gpu=true)
  ╠═╡ =#

# ╔═╡ 2df5adc1-130b-4982-a4bc-7e0c7417923e
#=╠═╡
cont_sarsa_fcann_test.value_function((0f0, 0f0); use_gpu = false, cont_sarsa_fcann_test.form_kwargs()...)
  ╠═╡ =#

# ╔═╡ 2306039b-7b4d-4013-be1b-1402231ef8e8
#=╠═╡
π_mountain_car2_fcann(s) = q̂_mountain_car2_fcann(s).maximizing_action
  ╠═╡ =#

# ╔═╡ b191d3f9-cf25-4fb4-8f5a-8da86e96e125
#=╠═╡
show_mountaincar_trajectory(π_mountain_car2_fcann, 1_000, "Differential Q-learning Learned Policy")
  ╠═╡ =#

# ╔═╡ c44dd6c6-8213-49fb-8d33-ba8f2c766b2e
#=╠═╡
plot_mountaincar_action_values(q̂_mountain_car2_fcann, 500, 500)
  ╠═╡ =#

# ╔═╡ a6e0c082-7f1f-4352-8c23-c3b64fd74493
md"""
#### Differential Dynamic Programming Methods
"""

# ╔═╡ 6f79c437-7264-412c-839f-5bc9252eede8
md"""
#### Tile Coding Linear Approximation
"""

# ╔═╡ 501b7284-6e04-4a15-b8e4-2601156b0345
#=╠═╡
function mountaincar_differential_dp_test(num_steps::Integer, α::Float32, β::Float32, ϵ::Float32; num_tiles = 20, num_tilings = 10, kwargs...)
	setup = setup_mountain_car_tiles((1f0/num_tiles, 1f0/num_tiles), num_tilings)
	semi_gradient_differential_dp_linear(mountain_car_differential_mdps.deterministic_mdp, num_steps, setup.feature_vector, setup.update_feature_vector!; α = α, β = β, ϵ = ϵ, kwargs...)
end
  ╠═╡ =#

# ╔═╡ 2441b61e-5954-41e2-8ee4-38b16ed04cef
#=╠═╡
const differential_linear_dp_mountaincar = mountaincar_differential_dp_test(1_000_000, 4f-3, 1f-2, 0.01f0)
  ╠═╡ =#

# ╔═╡ 6f4f8b64-0c17-446e-bfb6-0540871ad9e0
#=╠═╡
plot(differential_linear_dp_mountaincar.average_reward_history)
  ╠═╡ =#

# ╔═╡ 1a56e4dd-15dd-47b3-afd8-1dd7f5b690ac
#=╠═╡
show_mountaincar_trajectory(s ->differential_linear_dp_mountaincar.value_function(s).maximizing_action, 1_000, "Differential Linear DP Learned Policy")
  ╠═╡ =#

# ╔═╡ c94da551-06b2-4e2b-bf39-ceb5cb5c390c
#=╠═╡
plot_mountaincar_action_values(differential_linear_dp_mountaincar.value_function, 200, 200)
  ╠═╡ =#

# ╔═╡ 0e34a25b-f8ee-4da9-8664-b6c094163759
md"""
##### Non-linear Approximation
"""

# ╔═╡ 3b66c97b-ebad-4d13-987c-ac0172b349d1
# ╠═╡ skip_as_script = true
#=╠═╡
function mountaincar_differential_dp_nonlinear_test(num_steps::Integer, α::Float32, β::Float32, ϵ::Float32; num_layers = 3, layer_size = 8, kwargs...)
	feature_vector = zeros(Float32, 2)
	function update_feature_vector!(v::Vector{Float32}, s::NTuple{2, Float32})
		x1 = 3.45f0*(((s[1] - 1.2f0) / 1.7f0) - 0.5f0)
		x2 = 1.725f0*s[2] / 0.07f0
		v[1] = x1
		v[2] = x2
	end
	layers = fill(layer_size, num_layers)
	semi_gradient_differential_dp_fcann(mountain_car_differential_mdps.deterministic_mdp, num_steps, feature_vector, update_feature_vector!, layers; α = α, β = β, ϵ = ϵ, kwargs...)
end
  ╠═╡ =#

# ╔═╡ 86f7dcde-b27e-4096-bec8-c5d17fd553d2
#=╠═╡
const differential_nonlinear_dp_mountaincar = mountaincar_differential_dp_nonlinear_test(100_000, 1f-5, 4f-2, 0.01f0; layer_size = 16, num_layers = 4, use_gpu = false)
  ╠═╡ =#

# ╔═╡ defe9c74-d514-44ea-af09-fb77764dfaa4
#=╠═╡
const cont_dp_fcann_test = mountaincar_differential_dp_nonlinear_test(100, 1f-5, 1f-3, 0.01f0; num_layers = 4, layer_size = 64, reslayers = 1, use_gpu=true)
  ╠═╡ =#

# ╔═╡ 7ed745f4-5b7d-41b4-a40e-5b782ca12530
#=╠═╡
cont_dp_fcann_test.value_function((0.5f0, 0f0); use_gpu = true, cont_dp_fcann_test.form_kwargs()...)
  ╠═╡ =#

# ╔═╡ ad692a51-e93b-4480-8a6c-2ad86dc6766b
#=╠═╡
plot(differential_nonlinear_dp_mountaincar.average_reward_history)
  ╠═╡ =#

# ╔═╡ cf00a316-38e3-4423-9909-d5ffbd7c0b06
#=╠═╡
show_mountaincar_trajectory(s -> differential_nonlinear_dp_mountaincar.value_function(s).maximizing_action, 1_000, "Differential Non-Linear DP Learned Policy")
  ╠═╡ =#

# ╔═╡ 89288ce6-11e8-41f3-b32d-e19edee7db33
#=╠═╡
plot_mountaincar_action_values(differential_nonlinear_dp_mountaincar.value_function, 200, 200)
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

$R_1 = 1, R_2 = 0, R_3 = 1, \dots, R_t = \frac{(-1)^{t+1} + 1}{2} \implies \mathbb{E}[R_t] = \frac{(-1)^{t+1} + 1}{2}$

the average reward can be computed using the definition:

$r(\pi) = \lim_{h \rightarrow \infty} \frac{1}{2h}\sum_{t=1}^h \left [ (-1)^{t+1} + 1 \right ]= \lim_{h \rightarrow \infty} \frac{h}{2h} + \frac{1}{2h} \sum_{t=1}^h (-1)^{t+1} = \frac{1}{2} \left [ 1 + \lim_{h \rightarrow \infty} \frac{1}{h}\sum_{t=1}^h (-1)^{t+1} \right ]$

The remaining sum term is simply $1 - 1 + 1 - 1 \cdots$ which can be expressed as:

$1 \geq X_h = \sum_{t=1}^h (-1)^{t+1} = \frac{(-1)^{h+1} + 1}{2} \geq 0$

Since this expression is bounded, we can also bound the limit:

$\lim_{h \rightarrow \infty} \frac{X_h}{h} \geq \lim_{h \rightarrow \infty} \frac{0}{h} = 0$
$\lim_{h \rightarrow \infty} \frac{X_h}{h} \leq \lim_{h \rightarrow \infty} \frac{1}{h} = 0$

Since both bounds are 0, the limit must be 0, resulting in the following expression for the average reward:

$r(\pi) = \frac{1}{2}[1 + 0] = \frac{1}{2}$

---
The differetial value function is defined as: $v_\pi(s) \doteq \mathbb{E}_\pi[G_t \vert S_t = s]$ where $G_t \doteq R_{t+1} - r(\pi) + R_{t+2} - r(\pi) + R_{t+3} - r(\pi) + \cdots$  In this example, we know that $r(\pi) = 0.5$, but the reward sequence is periodic meaning the expected value of the differential return depends on $t$:

$v_\pi (s) \doteq \mathbb{E}_\pi[G_t \vert S_t = s] = \mathbb{E}_\pi \left [ \sum_{h = 1}^{\infty} [R_{t+h} - r(\pi) ]  \mid S_t = s \right ] = \sum_{h = 1}^{\infty} \mathbb{E}_\pi[R_{t+h} \vert S_t = s] - r(\pi)$ 

In particular, the sum does not converge since the values alternate and do not approach a single value.  Thus the differential value function is not well defined.  Therefore, we must consider the alternative definition: 

$v_\pi (s) \doteq \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} \sum_{t=0}^h \gamma^t \left ( \mathbb{E_\pi} [R_{t+1}|S_0=s]-r(\pi)  \right )$ 

This expression is identical to the original definition except for the presence of a discount rate in the sum which ensures the existence of a well defined value.  By taking the limit as $\gamma \rightarrow 1$, the impact of the discount rate can be eliminated after the calculation is done.

For state A, each parenthetical term in the sum will be: $1 - 0.5, 0 - 0.5, 1 - 0.5, 0 - 0.5, \dots = 0.5, -0.5, 0.5, -0.5, \dots$

For state B, each parenthetical term in the sum will be: $0 - 0.5, 1 - 0.5, 0 - 0.5, 1 - 0.5, \dots = -0.5, 0.5, -0.5, 0.5, \dots$

$\begin{flalign}
v_\pi (A) &= \lim_{\gamma \rightarrow 1} 0.5 - 0.5\gamma + 0.5 \gamma^2 - 0.5\gamma^3 + \cdots \\
&=0.5\lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty}\sum_{t=0}^h (-\gamma)^t \\
&=0.5\lim_{\gamma \rightarrow 1}\frac{1}{\gamma +1 } = 0.25
\end{flalign}$

$\begin{flalign}
v_\pi (B) &= \lim_{\gamma \rightarrow 1} -0.5 + 0.5\gamma - 0.5 \gamma^2 + 0.5\gamma^3 + \cdots \\
&=-0.5\lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty}\sum_{t=0}^h (-\gamma)^t \\
&=-0.5\lim_{\gamma \rightarrow 1}\frac{1}{\gamma +1 } = -0.25
\end{flalign}$
"""

# ╔═╡ f56743d1-d681-4bcf-b1bb-c4ef78a75180
md"""
> ### *Exercise 10.7* 
> Consider a Markov reward process consisting of a ring of three states A, B, and C, with state transitions going deterministically around the ring.  A reward of +1 is received upon arrival in A and otherwise the reward is 0.  What are the differential values of the three states, using (10.13)

From 10.13 we have 

$v_\pi (s) \doteq \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} \sum_{t=0}^h \gamma^t \left ( \mathbb{E_\pi} [R_{t+1}|S_0=s]-r(\pi)  \right )$

The average reward per step is $\frac{1}{3}$ so we can apply the same method used in exercise 10.6 where the elements inside the parentheses of the sum are: $\frac{2}{3}$ for $C \rightarrow A$ and $-\frac{1}{3}$ for the other two.  Starting in state A we transition twice and then on the third arrive in state A leading to the following mean corrected values of $-\frac{1}{3}$, $-\frac{1}{3}$, and $\frac{2}{3}$.  The other states will have these values cyclically permuted leading to the following infinite sums:

For state A:
$v_A = -\frac{1}{3} - \frac{1}{3}\gamma + \frac{2}{3}\gamma^2 - \frac{1}{3}\gamma^3 - \frac{1}{3}\gamma^4 + \frac{2}{3}\gamma^5 + \cdots = -\frac{1}{3} \times \left [ 1 + \gamma - 2 \gamma^2 + \gamma^3 + \gamma^4 - 2 \gamma^5 + \cdots \right ]$

For state B:
$v_B = -\frac{1}{3} + \frac{2}{3}\gamma - \frac{1}{3}\gamma^2 - \frac{1}{3}\gamma^3 + \frac{2}{3}\gamma^4 - \frac{1}{3}\gamma^5 + \cdots = -\frac{1}{3} \times \left [ 1 - 2\gamma + \gamma^2 + \gamma^3 - 2\gamma^4 + \gamma^5 + \cdots \right ]$

For state C:
$v_C = \frac{2}{3} - \frac{1}{3}\gamma - \frac{1}{3} \gamma^2 + \frac{2}{3}\gamma^3 - \frac{1}{3}\gamma^4 -\frac{1}{3}\gamma^5 + \cdots = -\frac{1}{3} \times \left [ -2 + \gamma + \gamma^2 - 2\gamma^3 + \gamma^4 + \gamma^5 \cdots \right ]$

"""

# ╔═╡ b242d3b2-396c-4cb6-8c9c-38d16dc18636
md"""
Comparing the above expressions we have:

$\begin{flalign}
v_A + v_B + v_C &= 0 \tag{1} \\
\gamma v_A &= v_C - \frac{2}{3} \tag{2}\\
\gamma v_B &= v_A + \frac{1}{3} \tag{3}\\
\end{flalign}$

Keep in mind we only care about the case where $\gamma = 1$, so we can enforce that now and see if the values exist.  Replacing terms to isolate $v_A$ yields: 

$\begin{flalign}
v_A &= -v_A - v_B - \frac{2}{3} \tag{by (1) and (2)}\\
2v_A &= - v_B - \frac{2}{3} \\
2v_A &= -v_A - \frac{1}{3} - \frac{2}{3} \tag{by (3)} \\
3v_A &= -1 \\
v_A &= -\frac{1}{3} \\
\end{flalign}$
"""

# ╔═╡ a28f57c1-e48c-4f4f-8795-bdd195b26135
md"""
With the value for state A known, the others will follow from (1), (2), and (3).

$\begin{flalign}
v_A &= -\frac{1}{3} \\
v_B &= v_A + \frac{1}{3} = -\frac{1}{3} + \frac{1}{3} = 0 \\
v_C &= -v_A - v_B = \frac{1}{3} \\
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
"""
    AccessControlState

State representation for the access control queueing system.

Encapsulates the current system state including server availability and
the priority level of the current incoming request.

# Fields
- `num_free_servers::Int64`: Number of currently unoccupied servers
- `top_priority::Float32`: Priority payment value of the current incoming request

# See Also
[`access_control_step`](@ref), [`create_access_control_task`](@ref)
"""
struct AccessControlState
	num_free_servers::Int64
	top_priority::Float32
end;

# ╔═╡ 014339eb-5b23-4ac5-a551-8eeb2238366f
begin
	"""
	    access_control_step(s, action, num_servers, priority_payments) -> Tuple{Float32, AccessControlState}
	
	Execute one step of the access control queueing system dynamics.
	
	Simulates server dynamics where occupied servers become free with probability 0.06 per step,
	and new requests arrive with random priority. Action determines whether to accept or reject
	the current request.
	
	# Arguments
	- `s::`[`AccessControlState`](@ref): Current system state
	- `action::`[`AccessControlAction`](@ref): Action to take ([`Accept`](@ref) or [`Reject`](@ref))
	- `num_servers::Integer`: Total number of servers in the system
	- `priority_payments::Vector{Float32}`: Possible priority payment values for requests
	
	# Returns
	- `Tuple{Float32, AccessControlState}`: (reward, next_state) where reward equals priority value if accepted, 0.0 if rejected
	
	# See Also
	[`AccessControlState`](@ref), [`Accept`](@ref), [`Reject`](@ref), [`create_access_control_task`](@ref)
	
	# Algorithm Details
	1. Simulates server liberation: each occupied server becomes free with probability 0.06
	2. For [`Reject`](@ref): returns 0.0 reward and updates state with new random priority
	3. For [`Accept`](@ref): 
	   - If no free servers, behaves as reject
	   - Otherwise, allocates one server, returns priority as reward
	4. Generates new incoming request with random priority from priority_payments
	"""
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

# ╔═╡ 78251726-a5ec-4fa3-adcb-09ff347ff54a
md"""
Approximation is not really needed here since we have a small number of states, but we can always use state aggregation where each state is assigned a unique group.  Here I use a matrix style indexing for the states where the number of free servers and the priority index are used and then converted into a linear index.
"""

# ╔═╡ 62839b2a-398a-4445-87d1-b15ff2acc1d1
"""
    create_access_control_task(num_servers, priority_payments) -> NamedTuple

Create access control queueing system MDP with state aggregation features.

Constructs a continuing task MDP modeling a server system that accepts/rejects requests
based on priority and capacity. Includes pre-configured state aggregation for linear function
approximation based on server count and priority groupings.

# Arguments
- `num_servers::Integer`: Total number of servers in the system
- `priority_payments::Vector{Float32}`: Possible priority payment values

# Returns
- `NamedTuple` with fields:
  - `mdp::`[`StateMDP`](@ref): The access control MDP with continuing task structure
  - `setup`: State aggregation setup from [`state_aggregation_feature_setup`](@ref)
# Example
```julia-repl
julia> 	# setup task
	(mdp, setup) = create_access_control_task(10, [1f0, 2f0, 4f0, 8f0]);
julia> 	# run learning algorithm
	output = semi_gradient_differential_sarsa_linear(mdp, 1, 100, setup...)
julia> 	# test value function on example initial state
	output.value_function(mdp.initialize_state());
```

# See Also
[`access_control_step`](@ref), [`AccessControlState`](@ref), [`state_aggregation_feature_setup`](@ref), [`StateMDPTransitionSampler`](@ref)

# Algorithm Details
1. Creates action space with [`Accept`](@ref) and [`Reject`](@ref) actions
2. Defines state initialization function for random priority assignment
3. Sets up transition sampler using [`access_control_step`](@ref) dynamics
4. Creates continuing task MDP (never terminates: `isterm(s) -> false`)
5. Configures state aggregation grouping states by:
   - Server availability (num_free_servers)
   - Priority level
6. Returns both MDP and feature setup for immediate use
"""
function create_access_control_task(num_servers::Integer, priority_payments::Vector{Float32})
	actions = [Accept(), Reject()]

	initialize_state() = AccessControlState(num_servers, rand(priority_payments))

	transition = StateMDPTransitionSampler((s, i_a) -> access_control_step(s, actions[i_a], num_servers, priority_payments), initialize_state())
	mdp = StateMDP(actions, transition, initialize_state, s -> false)
	states =  [AccessControlState(n, p) for n in 0:num_servers for p in priority_payments]
	assign_group(s::AccessControlState) = s.num_free_servers + 1 + (num_servers+1)*Int64(log2(s.top_priority))
	num_groups = (num_servers+1) * length(priority_payments)
	(mdp = mdp, setup = state_aggregation_feature_setup(initialize_state(), num_groups, assign_group))
end;

# ╔═╡ b4af8d87-a6e5-4e09-92b4-b07757f58f7f
# ╠═╡ skip_as_script = true
#=╠═╡
function run_access_control_differential_sarsa(max_steps::Int64; num_servers = 10, priority_payments = [1f0, 2f0, 4f0, 8f0], kwargs...)
	(mdp, setup) = create_access_control_task(num_servers, priority_payments)
	
	output = semi_gradient_differential_sarsa_linear(mdp, max_steps, setup...; kwargs...)
	
	v̂(num_free_servers::Int64, priority::Real) = output.value_function(AccessControlState(num_free_servers, Float32(priority)))

	(value_function = v̂, mdp = mdp, parameters = output.final_parameters, steprewards = output.average_reward_history)
end
  ╠═╡ =#

# ╔═╡ c914fc12-d650-400b-8aff-e2a55bb2d5cf
function sample_vector(v::Vector; npoints = min(length(v), 1000))
	l = length(v)
	inds = ceil.(Int64, LinRange(1, l, npoints))
	(inds, v[inds])
end

# ╔═╡ 546a775e-d3c9-4693-9f64-d4c47a84fb9f
# ╠═╡ skip_as_script = true
#=╠═╡
function figure_10_5(;numsteps = 2_000_000, α = 0.004f0, β = 0.001f0, ϵ = 0.1f0)
	access_control_output = run_access_control_differential_sarsa(numsteps; β = β, α = α, ϵ = ϵ)
	policy_output = BitArray(undef, (4, 10))
	priorities = [8, 4, 2, 1]
	actions = [true, false]
	value_function_outputs = [zeros(Float32, 11) for _ in 1:4]
	for num_free_servers in 0:10
		for priority in 1:4
			action_values, i_a, v = access_control_output.value_function(num_free_servers, priorities[priority])
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
	(rinds, vinds) = sample_vector(access_control_output.steprewards)
	p3 = plot(scatter(x = rinds, y = vinds), Layout(xaxis_title = "Step", yaxis_title = "Average Reward Estimate"))
	
	md"""
	Figure 10.5

	The policy and value function found by differential semi-gradient one-step Sarsa on the access-control queuing task after 2 million steps.  The value learned for $\bar R$ was about $(access_control_output.steprewards[end-10000:end] |> mean |> Float64 |> x -> round(x, sigdigits = 3))
	
	$([p1 p2])
	$p3
	"""
end
  ╠═╡ =#

# ╔═╡ 41c626c7-908d-4ff6-9730-4ad0b8c3cc25
#=╠═╡
figure_10_5()
  ╠═╡ =#

# ╔═╡ 708164fd-93ea-4720-ad6d-22e1c297c22a
md"""
#### Distributional Transition for Access Control Task

Even though the problem is stochastic, we can calculate all of the probabilities for a step transition.  The main challenge is using the 6% probability that an occupied server is freed on each step to compute the probabilities of transition states.
"""

# ╔═╡ c4ba34b5-e657-4b75-b853-0a2df081e34b
begin
	"""
	    access_control_step_distribution(s, action, num_servers, priority_payments) -> Tuple{Vector{Float32}, Vector{AccessControlState}, Vector{Float32}}
	
	Execute one step of the access control queueing system dynamics with full transition distribution.
	
	Returns the complete probability distribution over next states instead of sampling a single
	transition. Uses binomial distribution for server liberation events and uniform distribution
	over incoming request priorities.
	
	# Arguments
	- `s::`[`AccessControlState`](@ref): Current system state
	- `action::`[`AccessControlAction`](@ref): Action to take ([`Accept`](@ref) or [`Reject`](@ref))
	- `num_servers::Integer`: Total number of servers in the system
	- `priority_payments::Vector{Float32}`: Possible priority payment values for requests
	
	# Returns
	- `Tuple{Vector{Float32}, Vector{AccessControlState}, Vector{Float32}}`: (rewards, next_states, probabilities) where each vector has the same length and probabilities sum to 1.0
	
	# See Also
	[`access_control_step`](@ref), [`AccessControlState`](@ref), [`Accept`](@ref), [`Reject`](@ref), [`create_access_control_task_distribution`](@ref)
	
	# Algorithm Details
	1. **Server Liberation**: Computes binomial probabilities for all possible numbers of freed servers
	   - Each occupied server becomes free independently with probability 0.06
	   - Uses combinatorial formula: C(n,k) × (0.06)^k × (0.94)^(n-k)
	2. **Priority Assignment**: Uniform distribution over priority_payments for new requests
	3. **Action Processing**:
	   - [`Reject`](@ref): Returns 0.0 reward for all transitions, updates state with new priority
	   - [`Accept`](@ref): 
	     - If no free servers available, behaves identically to reject
	     - Otherwise, returns current priority as reward, allocates one server
	4. **Distribution Construction**: Creates parallel vectors of (reward, state, probability) tuples
	"""
	function access_control_step_distribution(s::AccessControlState, ::Reject, num_servers::Integer, priority_payments::Vector{Float32})
		occupied_servers = num_servers - s.num_free_servers
		rewards = Vector{Float32}()
		states = Vector{AccessControlState}()
		probabilities = Vector{Float32}()
		f = factorial(occupied_servers)
		for freed_servers in 0:occupied_servers
			new_occupied_servers = occupied_servers - freed_servers
			c = f / (factorial(new_occupied_servers) * factorial(freed_servers))
			pfree = c*(0.06f0^freed_servers) * (0.94f0^new_occupied_servers)
			for payment in priority_payments
				p = pfree/length(priority_payments)
				s′ = AccessControlState(num_servers - new_occupied_servers, payment)
				push!(rewards, 0f0)
				push!(states, s′)
				push!(probabilities, p)
			end
		end
		return (rewards, states, probabilities)
	end

	function access_control_step_distribution(s::AccessControlState, ::Accept, num_servers::Integer, priority_payments::Vector{Float32})
		(rewards, states, probabilities) = access_control_step_distribution(s, Reject(), num_servers, priority_payments)
		s.num_free_servers == 0 && return (rewards, states, probabilities)
		occupied_servers = num_servers - s.num_free_servers
		
		([s.top_priority for _ in rewards], [AccessControlState(s′.num_free_servers - 1, s′.top_priority) for s′ in states], probabilities)
	end
end

# ╔═╡ f9ad39d4-d2b6-44f3-a444-bcabd926a743
#=╠═╡
md"""
##### Access Control Step Transition Probabilities
Number of Free Servers: $(@bind num_free_servers Slider(0:10; show_value=true))
"""
  ╠═╡ =#

# ╔═╡ 84942647-8826-4864-b7d4-c31f9d78fd48
#=╠═╡
function plot_access_control_transition(free_servers::Int64)
	s = AccessControlState(free_servers, 1f0)
	(rewards, states, probabilities) = access_control_step_distribution(s, Accept(), 10, [1f0])
	μaccept = zeros(Float32, 11)
	for i in eachindex(states)
		s′ = states[i]
		μaccept[s′.num_free_servers + 1] += probabilities[i]
	end
	μaccept = bar(x = 0:10, y = μaccept, name = "accept")

	(rewards, states, probabilities) = access_control_step_distribution(s, Reject(), 10, [1f0])
	μreject = zeros(Float32, 11)
	for i in eachindex(states)
		s′ = states[i]
		μreject[s′.num_free_servers + 1] += probabilities[i]
	end
	μreject = bar(x = 0:10, y = μreject, name = "reject")
	plot([μaccept, μreject], Layout(xaxis_title = "Number of Free Servers", yaxis_title = "Probability"))
end
  ╠═╡ =#

# ╔═╡ dbde6c7c-a0ff-41bc-9a26-ffd38561a5ef
#=╠═╡
plot_access_control_transition(num_free_servers)
  ╠═╡ =#

# ╔═╡ 54700e88-2c70-4b3e-bc93-6960dc70efcb
"""
    create_access_control_tabular_task(num_servers, priority_payments) -> TabularMDP

Create access control queueing system as tabular MDP with precomputed transition matrices.

Constructs a tabular representation of the access control system where all state transitions
and reward distributions are precomputed and stored in sparse matrices. Enables exact
dynamic programming algorithms and analytical solutions by leveraging the finite state space.

# Arguments
- `num_servers::Integer`: Total number of servers in the system  
- `priority_payments::Vector{Float32}`: Possible priority payment values for requests

# Returns
- `TabularMDP{Float32, AccessControlState, AccessControlAction, TabularTransitionDistribution{Float32, 2}, Function}`

# See Also
[`access_control_step_distribution`](@ref), [`AccessControlState`](@ref), [`TabularMDP`](@ref), [`TabularTransitionDistribution`](@ref), [`create_access_control_task`](@ref)


# Examples
```julia-repl
julia> # Small system for demonstration
julia> mdp = create_access_control_tabular_task(3, [1f0, 2f0, 4f0]);
julia> length(mdp.states)
12
julia> size(mdp.ptf.state_transition_map)
(2, 12)
julia> # Verify probability distributions sum to 1
julia> sum(mdp.ptf.state_transition_map[1, 1]) ≈ 1.0f0
true
```

# Tabular vs Sampling Comparison
- **Tabular advantages**: Exact transitions, supports value/policy iteration, complete distributions
- **Sampling advantages**: Memory efficient for large state spaces, online learning capability
- **Use tabular for**: Small systems (≤1000 states), analytical study, exact DP algorithms
- **Use sampling for**: Large systems, function approximation, model-free online environments
"""
function create_access_control_tabular_task(num_servers::Integer, priority_payments::Vector{Float32})
	actions = [Accept(), Reject()]
	states =  [AccessControlState(n, p) for n in 0:num_servers for p in priority_payments]
	stateindex = makelookup(states)
	state_transition_map = Matrix{SparseVector{Float32, Int64}}(undef, length(actions), length(states))
	reward_transition_map = Matrix{Vector{Float32}}(undef, length(actions), length(states))
	for i_s in eachindex(states)
		for i_a in eachindex(actions)
			(rewards, states′, probabilities) = access_control_step_distribution(states[i_s], actions[i_a], num_servers, priority_payments)
			v = sparse(zeros(Float32, length(states)))
			for i in eachindex(states′)
				s′ = states′[i]
				i_s′ = stateindex[s′]
				v[i_s′] = probabilities[i]
			end
			state_transition_map[i_a, i_s] = copy(v)
			reward_transition_map[i_a, i_s] = copy(rewards)
		end
	end
	
	initialize_state_index() = stateindex[AccessControlState(num_servers, rand(priority_payments))]
	ptf = TabularTransitionDistribution(state_transition_map, reward_transition_map)
	TabularMDP(states, actions, ptf, initialize_state_index)
end;

# ╔═╡ 28f9d40a-4f4f-4bbf-ac36-4964afed7ab4
# ╠═╡ skip_as_script = true
#=╠═╡
const tabular_access_control_task = create_access_control_tabular_task(10, [1f0, 2f0, 4f0, 8f0])
  ╠═╡ =#

# ╔═╡ 32b3c5b4-cdb8-43be-a398-6e158254c4a7
#=╠═╡
md"""
##### Access Control Task with Discounting

Select Discount Rate: $(@bind γ_10_5 NumberField(0f0:0.01f0:1f0; default = 0.9))
"""
  ╠═╡ =#

# ╔═╡ 3985641e-2f07-4029-8047-51579904cd53
md"""
As the discount rate approaches 1, this solution should converge to the average reward solution shown below.  Numerically this is challenging because the values themselves diverge so there might not be enough precision to arrive at an answer.  If the priority payments are considered money, this solution is more desireable if the value of γ actually reflects some real time value of the money like an interest rate.
"""

# ╔═╡ 350e057e-154f-4d0b-91fb-ffde9cc9059f
#=╠═╡
function figure_10_5_tabular_discounted(γ::Float32)
	access_control_output = value_iteration_v(tabular_access_control_task, γ; show_message = false)
	policy_output = zeros(Float32, 4, 10)
	priorities = [8, 4, 2, 1]
	actions = [true, false]
	value_function_outputs = [zeros(Float32, 11) for _ in 1:4]
	for num_free_servers in 0:10
		for priority in 1:4
			s = AccessControlState(num_free_servers, Float32(priorities[priority]))
			i_s = tabular_access_control_task.state_index[s]
			v = access_control_output.final_value[i_s]
			i_a = 1f0 - access_control_output.optimal_policy[2, i_s]
			value_function_outputs[priority][num_free_servers+1] = v
			if num_free_servers > 0
				policy_output[priority, num_free_servers] = i_a
			end
		end
	end
	policy_trace = heatmap(x = 1:10, y = 1:4, z = Float32.(policy_output), colorscale="Greys", showscale = false)
	value_traces = [scatter(x = 0:10, y = value_function_outputs[i], name = "priority $(priorities[i])") for i in 1:4]
	p1 = plot(policy_trace, Layout(yaxis_tickvals = 1:4, yaxis_ticktext = priorities, xaxis_ticktext = 1:10, xaxis_tickvals = 1:10, xaxis_title = "Number of free servers", yaxis_title = "Priority", title = "Policy (black=reject, white=accept)"))
	p2 = plot(value_traces, Layout(xaxis_title = "Number of free servers", yaxis_title = "Discounted value of best action", title = "Value Function"))

	md"""
	Figure 10.5 Discounted Tabular

	The policy and value function found by value iteration on the access-control queuing task.
	$([p1 p2])
	"""
end
  ╠═╡ =#

# ╔═╡ 5a73ef20-dfdb-4d75-8790-805d6da27462
#=╠═╡
figure_10_5_tabular_discounted(γ_10_5)
  ╠═╡ =#

# ╔═╡ f009970f-bf6c-46dd-a534-a960582ce51b
md"""
##### Average Reward Solution to Access Control Task Solved with Policy Iteration
"""

# ╔═╡ f1f6f750-8c49-4435-82a1-e13a280b3738
#=╠═╡
function figure_10_5_tabular_average_reward()
	access_control_output = TabularRL.differential_policy_iteration_v(tabular_access_control_task; θ = 0.00001f0)
	@info "final average reward per step was $(access_control_output.average_reward[end])"
	policy_output = zeros(Float32, 4, 10)
	μs = zeros(Float32, 4, 11)
	priorities = [8, 4, 2, 1]
	actions = [true, false]
	value_function_outputs = [zeros(Float32, 11) for _ in 1:4]
	(minv, maxv) = extrema(access_control_output.value_history[end])
	meanv = (minv + maxv)/2
	for num_free_servers in 0:10
		for priority in 1:4
			s = AccessControlState(num_free_servers, Float32(priorities[priority]))
			i_s = tabular_access_control_task.state_index[s]
			v = access_control_output.value_history[end][i_s] .- meanv
			i_a = 1f0 - access_control_output.policy_history[end][2, i_s]
			value_function_outputs[priority][num_free_servers+1] = v
			μs[priority, num_free_servers+1] = access_control_output.steady_state_distribution.steady_state_distribution[i_s]
			if num_free_servers > 0
				policy_output[priority, num_free_servers] = i_a
			end
		end
	end
	policy_trace = heatmap(x = 1:10, y = 1:4, z = Float32.(policy_output), name = "optimal policy action", colorscale="Greys", showscale = false)
	value_traces = [scatter(x = 0:10, y = value_function_outputs[i], name = "priority $(priorities[i])") for i in 1:4]
	μs = heatmap(x = 1:10, y = 1:4, z = μs, colorscale="Greys", name = "steady state probability", showscale = false)
	p1 = plot(policy_trace, Layout(yaxis_tickvals = 1:4, yaxis_ticktext = priorities, xaxis_ticktext = 1:10, xaxis_tickvals = 1:10, xaxis_title = "Number of free servers", yaxis_title = "Priority", title = "Policy (black=reject, white=accept)"))
	p2 = plot(value_traces, Layout(xaxis_title = "Number of free servers", yaxis_title = "Differential value of best action", title = "Value Function"))
	p3 = plot(μs, Layout(yaxis_tickvals = 1:4, yaxis_ticktext = priorities, xaxis_ticktext = 1:10, xaxis_tickvals = 1:11, xaxis_title = "Number of free servers", yaxis_title = "Priority", title = "Steady State Distribution"))
	p4 = plot(bar(y = access_control_output.average_reward, showlegend = false), Layout(xaxis_title = "Policy Iteration", yaxis_title = "Average Reward"))

	@htl("""
	$(md"""
	Figure 10.5 Tabular Policy Iteration for Average Reward

	The policy and value function found by policy iteration on the access-control queuing task.
	""")

	<div style = "display: flex; flex-wrap: wrap;">
	<div style = "width: 50%">$p1</div>
	<div style = "width: 50%">$p2</div>
	<div style = "width: 50%">$p3</div>
	<div style = "width: 50%">$p4</div>
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ 64096af2-3cbe-4f6b-944d-6f4bdc2cd535
#=╠═╡
figure_10_5_tabular_average_reward()
  ╠═╡ =#

# ╔═╡ 662759be-282c-460b-adc3-8595475b53c2
md"""
## 10.4 Deprecating the Discounted Setting

In a special case of indistinguishable states, we can only use the actions and reward sequences to analyze a continuing task.  For a policy $\pi$, the average of the discounted returns with discount factor $\gamma$ is always $\frac{r(\pi)}{1-\gamma}$.  Therefore the *ordering* of all policies is independent of the discount rate and would match the ordering we get in the average reward setting.  This derivation however depends on states being indistinguishable allowing us to match up the weights on reward sequences from different policies.

We can use discounting in approximate solution methods regardless but then $\gamma$ changes from a problem parameter to a solution method parameter.  Unfortunately, discounting algorithms with function approximation do not optimize discounted value over the on-policy distribution, and thus are not guaranteed to optimze average reward.

The root cause of the problem applying discounting with function approximation is that we have lost the policy improvement theorem which states that a policy $\pi^\prime$ is better than policy $\pi$ if $v_{\pi^\prime}(s) \geq v_\pi(s) \forall s\in \mathcal{S}$.  Under this theorem we could take a deterministic policy, choose a specific state, and find a new action at that state with a higher expected reward than the current policy.  If the policy is an approximation function that uses states represented by feature vectors, then adjusting the parameters can in general affect the actions at many states including ones that have not been encountered yet.  In fact, with approximate solution methods we cannot guarantee  policy improvement in any setting.  Later we will introduce a theoretical guarantee called the "policy-gradient theorem" but for an alternative class of algorithms based on parametrized policies.
"""

# ╔═╡ b727e6e7-e019-4697-85a6-c2cf839ef34a
md"""
The objective that uses discounting here is $J(\pi) = \sum_s \mu_\pi (s) v_\pi^\gamma(s)$ rather than the original goal of getting the maximum state values for every state.  So this derivation is really only suited to the approximation setting and the altered objective.  If we are in the tabular objective, then we have access to the policy improvmeent theorem and can guarantee a maximum value for every state, although none of those values may correspond to the average reward.  We can see the example below that favoring this objective leads to a worse value function and policy which is not optimal given that discount rate.  The only reason we consider it here is because in the case of approximation we are forced into using this objective.  If we use an approximation method instead which does separate each state, then updates to one state will not affect the others and the optimal policy will be found anyway.  The sampling according to the steady state distribution only affects the optimization result when multiple states are affected by a single update.
"""

# ╔═╡ 68e3ef82-0706-449c-b00a-dc69e6c7b717
md"""
In the discounted setting, we can think about optimizing $D(\pi) = \sum_s v_\pi(s)$ where $v_*$ has the highest possible value of $D$ since we know the state values are highest everywhere.  Since all states are weighted equally, the result is different from optimizing $J$ and in general will depend on the discount rate.  But what if $\gamma \rightarrow 1$, can we say anything about $D(\pi)$?

Returning to the definition of $v_\pi(s) = \mathbb{E}_\pi \left [ \sum_{k=1}^\infty R_k \gamma^{k-1} \mid S_0 = s \right ]$

Let's say that $\lim_{t \rightarrow \infty} \mathbb{E}_\pi [R_t]$ exists and is independent of $S_0$ (those are the criteria for the average reward setting).  Then we can split this into a finite sum and an infinite sum where 

$v_\pi(s) \approx \sum_{k=1}^N \mathbb{E}_\pi[ R_k \gamma^{k-1} \mid S_0 = s] + \sum_{k = N+1}^\infty r(\pi) \gamma^{k-1}$

and that in the limit as $N \rightarrow \infty$ this becomes exact.  Now there is some $r_{max} \geq R_t \forall t < N$, so the first part of the sum can be bounded by $\sum_{k = 1}^N r_{max} \gamma^{k-1} = r_{max}\frac{\gamma^N - 1}{\gamma - 1}$ and in the limit of $\gamma \rightarrow 1$ this approaches $r_{max}N$.

$S1 \le r_{max}\frac{1 - \gamma^N}{1 - \gamma}$

$S2 = r(\pi) \frac{\gamma ^N}{1 - \gamma}$

$\frac{S2}{S1} \ge \frac{r(\pi) \gamma^N}{r_{max} (1 - \gamma^N)}$

Let $\gamma = 1 - \epsilon$ where $\epsilon \ll 1$  Then $\gamma^N = (1 - \epsilon)^N = 1 - N \epsilon + O(\epsilon^2)$ so we can rewrite the expression above as

$\frac{S2}{S1} \ge \frac{r(\pi)(1 - N\epsilon)}{r_{max} (1 - 1 + N\epsilon)} = \frac{r(\pi)}{r_\max}\frac{(1 - N\epsilon)}{N\epsilon} = \frac{r(\pi)}{r_\max}\left ( \frac{1}{N \epsilon} - 1 \right )$

So let's say that I demand that $\frac{S_2}{S_1} \gt D \gg 1 \implies \frac{r(\pi)}{r_\max} \left (\frac{1}{N\epsilon} - 1 \right ) \gt D \implies \frac{1}{N\epsilon} \gt \frac{D r_\max + r(\pi)}{r(\pi)} \implies \epsilon \lt \frac{r(\pi)N}{Dr_\max + r(\pi)} = \frac{N}{D \frac{r_\max}{r_\pi} + 1} \lt \frac{N}{D + 1}$  So for any N, I can always select $D \gg N \implies \epsilon \ll 1 \implies \frac{S2}{S1} \gg 1 \implies v_\pi \approx S2 \approx r(\pi) \frac{1 - N\epsilon}{\epsilon}$

$\frac{v_\pi - \hat v_\pi}{v_\pi} = 1 - \frac{\hat v_\pi}{v_\pi}$

The second sum is equal to $\frac{r(\pi) \gamma^N}{1-\gamma}$ which approaches infinite value as the discount rate approaches 1.  So for any finite N, there exists a $\gamma \lt 1$ for which the second sum is arbitrarily larger than the first which means that in the limit $v_\pi(s) \approx \frac{r(\pi) \gamma^N}{1-\gamma}$ which is also approximately equal to $J(\pi)$ in the limit of $\gamma \rightarrow 1$.  Thus optimizing the discounted value is equivalent to optimizing $r(\pi)$ in this limit.
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

# ╔═╡ 0d3d5304-0412-485d-8f56-f4362a74ea45
function gradient_monte_carlo_episode_update!(parameters, action_values::Vector{T}, ∇q̂, feature_vector, update_feature_vector!::Function, update_action_values!::Function, update_value_gradient!::Function, states::AbstractVector{S}, actions::AbstractVector{I}, rewards::AbstractVector{T}, γ::T, α::T, calculate_error::Function) where {T<:Real, I<:Integer, S}
	g = zero(T)
	l = length(states)
	episode_error = zero(T)
	for i in l:-1:1
		s = states[i]
		i_a = actions[i]
		update_feature_vector!(feature_vector, s)
		update_value_gradient!(∇q̂, action_values, feature_vector, i_a, parameters)
		g = γ * g + rewards[i]
		q̂ = action_values[i_a]
		δ = g - q̂
		c = α*δ
		update_params_with_gradient!(parameters, c, ∇q̂)
		episode_error += calculate_error(g, q̂, s)
	end
	return episode_error / l
end

# ╔═╡ 604a2621-aa73-42d9-9255-e5f5578d0b51
"""
    gradient_monte_carlo_action_policy_estimation!(parameters, mdp, π, γ, num_episodes,
                                                  feature_vector, update_feature_vector!,
                                                  update_action_values!, ∇q̂, update_value_gradient!;
                                                  α=0.1, action_values, calculate_error, epkwargs...) -> NamedTuple

Estimate action-value function for a given policy using gradient Monte Carlo method.

Performs policy evaluation by running episodes under policy π and updating action-value
function parameters via gradient descent on Monte Carlo returns. Uses function approximation
to represent the action-value function q̂(s,a).

# Arguments
- `parameters::Vector{T}`: Action-value function parameters (modified in-place)
- `mdp::`[`StateMDP`](@ref): Markov Decision Process environment
- `π::Function`: Policy function mapping states to action indices
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `num_episodes::Integer`: Number of episodes to run
- `feature_vector`: Pre-allocated storage for state feature representations
- `update_feature_vector!::Function`: Extract features: `(feature_vector, state) -> nothing`
- `update_action_values!::Function`: Compute action values: `(action_values, features, params) -> nothing`
- `∇q̂`: Pre-allocated storage for action-value function gradients
- `update_value_gradient!::Function`: Compute gradient: `(∇q̂, features, action_index, params) -> nothing`

# Keyword Arguments
- `α::Real = 0.1`: Learning rate (step size parameter)
- `action_values::Vector{T} = zeros(T, length(mdp.actions))`: Pre-allocated action value storage
- `calculate_error::Function = (g, q̂, s) -> (g - q̂)^2`: Error function for convergence tracking
- `epkwargs...`: Additional arguments passed to episode generation

# Returns
- `NamedTuple` with fields:
- `value_function`: Function q̂(s) that returns NamedTuple with fields (action_values, maximizing_action, maximizing_value)
  - `error_history::Vector{T}`: Episode-wise error progression
  - `parameters::Vector{T}`: Final function approximation parameters

# See Also
[`gradient_monte_carlo_episode_update!`](@ref), [`runepisode`](@ref), [`runepisode!`](@ref), [`form_value_function`](@ref), [`StateMDP`](@ref)

# Algorithm Details
1. **Initial Episode**: Generates first episode using [`runepisode`](@ref) with fresh storage
2. **Parameter Updates**: Calls [`gradient_monte_carlo_episode_update!`](@ref) for backward pass updates
3. **Episode Reuse**: Subsequent episodes use [`runepisode!`](@ref) to reuse storage arrays
4. **Error Tracking**: Records root mean square error for first episode, then average error
5. **Value Function Construction**: Uses [`form_value_function`](@ref) to create final evaluator
"""
function gradient_monte_carlo_action_policy_estimation!(parameters, mdp::StateMDP, π::Function, γ::T, num_episodes::Integer, feature_vector, update_feature_vector!::Function, update_action_values!::Function, ∇q̂, update_value_gradient!::Function; α = one(T)/10, action_values = zeros(T, length(mdp.actions)), calculate_error::Function = (g, v̂, s) -> (g - v̂) ^2, epkwargs...) where {T<:Real}
	(states, actions, rewards, sterm) = runepisode(mdp; π = π, epkwargs...)
	sqerr = gradient_monte_carlo_action_episode_update!(parameters, action_values, ∇q̂, feature_vector, update_feature_vector!, update_action_values!, update_value_gradient!, states, actions, rewards, γ, α, calculate_error)
	error_history = zeros(T, num_episodes)
	error_history[1] = sqrt(sqerr)
	for ep in 2:num_episodes
		(states, actions, rewards, sterm, n_steps) = runepisode!((states, actions, rewards), mdp; π = π, epkwargs...)
		error = gradient_monte_carlo_episode_update!(parameters, action_values, ∇q̂, feature_vector, update_feature_vector!, update_action_values!, update_value_gradient!, view(states, 1:n_steps), view(actions, 1:n_steps), view(rewards, 1:n_steps), γ, α, calculate_error)
		error_history[ep] = error
	end
	q̂ = form_value_function(mdp, update_feature_vector!, update_action_values!, feature_vector, parameters)
	return (value_function = q̂, error_history = error_history, parameters = parameters)
end;

# ╔═╡ 06834750-cc3a-468a-b0c2-81349c288a33
"""
    gradient_monte_carlo_control!(parameters, mdp, γ, num_episodes, feature_vector,
                                 update_feature_vector!, value_component, gradient_storage,
                                 update_value_gradient!; α=0.1, ϵ=0.1, suppress_warning=false,
                                 ignore_unfinished_episodes=false, action_values, calculate_error,
                                 epkwargs...) -> NamedTuple

Learn optimal policy using gradient Monte Carlo control with ε-greedy exploration.

Combines policy evaluation and improvement by alternating between value function updates 
(using Monte Carlo returns) and ε-greedy policy updates. Supports both direct action-value
function approximation and state-value function approximation with action-value computation.

## Action-Value Function Approximation Method
For direct q̂(s,a) approximation:

# Type Parameters
- `T <: Real`: Numeric type for rewards, parameters, and computations

# Arguments
- `parameters::Vector{T}`: Value function parameters (modified in-place)
- `mdp::`[`StateMDP`](@ref): Markov Decision Process environment
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `num_episodes::Integer`: Number of episodes to run
- `feature_vector`: Pre-allocated storage for state feature representations
- `update_feature_vector!::Function`: Extract features: `(feature_vector, state) -> nothing`
- `update_action_values!::Function`: Action-value computation: `(action_values, features, params) -> nothing`
- `∇q̂`: Pre-allocated storage for action-value function gradients
- `update_value_gradient!::Function`: Gradient computation: `(∇q̂, features, action_index, params) -> nothing`

## State-Value Function Approximation Method  
For v̂(s) approximation with action-value computation:

# Type Parameters
- `T <: Real`: Numeric type for rewards, parameters, and computations
- `S`: State type (scalar or N-dimensional tuple)
- `A`: Action type  
- `P <: StateMDPTransitionDistribution`: Transition probability distribution type
- `F1 <: Function`: State initialization function
- `F2 <: Function`: Transition function
- `F3 <: Function`: Termination function

# Arguments
- `parameters::Vector{T}`: Value function parameters (modified in-place)
- `mdp::StateMDP{T, S, A, P, F1, F2, F3}`: Markov Decision Process environment with matching numeric type
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `num_episodes::Integer`: Number of episodes to run
- `feature_vector`: Pre-allocated storage for state feature representations
- `update_feature_vector!::Function`: Extract features: `(feature_vector, state) -> nothing`
- `estimate_value::Function`: State-value estimation: `(features, params; kwargs...) -> Real`
- `∇v̂`: Pre-allocated storage for state-value function gradients
- `update_value_gradient!::Function`: Gradient computation: `(∇v̂, features, params) -> nothing`

# Keyword Arguments
- `α::Real = 0.1`: Learning rate for parameter updates
- `ϵ::Real = 0.1`: Exploration parameter for ε-greedy policy (probability of random action)
- `suppress_warning::Bool = false`: Whether to suppress episode termination warnings
- `ignore_unfinished_episodes::Bool = false`: Whether to update on incomplete episodes
- `action_values::Vector{T} = zeros(T, length(mdp.actions))`: Pre-allocated action value storage
- `calculate_error::Function = (g, v̂, s) -> (g - v̂)^2`: Error function for convergence tracking
- `epkwargs...`: Additional arguments passed to episode generation

# Returns
- `NamedTuple` with fields:
  - `value_function::Function`: Action-value function `(state; kwargs...) -> NamedTuple` created by [`form_value_function`](@ref), returns `(action_values, maximizing_action, maximizing_value)`
  - `step_history::Vector{Int64}`: Number of steps per successful episode
  - `reward_history::Vector{T}`: Total reward per successful episode
  - `error_history::Vector{T}`: Episode-wise approximation error
  - `parameters::Vector{T}`: Final function approximation parameters
  - `success_rate::Real`: Fraction of episodes that terminated successfully

# See Also
[`gradient_monte_carlo_episode_update!`](@ref), [`make_ϵ_greedy_policy!`](@ref), [`sample_action`](@ref), [`form_value_function`](@ref), [`update_action_values!`](@ref), [`StateMDP`](@ref)

# Algorithm Details
Monte Carlo control algorithm implementing the policy iteration framework:

1. **Episode Generation**: Generate episode under current ε-greedy policy
2. **Policy Evaluation**: Update value function parameters using Monte Carlo returns
3. **Policy Improvement**: Update ε-greedy policy based on improved value estimates
4. **Convergence Tracking**: Monitor episode success, rewards, and approximation error

The two methods differ in value function representation:
- **Action-value method**: Directly approximates q̂(s,a) and updates parameters for specific state-action pairs
- **State-value method**: Approximates v̂(s) and computes action values via [`update_action_values!`](@ref) using Bellman backup

Both methods use the same ε-greedy exploration strategy and Monte Carlo update mechanism.

# Performance Notes
- Reuses episode storage for memory efficiency across episodes
- Supports incomplete episode processing for continuing tasks
- Action-value method reduces computational overhead by avoiding Bellman backups
- State-value method provides more flexible value function approximation
"""
function gradient_monte_carlo_control!(parameters, mdp::StateMDP, γ::T, num_episodes::Integer, feature_vector, update_feature_vector!::Function, update_action_values!::Function, ∇q̂, update_value_gradient!::Function; α = one(T)/10, ϵ = one(T)/10, suppress_warning::Bool = false, use_unfinished_episodes::Bool = false, action_values::Vector{T} = zeros(T, length(mdp.actions)), calculate_error::Function = (g, v̂, s) -> (g - v̂) ^2, epkwargs...) where {T<:Real}

	step_history = Vector{Int64}()
	error_history = Vector{T}()
	reward_history = Vector{T}()
	num_success = 0
	
	function π_ϵ_greedy(s)
		update_feature_vector!(feature_vector, s)
		update_action_values!(action_values, feature_vector, parameters)
		make_ϵ_greedy_policy!(action_values; ϵ = ϵ)
		sample_action(action_values)
	end
	
	(states, actions, rewards, sterm, nsteps) = runepisode(mdp; epkwargs...)

	success = mdp.isterm(sterm)
	num_success += success
	if !suppress_warning && !success
		@info "Warning: Episode 1 did not conclude in $nsteps steps"
	end

	if mdp.isterm(sterm) || use_unfinished_episodes
		err = gradient_monte_carlo_episode_update!(parameters, action_values, ∇q̂, feature_vector, update_feature_vector!, update_action_values!, update_value_gradient!, states, actions, rewards, γ, α, calculate_error)
		push!(error_history, err)
		push!(step_history, nsteps)
		push!(reward_history, sum(rewards))
	end
	
	for ep in 2:num_episodes
		(states, actions, rewards, sterm, nsteps) = runepisode!((states, actions, rewards), mdp; π = π_ϵ_greedy, epkwargs...)

		success = mdp.isterm(sterm)
		num_success += success

		if !success && !suppress_warning
			@info "Warning: Episode $ep did not conclude in $nsteps steps"
		end

		if success || use_unfinished_episodes
			err = gradient_monte_carlo_episode_update!(parameters,action_values, ∇q̂, feature_vector, update_feature_vector!, update_action_values!, update_value_gradient!, view(states, 1:nsteps), view(actions, 1:nsteps), view(rewards, 1:nsteps), γ, α, calculate_error)
			push!(step_history, nsteps)
			push!(reward_history, sum(rewards[i] for i in 1:nsteps))
			push!(error_history, err)
		end
	end

	q̂, form_kwargs = form_value_function(mdp, update_feature_vector!, update_action_values!, feature_vector, parameters)

	success_rate = num_success / num_episodes
	
	return (value_function = q̂, step_history = step_history, reward_history = reward_history,  error_history = error_history, parameters = parameters, sucess_rate = success_rate, form_kwargs = form_kwargs)
end;

# ╔═╡ d04bf8ac-9905-4e80-93db-c5c28c31359b
function gradient_monte_carlo_control!(parameters, mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, num_episodes::Integer, feature_vector, update_feature_vector!::Function, estimate_value::Function, ∇v̂, update_value_gradient!::Function; α = one(T)/10, ϵ = one(T)/10, suppress_warning::Bool = false, use_unfinished_episodes::Bool = false, action_values::Array{T, N} = zeros(T, length(mdp.actions), 1), calculate_error::Function = (g, v̂, s) -> (g - v̂) ^2, epkwargs...) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1<:Function, F2<:Function, F3<:Function, N}

	step_history = Vector{Int64}()
	error_history = Vector{T}()
	reward_history = Vector{T}()
	num_success = 0

	action_value_args = form_action_value_args(mdp, feature_vector, parameters)

	function π_ϵ_greedy(s)
		update_action_values!(action_values, s, feature_vector, update_feature_vector!, estimate_value, parameters, mdp, γ, action_value_args...)
		make_ϵ_greedy_policy!(action_values; ϵ = ϵ)
		sample_action(action_values)
	end
	
	(states, actions, rewards, sterm, nsteps) = runepisode(mdp; epkwargs...)

	success = mdp.isterm(sterm)
	num_success += success
	if !suppress_warning && !success
		@info "Warning: Episode 1 did not conclude in $nsteps steps"
	end

	if mdp.isterm(sterm) || use_unfinished_episodes
		err = gradient_monte_carlo_episode_update!(parameters, ∇v̂, feature_vector, update_feature_vector!, estimate_value, update_value_gradient!, states, rewards, γ, α, calculate_error)
		push!(error_history, err)
		push!(step_history, nsteps)
		push!(reward_history, sum(rewards))
	end
	
	for ep in 2:num_episodes
		(states, actions, rewards, sterm, nsteps) = runepisode!((states, actions, rewards), mdp; π = π_ϵ_greedy, epkwargs...)

		success = mdp.isterm(sterm)
		num_success += success

		if !success && !suppress_warning
			@info "Warning: Episode $ep did not conclude in $nsteps steps"
		end

		if success || use_unfinished_episodes
			err = gradient_monte_carlo_episode_update!(parameters, ∇v̂, feature_vector, update_feature_vector!, estimate_value, update_value_gradient!, view(states, 1:nsteps), view(rewards, 1:nsteps), γ, α, calculate_error)
			push!(step_history, nsteps)
			push!(reward_history, sum(rewards[i] for i in 1:nsteps))
			push!(error_history, err)
		end
	end

	q̂, form_kwargs = form_value_function(mdp, γ, update_feature_vector!, estimate_value, feature_vector, parameters)

	success_rate = num_success / num_episodes
	
	return (value_function = q̂, step_history = step_history, reward_history = reward_history,  error_history = error_history, parameters = parameters, sucess_rate = success_rate, form_kwargs = form_kwargs)
end

# ╔═╡ 31260d29-6131-4e44-b6e6-e78399501c54
md"""
### Linear Approximation
"""

# ╔═╡ b4085947-f4c7-4664-8d94-8090a67ea6c4
#uses an action value function to learn optimal policy
"""
    gradient_monte_carlo_control_linear(mdp, γ, num_episodes, feature_vector,
                                       update_feature_vector!; init_value=0,
                                       parameters, kwargs...) -> NamedTuple

Learn optimal policy using gradient Monte Carlo control with linear function approximation.

Provides specialized implementations for linear value function approximation, automatically
selecting between action-value and state-value approaches based on MDP transition structure.
Uses [`LinearFeatureVector`](@ref) for efficient sparse feature representation.

## Action-Value Function Approximation Method
For direct q̂(s,a) linear approximation when transition distribution is not available:

# Type Parameters
- `T <: Real`: Numeric type for rewards, parameters, and computations

# Arguments
- `mdp::`[`StateMDP`](@ref): Markov Decision Process environment
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `num_episodes::Integer`: Number of episodes to run
- `feature_vector::`[`LinearFeatureVector`](@ref): Linear feature representation storage
- `update_feature_vector!::Function`: Extract features: `(feature_vector, state) -> nothing`

# Keyword Arguments
- `init_value::T = zero(T)`: Initial parameter values for action-value function
- `parameters::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, init_value)`: Action-value function parameters (|features| × |actions|)
- `kwargs...`: Additional arguments passed to [`gradient_monte_carlo_control!`](@ref)

## State-Value Function Approximation Method
For v̂(s) linear approximation when transition distribution is available:

# Type Parameters
- `T <: Real`: Numeric type for rewards, parameters, and computations
- `S`: State type (scalar or N-dimensional tuple)
- `A`: Action type
- `P <: StateMDPTransitionDistribution`: Transition probability distribution type
- `F1 <: Function`: State initialization function
- `F2 <: Function`: Transition function  
- `F3 <: Function`: Termination function

# Arguments
- `mdp::StateMDP{T, S, A, P, F1, F2, F3}`: Markov Decision Process with transition distribution
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `num_episodes::Integer`: Number of episodes to run
- `feature_vector::`[`LinearFeatureVector`](@ref): Linear feature representation storage
- `update_feature_vector!::Function`: Extract features: `(feature_vector, state) -> nothing`

# Keyword Arguments
- `init_value::T = zero(T)`: Initial parameter values for state-value function
- `parameters::Vector{T} = initialize_linear_parameters(feature_vector, init_value)`: State-value function parameters (length |features|)
- `kwargs...`: Additional arguments passed to [`gradient_monte_carlo_control!`](@ref)

# Returns
- `NamedTuple`: Same structure as [`gradient_monte_carlo_control!`](@ref) with fields:
  - `value_function`: Action-value function created by [`form_value_function`](@ref)
  - `step_history`: Episode step counts
  - `reward_history`: Episode reward totals
  - `error_history`: Approximation errors
  - `parameters`: Final function approximation parameters
  - `success_rate`: Episode success fraction

# See Also
[`gradient_monte_carlo_control!`](@ref), [`LinearFeatureVector`](@ref), [`initialize_linear_parameters`](@ref), [`linear_value_function`](@ref), [`update_linear_action_values!`](@ref), [`update_linear_value_gradient!`](@ref)

# Implementation
Convenience wrapper that automatically configures linear function approximation components:

1. **Method Selection**: Chooses action-value or state-value approach based on MDP transition type
2. **Parameter Initialization**: Creates appropriately sized parameter arrays using [`initialize_linear_parameters`](@ref)
3. **Component Setup**: Configures linear value functions and gradient computations
4. **Delegation**: Calls [`gradient_monte_carlo_control!`](@ref) with configured components

**Action-value method** uses [`update_linear_action_values!`](@ref) and [`LinearActionValueGradient`](@ref)
for direct q̂(s,a) approximation. **State-value method** uses [`linear_value_function`](@ref) and 
computes action values via transition distribution.

# Performance Notes
- Leverages sparse feature representations for memory efficiency
- Parameter matrices (action-value) vs vectors (state-value) sized automatically
- Reuses feature vector storage through `deepcopy` for gradient computations

# Examples
```julia-repl
julia> mountaincar_tile_setup = setup_mountain_car_tiles((1/10f0, 1/10f0), 12);
julia> output = gradient_monte_carlo_control_linear(mountain_car_mdp, 1f0, 1000, 
                                                   mountaincar_tile_setup.feature_vector, 
                                                   mountaincar_tile_setup.update_feature_vector!; 
                                                   α=1f-8, ϵ=0.1f0, max_steps=10_000, 
                                                   suppress_warning=true, 
                                                   ignore_unfinished_episodes=true);
julia> output.value_function(mountain_car_mdp.initialize_state())
(action_values = [0.045f0, 0.032f0, 0.051f0], maximizing_action = 3, maximizing_value = 0.051f0)
```
"""
gradient_monte_carlo_control_linear(mdp::StateMDP, γ::T, num_episodes::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), parameters::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, init_value), kwargs...) where T<:Real = gradient_monte_carlo_control!(parameters, mdp, γ, num_episodes, feature_vector, update_feature_vector!, update_linear_action_values!, LinearActionValueGradient(deepcopy(feature_vector), 0), update_linear_value_gradient!; kwargs...)

# ╔═╡ 164c68ef-01b8-43be-bc75-919dd99a6e03
#when the transition distribution is available uses the state value function to learn optimal policy
gradient_monte_carlo_control_linear(mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, num_episodes::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), parameters::Vector{T} = initialize_linear_parameters(feature_vector, init_value), kwargs...) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1<:Function, F2<:Function, F3<:Function} = gradient_monte_carlo_control!(parameters, mdp, γ, num_episodes, feature_vector, update_feature_vector!, linear_value_function, deepcopy(feature_vector), update_linear_value_gradient!; kwargs...)

# ╔═╡ cc285969-c33f-4d19-8e47-397b59e67299
# ╠═╡ skip_as_script = true
#=╠═╡
const mountaincar_tile_setup = setup_mountain_car_tiles((1/20f0, 1/20f0), 10)
  ╠═╡ =#

# ╔═╡ 0714a1cf-9288-4f1e-ba72-d82608704d69
# ╠═╡ skip_as_script = true
#=╠═╡
const mc_test = gradient_monte_carlo_control_linear(mountain_car_mdp, 1f0, 100, mountaincar_tile_setup.feature_vector, mountaincar_tile_setup.update_feature_vector!; α = 1f-8, ϵ = 0.01f0, max_steps = 100_000, suppress_warning = true, use_unfinished_episodes = true)
  ╠═╡ =#

# ╔═╡ c85033e1-3ee6-42ad-9ef0-144ce6238ce4
#=╠═╡
function smooth_error(error_history, n)
	l = length(error_history)
	[mean(error_history[i-n:i]) for i in n+1:l]
end
  ╠═╡ =#

# ╔═╡ c1388562-0708-4a6a-acfe-927413dab5d2
# ╠═╡ skip_as_script = true
#=╠═╡
plot(scatter(y = smooth_error(-episode_rewards, 100)), Layout(yaxis_type = "log"))
  ╠═╡ =#

# ╔═╡ 5c920177-8e46-49c9-9b95-1a657fdcae4e
# ╠═╡ skip_as_script = true
#=╠═╡
plot(scatter(y = -smooth_error(episode_rewards_dp, 1)), Layout(yaxis_type = "log"))
  ╠═╡ =#

# ╔═╡ 4ccb8a52-c6af-445d-a39e-d4d9b10c0d6a
# ╠═╡ skip_as_script = true
#=╠═╡
plot(scatter(y = -smooth_error(mountain_car_fcann_sarsa.episode_rewards, 1)), Layout(yaxis_type = "log"))
  ╠═╡ =#

# ╔═╡ 09088eee-4cb3-40ac-b127-658ce1332fba
#=╠═╡
plot(scatter(y = -smooth_error(episode_rewards_q, 100)), Layout(yaxis_type = "log", xaxis_title = "Episode", yaxis_title = "Number of Steps"))
  ╠═╡ =#

# ╔═╡ 01c1958a-0690-4a69-8158-8cacc69e1bff
#=╠═╡
plot(scatter(y = -smooth_error(mountaincar_fcann_results2.episode_rewards, 10)), Layout(yaxis_type = "log", xaxis_title = "Episode", yaxis_title = "Number of Steps"))
  ╠═╡ =#

# ╔═╡ 425fe768-c7bb-4d3e-87e6-47fa052ba612
#=╠═╡
plot(smooth_error(average_step_reward_fcann, 10)[round.(Int64, (LinRange(1, length(average_step_reward_fcann) - 10, 1000)))])
  ╠═╡ =#

# ╔═╡ b76551e0-c027-4682-b5ae-bba7ea2b987a
# ╠═╡ skip_as_script = true
#=╠═╡
plot(smooth_error(mc_test.step_history, 10), Layout(yaxis_type = "log"))
  ╠═╡ =#

# ╔═╡ 954848db-6dcc-4666-90f8-b5a900203242
#=╠═╡
show_mountaincar_trajectory(s -> mc_test.value_function(s).maximizing_action, 1000, "MC Learned Policy")
  ╠═╡ =#

# ╔═╡ b55d50a4-b039-4240-b434-42f7b724d24d
# ╠═╡ skip_as_script = true
#=╠═╡
plot_mountaincar_action_values(mc_test.value_function, 100, 100)
  ╠═╡ =#

# ╔═╡ e4e572b0-eea6-4cf3-85cd-bbe7f2c687e6
#=╠═╡
mc_test2 = gradient_monte_carlo_control_linear(MountainCarTask.deterministic_mdp, 1f0, 1000, mountaincar_tile_setup.feature_vector, mountaincar_tile_setup.update_feature_vector!; α = 1f-8, ϵ = 0.1f0, max_steps = 10_000, suppress_warning = true, use_unfinished_episodes = true)
  ╠═╡ =#

# ╔═╡ 4282d334-5c18-4805-99b1-59930165de98
#=╠═╡
plot(smooth_error(mc_test2.step_history, 10), Layout(yaxis_type = "log"))
  ╠═╡ =#

# ╔═╡ fa048d3d-6a5a-4f47-8e58-2a3b6a905e50
#=╠═╡
show_mountaincar_trajectory(s -> mc_test2.value_function(s).maximizing_action, 1000, "MC Learned Policy")
  ╠═╡ =#

# ╔═╡ ca4928fb-0fcb-4835-95ff-a65abf5102b8
#=╠═╡
plot_mountaincar_action_values(mc_test2.value_function, 100, 100)
  ╠═╡ =#

# ╔═╡ 8ae2f369-8c73-4116-a6d8-1a1e4aae35e0
md"""
### Non-linear Approximation
"""

# ╔═╡ c75dc51c-cbff-48b1-b0fd-108828929b51
#uses an action value function to learn optimal policy
"""
    gradient_monte_carlo_control_fcann(mdp, γ, num_episodes, update_feature_vector!,
                                      num_features, layers; reslayers=0, use_μP=true,
                                      parameters, dropout=0, activation_list, l2=0,
                                      kwargs...) -> NamedTuple

Learn optimal policy using gradient Monte Carlo control with fully-connected neural network approximation.

Provides specialized implementations for neural network value function approximation, automatically
selecting between action-value and state-value approaches based on MDP transition structure.
Uses [`FCANNParams`](@ref) for efficient neural network parameter storage and computation.

## Action-Value Function Approximation Method
For direct q̂(s,a) neural network approximation when transition distribution is not available:

# Type Parameters
- `T <: Real`: Numeric type for rewards, parameters, and computations

# Arguments
- `mdp::`[`StateMDP`](@ref): Markov Decision Process environment
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `num_episodes::Integer`: Number of episodes to run
- `update_feature_vector!::Function`: Extract features: `(feature_vector, state) -> nothing`
- `num_features::Integer`: Input feature dimension for neural network
- `layers::Vector{Int64}`: Hidden layer sizes for neural network architecture

# Keyword Arguments
- `reslayers::Integer = 0`: Number of residual connection layers
- `use_μP::Bool = true`: Whether to use μP (Maximal Update Parameterization) initialization
- `parameters::`[`FCANNParams{T}`](@ref) `= FCANN.initializeparams_saxe(num_features, layers, length(mdp.actions), reslayers; use_μP)`: Network parameters for action-value function (outputs |actions| values)
- `dropout::T = zero(T)`: Dropout probability for regularization
- `activation_list::Vector{Bool} = fill(true, length(layers))`: Activation function flags per layer
- `l2::T = zero(T)`: L2 regularization strength
- `kwargs...`: Additional arguments passed to [`gradient_monte_carlo_control!`](@ref)

## State-Value Function Approximation Method
For v̂(s) neural network approximation when transition distribution is available:

# Type Parameters
- `T <: Real`: Numeric type for rewards, parameters, and computations
- `S`: State type (scalar or N-dimensional tuple)
- `A`: Action type
- `P <: StateMDPTransitionDistribution`: Transition probability distribution type
- `F1 <: Function`: State initialization function
- `F2 <: Function`: Transition function
- `F3 <: Function`: Termination function

# Arguments
- `mdp::StateMDP{T, S, A, P, F1, F2, F3}`: Markov Decision Process with transition distribution
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `num_episodes::Integer`: Number of episodes to run
- `update_feature_vector!::Function`: Extract features: `(feature_vector, state) -> nothing`
- `num_features::Integer`: Input feature dimension for neural network
- `layers::Vector{Int64}`: Hidden layer sizes for neural network architecture

# Keyword Arguments
- `reslayers::Integer = 0`: Number of residual connection layers
- `use_μP::Bool = true`: Whether to use μP (Maximal Update Parameterization) initialization
- `parameters::`[`FCANNParams{T}`](@ref) `= FCANN.initializeparams_saxe(num_features, layers, 1, reslayers; use_μP)`: Network parameters for state-value function (outputs single value)
- `dropout::T = zero(T)`: Dropout probability for regularization
- `activation_list::Vector{Bool} = fill(true, length(layers))`: Activation function flags per layer
- `l2::T = zero(T)`: L2 regularization strength
- `kwargs...`: Additional arguments passed to [`gradient_monte_carlo_control!`](@ref)

# Returns
- `NamedTuple`: Same structure as [`gradient_monte_carlo_control!`](@ref) with fields:
  - `value_function::Function`: Action-value function `(state; activations, kwargs...) -> NamedTuple` with pre-allocated activations, returns `(action_values, maximizing_action, maximizing_value)`
  - `step_history::Vector{Int64}`: Episode step counts
  - `reward_history::Vector{T}`: Episode reward totals
  - `error_history::Vector{T}`: Approximation errors
  - `parameters::`[`FCANNParams{T}`](@ref): Final neural network parameters
  - `success_rate::Real`: Episode success fraction

# See Also
[`gradient_monte_carlo_control!`](@ref), [`FCANNParams`](@ref), [`setup_fcann_action_value_arguments`](@ref), [`setup_fcann_value_arguments`](@ref), [`FCANN.initializeparams_saxe`](@ref)

# Implementation
Convenience wrapper that automatically configures neural network function approximation components:

1. **Method Selection**: Chooses action-value or state-value approach based on MDP transition type
2. **Parameter Initialization**: Creates neural network parameters using [`FCANN.initializeparams_saxe`](@ref) with appropriate output dimensions
3. **Component Setup**: Configures network components via [`setup_fcann_action_value_arguments`](@ref) or [`setup_fcann_value_arguments`](@ref)
4. **Activation Management**: Returns value function with pre-allocated activation storage for efficiency
5. **Delegation**: Calls [`gradient_monte_carlo_control!`](@ref) with configured neural network components

**Action-value method** creates network with `|actions|` outputs for direct q̂(s,a) approximation.
**State-value method** creates network with single output and computes action values via transition distribution.

# Performance Notes
- Uses μP initialization for stable training across network widths
- Pre-allocates activation storage to avoid repeated memory allocation
- Supports dropout and L2 regularization for improved generalization
- Residual connections available for deeper networks

# Examples
```julia-repl
julia> mountaincar_tile_setup = setup_mountain_car_tiles((1/10f0, 1/10f0), 12);
julia> output = gradient_monte_carlo_control_fcann(mountain_car_mdp, 1f0, 1000,
                                                  mountaincar_tile_setup.update_feature_vector!,
                                                  mountaincar_tile_setup.num_features,
                                                  [64, 32]; α=1f-4, ϵ=0.1f0,
                                                  max_steps=10_000, suppress_warning=true,
                                                  ignore_unfinished_episodes=true);
julia> output.value_function(mountain_car_mdp.initialize_state())
(action_values = [0.12f0, -0.08f0, 0.15f0], maximizing_action = 3, maximizing_value = 0.15f0)
```
"""
function gradient_monte_carlo_control_fcann(mdp::StateMDP, γ::T, num_episodes::Integer, feature_vector, update_feature_vector!::Function, layers::Vector{Int64}; reslayers::Integer = 0, use_μP::Bool = true, parameters::FCANNParams{T} = initialize_fcann_params(feature_vector, layers, length(mdp.actions), reslayers, use_μP), dropout = zero(T), activation_list = fill(true, length(layers)), l2 = zero(T), use_gpu::Bool = false, kwargs...) where T<:Real
	setup = setup_fcann_action_value_arguments(parameters, l2, dropout, use_μP, activation_list; use_gpu = use_gpu)
	!use_gpu && return gradient_monte_carlo_control!(parameters, mdp, γ, num_episodes, feature_vector, update_feature_vector!, setup.update_action_values!, setup.gradient, setup.update_value_gradient!; kwargs...)

	isempty(setup.gpu_args) && error("GPU backend is not available")
	output = gradient_monte_carlo_control!(setup.gpu_args.params, mdp, γ, num_episodes, feature_vector, update_feature_vector!, setup.update_action_values!, setup.gpu_args.gradient, setup.update_value_gradient!; kwargs...)
	FCANN.GPU2Host(parameters.weights, setup.gpu_args.params.weights)
	setup.gpu_args.cleanup_vars()
	(;output..., parameters = parameters)
end

# ╔═╡ a9d1381b-566a-4422-81fc-38efde1d2608
#when the transition distribution is available uses the state value function to learn optimal policy
function gradient_monte_carlo_control_fcann(mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, num_episodes::Integer, feature_vector, update_feature_vector!::Function, layers::Vector{Int64}; reslayers::Integer = 0, use_μP::Bool = true, parameters::FCANNParams{T} = initialize_fcann_params(feature_vector, layers, 1, reslayers, use_μP), dropout = zero(T), activation_list = fill(true, length(layers)), l2 = zero(T), use_gpu::Bool = false, kwargs...) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1<:Function, F2<:Function, F3<:Function}
	setup = setup_fcann_value_arguments(parameters, l2, dropout, use_μP, activation_list; use_gpu = use_gpu)
	!use_gpu && return gradient_monte_carlo_control!(parameters, mdp, γ, num_episodes, feature_vector, update_feature_vector!, setup.value_function, setup.gradient, setup.update_gradient!; kwargs...)

	isempty(setup.gpu_args) && error("GPU backend is not available")
	output = gradient_monte_carlo_control!(setup.gpu_args.params, mdp, γ, num_episodes, feature_vector, update_feature_vector!, setup.value_function, setup.gpu_args.gradient, setup.update_gradient!; kwargs...)
	FCANN.GPU2Host(parameters.weights, setup.gpu_args.params.weights)
	setup.gpu_args.cleanup_vars()
	(;output..., parameters = parameters)
end

# ╔═╡ d81e0a66-626f-467f-9748-2f5d407a8815
#=╠═╡
const mc_fcann_sarsa = gradient_monte_carlo_control_fcann(mountain_car_mdp, 1f0, 10, zeros(Float32, 2), update_mountaincar_feature_vector!, fill(32, 4); reslayers = 1, α = 5f-9, ϵ = 0.01f0, max_steps = 10_000, suppress_warning = true, use_unfinished_episodes = true)
  ╠═╡ =#

# ╔═╡ 702927e2-23c0-48a8-85aa-f406710e3ac8
#=╠═╡
plot(smooth_error(mc_fcann_sarsa.step_history, 10), Layout(yaxis_type = "log"))
  ╠═╡ =#

# ╔═╡ ce3de885-8c9d-4ae1-b43b-c011e140af58
#=╠═╡
show_mountaincar_trajectory(s -> mc_fcann_sarsa.value_function(s).maximizing_action, 1000, "MC Learned Policy")
  ╠═╡ =#

# ╔═╡ 11ad4137-2145-452f-b01e-6fffb3a69cdd
#=╠═╡
plot_mountaincar_action_values(mc_fcann_sarsa.value_function, 100, 100)
  ╠═╡ =#

# ╔═╡ 9b3035f6-fe59-4748-a1cd-3c2ce61c6608
const mc_test3 = gradient_monte_carlo_control_fcann(MountainCarTask.deterministic_mdp, 1f0, 10, zeros(Float32, 2), update_mountaincar_feature_vector!, fill(64, 4); α = 4f-8, ϵ = 0.01f0, max_steps = 10_000, suppress_warning = true, use_unfinished_episodes = true)

# ╔═╡ 52ab5b04-8500-4310-8723-0fba097358da
#=╠═╡
plot(smooth_error(mc_test3.step_history, 10), Layout(yaxis_type = "log"))
  ╠═╡ =#

# ╔═╡ ae790d84-ebdb-4dd6-9abf-49096ca8567a
#=╠═╡
show_mountaincar_trajectory(s -> mc_test3.value_function(s).maximizing_action, 1000, "MC Learned Policy")
  ╠═╡ =#

# ╔═╡ 9e18c459-5f76-40cc-a8bc-c513492bb0ea
#=╠═╡
plot_mountaincar_action_values(mc_test3.value_function, 100, 100)
  ╠═╡ =#

# ╔═╡ 6cea9e69-bf8c-4079-9884-663a728d7b08
md"""
# Dependencies
"""

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
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
BenchmarkTools = "~1.6.3"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
PlutoDevMacros = "~0.9.1"
PlutoPlotly = "~0.6.5"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.73"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "3d1178c7404f262152d31e2855f70e2a3293599d"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "7fecfb1123b8d0232218e2da0c213004ff15358d"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.3"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "980f01d6d3283b3dbdfd7ed89405f96b7256ad57"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "2.0.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

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

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "d60eb76f37d7e5a40cc2e7c36974d864b82dc802"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.1"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

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

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

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
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "277779adfedf4a30d66b64edc75dc6bb6d52a16e"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.6"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

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
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "28278bb0053da0fd73537be94afd1682cc5a0a83"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.21"

    [deps.PlotlyBase.extensions]
    DataFramesExt = "DataFrames"
    DistributionsExt = "Distributions"
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyBase.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlutoDevMacros]]
deps = ["JuliaInterpreter", "Logging", "MacroTools", "Pkg", "TOML"]
git-tree-sha1 = "1cb861c9295d79dc6e23170d4b33bce013f69643"
uuid = "a0499f29-c39b-4c5c-807c-88074221b949"
version = "0.9.1"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "8acd04abc9a636ef57004f4c2e6f3f6ed4611099"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.5"

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
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "3faff84e6f97a7f18e0dd24373daa229fd358db5"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.73"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProfileCanvas]]
deps = ["FlameGraphs", "JSON", "Pkg", "Profile", "REPL"]
git-tree-sha1 = "41fd9086187b8643feda56b996eef7a3cc7f4699"
uuid = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
version = "0.1.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
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
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

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

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

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
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
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
# ╟─d88ebdb9-47bc-478c-b471-804a02ad2acf
# ╠═9043a684-6f16-48d0-83d4-2e00f9b7dbc2
# ╠═3273ed4a-6787-4635-8399-65ddf65b31ea
# ╠═0226d8a3-bb22-4a32-9700-e234abf518a6
# ╠═08c74b7d-7aa6-4085-a09b-b6191f8d098e
# ╠═1393f7a6-05c7-48a3-96a9-130eb6d45937
# ╠═7956c5af-4f26-4790-88c6-9e4d2a7b37f7
# ╠═585140d8-4c17-4adb-999c-ef4a72ab07b7
# ╠═94fa7f7d-c77c-4df5-a7b9-b3c931cb3bce
# ╠═d82faf3b-c975-4b23-ad62-473bd943c4e2
# ╟─fc0b88f3-fbf9-450d-b770-b34357ffad49
# ╟─f9057d17-00fe-4cc9-83a1-fef34c116b25
# ╠═991492f4-7dfc-43aa-ab6c-a6b1f3e38225
# ╠═8b7e1031-9864-439c-86eb-11aa08f53b90
# ╟─05e2fff5-4871-4468-a00e-9c1b7ba0ffc6
# ╟─57a6510f-bd42-4d1d-a550-d1442f79569f
# ╠═bf980179-911b-4dd6-8abe-16f6e497a0bc
# ╠═2ef47fe1-e082-406b-b131-5e2ae1bcb08b
# ╠═a4c6a5c0-29c5-440c-bf86-20d0f881ee06
# ╠═97e56e3f-1ef7-45a5-8261-c8fa103b9747
# ╠═b0761704-5447-4e64-8270-708d9dccef60
# ╠═de3e4afe-f935-4b33-9218-08d403743c60
# ╟─8513264e-6a14-41ab-8cfd-a335682a06aa
# ╟─b697c5ba-4647-4998-a153-1e97dd91cb23
# ╟─b8cd582e-26fc-4f21-85cc-950bac60bee0
# ╟─526689e2-85ea-47d5-9791-5aa730f8b1ab
# ╟─8d096d0d-8fea-421a-aa33-82269d3fe7e2
# ╠═be1ad356-de4b-469c-bb65-81d630f07674
# ╠═7e87f2ec-c96f-4897-bb61-c27913f6944f
# ╠═8cdf042f-2214-48e0-afc2-c6a7d385ee4e
# ╠═6d4b513d-2744-4f9c-8bee-e51fe9d0bade
# ╠═4c94be37-dcd7-4b32-8e7f-3371ddaa254a
# ╟─a22e5d34-4b8d-479c-985c-d6abd41a6c80
# ╟─b990ba67-42c8-4ab9-943d-085392204fdd
# ╠═f221fb13-4ef2-4ebe-b71b-fe6adbddb1e4
# ╠═1e9c537a-a731-4b81-8f6a-cb658b52c5be
# ╠═5b2ffd90-ead0-42ce-999a-584ed8995910
# ╟─f6e08689-040f-4565-9dfb-e9a65d1c1f18
# ╟─528533f7-68f1-4d19-9a37-6d4d0d7c38e2
# ╠═d42bb733-07e2-4932-aab4-09229ff67492
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
# ╟─742100ba-c38e-4840-8988-40990039b527
# ╠═e5c0b558-4902-455f-a370-cddb9b291c15
# ╟─9ffad966-a568-437a-b9ab-522c08ba681c
# ╠═7c5fb569-81f0-4b70-ae95-1fce0c51b6f4
# ╠═30ab21ba-3f5b-46a8-8b8c-753f2755d419
# ╠═ae1adf97-1a2d-44ff-98ab-422899afd096
# ╠═4afbb723-340b-4d85-9115-027a0ff8dfad
# ╠═f2201afe-8952-4dde-9e39-02beeb920f6f
# ╟─c1388562-0708-4a6a-acfe-927413dab5d2
# ╟─ddcb50be-5287-47f8-89f9-58c026a6b151
# ╟─af97f222-08d1-4200-a10b-8da178182175
# ╟─224b4bec-9ec5-434d-a950-f5974cd786d0
# ╠═b0cc6ff8-7296-461c-9db7-e52fa518e2e2
# ╠═d0cf3806-05c6-4a50-94c8-55c9042d51b7
# ╠═bd1f42e5-94cc-4aef-b82a-9bffd1c951d8
# ╟─7d21c4cd-ab79-4f40-9b8b-f637b3efcab0
# ╟─5c920177-8e46-49c9-9b95-1a657fdcae4e
# ╟─31fb07d2-1c34-44ec-b932-a598e78ec8dc
# ╠═c12070a9-df63-4b25-99e6-26ff876af1b4
# ╟─680561af-db37-440c-9c48-2969e8fd99fc
# ╠═c11aa069-93c2-435a-8f0e-353ced9633b6
# ╠═5fdbce61-ca25-45e0-b07d-94adf7138446
# ╠═7cef3dab-7091-4293-a2fb-edddb15a8af8
# ╟─fc3e0577-45aa-4bba-a275-fa7a352fc5cc
# ╠═4ccb8a52-c6af-445d-a39e-d4d9b10c0d6a
# ╟─6bcd0ce5-f059-4adc-9cec-c51d0b98ce19
# ╠═0f958535-6b18-46de-a1ba-81f64c217ee0
# ╠═ee59176e-24b6-4213-8f8e-759a70bc1d5e
# ╠═b3658e4d-ee8e-45cd-906a-06dd512a6921
# ╠═1e224a46-91ef-4a5f-ae35-ef4062147f2d
# ╠═00399548-b21c-43b5-90e2-30656ab1541e
# ╠═1a82ae95-3c3e-4281-bc1d-9eb19bf50286
# ╠═5db29488-a150-42ee-aedb-380a3a4fd548
# ╟─59ec5223-f23f-4f32-9e5f-8a08e450da85
# ╟─49249ac1-8964-4afc-89f2-3cd4d4322cc2
# ╟─e1abf8c7-06b8-4cd5-b557-1d187004bdf1
# ╟─98a5d65e-4253-4523-a74e-99d03be03b89
# ╠═8ed6f8fd-8574-4d5a-9964-ce8a32629c6f
# ╠═1410db13-4b73-4a87-af34-30a5232af4ba
# ╟─f7410fe7-e3d8-4047-8fa7-f076476e9d3a
# ╠═cbac1927-b087-4c4c-98ae-6aa5f0b824ad
# ╠═b5409b69-a254-4355-b2b9-99394eceb2f7
# ╠═f9ee13e8-7406-4fba-9a30-1e2714bd7cfc
# ╠═09088eee-4cb3-40ac-b127-658ce1332fba
# ╠═7c160da9-d546-42f8-ad99-7e74c96cabe5
# ╠═00bd6bfc-2ea6-4fcc-8c51-cb7aabb5ce25
# ╠═26712d1f-d5d1-4784-967c-f1682c3e07aa
# ╠═01c1958a-0690-4a69-8158-8cacc69e1bff
# ╟─d6ad1ff1-8fbf-4799-8b1b-ae1e3ce88c5b
# ╟─b8c031ca-7995-4501-a1e3-df3f34e5f0da
# ╟─5f9a2231-8c4a-4519-adbe-a0dd92838ba4
# ╟─4154e827-6d0b-4b94-9f14-64baa85739af
# ╟─e90a591d-0bd0-46a9-8327-f61bfb155a31
# ╟─7cb7ca66-3130-4f06-a0dc-a3335ef85fdb
# ╟─4a5805f0-0cff-4e30-8305-304340734232
# ╟─37db9d03-1978-4842-a016-f416c33ba1d7
# ╟─81a0a342-f92a-4f5a-a173-fd555188895f
# ╟─a92db2a7-3a6c-4328-a2a8-cb74d2e671e9
# ╟─44b7a560-d03b-4636-ad24-b30c8965ab8f
# ╠═5a9bcf45-a04b-4a81-b825-9891021c8a15
# ╠═25159f84-a120-4a20-aab8-010c110571a4
# ╠═e6bf5b6e-75cd-49b3-bf36-7ed6dee11aaf
# ╟─69a06405-57cd-42e5-96b1-5cc77d74aa03
# ╠═a9fdb1fd-3f62-4e1c-9157-c4eee6215261
# ╟─efee131c-318a-40d6-be83-ce24edbbe11c
# ╠═aceeb425-cd5f-4c4c-903e-d4359d2de88d
# ╟─9b629126-0b8f-4592-8727-cbe710bd4a24
# ╠═db778942-1bed-4c42-a2f0-a176a0364772
# ╟─063e6f33-8b65-463c-a96f-5411f0ba0326
# ╠═91447aff-5598-4f02-acd5-6a90c563f4f6
# ╠═4e955391-ac29-412e-8ed2-bad3b46961b0
# ╠═12fa7b75-d13f-4a16-8562-1142002f3f3f
# ╟─9b56eac4-10be-42c3-b3a9-a0c4852b7cce
# ╠═7c22d050-bd56-4b84-8a01-e575475db099
# ╟─571fad6e-ca32-4661-bc48-62f3f49d124b
# ╠═e04b9ac4-7e7f-4f6a-b068-d62b319a23fa
# ╟─1a7ba296-52ca-4069-85fa-792d08d77b0e
# ╠═eb28458f-b222-4f8e-9a5b-8203d3997f7b
# ╠═d66cd124-7111-401a-a3e8-1059b31c6db7
# ╠═bc1d7cce-c0f4-47a8-b674-8acb82491c7f
# ╟─0a494e3e-0af5-4497-b80e-e471acc1fabc
# ╠═49e43d51-05d6-415b-a685-76e50904c5bc
# ╠═db189316-e880-4cc8-9070-ccfe2b4fc545
# ╠═7bc49107-9de5-4985-8750-979f36b3aa81
# ╠═ab4cb3db-3a2d-4145-826b-b1001114eeff
# ╠═0e3e506d-1959-47fd-8da9-b3dfd294be67
# ╠═53c5558b-e713-4c72-bdf8-e162c3892e6f
# ╟─86cd431e-7b05-410a-b943-ba03b286f3f0
# ╠═d3ba78fa-f032-4bb9-9359-ef3bcff2252d
# ╠═ae5c5377-8b44-4c82-a63c-d2cb8a0d6667
# ╠═6d4016c6-8edd-4466-9cc4-015452c669ba
# ╠═2df5adc1-130b-4982-a4bc-7e0c7417923e
# ╠═2306039b-7b4d-4013-be1b-1402231ef8e8
# ╠═425fe768-c7bb-4d3e-87e6-47fa052ba612
# ╠═b191d3f9-cf25-4fb4-8f5a-8da86e96e125
# ╠═c44dd6c6-8213-49fb-8d33-ba8f2c766b2e
# ╟─a6e0c082-7f1f-4352-8c23-c3b64fd74493
# ╟─6f79c437-7264-412c-839f-5bc9252eede8
# ╠═501b7284-6e04-4a15-b8e4-2601156b0345
# ╠═2441b61e-5954-41e2-8ee4-38b16ed04cef
# ╟─6f4f8b64-0c17-446e-bfb6-0540871ad9e0
# ╠═1a56e4dd-15dd-47b3-afd8-1dd7f5b690ac
# ╠═c94da551-06b2-4e2b-bf39-ceb5cb5c390c
# ╟─0e34a25b-f8ee-4da9-8664-b6c094163759
# ╠═3b66c97b-ebad-4d13-987c-ac0172b349d1
# ╠═86f7dcde-b27e-4096-bec8-c5d17fd553d2
# ╠═defe9c74-d514-44ea-af09-fb77764dfaa4
# ╠═7ed745f4-5b7d-41b4-a40e-5b782ca12530
# ╠═ad692a51-e93b-4480-8a6c-2ad86dc6766b
# ╟─cf00a316-38e3-4423-9909-d5ffbd7c0b06
# ╠═89288ce6-11e8-41f3-b32d-e19edee7db33
# ╟─9df1a18d-137c-4ea5-8d15-05697f7bbf07
# ╟─0c7f5742-6c51-4c6a-b67f-217163935ba5
# ╟─a6c5ec28-b2d5-4893-a118-95c1318d1f7f
# ╟─44d43dbf-fe32-438e-b89d-c677bbc35893
# ╟─f56743d1-d681-4bcf-b1bb-c4ef78a75180
# ╟─b242d3b2-396c-4cb6-8c9c-38d16dc18636
# ╟─a28f57c1-e48c-4f4f-8795-bdd195b26135
# ╟─2d7679ad-a9b3-448b-a4bc-7e5b9bce6adb
# ╟─a9b74949-9392-4048-bcb6-5fd48c1d9b98
# ╠═fbf1c64f-1979-4384-a8c6-dc7875174d1f
# ╠═e7372e2b-a2db-4a93-9efc-f75aa74c197b
# ╠═014339eb-5b23-4ac5-a551-8eeb2238366f
# ╟─78251726-a5ec-4fa3-adcb-09ff347ff54a
# ╠═62839b2a-398a-4445-87d1-b15ff2acc1d1
# ╠═b4af8d87-a6e5-4e09-92b4-b07757f58f7f
# ╠═546a775e-d3c9-4693-9f64-d4c47a84fb9f
# ╠═41c626c7-908d-4ff6-9730-4ad0b8c3cc25
# ╠═c914fc12-d650-400b-8aff-e2a55bb2d5cf
# ╟─708164fd-93ea-4720-ad6d-22e1c297c22a
# ╠═c4ba34b5-e657-4b75-b853-0a2df081e34b
# ╟─f9ad39d4-d2b6-44f3-a444-bcabd926a743
# ╟─dbde6c7c-a0ff-41bc-9a26-ffd38561a5ef
# ╠═84942647-8826-4864-b7d4-c31f9d78fd48
# ╠═54700e88-2c70-4b3e-bc93-6960dc70efcb
# ╠═28f9d40a-4f4f-4bbf-ac36-4964afed7ab4
# ╟─32b3c5b4-cdb8-43be-a398-6e158254c4a7
# ╠═5a73ef20-dfdb-4d75-8790-805d6da27462
# ╟─3985641e-2f07-4029-8047-51579904cd53
# ╠═350e057e-154f-4d0b-91fb-ffde9cc9059f
# ╟─f009970f-bf6c-46dd-a534-a960582ce51b
# ╟─64096af2-3cbe-4f6b-944d-6f4bdc2cd535
# ╠═f1f6f750-8c49-4435-82a1-e13a280b3738
# ╟─662759be-282c-460b-adc3-8595475b53c2
# ╟─b727e6e7-e019-4697-85a6-c2cf839ef34a
# ╟─68e3ef82-0706-449c-b00a-dc69e6c7b717
# ╟─0e66a941-1ec1-4d3b-b064-e5f25cc93baf
# ╟─c316c5d3-f484-4e8e-bd56-be1e236d96bc
# ╟─bc220d14-97fd-486d-9880-6908135fe036
# ╟─39eada35-8c3e-4ddc-8df9-7cf9f120928d
# ╟─8752c98d-fac1-4b3b-b20b-70acc0677fcb
# ╟─50f6ff51-d81b-4e97-9f8a-0daf03af7192
# ╠═0d3d5304-0412-485d-8f56-f4362a74ea45
# ╠═604a2621-aa73-42d9-9255-e5f5578d0b51
# ╠═06834750-cc3a-468a-b0c2-81349c288a33
# ╠═d04bf8ac-9905-4e80-93db-c5c28c31359b
# ╟─31260d29-6131-4e44-b6e6-e78399501c54
# ╠═b4085947-f4c7-4664-8d94-8090a67ea6c4
# ╠═164c68ef-01b8-43be-bc75-919dd99a6e03
# ╠═cc285969-c33f-4d19-8e47-397b59e67299
# ╠═0714a1cf-9288-4f1e-ba72-d82608704d69
# ╠═c85033e1-3ee6-42ad-9ef0-144ce6238ce4
# ╠═b76551e0-c027-4682-b5ae-bba7ea2b987a
# ╠═954848db-6dcc-4666-90f8-b5a900203242
# ╠═b55d50a4-b039-4240-b434-42f7b724d24d
# ╠═e4e572b0-eea6-4cf3-85cd-bbe7f2c687e6
# ╟─4282d334-5c18-4805-99b1-59930165de98
# ╟─fa048d3d-6a5a-4f47-8e58-2a3b6a905e50
# ╟─ca4928fb-0fcb-4835-95ff-a65abf5102b8
# ╟─8ae2f369-8c73-4116-a6d8-1a1e4aae35e0
# ╠═c75dc51c-cbff-48b1-b0fd-108828929b51
# ╠═a9d1381b-566a-4422-81fc-38efde1d2608
# ╠═d81e0a66-626f-467f-9748-2f5d407a8815
# ╟─702927e2-23c0-48a8-85aa-f406710e3ac8
# ╠═ce3de885-8c9d-4ae1-b43b-c011e140af58
# ╠═11ad4137-2145-452f-b01e-6fffb3a69cdd
# ╠═9b3035f6-fe59-4748-a1cd-3c2ce61c6608
# ╟─52ab5b04-8500-4310-8723-0fba097358da
# ╠═ae790d84-ebdb-4dd6-9abf-49096ca8567a
# ╠═9e18c459-5f76-40cc-a8bc-c513492bb0ea
# ╟─6cea9e69-bf8c-4079-9884-663a728d7b08
# ╠═bee124fd-5605-4512-833b-945ef77c056e
# ╠═fa5fecfd-c039-4063-9acb-365a046e06f2
# ╠═9fb5dace-a799-4424-bcb3-8542e508dd4b
# ╠═ed1bd92c-8cc7-457f-9692-a10a9487c953
# ╠═dd472c0f-7b43-4abe-ada9-9dc8004a18cb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
