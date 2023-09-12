### A Pluto.jl notebook ###
# v0.19.42

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

# ╔═╡ 1fb1a518-e5ec-4777-80bc-bb55e8172100
begin
	using Random, Base.Threads, PlutoPlotly, PlutoUI, PlutoProfile, Latexify, LaTeXStrings, SpecialFunctions, Distributions, Statistics, StatsBase
	TableOfContents()
end

# ╔═╡ db30e6c0-36bb-4602-a257-5768f3833525
md"""
# Chapter 2: Multi-armed Bandits

## 2.1: A *k*-armed Bandit Problem

Consider a repeated choice among *k* different options.  A numerical reward is chosen from a stationary probability distribution that depends only on the action selected.  The objective is to maximize the accumulated reward over some time period, let's say 1000 action selections or *time steps*.

We denote the action selected on time step *t* as $$A_t,$$ and the corresponding reward as $$R_t.$$  The value then of an arbitrary action $$a$$, denoted $$q_*(a),$$ is the expected reward given that $a$ is selected:

$$q_*(a) \dot = \mathbf{E}[R_t|A_t=a]$$

If we know the values, then the problem is trivial, but we assume that we only have estimates of the values at a time step $a$ which we will denote $$Q_t(a).$$  At any given time step the greedy action is the one with the highest value estimate.  If we take non-greedy actions then we can improve our value estimate for other states.  To solve the problem in general we must balance *exploiting* the action estimated to be the best with *exploring* the values of other candidate actions.  What follows are various methods to balance these two choices.
"""

# ╔═╡ 31d8fba9-cd28-4b2b-ba46-70a068b9ecad
md"""
## 2.2: Action-value Methods

The true value is the mean reward when that action is selected.  One way to estimate this is by averaging the rewards actually received:

$$Q_t(a) \dot = \frac{\text{sum of rewards when } a \text{ taken prior to } t}{\text{number of times } a \text{ taken prior to } t} = \frac{\sum_{i=1}^{t-1}R_i \cdot \mathbf{1}_{A_i=a}}{\sum_{i=1}^{t-1} \mathbf{1}_{A_i=a}} \tag{2.1}$$

If the denominator is zero then we instead define $$Q_t(a)$$ by some default value such as 0.  We call this the *sample-average* method for estimating action values because each estmiate is an average of the sample of relevant rewards.

The simplest action selection rule is to select one of the actions with the higest estimated value.  This is the *greedy* action.

$$A_t \dot = \operatorname*{argmax}_a Q_t(a)$$
"""

# ╔═╡ 1e4ac085-7b72-4bad-ad87-21635930a6f7
md"""
> ### *Exercise 2.1* 
> In ϵ-greedy action selection, for the case of two actions and $$\epsilon = 0.5$$, what is the probability that the greedy action is selected?

The greedy action will be selected in two cases, each of which has probability 0.5.  For case 1 the greedy action is explicitely taken so the probability of selecting the greedy action in this case is 1.  For case 2, we select an action randomly, so the probability of selecting the greedy action is $$\frac{\text{num greedy actions}}{\text{num total actions}}=0.5$$
Since both cases are independent, the probabilities can be summed after multiplying each by the probability of that case which is 0.5 for both.

$$P(a = a_{greedy}) = 0.5 \times (1 + 0.5) = 0.5 + 0.25 = 0.75$$
"""

# ╔═╡ 083c721c-70dd-4ca3-8160-fc0b0531914f
md"""
## 2.3 The 10-armed Testbed
The following code recreates the 10-armed Testbed from section 2.3
"""

# ╔═╡ 97c94391-397d-4cf2-88b1-3bea3af56ed3
function create_bandit(k::Integer; offset::T = 0.0) where T<:AbstractFloat
    qs = randn(T, k) .+ offset #generate mean rewards for each arm of the bandit 
end

# ╔═╡ 7950e06b-e8ce-4bd7-9681-ab7b66dfec69
function plot_bandit_testbed(n; npoints = 5_000)
	qs = create_bandit(n)
	spreads = randn(npoints)
	violins = [violin(y = spreads .+ qs[i], name = i, points=false, meanline_visible=true) for i in 1:n]
 	PlutoPlotly.plot(violins, Layout(showlegend = false, yaxis = attr(title = "Reward distribution"), xaxis = attr(title = "Action"), annotations = [attr(x = i-1 + 0.5, y = qs[i], xref = "x", yref = "y", text = "q($i)", showhead=false, ax=0, ay=0) for i in 1:n]))
end

# ╔═╡ 3e5a226e-ecdb-43fb-a40a-a262da0ae542
md"""
Number of Arms in Testbed
$(@bind ktest confirm(NumberField(2:100, default = 10)))
"""

# ╔═╡ 0b951e6e-4b97-4bb5-87d0-6be7f0fd4802
md"""
### Figure 2.1
Shows the reward distribution for each of the $ktest arms in the testbed.  The mean value is marked with a dashed line for each.

$(plot_bandit_testbed(ktest))
"""

# ╔═╡ 14bd0549-747f-4513-809f-8bdb78027807
md"""
### $$\epsilon - Greedy$$ Action Value Method
The functions below implement the sample-average method for estimating the value of each action with the ϵ-greedy method of action selection.  Note that the `simple_algorithm` uses the incremental implementation of calculating the sample average which is described in section **2.4**.
"""

# ╔═╡ 55f89ca1-cd57-44e0-95e5-63a4be31418f
function sample_bandit(a::Integer, qs::Vector{T}) where T<:AbstractFloat
    randn(T) + qs[a] #generate a reward with mean q[a] and variance 1
end

# ╔═╡ 3a798da5-c309-48f2-aab1-6602ded8a650
function simple_algorithm(qs::Vector{Float64}, k::Integer, ϵ::AbstractFloat; 
	steps = 1000, 
	Qinit = 0.0, 
	α = 0.0, 
	c = 0.0, 
	#pre-allocated result vectors, can be provided to function to reduce allocations
	cum_reward_ideal = zeros(steps),
	step_reward_ideal = zeros(steps),
	cum_reward = zeros(steps),
    step_reward = zeros(steps),
	optimalstep = fill(false, steps),
	optimalaction_pct = zeros(steps))

	#define bandit sampling function and initialize action value estimates
    bandit(a) = sample_bandit(a, qs)
    N = zeros(k)
    Q = ones(k) .* Qinit
    accum_reward_ideal = 0.0
    accum_reward = 0.0
   
    bestaction = argmax(qs)
   
    optimalcount = 0
   
    actions = collect(1:k)
    for i = 1:steps
        shuffle!(actions) #so that ties are broken randomly with argmax
        a = if rand() < ϵ
            rand(actions)
		elseif c == 0.0
			#plain ϵ-greedy action selection
			actions[argmax(view(Q, actions))]	
		else
			#UCB action selection
           	actions[argmax(view(Q, actions) .+ (c .* sqrt.(log(i) ./ view(N, actions))))]
		end
        if a == bestaction
            optimalstep[i] = true
            optimalcount += 1
		else
			optimalstep[i] = false
		end

		#sample reward and 
        step_reward[i] = bandit(a) 
        step_reward_ideal[i] = bandit(bestaction)
        accum_reward_ideal += step_reward_ideal[i] 
        cum_reward_ideal[i] = accum_reward_ideal
        accum_reward += step_reward[i] 
        cum_reward[i] = accum_reward
        optimalaction_pct[i] = optimalcount / i
        N[a] += 1.0
        if α == 0.0
			#sample-average
            Q[a] += (1.0/N[a])*(step_reward[i] - Q[a])
        else 
			#exponential recency-weighted average
            Q[a] += α*(step_reward[i] - Q[a])
        end
    end
    return (;Q, step_reward, step_reward_ideal, cum_reward, cum_reward_ideal, optimalstep, optimalaction_pct)
end

# ╔═╡ 33009d31-6d66-4aba-b44c-edd911c2f392
function average_simple_runs(k, ϵ; steps = 1000, n = 2000, Qinit = 0.0, α=0.0, c = 0.0)
    runs = Vector{Vector{Vector{Float32}}}(undef, n)
	outnames = (:step_reward, :step_reward_ideal, :optimalstep, :cum_reward, :cum_reward_ideal, :optimalaction_pct)
	qs = create_bandit(k)
	run1 = simple_algorithm(qs, k, ϵ, steps = steps, Qinit = Qinit, α = α, c = c)
	out = [Float32.(run1[name]) for name in outnames]
	runs[1] = out
    @threads for i in 2:n
		runs[i] = deepcopy(out)
        qs = create_bandit(k)
        run = simple_algorithm(qs, k, ϵ; 
			steps = steps, Qinit = Qinit, α = α, c = c) 
			# cum_reward_ideal = run1.cum_reward_ideal, 
			# step_reward_ideal = run1.step_reward_ideal, 
			# cum_reward = run1.cum_reward, 
			# step_reward = run1.step_reward, 
			# optimalstep = run1.optimalstep, optimalaction_pct = run1.optimalaction_pct) 
    	for j in eachindex(outnames)
			# out[i] .+= Float32.(run[outnames[i]])
			runs[i][j] .= Float32.(run[outnames[j]])
		end
	end
	for i in eachindex(out)
		for j in 2:n
			out[i] .+= runs[j][i]
		end
		out[i] ./= n
	end
    # map(i -> mapreduce(a -> a[i], (a, b) -> a .+ b, runs)./n, (:step_reward, :step_reward_ideal, :optimalstep, :cum_reward, :cum_reward_ideal, :optimalaction_pct))
	return out
end

# ╔═╡ c12dab6a-92f0-41b1-a6b7-c404c74d9a83
function action_value_testbed_plot(; k = 10, ϵ1 = 0.01, ϵ2 = 0.1, steps = 1000)
	qs = create_bandit(k)
	ϵ_list = [0.0, ϵ1, ϵ2]
	runs = [average_simple_runs(k, ϵ; steps = steps) for ϵ in ϵ_list]
	labs = ["ϵ=0 (greedy)", "ϵ=$(ϵ1)", "ϵ=$(ϵ2)"]
	fig22a = PlutoPlotly.plot([map(enumerate(runs)) do (i, a)
		scatter(y = a[1], name = labs[i])
	end; scatter(y = runs[1][2], name = "Theoretical Limit")],
	Layout(xaxis_title = "Step", yaxis_title = "Reward Averaged Over Runs", legend = attr(orientation = "h", y = 1.2), height = 500))
	fig22b = PlutoPlotly.plot(map(enumerate(runs)) do (i, a)
		scatter(y = a[3], name = labs[i])
	end, Layout(xaxis_title="Step", yaxis_title = "% Runs Taking Optimal Action", height = 500, showlegend = false))
	md"""
	### Figure 2.2
	$fig22a
	$fig22b
	"""
end

# ╔═╡ 87f5ec29-623f-4666-824d-fad8c6448072
@bind testbedreset Button("reset")

# ╔═╡ 6034cc0f-cbae-4d2f-a43f-1bb738c00f0b
@bind ϵ_action_value_params confirm(PlutoUI.combine() do Child
	testbedreset
	md"""
	### Testbed Params
	The $(Child(:k, NumberField(2:10, default = 10))) armed testbed with	
	ϵ-greedy 1 = $(Child(:ϵ1, NumberField(0.01:0.01:0.1))) and	
	ϵ-greedy 2 = $(Child(:ϵ2, NumberField(0.1:0.1:1.0))) over $(Child(:steps, NumberField(500:100:10000, default = 1000))) steps
	"""
end)

# ╔═╡ 79082409-3182-4e0b-9c8c-37a94543fee9
action_value_testbed_plot(;ϵ_action_value_params...)

# ╔═╡ 965da91b-6a3f-456c-89ce-461c31e0fb7e
md"""
> ### *Exercise 2.2:* *Bandit example* 
> Consider a k-armed bandit problem with k = 4 actions, denoted 1, 2, 3, and 4. Consider applying to this problem a bandit algorithm using $\epsilon$-greedy action selection, sample-average action-value estimates, and initial estimates of $$Q_1(a) = 0$$, for all a. Suppose the initial sequence of actions and rewards is $$A_1 = 1,$$ $$R_1 = −1,$$ $$A_2 = 2,$$ $$R_2 = 1,$$ $$A_3 = 2,$$ $$R_3 = −2,$$ $$A_4 = 2,$$ $$R_4 = 2,$$ $$A_5 = 3,$$ $$R_5 = 0.$$ On some of these time steps the $$\epsilon$$ case may have occurred, causing an action to be selected at random. On which time steps did this definitely occur? On which time steps could this possibly have occurred?

The table below summarizes the actions taken leading into every step and the Q estimate for each action at the end of each step.  So step 0 shows the initial Q estimates of 0 for every action and the selected action 1 that generates the reward on step 1.  For the row in step 1 it shows the Q estimates after receiving the reward on step 1 and thus what actions are demanded by a greedy choice leading into the next step.  If the action selected is not in the set of greedy actions, then a random action **must** have occured.  Since a random action choice can also select one of the greedy actions, such a random choice is possible at every step.  Note that the answer in row 0 corresponds to action $$A_1$$, row 1 -> $$A_2$$ etc...

|Step| Action Selected | Reward | $$Q(1)$$ | $$Q(2)$$ | $$Q(3)$$ | $$Q(4)$$ | Greedy Action Set | Greedy Selection | $$\epsilon$$ Case |
|----|---- |---- | ---- | ---- |---- | ---- |  ---- | ----  | ---- |
|  1 |  1| -1  |  0  |  0  | 0    | 0 | $$\{1, 2, 3, 4\}$$ | True | possibly |
|  2 |  2 | 1 | -1 |  0  | 0    | 0 | $$\{2, 3, 4\}$$ | True | possibly            |
|  3 |  2 | -2 | -1 |  1  | 0 	| 0 | $$\{2\}$$ | True | possibly |
| 4  | 2 | 2 |   -1 | $$-\frac{1}{2}$$ | 0  | 0 | $$\{3, 4\}$$ | False | definitely |
| 5  |  3 | 0 | -1  | $$\frac{1}{3}$$ | 0 | 0 | $$\{2\}$$ | False | definitely |
"""

# ╔═╡ 464d43c0-cd59-49e6-88f6-12a767677418
md"""
> ### *Exercise 2.3* 
> In the comparison shown in Figure 2.2, which method will perform best in the long run in terms of cumulative reward and probability of selecting the best action? How much better will it be? Express your answer quantitatively.

In the long run both the $$\epsilon = 0.1$$ and $$\epsilon = 0.01$$ methods will have Q value estimates that converge to the true mean value of the reward distribution.  Since both methods will necessarily take random actions 10% and 1% of the time respectively, we'd expect each method to take the optimal action with probability $$(1-\epsilon) + \epsilon \times \frac{1}{10}=\frac{10 - 9 \times \epsilon}{10}$$.  So for each value of ϵ we have.  

$$Pr(a=a_{best}|\epsilon = 0.1) = 0.91$$

$$Pr(a=a_{best}|\epsilon = 0.01) = 0.991$$

For the $$\epsilon = 0$$ greedy case the expected reward and optimal action selection probability depends on the order of sampled actions and the likelihood of getting close to or on the optimal action enough to push its Q estimation to the top.  From the plots in figure 2.2 in practice that seems to lead to an average reward of ~1.05 and an optimal action selection probability of 0.3825.  For long term cummulative reward this case will have roughly $$1.05 \times num\_steps$$.  For the other two cases, the long term cummulative reward is based on the expected value of the highest reward mean which is empiracally ~1.55.  For a random action the expected reward should be 0 due to the normal distribution of the action mean rewards.  

$$E(long\_term\_step\_reward|\epsilon=0.1) = 0.91 \times 1.55 = 1.41$$
$$E(long\_term\_step\_reward|\epsilon=0.01) = 0.991 \times 1.55 = 1.536$$

For each case the long run cumulative reward is just this long term expected reward per step times the number of steps.  The statistical properties of the different arms of a generic bandit can be visualized below using the following function.
"""

# ╔═╡ 8b8a9449-04b7-4901-9a2c-fbbdc33dfdfa
function visualize_bandit_dist(;n = 10, samples = 100_000)
	maxdist = [maximum(randn(n)) for _ in 1:samples]
	rankdist = mapreduce(a -> sort(randn(n)), +, 1:samples) ./ samples
	p1 = histogram(x = maxdist) |> PlutoPlotly.plot
	p2 = bar(x = 1:n, y = rankdist) |> a -> PlutoPlotly.plot(a, Layout(xaxis_title = "Reward Rank", yaxis_title = "Mean Reward of Arm"))
	md"""
	#### Distribution of Best Action Reward for a $n Armed Bandit
	$p1

	#### Expected Value of Mean Reward for Arms Ranked from 1 to $n
	$p2
	"""
end

# ╔═╡ 21e56374-35e6-4488-b8da-15e383017c77
md"""
### Bandit Arm Reward Distributions
"""

# ╔═╡ 2bce1b80-2133-40a0-9367-fc2d491f6245
md"""
Visualize $(@bind n_arms_vis NumberField(1:100, default = 10)) Armed Bandit
"""

# ╔═╡ 4be9e81b-c3de-4c79-97b5-b41c03e0f187
visualize_bandit_dist(n = n_arms_vis, samples = 100_000)

# ╔═╡ e07a27c5-0c9a-4893-a1cf-cf565ab78761
md"""
## 2.4 Incremental Implementation
The sample-average for the action-value estimate is defined as $$Q_n \dot = \frac{R_1+R_2+\cdots+R_{n-1}}{n-1}$$.  To compute this we can maintain a record of every reward and every time we accumulate a new reward recompute the entire average.  However that is inefficient in terms of computational and memory resources.  It is possible to instead maintain a single value for the estimate at step n (and n itself) and update it incrementally every time we obtain a new reward.

The update formula for $$Q_{n+1}$$ when we obtain a new reward sample is derived below:

$$\begin{flalign}
Q_{n+1} &= \frac{1}{n} \sum_{i=1}^n R_i \\
&= \frac{1}{n} \left ( R_n + \sum_{i=1}^{n-1} R_i \right ) \\
&= \frac{1}{n} \left ( R_n + (n-1) \frac{1}{n-1} \sum_{i=1}^{n-1} R_i \right ) \\
&= \frac{1}{n} \left ( R_n + (n-1) Q_n \right ) \tag{definition of Q} \\
&= \frac{1}{n} \left ( R_n + n Q_n - Q_n \right ) \\
&= Q_n + \frac{1}{n} \left [ R_n + Q_n \right ] \tag{2.3}
\end{flalign}$$

The update rule (2.3) is of a form that occurs frequently whose general form is

$$NewEstimate \leftarrow OldEstimate + StepSize \left [ Target - OldEstimate \right ] \tag{2.4}$$

Notice that the step size parameter in this case is $$\frac{1}{n}$$ but in general it can be constant or depend on the step count and the action itself.  In this case it is denoted $$\alpha_t (a)$$.  The simple bandit algorithm that uses this incremental update rule is implemented above in section **2.3**. 
"""

# ╔═╡ 86350532-13f8-4035-bad8-f25f41c93163
md"""
## 2.5 Tracking a Nonstationary Problem
"""

# ╔═╡ a84f5393-bd7e-433e-b3d9-e9e7cfa1a329
md"""
In the case of a non-stationary problem, the sample-average method is not ideal because it weights samples from the past equally to the present.  We can change the incremental implementation of the average to weight more recent rewards higher than past ones.  A constant step-size parameter is one way of doing this.  In this case we will change the update rule (2.3) to:

$$Q_{n+1} \dot = Q_n + \alpha [R_n - Q_n] \tag{2.5}$$

By writing this as an explicite sum over all rewards, one can observe that this update rule computes a weights average whose weights exponentially decay into the past.  See a similar derivation in exercise 2.4.  In order to guarantee that Q converges to the true expected value, the step size parameter must obey the following relationships:

$$\sum_{n=1}^\infty = \infty \quad \quad \text{and} \quad \quad \sum_{n=1}^\infty \alpha_n^2(a) < \infty \tag{2.7}$$

The first condition insures steps are large enough to overcome initial conditions and the second condition insures steps are small enough to converge.  These are both met by the sample average step size of $$\frac{1}{n}$$ for not by the constant step size.  That is desireable in a non-stationary environment where there is no stable value to converge to in the first place.  
"""

# ╔═╡ 78c45162-dc8e-4faf-b1a9-8c71be86dcee
md"""
> ### *Exercise 2.4* 
> If the step-size parameters, $$\alpha_n$$, are not constant, then the estimate $$Q_n$$ is a weighted average of previously received rewards with a weighting different from that given by (2.6). What is the weighting on each prior reward for the general case, analogous to (2.6), in terms of the sequence of step-size parameters?

From (2.6):  $$Q_{n+1} = Q_n + \alpha[R_n - Q_n]$$ so here we consider the case where $$\alpha$$ is not a constant but rather can have a unique value for each step n.

$$\begin{flalign}
Q_{n+1}&=Q_n + \alpha_n[R_n - Q_n]\\
&=\alpha_nR_n+(1-\alpha_n)Q_n\\
&=\alpha_nR_n+(1-\alpha_n)[\alpha_{n-1}R_{n-1}+(1-\alpha_{n-1})Q_{n-1}] \tag{using recursive formula from first step}\\
&=\alpha_nR_n+(1-\alpha_n)\alpha_{n-1}R_{n-1}+(1-\alpha_n)(1-\alpha_{n-1})Q_{n-1}\\
&=\alpha_nR_n+(1-\alpha_n)\alpha_{n-1}R_{n-1}+(1-\alpha_n)(1-\alpha_{n-1})[\alpha_{n-2}R_{n-2}+(1-\alpha_{n-2})Q_{n-2}]\\
&=\alpha_nR_n+(1-\alpha_n)\alpha_{n-1}R_{n-1}+(1-\alpha_n)(1-\alpha_{n-1})\alpha_{n-2}R_{n-2}+...\\
&=Q_1\prod_{i=1}^n \left( 1-\alpha_i \right)+\sum_{i=1}^{n} \left[ (R_i\alpha_i)\prod_{j=i+1}^n(1-\alpha_j) \right]\\
\end{flalign}$$

For example if $$\alpha_i=1/i$$ then the product in the first term is 0 and the formula becomes:

$$\begin{flalign}
Q_{n+1} &= \sum_{i=1}^{n} \left[ \frac{R_i}{i}\prod_{j=i+1}^n\frac{j-1}{j} \right]\\
&= \sum_{i=1}^{n} \left[ \frac{R_i}{i}\frac{i}{i+1}\frac{i+1}{i+2}...\frac{n-1}{n} \right]\\
&=\sum_{i=1}^{n} \frac{R_i}{n}
\end{flalign}$$

	
from the expanded product we can see that all of the numerators and denominators cancel out leaving only $$\frac{R_i}{n}$$ which matches the sample-average as expected for this step-size.
"""

# ╔═╡ eed2a4f2-48b2-4684-846e-aa99bb6dafd9
md"""
> ### *Exercise 2.5 (programming)* 
> Design and conduct an experiment to demonstrate the difficulties that sample-average methods have for nonstationary problems. Use a modified version of the 10-armed testbed in which all the $$q_*(a)$$ start out equal and then take independent random walks (say by adding a normally distributed increment with mean 0 and standard deviation 0.01 to all the $$q_*(a)$$ on each step). Prepare plots like Figure 2.2 for an action-value method using sample averages, incrementally computed, and another action-value method using a constant step-size parameter, $$\alpha = 0.1$$. Use $$\epsilon = 0.1$$ and longer runs, say of 10,000 steps. 

See code and figures below for answer
"""

# ╔═╡ fbe3dc15-9171-4a7e-8eea-b8cc052c9ba5
function nonstationary_algorithm(k::Integer, ϵ::AbstractFloat; steps = 10000, σ = 0.01, α = 0.0, initR = 0.0)
    qs = initR .* ones(k)
    Q = initR .* ones(k)
    N = zeros(k)
    accum_reward = 0.0
    step_reward = zeros(steps)
    accum_reward_ideal = 0.0
    step_reward_ideal = zeros(steps)
    cum_reward_ideal = zeros(steps)
    cum_reward = zeros(steps)
    optimalcount = 0
    optimalaction_pct = zeros(steps)
    optimalstep = fill(false, steps)
    actions = collect(1:k)
    for i = 1:steps
        shuffle!(actions) #so that ties are broken randomly with argmax
        a = if rand() < ϵ
            rand(actions)
        else
            actions[argmax(Q[actions])]
        end
        optimalaction = argmax(qs)
        if a == optimalaction
            optimalcount += 1
            optimalstep[i] = true
        end
        bandit(a) = sample_bandit(a, qs)
        step_reward[i] = bandit(a)
        step_reward_ideal[i] = bandit(optimalaction)
        accum_reward_ideal += step_reward_ideal[i]
        accum_reward += step_reward[i] 
        cum_reward_ideal[i] = accum_reward_ideal
        cum_reward[i] = accum_reward
        optimalaction_pct[i] = optimalcount / i
        N[a] += 1.0
        if α == 0.0
            Q[a] += (1.0/N[a])*(step_reward[i] - Q[a])
        else 
            Q[a] += α*(step_reward[i] - Q[a])
        end
        qs .+= randn(k) .*σ #update q values with random walk
    end
    return (;Q, step_reward, optimalstep, step_reward_ideal, cum_reward, cum_reward_ideal, optimalaction_pct)
end

# ╔═╡ 647ab36b-641e-4024-ad2d-40ff33be28f4
function average_nonstationary_runs(k, ϵ, α; n = 2000, kwargs...)
    names = (:step_reward, :optimalstep, :step_reward_ideal, :cum_reward, :cum_reward_ideal, :optimalaction_pct)
	run1 = nonstationary_algorithm(k, ϵ; α = α, kwargs...)
	runs = Vector{Vector{Vector{Float32}}}(undef, n)
	runs[1] = [Float32.(run1[a]) for a in names]
	for i in 2:n
		runs[i] = deepcopy(runs[1])
	end
    @threads for i in 2:n
        run = nonstationary_algorithm(k, ϵ; α = α, kwargs...) 
		for (j, a) in enumerate(names)
			runs[i][j] .= Float32.(run[a])
		end
    end

	for j in eachindex(names)
		for i in 2:n
			runs[1][j] .+= runs[i][j]
		end
		runs[1][j] ./= n
	end

	return runs[1]
    # map(i -> mapreduce(a -> a[i], (a, b) -> a .+ b, runs)./n, eachindex(names))
end

# ╔═╡ 276779a3-9332-46bd-b511-a33a2fea4b5f
@bind nonstationaryparams confirm(PlutoUI.combine() do Child
	md"""
	### Figure Parameters for Exercise 2.5
	Action Count: $(Child(:n, NumberField(1:100, default = 10)))
	
	Exploration Parameter ϵ: $(Child(:ϵ, NumberField(0.01:0.01:0.1, default = 0.1)))
	
	Constant Step-Size Parameter α: $(Child(:α, NumberField(0.01:0.01:0.1, default = 0.1)))
	
	Initial Reward Value: $(Child(:initR, NumberField(-10.0:0.1:10.0, default = 0.0)))

	Reward Drift Standard Deviation: $(Child(:σ, NumberField(0.001:0.001:0.1, default = 0.01)))
	"""
end)

# ╔═╡ 7748ab8a-d186-49a4-b6ab-d1bd9ea34990
md"""
### Bandit Arm Reward Distributions
"""

# ╔═╡ bb16115a-a2d9-4b8d-9937-96ab1cdd1ce2
function nonstationary_bandit_statistics(k::Integer; steps = 2_000, σ = 0.01, initR = 0.0, nruns = 1000)
    function run()
		qs = initR .* ones(k)
	    q_history = [[q] for q in qs]
	    for i = 1:steps
	        qs .+= randn(k) .*σ #update q values with random walk
			sort!(qs, rev = true)
			for i in eachindex(qs) push!(q_history[i], qs[i]) end
	    end
		return q_history
	end

	runs = [run() for _ in 1:nruns]
	q_avg = [sum(run[i] for run in runs) ./ nruns for i in 1:k]
    traces = [scatter(x = 1:steps, y = q_avg[i], name = "Arm Rank $i", showlegend = false) for i in 1:k]
	benchtrace = scatter(x = 1:steps, y = σ .* sqrt.(1:steps), name = "Expected Reward STD", line = attr(dash = "dot", width = 5, color = "rgba(100, 100, 100, 0.9)"))
	PlutoPlotly.plot([traces; benchtrace], Layout(xaxis_title = "Step", yaxis_title = "Mean Reward", legend = attr(orientation = "h", y = 1.), title = "Expected Value for Arm Mean Rewards Ranked from 1 to $k", height = 500))
end

# ╔═╡ 9b625fc0-89bd-4064-a379-225e6a940af7
md"""
Number of Arms: $(@bind ktest_nonstationary NumberField(2:100, default = 10))
"""

# ╔═╡ 8584ece7-badc-486a-9a57-b60e77f92673
nonstationary_bandit_statistics(ktest_nonstationary)

# ╔═╡ f9e60b35-84d2-4b4b-8832-6b0f08152396
md"""
Shows the reward mean for the arms at each ranking from 1 to $ktest_nonstationary.  Each arm starts at 0 mean reward and is perturbed by a normal random variable with σ = 0.01 at each step.  The rewards for an arm at a particular ranking seem to track the standard deviation of the overall distribution for the drift process.  The functional form of this is $$\sqrt{n}$$ where $$n$$ is the number of steps so far.  Depending on the number of arms, the multiplicative factor on the curve changes but for the 10 armed case, the second best arm seems to match the value for 1 standard deviation above the mean of 0.
"""

# ╔═╡ 32bff269-e893-4907-b589-7ba2ae1314bd
md"""
## 2.6 Optmisitic Initial Values

The averaging methods discussed above have some bias towards the initial value of the estimates.  We can exploit this by initializing Q with a value much higher than we'd expect to receive as a reward from any action.  That way every observed reward at first will be dissappointing thus encouraging the agent to try unvisited actions.  See below an example whose performance can be observed under different conditions.
"""

# ╔═╡ 6292f449-8720-41f1-84de-1865fb5fddbf
function figure_2_3(;k = 10, ϵ = 0.1, Qinit = 5.0, α=0.1)
	optimistic_greedy_runs = average_simple_runs(k, 0.0, Qinit = Qinit, α = α)
	realistic_ϵ_runs = average_simple_runs(k, ϵ, α = α)
	steps = 1:1000
	t1 = scatter(x = steps, y = optimistic_greedy_runs[3], name = "Optimistic, greedy, Qinit = $Qinit")
	t2 = scatter(x = steps, y = realistic_ϵ_runs[3], name = "Realistic, ϵ-greedy,  Qinit = 0.0. ϵ = $ϵ")
	PlutoPlotly.plot([t1, t2], Layout(xaxis_title = "Step", yaxis = attr(title = "% Runs Taking Optimal Action", range = [0, 1]), legend = attr(orientation = "h", y = 1.1), width = 700, height = 400, title = "$k Armed Testbed, α = $α", hovermode = "x unified"))
end

# ╔═╡ 70e40b75-e7d8-4009-af80-3bf4086a28df
@bind params_2_3 confirm(PlutoUI.combine() do Child
	md"""
	### Figure 2.3 Parameters
	Default values match textbook 
	
	Number of Actions: $(Child(:k, NumberField(1:100, default = 10)))
	Initial Optimistic Estimate: $(Child(:Qinit, NumberField(0.0:0.1:10.0, default = 5.0)))
	
	Exploration Parameter ϵ: $(Child(:ϵ, NumberField(0.01:0.01:1.0, default = 0.1)))
	Constant Step Size α: $(Child(:α, NumberField(0.01:0.01:1.0, default = 0.1)))
	"""
end)

# ╔═╡ b3ec4673-af63-4d4a-8314-fa7e594f8a37
figure_2_3(;params_2_3...)

# ╔═╡ d4ce45ae-613e-41ee-b626-69b0dbcf6452
md"""
> ### *Exercise 2.6: Mysterious Spikes* 
> The results shown in Figure 2.3 should be quite reliable because they are averages over 2000 individual, randomly chosen 10-armed bandit tasks.  Why, then, are there oscillations and spikes in the early part of the curve for the optimistic method? In other words, what might make this method perform particularly better or worse, on average, on particular early steps?

The spike occurs on step 11.  Due to the initial Q values it is almost 100% likely that a given run will sample each of the 10 possible actions once before repeating any.  For this not to be the case, one of the samples would have to exceed the initial value of 5.0 which has a probability near zero since the expected q value for the best arm is around 1.55 which unit variance.  That would mean that at any given step only 10% of the runs would select the optimal action and indeed for the first 10 steps about 10% of the runs are selecting the optimal action as we'd expect from random chance.  

On the 11th step, the Q value estimate for each action is $$(0.9 \times 5) + (0.1 \times action\_reward)$$.  The optimal action will be selected on this step as long as the reward produced by the best action exceeded all the others.  Empirically, that probability is ~44% which is similar to the probability calculated for the expected value of the best action of ~1.55 exceeding the rewards from the other 9 arms.  For those 44% of the runs that do select the optimal action, they will obtain a reward with expected value 1.55.  If they received that reward during both samples, then the Q value estimate will be $$0.9 \times ((0.9 \times 5) + (0.1 \times 1.55)) + (0.1 \times 1.55) \approx 4.34$$.  Let's consider the second best arm which has an expected q value of ~1.  The updated estimate for that arm after receiving a reward equal to the expected value is $$0.9 \times 0.5 + 0.1 \times 1 \approx 4.6$$.  Following the same reasoning for the third best arm, the value is about 4.57.  In fact even a reward of zero will produce an estimate of $$0.9 \times 5 = 4.5$$ which still exceeds the estimate for the optimal action in our scenario.  That explains why the percentage of optimal actions drops in the 12th step because it is expected that the estimate of the action selected on step 11 will drop below at least one of the other arms, thus changing the maximizing action selection to a worse one.
"""

# ╔═╡ cb93c588-3dfa-45f4-9d83-f2de26cb1cea
md"""
> ### *Exercise 2.7: Unbiased Constant-Step-Size Tick* 
> In most of this chapter we have used sample averages to estimate action values because sample averages do not produce the initial bias that constant step sizes do (see analysis leading to (2.6)).  However, sample averages are not a completely satisfactory solution because they may perform poorly on nonstationary problems.  Is it possible to avoid the bias of constant sample sizes while retaining their advantages on nonstationary problems?  One way is to use a step size of $$\beta_n \dot= \alpha / \bar{o}_n,$$ 
>to process the nth reward for a particular action, where $$\alpha>0$$ is a conventional constant step size, and $$\bar{o}_n$$ is a trace of one that starts at 0:
> $$\bar{o}_n \dot= \bar{o}_{n-1}+\alpha(1-\bar{o}_{n-1}), \text{ for } n \geq 0, \text{ with } \bar{o}_0 \dot= 0.$$
> Carry out an analysis like that in (2.6) to show that $$Q_n$$ is an exponential recency-weighted average *without initial bias*.
"""

# ╔═╡ c695b7f9-76ca-419b-924d-8338a42c8188
md"""
$$Q_{n+1} = Q_n + \beta_n[R_n - Q_n]$$ where $$\beta_n \dot= \alpha / \bar{o}_n $$ and $$\bar{o}_n \dot= \bar{o}_{n-1}+\alpha(1-\bar{o}_{n-1})$$

$$\bar{o}_n = \bar{o}_{n-1} + \alpha(1-\bar{o}_{n-1})=\bar{o}_{n-1}(1-\alpha)+\alpha$$

We can expand $\bar{o}_n$ backwards to get an explicit formula.

$$\begin{flalign}
\bar{o}_n&=\bar{o}_{n-1}(1-\alpha)+\alpha\\
&=(\bar{o}_{n-2}(1-\alpha) + \alpha)(1-\alpha)+\alpha\\
&=\bar{o}_{n-2}(1-\alpha)^2 + \alpha((1-\alpha)+1)\\
&=(\bar{o}_{n-3}(1-\alpha)+\alpha)(1-\alpha)^2 + \alpha((1-\alpha)+1)\\
&=\bar{o}_{n-3}(1-\alpha)^3+\alpha((1-\alpha^2) + (1-\alpha)+1)\\
&\vdots \\
&=\bar{o}_0(1-\alpha)^n + \alpha\sum_{i=0}^{n-1}(1-\alpha)^i=\alpha\sum_{i=0}^{n-1}(1-\alpha)^i
\end{flalign}$$

This sum has an explicit formula as can be seen by:

$$\begin{flalign}
S &= 1 + (1-\alpha) + (1-\alpha)^2 + \cdots + (1-\alpha)^{n-1} \\
S(1-\alpha) &= (1-\alpha)+\cdots+(1-\alpha)^n=S-1+(1-\alpha)^n \\
-S\alpha &=-1+(1-\alpha)^n \\
S&=\frac{1-(1-\alpha)^n}{\alpha} \\
\end{flalign}$$

Therefore, $$\bar{o}_n=\alpha\frac{1-(1-\alpha)^n}{\alpha}=1-(1-\alpha)^n$$, and since $$0<\alpha<1$$, then $$(1 - \alpha)^n \rightarrow 0 \text{ as } n \rightarrow \infty.$$

$$\beta_n=\frac{\alpha}{\bar{o}_n}=\frac{\alpha}{1-(1-\alpha)^n} \implies \beta_1=1$$

From exercise 2.4, we have the formula for $$Q_n$$ with a non-constant coefficient $$\alpha_n$$ which we can trivially replace here with $$\beta_n$$

$$Q_n=Q_1\prod_{i=1}^n \left( 1-\beta_i \right)+\sum_{i=1}^{n} \left[ R_i\beta_i\prod_{j=i+1}^n(1-\beta_j) \right]$$

Since $$\beta_1=1$$, the product associated with $$Q_1$$ will be 0.  Since there is no dependency on the initial value of Q, we can say this formula for updating Q has *no initial bias*.
If we then make the substitution $$\beta_n=\frac{\alpha}{1-(1-\alpha)^n}$$, we have

$$\begin{flalign}
Q_n&=\sum_{i=1}^{n} \left[ R_i\frac{\alpha}{1-(1-\alpha)^i}\prod_{j=i+1}^n \left( 1-\frac{\alpha}{1-(1-\alpha)^j} \right) \right] \\
&=\alpha\sum_{i=1}^n \left[ \frac{R_i}{1-(1-\alpha)^i}\prod_{j=i+1}^n \left( \frac{(1-\alpha)(1-(1-\alpha)^{j-1})}{1-(1-\alpha)^j} \right) \right] \\
&=\alpha\sum_{i=1}^n \left[ \frac{R_i(1-\alpha)^{n-i}}{1-(1-\alpha)^i}\prod_{j=i+1}^n \left( \frac{1-(1-\alpha)^{j-1}}{1-(1-\alpha)^j} \right) \right]
\end{flalign}$$

Examining the product term on its own, we can see it simplifies.

$$\prod_{j=i+1}^n \left( \frac{1-(1-\alpha)^{j-1}}{1-(1-\alpha)^j} \right)$$
$$\frac{1-(1-\alpha)^{i}}{1-(1-\alpha)^{i+1}}\frac{1-(1-\alpha)^{i+1}}{1-(1-\alpha)^{i+2}}\cdots\frac{1-(1-\alpha)^{n-1}}{1-(1-\alpha)^{n}}=\frac{1-(1-\alpha)^i}{1-(1-\alpha)^n} \text{ for i≤n}$$

Replacing this expression for the product in the expression for $$Q_n$$ we have:

$$Q_n=\alpha\sum_{i=1}^n \left[ \frac{R_i(1-\alpha)^{n-i}}{1-(1-\alpha)^i}\frac{1-(1-\alpha)^i}{1-(1-\alpha)^n}\right]=\frac{\alpha}{1-(1-\alpha)^n}\sum_{i=1}^n R_i(1-\alpha)^{n-i}$$

If we expand this sum going backwards from $$i=n$$:

$$Q_n=\frac{\alpha}{1-(1-\alpha)^n} \left[ R_n+R_{n-1}(1-\alpha)+R_{n-2}(1-\alpha)^2+\cdots+R_1(1-\alpha)^{n-1} \right]$$

The constant term starts off at $$1$$ for $$n=1$$ and approaches $$\alpha$$ in the limit of $$n \rightarrow \infty$$.  If $$0<\alpha<1$$, then the coefficients in the sum section for $$R_i$$ decrease exponentially from 1 for $$i=n$$ to $$(1-\alpha)^{n-1}$$ for $$i=1.$$  So the average over rewards includes every reward back to $$R_1$$ like the simple average but the coefficients become exponentially smaller approaching 0 as $$n \rightarrow \infty$$.   
"""

# ╔═╡ 23b99305-c8d9-4129-85fb-a5e4aabc4a31
md"""
## 2.7 Upper-Confidence-Bound Action Selection

We can choose to explore non-greedy actions based on the probability that they are better than optimal.  This probability always exists due to the uncertainty inherent in our value estimates.  One way to implement this concept is to select actions according to:

$$A_t \dot = \operatorname*{argmax}_a \left [ Q_t(a) + c \sqrt{\frac{\ln{t}}{N_t(a)}}\right ] \tag{2.10}$$

where $$N_t(a)$$ is the number of times that action $a$ has been selected prior to time t and $$c>0$$ controls the degree of exploration.  This idea is called *upper confidence bound* (UCB) action selection.  Note that is $$N_t(a)=0$$ then that action will be selected or a random selection will be made among all actions with zero counts.  See below for a comparison between the ϵ-greedy exploration method and the UCB method with the ability to change parameters for both methods.
"""

# ╔═╡ 8fac4109-2e0d-4366-9118-018221e0b910
function figure_2_4(;k = 10, c = 2.0, ϵ = 0.1)
	Random.seed!(1234) #ensure both techniques are using the same set of random bandits
	ucb_runs = average_simple_runs(k, 0.0, c = c)
	Random.seed!(1234)
	ϵ_runs = average_simple_runs(k, 0.1)
	steps = 1:1000
	t1 = scatter(x = steps, y = ucb_runs[1], name = "UCB c = $c")
	t2 = scatter(x = steps, y = ϵ_runs[1], name = "ϵ-greedy ϵ = $ϵ")
	t3 = scatter(x = steps, y = ϵ_runs[2], name = "theoretical limit")
	PlutoPlotly.plot([t1, t2, t3], Layout(xaxis_title = "Step", yaxis_title = "Reward per Step Averaged Over Runs", title = "$k armed testbed average performance", legend = attr(orientation = "h", y = 1.1), hovermode = "x unified"))
end

# ╔═╡ 093f312b-d70d-4bf7-bd53-8a1c7b2bee31
@bind params_2_4 confirm(PlutoUI.combine() do Child
	md"""
	### Figure 2.4 Parameters
	Number of Actions: $(Child(:k, NumberField(1:100, default = 10)))
	Confidence Bound c: $(Child(:c, NumberField(0.0:0.1:5.0, default = 2.0)))
	Exploration parameter ϵ: $(Child(:ϵ, NumberField(0.01:0.01:1.0, default = 0.1)))
	"""
end)

# ╔═╡ 0de99ee5-d94d-4d07-8cef-a6f9caf5e742
figure_2_4(;params_2_4...)

# ╔═╡ f88029d6-3fc2-4552-8441-5ef37ac42638
md"""
> ### *Exercise 2.8: UCB Spikes* 
> In Figure 2.4 the UCB algorithm shows a distinct spike in performance on the 11th step.  Why is this?  Note that for your answer to be fully satisfactory it must explain both why the reward increases on the 11th step and why it decreases on the subsequent steps.  Hint: If $$c=1,$$ then the spike is less prominent.

By definition, actions with zero visits are always considered maximizing.  Therefore, for the first 10 steps, all 10 unique actions will be sampled once with each Q estimate updating from a single sample.  On the 11th step, the exploration incentive for each action will be equal, so the action with the highest Q estimate will be selected.  This is most likely to be the action with the highest $$q^*$$ value but there is a substantial probability it is the second best action and diminishing probabilities for the remaining actions ranked by true $$q^*$$.  It is on this step though that we expect the selection to be substantially better than random chance although it is only using a single sample to validate the estimates.  On step 12, that improved action will now have a visit count of 2 instead of 1 for every other action.  In the calculation, the exploration bonus for that action will be $$c\sqrt{\frac{\ln{12}}{2}}\approx 1.11465 \times c.$$  Every other action will have an exploration bonus of $$\approx 1.576 \times c.$$  In order for the 2 visit action to be considered maximizing after this it must have a Q estimate that is $$\approx 0.4617 \times c$$ greater than any other action value estimate. In particular for $$c = 2.0,$$ the estimate must be $$\approx 0.9234$$ greater than the others.  Since the q's are normally distributed, the difference in expected value between the best and second best action is only about 0.55.  As a rough heuristic for the probability of the action selection remaining unchanged, we can consider the probability that the 2 sample best action estimate exceeds the single sample second best action estimate by 0.9234 with the following calculation: $$1 - \operatorname*{cdf}(\operatorname*{Normal}(\mu = 1.54 - 1, \sigma = \sqrt{\frac{3}{2}}), x = 0.9234) \approx 0.377.$$  Therefore, on step 12, the average run will change the action selection to something less optimal.  The larger the value of c, the more likely the selection is to change due to the larger weight placed on exploration.  Empirically, as c approaches $$\infty$$ the expected reward on step 12 approaches 0.744 vs 1.145 on step 11.  That compares to the expected q value for the top 3 actions of approximately 0.656, 1, and 1.54.  
"""

# ╔═╡ b24a92fc-f6c6-44e4-9afc-fa4249e4ab83
md"""
## 2.8 Gradient Bandit Algorithms

As an alternative to estimating action values, we can attempt to learn a numerical *preference* for each action $a$ which we will denote $$H_t(a) \in \mathbf{R}.$$  This vector of preferences will be converted in a probability distribution using the *soft-max distribution*.

$$\Pr\{A_t = a\} \dot = \frac{e^{H_t(a)}}{\sum_{b=1}^k e^{H_t(b)}} \dot = \pi_t(a) \tag{2.11}$$

 $$\pi_t(a)$$ is the probability for this agent to select action $$a$$ at time $$t.$$  All action preferences are initialized at the same value.
"""

# ╔═╡ 9d7782f5-b530-40d5-9f75-280d3a762216
md"""
> ### *Exercise 2.9* 
> Show that in the case of two actions, the soft-max distribution is the same as that given by the logistic, or sigmoid, function often used in statistics and artificial neural networks.

The sigmoid function is defined as: $$S(x) = \frac{1}{1 + e^{-x}}.$$  For two actions, let's denote them $$a_1$$ and $$a_2.$$  Now for the action probabilities we have.

$$\pi(a_1) = \frac{e^{H_t(a_1)}}{e^{H_t(a_1)} + e^{H_t(a_2)}}=\frac{1}{1+e^{-(H_t(a_1) - H_t(a_2))}}$$

This expression for $$\pi(a_1)$$ is equivalent to $$S(x)$$ with $$x = H_t(a_1) - H_t(a_2)$$ which is the degree of preference for action 1 over action 2.  As expected, if the preferences are equal then it is equavalent to $$x=0$$ with a probability of 50%.  The same analysis applies to action 2 with the actions reversed from this case.
"""

# ╔═╡ 0f244fa0-7591-4478-b172-d9c1de51f6e1
md"""
One natural update rule for the action preferences is to use stochastic gradient ascent.  Using this technique we perform the following update on step $$t+1$$ after selecting action $$A_t$$ and receiving reward $$R_t$$ on step $$t$$.

$$\begin{flalign}
H_{t+1}(A_t) &\dot = H_t(A_t) + \alpha (R_t - \overline R_t)(1-\pi_t (A_t)) \\
H_{t+1}(a) & \dot = H_t(a) - \alpha(R_t - \overline R_t)\pi_t(a) \quad \forall a \neq A_t
\end{flalign} \tag{2.12}$$

where $$\alpha > 0$$ is a step-size parameter and $$\overline R_t \in \mathbf{R}$$ is the average rewards up to but not including time $$t.$$  This average can be computed by any of the techniques mentioned earlier.
"""

# ╔═╡ 99945570-0b2a-432a-ae83-3a7abc39cac8
function normalize_πvec!(πvec::Vector{T}) where T <: AbstractFloat
	s = sum(πvec)
	if (isinf(s) || isnan(s) || (s == 0))
		πvec .= one(T) / length(πvec)
	else
		πvec ./= s
	end
end

# ╔═╡ 54deaa09-8f87-4caf-b2a0-f15bcd5b40a5
#calculates a vector of probabilities for selecting each action given exponentiated "perferences" given in expH using the softmax distribution
function update_πvec!(πvec, H::AbstractVector)
	πvec .= exp.(H)
	normalize_πvec!(πvec)
end

# ╔═╡ 24759e26-e670-4330-b6e4-b313620660f1
function calculate_πvec(H::AbstractVector)
	πvec = exp.(H)
	normalize_πvec!(πvec)
end

# ╔═╡ ea6d7cad-47ad-4472-a9e9-1ee33c81058d
function update_H!(a::Integer, H::AbstractVector, π_vec::AbstractVector, α, R, R̄)
	v = α*(R - R̄)
	H .-= v .* π_vec
	H[a] += v
end

# ╔═╡ 51349e41-4696-4bd5-9bc1-cefbb82bea08
sample_action(actions, π_vec) = sample(actions, pweights(π_vec))

# ╔═╡ b5a2df21-4525-4320-b8dd-aea5ecdab832
@bind params_2_5 confirm(PlutoUI.combine() do Child
	md"""
	### Figure 2.5 Parameters
	Number of Actions: $(Child(:k, NumberField(1:100, default = 10)))
	Reward Offset: $(Child(:offset, NumberField(0.0:1.0:10.0, default = 4.0)))

	Step-Size Minimum With Baseline: $(Child(:αmin1, NumberField(0.001:0.001:0.1, default = 0.025)))
	
	Step-Size Minimum Without Baseline: $(Child(:αmin2, NumberField(0.001:0.0001:0.1, default = 0.0125)))
	"""
end)

# ╔═╡ 649e3d20-e276-4f4b-aeb0-89150f180ef5
md"""
## 2.9 Associate Search (Contextual Bandits)
"""

# ╔═╡ 1f9a98fd-ea29-415c-9f35-add34b513a34
md"""
> ### *Exercise 2.10* 
> Suppose you face a 2-armed bandit task whose true action values change randomly from time step to time step. Specifically, suppose that, for any time step, the true values of actions 1 and 2 are respectively 10 and 20 with probability 0.5 (case A), and 90 and 80 with probability 0.5 (case B). If you are not able to tell which case you face at any step, what is the best expected reward you can achieve and how should you behave to achieve it? Now suppose that on each step you are told whether you are facing case A or case B (although you still don’t know the true action values). This is an associative search task. What is the best expected reward you can achieve in this task, and how should you behave to achieve it?

When we do not know which case we are facing, we can calculate the expected reward for each action across all cases.

$$E[R_1] = 0.5 \times 10 + 0.5 \times 90 = 50$$

$$E[R_2] = 0.5 \times 20 + 0.5 \times 80 = 50$$

Since the expected reward of each action is equal, the best we can do is pick randomly which will have an expected reward of 50.

For the case in which we know if we are in case A or case B, we now can select the best action for each case which has a value of 20 (action 2) for case A and 90 (action 1) for case B.  However, we have a 50% probability of facing each case so the best achievable expected reward is.

$$E[R] = 20 \times 0.5 + 90 \times 0.5 = 55$$

To acheive this reward we could apply the action value estimate approach but separate our samples for case A and B.  That way we would have 4 estimates representing the expected reward of each action in each case.  We could perform any of the exploration strategies mentioned earlier such as ϵ-greedy action selection but being careful to update the estimate for that case only.
"""

# ╔═╡ 1c9b54cd-08dd-401e-9705-818741844e8d
md"""
## Code Refactoring
Due to the variety of algorithms and parameters for the bandit, I have rewritten the test environment with types that represent the different algorithms.  The run simulator will dispatch on the types to correctly simulate that method with its parameters.  Some of the previous simluations and plots are generated again.  Because of the style used, only one simulation function is needed with the flexibility to select any combination of techniques in the chapter.
	"""

# ╔═╡ c2347999-5ade-420b-903f-30523b38eb0f
abstract type Explorer{T <: AbstractFloat} end

# ╔═╡ f995d0af-50bc-4e33-9bbf-17a7ab06358a
struct ϵ_Greedy{T <: AbstractFloat} <: Explorer{T}
	ϵ::T
end

# ╔═╡ 852df31d-18d8-466c-8225-e06ba7f05e96
#creates additional constructor with default value
(::Type{ϵ_Greedy})() = ϵ_Greedy(0.1)

# ╔═╡ 44b9ff95-ea3d-41f5-8098-445a263738a9
#extends default value to other types
(::Type{ϵ_Greedy{T}})() where T<:AbstractFloat = ϵ_Greedy(T(0.1))

# ╔═╡ ca726a9d-364d-48e2-8882-20ddbc85b664
(::Type{ϵ_Greedy{T}})(e::ϵ_Greedy) where T<:AbstractFloat = ϵ_Greedy(T(e.ϵ))

# ╔═╡ 9d36934a-78cb-446b-b3db-1bbd88cf272d
#how to convert type of explorer when it is already the same
(::Type{T})(e::T) where T <: ϵ_Greedy = e

# ╔═╡ 60b2079e-0efa-427e-93cf-7f4646fe202e
struct UCB{T <: AbstractFloat} <: Explorer{T}
	c::T
end

# ╔═╡ 3118e102-aeac-42d9-98fc-ca29f40be4cd
(::Type{UCB})() = UCB(2.0)

# ╔═╡ 8c0f06f7-2ed0-4f3a-ab4e-90ac142f0cd9
(::Type{UCB{T}})() where T <: AbstractFloat = UCB(T(2.0))

# ╔═╡ a61c15eb-ed5f-4052-a3a3-3276940564a1
(::Type{T})(e::T) where T <: UCB = e

# ╔═╡ 47be3ae6-20f7-47d0-aae3-b67154afc1a8
(::Type{UCB{T}})(e::UCB) where {T<:AbstractFloat} = UCB(T(e.c))

# ╔═╡ 4c6ccbfe-a3ce-4f2d-bcb1-7f1a4b735c65
struct GradientSample{T <: AbstractFloat} <: Explorer{T} end

# ╔═╡ e50596ab-91db-42f0-a62c-77629a4e79c7
(::Type{T})(e::T) where T <: GradientSample = e

# ╔═╡ 839861db-676f-4544-a802-0abb5d0049e1
abstract type AverageMethod{T<:AbstractFloat} end

# ╔═╡ 1ac5588c-3c32-436e-8b40-41715223fba7
struct ConstantStep{T<:AbstractFloat} <: AverageMethod{T}
	α::T
end

# ╔═╡ f00ab44e-0d84-40c7-aa23-358c77a013e3
mutable struct UnbiasedConstantStep{T<:AbstractFloat}<:AverageMethod{T}
	α::T
	o::T
	UnbiasedConstantStep(α::T) where T<:AbstractFloat = new{T}(α, zero(T))
end

# ╔═╡ 262952a7-280e-4af1-99a6-0899518484a2
(::Type{ConstantStep})() = ConstantStep(0.1)

# ╔═╡ fbf7b108-dc68-4077-b63c-6f88161d2098
(::Type{ConstantStep{T}})() where T<:AbstractFloat = ConstantStep(T(0.1))

# ╔═╡ e2597cc6-a6f6-4260-887b-c587cacd3bc8
(::Type{U})(a::U) where U <: ConstantStep = a

# ╔═╡ 3b9bb9f0-ba9d-4320-935a-58912afe34b6
(::Type{ConstantStep{T}})(a::ConstantStep) where {T<:AbstractFloat} = ConstantStep{T}(a.α) 

# ╔═╡ 0f6b4e2d-dc09-4e1c-834f-dd8aaa8743ae
(::Type{UnbiasedConstantStep})() = UnbiasedConstantStep(0.1)

# ╔═╡ 7a31d1c5-260b-42f4-b997-967150881e21
(::Type{UnbiasedConstantStep{T}})(a::UnbiasedConstantStep) where T<:AbstractFloat=UnbiasedConstantStep(T(a.α))

# ╔═╡ 5712b303-0aa3-4501-b1b5-020136d6e655
(::Type{U})(a::U) where U<:UnbiasedConstantStep = a

# ╔═╡ f561b0a8-a086-4e1a-bc87-82c4205e89c9
struct SampleAverage{T<:AbstractFloat} <: AverageMethod{T} end

# ╔═╡ 30aa1e1b-0b51-40c4-a093-ef92c3ad519a
(::Type{SampleAverage})() = SampleAverage{Float64}()

# ╔═╡ 0e606680-dd65-444f-bc98-73de4abbcdd4
(::Type{U})(a::U) where U <: SampleAverage = a

# ╔═╡ 4ebd4a5a-3bd1-48e2-b03e-a5a3b2ec18a1
(::Type{U})(a::SampleAverage) where U <: SampleAverage = U() 

# ╔═╡ 73e5b719-8b91-41d2-b83d-471d981b027f
(::Type{Explorer{T}})(e::ϵ_Greedy) where T<:AbstractFloat = ϵ_Greedy{T}(e)

# ╔═╡ 49e45202-b9ae-42ab-9575-a57edb626a20
(::Type{Explorer{T}})(e::ϵ_Greedy{T}) where T<:AbstractFloat = e

# ╔═╡ 3a215c95-c595-4837-a842-1c1e1c6bfa3b
(::Type{Explorer{T}})(e::UCB{T}) where T<:AbstractFloat = e

# ╔═╡ a04da367-3f8d-422d-a443-4e3e666e30ef
(::Type{Explorer{T}})(e::UCB) where T<:AbstractFloat = UCB{T}(e)

# ╔═╡ 68470b1d-3cc2-4cb1-8dc2-53227e6300e7
(::Type{Explorer{T}})(e::GradientSample) where T<:AbstractFloat = GradientSample{T}()

# ╔═╡ fb3381b5-10e3-4307-b76d-672245fac9e7
(::Type{AverageMethod{T}})(a::SampleAverage) where T<:AbstractFloat = SampleAverage{T}()

# ╔═╡ ff6598fa-3366-416c-88a1-6bfcefeb1719
(::Type{AverageMethod{T}})(a::ConstantStep) where T<:AbstractFloat = ConstantStep{T}(a)

# ╔═╡ d2ebd908-387d-4e40-bc00-61ce5f45ebdd
(::Type{AverageMethod{T}})(a::UnbiasedConstantStep) where T<:AbstractFloat = UnbiasedConstantStep{T}(a)

# ╔═╡ 1e1f6d10-1b31-4e0a-96f7-23207e913154
abstract type BanditAlgorithm{T, E, A} end

# ╔═╡ 13f0adab-7660-49df-b26d-5f89cd73192b
struct ActionValue{T<:AbstractFloat, E <: Explorer{T}, A <: AverageMethod{T}} <: BanditAlgorithm{T, E, A}
	N::Vector{T}
	Q::Vector{T}
	explorer::E
	update_average::A
end

# ╔═╡ 45cc0a58-3534-4c67-bfd4-1c2b48d59a2e
function (::Type{ActionValue})(k::Integer, Qinit::T, explorer::Explorer, q_avg::AverageMethod) where T <: AbstractFloat 
	#use the type of Qinit to initialize vectors and convert the other types if necessary
	N = zeros(T, k)
	Q = ones(T, k) .* Qinit
	new_e = Explorer{T}(explorer)
	new_avg = AverageMethod{T}(q_avg)
	ActionValue(N, Q, new_e, new_avg)
end

# ╔═╡ 0fbbe455-79ce-44d6-b010-da0bb56adbb4
(::Type{ActionValue})(k::Integer; Qinit::T = 0.0, explorer::Explorer = ϵ_Greedy(), update_average::AverageMethod = SampleAverage()) where T <: AbstractFloat = ActionValue(k, Qinit, Explorer{T}(explorer), AverageMethod{T}(update_average))

# ╔═╡ c66f1676-aec1-489d-96ff-99d748dac0fe
(::Type{ActionValue})(;kwargs...) = k -> ActionValue(k; kwargs...)

# ╔═╡ 8a04adab-e97e-4ac4-a85e-5eae93b1c37b
mutable struct GradientReward{T<:AbstractFloat, E <: GradientSample{T}, A <: AverageMethod{T}} <: BanditAlgorithm{T, E, A}
	H::Vector{T} 
	πvec::Vector{T}
	α::T
	R̄::T
	update_average::A

	function GradientReward(H, πvec, α::T, R̄, update_average::A) where {T <: AbstractFloat, A <: AverageMethod{T}}
		new{T, GradientSample{T}, A}(H, πvec, α, R̄, update_average)
	end
end

# ╔═╡ 69b560c1-98ad-4cbf-89d2-e0516299bc69
function (::Type{GradientReward})(k::Integer; α::T=0.1, update_average::AverageMethod = SampleAverage()) where T <: AbstractFloat 
	H = zeros(T, k)
	πvec = ones(T, k) ./ k
	R̄ = zero(T)
	new_update = AverageMethod{T}(update_average)
	GradientReward(H, πvec, α, R̄, new_update)
end

# ╔═╡ d9265b98-cc3e-4a60-b16e-f54d9f78c9d3
(::Type{GradientReward})(;kwargs...) = k -> GradientReward(k; kwargs...)

# ╔═╡ 4982b489-ca15-4188-90e5-565c45f02e01
sample_action(est::GradientReward, i::Integer, actions) = sample(actions, weights(est.πvec))

# ╔═╡ 672a91c0-aa77-4257-8c83-d857f47cab6c
sample_action(est::ActionValue, i::Integer, actions::AbstractVector) = sample_action(est::ActionValue, est.explorer, i, actions)

# ╔═╡ 638f99e6-1cdc-414c-9b67-fd626ec0be3e
function sample_action(est::ActionValue, explorer::ϵ_Greedy, i::Integer, actions::AbstractVector)
	shuffle!(actions)
	# ϵ = explorer.ϵ
	if rand() < explorer.ϵ
		rand(actions)
	else
		actions[argmax(view(est.Q, actions))]
	end
end

# ╔═╡ aa238ebc-8730-46c7-8ad9-41c7cac70b18
function sample_action(est::ActionValue, explorer::UCB, i::Integer, actions::AbstractVector)
	(Q, N, c) = (est.Q, est.N, explorer.c)
	argmax(view(Q, actions) .+ c .* sqrt.(log(i) ./ view(N, actions)))
	# argmax(est.Q[actions] .+ (est.explorer.c .* sqrt.(log(i) ./ est.N[actions])))
end

# ╔═╡ f7519adc-7dfb-4030-86f0-7445699dd3db
function gradient_stationary_bandit_algorithm(qs::Vector{Float64}, k::Integer; steps = 1000, α = 0.1, baseline = true)
    bandit(a) = sample_bandit(a, qs)
    H = zeros(k)
	πvec = calculate_πvec(H)
	R̄ = 0.0
    accum_reward_ideal = 0.0
    accum_reward = 0.0
    cum_reward_ideal = zeros(steps)
    step_reward_ideal = zeros(steps)
    cum_reward = zeros(steps)
    step_reward = zeros(steps)
    bestaction = argmax(qs)
    optimalstep = fill(false, steps)
    optimalcount = 0
    optimalaction_pct = zeros(steps)
    actions = collect(1:k)
    for i = 1:steps
        a = sample_action(actions, πvec)
        if a == bestaction
            optimalstep[i] = true
            optimalcount += 1
        end
        step_reward[i] = bandit(a) 
        step_reward_ideal[i] = bandit(bestaction)
        accum_reward_ideal += step_reward_ideal[i] 
        cum_reward_ideal[i] = accum_reward_ideal
        accum_reward += step_reward[i] 
        cum_reward[i] = accum_reward
        optimalaction_pct[i] = optimalcount / i
		update_H!(a, H, πvec, α, step_reward[i], R̄)

		#update R̄ with running average if baseline is true
		if baseline
			R̄ += (1.0/i)*(step_reward[i] - R̄)
		end

		#update π_vec
		update_πvec!(πvec, H)
    end
    return (;step_reward, step_reward_ideal, cum_reward, cum_reward_ideal, optimalstep, optimalaction_pct)
end

# ╔═╡ 50fbdc85-82f1-4c52-936b-84eb14951d71
function average_gradient_stationary_runs(k; steps = 1000, n = 2000, α=0.1, offset = 0.0, baseline = true)
	names = (:step_reward, :step_reward_ideal, :optimalstep, :cum_reward, :cum_reward_ideal, :optimalaction_pct)
    runs = Vector{Vector{Vector{Float32}}}(undef, n)
	qs = create_bandit(k, offset = offset)
	run1 = gradient_stationary_bandit_algorithm(qs, k, steps = steps, α = α, baseline = baseline)
	runs[1] = [Float32.(run1[name]) for name in names]
    @threads for i in 2:n
        qs = create_bandit(k, offset = offset)
        run = gradient_stationary_bandit_algorithm(qs, k, steps = steps, α = α, baseline = baseline)
		runs[i] = [Float32.(run[name]) for name in names]
    end

	for i in eachindex(names)
		for j in 2:n
			runs[1][i] .+= runs[j][i]
		end
		runs[1][i] ./= n
	end
    return runs[1]
end

# ╔═╡ 8a6f3f85-64e4-4c31-9e69-43f50f42bbc9
function figure_2_5(;k = 10, offset = 4.0, αmin1 = 0.025, αmin2 = 0.025, α_list = [0.025, 0.05, 0.1, 0.2, 0.4])
	ylabel =  "% Runs Taking Optimal Action"
	steps = 1:1000

	function make_αlist(α::T, list = Vector{T}()) where T <: AbstractFloat 
		α > 0.4 && return list 
		make_αlist(2*α, push!(list, α))
	end
	
	function make_plot(baseline::Bool, α_list)
		results = [average_gradient_stationary_runs(k, α=α, offset = 4.0, baseline = baseline) for α in α_list]
		traces = [scatter(x = steps, y = a[3], name = "α = $(α_list[i])") for (i, a) in enumerate(results)]
		PlutoPlotly.plot(traces, Layout(xaxis_title = "Step", yaxis_title = ylabel, hovermode = "x unified", title = "Gradient Bandit $(baseline ? "With" : "Without") Baseline"))
	end

	md"""
	$(make_plot(true, make_αlist(αmin1)))
	$(make_plot(false, make_αlist(αmin2)))
	Average performance of the gradient bandit algorithm with and without a reward baseline on the $k-armed bandit testbed when the $$q_*(a)$$ are chosen to be near $offset rather than near 0.   For the gradient bandit with a baseline, the offset doesn't affect the curves at all, but if the baseline is removed then the results are worse as seen in the second plot. However, if $$\alpha$$ is made smaller it seems like it will also converge to a similar success rate just over a longer time. The optimal value of $$\alpha$$ is much lower than when the baseline is removed which is consistent with slower convergence properties.
	"""
end

# ╔═╡ 691aa77a-d6da-4fde-9024-c4195057179d
figure_2_5(;params_2_5...)

# ╔═╡ 62eb0650-96bc-4fd6-bfe0-bf05a4137a03
updatecoef(est::ActionValue{T, E, SampleAverage{T}}, a::Integer, step::Integer) where {T <: AbstractFloat, E <: Explorer{T}} = one(T) / est.N[a]

# ╔═╡ 30f05bd9-e939-4810-b710-edc7f5975921
updatecoef(est::GradientReward{T, GradientSample{T}, SampleAverage{T}}, a::Integer, step::Integer) where {T <: AbstractFloat} = one(T) / step

# ╔═╡ 96566aad-6d5c-460c-a924-ae0bad5d8b2d
updatecoef(est::BA, a::Integer, step::Integer) where {BA <: BanditAlgorithm{T, E, ConstantStep{T}} where {T <: AbstractFloat, E <: Explorer{T}}} = est.update_average.α

# ╔═╡ 4191ff98-f4ab-4f18-a148-d3d3fff3d0ad
function updatecoef(est::BA, a::Integer, step::Integer) where {BA <: BanditAlgorithm{T, E, UnbiasedConstantStep{T}} where {T <: AbstractFloat, E <: Explorer{T}}}
	avg = est.update_average
	avg.o += avg.α*(one(T) - avg.o)
	avg.α/avg.o
end

# ╔═╡ 09227386-1620-4093-b913-5786205aad13
function update_estimator!(est::GradientReward{T, E, A}, a::Integer, r::T, step::Integer) where {T <: AbstractFloat, E <: GradientSample{T}, A <: AverageMethod{T}}
	rdiff = r - est.R̄
	c = est.α * rdiff
	est.H .-= c .* est.πvec
	est.H[a] += c

	est.R̄ += updatecoef(est, a, step) * rdiff
	update_πvec!(est.πvec,est.H)
end

# ╔═╡ 781fa565-398c-4c89-89f3-0595455afc85
function update_estimator!(est::ActionValue{T, E, A}, a::Integer, r::T, step::Integer) where {T <: AbstractFloat, E <: Explorer{T}, A <: AverageMethod{T}} 	est.N[a] += one(T)	
	est.Q[a] += updatecoef(est, a, step) * (r - est.Q[a])
end

# ╔═╡ 1004eb4b-1fed-4328-a08b-6f5d9dd5080b
function run_bandit(qs::Vector{T}, algorithm::BanditAlgorithm{T}; steps = 1000, μ::T = zero(T), σ::T = zero(T), cumstart = 1, saveall = true) where T <: AbstractFloat
#if saveall is false, then only saves the average cumulative reward per step, so it is faster
	#in this case the bandit is not stationary
	updateq = (μ != 0) || (σ != 0)
	function qupdate!(qs)
		for i in eachindex(qs)
			qs[i] += (randn()*σ) + μ
		end
		return nothing
	end
	#initialize values to keep track of
	actions = collect(eachindex(qs))
	accum_reward_ideal = zero(T)
    accum_reward = zero(T)
	bestaction = argmax(qs)

	bandit(a) = sample_bandit(a, qs)

	if saveall
	    cum_reward_ideal = zeros(T, steps)
	    step_reward_ideal = zeros(T, steps)
	    cum_reward = zeros(T, steps)
	    step_reward = zeros(T, steps)
	    optimalstep = fill(false, steps)
	    optimalcount = 0
	    optimalaction_pct = zeros(T, steps)
	end
	
    for i = 1:steps
        a = sample_action(algorithm, i, actions)
		r = bandit(a)
		r_ideal = a == bestaction ? r : bandit(bestaction)

		if i >= cumstart
			accum_reward_ideal += r_ideal
			accum_reward += r
		end

		#update anything required before sampling the next action
		update_estimator!(algorithm, a, r, i)

		if saveall
			if a == bestaction
	            optimalstep[i] = true
	            optimalcount += 1
	        end
	        step_reward[i] = r 
	        step_reward_ideal[i] = r_ideal
	        cum_reward_ideal[i] = accum_reward_ideal
	        cum_reward[i] = accum_reward
	        optimalaction_pct[i] = optimalcount / i	
		end

		#will only update qs in the non-stationary case and get a new bestaction
		if updateq				
			qupdate!(qs)
			bestaction = argmax(qs)
		end
	
    end

	saveall && return (;step_reward, step_reward_ideal, cum_reward, cum_reward_ideal, optimalstep, optimalaction_pct)

	navg = steps - cumstart + 1
	step_cum_reward = accum_reward / navg
	step_cum_reward_ideal = accum_reward_ideal / navg
	return (;step_cum_reward, step_cum_reward_ideal)
end

# ╔═╡ 7678b06b-feef-4656-a801-33f630437bfb
function average_runs(k, algorithm::Function; offset::Float32 = 0.0f0, steps = 1000, n = 2000, make_bandit = create_bandit, kwargs...)
    names = (:step_reward, :step_reward_ideal, :optimalstep, :cum_reward, :cum_reward_ideal, :optimalaction_pct)
	runs = Vector{Vector{Vector{Float32}}}(undef, n)
	function loopbody()
		qs = make_bandit(k, offset=offset)
		est = algorithm(k)
        run = run_bandit(qs, est; steps = steps, kwargs...) 
		[Float32.(run[name]) for name in names]
	end
	runs[1] = loopbody()
    for i in 2:n
       runs[i] = loopbody()
    end

	for i in eachindex(names)
		for j in 2:n
			runs[1][i] .+= runs[j][i]
		end
		runs[1][i] ./= n
	end
	return runs[1]
end

# ╔═╡ 8a52acf7-5d57-490f-8bd9-6e1e0e322872
function average_runs_cum_reward(k, algorithm::Function; steps = 1000, n = 2000, offset::T = 0.0f0, make_bandit = create_bandit, kwargs...) where T <: AbstractFloat
    r_step = Atomic{T}(zero(T))
	r_step_ideal = Atomic{T}(zero(T))
	rdiff = Atomic{T}(zero(T))
	rdiff2 = Atomic{T}(zero(T))
	getvar(s, s2) = (s2 - (s*s/n))/(n-1)
    @threads for i in 1:n
		Random.seed!(i)
        qs = make_bandit(k, offset=offset)
		est = algorithm(k)
		Random.seed!(i)
        rewards = run_bandit(qs, est; steps = steps, saveall = false, kwargs...) 
    	atomic_add!(r_step, rewards[1])
		x = abs(rewards[2] - rewards[1])
		atomic_add!(rdiff, x)
		atomic_add!(rdiff2, x^2)
		atomic_add!(r_step_ideal, rewards[2])
	end
    (means = (r_step[]/n, r_step_ideal[]/n), rstd = r_step[]/n - sqrt(getvar(rdiff[], rdiff2[])))
end

# ╔═╡ 181d7eef-24a0-4775-a535-8ef901b7e4eb
average_nonstationary_runs(k, algorithm::Function; steps = 1_000, n = 2000, qinit::T=0.0f0) where T<:AbstractFloat = average_runs(k, algorithm; offset = qinit, steps = steps, n = n, make_bandit = (k; offset = qinit) -> ones(T, k) .* offset, σ = T(0.01))

# ╔═╡ 34f65898-cbcc-4832-afac-0f7a284e7f0b
function exercise2_5(; n=10, ϵ=0.1, α=0.1, kwargs...)
	sample_average_run = average_nonstationary_runs(n, ϵ, 0.0; kwargs...)
	constant_step_update_run = average_nonstationary_runs(n, ϵ, α; kwargs...)
	p1 = plot([scatter(y = sample_average_run[1], name = "Sample Average"), scatter(y = constant_step_update_run[1], name = L"\alpha = 0.1", hovertemplate = "%{y}<extra>α=0.1</extra>"), scatter(y=sample_average_run[3], name  = "Theoretical Limit")], Layout(yaxis_title = "Reward Averaged Over Runs", xaxis_title = "Step", width = 700, height = 400, legend_orientation = "h", legend_y = 1.1, hovermode = "x unified"))
	p2 = plot([scatter(y = sample_average_run[2], name = "Sample Average"), scatter(y = constant_step_update_run[2], name  = L"\alpha = 0.1", hovertemplate="%{y}<extra>α=0.1</extra>")], Layout(hovermode = "x", yaxis_title = "% Runs Taking Optimal Action", xaxis_title = "Step", width = 700, height = 400, legend_orientation = "h", legend_y = 1.1))
	md"""
	$p1
	$p2
	"""
end

# ╔═╡ d24bd737-9e09-441a-aa94-9279c80f566d
exercise2_5(;nonstationaryparams...)

# ╔═╡ 74024d96-d0c7-43c8-8379-caf843cbe4b8
average_nonstationary_runs_cum_reward(k, algorithm::Function; steps = 1_000, n = 2000, qinit::T=0.0f0) where T<:AbstractFloat = average_runs_cum_reward(k, algorithm; offset = qinit, steps = steps, n = n, make_bandit = (k; offset = qinit) -> ones(T, k) .* offset, σ = T(0.01), cumstart = floor(Int64, steps/2) + 1)

# ╔═╡ 46ce2b1f-02cf-4dae-bf02-f67543f38b91
md"""
## Parameter Studies
"""

# ╔═╡ c61630b0-29c1-4183-90d9-57c999187b53
function print_power2(n)
	if abs(n) > 7
		latexify("2^$n")
	elseif n < 0
		latexify("1/$(2^-n)")
	else
		latexify("$(2^n)")
	end
end

# ╔═╡ c72d35fc-c8fd-450e-9b95-d12ece5c2291
function get_param_list(n1::Integer, n2::Integer; base::T = 2.0f0) where T<:AbstractFloat
	nlist = collect(n1:n2)
	plist = base .^nlist	
	# namelist = print_power2.(nlist)
	return plist, nlist
end

# ╔═╡ 8f1c7b0d-121c-46bf-884d-729e3b593025
#parameter search with fixed powers of 2 range
function param_search(k, algorithm, n1, n2, f; base::T = 2.0f0, kwargs...) where T<:AbstractFloat
	plist, nlist = get_param_list(n1, n2, base = base)
	cum_rewards = Vector{T}(undef, length(plist))
	cum_rewards_ideal = similar(cum_rewards)
	[(param = p, ex = n, rewards = f(k, algorithm(p); kwargs...)) for (p, n) in zip(plist, nlist)]
end

# ╔═╡ a15e5d05-a238-440b-9a43-d830c6ea2f4d
#parameter search that automatically searches to find a maximum starting from a power of -1 for base 2
function param_search(k, algorithm, f; base::T = 2.0f0, exmin = -10, exmax = 5, kwargs...) where T<:AbstractFloat
	@info "Starting parameter search with algorithm: $algorithm"
	exmean = round(Int64, (exmin + exmax) / 2)
	function makerun(ex) 
		p = base^ex
		@info "Evaluating p = $base ^ $ex"
		return (param = p, ex = ex, rewards = f(k, algorithm(p); kwargs...))
	end
	
	function step(count, incr, ex, maxreward, runlist)
		(ex < exmin) && return runlist
		(ex > exmax) && return step(0, -1, -2, maxreward, runlist)
		if (count >= 3)
		#in this case we've generated 3 points to the right or left of the maximum which means either we change direction and reset left of the maximum or terminate
			incr == -1 && return runlist
			return step(0, -1, -2, maxreward, runlist)
		else
			run = makerun(ex)
			#mean reward per step
			r = run.rewards.means[1]

			#if we have a new maximum reset the count to 0 otherwise increment by 1
			if r > maxreward
				count = -1
				maxreward = r
			end
			return step(count+1, incr, ex+incr, maxreward, vcat(runlist, run))
		end
	end

	#initialize search starting from ex = -1
	firstrun = makerun(-1)
	firstreward = firstrun.rewards.means[1]
	step(0, 1, 0, firstreward, [firstrun])	
end

# ╔═╡ 22fa2b71-a98f-4b87-9e0b-9d373cd8915f
stationary_param_search(args...; kwargs...) = param_search(args..., average_runs_cum_reward; kwargs...)

# ╔═╡ 2447c4ea-7752-457c-80da-ac0dd72a64c1
save_data(varname, data) = jldsave("$varname.jld2"; data)	

# ╔═╡ 865610bb-ee82-4440-9f32-f00d0382783b
function run_or_load(varname::String, operation::Function)
	if !isfile("$varname.jld2")
		data = operation()
		jldsave("$varname.jld2"; data)
	else
		data = read(jldopen("$varname.jld2"), "data")
	end
	return data
end

# ╔═╡ 400e0de1-0101-4531-928f-08ca155da40c
#makes vectors suitable for plotting the different lines
function preparetrace(runlist)
	f(g) = [g(r) for r in runlist]
	plist = f(r -> r.param)
	nlist = f(r -> r.ex)
	inds = sortperm(plist)
	glist = [r -> r.rewards.means[1], r -> r.rewards.means[2], r -> r.rewards.rstd]
	yvecs = [f(g)[inds] for g in glist]
	(x = plist[inds], nlist = nlist[inds], ys = yvecs)
end

# ╔═╡ fb698663-e9d6-4368-bd53-d88da5e6f2c4
const colors = PlotlyBase.colors.cyclical[:tableau_colorblind];

# ╔═╡ 7c2015dd-a786-49f5-9fe9-9199335ebd09
#prepare parameter scan trace with standard deviation
function maketrace(prep, name, hovertemplate, color)
	hovertemplatelist = ["$hovertemplate, reward 1 std worse = $v" for v in prep.ys[3]]
	#mean line
	meantrace = scatter(x = prep.x, y = prep.ys[1], name = name, hovertemplate = hovertemplatelist, line_color = color, legendgroup = color)

	#dotted lines showing min and max
	extrematrace = scatter(x = prep.x, y = prep.ys[3], name = "", showlegend = false, legend = "legend2", line = attr(dash = "dot", width = 5, color = color), opacity = 0.25, hovertemplate = hovertemplatelist, legendgroup = color)

	vcat(meantrace, extrematrace)
end

# ╔═╡ 140f1e20-f86d-4a6f-9cff-99685e129e1c
function plot_stationary_param_search(;k = 10, steps = 1000, kwargs...)
	algorithms = [
		(p -> ActionValue(Qinit = 0.0f0, explorer = ϵ_Greedy(p)), -100, -1), 
		(p -> GradientReward(α=p), -100, 100), 
		(p -> ActionValue(Qinit = 0.0f0, explorer = UCB(p)), -100, 100), 
		(p -> ActionValue(Qinit = p, explorer = ϵ_Greedy(0.0), update_average = ConstantStep()), -10, 10)
	]
	
	names = [L"\epsilon\text{-greedy }", L"\text{gradient bandit }", "UCB", L"\text{greedy optimistic initialization } \alpha = 0.1"]

	hovertemplates = [
		"ϵ = %{x:.2g}, reward = %{y:.3g} <extra> ϵ-greedy</extra>",
		"α = %{x:.2g}, reward = %{y:.3g} <extra> gradient bandit</extra>",
		"c = %{x:.2g}, reward = %{y:.3g} <extra> UCB</extra>",
		"Q0 = %{x:.2g}, reward = %{y:.3g} <extra> greedy optimistic</extra>",
	]

	results = [stationary_param_search(k, algo[1]; exmin = algo[2], exmax = algo[3], steps = steps, kwargs...) for algo in algorithms]

	extracts = [preparetrace(runlist) for runlist in results]
	idealx = reduce(vcat, a.x for a in extracts)
	idealy = reduce(vcat, a.ys[2] for a in extracts) 
	nlist = sort(unique(reduce(vcat, a.nlist for a in extracts)))

	traces = reduce(vcat, [maketrace(extracts[i], names[i], hovertemplates[i], colors[i]) for i in eachindex(names)])

	idealtrace = scatter(x = idealx, y = idealy, name = "ideal", hovertemplate = "ideal reward = %{y:.3g}<extra></extra>", mode = "markers")
	Plot([idealtrace; traces], Layout(xaxis = attr(title = "Method Parameter (see hovertext)", type = "log", tickvals = sort(unique(idealx)), ticktext = print_power2.(nlist)), yaxis = attr(title = "Average Reward over first $steps steps"), legend = attr(orientation = "h", x = -.1, y = -.2), width = 700, height = 500))	
end

# ╔═╡ 1cde4625-f8ed-4403-835c-95cc30206699
const datapath = joinpath(@__DIR__, "parameter_studies")

# ╔═╡ 7c562867-55d3-4b4f-950d-c8efc4a9ff32
function loadplots()
	files = filter(f -> occursin(r"\.html$", f), readdir(datapath))
	isempty(files) && return Dict{String, HTML{String}}()
	Dict(split(f, ".") |> first |> String => HTML(String(read(joinpath(datapath, f)))) for f in files)
end

# ╔═╡ 6ce00349-1cf0-4a80-bddd-c1d26b66d051
const plotdict = loadplots();

# ╔═╡ 8a0462e9-fd65-4ccc-b795-446cf9ae7392
function make_or_lookup_param_plot(;f = plot_stationary_param_search, basename = "stationary_parameter_search", remakeplot = false, steps = 1000, kwargs...)
	corename = "$(basename)_$(steps)_steps"
	plotpath = joinpath(datapath, corename)
	# p = jldopen(jldpath, "a+") do f
		if !remakeplot && haskey(plotdict, corename)
			htmlplot = plotdict[corename]
		else
			io = IOBuffer()
			p = f(;steps = steps, kwargs...)
			PlotlyBase.to_html(io, p)
			htmlplot = io.data |> String |> HTML
			# delete!(jldfile, corename)
			# jldfile[corename] = p
			plotdict[corename] = htmlplot
			open("$plotpath.html", "w") do fplot
				PlotlyBase.to_html(fplot, p)
			end
		end
		# return p
	# end
	# return PlutoPlotly.plot(p)
	htmlplot
end

# ╔═╡ 88e43fed-fcf3-4071-996a-63f63c3d49b4
md"""
Number of Steps to Accumulate Reward: $(@bind stationary_numsteps confirm(NumberField(100:100:10000, default = 1000)))
"""

# ╔═╡ 97f6221d-3289-4e54-a80d-26c5c81f2651
md"""
By default, a plot will be loaded that matches this search criteria.  Check the box below to recalculate the search and save a new plot whever the submit button is clicked.

Recompute Parameter Search: $(@bind execute_stationary confirm(CheckBox()))
"""

# ╔═╡ bf3770ea-ee54-4296-ab33-340aea445670
# ╠═╡ show_logs = false
md"""
### Figure 2.6
Parameter study of bandit algorithms on the 10-armed testbed for stationary normally distributed bandit rewards.

$(make_or_lookup_param_plot(;remakeplot = execute_stationary, steps = stationary_numsteps))
"""

# ╔═╡ 51b7a645-269a-418c-b6d8-39c01d0609f1
make_or_lookup_param_plot(;remakeplot = execute_stationary, steps = stationary_numsteps)

# ╔═╡ d0111453-9a66-411d-9966-fc386d1bdcb7
md"""
> ### *Exercise 2.11 (programming)* 
> Make a figure analogous to Figure 2.6 for the nonstionary case outlined in Exercise 2.5.  Include the constant-step-size ϵ-greedy algorithm with α=0.1.  Use runs of 200,000 steps and, as a performance measure for each algorithm and parameter setting, use the average reward over the last 100,000 steps.
"""

# ╔═╡ 4a89cdd9-c20f-40e2-bc84-c3ea9cbf00e7
md"""
Below are functions which mimic those for the stationary bandit parameter search.  The final function produces a plot and saves it to disk or loads one that already exists.
"""

# ╔═╡ aa5acd7c-6a0b-454f-ab05-12a606dd9fc2
nonstationary_param_search(args...; kwargs...) = param_search(args..., average_nonstationary_runs_cum_reward; kwargs...)

# ╔═╡ 98f4d5e6-7569-457e-851d-713c572ae400
function plot_nonstationary_param_search(;k = 10, steps = 1_000, kwargs...)
	algorithms = [
		(p -> ActionValue(Qinit = 0.0f0, explorer = ϵ_Greedy(p)), -10, -1), 
		(p -> ActionValue(Qinit = 0.0f0, explorer = ϵ_Greedy(p), update_average = ConstantStep()), -10, -1), 
		(p -> GradientReward(α=p), -100, 100), 
		(p -> GradientReward(α=p, update_average = ConstantStep()), -100, 100), 
		(p -> ActionValue(Qinit = 0.0f0, explorer = UCB(p)), -100, 100), 
		(p -> ActionValue(Qinit = 0.0f0, explorer = UCB(p), update_average = ConstantStep()), -100, 100), 
		(p -> ActionValue(Qinit = p, explorer = ϵ_Greedy(0.0), update_average = ConstantStep()), -10, 10)
		]
	
	names = [L"\epsilon\text{-greedy sample average}", L"\epsilon\text{-greedy constant step average } (\alpha = 0.1)", L"\text{gradient bandit sample average}", L"\text{gradient bandit constant step average}", "UCB sample average", "UCB constant step average", L"\text{greedy optimistic initialization } \alpha = 0.1"]

	hovertemplates = [
		"ϵ = %{x:.2g}, reward = %{y:.3g} <extra> ϵ-greedy Sample Average</extra>",
		"ϵ = %{x:.2g}, reward = %{y:.3g} <extra> ϵ-greedy Constant Step Average</extra>",
		"α = %{x:.2g}, reward = %{y:.3g} <extra> gradient bandit Sample Average</extra>",
		"α = %{x:.2g}, reward = %{y:.3g} <extra> gradient bandit Constant Step Average</extra>",
		"c = %{x:.2g}, reward = %{y:.3g} <extra> UCB Sample Average</extra>",
		"c = %{x:.2g}, reward = %{y:.3g} <extra> UCB Constant Step</extra>",
		"Q0 = %{x:.2g}, reward = %{y:.3g} <extra> greedy optimistic</extra>",
	]

	results = [nonstationary_param_search(k, algo[1]; exmin = algo[2], exmax = algo[3], steps = steps) for algo in algorithms]

	extracts = [preparetrace(runlist) for runlist in results]
	idealx = reduce(vcat, a.x for a in extracts)
	idealy = reduce(vcat, a.ys[2] for a in extracts) 
	nlist = sort(unique(reduce(vcat, a.nlist for a in extracts)))

	traces = reduce(vcat, [maketrace(extracts[i], names[i], hovertemplates[i], colors[i]) for i in eachindex(names)])

	idealtrace = scatter(x = idealx, y = idealy, name = "ideal", hovertemplate = "ideal reward = %{y:.3g}<extra></extra>", mode = "markers")

	# idealx = reduce(vcat, [results[i][1] for i in eachindex(names)])
	# idealy = reduce(vcat, [results[i][3][2] for i in eachindex(names)])
	# nlist = sort(unique(reduce(vcat, [results[i][2] for i in eachindex(names)])))

	
	# traces = [scatter(x = results[i][1], y = results[i][3][1], name = names[i], hovertemplate = hovertemplates[i]) for i in  eachindex(names)]
	# idealtrace = scatter(x = idealx, y = idealy, name = "ideal", hovertemplate = "ideal reward = %{y:.2g}<extra></extra>", mode = "markers")
	Plot([idealtrace; traces], Layout(xaxis = attr(title = "Method Parameter (see hovertext)", type = "log", tickvals = sort(unique(idealx)), ticktext = print_power2.(nlist)), yaxis = attr(title = "Average Reward Final $(floor(Int64, steps / 2)) Steps"), legend = attr(orientation = "h", x = -0.1, y = -0.2), width = 700, height = 600))
	
end

# ╔═╡ 9bd99099-1dfa-477a-9896-3da94bcc0633
exercise_2_11(;kwargs...) = make_or_lookup_param_plot(;f = plot_nonstationary_param_search, basename = "nonstationary_parameter_search", kwargs...)

# ╔═╡ 3b88ec30-768b-44d0-88ee-b3ed989f22c3
@bind nonstationarysearchparams confirm(PlutoUI.combine() do Child
md"""
Number of Steps to Accumulate Reward (only measured on second half): $(Child(:steps, NumberField(1000:1000:1_000_000, default = 200_000)))

Reward Drift Rate Per Step $$\sigma$$: $(Child(:σ, NumberField(0.001:0.001:0.1, default = 0.01)))

By default, a plot will be loaded that matches this search criteria.  Check the box below to recalculate the search and save a new plot whever the submit button is clicked.

Recompute Parameter Search: $(Child(:remakeplot, CheckBox()))
"""
end)

# ╔═╡ d59126d7-5af0-4d06-a57b-e115eec32388
# ╠═╡ show_logs = false
exercise_2_11(;nonstationarysearchparams...)

# ╔═╡ 37446874-1c28-491b-b3cc-b4ad3282686e
md"""
Recreation of Figure 2.6 for the non-stationary case over $(nonstationarysearchparams.steps) steps with a drift rate of σ = $(nonstationarysearchparams.σ). These parameters can be adjusted above, but by default this is performed over 200,000 steps with the accumulated reward only being measured on the final 100,000 steps.  Unlike in the stationary case, the ϵ-greedy method with α=0.1 for updating the Q values performs the best at a very small ϵ value of $$2^{-7}.$$  The UCB method is the second best performer but requires a very large c value of 128 compared to ~1 for the stationary case in which it was the best performer.  This UCB method also uses the sample average which is not ideal for a non-stationary distribution.  That is one of the reasons why it was mentioned earlier in the chapter that it is difficult to adapt the UCB technique to the non-stationary problem.  We can use the constant step size method but that doesn't help the fact that the variance estimates are wrong.  
"""

# ╔═╡ 36602c38-8b29-4158-b299-94015a333762
md"""
# Dependencies and Settings
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
Distributions = "~0.25.100"
LaTeXStrings = "~1.3.0"
Latexify = "~0.16.1"
PlutoPlotly = "~0.3.9"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.52"
SpecialFunctions = "~2.3.1"
StatsBase = "~0.34.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "1572bd4e790356c6fdf468cf6358268b3b27299c"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

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

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "d9a8f86737b665e15a9641ecbac64deef9ce6724"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.23.0"

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
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "e460f044ca8b99be31d35fe54fc33a5c33dd8ed7"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.9.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "938fe2981db009f531b6332e31c58e9584a2f9bd"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.100"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "f372472e8672b1d993e93dada09e23139b509f9e"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.5.0"

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

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

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

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "b864cb409e8e445688bc478ef87c0afe4f6d1f8d"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "c3ce8e7420b3a6e071e0fe4745f5d4300e37b13f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.24"

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
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "f9b1e033c2b1205cf30fd119f4e50881316c1923"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.1"
weakdeps = ["Requires", "TOML"]

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
version = "1.9.2"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Colors", "Dates", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "PackageExtensionCompat", "PlotlyBase", "PlutoUI", "Reexport"]
git-tree-sha1 = "9a77654cdb96e8c8a0f1e56a053235a739d453fe"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.3.9"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"

[[deps.PlutoProfile]]
deps = ["AbstractTrees", "FlameGraphs", "Profile", "ProfileCanvas"]
git-tree-sha1 = "154819e606ac4205dd1c7f247d7bda0bf4f215c4"
uuid = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
version = "0.4.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

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

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
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

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "75ebe04c5bed70b91614d684259b661c9e6274a4"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.0"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

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
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─db30e6c0-36bb-4602-a257-5768f3833525
# ╟─31d8fba9-cd28-4b2b-ba46-70a068b9ecad
# ╟─1e4ac085-7b72-4bad-ad87-21635930a6f7
# ╟─083c721c-70dd-4ca3-8160-fc0b0531914f
# ╠═97c94391-397d-4cf2-88b1-3bea3af56ed3
# ╠═7950e06b-e8ce-4bd7-9681-ab7b66dfec69
# ╟─3e5a226e-ecdb-43fb-a40a-a262da0ae542
# ╟─0b951e6e-4b97-4bb5-87d0-6be7f0fd4802
# ╟─14bd0549-747f-4513-809f-8bdb78027807
# ╠═55f89ca1-cd57-44e0-95e5-63a4be31418f
# ╠═3a798da5-c309-48f2-aab1-6602ded8a650
# ╠═33009d31-6d66-4aba-b44c-edd911c2f392
# ╠═c12dab6a-92f0-41b1-a6b7-c404c74d9a83
# ╟─6034cc0f-cbae-4d2f-a43f-1bb738c00f0b
# ╟─87f5ec29-623f-4666-824d-fad8c6448072
# ╟─79082409-3182-4e0b-9c8c-37a94543fee9
# ╟─965da91b-6a3f-456c-89ce-461c31e0fb7e
# ╟─464d43c0-cd59-49e6-88f6-12a767677418
# ╠═8b8a9449-04b7-4901-9a2c-fbbdc33dfdfa
# ╟─21e56374-35e6-4488-b8da-15e383017c77
# ╟─2bce1b80-2133-40a0-9367-fc2d491f6245
# ╟─4be9e81b-c3de-4c79-97b5-b41c03e0f187
# ╟─e07a27c5-0c9a-4893-a1cf-cf565ab78761
# ╟─86350532-13f8-4035-bad8-f25f41c93163
# ╟─a84f5393-bd7e-433e-b3d9-e9e7cfa1a329
# ╟─78c45162-dc8e-4faf-b1a9-8c71be86dcee
# ╟─eed2a4f2-48b2-4684-846e-aa99bb6dafd9
# ╠═fbe3dc15-9171-4a7e-8eea-b8cc052c9ba5
# ╠═647ab36b-641e-4024-ad2d-40ff33be28f4
# ╠═34f65898-cbcc-4832-afac-0f7a284e7f0b
# ╟─276779a3-9332-46bd-b511-a33a2fea4b5f
# ╟─d24bd737-9e09-441a-aa94-9279c80f566d
# ╟─7748ab8a-d186-49a4-b6ab-d1bd9ea34990
# ╠═bb16115a-a2d9-4b8d-9937-96ab1cdd1ce2
# ╟─9b625fc0-89bd-4064-a379-225e6a940af7
# ╟─8584ece7-badc-486a-9a57-b60e77f92673
# ╟─f9e60b35-84d2-4b4b-8832-6b0f08152396
# ╟─32bff269-e893-4907-b589-7ba2ae1314bd
# ╠═6292f449-8720-41f1-84de-1865fb5fddbf
# ╟─70e40b75-e7d8-4009-af80-3bf4086a28df
# ╟─b3ec4673-af63-4d4a-8314-fa7e594f8a37
# ╟─d4ce45ae-613e-41ee-b626-69b0dbcf6452
# ╟─cb93c588-3dfa-45f4-9d83-f2de26cb1cea
# ╟─c695b7f9-76ca-419b-924d-8338a42c8188
# ╟─23b99305-c8d9-4129-85fb-a5e4aabc4a31
# ╠═8fac4109-2e0d-4366-9118-018221e0b910
# ╟─093f312b-d70d-4bf7-bd53-8a1c7b2bee31
# ╟─0de99ee5-d94d-4d07-8cef-a6f9caf5e742
# ╟─f88029d6-3fc2-4552-8441-5ef37ac42638
# ╟─b24a92fc-f6c6-44e4-9afc-fa4249e4ab83
# ╟─9d7782f5-b530-40d5-9f75-280d3a762216
# ╟─0f244fa0-7591-4478-b172-d9c1de51f6e1
# ╠═99945570-0b2a-432a-ae83-3a7abc39cac8
# ╠═54deaa09-8f87-4caf-b2a0-f15bcd5b40a5
# ╠═24759e26-e670-4330-b6e4-b313620660f1
# ╠═ea6d7cad-47ad-4472-a9e9-1ee33c81058d
# ╠═51349e41-4696-4bd5-9bc1-cefbb82bea08
# ╠═f7519adc-7dfb-4030-86f0-7445699dd3db
# ╠═50fbdc85-82f1-4c52-936b-84eb14951d71
# ╠═8a6f3f85-64e4-4c31-9e69-43f50f42bbc9
# ╟─b5a2df21-4525-4320-b8dd-aea5ecdab832
# ╟─691aa77a-d6da-4fde-9024-c4195057179d
# ╟─649e3d20-e276-4f4b-aeb0-89150f180ef5
# ╟─1f9a98fd-ea29-415c-9f35-add34b513a34
# ╟─1c9b54cd-08dd-401e-9705-818741844e8d
# ╠═1004eb4b-1fed-4328-a08b-6f5d9dd5080b
# ╠═c2347999-5ade-420b-903f-30523b38eb0f
# ╠═f995d0af-50bc-4e33-9bbf-17a7ab06358a
# ╠═852df31d-18d8-466c-8225-e06ba7f05e96
# ╠═44b9ff95-ea3d-41f5-8098-445a263738a9
# ╠═ca726a9d-364d-48e2-8882-20ddbc85b664
# ╠═9d36934a-78cb-446b-b3db-1bbd88cf272d
# ╠═60b2079e-0efa-427e-93cf-7f4646fe202e
# ╠═3118e102-aeac-42d9-98fc-ca29f40be4cd
# ╠═8c0f06f7-2ed0-4f3a-ab4e-90ac142f0cd9
# ╠═a61c15eb-ed5f-4052-a3a3-3276940564a1
# ╠═47be3ae6-20f7-47d0-aae3-b67154afc1a8
# ╠═4c6ccbfe-a3ce-4f2d-bcb1-7f1a4b735c65
# ╠═e50596ab-91db-42f0-a62c-77629a4e79c7
# ╠═839861db-676f-4544-a802-0abb5d0049e1
# ╠═1ac5588c-3c32-436e-8b40-41715223fba7
# ╠═f00ab44e-0d84-40c7-aa23-358c77a013e3
# ╠═262952a7-280e-4af1-99a6-0899518484a2
# ╠═fbf7b108-dc68-4077-b63c-6f88161d2098
# ╠═e2597cc6-a6f6-4260-887b-c587cacd3bc8
# ╠═3b9bb9f0-ba9d-4320-935a-58912afe34b6
# ╠═0f6b4e2d-dc09-4e1c-834f-dd8aaa8743ae
# ╠═7a31d1c5-260b-42f4-b997-967150881e21
# ╠═5712b303-0aa3-4501-b1b5-020136d6e655
# ╠═f561b0a8-a086-4e1a-bc87-82c4205e89c9
# ╠═30aa1e1b-0b51-40c4-a093-ef92c3ad519a
# ╠═0e606680-dd65-444f-bc98-73de4abbcdd4
# ╠═4ebd4a5a-3bd1-48e2-b03e-a5a3b2ec18a1
# ╠═73e5b719-8b91-41d2-b83d-471d981b027f
# ╠═49e45202-b9ae-42ab-9575-a57edb626a20
# ╠═3a215c95-c595-4837-a842-1c1e1c6bfa3b
# ╠═a04da367-3f8d-422d-a443-4e3e666e30ef
# ╠═68470b1d-3cc2-4cb1-8dc2-53227e6300e7
# ╠═fb3381b5-10e3-4307-b76d-672245fac9e7
# ╠═ff6598fa-3366-416c-88a1-6bfcefeb1719
# ╠═d2ebd908-387d-4e40-bc00-61ce5f45ebdd
# ╠═1e1f6d10-1b31-4e0a-96f7-23207e913154
# ╠═13f0adab-7660-49df-b26d-5f89cd73192b
# ╠═45cc0a58-3534-4c67-bfd4-1c2b48d59a2e
# ╠═0fbbe455-79ce-44d6-b010-da0bb56adbb4
# ╠═c66f1676-aec1-489d-96ff-99d748dac0fe
# ╠═8a04adab-e97e-4ac4-a85e-5eae93b1c37b
# ╠═69b560c1-98ad-4cbf-89d2-e0516299bc69
# ╠═d9265b98-cc3e-4a60-b16e-f54d9f78c9d3
# ╠═4982b489-ca15-4188-90e5-565c45f02e01
# ╠═672a91c0-aa77-4257-8c83-d857f47cab6c
# ╠═638f99e6-1cdc-414c-9b67-fd626ec0be3e
# ╠═aa238ebc-8730-46c7-8ad9-41c7cac70b18
# ╠═62eb0650-96bc-4fd6-bfe0-bf05a4137a03
# ╠═30f05bd9-e939-4810-b710-edc7f5975921
# ╠═96566aad-6d5c-460c-a924-ae0bad5d8b2d
# ╠═4191ff98-f4ab-4f18-a148-d3d3fff3d0ad
# ╠═09227386-1620-4093-b913-5786205aad13
# ╠═781fa565-398c-4c89-89f3-0595455afc85
# ╠═7678b06b-feef-4656-a801-33f630437bfb
# ╠═8a52acf7-5d57-490f-8bd9-6e1e0e322872
# ╠═181d7eef-24a0-4775-a535-8ef901b7e4eb
# ╠═74024d96-d0c7-43c8-8379-caf843cbe4b8
# ╟─46ce2b1f-02cf-4dae-bf02-f67543f38b91
# ╠═c61630b0-29c1-4183-90d9-57c999187b53
# ╠═c72d35fc-c8fd-450e-9b95-d12ece5c2291
# ╠═8f1c7b0d-121c-46bf-884d-729e3b593025
# ╠═a15e5d05-a238-440b-9a43-d830c6ea2f4d
# ╠═22fa2b71-a98f-4b87-9e0b-9d373cd8915f
# ╠═2447c4ea-7752-457c-80da-ac0dd72a64c1
# ╠═865610bb-ee82-4440-9f32-f00d0382783b
# ╠═400e0de1-0101-4531-928f-08ca155da40c
# ╠═fb698663-e9d6-4368-bd53-d88da5e6f2c4
# ╠═7c2015dd-a786-49f5-9fe9-9199335ebd09
# ╠═140f1e20-f86d-4a6f-9cff-99685e129e1c
# ╠═1cde4625-f8ed-4403-835c-95cc30206699
# ╠═7c562867-55d3-4b4f-950d-c8efc4a9ff32
# ╠═6ce00349-1cf0-4a80-bddd-c1d26b66d051
# ╠═8a0462e9-fd65-4ccc-b795-446cf9ae7392
# ╟─88e43fed-fcf3-4071-996a-63f63c3d49b4
# ╟─97f6221d-3289-4e54-a80d-26c5c81f2651
# ╟─bf3770ea-ee54-4296-ab33-340aea445670
# ╟─51b7a645-269a-418c-b6d8-39c01d0609f1
# ╟─d0111453-9a66-411d-9966-fc386d1bdcb7
# ╟─4a89cdd9-c20f-40e2-bc84-c3ea9cbf00e7
# ╠═aa5acd7c-6a0b-454f-ab05-12a606dd9fc2
# ╠═98f4d5e6-7569-457e-851d-713c572ae400
# ╠═9bd99099-1dfa-477a-9896-3da94bcc0633
# ╟─3b88ec30-768b-44d0-88ee-b3ed989f22c3
# ╟─d59126d7-5af0-4d06-a57b-e115eec32388
# ╟─37446874-1c28-491b-b3cc-b4ad3282686e
# ╟─36602c38-8b29-4158-b299-94015a333762
# ╠═1fb1a518-e5ec-4777-80bc-bb55e8172100
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
