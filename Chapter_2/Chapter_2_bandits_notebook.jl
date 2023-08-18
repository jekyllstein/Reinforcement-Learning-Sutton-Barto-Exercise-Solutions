### A Pluto.jl notebook ###
# v0.19.27

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

# ╔═╡ 9071a52c-a0a3-4262-93e7-bb8ff93bf44f
#necessary for doing weighted sampling
using StatsBase

# ╔═╡ 1fb1a518-e5ec-4777-80bc-bb55e8172100
begin
	using Random, Base.Threads, PlutoPlotly, BenchmarkTools, PlutoUI, PlutoProfile, JLD2, Latexify, LaTeXStrings
	TableOfContents()
end

# ╔═╡ db30e6c0-36bb-4602-a257-5768f3833525
md"""
# Chapter 2: Multi-armed Bandits

## 2.1: A *k*-armed Bandit Problem

Consider a repeated choice among *k* different options.  A numerical reward is chosen from a stationary probability distribution that depends only on the action selected.  The objective is to maximize the accumulated reward over some time period, let's say 1000 action selections or *time steps*.

We denote the action selected on time step *t* as $A_t$, and the corresponding reward as $R_t$.  The value then of an arbitrary action $a$, denoted $q_*(a)$, is the expected reward given that $a$ is selected:

$q_*(a) \dot = \mathbf{E}[R_t|A_t=a]$

If we know the values, then the problem is trivial, but we assume that we only have estimates of the values at a time step $a$ which we will denote $Q_t(a)$.  At any given time step the greedy action is the one with the highest value estimate.  If we take non-greedy actions then we can improve our value estimate for other states.  To solve the problem in general we must balance *exploiting* the action estimated to be the best with *exploring* the values of other candidate actions.  What follows are various methods to balance these two choices.
"""

# ╔═╡ 31d8fba9-cd28-4b2b-ba46-70a068b9ecad
md"""
## 2.2: Action-value Methods

The true value is the mean reward when that action is selected.  One way to estimate this is by averaging the rewards actually received:

$Q_t(a) \dot = \frac{\text{sum of rewards when } a \text{ taken prior to } t}{\text{number of times } a \text{ taken prior to } t} = \frac{\sum_{i=1}^{t-1}R_i \cdot \mathbf{1}_{A_i=a}}{\sum_{i=1}^{t-1} \mathbf{1}_{A_i=a}} \tag{2.1}$

If the denominator is zero then we instead define $Q_t(a)$ by some default value such as 0.  We call this the *sample-average* method for estimating action values because each estmiate is an average of the sample of relevant rewards.

The simplest action selection rule is to select one of the actions with the higest estimated value.  This is the *greedy* action.

$A_t \dot = \operatorname*{argmax}_a Q_t(a)$
"""

# ╔═╡ 1e4ac085-7b72-4bad-ad87-21635930a6f7
md"""
>*Exercise 2.1* In ϵ-greedy action selection, for the case of two actions and $\epsilon = 0.5$, what is the probability that the greedy action is selected?

The greedy action will be selected in two cases, each of which has probability 0.5.  For case 1 the greedy action is explicitely taken so the probability of selecting the greedy action in this case is 1.  For case 2, we select an action randomly, so the probability of selecting the greedy action is $\frac{\text{num greedy actions}}{\text{num total actions}}=0.5$
Since both cases are independent, the probabilities can be summed after multiplying each by the probability of that case which is 0.5 for both.

$P(a = a_{greedy}) = 0.5 \times (1 + 0.5) = 0.5 + 0.25 = 0.75$
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
 	plot(violins, Layout(showlegend = false, yaxis = attr(title = "Reward distribution"), xaxis = attr(title = "Action"), annotations = [attr(x = i-1 + 0.5, y = qs[i], xref = "x", yref = "y", text = "q($i)", showhead=false, ax=0, ay=0) for i in 1:n]))
end

# ╔═╡ 3e5a226e-ecdb-43fb-a40a-a262da0ae542
md"""
Number of Arms in Testbed
$(@bind ktest confirm(NumberField(2:100, default = 10)))
"""

# ╔═╡ 0b951e6e-4b97-4bb5-87d0-6be7f0fd4802
plot_bandit_testbed(ktest)

# ╔═╡ 2e46853d-f384-49fb-bccd-2fb320444080
md"""
### Figure 2.1
Shows the reward distribution for each of the $ktest arms in the testbed.  The mean value is marked with a dashed line for each.
"""

# ╔═╡ 14bd0549-747f-4513-809f-8bdb78027807
md"""
### $\epsilon$-Greedy Action Value Method
The functions below implement the sample-average method for estimating the value of each action with the ϵ-greedy method of action selection.  Note that the `simple_algorithm` uses the incremental implementation of calculating the sample average which is described in section **2.4**.
"""

# ╔═╡ 55f89ca1-cd57-44e0-95e5-63a4be31418f
function sample_bandit(a::Integer, qs::Vector{T}) where T<:AbstractFloat
    randn(T) + qs[a] #generate a reward with mean q[a] and variance 1
end

# ╔═╡ 3a798da5-c309-48f2-aab1-6602ded8a650
function simple_algorithm(qs::Vector{Float64}, k::Integer, ϵ::AbstractFloat; steps = 1000, Qinit = 0.0, α = 0.0, c = 0.0)
    bandit(a) = sample_bandit(a, qs)
    N = zeros(k)
    Q = ones(k) .* Qinit
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
        shuffle!(actions) #so that ties are broken randomly with argmax
        a = if rand() < ϵ
            rand(actions)
		elseif c == 0.0
			actions[argmax(view(Q, actions))]
		else
           	actions[argmax(view(Q, actions) .+ (c .* sqrt.(log(i) ./ view(N, actions))))]
		end
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
        N[a] += 1.0
        if α == 0.0
            Q[a] += (1.0/N[a])*(step_reward[i] - Q[a])
        else 
            Q[a] += α*(step_reward[i] - Q[a])
        end
    end
    return (;Q, step_reward, step_reward_ideal, cum_reward, cum_reward_ideal, optimalstep, optimalaction_pct)
end

# ╔═╡ 33009d31-6d66-4aba-b44c-edd911c2f392
function average_simple_runs(k, ϵ; steps = 1000, n = 2000, Qinit = 0.0, α=0.0, c = 0.0)
    runs = Vector{NamedTuple}(undef, n)
    @threads for i in 1:n
        qs = create_bandit(k)
        runs[i] = simple_algorithm(qs, k, ϵ, steps = steps, Qinit = Qinit, α = α, c = c) 
    end
    map(i -> mapreduce(a -> a[i], (a, b) -> a .+ b, runs)./n, (:step_reward, :step_reward_ideal, :optimalstep, :cum_reward, :cum_reward_ideal, :optimalaction_pct))
end

# ╔═╡ c12dab6a-92f0-41b1-a6b7-c404c74d9a83
function action_value_testbed_plot(; k = 10, ϵ1 = 0.01, ϵ2 = 0.1, steps = 1000)
	qs = create_bandit(k)
	ϵ_list = [0.0, ϵ1, ϵ2]
	runs = [average_simple_runs(k, ϵ; steps = steps) for ϵ in ϵ_list]
	labs = ["ϵ=0 (greedy)", "ϵ=$(ϵ1)", "ϵ=$(ϵ2)"]
	fig22a = plot([map(enumerate(runs)) do (i, a)
		scatter(y = a[1], name = labs[i])
	end; scatter(y = runs[1][2], name = "Theoretical Limit")],
	Layout(xaxis_title = "Step", yaxis_title = "Reward Averaged Over Runs", legend = attr(orientation = "h", y = 1.2), height = 500))
	fig22b = plot(map(enumerate(runs)) do (i, a)
		scatter(y = a[3], name = labs[i])
	end, Layout(xaxis_title="Step", yaxis_title = "% Runs Taking Optimal Action", height = 500, showlegend = false))
	md"""
	$fig22a
	$fig22b
	### Figure 2.2
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
>*Exercise 2.2:* *Bandit example* Consider a k-armed bandit problem with k = 4 actions, denoted 1, 2, 3, and 4. Consider applying to this problem a bandit algorithm using $\epsilon$-greedy action selection, sample-average action-value estimates, and initial estimates of $Q_1(a) = 0$, for all a. Suppose the initial sequence of actions and rewards is $A_1 = 1$, $R_1 = −1$, $A_2 = 2$, $R_2 = 1$, $A_3 = 2$, $R_3 = −2$, $A_4 = 2$, $R_4 = 2$, $A_5 = 3$, $R_5 = 0$. On some of these time steps the $\epsilon$ case may have occurred, causing an action to be selected at random. On which time steps did this definitely occur? On which time steps could this possibly have occurred?

The table below summarizes the actions taken leading into every step and the Q estimate for each action at the end of each step.  So step 0 shows the initial Q estimates of 0 for every action and the selected action 1 that generates the reward on step 1.  For the row in step 1 it shows the Q estimates after receiving the reward on step 1 and thus what actions are demanded by a greedy choice leading into the next step.  If the action selected is not equal to or in the set of greedy actions, then a random action **must** have occured.  Since a random action choice can also select one of the greedy actions, such a random choice is possible at every step.  Note that the answer in row 0 corresponds to action $A_1$, row 1 -> $A_2$ etc...

|Step|$Q(1)$|$Q(2)$|$Q(3)$ | Action Selected | Reward 	| Greedy Action | $\epsilon$ Case |
|----|---- |---- | ---- | ---- 			  | --- 	|  ----         | ----          |
|  0 |  0  |  0  | 0    | 1               | -1 		| 	1-3         | possibly 		    |
|  1 |  -1 |  0  | 0    | 2               | 1 | 2-3           | possibly            |
|  2 |  -1 |  1  | 0 	| 2 | -2 | 2 | possibly |
| 3  |  -1 | -.5 | 0  	| 2 | 2 |  3 | definitely |
| 4  | -1  | 0.5 | 0 	| 3 | 0 | 2 | definitely |
| 5  | -1  |  0.5 |  0 | n/a | n/a |  n/a | n/a |
"""

# ╔═╡ 464d43c0-cd59-49e6-88f6-12a767677418
md"""
>*Exercise 2.3* In the comparison shown in Figure 2.2, which method will perform best in the long run in terms of cumulative reward and probability of selecting the best action? How much better will it be? Express your answer quantitatively.

In the long run both the $\epsilon = 0.1$ and $\epsilon = 0.01$ methods will have Q value estimates that converge to the true mean value of the reward distribution.  Since both methods will necessarily take random actions 10% and 1% of the time respectively, we'd expect each method to take the optimal action with probability $(1-\epsilon) + \epsilon \times \frac{1}{10}=\frac{10 - 9 \times \epsilon}{10}$.  So for each value of ϵ we have.  

$Pr(a=a_{best}|\epsilon = 0.1) = 0.91$

$Pr(a=a_{best}|\epsilon = 0.01) = 0.991$

For the $\epsilon = 0$ greedy case the expected reward and optimal action selection probability depends on the order of sampled actions and the likelihood of getting close to or on the optimal action enough to push its Q estimation to the top.  From the plots in figure 2.2 in practice that seems to lead to an average reward of ~1.05 and an optimal action selection probability of 0.3825.  For long term cummulative reward this case will have roughly $1.05 \times num\_steps$.  For the other two cases, the long term cummulative reward is based on the expected value of the highest reward mean which is empiracally ~1.55.  For a random action the expected reward should be 0 due to the normal distribution of the action mean rewards.  

$E(long\_term\_step\_reward|\epsilon=0.1) = 0.91 \times 1.55 = 1.41$
$E(long\_term\_step\_reward|\epsilon=0.01) = 0.991 \times 1.55 = 1.536$

For each case the long run cumulative reward is just this long term expected reward per step times the number of steps.  The statistical properties of the different arms of a generic bandit can be visualized below using the following function.
"""

# ╔═╡ 8b8a9449-04b7-4901-9a2c-fbbdc33dfdfa
function visualize_bandit_dist(;n = 10, samples = 10_000)
	maxdist = [maximum(randn(n)) for _ in 1:samples]
	rankdist = mapreduce(a -> sort(randn(n)), +, 1:samples) ./ samples
	p1 = histogram(x = maxdist) |> plot
	p2 = bar(x = 1:10, y = rankdist) |> a -> plot(a, Layout(xaxis_title = "Reward Rank", yaxis_title = "Mean Reward of Arm"))
	md"""
	#### Distribution of Best Action Reward for a $n Armed Bandit
	$p1

	#### Expected Value of Mean Reward for Arms Ranked from 1 to $n
	$p2
	"""
end

# ╔═╡ 2bce1b80-2133-40a0-9367-fc2d491f6245
md"""
Visualize $(@bind n_arms_vis NumberField(1:100, default = 10)) Armed Bandit
"""

# ╔═╡ 4be9e81b-c3de-4c79-97b5-b41c03e0f187
visualize_bandit_dist(n = n_arms_vis)

# ╔═╡ e07a27c5-0c9a-4893-a1cf-cf565ab78761
md"""
## 2.4 Incremental Implementation
"""

# ╔═╡ 86350532-13f8-4035-bad8-f25f41c93163
md"""
## 2.5 Tracking a Nonstationary Problem
"""

# ╔═╡ 78c45162-dc8e-4faf-b1a9-8c71be86dcee
md"""
>*Exercise 2.4* If the step-size parameters, $\alpha_n$, are not constant, then the estimate $Q_n$ is a weighted average of previously received rewards with a weighting different from that given by (2.6). What is the weighting on each prior reward for the general case, analogous to (2.6), in terms of the sequence of step-size parameters?

From (2.6):  $Q_{n+1} = Q_n + \alpha[R_n - Q_n]$ so here we consider the case where $\alpha$ is not a constant but rather can have a unique value for each step n.

$\begin{flalign}
Q_{n+1}&=Q_n + \alpha_n[R_n - Q_n]\\
&=\alpha_nR_n+(1-\alpha_n)Q_n\\
&=\alpha_nR_n+(1-\alpha_n)[\alpha_{n-1}R_{n-1}+(1-\alpha_{n-1})Q_{n-1}]\\
&=\alpha_nR_n+(1-\alpha_n)\alpha_{n-1}R_{n-1}+(1-\alpha_n)(1-\alpha_{n-1})Q_{n-1}\\
&=\alpha_nR_n+(1-\alpha_n)\alpha_{n-1}R_{n-1}+(1-\alpha_n)(1-\alpha_{n-1})[\alpha_{n-2}R_{n-2}+(1-\alpha_{n-2})Q_{n-2}]\\
&=\alpha_nR_n+(1-\alpha_n)\alpha_{n-1}R_{n-1}+(1-\alpha_n)(1-\alpha_{n-1})\alpha_{n-2}R_{n-2}+...\\
&=Q_1\prod_{i=1}^n \left( 1-\alpha_i \right)+\sum_{i=1}^{n} \left[ (R_i\alpha_i)\prod_{j=i+1}^n(1-\alpha_j) \right]\\
\end{flalign}$

For example if $\alpha_i=1/i$ then the product in the first term is 0 and the formula becomes:

$\begin{flalign}
Q_{n+1} &= \sum_{i=1}^{n} \left[ \frac{R_i}{i}\prod_{j=i+1}^n\frac{j-1}{j} \right]\\
&= \sum_{i=1}^{n} \left[ \frac{R_i}{i}\frac{i}{i+1}\frac{i+1}{i+2}...\frac{n-1}{n} \right]\\
&=\sum_{i=1}^{n} \frac{R_i}{n}
\end{flalign}$

	
from the expanded product we can see that all of the numerators and denominators cancel out leaving only $\frac{R_i}{n}$ which we expect for tihs form of $\alpha$ which was derived earlier for a simple running average.
"""

# ╔═╡ eed2a4f2-48b2-4684-846e-aa99bb6dafd9
md"""
> Exercise 2.5 (programming) Design and conduct an experiment to demonstrate the difficulties that sample-average methods have for nonstationary problems. Use a modified version of the 10-armed testbed in which all the q⇤(a) start out equal and then take independent random walks (say by adding a normally distributed increment with mean 0 and standard deviation 0.01 to all the q⇤(a) on each step). Prepare plots like Figure 2.2 for an action-value method using sample averages, incrementally computed, and another action-value method using a constant step-size parameter, ↵ = 0.1. Use " = 0.1 and longer runs, say of 10,000 steps. 

See code and figures below for answer
"""

# ╔═╡ fbe3dc15-9171-4a7e-8eea-b8cc052c9ba5
function nonstationary_algorithm(k::Integer, ϵ::AbstractFloat; steps = 10000, σ = 0.01, α = 0.0)
    qs = zeros(k)
    Q = zeros(k)
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
function average_nonstationary_runs(k, ϵ, α; n = 2000)
    runs = Vector{NamedTuple}(undef, n)
    @threads for i in 1:n
        runs[i] = nonstationary_algorithm(k, ϵ, α = α) 
    end
    map(i -> mapreduce(a -> a[i], (a, b) -> a .+ b, runs)./n, (:step_reward, :optimalstep, :step_reward_ideal, :cum_reward, :cum_reward_ideal, :optimalaction_pct))
end

# ╔═╡ 32bff269-e893-4907-b589-7ba2ae1314bd
md"""
2.6 Optmisitic Initial Values
"""

# ╔═╡ c37922b9-6c23-4701-8499-9bb73ccc3684
optimistic_greedy_runs = average_simple_runs(k, 0.0, Qinit = 5.0, α = 0.1)

# ╔═╡ 7208da59-282b-4c27-8b0b-460540afb5af
realistic_ϵ_runs = average_simple_runs(k, 0.1, α = 0.1)

# ╔═╡ 0b39196d-a9c4-4d17-a7ab-b0d7264b07fa
plot([optimistic_greedy_runs[3] realistic_ϵ_runs[3]], xaxis = "Step", yaxis = ((0, 1), "% Runs Taking Optimal Action"), lab = ["Optimistic, greedy Qinit = 5" "Realistic, ϵ-greedy Qinit = 0, ϵ=0.1"], size = (700, 400))

# ╔═╡ d4ce45ae-613e-41ee-b626-69b0dbcf6452
md"""
>*Exercise 2.6: Mysterious Spikes* The results shown in Figure 2.3 should be quite reliable because they are averages over 2000 individual, randomly chosen 10-armed bandit tasks.  Why, then, are there oscillations and spikes in the early part of the curve for the optimistic method? In other words, what might make this method perform particularly better or worse, on average, on particular early steps?

The spike occurs on step 11.  Due to the initial Q values it is almost 100% likely that a given run with sample each of the 10 possible actions once before repeating any.  That would mean that at any given step only 10% of the runs would select the optimal action and indeed for the first 10 steps about 10% of the runs are selecting the optimal action as we'd expect from random chance.  On the 11th step, the Q value estimate for each action is $(0.9 \times 5) + (0.1 \times action\_reward)$.  The optimal action for each bandit has the highest mean reward, but there is some chance that one of the other 10 actions produced a higher reward the step it was sampled.  However, on the 11th step, the number of runs that select the optimal action will be equal to the probability that the optimal action produced the highest reward when it was sampled which empirally is ~44%.  Due to the Q value initialization though, the reward on step 11 for those cases that selected the optimal action will almost certainly lower the Q value estimate for that action below the others resulting in the sudden drop of the optimal action selection on step 12.  
"""

# ╔═╡ cb93c588-3dfa-45f4-9d83-f2de26cb1cea
md"""
>*Exercise 2.7: Unbiased Constant-Step-Size Tick* In most of this chapter we have used sample averages to estimate action values because sample averages do not produce the initial bias that constant step sizes do (see analysis leading to (2.6)).  However, sample averages are not a completely satisfactory solution because they may perform poorly on nonstationary problems.  Is it possible to avoid the bias of constant sample sizes while retaining their advantages on nonstationary problems?  One way is to use a step size of 

> $\beta_n \dot= \alpha / \bar{o}_n,$ 
>to process the nth reward for a particular action, where $\alpha>0$ is a conventional constant step size, and $\bar{o}_n$ is a trace of one that starts at 0:

> $\bar{o}_n \dot= \bar{o}_{n-1}+\alpha(1-\bar{o}_{n-1}), \text{ for } n \geq 0, \text{ with } \bar{o}_0 \dot= 0.$
>Carry out an analysis like that in (2.6) to show that $Q_n$ is an exponential recency-weighted average *without initial bias*.

$Q_{n+1} = Q_n + \beta_n[R_n - Q_n]$ where $\beta_n \dot= \alpha / \bar{o}_n $ and $\bar{o}_n \dot= \bar{o}_{n-1}+\alpha(1-\bar{o}_{n-1})$

$\bar{o}_n = \bar{o}_{n-1} + \alpha(1-\bar{o}_{n-1})=\bar{o}_{n-1}(1-\alpha)+\alpha$

We can expand $\bar{o}_n$ backwards to get an explicit formula.

$\bar{o}_n=\bar{o}_{n-1}(1-\alpha)+\alpha,$

$\bar{o}_n=(\bar{o}_{n-2}(1-\alpha) + \alpha)(1-\alpha)+\alpha$

$\bar{o}_n=\bar{o}_{n-2}(1-\alpha)^2 + \alpha((1-\alpha)+1)$

$\bar{o}_n=(\bar{o}_{n-3}(1-\alpha)+\alpha)(1-\alpha)^2 + \alpha((1-\alpha)+1)$

$\bar{o}_n=\bar{o}_{n-3}(1-\alpha)^3+\alpha((1-\alpha^2) + (1-\alpha)+1)$

$\vdots$

$\bar{o}_n=\bar{o}_0(1-\alpha)^n + \alpha\sum_{i=0}^{n-1}(1-\alpha)^i=\alpha\sum_{i=0}^{n-1}(1-\alpha)^i$

This sum has an explicit formula as can be seen by:

$S = 1 + (1-\alpha) + (1-\alpha)^2 + \cdots + (1-\alpha)^{n-1}$
$S(1-\alpha)=(1-\alpha)+\cdots+(1-\alpha)^n=S-1+(1-\alpha)^n$

$-S\alpha=-1+(1-\alpha)^n$
$S=\frac{1-(1-\alpha)^n}{\alpha}$

Therefore, $\bar{o}_n=\alpha\frac{1-(1-\alpha)^n}{\alpha}=1-(1-\alpha)^n$, and since $0<\alpha<1$, then $(1 - \alpha)^n \rightarrow 0 \text{ as } n \rightarrow \inf$.

$\beta_n=\frac{\alpha}{\bar{o}_n}=\frac{\alpha}{1-(1-\alpha)^n} \implies \beta_1=1$

From exercise 2.4, we have the formula for $Q_n$ with a non-constant coefficient $\alpha_n$ which we can trivially replace here with $\beta_n$

$Q_n=Q_1\prod_{i=1}^n \left( 1-\beta_i \right)+\sum_{i=1}^{n} \left[ R_i\beta_i\prod_{j=i+1}^n(1-\beta_j) \right]$

Since $\beta_1=1$, the product associated with $Q_1$ will be 0.  Since there is no dependency on the initial value of Q, we can say this formula for updating Q has *no initial bias*.
If we then make the substitution $\beta_n=\frac{\alpha}{1-(1-\alpha)^n}$, we have

$Q_n=\sum_{i=1}^{n} \left[ R_i\frac{\alpha}{1-(1-\alpha)^i}\prod_{j=i+1}^n \left( 1-\frac{\alpha}{1-(1-\alpha)^j} \right) \right]$
$Q_n=\alpha\sum_{i=1}^n \left[ \frac{R_i}{1-(1-\alpha)^i}\prod_{j=i+1}^n \left( \frac{(1-\alpha)(1-(1-\alpha)^{j-1})}{1-(1-\alpha)^j} \right) \right]$
$Q_n=\alpha\sum_{i=1}^n \left[ \frac{R_i(1-\alpha)^{n-i}}{1-(1-\alpha)^i}\prod_{j=i+1}^n \left( \frac{1-(1-\alpha)^{j-1}}{1-(1-\alpha)^j} \right) \right]$

Examining the product term on its own, we can see it simplifies.

$\prod_{j=i+1}^n \left( \frac{1-(1-\alpha)^{j-1}}{1-(1-\alpha)^j} \right)$
$\frac{1-(1-\alpha)^{i}}{1-(1-\alpha)^{i+1}}\frac{1-(1-\alpha)^{i+1}}{1-(1-\alpha)^{i+2}}\cdots\frac{1-(1-\alpha)^{n-1}}{1-(1-\alpha)^{n}}=\frac{1-(1-\alpha)^i}{1-(1-\alpha)^n} \text{ for i≤n}$

Replacing this expression for the product in the expression for $Q_n$ we have:

$Q_n=\alpha\sum_{i=1}^n \left[ \frac{R_i(1-\alpha)^{n-i}}{1-(1-\alpha)^i}\frac{1-(1-\alpha)^i}{1-(1-\alpha)^n}\right]=\frac{\alpha}{1-(1-\alpha)^n}\sum_{i=1}^n R_i(1-\alpha)^{n-i}$

If we expand this sum going backwards from $i=n$:

$Q_n=\frac{\alpha}{1-(1-\alpha)^n} \left[ R_n+R_{n-1}(1-\alpha)+R_{n-2}(1-\alpha)^2+\cdots+R_1(1-\alpha)^{n-1} \right]$

The constant term starts off at $1$ for $n=1$ and approaches $\alpha$ in the limit of $n \rightarrow \inf$.  If $0<\alpha<1$, then the coefficients in the sum section for $R_i$ decrease exponentially from 1 for $i=n$ to $(1-\alpha)^{n-1}$ for $i=1$.  So the average over rewards includes every reward back to $R_1$ like the simple average but the coefficients become exponentially smaller approaching 0 as $n \rightarrow \inf$.   

"""

# ╔═╡ 23b99305-c8d9-4129-85fb-a5e4aabc4a31
md"""
2.7 Upper-Confidence-Bound Action Selection
"""

# ╔═╡ 093f312b-d70d-4bf7-bd53-8a1c7b2bee31
c = 2.0

# ╔═╡ 3c58fbe9-38d1-40d2-a86d-3c328c0da517
ucb_runs = average_simple_runs(k, 0.0, c = c)

# ╔═╡ 53a7b41e-e6b1-4da7-8130-27627ee9af2c
ϵ_runs = average_simple_runs(k, 0.1)

# ╔═╡ f19dd53e-5ca1-4c8d-a04d-ea436394c089
plot([ucb_runs[1] ϵ_runs[1]], lab = ["UCB c=$c" "ϵ-greedy ϵ=0.1"], xaxis = "Step", yaxis = "Reward Averaged Over Runs", size = (700, 400))

# ╔═╡ f88029d6-3fc2-4552-8441-5ef37ac42638
md"""
*Exercise 2.8: UCB Spikes* In Figure 2.4 the UCB algorithm shows a distinct spike in performance on the 11th step.  Why is this?  Note that for your answer to be fully satisfactory it must explain both why the reward increases on the 11th step and why it decreases on the subsequent steps.  Hint: If $c=1$, then the spike is less prominent.
>Due to the UCB calculation, any state that has not been visited will have an infinite Q value, thus for the first 10 steps similar to the optimistic Q initialization each run will sample each of the 10 possible actions.  One the 11th step, the exploration incentive for each action will be equal, so the most likely action to be selected is the optimal action since it is the most likely to have produced a reward higher than any other action.  If c is very large, then on step 12 no matter how good of a reward we received for the optimal action, that action will be penalized compared to the others because it will have double the visit count.  In particular for c = 2.0, the exploration bonus for the optimal action if selected on step 11 will be 2.2293 vs 3.31527 for all other actions.  Since the q's are normally distributed it is unlikely that the reward average for the optimal action is >1 than the next best action.  The larger c is the more of a relative bonus the other actions have and the probability of selecting the optimal action twice in a row drops to zero.  As c changes the improved reward on step 11 remains similar but the dropoff on step 12 becomes more severe the larger c is.
"""

# ╔═╡ b24a92fc-f6c6-44e4-9afc-fa4249e4ab83
md"""
2.8 Gradient Bandit Algorithms
"""

# ╔═╡ 9d7782f5-b530-40d5-9f75-280d3a762216
md"""
>*Exercise 2.9* Show that in the case of two actions, the soft-max distribution is the same as that given by the logistic, or sigmoid, function often used in statistics and artificial neural networks.

The sigmoid function is defined as: $S(x) = \frac{1}{1 + e^{-x}}$.  For two actions, let's denote them $a_1$ and $a_2$.  Now for the action probabilities we have.

$\pi(a_1) = \frac{e^{H_t(a_1)}}{e^{H_t(a_1)} + e^{H_t(a_2)}}=\frac{1}{1+e^{-(H_t(a_1) - H_t(a_2))}}$

This expression for $\pi(a_1)$ is equivalent to $S(x)$ with $x = H_t(a_1) - H_t(a_2)$ which is the degree of preference for action 1 over action 2.  As expected, if the preferences are equal then it is equavalent to $x=0$ with a probability of 50%.  The same analysis applies to action 2 with the actions reversed from this case.
"""

# ╔═╡ 54deaa09-8f87-4caf-b2a0-f15bcd5b40a5
#calculates a vector of probabilities for selecting each action given exponentiated "perferences" given in expH using the softmax distribution
function calc_πvec(expH::AbstractVector)
	s = sum(expH)
	expH ./ s
end

# ╔═╡ ea6d7cad-47ad-4472-a9e9-1ee33c81058d
function update_H!(a::Integer, H::AbstractVector, π_vec::AbstractVector, α, R, R̄)
	for i in eachindex(H)
		H[i] = H[i] + α*(R - R̄)*(i == a ? (1.0 - π_vec[i]) : -π_vec[i])	
	end	
end

# ╔═╡ 51349e41-4696-4bd5-9bc1-cefbb82bea08
sample_action(actions, π_vec) = sample(actions, pweights(π_vec))

# ╔═╡ 04e5a0db-f47b-46a0-bf98-eeafce87a44b
α_list = [0.025, 0.05, 0.1, 0.2, 0.4]

# ╔═╡ c30715fe-8650-46ad-ba8d-30d34ead569a
md"""
For the plots below the bandit parameters were created with an offset of 4.0 so the expected reward value for any action is centered at 4.0 instead of 0.  For the gradient bandit with a baseline, it doesn't affect the curves at all, but if the baseline is removed then the results are worse as seen in the second plot.  However, if α is made smaller it seems like it will also converge to a similar success rate just over a longer time.  The optimal value of α is much lower than when the baseline is removed"
"""

# ╔═╡ 1c9b54cd-08dd-401e-9705-818741844e8d
md"""
# Code Refactoring
Due to the variety of algorithms and parameters for the bandit, I have rewritten the test environment with types that represent the different algorithms.  The run simulator will dispatch on the types to correctly simulate that method with its parameters.  Some of the previous simluations are plots are generated again.  Because of the style used, only one simulation function is needed with the flexibility to select any combination of techniques in the chapter.
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
#extrends default value to other types
(::Type{ϵ_Greedy{T}})() where T<:AbstractFloat = ϵ_Greedy(T(0.1))

# ╔═╡ ca726a9d-364d-48e2-8882-20ddbc85b664
(::Type{ϵ_Greedy{T}})(e::ϵ_Greedy) where T<:AbstractFloat = ϵ_Greedy(T(e.ϵ))

# ╔═╡ 9d36934a-78cb-446b-b3db-1bbd88cf272d
#how to convert type of explorer when it is already the same
(::Type{T})(e::T) where T <: ϵ_Greedy = e

# ╔═╡ 926b95b1-c188-4cfb-8272-10c3a3b9f8e5
#already knows how to convert if the argument passed is wrong
ϵ_Greedy{BigFloat}(1.0f0)

# ╔═╡ 53fca00d-69b8-42aa-aeec-41de02f553a3
ϵ_Greedy()

# ╔═╡ fba8da74-4a34-49ad-a60e-e1849d138cc8
ϵ_Greedy{Float32}()

# ╔═╡ bae0b1b4-149e-416c-9b92-1cf8ebde07a3
explorer = ϵ_Greedy(0.1)

# ╔═╡ cabfd2b9-307e-4026-a3ac-91d2674c58af
ϵ_Greedy{Float32}(explorer)

# ╔═╡ 68f7540b-a6d0-49f2-8d34-8b96888e3109
with_terminal() do 
	@code_lowered ϵ_Greedy{Float64}(explorer)
end

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

# ╔═╡ 3c4462d4-c1ab-4231-8c9e-75ab33734061
ex2 = UCB(1.0)

# ╔═╡ 2a70e0cb-311f-4ad4-b55d-77d299030d9c
UCB{Float32}(ex2)

# ╔═╡ 5dac18d3-dcaa-47ce-b050-09357dc41502
with_terminal() do 
	@code_lowered UCB{Float32}(ex2)
end

# ╔═╡ 33930d1a-ad7b-4359-b46d-fe84d47b16dc
ϵ_Greedy{Float64} <: Explorer{Float64}

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

# ╔═╡ f44d8016-984f-43c1-8541-0bcc74adf735
UnbiasedConstantStep()

# ╔═╡ 4d105237-947f-429a-be12-ea73e0ce362c
q_avg = ConstantStep()

# ╔═╡ c9c9c147-be36-4014-be1e-4e72e325f611
ConstantStep{BigFloat}()

# ╔═╡ 01929fe4-0ab7-4e5f-89cf-ade442c4f1a3
ConstantStep{Float32}(q_avg)

# ╔═╡ f561b0a8-a086-4e1a-bc87-82c4205e89c9
struct SampleAverage{T<:AbstractFloat} <: AverageMethod{T}
end

# ╔═╡ 30aa1e1b-0b51-40c4-a093-ef92c3ad519a
(::Type{SampleAverage})() = SampleAverage{Float64}()

# ╔═╡ 0e606680-dd65-444f-bc98-73de4abbcdd4
(::Type{U})(a::U) where U <: SampleAverage = a

# ╔═╡ 4ebd4a5a-3bd1-48e2-b03e-a5a3b2ec18a1
(::Type{U})(a::SampleAverage) where U <: SampleAverage = U() 

# ╔═╡ 3f8d5fc9-a153-4832-bd48-387679757a45
q_avg2 = SampleAverage()

# ╔═╡ 6d3076e4-8644-4436-9c95-480093689dc9
SampleAverage{BigFloat}(q_avg2)

# ╔═╡ 1e1f6d10-1b31-4e0a-96f7-23207e913154
abstract type BanditAlgorithm{T} end

# ╔═╡ 73e5b719-8b91-41d2-b83d-471d981b027f
(::Type{Explorer{T}})(e::ϵ_Greedy) where T<:AbstractFloat = ϵ_Greedy{T}(e)

# ╔═╡ 49e45202-b9ae-42ab-9575-a57edb626a20
(::Type{Explorer{T}})(e::ϵ_Greedy{T}) where T<:AbstractFloat = e

# ╔═╡ 3a215c95-c595-4837-a842-1c1e1c6bfa3b
(::Type{Explorer{T}})(e::UCB{T}) where T<:AbstractFloat = e

# ╔═╡ a04da367-3f8d-422d-a443-4e3e666e30ef
(::Type{Explorer{T}})(e::UCB) where T<:AbstractFloat = UCB{T}(e)

# ╔═╡ fb3381b5-10e3-4307-b76d-672245fac9e7
(::Type{AverageMethod{T}})(a::SampleAverage) where T<:AbstractFloat = SampleAverage{T}()

# ╔═╡ ff6598fa-3366-416c-88a1-6bfcefeb1719
(::Type{AverageMethod{T}})(a::ConstantStep) where T<:AbstractFloat = ConstantStep{T}(a)

# ╔═╡ d2ebd908-387d-4e40-bc00-61ce5f45ebdd
(::Type{AverageMethod{T}})(a::UnbiasedConstantStep) where T<:AbstractFloat = UnbiasedConstantStep{T}(a)

# ╔═╡ 13f0adab-7660-49df-b26d-5f89cd73192b
struct ActionValue{T<:AbstractFloat} <: BanditAlgorithm{T}
	N::Vector{T}
	Q::Vector{T}
	explorer::Explorer{T}
	q_avg::AverageMethod{T}
	#use the type of Qinit to initialize vectors and convert the other types if necessary
	function ActionValue(k::Integer, Qinit::T, explorer::Explorer, q_avg::AverageMethod) where {T<:AbstractFloat}
		N = zeros(T, k)
		Q = ones(T, k) .* Qinit
		new_e = Explorer{T}(explorer)
		new_avg = AverageMethod{T}(q_avg)
		new{T}(N, Q, new_e, new_avg)
	end
end

# ╔═╡ 4ce233ba-383e-48c5-a8af-b447b7f46f5f
est1 = ActionValue(10, 1.0f0, ϵ_Greedy(), SampleAverage())

# ╔═╡ 0fbbe455-79ce-44d6-b010-da0bb56adbb4
(::Type{ActionValue})(k::Integer; Qinit = 0.0, explorer::Explorer = ϵ_Greedy(), q_avg::AverageMethod = SampleAverage()) = ActionValue(k, Qinit, explorer, q_avg)

# ╔═╡ a03e2617-048e-44f8-8cf6-a2416190d768
ActionValue(10, Qinit = 0.0f0)

# ╔═╡ 0368db15-f875-45d1-8598-fe53732c58cc
est2 = ActionValue(10, Qinit = BigFloat(1.0), explorer = UCB(1.0f0), q_avg = ConstantStep())

# ╔═╡ 8a04adab-e97e-4ac4-a85e-5eae93b1c37b
mutable struct GradientReward{T<:AbstractFloat} <: BanditAlgorithm{T}
	H::Vector{T} 
	expH::Vector{T}
	πvec::Vector{T}
	α::T
	R̄::T
	update::AverageMethod{T}
	function GradientReward(k::Integer, α::T, update::AverageMethod) where T<:AbstractFloat
		H = zeros(T, k)
		expH = exp.(H)
		πvec = ones(T, k) ./ k
		R̄ = zero(T)
		new_update = AverageMethod{T}(update)
		new{T}(H, expH, πvec, α, R̄, new_update)
	end
end

# ╔═╡ 69b560c1-98ad-4cbf-89d2-e0516299bc69
(::Type{GradientReward})(k::Integer; α::T=0.1, update::AverageMethod = SampleAverage()) where T<:AbstractFloat = GradientReward(k, α, update)

# ╔═╡ 0785c5d2-e2eb-4176-9b04-7386dc3de82f
grad_est1 = GradientReward(10)

# ╔═╡ 003a6b9d-8e89-4063-8c56-4a15d1da2110
grad_est2 = GradientReward(10, update = ConstantStep(0.0f0))

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
	expH = exp.(H)
	π_vec = calc_πvec(expH)
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
        a = sample_action(actions, π_vec)
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
		update_H!(a, H, π_vec, α, step_reward[i], R̄)

		#update R̄ with running average if baseline is true
		if baseline
			R̄ += (1.0/i)*(step_reward[i] - R̄)
		end

		#update expH and π_vec
		expH .= exp.(H)
		π_vec .= calc_πvec(expH)
    end
    return (;step_reward, step_reward_ideal, cum_reward, cum_reward_ideal, optimalstep, optimalaction_pct)
end

# ╔═╡ 1f883942-89af-4a05-9455-828d43b860d4
gradient_stationary_bandit_algorithm(create_bandit(k), k)

# ╔═╡ 50fbdc85-82f1-4c52-936b-84eb14951d71
function average_gradient_stationary_runs(k; steps = 1000, n = 2000, α=0.1, offset = 0.0, baseline = true)
    runs = Vector{NamedTuple}(undef, n)
    @threads for i in 1:n
        qs = create_bandit(k, offset = offset)
        runs[i] = gradient_stationary_bandit_algorithm(qs, k, steps = steps, α = α, baseline = baseline) 
    end
    map(i -> mapreduce(a -> a[i], (a, b) -> a .+ b, runs)./n, (:step_reward, :step_reward_ideal, :optimalstep, :cum_reward, :cum_reward_ideal, :optimalaction_pct))
end

# ╔═╡ d75f4005-e905-4546-9839-75aed9d505e3
begin
	stationary_gradient_results = [average_gradient_stationary_runs(10, α=α, offset = 4.0) for α in α_list]
	plot(mapreduce(a -> a[3], hcat, stationary_gradient_results), lab = mapreduce(a -> "α = $a", hcat, α_list), ylabel = "% Runs Taking Optimal Action", xlabel = "Step", size = (700, 450), title = "Gradient Bandit With Baseline")
end

# ╔═╡ f3672a55-1de2-4925-9aa7-1865e6a5c64a
begin
	stationary_gradient_results2 = [average_gradient_stationary_runs(10, α=α, offset = 4.0, baseline = false) for α in α_list]
	plot(mapreduce(a -> a[3], hcat, stationary_gradient_results2), lab = mapreduce(a -> "α = $a", hcat, α_list), ylabel = "% Runs Taking Optimal Action", xlabel = "Step", size = (700, 450), title = "Gradient Bandit Without Baseline")
end

# ╔═╡ 555860b6-4ae6-411f-94d7-5c30efc5c339
with_terminal() do
	actions = collect(1:10)
	est = ActionValue(10)
	# @code_warntype sample_action(est, 10, actions)
	@benchmark sample_action($est, 10, $actions)
end

# ╔═╡ fb980aba-a703-44b8-a633-b59a085eee1b
with_terminal() do
	actions = collect(1:10)
	est = ActionValue(10, explorer = UCB())
	# @code_warntype sample_action(est, 10, actions)
	@benchmark sample_action($est, 10, $actions)
end

# ╔═╡ 32a4af32-c645-4342-b09a-6f4d964e046a
with_terminal() do
	actions = collect(1:10)
	est = GradientReward(10)
	# @code_warntype sample_action(est, 10, actions)
	@benchmark sample_action($est, 10, $actions)
end

# ╔═╡ e1909e7e-b691-4660-af5d-a7d72026195f
function update_estimator!(est::GradientReward{T}, a::Integer, r::T, step::Integer) where T <: AbstractFloat
	(H, expH, πvec, α, R̄, r_avg) = (est.H, est.expH, est.πvec, est.α, est.R̄, est.update)
	rdiff = r - R̄
	c = α*rdiff
	for i in eachindex(H)
		H[i] += c*(i == a ? (one(T) - πvec[i]) : -πvec[i])	
	end	
	#update R̄ with desired method
	updatecoef(::SampleAverage) = one(T)/step
	updatecoef(r_avg::ConstantStep) = r_avg.α
	function updatecoef(r_avg::UnbiasedConstantStep) 
		r_avg.o = r_avg.o + r_avg.α*(one(T) - r_avg.o)
		r_avg.α/r_avg.o
	end
	est.R̄ = R̄ + updatecoef(r_avg)*rdiff

	#update expH and π_vec
	expH .= exp.(H)
	πvec .= calc_πvec(expH)
end	

# ╔═╡ 7a034d15-60e4-4082-9383-6685c8561e33
function update_estimator!(est::ActionValue{T}, a::Integer, r::T, i::Integer) where T <: AbstractFloat
	(N, Q, q_avg) = (est.N, est.Q, est.q_avg)
	N[a] += one(T)
 	updatecoef(q_avg::SampleAverage) = one(T) / N[a]
	updatecoef(q_avg::ConstantStep) = q_avg.α
	function updatecoef(r_avg::UnbiasedConstantStep) 
		r_avg.o = r_avg.o + r_avg.α*(one(T) - r_avg.o)
		r_avg.α/r_avg.o
	end
	c = updatecoef(q_avg)
	Q[a] += c*(r - Q[a])
end	

# ╔═╡ 1004eb4b-1fed-4328-a08b-6f5d9dd5080b
function run_bandit(qs::Vector{T}, algorithm::BanditAlgorithm{T}; steps = 1000, μ::T = zero(T), σ::T = zero(T)) where T <: AbstractFloat 
	bandit(a) = sample_bandit(a, qs)
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
    cum_reward_ideal = zeros(T, steps)
    step_reward_ideal = zeros(T, steps)
    cum_reward = zeros(T, steps)
    step_reward = zeros(T, steps)
    optimalstep = fill(false, steps)
    optimalcount = 0
    optimalaction_pct = zeros(T, steps)
	bestaction = argmax(qs)
    for i = 1:steps
        a = sample_action(algorithm, i, actions)
        # a = 1
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
		#update anything required before sampling the next action
		update_estimator!(algorithm, a, step_reward[i], i)
		#will only update qs in the non-stationary case and get a new bestaction
		if updateq				
			qupdate!(qs)
			bestaction = argmax(qs)
		end
    end
    return (;step_reward, step_reward_ideal, cum_reward, cum_reward_ideal, optimalstep, optimalaction_pct)
end

# ╔═╡ 5915e7b4-bb03-4e29-a5bc-5cf2a6cb9f3d
function run_bandit_cumreward(qs::Vector{T}, algorithm::BanditAlgorithm{T}; steps = 1000, μ::T = zero(T), σ::T = zero(T), cumstart = 1) where T <: AbstractFloat
#same as run_bandit but only saved the average cumulative reward per step, so it is faster
	bandit(a) = sample_bandit(a, qs)
	#in this case the bandit is not stationary
	updateq = (μ != 0) || (σ != 0)
	function qupdate!(qs)
		for i in eachindex(qs)
			qs[i] += (randn(T)*σ) + μ
		end
		return nothing
	end
	#initialize values to keep track of
	actions = collect(eachindex(qs))
	accum_reward_ideal = zero(T)
    accum_reward = zero(T)
	bestaction = argmax(qs)
    for i = 1:steps
        a = sample_action(algorithm, i, actions)
        r = bandit(a) 
        r_ideal = bandit(bestaction)
        if i >= cumstart
			accum_reward_ideal += r_ideal 
        	accum_reward += r 
		end
		#update anything required before sampling the next action
		update_estimator!(algorithm, a, r, i)
		#will only update qs in the non-stationary case and get a new bestaction
		if updateq				
			qupdate!(qs)
			bestaction = argmax(qs)
		end
    end
	step_cum_reward = accum_reward/(steps - cumstart + 1)
	step_cum_reward_ideal = accum_reward_ideal/(steps - cumstart + 1)
    return (;step_cum_reward, step_cum_reward_ideal)
end

# ╔═╡ 21956a94-c846-4785-b081-2303bf90abcc
with_terminal() do
	actions = collect(1:10)
	est = ActionValue(10)
	# @code_warntype update_estimator!(est, 5, 1.0, 10)
	@benchmark update_estimator!($est, 5, 1.0, 10)
end

# ╔═╡ bd76e270-21c1-47f0-9924-8a8f8da48d00
with_terminal() do
	actions = collect(1:10)
	est = GradientReward(10)
	# @code_warntype update_estimator!(est, 5, 1.0, 10)
	@benchmark update_estimator!($est, 5, 1.0, 10)
end

# ╔═╡ 6af63fff-e58a-4868-a448-e08205cdd5b9
with_terminal() do 
	bandit = create_bandit(k)
	estimator = ActionValue(k, q_avg=ConstantStep())
	@benchmark run_bandit($bandit, $estimator)
end

# ╔═╡ a94d9e8f-a20d-4a46-bab4-83235592e198
with_terminal() do 
	bandit = create_bandit(k)
	@benchmark simple_algorithm($bandit, $k, 0.1)
end

# ╔═╡ 7678b06b-feef-4656-a801-33f630437bfb
function average_stationary_runs(k, algorithm; steps = 1000, n = 2000, offset = 0.0)
    runs = Vector{NamedTuple}(undef, n)
    @threads for i in 1:n
        qs = create_bandit(k, offset=offset)
		est = algorithm(k)
        runs[i] = run_bandit(qs, est, steps = steps) 
    end
    map(i -> mapreduce(a -> a[i], (a, b) -> a .+ b, runs)./n, (:step_reward, :step_reward_ideal, :optimalstep, :cum_reward, :cum_reward_ideal, :optimalaction_pct))
end

# ╔═╡ 8a52acf7-5d57-490f-8bd9-6e1e0e322872
function average_stationary_runs_cum_reward(k, algorithm; steps = 1000, n = 2000, offset::T = 0.0) where T <: AbstractFloat
    r_step = Atomic{T}(zero(T))
	r_step_ideal = Atomic{T}(zero(T))
    @threads for i in 1:n
        qs = create_bandit(k, offset=offset)
		est = algorithm(k)
        rewards = run_bandit_cumreward(qs, est, steps = steps) 
    	atomic_add!(r_step, rewards[1])
		atomic_add!(r_step_ideal, rewards[2])
	end
    (r_step[]/n, r_step_ideal[]/n)
end

# ╔═╡ 6a90f167-15a8-4269-80e4-ea7206d90470
function average_nonstationary_runs(k, algorithm; steps = 10000, n = 2000, qinit::T=0.0) where T<:AbstractFloat
    runs = Vector{NamedTuple}(undef, n)
    @threads for i in 1:n
        qs = ones(T, k) .* qinit
		est = algorithm(k)
        runs[i] = run_bandit(qs, est, steps = steps, σ=0.01) 
    end
    map(i -> mapreduce(a -> a[i], (a, b) -> a .+ b, runs)./n, (:step_reward, :step_reward_ideal, :optimalstep, :cum_reward, :cum_reward_ideal, :optimalaction_pct))
end

# ╔═╡ 24713318-8d43-4d92-a2e6-586414fc29e2
sample_average_run = average_nonstationary_runs(10, 0.1, 0.0)

# ╔═╡ a92f05b7-311f-4267-9943-2b0af0dc895b
constant_step_update_run = average_nonstationary_runs(10, 0.1, 0.1)

# ╔═╡ 59fcfd80-54b9-4e83-a1ed-7c7a5e7fac9a
plot([sample_average_run[1] constant_step_update_run[1] sample_average_run[3]], lab  = ["Sample Average" "α = 0.1" "Theoretical Limit"], ylabel = "Reward Averaged Over Runs", xlabel = "Step", size = (700, 400))

# ╔═╡ c8954a17-a94e-4c4a-86d0-217e322b4c30
plot([sample_average_run[2] constant_step_update_run[2]], lab  = ["Sample Average" "α = 0.1"], ylabel = "% Runs Taking Optimal Action", xlabel = "Step", size = (700, 400))

# ╔═╡ 041e54ab-c7ee-4fe8-96ec-1189a3b18380
function average_nonstationary_runs_cum_reward(k, algorithm; steps = 10000, n = 2000, qinit::T = 0.0) where T<:AbstractFloat
    r_step = Atomic{T}(zero(T))
	r_step_ideal = Atomic{T}(zero(T))
    @threads for i in 1:n
        qs = ones(T, k) .* qinit
		est = algorithm(k)
        rewards = run_bandit_cumreward(qs, est, steps = steps, σ=0.01, cumstart = floor(Int64, steps/2) + 1) 
    	atomic_add!(r_step, rewards[1])
		atomic_add!(r_step_ideal, rewards[2])
	end
    (r_step[]/n, r_step_ideal[]/n)
end

# ╔═╡ 68b09fc5-e0d2-4c58-a382-83fe2a058501
with_terminal() do 
	@benchmark average_stationary_runs_cum_reward(k, k -> ActionValue(k))
end

# ╔═╡ 95bfaee5-05da-4c4c-81d2-5f8aac98fb3b
with_terminal() do
#repeat benchmark with Float32 to see how much faster it is
	@benchmark average_stationary_runs_cum_reward(k, k -> ActionValue(k, Qinit = 0.0f0), offset = 0.0f0)
end

# ╔═╡ b6e3eca8-f174-4889-b801-2f69c801f8d1
nonstationaryruns = average_nonstationary_runs(k, k -> ActionValue(k, explorer = ϵ_Greedy(1/128), q_avg = ConstantStep()), steps = 100000)

# ╔═╡ 72e77d4f-229a-4ac1-a023-38059e34740a
plot(nonstationaryruns[1], legend = false)

# ╔═╡ c348c234-db5d-4f72-b41f-2e0969b69594
gradient_est = GradientReward(k)

# ╔═╡ 8fda57ef-08a9-4897-b127-a4283b959d06
run_bandit(create_bandit(k), gradient_est)

# ╔═╡ 66b8b455-b2d3-4e5f-beb4-2c7736eb6de7
αlist = [0.025, 0.05, 0.1, 0.2, 0.4]

# ╔═╡ 8585f70b-2f4e-47cd-a101-89ba9a9e16c2
gradientruns_baseline = [average_stationary_runs(k, k -> GradientReward(k, α=α), offset = 4.0) for α in αlist]

# ╔═╡ 99d7fc46-67bf-4b85-b3ee-62a6be79d5d3
gradientruns_nobaseline = [average_stationary_runs(k, k -> GradientReward(k, α=α, update = ConstantStep(0.0)), offset = 4.0) for α in αlist]

# ╔═╡ eb71f799-e192-4c20-b60c-8c523825261c
plot(mapreduce(a -> a[3], hcat, gradientruns_baseline), lab = mapreduce(a -> "α=$a", hcat, αlist), xaxis = "Step", yaxis = "% Runs Taking Optimal Action", title="Gradient Algorithm with Baseline and 4.0 Mean Reward", size = (700, 500))

# ╔═╡ 9271bbe3-f8f0-4223-ba3a-35575f5d9725
plot(mapreduce(a -> a[3], hcat, gradientruns_nobaseline), lab = mapreduce(a -> "α=$a", hcat, αlist), xaxis = "Step", yaxis = "% Runs Taking Optimal Action", title="Gradient Algorithm with No Baseline and 4.0 Mean Reward", size = (700, 500))

# ╔═╡ 1f9a98fd-ea29-415c-9f35-add34b513a34
md"""
>*Expercise 2.10* Suppose you face a 2-armed bandit task whose true action values change randomly from time step to time step. Specifically, suppose that, for any time step, the true values of actions 1 and 2 are respectively 10 and 20 with probability 0.5 (case A), and 90 and 80 with probability 0.5 (case B). If you are not able to tell which case you face at any step, what is the best expected reward you can achieve and how should you behave to achieve it? Now suppose that on each step you are told whether you are facing case A or case B (although you still don’t know the true action values). This is an associative search task. What is the best expected reward you can achieve in this task, and how should you behave to achieve it?

For the case in which we do not know which case we are facing, we can calculate the expected reward for each action across all cases.

$E[R_1] = 0.5 \times 10 + 0.5 \times 90 = 50$

$E[R_2] = 0.5 \times 20 + 0.5 \times 80 = 50$

Since the expected reward of each action is equal, the best we can do is pick randomly which will have an expected reward of 50.

For the case in which we know if we are in case A or case B, we now can select the best action for each case which has a value of 20 (action 2) for case A and 90 (action 1) for case B.  However, we have a 50% probability of facing each case so our overall expected reward is.

$E[R] = 20 \times 0.5 + 90 \times 0.5 = 55$
"""

# ╔═╡ 46ce2b1f-02cf-4dae-bf02-f67543f38b91
md"""
# Parameter Studies
"""

# ╔═╡ c61630b0-29c1-4183-90d9-57c999187b53
function print_power2(n)
	if abs(n) > 7
		"2^$n"
	elseif n < 0
		"1/$(2^-n)"
	else
		"$(2^n)"
	end
end

# ╔═╡ c72d35fc-c8fd-450e-9b95-d12ece5c2291
function get_param_list(n1::Integer, n2::Integer; base::T = 2.0) where T<:AbstractFloat
	nlist = collect(n1:n2)
	plist = base .^nlist	
	# namelist = print_power2.(nlist)
	return plist, nlist
end

# ╔═╡ 52bb3923-14bc-489e-bbb7-e3d1cff7e783
function stationary_param_search(algorithm, n1, n2; base::T = 2.0) where T<:AbstractFloat
	plist, nlist = get_param_list(n1, n2, base = base)
	runs = Vector{Tuple{T, T}}(undef, length(plist))
	cum_rewards = Vector{T}(undef, length(plist))
	cum_rewards_ideal = similar(cum_rewards)
	for i in eachindex(runs)
		run = average_stationary_runs_cum_reward(k, k -> algorithm(plist[i], k))
		cum_rewards[i] = run[1]
		cum_rewards_ideal[i] = run[2]
	end
	plist, nlist, (cum_rewards, cum_rewards_ideal)
end

# ╔═╡ 57d2fe04-1447-4d0f-a104-3a6034b85a14
function plotstationaryparamsearch(param_search, pname; addplot = false, ideal = false)
	(plist, nlist, runrewards) = param_search
	pltfunc = addplot ? plot! : plot
	rewardindex = ideal ? 2 : 1
	pltfunc(plist, runrewards[rewardindex], xaxis = :log, xticks = (plist, print_power2.(nlist)), lab = pname, yaxis = ("Average Reward over first 1000 steps", [0.5, 1.6]), size = (650, 450))
end

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

# ╔═╡ 2447c4ea-7752-457c-80da-ac0dd72a64c1
save_data(varname, data) = jldsave("$varname.jld2"; data)	

# ╔═╡ 23a09111-a43d-4da0-b1f4-e8907097e31e
ϵ_greedy_stationary_param_search = run_or_load("ϵ_greedy_stationary_param_search", ()-> stationary_param_search((p, k) -> ActionValue(k, explorer = ϵ_Greedy(p)), -7, 2))

# ╔═╡ 6e99a9bc-cec9-4bc2-a7a1-33c33f1f0988
plotstationaryparamsearch(ϵ_greedy_stationary_param_search, "ϵ-greedy")

# ╔═╡ 5c6592ba-91a2-4477-89e3-8d620773722e
plotstationaryparamsearch(ϵ_greedy_stationary_param_search, "ideal", addplot = true, ideal = true)

# ╔═╡ 0543d212-382e-4bbe-8ff3-d4db6eccc2ed
gradient_stationary_param_search = run_or_load("gradient_stationary_param_search", ()->stationary_param_search((p, k) -> 
	GradientReward(k, α=p), -7, 2))

# ╔═╡ 3b19b145-94a0-40c6-ad44-5a6c44ad4c15
plotstationaryparamsearch(gradient_stationary_param_search, "gradient bandit", addplot = true)

# ╔═╡ a1b15d7f-8c7c-4b38-910d-19c62098e37a
UCB_stationary_param_search = run_or_load("UCB_stationary_param_search", ()->stationary_param_search((p, k) -> ActionValue(k, explorer = UCB(p)), -7, 2))

# ╔═╡ 5597e20e-42ec-4470-be86-2fed94760f21
plotstationaryparamsearch(UCB_stationary_param_search, "UCB", addplot = true)

# ╔═╡ aeb7bc99-ef3e-4404-961d-fa5a1bea9d75
optin_stationary_param_search = run_or_load("optin_stationary_param_search", ()->stationary_param_search((p, k) -> ActionValue(k, Qinit = p, explorer = ϵ_Greedy(0.0), q_avg = ConstantStep()), -7, 2))

# ╔═╡ 2a25d0c3-d1d4-47fb-9262-2b328b8daa8b
plotstationaryparamsearch(optin_stationary_param_search, "greedy Qinit", addplot = true)

# ╔═╡ d0111453-9a66-411d-9966-fc386d1bdcb7
md"""
>*Exercise 2.11 (programming)* Make a figure analogous to Figure 2.6 for the nonstionary case outlined in Exercise 2.5.  Include the constant-step-size ϵ-greedy algorithm with α=0.1.  Use runs of 200,000 steps and, as a performance measure for each algorithm and parameter setting, use the average reward over the last 100,000 steps.
"""

# ╔═╡ 7117b87b-1570-4074-9eb2-a8931be341c9
numsteps = 200000

# ╔═╡ 51e40520-c6c4-41c3-b59e-b005012b9c9a
function param_search(run_function, algorithm, n1, n2; steps = 1000, base::T = 2.0) where T<:AbstractFloat
	(plist, nlist) = get_param_list(n1, n2, base=base)
	cum_rewards = Vector{T}(undef, length(plist))
	cum_rewards_ideal = similar(cum_rewards)
	for i in eachindex(plist)
		run = run_function(k, k -> algorithm(plist[i], k), steps = steps)
		cum_rewards[i] = run[1]
		cum_rewards_ideal[i] = run[2]
	end
	(plist, nlist, (cum_rewards, cum_rewards_ideal), steps)
end

# ╔═╡ 1770e92e-1056-4c7d-8a03-f9fe5cb7fb07
function plotnonstationaryparamsearch(param_search, pname; addplot = false, ideal = false)
	(plist, nlist, runrewards, steps) = param_search
	pltfunc = addplot ? plot! : plot
	rewardindex = ideal ? 2 : 1
	pltfunc(plist, runrewards[rewardindex], xaxis = :log, xticks = (plist, print_power2.(nlist)), lab = pname, yaxis = ("Average Reward over last $(floor(Int64, steps/2)) steps",), size = (700, 450))
end

# ╔═╡ 77f06b7a-6fd5-4b7a-8895-a8b32d72bea5
ϵ_greedy_nonstationary_paramsearch = run_or_load("ϵ_greedy_nonstationary_paramsearch", () -> param_search(average_nonstationary_runs_cum_reward, (p, k) -> ActionValue(k, explorer = ϵ_Greedy(p), q_avg = ConstantStep()), -14, -1, steps = numsteps))

# ╔═╡ 03dc9aa3-6750-45cf-a500-5f5274936410
plotnonstationaryparamsearch(ϵ_greedy_nonstationary_paramsearch, "ϵ-greedy")

# ╔═╡ ec2abc84-876c-49ca-be8a-5cba5af30367
ϵ_greedy_unbiased_nonstationary_paramsearch = run_or_load("ϵ_greedy_unbiased_nonstationary_paramsearch", () -> param_search(average_nonstationary_runs_cum_reward, (p, k) -> ActionValue(k, explorer = ϵ_Greedy(p), q_avg = UnbiasedConstantStep()), -14, -1, steps = numsteps))

# ╔═╡ 0988b634-2bd8-464c-a66f-9adc4da7c9fe
plotnonstationaryparamsearch(ϵ_greedy_unbiased_nonstationary_paramsearch, "ϵ-greedy-unbiased", addplot=true)

# ╔═╡ 4a17fb7c-ffdc-4cc6-97bc-35873c17c6ee
gradient_nonstationary_paramsearch = run_or_load("gradient_nonstationary_paramsearch", () ->  param_search(average_nonstationary_runs_cum_reward, (p, k) -> GradientReward(k, α=p, update = ConstantStep()), -16, -8, steps = numsteps))

# ╔═╡ 52d496bf-040b-436f-8126-9853abc46ebc
plotnonstationaryparamsearch(gradient_nonstationary_paramsearch, "gradient", addplot=true)

# ╔═╡ 283a2b3e-b1c2-41ff-bd29-cd15a321cc68
UCB_nonstationary_paramsearch = run_or_load("UCB_nonstationary_paramsearch", () -> param_search(average_nonstationary_runs_cum_reward, (p, k) -> ActionValue(k, explorer = UCB(p), q_avg = ConstantStep()), 2, 9, steps = numsteps))

# ╔═╡ 1fca9780-3a2d-4417-af09-1668f8df4c93
plotnonstationaryparamsearch(UCB_nonstationary_paramsearch, "UCB", addplot=true)

# ╔═╡ ec7f615a-47ef-492b-9c07-4d2d7f69fbcc
optinit_nonstationary_paramsearch = run_or_load("optinit_nonstationary_paramsearch", () -> param_search(average_nonstationary_runs_cum_reward, (p, k) -> ActionValue(k, Qinit = p, explorer = ϵ_Greedy(0.0), q_avg = ConstantStep()), -16, 9, steps = numsteps))

# ╔═╡ b13ad1ed-a617-48c8-94aa-4308b5a38e1b
plotnonstationaryparamsearch(optinit_nonstationary_paramsearch, "Qinit", addplot=true)

# ╔═╡ ae87fb9b-6829-46cb-aad4-5d292280519c
plotnonstationaryparamsearch(optinit_nonstationary_paramsearch, "ideal", addplot=true, ideal=true)

# ╔═╡ 37446874-1c28-491b-b3cc-b4ad3282686e
md"""
Recreation of Figure 2.6 for the non-stationary case.  In all cases where average values are updated, a constant step size of α=0.1 is used.  The parameters that vary along the x-axis correspond to the algorithms as follows: ϵ-greedy -> ϵ, gradient -> α, UCB -> c, Qinit -> initial Q estimate.  Finally the ideal reward if the optimal action is selected each step is also shown.  Unlike in the stationary case, the ϵ-greedy method with α=0.1 for updating the Q values performs the best at a very small ϵ value of $2^{-8}$.  The UCB is a close second but requires a very large c value of 64 compared to ~1 for the stationary case in which it was the best performer.  
	"""

# ╔═╡ 36602c38-8b29-4158-b299-94015a333762
md"""
# Dependencies and Settings
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
BenchmarkTools = "~1.3.2"
JLD2 = "~0.4.33"
LaTeXStrings = "~1.3.0"
Latexify = "~0.16.1"
PlutoPlotly = "~0.3.9"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.52"
StatsBase = "~0.34.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0-beta2"
manifest_format = "2.0"
project_hash = "fe3fe2cfba07872a4bd9a0a8b71bc7356f2b990f"

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

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

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
git-tree-sha1 = "e460f044ca8b99be31d35fe54fc33a5c33dd8ed7"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.9.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+1"

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
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

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

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

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

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "aa6ffef1fd85657f4999030c52eaeec22a279738"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.33"

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
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.0.1+1"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

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
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

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
version = "1.10.0"

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
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

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

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

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
# ╟─db30e6c0-36bb-4602-a257-5768f3833525
# ╟─31d8fba9-cd28-4b2b-ba46-70a068b9ecad
# ╟─1e4ac085-7b72-4bad-ad87-21635930a6f7
# ╟─083c721c-70dd-4ca3-8160-fc0b0531914f
# ╠═97c94391-397d-4cf2-88b1-3bea3af56ed3
# ╠═7950e06b-e8ce-4bd7-9681-ab7b66dfec69
# ╟─3e5a226e-ecdb-43fb-a40a-a262da0ae542
# ╟─0b951e6e-4b97-4bb5-87d0-6be7f0fd4802
# ╟─2e46853d-f384-49fb-bccd-2fb320444080
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
# ╟─2bce1b80-2133-40a0-9367-fc2d491f6245
# ╟─4be9e81b-c3de-4c79-97b5-b41c03e0f187
# ╟─e07a27c5-0c9a-4893-a1cf-cf565ab78761
# ╟─86350532-13f8-4035-bad8-f25f41c93163
# ╟─78c45162-dc8e-4faf-b1a9-8c71be86dcee
# ╟─eed2a4f2-48b2-4684-846e-aa99bb6dafd9
# ╠═fbe3dc15-9171-4a7e-8eea-b8cc052c9ba5
# ╟─647ab36b-641e-4024-ad2d-40ff33be28f4
# ╠═24713318-8d43-4d92-a2e6-586414fc29e2
# ╠═a92f05b7-311f-4267-9943-2b0af0dc895b
# ╟─59fcfd80-54b9-4e83-a1ed-7c7a5e7fac9a
# ╟─c8954a17-a94e-4c4a-86d0-217e322b4c30
# ╟─32bff269-e893-4907-b589-7ba2ae1314bd
# ╠═c37922b9-6c23-4701-8499-9bb73ccc3684
# ╠═7208da59-282b-4c27-8b0b-460540afb5af
# ╟─0b39196d-a9c4-4d17-a7ab-b0d7264b07fa
# ╟─d4ce45ae-613e-41ee-b626-69b0dbcf6452
# ╟─cb93c588-3dfa-45f4-9d83-f2de26cb1cea
# ╟─23b99305-c8d9-4129-85fb-a5e4aabc4a31
# ╠═093f312b-d70d-4bf7-bd53-8a1c7b2bee31
# ╠═3c58fbe9-38d1-40d2-a86d-3c328c0da517
# ╠═53a7b41e-e6b1-4da7-8130-27627ee9af2c
# ╟─f19dd53e-5ca1-4c8d-a04d-ea436394c089
# ╟─f88029d6-3fc2-4552-8441-5ef37ac42638
# ╟─b24a92fc-f6c6-44e4-9afc-fa4249e4ab83
# ╟─9d7782f5-b530-40d5-9f75-280d3a762216
# ╠═54deaa09-8f87-4caf-b2a0-f15bcd5b40a5
# ╠═ea6d7cad-47ad-4472-a9e9-1ee33c81058d
# ╠═9071a52c-a0a3-4262-93e7-bb8ff93bf44f
# ╠═51349e41-4696-4bd5-9bc1-cefbb82bea08
# ╠═f7519adc-7dfb-4030-86f0-7445699dd3db
# ╠═1f883942-89af-4a05-9455-828d43b860d4
# ╠═50fbdc85-82f1-4c52-936b-84eb14951d71
# ╠═04e5a0db-f47b-46a0-bf98-eeafce87a44b
# ╟─c30715fe-8650-46ad-ba8d-30d34ead569a
# ╟─d75f4005-e905-4546-9839-75aed9d505e3
# ╟─f3672a55-1de2-4925-9aa7-1865e6a5c64a
# ╟─1c9b54cd-08dd-401e-9705-818741844e8d
# ╠═1004eb4b-1fed-4328-a08b-6f5d9dd5080b
# ╠═5915e7b4-bb03-4e29-a5bc-5cf2a6cb9f3d
# ╠═c2347999-5ade-420b-903f-30523b38eb0f
# ╠═f995d0af-50bc-4e33-9bbf-17a7ab06358a
# ╠═852df31d-18d8-466c-8225-e06ba7f05e96
# ╠═44b9ff95-ea3d-41f5-8098-445a263738a9
# ╠═ca726a9d-364d-48e2-8882-20ddbc85b664
# ╠═9d36934a-78cb-446b-b3db-1bbd88cf272d
# ╠═926b95b1-c188-4cfb-8272-10c3a3b9f8e5
# ╠═53fca00d-69b8-42aa-aeec-41de02f553a3
# ╠═fba8da74-4a34-49ad-a60e-e1849d138cc8
# ╠═bae0b1b4-149e-416c-9b92-1cf8ebde07a3
# ╠═cabfd2b9-307e-4026-a3ac-91d2674c58af
# ╠═68f7540b-a6d0-49f2-8d34-8b96888e3109
# ╠═60b2079e-0efa-427e-93cf-7f4646fe202e
# ╠═3118e102-aeac-42d9-98fc-ca29f40be4cd
# ╠═8c0f06f7-2ed0-4f3a-ab4e-90ac142f0cd9
# ╠═a61c15eb-ed5f-4052-a3a3-3276940564a1
# ╠═47be3ae6-20f7-47d0-aae3-b67154afc1a8
# ╠═3c4462d4-c1ab-4231-8c9e-75ab33734061
# ╠═2a70e0cb-311f-4ad4-b55d-77d299030d9c
# ╠═5dac18d3-dcaa-47ce-b050-09357dc41502
# ╠═33930d1a-ad7b-4359-b46d-fe84d47b16dc
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
# ╠═f44d8016-984f-43c1-8541-0bcc74adf735
# ╠═4d105237-947f-429a-be12-ea73e0ce362c
# ╠═c9c9c147-be36-4014-be1e-4e72e325f611
# ╠═01929fe4-0ab7-4e5f-89cf-ade442c4f1a3
# ╠═f561b0a8-a086-4e1a-bc87-82c4205e89c9
# ╠═30aa1e1b-0b51-40c4-a093-ef92c3ad519a
# ╠═0e606680-dd65-444f-bc98-73de4abbcdd4
# ╠═4ebd4a5a-3bd1-48e2-b03e-a5a3b2ec18a1
# ╠═3f8d5fc9-a153-4832-bd48-387679757a45
# ╠═6d3076e4-8644-4436-9c95-480093689dc9
# ╠═1e1f6d10-1b31-4e0a-96f7-23207e913154
# ╠═73e5b719-8b91-41d2-b83d-471d981b027f
# ╠═49e45202-b9ae-42ab-9575-a57edb626a20
# ╠═3a215c95-c595-4837-a842-1c1e1c6bfa3b
# ╠═a04da367-3f8d-422d-a443-4e3e666e30ef
# ╠═fb3381b5-10e3-4307-b76d-672245fac9e7
# ╠═ff6598fa-3366-416c-88a1-6bfcefeb1719
# ╠═d2ebd908-387d-4e40-bc00-61ce5f45ebdd
# ╠═13f0adab-7660-49df-b26d-5f89cd73192b
# ╠═4ce233ba-383e-48c5-a8af-b447b7f46f5f
# ╠═0fbbe455-79ce-44d6-b010-da0bb56adbb4
# ╠═a03e2617-048e-44f8-8cf6-a2416190d768
# ╠═0368db15-f875-45d1-8598-fe53732c58cc
# ╠═8a04adab-e97e-4ac4-a85e-5eae93b1c37b
# ╠═69b560c1-98ad-4cbf-89d2-e0516299bc69
# ╠═0785c5d2-e2eb-4176-9b04-7386dc3de82f
# ╠═003a6b9d-8e89-4063-8c56-4a15d1da2110
# ╠═4982b489-ca15-4188-90e5-565c45f02e01
# ╠═672a91c0-aa77-4257-8c83-d857f47cab6c
# ╠═638f99e6-1cdc-414c-9b67-fd626ec0be3e
# ╠═aa238ebc-8730-46c7-8ad9-41c7cac70b18
# ╠═555860b6-4ae6-411f-94d7-5c30efc5c339
# ╠═fb980aba-a703-44b8-a633-b59a085eee1b
# ╠═32a4af32-c645-4342-b09a-6f4d964e046a
# ╠═e1909e7e-b691-4660-af5d-a7d72026195f
# ╠═7a034d15-60e4-4082-9383-6685c8561e33
# ╠═21956a94-c846-4785-b081-2303bf90abcc
# ╠═bd76e270-21c1-47f0-9924-8a8f8da48d00
# ╠═6af63fff-e58a-4868-a448-e08205cdd5b9
# ╠═a94d9e8f-a20d-4a46-bab4-83235592e198
# ╠═7678b06b-feef-4656-a801-33f630437bfb
# ╠═8a52acf7-5d57-490f-8bd9-6e1e0e322872
# ╠═6a90f167-15a8-4269-80e4-ea7206d90470
# ╠═041e54ab-c7ee-4fe8-96ec-1189a3b18380
# ╠═68b09fc5-e0d2-4c58-a382-83fe2a058501
# ╠═95bfaee5-05da-4c4c-81d2-5f8aac98fb3b
# ╠═b6e3eca8-f174-4889-b801-2f69c801f8d1
# ╠═72e77d4f-229a-4ac1-a023-38059e34740a
# ╠═c348c234-db5d-4f72-b41f-2e0969b69594
# ╠═8fda57ef-08a9-4897-b127-a4283b959d06
# ╠═66b8b455-b2d3-4e5f-beb4-2c7736eb6de7
# ╠═8585f70b-2f4e-47cd-a101-89ba9a9e16c2
# ╠═99d7fc46-67bf-4b85-b3ee-62a6be79d5d3
# ╟─eb71f799-e192-4c20-b60c-8c523825261c
# ╟─9271bbe3-f8f0-4223-ba3a-35575f5d9725
# ╟─1f9a98fd-ea29-415c-9f35-add34b513a34
# ╟─46ce2b1f-02cf-4dae-bf02-f67543f38b91
# ╠═c61630b0-29c1-4183-90d9-57c999187b53
# ╠═c72d35fc-c8fd-450e-9b95-d12ece5c2291
# ╠═52bb3923-14bc-489e-bbb7-e3d1cff7e783
# ╠═57d2fe04-1447-4d0f-a104-3a6034b85a14
# ╠═865610bb-ee82-4440-9f32-f00d0382783b
# ╠═2447c4ea-7752-457c-80da-ac0dd72a64c1
# ╠═23a09111-a43d-4da0-b1f4-e8907097e31e
# ╠═6e99a9bc-cec9-4bc2-a7a1-33c33f1f0988
# ╠═5c6592ba-91a2-4477-89e3-8d620773722e
# ╠═0543d212-382e-4bbe-8ff3-d4db6eccc2ed
# ╠═3b19b145-94a0-40c6-ad44-5a6c44ad4c15
# ╠═a1b15d7f-8c7c-4b38-910d-19c62098e37a
# ╠═5597e20e-42ec-4470-be86-2fed94760f21
# ╠═aeb7bc99-ef3e-4404-961d-fa5a1bea9d75
# ╠═2a25d0c3-d1d4-47fb-9262-2b328b8daa8b
# ╟─d0111453-9a66-411d-9966-fc386d1bdcb7
# ╠═7117b87b-1570-4074-9eb2-a8931be341c9
# ╠═51e40520-c6c4-41c3-b59e-b005012b9c9a
# ╠═1770e92e-1056-4c7d-8a03-f9fe5cb7fb07
# ╠═77f06b7a-6fd5-4b7a-8895-a8b32d72bea5
# ╠═03dc9aa3-6750-45cf-a500-5f5274936410
# ╠═ec2abc84-876c-49ca-be8a-5cba5af30367
# ╠═0988b634-2bd8-464c-a66f-9adc4da7c9fe
# ╠═4a17fb7c-ffdc-4cc6-97bc-35873c17c6ee
# ╠═52d496bf-040b-436f-8126-9853abc46ebc
# ╠═283a2b3e-b1c2-41ff-bd29-cd15a321cc68
# ╠═1fca9780-3a2d-4417-af09-1668f8df4c93
# ╠═ec7f615a-47ef-492b-9c07-4d2d7f69fbcc
# ╠═b13ad1ed-a617-48c8-94aa-4308b5a38e1b
# ╠═ae87fb9b-6829-46cb-aad4-5d292280519c
# ╟─37446874-1c28-491b-b3cc-b4ad3282686e
# ╟─36602c38-8b29-4158-b299-94015a333762
# ╠═1fb1a518-e5ec-4777-80bc-bb55e8172100
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
