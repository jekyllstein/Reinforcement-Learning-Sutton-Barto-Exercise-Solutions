### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ f6809bb4-587e-11ec-3ed8-4d2c83c99fad
begin
	using Random
	using Base.Threads
	using Plots
	using BenchmarkTools
	using PlutoUI
	using Profile
	using JLD2
	plotly()
end

# ╔═╡ 9071a52c-a0a3-4262-93e7-bb8ff93bf44f
#necessary for doing weighted sampling
using StatsBase

# ╔═╡ 1e4ac085-7b72-4bad-ad87-21635930a6f7
md"""
>*Exercise 2.1* In ϵ-greedy action selection, for the case of two actions and $\epsilon = 0.5$, what is the probability that the greedy action is selected?
The probability that the greedy action is selected is a combination of $P(u>\epsilon)$ where u is a sample from $U = uniform(0, 1)$ and the probability of selecting the greedy action at random.  Since both cases are independent, the probabilities can be summed.

$P(a = a_{greedy}) = P_1 + P_2$
$P_1 = P(u>\epsilon)=(1-\epsilon)=0.5$
$P_2=P(a_{rand} =a_{greedy})P(u <\epsilon)=\frac{\text{num greedy actions}}{\text{num total actions}} \times \epsilon = 0.5 \times 0.5 = 0.25$
$P(a = a_{greedy}) = 0.5 + 0.25 = 0.75$ 
"""

# ╔═╡ 083c721c-70dd-4ca3-8160-fc0b0531914f
md"""
# 2.3 The 10-armed Testbed
The following code recreates the 10-armed Testbed from section 2.3
"""

# ╔═╡ 97c94391-397d-4cf2-88b1-3bea3af56ed3
function create_bandit(k::Integer; offset::T = 0.0) where T<:AbstractFloat
    qs = randn(T, k) .+ offset #generate mean rewards for each arm of the bandit 
end

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

# ╔═╡ fc984183-7af6-42b2-9cee-8a2c1c15443c
k = 10

# ╔═╡ 98122c7d-664c-4cb4-957b-18f37ee84d22
simple_algorithm(create_bandit(k), k, 0.1)

# ╔═╡ 08842085-cb76-42d3-8df8-6a1ceaef7b08
ϵ_list = [0.0, 0.01, 0.1]

# ╔═╡ de13a958-25a3-4978-ad5f-cb9ff480fac3
qs = create_bandit(k)

# ╔═╡ 4ac4bfeb-70e1-44bb-a720-6acd85838ee0
runs = [average_simple_runs(k, ϵ) for ϵ in ϵ_list]

# ╔═╡ 6930629f-c35a-4502-893c-a7d519d9be72
md"""
The following two plots recreate Figure 2.2
"""

# ╔═╡ bd6fe118-c49f-41c9-8416-da5dc10c385d
plot(hcat(mapreduce(a -> a[1], hcat, runs), runs[1][2]), lab = ["ϵ=0 (greedy)" "ϵ=0.01" "ϵ=0.1" "Theoretical Limit"], xlabel="Step", ylabel = "Reward Averaged Over Runs", size = (700, 400))

# ╔═╡ ed2ad58b-c441-4512-8128-06f11d9c51c1
plot(mapreduce(a -> a[3], hcat, runs), lab = ["ϵ=0 (greedy)" "ϵ=0.01" "ϵ=0.1"], xlabel="Step", ylabel = "% Runs Taking Optimal Action", size = (700, 400))

# ╔═╡ 965da91b-6a3f-456c-89ce-461c31e0fb7e
md"""
>*Exercise 2.2:* *Bandit example* Consider a k-armed bandit problem with k = 4 actions, denoted 1, 2, 3, and 4. Consider applying to this problem a bandit algorithm using $\epsilon$-greedy action selection, sample-average action-value estimates, and initial estimates of $Q_1(a) = 0$, for all a. Suppose the initial sequence of actions and rewards is $A_1 = 1$, $R_1 = −1$, $A_2 = 2$, $R_2 = 1$, $A_3 = 2$, $R_3 = −2$, $A_4 = 2$, $R_4 = 2$, $A_5 = 3$, $R_5 = 0$. On some of these time steps the $\epsilon$ case may have occurred, causing an action to be selected at random. On which time steps did this definitely occur? On which time steps could this possibly have occurred?

The table below summarizes the actions taken leading into every step and the Q estimate for each action at the end of each step.  So step 0 shows the initial Q estimates of 0 for every action and the selected action 1 that generates the reward on step 1.  For the row in step 1 it shows the Q estimates after receiving the reward on step 1 and thus what actions are demanded by a greedy choice leading into the next step.  If the action selected is not equal to or in the set of greedy actions, then a random action **must** have occured.  Since a random action choice can also select one of the greedy actions, such a random choice is possible at every step and not necessary to list.  Note that the answer in row 0 corresponds to action $A_1$, row 1 -> $A_2$ etc...

|Step|$Q_1$|$Q_2$|$Q_3$ | Action Selected | Greedy Action | Surely Random |
|----|---- |---- | ---- | ---- 			  | ----          | ----          |
|  0 |  0  |  0  | 0    | 1               | 1-3           | no 			  |
|  1 |  -1 |  0  | 0    | 2               | 2-3           | no            |
|  2    |   -1   |  1  | 0| 2 | 2 | no |
| 3  |  -1 |  -.5 | 0  | 2 | 3 | yes |
| 4 | -1  |  0.5  |  0 | 3 | 2 | yes |
| 5 |   -1  |  0.5 |  0 | n/a | n/a | n/a |
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

For each case the long run cumulative reward is just this long term expected reward per step times the number of steps.
"""

# ╔═╡ 86350532-13f8-4035-bad8-f25f41c93163
md"""
2.5 Tracking a Nonstationary Problem
"""

# ╔═╡ 78c45162-dc8e-4faf-b1a9-8c71be86dcee
md"""
>*Exercise 2.4* If the step-size parameters, $\alpha_n$, are not constant, then the estimate $Q_n$ is a weighted average of previously received rewards with a weighting different from that given by (2.6). What is the weighting on each prior reward for the general case, analogous to (2.6), in terms of the sequence of step-size parameters?

From (2.6):  $Q_{n+1} = Q_n + \alpha[R_n - Q_n]$ so here we consider the case where $\alpha$ is not a constant but rather can have a unique value for each step n.  All expressions below are expansions of the right side of the equation.

$=Q_n + \alpha_n[R_n - Q_n]$
$=\alpha_nR_n+(1-\alpha_n)Q_n$
$=\alpha_nR_n+(1-\alpha_n)[\alpha_{n-1}R_{n-1}+(1-\alpha_{n-1})Q_{n-1}]$
$=\alpha_nR_n+(1-\alpha_n)\alpha_{n-1}R_{n-1}+(1-\alpha_n)(1-\alpha_{n-1})Q_{n-1}$
$=\alpha_nR_n+(1-\alpha_n)\alpha_{n-1}R_{n-1}+(1-\alpha_n)(1-\alpha_{n-1})[\alpha_{n-2}R_{n-2}+(1-\alpha_{n-2})Q_{n-2}]$
$=\alpha_nR_n+(1-\alpha_n)\alpha_{n-1}R_{n-1}+(1-\alpha_n)(1-\alpha_{n-1})\alpha_{n-2}R_{n-2}+...$
$=Q_1\prod_{i=1}^n \left( 1-\alpha_i \right)+\sum_{i=1}^{n} \left[ (R_i\alpha_i)\prod_{j=i+1}^n(1-\alpha_j) \right]$

For example if $\alpha_i=1/i$ then the product in the first term is 0 and the formula becomes:

$Q_{n+1} = \sum_{i=1}^{n} \left[ \frac{R_i}{i}\prod_{j=i+1}^n\frac{j-1}{j} \right]$
$= \sum_{i=1}^{n} \left[ \frac{R_i}{i}\frac{i}{i+1}\frac{i+1}{i+2}...\frac{n-1}{n} \right]$
$=\sum_{i=1}^{n} \frac{R_i}{n}$

	
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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Profile = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
BenchmarkTools = "~1.2.2"
JLD2 = "~0.4.16"
Plots = "~1.25.2"
PlutoUI = "~0.7.22"
StatsBase = "~0.33.13"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "abb72771fd8895a7ebd83d5632dc4b989b022b5b"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.2"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "940001114a0147b6e4d10624276d56d531dd9b49"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.2.2"

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
git-tree-sha1 = "4c26b4e9e91ca528ea212927326ece5918a04b47"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.2"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

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
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

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
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

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
git-tree-sha1 = "2db648b6712831ecb333eae76dbfd1c156ca13bb"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.2"

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
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "30f2b340c2fff8410d89bfcdc9c0a6dd661ac5f7"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.62.1"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fd75fa3a2080109a2c0ec9864a6e14c60cca3866"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.62.0+0"

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

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

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

[[deps.JLD2]]
deps = ["DataStructures", "FileIO", "MacroTools", "Mmap", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "5335c4c9a30b4b823d1776d2db09882cbfac9f1e"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.16"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

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
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

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
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

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
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

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
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun"]
git-tree-sha1 = "65ebc27d8c00c84276f14aaf4ff63cbe12016c70"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.25.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "565564f615ba8c4e4f40f5d29784aa50a8f7bbaf"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.22"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

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
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

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
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "0f2aa8e32d511f758a2ce49208181f7733a0936a"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.1.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2bb0cb32026a66037360606510fca5984ccc6b75"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.13"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

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
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

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
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

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
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

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
# ╟─1e4ac085-7b72-4bad-ad87-21635930a6f7
# ╟─083c721c-70dd-4ca3-8160-fc0b0531914f
# ╠═f6809bb4-587e-11ec-3ed8-4d2c83c99fad
# ╠═97c94391-397d-4cf2-88b1-3bea3af56ed3
# ╠═55f89ca1-cd57-44e0-95e5-63a4be31418f
# ╠═3a798da5-c309-48f2-aab1-6602ded8a650
# ╟─33009d31-6d66-4aba-b44c-edd911c2f392
# ╠═fc984183-7af6-42b2-9cee-8a2c1c15443c
# ╠═98122c7d-664c-4cb4-957b-18f37ee84d22
# ╠═08842085-cb76-42d3-8df8-6a1ceaef7b08
# ╠═de13a958-25a3-4978-ad5f-cb9ff480fac3
# ╠═4ac4bfeb-70e1-44bb-a720-6acd85838ee0
# ╟─6930629f-c35a-4502-893c-a7d519d9be72
# ╟─bd6fe118-c49f-41c9-8416-da5dc10c385d
# ╟─ed2ad58b-c441-4512-8128-06f11d9c51c1
# ╟─965da91b-6a3f-456c-89ce-461c31e0fb7e
# ╟─464d43c0-cd59-49e6-88f6-12a767677418
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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
