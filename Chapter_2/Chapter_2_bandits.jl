using Random
using Base.Threads
using Plots

function create_bandit(k::Integer)
    qs = randn(k) #generate mean rewards for each arm of the bandit 
end

function sample_bandit(a::Integer, qs::Vector{Float64})
    randn() + qs[a] #generate a reward with mean q[a] and variance 1
end

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
            actions[argmax(Q[actions])]
        else
            actions[argmax(Q[actions] .+ (c .* sqrt.(log(i) ./N[actions])))]
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

function average_simple_runs(k, ϵ; steps = 1000, n = 2000, Qinit = 0.0, α=0.0, c = 0.0)
    runs = Vector{NamedTuple}(undef, n)
    @threads for i in 1:n
        qs = create_bandit(k)
        runs[i] = simple_algorithm(qs, k, ϵ, steps = steps, Qinit = Qinit, α = α, c = c) 
    end
    map(i -> mapreduce(a -> a[i], (a, b) -> a .+ b, runs)./n, (:step_reward, :step_reward_ideal, :optimalstep, :cum_reward, :cum_reward_ideal, :optimalaction_pct))
end

k = 10
qs = create_bandit(k)
run1 = average_simple_runs(k, 0.0)
run2 = average_simple_runs(k, 0.01)
run3 = average_simple_runs(k, 0.1)

plot([run1[1] run2[1] run3[1] run1[2]], lab = ["ϵ=0 (greedy)" "ϵ=0.01" "ϵ=0.1" "Theoretical Limit"], xlabel="Step", ylabel = "Reward Averaged Over Runs")
# savefig("stationary_bandit_reward")
# plot([run1[3] run2[3] run3[3]], lab = ["ϵ=0 (greedy)" "ϵ=0.01" "ϵ=0.1"], xlabel="Step", ylabel = "% Runs Taking Optimal Action")
# savefig("stationary_bandit_percent_optimal")

#modification for exercise 2.5
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

function average_nonstationary_runs(k, ϵ, α; n = 2000)
    runs = Vector{NamedTuple}(undef, n)
    @threads for i in 1:n
        runs[i] = nonstationary_algorithm(k, ϵ, α = α) 
    end
    map(i -> mapreduce(a -> a[i], (a, b) -> a .+ b, runs)./n, (:step_reward, :optimalstep, :step_reward_ideal, :cum_reward, :cum_reward_ideal, :optimalaction_pct))
end

# sample_average_run = average_nonstationary_runs(10, 0.1, 0.0)
# constant_step_update_run = average_nonstationary_runs(10, 0.1, 0.1)

# plot([sample_average_run[1] constant_step_update_run[1] sample_average_run[3]], lab  = ["Sample Average" "α = 0.1" "Theoretical Limit"], ylabel = "Reward Averaged Over Runs", xlabel = "Step")
# savefig("non_stationary_rewards")
# plot([sample_average_run[2] constant_step_update_run[2]], lab  = ["Sample Average" "α = 0.1"], ylabel = "% Runs Taking Optimal Action", xlabel = "Step")
# savefig("non_stationary_percent_optimal")


#2.6 Optimistic Initial Values 
optimistic_greedy_runs = average_simple_runs(k, 0.0, Qinit = 5.0, α = 0.1)
realistic_ϵ_runs = average_simple_runs(k, 0.1, α = 0.1)
plot([optimistic_greedy_runs[3] realistic_ϵ_runs[3]], xaxis = "Step", yaxis = ((0, 1), "% Runs Taking Optimal Action"), lab = ["Optimistic, greedy Qinit = 5" "Realistic, ϵ-greedy Qinit = 0, ϵ=0.1"], size = (1100, 800))
savefig("optimistic_initial_value_estimate")

#Exercise 2.6
# The spike occurs on step 11.  Due to the initial Q values it is almost 100% likely that a given run with sample each of the 10 possible actions once before repeating any.  That would mean that at any given step only 10% of the runs would select the optimal action and indeed for the first 10 steps about 10% of the runs are selecting the optimal action as we'd expect from random chance.  On the 11th step, the Q value estimate for each action is 0.9*5 + 0.1*actionreward.  The optimal action for each bandit has the highest mean reward, but there is some chance that one of the other 10 actions produced a higher reward the step it was sampled.  However, on the 11th step, the number of runs that select the optimal action will be equal to the probability that the optimal action produced the highest reward when it was sampled which empirally appears to be slightly over 40%.  Due to the Q value initializationt though, the reward on step 11 for those cases that selected the optimal action will almost certainly lower the Q value estimate for that action below the others resulting in the sudden drop of the optimal action selection on step 11.  

#2.7 Upper-Confidence-Bound Action Selection
c = 5.0
ucb_runs = average_simple_runs(k, 0.0, c = c)
ϵ_runs = average_simple_runs(k, 0.1)
plot([ucb_runs[1] ϵ_runs[1]], lab = ["UCB c=$c" "ϵ-greedy ϵ=0.1"], xaxis = "Step", yaxis = "Reward Averaged Over Runs", size = (1100, 800))
savefig("ucb_vs_ϵ_greedy")

#Exercise 2.8
#Due to the UCB calculation, any state that has not been visited will have an infinite Q value, thus for the first 10 steps similar to the optimistic Q initialization each run will sample each of the 10 possible actions.  One the 11th step, the exploration incentive for each action will be equal, so the most likely action to be selected is the optimal action since it is the most likely to have produced a reward higher than any other action.  If c is very large, then on step 12 no matter how good of a reward we received for the optimal action, that action will be penalized compared to the others because it will have double the visit count.  In particular for c = 2.0, the exploration bonus for the optimal action if selected on step 11 will be 2.2293 vs 3.31527 for all other actions.  Since the q's are normally distributed it is unlikely that the reward average for the optimal action is >1 than the next best action.  The larger c is the more of a relative bonus the other actions have and the probability of selecting the optimal action twice in a row drops to zero.  As c changes the improved reward on step 11 remains similar but the dropoff on step 12 becomes more severe the larger c is.

#2.8 Gradient Bandit Algorithms
