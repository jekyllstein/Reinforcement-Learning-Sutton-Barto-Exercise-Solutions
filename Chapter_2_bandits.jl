using Random
using Base.Threads
using Plots

function create_bandit(k::Integer)
    qs = randn(k) #generate mean rewards for each arm of the bandit 
end

function sample_bandit(a::Integer, qs::Vector{Float64})
    randn() + qs[a] #generate a reward with mean q[a] and variance 1
end

function simple_algorithm(qs::Vector{Float64}, k::Integer, ϵ::AbstractFloat; steps = 1000)
    bandit(a) = sample_bandit(a, qs)
    Q = zeros(k)
    N = zeros(k)
    accum_reward_ideal = 0.0
    accum_reward = 0.0
    avg_reward_ideal = zeros(steps)
    avg_reward = zeros(steps)
    bestaction = argmax(qs)
    optimalcount = 0
    opitmalaction_pct = zeros(steps)
    actions = collect(1:k)
    for i = 1:steps
        shuffle!(actions) #so that ties are broken randomly with argmax
        a = if rand() < ϵ
            rand(actions)
        else
            actions[argmax(Q[actions])]
        end
        if a == bestaction
            optimalcount += 1
        end
        reward = bandit(a)
        accum_reward_ideal += bandit(bestaction)
        accum_reward += reward
        avg_reward_ideal[i] = accum_reward_ideal/i
        avg_reward[i] = accum_reward/i
        opitmalaction_pct[i] = optimalcount / i
        N[a] += 1.0
        Q[a] += (1.0/N[a])*(reward - Q[a])
    end
    return Q, avg_reward, opitmalaction_pct, avg_reward_ideal 
end

function average_simple_runs(k, ϵ; steps = 1000, n = 2000)
    runs = Vector{Tuple}(undef, n)
    @threads for i in 1:n
        qs = create_bandit(k)
        runs[i] = simple_algorithm(qs, k, ϵ, steps = steps) 
    end
    map(i -> mapreduce(a -> a[i], (a, b) -> a .+ b, runs)./n, (1, 2, 3, 4))
end

k = 10
qs = create_bandit(k)
run1 = average_simple_runs(k, 0.0)
run2 = average_simple_runs(k, 0.01)
run3 = average_simple_runs(k, 0.1)

plot([run1[2] run2[2] run3[2] run1[4]], lab = ["ϵ=0 (greedy)" "ϵ=0.01" "ϵ=0.1" "Theoretical Limit"], xlabel="Steps", ylabel = "Average Reward Per Step")
savefig("stationary_bandit_reward")
plot([run1[3] run2[3] run3[3]], lab = ["ϵ=0 (greedy)" "ϵ=0.01" "ϵ=0.1"], xlabel="Steps", ylabel = "% Optimal Action")
savefig("stationary_bandit_percent_optimal")

#modification for exercise 2.5
function nonstationary_algorithm(k::Integer, ϵ::AbstractFloat; steps = 10000, σ = 0.01, α = 0.0)
    qs = zeros(k)
    Q = zeros(k)
    N = zeros(k)
    accum_reward = 0.0
    accum_reward_ideal = 0.0
    avg_reward_ideal = zeros(steps)
    avg_reward = zeros(steps)
    optimalcount = 0
    opitmalaction_pct = zeros(steps)
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
        end
        bandit(a) = sample_bandit(a, qs)
        reward = bandit(a)
        accum_reward_ideal += bandit(optimalaction) 
        accum_reward += reward
        avg_reward_ideal[i] = accum_reward_ideal/i
        avg_reward[i] = accum_reward/i
        opitmalaction_pct[i] = optimalcount / i
        N[a] += 1.0
        if α == 0.0
            Q[a] += (1.0/N[a])*(reward - Q[a])
        else 
            Q[a] += α*(reward - Q[a])
        end
        qs .+= randn(k) .*σ #update q values with random walk
    end
    return Q, avg_reward, opitmalaction_pct, avg_reward_ideal
end

function average_nonstationary_runs(k, ϵ, α; n = 2000)
    runs = Vector{Tuple}(undef, n)
    @threads for i in 1:n
        runs[i] = nonstationary_algorithm(k, ϵ, α = α) 
    end
    map(i -> mapreduce(a -> a[i], (a, b) -> a .+ b, runs)./n, (1, 2, 3, 4))
end

sample_average_run = average_nonstationary_runs(10, 0.1, 0.0)
constant_step_update_run = average_nonstationary_runs(10, 0.1, 0.1)

plot([sample_average_run[2] constant_step_update_run[2] sample_average_run[4]], lab  = ["Sample Average" "α = 0.1" "Theoretical Limit"], ylabel = "Average Reward Per Step", xlabel = "Steps")
savefig("non_stationary_rewards")
plot([sample_average_run[3] constant_step_update_run[3]], lab  = ["Sample Average" "α = 0.1"], ylabel = "% Optimal Action", xlabel = "Steps")
savefig("non_stationary_percent_optimal")