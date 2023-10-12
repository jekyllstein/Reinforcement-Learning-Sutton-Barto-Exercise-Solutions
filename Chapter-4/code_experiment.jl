using MLStyle, Transducers, LinearAlgebra.BLAS, Base.Threads


@enum Action up down left right

#p should be a dictionary mapping state/action pairs to a distribution over state/reward pairs
function form_gridworld4x4_mdp()
    states = 1:14
    actions = [up, down, left, right]
    rewards = [-1]
    s_term = 0

    sa_pairs = ((s, a) for s in states for a in actions)

    #defines the properties of each of the 4 actions by 1. the state(s) that lead to a terminal state, 2. the states that leave the original state unchanges, and 3. the default state transition 
    actionprops = Dict([
        right => (Set([14]), Set([3, 7, 11]), s -> s + 1),
        left => (Set([1]), Set([4, 8, 12]), s -> s - 1),
        up => (Set([4]), Set([1, 2, 3]), s -> s - 4),
        down => (Set([11]), Set([12, 13, 14]), s -> s + 4)
    ])

    #convert a single state into a distribution over states
    makedist(s) = reshape(Float64.(s .== states), length(states), length(rewards))

    #generate a new state distribution from state action pair, since there is only one reward value, this is just a vector but in general could be a matrix
    function move(s, a)
        (s1, s2, f) = actionprops[a]
        snew =  in(s, s1) ? s_term  :
                in(s, s2) ? s       : 
                f(s)
        makedist(snew)
    end

    # ptr = Dict(s => Dict(a => move(s, a) for a in actions) for s in states)
    ptr = mapreduce(s -> mapreduce(a -> move(s,a), (a,b) -> cat(a,b,dims=4), actions), (a,b) -> cat(a,b,dims=3), states)
    # ptr = foldl((a,b) -> cat(a,b,dims=3), states |> Map(s -> foldl((a,b) -> cat(a,b,dims=4), Map(a -> move(s,a)), actions)))
    # ptr = mapreduce(Map(s -> mapreduce(Map(a -> move(s,a)), (a,b) -> cat(a,b,dims=3), actions)), (a,b) -> cat(a,b,dims=4), states)
    # [[move(s,a) for a in a?catctions] for s in states]
    (ptr = ptr, states = states, actions = actions, rewards = rewards)

    # dists = (move(sa...) for sa in sa_pairs)
    # Dict(zip(sa_pairs, dists))
end

function bellman_value!(V::Vector{Float64}, mdp::NamedTuple, π::Dict, γ::Real, reward_holder::Matrix{Float64})
	delt = 0.0
    l = length(mdp.actions)
    for s in mdp.states
        v = V[s]
        #sum of the value of each action weighted by the probability of taking that action
        # for a in eachindex(mdp.actions)    
        #     reward_holder[:, a] .= mdp.rewards
        #     gemv!('T', γ, view(mdp.ptr, :, :, s, a), V, 1.0, view(reward_holder, :, a))
        # end
        for a in eachindex(mdp.actions)    
            reward_holder[:, a] .= mdp.rewards
            gemv!('T', γ, view(mdp.ptr, :, :, s, a), V, 1.0, view(reward_holder, :, a))
        end
        gemv!('N', 1.0, view(reward_holder, :, 1:l), π[s], 0.0, view(reward_holder, :, l+1))
        V[s] = sum(view(reward_holder, :, l+1))
		delt = max(delt, abs(v - V[s]))
    end
	return delt
end

function iterative_policy_eval_v(π::Dict, θ::Real, mdp::NamedTuple, γ::Real, V::Vector{Float64}, nmax::Real, reward_holder::Matrix{Float64})
	delt = bellman_value!(V, mdp, π, γ, reward_holder)
    if nmax <= 1 || delt <= θ
		return V
	else 
		iterative_policy_eval_v(π, θ, mdp, γ, V, nmax - 1, reward_holder)	
	end
end

function iterative_policy_eval_v(π::Dict, θ::Real, mdp::NamedTuple, γ::Real, Vinit = 0.0; nmax = Inf)
	V = fill(Vinit, length(mdp.states))
    reward_holder = reduce(hcat, (Float64.(mdp.rewards) for _ in 1:length(mdp.actions)+1))
	iterative_policy_eval_v(π, θ, mdp, γ, V, nmax, reward_holder)	
end

#forms a random policy for a generic finite state mdp.  The policy is a dictionary that maps each state to a dictionary of action/probability pairs.
function form_random_policy(mdp)
    l = length(mdp.actions)
    dist = fill(1. /l, l)
    Dict(s => dist for s in mdp.states)
end

function run_4x4gridworld(nmax=Inf)
    gridworldmdp = form_gridworld4x4_mdp()
    π_rand = form_random_policy(gridworldmdp)
    V = iterative_policy_eval_v(π_rand, eps(0.0), gridworldmdp, 1.0, nmax = nmax)
end


function form_gridworld_modified_mdp()
    states = 1:15
    actions = [up, down, left, right]
    rewards = [-1]
    s_term = 0

    sa_pairs = ((s, a) for s in states for a in actions)

    #defines the properties of each of the 4 actions by 1. the state(s) that lead to a terminal state, 2. the states that leave the original state unchanges, and 3. the default state transition 
    actionprops = Dict([
        right => (Set([14]), Set([3, 7, 11]), Dict([15 => 14]), s -> s + 1),
        left => (Set([1]), Set([4, 8, 12]), Dict([15 => 12]), s -> s - 1),
        up => (Set([4]), Set([1, 2, 3]), Dict([15 => 13]), s -> s - 4),
        down => (Set([11]), Set([12, 15, 14]), Dict([13 => 15]), s -> s + 4)
    ])

    #convert a single state into a distribution over states
    makedist(s) = reshape(Float64.(s .== states), length(states), length(rewards))

    #generate a new state distribution from state action pair, since there is only one reward value, this is just a vector but in general could be a matrix
    function move(s, a)
        (s1, s2, s3, f) = actionprops[a]
        snew =  in(s, s1) ? s_term      :
                in(s, s2) ? s           :
                haskey(s3, s) ? s3[s]   : 
                f(s)
        makedist(snew)
    end

    # ptr = Dict(s => Dict(a => move(s, a) for a in actions) for s in states)
    ptr = mapreduce(s -> mapreduce(a -> move(s,a), (a,b) -> cat(a,b,dims=4), actions), (a,b) -> cat(a,b,dims=3), states)
    # ptr = foldl((a,b) -> cat(a,b,dims=3), states |> Map(s -> foldl((a,b) -> cat(a,b,dims=4), Map(a -> move(s,a)), actions)))
    # ptr = mapreduce(Map(s -> mapreduce(Map(a -> move(s,a)), (a,b) -> cat(a,b,dims=3), actions)), (a,b) -> cat(a,b,dims=4), states)
    # [[move(s,a) for a in a?catctions] for s in states]
    (ptr = ptr, states = states, actions = actions, rewards = rewards)

    # dists = (move(sa...) for sa in sa_pairs)
    # Dict(zip(sa_pairs, dists))
end

function run_modifiedgridworld(nmax=Inf)
    mdp = form_gridworld_modified_mdp()
    π_rand = form_random_policy(mdp)
    V = iterative_policy_eval_v(π_rand, eps(0.0), mdp, 1.0, nmax = nmax)
end