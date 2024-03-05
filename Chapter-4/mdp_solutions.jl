using MLStyle, Transducers, LinearAlgebra.BLAS, Base.Threads

function bellman_value!(V::Vector{Float64}, mdp::NamedTuple, π::Dict, γ::Real, reward_holder::Matrix{Float64})
	delt = 0.0
    l = length(mdp.actions)
    for s in mdp.states
        v = V[s]
        #sum of the value of each action weighted by the probability of taking that action
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

function policy_eval(π::Dict, θ::Real, mdp::NamedTuple, γ::Real, V::Vector{Float64}, nmax::Real, reward_holder::Matrix{Float64})
	delt = bellman_value!(V, mdp, π, γ, reward_holder)
    if nmax <= 1 || delt <= θ
		return V
	else 
		policy_eval(π, θ, mdp, γ, V, nmax - 1, reward_holder)	
	end
end

function policy_eval(π::Dict, θ::Real, mdp::NamedTuple, γ::Real, Vinit = 0.0; nmax = Inf)
	V = fill(Vinit, length(mdp.states))
    reward_holder = reduce(hcat, (Float64.(mdp.rewards) for _ in 1:length(mdp.actions)+1))
	policy_eval(π, θ, mdp, γ, V, nmax, reward_holder)	
end

#forms a random policy for a generic finite state mdp.  The policy is a dictionary that maps each state to a dictionary of action/probability pairs.
function form_random_policy(mdp)
    l = length(mdp.actions)
    dist = fill(1. /l, l)
    Dict(s => dist for s in mdp.states)
end

function policy_improvement(π::Dict, mdp::NamedTuple, γ::Real, V::Vector{Float64}, deterministic)
	(ptr, states, actions, rewards) = mdp
    reward_holder = Float64.(rewards)
    π_new = deepcopy(π)
    policy_stable = true
    for s in states
        old_action = argmax(π[s])
        v = V[s]
        for a in eachindex(actions)    
            gemv!('T', γ, view(mdp.ptr, :, :, s, a), V, 1.0, reward_holder) 
            π_new[s][a] = sum(reward_holder)
            reward_holder .= rewards
        end
        new_action = argmax(π_new[s])
        if deterministic #set the probability to 1 at the most valuable action
            map!(Map(i -> Float64(i == new_action)), π_new[s], eachindex(π_new[s]))
        else #make a proper distribution
            π_new[s] ./= sum(π_new[s])
        end
        (old_action != new_action) && (policy_stable = false)
    end

    return (policy_stable, π_new)
end

function policy_iteration(mdp::NamedTuple, π::Dict, γ::Real, Vold::Vector, iters, θ, evaln, policy_stable, resultlist, deterministic, reward_holder = reduce(hcat, (Float64.(mdp.rewards) for _ in 1:length(mdp.actions)+1)))
	policy_stable && return (true, resultlist)
	V = copy(Vold)
    policy_eval(π, θ, mdp, γ, V, evaln, reward_holder)
	(V == resultlist[end][1]) && return (true, resultlist)
	newresultlist = vcat(resultlist, (V, π))
	(iters <= 0) &&	return (false, newresultlist)
	(new_policy_stable, π_new) = policy_improvement(π, mdp, γ, V, deterministic)
	policy_iteration(mdp, π_new, γ, V, iters-1, θ, evaln, new_policy_stable, newresultlist, deterministic, reward_holder)
end

function begin_policy_iteration(mdp::NamedTuple, π::Dict, γ::Real; iters=Inf, θ=eps(0.0), evaln = Inf, V = policy_eval(π, θ, mdp, γ, nmax = evaln), deterministic=true)
	resultlist = [(V, π)]
	(policy_stable, π_new) = policy_improvement(π, mdp, γ, V, deterministic)
	policy_iteration(mdp, π_new, γ, V, iters-1, θ, evaln, policy_stable, resultlist, deterministic)
end