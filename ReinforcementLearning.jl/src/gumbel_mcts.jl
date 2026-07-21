module GumbelMCTS

using Random: rand
using Statistics: mean
using SparseArrays
using ..ReinforcementLearning

export gumbel_mcts_search, make_gumbel_mcts_policy,
       compute_completed_policy!, extract_root_training_data

"""
    compute_vmix(v_base, visit_counts, Q, policy_logits, num_actions)

Compute the mixed value estimator v_mix from Appendix D of the Gumbel paper.

    v_mix = 1/(1 + total_N) * (v̂_π + total_N * Σ_{a: N(a)>0} π(a) * Q(a) / Σ_{a: N(a)>0} π(a))

Interpolates between the value network estimate (no visits) and the weighted
average of observed Q-values (many visits). Falls back to v_base if no visits.

Returns the v_mix baseline value.
"""
function compute_vmix(
    v_base::T,
    visit_counts::SparseVector{T, Int64},
    Q::SparseVector{T, Int64},
    policy_logits::Vector{T},
    num_actions::Int
) where {T<:Real}
    total_N = sum(visit_counts)
    iszero(total_N) && return v_base

    # Compute π-weighted average of observed Q-values
    π_sum = zero(T)
    q_weighted_sum = zero(T)
    @inbounds for i in visit_counts.nzind
        π_i = exp(policy_logits[i])  # convert logit to probability weight
        π_sum += π_i
        q_weighted_sum += π_i * Q[i]
    end

    π_weighted_q = q_weighted_sum / π_sum
    return (v_base + total_N * π_weighted_q) / (one(T) + total_N)
end


"""
    gumbel_simulate!(visit_counts, Q, v_est_cache, policy_cache, mdp, γ,
                     π_dist!, pscale, s, c_visit, c_scale, v_hold, v_new,
                     σ_vals, step_kwargs, est_kwargs, depth; use_vmix)

Perform a single Gumbel MCTS simulation from state `s` at a non-root node.

Action selection follows Section 5 of "Policy Improvement By Planning with Gumbel":
  1. Compute completed Q-values for all actions (visited: Q[s][a], unvisited: baseline)
  2. Baseline is v_est_cache[s] unless `use_vmix=true`, in which case v_mix is used
  3. σ_completed[a] = (c_visit + Nmax) * c_scale * completedQ[a]
  4. π'_logits[a] = pscale * policy_cache[s][a] + σ_completed[a]
  5. Compute π'(a) = softmax(π'_logits)
  6. Select action: argmax_a (π'(a) - N(a) / (1 + sum(b) N(b)))
  7. Backup: running average

Caches are populated on first visit to each state.
Uses preallocated scratch vectors `v_hold` and `σ_vals` to avoid allocations.
"""
function gumbel_simulate!(
    visit_counts::Dict,
    Q::Dict,
    v_est_cache::Dict,
    policy_cache::Dict,
    mdp, γ::T, π_dist!::Function, min_reward::T, max_reward::T,pscale::T,
    s, c_visit::T, c_scale::T,
    v_hold::Vector{T}, v_new::SparseVector{T, Int64},
    σ_vals::Vector{T},
    step_kwargs::NamedTuple, est_kwargs::NamedTuple,
    depth::Int;
    use_vmix::Bool = false,
    rescale_values::Bool = true
) where {T<:Real}
    # Terminal state check
    mdp.isterm(s) && return zero(T)

    # Expand state if first visit
    if !haskey(visit_counts, s)
        v_est = π_dist!(v_hold, s)  # v_hold gets policy logits, returns v_est
        v_est_cache[s] = clamp(v_est, min_reward, max_reward)
        local policy = copy(v_hold)
        policy_cache[s] = policy
        visit_counts[s] = copy(v_new)
        Q[s] = copy(v_new)
        return v_est_cache[s]
    end

    # Depth limit: return cached value
    depth <= 0 && return v_est_cache[s]

    state_visits = visit_counts[s]
    state_qs = Q[s]
    state_policy = policy_cache[s]
    n_actions = length(state_policy)
    v_base = v_est_cache[s]

    # Compute Nmax and total_N
    Nmax = zero(T)
    total_N = zero(T)
    for n in state_visits.nzval
        Nmax = max(Nmax, T(n))
        total_N += T(n)
    end

    # Determine baseline for unvisited actions
    v_baseline = if use_vmix && total_N > 0
        compute_vmix(v_base, state_visits, state_qs, state_policy, n_actions)
    else
        v_base
    end

    σ_vals .= v_baseline  # default to baseline for unvisited actions, will be overridden by visited ones
    @inbounds @simd for i in state_visits.nzind
        σ_vals[i] = state_qs[i]
    end

    if rescale_values
        min_value, max_value = extrema(σ_vals)
        σ_vals .= (σ_vals .- min_value) ./ max(max_value - min_value, eps(T))
    end

    # Compute σ_completed[a] for all actions
    @inbounds for i in eachindex(σ_vals)
        σ_vals[i] = (c_visit + Nmax) * c_scale * σ_vals[i]
    end

    # Compute π'_logits[a] = pscale * policy_logits[a] + σ_vals[a]
    @inbounds for i in eachindex(v_hold)
        v_hold[i] = pscale * state_policy[i] + σ_vals[i]
    end

    # In-place softmax with numerical stability: max-shift for safety
    max_logit = maximum(v_hold)
    sum_exp = zero(T)
    @inbounds for i in eachindex(v_hold)
        v_hold[i] = exp(v_hold[i] - max_logit)
        sum_exp += v_hold[i]
    end
    inv_sum = inv(sum_exp)
    @inbounds for i in eachindex(v_hold)
        v_hold[i] *= inv_sum  # v_hold now holds π'(a) (improved policy)
    end

    # Action selection: argmax_a (π'(a) - N(a) / (1 + total_N))
    # For valid actions only
    best_i_a = 0
    best_score = typemin(T)
    @inbounds for i in eachindex(v_hold)
        if mdp.is_valid_action(s, i)
            n_i = if i in state_visits.nzind
                T(state_visits[i])
            else
                zero(T)
            end
            score = v_hold[i] - n_i / (one(T) + total_N)
            if score > best_score
                best_score = score
                best_i_a = i
            end
        end
    end

    # Safety: if no valid action found, return cached value
    if best_i_a == 0
        return v_base
    end

    # Take action in environment and recurse
    r, s′ = mdp.ptf(s, best_i_a; step_kwargs...)
    q = r + γ * gumbel_simulate!(
        visit_counts, Q, v_est_cache, policy_cache,
        mdp, γ, π_dist!, min_reward, max_reward, pscale, s′, c_visit, c_scale,
        v_hold, v_new, σ_vals, step_kwargs, est_kwargs, depth - 1;
        use_vmix=use_vmix
    )

    # Backup: update visit count and running-average Q
    n_a = state_visits[best_i_a] + 1
    state_visits[best_i_a] = n_a
    δq = (q - state_qs[best_i_a]) / T(n_a)
    state_qs[best_i_a] += δq

    return q
end

"""
    sequential_halving_gumbel!(remaining, gumbel_noise, prior_logits,
                                Q_root, visit_root, scores, valid_actions,
                                mdp, γ, π_dist!, pscale, s, c_visit, c_scale,
                                visit_counts, Q, v_est_cache, policy_cache,
                                v_hold, v_new, σ_vals, step_kwargs,
                                est_kwargs, depth, nsims; use_vmix)

Run Sequential Halving with Gumbel (Algorithm 2) to identify the best action
from the set of actions in `remaining`.
"""
function sequential_halving_gumbel!(
    remaining::Vector{Int},
    gumbel_noise::Vector{T},
    prior_logits::Vector{T},
    Q_root::SparseVector{T, Int64},
    visit_root::SparseVector{T, Int64},
    scores::Vector{T},
    valid_actions::Vector{Int},
    mdp, γ::T, π_dist!::Function, min_reward::T, max_reward::T, pscale::T,
    s, c_visit::T, c_scale::T,
    visit_counts::Dict,
    Q::Dict,
    v_est_cache::Dict,
    policy_cache::Dict,
    v_hold::Vector{T}, v_new::SparseVector{T, Int64},
    σ_vals::Vector{T},
    step_kwargs::NamedTuple, est_kwargs::NamedTuple,
    depth::Int, nsims::Int, infval::T;
    use_vmix::Bool = false,
    rescale_values::Bool = true,
    sim_message::Bool = false
) where {T<:Real}
    m = length(remaining)
    m == 0 && error("Sequential Halving: no remaining actions")
    m == 1 && return remaining[1]

    budget_remaining = nsims
    phases = max(1, ceil(Int, log2(m)))

    sim_message && @info "Starting sequential halving with the following $m candidate actions: $(remaining). Total budget: $nsims simulations over $phases phases."

    for phase in 1:phases
        n_remaining = length(remaining)
        n_remaining <= 1 && break

        # Allocate budget for this phase: equal visits per remaining action
        # n_per_action = max(1, budget_remaining ÷ (n_remaining * (phases - phase + 1)))
        n_per_action = if phase == phases
            budget_remaining ÷ n_remaining
        else
            max(1, floor(Int, nsims / (phases * n_remaining)))
        end

        sim_message && @info "In phase $phase, allocating $n_per_action simulations per action for the $n_remaining remaining actions for a total of $(n_per_action * n_remaining) simulations. Budget remaining after this phase: $(budget_remaining - n_per_action * n_remaining)."

        # Visit each remaining action n_per_action times
        for a in remaining
            for _ in 1:n_per_action
                r, s′ = mdp.ptf(s, a; step_kwargs...)
                q = r + γ * gumbel_simulate!(
                    visit_counts, Q, v_est_cache, policy_cache,
                    mdp, γ, π_dist!, min_reward, max_reward, pscale, s′, c_visit, c_scale,
                    v_hold, v_new, σ_vals, step_kwargs, est_kwargs, depth - 1;
                    use_vmix=use_vmix, rescale_values=rescale_values
                )
                # Backup at root
                n_a = visit_root[a] + 1
                visit_root[a] = n_a
                δq = (q - Q_root[a]) / T(n_a)
                Q_root[a] += δq
            end
        end

        # Score remaining actions by gumbel + log_prior + σ(Q)
        Nmax = maximum(visit_root; init = zero(T))
        scores .= typemin(T)

        f_norm = if rescale_values
            min_value = typemax(T)
            max_value = typemin(T)
            for i in visit_root.nzind
                min_value = min(min_value, Q_root[i])
                max_value = max(max_value, Q_root[i])
            end
            x -> (x - min_value) / max(max_value - min_value, eps(T))
        else
            identity
        end

        for (i, a) in enumerate(remaining)
            q_val = if visit_root[a] != 0
                f_norm(Q_root[a])
            else
                zero(T)
            end

            # Handle Infs in prior by clamping values to a range still affected by the mean gumbel noise 
            p = prior_logits[a]
            prior_value = clamp(p, -infval, infval)
            σ_val = (c_visit + Nmax) * c_scale * q_val
            scores[i] = gumbel_noise[a] + pscale * prior_value + σ_val
        end

        budget_remaining -= n_per_action * n_remaining
        
        if budget_remaining <= 0
            kept_action = remaining[argmax(scores)]
            empty!(remaining)
            push!(remaining, kept_action)
            break
        end

        # Eliminate bottom half
        kept_count = min(budget_remaining, max(1, n_remaining ÷ 2)) # If the remaining budget is less than the number of actions, adjust to only keep 1 action per remaining budget which will make it zero the next turn
        kept_indices = partialsortperm(scores, 1:kept_count; rev=true)
        kept_actions = remaining[kept_indices]
        empty!(remaining)
        append!(remaining, kept_actions)
    end

    return remaining[1]
end


"""
    gumbel_mcts_search(mdp, γ, π_dist!, pscale, s, v_est;
                       nsims, topk, c_visit, c_scale, depth, use_vmix,
                       visit_counts, Q, v_est_cache, policy_cache,
                       prior, v_hold, σ_vals, gumbel_scratch,
                       make_step_kwargs, make_est_kwargs, sim_message)

Perform Gumbel MCTS search from root state `s` (Algorithm 2).

The `π_dist!` callback fills `prior` with the policy logits and returns
the value function estimate for the state. It has the signature:
    v_est = π_dist!(dst_logits, s)

When `use_vmix=true`, the mixed value estimator from Appendix D is used
instead of the raw value network output for the baseline of unvisited actions.

Returns:
    (best_action, visit_counts, Q, v_est_cache, policy_cache)

The returned caches enable training data extraction.
"""
function gumbel_mcts_search(
    mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, π_dist!::Function, pscale::T, s::S, v_est::Function;
    nsims::Int = 100,
    topk::Int = 16,
    c_visit::T = T(50),
    c_scale::T = T(1.0),
    depth::Int = 10,
    use_vmix::Bool = false,
    min_reward::T = typemin(T),
    max_reward::T = typemax(T),
    visit_counts::Dict{S, SparseVector{T, Int64}} = Dict{S, SparseVector{T, Int64}}(),
    Q::Dict{S, SparseVector{T, Int64}} = Dict{S, SparseVector{T, Int64}}(),
    v_est_cache::Dict{S, T} = Dict{S, T}(),
    policy_cache::Dict{S, Vector{T}} = Dict{S, Vector{T}}(),
    prior::Vector{T} = zeros(T, length(mdp.actions)),
    v_hold::Vector{T} = zeros(T, length(mdp.actions)),
    σ_vals::Vector{T} = zeros(T, length(mdp.actions)),
    gumbel_scratch::Vector{T} = zeros(T, length(mdp.actions)),
    scores::Vector{T} = zeros(T, length(mdp.actions)),
    make_step_kwargs::Function = k -> NamedTuple(),
    make_est_kwargs::Function = k -> NamedTuple(),
    sim_message::Bool = false,
    rescale_values::Bool = true
) where {T<:Real, S, A, P, F1<:Function, F2<:Function, F3<:Function}
    num_actions = length(mdp.actions)
    v_new = SparseVector(num_actions, Vector{Int64}(), Vector{T}())

    if nsims < topk
        sim_message && @warn "$nsims simulations is less than the topk value of $topk.  Only $nsims actions will be evaluated at the root node due to the simulation limit.  To sample all topk actions at the root, increase nsims to at least $topk."
        topk = nsims
    end

    !ispow2(topk) && sim_message && @warn "topk=$topk is not a power of 2.  Sequential halving will still work, but some phases will have uneven numbers of actions eliminated."

    # ---- Phase 1: Ensure root state is in tree ----
    if !haskey(visit_counts, s)
        v_est_val = π_dist!(prior, s)
        v_est_cache[s] = clamp(v_est_val, min_reward, max_reward) # Clamp value estimate to the range of known rewards
        local policy = copy(prior)
        policy_cache[s] = policy
        visit_counts[s] = copy(v_new)
        Q[s] = copy(v_new)
    else
        # Copy cached policy into prior scratch for use below
        copyto!(prior, policy_cache[s])
    end

    # ---- Phase 2: Sample Gumbel noise for all actions ----
    @inbounds for i in 1:num_actions
        gumbel_scratch[i] = -T(log(-log(rand(T))))
    end

    # ---- Phase 3: Find top-k actions by Gumbel score ----
    #get the minimum absolute value for the gumbel noise
    noisemean = mean(abs, gumbel_scratch)
    #find the largest value of T that is still affected by the noise mean such that infval + noisemean > infval and the change isn't swallowed up by the floating point precision
    infval = prevfloat(typemax(T))
    while infval + noisemean == infval
        infval /= 2
    end

    @inbounds @simd for i in 1:num_actions
        p = prior[i]
        #handle Inf prior by clamping it to a range where the gumbel noise can still break ties
        prior_value = clamp(p, -infval, infval)
        v_hold[i] = gumbel_scratch[i] + pscale * prior_value
    end

    valid_actions = [i for i in 1:num_actions if mdp.is_valid_action(s, i)]

    if isempty(valid_actions)
        error("Gumbel MCTS: no valid actions in state $s")
    elseif length(valid_actions) < topk
        sim_message && @warn "Only $(length(valid_actions)) valid actions available at root, which is less than topk=$topk.  Reducing topk to $(length(valid_actions))."
    end

    m = min(topk, length(valid_actions))

    # Get top-m actions by sampling from policy distribution and ignoring invalid actions
    valid_scores = [v_hold[i] for i in valid_actions]
    topm_valid_indices = partialsortperm(valid_scores, 1:m; rev=true)
    topm_actions = valid_actions[topm_valid_indices]

    # ---- Phase 4: Sequential Halving with Gumbel ----
    t_start = time()
    last_time = t_start

    remaining = copy(topm_actions)

    best_action = sequential_halving_gumbel!(
        remaining, gumbel_scratch, prior,
        Q[s], visit_counts[s], scores, valid_actions,
        mdp, γ, π_dist!, min_reward, max_reward, pscale, s, c_visit, c_scale,
        visit_counts, Q, v_est_cache, policy_cache,
        v_hold, v_new, σ_vals,
        make_step_kwargs(rand(UInt64)), make_est_kwargs(rand(UInt64)),
        depth, nsims, infval;
        use_vmix=use_vmix,
        sim_message=sim_message,
        rescale_values=rescale_values
    )

    if sim_message
        elapsed = time() - t_start
        @info "Gumbel MCTS search completed in $(elapsed) seconds"
    end

    return best_action, visit_counts, Q, v_est_cache, policy_cache
end


"""
    compute_completed_policy!(dst_logits, mdp, s, visit_counts, Q, v_base,
                               policy_logits, pscale; c_visit, c_scale, use_vmix)

Compute the improved policy π' from completed Q-values (Equation 11).

Given the search results at a state `s`, for each action a:
    completedQ[a] = Q[a] if N(a) > 0, else v_baseline
    σ[a] = (c_visit + Nmax) * c_scale * completedQ[a]
    π'_logits[a] = pscale * policy_logits[a] + σ[a]

When `use_vmix=true`, the baseline for unvisited actions uses the mixed
value estimator from Appendix D instead of the raw value network output.

Invalid actions are masked to -Inf before softmax.
The result is softmax-normalized into `dst_logits`.

Returns whether any action was visited.
"""
function compute_completed_policy!(
    dst_logits::Vector{T},
    mdp, s,
    visit_counts::SparseVector{T, Int64},
    Q::SparseVector{T, Int64},
    v_base::T,
    policy_logits::Vector{T},
    pscale::T;
    c_visit::T = T(50),
    c_scale::T = T(1.0),
    use_vmix::Bool = false,
    rescale_values::Bool = true
) where {T<:Real}
    num_actions = length(mdp.actions)

    # Compute Nmax and total_N
    Nmax = zero(T)
    total_N = zero(T)
    for i in visit_counts.nzind
        n = visit_counts[i]
        if n > Nmax
            Nmax = n
        end
        total_N += n
    end

    has_visits = !iszero(Nmax)

    # Determine baseline (v_mix or raw value network)
    v_baseline = if use_vmix && total_N > 0
        compute_vmix(v_base, visit_counts, Q, policy_logits, num_actions)
    else
        v_base
    end

    dst_logits .= v_baseline  # default to baseline for unvisited actions, will be overridden by visited ones
    @inbounds @simd for i in visit_counts.nzind
        dst_logits[i] = Q[i]
    end

    if rescale_values
        min_value, max_value = extrema(dst_logits)
        dst_logits .= (dst_logits .- min_value) ./ max(max_value - min_value, eps(T))
    end

    # Compute π'_logits, masking invalid actions
    @inbounds for i in 1:num_actions
        if !mdp.is_valid_action(s, i)
            dst_logits[i] = typemin(T)
        else 
            dst_logits[i] *= (c_visit + Nmax) * c_scale
            dst_logits[i] += pscale * policy_logits[i]
        end
    end

    # In-place softmax with numerical stability
    max_logit = maximum(dst_logits)
    if isfinite(max_logit)
        sum_exp = zero(T)
        @inbounds for i in 1:num_actions
            dst_logits[i] = exp(dst_logits[i] - max_logit)
            sum_exp += dst_logits[i]
        end
        inv_sum = inv(sum_exp)
        @inbounds for i in 1:num_actions
            dst_logits[i] *= inv_sum
        end
    else
        # All logits are -Inf, fallback to uniform over valid actions
        n_valid = count(i -> mdp.is_valid_action(s, i), 1:num_actions)
        uniform = n_valid > 0 ? inv(T(n_valid)) : inv(T(num_actions))
        @inbounds for i in 1:num_actions
            dst_logits[i] = mdp.is_valid_action(s, i) ? uniform : zero(T)
        end
    end

    return has_visits
end


"""
    extract_root_training_data(mdp, s, pscale, visit_counts, Q,
                                v_est_cache, policy_cache;
                                c_visit, c_scale, use_vmix, v_hold_scratch)

Extract training data from the Gumbel MCTS search results for the root state.

The `mdp` argument is used to determine the number of actions, check action validity,
and verify state type consistency.

When `use_vmix=true`, the mixed value estimator from Appendix D is used
for the unvisited action baseline.

Returns:
    (improved_policy, q_values, state_value, policy_logits)

Where:
    - improved_policy: Vector{T} of π'(a) — the policy improvement target
    - q_values: Vector{T} of completed Q-values for each action
    - state_value: T, the value estimate at root (v_mix if use_vmix=true)
    - policy_logits: Vector{T}, the original policy logits from π_dist!
"""
function extract_root_training_data(
    mdp, s, pscale::T,
    visit_counts::Dict,
    Q::Dict,
    v_est_cache::Dict,
    policy_cache::Dict;
    c_visit::T = T(50),
    c_scale::T = T(1.0),
    use_vmix::Bool = false,
    rescale_values::Bool = true,
    v_hold_scratch::Vector{T} = zeros(T, length(mdp.actions))
) where {T<:Real}
    if !haskey(visit_counts, s)
        error("State $s not found in search results")
    end

    num_actions = length(mdp.actions)
    v_base = v_est_cache[s]
    policy_logits = policy_cache[s]
    vs = visit_counts[s]
    qs = Q[s]

    # Determine effective state value (v_mix or raw)
    total_N = sum(vs)
    state_value = if use_vmix && total_N > 0
        compute_vmix(v_base, vs, qs, policy_logits, num_actions)
    else
        v_base
    end

    # Compute improved policy (uses state_value as baseline for unvisited)
    compute_completed_policy!(
        v_hold_scratch, mdp, s, vs, qs, v_base,
        policy_logits, pscale; c_visit=c_visit, c_scale=c_scale,
        use_vmix=use_vmix, rescale_values=rescale_values
    )
    improved_policy = copy(v_hold_scratch)

    # Compute completed Q-values
    q_completed = zeros(T, num_actions)
    Nmax = zero(T)
    for i in vs.nzind
        n = T(vs[i])
        if n > Nmax
            Nmax = n
        end
    end
    @inbounds for i in 1:num_actions
        if i in qs.nzind
            q_completed[i] = qs[i]
        else
            q_completed[i] = state_value
        end
    end

    return (improved_policy=improved_policy, q_values=q_completed,
            state_value=state_value, policy_logits=copy(policy_logits))
end


"""
    make_gumbel_mcts_policy(mdp, γ, π_dist!, pscale, v_est; kwargs...)

Create a policy function that uses Gumbel MCTS search for action selection.

Returns a function `policy(s)` that performs search and returns the best action,
along with search results stored in closure.

Usage:
    policy, search_state = make_gumbel_mcts_policy(mdp, γ, π_dist!, pscale, v_est)
    best_action = policy(s)
    # Access training data:
    data = extract_root_training_data(mdp, s, pscale, search_state...;
                                       c_visit=50, c_scale=1.0)
"""
function make_gumbel_mcts_policy(
    mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, π_dist!::Function, pscale::T, v_est::Function;
    kwargs...
) where {T<:Real, S, A, P, F1<:Function, F2<:Function, F3<:Function}
    num_actions = length(mdp.actions)
    visit_counts = Dict{S, SparseVector{T, Int64}}()
    Q = Dict{S, SparseVector{T, Int64}}()
    v_est_cache = Dict{S, T}()
    policy_cache = Dict{S, Vector{T}}()

    # Preallocate scratch vectors
    prior = zeros(T, num_actions)
    v_hold = zeros(T, num_actions)
    σ_vals = zeros(T, num_actions)
    gumbel_scratch = zeros(T, num_actions)
    scores = zeros(T, num_actions)

    function policy(s)
        best_action, visit_counts, Q, v_est_cache, policy_cache = gumbel_mcts_search(
            mdp, γ, π_dist!, pscale, s, v_est;
            visit_counts=visit_counts, Q=Q,
            v_est_cache=v_est_cache, policy_cache=policy_cache,
            prior=prior, v_hold=v_hold, σ_vals=σ_vals,
            gumbel_scratch=gumbel_scratch, scores=scores,
            kwargs...
        )
        return best_action
    end

    function clear_search_tree!()
        empty!(visit_counts)
        empty!(Q)
    end

    # Store state for training data extraction
    return (policy = policy, clear_search_tree! = clear_search_tree!, search_state = (visit_counts, Q, v_est_cache, policy_cache))
end

end  # module GumbelMCTS