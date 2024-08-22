module TabularRL

using PrecompileTools: @setup_workload, @compile_workload
import PlutoDevMacros

include(joinpath("..", "..", "Tabular-Methods-Summary", "tabular_methods_overview_notebook.jl"))

#---------Types------------
#dynamic programming mdp types
export AbstractTransition, AbstractTabularTransition, TabularTransitionDistribution, TabularDeterministicTransition, TabularStochasticTransition, TabularMDPTransitionSampler, TabularMRPTransitionSampler, AbstractStateTransition, StateMDPTransitionDistribution, StateMDPTransitionSampler, StateMRPTransitionDistribution, StateMRPTransitionSampler

export AbstractMP, AbstractMDP, AbstractMRP, TabularMDP, TabularMRP, AbstractAfterstateMDP, TabularAfterstateMDP, StateMDP, StateMRP

#sampling and averaging types
export AbstractAveragingMethod, SampleAveraging, ConstantStepAveraging

#--------Functions---------
#utilities 
export initialize_state_action_value, initialize_state_value, find_terminal_states, find_available_actions, sample_action, make_random_policy, runepisode, runepisode!, make_greedy_policy!, make_ϵ_greedy_policy!, initialize_afterstate_value

#dynamic programming solution methods
export bellman_state_value, bellman_state_action_value, bellman_policy_update!, policy_evaluation!, policy_evaluation, mrp_evaluation!, mrp_evaluation, policy_evaluation_q, policy_evaluation_v, policy_iteration!, policy_iteration, policy_iteration_v, value_iteration!, value_iteration, value_iteration_v, value_iteration_q, bellman_afterstate_value, afterstate_policy_iteration!

#monte carlo solution methods
export monte_carlo_policy_prediction, monte_carlo_prediction, monte_carlo_policy_prediction_v, monte_carlo_policy_prediction_q, monte_carlo_control, monte_carlo_control_exploring_starts, monte_carlo_control_ϵ_soft, monte_carlo_off_policy_prediction, monte_carlo_off_policy_prediction_q, monte_carlo_off_policy_control

#temporal difference solution methods
export td0_policy_prediction, td0_prediction, td0_policy_prediction_v, td0_policy_prediction_q, generalized_sarsa!, sarsa, expected_sarsa, q_learning, double_expected_sarsa, double_q_learning

#planning solution methods
export monte_carlo_tree_search, sample_rollout, distribution_rollout, uct, apply_uct!, simulate!

#----------Gridworld Environment------------
export GridworldState, GridworldAction, rook_actions, make_deterministic_gridworld, make_stochastic_gridworld

@setup_workload begin
    γ = 0.9f0
    num_episodes = 10
    α = 0.1f0
    max_steps = 100
    @compile_workload begin
        for f in [make_deterministic_gridworld, make_stochastic_gridworld]
            mdp = f()
            policy_iteration_v(mdp, γ)
            value_iteration_v(mdp, γ)
            runepisode(mdp; max_steps = max_steps)
            monte_carlo_control_exploring_starts(mdp, γ, num_episodes; max_steps = max_steps)
            monte_carlo_control_exploring_starts(mdp, γ, num_episodes; averaging_method = ConstantStepAveraging(α), max_steps = max_steps)
            monte_carlo_control_ϵ_soft(mdp, γ, num_episodes; max_steps = max_steps)
            monte_carlo_off_policy_control(mdp, γ, num_episodes; max_steps = max_steps)
            sarsa(mdp, γ; max_steps = max_steps, α = α)
            expected_sarsa(mdp, γ; max_steps = max_steps, α = α)
            q_learning(mdp, γ; max_steps = max_steps, α = α)
            state_mdp = StateMDP(mdp)
            make_random_policy(state_mdp)
            runepisode(state_mdp)
            monte_carlo_tree_search(state_mdp, 0.99f0, state_mdp.initialize_state())

            mrp = create_random_walk_distribution(5, -1f0, 1f0)
            mrp_evaluation(mrp, 1f0)
            monte_carlo_prediction(mrp, 1f0, num_episodes)
            td0_prediction(mrp, 1f0; max_steps = max_steps)
            runepisode(mrp)
            state_mrp = StateMRP(mrp)
            runepisode(state_mrp)
        end
    end
end
end # module TabularRL
