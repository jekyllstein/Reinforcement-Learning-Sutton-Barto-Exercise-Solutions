module TabularRL

using PrecompileTools: @setup_workload, @compile_workload

include(joinpath(@__DIR__, "..", "..", "Tabular-Methods-Summary", "tabular_methods_overview_notebook.jl"))

#---------Types------------
#dynamic programming mdp types
export AbstractMDP, AbstractTabularMDP, AbstractCompleteMDP, FiniteDeterministicMDP, FiniteStochasticMDP

#sample mdp types
export AbstractSampleTabularMDP, SampleTabularMDP

#sampling and averaging types
export AbstractAveragingMethod, SampleAveraging, ConstantStepAveraging

#--------Functions---------
#utilities 
export initialize_state_action_value, initialize_state_value, find_terminal_states, make_isterm, sample_action, make_random_policy, runepisode, make_greedy_policy!, make_ϵ_greedy_policy!

#dynamic programming solution methods
export policy_evaluation_q, policy_evaluation_v, policy_iteration_v, value_iteration, value_iteration_v

#monte carlo solution methods
export monte_carlo_policy_prediction, monte_carlo_policy_prediction_v, monte_carlo_control, monte_carlo_control_exploring_starts, monte_carlo_control_ϵ_soft, monte_carlo_off_policy_prediction, monte_carlo_off_policy_prediction_q, monte_carlo_off_policy_control

#temporal difference solution methods
export td0_policy_prediction, td0_policy_prediction_v, td0_policy_prediction_q#, sarsa, expected_sarsa, q_learning

#----------Gridworld Environment------------
export GridworldState, GridworldAction, rook_actions, make_gridworld

@setup_workload begin
    γ = 0.9f0
    num_episodes = 10
    α = 0.1f0
    max_steps = 100
    @compile_workload begin
        (mdp, isterm, init_state) = make_gridworld()
        policy_iteration_v(mdp, γ)
        value_iteration_v(mdp, γ)
        sample_mdp = SampleTabularMDP(mdp, () -> init_state)
        runepisode(sample_mdp; max_steps = max_steps)
        monte_carlo_control_exploring_starts(sample_mdp, γ, num_episodes; max_steps = max_steps)
        monte_carlo_control_exploring_starts(sample_mdp, γ, num_episodes; averaging_method = ConstantStepAveraging(α), max_steps = max_steps)
        monte_carlo_control_ϵ_soft(sample_mdp, γ, num_episodes, max_steps = max_steps)
        monte_carlo_off_policy_control(sample_mdp, γ, num_episodes; max_steps = max_steps)
        # sarsa(sample_mdp, γ, num_episodes, α)
        # expected_sarsa(sample_mdp, γ, num_episodes, α)
        # q_learning(sample_mdp, γ, num_episodes, α)
    end
end
end # module TabularRL
