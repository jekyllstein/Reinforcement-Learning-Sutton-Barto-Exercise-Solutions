module NonTabularRL

using Reexport, PrecompileTools

import PlutoDevMacros

@reexport using ApproximationUtils

include(joinpath(@__DIR__, "..", "..", "Chapter-09", "Chapter_09_On-policy_Prediction_with_Approximation.jl"))

export gradient_monte_carlo_episode_update!, gradient_monte_carlo_policy_estimation!, gradient_monte_carlo_estimation!, semi_gradient_td0_update!, semi_gradient_td0_policy_estimation!, semi_gradient_td0_estimation!, semi_gradient_td0_policy_estimation, semi_gradient_td0_estimation, make_random_walk_mrp, state_aggregation_gradient_setup, run_state_aggregation_monte_carlo_policy_estimation, run_state_aggregation_semi_gradient_policy_estimation, order_features_gradient_setup, run_order_features_monte_carlo_policy_estimation, calc_poly_feature, calc_fourier_feature, tile_coding_gradient_setup, run_tile_coding_monte_carlo_policy_estimation

include(joinpath(@__DIR__, "..", "..", "Chapter-10", "Chapter_10_On_policy_Control_with_Approximation.jl"))

export run_fcann_semi_gradient_sarsa, run_fcann_semi_gradient_dp, run_linear_differential_semi_gradient_sarsa, run_linear_gradient_monte_carlo_control, run_linear_semi_gradient_sarsa, run_linear_semi_gradient_dp, run_nonlinear_differential_semi_gradient_dp, state_aggregation_action_gradient_setup, create_access_control_task, create_access_control_tabular_task, run_fcann_gradient_monte_carlo_control

include(joinpath(@__DIR__, "..", "..", "Chapter-11", "Chapter_11_Off_policy_Methods_with_Approximation.jl"))

export make_baird_ptf, make_baird_mdps, baird_update_state_vector!, tdc_estimation, tdc_control, tdc_dp_control

include(joinpath(@__DIR__, "..", "..", "Chapter-12", "Chapter_12_Eligibility_Traces.jl"))

export n_step_TD_prediction, run_linear_semi_gradient_TDλ, run_fcann_semi_gradient_TDλ, run_tabular_semi_gradient_TDλ, true_online_TDλ, true_online_tabular_TDλ, sarsa_λ, dp_λ, true_online_dp_λ!, dp_λ!, sarsa_λ!, expected_sarsa_λ!, true_online_sarsa_λ!, true_online_expected_sarsa_λ!, tile_coding_setup, get_active_features, create_cartpole_mdp, setup_cartpole_problem, test_sarsa_λ 

include(joinpath(@__DIR__, "..", "..", "Chapter-13", "Chapter_13_Policy_Gradient_Methods.jl"))

export make_corridor_mdp, soft_max!, BinaryFeatures, BinaryFeatureVector, BinaryEligibilityVector, setup_binary_policy_arguments, reinforce_monte_carlo_binary_features, reinforce_monte_carlo_linear_features, setup_fcann_policy_arguments, reinforce_monte_carlo_fcann, linear_value_function, binary_value_function, form_state_policy_function, form_state_value_function, reinforce_with_baseline_monte_carlo_control_binary_features, reinforce_with_baseline_monte_carlo_control_linear_features, setup_fcann_value_arguments, setup_fcann_policy_and_value_arguments, reinforce_with_baseline_monte_carlo_control_fcann, form_state_and_policy_function_outputs, one_step_actor_critic_binary_features, one_step_actor_critic_linear_features, one_step_actor_critic_fcann, actor_critic_with_eligibility_traces_binary_features, actor_critic_with_eligibility_traces_linear_features, actor_critic_with_eligibility_traces_fcann, make_corridor_continuing_mdp, actor_critic_linear_parameter_study, actor_critic_fcann_parameter_study, create_cartpole_functions, cartpole_continuing_step, mountaincar_continuing_step, create_mountaincar_continuing_mdp, fcann_feature_vector_setup, create_continuous_action_mountaincar, AbstractContinuousTransition, ContinuousMDPTransitionSampler, ContinuousMDP, BinaryGaussianEligibilityVector, BinaryBetaEligibilityVector, BinarySquashedGaussianEligibilityVector, bad_continuous_action, setup_binary_gaussian_policy_arguments, setup_binary_beta_policy_arguments, setup_binary_squashed_gaussian_policy_arguments, reinforce_with_baseline_monte_carlo_control_binary_features_gaussian_actions, reinforce_with_baseline_monte_carlo_control_linear_features_gaussian_actions, actor_critic_with_eligibility_traces_binary_features_gaussian_actions, actor_critic_with_eligibility_traces_binary_features_beta_actions, actor_critic_with_eligibility_traces_binary_features_squashed_gaussian_actions, actor_critic_binary_episodic_gaussian_parameter_study, actor_critic_binary_episodic_beta_parameter_study, actor_critic_binary_episodic_squashed_gaussian_parameter_study, setup_cartpole_continuous_problem, actor_critic_binary_episodic_parameter_study, create_continous

@setup_workload begin
    γ = 0.9f0
    num_episodes = 10
    α = 0.1f0
    max_steps = 100
    num_states = 1000
    initial_state = 500
    num_groups = 10
    randomwalk_state_ptf = StateMRPTransitionSampler((s) -> randomwalk_step(s, num_states), 1f0)
    randomwalk_state_init() = Float32(initial_state)
    random_walk_group_assign = make_random_walk_group_assign(num_states, num_groups)


    function run_access_control_differential_sarsa(max_steps::Int64; num_servers = 10, priority_payments = [1f0, 2f0, 4f0, 8f0], kwargs...)
        (mdp, gradient_setup, num_groups) = create_access_control_task(num_servers, priority_payments)
        parameters = [zeros(Float32, num_groups) for _ in eachindex(mdp.actions)]
        state_representation = zeros(Float32, num_groups)
        (_, _, steprewards) = differential_semi_gradient_sarsa!(parameters, mdp, 1, max_steps, gradient_setup...; kwargs...)
        action_values = zeros(Float32, length(mdp.actions))
        v̂(num_free_servers::Int64, priority::Real) = gradient_setup.value_function(action_values, AccessControlState(num_free_servers, Float32(priority)), parameters)
        (value_function = v̂, mdp = mdp, parameters = parameters, steprewards = steprewards)
    end

    @compile_workload begin
        random_walk_state_mrp = StateMRP(randomwalk_state_ptf, randomwalk_state_init, s -> randomwalk_isterm(s, num_states))
        run_state_aggregation_monte_carlo_estimation(random_walk_state_mrp, 1f0, 10, num_groups, random_walk_group_assign; α = α)
        run_access_control_differential_sarsa(10)

        for algo! in [sarsa_λ!, expected_sarsa_λ!, true_online_sarsa_λ!, true_online_expected_sarsa_λ!]
            test_sarsa_λ(;algo! = algo!)
        end
    end
end

end # module NonTabularRL
