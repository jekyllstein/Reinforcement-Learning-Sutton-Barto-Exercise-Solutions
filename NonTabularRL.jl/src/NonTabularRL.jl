module NonTabularRL

using Reexport, PrecompileTools

import PlutoDevMacros

@reexport using ApproximationUtils

include(joinpath(@__DIR__, "..", "..", "Chapter-09", "Chapter_09_On-policy_Prediction_with_Approximation.jl"))

export FCANNParams, FCANNActivations, LinearFeatures, AbstractBinaryFeatures, BinaryFeatureVector, StateAggregationFeatureVector, form_state_value_function, gradient_monte_carlo_episode_update!, gradient_monte_carlo_policy_estimation!, gradient_monte_carlo_estimation!, gradient_monte_carlo_estimation_linear, gradient_monte_carlo_policy_estimation_linear, semi_gradient_td0_estimation!, semi_gradient_td0_policy_estimation!, semi_gradient_td0_estimation_linear, semi_gradient_td0_policy_estimation_linear 

export state_aggregation_feature_setup, gradient_monte_carlo_estimation_state_aggregation, gradient_monte_carlo_policy_estimation_state_aggregation, semi_gradient_td0_estimation_state_aggregation,semi_gradient_td0_policy_estimation_state_aggregation 

export get_order_coefficients, order_features_setup, calc_poly_feature, calc_fourier_feature

export tile_coding_feature_setup

export setup_fcann_value_arguments, gradient_monte_carlo_estimation_fcann, gradient_monte_carlo_policy_estimation_fcann, semi_gradient_td0_estimation_fcann, semi_gradient_td0_policy_estimation_fcann

export make_random_walk_mrp, create_continuous_random_walk, make_random_walk_group_assign

export least_squares_td_estimation, least_squares_td_policy_estimation

include(joinpath(@__DIR__, "..", "..", "Chapter-10", "Chapter_10_On_policy_Control_with_Approximation.jl"))

export LinearActionValueGradient, form_value_function, compute_sarsa_value, semi_gradient_sarsa!, semi_gradient_double_sarsa!, semi_gradient_dp!, calculate_action_value, compute_expected_sarsa_value, compute_q_learning_value, semi_gradient_differential_sarsa!

export initialize_linear_parameters, semi_gradient_sarsa_linear, semi_gradient_double_sarsa_linear, semi_gradient_dp_linear, semi_gradient_expoeted_sarsa_linear, semi_gradient_q_learning_linear, semi_gradient_differential_sarsa_linear, semi_gradient_differential_sarsa_fcann

export setup_fcann_action_value_arguments, semi_gradient_sarsa_fcann, semi_gradient_dp_fcann

export make_tabular_mountaincar, update_mountaincar_feature_vector, create_differential_mountaincar_mdp

export form_differential_value_function, semi_gradient_differential_dp!, semi_gradient_differential_dp_linear, semi_gradient_differential_dp_fcann

export create_access_control_task, create_access_control_tabular_task

export gradient_monte_carlo_control!, gradient_monte_carlo_control_linear, gradient_monte_carlo_control_fcann

include(joinpath(@__DIR__, "..", "..", "Chapter-11", "Chapter_11_Off_policy_Methods_with_Approximation.jl"))

export make_baird_ptf, make_baird_mdps, baird_update_state_vector!, tdc_estimation, tdc_control, tdc_dp_control

include(joinpath(@__DIR__, "..", "..", "Chapter-12", "Chapter_12_Eligibility_Traces.jl"))

export n_step_TD_prediction, run_linear_semi_gradient_TDλ, run_fcann_semi_gradient_TDλ, run_tabular_semi_gradient_TDλ, true_online_TDλ, true_online_tabular_TDλ, sarsa_λ, dp_λ, true_online_dp_λ!, dp_λ!, sarsa_λ!, expected_sarsa_λ!, true_online_sarsa_λ!, true_online_expected_sarsa_λ!, tile_coding_setup, get_active_features, create_cartpole_mdp, setup_cartpole_problem, test_sarsa_λ 

# include(joinpath(@__DIR__, "..", "..", "Chapter-13", "Chapter_13_Policy_Gradient_Methods.jl"))

# export make_corridor_mdp, soft_max!, BinaryFeatures, BinaryFeatureVector, BinaryEligibilityVector, setup_binary_policy_arguments, reinforce_monte_carlo_binary_features, reinforce_monte_carlo_linear_features, setup_fcann_policy_arguments, reinforce_monte_carlo_fcann, linear_value_function, binary_value_function, form_state_policy_function, form_state_value_function, reinforce_with_baseline_monte_carlo_control_binary_features, reinforce_with_baseline_monte_carlo_control_linear_features, setup_fcann_value_arguments, setup_fcann_policy_and_value_arguments, reinforce_with_baseline_monte_carlo_control_fcann, form_state_and_policy_function_outputs, one_step_actor_critic_binary_features, one_step_actor_critic_linear_features, one_step_actor_critic_fcann, actor_critic_with_eligibility_traces_binary_features, actor_critic_with_eligibility_traces_linear_features, actor_critic_with_eligibility_traces_fcann, make_corridor_continuing_mdp, actor_critic_linear_parameter_study, actor_critic_fcann_parameter_study, create_cartpole_functions, cartpole_continuing_step, mountaincar_continuing_step, create_mountaincar_continuing_mdp, fcann_feature_vector_setup, create_continuous_action_mountaincar, AbstractContinuousTransition, ContinuousMDPTransitionSampler, ContinuousMDP, BinaryGaussianEligibilityVector, BinaryBetaEligibilityVector, BinarySquashedGaussianEligibilityVector, bad_continuous_action, setup_binary_gaussian_policy_arguments, setup_binary_beta_policy_arguments, setup_binary_squashed_gaussian_policy_arguments, reinforce_with_baseline_monte_carlo_control_binary_features_gaussian_actions, reinforce_with_baseline_monte_carlo_control_linear_features_gaussian_actions, actor_critic_with_eligibility_traces_binary_features_gaussian_actions, actor_critic_with_eligibility_traces_binary_features_beta_actions, actor_critic_with_eligibility_traces_binary_features_squashed_gaussian_actions, actor_critic_binary_episodic_gaussian_parameter_study, actor_critic_binary_episodic_beta_parameter_study, actor_critic_binary_episodic_squashed_gaussian_parameter_study, setup_cartpole_continuous_problem, actor_critic_binary_episodic_parameter_study, create_continous

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
        (mdp, setup) = create_access_control_task(num_servers, priority_payments)
        semi_gradient_differential_sarsa_linear(mdp, 1, max_steps, setup...; kwargs...)
    end

    @compile_workload begin
        random_walk_state_mrp = StateMRP(randomwalk_state_ptf, randomwalk_state_init, s -> randomwalk_isterm(s, num_states))
        gradient_monte_carlo_estimation_state_aggregation(random_walk_state_mrp, 1f0, 10, num_groups, random_walk_group_assign; α = α)
        run_access_control_differential_sarsa(10)
        run_access_control_differential_sarsa(10; compute_value = compute_expected_sarsa_value)
        run_access_control_differential_sarsa(10; compute_value = compute_q_learning_value)


        # for algo! in [sarsa_λ!, expected_sarsa_λ!, true_online_sarsa_λ!, true_online_expected_sarsa_λ!]
        #     test_sarsa_λ(;algo! = algo!)
        # end
    end
end

end # module NonTabularRL
