module NonTabularRL

using Reexport, PrecompileTools

import PlutoDevMacros

@reexport using TabularRL, FCANN, NVIDIALibraries, SparseArrays, LinearAlgebra, TailRec

include(joinpath(@__DIR__, "..", "..", "Chapter-9", "Chapter_9_On-policy_Prediction_with_Approximation.jl"))

export gradient_monte_carlo_episode_update!, gradient_monte_carlo_policy_estimation!, gradient_monte_carlo_estimation!, semi_gradient_td0_update!, semi_gradient_td0_policy_estimation!, semi_gradient_td0_estimation!, semi_gradient_td0_policy_estimation, semi_gradient_td0_estimation, make_random_walk_mrp, state_aggregation_gradient_setup, run_state_aggregation_monte_carlo_policy_estimation, run_state_aggregation_semi_gradient_policy_estimation, order_features_gradient_setup, run_order_features_monte_carlo_policy_estimation, calc_poly_feature, calc_fourier_feature, tile_coding_gradient_setup, run_tile_coding_monte_carlo_policy_estimation

include(joinpath(@__DIR__, "..", "..", "Chapter-10", "Chapter_10_On_policy_Control_with_Approximation.jl"))

export run_fcann_semi_gradient_sarsa

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
    @compile_workload begin
        random_walk_state_mrp = StateMRP(randomwalk_state_ptf, randomwalk_state_init, s -> randomwalk_isterm(s, num_states))
        run_state_aggregation_monte_carlo_estimation(random_walk_state_mrp, 1f0, 10, num_groups, random_walk_group_assign; α = α)
    end
end

end # module NonTabularRL
