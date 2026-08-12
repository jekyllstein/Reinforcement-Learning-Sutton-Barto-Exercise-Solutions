### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 22ccdf79-e8a7-4233-88b0-5678797d5cb8
using Revise

# ╔═╡ 18ed439c-09b0-4039-bdb7-12b47e25f82d
using PlutoDevMacros

# ╔═╡ 7035e568-65b9-4e90-8a7c-fb7354fb7aec
begin
	@fromparent begin
		import *
	end

	# include("GridCapture.jl")
	using .ReinforcementLearning.GridCapture

	include("gumbel_mcts.jl")
	using .GumbelMCTS
end

# ╔═╡ 662669a4-243d-4d9d-98c4-d2acac04358c
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI, PlutoPlotly, LaTeXStrings, PlutoProfile, HypertextLiteral, ProgressLogging, BenchmarkTools
	TableOfContents(;depth = 4)
end
  ╠═╡ =#

# ╔═╡ 657a72a3-933f-4e48-b357-7a73315fab65
md"""
# Grid Capture Environment

This environment is an extended tic-tac-toe game where the size of the square board can be scaled to any value of N and the win condition can be scaled with K.  The board state is parameterized by N and K and stores a bit matrix for each of the player's pieces.
"""

# ╔═╡ 1519aac0-b013-49c8-a222-0d26c05d5195
md"""
## Board and Moves
"""

# ╔═╡ 62c32e45-49bd-473c-b4d8-fc15a2dbaa17
md"""
Adjust board size
"""

# ╔═╡ ed609953-cf65-45cb-ba89-c5a6058dd4af
#=╠═╡
@bind board_N Slider(3:10, default = 6, show_value=true)
  ╠═╡ =#

# ╔═╡ 9e6c8d1d-2152-4b45-a2b6-38e48a4f0b8e
#=╠═╡
board = GameState{board_N, 4}()
  ╠═╡ =#

# ╔═╡ 819d1840-27a9-4cba-993a-e028584ed147
#=╠═╡
html_board(board; cell_size = 60) |> HTML
  ╠═╡ =#

# ╔═╡ 7cde31ab-ee50-4553-b9d0-45141ddfc40d
#=╠═╡
@bind test_x_move PlutoUI.combine() do Child
	md"""
	x position: $(Child(Slider(1:board_N)))
	
	y position: $(Child(Slider(1:board_N)))
	"""
end
  ╠═╡ =#

# ╔═╡ 606741df-ff6e-4a00-b8b4-400de9cec9b3
#=╠═╡
begin
	new_board = place_stone(board, test_x_move...)
	html_board(new_board) |> HTML
end
  ╠═╡ =#

# ╔═╡ 806ed2a1-0b2a-4958-9d10-5f3117b69f72
md"""
Note that a player can only make a move in an empty square
"""

# ╔═╡ 030e5494-4dcd-4456-8b7f-9335b1e148c5
#=╠═╡
@bind test_o_move PlutoUI.combine() do Child
	md"""
	x position: $(Child(Slider(1:board_N)))
	
	y position: $(Child(Slider(1:board_N; default = 2)))
	"""
end
  ╠═╡ =#

# ╔═╡ 2cd49671-b297-4d4a-8d5a-8c8937372088
#=╠═╡
place_stone(new_board, test_o_move...) |> html_board |> HTML
  ╠═╡ =#

# ╔═╡ 8ae4306e-1c38-4a76-81c4-741a00b9a1df
play_gridcapture_game(s0; x_player::Function = s -> rand(available_moves(s)), o_player::Function = s -> rand(available_moves(s)), kwargs...) = play_gridcapture_game(s0, x_player, o_player; kwargs...)

# ╔═╡ 2aed3faa-2e28-4400-b71d-d8b3ac5fa553
# ╠═╡ skip_as_script = true
#=╠═╡
const gridcapture_rewards = Dict([:x_win => 1f0, :o_win => -1f0, :draw => 0f0])
  ╠═╡ =#

# ╔═╡ 482d4aaa-f3bc-41df-adbb-23e588abdcd2
#=╠═╡
function score_game(s::GameState)
	result = check_game_result(s)

	result == :ongoing && error("Cannot score game that has not yet finished")
	gridcapture_rewards[result]
end
  ╠═╡ =#

# ╔═╡ a46fad01-1453-46ca-ab09-448ded3b8aae
#=╠═╡
function play_gridcapture_game(s0::S, x_player::Function, o_player::Function; save_states::Bool = true) where S<:GameState
	s = s0
	states = Vector{S}()
	save_states && push!(states, s)
	while !game_over(s)
		m = if player_turn(s) == 1
			x_player(s)
		else
			o_player(s)
		end
		s = place_stone(s, m...)
		save_states && push!(states, s)
	end
	return states, s, score_game(s)
end
  ╠═╡ =#

# ╔═╡ f0c23fbc-112b-4a47-8af2-e166dea16627
#=╠═╡
#calculates the average value from state s0 from the perspective of the x player based on the game outcome of 1, 0, -1 for xwin, draw, owin
function calculate_x_value(s0::S, args...; num_games::Integer = 10_000) where S<:GameState
	f(i) = play_gridcapture_game(s0, args...; save_states = false)[3]
	1:num_games |> Map(f) |> foldxt(+) |> x -> x / num_games
end
  ╠═╡ =#

# ╔═╡ e55fc618-5319-4059-9fee-6733329855f1
#=╠═╡
#calculate the winrate from the perspective of the x player
calculate_x_winrate(args...; kwargs...) = (calculate_x_value(args...; kwargs...) + 1) / 2
  ╠═╡ =#

# ╔═╡ a320f525-ca01-4bb8-a872-8c563107fad7
#=╠═╡
random_game = play_gridcapture_game(board)
  ╠═╡ =#

# ╔═╡ 3ece2292-f3dc-4d9b-9b24-6badb8246862
#=╠═╡
md"""
Random Game Step: $(@bind random_game_step Slider(1:length(random_game[1]); show_value=true))
"""
  ╠═╡ =#

# ╔═╡ 4b116755-9c16-43a9-932d-336c206777c6
#=╠═╡
random_game[1][random_game_step] |> s -> begin
	@htl("""
	<div style = "display: flex;">
		 <div>
	<div>Greedy Policy Distribution</div>
	
	$(HTML(html_board(s; cell_size = 30, policy = greedy_policy_distribution(s))))
    </div>
	<div>
		 <div>Defensive Policy Distribution</div>
		$(HTML(html_board(s; cell_size = 30, policy = defensive_policy_distribution(s))))
	</div>
		 <div>
		 <div>Positional Policy Distribution</div>
		 $(HTML(html_board(s; cell_size = 30, policy = positional_policy_distribution(s))))
		</div>
		 </div>
		 """)
end
  ╠═╡ =#

# ╔═╡ 2b89ab81-f8c5-458c-8a5e-fadd5418d8dd
#=╠═╡
# random policy faceoff
calculate_x_winrate(board)
  ╠═╡ =#

# ╔═╡ 6b694195-dba4-411b-a600-add1c2d1bed9
#=╠═╡
greedy_game = play_gridcapture_game(board; x_player = s->greedy_policy(s), o_player = s->greedy_policy(s))
  ╠═╡ =#

# ╔═╡ 607f0859-8a67-45f7-9e6d-e407d0febf26
#=╠═╡
md"""
Greedy Game Step: $(@bind greedy_game_step Slider(1:length(greedy_game[1]); show_value=true))
"""
  ╠═╡ =#

# ╔═╡ 3d1cf9bb-1b75-4070-bc3e-766a4da7ffba
#=╠═╡
greedy_game[1][greedy_game_step] |> s -> begin
	HTML(html_board(s; cell_size = 30))
end
  ╠═╡ =#

# ╔═╡ 36d2c036-0167-44a2-b9ec-343f831c1893
#=╠═╡
calculate_x_winrate(board, greedy_policy, greedy_policy)
  ╠═╡ =#

# ╔═╡ 14ef1213-3a2c-48c5-899d-986dfad9fef1
#=╠═╡
calculate_x_winrate(board, defensive_policy, defensive_policy)
  ╠═╡ =#

# ╔═╡ 8fcc9a0c-8ed5-4cc7-a907-1ff41c929f23
#=╠═╡
calculate_x_winrate(board, positional_policy, positional_policy)
  ╠═╡ =#

# ╔═╡ 90944844-7709-4547-ba6d-2821082648f6
#=╠═╡
calculate_x_winrate(board, greedy_policy, s -> rand(available_moves(s)))
  ╠═╡ =#

# ╔═╡ 9fdc2886-b390-42be-965a-227c56a0d629
#=╠═╡
calculate_x_winrate(board, s -> rand(available_moves(s)), greedy_policy)
  ╠═╡ =#

# ╔═╡ f1df3bd5-0019-4f6b-8073-42362048ed8a
md"""
## Strategy Winrates
"""

# ╔═╡ f0eb4b7b-5ae9-450b-973c-495ebe09a3e2
md"""
### Greedy vs Defensive
"""

# ╔═╡ 0f3e8379-22bc-4aad-95d0-7741b9d1ae03
#=╠═╡
calculate_x_winrate(board, greedy_policy, defensive_policy)
  ╠═╡ =#

# ╔═╡ dfb0c194-322b-445e-aace-409d7b80e99e
#=╠═╡
calculate_x_winrate(board, defensive_policy, greedy_policy)
  ╠═╡ =#

# ╔═╡ 268840e0-fefd-4d8a-a53b-3fabd9e70548
md"""
### Greedy vs Positional
"""

# ╔═╡ b4cd4c2c-6f27-499d-9510-27ac1b4b36b1
#=╠═╡
calculate_x_winrate(board, greedy_policy, positional_policy)
  ╠═╡ =#

# ╔═╡ 2a46b458-9f23-4659-9d00-bba1e90a5b74
#=╠═╡
calculate_x_winrate(board, positional_policy, greedy_policy)
  ╠═╡ =#

# ╔═╡ 59978a88-92d6-4964-a580-41407144959d
md"""
### Defensive vs Positional
"""

# ╔═╡ 4ccdca95-1e86-4877-91a4-f328aefb2915
#=╠═╡
calculate_x_winrate(board, defensive_policy, positional_policy)
  ╠═╡ =#

# ╔═╡ ee9d565d-8d26-4768-bfbc-938c145f887b
#=╠═╡
calculate_x_winrate(board, positional_policy, defensive_policy)
  ╠═╡ =#

# ╔═╡ 76d3fa3b-a229-4800-9dd2-38e80346ae47
md"""
# MDP Environment

We can turn this two player game into an MDP by fixing an opponent.  We could use a random opponent or any one of the heuristic strategies shown above such as the greedy policy.  Once we have the MDP environment, we can test various learning algorithms and compare them to MCTS.
"""

# ╔═╡ 40e3da31-3fd2-446c-9dc9-bbec430e794b
md"""
## Create MDP with Opponent Strategy
"""

# ╔═╡ 72ea5a15-2b07-4c0d-80aa-802f50bc433a
#=╠═╡
"""
    make_gridcapture_mdp(N, K, opponent::Function; agent_player=:x)

Create a GridCapture MDP with a specified opponent strategy.  N is the board size, K is the win condition (K-in-a-row), opponent is a function that returns actions given the state, and agent_player indicates whether the agent plays as :x or :o.
"""
function make_gridcapture_mdp(
    N::Int, K::Int, opponent::Function; agent_player=:x
)
    
    # Create list of all possible actions
    all_actions = vec(Tuple{Int, Int}[(r, c) for c in 1:N, r in 1:N])
    
    # Initial state function
    if agent_player == :x
        state_init = () -> GameState{N, K}()
    else
        state_init = () -> begin
            s = GameState{N, K}()
            opp_move = opponent(s)
            place_stone(s, opp_move...)
        end
    end
    
    isterm(s::GameState) = check_game_result(s) != :ongoing
    
    # Step function
    function step(s::GameState, i_a::Int)
        action = all_actions[i_a]
        
        # If game already over, return zero reward and same state
        isterm(s) && return (0.0f0, s)
        
        new_state = place_stone(s, action...)

        result = check_game_result(new_state)

        # If the game is over after one move then immediately return the reward and current state
        (result != :ongoing) && return (score_game(new_state), new_state)
        
        opp_move = opponent(new_state)
        final_state = place_stone(new_state, opp_move...)
        final_result = check_game_result(final_state)

        final_result == :ongoing && return (0f0, final_state)

        return (score_game(final_state), final_state)
    end
    
    transition = StateMDPTransitionSampler(step, state_init())
    is_valid_action(s::GameState, i_a::Int) = is_valid_move(s, all_actions[i_a]...)
    action_index = Dict(action => i for (i, action) in enumerate(all_actions))
    
    return StateMDP(all_actions, transition, state_init, isterm; 
                    is_valid_action=is_valid_action, action_index=action_index)
end
  ╠═╡ =#

# ╔═╡ bd4ba7de-9e3b-4b09-bad8-d20cce0c1c7a
random_policy(s::GameState) = rand(available_moves(s))

# ╔═╡ 49c0564a-9a29-4c2d-8559-7b8944fa5219
#=╠═╡
const gridcapture_random_mdp = make_gridcapture_mdp(6, 4, s -> rand(available_moves(s)))
  ╠═╡ =#

# ╔═╡ 684b1c27-2c72-41ba-841a-ec08b5bae3c4
#=╠═╡
const gridcapture_greedy_mdp = make_gridcapture_mdp(6, 4, greedy_policy)
  ╠═╡ =#

# ╔═╡ b4a16aa8-2411-4e26-a66a-f6fd1375430d
#=╠═╡
const gridcapture_defensive_mdp = make_gridcapture_mdp(6, 4, defensive_policy)
  ╠═╡ =#

# ╔═╡ 4f9fca26-f1b2-4fbc-954d-1d32211edb56
#=╠═╡
const gridcapture_positional_mdp = make_gridcapture_mdp(6, 4, positional_policy)
  ╠═╡ =#

# ╔═╡ 3924aab4-8d85-422a-892e-0f3950ffec3a
md"""
## Visualizing Episodes Against Fixed Opponents
"""

# ╔═╡ 25e97cff-adaa-41e8-864f-9d9d55069fa3
#=╠═╡
@bind mdp_select Select([gridcapture_random_mdp => "Random Opponent", gridcapture_greedy_mdp => "Greedy Opponent", gridcapture_defensive_mdp => "Defensive Opponent", gridcapture_positional_mdp => "Positional Opponent"])
  ╠═╡ =#

# ╔═╡ bf2cbd82-8bd6-41d5-b9d1-0de6f1521565
#=╠═╡
mdp_episode = runepisode(mdp_select)
  ╠═╡ =#

# ╔═╡ 2cf6d815-2099-473f-b4c3-99c712541787
#=╠═╡
md"""
Select Episode Step: $(@bind mdp_episode_step Slider(1:mdp_episode[5]+1))

Final Outcome: $(check_game_result(mdp_episode[4]))
"""
  ╠═╡ =#

# ╔═╡ 9da644b0-c096-4fc9-a8e3-7575a32fb670
#=╠═╡
html_board(vcat(mdp_episode[1], mdp_episode[4])[mdp_episode_step]) |> HTML
  ╠═╡ =#

# ╔═╡ c6a1d201-6f71-4a09-8cf5-8ca1f058ee7d
md"""
## Baseline Performance
"""

# ╔═╡ 0391c2b3-33f4-4ffa-bd09-ae381a3fd587
#=╠═╡
TabularRL.average_stochastic_rollout(10000, gridcapture_random_mdp, make_random_policy(gridcapture_random_mdp), 1f0)
  ╠═╡ =#

# ╔═╡ 74154178-6c1f-444e-bf23-d28bde19cb2c
#=╠═╡
TabularRL.average_stochastic_rollout(10000, gridcapture_greedy_mdp, make_random_policy(gridcapture_greedy_mdp), 1f0)
  ╠═╡ =#

# ╔═╡ 1146218c-da98-4012-9d51-e14c223b929e
#=╠═╡
TabularRL.average_stochastic_rollout(10000, gridcapture_defensive_mdp, make_random_policy(gridcapture_greedy_mdp), 1f0)
  ╠═╡ =#

# ╔═╡ ae69ab54-dfb1-4a8b-8e3e-cfb0281a99bb
#=╠═╡
TabularRL.average_stochastic_rollout(10000, gridcapture_positional_mdp, make_random_policy(gridcapture_greedy_mdp), 1f0)
  ╠═╡ =#

# ╔═╡ 179a2fce-e367-4db2-a563-604aadb00122
md"""
## Feature Vector Construction
"""

# ╔═╡ 0e237d91-ece1-485f-aedf-c29187955315
"""
    update_features!(features, state::GameState{N,K}) where {N,K}

Update feature vector with current state.  By default the feature vector will use a value of 1 for x, -1 for 0, and 0 for empty.  This is an in-place update version of state_to_features.
"""
function update_features!(features::Vector{T}, state::GameState{N,K}) where {N,K,T<:Real}
    features .= zero(T)
    @inbounds @simd for i in eachindex(features)
        features[i] += T(state.x_pieces[i])
        features[i] -= T(state.o_pieces[i])
    end
    return features
end

# ╔═╡ 16de0d9b-c411-490f-99c3-c70c6e874741
"""
    state_to_features(state::GameState{N,K}) where {N,K}

Convert GameState to feature vector for neural network.  By default the feature vector will use a value of 1 for x, -1 for 0, and 0 for empty.
"""
function state_to_features(state::GameState{N,K}) where {N,K}
    v = zeros(Float32, N*N)
    update_features!(v, state)
end

# ╔═╡ cfd60a54-6f93-453d-97c5-75574702a47e
md"""
## RL Value Training
"""

# ╔═╡ 68b919cd-51b1-4450-b6a5-a8f1d96248bf
#=╠═╡
const gridcapture_random_value_setup = setup_episodic_value_nonlinear_training(gridcapture_random_mdp, state_to_features(gridcapture_random_mdp.initialize_state()), update_features!; min_reward = -1f0)
  ╠═╡ =#

# ╔═╡ d0e04b83-9b6a-4653-afd4-2f2914dc067f
#=╠═╡
const gridcapture_greedy_value_setup = setup_episodic_value_nonlinear_training(gridcapture_greedy_mdp, state_to_features(gridcapture_greedy_mdp.initialize_state()), update_features!; min_reward = -1f0)
  ╠═╡ =#

# ╔═╡ a8bd57dc-239e-41bd-b1c4-a467a65da4d4
#=╠═╡
const gridcapture_defensive_value_setup = setup_episodic_value_nonlinear_training(gridcapture_defensive_mdp, state_to_features(gridcapture_defensive_mdp.initialize_state()), update_features!; min_reward = -1f0)
  ╠═╡ =#

# ╔═╡ 5c23b887-cd6d-4913-9704-c3552e8deb2b
#=╠═╡
const gridcapture_positional_value_setup = setup_episodic_value_nonlinear_training(gridcapture_positional_mdp, state_to_features(gridcapture_positional_mdp.initialize_state()), update_features!; min_reward = -1f0)
  ╠═╡ =#

# ╔═╡ 36056f87-3a19-49cb-b322-c44c3655e59e
#=╠═╡
const fcann_random_value_solution = gridcapture_random_value_setup.train_ϵ_decay([64, 64], 1, 1f0, 0.1f0, 0.99f0, 10_000; ϵ_min = 0.001f0)
  ╠═╡ =#

# ╔═╡ a37ff490-f724-413e-957e-9e137b090019
#=╠═╡
const fcann_greedy_value_solution = gridcapture_greedy_value_setup.train_ϵ_decay([64, 64], 1, 1f0, 0.1f0, 0.99f0, 10_000; ϵ_min = 0.001f0)
  ╠═╡ =#

# ╔═╡ 72fcf8f4-dab0-4f59-ad08-dbefd94157a8
#=╠═╡
const fcann_defensive_value_solution = gridcapture_defensive_value_setup.train_ϵ_decay([64, 64], 1, 1f0, 0.1f0, 0.99f0, 10_000; ϵ_min = 0.001f0)
  ╠═╡ =#

# ╔═╡ 2a3100f8-8f44-4d63-97da-2eebd2e0a8dc
#=╠═╡
const fcann_positional_value_solution = gridcapture_positional_value_setup.train_ϵ_decay([64, 64], 1, 1f0, 0.1f0, 0.99f0, 10_000; ϵ_min = 0.001f0)
  ╠═╡ =#

# ╔═╡ 94a708a1-7621-4390-8852-a37d65b4e0da
#=╠═╡
positional_value_game = play_gridcapture_game(board; x_player = s->action_index_to_move(fcann_positional_value_solution.value_function(s).maximizing_action, 6), o_player = positional_policy)
  ╠═╡ =#

# ╔═╡ c16dfc51-44e7-45c0-a85d-831b713f0e66
#=╠═╡
@bind positional_value_game_step Slider(1:length(positional_value_game[1]))
  ╠═╡ =#

# ╔═╡ 5c04c776-015b-4b51-ac68-c4e359fb323d
#=╠═╡
positional_value_game[1][positional_value_game_step] |> s -> begin
	HTML(html_board(s; cell_size = 40, policy = player_turn(s) == 1 ? fcann_positional_value_solution.value_function(s).action_values |> make_greedy_policy! : positional_policy_distribution(s)))
end
  ╠═╡ =#

# ╔═╡ 8c3565c6-09f6-4b4e-9e17-6d10fb637e77
md"""
## RL Policy Training
"""

# ╔═╡ a575b693-fae2-4c3c-b9d6-b125f2773530
#=╠═╡
begin
	const gridcapture_random_policy_setup = setup_episodic_policy_nonlinear_training(gridcapture_random_mdp, state_to_features(gridcapture_random_mdp.initialize_state()), update_features!; min_reward = -1f0)
	const gridcapture_greedy_policy_setup = setup_episodic_policy_nonlinear_training(gridcapture_greedy_mdp, state_to_features(gridcapture_greedy_mdp.initialize_state()), update_features!; min_reward = -1f0)
	const gridcapture_defensive_policy_setup = setup_episodic_policy_nonlinear_training(gridcapture_defensive_mdp, state_to_features(gridcapture_defensive_mdp.initialize_state()), update_features!; min_reward = -1f0)
	const gridcapture_positional_policy_setup = setup_episodic_policy_nonlinear_training(gridcapture_positional_mdp, state_to_features(gridcapture_positional_mdp.initialize_state()), update_features!; min_reward = -1f0)
end
  ╠═╡ =#

# ╔═╡ 48199c60-dc5b-4131-bd7d-43b75c0b6c1c
#=╠═╡
const fcann_random_policy_solution = gridcapture_random_policy_setup.train_rate_decay([64, 64], 1, 1f0, 0.3f0, 0.1f0, 0.99f0, 0.99f0, 10_000)
  ╠═╡ =#

# ╔═╡ f86fd6b6-1438-4d4b-8566-93a6dcd8e62b
#=╠═╡
const fcann_positional_policy_solution = gridcapture_positional_policy_setup.train_rate_decay([64, 64], 1, 1f0, 0.3f0, 0.1f0, 0.99f0, 0.99f0, 10_000)
  ╠═╡ =#

# ╔═╡ 5ed3ef2c-729e-4aa4-8c6c-12b8ccffa102
#=╠═╡
kwarg_test = fcann_positional_policy_solution.form_policy_kwargs()
  ╠═╡ =#

# ╔═╡ 883853ff-d4fb-4ad1-b610-2d8a86f571d8
#=╠═╡
positional_policy_game = play_gridcapture_game(board; x_player = s->gridcapture_positional_mdp.actions[fcann_positional_policy_solution.policy_sample_action(s)], o_player = positional_policy)
# positional_policy_game = play_gridcapture_game(board; x_player = positional_policy, o_player = positional_policy)
  ╠═╡ =#

# ╔═╡ 888a50ac-ef39-4705-b3c3-16c252ab7b40
#=╠═╡
@bind positional_policy_game_step Slider(1:length(positional_policy_game[1]))
  ╠═╡ =#

# ╔═╡ 90ce5f4d-5104-4011-8164-f8c95077109c
#=╠═╡
positional_policy_game[1][positional_policy_game_step] |> s -> begin
	HTML(html_board(s; cell_size = 30, policy = player_turn(s) == 1 ? fcann_positional_policy_solution.policy_function(s) : positional_policy_distribution(s)))
end
  ╠═╡ =#

# ╔═╡ 8e57a6c4-3d87-4ad0-a42d-619d9f96cf83
md"""
## AlphaZero Overview

The AlphaZero algorithm is based on combining trained value and policy networks from deep reinforcement learning with the real time planning capabilities of Monte Carlo tree search.  Although its primary use case is in turn taking two player games, the core ideas apply to episodic MDP environments as well.  For now, we will focus on the MDP case to avoid complications from turn taking games.  We can think of the algorithm as having the following major components:

1. Given an existing parameterized function that calculates a policy and associated value estimate ``(u, p) = f(s; \theta)`` where ``\pi(s) \sim p`` and ``u \approx v_\pi(s)`` use simulations through a search tree of the environment to select actions according to an improved policy.  From an environment state ``s``, a simulation budget is used to collect data within the search tree making use of ``f``.  That data is then used to select an action that may differ from ``\pi``.  The goal is that by following this *tree policy* at each state will result in better performance in the environment.

2. Use the tree search to collect data following trajectories in the environment.  Since we have an episodic environment guaranteed to terminate after a reasonable number of steps, we will have complete trajectories regardless of ``f``.  Since these trajectories represent the *tree policy*, if the tree policy is in fact an improvement, we can use the collected data to train ``f`` so that its new behavior matches the tree policy.  Each trajectory will have a reward outcome, so these outcomes can directly train the value component of ``f`` with Monte Carlo sampling.  The policy function is trained to directly match the behavior of the tree policy with cross entropy loss.

For this session, we will focus on part 1

## Policy Improvement Overview

Let's say we have a policy ``\pi(s)`` that can select actions in an environment.  We define the following value functions to measure the performance of such a policy:

``v_\pi(s) = \mathbb{E}_\pi[R_t + \gamma R_{t+1} + \cdots \vert S_t = s]``

``q_\pi(s, a) = \mathbb{E}_\pi[R_t + \gamma R_{t+1} + \cdots \vert S_t = s, A_t = a]``

where ``0 \gt \gamma \lt 1`` is the *discount factor* to ensure that the sum of discounted rewards converges for all states.

### Policy Improvement Theorem

We say that a policy ``\pi^\prime(s)`` is *improved* if ``v_{\pi^\prime}(s) \geq v_\pi(s) \forall s in \mathcal{S}``.  That is its expected value is higher at all states in an environment.  The *policy improvement theorem* provides a way to verify that a new policy is improved by using only the value functions from the original policy.  In particular, we need only verify that the followig is true at a single state:

``q_\pi(s, \pi^\prime(s)) \geq v_\pi(s) \implies v_{\pi^\prime} \geq v_\pi(s) \forall s \in \mathcal{S}``

So, if we define a new policy that has a different action selection at a particular state ``s`` such that the *action-value* measured under the orignal policy's value function is higher, then that policy is improved globally.  In order for this theorem to hold, we must have a way of constructing a policy that differs only at a single state where we can verify the improved action value.  Tabular solutions meet this criteria as well as the type of local tabular solutions we form when doing planning algorithms like MCTS.

### Rollout Algorithms

Let's say we have an environment that terminates after a finite number of steps even with random behvaior.  We have a policy function ``\pi(s)`` available to us whether it is predefined with heuristics or trained using an RL algorithm.  It could even be a neural network function with parameters that is a policy network or a value network from which a greedy action is selected.  The nature of the function is not important.  Starting from an environment state ``s``, we can use the policy function to select actions, but what if we have a computational and time budget to work with on each step of action selection?  We could then use a rollout algorithm to perform policy improvement in real time.  The idea behind a rollout algorithm is very simple: use Monte Carlo sampling to build estimates of ``q_\pi(s, a)`` and then select the action according to ``\mathrm{argmax}_a Q_\pi(s, a)`` where ``\mathbb{E}  \left [ Q_\pi(s, a) \right ] = q_\pi(s, a)``.  We build ``Q_\pi(s, a)`` as follows: 

Starting from state ``s`` select each available action ``s \in \mathcal{A}`` and sample a transition state ``s^\prime`` from the environment.  From each transition state, use the policy ``\pi(s^\prime)`` to select the next action and continue using the policy follow a trajectory until a terminal state is reached:

``s, \mathcal{A}_1 \overset{R_1}{\rightarrow} s^\prime, \pi(s^\prime) \overset{R_2}{\rightarrow} s^{\prime \prime} \cdots`` 

``s, \mathcal{A}_2 \overset{R_1}{\rightarrow} s^\prime, \pi(s^\prime) \overset{R_2}{\rightarrow} s^{\prime \prime} \cdots`` 

``s, \mathcal{A}_3 \overset{R_1}{\rightarrow} s^\prime, \pi(s^\prime) \overset{R_2}{\rightarrow} s^{\prime \prime} \cdots`` 

``\vdots``

Each of these trajectories could be sampled multiple times to get better estimates.  If we sample enough that the expected values converge to the true values, then we can guarantee selecting the argmax will produce a better policy.  Whatever our sampling budget is though, we will improve the policy in expectation.  One important caveat, however, is that we cannot use this approach unless we collect at least one sample trajectory for each action.  Otherwise we cannot select the argmax.

### Connection to MCTS

Whenever a transition state is visited for the first time using MCTS, it is evaluated using some value function.  In the absence of any trained function, we can fall back to using a rollout estimator with a policy (as a starting point we usually use the random policy).  Let's say we have an environment with ``k`` available actions from state ``s``.  If we run MCTS with ``k`` simulations, then each action will be sampled exactly once due to the MCTS selection criteria (unvisited actions are given an infinite bonus).  

Where MCTS can be more effective is when the simulation count greatly exceeds ``k``.  Where a rollout algorithm may evenly distribute simulations to each action, MCTS will prioritize actions based on the preliminary estimates of the action values.  Additionally, more transition states are added to the tree each with their own action value statistics, although for these estimates to be useful, all available actions must be sampled at those states as well.  We the number of actions is small compared to the number of simulations this might be fine.

Therefore, given a policy function, we can pursue policy improvement with a rollout strategy as long as we have enough simulations to cover the action space.

### Dealing with Large Action Spaces

What if the number of simulations we can practically execute in a time budget is smaller than the available actions?  This constraint makes vanilla MCTS useless since we never accumulate all of the action values.  Is there some other way to use an existing policy to make MCTS more practical?  If we have a policy function ``\pi(s)`` that provides not just an action sample, but a distribution over actions then we can use that policy distribution to narrow the score of tree search.  

"""

# ╔═╡ 87f9eb9b-d542-4ba6-9ba4-9590138a49a7
md"""
## Vanilla MCTS Policy
"""

# ╔═╡ 999f90e7-a87a-4edf-9e81-c4f8493f1c69
md"""
### Algorithm Explanation

An MCTS search is done from a root node at the state for which we need an action selection.  The goal is to estimate the action values at this state and then select an action as we usually would by taking ``\mathrm{argmax}_a \hat Q(s, a)``.  During the MCTS search, these action values are accumulated with temporal difference updates using the environment to make transitions.  The search maintains statistics of visited state-action pairs that is used to direct the search through actions that are promising.  Each such search is called a *simulation* because it simulates a trajectory in the environment using the *tree policy*.  

The tree policy uses the current statistics in the tree to make action selections based on the action values.  Typically in RL value learning, we follow an ``\epsilon`` greedy policy with respect to the action values and the factor of ``\epsilon`` guarantees some degree of exploration.  Since we are saving tables of values, MCTs is most easily comparable to tabular Q-learning.  Both methods perform temporal difference learning updates on the action values with the goal of policy improvement based on the Bellman Optimality Equations:

```math
\begin{flalign}
Q^*(s, a) &= \mathbb{E} \left [ R_{t+1} + \gamma * \max_{a^\prime} Q^* (S_{t+1}, {a^\prime}) \mid S_t = s, A_t = a \right ] \\
&= \mathbb{E} \left [ R_{t+1} + \gamma * V^* (S_{t+1}) \mid S_t = s, A_t = a \right ] 
\end{flalign}
```

The properties of this operator allow us to stochastically approach the optimal value function by applying this relationship to every visited state in an environement while sampling state-action pairs.  These stochastic updates match the Bellman equation in expectation.  The update below is what is used in tabular Q-learning:

```math
\hat Q(S_t, A_t) \leftarrow R_{t+1} + \gamma * \max_{a^\prime} \hat Q(S_{t+1}, a^\prime)
```

We can also perform n-step updates all the way to monte carlo sampling and keep the same expectation as long as the steps taken are done using the policy.

```math
\hat Q(S_t, A_t) \leftarrow R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^n \max_{a^\prime} \hat Q(S_{t+n, a^\prime})
```

#### Value Updates

Under this approach, we must visit all state-action pairs in an environment with some frequency because the solution is global.  The Q values are only correct when this relationship holds for all state action pairs.  With an MCTS search, we accumulate the same state-action value estimates, but we never complete a global solution.  Instead we pursue the tree until we reach an unvisited state which we have no action values for.  At this point in the traversal, we cannot use the Q-learning approach to complete our value update.  Instead, we must fallback on the second version of the Bellman equation in which we replace the maximum value of the action-value function with the state value function.  Of course, this update is only valid when the state value function is correct.  All we can do in MCTS is provide some attempt to estimate the state value which is commonly a rollout function which selects random actions until a terminal state is reached.  The average performance of such random rollouts is an unbiased state value estimate of the random policy.  Ideally, we want a state value estimate of the *tree policy* but that is never available.

Let's say we have some value estimation function ``\hat v(s)`` which we can use to estimate the value of our first unvisited state.  Note that I'm using capital letters like ``\hat Q(s, a)`` to indicate a tabular estimate with saved values and lowercase letters like ``\hat v(s)`` to indicate a function that can produce a value for any input.  From there, we can perform our temporal difference updates as follows:

```math
\begin{flalign}
&S_0, A_0 \overset{R_1}{\rightarrow} S_1, A_1 \overset{R_2}{\rightarrow} \cdots \overset{R_n}{\rightarrow} S_n \text{ (first unvisited state)} \\
&\hat Q(S_{n-1}, A_{n-1}) = R_n + \gamma \hat v(S_n) \\
&\hat Q(S_{n-2}, A_{n-2}) = R_{n-1} + \gamma R_n + \gamma \hat v(S_n) \\
&\vdots \\
& \hat Q(S_0, A_0) = R_1 + \gamma R_2 + \cdots + \gamma^n \hat v(S_n)
\end{flalign}
```

If the tree becomes deep enough that the first unvisited state reached is terminal, then a true monte carlo update can occur where the final state value is 0 and the only component in the tree updates are sampled reward values.  So the action-value updates are similar to n-step Q learning but with an additional state value estimate instead of the usual maximization over estimated action values, but how are the trajectories actually created?
"""

# ╔═╡ 7e0c91a5-c1f0-4ea1-b450-52daca234cde
md"""
#### Simulation Action Selection

In tabular Q-learning, we use ``\epsilon`` greedy action selection to ensure enough exploration to have a global solution.  Alternatively, we could just follow random actions, but that may be sample inefficient since we care about accurate values close to the optimal policy (even though we need accurate values in every state to ensure the accuracy anywhere).  In MCTS we have a limited simulation budget and a known inability to calculate a global solution, so the sampling strategy must be different.  Instead of ``\epsilon`` greedy action selection, we pursue the *tree policy* at every node, including the root node, for action selection during the simulation phase.  The goal of the tree policy is to target data collection towards states that appear to have a high value, while accounting for the sampling uncertainty or other states.  One important consequence of the UCB selection criteria used in the tree policy is that unvisited actions are always simulated at every state.  In other words, none of the actions are sampled more than once until every action has been sampled at least once.  If we have an environment with 100 actions at the root node, then running MCTS with 100 simulations would yield a single sample from each action and if these all map to a unique state then our only insight into action selection would be a single evaluation of our state value estimator at each transition state.  

In environements with large action spaces, this limitation can hamper the effectiveness of MCTS since the UCB selection does nothing but spread out the simluations evenly.  Ideally, we would have enough of a budget to start to see the non-uniform sampling that would indicate improved knowedge about the action values.  The extensions to MCTS discussed later address this problem by limiting the breadth of action selection through other means.  The U(pper) C(onfidence) B(ound) action selection criteria is a deterministic function given below:

```math
A_n = \begin{cases}
a &\text{ if } N(S_n, a) = 0 \\
\mathrm{argmax}_a \left ( \hat Q(S_n, a) + \sqrt{\frac{2 \ln N(S_n)}{N(S_n, a)}} \right ) &\text{ otherwise}
\end{cases}
```

where ``\hat Q`` are the tree action-values and ``N`` are the visit counts to each state-action pair.  Without the ``\sqrt{}`` term, the second case is just greedy action selection.  Assuming all actions have a non-zero count, this deterministic alternative to ``\epsilon`` greedy action selection attempts to account for sampling error but giving a bonus value to infrequently visited actions.  If the sampled action-values followed a normal distribution, this bonus would be equivalent to adding a term close to the standard deviation of the sample distribution in order to counteract the bias from low samples.  For actions with low samples, the low values could be the result of collecting a sample at the lower end of the distribution rather than a reflection of its true value.

"""

# ╔═╡ 3cff602c-a889-40f1-be74-74be117d3b0b
md"""
#### Final Action Selection
After performing all simulations, MCTS has a collection of tree values, but we only care about the final action selection at the root node.  During simulation, the goal was to improve the accuracy of value estimates, but for the policy itself, the goal is to perform as well as possible with the information at hand.  As such, the greedy policy is used to select the root node action with respect to the current state-action values: ``a = \mathrm{argmax}_{a^\prime} \hat Q(S_0, a^\prime)`` where now ``a`` represents the action selected by the MCTS policy.  This entire process is used to select just one action at the root node.  When the next state is encountered, the entire search is repeated, although it is possible to add to the existing tree accumulated so far rather than starting a fresh one.
"""

# ╔═╡ 9ff655da-7b92-4797-b953-d2413feb4aed
md"""
### Gridcapture Example
"""

# ╔═╡ 49a67540-f964-4209-a6c3-07dd66e83d84
function create_mcts_policy_evaluation(mdp::StateMDP, rollout_n::Integer; rollout_policy::Function = make_random_policy(mdp))
	v_est(mdp, s, γ) = TabularRL.average_stochastic_rollout(rollout_n, mdp, rollout_policy, γ)
	function π(s; kwargs...)
		monte_carlo_tree_search(mdp, 1f0, v_est, s; kwargs...)
	end

	function generate_episode(;kwargs...)
		s0 = mdp.initialize_state()
		(i_a0, counts0, qvals0) = π(s0; kwargs...)
		state_history = [s0]
		action_history = [i_a0]
		count_history = [counts0]
		Q_history = [qvals0]
		(r, s) = mdp.ptf(s0, i_a0)
		reward_history = [r]
		while !mdp.isterm(s)
			(i_a, counts, qvals) = π(s; kwargs...)
			push!(state_history, s)
			push!(action_history, i_a)
			push!(count_history, counts)
			push!(Q_history, qvals)
			(r, s) = mdp.ptf(s, i_a)
			push!(reward_history, r)
		end

		return (states = state_history, actions = action_history, rewards = reward_history, s_term = s, mcts_counts = count_history, mcts_qs = Q_history)
	end

	return (mcts_policy = π, generate_mcts_episode = generate_episode)
end

# ╔═╡ 17fe1925-5e87-4abf-ba34-dff183df9638
#=╠═╡
const random_mcts_evaluation = create_mcts_policy_evaluation(gridcapture_defensive_mdp, 10; rollout_policy = s -> gridcapture_defensive_mdp.action_index[defensive_policy(s)])
  ╠═╡ =#

# ╔═╡ 5735640d-c8a6-4753-8425-8cdb1e9a4bf5
#=╠═╡
random_mcts_episode = random_mcts_evaluation.generate_mcts_episode(; nsims = 200, depth = 1_000)
  ╠═╡ =#

# ╔═╡ 7f160dd5-654a-4bd9-abd8-62ef5baddaa8
#=╠═╡
@bind random_mcts_episode_step Slider(1:length(random_mcts_episode.states)+1; show_value=true)
  ╠═╡ =#

# ╔═╡ 08616411-c11d-4ac1-aab3-07a21d25a30d
#=╠═╡
function display_mcts_episode_step(mcts_episode::NamedTuple, step::Integer)
	if step > length(mcts_episode.states)
		s = mcts_episode.s_term
		HTML(html_board(s))
	else
		s = mcts_episode.states[step]
		counts = mcts_episode.mcts_counts[step][s]
		visit_dist = counts ./ sum(counts)
		action_values = zeros(Float32, length(visit_dist))
		action_values .= -Inf
		for i in counts.nzind
			action_values[i] = mcts_episode.mcts_qs[step][s][i]
		end
		policy = copy(action_values)
		greedy_policy = make_greedy_policy!(policy)

		(@htl("""
		<div style = "display: flex;">
		<div>
			 <div>Visit Count Distribution</div>
		$(HTML(html_board(s; policy = visit_dist)))
		</div>
			 <div>
		<div>Greedy Policy</div>
		$(HTML(html_board(s; policy = greedy_policy)))
	    </div>
		</div>
		"""), collect(counts), collect(action_values))
	end
end
  ╠═╡ =#

# ╔═╡ faa837bc-3c78-4198-b34a-6deddc872959
#=╠═╡
display_mcts_episode_step(random_mcts_episode, random_mcts_episode_step)
  ╠═╡ =#

# ╔═╡ f7633ae5-e036-428c-a85f-4352d400a91a
md"""
## Gumbel Top-K MCTS
"""

# ╔═╡ 0732370c-785f-4f4c-9d31-94d985fccc71
md"""
### Using Search to Improve an Existing Policy

Scenario: We have an agent that operates in an MDP environment with a policy function that can calculate for a given state: 1) a distribution over actions from which we can sample 2) a corresponding value estimate.  We could sample from the policy distribution to make action selections at the time of deployment, but we have a simulation budget to use for each action selection.  That is to say we can evaluate the policy function multiple times and simulate transitions in the environment with the hopes of improving the performance of the policy in real time.  Gumbel top-k MCTS is a method to guarantee policy improvement in expectation as follows:

#### Algorithm Hyperparameters

- Top-k: The number of actions to consider at the root node (environment state).  Given a very large action space, this value can reduce the breadth of the search at every layer of the tree.

- Nsims: The number of total simulations allowed for search.  Each simulation will traverse the tree until an unvisited state is reached.  That state will then be evaluated by the policy function to produce a value estimate and a distribution.  Each simulation will therefore entail one new evaluation of the policy function as well as the other steps needed to traverse the tree to a node.

- Cvisit and Cscale: Constants used to calculate improved policy.  Controls the relative importance of the tree action-values compared to the policy prior, however, all values guarantee expected policy improvement.  Usually remain fixed at 50 and 1 respectively.

#### Selecting Actions to Search at the Root Node

First, the algorithm selects which actions to consider from the search state ``s``.  The Top-k hyperparameter determines how many total actions will be considered.  With a fresh tree, the only information available to distinguish actions is from the policy prior.  Therefore, the algorithm simply samples from this distribution using the *Gumbel Top-k* trick.  Gumbel noise is generated for every action and then added to the logits of the policy distribution.  This score can then be used to rank actions and deterministically select the top-k which is equivalent to sampling k actions without replacement from the distribution.  The same gumbel noise values are used in combination with results from the tree search to allocate simulations to the actions.

#### Selecting Actions at the Root Node

Given a budget of Nsims, the algorithm must distribute simulations to the actions in order to collect more information about the action values.  Sequential halving is used to divide the search into phases with the goal of distributing the simulations evenlly between the phases.  In phase 1, the simulations are evenly divided among the top-k actions.  Let's say there are 16 actions and 200 simulations.  25 simulations would be allocated to 4 phases which would pare down the actions from 16 to 8 to 4 to 2.  Each phase allocates the available simualations evenly between the considered actions and then re-ranks them by score.  The score, however, now also uses the action values observed from the simulation.  Every time an action is selected, the next transition state is observed which will intially be approximated by the value function.  After subsequent visits, however, that value is replaced by the action value observed from that state using another selection process.  All of the actions under consideration for the next phase will have a similar or equal number of visits so the information from the prior and the new q-values will be on par for all actions.  At the end of sequential halving, the action with the highest score is selected from the set of the most visited actions.  If the final phase has 2 actions remaining, then only these two actions will be considered for the final selection.  Also, towards the end of the process, most of the score weight is on the action values rather than the logits.

#### Selecting Actions at Non-Root Nodes

An improved policy is derived from the tree statistics based on a combination of the logits and the action values.  This policy is calculated using the same score used to select actions at the root node.  At non-root nodes, however, some actions are never visited, so their score must default to a value which is either the value function estimate or some combination of the existing q-values and that.  Either way, all unvisited actions are given the same score, so they can only be distinguished by the logits.  Based on the current tree statistics, the action is selected that would minimize the difference between the empirical visit distribution and the improved policy distribution after the visit takes place.  Alternatively, actions could be sampled stochastically from the improved policy distribution.  Both methods have the same long run expectation but the deterministic action selection reduces the variance.

### Using Search to Collect Data for Policy and Value Training

#### Training Networks

After running mcts search from a root node, the root statistics include the visit counts and updated action values.  There is also the prior logits.  That information was used to calculate the score that is used in sequential halving but it can also be used to calculate the improved policy distribution after the search is complete.  It may not match the action selection during search because subsequent visits to states may lower their action values below that of actions that have already been eliminated.  This retrospective improved policy can be used as a target for a policy network by training to minimize KL divergence between the two policies.

The value network can be trained after search has been performed across every state in an environment episode.  Once the episode ends, we can add up the rewards and have a Monte Carlo sampling target to use directly for the value function training with squared error loss.

- How large should the replay buffer be?
- How much many training steps should occur before updating parameters used to generate data?
- How many simulations to use during training?
"""

# ╔═╡ d1060ffe-0ea7-4e6c-802c-3073c59f665f
md"""
# Environment Introduction

Tic-Tac-Toe variants with NxN boards and M in a row win conditions.  Normally two player game, but we can turn this into an MDP by fixing an opponent strategy.  Below is a visualization of how one of the benchmark policies plays on an 8x8 board with a 4-in-a-row win condition.
"""

# ╔═╡ 868602f2-1e57-4a9f-93ed-a73e7c461b99
md"""
## 8x4 Game Greedy Policy Demonstration

The visualizations below also introduce the way the game board is overlayed with policy information and other information to follow.
"""

# ╔═╡ 851ac26a-4f49-423a-9558-77d081b7791b
#=╠═╡
const greedy_8x4_game = play_gridcapture_game(GameState{8, 4}(), greedy_policy, random_policy)
  ╠═╡ =#

# ╔═╡ 0edf174b-98c1-4328-a5b9-d2d82a2b99b9
#=╠═╡
@bind greedy_8x4_game_step Slider(1:length(greedy_8x4_game[1]); show_value=true)
  ╠═╡ =#

# ╔═╡ aa79a188-8055-4ae0-8e3d-82632bb04b10
#=╠═╡
let
	step = greedy_8x4_game_step
	states = greedy_8x4_game[1]
	s = states[greedy_8x4_game_step]
	policy = iseven(step) ? random_policy_distribution(s) : greedy_policy_distribution(s)
	HTML(html_board(s; policy = policy))
end
  ╠═╡ =#

# ╔═╡ d508aae1-3c69-403f-b398-cd4481657215
md"""
The board above visualizes the board and the policy distribution of each player on their respective turn.  The greedy policy plays as follows considering the following in order of priority:

- If a winning move exists, take it
- If an opponent winning move exists, block it
- Favor the center of the board

It is not a sophisticated strategy, but can reliably win against a random opponent.  We can use this strategy to construct an MDP where our agent will always play against the greedy strategy.
"""

# ╔═╡ 3e645d32-26e2-4da8-a7fc-dfa1ca08d7e3
md"""
## MDP Construction with Greedy Opponent
"""

# ╔═╡ a17bdb13-472e-4f6a-ac56-3f17ffc1154b
#=╠═╡
const large_gridcapture_mdp = make_gridcapture_mdp(8, 4, greedy_policy)
  ╠═╡ =#

# ╔═╡ 2e9c422f-adc4-4762-8ef4-ae95cdc39cce
#=╠═╡
const large_gridcapture_hard_mdp = make_gridcapture_mdp(8, 4, positional_policy)
  ╠═╡ =#

# ╔═╡ ef531570-6180-46ab-981d-1a022a787373
md"""
### Random Performance
"""

# ╔═╡ 8c958b3a-c7e5-4d8c-b0e3-16d1987baafa
md"""
We can measure the benchmark performance of the random policy in this environment over the course of many episodes.
"""

# ╔═╡ a6ce047a-d577-46b0-afda-72076c1d96a3
#=╠═╡
const large_gridcapture_random_performance = TabularRL.average_stochastic_rollout(10_000, large_gridcapture_mdp, make_random_policy(large_gridcapture_mdp), 1f0)
  ╠═╡ =#

# ╔═╡ f76a0cc6-9265-4f04-aecb-da3593bb9303
#=╠═╡
const large_gridcapture_random_winrate = (large_gridcapture_random_performance + 1) / 2
  ╠═╡ =#

# ╔═╡ d05a1e26-f234-43d8-8503-88162d963bdd
md"""
Even against this basic opponent, the random policy has less than a 1% chance of winning which in this case means a reward of 1 at the end of the episode.  Now that we've fixed the opponent the episodes look like this.
"""

# ╔═╡ 90017d35-c18a-4914-8304-bfac94161733
#=╠═╡
const large_gridcapture_random_episode = runepisode(large_gridcapture_mdp)
  ╠═╡ =#

# ╔═╡ 49980b44-ba9d-4f53-b0a0-31553d57c642
md"""
### Random Episode Visualization
"""

# ╔═╡ ac04aa56-6c96-4f10-b798-62d9249a20fb
#=╠═╡
@bind large_gridcapture_random_episode_step Slider(1:length(large_gridcapture_random_episode[1])+1; show_value=true)
  ╠═╡ =#

# ╔═╡ 1b830b45-6800-49ce-ad25-ff20c0fef9b1
#=╠═╡
let
	step = large_gridcapture_random_episode_step
	states = large_gridcapture_random_episode[1]
	s = step > length(states) ? large_gridcapture_random_episode[4] : states[step]
	move = step > length(states) ? nothing : large_gridcapture_random_episode[2][step]
	HTML(html_board(s; candidate_move = move))
end
  ╠═╡ =#

# ╔═╡ 3693c086-4e76-4468-ba48-14725587d4d3
md"""
Now every move is followed by an environment transition which involves placing the agent move and the opponent selecting a response move from the strategy used to construct the MDP, in this case the greedy strategy.  Every transition has a 0 reward associated with it except for the final one in which the agent loses and receives a reward of -1.
"""

# ╔═╡ a1d02958-c9cd-464e-8668-196cab28e108
md"""
# Tranditional Policy Gradient Training Results

Given an environment, we are free to use any technique from reinforcement learning to learn optimal behavior to maximize reward.  As an undiscounted problem, maximizing reward in this case is equivalent to maximizing win rate.  Below is an example of using an actor-critic policy gradient algorithm to train an agent in the environment.
"""

# ╔═╡ 6d0e00ab-33a9-4db5-987c-e5cbdd58f521
md"""
## Training Function Setup

In order to train policy gradient, we need to provide a feature vector constructor that converts states to a vector that can be used an input by a variety of function approximators.  Since each board position is either empty or occupied by and "X" or "O" move, we can simply allocate one element in a vector to each board position and assign a numerical value to each of the 3 states: -1 for "O", 0 for "empty", 1 for "X".
"""

# ╔═╡ 63d77923-4fa9-4263-ba1b-6c5d5d219a2c
#=╠═╡
const large_gridcapture_policy_setup = setup_episodic_policy_nonlinear_training(large_gridcapture_mdp, state_to_features(large_gridcapture_mdp.initialize_state()), update_features!; min_reward = -1f0)
  ╠═╡ =#

# ╔═╡ 4d0682ae-c438-475c-9825-6a0c067b8b50
#=╠═╡
const large_gridcapture_hard_policy_setup = setup_episodic_policy_nonlinear_training(large_gridcapture_hard_mdp, state_to_features(large_gridcapture_hard_mdp.initialize_state()), update_features!; min_reward = -1f0)
  ╠═╡ =#

# ╔═╡ 37934b7f-5e5f-4f6b-8ed9-f4218a0f4e1e
md"""
#### Feature Vector Visualization

The vector has been reshaped into an 8x8 matrix, but each column will be stacked to form a lengh 64 feature vector.
"""

# ╔═╡ d5d54013-1c75-41a6-8f7e-b1721086c7df
#=╠═╡
let
	s = large_gridcapture_random_episode[1][5]
	HTML(html_board(s; cell_size = 20)) => reshape(state_to_features(s), 8, 8)
end
  ╠═╡ =#

# ╔═╡ b919b834-d65f-43b9-8ad8-0b494b4c4a0b
md"""
## Training a Neural Network Agent 

The architectur is a fully connected network with 2 layers of 64 neurons each.  Training occurs through rounds of 100,000 steps each and proceeds while performance improves from cycle to cycle.  Because this is an actor critic method, there is a policy function which provides a probability distribution over actions for each state as well as a value function that estimates a state value.  The value function is only used during training to reduce the variance of gradient updates and enable training updates to occur prior to episode termination.  The trained agent will only use the policy function.
"""

# ╔═╡ 1148e63c-a555-493b-aefb-ded46e748b49
#=╠═╡
const fcann_large_gridcapture_policy_solution = large_gridcapture_policy_setup.train_rate_decay([64, 64], 1, 1f0, 0.03f0, 0.01f0, 0.99f0, 0.99f0, 1_000_000; new_params = false, use_gpu = false)
  ╠═╡ =#

# ╔═╡ e3bb487d-c59a-4764-b926-0ac7672c5c5e
#=╠═╡
const fcann_large_gridcapture_hard_policy_solution = large_gridcapture_hard_policy_setup.train_rate_decay([64, 64], 1, 1f0, 0.03f0, 0.01f0, 0.99f0, 0.99f0, 1_000_000; new_params = false, use_gpu = false)
  ╠═╡ =#

# ╔═╡ 7eca6f17-2124-4e20-833f-3e4d3c81ec4e
md"""
## Actor-Critic Performance
"""

# ╔═╡ 6e78bd45-2659-4c2c-a1ce-f62b9c1eef69
#=╠═╡
const large_gridcapture_ac_performance = TabularRL.average_stochastic_rollout(10_000, large_gridcapture_mdp, fcann_large_gridcapture_policy_solution.policy_sample_action, 1f0)
  ╠═╡ =#

# ╔═╡ b1a8bfc9-73a7-4ce1-b3c9-c5b4263d284f
#=╠═╡
const large_gridcapture_hard_ac_performance = TabularRL.average_stochastic_rollout(10_000, large_gridcapture_hard_mdp, fcann_large_gridcapture_hard_policy_solution.policy_sample_action, 1f0)
  ╠═╡ =#

# ╔═╡ e098c17d-97c4-4f89-bce4-d9cd90ee09b8
#=╠═╡
const large_gridcapture_ac_winrate = (large_gridcapture_ac_performance + 1) / 2
  ╠═╡ =#

# ╔═╡ eaebd5ca-0669-4b96-b9ed-f28247113d30
md"""
With very little training on a small network, we have already achieved a substantial improvement to the random policy.  Below we can see what kind of strategy the agent has learned and how it behaves in winning and losing scenarios.
"""

# ╔═╡ cefe7851-b3ad-4e13-a8c7-74848ae42e66
md"""
## Actor-Critic Agent Episode Visualization

Now that we've trained an agent with some success in the environment, we can run an episode an observe the behavior as well as the policy and value function outputs.
"""

# ╔═╡ 220d4cef-141d-4a29-9f83-c48a25886131
#=╠═╡
const ac_large_gridcapture_win_episode = begin
	Random.seed!(2)
	runepisode(large_gridcapture_mdp; π = fcann_large_gridcapture_policy_solution.policy_sample_action)
end
  ╠═╡ =#

# ╔═╡ bab309d1-ade0-43f4-af2e-ecd9eff0dda7
#=╠═╡
@bind ac_large_gridcapture_win_episode_step Slider(1:ac_large_gridcapture_win_episode[5]+1; show_value=true)
  ╠═╡ =#

# ╔═╡ 51b60ecc-4609-4dd5-8537-a0b38443e00d
#=╠═╡
let
	step = ac_large_gridcapture_win_episode_step
	term_state = step > ac_large_gridcapture_win_episode[5]
	s = term_state ? ac_large_gridcapture_win_episode[4] : ac_large_gridcapture_win_episode[1][step]
	policy = term_state ? nothing : fcann_large_gridcapture_policy_solution.policy_function(s)
	action = term_state ? nothing : ac_large_gridcapture_win_episode[2][step]
	v̂ = fcann_large_gridcapture_policy_solution.value_function(s)
	@htl("""
    <div style = "font-size: 2em;">Value Estimate: $v̂</div>
		 
	$(HTML(html_board(s; policy = policy, candidate_move = action)))
		 """)
end
  ╠═╡ =#

# ╔═╡ 9dbed060-b0b4-4fe1-99c8-c9356637ccda
md"""
When successful, the trained policy tries to form a line in the center of the board.  If it successfully connects 3 in a row with a double threat, then the greedy policy cannot stop it.
"""

# ╔═╡ 7f2b59df-0022-4f7c-a769-75568e8aa607
#=╠═╡
const ac_large_gridcapture_lose_episode = begin
	Random.seed!(10)
	runepisode(large_gridcapture_mdp; π = fcann_large_gridcapture_policy_solution.policy_sample_action)
end
  ╠═╡ =#

# ╔═╡ 3bc2a8e3-4a26-4588-9c2b-d2f855cb211c
#=╠═╡
@bind ac_large_gridcapture_lose_episode_step Slider(1:ac_large_gridcapture_lose_episode[5]+1; show_value=true)
  ╠═╡ =#

# ╔═╡ 64ac08c1-7a4b-4645-97f4-54bb2e4c3ba1
#=╠═╡
let
	step = ac_large_gridcapture_lose_episode_step
	term_state = step > ac_large_gridcapture_lose_episode[5]
	s = term_state ? ac_large_gridcapture_lose_episode[4] : ac_large_gridcapture_lose_episode[1][step]
	policy = term_state ? nothing : fcann_large_gridcapture_policy_solution.policy_function(s)
	action = term_state ? nothing : ac_large_gridcapture_lose_episode[2][step]
	v̂ = fcann_large_gridcapture_policy_solution.value_function(s)
	@htl("""
    <div style = "font-size: 2em;">Value Estimate: $v̂</div>
		 
	$(HTML(html_board(s; policy = policy, candidate_move = action)))
		 """)
end
  ╠═╡ =#

# ╔═╡ d580e150-7f6d-4b63-bd24-ac39d9fe7709
md"""
In a losing scenario, the initial attempt to win in the center fails, and the policy function enters a space with much more random behavior.  In general, most of the games end successfully in 4 or 5 moves from the "X" player.  When games are extended, most end in a loss.
"""

# ╔═╡ a288fdce-5f46-42f7-a826-fbdc37db8094
md"""
# MCTS Scenario 1: Using Realtime Search for Policy Improvement

Consider a scenario in which the above policy and value function is available.  Evaluating these functions is very fast and we are given a time budget to make an action selection in the environment.  Therefore, we are free to evaluate the functions more than once at different states.  We can simulate trajectories in the environment since we know the rules.  MCTS provides a way to evaluate trajectories in an efficient manner and use the information gathered from these simluations to make a better decision.  In order to make use of our existing policy and value functions, we need to use a version of MCTS that can incorporate this prior information and improve upon it.  That is exactly what the algorithm described in [Danihelka, I., Guez, A., Schrittwieser, J., & Silver, D. (2022). Policy Improvement by Planning with Gumbel (ICLR 2022)](https://openreview.net/forum?id=bERaNdoegnO) is designed to do.  For the purposes of this notebook, I will refer to the general algorithm as Gumbel Top-K MCTS.  This algorithm also functions in a scenario where the number of simulations is smaller than the action space of the problem but also can improve when given a larger simulation budget.  In summary, the algorithm achieves the following:

- Improve upon an existing policy function by making use of a simulation budget
- Succeed even when the number of simulations is smaller than the number of environment actions (traditional MCTS cannot function in this regime)
"""

# ╔═╡ fa92cac5-71dd-4b04-99ed-2256cf50480d
md"""
## Algorithm Requirements

- Top-k: The number of actions to consider at the root node (environment state).  Given a very large action space, this value can reduce the breadth of the search at every layer of the tree and is also necessary to handle the scenario mentioned above when Nsims < ``\vert \mathcal{A} \vert``

- Nsims: The number of total simulations allowed for search.  Each simulation will traverse the tree until an unvisited state is reached.  That state will then be evaluated by the policy function to produce a value estimate and a distribution.  Each simulation will therefore entail one new evaluation of the policy function as well as the other steps needed to traverse the tree to a node.

- Cvisit and Cscale: Constants used to calculate improved policy.  Controls the relative importance of the tree action-values compared to the policy prior, however, all values guarantee expected policy improvement.  Usually remain fixed at 50 and 1 respectively.

- Policy function: `s -> (v̂, logits)` where v̂ is a state value estimate and logits represent the policy distribution such that exponentiating them (and normalizing if necessary) will result in π
"""

# ╔═╡ fc0f7cb6-792c-4dcc-bc3b-482593f9838f
md"""
Using our existing policy gradient solution, we can generate a function that provides what Gumbel Top-K MCTS needs.  We already have the value estimate and taking the log of the policy distribution produces the logits.  For efficiency, this function will populate an existing vector with the logits rather than allocating a new one every time.
"""

# ╔═╡ 2b9b64a2-69f4-4083-b7a6-13325bd1e672
#=╠═╡
function gumbel_π_dist_test!(v::Vector{T}, s) where T<:Real
	output = fcann_large_gridcapture_policy_solution.policy_and_value(s; policy = v)
	v .= log.(v) #convert distribution into logits
	return (output.value)
end
  ╠═╡ =#

# ╔═╡ 481e6e9c-63f0-4d81-9abb-305e692f5121
md"""
##  Search -> Policy Improvement

Given an environment state `s`, the search algorithm conducts `Nsims` simulations to gather information using the policy function.  The goal of the search is to select an action that on average will be an improvement over the action selected by the policy alone.  One of the hyperparameters of the algorithm is `K` which limits the number of actions considered at the root.  Part 1 of the search is to consider which `K` actions should be considered at all.

### Selecting Search Actions at the Root

For the examples below, the `K` value is 16 which corresponds to the number of actions considered at the root.

- Up to 16 actions are sampled from the policy distribution evaluated at the root state.
- For Nsims 16 or less, the value function is evaluated once at a transition state produced by each of the sampled actions.
"""

# ╔═╡ 1e874fe5-3c7e-4807-ae50-cb316e3bcd50
md"""
#### Limited Simulations less than ``K = 16``
"""

# ╔═╡ 0beed6bc-1d6a-4499-9315-842e74ad4ff8
#=╠═╡
@bind gumbel_mcts_search_test1_sims Slider(1:16; show_value=true, default = 6)
  ╠═╡ =#

# ╔═╡ 6ed31c1e-ba49-43c5-b67e-3b03a6f04301
#=╠═╡
const gumbel_mcts_search_test1 = gumbel_mcts_search(large_gridcapture_mdp, 1f0, gumbel_π_dist_test!, 1f0, large_gridcapture_mdp.initialize_state(), Returns(0f0);nsims = gumbel_mcts_search_test1_sims, depth = 1_000, sim_message = false)
  ╠═╡ =#

# ╔═╡ 69087af9-d37d-4736-ba9a-344d8cbdec12
md"""
The search function returns the following in order:

1) selected action 
2) visit_counts: A dictionary mapping visited states to a list of the number of times each action has been visited
3) Q: A dictionary mapping visited states to a list of averaged action values 
4) vest_cache: A dictionary mapping visited states to the value function output at that state
5) policy_cache: A dictionary mapping visited states to the policy function output at that state
"""

# ╔═╡ d61c6352-9156-47db-824f-9d5f5e8cc827
md"""
Below are the states that exist in the tree.  With 16 or fewer simulations, the only states in the tree correspond to a single move taken by the "X" player.
"""

# ╔═╡ b3818532-950b-49f7-8f6e-91b785e856b4
#=╠═╡
const gumbel_mcts_search_test1_branch_states = keys(gumbel_mcts_search_test1[2]) |> collect
  ╠═╡ =#

# ╔═╡ 58673ed3-7098-4745-a800-6b813cc40dd7
#=╠═╡
[HTML(html_board(s; cell_size = 10)) for s in gumbel_mcts_search_test1_branch_states]
  ╠═╡ =#

# ╔═╡ 23fadce5-7392-4916-8609-ad7aacad2306
md"""
If we look at the information stored for this deeper states in the tree we see there is no information because the simulation count isn't high enough to have visited any of them again.
"""

# ╔═╡ 91ed1cb2-ae03-4c56-a361-aa9434ab6352
#=╠═╡
gumbel_mcts_search_test1[2][gumbel_mcts_search_test1_branch_states[1]]
  ╠═╡ =#

# ╔═╡ 1a32d24d-7781-4774-b530-0379dd02f5d5
md"""
In contrast the root of the tree has visited as many actions as there are simulations and saved a value for each
"""

# ╔═╡ b46b076b-5177-4595-99de-931aa81ae9a3
#=╠═╡
let
	s = large_gridcapture_mdp.initialize_state()

	π_policy = fcann_large_gridcapture_policy_solution.policy_function(s)

	best_action, visit_counts, Q, v_est_cache, policy_cache = gumbel_mcts_search_test1

	
	HTML(html_board(s; policy = π_policy, visit_count = visit_counts[s], candidate_move = best_action)) =>
	(tree_action_values = reshape(Q[s], 8, 8),)
end
  ╠═╡ =#

# ╔═╡ 1ade37fa-e033-4134-b2f2-bc0ab535b90c
md"""
#### Simulation Count ``>> K = 16``

In this regime, the algorithm still samples the initial 16 actions from the policy distribution, but it distributes the simulations in stages using a procedure called *sequential halving*.  The goal of the procedure is to cut down the considered actions until just one remains and for that action to be the best possible selection. 
"""

# ╔═╡ 64cada0a-b9d1-4aa9-bb1c-0e697d11ffd0
#=╠═╡
const gumbel_mcts_search_test2 = gumbel_mcts_search(large_gridcapture_hard_mdp, 1f0, gumbel_π_dist_test!, 1f0, large_gridcapture_hard_mdp.initialize_state(), Returns(0f0); nsims = 200, depth = 1_000, sim_message = true)
  ╠═╡ =#

# ╔═╡ d2fbb761-6bf6-46db-b182-062ec9c10d8f
#=╠═╡
const gumbel_mcts_search_test2_branch_states = keys(gumbel_mcts_search_test2[2]) |> collect |> v -> filter(x -> !isequal(x, GameState{8, 4}()), v)
  ╠═╡ =#

# ╔═╡ eb378857-c047-456f-bb8c-dddbb6b64882
#=╠═╡
[HTML(html_board(s; cell_size = 10)) for s in gumbel_mcts_search_test2_branch_states]
  ╠═╡ =#

# ╔═╡ c8586d76-7923-4e13-918f-36e6e4e97ace
md"""
Now there are 200 states since each simulation explored further into the tree with some states having more than 1 made move.  Also if we select one of these states it should have information in the tree.
"""

# ╔═╡ 0260c420-dd3d-48fa-8ea9-287387a51d16
#=╠═╡
gumbel_mcts_search_test2[2][gumbel_mcts_search_test2_branch_states[2]].nzind #this branch state visited action 39 1 time
  ╠═╡ =#

# ╔═╡ efeba91e-522a-4c17-a50f-d1f610981139
#=╠═╡
let
	s = large_gridcapture_mdp.initialize_state()

	π_policy = fcann_large_gridcapture_policy_solution.policy_function(s)

	best_action, visit_counts, Q, v_est_cache, policy_cache = gumbel_mcts_search_test2

	
	HTML(html_board(s; policy = π_policy, visit_count = visit_counts[s], candidate_move = best_action)) =>
	(tree_action_values = reshape(Q[s], 8, 8),)
end
  ╠═╡ =#

# ╔═╡ 1e6595ef-aa9f-4084-8976-64605aeee54e
md"""
We can now see that there are two actions with 49 visits each and the final action is selected between the two of them.  The other 14 actions are visited progressively less with the least visited 8 actions seen only 3 times.  This scenario corresponds to the diagram in the paper with 200 simulations.
"""

# ╔═╡ 80e82c6b-4b6f-46ca-9162-a7c12aa9fc03
md"""
### Policy Improvement/Selecting Actions to Explore

For the first round of action selection, only the policy distribution is sampled.  The simulations are allocated evenly to all 16 selected actions.  On round 2 of sequential halving, the top 8 actions are selected from these 16 using a new distribution based on the policy and the collected action values.  This method of calculating the improved policy is also used to select actions at non-root nodes and to calculate the final improved policy after search is complete.

Prior Policy Logits: ``\text{logits}(a)``

Improved Policy Logits: ``\text{logits}(a) + \sigma(\hat q(a))``

``\sigma(\hat q(a)) = (c_{visit} + \max_b N(b)) c_{scale} \hat q (a)`` is a monotonic function of the action values

``N(b)`` is the count of visits to each available action from the state under consideration.  As a state is visited more often, this value increases which in turn increases the weight of the action-values on the improved policy compared to the prior.

At the root node after round 1, all actions under consideration have a ``\hat q(a)`` estimate.  Due to the sampling method using gumbel noise, this method always samples from the improved policy given the information available at the time.  Once the search is complete, it is possible that one of the earlier eliminated actions appears more favorable than the action selected in the end.  

For non-root nodes, the algorithm considers all available actions but samples from the improved distribution.  For states missing a ``\hat q(a)`` estimate, the state value estimate is used to fill in the gaps.
"""

# ╔═╡ 20dcb4c0-75e8-44ac-b1be-a27d5d4a7660
md"""
#### Generating Episodes with Search

By using the above search function, we can generate a new policy that performs this search on every step and selects the final action recommendation.  Once we construct this new policy we can benchmark it to verify its performance is improved over the policy function and we can visualize what the improved policy looks like.
"""

# ╔═╡ 9d623aba-7eaa-44a0-9617-f869cf423d5e
md"""
Note that now when episodes become longer, the search can eventually improve upon the policy that is mostly random.  Compared this behavior to the base policy that would lose almost all games that extended beyond several moves.
"""

# ╔═╡ 7e91919a-6d1f-4d02-a095-a30186c8362a
md"""
#### Verifying Policy Improvement of Search

Increasing the number of simulations increases the performance as expected.  Even a very low simulation count, however, produces an improvement.
"""

# ╔═╡ e4d8bf10-9214-4a9a-be18-f88cd9153f8b
#=╠═╡
large_gridcapture_hard_ac_performance
  ╠═╡ =#

# ╔═╡ 6297a0ff-c277-414b-a2b8-020c11da17e6
#=╠═╡
large_gridcapture_ac_performance
  ╠═╡ =#

# ╔═╡ 43f6f078-30f2-4951-b11c-95adb1dabafc
#=╠═╡
const large_gridcapture_mcts2_performance = TabularRL.average_stochastic_rollout(10_000, large_gridcapture_mdp, s -> gumbel_mcts_search(large_gridcapture_mdp, 1f0, gumbel_π_dist_test!, 1f0, s, Returns(0f0); nsims = 2, depth = 1000, use_vmix = false, min_reward = -1f0, max_reward = 1f0, rescale_values = false)[1], 1f0)
  ╠═╡ =#

# ╔═╡ fd290ce1-8cca-4a68-9a53-a0ecc9b95021
#=╠═╡
const large_gridcapture_mcts20_performance = TabularRL.average_stochastic_rollout(1_000, large_gridcapture_mdp, s -> gumbel_mcts_search(large_gridcapture_mdp, 1f0, gumbel_π_dist_test!, 1f0, s, Returns(0f0); nsims = 20, depth = 1000, use_vmix = true, min_reward = -1f0, max_reward = 1f0)[1], 1f0)
  ╠═╡ =#

# ╔═╡ 9148e4fb-e1ad-4d49-9211-ee874965f879
#=╠═╡
const large_gridcapture_mcts200_performance = TabularRL.average_stochastic_rollout(1_000, large_gridcapture_mdp, s -> gumbel_mcts_search(large_gridcapture_mdp, 1f0, gumbel_π_dist_test!, 1f0, s, Returns(0f0); nsims = 200, depth = 1000, use_vmix = true, min_reward = -1f0, max_reward = 1f0)[1], 1f0)
  ╠═╡ =#

# ╔═╡ f22a726c-d334-460f-a961-05cb486cd09e
#=╠═╡
const large_gridcapture_mcts1000_performance = TabularRL.average_stochastic_rollout(1_000, large_gridcapture_mdp, s -> gumbel_mcts_search(large_gridcapture_mdp, 1f0, gumbel_π_dist_test!, 1f0, s, Returns(0f0); nsims = 1000, depth = 1000, use_vmix = true, min_reward = -1f0, max_reward = 1f0)[1], 1f0)
  ╠═╡ =#

# ╔═╡ 4f1ee074-4f64-4088-b329-b2920489fb82
md"""
#### Benchmarking Policy Performance
"""

# ╔═╡ 850637e9-c707-4dd5-b4e8-fc5c67bebe26
md"""
Benchmarking the performance of both policies we can see the search version with 1000 simulations takes about 24k times longer to run.  Looking at the performance profile, most of this time is spent on the MDP step function so the policy evaluation isn't even a bottleneck.  With only 2 simulations, the slowdown is only about 20x.
"""

# ╔═╡ 90b11aa5-0136-46ad-9fdd-3d043e028f8c
#=╠═╡
@btime gumbel_mcts_search(large_gridcapture_mdp, 1f0, gumbel_π_dist_test!, 1f0, $(large_gridcapture_mdp.initialize_state()), Returns(0f0); nsims = 2, depth = 1000, use_vmix = true, min_reward = -1f0, max_reward = 1f0)
  ╠═╡ =#

# ╔═╡ 0c2ea530-bbe6-4b14-a202-1a9fe91405bd
# 102.5 μs

# ╔═╡ 935afffb-a305-462f-b92c-8a7048ee125f
#=╠═╡
@btime gumbel_mcts_search(large_gridcapture_mdp, 1f0, gumbel_π_dist_test!, 1f0, $(large_gridcapture_mdp.initialize_state()), Returns(0f0); nsims = 200, depth = 1000, use_vmix = true, min_reward = -1f0, max_reward = 1f0)
  ╠═╡ =#

# ╔═╡ 640b689b-af31-4309-baa0-4004d40a669d
# 12 ms

# ╔═╡ f802a3bd-42b8-4012-b06b-b7319cbb189d
# 128 ms

# ╔═╡ c7cd253f-8852-4f6e-9ea9-2baedc53b4d5
#=╠═╡
@btime fcann_large_gridcapture_policy_solution.policy_sample_action(large_gridcapture_mdp.initialize_state())
  ╠═╡ =#

# ╔═╡ 2e1b189b-f5d7-429b-8468-04f57dcfb06a
#5 μs

# ╔═╡ 02e1ea17-74ed-4523-bde7-fe48d9ea1f1d
md"""
# MCTS Scenario 2: Using Search as a Training Tool
"""

# ╔═╡ a07c2d69-227a-42ca-87c6-35d19f955e77
md"""
### Search -> Data Collection

The function used to generate an MCTS episode has another purpose, that of data collection.  Once an episode is complete, we have an outcome win/loss/draw.  That outcome corresponds to a sample of a value estimate for every state visited in the episode.  Also, we can use the tree root at from every search to compute the improved policy using the formula described above.  That policy can be used directly to train a policy network with KL divergence loss.

- Generate an episode with the MCTS search policy
- For each visited state save the following
  - Observed reward (for an game like this it will be -1/0/1 for every state depending on the outcome)
  - Tree improved policy

Later on we can use this collected data to update the value network prediction to match the outcome and the policy network to match the improved tree policy.  The function below takes the information from the MCTS episode as well as the feature vector format for the policy/value function and produces information that can be used for training.
"""

# ╔═╡ cf57bed0-ed83-48ed-9ecc-6d5740f7fbd7
md"""
#### Collecting Data From a Single Episode
"""

# ╔═╡ aabaa31b-7204-40f7-a3a8-2fd0f0e666ec
function generate_gumbel_mcts_training_data(mcts_episode::NamedTuple, mdp::StateMDP, feature_vector::V, update_feature_vector!::Function) where V
	states = mcts_episode.states
	value_targets = mcts_episode.Gs
	policy_targets = [a.improved_policy for a in mcts_episode.mcts_data]

	n = length(value_targets)
	feature_vectors = [copy(feature_vector) for i in 1:n]

	for i in 1:n
		v = feature_vectors[i]
		update_feature_vector!(v, states[i])
	end

	return (states = states, feature_vectors = feature_vectors, policy_targets = policy_targets, value_targets = value_targets)
end

# ╔═╡ 334e5d84-6b1a-4427-8ecc-cecd858625b2
md"""
#### Parallel Data Collection

To use the mcts search as a data collection method for training, we need to automate the collection of data at scale.  Later on we will extract mini batches from this data for training.
"""

# ╔═╡ 26ea9090-af3a-4bf8-946d-14896a54a020
md"""
The above setup creates utilities for collecting data and using it to train a policy/value function as described above.  For demonstration purposes, the tree search will use the function from the policy gradient solution rather than the randomly initialized neural network.  This way we can verify that training works by producing a function after training that matches or exceeds the performance of the starting function.
"""

# ╔═╡ 9a48ba14-f71f-4c60-907d-266df8a3ffdb
md"""
The setup contains circular buffers for each item produced by `generate_data`.  These buffers have a size equal to the `buffer_size` argument shown above.  Once the buffer has reached capacity, the oldest data in the buffer will be removed.  Intially it is empty.
"""

# ╔═╡ 75e5f820-1a7d-42c5-bd23-cf9b94faf727
md"""
We can accumulate new data in the buffer in a few ways with the `accumulate_data!` function.  The version called below generates data for 10 episodes and adds it to the buffer.  If we run the function repeatedly, we can see data being accumulated in the buffer.
"""

# ╔═╡ 30b603f2-5181-48f3-a1fc-59297b0cb7fc
md"""
We can also accumulate data across multiple threads by generating episodes in parallel and having a parent process accumulate the episodes as they are ready and add them to the buffers.  With 10 episodes across 10 threads, we can significantly speed up generation.
"""

# ╔═╡ 3363b2bd-7a38-42db-8225-da04fa1d8cb1
md"""
### Data Collection -> Policy/Value Training
- The data accumulation process adds examples to the replay buffer
- The training function extracts a minibatch from the replay buffer, formats it into input/output data, and performs gradient descent updates.
- We can measure the relative rates of both processes occuring unimpeded
"""

# ╔═╡ 91f5ca97-a747-483f-8697-1b1872fba236
md"""
Let's fill up the replay buffer and see how long it takes to accumulate 100,000 examples on 40 threads.
"""

# ╔═╡ 53837f4c-c246-41a0-a917-bae50376dc0f
md"""
#### Training on Existing Replay Buffer

Now that we've accumulated data in the buffer, we should be able to use it to train a policy function that exceeds the performance of the function used in the search process.  Below is a training loop using the utilities from the setup that demonstrates just that.
"""

# ╔═╡ b6f43053-1309-4139-b296-a46e8e864f34
md"""
After training, we can use `update_generation!()` to copy the parameters to the policy function output and calculate the performance of this generation on the MDP.  .  We can see the original generation had very poor performance being a random function and after training it matches or exceeds the performance of the search policy function.
"""

# ╔═╡ 93f009f0-a348-47de-b9e0-dd2e29f50973
md"""
We can see on the left side how the policy distilled from the function used to generate data with MCTS is different from the original.  That is because the dataset should produce an improvement based on the number of simulations conducted.  We can also see a boost in performance highlighted below.  While the policy on its own is better, it is also better than the original when enhanced with mcts search.
"""

# ╔═╡ 49afc4a2-3168-4926-99b1-e363295f3bd5
#=╠═╡
large_gridcapture_ac_performance
  ╠═╡ =#

# ╔═╡ aede5c00-628b-4b06-b086-9875ecd2d9ea
md"""
#### Data Generation Rate

We can measure how quickly we can generate training data with MCTS and how it depends on the number of simulations and the number of threads.
"""

# ╔═╡ ce9a53ea-9642-47df-b84d-4e5ce5bf123a
function measure_data_generation_rate(training_setup::NamedTuple, t = 1; num_data_threads::Integer = 1, minimum_examples = 1000, search_kwargs...)
	#begin data accumulation
	data_process = training_setup.accumulate_data!(num_data_threads; search_kwargs...)

	sleep(t)

	while data_process.step_count[] < minimum_examples
		sleep(0.01)
	end

	training_setup.stop_data_collection!()

	data_process.check_data_rate()
end

# ╔═╡ 3d553190-c829-4b65-bc83-c4cc58df4330
md"""
#### Training Rate

We can also measure the rate at which we can process mini batches of training examples as a function of the batch size.  Once we've fixed the network architecture, the training rate primarily depends on the batch size.
"""

# ╔═╡ 999d8928-12b8-4038-b6b6-e7087e44a356
function measure_data_rate(training_setup::NamedTuple; l::Integer = 1_000, batch_time::Real = 1, num_data_threads::Integer = 1, search_kwargs...)
	l > training_setup.buffer_size && error("Cannot test adding $l elements on a buffer of size $(training_setup.buffer_size)")

	training_setup.clear_data!()
	
	t = time()
	
	#accumulate data
	if num_data_threads > 1
		training_setup.accumulate_data!(typemax(Int64), l, num_data_threads; search_kwargs...)
	else
		training_setup.accumulate_data!(typemax(Int64), l; search_kwargs...)
	end

	runtime = time() - t

	l_actual = length(first(training_setup.saved_data))

	data_rate = l_actual / runtime

	@info "Generated $l_actual examples in $runtime seconds for a rate of $(round(data_rate; sigdigits = 4)) items per second"

	num_batches = 0
	t = time()
	while time() - t < batch_time
		training_setup.train_minibatch!(0.1f0, 0.1f0)
		num_batches += 1
	end

	runtime = time() - t
	batch_rate = num_batches / runtime
	item_rate = batch_rate * training_setup.batch_size

	@info "Processed $num_batches batches of size $(training_setup.batch_size) in $runtime seconds for a rate of $(round(item_rate; sigdigits = 3)) items per second"

	return (generation_rate = data_rate, process_rate = item_rate)
end


# ╔═╡ 6c9ee058-8a16-4332-b40a-112b30e30a59
#add graph for training rate as a function of network architecture and batch size, then also make this data generation plot while training is occuring

# ╔═╡ 9e375988-454e-463e-9d56-439c9855ffcc
md"""
#### Simultaneous Collection and Training

- Compare timing of filling buffer with training through it before.  Already have demonstration of policy distillation/improvement
- Show demo of accumulating buffer while training with interference vs non-interference.  Need to see if there's away to avoid the training loop blocking the buffer accumulation
- Discuss the rate of each and how much of the buffer would be trained on, the waiting cycle, how fast each process is occuring in real time
"""

# ╔═╡ e8b27cc5-25b6-4323-b0da-3a0d9990a717
#should I compare this over different batch sizes?  So that way we can see the crossover points or ratios with different settings.  Need to think about what kind of plot I want

# ╔═╡ 29033440-f242-44d7-9190-5506e0653d0a
md"""
The following two tests confirm that the simultaneous training loop is not slowing down the data collection rate.
"""

# ╔═╡ 9764c645-c07c-4c43-8231-56791426e241
md"""
### Policy/Value Training -> Mastering an Environment

We've seen that given a policy/value function we can accumulate MCTS derived data to train an updated policy/value function that matches or exceeds the original performance.  Now we consider using this training cycle to produce a sequence of improvements that eventually achieve optimal performance in an environment.

#### Managing Generations of Policy/Value Parameters

For our training pipeline, we will need to maintain multiple sets of parameters that correspond to different generations of functions.  Hopefully these generations will have progressively improving performance in the environment.  The rate at which we update generations relative to the rate of data accumulation will also determine the composition of the data buffer.

- Randomly initialize parameters for a policy/value function and save 2 copies
  - Search parameters: generates training data using MCTS that is accumulated into the buffer
  - Train parameters: updated with minibatch gradient descent training to minimize loss with buffer data
- Save a copy of the search parameters as the first generation and calculate its performance in the environment
- After a set interval, update the generation
  - Use current state of training parameters to evaluate performance in the environment
  - Copy training parameters into search parameters so new data accumulation is based on new generation
  - Keep track of which generation is the best so far (hopefully the most recent)
- Repeat process to accumulate parameter generations and check performance progress
"""

# ╔═╡ 5d0c5922-1d4c-4ab0-bb74-91cf39b9bf2b
md"""
#### Designing a Training Loop

The process above describes the initial setup and the process to train a new generation.  To complete a training loop, we can either train for a fixed number of cycles or use a criteria that checks how many consecutive generations have failed to produce an improvement:

- Select hyperparameters that determine ratio of training rate to data generation rate
  - Resource allocation to MCTS data collection
  - Minibatch size
  - Number of simulations for MCTS
- Select generation update frequency such as N minibatches where ``N = \frac{\text{buffer size}}{\text{batch size}}``
- Initialize failure count to 0
- Loop until failure count > threshold
  - Train parameters for N minibatches
    - Initialize parallel data accumulation using current generation search parameters
    - Repeat N times
      - Pull N examples from buffer to form minibatch
      - Perform gradient update for train parameters
  - Check environment performance of current train parameters
  - Copy train parameters into saved generations with performance and update search parameters
  - If performance improved over previous best generation 
    - Set failure count to 0
  - Otherwise
    - Increment failure count by 1
    - Lower learning rate
- After training is complete we can test the performance of any generation as a policy network and as an MCTS enhanced policy that uses the trained policy/value network together.  Performance should improve with a higher simulation budget
"""

# ╔═╡ 41978938-c540-4778-9177-12d4ef94bf9f
md"""
#### Gridcapture Example
"""

# ╔═╡ d573ebc9-2dc0-4d66-980c-580d3fcd5751
md"""
##### Testing Data Ratios

Below is a training test for the above setup that adjusts the remaining hyperparameters to see the true simultaneous generation/training comparison rate.  I've adjusted the settings to produce a ratio of about 10 which corresponds to each training example being used 10 times before being pushed out of the buffer.  
"""

# ╔═╡ 277336f2-3ce8-4c63-b2fe-fd4b36e8e353
md"""
We can also setup a test that allows us to tweak more variables such as the batch size and network architecture, both of which require a new pipeline setup since these two parameters imply certain memory structures that are used throughout training.
"""

# ╔═╡ b26f33c0-2a90-4405-b76c-8d53e257c62f
md"""
Should replicate the test above
"""

# ╔═╡ 9db64ad9-e2e1-4520-b29f-7acfb3f3ef4a
md"""
If I increase the batch size to 64, the training rate is higher, so to get a reasonable ratio we must use more search threads or lower the number of simulations.  If I lower the simulation count from 60 to 30, the generation rate doubles keeping the ratio reasonable.
"""

# ╔═╡ 6f3b990c-0f48-4ca2-8ae8-22171b5b4943
md"""
##### Running the Training Pipeline

Below I've prototyped a training loop that uses the hyperparameters tested above that produce a training rate to data rate ratio of 10.  The buffer size is 1,000,000 examples and the generation is updated after the number of training updates has covered as many examples as are in the buffer.  We can think of this as one epoch of training although the buffer data is being updated throughout the process.  I've set a consecutive failure limit of 5 and an initial learning rate of 0.1.  Training will continue until the failure limit is reached.  For a generation to be successful, it needs to strictly improve performance, so if a generation ever achieves the maximum possible performance of 1, the loop is guaranteed to terminate.
"""

# ╔═╡ 03e274c9-2c75-440a-bec5-7d2c208e059e
md"""
The best generation policy function has near perfect performance which we can make arbitrarily close to 1 using the MCTS enhanced version
"""

# ╔═╡ e446f5b2-068e-4c27-aacc-dcd99362a70b
md"""
I already have the performance characteristics of each generation for the policy function itself.  If I want to see how the performance improved over time with the benefit of MCTS, I can use the extraction function above
"""

# ╔═╡ c4ba3f81-8189-4059-8652-7edb0834c215
md"""
# Extra Stuff
"""

# ╔═╡ 6ee14f48-b46c-46cd-86a6-140e5f969a75
# ╠═╡ skip_as_script = true
#=╠═╡
BLAS.set_num_threads(4)
  ╠═╡ =#

# ╔═╡ 59ec3eed-559f-4243-92f8-d893768511be
#=╠═╡
const large_gridcapture_value_setup = setup_episodic_value_nonlinear_training(large_gridcapture_mdp, state_to_features(large_gridcapture_mdp.initialize_state()), update_features!; min_reward = -1f0)
  ╠═╡ =#

# ╔═╡ c472e9b0-14a6-46be-8914-08ee560fac1c
#=╠═╡
const gomoku_easy_mdp = make_gridcapture_mdp(15, 5, random_policy)
  ╠═╡ =#

# ╔═╡ 475dad7f-227f-4d94-ba85-ec654fa23bbd
#=╠═╡
const gomoku_medium_mdp = make_gridcapture_mdp(15, 5, greedy_policy)
  ╠═╡ =#

# ╔═╡ 46bac81b-29ac-4594-9459-11710e5b6e02
#=╠═╡
const gomoku_hard_mdp = make_gridcapture_mdp(15, 5, defensive_policy)
  ╠═╡ =#

# ╔═╡ 6d4a7252-d23a-43c7-878b-ac6eb1c8e9a6
#=╠═╡
const gomoku_expert_mdp = make_gridcapture_mdp(15, 5, positional_policy)
  ╠═╡ =#

# ╔═╡ 67453114-9ce0-424e-82a4-ff9069bfcc6b
#=╠═╡
const gomoku_easy_value_setup = setup_episodic_value_nonlinear_training(gomoku_easy_mdp, state_to_features(gomoku_easy_mdp.initialize_state()), update_features!; min_reward = -1f0)
  ╠═╡ =#

# ╔═╡ 3ec0e848-d1ef-4f5c-b712-507e7c6f83c8
#=╠═╡
const gomoku_medium_value_setup = setup_episodic_value_nonlinear_training(gomoku_medium_mdp, state_to_features(gomoku_medium_mdp.initialize_state()), update_features!; min_reward = -1f0)
  ╠═╡ =#

# ╔═╡ 63ba2858-b49d-4867-8a9e-0c16816df168
#=╠═╡
const gomoku_hard_value_setup = setup_episodic_value_nonlinear_training(gomoku_hard_mdp, state_to_features(gomoku_hard_mdp.initialize_state()), update_features!; min_reward = -1f0)
  ╠═╡ =#

# ╔═╡ e7ffd232-e520-467d-832d-9cdc898af79c
#=╠═╡
const gomoku_easy_policy_setup = setup_episodic_policy_nonlinear_training(gomoku_easy_mdp, state_to_features(gomoku_easy_mdp.initialize_state()), update_features!; min_reward = -1f0)
  ╠═╡ =#

# ╔═╡ 38f75388-7d9f-4538-b173-38d1f303d569
md"""
#### Value Training Solutions
"""

# ╔═╡ b51d007d-8135-4741-ab81-c00f8ee9a746
# ╠═╡ disabled = true
#=╠═╡
const fcann_large_gridcapture_value_solution = large_gridcapture_value_setup.train_ϵ_decay([64, 64], 1, 1f0, 0.1f0, 0.99f0, 10_000; ϵ_min = 0.001f0)
  ╠═╡ =#

# ╔═╡ 630e7012-d5ad-43be-a671-1582e532804e
# ╠═╡ disabled = true
#=╠═╡
# const fcann_gomoku_value_solution = gomoku_value_setup.train_dqn_rate_decay([128, 128, 128, 128], 1, 1f0, 0.01f0, 1_000_000; ϵ = 0.1f0, N = 10, batch_size = 256)

const fcann_gomoku_easy_value_solution = gomoku_easy_value_setup.train_ϵ_decay([64, 64], 1, 1f0, 0.1f0, 0.99f0, 10_000; ϵ_min = 0.001f0)
  ╠═╡ =#

# ╔═╡ a7c34d1a-169f-4189-99fd-e3cb6de744b5
# ╠═╡ disabled = true
#=╠═╡
const fcann_gomoku_medium_value_solution = gomoku_medium_value_setup.train_ϵ_decay([64, 64], 1, 1f0, 0.1f0, 0.99f0, 100_000; ϵ_min = 0.001f0)
  ╠═╡ =#

# ╔═╡ a366e369-21b5-45a5-903e-59ace7c61c3b
# ╠═╡ disabled = true
#=╠═╡
const fcann_gomoku_hard_value_solution = gomoku_hard_value_setup.train_ϵ_decay([64, 64], 1, 1f0, 0.1f0, 0.99f0, 100_000; ϵ_min = 0.001f0)
  ╠═╡ =#

# ╔═╡ e8961a70-08e1-410e-a60c-05c6c963482d
md"""
#### Policy Training Solutions
"""

# ╔═╡ 55127238-6d1d-41d0-9e4a-9c0ac150a50b
#=╠═╡
const fcann_gomoku_easy_policy_solution = gomoku_easy_policy_setup.train_rate_decay([64, 64], 1, 1f0, 0.01f0, 0.01f0, 0.99f0, 0.99f0, 100_000)
  ╠═╡ =#

# ╔═╡ 4648697d-c834-4a57-9e0e-dbe4f1757e35
#=╠═╡
TabularRL.average_stochastic_rollout(10_000, gomoku_medium_mdp, fcann_gomoku_easy_policy_solution.policy_sample_action, 1f0)
  ╠═╡ =#

# ╔═╡ 3669289b-4baa-4aa5-ad2b-4242703ffa0c
#=╠═╡
TabularRL.average_stochastic_rollout(10_000, gomoku_hard_mdp, fcann_gomoku_easy_policy_solution.policy_sample_action, 1f0)
  ╠═╡ =#

# ╔═╡ a268f2e4-51c0-452a-bec3-146a9028f25b
#=╠═╡
large_gridcapture_episode = runepisode(large_gridcapture_mdp; π = fcann_large_gridcapture_policy_solution.policy_sample_action)
  ╠═╡ =#

# ╔═╡ d7ea540e-c134-4389-b448-7186a1e6eb7f
#=╠═╡
@bind large_gridcapture_episode_step Slider(1:large_gridcapture_episode[5]+1; show_value=true)
  ╠═╡ =#

# ╔═╡ d29a25db-536d-4231-b56e-9bb28470d0c8
#=╠═╡
HTML(large_gridcapture_episode_step > large_gridcapture_episode[5] ? html_board(large_gridcapture_episode[4]; cell_size = 40) : html_board(large_gridcapture_episode[1][large_gridcapture_episode_step]; cell_size = 40, policy = fcann_large_gridcapture_policy_solution.policy_function(large_gridcapture_episode[1][large_gridcapture_episode_step])))
  ╠═╡ =#

# ╔═╡ 095cb92d-9a8b-45b1-bdf2-8614411a1398
md"""
Short episodes with this policy usually result in a win with the action selection being close to deterministic.  For longer episodes, the policy function becomes much more indecisive entering a part of the state space it is not familiar with.  These episodes are often losses with mostly random actions occuring after that.  We can see below, however, that when using Gumbel MCTS, some of these longer episodes end in a win owning to the tree search finding an improved action.
"""

# ╔═╡ b37c8a17-faaa-4786-a520-059a224eb6dc
#=╠═╡
const gumbel_test_output = gumbel_mcts_search(large_gridcapture_mdp, 1f0, gumbel_π_dist_test!, 1f0, large_gridcapture_mdp.initialize_state(), Returns(0f0); nsims = 1000, depth = 1_000, sim_message = true)
  ╠═╡ =#

# ╔═╡ e507d605-ce14-4dee-987b-ad1668534ac7
#=╠═╡
gumbel_policy, gumbel_state = make_gumbel_mcts_policy(large_gridcapture_mdp, 1f0, gumbel_π_dist_test!, 1f0, Returns(0f0); nsims = 50, depth = 1000, min_reward = -1f0, max_reward = 1f0)
  ╠═╡ =#

# ╔═╡ 99dd5ff5-75b3-41ec-88cd-b907159812d4
#=╠═╡
TabularRL.average_stochastic_rollout(10_000, large_gridcapture_mdp, fcann_large_gridcapture_policy_solution.policy_sample_action, 1f0)
  ╠═╡ =#

# ╔═╡ ed5d12e5-a277-4a6d-8048-d5f18bc6f23c
#=╠═╡
TabularRL.average_stochastic_rollout(1_000, large_gridcapture_mdp, s -> gumbel_mcts_search(large_gridcapture_mdp, 1f0, gumbel_π_dist_test!, 1f0, s, Returns(0f0); nsims = 1000, depth = 1000, use_vmix = true, min_reward = -1f0, max_reward = 1f0)[1], 1f0)
  ╠═╡ =#

# ╔═╡ 300799fa-b47e-432d-830e-0c5e822a9b02
# ╠═╡ skip_as_script = true
#=╠═╡
BLAS.set_num_threads(4)
  ╠═╡ =#

# ╔═╡ c1df080c-15e5-41cd-9553-29bf20cacce7
function train_gumbel_mcts(setup::NamedTuple, search_sims::Integer, search_threads::Integer; α_θ = 0.1f0, α_w = 0.1f0, failure_limit::Integer = 5, gen_epochs::Integer = 1, eval_sims = 1, target_ratio = Inf)
	#find the best performance consistent with the number of evaluation simulations being used
	(best_performance, best_generation) = findmax(a -> a.nsims != eval_sims ? typemin(Float32) : a.performance, setup.policy_and_value_generations)
	rate_adjust = 1f0
	@info "Prior to training, best performance is $best_performance from generation $best_generation"

	failure_count = 0
	# rounds = 10
	buffer_capacity = setup.saved_data.states |> capacity
	epoch_batches = ceil(Int64, buffer_capacity / setup.batch_size)
	num_batches = gen_epochs * epoch_batches
	setup.clear_data!()
	while failure_count < failure_limit
	# for round in 1:rounds
		setup.train_networks!(α_θ*rate_adjust, α_w*rate_adjust, num_batches, search_threads, target_ratio; clear_buffer = false, nsims = search_sims)
		setup.update_generation!(;nsims = eval_sims)
		generation_performance = setup.policy_and_value_generations[end].performance
		# @info "New generation performance is $generation_performance on round $round of $rounds"
		@info "New generation performance is $generation_performance"
		if generation_performance ≤ best_performance
			rate_adjust /= 2
			@info "New performance is worse than best performance of $best_performance from generation $best_generation.  Cutting learning rate adjustment in half to $rate_adjust."
			failure_count += 1
			@warn "Failed improvement $failure_count out of $failure_limit"
		else
			failure_count = 0
			@info "New best performance achieved on generation $(length(setup.policy_and_value_generations))."
			best_performance = generation_performance
			best_generation = length(setup.policy_and_value_generations)
		end
	end
	setup.restore_generation!(best_generation)
	setup.policy_and_value_generations
end

# ╔═╡ 7249110d-8b63-46a4-8772-642ddade0eb9
function test_gumbel_mcts_training_ratio(setup::NamedTuple, search_sims::Integer, search_threads::Integer; target_ratio = 8, test_batches = 1_000)
	rate = 0.1f0
	output = setup.train_networks!(rate, rate, test_batches, search_threads, target_ratio; clear_buffer = false, nsims = search_sims)
	true_ratio = output.training_rate / output.generation_rate
	(;output..., true_ratio = true_ratio)
end

# ╔═╡ b97ef521-903c-4493-af2d-47781be40530
# ╠═╡ disabled = true
#=╠═╡
TabularRL.average_stochastic_rollout(1_000, large_gridcapture_mdp, s -> gumbel_mcts_training_pipeline.mcts_policy_sample_action(s; nsims = 800), 1f0)
  ╠═╡ =#

# ╔═╡ d99a0f83-3927-4bc4-b60c-08ee7224e6c6
# ╠═╡ disabled = true
#=╠═╡
gumbel_mcts_search(large_gridcapture_mdp, 1f0, gumbel_mcts_training_pipeline.π_dist!, 1f0, large_gridcapture_mdp.initialize_state(), Returns(0f0); nsims = 20, depth = 1000, use_vmix = true, min_reward = -1f0, max_reward = 1f0)
  ╠═╡ =#

# ╔═╡ ca823968-ee51-45a6-9d52-a45c7586a5d0
# ╠═╡ disabled = true
#=╠═╡
@plutoprofview runepisode(gomoku_hard_mdp; π = s -> gumbel_mcts_search(gomoku_hard_mdp, 1f0, gumbel_mcts_training_pipeline3.π_dist!, 1f0, s, Returns(0f0); nsims = 100, depth = 1000, use_vmix = true, min_reward = -1f0, max_reward = 1f0)[1])
  ╠═╡ =#

# ╔═╡ 2b497bc2-f01b-4274-90f5-f5077c15913f
# ╠═╡ disabled = true
#=╠═╡
@btime runepisode(gomoku_easy_mdp; max_steps = 10)
  ╠═╡ =#

# ╔═╡ 5d1c1e17-51d3-430b-9bd0-077d6d32c21e
# ╠═╡ disabled = true
#=╠═╡
@btime runepisode(gomoku_medium_mdp; max_steps = 10)
  ╠═╡ =#

# ╔═╡ 8e700fca-3563-4cc6-a92a-dce1705b6dc8
# ╠═╡ disabled = true
#=╠═╡
@btime runepisode(gomoku_hard_mdp; max_steps = 10)
  ╠═╡ =#

# ╔═╡ 4acd4499-eee5-4bdf-a32d-0d41172eedb8
# ╠═╡ disabled = true
#=╠═╡
@btime runepisode(gomoku_expert_mdp; max_steps = 10)
  ╠═╡ =#

# ╔═╡ 065f9eaf-d180-4e6f-ab17-7e710a383ef7
# ╠═╡ disabled = true
#=╠═╡
@plutoprofview for _ in 1:1000 runepisode(gomoku_expert_mdp) end
  ╠═╡ =#

# ╔═╡ f1b3e26c-5ac5-48ea-9e82-e4d4fa84c8e4
function create_gumbel_mcts_policy_evaluation(mdp::StateMDP, π_dist!; nsims = 100, depth = 1000, γ = 1f0, use_vmix = false, persist_search_tree::Bool = false, kwargs...)
	mcts_policy = make_gumbel_mcts_policy(mdp, γ, π_dist!, 1f0, Returns(0f0); nsims = nsims, depth = depth, use_vmix = use_vmix, kwargs...)

	function generate_episode()
		s0 = mdp.initialize_state()
		i_a0 = mcts_policy.policy(s0)
		state_history = [s0]
		action_history = [i_a0]
		data = extract_root_training_data(mdp, s0, 1f0, mcts_policy.search_state...; use_vmix = use_vmix)
		data_history = [(;deepcopy(data)..., visit_counts = copy(mcts_policy.search_state[1][s0]))]
		(r, s) = mdp.ptf(s0, i_a0)
		reward_history = [r]
		while !mdp.isterm(s)
			#by default clear visit counts and values from the search tree after every step of an episode
			!persist_search_tree && mcts_policy.clear_search_tree!()
			i_a = mcts_policy.policy(s)
			data = extract_root_training_data(mdp, s, 1f0, mcts_policy.search_state...; use_vmix = use_vmix)
			push!(state_history, s)
			push!(action_history, i_a)
			push!(data_history, (;deepcopy(data)..., visit_counts = copy(mcts_policy.search_state[1][s])))
			(r, s) = mdp.ptf(s, i_a)
			push!(reward_history, r)
		end

		l = length(reward_history)
		Gs = copy(reward_history)
		if isone(γ)
			Gs .= last(reward_history)
		else
			for t in l - 1:1
				Gs[t] = γ*Gs[t+1]
			end
		end

		return (states = state_history, actions = action_history, rewards = reward_history, Gs = Gs, s_term = s, mcts_data = data_history)
	end

	return (mcts_policy = mcts_policy.policy, generate_mcts_episode = generate_episode)
end

# ╔═╡ 8c5e4b7f-a1b3-4d54-9879-04aee828561a
#=╠═╡
const gumbel_mcts_evaluation1 = create_gumbel_mcts_policy_evaluation(large_gridcapture_hard_mdp, gumbel_π_dist_test!; nsims = 1000, use_vmix = true, topk = 16)
  ╠═╡ =#

# ╔═╡ 5f129b73-a4c9-476b-a162-64c0b1cedc5b
#=╠═╡
const gumbel_mcts_episode1 = gumbel_mcts_evaluation1.generate_mcts_episode()
  ╠═╡ =#

# ╔═╡ aeff1449-b8ad-4a36-b96f-5c90fc11178f
#=╠═╡
@bind gumbel_mcts_episode1_step Slider(1:length(gumbel_mcts_episode1.states); show_value=true)
  ╠═╡ =#

# ╔═╡ 34b4ec27-5e37-429f-8873-c1bba4c20130
#=╠═╡
let
	step = gumbel_mcts_episode1_step
	s = gumbel_mcts_episode1.states[step]
	π = fcann_large_gridcapture_policy_solution.policy_function(s)
	π′ = gumbel_mcts_episode1.mcts_data[step].improved_policy
	visit_counts = gumbel_mcts_episode1.mcts_data[step].visit_counts
	
	@htl("""
		 <div style = "display: flex;">
		 <div>
		 <div>Original Policy</div>
		 $(HTML(html_board(s; cell_size = 40, policy = π, visit_count = visit_counts)))
		</div>

		 <div>
		 <div>Search Policy</div>
	   $(HTML(html_board(s; cell_size = 40, policy = π′, visit_count = visit_counts, candidate_move = gumbel_mcts_episode1.actions[step])))
		 </div>
		 </div>
		 """) => (action_values = gumbel_mcts_episode1.mcts_data[step].q_values |> x -> reshape(x, 8, 8),)
	   
end
  ╠═╡ =#

# ╔═╡ 7155e745-d3c2-466c-9c0d-aef997f0901f
#=╠═╡
const gumbel_mcts_episode1_data = generate_gumbel_mcts_training_data(gumbel_mcts_episode1, large_gridcapture_mdp, state_to_features(large_gridcapture_mdp.initialize_state()), update_features!)
  ╠═╡ =#

# ╔═╡ 8e0a58d4-ba9d-4b72-97ec-060fd4bd4767
#=╠═╡
NamedTuple{(:state, :feature_vector, :policy_target, :value_target)}([a[1] for a in gumbel_mcts_episode1_data])
  ╠═╡ =#

# ╔═╡ 26f5c1e0-5162-42ee-9aaf-61a8c5682b1b
#=╠═╡
@btime gumbel_mcts_evaluation1.mcts_policy($(large_gridcapture_mdp.initialize_state()))
  ╠═╡ =#

# ╔═╡ 7dbcc40d-c107-4322-bb5c-7f66ef40dc5b
#=╠═╡
@plutoprofview gumbel_mcts_evaluation1.mcts_policy(large_gridcapture_mdp.initialize_state())
  ╠═╡ =#

# ╔═╡ 76eefb47-cf41-414c-a187-756f5a8e6dbf
md"""
# Training Setup Function
"""

# ╔═╡ 1e79413b-b509-4adb-927c-5a99d931043b
function setup_gumbel_mcts_training(mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, feature_vector::V, update_feature_vector!::Function, hidden_layers::Vector{Int64}, buffer_size::Integer, batch_size::Integer; 
					reslayers::Integer = 0, 
					use_μP::Bool = true, 
					train_policy_params::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, length(mdp.actions), reslayers, use_μP),
					train_value_params::FCANNParams{T} = initialize_fcann_value_params(train_policy_params, use_μP), l2::T = zero(T), dropout::T = zero(T),
					search_policy_params::FCANNParams{T} = deepcopy(train_policy_params),
					search_value_params::FCANNParams{T} = deepcopy(train_value_params),
					activation_list::Vector{Bool} = fill(true, length(hidden_layers)), 
					use_gpu::Bool = false,
					eval_episodes::Integer = 1_000,
					eval_sims::Integer = 800,
					∇v̂::FCANNParams{T} = deepcopy(train_value_params), 
					∇π::FCANNParams{T} = deepcopy(train_policy_params),
					policy_and_value_components = form_policy_and_value_function(mdp, feature_vector, update_feature_vector!, search_policy_params, search_value_params),
					π_dist!::Function = function π_dist!(v::Vector{T}, s::S) where T<:Real
						output = policy_and_value_components.policy_and_value(s; policy = v)
						v .= log.(v) #convert distribution into logits
						return (output.value)
					end,		
					kwargs...) where {S, A, P, F1, F2, F3, V, T<:Real}

	#if search_policy_params or search_value_params are updated, then the π_dist! function will change behavior accordingly.  These are also the parameters copied into the generation vector every time !update_generation() is used

	#train_policy_params and train_value_params are the parameters updated by the training function with batch data
	
	
	#generate training data from one episode
	function generate_data(;search_kwargs...)
		mcts_setup = create_gumbel_mcts_policy_evaluation(mdp, π_dist!; kwargs..., γ = γ, search_kwargs...)
		episode = mcts_setup.generate_mcts_episode()
		generate_gumbel_mcts_training_data(episode, mdp, feature_vector, update_feature_vector!)
	end

	#add an episode of data to an existing set
	function accumulate_data!(data::NamedTuple; search_kwargs...)
		new_data = generate_data(;search_kwargs...)
		for k in keys(data)
			append!(data[k], new_data[k])
		end
		return data, length(first(new_data))
	end

	function form_mcts_policy_and_value(components::NamedTuple)
		function π_dist_mcts!(v::Vector{T}, s::S)
			output = components.policy_and_value(s; policy = v)
			v .= log.(v)
			return output.value
		end

		π_mcts(s; search_kwargs...) = gumbel_mcts_search(mdp, γ, π_dist_mcts!, 1f0, s, Returns(true); kwargs..., search_kwargs...)
		π_mcts_sample_action(s; search_kwargs...) = π_mcts(s; search_kwargs...)[1]
		(;components..., mcts_policy = π_mcts, mcts_policy_sample_action = π_mcts_sample_action)
	end

	function form_mcts_policy_and_value(policy_params::FCANNParams{T}, value_params::FCANNParams{T})
		components = form_policy_and_value_function(mdp, feature_vector, update_feature_vector!, policy_params, value_params)
		form_mcts_policy_and_value(components)
	end

	function extract_generation_params(generation::Integer)
		generation < 1 && error("Cannot extract a generation less than 1")
		l = length(policy_and_value_generations)
		generation > l && error("Attempting to use generation $generation but there are only $l generations")
		policy_and_value_generations[generation]
	end
	
	function extract_generation_mcts_policy_and_value(generation::Integer)
		gen = extract_generation_params(generation)
		form_mcts_policy_and_value(gen.policy_params, gen.value_params)
	end

	function compute_performance(policy_params::FCANNParams{T}, value_params::FCANNParams{T}; nsims = eval_sims, search_kwargs...)
		components = form_mcts_policy_and_value(policy_params, value_params)

		#if only one simulation is requested, then just use the policy function on its own since this is equivalent to what mcts will do without as much overhead
		isone(nsims) && return (nsims = 1, performance = TabularRL.average_stochastic_rollout(eval_episodes, mdp, components.policy_sample_action, γ))
		
		(nsims = nsims, performance = TabularRL.average_stochastic_rollout(eval_episodes, mdp, s -> components.mcts_policy_sample_action(s; nsims = nsims, search_kwargs...), γ))
	end

	function compute_performance(generation::Integer; kwargs...)
		gen = extract_generation_params(generation)
		compute_performance(gen.policy_params, gen.value_params; kwargs...)
	end

	function compute_performance(;kwargs...)
		l = length(policy_and_value_generations)
		compute_performance(l; kwargs...)
	end

	buffer_data = (states = CircularBuffer{S}(buffer_size), feature_vectors = CircularBuffer{V}(buffer_size), policy_targets = CircularBuffer{Vector{T}}(buffer_size), value_targets = CircularBuffer{T}(buffer_size))

	buffer_lock = ReentrantLock()
	
	policy_and_value_generations = [(; policy_params = deepcopy(search_policy_params), value_params = deepcopy(search_value_params), compute_performance(search_policy_params, search_value_params)...)]
	
	#copy training parameters into the fixed parameters and save a copy of both as a record
	function update_generation!(;search_kwargs...)
		copy!(search_policy_params, train_policy_params)
		copy!(search_value_params, train_value_params)
		performance = compute_performance(train_policy_params, train_value_params; search_kwargs...)
		push!(policy_and_value_generations, (;policy_params = deepcopy(search_policy_params), value_params = deepcopy(search_value_params), performance...))
	end

	function restore_generation!(i::Integer)
		#restore both the parameters used for the data generation and the training parameters to the last best known performance
		copy!(search_policy_params, policy_and_value_generations[i].policy_params)
		copy!(search_value_params, policy_and_value_generations[i].value_params)
		copy!(train_policy_params, policy_and_value_generations[i].policy_params)
		copy!(train_value_params, policy_and_value_generations[i].value_params)
	end

	#generate episodes of training data
	function accumulate_data!(max_episodes::Integer, max_steps::Integer; search_kwargs...)
		items_added = 0
		for i in 1:max_episodes
			_, l = accumulate_data!(buffer_data; search_kwargs...)
			items_added += l
			(items_added > max_steps) && break
		end

		return buffer_data
	end

	#generate episodes of training data in parallel and add to the buffer safely
	function accumulate_data!(max_episodes::Integer, max_steps::Integer, num_threads::Integer; search_kwargs...)
		data_channel = Channel{NamedTuple}(100)
		episode_count = Threads.Atomic{Int}(0)
		step_count = Threads.Atomic{Int}(0)

		stop = Threads.Atomic{Bool}(false)

		function add_data!()
			data = generate_data(;search_kwargs...)
			put!(data_channel, data)
		end

		for _ in 1:num_threads
			@spawn begin
				while !stop[]
					add_data!()
					if episode_count[] >= max_episodes || step_count[] >= max_steps
						stop[] = true
					end
				end
			end
		end
		
		
		while !stop[]
			new_data = take!(data_channel) #remove one episode of new data from the channel
			for k in keys(buffer_data)
				append!(buffer_data[k], new_data[k])
			end
				
			episode_count[] += 1
			step_count[] += length(first(new_data))
			if episode_count[] >= max_episodes || step_count[] >= max_steps
				stop[] = true
			end
		end

		
		return buffer_data
	end

	#this stop signal will work on any instance of `accumulate_data!` running from this setup
	stop_signal = Threads.Atomic{Bool}(false)
	stop_process!() = stop_signal[] = true

	#generate episodes of training data in parallel and add to the buffer safely until a stop signal is received
	function accumulate_data!(num_threads::Integer; search_kwargs...)
		stop_signal[] = false
		data_channel = Channel{NamedTuple}(100)
		
		step_count = Atomic{Int}(0)
		start_time = Atomic{Float64}(time())
		duration = Atomic{Float64}(0)
		

		function add_data!()
			data = generate_data(;search_kwargs...)
			put!(data_channel, data)
		end

		pause_signal = Threads.Atomic{Bool}(false)
		pause_process!() = pause_signal[] = true
		
		function resume_process!()
			  stop_signal[] && error("Cannot resume a stopped process")
			step_count[] = 0
			start_time[] = time()
			pause_signal[] = false
		end
		
		check_data_rate() = step_count[] / duration[]
		
		for _ in 1:num_threads
			@spawn begin
				while !stop_signal[]
					add_data!()
					while !stop_signal[] && pause_signal[]
						sleep(0.01)
					end
				end
			end #continue if pause signal is empty
		end

		@spawn while !stop_signal[] || !isempty(data_channel)
			new_data = take!(data_channel) #remove one episode of new data from the channel
			lock(buffer_lock)
			try 
				for k in keys(buffer_data)
					append!(buffer_data[k], new_data[k])
				end
			finally
				unlock(buffer_lock)
			end
			step_count[] += length(first(new_data))
			duration[] = time() - start_time[]
		end
		
		return (stop_process! = stop_process!, pause_process! = pause_process!, resume_process! = resume_process!, check_data_rate = check_data_rate, step_count = step_count)
	end

	function clear_data!()
		for k in keys(buffer_data)
			empty!(buffer_data[k])
		end
		return buffer_data
	end

	#training data variables
	input = form_feature_matrix(mdp, feature_vector, batch_size)
	value_output = zeros(T, batch_size, 1)
	policy_output = zeros(T, batch_size, length(mdp.actions))
	policy_output_transposed = zeros(T, length(mdp.actions), batch_size)
	batch_inds = collect(1:batch_size)

	#updates batch inds and then populates the input as well as the value and policy output
	function update_input_output!(input, policy_output, value_output, buffer_data::NamedTuple)
		live_buffer_size = length(buffer_data.states)
		if live_buffer_size < batch_size 
			@info "Buffer only has $live_buffer_size elements to fill a batch of $batch_size"
			return nothing
		end
		
		sample!(1:live_buffer_size, batch_inds; replace = false)
		@inbounds @simd for i in 1:batch_size
			ind = batch_inds[i]
			update_feature_matrix!(input, buffer_data.feature_vectors[ind], i)
			value_output[i] = buffer_data.value_targets[ind]
			update_feature_matrix!(policy_output_transposed, buffer_data.policy_targets[ind], i)
		end

		transpose!(policy_output, policy_output_transposed)
		return (input = input, policy_output = policy_output, value_output = value_output)
	end

	#setup training arguments for policy and value function
	policy_setup = setup_fcann_batch_policy_arguments(train_policy_params, batch_size, l2, dropout, use_μP, activation_list)
	value_setup = setup_fcann_batch_value_arguments(policy_setup, train_value_params, batch_size, l2, dropout, use_μP, activation_list)
	
	input_size = value_setup.value_gradient_args[1]
	value_tanh_grad_z = value_setup.value_gradient_args[5]
	value_activations = value_setup.value_gradient_args[6]
	value_deltas = value_setup.value_gradient_args[7]
	onesvec = value_setup.value_gradient_args[8]
	scales = value_setup.value_gradient_args[11] #note that these scales already multiply the gradient by -1 to minimize cost function

	policy_tanh_grad_z = policy_setup.policy_gradient_args[3]
	policy_activations = policy_setup.policy_gradient_args[4]
	policy_deltas = policy_setup.policy_gradient_args[5]
	
	function update_value_gradient!()
		FCANN.nnCostFunction(train_value_params.weights..., input_size, hidden_layers, input, value_output, l2, ∇v̂.weights..., value_tanh_grad_z, value_activations, value_deltas, onesvec, dropout; resLayers = reslayers, costFunc = "sqErr", activation_list = activation_list, input_orientation = 'T')
		scale_fcann_params!(∇v̂, scales)
	end

	function update_policy_gradient!()
		FCANN.nnCostFunction(train_policy_params.weights..., hidden_layers, input, policy_output, l2, ∇π.weights..., policy_tanh_grad_z, policy_activations, policy_deltas, onesvec, dropout; resLayers = reslayers, activation_list = activation_list, input_orientation = 'T')
		scale_fcann_params!(∇π, scales)
	end

	function train_minibatch!(α_θ::T, α_w::T)
		lock(buffer_lock)
		update_status = try
			#sample a new minibatch
			update_input_output!(input, policy_output, value_output, buffer_data)
		finally
			unlock(buffer_lock)
		end
		
		if isnothing(update_status) 
			@warn "Failed to update minibatch"
			return nothing
		end
		
		#update gradients
		update_policy_gradient!()
		update_value_gradient!()

		#update parameters with gradients and learning rate
		ReinforcementLearning.update_params_with_gradient!(train_policy_params, α_θ, ∇π)
		ReinforcementLearning.update_params_with_gradient!(train_value_params, α_w, ∇v̂)
	end

	return (;generate_data = generate_data, accumulate_data! = accumulate_data!, clear_data! = clear_data!, stop_data_collection! = stop_process!, saved_data = buffer_data, train_minibatch! = train_minibatch!, buffer_size = buffer_size, batch_size = batch_size, policy_and_value_generations = policy_and_value_generations, update_generation! = update_generation!, restore_generation! = restore_generation!, π_dist! = π_dist!, update_input_output! = update_input_output!, extract_generation_mcts_policy_and_value = extract_generation_mcts_policy_and_value, compute_performance = compute_performance,  form_mcts_policy_and_value(policy_and_value_components)...)
end

# ╔═╡ 0d676e61-5ec8-4fda-a748-dfb9a2abf5a8
#=╠═╡
const gumbel_mcts_training_setup1 = setup_gumbel_mcts_training(large_gridcapture_mdp, 1f0, state_to_features(large_gridcapture_mdp.initialize_state()), update_features!, 
															   [64, 64], #network architecture
															   100_000, #buffer size
															   16 #batch size
															   ; reslayers = 1, nsims = 20, topk = 16, depth = 1_000, use_vmix = true, π_dist! = gumbel_π_dist_test!, min_reward = -1f0, max_reward = 1f0, eval_sims = 1)
  ╠═╡ =#

# ╔═╡ 39f265e4-037e-42ab-b09c-3079f463f496
#=╠═╡
#creates training data from 1 episode
gumbel_mcts_training_setup1.generate_data()
  ╠═╡ =#

# ╔═╡ 0d3307b3-c50e-4d6c-a4b8-1cfd79ed0d27
#=╠═╡
gumbel_mcts_training_setup1.saved_data
  ╠═╡ =#

# ╔═╡ 0b22c86d-561a-4839-a088-2a097da28df6
#=╠═╡
gumbel_mcts_training_setup1.accumulate_data!(10, typemax(Int64))
  ╠═╡ =#

# ╔═╡ 88501e87-6bed-4c22-9524-24b46e1dc477
#=╠═╡
gumbel_mcts_training_setup1.accumulate_data!(10, typemax(Int64), 10)
  ╠═╡ =#

# ╔═╡ 7a00fa12-f66c-4e83-a1e7-e0aa51c211b7
#=╠═╡
gumbel_mcts_training_setup1.accumulate_data!(typemax(Int64), 100_000, 40)
  ╠═╡ =#

# ╔═╡ e1bd21eb-6d43-4d0a-8cf2-b86d18f3eab2
#=╠═╡
let
	setup = gumbel_mcts_training_setup1

	α_θ = 0.1f0
	α_w = 0.1f0

	buffer_batches = ceil(Int64, length(setup.saved_data.states) / setup.batch_size)
	training_epochs = 10
	batches = buffer_batches*training_epochs
	
	@info "Ready to start training with $(setup.batch_size) of accumulated data"
	t = time()
	wasted_training_time = 0.0
	for epoch in 1:training_epochs
		for i in 1:buffer_batches
			setup.train_minibatch!(α_θ, α_w)
		end

		#calculate performance after every training epoch
		setup.update_generation!()
	end
	
	runtime = time() - t
	data_processed = batches * setup.batch_size
	process_rate = data_processed / runtime

	(examples_trained = data_processed, training_rate = process_rate, generation_performance = [a.performance for a in setup.policy_and_value_generations])
end
  ╠═╡ =#

# ╔═╡ 699232e8-d39c-4623-9d23-6527adbc9baf
#=╠═╡
gumbel_mcts_training_setup1
  ╠═╡ =#

# ╔═╡ d5986e30-4b1f-42c2-9b96-05a024b513af
#=╠═╡
const distillation_episode = runepisode(large_gridcapture_mdp; π = gumbel_mcts_training_setup1.policy_sample_action)
  ╠═╡ =#

# ╔═╡ 8d422c7e-5784-4a86-b832-9a37af4bcc08
#=╠═╡
@bind distillation_step Slider(1:distillation_episode[5])
  ╠═╡ =#

# ╔═╡ 5d3e6956-3aae-4202-ae17-9dbb5f3f789e
#=╠═╡
let
	step = distillation_step
	s = distillation_episode[1][step]
	b1 = @htl("""
			  <div>
			  <div>Distilled Policy
			  Value Estimate: $(gumbel_mcts_training_setup1.value_function(s))
			  </div> 
			  
			  <div>$(HTML(html_board(s; policy = gumbel_mcts_training_setup1.policy_function(s))))</div>
			  </div>
			  """)

	b2 = @htl("""
			  <div>
			  <div>Original Policy
			  Value Estimate: $(fcann_large_gridcapture_policy_solution.value_function(s))
			  </div>
			  
			  $(HTML(html_board(s; policy = fcann_large_gridcapture_policy_solution.policy_function(s))))
			  </div>
			  """)

	(b1, b2)
end
  ╠═╡ =#

# ╔═╡ f78026ff-cf97-4d20-a705-8c82cc9f0d21
#=╠═╡
const large_gridcapture_distilled_ac_performance = TabularRL.average_stochastic_rollout(10_000, large_gridcapture_mdp, gumbel_mcts_training_setup1.policy_sample_action, 1f0)
  ╠═╡ =#

# ╔═╡ 0c3aa32d-1968-4338-b29c-51f3ab892f86
#=╠═╡
const large_gridcapture_distilled_mcts_performance10 = TabularRL.average_stochastic_rollout(1_000, large_gridcapture_mdp, s -> gumbel_mcts_training_setup1.mcts_policy_sample_action(s; nsims = 10, min_reward = -1f0, max_reward = 1f0, rescale_values=true), 1f0)
  ╠═╡ =#

# ╔═╡ 3f7e7968-9b7b-47bd-a3d4-3f06be6dc573
#=╠═╡
md"""
|Original Policy|Distilled Search Policy|Distilled Search Policy+10sim MCTS|
|---|---|---|
|$(round(Float64(large_gridcapture_ac_performance); sigdigits = 4))|$(round(Float64(large_gridcapture_distilled_ac_performance); sigdigits = 4))|$(round(Float64(large_gridcapture_distilled_mcts_performance10); sigdigits = 4))|
"""
  ╠═╡ =#

# ╔═╡ a1d18f21-000e-4ee8-8296-0304ccae5bdb
#=╠═╡
function benchmark_data_generation(threads::Integer; sim_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512], time_limit = 1, kwargs...)
	[begin
	 	sleep(1)
	 	rate = measure_data_generation_rate(gumbel_mcts_training_setup1, time_limit; num_data_threads = threads, nsims = nsims, kwargs...)
	 	(nsims = nsims, rate = rate) 
	 	end for nsims in sim_list]
end
  ╠═╡ =#

# ╔═╡ 2e7fecbc-1952-44bf-97ad-b050cdfce8be
#=╠═╡
data_generation_traces = let
	traces = [begin
			  output = benchmark_data_generation(threads)
			xvals = [x.nsims for x in output]
			yvals = [x.rate for x in output]
			scatter(x = xvals, y = yvals, name = "Threads = $threads")
	end
	for threads in [1, 2, 4, 8, 16, 32]]
end
  ╠═╡ =#

# ╔═╡ a9d34520-f805-4548-8d71-19797cc519e1
#=╠═╡
plot(data_generation_traces, Layout(xaxis = attr(type =:log, title = "Number of Simulations"), yaxis = attr(type = :log, title = "Examples Generated per Second")))
  ╠═╡ =#

# ╔═╡ 0e27a2bc-41ed-4575-8952-20ca904a92d5
#=╠═╡
measure_data_generation_rate(gumbel_mcts_training_setup1, 1; num_data_threads = 16, nsims = 2)
  ╠═╡ =#

# ╔═╡ 218862a6-0625-4650-920d-6b4424c2e235
#=╠═╡
measure_data_rate(gumbel_mcts_training_setup1; num_data_threads = 30, nsims = 2, l = 100_000)
  ╠═╡ =#

# ╔═╡ 96611207-eda1-4153-ae3e-799b198e5b93
#=╠═╡
measure_data_generation_rate(gumbel_mcts_training_setup1)
  ╠═╡ =#

# ╔═╡ 6175acee-1310-471a-8258-08e2ea4f4ff1
#=╠═╡
function measure_training_rate(batch_size::Integer; hidden_layers = [64, 64], time_limit = 1) 
	setup = setup_gumbel_mcts_training(large_gridcapture_hard_mdp, 1f0, state_to_features(large_gridcapture_hard_mdp.initialize_state()), update_features!, hidden_layers, 100_000, batch_size; reslayers = 1, eval_sims = 1)

	#fill buffer enough to test
	setup.accumulate_data!(typemax(Int64), 10*batch_size, max(1, nthreads()-10); nsims = 100)

	t = time()
	batches_trained = 0
	while time() - t < time_limit
		gumbel_mcts_training_setup1.train_minibatch!(0.1f0, 0.1f0)
		batches_trained += 1
	end
	elapsed = time() - t
	
	batches_trained*batch_size / elapsed
end
  ╠═╡ =#

# ╔═╡ aa2ec6b2-863e-4c44-b95e-d8582d0e20c2
#=╠═╡
training_rate_trace = let
	batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
	rates = [measure_training_rate(batch_size) for batch_size in batch_sizes]
	xvals = batch_sizes
	yvals = rates
	scatter(x = xvals, y = yvals)
end
  ╠═╡ =#

# ╔═╡ d6598ee5-92e7-4d24-a998-78b9b0da5da0
#=╠═╡
plot(training_rate_trace, Layout(xaxis = attr(title = "Batch Size", type = :log), yaxis = attr(title = "Examples per Second", type = :log), title = "Training Rate"))
  ╠═╡ =#

# ╔═╡ a4b7488b-85d2-4b93-b218-d0490b37170b
#=╠═╡
function test_data_ratio(nthreads::Integer, nsims::Integer; time_limit = 1, run_training=true, hidden_layers = [64, 64], batch_size = 16)
	setup = setup_gumbel_mcts_training(large_gridcapture_hard_mdp, 1f0, state_to_features(large_gridcapture_hard_mdp.initialize_state()), update_features!, hidden_layers, 100_000, batch_size; reslayers = 1, eval_sims = 1)

	#fill buffer enough to test
	setup.accumulate_data!(typemax(Int64), 10*batch_size, nthreads; nsims = nsims)
	
	t = time()
	data_process = setup.accumulate_data!(nthreads; nsims = nsims)

	batches_trained = 0
	while time() - t < time_limit
		if run_training
			setup.train_minibatch!(0.1f0, 0.1f0)
			batches_trained += 1
		else
			sleep(0.01)
		end
	end
	elapsed = time() - t
	data_process.stop_process!()

	generation_rate = data_process.check_data_rate()
	training_rate = batches_trained*setup.batch_size / elapsed

	ratio = training_rate / generation_rate

	sleep(0.1)

	#this test confirms the data generation rate is not slowed down by the presence of training
	pure_generation_rate = measure_data_generation_rate(setup, 1; num_data_threads = nthreads, nsims = nsims)
	
	(generation_rate = generation_rate, training_rate = training_rate, ratio = ratio, pure_generation_rate = pure_generation_rate)
end
  ╠═╡ =#

# ╔═╡ 36c416f5-2304-466b-b507-28d58962a8b0
#=╠═╡
data_ratio_traces = let
	nsims = [1, 2, 4, 8, 16, 32, 64]
	nthreads = [1, 2, 4, 8, 16, 32]
	traces = [begin
			 yvals = [test_data_ratio(nthreads, nsims; time_limit = 5).ratio for nsims in nsims]
		   sleep(1)
		  	scatter(x = nsims, y = yvals, name = "$nthreads Threads")
	end
	for nthreads in nthreads]
end
  ╠═╡ =#

# ╔═╡ 80d6264f-1def-47f2-8cfa-913b88898da3
#=╠═╡
plot([data_ratio_traces; scatter(x = [1, 64], y = [8, 8], line_color = "black", line_dash = "dash", mode = "lines", name = "Target Ratio")], Layout(xaxis = attr(title = "Number of Simulations", type = :log), yaxis = attr(title = "Training Rate / Generation Rate", type = :log), title = "Data Ratios at 16 Batch Size"))
  ╠═╡ =#

# ╔═╡ 28c04d73-d6e0-4167-81c6-186005b3f163
#=╠═╡
test_data_ratio(32, 64)
  ╠═╡ =#

# ╔═╡ d9199030-4562-4807-b3ef-ce0df0fc16d2
#=╠═╡
data_ratio_64_traces = let
	nsims = [1, 2, 4, 8, 16, 32, 64]
	nthreads = [1, 2, 4, 8, 16, 32]
	traces = [begin
			 yvals = [test_data_ratio(nthreads, nsims; time_limit = 5, batch_size = 64).ratio for nsims in nsims]
		   sleep(1)
		  	scatter(x = nsims, y = yvals, name = "$nthreads Threads")
	end
	for nthreads in nthreads]
end
  ╠═╡ =#

# ╔═╡ 635bee38-38df-4db9-b40c-418c3b451a5e
#=╠═╡
plot([data_ratio_64_traces; scatter(x = [1, 64], y = [8, 8], line_color = "black", line_dash = "dash", mode = "lines", name = "Target Ratio")], Layout(xaxis = attr(title = "Number of Simulations", type = :log), yaxis = attr(title = "Training Rate / Generation Rate", type = :log), title = "Data Ratios at 64 Batch Size"))
  ╠═╡ =#

# ╔═╡ e1657a7a-d4a5-4573-b7cb-fe3b0e1159c0
#=╠═╡
test_data_ratio(32, 16; batch_size = 64, time_limit = 5)
  ╠═╡ =#

# ╔═╡ 8229ce52-eb6c-4810-a223-e9706a9ffa20
function training_pipeline(setup_args...; setup_kwargs...)
	setup = setup_gumbel_mcts_training(setup_args...; setup_kwargs...)

	function train_networks!(α_θ::T, α_w::T, batches::Integer, data_threads::Integer, target_ratio::Real; sleep_time::Real = 0.01, clear_buffer::Bool = true, search_kwargs...) where T<:Real
		if clear_buffer
			@info "Clearing existing data"
			#clear the buffer of existing data
			setup.clear_data!()
		end
		
		#start data accumulation process
		data_process = setup.accumulate_data!(data_threads; search_kwargs...)
		@info "Started Data Generation Process"
		
		#wait until a mini batch of data has been accumulated
		while length(first(setup.saved_data)) < setup.batch_size
			sleep(0.1)
		end

		@info "Ready to start training with $(setup.batch_size) of accumulated data"
		t = time()
		wasted_training_time = 0.0
		for i in 1:batches
			setup.train_minibatch!(α_θ, α_w)
			#wait if data is not being collected fast enough
			while (data_process.step_count[]*target_ratio < i*setup.batch_size)
				sleep(sleep_time)
				wasted_training_time += sleep_time
			end
		end
		runtime = time() - t
		data_processed = batches * setup.batch_size
		process_rate = data_processed / runtime

		data_process.stop_process!()
		data_rate = data_process.check_data_rate()
		(generation_rate = data_rate, training_rate = process_rate, examples_generated = data_process.step_count[], examples_trained = data_processed, wasted_training_time = wasted_training_time, training_time = runtime)
	end

	return (;train_networks! = train_networks!, setup...)
end	

# ╔═╡ c1f1052c-1111-4453-bd19-72add618e489
#=╠═╡
const gumbel_mcts_training_pipeline = training_pipeline(large_gridcapture_mdp, 1f0, state_to_features(large_gridcapture_mdp.initialize_state()), update_features!, [64, 64], 1_000_000, 16; reslayers = 1, nsims = 40, depth = 1_000, use_vmix = true, eval_episodes = 100_000, eval_sims = 1, min_reward = -1f0, max_reward = 1f0)
  ╠═╡ =#

# ╔═╡ ad3b033b-06b4-42b1-afe9-cd6481889661
#=╠═╡
let
	rate = 0.1f0
	num_threads = 35
	target_ratio = 1000
	nsims = 60
	output = gumbel_mcts_training_pipeline.train_networks!(rate, rate, 1_000, num_threads, target_ratio; clear_buffer = false, nsims = nsims)
	true_ratio = output.training_rate / output.generation_rate
	(;output..., true_ratio = true_ratio)
end
  ╠═╡ =#

# ╔═╡ b31dcfa0-39af-4f14-b224-12c0da7217f7
#=╠═╡
begin #whenever the pipeline is redefined, reset the button so training doesn't proceed immediately
	gumbel_mcts_training_pipeline
	@bind begin_pipeline1 CounterButton("Begin Training")
end
  ╠═╡ =#

# ╔═╡ b1f8f9b3-4d02-4373-bbda-c22c23d911d7
#=╠═╡
if iszero(begin_pipeline1)
	md"""
	Waiting to begin training
	"""
else
	let
		search_sims = 60
		search_threads = 35
		target_ratio = 1000
		
		eval_sims = 1
		#find the best performance consistent with the number of evaluation simulations being used
		(best_performance, best_generation) = findmax(a -> a.nsims != eval_sims ? typemin(Float32) : a.performance, gumbel_mcts_training_pipeline.policy_and_value_generations)
		base_rate = 0.1f0
		rate_adjust = 1f0
		@info "Prior to training, best performance is $best_performance from generation $best_generation"
	
		failure_count = 0
		failure_limit = 5
		rounds = 10
		buffer_capacity = gumbel_mcts_training_pipeline.saved_data.states |> capacity
		epoch_batches = ceil(Int64, buffer_capacity / gumbel_mcts_training_pipeline.batch_size)
		gen_epochs = 1
		num_batches = gen_epochs * epoch_batches
		
		gumbel_mcts_training_pipeline.clear_data!()
		while failure_count < failure_limit
		# for round in 1:rounds
			gumbel_mcts_training_pipeline.train_networks!(base_rate*rate_adjust, base_rate*rate_adjust, num_batches, search_sims, target_ratio; clear_buffer = false, nsims = search_sims)
			gumbel_mcts_training_pipeline.update_generation!(;nsims = eval_sims)
			generation_performance = gumbel_mcts_training_pipeline.policy_and_value_generations[end].performance
			# @info "New generation performance is $generation_performance on round $round of $rounds"
			@info "New generation performance is $generation_performance"
			if generation_performance ≤ best_performance
				rate_adjust /= 2
				@info "New performance is worse than best performance of $best_performance from generation $best_generation.  Cutting learning rate adjustment in half to $rate_adjust."
				failure_count += 1
				@warn "Failed improvement $failure_count out of $failure_limit"
			else
				failure_count = 0
				@info "New best performance achieved on generation $(length(gumbel_mcts_training_pipeline.policy_and_value_generations))."
				best_performance = generation_performance
				best_generation = length(gumbel_mcts_training_pipeline.policy_and_value_generations)
			end
		end
		gumbel_mcts_training_pipeline.restore_generation!(best_generation)
		gumbel_mcts_training_pipeline.policy_and_value_generations
	end
end
  ╠═╡ =#

# ╔═╡ 5225bb20-b634-47b2-aa7b-317f3e4ce799
#=╠═╡
begin
	begin_pipeline1 #show generations every time training is run
	gumbel_mcts_training_pipeline.policy_and_value_generations
end
  ╠═╡ =#

# ╔═╡ 9dec854c-5abb-48c1-8170-e879b45dee47
#=╠═╡
begin
	begin_pipeline1 #show trained policy performance for best generation after training run
	TabularRL.average_stochastic_rollout(10_000, large_gridcapture_mdp, gumbel_mcts_training_pipeline.policy_sample_action, 1f0)
end
  ╠═╡ =#

# ╔═╡ af9a0734-e0c0-4553-bfdd-24f89e7ab8ec
#=╠═╡
begin
	begin_pipeline1 #show trained policy performance enhanced with MCTS after training run
	TabularRL.average_stochastic_rollout(1_000, large_gridcapture_mdp, s -> gumbel_mcts_training_pipeline.mcts_policy_sample_action(s; nsims = 200), 1f0)
end
  ╠═╡ =#

# ╔═╡ 1b27a559-6cc8-4659-8ec9-719b15dc5335
#=╠═╡
#using this function I can extract the policy function for any generation as well as an MCTS enhanced version
gumbel_mcts_training_pipeline.extract_generation_mcts_policy_and_value(1)
  ╠═╡ =#

# ╔═╡ ff6930d8-cf7b-453a-b25e-93e665dabfd0
#=╠═╡
function test_generation_mcts_performance(generation::Integer, nsims::Integer, nsamples::Integer)
	gen = gumbel_mcts_training_pipeline.extract_generation_mcts_policy_and_value(generation)

	TabularRL.average_stochastic_rollout(nsamples, large_gridcapture_mdp, s -> gen.mcts_policy_sample_action(s; nsims = nsims), 1f0)
end
  ╠═╡ =#

# ╔═╡ 33237e06-e784-47c8-8d45-8de457d46e8c
#=╠═╡
test_generation_mcts_performance(1, 10, 1_000)
  ╠═╡ =#

# ╔═╡ 021d04c1-245a-42c4-a193-2ec0d1e68094
#=╠═╡
const mcts_performances = let
	nsims = [10, 20, 40]
	Dict(n => [test_generation_mcts_performance(i, n, 1_000) for i in eachindex(gumbel_mcts_training_pipeline.policy_and_value_generations)] for n in nsims)
end
  ╠═╡ =#

# ╔═╡ f36dc75d-a14b-4e3e-b360-583455f6ec65
#=╠═╡
let
	f_win(x) = (x+1)/2 #convert performance to winrate
	f_win(x::Vector) = (x .+ 1) ./ 2
	tr1 = scatter(y = [f_win(a.performance) for a in gumbel_mcts_training_pipeline.policy_and_value_generations], name = "Raw Policy", mode = "lines+markers", line_dash = "dash")
	traces = [scatter(y = f_win(mcts_performances[n]), name = "$n sims", mode = "lines+markers") for n in keys(mcts_performances)]
	plot([tr1; traces], Layout(xaxis_title = "Generation", yaxis = attr(title = "Win Rate"), title = "Enhanced Policy Performance Over Generations"))
end
  ╠═╡ =#

# ╔═╡ 15a719e6-e399-469c-9aee-69a8bfcec175
#=╠═╡
const large_gridcapture_mcts_episode = begin
	Random.seed!(26)
	runepisode(large_gridcapture_mdp; π = s -> gumbel_mcts_training_pipeline.mcts_policy_sample_action(s; nsims = 200))
end
  ╠═╡ =#

# ╔═╡ 04b2a9de-6556-4389-993b-4b85aa16528e
#=╠═╡
@bind large_gridcapture_mcts_episode_step Slider(1:large_gridcapture_mcts_episode[5]+1; show_value=true)
  ╠═╡ =#

# ╔═╡ a24b8edd-8fcf-4d70-84c8-925463169230
#=╠═╡
let
	episode = large_gridcapture_mcts_episode
	step = large_gridcapture_mcts_episode_step
	term_state = step > episode[5]
	s = term_state ? episode[4] : episode[1][step]
	policy = term_state ? nothing : gumbel_mcts_training_pipeline.policy_function(s)
	action = term_state ? nothing : episode[2][step]
	v̂ = gumbel_mcts_training_pipeline.value_function(s)
	@htl("""
    <div style = "font-size: 2em;">Value Estimate: $v̂</div>
		 
	$(HTML(html_board(s; policy = policy, candidate_move = action)))
		 """)
end
  ╠═╡ =#

# ╔═╡ 38874107-272e-4bfc-b502-7aa84fd010f4
#=╠═╡
# const gumbel_mcts_evaluation = create_gumbel_mcts_policy_evaluation(large_gridcapture_mdp, gumbel_π_dist_test!; nsims = 1000, use_vmix = true, topk = 16)
const gumbel_mcts_evaluation = create_gumbel_mcts_policy_evaluation(large_gridcapture_mdp, gumbel_mcts_training_pipeline.π_dist!; nsims = 200, use_vmix = true, min_reward = -1f0, max_reward = 1f0)
  ╠═╡ =#

# ╔═╡ 41b1f97b-f847-4578-a762-1c8b7ccb3fc4
#=╠═╡
const gumbel_mcts_episode = gumbel_mcts_evaluation.generate_mcts_episode()
  ╠═╡ =#

# ╔═╡ 5f4cd84b-5d43-4f37-802c-840f0c9a6b36
#=╠═╡
@bind gumbel_mcts_episode_step Slider(1:length(gumbel_mcts_episode.states); show_value=true)
  ╠═╡ =#

# ╔═╡ 7d06c211-96ee-4af3-a9c9-e666988a155a
#=╠═╡
generate_gumbel_mcts_training_data(gumbel_mcts_episode, large_gridcapture_mdp, state_to_features(large_gridcapture_mdp.initialize_state()), update_features!)
  ╠═╡ =#

# ╔═╡ 9068f9d7-6006-490f-a53e-a1fea309e2c2
#=╠═╡
@htl("""
	 $(HTML(html_board(gumbel_mcts_episode.states[gumbel_mcts_episode_step]; cell_size = 40, policy = gumbel_mcts_training_pipeline.policy_function(gumbel_mcts_episode.states[gumbel_mcts_episode_step]))))

	  $(HTML(html_board(gumbel_mcts_episode.states[gumbel_mcts_episode_step]; cell_size = 40, policy = gumbel_mcts_episode.mcts_data[gumbel_mcts_episode_step].improved_policy, visit_count = gumbel_mcts_episode.mcts_data[gumbel_mcts_episode_step].visit_counts, candidate_move = gumbel_mcts_episode.actions[gumbel_mcts_episode_step])))
	 """)
	 
  ╠═╡ =#

# ╔═╡ 556cb9b8-1a57-49f0-a586-3d7541e7e839
function setup_data_ratio_test(mdp, γ::T, feature_vector, update_feature_vector!; buffer_size = 1_000_000, kwargs...) where T<:Real
	rate = one(T) / 10

	function test_setup(hidden_layers, reslayers::Integer, batch_size::Integer, num_threads::Integer, nsims::Integer, num_batches::Integer)
		setup = training_pipeline(mdp, γ, feature_vector, update_feature_vector!, hidden_layers, buffer_size, batch_size; reslayers = reslayers, kwargs..., eval_sims = 1)
		output = setup.train_networks!(rate, rate, num_batches, num_threads, 100_000; clear_buffer = false, nsims = nsims)
		true_ratio = output.training_rate / output.generation_rate
		(;output..., true_ratio = true_ratio)
	end

	return test_setup
end

# ╔═╡ 037e9bbe-4b1e-494a-a9a9-7b4785421731
#=╠═╡
const gumbel_mcts_training_ratio_test = setup_data_ratio_test(large_gridcapture_mdp, 1f0, state_to_features(large_gridcapture_mdp.initialize_state()), update_features!; use_vmix=true, depth = 1_000, min_reward = -1f0, max_reward = 1f0)
  ╠═╡ =#

# ╔═╡ 29b8619b-ae26-4145-9008-1c97893d1307
#=╠═╡
gumbel_mcts_training_ratio_test([64, 64], 1, 16, 35, 60, 10_000)
  ╠═╡ =#

# ╔═╡ 95a1544f-75d4-44ab-b818-2b32ec7cfe71
#=╠═╡
gumbel_mcts_training_ratio_test([64, 64], 1, 64, 35, 30, 10_000)
  ╠═╡ =#

# ╔═╡ 8dbf7751-6e07-417b-9359-e0510e5aed18
#=╠═╡
const gumbel_mcts_training_pipeline2 = training_pipeline(gomoku_easy_mdp, 1f0, state_to_features(gomoku_easy_mdp.initialize_state()), update_features!, fill(64, 2), 1_000_000, 16; reslayers = 1, nsims = 4, depth = 1_000, use_vmix = true, eval_episodes = 10_000, eval_sims = 1, min_reward = -1f0, max_reward = 1f0)
  ╠═╡ =#

# ╔═╡ 5f275280-55bd-4784-98d5-a2cd83bba977
#=╠═╡
test_gumbel_mcts_training_ratio(gumbel_mcts_training_pipeline2, 80, 35; target_ratio = 10000000, test_batches = 1_000)
  ╠═╡ =#

# ╔═╡ 47f2b03e-675d-4294-95ca-fc368e6181cb
#=╠═╡
begin #whenever the pipeline is redefined, reset the button so training doesn't proceed immediately
	gumbel_mcts_training_pipeline2
	@bind begin_pipeline2 CounterButton("Begin Training")
end
  ╠═╡ =#

# ╔═╡ c9834a8b-fbf8-4786-b589-ac90cb03cb28
#=╠═╡
if begin_pipeline2 > 0
	train_gumbel_mcts(gumbel_mcts_training_pipeline2, 80, 35; α_θ = 0.01f0, α_w = 0.01f0)
else
	md"""
	Waiting to train
	"""
end
  ╠═╡ =#

# ╔═╡ 1091b9bf-5063-4f2f-83f1-0cea199c68ac
#=╠═╡
TabularRL.average_stochastic_rollout(10_000, gomoku_easy_mdp, gumbel_mcts_training_pipeline2.policy_sample_action, 1f0)
  ╠═╡ =#

# ╔═╡ a6d4fb64-8e55-4551-9cba-547f7cc8e3d5
#=╠═╡
TabularRL.average_stochastic_rollout(1_000, gomoku_easy_mdp, s -> gumbel_mcts_search(gomoku_easy_mdp, 1f0, gumbel_mcts_training_pipeline2.π_dist!, 1f0, s, Returns(0f0); nsims = 800, depth = 1000, use_vmix = true, min_reward = -1f0, max_reward = 1f0)[1], 1f0)
  ╠═╡ =#

# ╔═╡ b5fdce32-079e-47a8-acb5-f891e99f3d7a
#=╠═╡
const gumbel_mcts_training_pipeline3 = training_pipeline(gomoku_hard_mdp, 1f0, state_to_features(gomoku_hard_mdp.initialize_state()), update_features!, fill(128, 4), 1_000_000, 16; reslayers = 1, nsims = 4, depth = 1_000, use_vmix = true, eval_episodes = 10_000, eval_sims = 1, min_reward = -1f0, max_reward = 1f0)
  ╠═╡ =#

# ╔═╡ 83798641-6218-480e-beed-183e676d1f08
#=╠═╡
test_gumbel_mcts_training_ratio(gumbel_mcts_training_pipeline3, 25, 35; target_ratio = 10000000, test_batches = 1_000)
  ╠═╡ =#

# ╔═╡ 88930b3e-d977-46db-a27f-9c3bcebeb2c9
#=╠═╡
begin #whenever the pipeline is redefined, reset the button so training doesn't proceed immediately
	gumbel_mcts_training_pipeline3
	@bind begin_pipeline3 CounterButton("Begin Training")
end
  ╠═╡ =#

# ╔═╡ 75e80bf5-3562-460c-8157-f46c28829d8f
#=╠═╡
if begin_pipeline3 > 0
	train_gumbel_mcts(gumbel_mcts_training_pipeline3, 25, 35; α_θ = 0.1f0, α_w = 0.1f0)
else
	md"""
	Waiting to train
	"""
end
  ╠═╡ =#

# ╔═╡ 77c6e7f9-aa96-4a44-9620-f403847c5120
#=╠═╡
let
	(best_performance, best_generation) = findmax(a -> a.performance, gumbel_mcts_training_pipeline3.policy_and_value_generations)
	base_rate2 = 0.01f0
	rate_adjust2 = 1f0
	@info "Prior to training, best performance is $best_performance from generation $best_generation"

	failure_count2 = 0
	gumbel_mcts_training_pipeline3.clear_data!()
	while failure_count2 < 3
		gumbel_mcts_training_pipeline3.train_networks!(base_rate2*rate_adjust2, base_rate2*rate_adjust2, 1_000_000, 40, 8; clear_buffer = false)
		gumbel_mcts_training_pipeline3.update_generation!()
		generation_performance2 = gumbel_mcts_training_pipeline3.policy_and_value_generations[end].performance
		@info "New generation performance is $generation_performance2"
		if generation_performance2 ≤ best_performance
			rate_adjust2 /= 2
			@info "New performance is worse than best performance of $best_performance from generation $best_generation.  Restoring best generation parameters.  Cutting learning rate adjustment in half to $rate_adjust2."
			gumbel_mcts_training_pipeline3.restore_generation!(best_generation)
			failure_count2 += 1
			@warn "Failed improvement $failure_count2 out of 3"
		else
			failure_count2 = 0
			@info "New best performance achieved on generation $(length(gumbel_mcts_training_pipeline3.policy_and_value_generations)).  Keeping parameters"
			best_performance = generation_performance2
			best_generation = length(gumbel_mcts_training_pipeline3.policy_and_value_generations)
		end
	end
	gumbel_mcts_training_pipeline3.policy_and_value_generations
end
  ╠═╡ =#

# ╔═╡ 12035440-3134-43d9-a970-cab2672aaf24
md"""
### Network Training

Using `setup_gumbel_mcts_training` we can accumulate new training data.  We would like to use this data to perform gradient updates on the policy and value network.  There is some rate of data consumption with mini-batch SGD training and a rate of new training examples generated.  Let's call these rates ``R_{SGD}`` and ``R_{MCTS}`` respectively.  We can define a ratio between the two ``\rho = \frac{R_{SGD}}{R_{MCTS}}``.  Based on open source implementations and some notes from deepmind, we would like ``\rho \approx 4 \text{ to } 8``.  At this ratio, each generated example will be used a handfull of times before getting replaced as long as we size the replay buffer appropriately.  

Let's work out a specific example with a target batch size of ``N``.  We'll define the buffer size in terms of a multiple of the batch size.  Now what happens if we start training with this setup?  After 1 unit of time we will have ``R_{MCTS}`` worth of data.  Let's say the buffer is of length ``M \times N``.  That means to fill up the buffer will take ``M \times N \div R_{MCTS}`` time.  What we really care about is how long on average something stays in the buffer.  If an example enters the buffer at time 0, then it will get pushed out after enough data has been generated to fill it up which is ``M \times N \div R_{MCTS}`` units of time.  How much data will be consumed in that time?  The ratio we have would suggest that in the time it takes to generate enough data to fill the buffer, SGD can consume the entire buffer ``\rho`` times.  That means that regardless of how big the buffer is, we'd expect any one example in the buffer to be used ``\rho`` times before being pushed out.  In the extreme case let's say the buffer is exactly the size of one minibatch.  Then during the time it takes to replace that minibatch, SGD can process that batch ``\rho`` times so if you imagine an example entering at the beginning of the batch and exiting at the end it will travel the length of the batch over four cycles of training.  If the buffer is much larger then both the travel time and the cycle time through the buffer will grow at the same multiple.  Every training batch should be sampled at random from the buffer.
"""

# ╔═╡ fd7a6996-d8f0-49d2-a65d-5280a306d0d6
md"""
After performing some number of SGD updates, we will publish a new network to the MCTS search function or update the parameters it is using.  Then the data being generated will be based on one network.  The buffer should contain a mixture of different networks so as not to overfit on the most recent one.

"""

# ╔═╡ 77977fd8-5f70-11f1-9a3f-7d549b180fa8
md"""
# Depedendencies
"""

# ╔═╡ a1810cd7-630a-4ece-9535-3d003e4662f9
html"""
	<style>
		main {
			margin: 0 auto;
			max-width: min(1600px, 90%);
			padding-left: max(10px, 5%);
			padding-right: max(200px, 5%);
			font-size: max(10px, min(24px, 2vw));
		}
	</style>
	"""

# ╔═╡ e7c35449-d409-4313-9442-06cef7bfce8d
md"""
# Visualization Tools
"""

# ╔═╡ 66b571d7-ef1d-441d-b64c-651147ec0380


# ╔═╡ 6d511d78-a24a-410c-a304-1a09c2781aab


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Revise = "295af30f-e4ad-537b-8983-00126c2a3abe"

[compat]
BenchmarkTools = "~1.8.0"
HypertextLiteral = "~1.0.0"
LaTeXStrings = "~1.4.0"
PlutoDevMacros = "~0.9.2"
PlutoPlotly = "~0.6.6"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.83"
ProgressLogging = "~0.1.6"
Revise = "~3.15.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "c7fe0f75738e334a17fea5a6ab1bc575fc8b9f3b"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "PrecompileTools", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "9670d3febc2b6da60a0ae57846ba74670290653f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.8.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "REPL", "UUIDs"]
git-tree-sha1 = "cfb7a2e89e245a9d5016b70323db412b3a7438d5"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "8e9c059d6857607253e837730dbf780b6b151acd"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.19.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.FlameGraphs]]
deps = ["AbstractTrees", "Colors", "FileIO", "FixedPointNumbers", "IndirectArrays", "LeftChildRightSiblingTrees", "Profile"]
git-tree-sha1 = "d9eee53657f6a13ee51120337f98684c9c702264"
uuid = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"
version = "0.2.10"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "58927c485919bf17ea308d9d82156de1adf4b006"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.12"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "b864cb409e8e445688bc478ef87c0afe4f6d1f8d"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoweredCodeUtils]]
deps = ["CodeTracking", "Compiler", "JuliaInterpreter"]
git-tree-sha1 = "3733419e9a71156b389f3e331672d2e95436783f"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.6.2"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "32a4e09c5f29402573d673901778a0e03b0807b9"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.6"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "6256ab3ee24ef079b3afa310593817e069925eeb"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.23"

    [deps.PlotlyBase.extensions]
    DataFramesExt = "DataFrames"
    DistributionsExt = "Distributions"
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyBase.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlutoDevMacros]]
deps = ["JuliaInterpreter", "Logging", "MacroTools", "Pkg", "TOML"]
git-tree-sha1 = "709c36a806ec0af91840184f3052bb3c6cc60915"
uuid = "a0499f29-c39b-4c5c-807c-88074221b949"
version = "0.9.2"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "2b9e3d771adfe535a4fdda855f4741fdaacd3f7f"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.6"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoProfile]]
deps = ["AbstractTrees", "FlameGraphs", "Profile", "ProfileCanvas"]
git-tree-sha1 = "154819e606ac4205dd1c7f247d7bda0bf4f215c4"
uuid = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
version = "0.4.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProfileCanvas]]
deps = ["FlameGraphs", "JSON", "Pkg", "Profile", "REPL"]
git-tree-sha1 = "41fd9086187b8643feda56b996eef7a3cc7f4699"
uuid = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
version = "0.1.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "f0803bc1171e455a04124affa9c21bba5ac4db32"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.6"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CRC32c", "CodeTracking", "FileWatching", "InteractiveUtils", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Preferences", "REPL", "UUIDs"]
git-tree-sha1 = "27e3ee13fc8739a59b380d6163d6a82f52c03bd7"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.15.1"

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

    [deps.Revise.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╟─657a72a3-933f-4e48-b357-7a73315fab65
# ╟─1519aac0-b013-49c8-a222-0d26c05d5195
# ╟─62c32e45-49bd-473c-b4d8-fc15a2dbaa17
# ╟─ed609953-cf65-45cb-ba89-c5a6058dd4af
# ╠═9e6c8d1d-2152-4b45-a2b6-38e48a4f0b8e
# ╟─819d1840-27a9-4cba-993a-e028584ed147
# ╟─7cde31ab-ee50-4553-b9d0-45141ddfc40d
# ╟─606741df-ff6e-4a00-b8b4-400de9cec9b3
# ╟─806ed2a1-0b2a-4958-9d10-5f3117b69f72
# ╟─030e5494-4dcd-4456-8b7f-9335b1e148c5
# ╠═2cd49671-b297-4d4a-8d5a-8c8937372088
# ╠═a46fad01-1453-46ca-ab09-448ded3b8aae
# ╠═8ae4306e-1c38-4a76-81c4-741a00b9a1df
# ╠═f0c23fbc-112b-4a47-8af2-e166dea16627
# ╠═e55fc618-5319-4059-9fee-6733329855f1
# ╠═2aed3faa-2e28-4400-b71d-d8b3ac5fa553
# ╠═482d4aaa-f3bc-41df-adbb-23e588abdcd2
# ╠═a320f525-ca01-4bb8-a872-8c563107fad7
# ╟─3ece2292-f3dc-4d9b-9b24-6badb8246862
# ╠═4b116755-9c16-43a9-932d-336c206777c6
# ╠═2b89ab81-f8c5-458c-8a5e-fadd5418d8dd
# ╠═6b694195-dba4-411b-a600-add1c2d1bed9
# ╟─607f0859-8a67-45f7-9e6d-e407d0febf26
# ╠═3d1cf9bb-1b75-4070-bc3e-766a4da7ffba
# ╠═36d2c036-0167-44a2-b9ec-343f831c1893
# ╠═14ef1213-3a2c-48c5-899d-986dfad9fef1
# ╠═8fcc9a0c-8ed5-4cc7-a907-1ff41c929f23
# ╠═90944844-7709-4547-ba6d-2821082648f6
# ╠═9fdc2886-b390-42be-965a-227c56a0d629
# ╟─f1df3bd5-0019-4f6b-8073-42362048ed8a
# ╟─f0eb4b7b-5ae9-450b-973c-495ebe09a3e2
# ╠═0f3e8379-22bc-4aad-95d0-7741b9d1ae03
# ╠═dfb0c194-322b-445e-aace-409d7b80e99e
# ╟─268840e0-fefd-4d8a-a53b-3fabd9e70548
# ╠═b4cd4c2c-6f27-499d-9510-27ac1b4b36b1
# ╠═2a46b458-9f23-4659-9d00-bba1e90a5b74
# ╟─59978a88-92d6-4964-a580-41407144959d
# ╠═4ccdca95-1e86-4877-91a4-f328aefb2915
# ╠═ee9d565d-8d26-4768-bfbc-938c145f887b
# ╟─76d3fa3b-a229-4800-9dd2-38e80346ae47
# ╟─40e3da31-3fd2-446c-9dc9-bbec430e794b
# ╠═72ea5a15-2b07-4c0d-80aa-802f50bc433a
# ╠═bd4ba7de-9e3b-4b09-bad8-d20cce0c1c7a
# ╠═49c0564a-9a29-4c2d-8559-7b8944fa5219
# ╠═684b1c27-2c72-41ba-841a-ec08b5bae3c4
# ╠═b4a16aa8-2411-4e26-a66a-f6fd1375430d
# ╠═4f9fca26-f1b2-4fbc-954d-1d32211edb56
# ╟─3924aab4-8d85-422a-892e-0f3950ffec3a
# ╟─25e97cff-adaa-41e8-864f-9d9d55069fa3
# ╠═bf2cbd82-8bd6-41d5-b9d1-0de6f1521565
# ╟─2cf6d815-2099-473f-b4c3-99c712541787
# ╠═9da644b0-c096-4fc9-a8e3-7575a32fb670
# ╟─c6a1d201-6f71-4a09-8cf5-8ca1f058ee7d
# ╠═0391c2b3-33f4-4ffa-bd09-ae381a3fd587
# ╠═74154178-6c1f-444e-bf23-d28bde19cb2c
# ╠═1146218c-da98-4012-9d51-e14c223b929e
# ╠═ae69ab54-dfb1-4a8b-8e3e-cfb0281a99bb
# ╟─179a2fce-e367-4db2-a563-604aadb00122
# ╠═0e237d91-ece1-485f-aedf-c29187955315
# ╠═16de0d9b-c411-490f-99c3-c70c6e874741
# ╟─cfd60a54-6f93-453d-97c5-75574702a47e
# ╠═68b919cd-51b1-4450-b6a5-a8f1d96248bf
# ╠═d0e04b83-9b6a-4653-afd4-2f2914dc067f
# ╠═a8bd57dc-239e-41bd-b1c4-a467a65da4d4
# ╠═5c23b887-cd6d-4913-9704-c3552e8deb2b
# ╠═36056f87-3a19-49cb-b322-c44c3655e59e
# ╠═a37ff490-f724-413e-957e-9e137b090019
# ╠═72fcf8f4-dab0-4f59-ad08-dbefd94157a8
# ╠═2a3100f8-8f44-4d63-97da-2eebd2e0a8dc
# ╠═94a708a1-7621-4390-8852-a37d65b4e0da
# ╟─c16dfc51-44e7-45c0-a85d-831b713f0e66
# ╟─5c04c776-015b-4b51-ac68-c4e359fb323d
# ╟─8c3565c6-09f6-4b4e-9e17-6d10fb637e77
# ╠═a575b693-fae2-4c3c-b9d6-b125f2773530
# ╠═48199c60-dc5b-4131-bd7d-43b75c0b6c1c
# ╠═f86fd6b6-1438-4d4b-8566-93a6dcd8e62b
# ╠═5ed3ef2c-729e-4aa4-8c6c-12b8ccffa102
# ╠═883853ff-d4fb-4ad1-b610-2d8a86f571d8
# ╟─888a50ac-ef39-4705-b3c3-16c252ab7b40
# ╟─90ce5f4d-5104-4011-8164-f8c95077109c
# ╟─8e57a6c4-3d87-4ad0-a42d-619d9f96cf83
# ╟─87f9eb9b-d542-4ba6-9ba4-9590138a49a7
# ╟─999f90e7-a87a-4edf-9e81-c4f8493f1c69
# ╟─7e0c91a5-c1f0-4ea1-b450-52daca234cde
# ╟─3cff602c-a889-40f1-be74-74be117d3b0b
# ╟─9ff655da-7b92-4797-b953-d2413feb4aed
# ╠═49a67540-f964-4209-a6c3-07dd66e83d84
# ╠═17fe1925-5e87-4abf-ba34-dff183df9638
# ╠═5735640d-c8a6-4753-8425-8cdb1e9a4bf5
# ╟─7f160dd5-654a-4bd9-abd8-62ef5baddaa8
# ╠═faa837bc-3c78-4198-b34a-6deddc872959
# ╠═08616411-c11d-4ac1-aab3-07a21d25a30d
# ╟─f7633ae5-e036-428c-a85f-4352d400a91a
# ╟─0732370c-785f-4f4c-9d31-94d985fccc71
# ╟─d1060ffe-0ea7-4e6c-802c-3073c59f665f
# ╟─868602f2-1e57-4a9f-93ed-a73e7c461b99
# ╠═851ac26a-4f49-423a-9558-77d081b7791b
# ╟─0edf174b-98c1-4328-a5b9-d2d82a2b99b9
# ╟─aa79a188-8055-4ae0-8e3d-82632bb04b10
# ╟─d508aae1-3c69-403f-b398-cd4481657215
# ╟─3e645d32-26e2-4da8-a7fc-dfa1ca08d7e3
# ╠═a17bdb13-472e-4f6a-ac56-3f17ffc1154b
# ╠═2e9c422f-adc4-4762-8ef4-ae95cdc39cce
# ╟─ef531570-6180-46ab-981d-1a022a787373
# ╟─8c958b3a-c7e5-4d8c-b0e3-16d1987baafa
# ╠═a6ce047a-d577-46b0-afda-72076c1d96a3
# ╠═f76a0cc6-9265-4f04-aecb-da3593bb9303
# ╟─d05a1e26-f234-43d8-8503-88162d963bdd
# ╠═90017d35-c18a-4914-8304-bfac94161733
# ╟─49980b44-ba9d-4f53-b0a0-31553d57c642
# ╟─ac04aa56-6c96-4f10-b798-62d9249a20fb
# ╟─1b830b45-6800-49ce-ad25-ff20c0fef9b1
# ╟─3693c086-4e76-4468-ba48-14725587d4d3
# ╟─a1d02958-c9cd-464e-8668-196cab28e108
# ╟─6d0e00ab-33a9-4db5-987c-e5cbdd58f521
# ╠═63d77923-4fa9-4263-ba1b-6c5d5d219a2c
# ╠═4d0682ae-c438-475c-9825-6a0c067b8b50
# ╟─37934b7f-5e5f-4f6b-8ed9-f4218a0f4e1e
# ╟─d5d54013-1c75-41a6-8f7e-b1721086c7df
# ╟─b919b834-d65f-43b9-8ad8-0b494b4c4a0b
# ╠═1148e63c-a555-493b-aefb-ded46e748b49
# ╠═e3bb487d-c59a-4764-b926-0ac7672c5c5e
# ╟─7eca6f17-2124-4e20-833f-3e4d3c81ec4e
# ╠═6e78bd45-2659-4c2c-a1ce-f62b9c1eef69
# ╠═b1a8bfc9-73a7-4ce1-b3c9-c5b4263d284f
# ╠═e098c17d-97c4-4f89-bce4-d9cd90ee09b8
# ╟─eaebd5ca-0669-4b96-b9ed-f28247113d30
# ╟─cefe7851-b3ad-4e13-a8c7-74848ae42e66
# ╠═220d4cef-141d-4a29-9f83-c48a25886131
# ╠═bab309d1-ade0-43f4-af2e-ecd9eff0dda7
# ╠═51b60ecc-4609-4dd5-8537-a0b38443e00d
# ╟─9dbed060-b0b4-4fe1-99c8-c9356637ccda
# ╟─7f2b59df-0022-4f7c-a769-75568e8aa607
# ╟─3bc2a8e3-4a26-4588-9c2b-d2f855cb211c
# ╟─64ac08c1-7a4b-4645-97f4-54bb2e4c3ba1
# ╟─d580e150-7f6d-4b63-bd24-ac39d9fe7709
# ╟─a288fdce-5f46-42f7-a826-fbdc37db8094
# ╟─fa92cac5-71dd-4b04-99ed-2256cf50480d
# ╟─fc0f7cb6-792c-4dcc-bc3b-482593f9838f
# ╠═2b9b64a2-69f4-4083-b7a6-13325bd1e672
# ╟─481e6e9c-63f0-4d81-9abb-305e692f5121
# ╟─1e874fe5-3c7e-4807-ae50-cb316e3bcd50
# ╟─0beed6bc-1d6a-4499-9315-842e74ad4ff8
# ╠═6ed31c1e-ba49-43c5-b67e-3b03a6f04301
# ╟─69087af9-d37d-4736-ba9a-344d8cbdec12
# ╟─d61c6352-9156-47db-824f-9d5f5e8cc827
# ╠═b3818532-950b-49f7-8f6e-91b785e856b4
# ╠═58673ed3-7098-4745-a800-6b813cc40dd7
# ╟─23fadce5-7392-4916-8609-ad7aacad2306
# ╠═91ed1cb2-ae03-4c56-a361-aa9434ab6352
# ╟─1a32d24d-7781-4774-b530-0379dd02f5d5
# ╠═b46b076b-5177-4595-99de-931aa81ae9a3
# ╟─1ade37fa-e033-4134-b2f2-bc0ab535b90c
# ╠═64cada0a-b9d1-4aa9-bb1c-0e697d11ffd0
# ╠═d2fbb761-6bf6-46db-b182-062ec9c10d8f
# ╠═eb378857-c047-456f-bb8c-dddbb6b64882
# ╟─c8586d76-7923-4e13-918f-36e6e4e97ace
# ╠═0260c420-dd3d-48fa-8ea9-287387a51d16
# ╟─efeba91e-522a-4c17-a50f-d1f610981139
# ╟─1e6595ef-aa9f-4084-8976-64605aeee54e
# ╟─80e82c6b-4b6f-46ca-9162-a7c12aa9fc03
# ╟─20dcb4c0-75e8-44ac-b1be-a27d5d4a7660
# ╠═8c5e4b7f-a1b3-4d54-9879-04aee828561a
# ╠═5f129b73-a4c9-476b-a162-64c0b1cedc5b
# ╟─aeff1449-b8ad-4a36-b96f-5c90fc11178f
# ╟─34b4ec27-5e37-429f-8873-c1bba4c20130
# ╟─9d623aba-7eaa-44a0-9617-f869cf423d5e
# ╟─7e91919a-6d1f-4d02-a095-a30186c8362a
# ╠═e4d8bf10-9214-4a9a-be18-f88cd9153f8b
# ╠═6297a0ff-c277-414b-a2b8-020c11da17e6
# ╠═43f6f078-30f2-4951-b11c-95adb1dabafc
# ╠═fd290ce1-8cca-4a68-9a53-a0ecc9b95021
# ╠═9148e4fb-e1ad-4d49-9211-ee874965f879
# ╠═f22a726c-d334-460f-a961-05cb486cd09e
# ╟─4f1ee074-4f64-4088-b329-b2920489fb82
# ╟─850637e9-c707-4dd5-b4e8-fc5c67bebe26
# ╠═90b11aa5-0136-46ad-9fdd-3d043e028f8c
# ╠═0c2ea530-bbe6-4b14-a202-1a9fe91405bd
# ╠═935afffb-a305-462f-b92c-8a7048ee125f
# ╠═640b689b-af31-4309-baa0-4004d40a669d
# ╠═26f5c1e0-5162-42ee-9aaf-61a8c5682b1b
# ╠═f802a3bd-42b8-4012-b06b-b7319cbb189d
# ╠═7dbcc40d-c107-4322-bb5c-7f66ef40dc5b
# ╠═c7cd253f-8852-4f6e-9ea9-2baedc53b4d5
# ╠═2e1b189b-f5d7-429b-8468-04f57dcfb06a
# ╟─02e1ea17-74ed-4523-bde7-fe48d9ea1f1d
# ╟─a07c2d69-227a-42ca-87c6-35d19f955e77
# ╟─cf57bed0-ed83-48ed-9ecc-6d5740f7fbd7
# ╠═aabaa31b-7204-40f7-a3a8-2fd0f0e666ec
# ╠═7155e745-d3c2-466c-9c0d-aef997f0901f
# ╠═8e0a58d4-ba9d-4b72-97ec-060fd4bd4767
# ╟─334e5d84-6b1a-4427-8ecc-cecd858625b2
# ╠═0d676e61-5ec8-4fda-a748-dfb9a2abf5a8
# ╟─26ea9090-af3a-4bf8-946d-14896a54a020
# ╠═39f265e4-037e-42ab-b09c-3079f463f496
# ╟─9a48ba14-f71f-4c60-907d-266df8a3ffdb
# ╠═0d3307b3-c50e-4d6c-a4b8-1cfd79ed0d27
# ╟─75e5f820-1a7d-42c5-bd23-cf9b94faf727
# ╠═0b22c86d-561a-4839-a088-2a097da28df6
# ╟─30b603f2-5181-48f3-a1fc-59297b0cb7fc
# ╠═88501e87-6bed-4c22-9524-24b46e1dc477
# ╟─3363b2bd-7a38-42db-8225-da04fa1d8cb1
# ╟─91f5ca97-a747-483f-8697-1b1872fba236
# ╠═7a00fa12-f66c-4e83-a1e7-e0aa51c211b7
# ╟─53837f4c-c246-41a0-a917-bae50376dc0f
# ╠═e1bd21eb-6d43-4d0a-8cf2-b86d18f3eab2
# ╟─b6f43053-1309-4139-b296-a46e8e864f34
# ╠═699232e8-d39c-4623-9d23-6527adbc9baf
# ╠═d5986e30-4b1f-42c2-9b96-05a024b513af
# ╟─8d422c7e-5784-4a86-b832-9a37af4bcc08
# ╟─5d3e6956-3aae-4202-ae17-9dbb5f3f789e
# ╟─93f009f0-a348-47de-b9e0-dd2e29f50973
# ╟─3f7e7968-9b7b-47bd-a3d4-3f06be6dc573
# ╠═49afc4a2-3168-4926-99b1-e363295f3bd5
# ╠═f78026ff-cf97-4d20-a705-8c82cc9f0d21
# ╠═0c3aa32d-1968-4338-b29c-51f3ab892f86
# ╟─aede5c00-628b-4b06-b086-9875ecd2d9ea
# ╠═ce9a53ea-9642-47df-b84d-4e5ce5bf123a
# ╠═a1d18f21-000e-4ee8-8296-0304ccae5bdb
# ╠═0e27a2bc-41ed-4575-8952-20ca904a92d5
# ╠═2e7fecbc-1952-44bf-97ad-b050cdfce8be
# ╠═a9d34520-f805-4548-8d71-19797cc519e1
# ╟─3d553190-c829-4b65-bc83-c4cc58df4330
# ╠═6175acee-1310-471a-8258-08e2ea4f4ff1
# ╠═aa2ec6b2-863e-4c44-b95e-d8582d0e20c2
# ╠═d6598ee5-92e7-4d24-a998-78b9b0da5da0
# ╠═999d8928-12b8-4038-b6b6-e7087e44a356
# ╠═218862a6-0625-4650-920d-6b4424c2e235
# ╠═96611207-eda1-4153-ae3e-799b198e5b93
# ╠═6c9ee058-8a16-4332-b40a-112b30e30a59
# ╟─9e375988-454e-463e-9d56-439c9855ffcc
# ╠═e8b27cc5-25b6-4323-b0da-3a0d9990a717
# ╠═36c416f5-2304-466b-b507-28d58962a8b0
# ╠═80d6264f-1def-47f2-8cfa-913b88898da3
# ╟─29033440-f242-44d7-9190-5506e0653d0a
# ╠═28c04d73-d6e0-4167-81c6-186005b3f163
# ╠═d9199030-4562-4807-b3ef-ce0df0fc16d2
# ╠═635bee38-38df-4db9-b40c-418c3b451a5e
# ╠═e1657a7a-d4a5-4573-b7cb-fe3b0e1159c0
# ╠═a4b7488b-85d2-4b93-b218-d0490b37170b
# ╟─9764c645-c07c-4c43-8231-56791426e241
# ╟─5d0c5922-1d4c-4ab0-bb74-91cf39b9bf2b
# ╟─41978938-c540-4778-9177-12d4ef94bf9f
# ╠═c1f1052c-1111-4453-bd19-72add618e489
# ╟─d573ebc9-2dc0-4d66-980c-580d3fcd5751
# ╠═ad3b033b-06b4-42b1-afe9-cd6481889661
# ╟─277336f2-3ce8-4c63-b2fe-fd4b36e8e353
# ╠═556cb9b8-1a57-49f0-a586-3d7541e7e839
# ╠═037e9bbe-4b1e-494a-a9a9-7b4785421731
# ╟─b26f33c0-2a90-4405-b76c-8d53e257c62f
# ╠═29b8619b-ae26-4145-9008-1c97893d1307
# ╟─9db64ad9-e2e1-4520-b29f-7acfb3f3ef4a
# ╠═95a1544f-75d4-44ab-b818-2b32ec7cfe71
# ╟─6f3b990c-0f48-4ca2-8ae8-22171b5b4943
# ╟─b31dcfa0-39af-4f14-b224-12c0da7217f7
# ╠═b1f8f9b3-4d02-4373-bbda-c22c23d911d7
# ╠═5225bb20-b634-47b2-aa7b-317f3e4ce799
# ╟─03e274c9-2c75-440a-bec5-7d2c208e059e
# ╠═9dec854c-5abb-48c1-8170-e879b45dee47
# ╠═af9a0734-e0c0-4553-bfdd-24f89e7ab8ec
# ╠═1b27a559-6cc8-4659-8ec9-719b15dc5335
# ╟─e446f5b2-068e-4c27-aacc-dcd99362a70b
# ╠═ff6930d8-cf7b-453a-b25e-93e665dabfd0
# ╠═33237e06-e784-47c8-8d45-8de457d46e8c
# ╠═021d04c1-245a-42c4-a193-2ec0d1e68094
# ╟─f36dc75d-a14b-4e3e-b360-583455f6ec65
# ╠═15a719e6-e399-469c-9aee-69a8bfcec175
# ╟─04b2a9de-6556-4389-993b-4b85aa16528e
# ╟─a24b8edd-8fcf-4d70-84c8-925463169230
# ╟─c4ba3f81-8189-4059-8652-7edb0834c215
# ╠═6ee14f48-b46c-46cd-86a6-140e5f969a75
# ╠═59ec3eed-559f-4243-92f8-d893768511be
# ╠═c472e9b0-14a6-46be-8914-08ee560fac1c
# ╠═475dad7f-227f-4d94-ba85-ec654fa23bbd
# ╠═46bac81b-29ac-4594-9459-11710e5b6e02
# ╠═6d4a7252-d23a-43c7-878b-ac6eb1c8e9a6
# ╠═67453114-9ce0-424e-82a4-ff9069bfcc6b
# ╠═3ec0e848-d1ef-4f5c-b712-507e7c6f83c8
# ╠═63ba2858-b49d-4867-8a9e-0c16816df168
# ╠═e7ffd232-e520-467d-832d-9cdc898af79c
# ╠═38f75388-7d9f-4538-b173-38d1f303d569
# ╠═b51d007d-8135-4741-ab81-c00f8ee9a746
# ╠═630e7012-d5ad-43be-a671-1582e532804e
# ╠═a7c34d1a-169f-4189-99fd-e3cb6de744b5
# ╠═a366e369-21b5-45a5-903e-59ace7c61c3b
# ╟─e8961a70-08e1-410e-a60c-05c6c963482d
# ╠═55127238-6d1d-41d0-9e4a-9c0ac150a50b
# ╠═4648697d-c834-4a57-9e0e-dbe4f1757e35
# ╠═3669289b-4baa-4aa5-ad2b-4242703ffa0c
# ╠═a268f2e4-51c0-452a-bec3-146a9028f25b
# ╟─d7ea540e-c134-4389-b448-7186a1e6eb7f
# ╠═d29a25db-536d-4231-b56e-9bb28470d0c8
# ╟─095cb92d-9a8b-45b1-bdf2-8614411a1398
# ╠═b37c8a17-faaa-4786-a520-059a224eb6dc
# ╠═e507d605-ce14-4dee-987b-ad1668534ac7
# ╠═99dd5ff5-75b3-41ec-88cd-b907159812d4
# ╠═ed5d12e5-a277-4a6d-8048-d5f18bc6f23c
# ╠═38874107-272e-4bfc-b502-7aa84fd010f4
# ╠═41b1f97b-f847-4578-a762-1c8b7ccb3fc4
# ╠═5f4cd84b-5d43-4f37-802c-840f0c9a6b36
# ╠═9068f9d7-6006-490f-a53e-a1fea309e2c2
# ╠═7d06c211-96ee-4af3-a9c9-e666988a155a
# ╠═300799fa-b47e-432d-830e-0c5e822a9b02
# ╠═c1df080c-15e5-41cd-9553-29bf20cacce7
# ╠═7249110d-8b63-46a4-8772-642ddade0eb9
# ╠═b97ef521-903c-4493-af2d-47781be40530
# ╠═d99a0f83-3927-4bc4-b60c-08ee7224e6c6
# ╠═8dbf7751-6e07-417b-9359-e0510e5aed18
# ╠═5f275280-55bd-4784-98d5-a2cd83bba977
# ╟─47f2b03e-675d-4294-95ca-fc368e6181cb
# ╠═c9834a8b-fbf8-4786-b589-ac90cb03cb28
# ╠═b5fdce32-079e-47a8-acb5-f891e99f3d7a
# ╠═83798641-6218-480e-beed-183e676d1f08
# ╟─88930b3e-d977-46db-a27f-9c3bcebeb2c9
# ╠═75e80bf5-3562-460c-8157-f46c28829d8f
# ╠═77c6e7f9-aa96-4a44-9620-f403847c5120
# ╠═1091b9bf-5063-4f2f-83f1-0cea199c68ac
# ╠═a6d4fb64-8e55-4551-9cba-547f7cc8e3d5
# ╠═ca823968-ee51-45a6-9d52-a45c7586a5d0
# ╠═2b497bc2-f01b-4274-90f5-f5077c15913f
# ╠═5d1c1e17-51d3-430b-9bd0-077d6d32c21e
# ╠═8e700fca-3563-4cc6-a92a-dce1705b6dc8
# ╠═4acd4499-eee5-4bdf-a32d-0d41172eedb8
# ╠═065f9eaf-d180-4e6f-ab17-7e710a383ef7
# ╠═f1b3e26c-5ac5-48ea-9e82-e4d4fa84c8e4
# ╟─76eefb47-cf41-414c-a187-756f5a8e6dbf
# ╠═1e79413b-b509-4adb-927c-5a99d931043b
# ╠═8229ce52-eb6c-4810-a223-e9706a9ffa20
# ╟─12035440-3134-43d9-a970-cab2672aaf24
# ╟─fd7a6996-d8f0-49d2-a65d-5280a306d0d6
# ╟─77977fd8-5f70-11f1-9a3f-7d549b180fa8
# ╠═18ed439c-09b0-4039-bdb7-12b47e25f82d
# ╠═22ccdf79-e8a7-4233-88b0-5678797d5cb8
# ╠═7035e568-65b9-4e90-8a7c-fb7354fb7aec
# ╠═662669a4-243d-4d9d-98c4-d2acac04358c
# ╠═a1810cd7-630a-4ece-9535-3d003e4662f9
# ╟─e7c35449-d409-4313-9442-06cef7bfce8d
# ╠═66b571d7-ef1d-441d-b64c-651147ec0380
# ╠═6d511d78-a24a-410c-a304-1a09c2781aab
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002