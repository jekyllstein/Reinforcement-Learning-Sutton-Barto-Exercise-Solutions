# GridCapture MDP Wrapper for TabularRL
# This provides a StateMDP interface for GridCapture

using ReinforcementLearning: StateMDP, StateMDPTransitionSampler
include("GridCapture.jl")
using .GridCapture

# ============================================
# ACTION TYPE
# ============================================

"""
    GridCaptureAction

Tuple type representing a move on the GridCapture board.
(row, col) where 1 <= row <= N and 1 <= col <= N
"""
const GridCaptureAction = Tuple{Int64, Int64}

# ============================================
# MDP CREATION
# ============================================

"""
    make_gridcapture_mdp(N, K, opponent_level; agent_player=:x)

Create a GridCapture MDP with the specified opponent strategy.

Args:
    N: Board size (N x N)
    K: Win condition (K-in-a-row)
    opponent_level: Symbol indicating opponent difficulty
    agent_player: :x or :o indicating which player the agent is

Returns:
    StateMDP for use with TabularRL algorithms
"""
function make_gridcapture_mdp(
    N::Int, K::Int, opponent_level::Symbol; agent_player=:x
)
    opponent = make_opponent(opponent_level, K)
    
    # Create list of all possible actions (all positions on board)
    # Flatten the matrix to a vector
    all_actions = vec(GridCaptureAction[(r, c) for r in 1:N, c in 1:N])
    
    # Determine initial state function based on agent player
    if agent_player == :x
        # Agent plays first as X
        state_init = () -> GameState(N, K)
    else
        # Agent plays second as O, opponent moves first
        state_init = () -> begin
            s = GameState(N, K)
            opp_move, _ = opponent(s)
            place_stone(s, opp_move[1], opp_move[2])
        end
    end
    
    # isterm function: check if game is over
    isterm(s::GameState) = check_game_result(s) != :ongoing
    
    # Step function: takes state and action index, returns (reward, new_state)
    function step(s::GameState, i_a::Int)
        action = all_actions[i_a]
        
        # Check if state is terminal
        if isterm(s)
            return (0.0f0, s)
        end
        
        # Make agent's move
        new_state = place_stone(s, action[1], action[2])
        
        # Check if agent's move resulted in a win
        result = check_game_result(new_state)
        if result == :x_win
            return (1.0f0, new_state)
        elseif result == :o_win
            return (-1.0f0, new_state)
        end
        
        # Make opponent's move (samples from their distribution)
        opp_move, _ = opponent(new_state)
        final_state = place_stone(new_state, opp_move[1], opp_move[2])
        
        # Check final result
        final_result = check_game_result(final_state)
        if final_result == :x_win
            return (1.0f0, final_state)
        elseif final_result == :o_win
            return (-1.0f0, final_state)
        else
            return (0.0f0, final_state)
        end
    end
    
    # Create transition distribution from step function
    transition = StateMDPTransitionSampler(step, state_init())
    
    # is_valid_action: check if action is valid for current state
    is_valid_action(s::GameState, i_a::Int) = is_valid_move(s, all_actions[i_a]...)
    
    # Create StateMDP with action_index lookup table
    action_index = Dict(action => i for (i, action) in enumerate(all_actions))
    return StateMDP(all_actions, transition, state_init, isterm; 
                    is_valid_action=is_valid_action, action_index=action_index)
end

"""
    make_gridcapture_mdp(N, K, opponent_func; agent_player=:x)

Create a GridCapture MDP with a custom opponent function.

Args:
    N: Board size (N x N)
    K: Win condition (K-in-a-row)
    opponent_func: Custom opponent function that takes state and returns (move, score)
    agent_player: :x or :o indicating which player the agent is

Returns:
    StateMDP for use with TabularRL algorithms
"""
function make_gridcapture_mdp(
    N::Int, K::Int, opponent_func::Function; agent_player=:x
)
    all_actions = vec(GridCaptureAction[(r, c) for r in 1:N, c in 1:N])
    
    if agent_player == :x
        state_init = () -> GameState(N, K)
    else
        state_init = () -> begin
            s = GameState(N, K)
            opp_move, _ = opponent_func(s)
            place_stone(s, opp_move[1], opp_move[2])
        end
    end
    
    isterm(s::GameState) = check_game_result(s) != :ongoing
    
    function step(s::GameState, i_a::Int)
        action = all_actions[i_a]
        
        if isterm(s)
            return (0.0f0, s)
        end
        
        new_state = place_stone(s, action[1], action[2])
        
        result = check_game_result(new_state)
        if result == :x_win
            return (1.0f0, new_state)
        elseif result == :o_win
            return (-1.0f0, new_state)
        end
        
        opp_move, _ = opponent_func(new_state)
        final_state = place_stone(new_state, opp_move[1], opp_move[2])
        
        final_result = check_game_result(final_state)
        if final_result == :x_win
            return (1.0f0, final_state)
        elseif final_result == :o_win
            return (-1.0f0, final_state)
        else
            return (0.0f0, final_state)
        end
    end
    
    # is_valid_action: check if action is valid for current state
    is_valid_action(s::GameState, i_a::Int) = is_valid_move(s, all_actions[i_a]...)
    
    transition = StateMDPTransitionSampler(step, state_init())
    action_index = Dict(action => i for (i, action) in enumerate(all_actions))
    
    return StateMDP(all_actions, transition, state_init, isterm; 
                    is_valid_action=is_valid_action, action_index=action_index)
end
