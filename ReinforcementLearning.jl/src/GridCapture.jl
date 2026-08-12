module GridCapture

using Random: rand, shuffle
using Statistics: mean
import ..TabularRL: sample_action

# Counter for generating unique board IDs
global _board_id_counter = 0

# Function to get next unique board ID
function next_board_id()
    global _board_id_counter += 1
    return _board_id_counter
end

export GameState, place_stone, player_turn, check_win, check_win_at, check_game_result, game_over,
       is_valid_move, available_moves, to_canonical, apply_symmetry, make_opponent,
       print_board, html_board,
       count_stones, total_moves, action_index_to_move,
       # Policy distribution functions
       random_policy_distribution, random_policy,
       greedy_policy_distribution, greedy_policy,
       defensive_policy_distribution, defensive_policy,
       positional_policy_distribution, positional_policy

# ============================================
# GAME STATE
# ============================================

"""
    GameState{N,K}

Represents the state of a K-in-a-row game on an N×N board.

Fields:
- `x_pieces`: Bit matrix where true = X stone
- `o_pieces`: Bit matrix where true = O stone  
- `K`: Number of consecutive stones needed to win (baked into type)
- `last_move`: The (row, col) of the most recent stone placed, or (0,0) if none
"""
struct GameState{N,K}
    x_pieces::BitMatrix
    o_pieces::BitMatrix
    last_move::Tuple{Int,Int}
    player_turn::Int
end

# Create new empty game with no last move
GameState{N, K} where {N, K} = GameState{N,K}(falses(N,N), falses(N,N), (0, 0), 1)

# Necessary functions for dictionary lookup to work with GameState as keys
Base.isequal(b1::GameState, b2::GameState) = false
Base.isequal(b1::GameState{N, K}, b2::GameState{N, K}) where {N, K} = (isequal(b1.x_pieces, b2.x_pieces) && isequal(b1.o_pieces, b2.o_pieces))

Base.hash(b::GameState, h::UInt) = hash(b.x_pieces, h) + hash(b.o_pieces, h) + hash(b.last_move, h)

# ============================================
# MOVE FUNCTIONS
# ============================================

"""
    player_turn(state::GameState)

Returns an integer representing whose turn it is: 1 for X or 2 for O.
Uses bit operations for efficient validation.
"""
function player_turn(state::GameState{N,K}) where {N,K}
    # Validate: X and O should not overlap using efficient bitwise AND
    # any(x .& y) checks if any position has both X and O pieces
    @inbounds for (x, o) in zip(state.x_pieces.chunks, state.o_pieces.chunks)
        if (x & o) != 0
            error("Invalid game state: Pieces overlap")
        end
    end
    
    # Use count(identity, ...) for BitMatrix - efficient without allocations
    x_count = count(identity, state.x_pieces)
    o_count = count(identity, state.o_pieces)
    
    # Check valid turn sequence
    if x_count == o_count
        return 1  # X's turn
    elseif x_count == o_count + 1
        return 2  # O's turn
    else
        error("Invalid game state: piece counts inconsistent (X=$x_count, O=$o_count)")
    end
end

"""
    is_valid_move(state, row, col)

Check if placing a stone at (row, col) is valid.
"""
function is_valid_move(state::GameState{N,K}, row::Int, col::Int) where {N,K}
    # Check bounds
    row < 1 || row > N && return false
    col < 1 || col > N && return false
    # Check if cell is empty
    return !state.x_pieces[row, col] && !state.o_pieces[row, col]
end

function available_moves(state::GameState{N,K}) where {N,K}
    moves = Tuple{Int, Int}[]
    for r in 1:N, c in 1:N
        if is_valid_move(state, r, c)
            push!(moves, (r, c))
        end
    end
    return moves
end

"""
    place_stone(state::GameState{N,K}, row::Int, col::Int)

Place a stone for the current player and return the new state.
Errors if the move is invalid.
"""
function place_stone(state::GameState{N,K}, row::Int, col::Int) where {N,K}
    # Validate move
    if !is_valid_move(state, row, col)
        error("Invalid move at ($row, $col)")
    end

    player = state.player_turn
    
    new_x = copy(state.x_pieces)
    new_o = copy(state.o_pieces)
    
    if player == 1
        new_x[row, col] = true
        new_player = 2
    else
        new_o[row, col] = true
        new_player = 1
    end
    
    return GameState{N,K}(new_x, new_o, (row, col), new_player)
end

# ============================================
# WIN DETECTION
# ============================================

"""
    check_win(board, K)

Check if the given board has K-in-a-row (horizontal, vertical, or diagonal).
Returns true if there's a win, false otherwise.
Works for any board size and K value.
"""
function check_win(board::AbstractMatrix, K::Int)
    n_rows, n_cols = size(board)
    
    # Need at least K rows and columns to have a line of length K
    (K > n_rows && K > n_cols) && return false
    
    # Check all starting positions
    for r in 1:n_rows, c in 1:n_cols
        # Only start if this cell has a stone
        !board[r,c] && continue
        
        # Check horizontal (right)
        if c + K - 1 <= n_cols
            won = true
            for i in 1:K
                if !board[r, c+i-1]
                    won = false
                    break
                end
            end
            won && return true
        end
        
        # Check vertical (down)
        if r + K - 1 <= n_rows
            won = true
            for i in 1:K
                if !board[r+i-1, c]
                    won = false
                    break
                end
            end
            won && return true
        end
        
        # Check diagonal down-right
        if r + K - 1 <= n_rows && c + K - 1 <= n_cols
            won = true
            for i in 1:K
                if !board[r+i-1, c+i-1]
                    won = false
                    break
                end
            end
            won && return true
        end
        
        # Check diagonal down-left
        if r + K - 1 <= n_rows && c - K + 1 >= 1
            won = true
            for i in 1:K
                if !board[r+i-1, c-i+1]
                    won = false
                    break
                end
            end
            won && return true
        end
    end
    
    return false
end

"""
    check_win_at(board, row, col, K)

Check if placing at (row, col) creates K-in-a-row.
Only checks the 4 directional lines passing through (row, col).
O(K) instead of O(N²·K) — much faster, no board copy needed.
"""
function check_win_at(board::AbstractMatrix, row::Int, col::Int, K::Int)
    n_rows, n_cols = size(board)
    
    # Directions: (dr, dc) pairs for horizontal, vertical, main diag, anti-diag
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for (dr, dc) in directions
        count = 1  # Count the stone at (row, col) itself
        
        # Count in positive direction
        r, c = row + dr, col + dc
        while 1 <= r <= n_rows && 1 <= c <= n_cols && board[r, c]
            count += 1
            r += dr; c += dc
        end
        
        # Count in negative direction
        r, c = row - dr, col - dc
        while 1 <= r <= n_rows && 1 <= c <= n_cols && board[r, c]
            count += 1
            r -= dr; c -= dc
        end
        
        if count >= K
            return true
        end
    end
    
    return false
end

# For GameState - uses last_move for incremental check when possible
function check_win(state::GameState{N,K}) where {N,K}
    row, col = state.last_move
    if row > 0 && col > 0
        # We know which stone was placed last — check only that player's board
        x_count = count(identity, state.x_pieces)
        o_count = count(identity, state.o_pieces)
        if x_count > o_count
            # X just moved
            return check_win_at(state.x_pieces, row, col, K)
        else
            # O just moved
            return check_win_at(state.o_pieces, row, col, K)
        end
    else
        # Empty board — no winner by default 
        return false
    end
end

"""
    check_full(state::GameState{N,K})

Check if the board is full.
"""
function check_full(state::GameState{N,K}) where {N,K}
    xcount = count(state.x_pieces)
    ocount = count(state.o_pieces)
    return xcount + ocount >= N * N
end

"""
    check_game_result(state)

Determine the game outcome.

Returns:
- `:x_win` if X has won
- `:o_win` if O has won  
- `:draw` if board is full with no winner
- `:ongoing` if game continues
"""
function check_game_result(state::GameState{N,K}) where {N,K}
    check_win(state) && return state.player_turn == 1 ? :o_win : :x_win
   
    check_full(state) && return :draw
    return :ongoing
end

"""
    game_over(state)

Check if the game is over.
"""
game_over(state::GameState{N,K}) where {N,K} = check_game_result(state) != :ongoing

# ============================================
# SYMMETRY OPERATIONS (D4 GROUP)
# ============================================

"""
    identity_transform(board)

Return a copy of the board (identity transformation).
"""
function identity_transform(board::AbstractArray) 
    return copy(board)
end

"""
    rotate90(board)

Rotate the board 90 degrees clockwise.
"""
function rotate90(board::AbstractArray)
    n_rows, n_cols = size(board)
    new_board = falses(n_cols, n_rows)
    for r in 1:n_rows, c in 1:n_cols
        # (r, c) -> (c, n_rows - r + 1)
        new_board[c, n_rows - r + 1] = board[r, c]
    end
    return new_board
end

"""
    rotate180(board)

Rotate the board 180 degrees.
"""
function rotate180(board::AbstractArray)
    n_rows, n_cols = size(board)
    new_board = falses(n_rows, n_cols)
    for r in 1:n_rows, c in 1:n_cols
        # (r, c) -> (n_rows - r + 1, n_cols - c + 1)
        new_board[n_rows - r + 1, n_cols - c + 1] = board[r, c]
    end
    return new_board
end

"""
    rotate270(board)

Rotate the board 270 degrees clockwise (90 counter-clockwise).
"""
function rotate270(board::AbstractArray)
    n_rows, n_cols = size(board)
    new_board = falses(n_cols, n_rows)
    for r in 1:n_rows, c in 1:n_cols
        # (r, c) -> (n_cols - c + 1, r)
        new_board[n_cols - c + 1, r] = board[r, c]
    end
    return new_board
end

"""
    flip_horizontal(board)

Flip the board horizontally (mirror across vertical axis).
"""
function flip_horizontal(board::AbstractArray)
    n_rows, n_cols = size(board)
    new_board = falses(n_rows, n_cols)
    for r in 1:n_rows, c in 1:n_cols
        # (r, c) -> (r, n_cols - c + 1)
        new_board[r, n_cols - c + 1] = board[r, c]
    end
    return new_board
end

"""
    flip_vertical(board)

Flip the board vertically (mirror across horizontal axis).
"""
function flip_vertical(board::AbstractArray)
    n_rows, n_cols = size(board)
    new_board = falses(n_rows, n_cols)
    for r in 1:n_rows, c in 1:n_cols
        # (r, c) -> (n_rows - r + 1, c)
        new_board[n_rows - r + 1, c] = board[r, c]
    end
    return new_board
end

"""
    flip_diag_main(board)

Flip across the main diagonal (transpose).
"""
function flip_diag_main(board::AbstractArray)
    return permutedims(board, (2, 1))
end

"""
    flip_diag_anti(board)

Flip across the anti-diagonal.
"""
function flip_diag_anti(board::AbstractArray)
    n_rows, n_cols = size(board)
    new_board = falses(n_cols, n_rows)
    for r in 1:n_rows, c in 1:n_cols
        # (r, c) -> (n_cols - c + 1, n_rows - r + 1) then transpose
        new_board[n_cols - c + 1, n_rows - r + 1] = board[r, c]
    end
    return new_board
end

"""
    apply_symmetry(board, sym_type)

Apply a symmetry transformation to the board.

sym_type can be:
- 0: identity
- 1: rotate 90°
- 2: rotate 180°
- 3: rotate 270°  
- 4: flip horizontal
- 5: flip vertical
- 6: flip main diagonal
- 7: flip anti-diagonal
"""
function apply_symmetry(board::AbstractArray, sym_type::Int)
    if sym_type == 0
        return identity_transform(board)
    elseif sym_type == 1
        return rotate90(board)
    elseif sym_type == 2
        return rotate180(board)
    elseif sym_type == 3
        return rotate270(board)
    elseif sym_type == 4
        return flip_horizontal(board)
    elseif sym_type == 5
        return flip_vertical(board)
    elseif sym_type == 6
        return flip_diag_main(board)
    elseif sym_type == 7
        return flip_diag_anti(board)
    else
        error("Invalid symmetry type: $sym_type")
    end
end

"""
    board_to_int(board)

Convert a bitboard to an integer representation for comparison.
"""
function board_to_int(board::AbstractArray)
    n_rows, n_cols = size(board)
    result = zero(UInt64)
    idx = 0
    for c in 1:n_cols, r in 1:n_rows
        if board[r, c]
            result |= (UInt64(1) << idx)
        end
        idx += 1
    end
    return result
end

"""
    to_canonical(state)

Return the canonical (smallest) representation of this state among all 8 symmetries.
Also returns which symmetry was applied (0-7).
"""
function to_canonical(state::GameState{N,K}) where {N,K}
    x_pieces = state.x_pieces
    o_pieces = state.o_pieces
    
    best_x = x_pieces
    best_o = o_pieces
    best_sym = 0
    best_hash = board_to_int(x_pieces), board_to_int(o_pieces)
    
    for sym in 1:7
        new_x = apply_symmetry(x_pieces, sym)
        new_o = apply_symmetry(o_pieces, sym)
        
        new_hash = board_to_int(new_x), board_to_int(new_o)
        
        if new_hash < best_hash
            best_hash = new_hash
            best_x = new_x
            best_o = new_o
            best_sym = sym
        end
    end
    
    return GameState{N,K}(best_x, best_o, state.last_move, state.player_turn), best_sym
end

# ============================================
# SAMPLING UTILITY
# ============================================

"""
    action_index_to_move(i_a, N)

Convert an action index to a (row, column) position. Assumes actions are ordered row-wise: (1,1), (1,2), ..., (N,N).
"""
function action_index_to_move(i_a::Int, N::Int)
    r = div(i_a-1, N) + 1
    c = mod(i_a-1, N) + 1
    return (r, c)
end

# ============================================
# POLICY DISTRIBUTIONS
# ============================================

"""
    random_policy_distribution(state)

Produce a uniform distribution over valid moves.
Returns a Vector of length N*N with probabilities for each action index.
"""
function random_policy_distribution(state::GameState{N,K}; dist::Vector{T}=zeros(Float32, N*N)) where {N,K,T<:Real}
    for i in eachindex(dist)
        #convert index to (r, c)
        (r, c) = action_index_to_move(i, N)
        valid = is_valid_move(state, r, c)
        dist[i] = valid
    end
    dist ./= sum(dist)  # Normalize to create a valid distribution
    return dist
end

"""
    random_policy(state)

Select a move at random from the set of valid moves.
"""
function random_policy(state::GameState{N,K}; kwargs...) where {N,K}
    dist = random_policy_distribution(state; kwargs...)
    i_a = sample_action(dist)
    return action_index_to_move(i_a, N)
end

"""
    greedy_policy_distribution(state)

Produce a distribution based on greedy evaluation.
Immediate wins or blocks get probability 1; otherwise preferences center positions.
Returns a Vector of length N*N with probabilities for each action index.
"""
function greedy_policy_distribution(state::GameState{N,K}; dist::Vector{T}=zeros(Float32, N*N)) where {N,K,T<:Real}
    
    current_player = state.player_turn
    pieces = current_player == 1 ? state.x_pieces : state.o_pieces
    opp_pieces = current_player == 1 ? state.o_pieces : state.x_pieces
    
    # FIRST PASS: Scan all moves for an immediate winning move
    for i in eachindex(dist)
        (r, c) = action_index_to_move(i, N)

        if !is_valid_move(state, r, c)
            continue
        end

        # Check if this move creates K-in-a-row (win immediately) — no board copy needed
        if check_win_at(pieces, r, c, K)
            dist .= zero(T)  # Clear all other probabilities
            dist[i] = one(T) # This move wins immediately, so it should be chosen with probability 1
            return dist      # Return immediately — winning move takes priority over everything
        end
    end
    
    # SECOND PASS: Scan all moves for a blocking move (only if no winning move exists)
    for i in eachindex(dist)
        (r, c) = action_index_to_move(i, N)

        if !is_valid_move(state, r, c)
            dist[i] = typemax(T)  # Invalid moves get zero probability after taking inverse
            continue
        end

        # Check if opponent would win next turn without blocking — no board copy needed
        if check_win_at(opp_pieces, r, c, K)
            dist .= zero(T)  # Clear all other probabilities
            dist[i] = one(T)  # Must block opponent's winning move
            return dist # Return immediately — blocking move is selected
        end
        
        # Prefer center positions on larger boards
        center_r = (N + 1) / 2
        center_c = (N + 1) / 2
        dist_to_center = abs(r - center_r) + abs(c - center_c)
        dist[i] = dist_to_center
    end
    
    # No winning or blocking move found — use positional scoring
    dist .= inv.(dist .+ 1f-5)  # Invert distances to prefer closer to center, add small value to avoid division by zero
    dsum = sum(dist)
    dist ./= dsum  # Normalize to create a valid distribution
    return dist
end

"""
    greedy_policy(state)

Select a move based on greedy evaluation scores using softmax sampling.
This creates a non-deterministic policy that reflects the greedy strategy.
"""
function greedy_policy(state::GameState{N,K}; kwargs...) where {N,K}
    dist = greedy_policy_distribution(state; kwargs...)
    i_a = sample_action(dist)
    return action_index_to_move(i_a, N)
end

"""
    defensive_policy_distribution(state)

Produce a distribution based on defensive evaluation.
Considers line creation, blocking opponent threats, and open-ended 3s.
Returns a Vector of length N*N with probabilities for each action index.

Scoring: higher raw scores mean worse moves. After inversion, lower raw scores
become higher probabilities. This follows the same pattern as greedy_policy_distribution.
- typemax for invalid moves (effectively zero probability after inversion)
- 0 for best moves (highest probability after inversion)
- Positive values for worse moves (lower probability after inversion)
- Immediate win or must-block: set probability = 1 directly
"""
function defensive_policy_distribution(state::GameState{N,K}; dist::Vector{T}=zeros(Float32, N*N)) where {N,K,T<:Real}
    
    current_player = player_turn(state)
    my_board = current_player == 1 ? state.x_pieces : state.o_pieces
    opp_board = current_player == 1 ? state.o_pieces : state.x_pieces
    
    # FIRST PASS: Scan all moves for an immediate winning move
    for i in eachindex(dist)
        (r, c) = action_index_to_move(i, N)

        if !is_valid_move(state, r, c)
            continue
        end

        # Win immediately? — no board copy needed
        if check_win_at(my_board, r, c, K)
            dist .= zero(T)
            dist[i] = one(T)
            return dist
        end
    end
    
    # SECOND PASS: Scan all moves for a blocking move (only if no winning move found)
    for i in eachindex(dist)
        (r, c) = action_index_to_move(i, N)

        if !is_valid_move(state, r, c)
            continue
        end

        # Must block opponent win? — no board copy needed
        if check_win_at(opp_board, r, c, K)
            dist .= zero(T)
            dist[i] = one(T)
            return dist
        end
    end
    
    # THIRD PASS: No winning or blocking move — evaluate positional scoring
    for i in eachindex(dist)
        (r, c) = action_index_to_move(i, N)

        if !is_valid_move(state, r, c)
            dist[i] = typemax(T)  # Invalid moves get near-zero probability after inversion
            continue
        end

        # No board copies needed — check lines directly using original boards
        # For partial lines evaluation, we check if the stone at (r,c) would create 
        # lines of various lengths. The count_partial_lines function already checks 
        # all cells in a line, which includes (r,c) itself.
        score = zero(Float64)
        
        # Evaluate creating lines of various lengths — better lines reduce score
        for len in 2:K-1
            if count_partial_lines_at(my_board, r, c, len) > 0
                score -= len * 0.1
            end
        end
        
        if K >= 4 && has_open_ended_3_at(my_board, r, c)
            score -= 0.5
        end
        
        # Evaluate blocking opponent
        for len in 2:K-1
            if count_partial_lines_at(opp_board, r, c, len) > 0
                score -= len * 0.12
            end
        end
        
        if has_open_ended_3_at(opp_board, r, c)
            score -= 0.6
        end
        
        dist[i] = score
    end
    
    # Shift scores to be non-negative (min score may be negative), then invert
    min_score = minimum(dist)
    if min_score < 0
        dist .-= min_score  # Shift so min becomes 0
    end
    
    dist .= inv.(dist .+ 1f-5)  # Invert: 0 → highest probability, larger → lower probability
    dsum = sum(dist)
    dist ./= dsum
    return dist
end

"""
    defensive_policy(state)

Select a move based on defensive evaluation.
Considers line creation, blocking opponent threats, and open-ended 3s.
"""
function defensive_policy(state::GameState{N,K}; kwargs...) where {N,K}
    dist = defensive_policy_distribution(state; kwargs...)
    i_a = sample_action(dist)
    return action_index_to_move(i_a, N)
end

"""
    positional_policy_distribution(state)

Produce a distribution based on comprehensive positional evaluation.
Considers line creation, blocking threats, center control, corner/edge preference,
and multiple simultaneous threats.
Returns a Vector of length N*N with probabilities for each action index.

Scoring: lower raw scores = better moves. After inversion, the best moves 
have the highest probability. This follows the same pattern as greedy_policy_distribution.
- typemax for invalid moves (near-zero probability after inversion)
- Negative values for beneficial moves (good lines, blocking threats, center)
- Must-block opponent win or immediate win: probability = 1
"""
function positional_policy_distribution(state::GameState{N,K}; dist::Vector{T}=zeros(Float32, N*N)) where {N,K,T<:Real}
    
    current_player = player_turn(state)
    my_board = current_player == 1 ? state.x_pieces : state.o_pieces
    opp_board = current_player == 1 ? state.o_pieces : state.x_pieces
    
    # FIRST PASS: Scan all moves for an immediate winning move
    for i in eachindex(dist)
        (r, c) = action_index_to_move(i, N)

        if !is_valid_move(state, r, c)
            continue
        end

        # Win immediately? — no board copy needed
        if check_win_at(my_board, r, c, K)
            dist .= zero(T)
            dist[i] = one(T)
            return dist
        end
    end
    
    # SECOND PASS: Scan all moves for a blocking move (only if no winning move found)
    for i in eachindex(dist)
        (r, c) = action_index_to_move(i, N)

        if !is_valid_move(state, r, c)
            continue
        end

        # Must block opponent win? — no board copy needed
        if check_win_at(opp_board, r, c, K)
            dist .= zero(T)
            dist[i] = one(T)
            return dist
        end
    end
    
    # THIRD PASS: No winning or blocking move — evaluate positional scoring
    for i in eachindex(dist)
        (r, c) = action_index_to_move(i, N)

        if !is_valid_move(state, r, c)
            dist[i] = typemax(T)  # Invalid moves get near-zero probability after inversion
            continue
        end

        # No board copies — use the "at" variants that evaluate mentally
        score = zero(Float64)
        
        # My creation: creating lines is good → negative score
        for len in 2:K-1
            n_lines = count_partial_lines_at(my_board, r, c, len)
            score -= n_lines * (5 ^ (len - 1))
        end
        
        if has_open_ended_3_at(my_board, r, c) && K >= 4
            score -= 50.0
        end
        
        if K >= 5 && has_double_threats_at(my_board, r, c)
            score -= 100.0
        end
        
        # Opponent blocking: blocking opponent lines is good → negative score
        for len in 2:K-1
            n_lines = count_partial_lines_at(opp_board, r, c, len)
            score -= n_lines * (4 ^ (len - 1))
        end
        
        if has_open_ended_3_at(opp_board, r, c) && K >= 4
            score -= 75.0
        end
        
        # ========== POSITIONAL BONUS ==========
        center_r, center_c = (N + 1) / 2, (N + 1) / 2
        dist_to_center = sqrt((r - center_r)^2 + (c - center_c)^2)
        max_dist = sqrt(N^2 + N^2)
        score += 50.0 * (dist_to_center / max_dist)
        
        if r == 1 || r == N || c == 1 || c == N
            score -= 5.0
        end
        
        dist[i] = score
    end
    
    # Shift scores to be non-negative (min score may be negative), then invert
    min_score = minimum(dist)
    if min_score < 0
        dist .-= min_score
    end
    
    dist .= inv.(dist .+ 1f-5)
    dsum = sum(dist)
    dist ./= dsum
    return dist
end

"""
    positional_policy(state)

Select a move based on comprehensive positional evaluation.
Considers line creation, blocking threats, center control, corner/edge preference,
and multiple simultaneous threats.
"""
function positional_policy(state::GameState{N,K}; kwargs...) where {N,K}
    dist = positional_policy_distribution(state; kwargs...)
    i_a = sample_action(dist)
    return action_index_to_move(i_a, N)
end

# ============================================
# OPPONENT GENERATION
# ============================================

"""
    make_opponent(level::Symbol, K::Int)

Generate an opponent policy function at the specified difficulty level.

Levels:
- `:random`: Pure random valid moves (easiest)
- `:greedy`: Tries to win or block immediate threats
- `:defensive`: Considers line creation and multiple threats
- `:positional`: Expert evaluation with positional understanding

Returns a function that takes a GameState and returns (row, col).
"""
function make_opponent(level::Symbol, K::Int)
    if level == :random
        return (state::GameState) -> random_policy(state)
    elseif level == :greedy
        return (state::GameState) -> greedy_policy(state)
    elseif level == :defensive
        return (state::GameState) -> defensive_policy(state)
    elseif level == :positional
        return (state::GameState) -> positional_policy(state)
    else
        error("Unknown opponent level: $level. Valid levels: :random, :greedy, :defensive, :positional")
    end
end

# ============================================
# HELPER FUNCTIONS FOR EVALUATION (NO-COPY VARIANTS)
# ============================================

"""
    count_partial_lines_at(board, row, col, len)

Count how many lines of given length include (row, col) in any direction.
This is the no-copy variant — it treats the board as-is and checks if there are
`len` consecutive occupied cells including (row, col) in any line direction.
Used to evaluate partial line creation without copying the board.
"""
function count_partial_lines_at(board::AbstractArray, row::Int, col::Int, len::Int)
    n_rows, n_cols = size(board)
    count = 0
    
    # Directions: horizontal, vertical, two diagonals
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for (dr, dc) in directions
        # Slide a window of length `len` that includes (row, col)
        # The window starts at `start_offset` positions back from (row, col)
        for start_offset in 0:(len-1)
            start_row = row - start_offset * dr
            start_col = col - start_offset * dc
            
            valid_line = true
            for offset in 0:(len-1)
                r = start_row + offset * dr
                c = start_col + offset * dc
                
                if r < 1 || r > n_rows || c < 1 || c > n_cols
                    valid_line = false
                    break
                end
                
                if !board[r, c]
                    valid_line = false
                    break
                end
            end
            
            if valid_line
                count += 1
                break  # Only count one line per direction
            end
        end
    end
    
    return count
end

"""
    has_open_ended_3_at(board, row, col)

Check if there is a line of exactly 3 including (row, col) that is open-ended
(can be extended at both ends). No board copy needed.
An open-ended 3-in-a-row can become 4-in-a-row with one more move.
"""
function has_open_ended_3_at(board::AbstractArray, row::Int, col::Int)
    n_rows, n_cols = size(board)
    
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for (dr, dc) in directions
        # Count consecutive stones in both directions from (row, col)
        pos_len = 0
        r, c = row + dr, col + dc
        while 1 <= r <= n_rows && 1 <= c <= n_cols && board[r, c]
            pos_len += 1
            r += dr; c += dc
        end
        # Check if the end is open
        pos_open = 1 <= r <= n_rows && 1 <= c <= n_cols && !board[r, c]
        
        neg_len = 0
        r, c = row - dr, col - dc
        while 1 <= r <= n_rows && 1 <= c <= n_cols && board[r, c]
            neg_len += 1
            r -= dr; c -= dc
        end
        # Check if the end is open
        neg_open = 1 <= r <= n_rows && 1 <= c <= n_cols && !board[r, c]
        
        total = 1 + pos_len + neg_len
        # Open-ended 3: exactly 3 in a row, both ends open
        if total >= 3 && pos_open && neg_open
            return true
        end
    end
    
    return false
end

"""
    has_double_threats_at(board, row, col)

Check if placing at (row, col) creates multiple lines of length >= 3.
No board copy needed. This is a strong strategic pattern.
"""
function has_double_threats_at(board::AbstractArray, row::Int, col::Int)
    n_rows, n_cols = size(board)
    
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    lines_count = 0
    
    for (dr, dc) in directions
        # Count consecutive stones in positive direction
        r, c = row + dr, col + dc
        pos_len = 0
        while 1 <= r <= n_rows && 1 <= c <= n_cols && board[r, c]
            pos_len += 1
            r += dr; c += dc
        end
        
        # Count consecutive stones in negative direction
        r, c = row - dr, col - dc
        neg_len = 0
        while 1 <= r <= n_rows && 1 <= c <= n_cols && board[r, c]
            neg_len += 1
            r -= dr; c -= dc
        end
        
        total = pos_len + 1 + neg_len
        
        if total >= 3
            lines_count += 1
        end
    end
    
    return lines_count >= 2
end

"""
    count_partial_lines(board, row, col, length)

Original variant — used for backward compatibility.
Count how many lines of given length pass through (row, col) in any direction,
assuming the board already has the stone at (row, col).
"""
function count_partial_lines(board::AbstractArray, row::Int, col::Int, len::Int)
    n_rows, n_cols = size(board)
    count = 0
    
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for (dr, dc) in directions
        start_row = row - (len - 1) * dr
        start_col = col - (len - 1) * dc
        
        valid_line = true
        for offset in 0:(len-1)
            r = start_row + offset * dr
            c = start_col + offset * dc
            
            if r < 1 || r > n_rows || c < 1 || c > n_cols
                valid_line = false
                break
            end
            
            if !board[r, c]
                valid_line = false
                break
            end
        end
        
        if valid_line
            count += 1
        end
    end
    
    return count
end

"""
    has_open_ended_3(board, row, col)

Original variant — used for backward compatibility.
Check if there's an open-ended 3-in-a-row including (row, col).
"""
function has_open_ended_3(board::AbstractArray, row::Int, col::Int)
    n_rows, n_cols = size(board)
    
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for (dr, dc) in directions
        cells_in_line = Tuple{Int, Int}[]
        
        r, c = row + dr, col + dc
        while 1 <= r <= n_rows && 1 <= c <= n_cols && board[r, c]
            push!(cells_in_line, (r, c))
            r += dr; c += dc
        end
        
        r, c = row - dr, col - dc
        while 1 <= r <= n_rows && 1 <= c <= n_cols && board[r, c]
            push!(cells_in_line, (r, c))
            r -= dr; c -= dc
        end
        
        n_stones = 1 + length(cells_in_line)
        
        if n_stones >= 3
            return true
        end
    end
    
    return false
end

"""
    has_double_threats(board, row, col)

Original variant — used for backward compatibility.
Check if placing at (row, col) creates multiple winning lines.
"""
function has_double_threats(board::AbstractArray, row::Int, col::Int)
    n_rows, n_cols = size(board)
    
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    lines_count = 0
    
    for (dr, dc) in directions
        r, c = row + dr, col + dc
        pos_len = 0
        while 1 <= r <= n_rows && 1 <= c <= n_cols && board[r, c]
            pos_len += 1
            r += dr; c += dc
        end
        
        r, c = row - dr, col - dc
        neg_len = 0
        while 1 <= r <= n_rows && 1 <= c <= n_cols && board[r, c]
            neg_len += 1
            r -= dr; c -= dc
        end
        
        total = pos_len + 1 + neg_len
        
        if total >= 3
            lines_count += 1
        end
    end
    
    return lines_count >= 2
end

# ============================================
# BOARD DISPLAY
# ============================================

"""
    print_board(my_board, opp_board; show_coords=true)

Print a human-readable representation of the board.
'X' = my stone, 'O' = opponent's stone, '.' = empty
"""
function print_board(my_board::AbstractArray, opp_board::AbstractArray; show_coords=true)
    n_rows, n_cols = size(my_board)
    
    if show_coords
        # Print column numbers
        print("   ")
        for c in 1:n_cols
            print(rpad(c, 3))
        end
        println()
        
        # Top border
        print("   " * "-" ^ (n_cols * 3 + 1) * "\n")
    end
    
    for r in 1:n_rows
        if show_coords
            print(lpad(r, 2) * "| ")
        else
            print("  ")
        end
        
        for c in 1:n_cols
            if my_board[r,c]
                print("X ")
            elseif opp_board[r,c]
                print("O ")
            else
                print(". ")
            end
        end
        println()
    end
    
    if show_coords
        print("   " * "-" ^ (n_cols * 3 + 1))
        println()
    end
end

# For GameState
function print_board(state::GameState; kwargs...)
    print_board(state.x_pieces, state.o_pieces; kwargs...)
    println("Game Status: ", check_game_result(state))
end

"""
    count_stones(board)

Count the number of stones on the board.
"""
count_stones(board::AbstractArray) = count(identity, board)

"""
    total_moves(state)

Return the total number of moves made so far.
"""
total_moves(state::GameState) = count_stones(state.x_pieces) + count_stones(state.o_pieces)

"""
    _add_to_dict!(d, key, v)

Helper to add a value to a dictionary. For integer types, only positive values are added.
For non-integer types (e.g. floats), all values are added.
"""
_add_to_dict!(d, key, v::Integer) = v > 0 ? (d[key] = v) : nothing
_add_to_dict!(d, key, v) = (d[key] = v)

"""
    _vector_to_dict(data, state)

Helper function to convert a vector to a Dict mapping grid positions to values.
If data is already a Dict, returns it as-is.
If data is a Vector, converts using action_index_to_move.
For integer vectors, non-positive values are skipped.
For float vectors, all values are included.
"""
function _vector_to_dict(data::Dict, state::GameState{N,K}) where {N,K}
    return data
end

function _vector_to_dict(data::AbstractVector{T}, state::GameState{N,K}) where {N,K,T}
    d = Dict{Tuple{Int,Int}, T}()
    for i in eachindex(data)
        (r, c) = action_index_to_move(i, N)
        _add_to_dict!(d, (r, c), data[i])
    end
    return d
end

"""
    html_board(state::GameState; kwargs...)

Return an HTML string displaying the game board with CSS-styled X and O pieces.

Arguments:
- `state`: The GameState to display
- `cell_size`: Size of each cell in pixels (default: 60)
- `show_status`: Whether to show the game status below the board (default: true)
- `policy`: Optional policy distribution for heatmap overlay.
           Can be a Dict{Tuple{Int,Int}, Float64} or a Vector of length N*N.
           Empty squares will be colored based on probability: dark blue (low) to bright green (high)
- `board_id`: Optional unique ID for this board (defaults to auto-generated)
- `candidate_move`: Optional move to highlight for the current player.
                    Can be an Int (action index) or a Tuple{Int,Int} (row, col).
                    The move is shown with a semi-transparent colored stone.
                    If nothing, no candidate move is displayed (default: nothing)
- `visit_count`: Optional visit counts for unoccupied squares.
                Can be a Dict{Tuple{Int,Int}, <:Integer} or a Vector of length N*N.
                Only positive integers are shown, displayed in a small font.
                If nothing, no visit counts are displayed (default: nothing)
- `visit_count_font_size`: Font size in pixels for visit count numbers (default: 13)

Returns HTML that can be displayed in a browser or notebook environment.
"""
function html_board(state::GameState{N,K}; cell_size=60, show_status=true, policy=nothing,
                    board_id=string(next_board_id()), candidate_move=nothing,
                    visit_count=nothing, visit_count_font_size=13) where {N,K}
    html = IOBuffer()
    
    # Convert policy to Dict form if it's a vector
    if policy !== nothing
        if !(policy isa Dict)
            policy = _vector_to_dict(policy, state)
        end
    end
    
    # Convert visit_count to Dict form if it's a vector
    if visit_count !== nothing
        if !(visit_count isa Dict)
            visit_count = _vector_to_dict(visit_count, state)
        end
    end
    
    # Convert candidate_move from action index to (r, c) if needed
    if candidate_move isa Int
        candidate_move = action_index_to_move(candidate_move, N)
    end
    
    # Calculate baseline probability for uniform distribution
    moves = available_moves(state)
    n_moves = length(moves)
    uniform_prob = n_moves > 0 ? 1.0 / n_moves : 0.0
    
    print(html, """<style>
        #gc-$(board_id) {
            display: inline-grid;
            grid-template-columns: repeat($N, $(cell_size)px);
            grid-template-rows: repeat($N, $(cell_size)px);
            gap: 2px;
            background-color: #8b5a2b;
            border: 4px solid #6d4c32;
            border-radius: 4px;
            padding: 4px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        #gc-$(board_id) .gc-cell {
            width: $(cell_size)px;
            height: $(cell_size)px;
            background-color: #eecfa1;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }
        #gc-$(board_id) .gc-cell.highlight-win {
            background-color: #ffeb3b !important;
        }
        #gc-$(board_id) .gc-cell.x, #gc-$(board_id) .gc-cell.o {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #gc-$(board_id) .gc-x, #gc-$(board_id) .gc-o {
            position: absolute;
            width: $(cell_size * 0.7)px;
            height: $(cell_size * 0.7)px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #gc-$(board_id) .gc-x::before, #gc-$(board_id) .gc-x::after {
            content: '';
            position: absolute;
            width: 100%;
            height: $(cell_size * 0.12)px;
            background-color: #1a1a1a;
            border-radius: 2px;
        }
        #gc-$(board_id) .gc-x::before {
            transform: rotate(45deg);
        }
        #gc-$(board_id) .gc-x::after {
            transform: rotate(-45deg);
        }
        #gc-$(board_id) .gc-o::before {
            content: '';
            position: absolute;
            width: 100%;
            height: 100%;
            border: $(cell_size * 0.12)px solid #1a1a1a;
            border-radius: 50%;
            box-sizing: border-box;
        }
        #gc-$(board_id) .gc-cell:hover {
            background-color: #dfd5b5;
        }
        #gc-$(board_id) .gc-policy-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            border-radius: 2px;
        }
        #gc-$(board_id) .gc-candidate-x::before, #gc-$(board_id) .gc-candidate-x::after {
            content: '';
            position: absolute;
            width: 100%;
            height: $(cell_size * 0.12)px;
            background-color: #ff6f00;
            border-radius: 2px;
            opacity: 0.85;
        }
        #gc-$(board_id) .gc-candidate-x::before {
            transform: rotate(45deg);
        }
        #gc-$(board_id) .gc-candidate-x::after {
            transform: rotate(-45deg);
        }
        #gc-$(board_id) .gc-candidate-o::before {
            content: '';
            position: absolute;
            width: 100%;
            height: 100%;
            border: $(cell_size * 0.12)px solid #ff6f00;
            border-radius: 50%;
            box-sizing: border-box;
            opacity: 0.85;
        }
        #gc-$(board_id) .gc-candidate-x, #gc-$(board_id) .gc-candidate-o {
            position: absolute;
            width: $(cell_size * 0.7)px;
            height: $(cell_size * 0.7)px;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 2;
            pointer-events: none;
            filter: drop-shadow(0 0 3px #ffffff);
        }
        #gc-$(board_id) .gc-visit-count {
            position: absolute;
            bottom: 2px;
            right: 3px;
            font-size: $(visit_count_font_size)px;
            font-family: sans-serif;
            font-weight: bold;
            color: #333333;
            z-index: 3;
            pointer-events: none;
            text-shadow: 0 0 2px #ffffff, 0 0 2px #ffffff;
        }
    </style>
    <div id="gc-$(board_id)" class="gc-board">
""")
    
    # Get winning line cells if there's a win
    winning_cells = Set{Tuple{Int,Int}}()
    result = check_game_result(state)
    if result == :x_win
        winning_cells = find_winning_line(state.x_pieces, K)
    elseif result == :o_win
        winning_cells = find_winning_line(state.o_pieces, K)
    end
    
    for r in 1:N
        for c in 1:N
            # Determine cell class
            cell_class = "gc-cell"
            if state.x_pieces[r, c]
                cell_class *= " x"
            elseif state.o_pieces[r, c]
                cell_class *= " o"
            end
            if (r, c) in winning_cells
                cell_class *= " highlight-win"
            end
            
            print(html, """        <div class="$cell_class" data-row="$r" data-col="$c">
""")
            
            # Add X or O content
            if state.x_pieces[r, c]
                print(html, """            <div class="gc-x"></div>
""")
            elseif state.o_pieces[r, c]
                print(html, """            <div class="gc-o"></div>
""")
            else
                # Add policy overlay for empty squares
                if policy !== nothing
                    prob = get(policy, (r, c), 0.0)
                    
                    if uniform_prob > 0
                        # Asymmetric color scale:
                        # Below uniform: smooth quadratic from dark navy (0) → neutral beige (uniform)
                        # Above uniform: linear from neutral beige (uniform) → bright green (1)
                        ratio = prob / uniform_prob
                        
                        if ratio <= 1.0
                            # Below uniform: use quadratic for smooth approach to neutral
                            # t ranges from 0 (prob=0) to 1 (prob=uniform_prob)
                            t = ratio^2.0
                            # Blend from dark navy (26,26,100) at t=0 to neutral beige (200,180,150) at t=1
                            r_val = Int(round(26 + (200 - 26) * t))
                            g_val = Int(round(26 + (180 - 26) * t))
                            b_val = Int(round(100 + (150 - 100) * t))
                        else
                            # Above uniform: linear from uniform to 1
                            max_ratio = 1.0 / uniform_prob
                            # t ranges from 0 (uniform_prob) to 1 (prob=1)
                            t = min((ratio - 1.0) / (max_ratio - 1.0), 1.0)
                            # Blend from neutral beige (200,180,150) at t=0 to bright green (26,230,26) at t=1
                            r_val = Int(round(200 + (26 - 200) * t))
                            g_val = Int(round(180 + (230 - 180) * t))
                            b_val = Int(round(150 + (26 - 150) * t))
                        end
                        
                        print(html, """            <div class="gc-policy-overlay" style="background-color: rgb($r_val, $g_val, $b_val);"></div>
""")
                    end
                end
                
                # Add candidate move overlay for empty squares
                if candidate_move !== nothing && (r, c) == candidate_move
                    turn = state.player_turn
                    if turn == 1
                        print(html, """            <div class="gc-candidate-x"></div>
""")
                    else
                        print(html, """            <div class="gc-candidate-o"></div>
""")
                    end
                end
                
                # Add visit count for empty squares
                if visit_count !== nothing
                    vc = get(visit_count, (r, c), 0)
                    if vc > 0
                        print(html, """            <span class="gc-visit-count">$(Int(vc))</span>
""")
                    end
                end
            end
            
            print(html, """        </div>
""")
        end
    end
    
    # Status centered under the board (inside the board container for proper centering)
    if show_status
        status = check_game_result(state)
        
        if status == :x_win
            status_text = "X Wins!"
            status_color = "#2e7d32"
        elseif status == :o_win
            status_text = "O Wins!"
            status_color = "#c62828"
        elseif status == :draw
            status_text = "Draw"
            status_color = "#5d4037"
        else
            turn = state.player_turn
            player_text = turn == 1 ? "X" : "O"
            status_text = "$player_text's Turn"
            status_color = "#ffffff"
        end
        
        print(html, """    <div style="margin-top: 12px; text-align: center; font-family: sans-serif; font-size: 16px; grid-column: 1 / -1;">
        <strong>Game Status:</strong> <span style="color: $status_color; font-weight: bold;">$status_text</span>
    </div>
""")
        
        # Conditional legend items
        has_legend = false
        legend_html = IOBuffer()
        
        # Policy color bar legend
        if policy !== nothing
            has_legend = true
            print(legend_html, """                <div style="display: flex; align-items: center; gap: 4px;">
                    <div style="width: 60px; height: 20px; border-radius: 2px; background: linear-gradient(to right, rgb(26,26,100), rgb(200,180,150), rgb(26,230,26)); position: relative; display: flex; align-items: center; justify-content: center;">
                        <span style="font-size: 9px; color: #ffffff; font-weight: bold; text-shadow: 0 0 3px #000000, 0 0 3px #000000;">Policy Action Probabilities</span>
                    </div>
                </div>
""")
        end
        
        # Candidate move legend
        if candidate_move !== nothing
            has_legend = true
            turn = state.player_turn
            print(legend_html, """                <div style="display: flex; align-items: center; gap: 4px;">
                    <div style="width: 14px; height: 14px; position: relative; display: flex; align-items: center; justify-content: center;">
                        <div style="position: absolute; width: 100%; height: 2px; background-color: #ff6f00; border-radius: 1px; transform: rotate(45deg); opacity: 0.85; filter: drop-shadow(0 0 2px #ffffff);"></div>
                        <div style="position: absolute; width: 100%; height: 2px; background-color: #ff6f00; border-radius: 1px; transform: rotate(-45deg); opacity: 0.85; filter: drop-shadow(0 0 2px #ffffff);"></div>
                    </div>
                    <span style="font-size: 11px; color: #ffffff;">Action Taken</span>
                </div>
""")
        end
        
        # Visit count legend
        if visit_count !== nothing
            has_legend = true
            print(legend_html, """                <div style="display: flex; align-items: center; gap: 4px;">
                    <span style="font-size: $(visit_count_font_size)px; font-family: sans-serif; font-weight: bold; color: #ffffff; text-shadow: 0 0 2px #000, 0 0 2px #000;">12</span>
                    <span style="font-size: 11px; color: #ffffff;">Visit Counts</span>
                </div>
""")
        end
        
        if has_legend
            print(html, """    <div style="margin-top: 8px; display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; grid-column: 1 / -1;">
$(String(take!(legend_html)))    </div>
""")
        end
    end
    
    print(html, """    </div>
""")
    
    return String(take!(html))
end

"""
    find_winning_line(board, K)

Find the winning line on the board and return the cell coordinates.
"""
function find_winning_line(board::AbstractMatrix, K::Int)
    n_rows, n_cols = size(board)
    winning_cells = Set{Tuple{Int,Int}}()
    
    # Check all starting positions
    for r in 1:n_rows, c in 1:n_cols
        !board[r,c] && continue
        
        # Check horizontal
        if c + K - 1 <= n_cols
            won = true
            for i in 1:K
                if !board[r, c+i-1]
                    won = false
                    break
                end
            end
            if won
                for i in 1:K
                    push!(winning_cells, (r, c+i-1))
                end
                continue
            end
        end
        
        # Check vertical
        if r + K - 1 <= n_rows
            won = true
            for i in 1:K
                if !board[r+i-1, c]
                    won = false
                    break
                end
            end
            if won
                for i in 1:K
                    push!(winning_cells, (r+i-1, c))
                end
                continue
            end
        end
        
        # Check diagonal down-right
        if r + K - 1 <= n_rows && c + K - 1 <= n_cols
            won = true
            for i in 1:K
                if !board[r+i-1, c+i-1]
                    won = false
                    break
                end
            end
            if won
                for i in 1:K
                    push!(winning_cells, (r+i-1, c+i-1))
                end
                continue
            end
        end
        
        # Check diagonal down-left
        if r + K - 1 <= n_rows && c - K + 1 >= 1
            won = true
            for i in 1:K
                if !board[r+i-1, c-i+1]
                    won = false
                    break
                end
            end
            if won
                for i in 1:K
                    push!(winning_cells, (r+i-1, c-i+1))
                end
                continue
            end
        end
    end
    
    return winning_cells
end
end  # module GridCapture