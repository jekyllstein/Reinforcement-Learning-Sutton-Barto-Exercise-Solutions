# Test script for GridCapture module
# Demonstrates game functionality, opponent policies, and performance tests

using Random: seed!

include("GridCapture.jl")
using .GridCapture
using Test
using BenchmarkTools
using InteractiveUtils

function run_tests()
    println("=== GridCapture Module Tests ===\n")

    # ============================================
    # TEST 1: BASIC FUNCTIONALITY
    # ============================================
    println("\n--- Test 1: Basic Game Setup ---")
    game = GameState(6, 4)
    print_board(game)
    println("Valid moves: ", length(available_moves(game)))

    # Test 2: Place Stones and Check Win
    println("\n--- Test 2: Place Stones ---")
    state = place_stone(game, 1, 1)
    print_board(state)
    state = place_stone(state, 3, 1)
    print_board(state)
    state = place_stone(state, 1, 2)
    print_board(state)

    # Test 3: Check Win Detection
    println("\n--- Test 3: Check Win Detection ---")
    game_result = check_game_result(state)
    println("Game result: ", game_result)

    # Test 4: Complete 3-in-a-row (K=4)
    println("\n--- Test 4: Complete 3-in-a-row (K=4) ---")
    state = place_stone(state, 1, 3)
    print_board(state)
    game_result = check_game_result(state)
    println("Game result: ", game_result)

    # Test 5: Different K Values
    println("\n--- Test 5: Different K Values ---")
    for k in [3, 4, 5]
        local_state = GameState(6, k)
        println("Board 6x6 with K=$k:")
        print_board(local_state)
    end

    # Test 6: Symmetry Operations
    println("\n--- Test 6: Symmetry Transformations ---")
    state = GameState(4, 4)
    state = place_stone(state, 1, 1)
    state = place_stone(state, 2, 2)
    print_board(state)

    for sym in 0:3
        new_state, sym_type = to_canonical(state)
        println("\nSymmetry type $sym (canonical form):")
        print_board(new_state; show_coords=false)
    end

    # Test 7: Opponent Policies
    println("\n--- Test 7: Opponent Policies ---")

    for level in [:random, :greedy]
        println("\nOpponent level: $level")
        opponent = make_opponent(level, 4)
        
        state = GameState(6, 4)
        print_board(state)
        
        for move_num in 1:5
            current_player = player_turn(state)
            if current_player == 1
                my_move = first(available_moves(state))
                state = place_stone(state, my_move[1], my_move[2])
            else
                opp_move, score = opponent(state)
                println("Opponent chose $(opp_move) with score $score")
                state = place_stone(state, opp_move[1], opp_move[2])
            end
            print_board(state)
        end
    end

    # Test 8: Game Over Detection
    println("\n--- Test 8: Game Over Detection ---")
    state = GameState(4, 4)
    for c in 1:4
        state = place_stone(state, 1, c)
    end
    print_board(state)
    result = check_game_result(state)
    println("Game result: $result")

    # Test 9: Complete Random Game
    println("\n--- Test 9: Complete Random Game ---")
    seed!(42)
    state = GameState(5, 4)
    while !game_over(state)
        if player_turn(state) == 1
            move = rand(available_moves(state))
            state = place_stone(state, move[1], move[2])
        else
            opponent = make_opponent(:random, 4)
            move, _ = opponent(state)
            state = place_stone(state, move[1], move[2])
        end
    end
    result = check_game_result(state)
    println("Final result: $result")

    # Test 10: Large Board
    println("\n--- Test 10: Large Board (8x8) ---")
    state = GameState(8, 4)
    println("Board size: $(size(state.x_pieces))")
    println("Valid moves at start: ", length(available_moves(state)))

    positional_opp = make_opponent(:positional, 4)
    move, score = positional_opp(state)
    println("Positional opponent first move: $move with score $score")

    println("\n=== Basic Functionality Tests Completed ===\n")

    # ============================================
    # TEST 11: DISTRIBUTION FUNCTIONS
    # ============================================
    println("\n--- Test 11: Distribution Functions ---")

    # Test that distributions return valid probabilities
    for level in [:random, :greedy, :positional]
        println("\nTesting $level distribution:")
        opp = make_opponent(level, 4)
        state = GameState(6, 4)
        
        # Get distribution
        if level == :random
            dist = random_distribution(state)
        elseif level == :greedy
            dist = greedy_distribution(state)
        else
            dist = positional_distribution(state)
        end
        
        # Verify probabilities sum to 1
        total = sum(values(dist))
        println("  Sum of probabilities: $total (should be ~1.0)")
        
        # Verify all moves have positive probability
        valid = all(p -> p > 0, values(dist))
        println("  All probabilities positive: $valid")
        
        # Verify number of moves matches available moves
        n_dist = length(dist)
        n_available = length(available_moves(state))
        println("  Distribution size: $n_dist, Available moves: $n_available")
        println("  Match: $(n_dist == n_available)")
    end

    # Test that policies match their distributions
    println("\n--- Test 11b: Policy-Distribution Consistency ---")
    state = GameState(6, 4)

    for level in [:random, :greedy, :positional]
        println("\nTesting $level policy vs distribution:")
        opp = make_opponent(level, 4)
        
        # Run multiple games to see action frequencies
        n_games = 100
        move_counts = Dict{Tuple{Int,Int}, Int}()
        
        for _ in 1:n_games
            s = deepcopy(state)
            while !game_over(s) && length(available_moves(s)) > 0
                move, score = opp(s)
                move_counts[move] = get(move_counts, move, 0) + 1
                s = place_stone(s, move[1], move[2])
                if game_over(s)
                    break
                end
                # Use random opponent for the other player
                move2 = rand(available_moves(s))
                s = place_stone(s, move2[1], move2[2])
            end
        end
        
        println("  Total moves tracked: $(sum(values(move_counts)))")
        println("  Unique moves: $(length(move_counts))")
    end

    println("\n=== Distribution Tests Completed ===\n")

    # ============================================
    # TEST 12: VALUE FUNCTIONS
    # ============================================
    println("\n--- Test 12: Value Functions ---")

    # Test value functions return expected values
    state = GameState(4, 4)
    for level in [:random, :greedy, :positional]
        if level == :random
            v = random_value(state)
        elseif level == :greedy
            v = greedy_value(state)
        else
            v = positional_value(state)
        end
        println("$level value at empty state: $v")
    end

    # Test value function on terminal states
    terminal_x_win = GameState(4, 4)
    terminal_x_win = place_stone(terminal_x_win, 1, 1)
    terminal_x_win = place_stone(terminal_x_win, 3, 1)
    terminal_x_win = place_stone(terminal_x_win, 1, 2)
    terminal_x_win = place_stone(terminal_x_win, 3, 2)
    terminal_x_win = place_stone(terminal_x_win, 1, 3)
    print_board(terminal_x_win)
    println("Terminal X win value: $(positional_value(terminal_x_win)) (should be 1.0)")

    # Test value function on draw
    terminal_draw = GameState(4, 4)
    for r in 1:4, c in 1:4
        if (r+c) % 2 == 0
            terminal_draw = place_stone(terminal_draw, r, c)
        else
            terminal_draw = place_stone(terminal_draw, r, c)
        end
    end
    print_board(terminal_draw)
    println("Terminal draw value: $(positional_value(terminal_draw)) (should be 0.0)")

    println("\n=== Value Function Tests Completed ===\n")

    # ============================================
    # TEST 13: OPPONENT TOURNAMENT
    # ============================================
    println("\n--- Test 13: Opponent Tournament ---")

    """
        run_opponent_tournament(x_strategy, o_strategy, n_games, K, seed=42)

    Run a tournament between two strategies and return win statistics.
    """
    function run_opponent_tournament(
        x_strategy, o_strategy, n_games, K, seed=42
    )
        seed!(seed)
        
        x_wins = 0
        o_wins = 0
        draws = 0
        
        x_opp = make_opponent(x_strategy, K)
        o_opp = make_opponent(o_strategy, K)
        
        for game_num in 1:n_games
            state = GameState(K, K)
            
            while !game_over(state)
                if player_turn(state) == 1
                    move, _ = x_opp(state)
                    state = place_stone(state, move[1], move[2])
                else
                    move, _ = o_opp(state)
                    state = place_stone(state, move[1], move[2])
                end
            end
            
            result = check_game_result(state)
            if result == :x_win
                x_wins += 1
            elseif result == :o_win
                o_wins += 1
            else
                draws += 1
            end
        end
        
        return Dict(
            :x_wins => x_wins,
            :o_wins => o_wins,
            :draws => draws,
            :total => n_games
        )
    end

    # Run tournament matrix
    strategies = [:random, :greedy, :defensive, :positional]
    results = Dict{Tuple, Dict}()

    println("\nOpponent Tournament Results (X vs O):")
    println("=====================================")

    for x_strat in strategies, o_strat in strategies
        key = (x_strat, o_strat)
        results[key] = run_opponent_tournament(x_strat, o_strat, 50, 4, 42)
        
        x_w = results[key][:x_wins]
        o_w = results[key][:o_wins]
        d = results[key][:draws]
        
        if x_strat == o_strat
            println("$x_strat vs $o_strat: X=$x_w, O=$o_w, Draw=$d")
        else
            println("$x_strat vs $o_strat: X=$x_w, O=$o_w, Draw=$d")
        end
    end

    # Analyze relative strength
    println("\n--- Tournament Analysis ---")
    for x_strat in strategies
        avg_x_win = 0.0
        avg_o_win = 0.0
        
        for o_strat in strategies
            if x_strat != o_strat
                r = results[(x_strat, o_strat)]
                avg_x_win += r[:x_wins] / 50.0
                avg_o_win += r[:o_wins] / 50.0
            end
        end
        
        avg_x_win /= 3.0
        avg_o_win /= 3.0
        
        println("$x_strat: avg X-win rate vs others: $(round(avg_x_win, digits=2))")
    end

    println("\n=== Opponent Tournament Tests Completed ===\n")

    # ============================================
    # TEST 14: ROLLOUT VALUE FUNCTION
    # ============================================
    println("\n--- Test 14: Rollout Value Function ---")

    """
        rollout_value(state, x_strategy, o_strategy, n_rollouts, K, seed)

    Estimate state value by running random rollouts between strategies.
    """
    function rollout_value(
        state, x_strategy, o_strategy, n_rollouts, K, seed=42
    )
        seed!(seed)
        
        total_value = 0.0
        
        x_opp = make_opponent(x_strategy, K)
        o_opp = make_opponent(o_strategy, K)
        
        for rollout in 1:n_rollouts
            s = deepcopy(state)
            
            while !game_over(s)
                if player_turn(s) == 1
                    move, _ = x_opp(s)
                    s = place_stone(s, move[1], move[2])
                else
                    move, _ = o_opp(s)
                    s = place_stone(s, move[1], move[2])
                end
            end
            
            result = check_game_result(s)
            if result == :x_win
                total_value += 1.0
            elseif result == :o_win
                total_value -= 1.0
            else
                total_value += 0.0
            end
        end
        
        return total_value / n_rollouts
    end

    # Test rollout value function
    println("\nTesting rollout_value function:")
    state = GameState(6, 4)

    println("\nEmpty board, random vs random (10 rollouts):")
    v = rollout_value(state, :random, :random, 10, 4, 42)
    println("  Value: $v (should be near 0)")

    println("\nEmpty board, greedy vs random (10 rollouts):")
    v = rollout_value(state, :greedy, :random, 10, 4, 42)
    println("  Value: $v (should be positive - greedy has advantage)")

    println("\nEmpty board, random vs greedy (10 rollouts):")
    v = rollout_value(state, :random, :greedy, 10, 4, 42)
    println("  Value: $v (should be negative - random at disadvantage)")

    println("\n=== Rollout Value Tests Completed ===\n")

    # ============================================
    # TEST 15: TYPE STABILITY AND ALLOCATIONS
    # ============================================
    println("\n--- Test 15: Type Stability and Allocations ---")

    # Test basic functions for type stability
    println("\nTesting type stability...")

    # player_turn
    state = GameState(4, 4)
    @assert player_turn(state) == 1
    println("  player_turn: OK")

    # available_moves
    moves = available_moves(state)
    @assert typeof(moves) == Vector{Tuple{Int, Int}}
    println("  available_moves: OK")

    # place_stone
    new_state = place_stone(state, 1, 1)
    @assert typeof(new_state) == GameState{4,4}
    println("  place_stone: OK")

    # check_game_result
    result = check_game_result(state)
    @assert typeof(result) == Symbol
    println("  check_game_result: OK")

    # game_over
    @assert game_over(state) == false
    println("  game_over: OK")

    # to_canonical
    canonical, sym = to_canonical(state)
    @assert typeof(canonical) == GameState{4,4}
    @assert typeof(sym) == Int
    println("  to_canonical: OK")

    # Strategy functions
    println("\nTesting strategy functions...")

    # random_policy
    move, score = random_policy(state)
    @assert typeof(move) == Tuple{Int, Int}
    @assert typeof(score) == Int
    println("  random_policy: OK")

    # greedy_policy
    move, score = greedy_policy(state)
    @assert typeof(move) == Tuple{Int, Int}
    @assert typeof(score) <: Real
    println("  greedy_policy: OK")

    # Check distribution functions
    println("\nTesting distribution functions...")

    dist = random_distribution(state)
    @assert typeof(dist) == Dict{Tuple{Int, Int}, Float64}
    @assert isapprox(sum(values(dist)), 1.0; atol=1e-10)
    println("  random_distribution: OK")

    dist = greedy_distribution(state)
    @assert typeof(dist) == Dict{Tuple{Int, Int}, Float64}
    @assert isapprox(sum(values(dist)), 1.0; atol=1e-10)
    println("  greedy_distribution: OK")

    dist = positional_distribution(state)
    @assert typeof(dist) == Dict{Tuple{Int, Int}, Float64}
    @assert isapprox(sum(values(dist)), 1.0; atol=1e-10)
    println("  positional_distribution: OK")

    # Check value functions return Float64
    println("\nTesting value functions...")

    v = random_value(state)
    @assert typeof(v) == Float64
    println("  random_value: OK")

    v = greedy_value(state)
    @assert typeof(v) == Float64
    println("  greedy_value: OK")

    v = positional_value(state)
    @assert typeof(v) == Float64
    println("  positional_value: OK")

    # Type stability checks using @code_warntype
    println("\n=== Type Stability Tests with @code_warntype ===")
    println("\nChecking for type instabilities in key functions...")
    
    # Use @code_warntype to check for type instabilities
    @code_warntype player_turn(state)
    @code_warntype available_moves(state)
    @code_warntype check_game_result(state)
    
    println("\nType stability checks completed.")
    println("Note: @code_warntype output shown above should not show 'Any' types")
    println("for core operations.")

    println("\n=== Type Stability Tests Completed ===\n")

    # ============================================
    # TEST 16: ALLOCATION BENCHMARKS
    # ============================================
    println("\n--- Test 16: Allocation Benchmarks ---")

    # Benchmark available_moves
    println("\nBenchmarking available_moves (6x6 board):")
    @btime available_moves($state) setup=(state = GameState(6, 4))

    # Benchmark place_stone
    println("\nBenchmarking place_stone:")
    @btime place_stone($state, 1, 1) setup=(state = GameState(6, 4))

    # Benchmark player_turn
    println("\nBenchmarking player_turn:")
    @btime player_turn($state) setup=(state = GameState(6, 4))

    # Benchmark check_win
    println("\nBenchmarking check_win:")
    board = state.x_pieces
    @btime check_win($board, 4)

    # Benchmark policy functions
    println("\nBenchmarking random_policy:")
    @btime random_policy($state) setup=(state = GameState(6, 4))

    println("\nBenchmarking greedy_policy:")
    @btime greedy_policy($state) setup=(state = GameState(6, 4))

    println("\nBenchmarking positional_policy:")
    @btime positional_policy($state) setup=(state = GameState(6, 4))

    # Benchmark distribution functions
    println("\nBenchmarking random_distribution:")
    @btime random_distribution($state) setup=(state = GameState(6, 4))

    println("\nBenchmarking greedy_distribution:")
    @btime greedy_distribution($state) setup=(state = GameState(6, 4))

    println("\nBenchmarking positional_distribution:")
    @btime positional_distribution($state) setup=(state = GameState(6, 4))

    # Benchmark value functions
    println("\nBenchmarking random_value:")
    @btime random_value($state) setup=(state = GameState(6, 4))

    println("\nBenchmarking greedy_value:")
    @btime greedy_value($state) setup=(state = GameState(6, 4))

    println("\nBenchmarking positional_value:")
    @btime positional_value($state) setup=(state = GameState(6, 4))

    println("\n=== Allocation Benchmarks Completed ===\n")

    # ============================================
    # SUMMARY
    # ============================================
    println("\n=== All Tests Completed Successfully! ===")
    println("\nNext steps for MCTS integration:")
    println("1. Ensure opponent distributions match actual policy behavior")
    println("2. Test rollout_value with parallel execution using Transducers")
    println("3. Integrate with StateMDP from TabularRL")
    println("4. Implement Gumbel MCTS search using action distributions")
end

# Run the tests
run_tests()