# Benchmark test for GridCapture optimizations
# Run: julia --project=ReinforcementLearning.jl ReinforcementLearning.jl/src/test_gridcapture_benchmark.jl
# This must be run from the Reinforcement-Learning-Sutton-Barto-Exercise-Solutions directory

using Random, BenchmarkTools

# Load the full RL package which provides TabularRL for sample_action
include(joinpath(@__DIR__, "GridCapture.jl"))
using .GridCapture

# ============================================
# Compile everything first
# ============================================
let s = GameState(3, 3)
    s = place_stone(s, 1, 1)
    s = place_stone(s, 2, 2)
    check_win(s)
    check_game_result(s)
    greedy_policy_distribution(s)
end

println("="^80)
println("  GRIDCAPTURE OPTIMIZATION BENCHMARKS")
println("="^80)

println("\n" * "-"^80)
println("  PART 1: check_win_at (incremental, O(K)) vs old copy+full-scan (O(N²·K))")
println("-"^80)

for (label, N, K, n_stones) in [("3×3 K=3", 3, 3, 2), ("6×6 K=4", 6, 4, 8), ("9×9 K=5", 9, 5, 15)]
    # Build a mid-game state
    state = GameState(N, K)
    rng = MersenneTwister(42)
    for _ in 1:min(n_stones, N*N-1)
        moves = collect(available_moves(state))
        isempty(moves) && break
        idx = rand(rng, 1:length(moves))
        state = place_stone(state, moves[idx][1], moves[idx][2])
    end
    
    moves = available_moves(state)
    isempty(moves) && continue
    (r, c) = moves[1]
    turn = player_turn(state)
    board = turn == 1 ? state.x_pieces : state.o_pieces
    
    println("\n  $label board, move=($r,$c):")
    
    # new: check_win_at
    t = @timed for _ in 1:100000; GridCapture.check_win_at(board, r, c, K); end
    b_at = @benchmark GridCapture.check_win_at($board, $r, $c, $K)
    println("    check_win_at:     $(BenchmarkTools.prettytime(median(b_at.time)))  ($(b_at.allocs) allocs)")
    
    # old: copy+place+full-scan
    function old_way(b, r, c, k)
        b2 = copy(b)
        b2[r, c] = true
        check_win(b2, k)
    end
    b_old = @benchmark old_way($board, $r, $c, $K)
    println("    old copy+scan:    $(BenchmarkTools.prettytime(median(b_old.time)))  ($(b_old.allocs) allocs)")
    
    speedup = round(Int, median(b_old.time) / max(median(b_at.time), 1))
    println("    ⚡ Speedup: $(speedup)×, allocations reduced: $(b_old.allocs) → $(b_at.allocs)")
end

println("\n" * "-"^80)
println("  PART 2: check_game_result (GameState dispatch: incremental via last_move)")
println("-"^80)

for (label, N, K, n_stones) in [("3×3 K=3", 3, 3, 3), ("6×6 K=4", 6, 4, 10), ("9×9 K=5", 9, 5, 20)]
    state = GameState(N, K)
    rng = MersenneTwister(42)
    for _ in 1:min(n_stones, N*N-1)
        moves = collect(available_moves(state))
        isempty(moves) && break
        idx = rand(rng, 1:length(moves))
        state = place_stone(state, moves[idx][1], moves[idx][2])
    end
    
    println("\n  $label board (last_move=$(state.last_move)):")
    
    b_new = @benchmark check_game_result($state)
    println("    incremental (new): $(BenchmarkTools.prettytime(median(b_new.time)))  ($(b_new.allocs) allocs)")
end

println("\n" * "-"^80)
println("  PART 3: greedy_policy_distribution (no board copies)")
println("-"^80)

for (label, N, K, n_stones) in [("3×3 K=3", 3, 3, 2), ("6×6 K=4", 6, 4, 8), ("9×9 K=5", 9, 5, 15)]
    state = GameState(N, K)
    rng = MersenneTwister(42)
    for _ in 1:min(n_stones, N*N-1)
        moves = collect(available_moves(state))
        isempty(moves) && break
        idx = rand(rng, 1:length(moves))
        state = place_stone(state, moves[idx][1], moves[idx][2])
    end
    
    println("\n  $label board:")
    b = @benchmark greedy_policy_distribution($state) samples=50 evals=3
    println("    greedy_policy_distribution: $(BenchmarkTools.prettytime(median(b.time)))  ($(b.allocs) allocs)")
    
    b2 = @benchmark defensive_policy_distribution($state) samples=50 evals=3
    println("    defensive_policy_distribution: $(BenchmarkTools.prettytime(median(b2.time)))  ($(b2.allocs) allocs)")
    
    b3 = @benchmark positional_policy_distribution($state) samples=30 evals=2
    println("    positional_policy_distribution: $(BenchmarkTools.prettytime(median(b3.time)))  ($(b3.allocs) allocs)")
end

println("\n" * "-"^80)
println("  PART 4: Real game play (full game vs random opponent)")
println("-"^80)

function play_game(opp_level)
    N, K = 6, 4
    state = GameState(N, K)
    opp = make_opponent(opp_level, K)
    rng = MersenneTwister(42)
    while !game_over(state)
        if player_turn(state) == 1
            move, _ = opp(state)
            state = place_stone(state, move[1], move[2])
        else
            moves = available_moves(state)
            isempty(moves) && break
            state = place_stone(state, rand(rng, moves)[1], rand(rng, moves)[2])
        end
    end
    return check_game_result(state)
end

for level in [:greedy, :defensive, :positional]
    b = @benchmark play_game($level) samples=20 evals=1
    println("  $level vs random: $(BenchmarkTools.prettytime(median(b.time)))  ($(b.allocs) allocs)")
end

println("\n" * "="^80)
println("  BENCHMARK COMPLETE")
println("="^80)