### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 9a4d0c70-ca15-4201-8a2e-56af95a60290
using PlutoDevMacros, Base.Threads

# ╔═╡ c77a29da-a2a4-4956-9795-56ce63337495
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI, PlutoPlotly, LaTeXStrings, PlutoProfile, HypertextLiteral, ProgressLogging, BenchmarkTools
	TableOfContents()
end
  ╠═╡ =#

# ╔═╡ e7c4aad1-562f-4027-b64b-39859ac8abbb
md"""
# Tic Tac Toe Environment

This state space is small enough that it is also possible to solve with effectively tabular techniques.  To make the solution more efficient consider the inherent symmetries in the problem.  In particular, the value of game states should be invariant with respenct to d4 symmetries which include: mirror reflection across horizontal, vertical and both diagonal axes, 90, 180, and 270 degree rotations.  So each board and the 7 transformed versions should be treated the same.

Each of the 9 elements of the board can contain an X, O, or nothing.  To encode a board into a unique integer, I will use a ternary encoding.  Associate each cell state with an integer: 0 = nothing, 1 = X, and 2 = O.  A board can then be represented with a 9 digit value where each digit is 0, 1, 2 and this vector is equivalent to a ternary number calculated as follows: `sum(v[i]*3^(i-1) for v in boardvector)`.  So to get a list of unique states, apply all the symmetry operations to a board, encode each board as an integer, select the transformation with lowest integer.  This procedure can be performed for every possible board and a mapping between those and the unique states can be saved for future use.  

Additional filters for valid boards should include ensuring that the O count either 1 less than the X count or equal to it.  Also boards where more than one player has 3 in a row are invalid.
"""

# ╔═╡ 00e79eb9-93aa-4afc-b665-aa410a1d3a9b
md"""
## Game Setup
"""

# ╔═╡ 52e59796-5eef-4630-98b8-65e62429abee
const d4_symmetries = SVector{9, UInt8}.([
		[1, 2, 3, 4, 5, 6, 7, 8, 9], #identity
		[3, 2, 1, 6, 5, 4, 9, 8, 7], #x axis flip
		[7, 8, 9, 4, 5, 6, 1, 2, 3], #y axis flip
		[7, 4, 1, 8, 5, 2, 9, 6, 3], #90 degree rotation
		[9, 8, 7, 6, 5, 4, 3, 2, 1], #180 degree rotation
		[3, 6, 9, 2, 5, 8, 1, 4, 7], #270 degree rotation
		[9, 6, 3, 8, 5, 2, 7, 4, 1], #diagonal flip 1
		[1, 4, 7, 2, 5, 8, 3, 6, 9] #diagonal flip 2
])

# ╔═╡ 5ffdff02-3de1-4761-beb3-b701501a6fc3
#indices to transform back after doing symmetry operation
const d4_inverted = [SVector{9, UInt8}(findfirst(==(i), v) for i in 1:9) for v in d4_symmetries]

# ╔═╡ be93b217-b9de-4183-a6d3-dd4574808597
const BoardTTT = SVector{9, UInt8}

# ╔═╡ 38c5c0a3-1605-4c43-8d8a-efd1d8e286c5
#if a player has claimed any of these inds then the game is over
const winning_inds = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (1, 4, 7), (2, 5, 8), (3, 6, 9), (1, 5, 9), (3, 5, 7))

# ╔═╡ 83509b3d-3acf-4b0e-b661-448f2131c906
const ttt_moves = SVector{9}(UInt8.(1:9))

# ╔═╡ 8fe13e56-a6ec-4ee9-bbca-7df6ef410e73
const X_VAL = 0x01

# ╔═╡ a10a69c9-7a16-4035-8821-e0f82fb24d1b
const O_VAL = 0x02

# ╔═╡ 97eb046f-e053-4547-a8ec-e763e8a025e1
const EMPTY = 0x00

# ╔═╡ 0618bd90-e6fc-468b-bb9b-219de9e32613
#check if a player associated with a given value has won on a given board
val_win(board::BoardTTT, val::UInt8) = any(all(board[i] == val for i in inds) for inds in winning_inds)

# ╔═╡ cd542047-0eb4-40df-9d2f-1f7830b7f817
val_win(board, val) = val_win(BoardTTT(board), UInt8(val))

# ╔═╡ b6aa747d-56d5-43b1-aff8-adf59c91ce66
x_win(board) = val_win(board, X_VAL)

# ╔═╡ f2f5f23a-8484-4ea3-b7dc-1f1d5d64ba60
o_win(board) = val_win(board, O_VAL)

# ╔═╡ 141b0b0b-28c5-48d7-962e-c203305a0565
canmove(board::BoardTTT, m::Integer) = board[m] == 0x00

# ╔═╡ 293d5ea2-342b-4005-939a-5b924282f83f
canmove(board, m) = canmove(BoardTTT(board), m)

# ╔═╡ 3993d911-73cb-49db-9478-c4c6eb9f4ce9
valid_moves(board::BoardTTT) = board .== 0 

# ╔═╡ 6c68868a-30f0-448b-8a42-84ce586145c7
valid_moves(board) = valid_moves(BoardTTT(board))

# ╔═╡ 4926525c-1584-4ce0-84a4-2d1d7bad8b9b
is_term(board) = !any(==(0), board)

# ╔═╡ 6705c31c-f92f-48ad-b0f4-274fce082359
is_winner(board) = x_win(board) || o_win(board)

# ╔═╡ ad4a926d-6238-4dcd-85c9-5184a66a91f1
is_draw(board) = is_term(board) && !is_winner(board)

# ╔═╡ 97a8c4a3-6191-4c34-bdfc-8d8f6a06e33a
is_active(board) = !is_term(board) && !is_winner(board)

# ╔═╡ 26a5e525-d69b-4e84-aa14-a3d273ae3333
#determine if it is O's turn to move because the board sum should be 1 off from a multiple of 3
function is_o_move(board::BoardTTT)
	s = sum(board)
	Bool(s % 0x0003)
end

# ╔═╡ 79108d50-064a-4eba-9044-53b2ca6f6ac6
is_o_move(board) = is_o_move(BoardTTT(board))

# ╔═╡ 99c1e2e6-743b-4112-baae-1360a8b39a5d
#check if a board is valid, i.e. can be reached during normal play where X starts, players alternate and the game ends after the first player gets 3 in a row
function isvalid(board::BoardTTT)
	winners = NamedTuple((Symbol(f), f(board)) for f in (x_win, o_win))
	#cannot have both x and o winning
	all(winners) && return false
	xnum = count(==(X_VAL), board)
	onum = count(==(O_VAL), board)
	#O count must be equal to or one less than X count
	!(0 <= xnum - onum <= 1) && return false
	#if O wins then the X count must be equal to the O count because if it is one greater then X played a move after O won
	winners.o_win && (xnum != onum) && return false
	#if X wins then it must have a count greater than O because otherwise O would have gone after X wins
	winners.x_win && (xnum == onum) && return false
	
	#in all other cases the board is fine
	return true
end

# ╔═╡ 6545da33-7250-42a9-b1a2-77cd8cc5f71f
isvalid(board) = isvalid(BoardTTT(board))

# ╔═╡ e84b32f3-dd81-4dbe-a80d-c198a08e0041
const score_functions = (x_win, o_win, is_draw, is_active)

# ╔═╡ a229d28f-2026-457f-8236-8002f9caa150
#list of functions to compute all relevant properties of a board
const status_functions = (score_functions..., is_o_move, valid_moves)

# ╔═╡ 62bd6663-101d-45c6-8aad-0425da307d3a
#rewards associated with arriving at a board with the following conditions for the X player.  rewards for the O player will be negative of this.  The value of draw differing from 0 is so it can be distinguished from an active board.  Also under these rewards a state with equal probability of win and loss would be 0 whereas a state with an expected draw would be valued at -0.5.
const rewardsX = NamedTuple(zip(Symbol.(score_functions), (1f0, -1f0, -0.5f0, 0f0)))

# ╔═╡ 5bd479bd-35a5-4ff8-b534-95bd628c9af9
const rewardsX_alt = NamedTuple(zip(Symbol.(score_functions), (1f0, -0.5f0, -0.5f0, 0f0)))

# ╔═╡ 02237d67-ada1-488e-bfff-f5f51fca757d
const BoardStatus = NamedTuple{Symbol.(status_functions)}

# ╔═╡ cee75d80-e58c-475e-ade5-e3c609a2fe24
#check a board and return game status of each check
get_board_status(board::BoardTTT) = NamedTuple((Symbol(f), f(board)) for f in status_functions)

# ╔═╡ 6a6ba736-df06-4049-a1b5-fb5b2a3e3bb5
#attempt to convert a different type to a valid board if possible
get_board_status(board) = get_board_status(BoardTTT(board))

# ╔═╡ bc259929-d37b-441b-af3d-a0f56f6e7387
#reward associated with arriving at a new board from the perspective of the x player, not that for valid boards only one of the values in status will be true so this will produce a value for invalid boards even though it isn't well defined
get_reward_x(status::BoardStatus; rewards = rewardsX) = sum(rewards[k]*status[k] for k in keys(rewards))

# ╔═╡ 8678870f-3a3a-4293-8295-a53b08cc1549
get_reward_x(board; kwargslll) = get_reward_x(get_board_status(board); kwargs...)

# ╔═╡ 2c17bef9-da02-4c9a-8aff-72176ed3ec10
get_reward_o(args...; kwargs...) = -get_reward_x(args...; kwargs...)

# ╔═╡ d5917084-6b70-4158-a32d-a2a9d375ac8a
#get reward for a board assuming the desired perspective is the player with the available move
get_reward(status::BoardStatus; kwargs...) = (1 - 2*status.is_o_move) * get_reward_x(status; kwargs...)

# ╔═╡ 857d205e-ff6d-4709-9b2e-f010bee240d1
get_reward(board; kwargs...) = get_reward(get_board_status(board); kwargs...)

# ╔═╡ a514f581-b3ec-41dd-a783-d60fbc56c255
# convert a board representation as a vector to an integer using powers of 3, need to use UInt16 here to have enough states.  Optionally permute the indices to calculate the state of a transformed board
mapboard(v::BoardTTT; inds = eachindex(v)) =  mapreduce(a -> v[last(a)]*0x0003^(first(a)-1), +, enumerate(inds)) 

# ╔═╡ 830cfb4b-2bc5-425a-b9b6-a8b6fd0ced63
mapboard(v; kwargs...) = mapboard(BoardTTT(v); kwargs...)

# ╔═╡ 461a1484-3fa3-42f3-a26c-8949117e6b44
# convert a number to a board representation vector
map_ttt_state(n::UInt16) = BoardTTT(digits(n, base = 0x03, pad=9))

# ╔═╡ 5b744d12-e0af-4864-9409-3927bfb700d6
map_ttt_state(n) = map_ttt_state(UInt16(n))

# ╔═╡ e25a0c4b-0a84-420d-a7e3-f18a7014aa2d
const unfiltered_ttt_boards = (map_ttt_state(n) for n in 0:(3^9-1))

# ╔═╡ 70b2d2db-bafc-47bb-ba32-28449ec1dd9e
const valid_ttt_boards = (b for b in unfiltered_ttt_boards if isvalid(b))

# ╔═╡ 58f14689-1add-48f3-b1d9-bab6d05ac992
#lookup table for getting board from a numerical state representation
const ttt_state_lookup = Dict(mapboard(b) => b for b in valid_ttt_boards)

# ╔═╡ 40bf9e7e-67d5-486c-9fbd-f17f4cdba947
#convert a board to its symmetry equivalent version and the index of the symmetry transformation used
function get_symmetric_board(board::BoardTTT)
	#only keep the board with the lowest state value
	(smin, imin) = findmin(inds -> mapboard(board; inds = inds), d4_symmetries)
	inds = d4_symmetries[imin]
	(BoardTTT(view(board, inds)), imin)
end

# ╔═╡ 31538c97-d99c-43db-8a32-dbc6e2726920
get_symmetric_board(board) = get_symmetric_board(BoardTTT(board))

# ╔═╡ b375c7eb-3b4c-4dff-af90-e6814b19cd34
#map a board to it's symmetric equivalent with the permutation indices
const symmetric_board_lookup = Dict(b => get_symmetric_board(b) for b in valid_ttt_boards)

# ╔═╡ de2d1f87-1530-4d31-9d8c-9af007a44bec
const symmetric_boards = unique(first(a) for a in values(symmetric_board_lookup))

# ╔═╡ 01a099ff-c9cf-44d5-827a-f8d587871dcb
const symmetric_board_index = makelookup(symmetric_boards)

# ╔═╡ fa1ee379-3476-4636-83b4-3af562f52050
#precompute the status of unique boards only
const ttt_status_lookup = Dict(b => get_board_status(b) for b in symmetric_boards)

# ╔═╡ 52ee5e7b-19f4-4017-a803-ead3fd45b082
function lookup_board_status(board::BoardTTT)
	sym_board, isym = symmetric_board_lookup[board]
	(status = ttt_status_lookup[sym_board], isym = isym)
end

# ╔═╡ 85dfb79c-1a06-45a6-af11-26b688ea0b3e
lookup_board_status(board) = lookup_board_status(BoardTTT(board))

# ╔═╡ d0d5b687-cff4-48e4-ac6c-bf19d866aaca
const active_ttt_boards = filter(b->ttt_status_lookup[b].is_active, symmetric_boards)

# ╔═╡ 835ee149-b79b-4293-ba1f-714aa76eb141
const active_ttt_board_indices = [symmetric_board_index[b] for b in active_ttt_boards]

# ╔═╡ 844d31b2-373e-48ef-882d-eb73e6b10391
const active_ttt_board_bits = BitVector(ttt_status_lookup[b].is_active for b in symmetric_boards)

# ╔═╡ 7b807838-a724-4f18-b90c-e6cfd52f38aa
const active_x_boards = filter(b -> !ttt_status_lookup[b].is_o_move, active_ttt_boards)

# ╔═╡ b823c99e-fd6e-46dd-bc24-f080befdf922
const active_x_board_indices = [symmetric_board_index[b] for b in active_x_boards]

# ╔═╡ 287d846d-4eba-4745-adaa-613c638262da
const active_x_board_bits = BitVector(!ttt_status_lookup[b].is_o_move for b in symmetric_boards)

# ╔═╡ d78c7ca7-7125-4547-804a-a8cd193746ed
const active_o_boards = filter(b -> ttt_status_lookup[b].is_o_move, active_ttt_boards)

# ╔═╡ a93fb40b-7eda-48f6-a2ea-4afce95140a0
const active_o_board_indices = [symmetric_board_index[b] for b in active_o_boards]

# ╔═╡ 216b2401-a416-4a6e-9812-7ef7061b55b9
const active_o_board_bits = BitVector(ttt_status_lookup[b].is_o_move for b in symmetric_boards)

# ╔═╡ 36deaa30-0485-4012-a838-1011161e7a3e
const inactive_boards = filter(b -> !ttt_status_lookup[b].is_active, symmetric_boards)

# ╔═╡ 81ae3b51-b666-4c51-b17b-32bea5b99357
const active_board_indices = [symmetric_board_index[b] for b in inactive_boards]

# ╔═╡ ad943e14-28a1-406c-9f98-d8572dfa16a6
const inactive_board_bits = BitVector(ttt_status_lookup[b].is_active for b in symmetric_boards)

# ╔═╡ d52f66cd-c4c2-4750-b182-42e08a9a27f4
md"""
## MDP Definitions
"""

# ╔═╡ b1df7a1e-5319-41d2-b8cd-ca25d55fe1ea
function π_random_ttt!(action_probabilities::Vector{T}, s::BoardTTT) where T<:Real
	n = zero(T)
	for i in eachindex(s)
		x = T(iszero(s[i])) #check if action is valid
		action_probabilities[i] = x
		n += x
	end
	action_probabilities ./= n
end

# ╔═╡ 14b5a3cd-31d9-4392-bbb8-25286bcc2ff8
π_random_ttt(s::BoardTTT; action_probabilities::Vector{T} = zeros(Float32, 9)) where T<:Real = π_random_ttt!(action_probabilities, s) 

# ╔═╡ d90008a6-e2eb-4e1e-b9ce-9a909ff0325f
symmetric_boards[760]

# ╔═╡ 11e24ec6-4222-425b-afc1-5d44948b2392
make_ttt_tabular_mdp(ptf::TabularStochasticTransition) = TabularMDP(symmetric_boards, UInt8.(1:9), ptf, () -> 760)

# ╔═╡ 109a9bfd-92bb-4307-9d3e-0bfb737082ab
const ttt_value_γ = 1f0

# ╔═╡ d92d52c1-d38a-4b03-9c59-e542c2bdde13
#next step is to see when the values also converge even if the policy remains unchanged

# ╔═╡ 9cf13709-615b-4183-8397-c146f7f91252
function get_symmetric_index(board::AbstractVector{I}) where I<:Integer
	sym_board, rot = get_symmetric_board(board)
	sym_index = symmetric_board_index[sym_board]
	(index = sym_index, rot_inds = d4_inverted[rot])
end

# ╔═╡ ae128a72-bf33-40eb-8a47-3b6569a3af7b
function make_π_value_iter(x_results, o_results)
	function π(b)
		board_index, rot_inds = get_symmetric_index(b)
		status = get_board_status(b)
		results = if status.is_o_move
			o_results
		else
			x_results
		end
		results.optimal_policy[:, board_index]
	end
end

# ╔═╡ 871202dd-4fe5-4a04-b11b-02534c729d8c
get_symmetric_index(symmetric_boards[5])

# ╔═╡ 2180c6dc-1f6d-4976-bd0f-cc8153e6b87d
#what is the ϵ greedy policy for ttt for different values and how does it differ from greedy

# ╔═╡ 9d2120cd-5582-4fa6-9a52-bad1121d9da3
# ╠═╡ disabled = true
#=╠═╡
const value_iter_board1_result = value_iter_board1_status.is_o_move ? value_iter_o_vs_rand : value_iter_x_vs_rand#value_iter_o_vs_iter4 : value_iter_x_vs_iter4
  ╠═╡ =#

# ╔═╡ 7479eb78-0aab-4a28-b433-968aec5b980b
d4_symmetries[8]

# ╔═╡ 4016eb02-ff4c-46dc-8167-d8c2a05f4a1f
value_iter_test.final_value[241]

# ╔═╡ 45be8610-a480-435c-b4e4-33aff7f3aead
#check to see if this policy makes invalid moves in any of the states, don't care about terminal states though since all actions are valid and do nothing

# ╔═╡ 74c3970f-49a7-4e35-bcb0-e4fe53c9cd58
struct TTTEnvironment{T, V}
	init_board::BoardTTT
	term_board::BoardTTT
	move::T
	apply_π::V
end

# ╔═╡ ec450941-30ec-4bd1-8628-330760287e5f
function make_ttt_environment()
	#the most straightforward board representation is a 3x3 matrix of a ternary value.  We could represent this with 2 bits that can take on 1 of 4 values so it would be one more value than is necessary.  With this representation an unocupied cell is 00, x cell is 01, and o cell is 10 with 11 being ignored.  We could use 2 bit matricies for this with each matrix representing the occupied positions of x and o respectively.  This could also be compressed down to a single number 9 bits long.  It would be nice to just use a 8 bit number though because that is a fundamental datatype UInt8.  Maybe we can ignore the last number because we'd never have a situation where every state was filled up by a single mark but these bits represent whether the mark is present in a given cell so we'd have some unintuitive mapping if we force ourselves to use UInt8.  We could also just use a vector or even static array of length 9.  The other approach is to generate all 3^9 possible boards and just have a lookup table from that maps a given board to one of those numbers.  We could do that by having 0, 1, 2 in each position and then calculating the ternary value of that.  For example let's say we have the following board where the cells are shown one row at a time [0 0 0; 0 1 0; 0 0 2].  This would map to 3^5 + 2*3^9.  

	init_board = BoardTTT(fill(0x00, 9))
	term_board = BoardTTT(fill(0x03, 9))

	#return a new board state after a move a where a should be the square where a mark is placed as a number from 1 to 9.
	function move(board::BoardTTT, a::UInt8)
		#if an illegal move is attempted 
		board[a] != 0x00 && begin @info "illegal move $a on board $board"; return term_board end
		#value to be filled into the board, 1 for X moves and 2 for O moves
		fillmove = 0x0001 + UInt8(lookup_board_status(board).status.is_o_move)
		state = mapboard(board) #convert board to integer to calculate new values and perform lookup
		newstate = state + (fillmove * 0x003^(a-0x0001)) #calculate new state
		newboard = ttt_state_lookup[newstate] #get new board from lookup table
	end

	move(board, a) = move(BoardTTT(board), UInt8(a))

	#take a policy π that is only defined for unique boards and calculate the action to take converting symmetries back to original board
	function apply_π(π, board::BoardTTT)
		(symboard, isym) = symmetric_board_lookup[board]
		prbs = copy(π(symboard))
		prbs[d4_inverted[isym]]
	end	

	TTTEnvironment(init_board, term_board, move, apply_π)
end

# ╔═╡ 1d7fccfb-10d3-47ae-a8f6-55d4a81344da
const ttt_environment = make_ttt_environment()

# ╔═╡ 91e35000-afb7-4170-9923-4ea25b157ae2
function get_random_move(board::BoardTTT)
	status, isym = lookup_board_status(board)
	wsample(ttt_moves, view(status.valid_moves, d4_inverted[isym]))
end

# ╔═╡ 90255ba0-304e-4ca8-a1e7-28bfc30ca0cd
get_random_move(board) = get_random_move(BoardTTT(board))

# ╔═╡ b1a5a634-9673-4c72-9a71-4da9dba8ea83
#clean up possible issues in softmax caused by infinite and undefined values
function clean_output!(v::AbstractVector{T}) where T <: AbstractFloat
	for (i, x) in enumerate(v)
		if isnan(x) || isinf(x)
			v[i] = zero(T)
		end
	end
	return v
end

# ╔═╡ af40d78d-4860-4f56-9ce1-ba4e631f7ffd
#move on a board but return the symmetric version
symmetric_move(board, m) = first(symmetric_board_lookup[ttt_environment.move(board, m)])

# ╔═╡ be80db21-c043-4a9b-8473-f2ab4a1e04ce
#take a step but map boards to symmetric versions and any inactive board maps to the terminal state. defaults to calculating rewards from the x player perspective
function ttt_step(board, m; reward_func = get_reward_x, kwargs...)
	newboard = symmetric_move(board, m)
	(status, isym) = lookup_board_status(newboard)
	r = reward_func(status; kwargs...)
	(newboard, r, status.is_active)
	# finalboard = status.is_active ? newboard : ttt_environment.term_board
	# (finalboard, r, status.is_active)
end

# ╔═╡ 5c59c4a2-e4d7-4ed7-9daa-10a12f5378c2
#define step for a player against an opponent
function ttt_step(board::BoardTTT, m::UInt8, get_opponent_action::Function; kwargs...)
	(newboard, r, is_active) = ttt_step(board, m; kwargs...)
	!is_active && return (newboard, r)
	m2 = get_opponent_action(newboard)
	ttt_step(newboard, m2; kwargs...)[[1, 2]]
end

# ╔═╡ 68b8fd79-09f9-43c5-bf53-b5b875d63c3f
function make_ttt_ptfs(π_opponent::Function; kwargs...)
	state_transition_map_x = Matrix{SparseVector{Float32, Int64}}(undef, 9, length(symmetric_boards))
	reward_transition_map_x = Matrix{Vector{Float32}}(undef, 9, length(symmetric_boards))
	state_transition_map_o = Matrix{SparseVector{Float32, Int64}}(undef, 9, length(symmetric_boards))
	reward_transition_map_o = Matrix{Vector{Float32}}(undef, 9, length(symmetric_boards))

	for i in eachindex(symmetric_boards)
		b = symmetric_boards[i]
		for i_a in 1:9
			state_transitions = SparseVector(zeros(Float32, length(symmetric_boards)))
			rewards = SparseVector(zeros(Float32, length(symmetric_boards)))
			if ttt_status_lookup[b].is_active
				if iszero(b[i_a])
					(s′, r1, active) = ttt_step(b, i_a; kwargs...)
					if active
						prbs = π_opponent(s′)
						for i_a2 in findall(!iszero, prbs)
							(s′′, r2, active2) = ttt_step(s′, i_a2; kwargs...)
							state_transitions[symmetric_board_index[s′′]] += prbs[i_a2]
							rewards[symmetric_board_index[s′′]] += (r1 + r2)*prbs[i_a2]
						end
						for i_s′ in state_transitions.nzind
							rewards[i_s′] /= state_transitions[i_s′]
						end
					else
						state_transitions[symmetric_board_index[s′]] = 1f0
						rewards[symmetric_board_index[s′]] = r1
					end
				end
				reward_output = rewards[state_transitions.nzind]
				
				if ttt_status_lookup[b].is_o_move
					state_transition_map_o[i_a, i] = state_transitions
					reward_transition_map_o[i_a, i] = -reward_output
					state_transition_map_x[i_a, i] = SparseVector(zeros(Float32, length(symmetric_boards)))
					reward_transition_map_x[i_a, i] = Vector{Float32}()
				else
					state_transition_map_x[i_a, i] = state_transitions
					reward_transition_map_x[i_a, i] = reward_output
					state_transition_map_o[i_a, i] = SparseVector(zeros(Float32, length(symmetric_boards)))
					reward_transition_map_o[i_a, i] = Vector{Float32}()
				end
			else
				state_transitions[i] = 1f0
				reward_output = [0f0]
				state_transition_map_o[i_a, i] = copy(state_transitions)
				reward_transition_map_o[i_a, i] = copy(reward_output)
				state_transition_map_x[i_a, i] = state_transitions
				reward_transition_map_x[i_a, i] = reward_output
			end
		end
		
	end

	ptf_x = TabularStochasticTransition(state_transition_map_x, reward_transition_map_x)
	ptf_o = TabularStochasticTransition(state_transition_map_o, reward_transition_map_o)
	(ptf_x, ptf_o)
end

# ╔═╡ fef5f0c4-46d7-477c-a508-3c221129cc60
function make_x_o_tabular_mdps(x_results, o_results; kwargs...)
	π = make_π_value_iter(x_results, o_results)
	ptfs = make_ttt_ptfs(π; kwargs...)
	Tuple(make_ttt_tabular_mdp(ptf) for ptf in ptfs)
end

# ╔═╡ cb12f8b6-cec1-40fe-9b29-0acae1d3291b
md"""
# Solution Techniques
"""

# ╔═╡ 16ebebb0-fe2f-4469-a834-8b7375cdb0dd
md"""
## Fixed Opponent

If we specific an opponent, then we can treat the game as an MDP environment with the states matching those when one of the player is to move.  We could alternate training each player and improve that way or just stop at the optimal strategy vs the fixed opponent.
"""

# ╔═╡ 37f279ed-c1d6-446f-9e63-89553ad4f721
md"""
### Value Function Techniques
"""

# ╔═╡ dec67e3a-d798-48c8-b58a-9cfdfd057d36
md"""
#### Policy Iteration
"""

# ╔═╡ b8044db9-73af-4834-948d-9c2a800d251d
md"""
#### Value Iteration
"""

# ╔═╡ 1aa7f969-7382-4ec4-99ad-af9a7c8f8354
md"""
#### Sampling with Eligibility Traces
"""

# ╔═╡ aef1ba88-8b77-4f12-a88c-56ec3e6bdbc7
md"""
### Policy Gradient Techniques
"""

# ╔═╡ c180b1a8-9540-4da7-9ba8-932c8d48b8f2
md"""
#### Reinforce
"""

# ╔═╡ 9701cad5-7897-40ed-a3b4-9e9ead8bd790
md"""
#### Actor-Critic with Eligibility Traces
"""

# ╔═╡ e65a55a2-a7ba-4c65-82f7-d3c04f168ccc
md"""
### Planning Techniques
"""

# ╔═╡ 15d3b0d1-a6a5-4524-a07a-5dd2580aac92
md"""
#### Dyna-Q
"""

# ╔═╡ 6894f2a4-b844-434b-80cf-03b52558e043
md"""
#### Trajectory Sampling
"""

# ╔═╡ 3a84f6ff-e6ac-42a5-8939-420430defeb2
md"""
#### Tree Search
"""

# ╔═╡ 88de5309-557b-4ba9-b627-90495d769747
md"""
## Self-Play

If we allow all game states for both players to exist, then the learning technique must change how the value function is defined depending on whose move it is.  The value function estimates the value of a state from the perspective of one of the players, and the states associated with that player will be handled as usual.  For the other set of states, the goal of maximizing the value fuction must be reversed to minimizing it.  That way both players can share the same value function and be optimized simultaneously.
"""

# ╔═╡ 2e454bf5-62fc-46fc-8076-74f1a6729fa7


# ╔═╡ 0d3b384b-c602-4ccd-b335-987e378a4230
md"""
## Actor-Critic vs Fixed Opponent
"""

# ╔═╡ 90240ffc-fd88-44e6-9b6c-367361326996
struct ActorCriticTTTAgent{Vest, Vgrad, Pfunc, Pgrad}
	v̂::Vest
	∇v̂::Vgrad
	π!::Pfunc
	∇lnπ!::Pgrad
	θ::Matrix{Float64}
	w::Vector{Float64}
	πoutput::Vector{Float64}
	∇output::Matrix{Float64}
end

# ╔═╡ 2459ba2e-b0ba-4271-9b3e-7cb0e2b847fa
#setup estimation functions for a player given a set of valid playable states for that player.  For example to create an X player, only valid X states should be selected and the corresponding step function should only produce those states
function setup_ttt_player(states::AbstractVector{T}) where T <: BoardTTT
	#convert states to index
	statelookup = Dict(zip(states, eachindex(states)))
	statelookup[ttt_environment.term_board] = lastindex(states) + 1

	#create state feature vectors, leave the terminal state at all zeros
	xs = [zeros(lastindex(states)+1) for i in 1:(lastindex(states)+1)]
	for i in eachindex(states)
		xs[i][i] = 1.0
	end

	#value function and gradient
	v̂(s::BoardTTT, w) = w[statelookup[s]]
	∇v̂(s::BoardTTT, w) = xs[statelookup[s]]
	#allocations for outputs
	πoutput = zeros(lastindex(ttt_moves))
	∇output = zeros(lastindex(states)+1, lastindex(ttt_moves))

	#policy function and gradient
	function π!(s::BoardTTT, θ::Matrix)
		πoutput .= view(θ, statelookup[s], :)
		πoutput .+= ((s .!= 0x00) .* -Inf) #set output preference to -Inf for occupied cells
		soft_max!(πoutput)
		clean_output!(πoutput)
	end

	#under the convension that we always use the x player reward for the value estimate, to get a valid policy for the o player we can reverse the gradient direction for board states on which the o player is taking a turn.  That way both players can use the same value function
	function ∇lnπ!(a::UInt8, s::BoardTTT, θ::Matrix)
		π!(s, θ)
		i = statelookup[s]
		f = 1.0 - (2.0 * lookup_board_status(s).status.is_o_move) #reverse policy gradient for o player
		for n in ttt_moves
			@inbounds @simd for m in eachindex(states)
				#apply gradient for soft-max but noticing all values are 0 for i != m which corresponds to other states
				∇output[m, n] = f * (i == m) * ((n == a) - πoutput[n])
			end
		end
		return ∇output
	end

	∇lnπ!(a, s, θ) = ∇lnπ!(UInt8(a), BoardTTT(s), θ)

	#parameters
	θ = zeros(lastindex(states)+1, lastindex(ttt_moves))
	w = zeros(lastindex(states)+1)

	#note that because there are internal allocated outputs for the policy and the gradient a new instance of this should be generated each time a learning procedure is done.  it may be better design to explicitely pass these holders into any running function so there's always a new copy
	ActorCriticTTTAgent(v̂, ∇v̂, π!, ∇lnπ!, θ, w, πoutput, ∇output)
end	

# ╔═╡ b795d8a3-9b54-44d9-9ce6-112651f39cd1
x_step_vs_random(board, move) = ttt_step(board, move, get_random_move)

# ╔═╡ e3d6ff52-4084-47b1-aeb1-d36a161882b0
#=╠═╡
@bind avgeps Slider(100:10000, show_value=true)
  ╠═╡ =#

# ╔═╡ 5b8adffa-0ef5-4bcd-95c3-01e62f6c6186
md"""
Compare these three policies on a single board state
"""

# ╔═╡ a417ae4c-7b97-442e-895b-ddd5eaa7f4e6
run_ttt_game(πx, πo) = run_ttt_game(πx, πo, [ttt_environment.init_board], Vector{UInt8}(), Vector{UInt8}())

# ╔═╡ fb917755-0db1-436c-9a07-3e1529d6c144
#play a game between two different policies for the x and o player
function run_ttt_game(πx::Function, πo::Function, board_history::Vector{BoardTTT}, xturns::Vector{UInt8}, oturns::Vector{UInt8})
	board = last(board_history)
	status = lookup_board_status(board)
	#if the board is no longer active then end the game
	!status.status.is_active && return (board_history, status, xturns, oturns)
	xmove = πx(board) #select move for x player
	push!(xturns, xmove)
	board′ = ttt_environment.move(board, xmove)
	push!(board_history, board′)
	status′ = lookup_board_status(board′)
	#if the board is no longer active then end the game
	!status′.status.is_active && return (board_history, status′, xturns, oturns)
	omove = πo(board′) #select move for o player
	push!(oturns, omove)
	board′′ = ttt_environment.move(board′, omove)
	push!(board_history, board′′)
	run_ttt_game(πx, πo, board_history, xturns, oturns)
end

# ╔═╡ c6dec053-7407-45ca-93fd-f0b893bb1345
function get_ttt_matchup_statistics(πx::Function, πo::Function; trials = 100_000)
	wld = 1:trials |> Map(n -> run_ttt_game(πx, πo)[2].status[(:x_win, :o_win, :is_draw)]) |> collect
	NamedTuple(outcome => count(a[outcome] for a in wld)/trials for outcome in (:x_win, :o_win, :is_draw))
end

# ╔═╡ a4f13c4a-152e-4aef-85a8-874f0cd0abaa
compare_ttt_policies(p1::Function, p2::Function; kwargs...) = get_ttt_matchup_statistics(p1, p2; kwargs...)

# ╔═╡ 45c9a02d-a7ce-49ab-b085-44bd40090b29
nrounds = 10

# ╔═╡ d4de69bd-f3b4-4cf5-a3c7-3a95d28052c6
#=╠═╡
md"""
Round:
$(@bind roundcount Slider(1:nrounds, show_value=true))

Player:
$(@bind playerselect Select([1 => "X", 2 => "O"]))
"""
  ╠═╡ =#

# ╔═╡ eaa5da0c-a2a1-4c87-9a83-791dacd67765
get_ttt_matchup_statistics(get_random_move, get_random_move)

# ╔═╡ 0d0e1329-ee71-46a7-a998-7c6a2cfa8e32
#=╠═╡
function optimize_λ(αθlist, αwlist, opt_setup; epavg = 100, nruns = nthreads(), λlist = [0.0, 0.1, 0.2, 0.4, 0.8, .9], kwargs...)
	function maketrace(αθ, αw) 
		@info "running for αθ = $αθ and αw = $αw"
		@progress rewards = [begin
			out = average_runs((;kwargs...) -> execute_actor_critic(opt_setup, αθ, αw; kwargs...), nruns; λθ = λ, λw = λ, kwargs...) 
			mean(out[max(1, end-epavg):end])
		end
		for λ in λlist]
		scatter(x = λlist, y = rewards, name = "αθ = $αθ, αw = $αw")
	end

	params = [(a, b) for a in αθlist for b in αwlist]
	@progress traces = [maketrace(p...) for p in params]
	plot(traces, Layout(xaxis_title = "λ", yaxis_title = "Average Reward Last $epavg Episodes", width = 900, height = 600))
end
  ╠═╡ =#

# ╔═╡ a4855984-90c8-4044-9358-9ea5748772eb
function showboard(board::AbstractVector)
	function f(n::Integer)
		n == 0 && return '-'
		n == 1 && return 'X'
		return 'O'
	end
	mapreduce(inds -> f.(board[inds]), vcat, [[1 2 3], [4 5 6], [7 8 9]])
end		

# ╔═╡ 3307f5b4-0a49-4895-b6df-decdbbde28ba
showboard(boardstate::UInt16) = boardstate == typemax(UInt16) ? "Terminal State" : showboard(mapstate(boardstate))

# ╔═╡ 38d7c4b4-eaac-4e92-9d94-0b833ee50c3b
md"""
## Value Iteration

For the previous two environments, value iteration was not feasible because defining the probability transition function was very inconvenient or impossible.  However for the tic tac toe game it may be possible assuming that the opponent is pursuing the same greedy policy as the player.  Alternatively we can train value iteration against the random policy which could very well find the same optimal strategy as playing against an optimal opponent.  To make the problem more tractable we will only consider states that are unique in terms of symmetries and use the mapping functions to enforce every state in our lookup is a symmetry mapped version.
"""

# ╔═╡ bc5c2d93-24f6-4ee5-ae1a-823229ee7c5d
function make_ttt_ptf(boards, π_opponent)
	function get_opponent_transitions(board, s, a)
		prbs = π_opponent(board)
		#add up probabilities for each transition accumulating them if the ending state is equivalent
		mapreduce(mergewith(+), keys(prbs)) do i
			(s′, r, active) = ttt_step(board, i)
			Dict((s′, r, s, a) => prbs[i])
		end
	end

	function get_transitions(board, a)
		(newboard, r, active) = ttt_step(board, a)
		!active && return Dict((newboard, r, board, a) => 1.)
		get_opponent_transitions(newboard, board, a) #if game isn't over get the transition from the subsequent move
	end

	function get_transitions(board::S) where S
		moves = findall(==(0), board)
		isempty(moves) && return Dict{Tuple{S, Float64, S, UInt8}, Float64}()
		mapreduce(mergewith(+), moves) do move
			get_transitions(board, move)
		end
	end

	#only calculate transitions from valid states for x player
	ptf = mapreduce(get_transitions, mergewith(+), boards)
	sa_keys = get_sa_keys(ptf)

	return (ptr = ptf, sa_keys = sa_keys)
end

# ╔═╡ b13134bc-9b08-435c-8c7a-9aa736c003b3
function π_random_ttt(b)
	inds = findall(==(0), b)
	v = 1/length(inds)
	Dict(i => v for i in inds)
end

# ╔═╡ 3056e679-8006-4396-8bb4-e76cfd909970
make_ttt_ptfs(π_random_ttt)

# ╔═╡ f215e88d-8029-4c54-873c-e8413f32a04b
const ttt_random_tabular_ptfs = make_ttt_ptfs(π_random_ttt)

# ╔═╡ 22406fa6-fb9b-4d9e-84d8-daa336542563
function make_x_o_tabular_mdps(;kwargs...)
	ptfs = make_ttt_ptfs(π_random_ttt; kwargs...)
	Tuple(make_ttt_tabular_mdp(ptf) for ptf in ptfs)
end

# ╔═╡ aa250079-6cc1-4f3e-acf3-a0b0bce75c7c
const x_rand_mdp, o_rand_mdp = make_x_o_tabular_mdps()

# ╔═╡ 9d9ba389-4346-4d37-a5c7-b6a6cfddaedc
const value_iter_x_vs_rand = value_iteration_v(x_rand_mdp, ttt_value_γ)

# ╔═╡ f4c9243b-ea5c-45d2-9eea-a6793e8081dd
const value_iter_o_vs_rand = value_iteration_v(o_rand_mdp, ttt_value_γ)

# ╔═╡ 21480455-a98c-4d41-91ba-08d069cb2cbf
const ttt_iter1_tabular_ptfs = make_ttt_ptfs(make_π_value_iter(value_iter_x_vs_rand, value_iter_o_vs_rand))

# ╔═╡ 2f7f4c11-ecd4-40bf-bcf1-62a63e81c68b
const test_episode_x = runepisode(x_rand_mdp; π = value_iter_x_vs_rand.optimal_policy)

# ╔═╡ 5a962d71-b56b-4166-8320-71c6baed452b
#=╠═╡
1:100_000 |> Map(_ -> runepisode(x_rand_mdp; π = value_iter_x_vs_rand.optimal_policy)[3][end]) |> tcollect |> x -> histogram(x = x) |> plot
  ╠═╡ =#

# ╔═╡ 2e57e843-0ab2-4338-b35b-afa5f606453b
const x_value_iter1_mdp, o_value_iter1_mdp = make_x_o_tabular_mdps(value_iter_x_vs_rand, value_iter_o_vs_rand)

# ╔═╡ 5fd3dfc2-3e47-474a-8b96-61459b05b4e7
const value_iter_x_vs_iter1 = value_iteration_v(x_value_iter1_mdp, ttt_value_γ)

# ╔═╡ efc0ed38-24dd-4d4a-ae44-a2dfd8bc239d
const value_iter_o_vs_iter1 = value_iteration_v(o_value_iter1_mdp, ttt_value_γ)

# ╔═╡ bc9965d2-437a-40e7-bced-30ccbb7867b7
const x_value_iter2_mdp, o_value_iter2_mdp = make_x_o_tabular_mdps(value_iter_x_vs_iter1, value_iter_o_vs_iter1)

# ╔═╡ 3ca15d74-a36d-4b45-b57c-cecd6c2a84c2
const value_iter_x_vs_iter2 = value_iteration_v(x_value_iter2_mdp, ttt_value_γ)

# ╔═╡ 494879df-0ecc-4610-9206-e5362bbf96d9
const value_iter_o_vs_iter2 = value_iteration_v(o_value_iter2_mdp, ttt_value_γ)

# ╔═╡ 68e65fd7-2885-4c28-9f58-bdc62d0fb9d5
const x_value_iter3_mdp, o_value_iter3_mdp = make_x_o_tabular_mdps(value_iter_x_vs_iter2, value_iter_o_vs_iter2)

# ╔═╡ 477452bc-12ff-4cea-bc3b-2207ec0a907d
const value_iter_x_vs_iter3 = value_iteration_v(x_value_iter3_mdp, ttt_value_γ)

# ╔═╡ a66beb65-f590-4c76-a5a8-8ba9e5829f04
const value_iter_o_vs_iter3 = value_iteration_v(o_value_iter3_mdp, ttt_value_γ)

# ╔═╡ 429bbe6d-3126-4e6f-852a-a232f129300c
const x_value_iter4_mdp, o_value_iter4_mdp = make_x_o_tabular_mdps(value_iter_x_vs_iter3, value_iter_o_vs_iter3)

# ╔═╡ 585d07a0-66e8-47dd-a655-845bbed0e79a
const value_iter_x_vs_iter4 = value_iteration_v(x_value_iter4_mdp, ttt_value_γ)

# ╔═╡ d8d3d75e-b35d-4077-a953-f2a83958b48a
const value_iter_o_vs_iter4 = value_iteration_v(o_value_iter4_mdp, ttt_value_γ)

# ╔═╡ 18f5244f-c88a-4e84-9b5b-a8d8c2b76b23
#=╠═╡
@bind strat_select Select([(value_iter_x_vs_rand, value_iter_o_vs_rand) => "Strat 1", (value_iter_x_vs_iter4, value_iter_o_vs_iter4) => "Strat 2"])
  ╠═╡ =#

# ╔═╡ 83bdfbb5-ed4e-4d67-9db5-15037c788a65
const x_rand_mdp_alt, o_rand_mdp_alt = make_x_o_tabular_mdps(rewards = rewardsX_alt)

# ╔═╡ 57dfcf6f-572b-4029-9c65-cb56b1d1b6bd
const value_iter_x_vs_rand_alt = value_iteration_v(x_rand_mdp_alt, ttt_value_γ)

# ╔═╡ 712f33b8-8774-46c6-973e-48c765799e6e
const value_iter_o_vs_rand_alt = value_iteration_v(o_rand_mdp_alt, ttt_value_γ)

# ╔═╡ 1e4cbd58-5b9b-4382-9150-9ef8afd331bb
const x_value_iter1_mdp_alt, o_value_iter1_mdp_alt = make_x_o_tabular_mdps(value_iter_x_vs_rand_alt, value_iter_o_vs_rand_alt; rewards = rewardsX_alt)

# ╔═╡ fd7742c5-472d-46ac-b885-c720326a82c5
const value_iter_x_vs_o1_alt = value_iteration_v(x_value_iter1_mdp_alt, ttt_value_γ)

# ╔═╡ b98ceb8e-5780-40a3-95bc-1e8df86466b6
const value_iter_o_vs_x1_alt = value_iteration_v(o_value_iter1_mdp_alt, ttt_value_γ)

# ╔═╡ 2584f0cf-ad4c-4dca-86a3-357ee024c52c
#=╠═╡
@bind strat_select2 Select([(value_iter_x_vs_rand_alt, value_iter_o_vs_rand_alt) => "Strat 1", (value_iter_x_vs_o1_alt, value_iter_o_vs_x1_alt) => "Strat 2"])
  ╠═╡ =#

# ╔═╡ 10868e8b-f243-45bd-a21e-94b5a7f06f44
const x_value_iter2_mdp_alt, o_value_iter2_mdp_alt = make_x_o_tabular_mdps(value_iter_x_vs_o1_alt, value_iter_o_vs_x1_alt; rewards = rewardsX_alt)

# ╔═╡ c33be92f-9be3-4bb2-82e2-4ac95f2467f1
const value_iter_x_vs_o2_alt = value_iteration_v(x_value_iter2_mdp_alt, ttt_value_γ)

# ╔═╡ 73041a13-49da-4bd6-a233-5bb88b4f5fd4
const value_iter_o_vs_x2_alt = value_iteration_v(o_value_iter2_mdp_alt, ttt_value_γ)

# ╔═╡ 6ac696c7-9f40-45bb-a556-9507f0b10f87
#compute the action probability distribution and value for a given board state from a value iteration result output
function value_policy_output(value_policy::ValueIterationResults{T, S}, board) where {T <: Dict, S} 
	(newplayboard, isym) = symmetric_board_lookup[board]
	!haskey(value_policy.πstar, newplayboard) && return (zeros(9), "Invalid State")
	board_value = value_policy.state_values[newplayboard]
	πs = value_policy.πstar[newplayboard]
	prbs = [haskey(πs, a) ? πs[a] : 0.0 for a in UInt8.(1:9)][d4_inverted[isym]]
	return (prbs, board_value)
end

# ╔═╡ 46b0b624-c841-42f4-aa5d-9dba21c981fe
abstract type ResultsTTT end

# ╔═╡ a5df2ddc-fb88-4374-b2b2-28217cc8d1ee
struct PolicyResultsTTT{T} <: ResultsTTT
	rewards::Vector{Float64} #rewards per episode of training
	θ::Matrix{Float64} #parameters for policy function
	w::Vector{Float64} #parameters for value function
	eval_board::T #function to evaluate a board
end

# ╔═╡ 24b430a3-b8b7-4ef1-8050-347b1dd12ac7
function execute_ttt_actor_critic(states, step, get_s0, αθ, αw; kwargs...)
	agent = setup_ttt_player(states)
	s0 = ttt_environment.init_board
	sterm = ttt_environment.term_board
	actions = ttt_moves

	# reinforce_monte_carlo_control(π!, ∇lnπ!, length(θ), s0, αθ, step, sterm, actions; θ = θ, kwargs...)
	(rewards, θout, wout) = actor_critic_eligibility(agent.π!, agent.∇lnπ!, agent.v̂, agent.∇v̂, length(agent.θ), length(agent.w), s0, αθ, αw, step, sterm, actions; θ = agent.θ, w = agent.w, get_s0=get_s0, kwargs...)
	# one_step_actor_critic(π!, ∇lnπ!, v̂, ∇v̂, length(θ), length(w), s0, αθ, αw, step, sterm, actions; θ = θ, w = w, kwargs...)

	function eval_board(b)
		(symboard, isym) = symmetric_board_lookup[b]
		prbs = agent.π!(symboard, θout)[d4_inverted[isym]]
		v = agent.v̂(symboard, wout)
		(prbs, v)
	end
	PolicyResultsTTT(rewards, θout, wout, eval_board)
end

# ╔═╡ a651f37c-ef11-445f-8839-a31112559c2e
x_step_vs_random_results = execute_ttt_actor_critic(active_x_boards, x_step_vs_random, () -> rand() < 0.1 ? ttt_environment.init_board : rand(active_x_boards), 0.5, 0.5; λθ = 0.5, λw = 0.5, max_episodes = 100_000, showprogress=true)

# ╔═╡ dc8f8c7e-05c4-4c97-8d37-109b56b7f1fa
#train O-player vs the first X policy
o_step_vs_x1(board, move) = ttt_step(board, move, b -> select_action(x_step_vs_random_results.eval_board(b)[1]))

# ╔═╡ 666169e0-f745-4858-a28e-9a7217fdd092
o_vs_x1_results = execute_ttt_actor_critic(active_o_boards, o_step_vs_x1, () -> rand(active_o_boards), 0.5, 0.5; λθ = 0.5, λw = 0.5, max_episodes = 100_000, showprogress=true)

# ╔═╡ 0a623b8c-a462-4742-9141-d2cf75eae19b
x_vs_o1(board, move) = ttt_step(board, move, b -> select_action(o_vs_x1_results.eval_board(b)[1]))

# ╔═╡ 79c02db7-709e-407b-86fc-571951ac7d8b
x_vs_o1_results = execute_ttt_actor_critic(active_x_boards, x_vs_o1, () -> rand(active_x_boards), 0.5, 0.5; λθ = 0.5, λw = 0.5, max_episodes = 100_000, showprogress=true)

# ╔═╡ 4e522a97-b341-40cc-be52-065f265f2a3e
ttt_selfplay_results = execute_ttt_actor_critic(active_ttt_boards, ttt_step, () -> rand() < 0.75 ? ttt_environment.init_board : rand(active_ttt_boards), 0.5, 0.1; λθ = 0.5, λw = 0.5, γ = 0.9, max_episodes = 100_000, showprogress=true)

# ╔═╡ 4f03e182-81d6-422a-9a85-667854264a8e
#=╠═╡
#modify this so that it uses the new functions and plots progress per round by showing the victory rate over the previous opponent
function execute_actor_critic_selfplay(αθ, αw, rounds; kwargs...)
	form_opponent(results) = (board, move) -> ttt_step(board, move, b -> select_action(results.eval_board(b)[1]))
	train_player(active_boards, opponent) = execute_ttt_actor_critic(active_boards, opponent, () -> rand(active_boards), αθ, αw; kwargs...)

	x_results = Vector{PolicyResultsTTT}(undef, rounds)
	o_results = Vector{PolicyResultsTTT}(undef, rounds)

	x_results[1] = train_player(active_x_boards, (board, move) -> ttt_step(board, move, get_random_move))
	o_results[1] = train_player(active_o_boards, form_opponent(x_results[1]))
	
	@progress for i in 2:rounds
		x_results[i] = train_player(active_x_boards, form_opponent(o_results[i-1]))
		o_results[i] = train_player(active_o_boards, form_opponent(x_results[i]))
	end
	
	return x_results, o_results
end
  ╠═╡ =#

# ╔═╡ 32485151-c606-4483-8d8c-4c36b87f9158
#=╠═╡
ttt_rounds_results = execute_actor_critic_selfplay(0.5, 0.5, nrounds; λθ = 0.5, λw = 0.5, max_episodes = 30_000)
  ╠═╡ =#

# ╔═╡ da6ff1d3-561c-405e-aac9-54db12e95299
#=╠═╡
function plot_tttresults(ttt_results::PolicyResultsTTT, avgeps = 100)
	plot([mean(ttt_results.rewards[i:avgeps+i-1]) for i in 1:lastindex(ttt_results.rewards)-avgeps])
end
  ╠═╡ =#

# ╔═╡ ff61ffce-612f-4bb5-ba45-fe8715ceb180
#=╠═╡
plot_tttresults(x_step_vs_random_results, avgeps)
  ╠═╡ =#

# ╔═╡ 0e309975-5338-426e-85a8-04839cfba5de
#=╠═╡
plot_tttresults(o_vs_x1_results, avgeps)
  ╠═╡ =#

# ╔═╡ be81fcbd-45b0-4555-b68c-ababe164fbd3
#=╠═╡
plot_tttresults(x_vs_o1_results, avgeps)
  ╠═╡ =#

# ╔═╡ 558f82e6-8b24-4a04-ba0e-48fdfb0db98c
#=╠═╡
plot_tttresults(ttt_rounds_results[playerselect][roundcount], 100)
  ╠═╡ =#

# ╔═╡ a7340ee4-9b64-4b50-ba63-6b8682318f74
#=╠═╡
plot_tttresults(ttt_selfplay_results, avgeps)
  ╠═╡ =#

# ╔═╡ 81622974-4b91-4f26-bb60-01edebfe4e2b
get_ttt_move(results::ResultsTTT) = b -> select_action(results.eval_board(b) |> first)

# ╔═╡ e3711fc9-10fd-4802-aaf3-b70bed5e4597
get_ttt_matchup_statistics(get_ttt_move(x_step_vs_random_results), get_ttt_move(o_vs_x1_results))

# ╔═╡ f62c6c04-c97c-4dc7-8270-44eaca693cf7
get_ttt_matchup_statistics(get_ttt_move(x_vs_o1_results), get_ttt_move(o_vs_x1_results))

# ╔═╡ 130facfd-c121-45e5-ad49-06d2d5a63976
function compare_ttt_policies(results1::ResultsTTT, results2::ResultsTTT; kwargs...)
	p1 = get_ttt_move(results1)
	p2 = get_ttt_move(results2)
	get_ttt_matchup_statistics(p1, p2; kwargs...)
end

# ╔═╡ d235d68a-ecf9-46c5-839e-96727a3b3226
compare_ttt_policies(results::ResultsTTT, p::Function; kwargs...) = get_ttt_matchup_statistics(get_ttt_move(results), p; kwargs...)

# ╔═╡ a3dd5b6b-feed-4488-aca9-8f5e6a5caf3a
compare_ttt_policies(p::Function, results::ResultsTTT; kwargs...) = get_ttt_matchup_statistics(p, get_ttt_move(results); kwargs...)

# ╔═╡ 344dd837-021d-44ed-8a1d-ad5a295438be
#=╠═╡
function plot_ttt_rounds(round_results; trials = 1000)
	xrounds = first(ttt_rounds_results) |> Map(x_results -> compare_ttt_policies(x_results, get_random_move, trials = trials)) |> tcollect
	x_traces = [scatter(x = eachindex(round_results[1]), y = [a[sym] for a in xrounds], name = String(sym)) for sym in (:x_win, :o_win, :is_draw)] 
	p1 = Plot(x_traces, Layout(title = "X Player vs Random Policy", xaxis_title = "Rounds"))
	orounds = last(ttt_rounds_results) |> Map(o_results -> compare_ttt_policies(get_random_move, o_results, trials = trials)) |> tcollect
	o_traces = [scatter(x = eachindex(round_results[1]), y = [a[sym] for a in orounds], name = String(sym)) for sym in (:x_win, :o_win, :is_draw)] 
	p2 = Plot(o_traces, Layout(title = "O Player vs Random Policy", xaxis_title = "Rounds"))
	plot([p1 p2])
end
  ╠═╡ =#

# ╔═╡ c9169432-ba22-4bf9-aa51-ac633cac69f7
#=╠═╡
plot_ttt_rounds(ttt_rounds_results; trials = 10_000)
  ╠═╡ =#

# ╔═╡ 58b93ab0-44df-4594-8941-a09f312def82
compare_ttt_policies(x_step_vs_random_results, get_random_move)

# ╔═╡ 5afd7bb9-cf9d-4495-a712-2d6713595fb5
compare_ttt_policies(get_random_move, o_vs_x1_results)

# ╔═╡ 795b1315-cba8-4548-a609-1873e6cc34eb
compare_ttt_policies(x_vs_o1_results, get_random_move)

# ╔═╡ cac650e8-ee30-4068-a50f-28e4f7203323
compare_ttt_policies(ttt_selfplay_results, o_vs_x1_results)

# ╔═╡ 345fe2cc-8d6a-49a3-959d-e7b21410bc43
#=╠═╡
compare_ttt_policies(ttt_selfplay_results, ttt_rounds_results[end][2])
  ╠═╡ =#

# ╔═╡ 8c409991-1729-47c2-8cf4-598279f37fc7
compare_ttt_policies(ttt_selfplay_results, get_random_move)

# ╔═╡ 6a0664ed-b3c8-445f-b355-13b1c23630dd
compare_ttt_policies(x_vs_o1_results, o_vs_x1_results)

# ╔═╡ 6f7b66e4-9fe5-43d7-b0c6-9410c1f0ec8f
compare_ttt_policies(x_vs_o1_results, get_random_move)

# ╔═╡ b95837e8-942a-45ca-b3e6-22e44c2942f4
compare_ttt_policies(x_step_vs_random_results, get_random_move)

# ╔═╡ a0f01732-fbb5-4bf3-a6a8-4ef8794ee572
compare_ttt_policies(ttt_selfplay_results, ttt_selfplay_results)

# ╔═╡ 9cc3d72d-3706-4af8-a62c-d143fa439664
struct ValueResultsTTT{V, S, T} <: ResultsTTT
	state_values::V
	πstar::Dict{S, Dict{Int64, Float64}}
	eval_board::T
end

# ╔═╡ 3f1f6c2f-f377-4f38-b4c4-4df3817bb7d3
function run_ttt_value_iteration(ptf; γ=1.0, savelist = false, kwargs...) 
	results = begin_value_iteration_v(ptf, ttt_environment.term_board, γ; θ = 0.0, nmax=Inf, Vinit=0.0, savelist=savelist, kwargs...)
	eval_board(b) = value_policy_output(results, b)
	ValueResultsTTT(results.state_values, results.πstar, eval_board)
end

# ╔═╡ a441ecdd-1837-451e-944c-b2fde5bd7276
md"""
#### Visualize O Player Policy Against Random Opponent
"""

# ╔═╡ b2c85ccd-7338-4eda-9491-d9f24e771dc8
#add a function to show boards where the policies differ

# ╔═╡ 541dc7b8-951f-4296-9f54-bc74ba831bab
#identify states where two policies differ 
function compare_actions(π1, π2, states)
	compactions = [s => (π1[s], π2[s]) for s in states]
	Dict(filter(a -> a[2][1] != a[2][2], compactions))
end

# ╔═╡ 96f5d75e-8b24-4bdb-82d6-d9173682633a
#can alternate this as well until each player's policy is identical for every state similar to how the value iteration stops running

# ╔═╡ 0facd26f-d6af-4574-9afa-4e90b5043cb7
#make probability transition function for a selfplay game of tic tac toe over all active states
function make_ttt_ptf()
	function get_transitions(board::S) where S
		moves = findall(==(0), board)
		isempty(moves) && return Dict{Tuple{S, Float64, S, UInt8}, Float64}()
		mapreduce(mergewith(+), moves) do move
			(newboard, r, active) = ttt_step(board, move)
			Dict((newboard, r, board, move) => 1.)
		end
	end

	#only calculate transitions from valid states for x player
	ptf = mapreduce(get_transitions, mergewith(+), active_ttt_boards)
	sa_keys = get_sa_keys(ptf)

	return (ptr = ptf, sa_keys = sa_keys)
end

# ╔═╡ 9d2ea20b-3250-4bcc-bcb6-193612ab8ff8
const x_vs_random_ptf = make_ttt_ptf(active_x_boards, π_random_ttt)

# ╔═╡ 0e51f9d7-c37e-471a-90a3-9d5ab43896c6
x_vs_random_value_results = run_ttt_value_iteration(x_vs_random_ptf; γ = 0.9)

# ╔═╡ 60477da6-eff0-4c01-ab39-492fa1c1a59e
const o_vs_random_ptf = make_ttt_ptf(active_o_boards, π_random_ttt)

# ╔═╡ c7f5c690-884f-4489-a9d6-64816dba925d
o_vs_random_value_results = run_ttt_value_iteration(o_vs_random_ptf; invert_state = s -> -1.0)

# ╔═╡ 75258904-d987-42bd-a9f5-69e6d4eff687
o_vs_x1_ptf = make_ttt_ptf(active_o_boards, b -> x_vs_random_value_results.πstar[b])

# ╔═╡ 6b5731f8-4d68-4007-b4fe-ac2351a532d2
o_vs_x1_value_results = run_ttt_value_iteration(o_vs_x1_ptf, invert_state = s -> -1.0)

# ╔═╡ 2d766054-08c1-40a4-ac79-9498fffad9ab
compare_actions(o_vs_random_value_results.πstar,  o_vs_x1_value_results.πstar, active_o_boards) |> length #this is how many states that have a different policy

# ╔═╡ 45eb3f00-4ae6-4f9b-af5e-ebb4a7c076f6
x_vs_o_ptf = make_ttt_ptf(active_x_boards, b -> o_vs_x1_value_results.πstar[b])

# ╔═╡ 80442053-1908-420c-b71e-f14c59da4d99
x_vs_o_value_results = run_ttt_value_iteration(x_vs_o_ptf)

# ╔═╡ 2a77ae50-ada6-4204-ba46-6b2d7c565138
o_vs_x2_ptf = make_ttt_ptf(active_o_boards, b -> x_vs_o_value_results.πstar[b])

# ╔═╡ 7e92bf5b-5d34-4a7f-8002-2471692a5fe8
o_vs_x2_value_results = run_ttt_value_iteration(o_vs_x2_ptf, invert_state = s -> -1.0)

# ╔═╡ 147aa41b-d390-4258-8023-eb36f4e9b7b3
o_policy_comp = compare_actions(o_vs_x2_value_results.πstar,  o_vs_x1_value_results.πstar, active_o_boards)

# ╔═╡ 18c9654e-36f4-49d6-97a5-ca03ddc511c5
x_vs_o2_ptf = make_ttt_ptf(active_x_boards, b -> o_vs_x2_value_results.πstar[b])

# ╔═╡ 4489fdc1-2f3d-4609-8c0c-0f6c3e67c4ab
x_vs_o2_value_results = run_ttt_value_iteration(x_vs_o2_ptf)

# ╔═╡ 42583b0d-dd48-4150-b7f4-a66c527326a3
#so these two policies are equivalent
x_vs_o_value_results.πstar == x_vs_o2_value_results.πstar

# ╔═╡ 759b0d57-4475-4643-ad79-fcc608846b5d
const selfplay_ptf = make_ttt_ptf()

# ╔═╡ 804dd7f6-c589-4f39-ba0f-20a1b46f9aaa
selfplay_value_results = run_ttt_value_iteration(selfplay_ptf, invert_state = s -> is_o_move(s) ? -1.0 : 1.0)

# ╔═╡ 3070b33a-fb35-4242-ab2d-5187e49a7d75
md"""
### Compare Learned Policies
"""

# ╔═╡ 25a74028-3575-4f4b-8083-707af22aaea6
function makepolicycomptable(xplayers, oplayers)
	tablenames = [:x_win, :o_win, :is_draw]
	tables = Dict(name => zeros(length(xplayers), length(oplayers)) for name in tablenames)
	for (i, x) in enumerate(xplayers) for (j, o) in enumerate(oplayers)
		results = compare_ttt_policies(x, o)
		for name in tablenames
			tables[name][i, j] = results[name]
		end
	end end
	return NamedTuple(tables)
end

# ╔═╡ 79885cfc-7cda-4a8f-aaa6-ab4d09fbca91
matchup_tables = makepolicycomptable([selfplay_value_results, x_vs_random_value_results, x_vs_o_value_results, get_random_move], [selfplay_value_results, o_vs_random_value_results, o_vs_x1_value_results, get_random_move])

# ╔═╡ 786e1340-1b1c-478f-9949-b48ed01cde33
compare_ttt_policies(selfplay_value_results, o_vs_x1_results)

# ╔═╡ 9a3059b0-ec9a-4c1b-beb1-901b2285feaa
compare_ttt_policies(selfplay_value_results, get_random_move)

# ╔═╡ cb143d38-8ceb-4c8d-aa59-8e4df8a67f3c
compare_ttt_policies(x_vs_random_value_results, get_random_move)

# ╔═╡ 94c3c1be-caf2-4093-b795-b1b3a80ffc81
compare_ttt_policies(x_vs_random_value_results, selfplay_value_results)

# ╔═╡ 4686cbb3-a6d0-49de-b7d6-7a6b6873cfe9
compare_ttt_policies(selfplay_value_results, selfplay_value_results)

# ╔═╡ 183b9738-4f81-423b-93ae-f0be2fdc190d
compare_ttt_policies(get_random_move, selfplay_value_results)

# ╔═╡ f1a47d6b-f394-470d-b92b-1d2efd0526b4
compare_ttt_policies(get_random_move, o_vs_random_value_results)

# ╔═╡ 3bceb7cb-acd6-49e6-bf33-02e377cf19f5
const boardnodes = Dict(begin
		moves = findall(==(0), b)
		nextboards = if isempty(moves) 
			Set{SVector{9, UInt8}}()
		else
			Set(symmetric_move(b, a) for a in moves)
		end
		b => nextboards
	
	end
	for b in active_ttt_boards)

# ╔═╡ d056f230-0310-48cc-98ea-dddc60f2a261
#should address this problem of having values for states that should be terminal.  The value of every terminal state should be 0.0 and the symmetry map should turn every such state into the terminal state.  Also states where more than one player has 3 in a row should be eliminated from the MDP

# ╔═╡ 63888652-d0e7-495d-9f03-4bf988441675
#next step is to implement the HTML program for adding moves to the state and updating a board object.  Ideally we could recompute the policy as well but another cell could actually update the style for these grid elements which would change the appearance.  Yeah so I can make the HTML where the bound variable is the board and then another cell styles that board with the correct policy.  But then I would need to just stick with one policy per board.  Also wanna implement the reset button.

# ╔═╡ 8a334293-b568-4dcc-b9d5-cfd542a8c9f7
function get_minimax_policy(minimaxvalues, board)
	moves = check_available_moves(board)
	c = is_o_move(board) ? -1.0 : 1.0
	prefs = [begin
		newboard = first(move(board, a))
		if haskey(minimaxvalues, newboard)
			c*minimaxvalues[newboard]
		else
			-Inf
		end
	end
	for a in UInt8.(1:9)]

	v = soft_max(1e2*prefs)
end

# ╔═╡ 170474a5-7a8e-4e6c-8c3a-9d6efbd85637
function minimax(board, o_max_player::Bool, boardvalues)
	c = o_max_player ? -1.0 : 1.0
	nextboards = boardnodes[board]
	if any(checkboard(board)) || isempty(nextboards)
		r = c*get_reward_x(board)
		boardvalues[board] = r
		return r
	end
	
	(value, f) = if (is_o_move(board) == o_max_player) #maximizing player
		(-Inf, max)
	else
		(Inf, min)
	end

	for newboard in nextboards
		value = f(value, minimax(newboard, o_max_player, boardvalues))
	end
	boardvalues[board] = value
	return value
end

# ╔═╡ 345af4aa-8bbe-46ce-917c-a5237114a92e
function run_minimax(startboard)
	boardvalues = Dict{SVector{9, UInt8}, Float64}()
	v = minimax(startboard, is_o_move(startboard), boardvalues)
	π = Dict(board => get_minimax_policy(boardvalues, board) for board in keys(boardvalues))
	return (v, boardvalues, π)
end

# ╔═╡ 8e31c353-e7fd-4d7d-b35a-cfb011f325fd
(baseval, minimaxvalues, minimax_policy) = run_minimax(ttt_environment.init_board)

# ╔═╡ 4f16565e-09bb-11f0-3729-7ffc5462cdc8
md"""
# Dependencies
"""

# ╔═╡ 43999574-da93-475c-9a67-e5024fb08202
# ╠═╡ skip_as_script = true
#=╠═╡
import HypertextLiteral.@htl
  ╠═╡ =#

# ╔═╡ 1b84943c-c8f5-4ad4-b95a-66dc818fa609
# ╠═╡ skip_as_script = true
#=╠═╡
html"""
	<style>
		main {
			margin: 0 auto;
			max-width: min(1600px, 90%);
			padding-left: max(10px, 5%);
			padding-right: max(10px, 5%);
			font-size: max(10px, min(24px, 2vw));
		}
	</style>
	"""
  ╠═╡ =#

# ╔═╡ afb3f1df-8aa7-4e57-bf03-9d901c9c2946
md"""
## Visualization Tools
"""

# ╔═╡ baad7afc-5104-4ea9-85bf-411c8cbb5c20
joinrow(a, b) = "$a|$b"

# ╔═╡ 27866a8d-a26b-4602-b10b-1e8748a77399
joinmdrows(r1, r2) = "$r1\n$r2"

# ╔═╡ b015036c-d045-4749-b0e8-33735b04c328
function make_md_row(v::AbstractVector)
	"""|$(reduce(joinrow, v))|"""
end

# ╔═╡ 74250a9c-958b-4689-abbb-92764ddf8904
function matrix_to_mdtable(M, header, rownames)
	body = mapreduce(joinmdrows, eachrow(hcat(rownames, M))) do row
		make_md_row(row)
	end
	h = make_md_row(header)
	n = make_md_row(["---" for _ in eachindex(header)])
	reduce(joinmdrows, [h, n, body])
end

# ╔═╡ 0c968db5-69a4-4452-80f7-3c22fd4f93aa
function display_matchup_comps(tables, xnames, onames; title = "Outcome Probabilities Per Matchup")
	out = 
	"""	
	##### $title

	Draw
	
	$(matrix_to_mdtable(tables.is_draw, [""; onames], xnames))

	X Win
	
	$(matrix_to_mdtable(tables.x_win, [""; onames], xnames))

	O Win
	
	$(matrix_to_mdtable(tables.o_win, [""; onames], xnames))
	"""
	Markdown.parse(out)
end

# ╔═╡ d8cd4939-9c00-4adf-bc29-49b6f02134bf
display_matchup_comps(matchup_tables, ["selfplay value", "x vs random", "x vs o1 value", "random"], ["selfplay value", "o vs random", "o vs x1 value", "random"]; title = "Value Iteration Outcome Probabilities")

# ╔═╡ 28799f85-5fb7-4617-9c3a-fa307bfc25d4
md"""
### Style and JavaScript
"""

# ╔═╡ 07fa2e67-42a8-4c2b-ad95-cd0f9f28e122
const base_cell_style = HTML("""
		<style>
		.grid-container {
			margin: 10px;
			display: grid;
			justify-content: center;
			align-content: center;
			grid-template-columns: repeat(3, auto);
			background-color: rgb(31, 31, 31);
		}

		.grid-container .gridcell.x::before,
		.grid-container .gridcell.x::after,
		.grid-container.x .gridcell:hover:not(.x):not(.o)::before,
		.grid-container.x .gridcell:hover:not(.x):not(.o)::after {
			content: '';
			position: absolute;
			background-color: black;
			width: 10%;
			height: 90%;
		}

		.grid-container .gridcell.x::before,
		.grid-container.x .gridcell:hover::before {
			transform: rotate(45deg);
		}

		.grid-container .gridcell.x::after,
		.grid-container.x .gridcell:hover::after {
			transform: rotate(-45deg);
		}

		.grid-container .gridcell.o::before, 
		.grid-container.o .gridcell:hover:not(.x):not(.o)::before
		{
			content: '';
			background-color: rgba(1, 1, 1, 0);
			border: 10px solid black;
			border-radius:50%;
			width: 65%;
			height: 65%;
		}

		.grid-container.x .gridcell:hover:not(.x):not(.o)::before,
		.grid-container.x .gridcell:hover:not(.x):not(.o)::after {
			background-color: gray;
		}

		.grid-container.o .gridcell:hover:not(.x):not(.o)::before {
			border-color: gray;
		}
		
		.gridcell {
			border: 1px solid black;
			display: flex;
			justify-content: center;
			align-items: center;
			position: relative;
			cursor: pointer;
			width: vw/10;
			height: vw/10;
		}

		.gridcell.x, .gridcell.o {
			cursor: not-allowed;
		}

		.gridcell:first-child,
		.gridcell:nth-child(2),
		.gridcell:nth-child(3) {
			border-top: none;
		}

		.gridcell:nth-child(3),
		.gridcell:nth-child(6),
		.gridcell:nth-child(9) {
			border-right: none;
		}

		.gridcell:nth-child(7),
		.gridcell:nth-child(8),
		.gridcell:nth-child(9) {
			border-bottom: none;
		}

		.gridcell:nth-child(1),
		.gridcell:nth-child(4),
		.gridcell:nth-child(7) {
			border-left: none;
		}
	</style>
""")

# ╔═╡ b4ab7ecb-c481-475f-8bca-7cc0a98a0287
function make_board_script(name) 
	"""
<script>
	const resetButton = document.querySelector(".$name .resetButton");
	console.log("got button")
	console.log(resetButton)
	resetButton.addEventListener("click", resetClick);
	resetButton.onclick = console.log("clicked");
	
	const X_CLASS = 'x'
	const CIRCLE_CLASS = 'o'
	const span = currentScript.parentElement
	const board = document.querySelector('.grid-container.$name')
	const cells = [...board.children];
	
	let circleTurn 

	span.value = [$(zeros(Int64, 9)), '$name']
	span.dispatchEvent(new CustomEvent('input'))

	cells.forEach ((child) => {
		child.addEventListener('click', handleClick, {once: true});    
	})

	function resetClick(e) {
		console.log('button pushed')
		restart()
	}

	function restart() {
		circleTurn = false
		cells.forEach((cell) => {
			var index = cells.indexOf(cell);
			cell.classList.remove(X_CLASS);
			cell.classList.remove(CIRCLE_CLASS);
			cell.removeEventListener('click', handleClick);
			cell.addEventListener('click', handleClick, {once: true});
			span.value[0][index] = 0;
		})
		setBoardHoverClass()
		span.dispatchEvent(new CustomEvent('input'))
	}

	function handleClick(e) {
		const cell = e.target;
		const index = cells.indexOf(cell);
		console.log('cell ', index, ' clicked');
		const currentClass = circleTurn ? CIRCLE_CLASS : X_CLASS;
		const fillValue = circleTurn ? 2 : 1;
		placeMark(cell, currentClass);
		swapTurns();
		setBoardHoverClass();
		span.value[0][index] = fillValue;
		span.dispatchEvent(new CustomEvent('input'));
	}

	function placeMark(cell, currentClass) {
		cell.classList.add(currentClass)
	}

	function setBoardHoverClass() {
		board.classList.remove(X_CLASS)
		board.classList.remove(CIRCLE_CLASS)
		if (circleTurn) {
			board.classList.add(CIRCLE_CLASS)
		} else {
			board.classList.add(X_CLASS)
		}
				
	}

	function swapTurns() {
		circleTurn = !circleTurn
	}
	
</script>
"""
end

# ╔═╡ 6bc3c3ee-84e8-44a9-90bc-7c9a91624dfd
md"""
### Board Display and Control
"""

# ╔═╡ 0e2591a9-77d5-41a5-90e7-f4bb750c820b
md"""
### Restyling Utilities
"""

# ╔═╡ a032a3c7-a4c2-4053-aab1-945405764f1b
const no_color = "rgba(0, 0, 0, 0)"

# ╔═╡ dd6d21c4-d04f-4f4b-ad71-901b50d1d23d
joinelements(a, b) =  """$a \n $b"""

# ╔═╡ ed165fdb-e2fe-4636-81f9-fdcdb2e3784a
make_elems(f, iter) = mapreduce(f, joinelements, iter)

# ╔═╡ c117f3dd-5122-4951-bbce-365cd1f2c5fb
function colorcell(name, i, c)
	"""
	.grid-container.$name .gridcell:nth-child($i) {
		background-color: $c;
	}
	"""
end

# ╔═╡ 5e091621-e9ac-46a5-8470-06ca5bc377f8
function colorboard(name::AbstractString, colors::AbstractVector{T}) where T <: AbstractString
	HTML("""
	<style>
	$(make_elems(i -> colorcell(name, i, colors[i]), 1:9))
	</style>
	""")
end

# ╔═╡ cce8f5ef-9df2-4ed2-803d-7e435bb23e82
#option to just make every cell the same color
colorboard(name, color) = colorboard(name, fill(color, 9))

# ╔═╡ 9dd13f7f-34a9-40c2-80f1-9b115faf73c4
#display boards in rows that wrap to the next line
function displayboards(boards)
	HTML("""
	<span class=multiboard>
	$(reduce(joinelements, boards))
	</span>
	<style>
		.multiboard {
			display: flex;
			flex-wrap: wrap;
		}
	</style>
""")
end

# ╔═╡ 91404294-9145-4314-986a-10ea23dbdda8
function resize_board(name, cellsize)
	HTML("""
	<style>
	.grid-container.$name .gridcell {
			width: $(cellsize)px;
			height: $(cellsize)px;
		}
	.grid-container.$name .gridcell.o::before, 
	.grid-container.$name.o .gridcell:hover:not(.x):not(.o)::before
	{
		border: $(cellsize/10)px solid black;
	}
	.grid-container.$name.o .gridcell:hover:not(.x):not(.o)::before {
			border-color: gray;
		}
	.$name .resetButton {
		font-size: $(min(20, cellsize/3))px;
	}
	.$name .board-value {
		font-size: $(min(20, cellsize/4))px;
	}
	</style>
""")
end

# ╔═╡ eaed2a5b-944a-453f-84ad-04f3917ca27a
resize_boards(boardnames::Union{AbstractVector{T}, Base.Generator}, size) where T <: AbstractString = HTML(reduce(joinelements, (resize_board(b, size).content for b in boardnames)))

# ╔═╡ 6d4217a2-5b1f-46bb-95d1-f73741a24690
function annotate_value(name, str)
	"""
	<style>
	.$name .board-value::after {
		content: '$str';
		background-color: "rgba(0, 0, 0, 0)";
		font-weight: normal;
		color: rgb(180, 180, 180);
		font-family: Arial;
		text-shadow: 1px 2px 1px black;
	}
	</style>
"""
end

# ╔═╡ 8f24e187-f9a6-4579-9f20-97f33ab657ed
value_board(name, v::AbstractFloat) = annotate_value(name, "Value Est: $(round(v, sigdigits = 2))")

# ╔═╡ af16a2a9-aa59-4c51-8e30-841eddd4aff5
value_board(name, v) = annotate_value(name, "Value Est: $v")

# ╔═╡ af8c173b-c310-4d3d-80cb-11343878025a
prb_to_color(p::AbstractFloat) = "rgb(40, $(max(40, .9*round(Int64, 255*(p .^(1/2))))), 40)"

# ╔═╡ 6a215dfb-8869-4efa-8708-99813b13c296
makecolors(prbs::AbstractVector{T}) where T <: AbstractFloat = prb_to_color.(prbs)

# ╔═╡ a90bbcb3-2119-41c3-b06b-66f59d60d0a5
colorboard(name::AbstractString, prbs::AbstractVector{T}) where T <: AbstractFloat = colorboard(name, makecolors(prbs)) 

# ╔═╡ e2ae92dc-34ec-48fa-a958-2d6c0170cc79
#color a TTT board with action probabilities based on a policy function
function style_value_policy(get_value_policy, board, boardname)
	(prbs, v) = try get_value_policy(board) catch; (zeros(9), "Invalid State") end
	c = colorboard(boardname, prbs).content
	htmlstr = if isa(v, Real)
		joinelements(c, value_board(boardname, round(v, sigdigits = 2)))
	else
		joinelements(c, value_board(boardname, v))
	end
	HTML(htmlstr)
end

# ╔═╡ 82b948a9-be98-409a-8d2a-e7ff73376507
style_value_policy(x_vs_o1_results.eval_board, xplayboard2...)

# ╔═╡ e42a7b19-5be5-4b4e-bceb-cb4bba24ee60
randomclassname(n = 20) = string(rand('a':'z'), String(rand(['a':'z'; '0':'9'; '_'; '-'], 20)))

# ╔═╡ ca4116b8-98a9-4de6-ba82-24ce27579eb5
function make_ttt_board_raw(board; colors = ["rgba(0, 0, 0, 0)" for _ in 1:9], cellsize = 100, name = randomclassname(), boardtitle = "", value = nothing)
	function makehtmlcell(v)
		str = if v == 1
			" x"
		elseif v == 2
			" o"
		else
			""
		end
		"""<div class = "gridcell$str"></div>"""
	end
	gridstr(board) = is_o_move(board) ? "o" : "x"
	function makecontainer(board, name)
		"""
		<div class = "grid-container $name $(gridstr(board))">
			$(makecells(board))
		</div>
		"""
	end
	
	makecells(board) = make_elems(makehtmlcell, board)

	addvalue(v::AbstractFloat) = value_board(name, v)
	addvalue(v::AbstractString) = annotate_value(name, v)
	addvalue(::Nothing) = """"""

	board = """
	<span class = $name>
	<div>$boardtitle</div>
	<div class = "board-value"></div>
	$(makecontainer(board, name))
	</span>
	$(colorboard(name, colors).content)
	$(resize_board(name, cellsize).content)
	<style>
		$name {
			display: flex;
			flex-direction: column;
		}
	</style>
	$(addvalue(value))
	"""
	(board = board, id = name)
end

# ╔═╡ 255d5d51-08b9-43f2-a373-1fd9a70374eb
function makecompboard_display(board, policies::AbstractVector{T}, titles; kwargs...) where T <: ResultsTTT
	@assert length(policies) == length(titles)
	policyoutputs = [try policy.eval_board(board) catch; (zeros(9), "Invalid State") end for policy in policies]
	rawboards = [make_ttt_board_raw(board; boardtitle = title, colors = policyoutputs[i][1], value = policyoutputs[i][2], kwargs...) for (i, title) in enumerate(titles)]
	displayboards = [a[1] for a in rawboards]
	boardids = [a[2] for a in rawboards]
	(htmlboards = displayboards, boardids = boardids)
end

# ╔═╡ 12d0c927-3733-4bf8-94d4-fbfc82dab098
compdisplayboards1 = makecompboard_display(compboard1[1], [x_step_vs_random_results, o_vs_x1_results, x_vs_o1_results], ["x vs random", "o vs x1", "x vs o1"]; cellsize = 70)

# ╔═╡ 008a1ccb-7155-43f8-a971-b798144f6a57
displayboards(compdisplayboards1.htmlboards)

# ╔═╡ ecde9687-8a3d-4acb-b854-50c030febf1c
function displayexamplegame(xplayer::PolicyResultsTTT, oselect::Function; cellsize = 50)
	game = run_ttt_game(b -> select_action(xplayer.eval_board(b)[1]), oselect)
	gameboards = [(board, make_ttt_board_raw(board, cellsize = cellsize)) for board in game[1]]
	style = mapreduce(joinelements, gameboards[1:end-1]) do board
		if !is_o_move(board[1])
			style_value_policy(xplayer.eval_board, board[1], board[2][2]).content
		else
			""""""
		end
	end
	base = joinelements(displayboards(a[2][1] for a in gameboards).content, style)
	outcomestr = game[2].status.x_win ? "X Wins" : game[2].status.o_win ? "O Wins" : "Draw"
	joinelements(base, annotate_value(gameboards[end][2][2], outcomestr)) |> HTML
end

# ╔═╡ d618c9fd-8d82-49b6-be5c-bdef436c9ef0
function displayexamplegame(xselect::Function, oplayer::PolicyResultsTTT; cellsize = 50)
	game = run_ttt_game(xselect, b -> select_action(oplayer.eval_board(b)[1]))
	gameboards = [(board, make_ttt_board_raw(board, cellsize = cellsize)) for board in game[1]]
	style = mapreduce(joinelements, gameboards[1:end-1]) do board
		if is_o_move(board[1])
			style_value_policy(oplayer.eval_board, board[1], board[2][2]).content
		else
			""""""
		end
	end
	base = joinelements(displayboards(a[2][1] for a in gameboards[2:end]).content, style)
	outcomestr = game[2].status.x_win ? "X Wins" : game[2].status.o_win ? "O Wins" : "Draw"
	joinelements(base, annotate_value(gameboards[end][2][2], outcomestr)) |> HTML
end

# ╔═╡ 844d3cfd-6d40-4bba-bc2d-2cdc8c6130c2
function displayexamplegame(xplayer::PolicyResultsTTT, oplayer::PolicyResultsTTT; cellsize = 50)
	game = run_ttt_game(b -> select_action(xplayer.eval_board(b)[1]), b -> select_action(oplayer.eval_board(b)[1]))
	gameboards = [(board, make_ttt_board_raw(board, cellsize = cellsize)) for board in game[1]]
	style = mapreduce(joinelements, gameboards[1:end-1]) do board
	result = if is_o_move(board[1])
		oplayer
	else
		xplayer
	end
	style_value_policy(result.eval_board, board[1], board[2][2]).content
	end
	base = joinelements(displayboards(a[2][1] for a in gameboards).content, style)
	outcomestr = game[2].status.x_win ? "X Wins" : game[2].status.o_win ? "O Wins" : "Draw"
	joinelements(base, annotate_value(gameboards[end][2][2], outcomestr)) |> HTML
end

# ╔═╡ afecb204-21e2-4971-aa73-ded9f8c5ce02
displayexamplegame(x_step_vs_random_results, get_random_move)

# ╔═╡ 18f69434-59c5-415b-b222-e4d932dfa2b1
displayexamplegame(x_step_vs_random_results, o_vs_x1_results)

# ╔═╡ c98cab58-64cb-4a05-99c1-4398989babd2
displayexamplegame(x_vs_o1_results, o_vs_x1_results)

# ╔═╡ 2d93e5d4-77d0-4542-ada2-c488975ff5cc
displayexamplegame(get_random_move, o_vs_x1_results)

# ╔═╡ 4ec75c16-e19e-4609-ba45-296e8ed5e694
displayexamplegame(ttt_selfplay_results, o_vs_x1_results)

# ╔═╡ a61157e7-2f22-4e99-a9ef-32f302d4b60b
displayexamplegame(ttt_selfplay_results, ttt_selfplay_results)

# ╔═╡ 95da63e5-edbc-49a2-9a34-1ccf297c04e2
displayexamplegame(ttt_selfplay_results, get_random_move)

# ╔═╡ 493504bd-17a4-4e0a-a445-effd82ca1807
#create interactive board that works with @bind
function TTTBoard(;cellsize = 100, alignment = "flex-start")
	(board, id) = make_ttt_board_raw(zeros(9); cellsize = cellsize) #make empty board
	js = make_board_script(id)
	HTML(
		"""
		<span class = $id>
			<button class="resetButton">Reset Board</button>
			$board
			$js
		</span>
		<style>
			.$id {
				display: flex;
				flex-direction: column;
				align-items: $alignment;
			}
		</style>
		"""
	)
end

# ╔═╡ cfbf20c5-9605-4595-952a-a90cabadab65
@bind testboard TTTBoard()

# ╔═╡ c28b68c9-a732-4019-aff4-6d80bd595957
testboard

# ╔═╡ a99f68ab-9f7d-474f-ae0e-6e99c707006e
get_board_status(testboard[1]), get_reward(testboard[1]), isvalid(testboard[1])

# ╔═╡ 4a940b85-8f56-414a-a7a8-b2895605ae45
@bind value_iter_fixed_board TTTBoard()

# ╔═╡ 4ef908c9-a2a7-4f85-b98d-c9dab9fc3ac7
const value_iter_rotated_board = get_symmetric_index(value_iter_fixed_board[1])

# ╔═╡ 8b6786f7-220c-4daa-b707-376c2c7bc14c
const value_iter_board1_status = get_board_status(value_iter_fixed_board[1])

# ╔═╡ 124519d9-b8b3-492b-a3af-36c48553bc52
#=╠═╡
const value_iter_board1_result = value_iter_board1_status.is_o_move ? strat_select[2] : strat_select[1]
  ╠═╡ =#

# ╔═╡ cb50cf2c-6adf-42de-be9a-085676876bbe
#=╠═╡
md"""
#### Value Iteration Policy Visualization

The board below is colored according to the policy with green indicating likely moves.

State Value for $(value_iter_board1_status.is_o_move ? "O" : "X") Player: $(round(value_iter_board1_result.final_value[value_iter_rotated_board.index] |> Float64; sigdigits = 3))
"""
  ╠═╡ =#

# ╔═╡ b8a5f496-5727-48ab-804f-3dc027cca0f1
get_symmetric_board(value_iter_fixed_board[1])

# ╔═╡ c23d9164-d96d-43fe-b8de-f1b7c6c6d30f
test_index = symmetric_board_index[get_symmetric_board(value_iter_fixed_board[1])[1]]

# ╔═╡ 6fea4c9f-b70e-4e64-92bf-0436c1203516
symmetric_boards[test_index]

# ╔═╡ 079f86e4-337e-4cf2-a50b-042e351f65c7
showboard(symmetric_boards[test_index])

# ╔═╡ 8ff4ceda-e4df-4652-b48f-4b9fe4bbce27
value_iter_test.optimal_policy[:, symmetric_board_index[get_symmetric_board(value_iter_fixed_board[1])[1]]]

# ╔═╡ bbfa172b-e519-4ee6-9410-935a8baf3cbf
@bind value_iter_fixed_board2 TTTBoard()

# ╔═╡ bc1b2683-89b1-4813-bf94-2fd41d23e6ed
const value_iter_rotated_board2 = get_symmetric_index(value_iter_fixed_board2[1])

# ╔═╡ 2a67bf2c-09d1-4dc3-b4f6-200ad9be940e
const value_iter_board2_status = get_board_status(value_iter_fixed_board2[1])

# ╔═╡ 7717eb2e-1bf7-4953-8892-d8f936a85756
#=╠═╡
const value_iter_board2_result = value_iter_board2_status.is_o_move ? strat_select2[2] : strat_select2[1]
  ╠═╡ =#

# ╔═╡ 32ee92b1-a61a-460a-b436-036ef9cef080
#=╠═╡
md"""
#### Value Iteration Policy Visualization for Alt MDP

The board below is colored according to the policy with green indicating likely moves.

State Value for $(value_iter_board2_status.is_o_move ? "O" : "X") Player: $(round(value_iter_board2_result.final_value[value_iter_rotated_board2.index] |> Float64; sigdigits = 3))
"""
  ╠═╡ =#

# ╔═╡ 436c72df-c0a1-428c-a005-25381c758deb
@bind xplayboard TTTBoard()

# ╔═╡ 3c4bc244-d91f-48f5-9b74-99b1a5d242f7
style_value_policy(x_step_vs_random_results.eval_board, xplayboard...)

# ╔═╡ d1b2f039-d3c1-4709-8a53-43dc85b10f50
@bind oplayboard TTTBoard()

# ╔═╡ 1b5f9e37-7ba4-44ed-a4dd-62ec0a0adafe
style_value_policy(o_vs_x1_results.eval_board, oplayboard...)

# ╔═╡ 36ea7865-bfd9-4ec1-bda4-213ee2f09466
@bind selfplayboard TTTBoard()

# ╔═╡ 7613cd78-fa2f-4dc1-9c6d-4331823f5371
style_value_policy(ttt_selfplay_results.eval_board, selfplayboard...)

# ╔═╡ 4d54ba83-2489-4d51-a75e-d27c321babf7
md"""
#### Visualize Learned X-Player Policy Against Random  

Higher probability moves appear more green.  Click on board to change state by adding moves.  The value estimate will be 1.0 for an expected win, -0.5 for a draw, and -1.0 for a loss.

$(@bind base_board1 TTTBoard())
"""

# ╔═╡ 5753d01d-61e5-4306-a10d-c2ac6d0f8e3e
style_value_policy(x_vs_random_value_results.eval_board, base_board1...)

# ╔═╡ 73f5697b-2928-4f4e-b3ff-8129c1d4f8ed
@bind o_vs_random_value_board TTTBoard()

# ╔═╡ 4e08ba5a-b5b0-4e4c-b4d3-f806bbf1d13c
style_value_policy(o_vs_random_value_results.eval_board, o_vs_random_value_board...)

# ╔═╡ 92dbe6ad-7f82-428c-b2b9-446844d8a7e0
@bind o_vs_x1_value_board TTTBoard()

# ╔═╡ 3e239392-9289-4c79-8a64-d80305e57ed2
style_value_policy(o_vs_x1_value_results.eval_board, o_vs_x1_value_board...)

# ╔═╡ dcedc66c-2bd5-403c-9a63-5df777014bcf
@bind x_vs_o_value_board TTTBoard()

# ╔═╡ 48d8f9e2-3336-4f10-8781-ac898ca286ad
style_value_policy(x_vs_o_value_results.eval_board, x_vs_o_value_board...)

# ╔═╡ 908894b3-b53b-4f6b-9175-c15fc69bf2c4
@bind o_vs_x2_value_board TTTBoard()

# ╔═╡ 3ac17b3b-3e2d-424f-9cf6-006589ae1cc9
style_value_policy(o_vs_x2_value_results.eval_board, o_vs_x2_value_board...)

# ╔═╡ da65c461-6ed8-44e1-b159-d3a86db7081f
@bind x_vs_o2_value_board TTTBoard()

# ╔═╡ 4c45762d-3f7d-4545-80ef-c4cefa3885a5
style_value_policy(x_vs_o2_value_results.eval_board, x_vs_o2_value_board...)

# ╔═╡ 9822e5de-0a95-4021-9d61-e43ea976a7c7
@bind selfplay_value_board TTTBoard()

# ╔═╡ baca1c58-1efb-43cb-8764-dcffc371b089
style_value_policy(selfplay_value_results.eval_board, selfplay_value_board...)

# ╔═╡ 6e768875-389d-40b8-997a-7c58f7d9592a
@bind policycompboard TTTBoard(cellsize = 80)

# ╔═╡ cb649c42-25a8-4b06-86be-3ab60a319156
comp1displayboards = makecompboard_display(policycompboard[1], [selfplay_value_results, ttt_selfplay_results, x_vs_random_value_results, x_step_vs_random_results, o_vs_random_value_results, o_vs_x1_value_results, x_vs_o_value_results], ["value iteration selfplay", "actor/critic selfplay", "value iteration x vs random", "actor critic vs random", "value iteration o vs random", "value iteration o vs x1", "value iteration x vs o1"]; cellsize = 70)

# ╔═╡ cc006471-8181-4cc2-a8ea-87ae865d7b35
displayboards(comp1displayboards.htmlboards)

# ╔═╡ febfb442-0f24-4c34-8104-ecbe43eeefc5
#=╠═╡
function makeboardselector() 
	PlutoUI.combine() do Child
		makechild() = @htl("""<div>$(Child(Select([0x00 => "", 0x01 => "X", 0x02 => "O", ])))</div>""")
		makechildren() = mapreduce(a -> makechild(), (a, b) -> @htl("""$a \n $b"""), 1:9)
		children = makechildren()
		@htl("""
		<div class = "button-grid">
			$(children)
		</div>
		<style>
			.button-grid {
				display: grid;
				grid-template-columns: repeat(3, auto);
				width: 100px;
				height: 100px;
			}
		</style>
		""")
	end
end
  ╠═╡ =#

# ╔═╡ ff6be080-7aed-4a8d-878a-13f447fd48ce
function color_board(boardname::AbstractString, action_prbs::AbstractVector{T}) where T <: Real
	mapcolor(x) = round(Int64, 255*(x .^(1/2)))
	colors = [begin
		c = mapcolor(x)
		"rgb($(40), $(max(40, .9*c)), $(40))"
	end
	for x in action_prbs]
	color_board(boardname, colors)
end

# ╔═╡ 369c4b38-3753-4931-a7f3-9a4f6ecf2fb8
function color_board(boardname::AbstractString, colors::AbstractVector{T}) where T <: AbstractString
	"""
	<style>
		$(mapreduce(a -> colorcell(boardname, a...), joinelements, enumerate(colors)))
	</style>
	"""
end

# ╔═╡ 6ada97f9-c1aa-4568-a951-b343e3a7d8ed
#=╠═╡
function compare_value_iter_results(result1::NamedTuple, result2::NamedTuple)
	π1 = result1.optimal_policy
	π2 = result2.optimal_policy
	inds = 1:size(π1, 2) |> Map(j -> any(π1[i, j] != π2[i, j] for i in 1:size(π1, 1))) |> collect |> findall

	isempty(inds) && return md"""Policies are identical"""

	boards1 = [make_ttt_board_raw(symmetric_boards[i]; cellsize = 20) for i in inds]
	boards2 = [make_ttt_board_raw(symmetric_boards[i]; cellsize = 20) for i in inds]

	colors1 = [color_board(boards1[i].id, π1[:, inds[i]]) for i in eachindex(inds)]
	colors2 = [color_board(boards2[i].id, π2[:, inds[i]]) for i in eachindex(inds)]

	boards_html = [HTML("""<div> <div style = "display: flex;"><div>$(round(result1.final_value[inds[i]], sigdigits = 2)) $(x[1].board)</div><div>$(round(result2.final_value[inds[i]], sigdigits = 2)) $(x[2].board)</div> <div style = "background-color: white; width: 2px; margin: 5px; height: 90px;"></div></div> </div>""") for (i, x) in enumerate(zip(boards1, boards2))]
@htl(
"""
<span>
States for which the policies differ.  Above each policy heatmap is the corresponding value function
<div style = "display: flex; flex-wrap: wrap;">
$boards_html
</div>
$(HTML(reduce(joinelements, colors1)))
$(HTML(reduce(joinelements, colors2)))
</span>
"""
)
end
  ╠═╡ =#

# ╔═╡ 6c8d5a2c-7a4d-49e5-961f-8729ab76273a
#=╠═╡
compare_value_iter_results(value_iter_x_vs_rand, value_iter_x_vs_iter1)
  ╠═╡ =#

# ╔═╡ 1ff8a3a5-8b93-4819-97d2-e7f73abf3dcc
#=╠═╡
compare_value_iter_results(value_iter_o_vs_rand, value_iter_o_vs_iter1)
  ╠═╡ =#

# ╔═╡ 9cf9c48f-6ca7-41aa-bc9b-0cf5d404e86a
#=╠═╡
compare_value_iter_results(value_iter_x_vs_iter1, value_iter_x_vs_iter2)
  ╠═╡ =#

# ╔═╡ ee91b5ea-8bd6-4a1c-84cc-e1e6b9114c45
#=╠═╡
compare_value_iter_results(value_iter_o_vs_iter1, value_iter_o_vs_iter2)
  ╠═╡ =#

# ╔═╡ 09c92c9d-9ba6-4ef6-a688-6a6c0e2486f7
#=╠═╡
compare_value_iter_results(value_iter_x_vs_iter2, value_iter_x_vs_iter3)
  ╠═╡ =#

# ╔═╡ fa482e14-aa0c-4ae5-a894-e891f16cc3fd
#=╠═╡
compare_value_iter_results(value_iter_o_vs_iter2, value_iter_o_vs_iter3)
  ╠═╡ =#

# ╔═╡ 5caf9528-6ae5-41ca-964e-4295d4010ac4
#=╠═╡
compare_value_iter_results(value_iter_x_vs_iter3, value_iter_x_vs_iter4)
  ╠═╡ =#

# ╔═╡ 5c83b11b-7756-4b35-895b-ff08c1260f0b
#=╠═╡
compare_value_iter_results(value_iter_o_vs_iter3, value_iter_o_vs_iter4)
  ╠═╡ =#

# ╔═╡ b83d7292-5082-43d5-bc3b-fa6c93c67a32
#=╠═╡
compare_value_iter_results(value_iter_x_vs_iter4, value_iter_x_vs_rand)
  ╠═╡ =#

# ╔═╡ 5eaebc86-e45b-4106-a7ed-2ce4f16728c5
#=╠═╡
compare_value_iter_results(value_iter_x_vs_o1_alt, value_iter_x_vs_rand_alt)
  ╠═╡ =#

# ╔═╡ 0950e770-bd38-4cf4-81e1-7e5e3bbb315f
#=╠═╡
compare_value_iter_results(value_iter_o_vs_x1_alt, value_iter_o_vs_rand_alt)
  ╠═╡ =#

# ╔═╡ 11c416e0-d21c-4571-99af-fb8bfdfb42e0
#=╠═╡
compare_value_iter_results(value_iter_x_vs_o2_alt, value_iter_x_vs_o1_alt)
  ╠═╡ =#

# ╔═╡ 56926c91-558f-4199-b84e-c1cf9260b435
#=╠═╡
compare_value_iter_results(value_iter_o_vs_x2_alt, value_iter_o_vs_x1_alt)
  ╠═╡ =#

# ╔═╡ ac994fa8-e45b-4656-a33f-e2e221044255
#=╠═╡
begin
	function show_tabular_policy(board::BoardTTT, policy::Matrix; cellsize = 100)
		sym_board, rot = get_symmetric_board(board)
		(board_viz, id) = make_ttt_board_raw(board)
		board_index = symmetric_board_index[sym_board]
		policy_vector = policy[:, board_index]
		rot_inds = d4_inverted[rot]
		policy_vector_rot = policy_vector[rot_inds]
		# @info "rotating indices with $rot_inds"
		# @info "Original policy vector $policy_vector"
		# @info "Updated policy vector $policy_vector_rot"
		color_style = color_board(id, policy_vector_rot)
		resize_style = resize_board(id, cellsize)
		HTML("""	
		$board_viz
		$color_style
		$(@htl("""$resize_style"""))
		""")
	end

	function show_tabular_policy(board_index::Integer, policy::Matrix; cellsize = 100)
		sym_board = symmetric_boards[board_index]
		(board_viz, id) = make_ttt_board_raw(sym_board)
		color_style = color_board(id, policy[:, board_index])
		resize_style = resize_board(id, cellsize)
		HTML("""	
		$board_viz
		$color_style
		$(@htl("""$resize_style"""))
		""")
	end
end
  ╠═╡ =#

# ╔═╡ ef532a60-e84c-45fd-8b83-e3ee77618410
#=╠═╡
color_board(value_iter_fixed_board[2], value_iter_board1_result.optimal_policy[value_iter_rotated_board.rot_inds, value_iter_rotated_board.index]) |> HTML
  ╠═╡ =#

# ╔═╡ 3d96abfe-681f-438c-8f7f-992d2b963c3f
#=╠═╡
color_board(value_iter_fixed_board2[2], value_iter_board2_result.optimal_policy[value_iter_rotated_board2.rot_inds, value_iter_rotated_board2.index]) |> HTML
  ╠═╡ =#

# ╔═╡ f7392c49-ee86-40ce-b7f1-b31e72326837
function show_value_policy(value_policy, board, boardname)
	v, prbs = value_policy_output(value_policy, board)
	c = color_board(boardname, prbs)
	htmlstr = if isa(v, Real)
		joinelements(c, value_board(boardname, round(v, sigdigits = 2)))
	else
		c
	end
	HTML(htmlstr)
end

# ╔═╡ 0a9a139e-88c2-4405-8ada-61c57e39928f
function heatmap_board(board, actions = zeros(9); cellsize = 100)
	hash_str = hash((board, actions))
	#push up non zero colors above linear range
	mapcolor(x) = round(Int64, 255*(x .^(1/2)))
	colors = [begin
		c = mapcolor(x)
		"rgb($(40), $(max(40, .9*c)), $(40))"
	end
	for x in actions]

	function makecell(i)
		"""
		.grid-container$hash_str .gridcell.cell$i {
			background-color: $(colors[i]);
		}
		"""
	end

	joinstr(a, b) =  """$a \n $b"""

	function makehtmlcell(i, v)
		str = if v == 1
			" x"
		elseif v == 2
			" o"
		else
			""
		end
		"""<div class = "gridcell cell$i$str" data-cell$hash_str></div>"""
	end

	htmlcells = mapreduce(i -> makehtmlcell(i, board[i]), joinstr, eachindex(board))

	cells = mapreduce(i -> makecell(i), joinstr, eachindex(actions))

	gridstr = is_o_move(board) ? "o" : "x"
	
	HTML("""
	<span class = "$hash_str">
	<div class = "grid-container$hash_str $gridstr" id="grid-container$hash_str">
		$htmlcells
	</div>
	
	<style>
		body {
			margin: 0;
		}
		.grid-container$hash_str {
			width: 100vw
			height: 100vh;
			display: grid;
			justify-content: center;
			align-content: center;
			grid-template-columns: repeat(3, auto);
			background-color: rgb(31, 31, 31);
		}

		.grid-container$hash_str .gridcell.x::before,
		.grid-container$hash_str .gridcell.x::after,
		.grid-container$hash_str.x .gridcell:hover:not(.x):not(.o)::before,
		.grid-container$hash_str.x .gridcell:hover:not(.x):not(.o)::after {
			content: '';
			position: absolute;
			width: 10px;
			height: 90px;
			background-color: black;
		}

		.grid-container$hash_str .gridcell.x::before,
		.grid-container$hash_str.x .gridcell:hover::before {
			transform: rotate(45deg);
		}

		.grid-container$hash_str .gridcell.x::after,
		.grid-container$hash_str.x .gridcell:hover::after {
			transform: rotate(-45deg);
		}

		.grid-container$hash_str .gridcell.o::before, 
		.grid-container$hash_str.o .gridcell:hover:not(.x):not(.o)::before
		{
			content: '';
			background-color: rgba(1, 1, 1, 0);
			border: 10px solid black;
			height: 70px;
			width: 70px;
			border-radius:50%;
		}

		.grid-container$hash_str.x .gridcell:hover:not(.x):not(.o)::before,
		.grid-container$hash_str.x .gridcell:hover:not(.x):not(.o)::after {
			background-color: gray;
		}

		.grid-container$hash_str.o .gridcell:hover:not(.x):not(.o)::before {
			border-color: gray;
		}
		
		.gridcell {
			border: 1px solid black;
			height: 100px;
			width: 100px;
			display: flex;
			justify-content: center;
			align-items: center;
			position: relative;
			cursor: pointer;
		}

		.gridcell.x, .gridcell.o {
			cursor: not-allowed;
		}

		.gridcell:first-child,
		.gridcell:nth-child(2),
		.gridcell:nth-child(3) {
			border-top: none;
		}

		.gridcell:nth-child(3),
		.gridcell:nth-child(6),
		.gridcell:nth-child(9) {
			border-right: none;
		}

		.gridcell:nth-child(7),
		.gridcell:nth-child(8),
		.gridcell:nth-child(9) {
			border-bottom: none;
		}

		.gridcell:nth-child(1),
		.gridcell:nth-child(4),
		.gridcell:nth-child(7) {
			border-left: none;
		}

		$cells
	</style>
	$(resize_board(hash_str, cellsize).content)
	</span> 
	""")
end

# ╔═╡ 95c6b961-bbd2-4ab2-b948-5f3b19b86138
@bind board4raw heatmap_board("fjehjkwio6786fe", zeros(9), ones(9))

# ╔═╡ fac6efe9-2242-4da0-b491-d3055fe9c3dd
board4 = UInt8.(board4raw)

# ╔═╡ 84f191f1-0d5a-44dd-991e-6cfdfd2dbbee
checkboard(state_symmetry_lookup[mapboard(board4)][1])

# ╔═╡ a242cca3-e425-4dae-8908-f27c571a7f3a
show_policy(board, f) = heatmap_board(hash(f), board, f(board))

# ╔═╡ 4dff1499-5799-4185-aee6-f8285009b2de
(value = minimaxvalues[state_symmetry_lookup[mapboard(board4)][1]], actions =  show_policy(board4, s -> apply_sym_π(minimax_policy, board4)))

# ╔═╡ 41fc0344-fac6-4231-b12f-c4eb27598b38
function eval_value_policy(board, results, name)
	(newplayboard, inds) = state_symmetry_lookup[mapboard(board)]
	invertinds = [findfirst(inds .== i) for i in 1:9]
	!haskey(results[3], newplayboard) && return (value = "Not a valid state for first player", actions = heatmap_board(name, board, zeros(9))) 
	πs = convertπs(results[3][newplayboard])
	(value = results[1][end][newplayboard], actions = heatmap_board(name, board, [haskey(πs, UInt8(a)) ? πs[UInt8(a)] : 0.0 for a in 1:9][invertinds])) 
end

# ╔═╡ 5328c966-d0ca-4e02-bfcf-9a585fe8a6c6
eval_value_policy(board4, selfplay_ttt_value_results, "value_selfplay")

# ╔═╡ b54bf47b-1278-4d21-a9dd-9b9e5a9f66fc
# ╠═╡ disabled = true
#=╠═╡
#for displaying plots that do not load by default when the notebook first runs.  Displays a placeholder markdown and then if the counter is more than 0 runs the function f with the provided arguments and caches the result in the appropriate dictionary
function show_or_lookup_plot(buttoncounter::Integer, args::Tuple, kwargs::NamedTuple, dict::Dict, f::Function, name::AbstractString)
	buttoncounter == 0 && return md"""
								 #### Placeholder for $name plot.  Click above button to run
								 """
	haskey(dict, (args, kwargs)) && return dict[(args, kwargs)]

	p = f(args...; kwargs...)
	dict[(args, kwargs)] = p
end
  ╠═╡ =#

# ╔═╡ acfabeef-f268-4c7f-a07c-4c05d1333305
@fromparent import *

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

[compat]
BenchmarkTools = "~1.6.3"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
PlutoDevMacros = "~0.9.2"
PlutoPlotly = "~0.6.5"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.79"
ProgressLogging = "~0.1.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "f741a345525e5271ed5191bd30efa830f4aa1961"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

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
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "7fecfb1123b8d0232218e2da0c213004ff15358d"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.3"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "b7231a755812695b8046e8471ddc34c8268cbad5"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.0"

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
git-tree-sha1 = "6522cfb3b8fe97bec632252263057996cbd3de20"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.18.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

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
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

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
git-tree-sha1 = "80580012d4ed5a3e8b18c7cd86cebe4b816d17a6"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.9"

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
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

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
git-tree-sha1 = "8acd04abc9a636ef57004f4c2e6f3f6ed4611099"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.5"

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
git-tree-sha1 = "3ac7038a98ef6977d44adeadc73cc6f596c08109"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.79"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

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
# ╟─e7c4aad1-562f-4027-b64b-39859ac8abbb
# ╟─00e79eb9-93aa-4afc-b665-aa410a1d3a9b
# ╠═52e59796-5eef-4630-98b8-65e62429abee
# ╠═5ffdff02-3de1-4761-beb3-b701501a6fc3
# ╠═be93b217-b9de-4183-a6d3-dd4574808597
# ╠═38c5c0a3-1605-4c43-8d8a-efd1d8e286c5
# ╠═83509b3d-3acf-4b0e-b661-448f2131c906
# ╠═8fe13e56-a6ec-4ee9-bbca-7df6ef410e73
# ╠═a10a69c9-7a16-4035-8821-e0f82fb24d1b
# ╠═97eb046f-e053-4547-a8ec-e763e8a025e1
# ╠═0618bd90-e6fc-468b-bb9b-219de9e32613
# ╠═cd542047-0eb4-40df-9d2f-1f7830b7f817
# ╠═b6aa747d-56d5-43b1-aff8-adf59c91ce66
# ╠═f2f5f23a-8484-4ea3-b7dc-1f1d5d64ba60
# ╠═141b0b0b-28c5-48d7-962e-c203305a0565
# ╠═293d5ea2-342b-4005-939a-5b924282f83f
# ╠═3993d911-73cb-49db-9478-c4c6eb9f4ce9
# ╠═6c68868a-30f0-448b-8a42-84ce586145c7
# ╠═4926525c-1584-4ce0-84a4-2d1d7bad8b9b
# ╠═6705c31c-f92f-48ad-b0f4-274fce082359
# ╠═ad4a926d-6238-4dcd-85c9-5184a66a91f1
# ╠═97a8c4a3-6191-4c34-bdfc-8d8f6a06e33a
# ╠═26a5e525-d69b-4e84-aa14-a3d273ae3333
# ╠═79108d50-064a-4eba-9044-53b2ca6f6ac6
# ╠═99c1e2e6-743b-4112-baae-1360a8b39a5d
# ╠═6545da33-7250-42a9-b1a2-77cd8cc5f71f
# ╠═e84b32f3-dd81-4dbe-a80d-c198a08e0041
# ╠═a229d28f-2026-457f-8236-8002f9caa150
# ╠═62bd6663-101d-45c6-8aad-0425da307d3a
# ╠═5bd479bd-35a5-4ff8-b534-95bd628c9af9
# ╠═02237d67-ada1-488e-bfff-f5f51fca757d
# ╠═cee75d80-e58c-475e-ade5-e3c609a2fe24
# ╠═6a6ba736-df06-4049-a1b5-fb5b2a3e3bb5
# ╠═bc259929-d37b-441b-af3d-a0f56f6e7387
# ╠═8678870f-3a3a-4293-8295-a53b08cc1549
# ╠═2c17bef9-da02-4c9a-8aff-72176ed3ec10
# ╠═d5917084-6b70-4158-a32d-a2a9d375ac8a
# ╠═857d205e-ff6d-4709-9b2e-f010bee240d1
# ╟─cfbf20c5-9605-4595-952a-a90cabadab65
# ╠═c28b68c9-a732-4019-aff4-6d80bd595957
# ╠═a99f68ab-9f7d-474f-ae0e-6e99c707006e
# ╠═a514f581-b3ec-41dd-a783-d60fbc56c255
# ╠═830cfb4b-2bc5-425a-b9b6-a8b6fd0ced63
# ╠═461a1484-3fa3-42f3-a26c-8949117e6b44
# ╠═5b744d12-e0af-4864-9409-3927bfb700d6
# ╠═e25a0c4b-0a84-420d-a7e3-f18a7014aa2d
# ╠═70b2d2db-bafc-47bb-ba32-28449ec1dd9e
# ╠═58f14689-1add-48f3-b1d9-bab6d05ac992
# ╠═40bf9e7e-67d5-486c-9fbd-f17f4cdba947
# ╠═31538c97-d99c-43db-8a32-dbc6e2726920
# ╠═b375c7eb-3b4c-4dff-af90-e6814b19cd34
# ╠═de2d1f87-1530-4d31-9d8c-9af007a44bec
# ╠═01a099ff-c9cf-44d5-827a-f8d587871dcb
# ╠═fa1ee379-3476-4636-83b4-3af562f52050
# ╠═52ee5e7b-19f4-4017-a803-ead3fd45b082
# ╠═85dfb79c-1a06-45a6-af11-26b688ea0b3e
# ╠═d0d5b687-cff4-48e4-ac6c-bf19d866aaca
# ╠═835ee149-b79b-4293-ba1f-714aa76eb141
# ╠═844d31b2-373e-48ef-882d-eb73e6b10391
# ╠═7b807838-a724-4f18-b90c-e6cfd52f38aa
# ╠═b823c99e-fd6e-46dd-bc24-f080befdf922
# ╠═287d846d-4eba-4745-adaa-613c638262da
# ╠═d78c7ca7-7125-4547-804a-a8cd193746ed
# ╠═a93fb40b-7eda-48f6-a2ea-4afce95140a0
# ╠═216b2401-a416-4a6e-9812-7ef7061b55b9
# ╠═36deaa30-0485-4012-a838-1011161e7a3e
# ╠═81ae3b51-b666-4c51-b17b-32bea5b99357
# ╠═ad943e14-28a1-406c-9f98-d8572dfa16a6
# ╟─d52f66cd-c4c2-4750-b182-42e08a9a27f4
# ╠═b1df7a1e-5319-41d2-b8cd-ca25d55fe1ea
# ╠═14b5a3cd-31d9-4392-bbb8-25286bcc2ff8
# ╠═3056e679-8006-4396-8bb4-e76cfd909970
# ╠═d90008a6-e2eb-4e1e-b9ce-9a909ff0325f
# ╠═f215e88d-8029-4c54-873c-e8413f32a04b
# ╠═ae128a72-bf33-40eb-8a47-3b6569a3af7b
# ╠═11e24ec6-4222-425b-afc1-5d44948b2392
# ╠═22406fa6-fb9b-4d9e-84d8-daa336542563
# ╠═fef5f0c4-46d7-477c-a508-3c221129cc60
# ╠═871202dd-4fe5-4a04-b11b-02534c729d8c
# ╠═21480455-a98c-4d41-91ba-08d069cb2cbf
# ╠═aa250079-6cc1-4f3e-acf3-a0b0bce75c7c
# ╠═109a9bfd-92bb-4307-9d3e-0bfb737082ab
# ╠═9d9ba389-4346-4d37-a5c7-b6a6cfddaedc
# ╠═f4c9243b-ea5c-45d2-9eea-a6793e8081dd
# ╠═2e57e843-0ab2-4338-b35b-afa5f606453b
# ╠═bc9965d2-437a-40e7-bced-30ccbb7867b7
# ╠═68e65fd7-2885-4c28-9f58-bdc62d0fb9d5
# ╠═429bbe6d-3126-4e6f-852a-a232f129300c
# ╠═5fd3dfc2-3e47-474a-8b96-61459b05b4e7
# ╠═efc0ed38-24dd-4d4a-ae44-a2dfd8bc239d
# ╠═3ca15d74-a36d-4b45-b57c-cecd6c2a84c2
# ╠═494879df-0ecc-4610-9206-e5362bbf96d9
# ╠═477452bc-12ff-4cea-bc3b-2207ec0a907d
# ╠═a66beb65-f590-4c76-a5a8-8ba9e5829f04
# ╠═585d07a0-66e8-47dd-a655-845bbed0e79a
# ╠═d8d3d75e-b35d-4077-a953-f2a83958b48a
# ╠═2f7f4c11-ecd4-40bf-bcf1-62a63e81c68b
# ╠═6ada97f9-c1aa-4568-a951-b343e3a7d8ed
# ╠═d92d52c1-d38a-4b03-9c59-e542c2bdde13
# ╠═6c8d5a2c-7a4d-49e5-961f-8729ab76273a
# ╠═1ff8a3a5-8b93-4819-97d2-e7f73abf3dcc
# ╠═9cf9c48f-6ca7-41aa-bc9b-0cf5d404e86a
# ╠═ee91b5ea-8bd6-4a1c-84cc-e1e6b9114c45
# ╠═09c92c9d-9ba6-4ef6-a688-6a6c0e2486f7
# ╠═fa482e14-aa0c-4ae5-a894-e891f16cc3fd
# ╠═5caf9528-6ae5-41ca-964e-4295d4010ac4
# ╠═5c83b11b-7756-4b35-895b-ff08c1260f0b
# ╠═b83d7292-5082-43d5-bc3b-fa6c93c67a32
# ╠═5a962d71-b56b-4166-8320-71c6baed452b
# ╠═83bdfbb5-ed4e-4d67-9db5-15037c788a65
# ╠═57dfcf6f-572b-4029-9c65-cb56b1d1b6bd
# ╠═712f33b8-8774-46c6-973e-48c765799e6e
# ╠═1e4cbd58-5b9b-4382-9150-9ef8afd331bb
# ╠═fd7742c5-472d-46ac-b885-c720326a82c5
# ╠═b98ceb8e-5780-40a3-95bc-1e8df86466b6
# ╠═5eaebc86-e45b-4106-a7ed-2ce4f16728c5
# ╠═0950e770-bd38-4cf4-81e1-7e5e3bbb315f
# ╠═10868e8b-f243-45bd-a21e-94b5a7f06f44
# ╠═c33be92f-9be3-4bb2-82e2-4ac95f2467f1
# ╠═73041a13-49da-4bd6-a233-5bb88b4f5fd4
# ╠═11c416e0-d21c-4571-99af-fb8bfdfb42e0
# ╠═56926c91-558f-4199-b84e-c1cf9260b435
# ╠═ac994fa8-e45b-4656-a33f-e2e221044255
# ╠═9cf13709-615b-4183-8397-c146f7f91252
# ╟─cb50cf2c-6adf-42de-be9a-085676876bbe
# ╠═2180c6dc-1f6d-4976-bd0f-cc8153e6b87d
# ╠═4a940b85-8f56-414a-a7a8-b2895605ae45
# ╠═18f5244f-c88a-4e84-9b5b-a8d8c2b76b23
# ╠═4ef908c9-a2a7-4f85-b98d-c9dab9fc3ac7
# ╠═8b6786f7-220c-4daa-b707-376c2c7bc14c
# ╠═9d2120cd-5582-4fa6-9a52-bad1121d9da3
# ╟─32ee92b1-a61a-460a-b436-036ef9cef080
# ╠═bbfa172b-e519-4ee6-9410-935a8baf3cbf
# ╟─2584f0cf-ad4c-4dca-86a3-357ee024c52c
# ╠═bc1b2683-89b1-4813-bf94-2fd41d23e6ed
# ╠═2a67bf2c-09d1-4dc3-b4f6-200ad9be940e
# ╠═124519d9-b8b3-492b-a3af-36c48553bc52
# ╠═7717eb2e-1bf7-4953-8892-d8f936a85756
# ╠═ef532a60-e84c-45fd-8b83-e3ee77618410
# ╠═3d96abfe-681f-438c-8f7f-992d2b963c3f
# ╠═b8a5f496-5727-48ab-804f-3dc027cca0f1
# ╠═7479eb78-0aab-4a28-b433-968aec5b980b
# ╠═c23d9164-d96d-43fe-b8de-f1b7c6c6d30f
# ╠═6fea4c9f-b70e-4e64-92bf-0436c1203516
# ╠═079f86e4-337e-4cf2-a50b-042e351f65c7
# ╠═8ff4ceda-e4df-4652-b48f-4b9fe4bbce27
# ╠═4016eb02-ff4c-46dc-8167-d8c2a05f4a1f
# ╠═45be8610-a480-435c-b4e4-33aff7f3aead
# ╠═68b8fd79-09f9-43c5-bf53-b5b875d63c3f
# ╠═74c3970f-49a7-4e35-bcb0-e4fe53c9cd58
# ╠═ec450941-30ec-4bd1-8628-330760287e5f
# ╠═1d7fccfb-10d3-47ae-a8f6-55d4a81344da
# ╠═91e35000-afb7-4170-9923-4ea25b157ae2
# ╠═90255ba0-304e-4ca8-a1e7-28bfc30ca0cd
# ╠═b1a5a634-9673-4c72-9a71-4da9dba8ea83
# ╠═af40d78d-4860-4f56-9ce1-ba4e631f7ffd
# ╠═be80db21-c043-4a9b-8473-f2ab4a1e04ce
# ╠═5c59c4a2-e4d7-4ed7-9daa-10a12f5378c2
# ╟─cb12f8b6-cec1-40fe-9b29-0acae1d3291b
# ╟─16ebebb0-fe2f-4469-a834-8b7375cdb0dd
# ╠═37f279ed-c1d6-446f-9e63-89553ad4f721
# ╟─dec67e3a-d798-48c8-b58a-9cfdfd057d36
# ╟─b8044db9-73af-4834-948d-9c2a800d251d
# ╟─1aa7f969-7382-4ec4-99ad-af9a7c8f8354
# ╟─aef1ba88-8b77-4f12-a88c-56ec3e6bdbc7
# ╟─c180b1a8-9540-4da7-9ba8-932c8d48b8f2
# ╟─9701cad5-7897-40ed-a3b4-9e9ead8bd790
# ╟─e65a55a2-a7ba-4c65-82f7-d3c04f168ccc
# ╟─15d3b0d1-a6a5-4524-a07a-5dd2580aac92
# ╟─6894f2a4-b844-434b-80cf-03b52558e043
# ╟─3a84f6ff-e6ac-42a5-8939-420430defeb2
# ╟─88de5309-557b-4ba9-b627-90495d769747
# ╠═2e454bf5-62fc-46fc-8076-74f1a6729fa7
# ╟─0d3b384b-c602-4ccd-b335-987e378a4230
# ╠═90240ffc-fd88-44e6-9b6c-367361326996
# ╠═2459ba2e-b0ba-4271-9b3e-7cb0e2b847fa
# ╠═a5df2ddc-fb88-4374-b2b2-28217cc8d1ee
# ╠═24b430a3-b8b7-4ef1-8050-347b1dd12ac7
# ╠═b795d8a3-9b54-44d9-9ce6-112651f39cd1
# ╠═a651f37c-ef11-445f-8839-a31112559c2e
# ╠═e3d6ff52-4084-47b1-aeb1-d36a161882b0
# ╠═ff61ffce-612f-4bb5-ba45-fe8715ceb180
# ╠═436c72df-c0a1-428c-a005-25381c758deb
# ╠═3c4bc244-d91f-48f5-9b74-99b1a5d242f7
# ╠═e2ae92dc-34ec-48fa-a958-2d6c0170cc79
# ╠═dc8f8c7e-05c4-4c97-8d37-109b56b7f1fa
# ╠═666169e0-f745-4858-a28e-9a7217fdd092
# ╠═0e309975-5338-426e-85a8-04839cfba5de
# ╠═d1b2f039-d3c1-4709-8a53-43dc85b10f50
# ╠═1b5f9e37-7ba4-44ed-a4dd-62ec0a0adafe
# ╠═0a623b8c-a462-4742-9141-d2cf75eae19b
# ╠═79c02db7-709e-407b-86fc-571951ac7d8b
# ╠═be81fcbd-45b0-4555-b68c-ababe164fbd3
# ╠═82b948a9-be98-409a-8d2a-e7ff73376507
# ╠═5b8adffa-0ef5-4bcd-95c3-01e62f6c6186
# ╠═008a1ccb-7155-43f8-a971-b798144f6a57
# ╠═12d0c927-3733-4bf8-94d4-fbfc82dab098
# ╠═255d5d51-08b9-43f2-a373-1fd9a70374eb
# ╠═ecde9687-8a3d-4acb-b854-50c030febf1c
# ╠═d618c9fd-8d82-49b6-be5c-bdef436c9ef0
# ╠═844d3cfd-6d40-4bba-bc2d-2cdc8c6130c2
# ╠═afecb204-21e2-4971-aa73-ded9f8c5ce02
# ╠═18f69434-59c5-415b-b222-e4d932dfa2b1
# ╠═c98cab58-64cb-4a05-99c1-4398989babd2
# ╠═2d93e5d4-77d0-4542-ada2-c488975ff5cc
# ╠═a417ae4c-7b97-442e-895b-ddd5eaa7f4e6
# ╠═fb917755-0db1-436c-9a07-3e1529d6c144
# ╠═c6dec053-7407-45ca-93fd-f0b893bb1345
# ╠═81622974-4b91-4f26-bb60-01edebfe4e2b
# ╠═130facfd-c121-45e5-ad49-06d2d5a63976
# ╠═d235d68a-ecf9-46c5-839e-96727a3b3226
# ╠═a3dd5b6b-feed-4488-aca9-8f5e6a5caf3a
# ╠═a4f13c4a-152e-4aef-85a8-874f0cd0abaa
# ╠═4f03e182-81d6-422a-9a85-667854264a8e
# ╠═45c9a02d-a7ce-49ab-b085-44bd40090b29
# ╠═32485151-c606-4483-8d8c-4c36b87f9158
# ╠═c9169432-ba22-4bf9-aa51-ac633cac69f7
# ╠═d4de69bd-f3b4-4cf5-a3c7-3a95d28052c6
# ╠═558f82e6-8b24-4a04-ba0e-48fdfb0db98c
# ╠═344dd837-021d-44ed-8a1d-ad5a295438be
# ╠═58b93ab0-44df-4594-8941-a09f312def82
# ╠═5afd7bb9-cf9d-4495-a712-2d6713595fb5
# ╠═795b1315-cba8-4548-a609-1873e6cc34eb
# ╠═e3711fc9-10fd-4802-aaf3-b70bed5e4597
# ╠═f62c6c04-c97c-4dc7-8270-44eaca693cf7
# ╠═eaa5da0c-a2a1-4c87-9a83-791dacd67765
# ╠═4e522a97-b341-40cc-be52-065f265f2a3e
# ╠═a7340ee4-9b64-4b50-ba63-6b8682318f74
# ╠═36ea7865-bfd9-4ec1-bda4-213ee2f09466
# ╠═7613cd78-fa2f-4dc1-9c6d-4331823f5371
# ╠═4ec75c16-e19e-4609-ba45-296e8ed5e694
# ╠═a61157e7-2f22-4e99-a9ef-32f302d4b60b
# ╠═cac650e8-ee30-4068-a50f-28e4f7203323
# ╠═345fe2cc-8d6a-49a3-959d-e7b21410bc43
# ╠═95da63e5-edbc-49a2-9a34-1ccf297c04e2
# ╠═8c409991-1729-47c2-8cf4-598279f37fc7
# ╠═6a0664ed-b3c8-445f-b355-13b1c23630dd
# ╠═6f7b66e4-9fe5-43d7-b0c6-9410c1f0ec8f
# ╠═b95837e8-942a-45ca-b3e6-22e44c2942f4
# ╠═a0f01732-fbb5-4bf3-a6a8-4ef8794ee572
# ╠═da6ff1d3-561c-405e-aac9-54db12e95299
# ╠═0d0e1329-ee71-46a7-a998-7c6a2cfa8e32
# ╠═a4855984-90c8-4044-9358-9ea5748772eb
# ╠═3307f5b4-0a49-4895-b6df-decdbbde28ba
# ╟─38d7c4b4-eaac-4e92-9d94-0b833ee50c3b
# ╠═bc5c2d93-24f6-4ee5-ae1a-823229ee7c5d
# ╠═9d2ea20b-3250-4bcc-bcb6-193612ab8ff8
# ╠═b13134bc-9b08-435c-8c7a-9aa736c003b3
# ╠═6ac696c7-9f40-45bb-a556-9507f0b10f87
# ╠═9cc3d72d-3706-4af8-a62c-d143fa439664
# ╠═46b0b624-c841-42f4-aa5d-9dba21c981fe
# ╠═3f1f6c2f-f377-4f38-b4c4-4df3817bb7d3
# ╠═0e51f9d7-c37e-471a-90a3-9d5ab43896c6
# ╠═4d54ba83-2489-4d51-a75e-d27c321babf7
# ╠═5753d01d-61e5-4306-a10d-c2ac6d0f8e3e
# ╠═60477da6-eff0-4c01-ab39-492fa1c1a59e
# ╠═c7f5c690-884f-4489-a9d6-64816dba925d
# ╠═a441ecdd-1837-451e-944c-b2fde5bd7276
# ╠═73f5697b-2928-4f4e-b3ff-8129c1d4f8ed
# ╠═4e08ba5a-b5b0-4e4c-b4d3-f806bbf1d13c
# ╠═75258904-d987-42bd-a9f5-69e6d4eff687
# ╠═6b5731f8-4d68-4007-b4fe-ac2351a532d2
# ╠═92dbe6ad-7f82-428c-b2b9-446844d8a7e0
# ╠═3e239392-9289-4c79-8a64-d80305e57ed2
# ╠═45eb3f00-4ae6-4f9b-af5e-ebb4a7c076f6
# ╠═80442053-1908-420c-b71e-f14c59da4d99
# ╠═dcedc66c-2bd5-403c-9a63-5df777014bcf
# ╠═48d8f9e2-3336-4f10-8781-ac898ca286ad
# ╠═2a77ae50-ada6-4204-ba46-6b2d7c565138
# ╠═7e92bf5b-5d34-4a7f-8002-2471692a5fe8
# ╠═908894b3-b53b-4f6b-9175-c15fc69bf2c4
# ╠═3ac17b3b-3e2d-424f-9cf6-006589ae1cc9
# ╠═18c9654e-36f4-49d6-97a5-ca03ddc511c5
# ╠═4489fdc1-2f3d-4609-8c0c-0f6c3e67c4ab
# ╠═da65c461-6ed8-44e1-b159-d3a86db7081f
# ╠═4c45762d-3f7d-4545-80ef-c4cefa3885a5
# ╠═42583b0d-dd48-4150-b7f4-a66c527326a3
# ╠═147aa41b-d390-4258-8023-eb36f4e9b7b3
# ╠═2d766054-08c1-40a4-ac79-9498fffad9ab
# ╠═b2c85ccd-7338-4eda-9491-d9f24e771dc8
# ╠═541dc7b8-951f-4296-9f54-bc74ba831bab
# ╠═96f5d75e-8b24-4bdb-82d6-d9173682633a
# ╠═0facd26f-d6af-4574-9afa-4e90b5043cb7
# ╠═759b0d57-4475-4643-ad79-fcc608846b5d
# ╠═804dd7f6-c589-4f39-ba0f-20a1b46f9aaa
# ╠═9822e5de-0a95-4021-9d61-e43ea976a7c7
# ╠═baca1c58-1efb-43cb-8764-dcffc371b089
# ╠═3070b33a-fb35-4242-ab2d-5187e49a7d75
# ╠═6e768875-389d-40b8-997a-7c58f7d9592a
# ╠═cc006471-8181-4cc2-a8ea-87ae865d7b35
# ╠═cb649c42-25a8-4b06-86be-3ab60a319156
# ╠═25a74028-3575-4f4b-8083-707af22aaea6
# ╠═79885cfc-7cda-4a8f-aaa6-ab4d09fbca91
# ╠═d8cd4939-9c00-4adf-bc29-49b6f02134bf
# ╠═0c968db5-69a4-4452-80f7-3c22fd4f93aa
# ╠═786e1340-1b1c-478f-9949-b48ed01cde33
# ╠═9a3059b0-ec9a-4c1b-beb1-901b2285feaa
# ╠═cb143d38-8ceb-4c8d-aa59-8e4df8a67f3c
# ╠═94c3c1be-caf2-4093-b795-b1b3a80ffc81
# ╠═4686cbb3-a6d0-49de-b7d6-7a6b6873cfe9
# ╠═183b9738-4f81-423b-93ae-f0be2fdc190d
# ╠═f1a47d6b-f394-470d-b92b-1d2efd0526b4
# ╠═3bceb7cb-acd6-49e6-bf33-02e377cf19f5
# ╠═95c6b961-bbd2-4ab2-b948-5f3b19b86138
# ╠═fac6efe9-2242-4da0-b491-d3055fe9c3dd
# ╠═84f191f1-0d5a-44dd-991e-6cfdfd2dbbee
# ╠═d056f230-0310-48cc-98ea-dddc60f2a261
# ╠═5328c966-d0ca-4e02-bfcf-9a585fe8a6c6
# ╠═63888652-d0e7-495d-9f03-4bf988441675
# ╠═4dff1499-5799-4185-aee6-f8285009b2de
# ╠═8e31c353-e7fd-4d7d-b35a-cfb011f325fd
# ╠═a242cca3-e425-4dae-8908-f27c571a7f3a
# ╠═8a334293-b568-4dcc-b9d5-cfd542a8c9f7
# ╠═345af4aa-8bbe-46ce-917c-a5237114a92e
# ╠═170474a5-7a8e-4e6c-8c3a-9d6efbd85637
# ╟─4f16565e-09bb-11f0-3729-7ffc5462cdc8
# ╠═9a4d0c70-ca15-4201-8a2e-56af95a60290
# ╠═acfabeef-f268-4c7f-a07c-4c05d1333305
# ╠═c77a29da-a2a4-4956-9795-56ce63337495
# ╠═43999574-da93-475c-9a67-e5024fb08202
# ╠═1b84943c-c8f5-4ad4-b95a-66dc818fa609
# ╟─afb3f1df-8aa7-4e57-bf03-9d901c9c2946
# ╠═b54bf47b-1278-4d21-a9dd-9b9e5a9f66fc
# ╠═baad7afc-5104-4ea9-85bf-411c8cbb5c20
# ╠═27866a8d-a26b-4602-b10b-1e8748a77399
# ╠═b015036c-d045-4749-b0e8-33735b04c328
# ╠═74250a9c-958b-4689-abbb-92764ddf8904
# ╟─28799f85-5fb7-4617-9c3a-fa307bfc25d4
# ╠═07fa2e67-42a8-4c2b-ad95-cd0f9f28e122
# ╠═b4ab7ecb-c481-475f-8bca-7cc0a98a0287
# ╠═6bc3c3ee-84e8-44a9-90bc-7c9a91624dfd
# ╠═ca4116b8-98a9-4de6-ba82-24ce27579eb5
# ╠═493504bd-17a4-4e0a-a445-effd82ca1807
# ╟─0e2591a9-77d5-41a5-90e7-f4bb750c820b
# ╠═a032a3c7-a4c2-4053-aab1-945405764f1b
# ╠═dd6d21c4-d04f-4f4b-ad71-901b50d1d23d
# ╠═ed165fdb-e2fe-4636-81f9-fdcdb2e3784a
# ╠═c117f3dd-5122-4951-bbce-365cd1f2c5fb
# ╠═5e091621-e9ac-46a5-8470-06ca5bc377f8
# ╠═cce8f5ef-9df2-4ed2-803d-7e435bb23e82
# ╠═9dd13f7f-34a9-40c2-80f1-9b115faf73c4
# ╠═91404294-9145-4314-986a-10ea23dbdda8
# ╠═eaed2a5b-944a-453f-84ad-04f3917ca27a
# ╠═8f24e187-f9a6-4579-9f20-97f33ab657ed
# ╠═af16a2a9-aa59-4c51-8e30-841eddd4aff5
# ╠═6d4217a2-5b1f-46bb-95d1-f73741a24690
# ╠═af8c173b-c310-4d3d-80cb-11343878025a
# ╠═6a215dfb-8869-4efa-8708-99813b13c296
# ╠═a90bbcb3-2119-41c3-b06b-66f59d60d0a5
# ╠═e42a7b19-5be5-4b4e-bceb-cb4bba24ee60
# ╠═febfb442-0f24-4c34-8104-ecbe43eeefc5
# ╠═ff6be080-7aed-4a8d-878a-13f447fd48ce
# ╠═369c4b38-3753-4931-a7f3-9a4f6ecf2fb8
# ╠═f7392c49-ee86-40ce-b7f1-b31e72326837
# ╠═0a9a139e-88c2-4405-8ada-61c57e39928f
# ╠═41fc0344-fac6-4231-b12f-c4eb27598b38
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
