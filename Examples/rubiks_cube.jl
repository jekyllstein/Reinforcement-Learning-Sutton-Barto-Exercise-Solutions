### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ b22c8404-b359-44f2-b81c-e0c9f4834db5
using DataFrames

# ╔═╡ c5256762-ca4a-4c81-805b-4f865efc2091
using PlutoDevMacros

# ╔═╡ 624eef76-16a7-4556-a466-14341346f7a5
PlutoDevMacros.@frompackage @raw_str(joinpath(@__DIR__, "..", "NonTabularRL.jl")) begin 
	using NonTabularRL
	using >.Random, >.Statistics, >.LinearAlgebra, >.Transducers, >.StaticArrays, >.DataStructures
end

# ╔═╡ c5c0f635-171d-4904-9675-d1b0a01f6d7a
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI,PlutoPlotly, PlutoProfile, BenchmarkTools, LaTeXStrings, HypertextLiteral
	TableOfContents()
end
  ╠═╡ =#

# ╔═╡ def37ed7-42e8-4f07-9a26-e0a0dc2bf9ea
md"""
# Cube State and Orientation Conventions

A Rubik's cube has 12 distinct edge pieces and 8 distinct corner pieces.  The 6 center pieces, one for each face, have an orientation with respect to eachother that cannot be changed.  Therefore consider a cube always from the perspective with the center prices arranged as white, green, orange, yellow, red, blue for the sides as front, top, right, back, left, bottom.  To solve a cube all of the other colors on that face must match the center square.

Other than the 6 center facets, there are 48 other facets that can be moved with simple rotations of the faces.  While there are only 6 colors, each piece is actually uniquely defined by the color combination of either 2 or 3 colors for edge and corner pieces respectively.  A solved cube will always have these pieces in a unique position which translates into a unique position for the 48 facets.  Therefore, only the index of a facet with respect to the solved cube is needed to define the state.  The color of the facet is uniquely determined by the index and can be looked up by referring to a solved cube.  Below is an example of the values of a solved cube as well as the indices.
"""

# ╔═╡ eb46363c-35fa-4c31-b611-f349d50e167f
const solved_cube_indices = UInt8.(1:48)

# ╔═╡ f1634b1a-cc22-42e4-90af-72fc5495ed17
const layer1_inds = [1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 25, 26, 27, 33, 34, 35]

# ╔═╡ 8e58e9a2-9831-40fa-9ad3-bd82f6629bf7
const layer2_inds = [4, 5, 20, 21, 28, 29, 36, 37]

# ╔═╡ 3696b6c0-804a-11ef-0d86-7b48f7a6697c
const square_values = UInt8.(collect(1:6))

# ╔═╡ 42816516-dcf0-40b7-8787-6608fcd07831
#each column represents a face of the cube
const solved_cube_values = mapreduce(i -> fill(i, 8), hcat, square_values)

# ╔═╡ f7201b43-6c8c-4b49-a0fe-7bd1378d1649
const square_colors = ("white", "green", "orange", "yellow", "red", "blue")

# ╔═╡ c6a76e1e-2121-4d80-a1f6-ddf78b6f6d55
md"""
The following displays a solved cube with the faces and squares listed in the convention described above.
"""

# ╔═╡ f1b1c2fc-6973-4ecd-ae9c-1f52a1fd2189
md"""
# Rubik's Cube Moves

Given our orientation convention, the only moves that need to be considered are face rotations.  Each of the 6 faces can be rotated clockwise or counter-clockwise with respect to the face itself.  Any other moves can be performed by repeatedly applying these "atomic" moves.
"""

# ╔═╡ c6ccf794-d7e4-4e7e-b4f7-7acbe2530722
begin
	abstract type Direction end
	struct Clockwise <: Direction end 
	struct CounterClockwise <: Direction end 
	abstract type Face end
	struct Top <: Face end 
	struct Left <: Face end 
	struct Right <: Face end 
	struct Bottom <: Face end 
	struct Front <: Face end 
	struct Back <: Face end 
	struct RubiksMove{F<:Face, D<:Direction} end
end

# ╔═╡ 1c8bc474-9b46-4e99-b858-2a9a980f93f4
const face_order = [Front, Top, Right, Back, Left, Bottom]

# ╔═╡ ebf0eee8-b1d4-4f59-9727-518e6dd546bd
const direction_order = [Clockwise, CounterClockwise]

# ╔═╡ a9748352-649d-45c4-b33d-a4c8f2b5b247
const rubiks_moves = [RubiksMove{F, D}() for F in face_order for D in direction_order]

# ╔═╡ ebc5bf2b-abef-4b4a-9f12-b0884c089350
const rubiks_move_index = TabularRL.makelookup(rubiks_moves)

# ╔═╡ 46bbd599-ae25-485e-b064-af28efde316f
md"""
Any move can be reversed by applying  the opposite rotation to the same face.  Any move followed by its reversal will result in the same cube state.
"""

# ╔═╡ 545ca130-8e89-48db-9ada-5f6467067ef3
begin
	reverse_move(move::RubiksMove{F, Clockwise}) where F<:Face= RubiksMove{F, CounterClockwise}()
	reverse_move(move::RubiksMove{F, CounterClockwise}) where F<:Face = RubiksMove{F, Clockwise}()
end

# ╔═╡ 44e167a7-ebe7-4cb0-8e4c-9a5444f5f224
md"""
## Face Rotation Permutations

From the standard orientation, the face squares are numbered from 1 to 8 in left to right top to bottom order.  For the 4 faces perpendicular to the surface, the squares for the faces will follow that convention when the cube is rotated along the axis perpendicular to the surface to face front.  The top face will be numbered in the case of rotating that face to face the front and same with the bottom.

When a face is rotated, the color values of some squares are swapped.  This transformation can be represented by a permutation of the matrix indices.  For a given face, the indices associated with that face are permuted as follows:
"""

# ╔═╡ 7aedab25-1129-4ad5-bfc5-bd673b8d2509
const clockwise_perm = SVector{8, UInt8}([6, 4, 1, 7, 2, 8, 5, 3])

# ╔═╡ 4dd63f4c-32ce-4672-a165-6a96ef75895f
md"""
The square value in position 1 of the transformed cube, will be have the color value that was in position 6 of the original cube and so forth for the other 7 squares.

These are also permutations for the other faces, but these permutations must map both a face and position to a new face and position.  For a clockwise rotation, these permutations are defined below:
"""

# ╔═╡ 8bec2716-88f5-4998-8102-4d63b0f8f99b
const clockwise_rotation_mapping = Dict([
	Front => [(2, (6, 7, 8)), (3, (1, 4, 6)), (6, (3, 2, 1)), (5, (8, 5, 3))],
	Top => [(1, (1, 2, 3)), (5, (1, 2, 3)), (4, (1, 2, 3)), (3, (1, 2, 3))],
	Right => [(1, (3, 5, 8)), (2, (3, 5, 8)), (4, (6, 4, 1)), (6, (3, 5, 8))], 
	Back => [(2, (1, 2, 3)), (5, (6, 4, 1)), (6, (8, 7, 6)), (3, (3, 5, 8))],
	Left => [(2, (1, 4, 6)), (1, (1, 4, 6)), (6, (1, 4, 6)), (4, (8, 5, 3))],
	Bottom => [(1, (6, 7, 8)), (3, (6, 7, 8)), (4, (6, 7, 8)), (5, (6, 7, 8))]
	])

# ╔═╡ d19bf533-22ea-427d-aa63-5556ba7880b4
md"""
For example, if the "front" face is rotated clockwise, then the squares on face 2 (top) in positions 6, 7, and 8 will be mapped to face 3 (right) in positions (1, 4, 6).  To perform the full transformation, the mapping must proceed around the loop ending by mapping the squares in the 4th list item to the squares in the 1st.  The function defined below, applies this permutation to a cube filled with its indices in order instead of colors.  The resulting cube then contains the updated indices for mapping a cube to its transformed state.  The counter-clockwise transformation is just the reverse permutation of the clockwise transformation
"""

# ╔═╡ 4aeac615-6129-4482-bdc4-a0549d069d35
md"""
These functions can be used to create a lookup table with the compete index transformation for each of the 12 moves.
"""

# ╔═╡ 0051d77d-e74c-4d34-b17f-e310528ef9f4
md"""
This lookup table can then be used to efficiently produce transformed cubes by simply applying the index permutation
"""

# ╔═╡ d4613756-bc2d-4632-8e74-6fa11d5ae6a4
md"""
## Cube Rotation Function and Display
"""

# ╔═╡ bb3d3886-cbbc-4584-aed2-a53ed7f2d878
md"""
# Defining Rubik's Cube MDP

In order to define an MDP for the Rubik's cube, we need to specify the reward and initialization functions.  This problem is most naturally defined as episodic in which we start with a randomized cube and the episode only ends when a solved cube has been produced  The reward function could be as simple as -1 per step until solved.  To inject more domain knowledge into the problem setup, consider a scoring function which measures how many facets of the cube are in the solved position. Then the reward for each step can be the improvement in score and only a solved cube will attain the maximum score.
"""

# ╔═╡ 78aaa31a-66ba-4325-915c-383e648fa801
md"""
## Scoring and the Transition Function
"""

# ╔═╡ 1157c771-1121-47da-8d4a-39c10a55df85
function score_cube(cube::AbstractVector{I}) where I <: UInt8
	s = 0f0
	@inbounds @simd for i in eachindex(cube)
		s += Float32(cube[i] == solved_cube_indices[i])
	end
	return s
end

# ╔═╡ 053f6365-ad13-45f2-851c-ddcc16f98e4a
md"""
## Initialization Function
"""

# ╔═╡ cced6b32-f170-4470-8fbf-02bc2e8ad0a1
#=╠═╡
md"""Number of scrambling actions: $(@bind init_actions NumberField(1:30, default = 5))"""
  ╠═╡ =#

# ╔═╡ db9454b3-4ee0-4f16-9649-791a9e246939
md"""
# Defining a Rubik's Feature Vector

Since the state is defined by the ordering of all of the facets, it may be natural to use all of those indices as labels.  Then the feature vector may be of length 48x48=2304 if a onehot vector is used for each facet.
"""

# ╔═╡ afad0f03-8b3d-4348-9570-9fe9489f0cb5
function make_onehot_vector(i::Integer)
	v = BitVector(zeros(48))
	v[i] = true
	return v
end

# ╔═╡ 6aec543d-eec2-47f3-9dbf-2631688e1112
function update_rubiks_feature!(v::AbstractVector{T}, cube::AbstractVector{I}) where {T<:Real, I <: Integer}
	v .= zero(T)
	@inbounds @simd for i in eachindex(cube)
		j = (i-1)*48
		v[j + cube[i]] = one(T)
	end
	return v
end

# ╔═╡ cd53cf28-6542-45aa-b9dc-38fa07f3e018
function update_rubiks_feature2!(v::AbstractVector{T}, cube::AbstractVector{I}) where {T<:Real, I <: Integer}
	v .= zero(T)
	@inbounds @simd for i in eachindex(cube)
		j = (i-1)*6
		v[j + solved_cube_values[cube[i]]] = one(T)
	end
	return v
end

# ╔═╡ 343aa4fe-f38d-42b7-967b-c589be65077d
md"""
# Linear Control Techniques
"""

# ╔═╡ 7e57369f-14d5-4a5e-b83d-951228fac2eb
md"""
## Semi-gradient Sarsa
"""

# ╔═╡ f78706e3-2b14-4bc8-9b96-2e4e98a2efa4
md"""
The problem with a technique like Sarsa is that the agent never experiences a solved cube.  Once the task starts, time will be spent exploring scrambled states with no limit to the number of steps.  For using temporal difference methods like this, it may be necessary to modify the task so that an episode ends after a maximum number of time steps regardless of whether the cube is solved or not.  That way many initialization states can be used including those that rae very close to solved.
"""

# ╔═╡ 409dd109-ad0c-4a94-8722-7bc1439f2625
md"""
## Gradient Monte Carlo Control
"""

# ╔═╡ d062abca-a910-462b-992a-fe713985d644
maximum_reward(s::Vector{UInt8}) = 48 - score_cube(s)

# ╔═╡ e6a19274-9069-419e-a6e1-0feed72d1dfe
# ╠═╡ disabled = true
#=╠═╡
const test_mc_output = run_mc_rubiks_linear_test(() -> rand(1:10); num_episodes = 100_000, max_steps = 10, α = 4f-4)
  ╠═╡ =#

# ╔═╡ bf27bdef-d382-4999-972a-5f3934566c95
# ╠═╡ disabled = true
#=╠═╡
plot_rewards(test_mc_output.reward_history, 1000, 1000)
  ╠═╡ =#

# ╔═╡ cf47193d-4c64-4dd5-836e-77dd59a2d586
#=╠═╡
show_rubiks_episode(test_mc_output.π_greedy; max_steps = 100, s0 = initialize_rubiks_cube(10))
  ╠═╡ =#

# ╔═╡ 592a404d-f788-4781-b614-69f90dede7d5
# ╠═╡ disabled = true
#=╠═╡
const test_mc_nonlinear_output = run_mc_rubiks_nonlinear_test(() -> rand(1:30), [32, 32, 32, 32]; num_episodes = 1_000_000, max_steps = 20, α =1f-4, ϵ = 0.01f0, c = 1_000f0)
  ╠═╡ =#

# ╔═╡ bbe646c0-0de0-4bbf-9c75-ac9cc9491492
#=╠═╡
plot(smooth_error(test_mc_nonlinear_output.reward_history, 1_000)[LinRange(1, length(test_mc_nonlinear_output.reward_history)-1000, 1000) |> v -> round.(Int64, v)])
  ╠═╡ =#

# ╔═╡ 69f09d4e-743c-45c6-bd96-9b60ba724992
#=╠═╡
const mc_nonlinear_episode = show_rubiks_episode(test_mc_nonlinear_output.π_greedy; max_steps = 40, s0 = mc_nonlinear_s0)
  ╠═╡ =#

# ╔═╡ 8d8c360f-66b7-4d50-bff8-803e77ef688e
md"""
## Dynamic Programming Gradient Control
"""

# ╔═╡ 7cd30b68-1796-4dd0-bf13-dfb71610637a
md"""
### Updating MDP for TD Methods

In order to expose an agent to a variety of states, it is important to reset states periodically to those close to a solution.  The original formulation of the Rubik's cube problem does not allow any episode termination short of a solved cube.  If we add to the state representation the number of moves so far as well as the number of moves used to scramble the cube, then the step function can use that information to reset the cube after a limit has been reached.  For a given number of scramble moves, an episode for the optimal policy should never exceed that count.  Furthermore, even if the number of scramble moves is extremely high, the number of moves to solve should not exceed 20.  Therefore, we can alter the problem so that there is some probability distribution of scramble moves that are used to initialize a state and that value is saved in the state to allow for a reset if the total number of moves exceeds that value or 20, whichever is less.  With these changes, it is possible to get valuable information from partially completed episodes and use TD methods, including the one with the lowest variance, i.e. dynamic programming.
"""

# ╔═╡ 62650c81-17cd-4e2e-ab8d-a92c6b3eefb3
md"""
### Linear Method
"""

# ╔═╡ 97f64a35-b10d-4c3d-ac28-468110f9177f
function update_rubiks_feature!(v::AbstractVector{T}, s::@NamedTuple{cube::Vector{UInt8}, scramble_moves::Int64, move_count::Int64}) where T<:Real
	v .= zero(T)
	@inbounds @simd for i in eachindex(s.cube)
		j = (i-1)*48
		v[j + s.cube[i]] = one(T)
	end
	return v
end

# ╔═╡ 09538cca-7da3-4d48-9541-90f00acce794
function make_rubiks_feature(cube::AbstractVector{I}) where I <: Integer
	v = zeros(Float32, 48*48)
	update_rubiks_feature!(v, cube)
	return v
end

# ╔═╡ 2040ffd4-f977-4c30-83a0-875b500099aa
const solved_cube_feature = make_rubiks_feature(solved_cube_indices)

# ╔═╡ 0403bc9e-05b0-4662-a3ec-f78f7205ab1f


# ╔═╡ 2f2a9538-8046-413f-945e-ea6fdc9809ed
#=╠═╡
show_rubiks_policy_eval(s -> test_dp_output.π_greedy((cube=s, move_count = 0, scramble_moves = 30)), 1000)
  ╠═╡ =#

# ╔═╡ 3f289f24-9992-45f0-81be-683e7e152199
#=╠═╡
show_rubiks_episode(s -> test_dp_output.π_greedy((cube=s, move_count = 0, scramble_moves = 30)); max_steps = 100, s0 = initialize_rubiks_cube(20))
  ╠═╡ =#

# ╔═╡ 7bc8ea32-92c8-4a80-a64e-44e6b2af408b
# ╠═╡ disabled = true
#=╠═╡
const test_dp_output2 = run_dp_rubiks_linear_test2(;num_steps = 1_000_000, α = 1f-6)
  ╠═╡ =#

# ╔═╡ 9899c955-e5f5-413a-ba9e-cd57dda60eb4
#=╠═╡
show_rubiks_policy_eval(test_dp_output2.π_greedy, 1000)
  ╠═╡ =#

# ╔═╡ 77bb49f2-1c14-48ca-8d10-34000e6b4541
#=╠═╡
plot([1:1_000 |> Map(_ -> test_dp_output2.value_function(initialize_rubiks_cube(i))) |> mean for i in 1:30])
  ╠═╡ =#

# ╔═╡ 7f12e90c-1dd5-49e4-91a1-786e3165c769
md"""
### Non-linear Method
"""

# ╔═╡ d20f376b-7973-495f-811e-75a7fff0cd9c
function update_rubiks_feature2!(v::AbstractVector{T}, s::@NamedTuple{cube::Vector{UInt8}, scramble_moves::Int64, move_count::Int64}) where T<:Real
	v .= zero(T)
	@inbounds @simd for i in eachindex(s.cube)
		j = (i-1)*6
		v[j + solved_cube_values[s.cube[i]]] = one(T)
	end
	return v
end

# ╔═╡ 5f6ed356-71c3-44a1-ae1f-ac8ed55a1c2b
function make_rubiks_feature2(cube::AbstractVector{I}) where I <: Integer
	v = zeros(Float32, 48*6)
	update_rubiks_feature2!(v, cube)
	return v
end

# ╔═╡ 80069af6-31fd-4042-8690-a388c213fe28
const solved_cube_feature2 = make_rubiks_feature2(solved_cube_indices)

# ╔═╡ c8c3a104-0183-4e1b-9d9d-b2e2c6248f4c
# ╠═╡ disabled = true
#=╠═╡
const test_dp_nonlinear_output = run_dp_rubiks_nonlinear_test(1, 20; layers = [256, 256, 256, 256, 256, 256, 256], num_steps = 10_000_000, α = 1f-4, c = 1000f0, dropout = 0.01f0, ϵ = 0.01f0)
  ╠═╡ =#

# ╔═╡ df483fd1-737b-44d2-8c25-a49d756fb05c
#=╠═╡
plot_rewards(test_dp_nonlinear_output.reward_history, 10000, 1000)
  ╠═╡ =#

# ╔═╡ 73ac41c1-0c82-4772-bf05-351bda74e444
#this might be working but to train faster need to add batch version of semi_gradient_dp

# ╔═╡ 4955995a-5051-4d5e-a336-008834ba8ca2
#=╠═╡
show_rubiks_policy_eval(s -> test_dp_nonlinear_output.π_greedy((cube=s, move_count = 0, scramble_moves = 30)), 100)
  ╠═╡ =#

# ╔═╡ f6adca6d-8cc2-4fd9-a446-3564b68c9499
#=╠═╡
show_rubiks_episode(s -> rand() < 0.1 ? rand(1:12) : test_dp_nonlinear_output.π_greedy((cube=s, move_count = 0, scramble_moves = 30)); max_steps = 3000, s0 = initialize_rubiks_cube(30))
  ╠═╡ =#

# ╔═╡ 3d5eb8a6-29bf-49d5-acec-f05897ca1416
# ╠═╡ disabled = true
#=╠═╡
const test_dp_nonlinear_output2 = run_dp_rubiks_nonlinear_test2(;layers = fill(16, 10), num_steps = 1_000_000, α = 1f-4, c = 1000f0, dropout = 0.01f0, ϵ = 0.01f0)
  ╠═╡ =#

# ╔═╡ 1c69a2e6-4396-4460-8f52-90c626bebc24
#=╠═╡
show_rubiks_policy_eval(test_dp_nonlinear_output2.π_greedy, 10)
  ╠═╡ =#

# ╔═╡ d127796c-9dbe-4ca6-a11d-0197875d095a
#=╠═╡
plot([1:1_000 |> Map(_ -> test_dp_nonlinear_output2.value_function(initialize_rubiks_cube(i))) |> mean for i in 1:30])
  ╠═╡ =#

# ╔═╡ c9bf5811-6a61-45f4-9070-6f2b7277611f
md"""
# MCTS Planning Control Method 
"""

# ╔═╡ 4c53d04e-df52-4960-ba46-ed3bff1a624b
function π_dist_rubiks_uniform!(prior, s)
	prior .= 1f0/12
	return 1
end

# ╔═╡ fca1535c-20f3-450e-95b3-0130e8a49b48
md"""
# Tabular Version of More Limited Problem Space
"""

# ╔═╡ d3feb1af-b300-4a36-844b-3ad6dfe9a758
# ╠═╡ disabled = true
#=╠═╡
const candidate_cube = initialize_rubiks_cube(30)
  ╠═╡ =#

# ╔═╡ c42847a7-7324-4c30-8275-0638fd289661
# ╠═╡ disabled = true
#=╠═╡
(state_list, score_list) = exhaustive_search(candidate_cube, 100; search_moves = 4, score_cube = tabular_similarity_score)
  ╠═╡ =#

# ╔═╡ b88b019b-dd73-464d-a720-0394ad3c712c
#=╠═╡
plot(score_list)
  ╠═╡ =#

# ╔═╡ d3383318-c11e-4616-a2de-26f27023043d
#=╠═╡
[render_cube(c) for c in state_list]
  ╠═╡ =#

# ╔═╡ 82e56315-c8f1-4c01-bbc6-dc1c5588a60a
#=╠═╡
in(last(state_list), rubiks_tabular_mdp.states)
  ╠═╡ =#

# ╔═╡ c6fd1a65-014d-4493-928d-c15d598d8415
#=╠═╡
in(first(state_list), rubiks_tabular_mdp.states)
  ╠═╡ =#

# ╔═╡ 914a8139-d934-48aa-ac4b-ae0a3f7363f7
# ╠═╡ disabled = true
#=╠═╡
function cube_similarity(cube1::Vector{UInt8}, cube2::Vector{UInt8})
	s = 0f0
	@inbounds @simd for i in eachindex(cube1)
		s += Float32(cube1[i] == cube2[i])
	end
	return s
end
  ╠═╡ =#

# ╔═╡ 0570b1ce-b71b-4ccf-88ee-a75e90c426d1
# ╠═╡ disabled = true
#=╠═╡
function tabular_similarity_score(cube::Vector{UInt8})
	five_move_list |> Map(s -> cube_similarity(cube, s)) |> foldxt(max)
end
  ╠═╡ =#

# ╔═╡ 5affaa6c-eef7-4682-9bce-47f5bfd7d5c0
# ╠═╡ disabled = true
#=╠═╡
const six_move_list = build_nmove_list(6)
  ╠═╡ =#

# ╔═╡ 645aad71-f90b-4d04-a12e-d267466a5665
# ╠═╡ disabled = true
#=╠═╡
const five_move_list = build_nmove_list(5)
  ╠═╡ =#

# ╔═╡ 20a5c2f2-b031-4d7f-91f5-d775aaf4f593
# ╠═╡ disabled = true
#=╠═╡
const four_move_list = build_nmove_list(4)
  ╠═╡ =#

# ╔═╡ 625dc7bb-8406-4165-893a-2b8d7a06d011
# ╠═╡ disabled = true
#=╠═╡
function score_cube2(cube::Vector{UInt8})
	s = 0f0
	@inbounds @simd for i in eachindex(cube)
		s += Float32(solved_cube_values[cube[i]] == solved_cube_values[i])
	end
	return s
end
  ╠═╡ =#

# ╔═╡ 04561ac3-a19d-4ece-86dc-1e5d024d8f84
# ╠═╡ disabled = true
#=╠═╡
const cube_edge_indices = Set(mapreduce(vcat, 1:6) do f
	[2, 4, 5, 7] .+ 8*(f-1)
end)
  ╠═╡ =#

# ╔═╡ a77ec593-5ee3-4bca-8064-326573a3edfe
# ╠═╡ disabled = true
#=╠═╡
function score_cube3(cube::Vector{UInt8}; edge_weight = 10f0)
	s = 0f0
	@inbounds @simd for i in eachindex(cube)
		edge = in(i, cube_edge_indices)
		weight = edge*edge_weight + !edge*1f0
		s += weight * Float32(cube[i] == solved_cube_indices[i])
	end
	return s
end
  ╠═╡ =#

# ╔═╡ 10e951ff-1b3b-46cd-9a3f-ba819804b65e
# ╠═╡ disabled = true
#=╠═╡
function score_cube4(cube::Vector{UInt8}; layer1_weight = 4f0, layer2_weight = 2f0)
	s = 0f0
	@inbounds @simd for i in eachindex(cube)
		layer1 = in(i, layer1_inds)
		layer2 = in(i, layer2_inds)
		weight = layer1*layer1_weight + layer2*layer2_weight + (!layer1*!layer2)*1f0
		s += weight * Float32(cube[i] == solved_cube_indices[i])
	end
	return s
end
  ╠═╡ =#

# ╔═╡ 46af888e-e57e-456e-af71-bdcf22dc7798
# ╠═╡ disabled = true
#=╠═╡
function max_score_search(s0::Vector{UInt8}, nmoves::Integer, score_cube::Function)
	initial_score = score_cube(s0)
	states = build_nmove_list(nmoves; s0 = s0)
	in(solved_cube_indices, states) && return (solved_cube_indices, score_cube(solved_cube_indices), initial_score)
	beststate = s0
	bestscore = initial_score
	for s in states
		score = score_cube(s)
		if score > bestscore
			bestscore = score
			beststate = s
		end
	end
	return (beststate, bestscore, initial_score)
end	
  ╠═╡ =#

# ╔═╡ 81cbe49d-364b-450e-8e2f-c496525e3aae
# ╠═╡ disabled = true
#=╠═╡
function exhaustive_search(s0::Vector{UInt8}, max_turns::Integer; search_moves::Integer = 5, score_cube = score_cube)
	state_list = [s0]
	score_list = [score_cube(s0)]
	s = copy(s0)
	for i in 1:max_turns
		(beststate, bestscore, initial_score) = max_score_search(s, search_moves, score_cube)
		push!(state_list, beststate)
		push!(score_list, bestscore)
		s == beststate && break
		s = beststate
		s == solved_cube_indices && break
	end
	return state_list, score_list
end
  ╠═╡ =#

# ╔═╡ 7464b10d-056b-47cc-8db8-44ace4b84d11
const nmove_lookup = Dict{Int64, Set{SVector{48, UInt8}}}()

# ╔═╡ 206a1146-4a01-412f-a17e-3d58cac83453
md"""
## Linear Approximation of Value Function
"""

# ╔═╡ 87c318f7-197a-4b51-b38f-137ffb52f3d7
function compute_score_averages(states::Vector{Vector{UInt8}}, state_values::Vector{Float32})
	unique_values = sort(unique(state_values))
	valuemap = Dict(v => i for (i, v) in enumerate(unique_values))
	value_sums = zeros(Float32, length(unique_values))
	value_counts = zeros(Float32, length(unique_values))
	@inbounds @simd for i in eachindex(states)
		v = state_values[i]
		s = score_cube(states[i])
		value_sums[valuemap[v]] += s
		value_counts[valuemap[v]] += 1f0
	end
	return unique_values, value_sums ./ value_counts
end	

# ╔═╡ 64c85a03-d88e-471e-8215-5b4ff51b5440
md"""
At the end of an MCTS attempt to improve score.  How often is the resulting state found in the state list for the 7 step MDP?
"""

# ╔═╡ 0624c4dc-d6e4-4a31-b13d-3618d124f857
# ╠═╡ disabled = true
#=╠═╡
compare_mcts_endpoint(10_000, 10, 1000f0, 10000, initialize_rubiks_cube(30))
  ╠═╡ =#

# ╔═╡ 7ff44be5-a4df-4862-b0e5-302588c127c4
md"""
# Supervised Learning on Backtracking Database
"""

# ╔═╡ cd398ac4-82f3-4c06-afd5-019864064197
md"""
# Other To DO
"""

# ╔═╡ 434df85d-fbdc-41e7-98f7-6948e4aacea2
function fcann_v̂(s, params)
	v = zeros(Float32, 6*8*6)
	update_rubiks_feature!(v, s)
	input = reshape(v, 1, length(v))
	predict(params..., input, 1)
end

# ╔═╡ ffbd0f1d-306a-4b3a-84a4-53c7d3f0fb27
# ╠═╡ disabled = true
#=╠═╡
# const rubiks_monte_carlo_control_test = NonTabularRL.run_linear_gradient_monte_carlo_control(rubiks_cube_mdp, 0.99f0, 10_000_000, Float32.(solved_cube_bits), update_rubiks_feature!; α = 1f-4, ϵ = 0.01f0, suppress_warning=true, max_steps = 7)

const rubiks_monte_carlo_control_test = NonTabularRL.run_fcann_gradient_monte_carlo_control(rubiks_cube_mdp, 0.99f0, 10_000_000, Float32.(solved_cube_bits), update_rubiks_feature!, [256, 256, 256]; α = 1f-5, c = 100f0, dropout = 0.05f0, ϵ = 0.01f0, suppress_warning=true, max_steps = 7)
  ╠═╡ =#

# ╔═╡ c2a1a4a9-c652-400f-ad3a-5a05d4e10073
#=╠═╡
function smooth_error(error_history, n)
	l = length(error_history)
	[mean(error_history[i-n:i]) for i in n+1:l]
end
  ╠═╡ =#

# ╔═╡ 6148ec6a-5d38-40d1-84ba-8b5253d5fdaa
#=╠═╡
plot_rewards(rewards::Vector, nsmooth::Integer, npoints::Integer) = plot(smooth_error(rewards, nsmooth)[round.(Int64, LinRange(1, length(rewards) - nsmooth, npoints))])
  ╠═╡ =#

# ╔═╡ df59ec2c-1032-4c44-962c-1014fb4760f4
12^4

# ╔═╡ 3ce3c964-4272-48f0-ae40-8f77f333f4ca
depth = 3

# ╔═╡ 7d3856b9-38e9-40dc-b9da-44ae94e94522
layer_size = 1024

# ╔═╡ 0acf8afd-166a-4057-8bdb-e7d95e761145
layers = fill(layer_size, depth)

# ╔═╡ f674e316-f3e1-44bd-98eb-f5812c045c77
#another idea is to do MCTS but for the prior distribution use the inner product with the solved cube which is also just a measure of how many facets are in the correct position

# ╔═╡ 635a7314-c6df-4ddd-863c-833166069de6
build_input_output_branches(max_moves::Integer) = build_input_output_branches(solved_cube_values, Vector{Int64}(); max_moves = max_moves)

# ╔═╡ ad7559e2-c938-4396-9fab-09b9ecfcbfc4
# ╠═╡ disabled = true
#=╠═╡
(θ, β, bestcost, record, timerecord) = FCANN.ADAMAXTrainNNGPU(((rubiks_input, rubiks_output),), batchsize, params..., epochs, input_size, layers, λ, c; dropout = dropout, alpha = α, costFunc = "sqErr", use_μP = true, lrschedule = LinRange(α, 1f-20, epochs)
)  
  ╠═╡ =#

# ╔═╡ 619ff1aa-61ba-495d-8869-047cbf7ca0a1
#=╠═╡
plot(record[2:end])
  ╠═╡ =#

# ╔═╡ 7999388a-eff6-4cca-8f26-373317654f25
#=╠═╡
fcann_v̂(initialize_rubiks_cube(;num_actions = 3), (θ, β))
  ╠═╡ =#

# ╔═╡ 6f227e33-370c-4f75-b10e-6d48d73e401e
#=╠═╡
test_cube_policy(initialize_rubiks_cube(;num_actions = 3), π_test2; maxsteps = 10)
  ╠═╡ =#

# ╔═╡ c8cc0023-6f5e-4696-8303-3c5e9c52d0dc
#=╠═╡
function π_test2(s::Matrix{UInt8})
	i_a = 1
	best_value = typemin(Float32)
	for i in eachindex(rubiks_moves)
		(r, s′) = rubiks_transition(s, i)
		v = fcann_v̂(s′, (θ, β))[1]
		if v > best_value
			best_value = v
			i_a = i
		end
	end
	return i_a
end
  ╠═╡ =#

# ╔═╡ 5d7746fd-296d-4866-9c7b-9019f1f3e02a


# ╔═╡ ee3461a8-26a8-4e80-9e4c-42fa3d8787fd
const square_vectors = make_onehot_vector.(square_values)

# ╔═╡ 474b56e9-c8ee-4db3-80c0-2408ae5008e5
const onehot_lookup = Dict(zip(square_values, square_vectors))

# ╔═╡ f022bd7b-9a34-4940-9a52-13b8ad2eefdc
# ╠═╡ disabled = true
#=╠═╡
const value_lookup = Dict(zip(square_vectors, square_values))
  ╠═╡ =#

# ╔═╡ 0d80de04-ac65-4cce-9db2-5f3053079a1b
value2onehot(v::Integer) = onehot_lookup[UInt8(v)]

# ╔═╡ 639b2d17-59c9-4605-a394-8fce6dc5449b
const solved_cube_bits = solved_cube_values |> Map(value2onehot) |> foldxl(vcat)

# ╔═╡ 7bb3fe7c-a8a5-41b0-b35c-2ca92a4d2eb0
function make_value_data(states::AbstractVector)
	l = length(states)
	v = copy(solved_cube_bits)
	X = zeros(Float32, l, 48*48)
	for i in 1:l
		s = states[i]
		update_rubiks_feature!(v, s)
		X[i, :] .= v
	end
	return X
end

# ╔═╡ 1af6a2f7-ea00-4d96-8045-2ce280b5a833
const face_names = ["Front", "Top", "Right", "Back", "Left", "Bottom"]

# ╔═╡ 0b826bd0-8a83-47d1-934b-e9ddc50c91a9
#=╠═╡
@bind root_move_eg PlutoUI.combine() do Child
	md"""
	The display below shows the impact of applying the selected move to a solved cube.
	
	Select Face: $(Child(:face, Select([a[1] => a[2] for a in zip(face_order, face_names)])))
	Select Rotation: $(Child(:rot, Select([Clockwise => "Clockwise", CounterClockwise => "Counter Clockwise"])))
	"""
end
  ╠═╡ =#

# ╔═╡ f8d68c24-71f0-4850-804d-67e5634ed60d
const face2value = Dict(zip(face_order, square_values))

# ╔═╡ eb1f5b65-e804-4578-9123-ea8d65e75397
begin
	function get_rotation_indices(::RubiksMove{F, Clockwise}) where F<:Face
		rotation_mapping = clockwise_rotation_mapping[F]
		
		fnum = face2value[F]
		cube = reshape(1:48, 8, 6)
		cube′ = copy(cube)

		#rotate face colors
		@inbounds @simd for i in 1:8
			cube′[i, fnum] = cube[clockwise_perm[i], fnum]
		end

		#rotate other colors
		for i in 2:4
			(origin_face, origin_inds) = rotation_mapping[i-1]
			(destination_face, destination_inds) = rotation_mapping[i]
			@inbounds @simd for j in 1:3
				cube′[destination_inds[j], destination_face] = cube[origin_inds[j], origin_face]
			end
		end
		(origin_face, origin_inds) = rotation_mapping[4]
		(destination_face, destination_inds) = rotation_mapping[1]
		@inbounds @simd for j in 1:3
			cube′[destination_inds[j], destination_face] = cube[origin_inds[j], origin_face]
		end
		return cube′[:]
	end

	function get_rotation_indices(::RubiksMove{F, CounterClockwise}) where F<:Face
		rotation_mapping = clockwise_rotation_mapping[F]
		
		fnum = face2value[F]
		cube = reshape(1:48, 8, 6)
		cube′ = copy(cube)

		#rotate face colors
		@inbounds @simd for i in 1:8
			cube′[clockwise_perm[i], fnum] = cube[i, fnum]
		end

	
		#rotate other colors
		for i in 1:3
			(origin_face, origin_inds) = rotation_mapping[i+1]
			(destination_face, destination_inds) = rotation_mapping[i]
			@inbounds @simd for j in 1:3
				cube′[destination_inds[j], destination_face] = cube[origin_inds[j], origin_face]
			end
		end
		(origin_face, origin_inds) = rotation_mapping[1]
		(destination_face, destination_inds) = rotation_mapping[4]
		@inbounds @simd for j in 1:3
			cube′[destination_inds[j], destination_face] = cube[origin_inds[j], origin_face]
		end

		
		return cube′[:]
	end
end

# ╔═╡ 95854163-4da3-4d67-b254-1eab96c9fcc6
const rotation_lookup = [get_rotation_indices(m) for m in rubiks_moves]

# ╔═╡ 22879750-f3c1-4053-994c-fb1262107be3
begin
	function rotate_cube!(cube′::Vector{UInt8}, cube::Vector{UInt8}, i_a::Integer)
		indices = rotation_lookup[i_a]
		@inbounds @simd for i in eachindex(cube)
			cube′[i] = cube[indices[i]]
		end
		return cube′
	end
	rotate_cube!(cube′, cube, m::RubiksMove) = rotate_cube!(cube′, cube, rubiks_move_index[m])
	rotate_cube(cube::Vector{UInt8}, x; cube′ = copy(cube)) = rotate_cube!(cube′, cube, x)
end

# ╔═╡ 361b9a94-ab5f-474d-8893-9987be4f0c5d
function rubiks_move(cube::Vector{UInt8}, i_a::Integer; kwargs...)
	score1 = score_cube(cube)
	cube′ = rotate_cube(cube, i_a; kwargs...)
	score2 = score_cube(cube′)
	reward = score2 - score1
	(reward, cube′)
end

# ╔═╡ 8fe5898c-2c50-4a36-a49d-37fec4eef1a3
function rubiks_dist_move(cube::Vector{UInt8}, i_a::Integer; kwargs...)
	(r, s′) = rubiks_move(cube, i_a; kwargs...)
	return ([r], [s′], [1f0])
end

# ╔═╡ 7982b2b3-3cad-4b2a-ba9e-c9b6ba4ef728
const rubiks_transition_distribution = StateMDPTransitionDistribution(rubiks_dist_move, solved_cube_indices)

# ╔═╡ 44499455-ab75-4a65-9fec-1d544afb8a33
const rubiks_transition = StateMDPTransitionSampler(rubiks_move, solved_cube_indices)

# ╔═╡ b68a651b-f16d-4662-83ee-2120daf73512
function rubik_episode!((states, actions, rewards)::Tuple{Vector{Matrix{UInt8}}, Vector{Int64}, Vector{T}}; s0::Matrix{UInt8} = solved_cube_values, i_a0 = rand(1:12), max_steps = 10) where {T<:Real}
	step = 1
	l = length(states)
	sterm = solved_cube_values
	a = rubiks_moves[i_a0]
	a_rev = reverse_move(a)
	i_a = rubiks_move_index[a_rev]
	(r, s) = rubiks_move(sterm, i_a0)
	for i in 1:min(l, max_steps)
		states[i] = s
		actions[i] = i_a
		rewards[i] = r
		s′ = solved_cube_values
		while s′ == solved_cube_values
			i_a = rand(1:12)
			(r, s′) = rubiks_move(s, i_a)
		end
		a = rubiks_moves[i_a]
		a_rev = reverse_move(a)
		i_a = rubiks_move_index[a_rev]
		s = s′
	end

	for i in min(l, max_steps)+1:max_steps
		push!(states, s)
		push!(rewards, r)
		push!(actions, i_a)
		s′ = solved_cube_values
		while s′ == solved_cube_values
			i_a = rand(1:12)
			(r, s′) = rubiks_move(s, i_a)
		end
		a = rubiks_moves[i_a]
		a_rev = reverse_move(a)
		i_a = rubiks_move_index[a_rev]
		(r, s′) = rubiks_move(s, i_a)
		s = s′
	end

	for (i, j) in enumerate(max_steps:-1:ceil(Int64, max_steps / 2))
		s1 = states[i]
		s2 = states[j]
		states[i] = s2
		states[j] = s1
		a1 = actions[i]
		a2 = actions[j]
		actions[i] = a2
		actions[j] = a1
		
		r1 = rewards[i] 
		r2 = rewards[j]
		rewards[i] = r2
		rewards[j] = r1
	end
	return (states, actions, rewards, sterm, max_steps)
end

# ╔═╡ daf5d43c-093f-4dd1-9a5a-a68afd40aa54
function build_rubiks_dataset(num_episodes; steps = 10)
	states = Vector{Matrix{UInt8}}()
	actions = Vector{Int64}()
	rewards = Vector{Float32}()
	input = zeros(Float32, num_episodes*(steps), 8*6*6)
	output = zeros(Float32, num_episodes*(steps), 1)
	v = zeros(Float32, 8*6*6)
	row = 1
	x = (steps - 1)/2f0
	for ep in 1:num_episodes
		rubik_episode!((states, actions, rewards); max_steps = steps)
		for i in 1:length(states) 
			s = states[i]
			update_rubiks_feature!(v, s)
			input[row, :] .= v
			output[row, 1] = (i - 1)/x - 1
			row += 1
		end
	end
	return (input, output)
end

# ╔═╡ 3c054a02-1c38-4df6-a559-d6f345d6d795
(rubiks_input, rubiks_output) = build_rubiks_dataset(50_000; steps = 4)

# ╔═╡ 65a1cb65-6f9e-43fa-b879-fa5ed448da3c
#=╠═╡
mean(rubiks_output .^2)
  ╠═╡ =#

# ╔═╡ 87ad6b4f-5322-459a-9aac-e2996f45a67c
begin
	batchsize = 1024
	input_size = size(rubiks_input, 2)
	λ = 0f0
	c = Inf
	dropout = 0.0f0
	α = 1f-1
	epochs = 100
end

# ╔═╡ 14768320-8400-43ca-ad15-5adf9c78fbbd
const params = FCANN.initializeparams_saxe(input_size, layers, 1, 1; use_μP=true)

# ╔═╡ 790aea25-52d1-41c0-8b80-5b987c294c54
function test_cube_policy(s0::Matrix{UInt8}, π::Function; maxsteps = 10_000)
	s = s0
	i_a = π(s)
	step = 1
	while (step < maxsteps) && (s != solved_cube_values)
		(r, s) = rubiks_move(s, i_a)
		i_a = π(s)
		step += 1
	end
	return step
end

# ╔═╡ b31848b1-3a25-4cd5-8ca2-748f35a010e2
function rubiks_dist_move2(cube::Vector{UInt8}, i_a::Integer; kwargs...)
	cube′ = rotate_cube(cube, i_a; kwargs...)
	return ([-1f0], [cube′], [1f0])
end

# ╔═╡ 4036aeb4-65fd-449a-a259-1683de4f9c30
const rubiks_transition_distribution2 = StateMDPTransitionDistribution(rubiks_dist_move2, solved_cube_indices)

# ╔═╡ 8c396914-675b-404f-9f0c-90e880c55658
function initialize_rubiks_cube(num_actions)
	cube = copy(solved_cube_indices)
	cube′ = copy(cube)
	for _ in 1:num_actions
		rotate_cube(cube, rand(eachindex(rubiks_moves)); cube′ = cube′)
		cube .= cube′
	end
	return cube
end

# ╔═╡ 1fbcf1f1-722a-4014-b8f6-11c2a5f12548
function make_rubiks_mdp(make_init_actions::Function, transition::AbstractStateTransition)
	StateMDP(rubiks_moves, transition, () -> initialize_rubiks_cube(make_init_actions()), s -> isequal(s, solved_cube_indices))
end

# ╔═╡ b28d50ad-ca0a-4743-bace-be46c0821ac2
const rubiks_cube_mdp = make_rubiks_mdp(() -> 0, rubiks_transition_distribution)

# ╔═╡ fe20a8bd-c608-4ac7-9bb0-3251bcf7d85c
runepisode(rubiks_cube_mdp; max_steps = 5)

# ╔═╡ f3ee2969-cafa-4e24-8295-0e5a1035eb4a
runepisode(rubiks_cube_mdp; max_steps = 10)

# ╔═╡ 03ed5828-ae46-4a9a-abfa-156c7c0ed989
fcann_test = NonTabularRL.run_fcann_monte_carlo_policy_estimation(rubiks_cube_mdp, make_random_policy(rubiks_cube_mdp), 1f0, 100_000, [128, 128], make_rubiks_feature(base_cube), update_rubiks_feature!; setup_kwargs = (λ = 0f-10, c = 10f0, dropout = 0.0f0), α = 1f-6, max_steps = 20)

# ╔═╡ d795f457-f7a8-453f-a440-6a0d3f6fce09
#=╠═╡
plot(scatter(x = 1:1000:length(fcann_test.error_history), y = smooth_error(sqrt.(fcann_test.error_history), 100)[1:1000:end]))
  ╠═╡ =#

# ╔═╡ efb5897b-4f97-4ba3-8257-93c6c24eebec
function π_test(s::RubiksCube)
	i_a = 1
	best_value = typemin(Float32)
	for i in eachindex(rubiks_moves)
		(r, s′) = rubiks_transition(s, i)
		v = fcann_test.v̂(s′)
		if v > best_value
			best_value = v
			i_a = i
		end
	end
	return i_a
end

# ╔═╡ 027d5ff6-b250-427f-9a00-ec9562236bc3
function run_sarsa_rubiks_linear_test(select_num_actions::Function; γ = 0.99f0, max_steps = 10_000, kwargs...)
	mdp =  make_rubiks_mdp(select_num_actions, rubiks_transition)
	NonTabularRL.run_linear_semi_gradient_sarsa(mdp, γ, typemax(Int64), max_steps, copy(solved_cube_feature), update_rubiks_feature!; kwargs...)
end

# ╔═╡ 35ae742a-db14-4868-bb2d-d59cd9300de4
const test_sarsa_output = run_sarsa_rubiks_linear_test(() -> 50, max_steps = 10_000, α = 0.001f0)

# ╔═╡ f55b65e0-cb95-4403-a632-d95c50c893a9
function run_dp_rubiks_linear_test2(;γ = 0.999f0, num_steps = 10_000, kwargs...)
	mdp =  make_rubiks_mdp(() -> 30, rubiks_transition_distribution)
	NonTabularRL.run_linear_semi_gradient_dp(mdp, γ, typemax(Int64), num_steps, copy(solved_cube_feature), update_rubiks_feature!; suppress_warning=true, kwargs...)
end

# ╔═╡ dd2c8eec-1e56-4315-8e14-7f031b44bcf0
function run_dp_rubiks_nonlinear_test2(;layers = [2, 2], num_steps = 100, kwargs...)
	mdp =  make_rubiks_mdp(() -> 30, rubiks_transition_distribution)
	NonTabularRL.run_fcann_semi_gradient_dp(mdp, 1f0, typemax(Int64), num_steps, copy(solved_cube_feature), update_rubiks_feature!, layers; suppress_warning=true, kwargs...)
end

# ╔═╡ 18fa27e3-d076-4543-92cc-679ee3c32156
const rubiks_mdp = make_rubiks_mdp(() -> 5, rubiks_transition)

# ╔═╡ 09cf4ae7-d1fb-44d6-a617-0dbc7f64c2f2
const rubiks_dist_mdp = make_rubiks_mdp(() -> 5, rubiks_transition_distribution)

# ╔═╡ 3617c5a0-79e6-4fc0-814e-45bbdd373b5d
rubiks_mcts_policy(s; kwargs...) = TabularRL.monte_carlo_tree_search2(rubiks_dist_mdp, 0.99f0, s, π_dist_rubiks_uniform!, 1f0, 1, -48f0, 48f0; depth = 5, c = 0.5f0, kwargs...)[1](s)

# ╔═╡ 89f0f0ec-f659-4752-93fe-d90c0ac248e4
function run_sarsa_rubiks_nonlinear_test(select_num_actions::Function, layers::Vector{Int64}; γ = 0.99f0, max_steps = 10_000, kwargs...)
	mdp =  make_rubiks_mdp(select_num_actions, rubiks_transition)
	run_fcann_semi_gradient_sarsa(mdp, γ, typemax(Int64), max_steps, copy(solved_cube_feature), update_rubiks_feature!, layers; kwargs...)
end

# ╔═╡ dddcd1ee-4bc7-4791-ba0f-c8ee77e12246
test_sarsa_output.value_function(initialize_rubiks_cube(100))

# ╔═╡ 92b86eb0-e506-4104-8bc0-3f2550398455
#=╠═╡
function expected_maximum_reward(select_num_actions::Function; num_samples = 1000)
	1:num_samples |> Map(i -> initialize_rubiks_cube(select_num_actions()) |> maximum_reward) |> mean
end
  ╠═╡ =#

# ╔═╡ bee4261d-76ce-440f-8285-4b0816726f35
#=╠═╡
function run_mc_rubiks_linear_test(select_num_actions::Function; γ = 0.999f0, num_episodes = 100, kwargs...)
	mdp =  make_rubiks_mdp(select_num_actions, rubiks_transition_distribution)
	output = NonTabularRL.run_linear_gradient_monte_carlo_control(mdp, γ, num_episodes, copy(solved_cube_feature), update_rubiks_feature!; suppress_warning=true, kwargs...)
	emr = expected_maximum_reward(select_num_actions)
	(;output..., maximum_reward = emr)
end
  ╠═╡ =#

# ╔═╡ 043fc456-0a9f-4023-aea9-d2775640dcd8
#=╠═╡
function run_mc_rubiks_nonlinear_test(select_num_actions::Function, layers::Vector{Int64}; γ = 0.999f0, num_episodes = 100, kwargs...)
	mdp =  make_rubiks_mdp(select_num_actions, rubiks_transition_distribution)
	output = NonTabularRL.run_fcann_gradient_monte_carlo_control(mdp, γ, num_episodes, copy(solved_cube_feature2), update_rubiks_feature2!, layers; suppress_warning=true, kwargs...)
	emr = expected_maximum_reward(select_num_actions)
	(;output..., maximum_reward = emr)
end
  ╠═╡ =#

# ╔═╡ ddecab58-0965-4718-b5b9-3eff15c9e9dd
const mc_nonlinear_s0 = initialize_rubiks_cube(30)

# ╔═╡ 21b2a4ec-8af9-4884-8439-f6868ac12804
#=╠═╡
function rubiks_mean_reward(π::Function, nsamples::Integer, num_scramble::Integer)
	1:nsamples |> Map() do i
		s0 = initialize_rubiks_cube(num_scramble)
		while s0 == solved_cube_indices
			s0 = initialize_rubiks_cube(num_scramble)
		end
		ep = runepisode(rubiks_cube_mdp; π = π, s0 = s0, max_steps = num_scramble)
		sterm = ep[4]
		sterm == solved_cube_indices
	end|> mean
end
  ╠═╡ =#

# ╔═╡ b492634c-5c93-4d8f-80df-7c77ad04e14e
#=╠═╡
function eval_rubiks_policy(π::Function, nsamples::Integer)
	solve_rate = zeros(Float32, 20)
	for i in 1:20
		solve_rate[i] = rubiks_mean_reward(π, nsamples, i)
	end
	return solve_rate
end
  ╠═╡ =#

# ╔═╡ 0b4860d3-8153-49a7-be54-16ccc589ec9b
#=╠═╡
function show_rubiks_policy_eval(π::Function, nsamples::Integer)
	solve_rate = eval_rubiks_policy(π, nsamples)
	tr = bar(x = 1:20, y = solve_rate)
	plot(tr, Layout(xaxis_title = "Scramble Moves", yaxis_title = "Solve Rate"))
end
  ╠═╡ =#

# ╔═╡ cfab1b66-884a-4db6-bdae-6df295163838
const mcts_s0 = initialize_rubiks_cube(10)

# ╔═╡ 2e177050-ee05-4de9-9fa8-4e23199fc669
monte_carlo_tree_search(rubiks_mdp, 0.99f0, (mdp, s, γ) -> 0f0, mcts_s0; depth = 5, c = 1f0, nsims = 1_000)[3][mcts_s0]

# ╔═╡ 30af4178-b39b-4389-a664-6c9e457f8ca1
const mcts2_results = TabularRL.monte_carlo_tree_search2(rubiks_dist_mdp, 0.99f0, mcts_s0, π_dist_rubiks_uniform!, 100f0, 12, -48f0, 48f0; depth = 3, c = 0.5f0, nsims = 100)

# ╔═╡ be7b4bb5-0deb-4c96-ad0c-90e785a3ff28
mcts2_results[2](mcts_s0)

# ╔═╡ 1bf2d626-12c4-4065-a63d-c007f6ddd15b
mcts2_results[1](mcts_s0)

# ╔═╡ 11d96bf0-c09d-411d-a2fa-1e172ab1b7aa
mcts2_results[3][mcts_s0]

# ╔═╡ acbbc9ff-f26f-49de-9be4-2de2faefc3c8
fcann_test.v̂(initialize_rubiks_cube(;num_actions = 8))

# ╔═╡ 613191fe-307e-4bf1-be1a-74df1ed11c95
test_cube_policy(initialize_rubiks_cube(;num_actions = 20), π_test; maxsteps = 1000)

# ╔═╡ be2e7ca8-ac9f-439b-aeb4-cf5f12aae716
#=╠═╡
mean(test_cube_policy(initialize_rubiks_cube(;num_actions = 20), π_test; maxsteps = 100) for _ in 1:1_000)
  ╠═╡ =#

# ╔═╡ 662729cc-54fa-449c-a90b-4c32196aef20
function plot_feature_changes(num_moves)
	z = zeros(Float32, num_moves+1, 48*48)
	z[1, :] .= solved_cube_feature
	cube = copy(solved_cube_indices)
	cube′ = copy(solved_cube_indices)
	v = zeros(Float32, 48*48)
	for i in 2:num_moves+1
		rotate_cube!(cube′, cube, rand(1:12))
		update_rubiks_feature!(v, cube′)
		view(z, i, :) .= v
		cube .= cube′
	end
	return (x = 1:48*48, y = 1:num_moves+1, z = z)
end

# ╔═╡ 68150249-6db1-4dca-b998-9521a6d7d98d
#=╠═╡
heatmap(;plot_feature_changes(50)..., colorscale = "Greys", showscale=false) |> tr -> plot(tr, Layout(xaxis_title = "vector index", yaxis_title = "rotations", title = "Feature Vector Changing Through Random Moves"))
  ╠═╡ =#

# ╔═╡ 013fe195-8a8a-4717-a95b-16bb86e8de29
#create a cube mdp which saves in the state the number of initial scramble moves as well as how many moves have been attempted since the initial scramble
function make_tdcube_mdp(min_moves::Integer, max_moves::Integer)
	initialize_state(nmoves::Integer) = (cube = initialize_rubiks_cube(nmoves), scramble_moves = nmoves, move_count = 0)
	function isterm(s)
		s.move_count > 50 && return true
		s.move_count > 2*s.scramble_moves && return true
		s.cube == solved_cube_indices
	end
	function step(s, i_a)
		isterm(s) && return ([0f0], [s], [1f0])
		# score1 = score_cube(s.cube)
		cube′ = rotate_cube(s.cube, i_a)
		s′ = (cube = cube′, scramble_moves = s.scramble_moves, move_count = s.move_count + 1)
		# score2 = score_cube(cube′)
		# r = score2 - score1
		r = Float32(s′.cube == solved_cube_indices)
		([r], [s′], [1f0])
	end
	ptf = StateMDPTransitionDistribution(step, initialize_state(1))
	
	StateMDP(rubiks_moves, ptf, () -> initialize_state(rand(min_moves:max_moves)), isterm)
end

# ╔═╡ fd053336-7b1f-4c79-afaa-f60b6fcc5938
const rubiks_mdp_td = make_tdcube_mdp(1, 5)

# ╔═╡ 4085110c-7b47-440c-b1bf-020b91168183
s0_test = rubiks_mdp_td.initialize_state()

# ╔═╡ 4edd1430-5779-471e-b12c-843f6abce802
runepisode(rubiks_mdp_td; s0 = s0_test)

# ╔═╡ d163d3cd-e8ca-44fb-81b2-66a2e1a92812
function run_dp_rubiks_linear_test(min_moves, max_moves; γ = 0.99f0, num_steps = 10_000, kwargs...)
	mdp =  make_tdcube_mdp(min_moves, max_moves)
	output = NonTabularRL.run_linear_semi_gradient_dp(mdp, γ, typemax(Int64), num_steps, copy(solved_cube_feature), update_rubiks_feature!; suppress_warning=true, kwargs...)
	# emr = expected_maximum_reward(() -> rand(min_moves:max_moves))
	(;output...) #, maximum_reward = emr)
end

# ╔═╡ a538ead8-188b-4dd6-a105-a0c7a5d7d64f
const test_dp_output = run_dp_rubiks_linear_test(1, 20; num_steps = 10_000_000, α = 1f-4)

# ╔═╡ 26fcb97c-9d79-4638-a100-8eb8c59f1ef8
test_dp_output.value_function(rubiks_mdp_td.initialize_state())

# ╔═╡ 7b0af922-aca3-4e3b-b521-a01c6cd6889c
#=╠═╡
plot_rewards(test_dp_output.reward_history, 1000, 1000)
  ╠═╡ =#

# ╔═╡ b86c7a49-5769-49b5-945d-e54a0a2a2c86
#=╠═╡
function run_dp_rubiks_nonlinear_test(min_moves, max_moves; γ = 0.99f0, layers = [2, 2], num_steps = 100, kwargs...)
	mdp =  make_tdcube_mdp(min_moves, max_moves)
	output = NonTabularRL.run_fcann_semi_gradient_dp(mdp, γ, typemax(Int64), num_steps, copy(solved_cube_feature), update_rubiks_feature!, layers; suppress_warning=true, kwargs...)
	emr = expected_maximum_reward(() -> rand(min_moves:max_moves))
	(;output..., maximum_reward = emr)
end
  ╠═╡ =#

# ╔═╡ fa2286f8-7b20-4f33-8f5a-bccacbe758e9
#builds a list of all unique cube states that are within n moves of the solved state.  Any state that is outside of this set can be considered terminal with a losing condition, need to test to see how large n can be before the problem is unweildy
function update_nmove_list!(statelist::Set{SVector{48, UInt8}}, s0::Vector{UInt8}, nmoves::Integer)
	push!(statelist, SVector{48}(s0))
	nmoves == 0 && return nothing
	for i_a in 1:12
		s′ = rotate_cube(s0, i_a)
		update_nmove_list!(statelist, s′, nmoves - 1)
	end
	return statelist
end

# ╔═╡ ef39bdd3-c085-441c-a8cd-6c1221ad4b21
function build_nmove_list(nmoves::Integer; s0 = copy(solved_cube_indices)) 
	haskey(nmove_lookup, nmoves) && return nmove_lookup[nmoves]
	# statelist = if haskey(nmove_lookup, nmoves - 1)
	# 	@info "Extending $(nmoves-1) move database by 1 step"
	# 	update_nmove_list!(deepcopy(nmove_lookup[nmoves-1]))
	# else
	# 	@info "Building $nmoves move database from scratch"
	statelist = update_nmove_list!(Set{SVector{48, UInt8}}(), s0, nmoves)
	# end
	nmove_lookup[nmoves] = deepcopy(statelist)
end

# ╔═╡ 725317cb-e761-49f3-9c03-a15bf9c34da9
function build_tabular_rubiks_mdp(nmoves::Integer)
	@info "Building list of cube states"
	statelist = collect(build_nmove_list(nmoves))
	@info "Done building list of $(length(statelist)) states for $nmoves moves"
	@info "Building state index map"
	state_index_map = Dict(s => i for (i, s) in enumerate(statelist))
	@info "Done building state index map"
	nstates = length(state_index_map)
	state_transition_map = zeros(Int64, 12, nstates)
	reward_transition_map = zeros(Float32, 12, nstates)
	s′ = copy(solved_cube_indices)
	s_vec = copy(solved_cube_indices)
	i_s_term = state_index_map[SVector{48}(solved_cube_indices)]
	@info "Building state and reward transition maps"
	for s in statelist
		i_s = state_index_map[s]
		if i_s == i_s_term
			state_transition_map[:, i_s] .= i_s
		else
			# score1 = score_cube(s)
			s_vec .= s
			for i_a in 1:12
				rotate_cube!(s′, s_vec, i_a)
				(r, i_s′) = haskey(state_index_map, SVector{48}(s′)) ? (-1f0, state_index_map[SVector{48}(s′)]) : (-nmoves - 2, i_s_term)
				state_transition_map[i_a, i_s] = i_s′
				reward_transition_map[i_a, i_s] = r
			end
		end
	end
	TabularMDP(statelist, rubiks_moves, TabularDeterministicTransition(state_transition_map, reward_transition_map), () -> state_index_map[initialize_rubiks_cube(nmoves)]; state_index = state_index_map)
end

# ╔═╡ 0d62e009-81be-43d3-add5-ce4ac591dbc7
const rubiks_tabular_mdp = build_tabular_rubiks_mdp(6)

# ╔═╡ ddf57ab2-2bf5-4914-99b2-c0b0e9562e40
in(mc_nonlinear_s0, rubiks_tabular_mdp.states)

# ╔═╡ 1938becf-e2cc-4c7b-b6ec-d1c3ff107ed9
const rubiks_value_iteration = value_iteration_v(rubiks_tabular_mdp, 1f0)

# ╔═╡ a487ccf0-c293-4093-b936-f92094e86fa7
const value_lookup = DataFrame(state = rubiks_tabular_mdp.states, value = rubiks_value_iteration.final_value) |> df -> groupby(df, :value)

# ╔═╡ a06c96f9-57c1-48e4-a725-08980892502e
function make_value_dataset(base_value, value_n::Integer)
	#idea here is to build a dataset where there is an equal representation of cubes for each score = number of turns until solved.  We want our approximation to be accurate with the distribution of states visited under the optimal policy which eventually will spend an equal amount of time in states at each step distance away from being solved
	base_data = value_lookup[(value=base_value,)]
	l = size(base_data, 1)

	minkey = minimum(a.value for a in keys(value_lookup))

	X = base_value:-1:minkey |> Map() do k
		df = value_lookup[(value = k,)]
		l′ = size(df, 1)
		if l′ ≥ value_n
			inds = shuffle(1:l′)[1:value_n]
		else
			mult = value_n / l′
			basemult = floor(mult)
			inds = reduce(vcat, [collect(1:l′) for _ in 1:basemult])
			remainder = value_n - length(inds)
			inds′ = shuffle(1:l′)[1:remainder]
			inds = vcat(inds, inds′)
		end
		make_value_data(df.state[inds])
	end |> foldxt(vcat)

	y = reduce(vcat, [fill(k, value_n, 1) for k in base_value:-1:minkey])

	l2 = size(X, 1)
	inds = shuffle(1:l2)
	return (X[inds, :], y[inds, :])
end	

# ╔═╡ 91e6576c-0142-409c-8749-194c66ba3c9d
const (X, y) = make_value_dataset(-1, 1_000)

# ╔═╡ abfc1d27-96a6-48dc-93d1-b2f5db4455ef
#=╠═╡
mean(abs.(y .- mean(y)))
  ╠═╡ =#

# ╔═╡ 1e6c3190-6b5f-4ada-a24f-98483658aa17
linear_training_output = fullTrain("test", X, y, 1, 128, Vector{Int64}(), 0f0, Inf, 0.1f0, 0.1f0, 1; costFunc = "absErr", writeFiles=false)

# ╔═╡ 116c751a-76e6-4139-9926-0e8f58d30fe0
nonlinear_training_output = fullTrain("test", X, y, 10, 64, [64, 64, 64], 0f0, Inf, 0.01f0, 0.1f0, 1; costFunc = "absErr", writeFiles=false, use_μP = true)

# ╔═╡ 8fc4500b-de4f-44f0-ac51-b2fb875a336a
md"""
Total nonlinear model parameters: $(FCANN.theta2Params(nonlinear_training_output[3], nonlinear_training_output[2]) |> length)
"""

# ╔═╡ 3d1ad253-1fe6-4e5e-9959-690d88a278e4
predict(nonlinear_training_output[2], nonlinear_training_output[3], reshape(make_rubiks_feature(initialize_rubiks_cube(8)), 1, 48*48))

# ╔═╡ d84d5e1c-bcfb-451c-9a85-11ec329c5d22
#=╠═╡
function test_values(v)
	states = value_lookup[(value=v,)].state
	X = make_value_data(states)
	output1 = predict(linear_training_output[2], linear_training_output[3], X)
	error1 = mean(abs.(output1 .- v))
	output2 = predict(nonlinear_training_output[2], nonlinear_training_output[3], X)
	error2 = mean(abs.(output2 .- v))
	(linear_results = (output1, error1), nonlinear_results = (output2, error2))
end
  ╠═╡ =#

# ╔═╡ 4d848bb9-2d35-4568-84db-edd229c07ec9
#=╠═╡
test_values(-3)
  ╠═╡ =#

# ╔═╡ 54c914bc-8822-4fd4-bd38-97fd1476558e
#=╠═╡
test_values(-4).nonlinear_results[1] |> extrema
  ╠═╡ =#

# ╔═╡ 17006174-caad-4043-b8d5-883aa0e10c80
onehot2value(v::BitVector) = value_lookup[v]

# ╔═╡ 8218ea8b-3ba6-45fb-ac36-89ba6cccf112
function bits2value(v::BitVector; output = zeros(UInt8, 8, 6))
	for (i, j) in enumerate(1:6:287)
		output[i] = onehot2value(v[j:j+5])
	end
	return output
end

# ╔═╡ 91a8ca59-c261-4e40-97e7-43b6fd2f87f3
branch_rubik_moves(s0_bits::BitVector) = branch_rubik_move(bits2value(s0_bits))

# ╔═╡ 27b38778-17fe-4aca-8cec-341e2333e536
bits2value(solved_cube_bits)

# ╔═╡ 338515c3-6818-4c23-9d98-178e74c5f148
score_averages = compute_score_averages(rubiks_tabular_mdp.states, rubiks_value_iteration.final_value)

# ╔═╡ 22e630e8-2f5b-40f6-9bbe-b0e6761b4e76
#=╠═╡
plot(scatter(x =score_averages[1], y = score_averages[2]), Layout(xaxis_title = "Turns Until Solution", yaxis_title = "Average Score"))
  ╠═╡ =#

# ╔═╡ 39c7bb0d-093c-4be7-a366-2adcbfe7a7a6
function value_iteration_π(s::Vector{UInt8})
	q_best = typemin(Float32)
	i_best = 1
	s′ = copy(s)
	for i_a in 1:12
		rotate_cube!(s′, s, i_a)
		q = haskey(rubiks_tabular_mdp.state_index, s′) ? rubiks_value_iteration.final_value[rubiks_tabular_mdp.state_index[s′]] : typemin(Float32)
		if q > q_best
			q_best = q
			i_best = i_a
		end
	end
	q_best == typemin(Float32) && return rand(1:12)
	return i_best
end

# ╔═╡ e74f5d40-99a4-414c-878f-0d429faddea2
function make_rubiks_batch(batchsize)
	input = zeros(Float32, batchsize, length(solved_cube_bits))
	output = zeros(Float32, batchsize, 1)
	value_sets = [Set{Vector{UInt8}}() for _ in 1:30]
	s = copy(solved_cube_indices)
	v = 0
	for i in 1:batchsize
		s′ = rotate_cube(s, rand(1:12))
		v′ = findfirst(x -> in(s′, x),  value_sets)
		if isnothing(v′)
			v += 1
			if v > length(value_sets)
				push!(value_sets, Set{Vector{UInt8}}())
			end
			push!(value_sets[v], s′)
		else
			v = v′
		end
		s = s′
	end
	return value_sets
end

# ╔═╡ 5a75cb17-56cb-4808-8c41-76fa671c50cf
make_rubiks_batch(128)

# ╔═╡ 1a7e54e3-8a2b-4680-88f3-643967af82a2
function branch_rubik_moves(s0::Matrix{UInt8}; v = BitVector(fill(false, 288)), output = sparse(Matrix(fill(false, 12, 288))))
	s = copy(s0)
	for i in eachindex(rubiks_moves)
		s .= view(s0, rotation_lookup[i])
		update_rubiks_feature!(v, s)
		view(output, i, :) .= v
	end
	return output
end

# ╔═╡ 600334bd-2c7d-4c6c-a59d-922692351ab2
function rubik_two_step_lookahead(s::Matrix{UInt8})
	output = branch_rubik_moves(s)
	scores = 1:12 |> Map() do i
		s = bits2value(BitVector(output[i, :]))
		output = branch_rubik_moves(s)
		scores = output * solved_cube_bits
		maximum(scores)
	end |> tcollect
	findmax(scores)
end

# ╔═╡ 5f0dbbf0-949b-41ea-b129-feb76698582c
function rubik_three_step_lookahead(s::Matrix{UInt8})
	output = branch_rubik_moves(s)
	scores = 1:12 |> Map() do i
		s = bits2value(BitVector(output[i, :]))
		(score, index) = rubik_two_step_lookahead(s)
		score
	end |> tcollect
	findmax(scores)
end

# ╔═╡ 010f49e4-84c9-4a07-9958-5e38114599c1
function rubik_four_step_lookahead(s::Matrix{UInt8})
	output = branch_rubik_moves(s)
	scores = 1:12 |> Map() do i
		s = bits2value(BitVector(output[i, :]))
		(score, index) = rubik_three_step_lookahead(s)
		score
	end |> tcollect
	findmax(scores)
end

# ╔═╡ d7802dee-7a35-4919-8e01-bd93908b307e
function rubik_five_step_lookahead(s::Matrix{UInt8})
	output = branch_rubik_moves(s)
	scores = 1:12 |> Map() do i
		s = bits2value(BitVector(output[i, :]))
		(score, index) = rubik_four_step_lookahead(s)
		score
	end |> tcollect
	findmax(scores)
end

# ╔═╡ 0cca0e5b-88e3-43c7-92ac-9b993e583f22
function build_input_output_branches(s0::Matrix{UInt8}, move_history::Vector{Int64}; max_moves = 4)
	output = branch_rubik_moves(s0)
	move_history′ = [vcat(move_history, i) for i in 1:12]
	move = length(move_history)
	move + 1 >= max_moves && return output, move_history′
	
	s_bits = BitVector(zeros(288))
	update_rubiks_feature!(s_bits, s0)
	(state_bits, movehistory) = 1:12 |> Map() do i
		s = bits2value(output[i, :])
		build_input_output_branches(s, vcat(move_history, i); max_moves = max_moves)
	end |> foldxt((a, b) -> (vcat(a[1], b[1]), vcat(a[2], b[2])))
	# (state_bits, moves) = mapreduce((a, b) -> (vcat(a[1], b[1]), vcat(a[2], b[2])), 1:12) do i
	# 	s = bits2value(output[i, :])
	# 	build_input_output_branches(s, move + 1; max_moves = max_moves)
	# end
	usedinds = findall(x -> x < 48, state_bits*s_bits)
	return vcat(output, state_bits[usedinds, :]), vcat(move_history′, movehistory[usedinds])
end

# ╔═╡ cbe5cfca-d3b2-4874-bf32-80b7ca90ca71
(input, output) = build_input_output_branches(4)

# ╔═╡ 0cb63a54-e691-4602-99e9-0617f7ad0cce
sp_input = SparseMatrixCSC(input)

# ╔═╡ a074993e-53b4-4323-9d20-0e1c3c55a846
(Base.summarysize(input), Base.summarysize(sp_input))

# ╔═╡ 954f663b-1dce-491f-b999-343d538e0125
sbit_eg = input[50, :]

# ╔═╡ 855b0735-5d9d-496a-94fc-d770861647c8
ssp_eg = SparseVector(sbit_eg)

# ╔═╡ fd85a503-b349-4d54-ba90-72a162726990
sp_input*ssp_eg

# ╔═╡ 4eef8010-f192-4142-8287-6cef8ea19f6c
findfirst(input*sbit_eg .== 48)

# ╔═╡ c773cd92-01af-4625-9cbf-7e3f74583682
build_input_output_branches(initialize_rubiks_cube())

# ╔═╡ eb5d2eac-5280-4269-8b2d-3c31226b4b2f
const test_episode = runepisode(rubiks_cube_mdp; max_steps = 1_000)

# ╔═╡ d8e31a4e-7178-4ec3-85c6-0c5eb5fe9b0b
const rubik_sarsa_test = NonTabularRL.run_linear_semi_gradient_sarsa(rubiks_cube_mdp, 0.99f0, typemax(Int64), 1_000_000, Float32.(solved_cube_bits), update_rubiks_feature!; α = 1f-4)
# const rubik_sarsa_test = run_fcann_semi_gradient_sarsa(rubiks_cube_mdp, 0.99f0, typemax(Int64), 10_000_000, Float32.(solved_cube_bits), update_rubiks_feature!, [128, 128]; λ = 0f0, c = 10f0, dropout = 0.01f0, α = 2f-3)

# ╔═╡ 73a7493f-9193-4f96-867b-f35296ecfdc4
# ╠═╡ disabled = true
#=╠═╡
# const rubiks_monte_carlo_control_test = NonTabularRL.run_linear_gradient_monte_carlo_control(rubiks_cube_mdp, 0.99f0, 10_000_000, Float32.(solved_cube_bits), update_rubiks_feature!; α = 1f-4, ϵ = 0.01f0, suppress_warning=true, max_steps = 7)

const rubiks_monte_carlo_control_test = NonTabularRL.run_fcann_gradient_monte_carlo_control(rubiks_cube_mdp, 0.99f0, 10_000_000, Float32.(solved_cube_bits), update_rubiks_feature!, [256, 256, 256]; α = 1f-5, c = 100f0, dropout = 0.05f0, ϵ = 0.01f0, suppress_warning=true, max_steps = 7)
  ╠═╡ =#

# ╔═╡ 46618a5b-1a6b-4815-bbb6-d5c050e44945
#=╠═╡
plot(smooth_error(rubiks_monte_carlo_control_test.reward_history, 1000)[round.(Int64, LinRange(1, length(rubiks_monte_carlo_control_test.reward_history)-1000, 1000))])
  ╠═╡ =#

# ╔═╡ b243bb1e-b941-44bd-acba-21e797677dee
#=╠═╡
show_rubiks_episode(rubiks_monte_carlo_control_test.π_greedy; max_steps = 50, s0 = initialize_rubiks_cube(;num_actions = 7))
  ╠═╡ =#

# ╔═╡ 0bd4f561-5736-4b61-9c10-1006b99e60e4
#=╠═╡
 48 - (1:100_000 |> Map(_ -> rubiks_cube_mdp.initialize_state() |> score_cube) |> mean)
  ╠═╡ =#

# ╔═╡ 1a5aee67-f3b3-4c0b-ac67-096115d39d42
#=╠═╡
test_cube_policy(initialize_rubiks_cube(), rubiks_monte_carlo_control_test.π_greedy; maxsteps = 100)
  ╠═╡ =#

# ╔═╡ 88851fe7-a08d-4b9d-8e31-4802d25ab745
#=╠═╡
rubiks_monte_carlo_control_test.value_function(initialize_rubiks_cube())
  ╠═╡ =#

# ╔═╡ 6b410101-edb1-4814-8aa6-344d0c969e01
md"""
# Dependencies
"""

# ╔═╡ e956ffaa-01c3-4e44-8c9f-2298347fea03
html"""
<style>
	main {
		margin: 0 auto;
		max-width: min(1200px, 90%);
		padding-left: max(10px, 5%);
		padding-right: max(10px, 5%);
		font-size: max(10px, min(24px, 2vw));
	}
</style>
"""

# ╔═╡ dcf12932-7854-481c-8bfa-f6a4b2956518
#=╠═╡
begin
function add_elements(a, b)
	@htl("""
	$a
	$b
	""")
end
add_elements(a::HTML, b::HTML) = add_elements(a.content, b.content)
add_elements(a::HTML, b::AbstractString) = add_elements(a.content, b)
add_elements(a::AbstractString, b::HTML) = add_elements(a, b.content)
end
  ╠═╡ =#

# ╔═╡ 1737d09d-046f-49d4-9287-e67bfee13468
#=╠═╡
function render_face(face::AbstractVector{I}, face_number::Integer; square_pixels = 20) where I <: Integer
	@htl("""
	<div style = "display: flex; flex-wrap: wrap; width: $(3*square_pixels)px;">
	$(mapreduce(add_elements, [face[1:4]; face_number; face[5:end]]) do v
	@htl("""
	<div style = "background-color: $(square_colors[v]); width: $(square_pixels)px; height: $(square_pixels)px; border: 1px solid black;"></div>
	""")
	end)
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ 083caabf-4122-49fa-b397-7746b77ca12b
#=╠═╡
begin
	function render_cube(cube_values::Matrix{I}; kwargs...) where I <: Integer
		@htl("""
		<div style = "display: flex;">
		$(mapreduce(add_elements, 1:6) do i
		@htl("""
		<div style = "margin-right: 5px;">
		$(face_names[i])
		$(render_face(cube_values[:, i], i; kwargs...))
		</div>
		""")
		end)
		</div>
		""")
	end

	render_cube(cube_indices::Vector{UInt8}; kwargs...) = render_cube(reshape(solved_cube_values[cube_indices], 8, 6); kwargs...)

	render_cube(cube_indices; kwargs...) = render_cube(Vector(cube_indices); kwargs...)
end
  ╠═╡ =#

# ╔═╡ 8aa9a6dd-130a-4439-83a8-5b0385118a49
#=╠═╡
render_cube(solved_cube_values)
  ╠═╡ =#

# ╔═╡ 0fd13622-f89b-472b-9c82-9142acf07359
#=╠═╡
render_cube(rotate_cube(solved_cube_indices, RubiksMove{root_move_eg.face, root_move_eg.rot}()))
  ╠═╡ =#

# ╔═╡ 5e621aeb-a106-4147-89c8-4b8635fc6342
#=╠═╡
begin
	eg_init_cube = initialize_rubiks_cube(init_actions)
	md"""
	Score = $(round(Int64, score_cube(eg_init_cube)))
	$(render_cube(eg_init_cube))
	"""
end
  ╠═╡ =#

# ╔═╡ c7cfee1d-4d1a-4e6e-ac89-a8fcfde55ecb
#=╠═╡
[render_cube(x) for x in runepisode(rubiks_cube_mdp; max_steps = 5)[1]]
  ╠═╡ =#

# ╔═╡ 44eb89fc-f006-432c-a633-c433f9524a63
#=╠═╡
render_cube(mcts_s0)
  ╠═╡ =#

# ╔═╡ 6ec48894-7cc4-46ea-90b5-5f0e8e566b5a
#=╠═╡
render_cube(test_episode[1][1])
  ╠═╡ =#

# ╔═╡ a2464600-d228-4f8c-96c0-6b8fa8c87afe
#=╠═╡
function show_rubiks_episode(π::Function; s0 = rubiks_cube_mdp.initialize_state(), kwargs...)
	(states, actions, rewards, sterm, steps) = runepisode(rubiks_cube_mdp; s0 = s0, π = π, kwargs...)
	initial_score = score_cube(s0)
	(max_reward = 48 - initial_score, reward_sum = sum(rewards), steps = steps, rendered_states = [render_cube(c) for c in [states[1], states[end], sterm]], states = states)
end
  ╠═╡ =#

# ╔═╡ 9c94259e-a42a-44e1-90db-fafa4cc23b1c
#=╠═╡
show_rubiks_episode(s -> test_sarsa_output.value_function(s)[2]; max_steps = 100, s0 = initialize_rubiks_cube(50))
  ╠═╡ =#

# ╔═╡ 1341aded-632b-4650-a4b6-de5c38fdda99
#=╠═╡
show_rubiks_episode(s -> test_dp_output.π_greedy((cube=s, move_count = 0, scramble_moves = 30)); max_steps = 100, s0 = initialize_rubiks_cube(6))
  ╠═╡ =#

# ╔═╡ e9e35805-f669-42ee-b0dc-51ab8c800d2f
#=╠═╡
mcts_output = show_rubiks_episode(s -> rubiks_mcts_policy(s; nsims = 100, depth = 10, c = 0.5f0); max_steps = 10, s0 = mcts_s0)
  ╠═╡ =#

# ╔═╡ f7155154-3062-4711-80b3-8139c792e25d
#=╠═╡
show_rubiks_episode(value_iteration_π; max_steps = 20, s0 = initialize_rubiks_cube(10)).states |> states -> [render_cube(s) for s in states]
  ╠═╡ =#

# ╔═╡ 323be52e-f967-44aa-a478-53b5f6575dab
#=╠═╡
function compare_mcts_endpoint(nsims, depth, c, steps, s0)
	mcts_output = show_rubiks_episode(s -> rubiks_mcts_policy(s; nsims = nsims, depth = depth, c = c); max_steps = steps, s0 = s0)
	x = any(haskey(rubiks_tabular_mdp.state_index, s) for s in mcts_output.states)
	(; success = x, mcts_output...)
end
  ╠═╡ =#

# ╔═╡ 3d9b2ecf-575b-44a8-9fb0-f62589a8abfa
#=╠═╡
show_rubiks_episode(s -> rubik_five_step_lookahead(s)[2]; max_steps = 20, s0 = initialize_rubiks_cube(;num_actions = 30))
  ╠═╡ =#

# ╔═╡ 04065d78-3d48-495e-917c-3b2e4d990342
#=╠═╡
show_rubiks_episode(s -> rubik_sarsa_test.value_function(s)[2]; max_steps = 100, s0 = initialize_rubiks_cube(;num_actions = 30))
  ╠═╡ =#

# ╔═╡ 00e7e9bf-3c9c-47e6-bbd5-a63d471bf6a3
#=╠═╡
render_cube(solved_cube_values)
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
BenchmarkTools = "~1.5.0"
DataFrames = "~1.7.0"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.3.1"
PlutoDevMacros = "~0.9.0"
PlutoPlotly = "~0.5.0"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.60"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.6"
manifest_format = "2.0"
project_hash = "fcfe33f0649fb625dc2d76f98dec33c70475d2e6"

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

[[deps.BaseDirs]]
git-tree-sha1 = "cb25e4b105cc927052c2314f8291854ea59bf70a"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.2.4"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b5278586822443594ff615963b0c09755771b3e0"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.26.0"

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
git-tree-sha1 = "3a3dfb30697e96a440e4149c8c51bf32f818c0f3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.17.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

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
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "62ca0547a14c57e98154423419d8a342dca75ca9"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.4"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

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
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.InlineStrings]]
git-tree-sha1 = "8594fac023c5ce1ef78260f24d1ad18b4327b420"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.4"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "2984284a8abcfcc4784d95a9e2ea4e352dd8ede7"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.36"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

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
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlutoDevMacros]]
deps = ["JuliaInterpreter", "Logging", "MacroTools", "Pkg", "TOML"]
git-tree-sha1 = "72f65885168722413c7b9a9debc504c7e7df7709"
uuid = "a0499f29-c39b-4c5c-807c-88074221b949"
version = "0.9.0"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "BaseDirs", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "Reexport", "TOML"]
git-tree-sha1 = "653b48f9c4170343c43c2ea0267e451b68d69051"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.5.0"

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
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProfileCanvas]]
deps = ["FlameGraphs", "JSON", "Pkg", "Profile", "REPL"]
git-tree-sha1 = "41fd9086187b8643feda56b996eef7a3cc7f4699"
uuid = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
version = "0.1.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
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
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

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
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

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
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─def37ed7-42e8-4f07-9a26-e0a0dc2bf9ea
# ╠═1c8bc474-9b46-4e99-b858-2a9a980f93f4
# ╠═eb46363c-35fa-4c31-b611-f349d50e167f
# ╠═42816516-dcf0-40b7-8787-6608fcd07831
# ╠═f1634b1a-cc22-42e4-90af-72fc5495ed17
# ╠═8e58e9a2-9831-40fa-9ad3-bd82f6629bf7
# ╠═3696b6c0-804a-11ef-0d86-7b48f7a6697c
# ╠═f7201b43-6c8c-4b49-a0fe-7bd1378d1649
# ╟─c6a76e1e-2121-4d80-a1f6-ddf78b6f6d55
# ╟─8aa9a6dd-130a-4439-83a8-5b0385118a49
# ╟─f1b1c2fc-6973-4ecd-ae9c-1f52a1fd2189
# ╠═c6ccf794-d7e4-4e7e-b4f7-7acbe2530722
# ╠═a9748352-649d-45c4-b33d-a4c8f2b5b247
# ╠═ebc5bf2b-abef-4b4a-9f12-b0884c089350
# ╠═ebf0eee8-b1d4-4f59-9727-518e6dd546bd
# ╟─46bbd599-ae25-485e-b064-af28efde316f
# ╠═545ca130-8e89-48db-9ada-5f6467067ef3
# ╟─44e167a7-ebe7-4cb0-8e4c-9a5444f5f224
# ╠═7aedab25-1129-4ad5-bfc5-bd673b8d2509
# ╟─4dd63f4c-32ce-4672-a165-6a96ef75895f
# ╠═8bec2716-88f5-4998-8102-4d63b0f8f99b
# ╟─d19bf533-22ea-427d-aa63-5556ba7880b4
# ╠═eb1f5b65-e804-4578-9123-ea8d65e75397
# ╟─4aeac615-6129-4482-bdc4-a0549d069d35
# ╠═95854163-4da3-4d67-b254-1eab96c9fcc6
# ╟─0051d77d-e74c-4d34-b17f-e310528ef9f4
# ╟─d4613756-bc2d-4632-8e74-6fa11d5ae6a4
# ╠═22879750-f3c1-4053-994c-fb1262107be3
# ╟─0b826bd0-8a83-47d1-934b-e9ddc50c91a9
# ╟─0fd13622-f89b-472b-9c82-9142acf07359
# ╟─bb3d3886-cbbc-4584-aed2-a53ed7f2d878
# ╟─78aaa31a-66ba-4325-915c-383e648fa801
# ╠═1157c771-1121-47da-8d4a-39c10a55df85
# ╠═361b9a94-ab5f-474d-8893-9987be4f0c5d
# ╠═8fe5898c-2c50-4a36-a49d-37fec4eef1a3
# ╠═b31848b1-3a25-4cd5-8ca2-748f35a010e2
# ╠═44499455-ab75-4a65-9fec-1d544afb8a33
# ╠═7982b2b3-3cad-4b2a-ba9e-c9b6ba4ef728
# ╠═4036aeb4-65fd-449a-a259-1683de4f9c30
# ╟─053f6365-ad13-45f2-851c-ddcc16f98e4a
# ╠═8c396914-675b-404f-9f0c-90e880c55658
# ╟─cced6b32-f170-4470-8fbf-02bc2e8ad0a1
# ╟─5e621aeb-a106-4147-89c8-4b8635fc6342
# ╠═1fbcf1f1-722a-4014-b8f6-11c2a5f12548
# ╠═b28d50ad-ca0a-4743-bace-be46c0821ac2
# ╠═fe20a8bd-c608-4ac7-9bb0-3251bcf7d85c
# ╠═c7cfee1d-4d1a-4e6e-ac89-a8fcfde55ecb
# ╟─db9454b3-4ee0-4f16-9649-791a9e246939
# ╠═afad0f03-8b3d-4348-9570-9fe9489f0cb5
# ╠═6aec543d-eec2-47f3-9dbf-2631688e1112
# ╠═cd53cf28-6542-45aa-b9dc-38fa07f3e018
# ╠═5f6ed356-71c3-44a1-ae1f-ac8ed55a1c2b
# ╠═09538cca-7da3-4d48-9541-90f00acce794
# ╠═2040ffd4-f977-4c30-83a0-875b500099aa
# ╠═80069af6-31fd-4042-8690-a388c213fe28
# ╠═68150249-6db1-4dca-b998-9521a6d7d98d
# ╠═662729cc-54fa-449c-a90b-4c32196aef20
# ╟─343aa4fe-f38d-42b7-967b-c589be65077d
# ╟─7e57369f-14d5-4a5e-b83d-951228fac2eb
# ╠═027d5ff6-b250-427f-9a00-ec9562236bc3
# ╠═35ae742a-db14-4868-bb2d-d59cd9300de4
# ╠═9c94259e-a42a-44e1-90db-fafa4cc23b1c
# ╟─f78706e3-2b14-4bc8-9b96-2e4e98a2efa4
# ╠═dddcd1ee-4bc7-4791-ba0f-c8ee77e12246
# ╟─409dd109-ad0c-4a94-8722-7bc1439f2625
# ╠═d062abca-a910-462b-992a-fe713985d644
# ╠═92b86eb0-e506-4104-8bc0-3f2550398455
# ╠═bee4261d-76ce-440f-8285-4b0816726f35
# ╠═e6a19274-9069-419e-a6e1-0feed72d1dfe
# ╠═bf27bdef-d382-4999-972a-5f3934566c95
# ╠═6148ec6a-5d38-40d1-84ba-8b5253d5fdaa
# ╠═cf47193d-4c64-4dd5-836e-77dd59a2d586
# ╠═043fc456-0a9f-4023-aea9-d2775640dcd8
# ╠═592a404d-f788-4781-b614-69f90dede7d5
# ╠═bbe646c0-0de0-4bbf-9c75-ac9cc9491492
# ╠═ddecab58-0965-4718-b5b9-3eff15c9e9dd
# ╠═ddf57ab2-2bf5-4914-99b2-c0b0e9562e40
# ╠═69f09d4e-743c-45c6-bd96-9b60ba724992
# ╟─8d8c360f-66b7-4d50-bff8-803e77ef688e
# ╟─7cd30b68-1796-4dd0-bf13-dfb71610637a
# ╠═fd053336-7b1f-4c79-afaa-f60b6fcc5938
# ╠═4085110c-7b47-440c-b1bf-020b91168183
# ╠═4edd1430-5779-471e-b12c-843f6abce802
# ╠═013fe195-8a8a-4717-a95b-16bb86e8de29
# ╟─62650c81-17cd-4e2e-ab8d-a92c6b3eefb3
# ╠═97f64a35-b10d-4c3d-ac28-468110f9177f
# ╠═d163d3cd-e8ca-44fb-81b2-66a2e1a92812
# ╠═a538ead8-188b-4dd6-a105-a0c7a5d7d64f
# ╠═26fcb97c-9d79-4638-a100-8eb8c59f1ef8
# ╠═7b0af922-aca3-4e3b-b521-a01c6cd6889c
# ╠═1341aded-632b-4650-a4b6-de5c38fdda99
# ╠═0403bc9e-05b0-4662-a3ec-f78f7205ab1f
# ╠═2f2a9538-8046-413f-945e-ea6fdc9809ed
# ╠═3f289f24-9992-45f0-81be-683e7e152199
# ╠═f55b65e0-cb95-4403-a632-d95c50c893a9
# ╠═7bc8ea32-92c8-4a80-a64e-44e6b2af408b
# ╠═9899c955-e5f5-413a-ba9e-cd57dda60eb4
# ╠═77bb49f2-1c14-48ca-8d10-34000e6b4541
# ╟─7f12e90c-1dd5-49e4-91a1-786e3165c769
# ╠═d20f376b-7973-495f-811e-75a7fff0cd9c
# ╠═b86c7a49-5769-49b5-945d-e54a0a2a2c86
# ╠═c8c3a104-0183-4e1b-9d9d-b2e2c6248f4c
# ╠═df483fd1-737b-44d2-8c25-a49d756fb05c
# ╠═73ac41c1-0c82-4772-bf05-351bda74e444
# ╠═4955995a-5051-4d5e-a336-008834ba8ca2
# ╠═0b4860d3-8153-49a7-be54-16ccc589ec9b
# ╠═21b2a4ec-8af9-4884-8439-f6868ac12804
# ╠═f3ee2969-cafa-4e24-8295-0e5a1035eb4a
# ╠═b492634c-5c93-4d8f-80df-7c77ad04e14e
# ╠═f6adca6d-8cc2-4fd9-a446-3564b68c9499
# ╠═dd2c8eec-1e56-4315-8e14-7f031b44bcf0
# ╠═3d5eb8a6-29bf-49d5-acec-f05897ca1416
# ╠═1c69a2e6-4396-4460-8f52-90c626bebc24
# ╠═d127796c-9dbe-4ca6-a11d-0197875d095a
# ╟─c9bf5811-6a61-45f4-9070-6f2b7277611f
# ╠═18fa27e3-d076-4543-92cc-679ee3c32156
# ╠═09cf4ae7-d1fb-44d6-a617-0dbc7f64c2f2
# ╠═3617c5a0-79e6-4fc0-814e-45bbdd373b5d
# ╟─44eb89fc-f006-432c-a633-c433f9524a63
# ╠═cfab1b66-884a-4db6-bdae-6df295163838
# ╠═2e177050-ee05-4de9-9fa8-4e23199fc669
# ╠═4c53d04e-df52-4960-ba46-ed3bff1a624b
# ╠═30af4178-b39b-4389-a664-6c9e457f8ca1
# ╠═be7b4bb5-0deb-4c96-ad0c-90e785a3ff28
# ╠═1bf2d626-12c4-4065-a63d-c007f6ddd15b
# ╠═11d96bf0-c09d-411d-a2fa-1e172ab1b7aa
# ╠═e9e35805-f669-42ee-b0dc-51ab8c800d2f
# ╟─fca1535c-20f3-450e-95b3-0130e8a49b48
# ╠═d3feb1af-b300-4a36-844b-3ad6dfe9a758
# ╠═c42847a7-7324-4c30-8275-0638fd289661
# ╠═b88b019b-dd73-464d-a720-0394ad3c712c
# ╠═d3383318-c11e-4616-a2de-26f27023043d
# ╠═82e56315-c8f1-4c01-bbc6-dc1c5588a60a
# ╠═c6fd1a65-014d-4493-928d-c15d598d8415
# ╠═914a8139-d934-48aa-ac4b-ae0a3f7363f7
# ╠═0570b1ce-b71b-4ccf-88ee-a75e90c426d1
# ╠═5affaa6c-eef7-4682-9bce-47f5bfd7d5c0
# ╠═645aad71-f90b-4d04-a12e-d267466a5665
# ╠═20a5c2f2-b031-4d7f-91f5-d775aaf4f593
# ╠═625dc7bb-8406-4165-893a-2b8d7a06d011
# ╠═04561ac3-a19d-4ece-86dc-1e5d024d8f84
# ╠═a77ec593-5ee3-4bca-8064-326573a3edfe
# ╠═10e951ff-1b3b-46cd-9a3f-ba819804b65e
# ╠═46af888e-e57e-456e-af71-bdcf22dc7798
# ╠═81cbe49d-364b-450e-8e2f-c496525e3aae
# ╠═7464b10d-056b-47cc-8db8-44ace4b84d11
# ╠═fa2286f8-7b20-4f33-8f5a-bccacbe758e9
# ╠═ef39bdd3-c085-441c-a8cd-6c1221ad4b21
# ╠═725317cb-e761-49f3-9c03-a15bf9c34da9
# ╠═0d62e009-81be-43d3-add5-ce4ac591dbc7
# ╠═1938becf-e2cc-4c7b-b6ec-d1c3ff107ed9
# ╠═b22c8404-b359-44f2-b81c-e0c9f4834db5
# ╠═a487ccf0-c293-4093-b936-f92094e86fa7
# ╠═7bb3fe7c-a8a5-41b0-b35c-2ca92a4d2eb0
# ╠═a06c96f9-57c1-48e4-a725-08980892502e
# ╟─206a1146-4a01-412f-a17e-3d58cac83453
# ╠═91e6576c-0142-409c-8749-194c66ba3c9d
# ╠═abfc1d27-96a6-48dc-93d1-b2f5db4455ef
# ╠═1e6c3190-6b5f-4ada-a24f-98483658aa17
# ╠═116c751a-76e6-4139-9926-0e8f58d30fe0
# ╠═8fc4500b-de4f-44f0-ac51-b2fb875a336a
# ╠═d84d5e1c-bcfb-451c-9a85-11ec329c5d22
# ╠═4d848bb9-2d35-4568-84db-edd229c07ec9
# ╠═54c914bc-8822-4fd4-bd38-97fd1476558e
# ╠═3d1ad253-1fe6-4e5e-9959-690d88a278e4
# ╠═87c318f7-197a-4b51-b38f-137ffb52f3d7
# ╠═338515c3-6818-4c23-9d98-178e74c5f148
# ╠═22e630e8-2f5b-40f6-9bbe-b0e6761b4e76
# ╠═39c7bb0d-093c-4be7-a366-2adcbfe7a7a6
# ╠═f7155154-3062-4711-80b3-8139c792e25d
# ╟─64c85a03-d88e-471e-8215-5b4ff51b5440
# ╠═0624c4dc-d6e4-4a31-b13d-3618d124f857
# ╠═323be52e-f967-44aa-a478-53b5f6575dab
# ╟─7ff44be5-a4df-4862-b0e5-302588c127c4
# ╠═5a75cb17-56cb-4808-8c41-76fa671c50cf
# ╠═e74f5d40-99a4-414c-878f-0d429faddea2
# ╟─cd398ac4-82f3-4c06-afd5-019864064197
# ╠═434df85d-fbdc-41e7-98f7-6948e4aacea2
# ╠═89f0f0ec-f659-4752-93fe-d90c0ac248e4
# ╠═ffbd0f1d-306a-4b3a-84a4-53c7d3f0fb27
# ╠═03ed5828-ae46-4a9a-abfa-156c7c0ed989
# ╠═d795f457-f7a8-453f-a440-6a0d3f6fce09
# ╠═acbbc9ff-f26f-49de-9be4-2de2faefc3c8
# ╠═613191fe-307e-4bf1-be1a-74df1ed11c95
# ╠═be2e7ca8-ac9f-439b-aeb4-cf5f12aae716
# ╠═efb5897b-4f97-4ba3-8257-93c6c24eebec
# ╠═c2a1a4a9-c652-400f-ad3a-5a05d4e10073
# ╠═b68a651b-f16d-4662-83ee-2120daf73512
# ╠═790aea25-52d1-41c0-8b80-5b987c294c54
# ╠═daf5d43c-093f-4dd1-9a5a-a68afd40aa54
# ╠═3c054a02-1c38-4df6-a559-d6f345d6d795
# ╠═65a1cb65-6f9e-43fa-b879-fa5ed448da3c
# ╠═df59ec2c-1032-4c44-962c-1014fb4760f4
# ╠═3ce3c964-4272-48f0-ae40-8f77f333f4ca
# ╠═7d3856b9-38e9-40dc-b9da-44ae94e94522
# ╠═0acf8afd-166a-4057-8bdb-e7d95e761145
# ╠═14768320-8400-43ca-ad15-5adf9c78fbbd
# ╠═87ad6b4f-5322-459a-9aac-e2996f45a67c
# ╠═cbe5cfca-d3b2-4874-bf32-80b7ca90ca71
# ╠═f674e316-f3e1-44bd-98eb-f5812c045c77
# ╠═0cb63a54-e691-4602-99e9-0617f7ad0cce
# ╠═a074993e-53b4-4323-9d20-0e1c3c55a846
# ╠═954f663b-1dce-491f-b999-343d538e0125
# ╠═855b0735-5d9d-496a-94fc-d770861647c8
# ╠═4eef8010-f192-4142-8287-6cef8ea19f6c
# ╠═fd85a503-b349-4d54-ba90-72a162726990
# ╠═1a7e54e3-8a2b-4680-88f3-643967af82a2
# ╠═600334bd-2c7d-4c6c-a59d-922692351ab2
# ╠═5f0dbbf0-949b-41ea-b129-feb76698582c
# ╠═010f49e4-84c9-4a07-9958-5e38114599c1
# ╠═d7802dee-7a35-4919-8e01-bd93908b307e
# ╠═3d9b2ecf-575b-44a8-9fb0-f62589a8abfa
# ╠═91a8ca59-c261-4e40-97e7-43b6fd2f87f3
# ╠═635a7314-c6df-4ddd-863c-833166069de6
# ╠═0cca0e5b-88e3-43c7-92ac-9b993e583f22
# ╠═c773cd92-01af-4625-9cbf-7e3f74583682
# ╠═ad7559e2-c938-4396-9fab-09b9ecfcbfc4
# ╠═619ff1aa-61ba-495d-8869-047cbf7ca0a1
# ╠═7999388a-eff6-4cca-8f26-373317654f25
# ╠═6f227e33-370c-4f75-b10e-6d48d73e401e
# ╠═c8cc0023-6f5e-4696-8303-3c5e9c52d0dc
# ╠═5d7746fd-296d-4866-9c7b-9019f1f3e02a
# ╠═ee3461a8-26a8-4e80-9e4c-42fa3d8787fd
# ╠═474b56e9-c8ee-4db3-80c0-2408ae5008e5
# ╠═f022bd7b-9a34-4940-9a52-13b8ad2eefdc
# ╠═0d80de04-ac65-4cce-9db2-5f3053079a1b
# ╠═17006174-caad-4043-b8d5-883aa0e10c80
# ╠═8218ea8b-3ba6-45fb-ac36-89ba6cccf112
# ╠═27b38778-17fe-4aca-8cec-341e2333e536
# ╠═639b2d17-59c9-4605-a394-8fce6dc5449b
# ╠═1af6a2f7-ea00-4d96-8045-2ce280b5a833
# ╠═f8d68c24-71f0-4850-804d-67e5634ed60d
# ╠═eb5d2eac-5280-4269-8b2d-3c31226b4b2f
# ╠═6ec48894-7cc4-46ea-90b5-5f0e8e566b5a
# ╠═d8e31a4e-7178-4ec3-85c6-0c5eb5fe9b0b
# ╠═04065d78-3d48-495e-917c-3b2e4d990342
# ╠═73a7493f-9193-4f96-867b-f35296ecfdc4
# ╠═46618a5b-1a6b-4815-bbb6-d5c050e44945
# ╠═b243bb1e-b941-44bd-acba-21e797677dee
# ╠═0bd4f561-5736-4b61-9c10-1006b99e60e4
# ╠═a2464600-d228-4f8c-96c0-6b8fa8c87afe
# ╠═1a5aee67-f3b3-4c0b-ac67-096115d39d42
# ╠═88851fe7-a08d-4b9d-8e31-4802d25ab745
# ╟─00e7e9bf-3c9c-47e6-bbd5-a63d471bf6a3
# ╠═1737d09d-046f-49d4-9287-e67bfee13468
# ╠═083caabf-4122-49fa-b397-7746b77ca12b
# ╟─6b410101-edb1-4814-8aa6-344d0c969e01
# ╠═c5256762-ca4a-4c81-805b-4f865efc2091
# ╠═624eef76-16a7-4556-a466-14341346f7a5
# ╠═c5c0f635-171d-4904-9675-d1b0a01f6d7a
# ╠═e956ffaa-01c3-4e44-8c9f-2298347fea03
# ╠═dcf12932-7854-481c-8bfa-f6a4b2956518
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
