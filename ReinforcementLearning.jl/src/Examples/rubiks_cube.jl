### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ df1dc37b-0c14-4a53-86e2-aab583e9c40a
using PlutoLinks, PlutoHooks, Base.Threads

# ╔═╡ c5256762-ca4a-4c81-805b-4f865efc2091
using PlutoDevMacros

# ╔═╡ 624eef76-16a7-4556-a466-14341346f7a5
begin
	PlutoDevMacros.@frompackage @raw_str(joinpath(@__DIR__, "..", "NonTabularRL.jl")) begin 
		using NonTabularRL
		using >.Random, >.Statistics, >.LinearAlgebra, >.Transducers, >.StaticArrays, >.DataStructures
	end
	switch_device(3)
end

# ╔═╡ c5c0f635-171d-4904-9675-d1b0a01f6d7a
# ╠═╡ show_logs = false
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI, PlutoPlotly, PlutoProfile, BenchmarkTools, LaTeXStrings, HypertextLiteral, DataFrames, Dates
	
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

Given our orientation convention, the only moves that need to be considered are face rotations.  Each of the 6 faces can be rotated clockwise or counter-clockwise with respect to the face itself.  A "Double" direction is just a half-turn which can be accomplished with a quarter turn repeated for either direction.  With these three directions no face needs to be moved more than once so any logical move list should switch the face being moved on every step.
"""

# ╔═╡ c6ccf794-d7e4-4e7e-b4f7-7acbe2530722
begin
	abstract type Direction end
	struct Clockwise <: Direction end 
	struct CounterClockwise <: Direction end 
	struct Double <: Direction end
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
const direction_order = [Clockwise, CounterClockwise, Double]

# ╔═╡ a9748352-649d-45c4-b33d-a4c8f2b5b247
const rubiks_moves = [RubiksMove{F, D}() for F in face_order for D in direction_order]

# ╔═╡ b31fda46-27e1-4b10-9cc3-de26557521f5
md"""
When a face is rotated, it affects the position of pieces on every other face except for the opposite one.  The opposite face is unaffected and the following vector lists the index of the single face unaffected by a move of the face at each index.  The purpose of this property is that it can be used to determine if a three move sequence is redundant since moves rotating two opposite faces should never be followed by a move of the starting face which would again be redundant.
"""

# ╔═╡ 42e03441-cd7c-4f3c-a30d-9814b36428c7
const face_independence_lookup = [4, 6, 5, 1, 3, 2]

# ╔═╡ ebc5bf2b-abef-4b4a-9f12-b0884c089350
const rubiks_move_index = TabularRL.makelookup(rubiks_moves)

# ╔═╡ 46bbd599-ae25-485e-b064-af28efde316f
md"""
Any move can be reversed by applying the opposite rotation to the same face.  Any move followed by its reversal will result in the same cube state.  A double rotation is its own reverse move.
"""

# ╔═╡ 545ca130-8e89-48db-9ada-5f6467067ef3
begin
	reverse_move(move::RubiksMove{F, Clockwise}) where F<:Face= RubiksMove{F, CounterClockwise}()
	reverse_move(move::RubiksMove{F, CounterClockwise}) where F<:Face = RubiksMove{F, Clockwise}()
	reverse_move(move::RubiksMove{F, Double}) where F<:Face = RubiksMove{F, Double}()
end

# ╔═╡ 44e167a7-ebe7-4cb0-8e4c-9a5444f5f224
md"""
## Face Rotation Permutations

From the standard orientation, the face squares are numbered from 1 to 8 in left to right top to bottom order.  For the 4 faces perpendicular to the surface, the squares for the faces will follow that convention when the cube is rotated so that face is in the front position.

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

# ╔═╡ fabd617b-cff1-4939-8e10-b2d79861fd47
reshape(solved_cube_indices, 8, 6)

# ╔═╡ 33d1609a-5c0d-4cb4-a7e1-1aca85da74e1
function get_piece_indices(piece::Tuple{NTuple{N, Int64}, NTuple{N, Int64}}) where N
	[begin
		f = piece[1][i]
		n = piece[2][i]
		UInt8((f-1)*8 + n)
	end
	for i in 1:N]
end

# ╔═╡ ec496cfb-238d-48a5-a482-cfbe35cf2854
#each piece is defined by a pair of tuples where the first defines the face and the second defines the index on that face
const corner_pieces = [
	((5, 2, 1), (3, 6, 1)),
	((1, 2, 3), (3, 8, 1)),
	((5, 1, 6), (8, 6, 1)),
	((6, 1, 3), (3, 8, 6)),
	((4, 2, 5), (3, 1, 1)),
	((3, 2, 4), (3, 3, 1)),
	((5, 6, 4), (6, 6, 8)),
	((6, 3, 4), (8, 8, 6))
]

# ╔═╡ 28e5f0e7-d26b-49ff-bc1a-e7f177466053
const corner_inds = get_piece_indices.(corner_pieces)

# ╔═╡ ef5827b1-11b7-414a-b1d2-4410684277fa
const corner_inds_flat = reduce(vcat, corner_inds)

# ╔═╡ 05114a47-f015-410e-8416-34cc6b598f3b
#each piece is defined by a pair of tuples where the first defines the face and the second defines the index on that face
const edge_pieces = [
	((1, 2), (2, 7)),
	((1, 3), (5, 4)),
	((1, 5), (4, 5)),
	((1, 6), (7, 2)),
	((3, 2), (2, 5)),
	((4, 2), (2, 2)),
	((5, 2), (2, 4)),
	((3, 4), (5, 4)),
	((3, 6), (7, 5)),
	((4, 5), (5, 4)),
	((4, 6), (7, 7)),
	((5, 6), (7, 4))
]

# ╔═╡ 50e9caf2-83fd-47f4-8838-9f6fc82aa2cd
const edge_inds = get_piece_indices.(edge_pieces)

# ╔═╡ 9bfc604d-ca3e-4ccc-852b-63adfe67e8ef
const edge_inds_flat = reduce(vcat, edge_inds)

# ╔═╡ 39a2ce90-9ba4-4079-9c7a-b91e2a1e57db
make_piece_lookup(piece_inds) = reduce(merge, (Dict(v => (i, j) for (j, v) in enumerate(piece_inds[i])) for i in eachindex(piece_inds)))

# ╔═╡ 85155ccc-ea07-4734-9e01-04fc723ce619
#these lookups will take any cube index and map it to which piece it belongs to and in which position it is in compared to the position in a solved cube.  The position is critical for getting the orientation information while the first number tells you which piece it corresponds to as they are ordered by the face ordering of a solved cube.  We only need to look up 20 total pieces instead of the usual 48.  That is the 12 edges and 8 corners.  

# ╔═╡ d30cf757-2edd-4341-a3b2-b611c58ebe2b
const corner_lookup = make_piece_lookup(corner_inds)

# ╔═╡ b62a3967-7642-43ed-9c0d-141f33c6de2b
const edge_lookup = make_piece_lookup(edge_inds)

# ╔═╡ bec84bcd-9b14-4793-b4de-a1f5c5a71247
const piece_check_inds = reduce(vcat, [[a[1] for a in piece_inds] for piece_inds in (corner_inds, edge_inds)])

# ╔═╡ 24aa60e2-44fb-43ec-b0e3-624005493d58
function calculate_cube_pieces(cube::AbstractVector{I}) where I <: Integer
	#I only need to check 20 faces to complete this lookup.  I should not repeat any pieces so I will select 20 faces that all appear on distinct pieces.  By convention I will choose the first index that belongs to each piece in their indices defined above
	corner_pieces = (corner_lookup[cube[first(inds)]] for inds in corner_inds) 
	edge_pieces = (edge_lookup[cube[first(inds)]] for inds in edge_inds)
	(corner_pieces = corner_pieces, edge_pieces = edge_pieces)
end

# ╔═╡ 44e0a960-97a9-4d08-aa3f-9ceb212c7f1d
calculate_cube_pieces(solved_cube_indices).corner_pieces |> collect

# ╔═╡ abe87ea5-0d1f-4890-978c-948c99afe493
#I can define the pieces just by a single one of these elements and map each one.  I need to see which permutation it is which I can check by having three different lookups for the corners and two different lookups for the edges.

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

# ╔═╡ 4442e1c0-1e16-4a26-8498-252053b7ee95
md"""
## Rubik's MDP
"""

# ╔═╡ 8838f73a-c9b7-4b15-aa4d-b484452e71e1
md"""
## Rubik's Scramble Reset MDP

In addition to this environment resetting the episode after a certain number of moves, it tells the learning algorithms that each transition is deterministic.  With that information, we can learn a state value function only instead of an action value function and then use the transition to calculate the action values.
"""

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

# ╔═╡ 36bd6bf0-daaa-4780-a58d-e87fa96ae19a
function get_active_rubiks_features(cube::AbstractVector{I}) where I <: Integer
	((i-1)*48 + cube[i] for i in eachindex(cube))
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

# ╔═╡ 5f6ed356-71c3-44a1-ae1f-ac8ed55a1c2b
function make_rubiks_feature2(cube::AbstractVector{I}) where I <: Integer
	v = zeros(Float32, 48*6)
	update_rubiks_feature2!(v, cube)
	return v
end

# ╔═╡ f1256bc8-a69e-4daf-a304-bd436ce8cfbe
const rubiks_binary_feature = BinaryFeatureVector(48*48)

# ╔═╡ 066ec351-b414-4232-a614-09b0430ede86
function update_rubiks_feature!(v::BinaryFeatureVector{Int64, 48*48}, cube::AbstractVector{I}) where I<:Integer
	active_features = ((i-1)*48 + cube[i] for i in 1:48)
	NonTabularRL.update_binary_feature_vector!(v, active_features)
end

# ╔═╡ 80069af6-31fd-4042-8690-a388c213fe28
const solved_cube_feature2 = make_rubiks_feature2(solved_cube_indices)

# ╔═╡ a6492883-111f-4422-8e0a-13b8364b9e66
md"""
## Essential Index Alternative

Due to the constraints of the cube, we do not need to identify the location of all 48 faces.  Instead we can use a single face for each of the unique pieces (8 corners and 12 edges).  Using the convention that we select the face that would appear on the first ordered side of the cube in the solved position.  For example, we can use the first face to identify the first corner piece and we would use the face on the front side.
"""

# ╔═╡ 5a325bc1-1a12-4534-91d1-74a0a20bf1e4
const essential_inds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 21, 23, 24, 29, 31, 32, 39]

# ╔═╡ e173c7b0-6360-4af7-8bc4-2377b30765c8
const cube_index_pieces = [
	#front face
	(:c, 1, 1),
	(:e, 1, 1),
	(:c, 2, 1),
	(:e, 2, 1),
	(:e, 3, 1),
	(:c, 3, 1),
	(:e, 4, 1),
	(:c, 4, 1),
	#top face
	(:c, 5, 1),
	(:e, 5, 1),
	(:c, 6, 1),
	(:e, 6, 1),
	(:e, 7, 1),
	(:c, 1, 2),
	(:e, 1, 2),
	(:c, 2, 2),
	#right face
	(:c, 2, 3),
	(:e, 7, 2),
	(:c, 6, 2),
	(:e, 3, 2),
	(:e, 8, 1),
	(:c, 4, 2),
	(:e, 9, 1),
	(:c, 7, 1),
	#back face
	(:c, 6, 3),
	(:e, 5, 2),
	(:c, 5, 2),
	(:e, 8, 2),
	(:e, 10, 1),
	(:c, 7, 2),
	(:e, 11, 1),
	(:c, 8, 1),
	#left face
	(:c, 5, 3),
	(:e, 6, 2),
	(:c, 1, 3),
	(:e, 10, 2),
	(:e, 2, 2),
	(:c, 8, 2),
	(:e, 12, 1),
	(:c, 3, 2),
	#bottom face
	(:c, 3, 3),
	(:e, 4, 2),
	(:c, 4, 3),
	(:e, 12, 2),
	(:e, 9, 2),
	(:c, 8, 3),
	(:e, 11, 2),
	(:c, 7, 3)
]

# ╔═╡ e58a058d-57c8-4eb8-b5dc-0ffb2825a8ae
const essential_vector_bits = length(essential_inds) * 48

# ╔═╡ f4318383-a402-4277-8160-2092f8ad46b3
function update_rubiks_essential_feature!(v::AbstractVector{T}, cube::AbstractVector{I}) where {T<:Real, I <: Integer}
	v .= zero(T)
	@inbounds @simd for i in eachindex(essential_inds)
		j = (i-1)*48
		v[j + cube[essential_inds[i]]] = one(T)
	end
	return v
end

# ╔═╡ c04d8282-2799-4a79-8f33-7dcf4213ae4b
function map_bits((piece_type, piece_number, piece_position)::Tuple{Symbol, Int64, Int64})
	if piece_type == :c
		piece_bits = zeros(Int64, 8)
		position_bits = zeros(Int64, 3)
	else
		piece_bits = zeros(Int64, 12)
		position_bits = zeros(Int64, 2)
	end
	piece_bits[piece_number] = 1
	position_bits[piece_position] = 1
	return BitVector(vcat(piece_bits, position_bits))
end

# ╔═╡ e9d9cfd6-01b7-4d89-9870-57514eb03d22
md"""
## Piece Based Alternative

For this alternative vector I will use only the 20 edge and corner pieces and their respective orientations.  I have already written a function to take the cube represented as a list of 48 indices and return the information about each set of pieces.  This information consists of which piece is in which position and orientation compared to what it would be in a solved cube.  For the edge pieces I only need 12+2 = 14 bits to represent each one with one hot encoding.  With the corner pieces, the number of bits is 8 + 3 = 11.
"""

# ╔═╡ c6482bad-7912-427e-bb19-4245e1818f56
function update_rubiks_piece_vector!(v::AbstractVector{T}, cube::AbstractVector{I}) where {T<:Real, I <: Integer}
	v .= zero(T)
	pieces = calculate_cube_pieces(cube)
	for (i, t) in enumerate(pieces.corner_pieces)
		base_ind = (i-1)*11
		v[base_ind + first(t)] = one(T)
		v[base_ind + 8 + last(t)] = one(T)
	end
	corner_last_ind = 11*8
	for (i, t) in enumerate(pieces.edge_pieces)
		base_ind = (i-1)*14 + corner_last_ind
		v[base_ind + first(t)] = one(T)
		v[base_ind + 12 + last(t)] = one(T)
	end
	return v
end

# ╔═╡ b7586067-6d80-49c4-b8c4-877aa3e2ce3f
function update_rubiks_feature!(x::NonTabularRL.BinaryFeatureVector{I1, 256}, cube::AbstractVector{I2}) where {I1 <: Integer, I2 <: Integer}
	pieces = calculate_cube_pieces(cube)
	ind = 1
	for (i, t) in enumerate(pieces.corner_pieces)
		base_ind = (i-1)*11
		x.active_features[ind] = base_ind + first(t)
		ind += 1
		x.active_features[ind] = base_ind + 8 + last(t)
		ind += 1
	end
	corner_last_ind = 11*8
	for (i, t) in enumerate(pieces.edge_pieces)
		base_ind = (i-1)*14 + corner_last_ind
		x.active_features[ind] = base_ind + first(t)
		ind += 1
		x.active_features[ind] = base_ind + 12 + last(t)
		ind += 1
	end
end

# ╔═╡ d2f662c5-dc52-45c2-a335-5cc7ce2eb486
function get_active_rubiks_piece_features(cube::AbstractVector{I}) where {I <: Integer}
	features = Vector{Int64}()
	pieces = calculate_cube_pieces(cube)
	for (i, t) in enumerate(pieces.corner_pieces)
		base_ind = (i-1)*11
		push!(features, base_ind + first(t))
		push!(features, base_ind + 8 + last(t))
	end
	corner_last_ind = 11*8
	for (i, t) in enumerate(pieces.edge_pieces)
		base_ind = (i-1)*14 + corner_last_ind
		push!(features, base_ind + first(t))
		push!(features, base_ind + 12 + last(t))
	end
	return features
end

# ╔═╡ 5a6a4b92-7b02-441a-9f05-2f7924eec600
function make_rubiks_piece_binary_vector(cube::AbstractVector{I}) where I <: Integer
	v = NonTabularRL.BinaryFeatureVector(14*12 + 11*8)
	v.num_features = 8*2 + 12*2
	v.active_features = get_active_rubiks_piece_features(cube)
	return v
end

# ╔═╡ d949035a-4258-46b2-b749-83397213c379
function initialize_piece_vector() 
	v = NonTabularRL.BinaryFeatureVector(256)
	v.num_features = 8*2 + 12*2
	v.active_features = get_active_rubiks_piece_features(solved_cube_indices)
	return v
end

# ╔═╡ 343aa4fe-f38d-42b7-967b-c589be65077d
md"""
# Traditional Reinforcement Learning Control Techniques
"""

# ╔═╡ 7e57369f-14d5-4a5e-b83d-951228fac2eb
md"""
## Semi-gradient Sarsa
"""

# ╔═╡ f78706e3-2b14-4bc8-9b96-2e4e98a2efa4
md"""
The problem with a technique like Sarsa is that the agent never experiences a solved cube.  Once the task starts, time will be spent exploring scrambled states with no limit to the number of steps.  For using temporal difference methods like this, it may be necessary to modify the task so that an episode ends after a maximum number of time steps regardless of whether the cube is solved or not.  That way many initialization states can be used including those that rae very close to solved.
"""

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
update_rubiks_feature!(v, s::@NamedTuple{cube::Vector{UInt8}, scramble_moves::Int64, move_count::Int64}) = update_rubiks_feature!(v, s.cube)

# ╔═╡ 09538cca-7da3-4d48-9541-90f00acce794
function make_rubiks_feature(cube::AbstractVector{I}) where I <: Integer
	v = zeros(Float32, 48*48)
	update_rubiks_feature!(v, cube)
	return v
end

# ╔═╡ 2040ffd4-f977-4c30-83a0-875b500099aa
const solved_cube_feature = make_rubiks_feature(solved_cube_indices)

# ╔═╡ cbf95b8e-e6e1-4f6b-aec7-af96cf1bb752
update_rubiks_piece_vector!(v::AbstractVector{T}, s::@NamedTuple{cube::Vector{UInt8}, scramble_moves::Int64, move_count::Int64}) where T<:Real = update_rubiks_piece_vector!(v, s.cube)

# ╔═╡ 67f2fe48-ef6e-4af2-9a4c-5af027cffabd
function make_rubiks_piece_vector(cube::AbstractVector{I}) where I <: Integer
	v = zeros(Float32, 14*12 + 11*8)
	update_rubiks_piece_vector!(v, cube)
	return v
end

# ╔═╡ 92b1d07f-6538-44f4-8b94-9be1ee83728a
const solved_cube_piece_feature = make_rubiks_piece_vector(solved_cube_indices)

# ╔═╡ 74d3cc6a-8486-4e01-8aee-a5e4c1b31a73
update_rubiks_essential_feature!(v::AbstractVector{T}, s::@NamedTuple{cube::Vector{UInt8}, scramble_moves::Int64, move_count::Int64}) where T<:Real = update_rubiks_essential_feature!(v, s.cube)

# ╔═╡ 42022ea3-2a4e-4167-bfd1-9bdd2a441d62
function make_rubiks_essential_feature(cube::AbstractVector{I}) where I <: Integer
	v = zeros(Float32, essential_vector_bits)
	update_rubiks_essential_feature!(v, cube)
	return v
end

# ╔═╡ e6cc1a1e-ab00-435f-a0ae-0afc09288023
const solved_cube_essential_feature = make_rubiks_essential_feature(solved_cube_indices)

# ╔═╡ 9e25905f-452d-4ca5-ae9b-4b03284ad03d
function run_dp_rubiks_linear_piece_test(min_moves, max_moves; γ = 0.9f0, num_steps = 10_000, kwargs...)
	mdp =  make_tdcube_mdp(min_moves, max_moves)
	semi_gradient_dp_linear(mdp, γ, typemax(Int64), num_steps, copy(solved_cube_piece_feature), update_rubiks_piece_vector!; suppress_warning=true, kwargs...)
end

# ╔═╡ 7f12e90c-1dd5-49e4-91a1-786e3165c769
md"""
### Non-linear Method
"""

# ╔═╡ dd36dbbd-1807-42d2-ae8e-95ea9ad91069
const sarsa_params_layers = fill(16, 3)

# ╔═╡ 0ac973d8-bd67-499f-a05d-1341fa9ca52b
const sarsa_fcann_params = NonTabularRL.initialize_fcann_params(48*48, sarsa_params_layers, length(rubiks_moves), 1, true)

# ╔═╡ 9e34bffc-e324-414d-a409-ef2cb13d365a
const dp_params_layers = fill(512, 2)

# ╔═╡ 6014ac2d-7d49-483c-92b7-7b1e24466b42
const dp_fcann_params = NonTabularRL.initialize_fcann_params(48*48, dp_params_layers, 1, 1, true)

# ╔═╡ 98e24e54-e285-4c2f-984f-159e315fbdeb
# display_learning_output(test_dp_fcann_output; max_scramble = 8)

# ╔═╡ 91e11772-3cfc-47db-a538-b439b669ccd4
#I want to compare this to an alternative MDP where the reward is designed to produce values from -1.75 to 1.75 for 20 moves away from solution down to 1 move away from solution.  This way it could be an undiscounted task with small negative rewards per step plus a large positive reward on solution.  Failing to solve in the time limit would just leave the negative rewards without anything positive but would still be capped at the double negative value of the range for states that should never be visited under the optimal policy.  Actually under the optimal policy even the "bad" states would be at worst -1.75 because you can't get further away from solution than that so that entire large state space of things that don't progress you to a solution would at worst leave the value flat.  It must be hard to identify a cube though that is in an early stage of being manipulated closer to a solution despite still looking very random vs a move that is redundant and just produces another scrambled cube.  Another option is to experiment with learning the human solution algorithm which is significantly less efficient but may be easier to learn.

# ╔═╡ cd20e905-2355-47cd-82f9-aa576bf6a53f
#I need to add a setup function to create the update test with the feature vector and parameters embedded together so the results are always done and when I go back to running something it uses the existing parameters that I have saved and can also be run in parallel updating the dictionary whenever a new result comes in

# ╔═╡ f0015a0e-b947-4a9b-b734-ddc01871c3c3
const dp_fcann_step_mastery_results = Dict{NamedTuple, NamedTuple}()

# ╔═╡ ed067ef4-0b0c-4f2c-9372-899cfc6449c5
# ╠═╡ disabled = true
#=╠═╡
const dp_mastery_params_layers = fill(512, 4)
  ╠═╡ =#

# ╔═╡ 33ffe90d-941f-4c84-93b5-628bd175140d
#=╠═╡
const dp_mastery_fcann_params = NonTabularRL.initialize_fcann_params(48*48, dp_mastery_params_layers, 1, 1, true)
  ╠═╡ =#

# ╔═╡ 2a0dc867-0b2b-4ad5-9bf9-ce8e1b0d545a
#=╠═╡
const step_mastery_dp_fcann_output = run_dp_step_mastery_fcann_test!(dp_mastery_fcann_params, 2, 3, deepcopy(rubiks_binary_feature), update_rubiks_feature!, 1_000; α = 1f-2, ϵ = 0.01f0)
  ╠═╡ =#

# ╔═╡ 5cafccd6-d525-44e8-a10c-3be0c9ff17ba
#=╠═╡
begin
	added_fcann_mastery_result
	@bind dp_fcann_step_mastery_layer_select Select(collect(keys(dp_fcann_step_mastery_results)), default = (layers = dp_mastery_params_layers,))
end
  ╠═╡ =#

# ╔═╡ 4f7eded4-f49f-4db1-b57a-da7160409199
#=╠═╡
display_learning_output(dp_fcann_step_mastery_results[dp_fcann_step_mastery_layer_select].output; max_scramble = 7)
  ╠═╡ =#

# ╔═╡ 1284f34a-da8b-473e-9e4a-5ef1bf8c1786
#=╠═╡
begin
	dp_fcann_step_mastery_results[(layers = dp_mastery_params_layers,)] = (output = deepcopy(step_mastery_dp_fcann_output), params = deepcopy(dp_mastery_fcann_params))
	added_fcann_mastery_result = true;
end
  ╠═╡ =#

# ╔═╡ 4f244f15-8f07-403c-a2b4-7b2cb9dc7284
function run_dp_rubiks_fcann_test(min_moves, max_moves; γ = 0.99f0, layers = [8, 8], num_steps = 10_000, kwargs...)
	mdp =  make_tdcube_mdp(min_moves, max_moves)
	semi_gradient_dp_fcann(mdp, γ, typemax(Int64), num_steps, update_rubiks_feature!, length(solved_cube_feature), layers; suppress_warning=true, kwargs...)
end

# ╔═╡ 6cab2983-d84f-49dd-b9ff-91933e45667c
function run_dp_rubiks_essential_fcann_test(min_moves, max_moves; γ = 0.9f0, layers = [8, 8], num_steps = 10_000, kwargs...)
	mdp =  make_tdcube_mdp(min_moves, max_moves)
	semi_gradient_dp_fcann(mdp, γ, typemax(Int64), num_steps, update_rubiks_essential_feature!, length(solved_cube_essential_feature), layers; suppress_warning=true, kwargs...)
end

# ╔═╡ 6d7d5e1d-6aa3-49f2-a85b-f34627868351
function run_dp_rubiks_piece_fcann_test(min_moves, max_moves; γ = 0.99f0, layers = [8, 8], num_steps = 10_000, kwargs...)
	mdp =  make_tdcube_mdp(min_moves, max_moves)
	semi_gradient_dp_fcann(mdp, γ, typemax(Int64), num_steps, update_rubiks_piece_vector!, length(solved_cube_piece_feature), layers; suppress_warning=true, kwargs...)
end

# ╔═╡ f13eb4cd-7a45-41c6-b954-6e5f305c461b
function get_hidden_layers(params::FCANNParams)
	β = params.weights[2]
	l = length(β)
	[length(β[x]) for x in 1:l-1]
end

# ╔═╡ 4ddc8f58-fb12-47d7-946a-6601aab2b072
#might need to add the training extension here without yielding a result outside the function.  that way I can extend if there's progress but you wouldn't be able to see that happening in real time

# ╔═╡ 42098e58-5246-4187-a0bd-089a00777793
#Make combined algorithm that trains network at n scramble moves for 1e6 steps.  Looks at the first half and second half of those steps and keeps training until there is no improvement in the second half within numerical noise limits.  Snapshots the parameters and the performance statistics for each scramble and then moves up one scramble step in the training process.  Continues on until some max scramble rate is reached or there is no improvement seen.  Need to figure out how to gracefully interrupt this process while monitoring what it looks like as it is going on.  Look at the MCTS notebook for guidance and check if plutohooks has been updated at all to improve this.

# ╔═╡ 0d1beb1b-d5b0-4ddd-b9ab-71130e5b5f14
#add a function to transfer this model to a higher parameter count by training it on data generated from the other model

# ╔═╡ 78fe374c-60f8-498c-87f8-7666e7e33412
md"""
#### Setup Variables
"""

# ╔═╡ edaf3a59-808c-47dd-a331-d9bb5465d9ac
const layers = [64, 64, 64]

# ╔═╡ c51f5247-732d-4096-9df9-4730cac95f5c
# ╠═╡ disabled = true
#=╠═╡
const dp_params = FCANN.initializeparams_saxe(48*48, layers, 1, 1; use_μP = true)
  ╠═╡ =#

# ╔═╡ 3089e389-0687-4e35-afe9-72a20f5a597b
#=╠═╡
const snapshot_params = deepcopy(dp_params)
  ╠═╡ =#

# ╔═╡ fc70f91d-40ea-4de5-9deb-d5863aabb806
md"""
---
"""

# ╔═╡ 613bd115-02f7-4b4b-b834-1dcad6016788
md"""
#### Background Thread Utility Variables
"""

# ╔═╡ 990be70c-2b10-4766-8f05-d0b535dbde07
test_fcann_dp_output, set_test_fcann_dp_output = @use_state(nothing)

# ╔═╡ 29188eb0-6cad-48dd-8ea3-d1fa66b37d45
test_fcann_dp_start_time, set_test_fcann_dp_start_time = @use_state(nothing)

# ╔═╡ 2d377bf3-f16e-4b55-aea0-7df7478975f6
test_fcann_dp_end_time, set_test_fcann_dp_end_time = @use_state(0.0)

# ╔═╡ b21966df-03b5-49d5-962b-878e5b0a1e2f
test_fcann_dp_curriculum_count, set_test_fcann_dp_curriculum_count = @use_state(0)

# ╔═╡ 1c83cd54-86c8-481d-b7a5-e5d4db60f27a
md"""
#### Curriculum Learning Function
"""

# ╔═╡ 6782e301-16ac-41e3-bd64-fe41dad938de
#=╠═╡
function check_reward_progress(rewards::AbstractVector{T}) where T<:Real
	l = length(rewards)
	b = floor(Int64, l/2)
	v1 = mean(view(rewards, 1:b))
	v2 = mean(view(rewards, b+1:l))

	isapprox(v1, v2) && return false
	v1 > v2 && return false
	v1 >= v2 && return true
	return true
end
  ╠═╡ =#

# ╔═╡ 2ee24bf5-f148-45ef-b2f4-ba8aa0bf96be
# ╠═╡ disabled = true
#=╠═╡
@use_effect([]) do
	@spawn begin 
		run_rubiks_dp_curriculum_eval_loop(7, 7; num_steps = 10_000, α = 1f-2, layers = layers, reslayers = 1, parameters = snapshot_params, ϵ = 0.01f0, γ = 0.9f0)
	end
end
  ╠═╡ =#

# ╔═╡ c29c507f-0c07-424e-94db-796c36c09143
md"""
---
"""

# ╔═╡ 8888a405-0200-49f2-8b02-ea61d7089629
if test_fcann_dp_start_time > test_fcann_dp_end_time
	md"""
	##### Process still running on iteration number $test_fcann_dp_curriculum_count
	"""
else
	md"""
	##### Reward progress stalled at on iteration number $test_fcann_dp_curriculum_count.  Ready to run new task.
	"""
end

# ╔═╡ f4435085-6e27-4acc-bfa2-2fe16b232ced
#=╠═╡
if isnothing(test_fcann_dp_start_time)
	md"""
	##### Waiting to run process
	"""
elseif test_fcann_dp_start_time > test_fcann_dp_end_time
	md"""
	##### Currently running task starting at $(unix2datetime(test_fcann_dp_start_time)) and displaying previous results completed at $(unix2datetime(test_fcann_dp_end_time))
	"""
else
	md"""
	##### Displaying most recent results completed at $(unix2datetime(test_fcann_dp_end_time)) in $(test_fcann_dp_end_time - test_fcann_dp_start_time) seconds.
	"""
end
  ╠═╡ =#

# ╔═╡ 917e90dc-0cea-4119-8779-ed8b15b1a73b
#5 -> .94
#6 -> .95
#7 -> 0.86

# ╔═╡ 6aa509fa-a762-4571-8d88-360959cbb670
show_rubiks_policy_eval(s -> test_fcann_dp_output.value_function((cube=s, scramble_moves = 30, move_count = 0)).maximizing_action, 1_000; max_scramble = 15)

# ╔═╡ 3c81903a-fb36-48e7-9790-c547a297b092
const layers2 = [256, 256, 256, 256, 256]

# ╔═╡ 7b0b3725-09dd-4f1f-91e3-32d157241bda
# ╠═╡ disabled = true
#=╠═╡
const dp_params2 = FCANN.initializeparams_saxe(48*48, layers2, 1, 1; use_μP = true)
  ╠═╡ =#

# ╔═╡ c0193517-5a19-4e09-9a9e-e8dac957ae74
#add a button here so every time you click to train again it saves a snapshot of the parameters and the performance histogram of how well it did for every scramble level

# ╔═╡ 803bf4fa-de6c-421e-93a4-a001f480a036
test_fcann_dp_output2, set_test_fcann_dp_output2 = @use_state(nothing)

# ╔═╡ 88a45407-9b12-412e-8c8d-4c9437d2d50d
test_fcann_dp_start_time2, set_test_fcann_dp_start_time2 = @use_state(nothing)

# ╔═╡ 4c0482a1-0320-45d3-9246-8c38bd1f3a05
test_fcann_dp_end_time2, set_test_fcann_dp_end_time2 = @use_state(0.0)

# ╔═╡ 5a0a75e8-6478-4c1e-ac7a-68d30481aa4a
# ╠═╡ disabled = true
#=╠═╡
@use_effect([]) do
	@spawn begin
		set_test_fcann_dp_start_time2(time())
		output = run_dp_rubiks_fcann_test(9, 9; num_steps = 10_000, α = 1f-2, layers = layers2, parameters = dp_params2, ϵ = 0.01f0, γ = 0.9f0, reslayers = 1)
		set_test_fcann_dp_output2(output)
		set_test_fcann_dp_end_time2(time())
	end
end
  ╠═╡ =#

# ╔═╡ be9fce6d-2fcf-493f-9634-0da89160866c
# 5 -> 0.988
# 6 -> 0.98
# 7 -> 0.867
# 8 -> 0.81
# 9 -> 0.68

# ╔═╡ 9a54a1ba-8e51-42c2-a44b-ab669f7533d4
#=╠═╡
begin
function create_spawn_message(start_time::Nothing, end_time::Real)
	md"""
	##### Waiting to run process
	"""
end
	
function create_spawn_message(start_time::Real, end_time::Nothing)
	md"""
	##### Currently running task started at $(unix2datetime(start_time)) and waiting for first result
	"""
end

function create_spawn_message(start_time::Real, end_time::Real)
	if start_time > end_time
	md"""
	##### Currently running task started at $(unix2datetime(start_time)) and displaying most recent result completed at $(unix2datetime(end_time))
	"""
	else
	md"""
	##### Displaying most recent result completed at $(unix2datetime(end_time)) in $(end_time - start_time) seconds.  Ready to start a new task
		"""
	end
end
end
  ╠═╡ =#

# ╔═╡ 2d82c02a-583a-4509-b396-cb97d1ebc67c
#=╠═╡
create_spawn_message(test_fcann_dp_start_time2, test_fcann_dp_end_time2)
  ╠═╡ =#

# ╔═╡ c6e67132-a618-4371-bb74-00a9f34d16f0
show_rubiks_policy_eval(s -> test_fcann_dp_output2.value_function((cube=s, scramble_moves = 30, move_count = 0)).maximizing_action, 100; max_scramble = 15)

# ╔═╡ 19933037-d49e-44cb-af5e-80373c3e7b29
test_fcann_dp_output2

# ╔═╡ 73d36792-1b08-44e5-9fbd-1d9fc3d29127
# ╠═╡ disabled = true
#=╠═╡
const layers3 = fill(256, 5)
  ╠═╡ =#

# ╔═╡ 51dece7f-4656-4b58-a7be-3ba2617c82d2
# ╠═╡ disabled = true
#=╠═╡
const layers_essential = [256, 256, 256]
  ╠═╡ =#

# ╔═╡ c313a8c7-8c49-4f54-bf25-3ec964ef1834
# ╠═╡ disabled = true
#=╠═╡
const layer3_piece = fill(256, 5)
  ╠═╡ =#

# ╔═╡ 02a0c6b5-73c1-43b8-b43d-ae073b74bd3f
#=╠═╡
const dp_params3 = FCANN.initializeparams_saxe(48*48, layers3, 1, 1; use_μP = true)
  ╠═╡ =#

# ╔═╡ a9423fb3-288a-4b44-92a3-41a246c882df
#=╠═╡
const dp_essential_params = FCANN.initializeparams_saxe(essential_vector_bits, layers_essential, 1, 1; use_μP = true)
  ╠═╡ =#

# ╔═╡ 53b51ce8-4370-422e-b018-279a2da6185d
#=╠═╡
const dp_piece_params3 = FCANN.initializeparams_saxe(256, layer3_piece, 1, 1; use_μP = true)
  ╠═╡ =#

# ╔═╡ a59a94ed-abd7-45df-9045-4502ad09064d
const curriculum_results = Dict{Vector{Int64}, NamedTuple}()

# ╔═╡ d1d14f85-2ce1-4b3f-970b-5ef4ad38ee83
test_fcann_dp_output3, set_test_fcann_dp_output3 = @use_state(nothing)

# ╔═╡ 3e3b7280-a3af-4a4c-9a98-21cbb22fef0a
test_fcann_dp_piece_output3, set_test_fcann_dp_piece_output3 = @use_state(nothing)

# ╔═╡ 155ceb16-d02b-42ea-8d6e-15d3133e577e
test_fcann_dp_essential_output, set_test_fcann_dp_essential_output = @use_state(nothing)

# ╔═╡ edc3c704-cc27-4089-bed5-44a93c454ce8
test_fcann_dp_start_time3, set_test_fcann_dp_start_time3 = @use_state(nothing)

# ╔═╡ f1bc4018-642a-46d7-b818-07eb3c26898c
test_fcann_dp_piece_start_time3, set_test_fcann_dp_piece_start_time3 = @use_state(nothing)

# ╔═╡ d4814bad-0fb7-4399-8059-ca31d8329db6
test_fcann_dp_essential_start_time, set_test_fcann_dp_essential_start_time = @use_state(nothing)

# ╔═╡ a1b140e8-a379-4a36-a109-bbe126b4edbd
test_fcann_dp_end_time3, set_test_fcann_dp_end_time3 = @use_state(0.0)

# ╔═╡ 699884e7-bc2b-44a1-8e04-dcd2001009c7
test_fcann_dp_essential_end_time, set_test_fcann_dp_essential_end_time = @use_state(0.0)

# ╔═╡ 5a1cd4ba-e5c3-40b1-a137-eef33b6bc721
test_fcann_dp_piece_end_time3, set_test_fcann_dp_piece_end_time3 = @use_state(0.0)

# ╔═╡ 67bcffbb-7107-41ba-89bc-0fddedb6eb0c
# ╠═╡ disabled = true
#=╠═╡
@use_effect([]) do
	@spawn begin
		set_test_fcann_dp_start_time3(time())
		# output = run_dp_rubiks_piece_fcann_test(4, 4; num_steps = 10_000_000, α = 1f-2, layers = layers3, parameters = dp_params3, ϵ = 0.01f0, γ = 0.9f0, reslayers = 0)
		# output = run_dp_step_mastery_fcann_test!(dp_params3, 5, make_rubiks_piece_vector(solved_cube_indices), update_rubiks_piece_vector!, 1_000_000; α = 1f-2, ϵ = 0.01f0, γ = 0.9f0, reslayers = 0)
		output = run_dp_curriculum_mastery_fcann_test!(dp_params3, 3, 8, make_rubiks_feature(solved_cube_indices), update_rubiks_feature!, 10_000, [1f-2]; ϵ = 0.01f0, γ = 0.9f0, reslayers = 1)
		set_test_fcann_dp_output3(output)
		set_test_fcann_dp_end_time3(time())
	end
end
  ╠═╡ =#

# ╔═╡ fd3f1da9-dbd8-4885-84bd-5f5fcb076a34
#=╠═╡
if isnothing(test_fcann_dp_start_time3)
	md"""
	##### Waiting to run process
	"""
elseif test_fcann_dp_start_time3 > test_fcann_dp_end_time3
	md"""
	##### Currently running task starting at $(unix2datetime(test_fcann_dp_start_time3)) and displaying previous results completed at $(unix2datetime(test_fcann_dp_end_time3))
	"""
else
	md"""
	##### Displaying most recent results completed at $(unix2datetime(test_fcann_dp_end_time3)) in $(test_fcann_dp_end_time3 - test_fcann_dp_start_time3) seconds.  Ready to start a new task.
	"""
end
  ╠═╡ =#

# ╔═╡ 72204679-8fdd-49a0-8fc2-24bf2f0e83ac
curriculum_results[get_hidden_layers(test_fcann_dp_output3.final_results.final_parameters)] = test_fcann_dp_output3

# ╔═╡ fb106a66-d9ea-4195-8c0e-4246205a794e
md"""
#### Select Results to View
"""

# ╔═╡ 299b2603-bc86-4803-b136-865faf0571e1
md"""
Select Architecture
"""

# ╔═╡ df8688bb-8cc7-4737-9ef4-a356c4d2de80
#=╠═╡
begin
	test_fcann_dp_output3
	@bind layer_select Select(collect(keys(curriculum_results)))
end
  ╠═╡ =#

# ╔═╡ 76be0781-2406-4679-8b3e-701fdbf84944
md"""
Select Scramble Moves
"""

# ╔═╡ bfbd5284-34f7-4346-ac5c-857bc75eb035
#=╠═╡
@bind move_select Select(collect(keys(curriculum_results[layer_select].dict_results)) |> sort)
  ╠═╡ =#

# ╔═╡ 2514edf2-3960-4c1b-a47a-13400e5edad8
md"""Select Learning Rate
"""

# ╔═╡ c05959f2-36d9-4de6-8a15-e21cc3cc610d
#=╠═╡
@bind α_select Select(collect(keys(curriculum_results[layer_select].dict_results[move_select])) |> v -> sort(v; rev=true))
  ╠═╡ =#

# ╔═╡ 37a6d563-f943-4bcc-8b0e-53af130a2bec
#=╠═╡
show_rubiks_policy_eval(s -> curriculum_results[layer_select].final_results.value_function((cube=s, scramble_moves = 30, move_count = 0)).maximizing_action, 100; max_scramble = 15)
  ╠═╡ =#

# ╔═╡ d513c0b8-6699-4cdd-8481-7adc0735b669
md"""
---
"""

# ╔═╡ 803182af-ac7b-4089-a4b0-b834856f725e
# ╠═╡ disabled = true
#=╠═╡
@use_effect([]) do
	@spawn begin
		set_test_fcann_dp_piece_start_time3(time())
		output = run_dp_rubiks_piece_fcann_test(7, 7; num_steps = 10_000, α = 1f-2, layers = layer3_piece, parameters = dp_piece_params3, ϵ = 0.01f0, γ = 0.9f0, reslayers = 1)
		# output = run_dp_rubiks_piece_fcann_test(1, 5; num_steps = 100_000, α = 8f-3, layers = layers3, ϵ = 0.01f0, γ = 0.9f0)

		set_test_fcann_dp_piece_output3(output)
		set_test_fcann_dp_piece_end_time3(time())
	end
end
  ╠═╡ =#

# ╔═╡ cdaac960-ffcf-43fb-b0e6-1f5a9be03252
#=╠═╡
if isnothing(test_fcann_dp_piece_start_time3)
	md"""
	##### Waiting to run process
	"""
elseif test_fcann_dp_piece_start_time3 > test_fcann_dp_piece_end_time3
	md"""
	##### Currently running task starting at $(unix2datetime(test_fcann_dp_piece_start_time3)) and displaying previous results completed at $(unix2datetime(test_fcann_dp_piece_end_time3))
	"""
else
	md"""
	##### Displaying most recent results completed at $(unix2datetime(test_fcann_dp_piece_end_time3)) in $(test_fcann_dp_piece_end_time3 - test_fcann_dp_piece_start_time3) seconds.  Ready to start a new task.
	"""
end
  ╠═╡ =#

# ╔═╡ f51df2ad-2c72-436d-81ca-a922dc41b1b8
#5 -> .94
#6 -> .9
#7 -> .75

# ╔═╡ 687c56a6-b87a-43ab-badd-d4cd7eddebcf
show_rubiks_policy_eval(s -> test_fcann_dp_piece_output3.value_function((cube=s, scramble_moves = 30, move_count = 0)).maximizing_action, 100; max_scramble = 15)

# ╔═╡ f68e5661-ebcb-46b8-8fe2-f35779b8deeb
md"""
#### Essential Indices with 48 Bit Representation
"""

# ╔═╡ 04bababf-cebe-4d37-940f-f17ffc9c8f36
@use_effect([]) do
	@spawn begin
		set_test_fcann_dp_essential_start_time(time())
		output = run_dp_rubiks_essential_fcann_test(6, 6; num_steps = 10_000, α = 1f-2, layers = layers_essential, parameters = dp_essential_params, ϵ = 0.01f0, reslayers = 1)
		set_test_fcann_dp_essential_output(output)
		set_test_fcann_dp_essential_end_time(time())
	end
end

# ╔═╡ 18061035-b795-4df9-aa9d-a0a32b7c6598
#=╠═╡
if isnothing(test_fcann_dp_essential_start_time)
	md"""
	##### Waiting to run process
	"""
elseif test_fcann_dp_essential_start_time > test_fcann_dp_essential_end_time
	md"""
	##### Currently running task starting at $(unix2datetime(test_fcann_dp_essential_start_time)) and displaying previous results completed at $(unix2datetime(test_fcann_dp_essential_end_time))
	"""
else
	md"""
	##### Displaying most recent results completed at $(unix2datetime(test_fcann_dp_essential_end_time)) in $(test_fcann_dp_essential_end_time - test_fcann_dp_essential_start_time) seconds.  Ready to start a new task.
	"""
end
  ╠═╡ =#

# ╔═╡ 1cea3b3e-8bba-496d-aaf3-de19eadab8b4
#5 -> .97
#6 -> 0.86

# ╔═╡ b2e54677-184f-460f-bd90-07fa513a5d56
show_rubiks_policy_eval(s -> test_fcann_dp_essential_output.value_function((cube=s, scramble_moves = 30, move_count = 0)).maximizing_action, 100; max_scramble = 15)

# ╔═╡ c9bf5811-6a61-45f4-9070-6f2b7277611f
md"""
# MCTS Planning Control Method 
"""

# ╔═╡ 8ca56795-bdc9-4740-8da7-9506d9063fe7
const rubiks_mcts_mdp = make_tdcube_mdp(10, 10)

# ╔═╡ b5feae0f-7196-445a-8a3f-b1a52a62b2de
const rubiks_mcts_s0 = rubiks_mcts_mdp.initialize_state()

# ╔═╡ ff5f38a5-f7f7-4417-87b6-f3d4b4591fad
#make a graph showing the degree to which the MCTS search improves the policy as a function of number of simulations, depth and other factors

# ╔═╡ 5e804f3b-a7a2-450a-8ff8-e56fffe662e1
test_fcann_dp_output

# ╔═╡ 0c0115d2-ecd7-4947-9086-9f8759e568bb
function make_fast_value_function(dp_output, params::Vector{T}) where T<:Real
	x = zeros(T, length(params))
	function v̂(s)
		dp_output.value_function(s; x = x).maximizing_value
	end
	return v̂
end

# ╔═╡ d2454f2a-c61d-4a76-8c6f-2234afefa0e4
function make_fast_value_function(dp_output, params::FCANNParams{T}) where T<:Real
	x = zeros(T, size(params[1][1], 2))
	function v̂(s)
		dp_output.value_function(s; x = x).maximizing_value
	end
	return v̂
end

# ╔═╡ 09c5e213-f175-4032-a5e5-dbd9f7b0d753
function calculate_mcts_improvement(min_scramble, max_scramble, dp_output; nsamples = 100, mcts_kwargs...)
	# vanilla_score = 0f0
	# mcts_score = 0f0
	# π = make_threadsafe_policy(dp_output)
	v̂ = make_fast_value_function(dp_output, dp_output.final_parameters)
	# v̂(s) = dp_output.value_function(s).maximizing_value
	(vanilla_score, mcts_score) = 1:nsamples |> Map() do _
		mdp = make_tdcube_mdp(min_scramble, max_scramble)
		
		s0 = mdp.initialize_state()
		while mdp.isterm(s0) #ensures that we don't begin an episode at a terminal state
			s0 = mdp.initialize_state()
		end
		vanilla_score = runepisode(mdp; π = s -> dp_output.value_function(s).maximizing_action, s0 = (cube = s0.cube, scramble_moves = 100, move_count = 0))[3][end]
		# vanilla_score = runepisode(mdp; π = π, s0 = (cube = s0.cube, scramble_moves = 100, move_count = 0))[3][end]
		mcts_score = runepisode(mdp; π = s -> monte_carlo_tree_search(mdp, 0.9f0, (mdp, s, γ) -> v̂(s), s; mcts_kwargs...)[1], s0 = (cube = s0.cube, scramble_moves = 100, move_count = 0))[3][end]
		# mcts_score = runepisode(mdp; π = s -> monte_carlo_tree_search(mdp, 0.9f0, (mdp, s, γ) -> π(s), s; mcts_kwargs...)[1], s0 = (cube = s0.cube, scramble_moves = 100, move_count = 0))[3][end]
		(vanilla_score, mcts_score)
	end |> foldxl((a, b) -> (a[1] + b[1], a[2] + b[2]))
	(vanilla_success_rate = vanilla_score / nsamples, mcts_success_rate = mcts_score / nsamples)
end

# ╔═╡ 831c8323-576e-4826-a868-514d21804986
#[64, 64, 64] Network Results
#-------10 Scramble Moves--------------
#nsims = 100, depth = 10, c = 1 -> 0.75
#nsims = 100, depth = 5, c = 1 -> 0.735 
#nsims = 100, depth = 2, c = 1 -> 0.728 
#nsims = 1000, depth = 10, c = 1 -> 0.807 success rate
#nsims = 1000, depth = 8, c = 1 -> 0.792 success rate
#nsims = 1000, depth = 20, c = 1 -> 0.785 success rate
#nsims = 1000, depth = 5, c = 1 -> 0.789 success rate

#------8 Scramble Moves--------------
#vanilla success rate ~ 0.87
#nsims = 100, depth = 10, c = 1 -> 0.926
#nsims = 1000, depth = 10, c = 1 -> 0.962

#------9 Scramble Moves--------------
#vanilla success rate ~ 0.76
#nsims = 1000, depth = 10, c = 1 -> .907 

#------12 Scramble Moves--------------
#vanilla success rate ~ 0.373
#nsims = 1000, depth = 10, c = 1 -> .581 

#------11 Scramble Moves--------------
#vanilla success rate ~ 0.49
#nsims = 1_000, depth = 10, c = 1 -> 705
#nsims = 10_000, depth = 10, c = 1 -> .735 

#------20 Scramble Moves--------------
#vanilla success rate ~ 0.012
#nsims = 1_000, depth = 10, c = 1 -> .043

# ╔═╡ 378a59e9-61d7-49e5-8d3f-0613c3632766
#so I can take an existing value function and make use of mcts to produce a much better policy, but then how can I use this to improve the value function?  One option is to make a dataset with the mcts search of true values and use that to train the value function in a monte carlo fashion.  The other option would be to continue with a kind of td training which would be a different dataset in which I would use the intermediate value update instead of the monte carlo target.  I guess I could train using both of these methods with a large dataset and then decide compare the greedy policy generated from both to see which method is better.

# ╔═╡ 61991d31-551a-42cc-a2ca-ce9a446f5952
#could turn this into a dp method where instead of the one step lookahead you do a full MCTS evaluation for each learning step using the parameter vector at the time and then use the selected action value as the target value.  It would be very slow though depending on how many iterations are done

# ╔═╡ 89de5822-ab28-4153-a244-798e6fcce1c5
#another idea is to generate episodes backwards and then train a value function from those trajectories.  Would have to be a custom function though where I reverse the episode information but I could also use the -1 per step reward since I would have a finite cap on this.  Maybe up to 30 scrambles.  I could see how easily the thing works for uhh accuracy.  This would be purely policy estimation though assuming this actulaly makes sense as a way to seed a strategy

# ╔═╡ 289bd333-b459-45e9-a427-863bb8fe709c
test_fcann_dp_output3

# ╔═╡ 95dec114-a1a4-4e51-9a9e-7a46d3f9928e
# ╠═╡ disabled = true
#=╠═╡
calculate_mcts_improvement(10, 10, curriculum_results[[128, 128, 128, 128, 128]].final_results; nsamples = 100, nsims = 1000, depth = 10)
  ╠═╡ =#

# ╔═╡ 68e6a2fd-bdaa-4973-89b3-d5eff791a39b


# ╔═╡ 02b73ed6-04e0-45a3-b071-d9ae4c347d78
#7 scramble, 100 sims, 0.76 -> 0.94
#7 scramble, 1000 sims, 0.65 -> 98
#8 scramble, 0.58 -> .91
#10 scramble, 100 sims, 0.57 -> 0.77
#10 scramble, 1000 sims, 0.58 -> 0.84

# ╔═╡ e8b2e0bb-2f96-4ffb-8be8-066090982ad2
# ╠═╡ disabled = true
#=╠═╡
function π_dp3_threadsafe(s)
	out = test_fcann_dp_output3.value_function(s)
	out.maximizing_action
end
  ╠═╡ =#

# ╔═╡ e8e05678-c0fa-46bf-bed5-96eb00dcf3f6
function make_threadsafe_policy(dp_output)
	function π(s)
		out = dp_output.value_function(s)
		out.maximizing_action
	end
	return π
end

# ╔═╡ 54149d2e-06ee-4a5c-8c55-fcd0cd1442b5
# ╠═╡ disabled = true
#=╠═╡
function make_threadunsafe_value_function(dp_output, update_feature_vector!::Function, reslayers::Integer)
	activations = FCANN.form_activations(dp_output.final_parameters[1])
	l = size(dp_output.final_parameters[1][1], 2)
	feature_vector = zeros(Float32, l)
	action_values = zeros(Float32, 12)
	function v̂(s)
		update_rubiks_feature!(feature_vector, s.cube)
		NonTabularRL.fcann_value_function!(activations, feature_vector, dp_output.final_parameters, reslayers)
		return first(last(activations))
	end
	return v̂
end
  ╠═╡ =#

# ╔═╡ ab2affbb-fe34-49d8-b63d-fab610b62f14
#=╠═╡
dp3_threadunsafe_value = make_threadunsafe_value_function(test_fcann_dp_output3, update_rubiks_feature!, 1)
  ╠═╡ =#

# ╔═╡ a815fae0-0e15-4bda-b1a9-34d773848c81
# ╠═╡ disabled = true
#=╠═╡
runepisode(rubiks_mcts_mdp; π = s -> monte_carlo_tree_search(rubiks_mcts_mdp, 0.9f0, (mdp, s, γ) -> dp3_threadunsafe_value(s), s, nsims = 100, depth = 2)[1], s0 = (cube = rubiks_mcts_s0.cube, scramble_moves = 100, move_count = 0))
  ╠═╡ =#

# ╔═╡ dc8bfe16-e8e9-463d-8e2d-214e1b018d32
# ╠═╡ disabled = true
#=╠═╡
runepisode(rubiks_mcts_mdp; π = s -> monte_carlo_tree_search(rubiks_mcts_mdp, 0.9f0, (mdp, s, γ) -> test_fcann_dp_output3.value_function(s).maximizing_value, s, nsims = 100, depth = 2)[1], s0 = (cube = rubiks_mcts_s0.cube, scramble_moves = 100, move_count = 0))
  ╠═╡ =#

# ╔═╡ 989c1456-dd26-4497-b8d3-91a18c474370
# ╠═╡ disabled = true
#=╠═╡
mcts_episode = runepisode(rubiks_mcts_mdp; π = s -> monte_carlo_tree_search(rubiks_mcts_mdp, 0.9f0, (mdp, s, γ) -> test_fcann_dp_output3.value_function(s).maximizing_action, s, nsims = 1_000, depth = 6)[1], s0 = (cube = rubiks_mcts_s0.cube, scramble_moves = 100, move_count = 0))
  ╠═╡ =#

# ╔═╡ 27c9d6ba-3f37-45a7-9039-d256cdda3958
#=╠═╡
(render_cube(mcts_episode[1][1].cube), render_cube(mcts_episode[1][end].cube))
  ╠═╡ =#

# ╔═╡ 3e12ae05-271e-4f31-baf8-6e926460f12a
# ╠═╡ disabled = true
#=╠═╡
runepisode(rubiks_mcts_mdp; π = s -> test_fcann_dp_output.value_function(s).maximizing_action, s0 = (cube = rubiks_mcts_s0.cube, scramble_moves = 100, move_count = 0))
  ╠═╡ =#

# ╔═╡ 4c53d04e-df52-4960-ba46-ed3bff1a624b
function π_dist_rubiks_uniform!(prior, s)
	prior .= 1f0/12
	return 1
end

# ╔═╡ fca1535c-20f3-450e-95b3-0130e8a49b48
md"""
# Tabular Version of More Limited Problem Space

We cannot address the Rubik's cube as a tabular problem because the state space is too large, about 43 quintillion.  If we instead consider cubes that are closer to the solved state, we can shrink the state space by a lot.  There is one unique solved cube and 12 atomic moves we can make to produce 12 unique cubes that are one step away from being solved.  If we continue to branch out the state space with new cubes we multiply the state space by 12 each new step as an upper bound.  It is likely that as we continue to rotate cubes, there will be repeats, so the true scaling is likely to be less than that.

So let's say we wish to consider cubes that are at most N steps from being solved.  We can recursively generate all such cubes by branching all possible moves from the solved cube N times and only including unique states.  In order for this to be a valid MDP though, we have to deal with the fact that the true state space is larger.  We also only care about optimal solutions and we know that any cube not in this list is further away from being solved than N.  

The goal of constructing such a tabular problem would be to get an exact solution for the minimum number of steps required to solve any cube in the state space.  If we provide a reward of -1 per step, then the value function will give us exactly that assessment.  Any cube in the state space will have a value of at worst -N, so if we provide a reward for exiting the state space of -N, then we would still be guaranteed to find the optimal solution since it would always be worse to move outside of the state space.

## State Space

##### All cubes at most N moves away from solution

## Reward Function

##### -1 per step (unless exiting state space)

The value function will then equal the minimum number of steps required to solve the cube

## Terminal States

##### Unique solved cube

##### Any cube outside the state space

Any move that would produce a cube that doesn't exist in the state space must not be part of the optimal policy.  Such moves will be punished with a reward of -N to ensure the value estimate is lower than any valid move.
"""

# ╔═╡ 7464b10d-056b-47cc-8db8-44ace4b84d11
const nmove_lookup = Dict{Int64, Set{SVector{48, UInt8}}}();

# ╔═╡ 4017ed01-5cbb-47b6-bc29-d7a30b03a0c0
md"""
Once the state list is constructed, we must build the state transition map for the deterministic problem.  This map is a matrix which contains the index of the transition state for all state action pairs.  The reward matrix is simple in that most values are -1 except for transitions that would be terminal.
"""

# ╔═╡ a10c5406-4d4e-45f6-9fb7-b4b5702abf07
function build_lookup(statelist)
	l = length(statelist)
	dict = Dict{eltype(statelist), Int64}()
	sizehint!(dict, l)
	for (i, s) in enumerate(statelist)
		dict[s] = i
	end
	return dict
end

# ╔═╡ 0d62e009-81be-43d3-add5-ce4ac591dbc7
# ╠═╡ disabled = true
#=╠═╡
const rubiks_tabular_mdp = build_tabular_rubiks_mdp(7)
  ╠═╡ =#

# ╔═╡ dee2e1c5-8423-47b3-868d-483b871da731
#=╠═╡
#this is the terminal state index and it appears in the transition map every time an invalid move occurs
findall(rubiks_tabular_mdp.terminal_states)
  ╠═╡ =#

# ╔═╡ 50f873e8-f59d-40f3-8adf-600ddf4b6e1a
md"""
We can also calculate a benchmark of how much we need to scramble a cube before it becomes completely out of reach of the 7 step solution.  We see here that after 10 scramble moves about 24% of these cubes are still solvable within 7 moves, but with 14 scramble moves that drops to 4%.
"""

# ╔═╡ 92c365b7-924d-4ec6-978d-e743d0237cfc
#=╠═╡
function compute_scramble_statistic(scramble::Integer; nsamples = 100_000)
	1:nsamples |> Map() do i
		cube = initialize_rubiks_cube(scramble)
		haskey(rubiks_tabular_mdp.state_index, cube)
	end |> foldxt(+) |> x -> x / nsamples
end
  ╠═╡ =#

# ╔═╡ c079e451-0487-4e02-8f0e-b61e8963eeb2
#=╠═╡
[(scramble_moves = n, percent_solvable = compute_scramble_statistic(n)) for n in vcat(6:20, [30, 40])] |> DataFrame
  ╠═╡ =#

# ╔═╡ 6e9091e1-afaa-46cf-8e21-6b38526640cb
md"""
## Value Iteration Solution
Once we set up the MDP, the value iteration function can easily solve it as an undiscounted problem
"""

# ╔═╡ 1938becf-e2cc-4c7b-b6ec-d1c3ff107ed9
#=╠═╡
const rubiks_value_iteration = value_iteration_v(rubiks_tabular_mdp, 1f0; usethreads=true, make_final_policy = TabularRL.make_greedy_bit_policy)
  ╠═╡ =#

# ╔═╡ d7c391b1-0d13-409b-a0ac-c02fb766b839
#=╠═╡
const rubiks_tabular_policy_lookup = [findfirst(!iszero, a) for a in eachcol(rubiks_value_iteration.optimal_policy)]
  ╠═╡ =#

# ╔═╡ 7f45ed2c-996b-40dc-b2cd-4f6cc7bddc4c
md"""
We can now query the solution for any cube in the state space and see how many steps away it is from a solution as well as the solution trajectory.
"""

# ╔═╡ bcb9aec3-2d1b-4361-8fc3-6a2cfa20ddb3
#=╠═╡
(initial_cube = render_cube(tabular_eval_cube), value = rubiks_value_iteration.final_value[rubiks_tabular_mdp.state_index[tabular_eval_cube]])
  ╠═╡ =#

# ╔═╡ afe35fcb-44f3-4deb-b1f6-8820b74679c1
#=╠═╡
const test_tabular_episode = runepisode(rubiks_tabular_mdp; π = rubiks_value_iteration.optimal_policy, i_s0 = rubiks_tabular_mdp.state_index[tabular_eval_cube])
  ╠═╡ =#

# ╔═╡ a487ccf0-c293-4093-b936-f92094e86fa7
#=╠═╡
const value_lookup = DataFrame(state = rubiks_tabular_mdp.states, value = rubiks_value_iteration.final_value) |> df -> groupby(df, :value)
  ╠═╡ =#

# ╔═╡ f26935db-a80a-45d7-b2cb-23febbfc3d15
#=╠═╡
[(key = k.value, num_states = size(value_lookup[k], 1)) for k in keys(value_lookup)] |> DataFrame
  ╠═╡ =#

# ╔═╡ 84a08fd1-f8b5-4ca8-b0c0-2c306044d694
#=╠═╡
[size(value_lookup[(value = k,)], 1) / size(value_lookup[(value = k+1,)], 1) for k in -6:-1]
  ╠═╡ =#

# ╔═╡ 9e1d861d-1f5a-40c8-ac24-13c2b35cd90c
md"""
## 2x2x2 Cube Exact Solution

This time I can consider the 24 unique facelets of the pocket cube and only 3 moves instead of 12.  That's because there is a relationship between the opposite faces such that if you rotate a face you produce the same cube as rotating the opposite face in the same direction.  So we should be able to produce all of the cube states by just looking at the front, top, and right faces and only doing clockwise rotations.
"""

# ╔═╡ 4409a59f-e138-4c5b-910c-828bf8bd4497
const solved_pocket_cube = SVector{24}(UInt8.(1:24))

# ╔═╡ 47ebe08c-5234-42bf-9750-1fca5bb63e99
#each column represents a face of the cube
const solved_pocket_cube_values = mapreduce(i -> fill(i, 4), hcat, square_values)

# ╔═╡ ccd01889-2619-4518-b7e5-aa2768cde226
const pocket_moves = [RubiksMove{F, Clockwise}() for F in face_order[1:3]]

# ╔═╡ 62dd2f4f-4ecb-4152-bb9e-7a6fb75222d9
const pocket_move_index = TabularRL.makelookup(pocket_moves)

# ╔═╡ dda10443-ec98-45a7-a935-86e65d608dd7
const pocket_clockwise_perm = SVector{4, UInt8}([3, 1, 4, 2])

# ╔═╡ 9b8f7aeb-d009-444a-9fea-b54677c30197
const pocket_clockwise_rotation_mapping = Dict([
	Front => [(2, (3, 4)), (3, (1, 3)), (6, (2, 1)), (5, (4, 2))],
	Top => [(1, (1, 2)), (5, (1, 2)), (4, (1, 2)), (3, (1, 2))],
	Right => [(1, (2, 4)), (2, (2, 4)), (4, (3, 1)), (6, (2, 4))], 
	])

# ╔═╡ bd55c435-a481-47cb-a032-f8ff9432f630
# ╠═╡ disabled = true
#=╠═╡
function build_pocket_list()
	explored_statelist = Set{SVector{24, UInt8}}()
	unexplored_statelist = [Set([solved_pocket_cube])]
	change = 1
	round = 1
	while change > 0
		l1 = length(explored_statelist)
		# @info "On round $round, have $l1 explored states and $(length(unexplored_statelist[round])) new to explore"
		new_unexplored = Set{SVector{24, UInt8}}()
		for s in unexplored_statelist[round]
				for i_a in 1:3
					s′ = rotate_pocket_cube(s, i_a)
					!in(s′, explored_statelist) && push!(new_unexplored, s′)
				end
			push!(explored_statelist, s)
		end
		push!(unexplored_statelist, new_unexplored)
		l2 = length(explored_statelist)
		change = l2 - l1
		round += 1
	end
	@info "Terminated search on round $(round - 1) after finding $(length(explored_statelist)) states"
	return explored_statelist
end	
  ╠═╡ =#

# ╔═╡ ed426ccc-2b42-4610-9eeb-f342907861d3
#=╠═╡
const pocket_states = collect(build_pocket_list())
  ╠═╡ =#

# ╔═╡ ce6556ef-a69d-48a4-8eab-a2146710c9ca
#=╠═╡
const pocket_state_index = TabularRL.makelookup(pocket_states)
  ╠═╡ =#

# ╔═╡ 6e20e3ee-6de3-46ce-bad6-ddad9694aeb4
#=╠═╡
function build_tabular_pocket_mdp()
	nstates = length(pocket_states)
	state_transition_map = zeros(Int64, 3, nstates)
	reward_transition_map = zeros(Float32, 3, nstates)
	s′ = copy(solved_cube_indices)
	s_vec = copy(solved_cube_indices)
	i_s_term = pocket_state_index[solved_pocket_cube]
	@info "Building state and reward transition maps"
	for s in pocket_states
		i_s = pocket_state_index[s]
		if i_s == i_s_term
			state_transition_map[:, i_s] .= i_s
			reward_transition_map[:, i_s] .= 0f0
		else
			for i_a in 1:3
				s′ = rotate_pocket_cube(s, i_a)
				i_s′ = pocket_state_index[s′]
				state_transition_map[i_a, i_s] = i_s′
				reward_transition_map[i_a, i_s] = -1f0
			end
		end
	end
	TabularMDP(pocket_states, pocket_moves, TabularDeterministicTransition(state_transition_map, reward_transition_map), () -> rand(eachindex(pocket_states)); state_index = pocket_state_index)
end
  ╠═╡ =#

# ╔═╡ 861a3436-6d46-425a-afa4-1f2be8994221
#=╠═╡
const pocket_mdp = build_tabular_pocket_mdp()
  ╠═╡ =#

# ╔═╡ e4d905b5-aefc-4a2f-8c41-e48c297b8663
#=╠═╡
const pocket_cube_solution = value_iteration_v(pocket_mdp, 1f0)
  ╠═╡ =#

# ╔═╡ b384cf65-0865-4b54-81af-0ad9fbd3db97
#=╠═╡
runepisode(pocket_mdp; π = pocket_cube_solution.optimal_policy)
  ╠═╡ =#

# ╔═╡ 8b8bcacd-0d00-4c72-86b4-e3bdf43d653f
#=╠═╡
extrema(pocket_cube_solution.final_value)
  ╠═╡ =#

# ╔═╡ 14f757fb-5f14-4f21-af1f-84be6249bdf5
md"""
## Pocket Cube Exhaustive Search Solution

Without the insight provided by reinforcement learning, we may instead consider an exhaustive forward search of forward trajectories in order to solve this problem from a given state.  In order to guarantee that we have the best possible solution, we would need to know ahead of time the upper limit on that length and then check all trajectories that are that length or less.  Out of the ones that reach a solved cube we would then select the unique one or ones that are the shortest.  

When generating these trajectories, we can notice that for every next step, we only need to consider the rotations of the two other faces that were not recently moved.  That is because the multiplicity of that face is handled on that step and it would be redundant to include it when analyzing the next step.  That means that if we start with 9 candidate moves (three possible face rotations of the three relevant faces), then every subsequent move will expand the trajectory space by 6 times (three possible face rotations for the two faces that were not just rotated).  The God's number for the pocket cube is 11 when we consider all of these possible face rotations as single moves.  That means we would have to check $9 \times 6^{10}$ trajectories which is only $(9*6^10).  Depending on how quickly we can through these that is entirely doable and ideally we would not save all of them but only a record of the best one so far.
"""

# ╔═╡ f8f98ed9-f936-41e6-b8de-02a71c0a06c7
const next_pocket_move_lookup = Dict([
		1 => [(f, n) for f in (2, 3) for n in 1:3]
		2 => [(f, n) for f in (1, 3) for n in 1:3]
		3 => [(f, n) for f in (1, 2) for n in 1:3]
	])

# ╔═╡ 0d41e787-eb70-4e47-9ecd-223caec70dd7
const all_pocket_moves = [(f, n) for f in 1:3 for n in 1:3]

# ╔═╡ b1d4a9f9-ebea-46ac-8c38-0ab3dfd2e905
get_next_pocket_moves(m::Tuple{Int64, Int64}) = next_pocket_move_lookup[m[1]]

# ╔═╡ b5f46f2f-2ea8-40ee-a28b-bd8c90e52149
function count_misplaced(s::AbstractVector{I}) where I<:Integer
	n = 0
	@inbounds @simd for i in eachindex(s)
		n += (s[i] != solved_pocket_cube[i])
	end
	return n
end

# ╔═╡ 5b910da9-52ad-4e3e-a264-2fa6c6df3cf3
#=╠═╡
const solved_rubiks_states = Set(rubiks_tabular_mdp.states)
  ╠═╡ =#

# ╔═╡ 1e2f7cb7-aabc-4ea1-a192-334d0788f53e
md"""
A better heuristic is the minimum number of turns needed to fix the corner pieces or edge pieces.  Taken separately, the maximum of these would still be a lower bound on the total moved needed.  Each quarter turn can fix at most 12 corner facelets and 8 edge facelets.

"""

# ╔═╡ 200ecf10-baf2-4f7e-a8ba-11a0dc3cab92
function count_misplaced_rubiks(s::AbstractVector{I}) where I<:Integer
	n = 0
	@inbounds @simd for i in eachindex(s)
		n += (s[i] != solved_cube_indices[i])
	end
	return n
end

# ╔═╡ 2af81c2a-5fa3-421d-88a0-a9865400cf47
function count_misplaced_rubiks_piece_heuristic(s::AbstractVector{I}) where I<:Integer
	n1 = 0
	@inbounds @simd for i in corner_inds_flat
		n1 += (s[i] != solved_cube_indices[i])
	end

	n2 = 0
	@inbounds @simd for i in edge_inds_flat
		n2 += (s[i] != solved_cube_indices[i])
	end

	max(ceil(Int64, n1/12), ceil(Int64, n2/8))
end

# ╔═╡ 73d40eb7-9213-4318-9bea-1a20edff2fbb
#=╠═╡
function check_next_moves_recur3(current_cube, trajectory::Vector{Int64}, cubes::Vector{Vector{UInt8}}, depth::Integer, threshold::Integer, states_checked)
	if in(current_cube, solved_rubiks_states)
		@info "Found a solution with $depth moves after checking $(states_checked[1]) states"
		return (true, trajectory[1:depth])
	end
	
	# misplaced = count_misplaced_rubiks(current_cube)
	# heuristic = ceil(Int64, misplaced / 20) #this value is a lower bound on the number of remaining moves needed to solve
	heuristic = count_misplaced_rubiks_piece_heuristic(current_cube)
	f = heuristic + depth
	
	(depth + heuristic > threshold) && return (false, f)
	
	min_overshoot = typemax(Int64)

	for m in get_valid_moves(view(trajectory, 1:depth))
		rotate_cube!(cubes[depth+1], current_cube, m)
		trajectory[depth + 1] = m
		states_checked[1] += 1
		(found, result) = check_next_moves_recur3(cubes[depth+1], trajectory, cubes, depth+1, threshold, states_checked)
		found && return (true, result)
		min_overshoot = min(min_overshoot, result)
	end

	return (false, min_overshoot)
end
  ╠═╡ =#

# ╔═╡ 1dee4e15-b9f6-47d4-bf7e-230946a6054d
#=╠═╡
function check_next_moves3(cube::SVector{48, UInt8}, maxdepth::Integer)
	in(cube, solved_rubiks_states) && return ([0], Vector{Int64}(), [Vector(cube)])
	trajectory = fill(1, maxdepth)
	best_trajectory = copy(trajectory)
	cubes = [Vector(cube) for i in 1:maxdepth]

	threshold = count_misplaced_rubiks_piece_heuristic(cube)
	# threshold = ceil(Int64, misplaced / 20)
	states_checked = [0]
	current_cube = Vector(cube)

	while threshold <= maxdepth
		@info "Starting search round with threshold: $threshold"
		found, result = check_next_moves_recur3(current_cube, trajectory, cubes, 0, threshold, states_checked)
		if found
			l = length(result)
			return (l, result, cubes[1:l], states_checked[1])
		else
			threshold = result
		end
	end
	@info "No solution found within maximum depth of $maxdepth"
	# return (best_depth, best_trajectory[1:best_depth[1]], best_cubes[1:best_depth[1]])
end
  ╠═╡ =#

# ╔═╡ b57a54e8-9876-4746-a1a3-f74812823a75
#=╠═╡
function solve_rubiks_cube_ida_star(s::SVector{48, UInt8}; maxdepth::Integer = 12)
	check_next_moves3(s, maxdepth)
end
  ╠═╡ =#

# ╔═╡ 5e6aca94-4efa-444a-82fc-cea805c20815
#=╠═╡
solve_rubiks_cube_ida_star(SVector{48}(initialize_rubiks_cube(30)); maxdepth = 9)
  ╠═╡ =#

# ╔═╡ acc8f2ae-0c21-404b-8cd1-34d498f7b4de
#=╠═╡
solve_pocket_cube_ida_star(rand(pocket_mdp.states))
  ╠═╡ =#

# ╔═╡ 4cbc56b2-85ac-47bb-80bf-963154cf3f48
#=╠═╡
const test_pocket_cube = rand(pocket_mdp.states)
  ╠═╡ =#

# ╔═╡ 700444c9-8ae0-4f6f-bfbe-7c4e2a014773
#=╠═╡
render_pocket_cube(test_pocket_cube)
  ╠═╡ =#

# ╔═╡ a58f1013-c5de-48d1-a6bc-a6c00adc9d1e
#=╠═╡
recursive_pocket_solution = solve_pocket_cube_exhaustive(test_pocket_cube)
  ╠═╡ =#

# ╔═╡ 60a41dea-32c0-43a6-b203-1eb07dbd7261
#=╠═╡
render_pocket_cube(recursive_pocket_solution[3][1])
  ╠═╡ =#

# ╔═╡ 318a221d-3a86-4419-9d87-e72eca3dd9c0
#=╠═╡
iterative_pocket_solution = solve_pocket_cube_exhaustive_iterative(test_pocket_cube)
  ╠═╡ =#

# ╔═╡ f156cea6-8d3e-4c08-aca0-ecd93ad961d1
#=╠═╡
render_pocket_cube(iterative_pocket_solution[3][2])
  ╠═╡ =#

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

# ╔═╡ a06c96f9-57c1-48e4-a725-08980892502e
#=╠═╡
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
	end |> foldxl(vcat)

	y = reduce(vcat, [fill(k, value_n, 1) for k in base_value:-1:minkey])

	l2 = size(X, 1)
	inds = shuffle(1:l2)
	return (X[inds, :], y[inds, :])
end	
  ╠═╡ =#

# ╔═╡ 16d6e982-c564-4d77-93b2-c4180eabbd2d
md"""
## Creating Markov Reward Process From Tabular Solution
"""

# ╔═╡ 37f28ba0-86b4-4c5d-96a4-e8adcec7c618
#=╠═╡
function check_rubiks_tabular_value(s::AbstractVector{I}) where I <: Integer
	haskey(rubiks_tabular_mdp.state_index, s) && return rubiks_value_iteration.final_value[rubiks_tabular_mdp.state_index[s]]
	return typemin(Float32)
end
  ╠═╡ =#

# ╔═╡ 3ac7bdb3-b4a8-4a64-82ea-dcd2dce14b6d
#=╠═╡
function rubiks_tabular_policy(s::AbstractVector{I}) where I <: Integer
	haskey(rubiks_tabular_mdp.state_index, s) && return rubiks_tabular_policy_lookup[rubiks_tabular_mdp.state_index[s]]

	best_value = typemin(Float32)
	best_action = 0
	s′ = copy(s)
	for i_a in eachindex(rubiks_moves)
		rotate_cube!(s′, s, i_a)
		candidate_value = check_rubiks_tabular_value(s′)
		if candidate_value > best_value
			best_value = candidate_value
			best_action = i_a
		end
	end

	iszero(best_action) && error("State is too far away from a known tabular solution")
	return best_action
end
  ╠═╡ =#

# ╔═╡ 30d9c367-01e6-4ee3-955a-7c1ad81cbbd5
#=╠═╡
function create_rubiks_mrp(;scramble_moves::Integer = 7)
	isterm(s) = s == solved_cube_indices
	initialize_state(;num_actions = scramble_moves) = initialize_rubiks_cube(num_actions)

	function step(s)
		i_a = rubiks_tabular_policy(s)
		s′ = rotate_cube(s, i_a)
		r = Float32(isterm(s′))
		(r, s′)
	end

	ptf = StateMRPTransitionSampler(step, initialize_state())
	StateMRP(ptf, initialize_state, isterm)
end
  ╠═╡ =#

# ╔═╡ 04bb99fd-2e99-4af7-8582-5cbc46fad29e
#=╠═╡
const rubiks_mrp = create_rubiks_mrp()
  ╠═╡ =#

# ╔═╡ 206a1146-4a01-412f-a17e-3d58cac83453
md"""
## Linear Approximation of Value Function
"""

# ╔═╡ 9084cb79-a89f-432d-a4f9-4b1f9ee65c4d
#=╠═╡
function run_rubiks_mrp_linear(num_episodes::Integer, λ::Float32, α::Float32; scramble_moves::Integer = 7, γ = 0.9f0, feature_vector = rubiks_binary_feature, kwargs...)
	mrp = create_rubiks_mrp(;scramble_moves = scramble_moves)
	output = NonTabularRL.semi_gradient_TDλ_linear(mrp, γ, λ, num_episodes, typemax(Int64), deepcopy(feature_vector), update_rubiks_feature!; α = α, kwargs...)
	
	q̂, form_kwargs = NonTabularRL.form_value_function(rubiks_reset_mdp, γ, update_rubiks_feature!, output.value_function, deepcopy(feature_vector), output.parameters)
	
	(episode_rewards = output.episode_history.errors, value_function = q̂, form_kwargs = form_kwargs)
end
  ╠═╡ =#

# ╔═╡ d9460da6-d51f-458b-bd90-888961ffe3a0
#=╠═╡
const linear_mrp_test = run_rubiks_mrp_linear(100_000, 0.75f0, 1f-4; scramble_moves = 6, trace_type = NonTabularRL.ReplacingTrace(), feature_vector = initialize_piece_vector())
  ╠═╡ =#

# ╔═╡ db57648d-ea66-4ece-85af-0df97a725ae5
#=╠═╡
display_learning_output(linear_mrp_test)
  ╠═╡ =#

# ╔═╡ 02523a61-5b7f-410c-ba3b-3564fcd8a35b
md"""
## Non-linear Approximation of Value Function
"""

# ╔═╡ 837479ed-a3c7-46dc-8fa0-31c98bb3ef6e
# ╠═╡ disabled = true
#=╠═╡
function initialize_rubiks_mrp_nonlinear_test(layers::Vector{Int64}; feature_vector = initialize_piece_vector(), kwargs...)

	params = NonTabularRL.initialize_fcann_params(length(feature_vector), layers, 1, 1, true)
	
	function f!(num_episodes, λ, α; scramble_moves = 7, γ = 0.9f0, kwargs...)
		mrp = create_rubiks_mrp(;scramble_moves = scramble_moves)
		output = NonTabularRL.semi_gradient_TDλ_fcann(mrp, γ, λ, num_episodes, typemax(Int64), deepcopy(feature_vector), update_rubiks_feature!, layers; α = α, parameters = params, kwargs...)
		q̂, form_kwargs = NonTabularRL.form_value_function(rubiks_reset_mdp, γ, update_rubiks_feature!, output.value_function, deepcopy(feature_vector), output.parameters)
		(episode_rewards = output.episode_history.errors, value_function = q̂, form_kwargs = form_kwargs)
	end
	(parameters = params, update! = f!)
end
  ╠═╡ =#

# ╔═╡ fb49c6c1-1a4c-4d57-92e8-b4abda1d6533
const mrp_nonlinear_layers = fill(256, 8)

# ╔═╡ b2324cff-2b72-4ec8-a51e-5123a2cc8ebd
#=╠═╡
const mrp_nonlinear_test = initialize_rubiks_mrp_nonlinear_test(mrp_nonlinear_layers)
  ╠═╡ =#

# ╔═╡ 91a90c89-a871-4776-a7d2-893fe8d7df63
#=╠═╡
const nonlinear_mrp_output = mrp_nonlinear_test.update!(100_000, 0.9f0, 1f-3; scramble_moves = 6)
  ╠═╡ =#

# ╔═╡ 949b2c5b-9657-4d8c-99e0-eb983c582ed1
#=╠═╡
display_learning_output(nonlinear_mrp_output; max_scramble = 6)
  ╠═╡ =#

# ╔═╡ 4453abd2-72c2-4a0b-8044-6345e0f87009
const mrp_nonlinear_layers2 = fill(32, 10)

# ╔═╡ 4210dabc-1ab4-46d8-950b-612201cdd0be
#=╠═╡
const mrp_nonlinear_test2 = initialize_rubiks_mrp_nonlinear_test(mrp_nonlinear_layers2)
  ╠═╡ =#

# ╔═╡ 01f86faa-224d-4509-acbf-8ee0d9b5881b
#=╠═╡
const nonlinear_mrp_output2 = mrp_nonlinear_test2.update!(100_000, 0.0f0, 1f-5; scramble_moves = 6)
  ╠═╡ =#

# ╔═╡ d84fc0d0-38bd-445d-8f03-4fe7f0db1d2a
#=╠═╡
display_learning_output(nonlinear_mrp_output2; max_scramble = 6)
  ╠═╡ =#

# ╔═╡ 6e8117cb-94ea-4216-8191-f250e03173e0
function setup_test_cases()
	function f!()
		sleep(5)
		return rand(10)
	end
	result = @use_state(nothing)
	stime = @use_state(nothing)
	ftime = @use_state(nothing)
	return (update! = f!, result = result, stime = stime, ftime = ftime)
end

# ╔═╡ 42b1cb51-1fbf-45b3-8c07-19484e9c9bbb
testcases = setup_test_cases()

# ╔═╡ eea81ca5-152b-45d6-94df-f05565376d2f
@use_effect([]) do
	@spawn begin
		testcases.stime[2](time())
		result = testcases.update!()
		testcases.result[2](result)
		testcases.ftime[2](time())
	end
end

# ╔═╡ 91e6576c-0142-409c-8749-194c66ba3c9d
# ╠═╡ disabled = true
#=╠═╡
const (X, y) = make_value_dataset(-1, 50_000)
  ╠═╡ =#

# ╔═╡ 3f459caa-c83d-4cdc-8b6e-78459b00d312
#=╠═╡
const ymod = 0.9f0 .^ -y
  ╠═╡ =#

# ╔═╡ abfc1d27-96a6-48dc-93d1-b2f5db4455ef
#=╠═╡
mean(abs.(ymod .- mean(ymod)))
  ╠═╡ =#

# ╔═╡ 1e6c3190-6b5f-4ada-a24f-98483658aa17
#=╠═╡
linear_training_output = fullTrain("test", X, ymod, 1, 64, Vector{Int64}(), 0f0, Inf, 0.0001f0, 0.1f0, 1; costFunc = "absErr", writeFiles=false)
  ╠═╡ =#

# ╔═╡ 2453b950-d1ff-45ec-8423-8d83d43afe8c
#=╠═╡
show_rubiks_policy_eval(make_greedy_policy(rubiks_dist_mdp, make_linear_value_function(Tuple(linear_training_output[2:3])), 0.9f0), 1000; max_scramble = 10)
  ╠═╡ =#

# ╔═╡ 4f9ae2a3-cc42-42cf-8fc9-c80d4588f675
function make_linear_value_function(params::NonTabularRL.FCANNParams)
	function v̂(s::Vector{UInt8})
		x = make_rubiks_feature(s)
		out = params[1][1] * x .+ params[2][1]
		return first(out)
	end

	v̂(s::@NamedTuple{cube::Vector{UInt8}, scramble_moves::Int64, move_count::Int64}) = v̂(s.cube)

	return v̂
end

# ╔═╡ ae4ddf00-85b2-4794-b92c-7ef76b8a1738
md"""
## Non-linear Approximation of Value Function
"""

# ╔═╡ 116c751a-76e6-4139-9926-0e8f58d30fe0
# ╠═╡ disabled = true
#=╠═╡
nonlinear_training_output = fullTrain("test", X, ymod, 100, 256, [256, 256, 256, 256, 256], 0.0f0, Inf, 0.01f0, 0.1f0, 1; costFunc = "absErr", writeFiles=false, use_μP = true)
  ╠═╡ =#

# ╔═╡ 075be7b6-e658-4818-90f6-6674c00c678a
#=╠═╡
show_rubiks_policy_eval(make_greedy_policy(rubiks_dist_mdp, make_nn_value_function(Tuple(nonlinear_training_output[2:3])), 0.9f0), 100; max_scramble = 10)
  ╠═╡ =#

# ╔═╡ 88087bbf-3bc3-4331-bf7a-44f4a023f54d
function make_nn_value_function(value_params::NonTabularRL.FCANNParams)
	value_activations = FCANN.form_activations(value_params[1])
	value_function(x, params) = NonTabularRL.fcann_value_function(x, params, value_activations, 0)
	
	
	
	
	function v̂(s::Vector{UInt8}) 
		x = make_rubiks_feature(s)
		value_function(x, value_params)
	end

	v̂(s::@NamedTuple{cube::Vector{UInt8}, scramble_moves::Int64, move_count::Int64}) = v̂(s.cube)

	return v̂
end

# ╔═╡ e8b7cf98-f5ed-4fb0-9d7a-4eb7fcd6bea4
function make_greedy_policy(mdp::StateMDP{T, S, A, P, F1, F2, F3}, v̂::Function, γ::T) where {T<:Real, S, A, P<:StateMDPTransitionDistribution, F1, F2, F3}
	num_actions = length(mdp.actions)
	function π_greedy(s)
		action_values = zeros(T, num_actions)
		for i_a in eachindex(action_values)
			(rewards, states, probabilities) = mdp.ptf.step(s, i_a)
			q = zero(T) 
			for i in eachindex(probabilities)
				v̂′ = !mdp.isterm(states[i])*v̂(states[i])
				q += probabilities[i]*(rewards[i] + γ*v̂′)
			end
			action_values[i_a] = q
		end
		make_greedy_policy!(action_values)
		i_a = sample_action(action_values)
	end
end

# ╔═╡ 8fc4500b-de4f-44f0-ac51-b2fb875a336a
#=╠═╡
md"""
Total nonlinear model parameters: $(FCANN.theta2Params(nonlinear_training_output[3], nonlinear_training_output[2]) |> length)
"""
  ╠═╡ =#

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
# ╠═╡ disabled = true
#=╠═╡
test_values(-3)
  ╠═╡ =#

# ╔═╡ 54c914bc-8822-4fd4-bd38-97fd1476558e
# ╠═╡ disabled = true
#=╠═╡
test_values(-4).nonlinear_results[1] |> extrema
  ╠═╡ =#

# ╔═╡ 3d1ad253-1fe6-4e5e-9959-690d88a278e4
# ╠═╡ disabled = true
#=╠═╡
predict(nonlinear_training_output[2], nonlinear_training_output[3], reshape(make_rubiks_feature(initialize_rubiks_cube(8)), 1, 48*48))
  ╠═╡ =#

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

# ╔═╡ 338515c3-6818-4c23-9d98-178e74c5f148
# ╠═╡ disabled = true
#=╠═╡
score_averages = compute_score_averages(rubiks_tabular_mdp.states, rubiks_value_iteration.final_value)
  ╠═╡ =#

# ╔═╡ 22e630e8-2f5b-40f6-9bbe-b0e6761b4e76
# ╠═╡ disabled = true
#=╠═╡
plot(scatter(x =score_averages[1], y = score_averages[2]), Layout(xaxis_title = "Turns Until Solution", yaxis_title = "Average Score"))
  ╠═╡ =#

# ╔═╡ 39c7bb0d-093c-4be7-a366-2adcbfe7a7a6
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ f7155154-3062-4711-80b3-8139c792e25d
#=╠═╡
show_rubiks_episode(value_iteration_π; max_steps = 20, s0 = initialize_rubiks_cube(10)).states |> states -> [render_cube(s) for s in states]
  ╠═╡ =#

# ╔═╡ 64c85a03-d88e-471e-8215-5b4ff51b5440
md"""
At the end of an MCTS attempt to improve score.  How often is the resulting state found in the state list for the 7 step MDP?
"""

# ╔═╡ 0624c4dc-d6e4-4a31-b13d-3618d124f857
# ╠═╡ disabled = true
#=╠═╡
compare_mcts_endpoint(10_000, 10, 1000f0, 10000, initialize_rubiks_cube(30))
  ╠═╡ =#

# ╔═╡ 323be52e-f967-44aa-a478-53b5f6575dab
#=╠═╡
function compare_mcts_endpoint(nsims, depth, c, steps, s0)
	mcts_output = show_rubiks_episode(s -> rubiks_mcts_policy(s; nsims = nsims, depth = depth, c = c); max_steps = steps, s0 = s0)
	x = any(haskey(rubiks_tabular_mdp.state_index, s) for s in mcts_output.states)
	(; success = x, mcts_output...)
end
  ╠═╡ =#

# ╔═╡ 92f16c16-5384-47cd-a8f6-31694b503ec8
md"""
# Multistep Version of MDP
So far we have considered the simplest possible cube moves, a single face rotation.  We have seen that with MCTS search or exhaustive forward search, we can improve the results significantly.  We could try to incorporate more of this forward search in the DP update, but we can also modify the MDP itself to turn each step into a much longer forward search.  Consider that for two forward moves, there are several sequences that yield the same cube.  Every time we rotate a face clockwise and then counterclockwise we return the cube to its original state.  The learning algorithm can use this information to infer a lack of progress, but we could also use an action space which selects moves in pairs and never repeats positions.  

For an MDP which expands the action space to two move sequences, we would want to first consider all 12 cube moves.  For each of these moves, there are only 11 subsequent moves that make sense, since each position has a single move out of the 12 which returns to the original cube.  Therefore, the action space would be $$12 \times 11 =$$ $(12*11) moves large instead of the original 12.  The action value selection process would then have to iterate over all of these possibilities turning each learning step into a two step foward lookahead.

What if the cube is actually solvable in only one move, but we are forced to do two by virtue of the action space design?  We should look at the intermediate states in the step function itself and terminate the action sequence if we reach a solved cube along the way.  Below is a method for defining a new MDP that uses this two move sequence action space and handles the termination condition as described above.
"""

# ╔═╡ bfefc8c5-8620-46c0-b6ef-5a8dd4106889
md"""
## N-Step Action Space
"""

# ╔═╡ 1735c965-8aa0-4c74-bb41-8800d145e09d
const face_index = TabularRL.makelookup(face_order)

# ╔═╡ 84d639ef-08d8-45c5-8df6-39e1dcb31abe
begin
	check_valid_move(::RubiksMove{F, D1}, ::RubiksMove{F, D2}) where {F<:Face, D1<:Direction, D2<:Direction} = false
	check_valid_move(::RubiksMove, ::RubiksMove) = true
	function check_valid_move(m1::RubiksMove{F1, D1}, m2::RubiksMove{F2, D2}, m3::RubiksMove{F3, D3}) where {F1<:Face, F2<:Face, F3<:Face, D1<:Direction, D2<:Direction, D3<:Direction}
		!check_valid_move(m1, m2) && return false
		!check_valid_move(m2, m3) && return false
		face_independence_lookup[face_index[F1]] != face_index[F2] && return true
		F3 != F1
	end
end

# ╔═╡ 44f7e193-61c2-4591-abe7-1800328abb18
begin
	get_valid_moves(m::RubiksMove) = filter(m2 -> check_valid_move(m, m2), rubiks_moves)
	get_valid_moves(m1::RubiksMove, m2::RubiksMove) = filter(m3 -> check_valid_move(m1, m2, m3), rubiks_moves)
end

# ╔═╡ 546635bc-83d8-4b07-8b0e-df6d8a29d45f
const valid_move_inds = Dict(begin
	i_a = rubiks_move_index[m]
	valid_moves = get_valid_moves(m)
	valid_inds = [rubiks_move_index[m2] for m2 in valid_moves]
	i_a => valid_inds
end
for m in rubiks_moves)

# ╔═╡ 1e8f8041-0cf2-47f7-b24f-d4b81ef36360
const valid_double_move_inds = mapreduce(vcat, keys(valid_move_inds)) do i_a
	[begin
		valid_moves = get_valid_moves(rubiks_moves[i_a], rubiks_moves[i_a2])
		valid_inds = [rubiks_move_index[m3] for m3 in valid_moves]
		(i_a, i_a2) => valid_inds
	end
	for i_a2 in valid_move_inds[i_a]]
end |> Dict

# ╔═╡ b609b7fc-3994-4488-beed-a151f1984d46
begin
	lookup_valid_moves() = eachindex(rubiks_moves)
	lookup_valid_moves(i_a::Integer) = valid_move_inds[i_a]
	lookup_valid_moves(move_inds::NTuple{1, Int64}) = lookup_valid_moves(first(move_inds))
	lookup_valid_moves(move_inds::NTuple{N, Int64}) where N = valid_double_move_inds[(move_inds[N-1], move_inds[N])]
	lookup_valid_moves(move_inds::Vector{Int64}) = lookup_valid_moves(Tuple(move_inds))
end

# ╔═╡ bd6ef1e0-0e75-4186-b597-01860adb10b4
begin
	function extend_move(move_inds::NTuple{N, Int64}) where N
		[(move_inds..., i_a) for i_a in valid_double_move_inds[(move_inds[N-1], move_inds[N])]]
	end
	
	function extend_move(move_inds::NTuple{1, Int64})
		[(move_inds..., i_a) for i_a in valid_move_inds[last(move_inds)]]
	end
end

# ╔═╡ 7b61d79d-e304-488d-8cc8-ed70a0e7d052
extend_move(move_index::Integer) = [(move, i_a) for i_a in valid_move_inds[move_index]]

# ╔═╡ 1222ffc4-82a4-4397-89f0-e0b5ce568fed
function make_rubiks_nstep_moves(previous_moves, n::Integer)
	n == 0 && return previous_moves
	newmoves = mapreduce(vcat, previous_moves) do moves
		extend_move(moves)
	end
	make_rubiks_nstep_moves(newmoves, n-1)
end

# ╔═╡ a67c9467-fe95-424d-92bf-1dd6eecfbd88
make_rubiks_nstep_moves(n::Integer) = make_rubiks_nstep_moves([(i,) for i in eachindex(rubiks_moves)], n-1)

# ╔═╡ ddcbd552-d3a2-46ed-9a5d-d705730501c7
const rubiks_nstep_moves = Dict(n => let
	actions = make_rubiks_nstep_moves(n)
	action_index = Dict(zip(actions, eachindex(actions)))
	(actions = actions, action_index = action_index)
	end
	for n in 1:4)

# ╔═╡ 1f3cc59d-81c4-4c2f-a6d2-702d803b1f1d
md"""
## N-step Transition Function
"""

# ╔═╡ ea436bfd-97df-4859-9bc4-14ee5f67ecaf
function rubiks_nstep_move(s::@NamedTuple{cube::Vector{UInt8}, scramble_moves::Int64, move_count::Int64}, move::NTuple{N, Int64}; kwargs...) where N
	(reward, cube′) = rubiks_nstep_move(s.cube, move; kwargs...)
	(reward, (cube = cube′, scramble_moves = s.scramble_moves, move_count = s.move_count + N))
end

# ╔═╡ 1060a7ff-4ed3-49e7-a4e4-12ec896e924d
md"""
## N-step MDP
"""

# ╔═╡ 1317a53a-c817-4950-aacb-528617ccb6e4
md"""
## DP λ Semi-gradient Control
"""

# ╔═╡ c3ece8b5-ddaa-46a1-bd44-1b7d99cb6fac
md"""
### Linear Method Curriculum Scramble
"""

# ╔═╡ f8180571-8869-4904-929b-dc89a0a612c6
function get_nstep_scramble_statistic(mdp::StateMDP{T, S, A, P, F1, F2, F3}, scramble::Integer, output, ntrials::Integer) where {N, T, S, A<:NTuple{N, Int64}, P, F1, F2, F3}
	1:ntrials |> Map() do i
		runepisode(mdp; π = s -> output.value_function(s; output.form_kwargs()...).maximizing_action, s0 = mdp.initialize_state(;nmoves = scramble))[3][end] 
	end |> foldxt(+) |> x -> x / ntrials
end

# ╔═╡ 8c76e9af-b50c-4d3c-b5f6-48c189a44c8b
function get_nstep_policy_scramble_statistic(mdp::StateMDP{T, S, A, P, F1, F2, F3}, scramble::Integer, output, ntrials::Integer) where {N, T, S, A<:NTuple{N, Int64}, P, F1, F2, F3}
	1:ntrials |> Map() do i
		runepisode(mdp; π = s -> output.policy_sample_action(s), s0 = mdp.initialize_state(;nmoves = scramble))[3][end] 
	end |> foldxt(+) |> x -> x / ntrials
end

# ╔═╡ fb4a36de-15af-41eb-8678-6418d55c5f65
function get_nstep_statistics(nstep_result, min_scramble, max_scramble; ntrials = 100)
	[(n = n, solve_rate = get_nstep_scramble_statistic(nstep_result.mdp, n, nstep_result.output, ntrials)) for n in min_scramble:max_scramble]
end

# ╔═╡ 3afc9543-8172-42e1-bb28-717dfb59e7ee
function get_nstep_policy_statistics(nstep_result, min_scramble, max_scramble; ntrials = 100)
	[(n = n, solve_rate = get_nstep_policy_scramble_statistic(nstep_result.mdp, n, nstep_result.output, ntrials)) for n in min_scramble:max_scramble]
end

# ╔═╡ 8e317749-78a6-4835-8a2d-e4f4690c7c68
md"""
## Non-linear TD Learning
"""

# ╔═╡ 990e31e1-a1ef-4ecc-8279-0b88824a032e
function calculate_max_performance(min_scramble::Integer, max_scramble::Integer, nmoves::Integer)
	ns = [ceil(Int64, n / nmoves) for n in min_scramble:max_scramble]
	avg_moves = sum(n^2 for n in ns) / sum(ns)
	return inv(avg_moves)
end 

# ╔═╡ 2308e4aa-7bcc-4c03-a496-fbe7f5d76bb1
#add these steup functions for linear training to do parameter studies

# ╔═╡ 45edbe8f-2c82-4ccd-9151-9e7551b4b62f
#128x8 with 3move mdp is 80 seconds on cpu and 30 seconds on gpu, only 40 seconds on gpu increasing to 1024x8
#1024x8 with 2 move mdp is 11 seconds on gpu, 128x8 is 6.1 seconds on gpu and 4.3 seconds on cpu
#256x8 with 2 move mdp is 13 seconds on cpu and 8.7 seconds on gpu
# rubiks_nstep_nonlinear_value_training(fill(256, 8), 1, 2, 1000; use_gpu = true)

# ╔═╡ c2dfc40a-9934-426f-8e46-48b142b2601b
const value_scrambles_2_step = (6, 7)

# ╔═╡ b67b017b-afb3-4caf-a3d5-7b3d3c530c25
#had 0.4 with this for 94% success rate on highest scramble
calculate_max_performance(value_scrambles_2_step..., 2)

# ╔═╡ e71ee180-2fdf-46b9-98a5-972a0a993cfd
#make display function that grabs the parameters from disk if they exist and displays the statistics, not the learning curve and then prints a message if that network doesn't exist.  Better yet have it scan the disk for all the options and show a menu for those and then run the display function

# ╔═╡ da030300-e2f4-4c18-a11b-29f8c7bad572
#another idea is to save this performance table for all the architectures after exhaustive training and then make a plot of the architectures and the final n for perfect performance and then the dropoff at the next n

# ╔═╡ 643a6ca5-d247-42b0-95b5-283cfa1e965f
const policy_scrambles_2_step = (2, 2)

# ╔═╡ d6a20337-857b-4775-8167-dcb1a2ec0ecf
calculate_max_performance(policy_scrambles_2_step..., 2)

# ╔═╡ 568815ac-9c48-4ccf-b22c-e9ed38831cad
const dp_λ_fcann_rubiks_nstep_mastery_results = Dict{NamedTuple, NamedTuple}()

# ╔═╡ 68a4f3ac-16d2-4b18-bdca-74718172ede5
save_fcann_params(parameters::FCANNParams, name::String) = FCANN.writeParams([parameters.weights], name)

# ╔═╡ b9f439b0-12cd-4c78-ac83-549451efc90a
function load_curriculum_dp_λ_nstep_params(hidden::Vector{Int64}, reslayers::Integer, n::Integer)
	name = "rubiks_cube_curriculum_dp_λ_fcann_params_$(n)_move_$(string(hidden))_$(reslayers)_reslayer.bin"
	if isfile(name)
		v = FCANN.readBinParams(name)
		θs = v[1][1]
		βs = v[1][2]
		(weights = (θs, βs), reslayers = reslayers)
	else
		initialize_fcann_params(48*48, hidden, 1, reslayers, true)
	end
end

# ╔═╡ 1478b8d9-a051-4f8d-b221-1468231b3ece
function save_curriculum_dp_λ_nstep_params(params::FCANNParams)
	input_size, hidden, num_layers = get_network_dimensions(params)
	name = "rubiks_cube_curriculum_dp_λ_fcann_params_$(n)_move_$(string(hidden))_$(params.reslayers)_reslayer.bin"
	save_fcann_params(params, name)
end

# ╔═╡ b54f218a-46f6-407c-bf7a-e41a87f351d6
function initialize_dp_λ_rubiks_nstep_fcann_test()
end

# ╔═╡ b903b3af-4c89-4054-88cd-4dfcc3a7d6f0
const fcann_2step_layers = fill(256, 4)

# ╔═╡ 58f6a2f1-c6eb-4031-9671-22c0d6416862
# ╠═╡ disabled = true
#=╠═╡
const fcann_2step_params = NonTabularRL.initialize_fcann_params(48*48, fcann_2step_layers, 1, 1, true)
  ╠═╡ =#

# ╔═╡ 8bc7748a-babe-46a9-9277-fc69d11220f9
# ╠═╡ disabled = true
#=╠═╡
const dp_mastery_2move_params_layers = fill(4096, 8)
  ╠═╡ =#

# ╔═╡ bd930459-d714-4cdd-a274-bc6ae44efb26
#=╠═╡
const dp_mastery_2move_fcann_params = NonTabularRL.initialize_fcann_params(48*48, dp_mastery_2move_params_layers, 1, 1, true)
  ╠═╡ =#

# ╔═╡ fc718149-0104-4893-8ace-3ffe1260ccdf
#=╠═╡
save_fcann_params(dp_mastery_2move_fcann_params, "dp_mastery_2move_fcann_4098x8_1_reslayer_params.bin")
  ╠═╡ =#

# ╔═╡ bca04135-4e8f-4d0d-9bd5-5af78a76455e
dp_step_mastery_2move_fcann_results = @use_state(nothing)

# ╔═╡ c873a5cd-2b65-4c80-80ee-9cdd57172b7f
dp_step_mastery_2move_fcann_start_time = @use_state(nothing)

# ╔═╡ de692498-e6e0-4d68-8975-1ff80abe0651
dp_step_mastery_2move_fcann_end_time = @use_state(nothing)

# ╔═╡ c74f80a7-71f6-49eb-9b9a-0c5c78cd7ec3
#=╠═╡
@use_effect([]) do
	schedule(Task() do
		dp_step_mastery_2move_fcann_start_time[2](time())
		result = run_dp_α_decay_step_mastery_nmove_fcann_test!(dp_mastery_2move_fcann_params, 4, 6, 2, zeros(Float32, 48*48), update_rubiks_feature!, 1_000, 1f-3; ϵ = 0.01f0, use_gpu = true)
		dp_step_mastery_2move_fcann_results[2](result)
		dp_step_mastery_2move_fcann_end_time[2](time())
	end)
end
  ╠═╡ =#

# ╔═╡ d2b7fac5-fcaa-4d0e-a611-87c50f6d4d84
#=╠═╡
create_spawn_message(dp_step_mastery_2move_fcann_start_time[1], dp_step_mastery_2move_fcann_end_time[1])
  ╠═╡ =#

# ╔═╡ 8918bd6e-4cc7-4f04-9f45-4ab9b2edb259
# ╠═╡ disabled = true
#=╠═╡
display_nstep_αdecay_output(dp_step_mastery_2move_fcann_results[1])
  ╠═╡ =#

# ╔═╡ fbee7bdb-e826-41bc-bca7-e900cc7c4dca
# ╠═╡ disabled = true
#=╠═╡
const dp_mastery_2move_params_layers2 = fill(512, 4)
  ╠═╡ =#

# ╔═╡ afd359a6-8f15-4d79-9267-07a2683f7635
#=╠═╡
const dp_mastery_2move_fcann_params2 = NonTabularRL.initialize_fcann_params(48*48, dp_mastery_2move_params_layers2, 1, 1, true)
  ╠═╡ =#

# ╔═╡ eff2c88c-a308-42d7-b3f2-98da787c930c
dp_step_mastery_2move_fcann_results2 = @use_state(nothing)

# ╔═╡ f453625b-ac72-46a5-a2a5-e21492dcdd86
dp_step_mastery_2move_fcann_start_time2 = @use_state(nothing)

# ╔═╡ 1f4ca1e0-d4c1-4ccf-aad4-3cba3ba822c2
dp_step_mastery_2move_fcann_end_time2 = @use_state(nothing)

# ╔═╡ 0648f541-aee5-451f-89a2-f8c55fb0105b
# ╠═╡ disabled = true
#=╠═╡
@use_effect([]) do
	@spawn begin
		dp_step_mastery_2move_fcann_start_time2[2](time())
		result = run_dp_α_decay_step_mastery_nmove_fcann_test!(dp_mastery_2move_fcann_params2, 5, 7, 2, deepcopy(rubiks_binary_feature), update_rubiks_feature!, 1_000_000, 1f-4; ϵ = 0.01f0)
		dp_step_mastery_2move_fcann_results2[2](result)
		dp_step_mastery_2move_fcann_end_time2[2](time())
	end
end
  ╠═╡ =#

# ╔═╡ 9864c34a-405b-466e-ae1b-e0844c2d4019
#=╠═╡
create_spawn_message(dp_step_mastery_2move_fcann_start_time2[1], dp_step_mastery_2move_fcann_end_time2[1])
  ╠═╡ =#

# ╔═╡ 5120365b-1317-4a2f-803c-bbac28046923
display_nstep_αdecay_output(::Nothing) = nothing

# ╔═╡ 7fe349ed-36fb-473e-80cc-6e7585610bf9
md"""
### Policy Gradient Methods
"""

# ╔═╡ 0e45945d-96f6-4eff-8276-162a72f8730d


# ╔═╡ 92f40a51-695c-459e-9861-3d3d59d55546
md"""
# Deterministic DP Learning
For problems where the transition function has known probabilities, we can take advantage of the Bellman equations to compute the action values from the state values.  We can therefore forgo approximating action values and instead focus on state estimation and use the target value from the Bellman optimality equation.  If our environment is also deterministic (only one transition for a state/action pair), then we can further simplify the Bellman update by computing all the transition state values in one forward pass.  That is because our action space has a fixed size, so there will be exactly as many states to evaluate as actions.  We could compute all of them at once using a preallocated input matrix for the state feature vectors and output vector for the action values.  

To see how this would work with the existing package, we need to define a new type and extend some of the previous methods.  First we will define a deterministic transition function which is identical to the transition sampler other than the name.  This designation means that although we only get a single output sample from the step function, that is in fact the only unique transition that exists so we can treat it differently than if we thought it was just a sample.
"""

# ╔═╡ d2ab9e40-1d90-40dc-b36b-6e778b821ac1
md"""
## Deterministic Transition
"""

# ╔═╡ a713e511-7c05-4cb4-9fee-2043fb0d4242
md"""
## Deterministic Action Value Update
When using a state value function and mdp, we iterate through the actions and compute the transition value for all possible transitions.  We can greatly simplify this function in the deterministic case, however, we want to make sure that our value function can operate on a matrix rather than just a single feature vector.  That way we can compute all the action values in one go.
"""

# ╔═╡ 4e0ede0b-bcde-42e6-baa6-0b0dea95a83c
md"""
## Non-linear Deterministic Cube Test
"""

# ╔═╡ 28a886a2-6fc0-4c16-9426-882f1745b67c
function run_deterministic_dp_rubiks_2step_fcann_test(min_moves, max_moves; γ = 0.9f0, layers = [8, 8], num_steps = 10_000, kwargs...)
	mdp =  make_deterministic_cube_2step_mdp(min_moves, max_moves)
	semi_gradient_deterministic_dp_fcann(mdp, γ, typemax(Int64), num_steps, update_rubiks_feature!, length(solved_cube_feature), layers; kwargs...)
end

# ╔═╡ 68cce5b5-a52e-47e3-8670-911dbbdb7509
fcann_deterministic_2step_output, set_fcann_deterministic_2step_output = @use_state(nothing)

# ╔═╡ 66f35e7b-c9d5-4a1f-8f08-fc83e419539c
fcann_deterministic_2step_start_time, set_fcann_deterministic_2step_start_time = @use_state(nothing)

# ╔═╡ 2935371e-fca5-47ef-89b9-781a1ff7b8b0
fcann_deterministic_2step_end_time, set_fcann_deterministic_2step_end_time = @use_state(nothing)

# ╔═╡ 125cc2d6-8712-4b41-96bc-243120eb4ff4
const fcann_deterministic_2step_results = Dict{NamedTuple, NamedTuple}()

# ╔═╡ 897476bc-bfc7-4e10-b965-0175902d9407
# ╠═╡ disabled = true
#=╠═╡
const deterministic_fcann_2step_params = FCANN.initializeparams_saxe(48*48, [256, 256, 256], 1; use_μP=true)
  ╠═╡ =#

# ╔═╡ 9d4fd48b-f0ca-4dd4-8648-e7a8827740b8
#=╠═╡
if isnothing(fcann_deterministic_2step_start_time)
	md"""
	##### Waiting to run process
	"""
elseif isnothing(fcann_deterministic_2step_end_time)
	md"""
	##### Currently running task started at $(unix2datetime(fcann_deterministic_2step_start_time)) for the first time
	"""
elseif fcann_deterministic_2step_start_time > fcann_deterministic_2step_end_time
	md"""
	##### Currently running task starting at $(unix2datetime(fcann_deterministic_2step_start_time)) and displaying previous results completed at $(unix2datetime(fcann_deterministic_2step_end_time))
	"""
else
	md"""
	##### Displaying most recent results completed at $(unix2datetime(fcann_deterministic_2step_end_time)) in $(fcann_deterministic_2step_end_time - fcann_deterministic_2step_start_time) seconds.  Ready to start a new task.
	"""
end
  ╠═╡ =#

# ╔═╡ 8fa47fbc-d097-4b7e-8ac7-8bcefb492ee3
#=╠═╡
if !isnothing(fcann_deterministic_2step_results)
	#update dictionary with most recent output
	fcann_deterministic_2step_results[fcann_deterministic_2step_output[1]] = fcann_deterministic_2step_output[2]
	@bind fcann_deterministic_2step_select PlutoUI.combine() do Child
		ks = sort(collect(keys(fcann_deterministic_2step_results)))
		layers = [k[1] for k in ks]
		moves = [k[2] for k in ks]
		αs = [k[3] for k in ks]
		md"""
		Select Results to View: $(Child(Select(ks)))
		"""
	end
else
	md"""
	Waiting for results
	"""
end
  ╠═╡ =#

# ╔═╡ f04f6ab1-6de1-4b38-8973-40064f496514
#try to load one of the other value functions into this technique and see how strong it is

# ╔═╡ 421add0b-6b3b-49fd-8f71-1898decd24af
function get_deterministic_2step_scramble_statistic(scramble::Integer, deterministic_output, ntrials::Integer)
	1:ntrials |> Map(i -> runepisode(make_deterministic_cube_2step_mdp(scramble, scramble); π = s -> deterministic_output.value_function(s).maximizing_action)[3][end]) |> foldxt(+) |> x -> x / ntrials
end

# ╔═╡ cc689336-57ac-4a0d-9a3e-81a17b2e0b39
function get_deterministic_2step_statistics(deterministic_output, min_scramble::Integer, max_scramble::Integer; ntrials = 1000) 
	[(n = n, solve_rate = get_deterministic_2step_scramble_statistic(n, deterministic_output, ntrials)) for n in min_scramble:max_scramble]
end

# ╔═╡ e2d1180d-e34e-45ad-8593-d6c1dd34cad2
#=╠═╡
function run_deterministic_dp_2step_scramble_mastery_fcann_test!(parameters::FCANNParams{T}, num_moves::Integer, feature_vector::Vector{T}, update_feature_vector!, step_interval, α::T; γ::T = 0.9f0, kwargs...) where T<:Real
	function check_reward_progress(episode_rewards::Vector{T}) 
		l = length(episode_rewards)
		episode_check = max(1000, ceil(Int64, l/2))
		mean(episode_rewards[max(1, l-episode_check):l])
	end
	layers = get_hidden_layers(parameters)
	mdp = make_deterministic_cube_2step_mdp(num_moves, num_moves+1) #always include an even and odd move to avoid a lack of certain states visted
	run_function() = semi_gradient_deterministic_dp_fcann(mdp, γ, typemax(Int64), step_interval, update_feature_vector!, length(feature_vector), layers; parameters = parameters, α = α, kwargs...)
	first_output = run_function()
	reward_check1 = check_reward_progress(first_output.episode_rewards)
	@info "After first learning pass for a scramble of $num_moves, average reward is $reward_check1"
	flush(stdout)

	second_output = run_function()
	reward_check2 = check_reward_progress(second_output.episode_rewards)
	@info "After second learning pass for a scramble of $num_moves, average reward is $reward_check2"

	episode_rewards = vcat(first_output.episode_rewards, second_output.episode_rewards)
	pass = 2
	while reward_check2 > reward_check1
		pass += 1
		@info "Reward still improving so proceeding with pass number $pass"
		reward_check1 = reward_check2
		second_output = run_function()
		reward_check2 = check_reward_progress(second_output.episode_rewards)
		episode_rewards = vcat(episode_rewards, second_output.episode_rewards)
		@info "After pass number $pass for a scramble of $num_moves, average reward is $reward_check2"
	end
	@info "Concluded learning after $pass passes"

	@info "Calculating performance statistics for final parameters"
	stats = get_deterministic_2step_statistics(second_output, 1, 15; ntrials = 1000)
	final_output = (value_function = second_output.value_function, episode_rewards = episode_rewards, final_reward = reward_check2, total_passes = pass, final_parameters = deepcopy(parameters), stats = stats, num_passes = pass)
	key = (layers = layers, scramble_moves = num_moves, α = α)

	return (key, final_output)
end
  ╠═╡ =#

# ╔═╡ a7a35c3d-b82d-48b0-b42a-5551ff38bc32
#=╠═╡
@use_effect([]) do
	@spawn begin
		set_fcann_deterministic_2step_start_time(time())
		output = run_deterministic_dp_2step_scramble_mastery_fcann_test!(deterministic_fcann_2step_params, 2, make_rubiks_feature(solved_cube_indices), update_rubiks_feature!, 10_000, 4f-3; ϵ = 0.01f0, γ = 0.9f0, reslayers = 1)
		set_fcann_deterministic_2step_output(output)
		set_fcann_deterministic_2step_end_time(time())
	end
end
  ╠═╡ =#

# ╔═╡ 9339a39a-bc74-4408-ac44-557366426840
function get_scramble_statistic(scramble::Integer, output::NamedTuple, ntrials::Integer)
	π(s) = output.value_function(s; output.form_kwargs()...).maximizing_action
	get_scramble_statistic(scramble, π, ntrials)
end

# ╔═╡ 53ca25d4-f0fd-471f-b509-519dc6cb5ecd
#next test is to add GPU version of this

# ╔═╡ d70eaa9f-ac2c-4303-89e4-a46443e65fe2
#=╠═╡
const transfered_deterministic_params = deepcopy(dp_params2)
  ╠═╡ =#

# ╔═╡ 04a77130-179b-48bf-8393-5c8d11c4d164
md"""
## MCTS Improvement on Best Two Step Solution
"""

# ╔═╡ 0bf5d23c-d0a8-4660-900c-43573c58faa1
const mcts_2step_eval_output = fcann_deterministic_2step_results[(layers = fill(128, 3), scramble_moves = 11, α = 4f-5)]

# ╔═╡ 8fb578c8-51d9-46ad-88ec-cd0cf0e73cf3
rubiks_2step_test_mdp.initialize_state()

# ╔═╡ ca5aad17-43a7-40af-abb8-5bc430510fce
function make_fast_deterministic_value_function(dp_output, mdp::StateMDP, params::FCANNParams{T}) where T<:Real
	feature_vector = zeros(T, size(params[1][1], 2))
	action_values = zeros(T, length(mdp.actions))
	reward_values = copy(action_values)
	feature_matrix = zeros(T, length(mdp.actions), length(feature_vector))
	function v̂(s)
		dp_output.value_function(s; feature_vector = feature_vector, action_values = action_values, reward_values = reward_values, feature_matrix = feature_matrix)
	end
	return v̂
end

# ╔═╡ f181e0a7-0e3f-43e2-af6e-70eb2ca98854
#=╠═╡
function run_mcts_2step_episode(scramble_moves, dp_output; mcts_kwargs...)
	mdp = make_rubiks_2step_mdp(scramble_moves, scramble_moves)
	v̂ = make_fast_deterministic_value_function(dp_output, mdp, dp_output.final_parameters)
	s0 = mdp.initialize_state()

	@info "checking ida* from starting state"
	solve_rubiks_cube_ida_star(SVector{48}(s0.cube); maxdepth=8)
	function vanilla_policy(s)
		if in(s.cube, rubiks_tabular_mdp.states)
			@info "Vanilla policy: Found tabular state on move count $(s.move_count)"
			i_s = rubiks_tabular_mdp.state_index[s.cube]
			i_a1 = findfirst(!iszero, rubiks_value_iteration.optimal_policy[:, i_s])
			cube′ = rotate_cube(s.cube, i_a1)
			i_s = rubiks_tabular_mdp.state_index[cube′]
			i_a2 = findfirst(!iszero, rubiks_value_iteration.optimal_policy[:, i_s])
			return mdp.action_index[(i_a1, i_a2)]
		else				
			v̂(s).maximizing_action
		end
	end
	
	vanilla_episode = runepisode(mdp; π = vanilla_policy, s0 = s0)

	vanilla_end = vanilla_episode[4].cube
	if vanilla_end != solved_cube_indices
		@info "Checking if vanilla solution is within 8 of known solution"
		solve_rubiks_cube_ida_star(SVector{48}(vanilla_end); maxdepth=8)
	else
		@info "Vanilla policy found good solution"
	end

	function mcts_policy(s)
		if in(s.cube, rubiks_tabular_mdp.states)
			@info "MCTS Policy: Found tabular state on move count $(s.move_count)"
			i_s = rubiks_tabular_mdp.state_index[s.cube]
			i_a1 = findfirst(!iszero, rubiks_value_iteration.optimal_policy[:, i_s])
			cube′ = rotate_cube(s.cube, i_a1)
			i_s = rubiks_tabular_mdp.state_index[cube′]
			i_a2 = findfirst(!iszero, rubiks_value_iteration.optimal_policy[:, i_s])
			return mdp.action_index[(i_a1, i_a2)]
		else				
			monte_carlo_tree_search(mdp, 0.9f0, (mdp, s, γ) -> v̂(s).maximizing_value, s; mcts_kwargs...)[1]
		end
	end
		
	mcts_episode = runepisode(mdp; π = mcts_policy, s0 = s0)

	mcts_end = mcts_episode[4].cube
	if mcts_end != solved_cube_indices
		@info "Checking if MCTS solution is within 8 of known solution"
		solve_rubiks_cube_ida_star(SVector{48}(mcts_end); maxdepth=8)
	else
		@info "MCTS policy found good solution"
	end
	
	return (vanilla_episode, mcts_episode)
end
  ╠═╡ =#

# ╔═╡ 8d40eb5b-ed23-4c13-abd9-6af0e63cb2b3
md"""
There are 132 2-step moves so doing MCTS with a depth of 1 and 133 sims effectly does an exhaustive one step lookahead search.  As a baseline we can see if this lookahead improves the performance of the policy
"""

# ╔═╡ c2ade361-6c96-431f-9237-98accc2de8ee
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
display_deterministic_2step_output(run_deterministic_dp_2step_scramble_mastery_fcann_test!(transfered_deterministic_params, 10, make_rubiks_feature(solved_cube_indices), update_rubiks_feature!, 10, 1f-4; ϵ = 0.01f0, γ = 0.9f0, reslayers = 1)[2])
  ╠═╡ =#

# ╔═╡ 432f8e64-c124-491d-9633-9d77bb1a9ff4
#=╠═╡
const mcts_2step_episodes = run_mcts_2step_episode(40, mcts_2step_eval_output; depth = 1, nsims = 133)
  ╠═╡ =#

# ╔═╡ b37cf65b-eb9a-4a78-be83-e4d4f54355d5
mcts_2step_episodes2, set_mcts_episodes = @use_state(nothing)

# ╔═╡ ae5fd508-1c9c-418c-a862-263bcc76630e
@use_effect([]) do
	@spawn begin
		output = run_mcts_2step_episode(20, mcts_2step_eval_output; depth = 5, nsims = 100_000)
		set_mcts_episodes(output)
	end
end

# ╔═╡ 1377a706-5a89-495e-98cf-d1b4ac1511f8
md"""
# Backwards Learning MDP and Policy Estimation
Since we can always scramble a solved cube and track the sequence of moves, we do have a way of seeing one possible trajectory to solving a cube from a particular scrambled state.  We simply need to reverse the sequence for the original scramble.  We may see repeated states since this reversal does not guarantee an efficient solution, but we can simply cut out those redundant sequences and hopefully arrive at an efficient example of a successful solution.  By doing this reversal, we can generate what appears to be a very good policy for which we can approximate a state value function.  This policy is an illusion though because it will only work on a particular scrambled state.  Nonetheless, we can generate an arbitrarily large amount of these specialized policies and try to learn a general function approximation.

## Reversing the Scramble
"""

# ╔═╡ b8a2150f-9be4-4c12-b645-37c9b91c7728
# ╠═╡ disabled = true
#=╠═╡
const valid_move_inds = Dict(begin
	valid_moves = get_valid_moves(rubiks_moves[i_a])
	i_a => [rubiks_move_index[x] for x in valid_moves]
end
							for i_a in 1:12)
  ╠═╡ =#

# ╔═╡ 4f79565b-860f-4862-92b0-fdc1b00f0c1d
md"""
Let's make a trajectory of scrambling a cube with 30 moves
"""

# ╔═╡ 3bcc2eb8-5c4d-46ad-a1f4-f380adf66812
md"""
If we only look at the unique states from this trajectory, it is in general less.  We would like to remove redundant states from the trajectory.  For example if we enumerate the cubes so that the solved one is 1 and then increasing unique scrambles go up from there, we may see a sequence like 1, 2, 3, 2, 5, 11.  In this case, the visit to state 3 and back to 2 is redundant because we can go straight from 2 to 5, or in the reverse direction we can go from 5 to 2 directly.  It doesn't matter which part we remove, but we can use the convention that we keep the first instance of a state and then remove everything between that and the next instance.  We should do the same with the actions.
"""

# ╔═╡ 2026135e-5d97-4755-8de7-cd3175e37298
md"""
Each action is the rotation of a face clockwise or counterclockwise.  Reversing these actions is as easy as switching one to the other.  For example, `rubiks_moves[5]` is rotating the right face counter clockwise.  Below you can see the impact of reversing this move.
"""

# ╔═╡ 1d332f0c-bb86-49f8-9f61-2ba03354b3dd
rubiks_moves[5] |> reverse_move

# ╔═╡ a35f5d8b-a9ea-4d9c-84c2-44ceea2fbe56
md"""
Using this reversal function, we should be able to generate a policy from this trajectory that will successfully solve the cube we end up with at the end of the scramble.  First let's create a lookup table that maps each rubiks move index to the index of the reverse move.
"""

# ╔═╡ bfbc45f1-900e-4596-965a-0dfe6e9bdc9c
const rubiks_move_index_reversal_lookup = Dict(i_a => rubiks_move_index[reverse_move(rubiks_moves[i_a])] for i_a in eachindex(rubiks_moves))

# ╔═╡ 75e1b19f-299b-4f05-a98a-87cc4ddb14d0
function make_unscramble_policy(states::AbstractVector{S}, actions::Vector{Int64}) where {I<: Integer, S <: AbstractVector{I}}
	l = length(actions)
	action_lookup = Dict(states[i] => rubiks_move_index_reversal_lookup[actions[i]] for i in l:-1:1)
	π(s) = action_lookup[s]
end

# ╔═╡ fa4fa0fe-113b-4537-a498-a1be486a9b8e
md"""
If we look at the first 7 scramble moves and compare it to the exact solution for the tabular problem, we see that the fastest way to unscramble these first 7 moves is simply to reverse them.  There is no redundancy for this early stage of the problem.  Hopefully that efficiency extends somewhat to the later scrambles.

Upon closer inspection, even in the early moves, matching the exact solution is very inconsistent.  The result is that often even without repeating any states, we do often repeat state values in terms of their distance away from the solution.  If we try to learn the state values which effectively tell us how far away we are from the solved state, we cannot learn good values.  This value function is only useful in so far that it can tell us how close we are to a solution.  This reversible policy does find a solution, but not in any learnable way because its success is purely due to random chance and there is no learnable pattern from its behavior.
"""

# ╔═╡ cb2dd4f3-b0ac-4148-877a-408831fea5ef
#=╠═╡
[haskey(rubiks_tabular_mdp.state_index, test_trajectory[1][i]) ? rubiks_value_iteration.final_value[rubiks_tabular_mdp.state_index[test_trajectory[1][i]]] : "N/A" for i in 1:30] 
  ╠═╡ =#

# ╔═╡ 267c92f0-2b55-421a-8210-adad5e6ad811
md"""
In order to do policy evaluation with this reversal trick, we need to initialize a scrambled state and have a policy that knows how to unscramble it.  We can cheat a bit by generating the policy lookup when we initialize the state, and then the policy function can simply use that lookup along the trajectory.  If everything works properly, we will never encounter a state that isn't in the lookup table.
"""

# ╔═╡ c71826b6-e295-4c8f-8d73-52dbde7a39e7
md"""
## Designing the Reverse MDP

In order to use this learning method with policy estimation, we need a way to execute the reverse policy for every episodic trajectory.  I can make the entire reverse policy available within the starting state and simply have that information passed to every subsequent state within an episode.  Even though the policy function will have access to that and be able to use it to make action decisions, the approximation function will only make use of the rubik's cube state itself.  That way I can train the value function appropriately without special information.
"""

# ╔═╡ a303ee2c-ac5c-4728-b191-ea5570538a4d
function rubiks_reversible_move(s::NamedTuple, i_a::Integer; kwargs...) 
	(r, cube′) = rubiks_reversible_move(s.cube, i_a; kwargs...)
	return (r, (cube = cube′, π = s.π))
end

# ╔═╡ 139cc887-f3ee-4810-aa97-2aec301c5238
function make_reversible_feature_update(update_feature_vector!)
	update!(x, s::NamedTuple) = update_feature_vector!(x, s.cube)
	return update!
end

# ╔═╡ a19597b5-fa79-4808-8f60-9bafb1a0dbb4
function π_cube_reverse(s::NamedTuple)
	s.π(s.cube)
end

# ╔═╡ ce885abc-23ab-47d1-87d8-bdf5bd512212
md"""
## Gradient Monte Carlo Policy Estimation
"""

# ╔═╡ cab6e8c8-5276-489d-8dc2-f3adcfda8ebc
md"""
### Linear Estimation
"""

# ╔═╡ 1e2456d3-8d14-4568-88df-5f5757ce928e
md"""
### Non-linear Estimation
"""

# ╔═╡ cbf5ed88-ef38-40f9-ad2c-649505f55997
const reversible_fcann_layers = fill(64, 3)

# ╔═╡ 94a15d87-790e-4917-ba9f-446f699cfe92
const reversible_fcann_params = NonTabularRL.initialize_fcann_params(48*48, reversible_fcann_layers, 1, 1, true)

# ╔═╡ 621f415b-ef9b-4ad5-bcc0-dcf51af1d1f9
md"""
## Semi-gradient TD Estimation
"""

# ╔═╡ d28b936a-5ca6-494c-a6fd-c96c60d24add
const reversible_fcann_td_layers = fill(128, 3)

# ╔═╡ 55f31d00-a946-46e2-b021-855120b73e77
const reversible_fcann_td_params = NonTabularRL.initialize_fcann_params(48*48, reversible_fcann_td_layers, 1, 1, true)

# ╔═╡ 7fdfe56e-681a-4708-ac7e-5cbc7957adc1
function form_policy_function(mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, value_function::Function) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1<:Function, F2<:Function, F3<:Function}
	function π(s::S; action_values::Vector{T} = zeros(T, length(mdp.actions)), kwargs...)
		maxq, i_a_max = NonTabularRL.update_action_values!(action_values, s, s -> value_function(s; kwargs...), mdp, γ)
		i_a_max
	end
end

# ╔═╡ c2a1a4a9-c652-400f-ad3a-5a05d4e10073
#=╠═╡
function smooth_error(error_history, n)
	l = length(error_history)
	[mean(error_history[i-n:i]) for i in n+1:l]
end
  ╠═╡ =#

# ╔═╡ 6148ec6a-5d38-40d1-84ba-8b5253d5fdaa
#=╠═╡
begin
	function plot_rewards(rewards::AbstractVector{T}, nsmooth::Integer, npoints::Integer) where T<:Real 
		isempty(rewards) && return plot()
		l = length(rewards)
		plot(smooth_error(rewards, nsmooth)[round.(Int64, LinRange(1, max(1, l - nsmooth), npoints))])
	end

	function plot_rewards(rewards::AbstractVector{A}, nsmooth::Integer, npoints::Integer) where A <: Union{Missing, T} where T<:Real 	
		isempty(rewards) && return plot()
			
		newrewards = [!ismissing(a) for a in rewards]
		plot_rewards(newrewards, nsmooth, npoints)
	end
end
  ╠═╡ =#

# ╔═╡ bec0a916-9689-482f-9008-15f035057838
#=╠═╡
plot_rewards(test_fcann_dp_output.episode_rewards, 1000, 1000)
  ╠═╡ =#

# ╔═╡ 2d67019e-557e-4751-8c08-429be62b76d4
#=╠═╡
plot_rewards(test_fcann_dp_output2.episode_rewards, 1000, 1000)
  ╠═╡ =#

# ╔═╡ 4f163fc4-beb6-43ff-a232-d3d8c19199c0
#=╠═╡
plot_rewards(curriculum_results[layer_select].dict_results[move_select][α_select].episode_rewards, 100, 1000)
  ╠═╡ =#

# ╔═╡ 5f2bdb07-5a8e-4a62-a288-ba4b0d29aaac
#=╠═╡
plot_rewards(test_fcann_dp_piece_output3.episode_rewards, 100, 1000)
  ╠═╡ =#

# ╔═╡ eb488bad-2830-406f-ad87-7cc98566e610
#=╠═╡
plot_rewards(test_fcann_dp_essential_output.episode_rewards, 100, 1000)
  ╠═╡ =#

# ╔═╡ c6fbbf97-1a25-4b80-bd5c-89034efa3f07
#=╠═╡
function display_deterministic_2step_output(output)
	reward_plot = output.episode_rewards |> v -> plot_rewards(v, 100, 1000)
	try
		md"""
		Training progress stalled after $(output.num_passes) passes and $(length(output.episode_rewards)) episodes
		$reward_plot
		$(DataFrame(output.stats))
		"""
	catch
		md"""
		$reward_plot
		$(DataFrame(output.stats))
		"""
	end
end
  ╠═╡ =#

# ╔═╡ a76fa602-642d-46d2-9540-4fa94b38e7cb
#=╠═╡
if !isnothing(fcann_deterministic_2step_results)
	display_deterministic_2step_output(fcann_deterministic_2step_results[fcann_deterministic_2step_select[1]])
else
	md"""
	Waiting for results
	"""
end
  ╠═╡ =#

# ╔═╡ 17006174-caad-4043-b8d5-883aa0e10c80
#=╠═╡
onehot2value(v::BitVector) = value_lookup[v]
  ╠═╡ =#

# ╔═╡ 8218ea8b-3ba6-45fb-ac36-89ba6cccf112
#=╠═╡
function bits2value(v::BitVector; output = zeros(UInt8, 8, 6))
	for (i, j) in enumerate(1:6:287)
		output[i] = onehot2value(v[j:j+5])
	end
	return output
end
  ╠═╡ =#

# ╔═╡ 93eefafa-60da-47b0-9ea5-6e9085e1c231
const square_vectors = make_onehot_vector.(square_values)

# ╔═╡ 15326e1a-96b7-414e-9967-46e222598bc6
const onehot_lookup = Dict(zip(square_values, square_vectors))

# ╔═╡ 0d80de04-ac65-4cce-9db2-5f3053079a1b
value2onehot(v::Integer) = onehot_lookup[UInt8(v)]

# ╔═╡ 27b38778-17fe-4aca-8cec-341e2333e536
#=╠═╡
bits2value(solved_cube_bits)
  ╠═╡ =#

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
	Select Rotation: $(Child(:rot, Select([Clockwise => "Clockwise", CounterClockwise => "Counter Clockwise", Double => "Double"])))
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

	function get_rotation_indices(::RubiksMove{F, Double}) where F<:Face
		cube = reshape(get_rotation_indices(RubiksMove{F, Clockwise}()), 8, 6)
		rotation_mapping = clockwise_rotation_mapping[F]
		
		fnum = face2value[F]
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

# ╔═╡ 44499455-ab75-4a65-9fec-1d544afb8a33
const rubiks_transition = StateMDPTransitionDeterministic(rubiks_move, solved_cube_indices)

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

# ╔═╡ 8c396914-675b-404f-9f0c-90e880c55658
function initialize_rubiks_cube(num_actions)
	cube = copy(solved_cube_indices)
	cube′ = copy(cube)
	action_list = Vector{Int64}()
	i_a = rand(eachindex(rubiks_moves))
	action_list = [i_a]
	for i in 1:num_actions
		rotate_cube(cube, i_a; cube′ = cube′)
		cube .= cube′
		i_a = rand(lookup_valid_moves(action_list))
		push!(action_list, i_a)
	end
	return cube
end

# ╔═╡ 1fbcf1f1-722a-4014-b8f6-11c2a5f12548
function make_rubiks_mdp(make_init_actions::Function, transition::AbstractStateTransition)
	StateMDP(rubiks_moves, transition, () -> initialize_rubiks_cube(make_init_actions()), s -> isequal(s, solved_cube_indices))
end

# ╔═╡ b28d50ad-ca0a-4743-bace-be46c0821ac2
const rubiks_cube_mdp = make_rubiks_mdp(() -> 30, rubiks_transition)

# ╔═╡ fe20a8bd-c608-4ac7-9bb0-3251bcf7d85c
runepisode(rubiks_cube_mdp; max_steps = 100)

# ╔═╡ 18fa27e3-d076-4543-92cc-679ee3c32156
const rubiks_mdp = make_rubiks_mdp(() -> 5, rubiks_transition)

# ╔═╡ 09cf4ae7-d1fb-44d6-a617-0dbc7f64c2f2
const rubiks_dist_mdp = make_rubiks_mdp(() -> 5, rubiks_transition_distribution)

# ╔═╡ 3617c5a0-79e6-4fc0-814e-45bbdd373b5d
rubiks_mcts_policy(s; kwargs...) = TabularRL.monte_carlo_tree_search2(rubiks_dist_mdp, 0.99f0, s, π_dist_rubiks_uniform!, 1f0, 1, -48f0, 48f0; depth = 5, c = 0.5f0, kwargs...)[1](s)

# ╔═╡ 3af88755-0e11-4a17-8867-b8687a484312
function initialize_reset_cube(nmoves::Integer)
	cube = initialize_rubiks_cube(nmoves)
	return (cube = cube, scramble_moves = nmoves, move_count = 0)
end

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

# ╔═╡ 52eb3090-66a0-41aa-90ed-f658847f9601
const tabular_eval_cube = initialize_rubiks_cube(6)

# ╔═╡ 5656dbff-1763-4464-bf7a-438983907e8e
testcube = initialize_rubiks_cube(10)

# ╔═╡ 71bd3296-99ba-4706-9c34-c8543f9fb020
ceil(Int64, count_misplaced_rubiks(testcube) / 20)

# ╔═╡ b9b7030d-3035-4401-bd5e-006bd2fd583f
count_misplaced_rubiks_piece_heuristic(testcube)

# ╔═╡ d86a6362-bd09-4eac-a0a3-2149c566d693
dp_step_mastery_2move_fcann_results2[1].output_dict[0.000125f0].value_function((cube = initialize_rubiks_cube(5), scramble_moves = 5, move_count = 0))

# ╔═╡ 004c8296-38db-4999-a134-3f009eba8034
monte_carlo_tree_search(rubiks_2step_test_mdp, 0.9f0, (mdp, s, γ) -> mcts_2step_eval_output.value_function(s).maximizing_value, (cube = initialize_rubiks_cube(10), scramble_moves = 10, move_count = 0); depth = 1, nsims = 200)

# ╔═╡ 013fe195-8a8a-4717-a95b-16bb86e8de29
#create a cube mdp which saves in the state the number of initial scramble moves as well as how many moves have been attempted since the initial scramble
function make_reset_mdp(min_moves::Integer, max_moves::Integer)
	initialize_state(nmoves::Integer) = (cube = initialize_rubiks_cube(nmoves), scramble_moves = nmoves, move_count = 0)
	function isterm(s)
		s.move_count > 50 && return true
		s.move_count > 2*s.scramble_moves && return true
		s.cube == solved_cube_indices
	end

	function step(s, i_a)
		isterm(s) && return (0f0, s)
		cube′ = rotate_cube(s.cube, i_a)
		s′ = (cube = cube′, scramble_moves = s.scramble_moves, move_count = s.move_count + 1)
		r = Float32(s′.cube == solved_cube_indices)
		(r, s′)
	end
	ptf = StateMDPTransitionDeterministic(step, initialize_state(1))
	
	StateMDP(rubiks_moves, ptf, () -> initialize_state(rand(min_moves:max_moves)), isterm)
end

# ╔═╡ fd053336-7b1f-4c79-afaa-f60b6fcc5938
const rubiks_reset_mdp = make_reset_mdp(1, 5)

# ╔═╡ 54f505d3-2a6b-4ce7-a0a4-96b62159dd4c
function form_rubiks_policy(v̂::Function)
	π_raw = form_policy_function(rubiks_reset_mdp, 1f0, s -> v̂((cube = s.cube, π = Returns(1))))

	# π(s::Vector{UInt8}) = π_raw((cube = s, scramble_moves = 100, move_count = 0))
end

# ╔═╡ f90125d1-04d9-4c0c-a0b3-cf2c98b933d8
function form_reversible_policy(mc_output::NamedTuple)
	q̂, form_kwargs = NonTabularRL.form_value_function(rubiks_reset_mdp, 0.9f0, update_rubiks_feature!, mc_output.value_function, deepcopy(rubiks_binary_feature), mc_output.parameters)

	π(s) = q̂(s; form_kwargs()...).maximizing_action
end

# ╔═╡ 128c8762-e4a7-46e1-a673-39d0ffbf2f72
function run_sarsa_rubiks_linear_test(min_scramble::Integer, max_scramble::Integer; γ = 0.9f0, max_steps = 10_000, kwargs...)
	mdp =  make_reset_mdp(min_scramble, max_scramble)
	NonTabularRL.semi_gradient_sarsa_linear(mdp, γ, typemax(Int64), max_steps, deepcopy(rubiks_binary_feature), update_rubiks_feature!; kwargs...)
end

# ╔═╡ 027d5ff6-b250-427f-9a00-ec9562236bc3
function run_sarsa_rubiks_linear_test(min_scramble::Integer, max_scramble::Integer, λ; γ = 0.9f0, max_steps = 10_000, kwargs...)
	mdp =  make_reset_mdp(min_scramble, max_scramble)
	NonTabularRL.sarsa_λ_linear(mdp, γ, λ, typemax(Int64), max_steps, deepcopy(rubiks_binary_feature), update_rubiks_feature!; kwargs...)
end

# ╔═╡ 35ae742a-db14-4868-bb2d-d59cd9300de4
const test_sarsa_output = run_sarsa_rubiks_linear_test(3, 5, 0.85f0; max_steps = 100_000, α = 0.004f0, ϵ = 0.01f0)

# ╔═╡ d163d3cd-e8ca-44fb-81b2-66a2e1a92812
function run_dp_rubiks_linear_test(min_moves, max_moves; γ = 0.9f0, num_steps = 10_000, kwargs...)
	mdp =  make_reset_mdp(min_moves, max_moves)
	output = semi_gradient_dp_linear(mdp, γ, typemax(Int64), num_steps, deepcopy(rubiks_binary_feature), update_rubiks_feature!; kwargs...)
end

# ╔═╡ 99a7d920-173b-4b01-ab16-5248e6849f12
function run_dp_rubiks_linear_test(min_moves, max_moves, λ; γ = 0.9f0, num_steps = 10_000, kwargs...)
	mdp =  make_reset_mdp(min_moves, max_moves)
	output = NonTabularRL.dp_λ_linear(mdp, γ, λ, typemax(Int64), num_steps, deepcopy(rubiks_binary_feature), update_rubiks_feature!; kwargs...)
end

# ╔═╡ a538ead8-188b-4dd6-a105-a0c7a5d7d64f
const test_dp_output = run_dp_rubiks_linear_test(3, 4; num_steps = 1_000_000, α = 1f-4, ϵ = 0.01f0)

# ╔═╡ ae2427cb-be0a-4792-b0d9-5d14c17d9fb4
monte_carlo_tree_search(rubiks_mcts_mdp, 0.9f0, (mdp, s, γ) -> test_dp_output.value_function(s).maximizing_value, rubiks_mcts_s0; depth = 1, nsims = 13)

# ╔═╡ 6f551c31-3082-452a-ad9d-d77432c61af9
function run_sarsa_rubiks_fcann_test(min_scramble::Integer, max_scramble::Integer, layers::Vector{Int64}; γ = 0.9f0, max_steps = 10_000, feature_vector = rubiks_binary_feature, kwargs...)
	mdp =  make_reset_mdp(min_scramble, max_scramble)
	NonTabularRL.semi_gradient_sarsa_fcann(mdp, γ, typemax(Int64), max_steps, deepcopy(feature_vector), update_rubiks_feature!, layers; kwargs...)
end

# ╔═╡ 03432a16-954e-4223-abdf-729aea9cbd26
function run_dp_rubiks_fcann_test(min_scramble::Integer, max_scramble::Integer, layers::Vector{Int64}; γ = 0.9f0, max_steps = 10_000, feature_vector = rubiks_binary_feature, kwargs...)
	mdp =  make_reset_mdp(min_scramble, max_scramble)
	NonTabularRL.semi_gradient_dp_fcann(mdp, γ, typemax(Int64), max_steps, deepcopy(feature_vector), update_rubiks_feature!, layers; kwargs...)
end

# ╔═╡ debc0525-2158-46ea-a7a7-cbf983fc40d6
function run_sarsa_rubiks_fcann_test(min_scramble::Integer, max_scramble::Integer, layers::Vector{Int64}, λ; γ = 0.9f0, max_steps = 10_000, feature_vector = rubiks_binary_feature, kwargs...)
	mdp =  make_reset_mdp(min_scramble, max_scramble)
	NonTabularRL.sarsa_λ_fcann(mdp, γ, λ, typemax(Int64), max_steps, deepcopy(feature_vector), update_rubiks_feature!, layers; kwargs...)
end

# ╔═╡ 699bf657-14cb-4575-925b-bedf97bc168c
const test_sarsa_fcann_output = run_sarsa_rubiks_fcann_test(3, 3, sarsa_params_layers, 0.0f0; max_steps = 100_000, α = 1f-2, ϵ = 0.01f0, parameters = sarsa_fcann_params) # feature_vector = make_rubiks_feature(solved_cube_indices))

# ╔═╡ 9b1e056c-1025-45a9-b508-f16691fb5696
function run_dp_rubiks_fcann_test(min_scramble::Integer, max_scramble::Integer, layers::Vector{Int64}, λ; γ = 0.9f0, max_steps = 10_000, feature_vector = rubiks_binary_feature, kwargs...)
	mdp =  make_reset_mdp(min_scramble, max_scramble)
	NonTabularRL.dp_λ_fcann(mdp, γ, λ, typemax(Int64), max_steps, deepcopy(feature_vector), update_rubiks_feature!, layers; kwargs...)
end

# ╔═╡ a2ef7212-d2b6-40ab-8af8-6d38ab9f39f2
const test_dp_fcann_output = run_dp_rubiks_fcann_test(4, 5, dp_params_layers; max_steps = 1_000, α = 1f-4, ϵ = 0.01f0, parameters = dp_fcann_params) #, feature_vector = zeros(Float32, 48*48), use_gpu = true)

# ╔═╡ bd50c529-c78f-48c4-8b30-bbbe4af87ca6
#=╠═╡
function run_rubiks_dp_curriculum_eval_loop(min_steps, max_steps; kwargs...)
	reward_progress = true
	curriculum_count = 1
	while reward_progress
		set_test_fcann_dp_curriculum_count(curriculum_count)
		set_test_fcann_dp_start_time(time())
		output = run_dp_rubiks_fcann_test(min_steps, max_steps; kwargs...)
		set_test_fcann_dp_output(output)
		set_test_fcann_dp_end_time(time())
		reward_progress = check_reward_progress(output.episode_rewards)
		curriculum_count += 1
		if !reward_progress
			break
		end
	end
end
  ╠═╡ =#

# ╔═╡ 738f63c0-4101-48df-b73f-610bea1553af
#=╠═╡
function run_dp_step_mastery_fcann_test!(parameters::FCANNParams{T}, min_moves::Integer, max_moves::Integer, feature_vector, update_feature_vector!, step_interval; γ::T = 0.9f0, kwargs...) where T<:Real
	function check_reward_progress(episode_rewards::Vector{T}) 
		l = length(episode_rewards)
		episode_check = max(1000, ceil(Int64, l/2))
		mean(episode_rewards[max(1, l-episode_check):l])
	end
	layers = get_hidden_layers(parameters)
	mdp = make_reset_mdp(min_moves, max_moves)
	first_output = semi_gradient_dp_fcann(mdp, γ, typemax(Int64), step_interval, feature_vector, update_feature_vector!, layers; parameters = parameters, kwargs...)
	reward_check1 = check_reward_progress(first_output.episode_rewards)
	@info "After first learning pass for a scramble of $min_moves to $max_moves, average reward is $reward_check1"

	second_output = semi_gradient_dp_fcann(mdp, γ, typemax(Int64), step_interval, feature_vector, update_feature_vector!, layers; parameters = parameters, kwargs...)
	reward_check2 = check_reward_progress(second_output.episode_rewards)
	@info "After second learning pass for a scramble of $min_moves to $max_moves, average reward is $reward_check2"

	episode_rewards = vcat(first_output.episode_rewards, second_output.episode_rewards)
	pass = 2
	while reward_check2 > reward_check1
		pass += 1
		@info "Reward still improving so proceeding with pass number $pass"
		reward_check1 = reward_check2
		second_output = semi_gradient_dp_fcann(mdp, γ, typemax(Int64), step_interval, feature_vector, update_feature_vector!, layers; parameters = parameters, kwargs...)
		reward_check2 = check_reward_progress(second_output.episode_rewards)
		episode_rewards = vcat(episode_rewards, second_output.episode_rewards)
		@info "After pass number $pass for a scramble of $min_moves to $max_moves, average reward is $reward_check2"
	end
	@info "Concluded learning after $pass passes"

	return (value_function = second_output.value_function, episode_rewards = episode_rewards, final_reward = reward_check2, total_passes = pass, final_parameters = deepcopy(parameters), form_kwargs = second_output.form_kwargs)
end
  ╠═╡ =#

# ╔═╡ 436bd68b-e3f0-4e22-8898-d1c586f7d69d
#=╠═╡
function run_dp_curriculum_mastery_fcann_test!(parameters::FCANNParams{T}, min_moves::Integer, max_moves::Integer, feature_vector::Vector{T}, update_feature_vector!, step_interval, αlist::Vector{T}; kwargs...) where T<:Real
	sorted_αlist = sort(αlist; rev=true)
	dict_results = Dict(begin
	 	@info "Starting curriculum mastery for $num_moves moves"
		out = Dict(begin
		 @info "Using learning rate of α = $α"
		 inner_out = run_dp_step_mastery_fcann_test!(parameters, num_moves, feature_vector, update_feature_vector!, step_interval; α = α, kwargs...) 
		α => inner_out 
		end
		for α in αlist)
		num_moves => out
	 end
	 for num_moves in min_moves:max_moves)
	final_results = dict_results[max_moves][last(sorted_αlist)]
	(dict_results = dict_results, final_results = final_results)
end
  ╠═╡ =#

# ╔═╡ 9e25ea7c-c3ff-4dae-873b-3decf08b4625
function get_scramble_statistic(scramble::Integer, π::Function, ntrials::Integer)
	1:ntrials |> Map(i -> runepisode(make_reset_mdp(scramble, scramble); π = π) |> out -> out[1][1].cube == solved_cube_indices ? 1f0 : out[3][end]) |> foldxt(+) |> x -> x / ntrials
end

# ╔═╡ 0d8fcded-38b2-40fc-a4d5-0da61767cc2d
function get_scramble_statistics(output, min_scramble::Integer, max_scramble::Integer; ntrials = 1000) 
	[(n = n, solve_rate = get_scramble_statistic(n, output, ntrials)) for n in min_scramble:max_scramble]
end

# ╔═╡ ded411aa-2927-4435-9e3e-3f01ae35b78c
#=╠═╡
function display_learning_output(output; min_scramble = 1, max_scramble = 15)
	reward_plot = output.episode_rewards |> v -> plot_rewards(v, 100, 1000)
	stats = get_scramble_statistics(output, min_scramble, max_scramble)
	@htl("""
	<div style = "display: flex;">
	<div style = "width: 75%">
	$reward_plot
	</div>
	$(DataFrame(stats))
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ ac46c949-a0b5-42fc-bcf0-707faf7c0742
#=╠═╡
display_learning_output(test_sarsa_output)
  ╠═╡ =#

# ╔═╡ 3479a6e6-91ca-4be6-9d0a-0d720b1f04dd
#=╠═╡
display_learning_output(test_dp_output)
  ╠═╡ =#

# ╔═╡ c07cc5dd-5ae5-4887-b7bd-c5a4c95ffa0b
#=╠═╡
display_learning_output(test_sarsa_fcann_output)
  ╠═╡ =#

# ╔═╡ 4e9029e2-1c7c-4fc6-901d-c10a617c3cc0
#=╠═╡
function display_reversible_solution(output::NamedTuple; nsmooth = 100, npoints = 1000, max_scramble = 10, kwargs...)
	p1 = try 
		plot_rewards(log.(output.error_history), nsmooth, npoints)
		catch
			try
				plot_rewards(log.(output.error_history.errors), nsmooth, npoints)
				catch
				plot_rewards(log.(output.episode_history.errors), nsmooth, npoints)
			end
	end

	π = form_reversible_policy(output)

	tbl = get_scramble_statistics(π, 1, max_scramble; kwargs...) |> DataFrame

	@htl("""
		 <div style = "display: flex;">
		 <div style = "width: 70%;">
		 $p1
	     </div>
		 $tbl
		 </div>
		 """)
end
  ╠═╡ =#

# ╔═╡ 7251946d-0899-465f-ba00-aa1c623b74da
testepisode = runepisode(make_reset_mdp(2, 2); π = s -> test_sarsa_output.value_function(s; test_sarsa_output.form_kwargs()...).maximizing_action)

# ╔═╡ 598a04c6-b290-4758-9480-e1704b9326a4
testepisode[1][1].cube == solved_cube_indices

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
PlutoPlotly.heatmap(;plot_feature_changes(50)..., colorscale = "Greys", showscale=false) |> tr -> plot(tr, Layout(xaxis_title = "vector index", yaxis_title = "rotations", title = "Feature Vector Changing Through Random Moves"))
  ╠═╡ =#

# ╔═╡ fa2286f8-7b20-4f33-8f5a-bccacbe758e9
#builds a list of all unique cube states that are within n moves of the solved state.  Any state that is outside of this set can be considered terminal with a losing condition, need to test to see how large n can be before the problem is unweildy
function update_nmove_list!(statelist::Set{SVector{48, UInt8}}, s0::Vector{UInt8}, nmoves::Integer)
	push!(statelist, SVector{48}(s0))
	nmoves == 0 && return nothing
	for i_a in eachindex(rubiks_moves)
		s′ = rotate_cube(s0, i_a)
		update_nmove_list!(statelist, s′, nmoves - 1)
	end
	return statelist
end

# ╔═╡ 23e34ba4-006f-4100-a4a8-c14f6c33e887
function extend_statelist!(statelist::Set{SVector{48, UInt8}})
	new_statelist = Set{SVector{48, UInt8}}()
	for s0 in statelist
		for i_a in eachindex(rubiks_moves)
			s′ = rotate_cube(Vector(s0), i_a)
			push!(new_statelist, SVector{48}(s′))
		end
	end
	union!(statelist, new_statelist)
	return statelist
end

# ╔═╡ ef39bdd3-c085-441c-a8cd-6c1221ad4b21
function build_nmove_list(nmoves::Integer; s0 = copy(solved_cube_indices)) 
	haskey(nmove_lookup, nmoves) && return nmove_lookup[nmoves]
	statelist = if haskey(nmove_lookup, nmoves-1)
		statelist = deepcopy(nmove_lookup[nmoves-1])
		extend_statelist!(statelist)
	else
		update_nmove_list!(Set{SVector{48, UInt8}}(), s0, nmoves)
	end
	nmove_lookup[nmoves] = deepcopy(statelist)
end

# ╔═╡ 725317cb-e761-49f3-9c03-a15bf9c34da9
function build_tabular_rubiks_mdp(nmoves::Integer)
	@info "Building list of cube states"
	statelist = collect(build_nmove_list(nmoves))
	@info "Done building list of $(length(statelist)) states for $nmoves moves"
	@info "Building state index map"
	state_index_map = build_lookup(statelist)
	@info "Done building state index map"
	nstates = length(state_index_map)
	@info "Allocating matrices for transition maps"
	state_transition_map = zeros(Int64, eachindex(rubiks_moves), nstates)
	reward_transition_map = zeros(Float32, eachindex(rubiks_moves), nstates)
	i_s_term = state_index_map[SVector{48}(solved_cube_indices)]
	@info "Building state and reward transition maps"
	@threads for s in statelist
		i_s = state_index_map[s]
		if i_s == i_s_term
			state_transition_map[:, i_s] .= i_s
		else
			s′ = copy(solved_cube_indices)
			s_vec = Vector(s)
			for i_a in eachindex(rubiks_moves)
				rotate_cube!(s′, s_vec, i_a)
				(r, i_s′) = haskey(state_index_map, s′) ? (-1f0, state_index_map[s′]) : (-nmoves - 2f0, i_s_term)
				state_transition_map[i_a, i_s] = i_s′
				reward_transition_map[i_a, i_s] = r
			end
		end
	end
	TabularMDP(statelist, rubiks_moves, TabularDeterministicTransition(state_transition_map, reward_transition_map), () -> rand(eachindex(statelist)); state_index = state_index_map)
end

# ╔═╡ a38e34b1-57ac-496e-be00-73501b865b6f
function rubiks_nstep_move(cube::Vector{UInt8}, move::NTuple{N, Int64}; cube′::Vector{UInt8} = copy(cube), cube′′::Vector{UInt8} = copy(cube)) where N
	reward = 0f0
	cube′ .= cube
	for i_a in move
		rotate_cube!(cube′′, cube′, i_a)
		if cube′′ == solved_cube_indices 
			reward = 1f0
			break
		end
		cube′ .= cube′′
	end
	(reward, cube′′)
end

# ╔═╡ 194595ee-5b5d-4b97-a48a-bba10a491fdd
function rubiks_nstep_continuing_move(s::@NamedTuple{cube::Vector{UInt8}, scramble_moves::Int64, move_count::Int64}, move::NTuple{N, Int64}; min_moves::Integer = 1, max_moves::Integer = 20, kwargs...) where N
	scramble_moves = rand(min_moves:max_moves)
	(s.cube == solved_cube_indices) && return (0f0, (cube = initialize_rubiks_cube(scramble_moves), scramble_moves = scramble_moves, move_count = 0))
	(s.move_count > 2*s.scramble_moves) && return (0f0, (cube = initialize_rubiks_cube(scramble_moves), scramble_moves = scramble_moves, move_count = 0))
	(s.move_count > 50) && return (0f0, (cube = initialize_rubiks_cube(scramble_moves), scramble_moves = scramble_moves, move_count = 0))
	(reward, cube′) = rubiks_nstep_move(s.cube, move; kwargs...)
	(cube′ == solved_cube_indices) && return (1f0, (cube = initialize_rubiks_cube(scramble_moves), scramble_moves = scramble_moves, move_count = 0))
	return (reward, (cube = cube′, scramble_moves = s.scramble_moves, move_count = s.move_count + N))
end

# ╔═╡ 96223544-4478-41ab-aba5-fcde3cac0768
#create a cube mdp which saves in the state the number of initial scramble moves as well as how many moves have been attempted since the initial scramble
function make_rubiks_nstep_continuing_mdp(min_moves::Integer, max_moves::Integer, moves_per_step::Integer)
	initialize_state(;nmoves::Integer = rand(min_moves:max_moves)) = (cube = initialize_rubiks_cube(nmoves), scramble_moves = nmoves, move_count = 0)

	!haskey(rubiks_nstep_moves, moves_per_step) && error("Have not computed these actions ahead of time")
	actions, action_index = rubiks_nstep_moves[moves_per_step]

	step(s, i_a; kwargs...) = rubiks_nstep_continuing_move(s, actions[i_a]; min_moves = min_moves, max_moves = max_moves, kwargs...)
	ptf = StateMDPTransitionDeterministic(step, initialize_state(;nmoves=1))
	
	StateMDP(actions, ptf, initialize_state, Returns(false); action_index = action_index)
end

# ╔═╡ 9d70d00d-39ba-430b-8a78-17c6dc4b6cb2
begin
	function rubiks_nstep_nonlinear_value_training(hidden_layers::Vector{Int64}, reslayers::Integer, nstep::Integer, training_steps::Integer; show_message = false, use_gpu = false)
		feature_vector = use_gpu ? zeros(Float32, 48*48) : deepcopy(rubiks_binary_feature)
		setup = setup_value_nonlinear_training("rubiks_cube_$(nstep)step_fcann", false, make_rubiks_nstep_continuing_mdp(2, 2, nstep), feature_vector, update_rubiks_feature!; show_message = show_message)
		setup.train(hidden_layers, reslayers, 0f0, 0f0, training_steps; use_dp = true, use_gpu = use_gpu, new_params = false)
	end

	function rubiks_nstep_nonlinear_policy_training(hidden_layers::Vector{Int64}, reslayers::Integer, nstep::Integer, training_steps::Integer; show_message = false, use_gpu = false)
		feature_vector = use_gpu ? zeros(Float32, 48*48) : deepcopy(rubiks_binary_feature)
		setup = setup_policy_nonlinear_training("rubiks_cube_$(nstep)step_fcann", false, make_rubiks_nstep_continuing_mdp(2, 2, nstep), feature_vector, update_rubiks_feature!; show_message = show_message)
		setup.train(hidden_layers, reslayers, 0f0, 0f0, 0f0, 0f0, training_steps; use_gpu = use_gpu, new_params = false)
	end
		
	function rubiks_nstep_nonlinear_value_training(α::Float32, λ::Float32, hidden_layers::Vector{Int64}, reslayers::Integer, min_scramble::Integer, max_scramble::Integer, nstep::Integer, training_steps::Integer; show_message = false, use_gpu = false, kwargs...)
		feature_vector = use_gpu ? zeros(Float32, 48*48) : deepcopy(rubiks_binary_feature)
		setup = setup_value_nonlinear_training("rubiks_cube_$(nstep)step_fcann", false, make_rubiks_nstep_continuing_mdp(min_scramble, max_scramble, nstep), feature_vector, update_rubiks_feature!; show_message = show_message)
		output = setup.train_rate_decay(hidden_layers, reslayers, α, λ, training_steps; use_dp = true, use_gpu = use_gpu, kwargs...)
		setup.save_params(; show_message = show_message)
		return output
	end

	function rubiks_nstep_nonlinear_policy_training(α_θ::Float32, α_w::Float32, λ_θ::Float32, λ_w::Float32, hidden_layers::Vector{Int64}, reslayers::Integer, min_scramble::Integer, max_scramble::Integer, nstep::Integer, training_steps::Integer; show_message = false, use_gpu = false, kwargs...)
		feature_vector = use_gpu ? zeros(Float32, 48*48) : deepcopy(rubiks_binary_feature)
		setup = setup_policy_nonlinear_training("rubiks_cube_$(nstep)step_fcann", false, make_rubiks_nstep_continuing_mdp(min_scramble, max_scramble, nstep), feature_vector, update_rubiks_feature!; show_message = show_message)
		output = setup.train_rate_decay(hidden_layers, reslayers, α_θ, α_w, λ_θ, λ_w, training_steps; use_gpu = use_gpu, kwargs...)
		setup.save_params(; show_message = show_message)
		return output
	end
end

# ╔═╡ 26a14497-12d2-4aec-b17c-72544ab23709
const dp_λ_2step_fcann_result = rubiks_nstep_nonlinear_value_training(4f-2, 0.1f0, fill(4096, 8), 1, value_scrambles_2_step..., 2, 1_000_000; use_gpu = true, ϵ = 0.001f0, α_r̄ = 0.1f0, l2 = 0.0f0, dropout = 0.0f0)

# ╔═╡ 6095a2c1-b33d-4f39-965d-c61e042bebe9
const ac_2step_fcann_result = rubiks_nstep_nonlinear_policy_training(2f-2, 2f-2, 0.5f0, 0.5f0, fill(1024, 8), 1, policy_scrambles_2_step..., 2, 1_000_000; use_gpu = true, α_r̄ = 0.1f0, l2 = 0.0f0)

# ╔═╡ c676f6ab-07cd-4e55-b853-ff7a13c03f80
rubiks_nstep_move(initialize_rubiks_cube(10), (2, 6))

# ╔═╡ 6f2724d4-dcf7-417b-966d-1ff8deeb2a65
#create a cube mdp which saves in the state the number of initial scramble moves as well as how many moves have been attempted since the initial scramble
function make_rubiks_nstep_mdp(min_moves::Integer, max_moves::Integer, moves_per_step::Integer)
	initialize_state(;nmoves::Integer = rand(min_moves:max_moves)) = (cube = initialize_rubiks_cube(nmoves), scramble_moves = nmoves, move_count = 0)
	
	function isterm(s)
		s.move_count > 50 && return true
		s.move_count > 2*s.scramble_moves && return true
		s.cube == solved_cube_indices
	end

	!haskey(rubiks_nstep_moves, moves_per_step) && error("Have not computed these actions ahead of time")
	actions, action_index = rubiks_nstep_moves[moves_per_step]

	step(s, i_a; kwargs...) = rubiks_nstep_move(s, actions[i_a]; kwargs...)
	ptf = StateMDPTransitionDeterministic(step, initialize_state(;nmoves=1))
	
	StateMDP(actions, ptf, initialize_state, isterm; action_index = action_index)
end

# ╔═╡ 440470e3-1f38-4041-b554-13a155f19cdf
const rubiks_2step_mdp = make_rubiks_nstep_mdp(1, 10, 2)

# ╔═╡ 174669a8-218c-49e8-a4d9-e4482d3850ae
@code_warntype rubiks_2step_mdp.ptf(initialize_rubiks_cube(10), rand(1:100))

# ╔═╡ d952334c-1953-49f5-8871-d96bfbfa64e3
const rubiks_3step_mdp = make_rubiks_nstep_mdp(1, 10, 3)

# ╔═╡ 07377754-fa04-4e5a-b15d-f6cfd9d52dab
function run_dp_λ_rubiks_nstep_linear_test(γ, λ, n, min_moves, max_moves; num_steps = 10_000, kwargs...)
	mdp =  make_rubiks_nstep_mdp(min_moves, max_moves, n)
	output = NonTabularRL.dp_λ_linear(mdp, γ, λ, typemax(Int64), num_steps, deepcopy(rubiks_binary_feature), update_rubiks_feature!; kwargs...)
	(mdp = mdp, output = output)
end

# ╔═╡ 56cf2e20-41ee-46fb-a5d4-0b79642ac3b9
const dp_λ_2step_result = run_dp_λ_rubiks_nstep_linear_test(0.9f0, 0.95f0, 2, 2, 5; num_steps = 50_000, α = 1f-2, ϵ = 0.001f0, trace_type = NonTabularRL.ReplacingTrace())

# ╔═╡ affd9578-11a2-4150-862a-aad1f7cfa565
function run_dp_λ_rubiks_nstep_linear_curriculum_test(γ, λ, steps_per_move, min_scramble, max_scramble; kwargs...)
	scramble = min_scramble
	output = run_dp_λ_rubiks_nstep_linear_test(γ, λ, steps_per_move, scramble, scramble+steps_per_move - 1; kwargs...).results
	results = Dict([scramble => output])
	scramble += 1
	while scramble ≤ max_scramble
		output = run_dp_λ_rubiks_nstep_linear_test(γ, λ, steps_per_move, scramble, scramble+steps_per_move - 1; parameters = output.final_parameters, kwargs...).results
		results[scramble] = output 
		scramble += 1
	end
	return results
end

# ╔═╡ bb6fea94-f0cc-46c9-90f3-4bed62b520c3
function run_sarsa_λ_rubiks_nstep_linear_test(γ, λ, n, min_moves, max_moves; num_steps = 10_000, kwargs...)
	mdp =  make_rubiks_nstep_mdp(min_moves, max_moves, n)
	output = NonTabularRL.sarsa_λ_linear(mdp, γ, λ, typemax(Int64), num_steps, deepcopy(rubiks_binary_feature), update_rubiks_feature!; kwargs...)
	(mdp = mdp, output = output)
end

# ╔═╡ dde9d10b-e259-4c86-98fd-d6459f36699b
const sarsa_λ_2step_result = run_sarsa_λ_rubiks_nstep_linear_test(0.9f0, 0.95f0, 2, 2, 5; num_steps = 100_000, α = 1f-2, ϵ = 0.01f0, trace_type = NonTabularRL.ReplacingTrace())

# ╔═╡ 74c9b56a-d890-4dca-a26e-a6d41f105cde
function get_nstep_statistics(nstep::Integer, output::NamedTuple, min_scramble, max_scramble; ntrials = 100)
	[(n = n, solve_rate = get_nstep_scramble_statistic(make_rubiks_nstep_mdp(n, n, nstep), n, output, ntrials)) for n in min_scramble:max_scramble]
end

# ╔═╡ fe719761-de47-4240-a50d-a1dccdf2d1e1
#=╠═╡
function display_nstep_output(nstep_result; nsmooth = 100, npoints = 1000, min_scramble = 1, max_scramble = 10, kwargs...)
	reward_plot = nstep_result.output.episode_rewards |> v -> plot_rewards(v, nsmooth, npoints)
	stats = get_nstep_statistics(nstep_result, min_scramble::Integer, max_scramble::Integer; kwargs...)
	@htl("""
	<div style = "display: flex;">
	<div style = "width = 0.75;">$reward_plot</div>
	$(DataFrame(stats))
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ b117af73-16a0-44cb-abfb-fd9df962bb97
#=╠═╡
display_nstep_output(sarsa_λ_2step_result)
  ╠═╡ =#

# ╔═╡ 1f0429c4-e6c7-4b09-8896-62fdf5569148
#=╠═╡
display_nstep_output(dp_λ_2step_result)
  ╠═╡ =#

# ╔═╡ ee8842f2-86af-495b-9b37-41aa59dd4cc1
#=╠═╡
function display_nstep_αdecay_output(results)
	αs = sort(collect(keys(results.output_dict)); rev = true)
	rewards = mapreduce(a -> results.output_dict[a].episode_rewards, vcat, αs)

	last_output = results.output_dict[last(αs)]
	result = (mdp = results.mdp, output = (value_function = last_output.value_function, episode_rewards = rewards, final_reward = last_output.final_reward, total_passes = sum(results.output_dict[a].total_passes for a in αs), final_parameters = last_output.final_parameters, form_kwargs = last_output.form_kwargs))
	@htl("""
	Showing results over the following learning rates: $(reduce((a, b) -> "$a, $b", αs)) for scrambles: $(results.min_scramble) to $(results.max_scramble) and $(result.output.total_passes) passes
		 
	$(display_nstep_output(result))
	""")
end
  ╠═╡ =#

# ╔═╡ ca57f9f0-4d9b-41d6-b7ec-931415b39641
#=╠═╡
display_nstep_αdecay_output(dp_step_mastery_2move_fcann_results2[1])
  ╠═╡ =#

# ╔═╡ 7f88d0d3-2069-4c97-a3a0-20196a7151a7
#=╠═╡
function display_nstep_continuing_output(n::Integer, output::NamedTuple; nsmooth = 100, npoints = 1000, min_scramble = 1, max_scramble = 10, kwargs...)
	reward_plot = output.reward_history |> v -> plot_rewards(v, nsmooth, npoints)
	stats = get_nstep_statistics(n::Integer, output, min_scramble::Integer, max_scramble::Integer; kwargs...)
	@htl("""
	<div style = "display: flex;">
	<div style = "width = 0.75;">$reward_plot</div>
	$(DataFrame(stats))
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ f0316e07-2873-4538-b7a5-01256b089561
#=╠═╡
display_nstep_continuing_output(2, dp_λ_2step_fcann_result; ntrials = 100)
  ╠═╡ =#

# ╔═╡ ef8c2a99-d462-4cb1-b433-d669cda52d31
function get_nstep_policy_statistics(nstep::Integer, output::NamedTuple, min_scramble, max_scramble; ntrials = 100)
	[(n = n, solve_rate = get_nstep_policy_scramble_statistic(make_rubiks_nstep_mdp(n, n, nstep), n, output, ntrials)) for n in min_scramble:max_scramble]
end

# ╔═╡ 273923d3-ab5e-4899-80bd-f257fa5c1d3c
#=╠═╡
function display_nstep_continuing_policy_output(n::Integer, output::NamedTuple; nsmooth = 100, npoints = 1000, min_scramble = 1, max_scramble = 10, kwargs...)
	reward_plot = output.reward_history |> v -> plot_rewards(v, nsmooth, npoints)
	stats = get_nstep_policy_statistics(n::Integer, output, min_scramble::Integer, max_scramble::Integer; kwargs...)
	@htl("""
	<div style = "display: flex;">
	<div style = "width = 0.75;">$reward_plot</div>
	$(DataFrame(stats))
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ b760fbbf-5e07-446f-9b37-6e184868dfdd
#=╠═╡
display_nstep_continuing_policy_output(2, ac_2step_fcann_result)
  ╠═╡ =#

# ╔═╡ 341ec41b-bc42-4548-b11a-f9b326179422
function run_dp_λ_rubiks_nstep_fcann_test(γ, λ, n, min_moves, max_moves, layers; num_steps = 10_000, feature_vector = rubiks_binary_feature, kwargs...)
	mdp =  make_rubiks_nstep_mdp(min_moves, max_moves, n)
	output = NonTabularRL.dp_λ_fcann(mdp, γ, λ, typemax(Int64), num_steps, deepcopy(feature_vector), update_rubiks_feature!, layers; kwargs...)
	(mdp = mdp, output = output)
end

# ╔═╡ cde3b13f-86e0-46ad-8a74-84684cc32811
#=╠═╡
function run_dp_step_mastery_nmove_fcann_test!(parameters::FCANNParams{T}, min_moves::Integer, max_moves::Integer, moves_per_step::Integer, feature_vector, update_feature_vector!, step_interval; γ::T = 0.9f0, kwargs...) where T<:Real
	function check_reward_progress(episode_rewards::Vector{T}) 
		l = length(episode_rewards)
		episode_check = max(1000, ceil(Int64, l/2))
		mean(episode_rewards[max(1, l-episode_check):l])
	end
	params = copy(parameters)
	layers = get_hidden_layers(parameters)
	mdp = make_rubiks_nstep_mdp(min_moves, max_moves, moves_per_step)
	first_output = semi_gradient_dp_fcann(mdp, γ, typemax(Int64), step_interval, feature_vector, update_feature_vector!, layers; parameters = parameters, kwargs...)
	reward_check1 = check_reward_progress(first_output.episode_rewards)
	@info "After first learning pass for a scramble of $min_moves to $max_moves, average reward is $reward_check1"

	second_output = semi_gradient_dp_fcann(mdp, γ, typemax(Int64), step_interval, feature_vector, update_feature_vector!, layers; parameters = parameters, kwargs...)
	reward_check2 = check_reward_progress(second_output.episode_rewards)
	@info "After second learning pass for a scramble of $min_moves to $max_moves, average reward is $reward_check2"

	episode_rewards = first_output.episode_rewards
	pass = 2
	while reward_check2 > reward_check1
		copy!(params, parameters)
		pass += 1
		@info "Reward still improving so proceeding with pass number $pass"
		reward_check1 = reward_check2
		first_output = second_output
		episode_rewards = vcat(episode_rewards, first_output.episode_rewards)
		second_output = semi_gradient_dp_fcann(mdp, γ, typemax(Int64), step_interval, feature_vector, update_feature_vector!, layers; parameters = parameters, kwargs...)
		reward_check2 = check_reward_progress(second_output.episode_rewards)
		
		@info "After pass number $pass for a scramble of $min_moves to $max_moves, average reward is $reward_check2"
	end
	@info "Concluded learning after $pass passes"
	copy!(parameters, params)
	output = (value_function = first_output.value_function, episode_rewards = episode_rewards, final_reward = reward_check1, total_passes = pass, final_parameters = copy(parameters), form_kwargs = first_output.form_kwargs)
	return (mdp = mdp, output = output)
end
  ╠═╡ =#

# ╔═╡ b9586038-58ca-467b-8cd5-f5a16663682e
#=╠═╡
function run_dp_α_decay_step_mastery_nmove_fcann_test!(parameters, min_moves, max_moves, moves_per_step, feature_vector, update_feature_vector!, step_interval, α0::T; decay = T(0.5), kwargs...) where T<:Real
	α = α0
	@info "Beginning training with learning rate $α"
	(mdp, output1) = run_dp_step_mastery_nmove_fcann_test!(parameters, min_moves, max_moves, moves_per_step, feature_vector, update_feature_vector!, step_interval; α = α, kwargs...)

	results = Dict([α => output1])

	α *= decay
	@info "Training second round with learning rate $α"
	(mdp, output2) = run_dp_step_mastery_nmove_fcann_test!(parameters, min_moves, max_moves, moves_per_step, feature_vector, update_feature_vector!, step_interval; α = α, kwargs...)
	results[α] = output2

	reward_check2 = output2.final_reward
	reward_check1 = output1.final_reward
	while reward_check2 > reward_check1
		reward_check1 = reward_check2
		α *= decay
		@info "Training next round with learning rate $α"
		(mdp, output) = run_dp_step_mastery_nmove_fcann_test!(parameters, min_moves, max_moves, moves_per_step, feature_vector, update_feature_vector!, step_interval; α = α, kwargs...)
		reward_check2 = output.final_reward
		results[α] = output
	end
	@info "Concluded rate decay with learning rate $(α*2)"
	return (mdp = mdp, output_dict = results, min_scramble = min_moves, max_scramble = max_moves)
end
  ╠═╡ =#

# ╔═╡ a56e5eb6-7ec3-45ce-b93d-2230691652b2
function make_scramble_trajectory(num_actions)
	cube = copy(solved_cube_indices)
	cube′ = copy(cube)
	states = Vector{typeof(cube)}(undef, num_actions)
	actions = Vector{Int64}(undef, num_actions)
	i_a = rand(eachindex(rubiks_moves))
	for i in 1:num_actions
		actions[i] = i_a
		rotate_cube(cube, i_a; cube′ = cube′)
		cube .= cube′
		states[i] = copy(cube)
		i_a = rand(valid_move_inds[i_a])
	end
	return states, actions
end

# ╔═╡ d41c9ab0-0712-4960-a2dc-aaf3661e0866
test_trajectory = make_scramble_trajectory(30)

# ╔═╡ 00609e53-d6bd-4630-9a28-8e69acc10901
length(unique(test_trajectory[1]))

# ╔═╡ 4aed6452-26e6-4a48-9a3a-b1be7d961157
π_unscramble_test = make_unscramble_policy(test_trajectory...)

# ╔═╡ fc8a9f05-2b66-4ef2-8e05-538f66b4b840
runepisode(rubiks_cube_mdp; s0 = test_trajectory[1][end], π = π_unscramble_test)

# ╔═╡ 4828ae7b-2f70-4b17-94f0-06505fac7887
function initialize_reversible_cube(num_moves::Integer)
	(states, actions) = make_scramble_trajectory(num_moves)
	π = make_unscramble_policy(states, actions)
	cube = states[end]
	(cube = cube, π = π)
end

# ╔═╡ 86ef6b80-f902-48da-9032-fca221c3ed6f
initialize_reversible_cube(30)

# ╔═╡ fe415c36-f2b2-4b25-9ab2-33011fdc9f28
function rubiks_reversible_move(cube::Vector{UInt8}, i_a::Integer; kwargs...)
	cube′ = rotate_cube(cube, i_a; kwargs...)
	r = 2*Float32(cube′ == solved_cube_indices) - 1f0/8 #this produces values roughly from -1.75 to 1.75 for states from 30 moves away to the solved state
	(r, cube′)
	# (-1f0/15, cube′)
end

# ╔═╡ af0c2dba-c7ee-4f75-a31d-1bc1dffd3e84
function rubiks_reversible_mrp_move(s::NamedTuple; kwargs...) 
	i_a = s.π(s.cube)
	(r, cube′) = rubiks_reversible_move(s.cube, i_a; kwargs...)
	return (r, (cube = cube′, π = s.π))
end

# ╔═╡ dd14d0ff-5163-43da-b1be-3fc420682fe3
const rubiks_reversible_mrp_transition = StateMRPTransitionSampler(rubiks_reversible_mrp_move, initialize_reversible_cube(1))

# ╔═╡ 46749d40-8c07-43dc-bbff-013797fd56cd
function make_rubiks_reversible_mrp(num_scramble::Integer)
	StateMRP(rubiks_reversible_mrp_transition, () -> initialize_reversible_cube(num_scramble), s -> isequal(s.cube, solved_cube_indices))
end

# ╔═╡ 242a57ef-7e7e-4439-95cb-9f747b23d6b3
const rubiks_reversible_mrp = make_rubiks_reversible_mrp(7)

# ╔═╡ 00c67f96-456c-48d1-9d1b-0fcd8ddac671
const reversible_linear_td_solution = semi_gradient_td0_estimation_linear(rubiks_reversible_mrp, 1f0, 400_000, typemax(Int64), deepcopy(rubiks_binary_feature), make_reversible_feature_update(update_rubiks_feature!); α = 1f-4)

# ╔═╡ 51105316-c191-44a5-b4f3-703f30efebad
#=╠═╡
display_reversible_solution(reversible_linear_td_solution)
  ╠═╡ =#

# ╔═╡ 9fffc847-87c2-41d6-9f87-d10d645708da
const reversible_fcann_td_solution = semi_gradient_td0_estimation_fcann(rubiks_reversible_mrp, 1f0, 1_000, typemax(Int64), deepcopy(rubiks_binary_feature), make_reversible_feature_update(update_rubiks_feature!), reversible_fcann_td_layers; α = 1f-5, params = reversible_fcann_td_params)

# ╔═╡ 42106894-5ae2-4a1e-a26e-0e8e7085d035
#=╠═╡
display_reversible_solution(reversible_fcann_td_solution)
  ╠═╡ =#

# ╔═╡ 99a280b6-a371-43ac-9a27-319501a47d96
const rubiks_reversible_transition = StateMDPTransitionSampler(rubiks_reversible_move, initialize_reversible_cube(10))

# ╔═╡ cb8d0b4a-39de-4a18-8048-7a525a6bfca8
function make_rubiks_reversible_mdp(num_scramble::Integer)
	StateMDP(rubiks_moves, rubiks_reversible_transition, () -> initialize_reversible_cube(num_scramble), s -> isequal(s.cube, solved_cube_indices))
end

# ╔═╡ 717209cb-24de-4fae-b434-bf8258f39fc6
const rubiks_reversible_mdp = make_rubiks_reversible_mdp(7)

# ╔═╡ 36f51608-7d65-4f21-ad71-98707f94b12a
runepisode(rubiks_reversible_mdp; π = π_cube_reverse)

# ╔═╡ 9d87dbb3-1a3c-47d3-a090-4eb76a5c6c2d
const reversible_linear_mc_solution = gradient_monte_carlo_policy_estimation_linear(rubiks_reversible_mdp, π_cube_reverse, 1f0, 400_000, deepcopy(rubiks_binary_feature), make_reversible_feature_update(update_rubiks_feature!); α = 4f-6)

# ╔═╡ a3555ab8-63b0-451f-a91d-995d7d8632bf
#=╠═╡
display_reversible_solution(reversible_linear_mc_solution)
  ╠═╡ =#

# ╔═╡ 9ffbb5d0-defa-4734-883e-7ff1f86e326b
const reversible_fcann_mc_solution = gradient_monte_carlo_policy_estimation_fcann(rubiks_reversible_mdp, π_cube_reverse, 1f0, 10_000, deepcopy(rubiks_binary_feature), make_reversible_feature_update(update_rubiks_feature!), reversible_fcann_layers; α = 1f-5, params = reversible_fcann_params)

# ╔═╡ af9571f9-9d82-4563-83b1-d3dc23668b6c
#=╠═╡
display_reversible_solution(reversible_fcann_mc_solution; ntrials = 100)
  ╠═╡ =#

# ╔═╡ 1a0d2678-c228-40a3-a376-01df24292556
function get_pocket_rotation_indices(::RubiksMove{F, Clockwise}) where F<:Face
	rotation_mapping = pocket_clockwise_rotation_mapping[F]
	
	fnum = face2value[F]
	cube = reshape(1:24, 4, 6)
	cube′ = copy(cube)

	#rotate face colors
	@inbounds @simd for i in 1:4
		cube′[i, fnum] = cube[pocket_clockwise_perm[i], fnum]
	end

	#rotate other colors
	for i in 2:4
		(origin_face, origin_inds) = rotation_mapping[i-1]
		(destination_face, destination_inds) = rotation_mapping[i]
		@inbounds @simd for j in 1:2
			cube′[destination_inds[j], destination_face] = cube[origin_inds[j], origin_face]
		end
	end
	(origin_face, origin_inds) = rotation_mapping[4]
	(destination_face, destination_inds) = rotation_mapping[1]
	@inbounds @simd for j in 1:2
		cube′[destination_inds[j], destination_face] = cube[origin_inds[j], origin_face]
	end
	return cube′[:]
end

# ╔═╡ 8e44a35b-ed6c-4a9e-9f68-9fe28aa5af22
const pocket_rotation_lookup = [get_pocket_rotation_indices(m) for m in pocket_moves]

# ╔═╡ a7ed00c6-6ada-41ee-9cbc-770c3eed776c
begin
	#always produce a copy of the cube since I plan to use this to generate an exhaustive list anyway
	function rotate_pocket_cube(cube::SVector{24, UInt8}, i_a::Integer)
		indices = pocket_rotation_lookup[i_a]
		SVector{24}(cube[indices[i]] for i in eachindex(cube))
	end
	rotate_pocket_cube(cube::SVector{UInt8}, m::RubiksMove) = rotate_pocket_cube(cube, pocket_move_index[m])
end

# ╔═╡ c5c1942a-86ff-4494-9e38-3ad0bc6c0517
rotate_pocket_cube(solved_pocket_cube, 1)

# ╔═╡ 76e07bdb-82c7-4dc8-893d-e7ca3d0d318a
function permute_cube!(s′′::Vector{UInt8}, s′, s0, m::Tuple{Int64, Int64})
	inds = pocket_rotation_lookup[m[1]]
	s′ .= s0
	s′′ .= s0
	for _ in 1:m[2]
		@inbounds for i in eachindex(inds)
			s′′[i] = s′[inds[i]]
		end
		s′ .= s′′
	end
end

# ╔═╡ 2968db78-f04a-47ea-b5a7-8dd383406c7e
function check_next_moves_recur(trajectory::Vector{Tuple{Int64, Int64}}, cubes::Vector{Vector{UInt8}}, cube′::Vector{UInt8}, depth::Integer, best_depth::Vector{Int64}, best_trajectory::Vector{Tuple{Int64, Int64}}, best_cubes::Vector{Vector{UInt8}}, maxdepth::Integer)

	threshold = min(maxdepth, best_depth[1])
	misplaced = count_misplaced(cubes[depth])
	heuristic = ceil(Int64, misplaced / 12) #this value is a lower bound on the number of remaining moves needed to solve
	
	(depth + heuristic >= threshold) && return nothing

	if (cubes[depth] == solved_pocket_cube)
		@info "Found a new best solution with $depth moves"
		for i in 1:depth
			best_trajectory[i] = trajectory[i]
			best_cubes[i] .= cubes[i]
		end
		best_depth[1] = depth
		return nothing
	end
	
	next_moves = get_next_pocket_moves(trajectory[depth])
	for m in next_moves
		permute_cube!(cubes[depth+1], cube′, cubes[depth], m)
		trajectory[depth + 1] = m
		check_next_moves_recur(trajectory, cubes, cube′, depth+1, best_depth, best_trajectory, best_cubes, maxdepth)
	end
end

# ╔═╡ 6aade30b-6784-495a-b559-a3483cb7dc49
function check_next_moves_recur2(current_cube, trajectory::Vector{Tuple{Int64, Int64}}, cubes::Vector{Vector{UInt8}}, cube′::Vector{UInt8}, depth::Integer, threshold::Integer, states_checked)
	if (current_cube == solved_pocket_cube)
		@info "Found a solution with $depth moves after checking $(states_checked[1]) states"
		return (true, trajectory[1:depth])
	end
	
	misplaced = count_misplaced(current_cube)
	heuristic = ceil(Int64, misplaced / 12) #this value is a lower bound on the number of remaining moves needed to solve
	f = heuristic + depth
	
	(depth + heuristic > threshold) && return (false, f)
	
	min_overshoot = typemax(Int64)
	next_moves = depth == 0 ? all_pocket_moves : get_next_pocket_moves(trajectory[depth])

	for m in next_moves
		permute_cube!(cubes[depth+1], cube′, current_cube, m)
		trajectory[depth + 1] = m
		states_checked[1] += 1
		(found, result) = check_next_moves_recur2(cubes[depth+1], trajectory, cubes, cube′, depth+1, threshold, states_checked)
		found && return (true, result)
		min_overshoot = min(min_overshoot, result)
	end

	return (false, min_overshoot)
end

# ╔═╡ 2e4fa908-f1c4-4594-a0cf-782cf6513208
function check_next_moves2(cube::SVector{24, UInt8}, cube′::Vector{UInt8}, maxdepth::Integer)
	cube == solved_pocket_cube && return ([0], Vector{Tuple{Int64, Int64}}(), [Vector(cube)])
	trajectory = fill((1, 1), 11)
	best_trajectory = copy(trajectory)
	cubes = [Vector(cube) for i in 1:11]

	misplaced = count_misplaced(cube)
	threshold = ceil(Int64, misplaced / 12)
	states_checked = [0]

	while threshold <= maxdepth
		@info "Starting search round with threshold: $threshold"
		found, result = check_next_moves_recur2(cube, trajectory, cubes, cube′, 0, threshold, states_checked)
		if found
			l = length(result)
			return (l, result, cubes[1:l], states_checked[1])
		else
			threshold = result
		end
	end
	@info "No solution found within maximum depth of $maxdepth"
	# return (best_depth, best_trajectory[1:best_depth[1]], best_cubes[1:best_depth[1]])
end

# ╔═╡ 435df0bb-ac20-4b6a-9575-03cd249b604c
function solve_pocket_cube_ida_star(s::SVector{24, UInt8}; maxdepth::Integer = 12)
	cube′ = Vector(s)
	check_next_moves2(s, cube′, maxdepth)
end

# ╔═╡ de5ea355-4ddc-46a1-8a55-a2f42a00d3a3
function check_next_moves(cube::SVector{24, UInt8}, cube′::Vector{UInt8}, maxdepth::Integer)
	cube == solved_pocket_cube && return ([0], Vector{Tuple{Int64, Int64}}(), [Vector(cube)])
	best_depth = [typemax(Int64)]
	trajectory = fill((1, 1), 11)
	best_trajectory = copy(trajectory)
	depth = 0
	cubes = [Vector(cube) for i in 1:11]
	best_cubes = deepcopy(cubes)
	
	for m in all_pocket_moves
		permute_cube!(cubes[1], cube′, cube, m)
		trajectory[1] = m
		check_next_moves_recur(trajectory, cubes, cube′, 1, best_depth, best_trajectory, best_cubes, maxdepth)
	end
	return (best_depth, best_trajectory[1:best_depth[1]], best_cubes[1:best_depth[1]])
end

# ╔═╡ 71dae97e-6df2-472a-81ef-663017d7b23e
function solve_pocket_cube_exhaustive(s::SVector{24, UInt8}; maxdepth::Integer = 12)
	cube′ = Vector(s)
	check_next_moves(s, cube′, maxdepth)
end

# ╔═╡ e0d2df20-2781-4589-9102-7189499536e7
# NOTE: The following constants and functions are assumed to be defined elsewhere,
# as they were in the original code:
#
# const solved_pocket_cube::SVector{24, UInt8}
# const all_pocket_moves::Vector{Tuple{Int64, Int64}}
# function get_next_pocket_moves(last_move::Tuple{Int64, Int64})::Vector{Tuple{Int64, Int64}}
# function permute_cube!(new_cube, buffer, old_cube, move)

"""
Performs an exhaustive, iterative depth-first search to find the shortest
solution for a pocket cube.

This function replaces the original recursive implementation with a stack-based
approach to avoid deep recursion and potential stack overflows.
"""
function solve_pocket_cube_exhaustive_iterative(s::SVector{24, UInt8}; maxdepth::Integer = 11)
	# If the cube is already solved, return immediately.
	if s == solved_pocket_cube
		return ([0], Tuple{Int64, Int64}[], [Vector(s)])
	end

	# --- State Initialization ---
	best_depth = [typemax(Int64)]
	best_trajectory = Vector{Tuple{Int64, Int64}}(undef, maxdepth)
	
	# Pre-allocate memory for cube states at each depth
	# cubes[d] will hold the cube state at depth d.
	cubes = [Vector{UInt8}(undef, 24) for _ in 1:(maxdepth + 1)]
	cube′ = similar(cubes[1]) # A temporary buffer for permutation

	# --- Iterative Search Setup ---
	# The stack will hold stateful iterators for the moves at each depth.
	stack = Vector{Base.Iterators.Stateful{Vector{Tuple{Int64, Int64}}, Union{Nothing, Tuple{Tuple{Int64, Int64}, Int64}}}}()
	
	# trajectory[d] stores the move taken at depth d.
	trajectory = Vector{Tuple{Int64, Int64}}(undef, maxdepth + 1)
	depth = 0

	# --- Main Search Loop ---
	# Start the search by "descending" to depth 1.
	depth = 1
	push!(stack, Iterators.Stateful(all_pocket_moves))

	while !isempty(stack)
		# Get the iterator for the moves at the current depth.
		move_iterator = last(stack)

		# If the iterator is exhausted, we have explored all paths from this node.
		# So, we backtrack by "returning" to the previous depth.
		if isempty(move_iterator)
			pop!(stack)
			depth -= 1
			continue
		end

		# Process the next available move at the current depth.
		move = popfirst!(move_iterator)
		trajectory[depth] = move
		
		# Determine the parent cube state to apply the move to.
		parent_cube = (depth == 1) ? s : cubes[depth - 1]
		
		# Apply the move to get the new cube state.
		permute_cube!(cubes[depth], cube′, parent_cube, move)

		# Check if this new state is the solved state.
		if cubes[depth] == solved_pocket_cube
			# If this solution is the best one found so far, record it.
			if depth < best_depth[1]
				@info "Found a new best solution with $depth moves"
				best_depth[1] = depth
				view(best_trajectory, 1:depth) .= view(trajectory, 1:depth)
			end
			# Prune this branch; no need to search deeper from a solved state.
			continue
		end
		
		# --- Descend to the Next Level ---
		# If we haven't hit maxdepth and this path is still shorter than our
		# best solution, we descend deeper into the search tree.
		if depth < maxdepth && depth < best_depth[1]
			depth += 1
			next_moves = get_next_pocket_moves(trajectory[depth-1])
			push!(stack, Iterators.Stateful(next_moves))
		end
	end

	# --- Format and Return the Result ---
	if best_depth[1] == typemax(Int64)
		# No solution was found within the given maxdepth.
		return (best_depth, Tuple{Int64, Int64}[], Vector{UInt8}[])
	else
		# A solution was found; reconstruct the sequence of cube states.
		final_cubes = [Vector(s)]
		current_cube = Vector(s)
		for i in 1:best_depth[1]
			next_cube = similar(current_cube)
			move = best_trajectory[i]
			permute_cube!(next_cube, cube′, current_cube, move)
			push!(final_cubes, next_cube)
			current_cube = next_cube
		end
		return (best_depth, best_trajectory[1:best_depth[1]], final_cubes)
	end
end

# ╔═╡ eb5d2eac-5280-4269-8b2d-3c31226b4b2f
const test_episode = runepisode(rubiks_cube_mdp; max_steps = 1_000)

# ╔═╡ 6b410101-edb1-4814-8aa6-344d0c969e01
md"""
# Dependencies
"""

# ╔═╡ e956ffaa-01c3-4e44-8c9f-2298347fea03
# ╠═╡ skip_as_script = true
#=╠═╡
html"""
<style>
	main {
		margin: 0 auto;
		max-width: min(1600px, 90%);
		padding-left: max(50px, 10%);
		padding-right: max(10px, 5%);
		font-size: max(10px, min(24px, 2vw));
	}
</style>
"""
  ╠═╡ =#

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
	function render_cube(cube_values::Matrix{I}; square_pixels = 20, kwargs...) where I <: Integer
		@htl("""
		<div style = "display: flex;">
		$(mapreduce(add_elements, 1:6) do i
		@htl("""
		<div style = "margin-right: 5px; font-size: $(square_pixels*.8)px;">
		$(face_names[i])
		$(render_face(cube_values[:, i], i; square_pixels = square_pixels, kwargs...))
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

# ╔═╡ 156aff30-918f-4f69-8830-815920a94d3c
#=╠═╡
@htl("""
	 <div>
	 <h3> Solved Cube</h3>
	 $(render_cube(solved_cube_indices))
	 </div>

	 
	 <div>
	 <h3> 1 Move Scramble </h3>
	 <div style = "display: flex; flex-wrap: wrap;">
	 $([render_cube(s; square_pixels = 10) for s in build_nmove_list(1)])
	 </div>

	 <div>
	 <h3> 2 Move Scramble </h3>
	 <div style = "display: flex; flex-wrap: wrap;">
	 $([render_cube(s; square_pixels = 5) for s in build_nmove_list(2)])
	 </div>
	 </div>
	 """)
  ╠═╡ =#

# ╔═╡ da55b6bc-ba84-4800-bab4-c5bbfae73b99
#=╠═╡
render_cube(rubiks_nstep_move(solved_cube_indices, (5, 12))[2])
  ╠═╡ =#

# ╔═╡ 8b26a4ce-d853-45a6-b83d-9d07f30b6621
#=╠═╡
function show_rubiks_nstep_episode(mdp::StateMDP, output::NamedTuple, nscramble::Integer; s0 = mdp.initialize_state(;nmoves = nscramble), kwargs...)
	π(s) = output.value_function(s).maximizing_action
	(states, actions, rewards, sterm, steps) = runepisode(mdp; s0 = s0, π = π, kwargs...)
	initial_score = score_cube(s0.cube)
	(max_reward = 48 - initial_score, reward_sum = sum(rewards), steps = steps, rendered_states = [render_cube(c.cube) for c in [states[1], states[end], sterm]], states = states)
end
  ╠═╡ =#

# ╔═╡ d506b58b-883e-466f-b440-9b9aaa77d460
#=╠═╡
function show_rubiks_nstep_episode(nstep_result::NamedTuple; kwargs...)
	π(s) = nstep_result.output.value_function(s; nstep_result.output.form_kwargs()...).maximizing_action
	(states, actions, rewards, sterm, steps) = runepisode(nstep_result.mdp; π = π, kwargs...)
	initial_score = score_cube(first(states).cube)
	(max_reward = 48 - initial_score, reward_sum = sum(rewards), steps = steps, rendered_states = [render_cube(c.cube) for c in [states[1], states[end], sterm]], states = states)
end
  ╠═╡ =#

# ╔═╡ 2e9374ef-f5dc-4212-89c1-235fd9ab86ca
#=╠═╡
function render_episode_states(states::Vector{Vector{UInt8}})
	@htl("""
		 $([render_cube(s) for s in states])
		 """)
end
  ╠═╡ =#

# ╔═╡ 1cf590a2-63cd-4857-9a55-1d8e50bcd00f
#=╠═╡
@htl("""
	 <div style = "display: flex;">
	 <div>
	$(render_episode_states([Vector(rubiks_tabular_mdp.states[i]) for i in test_tabular_episode[1]]))
	 </div>
	<div>
	 $([rubiks_tabular_mdp.actions[i] for i in test_tabular_episode[2]])
	 </div>
	 </div>
	 """)
  ╠═╡ =#

# ╔═╡ 9910ebb3-c074-4f68-8381-264e255c97a4
#=╠═╡
function render_tdcube_episode(episode)
	states = [s.cube for s in episode[1]]
	sterm = episode[4].cube
	push!(states, sterm)
	render_episode_states(states)
end
  ╠═╡ =#

# ╔═╡ f930fe2f-b6a2-434b-a9a5-a2a01b4994c4
#=╠═╡
@htl("""
	 <div style = "display: flex; justify-content: space-between;">
	 <div>
	 Vanilla Policy
	 $(render_tdcube_episode(mcts_2step_episodes[1]))
	 </div>

	 <div>
	 MCTS Improved Policy
	 $(render_tdcube_episode(mcts_2step_episodes[2]))
	 </div>
	 </div>
	 """)
  ╠═╡ =#

# ╔═╡ 43658ec0-9735-4fde-87c7-4aa344033226
#=╠═╡
@htl("""
	 <div style = "display: flex; justify-content: space-between;">
	 <div>
	 Vanilla Policy
	 $(render_tdcube_episode(mcts_2step_episodes2[1]))
	 </div>

	 <div>
	 MCTS Improved Policy
	 $(render_tdcube_episode(mcts_2step_episodes2[2]))
	 </div>
	 </div>
	 """)
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

# ╔═╡ 1341aded-632b-4650-a4b6-de5c38fdda99
#=╠═╡
show_rubiks_episode(s -> test_dp_output.value_function((cube=s, scramble_moves = 30, move_count = 0)).maximizing_action; max_steps = 100, s0 = initialize_rubiks_cube(6))
  ╠═╡ =#

# ╔═╡ ee4ac3ec-d6c4-48d2-ab35-304b2ebb2367
#=╠═╡
show_rubiks_episode(s -> test_fcann_dp_output.value_function((cube=s, scramble_moves = 30, move_count = 0)).maximizing_action; max_steps = 100, s0 = initialize_rubiks_cube(15))
  ╠═╡ =#

# ╔═╡ af124561-acb5-4ed3-8fb4-62e01e9b66f8
#=╠═╡
show_rubiks_episode(s -> test_fcann_dp_output2.value_function((cube=s, scramble_moves = 30, move_count = 0)).maximizing_action; max_steps = 100, s0 = initialize_rubiks_cube(15))
  ╠═╡ =#

# ╔═╡ e9e35805-f669-42ee-b0dc-51ab8c800d2f
#=╠═╡
mcts_output = show_rubiks_episode(s -> rubiks_mcts_policy(s; nsims = 100, depth = 10, c = 0.5f0); max_steps = 10, s0 = mcts_s0)
  ╠═╡ =#

# ╔═╡ 11278249-5cf0-419e-b881-87f34b97d99b
#=╠═╡
function show_rubiks_reset_episode(π::Function; s0 = initialize_reset_cube(40), kwargs...)
	(states, actions, rewards, sterm, steps) = runepisode(rubiks_reset_mdp; s0 = s0, π = π, kwargs...)
	initial_score = score_cube(s0.cube)
	(max_reward = 48 - initial_score, reward_sum = sum(rewards), steps = steps, rendered_states = [render_cube(c.cube) for c in [states[1], states[end], sterm]], states = states)
end
  ╠═╡ =#

# ╔═╡ 9c94259e-a42a-44e1-90db-fafa4cc23b1c
#=╠═╡
show_rubiks_reset_episode(s -> test_sarsa_output.value_function(s).maximizing_action; max_steps = 100, s0 = initialize_reset_cube(5))
  ╠═╡ =#

# ╔═╡ 00e7e9bf-3c9c-47e6-bbd5-a63d471bf6a3
#=╠═╡
render_cube(solved_cube_values; square_pixels = 20)
  ╠═╡ =#

# ╔═╡ f5b2ea91-ef69-4779-9ae6-6acd00d888bc
#=╠═╡
function render_pocket_face(face::AbstractVector{I}, face_number::Integer; square_pixels = 20) where I <: Integer
	@htl("""
	<div style = "display: flex; flex-wrap: wrap; width: $(2*square_pixels)px;">
	$(mapreduce(add_elements, face) do v
	@htl("""
	<div style = "background-color: $(square_colors[v]); width: $(square_pixels)px; height: $(square_pixels)px; border: 1px solid black;"></div>
	""")
	end)
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ ade9f039-48ce-49ac-b2d5-10d8a13dc8fd
#=╠═╡
begin
	function render_pocket_cube(cube_values::Matrix{I}; square_pixels = 20, kwargs...) where I <: Integer
		@htl("""
		<div style = "display: flex;">
		$(mapreduce(add_elements, 1:6) do i
		@htl("""
		<div style = "margin-right: 5px; font-size: $(square_pixels*.8)px;">
		$(face_names[i])
		$(render_pocket_face(cube_values[:, i], i; square_pixels = square_pixels, kwargs...))
		</div>
		""")
		end)
		</div>
		""")
	end

	render_pocket_cube(cube_indices::Vector{UInt8}; kwargs...) = render_pocket_cube(reshape(solved_pocket_cube_values[cube_indices], 4, 6); kwargs...)

	render_pocket_cube(cube_indices; kwargs...) = render_pocket_cube(Vector(cube_indices); kwargs...)
end
  ╠═╡ =#

# ╔═╡ afda9df9-a96f-4a57-a6dd-e18c0c4a8b37
#=╠═╡
render_pocket_cube(solved_pocket_cube)
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
BenchmarkTools = "~1.6.3"
DataFrames = "~1.8.1"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
PlutoDevMacros = "~0.9.2"
PlutoHooks = "~0.0.5"
PlutoLinks = "~0.1.6"
PlutoPlotly = "~0.6.5"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.75"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.2"
manifest_format = "2.0"
project_hash = "70bfae4c7dac084a83563779e88c9f1e32e2c856"

[deps]
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
# ╠═ebf0eee8-b1d4-4f59-9727-518e6dd546bd
# ╠═a9748352-649d-45c4-b33d-a4c8f2b5b247
# ╠═b31fda46-27e1-4b10-9cc3-de26557521f5
# ╠═42e03441-cd7c-4f3c-a30d-9814b36428c7
# ╠═ebc5bf2b-abef-4b4a-9f12-b0884c089350
# ╠═46bbd599-ae25-485e-b064-af28efde316f
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
# ╠═fabd617b-cff1-4939-8e10-b2d79861fd47
# ╠═33d1609a-5c0d-4cb4-a7e1-1aca85da74e1
# ╠═ec496cfb-238d-48a5-a482-cfbe35cf2854
# ╠═28e5f0e7-d26b-49ff-bc1a-e7f177466053
# ╠═ef5827b1-11b7-414a-b1d2-4410684277fa
# ╠═05114a47-f015-410e-8416-34cc6b598f3b
# ╠═50e9caf2-83fd-47f4-8838-9f6fc82aa2cd
# ╠═9bfc604d-ca3e-4ccc-852b-63adfe67e8ef
# ╠═39a2ce90-9ba4-4079-9c7a-b91e2a1e57db
# ╠═85155ccc-ea07-4734-9e01-04fc723ce619
# ╠═d30cf757-2edd-4341-a3b2-b611c58ebe2b
# ╠═b62a3967-7642-43ed-9c0d-141f33c6de2b
# ╠═bec84bcd-9b14-4793-b4de-a1f5c5a71247
# ╠═24aa60e2-44fb-43ec-b0e3-624005493d58
# ╠═44e0a960-97a9-4d08-aa3f-9ceb212c7f1d
# ╠═abe87ea5-0d1f-4890-978c-948c99afe493
# ╟─bb3d3886-cbbc-4584-aed2-a53ed7f2d878
# ╟─78aaa31a-66ba-4325-915c-383e648fa801
# ╠═1157c771-1121-47da-8d4a-39c10a55df85
# ╠═361b9a94-ab5f-474d-8893-9987be4f0c5d
# ╠═44499455-ab75-4a65-9fec-1d544afb8a33
# ╟─053f6365-ad13-45f2-851c-ddcc16f98e4a
# ╠═8c396914-675b-404f-9f0c-90e880c55658
# ╟─cced6b32-f170-4470-8fbf-02bc2e8ad0a1
# ╟─5e621aeb-a106-4147-89c8-4b8635fc6342
# ╟─4442e1c0-1e16-4a26-8498-252053b7ee95
# ╠═1fbcf1f1-722a-4014-b8f6-11c2a5f12548
# ╠═b28d50ad-ca0a-4743-bace-be46c0821ac2
# ╠═fe20a8bd-c608-4ac7-9bb0-3251bcf7d85c
# ╠═c7cfee1d-4d1a-4e6e-ac89-a8fcfde55ecb
# ╟─8838f73a-c9b7-4b15-aa4d-b484452e71e1
# ╠═013fe195-8a8a-4717-a95b-16bb86e8de29
# ╠═3af88755-0e11-4a17-8867-b8687a484312
# ╠═fd053336-7b1f-4c79-afaa-f60b6fcc5938
# ╟─db9454b3-4ee0-4f16-9649-791a9e246939
# ╠═afad0f03-8b3d-4348-9570-9fe9489f0cb5
# ╠═6aec543d-eec2-47f3-9dbf-2631688e1112
# ╠═36bd6bf0-daaa-4780-a58d-e87fa96ae19a
# ╠═cd53cf28-6542-45aa-b9dc-38fa07f3e018
# ╠═5f6ed356-71c3-44a1-ae1f-ac8ed55a1c2b
# ╠═09538cca-7da3-4d48-9541-90f00acce794
# ╠═2040ffd4-f977-4c30-83a0-875b500099aa
# ╠═f1256bc8-a69e-4daf-a304-bd436ce8cfbe
# ╠═066ec351-b414-4232-a614-09b0430ede86
# ╠═e6cc1a1e-ab00-435f-a0ae-0afc09288023
# ╠═80069af6-31fd-4042-8690-a388c213fe28
# ╠═68150249-6db1-4dca-b998-9521a6d7d98d
# ╠═662729cc-54fa-449c-a90b-4c32196aef20
# ╟─a6492883-111f-4422-8e0a-13b8364b9e66
# ╠═5a325bc1-1a12-4534-91d1-74a0a20bf1e4
# ╠═e173c7b0-6360-4af7-8bc4-2377b30765c8
# ╠═e58a058d-57c8-4eb8-b5dc-0ffb2825a8ae
# ╠═f4318383-a402-4277-8160-2092f8ad46b3
# ╠═42022ea3-2a4e-4167-bfd1-9bdd2a441d62
# ╠═c04d8282-2799-4a79-8f33-7dcf4213ae4b
# ╟─e9d9cfd6-01b7-4d89-9870-57514eb03d22
# ╠═c6482bad-7912-427e-bb19-4245e1818f56
# ╠═67f2fe48-ef6e-4af2-9a4c-5af027cffabd
# ╠═5a6a4b92-7b02-441a-9f05-2f7924eec600
# ╠═b7586067-6d80-49c4-b8c4-877aa3e2ce3f
# ╠═d2f662c5-dc52-45c2-a335-5cc7ce2eb486
# ╠═d949035a-4258-46b2-b749-83397213c379
# ╠═92b1d07f-6538-44f4-8b94-9be1ee83728a
# ╟─343aa4fe-f38d-42b7-967b-c589be65077d
# ╟─7e57369f-14d5-4a5e-b83d-951228fac2eb
# ╠═128c8762-e4a7-46e1-a673-39d0ffbf2f72
# ╠═027d5ff6-b250-427f-9a00-ec9562236bc3
# ╠═35ae742a-db14-4868-bb2d-d59cd9300de4
# ╠═ac46c949-a0b5-42fc-bcf0-707faf7c0742
# ╠═9c94259e-a42a-44e1-90db-fafa4cc23b1c
# ╟─f78706e3-2b14-4bc8-9b96-2e4e98a2efa4
# ╠═6148ec6a-5d38-40d1-84ba-8b5253d5fdaa
# ╟─8d8c360f-66b7-4d50-bff8-803e77ef688e
# ╟─7cd30b68-1796-4dd0-bf13-dfb71610637a
# ╟─62650c81-17cd-4e2e-ab8d-a92c6b3eefb3
# ╠═97f64a35-b10d-4c3d-ac28-468110f9177f
# ╠═cbf95b8e-e6e1-4f6b-aec7-af96cf1bb752
# ╠═74d3cc6a-8486-4e01-8aee-a5e4c1b31a73
# ╠═d163d3cd-e8ca-44fb-81b2-66a2e1a92812
# ╠═99a7d920-173b-4b01-ab16-5248e6849f12
# ╠═9e25905f-452d-4ca5-ae9b-4b03284ad03d
# ╠═a538ead8-188b-4dd6-a105-a0c7a5d7d64f
# ╠═3479a6e6-91ca-4be6-9d0a-0d720b1f04dd
# ╠═1341aded-632b-4650-a4b6-de5c38fdda99
# ╟─7f12e90c-1dd5-49e4-91a1-786e3165c769
# ╠═6f551c31-3082-452a-ad9d-d77432c61af9
# ╠═03432a16-954e-4223-abdf-729aea9cbd26
# ╠═debc0525-2158-46ea-a7a7-cbf983fc40d6
# ╠═9b1e056c-1025-45a9-b508-f16691fb5696
# ╠═dd36dbbd-1807-42d2-ae8e-95ea9ad91069
# ╠═0ac973d8-bd67-499f-a05d-1341fa9ca52b
# ╠═699bf657-14cb-4575-925b-bedf97bc168c
# ╠═c07cc5dd-5ae5-4887-b7bd-c5a4c95ffa0b
# ╠═9e34bffc-e324-414d-a409-ef2cb13d365a
# ╠═6014ac2d-7d49-483c-92b7-7b1e24466b42
# ╠═a2ef7212-d2b6-40ab-8af8-6d38ab9f39f2
# ╠═98e24e54-e285-4c2f-984f-159e315fbdeb
# ╠═738f63c0-4101-48df-b73f-610bea1553af
# ╠═91e11772-3cfc-47db-a538-b439b669ccd4
# ╠═cd20e905-2355-47cd-82f9-aa576bf6a53f
# ╠═f0015a0e-b947-4a9b-b734-ddc01871c3c3
# ╠═ed067ef4-0b0c-4f2c-9372-899cfc6449c5
# ╠═33ffe90d-941f-4c84-93b5-628bd175140d
# ╠═2a0dc867-0b2b-4ad5-9bf9-ce8e1b0d545a
# ╟─5cafccd6-d525-44e8-a10c-3be0c9ff17ba
# ╠═4f7eded4-f49f-4db1-b57a-da7160409199
# ╠═1284f34a-da8b-473e-9e4a-5ef1bf8c1786
# ╠═4f244f15-8f07-403c-a2b4-7b2cb9dc7284
# ╠═6cab2983-d84f-49dd-b9ff-91933e45667c
# ╠═6d7d5e1d-6aa3-49f2-a85b-f34627868351
# ╠═f13eb4cd-7a45-41c6-b954-6e5f305c461b
# ╠═436bd68b-e3f0-4e22-8898-d1c586f7d69d
# ╠═4ddc8f58-fb12-47d7-946a-6601aab2b072
# ╠═42098e58-5246-4187-a0bd-089a00777793
# ╠═0d1beb1b-d5b0-4ddd-b9ab-71130e5b5f14
# ╠═df1dc37b-0c14-4a53-86e2-aab583e9c40a
# ╟─78fe374c-60f8-498c-87f8-7666e7e33412
# ╠═edaf3a59-808c-47dd-a331-d9bb5465d9ac
# ╠═c51f5247-732d-4096-9df9-4730cac95f5c
# ╠═3089e389-0687-4e35-afe9-72a20f5a597b
# ╟─fc70f91d-40ea-4de5-9deb-d5863aabb806
# ╟─613bd115-02f7-4b4b-b834-1dcad6016788
# ╠═990be70c-2b10-4766-8f05-d0b535dbde07
# ╠═29188eb0-6cad-48dd-8ea3-d1fa66b37d45
# ╠═2d377bf3-f16e-4b55-aea0-7df7478975f6
# ╠═b21966df-03b5-49d5-962b-878e5b0a1e2f
# ╟─1c83cd54-86c8-481d-b7a5-e5d4db60f27a
# ╠═6782e301-16ac-41e3-bd64-fe41dad938de
# ╠═bd50c529-c78f-48c4-8b30-bbbe4af87ca6
# ╠═2ee24bf5-f148-45ef-b2f4-ba8aa0bf96be
# ╟─c29c507f-0c07-424e-94db-796c36c09143
# ╟─8888a405-0200-49f2-8b02-ea61d7089629
# ╟─f4435085-6e27-4acc-bfa2-2fe16b232ced
# ╠═917e90dc-0cea-4119-8779-ed8b15b1a73b
# ╟─bec0a916-9689-482f-9008-15f035057838
# ╠═6aa509fa-a762-4571-8d88-360959cbb670
# ╠═ee4ac3ec-d6c4-48d2-ab35-304b2ebb2367
# ╠═3c81903a-fb36-48e7-9790-c547a297b092
# ╠═7b0b3725-09dd-4f1f-91e3-32d157241bda
# ╠═c0193517-5a19-4e09-9a9e-e8dac957ae74
# ╠═803bf4fa-de6c-421e-93a4-a001f480a036
# ╠═88a45407-9b12-412e-8c8d-4c9437d2d50d
# ╠═4c0482a1-0320-45d3-9246-8c38bd1f3a05
# ╠═5a0a75e8-6478-4c1e-ac7a-68d30481aa4a
# ╠═be9fce6d-2fcf-493f-9634-0da89160866c
# ╠═9a54a1ba-8e51-42c2-a44b-ab669f7533d4
# ╠═2d82c02a-583a-4509-b396-cb97d1ebc67c
# ╟─2d67019e-557e-4751-8c08-429be62b76d4
# ╟─c6e67132-a618-4371-bb74-00a9f34d16f0
# ╠═af124561-acb5-4ed3-8fb4-62e01e9b66f8
# ╠═19933037-d49e-44cb-af5e-80373c3e7b29
# ╠═73d36792-1b08-44e5-9fbd-1d9fc3d29127
# ╠═51dece7f-4656-4b58-a7be-3ba2617c82d2
# ╠═c313a8c7-8c49-4f54-bf25-3ec964ef1834
# ╠═02a0c6b5-73c1-43b8-b43d-ae073b74bd3f
# ╠═a9423fb3-288a-4b44-92a3-41a246c882df
# ╠═53b51ce8-4370-422e-b018-279a2da6185d
# ╠═a59a94ed-abd7-45df-9045-4502ad09064d
# ╠═d1d14f85-2ce1-4b3f-970b-5ef4ad38ee83
# ╠═3e3b7280-a3af-4a4c-9a98-21cbb22fef0a
# ╠═155ceb16-d02b-42ea-8d6e-15d3133e577e
# ╠═edc3c704-cc27-4089-bed5-44a93c454ce8
# ╠═f1bc4018-642a-46d7-b818-07eb3c26898c
# ╠═d4814bad-0fb7-4399-8059-ca31d8329db6
# ╠═a1b140e8-a379-4a36-a109-bbe126b4edbd
# ╠═699884e7-bc2b-44a1-8e04-dcd2001009c7
# ╠═5a1cd4ba-e5c3-40b1-a137-eef33b6bc721
# ╠═67bcffbb-7107-41ba-89bc-0fddedb6eb0c
# ╟─fd3f1da9-dbd8-4885-84bd-5f5fcb076a34
# ╠═72204679-8fdd-49a0-8fc2-24bf2f0e83ac
# ╟─fb106a66-d9ea-4195-8c0e-4246205a794e
# ╟─299b2603-bc86-4803-b136-865faf0571e1
# ╟─df8688bb-8cc7-4737-9ef4-a356c4d2de80
# ╟─76be0781-2406-4679-8b3e-701fdbf84944
# ╟─bfbd5284-34f7-4346-ac5c-857bc75eb035
# ╟─2514edf2-3960-4c1b-a47a-13400e5edad8
# ╟─c05959f2-36d9-4de6-8a15-e21cc3cc610d
# ╠═4f163fc4-beb6-43ff-a232-d3d8c19199c0
# ╠═37a6d563-f943-4bcc-8b0e-53af130a2bec
# ╟─d513c0b8-6699-4cdd-8481-7adc0735b669
# ╠═803182af-ac7b-4089-a4b0-b834856f725e
# ╟─cdaac960-ffcf-43fb-b0e6-1f5a9be03252
# ╠═f51df2ad-2c72-436d-81ca-a922dc41b1b8
# ╟─5f2bdb07-5a8e-4a62-a288-ba4b0d29aaac
# ╠═687c56a6-b87a-43ab-badd-d4cd7eddebcf
# ╟─f68e5661-ebcb-46b8-8fe2-f35779b8deeb
# ╠═04bababf-cebe-4d37-940f-f17ffc9c8f36
# ╟─18061035-b795-4df9-aa9d-a0a32b7c6598
# ╠═1cea3b3e-8bba-496d-aaf3-de19eadab8b4
# ╠═eb488bad-2830-406f-ad87-7cc98566e610
# ╠═b2e54677-184f-460f-bd90-07fa513a5d56
# ╟─c9bf5811-6a61-45f4-9070-6f2b7277611f
# ╠═8ca56795-bdc9-4740-8da7-9506d9063fe7
# ╠═b5feae0f-7196-445a-8a3f-b1a52a62b2de
# ╠═ae2427cb-be0a-4792-b0d9-5d14c17d9fb4
# ╠═ff5f38a5-f7f7-4417-87b6-f3d4b4591fad
# ╠═5e804f3b-a7a2-450a-8ff8-e56fffe662e1
# ╠═0c0115d2-ecd7-4947-9086-9f8759e568bb
# ╠═d2454f2a-c61d-4a76-8c6f-2234afefa0e4
# ╠═09c5e213-f175-4032-a5e5-dbd9f7b0d753
# ╠═831c8323-576e-4826-a868-514d21804986
# ╠═378a59e9-61d7-49e5-8d3f-0613c3632766
# ╠═61991d31-551a-42cc-a2ca-ce9a446f5952
# ╠═89de5822-ab28-4153-a244-798e6fcce1c5
# ╠═289bd333-b459-45e9-a427-863bb8fe709c
# ╠═95dec114-a1a4-4e51-9a9e-7a46d3f9928e
# ╠═68e6a2fd-bdaa-4973-89b3-d5eff791a39b
# ╠═02b73ed6-04e0-45a3-b071-d9ae4c347d78
# ╠═e8b2e0bb-2f96-4ffb-8be8-066090982ad2
# ╠═e8e05678-c0fa-46bf-bed5-96eb00dcf3f6
# ╠═54149d2e-06ee-4a5c-8c55-fcd0cd1442b5
# ╠═ab2affbb-fe34-49d8-b63d-fab610b62f14
# ╠═a815fae0-0e15-4bda-b1a9-34d773848c81
# ╠═dc8bfe16-e8e9-463d-8e2d-214e1b018d32
# ╠═989c1456-dd26-4497-b8d3-91a18c474370
# ╠═27c9d6ba-3f37-45a7-9039-d256cdda3958
# ╠═3e12ae05-271e-4f31-baf8-6e926460f12a
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
# ╠═7464b10d-056b-47cc-8db8-44ace4b84d11
# ╠═fa2286f8-7b20-4f33-8f5a-bccacbe758e9
# ╠═23e34ba4-006f-4100-a4a8-c14f6c33e887
# ╠═ef39bdd3-c085-441c-a8cd-6c1221ad4b21
# ╟─156aff30-918f-4f69-8830-815920a94d3c
# ╟─4017ed01-5cbb-47b6-bc29-d7a30b03a0c0
# ╠═a10c5406-4d4e-45f6-9fb7-b4b5702abf07
# ╠═725317cb-e761-49f3-9c03-a15bf9c34da9
# ╠═0d62e009-81be-43d3-add5-ce4ac591dbc7
# ╠═dee2e1c5-8423-47b3-868d-483b871da731
# ╟─50f873e8-f59d-40f3-8adf-600ddf4b6e1a
# ╠═92c365b7-924d-4ec6-978d-e743d0237cfc
# ╠═c079e451-0487-4e02-8f0e-b61e8963eeb2
# ╟─6e9091e1-afaa-46cf-8e21-6b38526640cb
# ╠═1938becf-e2cc-4c7b-b6ec-d1c3ff107ed9
# ╠═d7c391b1-0d13-409b-a0ac-c02fb766b839
# ╟─7f45ed2c-996b-40dc-b2cd-4f6cc7bddc4c
# ╠═52eb3090-66a0-41aa-90ed-f658847f9601
# ╟─bcb9aec3-2d1b-4361-8fc3-6a2cfa20ddb3
# ╟─1cf590a2-63cd-4857-9a55-1d8e50bcd00f
# ╟─afe35fcb-44f3-4deb-b1f6-8820b74679c1
# ╠═a487ccf0-c293-4093-b936-f92094e86fa7
# ╠═f26935db-a80a-45d7-b2cb-23febbfc3d15
# ╠═84a08fd1-f8b5-4ca8-b0c0-2c306044d694
# ╟─9e1d861d-1f5a-40c8-ac24-13c2b35cd90c
# ╠═4409a59f-e138-4c5b-910c-828bf8bd4497
# ╠═47ebe08c-5234-42bf-9750-1fca5bb63e99
# ╠═afda9df9-a96f-4a57-a6dd-e18c0c4a8b37
# ╠═ccd01889-2619-4518-b7e5-aa2768cde226
# ╠═62dd2f4f-4ecb-4152-bb9e-7a6fb75222d9
# ╠═dda10443-ec98-45a7-a935-86e65d608dd7
# ╠═9b8f7aeb-d009-444a-9fea-b54677c30197
# ╠═1a0d2678-c228-40a3-a376-01df24292556
# ╠═8e44a35b-ed6c-4a9e-9f68-9fe28aa5af22
# ╠═a7ed00c6-6ada-41ee-9cbc-770c3eed776c
# ╠═c5c1942a-86ff-4494-9e38-3ad0bc6c0517
# ╠═bd55c435-a481-47cb-a032-f8ff9432f630
# ╠═ed426ccc-2b42-4610-9eeb-f342907861d3
# ╠═ce6556ef-a69d-48a4-8eab-a2146710c9ca
# ╠═6e20e3ee-6de3-46ce-bad6-ddad9694aeb4
# ╠═861a3436-6d46-425a-afa4-1f2be8994221
# ╠═e4d905b5-aefc-4a2f-8c41-e48c297b8663
# ╠═b384cf65-0865-4b54-81af-0ad9fbd3db97
# ╠═8b8bcacd-0d00-4c72-86b4-e3bdf43d653f
# ╟─14f757fb-5f14-4f21-af1f-84be6249bdf5
# ╠═f8f98ed9-f936-41e6-b8de-02a71c0a06c7
# ╠═0d41e787-eb70-4e47-9ecd-223caec70dd7
# ╠═b1d4a9f9-ebea-46ac-8c38-0ab3dfd2e905
# ╠═76e07bdb-82c7-4dc8-893d-e7ca3d0d318a
# ╠═b5f46f2f-2ea8-40ee-a28b-bd8c90e52149
# ╠═2968db78-f04a-47ea-b5a7-8dd383406c7e
# ╠═5b910da9-52ad-4e3e-a264-2fa6c6df3cf3
# ╠═1e2f7cb7-aabc-4ea1-a192-334d0788f53e
# ╠═200ecf10-baf2-4f7e-a8ba-11a0dc3cab92
# ╠═2af81c2a-5fa3-421d-88a0-a9865400cf47
# ╠═5656dbff-1763-4464-bf7a-438983907e8e
# ╠═71bd3296-99ba-4706-9c34-c8543f9fb020
# ╠═b9b7030d-3035-4401-bd5e-006bd2fd583f
# ╠═73d40eb7-9213-4318-9bea-1a20edff2fbb
# ╠═1dee4e15-b9f6-47d4-bf7e-230946a6054d
# ╠═b57a54e8-9876-4746-a1a3-f74812823a75
# ╠═5e6aca94-4efa-444a-82fc-cea805c20815
# ╠═6aade30b-6784-495a-b559-a3483cb7dc49
# ╠═2e4fa908-f1c4-4594-a0cf-782cf6513208
# ╠═435df0bb-ac20-4b6a-9575-03cd249b604c
# ╠═acc8f2ae-0c21-404b-8cd1-34d498f7b4de
# ╠═de5ea355-4ddc-46a1-8a55-a2f42a00d3a3
# ╠═71dae97e-6df2-472a-81ef-663017d7b23e
# ╠═e0d2df20-2781-4589-9102-7189499536e7
# ╠═4cbc56b2-85ac-47bb-80bf-963154cf3f48
# ╠═700444c9-8ae0-4f6f-bfbe-7c4e2a014773
# ╠═a58f1013-c5de-48d1-a6bc-a6c00adc9d1e
# ╠═60a41dea-32c0-43a6-b203-1eb07dbd7261
# ╠═318a221d-3a86-4419-9d87-e72eca3dd9c0
# ╠═f156cea6-8d3e-4c08-aca0-ecd93ad961d1
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
# ╠═7bb3fe7c-a8a5-41b0-b35c-2ca92a4d2eb0
# ╠═a06c96f9-57c1-48e4-a725-08980892502e
# ╟─16d6e982-c564-4d77-93b2-c4180eabbd2d
# ╠═37f28ba0-86b4-4c5d-96a4-e8adcec7c618
# ╠═3ac7bdb3-b4a8-4a64-82ea-dcd2dce14b6d
# ╠═30d9c367-01e6-4ee3-955a-7c1ad81cbbd5
# ╠═04bb99fd-2e99-4af7-8582-5cbc46fad29e
# ╟─206a1146-4a01-412f-a17e-3d58cac83453
# ╠═9084cb79-a89f-432d-a4f9-4b1f9ee65c4d
# ╠═d9460da6-d51f-458b-bd90-888961ffe3a0
# ╠═db57648d-ea66-4ece-85af-0df97a725ae5
# ╟─02523a61-5b7f-410c-ba3b-3564fcd8a35b
# ╠═837479ed-a3c7-46dc-8fa0-31c98bb3ef6e
# ╠═fb49c6c1-1a4c-4d57-92e8-b4abda1d6533
# ╠═b2324cff-2b72-4ec8-a51e-5123a2cc8ebd
# ╠═91a90c89-a871-4776-a7d2-893fe8d7df63
# ╠═949b2c5b-9657-4d8c-99e0-eb983c582ed1
# ╠═4453abd2-72c2-4a0b-8044-6345e0f87009
# ╠═4210dabc-1ab4-46d8-950b-612201cdd0be
# ╠═01f86faa-224d-4509-acbf-8ee0d9b5881b
# ╠═d84fc0d0-38bd-445d-8f03-4fe7f0db1d2a
# ╠═6e8117cb-94ea-4216-8191-f250e03173e0
# ╠═42b1cb51-1fbf-45b3-8c07-19484e9c9bbb
# ╠═eea81ca5-152b-45d6-94df-f05565376d2f
# ╠═91e6576c-0142-409c-8749-194c66ba3c9d
# ╠═3f459caa-c83d-4cdc-8b6e-78459b00d312
# ╠═abfc1d27-96a6-48dc-93d1-b2f5db4455ef
# ╠═1e6c3190-6b5f-4ada-a24f-98483658aa17
# ╠═2453b950-d1ff-45ec-8423-8d83d43afe8c
# ╠═4f9ae2a3-cc42-42cf-8fc9-c80d4588f675
# ╟─ae4ddf00-85b2-4794-b92c-7ef76b8a1738
# ╠═116c751a-76e6-4139-9926-0e8f58d30fe0
# ╠═075be7b6-e658-4818-90f6-6674c00c678a
# ╠═88087bbf-3bc3-4331-bf7a-44f4a023f54d
# ╠═e8b7cf98-f5ed-4fb0-9d7a-4eb7fcd6bea4
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
# ╟─92f16c16-5384-47cd-a8f6-31694b503ec8
# ╟─bfefc8c5-8620-46c0-b6ef-5a8dd4106889
# ╠═1735c965-8aa0-4c74-bb41-8800d145e09d
# ╠═84d639ef-08d8-45c5-8df6-39e1dcb31abe
# ╠═44f7e193-61c2-4591-abe7-1800328abb18
# ╠═546635bc-83d8-4b07-8b0e-df6d8a29d45f
# ╠═1e8f8041-0cf2-47f7-b24f-d4b81ef36360
# ╠═b609b7fc-3994-4488-beed-a151f1984d46
# ╠═bd6ef1e0-0e75-4186-b597-01860adb10b4
# ╠═7b61d79d-e304-488d-8cc8-ed70a0e7d052
# ╠═1222ffc4-82a4-4397-89f0-e0b5ce568fed
# ╠═a67c9467-fe95-424d-92bf-1dd6eecfbd88
# ╠═ddcbd552-d3a2-46ed-9a5d-d705730501c7
# ╟─1f3cc59d-81c4-4c2f-a6d2-702d803b1f1d
# ╠═a38e34b1-57ac-496e-be00-73501b865b6f
# ╠═ea436bfd-97df-4859-9bc4-14ee5f67ecaf
# ╠═194595ee-5b5d-4b97-a48a-bba10a491fdd
# ╠═c676f6ab-07cd-4e55-b853-ff7a13c03f80
# ╠═da55b6bc-ba84-4800-bab4-c5bbfae73b99
# ╟─1060a7ff-4ed3-49e7-a4e4-12ec896e924d
# ╠═6f2724d4-dcf7-417b-966d-1ff8deeb2a65
# ╠═96223544-4478-41ab-aba5-fcde3cac0768
# ╠═440470e3-1f38-4041-b554-13a155f19cdf
# ╠═d952334c-1953-49f5-8871-d96bfbfa64e3
# ╠═8b26a4ce-d853-45a6-b83d-9d07f30b6621
# ╠═d506b58b-883e-466f-b440-9b9aaa77d460
# ╠═174669a8-218c-49e8-a4d9-e4482d3850ae
# ╠═1317a53a-c817-4950-aacb-528617ccb6e4
# ╠═c3ece8b5-ddaa-46a1-bd44-1b7d99cb6fac
# ╠═07377754-fa04-4e5a-b15d-f6cfd9d52dab
# ╠═bb6fea94-f0cc-46c9-90f3-4bed62b520c3
# ╠═f8180571-8869-4904-929b-dc89a0a612c6
# ╠═8c76e9af-b50c-4d3c-b5f6-48c189a44c8b
# ╠═fb4a36de-15af-41eb-8678-6418d55c5f65
# ╠═3afc9543-8172-42e1-bb28-717dfb59e7ee
# ╠═74c9b56a-d890-4dca-a26e-a6d41f105cde
# ╠═ef8c2a99-d462-4cb1-b433-d669cda52d31
# ╠═fe719761-de47-4240-a50d-a1dccdf2d1e1
# ╠═7f88d0d3-2069-4c97-a3a0-20196a7151a7
# ╠═273923d3-ab5e-4899-80bd-f257fa5c1d3c
# ╠═dde9d10b-e259-4c86-98fd-d6459f36699b
# ╠═b117af73-16a0-44cb-abfb-fd9df962bb97
# ╠═56cf2e20-41ee-46fb-a5d4-0b79642ac3b9
# ╠═1f0429c4-e6c7-4b09-8896-62fdf5569148
# ╠═affd9578-11a2-4150-862a-aad1f7cfa565
# ╟─8e317749-78a6-4835-8a2d-e4f4690c7c68
# ╠═990e31e1-a1ef-4ecc-8279-0b88824a032e
# ╠═2308e4aa-7bcc-4c03-a496-fbe7f5d76bb1
# ╠═9d70d00d-39ba-430b-8a78-17c6dc4b6cb2
# ╠═45edbe8f-2c82-4ccd-9151-9e7551b4b62f
# ╠═c2dfc40a-9934-426f-8e46-48b142b2601b
# ╠═b67b017b-afb3-4caf-a3d5-7b3d3c530c25
# ╠═26a14497-12d2-4aec-b17c-72544ab23709
# ╠═e71ee180-2fdf-46b9-98a5-972a0a993cfd
# ╠═da030300-e2f4-4c18-a11b-29f8c7bad572
# ╠═f0316e07-2873-4538-b7a5-01256b089561
# ╠═643a6ca5-d247-42b0-95b5-283cfa1e965f
# ╠═d6a20337-857b-4775-8167-dcb1a2ec0ecf
# ╠═6095a2c1-b33d-4f39-965d-c61e042bebe9
# ╠═b760fbbf-5e07-446f-9b37-6e184868dfdd
# ╠═568815ac-9c48-4ccf-b22c-e9ed38831cad
# ╠═68a4f3ac-16d2-4b18-bdca-74718172ede5
# ╠═b9f439b0-12cd-4c78-ac83-549451efc90a
# ╠═1478b8d9-a051-4f8d-b221-1468231b3ece
# ╠═b54f218a-46f6-407c-bf7a-e41a87f351d6
# ╠═341ec41b-bc42-4548-b11a-f9b326179422
# ╠═b903b3af-4c89-4054-88cd-4dfcc3a7d6f0
# ╠═58f6a2f1-c6eb-4031-9671-22c0d6416862
# ╠═cde3b13f-86e0-46ad-8a74-84684cc32811
# ╠═b9586038-58ca-467b-8cd5-f5a16663682e
# ╠═8bc7748a-babe-46a9-9277-fc69d11220f9
# ╠═bd930459-d714-4cdd-a274-bc6ae44efb26
# ╠═fc718149-0104-4893-8ace-3ffe1260ccdf
# ╠═bca04135-4e8f-4d0d-9bd5-5af78a76455e
# ╠═c873a5cd-2b65-4c80-80ee-9cdd57172b7f
# ╠═de692498-e6e0-4d68-8975-1ff80abe0651
# ╠═c74f80a7-71f6-49eb-9b9a-0c5c78cd7ec3
# ╟─d2b7fac5-fcaa-4d0e-a611-87c50f6d4d84
# ╠═8918bd6e-4cc7-4f04-9f45-4ab9b2edb259
# ╠═fbee7bdb-e826-41bc-bca7-e900cc7c4dca
# ╠═afd359a6-8f15-4d79-9267-07a2683f7635
# ╠═eff2c88c-a308-42d7-b3f2-98da787c930c
# ╠═f453625b-ac72-46a5-a2a5-e21492dcdd86
# ╠═1f4ca1e0-d4c1-4ccf-aad4-3cba3ba822c2
# ╠═0648f541-aee5-451f-89a2-f8c55fb0105b
# ╟─9864c34a-405b-466e-ae1b-e0844c2d4019
# ╠═ca57f9f0-4d9b-41d6-b7ec-931415b39641
# ╠═d86a6362-bd09-4eac-a0a3-2149c566d693
# ╠═5120365b-1317-4a2f-803c-bbac28046923
# ╠═ee8842f2-86af-495b-9b37-41aa59dd4cc1
# ╟─7fe349ed-36fb-473e-80cc-6e7585610bf9
# ╠═0e45945d-96f6-4eff-8276-162a72f8730d
# ╟─92f40a51-695c-459e-9861-3d3d59d55546
# ╟─d2ab9e40-1d90-40dc-b36b-6e778b821ac1
# ╟─a713e511-7c05-4cb4-9fee-2043fb0d4242
# ╟─4e0ede0b-bcde-42e6-baa6-0b0dea95a83c
# ╠═28a886a2-6fc0-4c16-9426-882f1745b67c
# ╠═e2d1180d-e34e-45ad-8593-d6c1dd34cad2
# ╠═68cce5b5-a52e-47e3-8670-911dbbdb7509
# ╠═66f35e7b-c9d5-4a1f-8f08-fc83e419539c
# ╠═2935371e-fca5-47ef-89b9-781a1ff7b8b0
# ╠═125cc2d6-8712-4b41-96bc-243120eb4ff4
# ╠═897476bc-bfc7-4e10-b965-0175902d9407
# ╠═a7a35c3d-b82d-48b0-b42a-5551ff38bc32
# ╟─9d4fd48b-f0ca-4dd4-8648-e7a8827740b8
# ╟─8fa47fbc-d097-4b7e-8ac7-8bcefb492ee3
# ╠═a76fa602-642d-46d2-9540-4fa94b38e7cb
# ╠═f04f6ab1-6de1-4b38-8973-40064f496514
# ╠═c6fbbf97-1a25-4b80-bd5c-89034efa3f07
# ╠═cc689336-57ac-4a0d-9a3e-81a17b2e0b39
# ╠═421add0b-6b3b-49fd-8f71-1898decd24af
# ╠═ded411aa-2927-4435-9e3e-3f01ae35b78c
# ╠═0d8fcded-38b2-40fc-a4d5-0da61767cc2d
# ╠═9339a39a-bc74-4408-ac44-557366426840
# ╠═9e25ea7c-c3ff-4dae-873b-3decf08b4625
# ╠═7251946d-0899-465f-ba00-aa1c623b74da
# ╠═598a04c6-b290-4758-9480-e1704b9326a4
# ╠═53ca25d4-f0fd-471f-b509-519dc6cb5ecd
# ╠═d70eaa9f-ac2c-4303-89e4-a46443e65fe2
# ╟─04a77130-179b-48bf-8393-5c8d11c4d164
# ╠═0bf5d23c-d0a8-4660-900c-43573c58faa1
# ╠═8fb578c8-51d9-46ad-88ec-cd0cf0e73cf3
# ╠═004c8296-38db-4999-a134-3f009eba8034
# ╠═ca5aad17-43a7-40af-abb8-5bc430510fce
# ╠═f181e0a7-0e3f-43e2-af6e-70eb2ca98854
# ╟─8d40eb5b-ed23-4c13-abd9-6af0e63cb2b3
# ╠═c2ade361-6c96-431f-9237-98accc2de8ee
# ╠═432f8e64-c124-491d-9633-9d77bb1a9ff4
# ╟─f930fe2f-b6a2-434b-a9a5-a2a01b4994c4
# ╠═b37cf65b-eb9a-4a78-be83-e4d4f54355d5
# ╠═ae5fd508-1c9c-418c-a862-263bcc76630e
# ╟─43658ec0-9735-4fde-87c7-4aa344033226
# ╠═2e9374ef-f5dc-4212-89c1-235fd9ab86ca
# ╠═9910ebb3-c074-4f68-8381-264e255c97a4
# ╟─1377a706-5a89-495e-98cf-d1b4ac1511f8
# ╠═b8a2150f-9be4-4c12-b645-37c9b91c7728
# ╠═a56e5eb6-7ec3-45ce-b93d-2230691652b2
# ╟─4f79565b-860f-4862-92b0-fdc1b00f0c1d
# ╠═d41c9ab0-0712-4960-a2dc-aaf3661e0866
# ╟─3bcc2eb8-5c4d-46ad-a1f4-f380adf66812
# ╠═00609e53-d6bd-4630-9a28-8e69acc10901
# ╟─2026135e-5d97-4755-8de7-cd3175e37298
# ╠═1d332f0c-bb86-49f8-9f61-2ba03354b3dd
# ╟─a35f5d8b-a9ea-4d9c-84c2-44ceea2fbe56
# ╠═bfbc45f1-900e-4596-965a-0dfe6e9bdc9c
# ╠═75e1b19f-299b-4f05-a98a-87cc4ddb14d0
# ╠═4aed6452-26e6-4a48-9a3a-b1be7d961157
# ╠═fc8a9f05-2b66-4ef2-8e05-538f66b4b840
# ╟─fa4fa0fe-113b-4537-a498-a1be486a9b8e
# ╠═cb2dd4f3-b0ac-4148-877a-408831fea5ef
# ╟─267c92f0-2b55-421a-8210-adad5e6ad811
# ╟─c71826b6-e295-4c8f-8d73-52dbde7a39e7
# ╠═4828ae7b-2f70-4b17-94f0-06505fac7887
# ╠═86ef6b80-f902-48da-9032-fca221c3ed6f
# ╠═fe415c36-f2b2-4b25-9ab2-33011fdc9f28
# ╠═a303ee2c-ac5c-4728-b191-ea5570538a4d
# ╠═af0c2dba-c7ee-4f75-a31d-1bc1dffd3e84
# ╠═99a280b6-a371-43ac-9a27-319501a47d96
# ╠═dd14d0ff-5163-43da-b1be-3fc420682fe3
# ╠═cb8d0b4a-39de-4a18-8048-7a525a6bfca8
# ╠═46749d40-8c07-43dc-bbff-013797fd56cd
# ╠═139cc887-f3ee-4810-aa97-2aec301c5238
# ╠═717209cb-24de-4fae-b434-bf8258f39fc6
# ╠═242a57ef-7e7e-4439-95cb-9f747b23d6b3
# ╠═a19597b5-fa79-4808-8f60-9bafb1a0dbb4
# ╠═36f51608-7d65-4f21-ad71-98707f94b12a
# ╟─ce885abc-23ab-47d1-87d8-bdf5bd512212
# ╟─cab6e8c8-5276-489d-8dc2-f3adcfda8ebc
# ╠═9d87dbb3-1a3c-47d3-a090-4eb76a5c6c2d
# ╠═a3555ab8-63b0-451f-a91d-995d7d8632bf
# ╟─1e2456d3-8d14-4568-88df-5f5757ce928e
# ╠═cbf5ed88-ef38-40f9-ad2c-649505f55997
# ╠═94a15d87-790e-4917-ba9f-446f699cfe92
# ╠═9ffbb5d0-defa-4734-883e-7ff1f86e326b
# ╠═af9571f9-9d82-4563-83b1-d3dc23668b6c
# ╟─621f415b-ef9b-4ad5-bcc0-dcf51af1d1f9
# ╠═00c67f96-456c-48d1-9d1b-0fcd8ddac671
# ╠═51105316-c191-44a5-b4f3-703f30efebad
# ╠═d28b936a-5ca6-494c-a6fd-c96c60d24add
# ╠═55f31d00-a946-46e2-b021-855120b73e77
# ╠═9fffc847-87c2-41d6-9f87-d10d645708da
# ╠═42106894-5ae2-4a1e-a26e-0e8e7085d035
# ╠═4e9029e2-1c7c-4fc6-901d-c10a617c3cc0
# ╠═7fdfe56e-681a-4708-ac7e-5cbc7957adc1
# ╠═54f505d3-2a6b-4ce7-a0a4-96b62159dd4c
# ╠═f90125d1-04d9-4c0c-a0b3-cf2c98b933d8
# ╠═c2a1a4a9-c652-400f-ad3a-5a05d4e10073
# ╠═790aea25-52d1-41c0-8b80-5b987c294c54
# ╠═0d80de04-ac65-4cce-9db2-5f3053079a1b
# ╠═17006174-caad-4043-b8d5-883aa0e10c80
# ╠═8218ea8b-3ba6-45fb-ac36-89ba6cccf112
# ╠═93eefafa-60da-47b0-9ea5-6e9085e1c231
# ╠═15326e1a-96b7-414e-9967-46e222598bc6
# ╠═27b38778-17fe-4aca-8cec-341e2333e536
# ╠═639b2d17-59c9-4605-a394-8fce6dc5449b
# ╠═1af6a2f7-ea00-4d96-8045-2ce280b5a833
# ╠═f8d68c24-71f0-4850-804d-67e5634ed60d
# ╠═eb5d2eac-5280-4269-8b2d-3c31226b4b2f
# ╠═6ec48894-7cc4-46ea-90b5-5f0e8e566b5a
# ╠═a2464600-d228-4f8c-96c0-6b8fa8c87afe
# ╠═11278249-5cf0-419e-b881-87f34b97d99b
# ╠═00e7e9bf-3c9c-47e6-bbd5-a63d471bf6a3
# ╠═1737d09d-046f-49d4-9287-e67bfee13468
# ╠═083caabf-4122-49fa-b397-7746b77ca12b
# ╠═f5b2ea91-ef69-4779-9ae6-6acd00d888bc
# ╠═ade9f039-48ce-49ac-b2d5-10d8a13dc8fd
# ╟─6b410101-edb1-4814-8aa6-344d0c969e01
# ╠═c5256762-ca4a-4c81-805b-4f865efc2091
# ╠═624eef76-16a7-4556-a466-14341346f7a5
# ╠═c5c0f635-171d-4904-9675-d1b0a01f6d7a
# ╠═e956ffaa-01c3-4e44-8c9f-2298347fea03
# ╠═dcf12932-7854-481c-8bfa-f6a4b2956518
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
