### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 0574291d-263a-4836-8cb9-78ad7de3f095
begin
	using Statistics, Random, StatsBase, DataStructures, StaticArrays, Transducers, Serialization, PlutoHooks, Base.Threads, LinearAlgebra, SparseArrays, HypertextLiteral
end

# ╔═╡ cbcc1cd8-7319-4076-84cf-f7ae4d0b5794
@skip_as_script begin
	using PlutoUI, PlutoPlotly, PlutoProfile
	TableOfContents()
end

# ╔═╡ b6144c34-9f2b-4dc4-81cb-20e3a4cef298
@skip_as_script md"""
# Tabular Solution Methods for Markov Decision Processes

Code implementing the concepts as well as examples executing that code is interspersed throughout the document.  Any section containing code and examples will be italicized to distinguish it from other notes.
"""

# ╔═╡ 5340f896-674d-4675-b53a-8e22b536a269
@skip_as_script md"""
## Markov Decision Process Definitions
"""

# ╔═╡ 6a3e83b0-b4b4-4f4b-bd72-eb97df199465
@skip_as_script md"""
### Agent and Environment

We seek to find the optimum *behavior* for an *agent* interacting with an *environment*.  To properly define an *environment* we must first define the *state space* $s \in \mathcal{S}$ and the *action space* $a \in \mathcal{A}$.  An agent is something which can, at discrete time steps $t$, take actions in the environment.  Once an action has been taken, the *environement* will produce a *step transition* consisting of a numerical *reward* as well as an updated state.

An *environment* is defined by a *probability transition function*

$\begin{flalign}
p(s^\prime, r \vert s, a) &\doteq \Pr \{ S_{t+1} = s^\prime, R_{t+1} = r \mid S_t = s, A_t = a \}
\end{flalign}$

which specifies the probability of every *step transition* given a state-action pair.  By interacting with an environment repeatedly, an agent will produce a *trajectory* which consists of a sequence of as follows:

$S_0,A_0,R_1,S_1,A_1,R_2,S_2,A_2,R_3, \dots$

This sequence can continue indefinitely in the case of *continuing tasks* or terminate at some special state $S_T$.  If an environment has such a *terminal state* it is charaterized by the following property: $p(s^\prime, r \vert S_T, a) = \cases{1; \; r = 0, s^\prime = S_T \\ 0; \; \text{else}} \:\: \forall a$  

in other words, the only possible transitions from the terminal state remain there with 0 reward.

For some environments, only one transition state $s^\prime$ can be reached from any state-action pair $s, a$.  These environments are called *deterministic* (the reward may or may not follow some distribution of values).  All other environments are *stochastic*.
"""

# ╔═╡ da6ab60e-1677-41dc-82a1-bbc0c9234e25
@skip_as_script md"""
Often times, we visualize these *trajectories* with diagrams where open circles represent states, closed circles represent actions, and squares represent terminal states if they exist.  Even if an environment is *stochastic* a trajectory will have a single path as shown below.  For a *deterministic* environment, this path will represent the only possible trajectory given those actions.
"""

# ╔═╡ 9836edb5-5d95-4091-af9a-849b6d077cbf
@skip_as_script begin
	@htl("""
<div style="display: flex; flex-direction: row; align-items: flex-start; justify-content: center; background-color: rgb(100, 100, 100)">
	
	<div class="backup">
		<div>Example Trajectory</div>
		<div class="circlestate"></div>
		<div class="arrow"></div>
		<div class="circleaction"></div>
		<div class="arrow"></div>
		<div class="circlestate"></div>
		<div class="arrow"></div>
		<div class="circleaction"></div>
		<div style = "color: black; font-size: 30px;">&#8942;</div>
		<div class="circleaction"></div>
		<div class="arrow"></div>
		<div class="term"></div>
	</div>
	<div>
		<div class="q_backup"></div>
	</div>
</div>

<style>

	.backup {
		margin: 5px;
	}
	.backup, .backup * {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		color: black;
	}
	.circlestate, .circleaction {
		margin: 0;
	}
	.circlestate::before {
		content: '';
		display: inline-block;
		border: 1px solid black;
		border-radius: 50%;
		height: 20px;
		width: 20px;
		background-color: white;
	}
	.circleaction::before {
		content: '';
		display: inline-block;
		border: 1px solid black;
		border-radius: 50%;
		height: 10px;
		width: 10px;
		background-color: black;
	}
	.arrow {
		display: flex;
		justify-content: center;
		align-items: center;
	}
	.arrow::before {
		content: '';
		display: inline-block;
		width: 2px;
		height: 30px;
		background-color: black;
		margin-bottom: 0px;
	}
	.arrow::after {
		content: '';
		display: inline-block;
		width: 4px;
		height: 4px;
		border-bottom: 3px solid black;
		border-right: 3px solid black;
		transform: translateY(-5px) rotate(45deg);
		position: relative;
	}
	.term::before {
		content: '';
		display: inline-block;
		width: 20px;
		height: 20px;
		border: 2px solid black;
		background-color: rgb(50, 50, 50);
	}
</style>
""")
end

# ╔═╡ 4835bed5-a02a-49e9-8a01-63885109339c
@skip_as_script md"""
### *MDP Types and Definitions*

If we know in advance the entire probability transition function, then we can define an environment using those probabilities.  Below are datatypes and functions that implement such an environment.  Note that to implement such an environment, a complete list of all the states and actions must be known ahead of time.
"""

# ╔═╡ 872b6292-8318-4161-915c-c3d3b9ef1236
"""
    makelookup(v::AbstractVector)

Creates a lookup dictionary from elements to their indices.

# Arguments
- `v::AbstractVector`: A vector whose elements will be used to create the lookup dictionary.

# Returns
- `Dict{T, Int64}`: A dictionary mapping each element of the vector `v` to its index.

# Description
This function generates a dictionary where each key is an element from the vector `v` and the corresponding value is the index of that element in the vector. This is useful for quickly finding the index of an element in the vector.

# Examples
```julia
julia> v = ["a", "b", "c"]
julia> lookup = makelookup(v)
Dict{String, Int64} with 3 entries:
  "a" => 1
  "b" => 2
  "c" => 3
```
"""
makelookup(v::AbstractVector) = Dict(x => i for (i, x) in enumerate(v))

# ╔═╡ 43c6bb95-81a1-4988-878c-df376e3f7caa
begin
	"""
	    AbstractMDP{T<:Real, S, A}
	
	An abstract type representing a Markov Decision Process (MDP).
	
	# Description
	`AbstractMDP` is parameterized by three types:
	- `T<:Real`: The numerical type for the rewards.
	- `S`: The type representing the states.
	- `A`: The type representing the actions.
	
	This type serves as a base type for MDPs where the probability transition function might not be fully specified.  Also it may not be possible to enumerate all of the states.
	
	# Notes
	When the full probability transition function is known, this is represented by a subtype called `AbstractCompleteMDP`, which also specifies the numerical type `T` for the reward, the states `S`, and the actions `A`.
	"""
	abstract type AbstractMDP{T<:Real, S, A} end

	"""
	    AbstractTabularMDP{T<:Real, S, A} <: AbstractMDP{T, S, A}
	
	An abstract type representing a tabular Markov Decision Process (MDP).  While the probability transition function might not be fully specified, there is a complete list of states and actions.
	
	# Description
	`AbstractTabularMDP` is a subtype of `AbstractMDP`, parameterized by three types:
	- `T<:Real`: The numerical type for the rewards.
	- `S`: The type representing the states.
	- `A`: The type representing the actions.
	
	This type specifically represents MDPs where the state and action spaces are finite and discrete, and the transition dynamics and rewards are stored in tabular form.
	
	# Notes
	When the transition dynamics and rewards are fully specified and stored in a tabular form, this type can be used to define tabular MDPs.
	
	See also
	---------
	- `AbstractMDP`: The base type for Markov Decision Processes.
	"""
	abstract type AbstractTabularMDP{T<:Real, S, A} <: AbstractMDP{T, S, A} end

	"""
	    AbstractCompleteMDP{T<:Real, S, A} <: AbstractTabularMDP{T, S, A}
	
	An abstract type representing a complete Markov Decision Process (MDP) with fully specified transition dynamics.
	
	# Description
	`AbstractCompleteMDP` is a subtype of `AbstractTabularMDP`, which is in turn a subtype of `AbstractMDP`. It is parameterized by three types:
	- `T<:Real`: The numerical type for the rewards.
	- `S`: The type representing the states.
	- `A`: The type representing the actions.
	
	This type specifically represents MDPs where the state and action spaces are finite and discrete, and the transition dynamics and rewards are fully specified and stored in tabular form.
	
	# Notes
	`AbstractCompleteMDP` indicates that the probability transition function is known for all state-action pairs, and rewards are fully specified. This allows for a complete representation of the MDP.
	
	See also
	---------
	- `AbstractMDP`: The base type for Markov Decision Processes.
	- `AbstractTabularMDP`: An abstract type representing tabular MDPs.
	"""
	abstract type AbstractCompleteMDP{T<:Real, S, A} <: AbstractTabularMDP{T, S, A} end 

	#for the special case of a deterministic environment, every probability is 1 so the function can be represented as an injective map

	"""
	    struct FiniteDeterministicMDP{T<:Real, S, A} <: AbstractCompleteMDP{T, S, A}
	
	A struct representing a finite deterministic Markov Decision Process (MDP) with fully specified transition dynamics.
	
	# Fields
	- `states::Vector{S}`: The vector of possible states.
	- `actions::Vector{A}`: The vector of possible actions.
	- `state_index::Dict{S, Int64}`: A dictionary mapping states to their corresponding indices.
	- `action_index::Dict{A, Int64}`: A dictionary mapping actions to their corresponding indices.
	- `state_transition_map::Matrix{Int64}`: A matrix representing the state transition function. Each element (i, j) corresponds to the index of the state reached from state j when taking action i.
	- `reward_transition_map::Matrix{T}`: A matrix representing the reward function. Each element (i, j) corresponds to the (average) reward received for the transition from state j when taking action i.
	
	# Description
	`FiniteDeterministicMDP` is a subtype of `AbstractCompleteMDP`, parameterized by three types:
	- `T<:Real`: The numerical type for the rewards.
	- `S`: The type representing the states.
	- `A`: The type representing the actions.
	
	This struct represents a finite deterministic MDP where the transition dynamics and rewards are fully specified and deterministic.
	
	# Notes
	- The `state_transition_map` and `reward_transition_map` matrices encode the transition dynamics and rewards, respectively, for all state-action pairs.
	
	See also
	---------
	- `AbstractCompleteMDP`: An abstract type representing complete MDPs with fully specified transition dynamics.
	"""
	struct FiniteDeterministicMDP{T<:Real, S, A} <: AbstractCompleteMDP{T, S, A}
		states::Vector{S}
		actions::Vector{A}
		state_index::Dict{S, Int64}
		action_index::Dict{A, Int64}
		state_transition_map::Matrix{Int64} #index of state reached from the state corresponding to the column when taking action corresponding to the row
		reward_transition_map::Matrix{T} #(average) reward received for the transition from the state corresponding to the column when taking action corresponding to the row
	end

	"""
	    FiniteDeterministicMDP(states::Vector{S}, actions::Vector{A}, state_transition_map::Matrix{Int64}, reward_transition_map::Matrix{T}) where {T<:Real, S, A}
	
	Constructs a `FiniteDeterministicMDP` representing a deterministic Markov Decision Process (MDP) with fully specified transition dynamics.
	
	# Arguments
	- `states::Vector{S}`: The vector of possible states.
	- `actions::Vector{A}`: The vector of possible actions.
	- `state_transition_map::Matrix{Int64}`: A matrix representing the state transition function. Each element (i, j) corresponds to the index of the state reached from state j when taking action i.
	- `reward_transition_map::Matrix{T}`: A matrix representing the reward function. Each element (i, j) corresponds to the (average) reward received for the transition from state j when taking action i.
	
	# Returns
	- `FiniteDeterministicMDP{T, S, A}`: A `FiniteDeterministicMDP` object representing the deterministic MDP.
	
	# Description
	This method constructs a `FiniteDeterministicMDP` object, which is a subtype of `AbstractCompleteMDP`, representing a deterministic MDP with fully specified transition dynamics.
	
	# Notes
	- The `state_transition_map` and `reward_transition_map` matrices encode the transition dynamics and rewards, respectively, for all state-action pairs.
	
	# Examples
	```julia
	states = ["s1", "s2", "s3"]
	actions = ["a1", "a2"]
	state_transition_map = [2 3; 1 3; 1 2]  # Example state transition map
	reward_transition_map = [1.0 0.0; 0.5 1.0; 0.0 0.5]  # Example reward transition map
	mdp = FiniteDeterministicMDP(states, actions, state_transition_map, reward_transition_map)
	"""
	FiniteDeterministicMDP(states::Vector{S}, actions::Vector{A}, state_transition_map::Matrix{Int64}, reward_transition_map::Matrix{T}) where {T<:Real, S, A} = FiniteDeterministicMDP{T, S, A}(states, actions, makelookup(states), makelookup(actions), state_transition_map, reward_transition_map)

	"""
	    struct FiniteStochasticMDP{T<:Real, S, A} <: AbstractCompleteMDP{T, S, A}
	
	A struct representing a finite stochastic Markov Decision Process (MDP) with fully specified transition dynamics.
	
	# Fields
	- `states::Vector{S}`: The vector of possible states.
	- `actions::Vector{A}`: The vector of possible actions.
	- `state_index::Dict{S, Int64}`: A dictionary mapping states to their corresponding indices.
	- `action_index::Dict{A, Int64}`: A dictionary mapping actions to their corresponding indices.
	- `ptf::Dict{Tuple{Int64, Int64}, Dict{Int64, Tuple{T, T}}}`: A dictionary representing the probability transition function. For each state-action pair index, there is a corresponding dictionary mapping each transition state index to the probability of that transition and the average reward received.
	
	# Description
	`FiniteStochasticMDP` is a subtype of `AbstractCompleteMDP`, parameterized by three types:
	- `T<:Real`: The numerical type for the rewards.
	- `S`: The type representing the states.
	- `A`: The type representing the actions.
	
	This struct represents a finite stochastic MDP where the transition dynamics are fully specified and stochastic.
	
	# Notes
	- The `ptf` field encodes the probability transition function, representing the probability of transitioning to each state and the associated rewards for each state-action pair.
	
	See also
	---------
	- `AbstractCompleteMDP`: An abstract type representing complete MDPs with fully specified transition dynamics.
	"""
	struct FiniteStochasticMDP{T<:Real, S, A} <: AbstractCompleteMDP{T, S, A}
		states::Vector{S}
		actions::Vector{A}
		state_index::Dict{S, Int64}
		action_index::Dict{A, Int64}
		ptf::Dict{Tuple{Int64, Int64}, Dict{Int64, Tuple{T, T}}} #for each state action pair index there is a corresponding dictionary mapping each transition state index to the probability of that transition and the average reward received
	end

	"""
	    FiniteStochasticMDP(states::Vector{S}, actions::Vector{A}, ptf::Dict{Tuple{Int64, Int64}, Dict{Int64, Tuple{T, T}}}) where {T<:Real, S, A}
	
	Constructs a `FiniteStochasticMDP` representing a stochastic Markov Decision Process (MDP) with fully specified transition dynamics.
	
	# Arguments
	- `states::Vector{S}`: The vector of possible states.
	- `actions::Vector{A}`: The vector of possible actions.
	- `ptf::Dict{Tuple{Int64, Int64}, Dict{Int64, Tuple{T, T}}}`: A dictionary representing the probability transition function. For each state-action pair index, there is a corresponding dictionary mapping each transition state index to the probability of that transition and the average reward received.
	
	# Returns
	- `FiniteStochasticMDP{T, S, A}`: A `FiniteStochasticMDP` object representing the stochastic MDP.
	
	# Description
	This method constructs a `FiniteStochasticMDP` object, which is a subtype of `AbstractCompleteMDP`, representing a stochastic MDP with fully specified transition dynamics.
	
	# Notes
	- The `ptf` argument encodes the probability transition function, representing the probability of transitioning to each state and the associated rewards for each state-action pair.
	
	# Examples
	```julia
	states = ["s1", "s2", "s3"]
	actions = ["a1", "a2"]
	ptf = Dict{(Int64, Int64), Dict{Int64, Tuple{Float64, Float64}}}()
	ptf[(1, 1)] = Dict(1 => (0.8, 1.0), 2 => (0.2, 0.5))  # Example probability transition function for state-action pair (1, 1)
	ptf[(1, 2)] = Dict(2 => (1.0, 0.0))  # Example probability transition function for state-action pair (1, 2)
	mdp = FiniteStochasticMDP(states, actions, ptf)
	"""
	FiniteStochasticMDP(states::Vector{S}, actions::Vector{A}, ptf::Dict{Tuple{Int64, Int64}, Dict{Int64, Tuple{T, T}}}) where {T<:Real, S, A} = FiniteStochasticMDP{T, S, A}(states, actions, makelookup(states), makelookup(actions), ptf)
end

# ╔═╡ 3165f2d7-38a2-4852-98aa-afa4cabfb2ed
"""
    initialize_state_action_value(mdp::AbstractTabularMDP{T, S, A}; init_value = zero(T)) where {T<:Real, S, A}

Initializes the state-action value function for a tabular Markov Decision Process (MDP).

# Arguments
- `mdp::AbstractTabularMDP{T, S, A}`: The tabular MDP for which to initialize the state-action value function.
- `init_value::T`: (Optional) The initial value for each state-action pair. Default is `zero(T)`.

# Returns
- `Matrix{T}`: A matrix representing the initialized state-action value function.

# Description
This function initializes the state-action value function for a tabular MDP. Each element of the matrix represents the value of taking an action in a particular state represented by the row and column index respectively.
```
"""
initialize_state_action_value(mdp::AbstractTabularMDP{T, S, A}; init_value = zero(T)) where {T<:Real, S, A} = ones(T, length(mdp.actions), length(mdp.states)) .* init_value

# ╔═╡ fa07a49b-68fb-4478-a29b-9289f6a3d56a
"""
    initialize_state_value(mdp::AbstractTabularMDP{T, S, A}; init_value = zero(T)) where {T<:Real, S, A}

Initializes the state value function for a tabular Markov Decision Process (MDP).

# Arguments
- `mdp::AbstractTabularMDP{T, S, A}`: The tabular MDP for which to initialize the state value function.
- `init_value::T`: (Optional) The initial value for each state. Default is `zero(T)`.

# Returns
- `Vector{T}`: A vector representing the initialized state value function.

# Description
This function initializes the state value function for a tabular MDP. Each element of the vector represents the value of being in a particular state represented by the index.
"""
initialize_state_value(mdp::AbstractTabularMDP{T, S, A}; init_value = zero(T)) where {T<:Real, S, A} = ones(T, length(mdp.states)) .* init_value

# ╔═╡ 4e738b67-0d03-486a-a2cd-d18144a7fcd7
"""
    find_terminal_states(mdp::AbstractCompleteMDP)

Finds the terminal states in a tabular Markov Decision Process (MDP) wiht fully specified dynamics

# Returns
- `Set`: A set containing the indices of terminal states.

# Description
This function identifies the terminal states in a finite MDP. A terminal state is defined as a state from which no transitions occur to other states and where the reward is zero for all actions.

"""
find_terminal_states(mdp::AbstractCompleteMDP) = find_terminal_states(mdp)

# ╔═╡ 06f6647d-48c5-4ead-b7b5-90a968363215
@skip_as_script md"""
### *Example: Creating Deterministic and Stochastic Gridworld MDPs*
"""

# ╔═╡ 92556e91-abae-4ce3-aa15-b35c4a65cff5
begin
	abstract type GridworldAction end
	struct Up <: GridworldAction end
	struct Down <: GridworldAction end
	struct Left <: GridworldAction end
	struct Right <: GridworldAction end
	struct UpRight <: GridworldAction end
	struct DownRight <: GridworldAction end
	struct UpLeft <: GridworldAction end
	struct DownLeft <: GridworldAction end
	struct Stay <: GridworldAction end
	
	struct GridworldState
		x::Int64
		y::Int64
	end

	import Base.==, Base.hash

	function (==)(s1::GridworldState, s2::GridworldState)
		(s1.x == s2.x) && (s1.y == s2.y)
	end

	function hash(s::GridworldState) 
		hash([s.x, s.y])
	end

	#rectilinear actions
	const rook_actions = [Up(), Down(), Left(), Right()]
	
	move(::Up, x, y) = (x, y+1)
	move(::Down, x, y) = (x, y-1)
	move(::Left, x, y) = (x-1, y)
	move(::Right, x, y) = (x+1, y)
	move(::UpRight, x, y) = (x+1, y+1)
	move(::UpLeft, x, y) = (x-1, y+1)
	move(::DownRight, x, y) = (x+1, y-1)
	move(::DownLeft, x, y) = (x-1, y-1)
	move(::Stay, x, y) = (x, y)

	"""
		make_deterministic_gridworld(; kwargs...) -> NamedTuple{(:mdp, :isterm, :init_state), Tuple{FiniteDeterministicMDP, Function, Integer}}
	
	Create a deterministic Gridworld MDP with the given parameters.
	
	Keyword Arguments:
	- actions: The actions available in the environment (rook_actions)
	- start: The starting state (GridworldState(1, 4))
	- sterm: The terminal state (GridworldState(8, 4))
	- xmax: The maximum x-coordinate (10)
	- ymax: The maximum y-coordinate (7)
	- stepreward: The reward for each step (0.0f0)
	- termreward: The reward for reaching the terminal state (1.0f0)
	- iscliff: A function to check if a state is a cliff (s -> false)
	- iswall: A function to check if a state is a wall (s -> false)
	- cliffreward: The reward for falling off a cliff (-100f0)
	- goal2: The second goal state (GridworldState(start.x, ymax))
	- goal2reward: The reward for reaching the second goal state (0.0f0)
	- usegoal2: Whether to use the second goal state (false)
	- wind: The wind direction (zeros(Int64, xmax))
	- continuing: Whether the environment is continuing (false)
	
	Returns:
	- A named tuple containing:
	    - mdp: A FiniteDeterministicMDP instance
	    - isterm: A function to check if a state is terminal
	    - init_state: The initial state index
	"""
	function make_deterministic_gridworld(;
		actions = rook_actions, 
		start = GridworldState(1, 4),
		sterm = GridworldState(8, 4), 
		xmax = 10, 
		ymax = 7, 
		stepreward = 0.0f0, 
		termreward = 1.0f0, 
		iscliff = s -> false, 
		iswall = s -> false, 
		cliffreward = -100f0, 
		goal2 = GridworldState(start.x, ymax), 
		goal2reward = 0.0f0, 
		usegoal2 = false,
		wind = zeros(Int64, xmax),
		continuing = false)

		@assert length(wind) == xmax
		@assert all(x -> x >= 0, wind)
		
		#define the state space
		states = [GridworldState(x, y) for x in 1:xmax for y in 1:ymax]
		
		boundstate(x::Int64, y::Int64) = (clamp(x, 1, xmax), clamp(y, 1, ymax))

		#take a deterministic step in the environment and produce the transition state s′	
		function step(s::GridworldState, a::GridworldAction)
			w = wind[s.x]
			(x, y) = move(a, s.x, s.y)
			y += w
			s′ = GridworldState(boundstate(x, y)...)
			iswall(s′) && return s
			return s′
		end

		state_index = makelookup(states)
		action_index = makelookup(actions)

		i_start = state_index[start]
		i_sterm = state_index[sterm]
		i_goal2 = state_index[goal2]
		#determines if a state is terminal

		function isterm(i_s::Integer) 
			i_s == i_sterm && return true
			usegoal2 && (i_s == i_goal2) && return true
			return false
		end

		state_transition_map = zeros(Int64, length(actions), length(states))
		reward_transition_map = zeros(Float32, length(actions), length(states))
		for s in states
			i_s = state_index[s] #get index for starting state
			if isterm(i_s)
				if continuing
					state_transition_map[:, i_s] .= i_start
				else
					state_transition_map[:, i_s] .= i_s
				end
				reward_transition_map[:, i_s] .= 0f0
			else
				for a in actions
					i_a = action_index[a] #get index for action
					s′ = step(s, a)
					i_s′ = state_index[s′] #get index for transition state
					state_transition_map[i_a, i_s] = i_s′
					reward = if isterm(i_s)
						0f0
					elseif iscliff(s′)
						state_transition_map[i_a, i_s] = i_start
						cliffreward
					elseif usegoal2 && (s′ == goal2)
						goal2reward
					elseif isterm(i_s′)
						termreward
					else
						stepreward
					end
					reward_transition_map[i_a, i_s] = reward
				end
			end
		end
		(mdp = FiniteDeterministicMDP(states, actions, state_index, action_index, state_transition_map, reward_transition_map), isterm = isterm, init_state = state_index[start])
	end
end

# ╔═╡ 48954b7d-5165-4c4f-9af1-ee4217af5127
"""
    find_terminal_states(mdp::FiniteDeterministicMDP)

Finds the terminal states in a finite deterministic Markov Decision Process (MDP).
"""
function find_terminal_states(mdp::FiniteDeterministicMDP{T, S, A}) where {T<:Real, S, A}
    Set(findall(eachindex(mdp.states)) do i_s
        all((i_s′ == i_s) for i_s′ in view(mdp.state_transition_map, :, i_s)) && iszero(sum(view(mdp.reward_transition_map, :, i_s)))
    end)
end

# ╔═╡ ac91e5e8-a282-4c36-ad7e-3fed3b30855a
"""
    find_terminal_states(mdp::FiniteStochasticMDP)

Identify terminal states in a finite stochastic MDP.

# Examples
```julia
julia> states = [1, 2, 3]
julia> actions = [:a, :b]
julia> ptf = Dict((1, 1) => Dict(1 => (1.0, 0.0)), (2, 1) => Dict(2 => (0.8, 1.0)), (3, 1) => Dict(3 => (1.0, 0.0)))
julia> mdp = FiniteStochasticMDP(states, actions, ptf)
julia> find_terminal_states(mdp)
Set([1, 3])
```
"""
function find_terminal_states(mdp::FiniteStochasticMDP{T, S, A}) where {T<:Real, S, A}
    Set(findall(eachindex(mdp.states)) do i_s
        all(eachindex(mdp.actions)) do i_a
			if haskey(mdp.ptf, (i_s, i_a))
				d = mdp.ptf[(i_s, i_a)]
				if haskey(d, i_s)
					d[i_s] == (one(T), zero(T))
				else
					false
				end
			else
				true
			end
		end
	end)
end

# ╔═╡ 7d7527be-2cfa-4c7b-8344-8049d91835b0
"""
    make_isterm(mdp::AbstractCompleteMDP{T, S, A}) where {T<:Real, S, A}

Create a function to determine if a state is terminal in a given MDP.

# Arguments
- `mdp::AbstractCompleteMDP{T, S, A}`: An abstract complete Markov Decision Process with reward type `T`, state type `S`, and action type `A`.

# Returns
- `isterm::Function`: A function that takes a state index `i_s::Integer` and returns `true` if the state is terminal, and `false` otherwise.

# Description
This function identifies terminal states in the provided MDP and returns a function that can be used to check if a given state index corresponds to a terminal state.
"""
function make_isterm(mdp::AbstractCompleteMDP{T, S, A}) where {T<:Real, S, A}
	terminds = find_terminal_states(mdp)
	isterm(i_s::Integer) = in(i_s, terminds)
end

# ╔═╡ fef1b14a-5495-439d-9428-338be5c4f6e8
"""
	make_stochastic_gridworld(; kwargs...) -> NamedTuple{(:mdp, :isterm, :init_state), Tuple{FiniteStochasticMDP, Function, Integer}}

Create a stochastic Gridworld MDP with the given parameters.

Keyword Arguments:
- actions: The actions available in the environment (rook_actions)
- start: The starting state (GridworldState(1, 4))
- sterm: The terminal state (GridworldState(8, 4))
- xmax: The maximum x-coordinate (10)
- ymax: The maximum y-coordinate (7)
- stepreward: The reward for each step (0.0f0)
- termreward: The reward for reaching the terminal state (1.0f0)
- iscliff: A function to check if a state is a cliff (s -> false)
- iswall: A function to check if a state is a wall (s -> false)
- cliffreward: The reward for falling off a cliff (-100f0)
- goal2: The second goal state (GridworldState(start.x, ymax))
- goal2reward: The reward for reaching the second goal state (0.0f0)
- usegoal2: Whether to use the second goal state (false)
- wind: The wind direction (zeros(Int64, xmax))
- continuing: Whether the environment is continuing (false)

Returns:
- A named tuple containing:
    - mdp: A FiniteStochasticMDP instance
    - isterm: A function to check if a state is terminal
    - init_state: The initial state index
"""
function make_stochastic_gridworld(;
	actions = rook_actions, 
	start = GridworldState(1, 4),
	sterm = GridworldState(8, 4), 
	xmax = 10, 
	ymax = 7, 
	stepreward = 0.0f0, 
	termreward = 1.0f0, 
	iscliff = s -> false, 
	iswall = s -> false, 
	cliffreward = -100f0, 
	goal2 = GridworldState(start.x, ymax), 
	goal2reward = 0.0f0, 
	usegoal2 = false,
	wind = zeros(Int64, xmax),
	continuing = false)

	@assert length(wind) == xmax
	@assert all(x -> x >= 0, wind)

	#define the state space
	states = [GridworldState(x, y) for x in 1:xmax for y in 1:ymax]
	
	boundstate(x::Int64, y::Int64) = (clamp(x, 1, xmax), clamp(y, 1, ymax))

	#take a stochastic step in the environment and produce the transition states and their associated probabilities
	function step(s::GridworldState, a::GridworldAction)
		w = wind[s.x]
		(x, y) = move(a, s.x, s.y)
		s′ = GridworldState(boundstate(x, y)...)
		output = Dict{GridworldState, Float32}()
		if iszero(w)
			if iswall(s′)
				output[s] = 1f0
			else
				output[s′] = 1f0
			end
		else
			for w in w-1:w+1
				s′ = GridworldState(boundstate(x, y + w)...)
				if iswall(s′)
					s′ = s
				end
				if haskey(output, s′)
					output[s′] += 1f0/3
				else
					output[s′] = 1f0/3
				end
			end
		end
		return output
	end


	state_index = makelookup(states)
	action_index = makelookup(actions)

	i_start = state_index[start]
	i_sterm = state_index[sterm]
	i_goal2 = state_index[goal2]
	#determines if a state is terminal
	function isterm(i_s::Integer) 
		i_s == i_sterm && return true
		usegoal2 && (i_s == i_goal2) && return true
		return false
	end

	ptf = Dict{Tuple{Int64, Int64}, Dict{Int64, Tuple{Float32, Float32}}}()
	for s in states
		i_s = state_index[s] #get index for starting state
		if isterm(i_s)
			for i_a in eachindex(actions)
				if continuing
					ptf[(i_s, i_a)] = Dict([i_start => (1f0, 0f0)])
				else
					ptf[(i_s, i_a)] = Dict([i_s => (1f0, 0f0)])
				end
			end
		else
			for a in actions
				d = Dict{Int64, Tuple{Float32, Float32}}()
				i_a = action_index[a] #get index for action
				output = step(s, a)
				for s′ in keys(output)
					i_s′ = state_index[s′] #get index for transition state
					if iscliff(s′)
						d[i_start] = (output[s′], cliffreward)
					elseif usegoal2 && (s′ == goal2)
						d[i_s′] = (output[s′], goal2reward)
					elseif isterm(i_s′)
						d[i_s′] = (output[s′], termreward)
					else
						d[i_s′] = (output[s′], stepreward)
					end
				end
				ptf[(i_s, i_a)] = d
			end
		end
	end
	(mdp = FiniteStochasticMDP(states, actions, state_index, action_index, ptf), isterm = isterm, init_state = state_index[start])
end

# ╔═╡ 1188e680-cfbe-417c-ad61-83e145c39220
@skip_as_script md"""
##### Create a gridworld with all the necessary components shown below.  Included is an example of a deterministic gridworld without wind, a deterministic gridworld with wind, and a stochastic gridworld with wind.
"""

# ╔═╡ 10d4576c-9b86-469c-83b7-1e3d3bc21da1
@skip_as_script const deterministic_gridworld = make_deterministic_gridworld()

# ╔═╡ be227f6e-6d25-4a4a-97ab-21ecd6af917e
@skip_as_script const wind_values = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# ╔═╡ f750ec24-b9a0-4b4e-88ee-c6e4867103c7
@skip_as_script const windy_gridworld = make_deterministic_gridworld(;wind = wind_values)

# ╔═╡ b0059e3e-0351-4af7-a60b-56896e2b1a05
@skip_as_script const stochastic_gridworld = make_stochastic_gridworld(; wind = wind_values)

# ╔═╡ 3b3decd0-bb00-4fd2-a8eb-a5b14aede950
@skip_as_script md"""
##### Deterministic gridworld transition display.  Given a state action pair defined below, shows the corresponding state in the grid highlighted in blue and the transition state outlined in bold.  The start and goal states are also shown in green and gold respectively.  Notice that if the selected state is the goal, then all transitions remain in that state.
"""

# ╔═╡ e14350ea-5a00-4a8f-8b81-f751c69b67a6
@skip_as_script @htl("""
<div style = "display: flex; justify-content: flex-start; background-color:gray; color:black;">
<div>Selected State</div>
<div style = "width:20px; height:20px; background-color: rgb(0, 0, 255, 0.4); margin-top: 5px; margin-left: 10px; margin-right: 10px; border: 2px solid black;"></div>
<div>$(@bind highlight_state_index Slider(eachindex(deterministic_gridworld.mdp.states), show_value=true))</div>


</div>
</div>
<div style = "display: flex; background-color: gray; color:black">
Transition State 
<div style = "width:20px; height:20px; border: 4px solid black; background-color: white; margin-left: 10px">
</div>
""")

# ╔═╡ 770c4392-6285-4e00-8d72-5c6a132d8aa9
@skip_as_script md"""Selected Action $(@bind grid_action_selection Slider(1:4; show_value = true))"""

# ╔═╡ 0fca8f38-f282-4168-87d3-aab0ec0c6346
@skip_as_script md"""
##### Stochastic gridworld transition display. With stochastic wind, when wind is present there is an equal probability of experiencing w-1, w, and w+1 for the wind value
"""

# ╔═╡ 4b277cea-668e-43d6-bd2a-fcbf62be9b12
@skip_as_script md"""
### Agent Behavior: The Policy Function
An *agent* is often defined by a specific *policy* $\pi(a\vert s) = \text{Pr} \{A_t = a \mid S_t = s \}$ which defines the probabilities of taking an action given a state.  If there are multiple actions with non-zero probability for a given state, then this is a *stochastic* policy.  To handle stochastic policies in general, a generic policy can be defined as matrix of probabilities where each column represents the action distribution for the state represented by the column index.  Defining a policy like this takes advantage of the fact that we can enumerate all the state action pairs and thus represent them with a numerical index.  An agent following such a stochastic policy will sample from the action distribution every time it encounters a state.
"""

# ╔═╡ 82f710d7-6ae8-4794-af2d-762ee3a73a3f
@skip_as_script md"""
### *Policies, Action Selection, and Trajectories*
"""

# ╔═╡ 8cae3e2f-9fb8-485a-bdc7-3fff48a2f9b5
"""
    sample_action(π::Matrix{T}, i_s::Integer) where T<:AbstractFloat

Samples an action index from a probability distribution represented by a matrix.

# Arguments
- `π::Matrix{T}`: A matrix representing the probability distribution over actions for each state. Each column `π[:, i_s]` represents the probability distribution over actions in state `i_s`.
- `i_s::Integer`: The index of the current state.

# Returns
- `Int`: The sampled action index.

# Description
This function samples an action index from a probability distribution represented by a matrix `π`. The matrix `π` represents the probability distribution over actions for each state. The distribution for the current state `i_s` is given by the column `π[:, i_s]`. The sampling is performed using the `sample_action` function, which samples from a probability distribution represented by a vector using the Gumbel-max trick.
"""
function sample_action(π::Matrix{T}, i_s::Integer) where T<:AbstractFloat
	(n, m) = size(π)
	sample_action(view(π, :, i_s))
end

# ╔═╡ 26285297-5614-41bd-9ec4-428d37d1dd3e
"""
    sample_action(v::AbstractVector{T}) where T<:AbstractFloat

Samples an action index from a probability distribution represented by a vector.

# Arguments
- `v::AbstractVector{T}`: A vector representing the probability distribution over actions.

# Returns
- `Int`: The sampled action index.

# Description
This function samples an action index from a probability distribution represented by a vector `v`. Each element `v[i]` represents the probability of selecting action `i`. The sampling is performed using the Gumbel-max trick, which avoids the need to compute softmax probabilities explicitly.
"""
function sample_action(v::AbstractVector{T}) where T<:AbstractFloat 
	i_a = 1
	maxv = T(-Inf)
	@inbounds @fastmath for (i, x) in enumerate(v)
		g = log(x) - log(-log(rand(T)))
		newmax = (g > maxv)
		maxv = max(g, maxv)
		i_a += newmax*(i - i_a)
	end
	return i_a
	# sample(eachindex(v), weights(v))
end

# ╔═╡ 19114bac-a4b1-408e-a7ca-26454b894f72
"""
    make_random_policy(mdp::AbstractTabularMDP{T, S, A}) where {T, S, A}

Creates a random policy for a tabular Markov Decision Process (MDP).

# Returns
- `Matrix{T}`: A matrix representing the random policy. Each element `π[i, j]` denotes the probability of taking the action represented by index `i` in the state represented by index `j`.

# Description
This function creates a random policy for a tabular Markov Decision Process (MDP). The policy is represented as a matrix `π`, where each row corresponds to an action and each column corresponds to a state. Each element `π[i, j]` denotes the probability of taking the action represented by index `i` in the state represented by index `j`. In the random policy, each action in each state has an equal probability of being selected.
"""
make_random_policy(mdp::AbstractTabularMDP{T, S, A}) where {T, S, A} = ones(T, length(mdp.actions), length(mdp.states)) ./ length(mdp.actions)

# ╔═╡ 3709ca81-7ad9-453b-a8ba-929bb4031e9b
function get_transition(mdp::FiniteDeterministicMDP, i_s::Integer, i_a::Integer)
	i_s′ = mdp.state_transition_map[i_a, i_s]
	r = mdp.reward_transition_map[i_a, i_s]
	return (r, i_s′)
end

# ╔═╡ e1bd5582-c734-4597-9fdd-2ee0221fb35d
function get_transition(mdp::FiniteStochasticMDP, i_s::Integer, i_a::Integer)
	ptf = mdp.ptf[(i_s, i_a)]
	probabilities = [ptf[i_s′][1] for i_s′ in keys(ptf)]
	i_s′ = sample(collect(keys(ptf)), weights(probabilities))
	s′ = mdp.states[i_s′]
	r = ptf[i_s′][2]
	return (r, i_s′)
end

# ╔═╡ 035a6f5c-3bed-4f72-abe5-17558331f8ba
@skip_as_script md"""Matrix representation of a random policy"""

# ╔═╡ 84815181-244c-4f57-8bf0-7617379dda00
@skip_as_script md"""Visual representation of a random policy"""

# ╔═╡ 08b70e16-f113-4464-bb4b-3da393c8500d
@skip_as_script md"""
Random policy episode returns the trajectory as a list of states visited, actions taken, and rewards received.  The final state of the episode is also shown."""

# ╔═╡ 73c4f222-a405-493c-9127-0f950cd5fa0e
@skip_as_script md"""
## The Value Function

### Goals and Rewards

Our objective in *solving* and MDP is to maximize the expected value of what is called the *discounted future return*.  

$\begin{flalign}
G_t & \doteq \sum_{k=0}^\infty \gamma^k R_{t+k+1} \text{ or } \sum_{k = t+1} ^ T \gamma^{k-t-1}R_k \tag{3.8/3.11} \\
&= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \\
&= R_{t+1} + \gamma \left [ R_{t+2} + \gamma R_{t+3} + \cdots \right ] \\
&= R_{t+1} + \gamma G_{t+1} \tag{3.9}
\end{flalign}$

where $0 \lt \gamma \le 1$ in general and $0 \lt \gamma \lt 1$ for continuing tasks that do not have a terminal state.

The recursive expression for $G_t$ is important to defining our approach to solving the problem.  Given a specific policy $\pi$ and an environment with a state space $\mathcal{S}$, we can define the *value function* for a policy as follows:

### Policy Value Functions

$\begin{flalign}
v_\pi(s) &\doteq \mathbb{E}_\pi [G_t \mid S_t = s] \tag{3.12}\\
&= \sum_a \pi(a \vert s) \mathbb{E}_\pi [G_{t} \mid S_t = s, A_t = a] \tag{exp value def} \\
&= \sum_a \pi(a \vert s) q_\pi(s, a) \tag{by definition of q (1)} \\
&= \sum_a \pi(a \vert s) \sum_{s^\prime, r} p(s^\prime, r \vert s, a) [r + \gamma v_\pi(s^\prime)] \tag{by (4.6) (3.14)}\\

q_\pi(s, a) &\doteq \mathbb{E}_\pi[G_t \mid S_t=s,A_t=a] \tag{3.13} \\
& = \mathbb{E}_\pi \left [ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a \right ] \tag{by (3.9)} \\
& = \sum_{s^\prime, r} p(s^\prime, r \vert s, a) \mathbb{E}_\pi \left [ r + \gamma G_{t+1} \mid S_{t+1} = s^\prime \right ] \tag{exp value def}\\
& = \sum_{s^\prime, r} p(s^\prime, r \vert s, a) \left [ r + \gamma \mathbb{E}_\pi [G_{t+1} \mid S_{t+1} = s^\prime] \right ] \\
& = \sum_{s^\prime, r} p(s^\prime, r \vert s, a) [r + \gamma v_\pi(s^\prime)] \tag{by definition of v (4.6)} \\
& = \sum_{s^\prime, r} p(s^\prime, r \vert s, a) [r + \gamma\sum_{a^\prime} \pi(a^\prime \vert s^\prime) q_\pi(s^\prime, a^\prime)] \tag{by (1)} \\
\end{flalign}$
"""

# ╔═╡ c4e1d754-2535-40be-bbb3-075ca3fa64b9
@skip_as_script md"""
For a policy $\pi$, $v_\pi(s)$ is called the *state value function* and $q_\pi(s, a)$ is called the *state-action value function*. Notice that both expressions have a recursive form that defines values in terms of successor states.  Those recursive equations are known as the *Bellman Equations* for each value function.

Since we have a finite and countable number of state action pairs, each value function can be represented as a vector or matrix whose indices represent the states and actions corresponding to that value estimate.  Given a value function and a policy, we can verify whether or not it satisfies the Bellman Equation everywhere.  If it does, then we have the correct value function for that policy.  In other words, the correct value function is a *fixed point* of the *Bellman Operator* where the *Bellman Operator* is the act of updating the value function with the right hand side of the Bellman Equation.  

Verifying that a value function is correct is simple, but what is less obvious is that we can use the Bellman Operator to compute the correct value function without knowing it in advance.  It can be proven that if we initialize our value function arbitrarily and update those values with the Bellman Operator, that process will converge to the true value function.  This iterative approach is one method of computing the value functions when we have a well defined policy and the probability transition function for an environment.
"""

# ╔═╡ 478aa9a3-ac58-4520-9613-3fcf1a1c1952
@skip_as_script md"""
### *Bellman Policy Evaluation*

The following code shows how one can use the Bellman Operator to iteratively calculate the value function for a given policy.  The policy must be defined in terms of a probability distribution over actions for each state in the environment.  This implementation is an extension of the prior code in which every state action pair can be enumerated in advance.
"""

# ╔═╡ ed7c22bf-2773-4ff7-93d0-2bd05cfef738
calc_pct_change(x_old, x_new) = abs(x_old - x_new) / (eps(abs(x_old)) + abs(x_old))

# ╔═╡ 18bc3870-3261-43d0-924b-46ca44a9e8ce
function bellman_policy_update!(Q::Matrix{T}, π::Matrix{T}, i_s::Int64, i_a::Int64, mdp::FiniteDeterministicMDP{T, S, A}, γ::T) where {T <: Real, S, A}
	i_s′ = mdp.state_transition_map[i_a, i_s]
	r = mdp.reward_transition_map[i_a, i_s]
	
	#perform a bellman optimal update for a given state action pair index and return the percentage change in value
	v = zero(T)
	@inbounds @fastmath @simd for i_a′ in eachindex(mdp.actions)
		v += π[i_a′, i_s′] * Q[i_a′, i_s′]
	end
	x = r + (γ * v)
	
	delt = calc_pct_change(Q[i_a, i_s], x)
	Q[i_a, i_s] = x
	return delt
end

# ╔═╡ 125214ee-9fc5-4976-a622-23f0ce4e3cd7
function bellman_policy_update!(Q::Matrix{T}, π::Matrix{T}, i_s::Int64, i_a::Int64, mdp::FiniteStochasticMDP{T, S, A}, γ::T) where {T <: Real, S, A}
	#perform a bellman optimal update for a given state action pair index and return the percentage change in value
	q_avg = zero(T)
	r_avg = zero(T)
	ptf = mdp.ptf[(i_s, i_a)]
	x = zero(T)
	for i_s′ in keys(ptf)
		(p, r) = ptf[i_s′]
		v = zero(T)
		@inbounds @fastmath @simd for i_a′ in eachindex(mdp.actions)
			v += π[i_a′, i_s′] * Q[i_a′, i_s′]
		end
		x += p*(r + γ * v)
	end
	delt = calc_pct_change(Q[i_a, i_s], x)
	Q[i_a, i_s] = x
	return delt
end

# ╔═╡ 7c9c22ee-f245-45e1-b1b3-e8d029468f65
function uniform_bellman_policy_value!(Q::Matrix{T}, π::Matrix{T}, mdp::AbstractCompleteMDP{T, S, A}, γ::T) where {T <: Real, S, A}
	delt = zero(T)
	num_updates = 0
	for i_s in eachindex(mdp.states)
		for i_a in eachindex(mdp.actions)
			delt = max(delt, bellman_policy_update!(Q, π, i_s, i_a, mdp, γ))
			num_updates += 1
		end
	end
	return delt, num_updates
end

# ╔═╡ 021f942f-affa-4fb6-92da-65290680643a
function uniform_bellman_policy_value!(V::Vector{T}, π::Matrix{T}, mdp::FiniteDeterministicMDP{T, S, A}, γ::T) where {T <: Real, S, A}
	delt = zero(T)
	num_updates = 0
	for i_s in eachindex(mdp.states)
		x = zero(T)
		for i_a in eachindex(mdp.actions)
			r = mdp.reward_transition_map[i_a, i_s]
			i_s′ = mdp.state_transition_map[i_a, i_s]
			x += π[i_a, i_s] * (r + γ*V[i_s′])
		end
		delt = max(delt, calc_pct_change(V[i_s], x))
		num_updates += 1
		V[i_s] = x
	end
	return delt, num_updates
end

# ╔═╡ 9d335e8c-714f-40ad-be4f-6c2d357bf1c3
function uniform_bellman_policy_value!(V::Vector{T}, π::Matrix{T}, mdp::FiniteStochasticMDP{T, S, A}, γ::T) where {T <: Real, S, A}
	delt = zero(T)
	num_updates = 0
	for i_s in eachindex(mdp.states)
		x = zero(T)
		for i_a in eachindex(mdp.actions)
			ptf = mdp.ptf[(i_s, i_a)]
			v_avg = zero(T)
			for i_s′ in keys(ptf)
				(p, r) = ptf[i_s′]
				v_avg += p * (r + γ*V[i_s′])
			end
			x += π[i_a, i_s] * v_avg
		end
		delt = max(delt, calc_pct_change(V[i_s], x))
		num_updates += 1
		V[i_s] = x
	end
	return delt, num_updates
end

# ╔═╡ 9925509b-ee7e-430c-a646-fbf59bc75e62
"""
    policy_evaluation!(value_estimate::Array{T, N}, π::Matrix{T}, mdp::AbstractCompleteMDP{T, S, A}, γ::T; max_updates = typemax(Int64), θ = eps(zero(T))) where {T <: Real, S, A, N}

Performs policy evaluation in place to estimate value function.

# Arguments
- `value_estimate::Array{T, N}`: The array representing the initial value estimates for each state or state-action pair. This array is modified in place.
- `π::Matrix{T}`: The policy matrix representing the probability of selecting each action in each state.
- `mdp::AbstractCompleteMDP{T, S, A}`: The complete MDP for which policy evaluation is performed.
- `γ::T`: The discount factor.
- `max_updates::Int = typemax(Int64)`: (Optional) The maximum number of updates allowed during policy evaluation. Default is `typemax(Int64)`.
- `θ::T = eps(zero(T))`: (Optional) The threshold for convergence. Policy evaluation stops when the maximum change in value estimates is less than or equal to `θ`. Default is `eps(zero(T))`.

# Returns
- `Tuple{Array{T, N}, Int, Int}`: A tuple containing the updated value estimates (`value_estimate`), the number of iterations performed (`iter`), and the total number of updates made (`total_updates`).

# Description
This function performs policy evaluation in place to estimate the value function for a given policy `π` in a complete Markov Decision Process (MDP) `mdp`. The value estimates for each state or state-action pair are updated iteratively until convergence, where the maximum change in value estimates is less than or equal to the threshold `θ`, or the maximum number of updates `max_updates` is reached.

Policy evaluation is performed using the Bellman equation and the uniform Bellman update rule. The discount factor `γ` determines the weight given to future rewards.

The function modifies the `value_estimate` array in place and returns a tuple containing the updated value estimates (`value_estimate`), the number of iterations performed (`iter`), and the total number of updates made (`total_updates`).
"""
function policy_evaluation!(value_estimate::Array{T, N}, π::Matrix{T}, mdp::AbstractCompleteMDP{T, S, A}, γ::T; max_updates = typemax(Int64), θ = eps(zero(T))) where {T <: Real, S, A, N}
	delt, num_updates = uniform_bellman_policy_value!(value_estimate, π, mdp, γ)
	total_updates = num_updates
	iter = 1
	while (delt > θ) && (total_updates <= max_updates)
		delt, num_updates = uniform_bellman_policy_value!(value_estimate, π, mdp, γ)
		total_updates += num_updates
		iter += 1
	end
	return value_estimate, iter, total_updates
end

# ╔═╡ 43da70fd-e3c4-4d2d-9204-29aa5007df63
"""
    policy_evaluation_q(mdp::AbstractCompleteMDP{T, S, A}, π::Matrix{T}, γ::T; kwargs...) where {T<:Real,S, A}

Performs policy evaluation to estimate the action-value function (Q-function) for a given policy.

# Arguments
- `mdp::AbstractCompleteMDP{T, S, A}`: The complete MDP for which policy evaluation is performed.
- `π::Matrix{T}`: The policy matrix representing the probability of selecting each action in each state.
- `γ::T`: The discount factor.
- `kwargs...`: Additional keyword arguments to customize the behavior of the policy evaluation. See `policy_evaluation!` documentation for details.

# Returns
- `NamedTuple`: A named tuple containing the following fields:
  - `value_function::Array{T, 2}`: The estimated action-value function (Q-function).
  - `total_iterations::Int`: The total number of iterations performed during policy evaluation.
  - `total_updates::Int`: The total number of updates made during policy evaluation.

# Description
This function performs policy evaluation to estimate the action-value function (Q-function) for a given policy `π` in a complete Markov Decision Process (MDP) `mdp`. The Q-function represents the expected cumulative reward of taking a particular action in a particular state and following the given policy thereafter.

The policy evaluation is performed using the `policy_evaluation!` function, which updates the Q-function iteratively until convergence. The discount factor `γ` determines the weight given to future rewards.

Additional customization of the policy evaluation process can be achieved by providing keyword arguments `kwargs`, which are passed to the `policy_evaluation!` function.

The function returns a named tuple containing the estimated action-value function (`value_function`), the total number of iterations performed during policy evaluation (`total_iterations`), and the total number of updates made during policy evaluation (`total_updates`).
"""
function policy_evaluation_q(mdp::AbstractCompleteMDP{T, S, A}, π::Matrix{T}, γ::T; kwargs...) where {T<:Real,S, A}
	Q = initialize_state_action_value(mdp)
	(Q, total_iterations, total_updates) = policy_evaluation!(Q, π, mdp, γ; kwargs...)
	return (value_function = Q, total_iterations = total_iterations, total_updates = total_updates)
end

# ╔═╡ 823a8e5d-2092-480f-ad6c-4fc9e83e88c0
"""
    policy_evaluation_v(mdp::AbstractCompleteMDP{T, S, A}, π::Matrix{T}, γ::T; kwargs...) where {T<:Real,S, A}

Performs policy evaluation to estimate the state-value function (V-function) for a given policy.

# Arguments
- `mdp::AbstractCompleteMDP{T, S, A}`: The complete MDP for which policy evaluation is performed.
- `π::Matrix{T}`: The policy matrix representing the probability of selecting each action in each state.
- `γ::T`: The discount factor.
- `kwargs...`: Additional keyword arguments to customize the behavior of the policy evaluation. See `policy_evaluation!` documentation for details.

# Returns
- `NamedTuple`: A named tuple containing the following fields:
  - `value_function::Array{T, 1}`: The estimated state-value function (V-function).
  - `total_iterations::Int`: The total number of iterations performed during policy evaluation.
  - `total_updates::Int`: The total number of updates made during policy evaluation.

# Description
This function performs policy evaluation to estimate the state-value function (V-function) for a given policy `π` in a complete Markov Decision Process (MDP) `mdp`. The V-function represents the expected cumulative reward of starting from a particular state and following the given policy thereafter.

The policy evaluation is performed using the `policy_evaluation!` function, which updates the V-function iteratively until convergence. The discount factor `γ` determines the weight given to future rewards.

Additional customization of the policy evaluation process can be achieved by providing keyword arguments `kwargs`, which are passed to the `policy_evaluation!` function.

The function returns a named tuple containing the estimated state-value function (`value_function`), the total number of iterations performed during policy evaluation (`total_iterations`), and the total number of updates made during policy evaluation (`total_updates`).
"""
function policy_evaluation_v(mdp::AbstractCompleteMDP{T, S, A}, π::Matrix{T}, γ::T; kwargs...) where {T<:Real,S, A}
	V = initialize_state_value(mdp)
	(V, total_iterations, total_updates) = policy_evaluation!(V, π, mdp, γ; kwargs...)
	return (value_function = V, total_iterations = total_iterations, total_updates = total_updates)
end

# ╔═╡ 381bfc1e-9bc4-47f7-a8d3-116933382e25
@skip_as_script md"""
#### *Example: Random Policy Evaluation for Gridworlds*
"""

# ╔═╡ b991831b-f15d-493c-835c-c7e8a33f8d7b
@skip_as_script md"""
State values for the random policy.  Notice that at a discount rate of $\gamma=1$ all of the state values will be identical with a value of 1.  If the sole reward is for reaching the goal, a discount factor must be used to favor reaching the goal as fast as possible.  Otherwise any policy that eventually reaches the goal will be considered equally good.
"""

# ╔═╡ e6beff79-061c-4c01-b469-75dc5d4e059f
@skip_as_script md"""Select Discount Rate for State Policy Evaluation: $(@bind γ_gridworld_policy_evaluation Slider(0.01f0:0.01f0:1f0; show_value=true, default = 1f0))"""

# ╔═╡ cb96b24a-65aa-4832-bc7d-093f0c951f83
@skip_as_script md"""
### Optimal Policies and Value Functions

Every MDP has a unique optimal value function whose values are greater than or equal to every other value function at every state or state-action pair: $v_*(s) \geq v_\pi(s) \: \forall s, \pi$ and $q_*(s, a) \geq q_\pi(s, a) \: \forall s, a, \pi$.  This property can be used to derive a recursive relationship for both optimal value functions as shown below.
"""

# ╔═╡ 7df4fcbb-2f5f-4d59-ba0c-c7e635bb0503
@skip_as_script md"""
$\begin{flalign}
v_*(s) &\doteq \max_\pi v_\pi(s) \: \forall \: s \in \mathcal{S} \tag{3.15} \\
&= \max_{a \in \mathcal{A}(s)} q_{*}(s, a) \: \forall \: s \in \mathcal{S} \tag{meaning of optimal}\\
&= \max_{a \in \mathcal{A}(s)} \sum_{s^\prime, r} p(s^\prime, r \vert s, a) \left [ r + γ v_* (s^\prime) \right ] \quad \forall s \in \mathcal{S} \tag{by (3.21) (3.19)}\\
q_*(s, a) &\doteq \max_\pi q_\pi(s, a) \: \forall \: s \in \mathcal{S} \text{ and } a \in \mathcal{A}(s) \tag{3.16} \\
&=\mathbb{E} \left [ R_{t+1} + \gamma v_* (S_{t+1}) \mid S_t = s, A_t = a \right ] \tag{3.17} \\
&= \sum_{s^\prime, r} p(s^\prime, r \vert s, a) \left [ r + γ v_* (s^\prime) \right ] \tag{exp value def (3.21)} \\
&= \sum_{s^\prime, r} p(s^\prime, r \vert s, a) \left [ r + γ \max_{a^\prime} q_*(s^\prime, a^\prime) \right ] \tag{3.20} \\
\end{flalign}$
"""

# ╔═╡ 4f0f052d-b461-4040-b5ff-46aac74a24de
@skip_as_script md"""
Analogous to the previous Bellman equations, (3.19) and (3.20) are known as the *Bellman optimality equations* for the state and state-action value functions.  Every optimal policy will share the value function that has this property.  We can verify if a particular value function is optimal by checking whether it satisfies the Bellman optimality equation, but we also want methods to compute this function just like we did for a given policy.  In fact, our ability to compute the value function for a set policy can be used to derive the optimal value function.  This process is known as *policy improvement*.
"""

# ╔═╡ cf902114-94e3-4402-ae04-8f704dd6adad
@skip_as_script md"""
### Policy Improvement

Suppose we have a policy $\pi$ and the corresponding value functions $v_\pi, q_\pi$.  Recall that the optimal value functions $v_*, q_*$ have the property that their values are at least as good as the values for any other policy.  So, if we can find a modified policy whose value function is improved, we have moved our policy closer to the optimal one.  The approach in policy improvement will be to repeatedly improve a set policy until it is optimal.

As a starting point, consider a state $s$ and the corresponding value function at that state $v_\pi(s)$.  We can also consider $q_\pi(s, a)$ for all of the available actions.  Let's say we find an action $a$ such that $q_\pi(s, a) \geq v_\pi(s)$.  If we define a new policy $\pi^\prime$ which takes this action from state $s$ and otherwise follows $\pi$, then we know that $q_\pi(s, \pi^\prime(s)) \geq v_\pi(s)$.  This expression is using the value function for the original policy $\pi$ and assumes that our choice of action at state $s$ is a one time event.  If we encounter $s$ in the future, this expression is only correct if we revert to following $\pi$.  What we would like to know is whether $v_{\pi^\prime}(s) \geq v_\pi(s)$ for the state in question and every other state in the problem.  

The *policy improvement theorem* states that such a policy $\pi^\prime$ as we have defined it does in fact have that property.  In other words: 

$q_\pi(s, \pi^\prime(s)) \geq v_\pi(s) \implies v_{\pi^\prime}(s) \geq v_\pi(s) \: \forall \: s \in \mathcal{S}$

The $\pi^\prime$ defined above meets this property and uses $q_\pi$ to select a new action.  If we have access to the probability transition function, we can use $v_\pi$ to update the policy as follows:

$\begin{flalign}
\pi^\prime(s) &\doteq \mathrm{argmax}_a q_\pi(s, a) \\
& = \mathrm{argmax}_a \mathbb{E} [R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t = a] \\
& = \mathrm{argmax}_a \sum_{s^\prime, r} p(s^\prime, r \vert s, a) [r + \gamma v_\pi(s^\prime)] \\
\end{flalign}$

This policy is known as the *greedy* policy with respect to the value function.  We can apply this update at every state to improve the policy everywhere.  Let's say that $\pi^\prime = \pi$.  That would mean that the $v_{\pi^\prime} = v_\pi$, and $v_{\pi^\prime}(s) = \max_a q_{\pi^\prime}(s, a)$.  In other words, $\pi^\prime$ satisfies the Bellman optimaliy equation and we have found the optimal policy.
"""

# ╔═╡ a3e85772-9c67-454f-94d2-c2608b53c427
@skip_as_script md"""
### Policy Iteration

Since we can improve an arbitrary policy, one method to computing the optimal policy is to just repeat this process over an over until it converges.  Once the process converges, our policy is guaranteed to be optimal.  The procedure called *policy iteration* starts with an arbitrary policy $\pi_0$, computes its value function $v_{\pi_0}$, and then performs the greedy updateat every state to achieve an improved policy $\pi_1$.  Upon repetition this procedure will produce a sequence of policies and value functions until the update results in no change to the policy.  Since we are also computing the value functions at each step, we can also halt the process when the state values do not change at all or within some tolerance.
"""

# ╔═╡ f52b6f5d-3832-41aa-8ccd-78e514e65c8b
@skip_as_script md"""
### *Bellman Policy Iteration*
The following code implements policy iteration in the tabular case where the full probability transition function is available.  In this case, state values are sufficient, but one can also use state-action values with policy iteration.
"""

# ╔═╡ 1f9752c2-7bb9-4cd2-b90b-2995bcec7ae3
begin
"""
	make_greedy_policy!(π::Matrix{T}, mdp::FiniteDeterministicMDP{T, S, A}, V::Vector{T}, γ::T) where {T<:Real, S, A}

Modify π in-place to be a greedy policy for a given finite deterministic MDP environment, state value function, and discount factor. It iterates through each state and updates the policy matrix to prioritize actions leading to the highest expected future reward.

**Arguments:**

* `π::Matrix{T}`: A matrix representing the policy to be updated in-place. (rows represent actions, columns represent states). Elements `π[i_a, i_s]` specify the probability of taking action `i_a` in state `i_s`.
* `mdp::FiniteDeterministicMDP{T, S, A}`: The finite deterministic MDP environment for which the policy is being created.
* `V::Vector{T}`: A vector representing the state value function (V-values). Elements `V[i_s]` represent the value of state `i_s`.
* `γ::T`: The discount factor for future rewards (between 0 and 1).

**Returns:**

* `π::Matrix{T}`: The updated policy matrix where elements `π[i_a, i_s]` represent the probability of taking action `i_a` in state `i_s` based on the greedy policy.

**Notes:**

* This function assumes the provided `π` matrix has the same dimensions as the number of actions and states in the MDP environment.
"""
function make_greedy_policy!(π::Matrix{T}, mdp::FiniteDeterministicMDP{T, S, A}, V::Vector{T}, γ::T) where {T<:Real,S,A}
	for i_s in eachindex(mdp.states)
		maxv = -Inf
		for i_a in eachindex(mdp.actions)
			i_s′ = mdp.state_transition_map[i_a, i_s]
			r = mdp.reward_transition_map[i_a, i_s]
			x = r + γ*V[i_s′]
			maxv = max(maxv, x)
			π[i_a, i_s] = x
		end
		π[:, i_s] .= (π[:, i_s] .≈ maxv)
		π[:, i_s] ./= sum(π[:, i_s])
	end
	return π
end

function make_greedy_policy!(π::Matrix{T}, mdp::FiniteStochasticMDP{T, S, A}, V::Vector{T}, γ::T) where {T<:Real,S,A}
	for i_s in eachindex(mdp.states)
		maxv = -Inf
		for i_a in eachindex(mdp.actions)
			r_avg = zero(T)
			v_avg = zero(T)
			ptf = mdp.ptf[(i_s, i_a)]
			for i_s′ in keys(ptf)
				p = ptf[i_s′][1]
				v_avg += p*V[i_s′]
				r_avg += p*ptf[i_s′][2]
			end
			x = r_avg + γ*v_avg
			maxv = max(maxv, x)
			π[i_a, i_s] = x
		end
		π[:, i_s] .= (π[:, i_s] .≈ maxv)
		π[:, i_s] ./= sum(π[:, i_s])
	end
	return π
end

function make_greedy_policy!(π::Matrix{T}, mdp::AbstractTabularMDP{T, S, A}, Q::Matrix{T}, i_s::Integer) where {T<:Real,S,A}
	maxq = -Inf
	for i_a in eachindex(mdp.actions)
		maxq = max(maxq, Q[i_a, i_s])
	end
	π[:, i_s] .= (Q[:, i_s] .≈ maxq)
	π[:, i_s] ./= sum(π[:, i_s])
	return π
end

"""
	make_greedy_policy!(π::Matrix{T}, mdp::AbstractTabularMDP{T, S, A}, Q::Matrix{T}) where {T<:Real,S,A}

Modify `π` in-place to be a greedy policy for a given finite MDP environment (`mdp`) and state-action value function (`Q`). It iterates through each state and updates the policy matrix (`π`) to prioritize actions with the highest Q-value in that state. 

**Arguments:**

* `π::Matrix{T}`: A matrix representing the policy to be updated in-place. (rows represent actions, columns represent states). Elements `π[i_a, i_s]` specify the probability of taking action `i_a` in state `i_s`.
* `mdp::AbstractTabularMDP{T, S, A}`: The finite MDP environment (can be deterministic or stochastic) for which the policy is being created.
* `Q::Matrix{T}`: A matrix representing the state-action value function (Q-values). Elements `Q[i_a, i_s]` represent the Q-value for taking action `i_a` in state `i_s`.

**Returns:**

* `π::Matrix{T}`: The updated policy matrix where elements `π[i_a, i_s]` represent the probability of taking action `i_a` in state `i_s` based on the greedy policy.

**Notes:**

* This function assumes the provided `π` matrix has the same dimensions as the number of actions and states in the MDP environment.
* The function updates the policy matrix values in-place using the `≈` operator for element-wise comparison and the `./=` operator for element-wise division.

**Compared to `make_greedy_policy!` for deterministic MDPs:**

* This version works for both deterministic and stochastic MDP environments represented by `AbstractTabularMDP`.
* It uses the state-action value function (`Q`) directly instead of a separate state value function (`V`).

"""
function make_greedy_policy!(π::Matrix{T}, mdp::AbstractTabularMDP{T, S, A}, Q::Matrix{T}) where {T<:Real,S,A}
	for i_s in eachindex(mdp.states)
		make_greedy_policy!(π, mdp, Q, i_s)
	end
	return π
end
end

# ╔═╡ 4a80a7c3-6e9a-4973-b48a-b02509823830
@skip_as_script md"""
### *Example: Gridworld Optimal Policy Iteration*

If we apply policy iteration using the state value function, we can compute the optimal policy and value function for an arbitrary MDP.  This example applies the technique to a gridworld similar to the previous example but with a secondary goal in the upper left hand corner with half the reward.  The optimal solution changes depending on the discount rate since there are states for which the lower reward secondary goal is favorable due to the closer distance.  One can select the iteration to view both the policy and the corresponding value function as well as the discount rate and secondary goal reward to use for solving the MDP.
"""

# ╔═╡ 6467d0ee-d551-4558-a765-aa832373d125
@skip_as_script @bind policy_iteration_params PlutoUI.combine() do Child
	md"""
	Select reward for secondary goal: 
	
	$(Child(:goal2reward, Slider(-1f0:.01f0:1f0; show_value=true, default = 0.5f0)))
	
	Select Discount Rate for State Policy Iteration: 
	
	$(Child(:γ, Slider(0.01f0:0.01f0:1f0; show_value=true, default = 0.9)))

	Use Wind: $(Child(:usewind, CheckBox()))
	
	Use Stochastic Wind: $(Child(:stochastic, CheckBox()))

	Continuing Task: $(Child(:continuing, CheckBox()))
	"""
end |> confirm

# ╔═╡ 7cce54bb-eaf9-488a-a836-71e72ba66fcd
@skip_as_script const new_gridworld = begin
	policy_iteration_kwargs = (goal2 = GridworldState(1, 7), usegoal2=true, goal2reward = policy_iteration_params.goal2reward, wind = policy_iteration_params.usewind ? wind_values : zeros(Int64, 10), continuing = policy_iteration_params.continuing)
	if policy_iteration_params.stochastic
		make_stochastic_gridworld(;policy_iteration_kwargs...)
	else
		make_deterministic_gridworld(;policy_iteration_kwargs...)
	end
end

# ╔═╡ 6253a562-2a48-45da-b453-1ec7b51d2073
@skip_as_script md"""
### Value Iteration

When we introduced the Bellman optimality equations, it was noted that those equations can be used to verify if a policy is optimal.  It turns out that, just like with policy evaluation, we can use turn the Bellman optimality equations into an operator and use the operator directly to compute the optimal value function.  This procedure is called *value iteration* and proceeds by first initializing an arbitrary value function $v_0$.  Then that value function is updated with the Bellman optimality operator as follows:

$\begin{flalign}
v_{k+1}(s) = \max_a \sum_{s^\prime, r}p(s^\prime, r \vert s, a) \left [ r + \gamma v_k (s^\prime) \right ]
\end{flalign}$

This update can be performed at every state and repeated until the process converges.  It can be proven that starting with an arbitrary $v_0$, this procedure does converge to $v_*$ in the same manner that policy evaluation can compute $v_\pi$.  Here, the expected value under the policy is replaced with the maximization over actions.  This approach dispenses entirely with defining a policy as required by policy iteration and may converge faster than that process.  We can halt the process when the value function update becomes small within some tolerance.
"""

# ╔═╡ 0a7c9e73-81a7-45d9-bf9e-ebc61abeb552
@skip_as_script md"""
### *Bellman Value Iteration*
The following code implements value iteration in the tabular case where the value function can be represented as a vector of values for each state.  Given the probability transition function, state values are sufficient to perform value iteration, but it can also be done with state-action values.
"""

# ╔═╡ c2903e20-1be8-4d79-8716-798f5dc15bd4
function bellman_optimal_value!(V::Vector{T}, mdp::FiniteDeterministicMDP{T, S, A}, γ::T) where {T <: Real, S, A}
	delt = zero(T)
	@inbounds @fastmath @simd for i_s in eachindex(mdp.states)
		maxvalue = typemin(T)
		@inbounds @fastmath @simd for i_a in eachindex(mdp.actions)
			i_s′ = mdp.state_transition_map[i_a, i_s]
			r = mdp.reward_transition_map[i_a, i_s]
			x = r + γ*V[i_s′]
			maxvalue = max(maxvalue, x)
		end
		delt = max(delt, abs(maxvalue - V[i_s]) / (eps(abs(V[i_s])) + abs(V[i_s])))
		V[i_s] = maxvalue
	end
	return delt
end

# ╔═╡ 55d182d1-aa25-4ac9-802f-129756ffa302
function bellman_optimal_value!(V::Vector{T}, mdp::FiniteStochasticMDP{T, S, A}, γ::T) where {T <: Real, S, A}
	delt = zero(T)
	@inbounds @fastmath @simd for i_s in eachindex(mdp.states)
		maxvalue = typemin(T)
		@inbounds @fastmath @simd for i_a in eachindex(mdp.actions)
			r_avg = zero(T)
			v_avg = zero(T)
			ptf = mdp.ptf[(i_s, i_a)]
			for i_s′ in keys(ptf)
				p = ptf[i_s′][1]
				r = ptf[i_s′][2]
				v_avg += p*V[i_s′]
				r_avg += p*r
			end
			x = r_avg + γ*v_avg
			maxvalue = max(maxvalue, x)
		end
		delt = max(delt, abs(maxvalue - V[i_s]) / (eps(abs(V[i_s])) + abs(V[i_s])))
		V[i_s] = maxvalue
	end
	return delt
end

# ╔═╡ 40f6257d-db5c-4e21-9691-f3c9ffc9a9b5
@skip_as_script md"""
#### *Example: Gridworld Value Iteration*

If we apply value iteration using the state value function, we can compute the optimal value function for an arbitrary MDP.  The optimal policy will just be the greedy policy with respect to that value function.  The MDP shown is the same example as that used for the policy iteration example.  Even though value iteration requires more steps to converge, each step is much faster than those of policy iteration.
"""

# ╔═╡ bf12d9c9-c79d-4398-9f15-27cbde1ed476
@skip_as_script md"""Select discount rate for value iteration: $(@bind value_iteration_γ Slider(0.01f0:0.01f0:1f0; show_value=true, default = 0.9f0))"""

# ╔═╡ a6a3a31f-1411-4013-8bf7-fbdceac9c6ba
@skip_as_script md"""
### Generalized Policy Iteration

So far we presented two extreme cases of generalized policy iteration.  In the first case, policy iteration, we accurately compute a policy value function, and then update the policy to be greedy with respect to it.  In value iteration, we skip defining a policy altogether and just use the Bellman optimality operator to iteratively compute the optimal value function.  In general, we can use the Bellman operator to compute a value function for a policy that is not yet optimal and stop before that value function has converged.  Then our policy improvement step is not basing the new policy on an accurate version of the current value function, but we can continue to apply policy evaluation to the updated policy.  In this procedure, the policy evaluation is constantly playing catchup to the ever changing policy by chasing a moving target, but that target will stop moving once we reach the optimal policy.  It turns out that proceding with partial value function updates will still eventually converge to the optimal policy, and we can choose to wait until the value function is fully converged, dispense with it altogether, or anything in between.  This family of procedures all follow the same pattern and are known as *generalized policy iteration*.
"""

# ╔═╡ 1d555f77-c404-485a-9244-717c12c80d28
@skip_as_script md"""
## Monte Carlo Sampling Methods
The preceeding solution methods require the probability transition function to calculate value functions by using the Bellman equations.  It is also possible to compute value functions from *experience* with the environment.  Typically this experience is in the form of observed transitions in the environment: $(s, a) \rightarrow (s^\prime, r)$.  For a deterministic environment, only one state transition is possible, so even after one observation we may already have information equivalent to the probability transition function.  In general stochastic environments, we can only learn accurate value functions by observing many transitions from a single state action pair (usually an infinite number to guarantee convergence).  Our approach to computing the optimal value function will follow the same pattern of generalized policy iteration where we use the value function as a stepping stone for policy improvement.
"""

# ╔═╡ 3df86061-63f7-4c1f-a141-e1848f6e83e4
@skip_as_script md"""
### Policy Prediction

Experience can be used to do policy evaluation.  When we use experience instead of the probability transition function, this procedure is known as *Monte Carlo Prediction* and the environment will be used to *sample* experience that follows the probability transition function.  This method is the easiest to understand because it only relies upon the original definition of the value functions.  

$\begin{flalign}
v_\pi(s) &= \mathbb{E}_\pi \left [G_t \mid S_t = s \right] = \mathbb{E}_\pi \left [R_{t+1} + \gamma R_{t+2} + \cdots \mid S_t = s \right] \\
q_\pi(s, a) &= \mathbb{E}_\pi \left [G_t \mid S_t = s, A_t = a \right] = \mathbb{E}_\pi \left [R_{t+1} + \gamma R_{t+2} + \cdots \mid S_t = s, A_t = a \right]\\
\end{flalign}$

Instead of expanding the definition of $G_t$, we will directly sample it from episodes through the environment.  As such this method is only suitable for environments that are episodic and for policies that produce finite episodes.  Given such a policy, we can select a starting state either randomly or given naturally by the environment and then use the policy to generate transitions through the environment until termination.  Such an episode will look like:

$S_0 \overset{\pi}{\rightarrow} A_0 \rightarrow R_1, S_1 \overset{\pi}{\rightarrow} A_1 \rightarrow R_2, S_2 \overset{\pi}{\rightarrow} A_2 \rightarrow \cdots\rightarrow R_T, S_T$

From this episode, at each state $s = S_t$, we can estimate $G_t = \mathbb{E}_\pi \left [ R_t + R_{t+1} + \cdots + R_T \right ]$ by taking a single sample who's expected value matches the expected value in the definition of $G_t$.  A weighted average of these samples will produce an estimate of $G_t$ who's variance will shrink to 0 in the limit of infinite samples (this depends on the averaging method as some methods may not have variance that converges to 0 and also on the environment in the case of the reward distribution for a particular state having infinite variance).  If we instead wish to estimate state-action values, we can perform the same averaging but maintain a different estimate for each state action pair observed.    
"""

# ╔═╡ 8abba353-2309-4931-bf3f-6b1f500998a7
@skip_as_script md"""
### *Sampling MDP Definitions and Functions*

When the probability transition function is unavailable, we can use an MDP that only provides sample transitions given a state action pair.  Below is code implementing such a ```SampleTabularMDP{T<:Real, S, A, F, G, H}``` where we can fully enumerate all the states and actions.  In addition to a list of states and actions, such an MDP must also have three functions: 

```step(i_s::Integer, i_a::Integer)``` returns a tuple of $(r, i_s^\prime)$ where $r$ is of type ```T```

```state_init()``` produces an initial state index to start an episode

```isterm(i_s::Integer)``` returns a Boolean indicating whether an episode is a terminal state

Once these functions are defined, one can construct the mdp with ```SampleTabularMDP(states, actions, step, state_init, isterm)```.  Alternatively, one can use an existing ```FiniteDeterministicMDP``` or ```FiniteStochasticMDP``` to construct one by providing it and a ```state_init``` function: ```SampleTabularMDP(mdp::FiniteDeterministicMDP, state_init::Function)```
"""

# ╔═╡ 860650f0-c6bb-43d6-9ece-c6e6f39e010d
begin
	abstract type AbstractSampleTabularMDP{T<:Real, S, A, F, G, H} <: AbstractTabularMDP{T, S, A} end
	struct SampleTabularMDP{T<:Real, S, A, F, G, H} <: AbstractSampleTabularMDP{T, S, A, F, G, H}
		states::Vector{S}
		actions::Vector{A}
		state_index::Dict{S, Int64}
		action_index::Dict{A, Int64}
		step::F #step(i_s::Integer, i_a::Integer) must return a tuple (r::T, i_s′::Integer)
		state_init::G #state_init() must return an initial state index i_s_0::Integer
		isterm::H #isterm(i_s::Integer) must return a boolean inicating whether state s = states[i_s] is terminal
		function SampleTabularMDP(states::Vector{S}, actions::Vector{A}, step::F, state_init::G, isterm::H) where {S, A, F<:Function, G<:Function, H<:Function}
			i_s = state_init()
			!(typeof(i_s) <: Integer) && error("state init function is not returning a state index")
			transition = step(state_init(), 1)
			!(typeof(transition) <: Tuple) && error("step function is not returning a tuple of (r, s)")
			(r, i_s′) = transition
			T = typeof(r)
			!(typeof(i_s′) <: Integer)  && error("state transition is not an index")
			!(T <: Real) && error("Reward is not a real number")
			!isterm(i_s) #check to see if isterm function takes a state and returns a boolean
			new{T, S, A, F, G, H}(states, actions, makelookup(states), makelookup(actions), step, state_init, isterm)
		end
	end

	#once we have an AbstractCompleteMDP as defined above, we can always convert it into an AbstractSampleTabularMDP as long as we have a state_init function defined.  everything else can be derived from the TabularMDP
	function SampleTabularMDP(mdp::FiniteDeterministicMDP{T, S, A}, state_init::Function) where {T<:Real, S, A}
		step(i_s, i_a) = (mdp.reward_transition_map[i_a, i_s], mdp.state_transition_map[i_a, i_s])
		isterm = make_isterm(mdp)
		SampleTabularMDP(mdp.states, mdp.actions, step, state_init, isterm)
	end
	function SampleTabularMDP(mdp::FiniteStochasticMDP{T, S, A}, state_init::Function) where {T<:Real, S, A}
		transition_lookup = Dict(begin
			i_s = mdp.state_index[s]
			i_a = mdp.action_index[a]
			ptf = mdp.ptf[(i_s, i_a)]
			probabilities = [ptf[i_s′][1] for i_s′ in keys(ptf)]
			rewards = [ptf[i_s′][2] for i_s′ in keys(ptf)]
			(i_s, i_a) => (collect(keys(ptf)), weights(probabilities), rewards)
		end
		for s in mdp.states for a in mdp.actions)
		function step(i_s, i_a)
			states, weights, rewards = transition_lookup[(i_s, i_a)]
			if length(states) == 1
				i_s′ = first(states)
				r = first(rewards)
			else
				i = sample(eachindex(states), weights)
				i_s′ = states[i]
				r = rewards[i]
			end
			return (r, i_s′)
		end
		isterm = make_isterm(mdp)
		SampleTabularMDP(mdp.states, mdp.actions, step, state_init, isterm)
	end
end

# ╔═╡ ce8a7ed9-7719-4caa-a680-76fac3dea985
#construct a sample gridworld from the previously instantiated one
@skip_as_script const deterministic_sample_gridworld = SampleTabularMDP(deterministic_gridworld.mdp, () ->deterministic_gridworld.init_state)

# ╔═╡ c773d4a4-aade-4d14-9dbe-96b146450e3f
#construct a sample gridworld from the previously instantiated one
@skip_as_script const windy_sample_gridworld = SampleTabularMDP(windy_gridworld.mdp, () ->windy_gridworld.init_state)

# ╔═╡ 99667755-430c-4687-a6b2-b6c4e91f0e26
@skip_as_script const stochastic_sample_gridworld = SampleTabularMDP(stochastic_gridworld.mdp, () ->stochastic_gridworld.init_state)

# ╔═╡ 71d18d73-0bcb-48ee-91fd-8fa2f52a908c
function takestep(mdp::SampleTabularMDP{T, S, A, F, G, H}, π::Matrix{T}, i_s::Integer) where {T<:Real, S, A, F<:Function, G<:Function, H<:Function}
	i_a = sample_action(π, i_s)
	(r, i_s′) = mdp.step(i_s, i_a)
	return (r, i_s′, i_a)
end

# ╔═╡ 7c553f77-7783-439e-834b-53a2cd3bef5a
@skip_as_script md"""
### *Monte Carlo Policy Prediction*
"""

# ╔═╡ a502c80a-fe11-4184-9731-c634655a825d
begin
	abstract type AbstractAveragingMethod{T<:Real} end
	#to keep track of a sample average, a weight or count is required
	struct SampleAveraging{T<:Real, N} <: AbstractAveragingMethod{T}
		weights::Array{T, N}
		function SampleAveraging(values::Array{T, N}) where {T<:Real, N}
			weights = similar(values)
			weights .= 0
			new{T, N}(weights)
		end
	end
	#for constant step size averaging, no additional statistics are necessary
	struct ConstantStepAveraging{T<:Real} <: AbstractAveragingMethod{T}
		α::T
	end
	#note that to add additional averaging methods, one must also define a method of update_average! whose final argument matches the new averaging type.  see the examples below for those methods defined for SampleAveraging and ConstantStepAveraging
end

# ╔═╡ ad34ce87-d9cc-407b-9670-25ed535d2d8d
function update_average!(v_est::Vector{T}, target_value::T, step::Integer, states::Vector{Int64}, actions::Vector{Int64}, avg_method::SampleAveraging) where T<:Real
	i_s = states[step]
	avg_method.weights[i_s] += 1
	δ = target_value - v_est[i_s]
	v_est[i_s] += δ / avg_method.weights[i_s]
end

# ╔═╡ 67a0aac8-b022-4051-804c-cfda3e0c7357
function update_average!(v_est::Vector{T}, target_value::T, step::Integer, states::Vector{Int64}, actions::Vector{Int64}, avg_method::ConstantStepAveraging) where T<:Real
	i_s = states[step]
	δ = target_value - v_est[i_s]
	v_est[i_s] += avg_method.α * δ
end

# ╔═╡ 6dfbf4c3-8d66-45d3-ae1c-ad50a53eb570
function update_average!(q_est::Matrix{T}, target_value::T, step::Integer, states::Vector{Int64}, actions::Vector{Int64}, avg_method::SampleAveraging) where T<:Real
	i_s = states[step]
	i_a = actions[step]
	avg_method.weights[i_a, i_s] += 1
	δ = target_value - q_est[i_a, i_s]
	q_est[i_a, i_s] += δ / avg_method.weights[i_a, i_s]
end

# ╔═╡ 7136cb2a-6957-4d21-a6e2-381e571a113a
function update_average!(q_est::Matrix{T}, target_value::T, step::Integer, states::Vector{Int64}, actions::Vector{Int64}, avg_method::ConstantStepAveraging) where T<:Real
	i_s = states[step]
	i_a = actions[step]
	δ = target_value - q_est[i_a, i_s]
	q_est[i_a, i_s] += avg_method.α * δ
end

# ╔═╡ 3d86b788-9770-4356-ac6b-e80b0bfa1314
function monte_carlo_episode_update!(value_estimates::Array{T, N}, states::Vector{Int64}, actions::Vector{Int64}, rewards::Vector{T}, mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, γ::T, averaging_method::AbstractAveragingMethod; kwargs...) where {T<:Real, S, A, F<:Function, G<:Function, H<:Function, N}
	l = length(states)
	g = zero(T)
	for i in l:-1:1
		g = γ*g + rewards[i]
		update_average!(value_estimates, g, i, states, actions, averaging_method)
	end
end

# ╔═╡ 52e73547-ce0d-4696-8a3c-46ced9fa6582
begin
	update_value_history!(v_history, v::Vector, ep::Integer) = v_history[:, ep] .= v
	update_value_history!(v_history, q::Matrix, ep::Integer) = v_history[:, :, ep] .= q
end

# ╔═╡ a2027cca-4a12-4d7d-a721-6044c6255394
@skip_as_script @bind γ_mc_predict NumberField(0f0:0.01f0:1f0; default = 0.99f0)

# ╔═╡ 1b83b6c2-43cb-4ad4-b5a9-46e31d585a27
@skip_as_script md"""
### Monte Carlo Control

Recalling generalized policy iteration, we can use the episode as the point at which we update the policy with respect to whatever the value estimates are at that time.  Since we cannot apply Monte Carlo prediction before an episode is completed, this is the fastest we could possible update the policy.  We could always update our prediction of the value function over more episodes to make it more accurate, but we plan on updating the policy anyway so there is not need to have converged values until we have reached the optimal policy.  In order to guarantee convergence, however, we must visit have a non zero probability of visiting every state action pair an infinite number of times in the limit of conducting infinite episodes.  There are two main methods of achieving this property.  The first is to begin episodes with random state-action pairs sampled such that each pair has a non-zero probability of being selected.  The second method is to update the policy to be $\epsilon$-greedy with respect to the value function.  $\epsilon$-greedy policies have a non-zero probability $\epsilon$ of taking random actions and behave as the greedy policy otherwise.  Because of the random chance, such a policy is also guaranteed to visit all the state action pairs, but then our policy improvement is restricted to the case of the best $\epsilon$-greedy policy.  We could lower $\epsilon$ to zero during the learning process to converge to the optimal policy.

After applying MC state-action value prediction for a single episode, we have ${q_\pi}_k$ where $k$ is the current episode count.  To apply policy improvement just update $\pi_k(s) = \mathrm{argmax}_a {q_\pi}_k(s, a)$.  We estimate state-action values instead of state values because it makes the policy improvement step trivial.  The previous method required the probability transition function to compute $q(s, a)$ from $v(s)$.  Using state-action values instead frees us from needing the probability transition function at the cost of needing to store more estimates.
"""

# ╔═╡ 51fecb7e-65ff-4a11-b043-b5832fed5e02
@skip_as_script md"""
### *Monte Carlo Control with Exploring Starts*

The following code implements Monte Carlo control for estimating the optimal policy of a Tabular MDP from which we can only take samples.  If we update the target policy to be greedy with respect to the value function, then exploring starts are required to ensure that we could visit all the state action pairs an unlimited number of times over the course of multiple episodes.  The exploring starts method is defined by initializing each episode with a random state action pair and performing the greedy policy update after each episode.
"""

# ╔═╡ 105b8874-5cbc-4777-87c6-e8712cbcc78d
@skip_as_script md"""
#### *Example: Monte Carlo control with exploring starts on gridworld*
"""

# ╔═╡ 26d60dab-bab1-495d-a236-44f075c912bd
@skip_as_script @bind mc_control_γ NumberField(0.01f0:0.01f0:1f0; default = 0.88f0)

# ╔═╡ 4efee19f-c86c-44cc-8b4b-6eb45adf0aa1
@skip_as_script md"""
### *Monte Carlo Control for $\epsilon$-soft policies*

The following code implements Monte Carlo control without exploring starts.  In order to guarantee visits to all state-action pairs, we must force the policy to take random actions some percentage of the time.  Any policy that has non-zero probability for every state-action pair is called a *soft* policy.  For this algorithm we will select a value $\epsilon$ which controls the probability with which random actions are taken.  Such a policy is *soft* and thus this family of policies are called $\epsilon$-soft policies.  Once we set $\epsilon$, the behavior for the remaining probability can be arbitrary.  If we evenly divide it, then that would be the uniformly random policy which is also $\epsilon$-soft for any value of $\epsilon$.  If, instead, we select the greedy action for that probability, then such a policy is called $\epsilon$-greedy in addition to being an $\epsilon$-soft policy.  For any finite $\epsilon$, the learned policy will not necessarily be optimal since it is restricted to sometimes taking random actions, but as $\epsilon$ approaches zero, the learned policy can become arbitrarily close to the optimal policy.  Also, we are free to update the policy to be greedy with respect to the value function when we have completed the learning process.
"""

# ╔═╡ 1c829fde-e15d-42db-a608-2e5bdbaa4d8c
function make_ϵ_greedy_policy!(π::Matrix{T}, mdp::AbstractTabularMDP{T, S, A}, Q::Matrix{T}, ϵ::T) where {T<:Real,S,A}
	n = length(mdp.actions)
	for i_s in eachindex(mdp.states)
		maxq = -Inf
		for i_a in eachindex(mdp.actions)
			maxq = max(maxq, Q[i_a, i_s])
		end
		π[:, i_s] .= (Q[:, i_s] .≈ maxq)
		π[:, i_s] ./= (sum(π[:, i_s]) / (one(T) - ϵ))
		π[:, i_s] .+= ϵ / n
	end
	return π
end

# ╔═╡ f77cf1bb-6385-403d-a224-3c9c7313e591
function make_ϵ_greedy_policy!(π::Matrix{T}, mdp::AbstractTabularMDP{T, S, A}, Q::Matrix{T}, ϵ::T, i_s::Integer) where {T<:Real,S,A}
	n = length(mdp.actions)
	maxq = -Inf
	for i_a in eachindex(mdp.actions)
		maxq = max(maxq, Q[i_a, i_s])
	end
	π[:, i_s] .= (Q[:, i_s] .≈ maxq)
	π[:, i_s] ./= (sum(π[:, i_s]) / (one(T) - ϵ))
	π[:, i_s] .+= ϵ / n
	return π
end

# ╔═╡ 4bd400f3-4cb4-47a2-b0f5-31e6dedc253d
@skip_as_script md"""
#### *Example: Monte Carlo control with $\epsilon$ greedy policy*
"""

# ╔═╡ d7037f99-d3b8-4986-95c8-58f4f043e916
@skip_as_script md"""
### Off-policy Prediction via Importance Sampling

To learn the optimal policy through sampling experience, it is important to visit all state-action pairs.  Otherwise, we cannot compute all of the estimated values needed to update the value function.  So far, we have considered methods that sample the environment using a single policy who's behavior is updated to converge towards the optimal policy.  The optimal policy in general will not visit all the state action pairs, so it is possible that learned policies which are converging to the optimal policy will not visit all of the state-action pairs and therefore prevent us from collecting the experience necessary to continue generalized policy iteration.  We have considered two methods to avoid this problem: 1) exploring starts and 2) $\epsilon$-greedy action selection.  Now we consider a new type of solution that relies on *off-policy* learning in which the policy generating episodes in an environment is not the policy being optimized.

Such *off-policy* learning methods define a *target* policy and a *behavior* policy.  The target policy is the policy for which we are computing the value function and possibly updating though policy improvement.  The behavior policy is our source for episode samples.  The returns generated by the behavior policy will not have expected values that match the target policy value function, so the sampled values must be modified.  Recall that we are interested in calculating $\mathbb{E}_{\pi_{target}} [G_t \mid S_t = s, A_t = a]$ but we only have access to samples generated by $\pi_{behavior}$.  Our approach to estimating the expected value is just to average the returns observed for each state-action pair.  For off-policy prediction to work, we must compute a weighted sum of the sample returns that corrects for the difference in which trajectories you would observe for the target policy vs the behavior policy.  When such correction weights are applied to the sample average, that is called *importance sampling*.  The weight for each sample should be $\rho_{t:T-1} = \prod_{k=t}^{T-1}\frac{\pi_{target}(A_k \vert S_k)}{\pi_{behavior}(A_k \vert S_k)}$ which is equal to the probability of the trajectory beyond the current state for the target policy divided by that same probability under the behavior policy.  In other words, if a given trajectory would never be seen by the target policy, then do not include that term in the average.  If a trajectory is observed that is 10 times as likely under the target policy than the behavior policy, then weight it 10 times higher then trajectories that are equally expected under both policies.  

Once we compute the weighted sum of returns, the value estimate can be computed by dividing this sum by either the number of terms or the sum of weights.  The latter is called *weighted importance sampling* and both methods converge to the correct expected value in the limit of infinite samples.  The difference between the methods is that weighted importance sampling always has finite variance for the estimate as long as the returns themselves have finite variance.  Normal importance sampling can have infinite variance as long as the terms in the sum have infinite variance which is often the case with behavior policies that can generate long trajectories.  For weighted importance sampling, there is a bias towards the behavior policy, but that bias converges to zero with more samplies so it isn't usually a concern.  Therefore the more stable convergence properties of weighted importance sampling make it more favorable for Mpnte Carlo prediction and control.
"""

# ╔═╡ 39a1fc54-4024-4d89-9eeb-1fab0477e684
@skip_as_script md"""
### *Monte Carlo Off-policy Prediction*

Below is code implementing Monte Carlo prediction via importance sampling with the option of using ordinary or weighted importance sampling.  The MDPs are the same sampling types defined earlier and the weighted method is used by default.  Unlike on-policy Monte Carlo prediction, these algorithms require a behavior policy to be defined which is distinct from the target policy.  By default the random policy is used, but any other soft policy is suitable.  An error check will prevent prediction if the behavior policy is not soft.
"""

# ╔═╡ 46c11a87-10aa-46e2-8961-7acd33059b96
begin
	abstract type AbstractSamplingMethod end
	struct OrdinaryImportanceSampling <: AbstractSamplingMethod end
	struct WeightedImportanceSampling <: AbstractSamplingMethod end
end

# ╔═╡ 55fbc75b-44d2-49e4-830f-fdb88eadafdb
update_weight(ρ::T, ::OrdinaryImportanceSampling) where T<:Real = one(T)

# ╔═╡ f316a6f8-b462-4cec-b2ff-434330be579a
update_weight(ρ, ::WeightedImportanceSampling) = ρ

# ╔═╡ a1b90125-d3dd-409c-8231-ab0c3a85153e
function monte_carlo_episode_update!((q, weights)::Tuple{Matrix{T}, Matrix{T}}, states::Vector{Int64}, actions::Vector{Int64}, rewards::Vector{T}, π_target::Matrix{T}, π_behavior::Matrix{T}, sampling_method::AbstractSamplingMethod, mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, γ::T; kwargs...) where {T<:Real, S, A, F<:Function, G<:Function, H<:Function}
	l = length(states)
	g = zero(T)
	ρ = one(T)
	for i in l:-1:1
		i_s = states[i]
		i_a = actions[i]
		if iszero(π_target[i_a, i_s]) && isa(sampling_method, WeightedImportanceSampling)
			#in this case no further updates will occur
			break
		end
		p = π_target[i_a, i_s] / π_behavior[i_a, i_s]
		ρ *= p
		g = γ*g + rewards[i]
		weights[i_a, i_s] += update_weight(ρ, sampling_method)
		if !iszero(weights[i_a, i_s]) 
			q_old = q[i_a, i_s]
			δ = g - q_old
			q[i_a, i_s] += ρ * δ / weights[i_a, i_s] 
		end
	end
end

# ╔═╡ 900523ce-f8e7-4f33-a294-de86a7fb8869
@skip_as_script md"""
#### *Example: Off-policy prediction with Right gridworld policy*
"""

# ╔═╡ f3df4648-2884-4b01-823d-7e8ee2adc35b
@skip_as_script const π_target_gridworld =  mapreduce(_ -> [0f0, 0f0, 0f0, 1f0], hcat, 1:length(deterministic_sample_gridworld.states))

# ╔═╡ 73aece7b-314d-4f5f-bf7f-89852156e89e
@skip_as_script md"""
Select x value to view state estimate:  $(@bind x_off_policy_select Slider(1:7; default = 7, show_value=true))
"""

# ╔═╡ eebfe8e7-56dd-457c-a1e6-1a67b3b7ceec
@skip_as_script md"""
### Monte Carlo Off-policy Control
"""

# ╔═╡ 54cd4729-e4d3-4783-af1d-17df32ca6d69
@skip_as_script md"""
### *Monte Carlo Off-policy Control*
"""

# ╔═╡ 5979b5ec-5fef-40ef-a5c3-3a5b3d3040d9
@skip_as_script md"""
## Temporal Difference Learning

Both Monte Carlo and Temporal Difference methods use sampling from experience to learn value estimates and optimal policies.  With Monte Carlo methods we returned to the definition of the value function in terms of the expected value of the discounted future return:

$\begin{flalign}
v_\pi(s) &= \mathbb{E}_\pi \left [G_t \mid S_t = s \right] = \mathbb{E}_\pi \left [R_t + \gamma R_{t+1} + \cdots \mid S_t = s \right] \\
q_\pi(s, a) &= \mathbb{E}_\pi \left [G_t \mid S_t = s, A_t = a \right] = \mathbb{E}_\pi \left [R_t + \gamma R_{t+1} + \cdots \mid S_t = s, A_t = a \right]\\
\end{flalign}$

Using this form of the expression, we could sample an entire trajectory to a terminal state under a policy and then calculate a single unbiased sample of the value estimate.  Those samples can then be averaged in some way to compute the estimate.  For Temporal Difference Learning, we will instead use the Bellman Equations as inspiration for computing the value estimates from samples.  In particular recall that:

$\begin{flalign}
v_\pi(s) &= \mathbb{E}_\pi \left [G_t \mid S_t = s \right] \\
&= \sum_a \pi(a \vert s) \sum_{s^\prime, r} p(s^\prime, r \vert s, a) [r + \gamma v_\pi(s^\prime)] \\
&= \mathbb{E}_\pi [R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s] \\
q_\pi(s, a) &= \mathbb{E}_\pi \left [G_t \mid S_t = s, A_t = a \right] \\
&= \sum_{s^\prime, r} p(s^\prime, r \vert s, a) [r + \gamma\sum_{a^\prime} \pi(a^\prime \vert s^\prime) q_\pi(s^\prime, a^\prime)] \\
&= \mathbb{E}_\pi[R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a]\\
\end{flalign}$

Since both of these expressions are expected values under the policy, we can again simply take samples from a trajectory collected under the policy $\pi$ and average those samples to compute the value estimates.
"""

# ╔═╡ d250a257-4dc6-4369-90f0-fe186b3d9e7b
@skip_as_script md"""
### TD(0) Policy Prediction

Unlike Monte Carlo methods, TD(0) using the Bellman style update does not need an entire trajectory to a terminal state in order to perform a value update.  For the state value function, we only need to sample the reward and the next state.  For the state-action value function, we also need the action taken from the transition state.  Below is an example of the portion of the trajectory needed to perform the update.  For state value prediction we do not immediately need $A_{t+1}$ but if we evaluate it as part of the step we can use it on the next step.

$S_t \overset{\pi}{\rightarrow} A_t \rightarrow R_{t+1}, S_{t+1} \overset{\pi}{\rightarrow} A_{t+1}$

The sequence shown of state, action, reward, state, action is where the name *Sarsa* comes from since these are the necessary components for updating state-action value function $q_\pi(s, a)$
"""

# ╔═╡ b7506e65-60eb-4985-9a28-5a29cb400670
@skip_as_script md"""
### *Tabular TD(0) for Estimating Value Function* 

Typically for TD methods, we update the value estimates with constant step size averaging instead of sample averaging.  This requires selecting a step size $\alpha$ for the algorithm.  If $\alpha = \frac{1}{N}$ where $N$ is the number of observed samples, then this is equivalent to sample averaging.  Using a constant step size has the advantage that it is suitable for non-stationary problems.
"""

# ╔═╡ a858aeaa-29f5-4615-805c-0c6093cf9b5f
#note that for td learning with the state_action value function, it is necessary to perform a step and sample the action at the transition state.  The required information from a step is state, action, reward, new state, new action which is summarized by the acronym sarsa.  even though this term is reserved for the control case, the information from the transition is the same as that used for td0 q policy prediction
function sarsa_step(mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, π::Matrix{T}, i_s::Integer, i_a::Integer) where {T<:Real, S, A, F<:Function, G<:Function, H<:Function}
	(r, i_s′) = mdp.step(i_s, i_a)
	i_a′ = sample_action(π, i_s′)
	(r, i_s′, i_a′)
end

# ╔═╡ 0d8da60d-5e21-4398-a731-ed87754b63c8
function episode_update!(q::Matrix{T}, mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, π::Matrix{T}, γ::T, α::T) where {T<:Real, S, A, F<:Function, G<:Function, H<:Function}
	i_s = mdp.state_init()
	i_a = sample_action(π, i_s)
	while !mdp.isterm(i_s)
		(r, i_s′, i_a′) = sarsa_step(mdp, π, i_s, i_a)
		q[i_a, i_s] += α * (r + γ*q[i_a′, i_s′] - q[i_a, i_s])
		i_s = i_s′
		i_a = i_a′
	end
end

# ╔═╡ 9fb8f6ea-ca20-461c-b790-f651b13721b2
@skip_as_script md"""
### Sarsa: On-policy TD Control

Just as TD policy prediction uses the Bellman equations as an update target, Sarsa uses the Bellman optimality equations as the update target and performs something closer to value iteration where the value function is updated every step.
"""

# ╔═╡ c3c3bb5c-4bcf-442e-9718-c18a4a03fa82
@skip_as_script md"""
### *Sarsa for estimating $Q \approx q_{\star}$*
"""

# ╔═╡ 5aacf874-1519-4665-9207-f687b6e9944b
#try to make this not depend on episode but restart the episode if a terminal state is reached.  then it can stop iteration either based on total steps or total episodes.  need to decide for saving history though if it should be episode based or step based

# ╔═╡ cc09fc0b-bf88-464a-980b-59ae86bbd5d8
function sarsa_episode_update!((q, π)::Tuple{Matrix{T}, Matrix{T}}, update_policy!::Function, mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, γ::T, α::T) where {T<:Real,S, A, F<:Function, G<:Function, H<:Function}
	i_s = mdp.state_init()
	i_a = sample_action(π, i_s)
	total_reward = zero(T)
	total_steps = 0
	while !mdp.isterm(i_s)
		(r, i_s′, i_a′) = sarsa_step(mdp, π, i_s, i_a)
		q[i_a, i_s] += α * (r + γ*q[i_a′, i_s′] - q[i_a, i_s])
		i_s = i_s′
		i_a = i_a′
		update_policy!(π, q, i_s)
		total_reward += r
		total_steps += 1
	end
	return total_reward, total_steps
end

# ╔═╡ 41361309-8be9-464a-987e-981035e4b15a
@skip_as_script md"""
### Q-learning: Off-policy TD Control
"""

# ╔═╡ ee8a054e-64db-4c61-a5d3-b38e692887d9
md"""
### *Expected Sarsa for estimating $$Q \approx q_{\star}$$*

Q-learning is implemented as a version of expected sarsa where the target policy is updated with to be greedy while the behavior policy is updated to be $\epsilon$-greedy
"""

# ╔═╡ 7146eebf-bd16-424f-ae55-de51689bc0fe
function expected_sarsa_episode_update!((q, π_target, π_behavior)::Tuple{Matrix{T}, Matrix{T}, Matrix{T}}, update_target_policy!::Function, update_behavior_policy!::Function, mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, γ::T, α::T) where {T<:Real,S, A, F<:Function, G<:Function, H<:Function}
	i_s = mdp.state_init()
	i_a = sample_action(π_behavior, i_s)
	total_reward = zero(T)
	total_steps = 0
	while !mdp.isterm(i_s)
		(r, i_s′, i_a′) = sarsa_step(mdp, π_behavior, i_s, i_a)
		v′ = sum(π_target[i_a′, i_s′]*q[i_a′, i_s′] for i_a′ in eachindex(mdp.actions))
		q[i_a, i_s] += α * (r + γ*v′ - q[i_a, i_s])
		i_s = i_s′
		i_a = i_a′
		update_behavior_policy!(π_behavior, mdp, q, i_s)
		update_target_policy!(π_target, mdp, q, i_s)
		total_reward += r
		total_steps += 1
	end
	return total_reward, total_steps
end

# ╔═╡ 2bab0784-b185-44f0-9dec-c98bf164827b
@skip_as_script md"""
### Double Learning TD Methods
"""

# ╔═╡ be74f8fb-fd58-4170-8735-1af55a04d48f
md"""
### *Double Expected Sarsa for estimating $$Q \approx q_{\star}$$ and $\pi \approx \pi_{\star}$*
"""

# ╔═╡ 2688c9b8-07c0-4105-b2af-a7c71c48fb31
function make_ϵ_greedy_policy!(π::Matrix{T}, mdp::AbstractTabularMDP{T, S, A}, Q1::Matrix{T}, Q2::Matrix{T}, ϵ::T, i_s::Integer) where {T<:Real,S,A}
	n = length(mdp.actions)
	maxq = -Inf
	π[:, i_s] .= (Q1[:, i_s] .+ Q2[:, i_s]) ./ 2
	for i_a in eachindex(mdp.actions)
		maxq = max(maxq, π[i_a, i_s])
	end
	π[:, i_s] .= (π[:, i_s] .≈ maxq)
	π[:, i_s] ./= (sum(π[:, i_s]) / (one(T) - ϵ))
	π[:, i_s] .+= ϵ / n
	return π
end

# ╔═╡ afdd018f-c923-4906-9b70-c7b0a3e16831
function double_expected_sarsa_episode_update!((q1, q2, π_target1, π_target2, π_behavior)::Tuple{Matrix{T}, Matrix{T}, Matrix{T}, Matrix{T}, Matrix{T}}, update_target_policy!::Function, update_behavior_policy!::Function, mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, γ::T, α::T) where {T<:Real,S, A, F<:Function, G<:Function, H<:Function}
	i_s = mdp.state_init()
	i_a = sample_action(π_behavior, i_s)
	total_reward = zero(T)
	total_steps = 0
	while !mdp.isterm(i_s)
		(r, i_s′, i_a′) = sarsa_step(mdp, π_behavior, i_s, i_a)
		(π_target, q′, q) = if rand() < 0.5
			(π_target1, q2, q1)
		else
			(π_target2, q1, q2)
		end
		v′ = sum(π_target1[i_a′, i_s′]*q′[i_a′, i_s′] for i_a′ in eachindex(mdp.actions))
		q[i_a, i_s] += α * (r + γ*v′ - q[i_a, i_s])
		i_s = i_s′
		i_a = i_a′
		update_behavior_policy!(π_behavior, mdp, q1, q2, i_s)
		update_target_policy!(π_target, mdp, q, i_s)
		total_reward += r
		total_steps += 1
	end
	return total_reward, total_steps
end

# ╔═╡ 3dc94c4a-1072-4e9d-8408-439ea20a6029
md"""
## Afterstates

In the tic-tac-toe example we considered learning a value function for a state after the player's move but before the opponent's response.  This type of state is called an *afterstate*, and it is useful in situations when we know a portion of the dynamics in an environment, but then a portion of it is stochastic or unknown.  For example, we typically know the immediate effect of our moves, but not necessarily what happens after that.

It can be more efficient to learn based on afterstates because there are fewer values to represent than if we need to learn the full action value function.  Any state-action pair that maps to the same afterstate would be represented by a single value.  These afterstate value functions can also be learned with generalized policy iteration.
"""

# ╔═╡ 82f82d2a-beb4-4520-ac19-a498892d009c
md"""
To use afterstates with generalized policy iteration, we need to modify our MDP framework by considering the following trajectory:

$$(S, A) \longrightarrow (Y, P) \longrightarrow (S^\prime, R) \longrightarrow \cdots \longrightarrow (S_T, R_T)$$

where $(S, A, R)$ are the usual state, action, and reward.  We introduce $(Y, P)$ to indicate the afterstate and any intermediate reward that is received from the afterstate transition.

The probability transition function for a normal MDP is written as $p(s^\prime, r \vert s, a)$ and represents the probability of transitioning to state $s$ with reward $r$ under the condition that an agent takes action $a$ from state $s$.

When using afterstates, transitions can be represented with two functions:  

$p(y, \rho \vert s, a) \tag{a}$ is the probability of transitioning to afterstate $y$ with intermediate reward $\rho$ given an agent takes action $a$ from state $s$ 

$p(s^\prime, r \vert y) \tag{b}$ is the probability of transitioning to state $s^\prime$ with reward $r$ given an agent starts in afterstate $y$.  

Moreover, when an environment is modified to use afterstates, usually there are known deterministic dynamics that follow actions followed by some stochastic behavior after that.  A good example is tic-tac-toe where we fully know the dynamics after making a move, but there could be some unknown behavior from the opponent.  In this situation, the afterstate probability transition (a) is deterministic, so it could instead be represented by a mapping function that returns an afterstate and an intermediate reward given a state action pair.

$$f_1(s, a) = y \tag{b1′}$$

$$f_2(s, a) = \rho \tag{b2′}$$ 

where $y$ and $\rho$ are the afterstate and reward respectively after taking action $a$ in state $s$.  Now all of the stochastic dynamics of the environment are captured in (b) and the function only has 3 arguments instead of the usual 4.  We can now apply all of the previous techniques to the afterstate example and even combine dynamic programming and trajectory sampling.  
"""

# ╔═╡ 6b19aee6-a997-4eb4-9177-badd8ad2a540
md"""
Let's first consider the problem of prediction problem for afterstates and see how to compute the afterstate value function and how it could be used for policy improvement.  We will use the terminology $W(y)$ to represent the value of afterstate $y$ while $V(s)$ still means the value of state $s$.  From the earlier definitions, we can show the relationship between the state and afterstate value functions.

Recall that: 

$\begin{flalign} 
G_t &\doteq R_t + \gamma R_{t+1} + \cdots \\
V_\pi(s) &\doteq \mathbb{E}_\pi[G_t \mid S_t = s] \\
& = \mathbb{E}_\pi[R_t + \gamma V_\pi(S_{t+1}) \mid S_t = s] \\
&= \sum_a \pi(a \vert s) \sum_{r, s^\prime} p(r, s^\prime \vert s, a) \left ( r + \gamma V(s^\prime) \right )
\end{flalign}$

Representing the trajectory with afterstates and only considering the reward following an afterstate, we also know that: 

$\begin{flalign} 
G_t &\doteq R_t + \gamma(P_{t+1} + R_{t+1} + \gamma(P_{t+2} + R_{t+1} + \cdots))\\
W_\pi(y) &\doteq \mathbb{E}_\pi[G_t \mid Y_t = y] \\
& = \mathbb{E}_\pi[R_t + \gamma \left (P_{t+1} + W_\pi(Y_{t+1}) \right ) \mid Y_t = y] \\
&= \sum_{r, s^\prime} p(r, s^\prime \vert y) \left [r + \gamma \sum_{a^\prime} \left [ \pi(a \vert s^\prime) \left ( f_2(s^\prime, a^\prime) + W_\pi(f_1(s^\prime, a^\prime) \right ) \right ] \right ]
\end{flalign}$

Notice that compared to the value function, the policy only matters for this expected value when we consider the action taken from the transition state.  The initial transition from the afterstate to $s^\prime$ only depends on our new transition function which only conditioned on the afterstate.

Recall that to improve a policy $\pi$ for which we have a value function $V_\pi$, we must select the greedy policy with respect to $V_\pi$ meaning $\pi^{\prime} (s) = \mathrm{argmax}_a \sum_{r, s^\prime} p(r, s^\prime \vert s, a)(r + \gamma V(s^\prime))$.  If we do have access to the full probability transition function, we cannot compute this explicitely.  Furthermore, we cannot estimate this either from a single trajectory because from each state we would just have a single transition based on the behavior policy at the time.  That's why for MDPs that do not provide the full transition function, we prefer to estimate the state action value function $Q(s, a)$ because using that function policy improvement is much more trivial: $\pi^{\prime} (s) = \mathrm{argmax}_a Q(s, a)$.
"""

# ╔═╡ 610fc6de-6045-4c3f-8da1-95e9e5a4b986
md"""
Consider instead policy improvement with afterstate value estimates $W_\pi(y)$ where we seek to choose a policy that is greedy with respect to the afterstate values:

$\pi^\prime(s) = \mathrm{argmax}_a (f_2(s, a) + W_\pi(f_1(s, a))$

where $f_1$ and $f_2$ are the deterministic functions defined above that determine which afterstate is reached from $(s, a)$ and whether any intermediate reward is received.  This looks much closer to the policy improvement that occurs with $Q(s, a)$ and that is because $Q_\pi(s, a) = f_2(s, a) + W_\pi(f_1(s, a))$.  So, if we use afterstates, we can have the benefits of learning the state action value function while only saving values for the afterstates.  The functions $f_1$ and $f_2$ provide all the extra information needed to recover those values.

Continuing the comparison to value iteration, recall that we adapted the Bellman optimality equation for the state value function to have a single update rule to estimate $V^*(s)$:

$$V^*(s) = \max_a Q^*(s, a) = \max_a \sum_{r, s^\prime} p(r, s^\prime \vert s, a) (r + \gamma V^*(s^\prime))$$

We can only apply this update rule if we have $p(r, s^\prime \vert s, a)$ or if we instead estimate $Q^*$ and sample the transitions from the environment.  To estimate $W^*(y)$, we need to represent the Bellman optimality equation for the afterstate value function instead of the state value function:

$\begin{flalign}
W^*(y) &= \sum_{r, s^\prime} p(r, s^\prime \vert y)(r + \gamma \max_a(f_2(s^\prime, a) + W^*(f_1(s^\prime, a)))) \\
&= \sum_{r, s^\prime} p(r, s^\prime \vert y)r + \gamma \sum_{s^\prime}  p(s^\prime \vert y) \max_a(f_2(s^\prime, a) + W^*(f_1(s^\prime, a)))
\end{flalign}$

where $p(s^\prime \vert y) = \sum_r p(r, s^\prime \vert y)$

The outer sum is just represents an expected value based on the transition out of $y$, so if we don't have access to $p(r, s^\prime \vert y)$, we could sample the transitions from the environment.  The $\max_a$ term can now be calculated explicitely and will involve finding the maximum index of a vector for each transition state and does not depend on the reward.  Using state values, the maximization step involves evaluating a double sum every time, so each update with afterstates is less costly.  Also, the afterstates themselves might be more informative in the sense that they all have distinct values.  If many of the actions from a given state, lead to the same afterstate, this method will immediately treat them all as equal, whereas with usual value iterationthat equivalence would have to be calculated with the probability transition function.  The benefits of using an afterstate value function depend entirely on how effectively the environment transitions can be separated into informative deterministic steps and limited stochastic dynamics.  
"""

# ╔═╡ 903563ee-2f2a-48d4-991f-714d7da6808c
begin
	abstract type AbstractAfterstateMDP{T<:Real, S, AS, A} <: AbstractMDP{T, S, A} end
	abstract type AbstractTabularAfterstateMDP{T<:Real, S, AS, A} <: AbstractAfterstateMDP{T, S, AS, A} end

	struct FiniteAfterstateMDP{T<:Real, S, AS, A} <: AbstractTabularAfterstateMDP{T, S, AS, A}
		states::Vector{S}
		afterstates::Vector{AS}
		actions::Vector{A}
		rewards::Vector{T}
		#probability transition function now has probabilities for each state/reward transition from each afterstate
		ptf::Dict{AS, Dict{S, Tuple{T, T}}}
		#each column contains the index of the afterstate reached from the state represented by the column index while taking the action represented by the row index
		afterstate_map::Matrix{Int64}
		#each column contains the reward value received from the state represented by the column index while taking the action represented by the row index
		reward_interim_map::Matrix{T}
		state_index::Dict{S, Int64}
		afterstate_index::Dict{AS, Int64}
		action_index::Dict{A, Int64}
		function FiniteAfterstateMDP{T, S, AS, A}(states::Vector{S}, afterstates::Vector{AS}, actions::Vector{A}, rewards::Vector{T}, ptf::Array{T, 3}, afterstate_map::Matrix{Int64}, reward_interim_map::Matrix{T}) where {T <: Real, S, AS, A}
			new(states, afterstates, actions, rewards, ptf, afterstate_map, reward_interim_map, makelookup(states), makelookup(afterstates), makelookup(actions))
		end	
	end
	FiniteAfterstateMDP(states::Vector{S}, afterstates::Vector{AS}, actions::Vector{A}, rewards::Vector{T}, ptf::Array{T, 3}, afterstate_map::Matrix{Int64}, reward_interim_map::Matrix{T}) where {T <: Real, S, AS, A} = FiniteAfterstateMDP{T, S1, S2, A}(states, afterstates, actions, rewards, ptf, afterstate_map, reward_interim_map)
	#if a reward map is not provided, assume that there are no intermediate rewards
	FiniteAfterstateMDP(states::Vector{S}, afterstates::Vector{AS}, actions::Vector{A}, rewards::Vector{T}, ptf::Array{T, 3}, afterstate_map::Matrix{Int64}) where {T <: Real, S, AS, A} = FiniteAfterstateMDP{T, S, AS, A}(states, afterstates, actions, rewards, ptf, afterstate_map, zeros(T, length(actions), length(states)))

	struct AfterstateMDP{T<:Real, S, AS, A, F<:Function, G<:Function, H<:Function, I<:Function} <: AbstractAfterstateMDP{T, S, AS, A}
		actions::Vector{A}
		afterstate_step::F #deterministic function that produces a reward and afterstate given a state action pair
		afterstate_transition::G #deterministic function that preduces a probability distribution of reward, state pairs given an afterstate
		state_init::H #function that produces an initial state for an episode
		isterm::I #function that returns true if the state input is terminal
		action_index::Dict{A, Int64}
		function AfterstateMDP(state_init::H, actions::Vector{A}, afterstate_step::F, afterstate_transition::G, isterm::I) where {A, F<:Function, G<:Function, H<:Function, I<:Function}
			s0 = state_init()
			(r, w) = afterstate_step(s0, 1)
			action_index = makelookup(actions)
			new{typeof(r), typeof(s0), typeof(w), A, F, G, H, I}(actions, afterstate_step, afterstate_transition, state_init, isterm, action_index)
		end	
	end
end

# ╔═╡ 451e7b71-a495-4b95-803f-4fedd9268316
#perform a transition step for an afterstate MDP
function get_transition(mdp::FiniteAfterstateMDP, i_s::Integer, i_a::Integer)
	i_w = afterstate_map[i_a, i_s]
	r0 = rward_interim_map[i_a, i_s]
	ptf = mdp.ptf[i_w]
	probabilities = [ptf[i_s′][1] for i_s′ in keys(ptf)]
	i_s′ = sample(collect(keys(ptf)), weights(probabilities))
	s′ = mdp.states[i_s′]
	r = ptf[i_s′][2]
	return (r+r0, i_s′)
end

# ╔═╡ dc3e1ed4-3e48-4bf0-9cc0-a7ce0eab226e
function takestep(mdp::AbstractCompleteMDP{T, S, A}, π::Matrix{T}, i_s::Integer) where {T<:Real, S, A}
	i_a = sample_action(π, i_s)
	(r, i_s′) = get_transition(mdp, i_s, i_a)
	return (r, i_s′, i_a)
end

# ╔═╡ 74a4fba2-06b4-4198-9e57-6f5911760a72
function bellman_optimal_value!(V::Vector{T}, mdp::FiniteAfterstateMDP{T, S, AS, A}, γ::T) where {T <: Real, S, AS, A}
	delt = zero(T)
	q_vec = zeros(T, length(mdp.actions))
	@inbounds @fastmath @simd for i_y in eachindex(mdp.afterstates)
		q_total = zero(T)
		r_total = zero(T)
		@inbounds @fastmath @simd for i_s′ in eachindex(mdp.states)
			p_total = zero(T)
			q_vec .= mdp.reward_interim_map[:, i_s′] .+ V[mdp.afterstate_map[:, i_s′]]
			q_max = maximum(q_vec)
			@inbounds @fastmath for (i_r, r) in enumerate(mdp.rewards)
				p = mdp.ptf[i_s′, i_r, i_y]
				r_total += p*r
				p_total += p
			end
			q_total += q_max*p_total
		end
		v_new = r_total + γ*q_total
		delt = max(delt, abs(v_new - V[i_y]) / (eps(abs(V[i_y])) + abs(V[i_y])))
		V[i_y] = v_new
	end
	return delt
end

# ╔═╡ ecebce8b-0e2a-49d0-89f5-53bd0ffdd1a3
function value_iteration!(v_est, θ, mdp, γ, nmax, valuelist)
	nmax <= 0 && return valuelist
	
	#update value function
	delt = bellman_optimal_value!(v_est, mdp, γ)
	
	#add copy of value function to results list
	push!(valuelist, copy(v_est))

	#halt when value function is no longer changing
	delt <= θ && return valuelist
	
	value_iteration!(v_est, θ, mdp, γ, nmax - 1, valuelist)	
end

# ╔═╡ 42dab9c3-dd04-4129-ba3c-b6fb22e2afbe
function value_iteration!(v_est, θ, mdp, γ, nmax)
	nmax <= 0 && return v_est
	
	#update value function
	delt = bellman_optimal_value!(v_est, mdp, γ)

	#halt when value function is no longer changing
	delt <= θ && return v_est
	
	value_iteration!(v_est, θ, mdp, γ, nmax - 1)	
end

# ╔═╡ a617d0a8-f1d7-492a-98a9-357a8223d6b0
function make_greedy_policy!(π::Matrix{T}, mdp::FiniteAfterstateMDP{T, S, AS, A}, V::Vector{T}, γ::T) where {T<:Real,S,AS,A}
	for i_s in eachindex(mdp.states)
		π[:, i_s] .= mdp.reward_interim_map[:, i_s] .+ V[mdp.afterstate_map[:, i_s]]
		maxv = -T(Inf)
		@inbounds @fastmath @simd for i_a in eachindex(mdp.actions)
			maxv = max(maxv, π[i_a, i_s])
		end
		π[:, i_s] .= (π[:, i_s] .≈ maxv)
		x = zero(T)
		@fastmath @inbounds @simd for i_a in eachindex(mdp.actions)
			x += π[i_a, i_s]
		end
		π[:, i_s] ./= x
	end
	return π
end

# ╔═╡ 1dab32e6-9d81-4de3-9b97-6a2ac58a28c3
function generalized_episodic_sarsa(mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, γ::T, num_episodes::Integer, α::T, setup_function::Function; save_value_history = false, ϵ = T(0.1), kwargs...) where {T<:Real,S, A, F<:Function, G<:Function, H<:Function}
	(episode_update_args, episode_update!, get_q, get_πtarget) = setup_function(mdp, γ, ϵ, α; kwargs...)
	if save_value_history
		q_history = zeros(T, size(q)..., num_episodes)
	end
	reward_history = zeros(T, num_episodes)
	step_history = zeros(Int64, num_episodes)
	
	for ep in 1:num_episodes
		(total_reward, total_steps) = episode_update!(episode_update_args...)
		if save_value_history
			q_history[:, :, ep] .= get_q(episode_update_args)
		end
		reward_history[ep] = total_reward
		step_history[ep] = total_steps
	end
	final_q = get_q(episode_update_args)
	final_π = make_greedy_policy!(get_πtarget(episode_update_args), mdp, final_q)
	base_return = (final_policy = final_π, final_value_estimate = final_q, reward_history = reward_history, step_history = step_history)
	if save_value_history
		return (;base_return..., value_estimate_history = q_history)
	else
		return base_return
	end
end

# ╔═╡ 9a4027f9-243d-4fc6-916a-2f89a76120c9
episodic_expected_sarsa(args...; kwargs...) = generalized_episodic_sarsa(args..., setup_expected_sarsa; kwargs...)

# ╔═╡ adfc5819-6bb7-40fc-baef-9770efeb6a21
episodic_double_expected_sarsa(args...; kwargs...) = generalized_episodic_sarsa(args..., setup_double_expected_sarsa; kwargs...)

# ╔═╡ ad6e7b16-3819-4d59-9e7c-6bc83d6df468
#the value function in this case represents the value of each afterstate.  the afterstates are listed in mdp.afterstates while the states are listed in mdp.states
begin_value_iteration_v(mdp::FiniteAfterstateMDP{T,S,AS,A}, γ::T; Vinit::T = zero(T), kwargs...) where {T<:Real,S,AS,A} = begin_value_iteration_v(mdp, γ, Vinit .* ones(T, length(mdp.afterstates)); kwargs...)

# ╔═╡ 78ecd319-1f5c-4ba0-b9c4-da0dfadb4b2c
@skip_as_script md"""
## Planning and Learning
"""

# ╔═╡ a912feaa-b2b2-479e-befe-9e919e453e31
@skip_as_script md"""
### Monte Carlo Tree Search

For some MDP's, the state space may be too large to enumerate.  We could still sample from the environment and know ahead of time examples of states and actions, but in these problems we can never compute a complete solution.  Monte Carlo Tree Search (MCTS) allows us to update state-action value estimates but only for selected states that we encounter during interactions with the environment.  The data we collect can be used to build a partial model of the environment that we use to improve the value estimates and policy without having a full solution.  All of the previous MDP types we defined contain a complete state list.  This technique can apply to any MDP, even those for which we do not know all of the states.
"""

# ╔═╡ 305c2556-e269-40ea-ac90-6e75f03f07ce
begin
	struct SampleMDP{T<:Real, S, A, F<:Function, G<:Function, H<:Function} <: AbstractMDP{T, S, A}
		actions::Vector{A}
		action_index::Dict{A, Int64}
		step::F #step(s::S, a::A) must return a tuple (r::T, s′::S)
		state_init::G #state_init() must return an initial state s_0::S
		isterm::H #isterm(s::S) must return a boolean inicating whether state s is terminal
		function SampleMDP(actions::Vector{A}, step::F, state_init::G, isterm::H) where {A, F<:Function, G<:Function, H<:Function}
			s0 = state_init()
			transition = step(s0, first(actions))
			!(typeof(transition) <: Tuple) && error("step function is not returning a tuple of (r, s)")
			(r, s′) = transition
			T = typeof(r)
			!(typeof(s′) == typeof(s0))  && error("transition state is not of type $(typeof(s0))")
			!(T <: Real) && error("Reward is not a real number")
			!isterm(s′) #check to see if isterm function takes a state and returns a boolean
			new{T, typeof(s0), A, F, G, H}(actions, makelookup(actions), step, state_init, isterm)
		end
	end

	#convert a tabular MDP into a generic one
	function SampleMDP(mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}) where {T <:Real, S, A, F<:Function, G<:Function, H<:Function}
		function step(s::S, a::A) 
			(r, i_s′) = mdp.step(mdp.state_index[s], mdp.action_index[a])
			(r, mdp.states[i_s′])
		end

		state_init() = mdp.states[mdp.state_init()]

		isterm(s::S) = mdp.isterm(mdp.state_index[s])
		
		SampleMDP(mdp.actions, step, state_init, isterm)
	end

	#convert a tabular MDP into a generic one
	function SampleMDP(mdp::AbstractCompleteMDP, state_init::Function)
		sample_tabular_mdp = SampleTabularMDP(mdp, state_init)
		SampleMDP(sample_tabular_mdp)
	end
end

# ╔═╡ 221814d5-676a-4bbf-9617-a25cfe1c5f47
@skip_as_script const windy_mc_gridworld = SampleMDP(windy_sample_gridworld)

# ╔═╡ fc0d29f4-fd2e-45b0-ba19-f7552643efc7
make_random_policy(mdp::SampleMDP) = s -> rand(mdp.actions)

# ╔═╡ 62436d67-a417-476f-b508-da752796c774
@skip_as_script const example_gridworld_random_policy = make_random_policy(deterministic_gridworld.mdp)

# ╔═╡ ac5f7dcc-02ba-421c-a593-ca7ba60b3ff2
@skip_as_script deterministic_gridworld_random_policy_evaluation = policy_evaluation_v(deterministic_gridworld.mdp, example_gridworld_random_policy, γ_gridworld_policy_evaluation);

# ╔═╡ 7851e968-a5af-4b65-9591-e34b3404fb09
@skip_as_script md"""
Deterministic Gridworld
Converged after $(deterministic_gridworld_random_policy_evaluation.total_iterations) iterations
"""

# ╔═╡ 0f6cc7a9-4184-471f-86d5-4ad0c0e495ce
@skip_as_script windy_gridworld_random_policy_evaluation = policy_evaluation_v(windy_gridworld.mdp, example_gridworld_random_policy, γ_gridworld_policy_evaluation);

# ╔═╡ 8bfaa611-35fd-44d3-920f-c7c51d02216f
@skip_as_script md"""
Windy Gridworld
Converged after $(windy_gridworld_random_policy_evaluation.total_iterations) iterations
"""

# ╔═╡ 966eae0d-7556-4ff9-b9f7-d47a736524a4
@skip_as_script stochastic_gridworld_random_policy_evaluation = policy_evaluation_v(stochastic_gridworld.mdp, example_gridworld_random_policy, γ_gridworld_policy_evaluation);

# ╔═╡ 91ca282d-e857-41d7-b99d-d9449b82da09
@skip_as_script md"""
Stochastic Gridworld
Converged after $(stochastic_gridworld_random_policy_evaluation.total_iterations) iterations
"""

# ╔═╡ f87fd155-d6cf-4a27-bbc4-74cc64cbd84c
"""
    policy_iteration(mdp::AbstractCompleteMDP{T, S, A}, γ::T, initialize_value::Function; max_iterations = 10, save_history=true, kwargs...) where {T<:Real, S, A}

Performs policy iteration to find an optimal policy for a given MDP.

# Arguments
- `mdp::AbstractCompleteMDP{T, S, A}`: The complete MDP for which policy iteration is performed.
- `γ::T`: The discount factor.
- `initialize_value::Function`: A function to initialize the value function for policy evaluation.
- `max_iterations::Int`: The maximum number of iterations for policy iteration. Default is 10.
- `save_history::Bool`: A flag indicating whether to save the history of policies and value functions during iterations. Default is `true`.
- `kwargs...`: Additional keyword arguments to customize the behavior of policy evaluation. See `policy_evaluation!` documentation for details.

# Returns
- If `save_history` is `true`, returns a tuple `(π_list, v_list)` containing the history of policies and value functions.
  - `π_list::Vector{Matrix{T}}`: A vector of policy matrices representing the policies at each iteration.
  - `v_list::Vector{Array{T, 1}}`: A vector of value functions representing the value functions at each iteration.
- If `save_history` is `false`, returns a tuple `(πgreedy, v_π)` containing the final greedy policy and its associated value function.
  - `πgreedy::Matrix{T}`: The final greedy policy matrix.
  - `v_π::Array{T, 1}`: The value function associated with the final greedy policy.

# Description
This function performs policy iteration to find an optimal policy for the given MDP `mdp`. Policy iteration consists of alternating between policy evaluation and policy improvement steps until convergence.

The value function is initialized using the `initialize_value` function. Policy evaluation is performed to estimate the value function for the current policy, and then policy improvement is applied to update the policy to be greedy with respect to the estimated value function.

The process continues until the policy no longer changes between iterations, indicating convergence.

The `save_history` flag determines whether to save the history of policies and value functions during iterations. If `true`, the function returns the history; otherwise, it returns only the final greedy policy and its associated value function.

Additional customization of the policy evaluation process can be achieved by providing keyword arguments `kwargs`, which are passed to the `policy_evaluation!` function.
"""
function policy_iteration(mdp::AbstractCompleteMDP{T, S, A}, γ::T, initialize_value::Function; max_iterations = 10, save_history=true, kwargs...) where {T<:Real, S, A}
	πgreedy = make_random_policy(mdp)
	πlast = copy(πgreedy)
	v_π = initialize_value(mdp)
	(v_π, num_iterations, num_updates) = policy_evaluation!(v_π, πgreedy, mdp, γ; kwargs...)
	if save_history
		π_list = [πgreedy]
		v_list = [copy(v_π)]
	end
	make_greedy_policy!(πgreedy, mdp, v_π, γ)
	πlast .= πgreedy
	converged = false
	while !converged
		save_history && push!(π_list, copy(πgreedy))
		(v_π, num_iterations, num_updates) = policy_evaluation!(v_π, πgreedy, mdp, γ; kwargs...)
		save_history && push!(v_list, copy(v_π))
		make_greedy_policy!(πgreedy, mdp, v_π, γ)
		converged = (πgreedy == πlast)
		πlast .= πgreedy
	end

	if save_history
		return π_list, v_list
	else
		return πgreedy, v_π
	end
end

# ╔═╡ a59f0142-9f0c-452b-91ea-647f9201a8d6
"""
    policy_iteration_v(args...; kwargs...)

Performs policy iteration to find an optimal policy for a given MDP using state-value function initialization.

This function is a convenience wrapper around `policy_iteration` with the state-value function initialization.

# Arguments
- `args...`: Positional arguments for `policy_iteration`.
- `kwargs...`: Keyword arguments for `policy_iteration`.

# Returns
The return value of `policy_iteration`.

# Description
This function performs policy iteration to find an optimal policy for the given MDP using state-value function initialization. It is a convenience wrapper around `policy_iteration`, where the state-value function initialization is automatically applied.

# Example
```julia
# Define your MDP
mdp = ...

# Perform policy iteration with state-value function initialization
result = policy_iteration_v(mdp, γ=0.9)
```
"""
policy_iteration_v(args...; kwargs...) = policy_iteration(args..., initialize_state_value; kwargs...)

# ╔═╡ 6d74b5de-1fc9-48af-96dd-3e090f691641
@skip_as_script π_list, v_list = policy_iteration_v(new_gridworld.mdp, policy_iteration_params.γ);

# ╔═╡ f218de8b-6003-4bd2-9820-48165cfde650
@skip_as_script md"""Policy iteration converged after $(length(π_list) - 1) steps"""

# ╔═╡ 3a868cc5-4123-4b5f-be87-589430df389f
@skip_as_script md"""Number of Policy Iterations: $(@bind policy_iteration_count Slider(0:length(π_list) .- 1; show_value=true, default = length(π_list) - 1))"""

# ╔═╡ 7f3a1d41-dd16-493c-a59c-764aec13d076
"""
    policy_iteration_q(args...; kwargs...)

Performs policy iteration to find an optimal policy for a given MDP using state-action value function initialization.

This function is a convenience wrapper around `policy_iteration` with the state-action value function initialization.

# Arguments
- `args...`: Positional arguments for `policy_iteration`.
- `kwargs...`: Keyword arguments for `policy_iteration`.

# Returns
The return value of `policy_iteration`.

# Description
This function performs policy iteration to find an optimal policy for the given MDP using state-action value function initialization. It is a convenience wrapper around `policy_iteration`, where the state-action value function initialization is automatically applied.

# Example
```julia
using ReinforcementLearning

# Define your MDP
mdp = ...

# Perform policy iteration with state-action value function initialization
result = policy_iteration_q(mdp, γ=0.9)
```
"""
policy_iteration_q(args...; kwargs...) = policy_iteration(args..., initialize_state_action_value; kwargs...)

# ╔═╡ 1e24a0aa-dbf9-422e-92c9-834f293a0c02
function value_iteration(mdp::M, γ::T, v_est; θ = eps(zero(T)), nmax=typemax(Int64), save_history = true) where {T<:Real, M <: AbstractCompleteMDP{T, S, A} where {S, A}}
	if save_history
		valuelist = [copy(v_est)]
		value_iteration!(v_est, θ, mdp, γ, nmax, valuelist)
	else
		value_iteration!(v_est, θ, mdp, γ, nmax)
	end

	π = make_random_policy(mdp)
	make_greedy_policy!(π, mdp, v_est, γ)
	if save_history
		return (valuelist, π)
	else
		return (v_est, π)
	end
end

# ╔═╡ eec3017b-6d02-49e6-aedf-9a494b426ec5
value_iteration_v(mdp::AbstractCompleteMDP{T,S,A}, γ::T; init_value::T = zero(T), kwargs...) where {T<:Real,S,A} = value_iteration(mdp, γ, initialize_state_value(mdp; init_value = init_value); kwargs...)

# ╔═╡ 929c353b-f67c-49ff-85d3-0a27cafc59cf
@skip_as_script const value_iteration_grid_example = value_iteration_v(new_gridworld.mdp, value_iteration_γ);

# ╔═╡ 4f645ebc-27f4-4b68-93d9-2e35232cedcf
@skip_as_script const value_iteration_grid_example2 = value_iteration_v(deterministic_gridworld.mdp, mc_control_γ)

# ╔═╡ 2fe59959-5d89-4ae7-839c-ecf82e2c71d8
value_iteration_q(mdp::AbstractCompleteMDP{T,S,A}, γ::T; init_value::T = zero(T), kwargs...) where {T<:Real,S,A} = value_iteration(mdp, γ, initialize_state_action_value(mdp; init_value = init_value); kwargs...)

# ╔═╡ 59307ddd-c24b-444f-9723-badc7e6da897
function setup_episodic_sarsa(mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, γ::T, ϵ::T, α::T; q = initialize_state_action_value(mdp), π = make_random_policy(mdp)) where {T<:Real,S, A, F<:Function, G<:Function, H<:Function}
	update_policy!(π, q, i_s) = make_ϵ_greedy_policy!(π, mdp, q, ϵ, i_s)
	episode_update_args = ((q, π), update_policy!, mdp, γ, α)
	get_q(episode_update_args) = first(first(episode_update_args))
	get_πtarget(episode_update_args) = last(first(episode_update_args))
	(episode_update_args = episode_update_args, episode_update! = sarsa_episode_update!, get_q = get_q, get_πtarget = get_πtarget)
end

# ╔═╡ 2f70e03f-1556-4fce-b4f5-394df4266eb7
episodic_sarsa(args...; kwargs...) = generalized_episodic_sarsa(args..., setup_episodic_sarsa; kwargs...)

# ╔═╡ 6823a91e-c02e-495c-9e82-e22b18857df7
@skip_as_script sarsa_test = episodic_sarsa(deterministic_sample_gridworld, 0.9f0, 10_000, 0.1f0; ϵ = 0.5f0)

# ╔═╡ c75d9e65-f9be-4b8a-9bd4-9dbeeafec16e
@skip_as_script plot(cumsum(sarsa_test.step_history) ./ collect(1:length(sarsa_test.step_history)))

# ╔═╡ ae700906-4b17-45ce-b27e-e2a0d745e259
function setup_episodic_expected_sarsa(mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, γ::T, ϵ::T, α::T; q = initialize_state_action_value(mdp), π_target = make_random_policy(mdp), π_behavior = make_random_policy(mdp), update_target_policy! = (π, q, i_s) -> make_ϵ_greedy_policy!(π, mdp, q, ϵ, i_s), update_behavior_policy! = (π, mdp, q, i_s) -> make_ϵ_greedy_policy!(π, mdp, q, ϵ, i_s)) where {T<:Real,S, A, F<:Function, G<:Function, H<:Function}
	episode_update_args = ((q, π_target, π_behavior), update_target_policy!, update_behavior_policy!, mdp, γ, α)
	get_q(episode_update_args) = first(first(episode_update_args))
	get_πtarget(episode_update_args) = first(episode_update_args)[2]
	(episode_update_args = episode_update_args, episode_update! = expected_sarsa_episode_update!, get_q = get_q, get_πtarget = get_πtarget)
end

# ╔═╡ 823d65da-5636-4f7d-9582-2a0189a564ae
setup_episodic_q_learning(args...; kwargs...) = setup_episodic_expected_sarsa(args...; update_target_policy! = make_greedy_policy!, kwargs...)

# ╔═╡ 94193bc1-91c4-4d3e-8e44-cd37495481bf
episodic_q_learning(args...; kwargs...) = generalized_episodic_sarsa(args..., setup_episodic_q_learning; kwargs...)

# ╔═╡ d7de7be9-8d97-4476-ba09-9f84d2cebb00
@skip_as_script expected_sarsa_test = episodic_q_learning(deterministic_sample_gridworld, 0.9f0, 1_000, 0.1f0; ϵ = 0.5f0)

# ╔═╡ 5b66bf73-b7dd-4054-9efb-1c30a475bc6b
@skip_as_script plot(cumsum(expected_sarsa_test.step_history) ./ collect(1:length(expected_sarsa_test.step_history)))

# ╔═╡ 447cd06f-e110-450d-984f-ceb1d6361b43
function setup_episodic_double_expected_sarsa(mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, γ::T, ϵ::T, α::T; q1 = initialize_state_action_value(mdp), q2 = copy(q1), π_target1 = make_random_policy(mdp), π_target2 = copy(π_target1), π_behavior = make_random_policy(mdp), update_target_policy! = (π, q, i_s) -> make_ϵ_greedy_policy!(π, mdp, q, ϵ, i_s), update_behavior_policy! = (π, mdp, q1, q2, i_s) -> make_ϵ_greedy_policy!(π, mdp, q1, q2, ϵ, i_s)) where {T<:Real,S, A, F<:Function, G<:Function, H<:Function}
	episode_update_args = ((q1, q2, π_target1, π_target2, π_behavior), update_target_policy!, update_behavior_policy!, mdp, γ, α)
	get_q(episode_update_args) = (first(episode_update_args)[1] .+ first(episode_update_args)[2]) ./ 2
	get_πtarget(episode_update_args) = (first(episode_update_args)[3] .+ first(episode_update_args)[4]) ./ 2
	(episode_update_args = episode_update_args, episode_update! = double_expected_sarsa_episode_update!, get_q = get_q, get_πtarget = get_πtarget)
end

# ╔═╡ ffefe265-b7a3-4bc4-9e5f-2da6c406cc56
setup_episodic_double_q_learning(args...; kwargs...) = setup_episodic_double_expected_sarsa(args...; update_target_policy! = make_greedy_policy!, kwargs...)

# ╔═╡ cd834845-8ca9-407a-91da-d3104b0bd9b7
episodic_double_q_learning(args...; kwargs...) = generalized_episodic_sarsa(args..., setup_episodic_double_q_learning; kwargs...)

# ╔═╡ 3eca2837-16fb-4237-9ebd-8b6378ca13a8
@skip_as_script double_expected_sarsa_test = episodic_double_q_learning(deterministic_sample_gridworld, 0.9f0, 1_000, 0.1f0; ϵ = 0.5f0)

# ╔═╡ c87db76f-4c6a-4fe2-822b-8ee88079e30d
@skip_as_script plot(cumsum(double_expected_sarsa_test.step_history) ./ collect(1:length(double_expected_sarsa_test.step_history)))

# ╔═╡ 2abe4ba3-3834-4f20-9ce4-2496711fb0af
begin
	function takestep(mdp::SampleMDP{T, S, A, F, G, H}, π::Function, s::S) where {T<:Real, S, A, F<:Function, G<:Function, H<:Function}
		a = π[s]
		(r, s′) = mdp.step(s, a)
		return (r, s′, a)
	end

	function takestep(mdp::SampleMDP{T, S, A, F, G, H}, s::S, a::A) where {T<:Real, S, A, F<:Function, G<:Function, H<:Function}
		(r, s′) = mdp.step(s, a)
		return (r, s′, a)
	end
end

# ╔═╡ 2f7afb63-22de-49af-b907-4aeb75dc9f2a
begin
"""
    runepisode(mdp::AbstractCompleteMDP{T, S, A}, i_s0::Integer, isterm::Function, π::Matrix{T}; max_steps = Inf) where {T<:Real, S, A}

Runs an episode in a complete Markov Decision Process (MDP) using a given policy.

# Arguments
- `mdp::AbstractCompleteMDP{T, S, A}`: The complete MDP in which the episode is run.
- `i_s0::Integer`: The index of the initial state for the episode.
- `isterm::Function`: A function that determines whether a state is terminal.
- `π::Matrix{T}`: The policy matrix representing the probability of selecting each action in each state. Each element `π[i, j]` denotes the probability of taking the action represented by index `i` in the state represented by index `j`.
- `max_steps::Int = Inf`: The maximum number of steps allowed for the episode. Default is `Inf`.

# Returns
- `Tuple{Vector{Int64}, Vector{Int64}, Vector{T}, Int64}`: A tuple containing the indices of the visited states (`states`), the indices of the actions taken (`actions`), the rewards received at each step (`rewards`), and the index of the terminal state reached (`i_sterm`).

# Description
This function runs an episode in a complete Markov Decision Process (MDP) using a given policy `π`. The episode starts from the initial state `i_s0` and continues until a terminal state is reached or the maximum number of steps `max_steps` is exceeded. At each step, the agent selects an action according to the policy `π` and transitions to the next state based on the transition dynamics of the MDP. The episode terminates when a terminal state is reached or the maximum number of steps is exceeded.

The policy matrix `π` is a two-dimensional matrix where each row represents an action and each column represents a state. The element `π[i, j]` denotes the probability of taking the action represented by index `i` in the state represented by index `j`.

The function returns a tuple containing the indices of the visited states (`states`), the indices of the actions taken (`actions`), the rewards received at each step (`rewards`), and the index of the terminal state reached (`i_sterm`).
"""
function runepisode(mdp::AbstractCompleteMDP{T, S, A}, i_s0::Integer, isterm::Function, π::Matrix{T}; max_steps = Inf) where {T<:Real, S, A}
	i_s = i_s0
	states = Vector{Int64}()
	actions = Vector{Int64}()
	push!(states, i_s)
	(r, i_s′, i_a) = takestep(mdp, π, i_s)
	push!(actions, i_a)
	rewards = [r]
	step = 2
	i_sterm = i_s
	if isterm(i_s′)
		i_sterm = i_s′
	else
		i_sterm = i_s
	end
	i_s = i_s′

	#note that the terminal state will not be added to the state list
	while !isterm(i_s) && (step <= max_steps)
		push!(states, i_s)
		(r, i_s′, i_a) = takestep(mdp, π, i_s)
		push!(actions, i_a)
		push!(rewards, r)
		i_s = i_s′
		step += 1
		if isterm(i_s′)
			i_sterm = i_s′
		end
	end
	return states, actions, rewards, i_sterm
end

"""
    runepisode(mdp::AbstractCompleteMDP, i_s0::Integer, isterm::Function; kwargs...)

Runs an episode in a complete Markov Decision Process (MDP) using a given policy or a random policy.

# Keyword Arguments
- `π::Matrix{T}`: (Optional) The policy matrix representing the probability of selecting each action in each state. If not provided, a random policy is generated using `make_random_policy(mdp)`.
- `max_steps::Int = Inf`: (Optional) The maximum number of steps allowed for the episode. Default is `Inf`.
"""
runepisode(mdp::AbstractCompleteMDP{T, S, A}, i_s0::Integer, isterm::Function; kwargs...) where {T<:Real, S, A} = runepisode(mdp, i_s0, isterm, make_random_policy(mdp); kwargs...)
end

# ╔═╡ e4476a04-036e-4074-bd90-54475c00800a
begin
	
function runepisode(mdp::SampleTabularMDP{T, S, A, F, G, H}, π::Matrix{T}, i_s0::Integer, i_a0::Integer; max_steps = Inf) where {T<:Real, S, A, F<:Function, G<:Function, H<:Function}
	i_s = i_s0
	i_a = i_a0
	states = Vector{Int64}()
	actions = Vector{Int64}()
	push!(states, i_s)
	(r, i_s′) = mdp.step(i_s, i_a)
	push!(actions, i_a)
	rewards = [r]
	step = 2
	i_sterm = i_s
	if mdp.isterm(i_s′)
		i_sterm = i_s′
	else
		i_sterm = i_s
	end
	i_s = i_s′

	#note that the terminal state will not be added to the state list
	while !mdp.isterm(i_s) && (step <= max_steps)
		push!(states, i_s)
		(r, i_s′, i_a) = takestep(mdp, π, i_s)
		push!(actions, i_a)
		push!(rewards, r)
		i_s = i_s′
		step += 1
		if mdp.isterm(i_s′)
			i_sterm = i_s′
		end
	end
	return states, actions, rewards, i_sterm
end

"""
	runepisode(mdp::SampleTabularMDP{T, S, A, F, G, H}, π::Matrix{T}; i_s0::Integer = mdp.state_init(), i_a0 = sample_action(π, i_s0), kwargs...)
"""
runepisode(mdp::SampleTabularMDP{T, S, A, F, G, H}, π::Matrix{T}; i_s0::Integer = mdp.state_init(), i_a0 = sample_action(π, i_s0), kwargs...) where {T<:Real, S, A, F<:Function, G<:Function, H<:Function} = runepisode(mdp, π, i_s0, i_a0; kwargs...)

"""
	runepisode(mdp::SampleTabularMDP{T, S, A, F, G, H}, π::Matrix{T}; i_s0::Integer = mdp.state_init(), i_a0 = sample_action(π, i_s0), kwargs...)
"""
runepisode(mdp::SampleTabularMDP{T, S, A, F, G, H}; π::Matrix{T} = make_random_policy(mdp), kwargs...) where {T<:Real, S, A, F<:Function, G<:Function, H<:Function} = runepisode(mdp, π; kwargs...)

end

# ╔═╡ 1fed0e8d-0014-4484-8b61-29807caa8ef7
@skip_as_script runepisode(deterministic_gridworld.mdp, deterministic_gridworld.init_state, deterministic_gridworld.isterm)

# ╔═╡ 0fdaf201-2cdf-419d-9452-4ec14ea281dc
@skip_as_script runepisode(windy_gridworld.mdp, windy_gridworld.init_state, windy_gridworld.isterm)

# ╔═╡ 6e73940d-15fb-4f61-8100-05fdf7f50e10
@skip_as_script runepisode(stochastic_gridworld.mdp, stochastic_gridworld.init_state, stochastic_gridworld.isterm)

# ╔═╡ 0a81b18a-0ac8-45ba-ad46-02034ae8fb55
#verify that the episode function works with the sample mdp
@skip_as_script runepisode(deterministic_sample_gridworld, make_random_policy(deterministic_sample_gridworld))

# ╔═╡ ea19d77b-96bf-411f-8faa-6007c11e204b
function monte_carlo_policy_prediction(mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, π::Matrix{T}, γ::T, num_episodes::Integer, initialize_value_function::Function; v_est = initialize_value_function(mdp), averaging_method::AbstractAveragingMethod = SampleAveraging(v_est), save_history = false, kwargs...) where {T<:Real,S, A, F<:Function, G<:Function, H<:Function}
	if save_history
		v_history = zeros(T, size(v_est)..., num_episodes)
	end
	for ep in 1:num_episodes
		(states, actions, rewards, _) = runepisode(mdp, π)
		monte_carlo_episode_update!(v_est, states, actions, rewards, mdp, γ, averaging_method; kwargs...)
		save_history && update_value_history!(v_history, v_est, ep)
	end
	final_v = v_est
	if save_history
		return (final_value_estimate = final_v, value_estimate_history = v_history)
	else
		return v_est
	end
end

# ╔═╡ e375ca3a-57a7-4ca3-a672-4aa724cba34d
#by default values are updated with sample averaging, to use constant step size averaging instead use the keyword argument averaging_method = ConstantStepAveraging(α) where α is the step size and must match the numerical type of the value function
monte_carlo_policy_prediction_v(args...; kwargs...) = monte_carlo_policy_prediction(args..., initialize_state_value; kwargs...)

# ╔═╡ ad55c2d1-404f-4396-aff8-b8c207157ce4
#test state value policy prediction with gridworld random policy
@skip_as_script monte_carlo_policy_prediction_v(deterministic_sample_gridworld, make_random_policy(deterministic_sample_gridworld), 0.99f0, 1_000)

# ╔═╡ 4d6472e3-cbb6-4b5c-b06a-4210ff940409
#given an AbstractCompleteMDP, compare the results of policy prediction with mc sampling with dynamic programming policy evaluation.  computes the RMS error across all the states as it changes with learning episode and averaged over trials
function check_mc_error(mdp::AbstractCompleteMDP, state_init::Function, γ::T, num_episodes::Integer; num_trials = 100) where T<:Real
	mdp_sample = SampleTabularMDP(mdp, state_init)
	v_true = policy_evaluation_v(mdp, make_random_policy(mdp), γ)

	1:num_trials |> Map() do _
		v_sample = monte_carlo_policy_prediction_v(mdp_sample, make_random_policy(mdp_sample), γ, num_episodes; save_history = true)
		mean((v_sample.value_estimate_history .- v_true.value_function) .^ 2, dims = 1)[:]
	end |> foldxt((v1, v2) -> v1 .+ v2) |> v -> sqrt.(v ./ num_trials) 
end

# ╔═╡ 4e6b27be-79c3-4224-bfc1-7d4b83be6d39
@skip_as_script plot([scatter(y = check_mc_error(x[1].mdp, () -> deterministic_gridworld.init_state, γ_mc_predict, 100), name = x[2]) for x in zip([deterministic_gridworld, windy_gridworld, stochastic_gridworld], ["deterministic gridworld", "windy gridworld", "stochastic gridworld"])], Layout(xaxis_title = "Learning Episodes", yaxis_title = "Average RMS Error of State Values", title = "Monte Carlo State Value Prediction Error Decreases with More Episodes"))

# ╔═╡ 37a7a557-77ea-4440-8bf0-05f34b55ffc6
monte_carlo_policy_prediction_q(args...; kwargs...) = monte_carlo_policy_prediction(args..., initialize_state_action_value; kwargs...)

# ╔═╡ ba25b564-230b-4e06-aba5-c7d3197970ef
#test state-action value policy prediction with gridworld random policy
@skip_as_script monte_carlo_policy_prediction_q(deterministic_sample_gridworld, make_random_policy(deterministic_sample_gridworld), 0.99f0, 1_000)

# ╔═╡ e9fb9a9a-73cd-49ee-ab9f-e864b2dbd8bf
@skip_as_script const gridworld_right_policy_q = monte_carlo_policy_prediction_q(deterministic_sample_gridworld, π_target_gridworld, 0.9f0, 1)

# ╔═╡ 9a7e922b-44e5-4c5e-8288-e39a48e151d5
function monte_carlo_control(mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, γ::T, num_episodes::Integer, initialize_episode::Function, update_policy!::Function; π = make_random_policy(mdp), q = initialize_state_action_value(mdp), counts = zeros(T, length(mdp.actions), length(mdp.states)), compare_error = false, value_reference = zeros(T, length(mdp.states)), averaging_method::AbstractAveragingMethod{T} = SampleAveraging(q), kwargs...) where {T<:Real, S, A, F<:Function, G<:Function, H<:Function}
	if compare_error
		error_history = zeros(T, num_episodes)
	end
	reward_history = zeros(T, num_episodes)
	step_history = zeros(Int64, num_episodes)
	for ep in 1:num_episodes
		(states, actions, rewards) = runepisode(mdp, π, initialize_episode(mdp)...; kwargs...)
		monte_carlo_episode_update!(q, states, actions, rewards, mdp, γ, averaging_method)
		for i_s in states
			update_policy!(π, mdp, q, i_s)
		end
		if compare_error
			error_history[ep] = sqrt(mean((value_reference[i] - sum(q[i_a, i]*π[i_a, i] for i_a in eachindex(mdp.actions))) ^2 for i in eachindex(value_reference)))
		end
		reward_history[ep] = sum(rewards)
		step_history[ep] = length(rewards)
	end
	make_greedy_policy!(π, mdp, q)
	basereturn = (optimal_policy_estimate = π, optimal_value_estimate = q, reward_history = reward_history, step_history = step_history)
	!compare_error && return basereturn
	return (;basereturn..., error_history = error_history)
end

# ╔═╡ b40f0a76-9405-46d0-aae2-8987b296766a
monte_carlo_control_exploring_starts(args...; kwargs...) = monte_carlo_control(args..., mdp -> (rand((eachindex(mdp.states))), rand(eachindex(mdp.actions))), make_greedy_policy!; kwargs...)

# ╔═╡ faa17fdd-9660-43ab-8f94-9cd1c3ba7fec
@skip_as_script const mc_control_sample_gridworld = monte_carlo_control_exploring_starts(deterministic_sample_gridworld, mc_control_γ, 10_000; compare_error = true, value_reference = last(value_iteration_grid_example2[1]), max_steps = 10_000)

# ╔═╡ fbfeb350-d9a7-4960-8f9b-a9f70e19a4e2
@skip_as_script plot(mc_control_sample_gridworld.error_history, Layout(xaxis_title = "Episodes", yaxis_title = "Mean Squared Error", title = "Optimal Value Function Error Decreases with Episodes <br> Using Monte Carlo Control with Exploring Starts"))

# ╔═╡ 66886194-a2bd-4b1e-9bff-fbb419fddc78
#the ϵ-soft method is defined by using the normal episode initialization from the mdp and using an ϵ-greedy policy update
monte_carlo_control_ϵ_soft(args...; ϵ = 0.1f0, kwargs...) = monte_carlo_control(args..., mdp -> (), (π, mdp, q, i_s) -> make_ϵ_greedy_policy!(π, mdp, q, ϵ, i_s); kwargs...)

# ╔═╡ b666c289-de0f-4412-a5f7-8e5bb546a47c
@skip_as_script const mc_ϵ_soft_control_sample_gridworld = monte_carlo_control_ϵ_soft(deterministic_sample_gridworld, mc_control_γ, 10_000; compare_error = true, value_reference = last(value_iteration_grid_example2[1]), max_steps = 10_000, ϵ = 0.1f0)

# ╔═╡ a6b08af6-34e8-4316-8f8c-b8e4b5fbb98a
@skip_as_script plot(mc_ϵ_soft_control_sample_gridworld.error_history, Layout(xaxis_title = "Episodes", yaxis_title = "Mean Squared Error", title = "Optimal Value Function Error Decreases with Episodes <br> Using Monte Carlo Control with ϵ Greedy Policy"))

# ╔═╡ 5648561c-98cf-4aa6-9af4-16add4706c3b
function monte_carlo_off_policy_prediction(mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, π_target::Matrix{T}, γ::T, num_episodes::Integer, initialize_value::Function; π_behavior = make_random_policy(mdp), sampling_method = WeightedImportanceSampling(), save_history = false, kwargs...) where {T<:Real,S, A, F<:Function, G<:Function, H<:Function}
	any(iszero, π_behavior) && error("Behavior policy is not soft")
	v_est = initialize_value(mdp) #default is 0 initialization
	weights = zeros(T, size(v_est)...)
	save_history &&	(value_history = zeros(T, size(v_est)..., num_episodes))

	for ep in 1:num_episodes
		(states, actions, rewards, _) = runepisode(mdp, π_behavior)
		monte_carlo_episode_update!((v_est, weights), states, actions, rewards, π_target, π_behavior, sampling_method, mdp, γ; kwargs...)
		save_history && update_value_history!(value_history, v_est, ep)
	end
	final_value_estimate = v_est
	if save_history
		return (final_value_estimate = final_value_estimate, value_estimate_history = value_history)
	else
		return v_est
	end
end

# ╔═╡ 5db8f67c-17fe-4c08-81df-42b47143b0ba
monte_carlo_off_policy_prediction_q(args...; kwargs...) = monte_carlo_off_policy_prediction(args..., initialize_state_action_value; kwargs...)

# ╔═╡ 84d1f707-3a72-49a5-bf11-62316f69232a
@skip_as_script function plot_off_policy_state_value(mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, π_target::Matrix{T}, γ::T, num_traces::Integer, num_samples::Integer, sample_method::AbstractSamplingMethod, num_episodes::Integer, s::S, a::A, v_true::T; kwargs...) where {T, S, A, F, G, H}
	t_true = scatter(x = 1:num_episodes, y = fill(v_true, num_episodes), line_dash = "dash", name = "true value")
	traces = 1:num_traces |> Map(_ -> scatter(x = 1:num_episodes, y = (1:num_samples |> Map(_ -> monte_carlo_off_policy_prediction_q(mdp, π_target, γ, num_episodes; sampling_method = sample_method, save_history = true).value_estimate_history[mdp.action_index[a], mdp.state_index[s], :]) |> foldxt(+) |> v -> v ./ num_samples), showlegend = false, name = "")) |> collect
	plot([t_true; traces], Layout(xaxis_title = "Episodes", yaxis_title = "Value Estimate", yaxis_range = [0f0, v_true*3]; kwargs...))
end

# ╔═╡ a2b62ae3-13d2-4d5b-a8ac-5c1c3c1ee246
@skip_as_script function off_policy_figure(x::Integer)
	s = GridworldState(x, 4)
	i_s = deterministic_sample_gridworld.state_index[s]
	v_true = dot(gridworld_right_policy_q[:, i_s], π_target_gridworld[:, i_s])
	@htl("""
	<div style="display: flex;">
	$(plot_off_policy_state_value(deterministic_sample_gridworld, π_target_gridworld, 0.9f0, 3, 20, OrdinaryImportanceSampling(), 100, s, Right(), v_true; title = "Ordinary Importance Sampling", showlegend = false))
	$(plot_off_policy_state_value(deterministic_sample_gridworld, π_target_gridworld, 0.9f0, 3, 20, WeightedImportanceSampling(), 100, s, Right(), v_true; title = "Weighted Importance Sampling", yaxis_title = false))
	</div>
	""")
end

# ╔═╡ b0d184ed-4129-49bf-afb7-7a848c93f15b
@skip_as_script off_policy_figure(x_off_policy_select)

# ╔═╡ 138fb7ec-bfd3-4798-8cbc-cb1c8982b799
function monte_carlo_off_policy_control(mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, γ::T, num_episodes::Integer; π_target = make_random_policy(mdp), π_behavior = make_random_policy(mdp), q = initialize_state_action_value(mdp), weights = zeros(T, length(mdp.actions), length(mdp.states)), compare_error = false, value_reference = zeros(T, length(mdp.states)), sampling_method = WeightedImportanceSampling(), kwargs...) where {T<:Real, S, A, F<:Function, G<:Function, H<:Function}
	if compare_error
		error_history = zeros(T, num_episodes)
	end
	for ep in 1:num_episodes
		(states, actions, rewards) = runepisode(mdp, π_behavior; kwargs...)
		monte_carlo_episode_update!((q, weights), states, actions, rewards, π_target, π_behavior, sampling_method, mdp, γ; kwargs...)
		for i_s in states
			make_greedy_policy!(π_target, mdp, q, i_s)
		end
		if compare_error
			error_history[ep] = sqrt(mean((value_reference[i] - sum(q[i_a, i]*π_target[i_a, i] for i_a in eachindex(mdp.actions))) ^2 for i in eachindex(value_reference)))
		end
	end
	make_greedy_policy!(π_target, mdp, q)
	basereturn = (optimal_policy_estimate = π_target, optimal_value_estimate = q)
	!compare_error && return basereturn
	return (;basereturn..., error_history = error_history)
end

# ╔═╡ d4435765-167c-433b-99ea-5cb9f1f3ac82
@skip_as_script off_policy_control_gridworld = monte_carlo_off_policy_control(deterministic_sample_gridworld, 0.9f0, 10_000)

# ╔═╡ 854fa686-914c-4a56-a975-486a542c0a9b
function episode_update!(v::Vector{T}, mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, π::Matrix{T}, γ::T, α::T) where {T<:Real, S, A, F<:Function, G<:Function, H<:Function}
i_s = mdp.state_init()
	while !mdp.isterm(i_s)
		(r, i_s′, i_a) = takestep(mdp, π, i_s)
		v[i_s] += α * (r + γ*v[i_s′] - v[i_s])
		i_s = i_s′
	end
end

# ╔═╡ 337b9905-9284-4bd7-a06b-f3e8bb44679c
function td0_policy_prediction(mdp::AbstractSampleTabularMDP{T, S, A, F, G, H}, π::Matrix{T}, γ::T, num_episodes::Integer, α::T, initialize_value_function::Function; save_value_history = false, kwargs...) where {T<:Real,S, A, F<:Function, G<:Function, H<:Function}
	value_estimate = initialize_value_function(mdp) #default is 0 initialization
	if save_value_history
		value_history = zeros(T, size(value_estimate)..., num_episodes)
	end
	
	for ep in 1:num_episodes
		episode_update!(value_estimate, mdp, π, γ, α)
		save_value_history && update_value_history!(value_history, value_estimate, ep)
	end
	final_v = value_estimate
	if save_value_history
		return (final_value_estimate = final_v, value_estimate_history = value_history)
	else
		return value_estimate
	end
end

# ╔═╡ ac7606f4-5986-4110-9acb-d7b089e9c98a
td0_policy_prediction_v(args...; kwargs...) = td0_policy_prediction(args..., initialize_state_value; kwargs...)

# ╔═╡ 034734a7-e7f0-4ea5-b252-5916f67c65d4
@skip_as_script td0v = td0_policy_prediction_v(deterministic_sample_gridworld, example_gridworld_random_policy, 0.9f0, 10_000, 0.01f0)

# ╔═╡ 6e2a99bc-7f49-4455-8b23-11392e47f24d
td0_policy_prediction_q(args...; kwargs...) = td0_policy_prediction(args..., initialize_state_action_value; kwargs...)

# ╔═╡ 749b5691-506f-4c7f-baa2-6d3e9b2607b9
@skip_as_script td0q = td0_policy_prediction_q(deterministic_sample_gridworld, example_gridworld_random_policy, 0.9f0, 10_000, 0.01f0)

# ╔═╡ b67ca69e-24ab-4893-90a2-85275b6fa306
begin
#note that for this type of MDP we can no longer refer to everything by its index and the policy must now be a function of the state rather than a matrix
function runepisode(mdp::SampleMDP{T, S, A, F, G, H}, π::Function, s0::S, a0::A; max_steps = Inf) where {T<:Real, S, A, F<:Function, G<:Function, H<:Function}
	s = s0
	a = a0
	states = Vector{S}()
	actions = Vector{A}()
	push!(states, s)
	(r, s′, a) = takestep(mdp, s, a)
	push!(actions, a)
	rewards = [r]
	step = 2
	sterm = s
	if mdp.isterm(s′)
		sterm = s′
	else
		sterm = s
	end
	s = s′

	#note that the terminal state will not be added to the state list
	while !mdp.isterm(s) && (step <= max_steps)
		push!(states, s)
		a = π(s)
		(r, s′, a) = takestep(mdp, s, a)
		push!(actions, a)
		push!(rewards, r)
		s = s′
		step += 1
		if mdp.isterm(s′)
			sterm = s′
		end
	end
	return states, actions, rewards, sterm
end

#define default behavior for leaving out the policy function (default to random policy), the initial state (use the state_init function), or the initial action (use the policy from the starting state)
runepisode(mdp::SampleMDP{T, S, A, F, G, H}, π::Function, s0::S; kwargs...) where {T<:Real, S, A, F<:Function G<:Function, H<:Function} = runepisode(mdp, π, s0, π(s0); kwargs...)
runepisode(mdp::SampleMDP{T, S, A, F, G, H}, s0::S; kwargs...) where {T<:Real, S, A, F<:Function G<:Function, H<:Function} = runepisode(mdp, make_random_policy(mdp), s0; kwargs...)
runepisode(mdp::SampleMDP; kwargs...) = runepisode(mdp, make_random_policy(mdp), mdp.state_init(); kwargs...)
end

# ╔═╡ d31b4e4f-18bf-4649-82f8-c603712bdbf0
runepisode(windy_mc_gridworld)

# ╔═╡ 66f6cad5-cc5c-4a81-86d1-fb893bc4fe12
begin
#perform a rollout with an mdp from state s using a policy function π that produces an action selection given a state input. return value is an unbiased estimate of the value of this state under the policy
function rollout(mdp::AbstractSampleMDP{T, S, A, G, H}, s::S, π::Function, γ::T; max_steps = Inf) where {T<:Real,S, A, G<:Function, H<:Function}
	step = 0
	g = zero(T)
	while !mdp.isterm(s) && (step <= max_steps)
		a = π(s)
		r, s′, a = takestep(mdp, s, a)
		g += γ^step * r
		s = s′
		step += 1
	end
	return g
end

#if no policy is provided then the rollout will use a uniformly random policy
rollout(mdp::AbstractSampleMDP{T, S, A, G, H}, s::S, γ::T; kwargs...) where {T<:Real, S, A, G<:Function, H<:Function} = rollout(mdp, s, make_random_policy(mdp), γ; kwargs...)

end

# ╔═╡ 970f3789-f830-47af-938f-0faf5f36421b
#rollout will estimate the state value using a policy calculating the discounted reward to termination
@skip_as_script rollout(windy_mc_gridworld, windy_mc_gridworld.state_init(), 0.99f0)

# ╔═╡ 63cbbd0e-388d-4f1f-8e11-95366bce37ba
#get the value of a dictionary d at key k, defaulting to some value when the key is absent
get_dict_value(d::Dict{K, V}, k::K; default = zero(V)) where {K, V<:Real} = haskey(d, k) ? d[k] : default 

# ╔═╡ 300577bd-22b7-4fdd-8a85-4768ecf985c4
monte_carlo_tree_search(mdp::AbstractMDP{T, S, A}, γ::T, v_est::Function, s::S; kwargs...) = monte_carlo_tree_search(SampleMDP(mdp), γ, v_est, s; kwargs...)

# ╔═╡ 482d1c2d-0898-48eb-b122-51e22d51a265
#need to decide which tree statistics to collect like state values or afterstate values and what expansion means vs normal mcts.  I know that when I visit a new afterstate which is the same as a new action selection, I want to estimate it with a weighted sum of the value estimates of all the sucessor states but I don't necessarily want the tree search to continue down all those paths and split although it could so a single simulation would split into all the successor states avoiding the need to make a selection.  For doing sample updates though, I want to just pick one of those branches to go down by sampling from the distribution so then the simulation function itself should handle the case of an unvisited state which would look at the afterstate values that lead from that state if any exist and well this is the problem is which values should be saved and what does it mean to estimate the value of something for one of the unvisited states
function simulate!(s::S, visit::Bool, tree_values::Dict{S, Tuple{T, Dict{Int64, Tuple{T, T}}}}, mdp::AfterstateMDP_MC{S, AS, A, F, G, H, I}, γ::T, v_est::Function, depth::Integer, c::T, v_hold, update_tree_policy!, update_tree!, q_hold, apply_bonus!, step_kwargs, transition_kwargs, est_kwargs) where {T<:Real, S, AS, A, F<:Function, G<:Function, H<:Function, I<:Function}
	#if the state is terminal, produce a value of 0
	mdp.isterm(s) && return zero(T)

	depth ≤ 0 && return v_est(mdp, s, γ; est_kwargs...)
	
	#for a state where no actions have been attempted, expand a new node
	if !haskey(tree_values, s)
		v = v_est(mdp, s, γ; est_kwargs...)
		tree_values[s] = (v, Dict{Int64, Tuple{T, T}}()) 
		return v
	end

	!visit && return max(tree_values[s][1], maximum(t[2]/t[1] for t in values(tree_values[s][2]); init = zero(T))) #if not visiting this state then just return the best value estimate and do not update the tree values

	#compute value estimates and bonus applies to each potential action
	apply_bonus!(v_hold, tree_values, s, c)
	update_tree_policy!(v_hold, s)

	#select an action from the tree policy
	i_a = sample_action(v_hold)
	a = mdp.actions[i_a]
	r1, w = mdp.afterstate_step(s, a; step_kwargs...) #take a step with the action and get the afterstate
	v_w = simulate!(w, tree_values, mdp, γ, v_est, depth, c, v_hold, update_tree_policy!, update_tree!, q_hold, apply_bonus!, step_kwargs, transition_kwargs, est_kwargs)
	v_a = r1 + v_w #value for the visited action
	update_tree!(tree_values, v_a, s, i_a)
	return max(tree_values[s][1], maximum(t[2]/t[1] for t in values(tree_values[s][2]); init = zero(T))) #the value that was just updated will be included in this maximum
end

# ╔═╡ 0b2e6a3c-caaa-4d79-9a3a-6b1d85037fb2
function simulate!(w::AS, tree_values::Dict{S, Tuple{T, Dict{Int64, Tuple{T, T}}}}, mdp::AfterstateMDP_MC{S, AS, A, F, G, H, I}, γ::T, v_est::Function, depth::Integer, c::T, v_hold, update_tree_policy!, update_tree!, q_hold, apply_bonus!, step_kwargs, transition_kwargs, est_kwargs) where {T<:Real, S, AS, A, F<:Function, G<:Function, H<:Function, I<:Function}
	dist = mdp.afterstate_transition(w; transition_kwargs...) #get the distribution of states following the transition
	k_sample = sample(collect(keys(dist)), weights(collect(values(dist)))) #sample one of the transition states to visit in the tree
	sum(begin
		(r, s) = k
		p = dist[k]
		v′ = simulate!(s, k == k_sample, tree_values, mdp, γ, v_est, depth - 1, c, v_hold, update_tree_policy!, update_tree!, q_hold, apply_bonus!, step_kwargs, transition_kwargs, est_kwargs)
		p * (r + γ * v′) 
	end
	for k in keys(dist))
end

# ╔═╡ 00e567e7-ab21-4f4a-aec1-b90e45f3db2a
function simulate!(visit_counts, Q, mdp::AbstractSampleMDP, γ::T, v_est, s, depth, c::T, v_hold, update_tree_policy!, updateQ!, updateV!, q_hold, apply_bonus!, step_kwargs, est_kwargs) where T<:Real
	#if the state is terminal, produce a value of 0
	mdp.isterm(s) && return zero(T)
	
	depth ≤ 0 && return v_est(mdp, s, γ; est_kwargs...)
	
	#for a state where no actions have been attempted, expand a new node
	if !haskey(visit_counts, s)
		# Q[s] = sparse(q_hold)
		# visit_counts[s] = sparse(q_hold)
		Q[s] = Dict{Int64, T}()
		visit_counts[s] = Dict{Int64, T}()
		return v_est(mdp, s, γ; est_kwargs...)
	end

	apply_bonus!(v_hold, Q, visit_counts, s, c)
	
	update_tree_policy!(v_hold, s)
	i_a = sample_action(v_hold)
	a = mdp.actions[i_a]
	r, s′ = mdp.step(s, a; step_kwargs...)
	q = r + γ*simulate!(visit_counts, Q, mdp, γ, v_est, s′, depth - 1, c, v_hold, update_tree_policy!, updateQ!, updateV!, q_hold, apply_bonus!, step_kwargs, est_kwargs)
	
	updateV!(visit_counts, one(T), s, i_a)

	# δq = (q - Q[s][i_a]) / visit_counts[s][i_a]
	δq = (q - get_dict_value(Q[s], i_a)) / visit_counts[s][i_a]
	updateQ!(Q, δq, s, i_a)
	return q
end

# ╔═╡ fa267730-d67d-4cd4-a9d5-901e79e553e5
uct(counts::Dict{S, Dict{Int64, T}}, s::S, i_a::Int64, ntot::T) where {S, T<:Real} = sqrt(log(ntot)/counts[s][i_a])

# ╔═╡ b062a7a6-4776-4db0-9712-1c832d7f271c
uct(tree_values::Dict{S, Tuple{T, Dict{Int64, Tuple{T, T}}}}, s::S, i_a::Int64, ntot::T) where {S, T<:Real} = sqrt(log(ntot)/tree_values[s][2][i_a][1])

# ╔═╡ 3f35548e-1bfc-4262-9534-ad4bc159bcf9
function apply_uct!(v_hold::Vector{T}, tree_values::Dict{S, Tuple{T, Dict{Int64, Tuple{T, T}}}}, s::S, c::T) where {S, T<:Real}
	#for normal UCB selection, unvisited states have an infinite bonus
	v_hold .= T(Inf)

	d = tree_values[s][2]
	isempty(d) && return v_hold
	ntot = sum(t[1] for t in values(d))
	@inbounds @fastmath for i in keys(d)
		#note that the only bonus values computed here are for actions that have been visited
		v_hold[i] = (d[i][2] / d[i][1]) + c * uct(tree_values, s, i, ntot)
	end
	return v_hold
end

# ╔═╡ 78eda243-db35-4eb4-8e97-e845dd3da064
#perform action selection within an mdp for a given state s, discount factor γ, and state value estimation function v_est.  v_est must be a function that takes the arguments (mdp, s, γ) and produces a reward of the same type as γ
function monte_carlo_tree_search(mdp::SampleMDP{T, S, A, F, G, H}, γ::T, v_est::Function, s::S; 
	depth = 10, 
	nsims = 100, 
	c = one(T), 
	# visit_counts = Dict{S, SparseVector{T, Int64}}(), 
	visit_counts = Dict{S, Dict{Int64, T}}(),
	# Q = Dict{S, SparseVector{T, Int64}}(),
	Q = Dict{S, Dict{Int64, T}}(),
	update_tree_policy! = (v, s) -> make_greedy_policy!(v), 
	v_hold = zeros(T, length(mdp.actions)),
	updateQ! = function(Q, x, s, i_a)
		# Q[s][i_a] += x
		Q[s][i_a] = get_dict_value(Q[s], i_a) + x
	end,
	updateV! = function(V, x, s, i_a)
		# V[s][i_a] += x
		V[s][i_a] = get_dict_value(V[s], i_a) + x
	end,
	apply_bonus! = apply_uct!,
	make_step_kwargs = k -> NamedTuple(), #option to create mdp step arguments that depend on the simulation number, 
	make_est_kwargs = k -> NamedTuple(), #option to create state estimation arguments that depend on the simulation number
	sim_message = false
	) where {T<:Real, S, A, F, G, H}

	q_hold = zeros(T, length(mdp.actions))
	#I want to have a way of possible a kwargs such as the answer index to the simulator that can change with each simulation
	t = time()
	last_time = t
	for k in 1:nsims
		seed = rand(UInt64)
		if sim_message
			elapsed = time() - last_time
			if elapsed > 5
				last_time = time()
				pct_done = k/nsims
				total_time = time() - t
				ett = total_time / pct_done
				eta = ett - total_time
				@info """Completed simulation $k of $nsims after $(round(Int64, total_time/60)) minutes
				ETA: $(round(Int64, eta/60)) minutes"""
			end
		end
		simulate!(visit_counts, Q, mdp, γ, v_est, s, depth, c, v_hold, update_tree_policy!, updateQ!, updateV!, q_hold, apply_bonus!, make_step_kwargs(seed), make_est_kwargs(seed))
	end

	minv = minimum(Q[s][k] for k in keys(Q[s]))
	# v_hold .= Q[s]
	for i in eachindex(v_hold)
		v_hold[i] = get_dict_value(Q[s], i; default = minv)
	end
	make_greedy_policy!(v_hold)
	if sim_message
		@info "Finished MCTS evaluation of state $s"
	end
	return mdp.actions[sample_action(v_hold)], visit_counts, Q
end

# ╔═╡ b056168b-1f10-4046-9a0c-dbe89a713d6a
#perform action selection within an mdp for a given state s, discount factor γ, and state value estimation function v_est.  v_est must be a function that takes the arguments (mdp, s, γ) and produces a reward of the same type as γ
function monte_carlo_tree_search(mdp::AbstractAfterstateMDP{T, S, AS, A}, γ::T, v_est::Function, s::S; 
	depth = 10, 
	nsims = 100, 
	c = one(T), 
	tree_values = Dict{S, Tuple{T, Dict{Int64, Tuple{T, T}}}}(),
	update_tree_policy! = (v, s) -> make_greedy_policy!(v), 
	v_hold = zeros(T, length(mdp.actions)),
	update_tree! = function(tree_values, v::T, s::S, i_a::Integer)
		d = tree_values[s][2]
		new_value = if haskey(d, i_a)
			(d[i_a][1]+1, d[i_a][2]+v)
		else
			(1f0, v)
		end
		tree_values[s][2][i_a] = new_value
	end,
	apply_bonus! = apply_uct!,
	make_step_kwargs = k -> NamedTuple(), #option to create mdp afterstate step arguments that depend on the simulation number
	make_transition_kwargs = k -> NamedTuple(), #option to create mdp afterstate transition arguments that depend on the simulation number
	make_est_kwargs = k -> NamedTuple(), #option to create state estimation arguments that depend on the simulation number
	sim_message = false
	) where {S, AS, A, F, G, H, I, T<:Real}

	q_hold = zeros(T, length(mdp.actions))
	#I want to have a way of possible a kwargs such as the answer index to the simulator that can change with each simulation
	t = time()
	last_time = t
	for k in 1:nsims
		seed = rand(UInt64)
		if sim_message
			elapsed = time() - last_time
			if elapsed > 5
				last_time = time()
				pct_done = k/nsims
				total_time = time() - t
				ett = total_time / pct_done
				eta = ett - total_time
				@info """Completed simulation $k of $nsims after $(round(Int64, total_time/60)) minutes
				ETA: $(round(Int64, eta/60)) minutes"""
			end
		end
		simulate!(s, true, tree_values, mdp, γ, v_est, depth, c, v_hold, update_tree_policy!, update_tree!, q_hold, apply_bonus!, make_step_kwargs(seed), make_transition_kwargs(seed), make_est_kwargs(seed))
	end

	minv = minimum(t[2]/t[1] for t in values(tree_values[s][2]))
	for i in eachindex(v_hold)
		if haskey(tree_values[s][2], i)
			v_hold[i] = tree_values[s][2][i][2] / tree_values[s][2][i][1]
		else
			v_hold[i] = minv
		end
	end
	make_greedy_policy!(v_hold)
	if sim_message
		@info "Finished MCTS evaluation of state $s"
	end
	return mdp.actions[sample_action(v_hold)], tree_values
end

# ╔═╡ 4e906d8c-ca74-42e3-a9e3-b3980206fbe3
@skip_as_script md"""### *Example: Gridworld MCTS*"""

# ╔═╡ 3e4fc9d3-1d87-431b-b348-09e7567149f0
@skip_as_script monte_carlo_tree_search(dyna_maze, 0.95f0, rollout(;max_steps = 10_000),  dyna_maze.state_init(); nsims = 100, depth = 10, c = 1f0)

# ╔═╡ 796eeb6c-1152-11ef-00b7-b543ec85b526
@skip_as_script md"""# Dependencies"""

# ╔═╡ 7b4e1a9b-ef0b-41f6-a634-99af17a02f60
@skip_as_script html"""
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

# ╔═╡ 32c92099-f322-4086-983d-50b79ab28de8
@skip_as_script md"""
## Visualization Tools
"""

# ╔═╡ afaac0aa-d0e2-4e2c-a5ed-08b89b901541
@skip_as_script function addelements(e1, e2)
	@htl("""
	$e1
	$e2
	""")
end

# ╔═╡ a40d6dd3-1f8b-476a-9839-1bd1ae46751a
@skip_as_script show_grid_value(mdp::AbstractCompleteMDP{T, S, A}, isterm::Function, state_init::Function, Q, name; kwargs...) where {T<:Real, S, A} = show_grid_value(mdp.states, isterm, state_init, Q, name; kwargs...)

# ╔═╡ 31a3bb9e-4ef3-4876-87c2-12d462e60eab
@skip_as_script show_grid_value(mdp::AbstractSampleTabularMDP, Q, name; kwargs...) = show_grid_value(mdp.states, mdp.isterm, mdp.state_init, Q, name; kwargs...)

# ╔═╡ 4bfdde5d-857f-4955-809d-f4a21440000e
@skip_as_script HTML("""
<style>
	.windcell {
		display: flex;
		justify-content: center;
		align-items: center;
		border: 0px rgba(0, 0, 0, 0);
		color: black;
		background-color: white;
	}
</style>
""")

# ╔═╡ 7ad8dc82-5c60-493a-b78f-93e37a3f3ab8
@skip_as_script function show_grid_value(states, isterm, state_init, Q, name; scale = 1.0, title = "", sigdigits = 2, square_pixels = 20, highlight_state_index = 0)
	width = maximum(s.x for s in states)
	height = maximum(s.y for s in states)
	start = states[state_init()]
	terminds = findall(isterm, eachindex(states))
	sterms = states[terminds]
	ngrid = width*height

	displayvalue(Q::Matrix, i) = round(maximum(Q[:, i]), sigdigits = sigdigits)
	displayvalue(V::Vector, i) = round(V[i], sigdigits = sigdigits)

	highlight_style = if iszero(highlight_state_index)
		@htl("""""")
	else
		@htl("""
		.$name.value[x="$(states[highlight_state_index].x)"][y="$(states[highlight_state_index].y)"] {
			border: 3px solid black;
		}
		""")
	end
	@htl("""
		<div style = "display: flex; transform: scale($scale); background-color: white; color: black; font-size: 16px; justify-content: center;">
			<div>
				$title
				<div class = "gridworld $name value">
					$(HTML(mapreduce(i -> """<div class = "gridcell $name value" x = "$(states[i].x)" y = "$(states[i].y)" style = "grid-row: $(height - states[i].y + 1); grid-column: $(states[i].x); font-size: 12px; color: black;">$(displayvalue(Q, i))</div>""", *, eachindex(states))))
				</div>
			</div>
		</div>
	
		<style>
			.$name.value.gridworld {
				display: grid;
				grid-template-columns: repeat($width, $(square_pixels)px);
				grid-template-rows: repeat($height, $(square_pixels)px);
				background-color: white;
			}

			.$name.value[x="$(start.x)"][y="$(start.y)"] {
				content: '';
				background-color: rgba(0, 255, 0, 0.5);
			}

			$(mapreduce(addelements, sterms) do sterm
				@htl("""
				.$name.value[x="$(sterm.x)"][y="$(sterm.y)"] {
					content: '';
					background-color: rgba(255, 215, 0, 0.5);
				}
				""")
			end)

			$highlight_style
			
		</style>
	""")
end

# ╔═╡ bfef62c9-4186-4b01-afe2-e49432f04265
@skip_as_script show_grid_value(deterministic_gridworld.mdp, deterministic_gridworld.isterm, () -> deterministic_gridworld.init_state, deterministic_gridworld_random_policy_evaluation.value_function, "gridworld_random_values"; square_pixels = 50)

# ╔═╡ 900a2ece-9638-49fc-afbe-e012f9520b48
@skip_as_script show_grid_value(windy_gridworld.mdp, windy_gridworld.isterm, () -> windy_gridworld.init_state, windy_gridworld_random_policy_evaluation.value_function, "gridworld_random_values"; square_pixels = 50)

# ╔═╡ 5b53ef57-12d1-45e2-ad1e-28c490c336a6
@skip_as_script show_grid_value(stochastic_gridworld.mdp, stochastic_gridworld.isterm, () -> stochastic_gridworld.init_state, stochastic_gridworld_random_policy_evaluation.value_function, "gridworld_random_values"; square_pixels = 50)

# ╔═╡ e18328c7-837e-427c-b5c0-3be31dcb4c4b
@skip_as_script show_grid_value(deterministic_sample_gridworld, td0v, "td0_v_test"; square_pixels = 40)

# ╔═╡ 35df802b-41e6-4ede-8200-5658e3ee1328
@skip_as_script show_grid_value(deterministic_sample_gridworld, td0q, "td0_q_test"; square_pixels = 40)

# ╔═╡ 9b937c49-7216-47c9-a1ef-2ecfa6ff3b31
@skip_as_script function display_rook_policy(v::Vector{T}; scale = 1.0) where T<:AbstractFloat
	@htl("""
		<div style = "display: flex; align-items: center; justify-content: center; transform: scale($scale);">
		<div class = "downarrow" style = "position: absolute; transform: rotate(180deg); opacity: $(v[1]);"></div>	
		<div class = "downarrow" style = "position: absolute; opacity: $(v[2])"></div>
		<div class = "downarrow" style = "position: absolute; transform: rotate(90deg); opacity: $(v[3])"></div>
		<div class = "downarrow" style = "transform: rotate(-90deg); opacity: $(v[4])"></div>
		</div>
	""")
end

# ╔═╡ d5431c0e-ac46-4de1-8d3c-8c97b92306a8
@skip_as_script function show_selected_action(i)
	v = zeros(4)
	v[i] = 1
	display_rook_policy(v)
end

# ╔═╡ 93cbc453-152e-401e-bf53-c95f1ae962c0
@skip_as_script const rook_action_display = @htl("""
<div style = "display: flex; flex-direction: column; align-items: center; justify-content: center; color: black; background-color: rgba(100, 100, 100, 0.1);">
	<div style = "display: flex; align-items: center; justify-content: center;">
	<div class = "downarrow" style = "transform: rotate(90deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(180deg);"></div>
	<div class = "downarrow" style = "position: absolute; transform: rotate(270deg);"></div>
	<div class = "downarrow" style = "position: absolute;"></div>
	</div>
	<div>Actions</div>
</div>
""")

# ╔═╡ 5ab5f9d5-b60a-4556-a8c7-47c808e5d4f8
@skip_as_script function show_grid_transitions(states, isterm, state_init, name; scale = 1.0, title = "", action_display = rook_action_display, highlight_state = GridworldState(1, 1), transition_states::Dict{GridworldState, Float32} = Dict([GridworldState(1, 2) => 1f0]), reward_values = [(p = 1f0, r = 0f0)], width = maximum(s.x for s in states), wind = zeros(Int64, width), square_pixels = 30)
	height = maximum(s.y for s in states)
	start = states[state_init()]
	terminds = findall(isterm, eachindex(states))
	sterms = states[terminds]
	ngrid = width*height

	@htl("""
		<div style = "background-color: white; color: black;">
		Selected Action with Reward Distribution: $reward_values
		$action_display
		State Transitions
		<div style = "display: flex; transform: scale($scale); background-color: white; color: black; font-size: 16px; justify-content: center;">
			<div>
				$title
				<div class = "gridworld $name value">
					$(HTML(mapreduce(i -> """<div class = "gridcell $name value" x = "$(states[i].x)" y = "$(states[i].y)" style = "grid-row: $(height - states[i].y + 1); grid-column: $(states[i].x); font-size: 12px; color: black;"></div>""", *, eachindex(states))))
					$(HTML(mapreduce(i -> """<div class = "windcell $name" style = "grid-row: 0; grid-column: $i; font-size: 12px;">$(wind[i])</div>""", *, 1:width)))
					Wind Values
				</div>
			</div>
		</div>
		</div>
	
		<style>
			.$name.value.gridworld {
				display: grid;
				grid-template-columns: repeat($width, $(square_pixels)px);
				grid-template-rows: repeat($height, $(square_pixels)px);
				background-color: white;
				margin: 20px;
			}

			.$name.value[x="$(start.x)"][y="$(start.y)"] {
				background-color: rgba(0, 255, 0, 0.5);
				
			}

			.$name.value[x="$(highlight_state.x)"][y="$(highlight_state.y)"] {
				background-color: rgba(0, 0, 255, 0.5);
			}


			$(mapreduce(addelements, transition_states) do transition_state
				@htl("""
				.$name.value[x="$(transition_state[1].x)"][y="$(transition_state[1].y)"] {
					border: 4px solid black;
				}
				.$name.value[x="$(transition_state[1].x)"][y="$(transition_state[1].y)"]::before {
					content: '$(round(transition_state[2] |> Float64, sigdigits = 2))';
				}
				""")
			end)

			$(mapreduce(addelements, sterms) do sterm
				@htl("""
				.$name.value[x="$(sterm.x)"][y="$(sterm.y)"] {
					background-color: rgba(255, 215, 0, 0.5);
				}
				""")
			end)
			
		</style>
	""")
end

# ╔═╡ fed249aa-2d0a-4bc3-84ea-e3ad4b4e66fa
@skip_as_script function show_deterministic_gridworld(mdp::FiniteDeterministicMDP, isterm, init_state, highlight_state_index, grid_action_selection; name = "deterministic_gridworld_transitions", kwargs...)
	s = mdp.states[highlight_state_index]
	s′ = mdp.states[mdp.state_transition_map[grid_action_selection, highlight_state_index]]
	r = mdp.reward_transition_map[grid_action_selection, highlight_state_index]
	show_grid_transitions(mdp.states, isterm, () -> init_state, name; highlight_state = s, transition_states = Dict([s′ => 1f0]), action_display = show_selected_action(grid_action_selection), reward_values = [(p = 1, r = r |> Float64)], kwargs...)
end

# ╔═╡ 5994f7fd-ecd1-4c2b-8000-5eaa03262a63
@skip_as_script  show_deterministic_gridworld(windy_gridworld.mdp, windy_gridworld.isterm, windy_gridworld.init_state, highlight_state_index, grid_action_selection; wind = wind_values)

# ╔═╡ 97660b1c-e09c-4e52-a88c-55522141a39b
@skip_as_script function show_stochastic_gridworld(mdp::FiniteStochasticMDP, isterm, init_state, highlight_state_index, grid_action_selection; name = "stochastic_gridworld_transitions", kwargs...)
	s = mdp.states[highlight_state_index]
	output = mdp.ptf[highlight_state_index, grid_action_selection]
	show_grid_transitions(mdp.states, isterm, () -> init_state, name; highlight_state = s, transition_states = Dict(mdp.states[i_s′] => output[i_s′][1] for i_s′ in keys(output)), action_display = show_selected_action(grid_action_selection), reward_values = [(p = round(output[i_s′][1] |> Float64, sigdigits = 2), r = output[i_s′][2] |> Float64) for i_s′ in keys(output)], kwargs...)
end

# ╔═╡ 8e53fb6e-db4b-48e7-8cca-db3e6f16a3c3
@skip_as_script  show_stochastic_gridworld(stochastic_gridworld.mdp, stochastic_gridworld.isterm, stochastic_gridworld.init_state, highlight_state_index, grid_action_selection; wind = wind_values)

# ╔═╡ b70ec2b1-f8c2-4288-831a-041804d2ec43
@skip_as_script function show_grid_policy(states, state_init, isterm, π, name; display_function = display_rook_policy, action_display = rook_action_display, scale = 1.0)
	width = maximum(s.x for s in states)
	height = maximum(s.y for s in states)
	start = states[state_init()]
	termind = findfirst(isterm, eachindex(states))
	sterm = states[termind]
	ngrid = width*height
	@htl("""
		<div style = "display: flex; transform: scale($scale); background-color: white;">
			<div>
				<div class = "gridworld $name">
					$(HTML(mapreduce(i -> """<div class = "gridcell $name" x = "$(states[i].x)" y = "$(states[i].y)" style = "grid-row: $(height - states[i].y + 1); grid-column: $(states[i].x);">$(display_function(π[:, i], scale =0.8))</div>""", *, eachindex(states))))
				</div>
			</div>
			<div style = "display: flex; flex-direction: column; align-items: flex-start; justify-content: flex-end; color: black; font-size: 18px; width: 5em; margin-left: 1em;">
				$(action_display)
			</div>
		</div>
	
		<style>
			.$name.gridworld {
				display: grid;
				grid-template-columns: repeat($width, 40px);
				grid-template-rows: repeat($height, 40px);
				background-color: white;

			.$name[x="$(start.x)"][y="$(start.y)"]::before {
				content: 'S';
				position: absolute;
				color: green;
				opacity: 1.0;
			}

			.$name[x="$(sterm.x)"][y="$(sterm.y)"]::before {
				content: 'G';
				position: absolute;
				color: red;
				opacity: 1.0;
			}

		</style>
	""")
end

# ╔═╡ e30d2af4-b6e7-46fb-ad72-4672caa81de4
@skip_as_script show_grid_policy((deterministic_gridworld.mdp).states, () -> deterministic_gridworld.init_state, deterministic_gridworld.isterm, make_random_policy(deterministic_gridworld.mdp), "random_policy_deterministic_gridworld")

# ╔═╡ 2e4bdce5-6188-4c22-a56b-7051c63aa165
@skip_as_script @htl("""
<div style = "display: flex; justify-content: center; align-items: flex-start;">
	<div>Policy after Iteration $policy_iteration_count$(show_grid_policy((new_gridworld.mdp).states, () -> new_gridworld.init_state, new_gridworld.isterm, π_list[policy_iteration_count+1], "policy_iteration_deterministic_gridworld"))</div>
	<div>Corresponding Value Function$(show_grid_value(new_gridworld.mdp, new_gridworld.isterm, () -> new_gridworld.init_state, v_list[policy_iteration_count+1], "policy_iteration_values", square_pixels = 40))</div>
</div>
""")

# ╔═╡ 102d169a-8bd0-42f4-bfc9-3a32708afadc
@skip_as_script @htl("""
<div style = "display: flex; justify-content: center; align-items: flex-start;">
	<div style = "margin: 10px;">Optimal value function found after $(length(value_iteration_grid_example[1]) - 1) steps $(show_grid_value(new_gridworld.mdp, new_gridworld.isterm, () -> new_gridworld.init_state, last(value_iteration_grid_example[1]), "policy_iteration_values", square_pixels = 40))</div>
	<div style = "margin: 10px;">Corresponding greedy policy
	$(show_grid_policy((new_gridworld.mdp).states, () -> new_gridworld.init_state, new_gridworld.isterm, value_iteration_grid_example[2], "policy_iteration_deterministic_gridworld"))</div>
</div>
""")

# ╔═╡ 3cc38ba2-70ce-4250-be97-0a48c2c2b484
@skip_as_script @htl("""
<div style = "display: flex; justify-content: center; align-items: flex-start;">
	<div style = "margin: 10px;">Learned optimal value function found after 10,000 episodes $(show_grid_value(deterministic_sample_gridworld.states, deterministic_sample_gridworld.isterm, deterministic_sample_gridworld.state_init, sum(mc_control_sample_gridworld.optimal_policy_estimate .* mc_control_sample_gridworld.optimal_value_estimate, dims = 1), "mc_control_values", square_pixels = 40))</div>
	<div style = "margin: 10px;">Corresponding greedy policy
	$(show_grid_policy(deterministic_sample_gridworld.states, deterministic_sample_gridworld.state_init, deterministic_sample_gridworld.isterm, mc_control_sample_gridworld.optimal_policy_estimate, "mc_control_optimal_policy_gridworld"))</div>
</div>
""")

# ╔═╡ 97234d16-1455-4321-bb16-c09534a58594
@skip_as_script @htl("""
<div style = "display: flex; justify-content: center; align-items: flex-start;">
	<div style = "margin: 10px;">Learned optimal value function found after 10,000 episodes $(show_grid_value(deterministic_sample_gridworld.states, deterministic_sample_gridworld.isterm, deterministic_sample_gridworld.state_init, sum(mc_ϵ_soft_control_sample_gridworld.optimal_policy_estimate .* mc_ϵ_soft_control_sample_gridworld.optimal_value_estimate, dims = 1), "mc_control_values", square_pixels = 40))</div>
	<div style = "margin: 10px;">Corresponding greedy policy
	$(show_grid_policy(deterministic_sample_gridworld.states, deterministic_sample_gridworld.state_init, deterministic_sample_gridworld.isterm, mc_ϵ_soft_control_sample_gridworld.optimal_policy_estimate, "mc_control_optimal_policy_gridworld"))</div>
</div>
""")

# ╔═╡ 9c7c571e-ef14-4fe2-b3a9-aa66131226f8
@skip_as_script @htl("""
<div style = "display: flex; justify-content: center; align-items: flex-start;">
	<div style = "margin: 10px;">Values for Right Policy $(show_grid_value(deterministic_sample_gridworld, gridworld_right_policy_q, "gridworld_right_values", square_pixels = 40, highlight_state_index = deterministic_sample_gridworld.state_index[GridworldState(x_off_policy_select, 4)]))</div>
	<div style = "margin: 10px;">Right Target Policy
	$(show_grid_policy(deterministic_sample_gridworld.states, deterministic_sample_gridworld.state_init, deterministic_sample_gridworld.isterm, π_target_gridworld, "right_policy_gridworld"))</div>
</div>
""")

# ╔═╡ 0ad54e4b-ea9d-418c-bb6a-cd8fbe241c73
@skip_as_script @htl("""
<div style = "display: flex; justify-content: center; align-items: flex-start;">
	<div style = "margin: 10px;">Learned optimal value function found after 10,000 episodes $(show_grid_value(deterministic_sample_gridworld, sum(off_policy_control_gridworld.optimal_value_estimate .* off_policy_control_gridworld.optimal_policy_estimate, dims = 1), "mc_off_policy_control_values", square_pixels = 40))</div>
	<div style = "margin: 10px;">Corresponding greedy policy
	$(show_grid_policy(deterministic_sample_gridworld.states, deterministic_sample_gridworld.state_init, deterministic_sample_gridworld.isterm, off_policy_control_gridworld.optimal_policy_estimate, "mc_off_policy_control_optimal_policy_gridworld"))</div>
</div>
""")

# ╔═╡ 64d2a0e3-4ecd-4d44-b5cc-0ff23b3776dd
@skip_as_script @htl("""
<div style = "display: flex; justify-content: center; align-items: flex-start;">
	<div style = "margin: 10px;">Learned optimal value function found after 10,000 episodes $(show_grid_value(deterministic_sample_gridworld, sum(sarsa_test.final_value_estimate .* sarsa_test.final_policy, dims = 1), "sarsa_grid_world_values", square_pixels = 40))</div>
	<div style = "margin: 10px;">Corresponding greedy policy
	$(show_grid_policy(deterministic_sample_gridworld.states, deterministic_sample_gridworld.state_init, deterministic_sample_gridworld.isterm, sarsa_test.final_policy, "sarsa_optimal_policy_gridworld"))</div>
</div>
""")

# ╔═╡ acd5ff5b-f9d0-41bf-ae09-cf6842eab556
@skip_as_script @htl("""
<div style = "display: flex; justify-content: center; align-items: flex-start;">
	<div style = "margin: 10px;">Learned optimal value function found after 10,000 episodes $(show_grid_value(deterministic_sample_gridworld, sum(expected_sarsa_test.final_value_estimate .* expected_sarsa_test.final_policy, dims = 1), "sarsa_grid_world_values", square_pixels = 40))</div>
	<div style = "margin: 10px;">Corresponding greedy policy
	$(show_grid_policy(deterministic_sample_gridworld.states, deterministic_sample_gridworld.state_init, deterministic_sample_gridworld.isterm, expected_sarsa_test.final_policy, "sarsa_optimal_policy_gridworld"))</div>
</div>
""")

# ╔═╡ d32191cf-1d96-495a-bf04-f0bc5a5ecaa0
@skip_as_script @htl("""
<div style = "display: flex; justify-content: center; align-items: flex-start;">
	<div style = "margin: 10px;">Learned optimal value function found after 10,000 episodes $(show_grid_value(deterministic_sample_gridworld, sum(double_expected_sarsa_test.final_value_estimate .* expected_sarsa_test.final_policy, dims = 1), "sarsa_grid_world_values", square_pixels = 40))</div>
	<div style = "margin: 10px;">Corresponding greedy policy
	$(show_grid_policy(deterministic_sample_gridworld.states, deterministic_sample_gridworld.state_init, deterministic_sample_gridworld.isterm, double_expected_sarsa_test.final_policy, "sarsa_optimal_policy_gridworld"))</div>
</div>
""")

# ╔═╡ c11ab768-1da4-497b-afc1-fb64bc3fb457
@skip_as_script HTML("""
<style>
	.downarrow {
		display: flex;
		justify-content: center;
		align-items: center;
		flex-direction: column;
	}

	.downarrow::before {
		content: '';
		width: 2px;
		height: 40px;
		background-color: black;
	}
	.downarrow::after {
		content: '';
		width: 0px;
		height: 0px;
		border-left: 5px solid transparent;
		border-right: 5px solid transparent;
		border-top: 10px solid black;
	}

	.gridcell {
			display: flex;
			justify-content: center;
			align-items: center;
			border: 1px solid black;
		}

	.windbox {
		height: 40px;
		width: 40px;
		display: flex;
		justify-content: center;
		align-items: center;
		transform: rotate(180deg);
		background-color: green;
	}

	.windbox * {
		background-color: green;
		color: green;
	}

	.windbox[w="0"] {
		opacity: 0.0; 
	}

	.windbox[w="1"] {
		opacity: 0.5;
	}

	.windbox[w="2"] {
		opacity: 1.0;
	}
</style>
""")

# ╔═╡ 3279ba47-18a1-45a9-9d29-18b9875ed057
@skip_as_script function plot_path(episode_states::Vector{Int64}, i_sterm::Integer, gridworld_states::Vector{S}, i_s0::Integer, isterm::Function; title = "Policy <br> path example", iscliff = s -> false, iswall = s -> false, pathname = "Policy Path") where S <: GridworldState
	xmax = maximum([s.x for s in gridworld_states])
	ymax = maximum([s.y for s in gridworld_states])
	start = gridworld_states[i_s0]
	goal = gridworld_states[findlast(isterm(i_s) for i_s in eachindex(gridworld_states))]
	start_trace = scatter(x = [start.x + 0.5], y = [start.y + 0.5], mode = "text", text = ["S"], textposition = "left", showlegend=false)
	finish_trace = scatter(x = [goal.x + .5], y = [goal.y + .5], mode = "text", text = ["G"], textposition = "left", showlegend=false)
	
	path_traces = [scatter(x = [gridworld_states[episode_states[i]].x + 0.5, gridworld_states[episode_states[i+1]].x + 0.5], y = [gridworld_states[episode_states[i]].y + 0.5, gridworld_states[episode_states[i+1]].y + 0.5], line_color = "blue", mode = "lines", showlegend=false, name = pathname) for i in 1:length(episode_states)-1]
	finalpath = scatter(x = [gridworld_states[episode_states[end]].x + 0.5, gridworld_states[i_sterm].x + .5], y = [gridworld_states[episode_states[end]].y + 0.5, gridworld_states[i_sterm].y + 0.5], line_color = "blue", mode = "lines", showlegend=false, name = pathname)

	h1 = 30*ymax
	traces = [start_trace; finish_trace; path_traces; finalpath]

	cliff_squares = filter(iscliff, gridworld_states)
	for s in cliff_squares
		push!(traces, scatter(x = [s.x + 0.6], y = [s.y+0.5], mode = "text", text = ["C"], textposition = "left", showlegend = false))
	end


	wall_squares = filter(iswall, gridworld_states)
	for s in wall_squares
		push!(traces, scatter(x = [s.x + 0.8], y = [s.y+0.5], mode = "text", text = ["W"], textposition = "left", showlegend = false))
	end

	plot(traces, Layout(xaxis = attr(showgrid = true, showline = true, gridwith = 1, gridcolor = "black", zeroline = true, linecolor = "black", mirror=true, tickvals = 1:xmax, ticktext = fill("", 10), range = [1, xmax+1]), yaxis = attr(linecolor="black", mirror = true, gridcolor = "black", showgrid = true, gridwidth = 1, showline = true, tickvals = 1:ymax, ticktext = fill("", ymax), range = [1, ymax+1]), width = max(30*xmax, 200), height = max(h1, 200), autosize = false, padding=0, paper_bgcolor = "rgba(0, 0, 0, 0)", title = attr(text = title, font_size = 14, x = 0.5)))
end

# ╔═╡ cbeac89a-845c-4409-8067-8766fe3b8a24
@skip_as_script function plot_path(mdp::AbstractCompleteMDP, i_s0, isterm, π; max_steps = 100, kwargs...)
	(states, actions, rewards, sterm) = runepisode(mdp, i_s0, isterm, π; max_steps = max_steps)
	plot_path(states, sterm, mdp.states, i_s0, isterm; kwargs...)
end

# ╔═╡ 4f193af4-9925-4047-92f9-c67eec1f4c97
@skip_as_script plot_path(mdp::AbstractCompleteMDP, i_s0, isterm; title = "Random policy <br> path example", kwargs...) = plot_path(mdp, i_s0, isterm, make_random_policy(mdp); title = title, kwargs...)

# ╔═╡ 3a707040-a763-42f6-9f5c-8c56a5f869f7
@skip_as_script plot_path(deterministic_gridworld.mdp, deterministic_gridworld.init_state, deterministic_gridworld.isterm)

# ╔═╡ 2a2d1b60-be6f-4f9c-8190-7c0a2d77d510
@skip_as_script function show_mcts_solution()
	# visit_counts = Dict{GridworldState, SparseVector{Float32, Int64}}()
	visit_counts = Dict{GridworldState, Dict{Int64, Float32}}()
	# Q = Dict{GridworldState, SparseVector{Float32, Int64}}()
	Q = Dict{GridworldState, Dict{Int64, Float32}}()
	plot_path(dyna_maze, s -> monte_carlo_tree_search(dyna_maze, 0.95f0, rollout(;max_steps = 100),  s; nsims = 100, depth = 100, c = 1.0f0, visit_counts = visit_counts, Q = Q)[1]; max_steps = 1000, iswall = s -> in(s, maze_walls))
end

# ╔═╡ 8ffd78db-cfc5-4695-a1c1-6a4e6aa32348
@skip_as_script show_mcts_solution()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Transducers = "28d57a85-8fef-5791-bfe6-a80928e7c999"

[compat]
DataStructures = "~0.18.20"
HypertextLiteral = "~0.9.5"
PlutoHooks = "~0.0.5"
PlutoPlotly = "~0.4.6"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.59"
StaticArrays = "~1.9.3"
StatsBase = "~0.34.3"
Transducers = "~0.4.82"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "dd0a610c9ee8d847c62f16b793c49b11175a0e36"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown", "Test"]
git-tree-sha1 = "c0d491ef0b135fd7d63cbc6404286bc633329425"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.36"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BangBang]]
deps = ["Accessors", "Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires"]
git-tree-sha1 = "08e5fc6620a8d83534bf6149795054f1b1e8370a"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.2"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BaseDirs]]
git-tree-sha1 = "3e93fcd95fe8db4704e98dbda14453a0bfc6f6c3"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.2.3"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "4b270d6465eb21ae89b732182c20dc165f8bf9f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.25.0"

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
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "b1c55339b7c6c350ee89f2c1604299660525b248"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.15.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "260fd2400ed2dab602a7c15cf10c1933c59930a2"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.5"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

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
git-tree-sha1 = "82d8afa92ecf4b52d78d869f038ebfb881267322"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.FlameGraphs]]
deps = ["AbstractTrees", "Colors", "FileIO", "FixedPointNumbers", "IndirectArrays", "LeftChildRightSiblingTrees", "Profile"]
git-tree-sha1 = "d9eee53657f6a13ee51120337f98684c9c702264"
uuid = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"
version = "0.2.10"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

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
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "e7cbed5032c4c397a6ac23d1493f3289e01231c4"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.14"
weakdeps = ["Dates"]

    [deps.InverseFunctions.extensions]
    DatesExt = "Dates"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

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
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

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

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

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
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "BaseDirs", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "Reexport", "TOML"]
git-tree-sha1 = "1ae939782a5ce9a004484eab5416411c7190d3ce"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.4.6"

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
git-tree-sha1 = "ab55ee1510ad2af0ff674dbcced5e94921f867a9"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.59"

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

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProfileCanvas]]
deps = ["FlameGraphs", "JSON", "Pkg", "Profile", "REPL"]
git-tree-sha1 = "41fd9086187b8643feda56b996eef7a3cc7f4699"
uuid = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
version = "0.1.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

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

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "bf074c045d3d5ffd956fa0a461da38a44685d6b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.3"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

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
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

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

[[deps.Transducers]]
deps = ["Accessors", "Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "5215a069867476fc8e3469602006b9670e68da23"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.82"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─b6144c34-9f2b-4dc4-81cb-20e3a4cef298
# ╟─5340f896-674d-4675-b53a-8e22b536a269
# ╟─6a3e83b0-b4b4-4f4b-bd72-eb97df199465
# ╟─da6ab60e-1677-41dc-82a1-bbc0c9234e25
# ╟─9836edb5-5d95-4091-af9a-849b6d077cbf
# ╟─4835bed5-a02a-49e9-8a01-63885109339c
# ╟─872b6292-8318-4161-915c-c3d3b9ef1236
# ╠═43c6bb95-81a1-4988-878c-df376e3f7caa
# ╟─3165f2d7-38a2-4852-98aa-afa4cabfb2ed
# ╟─fa07a49b-68fb-4478-a29b-9289f6a3d56a
# ╠═48954b7d-5165-4c4f-9af1-ee4217af5127
# ╠═ac91e5e8-a282-4c36-ad7e-3fed3b30855a
# ╟─4e738b67-0d03-486a-a2cd-d18144a7fcd7
# ╟─7d7527be-2cfa-4c7b-8344-8049d91835b0
# ╟─06f6647d-48c5-4ead-b7b5-90a968363215
# ╠═92556e91-abae-4ce3-aa15-b35c4a65cff5
# ╠═fef1b14a-5495-439d-9428-338be5c4f6e8
# ╟─1188e680-cfbe-417c-ad61-83e145c39220
# ╠═10d4576c-9b86-469c-83b7-1e3d3bc21da1
# ╠═be227f6e-6d25-4a4a-97ab-21ecd6af917e
# ╠═f750ec24-b9a0-4b4e-88ee-c6e4867103c7
# ╠═b0059e3e-0351-4af7-a60b-56896e2b1a05
# ╟─3b3decd0-bb00-4fd2-a8eb-a5b14aede950
# ╟─e14350ea-5a00-4a8f-8b81-f751c69b67a6
# ╟─770c4392-6285-4e00-8d72-5c6a132d8aa9
# ╟─5994f7fd-ecd1-4c2b-8000-5eaa03262a63
# ╟─0fca8f38-f282-4168-87d3-aab0ec0c6346
# ╠═8e53fb6e-db4b-48e7-8cca-db3e6f16a3c3
# ╠═fed249aa-2d0a-4bc3-84ea-e3ad4b4e66fa
# ╠═97660b1c-e09c-4e52-a88c-55522141a39b
# ╟─4b277cea-668e-43d6-bd2a-fcbf62be9b12
# ╟─82f710d7-6ae8-4794-af2d-762ee3a73a3f
# ╟─8cae3e2f-9fb8-485a-bdc7-3fff48a2f9b5
# ╟─26285297-5614-41bd-9ec4-428d37d1dd3e
# ╟─19114bac-a4b1-408e-a7ca-26454b894f72
# ╠═3709ca81-7ad9-453b-a8ba-929bb4031e9b
# ╠═e1bd5582-c734-4597-9fdd-2ee0221fb35d
# ╠═dc3e1ed4-3e48-4bf0-9cc0-a7ce0eab226e
# ╠═2f7afb63-22de-49af-b907-4aeb75dc9f2a
# ╟─035a6f5c-3bed-4f72-abe5-17558331f8ba
# ╟─62436d67-a417-476f-b508-da752796c774
# ╟─84815181-244c-4f57-8bf0-7617379dda00
# ╟─e30d2af4-b6e7-46fb-ad72-4672caa81de4
# ╟─08b70e16-f113-4464-bb4b-3da393c8500d
# ╠═1fed0e8d-0014-4484-8b61-29807caa8ef7
# ╠═0fdaf201-2cdf-419d-9452-4ec14ea281dc
# ╠═6e73940d-15fb-4f61-8100-05fdf7f50e10
# ╠═3a707040-a763-42f6-9f5c-8c56a5f869f7
# ╟─73c4f222-a405-493c-9127-0f950cd5fa0e
# ╟─c4e1d754-2535-40be-bbb3-075ca3fa64b9
# ╟─478aa9a3-ac58-4520-9613-3fcf1a1c1952
# ╠═ed7c22bf-2773-4ff7-93d0-2bd05cfef738
# ╠═18bc3870-3261-43d0-924b-46ca44a9e8ce
# ╠═125214ee-9fc5-4976-a622-23f0ce4e3cd7
# ╠═7c9c22ee-f245-45e1-b1b3-e8d029468f65
# ╠═021f942f-affa-4fb6-92da-65290680643a
# ╠═9d335e8c-714f-40ad-be4f-6c2d357bf1c3
# ╠═9925509b-ee7e-430c-a646-fbf59bc75e62
# ╟─43da70fd-e3c4-4d2d-9204-29aa5007df63
# ╟─823a8e5d-2092-480f-ad6c-4fc9e83e88c0
# ╟─381bfc1e-9bc4-47f7-a8d3-116933382e25
# ╟─b991831b-f15d-493c-835c-c7e8a33f8d7b
# ╟─e6beff79-061c-4c01-b469-75dc5d4e059f
# ╟─7851e968-a5af-4b65-9591-e34b3404fb09
# ╟─bfef62c9-4186-4b01-afe2-e49432f04265
# ╟─ac5f7dcc-02ba-421c-a593-ca7ba60b3ff2
# ╟─8bfaa611-35fd-44d3-920f-c7c51d02216f
# ╟─900a2ece-9638-49fc-afbe-e012f9520b48
# ╟─0f6cc7a9-4184-471f-86d5-4ad0c0e495ce
# ╟─91ca282d-e857-41d7-b99d-d9449b82da09
# ╟─5b53ef57-12d1-45e2-ad1e-28c490c336a6
# ╟─966eae0d-7556-4ff9-b9f7-d47a736524a4
# ╟─cb96b24a-65aa-4832-bc7d-093f0c951f83
# ╟─7df4fcbb-2f5f-4d59-ba0c-c7e635bb0503
# ╟─4f0f052d-b461-4040-b5ff-46aac74a24de
# ╟─cf902114-94e3-4402-ae04-8f704dd6adad
# ╟─a3e85772-9c67-454f-94d2-c2608b53c427
# ╟─f52b6f5d-3832-41aa-8ccd-78e514e65c8b
# ╠═1f9752c2-7bb9-4cd2-b90b-2995bcec7ae3
# ╠═f87fd155-d6cf-4a27-bbc4-74cc64cbd84c
# ╟─a59f0142-9f0c-452b-91ea-647f9201a8d6
# ╟─7f3a1d41-dd16-493c-a59c-764aec13d076
# ╟─4a80a7c3-6e9a-4973-b48a-b02509823830
# ╟─6467d0ee-d551-4558-a765-aa832373d125
# ╟─f218de8b-6003-4bd2-9820-48165cfde650
# ╟─3a868cc5-4123-4b5f-be87-589430df389f
# ╟─2e4bdce5-6188-4c22-a56b-7051c63aa165
# ╟─7cce54bb-eaf9-488a-a836-71e72ba66fcd
# ╟─6d74b5de-1fc9-48af-96dd-3e090f691641
# ╟─6253a562-2a48-45da-b453-1ec7b51d2073
# ╟─0a7c9e73-81a7-45d9-bf9e-ebc61abeb552
# ╠═c2903e20-1be8-4d79-8716-798f5dc15bd4
# ╠═55d182d1-aa25-4ac9-802f-129756ffa302
# ╠═ecebce8b-0e2a-49d0-89f5-53bd0ffdd1a3
# ╠═42dab9c3-dd04-4129-ba3c-b6fb22e2afbe
# ╠═1e24a0aa-dbf9-422e-92c9-834f293a0c02
# ╠═eec3017b-6d02-49e6-aedf-9a494b426ec5
# ╠═2fe59959-5d89-4ae7-839c-ecf82e2c71d8
# ╟─40f6257d-db5c-4e21-9691-f3c9ffc9a9b5
# ╟─bf12d9c9-c79d-4398-9f15-27cbde1ed476
# ╟─102d169a-8bd0-42f4-bfc9-3a32708afadc
# ╠═929c353b-f67c-49ff-85d3-0a27cafc59cf
# ╟─a6a3a31f-1411-4013-8bf7-fbdceac9c6ba
# ╟─1d555f77-c404-485a-9244-717c12c80d28
# ╟─3df86061-63f7-4c1f-a141-e1848f6e83e4
# ╟─8abba353-2309-4931-bf3f-6b1f500998a7
# ╠═860650f0-c6bb-43d6-9ece-c6e6f39e010d
# ╠═ce8a7ed9-7719-4caa-a680-76fac3dea985
# ╠═c773d4a4-aade-4d14-9dbe-96b146450e3f
# ╠═99667755-430c-4687-a6b2-b6c4e91f0e26
# ╠═71d18d73-0bcb-48ee-91fd-8fa2f52a908c
# ╠═e4476a04-036e-4074-bd90-54475c00800a
# ╠═0a81b18a-0ac8-45ba-ad46-02034ae8fb55
# ╟─7c553f77-7783-439e-834b-53a2cd3bef5a
# ╠═a502c80a-fe11-4184-9731-c634655a825d
# ╠═ad34ce87-d9cc-407b-9670-25ed535d2d8d
# ╠═67a0aac8-b022-4051-804c-cfda3e0c7357
# ╠═6dfbf4c3-8d66-45d3-ae1c-ad50a53eb570
# ╠═7136cb2a-6957-4d21-a6e2-381e571a113a
# ╠═3d86b788-9770-4356-ac6b-e80b0bfa1314
# ╠═52e73547-ce0d-4696-8a3c-46ced9fa6582
# ╠═ea19d77b-96bf-411f-8faa-6007c11e204b
# ╠═e375ca3a-57a7-4ca3-a672-4aa724cba34d
# ╠═37a7a557-77ea-4440-8bf0-05f34b55ffc6
# ╠═ad55c2d1-404f-4396-aff8-b8c207157ce4
# ╠═ba25b564-230b-4e06-aba5-c7d3197970ef
# ╟─a2027cca-4a12-4d7d-a721-6044c6255394
# ╟─4e6b27be-79c3-4224-bfc1-7d4b83be6d39
# ╠═4d6472e3-cbb6-4b5c-b06a-4210ff940409
# ╟─1b83b6c2-43cb-4ad4-b5a9-46e31d585a27
# ╟─51fecb7e-65ff-4a11-b043-b5832fed5e02
# ╠═9a7e922b-44e5-4c5e-8288-e39a48e151d5
# ╠═b40f0a76-9405-46d0-aae2-8987b296766a
# ╟─105b8874-5cbc-4777-87c6-e8712cbcc78d
# ╟─26d60dab-bab1-495d-a236-44f075c912bd
# ╠═faa17fdd-9660-43ab-8f94-9cd1c3ba7fec
# ╟─3cc38ba2-70ce-4250-be97-0a48c2c2b484
# ╟─fbfeb350-d9a7-4960-8f9b-a9f70e19a4e2
# ╠═4f645ebc-27f4-4b68-93d9-2e35232cedcf
# ╟─4efee19f-c86c-44cc-8b4b-6eb45adf0aa1
# ╠═1c829fde-e15d-42db-a608-2e5bdbaa4d8c
# ╠═f77cf1bb-6385-403d-a224-3c9c7313e591
# ╠═66886194-a2bd-4b1e-9bff-fbb419fddc78
# ╟─4bd400f3-4cb4-47a2-b0f5-31e6dedc253d
# ╠═b666c289-de0f-4412-a5f7-8e5bb546a47c
# ╟─97234d16-1455-4321-bb16-c09534a58594
# ╟─a6b08af6-34e8-4316-8f8c-b8e4b5fbb98a
# ╟─d7037f99-d3b8-4986-95c8-58f4f043e916
# ╟─39a1fc54-4024-4d89-9eeb-1fab0477e684
# ╠═46c11a87-10aa-46e2-8961-7acd33059b96
# ╠═55fbc75b-44d2-49e4-830f-fdb88eadafdb
# ╠═f316a6f8-b462-4cec-b2ff-434330be579a
# ╠═a1b90125-d3dd-409c-8231-ab0c3a85153e
# ╠═5648561c-98cf-4aa6-9af4-16add4706c3b
# ╠═5db8f67c-17fe-4c08-81df-42b47143b0ba
# ╟─900523ce-f8e7-4f33-a294-de86a7fb8869
# ╠═f3df4648-2884-4b01-823d-7e8ee2adc35b
# ╠═e9fb9a9a-73cd-49ee-ab9f-e864b2dbd8bf
# ╟─73aece7b-314d-4f5f-bf7f-89852156e89e
# ╟─9c7c571e-ef14-4fe2-b3a9-aa66131226f8
# ╟─b0d184ed-4129-49bf-afb7-7a848c93f15b
# ╠═a2b62ae3-13d2-4d5b-a8ac-5c1c3c1ee246
# ╠═84d1f707-3a72-49a5-bf11-62316f69232a
# ╟─eebfe8e7-56dd-457c-a1e6-1a67b3b7ceec
# ╟─54cd4729-e4d3-4783-af1d-17df32ca6d69
# ╠═138fb7ec-bfd3-4798-8cbc-cb1c8982b799
# ╠═d4435765-167c-433b-99ea-5cb9f1f3ac82
# ╟─0ad54e4b-ea9d-418c-bb6a-cd8fbe241c73
# ╟─5979b5ec-5fef-40ef-a5c3-3a5b3d3040d9
# ╟─d250a257-4dc6-4369-90f0-fe186b3d9e7b
# ╟─b7506e65-60eb-4985-9a28-5a29cb400670
# ╠═854fa686-914c-4a56-a975-486a542c0a9b
# ╠═a858aeaa-29f5-4615-805c-0c6093cf9b5f
# ╠═0d8da60d-5e21-4398-a731-ed87754b63c8
# ╠═337b9905-9284-4bd7-a06b-f3e8bb44679c
# ╠═ac7606f4-5986-4110-9acb-d7b089e9c98a
# ╠═6e2a99bc-7f49-4455-8b23-11392e47f24d
# ╠═034734a7-e7f0-4ea5-b252-5916f67c65d4
# ╠═749b5691-506f-4c7f-baa2-6d3e9b2607b9
# ╟─e18328c7-837e-427c-b5c0-3be31dcb4c4b
# ╟─35df802b-41e6-4ede-8200-5658e3ee1328
# ╟─9fb8f6ea-ca20-461c-b790-f651b13721b2
# ╟─c3c3bb5c-4bcf-442e-9718-c18a4a03fa82
# ╠═5aacf874-1519-4665-9207-f687b6e9944b
# ╠═1dab32e6-9d81-4de3-9b97-6a2ac58a28c3
# ╠═cc09fc0b-bf88-464a-980b-59ae86bbd5d8
# ╠═59307ddd-c24b-444f-9723-badc7e6da897
# ╠═2f70e03f-1556-4fce-b4f5-394df4266eb7
# ╠═6823a91e-c02e-495c-9e82-e22b18857df7
# ╟─64d2a0e3-4ecd-4d44-b5cc-0ff23b3776dd
# ╠═c75d9e65-f9be-4b8a-9bd4-9dbeeafec16e
# ╟─41361309-8be9-464a-987e-981035e4b15a
# ╟─ee8a054e-64db-4c61-a5d3-b38e692887d9
# ╠═7146eebf-bd16-424f-ae55-de51689bc0fe
# ╠═ae700906-4b17-45ce-b27e-e2a0d745e259
# ╠═823d65da-5636-4f7d-9582-2a0189a564ae
# ╠═9a4027f9-243d-4fc6-916a-2f89a76120c9
# ╠═94193bc1-91c4-4d3e-8e44-cd37495481bf
# ╠═d7de7be9-8d97-4476-ba09-9f84d2cebb00
# ╟─acd5ff5b-f9d0-41bf-ae09-cf6842eab556
# ╠═5b66bf73-b7dd-4054-9efb-1c30a475bc6b
# ╟─2bab0784-b185-44f0-9dec-c98bf164827b
# ╟─be74f8fb-fd58-4170-8735-1af55a04d48f
# ╠═2688c9b8-07c0-4105-b2af-a7c71c48fb31
# ╠═afdd018f-c923-4906-9b70-c7b0a3e16831
# ╠═447cd06f-e110-450d-984f-ceb1d6361b43
# ╠═ffefe265-b7a3-4bc4-9e5f-2da6c406cc56
# ╠═adfc5819-6bb7-40fc-baef-9770efeb6a21
# ╠═cd834845-8ca9-407a-91da-d3104b0bd9b7
# ╠═3eca2837-16fb-4237-9ebd-8b6378ca13a8
# ╠═d32191cf-1d96-495a-bf04-f0bc5a5ecaa0
# ╠═c87db76f-4c6a-4fe2-822b-8ee88079e30d
# ╟─3dc94c4a-1072-4e9d-8408-439ea20a6029
# ╟─82f82d2a-beb4-4520-ac19-a498892d009c
# ╟─6b19aee6-a997-4eb4-9177-badd8ad2a540
# ╟─610fc6de-6045-4c3f-8da1-95e9e5a4b986
# ╠═903563ee-2f2a-48d4-991f-714d7da6808c
# ╠═451e7b71-a495-4b95-803f-4fedd9268316
# ╠═74a4fba2-06b4-4198-9e57-6f5911760a72
# ╠═a617d0a8-f1d7-492a-98a9-357a8223d6b0
# ╠═ad6e7b16-3819-4d59-9e7c-6bc83d6df468
# ╟─78ecd319-1f5c-4ba0-b9c4-da0dfadb4b2c
# ╟─a912feaa-b2b2-479e-befe-9e919e453e31
# ╠═305c2556-e269-40ea-ac90-6e75f03f07ce
# ╠═221814d5-676a-4bbf-9617-a25cfe1c5f47
# ╠═fc0d29f4-fd2e-45b0-ba19-f7552643efc7
# ╠═2abe4ba3-3834-4f20-9ce4-2496711fb0af
# ╠═b67ca69e-24ab-4893-90a2-85275b6fa306
# ╠═d31b4e4f-18bf-4649-82f8-c603712bdbf0
# ╠═66f6cad5-cc5c-4a81-86d1-fb893bc4fe12
# ╠═970f3789-f830-47af-938f-0faf5f36421b
# ╠═63cbbd0e-388d-4f1f-8e11-95366bce37ba
# ╠═300577bd-22b7-4fdd-8a85-4768ecf985c4
# ╠═78eda243-db35-4eb4-8e97-e845dd3da064
# ╠═b056168b-1f10-4046-9a0c-dbe89a713d6a
# ╠═482d1c2d-0898-48eb-b122-51e22d51a265
# ╠═0b2e6a3c-caaa-4d79-9a3a-6b1d85037fb2
# ╠═00e567e7-ab21-4f4a-aec1-b90e45f3db2a
# ╠═fa267730-d67d-4cd4-a9d5-901e79e553e5
# ╠═b062a7a6-4776-4db0-9712-1c832d7f271c
# ╠═3f35548e-1bfc-4262-9534-ad4bc159bcf9
# ╠═4e906d8c-ca74-42e3-a9e3-b3980206fbe3
# ╠═3e4fc9d3-1d87-431b-b348-09e7567149f0
# ╠═8ffd78db-cfc5-4695-a1c1-6a4e6aa32348
# ╠═2a2d1b60-be6f-4f9c-8190-7c0a2d77d510
# ╟─796eeb6c-1152-11ef-00b7-b543ec85b526
# ╠═0574291d-263a-4836-8cb9-78ad7de3f095
# ╠═cbcc1cd8-7319-4076-84cf-f7ae4d0b5794
# ╠═7b4e1a9b-ef0b-41f6-a634-99af17a02f60
# ╟─32c92099-f322-4086-983d-50b79ab28de8
# ╠═afaac0aa-d0e2-4e2c-a5ed-08b89b901541
# ╠═a40d6dd3-1f8b-476a-9839-1bd1ae46751a
# ╠═31a3bb9e-4ef3-4876-87c2-12d462e60eab
# ╠═d5431c0e-ac46-4de1-8d3c-8c97b92306a8
# ╠═5ab5f9d5-b60a-4556-a8c7-47c808e5d4f8
# ╠═4bfdde5d-857f-4955-809d-f4a21440000e
# ╠═7ad8dc82-5c60-493a-b78f-93e37a3f3ab8
# ╠═b70ec2b1-f8c2-4288-831a-041804d2ec43
# ╠═9b937c49-7216-47c9-a1ef-2ecfa6ff3b31
# ╠═93cbc453-152e-401e-bf53-c95f1ae962c0
# ╠═c11ab768-1da4-497b-afc1-fb64bc3fb457
# ╠═3279ba47-18a1-45a9-9d29-18b9875ed057
# ╠═cbeac89a-845c-4409-8067-8766fe3b8a24
# ╠═4f193af4-9925-4047-92f9-c67eec1f4c97
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
