### A Pluto.jl notebook ###
# v0.19.41

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

# ╔═╡ 4c0872f8-0fd5-4a44-9587-28cb04697d25
using BenchmarkTools

# ╔═╡ f5809bd3-64d4-47ee-9e41-e491f8c09719
begin
	using PlutoPlotly, PlutoUI, LinearAlgebra, LinearAlgebra.BLAS, Random, HypertextLiteral, Latexify
# 	using PlutoPlotly, PlutoUI, HypertextLiteral, Random, LinearAlgebra, LinearAlgebra.BLAS
	TableOfContents()
end

# ╔═╡ 4017d910-3635-4ffa-ac1b-919a7bff1e6e
md"""
# Chapter 4 

# Dynamic Programming
"""

# ╔═╡ fb497aa8-9d34-48e6-ae85-72df30c1adf3
md"""
Throughout this chapter we explore methods to solve the *Bellman optimality equations*.  Below are the equations for the *state-value function* as well as the *state-action value funtion*:

$\begin{flalign}
	v_*(s) &= \max_a \mathbb{E}[R_{t+1}+\gamma v_*(S_{t+1}) \vert S_t = s, A_t = a] \\
	&= \max_a \sum_{s^\prime,r}p(s^\prime,r\vert s,a)\left [ r + \gamma v_*(s^\prime) \right ] \tag{4.1}
\end{flalign}$

or

$\begin{flalign}
	q_*(s,a) &= \mathbb{E} \left [ R_{t+1}+\gamma \max_{a^\prime} q_*(S_{t+1}, a^\prime) \bigg | S_t = s, A_t = a \right ] \\
	&= \max_a \sum_{s^\prime,r}p(s^\prime,r\vert s,a)\left [ r + \gamma \max_{a^\prime} q_*(s^\prime, a^\prime) \right ] \tag{4.2}
\end{flalign}$

for all $s \in \mathcal{S}, \: a \in \mathcal{A}(s)$, and $s^\prime \in \mathcal{S}^+$
"""

# ╔═╡ 55276004-877e-47c0-b5b5-49dbe29aa6f7
md"""
## 4.1 Policy Evaluation (Prediction)
"""

# ╔═╡ 772d17b0-6fbc-4309-b55b-d17f9b4d3ddf
md"""
The state-value function for an arbitrary policy $\pi$ is defined for all $s \in \mathcal{S}$:

$\begin{flalign}
v_\pi(s) &\doteq \mathbb{E}_\pi [G_t \mid S_t = s] \\
&=\mathbb{E}_\pi [R_{t+1} + \gamma G_{t+1} \mid S_t = s] \tag{from (3.9)} \\ 
&=\mathbb{E}_\pi [R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s] \tag{4.3} \\
&=\sum_a \pi(a\vert s) \sum_{s^\prime,r}p(s^\prime,r \vert s, a) \left [ r + \gamma v_\pi(s^\prime) \right ] \tag{4.4}
\end{flalign}$

where $\pi(a \vert s)$ is the probability of taking action $a$ in state $s$ under policy $\pi$.  The existence and uniqueness of $v_\pi$ are guaranteed as long as either $\gamma < 1$ or eventual termination is guaranteed from all states under the policy $\pi$.

If we have the probability transition function $p(s^\prime, r \vert s, a)$, then (4.4) is a system of $\left | \mathcal{S} \right |$ simultaneous linear equations which can be solved tediously with linear algebra techniques.  However, there is an alternative iterative solution method that makes use of the Bellman equation for $v_\pi$ (4.4), turning it into an update rule.

$\begin{flalign}
v_{k+1}(s) &\doteq \mathbb{E}_\pi [R_{t+1} + \gamma v_k(S_{t+1}) \mid S_t = s] \\
&=\sum_a \pi(a\vert s) \sum_{s^\prime,r}p(s^\prime,r \vert s, a) \left [ r + \gamma v_k(s^\prime) \right ] \tag{4.5}
\end{flalign}$

for all $s \in \mathcal{S}$.  $v_k = v_\pi$ is a fixed point of this iteration because the Bellman equation assures us of equality in this case.  The sequence $\{ v_k \}$ in general converges to $v_\pi$ as $k \rightarrow \infty$ under the same conditions that guarantee the existence of $v_\pi$.  This algorithm is called *iterative policy evaluation*.

Each round of policy evaluation updates every state once.  These are called *expected* updates because they involve directly calculating the expected value using the true probabilities.  To follow this algorithm precisely, we must keep all the values of $v_k$ fixed while we compute $v_{k+1}$, but in practice we can update in place for each state.  As we sweep through the state space then the updates are computed and use new values as soon as they are available.  This method can converge faster than the strict version and does not require keeping a separate copy of the unmodified values.  This in-place version of the algorithm is usually what is implemented.  Below are examples of code that implement iterative policy evaluation.
"""

# ╔═╡ 4665aa5c-87d1-4359-8cfd-7502d8c5d2e2
md"""
### Iterative Policy Evaluation Implementation
"""

# ╔═╡ 86cd5d5f-f79f-4dc6-9d88-ae4753190de9
md"""
#### MDP Data Structures
"""

# ╔═╡ 7575b85a-c988-47e6-bdcc-fde4d92708a5
begin
	struct FiniteMDP{T<:Real, S, A} 
		states::Vector{S}
		actions::Vector{A}
		rewards::Vector{T}
		# ptf::Dict{Tuple{S, A}, Matrix{T}}
		ptf::Array{T, 4}
		action_scratch::Vector{T}
		state_scratch::Vector{T}
		reward_scratch::Vector{T}
		state_index::Dict{S, Int64}
		action_index::Dict{A, Int64}
		function FiniteMDP{T, S, A}(states::Vector{S}, actions::Vector{A}, rewards::Vector{T}, ptf::Array{T, 4}) where {T <: Real, S, A}
			new(states, actions, rewards, ptf, Vector{T}(undef, length(actions)), Vector{T}(undef, length(states)+1), Vector{T}(undef, length(rewards)), Dict(zip(states, eachindex(states))), Dict(zip(actions, eachindex(actions))))
		end	
	end
	FiniteMDP(states::Vector{S}, actions::Vector{A}, rewards::Vector{T}, ptf::Array{T, 4}) where {T <: Real, S, A} = FiniteMDP{T, S, A}(states, actions, rewards, ptf)
end

# ╔═╡ 5b65e817-5504-413f-9c1a-17880d238d80
function bellman_value!(V::Vector{T}, mdp::FiniteMDP{T, S, A}, π::Matrix{T}, γ::T) where {T <: Real, S, A}
	delt = zero(T)
	@inbounds @fastmath @simd for i_s in eachindex(mdp.states)
		newvalue = zero(T)
		@inbounds @fastmath @simd for i_a in eachindex(mdp.actions)
			# mdp.action_scratch[i_a] = zero(T)
			x = zero(T)
			for (i_r, r) in enumerate(mdp.rewards)
				@inbounds @fastmath @simd for i_s′ in eachindex(V)
					# mdp.action_scratch[i_a] += mdp.ptf[i_s′, i_r, i_s, i_a] * (r + γ * V)
					x += mdp.ptf[i_s′, i_r, i_a, i_s] * (r + γ * V[i_s′])
				end
				# mdp.state_scratch .= (r .+ γ .* V) .* mdp.ptf[:, i_r, i_a, i_s]
				# x += sum(mdp.state_scratch)
			end
			# mdp.action_scratch[i_a] = x
			newvalue += x * π[i_a, i_s]
		end
		# mdp.action_scratch .*= π[:, i_s]
		# newvalue = sum(mdp.action_scratch)

		# newvalue = dot(mdp.action_scratch, π[:, i_s])

		# for i in eachindex(mdp.actions)
			# newvalue += π[i, i_s] * mdp.action_scratch
		# newvalue = π[:, i_s]' * mdp.action_scratch
		# newvalue = π[:, i_s]' * [sum(mdp.ptf[:, :, i_s, i_a]' * (mdp.rewards' .+ γ .* V)) for i_a in eachindex(mdp.actions)]

		#convert this loop over actions into a matrix operation with the policy vector, need to loop over rewards in that case instead of actions and that means that the ptf representation needs to be different where the rewards are what separates things rather than the S,A tuple.  Could be a map from states to a matrix which is the action/newstate matrix so it matches where every matrix has actions in the rows and states in the columns.  Then there would just be a list of these matrices for each reward, but in the case of only having 1 reward, this would just be a single matrix
		# for (i_a, a) in enumerate(mdp.actions)
		# 	#this is the probability distribution across rewards and states for the transition
		# 	rdist = mdp.ptf[(s, a)]
		# 	newvalue += sum(π[i_a, i_s] * rdist * (mdp.rewards' .+ (γ .* V)))
		# end
		delt = max(delt, abs(newvalue - V[i_s]) / (eps(zero(T)) + abs(newvalue)))
		V[i_s] = newvalue
	end
	return delt
end

# ╔═╡ 59b91c65-3f8a-4015-bb08-d7455623101c
function bellman_value!(V::Dict, p::Dict, sa_keys::Tuple, π::Dict, γ::Real)
	delt = 0.0
	for s in intersect(keys(sa_keys[1]), keys(π))
		v = V[s]
		actions = intersect(sa_keys[1][s], keys(π[s]))
		# if !isempty(actions)
			V[s] = 	sum(π[s][a] * 
						sum(p[(s′,r,s,a)] * (r + γ*V[s′]) 
							for (s′,r) in sa_keys[2][(s,a)])
					for a in actions)
		# end
		delt = max(delt, abs(v - V[s]) / (eps(0.0) + abs(V[s])))
	end
	return delt
end

# ╔═╡ 88f8e335-968e-4e2f-8d3a-395667ad2ed3
function iterative_policy_eval_v(π::Matrix{T}, θ::T, mdp::FiniteMDP{T, S, A}, γ::T, V::Vector{T}, nmax::Real) where {T <: Real, S, A}
	nmax < 0 && return V
	delt = bellman_value!(V, mdp, π, γ)
	delt <= θ && return V
	iterative_policy_eval_v(π, θ, mdp, γ, V, nmax - 1)	
end

# ╔═╡ 5b912508-aa15-470e-be4e-430e88d8a68d
function iterative_policy_eval_v(π::Dict, θ::Real, mdp::NamedTuple, γ::Real, V::Dict, nmax::Real)
	nmax < 0 && return V
	(p, sa_keys) = mdp
	delt = bellman_value!(V, p, sa_keys, π, γ)
	delt <= θ && return V
	iterative_policy_eval_v(π, θ, mdp, γ, V, nmax - 1)	
end

# ╔═╡ d6b58b68-504d-463f-91e8-ee85d0f90000
#first call when the value function is initialized with a dictionary
iterative_policy_eval_v(π::Matrix{T}, θ::Real, mdp::FiniteMDP{T, S, A}, γ::T, Vinit::Vector{T}; nmax=Inf) where {T <: Real, S, A} = iterative_policy_eval_v(π, θ, mdp, γ, deepcopy(Vinit), nmax - 1)

# ╔═╡ 9253064c-7dfe-445f-b377-fc1acbb6886e
#first call when the value function is initialized with a dictionary
iterative_policy_eval_v(π::Dict, θ::Real, mdp::NamedTuple, γ::Real, Vinit::Dict; nmax=Inf) = iterative_policy_eval_v(π, θ, mdp, γ, deepcopy(Vinit), nmax - 1)

# ╔═╡ d5eda4ce-1c22-4d3d-b1b0-9c8d0357f6cf
#first call when the value function is initialized with a value
iterative_policy_eval_v(π::Matrix{T}, θ::Real, mdp::FiniteMDP{T, S, A}, γ::T, Vinit::T = zero(T); nmax = Inf) where {T <: Real, S, A} = iterative_policy_eval_v(π, θ, mdp, γ, zeros(T, size(mdp.ptf, 1)); nmax = nmax)

# ╔═╡ e6cdb2be-697d-4191-bd5a-9c129b32246d
#first call when the value function is initialized with a value
iterative_policy_eval_v(π::Dict, θ::Real, mdp::NamedTuple, γ::Real, Vinit::Real = 0.0; nmax = Inf) = iterative_policy_eval_v(π, θ, mdp, γ, Dict(s => Vinit for s in keys(mdp[2][1])); nmax = nmax)

# ╔═╡ 4dd32517-4c12-4635-8135-3019828af2b5
begin
	abstract type GridworldAction end
	struct North <: GridworldAction end
	struct South <: GridworldAction end
	struct East <: GridworldAction end
	struct West <: GridworldAction end
	const gridworld_actions = [North(), South(), East(), West()]
end

# ╔═╡ df9593ca-6d27-45de-9d46-79bddc7a3862
#p is the state transition function for an mdp which maps the 4 arguments to a probability.  This function uses p to generate two dictionaries.  The first maps each state to a set of possible actions in that state.  The second maps each state/action pair to a set of possible transition/reward pairs
function get_sa_keys(p::Dict{Tuple{A, B, A, C}, T}) where {T <: Real, A, B, C}
	#map from states to a list of possible actions
	state_actions = Dict{A, Set{C}}()

	#map from state action pairs to a list of possible newstate/reward pairs
	sa_s′rewards = Dict{Tuple{A, C}, Set{Tuple{A, B}}}()
	for k in keys(p)
		(s′, r, s, a) = k
		haskey(state_actions, s) ? push!(state_actions[s], a) : state_actions[s] = Set([a])
		haskey(sa_s′rewards, (s,a)) ? push!(sa_s′rewards[(s,a)], (s′, r)) : sa_s′rewards[(s,a)] = Set([(s′,r)])
	end
	return state_actions, sa_s′rewards
end	

# ╔═╡ c7bdf32a-2f89-4bf8-916b-7558ceedb628
function gridworld4x4_mdp()
	states = collect(1:14)
	s_term = 0
	#define p by iterating over all possible states and transitions: p(s′, r, s, a)
	#there is 0 reward and a probability of 1 staying in the terminal state for all 	actions taken from the terminal state
	p = Dict((0, 0.0, 0, a) => 1.0 for a in gridworld_actions)

	#positions that result in terminal states
	termset(::North) = (4,)
	termset(::South) = (11,)
	termset(::East) = (14,)
	termset(::West) = (1,)

	#positions that leave the state unchanged
	constset(::North) = (1, 2, 3)
	constset(::South) = (12, 13, 14)
	constset(::East) = (3, 7, 11)
	constset(::West) = (4, 8, 12)

	#usual movement rule
	delta(::North) = -4
	delta(::South) = +4
	delta(::East) = +1
	delta(::West) = -1
	
	function move(s, dir::GridworldAction)
		in(s, termset(dir)) && return s_term
		in(s, constset(dir)) && return s
		return s + delta(dir)
	end

	#add other transitions to p
	for s in states
		for a in gridworld_actions
			s′ = move(s, a)
			p[s′, -1.0, s, a] = 1.0
		end
	end
	sa_keys = get_sa_keys(p)
	return (p = p, sa_keys = sa_keys)
end

# ╔═╡ 6d04573a-c2a9-4cfa-88f1-2b2723a95aac
function create_4x4gridworld_mdp()
	states = collect(1:14)
	actions = gridworld_actions
	rewards = [-1.0]
	s_term = length(states) + 1

	#positions that result in terminal states
	termset(::North) = (4,)
	termset(::South) = (11,)
	termset(::East) = (14,)
	termset(::West) = (1,)

	#positions that leave the state unchanged
	constset(::North) = (1, 2, 3)
	constset(::South) = (12, 13, 14)
	constset(::East) = (3, 7, 11)
	constset(::West) = (4, 8, 12)

	#usual movement rule
	delta(::North) = -4
	delta(::South) = +4
	delta(::East) = +1
	delta(::West) = -1
	
	function move(s, dir::GridworldAction)
		in(s, termset(dir)) && return s_term
		in(s, constset(dir)) && return s
		return s + delta(dir)
	end

	function getmatrix(s, a)
		#initialize the matrix for s′, r transitions, each column runs over the transition states including the terminal state which is always in the last row by convention
			out = zeros(length(states) + 1, length(rewards))
			#since this mdp is deterministic, there will only be a single value of 1.0 for the transition which is valid, so I just need to replace the 0.0 with a 1.0 corresponding to the unique s′ reached by taking action a in state s.  also there is only one reward to consider
			s′ = move(s, a)
			out[s′, 1] = 1.0
			return out
	end

	#initialize probability function with all zeros
	ptf = zeros(length(states)+1, length(rewards), length(actions), length(states))

	for (i_s, s) in enumerate(states)
		for (i_a, a) in enumerate(actions)
			ptf[:, :, i_a, i_s] .= getmatrix(s, a)
		end
	end
	# ptf = Dict((s, a) => getmatrix(s, a) for s in states for a in actions)

	FiniteMDP(states, actions, rewards, ptf)
end

# ╔═╡ 7563d584-abfc-4d9f-9759-163ac55922a9
create_4x4gridworld_mdp()

# ╔═╡ 6048b106-458e-4e3b-bba9-5f3578458c7c
#forms a random policy for a generic finite state mdp.  The policy is a dictionary that maps each state to a dictionary of action/probability pairs.
function form_random_policy(sa_keys)
	Dict([begin
		s = k[1]
		actions = k[2]
		l = length(actions)
		p = inv(l) #equal probability for all actions
		s => Dict(a => p for a in actions)
	end
	for k in sa_keys[1]])
end

# ╔═╡ 4c46ea30-eeb2-4c25-9e6c-3bcddaf48de7
#forms a random policy for a generic finite state mdp.  The policy is a matrix where the rows represent actions and the columns represent states.  Each column is a probability distribution of actions over that state.
form_random_policy(mdp::FiniteMDP{T, S, A}) where {T, S, A} = ones(T, length(mdp.actions), length(mdp.states)) ./ length(mdp.actions)
	

# ╔═╡ 0d0e82e4-b3a4-4528-9288-285fdc5aa8af
function makefig4_1(nmax=Inf)
	gridworldmdp = gridworld4x4_mdp()
	π_rand = form_random_policy(gridworldmdp[2])
	V = iterative_policy_eval_v(π_rand, eps(0.0), gridworldmdp, 1.0, nmax = nmax)
	[(s, V[s]) for s in 0:14]
end

# ╔═╡ 00593e99-7fa9-49bb-8e60-5a0f083304ab
function makefig4_1_v2(nmax=Inf)
	gridworldmdp = create_4x4gridworld_mdp()
	π_rand = form_random_policy(gridworldmdp)
	V = iterative_policy_eval_v(π_rand, eps(0.0), gridworldmdp, 1.0; nmax = nmax)
	[(gridworldmdp.states[i], V[i]) for i in gridworldmdp.states]
end

# ╔═╡ cda46125-2cfa-4b57-adf8-164eed8f5af6
# ╠═╡ disabled = true
#=╠═╡
@btime makefig4_1()
  ╠═╡ =#

# ╔═╡ 0a17c3d0-7b46-45b2-8d8e-690fd498565c
# ╠═╡ disabled = true
#=╠═╡
@btime makefig4_1_v2()
  ╠═╡ =#

# ╔═╡ 17ff2d94-aea9-4a15-98b3-97437e2b70ab
#for the first attempt at doing this in a matrix style, the time for the 4_1 gridworld is 5.5ms vs ~12ms for the original method.  see how much improvement there is by fixing the action loop above.  Best result is writing everything as a nested loop and not using any linear algebra operations.  This was 366.6μs vs no better than 1 ms using linear algebra and dot fusion operators.  Reduced to 266μs using SIMD on inner loop.  Reduced to 242μs when using SIMD everywhere

# ╔═╡ 0297a007-5898-4936-ae2f-386a725700e4
md"""
### HTML Utilities
"""

# ╔═╡ e4370697-e6a7-40f0-974a-ed219102c13f
function linejoin(a, b) 
"""
$a
$b
"""
end

# ╔═╡ 6844dff1-bc0b-47c5-8496-efe46dafbb5b
function makehtmlgrid(n)
	mapreduce(linejoin, 1:n) do i
		"""
		<div class = "gridcell">$i
		</div>
		"""
	end
end

# ╔═╡ 34f0f670-483f-4add-bf25-34993d646e5e
gridworld_display = HTML("""
<div id = "gridcontainer">
<div class="gridworld">
	<div class = "nullcell"></div>
	$(makehtmlgrid(14))
	<div class = "nullcell"></div>
</div>
</div>
""");

# ╔═╡ 92e50901-7d10-470b-a985-be45adcad817
md"""
### Example 4.1
| |4x4 gridworld | |
|---|:---:|---|
|$$\leftarrow \uparrow \rightarrow \downarrow \text{actions}$$|$gridworld_display|$$R_t = -1$$ on all transitions|

The gray boxes in the corners represent the terminal state
"""

# ╔═╡ d0b4a71b-b574-4d62-be0b-14e03595a15c
function makevaluecell(value)
	"""
	<div class="valuecell">
		$value
	</div>
	"""
end	

# ╔═╡ aa19ffd4-69a0-44a9-8109-d6be003ae7b1
function show_gridworld_values(values; title = "")
	"""
	<div style="display: flex; flex-direction: column;">
	$title
	<div id="gridcontainer">
	<div class="gridworld">
		$(makevaluecell(0.0))
		$(mapreduce(makevaluecell, linejoin, values))
		$(makevaluecell(0.0))
	</div>
	</div>
	</div>
	"""
end

# ╔═╡ 39f4c75c-43f7-476f-bbc9-c704e5dee300
md"""
#### CSS for Policy Grids (hidden cell below by default)
"""

# ╔═╡ e76bd134-f4ac-4382-b56a-fca8f3ca27cd
html"""
<style>
	#gridcontainer {
		display: flex;
		flex-direction: row;
		background: rgba(0, 0, 0, 0);
		margin: 0px;
		padding: 0px;
	}
	.gridworld {
		aspect-ratio: 1 / 1;
		display: grid;
		grid-template-columns: repeat(4, 3vw);
		grid-auto-rows: auto;
		gap: 0px;
		border: 1px solid black;
	}
	
	.gridworld .gridcell,.valuecell,.nullcell,.blankcell {
		width: 3vw;
		height: 3vw;
		border: 1px solid black;
		display: flex;
		color: black;
	}
	
	.gridcell {
		background: white;
		writing-mode: horizontal-lr;
		align-items: end;
		padding-left: 4px;
	}
	
	.valuecell {
		background: white;
		align-items: center;
		justify-content: center;
		font: normal 1vw Veranda;
	}
	
	.nullcell {
		background: gray;
	}
	
	.blankcell {
		background: none;
		border: 0px;
	}

	.policycell {
		border: 1px solid black;
		width: 3vw;
		height: 3vw;
		display: grid;
		grid-template-columns: repeat(3, 1vw);
		grid-template-rows: repeat(3, 1vw);
		margin: 0px;
		padding: 0px;
		background: white;
		color: black;
	}

	.policycell * {
		display: flex;
		align-items: center;
		justify-content: center;
		font: bold 1.3vw Veranda;
	}

	.policycell *::before {
		content: "\2191";
		display: none;
	}

	.policycell[N] :nth-Child(2)::before {
		transform: rotate(0deg);
		display: inline;
	}

	.policycell[NW] :nth-Child(1)::before {
		transform: rotate(-45deg);
		display: inline;
	}

	.policycell[NE] :nth-Child(3)::before {
		transform: rotate(45deg);
		display: inline;
	}

	.policycell[W] :nth-Child(4)::before {
		transform: rotate(-90deg);
		display: inline;
	}

	.policycell[E] :nth-Child(6)::before {
		transform: rotate(90deg);
		display: inline;
	}

	.policycell[SW] :nth-Child(7)::before {
		transform: rotate(-135deg);
		display: inline;
	}

	.policycell[S] :nth-Child(8)::before {
		transform: rotate(180deg);
		display: inline;
	}

	.policycell[SE] :nth-Child(9)::before {
		transform: rotate(135deg);
		display: inline;
	}
</style>
"""

# ╔═╡ f4fce267-78a2-4fd3-aad5-a8298783c015
const directions = ("N", "S", "E", "W")

# ╔═╡ 1d098de3-592e-401b-a493-2728e8a6ffe9
function makepolicycell(πvec)
	inds = findall(πvec .!= 0)
	attr = isempty(inds) ? """""" : reduce((a, b) -> """ $a $b""", directions[inds])
	"""
	<div class="policycell" $attr>
		$(mapreduce(a -> """<div></div>""", linejoin, 1:9))
	</div>
	"""
end	

# ╔═╡ 2c23c4ec-f332-4e05-a730-06fa20a0227a
function show_gridworld_policy(policies)
	"""
	<div id="gridcontainer">
	<div class="gridworld">
		<div class = "nullcell"></div>
		$(mapreduce(makepolicycell, linejoin, policies))
		<div class = "nullcell"></div>
	</div>
	</div>
	"""
end

# ╔═╡ 7539da6f-1fb7-4a63-98ba-52b81bb27eca
function convertpolicy(π::Dict)
	function convertdist!(v::Vector)
		m = maximum(v)
		v .= isapprox.(v, m; atol = 0.01)
		v ./ sum(v)
		return v
	end
	[convertdist!([π[s][a] for a in gridworld_actions]) for s in 1:14]
end

# ╔═╡ f68b0587-1203-4985-9077-ded678ba4b8f
md"""
### Figure 4.1 Interactive
"""

# ╔═╡ 7dcb5621-17ce-4794-b70e-e639e5068a18
md"""
Click here to reset value: $(@bind reset_k Button())
"""

# ╔═╡ 7e9fe05a-c447-4a41-8c61-67c9a899411c
begin
	reset_k
	@bind user_k CounterButton("Click to increase k")
end

# ╔═╡ 20c5e03d-a1f4-4b2e-9893-efb9b03f00e8
md"""
Middle value set by user as k = $(user_k+1).  
"""

# ╔═╡ 5fae76af-ac80-49c5-b553-73d09a6e9098
md"""
### Figure 4.1
"""

# ╔═╡ f80580b3-f370-4a02-a9e2-ed791f380521
md"""
> ### *Exercise 4.1* 
> In Example 4.1, if $\pi$ is the equiprobable random policy, what is $q_{\pi}(11,\text{down})$?  What is $q_{\pi}(7,\text{down})$?
$q_{\pi}(11, \text{down}) = -1$ because this will transition into the terminal state and terminate the episode receiving the single reward of -1.

$q_{\pi}(7,\text{down})=-15$ because we are gauranteed to end up in state 11 and receive a reward of -1 from the first action.  Once we are in state 11, we can add $v_{\pi_{random}}(11)=-14$ to this value since the rewards are not discounted.

"""

# ╔═╡ 71abc452-cefc-47f8-8f9c-6fd3565f3ec6
md"""
> ### *Exercise 4.2* 
> In Example 4.1, supposed a new state 15 is added to the gridworld just below state 13, and its actions, $\text{left}$, $\text{up}$, $\text{right}$, and $\text{down}$, take the agent to states 12, 13, 14, and 15 respectively.  Assume that the transitions *from* the original states are unchanged.  What, then is $v_{\pi}(15)$ for the equiprobable random policy?  Now supposed the dynamics of state 13 are also changed, such that action $\text{down}$ from state 13 takes the agent to the new state 15.  What is $v_{\pi}(15)$ for the equiprobable random policy in this case?
"""

# ╔═╡ 54c73389-bb7a-48f1-b5d5-9d4972b1857a
gridworld_display_modified = HTML("""
<div id = "gridcontainer">
<div class="gridworld">
	<div class = "nullcell"></div>
	$(makehtmlgrid(14))
	<div class = "nullcell"></div>
	<div class = "blankcell"></div>
	<div class = "gridcell">15</div>
</div>
</div>
""")

# ╔═╡ 9e594059-39cd-4fc7-91ae-b8e9156db6df
md"""
In the first case, we can never re-enter state 15 from any other state, so we can use the average of the value function in the states it transitions into.  

$v_{\pi}(15) = -1 + 0.25 \times \left ( v_{\pi}(12) + v_{\pi}(13) + v_{\pi}(14)+ v_{\pi}(15) \right )$ 
$v_\pi(15) = -1 + 0.25 \times (-22 + -20 + -14 + v_\pi(15))$

Solving for the value at 15 yields:

$v_\pi(15) = \frac{0.25 \times -56 - 1}{0.75}=-20$

In the second case, the value function at 13 and 15 become coupled because transitions back and forth are allowed.  We can write down new Bellman equations for the equiprobable policy π of these states:

$v_{\pi}(13) = -1 + \frac{1}{4}(v_{\pi}(9) + v_{\pi}(14) + v_{\pi}(12) + v_{\pi}(15))$
$v_{\pi}(15) = -1 + \frac{1}{4}(v_{\pi}(13) + v_{\pi}(14) + v_{\pi}(12) + v_{\pi}(15))$

In the second equation we can simplify to get an equation for state 15 in terms of just 3 others.

$v_{\pi}(15) \times \frac{3}{4} = -1 + \frac{1}{4}(v_{\pi}(13) + v_{\pi}(14) + v_{\pi}(12))$
$v_{\pi}(15) = \frac{1}{3}(-4 + v_{\pi}(13) + v_{\pi}(14) + v_{\pi}(12))$

Let's try to approximate the new value at state 15 by substituting in the known values of the unmodified states.

$v_{\pi}(15) \approx \frac{1}{3}(-4 - 20 - 14 - 22) = \frac{1}{3}(-60) = -20$

Now let's get an implied updated value at state 13 by substituting in the approximate value at 15.

$v_{\pi_{new}}(13) = -1 + \frac{1}{4}(-20 - 14 - 22 - 20) = -1 - \frac{76}{4} = -1 - 19 = -20=v_{\pi_{old}}(13)$

So we assumed that the value at state 13 was unchanged to get the approximation for state 15.  Then using the self consistency equation for state 13 we confirmed that the original value is consistent with the approximate solution.  This step of approximating the value at state 15 with a previous value function is analogous to what we would do in policy evaluation.  However, when checking the value of state 13 we see that it remains unchanged after using state 15.  If we were to carry this out for the other states that depend on 15, we would find that no futher changes are needed since 13 is the only state with a transition to 15 and states 12, 13, and 9 all now have new trasitions to 15 which would have been transitions to 13 previously.  But the value estimate at 15 is identical to the original value at 13.  This is the stopping condition for policy evaluation.  Indeed if we carry out the full policy evaluation calculation below using the same method used to generate Figure 4.1, we see a value of -20 which is equal to the original value of state 13.
"""

# ╔═╡ 10c9b166-3a88-460e-82e8-a16c020c1378
#Exercise 4.2 part 2
function gridworld_modified_mdp()
	states = collect(1:15)
	s_term = 0

	#no discounting in this episodic task
	γ = 1.0
	
	#define p by iterating over all possible states and transitions
	#there is 0 reward and a probability of 1 staying in the terminal state for all 	actions taken from the terminal state
	p = Dict((0, 0.0, 0, a) => 1.0 for a in gridworld_actions)

	#positions that result in terminal states
	termset(::North) = (4,)
	termset(::South) = (11,)
	termset(::East) = (14,)
	termset(::West) = (1,)

	#positions that leave the state unchanged
	constset(::North) = (1, 2, 3)
	constset(::South) = (12, 14, 15)
	constset(::East) = (3, 7, 11)
	constset(::West) = (4, 8, 12)

	#usual movement rule
	delta(::North) = -4
	delta(::South) = +4
	delta(::East) = +1
	delta(::West) = -1

	#special moves for added square 15
	move15(s, ::North) = 13
	move15(s, ::East) = 14
	move15(s, ::West) = 12

	move13(s, ::North) = 9
	move13(s, ::West) = 12
	move13(s, ::East) = 14
	move13(s, ::South) = 15
	
	function move(s, dir::GridworldAction)
		in(s, termset(dir)) && return s_term
		in(s, constset(dir)) && return s
		(s == 15) && return move15(s, dir)
		(s == 13) && return move13(s, dir)
		return s + delta(dir)
	end

	#add other transitions to p
	for s in states
		for a in gridworld_actions
			s′ = move(s, a)
			p[s′, -1.0, s, a] = 1.0
		end
	end
	sa_keys = get_sa_keys(p)
	return (p = p, sa_keys = sa_keys)
end		

# ╔═╡ 1618ec46-6e13-42bf-a7f2-68f8dbe3714c
function exercise4_2(nmax=Inf)
	gridworldmdp = gridworld_modified_mdp()
	π_rand = form_random_policy(gridworldmdp[2])
	V = iterative_policy_eval_v(π_rand, eps(0.0), gridworldmdp, 1.0, nmax = nmax)
	[(s, V[s]) for s in 0:15]
end

# ╔═╡ 11365717-6dee-461c-8ecf-485144b53a93
function show_modified_gridworld_values(values)
	"""
	<div id="gridcontainer">
	<div class="gridworld">
		$(makevaluecell(0.0))
		$(mapreduce(makevaluecell, linejoin, values[2:end-1]))
		$(makevaluecell(0.0))
		<div class = "blankcell"></div>
		$(makevaluecell(values[end]))
	</div>
	</div>
	"""
end

# ╔═╡ 68a01f8e-769b-4362-b12a-48733e8b8dba
@bind k_4_2_reset Button("Click to reset k to 1")

# ╔═╡ ab98efa4-c793-40eb-8c1c-70a0bb929ab3
begin
	k_4_2_reset
	@bind k_4_2 CounterButton("Click to increase k")
end

# ╔═╡ e4bfdaca-3f3d-43bb-b8aa-7536adbff662
begin
	k_4_2_latex = Markdown.parse(latexify(string("k = ", k_4_2+1)))
	#calculates value function for gridworld example in part 2 of exercise 4.2 with an added state 15
	md"""
	|Modified Gridworld|$k_4_2_latex|$$k = \infty$$|
	|:---:|:---:|:---:|
	| $gridworld_display_modified |$(HTML(show_modified_gridworld_values([round(a[2], sigdigits = 2) for a in exercise4_2(k_4_2+1)]))) |  $(HTML(show_modified_gridworld_values([round(a[2], sigdigits = 2) for a in exercise4_2()]))) |
	"""
end

# ╔═╡ 35e1ffe5-d36a-449d-aa73-c618e2855042
md"""
> ### *Exercise 4.3* 
> What are the equations analogous to (4.3), (4.4), and (4.5), but for *action*-value functions instead of state-value functions?

Equation (4.3)

$v_\pi(s)=\mathbb{E}_\pi [R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s]$

action-value equivalent

$q_\pi(s,a)=\mathbb{E}_\pi [R_{t+1} + \gamma q_\pi(S_{t+1},A_{t+1}) | S_t = s, A_t=a]$

Equation (4.4)

$v_\pi(s)=\sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma v_\pi(s')]$

action-value equivalent

$q_\pi(s,a)=\sum_{s',r} p(s',r|s,a)[r + \gamma \sum_{a'} \pi(a'|s') q_\pi(s',a')]$

Equation (4.5)

$v_{k+1}(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma v_k(s')]$

action-value equivalent

$q_{k+1}(s,a) = \sum_{s',r} p(s',r|s,a)[r + \gamma \sum_{a'} \pi(a'|s') q_k(s',a')]$
"""

# ╔═╡ aa2e7334-af07-4152-8f21-e80bdcdd979b
md"""
## 4.2 Policy Improvement

From the value function $v_\pi$ we can systematically find a better policy $$\pi^\prime$$.  First consider a new policy with the following property:

$$q_\pi(s, \pi ^\prime(s)) \geq v_\pi(s) \tag{4.7}$$

If this is true for all $s \in \mathcal{S}$ then the policy $\pi^\prime$ must be as good or better than $\pi$ meaning it has a greater or equal expected return at every state:

$v_{\pi^\prime}(s) \geq v_\pi(s) \tag{4.8}$

Starting with $\pi$ consider a new policy that chooses action $a$ at state $s$ instead of the usual action:  $\pi^\prime(s) = a \neq \pi(s)$.  If $q_\pi(s, a) > v_\pi(s)$, then this new policy is better than $\pi$ since $v_{\pi^\prime}(s) \geq q_\pi(s, a) > v_\pi(s)$.  This relationship is shown in the proof of the policy improvement theorem which relies upon expanding out the expression for $q_\pi$ and repeatedly applying the inequality (4.7).  

One way of creating such a policy is using action-value function of the original policy:

$\begin{flalign}
\pi^\prime(s) &\doteq \underset{a}{\mathrm{argmax}} \: q_\pi(s, a) \\
&= \underset{a}{\mathrm{argmax}} \: \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t = a] \tag{4.9}\\
&= \underset{a}{\mathrm{argmax}} \: \sum_{s^\prime,r} p(s^\prime, r \vert s, a) \left [r + \gamma v_\pi(s) \right ] \\
\end{flalign}$

By construction, the greedy policy with respect to the action-value function for $\pi$ meets the criteria of the policy improvement theorem (4.7), so we know that it is as good or better than the original policy.

Suppose that our new policy $\pi^\prime$ is equally good to the original policy.  Then using (4.9):

$\begin{flalign}
v_{\pi^\prime}(s) &= \underset{a}{\mathrm{max}} \: \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t = a] \tag{4.9}\\
&= \underset{a}{\mathrm{max}} \: \sum_{s^\prime,r} p(s^\prime, r \vert s, a) \left [r + \gamma v_\pi(s) \right ] \\
\end{flalign}$


This is equal to the Bellman optimality equation (4.1), therefore $v_{\pi^\prime} = v_*$ and both $\pi$ and $\pi^\prime$ must be optimal policies.  So if we carry out this improvement procedure to the point where it converges and the value function remains unchanged, then we are guaranteed to compute the optimal policy.  Sometimes the greedy policy with respect to the equiprobable random policy is already optimal as in the case of the gridworld in example 4.1.  But in general many iterations could be required.
"""

# ╔═╡ 67b06f3b-13df-4b27-ad80-d112432e8f42
md"""
## 4.3 Policy Iteration

Once we've improved our policy to $\pi^\prime$ we can repeat the procedure to have a sequence of monotonically improving policies and value functions.

$\pi_0 \overset{\text{E}}{\longrightarrow}v_{\pi_0}\overset{\text{I}}{\longrightarrow}\pi_1 \overset{\text{E}}{\longrightarrow}v_{\pi_1}\overset{\text{I}}{\longrightarrow}\pi_2 \overset{\text{E}}{\longrightarrow} \cdots \overset{\text{I}}{\longrightarrow}\pi_* \overset{\text{E}}{\longrightarrow}v_*$

where $\overset{\text{E}}{\longrightarrow}$ denotes a policy *evaluation* and $\overset{\text{I}}{\longrightarrow}$ denotes a policy *improvement*.  Every policy is guaranteed to be a strict improvement over the previous one unless it is already optimal because the action selection at least at one state must be different.  If all the action selections are the same, then the process has converged.  This method of completing a full policy evaluation between steps of selecting the greedy policy is called *policy iteration$.  Code for carrying out policy iteration are shown below.
"""

# ╔═╡ 87718a9d-5624-4f18-9dbc-34458dd917fd
md"""
### Policy Iteration Implementation
"""

# ╔═╡ 160c59b0-a5ea-4046-b79f-7a6a6fc8db7e
#computes the greedy policy with respect to a given value function.  note that the full probability transition function for the MDP must be known.
function greedy_policy(mdp::NamedTuple, V::Dict, γ::Real)
	(p, sa_keys) = mdp
	Dict(begin
		actions = sa_keys[1][s]
		newdist = Dict(a => 
				sum(p[(s′,r,s,a)] * (r + γ*V[s′]) for (s′,r) in sa_keys[2][(s,a)])
				for a in actions)
		s => newdist
	end
	for s in keys(sa_keys[1]))
end

# ╔═╡ c078f6c3-7576-4933-bc95-d33e8193ee93
function make_4_1_row(k)
	v = makefig4_1(k)
	π = convertpolicy(greedy_policy(gridworld4x4_mdp(), Dict(v), 1.0))
	# """
	# | $(latexify(string("k = ", k))) | $(HTML(show_gridworld_values([round(a[2], sigdigits = 2) for a in v[2:end]]))) | $(HTML(show_gridworld_policy(π))) | $(latexify("longleftarrow")) random policy|
	# """

	ktext = Markdown.parse(latexify(string("k = ", k)))

	valuegrid = HTML(show_gridworld_values([round(a[2], sigdigits = 2) for a in v[2:end]]))
	policygrid = HTML(show_gridworld_policy(π))

	(ktext, valuegrid, policygrid)
end

# ╔═╡ 1b8bfddb-97c3-4756-ab8e-123d38afda64
function make_4_1_interactive(k = 1)
	results = [make_4_1_row(k) for k in [0, k, typemax(Int64)]]
	md"""
	| | $$v_k$$ for the random policy | greedy policy w.r.t. $$v_k$$ | |
	|:---:|:----:|:---:|--- |
	|$(results[1][1])|$(results[1][2])|$(results[1][3])|$$\longleftarrow$$ random policy|
	|$(results[2][1])|$(results[2][2])|$(results[2][3])||
	|$$k = \infty$$|$(results[3][2])|$(results[3][3])|$$\longleftarrow$$ optimal policy|
	"""
end

# ╔═╡ 38875afb-f8a3-4b8f-be7f-a34cc19efa7d
make_4_1_interactive(user_k+1)

# ╔═╡ b3ed0348-3d74-4726-878f-5eefcb1d72d0
function make_4_1_table(klist = [0, 1, 2, 3, 10, typemax(Int64)])
	results = [make_4_1_row(k) for k in klist]
	results[1][3]
	md"""
	| | $$v_k$$ for the random policy | greedy policy w.r.t. $$v_k$$ | |
	|:---:|:----:|:---:|--- |
	|$(results[1][1])|$(results[1][2])|$(results[1][3])|$$\longleftarrow$$ random policy|
	|$(results[2][1])|$(results[2][2])|$(results[2][3])||
	|$(results[3][1])|$(results[3][2])|$(results[3][3])||
	|$(results[4][1])|$(results[4][2])|$(results[4][3])||
	|$(results[5][1])|$(results[5][2])|$(results[5][3])||
	|$$k = \infty$$|$(results[6][2])|$(results[6][3])|$$\longleftarrow$$ optimal policy|
	"""
end

# ╔═╡ dae71267-9945-41d2-bec4-546c8c883ae0
make_4_1_table()

# ╔═╡ 56184148-aaad-470c-b79c-30b952e1142d
function make_greedy_policy!(π::Matrix{T}, mdp::FiniteMDP{T, S, A}, V::Vector{T}, γ::T) where {T<:Real,S,A}
	for i_s in eachindex(mdp.states)
		maxv = -Inf
		for i_a in eachindex(mdp.actions)
			x = zero(T)
			for i_r in eachindex(mdp.rewards)
				for i_s′ in eachindex(V)
					x += mdp.ptf[i_s′, i_r, i_a, i_s] * (mdp.rewards[i_r] + γ * V[i_s′])
				end
			end
			maxv = max(maxv, x)
			π[i_a, i_s] = x
		end
		π[:, i_s] .= (π[:, i_s] .≈ maxv)
		π[:, i_s] ./= sum(π[:, i_s])
	end
	return π
end

# ╔═╡ 8b4fb649-8aaa-4e17-8204-540caf8da343
function policy_improvement_v!(π::Matrix{T}, mdp::FiniteMDP{T, S, A}, γ::Real, V::Vector{T}) where {T<:Real, S, A}
	policy_stable = true
	dist = zeros(T, length(mdp.actions))
	for i_s in eachindex(mdp.states)
		maxv = -Inf
		for i_a in eachindex(mdp.actions)
			x = zero(T)
			for i_r in eachindex(mdp.rewards)
				for i_s′ in eachindex(V)
					x += mdp.ptf[i_s′, i_r, i_a, i_s] * (mdp.rewards[i_r] + γ * V[i_s′])
				end
			end
			maxv = max(maxv, x)
			dist[i_a] = x
		end
		dist .= (dist .≈ maxv)
		dist ./= sum(dist)
		for i_a in eachindex(mdp.actions)
			policy_stable = policy_stable && (sign(π[i_a, i_s]) == sign(dist[i_a]))
			π[i_a, i_s] = dist[i_a]
		end
		
	end
	return (policy_stable, π)
end 

# ╔═╡ f0e5d2e6-3d00-4ffc-962e-e98d4bb28e4e
function policy_improvement_v(π::Dict, mdp::NamedTuple, γ::Real, V::Dict)
	(p, sa_keys) = mdp
	π_new = Dict(begin
		actions = sa_keys[1][s]
		newdist = Dict(a => 
				sum(p[(s′,r,s,a)] * (r + γ*V[s′]) for (s′,r) in sa_keys[2][(s,a)])
				for a in actions)
		new_action = argmax(newdist)
		s => Dict(new_action => 1.0)
	end
	for s in keys(sa_keys[1]))

	policy_stable = mapreduce((a,b) -> a && b, keys(sa_keys[1])) do s
		argmax(π[s]) == argmax(π_new[s])
	end

	return (policy_stable, π_new)
end 

# ╔═╡ 31a9db21-c7d2-4053-8c18-2023f4720196
function policy_iteration_v(mdp::FiniteMDP{T, S, A}, π::Matrix{T}, γ::Real, Vold::Vector{T}, iters, θ, evaln, policy_stable, resultlist) where {T<:Real, S, A}
	policy_stable && return (true, resultlist)
	V = iterative_policy_eval_v(π, θ, mdp, γ, Vold, nmax = evaln)
	push!(resultlist, (copy(V), copy(π)))
	(V == resultlist[end-1][1]) && return (true, resultlist)
	(iters <= 0) &&	return (false, resultlist)
	(new_policy_stable, π_new) = policy_improvement_v!(π, mdp, γ, V)
	policy_iteration_v(mdp, π_new, γ, V, iters-1, θ, evaln, new_policy_stable, resultlist)
end

# ╔═╡ 4d15118f-f1ab-4115-bcc9-7f98246eca1c
function policy_iteration_v(mdp::NamedTuple, π::Dict, γ::Real, Vold::Dict, iters, θ, evaln, policy_stable, resultlist)
	policy_stable && return (true, resultlist)
	V = iterative_policy_eval_v(π, θ, mdp, γ, Vold, nmax = evaln)
	(V == resultlist[end][1]) && return (true, resultlist)
	newresultlist = vcat(resultlist, (V, π))
	(iters <= 0) &&	return (false, newresultlist)
	(new_policy_stable, π_new) = policy_improvement_v(π, mdp, γ, V)
	policy_iteration_v(mdp, π_new, γ, V, iters-1, θ, evaln, new_policy_stable, newresultlist)
end

# ╔═╡ 77250f6b-60d1-426f-85b2-497186b86c50
function begin_policy_iteration_v(mdp::NamedTuple, π::Dict, γ::Real; iters=Inf, θ=eps(0.0), evaln = Inf, V = iterative_policy_eval_v(π, θ, mdp, γ, nmax = evaln))
	resultlist = [(V, π)]
	(policy_stable, π_new) = policy_improvement_v(π, mdp, γ, V)
	policy_iteration_v(mdp, π_new, γ, V, iters-1, θ, evaln, policy_stable, resultlist)
end

# ╔═╡ 7021bc9e-716a-43ff-b2cb-40b499ae2706
function begin_policy_iteration_v(mdp::FiniteMDP{T, S, A}, π::Matrix{T}, γ::Real; iters=Inf, θ=eps(zero(T)), evaln = Inf, V = iterative_policy_eval_v(π, θ, mdp, γ, nmax = evaln)) where {T<:Real, S, A}
	resultlist = [(copy(V), copy(π))]
	(policy_stable, π_new) = policy_improvement_v!(π, mdp, γ, V)
	policy_iteration_v(mdp, π_new, γ, V, iters-1, θ, evaln, policy_stable, resultlist)
end

# ╔═╡ bb5fb8e6-0163-4fde-8238-63454f1c5128
md"""
### Gridworld Example
"""

# ╔═╡ c47882c7-ded1-440c-a9a3-0b89a0e7a011
function gridworld_policy_iteration(nmax=10; θ=eps(0.0), γ=1.0)
	gridworldmdp = gridworld4x4_mdp()
	π_rand = form_random_policy(gridworldmdp[2])
	(policy_stable, resultlist) = begin_policy_iteration_v(gridworldmdp, π_rand, γ, iters = nmax)
	(Vstar, πstar) = resultlist[end]
	(policy_stable, Vstar, [(s, first(keys(πstar[s]))) for s in 0:14])
end

# ╔═╡ ec5afe79-1b1b-4ccd-b672-450797b4e73e
function gridworld_policy_iteration_v2(nmax=10; θ=eps(0.0), γ=1.0)
	gridworldmdp = create_4x4gridworld_mdp()
	π_rand = form_random_policy(gridworldmdp)
	(policy_stable, resultlist) = begin_policy_iteration_v(gridworldmdp, π_rand, γ, iters = nmax)
	(Vstar, πstar) = resultlist[end]
	(policy_stable, Vstar, πstar, resultlist)
end

# ╔═╡ 0079b02d-8895-4dd4-9557-5f08ac341404
#seems to match optimal policy from figure 4.1
gridworld_policy_iteration_results = gridworld_policy_iteration()

# ╔═╡ 77d251d3-903b-4e96-9261-77a429a3eda7
gridworld_policy_iteration_results_v2 = gridworld_policy_iteration_v2()

# ╔═╡ a5174afc-04f2-4fc9-9b10-7aa7f249332a
md"""
Note that the value for each square matches the $L_1$ distance from the corner as is expected for the optimal policy.  The last row represents the optimal policy.  Note that only required 1 iteration to have the optimal value function.  The next iteration is shown to confirm that the values are the same.  The policy in states 6 and 9 actually change to being equiprobable instead of favoring just two directions, but any policy that simply selects one of these actions is also optimal.

|Policy Iteration Step|$v$|$\pi$|
|:---:|:---:|:---:|
|1|$(HTML(show_gridworld_values(round.(gridworld_policy_iteration_results_v2[4][1][1][1:end-1], sigdigits = 2))))| $(HTML(show_gridworld_policy(eachcol(gridworld_policy_iteration_results_v2[4][1][2]))))|
|2|$(HTML(show_gridworld_values(round.(gridworld_policy_iteration_results_v2[4][2][1][1:end-1], sigdigits = 2))))| $(HTML(show_gridworld_policy(eachcol(gridworld_policy_iteration_results_v2[4][2][2]))))|
|3|$(HTML(show_gridworld_values(gridworld_policy_iteration_results_v2[2][1:end-1])))| $(HTML(show_gridworld_policy(eachcol(gridworld_policy_iteration_results_v2[3]))))|
"""

# ╔═╡ c4a14f1c-ce17-40eb-b9ae-00b651d40714
md"""
### Example 4.2: Jack's Car Rental
"""

# ╔═╡ c3f4764f-b2e3-4004-b5ed-2f1cccd2cdde
#define car rental MDP with states, actions, probabilities as a tensor.  Use code below for poisson distribution etc..
md"""
- Each step is one day starting at the end of a given day
- State space: Number of cars at each location at end of day.  Refer to the locations as A and B
- Action space: Movement of cars from one location to another.  Maximum of 5 cars can be transfered in either direction.  Consider positive numbers to indicate transfering from A to B and negative numbers to mean transfering from B to A.  This way there are 11 actions from -5 to 5
- A maximum of 20 cars can be at any location, any extra cars are removed from the problem.
- During the day cars can be requested and returned at each location following a poisson distribution with the following constants: 
$p(n) = \frac{\lambda^n}{n!} e^{\lambda}$
$\lambda_{request, A} = 3, \: \lambda_{request, B}=4, \: \lambda_{return, A} = 3, \: \lambda_{return, B}=2$

- Returned cars are not available until the next day
- Continuing problem with $\gamma = 0.9$
- There are $21^2 = 441 states since each location could have 0 to 20 cars at the end of the day.
- \$10 positive reward for each car that is rented.  Note that if more cars are requested than are available, only the total number of available cars can be rented.
- \$2 cost for each car moved
"""

# ╔═╡ e722b7e0-63a3-4195-b13e-0449abb3cc39
poisson(n, λ) = exp(-λ) * (λ^n) / factorial(n)

# ╔═╡ bd41fffb-5d8c-4165-9f44-690f81c70113
function create_car_rental_mdp(;nmax=20, λs::@NamedTuple{request_A::T, request_B::T, return_A::T, return_B::T} = (request_A = 3f0, request_B = 4f0, return_A = 3f0, return_B = 2f0), movecost::T = 2f0, rentcredit::T = 10f0, movemax::Integer=5, maxovernight::Integer = 20, overnightpenalty::T = 4f0, employeeshuttle = false) where T <: Real
	#enumerate all states
	states = [(n_a, n_b) for n_a in 0:nmax for n_b in 0:nmax]
	
	actions = collect(-movemax:movemax)

	#enumerate all rewards by simply incrementing by 1 dollar from the worst to best case scenario
	rewards = collect(-movecost*movemax - 2*overnightpenalty:rentcredit*nmax*2)
	reward_lookup = Dict(zip(rewards, eachindex(rewards))) #mapping from rewards to the proper index

	#create a lookup for the probability of starting with n cars at the start of the day and ending up with n′ at the end of the day
	function create_probability_lookup(λ_request, λ_return)
		#can only rent from 0 to n cars.  if requests exceed n, all of those situations are equivalent and the probability is 1 - p(x < n-1)
		p_rent = Dict(n_request => poisson(n_request, λ_request) for n_request in 0:nmax-1)
	
		#car returns can be any number greater than or equal to 0, but all returns of nmax - (n - nrent) or more will result in the same state which is max cars
		p_return = Dict(n_return => poisson(n_return, λ_return) for n_return in 0:nmax-1)
		
		#initialize probabilities for each final value at 0
		prob_lookup = Dict((t, nrent) => 0f0 for t in states for nrent in 0:t[1])
			
		for n in 0:nmax
			for n_rent in 0:n-1
				for n_return in 0:(nmax - n + n_rent - 1)
					n′ = n - n_rent + n_return
					p = p_rent[n_rent]*p_return[n_return]
					prob_lookup[((n, n′), n_rent)] += p
				end
				prob_lookup[((n, nmax), n_rent)] += p_rent[n_rent]*(1 - sum(p_return[n_return] for n_return in 0:nmax-n+n_rent-1; init = zero(T)))
			end
			for n_return in 0:(nmax - 1)
				n′ = n_return
				p = (1 - sum(p_rent[n_rent] for n_rent in 0:n-1; init = zero(T)))*p_return[n_return]
				prob_lookup[((n, n′), n)] += p
			end
			prob_lookup[((n, nmax), n)] += (1 - sum(p_rent[n_rent] for n_rent in 0:n-1; init = zero(T)))*(1 - sum(p_return[n_return] for n_return in 0:nmax-1, init = zero(T)))
		end
		return prob_lookup
	end

	probabilities = (location_A = create_probability_lookup(λs.request_A, λs.return_A), location_B = create_probability_lookup(λs.request_B, λs.return_B))

	#calculate probability matrix for all the s′, r transitions given starting in state s and taking action a
	function getmatrix(s, a)
		#initialize the matrix for s′, r transitions, each column runs over the transition states
		out = zeros(length(states), length(rewards))
		(n_a, n_b) = s

		#calculate the number of cars moved with sign indicating direction + being A to B, normally this is simply a but if we try to move more cars than are available, it will be capped
		carsmoved = if a > 0
			min(a, n_a)
		elseif a < 0
			-min(abs(a), n_b)
		else
			0
		end
		
		#cars above nmax are returned to the company but we still incur the cost of transfering them
		aftercount_a = min(n_a - carsmoved, nmax)
		aftercount_b = min(n_b + carsmoved, nmax)

		cost = (abs(a) - (a > 0)*employeeshuttle)*movecost + (overnightpenalty * ((aftercount_a > maxovernight) + (aftercount_b > maxovernight))) #one free transfer from A to B if employee shuttle is true in modified version, overnight penalty if too many cars are left at a lot
		for (i_s′, s′) in enumerate(states)
			(n_a′, n_b′) = s′
			for n_rent_a in 0:aftercount_a
				for n_rent_b in 0:aftercount_b
					p_a = probabilities.location_A[((aftercount_a, n_a′), n_rent_a)]
					p_b = probabilities.location_B[((aftercount_b, n_b′), n_rent_b)]
					p_total = p_a*p_b
					r = rentcredit*(n_rent_a+n_rent_b) - cost
					out[i_s′, reward_lookup[r]] += p_total
				end
			end
		end
		return out
	end

	#initialize probability function with all zeros
	ptf = zeros(T, length(states), length(rewards), length(actions), length(states))
	for (i_s, s) in enumerate(states)
		for (i_a, a) in enumerate(actions)
			ptf[:, :, i_a, i_s] .= getmatrix(s, a)
		end
	end

	#find indices of the reward vector that never have non zero probability
	inds = reduce(intersect, [findall(0 .== [sum(ptf[:, i, j, k]) for i in 1:size(ptf, 2)]) for j in 1:size(ptf, 3) for k in 1:size(ptf, 4)])

	goodinds = setdiff(eachindex(rewards), inds)
	
	FiniteMDP(states, actions, rewards[goodinds], ptf[:, goodinds, :, :])
end

# ╔═╡ 871d78a4-cc74-4866-896b-027a6c626676
const car_rental_mdp_v2 = create_car_rental_mdp()

# ╔═╡ eb8fa5e6-c18d-41ba-a78e-cceccc90d9b6
function car_rental_policy_iteration_v2(mdp, nmax=10; θ=eps(0f0), γ=0.9f0)
	movezeroind = findfirst(mdp.actions .== 0)
	zerodist = [i == movezeroind ? 1f0 : 0f0 for i in eachindex(mdp.actions)]
	π0 = mapreduce(i -> zerodist, hcat, eachindex(mdp.states))
	# iterative_policy_eval_v(π_rand, θ, mdp, γ)
	(policy_stable, resultlist) = begin_policy_iteration_v(mdp, π0, γ; iters = nmax, θ = θ)
	(Vstar, πstar) = resultlist[end]
	(policy_stable, Vstar, πstar, resultlist)
end

# ╔═╡ 05512261-d0c7-4602-8280-cd1d4d45e875
example4_2_results = car_rental_policy_iteration_v2(car_rental_mdp_v2;θ = .001f0)

# ╔═╡ c4835f94-1ebc-43bf-b54a-5252e4280635
function makepolicyvaluemaps(mdp::FiniteMDP, v::Vector{T}, π::Matrix{T}) where T <: Real
	function getaction(dist)
		#default action will be 0
		sum(dist) == 0 && return 0
		(p, ind) = findmax(dist)
		mdp.actions[ind]
	end
	policymap = zeros(Int64, 21, 21)
	valuemap = zeros(T, 21, 21)
	for i in 1:size(π, 2)
		action = getaction(view(π, :, i))
		(n_a, n_b) = mdp.states[i]
		policymap[n_a+1, n_b+1] = action
		valuemap[n_a+1, n_b+1] = v[i]
	end
	(policymap, valuemap)
end

# ╔═╡ 8b0e875b-2644-4f52-83a0-a736e9330e78
function makepolicyvalueplots(mdp::FiniteMDP, v::Vector{T}, π::Matrix{T}, iter::Integer; policycolorscale = "RdBu", valuecolorscale = "Bluered", kwargs...) where T <: Real
	(policymap, valuemap) = makepolicyvaluemaps(mdp, v, π)
	layout = Layout(autosize = false, height = 220, width = 230, paper_bgcolor = "rgba(30, 30, 30, 1)", margin = attr(l = 0, t = 0, r = 0, b = 0, padding = 0), xaxis = attr(title = attr(text = "# Cars at second location", font_size = 10, standoff = 1, automargin = true), tickvals = [0, 20], linecolor = "white", mirror = true, linewidth = 2, yanchor = "bottom"), yaxis = attr(title = attr(text = "# Cars at first location", standoff = 1, automargin = true, pad_l = 0), tickvals = [0, 20], linecolor = "white", mirror = true, linewidth = 2), font_color = "gray", font_size = 9)
	
	function makeplot(z, colorscale; kwargs...) 
		tr = heatmap(;x = 0:20, y = 0:20, z = z, colorscale = colorscale, colorbar_thickness = 2)
		plot(tr, layout)
	end
	vtitle = L"v_{\pi_{%$(iter-1)}}"
	policyplot = relayout(makeplot(policymap, policycolorscale), (title = attr(text =  latexify("π_$(iter-1)"), x = 0.5, xanchor = "center", font_size = 20, automargin = true, yref = "paper", yanchor = "bottom", pad_b = 10)))
	valueplot = relayout(makeplot(valuemap, valuecolorscale), (title = attr(text = vtitle, x = 0.5, xanchor = "center", font_size = 20, automargin = true, yref = "paper", yanchor = "bottom", pad_b = 10)))
	
	# (π = relayout(policyplot, kwargs), v = relayout(valueplot, kwargs))
	(π = relayout(policyplot, kwargs), v = makeplot(valuemap, valuecolorscale))
end

# ╔═╡ e7f3cc36-56a7-4592-b3f6-05001675cc14
function plotcariterations(mdp, resultslist; kwargs...)
	[makepolicyvalueplots(mdp, value, policy, i; kwargs...) for (i, (value, policy)) in enumerate(resultslist)]
end

# ╔═╡ e94c14ac-9583-4b31-a392-996dcb6d79b7
carplots = plotcariterations(car_rental_mdp_v2, example4_2_results[4]);

# ╔═╡ 26566af8-152f-4553-b26b-303dff3d2f24
function figure4_2(carplots)
	l = L"\pi"
	function htljoin(a, b)
		@htl("""
		$a
		$b
		""")
	end
	cardivs = mapreduce(p -> @htl("""<div>$p</div>"""), htljoin, [a.π for a in carplots])
	
	@htl(
	"""
	<div class = "carplots">
		$cardivs
		<div>$(last(carplots).v)</div>
	</div>
	<style>
		.carplots {
			display: flex;
			flex-wrap: wrap;
			width: 700px;
		}
	</style>
	""")
end	

# ╔═╡ 3e1a1559-6dc0-46a1-a639-d8655b72e740
figure4_2(carplots)

# ╔═╡ 0d6936d1-38af-45f1-b496-da49b60f11f8
md"""
> ### *Exercise 4.4* 
> The policy iteration algorithm on page 80 has a subtle bug in that it may never terminate if the policy continually switches between two or more policies that are equally good.  This is okay for pedagogy, but not for actual use.  Modify the pseudocode so that convergence is guaranteed.

Initialize $V_{best}$ at the start randomly and replace it with the first value function calculated.  After each policy improvement, replace $V_{best}$ with the new value function, however add a check after step 2. that if the value function is the same as $V_{best}$ then stop.  This would ensure that no matter how many equivalent policies are optimal, they would all share the same value function and thus trigger the termination condition.
"""

# ╔═╡ ad1bf1d2-211d-44ca-a258-fc6e112785da
md"""
> ### *Exercise 4.5* 
> How would policy iteration be defined for action values?  Give a complete algorithm for computer $q_*$, analogous to that on page 80 for computing $v_*$.  Please pay special attention to this exercise, because the ideas involved will be used throughout the rest of the book.

**Policy Iteration (using iterative policy evaluation) for estimating $\pi \approx \pi_*$ using action-values**

1. Initialization
   * $Q(s,a) \in \mathbb{R} \text{ and } \pi(s) \in \mathcal{A}(s) \text{ arbitrarily for all } s \in \mathcal{S}; Q(terminal,a) \doteq 0 \space \forall \space a \in \mathcal{A}$

2. Policy Evaluation
   
   * Loop:
     *  $\Delta \leftarrow 0$

     * Loop for each $s \in \mathcal{S}$:

       * Loop for each $a \in \mathcal{A}(s)$:
$\begin{flalign} 
& \hspace{4em} q \leftarrow Q(s,a) \\
& \hspace{4em} Q(s,a) \leftarrow \ \sum_{s',r} p(s',r|s,a)[r + \gamma \sum_{a'} \pi(s',a') Q(s',a')] \\
& \hspace{4em} \Delta \leftarrow \max{(\Delta, |q - Q(s,a)|)}\\
\end{flalign}$
*
   * until $\Delta < \theta$ (a small positive number determining the accuracy of estimation)

3. Policy Improvement
   * *policy-stable* $\leftarrow$ *true*

   * For each $s \in \mathcal{S}$:

      *  $old-action \leftarrow \pi(s)$

      *  $\pi(s) \leftarrow \text{argmax}_a Q(s,a)$

      * If *old-action* $\neq \pi(s)$, then *policy-stable* $\leftarrow false$

   * If *policy-stable*, then stop and return $Q \approx q_*$ and $\pi \approx \pi_*$; else go to 2
"""

# ╔═╡ e316f59a-8070-4510-96f3-15498897347c
md"""
> ### *Exercise 4.6* 
> Suppose you are restricted to considering only policies that are $\epsilon \textendash soft$, meaning that the probability of selecting each action in each state, s, is at least $\epsilon / |\mathcal{A}(s)|$.  Describe qualitatively the changes that would be required in each of the steps 3,2,and 1, in that order, of the policy iteration algorithm for $v_*$ on page 80.

- For step 3 inside the loop over states: 
   -  $old \textendash action \textendash distribution \leftarrow \pi(s)$ which returns a probability distribution accross actions. 
   - Calculate $q(s, a) = \sum_{s^\prime, r} p(s^\prime, r \vert s, a)[r + \gamma V(s^\prime)]$ for each action  
   - Find the maximum value of $q(a | s)$ and create a policy distribution with equal probability weight on each maximizing action.  That means if there is a unique maximum, 100% probability weight on that action and if there are N equal maxima then each will have a weight of 1/N.  Call this distribution $\pi_{greedy}$
   - In the $\epsilon \textendash soft$ policy, there is a probability $\epsilon$ of a random action and a probability $1-\epsilon$ to follow the greedy policy, so to get the $\epsilon \textendash soft$ distribution we must calculate: $\pi(a \vert s) = (1-\epsilon) \times \pi_{greedy}(a \vert s) + \frac{\epsilon}{\vert \mathcal{A}(s) \vert }$ and $\pi_{greedy}$ was described above.

- For step 2 inside the loop over states:
   - Replace the state value update with: $V(s) \leftarrow \sum_a \pi(a \vert s) \sum_{s^\prime, r} p(s^\prime, r \vert s, a)[r + \gamma V(s^\prime)$

- For step 1:

   - The policy must be initialized as a distribution over actions for each state.  One natrual choice is just the random policy so $\pi(a \vert s) = \pi_{rand}(a \vert s) = \frac{1}{\vert \mathcal{A}(s) \vert} \: \forall s \in \mathcal{S}$
"""

# ╔═╡ 5dbfb100-49a8-4f9d-a752-bda4da54699e
md"""
> ### *Exercise 4.7 (programming)* 
> Write a program for policy iteration and re-solve Jack's car rental problem with the following changes.  One of Jack's employees at the first location rides a bus home each night and lives near the second location. She is happy to shuttle one car to the second location for free. Each additional car still costs $2, as do all cars moved in the other direction. In addition, Jack has limited parking space at each location.  If more than 10 cars are kept overnight at a location (after any moving of cars), then an additional cost of $4 must be incurred to use a second parking lot (independent of how many cars are kept there). These sorts of nonlinearities and arbitrary dynamics often occur in real problems and cannot easily be handled by optimization methods other than dynamic programming. To check your program, first replicate the results given for the original problem.
"""

# ╔═╡ 2360169d-291b-4976-815a-8092c89260a8
modified_car_rental_mdp = create_car_rental_mdp(; employeeshuttle = true, maxovernight = 10)

# ╔═╡ 45fe9d02-5b77-4dc4-868d-1c598121cd90
exercise4_7_results = car_rental_policy_iteration_v2(modified_car_rental_mdp; θ = 0.001f0)

# ╔═╡ 13ca35da-7f61-4a89-98ba-6418d754707f
modifiedcarplots = plotcariterations(modified_car_rental_mdp, exercise4_7_results[4]);

# ╔═╡ 19e7b5c2-a32d-47e7-a436-f95fd3397ac5
figure4_2(modifiedcarplots)

# ╔═╡ 2bedc22e-9615-4fb4-94bf-6a0e7114c417
md"""
## 4.4 Value Iteration

One drawback of policy iteration is that each iteration requires policy evaluation to converge.  We can truncate policy evaluation without losing the convergence guarantees.  One special case is when we stop policy iteration after just one sweep through the state space.  This is called value iteration and has one simple update operation that continues until the value function has converged:

$\begin{flalign}
v_{k+1} & \doteq \underset{a}{\mathrm{max}} \: \mathbb{E} \left [ R_{t+1} + \gamma v_k (S_{t+1} \mid S_t = s, A_t = a) \right ] \\
& = \underset{a}{\mathrm{max}} \sum_{s^\prime, r} p(s^\prime, r \vert s, a) \left [ r + \gamma v_k(s^\prime) \right ] \tag{4.10}
\end{flalign}$

If we compare this to the Bellman optimality equation (4.1), we can see that value iteration simply turns that equation into an update rule.  When this process converges, we are guaranteed to have the optimal value function since it satisfies the optimality equation.

Below are implementation examples of value iteration as well as results from the method being applied to the two previous examples in this chapter, the gridworld and Jack's Car Rental.
"""

# ╔═╡ dde0354c-81cb-4898-85ac-723ec7346116
md"""
### Value Iteration Implementation
"""

# ╔═╡ 7140970d-d4a7-45bc-9626-26cf1a2f945b
function bellman_optimal_value!(V::Dict, p::Dict, sa_keys::Tuple, γ::Real)
	delt = 0.0
	for s in keys(sa_keys[1])
		v = V[s]
		actions = sa_keys[1][s]
		V[s] = maximum(sum(p[(s′,r,s,a)] * (r + γ*V[s′]) for (s′,r) in sa_keys[2][(s,a)]) for a in actions)
		delt = max(delt, abs(v - V[s]))
	end
	return delt
end

# ╔═╡ 09a98ebf-4685-404b-b492-116c184d4a52
function compute_Q(V::Vector{T}, i_s::Integer, i_a::Integer, mdp::FiniteMDP{T, S, A}, γ::T) where {T <: Real, S, A}
	#compute the Q value for a given state action pair and value function
	x = zero(T)
	for (i_r, r) in enumerate(mdp.rewards)
		@inbounds @fastmath @simd for i_s′ in eachindex(V)
			x += mdp.ptf[i_s′, i_r, i_a, i_s] * (r + γ * V[i_s′])
		end
	end
	return x
end

# ╔═╡ 64dc17e3-a83f-4414-a74a-c6c8e3e498fa
compute_V(π::Matrix{T}, Q::Matrix{T}) where T <: Real = Q' * π

# ╔═╡ 16adacf1-2757-4a45-a2e5-8ec4bf04584e
function update_V!(V::Vector{T}, π::Matrix{T}, Q::Matrix{T}) where T <: Real 
	gemm!('T', 'N', one(T), Q, π, zero(T), V)
end

# ╔═╡ 057beec8-0ad7-4bf2-ac45-6753614c0b4d
function bellman_optimal_value!(V::Vector{T}, mdp::FiniteMDP{T, S, A}, γ::T) where {T <: Real, S, A}
	delt = zero(T)
	@inbounds @fastmath @simd for i_s in eachindex(mdp.states)
		maxvalue = typemin(T)
		@inbounds @fastmath @simd for i_a in eachindex(mdp.actions)
			x = zero(T)
			for (i_r, r) in enumerate(mdp.rewards)
				@inbounds @fastmath @simd for i_s′ in eachindex(V)
					x += mdp.ptf[i_s′, i_r, i_a, i_s] * (r + γ * V[i_s′])
				end
			end
			maxvalue = max(maxvalue, x)
		end
		delt = max(delt, abs(maxvalue - V[i_s]) / (eps(abs(V[i_s])) + abs(V[i_s])))
		V[i_s] = maxvalue
	end
	return delt
end

# ╔═╡ fa1849cb-663f-49c0-acc1-f7fdbcbb4189
function value_iteration_v!(V::Vector{T}, θ::Real, mdp::FiniteMDP{T, S, A}, γ::T, nmax::Real, valuelist) where {T<:Real, S, A}
	nmax <= 0 && return valuelist
	
	#update value function
	delt = bellman_optimal_value!(V, mdp, γ)
	
	#add copy of value function to results list
	push!(valuelist, copy(V))

	#halt when value function is no longer changing
	delt <= θ && return valuelist
	
	value_iteration_v!(V, θ, mdp, γ, nmax - 1, valuelist)	
end

# ╔═╡ 9c5144fa-9138-4d20-b1da-b69c6f7cae4c
function begin_value_iteration_v(mdp::FiniteMDP{T,S,A}, γ::T, V::Vector{T}; θ = eps(0.0), nmax=typemax(Int64)) where {T<:Real,S,A}
	valuelist = [copy(V)]
	value_iteration_v!(V, θ, mdp, γ, nmax, valuelist)

	π = form_random_policy(mdp)
	make_greedy_policy!(π, mdp, V, γ)
	return (valuelist, π)
end

# ╔═╡ 1ce9df3d-c989-499b-88b8-d59381d37adf
begin_value_iteration_v(mdp::FiniteMDP{T,S,A}, γ::T; Vinit::T = zero(T), kwargs...) where {T<:Real,S,A} = begin_value_iteration_v(mdp, γ, Vinit .* ones(T, size(mdp.ptf, 1)); kwargs...)

# ╔═╡ 39e0e313-79e7-4343-8e02-526f30a66aad
function calculatepolicy(mdp::NamedTuple, γ::Real, V::Dict)
	(p, sa_keys) = mdp
	πraw = Dict(begin
		actions = sa_keys[1][s]
		newdist = Dict(a => 
				sum(p[(s′,r,s,a)] * (r + γ*V[s′]) for (s′,r) in sa_keys[2][(s,a)])
				for a in actions)
		s => newdist
	end
	for s in keys(sa_keys[1]))
	πstar = Dict(s => Dict(argmax(πraw[s]) => 1.0) for s in keys(πraw))
	πstar, πraw
end 

# ╔═╡ f9b3f359-0c24-4a70-9ba8-be185cc01a62
function value_iteration_v(θ::Real, mdp::NamedTuple, γ::Real, V::Dict, delt::Real, nmax::Real, valuelist)
	(p, sa_keys) = mdp
	if nmax <= 0 || delt <= θ
		(πstar, πraw) = calculatepolicy(mdp, γ, V)
		return (valuelist, πstar, πraw)
	else 
		newV = deepcopy(V)
		delt = bellman_optimal_value!(newV, p, sa_keys, γ)
		value_iteration_v(θ, mdp, γ, newV, delt, nmax - 1, vcat(valuelist, newV))	
	end
end

# ╔═╡ a5f1a71f-3c28-49ed-8f29-ff164c4ea02c
function begin_value_iteration_v(mdp::NamedTuple, γ::Real; θ = eps(0.0), nmax=Inf, Vinit = 0.0)
	(p, sa_keys) = mdp
	V = Dict(s => Vinit for s in keys(sa_keys[1]))
	newV = deepcopy(V)
	delt = bellman_optimal_value!(newV, p, sa_keys, γ)
	value_iteration_v(θ, mdp, γ, newV, delt, nmax-1, [V, newV])
end

# ╔═╡ 3d7f4a19-316e-4874-8c28-8e8fe96a9002
function begin_value_iteration_v(mdp::NamedTuple, γ::Real, V; θ = eps(0.0), nmax=Inf)
	(p, sa_keys) = mdp
	newV = deepcopy(V)
	delt = bellman_optimal_value!(newV, p, sa_keys, γ)
	value_iteration_v(θ, mdp, γ, newV, delt, nmax-1, [V, newV])
end

# ╔═╡ 0587517b-4a4c-44f4-81e1-9e630043df9a
md"""
### Value Iteration Results for Gridworld 
"""

# ╔═╡ 5f365f8b-1448-4301-bb4c-2c43ee96b5ab
md"""
#### Value Function Iterations
"""

# ╔═╡ 9f6b1d57-87a0-494a-a844-5d8760513bb6
gridworld_value_iteration = begin_value_iteration_v(create_4x4gridworld_mdp(), 1.0)

# ╔═╡ c693bad0-31a8-42d8-a472-52f2c9825f1b
HTML("""
<div style="display: flex">
$(reduce(linejoin, [show_gridworld_values(values[1:14]; title = "Iteration $(i-1)") for (i, values) in enumerate(gridworld_value_iteration[1])]))
</div>
""")

# ╔═╡ d710264a-c567-415b-9106-a95eace8c622
md"""
#### Optimal Value Function and Policy
|$v_*$|$\pi_*$|
|:---:|:---:|
|$(HTML(show_gridworld_values(gridworld_value_iteration[1][end][1:14])))|$(HTML(show_gridworld_policy(eachcol(gridworld_value_iteration[2]))))|

Converged after $(length(gridworld_value_iteration[1]) - 1) iterations
"""


# ╔═╡ b4607b63-08f4-4a90-99e6-56860c3d9337
md"""
### Value Iteration Results for Jack's Car Rental
"""

# ╔═╡ a671f4ed-d758-4b42-a970-c088b6b59eb2
car_value_iteration = begin_value_iteration_v(car_rental_mdp_v2, 0.9f0; θ = 0.0001f0)

# ╔═╡ e337bb47-8309-4af1-8ed3-2b0de1875e57
begin
	carvalueiterplots = makepolicyvalueplots(car_rental_mdp_v2, car_value_iteration[1][end], car_value_iteration[2], length(car_value_iteration[1]))
	HTML("""
	<div class = carplots>
		$(mapreduce(a -> @htl("""<div>$(relayout(a, width = 350, height = 340))</div>"""), linejoin, carvalueiterplots))
	</div>
	""")
end

# ╔═╡ 315562d0-2bf6-431a-be3f-fb7d2af248b5
md"""
### Example 4.3: Gambler's Problem
"""

# ╔═╡ c456fbdc-0d52-41a9-8e18-dee3c2f4e258
function make_gambler_mdp(p::Real)
	ptf = Dict{Tuple{Int64, Int64, Int64, Int64}, Float64}()
	stermwin = 100
	stermlose = 0
	for s in 1:99
		for a in 0:min(s,100-s)
			swin = s+a
			slose = s-a
			if swin == stermwin
				ptf[(swin, 1, s, a)] = p
			else
				ptf[(swin, 0, s, a)] = p
			end

			ptf[(slose, 0, s, a)] = 1.0-p
		end
	end
	sa_keys = get_sa_keys(ptf)
	V = Dict(s => 0.0 for s in keys(sa_keys[1]))
	V[stermwin] = 0.0
	V[stermlose] = 0.0
	return (p = ptf, sa_keys = sa_keys, Vinit = V)
end		

# ╔═╡ 1016ad1f-36b5-4f4f-86f1-ac8b8c03dbff
function make_gambler_mdp_v2(p::T; winningcapital = 100, losingcapital = 0) where T <: Real
	states = collect(losingcapital+1:winningcapital-1)
	actions = collect(0:ceil(Int64, winningcapital/2)) #note that not all actions are possible in a given state
	rewards = [zero(T), one(T)]

	ptf = zeros(T, length(states)+1, length(rewards), length(actions), length(states))
	for (i_s, s) in enumerate(states)
		for a in 0:min(s, winningcapital-s) #never stake more than is needed to win
			s_win′ = s + a
			(i_s′, i_r) = s_win′ >= winningcapital ? (length(states)+1, 2) : (s_win′ - losingcapital, 1)
			ptf[i_s′, i_r, a+1, i_s] = p

			s_lose′ = s - a
			i_s′ = s_lose′ <= losingcapital ? length(states)+1 : s_lose′ - losingcapital
			ptf[i_s′, 1, a+1, i_s] = 1 - p
		end
	end
	FiniteMDP(states, actions, rewards, ptf)
end		

# ╔═╡ bb87aea9-7d4c-4d2f-b62d-3402fc309d50
function plot_gambler_results(p::T; winningcapital = 100, losingcapital = 0, γ = one(T), kwargs...) where T <: Real
	mdp = make_gambler_mdp_v2(p; winningcapital=winningcapital, losingcapital=losingcapital)
	results = begin_value_iteration_v(mdp, γ; kwargs...)
	numiter = length(first(results))
	valuetraces = [scatter(x = mdp.states, y = first(results)[i], name = "sweep $(i-1)") for i in [2:min(4, numiter); [numiter-1, numiter]]]
	valueplot = plot(valuetraces, Layout(xaxis_title = "Capital", yaxis_title = "Value estimates", width = 600, height = 350, legend_orientation = "h", legend_y = 1.2, margin_t = 10))
	policymap = results[2] |> z -> heatmap(x = mdp.states, y = mdp.actions, z = z, colorscale = "Greys", showscale = false) |> p -> plot(p, Layout(xaxis_title = "Capital", yaxis = attr(title = "Final policy (stake)"), height = 300, width = 600, margin_t = 0))

	str = Markdown.parse(L"""p_h = %$p""")
	
	@htl("""
	<div>$valueplot</div>
	<div>$policymap</div>
	The solution to the gambler's problem for $str  The upper graph shows the value function found by successive sweeps of value iteration.  The lower graph shows the final policy distribution accross actions (stakes) for each state (capital)</div>
	""")
end

# ╔═╡ 5f3dd95b-3563-4a29-ae6d-e772df4f53ad
md"""
### Figure 4.3
Probability of Heads for Gambler's Problem: $(@bind p_h NumberField(0.0:0.1:1.0, default = 0.4))

Winning Capital: $(@bind wc NumberField(1:1000, default = 100))

Discount Rate: $(@bind γ_gambler NumberField(0.01:0.01:1.0, default = 1.0))
"""

# ╔═╡ bb38c5a9-7916-489e-8d94-834f421d57e2
plot_gambler_results(Float32(p_h); winningcapital = wc, γ = Float32(γ_gambler))

# ╔═╡ 04e6f567-31c5-4f05-b5e2-8b46d22dffbc
md"""
> ### *Exercise 4.8* 
> Why does the optimal policy for the gambler's problem have such a curious form?  In particular, for capital of 50 it bets it all on one flip, but for capital of 51 it does not.  Why is this a good policy?

Since $p=0.4$, the expected outcome of a flip is a loss.  Therefore, repeated flips  will lower our chance of winning and discounting is not a factor in this problem so only the final probability for a series of flips is relevant.  That is why at a capital of 50, the strategy wagers all of the available capital resulting in a 40% chance to win and a 60% chance to lose.  Any smaller bet would avoid a loss but neessitate additional flips each of which is expected to lose.  In other words, if we have a chance to win in one flip, it should be taken immediately because otherwise we would be counting on having more winning than losing flips for repeated flips which is not expected.  Following this same logic, for every capital amount greater than 50, the optimal wager is the exact amount of capital needed to win in one flip and this strategy forms the upper policy line shown in the bottom half of figure 4.3.

For any capital amount less than 50, we can only win after repeated flips which is why these capital states have lower values than any state above them, in fact the entire value estimate curve is monotonic for this reason.  One optimal strategy for these capital amounts is to wager all of the available capital as this will advance the capital up as quickly as possible with a win.

There are other optimal policies though, in particular we see that for capital values between 51 and 62, there are two optimal wagers and then from 63 to 74 there are three optimal wagers.  The effect is mirrored at the capital values less than 50 as well.  For this particular value of winning capital the maximum number of optimal wagers is three for a given capital value, but for variations of this problem there could be additional splitting.  At a winning capital of 104, there are 4 optimal wagers starting at a capital of 33.  What is going on here can be understood by looking at the value function.  There are only two possible transitions from a wager so we can write down the Bellman optimality equation as $v_*(s) = \max_a \left [ pv_*(s+a) + (1-p)v_*(s-a) \right ]$.  Given the shape of the value function, there are certain states with non-unique pairs of outcomes that sum to the same optimal value.  For example at a capital state of 51, the wagers of 49 and 1 both lead to the same action value estimate simply because $0.4 + 0.6v_*(2) = 0.4v_*(51) + 0.6v_*(50)$.  A visualization of this equality is shown below.  
"""

# ╔═╡ 76c09949-ca38-4d96-b2aa-f7a1017ff322
md"""
#### Action Value Visualization
Capital State: $(@bind capitaleval NumberField(1:99, default = 50))

Winning Flip Probability: $(@bind pheval NumberField(0.0:0.01:1.0, default = 0.4)) 
"""

# ╔═╡ 9462c98d-1a0e-4c61-b7cf-31fb320ffe68
function evaluate_gambler(p::T; winningcapital = 100, losingcapital = 0, γ = one(T), kwargs...) where T <: Real
	mdp = make_gambler_mdp_v2(p; winningcapital=winningcapital, losingcapital=losingcapital)
	results = begin_value_iteration_v(mdp, γ; kwargs...)
	v = results |> first |> last
	v[1:end-1]
end

# ╔═╡ 4883c217-3f5f-4b30-a92d-615ae0deff7d
function evaluate_wager(p::T, v::Vector{T}, s) where T <: Real
	[begin 
		v1 = s + a > length(v) ? one(T) : v[s+a]
		v2 = s-a < 1 ? zero(T) : v[s-a]
		p*v1 + (1-p)*v2 
	end
	for a in 1:s]
end

# ╔═╡ f9dc45e8-1bfe-4115-a2e2-6dd2cbc427ff
function plot_gambler_Q(s, p::T; winningcapital = 100, losingcapital = 0, γ = one(T), kwargs...) where T <: Real
	v = evaluate_gambler(p; winningcapital = winningcapital, losingcapital = losingcapital, γ = γ, kwargs...)
	push!(v, one(T)) #add the winning value to v at the winning capital amount
	q = evaluate_wager(p, v, s)
	maxq = maximum(q)
	bestwagers = findall(q .≈ maxq)
	vtrace = scatter(x = eachindex(v), y = v, showlegend = false, name = "Value Function")
	vtraces = [scatter(x = [s + bestwager, s - bestwager], y = [v[s + bestwager], (s - bestwager) <= 0 ? zero(T) : v[s - bestwager]], name = "Wager of $bestwager", mode = "markers") for bestwager in bestwagers]
	vtrace3 = scatter(x = [s], y = [v[s]], name = "Capital State = $s")
	vplt = plot([vtrace; vtrace3; vtraces], Layout(xaxis_title = "Capital", yaxis_title = "Value estimates", title = "Optimal Value Function", height = 350))
	formatwagers(x::AbstractVector) = reduce((a, b) -> "$a, $b", x)
	t1 = scatter(x = eachindex(q), y = q, showlegend = false, name = "")
	t2 = scatter(x = bestwagers, y = maxq .* ones(length(bestwagers)), name = "Optimal Wager", mode = "markers")
	qplt = plot([t1, t2], Layout(xaxis_title = "Wager", yaxis_title = "Action Value", title = "Optimal Action Values for Capital $s", height = 350))
	msg = if length(bestwagers) == 1
		md"""Unique best wager of $(first(bestwagers))"""
	else
		md"""The following wagers are equally good: $(formatwagers(bestwagers))"""
	end
		
	md"""
	$qplt
	Gambler action value estimates for winning probability $p and winning capital of $winningcapital.  
	$msg
	$vplt
	"""
end

# ╔═╡ b5d83b23-e5f1-4280-b5d8-2b13191c8ffc
plot_gambler_Q(capitaleval, Float32(pheval))

# ╔═╡ 2f2f6821-8459-4bf9-b0d8-62deffbe5c6b
md"""
> ### *Exercise 4.9 (programming)* 
> Implement value iteration for the gamber's problem and solve it for $p_h=0.25$ and $p_h = 0.55$.  In programming, you may find it convenient to introduce two dummy states corresponding to termination with capital of 0 and 100, giving them values of 0 and 1 respectively.  Show your results graphically as in Figure 4.3  Are you results stable as $\theta \rightarrow 0$?

See code in the section for Example 4.3, below are plots for the desired p values.  In both cases, as the tolerance is made arbitrarily low the value estimates converge to a stable curve.  For $p_h>0.5$ the curves are smoother as the policy and solution are more predictable.
	"""

# ╔═╡ 64144caf-7b21-41b5-a002-6a86e5119f8b
plot_gambler_results(0.25)

# ╔═╡ d79c93ff-7945-435c-8db1-dfdd6518e34e
plot_gambler_results(0.55)

# ╔═╡ 42e4a3d6-26ef-48bb-9164-118186ec118b
md"""
> ### *Exercise 4.10* 
> What is the analog of the value iteration update (4.10) for action values, $q_{k+1}(s,a)$?

Copying equation 4.10 we have

$v_{k+1}(s) = \max_a \sum_{s',r} p(s',r|s,a)[r+\gamma v_k(s')]$

To create the equivalent for action values, we need to use the Bellman Optimality Equation for q rather than v

$q_{k+1}(s,a) = \sum_{s',r}p(s',r|s,a)[r + \gamma \max_{a'} q_k(s',a')]$
"""

# ╔═╡ cf8a9ce5-8204-4628-a21c-df52d986aca0
md"""
# Key Equation Reference

Add summary of Bellman equations and Bellman optimality equations for v and q as well as the mixtures and iterative update equations mentioned in chapters 3 and 4
"""

# ╔═╡ 1c8bf532-326d-4da7-905f-1ba05ea4d748
md"""
## Probability Transition Function
$\begin{flalign}
p(s^\prime, r \vert s, a) &\doteq \Pr \{ S_{t+1} = s^\prime, R_{t+1} = r \mid S_t = s, A_t = a \}
\end{flalign}$
"""

# ╔═╡ 75aeff5b-fb91-4567-84d8-1e617366a6f3
md"""
## Expected Value
$\begin{flalign}
\mathbb{E}[X \vert A] &= \sum_i \Pr \{ X = x_i \vert A \} x_i \\
\mathbb{E}[X + Y] &= \mathbb{E}[X] + \mathbb{E}[Y] \\
\mathbb{E}[cX] &= c \mathbb{E}[X] \: \forall \: \text{constants }c \\
\end{flalign}$
"""

# ╔═╡ 4bbf42a7-3d5e-4e7e-ac12-b135736d19d3
md"""
## Discounted Return
$\begin{flalign}
G_t & \doteq \sum_{k=0}^\infty \gamma^k R_{t+k+1} \text{ or } \sum_{k = t+1} ^ T \gamma^{k-t-1}R_k \tag{3.8/3.11} \\
&= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \\
&= R_{t+1} + \gamma \left [ R_{t+2} + \gamma R_{t+3} + \cdots \right ] \\
&= R_{t+1} + \gamma G_{t+1} \tag{3.9}
\end{flalign}$
"""

# ╔═╡ 14f7ccb6-facc-4d1e-9d15-1eb93253548a
md"""
## Policy Expectations
$\begin{flalign}
\mathbb{E}_\pi [R_{t+1} \vert S_t = s] &\doteq \sum_a \pi(a \vert s) \mathbb{E} [R_{t+1} \vert S_t = s, A_t = a] \\
& = \sum_a \pi(a \vert s) \sum_r r \sum_{s^\prime} \Pr \{S_{t+1} = s^\prime, R_{t+1} = r  \mid S_t = s, A_t = a \} \\
& = \sum_a \pi(a \vert s) \sum_r r \sum_{s^\prime} p(s^\prime, r \vert s, a) \\

\mathbb{E}_\pi [R_{t+2} \vert S_t = s] &\doteq \sum_{s^\prime, r, a} \pi(a \vert s) p(s^\prime, r \vert s, a) \sum_{a^\prime} \pi(a^\prime \vert s^\prime) \sum_{r^\prime} r^\prime \Pr \{ R_{t+2} = r^\prime \vert S_{t+1} = s^\prime, A_{t+1} = a^\prime \} \\
&= \sum_{s^\prime, r, a} \pi(a \vert s) p(s^\prime, r \vert s, a) \mathbb{E}_\pi [R_{t+2} \vert S_{t+1} = s^\prime] \\


\mathbb{E}_\pi [G_{t+1} \vert S_t = s] &= \mathbb{E}_\pi [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \vert S_t = s] \\
&= \sum_a \pi(a \vert s) \sum_r r \sum_{s^\prime} p(s^\prime, r \vert s, a) + \gamma\mathbb{E}_\pi [R_{t+2} + \gamma R_{t+3} + \cdots \vert S_t = s] \\
\end{flalign}$
"""

# ╔═╡ a3269e92-2e16-4173-9bbe-9771dfa291f6
md"""
## Value Functions
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

# ╔═╡ b59756bc-d268-45cf-8fe9-1981d7af1cb6
md"""
## Optimal Value Functions
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

# ╔═╡ 64f22fbb-5c6c-4e8f-bd79-3c9eb19bedca
md"""
# Dependencies and Settings
"""

# ╔═╡ 494c78f2-ae05-411b-a75d-1468be057449
html"""
	<style>
		main {
			margin: 0 auto;
			max-width: min(2000px, 90%);
	    	padding-left: max(10px, 5%);
	    	padding-right: max(10px, 5%);
			font-size: max(10px, min(18px, 2vw));
		}
	</style>
	"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
BenchmarkTools = "~1.3.2"
HypertextLiteral = "~0.9.5"
Latexify = "~0.16.2"
PlutoPlotly = "~0.4.5"
PlutoUI = "~0.7.58"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.3"
manifest_format = "2.0"
project_hash = "0785e3c85560030238c49c8380ed313ca3820d7b"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0f748c81756f2e5e6854298f11ad8b2dfae6911a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BaseDirs]]
git-tree-sha1 = "3e93fcd95fe8db4704e98dbda14453a0bfc6f6c3"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.2.3"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

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

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

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

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Format]]
git-tree-sha1 = "f3cf88025f6d03c194d73f5d13fee9004a108329"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.6"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cad560042a7cc108f5a4c24ea1431a9221f22c1b"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.2"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

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
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "BaseDirs", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "Reexport", "TOML"]
git-tree-sha1 = "fbf637823ec24c5669b1a66f3771c2306f60857c"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.4.5"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "71a22244e352aa8c5f0f2adde4150f62368a3f2e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.58"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

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

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

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
# ╟─4017d910-3635-4ffa-ac1b-919a7bff1e6e
# ╟─fb497aa8-9d34-48e6-ae85-72df30c1adf3
# ╟─55276004-877e-47c0-b5b5-49dbe29aa6f7
# ╟─772d17b0-6fbc-4309-b55b-d17f9b4d3ddf
# ╟─4665aa5c-87d1-4359-8cfd-7502d8c5d2e2
# ╟─86cd5d5f-f79f-4dc6-9d88-ae4753190de9
# ╠═7575b85a-c988-47e6-bdcc-fde4d92708a5
# ╠═5b65e817-5504-413f-9c1a-17880d238d80
# ╠═59b91c65-3f8a-4015-bb08-d7455623101c
# ╠═88f8e335-968e-4e2f-8d3a-395667ad2ed3
# ╠═5b912508-aa15-470e-be4e-430e88d8a68d
# ╠═d6b58b68-504d-463f-91e8-ee85d0f90000
# ╠═9253064c-7dfe-445f-b377-fc1acbb6886e
# ╠═d5eda4ce-1c22-4d3d-b1b0-9c8d0357f6cf
# ╠═e6cdb2be-697d-4191-bd5a-9c129b32246d
# ╟─92e50901-7d10-470b-a985-be45adcad817
# ╠═4dd32517-4c12-4635-8135-3019828af2b5
# ╠═df9593ca-6d27-45de-9d46-79bddc7a3862
# ╠═c7bdf32a-2f89-4bf8-916b-7558ceedb628
# ╠═6d04573a-c2a9-4cfa-88f1-2b2723a95aac
# ╠═7563d584-abfc-4d9f-9759-163ac55922a9
# ╠═6048b106-458e-4e3b-bba9-5f3578458c7c
# ╠═4c46ea30-eeb2-4c25-9e6c-3bcddaf48de7
# ╠═0d0e82e4-b3a4-4528-9288-285fdc5aa8af
# ╠═00593e99-7fa9-49bb-8e60-5a0f083304ab
# ╠═cda46125-2cfa-4b57-adf8-164eed8f5af6
# ╠═0a17c3d0-7b46-45b2-8d8e-690fd498565c
# ╠═17ff2d94-aea9-4a15-98b3-97437e2b70ab
# ╠═4c0872f8-0fd5-4a44-9587-28cb04697d25
# ╟─0297a007-5898-4936-ae2f-386a725700e4
# ╠═e4370697-e6a7-40f0-974a-ed219102c13f
# ╟─6844dff1-bc0b-47c5-8496-efe46dafbb5b
# ╟─34f0f670-483f-4add-bf25-34993d646e5e
# ╟─1d098de3-592e-401b-a493-2728e8a6ffe9
# ╟─d0b4a71b-b574-4d62-be0b-14e03595a15c
# ╟─2c23c4ec-f332-4e05-a730-06fa20a0227a
# ╠═aa19ffd4-69a0-44a9-8109-d6be003ae7b1
# ╟─39f4c75c-43f7-476f-bbc9-c704e5dee300
# ╟─e76bd134-f4ac-4382-b56a-fca8f3ca27cd
# ╠═f4fce267-78a2-4fd3-aad5-a8298783c015
# ╠═7539da6f-1fb7-4a63-98ba-52b81bb27eca
# ╠═c078f6c3-7576-4933-bc95-d33e8193ee93
# ╠═1b8bfddb-97c3-4756-ab8e-123d38afda64
# ╟─f68b0587-1203-4985-9077-ded678ba4b8f
# ╟─7e9fe05a-c447-4a41-8c61-67c9a899411c
# ╟─20c5e03d-a1f4-4b2e-9893-efb9b03f00e8
# ╟─7dcb5621-17ce-4794-b70e-e639e5068a18
# ╟─38875afb-f8a3-4b8f-be7f-a34cc19efa7d
# ╟─b3ed0348-3d74-4726-878f-5eefcb1d72d0
# ╟─5fae76af-ac80-49c5-b553-73d09a6e9098
# ╟─dae71267-9945-41d2-bec4-546c8c883ae0
# ╟─f80580b3-f370-4a02-a9e2-ed791f380521
# ╟─71abc452-cefc-47f8-8f9c-6fd3565f3ec6
# ╟─54c73389-bb7a-48f1-b5d5-9d4972b1857a
# ╟─9e594059-39cd-4fc7-91ae-b8e9156db6df
# ╠═10c9b166-3a88-460e-82e8-a16c020c1378
# ╠═1618ec46-6e13-42bf-a7f2-68f8dbe3714c
# ╠═11365717-6dee-461c-8ecf-485144b53a93
# ╟─ab98efa4-c793-40eb-8c1c-70a0bb929ab3
# ╟─68a01f8e-769b-4362-b12a-48733e8b8dba
# ╟─e4bfdaca-3f3d-43bb-b8aa-7536adbff662
# ╟─35e1ffe5-d36a-449d-aa73-c618e2855042
# ╟─aa2e7334-af07-4152-8f21-e80bdcdd979b
# ╟─67b06f3b-13df-4b27-ad80-d112432e8f42
# ╟─87718a9d-5624-4f18-9dbc-34458dd917fd
# ╠═160c59b0-a5ea-4046-b79f-7a6a6fc8db7e
# ╠═56184148-aaad-470c-b79c-30b952e1142d
# ╠═8b4fb649-8aaa-4e17-8204-540caf8da343
# ╠═f0e5d2e6-3d00-4ffc-962e-e98d4bb28e4e
# ╠═31a9db21-c7d2-4053-8c18-2023f4720196
# ╠═4d15118f-f1ab-4115-bcc9-7f98246eca1c
# ╠═77250f6b-60d1-426f-85b2-497186b86c50
# ╠═7021bc9e-716a-43ff-b2cb-40b499ae2706
# ╟─bb5fb8e6-0163-4fde-8238-63454f1c5128
# ╠═c47882c7-ded1-440c-a9a3-0b89a0e7a011
# ╠═ec5afe79-1b1b-4ccd-b672-450797b4e73e
# ╠═0079b02d-8895-4dd4-9557-5f08ac341404
# ╠═77d251d3-903b-4e96-9261-77a429a3eda7
# ╟─a5174afc-04f2-4fc9-9b10-7aa7f249332a
# ╟─c4a14f1c-ce17-40eb-b9ae-00b651d40714
# ╟─c3f4764f-b2e3-4004-b5ed-2f1cccd2cdde
# ╠═e722b7e0-63a3-4195-b13e-0449abb3cc39
# ╠═bd41fffb-5d8c-4165-9f44-690f81c70113
# ╠═871d78a4-cc74-4866-896b-027a6c626676
# ╠═eb8fa5e6-c18d-41ba-a78e-cceccc90d9b6
# ╠═05512261-d0c7-4602-8280-cd1d4d45e875
# ╠═c4835f94-1ebc-43bf-b54a-5252e4280635
# ╠═8b0e875b-2644-4f52-83a0-a736e9330e78
# ╠═e7f3cc36-56a7-4592-b3f6-05001675cc14
# ╠═e94c14ac-9583-4b31-a392-996dcb6d79b7
# ╠═26566af8-152f-4553-b26b-303dff3d2f24
# ╟─3e1a1559-6dc0-46a1-a639-d8655b72e740
# ╟─0d6936d1-38af-45f1-b496-da49b60f11f8
# ╟─ad1bf1d2-211d-44ca-a258-fc6e112785da
# ╟─e316f59a-8070-4510-96f3-15498897347c
# ╟─5dbfb100-49a8-4f9d-a752-bda4da54699e
# ╠═2360169d-291b-4976-815a-8092c89260a8
# ╠═45fe9d02-5b77-4dc4-868d-1c598121cd90
# ╠═13ca35da-7f61-4a89-98ba-6418d754707f
# ╟─19e7b5c2-a32d-47e7-a436-f95fd3397ac5
# ╟─2bedc22e-9615-4fb4-94bf-6a0e7114c417
# ╟─dde0354c-81cb-4898-85ac-723ec7346116
# ╠═7140970d-d4a7-45bc-9626-26cf1a2f945b
# ╠═09a98ebf-4685-404b-b492-116c184d4a52
# ╠═64dc17e3-a83f-4414-a74a-c6c8e3e498fa
# ╠═16adacf1-2757-4a45-a2e5-8ec4bf04584e
# ╠═057beec8-0ad7-4bf2-ac45-6753614c0b4d
# ╠═f9b3f359-0c24-4a70-9ba8-be185cc01a62
# ╠═fa1849cb-663f-49c0-acc1-f7fdbcbb4189
# ╠═9c5144fa-9138-4d20-b1da-b69c6f7cae4c
# ╠═1ce9df3d-c989-499b-88b8-d59381d37adf
# ╠═39e0e313-79e7-4343-8e02-526f30a66aad
# ╠═a5f1a71f-3c28-49ed-8f29-ff164c4ea02c
# ╠═3d7f4a19-316e-4874-8c28-8e8fe96a9002
# ╟─0587517b-4a4c-44f4-81e1-9e630043df9a
# ╟─5f365f8b-1448-4301-bb4c-2c43ee96b5ab
# ╟─c693bad0-31a8-42d8-a472-52f2c9825f1b
# ╟─d710264a-c567-415b-9106-a95eace8c622
# ╟─9f6b1d57-87a0-494a-a844-5d8760513bb6
# ╟─b4607b63-08f4-4a90-99e6-56860c3d9337
# ╠═a671f4ed-d758-4b42-a970-c088b6b59eb2
# ╟─e337bb47-8309-4af1-8ed3-2b0de1875e57
# ╟─315562d0-2bf6-431a-be3f-fb7d2af248b5
# ╠═c456fbdc-0d52-41a9-8e18-dee3c2f4e258
# ╠═1016ad1f-36b5-4f4f-86f1-ac8b8c03dbff
# ╟─bb87aea9-7d4c-4d2f-b62d-3402fc309d50
# ╟─5f3dd95b-3563-4a29-ae6d-e772df4f53ad
# ╟─bb38c5a9-7916-489e-8d94-834f421d57e2
# ╟─04e6f567-31c5-4f05-b5e2-8b46d22dffbc
# ╟─76c09949-ca38-4d96-b2aa-f7a1017ff322
# ╟─b5d83b23-e5f1-4280-b5d8-2b13191c8ffc
# ╟─9462c98d-1a0e-4c61-b7cf-31fb320ffe68
# ╟─4883c217-3f5f-4b30-a92d-615ae0deff7d
# ╟─f9dc45e8-1bfe-4115-a2e2-6dd2cbc427ff
# ╟─2f2f6821-8459-4bf9-b0d8-62deffbe5c6b
# ╟─64144caf-7b21-41b5-a002-6a86e5119f8b
# ╟─d79c93ff-7945-435c-8db1-dfdd6518e34e
# ╟─42e4a3d6-26ef-48bb-9164-118186ec118b
# ╟─cf8a9ce5-8204-4628-a21c-df52d986aca0
# ╠═1c8bf532-326d-4da7-905f-1ba05ea4d748
# ╟─75aeff5b-fb91-4567-84d8-1e617366a6f3
# ╠═4bbf42a7-3d5e-4e7e-ac12-b135736d19d3
# ╟─14f7ccb6-facc-4d1e-9d15-1eb93253548a
# ╠═a3269e92-2e16-4173-9bbe-9771dfa291f6
# ╟─b59756bc-d268-45cf-8fe9-1981d7af1cb6
# ╟─64f22fbb-5c6c-4e8f-bd79-3c9eb19bedca
# ╠═f5809bd3-64d4-47ee-9e41-e491f8c09719
# ╠═494c78f2-ae05-411b-a75d-1468be057449
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
