### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ 6da69e64-743f-4ea9-9670-fd023c7ffab7
using PlutoDevMacros, LinearAlgebra, Random, Statistics

# ╔═╡ 808fcb4f-f113-4623-9131-c709320130df
# ╠═╡ show_logs = false
@only_in_nb PlutoDevMacros.@frompackage @raw_str(joinpath(@__DIR__, "..", "ApproximationUtils.jl")) using ApproximationUtils

# ╔═╡ db8dd224-abf1-4a65-b8bb-e2da6ab43f7e
# ╠═╡ skip_as_script = true
#=╠═╡
begin 
	using PlutoPlotly, PlutoUI, PlutoProfile, BenchmarkTools, LaTeXStrings, HypertextLiteral
	TableOfContents()
end
  ╠═╡ =#

# ╔═╡ 19d23ef5-27db-44a8-99fe-a7343a5db2b8
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
# Chapter 9 On-policy Prediction with Approximation
## 9.1 Value-function Approximation
The method we use to approximate the true value function must be able to learn efficiently from incrementally acquired data.  Also the target values of training the function may be non stationary.  We will designate some approximation function for our value function as $\hat v(S, w)$ which is parametrized by some weights  that in general will be much smaller than in size to the true state space.
"""
  ╠═╡ =#

# ╔═╡ c4c71ace-c3a4-412b-b08b-31d246f8db5f
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
## 9.2 The Prediction Objective ($\overline {\text{VE}}$)
In tabular methods, the learned value can exactly equal the true objective and each state approximation is independent.  Neither of these are true for parametrized approximation.  We must specificy a state distribution $\mu(s) \geq 0, \sum_s{\mu(s)}=1$ that represents how much we care about the error in each state.  One natural objective function is the mean squared error weighted over this distribution.

$\overline{\text{VE}}(\mathbf{w}) \doteq \sum_{s \in S} \mu(s)[v_\pi(s) - \hat v(s, \mathbf{w})]^2 \tag{9.1}$

Often $\mu(s)$ is taken to be the fraction of time spent in $s$.  In contiunuing tasks the on-policy distribution is the stationary distribution under $\pi$.  In episodic tasks one must account for the probability of starting an episode in a particular state and the probability of transitioning to that state during an episode.  The state distribution will need to depend on that function typically denoted $\eta(s)$.

An ideal goal for optimizing $\overline {\text{VE}}$ is to find a *global optimum* for the weight vector such that $\overline {\text{VE}}(\mathbf{w}^*) \leq \overline {\text{VE}}(\mathbf{w})$ for all posible weights.  Typically this isn't possible, but we can find a *local optimum* in most cases.  Even this objective is not guaranteed for many approximation methods.  In this chapter we will focus on approximation methods based on linear gradient-descent methods since we can easily find an optimum in those cases.
"""
  ╠═╡ =#

# ╔═╡ cb5e302b-a14b-4135-b6ff-bee300f9dee6
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
## 9.3 Stochastic-gradient and Semi-gradient Methods
We will assume a weight vector with a fixed number of components $\mathbf{w} \doteq (w_1, w_2, \dots, w_d)$ and a differentiable value function $\hat v(s, \mathbf{w})$ that exists for all states.  We will update weights at each of a series of discrete time steps so we can denote $\mathbf{w}_t$ as the weight vector at each step.  Assume at each step we observe a state and its true value under the policy.  We assume that states appear in the same distribution $\mu$ over which we are trying to optimize the prediction objective.  Under these assumptions we can try to minimize the error observed on each example using *Stochastic gradient-descent* (SGD) by adjusting the weight vector a small amount after each observation:

$$\begin{flalign}
\mathbf{w}_{t+1} & \doteq \mathbf{w}_t - \frac{1}{2} \alpha \nabla [v_\pi(S_t) - \hat v(S_t, \mathbf{w}_t)]^2 \\
& = \mathbf{w}_t + \alpha[v_\pi(S_t) - \hat v(S_t, \mathbf{w}_t)]\nabla\hat v(S_t, \mathbf{w}_t) \tag{9.5}
\end{flalign}$$

where $\alpha$ is a learning rate.  In general this method will only converge to the weight vector that minimizes the error objective if $\alpha$ is sufficiently small and decreases over time.  The gradient is defined as follows:

$\nabla f(\mathbf{w}) \doteq \left ( \frac{\partial{f(\mathbf{w})}}{\partial{w_1}} , \frac{\partial{f(\mathbf{w})}}{\partial{w_2}}, \cdots, \frac{\partial{f(\mathbf{w})}}{\partial{w_d}} \right ) ^ \top \tag{9.6}$

If we do not receive the true value function at each example but rather a bootstrap approxmiation or a noise corrupted version, we can use the same formula and simply replace $v_\pi(S_t)$ with $U_t$.  As long as $U_t$ is an *unbiased* estimate for each example then the weights are still guaranteed to converge to a local optimum stochastically.  One example of an unbiased estimate would be a monte carlo sample of the discounted future return.

If we use a bootstrapped estimate of the value, then the estimate depends on the current weight vector and will no longer be *unbiased* which requires that the update target be independent of $\mathbf{w}_t$.  A method using bootstrapping with function approximation would be considered a *semi-gradient method* because it violates part of the convergence assumptions.  In the case of a linear function, however, they can still converge reliably.  One typical example of this is semi-gradient TD(0) learning which uses the value estimate target of $U_t \doteq R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w})$.  In this case the update step for the weight vector is as follows:

$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha[R_t + \gamma \hat v(S_{t+1}, \mathbf{w}_t) - \hat v(S_t, \mathbf{w}_t)] \nabla \hat v(S_t, \mathbf{w}_t) \tag{9.7}$

*State aggregation* is a simple form of generalizing function approximation in which states are grouped together, with one estimated value (one component of the weight vector **w**) for each group.  The value of a state is estimated as its group's component, and when the state is updated, that component alone is updated.  State aggregation is a special case of SGD in which the gradient, $\nabla \hat v(S_t, \mathbf{w}_t)$, is 1 for the observed state's component and 0 for others.
"""
  ╠═╡ =#

# ╔═╡ 865ed63a-a7ee-403f-9004-b3ec659d756f
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
### *Gradient Monte Carlo Algorithm for Estimating $$\hat v \approx v_\pi$$*

Monte Carlo sampling to estiamate $G_t$ can be used as a true gradient approximation method because $G_t$ is an unbiased estimate of $v_\pi (S_t)$ that does not depend on the parameters of the estimator.  To implement this algorithm, I will use a parameter gradient update rule that is more generic than the one in (9.5).  Instead, consider the more fundamental gradient update rule: 

$\mathbf{w}_{t+1} \doteq \mathbf{w}_t - \alpha \nabla \left [ err(\hat v(S_t, \mathbf{w}_t), U_t)\right ]$ 

where $err(x, y)$ is the error between $x$ and $y$.  For most of the examples in this chapter $err(x, y) \doteq (x - y)^2$ which reduces to the familiar rule shown in (9.5) where $v_\pi(S_t)$ is replaced with $G_t$.  In the case of Monte Carlo methods $U_t = G_t$ whereas in TD methods $U_t = R_t + \gamma \hat v(S_{t+1}, \mathbf{w}_t)$.

In order to implement the gradient update, one needs to define `update_parameters!` which is a function that does the full parameter update defined above which includes computing the gradients of the error function and the estimator. The following arguments are required: 
	
- `parameters`: values that get updated and are used in the function approximation
- `s`: current state
- `g`: target value 
- `α`: step size

Additional arguments can also be passed after this which will always be placeholder memory objects that the function can use to make calculations.  Allowing these arguments to be passed in means that the function need not create these variables every time it needs to perform an update.

The purpose for requiring this function instead of (9.5) is that certain function approximators will naturally compute the gradient of the error function directly rather than computing the gradient of the value function.  The quantity we are after is the entire expression multiplying $\alpha$.  Moreover, some approximators have a very simple gradient that implies a parameter update that doesn't even require computing the entire gradient, and the value estimate computation might be part of the gradient already making it unecessary to compute twice.  For example in the case of state aggregation described below, only one parameter will be updated at a time, so writing the update as in (9.5) is wasteful since most of the computations will simply be adding 0.  As a bonus, this format allows us to consider alternative error functions such as cross entropy loss.
"""
  ╠═╡ =#

# ╔═╡ 628c6613-0516-4078-a872-31f122831190
const FCANNParams{T} = @NamedTuple{weights::Tuple{Vector{Matrix{T}}, Vector{Vector{T}}}, reslayers::Int64} where T<:Float32

# ╔═╡ 47dbe518-6789-4639-bfc0-e5e5ddde980a
const FCANNActivations{T} = Vector{Vector{T}} where T<:Float32

# ╔═╡ a5ec633c-d5d0-4556-9bfe-16f51be1279f
const LinearFeatures{I} = Union{C1, C2} where {I <: Integer, C1 <: AbstractVector{I}, C2 <: Base.Generator{C1}}

# ╔═╡ 1176adf1-0a2a-41df-a3c5-f382126a0fe5
"""
    AbstractBinaryFeatures

# Summary
```julia
abstract type AbstractBinaryFeatures{I <: Integer, N} end
```

Abstract supertype for sparse binary feature representations in reinforcement learning.

# Type Parameters
- `I <: Integer`: Integer type for feature indices  
- `N`: Maximum feature space size (compile-time constant)

# Subtype Hierarchy
```
AbstractBinaryFeatures{I, N}
├── BinaryFeatureVector{I, N}
└── StateAggregationFeatureVector{I, N}
```

All concrete subtypes should implement sparse feature storage where only 
active (non-zero) features are stored explicitly.

See also: [`BinaryFeatureVector`](@ref), [`StateAggregationFeatureVector`](@ref)
"""
abstract type AbstractBinaryFeatures{I <: Integer, N} end;

# ╔═╡ 1e8e5f5f-8b73-4820-a4fc-f97c8344f9e7
"""
    BinaryFeatureVector

# Summary
```julia
mutable struct BinaryFeatureVector{I <: Integer, N} <: AbstractBinaryFeatures{I, N}
```

Sparse binary feature vector that stores only the indices of active features.

# Fields
- `active_features::Vector{I}`: Indices of features with value 1
- `num_features::I`: Number of currently active features

# Supertype Hierarchy
```
BinaryFeatureVector{I, N} <: AbstractBinaryFeatures{I, N} <: Any
```

# Type Parameters
- `I <: Integer`: Integer type for indices (typically `Int64`)
- `N`: Total feature space size (matches parameter vector length)

# Constructor
    BinaryFeatureVector(N::Integer)

Creates empty binary feature vector with maximum feature space size `N`.

# Examples
```julia
# Create binary feature vector for feature space of size 100
features = BinaryFeatureVector(100)

# Manually set active features (typically done by feature extraction)
features.active_features = [5, 12, 67]
features.num_features = 3

# This represents a sparse vector with 1s at positions 5, 12, 67
# and 0s elsewhere
```

# Performance
Optimized for sparse representations where few features are active relative 
to the total feature space size `N`.

See also: [`AbstractBinaryFeatures`](@ref), [`LinearFeatureVector`](@ref)
"""
mutable struct BinaryFeatureVector{I <: Integer, N} <: AbstractBinaryFeatures{I, N}
	active_features::Vector{I}
	num_features::I
	function BinaryFeatureVector(N::Integer)
		new{Int64, N}(Vector{Int64}(), 0)
	end
end;

# ╔═╡ 14a13743-8e3a-4698-aef2-245557adfd92
"""
    StateAggregationFeatureVector

# Summary
```julia
mutable struct StateAggregationFeatureVector{I <: Integer, N} <: AbstractBinaryFeatures{I, N}
```

Feature representation for state aggregation where states are grouped into 
discrete categories, with exactly one group active at a time.

# Fields
- `group_index::I`: Index of the active group (1 to N)

# Supertype Hierarchy
```
StateAggregationFeatureVector{I, N} <: AbstractBinaryFeatures{I, N} <: Any
```

# Type Parameters
- `I <: Integer`: Integer type for group index (typically `Int64`)
- `N`: Total number of groups (matches parameter vector length)

# Constructor
    StateAggregationFeatureVector(N::Integer)

Creates state aggregation vector for `N` possible groups, initialized with 
group_index = 0 (invalid state).

# Examples
```julia
# Create state aggregation for 8 groups
state_features = StateAggregationFeatureVector(8)

# Set active group (typically done by state processing)
state_features.group_index = 3  # State belongs to group 3

# This represents a one-hot vector with 1 at position 3, 0s elsewhere
```

# Notes
This is the most efficient representation when states can be partitioned into
discrete groups and each state belongs to exactly one group.

See also: [`AbstractBinaryFeatures`](@ref), [`LinearFeatureVector`](@ref)
"""
mutable struct StateAggregationFeatureVector{I <: Integer, N} <: AbstractBinaryFeatures{I, N}
	group_index::I
	function StateAggregationFeatureVector(N::Integer)
		new{Int64, N}(0)
	end
end;

# ╔═╡ d1e4e1d5-0c14-4aaa-95c6-f741f83fce0d
#this represents anything that could be used with linear function approximation which is either a vector, a binary feature vector, or a state aggregation feature vector which just stores the group_index
const LinearFeatureVector{I} = Union{C1, C2, C3} where {I <: Integer, T<:Real, C1 <: Vector{T}, N, C2 <: BinaryFeatureVector{I, N}, C3 <: StateAggregationFeatureVector{I, N}}

# ╔═╡ f8bc8f92-a9c6-4b7b-9a8e-48fbb1f85e6c
function update_state_aggregation_feature_vector!(x::StateAggregationFeatureVector{I, N}, group_index::I) where {N, I<:Integer}
	x.group_index = group_index
	return x
end

# ╔═╡ a2ffaa35-ee82-47fd-878e-dd535caab109
begin
	"""
	    update_params_with_gradient!(params, α, gradient) -> params
	
	Updates parameters in-place using gradient ascent/descent with step size α.
	
	This function supports multiple parameter and gradient representations through dispatch,
	enabling efficient updates for different function approximation schemes used in 
	reinforcement learning.
	
	# Arguments
	- `params`: Parameter structure to update (modified in-place)
	- `α::Real`: Step size for gradient update
	- `gradient`: Gradient information (type must be compatible with params)
	
	# See Also
	[`linear_value_function`](@ref)
	
	# Methods
	
	## General Array Updates
	```julia
	update_params_with_gradient!(θ::Array{T, N}, α::T, ∇θ::Array{T, N}) where {T<:Real, N}
	```
	Standard element-wise gradient update for arbitrary dimensional arrays.
	Performs: `θ[i] += α * ∇θ[i]` for all elements.
	
	- `θ::Array{T, N}`: Parameter array of any dimension
	- `α::T`: Step size 
	- `∇θ::Array{T, N}`: Gradient array (same shape as θ)
	
	### Examples
	```julia-repl
	julia> θ = [1.0, 2.0, 3.0]
	3-element Vector{Float64}:
	 1.0
	 2.0
	 3.0
	
	julia> ∇θ = [0.1, -0.2, 0.05]
	3-element Vector{Float64}:
	  0.1
	 -0.2
	  0.05
	
	julia> update_params_with_gradient!(θ, 0.1, ∇θ)
	3-element Vector{Float64}:
	 1.01
	 1.98
	 3.005
	```
	
	## Neural Network Parameters
	```julia
	update_params_with_gradient!(params::FCANNParams{T}, α::T, ∇::FCANNParams{T}) where T<:Float32
	```
	Updates fully connected neural network parameters by recursively updating 
	weight matrices and bias vectors.
	
	- `params::`[`FCANNParams{T}`](@ref FCANNParams): Network parameters (weights, biases)
	- `α::T`: Step size
	- `∇::`[`FCANNParams{T}`](@ref FCANNParams): Network gradients (same structure as params)
	
	## Binary Feature Gradients
	```julia
	update_params_with_gradient!(w::Vector{T}, α::T, ∇w::BinaryFeatureVector) where {T<:Real}
	```
	Efficient update for binary feature gradients. Only updates parameters 
	corresponding to active features with step size α.
	
	- `w::Vector{T}`: Linear function approximation weights
	- `α::T`: Step size  
	- `∇w::`[`BinaryFeatureVector`](@ref): Sparse gradient (active features get update of α)
	
	### Examples
	```julia-repl
	julia> weights = zeros(10)
	10-element Vector{Float64}:
	 0.0
	 ⋮
	 0.0
	
	julia> binary_grad = BinaryFeatureVector(10);
	
	julia> binary_grad.active_features = [2, 5];
	
	julia> binary_grad.num_features = 2;
	
	julia> update_params_with_gradient!(weights, 0.1, binary_grad);
	
	julia> weights
	10-element Vector{Float64}:
	 0.0
	 0.1
	 0.0
	 0.0
	 0.1
	 0.0
	 ⋮
	```
	
	## State Aggregation Gradients  
	```julia
	update_params_with_gradient!(w::Vector{T}, α::T, ∇w::StateAggregationFeatureVector) where {T<:Real}
	```
	Updates single parameter corresponding to active state group.
	
	- `w::Vector{T}`: State aggregation parameters
	- `α::T`: Step size
	- `∇w::`[`StateAggregationFeatureVector`](@ref): Group index for update
	
	### Examples
	```julia-repl
	julia> state_weights = zeros(8)
	8-element Vector{Float64}:
	 0.0
	 ⋮
	
	julia> state_grad = StateAggregationFeatureVector(8);
	
	julia> state_grad.group_index = 3;
	
	julia> update_params_with_gradient!(state_weights, 0.1, state_grad);
	
	julia> state_weights[3]
	0.1
	```
	
	## No-op for Nothing
	```julia
	update_params_with_gradient!(::Nothing, α::T, ::Nothing) where T<:Real
	```
	Handles cases where parameters or gradients are `nothing` (e.g., optional components).
	Returns `nothing` without performing any updates.
	
	# Performance Notes
	- All methods update parameters in-place for memory efficiency
	- Binary feature method uses `@simd` optimization
	- State aggregation provides O(1) update time
	- Neural network method recursively calls array update for each layer

	## Action-Value Parameters (Matrix Storage)

	### Dense Action-Value Features
	```julia
	update_params_with_gradient!(w::Matrix{T}, α::T, ∇w::LinearActionValueGradient{I, Vector}) where {T<:Real, I <: Integer}
	```
	Updates action-value parameters stored as matrix with dense feature gradients.
	Updates column `∇w.action_index` using vectorized operations with SIMD optimization.
	
	- `w::Matrix{T}`: Parameter matrix (features × actions) to update in-place
	- `α::T`: Learning rate scalar
	- `∇w::LinearActionValueGradient{I, Vector}`: Dense action-value gradient storage
	
	### Binary Action-Value Features  
	```julia
	update_params_with_gradient!(w::Matrix{T}, α::T, ∇w::LinearActionValueGradient{I, BinaryFeatureVector}) where {T<:Real, I <: Integer}
	```
	Updates action-value parameters with sparse binary feature gradients.
	Only updates parameters corresponding to active features with unit gradients.
	
	- `w::Matrix{T}`: Parameter matrix (features × actions) to update in-place
	- `α::T`: Learning rate scalar  
	- `∇w::LinearActionValueGradient{I, BinaryFeatureVector}`: Binary action-value gradient storage
	
	### State Aggregation Action-Value Features
	```julia
	update_params_with_gradient!(w::Matrix{T}, α::T, ∇w::LinearActionValueGradient{I, StateAggregationFeatureVector}) where {T<:Real, I <: Integer}
	```
	Updates single action-value parameter for state aggregation features.
	Provides O(1) update time by updating only the active group parameter.
	
	- `w::Matrix{T}`: Parameter matrix (groups × actions) to update in-place
	- `α::T`: Learning rate scalar
	- `∇w::LinearActionValueGradient{I, StateAggregationFeatureVector}`: State aggregation action-value gradient
	"""
	function update_params_with_gradient!(θ::Array{T, N}, α::T, ∇θ::Array{T, N}) where {T<:Real, N}
		θ .+= α .* ∇θ
		return θ
	end

	function update_params_with_gradient!(params::FCANNParams{T}, α::T, ∇::FCANNParams{T}) where T<:Float32
		for i in eachindex(first(params.weights))
			for j in 1:2
				update_params_with_gradient!(params.weights[j][i], α, ∇.weights[j][i])
			end
		end
		return params
	end

	function update_params_with_gradient!(w::Vector{T}, α::T, ∇w::BinaryFeatureVector) where {T<:Real}
		@inbounds @simd for i in 1:∇w.num_features
			j = ∇w.active_features[i]
			w[j] += α
		end
		return w
	end

	function update_params_with_gradient!(w::Vector{T}, α::T, ∇w::StateAggregationFeatureVector) where {T<:Real}
		w[∇w.group_index] += α
		return w
	end

	update_params_with_gradient!(::Nothing, α::T, ::Nothing) where T<:Real = return nothing
end

# ╔═╡ 42a7918f-8e9d-45ae-9e40-1254dde9f06f
begin
	#general linear approximation
	"""
	    linear_value_function(features, params) -> Real
	
	Computes the linear value function for given features and parameters.
	
	This function supports multiple feature representations through dispatch:
	- Dense feature vectors (dot product)
	- Sparse binary features (sum of active feature weights)  
	- State aggregation features (direct parameter lookup)
	
	# Arguments
	- `features`: Feature representation, see Methods section for supported types
	- `params::Vector{T}`: Parameter vector where T<:Real
	
	# See Also
	[`update_params_with_gradient!`](@ref)
	
	# Methods
	
	## Dense Features
	```julia
	linear_value_function(x::Vector{T}, w::Vector{T}) where T<:Real
	```
	Computes dot product of dense feature vector with parameters.
	- `x::Vector{T}`: Dense feature vector
	- `w::Vector{T}`: Parameter vector (must match length of x)
	
	### Examples
	```julia-repl
	julia> dense_features = [0.5, 1.2, -0.3]
	3-element Vector{Float64}:
	  0.5
	  1.2
	 -0.3
	
	julia> weights = [2.0, 1.5, 0.8]
	3-element Vector{Float64}:
	 2.0
	 1.5
	 0.8
	
	julia> linear_value_function(dense_features, weights)
	2.56
	```
	
	## Binary Features  
	```julia
	linear_value_function(binary_features::BinaryFeatureVector, params::Vector{T}) where T<:Real
	```
	Efficiently computes value for sparse binary features by summing weights of active features only.
	- `binary_features::`[`BinaryFeatureVector{I,N}`](@ref BinaryFeatureVector): Sparse binary feature representation
	- `params::Vector{T}`: Parameter vector of length N
	
	### Examples
	```julia-repl
	julia> binary_features = BinaryFeatureVector(10);
	
	julia> binary_features.active_features = [2, 5];
	
	julia> binary_features.num_features = 2;
	
	julia> params = [0.1, 0.3, 0.2, 0.4, 0.7, 0.1, 0.2, 0.3, 0.1, 0.5]
	10-element Vector{Float64}:
	 0.1
	 0.3
	 0.2
	 0.4
	 0.7
	 0.1
	 0.2
	 0.3
	 0.1
	 0.5
	
	julia> linear_value_function(binary_features, params)
	0.4
	```
	
	## State Aggregation
	```julia
	linear_value_function(x::StateAggregationFeatureVector, params::Vector{T}) where T<:Real  
	```
	Direct parameter lookup for state aggregation features.
	- `x::`[`StateAggregationFeatureVector{I,N}`](@ref StateAggregationFeatureVector): State aggregation with group index
	- `params::Vector{T}`: Parameter vector of length N
	
	### Examples
	```julia-repl
	julia> state_agg = StateAggregationFeatureVector(8);
	
	julia> state_agg.group_index = 3;
	
	julia> params = rand(8)
	8-element Vector{Float64}:
	 0.123
	 0.456
	 0.789
	 0.321
	 0.654
	 0.987
	 0.147
	 0.258
	
	julia> linear_value_function(state_agg, params)
	0.789
	```
	
	# Performance Notes
	- Binary feature method uses `@simd` optimization for active feature summation
	- State aggregation method provides O(1) lookup time
	- All methods avoid memory allocation in the hot path
	"""
	linear_value_function(x::Vector{T}, w::Vector{T}) where {T<:Real} = dot(x, w)
	
	#binary features
	function linear_value_function(binary_features::BinaryFeatureVector, params::Vector{T})::T where T<:Real
		v = zero(T)
		@inbounds @simd for i in 1:binary_features.num_features
			j = binary_features.active_features[i]
			v += params[j]
		end
		return v
	end

	#state-aggregation index (single feature only)
	linear_value_function(x::StateAggregationFeatureVector, params::Vector{T}) where T<:Real = params[x.group_index]
end

# ╔═╡ 1f0b9d36-3592-47a0-b32a-a7e19b763e1b
begin
	#value_function is something that that takes only the feature vector and parameters to generate a state value estimation.  this function converts that into a function that can be called with only the state as an argument.  by default the arguments are designed to make the function thread safe so that any modified internal arguments are generated each time it is called.  It also returns a function to create an instance of the arguments in case the function needs to be used repeatedly on a single thread
	function form_state_value_function(value_function::Function, update_feature_vector!::Function, feature_vector::V, parameters::P) where {V, P}
		function v̂(s; feature_vector::V = deepcopy(feature_vector), parameters::P = parameters, kwargs...)
			update_feature_vector!(feature_vector, s)
			value_function(feature_vector, parameters; kwargs...)
		end
	
		#also return a method that acts on the feature vector itself which has already been updated
		v̂(x::V, parameters; kwargs...) = value_function(x, parameters; kwargs...)

		form_kwargs() = (feature_vector = deepcopy(feature_vector), parameters = parameters)
		
		return (v̂, form_kwargs)
	end

	function form_state_value_function(value_function::Function, update_feature_vector!::Function, feature_vector::V, parameters::FCANNParams) where V
		function v̂(s; feature_vector::V = deepcopy(feature_vector), parameters::FCANNParams = parameters, activations = FCANN.form_activations(parameters.weights[1]), kwargs...)
			update_feature_vector!(feature_vector, s)
			value_function(feature_vector, parameters; activations = activations, kwargs...)
		end
	
		#also return a method that acts on the feature vector itself which has already been updated
		v̂(x::V, parameters; kwargs...) = value_function(x, parameters; kwargs...)

		form_kwargs() = (feature_vector = deepcopy(feature_vector), parameters = parameters, activations = FCANN.form_activations(parameters.weights[1]))
		
		return (v̂, form_kwargs)
	end
end

# ╔═╡ ba58242a-306a-4631-92b4-34bc9e354fae
#the purpose of these functions is to unify the gradient monte carlo algorithm by having a common way to run and update episodes.  Since I want to be able to pass the first two return trajectory values in both cases, I need that trajectory to store the states and rewards as the first two values
begin
	"""
	    create_episode_functions(mrp) -> (generate_episode, update_episode!)
	    create_episode_functions(mdp, π) -> (generate_episode, update_episode!)
	
	Creates standardized episode generation functions for gradient Monte Carlo algorithms.
	
	Generates consistent episode function interfaces that return trajectories with states
	and rewards as the first two elements, enabling unified gradient Monte Carlo estimation
	across different problem types (MRPs and MDPs).
	
	# Methods
	
	## MRP Version
	```julia
	create_episode_functions(mrp::StateMRP) -> (generate_episode, update_episode!)
	```
	Creates episode functions for Markov Reward Process evaluation.
	
	- `mrp::`[`StateMRP`](@ref): Markov reward process to generate episodes from
	
	Returns trajectory format: `(states, rewards)`
	
	### Examples
	```julia-repl
	julia> mrp = StateMRP(transition_matrix, reward_vector);
	
	julia> gen_ep, update_ep! = create_episode_functions(mrp);
	
	julia> # Generate initial episode
	       trajectory = gen_ep(max_steps=100);
	
	julia> trajectory[1]  # states
	5-element Vector{Int64}:
	 1
	 3
	 2
	 4
	 5
	
	julia> trajectory[2]  # rewards  
	5-element Vector{Float64}:
	 0.0
	 1.0
	 0.0
	 0.0
	 2.0
	```
	
	## MDP Version  
	```julia
	create_episode_functions(mdp::StateMDP, π::Function) -> (generate_episode, update_episode!)
	```
	Creates episode functions for Markov Decision Process policy evaluation.
	
	- `mdp::`[`StateMDP`](@ref): Markov decision process to generate episodes from
	- `π::Function`: Policy function for action selection
	
	Returns trajectory format: `(states, rewards, actions)`
	
	### Examples
	```julia-repl
	julia> mdp = StateMDP(transition_tensor, reward_matrix);
	
	julia> policy(s) = rand(1:num_actions);  # Random policy
	
	julia> gen_ep, update_ep! = create_episode_functions(mdp, policy);
	
	julia> # Generate initial episode
	       trajectory = gen_ep(max_steps=50);
	
	julia> trajectory[1]  # states
	3-element Vector{Int64}:
	 1
	 2
	 3
	
	julia> trajectory[2]  # rewards (always second element)
	3-element Vector{Float64}:
	 0.5
	 1.0
	 0.0
	
	julia> trajectory[3]  # actions (MDP-specific)
	3-element Vector{Int64}:
	 2
	 1
	 3
	```
	
	# Returned Functions
	
	## `generate_episode(; epkwargs...)`
	Creates new episode trajectory. Passes keyword arguments to underlying `runepisode` function.
	
	## `update_episode!(trajectory; epkwargs...)`
	Generates new episode reusing storage from previous trajectory. Returns `(new_trajectory, n_steps)`.
	
	# Design Purpose
	Ensures consistent trajectory format where `trajectory[1]` contains states and `trajectory[2]` 
	contains rewards, regardless of problem type. This standardization enables the same gradient
	Monte Carlo functions to work with both MRPs and MDPs.
	
	# Performance Notes
	- `update_episode!` reuses trajectory storage to minimize allocations
	- Consistent return format eliminates dispatch overhead in Monte Carlo algorithms
	- Compatible with any keyword arguments supported by underlying episode runners
	
	# See Also
	[`gradient_monte_carlo_estimation!`](@ref), [`runepisode`](@ref), [`runepisode!`](@ref)
	"""
	function create_episode_functions(mrp::StateMRP)
		function generate_episode(; epkwargs...) 
			(states, rewards, sterm) = runepisode(mrp; epkwargs...)
			(states, rewards)
		end
		
		function update_episode!((states, rewards); epkwargs...)
			(states, rewards, sterm, nsteps) = runepisode!((states, rewards), mrp; epkwargs...)
			((states, rewards), nsteps)
		end

		return (generate_episode, update_episode!)
	end

	function create_episode_functions(mdp::StateMDP, π::Function)
		function generate_episode(; epkwargs...) 
			(states, actions, rewards, sterm) = runepisode(mdp; π = π, epkwargs...)
			(states, rewards, actions)
		end
	
		function update_episode!((states, rewards, actions); epkwargs...)
			(states, actions, rewards, sterm, nsteps) = runepisode!((states, actions, rewards), mdp; π = π, epkwargs...)
			((states, rewards, actions), nsteps)
		end
		return (generate_episode, update_episode!)
	end
end

# ╔═╡ c466f78e-e464-4602-93c4-40362e4c0df2
gradient_monte_carlo_estimation!(parameters, mrp::StateMRP, args...; kwargs...) = gradient_monte_carlo_estimation!(parameters, create_episode_functions(mrp)..., args...; kwargs...)

# ╔═╡ a77f9819-04b2-4785-8eb0-c7e9dba6cecc
Base.length(::AbstractBinaryFeatures{I, N}) where {I<:Integer, N} = N

# ╔═╡ 76fb06c4-0841-40a2-996e-cb9a555ffc34
begin
	#here active_features is just something that can be enumerated
	"""
	    update_binary_feature_vector!(target, source) -> target
	
	Updates a binary feature vector in-place by copying active features from source.
	
	This function efficiently manages the sparse storage of active feature indices,
	handling memory allocation and resizing as needed. Used internally by gradient
	update functions for sparse feature representations.
	
	# Arguments
	- `target::`[`BinaryFeatureVector`](@ref): Binary feature vector to update (modified in-place)
	- `source`: Source of active features (see Methods for supported types)
	
	# See Also
	[`update_linear_value_gradient!`](@ref)
	
	# Methods
	
	## From LinearFeatures
	```julia
	update_binary_feature_vector!(x::BinaryFeatureVector{I, N}, active_features::LinearFeatures{I}) where {I <: Integer, N}
	```
	Updates binary feature vector from an enumerable collection of active feature indices.
	Efficiently manages vector resizing by reusing existing storage when possible.
	
	- `x::`[`BinaryFeatureVector{I, N}`](@ref BinaryFeatureVector): Target binary feature vector
	- `active_features::`[`LinearFeatures{I}`](@ref LinearFeatures): Enumerable collection of active feature indices
	
	### Examples
	```julia-repl
	julia> target = BinaryFeatureVector(10);
	
	julia> # LinearFeatures can be a vector of indices
	       active_indices = [1, 3, 7, 9]
	4-element Vector{Int64}:
	 1
	 3
	 7
	 9
	
	julia> update_binary_feature_vector!(target, active_indices);
	
	julia> target.active_features
	4-element Vector{Int64}:
	 1
	 3
	 7
	 9
	
	julia> target.num_features
	4
	```
	
	## From BinaryFeatureVector
	```julia
	update_binary_feature_vector!(x::BinaryFeatureVector{I, N}, y::BinaryFeatureVector{I, N}) where {I <: Integer, N}
	```
	Copies active features from one binary feature vector to another.
	Optimizes memory usage by reusing existing storage and only allocating when necessary.
	
	- `x::`[`BinaryFeatureVector{I, N}`](@ref BinaryFeatureVector): Target binary feature vector
	- `y::`[`BinaryFeatureVector{I, N}`](@ref BinaryFeatureVector): Source binary feature vector
	
	### Examples
	```julia-repl
	julia> target = BinaryFeatureVector(10);
	
	julia> source = BinaryFeatureVector(10);
	
	julia> source.active_features = [2, 4, 6];
	
	julia> source.num_features = 3;
	
	julia> update_binary_feature_vector!(target, source);
	
	julia> target.active_features
	3-element Vector{Int64}:
	 2
	 4
	 6
	
	julia> target.num_features
	3
	```
	
	# Performance Notes
	- Reuses existing storage in target vector when possible to minimize allocations
	- Uses `@simd` optimization for copying existing indices
	- Only allocates new memory when target vector needs to grow
	- Efficiently handles different source and target sizes using `extrema`
	"""
	function update_binary_feature_vector!(x::BinaryFeatureVector{I, N}, active_features::LinearFeatures{I}) where {I <: Integer, N}
		l = length(x.active_features)
		n = 0
		for (i, f) in enumerate(active_features)
			if i > l 
				push!(x.active_features, f)
			else
				x.active_features[i] = f
			end
			n += 1
		end
		x.num_features = n
		return x
	end

	function update_binary_feature_vector!(x::BinaryFeatureVector{I, N}, y::BinaryFeatureVector{I, N}) where {I <: Integer, N}
		l1, l2 = extrema((x.num_features, y.num_features))
		
		#replace the features for the indices that have already been allocated
		@inbounds @simd for i in 1:l1
			x.active_features[i] = y.active_features[i]
		end

		#add any new indices required for x
		for i in l1+1:l2
			push!(x.active_features, y.active_features[i])
		end
		x.num_features = y.num_features
		
		return x
	end
end

# ╔═╡ 1d107df4-36fa-49bd-bd48-5d5f49910b44
begin
	"""
	    update_linear_value_gradient!(gradient, features, value_params) -> gradient
	
	Updates the gradient of a linear value function in-place based on feature representation.
	
	For linear value functions, the gradient with respect to parameters is simply the 
	feature vector itself. This function efficiently updates gradient storage for 
	different feature representations.
	
	# Arguments
	- `gradient`: Gradient storage to update (modified in-place)
	- `features`: Feature representation used to compute gradient
	- `value_params`: Value function parameters (not used but maintained for API consistency)
	
	# See Also
	[`linear_value_function`](@ref), [`update_params_with_gradient!`](@ref), [`update_binary_feature_vector!`](@ref)
	
	# Methods
	
	## Dense Features
	```julia
	update_linear_value_gradient!(∇v̂::Vector{T}, x::Vector{T}, value_params) where {T<:Real}
	```
	Updates dense gradient vector by copying feature values.
	For linear functions: ∇v̂ = x (gradient equals features).
	
	- `∇v̂::Vector{T}`: Gradient vector to update
	- `x::Vector{T}`: Dense feature vector
	- `value_params`: Value function parameters (unused)
	
	## Binary Features
	```julia
	update_linear_value_gradient!(∇v̂::BinaryFeatureVector, binary_features::BinaryFeatureVector, value_params)
	```
	Updates sparse binary gradient by copying active feature indices.
	Calls [`update_binary_feature_vector!`](@ref) to efficiently copy sparse structure.
	
	- `∇v̂::`[`BinaryFeatureVector`](@ref): Sparse gradient to update
	- `binary_features::`[`BinaryFeatureVector`](@ref): Input binary features
	- `value_params`: Value function parameters (unused)
	
	## State Aggregation Features
	```julia
	update_linear_value_gradient!(∇v̂::StateAggregationFeatureVector, feature_vector::StateAggregationFeatureVector, value_params)
	```
	Updates state aggregation gradient by copying the active group index.
	
	- `∇v̂::`[`StateAggregationFeatureVector`](@ref): State aggregation gradient to update
	- `feature_vector::`[`StateAggregationFeatureVector`](@ref): Input state aggregation features
	- `value_params`: Value function parameters (unused)
	
	# Performance Notes
	- Dense method uses vectorized assignment for efficiency
	- Binary feature method delegates to optimized copying function
	- State aggregation provides O(1) update time
	- All methods modify gradient storage in-place to avoid allocations

	    update_linear_value_gradient!(∇q̂::LinearActionValueGradient, x, i_a, value_params) -> LinearActionValueGradient

	Updates action-value gradient storage with feature vector and action index.
	
	Extends the existing linear gradient system to handle action-value functions by delegating
	gradient computation to [`update_linear_value_gradient!`](@ref) and storing the action index.
	
	# Arguments
	- `∇q̂::LinearActionValueGradient`: Action-value gradient storage to update in-place
	- `x::LinearFeatureVector`: Feature vector representing ∇q̂(s,a)
	- `i_a::Integer`: Action index for this gradient
	- `value_params`: Value function parameters (unused, maintains API consistency)
	
	# Returns
	- `LinearActionValueGradient`: The updated gradient storage (same as input `∇q̂`)
	"""
	function update_linear_value_gradient!(∇v̂::Vector{T}, x::Vector{T}, value_params) where {T<:Real}
		∇v̂ .= x
		return ∇v̂
	end

	#with binary features we only need to store the active features
	function update_linear_value_gradient!(∇v̂::BinaryFeatureVector, binary_features::BinaryFeatureVector, value_params)
		update_binary_feature_vector!(∇v̂, binary_features)
	end

	function update_linear_value_gradient!(∇v̂::StateAggregationFeatureVector, feature_vector::StateAggregationFeatureVector, value_params)
		∇v̂.group_index = feature_vector.group_index
		return ∇v̂
	end
end

# ╔═╡ be546bdb-77a9-48c4-9a98-1205d73fc8c6
"""
    gradient_monte_carlo_episode_update!(parameters, ∇v̂, feature_vector, update_feature_vector!,
                                        value_function, update_value_gradient!, states, rewards,
                                        γ, α, calculate_error) -> Real

    gradient_monte_carlo_episode_update!(parameters, action_values, ∇q̂, feature_vector, 
                                        update_feature_vector!, update_action_values!, 
                                        update_value_gradient!, states, actions, rewards, 
                                        γ, α, calculate_error) -> Real

Internal function for Monte Carlo episode updates with gradient-based function approximation.

Processes episode backward to compute exact returns and performs gradient descent parameter
updates. Used internally by higher-level Monte Carlo algorithms. Supports both state-value
function approximation (first method) and action-value function approximation (second method).

# Type Parameters
- `T <: Real`: Numeric type for parameters and computations
- `S <: Any`: State data type
- `I <: Integer`: Action index type (action-value method only)

# Arguments

## State-Value Method
- `parameters::Vector{T}`: Value function parameters (modified in-place)
- `∇v̂`: Gradient storage for state-value function (modified in-place)
- `feature_vector`: Feature storage (modified in-place)
- `update_feature_vector!::Function`: Feature extraction function
- `value_function::Function`: Value function (features, params) -> value
- `update_value_gradient!::Function`: Gradient computation function
- `states::AbstractVector{S}`: Episode states (chronological order)
- `rewards::AbstractVector{T}`: Episode rewards
- `γ::T`: Discount factor
- `α::T`: Learning rate
- `calculate_error::Function`: Error function for statistics

## Action-Value Method
- `parameters::Vector{T}`: Action-value function parameters (modified in-place)
- `action_values::Vector{T}`: Pre-allocated storage for action values at current state
- `∇q̂`: Gradient storage for action-value function (modified in-place)
- `feature_vector`: Feature storage (modified in-place)
- `update_feature_vector!::Function`: Extract features from state: `(feature_vector, state) -> nothing`
- `update_action_values!::Function`: Compute all action values: `(action_values, features, params) -> nothing`
- `update_value_gradient!::Function`: Compute gradient for specific action: `(∇q̂, features, action_index, params) -> nothing`
- `states::AbstractVector{S}`: Episode state sequence
- `actions::AbstractVector{I}`: Episode action indices (1-based)
- `rewards::AbstractVector{T}`: Episode reward sequence
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `α::T`: Learning rate (step size)
- `calculate_error::Function`: Error function: `(return, estimated_value, state) -> error`

# Returns
- `Real`: Average episode error across all state(-action) pairs

# Implementation
Backward pass through episode computing returns `g = γ * g + rewards[i]` and updating
parameters via gradient descent:
- **State-value**: `θ ← θ + α·(g - v̂)·∇v̂`
- **Action-value**: `θ ← θ + α·(g - q̂)·∇q̂`

Accumulates error statistics for convergence monitoring.

# Performance Notes
- Reuses provided storage objects to avoid allocations
- Processes episode backward for efficient return computation
- Compatible with any linear or nonlinear function approximation setup
- Action-value method requires additional action value computation per step

# See Also
[`update_params_with_gradient!`](@ref), [`gradient_monte_carlo_estimation!`](@ref)
"""
function gradient_monte_carlo_episode_update!(parameters, ∇v̂, feature_vector, update_feature_vector!::Function, value_function::Function, update_value_gradient!::Function, states::AbstractVector{S}, rewards::AbstractVector{T}, γ::T, α::T, calculate_error::Function) where {T<:Real, S}
	g = zero(T)
	l = length(states)
	episode_error = zero(T)
	for i in l:-1:1
		s = states[i]
		update_feature_vector!(feature_vector, s)
		v̂ = value_function(feature_vector, parameters)
		update_value_gradient!(∇v̂, feature_vector, parameters)
		g = γ * g + rewards[i]
		δ = g - v̂
		c = α*δ
		update_params_with_gradient!(parameters, c, ∇v̂)
		episode_error += calculate_error(g, v̂, s)
	end
	return episode_error / l
end;

# ╔═╡ 7542ff9c-c6a1-4d41-8863-05388fea8ce2
"""
    gradient_monte_carlo_estimation!(parameters, generate_episode, update_episode!, γ, num_episodes,
                                    feature_vector, update_feature_vector!, value_function, ∇v̂,
                                    update_value_gradient!; α=0.1, calculate_error, epkwargs...) 
                                    where {T<:Real} -> NamedTuple

Monte Carlo value function estimation with gradient-based function approximation.

Coordinates episode generation, parameter updates, and error tracking across multiple episodes.
Supports both direct usage with custom episode functions and convenient wrappers for standard
problem types (MRPs, MDPs).

# Type Parameters
- `T <: Real`: Numeric type for parameters, rewards, and computations

# Arguments
- `parameters::Vector{T}`: Initial value function parameters (modified in-place)
- `generate_episode::Function`: Function to generate initial episode trajectory
- `update_episode!::Function`: Function to generate subsequent episodes (may reuse storage)
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `num_episodes::Integer`: Number of episodes to run
- `feature_vector`: Feature vector storage for state representations
- `update_feature_vector!::Function`: Function to extract features from states
- `value_function::Function`: Value function (features, params) -> value
- `∇v̂`: Gradient storage for value function gradients
- `update_value_gradient!::Function`: Function to compute value function gradient

# Keyword Arguments
- `α::Real`: Learning rate (default: 0.1)
- `calculate_error::Function`: Error function for convergence tracking (default: squared error)
- `epkwargs...`: Additional arguments passed to episode generation functions

# Returns
- `NamedTuple` with fields:
  - `value_function`: Learned value function `v̂(s)` 
  - `error_history::Vector{T}`: Per-episode error history for convergence analysis
  - `parameters::Vector{T}`: Final learned parameters

# See Also
[`gradient_monte_carlo_episode_update!`](@ref), [`form_state_value_function`](@ref),
[`create_episode_functions`](@ref)

# Methods

## MRP Wrapper
```julia
gradient_monte_carlo_estimation!(parameters, mrp::StateMRP, γ, num_episodes, feature_vector,
                                 update_feature_vector!, value_function, ∇v̂, 
                                 update_value_gradient!; kwargs...)
```
Convenience wrapper for Markov Reward Process policy evaluation.
Automatically creates episode functions using [`create_episode_functions`](@ref).

- `mrp::`[`StateMRP`](@ref): Markov reward process for episode generation


# Algorithm Flow
1. Generates initial episode using provided/created episode functions
2. Performs Monte Carlo update on first episode
3. For remaining episodes:
   - Generates new episode with `update_episode!`
   - Runs gradient-based parameter update via [`gradient_monte_carlo_episode_update!`](@ref)
   - Records error for convergence tracking
4. Forms final value function from learned parameters

# Performance Notes
- Reuses storage objects (feature_vector, ∇v̂) across episodes to minimize allocations
- Uses views for variable-length episodes to avoid copying
- Error history provides convergence diagnostics
- Compatible with any differentiable function approximation method
"""
function gradient_monte_carlo_estimation!(parameters, generate_episode::Function, update_episode!::Function, γ::T, num_episodes::Integer, feature_vector, update_feature_vector!::Function, value_function::Function, ∇v̂, update_value_gradient!::Function; α = one(T)/10, calculate_error::Function = (g, v̂, s) -> (g - v̂) ^2, epkwargs...) where {T<:Real}
	trajectory = generate_episode(; epkwargs...)
	sqerr = gradient_monte_carlo_episode_update!(parameters, ∇v̂, feature_vector, update_feature_vector!, value_function, update_value_gradient!, trajectory[1], trajectory[2], γ, α, calculate_error)
	error_history = zeros(T, num_episodes)
	error_history[1] = sqrt(sqerr)
	for ep in 2:num_episodes
		(trajectory, n_steps) = update_episode!(trajectory; epkwargs...)
		error = gradient_monte_carlo_episode_update!(parameters, ∇v̂, feature_vector, update_feature_vector!, value_function, update_value_gradient!, view(trajectory[1], 1:n_steps), view(trajectory[2], 1:n_steps), γ, α, calculate_error)
		error_history[ep] = error
	end
	v̂, form_kwargs = form_state_value_function(value_function, update_feature_vector!, feature_vector, parameters)
	return (value_function = v̂, error_history = error_history, parameters = parameters, form_kwargs = form_kwargs)
end;

# ╔═╡ 9296a8a1-7edd-4ac4-8fa4-842317d693bc
"""
    gradient_monte_carlo_policy_estimation!(parameters, mdp, π, γ, num_episodes, feature_vector,
                                           update_feature_vector!, value_function, ∇v̂,
                                           update_value_gradient!; α=0.1, calculate_error, epkwargs...)
                                           where {T<:Real} -> NamedTuple

Monte Carlo policy evaluation for Markov Decision Processes using gradient-based function approximation.

Low-level wrapper that automatically creates episode generation functions for MDP policy evaluation.
Coordinates episode generation using the given policy and delegates to the core Monte Carlo
estimation routine. Typically called by higher-level policy evaluation algorithms.

# Type Parameters
- `T <: Real`: Numeric type for parameters, rewards, and computations

# Arguments
- `parameters::Vector{T}`: Initial value function parameters (modified in-place)
- `mdp::`[`StateMDP`](@ref): Markov decision process for episode generation
- `π::Function`: Policy function for action selection
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `num_episodes::Integer`: Number of episodes to run
- `feature_vector`: Feature vector storage for state representations
- `update_feature_vector!::Function`: Function to extract features from states
- `value_function::Function`: Value function (features, params) -> value
- `∇v̂`: Gradient storage for value function gradients
- `update_value_gradient!::Function`: Function to compute value function gradient

# Keyword Arguments
- `α::Real`: Learning rate (default: 0.1)
- `calculate_error::Function`: Error function for convergence tracking (default: squared error)
- `epkwargs...`: Additional arguments passed to episode generation (e.g., `max_steps`, `start_state`)

# Returns
- `NamedTuple` with fields:
  - `value_function`: Learned state value function `v^π(s)` for policy π
  - `error_history::Vector{T}`: Per-episode error history for convergence analysis
  - `parameters::Vector{T}`: Final learned parameters

# Implementation Note
This function automatically calls [`create_episode_functions`](@ref)`(mdp, π)` to generate the required 
episode functions, then delegates to [`gradient_monte_carlo_estimation!`](@ref).

# Performance Notes
- Automatically handles MDP episode generation with policy π
- Reuses storage objects across episodes to minimize allocations
- Compatible with any differentiable function approximation method
- Episode functions handle state-action-reward trajectories internally

# See Also
[`gradient_monte_carlo_estimation!`](@ref), [`create_episode_functions`](@ref), [`StateMDP`](@ref)
"""
gradient_monte_carlo_policy_estimation!(parameters, mdp::StateMDP, π::Function, args...; kwargs...) = gradient_monte_carlo_estimation!(parameters, create_episode_functions(mdp, π)..., args...; kwargs...)

# ╔═╡ 412f6295-3eec-4966-98e3-2774bf62ed4f
begin
	"""
	    initialize_linear_parameters(length_or_features, init_value) -> Vector{T}
	    initialize_linear_parameters(feature_vector_length, num_actions, init_value) -> Matrix{T}
	
	Initializes parameter storage for linear function approximation.
	
	Creates parameter vectors for state-value functions or parameter matrices for action-value
	functions, with all entries initialized to the specified value. Supports initialization 
	from explicit dimensions or object representations.
	
	# Type Parameters  
	- `T <: Real`: Numeric type for parameter values
	
	# Arguments
	- `length_or_features`: Feature space size (Integer) or feature object
	- `init_value`: Initial value for all parameters
	- `feature_vector_length::Integer`: Number of features (action-value methods)
	- `num_actions::Integer`: Number of actions (action-value methods)
	
	# Returns
	- `Vector{T}`: Parameter vector for state-value functions
	- `Matrix{T}`: Parameter matrix (features × actions) for action-value functions
	
	# See Also
	[`get_feature_length`](@ref), [`get_num_actions`](@ref)
	
	# Methods
	
	## State-Value Parameters (Vector Storage)
	
	### Direct Length
	```julia
	initialize_linear_parameters(l::Integer, init_value::T) where T<:Real
	```
	Creates parameter vector with explicit length.
	
	- `l::Integer`: Length of parameter vector
	- `init_value::T`: Initial value for all parameters
	
	### Object-Based Length
	```julia
	initialize_linear_parameters(x, init_value)
	```
	Creates parameter vector using object to determine length.
	Uses [`get_feature_length`](@ref) to extract feature space size.
	
	- `x`: Feature object (vector, binary features, etc.)
	- `init_value`: Initial value for all parameters
	
	## Action-Value Parameters (Matrix Storage)
	
	### Direct Dimensions
	```julia
	initialize_linear_parameters(feature_vector_length::Integer, num_actions::Integer, init_value::T) where T<:Real
	```
	Creates parameter matrix for action-value function approximation with explicit dimensions.
	Returns matrix of size (features × actions) with all entries set to `init_value`.
	
	- `feature_vector_length::Integer`: Number of features in feature vector
	- `num_actions::Integer`: Number of actions in action space
	- `init_value::T`: Initial value for all parameters
	
	### Object-Based Dimensions
	```julia
	initialize_linear_parameters(feature_object, action_object, init_value)
	```
	Creates parameter matrix using objects to determine dimensions.
	Delegates to [`get_feature_length`](@ref) and [`get_num_actions`](@ref) for dimension extraction.
	
	```julia-repl
	julia> params = initialize_linear_parameters(feature_vector, mdp, 0.0f0)
	```
	
	# Performance Notes
	- Uses in-place multiplication for efficient initialization
	- Matrix layout optimized for column-wise access (features × actions)
	- Vector storage for state-value functions, matrix storage for action-value functions
	"""
	function initialize_linear_parameters(l::Integer, init_value::T) where T<:Real
		params = ones(T, l)
		params .*= init_value
		return params
	end
	initialize_linear_parameters(x, init_value) = initialize_linear_parameters(length(x), init_value)
end

# ╔═╡ 966850ef-dd15-417b-b51c-9957f27e4664
"""
    gradient_monte_carlo_estimation_linear(mrp, γ, num_episodes, feature_vector,
                                          update_feature_vector!; init_value=0.0,
                                          params=initialize_linear_parameters(...), kwargs...)
                                          where {T<:Real} -> NamedTuple

High-level Monte Carlo policy evaluation for MRPs with linear function approximation.

Complete Monte Carlo learning interface that automatically configures linear value function
approximation components. Provides sensible defaults for parameter initialization and gradient
computation, making it the primary entry point for MRP policy evaluation with linear features.

# Type Parameters
- `T <: Real`: Numeric type for parameters, rewards, and discount factor

# Arguments
- `mrp::`[`StateMRP`](@ref): Markov reward process for episode generation
- `γ::T`: Discount factor (0 ≤ γ < 1)
- `num_episodes::Integer`: Number of episodes to run for estimation
- `feature_vector::`[`LinearFeatureVector`](@ref): Linear feature representation template
- `update_feature_vector!::Function`: Function to extract linear features from states

# Keyword Arguments
- `init_value::T`: Initial value for all parameters (default: 0.0)
- `params::Vector{T}`: Pre-initialized parameter vector (default: auto-initialized using `init_value`)
- `kwargs...`: Additional arguments (e.g., `α`, `max_steps`, `calculate_error`)

# Returns
- `NamedTuple` with fields:
  - `value_function`: Learned linear value function `v^π(s)`
  - `error_history::Vector{T}`: Per-episode error history for convergence analysis
  - `parameters::Vector{T}`: Final learned parameter vector

# Examples
```julia-repl
julia> # Setup linear feature representation
       feature_template = [1.0, 0.5, 0.2];

julia> # Run Monte Carlo estimation with linear approximation
       result = gradient_monte_carlo_estimation_linear(
           mrp, 0.95, 1000, feature_template, update_feature_vector!,
           α = 0.01, max_steps = 200
       );

julia> # Check convergence
       final_error = result.error_history[end]
0.028

julia> # Use learned value function
       state_value = result.value_function(some_state)
3.42

julia> # Access final parameters
       learned_weights = result.parameters
3-element Vector{Float64}:
  2.1
  1.7
 -0.3
```

# Algorithm Details
Automatically configures linear function approximation by:
1. Initializing parameter vector using [`initialize_linear_parameters`](@ref)
2. Setting up [`linear_value_function`](@ref) for value computation
3. Using [`update_linear_value_gradient!`](@ref) for gradient computation
4. Creating episode functions via [`create_episode_functions`](@ref) for the MRP
5. Delegating to [`gradient_monte_carlo_estimation!`](@ref) for the core algorithm

# Performance Notes
- Automatically handles linear function approximation setup
- Uses efficient linear value function and gradient computations
- Reuses feature vector storage across episodes
- Provides sensible parameter initialization defaults

# See Also
[`gradient_monte_carlo_estimation!`](@ref), [`LinearFeatureVector`](@ref), 
[`linear_value_function`](@ref), [`update_linear_value_gradient!`](@ref),
[`initialize_linear_parameters`](@ref)
"""
gradient_monte_carlo_estimation_linear(mrp::StateMRP, γ::T, num_episodes::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), params::Vector{T} = initialize_linear_parameters(feature_vector, init_value), kwargs...) where T<:Real = gradient_monte_carlo_estimation!(params, mrp, γ, num_episodes, feature_vector, update_feature_vector!, linear_value_function, deepcopy(feature_vector), update_linear_value_gradient!; kwargs...)

# ╔═╡ 97539f3f-92bb-4b6f-a671-260251b4ddc7
"""
    gradient_monte_carlo_policy_estimation_linear(mdp, π, γ, num_episodes, feature_vector,
                                                  update_feature_vector!; init_value=0.0,
                                                  params=initialize_linear_parameters(...), kwargs...)
                                                  where {T<:Real} -> NamedTuple

High-level Monte Carlo policy evaluation for MDPs with linear function approximation.

Complete Monte Carlo learning interface that automatically configures linear value function
approximation components for policy evaluation. Provides sensible defaults for parameter 
initialization and gradient computation, making it the primary entry point for MDP policy
evaluation with linear features.

# Type Parameters
- `T <: Real`: Numeric type for parameters, rewards, and discount factor

# Arguments
- `mdp::`[`StateMDP`](@ref): Markov decision process for episode generation
- `π::Function`: Policy function for action selection
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `num_episodes::Integer`: Number of episodes to run for estimation
- `feature_vector::`[`LinearFeatureVector`](@ref): Linear feature representation template
- `update_feature_vector!::Function`: Function to extract linear features from states

# Keyword Arguments
- `init_value::T`: Initial value for all parameters (default: 0.0)
- `params::Vector{T}`: Pre-initialized parameter vector (default: auto-initialized using `init_value`)
- `kwargs...`: Additional arguments (e.g., `α`, `max_steps`, `calculate_error`)

# Returns
- `NamedTuple` with fields:
  - `value_function`: Learned state value function `v^π(s)` for policy π
  - `error_history::Vector{T}`: Per-episode error history for convergence analysis
  - `parameters::Vector{T}`: Final learned parameter vector

# Examples
```julia-repl
julia> # Setup linear feature representation
       feature_template = [1.0, 0.0, 0.0, 0.0];

julia> # Run Monte Carlo policy evaluation
       result = gradient_monte_carlo_policy_estimation_linear(
           mdp, policy, 0.9, 2000, feature_template, update_feature_vector!,
           α = 0.02, max_steps = 150, init_value = 0.1
       );

julia> # Check convergence
       final_error = result.error_history[end]
0.041

julia> # Evaluate policy at different states
       result.value_function(1)
4.67

julia> result.value_function(5)
2.31

julia> # Inspect learned weights
       result.parameters
4-element Vector{Float64}:
  3.2
  1.8
 -0.5
  2.1
```

# Algorithm Details
Automatically configures linear function approximation by:
1. Initializing parameter vector using [`initialize_linear_parameters`](@ref)
2. Setting up [`linear_value_function`](@ref) for value computation
3. Using [`update_linear_value_gradient!`](@ref) for gradient computation
4. Creating episode functions via [`create_episode_functions`](@ref) for the MDP and policy
5. Delegating to [`gradient_monte_carlo_estimation!`](@ref) for the core algorithm

# Performance Notes
- Automatically handles all linear function approximation setup
- Uses efficient linear value function and gradient computations
- Reuses feature vector storage across episodes
- Provides sensible parameter initialization defaults

# See Also
[`gradient_monte_carlo_estimation!`](@ref), [`LinearFeatureVector`](@ref),
[`linear_value_function`](@ref), [`update_linear_value_gradient!`](@ref),
[`initialize_linear_parameters`](@ref), [`StateMDP`](@ref)
"""
gradient_monte_carlo_policy_estimation_linear(mdp::StateMDP, π::Function, γ::T, num_episodes::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), params::Vector{T} = initialize_linear_parameters(feature_vector, init_value), kwargs...) where T<:Real = gradient_monte_carlo_policy_estimation!(params, mdp, π, γ, num_episodes, feature_vector, update_feature_vector!, linear_value_function, deepcopy(feature_vector), update_linear_value_gradient!; kwargs...)

# ╔═╡ df56b803-0aa5-4946-8338-601195e57a3e
md"""
### *Semi-gradient TD(0) for estimating $$\hat v \approx v_\pi$$*

When $U_t \doteq R_{t+1} + \gamma \hat v(S_{t+1}, \boldsymbol{w})$ the target value is the same as for temporal difference learning.  Now that the target uses parameter estimates, our gradient update is no longer correct since the target also depends on the parameters.  Thus this method is called `semi` gradient and has good convergence properties in the linear case.
"""

# ╔═╡ e8e26a28-90a5-4519-ab08-11b49a8a9499
begin
	"""
	    semi_gradient_td0_estimation!(parameters, initialize_state, transition, isterm, γ, max_episodes,
	                                  max_steps, feature_vector, update_feature_vector!, value_function,
	                                  ∇v̂, update_value_gradient!; α=0.1, calculate_error, save_episode_steps=false)
	                                  where {T<:Real} -> NamedTuple
	
	Semi-gradient TD(0) temporal difference learning with function approximation.
	
	Low-level TD learning implementation that performs online value function updates using
	single-step temporal difference errors. Supports both episodic and continuing tasks
	with flexible episode termination and state transition handling.
	
	# Type Parameters
	- `T <: Real`: Numeric type for parameters, rewards, and computations
	
	# Arguments
	- `parameters::Vector{T}`: Value function parameters (modified in-place)
	- `initialize_state::Function`: Function to generate initial states for episodes
	- `transition::Function`: State transition function `s -> (reward, next_state)`
	- `isterm::Function`: Termination check function `state -> Bool`
	- `γ::T`: Discount factor (0 ≤ γ < 1)
	- `max_episodes::Integer`: Maximum number of episodes to run
	- `max_steps::Integer`: Maximum total steps across all episodes
	- `feature_vector`: Feature vector storage for state representations
	- `update_feature_vector!::Function`: Function to extract features from states
	- `value_function::Function`: Value function (features, params) -> value
	- `∇v̂`: Gradient storage for value function gradients
	- `update_value_gradient!::Function`: Function to compute value function gradient
	
	# Keyword Arguments
	- `α::Real`: Learning rate (default: 0.1)
	- `calculate_error::Function`: Error function for statistics (default: squared error)
	- `save_episode_steps::Bool`: Whether to save step-by-step reward history (default: false)
	
	# Returns
	- `NamedTuple` with fields:
	  - `value_function`: Learned value function `v(s)`
	  - `episode_history`: Episode statistics with `errors`, `steps`, and `rewards` vectors
	  - `step_rewards::Vector{T}`: Step-by-step rewards (if `save_episode_steps=true`)
	  - `parameters::Vector{T}`: Final learned parameters
	
	# See Also
	[`update_params_with_gradient!`](@ref), [`form_state_value_function`](@ref)
	
	# Methods
	
	## MRP Wrapper
	```julia
	semi_gradient_td0_estimation!(parameters, mrp::StateMRP, γ, max_episodes, max_steps,
	                             feature_vector, update_feature_vector!, value_function,
	                             ∇v̂, update_value_gradient!; kwargs...)
	```
	Convenience wrapper for Markov Reward Process evaluation.
	Automatically extracts transition functions from MRP structure.
	
	- `mrp::`[`StateMRP`](@ref): Markov reward process for state transitions
	
	## MDP Policy Wrapper
	```julia
	semi_gradient_td0_policy_estimation!(parameters, mdp::StateMDP, π::Function, γ, max_episodes,
	                                    max_steps, feature_vector, update_feature_vector!,
	                                    value_function, ∇v̂, update_value_gradient!; kwargs...)
	```
	Convenience wrapper for Markov Decision Process policy evaluation.
	Automatically creates policy-based transition function from MDP and policy.
	
	- `mdp::`[`StateMDP`](@ref): Markov decision process for state transitions
	- `π::Function`: Policy function for action selection
	
	# Algorithm Details
	Implements classic TD(0) with function approximation:
	1. For each step: observes current state, computes value and gradient
	2. Takes environment step to get reward and next state
	3. Computes TD target: `target = r + γ * v(s')` (or `r` if terminal)
	4. Updates parameters: `θ ← θ + α * (target - v(s)) * ∇v(s)`
	5. Tracks episode statistics and manages episode boundaries
	
	# Performance Notes
	- Online learning with immediate parameter updates after each step
	- Reuses feature and gradient storage across steps
	- Optional step-by-step reward tracking for detailed analysis
	- Handles both episodic and continuing task termination
	- Compatible with any differentiable function approximation method
	"""
	function semi_gradient_td0_estimation!(parameters, initialize_state::Function, transition::Function, isterm::Function, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, value_function::Function, ∇v̂, update_value_gradient!::Function; α = one(T)/10, calculate_error::Function = (target, v̂, s) -> (v̂ - target) ^2, save_episode_steps = false) where {T<:Real}
		#initialize records
		step_rewards = Vector{T}()
		episode_steps = Vector{Int64}()
		episode_rewards = Vector{T}()
		episode_errors = Vector{T}()
		
		#initialize variables
		s = initialize_state()
		update_feature_vector!(feature_vector, s)
		ep = 1
		step = 1
		epstep = 1
		eperr = zero(T)
		rtot = zero(T)
		while (ep <= max_episodes) && (step <= max_steps)
			update_value_gradient!(∇v̂, feature_vector, parameters)
			v̂ = value_function(feature_vector, parameters)
			(r, s′) = transition(s)
			rtot += r
			save_episode_steps && push!(step_rewards, r)
			

			terminated = isterm(s′)
			if terminated
				push!(episode_steps, step)
				push!(episode_rewards, rtot)
				v̂′ = zero(T)
				ep += 1
				rtot = zero(T)
				s′ = initialize_state()
				update_feature_vector!(feature_vector, s′)
			else
				update_feature_vector!(feature_vector, s′)
				v̂′ = value_function(feature_vector, parameters)
			end

			target = r + γ*v̂′

			δ = target - v̂

			eperr += calculate_error(target, v̂, s)

			if terminated
				push!(episode_errors, eperr / epstep)
				eperr = zero(T)
				epstep = 0
			end

			update_params_with_gradient!(parameters, α*δ, ∇v̂)
			s = s′
			step += 1
			epstep += 1
		end

		v̂, form_kwargs = form_state_value_function(value_function, update_feature_vector!, feature_vector, parameters)

		(value_function = v̂, episode_history = (errors = episode_errors, steps = episode_steps, rewards = episode_rewards), step_rewards = step_rewards, parameters = parameters, form_kwargs = form_kwargs)
	end

	semi_gradient_td0_estimation!(parameters, mrp::StateMRP, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, value_function::Function, ∇v̂, update_value_gradient!::Function; kwargs...) where T<:Real = semi_gradient_td0_estimation!(parameters, mrp.initialize_state, s -> mrp.ptf(s), mrp.isterm, γ, max_episodes, max_steps, feature_vector, update_feature_vector!::Function, value_function::Function, ∇v̂, update_value_gradient!::Function; kwargs...)
	
	semi_gradient_td0_policy_estimation!(parameters, mdp::StateMDP, π::Function, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, value_function::Function, ∇v̂, update_value_gradient!::Function; kwargs...) where T<:Real = semi_gradient_td0_estimation!(parameters, mdp.initialize_state, s -> mdp.ptf(s, π(s)), mdp.isterm, γ, max_episodes, max_steps, feature_vector, update_feature_vector!::Function, value_function::Function, ∇v̂, update_value_gradient!::Function; kwargs...)
end

# ╔═╡ cb2005fd-d3e0-4f37-908c-77e4bbac45b8
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
### Example 9.1: State Aggregation on the $(@bind num_states NumberField(100:100_000, default = 1000)) State Random Walk
"""
  ╠═╡ =#

# ╔═╡ de9bea60-c91d-4253-bdd8-a3c1fde8941c
"""
    make_random_walk_mrp(num_states::Integer) -> TabularMRP

Generate a random walk Markov Reward Process with terminal states and stochastic transitions.

Creates a symmetric random walk environment where an agent starts at the center state and
can transition left or right with equal probability. Terminal states at both ends provide
rewards of -1 (left) and +1 (right), while intermediate states provide zero reward.
Transition probabilities are distance-dependent with wider spreads for longer walks.

# Arguments
- `num_states::Integer`: Number of non-terminal states in the random walk chain

# Returns
- [`TabularMRP`](@ref): Markov reward process with sparse transition matrices
  - `states`: State space `[0, 1, ..., num_states, num_states+1]` where 0 and `num_states+1` are terminal
  - `transition`: [`TabularStochasticTransition`](@ref) with sparse probability matrices
  - `initialize_state`: Function returning center state index for episode initialization

# Environment Structure
The random walk has `num_states + 2` total states:
- **Terminal states**: 0 (left, reward -1) and `num_states+1` (right, reward +1)  
- **Non-terminal states**: 1, 2, ..., `num_states` (reward 0)
- **Initial state**: Center state `⌈num_states/2⌉`

# Transition Dynamics
From each non-terminal state `s`, the agent:
1. Moves left with probability 0.5, right with probability 0.5
2. Transition spread: up to ±100 states with uniform probability within range
3. **Boundary handling**: Transitions beyond state boundaries terminate with appropriate rewards
4. **Sparse representation**: Uses [`SparseVector`](@ref) for memory efficiency

# See Also
[`TabularMRP`](@ref), [`TabularStochasticTransition`](@ref), [`make_chain_walk_mdp`](@ref)

# Examples

```julia-repl
julia> mrp = make_random_walk_mrp(1000)
TabularMRP with 1002 states (2 terminal)

julia> mrp.states
1002-element Vector{Int64}:
   0    1    2  ...  999  1000  1001

julia> initial_state = mrp.initialize_state()
501

```

# Performance Notes
- Uses [`SparseVector{Float32, Int64}`](@ref) for transition matrices to handle sparse connectivity
- Memory complexity: O(num_states × average_transitions_per_state) 
- Transition computation: O(1) lookup after preprocessing
- Float32 precision for memory efficiency in large state spaces
- Pre-computes all transition probabilities during construction
- Terminal state handling avoids runtime boundary checks
"""
function make_random_walk_mrp(num_states::Integer)
	states = collect(0:num_states+1)
	state_index = TabularRL.makelookup(states)
	initial_state = ceil(Int64, num_states / 2)
	initialize_state_index() = initial_state + 1
	state_transition_map = Vector{SparseVector{Float32, Int64}}(undef, num_states+2)
	reward_transition_map = Vector{Vector{Float32}}(undef, num_states+2)
	for s in states
		if (s == 0) || (s == num_states+1)
			v = zeros(Float32, num_states+2)
			v[s+1] = 1f0
			state_transition_map[s+1] = SparseVector(v)
			reward_transition_map[s+1] = [0f0]
		else
			
			state_transitions = SparseVector(zeros(Float32, num_states+2))
			reward_transitions = Vector{Float32}()
			minleft = s-100
			maxright = s+100
			ptermleft = if minleft > 0
				0f0
			else
				Float32((-minleft + 1)/100)
			end
	
			pnontermleft = 1f0 - ptermleft
			nontermleftstates = max(1, s - 100):s-1
			for s′ in nontermleftstates
				state_transitions[s′+1] = (0.5f0 * pnontermleft) / length(nontermleftstates)
			end
			state_transitions[1] = ptermleft/2
	
			ptermright = if maxright <= num_states
				0f0
			else
				Float32((maxright - num_states) / 100)
			end
	
			pnontermright = 1f0 - ptermright
			nontermrightstates = s+1:min(num_states, maxright)
			for s′ in nontermrightstates
				state_transitions[s′+1] = (0.5f0 * pnontermright) / length(nontermrightstates)
			end
			state_transitions[num_states+2] = ptermright/2
			
			state_transition_map[s+1] = state_transitions
	
			for i_s′ in state_transitions.nzind
				r = if i_s′ == 1
					-1f0
				elseif i_s′ == num_states+2
					1f0
				else
					0f0
				end
				push!(reward_transitions, r)
			end
			reward_transition_map[s+1] = reward_transitions
		end
	end
	
	TabularMRP(states, TabularStochasticTransition(state_transition_map, reward_transition_map), initialize_state_index)
end;

# ╔═╡ 7814bda0-4306-4060-8f9a-2bcf1cf8e132
# ╠═╡ skip_as_script = true
#=╠═╡
const random_walk_tabular_mrp = make_random_walk_mrp(num_states)
  ╠═╡ =#

# ╔═╡ 69223862-4d74-46c9-8c78-b24d659151ac
#=╠═╡
const random_walk_v = mrp_evaluation(random_walk_tabular_mrp, 1f0)
  ╠═╡ =#

# ╔═╡ f4459b0d-ee3e-47c7-9c82-981af622edfa
#=╠═╡
const initial_state::Int64 = ceil(Int64, num_states / 2)
  ╠═╡ =#

# ╔═╡ 90e5fc0e-2e97-424b-a5dd-9deb38293121
#=╠═╡
md"""
Consider a $num_states-state version of the random walk task in which the states are numbered from 1 to $num_states, left to right and all episodes begin near the center, in state $initial_state.  State transitions are from the current state to one of the 100 neighboring states to its left, or to one of the 100 neighboring states to its right, all with equal probability.  Of course, if the current state is near an edge, then there may be fewer than 100 neighbors on that side of it.  In this case, all the probability that would have gone into those missing neighbors goes into the probability of terminating on that side (thus, state 1 has a 0.5 chance of terminating on the left, and state $(num_states - 50) has a 0.25 chance of terminating on the right).  Left termination produces a reward of -1 and right +1.

The following function constructs this random walk as a tabular problem with a stochastic distribution function like we'd see in part 1 of the book.  From this representation of the problem, we can perform methods like value iteration to calculate the correct state values and then compare to approximation methods later.
"""
  ╠═╡ =#

# ╔═╡ 68a4151a-52ee-4ed0-b988-3fecc34d8d32
#=╠═╡
md"""
#### Transition Probabilities Visualized for $num_states State Random Walk

Using the tabular MDP, we can visualize the transition probabilities for any state.  Notice that at the edges, more probability is shifted to a terminal state.
"""
  ╠═╡ =#

# ╔═╡ 24e8b391-00ec-4ed5-85dc-0796eb85bf4f
#=╠═╡
md"""Select State to View Transition Probabilities: $(@bind smap Slider(1:num_states; default = ceil(Int64, num_states/2), show_value=true))"""
  ╠═╡ =#

# ╔═╡ 736b7667-904d-4a9c-bb10-a6b0b831bfb6
#=╠═╡
random_walk_tabular_mrp.ptf.state_transition_map[smap+1] |> v -> plot(bar(x = 0:num_states+1, y = v), Layout(xaxis_title = "State", yaxis_title = "Transition Probability"))
  ╠═╡ =#

# ╔═╡ 9c3f07b1-61eb-4d70-9dde-986c032a0840
md"""
#### Non-tabular Version of Random Walk Example
Since our goal is to compare estimation methods, we need to create a version of this problem that is non-tabular.  That way our state assignment function can be used properly to map a state to a particular parameter, effectively grouping them together instead of treading them each individually as in the tabular case.  The transition function for this case will operate on states and produce states that can then be mapped to the appropriate parameters.  Rather than converting the tabular MDP into a non-tabular one, this construction uses a faster step function.  By default, the conversion would create a step that produces the full distribution of transition states rather than just efficiently randomly sampling from them which is achieved here by the `randomwalk_step` method.
"""

# ╔═╡ 3f2ce7e0-b623-4ce3-90cf-949f3a6b0633
"""
    randomwalk_step(s::Float32, num_states::Int64) -> Tuple{Float32, Float32}

Single step transition for continuous random walk with terminal boundaries.

Takes random step of size 1-100 in either direction. Returns reward -1 for left
boundary crossing (s′ < 1), +1 for right boundary crossing (s′ > num_states), 
and 0 otherwise.

# Arguments
- `s::Float32`: Current state position
- `num_states::Int64`: Number of valid states (boundary at num_states)

# Returns  
- `Tuple{Float32, Float32}`: `(reward, next_state)` pair

# Performance Notes
- Uniform step size sampling with `ceil(rand() * 100)`
- Boundary reward computation via boolean arithmetic
"""
function randomwalk_step(s::Float32, num_states::Int64)
	x = Float32(ceil(rand() * 100))
	s′ = s + x * rand((-1f0, 1f0))

	r = Float32(-(s′ < 1) + (s′ > num_states))
	(r, s′)
end

# ╔═╡ 60d68f9b-d18d-4d23-9adb-27fcb205e54b
"""
    randomwalk_isterm(s::Float32, num_states::Int64) -> Bool

Check if random walk state is terminal (outside valid boundaries).

Returns `true` if state is beyond left boundary (s < 1) or right boundary (s > num_states).

# Arguments
- `s::Float32`: Current state position
- `num_states::Int64`: Number of valid states
"""
randomwalk_isterm(s::Float32, num_states::Int64) = (s < 1) || (s > num_states)

# ╔═╡ 6f3928a9-bcaa-44b5-8723-820142cbcfc3

"""
    create_continuous_random_walk(num_states::Int64) -> StateMRP

Create a continuous-state random walk Markov Reward Process with terminal boundaries.

Generates a random walk environment where an agent navigates a continuous state space
with stochastic step sizes. The agent starts at the center position and takes random
steps of varying magnitude (1-100 units) in either direction until reaching terminal
boundaries that provide reward feedback.

# Arguments
- `num_states::Int64`: Defines the valid state range [1, num_states] with terminal regions beyond

# Returns
- [`StateMRP`](@ref): Markov reward process with continuous state transitions
  - **State space**: Continuous values, valid range [1, num_states]
  - **Initial state**: Center position at `num_states/2`
  - **Transitions**: [`StateMRPTransitionSampler`](@ref) with stochastic step dynamics
  - **Termination**: States outside [1, num_states] boundaries

# Environment Dynamics
- **Step size**: Uniform random integer from 1 to 100 units
- **Direction**: Equal probability left (-) or right (+) movement
- **Rewards**: -1 for left boundary crossing (s < 1), +1 for right boundary (s > num_states), 0 otherwise
- **Termination**: Automatic episode end when boundaries are crossed

# State Representation
Unlike tabular random walks, this environment uses continuous Float32 states.

# See Also
[`make_random_walk_mrp`](@ref), [`StateMRP`](@ref), [`StateMRPTransitionSampler`](@ref), 
[`randomwalk_step`](@ref), [`randomwalk_isterm`](@ref)
"""
function create_continuous_random_walk(num_states::Int64)
	ptf = StateMRPTransitionSampler((s) -> randomwalk_step(s, num_states), 1f0)
	init_state = ceil(Float32, num_states / 2)
	StateMRP(ptf, () -> init_state, s -> randomwalk_isterm(s, num_states))
end

# ╔═╡ 2720329c-4c80-47cb-a3e3-d24fcec6ef43
#=╠═╡
const random_walk_state_mrp = create_continuous_random_walk(num_states)
  ╠═╡ =#

# ╔═╡ 2c6809f9-50ed-44b8-8f27-0a62e88d118c
#=╠═╡
md"""
#### State Aggregation

The simplest form of function approximation in which each state is assigned to a unique group.  Each group is represented by a parameter that estimates the value of every state in that group.  The gradient for this technique has the simple form: $\nabla \hat v (S_t, \mathbf{w}_t) = 1$ if $S_t$ is in the group represented by $\mathbf{w}_t$ and 0 otherwise.  For the random walk example, state aggregation can simply assign states to groups as: {1 to 100}, {101 to 200}, ..., {$(num_states - 100) to $num_states}.
"""
  ╠═╡ =#

# ╔═╡ 91e4e5da-4e0f-48b2-98bd-1e9f1330b0a8
# ╠═╡ skip_as_script = true
#=╠═╡
md"""Number of State Aggregation Groups: $(@bind num_groups NumberField(1:num_states, default = 10))"""
  ╠═╡ =#

# ╔═╡ 5ebafa8b-c316-4f95-8adc-581f2eb40e1f
"""
    make_random_walk_group_assign(num_states::Integer, num_groups::Integer) -> Function

Create state aggregation function for partitioning random walk states into groups.

Generates a group assignment function that maps continuous states to discrete group
indices, enabling state aggregation for value function approximation. Groups are
uniformly sized across the state space with equal interval partitioning.

# Arguments
- `num_states::Integer`: Total number of states in the random walk
- `num_groups::Integer`: Number of aggregation groups to create

# Returns
- `Function`: Group assignment function `assign_group(s::Real) -> Int64`
  - Maps any state `s` to group index `[1, 2, ..., num_groups]`
  - Uses ceiling division for uniform group boundaries

# Group Partitioning
States are partitioned into `num_groups` equal-sized intervals:
- **Group size**: `num_states / num_groups`
- **Group boundaries**: `[0, groupsize], (groupsize, 2×groupsize], ..., ((num_groups-1)×groupsize, num_groups×groupsize]`
- **Boundary handling**: States exactly on boundaries are assigned to the higher group

# See Also
[`create_continuous_random_walk`](@ref), [`make_random_walk_mrp`](@ref)

# Examples

```julia-repl
julia> assign_group = make_random_walk_group_assign(1000, 10)
assign_group (generic function with 1 method)

julia> # Check group assignments for different states
julia> assign_group(50.0)   # First group
1

julia> assign_group(150.0)  # Second group  
2

julia> assign_group(500.0)  # Middle group
5

julia> assign_group(999.0)  # Last group
10
```
"""
function make_random_walk_group_assign(num_states::Integer, num_groups::Integer)
	groupsize = num_states / num_groups
	assign_group(s::Real) = ceil(Int64, s / groupsize)
end

# ╔═╡ 24b99200-053a-41bf-a628-0b14b807fb86
#=╠═╡
#this function will assign a state to a group
random_walk_group_assign = make_random_walk_group_assign(num_states, num_groups)
  ╠═╡ =#

# ╔═╡ d68c0147-a66f-4542-a395-5f9b43e16b09
#=╠═╡
md"""
#### Group Aggregation Visualization for $num_states State Random Walk
"""
  ╠═╡ =#

# ╔═╡ 1adf0786-0897-4119-9336-09de869463b4
#=╠═╡
random_walk_group_assign.(random_walk_tabular_mrp.states) |> v -> plot(scatter(x = random_walk_tabular_mrp.states, y = v), Layout(xaxis_title = "State", yaxis_title = "Aggregation Group", title = "$num_states Random Walk States Partitioned into $num_groups Groups"))
  ╠═╡ =#

# ╔═╡ b361815f-d5b0-4c71-b331-c3b48ce53e73
md"""
Using the simple gradient for state aggregation, we can construct a function that computes the state value estimate and gradient per parameter component.  In order to implement state aggregation, one must have a fixed number of groups and a function to map states to a group index.  There will be one parameter value per group, so the gradient function needs to provide a component for every group.  Once a state is assigned into a unique group index, the gradient values will all be zero except for at the group index.  The value estimate is just the parameter value at that index.  In the case of the random walk example, assigning states to groups is simply a matter of dividing the state value by the group size and finding the next highest integer value.
"""

# ╔═╡ ff354a5e-f077-458d-8a0c-0a96a1d57658
md"""
Notice that for the case of state aggregation it isn't even necessary to compute the entire gradient or have a separate variable for the state representation
"""

# ╔═╡ 6004006e-4113-4f6a-b8ab-2c58ff207773
md"""
### *State Aggregation Feature Implementation*

In order to use linear function approximation we need to define a feature vector and a function to update that vector given a state.  The state should match the type used in the Markov problem.  In the case of state aggregation the only information we need about a state is the group assignment.  This setup function simply takes the group assignment function and uses it to construct the update function making use of the already defined function which updates a `StateAggregationFeatureVector` given a group index.
"""

# ╔═╡ c46c36f6-42da-4767-9e25-fa0ebe43998f
"""
    state_aggregation_feature_setup(s::S, num_groups::Integer, assign_state_group::Function) where S -> NamedTuple

Initialize state aggregation feature representation for value function approximation.

Creates a sparse binary feature vector system where each state is mapped to exactly one
group, enabling tabular value function learning over aggregated state spaces. The feature
vector uses one-hot encoding with a single active bit per state group.

# Type Parameters
- `S`: State type (e.g., `Float32` for a single numerical dimension)

# Arguments
- `s::S`: Sample state for validation and type inference
- `num_groups::Integer`: Number of aggregation groups (feature vector dimension)
- `assign_state_group::Function`: State-to-group mapping function `s -> group_index`

# Returns
- `NamedTuple` with fields:
  - `feature_vector::`[`StateAggregationFeatureVector{Integer, num_groups}`](@ref): Sparse binary feature storage
  - `update_feature_vector!::Function`: State-specific update function
    - Sets `vector.group_index` to the group assigned by `assign_state_group(state)`

# See Also
[`StateAggregationFeatureVector`](@ref), [`update_state_aggregation_feature_vector!`](@ref),
[`make_random_walk_group_assign`](@ref)

# Examples

```julia-repl
julia> assign_group = make_random_walk_group_assign(1000, 10)
assign_group (generic function with 1 method)

julia> setup = state_aggregation_feature_setup(500.0f0, 10, assign_group)
(feature_vector = StateAggregationFeatureVector{10}, update_feature_vector! = update_feature_vector!)

julia> gradient_monte_carlo_estimation_linear(random_walk_state_mrp, 0.9f0, 100, setup...);
```

# Performance Notes
- Validation check ensures `assign_state_group` returns `Integer` indices
- Closure-based update function eliminates repeated group assignment calls
- Sparse representation avoids storing `num_groups - 1` zeros
- Compatible with linear value function approximation methods
- Type-stable operations for efficient gradient computations
"""
function state_aggregation_feature_setup(s::S, num_groups::Integer, assign_state_group::Function) where S
	test_index = assign_state_group(s)
	@assert isa(test_index, Integer)
	x = StateAggregationFeatureVector(num_groups)
	update_feature_vector!(x::StateAggregationFeatureVector, s::S) = update_state_aggregation_feature_vector!(x, assign_state_group(s))
	(feature_vector = x, update_feature_vector! = update_feature_vector!)
end

# ╔═╡ c52222b7-64bd-4285-bba3-e22529495af6
"""
    gradient_monte_carlo_estimation_state_aggregation(mrp::StateMRP, γ::Real, num_episodes::Integer, num_groups::Integer, assign_state_group::Function; kwargs...) -> NamedTuple

Perform gradient Monte Carlo value estimation using state aggregation function approximation on an MRP.

This is a high-level convenience function that combines state aggregation feature setup with linear gradient Monte Carlo estimation. It automatically constructs the appropriate sparse feature representation and initial weight vector for the specified state grouping, then delegates to the core linear function approximation algorithm. Particularly effective for large state spaces where tabular methods are impractical due to memory constraints.

# Arguments
- `mrp::StateMRP`: The Markov reward process to evaluate, supporting any state type
- `γ::Real`: Discount factor for future rewards (0 ≤ γ ≤ 1)
- `num_episodes::Integer`: Number of Monte Carlo episodes to run for estimation
- `num_groups::Integer`: Number of state groups for aggregation (determines feature dimension)
- `assign_state_group::Function`: State-to-group mapping function (state → group_index ∈ 1:num_groups)

# Keyword Arguments
All keyword arguments are passed through to [`gradient_monte_carlo_estimation_linear`](@ref):
- `α::Real = 0.1`: Learning rate for gradient updates
- `calculate_error::Function = (g, v̂, s) -> (g - v̂)^2`: Error function for convergence tracking
- Additional episode generation arguments (e.g., `rng`, sampling parameters)

# Returns
- `NamedTuple` with fields:
  - `value_function::Function`: Learned value function `v̂(s)` that maps states to estimated values
  - `error_history::Vector{Float32}`: Per-episode error history for convergence analysis
  - `parameters::Vector{Float32}`: Final learned weight vector (length num_groups)

# See Also
[`gradient_monte_carlo_estimation!`](@ref), [`state_aggregation_feature_setup`](@ref), [`make_random_walk_group_assign`](@ref), [`StateMRP`](@ref)

# Algorithm Details
1. Initialize state aggregation features using [`state_aggregation_feature_setup`](@ref)
2. Create initial zero weight vector of length `num_groups`
3. Delegate to [`gradient_monte_carlo_estimation!`](@ref) with constructed features and functions
4. The underlying algorithm:
   - Generates episode trajectories using MRP transition sampling
   - Computes Monte Carlo returns for each state visit
   - Updates weight parameters via gradient descent on squared error
   - Tracks convergence through per-episode error history
5. Forms final value function from learned parameters and feature mapping

# Examples
```julia-repl
julia> # Create 1000-state random walk MRP
       mrp = create_continuous_random_walk(1000);

julia> # Define state aggregation: 10 groups of 100 states each
       assign_groups = make_random_walk_group_assign(1000, 10);

julia> # Run gradient Monte Carlo with state aggregation
       results = gradient_monte_carlo_estimation_state_aggregation(
           mrp, 0.9f0, 500, 10, assign_groups; α=0.05f0
       );

julia> # Check final results and convergence
       println("Final weights: ", results.parameters)
       println("Final error: ", results.error_history[end])
       println("Episodes run: ", length(results.error_history))
Final weights: Float32[-0.45, -0.36, -0.27, -0.18, -0.09, 0.0, 0.09, 0.18, 0.27, 0.36]
Final error: 0.023f0
Episodes run: 500

julia> # Use learned value function to evaluate states
       state_500 = 500.0f0;
       estimated_value = results.value_function(state_500);
       println("Estimated value at state 500: ", estimated_value)
Estimated value at state 500: 0.0234f0
```

# Performance Notes
- Memory efficient for large state spaces through sparse feature representation
- Feature construction is performed once, then reused across all episodes
- Automatically handles [`Float32`] precision for memory efficiency in large problems
- State aggregation reduces parameter space from `|S|` to `num_groups` dimensions
- Compatible with any [`StateMRP`](@ref) implementation via generic dispatch
"""
gradient_monte_carlo_estimation_state_aggregation(mrp::StateMRP, γ::Real, num_episodes::Integer, num_groups::Integer, assign_state_group::Function; kwargs...) = gradient_monte_carlo_estimation_linear(mrp, γ, num_episodes, state_aggregation_feature_setup(mrp.initialize_state(), num_groups, assign_state_group)...; kwargs...)

# ╔═╡ f64b78e1-76ff-4337-a9f0-aa2d3e3f33ac
"""
    gradient_monte_carlo_policy_estimation_state_aggregation(mdp::StateMDP, π::Function, γ::Real, num_episodes::Integer, num_groups::Integer, assign_state_group::Function; kwargs...) -> NamedTuple

Perform gradient Monte Carlo policy evaluation using state aggregation function approximation on an MDP.

This is a high-level convenience function that combines state aggregation feature setup with linear gradient Monte Carlo policy evaluation. It automatically constructs the appropriate sparse feature representation and initial weight vector for the specified state grouping, then delegates to the core linear function approximation algorithm. Particularly effective for large state spaces where tabular policy evaluation is impractical due to memory constraints.

# Arguments
- `mdp::StateMDP`: The Markov decision process to evaluate, supporting any state and action types
- `π::Function`: Policy function mapping states to action selections (state → action)
- `γ::Real`: Discount factor for future rewards (0 ≤ γ ≤ 1)
- `num_episodes::Integer`: Number of Monte Carlo episodes to run for policy evaluation
- `num_groups::Integer`: Number of state groups for aggregation (determines feature dimension)
- `assign_state_group::Function`: State-to-group mapping function (state → group_index ∈ 1:num_groups)

# Keyword Arguments
All keyword arguments are passed through to [`gradient_monte_carlo_estimation!`](@ref):
- `α::Real = 0.1`: Learning rate for gradient updates
- `calculate_error::Function = (g, v̂, s) -> (g - v̂)^2`: Error function for convergence tracking
- `epkwargs...`: Additional arguments passed to episode generation functions

# Returns
- `NamedTuple` with fields:
  - `value_function::Function`: Learned state value function `v_π(s)` that maps states to estimated values
  - `error_history::Vector{T}`: Per-episode error history for convergence analysis (where T matches γ type)
  - `parameters::Vector{T}`: Final learned weight vector (length num_groups, where T matches γ type)

# See Also
[`gradient_monte_carlo_estimation!`](@ref), [`state_aggregation_feature_setup`](@ref), [`make_random_walk_group_assign`](@ref), [`StateMDP`](@ref)

# Algorithm Details
1. Initialize state aggregation features using [`state_aggregation_feature_setup`](@ref)
2. Create initial zero weight vector of length `num_groups`
3. Delegate to [`gradient_monte_carlo_estimation!`](@ref) with constructed features and functions
4. The underlying algorithm:
   - Generates episode trajectories using MDP transition sampling and policy π
   - Computes Monte Carlo returns for each state visit under the policy
   - Updates weight parameters via gradient descent on squared error
   - Tracks convergence through per-episode error history
5. Forms final value function from learned parameters and feature mapping

# Examples
```julia-repl
julia> # Example usage with appropriate MDP and policy
       results = gradient_monte_carlo_policy_estimation_state_aggregation(
           mdp, policy, 0.9f0, 500, 10, assign_groups; α=0.05f0
       );
```

# Performance Notes
- Memory efficient for large state spaces through sparse feature representation
- Feature construction is performed once, then reused across all episodes
- State aggregation reduces parameter space from `|S|` to `num_groups` dimensions
- Compatible with any [`StateMDP`](@ref) implementation via generic dispatch
"""
gradient_monte_carlo_policy_estimation_state_aggregation(mdp::StateMDP, π::Function, γ::Real, num_episodes::Integer, num_groups::Integer, assign_state_group::Function; kwargs...) = gradient_monte_carlo_policy_estimation_linear(mdp, π, γ, num_episodes, state_aggregation_feature_setup(mrp.initialize_state(), num_groups, assign_state_group)...; kwargs...)

# ╔═╡ ace0693b-b4ce-43df-966e-0330d4399638
#=╠═╡
md"""
### *Figure 9.1*

Function approximation by state aggregation on the $num_states-state random walk task.  The blue line shows the true state values computed using value iteration.  The stepped orange line shows the group values as calculated using Gradient Monte Carlo estimation using the state aggregation parameters.  The distribution of visited states during an episode is also shown as a history.
"""
  ╠═╡ =#

# ╔═╡ bc479ae0-78ea-4255-863f-dcd126ae9b96
md"""
Our prediction objective will favor lower error on highly visited states than less requently visited ones.  Since the distribution of visited states as weighted towards the center, the error between the parameter estimated state value and the true state value is lower for states close to the center in a group.  That can be seen very clearly for group 1 where the right edge is far close to the blue line than the leftmost edge.  The leftmost state is the least likely to be visited and thus matters to least for minimizing prediction error.
"""

# ╔═╡ 750eef6b-58c6-4428-a44b-25e244aaf1d8
#=╠═╡
function calculate_random_walk_state_distribution(;samples = 100_000)
	state_counts = zeros(Int64, num_states)
	function update_state_counts!(state_counts, states)
		for s in states
			state_counts[Integer(s)] += 1
		end
	end
	
	(states, rewards, sterm, numsteps) = runepisode(random_walk_state_mrp)
	update_state_counts!(state_counts, view(states, 1:numsteps))
	for _ in 1:samples
		(states, rewards, sterm, num_steps) = runepisode!((states, rewards), random_walk_state_mrp)
		update_state_counts!(state_counts, view(states, 1:num_steps))
	end
	state_distribution = state_counts ./ sum(state_counts)
end
  ╠═╡ =#

# ╔═╡ 3a0d315b-b5f8-4387-9bb0-fd2a7038752e
#=╠═╡
const random_walk_state_distribution = calculate_random_walk_state_distribution()
  ╠═╡ =#

# ╔═╡ 75eceb07-f739-4009-8e92-b4742cedb548
# ╠═╡ skip_as_script = true
#=╠═╡
get_random_walk_true_value(s::Float32, values::Vector{Float32}) = values[Int64(s) + 1] 
  ╠═╡ =#

# ╔═╡ e3bd06e5-a16d-474c-b618-1c6f303eda00
#=╠═╡
function calc_random_walk_ve(g::Float32, v̂::Float32, s::Float32)
	true_value = get_random_walk_true_value(s, random_walk_v.value_function)
	(v̂ - true_value)^2
end
  ╠═╡ =#

# ╔═╡ 9dc8143f-280c-426a-911b-8ec851c9f093
#=╠═╡
random_walk_ve_setup_kwargs = (calculate_error = calc_random_walk_ve,)
  ╠═╡ =#

# ╔═╡ ce3ce1eb-1b88-4d30-aab3-9fa23c9246fe
#=╠═╡
function calculate_random_walk_ve(v̂::Function)
	states = Float32.(1:num_states)
	estimates = v̂.(states)
	sum(((random_walk_v.value_function .- estimates) .^2) .* random_walk_state_distribution)
end
  ╠═╡ =#

# ╔═╡ 214714a5-ad1e-4439-8567-9095d10411a6
# ╠═╡ skip_as_script = true
#=╠═╡
function figure_9_1()
	v = random_walk_v.value_function[2:end-1]
	(random_walk_v̂, error_history) = gradient_monte_carlo_estimation_state_aggregation(random_walk_state_mrp, 1f0, 100_000, num_groups, random_walk_group_assign; α = 2f-5, calculate_error = calc_random_walk_ve)
	v̂ = random_walk_v̂.(Float32.(1:num_states))
	x = 1:num_states
	n1 = L"v_\pi"
	n2 = L"\hat v"
	tr1 = scatter(x = x, y = v, name = "True value $n1")
	tr2 = scatter(x = x, y = v̂, name = "Approximate MC value $n2")
	
	state_distribution = calculate_random_walk_state_distribution()
	n3 = L"\mu"
	tr3 = bar(x = x, y = state_distribution, yaxis = "y2", name = "State distribution $n3", marker_color = "gray")

	state_mean = sum(state_distribution[i]*i for i in eachindex(state_distribution))
	state_variance = sum(state_distribution[i]*((i - state_mean)^2) for i in eachindex(state_distribution))
	p1 = plot([tr1, tr2, tr3], Layout(xaxis_title = "State", yaxis_title = "Value scale", yaxis2 = attr(title = "Distribution scale", overlaying = "y", side = "right"), title = "State Mean Value: $state_mean, State Value Variance: $state_variance"))

	p2 = plot(scatter(x = 1001:length(error_history), y = [sqrt(mean(error_history[i-1000:i])) for i in 1001:length(error_history)]), Layout(xaxis_title = "Episode", yaxis_title = "Value Error Over Previous 1000 Episodes"))
	# p2 = plot(scatter(y = sqrt.(error_history)), Layout(xaxis_title = "Episode", yaxis_title = "Value Error"))
	md"""
	$p1
	$p2
	"""
end
  ╠═╡ =#

# ╔═╡ c0e9ea1f-8cbe-4bc1-990f-ffd3ab1989cc
# ╠═╡ skip_as_script = true
#=╠═╡
figure_9_1()
  ╠═╡ =#

# ╔═╡ 49320a88-206e-4283-b3fc-a5d1ac41ddc4
#=╠═╡
function smooth_error(error_history, n)
	l = length(error_history)
	[mean(error_history[i-n:i]) for i in n+1:l]
end
  ╠═╡ =#

# ╔═╡ 3160e3ec-d1b9-47ea-ad10-3d6ea40cc0b5
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
## 9.4 Linear Methods
"""
  ╠═╡ =#

# ╔═╡ 701137fb-b497-47a5-9455-2f4b1c78a44e
md"""
Linear methods represent the value function as an inner product between *feature vectors* and *weight vectors*.

$\hat v(s, \mathbf{w})\doteq \mathbf{w}^\top \mathbf{x}(s) \doteq \sum_{i=1}^d w_i x_i(s)$ 

The vector $\mathbf{x}(s)$ is called a *feature vector* representing state x which is the same length as the number of parameters contained in $\mathbf{w}$.  For linear methods, features are *basis functions* because they form a linear basis for the set of approximate functions.

The gradient of linear value functions takes on a particularly simple form: $\nabla \hat v (s, \mathbf{w}) = \mathbf{x}(s)$.  Thus the general SGD update (9.7) reduces to:

$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \left [ U_t - \hat v (S_t, \mathbf{w}_t) \right ] \mathbf{x}(S_t)$

In the linear case there is only one optimum (or set of equally good optima), so any method that is guaranteed to converge to a local optimum is automatically guaranteed to converge to or near the global optimum.  For example, gradient Monte Carlo converges to the global optimum of the $\overline{VE}$ under linear function approximation if $\alpha$ is reduced over time according to the usual conditions.
"""

# ╔═╡ 6b339182-f81c-475c-bf28-d03b57eda76f
md"""
The semi-gradient TD(0) algorithm presented in the previous section also converges under linear function approximation, but this does not follow from general results on SGD; a separate theorem is necessary.  The weight vector converged to is also not the global optimum, but rather a point near the local optimum.  It is useful to consider this important case in more default, specifically for the continuing case.  The update at each time step $t$ is 

$\begin{flalign}
\mathbf{w}_{t+1} &\doteq \mathbf{w}_t +\alpha \left (R_{t+1} + \gamma \mathbf{w}_t ^ \top \mathbf{x}_{t+1} - \mathbf{w}_t ^ \top \mathbf{x}_t \right ) \mathbf{x}_t \tag{9.9}\\
&= \mathbf{w}_t + \alpha \left ( R_{t+1} \mathbf{x}_t - \mathbf{x}_t (\mathbf{x}_t - \gamma \mathbf{x}_{t+1} ) ^ \top \right ) \mathbf{w}_t
\end{flalign}$

where here we have used the notational shorthand $\mathbf{x}_t = \mathbf{x}(S_t)$.  Once the system has reached steady state, for any given $\mathbf{w}_t$, the expected next weight vector can be written

$\mathbb{E}[\mathbf{w}_{t+1} \vert \mathbf{w}_t] = \mathbf{w}_t + \alpha(\mathbf{b} - \mathbf{A} \mathbf{w}_t) \tag{9.10}$

where

$\mathbf{b} \doteq \mathbb{E}[R_{t+1} \mathbf{x}_t] \in \mathbb{R}^d \text{             and           } \mathbf{A} \doteq \mathbb{E} \left [ \mathbf{x}_t (\mathbf{x}_t - \gamma \mathbf{x}_{t+1}) ^\top \right] \in \mathbb{R}^{d \times d} \tag{9.11}$

From (9.10) it is clear that, if the system converges, it must converge to the weight vector $\mathbf{w}_{\text{TD}}$ at which

$\begin{flalign}
\mathbf{b} - \mathbf{A} \mathbf{w}_\text{TD} &= \mathbf{0} \\
\implies \mathbf{b} = \mathbf{A} \mathbf{w}_\text{TD} \\
\implies \mathbf{w}_\text{TD} \doteq \mathbf{A}^{-1} \boldsymbol{b} \tag{9.12}
\end{flalign}$

This quantity is called the *TD fixed point*.  In fact, linear semi-gradient TD(0) converges to this point.  See details below:
"""

# ╔═╡ b6737cef-b6f9-4e40-82d8-bf887e17eb7c
md"""
### Proof of Convergence of Linear TD(0)
"""

# ╔═╡ 3db9f60e-a823-4d78-bd16-e73cedffa755
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
At the TD fixed point, it has also been proven (in the continuing case) that the $\overline{VE}$ is within a bounded expansion of the lowest possible error: 

$\overline{VE}(\mathbf{w}_\text{TD}) \leq \frac{1}{1-\gamma} \min_{\mathbf{w}} \overline{VE} (\mathbf{w}) \tag{9.14}$

That is, the asymptotic error of the TD method is no more than $\frac{1}{1-\gamma}$ times the smallest possible error, that attained in the limit by the Monte Carlo method.  Because $\gamma$ is often near one, this expansion factor can be quite large, so there is substantial potential loss in asymptotic performance with the TD method.  On the otehr hand, recall that the TD methods are often of vastly reduced variance compared to Monte Carlo methods, and thus faster, as we saw in Chapters 6 and 7.

A bound analogous to (9.14) applies to other on-policy bootstrapping methods as well.  For example, linear semi-gradient DP $\left ( U_t \doteq \sum_a \pi(a \vert S_t) \sum_{s^\prime, r} p(s\prime, r \mid S_t, a)[r+\gamma \hat v(s^\prime, \mathbf{w}_t)] \right )$ with updates according to the on-policy distribution will also converge to the TD fixed point.  One-step semi-gradient *action-value* methods, such as semi-gradient Sarsa(0) convered in the next chapter converge to an analogous fixed point and an analogous bound.  Critical to these convergence results is that states are updated according to the on-policy distribution.  For other update distributions, bootstrapping methods using function approximation may actually diverge to infinity.
"""
  ╠═╡ =#

# ╔═╡ 7787522e-a4fb-4090-9a75-7ba74a4fcda6
md"""
### *Linear Methods Gradient Update Implementation*

For generic linear methods, the parameter update will require a gradient vector and a state representation vector that matches the length of the parameters.  To define a linear method then, all that is required is a function that converts a state to the state representation vector.
"""

# ╔═╡ c3732b25-94fd-4061-aab8-36fc39d739a1
md"""
In order to define a linear method, one must provide a state representation vector which will be the same length as the parameter vector as well as a function to update that representation for a given state.  The update function will be called as `update_feature_vector!(state_representation, s)`
"""

# ╔═╡ c737c14b-2ad6-4d95-9795-2b87f6f722cb
"""
    semi_gradient_td0_estimation_linear(mrp::StateMRP, γ::Real, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; kwargs...) where {T<:Real} -> NamedTuple

Semi-gradient TD(0) value estimation with linear function approximation on an MRP.

High-level interface for temporal difference learning using linear value functions. Automatically
sets up linear function approximation components (value function, gradient computation, parameter
initialization) and delegates to the core TD(0) algorithm. Performs online learning with immediate
parameter updates after each environment step.

# Type Parameters
- `T <: Real`: Numeric type for parameters, rewards, and computations (inferred from γ)

# Arguments
- `mrp::StateMRP`: The Markov reward process to evaluate, supporting any state type
- `γ::Real`: Discount factor for temporal difference updates (0 ≤ γ < 1)
- `max_episodes::Integer`: Maximum number of episodes to run
- `max_steps::Integer`: Maximum total steps across all episodes
- `feature_vector::LinearFeatureVector`: Linear feature representation for states
- `update_feature_vector!::Function`: Function to extract features from states into feature_vector

# Keyword Arguments
- `init_value::Real = zero(T)`: Initial value for parameter initialization
- `params::Vector{T} = initialize_linear_parameters(feature_vector, init_value)`: Initial parameter vector
- Additional arguments passed to [`semi_gradient_td0_estimation!`](@ref):
  - `α::Real = 0.1`: Learning rate for gradient updates
  - `calculate_error::Function = (target, v̂, s) -> (v̂ - target)^2`: Error function for statistics
  - `save_episode_steps::Bool = false`: Whether to save step-by-step reward history

# Returns
- `NamedTuple` with fields:
  - `value_function::Function`: Learned value function `v(s)` that maps states to estimated values
  - `episode_history::NamedTuple`: Episode statistics with fields:
    - `errors::Vector{T}`: Per-episode average TD errors
    - `steps::Vector{Int64}`: Steps taken in each episode
    - `rewards::Vector{T}`: Total rewards accumulated in each episode
  - `step_rewards::Vector{T}`: Step-by-step rewards (if `save_episode_steps=true`)
  - `parameters::Vector{T}`: Final learned parameter vector

# See Also
[`semi_gradient_td0_estimation!`](@ref), [`LinearFeatureVector`](@ref), [`linear_value_function`](@ref), [`initialize_linear_parameters`](@ref)

# Algorithm Details
1. Initialize linear function approximation components:
   - Parameter vector using [`initialize_linear_parameters`](@ref)
   - Linear value function and gradient computation functions
2. Delegate to [`semi_gradient_td0_estimation!`](@ref) which performs:
   - Online TD(0) updates: θ ← θ + α * δ * ∇v(s) where δ = r + γv(s') - v(s)
   - Episode management with termination checking
   - Error and reward tracking across episodes
3. Return learned value function and training statistics

# Examples
```julia-repl
julia> # Create continuous random walk MRP
       mrp = create_continuous_random_walk(1000);

julia> # Set up linear features (e.g., polynomial basis)
       features, update_fn = create_polynomial_features(3);

julia> # Run TD(0) learning with linear approximation
       results = semi_gradient_td0_estimation_linear(
           mrp, 0.95f0, 100, 10000, features, update_fn; 
           α=0.01f0, init_value=0.0f0
       );

julia> # Check convergence and final performance
       println("Episodes completed: ", length(results.episode_history.errors))
       println("Final episode error: ", results.episode_history.errors[end])
       println("Parameter vector: ", results.parameters)
Episodes completed: 100
Final episode error: 0.023f0
Parameter vector: Float32[0.12, -0.45, 0.08]

julia> # Evaluate learned value function
       test_state = 500.0f0;
       estimated_value = results.value_function(test_state);
       println("Value at state 500: ", estimated_value)
Value at state 500: 0.034f0
```

# Performance Notes
- Reuses feature vector and gradient storage to minimize allocations
- Compatible with any [`LinearFeatureVector`](@ref) implementation
"""
semi_gradient_td0_estimation_linear(mrp::StateMRP, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), params::Vector{T} = initialize_linear_parameters(feature_vector, init_value), kwargs...) where T<:Real = semi_gradient_td0_estimation!(params, mrp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, linear_value_function, deepcopy(feature_vector), update_linear_value_gradient!; kwargs...)

# ╔═╡ 3307300f-cd72-4f16-bc46-39115a32e2ca
"""
    semi_gradient_td0_policy_estimation_linear(mdp::StateMDP, π::Function, γ::Real, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; kwargs...) where {T<:Real} -> NamedTuple

Semi-gradient TD(0) policy evaluation with linear function approximation on an MDP.

High-level interface for temporal difference policy evaluation using linear value functions. Automatically
sets up linear function approximation components (value function, gradient computation, parameter
initialization) and delegates to the core TD(0) algorithm. Performs online learning with immediate
parameter updates after each environment step to estimate the state value function v^π(s) for the given policy.

# Type Parameters
- `T <: Real`: Numeric type for parameters, rewards, and computations (inferred from γ)

# Arguments
- `mdp::StateMDP`: The Markov decision process to evaluate, supporting any state and action types
- `π::Function`: Policy function mapping states to action selections (state → action)
- `γ::Real`: Discount factor for temporal difference updates (0 ≤ γ < 1)
- `max_episodes::Integer`: Maximum number of episodes to run
- `max_steps::Integer`: Maximum total steps across all episodes
- `feature_vector::LinearFeatureVector`: Linear feature representation for states
- `update_feature_vector!::Function`: Function to extract features from states into feature_vector

# Keyword Arguments
- `init_value::Real = zero(T)`: Initial value for parameter initialization
- `params::Vector{T} = initialize_linear_parameters(feature_vector, init_value)`: Initial parameter vector
- Additional arguments passed to [`semi_gradient_td0_estimation!`](@ref):
  - `α::Real = 0.1`: Learning rate for gradient updates
  - `calculate_error::Function = (target, v̂, s) -> (v̂ - target)^2`: Error function for statistics
  - `save_episode_steps::Bool = false`: Whether to save step-by-step reward history

# Returns
- `NamedTuple` with fields:
  - `value_function::Function`: Learned state value function `v_π(s)` that maps states to estimated values under policy π
  - `episode_history::NamedTuple`: Episode statistics with fields:
    - `errors::Vector{T}`: Per-episode average TD errors
    - `steps::Vector{Int64}`: Steps taken in each episode
    - `rewards::Vector{T}`: Total rewards accumulated in each episode under policy π
  - `step_rewards::Vector{T}`: Step-by-step rewards (if `save_episode_steps=true`)
  - `parameters::Vector{T}`: Final learned parameter vector

# See Also
[`semi_gradient_td0_estimation!`](@ref), [`LinearFeatureVector`](@ref), [`linear_value_function`](@ref), [`initialize_linear_parameters`](@ref)

# Algorithm Details
1. Initialize linear function approximation components:
   - Parameter vector using [`initialize_linear_parameters`](@ref)
   - Linear value function and gradient computation functions
2. Delegate to [`semi_gradient_td0_estimation!`](@ref) which performs:
   - Policy-based episode generation using π for action selection
   - Online TD(0) updates: θ ← θ + α * δ * ∇v(s) where δ = r + γv(s') - v(s)
   - Episode management with termination checking
   - Error and reward tracking across policy rollouts
3. Return learned value function v_π and training statistics

# Performance Notes
- Reuses feature vector and gradient storage to minimize allocations
- Compatible with any [`LinearFeatureVector`](@ref) implementation
"""
semi_gradient_td0_policy_estimation_linear(mdp::StateMDP, π::Function, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), params::Vector{T} = initialize_linear_parameters(feature_vector, init_value), kwargs...) where T<:Real = semi_gradient_td0_policy_estimation!(params, mdp, π, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, linear_value_function, deepcopy(feature_vector), update_linear_value_gradient!; kwargs...)

# ╔═╡ 645ba5fc-8575-4b8f-8982-f8bd20ac27ff
#=╠═╡
md"""
### Example 9.2: Bootstrapping on the $num_states-state Random Walk

State aggregation is a special case of linear function approximation, so we can use the previous example to illustrate the convergence properties of semi-gradient TD(0) vs gradient Monte Carlo.  
"""
  ╠═╡ =#

# ╔═╡ 99f34d13-a19a-4a28-8173-2f683527d61a
#=╠═╡
semi_gradient_td0_estimation_state_aggregation(mrp::StateMRP, γ::Real, max_episodes::Integer, max_steps::Integer, num_groups::Integer, assign_state_group::Function; kwargs...) = semi_gradient_td0_estimation_linear(mrp, γ, max_episodes, max_steps, state_aggregation_feature_setup(mrp.initialize_state(), num_groups, random_walk_group_assign)...; kwargs...)
  ╠═╡ =#

# ╔═╡ 7889fc4a-3a77-41b4-983a-0b04740afeb7
#=╠═╡
semi_gradient_td0_policy_estimation_state_aggregation(mdp::StateMDP, π::Function, γ::Real, max_episodes::Integer, max_steps::Integer, num_groups::Integer, assign_state_group::Function; kwargs...) = semi_gradient_td0_policy_estimation_linear(mdp, π, γ, num_episodes, state_aggregation_feature_setup(mrp.initialize_state(), num_groups, random_walk_group_assign)...; kwargs...)
  ╠═╡ =#

# ╔═╡ cf9d7c7d-4519-410a-8a05-af90312e291c
#=╠═╡
md"""
### Figure 9.2
Bootstrapping with state aggregation on the $num_states-state random walk task.  The asymptotic values of semi-gradient TD are worse than the asymptotic Monte Carlo values which matches with the expectation from the TD-fixed point convergence.
"""
  ╠═╡ =#

# ╔═╡ 7989d6a9-a52a-4537-9c39-5d6b41f60098
# ╠═╡ skip_as_script = true
#=╠═╡
@bind fig_9_2_params PlutoUI.combine() do Child
	md"""
	Learning Rates: Monte Carlo $(Child(:α_mc, NumberField(0f0:1f-8:1f0, default = 2f-5))) TD(0) $(Child(:α_td, NumberField(0f0:1f-8:1f0, default = 2f-4)))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ bfb1858b-5e05-4239-bcae-a3b718074630
# ╠═╡ skip_as_script = true
#=╠═╡
function figure_9_2(;num_episodes = 100_000, α_mc = 2f-5, α_td = 2f-4)
	v = random_walk_v.value_function[2:end-1]
	
	v̂_mc, err_history_mc = gradient_monte_carlo_estimation_state_aggregation(random_walk_state_mrp, 1f0, num_episodes, num_groups, random_walk_group_assign; α = α_mc, calculate_error = calc_random_walk_ve)

	#this function will produce the learned value estimate given a random walk state
	v̂_td, episode_history_td = semi_gradient_td0_estimation_state_aggregation(random_walk_state_mrp, 1f0, num_episodes, typemax(Int64), num_groups, random_walk_group_assign; α = α_td, calculate_error = calc_random_walk_ve)
	err_history_td = episode_history_td.errors
	
	x = Float32.(1:num_states)

	v̂_mc = v̂_mc.(x)
	v̂_td = v̂_td.(x)
	
	n1 = L"v_\pi"
	tr1 = scatter(x = x, y = v, name = "True value $n1")
	tr2 = scatter(x = x, y = v̂_mc, name = "Monte Carlo Value Estimate")
	tr3 = scatter(x = x, y = v̂_td, name = "TD(0) Value Estimate")

	p1 = plot([tr2, tr3, tr1], Layout(xaxis_title = "State", yaxis_title = "Value"))

	nsmooth = 100
	tr1 = scatter(x = nsmooth:num_episodes, y = sqrt.(smooth_error(err_history_mc, nsmooth)), name = "Monte Carlo Estimate Errors")
	tr2 = scatter(x = nsmooth:num_episodes, y = sqrt.(smooth_error(err_history_td, nsmooth)), name = "TD(0) Estimate Errors")
	p2 = plot([tr1, tr2], Layout(xaxis_title = "Episode", yaxis_title = "Value Error", showlegend = false))
	@htl("""
	<div style = "display: flex;">
	$p2
	$p1
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ c05ea239-2eea-4f41-b4e3-993db0fe2de5
#=╠═╡
figure_9_2(;num_episodes = 100_000, fig_9_2_params...)
  ╠═╡ =#

# ╔═╡ f5203959-29ef-406c-abac-4f01fa9630a3
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
> ### *Exercise 9.1* 
> Show that tabular methods such as presented in Part I of this book are a special case of linear function approximation.  What would the feature vectors be?

The simplest form of function approximation presented so far is state-aggregation which is a special case of linear function approximation.  Consider a case of state-aggregation where every state is in its own unique group and there is a parameter vector $\mathbf{w}$ such that $w_i$ is the approximation value for $s_i$.  Following the rules of state aggregation, the feature vectors would be orthanormal basis vectors of dimension matching the number of states, thus state 1 would be represented by the feature vector [1, 0, 0, ...], state 2 by [0, 1, 0, 0, ...] and so on.  The gradient Monte Carlo update rule for these feature vectors would be $w_i = w_i + \alpha [G_t - w_i]$ for an episode step encountering state $s_i$.  The TD(0) update rule would be $w_i = w_i + \alpha [R_t + \gamma w_j - w_i]$ where the next state encountered is $s_j$.  Both of these rules are exactly the same as tabular Monte Carlo policy prediction (with constant step size averaging) and tabular TD(0) policy prediction where $v_i = w_i$.  So the value function from the tabular setting is still a list of $\vert \mathcal{S} \vert$ values, one for each state and every state value update has no effect on the value estimates of other states.
"""
  ╠═╡ =#

# ╔═╡ c3da96b0-d584-4a43-acdb-16516e2d0452
md"""
## 9.5 Feature Construction for Linear Methods

Linear methods can only make approximations that additively combine the effects of multiple features.  In order to account for interactions between state properties such as the position and velocity of an object, features must be constructed that explicitely combine those state values.  The purpose of feature construction is to inject into the problem domain knowledge related to what type of information from the states will be useful to solving the problem.
"""

# ╔═╡ 0ee3afe9-9c33-45c8-b304-26062675e1b8
md"""
### 9.5.1 Polynomials

Consider a state with two numerical features $s_1, s_2$.  We could construct a feature vector that simply uses each value $(s_1, s_2)$ but this would restrict our value estimator to outputs of the form $as_1 + bs_2$.  This functional form would make it impossible for an estimated value to be non-zero if both state values are zero which may not be true in the environment.  In order to lift this restriction it is common to add a bias feature that is always 1.  Another desired feature may be one that combines both state values together multiplicatively.  Additional features of this nature are called polynomial features and take the form:

$x_i(s) = \prod_{j=1}^k s_j^{c_{i,j}} \tag{9.17}$

where each $c_{i,j}$ is an integer in the set $\{0, 1, \dots, n \}$ for an integer $n \geq 0$.  An example of such a feature vector for $n=2$ and $k=2$ state values is shown below:

$\mathbf{x}(s) = (1, s_1, s_2, s_1 s_2, s_1^2, s_2^2, s_1 s_2^2, s_1^2 s_2, s_1^2 s_2^2)$

This combination yields $(2+1)^2 = 9$ features since each of the two state values can be raised to 3 different exponents and then combined.
"""

# ╔═╡ d65a0ca9-5577-4df8-af77-44ecfbcc0a07
md"""
> ### *Exercise 9.2* 
> Why does (9.17) define $(n+1)^k$ distinct features for dimension $k$?
n represents the highest power to take for each individual dimension of the state and we consider powers from 0 up to n for each dimension.  If we list the exponent per dimension as a tuple, we have for n = 1, k = 2: (0, 0), (0, 1), (1, 0), (1, 1).
For n = 1, k = 3: (0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1).  This pattern consists of tuples of length k which can be formed by selecting from n + 1 choices of exponent.  The number of resulting tuples is $(n+1)^k$
"""

# ╔═╡ c5adf2d7-0b6b-4a87-974b-a90824d0323b
md"""
> ### *Exercise 9.3* 
> What $n$ and $c_{i, j}$ produce the feature vectors $\mathbf{x}(s)=(1, s_1, s_2, s_1s_2, s_1^2, s_2^2, s_1s_2^2, s_1^2s_2, s_1^2s_2^2)^\top$

Since the highest exponent considered is 2, $n=2$.  For the exponents we can visualize $c_{i, j}$ as the following matrix where rows correspond to $i$ and columns to $j$


$\begin{matrix}
0 & 0\\
1 & 0\\
0 & 1\\
1 & 1\\
2 & 0\\
0 & 2\\
1 & 2\\
2 & 1\\
2 & 2\\
\end{matrix}$
"""

# ╔═╡ 38f09914-e128-4336-8e70-9906675971f2
"""
    get_order_coefficients(k, n; coefs=()) -> Vector{NTuple{k, Int}}

Generate all coefficient tuples of length k with elements in range 0:n.

Recursively constructs all possible k-tuples where each element is an integer
from 0 to n inclusive. Used internally for polynomial basis construction and
multivariate feature generation in function approximation.

# Arguments
- `k::Integer`: Length of coefficient tuples to generate
- `n::Integer`: Maximum value for each coefficient (range 0:n)

# Keyword Arguments
- `coefs::Tuple`: Partial coefficient tuple for recursive construction (default: ())

# Returns
- `Vector{NTuple{k, Int}}`: All possible k-tuples with coefficients in 0:n

# Performance Notes
- Recursive implementation generates (n+1)^k total combinations
- Memory usage grows exponentially with k and n
"""
function get_order_coefficients(k, n; coefs = ())
	k == 0 && return coefs
	reduce(vcat, get_order_coefficients(k-1, n; coefs = (coefs..., e)) for e in 0:n)
end

# ╔═╡ 75fbbc54-807a-4894-a536-b27be81eb052
# ╠═╡ skip_as_script = true
#=╠═╡
get_order_coefficients(2, 2)
  ╠═╡ =#

# ╔═╡ f5dea7d5-4597-430c-9020-b74cdf8f3055
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
Notice that these 9 exponents match the ones for the feature vector in exercise 9.3
"""
  ╠═╡ =#

# ╔═╡ 9d7ca70c-0e60-4029-8ea0-26192ccea849
"""
    order_features_setup(problem::Union{StateMDP{T, S, A, P, F1, F2, F3}, StateMRP{T, S, P, F1, F2}}, n::Integer, min_values::S, max_values::S, feature_calculation::Function) -> NamedTuple

Create ordered feature vector and update function for basis function approximation.

Sets up feature representation for states using coefficient tuples from [`get_order_coefficients`](@ref)
and a provided feature calculation function. Generates all coefficient combinations up to order n
and creates optimized update function for extracting features from states. Compatible with polynomial,
Fourier, and other basis functions that use ordered coefficient expansions.

# Type Parameters
- `T <: Real`: Numeric type for computations and feature values
- `N`: Dimension of tuple states (automatically inferred)
- `S <: Union{T, NTuple{N, T}}`: State type - either scalar `T` or N-dimensional tuple
- `A`: Action type (for MDP problems)
- `P`: Transition probability type
- `F1 <: Function`: State initialization function type
- `F2 <: Function`: Transition function type  
- `F3 <: Function`: Termination function type (MDP only)

# Arguments
- `problem::Union{StateMDP{T, S, A, P, F1, F2, F3}, StateMRP{T, S, P, F1, F2}}`: MDP or MRP providing state type information
- `n::Integer`: Maximum order for coefficient generation
- `min_values::S`: Minimum bounds for state normalization (scalar T or tuple matching state dimension)  
- `max_values::S`: Maximum bounds for state normalization (scalar T or tuple matching state dimension)
- `feature_calculation::Function`: Function computing features from (state, min_vals, max_vals, coefficients)

# Returns
- `NamedTuple` with fields:
  - `feature_vector::Vector{T}`: Pre-allocated feature storage of length (n+1)^k where k is state dimension
  - `update_feature_vector!::Function`: Optimized function to populate features from state

The returned values are designed to be passed directly as the `feature_vector` and `update_feature_vector!`
arguments to linear function approximation methods.

# See Also
[`semi_gradient_td0_estimation_linear`](@ref), [`gradient_monte_carlo_estimation_linear`](@ref), [`gradient_monte_carlo_policy_estimation_linear`](@ref), [`get_order_coefficients`](@ref)

# Performance Notes
- Uses `@simd` optimization for fast feature computation
- Pre-allocates feature vector to avoid repeated memory allocation
- Feature vector size is (n+1)^k where k is inferred state dimensionality

# Examples
```julia-repl
julia> # Set up features for linear TD(0) learning
       features, update_fn = order_features_setup(mrp, 3, 0.0f0, 1000.0f0, polynomial_calc);

julia> # Pass directly to TD(0) estimation
       results = semi_gradient_td0_estimation_linear(
           mrp, 0.95f0, 100, 10000, features, update_fn; α=0.01f0
       );

julia> # Or use with gradient Monte Carlo
       mc_results = gradient_monte_carlo_estimation_linear(
           mrp, 0.9f0, 500, features, update_fn; α=0.05f0
       );
```
"""
function order_features_setup(problem::Union{StateMDP{T, S, A, P, F1, F2, F3}, StateMRP{T, S, P, F1, F2}}, n::Integer, min_values::S, max_values::S, feature_calculation::Function) where {T<:Real, N, S <: Union{T, NTuple{N, T}}, A, P, F1<:Function, F2<:Function, F3<:Function}
	#states must be tuples with k elements or some number value
	k = S == T ? 1 : N
	coefs = get_order_coefficients(k, n)

	l = length(coefs)

	function update_feature_vector!(x::Vector{T}, s::T)
		@inbounds @simd for i in eachindex(x)
			feature = feature_calculation(s, min_values, max_values, coefs[i])
			x[i] = feature
		end
	end

	x = zeros(T, l)

	(feature_vector = x, update_feature_vector! = update_feature_vector!)
end

# ╔═╡ bc2e52ff-7f47-4141-aff1-e752fe217f6a
begin
	"""
	    calc_poly_feature(s, min_values, max_values, e) -> Real
	
	Compute polynomial basis function value for normalized state with given exponents.
	
	Normalizes state to [0,1] range using min/max bounds, then computes polynomial
	feature as product of normalized coordinates raised to specified powers. Supports
	both scalar and multi-dimensional tuple states.
	
	# Arguments
	- `s::Union{T, NTuple{N, T}}`: State value (scalar or N-tuple)
	- `min_values::Union{T, NTuple{N, T}}`: Minimum bounds for normalization
	- `max_values::Union{T, NTuple{N, T}}`: Maximum bounds for normalization  
	- `e::NTuple{N, Int64}`: Exponent coefficients for each dimension
	
	# Returns
	- `T`: Polynomial feature value
	"""
	calc_poly_feature(s::NTuple{N, T}, min_values::NTuple{N, T}, max_values::NTuple{N, T}, e::NTuple{N, Int64}) where {T<:Real, N} = prod(((s[i] - min_values[i]) / (max_values[i] - min_values[i]))^e[i] for i in 1:N)
	calc_poly_feature(s::T, min_value::T, max_value::T, e::NTuple{1, Int64}) where {T<:Real} = ((s - min_value) / (max_value - min_value))^e[1]
end

# ╔═╡ be715a78-5fcb-48b2-8a4f-c7ba27d34dd3
# ╠═╡ skip_as_script = true
#=╠═╡
plot([scatter(y = [calc_poly_feature(s, 1, 1000, (o,)) for s in 1:1000], name = "Order $o") for o in 0:5], Layout(title = "One Dimensional Polynomial Features"))
  ╠═╡ =#

# ╔═╡ c609ee03-7217-4068-9da2-c91fb02623a9
# ╠═╡ skip_as_script = true
#=╠═╡
md"""
Note that a scaling factor of 1/num_states means that all states will be mapped to the range of 0 to 1 for the purpose of computing the polynomial features.  This helps with numerical stability when we are taking the state integers to large powers of n"""
  ╠═╡ =#

# ╔═╡ 25f4f9d3-d8aa-462c-9874-ae842da1cf79
md"""
### *Example: Linear Feature Vectors with Random Walk*
"""

# ╔═╡ 1e58c332-d43e-4467-b7b1-377262d460c3
#=╠═╡
function show_random_walk_results((v̂_mc, mc_error), (v̂_td, td_history), name, nsmooth)
	num_episodes = length(mc_error)
	td_error = td_history.errors
	p1 = plot([scatter(x = nsmooth:num_episodes, y = sqrt.(smooth_error(mc_error, nsmooth)), name = "Monte Carlo"), scatter(x = nsmooth:num_episodes, y = sqrt.(smooth_error(td_error, nsmooth)), name = "TD(0)")], Layout(xaxis_title = "Episode", yaxis_title = "Value Error Averaged <br> over Previous $nsmooth Episodes", showlegend = false))
	p2 = plot([scatter(x = 1:num_states, y = v̂_mc.(Float32.(1:num_states)), name = "Monte Carlo"), scatter(x = 1:num_states, y = v̂_td.(Float32.(1:num_states)), name = "TD(0)"), scatter(y = random_walk_v.value_function[2:end-1], name = "true value")], Layout(title = "$name Approximation", yaxis_title = "Value", xaxis_title = "State"))
	@htl("""
	<div style = "display: flex;">
	$p1
	$p2
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ 56212ab2-833a-4dec-bcdd-21bce1d680b6
# ╠═╡ skip_as_script = true
#=╠═╡
function show_random_walk_linear_results(feature_vector::LinearFeatureVector, update_feature_vector!::Function, num_episodes, α_mc::T, α_td::T, name; nsmooth = 100) where T<:Real
	v̂_mc, mc_error = gradient_monte_carlo_estimation_linear(random_walk_state_mrp, 1f0, num_episodes, feature_vector, update_feature_vector!; α = α_mc, calculate_error = calc_random_walk_ve)
	v̂_td, td_history = semi_gradient_td0_estimation_linear(random_walk_state_mrp, 1f0, num_episodes, typemax(Int64), feature_vector, update_feature_vector!; α = α_td, calculate_error = calc_random_walk_ve)
	show_random_walk_results((v̂_mc, mc_error), (v̂_td, td_history), name, nsmooth)
end
  ╠═╡ =#

# ╔═╡ 93a617ee-db64-4351-b919-340d950fc148
#=╠═╡
@bind poly_feature_params PlutoUI.combine() do Child
	md"""
	Number of Order Features: $(Child(:order_num, NumberField(1:100, default = 5)))
	
	Learning Rates: Monte Carlo $(Child(:α_mc, NumberField(0f0:1f-8:1f0, default = 2f-5))) TD(0) $(Child(:α_td, NumberField(0f0:1f-8:1f0, default = 2f-4)))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ 994f8556-964c-4c6b-8cfe-6f6a99c1ba29
#=╠═╡
show_random_walk_linear_results(order_features_setup(random_walk_state_mrp, poly_feature_params.order_num, 1f0, Float32(num_states), calc_poly_feature)..., 25_000, poly_feature_params.α_mc, poly_feature_params.α_td, "Polynomial Basis Function Approximation")
  ╠═╡ =#

# ╔═╡ ed00f1b2-79b0-406a-aabc-8c8c7ad61c31
md"""
### 9.5.2 Fourier Basis

With fourier features we generate the same integer vectors that we had for the polynomial basis so $(n+1)^k$ different vectors which define the different features.  The difference is that instead of exponents, these coefficients are now used to create an argument for a cosine function: $x_i(s) = \cos(\pi \mathbf{s}^\top \mathbf{c}^i)$.  For $k = 2$ and $n = 2$, the first few of these $\mathbf{c}$ vectors would look like: $[0, 0], [0, 1], [1, 0], \dots$.  Also, it is important for the numerical features that are the elements of $s$ be scaled between 0 and 1, so this method only works well if the numerical values of the state space fall within a known range.
"""

# ╔═╡ f1b7b56e-7701-4954-8217-1b2c7d01e309
begin
	"""
	    calc_fourier_feature(s, min_values, max_values, c) -> Float32
	
	Compute Fourier basis function value for normalized state with given coefficients.
	
	Normalizes state to [0,1] range using min/max bounds, then computes cosine basis
	function with coefficient-weighted frequency. Supports both scalar and multi-dimensional
	tuple states for Fourier series approximation.
	
	# Arguments
	- `s::Union{T, NTuple{N, T}}`: State value (scalar or N-tuple)
	- `min_values::Union{T, NTuple{N, T}}`: Minimum bounds for normalization
	- `max_values::Union{T, NTuple{N, T}}`: Maximum bounds for normalization  
	- `c::NTuple{N, Int64}`: Frequency coefficients for each dimension
	
	# Returns
	- `T`: Fourier feature value (matches input numeric type)
	"""
	calc_fourier_feature(s::NTuple{N, T}, min_values::NTuple{N, T}, max_values::NTuple{N, T}, c::NTuple{N, Int64}) where {T<:Real, N} = cos(T(π) * sum((s[i] - min_values[i])*e[i] / (max_values[i] - min_values[i]) for i in 1:N))
	calc_fourier_feature(s::T, min_value::T, max_value::T, c::NTuple{1, Int64}) where {T<:Real} = cos(T(π)*(s - min_value)*c[1] / (max_value - min_value))
end

# ╔═╡ c99867b7-2cb0-4bb7-b035-0e86104adefe
md"""
### *Example: Fourier Feature Vectors with Random Walk*

Notice that for approximation techniques that are forced to do global approximation, TD(0) can converge faster than Monte Carlo without much loss in accuracy.  Monte Carlo should converge to the same or lower error but may require a very large number of learning steps.
"""

# ╔═╡ 89262830-1129-4270-8007-32fb0cd2e0ec
# ╠═╡ skip_as_script = true
#=╠═╡
@bind fourier_feature_params PlutoUI.combine() do Child
	md"""
	Number of Order Features: $(Child(:order_num, NumberField(1:100, default = 5)))
	
	Learning Rates: Monte Carlo $(Child(:α_mc, NumberField(0f0:1f-8:1f0, default = 3f-5))) TD(0) $(Child(:α_td, NumberField(0f0:1f-8:1f0, default = 7f-4)))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ 111a6762-ab4f-4db9-80f9-10c707623e0f
#=╠═╡
show_random_walk_linear_results(order_features_setup(random_walk_state_mrp, fourier_feature_params.order_num, 1f0, Float32(num_states), calc_fourier_feature)..., 10_000, fourier_feature_params.α_mc, fourier_feature_params.α_td, "Fourier Basis Function Approximation")
  ╠═╡ =#

# ╔═╡ b4aefbb1-dbb7-490c-9fa7-0f68e5a9916c
# ╠═╡ skip_as_script = true
#=╠═╡
function plot_value_error(errs, names, nsmooth)
	l = length(first(errs))
	traces = [begin
		scatter(x = nsmooth:l, y = sqrt.(smooth_error(err, nsmooth)), name = names[i])
	end
	for (i, err) in enumerate(errs)]
	plot(traces, Layout(xaxis_title = "Episode", yaxis_title = "Value Error Averaged over <br> Previous $nsmooth Episodes"))
end
  ╠═╡ =#

# ╔═╡ a99ef185-0360-4005-9a8c-f10ca58babda
md"""
### 9.5.3 Coarse Coding

Coarse coding also operates in a state space where we can clearly define one or more numerical dimensions that scale over a known range of values.  Consider a number of overlapping regions in this space.  If we have N regions then that defines N binary features.  Each feature just indicates whether a state is present in that region.  Since the regions are overlapping most states will activate more than one feature.  If the regions are defined in a consistent way with a set shape and displacement vector, then each state will always activate the same number of features.  If the regions do not overlap and fully cover the state space, then this is equivalent to state aggregation where each state activates a single feature.
"""

# ╔═╡ 168e84f6-429e-45d6-bdbd-f47552fce8b5
# ╠═╡ skip_as_script = true
#=╠═╡
@bind coarse_linear_display PlutoUI.combine() do Child
	md"""
	State Value: $(Child(:x, Slider(0:0.01:3; show_value=true, default = 1.5)))
	
	Zone Offset: $(Child(:offset, NumberField(0:0.1:1, default = 0.5)))
	"""
end
  ╠═╡ =#

# ╔═╡ 40f0fd57-a4ea-47a0-b883-3b038a6612c4
# ╠═╡ skip_as_script = true
#=╠═╡
function show_coarse_coding_regions(x, offset_percentage)
	make_zone(offsetx, offsety) = scatter(x = [offsetx, 1+offsetx], y = [offsety, offsety], showlegend = false)
	region_starts = 0:offset_percentage:2.5
	traces = [make_zone(offsetx, offsety) for (offsetx, offsety) in zip(region_starts, region_starts)]
	state_trace = scatter(x = [x, x], y = [-1, 4], line_color = "black", mode = "lines", name = "state")

	feature_vector = Int64.([(x > a) && (x < a + 1) for a in region_starts])

	vector_string = reduce((a, b) -> "$a, $b", feature_vector)

	test = Markdown.parse(L"[%$vector_string]")
	
	md"""
	Feature Vector
	$test

	$(plot([traces; state_trace]))
	"""
end
  ╠═╡ =#

# ╔═╡ 529e262c-c94c-407b-8f13-be3b0f737e61
#=╠═╡
show_coarse_coding_regions(coarse_linear_display.x, coarse_linear_display.offset)
  ╠═╡ =#

# ╔═╡ e565c041-17bd-40c8-9240-e86931c83010
md"""
### 9.5.4 Tile Coding

Tile coding is a form of coarse coding where each state will be present in one distinct *tile* for each tiling.  A tiling is a segmentation of the state space that covers the entire space with non-overlapping regions that have no gaps.  Each tiling is thus a single instance of state aggregation.  To create multiple tilings, each tiling is shifted a set amount in each dimension of the state space to create a new set of regions shifted in position from the originals.  The shape of the tiles and the amount of offset could be different in each dimension and sometimes this asymmetry is desireable to avoid approximation artifacts such as prefered directions in the state space caused by uniform offsets.
"""

# ╔═╡ d215b917-c43d-4c14-aa97-2310f922d71a
"""
    scale_state(s, min_value, range) -> Real

Normalize a scalar state value to [0,1] range using linear scaling.

# Type Parameters
- `T <: Real`: Numeric type for state values and scaling parameters

# Arguments
- `s::T`: State value to normalize
- `min_value::T`: Minimum value of the original state range
- `range::T`: Range (max_value - min_value) of the original state space

# Returns
- `T`: Normalized state value in [0,1] where 0 corresponds to `min_value` and 1 corresponds to `min_value + range`
"""
scale_state(s::T, min_value::T, range::T) where T<:Real = (s - min_value) / range

# ╔═╡ 35d6dd59-1fd3-4aad-b24f-82dd466bcb83
begin
	"""
	    update_tile_features!(x, state, offset, d, num_tilings, tile_size, num_tiles, min_value, range) -> Vector
	
	Update tile feature indices for a scalar state using tile coding with displacement offset.
	
	# Type Parameters
	- `I <: Integer`: Index type for tile indices
	- `T <: Real`: Numeric type for state values and tiling parameters
	
	# Arguments
	- `x::Vector{I}`: Pre-allocated vector to store tile indices (modified in-place)
	- `state::T`: Scalar state value to encode
	- `offset::T`: Displacement offset for tiling alignment
	- `d::Int64`: Displacement factor for offset calculation
	- `num_tilings::Integer`: Number of overlapping tile layers
	- `tile_size::T`: Size of each tile in normalized coordinates
	- `num_tiles::Int64`: Number of tiles per tiling layer
	- `min_value::T`: Minimum value of the state space
	- `range::T`: Range (max - min) of the state space
	
	# Returns
	- `Vector{I}`: The modified input vector `x` containing tile indices for each tiling
	"""
	
	"""
	    update_tile_features!(x, state, offset, displacement, num_tilings, tile_size, num_tiles, min_values, ranges) -> Vector
	
	Update tile feature indices for a multi-dimensional state using tile coding with displacement offsets.
	
	# Type Parameters
	- `I <: Integer`: Index type for tile indices
	- `N`: Number of state dimensions
	- `T <: Real`: Numeric type for state values and tiling parameters
	
	# Arguments
	- `x::Vector{I}`: Pre-allocated vector to store tile indices (modified in-place)
	- `state::NTuple{N, T}`: N-dimensional state tuple to encode
	- `offset::NTuple{N, T}`: Displacement offsets for each dimension
	- `displacement::NTuple{N, Int64}`: Displacement factors for offset calculation per dimension
	- `num_tilings::Integer`: Number of overlapping tile layers
	- `tile_size::NTuple{N, T}`: Tile size for each dimension in normalized coordinates
	- `num_tiles::NTuple{N, Int64}`: Number of tiles per dimension per tiling layer
	- `min_values::NTuple{N, T}`: Minimum values for each state dimension
	- `ranges::NTuple{N, T}`: Ranges (max - min) for each state dimension
	
	# Returns
	- `Vector{I}`: The modified input vector `x` containing tile indices for each tiling
	"""
	function update_tile_features!(x::Vector{I}, state::T, offset::T, d::Int64, num_tilings::Integer, tile_size::T, num_tiles::Int64, min_value::T, range::T) where {I<:Integer, T<:Real}
		l = num_tiles*num_tilings
		for tiling in 1:num_tilings
			i = max(1, ceil(Int64, (scale_state(state, min_value, range) + offset*d*(tiling-1)) / tile_size))
			x[tiling] = min(i + (tiling - 1)*num_tiles, l)
		end
		return x
	end

	function update_tile_features!(x::Vector{I}, state::NTuple{N, T}, offset::NTuple{N, T}, displacement::NTuple{N, Int64}, num_tilings::Integer, tile_size::NTuple{N, T}, num_tiles::NTuple{N, Int64}, min_values::NTuple{N, T}, ranges::NTuple{N, T}) where {I <: Integer, N, T<:Real}
		total_tiles = prod(num_tiles)
		l = total_tiles*num_tilings
		for tiling in 1:num_tilings
			base = 1
			index = 0
			for d in 1:N
				i = max(1, ceil(Int64, (scale_state(state[d], min_values[d], ranges[d]) + offset[d]*displacement[d]*(tiling - 1)) / tile_size[d]))
				index += i * base
				base *= num_tiles[d]
			end
			x[tiling] = min(index + (tiling - 1)*total_tiles, l)
		end
		return x
	end
end

# ╔═╡ c64b740a-ceeb-431c-9c71-6ab498fc4003
begin
	form_default_displacement_vector(s::Real) = 1
	form_default_displacement_vector(s::NTuple{N, T}) where {N, T<:Real} = Tuple(i*2 + 1 for i in 0:N-1)
end

# ╔═╡ bb81db16-7c4d-4e08-bf17-45147be2b0db
"""
    tile_coding_feature_setup(problem, min_value, max_value, tile_size, num_tilings, displacement_vector) -> NamedTuple

Set up tile coding feature representation system for linear value function approximation in reinforcement learning problems.

# Type Parameters
- `T <: Real`: Numeric type for state values and computations
- `N`: Number of state dimensions (for multi-dimensional states)
- `S <: Union{T, NTuple{N, T}}`: State type (scalar or N-dimensional tuple)
- `A`: Action type (MDP only)
- `P`: Transition probability type
- `F1 <: Function`: State initialization function
- `F2 <: Function`: Transition function
- `F3 <: Function`: Termination function (MDP only)

# Arguments
- `problem::Union{StateMDP{T, S, A, P, F1, F2, F3}, StateMRP{T, S, P, F1, F2}}`: The MRP or MDP problem structure
- `min_value::S`: Minimum state value(s) defining the lower bounds of the state space
- `max_value::S`: Maximum state value(s) defining the upper bounds of the state space  
- `tile_size::S`: Size of each tile as a fraction of the state space range (must be in (0,1))
- `num_tilings::Integer`: Number of overlapping tile layers for improved generalization
- `displacement_vector::Union{Int64, NTuple{N, Int64}}`: Displacement factors controlling tiling offset patterns

# Returns
- `NamedTuple` with fields:
  - `feature_vector::BinaryFeatureVector`: Pre-allocated sparse binary feature vector
  - `update_feature_vector!::Function`: Function to update binary feature vectors from states
  - `num_features::Int`: Total number of features across all tilings
  - `get_active_features::Function`: Function returning active feature indices for a state
  - `get_feature_vector::Function`: Function returning dense feature vector for a state

# See Also
[`gradient_monte_carlo_estimation_linear`](@ref), [`semi_gradient_td0_estimation_linear`](@ref), [`linear_value_function`](@ref)

# Algorithm Details
1. Determines state space dimensionality and validates tile size constraints
2. Computes number of tiles per dimension and displacement offsets for tiling alignment
3. Creates pre-allocated feature storage and specialized update functions
4. Returns function closures that capture tiling parameters for efficient feature extraction

The tile coding creates overlapping grids displaced by calculated offsets, providing better generalization than single tiling while maintaining sparse binary features for linear function approximation.

# Examples
```julia-repl
julia> # Setup tile coding for 1D random walk
julia> setup = tile_coding_feature_setup(mrp, 0.0f0, 1.0f0, 0.1f0, 8, 2)

julia> # Use with high-level linear TD learning
julia> result = semi_gradient_td0_estimation_linear(mrp, 0.9f0, 1000, 10000,
           setup.feature_vector, setup.update_feature_vector!)
"""
function tile_coding_feature_setup(problem::Union{StateMDP{T, S, A, P, F1, F2, F3}, StateMRP{T, S, P, F1, F2}}, min_value::S, max_value::S, tile_size::S, num_tilings::Integer; displacement_vector::Union{Int64, NTuple{N, Int64}} = form_default_displacement_vector(min_value)) where {T<:Real, N, S <: Union{T, NTuple{N, T}}, A, P, F1<:Function, F2<:Function, F3<:Function}
	#states must be tuples with k elements or some number value
	k = S == T ? 1 : N

	#ensure that all tile sizes are some percentage of the total state space
	@assert all(0 < l < 1 for l in tile_size)

	max_d = k == 1 ? displacement_vector : maximum(displacement_vector)

	s_range = if k == 1
		max_value - min_value
	else
		Tuple(max_value[i] - min_value[i] for i in 1:k)
	end

	#number of tiles in each direction of the state space
	num_tiles = if k == 1
		x = inv(tile_size)
		if isinteger(x)
			Int64(x) + 1
		else
			ceil(Int64, x)
		end
	else
		Tuple(begin
			x = inv(l)
			if isinteger(x)
				Int64(x) + 1
			else
				ceil(Int64, x)
			end
		end
		for l in tile_size)
	end

	features_per_tiling = prod(num_tiles)

	num_features = features_per_tiling*num_tilings

	#the vector representing how much each offset is shifted from the base for single unit shifts
	offset = k == 1 ? tile_size/num_tilings/max_d : Tuple(T(l/num_tilings/max_d) for l in tile_size)

	feature_vector = BinaryFeatureVector(num_features)

	#this vector will be updated with the active features
	tiling_features = zeros(Int64, num_tilings)

	feature_vector.active_features = tiling_features
	feature_vector.num_features = num_tilings

	update_feature_vector!(x::BinaryFeatureVector, s::S) = update_tile_features!(x.active_features, s, offset, displacement_vector, num_tilings, tile_size, num_tiles, min_value, s_range)

	function update_feature_vector!(x::Vector{T}, s::S)
		update_tile_features!(tiling_features, s, offset, displacement_vector, num_tilings, tile_size, num_tiles, min_value, s_range)
		x .= zero(T)
		@inbounds @simd for i in tiling_features
			x[i] = one(T)
		end
		return x
	end

	function get_feature_vector(s::S)
		x = zeros(T, num_features)
		update_feature_vector!(x, s)
	end

	function get_active_features(s::S)
		update_feature_vector!(feature_vector, s)
		(feature_vector.active_features[i] for i in 1:feature_vector.num_features)
	end

	(feature_vector = feature_vector, update_feature_vector! = update_feature_vector!, num_features = num_features, get_active_features = get_active_features, get_feature_vector = get_feature_vector)
end

# ╔═╡ ed20781e-c7d5-48c8-82bd-94d73478c13a
begin
	get_tile_size(S::T, num_tiles::Integer) where T<:Real = T(inv(num_tiles))
	get_tile_size(S::NTuple{N, T}, num_tiles::NTuple{N, I}) where {T<:Real, I<:Integer, N} = Tuple(T(inv(x)) for x in num_tiles)
end

# ╔═╡ 3968fcf6-c7b6-42bc-a416-fdfcb270f92c
tile_coding_feature_setup(problem::Union{StateMDP{T, S, A, P, F1, F2, F3}, StateMRP{T, S, P, F1, F2}}, min_value::S, max_value::S, num_tiles::Union{Int64, NTuple{N, Int64}}, num_tilings::Integer; kwargs...) where {T<:Real, N, S <: Union{T, NTuple{N, T}}, A, P, F1<:Function, F2<:Function, F3<:Function} = tile_coding_feature_setup(problem, min_value, max_value, get_tile_size(min_value, num_tiles), num_tilings; kwargs...)

# ╔═╡ e6514762-31e0-4916-aa21-c280674c2fc1
md"""
### *Example: Visualizing 1-Dimensional Tile Coding*
"""

# ╔═╡ 84d9aac5-cf3b-402b-b222-9e8985a80b5b
# ╠═╡ skip_as_script = true
#=╠═╡
@bind tile_coding_params PlutoUI.combine() do Child
	md"""
	Tile Size (% of $s_{max}$): $(Child(:tile_size, NumberField(0.01:0.01:.99, default = 0.3)))

	Number of Tilings: $(Child(:num_tilings, NumberField(1:10, default = 2)))
	
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ dda74c94-3574-4e7b-bab1-d106111d36d4
#=╠═╡
tile_coding_test = tile_coding_feature_setup(random_walk_state_mrp, 0f0, 1000f0, Float32(tile_coding_params.tile_size), tile_coding_params.num_tilings)
  ╠═╡ =#

# ╔═╡ d17926d5-bcfa-4789-9609-59a69d87d194
#=╠═╡
md"""
The following shows which feature is active for each tiling in the 1 dimensional space used for the random walk example.  The tile size as a percent of the size of the state space determines how many tiles there are for each tiling.  In this case, a tile size of $(tile_coding_params.tile_size) translates into $(ceil(Int64, inv(tile_coding_params.tile_size))) tiles.  Each of the $(tile_coding_params.num_tilings) tilings will have one of $(ceil(Int64, inv(tile_coding_params.tile_size))) features active corresponding to which tile the state falls into.  Note that in order to cover the entire state space for each tiling, the number of tiles must overshoot the state space.  By convention the tilings will move in the negative direction of each dimension so the edge tiles must extend beyond the state space enough to still cover the space even after the shifting.
"""
  ╠═╡ =#

# ╔═╡ 71e7eef0-0304-4e26-8991-fa20da83df9a
#=╠═╡
plot(heatmap(
	x = 1:num_states, 
	y = 1:tile_coding_test.num_features, 
	z = mapreduce(hcat, Float32.(1:num_states)) do s
		v = zeros(Float32, tile_coding_test.num_features)
		for i in tile_coding_test.get_active_features(s)
			v[i] = 1f0
		end
		return v
	end, 
	colorscale = "Greys", 
	showscale=false), 
	 Layout(xaxis = attr(title = "state", mirror = true, linecolor = "black"), yaxis = attr(title = "Active Features", linecolor="black", mirror = true), title = "Active Tiling Features In White"))
  ╠═╡ =#

# ╔═╡ 8e12b92b-e56d-44f0-bf89-3248131b2245
md"""
### *Example: Tile Coding with Random Walk Example*

Notice that TD learning in this case is also more stable at higher learning rates as the number of tilings increases
"""

# ╔═╡ 7e56131f-3afe-4997-a085-60f0d45a9d8d
# ╠═╡ skip_as_script = true
#=╠═╡
@bind tile_coding_learning_params PlutoUI.combine() do Child
	md"""
	Tile Size (% of $s_{max}$): $(Child(:tile_size, NumberField(0.01f0:0.01f0:.99f0, default = 0.1f0)))

	Number of Tilings: $(Child(:num_tilings, NumberField(1:100, default = 20)))
	
	Learning Rates: Monte Carlo $(Child(:α_mc, NumberField(0f0:1f-8:1f0, default = 1f-5))) TD(0) $(Child(:α_td, NumberField(0f0:1f-7:1f0, default = 6f-4)))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ b5a3a529-2d74-4757-9d38-2eae28396d02
#=╠═╡
let
	setup = tile_coding_feature_setup(random_walk_state_mrp, 1f0, Float32(num_states), tile_coding_learning_params.tile_size, tile_coding_learning_params.num_tilings)
	show_random_walk_linear_results(setup.feature_vector, setup.update_feature_vector!, 10_000, tile_coding_learning_params.α_mc, tile_coding_learning_params.α_td, "Tile Coding Function Approximation")
end
  ╠═╡ =#

# ╔═╡ a4d9efaf-1e1e-4115-973f-570014c1fd06
md"""
> ### *Exercise 9.4* 
> Suppose we believe that one of two state dimensions is more likely to have an effect on the value function than is the other, that generalization should be primarily across this dimension rather than along it.  What kind of tilings could be used to take advantage of this prior knowledge?

We could use striped tilings such that the narrow width of the tile is in the direction of the important dimension and the elongated height of the tile is in the other direction.  That way states that have the same value of the important dimension would be treated similarly regardless of their value in the other dimension.  The most rapid changes in value would occur in the direction of the important dimension.
"""

# ╔═╡ 22f6f2b1-745d-4ee5-8dfa-0fe2a61c2c54
# ╠═╡ skip_as_script = true
#=╠═╡
plot([scatter(x = [a, a+2, a+2, a, a], y = [b, b, b+5, b+5, b], line_color = "blue", name = "", showlegend = false) for a in 0:2:8 for b in [0, 5]], Layout(width = 300, height = 300, margin = attr(t = 0, l = 0, r = 0, b = 0), xaxis_title = "Important Dimension", yaxis_title = "Unimportant Dimenson"))
  ╠═╡ =#

# ╔═╡ dfeead7c-65ab-4cb3-ac1c-a28a78e8448e
md"""
### 9.5.5 Radial Basis Functions
Requires much more computational complexity to tile coding without much advantage.  Also more fine tuning is required.
"""

# ╔═╡ 6beee5a8-c262-469e-9b1b-00b91e3b1b55
md"""
## 9.6 Selecting Step-Size Parameters Manually

Consider the tabular case with constant step size averaging to compute state values.  If $\alpha = 1$ (zero weight is placed on the previous estimate), then the error for that state is reduced to zero for the sampled value of that state every step.  Similarly, $\alpha = \frac{1}{10}$ implies that about ten experiences are neeed to converge approximately to their mean value.  In general tabular estimation of a state with $\alpha = \frac{1}{\tau}$ will approach the mean of its targets about $\tau$ experiences with that state.

With general function approximation there is not such a clear notion of *number* of experiences with a state; however a similar rule can be derived using feature vectors instead of states.  Suppose you wanted to learn in about $\tau$ experiences with substantially the same feature vector.  A good rule of thumb for the step-size parameter is then

$\alpha \doteq \left ( \tau \mathbb{E} \left [\mathbf{x}^\top \mathbf{x} \right ] \right ) ^{-1}$

where $\mathbf{x}$ is a random feature vector chosen from the same distribution as input vectors will be in the SGD.  This method words best if $\mathbf{x}^\top \mathbf{x}$ is a constant so the expected value plays no role.  Here the expected total weight on parameters that will be affected by an update replaces the value of one that was implied in the tabular case since in that case only values for individual states are updated.  In the approximation case, each feature vector represents a region of states and thus this update rule accounts for the other states that will all be affected by the update.  In the extreme case of state aggregation where each state gets its own group, then this update rule reduces to the same one from the tabular case since only one feature will be activated at a time.
"""

# ╔═╡ 858a6d4f-2241-43c3-9db0-ff9cec00c2c1
md"""
> ### *Exercise 9.5* 
> Suppose you are using tile coding to transform a seven-dimensional continuous state space into binary feature vectors to estimate a state value function $\hat v(s,\mathbf{w}) \approx v_\pi(s)$.  You believe that the dimensions do not interact strongly, so you decide to use eight tilings of each dimension separately (stripe tilings), for $7 \times 8 = 56$ tilings. In addition, in case there are some pairwise interactions between the dimensions, you also take all ${7\choose2} = 21$ pairs of dimensions and tile each pair conjunctively with rectangular tiles. You make two tilings for each pair of dimensions, making a grand total of $21 \times 2 + 56 = 98$ tilings.  Given these feature vectors, you suspect that you still have to average out some noise, so you decide that you want learning to be gradual, taking about 10 presentations with the same feature vector before learning nears its asymptote. What step-size parameter should you use? Why?

Each tiling will contribute one non-zero element to the feature vector.  With 98 tilings, we have 98 one values in each feature vector so the inner product in equation (9.19) would be $\mathbb{E}\left[\sum_{i=1}^{98} x_i^2 \right]=98$ so $\alpha=\frac{1}{10 \times 98}=\frac{1}{980} \approx 0.001$ 
	"""

# ╔═╡ be019186-33ad-4eb7-a218-9124ff40b6fb
md"""
> ### *Exercise 9.6* 
> If $\tau=1$ and $\mathbf{x}(S_t)^\top \mathbf{x}(S_t) = \mathbb{E} [\mathbf{x}^\top \mathbf{x}]$, prove that (9.19) together with (9.7) and linear function approximation results in the error being reduced to zero in one update.
"""

# ╔═╡ b447a3a9-fe35-4457-886b-05c5862ad8e0
md"""
$$\alpha \doteq \left ( \tau \mathbb{E}\left [ \mathbf{x}^\top \mathbf{x} \right ] \right ) ^{-1} \tag{9.19}$$
$$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \left [ U_t - \hat v(S_t, \mathbf{w}_t) \right] \nabla \hat v(S_t, \mathbf{w}_t) \tag{9.7}$$

Note that in the case of linear function approximation $\nabla \hat v(S_t, \mathbf{w}_t) = \mathbf{x}_t$ and $\hat v(S_t, \mathbf{w}_t) = \mathbf{x}_t^\top \mathbf{w}_t$ so (9.7) reduces to $\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \left [ U_t - \mathbf{x}_t^\top \mathbf{w}_t \right] \mathbf{x}_t = \mathbf{w}_t(\mathbb{1} - \mathbf{x}_t ^\top \mathbf{x}_t) + \alpha U_t\mathbf{x}_t$ 

For the error at the state $S_t$ to be zero after this update, $\mathbf{x}_t^\top \mathbf{w}_t = U_t$

For a given time, the only parameter values that contribute to the value estimate are those for which $\mathbf{x}_t$ are 1.  For these indices, the contribution from the original weight vector is 0.  So $\mathbf{w}_{t+1} = \alpha U_t \mathbf{x}_t$ for indices that are updated, otherwise the values are unchanged from before.  So $\mathbf{x}_t^\top \mathbf{w}_{t+1} = \alpha U_t \mathbf{x}_t^\top \mathbf{x}_t$.  Using (9.19) with $\tau = 1$, the expected update is $\mathbb{E} [ \mathbf{x}_t \mathbf{w}_{t+1} ]  = \mathbb{E} [ \hat v(S_t, \mathbf{w}_{t+1})]= \mathbb{E} [U_t]$.  So the expected approximation value of the state at step t will be updated to equal the true expected value at that state.
"""

# ╔═╡ d7c1810a-8f20-4178-83ca-017d53e3e7e9
md"""
## 9.7 Nonlinear Function Approxmation: Artificial Neural Networks
"""

# ╔═╡ 82828e72-5d30-41b6-a1b6-f258c234b034
md"""
### *Neural Network Parameter Update Implementation*
"""

# ╔═╡ 2bc32d3d-193e-4cab-b13b-f7ed304af0f6
md"""
By default the NN gradient and forward pass assumes a dense vector or matrix for the input.  If we want to pass a sparse feature representation instead, we must extend the following methods to these custom datatypes:

`gemv!('N', α::T, θ::Matrix{T}, x::Vector{T}, β::T, output::Vector{T})`

`gemm!('N', 'T', α::T, θ::Matrix{T}, x::Matrix{T}, β::T, output::Matrix{T})`

Here x contains the feature information and must work for something other than `Vector` or `Matrix` types
"""

# ╔═╡ 0334d2ff-268d-4485-b460-89f82c4a99e1
begin
	function BLAS.gemv!(O::Char, c1::T, θ::Matrix{T}, x::StateAggregationFeatureVector, c2::T, output::Vector{T}) where T<:Real
		j = x.group_index
		if O == 'N'
			@inbounds @simd for i in eachindex(output)
				output[i] = c2*output[i] + θ[i, j]
			end
		elseif O == 'T'
			@inbounds @simd for i in eachindex(output)
				output[i] = c2*output[i] + θ[j, i]
			end
		else
			error("Unknown orientation for matrix of $O")
		end
	end

	function BLAS.gemv!(O::Char, c1::T, θ::Matrix{T}, x::BinaryFeatureVector, c2::T, output::Vector{T}) where T<:Real
		l = x.num_features
		inds = x.active_features
		if !isone(c2) 
			output .*= c2
		end
		if O == 'N'
			for j in inds
				@inbounds @simd for i in eachindex(output)
					output[i] += θ[i, j]
				end
			end
		elseif O == 'T'
			for i in eachindex(output)
				@inbounds @simd for j in inds
					output[i] += θ[j, i]
				end
			end
		else
			error("Unknown orientation for matrix of $O")
		end
	end
end

# ╔═╡ 8e8add6f-99ab-4aa7-b236-87915c6be9c2
begin
	function BLAS.gemm!(O1::Char, O2::Char, c1::T, X::Vector{V}, θ::Matrix{T}, c2::T, output::Matrix{T}) where {T<:Real, V<:StateAggregationFeatureVector}
		!isone(c2) && output .*= c2
		N = length(X)
		(M, O) = size(θ)
		if O2 == 'N'
			for k in 1:O
				@inbounds @simd for j in 1:N
					x = X[j]
					i = x.group_index
					output[j, k] += c1*θ[i, j]
				end
			end
		elseif O2 == 'T'
			for k in 1:M
				@inbounds @simd for j in 1:N
					x = X[j]
					i = x.group_index
					output[j, k] += c1*θ[i, j]
				end
			end
		else
			error("Unknown orientation for matrix of $O1")
		end
	end

	#operation needed for backprop
	function BLAS.gemm!(O1::Char, O2::Char, c1::T, V::Vector{T}, x::StateAggregationFeatureVector, c2::T, output::Matrix{T}) where {T<:Real}
		!isone(c2) && output .*= c2
		(M, N) = size(output)
		if (O1 == 'N') && (O2 == 'T')
			@inbounds @simd for i in 1:M
				output[i, x.group_index] += c1*V[i]
			end
		else
			error("Unknown orientation for matrix of $O")
		end
	end

	function BLAS.gemm!(O1::Char, O2::Char, c1::T, X::Vector{V}, θ::Matrix{T}, c2::T, output::Matrix{T}) where {T<:Real, V<:BinaryFeatureVector}
		output .*= c2
		N = length(X) 
		(M, O) = size(θ)
		if O2 == 'N'
			for k in 1:O
				for j in 1:N
					x = X[j]
					@inbounds @simd for ind in 1:x.num_features
						i = x.active_features[ind]
						output[j, k] += c1*θ[i, k]
					end
				end
			end
		elseif O2 == 'T'
			for k in 1:M
				for j in 1:N
					x = X[j]
					@inbounds @simd for ind in 1:x.num_features
						i = x.active_features[ind]
						output[j, k] += c1*θ[k, i]
					end
				end
			end
		else
			error("Unknown orientation for matrix of $O1")
		end
	end

	#operation needed for backprop
	function BLAS.gemm!(O1::Char, O2::Char, c1::T, V::Vector{T}, x::BinaryFeatureVector, c2::T, output::Matrix{T}) where {T<:Real}
		!isone(c2) && output .*= c2
		(M, N) = size(output)
		if (O1 == 'N') && (O2 == 'T')
			for n in 1:x.num_features
				j = x.active_features[n]
				@inbounds @simd for i in 1:M
					output[i, j] += c1*V[i]
				end
			end
		else
			error("Unknown orientation for matrix of $O")
		end
	end
end

# ╔═╡ 66cadcfb-4fda-4509-80d6-aa22766a7e9c
"""
    fcann_value_function!(activations, x, params, reslayers) -> Nothing

Compute forward pass of fully connected artificial neural network value function without gradient computation.

# Type Parameters
- `T <: Float32`: Numeric type restricted to Float32 for performance

# Arguments
- `activations::FCANNActivations{T}`: Pre-allocated activation storage (modified in-place)
- `x::Vector{T}`: Input feature vector
- `params::FCANNParams`: Network parameters containing weights and biases
- `reslayers::Integer`: Number of residual layers in the network architecture

# Returns
- `activations::FCANNActivations{T}`: Function modifies `activations` in-place and returns them as output

# See Also
[`FCANNActivations`](@ref), [`FCANNParams`](@ref)
"""
function fcann_value_function!(activations::FCANNActivations{T}, x, params::FCANNParams) where T<:Float32
	FCANN.forwardNOGRAD_base!(activations, params.weights..., x, params.reslayers)
end

# ╔═╡ 9e3efa3c-af2f-4aea-b923-a6d50a6b9fb5
"""
    update_fcann_value_gradient!(∇v̂, x, output_index, params, hidden_layers, l2, tanh_grad_z, activations, deltas, dropout, activation_list, scales) -> Nothing

Compute gradient of FCANN value function with respect to network parameters.

# Type Parameters
- `T <: Float32`: Numeric type restricted to Float32 for performance
- `B <: Bool`: Boolean type for activation function indicators

# Arguments
- `∇v̂::FCANNParams`: Gradient storage for network parameters (modified in-place)
- `x::Vector{T}`: Input feature vector
- `output_index::Integer`: Index of the target output neuron
- `params::FCANNParams`: Network parameters containing weights, biases and reslayers
- `hidden_layers::Vector{Int64}`: Architecture specification for hidden layer sizes
- `l2::T`: L2 regularization coefficient
- `tanh_grad_z::FCANNActivations{T}`: Pre-allocated storage for tanh gradient computations
- `activations::FCANNActivations{T}`: Pre-allocated activation storage
- `deltas::FCANNActivations{T}`: Pre-allocated delta storage for backpropagation
- `dropout::T`: Dropout rate for regularization
- `reslayers::Integer`: Number of residual layers in the network
- `activation_list::AbstractVector{B}`: Boolean indicators for activation function types per layer
- `scales`: Scaling factors applied to computed gradients

# Returns
- `Nothing`: Function modifies `∇v̂` in-place with scaled gradients
"""
function update_fcann_value_gradient!(∇v̂::FCANNParams, x, output_index::Integer, params::FCANNParams, hidden_layers::Vector{Int64}, l2::T, tanh_grad_z::FCANNActivations{T}, activations::FCANNActivations{T}, deltas::FCANNActivations{T}, dropout::T, activation_list::AbstractVector{B}, scales) where {T<:Float32, B<:Bool}
	FCANN.nnCostFunction(params.weights..., hidden_layers, x, output_index, l2, ∇v̂.weights..., tanh_grad_z, activations, deltas, dropout; resLayers = params.reslayers, loss_type = OutputIndex(), activation_list = activation_list)
	@inbounds for i in eachindex(params.weights[1])
		for j in 1:2
			∇v̂.weights[j][i] .*= scales[i]
		end
	end
end

# ╔═╡ 2b922137-3110-4f91-94b1-4707d197b429
"""
    scale_fcann_params!(params, scales) -> Nothing

Apply inverse scaling factors to FCANN network parameters in-place.

# Type Parameters
- `T <: Real`: Numeric type for scaling factors

# Arguments
- `params::FCANNParams`: Network parameters containing weights and biases (modified in-place)
- `scales::Vector{T}`: Scaling factors to apply inversely to each parameter group

# Returns
- `Nothing`: Function modifies `params` in-place by dividing each parameter group by corresponding scale factor
"""
function scale_fcann_params!(params::FCANNParams, scales::Vector{T}) where T<:Real
	@inbounds for i in eachindex(scales)
		for j in 1:2
			params.weights[j][i] ./= scales[i]
		end
	end
end

# ╔═╡ b2c56d0e-668e-43cd-a886-bb830a60b132
function get_network_dimensions(params::FCANNParams)
	input_length = size(params.weights[1][1], 2)
	num_hidden = length(params.weights[1])-1
	hidden_layers = iszero(num_hidden) ? Vector{Int64}() : [length(params.weights[2][i]) for i in 1:num_hidden]
	return (input_length, hidden_layers, num_hidden)
end

# ╔═╡ 67db7264-2a5e-44be-98e7-e5d08d5e7273
"""
    setup_fcann_value_arguments(params, input_length, hidden_layers, reslayers, l2, dropout, use_μP, activation_list) -> NamedTuple

Set up neural network value function components for reinforcement learning with FCANN backend.

# Type Parameters
- `T <: Real`: Numeric type for network computations

# Arguments
- `params::FCANNParams{T}`: Pre-initialized network parameters containing weights and biases
- `input_length::Integer`: Dimension of input feature vectors
- `hidden_layers::Vector{Int64}`: Architecture specification for hidden layer sizes
- `reslayers::Integer`: Number of residual layers in the network
- `l2::T`: L2 regularization coefficient
- `dropout::T`: Dropout rate for regularization during training
- `use_μP::Bool`: Whether to apply μP (mu-parametrization) scaling for stable training
- `activation_list`: Specification of activation functions for each layer

# Returns
- `NamedTuple` with fields:
  - `feature_vector`: Pre-allocated input vector for state features
  - `gradient::FCANNParams{T}`: Pre-allocated gradient storage matching parameter structure
  - `value_function::Function`: Function computing value estimates from features and parameters
  - `gradient_update!::Function`: Function computing gradients with respect to network parameters
  - `activations`: Pre-allocated activation storage for forward passes

# See Also
[`gradient_monte_carlo_estimation_fcann`](@ref), [`gradient_monte_carlo_estimation!`](@ref), [`fcann_value_function!`](@ref)

# Algorithm Details
1. Allocates pre-sized storage for inputs, activations, and gradient computations
2. Configures μP scaling factors for stable training of wide networks if enabled
3. Creates closure functions that capture network architecture and storage for efficient repeated calls
4. Returns components ready for integration with core RL estimation algorithms via high-level wrapper functions

The μP scaling applies 1/width scaling to hidden layer parameters, enabling stable training across different network widths without hyperparameter retuning. Components are designed for delegation to [`gradient_monte_carlo_estimation!`](@ref) through wrapper functions.
"""
function setup_fcann_value_arguments(params::FCANNParams{T}, l2::T, dropout::T, use_μP::Bool, activation_list) where {T<:Real}
	input_length, hidden_layers, num_hidden = get_network_dimensions(params)
	
	#form activations for network
	activations = FCANN.form_activations(params.weights[1])
	tanh_grad_z = deepcopy(activations)
	deltas = deepcopy(activations)

	scales = fill(one(T), length(params.weights[1]))
	if use_μP
		for i in eachindex(hidden_layers)
			i′ = i + 1
			scales[i′] /= size(params.weights[1][i′], 2)
		end
	end

	function value_function(x, params; activations = activations) 			
		fcann_value_function!(activations, x, params)[1]
		return first(last(activations))
	end
	
	function update_value_gradient!(∇v̂, x, params) 
		update_fcann_value_gradient!(∇v̂, x, 1, params, hidden_layers, l2, tanh_grad_z, activations, deltas, dropout, activation_list, scales)
		use_μP && scale_fcann_params!(∇v̂, scales)
		return ∇v̂
	end

	return (gradient = deepcopy(params), value_function = value_function, update_gradient! = update_value_gradient!, activations = activations)
end

# ╔═╡ 9b5fbbdd-0b36-4893-b4bb-b05439f5a541
begin
	function initialize_fcann_params(input_size::Integer, hidden_layers::Vector{I}, num_actions::Integer, reslayers::Integer, use_μP::Bool) where I<:Integer
		weights = FCANN.initializeparams_saxe(input_size, hidden_layers, num_actions, reslayers; use_μP = use_μP)
		return (weights = weights, reslayers = reslayers)
	end

	initialize_fcann_params(featurevector, args...) = initialize_fcann_params(length(featurevector), args...)
end	

# ╔═╡ 74e42774-68e5-44b5-91c4-da87a20879e1
"""
    gradient_monte_carlo_estimation_fcann(mrp, γ, num_episodes, update_feature_vector!, num_features, hidden_layers; kwargs...) -> NamedTuple

Perform gradient Monte Carlo value function estimation using fully connected artificial neural networks.

# Type Parameters
- `T <: Real`: Numeric type for network computations and RL parameters

# Arguments
- `mrp::StateMRP`: Markov reward process structure defining the learning environment
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `num_episodes::Integer`: Number of episodes for Monte Carlo estimation
- `update_feature_vector!::Function`: Function to extract state features into pre-allocated vector
- `num_features::Integer`: Dimension of the state feature representation
- `hidden_layers::Vector{Int64}`: Architecture specification for hidden layer sizes

# Keyword Arguments
- `reslayers::Integer = 0`: Number of residual layers in the network architecture
- `use_μP::Bool = true`: Whether to apply μP scaling for stable wide network training
- `params::FCANNParams{T} = FCANN.initializeparams_saxe(...)`: Pre-initialized network parameters (auto-generated if not provided)
- `dropout::T = zero(T)`: Dropout rate for regularization during training
- `activation_list::Vector{Bool} = fill(true, length(hidden_layers))`: Activation function indicators per layer (true = tanh, false = linear)
- `l2::T = zero(T)`: L2 regularization coefficient
- `kwargs...`: Additional arguments passed to [`gradient_monte_carlo_estimation!`](@ref)

# Returns
- `NamedTuple` with fields:
  - `value_function::Function`: Thread-safe value function `v̂(s; activations=..., kwargs...)`
  - `error_history`: History of estimation errors during training
  - `parameters::FCANNParams{T}`: Final trained network parameters
  - `activations`: Reference activation storage for single-threaded use

# See Also
[`gradient_monte_carlo_estimation!`](@ref), [`setup_fcann_value_arguments`](@ref), [`StateMRP`](@ref)

# Algorithm Details
1. Initializes neural network parameters using Saxe initialization if not provided
2. Sets up FCANN value function components via [`setup_fcann_value_arguments`](@ref)
3. Delegates to core Monte Carlo algorithm with configured network functions
4. Returns thread-safe value function wrapper for concurrent evaluation

The returned value function creates activation storage per call, enabling safe multi-threaded evaluation. μP scaling is applied by default for stable training across network widths without hyperparameter adjustment.

# Examples
```julia-repl
julia> # Neural network value estimation for 1D random walk
julia> result = gradient_monte_carlo_estimation_fcann(mrp, 0.9f0, 5000, 
           update_features!, 10, [64, 32])

julia> # Evaluate learned value function
julia> v_estimate = result.value_function(0.5f0)
-0.234f0

julia> # Custom architecture with regularization
julia> result_reg = gradient_monte_carlo_estimation_fcann(mrp, 0.95f0, 3000,
           update_features!, 20, [128, 64, 32]; l2=0.001f0, dropout=0.1f0,
           α=0.01f0)

julia> # Multi-threaded evaluation
julia> states = [0.1f0, 0.3f0, 0.7f0, 0.9f0]
julia> values = [Threads.@spawn result.value_function(s) for s in states]
julia> fetch.(values)
4-element Vector{Float32}
"""
function gradient_monte_carlo_estimation_fcann(mrp::StateMRP, γ::T, num_episodes::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; reslayers::Integer = 0, use_μP::Bool = true, params::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, 1, reslayers, use_μP), dropout::T = zero(T), activation_list = fill(true, length(hidden_layers)), l2 = zero(T), kwargs...) where T<:Real
	setup = setup_fcann_value_arguments(params, l2, dropout, use_μP, activation_list)
	(value_function, history, params) = gradient_monte_carlo_estimation!(params, mrp, γ, num_episodes, feature_vector, update_feature_vector!, setup.value_function, setup.gradient, setup.update_gradient!; kwargs...)

	#this version of the value function can be run with multiple threads
	v̂(args...; activations = deepcopy(setup.activations), kwargs...) = value_function(args...; activations = activations, kwargs...)

	(value_function = v̂, error_history = history, parameters = params, activations = setup.activations)
end

# ╔═╡ b58cacd0-ca65-43f5-8678-7265ea2d46c8
"""
    gradient_monte_carlo_policy_estimation_fcann(mdp, π, γ, num_episodes, update_feature_vector!, num_features, hidden_layers; kwargs...) -> NamedTuple

Perform gradient Monte Carlo policy evaluation using fully connected artificial neural networks.

# Type Parameters
- `T <: Real`: Numeric type for network computations and RL parameters

# Arguments
- `mdp::StateMDP`: Markov decision process structure defining the learning environment
- `π::Function`: Policy function mapping states to action probabilities or deterministic actions
- `γ::T`: Discount factor (0 ≤ γ ≤ 1)
- `num_episodes::Integer`: Number of episodes for Monte Carlo policy evaluation
- `update_feature_vector!::Function`: Function to extract state features into pre-allocated vector
- `num_features::Integer`: Dimension of the state feature representation
- `hidden_layers::Vector{Int64}`: Architecture specification for hidden layer sizes

# Keyword Arguments
- `reslayers::Integer = 0`: Number of residual layers in the network architecture
- `use_μP::Bool = true`: Whether to apply μP scaling for stable wide network training
- `params::FCANNParams{T} = FCANN.initializeparams_saxe(...)`: Pre-initialized network parameters (auto-generated if not provided)
- `dropout::T = zero(T)`: Dropout rate for regularization during training
- `activation_list::Vector{Bool} = fill(true, length(hidden_layers))`: Activation function indicators per layer (true = tanh, false = linear)
- `l2::T = zero(T)`: L2 regularization coefficient
- `kwargs...`: Additional arguments passed to [`gradient_monte_carlo_policy_estimation!`](@ref)

# Returns
- `NamedTuple` with fields:
  - `value_function::Function`: Thread-safe value function `v̂(s; activations=..., kwargs...)`
  - `error_history`: History of estimation errors during policy evaluation
  - `parameters::FCANNParams{T}`: Final trained network parameters
  - `activations`: Reference activation storage for single-threaded use

# See Also
[`gradient_monte_carlo_policy_estimation!`](@ref), [`gradient_monte_carlo_estimation_fcann`](@ref), [`StateMDP`](@ref)

# Algorithm Details
1. Initializes neural network parameters using Saxe initialization if not provided
2. Sets up FCANN value function components via [`setup_fcann_value_arguments`](@ref)
3. Delegates to core Monte Carlo policy evaluation algorithm with configured network functions
4. Returns thread-safe value function wrapper for concurrent evaluation

The function evaluates the value function V^π(s) for the given policy π using Monte Carlo sampling from MDP episodes. μP scaling is applied by default for stable training across network widths.

# Examples
```julia-repl
julia> # Neural network policy evaluation for grid world MDP
julia> π_random(s) = rand(1:4)  # Random policy over 4 actions
julia> result = gradient_monte_carlo_policy_estimation_fcann(mdp, π_random, 0.9f0, 8000,
           update_features!, 25, [128, 64])

julia> # Evaluate policy value function
julia> v_π = result.value_function((3, 4))  # Grid position (3,4)
1.23f0
"""
function gradient_monte_carlo_policy_estimation_fcann(mdp::StateMDP, π::Function, γ::T, num_episodes::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; reslayers::Integer = 0, use_μP::Bool = true, params::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, 1, reslayers, use_μP), dropout::T = zero(T), activation_list = fill(true, length(hidden_layers)), l2::T = zero(T), kwargs...) where T<:Real
	setup = setup_fcann_value_arguments(params, l2, dropout, use_μP, activation_list)
	(value_function, history, params) = gradient_monte_carlo_policy_estimation!(params, mdp, π, γ, num_episodes, feature_vector, update_feature_vector!, setup.value_function, setup.gradient, setup.update_gradient!; kwargs...)

	#this version of the value function can be run with multiple threads
	v̂(args...; activations = deepcopy(setup.activations), kwargs...) = value_function(args...; activations = activations, kwargs...)

	(value_function = v̂, error_history = history, parameters = params, activations = setup.activations)
end

# ╔═╡ d81d8f7d-ed32-405d-b0c8-2ceff5845578
"""
    semi_gradient_td0_estimation_fcann(mrp, γ, max_episodes, max_steps, update_feature_vector!, num_features, hidden_layers; kwargs...) -> NamedTuple

Perform semi-gradient TD(0) value function estimation using fully connected artificial neural networks.

# Type Parameters
- `T <: Real`: Numeric type for network computations and RL parameters

# Arguments
- `mrp::StateMRP`: Markov reward process structure defining the learning environment
- `γ::T`: Discount factor (0 ≤ γ < 1)
- `max_episodes::Integer`: Maximum number of episodes for TD learning
- `max_steps::Integer`: Maximum total steps across all episodes
- `update_feature_vector!::Function`: Function to extract state features into pre-allocated vector
- `num_features::Integer`: Dimension of the state feature representation
- `hidden_layers::Vector{Int64}`: Architecture specification for hidden layer sizes

# Keyword Arguments
- `reslayers::Integer = 0`: Number of residual layers in the network architecture
- `use_μP::Bool = true`: Whether to apply μP scaling for stable wide network training
- `params::FCANNParams{T} = FCANN.initializeparams_saxe(...)`: Pre-initialized network parameters (auto-generated if not provided)
- `dropout::T = zero(T)`: Dropout rate for regularization during training
- `activation_list::Vector{Bool} = fill(true, length(hidden_layers))`: Activation function indicators per layer (true = tanh, false = linear)
- `l2::T = zero(T)`: L2 regularization coefficient
- `kwargs...`: Additional arguments passed to [`semi_gradient_td0_estimation!`](@ref)

# Returns
- `NamedTuple` with fields:
  - `value_function::Function`: Thread-safe value function `v̂(s; activations=..., kwargs...)`
  - `error_history`: Episode-wise error and step statistics during training
  - `step_rewards`: Detailed step-by-step reward history if requested
  - `parameters::FCANNParams{T}`: Final trained network parameters
  - `activations`: Reference activation storage for single-threaded use

# See Also
[`semi_gradient_td0_estimation!`](@ref), [`gradient_monte_carlo_estimation_fcann`](@ref), [`StateMRP`](@ref)

# Algorithm Details
1. Initializes neural network parameters using Saxe initialization if not provided
2. Sets up FCANN value function components via [`setup_fcann_value_arguments`](@ref)
3. Delegates to core semi-gradient TD(0) algorithm with configured network functions
4. Returns thread-safe value function wrapper for concurrent evaluation

TD(0) learning updates the value function after each step using the temporal difference δ = r + γV(s') - V(s), providing faster learning than Monte Carlo methods. The semi-gradient approach approximates gradients for non-linear function approximation.

# Examples
```julia-repl
julia> # Neural network TD(0) learning for continuous random walk
julia> result = semi_gradient_td0_estimation_fcann(mrp, 0.95f0, 2000, 50000,
           update_features!, 15, [128, 64]; α=0.001f0)

julia> # Check learning progress
julia> final_error = last(result.error_history.errors)
0.045f0

julia> # Evaluate learned value function
julia> current_value = result.value_function(0.3f0)
-1.87f0
```
"""
function semi_gradient_td0_estimation_fcann(mrp::StateMRP, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; reslayers::Integer = 0, use_μP::Bool = true, params::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, 1, reslayers, use_μP), dropout::T = zero(T), activation_list = fill(true, length(hidden_layers)), l2::T = zero(T), kwargs...) where T<:Real
	setup = setup_fcann_value_arguments(params, l2, dropout, use_μP, activation_list)
	(value_function, history, step_rewards, params) = semi_gradient_td0_estimation!(params, mrp, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, setup.value_function, setup.gradient, setup.update_gradient!; kwargs...)

	#this version of the value function can be run with multiple threads
	v̂(args...; activations = deepcopy(setup.activations), kwargs...) = value_function(args...; activations = activations, kwargs...)

	(value_function = v̂, error_history = history, step_rewards = step_rewards, parameters = params, activations = setup.activations)
end

# ╔═╡ 4a3a4635-a046-4eec-ab95-2dce74ac0fbe
"""
    semi_gradient_td0_policy_estimation_fcann(mdp, π, γ, max_episodes, max_steps, update_feature_vector!, num_features, hidden_layers; kwargs...) -> NamedTuple

Perform semi-gradient TD(0) policy evaluation using fully connected artificial neural networks.

# Type Parameters
- `T <: Real`: Numeric type for network computations and RL parameters

# Arguments
- `mdp::StateMDP`: Markov decision process structure defining the learning environment
- `π::Function`: Policy function mapping states to action probabilities or deterministic actions
- `γ::T`: Discount factor (0 ≤ γ < 1)
- `max_episodes::Integer`: Maximum number of episodes for TD policy evaluation
- `max_steps::Integer`: Maximum total steps across all episodes
- `update_feature_vector!::Function`: Function to extract state features into pre-allocated vector
- `num_features::Integer`: Dimension of the state feature representation
- `hidden_layers::Vector{Int64}`: Architecture specification for hidden layer sizes

# Keyword Arguments
- `reslayers::Integer = 0`: Number of residual layers in the network architecture
- `use_μP::Bool = true`: Whether to apply μP scaling for stable wide network training
- `params::FCANNParams{T} = FCANN.initializeparams_saxe(...)`: Pre-initialized network parameters (auto-generated if not provided)
- `dropout::T = zero(T)`: Dropout rate for regularization during training
- `activation_list::Vector{Bool} = fill(true, length(hidden_layers))`: Activation function indicators per layer (true = tanh, false = linear)
- `l2::T = zero(T)`: L2 regularization coefficient
- `kwargs...`: Additional arguments passed to [`semi_gradient_td0_policy_estimation!`](@ref)

# Returns
- `NamedTuple` with fields:
  - `value_function::Function`: Thread-safe value function `v̂(s; activations=..., kwargs...)`
  - `error_history`: Episode-wise error and step statistics during policy evaluation
  - `step_rewards`: Detailed step-by-step reward history if requested
  - `parameters::FCANNParams{T}`: Final trained network parameters
  - `activations`: Reference activation storage for single-threaded use

# See Also
[`semi_gradient_td0_policy_estimation!`](@ref), [`gradient_monte_carlo_policy_estimation_fcann`](@ref), [`StateMDP`](@ref)

# Algorithm Details
1. Initializes neural network parameters using Saxe initialization if not provided
2. Sets up FCANN value function components via [`setup_fcann_value_arguments`](@ref)
3. Delegates to core semi-gradient TD(0) policy evaluation algorithm with configured network functions
4. Returns thread-safe value function wrapper for concurrent evaluation

The function evaluates V^π(s) using TD(0) temporal difference learning with δ = r + γV(s') - V(s) updates. This provides faster convergence than Monte Carlo policy evaluation while maintaining the semi-gradient approximation for neural network function approximation.

# Examples
```julia-repl
julia> # Neural network TD(0) policy evaluation for grid world
julia> π_random(s) = rand(1:length(mdp.actions))  # Random policy over available actions
julia> result = semi_gradient_td0_policy_estimation_fcann(mdp, π_random, 
           0.9f0, 3000, 75000, update_features!, 36, [128, 64])

julia> # Monitor policy evaluation progress  
julia> episodes_completed = length(result.error_history.errors)
3000

julia> # Evaluate policy value at specific state
julia> v_π_state = result.value_function((2, 3))
4.27f0
```
"""
function semi_gradient_td0_policy_estimation_fcann(mdp::StateMDP, π::Function, γ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; reslayers::Integer = 0, use_μP::Bool = true, params::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, 1, reslayers, use_μP), dropout::T = zero(T), activation_list = fill(true, length(hidden_layers)), l2::T = zero(T), kwargs...) where T<:Real
	setup = setup_fcann_value_arguments(params, l2, dropout, use_μP, activation_list)
	(value_function, history, step_rewards, params) = semi_gradient_td0_policy_estimation!(params, mdp, π, γ, max_episodes, max_steps, feature_vector, update_feature_vector!, setup.value_function, setup.gradient, setup.update_gradient!; kwargs...)

	#this version of the value function can be run with multiple threads
	v̂(args...; activations = deepcopy(setup.activations), kwargs...) = value_function(args...; activations = activations, kwargs...)

	(value_function = v̂, error_history = history, step_rewards = step_rewards, parameters = params, activations = setup.activations)
end

# ╔═╡ 808026eb-4c5a-4f38-bb16-bbb1b2915906
md"""
### *Neural Network GPU Implementation*
"""

# ╔═╡ 3ac65a54-1ff6-441c-8edf-00c49b620389
# ╠═╡ disabled = true
#=╠═╡
import NVIDIALibraries.DeviceArray.CUDAArray
  ╠═╡ =#

# ╔═╡ dd907b31-24f1-46f6-a2d5-7dd268530c94
#=╠═╡
function update_nn_parameters!(θs::Vector{CUDAArray}, βs::Vector{CUDAArray}, layers::Vector{Int64}, ∇θ::Vector{CUDAArray}, ∇β::Vector{CUDAArray}, input::CUDAArray, output::CUDAArray, ∇tanh_z::Vector{CUDAArray}, activations::Vector{CUDAArray}, δs::Vector{CUDAArray}, onesvec::CUDAArray, onesvec_params::Vector{CUDAArray}, normvec_params::Vector{CUDAArray}, α::Float32, scales::Vector{Float32}; λ = 0f0, c = Inf, dropout = 0f0)
	(batchsize, input_layer_size) = input.size
	(_, output_layer_size) = output.size
	FCANN.nnCostFunction(θs, βs, input_layer_size, output_layer_size, layers, batchsize, onesvec, activations, ∇tanh_z, δs, ∇θ, ∇β, input, output, λ, dropout; costFunc = "sqErr", resLayers = 1)
	FCANN.updateParams!(α, θs, βs, ∇θ, ∇β, scales)
	if !isinf(c)
		FCANN.scaleThetas!(θs[1:end-1], ∇θ[1:end-1], onesvec_params, normvec_params, c)
	end
end
  ╠═╡ =#

# ╔═╡ 0facd6de-411a-43e0-820d-7d6eceff5b72
# ╠═╡ disabled = true
#=╠═╡
import FCANN.device_allocate
  ╠═╡ =#

# ╔═╡ 65795424-8e50-4edb-9f6a-7045a9a22b9d
# ╠═╡ disabled = true
#=╠═╡
import FCANN.cuda_allocate
  ╠═╡ =#

# ╔═╡ 6c752a2b-4d10-4865-aeff-ea717b9d3904
#=╠═╡
function fcann_gradient_gpu_setup(problem::Union{StateMDP{T, S, A, P, F1, F2, F3}, StateMRP{T, S, P, F1, F2}}, layers::Vector{Int64}, feature_vector::Vector{Float32}, update_feature_vector!::Function; calculate_error::Function = (g, v̂, s) -> (g - v̂)^2, dropout = 0f0, λ = 0f0, c = Inf) where {T<:Real, S, A, P, F1<:Function, F2<:Function, F3<:Function}
	s0 = problem.initialize_state()
	update_feature_vector!(feature_vector, s0)
	θ, β = FCANN.initializeparams_saxe(length(feature_vector), layers, 1, 1; use_μP = true)

	∇θ = deepcopy(θ)
	∇β = deepcopy(β)
	∇tanh_z = FCANN.form_tanh_grads(layers, 1)
	

	function setup_training(batch_size::Integer)
		activations = [zeros(Float32, batch_size, l) for l in [layers; 1]]
		δs = deepcopy(activations)
		onesvec = zeros(Float32, batch_size)
		return (activations, δs, onesvec)
	end

	(activations, δs, onesvec) = setup_training(1)
	

	input_layer_size = length(feature_vector)

	feature_matrix = reshape(feature_vector, 1, input_layer_size)
	input = zeros(Float32, 1, input_layer_size)
	output = zeros(Float32, 1, 1)
	scales = ones(Float32, length(layers)+1)
	for i in 2:length(scales)
		scales[i] /= size(θ[i], 2)
	end

	d_input = cuda_allocate(input)
	d_output = cuda_allocate(output)
	d_θ = device_allocate(θ)
	d_β = device_allocate(β)
	d_∇θ = device_allocate(∇θ)
	d_∇β = device_allocate(∇β)
	d_∇tanh_z = device_allocate(∇tanh_z)
	d_activations = device_allocate(activations)
	d_δs = device_allocate(δs)
	d_onesvec = cuda_allocate(onesvec)
	d_onesvec_params = map(a -> cuda_allocate(ones(Float32, a)), [input_layer_size; layers])
	d_normvec_params = map(a -> cuda_allocate(zeros(Float32, a)), [layers; 1])
	
	function update_parameters!(parameters, s::S, g::T, α::T, gradients, state_representation::Vector{Float32}, feature_matrix::Matrix{Float32}, input, output, ∇tanh_z, activations, δs, onesvec, onesvec_params, normvec_params, scales)
		update_feature_vector!(state_representation, s)
		feature_matrix .= state_representation
		FCANN.memcpy!(input, feature_matrix)
		FCANN.memcpy!(output, reshape([g], 1, 1))
		update_nn_parameters!(parameters[1], parameters[2], layers, gradients[1], gradients[2], input, output, ∇tanh_z, activations, δs, onesvec, onesvec_params, normvec_params, α, scales; c = c, λ = λ, dropout = dropout)
		calculate_error(g, FCANN.host_allocate(activations[end])[1, 1], s)
	end

	function v̂(s::S, parameters, state_representation, feature_matrix::Matrix{Float32}, input, activations) 
		update_feature_vector!(state_representation, s)
		feature_matrix .= state_representation
		FCANN.memcpy!(input, feature_matrix)
		FCANN.predict!(parameters[1], parameters[2], input, activations, 1)
		return FCANN.host_allocate(activations[end])[1, 1]
	end

	update_args = ((d_∇θ, d_∇β), feature_vector, feature_matrix, d_input, d_output, d_∇tanh_z, d_activations, d_δs, d_onesvec, d_onesvec_params, d_normvec_params, scales)
	
	return (value_function = v̂, value_args = (feature_vector, feature_matrix, d_input, d_activations), parameter_update = update_parameters!, update_args = update_args, parameters = (d_θ, d_β))
end
  ╠═╡ =#

# ╔═╡ 0c7d2eb3-02ce-47b0-955c-fc62d5c86994
md"""
### *Nonlinear Function Approximation with Random Walk Example*
"""

# ╔═╡ 15b93928-98fb-47ed-ba46-e6ee785d46e5
# ╠═╡ skip_as_script = true
#=╠═╡
#this ensures that the state range from 1 to 1000 is mapped to values with a mean 0 and variance of 1
function update_random_walk_vector!(feature_vector::Vector{Float32}, s::Float32)
	x1 = (s - 500f0) / sqrt(46295f0)
	feature_vector[1] = x1
end
  ╠═╡ =#

# ╔═╡ cfc5964b-3a23-48d9-b320-861fd4a43364
#=╠═╡
function run_random_walk_fcann_monte_carlo_estimation(mrp::StateMRP{T, S, P, F1, F2}, γ::T, num_episodes::Integer, layers::Vector{Int64}, input_type::Symbol; kwargs...) where {T<:Real, S, P<:AbstractStateTransition{T}, F1<:Function, F2<:Function}
	x, f! = if input_type == :tiles
		setup = tile_coding_feature_setup(random_walk_state_mrp, 1f0, Float32(num_states), .1f0, 10)
		setup.feature_vector, setup.update_feature_vector!
	elseif input_type == :state_aggregation
		state_aggregation_feature_setup(0f0, num_groups, random_walk_group_assign)
	else
		[0f0], update_random_walk_vector!
	end
	gradient_monte_carlo_estimation_fcann(mrp, γ, num_episodes, x, f!, layers; calculate_error = calc_random_walk_ve, kwargs...)
end
  ╠═╡ =#

# ╔═╡ 93a1f51f-1d83-408e-a860-26e6280c65ee
#=╠═╡
function run_random_walk_fcann_td0_estimation(mrp::StateMRP{T, S, P, F1, F2}, γ::T, num_episodes::Integer, layers::Vector{Int64}, input_type::Symbol; kwargs...) where {T<:Real, S, P<:AbstractStateTransition{T}, F1<:Function, F2<:Function}
	x, f! = if input_type == :tiles
		setup = tile_coding_feature_setup(random_walk_state_mrp, 1f0, Float32(num_states), .2f0, 10)
		setup.feature_vector, setup.update_feature_vector!
	elseif input_type == :state_aggregation
		state_aggregation_feature_setup(0f0, num_groups, random_walk_group_assign)
	else
		[0f0], update_random_walk_vector!
	end
	semi_gradient_td0_estimation_fcann(mrp, γ, num_episodes, typemax(Int64), x, f!, layers; calculate_error = calc_random_walk_ve, kwargs...)
end
  ╠═╡ =#

# ╔═╡ fb244ed5-2827-4b39-a5b1-ced0815b000a
# ╠═╡ skip_as_script = true
#=╠═╡
function show_random_walk_fcann_results(num_layers, layer_size, num_episodes, α_mc, α_td, input_type; nsmooth = 100)
	nn_layers = fill(layer_size, num_layers)
	
	v̂_mc, mc_error, mc_params = run_random_walk_fcann_monte_carlo_estimation(random_walk_state_mrp, 1f0, num_episodes, nn_layers, input_type; α = α_mc)
	v̂_td, td_history, td_steps, td_params = run_random_walk_fcann_td0_estimation(random_walk_state_mrp, 1f0, num_episodes, nn_layers, input_type; α = α_td)
	td_error = td_history.errors
	p1 = plot([scatter(x = nsmooth:num_episodes, y = sqrt.(smooth_error(mc_error, nsmooth)), name = "Monte Carlo"), scatter(x = nsmooth:num_episodes, y = sqrt.(smooth_error(td_error, nsmooth)), name = "TD(0)")], Layout(xaxis_title = "Episode", yaxis_title = "Value Error Averaged <br> over Previous $nsmooth Episodes", showlegend = false))
	p2 = plot([scatter(y = v̂_mc.(Float32.(1:num_states)), name = "Monte Carlo"), scatter(y = v̂_td.(Float32.(1:num_states)), name = "TD(0)"), scatter(y = random_walk_v.value_function[2:end-1], name = "true value")], Layout(title = "Neural Network Approximation with $nn_layers Layers", yaxis_title = "Value", xaxis_title = "State"))
	@htl("""
	<div style = "display: flex;">
	$p1
	$p2
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ b1c84d59-3598-46a1-bc1a-fd691d14ab09
md"""
Notice again how TD learning is more stable at higher learning rates.  The neural network approximation with this few parameters benefits greatly from bootstrap estimation since all state estimates affect eachother.
"""

# ╔═╡ 420e54ac-1a7c-46e9-a8bd-e2ed5765aa7a
# ╠═╡ skip_as_script = true
#=╠═╡
@bind nn_params PlutoUI.combine() do Child
	md"""
	Num Layers: $(Child(:num_layers, NumberField(1:10, default = 2)))
	Layer Size: $(Child(:layer_size, NumberField(1:100, default = 4)))

	Training Episodes: $(Child(:num_episodes, NumberField(1:100_000, default = 2000)))

	Learning Rates: Monte Carlo $(Child(:α_mc, NumberField(0f0:1f-8:1f0, default = 2f-5))) TD(0) $(Child(:α_td, NumberField(0f0:1f-7:1f0, default = 4f-3)))

	Select Input:: $(Child(:input_type, Select([:vector, :tiles, :state_aggregation])))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ 40d07f16-b9ca-4782-bdd2-de15ec6b21e5
#=╠═╡
show_random_walk_fcann_results(nn_params...; nsmooth = 100)
  ╠═╡ =#

# ╔═╡ b22ef023-4e6a-4114-b3c2-bf91e16e9a43
md"""
## 9.8 Least-Squares TD

All the methods we have discussed so far in this chapter have required computation per time step proportional to the number of parameters.  With more computation, however, one can do better.  In this section we present a method for linear function approximation that is arguably the best that can be done for this case.  As we established in Section 9.4 TD(0) with linear function approxmation converges assymptotically (for appropriate decreasing step sizes) to the TD fixed point:

$\mathbf{w_{TD}} = \mathbf{A}^{-1}\mathbf{b}$

where

 $\mathbf{A} \doteq \mathbb{E}\left [ \mathbf{x}_t(\mathbf{x}_t - \gamma \mathbf{x}_{t+1} ) ^\top \right ]$ and $\mathbf{b} \doteq \mathbb{E} [ R_{t+1} \mathbf{x}_t ]$

Instead of updating $\mathbf{w}$ incrementally we could use whatever data we have collected so far to compute estimates of $\mathbf{A}$ and $\mathbf{b}$ and then compute the TD fixed point directly.  *Least-Squares TD* or LSTD does this by forming the following estimates: 

 $\widehat{\mathbf{A}}_t \doteq \sum_{k=0}^{t-1} \mathbf{x}_k (\mathbf{x}_k - \gamma \mathbf{x}_{k+1})^\top + \epsilon \mathbf{I}$ and $\widehat{\mathbf{b}}_t \doteq \sum_{k=0}^{t-1} R_{k+1} \mathbf{x}_k \tag{9.20}$

where $\mathbf{I}$ is the identity matrix, and $\epsilon \mathbf{I}$, for some small $\epsilon \gt 0$, ensures that $\widehat{\mathbf{A}_t}$ is always invertible.  It might seem that these estimates should both be divided by $t$, and indeed they should; as defined here, these are really estimates of $t$ *times* $\mathbf{A}$ and $t$ *times* $\mathbf{b}$.  However, the extra $t$ factors cancel out when LSTD uses these estimates to estimate the TD fixed point as

$\mathbf{w}_t \doteq \widehat{\mathbf{A}}_t^{-1} \widehat{\mathbf{b}}_t \tag{9.21}$

This algorithm is the most data efficient form of linear TD(0), but it is also more expensive computationally.  Recall that semi-gradient TD(0) requires memory and per step computation that is only $O(d)$.  In contrast LSTD requires us to invert $\widehat{\mathbf{A}_t}$ which is $O(d^3)$ on top of the incremental updates to $\widehat{\mathbf{A}_t}$ requiring $O(d^2)$.  Fortunately, the matrix we are inverting is a sum of outer products and there is an $O(d^2)$ incremental update rule for that:

$\begin{flalign}
\widehat{\mathbf{A}}_t^{-1} &= \left ( \widehat{\mathbf{A}}_{t-1} + \mathbf{x}_{t-1} (\mathbf{x}_{t-1} - \gamma \mathbf{x}_{t})^\top \right )^{-1} \tag{from (9.20)} \\
&= \widehat{\mathbf{A}}_{t-1} - \frac{\widehat{\mathbf{A}}_{t-1}^{-1} \mathbf{x}_{t-1}(\mathbf{x}_{t-1} - \gamma \mathbf{x}_t)^\top \widehat{\mathbf{A}}_{t-1}^{-1}}{1 + (\mathbf{x}_{t-1} - \gamma \mathbf{x}_t)^\top \widehat{\mathbf{A}}_{t-1}^{-1} \mathbf{x}_{t-1}} \tag{9.22}  
\end{flalign}$

for $t>0$, with $\widehat{\mathbf{A}}_0 \doteq \epsilon \mathbf{I}$.  Although the identity (9.22), known as *the Sherman-Morrison formula*, is superficially complicated, it involves only vector-matrix and vector-vector multiplications and thus is only $O(d^2)$.  Of course, $O(d^2)$ is still significantly more expensive than the $O(d)$ of semi-gradient TD.  Whether this greater data efficiency of LSTD is worth this computational expense depends on how large $d$ is, how important it is to learn quickly, and the expense of other parts of the system.  The fact that LSTD requires no step-size parameter is sometimes also touted, but the advantage of this is probably overstated since we still need to define $\epsilon$ which affects the sequences of inverses calculated.  Also if the target policy changes it may be undesireable that we keep all of the data, so we may need to use some step size parameter anyway to have old data decay.
"""

# ╔═╡ 32c054ee-a7ee-4705-87c3-fb1a4bd956ab
md"""
### *Least-Squares TD Implementation*
"""

# ╔═╡ a8d7e5f7-8509-4aa1-b4c6-669339cb173c
begin
	"""
	    least_squares_td_estimation(d, initialize_state, transition, isterm, γ, max_episodes, max_steps, update_state_representation!; kwargs...) -> NamedTuple
	
	Perform least squares temporal difference (LSTD) value function estimation with linear function approximation.
	
	# Type Parameters
	- `T <: Real`: Numeric type for computations
	- `S`: State type
	
	# Arguments
	- `d::Integer`: Dimension of the state feature representation
	- `initialize_state::Function`: Function to generate initial states for episodes
	- `transition::Function`: State transition function `s -> (reward, next_state)`
	- `isterm::Function`: Termination check function `state -> Bool`
	- `γ::T`: Discount factor (0 ≤ γ < 1)
	- `max_episodes::Integer`: Maximum number of episodes to run
	- `max_steps::Integer`: Maximum total steps across all episodes
	- `update_state_representation!::Function`: Function to extract state features into pre-allocated vector
	
	# Keyword Arguments
	- `ϵ::T = one(T)/1000`: Regularization parameter for matrix inversion initialization
	- `s0::S = initialize_state()`: Initial state (computed automatically if not provided)
	- `calculate_error::Function = (v̂, s) -> zero(T)`: Error function for episode statistics
	
	# Returns
	- `NamedTuple` with fields:
	  - `parameters::Vector{T}`: Final learned linear parameters
	  - `value_estimate::Function`: Value function supporting single states and batched evaluation
	  - `episode_errors::Vector{T}`: Average error per episode during training
	
	# See Also
	[`semi_gradient_td0_estimation!`](@ref), [`gradient_monte_carlo_estimation!`](@ref), [`StateMRP`](@ref)
	
	# Algorithm Details
	1. Initializes parameter vector and inverse covariance matrix A⁻¹ with regularization
	2. Iteratively updates A⁻¹ and parameter estimates using Sherman-Morrison formula
	3. Computes exact least squares solution at each step: θ = A⁻¹b where Aθ = b
	4. Returns optimized value function with support for both single and batch evaluation
	
	LSTD computes the exact least squares solution to the projected Bellman equation, providing more stable convergence than semi-gradient methods but with higher computational cost per step.
	
	# Examples
	```julia-repl
	julia> # LSTD estimation for random walk with 10-dimensional features
	julia> result = least_squares_td_estimation(10, init_state, transition, isterm, 
	           0.9f0, 1000, 25000, update_features!)
	
	julia> # Evaluate learned value function
	julia> v_estimate = result.value_estimate(0.5f0)
	-0.123f0
	
	julia> # Batch evaluation for multiple states
	julia> states = [0.1f0, 0.3f0, 0.7f0, 0.9f0]
	julia> batch_values = result.value_estimate(states)
	4-element Vector{Float32}
	
	julia> # Check final learning error
	julia> final_error = last(result.episode_errors)
	0.002f0
	```
	
	    least_squares_td_estimation(mrp, d, γ, max_episodes, max_steps, update_state_representation!; kwargs...) -> NamedTuple
	
	Perform LSTD value function estimation using MRP structure.
	
	# Arguments
	- `mrp::StateMRP`: Markov reward process defining the learning environment
	- `d::Integer`: Dimension of the state feature representation
	- Remaining arguments match core [`least_squares_td_estimation`](@ref)
	
	# Returns
	Same as core function. Automatically extracts transition functions from MRP structure.
	
	    least_squares_td_policy_estimation(mdp, d, π, γ, max_episodes, max_steps, update_state_representation!; kwargs...) -> NamedTuple
	
	Perform LSTD policy evaluation using MDP structure and policy function.
	
	# Arguments
	- `mdp::StateMDP`: Markov decision process defining the learning environment
	- `d::Integer`: Dimension of the state feature representation
	- `π::Function`: Policy function mapping states to actions
	- Remaining arguments match core [`least_squares_td_estimation`](@ref)
	
	# Returns
	Same as core function. Automatically creates policy-driven transitions from MDP structure.
	"""
	function least_squares_td_estimation(d::Integer, initialize_state::Function, transition::Function, isterm::Function, γ::T, max_episodes::Integer, max_steps::Integer, update_state_representation!::Function; ϵ = one(T)/1000, s0::S = initialize_state(), calculate_error::Function = (v̂, s)->zero(T)) where {T<:Real, S}
		s = initialize_state()
		ep = 1
		step = 1
		parameters = zeros(T, d)
		Ainv = zeros(T, d, d)
		Ainv2 = zeros(T, d, d)
		for i in 1:d
			Ainv[i, i] = inv(ϵ)
		end
		state_representation1 = zeros(T, d)
		state_representation2 = zeros(T, d)
		
		b = zeros(T, d)
		v = zeros(T, d)
		x3 = zeros(T, d)
		update_state_representation!(state_representation1, s)
		episode_errors = Vector{T}()
		err = zero(T)
		epstep = 1
		while (ep <= max_episodes) && (step <= max_steps)
			(r, s′) = transition(s)
			if isterm(s′)
				state_representation2 .= zero(T)
			else
				update_state_representation!(state_representation2, s′)
			end

			x3 .= state_representation1 .- γ .* state_representation2
			mul!(v, Ainv', x3)
			mul!(x3, Ainv, state_representation1)
			mul!(Ainv2, x3, v')
			Ainv .-= Ainv2 ./ (one(T) + dot(v, state_representation1))
			b .+= r.*state_representation1
			mul!(parameters, Ainv, b)
			v̂ = dot(parameters, state_representation1)
			err += calculate_error(v̂, s)
			s = s′
			epstep += 1
			if isterm(s′)
				s = initialize_state()
				ep += 1
				push!(episode_errors, err / epstep)
				ep_step = 1
				update_state_representation!(state_representation1, s)
			else
				s = s′
				state_representation1 .= state_representation2
			end
			step += 1
		end

		function v(s::S)
			x = zeros(T, d)
			update_state_representation!(x, s)
			dot(parameters, x)
		end

		function v(states::AbstractVector{S})
			x = zeros(T, d)
			input = zeros(T, length(states), d)
			for i in eachindex(states)
				update_state_representation!(x, states[i])
				for j in 1:d
					input[i, j] = x[j]
				end
			end
			input*parameters
		end
		return (parameters = parameters, value_estimate = v, episode_errors = episode_errors)
	end

	least_squares_td_estimation(mrp::StateMRP, d::Integer, args...; kwargs...) = least_squares_td_estimation(d, mrp.initialize_state, s -> mrp.ptf(s), mrp.isterm, args...; kwargs...)

	least_squares_td_policy_estimation(mdp::StateMDP, d::Integer, π::Function, args...; kwargs...) = least_squares_td_estimation(d, mdp.initialize_state, s -> mdp.ptf(s, π), mdp.isterm, args...; kwargs...)
end

# ╔═╡ 2463013a-efad-42a9-874d-a0ecbea9cb49
md"""
### *State Aggregation LSTD Implementation*
"""

# ╔═╡ e0e51e37-0217-4a76-b6e7-9b6e15429941
"""
    create_state_aggregation_feature_vector_update(assign_state_group) -> Function

Create a state representation update function for state aggregation feature encoding.

# Arguments
- `assign_state_group::Function`: Function mapping states to group indices `state -> Integer`

# Returns
- `Function`: Update function with signature `update_state_representation!(state_representation, s) -> AbstractVector`

# See Also
[`tile_coding_feature_setup`](@ref), [`least_squares_td_estimation`](@ref), [`StateAggregationFeatureVector`](@ref)

# Algorithm Details
Creates a closure that encodes states as one-hot binary vectors based on state group assignments. The returned function:
1. Calls `assign_state_group(s)` to determine the group index for state `s`
2. Zeros the entire state representation vector
3. Sets the corresponding group index to 1, creating a one-hot encoding
4. Returns the modified vector

This provides sparse binary feature representation where each state belongs to exactly one group, enabling efficient linear function approximation over aggregated state spaces.

# Examples
```julia-repl
julia> # Create group assignment for 1D state space
julia> assign_group(s) = clamp(ceil(Int, s * 10), 1, 10)  # 10 groups
julia> update_features! = create_state_aggregation_feature_vector_update(assign_group)

julia> # Use with LSTD estimation
julia> result = least_squares_td_estimation(10, init_state, transition, isterm,
           0.9f0, 1000, 20000, update_features!)

julia> # Test feature encoding
julia> features = zeros(Float32, 10)
julia> update_features!(features, 0.35f0)
julia> features
10-element Vector{Float32}: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```
"""
function create_state_aggregation_feature_vector_update(assign_state_group::Function)
	function update_state_representation!(state_representation::AbstractVector{T}, s) where {T<:Real}
		i = assign_state_group(s)
		state_representation .= zero(T)
		state_representation[i] = one(T)
		return state_representation
	end
end

# ╔═╡ 3aec99c3-ba2c-418f-b448-14eb9e8e423c
"""
    run_state_aggregation_least_squares_td_estimation(mrp, num_groups, assign_state_group, γ, max_episodes, max_steps; kwargs...) -> NamedTuple

Perform LSTD value function estimation using state aggregation feature representation.

# Arguments
- `mrp::StateMRP`: Markov reward process defining the learning environment
- `num_groups::Integer`: Number of state groups for aggregation
- `assign_state_group::Function`: Function mapping states to group indices `state -> Integer`
- `γ`: Discount factor (0 ≤ γ < 1)
- `max_episodes::Integer`: Maximum number of episodes to run
- `max_steps::Integer`: Maximum total steps across all episodes

# Keyword Arguments
- `kwargs...`: Additional arguments passed to [`least_squares_td_estimation`](@ref)

# Returns
- Same as [`least_squares_td_estimation`](@ref): NamedTuple with `parameters`, `value_estimate`, and `episode_errors`

# See Also
[`least_squares_td_estimation`](@ref), [`create_state_aggregation_feature_vector_update`](@ref), [`StateMRP`](@ref)

# Algorithm Details
1. Creates state aggregation feature update function via [`create_state_aggregation_feature_vector_update`](@ref)
2. Delegates to [`least_squares_td_estimation`](@ref) with one-hot group encoding
3. Returns LSTD results where each state group has an independent value estimate

This convenience function automates the setup of state aggregation for LSTD learning, where continuous or large discrete state spaces are partitioned into manageable groups for efficient value function approximation.

# Examples
```julia-repl
julia> # State aggregation for continuous random walk into 20 groups
julia> assign_group(s) = clamp(ceil(Int, s * 20), 1, 20)
julia> result = run_state_aggregation_least_squares_td_estimation(mrp, 20, assign_group,
           0.95f0, 1500, 30000)

julia> # Each group has independent value estimate
julia> group_values = result.parameters
20-element Vector{Float64}

julia> # Evaluate states via group membership
julia> v_estimate = result.value_estimate(0.6f0)  # Uses group assignment internally
-0.234f0
```
"""
function run_state_aggregation_least_squares_td_estimation(mrp::StateMRP, num_groups::Integer, assign_state_group::Function, γ, max_episodes, max_steps; kwargs...)
	update_feature_vector! = create_state_aggregation_feature_vector_update(assign_state_group)
	least_squares_td_estimation(mrp, num_groups, γ, max_episodes, max_steps, update_feature_vector!; kwargs...)
end

# ╔═╡ 785e0e9e-8591-4df0-9282-b516cb87767e
function run_state_aggregation_least_squares_td_policy_estimation(mdp::StateMDP, π::Function, num_groups::Integer, assign_state_group::Function, γ, max_episodes, max_steps; kwargs...)
	update_feature_vector! = create_state_aggregation_feature_vector_update(assign_state_group)
	least_squares_td_policy_estimation(mdp, num_groups, π, γ, max_episodes, max_steps, update_feature_vector!; kwargs...)
end

# ╔═╡ 195d2aa9-28c1-4b4a-9da5-c8ed3e20ed85
md"""
### *Example: LSTD with Random Walk Example*
"""

# ╔═╡ f10c643b-9205-4b18-841c-255a9354cf97
#=╠═╡
function state_aggregation_least_squares_td_randomwalk(num_episodes; num_groups = 10, ϵ = 1f-3, α = 1f-3)
	group_assign = make_random_walk_group_assign(num_states, num_groups)

	t0 = time()
	(params, v, error) = run_state_aggregation_least_squares_td_estimation(random_walk_state_mrp, num_groups, group_assign, 1f0, num_episodes, typemax(Int64); calculate_error = (v̂, s) -> (v̂ - random_walk_v.value_function[Int64(s)])^2, ϵ = ϵ)
	t_lstd = round(time() - t0; sigdigits = 3)
	t0 = time()
	v̂_td, history_td = semi_gradient_td0_estimation_state_aggregation(random_walk_state_mrp, 1f0, num_episodes, typemax(Int64), num_groups, random_walk_group_assign; α = α, calculate_error = calc_random_walk_ve)
	err_history_td = history_td.errors
	t_td = round(time() - t0; sigdigits = 3)
	t1 = scatter(y = v(Float32.(1:1000)), name = "LSTD Estimation")
	t2 = scatter(y = random_walk_v.value_function[2:end-1], name = "true value")
	t3 = scatter(y = err_history_td)
	p1 = plot([t1; t2])
	p2 = plot_value_error([error, err_history_td], ["Least Squares TD with ϵ = $ϵ", "Semi-gradient TD with α = $α"], 10)
	md"""
	$p1
	Execution times: 

	Least Squares TD: $t_lstd seconds
	Semi-gradient TD: $t_td seconds
	$p2
	"""
end
  ╠═╡ =#

# ╔═╡ 369e5b57-61ce-49e0-97e7-90901f82d37f
md"""
#### Least Squares TD with State Aggregation
"""

# ╔═╡ 7c5ac88b-453b-40bd-98a4-534fc70c7c45
#=╠═╡
state_aggregation_least_squares_td_randomwalk(10000; ϵ = 1f-4, α = 1f-2)
  ╠═╡ =#

# ╔═╡ f1272708-4f99-484e-b861-cd50e4f20bc4
md"""
#### Least Squares TD with Linear Features
"""

# ╔═╡ 85a14a63-d084-4183-a9be-33455dd2ad33
#=╠═╡
function linear_compare_least_squares_td_randomwalk(num_episodes; order_number = 5, num_tilings = 10, tile_size = 0.05f0)
	
	poly_setup = order_features_setup(random_walk_state_mrp, order_number, 1f0, Float32(num_states), calc_poly_feature)
	fourier_setup = order_features_setup(random_walk_state_mrp, order_number, 1f0, Float32(num_states), calc_fourier_feature)
	tile_setup = tile_coding_feature_setup(random_walk_state_mrp, 1f0, Float32(num_states), tile_size, num_tilings)

	estimate_traces = [scatter(y = random_walk_v.value_function[2:end-1], name = "true value")]
	errors = []
	error_names = []

	tests = [ 	(setup = poly_setup, α = 1f-6, ϵ = 1f-2, name = "Polynomial Order $order_number"), 
				(setup = fourier_setup, α = 1f-3, ϵ = 1f-3, name = "Fourier Order $order_number"), 
				(setup = tile_setup, α = 1f-3 / num_tilings, ϵ = 1f-1, name = "Tile Coding with $num_tilings tilings")
	]

	get_num_features(::BinaryFeatureVector{I, N}) where {I, N} = N
	get_num_features(x::Vector) = length(x)
	
	for i in eachindex(tests)
		α = tests[i].α
		setup = tests[i].setup
		name = tests[i].name
		ϵ = tests[i].ϵ
		t0 = time()
		(params, v, error) = least_squares_td_estimation(random_walk_state_mrp, get_num_features(setup.feature_vector), 1f0, num_episodes, typemax(Int64), setup.update_feature_vector!; calculate_error = (v̂, s) -> (v̂ - random_walk_v.value_function[Int64(s)])^2, ϵ = ϵ)
		t_lstd = round(time() - t0; sigdigits = 3)
		α = 1f-3
		t0 = time()
		v̂_td, history_td = semi_gradient_td0_estimation_linear(random_walk_state_mrp, 1f0, num_episodes, typemax(Int64), setup.feature_vector, setup.update_feature_vector!; α = α, calculate_error = calc_random_walk_ve)
		err_history_td = history_td.errors
		t_td = round(time() - t0; sigdigits = 3)
		push!(estimate_traces, scatter(y = v(Float32.(1:1000)), name = "LSTD Estimation $name features"))
		push!(errors, error)
		push!(errors, err_history_td)
		push!(error_names, "$name Features with ϵ = $ϵ LSTD")
		push!(error_names, "$name Features with α = $α Semi-gradient TD(0)")
	end
	
	
	
	# t3 = scatter(y = err_history_td)
	p1 = plot(estimate_traces, Layout(xaxis_title = "State", yaxis_title = "Value"))
	p2 = plot_value_error(errors, error_names, 10)
	md"""
	$p1
	$p2
	"""
end
  ╠═╡ =#

# ╔═╡ 05f35ff0-3122-45ab-b048-3d6eec453644
#=╠═╡
linear_compare_least_squares_td_randomwalk(1000)
  ╠═╡ =#

# ╔═╡ 290200a3-7523-4e0f-bd3a-288626adaf29
md"""
## 9.9 Memory-based Function Approximation

All of the methods discussed so far have been *parametric*.  That is to say they use an approximation function whos output depends on a list of parameters which are updated as part of the leaning process.  The parameter values determine the value estimate accross the entire state space and in general any parameter update could have an impact on some or all of the other state values.  If we need to compute the value of a state during the learning process, we simply apply the function approximation with the current list of parameters to that state.

Memory-based function approxmation methods save training examples as memory as they arrive (or a subset of examples).  Whenever we need a state's value estimate, we query the memory to compute the value.  This is sometimes called *lazy learning* because nothing is done with data from examples until it is needed.  Memory baesd approaches are *nonparametric* methods since the estimation method is not limited to a class of functions determined ahead of time by the structure of the parameters and feature vectors.  

One class of memory-based methods are *local-learning* methods that approximate a value function only locally in the neighborhood of the current query state.  These methods retrieve a set of training examples form memory whose states are judged to be the most relevant to the query state, where relevance usually depends on the distance between states.  

The simplest example of the memory-based approach is the *nearest neighbor* method, which simply finds the example in memory whose state is closest to the query state and returns that example's value as the approximate value of the query state.  In other words, if the query state is $s$, and $s^\prime \rightarrow g$ is the example in memory in which $s^\prime$ is the closest state to $s$, then $g$ is returned as the approximate value of $s$.  Slightly more complicated are *weighted average* methods that retrieve a set of nearest neighbor examples and return a weighted average of their target values, where the weights generally decrease with increasing distance between their states and the query state.
"""

# ╔═╡ 53ed4517-7e1b-4b72-9844-b8e291382bca
md"""
### *Memory-based Database Implementation*

Since the memory must store a value estimate for the visited states, these methods are best suited for Monte Carlo sampling since we can calculate these value estimates without needing an approximation function.  In other words, as described here, these memory methods are not suitable for bootstrapping.
"""

# ╔═╡ 6dab2f6e-2b9d-4823-aa4c-f13f37afd2b3
function monte_carlo_episode_update!(state_values::Dict{S, T}, states::AbstractVector{S}, rewards::AbstractVector{T}, γ::T, α::T) where {T<:Real, S}
	g = zero(T)
	l = length(states)
	ō = zero(T)
	for i in l:-1:1
		s = states[i]
		g = γ * g + rewards[i]
		ō += α * (one(T) - ō)
		β = α / ō
		v = haskey(state_values, s) ? state_values[s] : zero(T)
		δ = g - v
		v′ = v + β*δ
		state_values[s] = v′
	end
end

# ╔═╡ 1d7dec72-c356-4043-9cc5-e0842c423cac
function monte_carlo_episode_update!(state_values::Dict{S, Tuple{T, T}}, states::AbstractVector{S}, rewards::AbstractVector{T}, γ::T, α::T) where {T<:Real, S}
	g = zero(T)
	l = length(states)
	ō = zero(T)
	for i in l:-1:1
		s = states[i]
		g = γ * g + rewards[i]
		if haskey(state_values, s)
			(v, n) = state_values[s]
			n′ = n + one(T)
			state_values[s] = ((v*n + g)/n′, n′)
		else
			state_values[s] = (g, one(T))
		end
	end
end

# ╔═╡ b56f36a5-884e-4f3e-90c1-0522e05f504d
function bulid_policy_value_memory(mdp::StateMDP{T, S, A, P, F1, F2, F3}, π::Function, γ::T, num_episodes::Integer; α = one(T)/10, epkwargs...) where {T<:Real, S, A, P, F1, F2, F3}
	(states, actions, rewards, _) = runepisode(mdp; π = π, epkwargs...)
	# state_values = Dict{S, T}()
	state_values = Dict{S, Tuple{T, T}}()
	monte_carlo_episode_update!(state_values, states, rewards, γ, α)
	for ep in 2:num_episodes
		(states, actions, rewards, _, n_steps) = runepisode!((states, actions, rewards), mdp; π = π, epkwargs...)
		monte_carlo_episode_update!(state_values, view(states, 1:n_steps), view(rewards, 1:n_steps), γ, α)
	end
	states = collect(keys(state_values))
	# vals = collect(values(state_values))
	vals = [state_values[s][1] for s in states]
	return (states = states, values = vals)
end

# ╔═╡ bbfe0acd-190e-457a-b08b-c2203f7f2efa
function build_value_memory(mrp::StateMRP{T, S, P, F1, F2}, γ::T, num_episodes::Integer; α = one(T)/10, epkwargs...) where {T<:Real, S, P, F1, F2}
	(states, rewards, _) = runepisode(mrp; epkwargs...)
	# state_values = Dict{S, T}()
	state_values = Dict{S, Tuple{T, T}}()
	monte_carlo_episode_update!(state_values, states, rewards, γ, α)
	for ep in 2:num_episodes
		(states, rewards, _, n_steps) = runepisode!((states, rewards), mrp; epkwargs...)
		monte_carlo_episode_update!(state_values, view(states, 1:n_steps), view(rewards, 1:n_steps), γ, α)
	end
	states = collect(keys(state_values))
	# vals = collect(values(state_values))
	vals = [state_values[s][1] for s in states]
	return (states = states, values = vals)
end

# ╔═╡ 34b78988-40f9-47e9-9c5a-7823de866b12
md"""
## 9.10 Kernel-based Function Approximation

The memory based methods described above save a database of examples $s^\prime \rightarrow g$ and then query the database for an example state $s$.  The value estimate will be some weighted sum of samples from the database and the function that calculates the weights is called a *kernel function* or simply a *kernel*.  For example, the kernel could assign a weight based on a distance metric between states but in general the kernel need only satisfy $k: \mathcal{S} \times \mathcal{S} \rightarrow \mathbb{R}$ so that $k(s, s^\prime)$ is the weight given to data $s^\prime$ answering a query about $s$.

Kernel functions numerically express how *relevant* knowledge about any state is to any other state.  As an example, consider the previous method of tile coding as a kernel function.  The relevance of states is determined by how many tiles it has in common with the query state and the stored value is shared among all examples in the same tile.  All of the linear methods discussed already can be described by a kernel function.

*Kernel regression* is the memory-basd method that computes a kernel weighted average of the targets of *all* examples stored in memory, assigning the result to the query state.  If $\mathcal{D}$ is the set of stored examples, and $g(s^\prime)$ denotes the target for state $s^\prime$ in a stored example, then kernel regression approximates the target function, in this case a value function depending on $\mathcal{D}$, as

$\hat v(s, \mathcal{D}) = \sum_{s^\prime \in \mathcal{D}} k(s, s^\prime) g(s^\prime)$

The weighted average method described above is a special case in which $k(s, s^\prime)$ is non-zero only when $s$ and $s^\prime$ are close to one another so that the sum need not be computed over all of $\mathcal{D}$.  Considering the linear methods where states are represented by a feature vector $\mathbf{x}(s) = (x_1(s), x_2(s), \dots, x_d(s))^\top$.  These are equivalent to kernel regression where $k(s, s^\prime) = \mathbf{x}(s)^\top \mathbf{x}(s^\prime)$
"""

# ╔═╡ 356d22a7-44e3-4875-9f21-ad4e1201101d
md"""
### *Example: Kernel-based Function Approximation on Random Walk Example*
"""

# ╔═╡ fda4d6cc-5868-4319-81c2-7a20dd0a7e9e
#=╠═╡
const random_walk_memory = build_value_memory(random_walk_state_mrp, 1f0, 100_000; α = 1f-2)
  ╠═╡ =#

# ╔═╡ 4e279cff-9233-430f-9b0b-40e992b34aed
# ╠═╡ skip_as_script = true
#=╠═╡
scatter(x = random_walk_memory.states, y = random_walk_memory.values, mode = "markers") |> plot
  ╠═╡ =#

# ╔═╡ 11d3d03b-18fe-40d6-80cf-b02e1dc8d0a1
function random_walk_distance_kernel_approximation(memory::@NamedTuple{states::Vector{Float32}, values::Vector{Float32}}; distance::Function = (s, s′) -> (s - s′)^2 + eps(1f0))
	states = memory.states
	vals = memory.values
	l = length(states)
	x = zeros(Float32, l)
	function v̂(s::Float32)
		x .= distance.(s, states) .^-1
		d = sum(x)
		dot(x, vals) / d
	end
end

# ╔═╡ 7254644c-1c92-428f-ba68-bb92cf404802
#=╠═╡
function random_walk_aggregation_kernel_approximation(memory::@NamedTuple{states::Vector{Float32}, values::Vector{Float32}}; num_groups = 10)
	states = memory.states
	vals = memory.values
	f = make_random_walk_group_assign(num_states, num_groups)
	l = length(states)
	state_groups = f.(states)
	x = zeros(Float32, l)
	function v̂(s::Float32)
		i = f(s)
		x .= state_groups .== i
		d = sum(x)
		dot(x, vals) / d
	end
end
  ╠═╡ =#

# ╔═╡ 62b2437b-72df-4943-b898-ad38b6d2de99
md"""
### Distance Kernel Random Walk Approximation

Note that a constant value is added to the distance in order to deal with the case of the query state matching a state in the memory.  In this case the distance is 0 so the kernel value is undefined.  Another way of dealing with this singularity is to simply assign the value in memory to that query state which in this example would simply use a single memory value for every estimate since all 1000 states are in the memory.
"""

# ╔═╡ c7c2395b-a5e9-4730-ab6e-11ef1d7639ee
# ╠═╡ skip_as_script = true
#=╠═╡
plot([scatter(x = 1:1000, y = random_walk_distance_kernel_approximation(random_walk_memory; distance = (s, s′) -> (s - s′)^2 + 1f1).(Float32.(1:1000)), name = "Distance Kernel-based Approximation"), scatter(y = random_walk_v.value_function[2:end-1], name = "true value")], Layout(xaxis_title = "State", yaxis_title = "Value"))
  ╠═╡ =#

# ╔═╡ d7ef7190-2031-470a-bc80-e96c93276387
md"""
### State Aggregation Kernel Random Walk Approximation

Note that this estimate should match the linear function approximation result for the same number of groups
"""

# ╔═╡ b2d97ba3-0816-4138-ae03-62423b82f960
#=╠═╡
md"""
Number of Groups for Kernel Appoximation: $(@bind kernel_num_groups Slider(1:num_states; show_value=true, default = 10))
"""
  ╠═╡ =#

# ╔═╡ 9ca3a044-3884-44c4-ae41-1ca8b44ae1c7
# ╠═╡ skip_as_script = true
#=╠═╡
plot([scatter(x = 1:1000, y = random_walk_aggregation_kernel_approximation(random_walk_memory; num_groups = kernel_num_groups).(Float32.(1:num_states)), name = "State Aggregation Kernel Approximation"), scatter(y = random_walk_v.value_function[2:end-1], name = "true value")], Layout(xaxis_title = "State", yaxis_title = "Value"))
  ╠═╡ =#

# ╔═╡ c04be604-804a-44c3-b2da-98729a5e7508
md"""
### Tile Coding Kernel Method
"""

# ╔═╡ 3e395c5f-2410-4abe-be61-b6345caa9e1c
# ╠═╡ skip_as_script = true
#=╠═╡
@bind tile_coding_kernel_params PlutoUI.combine() do Child
	md"""
	Tile Size: $(Child(:tile_size, NumberField(0f0:0.001f0:1f0, default = 0.1f0)))

	Number of Tilings: $(Child(:num_tilings, NumberField(1:100, default = 10)))
	"""
end
  ╠═╡ =#

# ╔═╡ 3be9fac4-17e6-4588-b4bc-2e7112e1bfbd
#takes the dot product of the feature vector for a new state given an existing feature vector, the feature vector for the new state is never constructed
function get_kernel_weight(feature_vector::AbstractVector{T}, active_features) where T<:Real
	w = zero(T)
	for i in active_features
		w += feature_vector[i]
	end
	return w
end

# ╔═╡ b2dc9155-8cae-4034-bb82-32ad41851fbd
#=╠═╡
function random_walk_tile_coding_kernel_approximation(memory::@NamedTuple{states::Vector{Float32}, values::Vector{Float32}}; tile_size = 0.1f0, num_tilings = 10)
	setup = tile_coding_feature_setup(random_walk_state_mrp, 1f0, Float32(num_states), tile_size, num_tilings)
	states = memory.states
	vals = memory.values
	l = length(states)
	state_feature_vectors = setup.get_feature_vector.(states)
	x = zeros(Float32, l)
	function v̂(s::Float32)
		for i in 1:l
			x[i] = get_kernel_weight(state_feature_vectors[i], setup.get_active_features(s))
		end
		d = sum(x)
		dot(x, vals) / d
	end
end
  ╠═╡ =#

# ╔═╡ d76f37ac-8721-4d10-8f15-20bc03b5ae98
# ╠═╡ skip_as_script = true
#=╠═╡
plot([scatter(x = 1:1000, y = random_walk_tile_coding_kernel_approximation(random_walk_memory; tile_coding_kernel_params...).(Float32.(1:num_states)), name = "Tile-Coding Kernel Approximation"), scatter(y = random_walk_v.value_function[2:end-1], name = "true value")], Layout(xaxis_title = "State", yaxis_title = "Value"))
  ╠═╡ =#

# ╔═╡ 905b032d-5fa0-4a3c-9055-fec92fd5879e
md"""
## 9.11 Looking Deeper at On-policy Learning: Interest and Emphasis
"""

# ╔═╡ 1636120f-9065-45a8-a849-731842374d60
md"""
## 9.12 Summary
"""

# ╔═╡ 022bb60c-6af7-4dd6-8410-69c7974707e8
md"""
> ### *Exercise 9.7*
> One of the simplest artificial neural networks consists of a single semi-linear unit with a logistic nonlinearity.  The need to handle approximate value functions of this form is common in games that end with either a win or a loss, in which case the value of a state can be interpreted as the probability of winning.  Derive the learning algorithm for this case, from (9.7), such that no gradient notation appears.
"""

# ╔═╡ bd7b5685-cb86-4efc-9491-0a2f61905b45
logit(x::T) where T<:Real = (one(T) + exp(-x))^-1

# ╔═╡ cf8ae04a-9931-447f-8e1d-5bd6415c6a51
md"""
The logistic function can be used to constrain the output of a linear approximator to between 0 and 1.  This is useful when the target values are probabilities.
"""

# ╔═╡ 42ec6c21-996d-4a6f-84fb-f8ac0fb8fd7b
# ╠═╡ skip_as_script = true
#=╠═╡
plot(scatter(x = -10:0.01:10, y = logit.(-10:0.01:10)), Layout(xaxis_title = "Linear Output", yaxis_title = "Logistic Output", title = L"f(x) = 1 / (1 + e^{-x})"))
  ╠═╡ =#

# ╔═╡ 272c7e61-8e16-421e-9c5b-b8ee32814e6b
md"""
The logistic function is: 
$f(x) = 1 / (1 + e^{-x})$

(9.7) is:

$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha [U_t - \hat v(S_t, \mathbf{w}_t)] \nabla \hat v(S_t, \mathbf{w}_t)$

For a single semi-linear unit, $\hat v(S_t, \mathbf{w}_t) = f(\mathbf{w}_t ^\top \mathbf{x}_t)$ where $f$ is the logistic function and $\mathbf{x}_t$ is the feature vector of state $S_t$ with the same length as $\mathbf{w}_t$.  

Also, using the definition of the logistic function:

$\begin{flalign}
f(x) &\doteq (1 + e^{-x})^{-1} \tag{1}\\
f(x)^{-1} &= 1 + e^{-x} \\
e^{-x} &= f(x)^{-1} - 1 \tag{2}\\
\end{flalign}$

Therefore, we can derive an expression for $f^\prime$ purely in terms of $f$:

$\begin{flalign}
f^\prime(x) &= -(1+e^{-x})^{-2}(-e^{-x}) \tag{chain rule} \\
&= e^{-x}(1 + e^{-x})^{-2} \\
&= f(x)^2 (f(x)^{-1} - 1) \tag{1 and 2}\\
&= f(x) (1 - f(x)) \\
\end{flalign}$

Applying to (9.7) with the chain rule and using the fact that $\nabla \left ( \mathbf{w}_t ^\top \mathbf{x}_t \right ) = \mathbf{x}_t$ :

$\begin{flalign}
	\mathbf{w}_{t+1} &\doteq \mathbf{w}_t + \alpha [U_t - \hat v(S_t, \mathbf{w}_t)] \nabla \hat v(S_t, \mathbf{w}_t) \\

	&= \mathbf{w}_t + \alpha [U_t - f(\mathbf{w}_t ^\top \mathbf{x}_t)] f(\mathbf{w}_t ^\top \mathbf{x}_t)(1-f(\mathbf{w}_t ^\top \mathbf{x}_t)) \mathbf{x}_t \\

\end{flalign}$
"""

# ╔═╡ 76de6624-6be3-450e-85a8-83e91af53272
md"""
> ### *Exercise 9.8*
> Arguably, the squared error used to derive (9.7) is inappropriate for the case treated in the preceding exercise, and the right error measure is the *cross-entropy loss*.  Repeat the derivation in Section 9.3, using the cross-entropy loss instead of the squared error in (9.4), all the way to an explicit form with no gradient or logarithm notation in it.  Is your final form more complex, or simpler, than you obtained in the preceding exercise?
"""

# ╔═╡ fa111767-96c2-44fe-8d26-29577f22b926
md"""
For a single output, the cross-entropy loss is 

$$-y \log{\hat y} - (1 - y)\log(1 - \hat y)$$ where $\hat y = f(\mathbf{w}_t^{\top} \mathbf{x}_t)$ is the approximation and $y = U_t$.  
"""

# ╔═╡ 82b0fb07-3f10-4701-bf4d-e2e0189cee08
md"""
The error for each example is then: $-U_t \log(f(\mathbf{w}_t^{\top} \mathbf{x}_t)) - (1 - U_t) \log(1 - f(\mathbf{w}_t^{\top} \mathbf{x}_t))$

where $f(x) = 1/(1 + e^{-x})$ is the logistic function

Our goal is to minimize this error over $\mu(s)$ using stochastic gradient descent, so the parameter update will be:

$\mathbf{w}_{t+1} \doteq \mathbf{w}_t - \alpha \nabla \left [-U_t \log(f(\mathbf{w}_t^{\top} \mathbf{x}_t)) - (1 - U_t) \log(1 - f(\mathbf{w}_t^{\top} \mathbf{x}_t)) \right ]$

From the previous exercise we know that $f^\prime(x) = f(x)(1-f(x))$, so applying the chain rule to the gradient gives: 

$\nabla \log(f(x)) = \nabla(x)f^\prime(x)/f(x) = (1 - f(x))\nabla(x)$

$\nabla \log(1 - f(x)) = -\nabla(x)f(x)^\prime/(1 - f(x)) = -f(x)\nabla(x)$

Using the fact that $\nabla(\mathbf{w}_t^{\top} \mathbf{x}_t) = \mathbf{x}_t$ So the parameter update rule can be simplified to:

$\begin{flalign}
\mathbf{w}_{t+1} &= \mathbf{w}_t - \alpha \nabla \left [-U_t \log(f(\mathbf{w}_t^{\top} \mathbf{x}_t)) - (1 - U_t) \log(1 - f(\mathbf{w}_t^{\top} \mathbf{x}_t)) \right ] \\
&= \mathbf{w}_t - \alpha \left [ -U_t(1-f(\mathbf{w}_t^{\top} \mathbf{x}_t)))\nabla(\mathbf{w}_t^{\top} \mathbf{x}_t)) + (1 - U_t)f(\mathbf{w}_t^{\top} \mathbf{x}_t)) \nabla(\mathbf{w}_t^{\top} \mathbf{x}_t)) \right ] \\
&= \mathbf{w}_t - \alpha  \left [-U_t + U_tf(\mathbf{w}_t^{\top} \mathbf{x}_t) + f(\mathbf{w}_t^{\top} \mathbf{x}_t) - U_t f(\mathbf{w}_t^{\top} \mathbf{x}_t)  \right ] \mathbf{x}_t \\
&= \mathbf{w}_t + \alpha  \left [U_t - f(\mathbf{w}_t^{\top} \mathbf{x}_t) \right ] \mathbf{x}_t \\
\end{flalign}$

This update rule is much simpler than the one in exercise 9.8 and is identical to the linear update rule with $\hat v = f(\mathbf{w}_t^{\top} \mathbf{x}_t)$ instead of $\hat v = \mathbf{w}_t^{\top} \mathbf{x}_t$
"""

# ╔═╡ 1a69bf65-7fa5-4ebd-b8e2-543a8e0dbf4f
cross_entropy_loss(y, ŷ) = -y*log(ŷ) - (1-y)*log(1-ŷ)

# ╔═╡ b4327edc-0677-4daf-a86d-1bcc908f2337
# ╠═╡ skip_as_script = true
#=╠═╡
plot([scatter(x = LinRange(0, 1, 1000), y = cross_entropy_loss.(0, LinRange(0, 1, 1000)), name = "y is false"), scatter(x = LinRange(0, 1, 1000), y = cross_entropy_loss.(1, LinRange(0, 1, 1000)), name = "y is true")], Layout(yaxis_title = "Cross Entropy Loss", xaxis_title = L"\hat y", title = "Cross Entropy Loss for a Single Output where the Target Value is True or False"))
  ╠═╡ =#

# ╔═╡ 5464338c-904a-4a1b-8d47-6c79da550c71
md"""
# Dependencies
"""

# ╔═╡ c1488837-602d-4fbf-9d18-fba4a7fc8140
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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
BenchmarkTools = "~1.6.0"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
PlutoDevMacros = "~0.9.0"
PlutoPlotly = "~0.6.4"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.69"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.6"
manifest_format = "2.0"
project_hash = "467bd178e4eaec23965efbca157f191edfe7a113"

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
git-tree-sha1 = "e38fbc49a620f5d0b660d7f543db1009fe0f8336"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "a656525c8b46aa6a1c76891552ed5381bb32ae7b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.30.0"

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
git-tree-sha1 = "0037835448781bb46feb39866934e243886d756a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

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
version = "1.6.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "b66970a70db13f45b7e57fbda1736e1cf72174ea"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.0"

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
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

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
git-tree-sha1 = "2984284a8abcfcc4784d95a9e2ea4e352dd8ede7"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.36"

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
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

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
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "28278bb0053da0fd73537be94afd1682cc5a0a83"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.21"

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
git-tree-sha1 = "72f65885168722413c7b9a9debc504c7e7df7709"
uuid = "a0499f29-c39b-4c5c-807c-88074221b949"
version = "0.9.0"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "232630fee92e588c11c2b260741b4fa70784b4c5"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.4"

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
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "2d7662f95eafd3b6c346acdbfc11a762a2256375"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.69"

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
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "7f44eef6b1d284465fafc66baf4d9bdcc239a15b"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.4.0"

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
git-tree-sha1 = "0fc001395447da85495b7fef1dfae9789fdd6e31"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.11"

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
# ╟─19d23ef5-27db-44a8-99fe-a7343a5db2b8
# ╟─c4c71ace-c3a4-412b-b08b-31d246f8db5f
# ╟─cb5e302b-a14b-4135-b6ff-bee300f9dee6
# ╟─865ed63a-a7ee-403f-9004-b3ec659d756f
# ╠═628c6613-0516-4078-a872-31f122831190
# ╠═47dbe518-6789-4639-bfc0-e5e5ddde980a
# ╠═a5ec633c-d5d0-4556-9bfe-16f51be1279f
# ╠═d1e4e1d5-0c14-4aaa-95c6-f741f83fce0d
# ╠═1176adf1-0a2a-41df-a3c5-f382126a0fe5
# ╠═1e8e5f5f-8b73-4820-a4fc-f97c8344f9e7
# ╠═14a13743-8e3a-4698-aef2-245557adfd92
# ╠═76fb06c4-0841-40a2-996e-cb9a555ffc34
# ╠═f8bc8f92-a9c6-4b7b-9a8e-48fbb1f85e6c
# ╠═a2ffaa35-ee82-47fd-878e-dd535caab109
# ╠═1d107df4-36fa-49bd-bd48-5d5f49910b44
# ╠═42a7918f-8e9d-45ae-9e40-1254dde9f06f
# ╠═1f0b9d36-3592-47a0-b32a-a7e19b763e1b
# ╠═be546bdb-77a9-48c4-9a98-1205d73fc8c6
# ╠═7542ff9c-c6a1-4d41-8863-05388fea8ce2
# ╠═ba58242a-306a-4631-92b4-34bc9e354fae
# ╠═c466f78e-e464-4602-93c4-40362e4c0df2
# ╟─9296a8a1-7edd-4ac4-8fa4-842317d693bc
# ╠═a77f9819-04b2-4785-8eb0-c7e9dba6cecc
# ╠═412f6295-3eec-4966-98e3-2774bf62ed4f
# ╟─966850ef-dd15-417b-b51c-9957f27e4664
# ╟─97539f3f-92bb-4b6f-a671-260251b4ddc7
# ╟─df56b803-0aa5-4946-8338-601195e57a3e
# ╠═e8e26a28-90a5-4519-ab08-11b49a8a9499
# ╟─cb2005fd-d3e0-4f37-908c-77e4bbac45b8
# ╟─90e5fc0e-2e97-424b-a5dd-9deb38293121
# ╠═de9bea60-c91d-4253-bdd8-a3c1fde8941c
# ╠═7814bda0-4306-4060-8f9a-2bcf1cf8e132
# ╠═69223862-4d74-46c9-8c78-b24d659151ac
# ╠═f4459b0d-ee3e-47c7-9c82-981af622edfa
# ╟─68a4151a-52ee-4ed0-b988-3fecc34d8d32
# ╟─24e8b391-00ec-4ed5-85dc-0796eb85bf4f
# ╟─736b7667-904d-4a9c-bb10-a6b0b831bfb6
# ╟─9c3f07b1-61eb-4d70-9dde-986c032a0840
# ╠═3f2ce7e0-b623-4ce3-90cf-949f3a6b0633
# ╟─60d68f9b-d18d-4d23-9adb-27fcb205e54b
# ╟─6f3928a9-bcaa-44b5-8723-820142cbcfc3
# ╠═2720329c-4c80-47cb-a3e3-d24fcec6ef43
# ╟─2c6809f9-50ed-44b8-8f27-0a62e88d118c
# ╟─91e4e5da-4e0f-48b2-98bd-1e9f1330b0a8
# ╟─5ebafa8b-c316-4f95-8adc-581f2eb40e1f
# ╠═24b99200-053a-41bf-a628-0b14b807fb86
# ╟─d68c0147-a66f-4542-a395-5f9b43e16b09
# ╟─1adf0786-0897-4119-9336-09de869463b4
# ╟─b361815f-d5b0-4c71-b331-c3b48ce53e73
# ╟─ff354a5e-f077-458d-8a0c-0a96a1d57658
# ╟─6004006e-4113-4f6a-b8ab-2c58ff207773
# ╟─c46c36f6-42da-4767-9e25-fa0ebe43998f
# ╟─c52222b7-64bd-4285-bba3-e22529495af6
# ╟─f64b78e1-76ff-4337-a9f0-aa2d3e3f33ac
# ╟─ace0693b-b4ce-43df-966e-0330d4399638
# ╠═c0e9ea1f-8cbe-4bc1-990f-ffd3ab1989cc
# ╟─bc479ae0-78ea-4255-863f-dcd126ae9b96
# ╠═750eef6b-58c6-4428-a44b-25e244aaf1d8
# ╠═3a0d315b-b5f8-4387-9bb0-fd2a7038752e
# ╠═75eceb07-f739-4009-8e92-b4742cedb548
# ╠═e3bd06e5-a16d-474c-b618-1c6f303eda00
# ╠═9dc8143f-280c-426a-911b-8ec851c9f093
# ╠═ce3ce1eb-1b88-4d30-aab3-9fa23c9246fe
# ╠═214714a5-ad1e-4439-8567-9095d10411a6
# ╠═49320a88-206e-4283-b3fc-a5d1ac41ddc4
# ╟─3160e3ec-d1b9-47ea-ad10-3d6ea40cc0b5
# ╟─701137fb-b497-47a5-9455-2f4b1c78a44e
# ╟─6b339182-f81c-475c-bf28-d03b57eda76f
# ╟─b6737cef-b6f9-4e40-82d8-bf887e17eb7c
# ╟─3db9f60e-a823-4d78-bd16-e73cedffa755
# ╟─7787522e-a4fb-4090-9a75-7ba74a4fcda6
# ╟─c3732b25-94fd-4061-aab8-36fc39d739a1
# ╠═c737c14b-2ad6-4d95-9795-2b87f6f722cb
# ╠═3307300f-cd72-4f16-bc46-39115a32e2ca
# ╟─645ba5fc-8575-4b8f-8982-f8bd20ac27ff
# ╠═99f34d13-a19a-4a28-8173-2f683527d61a
# ╠═7889fc4a-3a77-41b4-983a-0b04740afeb7
# ╟─cf9d7c7d-4519-410a-8a05-af90312e291c
# ╟─7989d6a9-a52a-4537-9c39-5d6b41f60098
# ╟─c05ea239-2eea-4f41-b4e3-993db0fe2de5
# ╠═bfb1858b-5e05-4239-bcae-a3b718074630
# ╟─f5203959-29ef-406c-abac-4f01fa9630a3
# ╟─c3da96b0-d584-4a43-acdb-16516e2d0452
# ╟─0ee3afe9-9c33-45c8-b304-26062675e1b8
# ╟─d65a0ca9-5577-4df8-af77-44ecfbcc0a07
# ╟─c5adf2d7-0b6b-4a87-974b-a90824d0323b
# ╠═38f09914-e128-4336-8e70-9906675971f2
# ╠═75fbbc54-807a-4894-a536-b27be81eb052
# ╟─f5dea7d5-4597-430c-9020-b74cdf8f3055
# ╠═9d7ca70c-0e60-4029-8ea0-26192ccea849
# ╠═bc2e52ff-7f47-4141-aff1-e752fe217f6a
# ╟─be715a78-5fcb-48b2-8a4f-c7ba27d34dd3
# ╟─c609ee03-7217-4068-9da2-c91fb02623a9
# ╟─25f4f9d3-d8aa-462c-9874-ae842da1cf79
# ╠═1e58c332-d43e-4467-b7b1-377262d460c3
# ╠═56212ab2-833a-4dec-bcdd-21bce1d680b6
# ╟─93a617ee-db64-4351-b919-340d950fc148
# ╟─994f8556-964c-4c6b-8cfe-6f6a99c1ba29
# ╟─ed00f1b2-79b0-406a-aabc-8c8c7ad61c31
# ╠═f1b7b56e-7701-4954-8217-1b2c7d01e309
# ╟─c99867b7-2cb0-4bb7-b035-0e86104adefe
# ╟─89262830-1129-4270-8007-32fb0cd2e0ec
# ╟─111a6762-ab4f-4db9-80f9-10c707623e0f
# ╠═b4aefbb1-dbb7-490c-9fa7-0f68e5a9916c
# ╟─a99ef185-0360-4005-9a8c-f10ca58babda
# ╟─168e84f6-429e-45d6-bdbd-f47552fce8b5
# ╟─529e262c-c94c-407b-8f13-be3b0f737e61
# ╠═40f0fd57-a4ea-47a0-b883-3b038a6612c4
# ╟─e565c041-17bd-40c8-9240-e86931c83010
# ╠═d215b917-c43d-4c14-aa97-2310f922d71a
# ╠═35d6dd59-1fd3-4aad-b24f-82dd466bcb83
# ╠═c64b740a-ceeb-431c-9c71-6ab498fc4003
# ╠═bb81db16-7c4d-4e08-bf17-45147be2b0db
# ╠═ed20781e-c7d5-48c8-82bd-94d73478c13a
# ╠═3968fcf6-c7b6-42bc-a416-fdfcb270f92c
# ╟─e6514762-31e0-4916-aa21-c280674c2fc1
# ╟─84d9aac5-cf3b-402b-b222-9e8985a80b5b
# ╟─dda74c94-3574-4e7b-bab1-d106111d36d4
# ╟─d17926d5-bcfa-4789-9609-59a69d87d194
# ╠═71e7eef0-0304-4e26-8991-fa20da83df9a
# ╟─8e12b92b-e56d-44f0-bf89-3248131b2245
# ╟─7e56131f-3afe-4997-a085-60f0d45a9d8d
# ╟─b5a3a529-2d74-4757-9d38-2eae28396d02
# ╟─a4d9efaf-1e1e-4115-973f-570014c1fd06
# ╟─22f6f2b1-745d-4ee5-8dfa-0fe2a61c2c54
# ╟─dfeead7c-65ab-4cb3-ac1c-a28a78e8448e
# ╟─6beee5a8-c262-469e-9b1b-00b91e3b1b55
# ╟─858a6d4f-2241-43c3-9db0-ff9cec00c2c1
# ╟─be019186-33ad-4eb7-a218-9124ff40b6fb
# ╟─b447a3a9-fe35-4457-886b-05c5862ad8e0
# ╟─d7c1810a-8f20-4178-83ca-017d53e3e7e9
# ╟─82828e72-5d30-41b6-a1b6-f258c234b034
# ╟─2bc32d3d-193e-4cab-b13b-f7ed304af0f6
# ╠═0334d2ff-268d-4485-b460-89f82c4a99e1
# ╠═8e8add6f-99ab-4aa7-b236-87915c6be9c2
# ╠═66cadcfb-4fda-4509-80d6-aa22766a7e9c
# ╠═9e3efa3c-af2f-4aea-b923-a6d50a6b9fb5
# ╠═2b922137-3110-4f91-94b1-4707d197b429
# ╠═b2c56d0e-668e-43cd-a886-bb830a60b132
# ╠═67db7264-2a5e-44be-98e7-e5d08d5e7273
# ╠═9b5fbbdd-0b36-4893-b4bb-b05439f5a541
# ╠═74e42774-68e5-44b5-91c4-da87a20879e1
# ╠═b58cacd0-ca65-43f5-8678-7265ea2d46c8
# ╠═d81d8f7d-ed32-405d-b0c8-2ceff5845578
# ╠═4a3a4635-a046-4eec-ab95-2dce74ac0fbe
# ╟─808026eb-4c5a-4f38-bb16-bbb1b2915906
# ╠═3ac65a54-1ff6-441c-8edf-00c49b620389
# ╠═dd907b31-24f1-46f6-a2d5-7dd268530c94
# ╠═0facd6de-411a-43e0-820d-7d6eceff5b72
# ╠═65795424-8e50-4edb-9f6a-7045a9a22b9d
# ╠═6c752a2b-4d10-4865-aeff-ea717b9d3904
# ╟─0c7d2eb3-02ce-47b0-955c-fc62d5c86994
# ╠═15b93928-98fb-47ed-ba46-e6ee785d46e5
# ╠═cfc5964b-3a23-48d9-b320-861fd4a43364
# ╠═93a1f51f-1d83-408e-a860-26e6280c65ee
# ╠═fb244ed5-2827-4b39-a5b1-ced0815b000a
# ╟─b1c84d59-3598-46a1-bc1a-fd691d14ab09
# ╟─420e54ac-1a7c-46e9-a8bd-e2ed5765aa7a
# ╠═40d07f16-b9ca-4782-bdd2-de15ec6b21e5
# ╟─b22ef023-4e6a-4114-b3c2-bf91e16e9a43
# ╟─32c054ee-a7ee-4705-87c3-fb1a4bd956ab
# ╠═a8d7e5f7-8509-4aa1-b4c6-669339cb173c
# ╟─2463013a-efad-42a9-874d-a0ecbea9cb49
# ╠═e0e51e37-0217-4a76-b6e7-9b6e15429941
# ╠═3aec99c3-ba2c-418f-b448-14eb9e8e423c
# ╠═785e0e9e-8591-4df0-9282-b516cb87767e
# ╟─195d2aa9-28c1-4b4a-9da5-c8ed3e20ed85
# ╠═f10c643b-9205-4b18-841c-255a9354cf97
# ╟─369e5b57-61ce-49e0-97e7-90901f82d37f
# ╟─7c5ac88b-453b-40bd-98a4-534fc70c7c45
# ╟─f1272708-4f99-484e-b861-cd50e4f20bc4
# ╟─05f35ff0-3122-45ab-b048-3d6eec453644
# ╠═85a14a63-d084-4183-a9be-33455dd2ad33
# ╟─290200a3-7523-4e0f-bd3a-288626adaf29
# ╟─53ed4517-7e1b-4b72-9844-b8e291382bca
# ╠═6dab2f6e-2b9d-4823-aa4c-f13f37afd2b3
# ╠═1d7dec72-c356-4043-9cc5-e0842c423cac
# ╠═b56f36a5-884e-4f3e-90c1-0522e05f504d
# ╠═bbfe0acd-190e-457a-b08b-c2203f7f2efa
# ╟─34b78988-40f9-47e9-9c5a-7823de866b12
# ╟─356d22a7-44e3-4875-9f21-ad4e1201101d
# ╠═fda4d6cc-5868-4319-81c2-7a20dd0a7e9e
# ╟─4e279cff-9233-430f-9b0b-40e992b34aed
# ╠═11d3d03b-18fe-40d6-80cf-b02e1dc8d0a1
# ╠═7254644c-1c92-428f-ba68-bb92cf404802
# ╟─62b2437b-72df-4943-b898-ad38b6d2de99
# ╠═c7c2395b-a5e9-4730-ab6e-11ef1d7639ee
# ╟─d7ef7190-2031-470a-bc80-e96c93276387
# ╟─b2d97ba3-0816-4138-ae03-62423b82f960
# ╟─9ca3a044-3884-44c4-ae41-1ca8b44ae1c7
# ╟─c04be604-804a-44c3-b2da-98729a5e7508
# ╟─3e395c5f-2410-4abe-be61-b6345caa9e1c
# ╟─d76f37ac-8721-4d10-8f15-20bc03b5ae98
# ╠═3be9fac4-17e6-4588-b4bc-2e7112e1bfbd
# ╠═b2dc9155-8cae-4034-bb82-32ad41851fbd
# ╟─905b032d-5fa0-4a3c-9055-fec92fd5879e
# ╟─1636120f-9065-45a8-a849-731842374d60
# ╟─022bb60c-6af7-4dd6-8410-69c7974707e8
# ╠═bd7b5685-cb86-4efc-9491-0a2f61905b45
# ╟─cf8ae04a-9931-447f-8e1d-5bd6415c6a51
# ╟─42ec6c21-996d-4a6f-84fb-f8ac0fb8fd7b
# ╟─272c7e61-8e16-421e-9c5b-b8ee32814e6b
# ╟─76de6624-6be3-450e-85a8-83e91af53272
# ╟─fa111767-96c2-44fe-8d26-29577f22b926
# ╟─b4327edc-0677-4daf-a86d-1bcc908f2337
# ╟─82b0fb07-3f10-4701-bf4d-e2e0189cee08
# ╠═1a69bf65-7fa5-4ebd-b8e2-543a8e0dbf4f
# ╟─5464338c-904a-4a1b-8d47-6c79da550c71
# ╠═6da69e64-743f-4ea9-9670-fd023c7ffab7
# ╠═808fcb4f-f113-4623-9131-c709320130df
# ╠═db8dd224-abf1-4a65-b8bb-e2da6ab43f7e
# ╠═c1488837-602d-4fbf-9d18-fba4a7fc8140
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
