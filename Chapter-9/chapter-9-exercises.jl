### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 24a0d98e-b015-4ba1-9eb3-c274a9669fef
using Random, Statistics, PlutoPlotly

# ╔═╡ 2e02b042-aa06-4c24-ad1b-65458d7e1999
using PlutoUI

# ╔═╡ 61958d96-3dea-11ed-20d2-2f87ba57e036
md"""
# Chapter 9 On-policy Prediction with Approximation
## 9.1 Value-function Approximation
The method we use to approximate the true value function must be able to learn efficiently from incrementally acquired data.  Also the target values of training the function may be non stationary.  We will designate some approximation function for our value function as $\hat v(S, w)$ which is parametrized by some weights  that in general will be much smaller than in size to the true state space.

## 9.2 The Prediction Objective ($\overline {VE}$)
In tabular methods, the learned value can exactly equal the true objective and each state approximation is independent.  Neither of these are true for parametrized approximation.  We must specificy a state distribution $\mu(s) \geq 0, \sum_s{\mu(s)}=1$ that represents how much we care about the error in each state.  One natural objective function is the mean squared error weighted over this distribution.

$\overline{VE}(\boldsymbol{w}) \dot = \sum_{s \in S} \mu(s)[v_\pi(s) - \hat v(s, \boldsymbol{w})]^2$

Often $\mu(s)$ is taken to be the fraction of time spent in $s$.  In contiunuing tasks the on-policy distribution is the stationary distribution under $\pi$.  In episodic tasks one must account for the probability of starting an episode in a particular state and the probability of transitioning to that state during an episode.  The state distribution will need to depend on that function typically denoted $\eta(s)$.

An ideal goal for optimizing $\overline {VE}$ is to find a *global optimum* for the weight vector such that $\overline {VE}(\boldsymbol{w}^*) \leq \overline {VE}(\boldsymbol{w})$ for all posible weights.  Typically this isn't possible but we can find a *local optimum* but even this objective is not guaranteed for many approximation methods.  In this chapter we will focus on approximation methods based on linear gradient-descent methods to we have easily find an optimum.

## 9.3 Stochastic-gradient and Semi-gradient Methods
We will assume a weight vector with a fixed number of components $\boldsymbol{w} \dot = (w_1, w_2, \dots, w_d)$ and a differentiable value function $\hat v(s, \boldsymbol{w})$ that exists for all states.  We will update weights at each of a series of discrete time steps so we can denote $\boldsymbol{w}_t$ as the weight vector at each step.  Assume at each step we observe a state and its true value under the policy.  We assume that states appear in the same distribution $\mu$ over which we are trying to optimize the prediction objective.  Under these assumptions we can try to minimize the error observed on each example using *Stochastic gradient-descent* (SGD) by adjusting the weight vector a small amount after each observation:

$\boldsymbol{w}_{t+1} \dot = \boldsymbol{w}_t - \frac{1}{2} \alpha \nabla [v_\pi(S_t) - \hat v(S_t, \boldsymbol{w}_t)]^2$
$= \boldsymbol{w}_t + \alpha[v_\pi(S_t) - \hat v(S_t, \boldsymbol{w}_t)]\nabla\hat v(S_t, \boldsymbol{w}_t)$

where $\alpha$ is a learning rate.  In general this method will only converge to the weight vector that minimizes the error objective if $\alpha$ is sufficiently small and decreases over time.

If we do not receive the true value function at each example but rather a bootstrap approxmiation or a noise corrupted version, we can use the same formula and simply replace $v_\pi(S_t)$ with $U_t$.  As long as $U_t$ is an *unbiased* estimate for each example then the weights are still guaranteed to converge to a local optimum stochastically.  One example of an unbiased estimate would be a monte carlo sample of the discounted future return.

If we use a bootstrapped estimate of the value, then the estimate depends on the current weight vector and will no longer be *unbiased* which requires that the update target be independent of $\boldsymbol{w}_t$.  A method using bootstrapping with function approximation would be considered a *semi-gradient method* because it violates part of the convergence assumptions.  In the case of a linear function, however, they can still converge reliably.  One typical example of this is semi-gradient TD(0) learning which uses the value estimate target of $U_t \dot = R_{t+1} + \gamma \hat v(S_{t+1}, \boldsymbol{w})$.  In this case the update step for the weight vector is as follows:

$\boldsymbol{w}_{t+1} = \boldsymbol{w}_t + \alpha[R_t + \gamma \hat v(S_{t+1}, \boldsymbol{w}_t) - \hat v(S, \boldsymbol{w}_t)] \nabla \hat v(S_t, \boldsymbol{w}_t)$

*State aggregation* is a simple form of generalizing function approximation in which states are grouped together, with one estimated value (one component of the weight vector **w**) for each group.  The value of a state is estimated as its group's component, and when the state is updated, that component alone is updated.  State aggregation is a special case of SGD in which the gradient, $\nabla \hat v(S_t, \boldsymbol{w}_t)$, is 1 for the observed state's component and 0 for others.
"""

# ╔═╡ cb2005fd-d3e0-4f37-908c-77e4bbac45b8
md"""
### Example 9.1: State Aggregation on the 1000-state Random Walk
"""

# ╔═╡ c3387396-4490-4225-88bf-db6b81f2addc
function randomwalk_step(states::Vector{Int64}, s::Int64)
	l = length(states)
	#randomly choose left or right
	newrange = if rand() < 0.5
		s-100:s-1
	else
		s+1:s+100
	end
	snew = rand(newrange)

	#terminating to the left is a reward of -1.0, the right is 1.0 and all other transitions are 0.0
	snew < 1 && return (0, -1.0)
	snew > l && return (l+1, 1.0)
	return (snew, 0.0)
end

# ╔═╡ 27ce81c9-4223-4adb-b29c-60ca272a0195
function gradient_monte_carlo_state_aggregate(s0, states, run_episode; α = 2e-5, nweights = 10, w0 = 0.0, maxepisodes = 1000)
	w = fill(w0, nweights)
	nstates = length(states)
	getweight(s) = ceil(Int64, nweights*s/nstates)
	v̂(s, w) = w[getweight(s)]
	function v̂_grad(s, w)
		out = zeros(length(w))
		out[getweight(s)] = 1.0
		return out
	end

	totaldist = zeros(1000)
	for i in 1:maxepisodes
		(traj, counts) = run_episode(s0)
		g = traj[end][2]
		for (s, r) in traj
			w .+= α*(g - v̂(s, w)).*v̂_grad(s, w)
		end
		totaldist .+= counts
	end
	totaldist = totaldist ./ sum(totaldist)
	return s -> v̂(s, w), totaldist
end

# ╔═╡ c0c07d6d-4d9b-4cb2-b298-dfe2e4885e76
function example_9_1()
	nstates = 1000
	states = collect(1:nstates)
	s0 = 500
	isterm(s) = (s == 0) || (s == nstates + 1)
	function run_episode(s, traj = [], counts = zeros(nstates))
		isterm(s) && return traj, counts
		(snew, reward) = randomwalk_step(states, s)
		push!(traj, (s, reward))
		counts[s] += 1.0
		run_episode(snew, traj, counts)
	end

	(v̂, totaldist) = gradient_monte_carlo_state_aggregate(s0, states, run_episode, maxepisodes = 100_000)
	
	[plot(totaldist), plot(v̂.(states))]
end

# ╔═╡ 4ca754f6-6646-410e-9172-e16db829fa09
example_9_1()

# ╔═╡ 3160e3ec-d1b9-47ea-ad10-3d6ea40cc0b5
md"""
## 9.4 Linear Methods
"""

# ╔═╡ 6c6c0ef4-0e68-4f50-8c3a-76ed3afb2d20
md"""
$\hat v(s, \mathbf{w})\dot = \mathbf{w}^\top \mathbf{x}(x) \dot =\sum_{i=1}^d w_i x_i(s)$ 

The vector $\mathbf{x}(s)$ is called a *feature vector* representing state x which is the same length as the number of parameters contained in $\mathbf{w}$.  For linear methods, features are *basis functions* because they form a linear basis for the set of approximate functions.
"""

# ╔═╡ f5203959-29ef-406c-abac-4f01fa9630a3
md"""
> *Exercise 9.1* Show that tabular methods such as presented in Part I of this book are a special case of linear function approximation.  What would the feature vectors be?

The tabular methods in Part I store a single value estimate for each state.  Such a lookup can be thought of as a function that is simply a map from each state to a value.  The feature vectors for this function would be orthanormal basis vectors of dimension matching the number of states, thus state 1 would be represented by the feature vector [1, 0, 0, ...], state 2 by [0, 1, 0, 0, ...] and so on.  The parameters would simply be the value estimate for each state.
"""

# ╔═╡ c3da96b0-d584-4a43-acdb-16516e2d0452
md"""
## 9.5 Feature Construction for Linear Methods
"""

# ╔═╡ 0ee3afe9-9c33-45c8-b304-26062675e1b8
md"""
### 9.5.1 Polynomials
"""

# ╔═╡ d65a0ca9-5577-4df8-af77-44ecfbcc0a07
md"""
> *Exercise 9.2* Why does (9.17) define $(n+1)^k$ distinct features for dimension $k$?
n represents the highest power to take for each individual dimension of the state and we consider powers from 0 up to n for each dimension.  If we list the exponent per dimension as a tuple, we have for n = 1, k = 2: (0, 0), (0, 1), (1, 0), (1, 1).
For n = 1, k = 3: (0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1).  This pattern consists of tuples of length k which can be formed by selecting from n + 1 choices of exponent.  The number of resulting tuples is $(n+1)^k$
"""

# ╔═╡ c5adf2d7-0b6b-4a87-974b-a90824d0323b
md"""
>*Exercise 9.3* What $n$ and $c_{i, j}$ produce the feature vectors $\mathbf{x}(s)=(1, s_1, s_2, s_1s_2, s_1^2, s_2^2, s_1s_2^2, s_1^2s_2, s_1^2s_2^2)^\top$

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

# ╔═╡ f5501489-46b8-4e5e-aa4f-427d8bc7a9b9
md"""
### 9.5.2 Fourier Basis
### 9.5.3 Coarse Coding
### 9.5.4 Tile Coding

> *Exercise 9.4* Suppose we believe that one of two state dimensions is more likely to have an effect on the value function than is the other, that generalization should be primarily across this dimension rather than along it.  What kind of tilings could be used to take advantage of this prior knowledge?

We could use striped tilings such that each stripe is the width of several of the important dimension but completely covers the entire space of the other dimension.  That way states that have the same value of the important dimension would be treated similarly regardless of their value in the other dimension and the overlap in the direction of the first dimension would allow some generalization if those states are close to each other along that dimension.
"""

# ╔═╡ dfeead7c-65ab-4cb3-ac1c-a28a78e8448e
md"""
### 9.5.5 Radial Basis Functions
Requires much more computational complexity to tile coding without much advantage.  Also more fine tuning is required.
"""

# ╔═╡ 6beee5a8-c262-469e-9b1b-00b91e3b1b55
md"""
## 9.6 Selecting Step-Size Parameters Manually
"""

# ╔═╡ 858a6d4f-2241-43c3-9db0-ff9cec00c2c1
md"""
> *Exercise 9.5* Suppose you are using tile coding to transform a seven-dimensional continuous state space into binary feature vectors to estimate a state value function $\hat v(s,\mathbf{w}) \approx v_\pi(s)$.  You believe that the dimensions do not interact strongly, so you decide to use eight tilings of each dimension separately (stripe tilings), for $7 \times 8 = 56$ tilings. In addition, in case there are some pairwise interactions between the dimensions, you also take all ${7\choose2} = 21$ pairs of dimensions and tile each pair conjunctively with rectangular tiles. You make two tilings for each pair of dimensions, making a grand total of $21 \times 2 + 56 = 98$ tilings.  Given these feature vectors, you suspect that you still have to average out some noise, so you decide that you want learning to be gradual, taking about 10 presentations with the same feature vector before learning nears its asymptote. What step-size parameter should you use? Why?

Each tiling will contribute one non-zero element to the feature vector.  With 98 tilings, we have 98 one values in each feature vector so the inner product in equation (9.19) would be $\mathbb{E}\left[\sum_{i=1}^{98} x_i^2 \right]=98$ so $\alpha=\frac{1}{10 \times 98}=\frac{1}{980} \approx 0.001$ 
	"""

# ╔═╡ be019186-33ad-4eb7-a218-9124ff40b6fb
md"""
> *Exercise 9.6* If $\tau=1$ and $\mathbf{x}(S_t)^\top \mathbf{x}(S_t) = \mathbb{E} [\mathbf{x}^\top \mathbf{x}]$, prove that (9.19) together with (9.7) and linear function approximation results in the error being reduced to zero in one update.
"""

# ╔═╡ 3c91e23b-dff8-4db3-b209-0c6b7f00faba
PlutoUI.TableOfContents(indent=true)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
PlutoPlotly = "~0.3.4"
PlutoUI = "~0.7.48"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "ca4f411998e3e99e4a3df66f98803271c5068019"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "5856d3031cdb1f3b2b6340dfdc66b6d9a149a374"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.2.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

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

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "3d5bf43e3e8b412656404ed9466f1dcbf7c50269"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Dates", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "PlotlyBase", "PlutoUI", "Reexport"]
git-tree-sha1 = "b470931aa2a8112c8b08e66ea096c6c62c60571e"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.3.4"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "efc140104e6d0ae3e7e30d56c98c4a927154d684"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.48"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
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
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

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
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─61958d96-3dea-11ed-20d2-2f87ba57e036
# ╟─cb2005fd-d3e0-4f37-908c-77e4bbac45b8
# ╠═24a0d98e-b015-4ba1-9eb3-c274a9669fef
# ╠═c3387396-4490-4225-88bf-db6b81f2addc
# ╠═27ce81c9-4223-4adb-b29c-60ca272a0195
# ╠═c0c07d6d-4d9b-4cb2-b298-dfe2e4885e76
# ╠═4ca754f6-6646-410e-9172-e16db829fa09
# ╟─3160e3ec-d1b9-47ea-ad10-3d6ea40cc0b5
# ╟─6c6c0ef4-0e68-4f50-8c3a-76ed3afb2d20
# ╟─f5203959-29ef-406c-abac-4f01fa9630a3
# ╟─c3da96b0-d584-4a43-acdb-16516e2d0452
# ╟─0ee3afe9-9c33-45c8-b304-26062675e1b8
# ╟─d65a0ca9-5577-4df8-af77-44ecfbcc0a07
# ╟─c5adf2d7-0b6b-4a87-974b-a90824d0323b
# ╟─f5501489-46b8-4e5e-aa4f-427d8bc7a9b9
# ╟─dfeead7c-65ab-4cb3-ac1c-a28a78e8448e
# ╟─6beee5a8-c262-469e-9b1b-00b91e3b1b55
# ╟─858a6d4f-2241-43c3-9db0-ff9cec00c2c1
# ╟─be019186-33ad-4eb7-a218-9124ff40b6fb
# ╟─2e02b042-aa06-4c24-ad1b-65458d7e1999
# ╟─3c91e23b-dff8-4db3-b209-0c6b7f00faba
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
