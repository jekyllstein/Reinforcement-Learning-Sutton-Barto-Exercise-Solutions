### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# ╔═╡ e681ab8e-5106-44e1-8ab0-3487aa876755
using PlutoUI

# ╔═╡ f229a1ef-76f7-4510-ae33-f54dd988636b
md"""
# Reinforcement Learning Methods

- Exploit Markov property of system to learn value function, policy function or both
- Use information related to specific states in the system
- Attempt to find a global solution with optimal behavior for every state
- Requires training ahead of time using environment definition or model
"""

# ╔═╡ 8ae1e4ae-fa8f-432a-9c8b-76468b7480d4
md"""
## Goals and Rewards

Our objective in *solving* and MDP is to maximize the expected value of what is called the *discounted future return*.  

$\begin{flalign}
G_t & \doteq \sum_{k=0}^\infty \gamma^k R_{t+k+1} \text{ or } \sum_{k = t+1} ^ T \gamma^{k-t-1}R_k \\
&= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \\
&= R_{t+1} + \gamma \left [ R_{t+2} + \gamma R_{t+3} + \cdots \right ] \\
&= R_{t+1} + \gamma G_{t+1}
\end{flalign}$

where $0 \lt \gamma \le 1$ in general and $0 \lt \gamma \lt 1$ for continuing tasks that do not have a terminal state.

## Policy Value Functions

Summarize information about $G_t$ for every state

$\begin{flalign}
v_\pi(s) &\doteq \mathbb{E}_\pi [G_t \mid S_t = s] \\
&= \sum_a \pi(a \vert s) \sum_{s^\prime, r} p(s^\prime, r \vert s, a) [r + \gamma v_\pi(s^\prime)] \\

q_\pi(s, a) &\doteq \mathbb{E}_\pi[G_t \mid S_t=s,A_t=a] \\
& = \sum_{s^\prime, r} p(s^\prime, r \vert s, a) [r + \gamma v_\pi(s^\prime)] \\
& = \sum_{s^\prime, r} p(s^\prime, r \vert s, a) [r + \gamma\sum_{a^\prime} \pi(a^\prime \vert s^\prime) q_\pi(s^\prime, a^\prime)] \\
\end{flalign}$
"""

# ╔═╡ aab00166-e104-4381-b601-1c5c6bb7e7db
md"""
## Optimal Policies and Value Functions

Every MDP has a unique optimal value function whose values are greater than or equal to every other value function at every state or state-action pair: 

$v_*(s) \geq v_\pi(s) \: \forall s, \pi$  

$q_*(s, a) \geq q_\pi(s, a) \: \forall s, a, \pi$
"""

# ╔═╡ d4cccc58-41c6-46f7-ba74-7bfe9560b705
md"""
$\begin{flalign}
v_*(s) &\doteq \max_\pi v_\pi(s) \: \forall \: s \in \mathcal{S} \\
&= \max_{a \in \mathcal{A}(s)} \sum_{s^\prime, r} p(s^\prime, r \vert s, a) \left [ r + γ v_* (s^\prime) \right ] \quad \forall s \in \mathcal{S} \\
q_*(s, a) &\doteq \max_\pi q_\pi(s, a) \: \forall \: s \in \mathcal{S} \text{ and } a \in \mathcal{A}(s) \\
&= \sum_{s^\prime, r} p(s^\prime, r \vert s, a) \left [ r + γ v_* (s^\prime) \right ] \\
&= \sum_{s^\prime, r} p(s^\prime, r \vert s, a) \left [ r + γ \max_{a^\prime} q_*(s^\prime, a^\prime) \right ] \\
\end{flalign}$
"""

# ╔═╡ 3457196c-bb39-4590-af4d-a13a6205a0f0
md"""
## Policy Improvement

- Given two policies $\pi(s)$ and $\pi^\prime(s)$ and the value functions for $\pi(s)$

- If $\pi^\prime(s)$ has the following property then the *policy improvement theorem* proves the following:

$q_\pi(s, \pi^\prime(s)) \geq v_\pi(s) \implies v_{\pi^\prime}(s) \geq v_\pi(s) \: \forall \: s \in \mathcal{S}$

- We can get a $\pi^\prime$ using only the value functions of $\pi(s)$:

$\begin{flalign}
\pi^\prime(s) &\doteq \mathrm{argmax}_a q_\pi(s, a) \\
& = \mathrm{argmax}_a \mathbb{E} [R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t = a] \\
& = \mathrm{argmax}_a \sum_{s^\prime, r} p(s^\prime, r \vert s, a) [r + \gamma v_\pi(s^\prime)] \\
\end{flalign}$

- This policy is known as the *greedy* policy with respect to the value function

- If the greedy policy is identical to itself, then it is guaranteed to be the optimal policy with the optimal value function: 

$\begin{flalign}
π_*(s) &= \mathrm{argmax}_a \sum_{s^\prime, r} p(s^\prime, r \vert s, a) [r + \gamma v_*(s^\prime)] \\
&= \mathrm{argmax}_a q_*(s, a)
\end{flalign}$
"""

# ╔═╡ 38671154-91fe-11f0-01c6-451cf9c6062d
md"""
## Tabular Methods

- Requires finite, enumerable state list
- Builds value function as a list of values for each state or state/action pair
- Uses policy improvement on learned values or directly calculates optimal value function

### Exact Solution Methods
- Require complete knowledge of environment (all transition probabilities)
- Value iteration uses fixed point iteration on Bellman optimality equation
- Policy iteration uses a sequence of policies/value functions and performs policy improvement on each value function
- Sufficient to only calculate state value function $v_\pi(s)$ and we can form the optimal policy with $v_*(s)$

### Estimation Methods
- Requires sampling transitions from environment to collect data
- Data is averaged to form estimates: $Q(s, a) \approx q_\pi(s, a)$
- Policy improvement is done on approximate values until convergence
- Note that we can always perform policy improvement with $q_\pi(s, a)$ and $q_*(s, a)$ can form the optimal policy
"""

# ╔═╡ 293c909f-34e9-43b2-a2b3-5d37b6f3f5aa
md"""
## Approximation Methods
- Necessary when state space is too large or infinite
- At minimum requires a mapping from states to feature vectors: $f(s) \rightarrow \mathbf{x}(s)$
- Uses approximation functions with parameters $\boldsymbol{\theta}$ that operate on the feature vectors
- Samples transitions from the environment to update parameters and improve function approximation
"""

# ╔═╡ 4af12e7a-1e21-4d3d-a42d-1ff4b940df3e
md"""
### Value Function Approximation

- Attempts to learn approximation of exact value function:

$\hat v(\mathbf{x}(s), \boldsymbol{\theta}) \approx v(s) \: \text{ or } \: \hat q(\mathbf{x}(s), a, \boldsymbol{\theta}) \approx q(s, a)$

- Must use value error objective which prioritizes states based on visit frequency to derive gradient updates to parameters:

$\overline{\text{VE}}(\mathbf{w}) \doteq \sum_{s \in S} \mu(s)[v_\pi(s) - \hat v(s, \mathbf{w})]^2$

- Updates function parameters with stochastic gradient updates: 

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha[v_\pi(S_t) - \hat v(S_t, \mathbf{w}_t)]\nabla\hat v(S_t, \mathbf{w}_t)$$

- Since $v_\pi$ is not known, must use an unbiased estimate from the environment or bootstrapping (violates gradient assumption)

- Uses policy improvement on approximate value function (note with with approximation the policy improvement theorem does not apply so this process is not guaranteed)
  - General case: $\pi(s) = \mathrm{argmax}_{a} \hat q(s, a)$
  - Known transition distribution: $\pi(s) = \mathrm{argmax}_{a} \sum_{s^\prime, r} p(r, s^\prime \vert s, a) r + \gamma \hat v(s^\prime)$
  - Take actions according to $\epsilon$-greedy policy and update value function with gradient as samples are collected
  - With approximation the policy improvement theorem does not apply so performance may not improve
- The feature vector $\mathbf{x}(s)$ will have some fixed length $n$ to be used in function approximation below

#### Linear Approximation
- State value function 
  -  $\mathbf{w}$ is a vector of length $n$ 
  - Output is a single state value
  -  $\hat v(s, \mathbf{w}) = \mathbf{x}(s) \cdot \mathbf{w}$
  -  $\nabla \hat v(s, \mathbf{w}) = \mathbf{x}(s)$
- State action value function
  -  $\mathbf{w}$ is a matrix with $n$ rows and $m$ columns where $m$ is the number of MDP actions
  - Output is a vector of $m$ action values
  -  $\hat q(s, \mathbf{w}) = \mathbf{w} ^ \top \mathbf{x}(s)$
  -  $\nabla \hat q(s, i_a, \mathbf{w}) = \begin{cases} \mathbf{x}(s); &\text{column = } i_a\\ \mathbf{0}; &\text{else} \end{cases}$

#### Non-linear Approximation
- Define architecture that can accomodate $n$ length vector input and output either a single value for $\hat v$ or $m$ values for $\hat q$
- Use backprop to compute gradient with respect to an output index (single value for $\hat v$ or action index for $\hat q$)
- Gradient will have same shape as parameters and will update parameters elementwise with learning rate

#### Gradient Monte Carlo Methods
- Uses sample return: $\mathbb{E}_\pi [ G_t ] = v_\pi(S_t) \implies \mathbf{w}_{t+1} = \mathbf{w}_t + \alpha[G_t - \hat v(S_t, \mathbf{w}_t)]\nabla\hat v(S_t, \mathbf{w}_t)$
- Converges to minimum value error solution
- Updates parameters after episode completion 
- Not suitable for continuing tasks (no terminal state)

#### Semi-gradient Methods
- Uses bootstrap estimate: $\mathbb{E}_\pi [ \sum_{i = 1}^ n \gamma ^{i-1} R_{t+i-1} + \gamma ^n \hat v(S_{t+1+n}) ] \approx v_\pi(S_t) \implies \mathbf{w}_{t+1} = \mathbf{w}_t + \alpha[R_t + \gamma \hat v(S_{t+1}) - \hat v(S_t, \mathbf{w}_t)]\nabla\hat v(S_t, \mathbf{w}_t)$ 
- Target value also depends on parameters so not a true gradient method
- Converges to TD fixed point in on policy case (projected Bellman error is zero)
- Risk of diverging values with off-policy sampling
- Updates parameters after every step so suitable for both continuing and episodic tasks
"""

# ╔═╡ 3a1b6e8c-89ad-4384-a27e-10791aa50d52
md"""
### Policy Gradient Methods

- Attempts to learn policy function directly: $\pi(a \vert s, \boldsymbol{\theta})$

- Policy gradient theorem allows optimization of policy parameters to maximize policy performance metric $J(\boldsymbol{\theta})$: 

$J(\boldsymbol{\theta}) = v_{\pi_\boldsymbol{\theta}}(s_0) \implies \nabla J(\boldsymbol{\theta}) \propto \mathbb{E}_\pi \left [ \gamma^t q_\pi (S_t, A_t) \frac{\nabla \pi(A_t|S_t, \boldsymbol{\theta})}{\pi(A_t|S_t, \boldsymbol{\theta})} \right ] \tag{where q is the discounted value function}$

$J(\boldsymbol{\theta}) = r(\pi) \implies \nabla J(\boldsymbol{\theta}) = \mathbb{E}_\pi \left [\frac{\nabla \pi(A_t \vert S_t, \boldsymbol{\theta})}{\pi(A_t \vert S_t, \boldsymbol{\theta})} q_\pi(S_t, A_t) \right ] \tag{where q is the differential value function}$

- Since $q_\pi$ is unknown, must replace with a sample from the environment

- No need to manage exploration with ϵ parameter or deal with off-policy learning

- Policy function maybe easier to learn than value function

- Optimal policy for a given feature vector might be stochastic which cannot be approximated by value methods

#### Reinforce Monte Carlo

- Only suitable for episodic tasks (parameter updates happen after episode is complete)
- Uses $G_t$ as a sample of $q_\pi(s, a)$ for each step in episode
- Gradient updates have high variance if a baseline value is not used

#### Actor Critic
- Suitable for episodic or continuing tasks (parameter updates happen after each step)
- Requires value function approximation $\hat v(s, \mathbf{w})$ which must also be trained with semi-gradient methods above
- Uses $R_t + \gamma \hat v(S_{t}, \mathbf{w})$ to sample $q_\pi(s, a)$
"""

# ╔═╡ c11cb109-547a-462b-8e13-9aed74accf36
md"""
### Comparison
"""

# ╔═╡ 2df524e6-7a6e-4dd4-bded-601b9c249779
md"""
|Element|Policy Gradient | Value Function Approximation|
|---|---|---|
|Feature Vector Mapping: ``\mathbf{x}(s)``| Required | Required |
|Target Value: ``\mathbb{E}_\pi[G_t \vert S_t, (A_t)]``|Required|Required|
|Value Function: ``\hat v(\mathbf{x}(s), \mathbf{w})`` or ``\hat q(\mathbf{x}(s), a, \mathbf{w})``| ``\hat v`` Optional| Either ``\hat q`` or ``\hat v`` Required|
|Policy Function: ``\hat \pi(a \vert \mathbf{x}(s))``| Explicit ``\pi(a \vert \mathbf{x}(s), \boldsymbol{\theta})``| Derived from ``\hat q`` or ``\hat v`` and deterministic|
|Gradient wrt Parameters| ``\nabla \pi(a \vert \mathbf{x}(s), \boldsymbol{\theta})`` (``\nabla \hat v`` optional)| ``\nabla \hat v(\mathbf{x}(s), \mathbf{w})`` or ``\nabla \hat q(\mathbf{x}(s), \mathbf{w})``|
|Theoretical Principle|Policy Gradient Theorem to find ``\max_{\boldsymbol{\theta}} J(\boldsymbol{\theta})`` |Use greedy policy wrt ``\hat v`` and hope policy improvement applies|
|Performance Notes|Accounts for limitations of ``\pi(a \vert \mathbb{x}(s), \boldsymbol{\theta})``|Limitations of ``\hat v / \hat q`` may sabotage policy improvement|
|Exploration Notes|Naturally stochastic policy can explore|Requires explicit use of ``\epsilon`` greedy action selection|
"""

# ╔═╡ ee9f5938-6ee1-4061-8e37-31c14bfbf8e2
md"""
# Search Methods
- Iterate through future possibilities from a starting state
- Build decisions in real time without any training process
- Often uses heuristics to intelligently prune branches
"""

# ╔═╡ 34f45e2f-4cf7-4356-b139-aaf2b628d762
md"""
## Monte Carlo Tree Search

- Requires guaranteed terminal state or an existing value function
- Suitable for discrete problems

## Breadth/Depth First Search

- Suitable for discrete problems with equal goal states
- Suffers from high memory use

## A* / IDA*

- Shortest path algorithms on a weighted graph
- Suitable for discrete problems with equal goal states
- More efficient but worst case scenario is as bad as exhaustive search
- Useful when entire graph is too large to process at once
"""

# ╔═╡ e203f464-80c8-4ab5-8a01-e6c9a5c7c530
md"""
# Hybrid Methods

## MCTS with Learned Value Function

"""

# ╔═╡ 46f9ec80-5f6f-4dbf-b943-ded02f264e58
md"""
# Settings and Dependencies
"""

# ╔═╡ cda55570-c3da-4953-8879-737d82f0b88e
TableOfContents()

# ╔═╡ 1ed57488-097d-4259-be41-af2e11d4270e
html"""
<style>
	main {
		margin: 0 auto;
		max-width: min(1600px, 90%);
		padding-left: max(10px, 10%);
		padding-right: max(10px, 5%);
		font-size: max(10px, min(24px, 2vw));
	}
</style>
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.71"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.7"
manifest_format = "2.0"
project_hash = "0c76a76c3ac8f04e01e91e0dc955aee1f9d81e4a"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

    [deps.ColorTypes.weakdeps]
    StyledStrings = "f489334b-da3d-4c2e-b8f0-e476e12c162b"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

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

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "8329a3a4f75e178c11c1ce2342778bcbbbfa7e3c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.71"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
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

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

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
# ╟─f229a1ef-76f7-4510-ae33-f54dd988636b
# ╟─8ae1e4ae-fa8f-432a-9c8b-76468b7480d4
# ╟─aab00166-e104-4381-b601-1c5c6bb7e7db
# ╟─d4cccc58-41c6-46f7-ba74-7bfe9560b705
# ╟─3457196c-bb39-4590-af4d-a13a6205a0f0
# ╟─38671154-91fe-11f0-01c6-451cf9c6062d
# ╟─293c909f-34e9-43b2-a2b3-5d37b6f3f5aa
# ╟─4af12e7a-1e21-4d3d-a42d-1ff4b940df3e
# ╟─3a1b6e8c-89ad-4384-a27e-10791aa50d52
# ╟─c11cb109-547a-462b-8e13-9aed74accf36
# ╟─2df524e6-7a6e-4dd4-bded-601b9c249779
# ╟─ee9f5938-6ee1-4061-8e37-31c14bfbf8e2
# ╟─34f45e2f-4cf7-4356-b139-aaf2b628d762
# ╟─e203f464-80c8-4ab5-8a01-e6c9a5c7c530
# ╟─46f9ec80-5f6f-4dbf-b943-ded02f264e58
# ╠═e681ab8e-5106-44e1-8ab0-3487aa876755
# ╠═cda55570-c3da-4953-8879-737d82f0b88e
# ╠═1ed57488-097d-4259-be41-af2e11d4270e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
