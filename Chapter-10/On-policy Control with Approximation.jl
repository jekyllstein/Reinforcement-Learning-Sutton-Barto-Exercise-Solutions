### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 310988d9-80f1-4fcd-9272-91908d1d367b
using PlutoUI

# ╔═╡ 6dec0d08-5ef7-11ed-2cb5-071408d50250
md"""
# 10.1 Episodic Semi-gradient Control
# 10.2 Semi-gradient *n*-step Sarsa
"""

# ╔═╡ 37d3812c-2710-4f97-b2f8-4dfd6f9b8390
md"""
> *Exercise 10.1* We have not explicitely considered or given pseudocode for any Monte Carlo methods in this chapter.  What would they be like?  Why is it reasonable not to give pseudocode for them?  How would they perform on the Mountain Car task?

Monte Carlo methods require an episode to terminate prior to updating any action value estimates.  After the final reward is retrieved then all the action value pairs visited along the trajectory can be updated and the policy can be updated prior to starting the next episode.  For tasks such as the Mountain Car task where a random policy will likely never terminate, such a method will never be able to complete a single episode worth of updates.  We saw in earlier chapters with the racetrack and gridworld examples that for some environments a bootstrap method is the only suitable one given this possibility of an episode never terminating.
"""

# ╔═╡ 6289fd48-a2ea-43d3-bcf8-bcc29447d425
md"""
> *Exercise 10.2* Give pseudocode for semi-gradient one-step *Expected* Sarsa for control.

Use the same pseudocode given for semi-gradient one-step Sarsa but with the following change to the weight update step in the non-terminal case:

$\mathbf{w} \leftarrow \mathbf{w} + \alpha[R + \gamma \sum_a \pi(a|S^\prime)\hat q(S^\prime, a, \mathbf{w}) - \hat q(S, A, \mathbf{w}) ] \nabla \hat q(S, A, \mathbf{w})$

where π is the currently used policy which is ϵ greedy with respect to q̂
"""

# ╔═╡ 15fe88ba-43a3-42cd-ba55-45f1586276e3
md"""
> *Exercise 10.3* Why do the results shown in Figure 10.4 have higher standard errors at large *n* than at small *n*?

At large n more of the reward function comes from the actual trajectory observed during a run.  Since random actions are taken initially there will be more spread in the observed reward estimates than with 1 step bootstrapping which is more dependent on the initialization of the action value function.  If ties are broken randomly then you would select random actions for the first n-steps of bootstrapping thus experience more spread in the early trajectories for higher n.
"""

# ╔═╡ 87b277b6-5c79-45fd-b6f3-e2e4ccf18f61
md"""
# 10.3 Average Reward: A New Problem Setting for Continuing Tasks
"""

# ╔═╡ 60901786-2f6f-451d-971d-27e684d079fa
md"""
> *Exercise 10.4* Give pseudocode for a differential version of semi-gradient Q-learning.

Given the pseudocode for semi-gradient Sarsa, make the following changes:

$\vdots$

Initialize S

Loop for each step of episode:

Choose A from S using ϵ-greedy policy
Take action A, observe R, S'

$\delta \leftarrow R - \bar R + \max_a \hat q(S^\prime, a, \mathbf{w}) - \hat q(S, A, \mathbf{w})$

$\vdots$
$S \leftarrow S^\prime$

"""

# ╔═╡ d06375b3-f377-45a6-be16-01b22c5a2b3f
md"""
> *Exercise 10.5* What equations are needed (beyond 10.10) to specify the differential version of TD(0)?
"""

# ╔═╡ 2c6951f9-33cb-400e-a83a-1a16f2ee0870
md"""
> *Exercise 10.6* Suppose there is an MDP that under any policy produces the deterministic sequence of rewards +1, 0, +1, 0, +1, 0, . . . going on forever. Technically, this violates ergodicity; there is no stationary limiting distribution $μ_\pi$ and the limit (10.7) does not exist. Nevertheless, the average reward (10.6) is well defined. What is it? Now consider two states in this MDP. From A, the reward sequence is exactly as described above, starting with a +1, whereas, from B, the reward sequence starts with a 0 and then continues with +1, 0, +1, 0, . . .. We would like to compute the di↵erential values of A and B. Unfortunately, the di↵erential return (10.9) is not well defined when starting from these states as the implicit limit does not exist. To repair this, one could alternatively define the differential value of a state as $v_\pi (s) \dot = \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} \sum_{t=0}^h \gamma^t \left ( \mathbb{E_\pi} [R_{t+1}|S_0=s]-r(\pi)  \right )$.  Under this definition what are the differential values of states A and B?

The average reward is 0.5 per step.

$v_\pi (A) = \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} 0.5 - 0.5\gamma + 0.5 \gamma^2 - 0.5\gamma^3 + \cdots =0.5\lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty}\sum_{t=0}^h (-\gamma)^t$
$=0.5\lim_{\gamma \rightarrow 1}\frac{1}{\gamma +1 } = 0.25$

$v_\pi (B) = \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} -0.5 + 0.5\gamma - 0.5 \gamma^2 + 0.5\gamma^3 + \cdots =-0.5\lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty}\sum_{t=0}^h (-\gamma)^t$
$=-0.5\lim_{\gamma \rightarrow 1}\frac{1}{\gamma +1 } = -0.25$
"""

# ╔═╡ 4a67aeba-dfaf-480d-84eb-7b8bcda549cb
md"""
> *Exercise 10.7* Consider a Markov reward process consisting of a ring of three states A, B, and C, with state transitions going deterministically around the ring.  A reward of +1 is received upon arrival in A and otherwise the reward is 0.  What are the differential values of the three states, using (10.13)

From 10.13 we have 

$v_\pi (s) \dot = \lim_{\gamma \rightarrow 1} \lim_{h \rightarrow \infty} \sum_{t=0}^h \gamma^t \left ( \mathbb{E_\pi} [R_{t+1}|S_0=s]-r(\pi)  \right )$

The average reward per step is $\frac{1}{3}$ so we can apply the same method used in exercise 10.6.  Now we need the value of the following infinite sums:

For state A:
$\frac{2}{3} - \frac{1}{3}\gamma - \frac{1}{3} \gamma^2 + \frac{2}{3}\gamma^3 + \cdots = 3 \times (2 - \gamma - \gamma^2 + 2\gamma^3 + \cdots)$

For state B:
$-\frac{1}{3} - \frac{1}{3}\gamma + \frac{2}{3}\gamma^2 - \frac{1}{3}\gamma^3 - \frac{1}{3}\gamma^4 + \cdots$

For state C:
$-\frac{1}{3} + \frac{2}{3}\gamma - \frac{1}{3}\gamma^2 - \frac{1}{3}\gamma^3 + \frac{2}{3}\gamma^4 + \cdots$

Comparing these sequences we have:

$\gamma \times v(A) = v(C) + \frac{1}{3}$
$\gamma \times v(B) = v(A) - \frac{2}{3}$
$\gamma \times v(C) = v(B) + \frac{1}{3}$


$\gamma \times v(A) = \frac{\frac{v(A) - \frac{2}{3}}{\gamma} + \frac{1}{3}}{\gamma} + \frac{1}{3}$

$\gamma^2 \times v(A) = \frac{v(A) - \frac{2}{3}}{\gamma} + \frac{1}{3} + \frac{\gamma}{3}$

$\gamma^3 \times v(A) = v(A) - \frac{2}{3} + \frac{\gamma}{3} + \frac{\gamma^2}{3}$

$v(A) (\gamma^3 - 1) = - \frac{2}{3} + \frac{\gamma}{3} + \frac{\gamma^2}{3}$

$v(A) = \frac{-2 +\gamma + \gamma^2}{3 \times (\gamma^3 - 1)}$

$v(A) = \frac{(\gamma + 1)(\gamma - 1)}{3 \times (\gamma-1)(\gamma^2+\gamma+1)} = \frac{(\gamma + 1)}{3 \times (\gamma^2+\gamma+1)}$

Therefore, 

$\lim_{\gamma \rightarrow 1} v(A) =\frac{2}{9}$
$\lim_{\gamma \rightarrow 1} v(B) = v(A) - \frac{2}{3}=\frac{2}{9}-\frac{6}{9}=-\frac{4}{9}$
$\lim_{\gamma \rightarrow 1} v(C) = v(B) + \frac{1}{3}=-\frac{4}{9}+\frac{3}{9}=-\frac{1}{9}$
"""

# ╔═╡ 685d31f0-7394-4a20-b9d0-3838b6d5645c
TableOfContents()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.48"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "502a5e5263da26fcd619b7b7033f402a42a81ffc"

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

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

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

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

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

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "cceb0257b662528ecdf0b4b4302eb00e767b38e7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "efc140104e6d0ae3e7e30d56c98c4a927154d684"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.48"

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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

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
# ╟─6dec0d08-5ef7-11ed-2cb5-071408d50250
# ╟─37d3812c-2710-4f97-b2f8-4dfd6f9b8390
# ╟─6289fd48-a2ea-43d3-bcf8-bcc29447d425
# ╟─15fe88ba-43a3-42cd-ba55-45f1586276e3
# ╟─87b277b6-5c79-45fd-b6f3-e2e4ccf18f61
# ╟─60901786-2f6f-451d-971d-27e684d079fa
# ╟─d06375b3-f377-45a6-be16-01b22c5a2b3f
# ╟─2c6951f9-33cb-400e-a83a-1a16f2ee0870
# ╟─4a67aeba-dfaf-480d-84eb-7b8bcda549cb
# ╠═310988d9-80f1-4fcd-9272-91908d1d367b
# ╠═685d31f0-7394-4a20-b9d0-3838b6d5645c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
