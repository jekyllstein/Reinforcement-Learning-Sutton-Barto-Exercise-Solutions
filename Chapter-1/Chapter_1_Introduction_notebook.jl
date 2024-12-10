### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 10021908-6ebe-4682-b993-a1cdd73a4aa2
begin
	using PlutoUI
	TableOfContents()
end

# ╔═╡ 4394105f-ba75-4d38-97bb-7d97b685c734
md"""
# Chapter 1: Introduction
"""

# ╔═╡ f1fb61f6-4066-11ee-3c81-790e7afab21b
md"""
> ### *Exercise 1.1: Self-Play* 
> Suppose, instead of playing against a random opponent, the reinforcement learning algorithm described above played against itself, with both sides learning. What do you think would happen in this case? Would it learn a different policy for selecting moves?

With self play learning, the value function would include a value for every game state rather than just X moves.  Let us assume that the value represents the value for the X player.  In that case the policy that the O player should follow would be to select the move that leads to the lowest value state transition rather than the highest for the X player.  If we use temporal difference learning, sampling moves from this policy, then eventually it will converge to the value function for the minimax solution in which both sides exhibit optimal play.  The exploration parameter in the training process must be large enough that the entire state space is explored, otherwise there will not be accurate value estimates for certain states.  This policy would never lose to a suboptimal opponent, but there might be cases where it considers a number of candidate moves identical because they all should lead to a draw.  The policy trained against an imperfect opponent might favor a particular move that should not be different under perfect play but has a chance of leading to a win only against that opponent.  For example, in perfect play no starting move is preferable over another because any starting move results in a draw.  But against an opponent that plays randomly, certain moves like in the corners lead to more game states with winning paths if the opponent makes mistakes.
"""

# ╔═╡ dcbd0ceb-4d6c-4045-bbd1-ac51ffe232f9
md"""
> ### *Exercise 1.2: Symmetries* 
> Many tic-tac-toe positions appear different but are really the same because of symmetries. How might we amend the learning process described above to take advantage of this? In what ways would this change improve the learning process? Now think again. Suppose the opponent did not take advantage of symmetries.  In that case, should we? Is it true, then, that symmetrically equivalent positions should necessarily have the same value?

The value function table can be modified so that symmetrically equivalent positions are mapped to the same value rather than storing a separate value for each position.  For evaluating candidate moves that result in symmetrically equivalent positions we might pick randomly if they all contain the highest value function estimate.  If we play an opponent that does not treat equivalent positions the same, then we should not consider the symmetries for our value function because the behavior of the opponent will cause them to not be equivalent regarding our ability to win the game.
"""

# ╔═╡ 937eaf38-b2fa-459f-a8ee-8857c294a823
md"""
> ### *Exercise 1.3: Greedy Play* 
> Suppose the reinforcement learning player was greedy, that is, it always played the move that brought it to the position that it rated the best. Might it learn to play better, or worse, than a nongreedy player? What problems might occur?

When we begin the learning process, the value function is an inaccurate estimation that is continually updated.  The first move or set of moves that lead to a winning outcome will be followed by the greedy policy unless in future sampling they in fact lead to losing outcomes.  In either case the greedy policy will converge to playing the first set of moves that on average are marginally better than the average outcome.  This policy is not necessarily the optimal one and would not have accurate value function estimates for states that are never explored.  A non-greedy player will explore some additional moves that may not have a higher value estimate at the moment.  However after the value function is updated there is a greater chance of finding a move that has a higher value that what was previously considered optimal.  The greedy player could also get stuck in a bad policy because it has inaccurate value estimates for the other states.  Unless a player is allowed to explore the entire state space, it is not guaranteed to find the optimal policy.
"""

# ╔═╡ 29760a2e-d509-4d0e-8331-cd5de8723321
md"""
> ### *Exercise 1.4: Learning from Exploration* 
> Suppose learning updates occurred after all moves, including exploratory moves. If the step-size parameter is appropriately reduced over time (but not the tendency to explore), then the state values would converge to a different set of probabilities. What (conceptually) are the two sets of probabilities computed when we do, and when we do not, learn from exploratory moves? Assuming that we do continue to make exploratory moves, which set of probabilities might be better to learn? Which would result in more wins?

The two sets of probabilities represent two different value functions.  The original method that does not update after exploritory moves represents the value estimates for the greedy optimal policy; that is the policy that only considers moves that maximize the value estimate of the state transition.  If we also update from exploritory moves, then we are calculating the value estimate for policy that takes exploratory moves with some probability.  Then the policy optimization is occuring under the constraint that sometimes random moves are made, so the algorithm will find the best policy under that restriction.  If we call the probability of making random exploratory moves $\epsilon$, then this type of optimal policy can be called the $\epsilon$-greedy policy since it makes greedy moves $1 - \epsilon$ percent of the time.  The formal definition of this type of policy appears in subsequent chapters.  Since this policy is restricted, it will result in a worse final policy with fewer wins than the policy that ignores exploratory moves for learning updates..  Also, if we plan to eventually follow the greedy policy with respect to the value function when we deploy the agent, the value function will no longer accurately represent the policy being followed as it will have failed to converge to the optimal greedy policy (converging instead to the optimal $\epsilon$-greedy policy.  If, however, we consider an agent that continues to make exploratory moves after being deployed, then updating after exploratory moves will lead to a more accurate value function for that agent's policy.  It should lead to more wins than the original method because it accounts for a certain degree of random moves in the future.
"""

# ╔═╡ 20d261de-4121-4b81-8ead-f3022f8efedc
md"""
> ### *Exercise 1.5: Other Improvements* 
> Can you think of other ways to improve the reinforcement learning player? Can you think of any better way to solve the tic-tac-toe problem as posed?

Since the game is very simple, rather than estimating the value function and updating it with one forward step perhaps we could see what happens to the value function multiple steps into the future considering every possible response from our opponent.  That way we could explore more of the action space even if the opponent does not actually play those moves.  Even if we ignore all the value function estimation techniques, we can do this type of exhaustive search to just evaluate every starting position and subsequent positions to see which states lead to wins, losses, and draws.  This approach would simply simulate all the game states and have a policy based on every possible outcome from each state.  For a game this simple that approach is possible, but for games with many more states such an exhaustive search is intractable.  Also having a different reward assigned to draws and losses would help an agent distinguish between the two outcomes.  It may not increase the win rate, but it could avoid losses in cases where a draw is possible. 
"""

# ╔═╡ 7f53cffe-982e-426e-a76b-f2dce514a518
md"""
# Dependencies
"""

# ╔═╡ 65014025-5486-4e80-942b-e765aa3808c7
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
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.60"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.2"
manifest_format = "2.0"
project_hash = "8aa109ae420d50afa1101b40d1430cf3ec96e03e"

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
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

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
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

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
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

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
# ╟─4394105f-ba75-4d38-97bb-7d97b685c734
# ╟─f1fb61f6-4066-11ee-3c81-790e7afab21b
# ╟─dcbd0ceb-4d6c-4045-bbd1-ac51ffe232f9
# ╟─937eaf38-b2fa-459f-a8ee-8857c294a823
# ╟─29760a2e-d509-4d0e-8331-cd5de8723321
# ╟─20d261de-4121-4b81-8ead-f3022f8efedc
# ╟─7f53cffe-982e-426e-a76b-f2dce514a518
# ╠═10021908-6ebe-4682-b993-a1cdd73a4aa2
# ╠═65014025-5486-4e80-942b-e765aa3808c7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
