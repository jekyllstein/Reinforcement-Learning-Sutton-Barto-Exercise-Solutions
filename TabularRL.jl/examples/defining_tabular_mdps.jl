### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ 88fc8fef-c10b-408c-82d5-cb97c87aa363
using PlutoDevMacros, LinearAlgebra, Random, Statistics

# ╔═╡ 98c33627-bd5e-4d9e-b3c0-0c215b3abc1c
PlutoDevMacros.@frompackage @raw_str(joinpath(@__DIR__, "..", "..", "TabularRL.jl")) using TabularRL

# ╔═╡ 9a3df08c-7bc5-4508-a48a-44e01fcbbf5c
begin
	using PlutoUI
	TableOfContents()
end

# ╔═╡ 1504a12d-940e-4c03-9b83-6301b303d64b
md"""
# Installation

Both methods require you to [install julia](https://julialang.org/install/) on your computer.  Julia should then be accessible in your shell environment with the commmand `julia`.  Using this command will open the REPL, an example of which is shown in the first method below.

## Manual Setup in the REPL

After opening the REPL, you can access the package management system by typing `]`.  From there you can install `TabularRL.jl` with the command: `add https://github.com/jekyllstein/Reinforcement-Learning-Sutton-Barto-Exercise-Solutions:TabularRL.jl`.  If you do not want to add this package to your main environment, then activate a temporary one and add the package there.  Below is an example of installing the package into a temporary environment from the REPL

```julia
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.11.5 (2025-04-14)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia>]
(@v1.11) pkg> activate --temp
  Activating new project at `/tmp/jl_w0iM0P`
(jl_w0iM0P) pkg> add https://github.com/jekyllstein/Reinforcement-Learning-Sutton-Barto-Exercise-Solutions:TabularRL.jl
    Updating git-repo `https://github.com/jekyllstein/Reinforcement-Learning-Sutton-Barto-Exercise-Solutions`
   Resolving package versions...
    Updating `/tmp/jl_w0iM0P/Project.toml`
  [70984187] + TabularRL v0.1.0 `https://github.com/jekyllstein/Reinforcement-Learning-Sutton-Barto-Exercise-Solutions#jolin_dev_dev:TabularRL.jl`
    Updating `/tmp/jl_w0iM0P/Manifest.toml`
  [7d9f7c33] + Accessors v0.1.42
	.
	.
	.

```
"""

# ╔═╡ 2d0a1689-93d6-4be8-a991-64a2f57f8475
md"""
After the package is installed you can return to normal REPL mode by hitting `backspace` and then enter the command `using TabularRL`
"""

# ╔═╡ 12d83219-d470-40e9-8bee-a3db0b2bd2a8
md"""
```julia
julia> using TabularRL
```
"""

# ╔═╡ 761318f2-095b-4e3f-a320-061e9f50f166
md"""
## Automatic Setup with Pluto Notebooks

Alternatively, you can clone the entire reinforcement learning exercise repository and have access to every notebook and package contained therein.  Check to see if you have `git` installed on your computer with `git --version`.  If you receive an error message or do not see a version number then [install git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

Clone the repository to your system inside a directory where you have read/write access:  

```shell
> git clone https://github.com/jekyllstein/Reinforcement-Learning-Sutton-Barto-Exercise-Solutions.git
```

Navigate inside of the repository directory where you should find the following shell scripts: `setup.sh`, `start.sh`, `update.sh`.  Note that you may need to make these files executable with 

```shell
> chmod +x setup.sh```
```

You can begin by running the start shell script which will automatically run the setup script if this is your first time using the repository:

```shell
> ./start.sh
```

After some precompilation and setup, you should see the following at the bottom of the terminal:

```julia
[ Info: Loading...
┌ Info:
└ Go to http://localhost:1234/?secret=3Ah66MkG in your browser to start writing ~ have fun!
┌ Info:
│ Press Ctrl+C in this terminal to stop Pluto
```

However, note that the secret and port number may differ on your system.  The URL that contains `localhost` is what you should copy into your web browser to see the Pluto welcome screen.
"""

# ╔═╡ 1519dfbc-e593-4f1e-9b09-9af8157b04b8
md"""
![pluto_welcome](https://raw.githubusercontent.com/jekyllstein/Reinforcement-Learning-Sutton-Barto-Exercise-Solutions/02a1ea29b5cf9e8ce783d23dadcb3f33995e48c9/TabularRL.jl/examples/pluto_welcome.jpeg)
"""

# ╔═╡ 418687b7-73f4-476d-8eeb-9791830f44e3
md"""
If you click in the text box under `Open a notebook` a navigation menu will appear that shows the directory structure.  If you open any of the `Chapter...` folders, you will see notebook files which can be opened and used interactively.  For our purposes, however, we will open a template notebook which loads all of the required tools.  This notebook is contained at `Examples/template.jl` and can be opened from the text box (see below).
"""

# ╔═╡ 206588a7-0f0f-44a0-b982-abc4fdaa5582
md"""
![](https://raw.githubusercontent.com/jekyllstein/Reinforcement-Learning-Sutton-Barto-Exercise-Solutions/02a1ea29b5cf9e8ce783d23dadcb3f33995e48c9/TabularRL.jl/examples/template_opening.png)
"""

# ╔═╡ 20ebb844-bde2-41fd-a512-d62991e2f6d0
md"""
By default, the notebook will open in a preview mode (see below).  Click `Run notebook code` at the top to run the notebook and have access to all the tools.  From there you can add cells to the notebook and enter commands in them just like you would in the REPL.  The code examples which follow can work either in the REPL or the notebook.
"""

# ╔═╡ 40475cd8-80d9-4090-9f92-61b746923517
md"""
![](https://raw.githubusercontent.com/jekyllstein/Reinforcement-Learning-Sutton-Barto-Exercise-Solutions/02a1ea29b5cf9e8ce783d23dadcb3f33995e48c9/TabularRL.jl/examples/template_notebook.png)
"""

# ╔═╡ 908e9a26-f0c3-4e51-995e-f3b474fbc477
md"""
# Basic Usage

## Defining a Markov Reward Process
"""

# ╔═╡ 95efac9a-559b-4cb0-aaed-8116fa09f45a


# ╔═╡ b328f330-368c-11f0-107d-4b2801866e56
md"""
# Dependencies
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
PlutoDevMacros = "~0.9.0"
PlutoUI = "~0.7.62"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "f87f53f7371c1d9d6afcb48a3d9214024f41503e"

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

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "062c5e1a5bf6ada13db96a4ae4749a4c2234f521"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.9"

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

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "c47892541d03e5dc63467f8964c9f2b415dfe718"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.46"

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
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

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

[[deps.PlutoDevMacros]]
deps = ["JuliaInterpreter", "Logging", "MacroTools", "Pkg", "TOML"]
git-tree-sha1 = "72f65885168722413c7b9a9debc504c7e7df7709"
uuid = "a0499f29-c39b-4c5c-807c-88074221b949"
version = "0.9.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "d3de2694b52a01ce61a036f18ea9c0f61c4a9230"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.62"

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
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "cbbebadbcc76c5ca1cc4b4f3b0614b3e603b5000"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.2"

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
# ╟─1504a12d-940e-4c03-9b83-6301b303d64b
# ╟─2d0a1689-93d6-4be8-a991-64a2f57f8475
# ╟─12d83219-d470-40e9-8bee-a3db0b2bd2a8
# ╟─761318f2-095b-4e3f-a320-061e9f50f166
# ╟─1519dfbc-e593-4f1e-9b09-9af8157b04b8
# ╟─418687b7-73f4-476d-8eeb-9791830f44e3
# ╟─206588a7-0f0f-44a0-b982-abc4fdaa5582
# ╟─20ebb844-bde2-41fd-a512-d62991e2f6d0
# ╟─40475cd8-80d9-4090-9f92-61b746923517
# ╟─908e9a26-f0c3-4e51-995e-f3b474fbc477
# ╠═95efac9a-559b-4cb0-aaed-8116fa09f45a
# ╟─b328f330-368c-11f0-107d-4b2801866e56
# ╠═88fc8fef-c10b-408c-82d5-cb97c87aa363
# ╠═98c33627-bd5e-4d9e-b3c0-0c215b3abc1c
# ╠═9a3df08c-7bc5-4508-a48a-44e01fcbbf5c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
