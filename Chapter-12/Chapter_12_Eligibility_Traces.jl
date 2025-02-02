### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 67f08f89-698c-4aa4-80d5-1ebcb830fc0c
using PlutoDevMacros, Random, Statistics, LinearAlgebra, StaticArrays, Transducers

# ╔═╡ 8a581882-c97d-4a3b-873a-212024a529a9
# ╠═╡ show_logs = false
PlutoDevMacros.@frompackage @raw_str(joinpath(@__DIR__, "..", "ApproximationUtils.jl")) using ApproximationUtils

# ╔═╡ f6125f11-8719-4c10-be91-3fe981e2d921
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI, PlutoPlotly ,StatsBase, BenchmarkTools, PlutoProfile, HypertextLiteral, LaTeXStrings
	TableOfContents()
end
  ╠═╡ =#

# ╔═╡ 062f756b-6640-4928-9216-c54316503944
begin
	include(joinpath(@__DIR__, "..", "Chapter-9", "Chapter_9_On-policy_Prediction_with_Approximation.jl"))
	include(joinpath(@__DIR__, "..", "Chapter-10", "Chapter_10_On_policy_Control_with_Approximation.jl"))
	include(joinpath(@__DIR__, "..", "Chapter-11", "Chapter_11_Off_policy_Methods_with_Approximation.jl"))
end

# ╔═╡ c62195dd-aa6e-4fd2-b9a9-848837a072d8
md"""
# Chapter 12: Eligibility Traces
"""

# ╔═╡ 3bc13de6-b767-4e2c-95bc-a44c1e688a77
md"""
Eligibility traces unify and generalize TD and Monte Carlo methods.  When TD methods are augmented with eligibility traces, they produce a family of methods spanning a spectrum that has Monte Carlo methods at one end (λ = 1) and one-step TD methods at the other (λ = 0).  In between are intermediate methods that are often better than either extreme.  Eligibility traces also provide a way of implementing Monte Carlo methods online and on continuing problems without episodes.

Of course, we have already seen one way of unifying TD and Monte Carlo methods: the n-step TD methods of Chapter 7.  What eligibility traces offer beyond these is an elegant algorithmic mechanism with significant computational advantages.  The mechanism is a short-term memory vector, the *eligibility trace* $\mathbf{z}_t \in \mathbb{R}^d$, that parallels the long-term weight vector $\mathbf{w}_t \in \mathbb{R}^d$.  The rough idea is that when a component of $\mathbf{w}_t$ participates in producing an estimated value, then the corresponding component of $\mathbf{z}_t$ is bumped up and then begins to fade away.  Learning will then occur in that component of $\mathbf{w}_t$ if a nonzero TD error occurs before the trace falls back to zero.  The trace-decay parameter $\lambda \in [0,1]$ determines the rate at which the trace falls.

The primary computational advantage of eligibility traces over n-step methods is that only a single trace vector is required rather than a store of the last n feature vectors.  Learning also occurs continually and uniformly rather than being delayed and then catching up at the end of the episode.  In addition, learning can occur and affect behavior immediately after a state is encountered rather than being belayed n steps.

Eligibility traces illustrate that a learning algorithm can sometimes be implemented in a different way to obtain computational advantages.  Many algorithms are most naturally formulated and understood as an update of a state's value based on events that follow that state over multiple future time steps.  For example Monte Carlo methods (Chapter 5) update a state based on all the future rewards, and n-step TD methods (Chapter 7) update based on the next n rewards and state n steps in the future.  Such formulations, based on looking forward from the updated state, are called *forward views*.  Forward views are always somewhat complex to implement because the update depends on later things that are not available at the time.  However, as we show in this chapter, it is often possible to achieve nearly the same updates -- and sometimes *exactly* the same updates -- with an algorithm that uses the current TD error, looking backwards to recently visited states using an eligibility trace.  These alternate ways of looking at and implementing learing algorithms are called *backward views*.  Backward views, transformations between forward views and backward views, and equivalences between them, date back to the introduction of temporal difference learning but have become much more powerful and sophisticated since 2014.  Here we present the basics of the modern view.

As usual, first we fully develop the ideas for state values and prediction, then extend them to action values and control.  We develop them first for the on-policy case then extend them to off-policy learning.  Our treatment pays special attention to the case of linear function approximation, for which the results with eligibility traces are stronger.  All these results apply also to the tabular and state aggregation casesbecause these are special cases of linear function approximation.
"""

# ╔═╡ 6426347e-4264-4a7f-9393-1065f8365efb
md"""
## 12.1 The λ-return

In Chapter 7 we defined an n-step return as the sum of the first n rewards plus the estimated value of the state reached in n steps, each appropriately discounted (7.1).  The general form of that equation, for any parametrized function approximator is:

$G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n \hat v(S_{t+n}, \mathbf{w}_{t+n-1}), \; 0 \leq t \leq T-n \tag{12.1}$

where $\hat v(s, \mathbf{w})$ is the approximate value of state $s$ given the weight vector $\mathbf{w}$ (Chapter 9), and $T$ is the time of episode termination, if any.  We noted in Chapter 7 that each n-step return for $n \geq 1$, is a valid update target for a tabularlearning update, just as it is for an approximate SGD learning update such as (9.7).

Now we noted that a valid update can bbe done not just toward anyn-step return, but toward any *average* of n-step returns for different ns.  For example, an update can be done toward a target that is half of a two-step return and half of a four-step return: $\frac{1}{2}G_{t:t+2} + \frac{1}{2}G_{t:t+4}$.  Any ste of n-step returns can be averaged in this way, even an infinite set, as long as the weights on the component returns are positive and sum to 1.  The composite return possesses an error reduction property similar to that of individual n-step returns (7.3) and thus can be used to construct updates with guaranteed convergence properties.  Averaging produces a substantial new range of algorithms.  For example, one could average one-step and infinite-step returns to obtain another way of interrelating TD and Monte Carlo methods.  In principle, one could even average experience based updates with DP updates to get a simple combination of experience-based and model-based methods (cf. Chapter 8).

An update that averages simpler component updates is called a *compound update*.  The backup diagram for a compound update consists of the backup diagrams for each of the component updates with a horizontal line above them and the weighting fractions below.  For example, the compound update for the case mentioned at the start of this section, mixing half of a two-step return and half of a four-step return, has the diagram shown below.  A compound update can only be done when the longest of its component updates is complete.  The update below, for example, could only be done at the time $t+4$ for the estimate formed at time $t$.  In general, one would like to limit the length of the longest component update because of the corresponding delay in the updates.
"""

# ╔═╡ c126b7c4-73df-4a0f-9ee4-a766eb19c5ba
#=╠═╡
@htl("""
<div style = "width: min(300px, 70vw); background-color: white;">
<div style = "height: 1px;"></div>
<hr style = "background-color: black; height: 2px; margin-left: 40px; margin-right: 40px; border: 0;">
<div style = "display: flex; align-items: flex-start; justify-content: space-around; padding: 5px; font-size: 1.2em;">
<div class = "backup-diagram">
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	$(md"""$\frac{1}{2}$""")
</div>
<div class = "backup-diagram">
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	$(md"""$\frac{1}{2}$""")
</div>
</div>
</div>
""")
  ╠═╡ =#

# ╔═╡ 01144f94-7a2e-4137-9cf8-4264d87a50a2
HTML("""
<style>
.backup-diagram {
	display: flex;
	flex-direction: column;
	align-items: center;
	justify-content: center;
	background-color: white;
	color: black;
	width: max(100px, 10vw); 
}

.down-arrow {
	display: flex;
	flex-direction: column;
	align-items: center;
	justify-content: flex-end;
	width: 2px;
	height: 35px;
	background-color: black;
	padding-bottom: 0px;
	margin-bottom: 1px;
}

.down-arrow::before {
	content: '';
	width: 0;
	height: 0;
	border-left: 4px solid transparent;
	border-right: 4px solid transparent;
	border-top: 8px solid black;
	transform: translateY(1px);
}

.state {
	width: 30px;
	height: 30px;
	border: 2px solid black;
	background-color: white;
	border-radius: 50%;
}
.action {
	width: 20px;
	height: 20px;
	background-color: black;
	border-radius: 50%;
}
.term {
	width: 30px;
	height: 30px;
	background-color: gray;
	border: 2px solid black;
}
</style>
""")

# ╔═╡ 50a4de5c-3856-45be-b552-011966faf9aa
md"""
The TD(λ) algorithm can be understood as one particular way of averaging n-step updates.  This average contains all the n-step updates, each weighted proportionally to $\lambda^{n-1}$ $($where $\lambda \in [0,1])$, and is normalized by a factor of $1-\lambda$ to ensure that the weights sum to 1 (Figure 12.1).  The resulting update is toward a return, called the λ-*return*, defined in its state-based form by

$G_t^\lambda \doteq (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} G_{t:t+n} \tag{12.2}$
"""

# ╔═╡ cd03392d-c19e-48ca-8d02-00bd342fbbb3
md"""
### Figure 12.1

The blackup diagram for TD$(\lambda)$.  If $\lambda = 0$ the overall update reduces to its first component, the one-step TD update, whereas if $\lambda=1$, then the overall update reduces to its last component, the Monte Carlo update.
"""

# ╔═╡ 2dcc2fa1-093d-4e6c-b168-4878d4e7ee86
#=╠═╡
@htl("""
<div style = "width: min(600px, 70vw); background-color: white;">
<div style = "color: black; font-size: 2em; display: flex; justify-content: center;">$(md"""TD$(\lambda)$""")</div>
<hr style = "background-color: black; height: 2px; margin-left: 40px; margin-right: 40px; border: 0;">
<div style = "display: flex; align-items: flex-start; justify-content: space-around; padding: 5px; font-size: 1.2em; color: black;">
<div class = "backup-diagram">
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	$(md"""$1-\lambda$""")
</div>
<div class = "backup-diagram">
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	$(md"""$(1-\lambda)\lambda$""")
</div>
<div class = "backup-diagram">
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	$(md"""$(1-\lambda)\lambda^2$""")
</div>
<div>
<div style = "font-size: 60px; color: black; transform: translateY(150px);">&hellip;</div>
<div style = "font-size: 60px; color: black; transform: translateY(420px) rotate(45deg) translateX(20px);">&hellip;</div>
</div>
<div class = "backup-diagram">
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "state"></div>
	<div class = "down-arrow"></div>
	<div class = "action"></div>
	<div style = "font-size: 40px; padding: 10px; color: black;">&vellip;</div>
	<div class = "action"></div>
	<div class = "down-arrow"></div>
	<div class = "term"></div>
	$(md"""$\lambda^{T-t-1}$""")
</div>
<div>
</div>
</div>
</div>
""")
  ╠═╡ =#

# ╔═╡ b670fe33-3db3-40f3-beb5-8508c260b3d0
md"""
Figure 12.2 further illustrates the weighting on the sequence of n-step returns in the λ-return.  The one-step return is given the largest weight, $1-\lambda$; the two-step return is given the next largest weight, $(1-\lambda)\lambda$; the three-step return si given the weight $(1-\lambda)\lambda^2$; and so on.  The weight fades by $\lambda$ with each additional step.  After a terminal state has been reached, all subsequent n-step returns are equal to the conventional return $G_t$.  If we want, we can separate these post-termination terms from the main sum, yielding

$G_t^\lambda \doteq (1 - \lambda) \sum_{n=1}^{T-t+1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t+1}G_t \tag{12.3}$

"""

# ╔═╡ 6ffa37d5-b587-4989-93fb-7003d818c082
md"""
### Figure 12.2

The weighting given in the λ-return to each of the n-step returns for an episode that terminates at step T.
"""

# ╔═╡ 2c1afacc-7956-479b-898d-eadf02a2ec19
#=╠═╡
@bind fig_12_2_params PlutoUI.combine() do Child
md"""
|λ|Steps Until Termination|
|:-:|:-:|
|$(Child(:λ, Slider(vcat(0:0.1:0.9, 0.91:.01:1), default = 0.5, show_value=true)))|$(Child(:T, Slider(1:50, default = 25, show_value=true)))|
"""
end
  ╠═╡ =#

# ╔═╡ 7f43afbf-3375-4ad1-acee-f6b74f98e20f
#=╠═╡
function figure_12_2(λ::Real, T::Integer)
	weights = vcat([(1-λ)*λ^(n-1) for n in 1:T], λ^(T-1))
	tr = bar(x = 1:T+1, y = weights)
	plot(tr, Layout(yaxis = attr(title = "Weighting", tickvals = [0, 1-λ, λ^(T-1), 1], range = [0, 1], ticktext = ["0", L"1-λ", L"λ^{T-t-1}", "1"]), xaxis = attr(title = "Time", tickvals = [1,2, T, T+1], ticktext = ["t", "t+1", "T-1", "T"])))
end
  ╠═╡ =#

# ╔═╡ 1035d33b-5e02-4d41-81cc-66546383db68
#=╠═╡
figure_12_2(fig_12_2_params...)
  ╠═╡ =#

# ╔═╡ dccc9b45-b711-44e8-8788-93de05f26543
md"""
> ### *Exercise 12.1* 
> Just as the return can be written recursively in terms of the first reward and itself one-step later (3.9), so can the λ-return. Derive the analogous recursive relationship from (12.2) and (12.1).

Revisiting (3.9): $G_t = R_{t+1} + \gamma G_{t+1}$.  We are looking for an equation for $G_t^\lambda$ of a similar form, i.e. $G_t^\lambda = \cdots + \gamma G_{t+1}^\lambda$

Using (12.1):

$\begin{flalign}
G_{t:t+n} &\doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n \hat v(S_{t+n}, \mathbf{w}_{t+n-1}) \\
G_{t+1:t+n} &= R_{t+2} + \gamma R_{t+3} +\cdots+\gamma^{n-2}R_{t+n} + \gamma^{n-1} \hat v(S_{t+n}, \mathbf{w}_{t+n-1})\\
\therefore \\
G_{t:t+n} &= R_{t+1} + \gamma G_{t+1:t+n} \tag{a}
\end{flalign}$

Using (12.2): 

$\begin{flalign}
G_{t}^\lambda &= (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} G_{t:t+n} \\
&= (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} (R_{t+1} + \gamma G_{t+1:t+n}) \tag{using (a)}\\
&= R_{t+1}(1-\lambda)\sum_{n=0}^\infty \lambda^n + \gamma(1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} G_{t+1:t+n}\\
&= R_{t+1}\frac{1-\lambda}{1-\lambda} + \gamma(1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} G_{t+1:t+n}\\
&= R_{t+1} + \gamma(1-\lambda)\left [ G_{t+1:t+1} + \lambda G_{t+1:t+2} + \lambda^2 G_{t+1:t+3} + \cdots \right ]\\
&= R_{t+1} + \gamma(1-\lambda)G_{t+1:t+1} + \gamma \lambda (1-\lambda)  \left [ G_{t+1:t+2} + \lambda G_{t+1:t+3} + \cdots \right ]\\

&\text{Note that also} \\ 
G_{t+1}^\lambda &= (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} G_{t+1:t+n+1} \\
&= (1-\lambda)(G_{t+1:t+2} + \lambda G_{t+1:t+3} + \lambda^2 G_{t+1:t+4} + \cdots) \\
&\text{and} \\

G_{t:t} &\doteq \hat v(S_t, \mathbf{w}_t) \\

&\text{So we can replace them in the above expression } \therefore \\

G_{t}^\lambda &= R_{t+1} + \gamma(1-\lambda)G_{t+1:t+1} + \gamma \lambda G_{t+1}^\lambda \\

 &= R_{t+1} + \gamma \left [ (1-\lambda)\hat v(S_{t+1}, \mathbf{w}_t) + \lambda G_{t+1}^\lambda \right ]\\

\end{flalign}$
From this expression it is clear that for $\lambda = 1$ we simply get $R_{t+1} + \gamma R_{t+2} + \cdots$ which is simply the monte carlo return.  For $\lambda = 0$, we get $R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w}_t)$ which is the 1 step TD return.

"""

# ╔═╡ 54e578de-d12c-4257-91b5-a257ea9c6ba6
md"""
Repeating the calculation with a terminal state

$\begin{flalign}
G_{t}^\lambda &= (1-\lambda)\sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t-1}G_t \\
G_{t+1}^\lambda &= (1-\lambda)\sum_{n=1}^{T-t-2} \lambda^{n-1} G_{t+1:t+n+1} + \lambda^{T-t-2}G_{t+1} \\

\\
G_{t}^\lambda &= (1-\lambda)\sum_{n=1}^{T-t-1} \lambda^{n-1} (R_{t+1} + \gamma G_{t+1:t+n}) + \lambda^{T-t-1}(R_{t+1} + \gamma G_{t+1}) \\
&= (1-\lambda)\left [ R_{t+1}\sum_{n=1}^{T-t-1} \lambda^{n-1} + \gamma \sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t+1:t+n}) \right ] + \lambda^{T-t-1}(R_{t+1} + \gamma G_{t+1}) \\
&= (1-\lambda)\left [ R_{t+1}\frac{1 - \lambda^{T-t-1}}{1 - \lambda} + \gamma \sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t+1:t+n}) \right ] + \lambda^{T-t-1}(R_{t+1} + \gamma G_{t+1}) \\
&= R_{t+1}(1 - \lambda^{T-t-1} + \lambda^{T-t-1}) + (1-\lambda)\gamma \sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t+1:t+n} + \lambda^{T-t-1} \gamma G_{t+1} \\
&= R_{t+1} + \gamma \left [(1-\lambda) \sum_{n=1}^{T-t-1} \lambda^{n-1}G_{t+1:t+n} + \lambda^{T-t-1} G_{t+1} \right ] \\
&= R_{t+1} + \gamma \left [(1-\lambda) \left ( \hat v (S_{t+1}) + \sum_{n=2}^{T-t-1} \lambda^{n-1}G_{t+1:t+n} \right ) + \lambda^{T-t-1} G_{t+1} \right ] \\
&= R_{t+1} + \gamma \left [(1-\lambda) \left ( \hat v (S_{t+1}) + \sum_{m=1}^{T-t-2} \lambda^{m}G_{t+1:t+m+1} \right ) + \lambda^{T-t-1} G_{t+1} \right ] \\
&= R_{t+1} + \gamma \left [(1-\lambda) \hat v (S_{t+1}) + \lambda  \left ( (1-\lambda) \sum_{m=1}^{T-t-2} \lambda^{m-1}G_{t+1:t+m+1} + \lambda^{T-t-2} G_{t+1} \right )  \right ] \\
&= R_{t+1} + \gamma \left [(1-\lambda) \hat v (S_{t+1}) + \lambda  G_{t+1}^\lambda  \right ] \\
\end{flalign}$
From this expression it is clear that for $\lambda = 1$ we simply get $R_{t+1} + \gamma R_{t+2} + \cdots$ which is simply the monte carlo return.  For $\lambda = 0$, we get $R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w}_t)$ which is the 1 step TD return.

"""

# ╔═╡ 9d131051-eeee-4aba-8f78-9ddff9babab4
#=╠═╡
function plot_hl()
	τ(λ) = - log(2) / log(λ)
	λs = 0:0.001:1
	plot(λs, τ.(λs), Layout(xaxis_title = "λ", yaxis_title = L"τ_λ", yaxis_range = [0, 5]))
end
  ╠═╡ =#

# ╔═╡ 752a80ea-1da6-49ef-91ef-a03c590b825d
#=╠═╡
md"""
> ### *Exercise 12.2* 
> The parameter λ characterizes how fast the exponential weighting in Figure 12.2 falls off, and thus how far into the future the λ-return algorithm looks in determining its update. But a rate factor such as λ is sometimes an awkward way of characterizing the speed of the decay. For some purposes it is better to specify a time constant, or half-life. What is the equation relating λ and the half-life, $\tau_\lambda$, the time by which the weighting sequence will have fallen to half of its initial value?

The initial weight for $n=1$ is $\lambda^0=1$, so the question is at what n will the weight value be $\frac{1}{2}$.  That will occur when:

$\begin{flalign}
\lambda^{n_\tau-1} &= \frac{1}{2}\\
(n_\tau-1) \log{\lambda} &= \log{\frac{1}{2}} \\
n_\tau-1 &= \frac{\log{1} - \log{2}}{\log{\lambda}} \\
\end{flalign}$

 $n = 1$ corresponds to the reference time, so $\tau_\lambda = n_\tau - 1 = - \frac{log{2}}{\log{\lambda}}$

From the plot we can see that the halflife approaches infinity as λ approaches 1 which we expect from the monte-carlo return.  Also $\lambda = \frac{1}{2} \implies \tau_\lambda = 1$.
$(plot_hl())
"""
  ╠═╡ =#

# ╔═╡ 134ce360-6290-4aea-b6c0-eaa825d6f9a5
md"""
We are now ready to define our first learning algorithm based on the λ-return: the *off-line λ-return algorithm*.  As an off-line algorithm, itmakes no changes to the weight vector during the episode.  Then, at the end of the episode, a whole sequence of off-line updates are made according to our usual semi-gradient rule, using the λ-return as the target:

$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \left [ G_t^\lambda - \hat v (S_t, \mathbf{w}_t) \right ] \nabla \hat v (S_t, \mathbf{w}_t), \; t = 0, \dots, T-1 \tag{12.4}$

The λ-return gives us an alternative way of moving smoothly between Monte Carlo and one-step TD methods that can be compared with the n-step bootstrapping way developed in Chapter 7.  There we assessed effectiveness on a 19-state random walk task (Example 7.1).  Figure 12.3 shows the performance of the off-line λ-return algorithm on this task alongside that of the n-step methods (repeated from Figure 7.2).  The experiment was just as described earlier except that for the λ-return algorithm λ is varied instead of n.  The performance measure used is the estimated root-mean-square error between the correct and estimated values of each state measured at the end of the episode, averaged over the first 10 episodes and the 19 states.  Note that overall the performance of the off-line λ-return algorithms in comparable to that of the n-step algorithms.  In both cases we get the best performance with an intermediate value of the bootstrapping parameter, n for the n-step methods and λ for the off-line λ-return algorithm.  The preference for higher values of n and λ increases with a larger random walk and vice versa.
"""

# ╔═╡ 37ffc88c-8418-468b-a537-37b8e6bf5922
md"""
### *Off-line λ-return and random walk example*
"""

# ╔═╡ 9d512c7b-3d49-439a-a971-1a3dad065d6e
function n_step_TD_prediction(mrp::TabularMRP{X, S, P, F}, γ::X, num_episodes, n::Integer; v_est::Vector{X} = initialize_state_value(mrp), α::X = one(X)/100, calc_err::Function = (v_est) -> zero(T), static_values = false, save_error = false, epkwargs...) where {X<:Real, S, P, F}
	#initialize
	stateindexbuffer = MVector{n+1, Int64}(zeros(Int64, n+1))
	rewardbuffer = MVector{n+1, X}(zeros(X, n+1))
	get_state_index(i) = stateindexbuffer[mod(i, n+1) + 1]
	get_reward(i) = rewardbuffer[mod(i, n+1) + 1]
	get_value(buffer, i) = buffer[mod(i, n+1)+1]
	v_est[mrp.terminal_states] .= zero(X) #terminal state must always have 0 value
	if static_values
		v_est2 = copy(v_est)
	end

	error_history = Vector{X}(undef, num_episodes)

	#simulate and episode and update the value function every step
	function runepisode!(V, j)
		i_s = mrp.initialize_state_index()
		T = typemax(Int64)
		t = 0
		τ = 0
		stateindexbuffer[1] = i_s
		while τ != T - 1
			if t < T
				(r, i_s′) = mrp.ptf(i_s)
				i_s = i_s′
				i = mod(t+1, n+1) + 1
				stateindexbuffer[i] = i_s′
				rewardbuffer[i] = r
				if mrp.terminal_states[i_s′]
					T = t + 1
				end
			end
			τ = t - n + 1
			if τ >= 0
				G = zero(X)
				for i in τ+1:min(τ+n, T)
					G += (γ^(i - τ - 1))*get_reward(i)
				end
				if τ+n < T
					G += γ^n * v_est[get_state_index(τ+n)]
				end
				i_τ = get_value(stateindexbuffer, τ)
				update_value = V[i_τ] + α*(G-v_est[i_τ])
				if static_values
					v_est2[i_τ] = update_value
				else
					v_est[i_τ] = update_value
				end
			end
			t += 1
		end
	
		if static_values
			v_est .= v_est2
		end

		if save_error
			error_history[j] = calc_err(v_est)
		end
		return V
	end
		
	for i = 1:num_episodes; runepisode!(v_est, i); end
	
	return v_est, error_history
end

# ╔═╡ d55aca40-b03a-4f6b-84e6-ced6c8f67da1
function offline_λ_return_prediction!(params, mrp::StateMRP{T, S, P, F1, F2}, γ::T, λ::T, num_episodes::Integer, state_representation, update_state_representation!::Function, estimate_value::Function, update_params!::Function; α::T = one(T)/100, calc_err::Function = params -> zero(T), static_params = false, save_error = false, epkwargs...) where {T<:Real, S, P, F1, F2}
	#initialize
	if static_params
		params2 = copy(params)
	end

	error_history = Vector{T}(undef, num_episodes)

	(states, rewards, sterm, num_steps) = runepisode(mrp; epkwargs...)

	params_arg = if static_params
		params2
	else
		params
	end

	function episode_update!(params, states, rewards)
		l = length(rewards)
		g = zero(T)
		g_λ = rewards[l]
		update_state_representation!(state_representation, states[l])
		v̂ = estimate_value(state_representation, params)
		update_params!(params_arg, g_λ, v̂, state_representation, α)
		for i = l-1:-1:1
			g_λ = rewards[i] + γ*((1-λ)*v̂ + λ*g_λ)
			update_state_representation!(state_representation, states[i])
			v̂ = estimate_value(state_representation, params)
			update_params!(params_arg, g_λ, v̂, state_representation, α)
		end
		if static_params
			params .= params2
		end
	end

	episode_update!(params, states, rewards)
	if save_error
		error_history[1] = calc_err(params)
	end

	for ep in 2:num_episodes
		(states, rewards, sterm, num_steps) = runepisode!((states, rewards), mrp; epkwargs...)
		episode_update!(params, view(states, 1:num_steps), view(rewards, 1:num_steps))
		if save_error
			error_history[ep] = calc_err(params)
		end
	end
	
	return params, error_history
end

# ╔═╡ a5877ac6-3bd8-4832-bf19-618b01ba16e2
#=╠═╡
function run_random_walk_offline_λ_estimation(mrp, nstates, calc_err::Function, α, λ; num_episodes = 10, kwargs...)
	params = zeros(Float32, nstates)
	state_representation = zeros(Float32, nstates)
	estimate_value(params, state_representation) = dot(params, state_representation)
	function update_state_representation!(state_representation, state)
		state_representation .= 0f0
		state_representation[state] = 1f0
	end

	function update_params!(params, g_λ, v̂, state_representation, α)
		params .+= α .* (g_λ - v̂) .* state_representation
	end

	params, error_history = offline_λ_return_prediction!(params, mrp, 1f0, λ, num_episodes, state_representation, update_state_representation!, estimate_value, update_params!; α = α, calc_err = calc_err, save_error = true, static_params=false, kwargs...)
	return mean(error_history)
end
  ╠═╡ =#

# ╔═╡ fba683ec-c923-498c-b379-9d23a1d4aa76
#=╠═╡
run_random_walk_offline_λ_estimation_trials(mrp, nstates, calc_err, α, λ; num_trials = 100, kwargs...) = (1:num_trials |> Map(_ -> run_random_walk_offline_λ_estimation(mrp, nstates, calc_err, α, λ; num_episodes = 10, kwargs...)) |> foldxt(+)) / num_trials
  ╠═╡ =#

# ╔═╡ ce8b9ebf-942a-4807-a36f-ced03c3c7916
function value_estimate_random_walk(nstates, α, n; kwargs...)
	mrp = TabularRL.create_random_walk_distribution(nstates, -1f0, 1f0)
	c = (nstates + 1)/2
	v_true = [(s-c)/c for s in 1:nstates]
	value_estimate_random_walk(mrp, v_true, α, n; kwargs...)
end

# ╔═╡ c240d631-3095-4880-b454-66e05a59e4ea
#=╠═╡
function value_estimate_random_walk(mrp, v_true, α, n; num_trials = 100, num_episodes = 10, kwargs...)
	calc_err(v) = sqrt(mean(i -> (v[i] - v_true[i])^2, 1:length(v_true)))
	(1:num_trials |> Map(i -> mean(n_step_TD_prediction(mrp, 1f0, num_episodes, n; α = α, save_error = true, calc_err = calc_err, kwargs...)[2])) |> foldxt(+)) / num_trials
end
  ╠═╡ =#

# ╔═╡ eefb36bb-d988-4f5d-bfbb-c3df2f869ab6
#=╠═╡
function offline_λ_error_random_walk(nstates; kwargs...)
	α_vec = vcat(Float32.(0.0:0.02:0.1), 0.15f0, Float32.(0.2:0.1:1.0))
	λ_vec = [0f0, 0.4f0, 0.8f0, 0.9f0, 0.95f0, 0.975f0, 0.99f0, 1f0]
	mrp = StateMRP(TabularRL.create_random_walk_distribution(nstates, -1f0, 1f0))
	c = (nstates + 1)/2
	v_true = [(s-c)/c for s in 1:nstates]
	calc_err(v) = sqrt(mean(i -> (v[i] - v_true[i])^2, eachindex(v_true)))
	get_α_line(λ) = α_vec |> Map(α -> run_random_walk_offline_λ_estimation_trials(mrp, nstates, calc_err, α, λ; kwargs...)) |> collect
	lines = λ_vec |> Map(λ -> get_α_line(λ)) |> collect
	traces = [scatter(x = α_vec, y = lines[i], name = "λ = $λ", mode = "lines", line_shape = "spline") for (i, λ) in enumerate(λ_vec)]
	plot(traces, Layout(title = "Off-line λ-return algorithm", xaxis_title = "α", height = 500, yaxis_title = "Average RMS error over $nstates <br> states and first 10 episodes", yaxis_range = [minimum(minimum(x) for x in lines) - 0.01, first(first(lines))]))
end
  ╠═╡ =#

# ╔═╡ 583d2f42-692a-4028-93cc-47c2e178c84e
#=╠═╡
function nsteptd_error_random_walk(nstates; kwargs...)
	α_vec = vcat(Float32.(0.0:0.02:0.1), 0.15f0, Float32.(0.2:0.1:1.0))
	n_vec = 2 .^ (0:9)
	mrp = TabularRL.create_random_walk_distribution(nstates, -1f0, 1f0)
	c = (nstates + 1)/2
	v_true = [(s-c)/c for s in 1:nstates]
	get_α_line(n) = α_vec |> Map(α -> value_estimate_random_walk(mrp, v_true, α, n; kwargs...)) |> collect
	lines = n_vec |> Map(n -> get_α_line(n)) |> collect
	traces = [scatter(x = α_vec, y = lines[i], name = "n = $n", mode = "lines", line_shape = "spline") for (i, n) in enumerate(n_vec)]
	plot(traces, Layout(title = "n-step TD methods", xaxis_title = "α", height = 500, yaxis_title = "Average RMS error over $nstates <br> states and first 10 episodes", yaxis_range =  [minimum(minimum(x) for x in lines) - 0.01, first(first(lines))]))
end
  ╠═╡ =#

# ╔═╡ 176ee625-51c6-48f1-8f01-d3fb7008db6e
#=╠═╡
md"""
### Figure 12.3

 $(@bind fig_12_3_n NumberField(1:30, default = 19))-state Random walk results: Performance of the off-line λ-return algorithm alongside that of the n-step TD methods.  In both cases intermediate values of the bootstrapping parameter (λ or n) performed bets.  The results with the off-line λ-return algorithm are slightly better at the best values of α and λ, and at high α.  The importance of bootstrapping diminishes the smaller the random walk chain is.
"""
  ╠═╡ =#

# ╔═╡ 57cf5ae7-d4dd-47e8-8090-c04fb39e0763
md"""
## 12.2 TD(λ)
TD $(\lambda)$ uses eligibility traces to look backward and compute something that approaches the theoretical forward view of the off-line λ-return.  It improves over the off-line λ-return algorithm by performing updates on every step rather than at the end of an episode.  Thus it can also be applied to continuing problems instead of just episodic problems which require an episode to reach a terminal state.  A semi-gradient version of TD $(\lambda)$ can be applied to function approximation which can also apply to tabular problems in the simple case of state aggregation with one state per parameter.

With function approximation, the eligibility trace is a vector $\mathbf{z}_t \in \mathbb{R}^d$ with the same number of components as the weight vector $\mathbf{w}_t$.  Whereas the weight vector is a long term memory accumulating over the lifetime of the system, the eligibility trace is a short-term memory, typically lasting less than the length of an episode.  In TD $(\lambda)$, the eligibility trace vector is initialized to zero at the beginning of the episode, is incremented on each time step by the value graient, and then fades away by $\lambda \gamma$:
"""

# ╔═╡ 34dda4bf-f78f-4c83-ba10-9b206d2fbcb8
md"""
$\begin{flalign}
\mathbf{z}_{-1} &\doteq \mathbf{0} \\
\mathbf{z_t} &\doteq \gamma \lambda \mathbf{z}_{t-1} + \nabla \hat v(S_t, \mathbf{w_{t}}),  \hspace{5 mm} 0 \leq t \leq T-1 \tag{12.5} \\
\end{flalign}$

where $\gamma$ is the discount rate and $\lambda$ is the parameter introduce with the $\lambda$-return and called the trace-decay parameter.  The eligibility trace keeps track of which components of the weight vector have contributed, positively or negatively, to recentstate valuations, where "recent" is defined in terms of $\gamma \lambda$.  (Recall that in linear function approximation, $\nabla \hat v (S_t, \mathbf{w}_t)$ is the feature vector, $\mathbb{x}_t$, in which case the eligibility trace vector is just a sum of past, fading, input vectors.)  The trace is said to indicate the eligibility of each component of hte weight vector for undergoing learning changes should a reinforcing even occur.  The reinforcing events we are concerned with are the moment-by-moment one-step TD errors.  The TD error for state-value prediction is:

$\delta_t \doteq R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w}_t) - \hat v(S_t, \mathbf{w}_t) \tag{12.6}$

In TD $(\lambda)$, the weight vector is updated on each step proportional to the scalar TD error and the vector eligibility trace:

$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t \mathbf{z}_t \tag{12.7}$
"""

# ╔═╡ 6f5168dc-f1f3-4533-a59e-bb85895f3b13
md"""
### *Semi-gradient TD(λ) for estimating $\hat v \approx v_\pi$*
"""

# ╔═╡ 5610a0ba-60a8-4da6-8f68-50b1c5e82686
begin
	#note that this function will modify both parameters and the state representation vector as well as some of the keyword arguments
	function semi_gradient_TDλ!(parameters::P, state_representation::X, initialize_state::Function, transition::Function, isterm::Function, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, update_state_representation!::Function, estimate_value::Function, update_gradient!::Function; α = one(T)/10, calculate_error::Function = params -> zero(T), ∇v::P = copy(parameters), z::P = copy(parameters), save_step_errors::Bool = false, save_episode_errors::Bool = false, epkwargs...) where {P, X, T<:Real}
		s = initialize_state()
		update_state_representation!(state_representation, s)
		v̂ = estimate_value(state_representation, parameters)
		update_gradient!(∇v, state_representation, parameters)
		(r, s′) = transition(s)
		if isterm(s′)
			v̂′ = zero(T)
		else
			update_state_representation!(state_representation, s′)
			v̂′ = estimate_value(state_representation, parameters)
		end
		ep = 1
		step = 1
		episode_error_history = Vector{T}()
		step_error_history = Vector{T}()
		
		#initialize eligibility vector to 0
		z .= zero(T)
		
		while (ep <= max_episodes) && (step <= max_steps)
			z .= (γ*λ .* z) .+ ∇v
			δ = r + γ*v̂′ - v̂
			parameters .+= α*δ .* z
	
			save_step_errors && push!(step_error_history, calculate_error(parameters))
	
			if isterm(s′)
				s = initialize_state()
				update_state_representation!(state_representation, s)
				#reset eligibility vector to 0 at the start of a new episode
				z .= zero(T)
				ep += 1
				save_episode_errors && push!(episode_error_history, calculate_error(parameters))
			else
				s = s′
			end

			#note that the state representation here will be for s on the next step
			v̂ = estimate_value(state_representation, parameters)
			update_gradient!(∇v, state_representation, parameters)
			
			(r, s′) = transition(s)
			
			if isterm(s′)
				v̂′ = zero(T)
			else
				update_state_representation!(state_representation, s′)
				v̂′ = estimate_value(state_representation, parameters)
			end
			step += 1
		end
		return (episode_errors = episode_error_history, step_errors = step_error_history)
	end

	#when evaluating an MRP, there is no policy and the transition is just from the environment
	semi_gradient_TDλ!(parameters::P, state_representation::X, mrp::StateMRP, args...; kwargs...) where {P, X} = semi_gradient_TDλ!(parameters, state_representation, mrp.initialize_state, s -> mrp.ptf(s), mrp.isterm, args...; kwargs...)

	#when evaluating an MDP, there is a policy and the transition uses it to select actions
	semi_gradient_TDλ!(parameters::P, state_representation::X, mdp::StateMDP, π::Function, args...; kwargs...) where {P, X} = semi_gradient_TDλ!(parameters, state_representation, mdp.initialize_state, s -> mdp.ptf(s, π), mdp.isterm, args...; kwargs...)
end

# ╔═╡ 5e5fdcee-356e-46d4-a5b0-3c433aee989d
md"""
Note that in the case of linear approximation, there is some function $\mathbf{x}(s)$ which produces a state representation vector of length $d$ and a parameter vector $\mathbf{w} \in \mathbb{R}^d$.  The value function and gradient then take on the form:

$\begin{flalign}
\hat v(s, \mathbf{w}) &= \mathbf{w}^\top \mathbf{x}(s) = \sum_i w_i x_i \\
\nabla \hat v(s, \mathbf{w}) &= \mathbf{x}(s) = [x_1, x_2, x_3, ..., x_d]
\end{flalign}$

So to implement this algorithm in the linear case, the only function that requires definition is $\mathbf{x}(s)$.  To use the linear version of the algorithm defined below, one need only specify the number of parameters $d$ and `update_state_representation!(x, s)` which updates a vector x given state s.
"""

# ╔═╡ 24468748-009d-42a3-918d-4ba18b23c9ed
#for linear function approximation the number of parameters also define the size of the state representation.  the function that updates the state representation is all that is required to calculate the updates.  problem will either be an MRP or and MDP plus a policy to evaluate
function run_linear_semi_gradient_TDλ(problem, num_params::Integer, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, update_state_representation!::Function; parameters::Vector{T} = zeros(T, num_params), state_representation::Vector{T} = zeros(T, num_params), kwargs...) where {T<:Real}
	@assert length(parameters) == length(state_representation) == num_params
	
	#the value estimation function is just the dot product of the parameters with the state representation
	estimate_value(x, w) = dot(x, w)

	#the gradient is just identical to the state representation
	function update_gradient!(∇v, x, w)
		∇v .= x
	end

	error_history = semi_gradient_TDλ!(parameters, state_representation, problem..., γ, λ, max_episodes, max_steps, update_state_representation!; kwargs...)

	#once the learning is done we can estimate values with the final version of the parameters.  these versions of the value function allow the computation to occur with a passed state representation vector and set of parameters or to use the existing parameters and define a new state vector each time
	function v!(x::Vector{T}, s, w::Vector{T})
		update_state_representation!(x, s)
		estimate_value(x, w)
	end

	v!(x::Vector{T}, s) = v!(x, s, parameters)

	function v(args...)
		x = zeros(T, num_params)
		v!(x, args...)
	end
	
	return (value_function = v, error_history = error_history)
end

# ╔═╡ e99caf5c-7c13-4edd-b55b-dce93cc850c6
md"""
In the case of a tabular problem with $d$ states: $[s_1, s_2, \dots, s_d]$, then the state representation can be ignored in favor of using the state index to compute the gradient and function approximation.  Now the size of the parameter vector is defined by the number of states $d$ and each parameter $w_i$ corresponds to the value of $s_i$.

$\begin{flalign}
\mathbf{w} &= [w_1, w_2, \dots, w_d] \\
\hat v(s_i, \mathbf{w}) &= w_i \\
\nabla \hat v (s_1, \mathbf{w}) &= [1, 0, 0, \dots, 0] \\ 
\nabla \hat v (s_2, \mathbf{w}) &= [0, 1, 0, \dots, 0] \\ 
&\vdots \\
\nabla \hat v (s_d, \mathbf{w}) &= [0, 0, 0, \dots, 1] \\ 
\end{flalign}$

The algorithm below runs semi-gradient TD(λ) for a tabular problem without needing to recast it as a StateMDP.  The definition of the parameters, value estimation function, and gradient update are all handled automatically according to the above rules.
"""

# ╔═╡ 900760f0-b253-4db7-8c4f-4ca34777198d
begin
	#in the case of a tabular problem, this algorithm can be used with a trivial version of the linear algorithm
	function run_tabular_semi_gradient_TDλ(states::Vector{S}, initialize_state_index::Function, transition::Function, terminal_states::BitVector, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; parameters::Vector{T} = zeros(T, length(states)), state_representation::Vector{Int64} = zeros(Int64, 1), kwargs...) where {T<:Real, S}
		@assert length(parameters) == length(states)

		#the state representation just stores the index of the state
		function update_state_representation!(x::Vector{Int64}, i_s::Integer)
			x[1] = i_s
		end
		
		#the value estimation function is just the parameter at the state index
		estimate_value(x, w) = w[x[1]]

		#the gradient is just 1 at the state index and zero elsewhere
		function update_gradient!(∇v, x, w)
			∇v .= zero(T)
			∇v[x[1]] = one(T)
		end

		error_history = semi_gradient_TDλ!(parameters, state_representation, initialize_state_index, transition, i_s -> terminal_states[i_s], γ, λ, max_episodes, max_steps, update_state_representation!, estimate_value, update_gradient!; kwargs...)
		
		return (state_values = parameters, error_history = error_history)
	end

	run_tabular_semi_gradient_TDλ(mrp::TabularMRP, args...; kwargs...) = run_tabular_semi_gradient_TDλ(mrp.states, mrp.initialize_state_index, i_s -> mrp.ptf(i_s), mrp.terminal_states, args...; kwargs...)

	run_tabular_semi_gradient_TDλ(mdp::TabularMDP, π::Function, args...; kwargs...) = run_tabular_semi_gradient_TDλ(mdp.states, mdp.initialize_state_index, i_s -> mdp.ptf(i_s, π), mdp.terminal_states, args...; kwargs...)
end

# ╔═╡ 373a89e3-0b8d-49a0-982e-8bb300538429
md"""
TD$(\lambda)$ is oriented backward in time.  At each moment we look at the current TD error and assign it backward to each prior state according to how much that state contributed to the current eligibility trace at that time.  We might imagine ourselves riding along hte stream of states, computing TD errors, and shouting them back to the previously visited states.  Where the TD error and traces come together, we get the update given by (12.7), changing the values of those past states for when they once again occurin the future.

If $\lambda = 0$, then (12.5) implies that the trace at $t$ is exactly the value gradient corresponding to $S_t$.  Thus the TD$(\lambda)$ update reduces to the one-step semi-gradient TD update treated in Chapter 9 or the simple TD rule (6.2) in the tabular case.  This is why the algorithm was called TD(0).  TD(0) is the case in which only the one state preceding the current one is updated by the TD error (other states may have their value estimates changed by generalization due to the function approximation).  For larger values of $\lambda$, but still $\lambda \lt 1$, more of the preceding states are updated, but each more temporally distant state is updated less because the correspondong eligibility trace is smaller.  We say that the earlier states are given less *credit* for the TD error.

If $\lambda = 1$, then the credit given to earlier states falls only by $\gamma$ per step.  This turns out to be just the right thing to do to achieve Monte Carlo behavior.  For example, remember that the TD error $\delta_t$, includes an undiscounted term of $R_{t+1}$.  In passing this back $k$ steps it needs to be discounted, like any reward in a return, by $\gamma^k$, which is just what the falling eligibility trace achieves.  If $\lambda = 1$ and $\gamma = 1$, then the eligibility traces do not decay at all with time.  In this case the method behaves like a Monte Carlo method for an undiscounted, episodic task.  If $\lambda = 1$, the algorithm is known as TD(1).

TD(1) is a way of implementing Monte Carlo algorithms that is more general than those presented earlier and that significantly increases their range of applicability.  Whereas the earlier Monte Carlo methods were limited to episodic tasks, TD(1) can be applied to discounted continuing tasks as well.  Moreover, TD(1) can be performed incrementally and online.  One disadvantage of Monte Carlo methods is that they learn nothing from an episode until it is over.  For example, if a Monte Carlo control method takes an action that produces a very poor reward but does not end the episode, then the agent's gendency to repeat the action will be undimiished during the episode.  Online TD(1), on the other hand, learns in an *n*-step TD way from the incomplete ongoing episode where the *n* steps are all the way up to the current step.  If something unusually good or bad happens during an episode, control methods based on TD(1) can learn immediately and alter their behavior on that same episode.
"""

# ╔═╡ 2c3b163d-b4cd-4b40-a597-cbd103e135b6
md"""
### Comparing TD(λ) and Off-line λ-return algorithm on random walk example

If is revealing to revisit the random walk example (Example 7.1) to see how well TD(λ) does in approximating the off-line λ-return algorithm.  The code below compares the two algorithms for different values of λ and learning rates.  For each λ value, if α is selected optimally for it (or smaller), then the two algorithms perform virtually indentically.  If α is chosen larger than is optimal, however, then the λ-return algorithm is only a little worse whereas TD(λ) is much worse and may even be unstable.  This is not catastrophic for TD(λ) on this problem, as these higher parameter values are not what one would want to use anyway, but for other problems it can be a significant weakness. 
"""

# ╔═╡ 36d87b2a-6e1a-47b7-8af5-825d47e55eec
#=╠═╡
function run_random_walk_TDλ_estimation(mrp::TabularMRP, calc_err::Function, α, λ; num_episodes = 10, kwargs...)
	output = run_tabular_semi_gradient_TDλ(mrp, 1f0, λ, num_episodes, typemax(Int64); save_episode_errors = true, α = α, calculate_error = calc_err, kwargs...)
	return mean(output.error_history.episode_errors)
end
  ╠═╡ =#

# ╔═╡ b68f1171-6274-4d93-bf68-05b95cb5b2f8
#=╠═╡
run_random_walk_TDλ_estimation_trials(mrp::TabularMRP, calc_err, α, λ; num_trials = 100, kwargs...) = (1:num_trials |> Map(_ -> run_random_walk_TDλ_estimation(mrp, calc_err, α, λ; num_episodes = 10, kwargs...)) |> foldxt(+)) / num_trials
  ╠═╡ =#

# ╔═╡ 8b67e90f-3305-4702-b7a3-393055d15e46


# ╔═╡ da8c5f8b-5ab6-4a2b-93e8-18be4284b932
#=╠═╡
function tdλ_vs_offline_λ_error_random_walk(nstates, num_episodes; kwargs...)
	α_vec = vcat(Float32.(0.0:0.02:0.1), 0.15f0, Float32.(0.2:0.1:1.0))
	λ_vec = [0f0, 0.4f0, 0.8f0, 0.9f0, 0.95f0, 0.975f0, 0.99f0, 1f0]
	tabular_mrp = TabularRL.create_random_walk_distribution(nstates, -1f0, 1f0)
	mrp = StateMRP(tabular_mrp)
	c = (nstates + 1)/2
	v_true = [(s-c)/c for s in 1:nstates]
	calc_err(v) = sqrt(mean(i -> (v[i] - v_true[i])^2, eachindex(v_true)))
	get_α_line1(λ) = α_vec |> Map(α -> run_random_walk_offline_λ_estimation_trials(mrp, nstates, calc_err, α, λ; num_episodes = num_episodes, kwargs...)) |> collect
	get_α_line2(λ) = α_vec |> Map(α -> run_random_walk_TDλ_estimation_trials(tabular_mrp, calc_err, α, λ; num_episodes = num_episodes, kwargs...)) |> collect
	lines1 = λ_vec |> Map(λ -> get_α_line1(λ)) |> collect
	lines2 = λ_vec |> Map(λ -> get_α_line2(λ)) |> collect

	yaxis_lims = [minimum(minimum(x) for x in lines1) - 0.05, first(first(lines1))]
	
	traces1 = [scatter(x = α_vec, y = lines1[i], name = "λ = $λ", mode = "lines", line_shape = "spline", showlegend = false) for (i, λ) in enumerate(λ_vec)]
	p1 = plot(traces1, Layout(title = "Off-line λ-return algorithm", xaxis_title = L"α", yaxis_title = "Average RMS error over $nstates <br> states and first $num_episodes episodes", yaxis_range = yaxis_lims))

	traces2 = [scatter(x = α_vec, y = lines2[i], name = "λ = $λ", mode = "lines", line_shape = "spline") for (i, λ) in enumerate(λ_vec)]
	p2 = plot(traces2, Layout(title = "TD(λ)", xaxis_title = L"α", yaxis_range = yaxis_lims))

	@htl("""
	<div style = "display: flex; height: 500px;">
	$p1 
	$p2
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ 83a645f8-f806-4828-bc42-d24cfd26bad3
#=╠═╡
@bind fig_12_6_params PlutoUI.combine() do Child
md"""
### Figure 12.6

 $(Child(:n, NumberField(2:30, default = 19)))-state Random walk results: Performance of TD$$(\lambda)$$ alongside that of the off-line λ-return algorithm.  The two algorithms performed virtually identically at low (less than optimal) $$\alpha$$ values, but TD$$(\lambda)$$ was worse at high $$\alpha$$ values.  The importance of bootstrapping diminishes the smaller the random walk chain is.  These results are for the first $(Child(:num_episodes, NumberField(1:1000, default = 10))) episodes and averaged over 100 trials.  TD$$(\lambda)$$ has faster diverging behavior the longer the episode count.
"""
end
  ╠═╡ =#

# ╔═╡ 9a75dc05-883b-47a6-b8f0-ae0799c5fc19
#=╠═╡
tdλ_vs_offline_λ_error_random_walk(fig_12_6_params...)
  ╠═╡ =#

# ╔═╡ addedc75-375f-429f-8e2e-90ba2151dee0
md"""
Linear TD(λ) has been proved to converge in the on-policy case if the step-size parameter is reduced over time according to the usual conditions (2.7).  Just as discussed in Section 9.4, convergence is not to the minimum-error weight vector, but to a nearby weight vector that depends on $λ$.  The bound on solution quality presented in that section (9.14) can now be generalized to apply for any $\lambda$.  For the continuing discounted case,

$\overline{\text{VE}}(\mathbf{w}_\infty) \leq \frac{1-\gamma \lambda}{1-\gamma} \min_{\mathbf{w}} \overline{\text{VE}}(\mathbf{w}) \tag{12.8}$

That is, the asymptotic error is no more than $\frac{1-\gamma \lambda}{1-\gamma}$ times the smallest possible error.  As $\lambda$ approches 1, the bound approaches the minimum error (and it is loosest at $\lambda$ = 0).  In practice, however, $\lambda = 1$ is often the poorest choice as will be illustrated later in Figure 12.14
"""

# ╔═╡ 4f9cbb26-6c9b-458a-b7e6-102f0dbf64cb
#=╠═╡
function plot_bound()
	λs = LinRange(0, 1, 1000)
	γs = [0.5, 0.6, 0.7, 0.8, 0.9]
	
	traces = [begin
		ys = (1 .- (λs .* γ)) ./ (1 - γ)
		scatter(x = λs, y = ys, name = "γ = $γ")
	end
	for γ in γs]
	plot(traces, Layout(xaxis_title = "λ", yaxis_title = "Error Multiplicative Bound", width = 600))
end
  ╠═╡ =#

# ╔═╡ 5e1366cc-05cd-43b3-8a00-e56242a30d8f
#=╠═╡
plot_bound()
  ╠═╡ =#

# ╔═╡ e597a042-9c03-4d49-a48f-6dff39283c54
md"""
> ### *Exercise 12.3* 
> Some insight into how TD$(λ)$ can closely approximate the on-line λ-return algorithm can be gained by seeing that the latter’s error term (in brackets in (12.4)) can be written as the sum of TD errors (12.6) for a single fixed w. Show this, following the pattern of (6.6), and using the recursive relationship for the λ-return you obtained in Exercise 12.1.

The error term at step t is: $G_t^\lambda - \hat v(S_t, \mathbf{w_t})$

The TD error at step t is given by : $\delta_t \doteq R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w_t}) - \hat v(S_t, \mathbf{w_t})$

The recursive relationship for the λ-return is given by: $G_t^\lambda = R_{t+1} + \gamma \left [ (1-\lambda)\hat v(S_{t+1}, \mathbf{w}_t) + \lambda G_{t+1}^\lambda \right ]$

Our goal is to show that the error term can be written as a sum of TD errors.  To start, we should express the error recursively and try to replace part of the resulting expression with the TD error:

$\begin{flalign}
\text{VE}_t &= G_t^\lambda - \hat v(S_t, \mathbf{w_t}) \\
&= R_{t+1} + \gamma \left [ (1-\lambda)\hat v(S_{t+1}, \mathbf{w}_t) + \lambda G_{t+1}^\lambda \right ] - \hat v(S_t, \mathbf{w_t}) \tag{recursive relationship} \\
&= R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w}_t) - \hat v(S_t, \mathbf{w_t}) + \gamma \lambda \left [ G_{t+1}^\lambda - \hat v(S_{t+1}, \mathbf{w}_t) \right ] \tag{grouping terms} \\
&= \delta_t + \gamma \lambda \left [ G_{t+1}^\lambda - \hat v(S_{t+1}, \mathbf{w}_t) \right ] \tag{TD error definition}\\
&= \delta_t + \gamma \lambda \text{VE}_{t+1} \tag{value error expression}\\
&= \delta_t + \gamma \lambda \left [ \delta_{t+1} + \gamma \lambda \text{VE}_{t+2} \right ] \tag{using recurssion once}\\
&= \delta_t + \gamma \lambda \delta_{t+1} +  (\gamma \lambda)^2 \text{VE}_{t+2} \tag{grouping δ terms}\\
\end{flalign}$

For a continuing task this recurssion can be applied repeatedly to yield:

$\text{VE}_t = \sum_{n = 0}^\infty (\gamma \lambda)^n \delta_{t+n}$

For episodic tasks the sum will be finite and the final TD error $\delta_{T-1} = R_{T} + \gamma \hat v(S_{T}, \mathbf{w_{T-1}}) - \hat v(S_{T-1}, \mathbf{w_{T-1}}) = R_{T} - \hat v(S_{T-1}, \mathbf{w_{T-1}})$ using the fact that the value estimate at the terminal state is always 0.  So the sum will be cut off at $n + t = T - 1 \implies n = T - 1 - t$.
"""

# ╔═╡ 0c6ebdeb-77f4-44f0-9bf3-c539d54bcaec
md"""
> ### *Exercise 12.4* 
> Use your result from the preceding exercise to show that, if the weight updates over an episode were computed on each step but not actually used to change the weights (w remained fixed), then the sum of TD$(λ)$’s weight updates would be the same as the sum of the off-line λ-return algorithm’s updates.

The TD$(λ)$ updates are given by: 

$\begin{flalign}
\mathbf{z_t} &\doteq \gamma \lambda \mathbf{z_{t-1}} + \nabla \hat v(S_{t}, \mathbf{w_{t}}) \\
\mathbf{w_{t+1}} &\doteq \mathbf{w_t} + \alpha \delta_t \mathbf{z_t} \\
\end{flalign}$.  

Let's write down all of the updates that will occur from t = 0 assuming the weights themselves are held constant the entire episode.

$\begin{flalign}
\mathbf{z_0} &= \nabla \hat v(S_{0}, \mathbf{w}) \\
\mathbf{w} \text{ update 1} &= \alpha \delta_0 \nabla \hat v(S_{0}, \mathbf{w})\\
\mathbf{z_1} &= \gamma \lambda \nabla \hat v(S_{0}, \mathbf{w}) + \nabla \hat v(S_{1}, \mathbf{w}) \\
\mathbf{w} \text{ update 2} &= \alpha \delta_1 \mathbf{z_1} \\
\mathbf{z_2} &= \gamma \lambda \left [ \gamma \lambda \nabla \hat v(S_{0}, \mathbf{w}) + \nabla \hat v(S_{1}, \mathbf{w}) \right ] + \nabla \hat v(S_{2}, \mathbf{w}) \\
&= (\gamma \lambda)^2 \nabla \hat v(S_{0}, \mathbf{w}) +  \gamma \lambda \nabla \hat v(S_{1}, \mathbf{w}) + \nabla \hat v(S_{2}, \mathbf{w}) \\
&= \sum_{n = 0}^2 (\gamma \lambda)^{2-n} \nabla \hat v(S_{n}, \mathbf{w}) \\
\mathbf{w} \text{ update 3} &= \alpha \delta_2 \mathbf{z_2} \\
& \vdots \\
\mathbf{z_t} &= \sum_{n = 0}^t (\gamma \lambda)^{t-n} \nabla \hat v(S_{n}, \mathbf{w}) \\
\mathbf{w} \text{ update t + 1} &= \alpha \delta_{t} \mathbf{z_{t}} \\
\end{flalign}$

Let's group the coefficients of the TD errors, i.e. $\delta_0$, $\delta_1$, ....  From the weight updates it is clear that a given TD error will only occur once per step with coefficients:

$\delta_t \text{ coefficient} = \alpha \mathbf{z_t} = \alpha \sum_{n = 0}^t (\gamma \lambda)^{t-n} \nabla \hat v(S_{n}, \mathbf{w})$

Now we can compare these coefficients to the off-line  λ-return updates.  Those weight updates are given by:

$\mathbf{w_{t+1}} \doteq \mathbf{w_t} + \alpha \left [ G_t^\lambda - \hat v(S_t, \mathbf{w_t}) \right ] \nabla \hat v(S_t, \mathbf{w_t}) = \mathbf{w_t} + \alpha \text{VE}_t \nabla \hat v(S_t, \mathbf{w_t})$

From the previous exercise we expressed the term in the brackets as follows for an episodic task ending at step T:

$\text{VE}_t = \sum_{n=0}^{T - 1 - t} (\gamma \lambda)^n \delta_{t+n}$

Assuming the weights are not updated until the end of an episode, the contribution per step is given by:

$\begin{flalign}
\mathbf{w} \text{ update t + 1} &= \alpha \left [ \sum_{n=0}^{T - 1 - t} (\gamma \lambda)^n \delta_{t + n} \right ] \nabla \hat v(S_{t}, \mathbf{w}) \\
\end{flalign}$

Writing out each update and aligning terms according to $\delta_t$ can reveal the coefficients of each TD error in the total update sum.  Consider $\delta_0$ which will only occur when $t = 0$ and $n = 0$.  This only occurs once for the update at $t = 0$.  Similarly $\delta_1$ will have terms from $t = 0$ and $n = 1$ but also from $t = 1$ and $n = 0$.  This pattern will continue for every $\delta_t$ resulting in $t+1$ terms for each t.

$\begin{flalign}
\mathbf{w} \text{ update 1} &= \alpha \left [ \sum_{n=0}^{T - 1} (\gamma \lambda)^n \delta_{n} \right ] \nabla \hat v(S_{0}, \mathbf{w}) \\
&= \delta_0 \left [ \alpha \nabla \hat v(S_{0}, \mathbf{w}) \right ] + \delta_1 \left [ \alpha \nabla \hat v(S_{0}, \mathbf{w}) (\gamma \lambda) \right ] + \delta_2 \left [ \alpha \nabla \hat v(S_{0}, \mathbf{w}) (\gamma \lambda)^2 \right ] + \cdots \\
\mathbf{w} \text{ update 2} &= \alpha \left [ \sum_{n=0}^{T - 2} (\gamma \lambda)^n \delta_{1+n} \right ] \nabla \hat v(S_{1}, \mathbf{w}) \\
&= \delta_1 \left [ \alpha \nabla \hat v(S_{1}, \mathbf{w}) \right ] + \delta_2 \left [ \alpha \nabla \hat v(S_{1}, \mathbf{w}) (\gamma \lambda) \right ] + \delta_3 \left [ \alpha \nabla \hat v(S_{1}, \mathbf{w}) (\gamma \lambda)^2 \right ] + \cdots \\
\mathbf{w} \text{ update 3} &= \alpha \left [ \sum_{n=0}^{T - 3} (\gamma \lambda)^n \delta_{2+n} \right ] \nabla \hat v(S_{2}, \mathbf{w}) \\
&= \delta_2 \left [ \alpha \nabla \hat v(S_{2}, \mathbf{w}) \right ] + \delta_3 \left [ \alpha \nabla \hat v(S_{2}, \mathbf{w}) (\gamma \lambda) \right ] + \delta_4 \left [ \alpha \nabla \hat v(S_{2}, \mathbf{w}) (\gamma \lambda)^2 \right ] + \cdots \\
\end{flalign}$

The pattern is already evident and the coefficient for each $\delta_t$ can be read off diagonally.  As an example, here are the first 3 coefficients and the resulting general pattern:

$\begin{flalign}
\delta_0 \text{ coefficient} &= \alpha \nabla \hat v(S_{0}, \mathbf{w})\\ 
\delta_1 \text{ coefficient} &= \alpha \nabla \hat v(S_{0}, \mathbf{w}) (\gamma \lambda) +  \alpha \nabla \hat v(S_{1}, \mathbf{w}) \\ 
\delta_2 \text{ coefficient} &= \alpha \nabla \hat v(S_{0}, \mathbf{w}) (\gamma \lambda)^2 + \alpha \nabla \hat v(S_{1}, \mathbf{w}) (\gamma \lambda) + \alpha \nabla \hat v(S_{2}, \mathbf{w}) \\ 
\vdots \\
\delta_t \text{ coefficient} &= \alpha \sum_{n = 0}^t (\gamma \lambda)^{t - n} \nabla \hat v(S_{n}, \mathbf{w})
\end{flalign}$

But this coefficient is the same as we got previously for the TD(λ) weight updates, so we've shown that if weight updates are delayed unti the end of an episode both methods will perform exactly the same weight updates.
"""

# ╔═╡ 27f535a4-2245-45aa-aefa-4c0fc6bb218d
md"""
## 12.3 n-step Truncated λ-return Methods

The off-line λ-return algorithm is an important ideal, but it is of limited utility because it uses the λ-return (12.2), which is not known until the end of the episode.  In the continuing case, the λ-return is technically never known, as it depends on n-step returns for arbitrarility large n, and thus on rewards arbitrariliy far in the future.  However, the dependence becomes weaker for longer-delayed rewards, falling by $\gamma \lambda$ for each step of delay.  A natural approximation, then, would be to truncate the sequence after some number of steps.  Our existing notion of n-step returns provides a natural way to do this in which the missing rewards replaced with estimated values.

Define the *truncated λ-return* for time t, given data only up to some later horizon, h, as 

$\begin{flalign}
G_{t:h}^\lambda \doteq (1-\lambda) \sum_{n=1}^{h-t-1} \lambda ^{n-1} G_{t:t+n} + \lambda^{h-t-1} G_{t:h}, \hspace{5mm} 0 \leq t < h \leq T \tag{12.9}
\end{flalign}$

If you compare this equation with the λ-return (12.3), it is clear that the horizon $h$ is playing the same role as was previously played by $T$, the time of termination.  Whereas in the λ-return there is a residual weight given to the conventional return $G_t$, here it is given to the longest available n-step return, $G_{t:h}$ (Figure 12.2).

The truncated λ-return immediately gives rise to a family of n-step λ-return algorithms similar to the n-step methods of Chapter 7.  In all of these algorithms, updates are delayed by n steps and only take into account the first n rewards, but now all the k-step returns are included for $1 \leq k \leq n$ (whereas the earlier n-step algorithms used only the n-step return), weighted geometrically as in Figure 12.2.  In the state-value case, this family of algorithms is known as Truncated TD$(\lambda)$, or TTD$(\lambda)$.  The compound backup diagram, shown in figure 12.7, is similar to that for the TD$(\lambda)$ (Figure 12.1) except that the longest component update is at most n steps rather than always going all the way to the end of the episode.

The weight updates for Truncated TD$(λ)$ or TTD$(λ)$ is given by:

$\mathbf{w}_{t+n} \doteq \mathbf{w}_{t+n-1} + \alpha \left [ G_{t:t+n}^\lambda - \hat v(S_t, \mathbf{w}_{t+n-1}) \right ] \nabla \hat v (S_t, \mathbf{w}_{t+n-1})$

where the maximum number of steps in the future to consider returns for is n.  Much as in *n*-step TD methods, no updates are made on the first n-1 steps of each episode, and n-1 additional updates are made upon termination.  Efficient imlementation relies on the fact that the *k*-step λ-return can be written exactly as 

$\begin{flalign}
G_{t:t+k}^\lambda = \hat v(S_t, \mathbf{w}_{t-1}) + \sum_{i=t}^{t+k-1} (\gamma \lambda)^{i-t} \delta_i ^\prime \tag{12.10}
\end{flalign}$

where 

$\delta_i ^\prime \doteq R_{t+1} + \gamma \hat v (S_{t+1}, \mathbf{w}_t) - \hat v(S_t, \mathbf{w}_{t-1})$

"""

# ╔═╡ e1e9f2eb-4751-4f5c-aa4d-a0cf75e193b2
md"""
> ### *Exercise 12.5* 
> Several times in this book (often in exercises) we have established that returns can be written as sums of TD errors if the value function is held constant.  Why is (12.10) another instance of this?  Prove (12.10).

To prove (12.10) let's return to the definition of $G_{t:k}^\lambda$ given in (12.9) and compare it to (12.10)

$\begin{flalign}
G_{t:h}^\lambda &\doteq (1-\lambda) \sum_{n=1}^{h-t-1} \lambda ^{n-1} G_{t:t+n} + \lambda^{h-t-1} G_{t:h}, \hspace{5mm} 0 \leq t < h \leq T \tag{12.9} \\
G_{t:k}^\lambda &= \hat v(S_t, \mathbf{w}_{t-1}) + \sum_{i=t}^{t+k-1} (\gamma \lambda)^{i-t} \delta_i ^\prime \tag{12.10}\\
\delta_t^\prime & \doteq R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w}_t) - \hat v(S_t, \mathbf{w}_{t-1})
\end{flalign}$

Also note that from (12.9) we can conclude:

$G_{t+1:h}^\lambda = (1-\lambda) \sum_{n=1}^{h-t-2} \lambda ^{n-1} G_{t+1:t+1+n} + \lambda^{h-t-2} G_{t+1:h}$
"""

# ╔═╡ 531263cf-274e-4a64-932f-821e8583a316
md"""
$\begin{flalign}
G_{t:h}^\lambda &= (1-\lambda) \sum_{n=1}^{h-t-1} \lambda ^{n-1} (R_{t+1} + \gamma G_{t+1:t+n}) + \lambda^{h-t-1} G_{t:h}\\
 &= (1-\lambda) \left [ R_{t+1} \sum_{n=1}^{h-t-1} \lambda ^{n-1}  + \gamma \hat v(S_{t+1}, \mathbf{w}) + \gamma \sum_{n=2}^{h-t-1} \lambda ^{n-1} G_{t+1:t+n} \right ] + \lambda^{h-t-1} G_{t:h} \tag{separating sum}\\
 &= (1-\lambda) \left [ R_{t+1} \frac{\lambda^{h-t-1} - 1}{\lambda - 1}  + \gamma \hat v(S_{t+1}, \mathbf{w}) + \gamma \sum_{n=2}^{h-t-1} \lambda ^{n-1} G_{t+1:t+n} \right ] + \lambda^{h-t-1} G_{t:h} \tag{simplifying sum}\\
 &= R_{t+1}(1 - \lambda^{h-t-1}) + \gamma (1-\lambda) \left [ \hat v(S_{t+1}, \mathbf{w}) + \sum_{n=2}^{h-t-1} \lambda ^{n-1} G_{t+1:t+n} \right ] + \lambda^{h-t-1} G_{t:h}\\
\end{flalign}$
"""

# ╔═╡ 0df08e27-18d3-4f2c-a7e1-75674418ba01
md"""
Let's look at just the sum expression and reindex by m = n - 1

$\begin{flalign}
& \sum_{m=1}^{h-t-2} \lambda^m G_{t+1:t+m+1} \\
& \lambda \sum_{m=1}^{h-t-2} \lambda^{m-1} G_{t+1:t+m+1} \tag{dividing sum by λ} \\ 
& \lambda \sum_{n=1}^{h-t-2} \lambda^{n-1} G_{t+1:t+n+1} \tag{renaming m to n} \\ 
\end{flalign}$

Also let's rewrite the final term as follows:
$\lambda^{h-t-1} G_{t:h} = \lambda^{h-t-1} (R_{t+1} + \gamma G_{t+1:h})$
"""

# ╔═╡ c65dc168-9fa4-4e1b-af39-02f80c9ec0e3
md"""
Using this new sum expression and the final term we can group some terms at the end to get a recurrance relationship.

$\begin{flalign}
G_{t:h}^\lambda & = R_{t+1}(1 - \lambda^{h-t-1}) + \gamma (1-\lambda) \left [ \hat v(S_{t+1}, \mathbf{w}) + \lambda \sum_{n=1}^{h-t-2} \lambda^{n-1} G_{t+1:t+n+1} \right ] + \lambda^{h-t-1} (R_{t+1} + \gamma G_{t+1:h})\\
& = R_{t+1} + \gamma (1-\lambda) \left [ \hat v(S_{t+1}, \mathbf{w}) + \lambda \sum_{n=1}^{h-t-2} \lambda^{n-1} G_{t+1:t+n+1} \right ] + \gamma \lambda^{h-t-1}G_{t+1:h} \tag{cancelling out R terms}\\
& = R_{t+1} + \gamma (1-\lambda)\hat v(S_{t+1}, \mathbf{w}) + \gamma \lambda \left [ (1 - \lambda) \sum_{n=1}^{h-t-2} \lambda^{n-1} G_{t+1:t+n+1} + \lambda^{h-t-2}G_{t+1:h} \right ] \tag{grouping terms}\\
& = R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w}) + \gamma \lambda \left [ G_{t+1:h}^\lambda - \hat v(S_{t+1}, \mathbf{w}) \right ] \tag{using recurrence relation for G}\\
& = R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w}) + \gamma \lambda \left [ R_{t+2} + \gamma \hat v(S_{t+2}, \mathbf{w}) + \gamma \lambda \left [ G_{t+2:h}^\lambda - \hat v(S_{t+2}, \mathbf{w}) \right ]- \hat v(S_{t+1}, \mathbf{w}) \right ] \tag{noticing recurssion}\\
& = R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w}) + \gamma \lambda \left [ R_{t+2} + \gamma \hat v(S_{t+2}, \mathbf{w}) - \hat v(S_{t+1}, \mathbf{w}) + \gamma \lambda \left [ G_{t+2:h}^\lambda - \hat v(S_{t+2}, \mathbf{w}) \right ] \right ] \tag{grouping terms} \\
& \vdots \\
& \text{when will this sum terminate?  The horizon return is only well defined up to t = h - 1 }\\
G_{h-1:h}^\lambda &= G_{h-1:h} = R_{h} + \gamma \hat v(S_h, \mathbf{w})\\  
& \text{so the final reward subscript is h which can be achieved with the following sum}\\
& = R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w}) + \sum_{i=t+1}^{h-1} (\gamma \lambda)^{i-t} \left [ R_{i+1} + \gamma \hat v(S_{i+1}, \mathbf{w}) - \hat v(S_{i}, \mathbf{w}) \right ]\\
\end{flalign}$
"""

# ╔═╡ 8d41a846-3a12-4e32-bc1a-50be12629eb2
md"""
Going back to equation (12.10), let's see how the terms line up noticing the cancelation of the estimator at state t and keeping the parameters fixed.

$\begin{flalign}
G_{t:t+k}^\lambda &= R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w}) + \gamma \lambda \left [ R_{t+2} + \gamma \hat v(S_{t+2}, \mathbf{w}) - \hat v(S_{t+1}, \mathbf{w}) \right ] + \cdots + (\gamma \lambda)^{k-1} \left [ R_{t+k} + \gamma \hat v(S_{t+k}, \mathbf{w}) - \hat v(S_{t+k-1}, \mathbf{w}) \right ]  
\end{flalign}$

If we compare to the expression we have $h = t+k$, the sum terminates at $h-1=t+k-1$.  The starting terms are the same and the ending terms also share the same exponent of $k-1$ and a reward term of $R_{t+k}$, so this proves (12.10).  The only difference is the expression in the book begins the sum at $i = t$ instead of $i = t+1$ so the added terms are different.  Either way an addtional term outside the sum is required due to the starting point.

"""

# ╔═╡ 8f211c8d-6b20-48de-ae91-435a977c930c
md"""
## 12.4 Redoing Updates: Online λ-return Algorithm

Choosing the truncation parameter n in Truncated TD$(\lambda)$ involves a tradeoff.  n should be large so that hte method closely approximates the off-line λ-return algorithm, but it should also be small so that the updates can be made sooner and can influence behavior sooner.  Can we get the best of both?  Well, yes, in principle we can, albeit at the cost of computational complexity.

The idea is that, on each time step as you gather a new increment of data, you go back and redo all the updates since the beginning of the current episode.  The new updates will be better than the ones you previously made because now they can take into account the time step's new data.  That is, the updates are always towards an n-step truncated λ-return target, but they always use the latest horizon.  In each pass over that episode you can use a slightly longer horizon and obtain slightly better results.  Recall that the truncated λ-return is defined in (12.9) as:


"""

# ╔═╡ 21df1418-8808-42f1-9fcd-26048705c5ce
md"""
## 12.5 True Online TD(λ)
The online λ-return algorithm just presented is currently the best performing temporal-difference algorithm. It is an ideal which online TD$(λ)$ only approximates.  (why is this the case?  I thought TD$(λ)$ was equivalent to the full λ return, they mentioned in figures that at higher learning rates it can be unstable though.  True online TD$(λ)$ doesn't have that problem.  In the plot there isn't even a horizon anymore but this was truncated.  So what happened to the cutoff point?)  So at each step in the episode, the target is the n-step λ return for that step so there is no selection of the horizon.  The largest possible horizon for every previous state is always being used in the update target.

In the linear case for which $\hat v(s_\mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s)$, then we arrive at the true online TD$(λ)$ algorithm:

$\begin{flalign}
\mathbf{w}_{t+1} & \doteq \mathbf{w}_t + \alpha \delta_t \mathbf{z}_t + \alpha (\mathbf{w}_t^\top \mathbf{x}_t - \mathbf{w}_{t-1}^\top \mathbf{x}+t)(\mathbf{z}_t - \mathbf{x}_t) \\
\mathbf{x}_t & \doteq \mathbf{x}(S_t) \\
\mathbf{z}_t & \doteq \gamma \lambda \mathbf{z}_{t-1} + (1 - \alpha\gamma\lambda \mathbf{z}_{t-1}^\top \mathbf{x}_t)\mathbf{x}_t \tag{12.11}

\end{flalign}$
"""

# ╔═╡ 7f8fb89d-1a2e-4acd-9118-2ce3d3874341
begin
	#note that this function will modify both parameters and the state representation vector as well as some of the keyword arguments
	function true_online_TDλ!(parameters::P, state_representation::X, initialize_state::Function, transition::Function, isterm::Function, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, update_state_representation!::Function; α = one(T)/10, calculate_error::Function = params -> zero(T), ∇v::P = copy(parameters), z::P = copy(parameters), save_step_errors::Bool = false, save_episode_errors::Bool = false, epkwargs...) where {P, X, T<:Real}
		s = initialize_state()
		update_state_representation!(state_representation, s)
		v̂ = dot(state_representation, parameters)
		∇v .= state_representation
		(r, s′) = transition(s)
		if isterm(s′)
			v̂′ = zero(T)
		else
			update_state_representation!(state_representation, s′)
			v̂′ = dot(state_representation, parameters)
		end
		ep = 1
		step = 1
		episode_error_history = Vector{T}()
		step_error_history = Vector{T}()
		
		#initialize eligibility vector to 0
		z .= zero(T)
		v_old = zero(T)
		
		while (ep <= max_episodes) && (step <= max_steps)
			z .= (γ*λ .* z) .+ (one(T) - α*γ*λ*dot(z, ∇v)) .* ∇v
			δ = r + γ*v̂′ - v̂
			a = v̂ - v_old
			parameters .+= α .* ((δ + a) .* z .- (a .* ∇v))
	
			save_step_errors && push!(step_error_history, calculate_error(parameters))
	
			if isterm(s′)
				s = initialize_state()
				update_state_representation!(state_representation, s)
				#reset eligibility vector to 0 at the start of a new episode
				z .= zero(T)
				v_old = zero(T)
				ep += 1
				save_episode_errors && push!(episode_error_history, calculate_error(parameters))
			else
				s = s′
				v_old = v̂′
			end

			#note that the state representation here will be for s on the next step
			v̂ = dot(state_representation, parameters)
			∇v .= state_representation
			
			(r, s′) = transition(s)
			
			if isterm(s′)
				v̂′ = zero(T)
			else
				update_state_representation!(state_representation, s′)
				v̂′ = dot(state_representation, parameters)
			end
			step += 1
		end
		return (episode_errors = episode_error_history, step_errors = step_error_history)
	end

	#when evaluating an MRP, there is no policy and the transition is just from the environment
	true_online_TDλ!(parameters::P, state_representation::X, mrp::StateMRP, args...; kwargs...) where {P, X} = true_online_TDλ!(parameters, state_representation, mrp.initialize_state, s -> mrp.ptf(s), mrp.isterm, args...; kwargs...)

	#when evaluating an MDP, there is a policy and the transition uses it to select actions
	true_online_TDλ!(parameters::P, state_representation::X, mdp::StateMDP, π::Function, args...; kwargs...) where {P, X} = true_online_TDλ!(parameters, state_representation, mdp.initialize_state, s -> mdp.ptf(s, π), mdp.isterm, args...; kwargs...)
end

# ╔═╡ 34a28cfa-bf18-4dcf-8cf4-f6e9031d6fc2
function true_online_TDλ(problem, state_representation::Vector{T}, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, update_state_representation!::Function; parameters::Vector{T} = zeros(T), kwargs...) where T<:Real
	errors = true_online_TDλ!(parameters, state_representation, problem..., γ, λ, max_episodes, max_steps, update_state_representation!)

	function v!(x, s)
		update_state_representation!(x, s)
		dot(parameters, x)
	end

	v(s; x = copy(state_representation)) = v!(x, s)
	return (value_function = v, error_history = errors)
end

# ╔═╡ 4dc03d69-c90c-4d64-bb8e-600ecfe30eb8
begin
	#note that this function will modify both parameters and the state representation vector as well as some of the keyword arguments
	function true_online_tabular_TDλ!(parameters::Vector{T}, initialize_state::Function, transition::Function, termstates::BitVector, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; α = one(T)/10, calculate_error::Function = params -> zero(T), z::Vector{T} = copy(parameters), save_step_errors::Bool = false, save_episode_errors::Bool = false, epkwargs...) where T<:Real
		i_s = initialize_state()
		v̂ = parameters[i_s]
		(r, i_s′) = transition(i_s)
		if termstates[i_s′]
			v̂′ = zero(T)
		else
			v̂′ = parameters[i_s′]
		end
		ep = 1
		step = 1
		episode_error_history = Vector{T}()
		step_error_history = Vector{T}()
		
		#initialize eligibility vector to 0
		z .= zero(T)
		v_old = zero(T)
		
		while (ep <= max_episodes) && (step <= max_steps)
			z .*= γ*λ
			z[i_s] += one(T) - α*γ*λ*z[i_s] 
			δ = r + γ*v̂′ - v̂
			a = v̂ - v_old
			parameters .+= α .* (δ + a) .* z
			parameters[i_s] -= α*a
	
			save_step_errors && push!(step_error_history, calculate_error(parameters))
	
			if termstates[i_s′]
				i_s = initialize_state()
				#reset eligibility vector to 0 at the start of a new episode
				z .= zero(T)
				v_old = zero(T)
				ep += 1
				save_episode_errors && push!(episode_error_history, calculate_error(parameters))
			else
				i_s = i_s′
				v_old = v̂′
			end

			#note that the state representation here will be for s on the next step
			v̂ = parameters[i_s]
			(r, i_s′) = transition(i_s)
			
			if termstates[i_s′]
				v̂′ = zero(T)
			else
				v̂′ = parameters[i_s′]
			end
			step += 1
		end
		return (episode_errors = episode_error_history, step_errors = step_error_history)
	end

	#when evaluating an MRP, there is no policy and the transition is just from the environment
	true_online_tabular_TDλ!(parameters, mrp::TabularMRP, args...; kwargs...) = true_online_tabular_TDλ!(parameters, mrp.initialize_state_index, s -> mrp.ptf(s), mrp.terminal_states, args...; kwargs...)

	#when evaluating an MDP, there is a policy and the transition uses it to select actions
	true_online_tabular_TDλ!(parameters, mdp::TabularMDP, π, args...; kwargs...) = true_online_tabular_TDλ!(parameters, mdp.initialize_state_index, s -> mdp.ptf(s, π), mdp.terminal_states, args...; kwargs...)

	function true_online_tabular_TDλ(env::Union{TabularMRP{T, S, P, F}, TabularMDP{T, S, A, P, F}}, args...; parameters = zeros(T, length(env.states)), kwargs...) where {T<:Real, S, A, P, F}
		@assert length(parameters) == length(env.states)
		errors = true_online_tabular_TDλ!(parameters, env, args...; kwargs...)
		return (parameters = parameters, error_history = errors)
	end
end

# ╔═╡ d9f89b2c-8df8-415a-a0a8-21744be88cec
#=╠═╡
function run_random_walk_true_online_TDλ_estimation(mrp::TabularMRP, calc_err::Function, α, λ; num_episodes = 10, kwargs...)
	output = true_online_tabular_TDλ(mrp, 1f0, λ, num_episodes, typemax(Int64); save_episode_errors = true, α = α, calculate_error = calc_err, kwargs...)
	return mean(output.error_history.episode_errors)
end
  ╠═╡ =#

# ╔═╡ 00015316-0d9d-41ce-a001-09aede10049c
#=╠═╡
run_random_walk_true_online_TDλ_estimation_trials(mrp::TabularMRP, calc_err, α, λ; num_trials = 100, kwargs...) = (1:num_trials |> Map(_ -> run_random_walk_true_online_TDλ_estimation(mrp, calc_err, α, λ; num_episodes = 10, kwargs...)) |> foldxt(+)) / num_trials
  ╠═╡ =#

# ╔═╡ 9c287f15-a78a-4e3f-a0b7-c901874b6cca
#=╠═╡
function true_online_tdλ_vs_offline_λ_error_random_walk(nstates, num_episodes; kwargs...)
	α_vec = vcat(Float32.(0.0:0.02:0.1), 0.15f0, Float32.(0.2:0.1:1.0))
	λ_vec = [0f0, 0.4f0, 0.8f0, 0.9f0, 0.95f0, 0.975f0, 0.99f0, 1f0]
	tabular_mrp = TabularRL.create_random_walk_distribution(nstates, -1f0, 1f0)
	mrp = StateMRP(tabular_mrp)
	c = (nstates + 1)/2
	v_true = [(s-c)/c for s in 1:nstates]
	calc_err(v) = sqrt(mean(i -> (v[i] - v_true[i])^2, eachindex(v_true)))
	get_α_line1(λ) = α_vec |> Map(α -> run_random_walk_offline_λ_estimation_trials(mrp, nstates, calc_err, α, λ; num_episodes = num_episodes, kwargs...)) |> collect
	get_α_line2(λ) = α_vec |> Map(α -> run_random_walk_true_online_TDλ_estimation_trials(tabular_mrp, calc_err, α, λ; num_episodes = num_episodes, kwargs...)) |> collect
	lines1 = λ_vec |> Map(λ -> get_α_line1(λ)) |> collect
	lines2 = λ_vec |> Map(λ -> get_α_line2(λ)) |> collect

	yaxis_lims = [minimum(minimum(x) for x in lines1) - 0.05, first(first(lines1))]
	
	traces1 = [scatter(x = α_vec, y = lines1[i], name = "λ = $λ", mode = "lines", line_shape = "spline", showlegend = false) for (i, λ) in enumerate(λ_vec)]
	p1 = plot(traces1, Layout(title = "Off-line λ-return algorithm", xaxis_title = L"α", yaxis_title = "Average RMS error over $nstates <br> states and first $num_episodes episodes", yaxis_range = yaxis_lims))

	traces2 = [scatter(x = α_vec, y = lines2[i], name = "λ = $λ", mode = "lines", line_shape = "spline") for (i, λ) in enumerate(λ_vec)]
	p2 = plot(traces2, Layout(title = "True Online TD(λ)", xaxis_title = L"α", yaxis_range = yaxis_lims))

	@htl("""
	<div style = "display: flex; height: 500px;">
	$p1 
	$p2
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ ccf903ab-1509-4c53-bfe0-435152ecea4b
#=╠═╡
@bind fig_12_8_params PlutoUI.combine() do Child
md"""
### Figure 12.8

 $(Child(:n, NumberField(2:30, default = 19)))-state Random walk results: Performance of True Online TD$$(\lambda)$$ alongside that of the off-line λ-return algorithm.  The off-line algorithm should be an ideal but the online version outperforms it on this problem.  These results are for the first $(Child(:num_episodes, NumberField(1:1000, default = 10))) episodes and averaged over 100 trials.
"""
end
  ╠═╡ =#

# ╔═╡ f7cc35c5-c1ed-440d-8723-8c1b8b966b8f
#=╠═╡
true_online_tdλ_vs_offline_λ_error_random_walk(fig_12_8_params...)
  ╠═╡ =#

# ╔═╡ b36896b1-6802-48e1-8cd3-f08bf3b99e3e
md"""
## 12.6 Dutch Traces in Monte Carlo Learning

It can be shown that the linear MC algorithm can be used to drive an equivalent yet computationally cheapter backward-view algorithm using dutch traces.  This equivalence gives some flavor of teh proof of equivalence of true online TD(λ) and the online λ-return algorithm, but is much simpler.

The linear version of gradient Monte Carlo prediction algorithm makes the following sequence updates, one for each time step of the episode:

$\begin{flalign}
\mathbf{w}_{t+1} & \doteq \mathbf{w}_t + \alpha \left [ G - \mathbf{w}_t^\top \mathbf{x}_t \right ] \mathbf{x}_t, \hspace{4 mm} 0 \leq t < T \tag{12.13} \\
\end{flalign}$

To simplify assume that the return $G$ is a single reward received at the end of teh episode and that there is no discounting.  In this case the update is also known as the Least Mean Square (LMS) rule.  As a Monte Carlo algirithm, all the updates depend on teh final reward/return, so none can be made until the end of the episode.  We seek an implementation of this algorithm with computational advantages by doing some computation during each step of the episode.

$\begin{flalign}
\mathbf{w}_T & = a_{t-1} + \alpha G\mathbf{z}_{T-1} \tag{12.14} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} + (1 - \alpha \mathbf{z}_{t-1}^\top \mathbf{x}_t)\mathbf{x}_t \text{,  with } \mathbf{z}_0 = \mathbf{x}_0 \\
\mathbf{a}_t & = \mathbf{a}_{t-1} - \alpha \mathbf{x}_t \mathbf{x}_t^\top \mathbf{a}_{t-1} \text{,  with } \mathbf{a}_0 = \mathbf{w}_0
\end{flalign}$


This computation uses the dutch trace for the case of $\gamma \lambda = 1$.  This is not specific to temporal-difference learning but is useful any time long term predictions are computed in an efficient manner.
"""

# ╔═╡ 0086dc4a-e0ba-43f0-a721-296cd50e1a76
md"""
## 12.7 Sarsa(λ)
"""

# ╔═╡ 21479229-c2ad-425f-98bb-77717ab40b02
#given a state_action_value function, an exploration parameter ϵ, and a state index, produce a sampled action according to the ϵ-greedy policy
function select_action!(action_values::Vector{T}, state_action_values::Matrix{T}, ϵ::T, i_s::Integer) where T<:Real
	for i_a in eachindex(action_values)
		action_values[i_a] = state_action_values[i_a, i_s]
	end
	make_ϵ_greedy_policy!(action_values; ϵ = ϵ)
	sample_action(action_values)
end

# ╔═╡ cf4fb06d-98e5-47f0-9e9a-0f89d83ccf1f
begin
	#tabular problem where the parameters are just the state action values and each state action pair only has one active feature
	function sarsa_λ!(state_action_values::Matrix{T}, mdp::TabularMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; α = one(T)/10, ϵ = one(T) / 10, z::Matrix{T} = copy(state_action_values), action_values::Vector{T} = zeros(T, length(mdp.actions)), save_episode_steps::Bool = false, save_step_rewards::Bool = false, use_accumulating_traces::Bool = false, use_dutch_traces::Bool = false, epkwargs...) where {T<:Real}
		#set the state action values of all terminal states to 0
		for i_s in eachindex(mdp.states)
			if mdp.terminal_states[i_s]
				state_action_values[:, i_s] .= zero(T)
			end
		end

		#initialize records
		episode_step_history = Vector{T}()
		step_rewards = Vector{T}()

		#initialize episode
		i_s = mdp.initialize_state_index()
		i_a = select_action!(action_values, state_action_values, ϵ, i_s)
		z .= zero(T)
		ep = 1
		step = 1
		
		while (ep <= max_episodes) && (step <= max_steps)
			#take action and observe transition
			(r, i_s′) = mdp.ptf(i_s, i_a)
			
			save_step_rewards && push!(step_rewards, r)
			
			δ = r - state_action_values[i_a, i_s]
			z[i_a, i_s] = one(T) + use_accumulating_traces*!use_dutch_traces*z[i_a, i_s] - use_dutch_traces*!use_accumulating_traces*α*γ*λ*z[i_a, i_s]

			if !mdp.terminal_states[i_s′]
				i_a′ = select_action!(action_values, state_action_values, ϵ, i_s′)
				δ += γ*state_action_values[i_a′, i_s′]
			end

			state_action_values .+= α*δ .* z

			if mdp.terminal_states[i_s′]
				i_s = mdp.initialize_state_index()
				i_a = select_action!(action_values, state_action_values, ϵ, i_s)
				#reset eligibility vector to 0 at the start of a new episode
				z .= zero(T)
				ep += 1
				save_episode_steps && push!(episode_step_history, step)
			else
				i_s = i_s′
				i_a = i_a′
				z .*= γ*λ
			end
			step += 1
		end
		
		return (episode_steps = episode_step_history, step_rewards = step_rewards)
	end

	#add a version of this where only state values are learned and the transition output is a distribution function, also can add expected sarsa versions of both where the value at the future time step gets weighted by the action selection of all the possible actions
	#non-tabular problem with binary features.  Each column represents the state feature values for the action of the column index
	function sarsa_λ!(parameters::Matrix{T}, active_features::Function, mdp::StateMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; α = one(T)/10, ϵ = one(T) / 10, z::Matrix{T} = copy(parameters), action_values::Vector{T} = zeros(T, length(mdp.actions)), save_episode_steps::Bool = false, save_step_rewards::Bool = false, use_dutch_traces::Bool = false, use_accumulating_traces::Bool = false, epkwargs...) where {T<:Real}
		#initialize records
		episode_step_history = Vector{T}()
		step_rewards = Vector{T}()

		function select_action!(action_values, parameters, ϵ, active_features)
			for i_a in eachindex(action_values)
				q = zero(T)
				for i in active_features
					q += parameters[i, i_a]
				end
				action_values[i_a] = q
			end
			make_ϵ_greedy_policy!(action_values; ϵ = ϵ)
			sample_action(action_values)
		end
			

		#initialize episode
		s = mdp.initialize_state()
		i_a = select_action!(action_values, parameters, ϵ, active_features(s))
		z .= zero(T)
		ep = 1
		step = 1
		
		while (ep <= max_episodes) && (step <= max_steps)
			#take action and observe transition
			(r, s′) = mdp.ptf(s, i_a)
			
			save_step_rewards && push!(step_rewards, r)

			δ = r
			
			for i in active_features(i_s)
				δ -= parameters[i, i_a]
				z[i, i_a] = one(T) + !use_dutch_traces*use_accumulating_traces*z[i, i_a] - use_dutch_traces*!use_accumulating_traces*α*γ*λ*z[i, i_a]
			end

			if !mdp.isterm(s′)
				i_a′ = select_action!(action_values, state_action_values, ϵ, active_features(s′))
				for i in active_features(s′)
					δ += γ*parameters[i, i_a′]
				end
			end

			parameters .+= α*δ .* z

			if mdp.isterm(s′)
				s = mdp.initialize_state()
				i_a = select_action!(action_values, state_action_values, ϵ, active_features(s))
				#reset eligibility vector to 0 at the start of a new episode
				z .= zero(T)
				ep += 1
				save_episode_steps && push!(episode_step_history, step)
			else
				s = s′
				i_a = i_a′
				z .*= γ*λ
			end
			step += 1
		end
		
		return (episode_steps = episode_step_history, step_rewards = step_rewards)
	end

	#non-tabular problem with general function approximation. 
	function sarsa_λ!(parameters::Vector{P}, feature_vector::Vector{T}, update_feature_vector!::Function, value_function::Function, update_gradient!::Function, mdp::StateMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; α = one(T)/10, ϵ = one(T) / 10, gradient::Vector{P} = deepcopy(parameters), z::Vector{P} = deepcopy(parameters), action_values::Vector{T} = zeros(T, length(mdp.actions)), save_episode_steps::Bool = false, save_step_rewards::Bool = false, use_accumulating_traces::Bool = false, use_dutch_traces::Bool = false, epkwargs...) where {T<:Real, P}
		#initialize records
		episode_step_history = Vector{T}()
		step_rewards = Vector{T}()

		function select_action!(action_values, parameters, ϵ, x)
			for i_a in eachindex(action_values)
				action_values[i_a] = value_function(x, parameters[i_a])
			end
			make_ϵ_greedy_policy!(action_values; ϵ = ϵ)
			sample_action(action_values)
		end
	
		#initialize episode
		s = mdp.initialize_state()
		update_feature_vector!(feature_vector, s)
		i_a = select_action!(action_values, parameters, ϵ, feature_vector)
		z .= zero(T)
		ep = 1
		step = 1
		
		while (ep <= max_episodes) && (step <= max_steps)
			#take action and observe transition
			(r, s′) = mdp.ptf(s, i_a)
			
			save_step_rewards && push!(step_rewards, r)
			δ = r - value_function(feature_vector, parameters[i_a])

			update_gradient!(gradient[i_a], feature_vector, parameters[i_a])
			
			if use_dutch
				d = dot(z[i_a], feature_vector)
			end
			
			z[i_a] .= gradient[i_a] .+ (!use_dutch_traces * use_accumulating_traces) .* z[i_a]
			if use_dutch_traces * !use_accumulating_traces
				z[i_a] .-= α*λ*γ*d .* feature_vector
			end

			if !mdp.isterm(s′)
				update_feature_vector!(feature_vector, s′)
				i_a′ = select_action!(action_values, parameters, ϵ, feature_vector)
				δ += γ*value_function(feature_vector, parameters[i_a′])
			end

			for i_a in eachindex(mdp.actions)
				parameters[i_a] .+= α*δ .* z[i_a]
			end

			if mdp.isterm(s′)
				s = mdp.initialize_state()
				update_feature_vector!(feature_vector, s)
				i_a = select_action!(action_values, parameters, ϵ, feature_vector)
				#reset eligibility vector to 0 at the start of a new episode
				for i_a in eachindex(mdp.actions) z[i_a] .= zero(T) end
				ep += 1
				save_episode_steps && push!(episode_step_history, step)
			else
				s = s′
				i_a = i_a′
				for i_a in eachindex(mdp.actions) z[i_a] .*= γ*λ end
			end
			step += 1
		end
		
		return (episode_steps = episode_step_history, step_rewards = step_rewards)
	end
end

# ╔═╡ 51274911-2eaa-4b18-b977-d0f735746bec
md"""
### *Example: Sarsa(λ) Gridworld Solution*
"""

# ╔═╡ afa6843b-9852-42c0-9ecd-06c408262334
md"""
#### Wind Values
"""

# ╔═╡ 078eecc3-05b3-4a58-91c8-fc8c28c9b144
md"""
#### Exact Solution
"""

# ╔═╡ b3c83ae6-9f3a-49a5-b726-872ef3a15853
md"""
#### Sarsa λ Parameter Study
"""

# ╔═╡ a4e10d47-2d41-4586-a328-11ea7234d7bd
#=╠═╡
@bind gridworld_λ_params PlutoUI.combine() do Child
	md"""
	|$λ$|Training Episodes|Learning Rate|
	|:--|:--|:--|
	|$(Child(:λ, Slider(0f0:0.01f0:1f0; default = 0.5f0, show_value=true)))|$(Child(:num_episodes, Slider(1:100; default = 50, show_value = true)))|$(Child(:α, Slider(0f0:0.01f0:1f0; default = 0.5f0, show_value = true)))|
	"""
end
  ╠═╡ =#

# ╔═╡ 4fa824ba-51a3-4f5a-a990-dd05bbf2526a
md"""
### True Online Sarsa(λ) for Tabular Problems or Linear Approximation
"""

# ╔═╡ 771cca22-d61d-498a-98be-90fa59e09571
begin
	#tabular problem where the parameters are just the state action values and each state action pair only has one active feature
	function true_online_sarsa_λ!(state_action_values::Matrix{T}, mdp::TabularMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; α = one(T)/10, ϵ = one(T) / 10, z::Matrix{T} = copy(state_action_values), action_values::Vector{T} = zeros(T, length(mdp.actions)), save_episode_steps::Bool = false, save_step_rewards::Bool = false, epkwargs...) where {T<:Real}
		#set the state action values of all terminal states to 0
		for i_s in eachindex(mdp.states)
			if mdp.terminal_states[i_s]
				state_action_values[:, i_s] .= zero(T)
			end
		end

		#initialize records
		episode_step_history = Vector{T}()
		step_rewards = Vector{T}()

		#initialize episode
		i_s = mdp.initialize_state_index()
		i_a = select_action!(action_values, state_action_values, ϵ, i_s)
		z .= zero(T)
		q_old = zero(T)
		ep = 1
		step = 1
		
		while (ep <= max_episodes) && (step <= max_steps)
			#take action and observe transition
			(r, i_s′) = mdp.ptf(i_s, i_a)
			
			save_step_rewards && push!(step_rewards, r)

			i_a′ = select_action!(action_values, state_action_values, ϵ, i_s′)

			q = state_action_values[i_a, i_s]
			q′ = state_action_values[i_a′, i_s′]
			
			δ = r + γ*q′ - q

			dt = z[i_a, i_s]
			z .*= γ*λ

			z[i_a, i_s] += one(T) - α*γ*λ*dt

			state_action_values .+= α*(δ + q - q_old) .* z
			state_action_values[i_a, i_s] -= α*(q - q_old)

			if mdp.terminal_states[i_s′]
				i_s = mdp.initialize_state_index()
				i_a = select_action!(action_values, state_action_values, ϵ, i_s)
				#reset eligibility vector to 0 at the start of a new episode
				z .= zero(T)
				q_old = zero(T)
				ep += 1
				save_episode_steps && push!(episode_step_history, step)
			else
				i_s = i_s′
				i_a = i_a′
				q_old = q′
			end
			step += 1
		end
		
		return (episode_steps = episode_step_history, step_rewards = step_rewards)
	end

	#non-tabular problem with linear function approximation. 
	function sarsa_λ!(parameters::Vector{Vector{T}}, feature_vector::Vector{T}, update_feature_vector!::Function, mdp::StateMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; α = one(T)/10, ϵ = one(T) / 10, x2::Vector{T} = copy(feature_vector), z::Vector{Vector{T}} = copy(parameters), action_values::Vector{T} = zeros(T, length(mdp.actions)), save_episode_steps::Bool = false, save_step_rewards::Bool = false, use_accumulating_traces::Bool = false, use_dutch_traces::Bool = false, epkwargs...) where {T<:Real}
		#initialize records
		episode_step_history = Vector{T}()
		step_rewards = Vector{T}()

		function select_action!(action_values, parameters, ϵ, x)
			for i_a in eachindex(action_values)
				action_values[i_a] = dot(x, parameters[i_a])
			end
			make_ϵ_greedy_policy!(action_values; ϵ = ϵ)
			sample_action(action_values)
		end
	
		#initialize episode
		s = mdp.initialize_state()
		update_feature_vector!(feature_vector, s)
		i_a = select_action!(action_values, parameters, ϵ, feature_vector)
		z .= zero(T)
		q_old = zero(T)
		ep = 1
		step = 1
		
		while (ep <= max_episodes) && (step <= max_steps)
			q = dot(feature_vector, parameters[i_a])
			dt = dot(z[i_a], feature_vector)
			x2 .= one(T) .+ α*γ*λ*dt .* feature_vector

			parameters[i_a] .-= α*(q - q_old) .* feature_vector
			
			#take action and observe transition
			(r, s′) = mdp.ptf(s, i_a)
			
			save_step_rewards && push!(step_rewards, r)

			
			if mdp.isterm(s′)
				q′ = zero(T)
			else
				update_feature_vector!(feature_vector, s′)
				i_a′ = select_action!(action_values, parameters, ϵ, feature_vector)
				q′ = dot(feature_vector, parameters[i_a′])
			end
			
			δ = r + γ*q′ - q
			
			z .*= γ*λ .+ x2

			for i_a in eachindex(mdp.actions)
				parameters[i_a] .+= α*(δ + q - q_old) .* z[i_a]
			end
			
			if mdp.isterm(s′)
				s = mdp.initialize_state()
				update_feature_vector!(feature_vector, s)
				i_a = select_action!(action_values, parameters, ϵ, feature_vector)
				#reset eligibility vector to 0 at the start of a new episode
				for i_a in eachindex(mdp.actions) z .= zero(T) end
				q_old = zero(T)
				ep += 1
				save_episode_steps && push!(episode_step_history, step)
			else
				s = s′
				i_a = i_a′
				q_old = q′
			end
			step += 1
		end
		
		return (episode_steps = episode_step_history, step_rewards = step_rewards)
	end
end

# ╔═╡ c5edfcbf-8d31-4dc4-b9d0-1a5439540710
begin
	function sarsa_λ(mdp::TabularMDP{T, S, A, P, F}, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; state_action_values::Matrix{T} = zeros(T, length(mdp.actions), length(mdp.states)), kwargs...) where {T<:Real, S, A, P, F}
		history = sarsa_λ!(state_action_values, mdp, γ, λ, max_episodes, max_steps; kwargs...)
		greedy_policy_lookup = [argmax(state_action_values[:, i_s]) for i_s in eachindex(mdp.states)]
		
		greedy_policy(s::S) = greedy_policy_lookup[mdp.state_index[s]]
		greedy_state_values = [maximum(state_action_values[:, i_s]) for i_s in eachindex(mdp.states)]
	
		(state_action_values = state_action_values, state_values = greedy_state_values, greedy_policy_lookup = greedy_policy_lookup, greedy_policy_function = greedy_policy, history = history)
	end
end

# ╔═╡ a36d205c-9a77-4b24-8e57-7bfceee9f4af
#=╠═╡
function gridworld_sarsaλ_parameter_study(mdp, steps; nruns = 50, λ_list = [0f0, 0.5f0, 0.7f0, 0.8f0, 0.9f0, 0.99f0], α_list = Base.LogRange(0.04f0, 0.8f0, 10), kwargs...)
	value_iteration_output = value_iteration_v(mdp, 1f0)
	best_steps = abs(value_iteration_output.final_value[mdp.initialize_state_index()])
	traces = [begin 
		output = [begin
			1:nruns |> Map() do _ 
				output = sarsa_λ(mdp, 1f0, λ, typemax(Int64), steps; save_episode_steps=true, α = α, kwargs...)
				step_history = output.history.episode_steps
				isempty(step_history) && return NaN
				step_history[end]/length(step_history)
			end |> foldxt(+) |> a -> a/nruns 
		end for α in α_list]
		scatter(x = α_list, y = output, name = "λ = $λ")
	end for λ in λ_list]
	plot(traces, Layout(xaxis_title = "Learning Rate", yaxis_title = "Steps per Episode <br> Averaged Over $steps Steps & $nruns Runs", yaxis_range = [1.5*best_steps, best_steps*3]))
end
  ╠═╡ =#

# ╔═╡ c57219cc-00ce-441d-a944-03c79c554708
begin
	function true_online_sarsa_λ(mdp::TabularMDP{T, S, A, P, F}, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; state_action_values::Matrix{T} = zeros(T, length(mdp.actions), length(mdp.states)), kwargs...) where {T<:Real, S, A, P, F}
		history = true_online_sarsa_λ!(state_action_values, mdp, γ, λ, max_episodes, max_steps; kwargs...)
		greedy_policy_lookup = [argmax(state_action_values[:, i_s]) for i_s in eachindex(mdp.states)]
		
		greedy_policy(s::S) = greedy_policy_lookup[mdp.state_index[s]]
		greedy_state_values = [maximum(state_action_values[:, i_s]) for i_s in eachindex(mdp.states)]
	
		(state_action_values = state_action_values, state_values = greedy_state_values, greedy_policy_lookup = greedy_policy_lookup, greedy_policy_function = greedy_policy, history = history)
	end

	function true_online_sarsa_λ(mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, feature_vector::Vector{T}, update_feature_vector!::Function; parameters::Vector{Vector{T}} = [zeros(T, length(feature_vector)) for i_a in eachindex(mdp.actions)], kwargs...) where {T<:Real, S, A, P, F1, F2, F3}
		history = true_online_sarsa_λ!(parameters, feature_vector, update_feature_vector!, mdp, γ, λ, max_episodes, max_steps; kwargs...)
		
		function value_function(s::S)
			x = copy(feature_vector)
			update_feature_vector!(x, s)
			action_values = [dot(x, parameters[i_a]) for i_a in eachindex(mdp.actions)]
		end
	
		function greedy_policy(s::S)
			action_values = value_function(s)
			argmax(action_values)
		end
	
		(value_function = value_function, greedy_policy = greedy_policy, history = history)
	end
end

# ╔═╡ d5bf43e7-07de-49c4-a2f2-39af6301d4b2
#=╠═╡
function gridworld_true_online_sarsaλ_parameter_study(mdp, steps; nruns = 50, λ_list = [0f0, 0.5f0, 0.7f0, 0.8f0, 0.9f0, 0.99f0], α_list = Base.LogRange(0.1f0, 1f0, 10), kwargs...)
	value_iteration_output = value_iteration_v(mdp, 1f0)
	best_steps = abs(value_iteration_output.final_value[mdp.initialize_state_index()])
	traces = [begin 
		output = [begin
			1:nruns |> Map() do _ 
				output = true_online_sarsa_λ(mdp, 1f0, λ, typemax(Int64), steps; save_episode_steps=true, α = α, kwargs...)
				step_history = output.history.episode_steps
				isempty(step_history) && return NaN
				step_history[end]/length(step_history)
			end |> foldxt(+) |> a -> a/nruns 
		end for α in α_list]
		scatter(x = α_list, y = output, name = "λ = $λ")
	end for λ in λ_list]
	plot(traces, Layout(xaxis_title = "Learning Rate", yaxis_title = "Steps per Episode <br> Averaged Over $steps Steps & $nruns Runs", yaxis_range = [1.5*best_steps, best_steps*3]))
end
  ╠═╡ =#

# ╔═╡ 8b6b5084-3972-4bd4-9ca2-423f1c627788
md"""
### *Example: Mountain Car Sarsa(λ) Variations*

Show similar parameter study for sarsa(λ) with different methods of traces and the true online version.  Can also add implementations of Sarsa(λ) that use q-learning and use the distribution function over output to bypass the action value estimation
"""

# ╔═╡ e7b274c5-4f0c-4e5f-8a4a-b574130e64c0
function getmaxinds(v::AbstractVector)
	maxinds = [1]
	nmax = 1
	maxval = first(v)

	function update!(i, maxval, maxinds, nmax)
		if v[i] > maxval
			maxinds[1] = i
			return (v[i], 1)
		end
		
		if v[i] == maxval
			push!(maxinds, i)
			return (maxval, nmax+1)
		end

		return (maxval, nmax)
	end
	
	for i in 2:lastindex(v)
		(maxval, nmax) = update!(i, maxval, maxinds, nmax)
	end

	maxinds[1:nmax]
end	

# ╔═╡ 2fbdb817-da15-4011-bb1c-126f1f311e7a
function findmaxrand(v::AbstractVector)
	maxval = first(v)
	imax = 1
	l = length(v)
	p = inv(l)

	function update!(i, maxval, imax)
		v[i] > maxval && return (v[i], i)
		v[i] < maxval && return (maxval, imax)

		r = rand()
		if r < p 
			(maxval, i) #randomly accept if the two are equal
		else
			(maxval, imax)
		end
	end
	
	for i in 2:lastindex(v)
		(maxval, imax) = update!(i, maxval, imax)
	end

	maxval, imax
end	

# ╔═╡ 72895891-9212-4722-b2a1-0e13c30a8ecf
#=╠═╡
function sarsaλ_linear(ℱ, w, states, actions, sterm, step, λ, γ, α, numepisodes, s_init, ϵ, usedutch = false)
	function ϵ_greedy(s, ϵ)
		rand() < ϵ && return rand(actions)
		qa = [sum(w[i] for i in ℱ(s, a)) for a in actions]
		# (inds, val) = getmaxinds(qa)
		(maxq, ind) = findmaxrand(qa)
		# inds = findall(q -> q == maxq, qa)
		# isempty(inds) && return rand(actions)
		# rand(actions[inds])
		actions[ind]
	end
	stepcounts = zeros(Int64, numepisodes)
	z = zeros(length(w))
	for ep in 1:numepisodes
		s = s_init()
		z .= 0.0
		stepcount = 0
		a = ϵ_greedy(s, ϵ)
		while true
			(s′, r) = step(s, a)
			stepcount += 1
			δ = r
			for i in ℱ(s, a)
				δ -= w[i]
				# accumulating traces
				z[i] += 1.0 - usedutch*(α*γ*z[i])
				# replacing traces
				# z[i] = 1.0
			end
			if s′ == sterm 
				w .+= α*δ .* z
				break
			end
			a′ = ϵ_greedy(s′, ϵ)
			for i in ℱ(s′, a′)
				δ += γ*w[i]
			end
			w .+= α*δ .* z
			z .*= γ*λ
			s = s′
			a = a′
		end
		stepcounts[ep] = stepcount
	end
	return w, stepcounts, s -> ϵ_greedy(s, 0.0)
end
  ╠═╡ =#

# ╔═╡ 07245a98-cab2-4b0c-a17a-4eaaa8a30703
#=╠═╡
function gridworld_sarsa(width, height, goal, λ, ϵ, α, numepisodes, usedutch = false; f = sarsaλ_linear)
	#states are tuples in an nxm grid
	states = [(x, y) for x in 1:width for y in 1:height]
	actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
	move(s, a) = (clamp(s[1] + a[1], 1, width), clamp(s[2] + a[2], 1, height))
	function step(s, a)
		s′ = move(s, a)
		(s′, Float64(s′ == goal))
	end

	function run_episode(π, s, maxsteps = 1e6)
		slist = [s]
		steps = 0
		while (s != goal) && (steps < maxsteps)
			(s, r) = step(s, π(s))
			steps +=1 
			push!(slist, s)
		end
		return slist
	end

	π_rand(s) = rand(actions)
		

	state_action_list = [(s, a) for s in states for a in actions]
	state_action_lookup = Dict(zip(state_action_list, eachindex(state_action_list)))
	
	#map state action pair to active features, the simplest case is to have a separate feature for every state action pair which is equivalent to the tabular method.  to do that we assign each state action pair a number.  Using a feature vector that just selects a single feature for states according to that index and then appends a later value for action would result in vectors that are not suitable for learning.  If we had a 5x5 gridworld that would be 25 states and 4 actions, so we could have activated features like [1, 26] and [2, 26].  Note that in this case whatever the weights are for indices 26 through 29 would impose a strict ordering on actions for every state because the q value for each (s, a) pair is just w[i_1] + w[i_2].  Short of the tabular case, our tilings need to allow the actions to be handled differently in different states so there are more than two activated features for a given state/action pair
	w_init() = zeros(length(states)*length(actions))
	s_init() = (1, 1)
	ℱ(s, a) = state_action_lookup[(s, a)]

	w, steps::Vector{Int64}, π = f(ℱ, w_init(), states, actions, goal, step, λ, 1.0, α, numepisodes, s_init, ϵ, usedutch)

	# return steps
	# run_episode(π_rand, (1, 1))
	(w, steps, π, π_rand, run_episode, step, states)
end
  ╠═╡ =#

# ╔═╡ f5c3d5a4-7fe8-420e-af0a-4318b1eeda2c
#=╠═╡
function eval_grid(lmax, wmax, goal; ϵ = 0.1, f = sarsaλ_linear, usedutch=false,  αlist = [0.05, 0.1, 0.2, 0.4, 0.8], λlist = [0.0, 0.4, 0.8, 0.9, 0.99])
	function runtrial(α, λ)::Float64
		(w, steps, π, π_rand, makepath, step, states) = gridworld_sarsa(lmax, wmax, goal, λ, ϵ, α, 100, usedutch, f = f)
		sum(steps[51:end])/50
	end


	runtrials(α, λ) = mean(runtrial(α, λ) for _ in 1:100)

	results = [[runtrials(α, λ) for α in αlist] for λ in λlist]
	(results = results, αlist=αlist, λlist=λlist)
end
  ╠═╡ =#

# ╔═╡ 4bd9d7a4-979d-492f-b863-8359864004ea
#=╠═╡
function plot_grid(lmax, wmax, goal; ϵ = 0.1, f = sarsaλ_linear, usedutch=false, αlist = [0.05, 0.1, 0.2, 0.4, 0.8], λlist = [0.0, 0.4, 0.8, 0.9, 0.99])
	(results, αlist, λlist) = eval_grid(lmax, wmax, goal, ϵ=ϵ, f = f, usedutch=usedutch, αlist = αlist, λlist = λlist)
	traces = [begin
		scatter(x = αlist, y = results[i], name = "λ = $(λlist[i])")
	end
	for i in eachindex(results)]
	plot(traces, Layout(yaxis_title = "Steps", xaxis_title = "α", title = "Mean Steps For 100 Episodes"))
end
  ╠═╡ =#

# ╔═╡ 32832503-d48b-48bb-be7b-cf2cb6855a57
# ╠═╡ disabled = true
#=╠═╡
plot_grid(10, 10, (5, 8), usedutch=false)
  ╠═╡ =#

# ╔═╡ fbe8691b-6d71-4cba-90e4-5de63421f634
md"""
> ### *Exercise 12.6* 
> Modify the pseudocode for Sarsa(λ) to use dutch traces (12.11) without the other distinctive features of a true online algorithm.  Assume linear function approximation and binary features.

See the above function `sarsaλ_linear`.  In the step where $z_i$ is updated an additional term is subtracted in the case of using dutch traces which matches equation (12.11)
"""

# ╔═╡ b1d56779-9a06-4b25-9a1b-09a12923e646
# ╠═╡ disabled = true
#=╠═╡
function true_online_sarsaλ_binary(ℱ, w, states, actions, sterm, step, λ, γ, α, numepisodes, s_init, ϵ, usedutch=true)
	function ϵ_greedy(s, ϵ)
		rand() < ϵ && return rand(actions)
		qa = [sum(w[i] for i in ℱ(s, a)) for a in actions]
		(maxq, ind) = findmaxrand(qa)
		# inds = findall(q -> q == maxq, qa)
		# isempty(inds) && return rand(actions)
		# rand(actions[inds])
		actions[ind]
	end
	stepcounts = zeros(Int64, numepisodes)
	z = zeros(length(w))
	for ep in 1:numepisodes
		s = s_init()
		z .= 0.0
		q_old = 0.0
		stepcount = 0
		a = ϵ_greedy(s, ϵ)
		while s != sterm
			(s′, r) = step(s, a)
			stepcount += 1
			a′ = ϵ_greedy(s′, ϵ)

			q = 0.0
			q′ = 0.0
			for i in ℱ(s, a)
				q += w[i]
				z[i] += 1.0 - α*γ*λ*z[i]
			end

			for i in ℱ(s′, a′)
				q′ += w[i]
			end
			
			δ = r + γ*q′ - q
			
			w .+= (α*(δ + q + q_old) .* z)
			for i in ℱ(s, a)
				w[i] -= α*(q - q_old)
			end
			q_old = q′
			z .*= γ*λ
			s = s′
			a = a′
		end
		stepcounts[ep] = stepcount
	end
	
	return w, stepcounts, s -> ϵ_greedy(s, 0.0)
end
  ╠═╡ =#

# ╔═╡ f3c3f934-6601-4383-8204-55d04c973881
# ╠═╡ disabled = true
#=╠═╡
plot_grid(10, 10, (5, 8), f = true_online_sarsaλ_binary, λlist = [0.0, 0.01, 0.02], αlist = [0.00001, 0.0001, 0.001, 0.002, 0.004])
  ╠═╡ =#

# ╔═╡ 5a88de5e-5837-41c8-8150-b8d65ffc2fdf
function update_action_values!(action_values::Vector{T}, x::Vector{T}, parameters::Vector{Vector{T}}) where T<:Real
	i_a_best = 1
	q_max = typemin(T)
	for (i_a, p) in enumerate(parameters)
		q = dot(p, x)
		action_values[i_a] = q
		newmax = q > q_max
		i_a_best = newmax*i_a + !newmax*i_a_best
		q_max = newmax*q + !newmax*q_max
	end
	return i_a_best, q_max
end

# ╔═╡ d5366fcf-39ee-48dc-8ee6-ca5c3141418a
function update_action_values!(action_values::Vector{T}, i_s::Integer, parameters::Vector{Vector{T}}) where T<:Real
	i_a_best = 1
	q_max = typemin(T)
	for (i_a, p) in enumerate(parameters)
		q = p[i_s]
		action_values[i_a] = q
		newmax = q > q_max
		i_a_best = newmax*i_a + !newmax*i_a_best
		q_max = newmax*q + !newmax*q_max
	end
	return i_a_best, q_max
end

# ╔═╡ 8f158e95-a308-430e-b0a5-265a3c1ea643
function true_online_sarsa_λ2(mdp::StateMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, state_representation::Vector{T}, update_state_representation!::Function; parameters = [zeros(T, length(state_representation)) for i_a in eachindex(mdp.actions)], kwargs...) where T<:Real
	reward_history = true_online_sarsa_λ!(parameters, state_representation, mdp, γ, λ, max_episodes, max_steps, update_state_representation!; kwargs...)

	function action_values(s)
		x = copy(state_representation)
		update_state_representation!(x, s)
		action_values = zeros(T, length(mdp.actions))
		i_a_best, q_max = update_action_values!(action_values, x, parameters)
		return action_values, i_a_best, q_max
	end

	greedy_policy(s) = action_values(s)[2]

	return (reward_history = reward_history, greedy_policy = greedy_policy, action_values = action_values)
end

# ╔═╡ 21e8ecc5-fe50-4cdf-8402-fe02db5dc13a
#note that this function will modify both parameters and the state representation vector as well as some of the keyword arguments
function true_online_sarsa_λ2!(parameters::Vector{Vector{T}}, state_representation::X, mdp::StateMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, update_state_representation!::Function; α = one(T)/10, ϵ = one(T)/10, ∇v::P = copy(parameters), z::P = copy(parameters), save_step_rewards::Bool = false, action_values::Vector{T} = zeros(T, length(mdp.actions)), save_episode_rewards::Bool = false, epkwargs...) where {P, X, T<:Real}
	@assert length(parameters) == length(mdp.actions)
	@assert length(z) == length(parameters)
	@assert length(action_values) = length(mdp.actions)

	s = mdp.initialize_state()
	update_state_representation!(state_representation, s)
	i_a = rand(eachindex(mdp.actions))
	v̂ = dot(state_representation, parameters[i_a])
	∇v .= state_representation
	(r, s′) = mdp.ptf(s, i_a)
	if mdp.isterm(s′)
		v̂′ = zero(T)
	else
		update_state_representation!(state_representation, s′)
		i_a′ = rand(eachindex(mdp.actions))
		v̂′ = dot(state_representation, parameters[i_a′])
	end
	ep = 1
	step = 1
	episode_reward_history = Vector{T}()
	step_reward_history = Vector{T}()
	
	#initialize eligibility vector to 0
	for i_a in eachindex(z) z[i_a] .= zero(T) end
	v_old = zero(T)
	rtot = r
	
	while (ep <= max_episodes) && (step <= max_steps)
		δ = r + γ*v̂′ - v̂
		for i_a in eachindex(z)
			z[i_a] .= (γ*λ .* z[i_a]) .+ (one(T) - α*γ*λ*dot(z[i_a], ∇v)) .* ∇v
		end
		a = v̂ - v_old
		for i_a in eachindex(parameters)
			parameters[i_a] .+= α .* ((δ + a) .* z[i_a])
		end
		parameters[i_a] .-= α .* a .* ∇v

		save_step_rewards && push!(step_reward_history, r)

		if mdp.isterm(s′)
			s = mdp.initialize_state()
			update_state_representation!(state_representation, s)
			#reset eligibility vector to 0 at the start of a new episode
			for i_a in eachindex(z) z[i_a] .= zero(T) end
			v_old = zero(T)
			rtot = zero(T)
			ep += 1
			save_episode_rewards && push!(episode_reward_history, rtot)
		else
			s = s′
			v_old = v̂′
		end
		
		#note that the state representation here will be for s on the next step
		v̂ = dot(state_representation, parameters[i_a′])
		∇v .= state_representation
		
		(r, s′) = mdp.ptf(s, i_a′)
		rtot += r
		
		if isterm(s′)
			v̂′ = zero(T)
		else
			update_state_representation!(state_representation, s′)
			i_a_best, q_best = update_action_values!(action_values, state_representation)
			#select ϵ greedy action
			i_a′ = rand() < ϵ ? rand(eachindex(action_values)) : i_a_best
			v̂′ = action_values[i_a′]
		end
		step += 1
	end
	return (episode_rewards = episode_reward_history, step_rewards = step_reward_history)
end

# ╔═╡ 2d7b4d32-d571-42eb-b31f-90a3bfe8a8c5
true_online_sarsa_λ(make_stochastic_gridworld(;stepreward = -1f0, termreward = 0f0), 1f0, .5f0, 1000, 1_000_000; α = .001f0, ϵ = 1f0, save_episode_rewards = true)

# ╔═╡ a9b96979-0c20-4f94-9903-34630af55668
function true_online_sarsa_λ2(mdp::TabularMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; parameters = [zeros(T, length(mdp.states)) for i_a in eachindex(mdp.actions)], kwargs...) where T<:Real
	reward_history = true_online_sarsa_λ!(parameters, mdp, γ, λ, max_episodes, max_steps; kwargs...)

	function action_values(i_s)
		action_values = zeros(T, length(mdp.actions))
		i_a_best, q_max = update_action_values!(action_values, i_s, parameters)
		return action_values, i_a_best, q_max
	end

	greedy_policy(s) = action_values(s)[2]

	return (reward_history = reward_history, greedy_policy = greedy_policy, action_values = action_values, parameters = parameters)
end

# ╔═╡ e199d7ed-4a99-4eff-a936-e8b860e73f2f
#note that this function will modify both parameters and the state representation vector as well as some of the keyword arguments
function true_online_sarsa_λ2!(parameters::Vector{Vector{T}}, mdp::TabularMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; α = one(T)/10, ϵ = one(T)/10, ∇v::P = copy(parameters), z::P = copy(parameters), save_step_rewards::Bool = false, action_values::Vector{T} = zeros(T, length(mdp.actions)), save_episode_rewards::Bool = false, epkwargs...) where {P, T<:Real}
	# @assert length(parameters) == length(mdp.actions)
	# @assert length(z) == length(parameters)
	# @assert length(action_values) = length(mdp.actions)

	i_s = mdp.initialize_state_index()
	i_a = rand(eachindex(mdp.actions))
	v̂ = parameters[i_a][i_s]
	(r, i_s′) = mdp.ptf(i_s, i_a)
	if mdp.terminal_states[i_s′]
		v̂′ = zero(T)
	else
		i_a′ = rand(eachindex(mdp.actions))
		v̂′ = parameters[i_a′][i_s′]
	end
	ep = 1
	step = 1
	episode_reward_history = Vector{T}()
	step_reward_history = Vector{T}()
	
	#initialize eligibility vector to 0
	for i_a in eachindex(z) z[i_a] .= zero(T) end
	v_old = zero(T)
	rtot = r
	
	while (ep <= max_episodes) && (step <= max_steps)
		δ = r + γ*v̂′ - v̂
		for i_a in eachindex(z)
			z[i_a] .*= γ*λ 
			z[i_a][i_s] += one(T) - α*γ*λ*z[i_a][i_s]
		end
		a = v̂ - v_old
		for i_a in eachindex(parameters)
			parameters[i_a] .+= α .* ((δ + a) .* z[i_a])
		end
		parameters[i_a][i_s] -= α * a

		save_step_rewards && push!(step_reward_history, r)

		if mdp.terminal_states[i_s′]
			i_s = mdp.initialize_state_index()
			#reset eligibility vector to 0 at the start of a new episode
			for i_a in eachindex(z) z[i_a] .= zero(T) end
			v_old = zero(T)
			i_a_best, q_best = update_action_values!(action_values, i_s′, parameters) 
			i_a′ = rand() < ϵ ? rand(eachindex(action_values)) : i_a_best
			ep += 1
			save_episode_rewards && push!(episode_reward_history, rtot)
			rtot = zero(T)
		else
			i_s = i_s′
			v_old = v̂′
		end
		
		#note that the state representation here will be for s on the next step
		v̂ = parameters[i_a′][i_s]
		
		(r, i_s′) = mdp.ptf(i_s, i_a′)
		rtot += r
		i_a = i_a′
		if mdp.terminal_states[i_s′]
			v̂′ = zero(T)
		else
			i_a_best, q_best = update_action_values!(action_values, i_s′, parameters)
			#select ϵ greedy action
			i_a′ = rand() < ϵ ? rand(eachindex(action_values)) : i_a_best
			v̂′ = action_values[i_a′]
		end
		
		step += 1
	end
	return (episode_rewards = episode_reward_history, step_rewards = step_reward_history)
end

# ╔═╡ 862026e9-ebe6-4f2e-8832-086bbba8db17
md"""
## 12.8 Variable λ and γ
"""

# ╔═╡ 8f894492-260e-4ab0-87b6-c02216a631e6
md"""
$\begin{flalign}
G_t &= \sum_{k=t}^\infty \left ( \prod_{i=t+1}^k \gamma_i \right ) R_{k+1} \tag{12.17}\\
G_t^{\lambda_s} &\doteq R_{t+1} + \gamma_{t+1} \left ( (1-\lambda_{t+1}) \hat v (S_{t+1}, \mathbf{w}_t) + \lambda_{t+1}G_{t+1}^{\lambda_s} \right ) \tag{12.18}\\
G_t^{\lambda_a} &\doteq R_{t+1} + \gamma_{t+1} \left ( (1-\lambda_{t+1}) \hat q (S_{t+1}, A_{t+1}, \mathbf{w}_t) + \lambda_{t+1}G_{t+1}^{\lambda_a} \right ) \tag{12.19}\\
G_t^{\lambda_a} &\doteq R_{t+1} + \gamma_{t+1} \left ( (1-\lambda_{t+1}) \overline V_t(S_{t+1}) + \lambda_{t+1}G_{t+1}^{\lambda_a} \right ) \tag{12.19}\\
\overline V_t(s) & \doteq \sum_a \pi(a|s)\hat q(s, a, \mathbf{w}_t) \tag{12.21}
\end{flalign}$
"""

# ╔═╡ c80256a7-be4f-4407-b0bf-7a13415482ad
md"""
> ### *Exercise 12.7* 
> Generalize the three recursive equations above to their truncated versions, defining $G_{t:h}^{\lambda_s}$ and $G_{t:h}^{\lambda_a}$

Starting with (12.18) we want to get a truncated version $G_{t:h}^{\lambda_s}$.  We can also use as a model the truncated λ-return

$\begin{flalign}
G_t^{\lambda_s} &\doteq R_{t+1} + \gamma_{t+1} \left ( (1-\lambda_{t+1}) \hat v (S_{t+1}, \mathbf{w}_t) + \lambda_{t+1}G_{t+1}^{\lambda_s} \right )\\
G_{t:h}^\lambda &\doteq (1-\lambda) \sum_{n=1}^{h-t-1}\lambda^{n-1} G_{t:t+n} + \lambda^{h-t-1}G_{t:h}\\
G_{t:h}^{\lambda_s} &\doteq (1-\lambda) \sum_{n=1}^{h-t-1} \left ( \prod_{i=t}^{t+n-1} \lambda_i \right ) G_{t:t+n} + \left ( \prod_{i=h-t-1}^{\infty} \lambda_i \right ) G_{t:h}
\end{flalign}$

"""

# ╔═╡ e6782d51-175c-4de7-9c75-1fc3f75a92f0
md"""
## Chapter 7 Code For Random Walk Comparison
"""

# ╔═╡ 013c2268-6ab8-441a-9fb4-5118dc3ae18a
#=╠═╡
#based on pseudocode described in book for n-step TD value estimation
function n_step_TD_Vest(π, α, n, states, sterm, sim, γ; v0 = 0.0, numep = 1000, Vtrue = Dict(s => v0 for s in states))
	V = Dict(s => v0 for s in states)
	V[sterm] = 0.0
	Svec = Vector{eltype(states)}(undef, n+1)
	Rvec = Vector{Float64}(undef, n+1)
	rmserr() = sqrt(mean((V[s] - Vtrue[s])^2 for s in states))
	rmserrs = Vector{Float64}(undef, numep)
	for ep in 1:numep
		#for each episode save a record of states and rewards
		s0 = rand(states)
		Svec[1] = s0
		s = s0
		T = typemax(Int64)
		t = 0
		while true
			if t < T
				a = π(s)
				(s, r) = sim(Svec[mod(t, n+1)+1], a)
				storeind = mod(t+1, n+1) + 1
				Svec[storeind] = s
				Rvec[storeind] = r
				(s == sterm) && (T = t + 1)
			end
			τ = t - n + 1

			if τ >= 0
				G = sum(γ^(i - τ - 1) * Rvec[mod(i, n+1)+1] for i in (τ + 1):min(τ+n, T))
				if τ+n < T
					G += γ^n * V[Svec[mod(τ+n, n+1)+1]]
				end
				if τ == 0
					V[s0] += α*(G - V[s0])
				else
					V[Svec[mod(τ, n+1)+1]] += α*(G - V[Svec[mod(τ, n+1)+1]])
				end
			end
			t += 1
			(τ == T - 1) && break
		end
		rmserrs[ep] = rmserr()
	end
	return V, rmserrs
end
  ╠═╡ =#

# ╔═╡ 44a16c0a-9d0d-4e9b-9ae5-aef791c4f544
begin
	abstract type LinearMoves end
	struct Left <: LinearMoves end
	struct Right <: LinearMoves end
end

# ╔═╡ 13756b5d-b496-45fa-875b-7f6ee6468dcf
#create a random walk mdp of length n where the left terminal state produces a reward of -1 and the right a reward of 1
function create_random_walk(n::Int64)
	states = Tuple(1:n)
	sterm = 0
	
	function step(s0, action)
		move(s, ::Left) = s - 1
		move(s, ::Right) = s + 1

		s = move(s0, action)
		(s == 0) && return (sterm, -1.0)
		(s > n) && return (sterm, 1.0)
		return (s, 0.0)
	end
	(states, sterm, step)
end

# ╔═╡ 2cafed7d-22c6-420f-9c8e-8ae734bfbad2
#=╠═╡
function nsteptd_error_random_walk(nstates, estimator; v0=0.0, nruns = 10)
	#estimate random policy
	π(s) = rand([Left(), Right()])

	c = (nstates + 1)/2
	Vtrue = [(s-c)/c for s in 1:nstates]

	maxerr = sqrt(mean(Vtrue .^2))

	(states, sterm, step) = create_random_walk(nstates)
	
	function get_nstep_error(α, n)
		Vest, rmserrs = estimator(π, α, n, states, sterm, step, 1.0, numep = 10, v0=v0, Vtrue = Vtrue)
		mean(rmserrs)
	end

	α_vec = 1.1 .^ (-30:0)
	n_vec = [2 .^ (0:7); 1_000]
	rmsvecs = [[mean(get_nstep_error(α, n) for _ in 1:nruns) for α in α_vec] for n in n_vec]
	(rmsvecs, α_vec, n_vec, ymax = maxerr)
end
  ╠═╡ =#

# ╔═╡ 14b3b28e-1351-4b45-9a57-bfab846a2ffd
#=╠═╡
@htl("""
<div style = "display: flex; height: 500px;">
$(offline_λ_error_random_walk(fig_12_3_n; num_trials = 200)) 
$(nsteptd_error_random_walk(fig_12_3_n; num_trials = 200))
</div>
""")
  ╠═╡ =#

# ╔═╡ a3484638-ae83-4810-9226-0a25b3fc58dc
#=╠═╡
function optimize_n_randomwalk(nstates; estimator = n_step_TD_Vest, v0=0.0, nruns = 10)
	(rmsvecs, α_vec, n_vec, ymax) = nsteptd_error_random_walk(nstates, estimator, v0=v0, nruns = nruns)
	traces = [scatter(x = α_vec, y = rmsvecs[i], name = "n=$(n_vec[i])") for i in eachindex(rmsvecs)]
	ymin = minimum(minimum(v) for v in rmsvecs) * 0.9
	plot(traces, Layout(title="RMS Error for $nstates State Chain with Random Policy Over the First 10 Episodes, n-step TD Estimator", xaxis_title = "α", yaxis_range = [ymin, ymax]))
end
  ╠═╡ =#

# ╔═╡ 0358288e-be4e-46c2-ac4c-16ace6f50187
md"""
# Dependencies
"""

# ╔═╡ 326b3355-7941-403b-bf1e-3031f585f666
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

# ╔═╡ 5616d294-892a-40bc-a35f-35e9e0ee55e2
md"""
## Visualization Tools
"""

# ╔═╡ 2b047cbe-4da3-4e40-8897-8ea83e70a84d
#=╠═╡
function addelements(e1, e2)
	@htl("""
	$e1
	$e2
	""")
end
  ╠═╡ =#

# ╔═╡ 4b1d86d8-e6b9-4d09-b2a1-8c297414094c
#=╠═╡
@bind wind_values PlutoUI.combine() do Child
	@htl("""
	$(mapreduce(addelements, 1:10) do i
	Child(NumberField(0:5))
	end)
	""")
end |> confirm
  ╠═╡ =#

# ╔═╡ 6d671599-f635-4e1d-bef5-707891a756cc
#=╠═╡
const gridworld_mdp = make_stochastic_gridworld(;stepreward = -1f0, termreward = 0f0, wind = wind_values)
  ╠═╡ =#

# ╔═╡ 8d52f740-e1bf-4ac7-a299-38e7767a0831
#=╠═╡
gridworld_sarsaλ_parameter_study(gridworld_mdp, 50_000; use_dutch_traces = false)
  ╠═╡ =#

# ╔═╡ e047cce1-11a5-4bcb-8668-a767628da140
#=╠═╡
gridworld_true_online_sarsaλ_parameter_study(gridworld_mdp, 50_000)
  ╠═╡ =#

# ╔═╡ 9db3ed98-a94d-4adc-a45f-75eca432a1e9
show_grid_value(mdp::TabularMDP, Q, name; kwargs...) = show_grid_value(mdp.states, mdp.terminal_states, mdp.initialize_state_index, Q, name; kwargs...)

# ╔═╡ 4a8bc15c-8f4d-4017-915f-d2b27c1a6bd0
show_grid_probabilities(mdp::TabularMDP, Q, name; kwargs...) = show_grid_probabilities(mdp.states, mdp.terminal_states, mdp.initialize_state_index, Q, name; kwargs...)

# ╔═╡ 4160de31-3c8e-4051-b618-24112bbcc70e
HTML("""
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

# ╔═╡ 359a682a-add2-4fe2-af09-f67ffbd985a8
#=╠═╡
function show_grid_probabilities(states, terminds::BitVector, state_init, μ::Vector, name; scale = 1.0, title = "", sigdigits = 2, square_pixels = 20, highlight_state_index = 0)
	width = maximum(s.x for s in states)
	height = maximum(s.y for s in states)
	start = states[state_init()]
	sterms = any(terminds) ? states[terminds] : [GridworldState(0, 0)]
	ngrid = width*height

	displayvalue(Q::Matrix, i) = round(maximum(Q[:, i]), sigdigits = sigdigits)
	displayvalue(V::Vector, i) = round(V[i], sigdigits = sigdigits)
	
	maxp = maximum(μ)
	function calculate_color(p::Real) 
		v = round(Int64, 255*p/maxp)
		"rgb($v, $v, $v)"
	end

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
		Maximum probability $maxp shown in white
		<div style = "display: flex; transform: scale($scale); background-color: rgba(0, 0, 0, 0); color: black; font-size: 16px; justify-content: center;">
			<div>
				$title
				<div class = "gridworld $name value">
					$(HTML(mapreduce(i -> """<div class = "gridcell $name value" x = "$(states[i].x)" y = "$(states[i].y)" style = "grid-row: $(height - states[i].y + 1); grid-column: $(states[i].x); background-color: $(calculate_color(μ[i])); font-size: 12px; color: black;">$(displayvalue(μ, i))</div>""", *, eachindex(states))))
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
  ╠═╡ =#

# ╔═╡ f22355c6-1b48-4a8d-b5b0-b851f3dadd52
#=╠═╡
function show_grid_value(states, terminds::BitVector, state_init, Q, name; scale = 1.0, title = "", sigdigits = 2, square_pixels = 20, highlight_state_index = 0)
	width = maximum(s.x for s in states)
	height = maximum(s.y for s in states)
	start = states[state_init()]
	sterms = any(terminds) ? states[terminds] : [GridworldState(0, 0)]
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
  ╠═╡ =#

# ╔═╡ c8be71e0-c82a-4260-9dcf-944962947ca2
show_grid_policy(mdp::TabularMDP, π, name; kwargs...) = show_grid_policy(mdp.states, mdp.initialize_state_index, mdp.terminal_states, π, name; kwargs...)

# ╔═╡ c231090d-6faf-46a8-ae08-fd8715ade241
#=╠═╡
function display_rook_policy(v::Vector{T}; scale = 1.0) where T<:AbstractFloat
	@htl("""
		<div style = "display: flex; align-items: center; justify-content: center; transform: scale($scale);">
		<div class = "downarrow" style = "position: absolute; transform: rotate(180deg); opacity: $(v[1]);"></div>	
		<div class = "downarrow" style = "position: absolute; opacity: $(v[2])"></div>
		<div class = "downarrow" style = "position: absolute; transform: rotate(90deg); opacity: $(v[3])"></div>
		<div class = "downarrow" style = "transform: rotate(-90deg); opacity: $(v[4])"></div>
		</div>
	""")
end
  ╠═╡ =#

# ╔═╡ 962ff704-5d43-4ecb-8e2d-4538c6ee71c5
#=╠═╡
function show_selected_action(i)
	v = zeros(4)
	v[i] = 1
	display_rook_policy(v)
end
  ╠═╡ =#

# ╔═╡ afa7fa78-283c-4053-9dff-7cc2adc0cc0e
#=╠═╡
const rook_action_display = @htl("""
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
  ╠═╡ =#

# ╔═╡ 55727cd1-ebb0-4e5c-80e5-c6d047c4b1ba
#=╠═╡
function show_grid_transitions(states, terminds, state_init, name; scale = 1.0, title = "", action_display = rook_action_display, highlight_state = GridworldState(1, 1), transition_states::Dict{GridworldState, Float32} = Dict([GridworldState(1, 2) => 1f0]), reward_values = [(p = 1f0, r = 0f0)], width = maximum(s.x for s in states), wind = zeros(Int64, width), square_pixels = 30)
	height = maximum(s.y for s in states)
	start = states[state_init()]
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
  ╠═╡ =#

# ╔═╡ f9137b59-24cc-4636-8a42-3a751309f42b
#=╠═╡
function show_grid_policy(states, state_init, terminds, π, name; display_function = display_rook_policy, action_display = rook_action_display, scale = 1.0)
	width = maximum(s.x for s in states)
	height = maximum(s.y for s in states)
	start = states[state_init()]
	sterms = any(terminds) ? states[terminds] : [GridworldState(0, 0)]
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

			$(mapreduce(addelements, sterms) do sterm
				@htl("""
				.$name[x="$(sterm.x)"][y="$(sterm.y)"]::before {
					content: 'G';
					position: absolute;
					color: red;
					opacity: 1.0;
				}
				""")
			end)

		</style>
	""")
end
  ╠═╡ =#

# ╔═╡ f9ef6328-06f0-47a5-9de3-9b00d02af7f6
HTML("""
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

# ╔═╡ e89b2c6f-7433-434f-8e46-836870f272b6
#=╠═╡
function plot_path(episode_states::Vector{S}, goal::S, start::S; title = "Policy <br> path example", iscliff = s -> false, iswall = s -> false, pathname = "Policy Path", xmax = maximum([s.x for s in episode_states]), ymax = maximum([s.y for s in episode_states])) where S <: GridworldState
	start_trace = scatter(x = [start.x + 0.5], y = [start.y + 0.5], mode = "text", text = ["S"], textposition = "left", showlegend=false)
	finish_trace = scatter(x = [goal.x + .5], y = [goal.y + .5], mode = "text", text = ["G"], textposition = "left", showlegend=false)
	
	path_traces = [scatter(x = [episode_states[i].x + 0.5, episode_states[i+1].x + 0.5], y = [episode_states[i].y + 0.5, episode_states[i+1].y + 0.5], line_color = "blue", mode = "lines", showlegend=false, name = pathname) for i in 1:length(episode_states)-1]
	finalpath = scatter(x = [episode_states[end].x + 0.5, goal.x + .5], y = [episode_states[end].y + 0.5, goal.y + 0.5], line_color = "blue", mode = "lines", showlegend=false, name = pathname)

	h1 = 30*ymax
	traces = [start_trace; finish_trace; path_traces; finalpath]

	cliff_squares = filter(iscliff, episode_states)
	for s in cliff_squares
		push!(traces, scatter(x = [s.x + 0.6], y = [s.y+0.5], mode = "text", text = ["C"], textposition = "left", showlegend = false))
	end


	wall_squares = filter(iswall, episode_states)
	for s in wall_squares
		push!(traces, scatter(x = [s.x + 0.8], y = [s.y+0.5], mode = "text", text = ["W"], textposition = "left", showlegend = false))
	end

	plot(traces, Layout(xaxis = attr(showgrid = true, showline = true, gridwith = 1, gridcolor = "black", zeroline = true, linecolor = "black", mirror=true, tickvals = 1:xmax, ticktext = fill("", 10), range = [1, xmax+1]), yaxis = attr(linecolor="black", mirror = true, gridcolor = "black", showgrid = true, gridwidth = 1, showline = true, tickvals = 1:ymax, ticktext = fill("", ymax), range = [1, ymax+1]), width = max(30*xmax, 200), height = max(h1, 200), autosize = false, padding=0, paper_bgcolor = "rgba(0, 0, 0, 0)", title = attr(text = title, font_size = 14, x = 0.5)))
end
  ╠═╡ =#

# ╔═╡ b39b1b8e-b70e-4860-b75c-86506433efa7
#=╠═╡
function plot_path(episode_states::Vector{Int64}, i_sterm::Integer, gridworld_states::Vector{S}, i_s0::Integer, terminal_states::BitVector; title = "Policy <br> path example", iscliff = s -> false, iswall = s -> false, pathname = "Policy Path", square_pixels = 30) where S <: GridworldState
	xmax = maximum([s.x for s in gridworld_states])
	ymax = maximum([s.y for s in gridworld_states])
	start = gridworld_states[i_s0]
	goal = gridworld_states[findlast(terminal_states)]
	start_trace = scatter(x = [start.x + 0.5], y = [start.y + 0.5], mode = "text", text = ["S"], textposition = "left", showlegend=false)
	finish_trace = scatter(x = [goal.x + .5], y = [goal.y + .5], mode = "text", text = ["G"], textposition = "left", showlegend=false)
	
	path_traces = [scatter(x = [gridworld_states[episode_states[i]].x + 0.5, gridworld_states[episode_states[i+1]].x + 0.5], y = [gridworld_states[episode_states[i]].y + 0.5, gridworld_states[episode_states[i+1]].y + 0.5], line_color = "blue", mode = "lines", showlegend=false, name = pathname) for i in 1:length(episode_states)-1]
	finalpath = scatter(x = [gridworld_states[episode_states[end]].x + 0.5, gridworld_states[i_sterm].x + .5], y = [gridworld_states[episode_states[end]].y + 0.5, gridworld_states[i_sterm].y + 0.5], line_color = "blue", mode = "lines", showlegend=false, name = pathname)

	h1 = square_pixels*ymax
	traces = [start_trace; finish_trace; path_traces; finalpath]

	cliff_squares = filter(iscliff, gridworld_states)
	for s in cliff_squares
		push!(traces, scatter(x = [s.x + 0.6], y = [s.y+0.5], mode = "text", text = ["C"], textposition = "left", showlegend = false))
	end


	wall_squares = filter(iswall, gridworld_states)
	for s in wall_squares
		push!(traces, scatter(x = [s.x + 0.8], y = [s.y+0.5], mode = "text", text = ["W"], textposition = "left", showlegend = false))
	end

	plot(traces, Layout(xaxis = attr(showgrid = true, showline = true, gridwith = 1, gridcolor = "black", zeroline = true, linecolor = "black", mirror=true, tickvals = 1:xmax, ticktext = fill("", 10), range = [1, xmax+1]), yaxis = attr(linecolor="black", mirror = true, gridcolor = "black", showgrid = true, gridwidth = 1, showline = true, tickvals = 1:ymax, ticktext = fill("", ymax), range = [1, ymax+1]), width = max(square_pixels*xmax, 200), height = max(h1, 200), autosize = false, padding=0, paper_bgcolor = "rgba(0, 0, 0, 0)", title = attr(text = title, font_size = 14, x = 0.5)))
end
  ╠═╡ =#

# ╔═╡ 0e3ae279-be7e-4e13-b1ea-2c0efced3162
function plot_path(mdp::TabularMDP, π; i_s0 = mdp.initialize_state_index(), max_steps = 100, kwargs...)
	(states, actions, rewards, sterm) = runepisode(mdp; i_s0 = i_s0, π = π, max_steps = max_steps)
	plot_path(states, sterm, mdp.states, i_s0, mdp.terminal_states; kwargs...)
end

# ╔═╡ 82adce34-923c-46b9-a1ed-f1c06be09e0f
plot_path(mdp::TabularMDP; title = "Random policy <br> path example", kwargs...) = plot_path(mdp, make_random_policy(mdp); title = title, kwargs...)

# ╔═╡ 1441e34f-e8c5-46b5-b04c-31ea7f9f605a
#=╠═╡
function show_gridworld_exact_solution(mdp)
	output = value_iteration_v(mdp, 1f0)
	@htl("""
	<div style = "display: flex">
	<div>
	Optimal Policy
	$(show_grid_policy(mdp, output.optimal_policy, "value_iteration_gridworld"))
	</div>
	<div>
	Optimal Value Function (-1 reward per step)
	$(show_grid_value(mdp, output.final_value, "value_iteration_values"; square_pixels = 40))
	</div>
	$(plot_path(mdp, output.optimal_policy; square_pixels = 40))
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ d08aaab5-4065-41ef-867d-ea689c87a4f6
#=╠═╡
show_gridworld_exact_solution(gridworld_mdp)
  ╠═╡ =#

# ╔═╡ bb71215f-fa97-4eca-a950-be4cb037bb00
#=╠═╡
function show_gridworld_sarsaλ_solution(mdp, λ, episodes; kwargs...)
	output = sarsa_λ(mdp, 1f0, λ, episodes, typemax(Int64); kwargs...)
	greedy_policy = TabularRL.make_greedy_policy(output.state_action_values)
	
	@htl("""
	<div style = "display: flex">
	$(show_grid_policy(mdp, greedy_policy, "sarsa_λ_policy"))
	$(show_grid_value(mdp, output.state_values, "sarsa_λ_values"; square_pixels = 40))
	$(plot_path(mdp, greedy_policy; square_pixels = 40))
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ 26a64c5d-6c21-4e3b-af75-e8682e8d5ea1
#=╠═╡
show_gridworld_sarsaλ_solution(gridworld_mdp, gridworld_λ_params.λ, gridworld_λ_params.num_episodes; α = gridworld_λ_params.α)
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
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Transducers = "28d57a85-8fef-5791-bfe6-a80928e7c999"

[compat]
BenchmarkTools = "~1.3.2"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
PlutoDevMacros = "~0.9.0"
PlutoPlotly = "~0.3.6"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.49"
StaticArrays = "~1.9.10"
Statistics = "~1.11.1"
StatsBase = "~0.33.21"
Transducers = "~0.4.84"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "eeaf5e83f39dc2c9adbdb2cb6d84b3370de6d0e8"

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
deps = ["CompositionsBase", "ConstructionBase", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown"]
git-tree-sha1 = "96bed9b1b57cf750cca50c311a197e306816a1cc"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.39"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsDatesExt = "Dates"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsTestExt = "Test"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.ArgCheck]]
git-tree-sha1 = "680b3b8759bd4c54052ada14e52355ab69e07876"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires"]
git-tree-sha1 = "e2144b631226d9eeab2d746ca8880b7ccff504ae"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.3"

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
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "c785dfb1b3bfddd1da557e861b919819b82bbe5b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.1"

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
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
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
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
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
version = "1.11.0"

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
version = "1.11.0"

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
git-tree-sha1 = "2dd20384bf8c6d411b5c7370865b1e9b26cb2ea3"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.6"

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

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

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

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "10da5154188682e5c0726823c2b5125957ec3778"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.38"

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

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

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
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

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
deps = ["AbstractPlutoDingetjes", "Colors", "Dates", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "PackageExtensionCompat", "PlotlyBase", "PlutoUI", "Reexport"]
git-tree-sha1 = "9a77654cdb96e8c8a0f1e56a053235a739d453fe"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.3.9"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"

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
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "47091a0340a675c738b1304b58161f3b0839d454"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.10"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

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
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

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

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

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
# ╟─c62195dd-aa6e-4fd2-b9a9-848837a072d8
# ╟─3bc13de6-b767-4e2c-95bc-a44c1e688a77
# ╟─6426347e-4264-4a7f-9393-1065f8365efb
# ╟─c126b7c4-73df-4a0f-9ee4-a766eb19c5ba
# ╟─01144f94-7a2e-4137-9cf8-4264d87a50a2
# ╟─50a4de5c-3856-45be-b552-011966faf9aa
# ╟─cd03392d-c19e-48ca-8d02-00bd342fbbb3
# ╟─2dcc2fa1-093d-4e6c-b168-4878d4e7ee86
# ╟─b670fe33-3db3-40f3-beb5-8508c260b3d0
# ╟─6ffa37d5-b587-4989-93fb-7003d818c082
# ╟─2c1afacc-7956-479b-898d-eadf02a2ec19
# ╟─1035d33b-5e02-4d41-81cc-66546383db68
# ╟─7f43afbf-3375-4ad1-acee-f6b74f98e20f
# ╟─dccc9b45-b711-44e8-8788-93de05f26543
# ╟─54e578de-d12c-4257-91b5-a257ea9c6ba6
# ╟─752a80ea-1da6-49ef-91ef-a03c590b825d
# ╠═9d131051-eeee-4aba-8f78-9ddff9babab4
# ╟─134ce360-6290-4aea-b6c0-eaa825d6f9a5
# ╟─37ffc88c-8418-468b-a537-37b8e6bf5922
# ╠═9d512c7b-3d49-439a-a971-1a3dad065d6e
# ╠═d55aca40-b03a-4f6b-84e6-ced6c8f67da1
# ╠═a5877ac6-3bd8-4832-bf19-618b01ba16e2
# ╠═fba683ec-c923-498c-b379-9d23a1d4aa76
# ╠═ce8b9ebf-942a-4807-a36f-ced03c3c7916
# ╠═c240d631-3095-4880-b454-66e05a59e4ea
# ╠═eefb36bb-d988-4f5d-bfbb-c3df2f869ab6
# ╠═583d2f42-692a-4028-93cc-47c2e178c84e
# ╟─176ee625-51c6-48f1-8f01-d3fb7008db6e
# ╠═14b3b28e-1351-4b45-9a57-bfab846a2ffd
# ╟─57cf5ae7-d4dd-47e8-8090-c04fb39e0763
# ╟─34dda4bf-f78f-4c83-ba10-9b206d2fbcb8
# ╟─6f5168dc-f1f3-4533-a59e-bb85895f3b13
# ╠═5610a0ba-60a8-4da6-8f68-50b1c5e82686
# ╟─5e5fdcee-356e-46d4-a5b0-3c433aee989d
# ╠═24468748-009d-42a3-918d-4ba18b23c9ed
# ╟─e99caf5c-7c13-4edd-b55b-dce93cc850c6
# ╠═900760f0-b253-4db7-8c4f-4ca34777198d
# ╟─373a89e3-0b8d-49a0-982e-8bb300538429
# ╟─2c3b163d-b4cd-4b40-a597-cbd103e135b6
# ╠═36d87b2a-6e1a-47b7-8af5-825d47e55eec
# ╠═b68f1171-6274-4d93-bf68-05b95cb5b2f8
# ╠═8b67e90f-3305-4702-b7a3-393055d15e46
# ╠═da8c5f8b-5ab6-4a2b-93e8-18be4284b932
# ╟─83a645f8-f806-4828-bc42-d24cfd26bad3
# ╟─9a75dc05-883b-47a6-b8f0-ae0799c5fc19
# ╟─addedc75-375f-429f-8e2e-90ba2151dee0
# ╟─5e1366cc-05cd-43b3-8a00-e56242a30d8f
# ╠═4f9cbb26-6c9b-458a-b7e6-102f0dbf64cb
# ╟─e597a042-9c03-4d49-a48f-6dff39283c54
# ╟─0c6ebdeb-77f4-44f0-9bf3-c539d54bcaec
# ╟─27f535a4-2245-45aa-aefa-4c0fc6bb218d
# ╟─e1e9f2eb-4751-4f5c-aa4d-a0cf75e193b2
# ╟─531263cf-274e-4a64-932f-821e8583a316
# ╟─0df08e27-18d3-4f2c-a7e1-75674418ba01
# ╟─c65dc168-9fa4-4e1b-af39-02f80c9ec0e3
# ╟─8d41a846-3a12-4e32-bc1a-50be12629eb2
# ╟─8f211c8d-6b20-48de-ae91-435a977c930c
# ╟─21df1418-8808-42f1-9fcd-26048705c5ce
# ╠═7f8fb89d-1a2e-4acd-9118-2ce3d3874341
# ╠═34a28cfa-bf18-4dcf-8cf4-f6e9031d6fc2
# ╠═4dc03d69-c90c-4d64-bb8e-600ecfe30eb8
# ╠═d9f89b2c-8df8-415a-a0a8-21744be88cec
# ╠═00015316-0d9d-41ce-a001-09aede10049c
# ╠═9c287f15-a78a-4e3f-a0b7-c901874b6cca
# ╟─ccf903ab-1509-4c53-bfe0-435152ecea4b
# ╟─f7cc35c5-c1ed-440d-8723-8c1b8b966b8f
# ╟─b36896b1-6802-48e1-8cd3-f08bf3b99e3e
# ╟─0086dc4a-e0ba-43f0-a721-296cd50e1a76
# ╠═21479229-c2ad-425f-98bb-77717ab40b02
# ╠═c5edfcbf-8d31-4dc4-b9d0-1a5439540710
# ╠═cf4fb06d-98e5-47f0-9e9a-0f89d83ccf1f
# ╟─51274911-2eaa-4b18-b977-d0f735746bec
# ╟─afa6843b-9852-42c0-9ecd-06c408262334
# ╟─4b1d86d8-e6b9-4d09-b2a1-8c297414094c
# ╟─078eecc3-05b3-4a58-91c8-fc8c28c9b144
# ╟─d08aaab5-4065-41ef-867d-ea689c87a4f6
# ╟─b3c83ae6-9f3a-49a5-b726-872ef3a15853
# ╠═8d52f740-e1bf-4ac7-a299-38e7767a0831
# ╟─a4e10d47-2d41-4586-a328-11ea7234d7bd
# ╟─26a64c5d-6c21-4e3b-af75-e8682e8d5ea1
# ╠═6d671599-f635-4e1d-bef5-707891a756cc
# ╠═1441e34f-e8c5-46b5-b04c-31ea7f9f605a
# ╠═a36d205c-9a77-4b24-8e57-7bfceee9f4af
# ╠═bb71215f-fa97-4eca-a950-be4cb037bb00
# ╟─4fa824ba-51a3-4f5a-a990-dd05bbf2526a
# ╠═e047cce1-11a5-4bcb-8668-a767628da140
# ╠═d5bf43e7-07de-49c4-a2f2-39af6301d4b2
# ╠═c57219cc-00ce-441d-a944-03c79c554708
# ╠═771cca22-d61d-498a-98be-90fa59e09571
# ╟─8b6b5084-3972-4bd4-9ca2-423f1c627788
# ╠═e7b274c5-4f0c-4e5f-8a4a-b574130e64c0
# ╠═2fbdb817-da15-4011-bb1c-126f1f311e7a
# ╠═72895891-9212-4722-b2a1-0e13c30a8ecf
# ╠═07245a98-cab2-4b0c-a17a-4eaaa8a30703
# ╠═f5c3d5a4-7fe8-420e-af0a-4318b1eeda2c
# ╠═4bd9d7a4-979d-492f-b863-8359864004ea
# ╠═32832503-d48b-48bb-be7b-cf2cb6855a57
# ╟─fbe8691b-6d71-4cba-90e4-5de63421f634
# ╠═b1d56779-9a06-4b25-9a1b-09a12923e646
# ╠═f3c3f934-6601-4383-8204-55d04c973881
# ╠═8f158e95-a308-430e-b0a5-265a3c1ea643
# ╠═5a88de5e-5837-41c8-8150-b8d65ffc2fdf
# ╠═d5366fcf-39ee-48dc-8ee6-ca5c3141418a
# ╠═21e8ecc5-fe50-4cdf-8402-fe02db5dc13a
# ╠═2d7b4d32-d571-42eb-b31f-90a3bfe8a8c5
# ╠═a9b96979-0c20-4f94-9903-34630af55668
# ╠═e199d7ed-4a99-4eff-a936-e8b860e73f2f
# ╟─862026e9-ebe6-4f2e-8832-086bbba8db17
# ╟─8f894492-260e-4ab0-87b6-c02216a631e6
# ╟─c80256a7-be4f-4407-b0bf-7a13415482ad
# ╟─e6782d51-175c-4de7-9c75-1fc3f75a92f0
# ╠═013c2268-6ab8-441a-9fb4-5118dc3ae18a
# ╠═44a16c0a-9d0d-4e9b-9ae5-aef791c4f544
# ╠═13756b5d-b496-45fa-875b-7f6ee6468dcf
# ╠═2cafed7d-22c6-420f-9c8e-8ae734bfbad2
# ╠═a3484638-ae83-4810-9226-0a25b3fc58dc
# ╟─0358288e-be4e-46c2-ac4c-16ace6f50187
# ╠═67f08f89-698c-4aa4-80d5-1ebcb830fc0c
# ╠═8a581882-c97d-4a3b-873a-212024a529a9
# ╠═062f756b-6640-4928-9216-c54316503944
# ╠═f6125f11-8719-4c10-be91-3fe981e2d921
# ╠═326b3355-7941-403b-bf1e-3031f585f666
# ╟─5616d294-892a-40bc-a35f-35e9e0ee55e2
# ╠═2b047cbe-4da3-4e40-8897-8ea83e70a84d
# ╠═9db3ed98-a94d-4adc-a45f-75eca432a1e9
# ╠═4a8bc15c-8f4d-4017-915f-d2b27c1a6bd0
# ╠═962ff704-5d43-4ecb-8e2d-4538c6ee71c5
# ╠═55727cd1-ebb0-4e5c-80e5-c6d047c4b1ba
# ╠═4160de31-3c8e-4051-b618-24112bbcc70e
# ╠═359a682a-add2-4fe2-af09-f67ffbd985a8
# ╠═f22355c6-1b48-4a8d-b5b0-b851f3dadd52
# ╠═c8be71e0-c82a-4260-9dcf-944962947ca2
# ╠═f9137b59-24cc-4636-8a42-3a751309f42b
# ╠═c231090d-6faf-46a8-ae08-fd8715ade241
# ╠═afa7fa78-283c-4053-9dff-7cc2adc0cc0e
# ╠═f9ef6328-06f0-47a5-9de3-9b00d02af7f6
# ╠═e89b2c6f-7433-434f-8e46-836870f272b6
# ╠═b39b1b8e-b70e-4860-b75c-86506433efa7
# ╠═0e3ae279-be7e-4e13-b1ea-2c0efced3162
# ╠═82adce34-923c-46b9-a1ed-f1c06be09e0f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
