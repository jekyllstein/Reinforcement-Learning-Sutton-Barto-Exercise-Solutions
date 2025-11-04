### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ bd3ad49f-f076-46de-a159-b7cbffabe3dc
using PlutoDevMacros

# ╔═╡ 7153065a-6e9e-4a03-9369-fcf63f8c238e
using StaticArrays, Random

# ╔═╡ 8a581882-c97d-4a3b-873a-212024a529a9
# ╠═╡ show_logs = false
@only_in_nb PlutoDevMacros.@frompackage @raw_str(joinpath(@__DIR__, "..", "ApproximationUtils.jl")) using ApproximationUtils

# ╔═╡ f6125f11-8719-4c10-be91-3fe981e2d921
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoUI, PlutoPlotly ,StatsBase, BenchmarkTools, PlutoProfile, HypertextLiteral, LaTeXStrings
	TableOfContents()
end
  ╠═╡ =#

# ╔═╡ 062f756b-6640-4928-9216-c54316503944
@only_in_nb begin
	include(joinpath(@__DIR__, "..", "Chapter-09", "Chapter_09_On-policy_Prediction_with_Approximation.jl"))
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
# ╠═╡ skip_as_script = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ d55aca40-b03a-4f6b-84e6-ced6c8f67da1
# ╠═╡ skip_as_script = true
#=╠═╡
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
  ╠═╡ =#

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

# ╔═╡ 14b3b28e-1351-4b45-9a57-bfab846a2ffd
#=╠═╡
@htl("""
<div style = "display: flex; height: 500px;">
$(offline_λ_error_random_walk(fig_12_3_n; num_trials = 200)) 
$(nsteptd_error_random_walk(fig_12_3_n; num_trials = 200))
</div>
""")
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

# ╔═╡ 0b2e26e4-e87a-49a8-a7a6-c5b225af82b3
begin
	decay_trace!(z, c) = z .*= c
	function decay_trace!(z::FCANNParams{T}, c::T) where T<:Real
		for i in eachindex(z.weights[1])
			decay_trace!(z.weights[1][i], c)
			decay_trace!(z.weights[2][i], c)
		end
		return z
	end
	function decay_trace!(z::FCANNParamsGPU, c::T) where T<:Real
		for i in eachindex(z.weights[1])
			for j in 1:2
				FCANN.cublasSscal(FCANN.cublas_handle, c, z.weights[j][i])
			end
		end
		return z
	end
end

# ╔═╡ 5542897d-eb37-4e96-ac33-d36fc8adb603
begin
	function update_trace!(z, γ, λ, ∇v) 
		z .= (γ*λ .* z) .+ ∇v
	end

	function update_trace!(z::Vector{T}, γ::T, λ::T, ∇v::BinaryFeatureVector) where T<:Real
		z .*= γ*λ 
		for i in 1:∇v.num_features
			j = ∇v.active_features[i] 
			z[j] .+ one(T)
		end
		return z
	end

	function update_trace!(z::Vector{T}, γ::T, λ::T, ∇v::StateAggregationFeatureVector) where T<:Real
		z .*= γ*λ 
		z[∇v.group_index] += one(T)
		return z
	end

	function update_trace!(z::FCANNParams, γ::Float32, λ::Float32, ∇v::FCANNParams)
		for i in eachindex(z.weights[1])
			update_trace!(z.weights[1][i], γ, λ, ∇v.weights[1][i])
			update_trace!(z.weights[2][i], γ, λ, ∇v.weights[2][i])
		end
		return z
	end

	function update_trace!(z::FCANNParamsGPU, γ::Float32, λ::Float32, ∇v::FCANNParamsGPU)
		for i in eachindex(z.weights[1])
			for j in 1:2
				FCANN.cublasSscal(FCANN.cublas_handle, γ*λ, z.weights[j][i])
				FCANN.cublasSaxpy(FCANN.cublas_handle, 1f0, ∇v.weights[j][i], z.weights[j][i])
			end
		end
		return z
	end
end

# ╔═╡ 9ec58129-a14f-40a9-9c41-809500181bdd
begin
	function zero_trace!(z::AbstractArray{T, N}) where {T<:Real, N}
		z .= zero(T)
	end
	function zero_trace!(z::FCANNParams)
		for i in eachindex(first(z.weights))
			zero_trace!(z.weights[1][i])
			zero_trace!(z.weights[2][i])
		end
	end
	zero_trace!(z::FCANNParamsGPU) = decay_trace!(z, 0f0)
end

# ╔═╡ 5610a0ba-60a8-4da6-8f68-50b1c5e82686
begin
	#note that this function will modify both parameters and the state representation vector as well as some of the keyword arguments
	function semi_gradient_TDλ!(parameters::P, initialize_state::Function, transition::Function, isterm::Function, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, state_representation::X, update_state_representation!::Function, estimate_value::Function, ∇v̂, update_value_gradient!::Function; α = one(T)/10, calculate_error::Function = (target, v̂, s) -> (v̂ - target) ^2, z::P = copy(parameters), save_episode_steps::Bool = false, kwargs...) where {P, X, T<:Real}
		#initialize records
		step_rewards = Vector{T}()
		episode_steps = Vector{Int64}()
		episode_rewards = Vector{T}()
		episode_errors = Vector{T}()

		#initialize variables
		s = initialize_state()
		update_state_representation!(state_representation, s)
		ep = 1
		step = 1
		epstep = 1
		eperr = zero(T)
		rtot = zero(T)
		zero_trace!(z)
		
		while (ep <= max_episodes) && (step <= max_steps)
			v̂ = update_value_gradient!(∇v̂, state_representation, parameters)
			
			#by default does: 
			# z .= (γ*λ .* z) .+ ∇v
			decay_trace!(z, γ*λ)
			update_params_with_gradient!(z, one(T), ∇v̂)
			# update_trace!(z, γ, λ, ∇v̂)
			
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
				update_state_representation!(state_representation, s′)
			else
				update_state_representation!(state_representation, s′)
				v̂′ = estimate_value(state_representation, parameters)
			end

			target = r + γ*v̂′
			δ = target - v̂

			eperr += calculate_error(target, v̂, s)

			if terminated
				push!(episode_errors, eperr / epstep)
				eperr = zero(T)
				epstep = 0
			end

			#by default does: 
			# parameters .+= α*δ .* z
			update_params_with_gradient!(parameters, α*δ, z)
			s = s′
			step += 1
			epstep += 1
			terminated && zero_trace!(z)
		end

		v̂, form_kwargs = form_state_value_function(estimate_value, update_state_representation!, state_representation, parameters)
		(value_function = v̂, episode_history = (errors = episode_errors, steps = episode_steps, rewards = episode_rewards), step_rewards = step_rewards, parameters = parameters, form_kwargs = form_kwargs, trace = z)
	end

	function semi_gradient_TDλ!(parameters::P, initialize_state::Function, transition::Function, isterm::Function, λ::T, num_steps::Integer, state_representation::X, update_state_representation!::Function, estimate_value::Function, ∇v̂, update_value_gradient!::Function; α = one(T)/10, α_r̄ = one(T)/100, calculate_error::Function = (target, v̂, s) -> (v̂ - target) ^2, z::P = copy(parameters), kwargs...) where {P, X, T<:Real}
		#initialize records
		reward_history = zeros(T, num_steps)
		error_history = zeros(T, num_steps)
		average_reward_history = zeros(T, num_steps)

		#initialize variables
		s = initialize_state()
		update_state_representation!(state_representation, s)
		zero_trace!(z)
		r̄ = zero(T)
		
		for step in 1:num_steps
			v̂ = update_value_gradient!(∇v̂, state_representation, parameters)
			
			#by default does: 
			# z .= (λ .* z) .+ ∇v
			decay_trace!(z, λ)
			update_params_with_gradient!(z, one(T), ∇v̂)
			
			(r, s′) = transition(s)
			reward_history[step] = r
			average_reward_history[step] = r̄

			mdp.isterm(s′) && error("$s′ is a terminal state and this method only applies to continuing tasks")
			
			update_state_representation!(state_representation, s′)
			v̂′ = estimate_value(state_representation, parameters)

			target = r - r̄ + v̂′
			δ = target - v̂

			r̄ += α_r̄*δ

			error_history[step] = calculate_error(target, v̂, s)

			#by default does: 
			# parameters .+= α*δ .* z
			update_params_with_gradient!(parameters, α*δ, z)
			s = s′
			step += 1
			epstep += 1
		end

		v̂, form_kwargs = form_state_value_function(estimate_value, update_state_representation!, state_representation, parameters)
		(value_function = v̂, history = (errors = error_history, rewards = reward_history, average_rewards = average_reward_history), parameters = parameters, form_kwargs = form_kwargs, trace = z)
	end

	#when evaluating an MRP, there is no policy and the transition is just from the environment
	semi_gradient_TDλ!(parameters, mrp::StateMRP, args...; kwargs...) = semi_gradient_TDλ!(parameters, mrp.initialize_state, s -> mrp.ptf(s), mrp.isterm, args...; kwargs...)

	#when evaluating an MDP, there is a policy and the transition uses it to select actions
	semi_gradient_TDλ!(parameters, mdp::StateMDP, π::Function, args...; kwargs...) = semi_gradient_TDλ!(parameters, mdp.initialize_state, s -> mdp.ptf(s, π), mdp.isterm, args...; kwargs...)
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

# ╔═╡ 1d3144f8-fecf-4c8f-8e47-626ad94ed15a
md"""
#### Linear Verison of TD$(\lambda)$
"""

# ╔═╡ 24468748-009d-42a3-918d-4ba18b23c9ed
begin
	semi_gradient_TDλ_linear(mrp::StateMRP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!; init_value::T = zero(T), parameters = initialize_linear_parameters(feature_vector, init_value), kwargs...) where T<:Real = semi_gradient_TDλ!(parameters, mrp, γ, λ, max_episodes, max_steps, feature_vector, update_feature_vector!, linear_value_function, deepcopy(feature_vector), update_linear_value_gradient!; kwargs...)

	semi_gradient_TDλ_linear(mrp::StateMRP, λ::T, num_steps::Integer, feature_vector, update_feature_vector!; init_value::T = zero(T), parameters = initialize_linear_parameters(feature_vector, init_value), kwargs...) where T<:Real = semi_gradient_TDλ!(parameters, mrp, λ, num_steps, feature_vector, update_feature_vector!, linear_value_function, deepcopy(feature_vector), update_linear_value_gradient!; kwargs...)
	
	semi_gradient_TDλ_linear(mdp::StateMDP, π::Function, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!; init_value::T = zero(T), parameters = initialize_linear_parameters(feature_vector, init_value), kwargs...) where T<:Real = semi_gradient_TDλ!(parameters, mdp, π, γ, λ, max_episodes, max_steps, feature_vector, update_feature_vector!, linear_value_function, deepcopy(feature_vector), update_linear_value_gradient!; kwargs...)

	semi_gradient_TDλ_linear(mdp::StateMDP, π::Function, λ::T, num_steps::Integer, feature_vector, update_feature_vector!; init_value::T = zero(T), parameters = initialize_linear_parameters(feature_vector, init_value), kwargs...) where T<:Real = semi_gradient_TDλ!(parameters, mdp, π, λ, num_steps, feature_vector, update_feature_vector!, linear_value_function, deepcopy(feature_vector), update_linear_value_gradient!; kwargs...)
end

# ╔═╡ be33e2cc-b6d7-48e1-bfbd-71a01f7ae161
md"""
#### Non-Linear Version of TD$(\lambda)$ with Neural Network
"""

# ╔═╡ 52098913-ea06-497d-afad-a9fef99fb428
begin
	function semi_gradient_TDλ_fcann(problem::Tuple, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; reslayers::Integer = 0, use_μP::Bool = true, parameters::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, 1, reslayers, use_μP), dropout = zero(T), activation_list = fill(true, length(hidden_layers)), l2 = zero(T), use_gpu::Bool = false, kwargs...) where T<:Real
		setup = setup_fcann_value_arguments(parameters, l2, dropout, use_μP, activation_list; use_gpu = use_gpu)
		
		!use_gpu && return semi_gradient_TDλ!(parameters, problem..., γ, λ, max_episodes, max_steps, feature_vector, update_feature_vector!, setup.value_function, setup.gradient, setup.update_gradient!; kwargs...)
		
		isempty(setup.gpu_args) && error("GPU backend is not available")
		
		output = semi_gradient_TDλ!(setup.gpu_args.params, problem..., γ, λ, max_episodes, max_steps, feature_vector, update_feature_vector!, setup.value_function, setup.gpu_args.gradient, setup.update_gradient!; kwargs...)
		FCANN.GPU2Host(parameters.weights, setup.gpu_args.params.weights)
		setup.gpu_args.cleanup_vars()
		FCANN.clear_gpu_data(output.trace.weights[1])
		FCANN.clear_gpu_data(output.trace.weights[2])
		(;output..., parameters = parameters)
	end

	function semi_gradient_TDλ_fcann(problem::Tuple, λ::T, num_steps::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; reslayers::Integer = 0, use_μP::Bool = true, parameters::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, 1, reslayers, use_μP), dropout = zero(T), activation_list = fill(true, length(hidden_layers)), l2 = zero(T), use_gpu::Bool = false, kwargs...) where T<:Real
		setup = setup_fcann_value_arguments(parameters, l2, dropout, use_μP, activation_list; use_gpu = use_gpu)

		!use_gpu && return semi_gradient_TDλ!(parameters, problem..., λ, num_steps, feature_vector, update_feature_vector!, setup.value_function, setup.gradient, setup.update_gradient!; kwargs...)
		
		isempty(setup.gpu_args) && error("GPU backend is not available")
		output = semi_gradient_TDλ!(setup.gpu_args.params, problem..., λ, num_steps, feature_vector, update_feature_vector!, setup.value_function, setup.gpu_args.gradient, setup.update_gradient!; kwargs...)
		FCANN.GPU2Host(parameters.weights, setup.gpu_args.params.weights)
		setup.gpu_args.cleanup_vars()
		FCANN.clear_gpu_data(output.trace.weights[1])
		FCANN.clear_gpu_data(output.trace.weights[2])
		(;output..., parameters = parameters)
	end

	semi_gradient_TDλ_fcann(mdp::StateMDP, π::Function, args...; kwargs...) = semi_gradient_TDλ_fcann((mdp, π), args...; kwargs...)
	
	semi_gradient_TDλ_fcann(mrp::StateMRP, args...; kwargs...) = semi_gradient_TDλ_fcann((mrp,), args...; kwargs...)
end

# ╔═╡ f92b8423-b05f-4058-ac91-4b3c6d447820
md"""
### *Continuous Random Walk Example*
"""

# ╔═╡ 3839f146-107c-45e9-bb94-b707982f4ce1
#=╠═╡
function test_fcann_tdλ_random_walk(λ::Float32, hidden_layers::Vector{Int64}; nstates = 1000, max_episodes = 100, max_steps = typemax(Int64), kwargs...)
	mrp = create_continuous_random_walk(1000)
	xmin = 1
	xmax = nstates
	scalex(x) = (2f0*(x - 1f0) / (nstates - 1)) - 1f0 #scale x to between -1 and 1
	function update_state_representation!(x::Vector{Float32}, s::Float32) 
		x[1] = scalex(s)
	end
	output = semi_gradient_TDλ_fcann(mrp, 1f0, λ, max_episodes, max_steps, [0f0], update_state_representation!, hidden_layers; kwargs...)
	plot(output.value_function.(1f0:1000f0), Layout(yaxis_range = [-1, 1]))
end
  ╠═╡ =#

# ╔═╡ 3ab94b9e-4f50-4162-8b27-f6a81595f42f
#=╠═╡
test_fcann_tdλ_random_walk(0.5f0, fill(8, 2); res_layers = 1, max_episodes = 1_000, α = 0.01f0, use_gpu=false)
  ╠═╡ =#

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
	function semi_gradient_TDλ(states::Vector{S}, initialize_state_index::Function, transition::Function, terminal_states::BitVector, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; parameters::Vector{T} = zeros(T, length(states)), kwargs...) where {T<:Real, S}
		@assert length(parameters) == length(states)

		feature_vector = StateAggregationFeatureVector(length(parameters))

		#the state representation just stores the index of the state
		function update_feature_vector!(x::StateAggregationFeatureVector, i_s::Integer)
			x.group_index = i_s
		end
		
		semi_gradient_TDλ!(parameters, initialize_state_index, transition, i_s -> terminal_states[i_s], γ, λ, max_episodes, max_steps, feature_vector, update_feature_vector!, linear_value_function, deepcopy(feature_vector), update_linear_value_gradient!; kwargs...)
	end

	semi_gradient_TDλ(mrp::TabularMRP, args...; kwargs...) = semi_gradient_TDλ(mrp.states, mrp.initialize_state_index, i_s -> mrp.ptf(i_s), mrp.terminal_states, args...; kwargs...)

	semi_gradient_TDλ(mdp::TabularMDP, π::Function, args...; kwargs...) = semi_gradient_TDλ(mdp.states, mdp.initialize_state_index, i_s -> mdp.ptf(i_s, π), mdp.terminal_states, args...; kwargs...)
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
	output = semi_gradient_TDλ(mrp, 1f0, λ, num_episodes, typemax(Int64); save_episode_errors = true, α = α, calculate_error = calc_err, kwargs...)
	return sqrt(mean(output.episode_history.errors))
end
  ╠═╡ =#

# ╔═╡ b68f1171-6274-4d93-bf68-05b95cb5b2f8
#=╠═╡
run_random_walk_TDλ_estimation_trials(mrp::TabularMRP, calc_err, α, λ; num_trials = 100, kwargs...) = (1:num_trials |> Map(_ -> run_random_walk_TDλ_estimation(mrp, calc_err, α, λ; num_episodes = 10, kwargs...)) |> foldxt(+)) / num_trials
  ╠═╡ =#

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
	calc_err2(target, v̂, i_s) = (v̂ - v_true[i_s])^2
	get_α_line1(λ) = α_vec |> Map(α -> run_random_walk_offline_λ_estimation_trials(mrp, nstates, calc_err, α, λ; num_episodes = num_episodes, kwargs...)) |> collect
	get_α_line2(λ) = α_vec |> Map(α -> run_random_walk_TDλ_estimation_trials(tabular_mrp, calc_err2, α, λ; num_episodes = num_episodes, kwargs...)) |> collect
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
	function true_online_TDλ!(parameters::Vector{T}, initialize_state::Function, transition::Function, isterm::Function, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, state_representation::X, update_state_representation!::Function; α = one(T)/10, calculate_error::Function = (target, v̂, s) -> (target - v̂)^2, ∇v̂::P = deepcopy(state_representation), z::Vector{T} = copy(parameters), save_episode_steps::Bool = false, kwargs...) where {P, X, T<:Real}
		#initialize records
		step_rewards = Vector{T}()
		episode_steps = Vector{Int64}()
		episode_rewards = Vector{T}()
		episode_errors = Vector{T}()

		#initialize variables
		s = initialize_state()
		update_state_representation!(state_representation, s)
		ep = 1
		step = 1
		epstep = 1
		eperr = zero(T)
		rtot = zero(T)

		#initialize eligibility vector to 0
		z .= zero(T)
		v_old = zero(T)
	
		while (ep <= max_episodes) && (step <= max_steps)
			v̂ = update_linear_value_gradient!(∇v̂, state_representation, parameters)
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
			else
				update_state_representation!(state_representation, s′)
				v̂′ = linear_value_function(state_representation, parameters)
			end

			
			z .*= γ*λ
			c = one(T) - α*γ*λ*linear_value_function(∇v̂, z)
			update_params_with_gradient!(z, c, ∇v̂)
			#by default computes this: z .= (γ*λ .* z) .+ (one(T) - α*γ*λ*dot(z, ∇v)) .* ∇v

			target = r + γ*v̂′
			eperr += calculate_error(target, v̂, s)
			δ = target - v̂
			a = v̂ - v_old

			update_params_with_gradient!(parameters, α*(δ + a), z)
			update_params_with_gradient!(parameters, -α*a, ∇v̂)
			#by default computes this: parameters .+= α .* ((δ + a) .* z .- (a .* ∇v))

			if terminated
				s = initialize_state()
				update_state_representation!(state_representation, s)
				#reset eligibility vector to 0 at the start of a new episode
				z .= zero(T)
				v_old = zero(T)
				push!(episode_errors, eperr / epstep)
				eperr = zero(T)
				epstep = 0
			else
				s = s′
				v_old = v̂′
			end
			step += 1
			epstep += 1
		end
		v̂, form_kwargs = form_state_value_function(linear_value_function, update_state_representation!, state_representation, parameters)
		(value_function = v̂, episode_history = (errors = episode_errors, steps = episode_steps, rewards = episode_rewards), step_rewards = step_rewards, parameters = parameters, form_kwargs = form_kwargs)
	end

	#when evaluating an MRP, there is no policy and the transition is just from the environment
	true_online_TDλ!(parameters::Vector{T}, mrp::StateMRP, args...; kwargs...) where T<:Real = true_online_TDλ!(parameters, mrp.initialize_state, s -> mrp.ptf(s), mrp.isterm, args...; kwargs...)

	#when evaluating an MDP, there is a policy and the transition uses it to select actions
	true_online_TDλ!(parameters::Vector{T}, mdp::StateMDP, π::Function, args...; kwargs...) where {T<:Real} = true_online_TDλ!(parameters, mdp.initialize_state, s -> mdp.ptf(s, π), mdp.isterm, args...; kwargs...)
end

# ╔═╡ 34a28cfa-bf18-4dcf-8cf4-f6e9031d6fc2
begin
	#convenience functions for calling true online algorithm on an MDP+policy or MRP problem
	true_online_TDλ(problem::Tuple, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, state_representation::LinearFeatureVector, update_state_representation!::Function; init_value::T = zero(T), parameters::Vector{T} = initialize_linear_parameters(state_representation, init_value), kwargs...) where T<:Real = true_online_TDλ!(parameters, problem..., γ, λ, max_episodes, max_steps, state_representation, update_state_representation!; kwargs...)
	
	true_online_TDλ(mdp::StateMDP, π::Function, args...; kwargs...) = true_online_TDλ((mdp, π), args...; kwargs...)
	
	true_online_TDλ(mrp::StateMRP, args...; kwargs...) = true_online_TDλ((mrp,), args...; kwargs...)
end

# ╔═╡ d1903e34-0463-4a63-a74b-fb827451e542
begin
	#in the case of a tabular problem, this algorithm can be used with a trivial version of the linear algorithm
	function true_online_TDλ(states::Vector{S}, initialize_state_index::Function, transition::Function, terminal_states::BitVector, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; parameters::Vector{T} = zeros(T, length(states)), kwargs...) where {T<:Real, S}
		@assert length(parameters) == length(states)

		feature_vector = StateAggregationFeatureVector(length(parameters))

		#the state representation just stores the index of the state
		function update_feature_vector!(x::StateAggregationFeatureVector, i_s::Integer)
			x.group_index = i_s
		end
		
		true_online_TDλ!(parameters, initialize_state_index, transition, i_s -> terminal_states[i_s], γ, λ, max_episodes, max_steps, feature_vector, update_feature_vector!; kwargs...)
	end

	true_online_TDλ(mrp::TabularMRP, args...; kwargs...) = true_online_TDλ(mrp.states, mrp.initialize_state_index, i_s -> mrp.ptf(i_s), mrp.terminal_states, args...; kwargs...)

	true_online_TDλ(mdp::TabularMDP, π::Function, args...; kwargs...) = true_online_TDλ(mdp.states, mdp.initialize_state_index, i_s -> mdp.ptf(i_s, π), mdp.terminal_states, args...; kwargs...)
end

# ╔═╡ d9f89b2c-8df8-415a-a0a8-21744be88cec
#=╠═╡
function run_random_walk_true_online_TDλ_estimation(mrp::TabularMRP, calc_err::Function, α, λ; num_episodes = 10, kwargs...)
	output = true_online_TDλ(mrp, 1f0, λ, num_episodes, typemax(Int64); α = α, calculate_error = calc_err, kwargs...)
	return sqrt(mean(output.episode_history.errors))
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
	calc_err2(target, v̂, i_s) = (v̂ - v_true[i_s])^2
	get_α_line1(λ) = α_vec |> Map(α -> run_random_walk_offline_λ_estimation_trials(mrp, nstates, calc_err, α, λ; num_episodes = num_episodes, kwargs...)) |> collect
	get_α_line2(λ) = α_vec |> Map(α -> run_random_walk_true_online_TDλ_estimation_trials(tabular_mrp, calc_err2, α, λ; num_episodes = num_episodes, kwargs...)) |> collect
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

# ╔═╡ 43ba01e3-27d0-481c-8f93-8cd9c30b2a1d
begin
	abstract type AbstractEligibilityTrace end
	struct ReplacingTrace <: AbstractEligibilityTrace end
	struct AccumulatingTrace <: AbstractEligibilityTrace end
	struct DutchTrace <: AbstractEligibilityTrace end
end

# ╔═╡ 58ea2a87-3a8c-4fa3-877e-b1aa4b0ba36f
begin
	#accumulating traces just add the gradient to z
	update_trace_with_gradient!(z::Union{Array{T, N}, FCANNParams{T}}, ∇, ::AccumulatingTrace) where {T<:Real, N} = update_params_with_gradient!(z, one(T), ∇)
	update_trace_with_gradient!(z::FCANNParamsGPU, ∇, ::AccumulatingTrace) = update_params_with_gradient!(z, 1f0, ∇)

	#the first step to a dutch trace is to apply the accumulating trace
	update_trace_with_gradient!(z, ∇q̂, ::DutchTrace) = update_trace_with_gradient!(z, ∇q̂, AccumulatingTrace())

	#replacing traces involve overwriting the eligibility trace value for any active feature of the current state to 1.  it is a technique that only has meaning for binary features and linear approximation, therefore I will throw an error whenever a replacing trace is attempted with the wrong technique
	function update_trace_with_gradient!(z::Vector{T}, ∇::BinaryFeatureVector, ::ReplacingTrace) where T<:Real 
		for i in 1:∇.num_features
			j = ∇.active_features[i]
			z[j] = one(T)
		end
		return z
	end

	function update_trace_with_gradient!(z::Vector{T}, ∇::StateAggregationFeatureVector, ::ReplacingTrace) where T<:Real 
		z[∇.group_index] = one(T)
		return z
	end

	function update_trace_with_gradient!(z::Matrix{T}, ∇::LinearActionValueGradient{I, V}, ::ReplacingTrace) where {T<:Real, I<:Integer, V<:BinaryFeatureVector}
		for i in 1:∇.action_gradient.num_features
			j = ∇.action_gradient.active_features[i]
			z[j, ∇.action_index] = one(T)
		end
		return z
	end

	function update_trace_with_gradient!(z::Matrix{T}, ∇::LinearActionValueGradient{I, V}, ::ReplacingTrace) where {T<:Real, I<:Integer, V<:StateAggregationFeatureVector}
		z[∇.action_gradient.group_index, ∇.action_index] = one(T)
		return z
	end

	update_trace_with_gradient!(z::Vector{T}, ∇q̂::Vector{T}, ::ReplacingTrace) where {T<:Real} = error("Attempting to use replacing trace with non-binary features")

	update_trace_with_gradient!(z::Matrix{T}, ∇q̂::LinearActionValueGradient{I, V}, ::ReplacingTrace) where {T<:Real, I<:Integer, V<:Vector{T}} = error("Attempting to use replacing trace with non-binary features")

	update_trace_with_gradient!(z::FCANNParams{T}, ∇::FCANNParams{T}, ::ReplacingTrace) where {T<:Real} = error("Attempting to use replacing trace with non-linear approximation")
	update_trace_with_gradient!(z::FCANNParamsGPU, ∇::FCANNParamsGPU, ::ReplacingTrace) = error("Attempting to use replacing trace with non-linear approximation")
end

# ╔═╡ 01024201-7989-43f0-86f2-d3cd5ee219ae
begin
	#for accumulating and replacing traces do not do anything to z
	apply_dutch_trace!(z, c, feature_vector, value_function, ∇, ::AccumulatingTrace) = z
	apply_dutch_trace!(z, c, feature_vector, value_function, ∇, ::ReplacingTrace) = z
	apply_dutch_trace!(z, c, feature_vector, i_a, update_action_values!, action_values, ∇, ::AccumulatingTrace) = z
	apply_dutch_trace!(z, c, feature_vector, i_a, update_action_values!, action_values, ∇, ::ReplacingTrace) = z

	#for linear approximation just apply the parameter update with a constant c which is computed ahead of time and passed into the function and the value function at the state using the eligibility trace as the parameters 
	function apply_dutch_trace!(z, c, feature_vector, value_function::Function, ∇, ::DutchTrace)
		vz = value_function(feature_vector, z)
		update_params_with_gradient!(z, c*vz, ∇)
	end

	#when we use action values, we only need the action value associated with the action taken
	function apply_dutch_trace!(z, c::T, feature_vector, i_a::Integer, update_action_values!::Function, action_values::Vector{T}, ∇, ::DutchTrace)  where {T<:Real}
		update_action_values!(action_values, feature_vector, z)
		update_params_with_gradient!(z, c*action_values[i_a], ∇)
	end
end

# ╔═╡ 0525812d-7a86-4c5b-b5a8-36b4cfbd51fe
md"""
### *Vanilla Implementation*
"""

# ╔═╡ 3a03bd95-f43d-47b5-93d6-88a9ef61d2b4
get_num_actions(mdp::TabularMDP) = length(mdp.actions)

# ╔═╡ 31926565-8c2f-42a9-bc73-4f3001a38bf4
md"""
### *Dynamic Programming Implementation*
"""

# ╔═╡ 9c8765f5-0101-47e3-8780-65c197c14d6b
function dp_λ!(parameters::P, mdp::StateMDP{T, S, A, TR, F1, F2, F3}, λ::T, num_steps::Integer, feature_vector, update_feature_vector!::Function, value_function::Function, ∇v̂, update_value_gradient!::Function; α = one(T)/10, α_r̄::T = one(T)/10, ϵ = one(T) / 10, z::P = copy(parameters), action_values::Matrix{T} = zeros(T, length(mdp.actions), 1), compute_value::Function = compute_sarsa_value, trace_type::AbstractEligibilityTrace = AccumulatingTrace(), α_decay::T = one(T), decay_step::Integer = typemax(Int64), kwargs...) where {T<:Real, P, S, A, TR <: Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1, F2, F3}
	#initialize records
	reward_history = zeros(T, num_steps)
	average_reward_history = zeros(T, num_steps)

	action_value_args = form_action_value_args(mdp, feature_vector, parameters)

	#initialize variables
	decay = one(T)
	s = mdp.initialize_state()
	zero_trace!(z)
	
	policy = copy(action_values)
	r̄ = zero(T)
	
	for step in 1:num_steps
		update_feature_vector!(feature_vector, s)
		v̂ = update_value_gradient!(∇v̂, feature_vector, parameters)
		
		update_trace_with_gradient!(z, ∇v̂, trace_type)
		apply_dutch_trace!(z, -α*λ, feature_vector, value_function, ∇v̂, trace_type)

		average_reward_history[step] = r̄
		target, i_a_max = update_differential_action_values!(action_values, s, feature_vector, update_feature_vector!, value_function, parameters, mdp, r̄, action_value_args...; kwargs...)
		δ = target - v̂
		r̄ += α_r̄*δ
		
		decay *= (step > decay_step)*α_decay + (step <= decay_step)
		update_params_with_gradient!(parameters, α*decay*δ, z)

		policy .= action_values
		make_ϵ_greedy_policy!(policy; ϵ = ϵ)
		i_a = sample_action(policy)

		#take action and observe transition
		(r, s′) = mdp.ptf(s, i_a)
		mdp.isterm(s′) && error("$s′ is a terminal state and this method only applies to continuing tasks")
		reward_history[step] = r
		

		if !isapprox(target, action_values[i_a])
			#if the selected action is not the greedy action then zero out the trace since it is an off policy selection
			zero_trace!(z)
		else
			decay_trace!(z, λ)
		end

		s = s′
		step += 1
	end
	
	q̂, form_kwargs = form_differential_value_function(mdp, r̄, update_feature_vector!, value_function, feature_vector, parameters)
	
	return (value_function = q̂, reward_history = reward_history, average_reward_history = average_reward_history, final_parameters = deepcopy(parameters), form_kwargs = form_kwargs, trace = z)
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
#### Sarsa$(λ)$ Parameter Study
"""

# ╔═╡ 38512c69-54dc-4eb6-9a22-2a6bb3e38d89
#=╠═╡
@bind study_trace_type Select([ReplacingTrace() => "replacing trace", AccumulatingTrace() => "acculating trace", DutchTrace() => "dutch trace"])
  ╠═╡ =#

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

# ╔═╡ 5c424e53-aeba-4a11-97c8-a9936d6ddb72
function setup_parameter_study(f::Function, mandatory_args::NTuple{N, T}, default_args::NamedTuple) where {N, T<:Symbol}
	results = Dict{NamedTuple, Real}()

	function update_results!(args...; overwrite_results::Bool = false, usethreads::Bool = true, num_trials = Base.Threads.nthreads(), kwargs...)
		key = (;NamedTuple{mandatory_args}(args)..., default_args..., kwargs..., num_trials = num_trials)
		haskey(results, key) && !overwrite_results && return results[key]
		
		Random.seed!(num_trials)
		tr = 1:num_trials |> Map(_ -> f(args...; default_args..., kwargs...)) 
		output = usethreads ? foldxt(+, tr) : foldxl(+, tr)
		results[key] = output/num_trials
	end

	return (results = results, update_results! = update_results!)
end

# ╔═╡ 06885c43-2ace-45ff-b77f-f3ceaaf999de
md"""
#### Expected Sarsa$(λ)$ Parameter Study
"""

# ╔═╡ 4fa824ba-51a3-4f5a-a990-dd05bbf2526a
md"""
### True Online Sarsa(λ) for Tabular Problems or Linear Approximation
"""

# ╔═╡ 7aa62007-0685-40d2-88ab-9c03add8e75a
md"""
#### True Online Sarsa$(λ)$ Parameter Study
"""

# ╔═╡ a09643e9-d2c8-4fe0-a320-205badde72bf
md"""
#### True Online Expected Sarsa$(λ)$ Parameter Study
"""

# ╔═╡ 1d49a787-2a82-4f6d-a986-f2351fa82d18
function update_action_values!(action_values::Vector{T}, i_s::Integer, state_action_values::Matrix{T}) where T<:Real
	@inbounds @simd for i_a in eachindex(action_values)
		action_values[i_a] = state_action_values[i_a, i_s]
	end
	return action_values
end

# ╔═╡ 65aa9202-7e9a-4590-865e-0b08c7b98e1e
function form_value_function(mdp::TabularMDP, parameters::Matrix{T}) where T<:Real
	function q̂(s; action_values::Vector{T} = zeros(T, length(mdp.actions)))
		i_s = mdp.state_index[s]
		update_action_values!(action_values, i_s, parameters)
		(qmax, i_a_max) = findmax(action_values)
		return (maximizing_action = i_a_max, maximizing_value = qmax)
	end

	form_kwargs() = (action_values = zeros(T, length(mdp.actions)),)
	return (value_function = q̂, form_kwargs = form_kwargs)
end

# ╔═╡ b320dc0e-95dc-44d5-8ee4-455c4a858835
function sarsa_λ!(parameters::P, mdp::StateMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, update_action_values!::Function, ∇q̂, update_value_gradient!::Function; α = one(T)/10, ϵ = one(T) / 10, z::P = copy(parameters), action_values::Vector{T} = zeros(T, length(mdp.actions)), compute_value::Function = compute_sarsa_value, save_parameter_history::Bool = false, trace_type::AbstractEligibilityTrace = AccumulatingTrace(), α_decay::T = one(T), decay_step::Integer = typemax(Int64), kwargs...) where {T<:Real, P}
	#initialize records
	episode_rewards = Vector{T}()
	episode_steps = Vector{Int64}()
	parameter_history = Vector{P}()
	save_parameter_history && push!(parameter_history, deepcopy(parameters))

	#initialize variables
	ep = 1
	step = 1
	epreward = zero(T)
	decay = one(T)
	s = mdp.initialize_state()
	update_feature_vector!(feature_vector, s)
	update_action_values!(action_values, feature_vector, parameters)
	policy = copy(action_values)
	make_ϵ_greedy_policy!(policy; ϵ = ϵ)
	i_a = sample_action(policy)
	zero_trace!(z)
	
	
	while (ep <= max_episodes) && (step <= max_steps)
		update_value_gradient!(∇q̂, action_values, feature_vector, i_a, parameters)
		q̂ = action_values[i_a]

		update_trace_with_gradient!(z, ∇q̂, trace_type)
		apply_dutch_trace!(z, -α*γ*λ, feature_vector, i_a, update_action_values!, action_values, ∇q̂, trace_type)

		#take action and observe transition
		(r, s′) = mdp.ptf(s, i_a)
		epreward += r

		terminated = mdp.isterm(s′)

		if terminated
			s′ = mdp.initialize_state()
			push!(episode_rewards, epreward)
			push!(episode_steps, step)
			epreward = zero(T)
			ep += 1
		end

		update_feature_vector!(feature_vector, s′)
		update_action_values!(action_values, feature_vector, parameters)
		policy .= action_values
		make_ϵ_greedy_policy!(policy; ϵ = ϵ)
		i_a′ = sample_action(policy)

		q̂′ = terminated ? zero(T) : compute_value(action_values, policy, i_a′)

		target = r + γ*q̂′

		δ = target - q̂

		decay *= (step > decay_step)*α_decay + (step <= decay_step)

		update_params_with_gradient!(parameters, α*decay*δ, z)

		save_parameter_history && push!(parameter_history, deepcopy(parameters))
		s = s′
		i_a = i_a′
		step += 1

		if terminated 
			zero_trace!(z)
		else
			decay_trace!(z, λ*γ)
		end
	end
	
	q̂, form_kwargs = form_value_function(mdp, update_feature_vector!, update_action_values!, feature_vector, parameters)
	
	return (value_function = q̂, episode_rewards = episode_rewards, episode_steps = episode_steps, parameter_history = parameter_history, final_parameters = deepcopy(parameters), form_kwargs = form_kwargs, trace = z)
end

# ╔═╡ 3df4cd98-f754-4eca-8e16-e654576e283d
function sarsa_λ!(parameters::P, mdp::StateMDP, λ::T, num_steps::Integer, feature_vector, update_feature_vector!::Function, update_action_values!::Function, ∇q̂, update_value_gradient!::Function; α = one(T)/10, ϵ = one(T) / 10, α_r̄::T = one(T)/10, z::P = copy(parameters), action_values::Vector{T} = zeros(T, length(mdp.actions)), compute_value::Function = compute_sarsa_value, trace_type::AbstractEligibilityTrace = AccumulatingTrace(), α_decay::T = one(T), decay_step::Integer = typemax(Int64), kwargs...) where {T<:Real, P}
	#initialize records
	reward_history = zeros(T, num_steps)
	average_reward_history = zeros(T, num_steps)
	
	#initialize variables
	ep = 1
	step = 1
	decay = one(T)
	s = mdp.initialize_state()
	update_feature_vector!(feature_vector, s)
	update_action_values!(action_values, feature_vector, parameters)
	policy = copy(action_values)
	make_ϵ_greedy_policy!(policy; ϵ = ϵ)
	i_a = sample_action(policy)
	zero_trace!(z)
	r̄ = zero(T)
	
	
	for step in 1:num_steps
		update_value_gradient!(∇q̂, action_values, feature_vector, i_a, parameters)
		q̂ = action_values[i_a]
		
		update_trace_with_gradient!(z, ∇q̂, trace_type)
		apply_dutch_trace!(z, -α*λ, feature_vector, i_a, update_action_values!, action_values, ∇q̂, trace_type)

		#take action and observe transition
		(r, s′) = mdp.ptf(s, i_a)
		reward_history[step] = r
		average_reward_history[step] = r̄

		mdp.isterm(s′) && error("$s′ is a terminal state and this method only applies to continuing tasks")

		update_feature_vector!(feature_vector, s′)
		update_action_values!(action_values, feature_vector, parameters)
		policy .= action_values
		make_ϵ_greedy_policy!(policy; ϵ = ϵ)
		i_a′ = sample_action(policy)

		q̂′ = compute_value(action_values, policy, i_a′)

		target = r - r̄ + q̂′

		δ = target - q̂
		r̄ += α_r̄*δ

		decay *= (step > decay_step)*α_decay + (step <= decay_step)

		update_params_with_gradient!(parameters, α*decay*δ, z)

		s = s′
		i_a = i_a′
		step += 1

		decay_trace!(z, λ)
		
	end
	
	q̂, form_kwargs = form_value_function(mdp, update_feature_vector!, update_action_values!, feature_vector, parameters)
	
	return (value_function = q̂, reward_history = reward_history, average_reward_history = average_reward_history, final_parameters = deepcopy(parameters), form_kwargs = form_kwargs, trace = z)
end

# ╔═╡ efa11915-d86b-4686-85f9-84d7539e27cf
sarsa_λ_linear(mdp::StateMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), parameters::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, init_value), kwargs...) where T<:Real = sarsa_λ!(parameters, mdp, γ, λ, max_episodes, max_steps, feature_vector, update_feature_vector!, update_linear_action_values!, LinearActionValueGradient(deepcopy(feature_vector), 0), update_linear_value_gradient!; kwargs...)

# ╔═╡ fc9149ee-6d28-41b5-ab47-2c2a36e7a8d1
sarsa_λ_linear(mdp::StateMDP, λ::T, num_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), parameters::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, init_value), kwargs...) where T<:Real = sarsa_λ!(parameters, mdp, λ, num_steps, feature_vector, update_feature_vector!, update_linear_action_values!, LinearActionValueGradient(deepcopy(feature_vector), 0), update_linear_value_gradient!; kwargs...)

# ╔═╡ c7caa90d-26bf-4179-b869-3385cb75b943
function sarsa_λ(mdp::TabularMDP{T, S, A, P, F}, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; feature_vector = StateAggregationFeatureVector(length(mdp.states)), init_value::T = zero(T), parameters::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, init_value), kwargs...) where {T<:Real, S, A, P, F}
	function update_feature_vector!(v, s::S)
		i_s = mdp.state_index[s]
		v.group_index = i_s
		return v
	end
	
	sarsa_λ_linear(StateMDP(mdp), γ, λ, max_episodes, max_steps, feature_vector, update_feature_vector!; kwargs...)
end

# ╔═╡ 58739629-eff4-416a-b133-85ab5ec563fc
function sarsa_λ(mdp::TabularMDP{T, S, A, P, F}, λ::T, num_steps::Integer; feature_vector = StateAggregationFeatureVector(length(mdp.states)), init_value::T = zero(T), parameters::Matrix{T} = initialize_linear_parameters(feature_vector, mdp, init_value), kwargs...) where {T<:Real, S, A, P, F}
	function update_feature_vector!(v, s::S)
		i_s = mdp.state_index[s]
		v.group_index = i_s
		return v
	end
	
	sarsa_λ_linear(StateMDP(mdp), λ, num_steps, feature_vector, update_feature_vector!; kwargs...)
end

# ╔═╡ 2c8beba6-4436-4603-88f2-20f847c5e916
function test_sarsa_λ(; kwargs...)
	mdp = make_stochastic_gridworld(;stepreward = -1f0, termreward = 0f0)
	sarsa_λ(mdp, 1f0, -.5f0, 10, 10000; kwargs...)
end

# ╔═╡ 84d71a29-7e38-4877-bfb8-57d4af2ec0d0
test_sarsa_λ()

# ╔═╡ 84870ff4-d7a5-4214-abcb-3d74b7a8fe7b
function sarsa_λ_fcann(mdp::StateMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; reslayers::Integer = 0, use_μP::Bool = true, parameters::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, length(mdp.actions), reslayers, use_μP), dropout = zero(T), activation_list = fill(true, length(hidden_layers)), l2 = zero(T), use_gpu::Bool = false, kwargs...) where T<:Real 
	setup = setup_fcann_action_value_arguments(parameters, l2, dropout, use_μP, activation_list; use_gpu = use_gpu)
	!use_gpu && return sarsa_λ!(parameters, mdp, γ, λ, max_episodes, max_steps, feature_vector, update_feature_vector!, setup.update_action_values!, setup.gradient, setup.update_value_gradient!; kwargs...)

	isempty(setup.gpu_args) && error("GPU backend is not available")
	gpu_feature_update! = setup_gpu_feature(feature_vector, update_feature_vector!)
	output = sarsa_λ!(setup.gpu_args.params, mdp, γ, λ, max_episodes, max_steps, setup.gpu_args.feature_vector, gpu_feature_update!, setup.update_action_values!, setup.gpu_args.gradient, setup.update_value_gradient!; kwargs...)
	FCANN.GPU2Host(parameters.weights, setup.gpu_args.params.weights)
	setup.gpu_args.cleanup_vars()
	FCANN.clear_gpu_data(output.trace.weights[1])
	FCANN.clear_gpu_data(output.trace.weights[2])
	(;output..., final_parameters = parameters)
end

# ╔═╡ 747700de-0a87-4ac9-a9cd-0bc11721836e
function sarsa_λ_fcann(mdp::StateMDP, λ::T, num_steps::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; reslayers::Integer = 0, use_μP::Bool = true, parameters::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, length(mdp.actions), reslayers, use_μP), dropout = zero(T), activation_list = fill(true, length(hidden_layers)), l2 = zero(T), use_gpu::Bool = false, kwargs...) where T<:Real 
	setup = setup_fcann_action_value_arguments(parameters, l2, dropout, use_μP, activation_list; use_gpu = use_gpu)
	!use_gpu && return sarsa_λ!(parameters, mdp, λ, num_steps, feature_vector, update_feature_vector!, setup.update_action_values!, setup.gradient, setup.update_value_gradient!; kwargs...)

	isempty(setup.gpu_args) && error("GPU backend is not available")
	gpu_feature_update! = setup_gpu_feature(feature_vector, update_feature_vector!)
	output = sarsa_λ!(setup.gpu_args.params, mdp, λ, num_steps, setup.gpu_args.feature_vector, gpu_feature_update!, setup.update_action_values!, setup.gpu_args.gradient, setup.update_value_gradient!; kwargs...)
	FCANN.GPU2Host(parameters.weights, setup.gpu_args.params.weights)
	setup.gpu_args.cleanup_vars()
	FCANN.clear_gpu_data(output.trace.weights[1])
	FCANN.clear_gpu_data(output.trace.weights[2])
	(;output..., final_parameters = parameters)
end

# ╔═╡ 4cab7b59-f080-4bea-86dc-3c860a618c35
function dp_λ!(parameters::P, mdp::StateMDP{T, S, A, TR, F1, F2, F3}, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, value_function::Function, ∇v̂, update_value_gradient!::Function; α = one(T)/10, ϵ = one(T) / 10, z::P = copy(parameters), action_values::Matrix{T} = zeros(T, length(mdp.actions), 1), compute_value::Function = compute_sarsa_value, save_parameter_history::Bool = false, trace_type::AbstractEligibilityTrace = AccumulatingTrace(), α_decay::T = one(T), decay_step::Integer = typemax(Int64), kwargs...) where {T<:Real, P, S, A, TR <: Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1, F2, F3}
	#initialize records
	episode_rewards = Vector{T}()
	episode_steps = Vector{Int64}()
	parameter_history = Vector{P}()
	save_parameter_history && push!(parameter_history, deepcopy(parameters))

	action_value_args = form_action_value_args(mdp, feature_vector, parameters)

	#initialize variables
	ep = 1
	step = 1
	epreward = zero(T)
	decay = one(T)
	s = mdp.initialize_state()
	zero_trace!(z)
	
	policy = zeros(T, length(mdp.actions))
	
	while (ep <= max_episodes) && (step <= max_steps)
		update_feature_vector!(feature_vector, s)
		v̂ = update_value_gradient!(∇v̂, feature_vector, parameters)
		
		update_trace_with_gradient!(z, ∇v̂, trace_type)
		apply_dutch_trace!(z, -α*γ*λ, feature_vector, value_function, ∇v̂, trace_type)

		target, i_a_max = update_action_values!(action_values, s, feature_vector, update_feature_vector!, value_function, parameters, mdp, γ, action_value_args...; kwargs...)
		δ = target - v̂
		
		decay *= (step > decay_step)*α_decay + (step <= decay_step)
		update_params_with_gradient!(parameters, α*decay*δ, z)

		policy .= action_values
		make_ϵ_greedy_policy!(policy; ϵ = ϵ)
		i_a = sample_action(policy)

		#take action and observe transition
		(r, s′) = mdp.ptf(s, i_a)
		epreward += r

		if mdp.isterm(s′)
			s′ = mdp.initialize_state()
			push!(episode_rewards, epreward)
			push!(episode_steps, step)
			epreward = zero(T)
			zero_trace!(z)
			ep += 1
		elseif !isapprox(target, action_values[i_a])
			#if the selected action is not the greedy action then zero out the trace since it is an off policy selection
			zero_trace!(z)
		else
			decay_trace!(z, λ*γ)
		end

		save_parameter_history && push!(parameter_history, deepcopy(parameters))
		s = s′
		step += 1
	end
	
	q̂, form_kwargs = form_value_function(mdp, γ, update_feature_vector!, value_function, feature_vector, parameters)
	
	return (value_function = q̂, episode_rewards = episode_rewards, episode_steps = episode_steps, parameter_history = parameter_history, final_parameters = deepcopy(parameters), form_kwargs = form_kwargs, trace = z)
end

# ╔═╡ 258be19f-0b01-4d5e-ae82-2ef07ba4cc9c
dp_λ_linear(mdp::StateMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), parameters::Vector{T} = initialize_linear_parameters(feature_vector, init_value), kwargs...) where T<:Real = dp_λ!(parameters, mdp, γ, λ, max_episodes, max_steps, feature_vector, update_feature_vector!, linear_value_function, deepcopy(feature_vector), update_linear_value_gradient!; kwargs...)

# ╔═╡ c3fb9a00-3aa0-43e5-98c8-306441166cc4
dp_λ_linear(mdp::StateMDP, λ::T, num_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), parameters::Vector{T} = initialize_linear_parameters(feature_vector, init_value), kwargs...) where T<:Real = dp_λ!(parameters, mdp, λ, num_steps, feature_vector, update_feature_vector!, linear_value_function, deepcopy(feature_vector), update_linear_value_gradient!; kwargs...)

# ╔═╡ da6944b0-1d1b-4b92-a765-5e28c133cff5
function test_dp_λ(; kwargs...)
	mdp = make_stochastic_gridworld(;stepreward = -1f0, termreward = 0f0)
	function update_feature_vector!(v, s) 
		v.group_index = mdp.state_index[s]
	end
	dp_λ_linear(StateMDP(mdp), 1f0, -.5f0, 10, 10000, StateAggregationFeatureVector(length(mdp.states)), update_feature_vector!; kwargs...)
end

# ╔═╡ 08c7beae-9dee-468e-b276-aa56a8f24f1f
# ╠═╡ skip_as_script = true
#=╠═╡
test_dp_λ()
  ╠═╡ =#

# ╔═╡ 5f623b73-4d7d-4c69-acaf-9a668c352bf9
function dp_λ_fcann(mdp::StateMDP{T, S, A, P, F1, F2, F3}, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; reslayers::Integer = 0, use_μP::Bool = true, parameters::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, 1, reslayers, use_μP), dropout = zero(T), activation_list = fill(true, length(hidden_layers)), l2 = zero(T), use_gpu::Bool = false, kwargs...) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1, F2, F3} 
	setup = setup_fcann_value_arguments(parameters, l2, dropout, use_μP, activation_list; use_gpu = use_gpu)
	!use_gpu && return dp_λ!(parameters, mdp, γ, λ, max_episodes, max_steps, feature_vector, update_feature_vector!, setup.value_function, setup.gradient, setup.update_gradient!; kwargs...)

	isempty(setup.gpu_args) && error("GPU backend is not available")
	output = dp_λ!(setup.gpu_args.params, mdp, γ, λ, max_episodes, max_steps, feature_vector, update_feature_vector!, setup.value_function, setup.gpu_args.gradient, setup.update_gradient!; kwargs...)
	FCANN.GPU2Host(parameters.weights, setup.gpu_args.params.weights)
	setup.gpu_args.cleanup_vars()
	FCANN.clear_gpu_data(output.trace.weights[1])
	FCANN.clear_gpu_data(output.trace.weights[2])
	(;output..., final_parameters = parameters)
end

# ╔═╡ c5ae58b0-6f89-476a-8a72-9bb1cfd1a6be
function dp_λ_fcann(mdp::StateMDP{T, S, A, P, F1, F2, F3}, λ::T, num_steps::Integer, feature_vector, update_feature_vector!::Function, hidden_layers::Vector{Int64}; reslayers::Integer = 0, use_μP::Bool = true, parameters::FCANNParams{T} = initialize_fcann_params(feature_vector, hidden_layers, 1, reslayers, use_μP), dropout = zero(T), activation_list = fill(true, length(hidden_layers)), l2 = zero(T), use_gpu::Bool = false, kwargs...) where {T<:Real, S, A, P<:Union{StateMDPTransitionDistribution, StateMDPTransitionDeterministic}, F1, F2, F3} 
	setup = setup_fcann_value_arguments(parameters, l2, dropout, use_μP, activation_list; use_gpu = use_gpu)
	!use_gpu && return dp_λ!(parameters, mdp, λ, num_steps, feature_vector, update_feature_vector!, setup.value_function, setup.gradient, setup.update_gradient!; kwargs...)

	isempty(setup.gpu_args) && error("GPU backend is not available")
	output = dp_λ!(setup.gpu_args.params, mdp, λ, num_steps, feature_vector, update_feature_vector!, setup.value_function, setup.gpu_args.gradient, setup.update_gradient!; kwargs...)
	FCANN.GPU2Host(parameters.weights, setup.gpu_args.params.weights)
	setup.gpu_args.cleanup_vars()
	FCANN.clear_gpu_data(output.trace.weights[1])
	FCANN.clear_gpu_data(output.trace.weights[2])
	(;output..., final_parameters = parameters)
end

# ╔═╡ 65ca967c-2425-4c85-92e9-3f957e7ede2f
begin
	function linear_action_value_function(feature_vector::Vector{T}, i_a::Integer, parameters::Matrix{T}) where T<:Real
		x = zero(T)
		@inbounds @simd for i in eachindex(feature_vector)
			x += parameters[i, i_a]
		end
		return x
	end

	function linear_action_value_function(feature_vector::BinaryFeatureVector, i_a::Integer, parameters::Matrix{T}) where T<:Real
		x = zero(T)
		@inbounds @simd for i in 1:feature_vector.num_features
			j = feature_vector.active_features[i]
			x += parameters[j, i_a]
		end
		return x
	end

	function linear_action_value_function(feature_vector::StateAggregationFeatureVector, i_a::Integer, parameters::Matrix{T}) where T<:Real
		i = feature_vector.group_index
		return parameters[i, i_a]
	end
end

# ╔═╡ 771cca22-d61d-498a-98be-90fa59e09571
begin
	#tabular problem where the parameters are just the state action values and each state action pair only has one active feature
	function true_online_sarsa_λ!(state_action_values::Matrix{T}, mdp::TabularMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; α = one(T)/10, ϵ = one(T) / 10, z::Matrix{T} = copy(state_action_values), action_values::Vector{T} = zeros(T, length(mdp.actions)), compute_value::Function = compute_sarsa_value, save_parameter_history::Bool = false, save_step_rewards::Bool = false, kwargs...) where {T<:Real}
		#set the state action values of all terminal states to 0
		for i_s in eachindex(mdp.states)
			if mdp.terminal_states[i_s]
				state_action_values[:, i_s] .= zero(T)
			end
		end

		#initialize records
		episode_rewards = Vector{T}()
		episode_steps = Vector{Int64}()
		parameter_history = Vector{Matrix{T}}()

		policy = copy(action_values)
		
		#initialize episode
		i_s = mdp.initialize_state_index()
		update_action_values!(action_values, i_s, state_action_values)
		policy .= action_values
		make_ϵ_greedy_policy!(policy; ϵ = ϵ)
		i_a = sample_action(policy)
		z .= zero(T)
		q_old = zero(T)
		ep = 1
		step = 1
		rtot = zero(T)
		
		while (ep <= max_episodes) && (step <= max_steps)
			#take action and observe transition
			q = state_action_values[i_a, i_s]
			(r, i_s′) = mdp.ptf(i_s, i_a)
			rtot += r
			
			save_step_rewards && push!(step_rewards, r)

			if mdp.terminal_states[i_s′]
				q′ = zero(T)
			else
				update_action_values!(action_values, i_s′, state_action_values)
				policy .= action_values
				make_ϵ_greedy_policy!(policy; ϵ = ϵ)
				i_a′ = sample_action(policy)
				q′ = compute_value(action_values, policy, i_a′)
			end

			δ = r + γ*q′ - q

			dt = z[i_a, i_s]
			z .*= γ*λ
			z[i_a, i_s] += one(T) - α*γ*λ*dt

			state_action_values .+= α*(δ + q - q_old) .* z
			state_action_values[i_a, i_s] -= α*(q - q_old)

			save_parameter_history && push!(parameter_history, copy(state_action_values))

			if mdp.terminal_states[i_s′]
				i_s = mdp.initialize_state_index()
				update_action_values!(action_values, i_s, state_action_values)
				policy .= action_values
				make_ϵ_greedy_policy!(policy; ϵ = ϵ)
				i_a = sample_action(policy)
				#reset eligibility vector to 0 at the start of a new episode
				z .= zero(T)
				q_old = zero(T)
				ep += 1
				push!(episode_steps, step)
				push!(episode_rewards, rtot)
				rtot = zero(T)
			else
				i_s = i_s′
				i_a = i_a′
				q_old = q′
			end
			step += 1
		end

		q̂, form_kwargs = form_value_function(mdp, state_action_values)
	
		return (value_function = q̂, episode_rewards = episode_rewards, episode_steps = episode_steps, parameter_history = parameter_history, final_parameters = deepcopy(state_action_values), form_kwargs = form_kwargs)
	end

	#true online sarsaλ for linear features
	function true_online_sarsa_λ!(parameters::Matrix{T}, mdp::StateMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; α = one(T)/10, ϵ = one(T) / 10, z::Matrix{T} = copy(parameters), action_values::Vector{T} = zeros(T, length(mdp.actions)), compute_value::Function = compute_sarsa_value, save_parameter_history::Bool = false, kwargs...) where {T<:Real}
		policy = copy(action_values)
		
		#initialize records
		episode_rewards = Vector{T}()
		episode_steps = Vector{Int64}()
		parameter_history = Vector{Matrix{T}}()
	
		#initialize episode
		s = mdp.initialize_state()
		update_feature_vector!(feature_vector, s)
		update_linear_action_values!(action_values, feature_vector, parameters)
		policy .= action_values
		make_ϵ_greedy_policy!(policy; ϵ = ϵ)
		i_a = sample_action(policy)
		z .= zero(T)
		q_old = zero(T)
		ep = 1
		step = 1
		rtot = zero(T)
		
		while (ep <= max_episodes) && (step <= max_steps)
			q = action_values[i_a]

			#represents the portion of the eligibility trace update that depends on the current feature vector
			dt =  one(T) - α*γ*λ*linear_action_value_function(feature_vector, i_a, z)

			z .*= γ*λ

			update_params_with_gradient!(z, dt, feature_vector, i_a)

			update_params_with_gradient!(parameters, -α*(q - q_old), feature_vector, i_a)
			
			#take action and observe transition
			(r, s′) = mdp.ptf(s, i_a)
			rtot += r

			terminated = mdp.isterm(s′)
			if terminated
				push!(episode_rewards, rtot)
				push!(episode_steps, step)
				rtot = zero(T)
				q′ = zero(T)
			else
				update_feature_vector!(feature_vector, s′)
				update_linear_action_values!(action_values, feature_vector, parameters)
				policy .= action_values
				make_ϵ_greedy_policy!(policy; ϵ = ϵ)
				i_a′ = sample_action(policy)
				q′ = compute_value(action_values, policy, i_a′)
			end
			
			δ = r + γ*q′ - q

			parameters .+= α*(δ + q - q_old) .* z

			save_parameter_history && push!(parameter_history, copy(parameters))
			
			if terminated
				s = mdp.initialize_state()
				update_feature_vector!(feature_vector, s)
				update_linear_action_values!(action_values, feature_vector, parameters)
				make_ϵ_greedy_policy!(action_values; ϵ = ϵ)
				policy .= action_values
				i_a = sample_action(policy)
				#reset eligibility vector to 0 at the start of a new episode
				z .= zero(T)
				q_old = zero(T)
				ep += 1
			else
				s = s′
				i_a = i_a′
				q_old = q′
			end
			step += 1
		end

		q̂, form_kwargs = form_value_function(mdp, update_feature_vector!, update_linear_action_values!, feature_vector, parameters)
	
		return (value_function = q̂, episode_rewards = episode_rewards, episode_steps = episode_steps, parameter_history = parameter_history, final_parameters = deepcopy(parameters), form_kwargs = form_kwargs)
	end
end

# ╔═╡ b9467ce7-ace9-4034-a306-050bf1e53573
begin
	true_online_sarsa_λ(mdp::TabularMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; init_value::T = zero(T), state_action_values::Matrix{T} = initialize_state_action_value(mdp; init_value = init_value), kwargs...) where T<:Real = true_online_sarsa_λ!(state_action_values, mdp, γ, λ, max_episodes, max_steps; kwargs...)
	
	true_online_sarsa_λ(mdp::StateMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), parameters = initialize_linear_parameters(feature_vector, mdp, init_value), kwargs...) where T<:Real = true_online_sarsa_λ!(parameters, mdp, γ, λ, max_episodes, max_steps, feature_vector, update_feature_vector!; kwargs...)
end

# ╔═╡ ded7c8e0-f44c-44c9-afad-070d325c180b
#true online dp λ for linear features
function true_online_dp_λ!(parameters::Vector{T}, mdp::StateMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; α = one(T)/10, ϵ = one(T) / 10, z::Vector{T} = copy(parameters), action_values::Matrix{T} = zeros(T, length(mdp.actions), 1), save_parameter_history::Bool = false, kwargs...) where {T<:Real}
	policy = zeros(T, length(mdp.actions))
	action_value_args = form_action_value_args(mdp, feature_vector, parameters)
	
	#initialize records
	episode_rewards = Vector{T}()
	episode_steps = Vector{Int64}()
	parameter_history = Vector{Vector{T}}()
	
	#initialize episode
	s = mdp.initialize_state()
	target, i_a_max = update_action_values!(action_values, s, feature_vector, update_feature_vector!, linear_value_function, parameters, mdp, γ, action_value_args...)
	policy .= action_values
	make_ϵ_greedy_policy!(policy; ϵ = ϵ)
	i_a = sample_action(policy)
	z .= zero(T)
	v_old = zero(T)
	ep = 1
	step = 1

	rtot = zero(T)
	while (ep <= max_episodes) && (step <= max_steps)
		update_feature_vector!(feature_vector, s)
		v = linear_value_function(feature_vector, parameters)
		#represents the portion of the eligibility trace update that depends on the current feature vector
		dt =  one(T) - α*γ*λ*linear_value_function(feature_vector, z)
		z .*= γ*λ
		update_params_with_gradient!(z, dt, feature_vector)
		update_params_with_gradient!(parameters, -α*(v - v_old), feature_vector)
		
		target, i_a_max = update_action_values!(action_values, s, feature_vector, update_feature_vector!, linear_value_function, parameters, mdp, γ, action_value_args...)
		policy .= action_values
		make_ϵ_greedy_policy!(policy; ϵ = ϵ)
		i_a = sample_action(policy)
		
		δ = target - v
		
		#take action and observe transition
		(r, s′) = mdp.ptf(s, i_a)
		rtot += r

		terminated = mdp.isterm(s′)

		if terminated
			v′ = zero(T)
		else
			update_feature_vector!(feature_vector, s′)
			v′ = linear_value_function(feature_vector, parameters)
		end

		parameters .+= α*(δ + v - v_old) .* z

		if terminated
			s′ = mdp.initialize_state()
			#reset eligibility vector to 0 at the start of a new episode
			z .= zero(T)
			ep += 1
			push!(episode_rewards, rtot)
			push!(episode_steps, step)
			rtot = zero(T)
		end
			
		s = s′
		v_old = v′
		
		step += 1
	end
	
	q̂, form_kwargs = form_value_function(mdp, γ, update_feature_vector!, linear_value_function, feature_vector, parameters)

	return (value_function = q̂, episode_rewards = episode_rewards, episode_steps = episode_steps, parameter_history = parameter_history, final_parameters = deepcopy(parameters), form_kwargs = form_kwargs)
end

# ╔═╡ b1ae33a5-272f-43de-b9e2-eaa0423c34b8
#true online dp λ for linear features
function true_online_dp_λ!(parameters::Vector{T}, mdp::StateMDP, λ::T, num_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; α = one(T)/10, α_r̄::T = one(T)/10, ϵ = one(T) / 10, z::Vector{T} = copy(parameters), action_values::Matrix{T} = zeros(T, length(mdp.actions), 1), kwargs...) where {T<:Real}
	policy = zeros(T, length(mdp.actions))
	action_value_args = form_action_value_args(mdp, feature_vector, parameters)
	
	#initialize records
	reward_history = zeros(T, num_steps)
	average_reward_history = zeros(T, num_steps)
	
	#initialize episode
	s = mdp.initialize_state()
	target, i_a_max = update_differential_action_values!(action_values, s, feature_vector, update_feature_vector!, linear_value_function, parameters, mdp, r̄, action_value_args...)
	policy .= action_values
	make_ϵ_greedy_policy!(policy; ϵ = ϵ)
	i_a = sample_action(policy)
	z .= zero(T)
	v_old = zero(T)

	
	for step in 1:num_steps
		update_feature_vector!(feature_vector, s)
		v = linear_value_function(feature_vector, parameters)
		#represents the portion of the eligibility trace update that depends on the current feature vector
		dt =  one(T) - α*λ*linear_value_function(feature_vector, z)
		z .*= λ
		update_params_with_gradient!(z, dt, feature_vector)
		update_params_with_gradient!(parameters, -α*(v - v_old), feature_vector)
		average_reward_history[step] = r̄
		target, i_a_max = update_differential_action_values!(action_values, s, feature_vector, update_feature_vector!, linear_value_function, parameters, mdp, r̄, action_value_args...)
		policy .= action_values
		make_ϵ_greedy_policy!(policy; ϵ = ϵ)
		i_a = sample_action(policy)
		
		δ = target - v

		r̄ += α_r̄*δ
		
		#take action and observe transition
		(r, s′) = mdp.ptf(s, i_a)
		reward_history[step] = r

		mdp.isterm(s′) && error("$s′ is a terminal state and this method only applies to continuing tasks")

		
		update_feature_vector!(feature_vector, s′)
		v′ = linear_value_function(feature_vector, parameters)

		parameters .+= α*(δ + v - v_old) .* z
			
		s = s′
		v_old = v′
	end
	
	q̂, form_kwargs = form_differential_value_function(mdp, r̄, update_feature_vector!, linear_value_function, feature_vector, parameters)

	return (value_function = q̂, episode_rewards = episode_rewards, episode_steps = episode_steps, parameter_history = parameter_history, final_parameters = deepcopy(parameters), form_kwargs = form_kwargs)
end

# ╔═╡ a2a6c291-3ea7-45d4-9608-40e25f1cfe7c
true_online_dp_λ(mdp::StateMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer, feature_vector::LinearFeatureVector, update_feature_vector!::Function; init_value::T = zero(T), parameters = initialize_linear_parameters(feature_vector, init_value), kwargs...) where T<:Real = true_online_dp_λ!(parameters, mdp, γ, λ, max_episodes, max_steps, feature_vector, update_feature_vector!; kwargs...)

# ╔═╡ 8b6b5084-3972-4bd4-9ca2-423f1c627788
md"""
### *Example: Mountain Car Sarsa(λ) Variations*
"""

# ╔═╡ 5bc128ec-2934-4aa5-a922-9017f647e1b3
md"""
#### Sarsa(λ) Parameter Studies With Mountain Car Tile Coding
"""

# ╔═╡ 5ad7c72e-b276-4dd8-a6de-df4e2e01f048
setup_mountaincar_tiles(num_tiles::Integer, num_tilings::Integer) = tile_coding_feature_setup(MountainCarTask.mdp, (-1.2f0, -0.07f0), (0.5f0, 0.07f0), (1f0/num_tiles, 1f0/num_tiles), num_tilings)

# ╔═╡ c35b4242-8477-468f-bd86-32cda00229a4
function run_mountaincar_λ_linear(α, λ, algo; num_steps = 50_000, num_tiles = 10, num_tilings = 10, kwargs...)
	tile_coding = setup_mountaincar_tiles(num_tiles, num_tilings)
	algo(MountainCarTask.deterministic_mdp, 1f0, λ, typemax(Int64), num_steps, tile_coding.feature_vector, tile_coding.update_feature_vector!; α = α, kwargs...)
end

# ╔═╡ cd0b96eb-150c-4441-ad79-8c0305213cbd
function run_mountaincar_λ_linear_trial(α, λ, algo; kwargs...)
	output = run_mountaincar_λ_linear(α, λ, algo; kwargs...)
	step_history = output.episode_steps
	isempty(step_history) && return NaN
	l = length(step_history)
	return step_history[end] / l
end

# ╔═╡ 4bd74da4-382b-47af-bfca-78b3318e2df7
# ╠═╡ skip_as_script = true
#=╠═╡
const mountaincar_semi_gradient_linear_parameter_study = setup_parameter_study(run_mountaincar_λ_linear_trial, (:α, :λ, :algo), (num_steps = 50_000, num_tiles = 10, num_tilings = 10, compute_value = compute_sarsa_value, ϵ = 0.01f0, trace_type = AccumulatingTrace()))
  ╠═╡ =#

# ╔═╡ 0324b4e2-2544-4bd6-b310-8a330b5a92c5
#=╠═╡
function display_mountaincar_λ_parameter_study(α_list, λ_list, study::NamedTuple; num_trials = Base.Threads.nthreads(), algo = sarsa_λ_linear, ymin = 100, ymax = 400, num_steps = 50_000, kwargs...)
	traces = [begin
		y = [begin
			study.update_results!(α, λ, algo; num_trials = num_trials, num_steps = num_steps, kwargs...)
		end
		for α in α_list]
		scatter(x = α_list, y = y, name = "λ = $λ")
	end
	for λ in λ_list]
	plot(traces, Layout(xaxis_title = "Learning Rate", yaxis_title = "Average Steps Per Episode Averaged <br> Over the First $num_steps Steps and $num_trials Runs", yaxis_range = [ymin, ymax], xaxis_type = "log"))
end
  ╠═╡ =#

# ╔═╡ c19209dc-bddf-4390-95a9-fc1d1d836a8a
md"""
##### Sarsa$$(λ)$$ with $$\epsilon = 0.01$$
"""

# ╔═╡ 5652f3fd-ec23-4dfb-a171-1e1ed0de275a
#=╠═╡
@bind run_mountaincar_λ_study1 CounterButton("Run Parameter Study (could take several minutes)")
  ╠═╡ =#

# ╔═╡ a29d560d-fe30-4efc-8bce-90d652c738fa
#=╠═╡
@bind sarsa_λ_mountaincar_study_params PlutoUI.combine() do Child
	md"""
	Target Value: $(Child(:compute_value, Select([compute_expected_sarsa_value => "Expected Sarsa", compute_sarsa_value => "Sarsa", compute_q_learning_value => "Q Learning"])))
	Number of Tiles Per Dimension: $(Child(:num_tiles, NumberField(2:20, default = 4)))
	Number of Tilings: $(Child(:num_tilings, NumberField(2:20, default = 4)))
	Trace Type: $(Child(:trace_type, Select([AccumulatingTrace() => "Accumulating Trace", ReplacingTrace() => "Replacing Trace", DutchTrace() => "Dutch Trace"])))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ 111cda26-bd25-49ed-9ba7-4ee8f71b063f
#=╠═╡
if run_mountaincar_λ_study1 > 0
	display_mountaincar_λ_parameter_study(Base.LogRange(0.002f0, 0.4f0, 8), [0f0, 0.5f0, 0.8f0, 0.9f0, 0.95f0, 0.99f0], mountaincar_semi_gradient_linear_parameter_study; ymin = 100, ymax = 400, sarsa_λ_mountaincar_study_params...)
else
	md"""Waiting to run parameter study"""
end
  ╠═╡ =#

# ╔═╡ aea15e6d-9873-406b-993b-04717dad01c6
md"""
##### DP$$(λ)$$ with $$\epsilon = 0.01$$

In this method the full transition distribution is used and only state values are estimated.
"""

# ╔═╡ c57b4792-928a-4450-9364-786e9f186cc8
#=╠═╡
@bind run_mountaincar_λ_study3 CounterButton("Run Parameter Study (could take several minutes)")
  ╠═╡ =#

# ╔═╡ 118545ac-5ee9-4b61-b384-52dfb41a533b
#=╠═╡
@bind dp_λ_mountaincar_study_params PlutoUI.combine() do Child
	md"""
	Number of Tiles Per Dimension: $(Child(:num_tiles, NumberField(2:20, default = 14)))
	Number of Tilings: $(Child(:num_tilings, NumberField(2:20, default = 10)))
	Trace Type: $(Child(:trace_type, Select([AccumulatingTrace() => "Accumulating Trace", ReplacingTrace() => "Replacing Trace", DutchTrace() => "Dutch Trace"])))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ f1a8df55-a5ef-475e-a0c4-ed31b1c6c9f5
#=╠═╡
if run_mountaincar_λ_study3 > 0
	display_mountaincar_λ_parameter_study(Base.LogRange(0.002f0, 0.4f0, 8), [0f0, 0.5f0, 0.8f0, 0.9f0, 0.95f0, 0.99f0], mountaincar_semi_gradient_linear_parameter_study; ymin = 100, ymax = 400, algo = dp_λ_linear, dp_λ_mountaincar_study_params...)
else
	md"""Waiting to run parameter study"""
end
  ╠═╡ =#

# ╔═╡ b28f47cc-eda7-4961-b6b3-569753386249
md"""
##### True Online Sarsa$$(λ)$$ with $$ϵ = 0.01$$

Notice that here a slightly lower value of $\lambda$ is optimal which increases the degree of bootstrapping compared to Sarsa$(\lambda)$
"""

# ╔═╡ 31633123-0249-4d15-b6fe-59480d3038eb
#=╠═╡
@bind run_mountaincar_λ_study4 CounterButton("Run Parameter Study (could take several minutes)")
  ╠═╡ =#

# ╔═╡ 4b9f3ab7-bb96-46f4-8af5-145f92f82d93
#=╠═╡
@bind true_online_sarsa_λ_mountaincar_study_params PlutoUI.combine() do Child
	md"""
	Target Value: $(Child(:compute_value, Select([compute_expected_sarsa_value => "Expected Sarsa", compute_sarsa_value => "Sarsa", compute_q_learning_value => "Q Learning"])))
	Number of Tiles Per Dimension: $(Child(:num_tiles, NumberField(2:20, default = 4)))
	Number of Tilings: $(Child(:num_tilings, NumberField(2:20, default = 4)))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ 6b449c6c-249e-4193-96ea-caccee683de0
#=╠═╡
if run_mountaincar_λ_study4 > 0
	display_mountaincar_λ_parameter_study(Base.LogRange(0.002f0, 0.4f0, 8), [0f0, 0.5f0, 0.8f0, 0.9f0, 0.95f0, 0.99f0], mountaincar_semi_gradient_linear_parameter_study; ymin = 100, ymax = 400, algo = true_online_sarsa_λ, true_online_sarsa_λ_mountaincar_study_params...)
else
	md"""Waiting to run parameter study"""
end
  ╠═╡ =#

# ╔═╡ 438726e5-f9a1-4bf7-abda-e5bb0eb30c39
md"""
##### True Online DP$$(λ)$$ with $$ϵ = 0.01$$

Bests results so far which also favor a higher value of $\lambda$ which indicates less reliance on bootstrapping.
"""

# ╔═╡ 4d00dfcc-7b01-4335-95ba-0b31fa0e62ad
#=╠═╡
@bind run_mountaincar_λ_study6 CounterButton("Run Parameter Study (could take several minutes)")
  ╠═╡ =#

# ╔═╡ f475a176-c2b0-4a22-8975-f5d2f54b6530
#=╠═╡
@bind true_online_dp_λ_mountaincar_study_params PlutoUI.combine() do Child
	md"""
	Number of Tiles Per Dimension: $(Child(:num_tiles, NumberField(2:20, default = 14)))
	Number of Tilings: $(Child(:num_tilings, NumberField(2:20, default = 10)))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ cc14f0a2-d0bc-40fa-83fa-b99e62351282
#=╠═╡
if run_mountaincar_λ_study6 > 0
	display_mountaincar_λ_parameter_study(Base.LogRange(0.001f0, 0.04f0, 8), [0f0, 0.5f0, 0.8f0, 0.9f0, 0.95f0, 0.99f0], mountaincar_semi_gradient_linear_parameter_study; ymin = 100, ymax = 400, algo = true_online_dp_λ, true_online_dp_λ_mountaincar_study_params...)
else
	md"""Waiting to run parameter study"""
end
  ╠═╡ =#

# ╔═╡ 0a5bec4a-0e65-4753-a1e8-f7b3c6a061df
md"""
##### Results Visualization for Best Training Parameters
"""

# ╔═╡ 2e6b7c33-c6b2-4fa0-9d71-acf7cb818b9b
#=╠═╡
function smooth_error(error_history, n)
	l = length(error_history)
	[mean(error_history[max(1, i-n):i]) for i in n+1:l]
end
  ╠═╡ =#

# ╔═╡ 315db1e4-c730-46bc-8f5b-03cdfe5467f9
# ╠═╡ skip_as_script = true
#=╠═╡
const mountaincar_tilecoding_λ_best = run_mountaincar_λ_linear(0.002f0, 0.99f0, true_online_dp_λ; num_steps = 50_000, num_tiles = 14, num_tilings = 10, ϵ = 0.01f0)
  ╠═╡ =#

# ╔═╡ 20836990-b332-478a-b99b-6f4ef4659392
md"""
#### Sarsa λ Parameter Studies with Non-linear Approximation
"""

# ╔═╡ a1c173d7-f5ac-4eb3-96c7-c2f020cdf3d5
md"""
##### Sarsa$$(λ)$$ with $$\epsilon = 0.01$$
"""

# ╔═╡ 03f0034a-9acf-4745-a927-51fc2554a7e4
#=╠═╡
@bind run_mountaincar_λ_fcann_study1 CounterButton("Run Parameter Study (could take several minutes)")
  ╠═╡ =#

# ╔═╡ 10c75d09-3966-4511-83f8-e8365a2a82da
md"""
##### DP$$(λ)$$ with $$\epsilon = 0.01$$
"""

# ╔═╡ 3d6df2fa-35e5-4a02-b53f-30936f404b3f
#=╠═╡
@bind run_mountaincar_λ_fcann_study3 CounterButton("Run Parameter Study (could take several minutes)")
  ╠═╡ =#

# ╔═╡ fbe8691b-6d71-4cba-90e4-5de63421f634
md"""
> ### *Exercise 12.6* 
> Modify the pseudocode for Sarsa(λ) to use dutch traces (12.11) without the other distinctive features of a true online algorithm.  Assume linear function approximation and binary features.

See the above function `sarsa_λ`.  In the step where $z_i$ is updated an additional term is subtracted in the case of using dutch traces which matches equation (12.11)
"""

# ╔═╡ 5062690c-96b9-450a-9927-6a6707dfc511
md"""
### Method Variation Summary

#### Base Algorithms

##### Sarsa$(\lambda)$

- For episodic tasks, performs the same value updates as offline TD$(\lambda)$
- Applies to any type of function approximation to update eligibility traces

##### True Online Sarsa $(\lambda)$

- Uses true online TD$(\lambda)$ value updates which equal the truncated n-step $\lambda$ return up to the maximum available horizon for each step
- Applies only to linear function approximation

#### Target Value Options

##### Sample Transition State
- Uses the action selected at the transition state to estimate the future q-value
-  $\delta = R + \gamma \hat q(s^\prime, a^\prime) - \hat q(s, a)$
- Default choice for both base algorithms

##### Expected Value of Transition State
- Uses the probability distribution over actions at the transition state: $\pi(s^\prime)$
-  $\delta = R + \gamma \sum_{a^\prime} \pi(a^\prime \vert s^\prime) \hat q(s^\prime, a^\prime) - \hat q(s, a)$
- When used results in "expected" versions of both algorithms which can reduce variance.

#### Dynamic Programming Alternative
- Requires knowledge of the probability distribution over transitions: $p(s^\prime, r \vert s, a)$
- Uses $\hat v(s)$ instead of $\hat q(s, a)$ and derives state-action values for action selection from the transition function above
- There is no expected value version of this because the transition value being used of $\hat v(s^\prime)$ already does not depend on the future action selection.
"""

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

# ╔═╡ ba274806-6e16-447d-8c70-259787941495
md"""
## 12.9 Off-policy Traces with Control Variates
"""

# ╔═╡ bc0073ab-fc41-4333-aecc-41501d89f15b
md"""
$\mathbf{z}_t \doteq \rho_t \left ( \gamma_t \lambda_t \mathbf{z}_{t-1} + \nabla\hat v (S_t, \mathbf{w}_t)\right ) \tag{12.25}$
$\delta_t^a = R_{t+1} + \gamma_{t+1} \bar V_t(S_{t+1}) - \hat q(S_t, A_t, \mathbf{w}_t) \tag{12.28}$
$\mathbf{z}_t \doteq \gamma_t \lambda_t \rho_t \mathbf{z}_{t-1} + \nabla\hat q (S_t, A_t, \mathbf{w}_t) \tag{12.29}$
"""

# ╔═╡ 1f3de2ad-65c5-4aaf-9c12-623de2257619


# ╔═╡ b6123560-90fd-4cd5-83ff-f73234d8a897
md"""
## 12.10 Watkins's $Q(\lambda)$ to Tree-Backup$(\lambda)$

This eligibility trace update along with the usual semi-gradient parameter-update rule defines the TB $(\lambda)$ algorithm.  It is not guaranteed to be stable when used with off-policy data and function approximation.  For that we need the techniques in the next section. 

$\mathbf{z}_t \doteq \gamma_t \lambda_t \pi(A_t \vert S_t)\mathbf{z}_{t-1} + \nabla \hat q (S_t, A_t, \mathbf{w}_t)$
"""

# ╔═╡ bcd35714-d664-4347-af27-4bdf131bad89


# ╔═╡ 4a474bb7-c932-4cbb-8442-2c0972a7da6c
md"""
## 12.11 Stable Off-policy Methods with Traces
"""

# ╔═╡ cacab854-8b62-4e45-bc88-85038461e667
md"""
$GTD(\lambda)$ is the eligibility-trace algorithm analogous to TDC discussed in Chapter 11 which is stable under off-policy learning.  Its goal is to learn a parameter $\mathbf{w}_t$ such that $\hat v(s, \mathbf{w}) \doteq \mathbf{w}_t ^\top \mathbf{x}(s) \approx v_\pi (s)$, even from data that is due to another policy $b$.  Its update is

$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t^s \mathbf{z}_t - \alpha \gamma_{t+1} (1- \lambda_{t+1}) (\mathbf{z}_t ^\top \mathbf{v}_t) \mathbf{x}_{t+1}$

and

$\mathbf{v}_{t+1} \doteq \mathbf{v}_t + \beta \delta_t^s \mathbf{z}_t - \beta (\mathbf{v}_t^\top \mathbf{x}_t) \mathbf{x}_t \tag{12.30}$

where, as in Section 11.7, $\mathbf{v} \in \mathbb{R}^d$ is a vector of the same dimension as $\mathbf{w}$, initialized to $\mathbf{v}_0 = \mathbf{0}$, and $\beta > 0$ is a second step-size parameter.

 $GQ(\lambda)$ is the Gradient-TD algorithm for action values with eligibility traces.  Its goal is to learn a parameter $\mathbf{w}_t$ such that $\hat q (s, a, \mathbf{w}_t) \doteq \mathbf{w}_t^\top \mathbf{x}(s, a) \approx q_\pi (s, a)$ from off-policy data.  If the target policy is $\epsilon$-greedy, or otherwise biased toward the greedy policy for $\hat q$, then GQ$(\lambda)$ can be used as a control algorithm.  Its update is

$\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \delta_t^s \mathbf{z}_t - \alpha \gamma_{t+1} (1- \lambda_{t+1}) (\mathbf{z}_t ^\top \mathbf{v}_t) \overline{\mathbf{x}}_{t+1}$

where $\overline{\mathbf{x}_t}$ is the average eature vector for $S_t$ under the target policy,

$\overline{\mathbf{x}}_t \doteq \sum_a \pi(a \mid S_t) \mathbf{x}(S_t, a)$,

 $\delta_t^a$ is the expectation form of the TD error, which can be written 

$\delta_t^a \doteq R_{t+1} + \gamma_{t+1} \mathbf{w}_t^\top \overline{\mathbf{x}}_{t+1} - \mathbf{w}_t^\top \mathbf{x}_t$,

 $\mathbf{z}_t$ is defined in the usual way for action values (12.29), and the rest is as in GTD$(\lambda)$, including hte update for $\mathbf{v}_t$ (12.30).

 $HTD(\lambda)$ is a hygrid state-value algorithm combining aspects of GTD$(\lambda)$ and TD$(\lambda)$.  Its most appealing feature is that it is a strict generalization of TD$(\lambda)$ to off-policy learning, meaning that if the behavior policy happens to be the same as the target policy, then HTD$(\lambda)$ becomes the same as TD$(\lambda)$, which is not true for GTD$(\lambda)$.  This is appealing because TD$(\lambda)$ is often faster than GTD$(\lambda)$ when both algorithms converge, and TD$(\lambda)$ requires setting only a single step size.  HTD$(\lambda)$ is defined by:

$\begin{flalign}
\mathbf{w}_{t+1} &\doteq \mathbf{w}_t \dots \\
\mathbf{v}_{t+1} &\doteq \mathbf{v}_t \dots \\
\mathbf{z}_t &\doteq \rho_t \dots \\
\mathbf{z}_t^b &\doteq \gamma_t \lambda_t \dots
\end{flalign}$

where $\beta > 0$ again is a second step-size parameter.  In addition to the second set of weights, $\mathbf{v}_t$, HTD$(\lambda)$ also has a second set of eligibility traces, $\mathbf{z}_t^b$.  These are conventional accumulating eligibility traces for the behavior policy and become equal to $\mathbf{z}_t$ if all the $rho_t$ are 1, which causes the last term in the $\mathbf{w}_t$ update to be zero and the overall update to reduce to TD$(\lambda)$.
"""

# ╔═╡ 44100481-4e66-4b38-8262-87e337148bfc
md"""
### *HTD$(\lambda)$ Implementation*
"""

# ╔═╡ b525e0c8-e673-448d-8143-2a9a8be342f5
begin
	#htdλ for binary features, WORK IN PROGRESS
	function htd_λ!(parameters::Matrix{T}, get_active_features::Function, mdp::StateMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; α = one(T)/10, ϵ = one(T) / 10, z_target::Matrix{T} = copy(parameters), z_behavior::Matrix{T} = copy(parameters), v::Matrix{T} = copy(parameters), action_values::Vector{T} = zeros(T, length(mdp.actions)), save_episode_steps::Bool = false, save_step_rewards::Bool = false, use_accumulating_traces::Bool = false, use_dutch_traces::Bool = false, epkwargs...) where {T<:Real}
		#initialize records
		episode_step_history = Vector{T}()
		step_rewards = Vector{T}()
	
		#initialize episode
		s = mdp.initialize_state()
		active_features = get_active_features(s)
		update_action_values!(action_values, parameters, active_features)
		make_ϵ_greedy_policy!(action_values; ϵ = ϵ)
		i_a = sample_action(action_values)
		p_b = action_values[i_a]
		make_greedy_policy!(action_values)
		p_t = action_values[i_a]
		ρ = p_t / p_b
		z_target .= zero(T)
		z_behavior .= zero(T)
		q_old = zero(T)
		ep = 1
		step = 1

		#uses the active features to compute the effective dot product of the feature vector of a state with the parameter or eligibility trace values for the specified action
		function get_feature_values(m::Matrix{T}, i_a::Integer, active_features)
			x = zero(T)
			for i in active_features
				x += m[i, i_a]
			end
			return x
		end
		
		while (ep <= max_episodes) && (step <= max_steps)
			q = get_feature_values(parameters, i_a, active_features)

			c = dot(z_target - z_behavior, v)
			
			#this portion of the parameter update only depends on the current feature vector, state-action value and old state-action value
			for i in active_features
				parameters[i, i_a] += α * c
			end

			z .*= γ*λ

			
			
			#take action and observe transition
			(r, s′) = mdp.ptf(s, i_a)
			
			save_step_rewards && push!(step_rewards, r)

			if mdp.isterm(s′)
				q′ = zero(T)
				active_features = []
			else
				active_features = get_active_features(s′)
				update_action_values!(action_values, parameters, active_features)
				make_ϵ_greedy_policy!(action_values; ϵ = ϵ)
				i_a′ = sample_action(action_values)
				q′ = get_feature_values(parameters, i_a′, active_features)
			end
			
			δ = r + γ*q′ - q

			for i in active_features
				parameters[i, i_a′] -= γ*c
			end
			
			parameters .+= α*δ .* z_target

			v .+= β*δ .* z_target .- β*dot(z_behavior, v)
			
			if mdp.isterm(s′)
				s = mdp.initialize_state()
				active_features = get_active_features(s)
				update_action_values!(action_values, parameters, active_features)
				make_ϵ_greedy_policy!(action_values; ϵ = ϵ)
				i_a = sample_action(action_values)
				#reset eligibility vector to 0 at the start of a new episode
				z .= zero(T)
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

	#non-tabular problem with linear function approximation. 
	# function true_online_sarsa_λ!(parameters::Vector{Vector{T}}, feature_vector::Vector{T}, update_feature_vector!::Function, mdp::StateMDP, γ::T, λ::T, max_episodes::Integer, max_steps::Integer; α = one(T)/10, ϵ = one(T) / 10, x2::Vector{T} = copy(feature_vector), z::Vector{Vector{T}} = copy(parameters), action_values::Vector{T} = zeros(T, length(mdp.actions)), save_episode_steps::Bool = false, save_step_rewards::Bool = false, use_accumulating_traces::Bool = false, use_dutch_traces::Bool = false, epkwargs...) where {T<:Real}
	# 	#initialize records
	# 	episode_step_history = Vector{T}()
	# 	step_rewards = Vector{T}()

	# 	function select_action!(action_values, parameters, ϵ, x)
	# 		for i_a in eachindex(action_values)
	# 			action_values[i_a] = dot(x, parameters[i_a])
	# 		end
	# 		make_ϵ_greedy_policy!(action_values; ϵ = ϵ)
	# 		sample_action(action_values)
	# 	end
	
	# 	#initialize episode
	# 	s = mdp.initialize_state()
	# 	update_feature_vector!(feature_vector, s)
	# 	i_a = select_action!(action_values, parameters, ϵ, feature_vector)
	# 	z .= zero(T)
	# 	q_old = zero(T)
	# 	ep = 1
	# 	step = 1
		
	# 	while (ep <= max_episodes) && (step <= max_steps)
	# 		q = dot(feature_vector, parameters[i_a])
	# 		dt = dot(z[i_a], feature_vector)
	# 		x2 .= one(T) .+ α*γ*λ*dt .* feature_vector

	# 		parameters[i_a] .-= α*(q - q_old) .* feature_vector
			
	# 		#take action and observe transition
	# 		(r, s′) = mdp.ptf(s, i_a)
			
	# 		save_step_rewards && push!(step_rewards, r)

			
	# 		if mdp.isterm(s′)
	# 			q′ = zero(T)
	# 		else
	# 			update_feature_vector!(feature_vector, s′)
	# 			i_a′ = select_action!(action_values, parameters, ϵ, feature_vector)
	# 			q′ = dot(feature_vector, parameters[i_a′])
	# 		end
			
	# 		δ = r + γ*q′ - q
			
	# 		for i_a in eachindex(mdp.actions)
	# 			z .*= γ*λ
	# 		end
	# 		z[i_a] .+ x2

	# 		for i_a in eachindex(mdp.actions)
	# 			parameters[i_a] .+= α*(δ + q - q_old) .* z[i_a]
	# 		end
			
	# 		if mdp.isterm(s′)
	# 			s = mdp.initialize_state()
	# 			update_feature_vector!(feature_vector, s)
	# 			i_a = select_action!(action_values, parameters, ϵ, feature_vector)
	# 			#reset eligibility vector to 0 at the start of a new episode
	# 			for i_a in eachindex(mdp.actions) z .= zero(T) end
	# 			q_old = zero(T)
	# 			ep += 1
	# 			save_episode_steps && push!(episode_step_history, step)
	# 		else
	# 			s = s′
	# 			i_a = i_a′
	# 			q_old = q′
	# 		end
	# 		step += 1
	# 	end
		
	# 	return (episode_steps = episode_step_history, step_rewards = step_rewards)
	# end
end

# ╔═╡ 9c9c5f0a-4079-4848-a822-ea9dcc460660
md"""
## Cart Pole Simulation Environment
"""

# ╔═╡ 560fa6a4-ac3a-43ae-931e-6699294b304a
md"""
### Data Structures
"""

# ╔═╡ 33a36f03-959d-4921-a476-68a75234f47c
md"""
The problem is defined by the properties of the vehicle which includes the mass of the cart and the point mass balanced atop it.  This struct contains all the relevant properties to determinte the physics of the problem.
"""

# ╔═╡ 6630a9a0-2ec9-4c18-b9eb-e263ddc5d18c
struct CartPoleVehicle{T <: Real}
	m::T 	#point mass
	m_c::T 	#cart mass
	l::T 	#length of pendulum
	k::T 	#inertia constant
	m_f::T  #moment of friction between cart and pole
	μ_c::T  #coefficient of friction between cart and track
end

# ╔═╡ 7cdc5c62-ddae-41fe-9ea2-aba25ac0ac3f
md"""
To simulate the movement of the cart, each time step requires knowledge of the position and velocity of the horizontal position of the cart as well as the angle of the pole.  The angle is defined to be 0 when the pole is vertical.
"""

# ╔═╡ 1fa542f3-1e0e-41fc-ab09-e7eb0bd22483
begin
	CartPoleState{T} = @NamedTuple{x::T, θ::T, ẋ::T, θ̇::T, t::T}
	# struct CartPoleState{T <: Real}
	# 	x::T 	#horizontal position on track
	# 	θ::T 	#Angle of pendulum in radians measured as deviation from the vertical, 90° is horizontal and to the right
	# 	ẋ::T 	#horizontal velocity on track
	# 	θ̇::T 	#Range of change of pendulum angle
	# 	t::T 	#Time in seconds
	# end
	function CartPoleState(x::A, θ::B, ẋ::C, θ̇::D) where {A<:Real, B<:Real, C<:Real, D<:Real}
		T = promote_type(A, B, C, D)
		(x = T(x), θ = T(θ), ẋ = T(ẋ), θ̇ = T(θ̇), t = zero(T))
	end

	function CartPoleState(x::A, θ::B, ẋ::C, θ̇::D, t::E) where {A<:Real, B<:Real, C<:Real, D<:Real, E<:Real}
		T = promote_type(A, B, C, D, E)
		(x = T(x), θ = T(θ), ẋ = T(ẋ), θ̇ = T(θ̇), t = T(t))
	end

	CartPoleState() = CartPoleState(0f0, 0f0, 0f0, 0f0)
end

# ╔═╡ 88e61bb4-fd6e-4363-be94-4166a7a39983
md"""
### Physics Simulation

To simulate the vehicle, we will choose a step size and calculate the state of the vehicle one step forward in time based on the initial positions and velocities.  The only external force on the vehicle will be a horizontal force representing force applied by some motor on the wheels to move the cart forward or backwards.  The following functions calculate the acceleration for both the horizontal position and the angular position of the pole.
"""

# ╔═╡ 1442dda6-f5a9-4a23-9075-39a9c7fcb899
md"""
#### Accelerations
"""

# ╔═╡ a566cd6b-19b6-4cfb-80e3-c74ed58705ba
cartpole_ẍ(m, m_c, l, g, θ, f, μ_c, m_f, k, ẋ, θ̇) = (m*g*sin(θ)*cos(θ) - (1+k)*(f+ m*l*θ̇^2 * sin(θ) - μ_c*ẋ) - m_f *cos(θ)/l) / (m*cos(θ)^2 - (1+k)*(m + m_c))

# ╔═╡ 7f83988f-6e2c-4d90-899f-b4f5cdb1de48
cartpole_θ̈(m, l, k, m_f, g, θ, ẍ) = (g*sin(θ) - ẍ*cos(θ) - m_f / (m*l)) / ((1+k)*l)

# ╔═╡ 006fc67d-c3f9-46d3-b631-3002d9e50dd6
cartpole_ẍ(cart::CartPoleVehicle{T}, state::CartPoleState{T}, g::T, f::T) where T<:Real = cartpole_ẍ(cart.m, cart.m_c, cart.l, g, state.θ, f, cart.μ_c, cart.m_f, cart.k, state.ẋ, state.θ̇)

# ╔═╡ f4e54d48-e2b6-45e7-b672-279cc3b2a3f0
cartpole_θ̈(cart::CartPoleVehicle{T}, state::CartPoleState{T}, g::T, ẍ::T) where T<:Real = cartpole_θ̈(cart.m, cart.l, cart.k, cart.m_f, g, state.θ, ẍ)

# ╔═╡ fd9c8373-90f9-4c1a-8c85-5e280311d381
md"""
#### Numerical Integration Step

Using the acceration functions, we can perform a multi-part integration step method known as the Runge Kutta method.  Unlike a simple Euler step, this approach calculates the accelerations at the halfway point of a step as well as the endpoints and uses all of the values together to reduce the error from a finite step size.  Using this method should enable more stable results even at larger step sizes.
"""

# ╔═╡ e820833d-db94-4a74-a637-5c3356b07906
function cartpole_runge_kutta_step(cart::CartPoleVehicle{T}, state::CartPoleState{T}, g::T, f::T, h::T) where T<:Real
	# acceleration of x and θ at the beginning of the interval
	k1_ẍ = cartpole_ẍ(cart, state, g, f)
	k1_θ̈ = cartpole_θ̈(cart, state, g, k1_ẍ)

	#acceleration of x and θ at the midpoint of the interval using the initial acceleration
	midpoint_state1 = CartPoleState(state.x + state.ẋ*h/2, state.θ + state.θ̇*h/2, state.ẋ + k1_ẍ*h/2, state.θ̇ + k1_θ̈*h/2)
	k2_ẍ = cartpole_ẍ(cart, midpoint_state1, g, f)
	k2_θ̈ = cartpole_θ̈(cart, midpoint_state1, g, k2_ẍ)

	#acceleration of x and θ at midpoint using k2
	midpoint_state2 = CartPoleState(state.x + midpoint_state1.ẋ*h/2, state.θ + midpoint_state1.θ̇*h/2, state.ẋ + k2_ẍ*h/2, state.θ̇ + k2_θ̈*h/2)
	k3_ẍ = cartpole_ẍ(cart, midpoint_state2, g, f)
	k3_θ̈ = cartpole_θ̈(cart, midpoint_state2, g, k3_ẍ)

	#acceleration of x and θ at end of interval using k3
	endpoint_state = CartPoleState(state.x + midpoint_state2.ẋ*h, state.θ + midpoint_state2.θ̇*h, state.ẋ + k3_ẍ*h, state.θ̇ + k3_θ̈*h)
	k4_ẍ = cartpole_ẍ(cart, endpoint_state, g, f)
	k4_θ̈ = cartpole_θ̈(cart, endpoint_state, g, k4_ẍ)

	#final state estimation
	x′ = state.x + (h/6) * (state.ẋ + 2*midpoint_state1.ẋ + 2*midpoint_state2.ẋ + endpoint_state.ẋ)
	θ′ = state.θ + (h/6) * (state.θ̇ + 2*midpoint_state1.θ̇ + 2*midpoint_state2.θ̇ + endpoint_state.θ̇)
	ẋ′ = state.ẋ + (h/6) * (k1_ẍ + 2*k2_ẍ + 2*k3_ẍ + k4_ẍ)
	θ̇′ = state.θ̇ + (h/6) * (k1_θ̈ + 2*k2_θ̈ + 2*k3_θ̈ + k4_θ̈)
	CartPoleState(x′, θ′, ẋ′, θ̇′, state.t + h)
end

# ╔═╡ d746e585-a734-4fea-a534-ab366c12a87f
md"""
### MDP Creation

In order to turn this into an MDP environment, we need to define a few other constraints on the problem including the reward function and whether or not the problem is episodic.  The goal of the environment is to keep the pole balanced vertically for as long as possible.  The equilibrium point at the top is unstable to any small perturbation will require intervention in order to prevent it from toppling.  Given this goal, one natural choice is to create an episodic task with a reward of +1 for every step.  Episodes will terminate when the angle of the pole exceeds a certain value such as horizontal.  The total reward at the end of an episode will be the number of steps survived, and the training process will incentivize an agent to balance the pole for as long as possible.  The following function will create an MDP based on the cart pole physics simulation but with the additional constraints described here to make it an MDP task.
"""

# ╔═╡ 2776aeba-4d0f-49c9-8395-d0f7242f2429
#create a cart pole MDP environment
function create_cartpole_mdp(;
	m::T = 1f0, 		#mass at the end of the pole in kg
	m_c::T = 10f0,  	#mass of the cart in kg
	l::T = 1f0, 		#length of the pole in meters
	g::T = 9.8f0, 		#gravitational constant in meters per second squared
	h::T = 1f-3, 		#step size parameter of simulation in seconds
	k::T = 1f0, 		#inertial constant of pendulum,
	m_f::T = 0f0, 		#friction of the rotating pole
	μ_c::T = 0f0, 		#friction of the cart wheels against the track
	f::T = 100f0, 		#force applied by throttle
	x_max::T = Inf32,  	#maximum horizontal position
    θ_max::T = π/2f0,   #maximum pole angle
	init_x::Function = () -> 0f0,  #initialize each of the 4 state variables
	init_θ::Function = () -> 0f0,
	init_ẋ::Function = () -> 0f0,
	init_θ̇::Function = () -> 0f0) where T<:Real

	#the action space is full throttle forward or backwards or idle
	actions = [-f, zero(T), f]

	#create a vehicle to use in simulation steps
	vehicle = CartPoleVehicle(m, m_c, l, k, m_f, μ_c)
	
	function failure(s::CartPoleState)
		(abs(s.x) > x_max) || (abs(s.θ) > θ_max)
	end

	function step(s::CartPoleState{T}, f::T)
		s′ = cartpole_runge_kutta_step(vehicle, s, g, f, h)
		reward = -T(failure(s′)) 
		return (reward, s′)
	end

	initialize_state() = CartPoleState(init_x(), init_θ(), init_ẋ(), init_θ̇())

	ptf_deterministic = StateMDPTransitionDeterministic((s, i_a) -> step(s, actions[i_a]), initialize_state())

	TabularRL.StateMDP(actions, ptf_deterministic, initialize_state, failure)
end

# ╔═╡ a81603a0-34ee-4a9e-a8f8-7994c4d09cee
md"""
### Episode Testing and Visualizaiton

Now that we have the ability to create MDPs with different constraints, we can test different parameters and see what makes the most sense for our problem.  As a starting point, we can test the behavior of the cart under simple single action policies and see the behavior.  Once we have a trajectory, we can also decide how to display the data to get the most insight.
"""

# ╔═╡ 7356e02e-7445-439d-a386-0b244541a443
# ╠═╡ skip_as_script = true
#=╠═╡
const test_cartpole_mdp = create_cartpole_mdp()
  ╠═╡ =#

# ╔═╡ 116bac12-7406-4f6d-9dab-ef4a75a98495
md"""
Notice that this function creates two MDPs, one that provides a distribution of transition states and one that samples them.  Since the problem is deterministic right now, both forms are equivalent but in the future that could be changed.  We can use either MDP to run an episode.  By default this will run an episode with the random policy.
"""

# ╔═╡ ab796133-dd92-4535-ab8a-7ebc8875eb45
#=╠═╡
const cartpole_episode_sample = runepisode(test_cartpole_mdp)
  ╠═╡ =#

# ╔═╡ 8b3c3da4-0ab2-4294-a6f6-84470669a5d9
md"""
From this episode we get the usual sequence of states which represent the positions and velocities of $x$ and $\theta$ for the vehicle.  As a starting point we can simply plot each of these values through time along with the action taken.
"""

# ╔═╡ 91a7b6c3-17aa-43cf-93aa-4ecc5f5019dc
#=╠═╡
function display_cartpole_episode(states::Vector{S}, actions::Vector{Int64}) where S<:CartPoleState
	fields = [:x, :θ, :ẋ, :θ̇]
	names = ["x", "θ", "ẋ", "θ̇"]
	yaxes = ["y", "y2", "y", "y2"]
	x = [s.t for s in states] #time history in seconds
	state_traces = [begin
		y = [getfield(s, f) for s in states]
		scatter(x = x, y = y, name = names[i], yaxis = yaxes[i])
	end
	for (i, f) in enumerate(fields)]
	plot(state_traces, Layout(xaxis_title = "Time(s)", yaxis_title = "Horizontal Position", yaxis2 = attr(title = "Pole Angle (Radians)", overlaying = "y", side = "right"), legend_orientation = "h"))
end
  ╠═╡ =#

# ╔═╡ 7fa7d6f4-87ac-4e7b-b09f-588800c97664
#=╠═╡
display_cartpole_episode(cartpole_episode_sample[1], [1])
  ╠═╡ =#

# ╔═╡ f430c6c9-914f-4db9-962a-012871a91a71
md"""Episode Step"""

# ╔═╡ cb5f26a2-cca2-4450-ae84-3cebd702a086
#=╠═╡
@bind display_step Slider(1:length(cartpole_episode_sample[1]); show_value=true)
  ╠═╡ =#

# ╔═╡ b45914c8-766b-4509-a6e6-92b093fa83b8
md"""
Running this repeatedly we see that after some initial movement, the pole eventually falls to one side or the other in a little over 2 seconds.  This MDP was initialized with the default values so the step size is 0.001 seconds.  Now that we have a working simulator, we can test different step sizes with the same initial conditions to see how accurate the simulation is.
"""

# ╔═╡ c308859b-7f95-461b-b9d8-98249aa92111
#=╠═╡
function evaluate_cartpole_stepsize(step_size_multiples::Vector{Int64}; θ_init = 0.00001f0, reference_step = 1f-3)
	make_mdp(h) = create_cartpole_mdp(h = h, init_θ = () -> θ_init)
	π(s) = 2 #idle action
	reference_episode = runepisode(make_mdp(reference_step); π = π)
	comparison_episodes = [begin
		mdp = make_mdp(reference_step*m)
		output = runepisode(mdp; π = π)
	end
	for m in step_size_multiples]

	error_traces = [begin
		reference_θs = [s.θ for s in reference_episode[1]][1:step_size_multiples[i]:end]
		θs = [s.θ for s in comparison_episodes[i][1]]
		times = [s.t for s in comparison_episodes[i][1]]
		deltas = θs .- reference_θs
		rmse = sqrt(mean((deltas) .^2))
		h = comparison_episodes[i][1][2].t
		scatter(x = times, y = deltas, name = "Step Size = $h, RMSE = $rmse")
	end
	for i in eachindex(step_size_multiples)]
	
	traces = [begin
		states = output[1]
		sterm = output[4]
		times = vcat([s.t for s in states], sterm.t)
		angles = vcat([s.θ for s in states], sterm.θ)
		h = states[2].t
		scatter(x = times, y = angles, name = "Step Size = $h")
	end
	for output in [reference_episode; comparison_episodes]]
	
	p1 = plot(traces, Layout(xaxis_title = "Time(s)", yaxis_title = "Pole Angle in Radians", title = "Idle Policy with Initial Angle of $θ_init Radians"))
	p2 = plot(error_traces, Layout(xaxis_title = "Time(s)", yaxis_title = "Angle Difference from Reference"))
	md"""
	$p1 $p2
	"""
end
  ╠═╡ =#

# ╔═╡ 6d61cc94-9578-4232-848f-8a74ec42daae
#=╠═╡
evaluate_cartpole_stepsize([10, 100, 1000, 2000, 4000, 5000, 8000, 10000]; reference_step = 1f-5, θ_init = 0.1f0)
  ╠═╡ =#

# ╔═╡ a0df5198-62e1-47bb-86dc-82fb501e24eb
md"""
The errors are all within numerical noise limits up to a step size of 0.04.  Beyond that we start to see an increase in the RMSE, so choosing 0.04 should ensure accurate simulations.  Next we can consider the behavior under force.
"""

# ╔═╡ 2d28b4af-0302-4dc4-9462-1ac6a083375f
md"""
#### Choosing Force and Angle Limits

Given a maximum horizontal force we can apply to the vehicle, there is always some angle beyond which the pole will necessarily fall.  We can explore what this limit is by initializing the pole at a given angle and applying the maximum throttle.
"""

# ╔═╡ fbfdf045-e627-442e-8ecf-81e9c8007679
# ╠═╡ skip_as_script = true
#=╠═╡
function test_cartpole_throttle(θ_init, throttle)
	mdp = create_cartpole_mdp(h = 4f-2, init_θ = () -> θ_init, f = throttle)
	π(s) = 3 #maximum throttle forward
	output = runepisode(mdp; π = π, max_steps = 25_000)
end
  ╠═╡ =#

# ╔═╡ c03d9058-25f0-49a1-9283-9d7d7492afd2
#=╠═╡
@bind throttle_params PlutoUI.combine() do Child
	md"""
	Initial Pole Angle in Degrees: $(Child(NumberField(0f0:90f0, default = 70f0)))
	Throttle Force: $(Child(NumberField(1f0:10_000f0, default = 300)))
	"""
end
  ╠═╡ =#

# ╔═╡ b2277d3b-7bc3-42ec-a685-bf45c4285caf
#=╠═╡
const throttle_episode = test_cartpole_throttle(deg2rad(throttle_params[1]), throttle_params[2])[1]
  ╠═╡ =#

# ╔═╡ 38d20b01-e6a2-46fa-8d92-a1725565a7d8
#=╠═╡
display_cartpole_episode(throttle_episode, [1])
  ╠═╡ =#

# ╔═╡ 798635bb-baf7-4069-8f40-a80f04d372ab
md"""
For an angle of 70°, we see that the inflection point is 296 to 297 throttle below which the pole falls to the right and above which the pole falls to the left.  Therefore we can set a failure angle of 70° and a throttle force value of 300 which will give it plenty of force to save the pole at any angle shy of 70° (assuming it doesn't already have velocity in the falling direction there).  We can also test here how far the cart moves under this force with different starting positions.  Under the full acceleration starting at 70° the cart reaches a position of 50.4, a horizontal velocity of 52.4, and an angular velocity of -7.44.  We can use these ranges to think about encoding the state into something like tile coding.
"""

# ╔═╡ b762a7f7-0a84-47e8-9425-f8982665ab7c
#=╠═╡
@bind display_step2 Slider(1:length(throttle_episode))
  ╠═╡ =#

# ╔═╡ 86c8efc2-970a-45a7-bc5e-10010cb39086
md"""
### Cart Pole Tile Coding

To use tile coding with this type of state, we need to define the range of each relevant variable for the state, which is the 4 non-time parameters.  By default, we will use a throttle value of 300 and a step size of 0.04.  Based on the analysis above, we can constrain the other variables to x = [-50, 50], ẋ = [-50, 50], θ̇ = [-10, 10] with a failure angle of 70°.
"""

# ╔═╡ dda1399e-d232-478d-9a38-6891430b8755
function setup_cartpole_problem(;h = 4f-2, f = 300f0, x_max = 50f0, θ_max = deg2rad(70f0), ẋ_max = 50f0, θ̇_max = 10f0, num_tiles = (8, 8, 8, 8), θ_range = 0.2f0, num_tilings = 8, kwargs...)
	init_θ(;θ_range = θ_range) = rand([-θ_range, θ_range])
	mdp = create_cartpole_mdp(h = h, f = f, x_max = x_max, θ_max = θ_max, init_θ = init_θ, kwargs...)
	setup = tile_coding_feature_setup(mdp, (-x_max, -θ_max, -ẋ_max, -θ̇_max), (x_max, θ_max, ẋ_max, θ̇_max), num_tiles, num_tilings; value_inds = (:x, :θ, :ẋ, :θ̇))
	(mdp = mdp, setup = setup)
end

# ╔═╡ ca515bd9-6ff9-4642-b0ac-f7cfd522e7f6
# ╠═╡ skip_as_script = true
#=╠═╡
const cartpole_tile_setup = setup_cartpole_problem()
  ╠═╡ =#

# ╔═╡ 59766450-1f4d-451a-9fe9-bca26596d955
md"""
### Eligibility Trace Control Solutions
"""

# ╔═╡ 75871e7e-2834-4e81-940b-9dd063733e1e
md"""
#### Linear Approximation with Tile Coding
"""

# ╔═╡ ec0ba4b3-3af4-464e-916e-e34df1605c8e
function run_cartpole_tilecoding(α, λ; algo::Function = sarsa_λ_linear, γ = 0.9f0, num_steps = 10_000, ϵ = 0.01f0, num_tiles = (8, 8, 8, 8), num_tilings = 8, kwargs...)
	mdp, setup = setup_cartpole_problem(;num_tiles = num_tiles, num_tilings = num_tilings)
	algo(mdp, γ, λ, typemax(Int64), num_steps, setup.feature_vector, setup.update_feature_vector!; α = α, ϵ = ϵ, kwargs...)
end

# ╔═╡ 70f0a00c-86f5-4a03-8b95-1333afba30e7
#=╠═╡
function run_cartpole_tilecoding_trial(args...; kwargs...)
	output = run_cartpole_tilecoding(args...; kwargs...)
	episode_steps = output.episode_steps[2:end] .- output.episode_steps[1:end-1]
	mean(episode_steps)
end
  ╠═╡ =#

# ╔═╡ 76b7abfb-e95a-4cf8-9ac1-8354882845cc
#=╠═╡
const cartpole_tilecoding_λ_param_study = setup_parameter_study(run_cartpole_tilecoding_trial, (:α, :λ), (algo = sarsa_λ_linear, compute_value = compute_sarsa_value, trace_type = AccumulatingTrace(), num_steps = 10_000, num_tiles = (8, 8, 8, 8), num_tilings = 8, γ = 0.9f0))
  ╠═╡ =#

# ╔═╡ 752297a5-6b66-4d96-931c-82352f9f0a35
#=╠═╡
function display_cartpole_λ_parameter_study(α_list, λ_list, study::NamedTuple; num_trials = Base.Threads.nthreads(), ymin = 100, ymax = 400, num_steps = 50_000, kwargs...)
	traces = [begin
		y = [begin
			study.update_results!(α, λ; num_steps = num_steps, num_trials = num_trials, kwargs...)
		end
		for α in α_list]
		scatter(x = α_list, y = y, name = "λ = $λ")
	end
	for λ in λ_list]
	plot(traces, Layout(xaxis_title = "Learning Rate", yaxis_title = "Average Steps Per Episode Averaged <br> Over the First $num_steps Steps and $num_trials Runs", yaxis_range = [ymin, ymax], xaxis_type = "log"))
end
  ╠═╡ =#

# ╔═╡ bf5c80bf-687f-4ade-afe5-6927c1733f64
md"""
##### Linear Parameter Study
"""

# ╔═╡ 36f54330-4d8a-4f5e-ac64-aab02468ead0
#add interactive elements to parameter studies to select the number of tiles, type of trace etc... and layer size for the fcann studies
#add function to set up a function for λ param studies where it generates results over a list of learning rate and λ values.  This can also be done for the non-λ algos that would only have the learning rate

# ╔═╡ fbef62e4-404b-4859-9a9f-92b248395d7c
#=╠═╡
@bind run_cartpole_tilecoding_param_studies CounterButton("Run Parameter Studies (could take several minutes)")
  ╠═╡ =#

# ╔═╡ de341cf8-822d-43ca-ba10-cc350afe8509
#=╠═╡
if run_cartpole_tilecoding_param_studies > 0
	display_cartpole_λ_parameter_study(Base.LogRange(1f-5, 1f-1, 7), [0f0, 0.2f0, 0.3f0, 0.4f0, 0.5f0, 0.9f0], cartpole_tilecoding_λ_param_study; trace_type = ReplacingTrace(), num_tiles = (4, 4, 4, 4), num_tilings = 8, num_steps = 100_000, algo = true_online_dp_λ, ymax = 1000)
else
	md"""
	Waiting to run parameter study
	"""
end
  ╠═╡ =#

# ╔═╡ 0ee1882f-f9b2-4292-8b92-adc2eb2edb0e
# ╠═╡ skip_as_script = true
#=╠═╡
const cartpole_tilecoding_λ_best = run_cartpole_tilecoding(5f-5, 0.2f0; algo = true_online_dp_λ, num_tiles = (10, 4, 4, 4), num_tilings = 8, num_steps = 100_000)
  ╠═╡ =#

# ╔═╡ bee67ec3-98b8-41b9-895c-7d2db4cebfab
#=╠═╡
function display_cartpole_result(solution::NamedTuple)
	episode_steps = solution.episode_steps[2:end] .- solution.episode_steps[1:end-1]
	episode = runepisode(cartpole_tile_setup.mdp; π = s -> solution.value_function(s).maximizing_action, max_steps = 25_000)
	p1 = display_cartpole_episode(episode[1], [1])
	p2 = plot(scatter(y = 0.04*cumsum(episode_steps) ./ (1:length(episode_steps))), Layout(xaxis_title = "Episode", yaxis_title = "Seconds Per Episode"))
	md"""
	$p1 $p2
	"""
end
  ╠═╡ =#

# ╔═╡ 193e034f-1278-436f-b534-defc870cd36b
#=╠═╡
display_cartpole_result(cartpole_tilecoding_λ_best)
  ╠═╡ =#

# ╔═╡ 2705f545-7c2c-446e-a625-97908a04fefe
md"""
#### Non-linear Approximation with Neural Network
"""

# ╔═╡ 9bdd4ce4-e9b9-4cc1-8c5d-fbc4c7a6f74a
function normalized_feature_setup(problem::Union{StateMDP{T, S, A, P, F1, F2, F3}, StateMRP{T, S, P, F1, F2}}, extract_values::Function, min_value::V, max_value::V; range::T = one(T)) where {T<:Real, N, S, V <: Union{T, NTuple{N, T}}, A, P, F1<:Function, F2<:Function, F3<:Function}
	#extract_values must transform a state into type V where V is either a tuple of values or a value
	
	#states must be tuples with k elements or some number value
	k = V == T ? 1 : N

	s_range = if k == 1
		max_value - min_value
	else
		Tuple(max_value[i] - min_value[i] for i in 1:k)
	end

	feature_vector = zeros(T, k)

	function update_feature_vector!(x::Vector{T}, s::S)
		values = extract_values(s)
		@inbounds @simd for i in 1:k
			x[i] = (2*range)*scale_state(values[i], min_value[i], s_range[i]) - range
		end
		return x
	end

	(feature_vector = feature_vector, update_feature_vector! = update_feature_vector!, num_features = k)
end

# ╔═╡ d9635e75-6e5c-41d2-b906-5025b58f9d0f
function run_mountaincar_λ_fcann(α, λ, algo; num_steps = 50_000, layers = [16, 16], kwargs...)
	mdp = MountainCarTask.deterministic_mdp
	setup = normalized_feature_setup(mdp, identity, (-1.2f0, -0.07f0), (0.5f0, 0.07f0); range = 1.725f0)
	algo(mdp, 1f0, λ, typemax(Int64), num_steps, setup.feature_vector, setup.update_feature_vector!, layers; α = α, kwargs...)
end

# ╔═╡ e2ccff6e-6791-4338-8ec2-eef47f388bb1
function run_mountaincar_λ_fcann_trial(α, λ, algo; kwargs...)
	output = run_mountaincar_λ_fcann(α, λ, algo; kwargs...)
	step_history = output.episode_steps
	isempty(step_history) && return NaN
	l = length(step_history)
	return step_history[end] / l
end

# ╔═╡ 48923864-c40c-45ca-907e-2c0c03587f2c
# ╠═╡ skip_as_script = true
#=╠═╡
const mountaincar_semi_gradient_fcann_parameter_study = setup_parameter_study(run_mountaincar_λ_fcann_trial, (:α, :λ, :algo), (num_steps = 50_000, layers = [16, 16], reslayers = 1, compute_value = compute_sarsa_value, ϵ = 0.01f0, trace_type = AccumulatingTrace()))
  ╠═╡ =#

# ╔═╡ 7af6f5ed-178d-4b90-9226-02be6b13ed5a
#=╠═╡
if run_mountaincar_λ_fcann_study1 > 0
	display_mountaincar_λ_parameter_study(Base.LogRange(1f-5, 1f-2, 6), [0.9f0, 0.99f0, 0.999f0], mountaincar_semi_gradient_fcann_parameter_study; algo = sarsa_λ_fcann, ymin = 150, ymax = 5000, compute_value = compute_expected_sarsa_value, layers = fill(32, 4), num_steps = 100_000, trace_type = AccumulatingTrace())
else
	md"""Waiting to run parameter study"""
end
  ╠═╡ =#

# ╔═╡ bf3584aa-2a17-43ef-95aa-035aa9505dd2
#=╠═╡
if run_mountaincar_λ_fcann_study3 > 0
	display_mountaincar_λ_parameter_study(Base.LogRange(1f-8, 1f-3, 8), [0.9f0, 0.99f0, 0.999f0], mountaincar_semi_gradient_fcann_parameter_study; algo = dp_λ_fcann, ymin = 150, ymax = 5000, reslayers = 1, layers = fill(32, 4), num_steps = 100_000, trace_type = AccumulatingTrace())
else
	md"""Waiting to run parameter study"""
end
  ╠═╡ =#

# ╔═╡ d85e292e-5b13-4b51-9cb9-7c5161ba70a6
# ╠═╡ skip_as_script = true
#=╠═╡
const mountaincar_fcann_λ_best = run_mountaincar_λ_fcann(8f-6, 0.99f0, dp_λ_fcann; num_steps = 100_000, layers = fill(32, 4), reslayers = 1, use_gpu=false)
  ╠═╡ =#

# ╔═╡ 6ee6d7d0-9381-4413-a361-e836ac5240de
function setup_cartpole_problem_fcann(;h = 4f-2, f = 300f0, x_max = 50f0, θ_max = deg2rad(70f0), ẋ_max = 50f0, θ̇_max = 10f0, kwargs...)
	init_θ() = rand([-0.02f0, 0.02f0])
	mdp = create_cartpole_mdp(h = h, f = f, x_max = x_max, θ_max = θ_max, init_θ = init_θ, kwargs...)
	extract_values(s) = (s.x, s.θ, s.ẋ, s.θ̇)
	setup = normalized_feature_setup(mdp, extract_values, (-x_max, -θ_max, -ẋ_max, -θ̇_max), (x_max, θ_max, ẋ_max, θ̇_max))
	(mdp = mdp, setup = setup)
end

# ╔═╡ 834055ef-0bdc-4b4b-9c3f-ce7f43864ef7
function run_cartpole_fcann(α, λ; algo::Function = sarsa_λ_fcann, γ = 0.9f0, num_steps = 10_000, ϵ = 0.01f0, layers = [4, 4], kwargs...)
	mdp, setup = setup_cartpole_problem_fcann()
	algo(mdp, γ, λ, typemax(Int64), num_steps, setup.feature_vector, setup.update_feature_vector!, layers; α = α, ϵ = ϵ, kwargs...)
end

# ╔═╡ 4b13b020-0dd2-45ff-adb5-d67cdd3a77f6
#=╠═╡
function run_cartpole_fcann_trial(args...; kwargs...)
	output = run_cartpole_fcann(args...; kwargs...)
	episode_steps = output.episode_steps[2:end] .- output.episode_steps[1:end-1]
	mean(episode_steps)
end
  ╠═╡ =#

# ╔═╡ 9305f6ca-bec7-4002-af6b-f4142cc78d91
#=╠═╡
const cartpole_fcann_λ_param_study = setup_parameter_study(run_cartpole_fcann_trial, (:α, :λ), (algo = sarsa_λ_fcann, compute_value = compute_sarsa_value, trace_type = AccumulatingTrace(), num_steps = 10_000, γ = 0.9f0, layers = fill(16, 4), reslayers = 1))
  ╠═╡ =#

# ╔═╡ 8bdb4ac6-164f-4b3b-9f07-3bb5545bd132
md"""
##### Non-linear Parameter Study
"""

# ╔═╡ c9a83977-bb0a-4f58-940a-1647b2864aec
#=╠═╡
@bind run_cartpole_fcann_param_studies CounterButton("Run Parameter Studies (could take several minutes)")
  ╠═╡ =#

# ╔═╡ 9601eb2d-de36-436f-91c0-906dcd2921f8
#=╠═╡
@bind fcann_study_layers PlutoUI.combine() do Child
	md"""
	Layer Size: $(Child(NumberField(1:256, default = 8)))
	Num Layers: $(Child(NumberField(1:100, default = 3)))
	"""
end |> confirm
  ╠═╡ =#

# ╔═╡ 565ac829-0b95-449f-b787-233a8d6b935a
#=╠═╡
if run_cartpole_fcann_param_studies > 0
	display_cartpole_λ_parameter_study(Base.LogRange(1f-3, 5f-1, 6), [0f0, 0.2f0, 0.5f0, 0.9f0], cartpole_fcann_λ_param_study; layers = fill(fcann_study_layers[1], fcann_study_layers[2]), num_steps = 100_000, algo = dp_λ_fcann, ymax = 1000)
else
	md"""
	Waiting to run parameter study
	"""
end
  ╠═╡ =#

# ╔═╡ d0e96989-c165-48d6-ba4b-4eab05fcb638
const cartpole_fcann_λ_best = run_cartpole_fcann(.14f0, 0.0f0; algo = dp_λ_fcann, layers = fill(4, 4), num_steps = 100_000)

# ╔═╡ 993f8193-7488-415d-85d5-9a7a83f6bf71
#=╠═╡
display_cartpole_result(cartpole_fcann_λ_best)
  ╠═╡ =#

# ╔═╡ 0358288e-be4e-46c2-ac4c-16ace6f50187
md"""
# Dependencies
"""

# ╔═╡ 2fb6e491-be69-44e8-ae2d-9cb13ec0b66f
md"""
## MDP Tools
"""

# ╔═╡ 2394cac9-3349-4684-9f08-506e4fe77a0d
md"""
## Notebook Only
"""

# ╔═╡ 326b3355-7941-403b-bf1e-3031f585f666
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

# ╔═╡ 5616d294-892a-40bc-a35f-35e9e0ee55e2
md"""
## Visualization Tools
"""

# ╔═╡ e4112acf-af6b-4cd7-be24-cff2ed77200d
md"""
### Gridworld
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

# ╔═╡ b9559522-69e1-415e-8cdc-2454d1edebb8
#=╠═╡
function gridworld_sarsaλ_trial(λ, α; steps = 50_000, kwargs...)
	output = sarsa_λ(gridworld_mdp, 1f0, λ, typemax(Int64), steps; α = α, kwargs...)
	step_history = output.episode_steps
	isempty(step_history) && return NaN
	step_history[end]/length(step_history)
end
  ╠═╡ =#

# ╔═╡ 3b96d763-66cf-414b-81c3-5902ffecc6d8
#=╠═╡
const gridworld_sarsaλ_study = setup_parameter_study(gridworld_sarsaλ_trial, (:λ, :α), (steps = 50_000, ϵ = 0.01f0, compute_value = compute_sarsa_value, trace_type = ReplacingTrace()))
  ╠═╡ =#

# ╔═╡ 1a35841b-5316-458c-a722-43c2c99d71f3
#=╠═╡
function gridworld_true_online_sarsaλ_trial(λ, α; steps = 50_000, kwargs...)
	parameters = initialize_state_action_value(gridworld_mdp)
	output = true_online_sarsa_λ!(parameters, gridworld_mdp, 1f0, λ, typemax(Int64), steps; α = α, kwargs...)
	# (; output..., params = parameters)
	step_history = output.episode_steps
	isempty(step_history) && return NaN
	step_history[end]/length(step_history)
end
  ╠═╡ =#

# ╔═╡ 5b42fa92-1ebf-4b0a-ac69-f147bd0defd7
#=╠═╡
const gridworld_true_online_sarsaλ_study = setup_parameter_study(gridworld_true_online_sarsaλ_trial, (:λ, :α), (steps = 50_000, ϵ = 0.01f0, compute_value = compute_sarsa_value))
  ╠═╡ =#

# ╔═╡ a36d205c-9a77-4b24-8e57-7bfceee9f4af
#=╠═╡
function display_gridworld_sarsaλ_parameter_study(steps::Integer, nruns::Integer; λ_list = [0f0, 0.5f0, 0.6f0, 0.7f0, 0.8f0, 0.9f0, 0.99f0], α_list = Base.LogRange(0.1f0, 1f0, 10), kwargs...)
	value_iteration_output = value_iteration_v(gridworld_mdp, 1f0; show_message = false)
	best_steps = abs(value_iteration_output.final_value[gridworld_mdp.initialize_state_index()])
	traces = [begin 
		output = [gridworld_sarsaλ_study.update_results!(λ, α; steps = steps, num_trials = nruns, kwargs...) for α in α_list]
		scatter(x = α_list, y = output, name = "λ = $λ")
	end for λ in λ_list]
	plot(traces, Layout(xaxis_title = "Learning Rate", yaxis_title = "Steps per Episode <br> Averaged Over $steps Steps & $nruns Runs", yaxis_range = [best_steps, best_steps*2.6], xaxis_type = "log"))
end
  ╠═╡ =#

# ╔═╡ 8d52f740-e1bf-4ac7-a299-38e7767a0831
#=╠═╡
display_gridworld_sarsaλ_parameter_study(10_000, 50; ϵ = 0.01f0, trace_type = study_trace_type, α_list = Base.LogRange(0.01f0, 1f0, 8))
  ╠═╡ =#

# ╔═╡ e323cbc2-1396-43fb-969a-1837bb60c5b5
#=╠═╡
display_gridworld_sarsaλ_parameter_study(10_000, 50; ϵ = 0.01f0, computer_value = compute_expected_sarsa_value, α_list = Base.LogRange(0.01f0, .8f0, 8))
  ╠═╡ =#

# ╔═╡ 1277b3b2-8733-4c26-a9aa-39b07769bca4
#=╠═╡
function display_gridworld_true_online_sarsaλ_parameter_study(steps::Integer, nruns::Integer; λ_list = [0f0, 0.5f0, 0.6f0, 0.7f0, 0.8f0, 0.9f0, 0.99f0], α_list = Base.LogRange(0.1f0, 1f0, 10), kwargs...)
	value_iteration_output = value_iteration_v(gridworld_mdp, 1f0; show_message = false)
	best_steps = abs(value_iteration_output.final_value[gridworld_mdp.initialize_state_index()])
	traces = [begin 
		output = [gridworld_true_online_sarsaλ_study.update_results!(λ, α; steps = steps, num_trials = nruns, kwargs...) for α in α_list]
		scatter(x = α_list, y = output, name = "λ = $λ")
	end for λ in λ_list]
	plot(traces, Layout(xaxis_title = "Learning Rate", yaxis_title = "Steps per Episode <br> Averaged Over $steps Steps & $nruns Runs", yaxis_range = [best_steps, best_steps*2.6], xaxis_type = "log"))
end
  ╠═╡ =#

# ╔═╡ e047cce1-11a5-4bcb-8668-a767628da140
#=╠═╡
display_gridworld_true_online_sarsaλ_parameter_study(10_000, 50; ϵ = 0.01f0, α_list = Base.LogRange(0.01f0, .8f0, 8))
  ╠═╡ =#

# ╔═╡ f5a8cc64-f7a3-44ef-b925-d11df6a414f6
#=╠═╡
display_gridworld_true_online_sarsaλ_parameter_study(10_000, 50; ϵ = 0.01f0, α_list = Base.LogRange(0.01f0, 1.5f0, 8), compute_value = compute_expected_sarsa_value)
  ╠═╡ =#

# ╔═╡ 9db3ed98-a94d-4adc-a45f-75eca432a1e9
# ╠═╡ skip_as_script = true
#=╠═╡
show_grid_value(mdp::TabularMDP, Q, name; kwargs...) = show_grid_value(mdp.states, mdp.terminal_states, mdp.initialize_state_index, Q, name; kwargs...)
  ╠═╡ =#

# ╔═╡ 4a8bc15c-8f4d-4017-915f-d2b27c1a6bd0
# ╠═╡ skip_as_script = true
#=╠═╡
show_grid_probabilities(mdp::TabularMDP, Q, name; kwargs...) = show_grid_probabilities(mdp.states, mdp.terminal_states, mdp.initialize_state_index, Q, name; kwargs...)
  ╠═╡ =#

# ╔═╡ 4160de31-3c8e-4051-b618-24112bbcc70e
# ╠═╡ skip_as_script = true
#=╠═╡
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
  ╠═╡ =#

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
# ╠═╡ skip_as_script = true
#=╠═╡
show_grid_policy(mdp::TabularMDP, π, name; kwargs...) = show_grid_policy(mdp.states, mdp.initialize_state_index, mdp.terminal_states, π, name; kwargs...)
  ╠═╡ =#

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
function show_gridworld_exact_solution(mdp; kwargs...)
	output = value_iteration_v(mdp, 1f0; kwargs...)
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
show_gridworld_exact_solution(gridworld_mdp; show_message = false)
  ╠═╡ =#

# ╔═╡ bb71215f-fa97-4eca-a950-be4cb037bb00
#=╠═╡
function show_gridworld_sarsaλ_solution(mdp, λ, episodes; kwargs...)
	output = sarsa_λ(mdp, 1f0, λ, episodes, typemax(Int64); kwargs...)
	greedy_policy = TabularRL.make_greedy_policy(output.final_parameters' |> collect)
	
	@htl("""
	<div style = "display: flex">
	$(show_grid_policy(mdp, greedy_policy, "sarsa_λ_policy"))
	$(show_grid_value(mdp, [output.value_function(s).maximizing_value for s in mdp.states], "sarsa_λ_values"; square_pixels = 40))
	$(plot_path(mdp, greedy_policy; square_pixels = 40))
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ 26a64c5d-6c21-4e3b-af75-e8682e8d5ea1
#=╠═╡
show_gridworld_sarsaλ_solution(gridworld_mdp, gridworld_λ_params.λ, gridworld_λ_params.num_episodes; α = gridworld_λ_params.α, ϵ = 0.01f0)
  ╠═╡ =#

# ╔═╡ f0fa43fc-6221-469d-accc-fff87b005a17
md"""
### Mountaincar
"""

# ╔═╡ 17083d16-6a9c-47e8-99f5-099067210029
#=╠═╡
function plot_mountaincar_values(v̂_mountain_car, π; n1 = 100, n2 = 100)
	xvals = LinRange(-1.2f0, 0.5f0, n1)
	vvals = LinRange(-0.07f0, 0.07f0, n2)
	values = zeros(Float32, n1, n2)
	actions = zeros(Float32, n1, n2)
	for (i, x) in enumerate(xvals)
		for (j, v) in enumerate(vvals)
			v̂ = v̂_mountain_car((x, v))
			values[j, i] = v̂
			actions[j, i] = π((x, v))
		end
	end
	p1 = plot(heatmap(x = xvals, y = vvals, z = values), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Learned Value Function"))
	p2 = plot(heatmap(x = xvals, y = vvals, z = actions, colorscale = "rb", showscale = false), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Policy (blue = accelerate left, <br>red = accelerate right, gray = no acceleration)"))
	@htl("""
	<div style = "display:flex; height: 400px;">
	$p1 
	$p2
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ 2e6c8ff9-4710-410d-b7e9-80563cc2af21
#=╠═╡
function show_mountaincar_trajectory(π::Function, max_steps::Integer, name)
	states, actions, rewards, sterm, nsteps = runepisode(MountainCarTask.mdp; π = π, max_steps = max_steps)
	positions = [s[1] for s in states]
	velocities = [s[2] for s in states]
	tr1 = scatter(x = positions, y = velocities, mode = "markers", showlegend = false)
	tr2 = scatter(y = positions, showlegend = false)
	tr3 = scatter(y = [MountainCarTask.actions[i] for i in actions], showlegend = false)
	p1 = plot(tr1, Layout(xaxis_title = "position", yaxis_title = "velocity", xaxis_range = [-1.2, 0.5], yaxis_range = [-0.07, 0.07]))
	p2 = plot(tr2, Layout(xaxis_title = "time", yaxis_title = "position"))
	p3 = plot(tr3, Layout(xaxis_title = "time", yaxis_title = "action"))
	mdname = Markdown.parse(name)
	@htl("""
	$mdname
	Total Reward: $(sum(rewards))
	<div style = "display:flex; height: 400px">
	$p1 
	$p2 
	$p3
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ cb992290-4bc2-4ebb-be3c-dc5d513ee5ef
#=╠═╡
function plot_mountaincar_action_values(q̂_mountain_car, n1, n2)
	xvals = LinRange(-1.2f0, 0.5f0, n1)
	vvals = LinRange(-0.07f0, 0.07f0, n2)
	values = zeros(Float32, n1, n2)
	actions = zeros(Float32, n1, n2)
	for (i, x) in enumerate(xvals)
		for (j, v) in enumerate(vvals)
			q̂ = q̂_mountain_car((x, v))
			values[j, i] = q̂.maximizing_value
			actions[j, i] = MountainCarTask.actions[q̂.maximizing_action]
		end
	end
	p1 = plot(heatmap(x = xvals, y = vvals, z = values), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Learned Value Function"))
	p2 = plot(heatmap(x = xvals, y = vvals, z = actions, colorscale = "rb", showscale = false), Layout(xaxis_title = "position", yaxis_title = "velocity", title = "Policy (blue = accelerate left, <br>red = accelerate right, gray = no acceleration)"))
	[p1 p2]
end
  ╠═╡ =#

# ╔═╡ abe1f077-d211-4488-a10c-ec2ca5aac328
#=╠═╡
function display_mountaincar_results(output::NamedTuple; nsmooth = 100, npoints = 1000, ϵ = 0.05f0)
	p1 = show_mountaincar_trajectory(s -> rand() < ϵ ? rand(1:3) : output.value_function(s).maximizing_action, 1000, "")

	kwargs = output.form_kwargs()
	v̂(s) = output.value_function(s; kwargs...)
	p2 = plot_mountaincar_action_values(v̂, 200, 200)

	rewards = output.episode_rewards
	if isempty(rewards)
		p3 = nothing
	elseif length(rewards) ≤ npoints
		p3 = plot(rewards)
	else
		rewards = smooth_error(rewards, nsmooth)
		l = length(rewards)
		sample_inds = round.(Int64, LinRange(1, l, npoints))
		p3 = plot(rewards[sample_inds])
	end
 
	@htl("""
		 $p1
		 <div style = "display: flex">
		 $p2
		 </div>
		 $p3
		 """)
end
  ╠═╡ =#

# ╔═╡ da3cde1e-d11a-4ce5-b134-347c3baba11d
#=╠═╡
display_mountaincar_results(mountaincar_tilecoding_λ_best)
  ╠═╡ =#

# ╔═╡ c3cd5a5f-6445-4df3-b492-0d22e641f37c
#=╠═╡
display_mountaincar_results(mountaincar_fcann_λ_best)
  ╠═╡ =#

# ╔═╡ 214ceb34-7e31-4c89-a328-a492244fd4cf
md"""
### Cart Pole
"""

# ╔═╡ 909104f3-44c2-44ae-8186-11fd74b3ba4e
#=╠═╡
function plot_cart(s::CartPoleState; xmin = -50, xmax = 50, θ̇_min = -10, θ̇_max = 10)
	s.x
	s.θ
	rad_angle = string(round(s.θ; sigdigits = 2), " Rads")
	deg_angle = string(round(rad2deg(s.θ); sigdigits = 2), "°")
	t1 = scatter(x = [0, sin(s.θ)], y = [0, cos(s.θ)], mode = "lines", color = "black")
	t2 = scatter(x = [sin(s.θ)], y = [cos(s.θ)], mode = "markers", color = "black")
	p1 = plot([t1, t2], Layout(yaxis_range = [-.1, 1.2], xaxis_range = [-1.2, 1.2], xaxis_scaleanchor = "y", height = 200, showlegend = false, title = "Pole Angle", xaxis_tickvals = [], yaxis_tickvals = [], annotations = [attr(x = 1.75, y = 0.5, showarrow=false, text = rad_angle, font_size = 20), attr(x = -1.75, y = 0.5, showarrow=false, text = deg_angle, font_size = 20)]))
	p2 = plot(bar(y = [0], x = [s.x], orientation = "h"), Layout(height = 200, yaxis = attr(tickvals = [], ticknames = []), xaxis_range = [xmin, xmax], xaxis_title = "Horizontal Position = $(s.x)"))
	p3 = plot(indicator(mode = "gauge+number+delta", value = s.θ̇, title_text = "Angular Speed in Radians per Second", delta_reference = 0, gauge_axis_range = [-10, 10]), Layout(height = 200))
	@htl("""
	<div style = "display: flex;">
	$p1 
	$p2 
	$p3
	</div>
	""")
end
  ╠═╡ =#

# ╔═╡ 61e6b0a3-a344-4fc6-b77c-36ef7cd138cd
#=╠═╡
plot_cart(cartpole_episode_sample[1][display_step])
  ╠═╡ =#

# ╔═╡ d1eef08b-60b5-4475-bb48-d8e8cb52235f
#=╠═╡
plot_cart(throttle_episode[display_step2])
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
PlutoDevMacros = "a0499f29-c39b-4c5c-807c-88074221b949"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoProfile = "ee419aa8-929d-45cd-acf6-76bd043cd7ba"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
BenchmarkTools = "~1.6.3"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
PlutoDevMacros = "~0.9.1"
PlutoPlotly = "~0.6.5"
PlutoProfile = "~0.4.0"
PlutoUI = "~0.7.73"
StaticArrays = "~1.9.15"
StatsBase = "~0.34.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "b9a830e646eb02d1bdf90c89ded62dd35e7c8982"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

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
git-tree-sha1 = "7fecfb1123b8d0232218e2da0c213004ff15358d"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.3"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "980f01d6d3283b3dbdfd7ed89405f96b7256ad57"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "2.0.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

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
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6c72198e6a101cccdd4c9731d3985e904ba26037"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.1"

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
git-tree-sha1 = "d60eb76f37d7e5a40cc2e7c36974d864b82dc802"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.1"

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
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "277779adfedf4a30d66b64edc75dc6bb6d52a16e"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.6"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

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
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.11.1+1"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

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
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

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
version = "2025.5.20"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.1+0"

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
version = "1.12.0"
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
git-tree-sha1 = "1cb861c9295d79dc6e23170d4b33bce013f69643"
uuid = "a0499f29-c39b-4c5c-807c-88074221b949"
version = "0.9.1"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "8acd04abc9a636ef57004f4c2e6f3f6ed4611099"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.5"

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
git-tree-sha1 = "3faff84e6f97a7f18e0dd24373daa229fd358db5"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.73"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProfileCanvas]]
deps = ["FlameGraphs", "JSON", "Pkg", "Profile", "REPL"]
git-tree-sha1 = "41fd9086187b8643feda56b996eef7a3cc7f4699"
uuid = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
version = "0.1.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
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
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

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

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

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
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "a136f98cefaf3e2924a66bd75173d1c891ab7453"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.7"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

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
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

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
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.5.0+2"
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
# ╟─14b3b28e-1351-4b45-9a57-bfab846a2ffd
# ╟─57cf5ae7-d4dd-47e8-8090-c04fb39e0763
# ╟─34dda4bf-f78f-4c83-ba10-9b206d2fbcb8
# ╟─6f5168dc-f1f3-4533-a59e-bb85895f3b13
# ╠═0b2e26e4-e87a-49a8-a7a6-c5b225af82b3
# ╠═5542897d-eb37-4e96-ac33-d36fc8adb603
# ╠═9ec58129-a14f-40a9-9c41-809500181bdd
# ╠═5610a0ba-60a8-4da6-8f68-50b1c5e82686
# ╟─5e5fdcee-356e-46d4-a5b0-3c433aee989d
# ╟─1d3144f8-fecf-4c8f-8e47-626ad94ed15a
# ╠═24468748-009d-42a3-918d-4ba18b23c9ed
# ╟─be33e2cc-b6d7-48e1-bfbd-71a01f7ae161
# ╠═52098913-ea06-497d-afad-a9fef99fb428
# ╟─f92b8423-b05f-4058-ac91-4b3c6d447820
# ╠═3839f146-107c-45e9-bb94-b707982f4ce1
# ╠═3ab94b9e-4f50-4162-8b27-f6a81595f42f
# ╟─e99caf5c-7c13-4edd-b55b-dce93cc850c6
# ╠═900760f0-b253-4db7-8c4f-4ca34777198d
# ╟─373a89e3-0b8d-49a0-982e-8bb300538429
# ╟─2c3b163d-b4cd-4b40-a597-cbd103e135b6
# ╠═36d87b2a-6e1a-47b7-8af5-825d47e55eec
# ╠═b68f1171-6274-4d93-bf68-05b95cb5b2f8
# ╠═da8c5f8b-5ab6-4a2b-93e8-18be4284b932
# ╟─83a645f8-f806-4828-bc42-d24cfd26bad3
# ╠═9a75dc05-883b-47a6-b8f0-ae0799c5fc19
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
# ╠═d1903e34-0463-4a63-a74b-fb827451e542
# ╠═d9f89b2c-8df8-415a-a0a8-21744be88cec
# ╠═00015316-0d9d-41ce-a001-09aede10049c
# ╠═9c287f15-a78a-4e3f-a0b7-c901874b6cca
# ╟─ccf903ab-1509-4c53-bfe0-435152ecea4b
# ╟─f7cc35c5-c1ed-440d-8723-8c1b8b966b8f
# ╟─b36896b1-6802-48e1-8cd3-f08bf3b99e3e
# ╟─0086dc4a-e0ba-43f0-a721-296cd50e1a76
# ╠═43ba01e3-27d0-481c-8f93-8cd9c30b2a1d
# ╠═58ea2a87-3a8c-4fa3-877e-b1aa4b0ba36f
# ╠═01024201-7989-43f0-86f2-d3cd5ee219ae
# ╟─0525812d-7a86-4c5b-b5a8-36b4cfbd51fe
# ╠═b320dc0e-95dc-44d5-8ee4-455c4a858835
# ╠═3df4cd98-f754-4eca-8e16-e654576e283d
# ╠═efa11915-d86b-4686-85f9-84d7539e27cf
# ╠═fc9149ee-6d28-41b5-ab47-2c2a36e7a8d1
# ╠═84870ff4-d7a5-4214-abcb-3d74b7a8fe7b
# ╠═747700de-0a87-4ac9-a9cd-0bc11721836e
# ╠═3a03bd95-f43d-47b5-93d6-88a9ef61d2b4
# ╠═c7caa90d-26bf-4179-b869-3385cb75b943
# ╠═58739629-eff4-416a-b133-85ab5ec563fc
# ╠═2c8beba6-4436-4603-88f2-20f847c5e916
# ╠═84d71a29-7e38-4877-bfb8-57d4af2ec0d0
# ╟─31926565-8c2f-42a9-bc73-4f3001a38bf4
# ╠═4cab7b59-f080-4bea-86dc-3c860a618c35
# ╠═9c8765f5-0101-47e3-8780-65c197c14d6b
# ╠═258be19f-0b01-4d5e-ae82-2ef07ba4cc9c
# ╠═c3fb9a00-3aa0-43e5-98c8-306441166cc4
# ╠═5f623b73-4d7d-4c69-acaf-9a668c352bf9
# ╠═c5ae58b0-6f89-476a-8a72-9bb1cfd1a6be
# ╠═da6944b0-1d1b-4b92-a765-5e28c133cff5
# ╠═08c7beae-9dee-468e-b276-aa56a8f24f1f
# ╟─51274911-2eaa-4b18-b977-d0f735746bec
# ╟─afa6843b-9852-42c0-9ecd-06c408262334
# ╟─4b1d86d8-e6b9-4d09-b2a1-8c297414094c
# ╟─078eecc3-05b3-4a58-91c8-fc8c28c9b144
# ╟─d08aaab5-4065-41ef-867d-ea689c87a4f6
# ╟─b3c83ae6-9f3a-49a5-b726-872ef3a15853
# ╟─38512c69-54dc-4eb6-9a22-2a6bb3e38d89
# ╠═8d52f740-e1bf-4ac7-a299-38e7767a0831
# ╟─a4e10d47-2d41-4586-a328-11ea7234d7bd
# ╠═26a64c5d-6c21-4e3b-af75-e8682e8d5ea1
# ╠═6d671599-f635-4e1d-bef5-707891a756cc
# ╠═1441e34f-e8c5-46b5-b04c-31ea7f9f605a
# ╠═5c424e53-aeba-4a11-97c8-a9936d6ddb72
# ╠═b9559522-69e1-415e-8cdc-2454d1edebb8
# ╠═1a35841b-5316-458c-a722-43c2c99d71f3
# ╠═3b96d763-66cf-414b-81c3-5902ffecc6d8
# ╠═5b42fa92-1ebf-4b0a-ac69-f147bd0defd7
# ╠═a36d205c-9a77-4b24-8e57-7bfceee9f4af
# ╠═1277b3b2-8733-4c26-a9aa-39b07769bca4
# ╠═bb71215f-fa97-4eca-a950-be4cb037bb00
# ╟─06885c43-2ace-45ff-b77f-f3ceaaf999de
# ╠═e323cbc2-1396-43fb-969a-1837bb60c5b5
# ╟─4fa824ba-51a3-4f5a-a990-dd05bbf2526a
# ╟─7aa62007-0685-40d2-88ab-9c03add8e75a
# ╠═e047cce1-11a5-4bcb-8668-a767628da140
# ╟─a09643e9-d2c8-4fe0-a320-205badde72bf
# ╠═f5a8cc64-f7a3-44ef-b925-d11df6a414f6
# ╠═1d49a787-2a82-4f6d-a986-f2351fa82d18
# ╠═65aa9202-7e9a-4590-865e-0b08c7b98e1e
# ╠═65ca967c-2425-4c85-92e9-3f957e7ede2f
# ╠═771cca22-d61d-498a-98be-90fa59e09571
# ╠═b9467ce7-ace9-4034-a306-050bf1e53573
# ╠═ded7c8e0-f44c-44c9-afad-070d325c180b
# ╠═b1ae33a5-272f-43de-b9e2-eaa0423c34b8
# ╠═a2a6c291-3ea7-45d4-9608-40e25f1cfe7c
# ╟─8b6b5084-3972-4bd4-9ca2-423f1c627788
# ╟─5bc128ec-2934-4aa5-a922-9017f647e1b3
# ╠═5ad7c72e-b276-4dd8-a6de-df4e2e01f048
# ╠═c35b4242-8477-468f-bd86-32cda00229a4
# ╠═cd0b96eb-150c-4441-ad79-8c0305213cbd
# ╠═4bd74da4-382b-47af-bfca-78b3318e2df7
# ╠═0324b4e2-2544-4bd6-b310-8a330b5a92c5
# ╟─c19209dc-bddf-4390-95a9-fc1d1d836a8a
# ╟─5652f3fd-ec23-4dfb-a171-1e1ed0de275a
# ╟─a29d560d-fe30-4efc-8bce-90d652c738fa
# ╟─111cda26-bd25-49ed-9ba7-4ee8f71b063f
# ╟─aea15e6d-9873-406b-993b-04717dad01c6
# ╟─c57b4792-928a-4450-9364-786e9f186cc8
# ╟─118545ac-5ee9-4b61-b384-52dfb41a533b
# ╟─f1a8df55-a5ef-475e-a0c4-ed31b1c6c9f5
# ╟─b28f47cc-eda7-4961-b6b3-569753386249
# ╟─31633123-0249-4d15-b6fe-59480d3038eb
# ╟─4b9f3ab7-bb96-46f4-8af5-145f92f82d93
# ╟─6b449c6c-249e-4193-96ea-caccee683de0
# ╟─438726e5-f9a1-4bf7-abda-e5bb0eb30c39
# ╟─4d00dfcc-7b01-4335-95ba-0b31fa0e62ad
# ╟─f475a176-c2b0-4a22-8975-f5d2f54b6530
# ╟─cc14f0a2-d0bc-40fa-83fa-b99e62351282
# ╟─0a5bec4a-0e65-4753-a1e8-f7b3c6a061df
# ╠═2e6b7c33-c6b2-4fa0-9d71-acf7cb818b9b
# ╠═abe1f077-d211-4488-a10c-ec2ca5aac328
# ╠═315db1e4-c730-46bc-8f5b-03cdfe5467f9
# ╠═da3cde1e-d11a-4ce5-b134-347c3baba11d
# ╟─20836990-b332-478a-b99b-6f4ef4659392
# ╠═d9635e75-6e5c-41d2-b906-5025b58f9d0f
# ╠═e2ccff6e-6791-4338-8ec2-eef47f388bb1
# ╠═48923864-c40c-45ca-907e-2c0c03587f2c
# ╟─a1c173d7-f5ac-4eb3-96c7-c2f020cdf3d5
# ╟─03f0034a-9acf-4745-a927-51fc2554a7e4
# ╟─7af6f5ed-178d-4b90-9226-02be6b13ed5a
# ╟─10c75d09-3966-4511-83f8-e8365a2a82da
# ╟─3d6df2fa-35e5-4a02-b53f-30936f404b3f
# ╟─bf3584aa-2a17-43ef-95aa-035aa9505dd2
# ╠═d85e292e-5b13-4b51-9cb9-7c5161ba70a6
# ╠═c3cd5a5f-6445-4df3-b492-0d22e641f37c
# ╟─fbe8691b-6d71-4cba-90e4-5de63421f634
# ╟─5062690c-96b9-450a-9927-6a6707dfc511
# ╟─862026e9-ebe6-4f2e-8832-086bbba8db17
# ╟─8f894492-260e-4ab0-87b6-c02216a631e6
# ╟─c80256a7-be4f-4407-b0bf-7a13415482ad
# ╟─ba274806-6e16-447d-8c70-259787941495
# ╟─bc0073ab-fc41-4333-aecc-41501d89f15b
# ╠═1f3de2ad-65c5-4aaf-9c12-623de2257619
# ╟─b6123560-90fd-4cd5-83ff-f73234d8a897
# ╠═bcd35714-d664-4347-af27-4bdf131bad89
# ╟─4a474bb7-c932-4cbb-8442-2c0972a7da6c
# ╟─cacab854-8b62-4e45-bc88-85038461e667
# ╟─44100481-4e66-4b38-8262-87e337148bfc
# ╠═b525e0c8-e673-448d-8143-2a9a8be342f5
# ╟─9c9c5f0a-4079-4848-a822-ea9dcc460660
# ╟─560fa6a4-ac3a-43ae-931e-6699294b304a
# ╟─33a36f03-959d-4921-a476-68a75234f47c
# ╠═6630a9a0-2ec9-4c18-b9eb-e263ddc5d18c
# ╟─7cdc5c62-ddae-41fe-9ea2-aba25ac0ac3f
# ╠═1fa542f3-1e0e-41fc-ab09-e7eb0bd22483
# ╟─88e61bb4-fd6e-4363-be94-4166a7a39983
# ╟─1442dda6-f5a9-4a23-9075-39a9c7fcb899
# ╠═a566cd6b-19b6-4cfb-80e3-c74ed58705ba
# ╠═7f83988f-6e2c-4d90-899f-b4f5cdb1de48
# ╠═006fc67d-c3f9-46d3-b631-3002d9e50dd6
# ╠═f4e54d48-e2b6-45e7-b672-279cc3b2a3f0
# ╟─fd9c8373-90f9-4c1a-8c85-5e280311d381
# ╠═e820833d-db94-4a74-a637-5c3356b07906
# ╟─d746e585-a734-4fea-a534-ab366c12a87f
# ╠═2776aeba-4d0f-49c9-8395-d0f7242f2429
# ╟─a81603a0-34ee-4a9e-a8f8-7994c4d09cee
# ╠═7356e02e-7445-439d-a386-0b244541a443
# ╟─116bac12-7406-4f6d-9dab-ef4a75a98495
# ╠═ab796133-dd92-4535-ab8a-7ebc8875eb45
# ╟─8b3c3da4-0ab2-4294-a6f6-84470669a5d9
# ╠═91a7b6c3-17aa-43cf-93aa-4ecc5f5019dc
# ╟─7fa7d6f4-87ac-4e7b-b09f-588800c97664
# ╟─f430c6c9-914f-4db9-962a-012871a91a71
# ╟─cb5f26a2-cca2-4450-ae84-3cebd702a086
# ╠═61e6b0a3-a344-4fc6-b77c-36ef7cd138cd
# ╟─b45914c8-766b-4509-a6e6-92b093fa83b8
# ╠═c308859b-7f95-461b-b9d8-98249aa92111
# ╠═6d61cc94-9578-4232-848f-8a74ec42daae
# ╟─a0df5198-62e1-47bb-86dc-82fb501e24eb
# ╟─2d28b4af-0302-4dc4-9462-1ac6a083375f
# ╠═fbfdf045-e627-442e-8ecf-81e9c8007679
# ╟─c03d9058-25f0-49a1-9283-9d7d7492afd2
# ╟─38d20b01-e6a2-46fa-8d92-a1725565a7d8
# ╠═b2277d3b-7bc3-42ec-a685-bf45c4285caf
# ╟─798635bb-baf7-4069-8f40-a80f04d372ab
# ╟─b762a7f7-0a84-47e8-9425-f8982665ab7c
# ╠═d1eef08b-60b5-4475-bb48-d8e8cb52235f
# ╟─86c8efc2-970a-45a7-bc5e-10010cb39086
# ╠═dda1399e-d232-478d-9a38-6891430b8755
# ╠═ca515bd9-6ff9-4642-b0ac-f7cfd522e7f6
# ╟─59766450-1f4d-451a-9fe9-bca26596d955
# ╟─75871e7e-2834-4e81-940b-9dd063733e1e
# ╠═ec0ba4b3-3af4-464e-916e-e34df1605c8e
# ╠═70f0a00c-86f5-4a03-8b95-1333afba30e7
# ╠═76b7abfb-e95a-4cf8-9ac1-8354882845cc
# ╠═752297a5-6b66-4d96-931c-82352f9f0a35
# ╟─bf5c80bf-687f-4ade-afe5-6927c1733f64
# ╠═36f54330-4d8a-4f5e-ac64-aab02468ead0
# ╟─fbef62e4-404b-4859-9a9f-92b248395d7c
# ╟─de341cf8-822d-43ca-ba10-cc350afe8509
# ╠═0ee1882f-f9b2-4292-8b92-adc2eb2edb0e
# ╟─193e034f-1278-436f-b534-defc870cd36b
# ╠═bee67ec3-98b8-41b9-895c-7d2db4cebfab
# ╟─2705f545-7c2c-446e-a625-97908a04fefe
# ╠═9bdd4ce4-e9b9-4cc1-8c5d-fbc4c7a6f74a
# ╠═6ee6d7d0-9381-4413-a361-e836ac5240de
# ╠═834055ef-0bdc-4b4b-9c3f-ce7f43864ef7
# ╠═4b13b020-0dd2-45ff-adb5-d67cdd3a77f6
# ╠═9305f6ca-bec7-4002-af6b-f4142cc78d91
# ╟─8bdb4ac6-164f-4b3b-9f07-3bb5545bd132
# ╟─c9a83977-bb0a-4f58-940a-1647b2864aec
# ╟─9601eb2d-de36-436f-91c0-906dcd2921f8
# ╠═565ac829-0b95-449f-b787-233a8d6b935a
# ╠═d0e96989-c165-48d6-ba4b-4eab05fcb638
# ╟─993f8193-7488-415d-85d5-9a7a83f6bf71
# ╟─0358288e-be4e-46c2-ac4c-16ace6f50187
# ╟─2fb6e491-be69-44e8-ae2d-9cb13ec0b66f
# ╠═bd3ad49f-f076-46de-a159-b7cbffabe3dc
# ╠═7153065a-6e9e-4a03-9369-fcf63f8c238e
# ╠═8a581882-c97d-4a3b-873a-212024a529a9
# ╠═062f756b-6640-4928-9216-c54316503944
# ╟─2394cac9-3349-4684-9f08-506e4fe77a0d
# ╠═f6125f11-8719-4c10-be91-3fe981e2d921
# ╠═326b3355-7941-403b-bf1e-3031f585f666
# ╟─5616d294-892a-40bc-a35f-35e9e0ee55e2
# ╟─e4112acf-af6b-4cd7-be24-cff2ed77200d
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
# ╟─f0fa43fc-6221-469d-accc-fff87b005a17
# ╠═17083d16-6a9c-47e8-99f5-099067210029
# ╠═2e6c8ff9-4710-410d-b7e9-80563cc2af21
# ╠═cb992290-4bc2-4ebb-be3c-dc5d513ee5ef
# ╟─214ceb34-7e31-4c89-a328-a492244fd4cf
# ╠═909104f3-44c2-44ae-8186-11fd74b3ba4e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
