### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ f6125f11-8719-4c10-be91-3fe981e2d921
using PlutoUI, PlutoPlotly, Statistics ,StatsBase

# ╔═╡ b5479c7a-9140-11ed-257a-b342885b47fa
md"""
# 12.1 The λ-return

$\begin{flalign}
G_{t:t+n} \hspace{2mm}&  \dot = \hspace{2mm} R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n \hat v(S_{t+n}, \mathbf{w}_{t+n-1}) \tag{12.1}\\
G_t^\lambda \hspace{2mm}&  \dot = \hspace{2mm} (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} G_{t:t+n} \tag{12.2}\\

&\text{We can rewrite (12.2) as follows:}\\
G_t^\lambda \hspace{2mm}&  \dot = \hspace{2mm} (1 - \lambda) \sum_{n=1}^{T-t+1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t+1}G_t \tag{12.3}\\
\end{flalign}$

"""

# ╔═╡ dccc9b45-b711-44e8-8788-93de05f26543
md"""
> *Exercise 12.1* Just as the return can be written recursively in terms of the first reward and itself one-step later (3.9), so can the λ-return. Derive the analogous recursive relationship from (12.2) and (12.1).

Revisiting (3.9): $G_t = R_{t+1} + \gamma G_{t+1}$.  We are looking for an equation for $G_t^\lambda$ of a similar form, i.e. $G_t^\lambda = \cdots + \gamma G_{t+1}^\lambda$

Using (12.1):

$\begin{flalign}
G_{t:t+n} \hspace{2mm}&  \dot = \hspace{2mm} R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n \hat v(S_{t+n}, \mathbf{w}_{t+n-1}) \\
G_{t+1:t+n} &= R_{t+2} + \gamma R_{t+3} +\cdots+\gamma^{n-2}R_{t+n} + \gamma^{n-1} \hat v(S_{t+n}, \mathbf{w}_{t+n-1})\\
\therefore \\
G_{t:t+n} &= R_{t+1} + \gamma G_{t+1:t+n}
\end{flalign}$

Using (12.2): 

$\begin{flalign}
G_{t}^\lambda &= (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} G_{t:t+n} \\
&= (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} (R_{t+1} + \gamma G_{t+1:t+n}) \\
&= R_{t+1}(1-\lambda)\sum_{n=0}^\infty \lambda^n + \gamma(1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} G_{t+1:t+n}\\
&= R_{t+1}\frac{1-\lambda}{1-\lambda} + \gamma(1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} G_{t+1:t+n}\\
&= R_{t+1} + \gamma(1-\lambda)\left [ G_{t+1:t+1} + \lambda G_{t+1:t+2} + \lambda^2 G_{t+1:t+3} + \cdots \right ]\\
&= R_{t+1} + \gamma(1-\lambda)G_{t+1:t+1} + \gamma \lambda (1-\lambda)  \left [ G_{t+1:t+2} + \lambda G_{t+1:t+3} + \cdots \right ]\\

&\text{Note that also} \\ 
G_{t+1}^\lambda &= (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} G_{t+1:t+n+1} \\
&= (1-\lambda)(G_{t+1:t+2} + \lambda G_{t+1:t+3} + \lambda^2 G_{t+1:t+4} + \cdots) \\

&\text{So we can replace it in the above expression } \therefore \\

G_{t}^\lambda &= R_{t+1} + \gamma(1-\lambda)G_{t+1:t+1} + \gamma \lambda G_{t+1}^\lambda \\

 &= R_{t+1} + \gamma \left [ (1-\lambda)\hat v(S_{t+1}, \mathbf{w}_t) + \lambda G_{t+1}^\lambda \right ]\\

\end{flalign}$
From this expression it is clear that for $\lambda = 1$ we simply get $R_{t+1} + \gamma R_{t+2} + \cdots$ which is simply the monte carlo return.  For $\lambda = 0$, we get $R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w}_t)$ which is the 1 step TD return.

"""

# ╔═╡ 9d131051-eeee-4aba-8f78-9ddff9babab4
function plot_hl()
	τ(λ) = (log(λ) - log(2)) / log(λ)
	λs = 0:0.01:1
	plot(λs, τ.(λs), Layout(xaxis_title = "λ", yaxis_title = "τ_λ"))
end

# ╔═╡ 752a80ea-1da6-49ef-91ef-a03c590b825d
md"""
> *Exercise 12.2* The parameter λ characterizes how fast the exponential weighting in Figure 12.2 falls off, and thus how far into the future the λ-return algorithm looks in determining its update. But a rate factor such as λ is sometimes an awkward way of characterizing the speed of the decay. For some purposes it is better to specify a time constant, or half-life. What is the equation relating λ and the half-life, $\tau_\lambda$, the time by which the weighting sequence will have fallen to half of its initial value?

The initial weight is $\lambda^0=1$, so the question is at what n will the weight value be $\frac{1}{2}$.  That will occur when:

$\begin{flalign}
\lambda^{n_\tau-1} &= \frac{1}{2}\\
(n_\tau-1) \log{\lambda} &= \log{\frac{1}{2}} \\
n_\tau-1 &= \frac{\log{1} - \log{2}}{\log{\lambda}} \\
n_\tau &= \frac{\log{\lambda} - \log{2}}{\log{\lambda}}
\end{flalign}$

From the plot we can see that the halflife approaches infinity as λ approaches 1 which we expect from the monte-carlo return.
$(plot_hl())
"""

# ╔═╡ 57cf5ae7-d4dd-47e8-8090-c04fb39e0763
md"""
# 12.2 TD(λ)
"""

# ╔═╡ 34dda4bf-f78f-4c83-ba10-9b206d2fbcb8
md"""
$\begin{flalign}
& \mathbf{z}_{-1} \dot = \mathbf{0}\\
& \mathbf{z_t} \dot = \gamma \lambda \mathbf{z}_{t-1} + \nabla \hat v(S_t, \mathbf{w_{t}}),  \hspace{5 mm} 0 \leq t \leq T-1 \tag{12.5} \\
& \delta_t \dot = R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w}_t) - \hat v(S_t, \mathbf{w}_t) \tag{12.6} \\
& \mathbf{w}_{t+1} \dot = \mathbf{w}_t + \alpha \delta_t \mathbf{z}_t \tag{12.7}
\end{flalign}$
"""

# ╔═╡ 6f5168dc-f1f3-4533-a59e-bb85895f3b13
md"""
## Semi-gradient TD(λ) for estimating $\hat v \approx v_\pi$
"""

# ╔═╡ bded7e14-0c02-4e55-b75c-cbb2c01c4e5d
function semi_gradient_TDλ(π, v̂, ∇v̂, w, states, sterm, step, λ, γ, α, numepisodes, s_init, Vtrue)
	rmserr() = sqrt(mean((Vtrue[s] - v̂(s, w))^2 for s in states))
	rmserrs = zeros(numepisodes)
	for ep in 1:numepisodes
		s = s_init()
		z = zeros(length(w))
		function update!(s)
			s == sterm && return
			a = π(s)
			(s′, r) = step(s, a)
			z .= (γ*λ .* z) .+ ∇v̂(s, w)
			δ = r + γ*v̂(s′, w) - v̂(s, w)
			w .+= α*δ .* z 
			update!(s′)
		end
		update!(s)
		rmserrs[ep] = rmserr()
	end
	return w, rmserrs
end		

# ╔═╡ e597a042-9c03-4d49-a48f-6dff39283c54
md"""
> *Exercise 12.3* Some insight into how TD(λ) can closely approximate the on-line 
λ-return algorithm can be gained by seeing that the latter’s error term (in brackets in (12.4)) can be written as the sum of TD errors (12.6) for a single fixed w. Show this, following the pattern of (6.6), and using the recursive relationship for the λ-return you obtained in Exercise 12.1.

The error term at step t is: $G_t^\lambda - \hat v(S_t, \mathbf{w_t})$

The TD error at step t is given by : $\delta_t \dot = R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w_t}) - \hat v(S_t, \mathbf{w_t})$

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
> *Exercise 12.4* Use your result from the preceding exercise to show that, if the weight updates over an episode were computed on each step but not actually used to change the weights (w remained fixed), then the sum of TD(λ)’s weight updates would be the same as the sum of the off-line λ-return algorithm’s updates.

The TD(λ) updates are given by: $\mathbf{w_{t+1}} \dot = \mathbf{w_t} + \alpha \delta_t \mathbf{z_t}$ with $\mathbf{z_t} = \gamma \lambda \mathbf{z_{t-1}} + \nabla \hat v(S_{t}, \mathbf{w_{t}})$.  Let's write down all of the updates that will occur from t = 0 assuming the weights themselves are held constant the entire episode.

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

$\mathbf{w_{t+1}} \dot = \mathbf{w_t} + \alpha \left [ G_t^\lambda - \hat v(S_t, \mathbf{w_t}) \right ] \nabla \hat v(S_t, \mathbf{w_t})$

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
\delta_t \text{ coefficient} &= \alpha \sum_{n = 0}^t (\gamma \lambda)^{t - n} \nabla \hat v(S_{t}, \mathbf{w})
\end{flalign}$

But this coefficient is the same as we got previously for the TD(λ) weight updates, so we've shown that if weight updates are delayed unti the end of an episode both methods will perform exactly the same weight updates.
"""

# ╔═╡ 27f535a4-2245-45aa-aefa-4c0fc6bb218d
md"""
# 12.3 n-step Truncated λ-return Methods

Define the *truncated λ-return* for time t, given data only up to some later horizon, h, as 

$\begin{flalign}
G_{t:h}^\lambda \dot = (1-\lambda) \sum_{n=1}^{h-t-1} \lambda ^{n-1} G_{t:t+n} + \lambda^{h-t-1} G_{t:h}, \hspace{5mm} 0 \leq t < h \leq T \tag{12.9}
\end{flalign}$

The weight updates for Truncated TD(λ) or TTD(λ) is given by:

$\mathbf{w}_{t+n} \dot = \mathbf{w}_{t+n-1} + \alpha \left [ G_{t:t+n}^\lambda - \hat v(S_t, \mathbf{w}_{t+n-1}) \right ] \nabla \hat v (S_t, \mathbf{w}_{t+n-1})$

where the maximum number of steps in the future to consider returns for is n.  Much as in *n*-step TD methods, no updates are made on the first n-1 steps of each episode, and n-1 additional updates are made upon termination.  Efficient imlementation relies on the fact that the *k*-step λ-return can be written exactly as 

$\begin{flalign}
G_{t:t+k}^\lambda = \hat v(S_t, \mathbf{w}_{t-1}) + \sum_{i=t}^{t+k-1} (\gamma \lambda)^{i-t} \delta_i ^\prime \tag{12.10}
\end{flalign}$

where 

$\delta_i ^\prime \dot = R_{t+1} + \gamma \hat v (S_{t+1}, \mathbf{w}_t) - \hat v(S_t, \mathbf{w}_{t-1})$

"""

# ╔═╡ e1e9f2eb-4751-4f5c-aa4d-a0cf75e193b2
md"""
> *Exercise 12.5* Several times in this book (often in exercises) we have established that returns can be written as sums of TD errors if the value function is held constant.  Why is (12.10) another instance of this?  Prove (12.10).

To prove (12.10) let's return to the definition of $G_{t:k}^\lambda$ given in (12.9) and compare it to (12.10)

$\begin{flalign}
G_{t:h}^\lambda &\dot = (1-\lambda) \sum_{n=1}^{h-t-1} \lambda ^{n-1} G_{t:t+n} + \lambda^{h-t-1} G_{t:h}, \hspace{5mm} 0 \leq t < h \leq T \tag{12.9} \\
G_{t:k}^\lambda &= \hat v(S_t, \mathbf{w}_{t-1}) + \sum_{i=t}^{t+k-1} (\gamma \lambda)^{i-t} \delta_i ^\prime \tag{12.10}\\
\delta_t & \dot = R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf{w}_t) - \hat v(S_t, \mathbf{w}_{t-1})
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

# ╔═╡ 2c664592-eddf-4438-b153-075282f6e491
md"""
# 12.4 Redoing Updates: Online λ-return Algorithm
# 12.5 True Online TD(λ)
The online λ-return algorithm just presented is currently the best performing temporal-difference algorithm. It is an ideal which online TD(λ) only approximates.  (why is this the case?  I thought TD(λ) was equivalent to the full λ return, they mentioned in figures that at higher learning rates it can be unstable though.  True online TD(λ) doesn't have that problem.  In the plot there isn't even a horizon anymore but this was truncated.  So what happened to the cutoff point?)  So at each step in the episode, the target is the n-step λ return for that step so there is no selection of the horizon.  The largest possible horizon for every previous state is always being used in the update target.

In the linear case for which $\hat v(s_\mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s)$, then we arrive at the true online TD(λ) algorithm:

$\begin{flalign}
\mathbf{w}_{t+1} & \dot = \mathbf{w}_t + \alpha \delta_t \mathbf{z}_t + \alpha (\mathbf{w}_t^\top \mathbf{x}_t - \mathbf{w}_{t-1}^\top \mathbf{x}+t)(\mathbf{z}_t - \mathbf{x}_t) \\
\mathbf{x}_t & \dot = \mathbf{x}(S_t) \\
\mathbf{z}_t & \dot = \gamma \lambda \mathbf{z}_{t-1} + (1 - \alpha\gamma\lambda \mathbf{z}_{t-1}^\top \mathbf{x}_t)\mathbf{x}_t \tag{12.11}

\end{flalign}$
"""

# ╔═╡ 5324724c-93d1-4186-9dcf-55afd410aa72
function true_online_TDλ(π, x, w, states, sterm, step, λ, γ, α, numepisodes, s_init, Vtrue)
	rmserr() = sqrt(mean((Vtrue[s] - w'*x(s))^2 for s in states))
	rmserrs = zeros(numepisodes)
	for ep in 1:numepisodes
		s = s_init()
		z = zeros(length(w))
		function update!(s, v_old = 0.0)
			s == sterm && return
			a = π(s)
			(s′, r) = step(s, a)
			v = w' * x(s)
			v′ = w' * x(s′)
			δ = r + γ*v′-v
			z .= (γ*λ .* z) .+ (1-α*γ*λ*(z'*x(s)))*x(s)
			w .+= α*(δ + v - v_old) .* z .- α*(v - v_old) 
			update!(s′, v′)
		end
		update!(s)
		rmserrs[ep] = rmserr()
	end
	return w, rmserrs
end	

# ╔═╡ b36896b1-6802-48e1-8cd3-f08bf3b99e3e
md"""
# 12.6 Dutch Traces in Monte Carlo Learning

It can be shown that the linear MC algorithm can be used to drive an equivalent yet computationally cheapter backward-view algorithm using dutch traces.  This equivalence gives some flavor of teh proof of equivalence of true online TD(λ) and the online λ-return algorithm, but is much simpler.

The linear version of gradient Monte Carlo prediction algorithm makes the following sequence updates, one for each time step of the episode:

$\begin{flalign}
\mathbf{w}_{t+1} & \dot = \mathbf{w}_t + \alpha \left [ G - \mathbf{w}_t^\top \mathbf{x}_t \right ] \mathbf{x}_t, \hspace{4 mm} 0 \leq t < T \tag{12.13} \\
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
# 12.7 Sarsa(λ)
"""

# ╔═╡ 72895891-9212-4722-b2a1-0e13c30a8ecf
function sarsaλ_linear(ℱ, w, states, actions, sterm, step, λ, γ, α, numepisodes, s_init, ϵ, usedutch = false)
	function ϵ_greedy(s, ϵ)
		rand() < ϵ && return rand(actions)
		qa = [sum(w[i] for i in ℱ(s, a)) for a in actions]
		(maxq, ind) = findmax(qa)
		inds = findall(q -> q == maxq, qa)
		isempty(inds) && return rand(actions)
		rand(actions[inds])
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

# ╔═╡ 07245a98-cab2-4b0c-a17a-4eaaa8a30703
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

	w, steps, π = f(ℱ, w_init(), states, actions, goal, step, λ, 1.0, α, numepisodes, s_init, ϵ, usedutch)

	# run_episode(π_rand, (1, 1))
	(w, steps, π, π_rand, run_episode, step, states)
end

# ╔═╡ f5c3d5a4-7fe8-420e-af0a-4318b1eeda2c
function eval_grid(lmax, wmax, goal; ϵ = 0.1, f = sarsaλ_linear, usedutch=false,  αlist = [0.05, 0.1, 0.2, 0.4, 0.8], λlist = [0.0, 0.4, 0.8, 0.9, 0.99])
	function runtrial(α, λ)
		(w, steps, π, π_rand, makepath, step, states) = gridworld_sarsa(lmax, wmax, goal, λ, ϵ, α, 100, usedutch, f = f)
		sum(steps[51:end])/50
	end


	runtrials(α, λ) = mean(runtrial(α, λ) for _ in 1:100)

	results = [[runtrials(α, λ) for α in αlist] for λ in λlist]
	(results = results, αlist=αlist, λlist=λlist)
end

# ╔═╡ 4bd9d7a4-979d-492f-b863-8359864004ea
function plot_grid(lmax, wmax, goal; ϵ = 0.1, f = sarsaλ_linear, usedutch=false, αlist = [0.05, 0.1, 0.2, 0.4, 0.8], λlist = [0.0, 0.4, 0.8, 0.9, 0.99])
	(results, αlist, λlist) = eval_grid(lmax, wmax, goal, ϵ=ϵ, f = f, usedutch=usedutch, αlist = αlist, λlist = λlist)
	traces = [begin
		scatter(x = αlist, y = results[i], name = "λ = $(λlist[i])")
	end
	for i in eachindex(results)]
	plot(traces, Layout(yaxis_title = "Steps", xaxis_title = "α", title = "Mean Steps For 100 Episodes"))
end

# ╔═╡ 32832503-d48b-48bb-be7b-cf2cb6855a57
plot_grid(10, 10, (5, 8), usedutch=true)

# ╔═╡ fbe8691b-6d71-4cba-90e4-5de63421f634
md"""
> *Exercise 12.6* Modify the pseudocode for Sarsa(λ) to use dutch traces (12.11) without the other distinctive features of a true online algorithm.  Assume linear function approximation and binary features.

See the above function `sarsaλ_linear`.  In the step where $z_i$ is updated an additional term is subtracted in the case of using dutch traces which matches equation (12.11)
"""

# ╔═╡ b1d56779-9a06-4b25-9a1b-09a12923e646
function true_online_sarsaλ_binary(ℱ, w, states, actions, sterm, step, λ, γ, α, numepisodes, s_init, ϵ, usedutch=true)
	function ϵ_greedy(s, ϵ)
		rand() < ϵ && return rand(actions)
		qa = [sum(w[i] for i in ℱ(s, a)) for a in actions]
		(maxq, ind) = findmax(qa)
		inds = findall(q -> q == maxq, qa)
		isempty(inds) && return rand(actions)
		rand(actions[inds])
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

# ╔═╡ f3c3f934-6601-4383-8204-55d04c973881
plot_grid(10, 10, (5, 8), f = true_online_sarsaλ_binary, λlist = [0.0, 0.01, 0.02], αlist = [0.00001, 0.0001, 0.001, 0.002, 0.004])

# ╔═╡ 862026e9-ebe6-4f2e-8832-086bbba8db17
md"""
# 12.8 Variable λ and γ
"""

# ╔═╡ 8f894492-260e-4ab0-87b6-c02216a631e6
md"""
$\begin{flalign}
G_t &= \sum_{k=t}^\infty \left ( \prod_{i=t+1}^k \gamma_i \right ) R_{k+1} \tag{12.17}\\
G_t^{\lambda_s} &\dot = R_{t+1} + \gamma_{t+1} \left ( (1-\lambda_{t+1}) \hat v (S_{t+1}, \mathbf{w}_t) + \lambda_{t+1}G_{t+1}^{\lambda_s} \right ) \tag{12.18}\\
G_t^{\lambda_a} &\dot = R_{t+1} + \gamma_{t+1} \left ( (1-\lambda_{t+1}) \hat q (S_{t+1}, A_{t+1}, \mathbf{w}_t) + \lambda_{t+1}G_{t+1}^{\lambda_a} \right ) \tag{12.19}\\
G_t^{\lambda_a} &\dot = R_{t+1} + \gamma_{t+1} \left ( (1-\lambda_{t+1}) \overline V_t(S_{t+1}) + \lambda_{t+1}G_{t+1}^{\lambda_a} \right ) \tag{12.19}\\
\overline V_t(s) & \dot = \sum_a \pi(a|s)\hat q(s, a, \mathbf{w}_t) \tag{12.21}
\end{flalign}$
"""

# ╔═╡ c80256a7-be4f-4407-b0bf-7a13415482ad
md"""
> *Exercise 12.7* Generalize the three recursive equations above to their truncated versions, defining $G_{t:h}^{\lambda_s}$ and $G_{t:h}^{\lambda_a}$

Starting with (12.18) we want to get a truncated version $G_{t:h}^{\lambda_s}$.  We can also use as a model the truncated λ-return

$\begin{flalign}
G_t^{\lambda_s} &\dot = R_{t+1} + \gamma_{t+1} \left ( (1-\lambda_{t+1}) \hat v (S_{t+1}, \mathbf{w}_t) + \lambda_{t+1}G_{t+1}^{\lambda_s} \right )\\
G_{t:h}^\lambda &\dot = (1-\lambda) \sum_{n=1}^{h-t-1}\lambda^{n-1} G_{t:t+n} + \lambda^{h-t-1}G_{t:h}\\
G_{t:h}^{\lambda_s} &\dot = (1-\lambda) \sum_{n=1}^{h-t-1} \left ( \prod_{i=t}^{t+n-1} \lambda_i \right ) G_{t:t+n} + \left ( \prod_{i=h-t-1}^{\infty} \lambda_i \right ) G_{t:h}
\end{flalign}$

"""

# ╔═╡ e6782d51-175c-4de7-9c75-1fc3f75a92f0
md"""
# Chapter 7 Code For Random Walk Comparison
"""

# ╔═╡ 013c2268-6ab8-441a-9fb4-5118dc3ae18a
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
		τ = 0
		t = 0
		while τ != T - 1
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
		end
		rmserrs[ep] = rmserr()
	end
	return V, rmserrs
end

# ╔═╡ 44a16c0a-9d0d-4e9b-9ae5-aef791c4f544
begin
	abstract type LinearMoves end
	struct Left <: LinearMoves end
	struct Right <: LinearMoves end
end

# ╔═╡ 13756b5d-b496-45fa-875b-7f6ee6468dcf
#create a random walk mdp of length n where the left terminal state produces a reward of -1 and the right a reward of 1
function create_random_walk(n)
	states = Tuple(1:n)
	sterm = 0
	actions = (:left, :right)
	move(s, ::Left) = s - 1
	move(s, ::Right) = s + 1
	function step(s0, action)
		s = move(s0, action)
		(s == 0) && return (sterm, -1.0)
		(s > n) && return (sterm, 1.0)
		return (s, 0.0)
	end
	(states, sterm, step)
end

# ╔═╡ 5f94ada5-5aa5-4c4a-8ea7-578931e04b6b
create_random_walk(19)

# ╔═╡ f7ac4e92-64b0-4bdb-ab00-9edbbfdd2898
function random_walk_TDλ(nstates = 19; numepisodes = 10, nruns = 10)
	#estimate random policy
	π(s) = rand([Left(), Right()])

	c = (nstates + 1)/2
	Vtrue = [(s-c)/c for s in 1:nstates]

	maxerr = sqrt(mean(Vtrue .^2))

	(states, sterm, step) = create_random_walk(nstates)

	make_w() = zeros(nstates) #using weight vector that keeps a value for each state
	v̂(s, w) = s == sterm ? 0.0 : w[s] #take weight value for that state
	∇v̂(s, w) = [i == s ? 1.0 : 0.0 for i in eachindex(w)]

	s_init() = rand(1:nstates)
	
	function get_λ_error(α, λ)
		w, rmserrs = semi_gradient_TDλ(π, v̂, ∇v̂, make_w(), states, sterm, step, λ, 1.0, α, numepisodes, s_init, Vtrue)
		mean(rmserrs)
	end

	α_vec = 1.1 .^ (-30:0)
	λ_vec = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0]
	rmsvecs = [[mean(get_λ_error(α, λ) for _ in 1:nruns) for α in α_vec] for λ in λ_vec]

	traces = [scatter(x = α_vec, y = rmsvecs[i], name = "λ=$(λ_vec[i])") for i in eachindex(rmsvecs)]
	ymin = minimum(minimum(filter(!isnan, v)) for v in rmsvecs) * 0.9
	ymax = maxerr
	plot(traces, Layout(yaxis_title="RMS Error for $nstates State Chain with Random Policy Over the First $numepisodes Episodes", title = "TD(λ) Estimator", xaxis_title = "α", yaxis_range = [ymin, ymax]))
end

# ╔═╡ 5cbe472f-4d96-483f-975f-07d41d809dc9
random_walk_TDλ(5, nruns = 100)

# ╔═╡ 9fc1b81a-a1c1-43ea-adb9-af0e8b3abaa9
random_walk_TDλ(nruns = 100)

# ╔═╡ 2336e059-34a5-4c81-be53-fa3f66733bd9
function random_walk_true_onlineTDλ(nstates = 19; numepisodes = 10, nruns = 10)
	#estimate random policy
	π(s) = rand([Left(), Right()])

	c = (nstates + 1)/2
	Vtrue = [(s-c)/c for s in 1:nstates]

	maxerr = sqrt(mean(Vtrue .^2))

	(states, sterm, step) = create_random_walk(nstates)

	make_w() = zeros(nstates) #using weight vector that keeps a value for each state
	function x(s)
		s == sterm && return zeros(nstates)
		[i == s ? 1.0 : 0.0 for i in 1:nstates]
	end

	s_init() = rand(1:nstates)
	
	function get_λ_error(α, λ)
		w, rmserrs = true_online_TDλ(π, x, make_w(), states, sterm, step, λ, 1.0, α, numepisodes, s_init, Vtrue)
		mean(rmserrs)
	end

	α_vec = 1.1 .^ (-30:0)
	λ_vec = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0]
	rmsvecs = [[mean(get_λ_error(α, λ) for _ in 1:nruns) for α in α_vec] for λ in λ_vec]

	traces = [scatter(x = α_vec, y = rmsvecs[i], name = "λ=$(λ_vec[i])") for i in eachindex(rmsvecs)]
	ymin = minimum(minimum(filter(!isnan, v)) for v in rmsvecs) * 0.9
	ymax = maxerr
	plot(traces, Layout(yaxis_title="RMS Error for $nstates State Chain with Random Policy Over the First $numepisodes Episodes", title = "True online TD(λ) Estimator", xaxis_title = "α", yaxis_range = [ymin, ymax]))
end

# ╔═╡ 9123aa11-9187-4203-b671-d5f5feaf5813
random_walk_true_onlineTDλ(nruns = 100)

# ╔═╡ 2cafed7d-22c6-420f-9c8e-8ae734bfbad2
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

# ╔═╡ a3484638-ae83-4810-9226-0a25b3fc58dc
function optimize_n_randomwalk(nstates; estimator = n_step_TD_Vest, v0=0.0, nruns = 10)
	(rmsvecs, α_vec, n_vec, ymax) = nsteptd_error_random_walk(nstates, estimator, v0=v0, nruns = nruns)
	traces = [scatter(x = α_vec, y = rmsvecs[i], name = "n=$(n_vec[i])") for i in eachindex(rmsvecs)]
	ymin = minimum(minimum(v) for v in rmsvecs) * 0.9
	plot(traces, Layout(title="RMS Error for $nstates State Chain with Random Policy Over the First 10 Episodes, n-step TD Estimator", xaxis_title = "α", yaxis_range = [ymin, ymax]))
end

# ╔═╡ f70fe1bd-f3ba-48c0-ba93-aa647224a8bf
walk19_plot1 = optimize_n_randomwalk(19, nruns = 100)

# ╔═╡ 773a0bed-4d14-4643-818f-02e9d93898eb
TableOfContents()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
PlutoPlotly = "~0.3.6"
PlutoUI = "~0.7.49"
StatsBase = "~0.33.21"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "fba562a8540152a8e4888628ad8394fbbc366253"

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
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "00a2cccc7f098ff3b66806862d275ca3db9e6e5a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.5.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

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
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

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
git-tree-sha1 = "946607f84feb96220f480e0422d3484c49c00239"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.19"

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

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

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
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "6466e524967496866901a78fca3f2e9ea445a559"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.2"

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
deps = ["AbstractPlutoDingetjes", "Colors", "Dates", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "PlotlyBase", "PlutoUI", "Reexport"]
git-tree-sha1 = "dec81dcd52748ffc59ce3582e709414ff78d947f"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.3.6"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eadad7b14cf046de6eb41f13c9275e5aa2711ab6"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.49"

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

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

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

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

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
git-tree-sha1 = "ac00576f90d8a259f2c9d823e91d1de3fd44d348"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.1"

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
# ╟─b5479c7a-9140-11ed-257a-b342885b47fa
# ╟─dccc9b45-b711-44e8-8788-93de05f26543
# ╟─752a80ea-1da6-49ef-91ef-a03c590b825d
# ╠═9d131051-eeee-4aba-8f78-9ddff9babab4
# ╟─57cf5ae7-d4dd-47e8-8090-c04fb39e0763
# ╟─34dda4bf-f78f-4c83-ba10-9b206d2fbcb8
# ╟─6f5168dc-f1f3-4533-a59e-bb85895f3b13
# ╠═bded7e14-0c02-4e55-b75c-cbb2c01c4e5d
# ╠═5f94ada5-5aa5-4c4a-8ea7-578931e04b6b
# ╠═f7ac4e92-64b0-4bdb-ab00-9edbbfdd2898
# ╠═5cbe472f-4d96-483f-975f-07d41d809dc9
# ╠═9fc1b81a-a1c1-43ea-adb9-af0e8b3abaa9
# ╠═f70fe1bd-f3ba-48c0-ba93-aa647224a8bf
# ╟─e597a042-9c03-4d49-a48f-6dff39283c54
# ╟─0c6ebdeb-77f4-44f0-9bf3-c539d54bcaec
# ╟─27f535a4-2245-45aa-aefa-4c0fc6bb218d
# ╟─e1e9f2eb-4751-4f5c-aa4d-a0cf75e193b2
# ╟─531263cf-274e-4a64-932f-821e8583a316
# ╟─0df08e27-18d3-4f2c-a7e1-75674418ba01
# ╟─c65dc168-9fa4-4e1b-af39-02f80c9ec0e3
# ╟─8d41a846-3a12-4e32-bc1a-50be12629eb2
# ╟─2c664592-eddf-4438-b153-075282f6e491
# ╠═5324724c-93d1-4186-9dcf-55afd410aa72
# ╠═2336e059-34a5-4c81-be53-fa3f66733bd9
# ╠═9123aa11-9187-4203-b671-d5f5feaf5813
# ╟─b36896b1-6802-48e1-8cd3-f08bf3b99e3e
# ╟─0086dc4a-e0ba-43f0-a721-296cd50e1a76
# ╠═72895891-9212-4722-b2a1-0e13c30a8ecf
# ╠═07245a98-cab2-4b0c-a17a-4eaaa8a30703
# ╠═f5c3d5a4-7fe8-420e-af0a-4318b1eeda2c
# ╠═4bd9d7a4-979d-492f-b863-8359864004ea
# ╠═32832503-d48b-48bb-be7b-cf2cb6855a57
# ╟─fbe8691b-6d71-4cba-90e4-5de63421f634
# ╠═b1d56779-9a06-4b25-9a1b-09a12923e646
# ╠═f3c3f934-6601-4383-8204-55d04c973881
# ╟─862026e9-ebe6-4f2e-8832-086bbba8db17
# ╟─8f894492-260e-4ab0-87b6-c02216a631e6
# ╟─c80256a7-be4f-4407-b0bf-7a13415482ad
# ╟─e6782d51-175c-4de7-9c75-1fc3f75a92f0
# ╠═013c2268-6ab8-441a-9fb4-5118dc3ae18a
# ╠═44a16c0a-9d0d-4e9b-9ae5-aef791c4f544
# ╠═13756b5d-b496-45fa-875b-7f6ee6468dcf
# ╠═2cafed7d-22c6-420f-9c8e-8ae734bfbad2
# ╠═a3484638-ae83-4810-9226-0a25b3fc58dc
# ╠═f6125f11-8719-4c10-be91-3fe981e2d921
# ╠═773a0bed-4d14-4643-818f-02e9d93898eb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
