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

# ╔═╡ 86d53794-2251-47d5-a45e-f1da53cd8ef5
begin
	using PlutoPlotly, PlutoUI
	TableOfContents()
end

# ╔═╡ 17f36458-139b-4f8b-aba9-d0dd586dd82c
md"""
# Chapter 3

# Finite Markov Decision Processes

## 3.1 The Agent-Environment Interface

> ### *Exercise 3.1*
> Devise three example tasks of your own that fit into the MDP framework, identifying for each its states, actions, and rewards. Make the three examples as *different* from each other as possible. The framework is abstract and flexible and can be applied in many different ways. Stretch its limits in some way in at least one of your examples.

- Example Task 1: Fill water bottle
  
  - Task Description: Consider a water dispenser that has a valve to let water flow into a bottle placed below it.
  - State space: The state space could be the total volume of the bottle being filled, the total volume of water currently in the bottle, and the total volume of water that enters the drain.
  - Action space: The action space in the simplest case could be whether the valve is open or closed with a constant flow rate of water occurring for an open valve and zero flow for a closed valve. The reward could be
  - Rewards: The reward could be $1 - \text{unfilled volume fraction} - (10 \times \text{overfilled volume fraction})$. This way the reward would be maximized when the bottle is completely filled and would be penalized for any overfilling that occurs. Since the overfilling negative reward could not be removed into the future it would accumulate forever discouraging this behavior. In contract, there will always be a benefit for the system to get arbitrarily close to being perfectly full without overfilling.
- Example Task 2: Automatic Vacuum Cleaner
  
  - Task Description: Consider a battery powered robotic vacuum with wheels that can drive around a room and clean the floor. The robot will have to decide which path to take through the room before it has to recharge in its charging dock. To simplify the problem consider a driving scheme where the robot can only move forward in a straight line and can stop and turn in place to one of 4 directions: N, S, E, W.
  - State space: Consider a simple vacuum that can only detect its own battery charge state as a percent of full, how much material it is collecting at each moment in time, and whether it has a clear path immediately in front of it in each of 4 possible directions.
  - Action space: The robot can select to drive forward, not drive forward, or turn to a new direction.
  - Rewards: It is critical that the robot not get stuck with 0 charge outside of its charging dock while still trying to collect as much material from the floor while it has available charge. So, the reward at each timestep could be $+1 \times \text{material collected} -1 \times \text{missing charge percentage}$. Depending on the units of material collected, these coefficients would have to be normalized to ensure that the robot doesn't sit in the charging dock indefinitely to avoid negative reward.
- Example Task 3: Firing Range Gun Aimer
  
  - Task Description: Consider a mechanism for holding a gun at a particular angle with the ability to pull the trigger and fire a round. The aim is to hit the target as close to the bullseye as possible. Consider an outdoor environment with wind and a target that can be placed at some arbitrary fixed distance away from the gun.
  - State space: The state space could be the vertical and horizontal angle of the gun relative to the straight line connecting the pivot point of the gun to the center of the target. In addition a wind speed sensor could detect the angle and speed of any airflow and a rangefinder can detect the distance to the target.
  - Action space: The agent can decide each of the two angles to position the gun restricted by some finite number of precise steps. If an angle is not being selected then the agent can decide whether to pull the trigger or wait.
  - Rewards: The agent could receive a large positive reward depending on how close the bullet hits to the bullseye and an infinitely negative reward for hitting too far away from the bullseye. Something like $r = \frac{1}{\max(0,\text{max distance} - \text{distance to bullseye})}+\frac{1}{\text{distance to bullseye}}$
"""

# ╔═╡ c5abf826-9ce8-4319-a2e5-6cf7fcc61400
md"""
> ### *Exercise 3.2*
> Is the MDP framework adequate to usefully represent *all* goal-directed learning tasks? Can you think of any clear exceptions?

If the environment depends heavily on the past history in terms of future rewards, but none of that information can be encoded into the current state, then the agent would be unable to learn the correct actions to take unless it had some internal memory of its own. But that would be akin to the agent having an internal state that varied over the course of a trajectory which is not part of the current framework. This could be solved by augmenting the environment state to contain whatever past information is necessary to specify the current state but that information may not always be accessible. In particular consider an environment with a person that has a particular action in mind that if repeated will cause large negative reward. The environment itself provides no record of which actions the agent has taken, so unless the agent saves that information itself, it would have no way of knowing.
"""

# ╔═╡ 85905a6e-1807-4b77-b313-dbadb8b898c8
md"""
> ### *Exercise 3.3*
> Consider the problem of driving. You could define the actions in terms of the accelerator, steering wheel, and brake, that is, where your body meets the machine. Or you could define them farther out—say, where the rubber meets the road, considering your actions to be tire torques. Or you could define them farther in—say, where your brain meets your body, the actions being muscle twitches to control your limbs. Or you could go to a really high level and say that your actions are your choices of *where* to drive. What is the right level, the right place to draw the line between agent and environment? On what basis is one location of the line to be preferred over another? Is there any fundamental reason for preferring one location over another, or is it a free choice?

If you have a system like a car, it already has a mechanism for translating the pedal position and steering wheel into forces on the tires. If instead, we tried to have the system directly control the torque on the wheel, it would still have to control that through the pedal and steering wheel and rely on some other learned or explicit mechanism for translating those desires. Using the action space that relates most closely to what can actually be controlled would be the least prone to errors in which the desired actions are not implemented accurately. If we already had a built navigation system for the car in question and the desired task involves choosing the optimal path to navigate between many locations, then it might be appropriate to have the action space in terms of *where* to drive. If the car only has simple controls like accelerator and steering wheel as described above, then even if the ultimate task is more complicated, the natural action space is still the controls we have access to. The agent may effectively learn an intermediate task of how to navigate to a particular city, but putting that in the action space would give the agent no obvious way of performing that action. If we consider a human driving, it would be natural to have the action space in terms of actions that a human would know how to perform such as pressing the accelerator a set amount. Since people already know how to control muscles with electrical impulses, it would be an unnecessary layer of complexity to have an agent learn how to directly control the muscles of a person.
"""

# ╔═╡ 090c50ed-6772-457a-afbb-cf2cde0e2ec4
md"""
> ### *Exercise 3.4*
> Give a table analogous to that in Example 3.3 but for $p(s',r|s,a)$. It should have columns for $$s, \space a, \space s', \space r$$ and $$p(s',r|s,a)$$, and a row for every 4-tuple for which $p(s',r|s,a)>0$

| $s$ | $a$ | $s'$ | $r$ | $p(s',r \vert s,a)$ |
| --- | --- | --- | --- | --- |
| high | search | high | $r_{search}$ | $\alpha$ |
| high | search | low | $r_{search}$ | $1-\alpha$ |
| low | search | low | $r_{search}$ | $\beta$ |
| low | search | high | -3  | $1-\beta$ |
| high | wait | high | $r_{wait}$ | 1   |
| low | wait | low | $r_{wait}$ | 1   |
| low | recharge | high | 0   | 1   |
"""

# ╔═╡ 768833aa-ceee-4fe8-958f-c00b778ec764
md"""
## 3.3 Returns and Episodes

> ### *Exercise 3.5*
> The equations in Section 3.1 are for the continuing case and need to be modified (very slightly) to apply to episodic tasks. Show that you know the modifications needed by giving the modified version of (3.3).

From Section 3.1 we have equation (3.3):

$\sum_{s' \in \mathcal{S}}\sum_{r \in \mathcal{R}}p(s',r|s,a)=1, \text{ for all } s\in\mathcal{S},a\in\mathcal{A}(s)$

In the episodic case there is an additional state outside of $\mathcal{S}$ called the terminal state, and the union of this with every other state is denoted $\mathcal{S}^+$. So equation (3.3) still applies, but only if we ensure that there is some non-zero probability of entering the terminal state.  The modified version of equation 3.3 is:

$\sum_{s^ \prime \in \mathcal{S}^+}\sum_{r \in \mathcal{R}}p(s^ \prime,r|s,a)=1, \text{ for all } s\in\mathcal{S},a\in\mathcal{A}(s)$

"""

# ╔═╡ 9ce34899-c8d0-428f-a3d3-91f0bf37198e
md"""
> ### *Exercise 3.6*
> Suppose you treated pole-balancing as an episodic task but also used discounting, with all rewards zero except for -1 upon failure. What then would the return be at each time? How does this return differ from that in the discounted, continuing formulation of this task?

The return at each time would be $G_t=-\gamma^{T-t-1}$ where T is the total number of steps in the episode. In the continuing formulation of the task there will be a series of failures each one occurring at a different time $T_i$. At any given time t, only the failures that occur in the future will contribute to the return: $G_t=\sum_{T_i >t}-\gamma^{T_i-t-1}$
"""

# ╔═╡ 9a3d9a79-e44a-40e5-b7bb-947f3327c598
md"""
> ### *Exercise 3.7* 
> Imagine you are designing a robot to run a maze. You decide to give it a reward of +1 for escaping the maze and a reward of zero at all other times. The task seems to break down naturally into episodes-the successive runs through the maze-so you decide to treat it as an episodic task, where the goal is to maximize the expected total reward (3.7). After running the learning agent for a while you find that it is showing no improvement in escaping from the maze. What is going wrong? Have you effectively communicated to the agent what you want it to achieve?

According to equation 3.7, there is no discount factor for the reward signal. If we assume the maze is simple enough that an agent escapes in some finite time, then it will receive a reward signal of 1. Any agent that solves the maze faster will receive the same reward as do agents that take longer. Because of the lack of discounting within an episode, there is no incentive for agents to solve the maze faster so any agent that can solve the maze is equally good, which in this case would be almost any arbitrary agent except one that simply stands still or only goes in one direction that gets stuck.
"""

# ╔═╡ 1b42d235-5f48-4625-a83e-48b50cbbc347
md"""
> ### *Exercise 3.8* 
>Suppose $\gamma=0.5$ and the following sequence of rewards is received $R_1=-1$, $R_2=2$, $R_3=6$, $R_4=3$, and $R_5=2$, with $T=5$. What are $G_0, \space G_1, \dots,G_5$

$\begin{flalign}

G_5&=0 \tag{by definition}\\

G_4&=R_5+\gamma G_5=2\\

G_3&=R_4+\gamma G_4=3+(0.5\times2)=4\\

G_2&=R_3+\gamma G_3=6+(0.5\times4)=8\\

G_1&=R_2+\gamma G_2=2+(0.5\times8)=6\\

G_0&=R_1+\gamma G_1=-1+(0.5\times6)=2
\end{flalign}$
"""

# ╔═╡ 30513c5c-20fe-4a31-9a59-67b73fa1e3a7
md"""
> ### *Exercise 3.9* 
> Suppose $\gamma=0.9$ and the reward sequence is $R_1=2$ followed by an infinite sequence of 7s. What are $G_1$ and $G_0$?

$\begin{flalign}
G_1&=7\times\sum_{k=0}^{\infty}\gamma^k=\frac{7}{1-\gamma}=70\\

G_0&=R_1+\gamma G_{1}=2+(0.9 \times 70)=65
\end{flalign}$
"""

# ╔═╡ 08ad2c4f-a3c1-4d9b-aafc-9b2394f68f53
md"""
> ### *Exercise 3.10* 
> Prove the second equality in (3.10).

Equation (3.10) is: 


$G_t=\sum_{k=0}^{\infty}\gamma^k=\frac{1}{1-\gamma}$

To prove the second equality we need to calulate the infinite sum:

$\begin{flalign}
G_t&=\gamma^0+\gamma^1+\gamma^2+\cdots\\

& = 1 + \gamma^1 + \gamma^2 + \cdots \\

&\therefore \\

\gamma \times G_t&=\gamma^1+\gamma^2+\cdots \\
&=G_t-1 \tag{comparing to above}\\

&\therefore \\

\gamma \times G_t&=G_t-1 \\

1 &= G_t\times(1 - \gamma) \tag{adding 1 and subtracting G}\\
G_t&=\frac{1}{1-\gamma} \tag{3.10 equality}
\end{flalign}$
"""

# ╔═╡ e9a973b7-889b-404e-bc31-1ad1b02f3864
md"""
## 3.5 Policies and Value Functions

> ### *Exercise 3.11* 
> If the current state is $S_t$, and actions are selected according to a stochastic policy $\pi$, then what is the expectation of $R_{t+1}$ in terms of $\pi$ and the four-argument function $p(s',r|s,a) \dot{=}Pr\{S_t=s', R_t=r|S_{t-1}=s,A_{t-1}=a\}$

$$\mathbb{E}[R_{t+1}]=\sum_{r \in \mathcal{R}}r\times Pr\{R_{t+1}=r|S_t=s\}$$

$\begin{flalign}
Pr\{R_{t+1}=r|S_t = s\} &= \mathbb{E}_\pi \left [ \sum_{s^\prime \in \mathcal{S}}Pr\{S_{t+1} = s^\prime, R_{t+1}=r|S_t = s, A_t = a\} \right ] \\
&= \mathbb{E}_\pi \left [ \sum_{s^\prime \in \mathcal{S}}p(s^\prime, r \vert s, a) \right ] \\
&=\sum_{a \in \mathcal{A(s)}}\pi(a|s)\sum_{s' \in \mathcal{S}}p(s', r|s,a) \\
&\therefore \\
\mathbb{E}[R_{t+1}] &=\sum_{r \in \mathcal{R}} \left[ r \times \left[ \sum_{a \in \mathcal{A(s)}}\pi(a|s) \left[ \sum_{s' \in \mathcal{S}}p(s', r|s,a) \right] \right] \right]
\end{flalign}$
"""

# ╔═╡ 7ecbbbd8-9823-41d3-8d38-10c81b59216a
md"""
> ### *Exercise 3.12* 
> Give an equation for $v_{\pi}$ in terms of $q_{\pi}$ and $\pi$.

From (3.12) we have 
$$v_{\pi}(s)= \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \vert S_t=s \right], \text{ for all } s \in \mathcal{S}$$

and from (3.13) we have

$$q_{\pi}(s,a)= \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \vert S_t=s,A_t=a\right]$$

So we need to average q over the actions, weighting them by the probability of taking those actions.

$$v_{\pi}(s)= \sum_{a \in \mathcal{A(s)}}\pi(a|s)q_{\pi}(s, a)$$
"""

# ╔═╡ db2157d9-abc1-43c1-8d37-0fe5e803667c
md"""
> ### *Exercise 3.13* 
> Give an equation for $$q_{\pi}$$ in terms of $$v_{\pi}$$ and $$p(s',r|s,a)$$

$\begin{flalign}
q_{\pi}(s,a) & = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t=s,A_t=a \right] \\

&=\mathbb{E}_{\pi} \left [ R_{t+1} + \sum_{k=1}^{\infty} \gamma^k R_{t+k+1} \mid S_t=s,A_t=a \right] \\

&=\sum_{r, s^\prime} p(s^\prime, r \vert s, a) \left [ r + \gamma \mathbb{E}_{\pi} \left[ \sum_{k=1}^{\infty} \gamma^{k-1} R_{t+k+1} \vert S_{t+1}=s^\prime\right] \right ] \tag{by 3.5} \\

&=\sum_{r, s^\prime} p(s^\prime, r \vert s, a) \left [ r + \gamma \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^{k} R_{t+k+2} \vert S_{t+1}=s^\prime\right] \right ] \\

&=\sum_{r, s^\prime} p(s^\prime, r \vert s, a) \left [ r + \gamma \mathbb{E}_{\pi} \left[ G_{t+1} \mid S_{t+1} = s^\prime \right ] \right ] \tag{by 3.9} \\

&=\sum_{r \in \mathcal{R}} \sum_{s ^ \prime \in \mathcal{S}} p(s^ \prime, r|s, a) \left ( r + \gamma v_\pi (s^\prime) \right ) \tag{by 3.12} \\
\end{flalign}$
"""

# ╔═╡ 7e4d7ca2-c4a2-49a0-a2cd-cd1e50a048de
md"""
> ### *Exercise 3.14* 
> The Bellman equation (3.14) must hold for each state for the value function $v_{\pi}$ shown in Figure 3.2 (right) of Example 3.5. Show numerically that this equation holds for the center state, valued at $+0.7$, with respect to its four neighboring states, valued at $+2.3$, $+0.4$, and $+0.7$. (These numbers are accurate only to one decimal place.)

The Bellman equation states

$$v_{\pi}(s)=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)\left[r+\gamma v_{\pi}(s')\right], \text{ for all } s\in\mathcal{S}$$

The value function in Figure 3.2 shows a policy in which each of the four possible actions is selected with equal probability so we can consider each action component of the sum separately.

north case: for a move north from the center, the agent with 100% probability receives a reward of 0 and ends up in the square directly north of the current square which has a value estimate. So the sums can be replaced with the certain outcomes and the term becomes.  
$$\sum_{s',r}p(s',r|s,north)\left[r+\gamma v_{\pi}(s')\right], \text{ for all } s\in\mathcal{S}$$
$$0+0.9 \times 2.3=2.07$$

Since every other possible action has a completely deterministic state transition and reward outcome, we can directly write down the sum contribution for each one.

south case: $0+0.9\times -0.4=-0.36$

east case: $0+0.9\times 0.4=0.36$

west case: $0+0.9\times 0.7=0.63$

Now applying the Bellman equation: $v_{\pi}(s)=0.25 \times [2.07-0.36+0.36+0.63]=0.25 \times 2.7=0.675$, which rounded to the nearest decimal value matches the value in the figure of 0.7.
"""

# ╔═╡ be824355-6fab-4367-ab09-4efaa96b9aae
md"""
> ### *Exercise 3.15* 
> In the gridworld example, rewards are positive for goals, negative for running into the edge of the world, and zero the rest of the time. Are the signs of these rewards important, or only the intervals between them? Prove, using (3.8), that adding a constant $c$ to all rewards adds a constant, $v_c$, to the values of all states, and thus does not affect the relative values of any states under any policies. What is $v_c$ in terms of $c$ and $\gamma$?

Equation 3.8 states:

$$G_t=\sum_{k=0}^\infty \gamma^k R_{t+k+1}$$

So if we add a constant $c$ to every reward value then the expected discounted return becomes:

$$G_t'=\sum_{k=0}^\infty \gamma^k (c+R_{t+k+1})=\sum_{k=0}^\infty c\gamma^k+\gamma^k R_{t+k+1}=\left(c\sum_{k=0}^\infty \gamma^k\right) + G_t=\frac{c}{1-\gamma}+G_t$$

So the modified estimate is simply the previous estimate plus a constant value which does not change under any state or policy. The constant added is just $\frac{c}{1-\gamma}$. So if a constant value is added to all rewards to remove negative values, it will not affect the relative differences between value estimates of any state under any policy.
"""

# ╔═╡ 97a605e6-31dc-4ba9-acc5-e0d81093c3ee
md"""
> ### *Exercise 3.16* 
> Now consider adding a constant c to all rewards in an episodic task, such as maze running. Would this have any effect, or would it leave the task unchanged as in the continuing task above? Why or why not? Give an example.

In an episodic case, equation 3.8 becomes:

$$G_t=\sum_{k=0}^{T-t-1}\gamma^k R_{t+k+1}$$

where T is the length of a particular episode. We can try the same substitution here of adding a constant value to each reward.

$$G_t'=\sum_{k=0}^{T-t-1} \gamma^k (c+R_{t+k+1})=\sum_{k=0}^{T-t-1}c\gamma^k+\gamma^k R_{t+k+1}=\left(c\sum_{k=0}^{T-t-1}\gamma^k\right) + G_t$$

Unlike in the previous case, the sum term does not become a constant of $c$ and $\gamma$, but is a sum of the form $\sum_{k=0}^{N}\gamma^k$ which can be simplified as follows:

$$S = \sum_{k=0}^N \gamma^k=\gamma^0+\gamma^1+\gamma^2+\cdots+\gamma^N$$
$$S\gamma =\gamma^1+\gamma^2+\cdots+\gamma^{N+1}=S-1+\gamma^{N+1}$$
$$S(\gamma-1)=-1+\gamma^{N+1} \implies S=\frac{\gamma^{N+1}-1}{\gamma-1}$$

Substituting this into the modified equation for G we get:

$$G_t'=c\frac{\gamma^{T-t}-1}{\gamma-1} + G_t = \left( G_t+\frac{c}{1-\gamma}\right) + \frac{c\gamma^{T-t}}{\gamma-1}$$

The part of this equation in parentheses is identical to what we had in the continuing case, but there is an additional term that depends on T (the total episode length) and t (the step we are on of the current episode). To see what this term does to G, we can plot its value for each step of an episode.
"""

# ╔═╡ c12ca18c-0780-4c02-9396-82b97f019bc6
@bind params PlutoUI.combine() do Child
	md"""
	Discount Factor $\gamma$: $(Child(:γ, Slider(0.0:0.001:1.0, default = 0.9, show_value = true)))
	
	Terminal Step: $(Child(:t_final, Slider(10:1000, default = 101, show_value = true)))
	"""
end

# ╔═╡ e4d73777-cf4b-40c5-8922-b9df28d25aa0
begin
	t_final = params.t_final
	t = 0:t_final
	f(t, t_final, γ) = (γ^(t_final - t) - 1) / (γ - 1)
	plot(t, f.(t, t_final, params.γ), Layout(xaxis_title = "Time Step", title = "Episodic Reward Factor for $t_final Step Episode"))
end

# ╔═╡ ede978b8-dd9d-4b26-88c1-7def0dae42ee
md"""
On early timesteps the additional value is near zero because the numerator takes γ to a large power (unless the length of the episode is very small).  This means the reward values would all be shifted up by the same amount as in the continuing case but uniformly  As we get closer to the end of the episode, the factor approaches 1 for the step right before the terminal state.  This reflects the expectation of receiving one more c value before the episode ends.  The values near the beginning of the episode match the expectation of receiving the added c value indefinitely.   

For the agent's value function, the expected value of the return is what is relevant:

$\mathbb{E}_\pi[G^\prime_t \vert S_t = s] = \mathbb{E}_\pi[G_t \vert S_t = s] + \frac{c}{1 - \gamma}\left [1 - \mathbb{E}_\pi[\gamma^{T-t} \vert S_t = s] \right ]$

The impact on the value function will be:

$v^\prime_\pi(s) = v_\pi(s) + \frac{c}{1 - \gamma}\left [1 - \mathbb{E}_\pi[\gamma^{T-t} \vert S_t = s] \right ]$

So it seems that rather than each value being shifted by a constant, there is also a factor that depends on an expected value related to the number of steps until termination.  Since each state could be a different number of steps away from termination, the value function will fundamentally change.  States that are far from termination will be valued higher by a factor close to $\frac{c}{1 - \gamma}$ whereas states that are close to termination will have a value that is increased by a smaller factor.  This effect is reversed if c is negative.  Let's say that c is large and positive in the gridword task.  If the states that produce reward are terminal states, then the agent would value states higher than avoid them for additional time steps because once the episode ends, the opportunity to accumulate more values of c is over.  Similarly, if c was largely negative, then an agent may value states that approach a low reward terminal state transition, if it avoids taking too many steps towards a higher reward terminal state transition that would accumulate too many penalties of c along the way.  So, in the case of episodes that are expected to be of finite length, the shifting of rewards fundamentally changes the task. 
"""

# ╔═╡ b5871733-c403-4b39-8b51-2f3941c8a634
md"""
> ### *Exercise 3.17* 
> What is the Bellman equation for action values, that is, for $q_{\pi}$? It must give the action value $q_{\pi}(s,a)$ in terms of the action values, $q_{\pi}(s',a')$, of possible successors to the state-action pair $(s,a)$.  
> Hint: The backup diagram to the right corresponds to this equation. Show the sequence of equations analogous to (3.14), but for action values.

Following the example in (3.14) but for $q_{\pi}(s, a)$ intsead of $v_{\pi}(s)$ we have:

$\begin{flalign}
q_{\pi}(s,a) & \doteq \mathbb{E}_\pi [G_t|S_t=s,A_t=a] \\
&= \mathbb{E}_\pi [R_{t+1}+\gamma G_{t+1}|S_t=s,A_t=a]\\
&=\sum_{s',r} p(s',r|s,a)\left[r+\gamma\mathbb{E}_\pi [G_{t+1}|S_{t+1}=s'] \right]\\
&=\sum_{s',r} p(s',r|s,a)\left[r+\gamma \sum_{a'} \pi(a', s')\mathbb{E}_\pi [G_{t+1}|S_{t+1}=s',A_{t+1}=a'] \right]\\
&=\sum_{s',r} p(s',r|s,a)\left[r+\gamma \sum_{a'} \pi(a', s')q_{\pi}(s',a') \right], \text{ for all } s \in \mathcal{S}, a \in \mathcal{A(s)}\\
\end{flalign}$
"""

# ╔═╡ 60aa5fa8-ea7d-45c9-8528-22ddd3ba74e2
md"""
> ### *Exercise 3.18* 
> The value of a state depends on the values of the actions possible in that state and how likely each action is to be taken under the current policy. We can think of this in terms of a small backup diagram rooted at the state and considering each possible action. Give the equation corresponding to this intuition and diagram for the value at the root node, $v_\pi(s)$, in terms of the value at the expected leaf node, $q_\pi(s,a)$, given $S_t=s$. This equation should include an expectation conditioned on following the policy, $\pi$. Then give a second equation in which the expected value is written out explicitly in terms of $\pi(a|s)$ such that no expected value notation appears in the equation.

In the diagram we see the value function at the root connecting to all of the the possible actions from that state with a corresponding q value. Each action is taken with a probability given by the policy. Since $v_\pi(s)$ is an average over the value of all actions that could be taken by the policy from this point, we can write it in terms of the expected action.

$$v_\pi(s)=\mathbb{E}_\pi[q_\pi(s, a) \vert S_t = s]$$

We can rewrite this using the probabilities given by the policy at each action explicitly:

$$v_\pi(s)=\sum_{a}\pi(a|s)q_\pi(s,a) \text{ for all } a\in\mathcal{A(s)}$$
"""

# ╔═╡ a0257b35-b3c8-4bf6-948a-48ced42addf7
md"""
> ### *Exercise 3.19* 
> The value of an action, $q_\pi(s,a)$, depends on the expected next reward and the expected sum of the remaining rewards. Again we can think of this in terms of a small backup diagram, this one rooted at an action (state-action pair) and branching to the possible next states.  
> Give the equation corresponding to this intuition and diagram for the action value, $q_\pi(s,a)$, in terms of the expected next reward, $R_{t+1}$, and the expected next state value, $v_\pi(S_{t+1})$, given that $S_t=s$ and $A_t=a$. This equation should include an expectation but *not* one conditioned on the following policy. Then give a second equation, writing out the expected value explicitly in terms of $p(s',r|s,a)$ defined by (3.2), such that no expected value notation appears in the equation.

The diagram shows a root for the action value estimate and all of the (future state, reward) pairs that are possible from that action. Since there is a distribution over these pairs, we can write the equation in terms of expected value:

$$q_\pi(s,a)=\mathbb{E}[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s,A_t=a] $$

From equation 3.2 we have:

$$p(s',r|s,a) \doteq \text{Pr}\{S_t=s',R_t=r|S_{t-1}=s,A_{t-1}=a\}$$
Since this provides the probability for each (future-state, reward) pair that is possible after the current (state, action) pair, we can directly compute the expected action value:

$$q_\pi(s,a)=\sum_{r,s'}p(s',r|s,a)(r+\gamma v_\pi(s')) \text{ for all } r\in\mathcal{R},s'\in\mathcal{S}$$
"""

# ╔═╡ 65457924-9c1f-4e13-834f-22b68e7e9062
md"""
## 3.6 Optimal Policies and Optimal Value Functions
"""

# ╔═╡ 0433fbf6-c753-4621-8ef3-8229cf77b9b2
md"""
> ### *Exercise 3.20* 
> Draw or describe the optimal state-value function for the golf example.

The optimal state-value function assumes that the optimal action is taken which in this case is simply the choice between putter and driver. The optimal action-value function is already shown, so as long as the optimal choice in each region is driver, this function will be identical to the state-value for that state. For all states on the green, the optimal action is putter, not driver, so unlike the bottom of Figure 3.3, any point on the green should have a value of -1. Any point outside the green but within the -2 contour will still have a value of -2 since these states can all reach the green by using the driver. Any point that could reach the green already shares the same value as the optimal state-value function because the optimal action is selected for the subsequent shot.
"""

# ╔═╡ 23516799-1bce-41a2-8dff-f8b8268b54d1
md"""
> ### *Exercise 3.21* 
> Draw of describe the contours of the optimal action-value function for putting, $q_{*}(s,\text{putter})$, for the golf example.

The $q_*(s,\text{putter})$ action value function restricts the selected initial action to be putter, but any subsequent action selected will be the optimal one. Starting on the green, we still have a value of -1 because any ball on the green can reach the hole in one shot. Next we have the contour shown in the top of Figure 3.3 for -2 which will also be the same since any ball in this state can reach the green and then the hole in the next two strokes. The sand pit however will also share a value of -3 with the rest of the -3 contour. Balls in the sandpit cannot move with the putter so a shot will be wasted, but the driver will be used on the next shot to reach the green and then the hole using a total of 3 shots. The -3 contour will remain -3 for the optimal policy because one putt will be used to reach the -2 contour and then whether driver or putter is selected next, it will take exactly 2 strokes to reach the hole. The contour labeled -4 can reach the -3 region with a putt; however at that point the optimal action would be to use the driver and get a hole in another 2 strokes. Therefore, the -4 contour will be merged into the -3 contour, taking on its value. The -5 contour can reach the -4 contour with a putt. If we approximate that this lies within the driver range of the green then much of this region will also share a value of -3. Only the remaining contour of -6 will take on a value of -4 since puts from this region can only reach a region that is still 3 shots away from a hole.
"""

# ╔═╡ 07f0e0a7-8c6e-4474-bc8b-ddf6eaa19a34
md"""
> ### *Exercise 3.22* 
> Consider the continuing MDP shows to the right. The only decision to be made is that in the top state, where two actions are available, left and right. The numbers show the rewards that are received deterministically after each action. There are exactly two deterministic policies, $\pi_{\text{left}}$ and $\pi_{\text{right}}$. What policy is optimal if $\gamma=0$? If $\gamma=0.9$? If $\gamma=0.5$?

For $\gamma=0$ the only reward considered is the immediate one from the chosen action. If we select left, the immediate reward is +1 vs 0 so $\pi_{\text{left}}$ is optimal.

For $\gamma \not=0$, we can calculate the future discounted reward of each policy:

$$G_{\pi_{\text{left}}}=1+\gamma^2+\gamma^4+\cdots=\frac{1}{1-\gamma^2}$$
$$G_{\pi_{\text{right}}}=2\times(\gamma+\gamma^3+\cdots)=\frac{2\gamma}{1-\gamma^2}=2\gamma G_{\pi_{\text{left}}}$$

So it is clear that if $\gamma>0.5$ then $\pi_{right}$ is more optimal than $\pi_{left}$ and they are equal if $\gamma=0.5$.
"""

# ╔═╡ 7ca226e2-0d8e-4f31-94e1-b0f5301f32ba
md"""
> ### *Exercise 3.23* 
> Give the Bellman equation for $q_*$ for the recycling robot.

$$q_*(s,a)=\sum_{s',r}p(s',r|s,a)\left[r+\gamma \max_{a'}q_*(s',a') \right]$$

As in example 3.9 we will abbreviate the two states high and low with $h$, $l$ and the three possible actions of search, wait, and recharge by $s$, $w$, $re$.

Starting with the h state, there are two possible actions of w and s.

$\begin{flalign}
q_*(h,s)&=p(h|h,s)[r(h,s,h)+\gamma\max_{a'}q_*(h,a')]+p(l|h,s)[r(h,s,l)+\gamma\max_{a'}q_*(l,a')] \\
&=\alpha[r_s+\gamma\max_{a'}q_*(h,a')]+(1-\alpha)[r_s+\gamma\max_{a'}q_*(l,a')] \\
&=r_s+\gamma[\alpha\max_{a'}q_*(h,a')+(1-\alpha)\max_{a'}q_*(l,a')] \\
q_*(h,w)&=r_w+\gamma\max_{a'}q_*(h,a')\\
\end{flalign}$

Starting with the l state, there are three possible actions: $w$, $s$, and $re$.

$\begin{flalign}
q_*(l,s)&=\beta[r_s+\gamma\max_{a'}q_*(l,a')]+(1-\beta)[-3+\gamma\max_{a'}q_*(h,a')] \\
q_*(l,w)&=r_w+\gamma\max_{a'}q_*(l,a')\\
q_*(l,re)&=\gamma\max_{a'}q_*(h,a')\\
\end{flalign}$

Together these five non-linear equations specify $q_*$ for each of the five state-action pairs given the constants $\alpha$, $\beta$, and $\gamma$ as well as the reward values $r_s$ and $r_w$.
"""

# ╔═╡ 9814c35f-ae0f-436b-ab6a-12d2da7922e0
md"""
> ### *Exercise 3.24* 
> Figure 3.5 gives the optimal value of the best state of the gridworld as 24.4, to one decimal place. Use your knowledge of the optimal policy and (3.8) to express this value symbolically, and then to compute it to three decimal places.

Equation 3.8 provides the expected discounted return as:

$$G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}$$

If we assume the same discount factor as when the problem was introduced of $$\gamma=0.9$$, then we can iterate through the grid using the optimal policy and collect a sequence of rewards.  Rewards are -1 for actions that try to leave the grid, +10 for any action in A, +5 for any action in square B, and 0 otherwise.  Starting at square A, all actions are optimal and result in a reward of +10.  The optimal policy is then to move vertically back to A from A'.  This sequence of actions will result in the reward sequence: $\{+10, 0, 0, 0, 0, +10, \cdots\}$ leading to the discounted return of:

$\begin{flalign}
G_t&=10\gamma^0 + 0\gamma^1 + 0\gamma^2 + 0\gamma^3 + 0 \gamma^4 + 10\gamma^5 + \cdots\\
&=10 [1 + \gamma^5 + \gamma^{10} + \cdots ]
\end{flalign}$

Taking $c = \gamma^5$, the term in the brackets is the series $\sum_{i = 0}^{\infty} c^i = (1-c)^{-1} = (1-\gamma^5)^{-1}$.  Therefore the discounted return is:

$$G_t=\frac{10}{1-\gamma^5}$$

And for $\gamma=0.9$: 

$G_t=\frac{10}{1-.9^5} \approx 24.419$

which is consistent with figure 3.5 when rounded to one decimal place.
"""

# ╔═╡ 6997e43e-ae25-4d71-a165-65fdb37f860c
md"""
> ### *Exercise 3.25* 
> Give an equation for $v_*$ in terms of $q_*$.

$$v_*(s)=\max_{a\in \mathcal{A} (s)}q_*(s,a)$$
"""

# ╔═╡ 5f0b3ca5-2bec-49b3-8ca2-f4381033c15b
md"""
> ### *Exercise 3.26* 
> Give an equation for $q_*$ in terms of $v_*$ and the four-argument $p$.

$\begin{flalign}
q_*(s,a)&=\mathbb{E} \left [ R_{t+1}+\gamma v_*(S_{t+1})|S_t=s,A_t=a \right ] \\
&=\sum_{r,s'}p(s',r|s,a)[r+\gamma v_*(s')]
\end{flalign}$
"""

# ╔═╡ fdd8ca66-00e7-4ce3-85ec-52cafc27bdba
md"""
> ### *Exercise 3.27* 
> Give an equation for $\pi_*$ in terms of $q_*$.

$\pi_*(s) = \begin{cases}
1 & a = \underset{a \in \mathcal{A}(s)}{\mathrm{argmax}}[q_*(s,a)]\\
0 & \text{else}
\end{cases}$
"""

# ╔═╡ cfb040e5-663f-491d-a949-81ad7630a1f3
md"""
> ### *Exercise 3.28* 
> Give an equation for $\pi_*$ in terms of $v_*$ and the four-argument $p$.

In exercise 3.27 for the case of $\pi_* = 1$, we can rewrite the expression in terms of $v_*$ by using the expression in exercise 3.26:

$\underset{a \in \mathcal{A}(s)}{\mathrm{argmax}}[q_*(s,a)]=\underset{a \in \mathcal{A}(s)}{\mathrm{argmax}} \left [ \sum_{r,s'}p(s',r|s,a)[r+\gamma v_*(s')] \right ]$

So the expression for the optimal policy is just:

$\pi_*(s) = \begin{cases}
1 & a = \underset{a \in \mathcal{A}(s)}{\mathrm{argmax}} \left [ \sum_{r,s'}p(s',r|s,a)[r+\gamma v_*(s')] \right ] \\
0 & \text{else}
\end{cases}$
"""

# ╔═╡ 2ecb796a-4d41-11ee-2293-2f0ee0eeff79
md"""
> ### *Exercise 3.29* 
> Rewrite the four Bellman equations for the four value functions $(v_\pi, \space v_*, \space q_\pi, \text{ and } q_*)$ in terms of the three argument function $p$ (3.4) and the two-argument function $r$ (3.5).

From (3.4) we have:

$$p(s'|s,a)=\sum_{r\in\mathcal{R}}p(s',r|s,a)$$

and from (3.5) we have:

$$r(s,a)=\sum_{r \in \mathcal{R}}r\sum_{s' \in \mathcal{S}}p(s',r|s,a)$$

Starting with $v_\pi$:

$\begin{flalign}
v_\pi(s)&=\sum_{a}\pi(a,s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_\pi(s')]\\
&=\sum_{a}\pi(a,s)\left[r(s,a)+ \gamma \sum_{s'}p(s'|s,a) v_\pi(s')\right]
\end{flalign}$

Next for $v_*$:

$\begin{flalign}
v_*(s)&=\max_{a} \left [ \sum_{s',r}p(s',r|s,a)[r+\gamma v_*(s')] \right ]\\
&=\max_{a} \left [ r(s,a)+ \gamma \sum_{s'}p(s'|s,a) v_*(s') \right ]
\end{flalign}$

Next for $q_\pi$:

$\begin{flalign}
q_\pi(s,a)&=\sum_{s',r} p(s',r|s,a)\left[r+\gamma \sum_{a'} \pi(a', s')q_{\pi}(s',a') \right] \\
&=r(s,a)+\gamma\sum_{s'} p(s'|s,a)\sum_{a'} \pi(a', s')q_{\pi}(s',a')
\end{flalign}$

Next for $q_*$:

$\begin{flalign}
q_*(s,a)&=\sum_{s',r}p(s',r|s,a)\left[r+\gamma \max_{a'} \left [ q_*(s',a') \right ] \right] \\
&=r(s,a)+ \gamma \sum_{s'}p(s'|s,a) \max_{a'} \left [ q_*(s',a') \right ]
\end{flalign}$
"""

# ╔═╡ 4ca58fcf-3115-4100-9f83-b8a389e4eaa0
md"""
# Dependencies and Settings
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoPlotly = "~0.3.9"
PlutoUI = "~0.7.52"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.3"
manifest_format = "2.0"
project_hash = "518adb648c80095d555fe737933aaac06e6c2875"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "d9a8f86737b665e15a9641ecbac64deef9ce6724"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.23.0"

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
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

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
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "f9b1e033c2b1205cf30fd119f4e50881316c1923"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.1"
weakdeps = ["Requires", "TOML"]

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
deps = ["AbstractPlutoDingetjes", "Colors", "Dates", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "PackageExtensionCompat", "PlotlyBase", "PlutoUI", "Reexport"]
git-tree-sha1 = "9a77654cdb96e8c8a0f1e56a053235a739d453fe"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.3.9"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

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
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "b7a5e99f24892b6824a954199a45e9ffcc1c70f0"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.0"

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
# ╟─17f36458-139b-4f8b-aba9-d0dd586dd82c
# ╟─c5abf826-9ce8-4319-a2e5-6cf7fcc61400
# ╟─85905a6e-1807-4b77-b313-dbadb8b898c8
# ╟─090c50ed-6772-457a-afbb-cf2cde0e2ec4
# ╟─768833aa-ceee-4fe8-958f-c00b778ec764
# ╟─9ce34899-c8d0-428f-a3d3-91f0bf37198e
# ╟─9a3d9a79-e44a-40e5-b7bb-947f3327c598
# ╟─1b42d235-5f48-4625-a83e-48b50cbbc347
# ╟─30513c5c-20fe-4a31-9a59-67b73fa1e3a7
# ╟─08ad2c4f-a3c1-4d9b-aafc-9b2394f68f53
# ╟─e9a973b7-889b-404e-bc31-1ad1b02f3864
# ╟─7ecbbbd8-9823-41d3-8d38-10c81b59216a
# ╟─db2157d9-abc1-43c1-8d37-0fe5e803667c
# ╟─7e4d7ca2-c4a2-49a0-a2cd-cd1e50a048de
# ╟─be824355-6fab-4367-ab09-4efaa96b9aae
# ╟─97a605e6-31dc-4ba9-acc5-e0d81093c3ee
# ╟─c12ca18c-0780-4c02-9396-82b97f019bc6
# ╟─e4d73777-cf4b-40c5-8922-b9df28d25aa0
# ╟─ede978b8-dd9d-4b26-88c1-7def0dae42ee
# ╟─b5871733-c403-4b39-8b51-2f3941c8a634
# ╟─60aa5fa8-ea7d-45c9-8528-22ddd3ba74e2
# ╟─a0257b35-b3c8-4bf6-948a-48ced42addf7
# ╟─65457924-9c1f-4e13-834f-22b68e7e9062
# ╟─0433fbf6-c753-4621-8ef3-8229cf77b9b2
# ╟─23516799-1bce-41a2-8dff-f8b8268b54d1
# ╟─07f0e0a7-8c6e-4474-bc8b-ddf6eaa19a34
# ╟─7ca226e2-0d8e-4f31-94e1-b0f5301f32ba
# ╟─9814c35f-ae0f-436b-ab6a-12d2da7922e0
# ╟─6997e43e-ae25-4d71-a165-65fdb37f860c
# ╟─5f0b3ca5-2bec-49b3-8ca2-f4381033c15b
# ╟─fdd8ca66-00e7-4ce3-85ec-52cafc27bdba
# ╟─cfb040e5-663f-491d-a949-81ad7630a1f3
# ╟─2ecb796a-4d41-11ee-2293-2f0ee0eeff79
# ╟─4ca58fcf-3115-4100-9f83-b8a389e4eaa0
# ╠═86d53794-2251-47d5-a45e-f1da53cd8ef5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
