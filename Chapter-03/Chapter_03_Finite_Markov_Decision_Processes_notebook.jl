### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ 86d53794-2251-47d5-a45e-f1da53cd8ef5
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using PlutoPlotly, PlutoUI
	TableOfContents()
end
  ╠═╡ =#

# ╔═╡ 1c7d2750-3625-4eed-8408-53626b67d749
md"""
# Chapter 3: Finite Markov Decision Processes

Finite Markov decision processes or MDPs are a mathematical framework to study the idealized reinforcement learning problem.  These tasks involve sequential decision making where actions affect not just immediate rewards but also subsequent situations or states which in turn also affect future rewards.  In the bandit problem we sought to learn a value estimate for each action $q_*(a)$.  In MDPs we estimate the value of $q_*(s, a)$ which includes both the state $s$ and the action $a$.  These state-dependent quantities are essential to accurately assigning credit for long-term consequences of individual action selections.

## 3.1 The Agent-Environment Interface

MDPs are meant to be a straightforward framing of the problem of learning from interaction to achieve a goal.  The learner and decision maker is called the *agent*.  The thing it interacts with, comprising everything outside the agent, is called the *environment*.  These interact continually, the agent selecting actions and the environment responding to these actions and presenting new situations to the agent.  The environment also gives rise to rewards, special numerical values that the agent seeks to maximize over time through its choice of actions.

More specifically, the agent and environment interact at each of a sequence of discrete time steps, $t=0,1,2,3,\dots$.  At each time step $t$, the agent receives some representation of the environment's *state*, $S_t \in \mathcal{S}$, and on that basis selects an *action*, $A_t \in \mathcal{A}(s)$.  One time step later, in part as a concequence of its action, the agent receives a numerical *reward*, $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$ and finds itself in a new state, $S_{t+1}$.  The MDP and agent together thereby give rise to a sequence of *trajectory* that begins like this:

$S_0,A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \dots \tag{3.1}$

In a *finite* MDP, the sets of states, actions, and rewards all have a finite number of elements.  In this case, the random variables $R_t$ and $S_t$ have well defined discrete probability distributions dependent only on the preceding state and action.  That is, for particular values of these random variables, $s^\prime \in \mathcal{S}$ and $r \in \mathcal{R}$, there is a probability of those values occurring at time $t$, given particular values of the preceding state and action: 

$p(s^\prime, r \vert s, a) \doteq \Pr \{ S_t = s^\prime, R_t = r \mid S_{t-1} = s, A_{t-1}=a \}, \tag{3.2}$

for all $s^\prime, s \in \mathcal{S}, r \in \mathcal{R}, \text{ and } a \in \mathcal{A}(s)$.  The function $p$ defines the *dynamics* of the MDP.  The dynamics function $p : \mathcal{S} \times \mathcal{R} \times \mathcal{S} \times \mathcal{A} \rightarrow [0,1]$ is an ordinary deterministic function of four arguments.  The '|' in the middle of it comes from the notation for conditional probability, but here it just reminds us that $p$ satisfies a probability distribution for each choice of $s$ and $a$, that is, that

$\sum_{s^\prime \in \mathcal{S}}\sum_{r \in \mathcal{R}} p(s^\prime, r \vert s, a) = 1, \text{ for all } s \in \mathcal{S}, a \in \mathcal{A}(s). \tag{3.3}$

In a *Markov* decision process, the probabilities given by $p$ completely characterize the environment's dynamics.  That is, the probability of each possible value for $S_t$ and $S_t$ depends on the immediately preceding state and action, $S_{t-1}$ and $A_{t-1}$, and, given them, not at all on earlier states and actions.  This is best viewed as a restriction not on the decision process, but on the *state*.  The state must include information about all aspects of the past agent-environment interaction that make a difference for the future.  If it does, then the state is said to have the *Markov property*.  We will assume the Markov property throughout this book.

From the four-argument dynamics function, $p$, one can compute anything else one might want to know about the environment, such as the *state-transition probabilities* (which we denote, with a slight abuse of notation, as a three-argument function $p : \mathcal{S} \times \mathcal{S} \times \mathcal{A} \rightarrow [0, 1]$),

$p(s^\prime \vert s, a) \doteq \Pr \{ S_t = s^\prime \mid S_{t-1} = s, A_{t-1} = a \} = \sum_{r \in \mathcal{R}} p(s^\prime, r \vert s, a). \tag{3.4}$

We can also compute the expected rewards for state-action pairs as a two-argument function $r : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$:

$r(s, a) \doteq \mathbb{E}[R_t \mid S_{t-1} = s, A_{t-1} = a] = \sum_{r \in \mathcal{R}} r \sum_{s^\prime \in \mathcal{S}} p(s^\prime, r \vert s, a), \tag{3.5}$

and the expected rewrads for state-action-next-state triples as a three-argument function $r : \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R},$,

$r(s, a, s^\prime) \doteq \mathbb{E}[R_t \mid S_{t-1} = s, A_{t-1} = a, S_t = s^\prime] = \sum_{r \in \mathcal{R}} r \frac{p(s^\prime, r \vert s, a)}{p(s^\prime \vert s, a)}. \tag{3.6}$


"""

# ╔═╡ e7a4f148-bf74-44f3-98e7-e4d9b4cac5ab
md"""
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
| high | search | high | $r_{\text{search}}$ | $\alpha$ |
| high | search | low | $r_{\text{search}}$ | $1-\alpha$ |
| low | search | low | $r_{\text{search}}$ | $\beta$ |
| low | search | high | -3  | $1-\beta$ |
| high | wait | high | $r_{\text{wait}}$ | 1   |
| low | wait | low | $r_{\text{wait}}$ | 1   |
| low | recharge | high | 0   | 1   |
"""

# ╔═╡ f376a2e1-69d3-46ca-9bc2-33e447c6834b
md"""
## 3.2 Goals and Rewards

In reinforcement learning, the purpose or goal of the agent is formalized in terms of a special signal, called the reward, passing from the environment to the agent. At each time step, the reward is a simple number, $R_t \in \mathbb{R}$. Informally, the agent’s goal is to maximize the total amount of reward it receives. This means maximizing not immediate reward, but cumulative reward in the long run. We can clearly state this informal idea as the reward hypothesis:

>That all of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward).


The use of a reward signal to formalize the idea of a goal is one of the most distinctive
features of reinforcement learning.

Although formulating goals in terms of reward signals might at first appear limiting, in practice it has proved to be flexible and widely applicable.  The best way to see this is to consider examples of how it has been, or could be, used.  For example, to make a robot learn how to escape a maze, the reward is often -1 for every time step that passes prior to escape; this encourages the agent to escape as quickly as possible.  For an agent to learn to play a win/loss/draw game the natural rewards are +1 for winning, -1 for losing, and 0 for draws and all nonterminal positions.

If we want an agent to do something for us, we must provide rewards to it in a raw that in maximizing them the agent will also acheive our goals.  It is thus critical that the rewards truly indicate what we want accomplished.  In particular, the reward signal is not the place to impart to the agent prior knowledge about *how* to acheive what we want to do.  For example, a chess-playing agent should be rewarded only for actually winning, not for achieving subgoals such as taking its opponent's pieces or gaining control of the center of hte board.  If achieving these shorts of subgoals were rewarded, then the agent might find a way to achieve them without achieving the real goal.  For example, it might find a way to take the opponent's pieces even at the cost of losing the game.  **The reward signal is your way of communicating to the agent *what* you want achieved, not *how* you want it achieved.**
"""

# ╔═╡ 6a85a90d-f345-4604-9637-086a71928af5
md"""
## 3.3 Returns and Episodes

Now we formalize the notation for describing in detail the cummulative long term reward of an agent.  If the sequence of rewards after time step $t$ is denoted, $R_{t+1}, R_{t+2}, R_{t+3}, \dots$, then what precise aspect of this sequence do we wish to maximize?  In general, we seek to maximize the *expected return*, where the return, denoted $G_t$, is defined as some specific function of the reward sequence.  In the simplest case the return is the sum of the rewards:

$G_t \doteq R_{t+1}+R_{t+2}+R_{t+3} + \cdots + R_T, \tag{3.7}$

where $T$ is a final time step.  This approach makes sense in applications in which there is a natural notation of the final time step, that is, when the agent-environment interaction breaks naturally into subsequences, which we call *episodes*, such as plays of a agame, trips through a maze, or any sort of repeated interaction.  Each episode ends in a special state called the *terminal state*, followed by a reset to a standard starting state or to a sample from a standard distribution of starting states which is independent from the previous episode.  Thus the episodes can all be considered to independently end in the same terminal state, with different rewards for different outcomes.  Tasks with episodes of this kind are called *episodic tasks*.  In episodic tasks we sometimes need to distinguish the set of all nonterminal states, denoted $\mathcal{S}$, from the set of all states plus the terminal state, denoted $\mathcal{S}^+$.  The time of termination, $T$, is a random variable that normally varies from episode to episode.

On the other hand, in many cases the agent-environment interaction does not break naturally into identifiable episodes, but goes on continually without limit.  For example, this would be the natural way to formulate an on-going process-control task.  We call these *continuing tasks*.  The return formulation (3.7) is problematic for continuing tasks because the final time step would be $T = \infty$, and the return, which is what we are trying to maximize, could easily be infinite.

To fix these problems we need the concept of *discounting*.  According to this approach, the agent tries to select actions so that the sum of the discounted rewards it receives over the future is maximized.  In particular, it chooses $A_t$ to maximize the expected *discounted return:*

$G_t \doteq R_{t+1}+\gamma R_{t+2}+ \gamma^2 R_{t+3} + \cdots = \sum_{k = 0}^\infty \gamma^k R_{t+k+1}, \tag{3.8}$

where $\gamma$ is a parameter, $0 \leq \gamma \leq 1$, called the *discount rate.*

The discount rate determines the present value of future rewards: a reward received $k$ time steps in the future is woth only $\gamma^{k-1}$ times what it would be worth if it were received immediately.  If $\gamma \lt 1$, the infinite sum in (3.8) has a finite value as long as the reward sequence $\{ R_k \}$ is bounded.  If $\gamma = 0$, the agent is "myopic" in being concerned only to maximize (3.8) by separately maximizing each immediate reward.  But in general, acting to maximize immediate reward can reduce access to future rewards so that the return is reduced.  As $\gamma$ approaches 1, the return objective takes future rewards into account more strongly, the agent becomes more farsighted.

Returns at successive time steps are related to each other in a way that is important for the theory and algorithms of reinforcement learning:

$\begin{flalign}
G_t &\doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \cdots \\
&= R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \cdots) \tag{using definition with variable replacement}\\
&= R_{t+1} + \gamma G_{t+1}
\end{flalign}$

Note that this works for all time steps $t \lt T$, even if termination occurs at $t+1$, provided we define $G_T = 0$.

Note that although the return (3.8) is a sum of an infinite number of terms, it is still finite if the return is nonzero and constant if $\gamma \lt 1$.  For example, if the reward is a constant +1, then the return is

$G_t = \sum_{k=0}^\infty \gamma^k = \frac{1}{1-\gamma} \tag{3.10}$
"""

# ╔═╡ 12fa078b-6b5a-4c25-97a2-38fd9d65cf52
md"""
> ### *Exercise 3.5*
> The equations in Section 3.1 are for the continuing case and need to be modified (very slightly) to apply to episodic tasks. Show that you know the modifications needed by giving the modified version of (3.3).

From Section 3.1 we have the following equations.  Written below each is the modified version for episodic tasks and an explanation:

$\begin{flalign}

&S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, \dots \tag{3.1} \\
&S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T, S_T \tag{3.1'}
\end{flalign}$

Instead of continuing indefinitely, in an episodic task there is always a transition into a terminal state at step $T$ which concludes the episode.  No actions or future transitions occur after reaching $S_T$

$\begin{flalign}
p(s^\prime, r \vert s, a) &\doteq \Pr \{ S_t = s^\prime, R_t = r \mid S_{t-1} = s, A_{t-1} = a \}, \forall s^\prime, s \in \mathcal{S}, r \in \mathcal{R}, a \in \mathcal{A}(s). \tag{3.2} \\
p(s^\prime, r \vert s, a) &\doteq \Pr \{ S_t = s^\prime, R_t = r \mid S_{t-1} = s, A_{t-1} = a \}, \forall s \in \mathcal{S}, s^\prime \in \mathcal{S}^+, r \in \mathcal{R}, a \in \mathcal{A}(s) \tag{3.2'} \\
p(s^\prime, 0 \vert s, a) &\doteq 1, \forall s, s^\prime \in \mathcal{S}^+ \setminus \mathcal{S}, a \in \mathcal{A}(s) \tag{3.2''}\\
p(s^\prime, r \vert s, a) &\doteq 0, \forall s \in \mathcal{S}^+ \setminus \mathcal{S}, s^\prime \in \mathcal{S}, r \in \mathcal{R}, a \in \mathcal{A}(s)  \tag{3.2'''}\\
\end{flalign}$

The probability transition function for an episodic task is defined for all state action pairs in the set of nonterminal states.  However, for the transition states, we must also include the set of terminal states in the possibilities.  Moreover, if we begin with a terminal state, denoted above by the set difference operator $\mathcal{S}^+ \setminus \mathcal{S}$, then the only possible transitions are to the same terminal state with a reward of 0.  All other transitions are forbidden.  These extra equations are only needed to formally define why there is no need to write down transitions beyond the terminal state.  They have no impact on the discounted return.  Alternatively, one can choose to not even define the transition function from terminal states which is true of the definition 3.2'.  The other two equations are only included for completeness.

$\begin{flalign}
\sum_{s' \in \mathcal{S}}\sum_{r \in \mathcal{R}}p(s',r|s,a) &= 1, \text{ for all } s\in\mathcal{S},a\in\mathcal{A}(s) \tag{3.3}\\
\sum_{s^ \prime \in \mathcal{S}^+}\sum_{r \in \mathcal{R}}p(s^ \prime,r|s,a) &= 1, \text{ for all } s\in\mathcal{S},a\in\mathcal{A}(s) \tag{3.3'}\\
\sum_{s^ \prime \in \mathcal{S}}\sum_{r \in \mathcal{R}}p(s^ \prime,r|s,a) &= \Pr \{ S_t \neq S_T \mid S_{t-1} = s, A_{t-1} = a \}, \text{ for all } s\in\mathcal{S},a\in\mathcal{A}(s) \tag{3.3''}\\
\end{flalign}$

To cover the entire probability space we must sum over every possible transition states which is the space $\mathcal{S}^+$. If we leave 3.3 unmodified, then this sum equals the probability of not entering a terminal state after a single step transition out of $s$ with action $a$.

$\begin{flalign}
p(s^\prime \vert s, a) &\doteq \Pr \{ S_t = s^\prime \mid S_{t-1} = s, A_{t-1} = a \} = \sum_{r \in \mathcal{R}} p(s^\prime, r \vert s, a) \tag{3.4}\\
\end{flalign}$

does not require any modification, but we must note that the first argument can take on terminal state values as well.

$\begin{flalign}
r(s, a) &\doteq \mathbb{E}[R_t \mid S_{t-1} = s, A_{t-1} = a] = \sum_{r \in \mathcal{R}} r \sum_{s^\prime \in \mathcal{S}} p(s^\prime, r \vert s, a), \tag{3.5} \\
r(s, a) &\doteq \mathbb{E}[R_t \mid S_{t-1} = s, A_{t-1} = a] = \sum_{r \in \mathcal{R}} r \sum_{s^\prime \in \mathcal{S}^+} p(s^\prime, r \vert s, a), \tag{3.5'} \\
\end{flalign}$

The sum over transition states must include terminal states.

$\begin{flalign}
r(s, a, s^\prime) &\doteq \mathbb{E}[R_t \mid S_{t-1} = s, A_{t-1} = a, S_t = s^\prime] = \sum_{r \in \mathcal{R}} r \frac{p(s^\prime, r \vert s, a)}{p(s^\prime \vert s, a)} \tag{3.6} \\
\end{flalign}$

does not require any modification, but we must note that the third argument can take terminal state values as well.
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

# ╔═╡ 5bd4bc89-4aa4-4513-ac0c-60f3aa062f0f
md"""
## 3.4 Unified Notation for Episodic and Continuing Tasks

$G_t \doteq \sum_{k = t+1}^T \gamma^{k-t-1}R_k \tag{3.11}$

including the possibility that $T = \infty$ or $\gamma = 1$ (but not both).
"""

# ╔═╡ 33665bd0-49be-4e93-acee-a8da79e1be77
md"""
## 3.5 Policies and Value Functions

A value function represents *how good* it is for the agent to be in a given state (or how good it is to perform a given action in a given state).  The notion of how good is defined in terms of the expected return.  Of course, the future rewards depend on the actions taken, so value functions are defined with respect to a policy.

A policy is a mapping from states to the probabilities of selecting each possible action.  If the agent is following a policy $\pi$ at time $t$, then $\pi(a \vert s)$ is the probability that $A_t = a$ if $S_t = s$.  Like $p$, $\pi$ is an ordinary function; the "|" in the middle of $\pi(a \vert s)$ merely reminds us that it defines a probability distribution over $a \in \mathcal{A}(s)$ for each $s \in \mathcal{S}$.  Reinforcement learning methods specify how the agent's policy is changed as a result of its experience.
"""

# ╔═╡ 7582815d-da34-41b6-90f4-b2602e7a81f3
md"""
> ### *Exercise 3.11* 
> If the current state is $S_t$, and actions are selected according to a stochastic policy $\pi$, then what is the expectation of $R_{t+1}$ in terms of $\pi$ and the four-argument function $p(s',r|s,a) \doteq \Pr\{S_t=s', R_t=r \mid S_{t-1}=s,A_{t-1}=a\}$

$$\mathbb{E}_\pi[R_{t+1} \mid S_t = s]=\sum_{r \in \mathcal{R}}r\times \Pr\{R_{t+1}=r \mid S_t=s, A_t \sim \pi(s) \}$$

$\begin{flalign}
\Pr\{R_{t+1}=r \mid S_t = s, A_t \sim \pi(s) \} &= \mathbb{E}_\pi \left [ \sum_{s^\prime \in \mathcal{S}}\Pr\{S_{t+1} = s^\prime, R_{t+1}=r \mid S_t = s\} \right ] \\
&= \sum_{a \in \mathcal{S}} \pi(a \vert s) \left [ \sum_{s^\prime \in \mathcal{S}}\Pr\{S_{t+1} = s^\prime, R_{t+1}=r \mid S_t = s, A_t = a \} \right ] \\
&=\sum_{a \in \mathcal{A(s)}}\pi(a|s)\sum_{s' \in \mathcal{S}}p(s', r|s,a) \\
&\therefore \\
\mathbb{E}_\pi[R_{t+1} \mid S_t = s] &=\sum_{r \in \mathcal{R}} \left[ r \times \left[ \sum_{a \in \mathcal{A(s)}}\pi(a|s) \left[ \sum_{s' \in \mathcal{S}}p(s', r|s,a) \right] \right] \right]
\end{flalign}$
"""

# ╔═╡ fe901beb-2fca-4850-9429-b27ec96d784e
md"""
---
"""

# ╔═╡ a4c34cb9-1195-4ab9-8036-fcc0de9f5ffb
md"""
The *value function* of a state $s$ under a policy $\pi$, denoted $v_\pi(s)$, is the expected return when starting in $s$ and following $\pi$ thereafter.  For MDPs, we can define $v_\pi$ formally by 

$v_\pi(s) \doteq \mathbb{E}_\pi [G_t \mid S_t = s ] = \mathbb{E}_\pi \left [ \sum_{k = 0}^\infty \gamma^k R_{t+k+1} \; \middle\vert \; S_t = s \right ], \forall s \in \mathcal{S}, \tag{3.12}$

where $\mathbb{E}_\pi[\cdot]$ denotes the expected value of a random variable given that the agent follows policy $\pi$, and $t$ is any time step.  Note that the value of the terminal state, if any, is always zero.  We call this function $v_\pi$ the *state-value function for policy $\pi$*.

Similarly, we define the value of taking action $a$ in state $s$ under a policy $\pi$, denoted $q_\pi(s, a)$, as the expected return starting from $s$, taking the action $a$, and thereafter following policy $\pi$:

$q_\pi(s, a) \doteq \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi \left [ \sum_{k = 0}^\infty \gamma^k R_{t+k+1} \; \middle\vert \; S_t = s, A_t = a \right ], \forall s \in \mathcal{S}, a \in \mathcal{A}(s). \tag{3.13}$

We call $q_\pi$ the *action-value function for policy $\pi$*.
"""

# ╔═╡ 2eb0bad0-7185-4745-8141-fe97201b06a4
md"""
> ### *Exercise 3.12* 
> Give an equation for $v_{\pi}$ in terms of $q_{\pi}$ and $\pi$.
"""

# ╔═╡ 6f9ea632-dfb8-4647-8550-78e4138317fd
md"""

$\begin{flalign}
v_{\pi}(s) &= \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \; \middle\vert \; S_t=s \right], \text{ for all } s \in \mathcal{S} \tag{3.12}\\
q_{\pi}(s, a) &= \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \; \middle\vert \; S_t=s, A_t = a \right], \text{ for all } s \in \mathcal{S}, a \in \mathcal{A}(s) \tag{3.13}\\
\end{flalign}$

Note that expected values have the following property:

$\begin{flalign}
\mathbb{E}[X] &= \sum_x x \times \Pr \{X = x \} \tag{expected value definition}\\
&= \sum_x x \sum_y \Pr \{X = x, Y = y \} \tag{joint probability definition}\\
&= \sum_x x \sum_y \Pr \{Y = y \} \Pr \{X = x \mid Y = y \} \tag{conditional probability definition}\\
&= \sum_y \Pr \{Y = y \} \sum_x x \times \Pr \{X = x \mid Y = y \} \tag{rearranging sum order}\\
&= \sum_y \Pr \{Y = y \} \mathbb{E}[X \mid Y = y] \tag{expected value definition}\\
\end{flalign}$

So we can always preserve an expected value by conditioning it on something and taking the sum of that conditional expectation weighted by the probability of each conditional value.  Returning to the two value function definitions, $q_\pi(s, a)$ is the expected value of $v_\pi(s)$ conditioned on a particular value of $A_t = a$.  Using the above property we can compute the expected value for $v_\pi(s)$ by summing $q_\pi(s)$ weighted by the probability of each action.  Those probabilities are given by policy function $\pi(a \vert s)$.

$v_\pi(s) = \sum_{a \in \mathcal{A}(s)} \pi(a \vert s)q_\pi(s, a)$
"""

# ╔═╡ db2157d9-abc1-43c1-8d37-0fe5e803667c
md"""
> ### *Exercise 3.13* 
> Give an equation for $$q_{\pi}$$ in terms of $$v_{\pi}$$ and $$p(s',r|s,a)$$

From (3.13) we have

$\begin{flalign}
q_{\pi}(s,a) &= \mathbb{E}_{\pi} \left[ G_t \mid S_t=s,A_t=a\right] \\
&= \mathbb{E}_{\pi} \left[ R_{t+1} + \gamma G_{t+1} \mid S_t=s,A_t=a\right] \tag{separating first sum term}\\
&= \sum_{s^\prime, r} r \times p(s^\prime, r \vert s, a) + \gamma \mathbb{E}_{\pi} \left[ G_{t+1} \mid S_t=s,A_t=a\right] \tag{expected value definition}\\
&= \sum_{s^\prime, r} r \times p(s^\prime, r \vert s, a) +  \gamma \sum_{s^\prime, r} p(s^\prime, r \vert s, a) \mathbb{E}_{\pi} \left[ G_{t+1} \mid S_{t+1}=s^\prime \right] \tag{conditional expectation definition}\\
&= \sum_{s^\prime, r} r \times p(s^\prime, r \vert s, a) +  \gamma \sum_{s^\prime, r} p(s^\prime, r \vert s, a) v_\pi(s^\prime) \tag{value function definition}\\
&= \sum_{s^\prime, r} p(s^\prime, r \vert s, a) (r + \gamma v_\pi(s^\prime))\\
\end{flalign}$

Note that $G_{t+1}$ only depends on $S_{t+1}$ and not any previous state.  However, we must evaluate the probability of every possible value of $S_{t+1}$ given the assumption that $S_t = s$ and $A_t = a$.  Those probabilities are given by $p(s^\prime \vert s, a) = \sum_r p(s^\prime, r \vert s, a)$ which is seen in the fourth line above.  We can then omit the condition on $S_t$ and $A_t$ since it is present in the probability transition function.
"""

# ╔═╡ fb893169-9457-4ce7-9c92-2d55bd8e7295
md"""
We can derive recursive relationships for value functions similar to what we have already derived for the return (3.9).  For any policy $\pi$ and any state $s$, the following consistency condition holds between the value of $s$ and the value of its possible successor states:

$\begin{flalign}
v_\pi(s) &\doteq \mathbb{E}_\pi[G_t \mid S_t = s]\\
&= \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s] \tag{by (3.9)}\\
&= \sum_a \pi(a \vert s) \sum_{s^\prime} \sum_r p(s^\prime, r \vert s, a) \left [r + \gamma \mathbb{E}_\pi[G_{t+1} \vert S_{t+1} = s^\prime]\right] \\
&= \sum_a \pi(a \vert s) \sum_{s^\prime, r} p(s^\prime, r \vert s, a) \left [r + \gamma v_\pi(s^\prime)\right], \text{ for all } s \in \mathcal{S} \tag{3.14} \\
\end{flalign}$

The final expression is an expected value where we sum over the three variables of $a, s^\prime, r$ and compute the return observed for every combination.  Equation (3.14) is the *Bellman equation for* $v_\pi$.  It expresses a relationship between the value of a state and the values of its successor states.

The value function $v_\pi$ is the unique solution to its Bellman equation.  We show in subsequent chapters how this Bellman equation forms the basis of a number of ways to compute, approximate, and learn $v_\pi$.
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
#=╠═╡
@bind params PlutoUI.combine() do Child
	md"""
	Discount Factor $\gamma$: $(Child(:γ, Slider(0.0:0.001:1.0, default = 0.9, show_value = true)))
	
	Terminal Step: $(Child(:t_final, Slider(10:1000, default = 101, show_value = true)))
	"""
end
  ╠═╡ =#

# ╔═╡ e4d73777-cf4b-40c5-8922-b9df28d25aa0
#=╠═╡
begin
	t_final = params.t_final
	t = 0:t_final
	f(t, t_final, γ) = (γ^(t_final - t) - 1) / (γ - 1)
	plot(t, f.(t, t_final, params.γ), Layout(xaxis_title = "Time Step", title = "Episodic Reward Factor for $t_final Step Episode"))
end
  ╠═╡ =#

# ╔═╡ ede978b8-dd9d-4b26-88c1-7def0dae42ee
md"""
On early timesteps the additional value is near zero because the numerator takes γ to a large power (unless the length of the episode is very small).  This means the reward values would all be shifted up by the same amount as in the continuing case but uniformly  As we get closer to the end of the episode, the factor approaches 1 for the step right before the terminal state.  This reflects the expectation of receiving one more c value before the episode ends.  The values near the beginning of the episode match the expectation of receiving the added c value indefinitely.   

For the agent's value function, the expected value of the return is what is relevant:

$\mathbb{E}_\pi[G^\prime_t \mid S_t = s] = \mathbb{E}_\pi[G_t \mid S_t = s] + \frac{c}{1 - \gamma}\left [1 - \mathbb{E}_\pi[\gamma^{T-t} \mid S_t = s] \right ]$

The impact on the value function will be:

$v^\prime_\pi(s) = v_\pi(s) + \frac{c}{1 - \gamma}\left [1 - \mathbb{E}_\pi[\gamma^{T-t} \mid S_t = s] \right ]$

So it seems that rather than each value being shifted by a constant, there is also a factor that depends on an expected value related to the number of steps until termination.  Since each state could be a different number of steps away from termination, the value function will fundamentally change.  States that are far from termination will be valued higher by a factor close to $\frac{c}{1 - \gamma}$ whereas states that are close to termination will have a value that is increased by a smaller factor.  This effect is reversed if c is negative.  Let's say that c is large and positive in a maze task where exiting the maze produces a reward of 1+c and terminates the episode while all other transitions produce a reward of c.  Normally, the states close to the exit would have the highest value, but the agent will accumulate the most reward far away from the exit since it will have more chances to accumulate values of c.
"""

# ╔═╡ b5871733-c403-4b39-8b51-2f3941c8a634
md"""
> ### *Exercise 3.17* 
> What is the Bellman equation for action values, that is, for $q_{\pi}$? It must give the action value $q_{\pi}(s,a)$ in terms of the action values, $q_{\pi}(s',a')$, of possible successors to the state-action pair $(s,a)$.  
> Hint: The backup diagram to the right corresponds to this equation. Show the sequence of equations analogous to (3.14), but for action values.

Following the example in (3.14) but for $q_{\pi}(s, a)$ intsead of $v_{\pi}(s)$ we have:

$\begin{flalign}
q_{\pi}(s,a) & \doteq \mathbb{E}_\pi [G_t \mid S_t=s,A_t=a] \\
&= \mathbb{E}_\pi [R_{t+1}+\gamma G_{t+1} \mid S_t=s,A_t=a] \tag{by (3.9)} \\
&=\sum_{s',r} p(s',r|s,a)\left[r+\gamma\mathbb{E}_\pi [G_{t+1} \mid S_{t+1}=s'] \right] \tag{expected value definition}\\
&=\sum_{s',r} p(s',r|s,a)\left[r+\gamma \sum_{a'} \pi(a', s')\mathbb{E}_\pi [G_{t+1} \mid S_{t+1}=s',A_{t+1}=a'] \right] \tag{policy expectation definition}\\
&=\sum_{s',r} p(s',r|s,a)\left[r+\gamma \sum_{a'} \pi(a', s')q_{\pi}(s',a') \right], \text{ for all } s \in \mathcal{S}, a \in \mathcal{A(s)} \tag{state-action value definition}\\
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

The *optimal state-value function* is defined as:

$v_*(s) \doteq \max_\pi v_\pi(s), \forall s \in \mathcal{S} \tag{3.15}$

Optimal policies also share the same *optimal action-value function*, defined as:

$q_*(s, a) \doteq \max_\pi q_\pi(s, a), \forall s \in \mathcal{S}, a \in \mathcal{A}(s) \tag{3.16}$

For the state-action pair $(s, a)$, this function gives the expected return for taking action $a$ in state $s$ and thereafter following an optimal policy.  Thus, we can write $q_*$ in terms of $v_\pi$ as follows:

$q_*(s, a) = \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a]. \tag{3.17}$

Because $v_*$ is the value function for a policy, it must satisfy the self-consistency condition given by the Bellman equation for state values (3.14).  Because it is the optimal value function, however, $v_*$'s consistency condition can be written in a special form without reference to any specific policy.  This is the Bellman equation for $v_*$, or the *Bellman optimality equation*.  Intuitively, the Bellman optimality equation expresses the fact that the value of a state under an optimal policy must equal the expected return for the best action from that state:

$\begin{flalign}
v_*(s) &= \max_{a \in \mathcal{A}(s)} q_{\pi_*}(s, a) \\
&= \max_a \mathbb{E}_{\pi_*}[G_t \mid S_t = s, A_t = a] \tag{definition of q}\\
&= \max_a \mathbb{E}_{\pi_*}[R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a] \tag{by (3.9)}\\
&= \max_a \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a] \tag{3.18}\\
&= \max_a \sum_{s^\prime, r} p(s^\prime, r \vert s, a)[r + \gamma v_*(s^\prime)] \tag{3.19}
\end{flalign}$

The last two equations follow from the definitions of the expected value and the existence of an optimal policy.  They are two forms of the Bellman optimality equation for $v_*$.  The Bellman optimality equation for $q_*$ is:

$\begin{flalign}
q_*(s, a) &= \mathbb{E} \left [ R_{t+1} + \gamma \max_{a^\prime} q_*(S_{t+1}, a^\prime) \; \middle\vert \; S_t = s, A_t = a \right ]\\
&= \sum_{s^\prime, r} p(s^\prime, r \vert s, a) \left [ r + \gamma \max_{a^\prime} q_*(s^\prime, a^\prime) \right ] \tag{3.20}
\end{flalign}$

For finite MDPs, the Bellman optimality equation for $v_*$ (3.19) has a unique solution.  The Bellman optimality equation is actually a system of equations, one for each state, so if there are $n$ states, then there are $n$ equations and $n$ unknowns.  If the dynamics $p$ of the environment are known, then in principle one can solve this system of equations for $v_*$ using any of a variety of methods for solving systems of nonlinear equations.  One can solve a related set of equations for $q_*$.

Once one has $v_*$, it is relatively easy to determine an optimal policy.  For each state $s$, there will be one or more actions at which the maximum is obtained in the Bellman optimality equation.  Any policy that assigns nonzero probability only to these actions is an optimal policy.  You can think of this as a one-step search.  If you have the optimal value function, $v_*$, then the actions that appear best after a one-step search will be optimal actions.  Another way of saying this is that any policy that is *greedy* with respect to the optimal evaluation function $v_*$ is an optimal policy.  The term greedy is used in computer science to describe any search or decision procedure that selects alternatives based only on local or immediate considerations, without considering the possibility that such a selection may prevent future access to even better alternatives.  Consequently, it describes policies that select actions based only on short-term consequences.  The beauty of $v_*$ is that if one uses it to evaluate the short-term consequences of actions -- specifically, the one-step consequences -- then a greedy policy is actually optimal in the long-term sense in which we are interested because $v_*$ already takes into account the reward consequences of all possible future behavior.  By means of $v_*$, the optimal expected long-term return is turned into a quantity that is locally and immediatley available for each state.  Hence, a one-step-ahead search yields the long-term optimal actions.

Having $q_*$ makes choosing optimal actions even easier.  With $q_*$, the agent does not even have to do a one-step-ahead search: for any state $s$, it cacn simply find any action that maximizes $q_*(s, a)$.  The action-value function effectively caches the results of all one-step-ahead searches.  It provides the optimal expected long-term return as a value that is locally and immediately available for each state-action pair.  **Hence, at the cost of representing a function of state-action pairs, instead of just states, the optimal action-value function allows optimal actions to be selected without having to know anything about the possible successor states and their values, that is, without having ot know anything about the environment's dynamics.**

Explicitely solving the Bellman optimality equation provides one route to finding an optimal policy, and thus to solving the reinforcement learning problem.  Often this is not possible either due to lack of knowledge about the environment or a lack of computational resources.  When we do not have these limitations, the methods of dynamic programming can be used to solve these equations in an iterative process which will be introduced in Chapter 4.  Many other methods introduced later can be understood as approximately solving the Bellman optimality equation, usuing actual experienced transitions in place of knowledge of the expected transitions.
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

# ╔═╡ 445163e2-c345-4679-acda-c8657c2b5564
md"""
## 3.7 Optimality and Approximation

Often agents cannot execute an optimal policy, but it is a useful theoretical ideal for agents to approximate.  We may have the resources to represent a problem and begin approaching a solution, but the time required to converge to an optimal policy may be impractically long.  In these cases, an agent must decide to step computing at a certain point to actually begin taking actions.  If a problem has a small enough state space, then such an approximate solution could be stored as a table of values in memory.  Such problems are called the *tabular* case and the corresponding methods are *tabular* methods.  If the number of states is too large to represent in this manner, then some sort of parameterized function representation may be required to cover the state space.

Since our value function approach approximates values for individual states, we have the option of putting more computational effort into states that occur frequently when following close to optimal behavior.  This prioritization is another way to make approximation more useful since we really only need accurate values for states that are visited while following the optimal policy.  If these states turn out to be a tiny fraction of the total, then we may have a very accurate approximation where it matters despite having a very poor representation of the entire value function.
"""

# ╔═╡ 4ca58fcf-3115-4100-9f83-b8a389e4eaa0
md"""
# Dependencies and Settings
"""

# ╔═╡ 5d96ee10-24c0-4bf5-82a1-9050200066c5
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
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoPlotly = "~0.3.9"
PlutoUI = "~0.7.52"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "7b68b9ca367436795e4fde755dda7ecca58c89d7"

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

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

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
# ╟─1c7d2750-3625-4eed-8408-53626b67d749
# ╟─e7a4f148-bf74-44f3-98e7-e4d9b4cac5ab
# ╟─c5abf826-9ce8-4319-a2e5-6cf7fcc61400
# ╟─85905a6e-1807-4b77-b313-dbadb8b898c8
# ╟─090c50ed-6772-457a-afbb-cf2cde0e2ec4
# ╟─f376a2e1-69d3-46ca-9bc2-33e447c6834b
# ╟─6a85a90d-f345-4604-9637-086a71928af5
# ╟─12fa078b-6b5a-4c25-97a2-38fd9d65cf52
# ╟─9ce34899-c8d0-428f-a3d3-91f0bf37198e
# ╟─9a3d9a79-e44a-40e5-b7bb-947f3327c598
# ╟─1b42d235-5f48-4625-a83e-48b50cbbc347
# ╟─30513c5c-20fe-4a31-9a59-67b73fa1e3a7
# ╟─08ad2c4f-a3c1-4d9b-aafc-9b2394f68f53
# ╟─5bd4bc89-4aa4-4513-ac0c-60f3aa062f0f
# ╟─33665bd0-49be-4e93-acee-a8da79e1be77
# ╟─7582815d-da34-41b6-90f4-b2602e7a81f3
# ╟─fe901beb-2fca-4850-9429-b27ec96d784e
# ╟─a4c34cb9-1195-4ab9-8036-fcc0de9f5ffb
# ╟─2eb0bad0-7185-4745-8141-fe97201b06a4
# ╟─6f9ea632-dfb8-4647-8550-78e4138317fd
# ╟─db2157d9-abc1-43c1-8d37-0fe5e803667c
# ╟─fb893169-9457-4ce7-9c92-2d55bd8e7295
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
# ╟─445163e2-c345-4679-acda-c8657c2b5564
# ╟─4ca58fcf-3115-4100-9f83-b8a389e4eaa0
# ╠═86d53794-2251-47d5-a45e-f1da53cd8ef5
# ╠═5d96ee10-24c0-4bf5-82a1-9050200066c5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
