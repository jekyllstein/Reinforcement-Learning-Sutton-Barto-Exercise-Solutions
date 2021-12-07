*Exercise 2.1* In $\epsilon$-greedy action selection, for the case of two actions and " = 0.5, what is
the probability that the greedy action is selected?

> If there are two possible actions then there are 3 possibilities.  With 50% probability the greedy action is selected and with 50% probability a random action is selected out of the possible 2.  So the probability that the greedy action is selected is $0.5 + 0.5 \times 0.5 = 0.75$.

*Exercise 2.2:* *Bandit example* Consider a k-armed bandit problem with k = 4 actions,
denoted 1, 2, 3, and 4. Consider applying to this problem a bandit algorithm using
$\epsilon$-greedy action selection, sample-average action-value estimates, and initial estimates
of Q1(a) = 0, for all a. Suppose the initial sequence of actions and rewards is A1 = 1,
R1 = −1, A2 = 2, R2 = 1, A3 = 2, R3 = −2, A4 = 2, R4 = 2, A5 = 3, R5 = 0. On some
of these time steps the $\epsilon$ case may have occurred, causing an action to be selected at
random. On which time steps did this definitely occur? On which time steps could this
possibly have occurred?

> On step 1 all the estimates are equal so there is no distinction between the $\epsilon$ case and the greedy case.  On step 2 if all actions were initialized with 0 value estimate then the greedy selection would be indifferent to actions 2, 3, 4 but a selection of action 2 is still possible with the $\epsilon$ case.  On step 3, the value estimate for action 2 is greater than any other because it is the only action to have experienced a positive reward.  Since step 3 saw an action of 2 that is the greedy choice but also could have been chosen in the $\epsilon$ case.  On step 4, action 2 has a negative value estimate so the greedy choice is indifferent between 3 and 4.  Since action 2 was selected it cannot be the greedy case.  On step 5, action 2 is the only action with a positive estimate.  Since action 3 was selected on step 5 this could only have occurred in the $\epsilon$ case so it cannot be the greedy case.

*Exercise 2.3* In the comparison shown in Figure 2.2, which method will perform best in
the long run in terms of cumulative reward and probability of selecting the best action?
How much better will it be? Express your answer quantitatively.

> In figure 2.2 the average reward for up to 1000 steps is shown.  At the end of 1000 steps the cumulative reward can be calculated by multiplying the average reward per step by the final value of the curve.  The blue curve for $\epsilon$ = 0.1 has the highest final value so it has the largest cumulative reward for this number of steps at about 1,540 vs the $\epsilon$ = 0 greedy strategy at about 1,000.  By the end of 1000 steps the $\epsilon$ = 0.1 strategy also selects correct actions about 80% of the time vs only about 50% for $\epsilon$ = 0.01 and less than 40% for $\epsilon$ = 0.

*Exercise 2.4* If the step-size parameters, $\alpha_n$, are not constant, then the estimate $Q_n$ is
a weighted average of previously received rewards with a weighting different from that
given by (2.6). What is the weighting on each prior reward for the general case, analogous
to (2.6), in terms of the sequence of step-size parameters?

> From (2.6):  $Q_{n+1} = Q_n + \alpha[R_n - Q_n]$ so here we consider the case where $\alpha$ is not a constant but rather can have a unique value for each step n.
>
> $Q_{n+1}$   $=Q_n + \alpha_n[R_n - Q_n]$
>
> ​			$=\alpha_nR_n+(1-\alpha_n)Q_n$
>
> ​			$=\alpha_nR_n+(1-\alpha_n)[\alpha_{n-1}R_{n-1}+(1-\alpha_{n-1})Q_{n-1}]$
>
> ​			$=\alpha_nR_n+(1-\alpha_n)\alpha_{n-1}R_{n-1}+(1-\alpha_n)(1-\alpha_{n-1})Q_{n-1}$
>
> ​			$=\alpha_nR_n+(1-\alpha_n)\alpha_{n-1}R_{n-1}+(1-\alpha_n)(1-\alpha_{n-1})[\alpha_{n-2}R_{n-2}+(1-\alpha_{n-2})Q_{n-2}]$
>
> ​			$=\alpha_nR_n+(1-\alpha_n)\alpha_{n-1}R_{n-1}+(1-\alpha_n)(1-\alpha_{n-1})\alpha_{n-2}R_{n-2}+(1-\alpha_n)(1-\alpha_{n-1})(1-\alpha_{n-2})Q_{n-2}$
>
> ​			$=Q_1\prod_{i=1}^n(1-\alpha_i)+\alpha_nR_n+\sum_{i=1}^{n-1}[(R_i\alpha_i)\prod_{j=i+1}^n(1-\alpha_j)]$
>
> For example if $\alpha_i=1/i$ then the formula becomes:
>
> $Q_{n+1}$ 	$= R_n/n + \sum_{i=1}^{n-1}[(R_i/i)\prod_{j=i+1}^n(1-1/j)]$
>
> ​			$= R_n/n + \sum_{i=1}^{n-1}[(R_i/i)\prod_{j=i+1}^n\frac{j-1}{j}]$
>
> ​			$= R_n/n + \sum_{i=1}^{n-1}[\frac{R_i}{i}\frac{i}{i+1}\frac{i+1}{i+2}...\frac{n-1}{n}]$
>
> from the expanded product we can see that all of the numerators and denominators cancel out leaving only $\frac{R_i}{n}$ which we expect for tihs form of $\alpha$ which was derived earlier for a simple running average.  To see a concrete example of how the product is  always equal to $i/n$, let's consider $i=n-1$.  Then there is only one term in the product for $j=n$ which is $\frac{n-1}{n}$ so the total expression in the sum for that term is $\frac{R_{n-1}}{(n-1)} \frac{n-1}{n}=R_{n-1}/n$ as expected.  

