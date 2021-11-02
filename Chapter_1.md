*Exercise 1.1: Self-Play* Suppose, instead of playing against a random opponent, the
reinforcement learning algorithm described above played against itself, with both sides
learning. What do you think would happen in this case? Would it learn a di↵erent policy
for selecting moves?



*Exercise 1.2: Symmetries* Many tic-tac-toe positions appear di↵erent but are really
the same because of symmetries. How might we amend the learning process described
above to take advantage of this? In what ways would this change improve the learning
process? Now think again. Suppose the opponent did not take advantage of symmetries.
In that case, should we? Is it true, then, that symmetrically equivalent positions should
necessarily have the same value?


*Exercise 1.3: Greedy Play* Suppose the reinforcement learning player was greedy, that is,
it always played the move that brought it to the position that it rated the best. Might it 
learn to play better, or worse, than a nongreedy player? What problems might occur?

*Exercise 1.4: Learning from Exploration* Suppose learning updates occurred after all
moves, including exploratory moves. If the step-size parameter is appropriately reduced
over time (but not the tendency to explore), then the state values would converge to
a di↵erent set of probabilities. What (conceptually) are the two sets of probabilities
computed when we do, and when we do not, learn from exploratory moves? Assuming
that we do continue to make exploratory moves, which set of probabilities might be better
to learn? Which would result in more wins?

*Exercise 1.5: Other Improvements* Can you think of other ways to improve the reinforcement
learning player? Can you think of any better way to solve the tic-tac-toe problem
as posed? ⇤