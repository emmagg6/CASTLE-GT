# Creation on MARL system in CC2 and CC4

CC2: https://github.com/cage-challenge/cage-challenge-2 

This is 2 agent adversarial setup in the Cyber Operations Research Gym (CybORG). The two agents are PPO RL agents trainging/learning/optimising *together* against eachother - each agent takes steps in the environment before the other agent takes its turn training. This iterative, sequential scheme will take into acount an adversial 2-player game theoretic equilibrium computation analogous to the similified verion in the *DEMO* folder.


CC4: https://github.com/cage-challenge/cage-challenge-4


In this complex, dynamic environment that is a random size (initialized randomly at start) and provides an asyncronous setting due to assymetric time-ticks for actions, an approximate **course correllated equilibrium** will be computed "on top of" the **5 cooperative MARL agents**, taking the adversarial invaders into account as part of the environment. 

Shared plan for this set up: https://docs.google.com/document/d/1TatAvdc4zCKJRPvLwYVwlKADEZPRA3ntakyPmh830jk/edit
Shared write-up: https://www.overleaf.com/project/660c8491198c0d7cd9334f72 

For CybORG setup, refer to 'cd ~/cage-challenge-4/documentation/docs/pages/tutorials/01_Getting_Started/1_Introduction.md'