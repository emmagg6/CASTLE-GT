# CASTLE-GT
Game Theory Model for heuristic of the CASTLE MARL interaction in adversarial security networks.

From most to least basic:

> In the folder *basic-GameTheory-SetUp* contains some basic (first pass), very simple, versions of the 2-player interaction with game theoretic dynamics.

> In the folder *DEMO* is a pygame interface of an OpenSpiel game set-up for the DARPA CASTLE PI meeting March 2024.

> In the folder *RL* is the current worl towards a trained PPO RL agent integration into CC2: https://github.com/cage-challenge/cage-challenge-2. In this dynamic 2-player environment of a consistant topology, an approximate course correllated equilibrium is to be computed "on top of" the PPO Agent, taking the adversarial invader into account as part of the environment.

> In the folder *MARL* is the current work toward a trained MARL integration into CC4: https://github.com/cage-challenge/cage-challenge-4. In this complex, dynamic environment that is a random size (initialized randomly at start) and provides an asyncronous setting due to assymetric time-ticks for actions, an approximate course correllated equilibrium will be computed "on top of" the MARL agents, taking the adversarial invaders into account as part of the environment. 
