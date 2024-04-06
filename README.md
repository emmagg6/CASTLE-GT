# CASTLE-GT
Game Theory Model for heuristic of the CASTLE MARL interaction in the 2-player adversarial game.

In the folder *GameTheory-SetUp* contains some basic (first pass), very simple, versions of the 2-player interaction game with game theoretic dynamics.

In the folder *DEMO* is the current work toward a pygame interface of an OpenSpiel game set-up for the DARPA PI meeting.

In the folder *MARL* is the current work toward a trained MARL integration into CC4: https://github.com/cage-challenge/cage-challenge-4. In this complex, dynamic environment that is a random size (initialized randomly at start) and provides an asyncronous setting due to assymetric time-ticks for actions, a approximate course correllated equilibrium will be computed "on top of" the MARL agents, taking the adversarial invaders into account as part of the environment. 
