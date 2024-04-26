## RL Folder for CCE in Single Agent, Dynamic Environment of DARPA's CASTLE CC2 security network.


CC2 is decribed in the CybORG/README.md and shows an adversarial, dynamic network environment with a consistant topology and single-time actions. 

To install CybORG (v2) : https://github.com/cage-challenge/cage-challenge-2/blob/main/CybORG/CybORG/Tutorial/0.%20Installation.md 


### IN THIS FOLDER

This RL folder contains a 'implementation' folder and 'cage-challenge-2-main' adapted from our private repo, as well as a 'model-EQapproximation'. 
> The 'implementation' folder contains raining and evaluating methods for custom PPO Blue Agents and Approximate CCE methods. 
> The 'cage-challenge-2-main' folder contains a 'fresh' import of CybORG from CC2 -- added as a submodule from the CASTLE CHALLENGE 2 repository in the 'CyBORG' folder.
> The 'model-EQapproximation' folder contains a small model environment and Q-learning Blue agent as well as implementations of the classic EXP3-IX algorithm, the EXP3-IX algorithm not involved in action-selection but view from 'above' the Q-learner's actions, and the extended EXP3-IX algorithm to large networks and unknown action spaces also running above an RL agent (the method created and tested at scale for the CC2 adaptation).



