import pyspiel
import random
import numpy as np

from MARL.environment import Environment
from MARL.game import APIState
from MARL.game import APIGame

from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import tabular_qlearner

################ INITIALIZATOINS ###############
player_count = 2
host_count = 3
critical_count = 2
connections = np.array([[0, 1, 0, 1, 0],
                        [1, 0, 1, 0, 1],
                        [0, 1, 0, 0, 1],
                        [1, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0]])
infected_hosts = np.array([0,1,1])
infected_criticals = np.array([0,0])
w_blue = np.array([0.5, 1, 0.3, 0.2])
w_red = np.array([0.75, 1.0])


################ ENVIRONMENT #################

env = Environment(player_count, host_count, critical_count, 
                  connections, infected_hosts, infected_criticals,
                  w_blue, w_red)


############# AGENTS ########
player_count = 2
num_actions_per_agent = [5, 2]

agents = [
    tabular_qlearner.QLearner(player_id = idx, num_actions = num_actions_per_agent[idx])
    for idx in range(player_count)
]

################ GAME ################
security_game = APIGame(env, max_rounds=5)

state = security_game.new_initial_state()


action_blue = None
target_blue = None
action_red = None
target_red = None

running = True




while running:
    pass
