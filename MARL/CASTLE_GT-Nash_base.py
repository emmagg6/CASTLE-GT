import pyspiel
import random
import numpy as np

from MARL.environment import Environment
from MARL.game import APIState
from MARL.game import APIGame

env = Environment(
        num_players=2,
        num_hosts=3,
        num_criticals=2,
        connection_matrix=np.array([
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0]
        ]),
        initial_hosts_infected=np.array([0, 1, 0]),
        initial_criticals_infected=np.array([0, 0]),
        weights_blue=np.array([0.5, 1, 0.3, 0.2]),
        weights_red=np.array([0.75, 1.0])
)

# AGENTS : Right now, the red agent will be a random agent, and the blue agent will be another random agent.

# Instantiate the game with the appropriate parameters
security_game = APIGame(env, max_rounds=5)

# You can interact with the cyber security game using OpenSpiel's API
state = security_game.new_initial_state()
while not state.is_terminal():
    actions_targets = [state.legal_actions_on_hosts(player) for player in range(security_game.num_players())]
    # print(actions_targets)  
    joint_action = [random.choice(list(actions_targets[player].keys())) for player in range(security_game.num_players())]
    # print(joint_action)
    target_hosts = [random.choice(actions_targets[player][joint_action[player]]) for player in range(security_game.num_players())]
    # print(target_hosts)
    state.apply_actions(joint_action, target_hosts)
    print(state)

print(f"End of Game. Resulting Returns: {state.returns()}")