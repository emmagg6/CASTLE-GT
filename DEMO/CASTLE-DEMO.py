import pyspiel
import random
import numpy as np
import pygame
import sys

from pygameENV import Environment
from osGAME import APIState
from osGAME import APIGame


pygame.init()

################ PYGAME SETTINGS ##############
GRAY = (200, 200, 200)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
HOST_RADIUS = 20
SERVER_SIZE = 30

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Network Security Game")
clock = pygame.time.Clock()

################ INITIALIZATOINS ###############
player_count = 2
host_count = 3
critical_count = 2
connections = np.array([[0, 1, 0, 1, 0],
                        [1, 0, 1, 0, 1],
                        [0, 1, 0, 0, 1],
                        [1, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0]])
infected_hosts = np.array([0,1,0])
infected_criticals = np.array([0,0])
w_blue = np.array([0.5, 1, 0.3, 0.2])
w_red = np.array([0.75, 1.0])

################ ENVIRONMENT #################

env = Environment(player_count, host_count, critical_count, 
                  connections, infected_hosts, infected_criticals,
                  w_blue, w_red)

################ GAME ################
security_game = APIGame(env, max_rounds=5)

state = security_game.new_initial_state()
env.draw_network()

################ ENVIRONMENT #################

env = Environment(player_count, host_count, critical_count, 
                  connections, infected_hosts, infected_criticals,
                  w_blue, w_red)

################ GAME ################
security_game = APIGame(env, max_rounds=5)

state = security_game.new_initial_state()
env.draw_network()


action_blue = None
target_blue = None
action_red = None
target_red = None

running = True

while running and not state.is_terminal():
    # 'draw'' the environment and available actions
    env.draw_network()
    if state.current_player() == 0:
        env.draw_text("Your turn, select an action", (WIDTH // 2, HEIGHT - 50))
        actions_targets = state.legal_actions_on_hosts(0)
        for idx, (action, targets) in enumerate(actions_targets.items()):
            env.draw_text(f"{idx + 1}: {action} -> {targets}", (WIDTH // 2, HEIGHT - 30 * (idx + 2)))
    pygame.display.flip()
    
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and state.current_player() == 0:
            mouse_pos = pygame.mouse.get_pos()
            # need to implement the logic to check if the mouse click is on one of the action texts
            # once obtained, set action_blue to that action and ask for the host to target
            
            # example: action_blue, target_blue = get_action_and_target_from_mouse_position(mouse_pos, actions_targets)
            
        # get Red agent's random everything
        elif state.current_player() == 1:
            actions_targets = state.legal_actions_on_hosts(1)
            action_red = random.choice(list(actions_targets.keys()))
            target_red = random.choice(actions_targets[action_red])
    
    if action_blue is not None and target_blue is not None:
        state.apply_actions((action_blue, action_red), (target_blue, target_red))
        action_blue, target_blue, action_red, target_red = None, None, None, None  # Reset actions and targets
print(f"End of Game. Resulting Returns: {state.returns()}")