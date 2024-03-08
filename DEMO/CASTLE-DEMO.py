# import pyspiel
import random
import numpy as np
import pygame
import sys

from pygameENV import Environment, PyGameInterface
from osGAME import APIState
from osGAME import APIGame


pygame.init()

################ PYGAME SETTINGS ##############
GRAY = (200, 200, 200)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GREEN = (0, 120, 0)
HOST_RADIUS = 20
SERVER_SIZE = 30

WIDTH, HEIGHT = 800, 600
# screen = pygame.display.set_mode((WIDTH, HEIGHT))

# # Initialize screen
# screen = pygame.display.set_mode((WIDTH, HEIGHT))
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
game_interface = PyGameInterface(env)

state = security_game.new_initial_state()
game_interface.draw_network()

action_blue = None
target_blue = None
action_red = None
target_red = None

running = True


while running and not state.is_terminal():
    # 'draw'' the environment and available actions
    # env.draw_network()

    if state.current_player() == 0:
        ######## HEADING #########
        game_interface.draw_text("Security Game", (WIDTH//2, HEIGHT//20), align='center', font_size = 50)
        game_interface.draw_text("Goal: Prevent the infection of your network while minimizing operational costs.", 
                                (WIDTH//2, HEIGHT//20 + 30), align = "center", color = GREEN)

        ######## ACTIONS WITH HOST BUTTONS ##########
        game_interface.draw_text("Available Hosts for Each Action", (20, HEIGHT//2 - 50), align='midleft')
        for idx, action in enumerate(env.actions):
            game_interface.draw_text(f"{action}", (20, HEIGHT//2 + 30 * (idx)), align="midleft")
            for host_idx in range(env.H):
                if host_idx not in state.legal_actions_on_hosts(0).get(action, []):
                    game_interface.draw_text(f"{host_idx}", game_interface.get_button_position(action, host_idx), align="center", color = GRAY)
                else:
                    game_interface.draw_text(f"{host_idx}", game_interface.get_button_position(action, host_idx), align="center", color = GREEN)
    pygame.display.flip()
    
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and state.current_player() == 0:
            mouse_pos = pygame.mouse.get_pos()
            # need to implement the logic to check if the mouse click is on one of the action texts
            # once obtained, set action_blue to that action and ask for the host to target
            for action in env.actions:
                for host_idx in range(env.H):
                    #check if button has been pressed
                    button_width, button_height = game_interface.get_button_position(action, host_idx)
                    if button_width-game_interface.button_width//2 <= mouse_pos[0] <= button_width+game_interface.button_width//2 and button_height-game_interface.button_height//2 <= mouse_pos[1] <= button_height+game_interface.button_height//2: 
                    #check if the action chosen is valid
                        if host_idx in state.legal_actions_on_hosts(0).get(action, []):
                            #if so, we apply the action
                            action_blue = action
                            target_blue = host_idx
                            state._current_player = 1
   
    # get Red agent's random everything
    if state.current_player() == 1:
        actions_targets = state.legal_actions_on_hosts(1)
        action_red = random.choice(list(actions_targets.keys()))
        target_red = random.choice(actions_targets[action_red])
    
    if action_blue is not None and target_blue is not None:
        state.apply_actions((action_blue, action_red), (target_blue, target_red))

        game_interface.draw_text(f"Selected {action_blue} on Host {target_blue}", (WIDTH//4, HEIGHT//10), align='midleft', color=GRAY)
        pygame.time.wait(100)
        game_interface.draw_text(f"Adversary chose {action_red} on Host {target_red}", (WIDTH//4, HEIGHT//10 + 20), align='midleft', color=GRAY)
        pygame.time.wait(100)

        game_interface.draw_network()
        action_blue, target_blue, action_red, target_red = None, None, None, None  # Reset actions and targets
        state._current_player = 0


print(f"End of Game. Resulting Returns: {state.returns()}")