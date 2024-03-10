# import pyspiel
import random
import numpy as np
import pygame
import sys

from pygameENV import Environment
from osPYGAME import PyGameInterface
from osGAME import APIState
from osGAME import APIGame


pygame.init()

################ PYGAME SETTINGS ##############
LIGHTGREY = (175, 175, 175)
GREY = (100, 100, 100)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GREEN = (0, 100, 0)
RED_agent = (100, 0, 0)
BLUE_agent = (0, 0, 100)


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
infected_hosts = np.array([0,1,1])
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




while running:
    game_interface.draw_text("Security Game", (WIDTH//2, HEIGHT//20), align='center', font_size = 50, font = 'bold')
    game_interface.draw_text("Goal: Prevent the infection of your network while minimizing operational costs.", 
                        (WIDTH//2, HEIGHT//20 + 50), align = "center", color = GREEN)
    ######## ACTIONS WITH HOST BUTTONS ##########
    game_interface.draw_text("Available Hosts for Each Action", (20, HEIGHT//2 - 50), align='midleft')
    for idx, action in enumerate(env.actions):
        game_interface.draw_text(f"{action}", (20, HEIGHT//2 + 30 * (idx)), align="midleft")
        for host_idx in range(env.H):
            if host_idx not in state.legal_actions_on_hosts(0).get(action, []):
                game_interface.draw_selection_button(action = action, host = host_idx, color = LIGHTGREY)
                game_interface.draw_text(f"H{host_idx}", game_interface.get_button_position(action, host_idx), align="center", color = GREY)
            else:

                game_interface.draw_selection_button(action = action, host = host_idx, color = GREEN)
                game_interface.draw_text(f"H{host_idx}", game_interface.get_button_position(action, host_idx), align="center", color = GREEN)
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
                            state.apply_action(action_blue, target_blue)
                            state._current_player = 1

    # Checking if terminal now just incase all infected have now been cleared
    if state.is_terminal():
        game_interface.draw_text(f"Round Score",
                                    (WIDTH - 120, HEIGHT//5), align='center', font='bold', color=BLACK)
        game_interface.draw_text(f"( {state.rewards()[0]:.2f}, ",
                                    (WIDTH - 150, HEIGHT//5 + 20), align='center', font='bold', color=BLUE_agent)
        game_interface.draw_text(f"{state.rewards()[1]:.2f} )",
                                    (WIDTH - 75, HEIGHT//5 + 20), align='center', font='bold', color=RED_agent)

        ############ Update Running Return ############
        game_interface.draw_text(f"GAME RETURN",
                                    (WIDTH//2, HEIGHT - HEIGHT//6), align='center', font='bold', color=BLACK, font_size=30)
        game_interface.draw_text(f"( {state.returns()[0]:.2f}, ",
                                    (WIDTH//2 - 50, HEIGHT - HEIGHT//10), align='center', font='bold', color=BLUE_agent,  font_size=30)
        game_interface.draw_text(f"{state.returns()[1]:.2f} )",
                                    (WIDTH//2 + 50, HEIGHT - HEIGHT//10), align='center', font='bold', color=RED_agent,  font_size=30)
        pygame.display.flip()
        break

    ######## ACTIONS WITH HOST BUTTONS ##########
    game_interface.draw_text("Available Hosts for Each Action", (20, HEIGHT//2 - 50), align='midleft')
    for idx, action in enumerate(env.actions):
        game_interface.draw_text(f"{action}", (20, HEIGHT//2 + 30 * (idx)), align="midleft")
        for host_idx in range(env.H):
            if host_idx not in state.legal_actions_on_hosts(0).get(action, []):
                game_interface.draw_selection_button(action = action, host = host_idx, color = LIGHTGREY)
                game_interface.draw_text(f"H{host_idx}", game_interface.get_button_position(action, host_idx), align="center", color = GREY)
            else:

                game_interface.draw_selection_button(action = action, host = host_idx, color = GREEN)
                game_interface.draw_text(f"H{host_idx}", game_interface.get_button_position(action, host_idx), align="center", color = GREEN)



    # get Red agent's random everything
    if state.current_player() == 1:
        actions_targets = state.legal_actions_on_hosts(1)
        action_red = random.choice(list(actions_targets.keys()))
        target_red = random.choice(actions_targets[action_red])
        state.apply_action(action_red, target_red)

    
    if action_blue is not None and target_blue is not None:
        state.payoff()

        game_interface.draw_network()

        game_interface.draw_text(f"{action_blue} selection on Host {target_blue}", 
                                 (20, HEIGHT//5), align='midleft', color=GREY)
        game_interface.draw_text(f"change in network connectivity of {env.Delta_connections} induced",
                                 (50, HEIGHT//5 + 20), align='midleft', color=LIGHTGREY, font='italic', font_size=15)
        game_interface.draw_text(f"a change in load of {env.l[target_blue]:.2f}  on the connections to the host",
                                 (50, HEIGHT//5 + 33), align='midleft', color=LIGHTGREY, font='italic', font_size=15)
        game_interface.draw_text(f"Adversary chose: {action_red} on Host {target_red}", (20, HEIGHT//5 + 70), align='midleft', color=GREY)
        action_blue, target_blue, action_red, target_red = None, None, None, None  # Reset actions and targets
        state._current_player = 0


    ############## Display Round Score #############
    game_interface.draw_text(f"Round Score",
                                (WIDTH - 120, HEIGHT//5), align='center', font='bold', color=BLACK)
    game_interface.draw_text(f"( {state.rewards()[0]:.2f}, ",
                                (WIDTH - 150, HEIGHT//5 + 20), align='center', font='bold', color=BLUE_agent)
    game_interface.draw_text(f"{state.rewards()[1]:.2f} )",
                                (WIDTH - 75, HEIGHT//5 + 20), align='center', font='bold', color=RED_agent)

    ############ Update Running Return ############
    game_interface.draw_text(f"GAME RETURN",
                                (WIDTH//2, HEIGHT - HEIGHT//6), align='center', font='bold', color=BLACK, font_size=30)
    game_interface.draw_text(f"( {state.returns()[0]:.2f}, ",
                                (WIDTH//2 - 50, HEIGHT - HEIGHT//10), align='center', font='bold', color=BLUE_agent,  font_size=30)
    game_interface.draw_text(f"{state.returns()[1]:.2f} )",
                                (WIDTH//2 + 50, HEIGHT - HEIGHT//10), align='center', font='bold', color=RED_agent,  font_size=30)


    if state.is_terminal():

        break

# print(f"End of Game. Resulting Returns: {state.returns()}")

game_interface.screen_fill()
score_reveal = True
while score_reveal == True :
    pygame.display.flip()

    ######## HEADING #########
    game_interface.draw_text("Security Game", (WIDTH//2, HEIGHT//20), align='center', font_size = 50, font = 'bold')
    game_interface.draw_text("Goal: Prevent the infection of your network while minimizing operational costs.", 
                            (WIDTH//2, HEIGHT//20 + 50), align = "center", color = GREEN)
    # pygame.display.flip()

    game_interface.draw_text(f"End Game", (WIDTH//2, HEIGHT//2), align="center", font_size= 50, color=BLACK, font='bold')
    # pygame.display.flip()

    ############ Update Running Return ############
    game_interface.draw_text(f"GAME RETURN",
                                (WIDTH//2, HEIGHT - HEIGHT//3), align='center', font='bold', color=BLACK, font_size=30)
    game_interface.draw_text(f"( {state.returns()[0]:.2f}, ",
                                (WIDTH//2 - 50, HEIGHT - HEIGHT//4), align='center', font='bold', color=BLUE_agent,  font_size=30)
    game_interface.draw_text(f"{state.returns()[1]:.2f} )",
                                (WIDTH//2 + 50, HEIGHT - HEIGHT//4), align='center', font='bold', color=RED_agent,  font_size=30)
    pygame.display.flip()

    # ok this does not work but trying to get it to exit once the screen is cliecked. but whatever just exit from terminal
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and state.current_player() == 0:
            mouse_pos = pygame.mouse.get_pos()
            if 0 <= mouse_pos[0] <= WIDTH and 0 <= mouse_pos[1] <= HEIGHT:
                pygame.display.quit() 
                pygame.quit()
                sys.exit()
                # so break when screen is clicked  (hypothetically - to end need to ^C in the terminal bc this doesn't want to work)