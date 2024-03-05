import pyspiel
import random
import numpy as np
import pygame
import sys

class Environment:
    def __init__(self, num_players, num_hosts, num_criticals, connection_matrix, initial_hosts_infected,
                 initial_criticals_infected, weights_blue, weights_red, 
                 screen_dimensions=(800,600), host_radius=20, server_size=30, 
                 host_color=(200, 200, 200), infected_color=(200, 0, 0), server_color=(0, 128, 0), 
                 connection_color=(0,0,0), disconnected_color=(192, 192, 192)):
        
        self.num_players = num_players
        self.H = num_hosts
        self.C = num_criticals
        self.O = connection_matrix
        self.O_original = connection_matrix

        self.h = initial_hosts_infected
        self.c = initial_criticals_infected

        self.l = np.zeros(self.H)
        self.psi = np.zeros(self.H)

        self.w_blue = weights_blue
        self.w_red = weights_red

        self.screen_width, self.screen_height = screen_dimensions
        self.host_radius = host_radius
        self.server_size = server_size
        self.host_color = host_color
        self.infected_color = infected_color
        self.server_color = server_color
        self.connection_color = connection_color
        self.disconnected_color = disconnected_color

        self.text_color = (0,0,0)
        pygame.font.init()
        self.font = pygame.font.Font(None, 30)

        ####### PYGAME SCREEN ######
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))


    def initialize(self) :
        for host in range(self.H) :
            if self.h[host] == 1 :
                neighbors = np.sum(self.O[host, :])
                self.l[host] = np.cos((neighbors / (self.H + self.C)))

                min_steps = self.min_steps_to_critical(host)
                self.psi[host] = (1 - ((min_steps - 1) / self.C))

    def interact(self, a_blue, h_blue, a_red, h_red):
        """
        Update the environment based on the actions taken by Blue and Red agents

        Parameters:
        - a_blue: The action taken by the Blue agent
        - a_red: The action taken by the Red agent
        - host: The index of the host that the Blue and Red agents are acting upon
        """
        self.initialize()
        h_tm1 = np.copy(self.h)
        c_tm1 = np.copy(self.c)
        psi_tm1 = np.copy(self.psi)
        l_tm1 = np.copy(self.l)
        
        for host in h_blue : 
            if self.h[host] == 1 :
                if a_blue == 'Block' :
                    for i in range(self.C):
                        i = i + self.H
                        self.O[host, i] = 0
                        self.O[i, host] = 0
                elif a_blue == 'Isolate' :
                    self.O[host, :] = 0
                    self.O[:, host] = 0
                elif a_blue == 'Unblock':
                    for i in range(self.C):
                        i = i + self.H
                        self.O[host, i] = self.O_original[host, i]
                        self.O[i, host] = self.O_original[i, host]
                elif a_blue == 'Unisolate':
                    self.O[host, :] = self.O_original[host, :]
                    self.O[:, host] = self.O_original[:, host]

                neighbors = np.sum(self.O[host, :])
                self.l[host] = np.cos((neighbors / (self.H + self.C)))

                min_steps = self.min_steps_to_critical(host)
                self.psi[host] = (1 - ((min_steps - 1) / self.C))

        for host in h_red :
                if a_red == 'Spread' and self.h[host]==1:
                    for neighbor, connection in enumerate(self.O[host]):
                        if connection == 1 :
                            if neighbor < self.H :
                                self.h[neighbor] = 1
                            else :
                                self.c[neighbor - self.H - 1] = 1

        print(f"Updated Connection Matrix:\n {self.O}")
        print(f"Updated Hosts Infected:\n {self.h}")
        print(f"Updated Criticals Infected:\n {self.c} \n")
        payoff_blue, payoff_red = self.payoffs(h_tm1, c_tm1, psi_tm1, l_tm1)

        return payoff_blue, payoff_red
    
    def legal_actions_on_hosts(self, player):
        # create a dictionary between the legal actions and the hosts that they can be applied to
        actions_hosts = {}
        if player == 0:
            actions = ['Neutral']
            # print(actions[0])
            targets = [i for i in range(self.H)]
            actions_hosts[actions[0]] = [targets[0]]
            if np.any(self.O[self.H:] != self.O_original[self.H:]):
                actions.append(['Unblock'])
                targets.append([i for i in range(self.H) if self.O[i, :].sum() > 0])
                actions_hosts[actions[1]] = [targets[1]]
            if np.any(self.O[:self.H, :self.H] != self.O_original[:self.H, :self.H]):
                actions.append(['Unisolate'])
                targets.append([i for i in range(self.H) if self.O[i, :].sum() > 0])
                actions_hosts[actions[1]] = [targets[1]]
            if np.sum(self.h) > 0:
                actions.append(['Block', 'Isolate'])
                targets.append([i for i in range(self.H) if self.h[i] == 1])
                targets.append([i for i in range(self.H) if self.O[i, :].sum() > 0])
                actions_hosts[actions[1][0]] = [targets[1]]
                actions_hosts[actions[1][1]] = [targets[2]]
            return actions_hosts
        else:
            if np.sum(self.h) == 0:
                actions_hosts['Neutral'] = [i for i in range(self.H) if self.h[i] == 1]
            else:
                actions_hosts['Neutral'] = [i for i in range(self.H) if self.h[i] == 1]
                actions_hosts['Spread'] =[i for i in range(self.H) if self.h[i] == 1]
            return actions_hosts
        

    def min_steps_to_critical(self, host):
        return 1
    
    def clone(self):
        return Environment(self.H, self.C, self.O, self.h, self.c, self.w_blue, self.w_red)

    def payoffs(self, h_tm1, c_tm1, psi_tm1, l_tm1):

        H_min_h = self.H - np.sum(self.h)
        C_min_c = self.C - np.sum(self.c)
        
        prev_H_min_h = self.H - np.sum(h_tm1)
        prev_C_min_c = self.C - np.sum(c_tm1)
        
        delta_H_min_h = H_min_h - prev_H_min_h
        delta_h = - delta_H_min_h
        delta_C_min_c = C_min_c - prev_C_min_c
        delta_c = - delta_C_min_c
        
        delta_psi = np.sum(self.psi - psi_tm1)
        delta_l = np.sum(self.l - l_tm1)
        
        w_1, w_2, w_3, w_4 = self.w_blue
        ŵ_1, ŵ_2 = self.w_red
        
        
        payoff_blue = w_1 * delta_H_min_h + w_2 * delta_C_min_c - (w_3 * delta_psi + w_4 * delta_l)
        payoff_red = ŵ_1 * delta_h + ŵ_2 * delta_c
    
        return payoff_blue, payoff_red
    
    ################ PYGAME ADDITIONALS ################


    def draw_network(self):
        # Clear the screen
        self.screen.fill((255, 255, 255))

        # Draw the network based on the connection matrix
        for i in range(self.H + self.C):
            # Determine the color and shape of the node
            if i < self.H:  # It's a host
                color = self.infected_color if self.h[i] == 1 else self.host_color
                position = self.get_node_position(i)
                pygame.draw.circle(self.screen, color, position, self.host_radius)
            else:  # It's a server
                color = self.server_color if self.c[i - self.H] == 0 else self.infected_color  #  server color based on infection status
                position = self.get_node_position(i)
                pygame.draw.rect(self.screen, color, (position[0] - self.server_size // 2,
                                                      position[1] - self.server_size // 2,
                                                      self.server_size, self.server_size))

            # Creating connections (or lack from original)
            for j in range(self.H + self.C):
                neighbor_position = self.get_node_position(j)
                if self.O[i, j] == 1:
                    pygame.draw.line(self.screen, self.connection_color,
                                     position, neighbor_position, 1)
                elif i < self.H and j >= self.H and self.O_original[i, j] == 1:  # grey lines for original connections that are now disconnected
                    pygame.draw.line(self.screen, self.disconnected_color,
                                     position, neighbor_position, 1)
                    
            # Enumerating Hosts / Servers
            text_surface = self.font.render(str(i if i < self.H else i - self.H), True, self.text_color)
            text_rect = text_surface.get_rect(center=position)
            self.screen.blit(text_surface, text_rect)

        # display update
        pygame.display.flip()

    def get_node_position(self, index):
        #  places hosts and servers in a circle --- should probably change
        angle = index * 2 * np.pi / (self.H + self.C)
        center_x, center_y = self.screen_width // 2, self.screen_height // 2
        radius = min(self.screen_width, self.screen_height) // 3
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        return (x, y)
        
    def draw_text(self, text, position, color=(255, 255, 255)):
        font = pygame.font.Font(None, 36)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=position)
        self.screen.blit(text_surface, text_rect)

