import numpy as np
import pygame
import sys

class PyGameInterface:
    def __init__(self, environment, screen_dimensions=(800,600), host_radius=20, server_size=30, 
                 host_color=(200, 200, 200), infected_color=(200, 0, 0), server_color=(0, 128, 0), 
                 connection_color=(0,0,0), disconnected_color=(192, 192, 192)):
        self.env = environment
        self.screen_width, self.screen_height = screen_dimensions
        self.host_radius = host_radius
        self.server_size = server_size
        self.host_color = host_color
        self.infected_color = infected_color
        self.server_color = server_color
        self.connection_color = connection_color
        self.disconnected_color = disconnected_color
        self.button_width = 20
        self.button_height = 30

        self.text_color = (0,0,0)
        pygame.font.init()
        self.font = pygame.font.Font(None, 30)

        ### Arranging the hosts / servers ##
        self.host_radius = min(self.screen_width, self.screen_height) // 30
        self.host_center = (self.screen_width // 2 + self.screen_width //10, self.screen_height // 2)
        self.server_start_po = (self.host_center[0] + self.host_radius + 200, 
                                self.screen_height// 2 - (self.env.C - 1)*50)

        ####### PYGAME SCREEN ######
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

    def screen_fill(self):
        self.screen.fill((255, 255, 255))

    def draw_network(self):
        # Clear the screen
        self.screen_fill()

        # Draw the network based on the connection matrix
        for i in range(self.env.H + self.env.C):

            if i < self.env.H:  # It's a host
                color = self.infected_color if self.env.h[i] == 1 else self.host_color
                position = self.get_node_position(i)
            else:  # It's a server
                color = self.server_color if self.env.c[i - self.env.H] == 0 else self.infected_color  #  server color based on infection status
                position = self.get_node_position(i)

            # Creating connections (or lack from original)
            for j in range(self.env.H + self.env.C):
                neighbor_position = self.get_node_position(j)
                if self.env.O[i, j] == 1:
                    pygame.draw.line(self.screen, self.connection_color,
                                     position, neighbor_position, 2)
                elif self.env.O_original[i, j] == 1:  # grey lines for original connections that are now disconnected
                    pygame.draw.line(self.screen, self.disconnected_color,
                                     position, neighbor_position, 2)
        for i in range(self.env.H + self.env.C):
            # Determine the color and shape of the node
            if i < self.env.H:  # It's a host
                color = self.infected_color if self.env.h[i] == 1 else self.host_color
                position = self.get_node_position(i)
                pygame.draw.circle(self.screen, color, position, self.host_radius)
            else:  # It's a server
                color = self.server_color if self.env.c[i - self.env.H] == 0 else self.infected_color  #  server color based on infection status
                position = self.get_node_position(i)
                pygame.draw.rect(self.screen, color, (position[0] - self.server_size // 2,
                                                      position[1] - self.server_size // 2,
                                                      self.server_size, self.server_size))
     
            # Enumerating Hosts / Servers
            text_surface = self.font.render("H"+ str(i) if i < self.env.H else "S"+ str(i - self.env.H), True, self.text_color)
            text_rect = text_surface.get_rect(center=position)
            self.screen.blit(text_surface, text_rect)

        # display update
        pygame.display.flip()

    # def get_node_position(self, index):
    #     #  places hosts and servers in a circle --- should probably change
    #     angle = index * 2 * np.pi / (self.env.H + self.env.C)
    #     center_x, center_y = self.screen_width // 2, self.screen_height // 2
    #     radius = min(self.screen_width, self.screen_height) // 3
    #     x = int(center_x + radius * np.cos(angle))
    #     y = int(center_y + radius * np.sin(angle))
    #     return (x, y)
    def get_node_position(self, index):
        if index < self.env.H:  # It's a host
            # Arrange hosts in a circle at the center
            angle = index * 2 * np.pi / self.env.H + np.pi
            x = int(self.host_center[0] + self.host_radius * np.cos(angle) * 5)
            y = int(self.host_center[1] + self.host_radius * np.sin(angle) * 5)
        else:  # It's a server
            # Arrange servers in a vertical line to the right
            server_index = index - self.env.H
            x = self.server_start_po[0] 
            y = self.server_start_po[1] + server_index * (self.server_size + 50)  # 50 is spacing between servers

        return (x, y)


    def get_button_position(self, action, host):
        height = self.screen_height//2 + self.button_height * (self.env.actions.index(action))
        width = 130 + self.button_width * host * 2
        return (width, height)
        
    def draw_text(self, text, position, color=(0, 0, 0), align="center", font_size = 20, font = 'regular'):
        if font == 'light' : 
            font = pygame.font.Font('OpenSans-Light.ttf', font_size)
        elif font == 'bold' : 
            font = pygame.font.Font('OpenSans-Bold.ttf', font_size)
        elif font == 'italic' :
            font = pygame.font.Font('OpenSans-Italic.ttf', font_size)
        else : 
            font = pygame.font.Font('OpenSans-Regular.ttf', font_size)
        
        text_surface = font.render(text, True, color)
        if align == "center":
            text_rect = text_surface.get_rect(center=position)
        elif align == "midleft":
            text_rect = text_surface.get_rect(midleft=position)
        self.screen.blit(text_surface, text_rect)

    def draw_selection_button(self, action, host, color) : 
        position= self.get_button_position(action, host)
        w, h = position
        rect = pygame.Rect(w-16, h-11, 32, 24)
        pygame.draw.rect(self.screen, color, rect, width=2)


