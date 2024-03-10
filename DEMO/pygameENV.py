# import pyspiel
import random
import numpy as np

class Environment:
    def __init__(self, num_players, num_hosts, num_criticals, connection_matrix, initial_hosts_infected,
                 initial_criticals_infected, weights_blue, weights_red):
        
        self.num_players = num_players
        self.H = num_hosts
        self.C = num_criticals
        self.O = connection_matrix
        self.O_original = np.copy(connection_matrix)
        self.actions = ["Isolate", "Block", "Neutral", "Unblock", "Unisolate", "Restore"]

        #h, b, and i are lists of 0,1 indicating infected, blocked, and isolated hosts respectively
        self.h = initial_hosts_infected
        self.b = [0 for i in range(self.H)]
        self.i = [0 for i in range(self.H)]
        self.c = initial_criticals_infected

        self.h_tm1 = initial_hosts_infected
        self.c_tm1 = initial_criticals_infected

        self.l = np.zeros(self.H)
        self.l_tm1 = np.zeros(self.H)
        self.psi = np.zeros(self.H)
        self.psi_tm1 = np.zeros(self.H)

        self.Delta_connections = 0
        self.num_connections = 0
        self.connections_tm1 = 0


        self.w_blue = weights_blue
        self.w_red = weights_red

        # self.reward_blue = 0 
        # self.reward_red = 0
        # self.return_blue = 0
        # self.return_red = 0



    def initialize(self) :
        # for host in range(self.H) :
        #     if self.h[host] == 1 :
                # neighbors = np.sum(self.O[host, :])
                # self.l[host] = np.cos((neighbors / (self.H + self.C)))

                # min_steps = self.min_steps_to_critical(host)
                # self.psi[host] = (1 - ((min_steps - 1) / self.C))

        self.num_connections = self.O.sum() //2

    def interact_blue(self, a_blue, h_blue):
        """
        Update the environment based on the actions taken by Blue and Red agents

        Parameters:
        - a_blue: The action taken by the Blue agent
        - a_red: The action taken by the Red agent
        - host: The index of the host that the Blue and Red agents are acting upon
        """
        self.initialize()
        self.h_tm1 = np.copy(self.h)
        self.c_tm1 = np.copy(self.c)
        self.psi_tm1 = np.copy(self.psi)
        self.l_tm1 = np.copy(self.l)
        self.connections_tm1 = np.copy(self.num_connections)
        host = h_blue
        print('thinks :', a_blue)
        print(host)
        

        if self.h[host] == 1 :
            if a_blue == 'Block' :
                self.b[host] = 1
                for i in range(self.C):
                    i = i + self.H
                    self.O[host, i] = 0
                    self.O[i, host] = 0
            elif a_blue == 'Isolate' :
                self.O[host, :] = 0
                self.O[:, host] = 0
                self.i[host] = 1
            elif a_blue == 'Unblock':
                self.b[host] = 0
                for i in range(self.C):
                    i = i + self.H
                    self.O[host, i] = self.O_original[host, i]
                    self.O[i, host] = self.O_original[i, host]
            elif a_blue == 'Unisolate':
                self.O[host, :] = self.O_original[host, :]
                self.O[:, host] = self.O_original[:, host]
                self.i[host] = 0
            elif a_blue == "Restore" :
                self.h[host] = 0

                # neighbors = np.sum(self.O[host, :])
                # self.l[host] = np.cos((neighbors / (self.H + self.C)))

                # min_steps = self.min_steps_to_critical(host)
                # self.psi[host] = (1 - ((min_steps - 1) / self.C))
                    
        self.num_connections = self.O.sum() // 2
        self.Delta_connections = self.num_connections - self.connections_tm1

        self.l[host] = - self.Delta_connections / 10

    def interact_red(self, a_red, h_red) :

        host = h_red
        if a_red == 'Spread' and self.h[host]==1:
            for neighbor, connection in enumerate(self.O[host]):
                if connection == 1 :
                    if neighbor < self.H :
                        self.h[neighbor] = 1
                    else :
                        self.c[neighbor - self.H] = 1

    
    def legal_actions_on_hosts(self, player):
        # create a dictionary between the legal actions and the hosts that they can be applied to
        actions_hosts = {}
        if player == 0:
            # print(actions[0])
            targets = [i for i in range(self.H)]
            actions_hosts["Neutral"] = targets
            if np.any(self.b):
                targets =[i for i in range(self.H) if self.b[i] == 1]
                actions_hosts["Unblock"] = targets
            if np.any(self.i):
                targets = [i for i in range(self.H) if self.i[i] == 1]
                actions_hosts["Unisolate"] = targets
            if np.sum(self.h) > 0:
                targets1 = [i for i in range(self.H) if self.O[i, self.H:self.H+self.C].sum() > 0 and self.h[i] == 1 and self.b[i]==0]
                targets2 =[i for i in range(self.H) if self.h[i] == 1 and self.i[i]==0]
                targets3 = [i for i in range(self.H) if self.h[i] == 1]
                actions_hosts["Block"] = targets1
                actions_hosts["Isolate"] = targets2
                actions_hosts["Restore"] = targets3
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

    def payoffs(self):
        print(f"Updated Connection Matrix:\n {self.O}")
        print(f"Updated Hosts Infected:\n {self.h}")
        print(f"Updated Criticals Infected:\n {self.c} \n")

        H_min_h = self.H - np.sum(self.h)
        C_min_c = self.C - np.sum(self.c)
        
        prev_H_min_h = self.H - np.sum(self.h_tm1)
        prev_C_min_c = self.C - np.sum(self.c_tm1)
        
        delta_H_min_h = H_min_h - prev_H_min_h
        delta_h = - delta_H_min_h
        delta_C_min_c = C_min_c - prev_C_min_c
        delta_c = - delta_C_min_c
        
        delta_psi = np.sum(self.psi - self.psi_tm1)
        delta_l = np.sum(self.l - self.l_tm1)
        
        w_1, w_2, w_3, w_4 = self.w_blue
        ŵ_1, ŵ_2 = self.w_red
        
        
        payoff_blue = w_1 * delta_H_min_h + w_2 * delta_C_min_c - (w_3 * delta_psi + w_4 * delta_l)
        payoff_red = ŵ_1 * delta_h + ŵ_2 * delta_c
    
        return payoff_blue, payoff_red