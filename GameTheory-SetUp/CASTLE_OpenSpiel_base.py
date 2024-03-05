import pyspiel
import random
import numpy as np

# Create 2 Player game
'''
Need to create a dynamic, asymmetric payoff matrix that is impaced by the actions of the other player.

PLAYERS :
Player 1: Blue Agent
Player 2: Red Agent


AGENTS :
Right now, the red agent will be a random agent, and the blue agent will be the user.


ACTION SPACE :
Blue Agent Actions: {'Neutral': 0, 'Block': 1, 'Isolate': 2}
Red Agent Actions: {'Neutral': 0, 'Spread': 1}


ENVIRONMENTAL DYNAMICS :
- The environment will be a network of hosts and critical servers. The network will be represented as a connection matrix
where each row represents a host or critical server, and each column represents a connection to another host or critical server. 
The connection matrix will be a binary matrix, where 1 represents a connection and 0 represents no connection, and can be thought of as a directed graph.
- The environment will also keep track of the number of infected hosts and critical servers, the load on each host, 
and the number of step connections to critical servers for each host. 
- The environment will also have a set of weights that will be used to calculate the payoffs for each agent based on the changes 
in the environment due to their actions. The weights will be inputs by the user and will be used to calculate the payoffs for each agent.
- The actions available to the agents will be dictated by the updated connection matrix.
- When an action is taken, the connection matrix will be updated to reflect the changes in the environment due to the action.

If the blue agent takes the action 'Neutral', the connection matrix will not change.
If the blue agent takes the action 'Block', the connection matrix will be updated to reflect that the host is now blocked from all critical servers if any are neighbors.)
If the blue agent takes the action 'Isolate', the connection matrix will be updated to reflect that the host is now isolated from all of its neighbors in the network.
The load will be updated to be the cos(1/2 * pi * (number of neighbors / total number of hosts)).
The host step connections to critical servers will be updated to be the (1 - ((minimum number of step connections to critical servers for the host) - 1)/(number of servers)).

If the red agent takes the action 'Neutral', the infected hosts will not change.
If the red agent takes the action 'Spread', the infected hosts and the infected critical servers will be updated


VARIABLES :
H - Total Hosts
h - Hosts nfected
C - Total Critical Servers
c - Critical Servers Infected
psi - Host step connections to critical servers
l = load on host
Omega - Connection Matrix: Host and Critical Server, (binary)
'Neutral' - no change by the player to the network
'Block' - block all access to critical servers
'Isolate' - isolate the host from all of its neighbors in the network
'Spread' - spread the infection to all of the host's neighbors in the network (this will be start as spread to everyone, but can be changed to a selection of neighbors)


PAYOFFS :
Payoff for Blue Agent = w_1 * Delta(H - h) + w_2 * Delta(C - c) - ( w_3 * Delta(psi) + w_4 * Delta(l) )
Payoff for Red Agent = ŵ_1 * Delta(h) + ŵ_2 * Delta(c)

'''

print('Creating the environment...')

class Environment:
    def __init__(self, num_hosts, num_criticals, connection_matrix, initial_hosts_infected,
                 initial_critials_infected, weights_blue, weights_red):
        self.H = num_hosts
        self.C = num_criticals
        self.O = connection_matrix

        self.h = initial_hosts_infected
        self.c = initial_critials_infected

        self.l = np.zeros(self.H)
        self.psi = np.zeros(self.H)

        self.w_blue = weights_blue
        self.w_red = weights_red

    def initialize(self) :
        for host in range(self.H) :
            if self.h[host] == 1 :
                neighbors = np.sum(self.O[host, :])
                self.l[host] = np.cos((neighbors / (self.H + self.C)))

                min_steps = self.min_steps_to_critical(host)
                self.psi[host] = (1 - ((min_steps - 1) / self.C))

    def interact(self, a_blue, a_red):
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
        
        for host in range(self.H) : 
            if self.h[host] == 1 :
                if a_blue == 'Block' :
                    for i in range(self.C):
                        i = i + self.H
                        self.O[host, i] = 0
                        self.O[i, host] = 0
                elif a_blue == 'Isolate' :
                    self.O[host, :] = 0
                    self.O[:, host] = 0

                neighbors = np.sum(self.O[host, :])
                self.l[host] = np.cos((neighbors / (self.H + self.C)))

                min_steps = self.min_steps_to_critical(host)
                self.psi[host] = (1 - ((min_steps - 1) / self.C))


                if a_red == 'Spread' and self.h[host]==1:
                    for neighbor, connection in enumerate(self.O[host]):
                        if connection == 1 :
                            if neighbor < self.H :
                                self.h[neighbor] = 1
                            else :
                                self.c[neighbor - self.H - 1] = 1


        payoff_blue, payoff_red = self.payoffs(h_tm1, c_tm1, psi_tm1, l_tm1)

        return payoff_blue, payoff_red

    def min_steps_to_critical(self, host):
        return 1

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
    

class Game:
    def __init__(self, environment, max_rounds=10):
        self.environment = environment
        self.blue_actions = {'Neutral': 0, 'Block': 1, 'Isolate': 2}
        self.red_actions = {'Neutral': 0, 'Spread': 1}
        self.R_blue = 0 # Total return for Blue Agent
        self.R_red = 0 # Total return for Red Agent
        self.eps = max_rounds

    def play_game(self):
        round = 0
        while round < self.eps:
            print("Network Connections:")
            print(np.array(self.environment.O))
            print()
            print(f"Infected Hosts: {self.environment.h}")
            print(f"Infected Critical Servers: {self.environment.c}")
            print()

            print(f"Round {round + 1}/{self.eps}")
            
            # Blue agent (user) action
            print("Choose an action for the Blue Agent:")
            for action in self.blue_actions:
                print(f"- {action}")
            blue_action = input("Enter the action: ").strip()
            if blue_action.lower() == 'stop':
                print("Game stopped by the user.")
                break
            while blue_action not in self.blue_actions:
                print("Invalid action. Please enter one of the following actions:")
                for action in self.blue_actions:
                    print(f"- {action}")
                blue_action = input("Enter the action: ").strip()
                if blue_action.lower() == 'stop':
                    print("Game stopped by the user.")
                    return
            
            # Red agent (random) action
            red_action = random.choice(list(self.red_actions.keys()))
            
            print()
            print(f"Blue Agent chooses to {blue_action}")
            print(f"Red Agent chooses to {red_action}")
            print()
            
            payoff_blue, payoff_red = self.environment.interact(blue_action, red_action)
            print(f"Infected Hosts: {self.environment.h}")
            print(f"Infected Critical Servers: {self.environment.c}")
            print()
            print(f"Episode reward for Blue Agent: {payoff_blue}")
            print(f"Episode reward for Red Agent: {payoff_red}")
            print()
            self.R_blue += payoff_blue
            self.R_red += payoff_red

            print()            
            round += 1
            print()

            if (self.environment.h == np.ones(self.environment.H)).all() and (self.environment.c == np.ones(self.environment.C)).all() :
                print("All hosts and critical servers are infected. Game Over.")
                break

        print("Game Completed")
        print(f"Total return for Blue Agent: {self.R_blue}")
        print(f"Total return for Red Agent: {self.R_red}")
        


#################### PLAY ####################


# Initialize 
blue_actions = {'Neutral': 0, 'Block': 1, 'Isolate': 2}
red_actions = {'Neutral': 0, 'Spread': 1}

num_hosts = 3
num_criticals = 2

weights_blue = np.array([0.5, 1, 0.3, 0.2]) # {'w_1': 0.5, 'w_2': 1.0, 'w_3': 0.3, 'w_4': 0.2}
weights_red = np.array([0.75, 1.0]) # {'ŵ_1': 0.75, 'ŵ_2': 1.0}

initial_hosts_infected = np.array([0, 1, 0]) # Host 2 is initially infected
initial_critials_infected = np.array([0, 0]) # No critical server is initially infected

# Connection matrix: (host_1, host_2, host_3, server_1, server_2) x (host_1, host_2, host_3, server_1, server_2)
                    # no self connections
connection_matrix = np.array([[0, 1, 0, 1, 0],  # Host 1 is connected to Host 3 and Server 1
                              [1, 0, 1, 0, 1],  # Host 2 is connected to Host 1 and 3 and Server 2
                              [0, 1, 0, 0, 1],  # Host 3 is connected to Host 2 and Server 2
                              [1, 0, 0, 0, 0],  # Server 1 is connected to Host 1
                              [0, 1, 1, 0, 0]   # Server 2 is connected to Host 2 and Host 3
                              ])


env = Environment(num_hosts, num_criticals, connection_matrix, initial_hosts_infected, initial_critials_infected, weights_blue, weights_red)

# Start the game
game = Game(env)
game.play_game()
