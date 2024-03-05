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
Right now, the red agent will be a random agent, and the blue agent will be another random agent.


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
'Block' - block all access of the select host to critical servers
'Isolate' - isolate the selecthost from all of its neighbors in the network
'Spread' - spread the infection of select host's neighbor in the network (can be changed to a selection of neighbors)


PAYOFFS :
Payoff for Blue Agent = w_1 * Delta(H - h) + w_2 * Delta(C - c) - ( w_3 * Delta(psi) + w_4 * Delta(l) )
Payoff for Red Agent = ŵ_1 * Delta(h) + ŵ_2 * Delta(c)

'''

print('Creating the environment...')

class Environment:
    def __init__(self, num_players, num_hosts, num_criticals, connection_matrix, initial_hosts_infected,
                 initial_criticals_infected, weights_blue, weights_red):
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


#################### GAME ####################
print('Creating the game...')

class APIState(pyspiel.State):
    def __init__(self, game, environment):
        super().__init__(game)
        self._environment = environment
        self._is_terminal = False
        self._round = 0
        self._returns = [0, 0]  # format: [Blue, Red]
        self._current_player = 0  # Blue starts

    def clone(self):
        # Create a deep copy of this state
        cloned_state = APIState(self.get_game())
        cloned_state._environment = self._environment.clone()
        cloned_state._is_terminal = self._is_terminal
        cloned_state._round = self._round
        cloned_state._returns = list(self._returns)
        cloned_state._current_player = self._current_player
        cloned_state._pending_action_blue = self._pending_action_blue
        cloned_state._pending_action_red = self._pending_action_red
        return cloned_state

    def current_player(self):
        # If the game is over, return Terminal player id
        # Otherwise, return the id of the current player
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._current_player

    def take_actions(self, action_blue, action_red, host_blue, host_red):
        print(f"Applying actions: {action_blue}, {action_red}")
        print(f"To the Hosts: {host_blue}, {host_red}")
        # Apply actions in the environment and calculate payoffs
        payoff_blue, payoff_red = self._environment.interact(action_blue, [host_blue], action_red, [host_red])
        print(f"Payoffs: {payoff_blue}, {payoff_red}")

        self._returns[0] += payoff_blue
        self._returns[1] += payoff_red

        self._current_player = 1 - self._current_player

        self._round += 1
        if self._round >= self.get_game().max_game_length():
            self._is_terminal = True

    def legal_actions_on_hosts(self, player):
        return self._environment.legal_actions_on_hosts(player)

    def apply_actions(self, actions, host_targets):
        # print(f"Actions: {actions[0]}, {actions[1]}")
        # print(f"Hosts: {host_targets[0]}, {host_targets[1]}")
        # Apply the joint action
        if not self._is_terminal:
            self.take_actions(actions[0], actions[1], host_targets[0], host_targets[1])

    def is_terminal(self):
        return self._is_terminal

    def returns(self):
        return self._returns

    def __str__(self):
        # Update to reflect your environment's state representation
        state_str = f"Round: {self._round} Results\nPlayer: {'Blue' if self._current_player == 0 else 'Red'}\n" \
                    f"Hosts Infected: {np.sum(self._environment.h)}\nCriticals Infected: {np.sum(self._environment.c)}\n"
        # Add additional details from the environment as needed
        return state_str


class APIGame(pyspiel.Game):
    def __init__(self, environment, max_rounds=10):
        # Define the game type
        game_type = pyspiel.GameType(
            short_name="api_game",
            long_name="API Game",
            dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
            chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
            information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
            utility=pyspiel.GameType.Utility.GENERAL_SUM,
            reward_model=pyspiel.GameType.RewardModel.TERMINAL,
            max_num_players=2,
            min_num_players=2,
            provides_information_state_string=True,
            provides_information_state_tensor=False,
            provides_observation_string=True,
            provides_observation_tensor=False,
            parameter_specification={}
        )
        
        # Define the game information
        game_info = pyspiel.GameInfo(
            num_distinct_actions=3,  # Maximum number of actions between both agents
            max_chance_outcomes=0,
            num_players=2,
            min_utility=-float('inf'),  # Placeholder, set to minimum possible payoff
            max_utility=float('inf'),  # Placeholder, set to maximum possible payoff
            utility_sum=None,  # This could be None since it's not constant-sum or zero-sum
            max_game_length=max_rounds
        )
        
        # Initialize the base game class
        super().__init__(game_type, game_info, {})

        # Additional environment and game-specific initialization
        self._environment = environment
        self.max_rounds = max_rounds

    def new_initial_state(self):
        return APIState(self, self._environment)

    def num_distinct_actions(self):
        # Assuming Environment has a method to get all distinct actions
        return len(self._environment.get_all_actions())

    def max_game_length(self):
        return self.max_rounds

    def num_players(self):
        # Assuming Environment knows the number of players
        return self._environment.num_players

    


#################### PLAY ####################


# Initialize 
print('Initializing ...')

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

# Now, run Nash equilibrium algorithm
# nash_conv = pyspiel.nash_conv(security_game, num_points=100)