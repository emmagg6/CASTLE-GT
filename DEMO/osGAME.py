import pyspiel
import random
import numpy as np
import pygame
import sys

class APIState:
    def __init__(self, game, environment):
        # super().__init__(game)
        self._environment = environment
        self._is_terminal = False
        self._round = 0
        self._returns = [0, 0]  # format: [Blue, Red]
        self._rewards = [0, 0] # formet: [Blue, Red]
        self._current_player = 0  # Blue starts

    def clone(self):
        # deep copy of this state
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
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._current_player

    def take_action_blue(self, a, h):
        print(f"Blue is applying: {a}")
        print(f"To the Host: {h}")

        self._environment.interact_blue(a, h)
        self._current_player = 1 - self._current_player


    def take_action_red(self, a, h):
        print(f"Red is applying: {a}")
        print(f"To the Host: {h}")

        self._environment.interact_red(a, h)
        self._current_player = 1 - self._current_player

    def payoff(self) :
        reward_blue, reward_red = self._environment.payoffs()
        self._rewards[0] = reward_blue
        self._rewards[1] = reward_red
        self._returns[0] += reward_blue
        self._returns[1] += reward_red
        return self._rewards
    
    def rewards(self) :
        return self._rewards

    def legal_actions_on_hosts(self, player):
        return self._environment.legal_actions_on_hosts(player)

    def apply_action(self, action, host_target):
        # print(f"Actions: {actions[0]}, {actions[1]}")
        # print(f"Hosts: {host_targets[0]}, {host_targets[1]}")
        # Apply the sequential just incase the host is no longer infected -> if red choses spread, nothing happens

        if not self.is_terminal():
            if self._current_player == 0 :
                self.take_action_blue(action, host_target)
            else :
                self.take_action_red(action, host_target)

        self._round += 1
        if self._round // 2 >= 20 : # round is when EACH agent has taken an action
            self._is_terminal = True

    def is_terminal(self):
        if np.all(self._environment.h == 1) and np.all(self._environment.c == 1) : 
            self._is_terminal = True
        if np.all(self._environment.h == 0) : 
            self._is_terminal = True
        return self._is_terminal

    def returns(self):
        return self._returns

    def __str__(self):
        # update to reflect your environment's state representation
        state_str = f"Round: {self._round} Results\nPlayer: {'Blue' if self._current_player == 0 else 'Red'}\n" \
                    f"Hosts Infected: {np.sum(self._environment.h)}\nCriticals Infected: {np.sum(self._environment.c)}\n"
        return state_str


class APIGame:
    def __init__(self, environment, max_rounds=10):
        #  the game type
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
        
        # #  the game information
        game_info = pyspiel.GameInfo(
            num_distinct_actions=7,
            max_chance_outcomes=0,
            num_players=2,
            min_utility=-float('inf'),  #  minimum possible payoff
            max_utility=float('inf'),  # maximum possible payoff
            utility_sum=None,  #  could be None since it's not constant-sum or zero-sum ?
            max_game_length=max_rounds
        )
        
        # super().__init__(game_type, game_info)  # not required if APIGame is not supposed t extend any other class
        self._environment = environment
        self.max_rounds = max_rounds

    def new_initial_state(self):
        return APIState(self, self._environment)

    def num_distinct_actions(self):
        return len(self._environment.get_all_actions())

    def max_game_length(self):
        return self.max_rounds

    def num_players(self):
        return self._environment.num_players

    