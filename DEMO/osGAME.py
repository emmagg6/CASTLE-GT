# import pyspiel
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

    def take_actions(self, action_blue, action_red, host_blue, host_red):
        print(f"Applying actions: {action_blue}, {action_red}")
        print(f"To the Hosts: {host_blue}, {host_red}")

        payoff_blue, payoff_red = self._environment.interact(action_blue, [host_blue], action_red, [host_red])
        print(f"Payoffs: {payoff_blue}, {payoff_red}")

        self._returns[0] += payoff_blue
        self._returns[1] += payoff_red

        self._current_player = 1 - self._current_player

        self._round += 1
        if self._round >= 10: #TODO: don't hardcode this! get it from the game
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
        # update to reflect your environment's state representation
        state_str = f"Round: {self._round} Results\nPlayer: {'Blue' if self._current_player == 0 else 'Red'}\n" \
                    f"Hosts Infected: {np.sum(self._environment.h)}\nCriticals Infected: {np.sum(self._environment.c)}\n"
        return state_str


class APIGame:
    def __init__(self, environment, max_rounds=10):
        #  the game type
        # game_type = pyspiel.GameType(
        #     short_name="api_game",
        #     long_name="API Game",
        #     dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
        #     chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
        #     information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
        #     utility=pyspiel.GameType.Utility.GENERAL_SUM,
        #     reward_model=pyspiel.GameType.RewardModel.TERMINAL,
        #     max_num_players=2,
        #     min_num_players=2,
        #     provides_information_state_string=True,
        #     provides_information_state_tensor=False,
        #     provides_observation_string=True,
        #     provides_observation_tensor=False,
        #     parameter_specification={}
        # )
        
        # #  the game information
        # game_info = pyspiel.GameInfo(
        #     num_distinct_actions=3,
        #     max_chance_outcomes=0,
        #     num_players=2,
        #     min_utility=-float('inf'),  #  minimum possible payoff
        #     max_utility=float('inf'),  # maximum possible payoff
        #     utility_sum=None,  #  could be None since it's not constant-sum or zero-sum ?
        #     max_game_length=max_rounds
        # )
        
        # super().__init__(game_type, game_info, {})
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

    