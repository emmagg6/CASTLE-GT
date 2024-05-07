from random import random
from CybORG.Shared import Results
from CybORG.Agents import BaseAgent
from .RandomAgent import RandomAgent


class EvadeByRandom(BaseAgent):

    def __init__(self, sub_agent, percent_random=100):
        if percent_random > 100 or percent_random < 0:
            raise ValueError(f'percent random must be between 0 and 100, invalid: {percent_random}')

        self.sub_agent = sub_agent()
        self.random_sub_agent = RandomAgent()
        self.percent_random = percent_random/100
        

    def train(self, results: Results):
        pass

    # For a percentage of turns, take a random action
    def get_action(self, observation, action_space):
        take_random_action = random() < self.percent_random
        if take_random_action:
            action = self.random_sub_agent.get_action(observation,action_space)
            # print(True, action)
        else:
            raise NotImplementedError('There are issues with all subagents if we alternate between random and subagent actions -- to be implemented later')
            # action = self.sub_agent.get_action(observation, action_space)
            # print(False, action)
        return action

    def end_episode(self):
        self.action_counter = 0
        self.initial_observation = None
        self.sub_agent.end_episode()

    def set_initial_values(self, action_space, observation):
        self.sub_agent.set_initial_values(action_space,observation)


