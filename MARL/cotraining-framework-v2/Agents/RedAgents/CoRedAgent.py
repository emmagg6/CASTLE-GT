import copy

from .PPOAgent import PPOAgent
import numpy as np
import os

class CoRedAgent(PPOAgent):
    def __init__(self, model_dir, model_file="model.pth"):
        self.model_dir = model_dir
        self.model_file = model_file
        self.action_space=list(range(145)) # not quite right, this is blue, not red
        #self.action_space=list(range(888))
        self.end_episode()

    def get_action(self, observation, action_space=None):
        if self.agent_loaded is False:
            self.agent = self.load_generic()
            self.agent_loaded = True

        print("Obs:", observation)
        # print("Action space:", action_space['action'])
        #print(list(action_space.keys()))

        #env_red = RedTableWrapper(env, output_mode='vector')
        #obs = self.observation_change(observation)

        # take action of agent
        action = self.agent.get_action(observation)

        return action


    def load_generic(self):
        ckpt = os.path.join(os.getcwd(),"Models",self.model_dir,self.model_file)
        # or is it dim=40 instead of 52?
        input_dims = 50 # just to match the +10 from training
        return PPOAgent(input_dims, self.action_space, restore=True, ckpt=ckpt,
                       deterministic=True, training=False)

    def end_episode(self):
        self.agent_loaded = False

    def set_initial_values(self, action_space, observation=None):
        pass

    '''
    def train(self, results: Results):
        """allows an agent to learn a policy"""
        pass
    '''

