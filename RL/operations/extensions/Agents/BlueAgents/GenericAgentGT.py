'''
TO BE COMPLETE FIXED THIS IS JUST MARKED UP FOR AN OUTLINE

'''


import copy

from .PPOAgent import PPOAgent
from .BlueSleepAgent import BlueSleepAgent
import numpy as np
import os

class GenericAgent(PPOAgent):
    def __init__(self, model_dir_PPO, model_file_PPO="model.pth", model_dir_GT, model_file_GT="model.pth"):
        self.model_dir_ppo = model_dir_PPO
        self.model_file_ppo = model_file_PPO

        self.model_dir_gt = model_dir_GT
        self.model_file_gt = model_file_GT

        self.action_space = [133, 134, 135, 139, 3, 4, 5, 9, 16, 17, 18, 22, 11, 12, 13, 14, 141, 142, 143, 144,
                             132, 2, 15, 24, 25, 26, 27]
        self.end_episode()

    def get_action_PPO(self, observation, action_space=None):
        action = None
        # keep track of scans
        old_scan_state = copy.copy(self.scan_state)
        super().add_scan(observation)
        # start actions
        if len(self.start_actions) > 0:
            action = self.start_actions[0]
            self.start_actions = self.start_actions[1:]

        # load agent based on fingerprint
        elif self.agent_loaded is False:
            self.agent = self.load_generic()

            self.agent_loaded = True
            # add decoys and scan state
            self.agent.current_decoys = {1000: [55], # enterprise0
                                         1001: [], # enterprise1
                                         1002: [], # enterprise2
                                         1003: [], # user1
                                         1004: [51, 116], # user2
                                         1005: [], # user3
                                         1006: [], # user4
                                         1007: [], # defender
                                         1008: []} # opserver0
            # add old since it will add new scan in its own action (since recieves latest observation)
            self.agent.scan_state = old_scan_state


        # take action of agent
        if action is None:
            action = self.agent.get_action(observation)
        return action
    
    def get_action_cce():
        '''
        Get the action from the CCE agent.

        Returns:
            action: optimal action from the CCE policy
        '''
        #TODO




    def select_action(): 
        '''
        Select action from a combination of the PPO and CCE action-selection
        policies -- depending on whether the equilibrium approx has been 
        statistically estabilished as valid. 

        Returns:
            action: optimal action from either PPO or CCE policy
        '''
        # TODO




    def load_sleep(self):
        return BlueSleepAgent()

    def load_generic(self):
        ckpt = os.path.join(os.getcwd(),"Models",self.model_dir,self.model_file)
        return PPOAgent(52, self.action_space, restore=True, ckpt=ckpt,
                       deterministic=True, training=False)

    def end_episode(self):
        self.scan_state = np.zeros(10)
        self.start_actions = [51, 116, 55]
        self.agent_loaded = False




