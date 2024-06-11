import copy

from .PPOAgent import PPOAgent
from .BlueSleepAgent import BlueSleepAgent
import numpy as np
import os

from .Exp3IXrl import CCE


class MainAgent(PPOAgent):
    def __init__(self, model_dir, balance, model_file_PPO="model.pth",  model_file_GT="modelcce.pkl"):
        self.model_dir = model_dir

        self.model_file_ppo = model_file_PPO
        self.model_file_gt = model_file_GT

        self.cce = {}
        self.cce_loaded = False

        self.ppo_action_count = 0
        self.cce_action_count = 0

        self.balance_point = balance

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
            self.agent = self.load_bline()

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
    
    def get_action_and_visits_cce(self, observation):
        '''
        Get the optimal (equilibrium) action from the calculated CCE,
        but only the for those visited more than the threshold (balance).

        Returns:
            action: optimal action from the CCE policy
            visits: number of visits to the state-action pair in the CCE computation
        '''
        if not self.cce_loaded:
            self.load_CCE()
            self.cce_loaded = True
    
        # avg_eq = 0.010 # ~ half average equilibrium value for those with visit counts over 1000
        
        eq_action, eq_visits = self.cce.get_eq_action_visits(tuple(observation), 
                                                             balance=self.balance_point)
        return eq_action, eq_visits



    def get_action(self, observation, action_space=None): 
        '''
        Select action from a combination of the PPO and CCE action-selection
        policies -- depending on whether the equilibrium approx has been 
        statistically estabilished as valid. 

        Returns:
            action: optimal action from either PPO or CCE policy
        '''

        cce_action, cce_visits = self.get_action_and_visits_cce(observation)
        # print("CCE visits: ", cce_visits)
        if cce_visits:
            self.cce_action_count += 1
            return int(cce_action)
        else:
            self.ppo_action_count += 1
            return self.get_action_PPO(observation, action_space)


    def load_sleep(self):
        return BlueSleepAgent()
    
    def load_bline(self):
        ckpt = os.path.join(os.getcwd(),"Models","bline","model.pth")
        return PPOAgent(52, self.action_space, restore=True, ckpt=ckpt,
                       deterministic=True, training=False)

    def load_generic(self):
        ckpt = os.path.join(os.getcwd(),"Models", self.model_dir, self.model_file_ppo)
        return PPOAgent(52, self.action_space, restore=True, ckpt=ckpt,
                       deterministic=True, training=False)

    # def load_CCE(self):
    #     cce_trained= CCE()
    #     ckpt = os.path.join(os.getcwd(),"Models",self.model_dir,self.model_file_gt)
    #     cce = cce_trained.load_eq(ckpt)
    #     self.cce = cce

  # Ensure self.cce is an instance of CCE
    def load_CCE(self):
        # uncomment for non-specific CCE
        # self.cce = CCE(action_space=self.action_space+[55, 51, 116])
        self.cce = CCE()
        ckpt = os.path.join(os.getcwd(), "Models", self.model_dir, self.model_file_gt)
        self.cce.load_eq(ckpt)


    def end_episode(self):
        self.scan_state = np.zeros(10)
        self.start_actions = [51, 116, 55]
        self.agent_loaded = False

    def get_proportions(self):
        return self.ppo_action_count, self.cce_action_count


