# copied from https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py
# only changes involve keeping track of decoys, and reduction of action space

from PPO.ActorCritic import ActorCritic
from PPO.Memory import Memory
import torch
import torch.nn as nn
from CybORG.Agents import BaseAgent
from CybORG.Shared import Results
import numpy as np
import copy
import os
import inspect

from Wrappers.BlueVectorize import BlueVectorize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BluePPOAgent(BaseAgent):
    def __init__(self, input_dims=52, action_space=[i for i in range(158)], lr=0.002, betas=[0.9, 0.990], gamma=0.99, K_epochs=4, eps_clip=0.2, restore=False, start_actions=[], ckpt=None,
                 deterministic=False, training=True, time_step = 0, rewards = [], internal=False):

        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.input_dims = input_dims
        self.restore = restore
        self.ckpt = ckpt
        self.deterministic = deterministic
        self.training = training
        self.start = start_actions
        self.time_step = time_step
        self.rewards = rewards
        self.internal=internal
        self.possible_actions = None
        self.action_signature = {}
        
        self.observation_dict = None
        self.possible_actions = None

        self.last_action = None # need last_action in BlueVectorize; the observation is updated based on it
        self.blueVect = BlueVectorize()
        self.action_space = [133, 134, 135, 139, 3, 4, 5, 9, 16, 17, 18, 22, 11, 12, 13, 14, 141, 142, 143, 144,
                             132, 2, 15, 24, 25, 26, 27]

        # reset decoys
        self.end_episode()

        # initialise
        self.set_initial_values_train(action_space = self.action_space)
    
    def save_rewards(self, path):
        # Check if the rewards file already exists
        if os.path.isfile(path):
            existing_rewards = torch.load(path, map_location=lambda storage, loc: storage) # existing rewards
            updated_rewards = existing_rewards + self.rewards # add the new rewards to the existing rewards
        else:
            # If the file does not exist, just use the new rewards
            updated_rewards = self.rewards
        torch.save(updated_rewards, path)

    def add_reward(self, rewards):
        self.rewards = self.rewards + rewards

    # add a decoy to the decoy list
    def add_decoy(self, id, host):
        # add to list of decoy actions
        if id not in self.current_decoys[host]:
            self.current_decoys[host].append(id)

    # remove a decoy from the decoy list
    def remove_decoy(self, id, host):
        # remove from decoy actions
        if id in self.current_decoys[host]:
            self.current_decoys[host].remove(id)

    def get_action(self, observation, action_space=None):
        # print("act_space:", action_space)

        if self.internal:  # internal to the cyborg environment; need to convert to the vector representation
            # getting the action space mapping
            if self.possible_actions == None:
                self.action_space_change(action_space)

            #print("Blue Observation:", observation)
            if not self.blueVect.baseline:
                observation = self.blueVect.reset(observation, self.last_action)
            else:
                observation = self.blueVect.observation_change(observation, self.last_action)

            #print("Blue Vect Observation:", observation)
        
        state = torch.FloatTensor(observation.reshape(1, -1)).to(device)
        action = self.old_policy.act(state, self.memory, deterministic=self.deterministic)

        action_ = self.action_space[action]

        #print("\nBlue is choosing:", action, action_)
        #print("\nBlue possible act:", self.possible_actions)

        # force start actions, ignore policy. only for training
        if len(self.start_actions) > 0:
            action_ = self.start_actions[0]
            self.start_actions = self.start_actions[1:]

        if action_ in self.decoy_ids:
            host = action_
            # select a decoy from available ones
            action_ = self.select_decoy(host, observation=observation)

        # if action is a restore, delete all decoys from decoy list for that host
        if action_ in self.restore_decoy_mapping.keys():
            for decoy in self.restore_decoy_mapping[action_]:
                for host in self.decoy_ids:
                    if decoy in self.current_decoys[host]:
                        self.remove_decoy(decoy, host)
        #print("xxxxxxFinished BluePPO get_action, chose:", action_, self.possible_actions[action_])
        #print("xxxxBlue's possible actions:", self.possible_actions)
        if self.internal:  # internal to the cyborg environment; need to convert to the correct action object
            #self.last_action = self.possible_actions[action_]
            #return self.possible_actions[action_]

            action_class = self.possible_actions[action_][0]
            action_param_dict = self.possible_actions[action_][1]
        
            self.last_action = action_class(**action_param_dict)

            return action_class(**action_param_dict)

        return action_

    def save(self, path):
        print("----Blue saves policy to path ", path)
        torch.save(self.policy.state_dict(), path)

    def save_old(self, path):
        torch.save(self.old_policy.state_dict(), path)

    def store(self, reward, done):
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

    def clear_memory(self):
        self.memory.clear_memory()

    def select_decoy(self, host, observation):
        try:
            # pick the top remaining decoy
            action = [a for a in self.greedy_decoys[host] if a not in self.current_decoys[host]][0]
            self.add_decoy(action, host)
        except:
            # # otherwise just use the remove action on that host
            # action = self.host_to_remove[host]

            # pick the top decoy again (a non-action)
            if self.training:
                action = self.greedy_decoys[host][0]

            # pick the next best available action (deterministic)
            else:
                state = torch.FloatTensor(observation.reshape(1, -1)).to(device)
                actions = self.old_policy.act(state, self.memory, full=True)

                max_actions = torch.sort(actions, dim=1, descending=True)
                max_actions = max_actions.indices
                max_actions = max_actions.tolist()

                # don't need top action since already know it can't be used (hence could put [1:] here, left for clarity)
                for action_ in max_actions[0]:
                    a = self.action_space[action_]
                    # if next best action is decoy, check if its full also
                    if a in self.current_decoys.keys():
                        if len(self.current_decoys[a]) < len(self.greedy_decoys[a]):
                            action = self.select_decoy(a,observation)
                            self.add_decoy(action, a)
                            break
                    else:
                        # don't select a next best action if "restore", likely too aggressive for 30-50 episodes
                        if a not in self.restore_decoy_mapping.keys():
                            action = a
                            break
        return action

#    def train(self):
    def train(self, results=None):  # the cyborg format requires the results parameter
        """allows an agent to learn a policy"""

        if self.internal:
            return
        
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.squeeze(torch.stack(self.memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(self.memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs)).to(device).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = - torch.min(surr1, surr2)

            critic_loss = 0.5 * self.MSE_loss(rewards, state_values) - 0.01 * dist_entropy

            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())

    def end_episode(self):
        # 9 possible decoys: enterprise 0-2 and user 1-4, defender, opserver0 (cant do actions on user0)
        self.current_decoys = {1000: [], # enterprise0
                               1001: [], # enterprise1
                               1002: [], # enterprise2
                               1003: [], # user1
                               1004: [], # user2
                               1005: [], # user3
                               1006: [], # user4
                               1007: [], # defender
                               1008: []} # opserver0
        # add start actions
        self.start_actions = copy.copy(self.start)
        self.action_signature = {}
        self.possible_actions = None
        self.blueVect = BlueVectorize()
        self.last_action = None # need last_action in BlueVectorize; the observation is updated based on it


    # similar to enumactionwrapper code
    def action_space_change(self, action_space: dict) -> int:
        assert type(action_space) is dict, \
            f"Wrapper required a dictionary action space. " \
            f"Please check that the wrappers below the ReduceActionSpaceWrapper return the action space as a dict "
        possible_actions = []
        temp = {}
        params = ['action']
        # for action in action_space['action']:
        for i, action in enumerate(action_space['action']):
            if action not in self.action_signature:
                self.action_signature[action] = inspect.signature(action).parameters
            p_dict = {}
            param_list = [{}]
            for p in self.action_signature[action]:
                if p == 'priority':
                    continue
                temp[p] = []
                if p not in params:
                    params.append(p)

                if len(action_space[p]) == 1:
                    for p_dict in param_list:
                        p_dict[p] = list(action_space[p].keys())[0]
                else:
                    new_param_list = []
                    for p_dict in param_list:
                        for key, val in action_space[p].items():
                            p_dict[p] = key
                            new_param_list.append({key: value for key, value in p_dict.items()})
                    param_list = new_param_list
            for p_dict in param_list:
                #possible_actions.append(action(**p_dict))
                possible_actions.append([action, p_dict])

        self.possible_actions = possible_actions
        #print(possible_actions, len(possible_actions))
        return len(possible_actions)


    def set_initial_values(self, action_space=[], observation=None):
        pass

    def set_initial_values_train(self, action_space=[], observation=None):

        #print("--BBB--- Blue sets initial values", action_space)

        self.memory = Memory()

        self.greedy_decoys = {1000: [55, 107, 120, 29],  # enterprise0 decoy actions
                              1001: [43],  # enterprise1 decoy actions
                              1002: [44],  # enterprise2 decoy actions
                              1003: [37, 115, 76, 102],  # user1 decoy actions
                              1004: [51, 116, 38, 90],  # user2 decoy actions
                              1005: [130, 91],  # user3 decoy actions
                              1006: [131],  # user4 decoys
                              1007: [54, 106, 28, 119], # defender decoys
                              1008: [61, 35, 113, 126]} # opserver0 decoys

        # added to simplify / for clarity
        self.all_decoys = {55: 1000, 107: 1000, 120: 1000, 29: 1000,
                           43: 1001,
                           44: 1002,
                           37: 1003, 115: 1003, 76: 1003, 102: 1003,
                           51: 1004, 116: 1004, 38: 1004, 90: 1004,
                           130: 1005, 91: 1005,
                           131: 1006,
                           54: 1007, 106: 1007, 28: 1007, 119: 1007,
                           126: 1008, 61: 1008, 113: 1008, 35: 1008}

        # make a mapping of restores to decoys
        self.restore_decoy_mapping = dict()
        # decoys for defender host
        base_list = [28, 41, 54, 67, 80, 93, 106, 119]
        # add for all hosts
        for i in range(13):
            self.restore_decoy_mapping[132 + i] = [x + i for x in base_list]

        # we statically add 9 decoy actions
        action_space_size = len(self.action_space)
        self.n_actions = action_space_size + 9
        self.decoy_ids = list(range(1000, 1009))

        # add decoys to action space (all except user0)
        self.action_space = self.action_space + self.decoy_ids
        #print("Blue Action Space: ", self.action_space)

        self.policy = ActorCritic(self.input_dims, self.n_actions).to(device)
        #print('Restore:', self.restore)
        #print('ckpt dir: ', self.ckpt)
        #print('self.input_dims, self.n_actions:', self.input_dims, self.n_actions)
        
        if self.restore:
            print("Restoring Blue from ", self.ckpt, 'internal', self.internal)
            pretained_model = torch.load(self.ckpt, map_location=lambda storage, loc: storage)
            self.policy.load_state_dict(pretained_model)
        #else:
        #    print("Not Restoring Blue, internal", self.internal)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)

        self.old_policy = ActorCritic(self.input_dims, self.n_actions).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()
