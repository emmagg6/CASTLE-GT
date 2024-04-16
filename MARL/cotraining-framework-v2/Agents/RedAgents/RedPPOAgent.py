import random
import torch
import copy,os
import numpy as np
import torch.nn as nn
import inspect

from PPO.ActorCritic import ActorCritic
from PPO.Memory import Memory
from CybORG.Agents import BaseAgent
from CybORG.Shared import Results
from CybORG.Shared.Actions import PrivilegeEscalate, ExploitRemoteService, DiscoverRemoteSystems, Impact, \
    DiscoverNetworkServices, Sleep

from Wrappers.RedVectorize import RedVectorize
from pprint import pprint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RedPPOAgent(BaseAgent):
    def __init__(self, input_dims=50, action_space=list(range(888)), lr=0.002, betas=[0.9, 0.990], gamma=0.99, K_epochs=4, eps_clip=0.2, start_actions=[], restore=False, ckpt=None,
                 deterministic=False, training=True, time_step = 0, rewards = [], internal=False):
        
        # keeping track of valid actions, based on Meander
        self.scanned_subnets = []
        self.scanned_ips = []
        self.exploited_ips = []
        self.escalated_hosts = []
        self.host_ip_map = {}
        self.last_host = None
        self.last_ip = None

        self.action_space_dict = None
        self.observation_dict = None
        self.possible_actions = None
        self.action_signature = {}

        # PPO
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
        self.start_actions = start_actions
        self.time_step = time_step
        self.rewards = rewards
        self.internal = internal
    
        # self.action_space=list(range(888))
        self.redVect = RedVectorize()
        self.last_action = None # need last_action in RedVectorize; the observation is updated based on it
        self.action_space = list(range(888))

        self.set_initial_values_train(action_space = self.action_space)

    def save(self, path):
        print("----Red saves policy to path ", path)
        torch.save(self.policy.state_dict(), path)
    
    def save_old(self, path):
        torch.save(self.old_policy.state_dict(), path)

    def store(self, reward, done):
        # print("Red stores: ", reward, done)
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

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

    def clear_memory(self):
        self.memory.clear_memory()

#    def train(self):
    def train(self, results=None):  # the cyborg format requires the results parameter
        #print(results)
        """allows an agent to learn a policy"""
        if self.internal:
            return

        print("Training external RedPPO")
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


    def get_action_internal(self, obs, action_space=None):

        if self.possible_actions == None:
            self.action_space_change(action_space) # this only when works when red internal
        #print("Possible actions: ", self.possible_actions)

        valid_actions = self.get_valid_actions(obs, action_space)
        valid_indexes = []

        for vact in valid_actions:
            i = -1
            for pact in self.possible_actions:
                i = i+1
                if vact[0] != pact[0]: continue # different action class
                res = all((pact[1].get(k) == v for k, v in vact[1].items()))
                if res:
                    valid_indexes.append(i)  # index of vact in the list of possible actions 
                    break

        obs = self.redVect.observation_change(obs, self.last_action)
        state = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        action = self.old_policy.act(state, self.memory, deterministic=self.deterministic)
        action_ = self.action_space[action]   # in vector space

        # if action index is not valid, randomly choose a valid one
        if action_ not in valid_indexes:
            action_ = random.sample(valid_indexes, 1)[0]

        # need to keep track of these, see Meander
        self.set_action_info(self.possible_actions[action_])

        # self.last_action = self.possible_actions[action_]
        action_class = self.possible_actions[action_][0]
        action_param_dict = self.possible_actions[action_][1]
        self.last_action = action_class(**action_param_dict)

        return action_class(**action_param_dict)

    
    def get_action_external(self, obs, action_space=None):

        self.action_space_change(self.action_space_dict)

        #print("\nPossible actions: ", self.possible_actions)
        valid_actions = self.get_valid_actions(self.observation_dict, self.action_space_dict)
        valid_indexes = []

        #print("\nValid actions:", valid_actions)

        for vact in valid_actions:
            i = -1
            for pact in self.possible_actions:
                i = i+1
                if vact[0] != pact[0]: continue # different action class
                res = all((pact[1].get(k) == v for k, v in vact[1].items()))
                if res:
                    valid_indexes.append(i)  # index of vact in the list of possible actions 
                    break
        
        #print("Valid indexes:", valid_indexes)

        state = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        action = self.old_policy.act(state, self.memory, deterministic=self.deterministic)
        action_ = self.action_space[action]   # in vector space
        #print("PPO chose", action_)

        # if action index is not valid, randomly choose a valid one
        if action_ not in valid_indexes:
            action_ = random.sample(valid_indexes, 1)[0]
        
        #print("FInal action chosen: ", action_, self.possible_actions[action_])

        # need to keep track of these, see Meander
        self.set_action_info(self.possible_actions[action_])

        return action_


    def get_action(self, obs, action_space=None):
        if self.internal:
            return self.get_action_internal(obs, action_space)
        else:
            return self.get_action_external(obs, action_space)


    def set_initial_values(self, action_space, observation=None):
        pass

    def set_initial_values_train(self, action_space, observation=None):
        self.memory = Memory()
        self.n_actions = len(action_space)
        self.policy = ActorCritic(self.input_dims, self.n_actions).to(device)

        #print('Restore:', self.restore)
        #print('ckpt dir: ', self.ckpt)

        if self.restore:
            print("Restoring Red from", self.ckpt, 'internal', self.internal)
            pretained_model = torch.load(self.ckpt, map_location=lambda storage, loc: storage)
            self.policy.load_state_dict(pretained_model)
        #else:
        #    print("Not Restoring Red,internal", self.internal)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)

        self.old_policy = ActorCritic(self.input_dims, self.n_actions).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()



    # based on Meander
    def get_valid_actions(self, observation, action_space):
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        valid_actions = []

        self._process_success(observation)

        session = list(action_space['session'].keys())[0]

        # Always impact if able
        if 'Op_Server0' in self.escalated_hosts:
            valid_actions.append([Impact, {"agent": 'Red', "hostname": 'Op_Server0', "session": session}])

        # start by scanning
        for subnet in action_space["subnet"]:
            if not action_space["subnet"][subnet] or subnet in self.scanned_subnets:
                continue
            valid_actions.append([DiscoverRemoteSystems, {"subnet": subnet, "agent": 'Red', "session": session}])

        # discover network services
        # # act on ip addresses discovered in first subnet
        addresses = [i for i in action_space["ip_address"]]
        for address in addresses:
            if not action_space["ip_address"][address] or address in self.scanned_ips:
                continue
            valid_actions.append([DiscoverNetworkServices, {"ip_address": address, "agent": 'Red', "session": session}])

        # priv esc on owned hosts
        hostnames = [x for x in action_space['hostname'].keys()]
        for hostname in hostnames:
            # test if host is not known
            if not action_space["hostname"][hostname]:
                continue
            # test if host is already priv esc
            if hostname in self.escalated_hosts:
                continue
            # test if host is exploited
            if hostname in self.host_ip_map and self.host_ip_map[hostname] not in self.exploited_ips:
                continue
            valid_actions.append([PrivilegeEscalate, {"hostname": hostname, "agent": 'Red', "session": session}])

        # access unexploited hosts
        for address in addresses:
            # test if output of observation matches expected output
            if not action_space["ip_address"][address] or address in self.exploited_ips:
                continue
            valid_actions.append([ExploitRemoteService, {"ip_address": address, "agent": 'Red', "session": session}])

        if valid_actions == []:
            print("Red has no valid actions!!")
            exit()
    
        return valid_actions


    def set_action_info(self, action):
        
        action_class = action[0]
        params = action[1]

        # print("set_action_info:", action)

        if action_class == Impact:
            self.last_host = 'Op_Server0'
        elif action_class == DiscoverRemoteSystems:
            subnet = params["subnet"]
            self.scanned_subnets.append(subnet)
        elif action_class == DiscoverNetworkServices:
            address = params["ip_address"]
            self.scanned_ips.append(address)
        elif action_class == PrivilegeEscalate:
            hostname = params["hostname"]
            self.escalated_hosts.append(hostname)
            self.last_host = hostname
        elif action_class == ExploitRemoteService:
            address = params["ip_address"]
            self.exploited_ips.append(address)
            self.last_ip = address


    def _process_success(self, observation):
        # print("_process_success:", observation)
        if self.last_ip is not None:
            if observation['success'] == True:
                self.host_ip_map[[value['System info']['Hostname'] for key, value in observation.items()
                                  if key != 'success' and 'System info' in value
                                  and 'Hostname' in value['System info']][0]] = self.last_ip
            else:
                self._process_failed_ip()
            self.last_ip = None
        if self.last_host is not None:
            if observation['success'] == False:
                if self.last_host in self.escalated_hosts:
                    self.escalated_hosts.remove(self.last_host)
                if self.last_host in self.host_ip_map and self.host_ip_map[self.last_host] in self.exploited_ips:
                    self.exploited_ips.remove(self.host_ip_map[self.last_host])
            self.last_host = None

    def _process_failed_ip(self):
        self.exploited_ips.remove(self.last_ip)
        hosts_of_type = lambda y: [x for x in self.escalated_hosts if y in x]
        if len(hosts_of_type('Op')) > 0:
            for host in hosts_of_type('Op'):
                self.escalated_hosts.remove(host)
                ip = self.host_ip_map[host]
                self.exploited_ips.remove(ip)
        elif len(hosts_of_type('Ent')) > 0:
            for host in hosts_of_type('Ent'):
                self.escalated_hosts.remove(host)
                ip = self.host_ip_map[host]
                self.exploited_ips.remove(ip)

    def end_episode(self):
        self.scanned_subnets = []
        self.scanned_ips = []
        self.exploited_ips = []
        self.escalated_hosts = []
        self.host_ip_map = {}
        self.last_host = None
        self.last_ip = None
        self.action_signature = {}
        self.possible_actions = None
        self.redVect = RedVectorize()
        self.last_action = None # need last_action in RedVectorize; the observation is updated based on it
        #print("\nFinished episode")

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


