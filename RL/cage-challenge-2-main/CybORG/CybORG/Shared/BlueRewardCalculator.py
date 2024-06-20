from collections import namedtuple

from CybORG.Shared import Scenario
from CybORG.Shared.RedRewardCalculator import DistruptRewardCalculator, PwnRewardCalculator
from CybORG.Shared.RewardCalculator import RewardCalculator

from CybORG.Shared.Actions import Restore
from CybORG.Shared.Actions.Action import Action


HostReward = namedtuple('HostReward','confidentiality availability operations')
# HostReward = namedtuple('HostReward','confidentiality availability')

class ConfidentialityRewardCalculator(RewardCalculator):
    # Calculate punishment for defending agent based on compromise of hosts/data
    def __init__(self, agent_name: str, scenario: Scenario):
        self.scenario = scenario
        self.adversary = scenario.get_agent_info(agent_name).adversary
        super(ConfidentialityRewardCalculator, self).__init__(agent_name)
        self.infiltrate_rc = PwnRewardCalculator(self.adversary, scenario)
        self.compromised_hosts = {}

    def reset(self):
        self.infiltrate_rc.reset()

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        self.compromised_hosts = {}
        reward = -self.infiltrate_rc.calculate_reward(current_state, action, agent_observations, done)
        self._calculate_compromised_hosts()
        return reward

    def _calculate_compromised_hosts(self):
        for host, value in self.infiltrate_rc.compromised_hosts.items():
            self.compromised_hosts[host] = -1 * value


class AvailabilityRewardCalculator(RewardCalculator):
    # Calculate punishment for defending agent based on reduction in availability
    def __init__(self, agent_name: str, scenario: Scenario):
        super(AvailabilityRewardCalculator, self).__init__(agent_name)
        self.adversary = scenario.get_agent_info(agent_name).adversary
        self.disrupt_rc = DistruptRewardCalculator(self.adversary, scenario)
        self.impacted_hosts = {}

    def reset(self):
        self.disrupt_rc.reset()

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        self.impacted_hosts = {}
        reward = -self.disrupt_rc.calculate_reward(current_state, action, agent_observations, done)
        self._calculate_impacted_hosts()
        return reward

    def _calculate_impacted_hosts(self):
        for host, value in self.disrupt_rc.impacted_hosts.items():
            self.impacted_hosts[host] = -1 * value


class OperationsRewardCalculator(RewardCalculator):
    # Calculate punishment for defending agent based on reduction in network operations
    def __init__(self, agent_name: str, scenario: Scenario):
        super(OperationsRewardCalculator, self).__init__(agent_name)
        self.scenario = scenario
        self.impacted_hosts = {}
        
    def reset(self):
        self.impacted_hosts = {}

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        reward = 0.0
        self.impacted_hosts = {}
        
        agent_action = action[self.agent_name]
        hostname = agent_action.hostname

        if type(agent_action) is Restore and agent_observations[self.agent_name].data['success'] == True:
            # print("\n\n-------Will do a Restore on", hostname, "operations cost: ", agent_action.operations_cost(self.scenario), '\n\n')
            reward = agent_action.operations_cost(self.scenario)
            self.impacted_hosts[hostname] = reward

        return reward

    def _calculate_impacted_hosts(self):
        pass 


class HybridAvailabilityConfidentialityRewardCalculator(RewardCalculator):
    # Hybrid of availability and confidentiality reward calculator
    def __init__(self, agent_name: str, scenario: Scenario):
        super(HybridAvailabilityConfidentialityRewardCalculator, self).__init__(agent_name)
        self.availability_calculator = AvailabilityRewardCalculator(agent_name, scenario)
        self.confidentiality_calculator = ConfidentialityRewardCalculator(agent_name, scenario)

    def reset(self):
        self.availability_calculator.reset()
        self.confidentiality_calculator.reset()

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        reward = self.availability_calculator.calculate_reward(current_state, action, agent_observations, done) \
                 + self.confidentiality_calculator.calculate_reward(current_state, action, agent_observations, done)
        self._compute_host_scores(current_state.keys())
        return reward

    def _compute_host_scores(self, hostnames):
        self.host_scores = {}
        compromised_hosts = self.confidentiality_calculator.compromised_hosts
        impacted_hosts = self.availability_calculator.impacted_hosts
        for host in hostnames:
            if host == 'success':
                continue
            compromised = compromised_hosts[host] if host in compromised_hosts else 0
            impacted = impacted_hosts[host] if host in impacted_hosts else 0
            reward_state = HostReward(compromised,impacted)  
                                    # confidentiality, availability
            self.host_scores[host] = reward_state


class HybridAvailabilityConfidentialityOperationsRewardCalculator(RewardCalculator):
    # Hybrid of availability and confidentiality and operations reward calculator
    def __init__(self, agent_name: str, scenario: Scenario):
        super(HybridAvailabilityConfidentialityOperationsRewardCalculator, self).__init__(agent_name)
        self.availability_calculator = AvailabilityRewardCalculator(agent_name, scenario)
        self.confidentiality_calculator = ConfidentialityRewardCalculator(agent_name, scenario)
        self.operations_calculator = OperationsRewardCalculator(agent_name, scenario)

    def reset(self):
        self.availability_calculator.reset()
        self.confidentiality_calculator.reset()
        self.operations_calculator.reset()

    def calculate_reward(self, current_state: dict, action: dict, agent_observations: dict, done: bool) -> float:
        reward = self.availability_calculator.calculate_reward(current_state, action, agent_observations, done) \
                 + self.confidentiality_calculator.calculate_reward(current_state, action, agent_observations, done) \
                 + self.operations_calculator.calculate_reward(current_state, action, agent_observations, done)
        self._compute_host_scores(current_state.keys())
        return reward

    def _compute_host_scores(self, hostnames):
        self.host_scores = {}
        compromised_hosts = self.confidentiality_calculator.compromised_hosts
        impacted_hosts = self.availability_calculator.impacted_hosts
        op_impacted_hosts = self.operations_calculator.impacted_hosts
        for host in hostnames:
            if host == 'success':
                continue
            compromised = compromised_hosts[host] if host in compromised_hosts else 0
            impacted = impacted_hosts[host] if host in impacted_hosts else 0
            op_impacted = op_impacted_hosts[host] if host in op_impacted_hosts else 0
            reward_state = HostReward(compromised,impacted,op_impacted)
                                    # confidentiality, availability, operations
            self.host_scores[host] = reward_state

