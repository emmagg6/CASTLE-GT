from CybORG.Shared import Results
from CybORG.Shared.Actions import Sleep
from CybORG.Agents import BaseAgent

class EvadeBySleep(BaseAgent):

    def __init__(self, sub_agent, len_sleep=2):
        # 2 is the optimal sleep for evading B_lineAgent and RedMeanderAgent
        self.sub_agent = sub_agent()
        self.len_sleep = len_sleep
        

    def train(self, results: Results):
        pass

    # Sleep for len_sleep, then act using sub-agent actions
    def get_action(self, observation, action_space):

        # Save initial state and observation
        if self.initial_observation == None:
            self.initial_observation = observation
        
        action = None
        if self.action_counter < self.len_sleep:
            # Sleep
            action = Sleep()
        elif self.action_counter == self.len_sleep:
            # First action -- use initial observation
            action = self.sub_agent.get_action(self.initial_observation, action_space)
        elif self.action_counter > self.len_sleep:
            # actions after first action -- use current observation
            action = self.sub_agent.get_action(observation, action_space)

        self.action_counter += 1

        return action

    def end_episode(self):
        self.action_counter = 0
        self.initial_observation = None
        self.sub_agent.end_episode()

    def set_initial_values(self, action_space, observation):
        self.sub_agent.set_initial_values(action_space,observation)


