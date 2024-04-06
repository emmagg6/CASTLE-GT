from gym import Env
#from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, RedTableWrapper, EnumActionWrapper
from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, EnumActionWrapper

# corrected BlueTableWrapper
from .BlueTableWrapper import BlueTableWrapper
from .RedTableWrapper import RedTableWrapper


class ChallengeWrapper2(Env, BaseWrapper):
    def __init__(self, agent_name: str, env, agent=None,
                 reward_threshold=None, max_steps=None):
        super().__init__(env, agent)
        self.agent_name = agent_name
        if agent_name.lower() == 'red':
            table_wrapper = RedTableWrapper
        elif agent_name.lower() == 'blue':
            table_wrapper = BlueTableWrapper
        else:
            raise ValueError('Invalid Agent Name')

        self.env_orig = env

        env = table_wrapper(env, output_mode='vector')
        action_space = env.get_action_space(agent_name)

        env = EnumActionWrapper(env)
        env.action_space_change(action_space)
        self.possible_actions = env.possible_actions

        env = OpenAIGymWrapper(agent_name=agent_name, env=env)

        self.env = env
        self.action_space = self.env.action_space
        #print("wrapper2: ", self.action_space)
        self.observation_space = self.env.observation_space
        self.reward_threshold = reward_threshold
        self.max_steps = max_steps
        self.step_counter = None

    def step(self, action=None):
        obs, reward, done, info = self.env.step(action=action)

        self.step_counter += 1
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            done = True

        return obs, reward, done, info

    def get_possible_actions(self):
        return self.possible_actions

    def reset(self):
        self.step_counter = 0
        return self.env.reset()

    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_observation_dict(self, agent: str):
        return self.env_orig.get_observation(agent)
    
    def get_agent_state(self, agent: str):
        return self.env.get_agent_state(agent)

    def get_action_space(self, agent=None) -> dict:
        return self.env.get_action_space(self.agent_name)
    
    def get_action_space_dict(self, agent=None) -> dict:
        return self.env_orig.get_action_space(self.agent_name)

    def get_last_action(self, agent):
        return self.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.get_attr('get_rewards')()

    def get_reward_breakdown(self, agent: str):
        return self.get_attr('get_reward_breakdown')(agent)

    def test_valid_action(self, action, agent):
        return self.env.test_valid_action(action, agent)


