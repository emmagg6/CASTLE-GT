from gym import Env
from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, EnumActionWrapper

# corrected BlueTableWrapper
from .BlueTableWrapper import BlueTableWrapper
from .RedTableWrapper import RedTableWrapper


class CoChallengeWrapper2(Env, BaseWrapper):
    def __init__(self, agent_name: str, agent_name2: str, env, agent=None,
                 reward_threshold=None, max_steps=None):
        super().__init__(env, agent)
        self.agent_name = agent_name
        self.agent_name2 = agent_name2
        
        env_blue = BlueTableWrapper(env, output_mode='vector')
        env_red = RedTableWrapper(env, output_mode='vector') 

        #env = table_wrapper(env, output_mode='vector')
        env_blue = EnumActionWrapper(env_blue)
        env_blue = OpenAIGymWrapper(agent_name=agent_name, env=env_blue)
        
        env_red = EnumActionWrapper(env_red)
        env_red = OpenAIGymWrapper(agent_name=agent_name2, env=env_red)

        self.env_red = env_red
        self.env_blue = env_blue
        self.env = env

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        self.reward_threshold = reward_threshold
        self.max_steps = max_steps
        self.step_counter = None

    def step(self, action=None, action2=None):
        obs, reward, done, info = self.env.step(action=action)

        self.step_counter += 1
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            done = True

        return obs, reward, done, info

    def reset(self):
        self.step_counter = 0
        return self.env.reset()

    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self, agent: str):
        return self.env.get_agent_state(agent)

    def get_action_space(self, agent=None) -> dict:
        return self.env.get_action_space(self.agent_name)

    def get_last_action(self, agent):
        return self.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.get_attr('get_rewards')()

    def get_reward_breakdown(self, agent: str):
        return self.get_attr('get_reward_breakdown')(agent)



