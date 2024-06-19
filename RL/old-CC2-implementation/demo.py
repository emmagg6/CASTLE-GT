import sys
import subprocess
import inspect
import time
from statistics import mean, stdev
from functools import partial

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.RedAgents.EvadeByRandom import EvadeByRandom
from Agents.RedAgents.EvadeBySleep import EvadeBySleep
from Agents.RedAgents.GreedyAgent import GreedyAgent
from Agents.BlueAgents.BlueSleepAgent import BlueSleepAgent
from Agents.BlueAgents.MainAgent import MainAgent
from Agents.BlueAgents.GenericAgent import GenericAgent
import random

MAX_EPS = 1
num_steps = 30
agent_name = 'Blue'
random.seed(0)

if __name__ == "__main__":
    try:
        blue_agent = sys.argv[1]
    except IndexError as e:
        raise IndexError(f'Must provide the blue agent as input, try cardiff, generic, or generic_long (original error: {e})')
    
    try:
        red_agent_name = sys.argv[2]
    except IndexError as e:
        raise IndexError(f'Must provide the blue agent as input, try cardiff, generic, or generic_long (original error: {e})')
    
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'

    # load blue agent
    if blue_agent == 'cardiff':
        agent = MainAgent()
    elif 'generic' in blue_agent:
        agent = GenericAgent(model_dir=blue_agent)
    elif blue_agent == 'sleep':
        agent = BlueSleepAgent()
    else:
        raise NotImplementedError(f'No blue agent for {blue_agent}, try cardiff, generic, or generic_long')

    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'
    
    if red_agent_name == 'random':
        red_agent, name = (partial(EvadeByRandom, sub_agent=SleepAgent,percent_random=100), 'RandomAgent')
    elif red_agent_name == 'bline_evade':
        red_agent, name = (partial(EvadeBySleep, sub_agent=B_lineAgent), 'EvadeBySleep_Bline')
    elif red_agent_name == 'meander_evade':
        red_agent, name = (partial(EvadeBySleep, sub_agent=RedMeanderAgent), 'EvadeBySleep_RedMeander')
    elif red_agent_name == 'bline':
        red_agent, name = (partial(B_lineAgent), 'B_lineAgent')
    elif red_agent_name == 'meander':
        red_agent, name = (partial(RedMeanderAgent), 'RedMeanderAgent')
    elif red_agent_name == 'greedy':
        red_agent, name = (partial(GreedyAgent), 'GreedyAgent')
    elif red_agent_name == 'sleep':
        red_agent, name = (partial(SleepAgent), 'SleepAgent')

    cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
    wrapped_cyborg = ChallengeWrapper2(env=cyborg, agent_name='Blue')
    observation = wrapped_cyborg.reset()

    action_space = wrapped_cyborg.get_action_space(agent_name)
    for i in range(MAX_EPS):
        for j in range(num_steps):
            print(f'\nTime Step {j}')
            while True:
                try:
                    action = agent.get_action(observation, action_space)
                    observation, rew, done, info = wrapped_cyborg.step(action)
                    print(f"Blue Agent Action: {cyborg.get_last_action('Blue')}")
                    print(f"Red Agent Action: {cyborg.get_last_action('Red')}")
                    # print(f"ip_map:", wrapped_cyborg.get_ip_map())
                except KeyError as e:
                    print(f'Bad action on Episode {i}, step {j}, Action: {action}, Error: {e}. Re-rolling.')
                else:
                    break

        agent.end_episode()
        observation = wrapped_cyborg.reset()
    # print(f'Average reward for red agent {name} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
