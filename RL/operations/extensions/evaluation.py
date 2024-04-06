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
from Agents.BlueAgents.MainAgent import MainAgent
from Agents.BlueAgents.GenericAgent import GenericAgent
#from Agents.BlueAgents.GenericLongAgent import GenericLongAgent
import random

MAX_EPS = 1 #1000
agent_name = 'Blue'
random.seed(0)


# changed to ChallengeWrapper2
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name='Blue')

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

if __name__ == "__main__":
    try:
        blue_agent = sys.argv[1]
    except IndexError as e:
        raise IndexError(f'Must provide the blue agent as input, try cardiff, generic, or generic_long (original error: {e})')
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    # commit_hash = get_git_revision_hash()
    commit_hash = "Not using git"
    # ask for a name
    name = "John Hannay"
    # ask for a team
    team = "CardiffUni"
    # ask for a name for the agent
    name_of_agent = "PPO + Greedy decoys"

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    # load blue agent
    if blue_agent == 'cardiff':
        agent = MainAgent()
    elif 'generic' in blue_agent:
        agent = GenericAgent(model_dir=blue_agent)
    else:
        raise NotImplementedError(f'No blue agent for {blue_agent}, try cardiff, generic, or generic_long')

    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{cyborg_version}, {scenario}, Commit Hash: {commit_hash}\n')
        data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n")

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    '''
    file_name_csv = './results/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.csv'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name_csv, 'a+') as data:
        data.write('blue_agent,red_agent,steps,avg_reward,std')

    '''

    red_agents = [(partial(B_lineAgent), 'B_lineAgent')]
    # red_agents = [(partial(SleepAgent), 'SleepAgent')]
    # red_agents = [(partial(RedMeanderAgent), 'RedMeanderAgent')]

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in [30]: #, 50, 100]:
        for red_agent, name in red_agents:
            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg)

            observation = wrapped_cyborg.reset()
            # observation = cyborg.reset().observation

            action_space = wrapped_cyborg.get_action_space(agent_name)
            # action_space = cyborg.get_action_space(agent_name)
            total_reward = []
            actions = []
            for i in range(MAX_EPS):
                r = []
                a = []
                # cyborg.env.env.tracker.render()
                for j in range(num_steps):
                    while True:
                        try:
                            action = agent.get_action(observation, action_space)
                            observation, rew, done, info = wrapped_cyborg.step(action)
                            print(f"Blue Agent Action: {cyborg.get_last_action('Blue')}")
                            print(f"Red Agent Action: {cyborg.get_last_action('Red')}")
                        except KeyError as e:
                            print(f'Bad action on Episode {i}, step {j}, Action: {action}, Error: {e}. Re-rolling.')
                        else:
                            break
                    # result = cyborg.step(agent_name, action)
                    r.append(rew)
                    # r.append(result.reward)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

                agent.end_episode()
                total_reward.append(sum(r))
                actions.append(a)
                # observation = cyborg.reset().observation
                observation = wrapped_cyborg.reset()
            print(f'Average reward for red agent {name} and steps {num_steps} is: {mean(total_reward)}')

            '''
            print(f'Average reward for red agent {name} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {name}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')
            with open(file_name_csv, 'a+') as data:
                data.write(f'\n{blue_agent},{name},{num_steps},{mean(total_reward)},{stdev(total_reward)}')
            '''
