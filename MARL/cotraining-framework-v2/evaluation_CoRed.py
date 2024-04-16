import sys
import subprocess
import inspect
import time
from statistics import mean, stdev
import functools as ft

from CybORG import CybORG, CYBORG_VERSION
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.BlueAgents.BluePPOAgent import BluePPOAgent
from Agents.RedAgents.RedPPOAgent import RedPPOAgent
from CybORG.Agents import BlueReactRemoveAgent
import random

MAX_EPS = 100
#MAX_EPS = 1
agent_name = 'Red'
random.seed(0)

# PPO Params
K_epochs = 6
eps_clip = 0.2 # not used in evaluation
gamma = 0.99 # not used in evaluation
betas=[0.9, 0.990] # not used in evaluation
lr = 0.002 # not used in evaluation


# changed to ChallengeWrapper2
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name='Red')

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    commit_hash = "Not using git"
    name = "noname"
    team = "Dartmouth_and_Northeastern"
    # ask for a name for the agent
    name_of_agent = "CoBlue"

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    '''
    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{cyborg_version}, {scenario}, Commit Hash: {commit_hash}\n')
        data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n")
    '''

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    '''
    file_name_csv = './results/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.csv'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name_csv, 'a+') as data:
        data.write('blue_agent,red_agent,steps,avg_reward,std')
    '''

    '''
    red_agents = [(partial(B_lineAgent), 'B_lineAgent'), 
                  (partial(RedMeanderAgent), 'RedMeanderAgent'), 
                  (partial(SleepAgent), 'SleepAgent'),
                  ]
    '''

    # define ppo red agent
    start_actions_red=[]
    red_action_space=list(range(888))
    red_load_ckpt = "/net/data/idsgnn/cotrain_timesteps1000/Red_314.pth"
    
    agent = RedPPOAgent(40, red_action_space, lr, betas, gamma, K_epochs, eps_clip, start_actions=start_actions_red,
                                ckpt=red_load_ckpt, restore=True, internal=False)

    print("-------Just getting actions")
    env = CybORG(path, 'sim', agents={'Blue':(ft.partial(BlueReactRemoveAgent))})
    env = ChallengeWrapper2(env=env, agent_name='Red')
    possible_actions = env.get_possible_actions()
    print("\nBlue's actions: ", possible_actions)




    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    #for num_steps in [30, 50, 100]:
    for num_steps in [1000]:
        for bluea, name in blue_agents:
            cyborg = CybORG(path, 'sim', agents={'Blue': bluea})
            wrapped_cyborg = wrap(cyborg)
            
            observation = wrapped_cyborg.reset()
            action_space = wrapped_cyborg.get_action_space(agent_name)
            print(observation)
            print(action_space)

            total_reward = []
            actions = []
            for i in range(MAX_EPS):
                r = []
                a = []
                for j in range(num_steps):
                    while True:
                        try:
                            action = agent.get_action(observation)
                            observation, rew, done, info = wrapped_cyborg.step(action)
                            #print('\n--- Step {}'.format(j), action)
                            #reds_action = cyborg.get_last_action('Red')
                            #if str(reds_action) != "InvalidAction":
                            #    print(f"Red Agent Action: {cyborg.get_last_action('Red')}")
                            #print(f"Blue Agent Action: {cyborg.get_last_action('Blue')}\n")
                            #print(rew)
                            #print(info)

                        except KeyError as e:
                            print(f'Bad action on Episode {i}, step {j}, Action: {action}, Error: {e}. Re-rolling.')
                            #exit()
                        else:
                            break
                    r.append(rew)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

                agent.end_episode()
                total_reward.append(sum(r))
                actions.append(a)
                observation = wrapped_cyborg.reset()
            print(f'Average reward against red agent {name} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            '''
            with open(file_name, 'a+') as data:
                da ta.write(f'steps: {num_steps}, adversary: {name}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')
            with open(file_name_csv, 'a+') as data:
                data.write(f'\n{blue_agent},{name},{num_steps},{mean(total_reward)},{stdev(total_reward)}')
            '''
