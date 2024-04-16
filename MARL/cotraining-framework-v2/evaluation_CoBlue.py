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

from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent

MAX_EPS = 1000
#MAX_EPS = 1
agent_name = 'Blue'
random.seed(0)

# PPO Params
K_epochs = 6
eps_clip = 0.2 # not used in evaluation
gamma = 0.99 # not used in evaluation
betas=[0.9, 0.990] # not used in evaluation
lr = 0.002 # not used in evaluation


# changed to ChallengeWrapper2
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name='Blue')

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

    # load blue agent
    '''
    blue_agent = 'BlueGen'
    agent = GenericAgent(model_dir=blue_agent)
    '''

    blue_agent = 'CoBlue'
    
    # define ppo blue agent
    #start_actions_blue = [1004, 1004, 1000] # user 2 decoy * 2, ent0 decoy

    start_actions_blue = [51, 116, 55] # based on GenericAgent

    blue_action_space = [133, 134, 135, 139]  # restore enterprise and opserver
    blue_action_space += [3, 4, 5, 9]  # analyse enterprise and opserver
    blue_action_space += [16, 17, 18, 22]  # remove enterprise and opserer
    blue_action_space += [11, 12, 13, 14]  # analyse user hosts
    blue_action_space += [141, 142, 143, 144]  # restore user hosts
    blue_action_space += [132]  # restore defender
    blue_action_space += [2]  # analyse defender
    blue_action_space += [15, 24, 25, 26, 27]  # remove defender and user hosts

    #blue_load_ckpt = "/net/data/idsgnn/cotrain_timesteps1000/Blue_317.pth"
    blue_load_ckpt = "/net/data/idsgnn/cotrain_both/Blue_25000.pth"
    
    agent = BluePPOAgent(52, blue_action_space, lr, betas, gamma, K_epochs, eps_clip, 
                                start_actions=start_actions_blue, restore=True,  # need to set restore!!
                                ckpt=blue_load_ckpt, internal=False)

    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

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
    red_agents = [#(ft.partial(B_lineAgent), 'B_lineAgent'), 
                  (ft.partial(RedMeanderAgent), 'RedMeanderAgent'), 
                 # (ft.partial(SleepAgent), 'SleepAgent'),
                  ]

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    #for num_steps in [30, 50, 100]:
    for num_steps in [100]:
        for red_agent, name in red_agents:
            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg)
            
            observation = wrapped_cyborg.reset()
            #action_space = wrapped_cyborg.get_action_space(agent_name)
            #print(observation)
            #print(action_space)

            total_reward = []
            actions = []
            for i in range(MAX_EPS):
                r = []
                a = []
                for j in range(num_steps):
                    while True:
                        try:
                            agent.action_space_dict = wrapped_cyborg.get_action_space_dict('Blue')
                            agent.observation_dict = wrapped_cyborg.get_observation_dict('Blue')

                            action = agent.get_action(observation)
                            observation, rew, done, info = wrapped_cyborg.step(action)
                            #print('\n--- Step {}'.format(j), action)
                            #reds_action = cyborg.get_last_action('Red')
                            #if str(reds_action) != "InvalidAction":
                            #print(f"Red Agent Action: {cyborg.get_last_action('Red')}")
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
