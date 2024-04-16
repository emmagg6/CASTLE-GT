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
from Agents.BlueAgents.MainAgent import MainAgent
from Agents.BlueAgents.GenericAgentGT import GenericAgent
import random
from pprint import pprint

MAX_EPS = 1000
agent_name = 'PPOxCCE'
random.seed(0)


# changed to ChallengeWrapper2
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name='Blue')

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2_Operations'
    #scenario = 'Scenario2_User5'
    #scenario = 'Scenario2'
    commit_hash = "Not using git"
    name = "noname"
    team = "Dartmouth_and_Northeastern"
    # ask for a name for the agent
    name_of_agent = "PPOxCCE"

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    # load blue agent
    blue_agent = 'gt-test'

    # agent = GenericAgent(model_dir_PPO=blue_agent)
    agent = GenericAgent(model_dir_PPO=blue_agent, model_file_PPO='/scratch/egraham/CASTLE-GT/RL/operations/extensions/Models/gt-test/25000.pth',
                         model_file_GT='/scratch/egraham/CASTLE-GT/RL/operations/extensions/Models/gt-test/25000.cce.pth')

    #blue_agent = 'cardiff'
    #agent = MainAgent()

    print("model: ", blue_agent)
    
    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'


    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{cyborg_version}, {scenario}, Commit Hash: {commit_hash}\n')
        data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n")


    '''
    file_name_csv = './results/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.csv'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name_csv, 'a+') as data:
        data.write('blue_agent,red_agent,steps,avg_reward,std')
    
    red_agents = [(partial(B_lineAgent), 'B_lineAgent'), 
                  (partial(RedMeanderAgent), 'RedMeanderAgent'), 
                  (partial(SleepAgent), 'SleepAgent'),
                  ]
    '''

    #red_agents = [(partial(EvadeBySleep, sub_agent=B_lineAgent), 'EvadeBySleep_Bline')]
    red_agents = [(partial(B_lineAgent), 'B_lineAgent')] 
    #red_agents = [(partial(RedMeanderAgent), 'RedMeanderAgent')] 

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    
    finalrew = {}
    final_action_counts = {}
    final_reward_breakdown = {}

    #for num_steps in [30,50,100]:
    for num_steps in [1]:
        for red_agent, name in red_agents:
            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            hostnames = list(cyborg.get_ip_map().keys())
            print(hostnames)

            wrapped_cyborg = wrap(cyborg)

            observation = wrapped_cyborg.reset()
            action_space = wrapped_cyborg.get_action_space(agent_name)
            total_reward = []
            actions = []
            total_action_counts = {}
            total_reward_breakdown = {}
            total_reward_breakdown['confidentiality'] = []
            total_reward_breakdown['availability'] = []
            total_reward_breakdown['operations'] = []

            for i in range(MAX_EPS):
                r = []
                a = []
                hosts_actions = {}
                reward_breakdown = {}
                reward_breakdown['confidentiality'] = []
                reward_breakdown['availability'] = []
                reward_breakdown['operations'] = []

                for j in range(num_steps):
                    while True:
                        try:
                            #print('\n--- Step {}, Before step: actions, action_space, observation: '.format(j), actions, action_space, observation)
                            action = agent.get_action(observation, action_space)
                            observation, rew, done, info = wrapped_cyborg.step(action)
                            #print('\n--- Step {}'.format(j), action)
                            #print(f"Red Agent Action: {cyborg.get_last_action('Red')} {cyborg.get_reward_breakdown('Red')}\n")
                            #print(f"Blue Agent Action: {cyborg.get_last_action('Blue')} {cyborg.get_reward_breakdown('Blue')}\n")
                            #print(rew)
                            #print(info)
                        except KeyError as e:
                            print(f'Bad action on Episode {i}, step {j}, Action: {action}, Error: {e}. Re-rolling.')
                            exit()
                        else:
                            break
                    
                    # statistics for the current step
                    baction = str(cyborg.get_last_action('Blue')).split()
                    #if baction[0] == 'Restore':
                    #    print(f"Red Agent Action: {cyborg.get_last_action('Red')} {cyborg.get_reward_breakdown('Red')}\n")
                    #    print(f"Blue Agent Action: {cyborg.get_last_action('Blue')} {cyborg.get_reward_breakdown('Blue')}\n")
                    #    print(rew)
                    #    print(info)

                    if baction[0] not in hosts_actions:
                            hosts_actions[baction[0]] = {}
                    
                    if len(baction) == 1:
                        where = 'blue'
                    else:
                        where = baction[1]

                    if where not in hosts_actions[baction[0]]:
                        hosts_actions[baction[0]][where] = 0
                    hosts_actions[baction[0]][where] += 1

                    raction = str(cyborg.get_last_action('Red')).split()
                    if raction[0] == 'Impact':
                        if raction[0] not in hosts_actions:
                            hosts_actions[raction[0]] = {'red': 0}
                        hosts_actions[raction[0]]['red'] += 1

                    # compute reward by reward type
                    # discrepancy between rew (computed before blue's action, but after red) 
                    # and cyborg.get_reward_breakdown (computed after both red/blue finished
                    # if red does priv esc followed by blue restore can show rew=-1.1, but cyborg.get_reward_breakdown=-0.1
                    rbreakdown = cyborg.get_reward_breakdown('Blue')
                    rc = 0
                    ra = 0
                    ro = 0
                    for hn in hostnames:  # sum over all hosts
                        # print(hn, rbreakdown[hn][0])
                        rc += rbreakdown[hn][0]
                        ra += rbreakdown[hn][1]
                        if len(rbreakdown[hn]) > 2:
                            ro += rbreakdown[hn][2]
                      
                    reward_breakdown['confidentiality'].append(rc)
                    reward_breakdown['availability'].append(ra)
                    reward_breakdown['operations'].append(ro)

                    # print('After step: observation: ', observation)
                    # print('info: ', info)
                    r.append(rew)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))
                    #print(reward_breakdown)
                    #print(r)
                agent.end_episode()

                # statistiics for the current episode
                # sum across all steps that occurred during this episode
                # save the reward per episode
                total_reward.append(sum(r))
                for elem in reward_breakdown:
                    total_reward_breakdown[elem].append(sum(reward_breakdown[elem]))

                actions.append(a)
                observation = wrapped_cyborg.reset()
                #print(hosts_actions)
                for ba in hosts_actions:
                    if ba not in total_action_counts:
                        total_action_counts[ba] = {}
                    for bh in hosts_actions[ba]:
                        if bh not in total_action_counts[ba]:
                            total_action_counts[ba][bh] = []
                    total_action_counts[ba][bh].append(hosts_actions[ba][bh])

            # average across all episodes
            finalrew[num_steps] = round(mean(total_reward), 2)

            final_reward_breakdown[num_steps] = {}
            for elem in total_reward_breakdown:
                final_reward_breakdown[num_steps][elem] = round(mean(total_reward_breakdown[elem]), 2)

            final_action_counts[num_steps] = {}
            
            print(f'\n========Average reward for red agent {name} and steps {num_steps} is: {round(mean(total_reward), 2)}')
            for ba in total_action_counts:
                final_action_counts[num_steps][ba] = {}
                print(f'\n{ba}:')
                for bh in total_action_counts[ba]:
                    final_action_counts[num_steps][ba][bh] = round(sum(total_action_counts[ba][bh]) / MAX_EPS, 2)
                    print(f'{bh}: {sum(total_action_counts[ba][bh])}')
           
            print('\n========Final: Average Blue Policy Proportions')
            ppo_action_cnt, cce_action_cnt = agent.get_proportions()
            print(f'PPO: {ppo_action_cnt}, CCE: {cce_action_cnt}, CCE precent: {round(cce_action_cnt / (ppo_action_cnt + cce_action_cnt), 2)}\n')

            
            
            print(f'Average reward for red agent {name} and steps {num_steps} is: {round(mean(total_reward), 2)} with a standard deviation of {round(stdev(total_reward), 2)}')
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {name}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')

            '''
            with open(file_name_csv, 'a+') as data:
                data.write(f'\n{blue_agent},{name},{num_steps},{mean(total_reward)},{stdev(total_reward)}')
            '''

    print(f'\n========Final: Average Blue reward against red agent {name}')
    print(finalrew)
    pprint(final_reward_breakdown)
    pprint(final_action_counts)


