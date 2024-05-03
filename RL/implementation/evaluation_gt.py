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
from Agents.BlueAgents.GenericAgentGT import GenericAgent
#from Agents.BlueAgents.GenericLongAgent import GenericLongAgent
import random

MAX_EPS = 1000
agent_name = 'B-PPOxCCE'
random.seed(0)


blue_agent = 'cce-bline'
balance_points = list(range(11000, 20000, 1000))
# balance_points = [1000000000] # for just PPO

# load blue agent
for balance_point in balance_points:
    agent = GenericAgent(model_dir=blue_agent, balance=balance_point)

    selected_actions = []

    # changed to ChallengeWrapper2
    def wrap(env):
        return ChallengeWrapper2(env=env, agent_name='Blue')

    def get_git_revision_hash() -> str:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

    if __name__ == "__main__":
        cyborg_version = CYBORG_VERSION
        scenario = 'Scenario2'
        # commit_hash = get_git_revision_hash()
        commit_hash = "Not using git"
        # ask for a name
        name = "Dartmouth_Northeastern"
        # ask for a team
        team = "BlueSTAR"
        # ask for a name for the agent
        name_of_agent = "PPO + Greedy decoys + GT"

        lines = inspect.getsource(wrap)
        wrap_line = lines.split('\n')[1].split('return ')[1]


        print(f'Using agent {agent_name}, if this is incorrect please update the code to load in your agent')

        file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' f'{agent_name}_balance{balance_point}.txt'
        print(f'Saving evaluation results to {file_name}')
        with open(file_name, 'a+') as data:
            data.write(f'CybORG v{cyborg_version}, {scenario}, Commit Hash: {commit_hash}\n')
            data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
            data.write(f"wrappers: {wrap_line}\n")
            data.write(f"Balance Point, {balance_point}\n\n\n")

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

        cce_action_cnts, ppo_action_cnts = [], []

        print(f'using CybORG v{cyborg_version}, {scenario}\n')
        for i, num_steps in enumerate([30, 50, 100]):

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
                                action_t = agent.get_action(observation, action_space)
                                observation_t, reward_t, done, info = wrapped_cyborg.step(action_t)
                                # print(f"Blue Agent Action: {cyborg.get_last_action('Blue')}")
                                # print(f"Red Agent Action: {cyborg.get_last_action('Red')}")
                            except KeyError as e:
                                print(f'Bad action on Episode {i}, step {j}, Action: {action_t}, Error: {e}. Re-rolling.')
                            else:
                                break
                        # result = cyborg.step(agent_name, action)
                        r.append(reward_t)
                        # r.append(result.reward)
                        a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))
                        selected_actions.append(action_t)

                    agent.end_episode()
                    total_reward.append(sum(r))
                    actions.append(a)
                    # observation = cyborg.reset().observation
                    observation = wrapped_cyborg.reset()

                print('\n========Final: Average Blue Policy Proportions')
                ppo_action_cnt, cce_action_cnt = agent.get_proportions()
                ppo_action_cnts.append(ppo_action_cnt)
                cce_action_cnts.append(cce_action_cnt)
                print(f'PPO: {ppo_action_cnt}, CCE: {cce_action_cnt}, CCE precent: {round(cce_action_cnt / (ppo_action_cnt + cce_action_cnt), 2)}\n')

                
                
                print(f'Average reward for red agent {name} and steps {num_steps} is: {round(mean(total_reward), 2)} with a standard deviation {stdev(r)}')
                with open(file_name, 'a+') as data:
                    data.write(f'\nsteps: {num_steps}, adversary: {name}, mean: {mean(total_reward)}, standard deviation {stdev(r)}\n')
                    # data.write(f'Balance point: {balance_point}, PPO action-selection count: {ppo_action_cnt}, CCE action-selection count: {cce_action_cnt}, CCE precent: {round(cce_action_cnt / (ppo_action_cnt + cce_action_cnt), 2)}\n')

        for i, num_steps in enumerate([30, 50, 100]):
            with open(file_name, 'a+') as data:
                data.write('\n')
                data.write(f'Steps: {num_steps}: PPO action-selection count: {ppo_action_cnt}, CCE action-selection count: {cce_action_cnt}, CCE precent: {round(cce_action_cnt / (ppo_action_cnt + cce_action_cnt), 2)}\n')



    with open(file_name, 'a+') as data:
        data.write('\n\n\n')
        data.write(f'Note: Red is Meander Agent. Balance point: {balance_point}, PPO: 10000.pth, CCE: 10000cce.pkl \n')
    #     # data.write(f'\n\n\nActions: {selected_actions}\n')