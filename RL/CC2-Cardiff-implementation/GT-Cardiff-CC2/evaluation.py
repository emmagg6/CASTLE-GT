import subprocess
import inspect
import time
from statistics import mean, stdev

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
# from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.MainAgentGT import MainAgent
import random

MAX_EPS = 1000
agent_name = 'PPOxCCE'
random.seed(0)

zetas = list(range(30000, 35000, 500))
# zetas = [10]

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
    name = "ppoxcce"
    # ask for a team
    team = "BlueSTAR"
    # ask for a name for the agent
    name_of_agent = "PPO + Greedy decoys + CCE"

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    for zeta in zetas : 
        print(f'Running evaluation for zeta = {zeta}')

        # Change this line to load your agent
        agent = MainAgent(model_dir=agent_name, balance=zeta)

        print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

        # file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
        file_name = '/scratch/egraham/CASTLE-GT/RL/CC2-Cardiff-implementation/GT-Cardiff-CC2/Evaluation/' + f'PPOxCCE_z{zeta}.txt'

        print(f'Saving evaluation results to {file_name}')
        with open(file_name, 'a+') as data:
            data.write(f'CybORG v{cyborg_version}, {scenario}, Commit Hash: {commit_hash}\n')
            data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
            data.write(f"wrappers: {wrap_line}\n")

        path = str(inspect.getfile(CybORG))
        path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

        print(f'using CybORG v{cyborg_version}, {scenario}\n')

        for num_steps in [30, 50, 100]:
            for red_agent in [B_lineAgent]:

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
                        action = agent.get_action(observation, action_space)
                        observation, rew, done, info = wrapped_cyborg.step(action)
                        # result = cyborg.step(agent_name, action)
                        r.append(rew)
                        # r.append(result.reward)
                        a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

                    agent.end_episode()
                    total_reward.append(sum(r))
                    actions.append(a)
                    # observation = cyborg.reset().observation
                    observation = wrapped_cyborg.reset()
                print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
                print('\n========Final: Average Blue Policy Proportions')
                ppo_action_cnt, cce_action_cnt = agent.get_proportions()
                print(f'PPO: {ppo_action_cnt}, CCE: {cce_action_cnt}, CCE precent: {round(cce_action_cnt / (ppo_action_cnt + cce_action_cnt), 2)}\n')
                with open(file_name, 'a+') as data:
                    data.write(f'\nsteps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}, proportion CCE: {round(cce_action_cnt / (ppo_action_cnt + cce_action_cnt), 2)}\n')
                    # for act, sum_rew in zip(actions, total_reward):
                        # data.write(f'actions: {act}, total reward: {sum_rew}')
