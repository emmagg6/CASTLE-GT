import subprocess
import inspect
import time,os
import torch
from statistics import mean, stdev

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import BlueReactRemoveAgent,BlueReactRestoreAgent,BlueMonitorAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2


from Agents.RedAgents.PPOAgent import PPOAgent as RedPPOAgent
import random

MAX_EPS = 1000
agent_name = 'Red'
random.seed(0)

# changed to ChallengeWrapper2
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name='Red')

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    # commit_hash = get_git_revision_hash()
    commit_hash = "Not using git"
    # ask for a name
    name = ""
    # ask for a team
    team = ""
    # ask for a name for the agent
    name_of_agent = "PPO"

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'
    
    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in [30, 50, 100]:
        for blue_agent in [BlueReactRemoveAgent,BlueReactRestoreAgent,BlueMonitorAgent]:

            cyborg = CybORG(path, 'sim', agents={'Blue': blue_agent})
            wrapped_cyborg = wrap(cyborg)

            # LOAD RED AGENT

            input_dims = wrapped_cyborg.observation_space.shape[0]
            action_space=list(range(888))
            start_actions = [] # user 2 decoy * 2, ent0 decoy
            agent = RedPPOAgent(input_dims, action_space, start_actions=start_actions)

            folder = 'redppo'
            ckpt_folder = os.path.join(os.getcwd(), "Models", folder)
            load_epoch=42000 # 50000
            load_dir = os.path.join(ckpt_folder, f'{load_epoch}.pth')
            load_dir_old = os.path.join(ckpt_folder, f'{load_epoch}_old.pth')
            if os.path.exists(load_dir):
                agent.policy.load_state_dict(torch.load(load_dir, map_location='cpu'))
                agent.old_policy.load_state_dict(torch.load(load_dir_old, map_location='cpu'))
                print('Checkpoint loaded',load_dir)
            else:
                raise ValueError(f'No checkpoint found at {load_dir}')

            print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

            
            eval_dir = os.path.join(os.getcwd(), "Evaluation")
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)
            file_name = './Evaluation/' + time.strftime("%Y%m%d_%H%M%S") + f'_{folder}.txt'
            print(f'Saving evaluation results to {file_name}')
            with open(file_name, 'a+') as data:
                data.write(f'CybORG v{cyborg_version}, {scenario}, Commit Hash: {commit_hash}\n')
                data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
                data.write(f"wrappers: {wrap_line}\n")



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
                    while True:
                        try:
                            observation, rew, done, info = wrapped_cyborg.step(action)
                            break
                        except: # if action is buggy
                            action=random.choice(agent.action_space)
                    # result = cyborg.step(agent_name, action)
                    r.append(rew)
                    # r.append(result.reward)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))

                agent.end_episode()
                total_reward.append(sum(r))
                actions.append(a)
                # observation = cyborg.reset().observation
                observation = wrapped_cyborg.reset()
            print(f'Average reward for blue agent {blue_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            with open(file_name, 'a+') as data:
                data.write(f'steps: {num_steps}, adversary: {blue_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')



#   File "/scratch/ows/CybORG/CybORG-2.1/CybORG/Shared/Actions/AbstractActions/PrivilegeEscalate.py", line 44, in get_escalate_action
#     if state.sessions[agent][session].operating_system[hostname] == OperatingSystemType.WINDOWS:
# KeyError: 'User1

