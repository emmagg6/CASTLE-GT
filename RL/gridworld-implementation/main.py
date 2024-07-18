import numpy as np
import matplotlib.pyplot as plt

from env import CircleGridWorld
from agentQ import QLearningAgent
from visuals import q_table_comparison, visualize_environment, visualize_path
from train import train
from test import evaluate

import pickle


if __name__ == '__main__':

    # ------------------------------------------------
    # Parameters
    # ------------------------------------------------
    size = 15
    radius = size // 2
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # left, right, up, down
    base_goal_state = [13, 7]
    goal_states = [base_goal_state, [13, 8], [13, 9], [12, 10], [12, 11], [11, 12], 
                   [10, 12], [9, 13], [8, 13], [7, 13], [6, 13], [5, 13], [4, 12], [3, 12], [2, 11], 
                   [2, 10], [1, 9], [1,8], [1,7]]
    init_q_table = np.ones((size, size, len(actions)))
    q_tables = []
    base_q_tables = []
    trn_dists_1 = []
    trn_dists_2 = []
    trn_dists_3 = []
    trn_dists_4 = []
    trn_dists_full = []
    own_dists = []
    for state in range(len(goal_states)):
        q_tables.append([])
        trn_dists_1.append([])
        trn_dists_2.append([])
        trn_dists_3.append([])
        trn_dists_4.append([])
        trn_dists_full.append([])
        own_dists.append([])

    for trial in range(100): 
        print(f'\n\n\nTrial {trial}\n\n')

        # ------------------------------------------------
        # Base Environment & Task
        # ------------------------------------------------

        env = CircleGridWorld(size, radius, base_goal_state)
        if trial == 0:
            visualize_environment(env, env.agent_pos, base_goal_state, 'base_env.png')

        # ------------------------------------------------
        # Initialized Agent - epsilon-greedy w/ dyanmic epsilon = sqrt(e_0 / t)
        # ------------------------------------------------
        agent = QLearningAgent(size=size, actions=actions, q_table=init_q_table, alpha=0.5, gamma=0.9, epsilon=0.25)


        # ------------------------------------------------
        # Train on Base Task
        # ------------------------------------------------
        num_episodes = 1000
        print(f'Training ({num_episodes} eps)')
        lst_rewards, base_q_table = train(agent, env, num_episodes)
        base_q_tables.append(base_q_table)

        plt.plot(lst_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Results')
        plt.savefig('base_training_results.png')
        plt.close()


        # ------------------------------------------------
        # Evaluate Base Training on Base Task
        # ------------------------------------------------
        print('Evaluating')
        base_distance, states = evaluate(agent, env, base_goal_state)
        visualize_path(states, base_goal_state, size, radius, 'base_path.png')


        # ------------------------------------------------
        # Similar Tasks
        # ------------------------------------------------
        for i, goal_state in enumerate(goal_states):
            # ------------------------------------------------
            # Initialize for Similar Tasks
            # ------------------------------------------------
            print(f'For {i}-level Task Difference (Goal State: {goal_state})')
            agent.q_table = base_q_table
            env_file_name = f'env{i}.png'
            agent.epsilon = 0.25
            _ = env.reset()
            visualize_environment(env, env.agent_pos, goal_state, env_file_name)
            agent.q_table = base_q_table

            # ------------------------------------------------
            # Train for Similar Tasks & Eval on Original Task
            # ------------------------------------------------
            for trn_intervals in range(5):
                eps = num_episodes // 5
                env.goal_pos = goal_state
                rewards, q_table = train(agent, env, eps)
                if i == 0:
                    plt.plot(rewards)
                    plt.xlabel('Episode')
                    plt.ylabel('Total Reward')
                    plt.title(f'Additional Training on Same Task')
                    plt.savefig(f'base_training_results_ctn.png')
                    plt.close()
                agent.q_table = q_table
                distance, states = evaluate(agent, env, base_goal_state)
                if trn_intervals == 0:
                    trn_dists_1[i].append(distance)
                elif trn_intervals == 1:
                    trn_dists_2[i].append(distance)
                elif trn_intervals == 2:
                    trn_dists_3[i].append(distance)
                elif trn_intervals == 3:
                    trn_dists_4[i].append(distance)
                else:
                    # ------------------------------------------------
                    # Eval Trained on Similar Task on Original Task
                    # ------------------------------------------------
                    print('trained {i} level difference')
                    trn_dists_full[i].append(distance)
                    if trial == 0:
                        visualize_path(states, base_goal_state, size, radius, f'level_{i}_difference_path_on_base.png')
                    q_tables[i].append(q_table)
                    own_distances, states = evaluate(agent, env, goal_state)
                    own_dists[i].append(own_distances)

    #------------------------------------------------
    # Save all tracked results to visualize in another file
    #------------------------------------------------
    with open('base_q_tables.pkl', 'wb') as f:
        pickle.dump(base_q_tables, f)
    with open('q_tables.pkl', 'wb') as f:
        pickle.dump(q_tables, f)
    with open('trn_dists_1.pkl', 'wb') as f:
        pickle.dump(trn_dists_1, f)
    with open('trn_dists_2.pkl', 'wb') as f:
        pickle.dump(trn_dists_2, f)
    with open('trn_dists_3.pkl', 'wb') as f:
        pickle.dump(trn_dists_3, f)
    with open('trn_dists_4.pkl', 'wb') as f:
        pickle.dump(trn_dists_4, f)
    with open('trn_dists_full.pkl', 'wb') as f:
        pickle.dump(trn_dists_full, f)
    with open('own_dists.pkl', 'wb') as f:
        pickle.dump(own_dists, f)
    with open('goal_states.pkl', 'wb') as f:
        pickle.dump(goal_states, f)
    with open('base_distance.pkl', 'wb') as f:
        pickle.dump(base_distance, f)

