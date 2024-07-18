from tqdm import tqdm
import numpy as np


def train(agent, env, num_episodes=1000):
    rewards = []
    with tqdm(total=num_episodes) as pbar:
        for episode in range(num_episodes):
            pbar.set_description(f'Episode {episode + 1}')
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action, action_index = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.update_q_table(state, action_index, reward, next_state)
                total_reward += reward
                state = next_state
            rewards.append(total_reward)
            pbar.update(1)
            agent.epsilon = np.sqrt(agent.epsilon / (episode + 1))
    return rewards, agent.q_table
