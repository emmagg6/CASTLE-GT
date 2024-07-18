import numpy as np

def softmax(q_values, temperature=1.0):
    """Compute softmax values for each set of scores in q_values."""
    max_q = np.max(q_values)
    exp_q = np.exp((q_values - max_q) / temperature)
    return exp_q / np.sum(exp_q)

def evaluate(agent, env, goal_state, max_episodes=10000, temperature=1.0):
    done = False
    state = env.reset()
    env.goal_pos = goal_state
    steps = 0
    states = []
    episode = 0
    while not done and episode < max_episodes:
        q_values = [agent.q_table[state[0], state[1], i] for i in range(len(agent.actions))]
        action_probs = softmax(q_values, temperature)
        action_index = np.random.choice(range(len(agent.actions)), p=action_probs)
        action = agent.actions[action_index]

        state, _, done = env.step(action)
        steps += 1
        states.append(state)
        episode += 1

    distance = np.sqrt((state[0] - goal_state[0]) ** 2 + (state[1] - goal_state[1]) ** 2)

    return distance, states