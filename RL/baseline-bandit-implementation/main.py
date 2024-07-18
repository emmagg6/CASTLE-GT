'''
Train and evaluate different bandit algorithms.
'''

import numpy as np
from Algorithms import EpsilonGreedy, Algorithm, UCB, GradientBandit, EXP3
from BanditEnvironment import BanditEnvironment, BanditNonstationary, AdversarialBandit
from tqdm import tqdm
from typing import Union, Tuple
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import math

def train_and_evaluate(bandit: BanditEnvironment, num_steps: int, num_runs: int, algorithm: Algorithm, regret: bool = False, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Train and evaluate the EpsilonGreedy algorithm.

    Parameters
    ----------
    bandit : BanditEnvironment
        The bandit environment.
    num_steps : int
        The number of steps to run the algorithm.
    num_runs : int
        The number of runs to average over.
    algorithm : Algorithm
        The training algorithm to use. E.g. EpsilonGreedy, UCB, etc.
    *args
        Additional arguments to pass to the algorithm. E.g. epsilon for EpsilonGreedy.

    Returns
    -------
    average_rewards : np.ndarray
        The average reward at each step.
    precents_optimal : np.ndarray
        The percentage of optimal actions the algorithm takes at each step.
    '''
    n = bandit.n

    average_measures = np.zeros(num_steps)
    precents_optimal = np.zeros(num_steps)
    regret_bound = np.zeros(num_steps)

    algorithm = algorithm(n, *args, **kwargs)
    for _ in tqdm(range(num_runs), desc=f'Running {str(algorithm)}'):
        bandit.reset()
        algorithm.reset()
        rewards = np.zeros(num_steps)

        picked_actions = np.zeros(num_steps)
        cummulative_rewards = [0] * num_steps
        for i in range(num_steps):
            action = algorithm.select_action()
            reward = bandit.get_reward(action)
            algorithm.train(action, reward)

            # If not regret, calculate rewards. Otherwise, calculate regret as G_max - G_t
            if not regret:
                rewards[i] = reward

                optimal_action = bandit.get_optimal_action()
                precents_optimal[i] += action == optimal_action
            else:
                rewards[i] = rewards[i-1] + reward if i > 0 else reward # Cummulative rewards
                cummulative_rewards[i] = bandit.get_optimal_value() # If regret - returns cummulative rewards

                picked_actions[i] = action # Prepare for calculating the optimal action

                # Calculate the weak regret bound with g >= G_max and G_max <= T => g = T
                regret_bound[i] += 2 * math.sqrt(math.e - 1) * math.sqrt(algorithm.t * n * math.log(n))

        if regret:
            optimal_action = bandit.get_optimal_action() # Overall optimal action
            cummulative_rewards = np.vstack(cummulative_rewards)[:, optimal_action]
            average_measures += cummulative_rewards - rewards # Weak regret            

            # Calculate the weak regret bound without upper bound on G_max
            # regret_bound += (math.exp(1) - 1) * algorithm.gamma * cummulative_rewards + (n * math.log(n)) / algorithm.gamma

            # Calculate the optimal action
            precents_optimal += picked_actions == optimal_action
        else:
            average_measures += rewards

    average_measures /= num_runs
    precents_optimal /= num_runs

    if regret:
        regret_bound /= num_runs
        return (average_measures, regret_bound), precents_optimal
    
    return average_measures, precents_optimal


def main():
    # epsilon_greedy()
    # ucb()
    # gradient_bandit()
    # gradient_bandit_no_baseline()
    # epsilon_greedy_optimistic()
    # nonstationary()
    
    run_exp3()

def run_exp3():
    num_steps = 1000
    num_runs = 2000
    algos = [(EXP3, (), {'time_horizon': num_steps})]

    average_regret = np.zeros((len(algos), num_steps))
    precents_optimal = np.zeros((len(algos), num_steps))
    regret_bound = np.zeros((len(algos), num_steps))

    # bandit = AdversarialBandit(10, reward_update_func=lambda q_values: np.random.rand(len(q_values)))
    bandit = AdversarialBandit(10)
    for i, (algorithm, args, kwargs) in enumerate(algos):
        (average_regret[i], regret_bound[i]), precents_optimal[i] = train_and_evaluate(bandit, num_steps, num_runs, algorithm, regret=True, *args, **kwargs)
    folder = './results/adversarial/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(folder + 'average_regret.npy', average_regret)
    np.save(folder + 'percent_optimal.npy', precents_optimal)
    print('saved average regret and percent optimal')

    # average_rewards = np.load('average_rewards.npy')
    # precents_optimal = np.load('percent_optimal.npy')

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    plt.plot(average_regret[0], label='EXP3')
    plt.plot(regret_bound[0], label='Regret Bound')
    plt.xlabel('steps')
    plt.ylabel('average regret')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(precents_optimal[0] * 100, label='EXP3')
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    # Define a formatter function for percentages
    def to_percent(y, position):
        # Format tick label as percentage
        return '{:.0f}%'.format(y)

    # Create a FuncFormatter object using the formatter function
    formatter = FuncFormatter(to_percent)

    # Apply the formatter to the y-axis of the second subplot
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.savefig(folder + 'adversarial.png')
    plt.close()

def epsilon_greedy():
    num_steps = 1000
    num_runs = 5000
    epsilons = [0, 0.01, 0.1]
    average_rewards = np.zeros((len(epsilons), num_steps))
    precents_optimal = np.zeros((len(epsilons), num_steps))
    bandit = BanditEnvironment(10)
    for i, epsilon in enumerate(epsilons):
        average_rewards[i], precents_optimal[i] = train_and_evaluate(bandit, num_steps, num_runs, EpsilonGreedy, epsilon)
    folder = './results/epsilon-greedy/'
    np.save(folder + 'average_rewards.npy', average_rewards)
    np.save(folder + 'percent_optimal.npy', precents_optimal)
    print('saved average rewards and percent optimal')

    # average_rewards = np.load('average_rewards.npy')
    # precents_optimal = np.load('percent_optimal.npy')

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    plt.plot(average_rewards[0], label='$\\epsilon = 0$')
    plt.plot(average_rewards[1], label='$\\epsilon = 0.01$')
    plt.plot(average_rewards[2], label='$\\epsilon = 0.1$')
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(precents_optimal[0] * 100, label='$\\epsilon = 0$')
    plt.plot(precents_optimal[1] * 100, label='$\\epsilon = 0.01$')
    plt.plot(precents_optimal[2] * 100, label='$\\epsilon = 0.1$')
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    # Define a formatter function for percentages
    def to_percent(y, position):
        # Format tick label as percentage
        return '{:.0f}%'.format(y)

    # Create a FuncFormatter object using the formatter function
    formatter = FuncFormatter(to_percent)

    # Apply the formatter to the y-axis of the second subplot
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.savefig(folder + 'epsilon-greedy.png')
    plt.close()

def ucb():
    num_steps = 1000
    num_runs = 2000
    algos = [(EpsilonGreedy, {'epsilon': 0.1}), (UCB, {'c': 2})]
    average_rewards = np.zeros((len(algos), num_steps))
    precents_optimal = np.zeros((len(algos), num_steps))
    bandit = BanditEnvironment(10)
    for i, (algorithm, kwargs) in enumerate(algos):
        average_rewards[i], precents_optimal[i] = train_and_evaluate(bandit, num_steps, num_runs, algorithm, **kwargs)
    folder = './results/ucb/'
    np.save(folder + 'average_rewards.npy', average_rewards)
    np.save(folder + 'percent_optimal.npy', precents_optimal)
    print('saved average rewards and percent optimal')

    # average_rewards = np.load(folder + 'average_rewards.npy')
    # precents_optimal = np.load(folder + 'percent_optimal.npy')

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    plt.plot(average_rewards[0], label='$\\epsilon = 0.1$ Epsilon-Greedy')
    plt.plot(average_rewards[1], label='$c = 2$ UCB')
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(precents_optimal[0]*100, label='$\\epsilon = 0.1$ Epsilon-Greedy')
    plt.plot(precents_optimal[1]*100, label='$c = 2$ UCB')
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    # Define a formatter function for percentages
    def to_percent(y, position):
        # Format tick label as percentage
        return '{:.0f}%'.format(y)

    # Create a FuncFormatter object using the formatter function
    formatter = FuncFormatter(to_percent)

    # Apply the formatter to the y-axis of the second subplot
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.savefig(folder + 'ucb.png')
    plt.close()

def gradient_bandit():
    num_steps = 1000
    num_runs = 2000
    algos = [(GradientBandit, (), {}), (GradientBandit, (), {'alpha': 0.4}), (GradientBandit, (), {'alpha': 0.1})]
    average_rewards = np.zeros((len(algos), num_steps))
    precents_optimal = np.zeros((len(algos), num_steps))
    bandit = BanditEnvironment(10, q_dist_func=lambda n: np.random.normal(4, 1, 10)) # Shift the true action values up +4
    for i, (algorithm, args, kwargs) in enumerate(algos):
        average_rewards[i], precents_optimal[i] = train_and_evaluate(bandit, num_steps, num_runs, algorithm, **kwargs)
    folder = './results/gradient-bandit/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(folder + 'average_rewards.npy', average_rewards)
    np.save(folder + 'percent_optimal.npy', precents_optimal)
    print('saved average rewards and percent optimal')

    # average_rewards = np.load(folder + 'average_rewards.npy')
    # precents_optimal = np.load(folder + 'percent_optimal.npy')

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    plt.plot(average_rewards[0], label='$\\alpha = 1/k$')
    plt.plot(average_rewards[1], label='$\\alpha = 0.4$')
    plt.plot(average_rewards[2], label='$\\alpha = 0.1$')
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(precents_optimal[0]*100, label='$\\alpha = 1/k$')
    plt.plot(precents_optimal[1]*100, label='$\\alpha = 0.4$')
    plt.plot(precents_optimal[2]*100, label='$\\alpha = 0.1$')
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    # Define a formatter function for percentages
    def to_percent(y, position):
        # Format tick label as percentage
        return '{:.0f}%'.format(y)

    # Create a FuncFormatter object using the formatter function
    formatter = FuncFormatter(to_percent)

    # Apply the formatter to the y-axis of the second subplot
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.savefig(folder + 'gradient-bandit.png')
    plt.close()

def gradient_bandit_no_baseline():
    num_steps = 1000
    num_runs = 2000
    algos = [(GradientBandit, (), {'alpha': 0.4}), (GradientBandit, (), {'alpha': 0.4, 'baseline': False}), 
             (GradientBandit, (), {'alpha': 0.1}), (GradientBandit, (), {'alpha': 0.1, 'baseline': False})]
    average_rewards = np.zeros((len(algos), num_steps))
    precents_optimal = np.zeros((len(algos), num_steps))
    bandit = BanditEnvironment(10, q_dist_func=lambda n: np.random.normal(4, 1, 10)) # Shift the true action values up +4
    for i, (algorithm, args, kwargs) in enumerate(algos):
        average_rewards[i], precents_optimal[i] = train_and_evaluate(bandit, num_steps, num_runs, algorithm, **kwargs)
    folder = './results/gradient-bandit_no-baseline/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(folder + 'average_rewards.npy', average_rewards)
    np.save(folder + 'percent_optimal.npy', precents_optimal)
    print('saved average rewards and percent optimal')

    # average_rewards = np.load(folder + 'average_rewards.npy')
    # precents_optimal = np.load(folder + 'percent_optimal.npy')

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    plt.plot(average_rewards[0], label='$\\alpha = 0.4$')
    plt.plot(average_rewards[1], label='$\\alpha = 0.4$ Without Baseline')
    plt.plot(average_rewards[2], label='$\\alpha = 0.1$')
    plt.plot(average_rewards[3], label='$\\alpha = 0.1$ Without Baseline')
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(precents_optimal[0]*100, label='$\\alpha = 0.4$')
    plt.plot(precents_optimal[1]*100, label='$\\alpha = 0.4$ Without Baseline')
    plt.plot(precents_optimal[2]*100, label='$\\alpha = 0.1$')
    plt.plot(precents_optimal[3]*100, label='$\\alpha = 0.1$ Without Baseline')
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    # Define a formatter function for percentages
    def to_percent(y, position):
        # Format tick label as percentage
        return '{:.0f}%'.format(y)

    # Create a FuncFormatter object using the formatter function
    formatter = FuncFormatter(to_percent)

    # Apply the formatter to the y-axis of the second subplot
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.savefig(folder + 'gradient-bandit_no-baseline.png')
    plt.close()

def epsilon_greedy_optimistic():
    num_steps = 1000
    num_runs = 2000
    algos = [(EpsilonGreedy, (), {'epsilon': 0, 'alpha': 0.1, 'q_estimates_func': lambda n: np.zeros(n) + 5}), (EpsilonGreedy, (), {'epsilon': 0.1, 'alpha': 0.1})]
    average_rewards = np.zeros((len(algos), num_steps))
    precents_optimal = np.zeros((len(algos), num_steps))
    bandit = BanditEnvironment(10)
    for i, (algorithm, args, kwargs) in enumerate(algos):
        average_rewards[i], precents_optimal[i] = train_and_evaluate(bandit, num_steps, num_runs, algorithm, *args, **kwargs)
    folder = './results/epsilon-greedy_optimistic/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(folder + 'average_rewards.npy', average_rewards)
    np.save(folder + 'percent_optimal.npy', precents_optimal)
    print('saved average rewards and percent optimal')

    # average_rewards = np.load('average_rewards.npy')
    # precents_optimal = np.load('percent_optimal.npy')

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    plt.plot(average_rewards[0], label='Optimistic $Q_1 = 5$, Greedy $\\epsilon = 0$')
    plt.plot(average_rewards[1], label='Realistic $Q_1 = 0$, Epsilon-Greedy $\\epsilon = 0.1$')
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(precents_optimal[0] * 100, label='Optimistic $Q_1 = 5$, Greedy $\\epsilon = 0$')
    plt.plot(precents_optimal[1] * 100, label='Realistic $Q_1 = 0$, Epsilon-Greedy $\\epsilon = 0.1$')
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    # Define a formatter function for percentages
    def to_percent(y, position):
        # Format tick label as percentage
        return '{:.0f}%'.format(y)

    # Create a FuncFormatter object using the formatter function
    formatter = FuncFormatter(to_percent)

    # Apply the formatter to the y-axis of the second subplot
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.savefig(folder + 'epsilon-greedy_optimistic.png')
    plt.close()

def nonstationary():
    num_steps = 1000
    num_runs = 2000
    algos = [(EpsilonGreedy, (), {'epsilon': 0, 'alpha': 0.1, 'q_estimates_func': lambda n: np.zeros(n) + 5}), 
             (EpsilonGreedy, (), {'epsilon': 0.1, 'alpha': 0.1}), (EpsilonGreedy, (), {'epsilon': 0.1}), 
             (UCB, (), {'c': 2}), (GradientBandit, (), {'alpha': 0.1}), (GradientBandit, (), {})]
    average_rewards = np.zeros((len(algos), num_steps))
    precents_optimal = np.zeros((len(algos), num_steps))
    bandit = BanditNonstationary(10)
    for i, (algorithm, args, kwargs) in enumerate(algos):
        average_rewards[i], precents_optimal[i] = train_and_evaluate(bandit, num_steps, num_runs, algorithm, *args, **kwargs)
    folder = './results/nonstationary/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(folder + 'average_rewards.npy', average_rewards)
    np.save(folder + 'percent_optimal.npy', precents_optimal)
    print('saved average rewards and percent optimal')

    # average_rewards = np.load('average_rewards.npy')
    # precents_optimal = np.load('percent_optimal.npy')

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    plt.plot(average_rewards[0], label='Optimistic $Q_1 = 5$, Greedy $\\epsilon = 0$, $\\alpha = 0.1$')
    plt.plot(average_rewards[1], label='Epsilon-Greedy $\\epsilon = 0.1$, $\\alpha = 0.1$')
    plt.plot(average_rewards[2], label='Epsilon-Greedy $\\epsilon = 0.1$')
    plt.plot(average_rewards[3], label='UCB $c = 2$')
    plt.plot(average_rewards[4], label='GradientBandit $\\alpha = 0.1$')
    plt.plot(average_rewards[5], label='GradientBandit')
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(precents_optimal[0] * 100, label='Optimistic $Q_1 = 5$, Greedy $\\epsilon = 0$, $\\alpha = 0.1$')
    plt.plot(precents_optimal[1] * 100, label='Epsilon-Greedy $\\epsilon = 0.1$, $\\alpha = 0.1$')
    plt.plot(precents_optimal[2] * 100, label='Epsilon-Greedy $\\epsilon = 0.1$')
    plt.plot(precents_optimal[3] * 100, label='UCB $c = 2$')
    plt.plot(precents_optimal[4] * 100, label='GradientBandit $\\alpha = 0.1$')
    plt.plot(precents_optimal[5] * 100, label='GradientBandit')
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    # Define a formatter function for percentages
    def to_percent(y, position):
        # Format tick label as percentage
        return '{:.0f}%'.format(y)

    # Create a FuncFormatter object using the formatter function
    formatter = FuncFormatter(to_percent)

    # Apply the formatter to the y-axis of the second subplot
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.savefig(folder + 'nonstationary.png')
    plt.close()

if __name__ == '__main__':
    main()