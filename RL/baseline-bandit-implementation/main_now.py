'''
Train and evaluate different bandit algorithms.
'''

import numpy as np
from Algorithms import EpsilonGreedy, Algorithm, UCB, GradientBandit, EXP3, EXP3IX, EXP3IXrl
from BanditEnvironment import BanditEnvironment, BanditNonstationary, AdversarialBandit
from tqdm import tqdm
from typing import Union, Tuple
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import math

def train_and_evaluate(bandit: BanditEnvironment, num_steps: int, num_runs: int, algorithm: Algorithm, regret: bool = False, trial=False, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
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
    regret : bool, optional
        If True, calculate based on regret. Defaults to False.
    trial : bool, optional
        If True, use a hardcoded switch of optimal action. Defaults to False.
    *args, **kwargs
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

                if trial:
                    optimal_action = 8 if i < num_steps/2 else 9 #bandit.get_optimal_action()
                    cummulative_rewards[i] = cummulative_rewards[i][optimal_action]
                    precents_optimal[i] += action == optimal_action

                picked_actions[i] = action # Prepare for calculating the optimal action

                # Calculate the weak regret bound with g >= G_max and G_max <= T -> g = T
                regret_bound[i] += 2 * math.sqrt(math.e - 1) * math.sqrt(algorithm.t * n * math.log(n))

        if regret:
            if not trial:
                optimal_action = bandit.get_optimal_action() # Overall optimal action
                # print(f'Optimal action: {optimal_action + 1}')
                cummulative_rewards = np.vstack(cummulative_rewards)[:, optimal_action]
            average_measures += cummulative_rewards - rewards # Weak regret            

            # Calculate the weak regret bound without upper bound on G_max
            # regret_bound += (math.exp(1) - 1) * algorithm.gamma * cummulative_rewards + (n * math.log(n)) / algorithm.gamma

            if not trial:
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

def train_and_evaluate_integrated(gt_algo: EXP3IXrl, bandit: BanditEnvironment, num_steps: int, num_runs: int, rl_algo: Algorithm, certainty: int, regret: bool = True, trial=False, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Train and evaluate the bandit algorithms integrated with Exp3IXrl.

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
    regret : bool, optional
        If True, calculate based on regret. Defaults to True since set up for True in main().
    trial : bool, optional
        If True, use a hardcoded switch of optimal action. Defaults to False.
    *args, **kwargs
        Additional arguments to pass to the algorithm. E.g. epsilon for EpsilonGreedy.

    Returns
    -------
    average_rewards : np.ndarray
        The average reward at each step.
    precents_optimal : np.ndarray
        The percentage of optimal actions the algorithm takes at each step.
    '''
    n = bandit.n
    print("pringing out n = bandit.n", n)

    average_measures = np.zeros(num_steps)
    average_measures_exp3ix = np.zeros(num_steps)
    precents_optimal = np.zeros(num_steps)
    precents_optimal_exp3ix = np.zeros(num_steps)
    regret_bound = np.zeros(num_steps)

    # Filter out kwargs that are not needed by the current algorithm
    if rl_algo == EpsilonGreedy:
        algo_kwargs = {k: v for k, v in kwargs.items() if k in ['epsilon', 'alpha']}
    elif rl_algo == UCB:
        algo_kwargs = {k: v for k, v in kwargs.items() if k in ['c']}
    elif rl_algo == GradientBandit:
        algo_kwargs = {k: v for k, v in kwargs.items() if k in ['alpha', 'baseline']}
    else:
        algo_kwargs = kwargs

    print("printing out rl_algo", rl_algo)
    print("printing out algo_kwargs", algo_kwargs)
    rl_algo = rl_algo(n, *args, **algo_kwargs)
    gt_algo = gt_algo()

    trained_rl_algos = []
    trained_gt_algos = []

    #   TRAINING   #
    for _ in tqdm(range(num_runs), desc=f'Training {str(gt_algo)} with {str(rl_algo)}'):
        bandit.reset()
        rl_algo.reset()
        gt_algo.reset()
        rewards = np.zeros(num_steps)

        picked_actions = np.zeros(num_steps)
        cummulative_rewards = [0] * num_steps
        for i in range(num_steps):
            action = rl_algo.select_action()
            reward = bandit.get_reward(action)
            rl_algo.train(action, reward)
            # Integrate Exp3IXrl training
            gt_algo.train_step(act = action, rew = reward)

            # If not regret, calculate rewards. Otherwise, calculate regret as G_max - G_t
            if not regret:
                rewards[i] = reward

                optimal_action = bandit.get_optimal_action()
                precents_optimal[i] += action == optimal_action
            else:
                rewards[i] = rewards[i-1] + reward if i > 0 else reward # Cummulative rewards
                cummulative_rewards[i] = bandit.get_optimal_value() # If regret - returns cummulative rewards

                if trial:
                    optimal_action = 8 if i < num_steps/2 else 9 #bandit.get_optimal_action()
                    cummulative_rewards[i] = cummulative_rewards[i][optimal_action]
                    precents_optimal[i] += action == optimal_action

                picked_actions[i] = action # Prepare for calculating the optimal action

                # Calculate the weak regret bound with g >= G_max and G_max <= T -> g = T
                regret_bound[i] += 2 * math.sqrt(math.e - 1) * math.sqrt(rl_algo.t * n * math.log(n))

            
        
        # Trained Algos
        trained_rl_algos.append(rl_algo)
        eq, visits = gt_algo.get_equilibrium()
        trained_gt_algos.append((eq, visits))


        if regret:
            if not trial:
                optimal_action = bandit.get_optimal_action() # Overall optimal action
                # print(f'Optimal action: {optimal_action + 1}')
                cummulative_rewards = np.vstack(cummulative_rewards)[:, optimal_action]
            average_measures += cummulative_rewards - rewards # Weak regret            

            # Calculate the weak regret bound without upper bound on G_max
            # regret_bound += (math.exp(1) - 1) * algorithm.gamma * cummulative_rewards + (n * math.log(n)) / algorithm.gamma

            if not trial:
                # Calculate the optimal action
                precents_optimal += picked_actions == optimal_action
        else:
            average_measures += rewards

    average_measures /= num_runs
    precents_optimal /= num_runs

    if regret:
        regret_bound /= num_runs
        # return (average_measures, regret_bound), precents_optimal
    
    #   EVALUATION   #
    for i, trained_rl_algo in enumerate(trained_rl_algos):
        eq, visits = trained_gt_algos[i]
        for _ in tqdm(range(num_runs), desc=f'Evaluating {str(gt_algo)} with {str(trained_rl_algo)}'):
            bandit.reset()
            rewards = np.zeros(num_steps)

            picked_actions = np.zeros(num_steps)
            exp3ixrl_picked_actions_cnt = np.zeros(num_steps)
            cummulative_rewards = [0] * num_steps
            for i in range(num_steps):
                action = gt_algo.select_action(eq, visits, certainty)
                if action == -1:
                    action = trained_rl_algo.select_action()
                else:
                    exp3ixrl_picked_actions_cnt += 1
                reward = bandit.get_reward(action)

                # If not regret, calculate rewards. Otherwise, calculate regret as G_max - G_t
                if not regret:
                    rewards[i] = reward

                    optimal_action = bandit.get_optimal_action()
                    precents_optimal[i] += action == optimal_action
                else:
                    rewards[i] = rewards[i-1] + reward if i > 0 else reward

            if regret:
                if not trial:
                    optimal_action = bandit.get_optimal_action() # Overall optimal action
                    # print(f'Optimal action: {optimal_action + 1}')
                    cummulative_rewards = np.vstack(cummulative_rewards)[:, optimal_action]
                average_measures_exp3ix += cummulative_rewards - rewards # Weak regret            

                # Calculate the weak regret bound without upper bound on G_max
                # regret_bound += (math.exp(1) - 1) * algorithm.gamma * cummulative_rewards + (n * math.log(n)) / algorithm.gamma

                if not trial:
                    # Calculate the optimal action
                    precents_optimal_exp3ix += picked_actions == optimal_action
            else:
                average_measures_exp3ix += rewards

    average_measures_exp3ix /= num_runs
    precents_optimal_exp3ix /= num_runs
    precent_exp3ixrl_picked_actions = exp3ixrl_picked_actions_cnt / num_runs

    if regret:
        regret_bound /= num_runs
        return (average_measures, regret_bound), precents_optimal, (average_measures_exp3ix, regret_bound), precents_optimal_exp3ix, precent_exp3ixrl_picked_actions
    
    return average_measures, precents_optimal, average_measures_exp3ix, precents_optimal_exp3ix


def main():
    # epsilon_greedy()
    # ucb()
    # gradient_bandit()
    # gradient_bandit_no_baseline()
    # epsilon_greedy_optimistic()
    # nonstationary()
    
    # run_exp3()
    # run_exp3ix_adversarial()
    # exp3ix_trial()
    # exp3ix_dynamic()
    gt_all()

def gt_all():
    num_steps = 10000
    num_runs = 2000
    algos = [(EXP3, (), {'time_horizon': num_steps}), (EXP3IX, (), {'time_horizon': num_steps})]
    # ['epsilon', 'alpha', 'q_estimates_func', 'c', 'baseline', 'preference_func']
    ours = (EXP3IXrl, (), {'time_horizon': num_steps, 'epsilon': 0.1, 'alpha': 0.1, 'dynamic_eta': True, 'c': 2, 'baseline': False})
    action_selectors = [EpsilonGreedy, UCB, GradientBandit]
    certainty = [1]

    average_regret = np.zeros((len(algos), num_steps))
    precents_optimal = np.zeros((len(algos), num_steps))
    regret_bound = np.zeros((len(algos), num_steps))

    average_regret_greedyselector = np.zeros(num_steps)
    precents_optimal_greedyselector = np.zeros(num_steps)
    regret_bound_greedyselector = np.zeros(num_steps)

    average_regret_ucbselector = np.zeros(num_steps)
    precents_optimal_ucbselector = np.zeros(num_steps)
    regret_bound_ucbselector = np.zeros(num_steps)

    average_regret_gradientselector = np.zeros(num_steps)
    precents_optimal_gradientselector = np.zeros(num_steps)
    regret_bound_gradientselector = np.zeros(num_steps)

    average_regret_ours_greedy = np.zeros((len(ours), num_steps))
    precents_optimal_ours_greedy = np.zeros((len(ours), num_steps))
    regret_bound_ours_greedy = np.zeros((len(ours), num_steps))

    average_regret_ours_ucb = np.zeros((len(ours), num_steps))
    precents_optimal_ours_ucb = np.zeros((len(ours), num_steps))
    regret_bound_ours_ucb = np.zeros((len(ours), num_steps))

    average_regret_ours_gradient = np.zeros((len(ours), num_steps))
    precents_optimal_ours_gradient = np.zeros((len(ours), num_steps))
    regret_bound_ours_gradient = np.zeros((len(ours), num_steps))




    bandit = AdversarialBandit(10, reward_update_func=lambda q_values, t: np.concatenate([np.random.binomial(1, 0.5, len(q_values) - 2),
                                                                                       np.random.binomial(1, 0.5 + 0.1, 1),
                                                                                       np.random.binomial(1, 0.5 - 0.1 if t < num_steps/2 else 0.5 + 4*0.1, 1)]))
    # for i, (algorithm, args, kwargs) in enumerate(algos):
    #     (average_regret[i], regret_bound[i]), precents_optimal[i] = train_and_evaluate(bandit, num_steps, num_runs, algorithm, regret=True, trial=True, *args, **kwargs)
    
    (algo, args, kwargs) = ours
    action_selector = action_selectors[0]
    for c in certainty:
        # algo: EXP3IXrl, bandit: BanditEnvironment, num_steps: int, num_runs: int, algorithm: Algorithm, certainty: int, regret: bool = True, trial=False, *args, **kwargs
        (average_regret_greedyselector, regret_bound_greedyselector), precents_optimal_greedyselector, (average_regret_ours_greedy, regret_bound_ours_greedy), precents_optimal_ours_greedy, precent_ours_greedy = train_and_evaluate_integrated(algo, bandit, num_steps, num_runs, action_selector, certainty=c, regret=True, trial=True, *args, **kwargs)
    action_selector = action_selectors[1]
    for c in certainty:
        (average_regret_ucbselector, regret_bound_ucbselector), precents_optimal_ucbselector, (average_regret_ours_ucb, regret_bound_ours_ucb), precents_optimal_ours_ucb, precent_ours_ucb = train_and_evaluate_integrated(algo, bandit, num_steps, num_runs, action_selector, certainty=c, regret=True, trial=True, *args, **kwargs)
    action_selector = action_selectors[2]
    for c in certainty:
        (average_regret_gradientselector, regret_bound_gradientselector), precents_optimal_gradientselector, (average_regret_ours_gradient, regret_bound_ours_gradient), precents_optimal_ours_gradient, precent_ours_gradient = train_and_evaluate_integrated(algo, bandit, num_steps, num_runs, action_selector, certainty=c, regret=True, trial=True, *args, **kwargs)
    
    
    folder = './results/all_trial/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    # np.save(folder + 'average_regret.npy', average_regret)
    # np.save(folder + 'percent_optimal.npy', precents_optimal)
    np.save(folder + 'average_regret_selector.npy', average_regret_greedyselector)
    np.save(folder + 'percent_optimal_selector.npy', precents_optimal_greedyselector)
    np.save(folder + 'average_regret_ucbselector.npy', average_regret_ucbselector)
    np.save(folder + 'percent_optimal_ucbselector.npy', precents_optimal_ucbselector)
    np.save(folder + 'average_regret_gradientselector.npy', average_regret_gradientselector)
    np.save(folder + 'percent_optimal_gradientselector.npy', precents_optimal_gradientselector)
    np.save(folder + 'average_regret_ours_greedy.npy', average_regret_ours_greedy)
    np.save(folder + 'percent_optimal_ours_greedy.npy', precents_optimal_ours_greedy)
    np.save(folder + 'average_regret_ours_ucb.npy', average_regret_ours_ucb)
    np.save(folder + 'percent_optimal_ours_ucb.npy', precents_optimal_ours_ucb)
    np.save(folder + 'average_regret_ours_gradient.npy', average_regret_ours_gradient)
    np.save(folder + 'percent_optimal_ours_gradient.npy', precents_optimal_ours_gradient)
    np.save(folder + 'precent_ours_greedy.npy', precent_ours_greedy)
    np.save(folder + 'precent_ours_ucb.npy', precent_ours_ucb)
    np.save(folder + 'precent_ours_gradient.npy', precent_ours_gradient)

    print('saved average regret and percent optimal')

    # average_rewards = np.load('average_rewards.npy')
    # precents_optimal = np.load('percent_optimal.npy')

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    # plt.plot(average_regret[0], label='EXP3')
    # plt.plot(average_regret[1], label='EXP3IX')
    plt.plot(average_regret_ours_greedy, label='EXP3IXrl_greedy')
    plt.plot(average_regret_ours_ucb, label='EXP3IXrl_ucb')
    plt.plot(average_regret_ours_gradient, label='EXP3IXrl_gradient')
    plt.plot(regret_bound[0], label='Regret Bound')
    plt.xlabel('steps')
    plt.ylabel('average regret')
    plt.legend()

    plt.subplot(2, 1, 2)
    # plt.plot(precents_optimal[0] * 100, label='EXP3')
    # plt.plot(precents_optimal[1] * 100, label='EXP3IX')
    plt.plot(precents_optimal_ours_greedy * 100, label='EXP3IXrl_greedy')
    plt.plot(precents_optimal_ours_ucb * 100, label='EXP3IXrl_ucb')
    plt.plot(precents_optimal_ours_gradient * 100, label='EXP3IXrl_gradient')
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

    plt.savefig(folder + 'all_trial.png')
    plt.close()

def exp3ix_dynamic():
    num_steps = 10000
    num_runs = 2000
    algos = [(EXP3, (), {'time_horizon': num_steps}), (EXP3IX, (), {'time_horizon': num_steps}),
             (EXP3IX, (), {'time_horizon': num_steps, 'dynamic_eta': True})]

    average_regret = np.zeros((len(algos), num_steps))
    precents_optimal = np.zeros((len(algos), num_steps))
    regret_bound = np.zeros((len(algos), num_steps))

    bandit = AdversarialBandit(10, reward_update_func=lambda q_values, t: np.concatenate([np.random.binomial(1, 0.5, len(q_values) - 2),
                                                                                       np.random.binomial(1, 0.5 + 0.1, 1),
                                                                                       np.random.binomial(1, 0.5 - 0.1 if t < num_steps/2 else 0.5 + 4*0.1, 1)]))
    for i, (algorithm, args, kwargs) in enumerate(algos):
        (average_regret[i], regret_bound[i]), precents_optimal[i] = train_and_evaluate(bandit, num_steps, num_runs, algorithm, regret=True, trial=True, *args, **kwargs)
    folder = './results/exp3ix_dynamic/'
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
    plt.plot(average_regret[1], label='EXP3IX')
    plt.plot(average_regret[2], label='EXP3IX_dynamic')
    plt.plot(regret_bound[0], label='Regret Bound')
    plt.xlabel('steps')
    plt.ylabel('average regret')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(precents_optimal[0] * 100, label='EXP3')
    plt.plot(precents_optimal[1] * 100, label='EXP3IX')
    plt.plot(precents_optimal[2] * 100, label='EXP3IX_dynamic')
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

    plt.savefig(folder + 'exp3ix_dynamic.png')
    plt.close()

def exp3ix_trial():
    num_steps = 10000
    num_runs = 2000
    algos = [(EXP3, (), {'time_horizon': num_steps}), (EXP3IX, (), {'time_horizon': num_steps})]

    average_regret = np.zeros((len(algos), num_steps))
    precents_optimal = np.zeros((len(algos), num_steps))
    regret_bound = np.zeros((len(algos), num_steps))

    bandit = AdversarialBandit(10, reward_update_func=lambda q_values, t: np.concatenate([np.random.binomial(1, 0.5, len(q_values) - 2),
                                                                                       np.random.binomial(1, 0.5 + 0.1, 1),
                                                                                       np.random.binomial(1, 0.5 - 0.1 if t < num_steps/2 else 0.5 + 4*0.1, 1)]))
    for i, (algorithm, args, kwargs) in enumerate(algos):
        (average_regret[i], regret_bound[i]), precents_optimal[i] = train_and_evaluate(bandit, num_steps, num_runs, algorithm, regret=True, trial=True, *args, **kwargs)
    folder = './results/exp3ix_trial/'
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
    plt.plot(average_regret[1], label='EXP3IX')
    plt.plot(regret_bound[0], label='Regret Bound')
    plt.xlabel('steps')
    plt.ylabel('average regret')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(precents_optimal[0] * 100, label='EXP3')
    plt.plot(precents_optimal[1] * 100, label='EXP3IX')
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

    plt.savefig(folder + 'exp3ix_trial.png')
    plt.close()

def run_exp3ix_adversarial():
    num_steps = 10000
    num_runs = 2000
    algos = [(EXP3, (), {'time_horizon': num_steps}), (EXP3IX, (), {'time_horizon': num_steps})]

    average_regret = np.zeros((len(algos), num_steps))
    precents_optimal = np.zeros((len(algos), num_steps))
    regret_bound = np.zeros((len(algos), num_steps))

    bandit = AdversarialBandit(10, reward_update_func=lambda q_values, t: np.concatenate([np.random.binomial(1, 0.5, len(q_values) - 2),
                                                                                       np.random.binomial(1, 0.5 + 0.1, 1),
                                                                                       np.random.binomial(1, 0.5 - 0.1 if t < num_steps/2 else 0.5 + 4*0.1, 1)]))
    for i, (algorithm, args, kwargs) in enumerate(algos):
        (average_regret[i], regret_bound[i]), precents_optimal[i] = train_and_evaluate(bandit, num_steps, num_runs, algorithm, regret=True, *args, **kwargs)
    folder = './results/EXP3IX_Adversary/'
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
    plt.plot(average_regret[1], label='EXP3IX')
    plt.plot(regret_bound[0], label='Regret Bound')
    plt.xlabel('steps')
    plt.ylabel('average regret')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(precents_optimal[0] * 100, label='EXP3')
    plt.plot(precents_optimal[1] * 100, label='EXP3IX')
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

    plt.savefig(folder + 'EXP3IX_Adversary.png')
    plt.close()

def run_exp3():
    num_steps = 10000
    num_runs = 2000
    algos = [(EXP3, (), {'time_horizon': num_steps}), (EXP3IX, (), {'time_horizon': num_steps})]

    average_regret = np.zeros((len(algos), num_steps))
    precents_optimal = np.zeros((len(algos), num_steps))
    regret_bound = np.zeros((len(algos), num_steps))

    # bandit = AdversarialBandit(10, reward_update_func=lambda q_values, t: np.random.rand(len(q_values)))
    bandit = AdversarialBandit(10)
    for i, (algorithm, args, kwargs) in enumerate(algos):
        (average_regret[i], regret_bound[i]), precents_optimal[i] = train_and_evaluate(bandit, num_steps, num_runs, algorithm, regret=True, *args, **kwargs)
    folder = './results/EXP3IX/'
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
    plt.plot(average_regret[1], label='EXP3IX')
    plt.plot(regret_bound[0], label='Regret Bound')
    plt.xlabel('steps')
    plt.ylabel('average regret')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(precents_optimal[0] * 100, label='EXP3')
    plt.plot(precents_optimal[1] * 100, label='EXP3IX')
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

    plt.savefig(folder + 'EXP3IX.png')
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