import matplotlib.pyplot as plt
import numpy as np

def plot_graph(losses_list, names_list, graph_title):
    """
    Plot the cumulative loss over time steps for multiple datasets.
    """
    plt.figure(figsize=(10, 6))


    for losses, name in zip(losses_list, names_list):
        time_steps = np.arange(1, len(losses) + 1)
        cumulative_losses = np.cumsum(losses)
        plt.plot(time_steps, cumulative_losses, label=name)

    plt.xlabel('Time Steps')
    plt.ylabel(f'Cumulative {graph_title}')
    plt.title(f'Cumulative {graph_title} Over Time Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'cumulative_{graph_title}.png')

def plot_regret(names_list,regretper_stateIterationEQObserver,regretper_stateEQObserver,regretEQAgent,regretper_stateEQAgent,regretper_stateIterationEQAgent):
    """
    total + y=x
    Plots cumulative regret per state for multiple agents.

    :param names_list: List of names for each agent.
    :param regret_per_state_list: List of dictionaries with cumulative regrets per state for each agent.
    :param regret_per_state_iteration_list: List of dictionaries with iteration counts for regrets per state for each agent.
    """
    plt.figure(figsize=(15, 10))  
    plt.rcParams.update({'font.size': 20})
    time_steps = np.arange(1, len(regretEQAgent) + 1)
    cumulative_losses = np.cumsum(regretEQAgent)
    # plt.plot(time_steps, cumulative_losses, label="EXP3-IX", color="red",linestyle="solid")

    # print(regretper_stateIterationEQObserver)
    islabeled =False
    for state, regrets in regretper_stateEQObserver.items():
        iterations = regretper_stateIterationEQObserver[state]
        if islabeled == False:
            plt.plot(iterations, regrets,label=f'Agent-Agnostic EXP3-IX (per state)', color="black",linestyle="solid")
            islabeled = True
        else:
            plt.plot(iterations, regrets, color="black",linestyle="solid")

    y_equals_x = np.arange(1, 276)
    plt.plot(y_equals_x, y_equals_x, label='Linear Regret Line', color='blue', linestyle='dashed')

    plt.title('Cumulative Regret per State Over Time')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    plt.ylim(0,275)
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('cumulative_regret_comparison.png')  
