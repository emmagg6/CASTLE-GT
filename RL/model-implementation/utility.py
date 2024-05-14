import matplotlib.pyplot as plt
import numpy as np
import itertools

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


def plot_scaled_sum_policy_over_time(policy_sums_over_time, checkpoints, blueAgentActions):
    """
    Plot the scaled sum of policy over time for each state.

    Args:
    policy_sums_over_time (dict): A dictionary where each key is a state and each value is a list of scaled policy sums over time.
    checkpoints (list): List of iteration numbers at which policy sums were recorded.
    """
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10.colors  # Get a colormap from matplotlib (10 colors)
    line_styles = ['-', '--', ':', '-.']  # Define line styles
    state_colors = dict(zip(policy_sums_over_time.keys(), itertools.cycle(colors)))
    
    for state, sums_list in policy_sums_over_time.items():
        # Unpack each action's sum values from each array and plot them
        color = state_colors[state]  # Get consistent color for the state
        for action_index in range(len(sums_list[0])):  # Assuming all arrays have the same size
            # Extract the sum for this action across all checkpoints
            action_sums = [sums[action_index] for sums in sums_list]
            line_style = line_styles[action_index % len(line_styles)]  # Cycle through line styles
            plt.plot(checkpoints, action_sums, label=f'State {state} Action {blueAgentActions[action_index]}', 
                     color=color, linestyle=line_style)
    
    plt.title('Scaled Sum of Policy Over Time for Each State')
    plt.xlabel('Iteration')
    plt.ylabel('Scaled Sum of Policy')
    plt.legend()
    plt.save('scaled_sum_policy_over_time.png')


def plot_regret(names_list,regretper_stateIterationEQObserver,regretper_stateEQObserver,regretEQAgent,regretper_stateEQAgent,regretper_stateIterationEQAgent):
    """
    total + y=x
    Plots cumulative regret per state for multiple agents.

    :param names_list: List of names for each agent.
    :param regret_per_state_list: List of dictionaries with cumulative regrets per state for each agent.
    :param regret_per_state_iteration_list: List of dictionaries with iteration counts for regrets per state for each agent.
    """
    plt.figure(figsize=(15, 10))  

    time_steps = np.arange(1, len(regretEQAgent) + 1)
    cumulative_losses = np.cumsum(regretEQAgent)
    plt.plot(time_steps, cumulative_losses, label="EXP3-IX", color="red",linestyle="solid")

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
