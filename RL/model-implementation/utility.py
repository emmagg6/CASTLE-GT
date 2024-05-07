import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_graph(losses_list, names_list):
    """
    Plot the cumulative loss over time steps for multiple datasets.
    """
    plt.figure(figsize=(10, 6))


    for losses, name in zip(losses_list, names_list):
        time_steps = np.arange(1, len(losses) + 1)
        cumulative_losses = np.cumsum(losses)
        plt.plot(time_steps, cumulative_losses, label=name)

    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Loss')
    plt.title('Cumulative Loss Over Time Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative_loss.png')
    plt.show()


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
    plt.show()   



def find_most_favored_action(sumOfPolicy):
    max_value = -float('inf')  
    favored_state = None
    favored_action = None

    # Iterate through each state's policy sums
    for state, policy_sums in sumOfPolicy.items():
        # Find the action with the highest cumulative policy sum in this state
        max_action_index = np.argmax(policy_sums)  # Assuming policy_sums is a numpy array
        max_action_value = policy_sums[max_action_index]
        
        # Check if this action's max is the highest found so far across all states
        if max_action_value > max_value:
            max_value = max_action_value
            favored_state = state
            favored_action = max_action_index

    return favored_state, favored_action, max_value
