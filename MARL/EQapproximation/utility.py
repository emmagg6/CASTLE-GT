import matplotlib.pyplot as plt
import numpy as np

def plot_graph(cumulative_losses,name):
    """
        plot the cumulative loss over time steps
    """
    time_steps = np.arange(1, len(cumulative_losses) + 1)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, cumulative_losses, label='Cumulative Loss')
    # plt.plot(time_steps, np.sqrt(time_steps), label='√n', linestyle='--')  # Reference sublinear function

    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Loss')
    plt.title(name)
    plt.legend()
    plt.grid(True)

    plt.show()

def plot_scaled_sum_policy_over_time(policy_sums_over_time, checkpoints,blueAgentActions):
    """
    Plot the scaled sum of policy over time for each state.

    Args:
    policy_sums_over_time (dict): A dictionary where each key is a state and each value is a list of scaled policy sums over time.
    checkpoints (list): List of iteration numbers at which policy sums were recorded.
    """
    plt.figure(figsize=(12, 8))
    for state, sums in policy_sums_over_time.items():
        plt.plot(checkpoints, sums, label=f'State {state}+')
    
    plt.title('Scaled Sum of Policy Over Time for Each State')
    plt.xlabel('Iteration')
    plt.ylabel('Scaled Sum of Policy')
    plt.legend()
    plt.grid(True)
    plt.show()