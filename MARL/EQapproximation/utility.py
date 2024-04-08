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