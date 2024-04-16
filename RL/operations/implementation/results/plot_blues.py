import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

all_files = glob.glob("*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True)

fig, axes = plt.subplots(2,3, figsize=(12,8))
fig.tight_layout(pad=2.0)
markers = ['o', '^', '*','+','.']

for red_agent, ax in zip(reversed(df['red_agent'].unique()), axes.ravel()):
    for i, blue_agent in enumerate(df['blue_agent'].unique()):
        marker = markers[i]
        agent_df = df.loc[df['blue_agent'] == blue_agent].loc[df['red_agent'] == red_agent]
        ax.errorbar(agent_df['steps'], agent_df['avg_reward'], yerr=agent_df['std'], label=blue_agent, marker=marker, capsize=10)
    if red_agent == 'SleepAgent':
        ax.legend()
    ax.set_title(red_agent)

# axes[1,2].remove()
# fig.set_title('Comparing Blue Agent Reward Against Each Red Agent')
fig.savefig('fig.pdf')


