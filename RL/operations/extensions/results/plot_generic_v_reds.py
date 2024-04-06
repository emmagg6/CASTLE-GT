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

fig, axes = plt.subplots(1,2, figsize=(10,6))
fig.tight_layout(pad=5.0)
markers = ['o', '^', '*', '+','.','p']


for i, red_agent in enumerate(reversed(df['red_agent'].unique())):
    if 'Evade' in red_agent:
        ax = axes[0]
        ax.set_title("Generic Blue Agent Reward Against Evasive Red Agents")
    else:
        ax = axes[1]
        ax.set_title("Generic Blue Agent Reward Against Non-Evasive Red Agents")
    marker = markers[i]
    agent_df = df.loc[df['blue_agent'] == 'generic'].loc[df['red_agent'] == red_agent]
    ax.errorbar(agent_df['steps'], agent_df['avg_reward'], yerr=agent_df['std'], label=red_agent, marker=marker, capsize=10)
    ax.legend()
    

fig.subplots_adjust(top=0.88)
fig.savefig('plot_generic_v_reds_fig.pdf')

