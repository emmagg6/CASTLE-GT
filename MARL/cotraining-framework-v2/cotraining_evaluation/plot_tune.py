import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# take data

fig, ax = plt.subplots()
data = pd.read_csv('results_tuned.csv')
cols = ['blue_agent','red_agent','steps','avg_reward','std','label']
df = pd.DataFrame(data, columns=cols)
print(df)

sns.set(font_scale=1.5)
sns.set_style("ticks")

g = sns.catplot(
    data=df,kind="bar",
    x="red_agent", y="avg_reward", hue="blue_agent",
    palette="bright"
)

# extract the matplotlib axes_subplot objects from the FacetGrid
ax = g.facet_axis(0, 0)  # or ax = g.axes.flat[0]

# iterate through the axes containers
for c in ax.containers:
    #labels = [f'{(v.get_height() / 1000):.1f}K' for v in c]
    labels = [(int)(v.get_height()) for v in c]
    ax.bar_label(c, labels=labels, label_type='edge', padding=3)

g.set_axis_labels("", "Blue Reward")
g.legend.set_title("")

'''
ax2 = plt.axes([0.2, 0.6, .2, .2])
g2 = sns.barplot(
    data=df,
    x="red_agent", y="avg_reward", hue="blue_agent",
    palette="bright", ax=ax2, legend=None
)
ax2.set_title('zoom')
ax2.set_ylim([-130,0])
'''

plt.tight_layout()
plt.ylim([-840, 0])

path_plots = 'cotraining_plots'
os.makedirs(path_plots, exist_ok=True)
filename_plot = "tuning.png"
plotfile = os.path.join(path_plots, filename_plot)
plt.savefig(plotfile)
plt.clf()
print("Plot saved in: ", plotfile)



