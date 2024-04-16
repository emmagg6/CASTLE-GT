import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# take data

data = pd.read_csv('results_half.csv')
cols = ['blue_agent','red_agent','steps','avg_reward','std','label']
df = pd.DataFrame(data, columns=cols)
print(df)

sns.set(font_scale=1.5)
sns.set_style("ticks")
sns.color_palette("Blues", as_cmap=True)

'''
g = sns.catplot(
    data=df, kind="bar",
    x="red_agent", y="avg_reward", hue="label",
    errorbar="sd", palette="dark", alpha=.6, height=6
)
'''
g = sns.catplot(
    data=df, kind="bar",
    x="red_agent", y="avg_reward", hue="label", palette="Blues"
)
g.set_axis_labels("", "Blue Reward")
g.legend.set_title("")
#plt.tight_layout()

#plt.title('coBlue-BASE Evaluation')
#plt.gca().invert_yaxis()

path_plots = 'cotraining_plots'
os.makedirs(path_plots, exist_ok=True)
filename_plot = "half_training.png"
plotfile = os.path.join(path_plots, filename_plot)
plt.savefig(plotfile)
plt.clf()
print("Plot saved in: ", plotfile)

