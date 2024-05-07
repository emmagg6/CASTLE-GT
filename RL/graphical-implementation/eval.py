from runs.loading import loading
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

folder = '/scratch/egraham/CASTLE-GT/RL/graphical-implementation/runs'
load = loading()
load.load_all(path = folder)

dists = load.distances
paths = load.paths
Qs = load.Q_tables
cces = load.cces
prop = load.cce_precents
zetas = load.zetas
zetas_lst = load.list_zetas
graph = load.graph

'''
Plot the precentage of cce actions per zeta value
'''
# print(f"Zetas: {zetas}")
# print(f"Percentage: {prop}")
# plt.scatter(zetas_lst, prop, color='black', alpha=0.1)
# plt.xlabel('Zeta Value')
# plt.ylabel('Percentage of CCE Actions')

# plt.savefig('cce_percentage.png')



'''
Plot the distances of the paths by the zeta values
'''

data = pd.DataFrame({
    'Zeta': zetas_lst,
    'Distance': dists
})

data_exploded = data.apply(pd.Series.explode)
mean_dists = data_exploded.groupby('Zeta').mean()
std_devs = data_exploded.groupby('Zeta').std()

plt.figure(figsize=(10, 6))
plt.errorbar(mean_dists.index, mean_dists['Distance'], yerr=std_devs['Distance'], fmt='o:', capsize=5, color='black', label='RLxCCE Agent')

plt.title('Average Distance vs CCE Certainty')
plt.xlabel('Minimum observation-action visits (certainty) of CCE')
plt.ylabel('Distance to Goal State')
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')

plt.savefig('distance_vs_cce.png')

