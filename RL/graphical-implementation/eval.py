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
############## TRAINED VARIABLES ##############
# print(f"Q tables for trial 0: {Qs[0]}")   
for state_action in [list(Qs[0].keys())][0]:
    state, action = state_action
    print(f"State: {state}, Action: {action}, Q value: {Qs[0][state_action]}")


for state in cces[0]:
    for action in cces[0][state]:
        cce_eq = cces[0][state][action][0]
        cce_count = cces[0][state][action][1]
        print('State:', state, 'Action:', action, 'CCE:', cce_eq, 'Count:', cce_count)

# # what is the largest cce value? smallest? average?
cce_values = []
for cce in cces:
    for state in cce:
        for action in cce[state]:
            cce_values.append(cce[state][action][0])

print('Max CCE:', max(cce_values))
print('Min CCE:', min(cce_values))
print('Average CCE:', np.mean(cce_values))

# # what is the largest cce count? smallest? average?
cce_counts = []
for cce in cces:
    for state in cce:
        for action in cce[state]:
            cce_counts.append(cce[state][action][1])

print('Max CCE Count:', max(cce_counts))
print('Min CCE Count:', min(cce_counts))
print('Average CCE Count:', np.mean(cce_counts))
'''


#################### Plotting ####################


'''
Plot the precentage of cce actions per zeta value
'''
plt.scatter(zetas_lst, prop, color='black', alpha=0.1)
plt.xlabel('Zeta Value')
plt.ylabel('Percentage of CCE Actions')

plt.savefig('ppcce_percentage.png')



'''
Plot the distances of the paths by the zeta values
'''

TRIALS = 100

data = pd.DataFrame({
    'Zeta': zetas_lst,
    'Distance': dists
})

data_exploded = data.apply(pd.Series.explode)
mean_dists = data_exploded.groupby('Zeta').mean()
std_devs = data_exploded.groupby('Zeta').std()


plt.figure(figsize=(10, 6))
plt.errorbar(mean_dists.index, mean_dists['Distance'], yerr=std_devs['Distance'], fmt='o:', capsize=5, color='black', label='RLxCCE Agent')

# add green horizonatal line at 34163.47381063614 with std 2305.177588945328 with label 'RL Agent'
plt.axhline(y=34163.47381063614, color='green', linestyle='--', label='RL Agent')

plt.title('Average Distance vs CCE Certainty')
plt.xlabel('Minimum observation-action visits (certainty) of CCE')
plt.ylabel('Distance to Goal State')
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')

plt.savefig('mppdistance_vs_cce.png')

