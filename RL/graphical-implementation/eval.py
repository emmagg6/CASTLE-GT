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



#################### Plotting ####################


'''
# Plot the precentage of cce actions per zeta value
'''
plt.scatter(zetas_lst, prop, color='black', alpha=0.1)
plt.xlabel('Minimum observation-action visits (certainty measure) of EXP3-IXRL approximation')
plt.ylabel('Percentage of EXP3-IXRL Actions')
plt.xlim(25000, 350000)
plt.savefig('final_cce_percentage.png')



'''
# Plot the distances of the paths by the zeta values
'''

TRIALS = 100

data = pd.DataFrame({
    'Zeta': zetas_lst,
    'Distance': dists,
    # 'CCE Percentage': prop,
})

data_exploded = data.apply(pd.Series.explode)
mean_dists = data_exploded.groupby('Zeta').mean()
std_devs = data_exploded.groupby('Zeta').std()
# prop = data_exploded.groupby('Zeta').mean()

# # # ave distance of the last zeta value
# print("Average distance of the last zeta value:")
# print(mean_dists.iloc[-1])
# # # std dev of dists for the last zeta value
# print("Standard deviation of the last zeta value:")
# print(std_devs.iloc[-1])
# # # proportion of cce actions for the last zeta value
# print("Proportion of CCE actions for the last zeta value:")
# print(prop.iloc[-1])

# #  # ave distance of the first zeta value
# print("Average distance of the first zeta value:")
# print(mean_dists.iloc[0])
# # # std dev of dists for the first zeta value
# print("Standard deviation of the first zeta value:")
# print(std_devs.iloc[0])
# # # proportion of cce actions for the first zeta value
# print("Proportion of CCE actions for the first zeta value:")
# print(prop.iloc[0])



plt.figure(figsize=(10, 6))
plt.errorbar(mean_dists.index, mean_dists['Distance'], yerr=std_devs['Distance'], 
             fmt='o:', capsize=1, color='black', label='EXP3-IXRL',
             elinewidth=1, capthick=1)

# add green horizonatal line at 31229.347695 with std 160.103771 with label 'RL Agent'
plt.axhline(y=31229.347695, color='blue', linestyle='--', label='RL Agent')
# plt.axhline(y=31229.347695 + 160.103771, color='green', linestyle=':', label='RL Agent + STD', alpha=0.15)
# plt.axhline(y=31229.347695 - 160.103771, color='green', linestyle=':', label='RL Agent - STD', alpha=0.15)

# plt.title('Average Distance vs Certainty')
plt.xlabel('Minimum observation-action visits (certainty measure) of EXP3-IXRL approximation')
plt.ylabel('Distance to Goal State')
# plt.xlim(12000, 305000)
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')

plt.savefig('final_distance_vs_cce.png')



# evaluation plot by percentage of cce actions instead of zeta values

data = pd.DataFrame({
    'CCE Percentage': prop,
    'Distance': dists
})

data_exploded = data.apply(pd.Series.explode)
mean_dists = data_exploded.groupby('CCE Percentage').mean()
std_devs = data_exploded.groupby('CCE Percentage').std()

plt.figure(figsize=(10, 6))
plt.errorbar(mean_dists.index, mean_dists['Distance'], yerr=std_devs['Distance'], fmt='o:', capsize=0, color='black', label='RLxCCE Agent',elinewidth=0)

# add green horizonatal line at 31229.347695 with std 160.103771 with label 'RL Agent'
plt.axhline(y=31229.347695, color='green', linestyle='--', label='RL Agent')
# plt.axhline(y=31229.347695 + 160.103771, color='green', linestyle=':', label='RL Agent + STD', alpha=0.15)
# plt.axhline(y=31229.347695 - 160.103771, color='green', linestyle=':', label='RL Agent - STD', alpha=0.15)

plt.title('Average Distance vs Proportion of EXP3-IXRL Action-Selection ')
plt.xlabel('Percentage of EXP3-IXRL Actions')
plt.ylabel('Distance to Goal State')
plt.grid(True)

plt.legend(loc='upper right', fontsize='small')

# plt.savefig('final_distance_vs_cce_percentage.png')




'''
Does a trial have a higher distance average than all the others?
'''

# TRIALS = 100
# # # average distance of each trial
# for i in range(TRIALS):
#     print(f'Trial {i}: {np.mean(dists[i])}')

# # # average distance of all trials
# print(f'Average distance of all trials: {np.mean(dists)}')

# # highest distance average
# print(f'Highest distance average: {max([np.mean(dists[i]) for i in range(TRIALS)])}')
