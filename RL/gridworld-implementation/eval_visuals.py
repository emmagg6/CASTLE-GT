import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import pickle

from visuals import visualize_q_table_differences
# ------------------------------------------------
# Load the data:
# ------------------------------------------------

with open('base_q_tables.pkl', 'rb') as f:
    base_q_tables = pickle.load(f)
with open('q_tables.pkl', 'rb') as f:
    q_tables = pickle.load(f)
with open('trn_dists_1.pkl', 'rb') as f:
    trn_dists_1 = pickle.load(f)
with open('trn_dists_2.pkl', 'rb') as f:
    trn_dists_2 = pickle.load(f)
with open('trn_dists_3.pkl', 'rb') as f:
    trn_dists_3 = pickle.load(f)
with open('trn_dists_4.pkl', 'rb') as f:
    trn_dists_4 = pickle.load(f)
with open('trn_dists_full.pkl', 'rb') as f:
    trn_dists_full = pickle.load(f)
with open('own_dists.pkl', 'rb') as f:
    own_dists = pickle.load(f)
with open('goal_states.pkl', 'rb') as f:
    goal_states = pickle.load(f)
with open('base_distance.pkl', 'rb') as f:
    base_distance = pickle.load(f)



# ------------------------------------------------
# Visualize all the learning dynamics and Q-table differences
# ------------------------------------------------
# Q-table for each task (goal_state) then has shape (num_tasks, num_states, num_actions) and there are 100 trials for each task
# show the difference in the q_value: average over the 100 trials for each task : from that of the average base_q_table (over 100 trials)

# Q-table comparison
# visualize_q_table_differences(base_q_tables, q_tables, goal_states)# DOES NOT WORK




# Plot all the learning dynamics on original task as training on similar
plt.figure()
cmap = mpl.colormaps['viridis']
for i, tasks in enumerate(goal_states):
    ave_base_distance = np.mean(base_distance)
    ave_trn_dists_1 = np.mean(trn_dists_1[i])
    ave_trn_dists_2 = np.mean(trn_dists_2[i])
    ave_trn_dists_3 = np.mean(trn_dists_3[i])
    ave_trn_dists_4 = np.mean(trn_dists_4[i])
    ave_trn_dists_full = np.mean(trn_dists_full[i])
    trn_dists_dynamic = [ave_base_distance, ave_trn_dists_1, ave_trn_dists_2, ave_trn_dists_3, ave_trn_dists_4, ave_trn_dists_full]
    plt.plot(trn_dists_dynamic, label=f'Task {i}', color = cmap(i / len(goal_states)))
plt.xlabel('Mid-training Evaluation Interval')
plt.ylabel('Distance')
plt.title('Performance Dynamics of Original Task while Training on Similar Tasks')
plt.legend()
plt.savefig('learning_dynamics.png')
plt.close()
plt.legend()
plt.figure()

# Plot all the evaluations on their task  after training completed
plt.figure()
ave_own_dists = np.mean(own_dists, axis=0)  
plt.plot(ave_own_dists)
plt.xlabel('Task Difference Level')
plt.ylabel('Performance on Similar Task')
plt.title('Performance on Similar Task after Training on Similar Tasks')
plt.savefig('evaluations_on_own.png')
plt.close()