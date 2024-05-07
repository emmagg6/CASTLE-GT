from runs.loading import loading
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

load = loading()
load.load()

dists = load.distances
paths = load.paths
Qs = load.Q_tables
cces = load.cces
precentages = load.cce_precents
zetas = load.zetas

'''
Plot the precentage of cce actions per zeta value
'''

plt.scatter(zetas, precentages, color='black', alpha=0.1)
plt.xlabel('Zeta Value')
plt.ylabel('Percentage of CCE Actions')

plt.savefig('cce_precentage.png')