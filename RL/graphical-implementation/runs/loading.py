import pickle
import os

class loading():
    def __init__(self):
        self.cces, self.Q_tables = None, None
        self.distances, self.cce_precents = None, None
        self.zetas, self.list_zetas, self.graph = None, None, None

    def load_train(self, path):
        with open(os.path.join(path, 'SARSA_q_tables.pkl'), 'rb') as f:
            self.Q_tables = pickle.load(f)

        with open(os.path.join(path, 'SARSA_cce.pkl'), 'rb') as f:
            self.cces = pickle.load(f)

        with open(os.path.join(path, 'graphs.pkl'), 'rb') as f:
            self.graphs = pickle.load(f)

    def load_all(self, path):
        with open(os.path.join(path, 'SARSA_dists.pkl'), 'rb') as f:
            self.distances = pickle.load(f)

        with open(os.path.join(path, 'SARSA_paths.pkl'), 'rb') as f:
            self.paths = pickle.load(f)

        with open(os.path.join(path, 'SARSA_q_tables.pkl'), 'rb') as f:
            self.Q_tables = pickle.load(f)

        with open(os.path.join(path, 'SARSA_cce.pkl'), 'rb') as f:
            self.cces = pickle.load(f)

        with open(os.path.join(path, 'SARSA_cce_percentages.pkl'), 'rb') as f:
            self.cce_precents = pickle.load(f)

        with open(os.path.join(path, 'zetavalues.pkl'), 'rb') as f:
            self.zetas = pickle.load(f)

        with open(os.path.join(path, 'all_zetas.pkl'), 'rb') as f:
            self.list_zetas = pickle.load(f)

        with open(os.path.join(path, 'graphs.pkl'), 'rb') as f:
            self.graphs = pickle.load(f)

