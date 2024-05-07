import pickle

class loading():
    def __init__(self):
        self.cces, self.Q_tables = None, None
        self.distances, self.cce_precents = None, None
        self.zetas = None

    def load(self, path):
        with open('SARSA_dists.pkl', 'rb') as f:
            self.distances = pickle.load(f)

        with open('SARSA_paths.pkl', 'rb') as f:
            self.paths = pickle.load(f)

        with open('SARSA_q_tables.pkl', 'rb') as f:
            self.Q_tables = pickle.load(f)

        with open('SARSA_cce.pkl', 'rb') as f:
            self.cces = pickle.load(f)

        with open('SARSA_cce_percentages.pkl', 'rb') as f:
            self.cce_precents = pickle.load(f)

        with open('zetavalues.pkl', 'rb') as f:
            self.zetas = pickle.load(f)

