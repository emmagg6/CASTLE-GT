import numpy as np
from typing import Callable

class Algorithm:
    def select_action(self) -> int:
        '''
        Select an action using the algorithm.

        Returns
        -------
        action : int
            The selected action.
        '''
        raise NotImplementedError
    
    def train(self, action: int, reward: float):
        '''
        Train the algorithm.

        Parameters
        ----------
        action : int
            The action to take.
        reward : float
            The reward received from the action.
        '''
        raise NotImplementedError
    
    def reset(self) -> None:
        '''
        Reset the algorithm.
        '''
        raise NotImplementedError

class EpsilonGreedy(Algorithm):
    '''
    Implementation of the EpsilonGreedy exploration-exploitation algorithm from Sutton and Barto's book.
    '''
    def __init__(self, n: int, epsilon: float, alpha: float = None, q_estimates_func: Callable = lambda n: np.zeros(n)) -> None:
        '''
        Initialize the EpsilonGreedy algorithm.

        Parameters
        ----------
        n : int
            Number of actions.
        epsilon : float
            The probability of selecting a random action.
        alpha : float, optional
            The step size parameter. If None, the step size is 1/k, where k is count of picked action. Defaults to None.
        q_estimates_func : function, optional
            A function that takes n as input and returns a numpy array of 
            length n containing the initial Q-value estimates. Defaults to np.zeros(n).
        '''
        # super().__init__(n, q_dist_func)
        self.n = n
        self.epsilon = epsilon
        self.q_estimates_func = q_estimates_func
        self.q_estimates = q_estimates_func(n) # Q_t(a) in the book
        self.action_counts = np.zeros(n) # N_t(a) in the book
        self.alpha = alpha

    def select_action(self) -> int:
        '''
        Select an action using the EpsilonGreedy algorithm.

        Returns
        -------
        action : int
            The selected action.
        '''
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n)
        else:
            # Greedly select the action with the highest Q-value estimate, randomly breaking ties
            best_actions = np.where(self.q_estimates == np.max(self.q_estimates))[0]
            return np.random.choice(best_actions)

    def train(self, action: int, reward: float):
        '''
        Train the EpsilonGreedy algorithm.

        Parameters
        ----------
        action : int
            The action to take.
        reward : float
            The reward received from the action.
        '''
        if (self.alpha is None):
            self.action_counts[action] += 1  # Em - good job this is a traditional bias for exploration
            alpha = 1 / self.action_counts[action]
        else:
            alpha = self.alpha
        self.q_estimates[action] += (reward - self.q_estimates[action]) * alpha # nice this is the action-value for this bandit setting

    def reset(self) -> None:
        '''
        Reset the Q-value estimates and action counts.
        '''
        self.q_estimates = self.q_estimates_func(self.n) 
        self.action_counts = np.zeros(self.n)

    def __str__(self) -> str:
        return f'EpsilonGreedy(epsilon={self.epsilon})'


class UCB(Algorithm):  # Em - this is great and traditionally has shown improvements over the static epsilon greedy strategy. notice the c value influces the exploration-exploitation tradeoff
    '''
    Implementation of the Upper Confidence Bound algorithm from Sutton and Barto's book.
    '''
    def __init__(self, n: int, c: float, q_estimates_func: Callable = lambda n: np.zeros(n)) -> None:
        '''
        Initialize the UCB algorithm.

        Parameters
        ----------
        n : int
            Number of actions.
        c : float
            The exploration parameter.
        q_estimates_func : function, optional
            A function that takes n as input and returns a numpy array of 
            length n containing the initial Q-value estimates. Defaults to np.zeros(n).
        '''
        self.n = n
        self.c = c
        self.q_estimates_func = q_estimates_func
        self.q_estimates = q_estimates_func(n) # Q_t(a) in the book
        self.action_counts = np.zeros(n) # N_t(a) in the book
        self.t = 0

    def select_action(self) -> int:
        '''
        Select an action using the UCB algorithm.

        Returns
        -------
        action : int
            The selected action.
        '''
        self.t += 1

        # If N_t(a) = 0, then a is considered to be a maximizing action.
        if 0 in self.action_counts:
            return np.random.choice(np.where(self.action_counts == 0)[0])
        
        ucb_values = self.q_estimates + self.c * np.sqrt(np.log(self.t) / self.action_counts)
        return np.argmax(ucb_values)

    def train(self, action: int, reward: float):
        '''
        Train the UCB algorithm.

        Parameters
        ----------
        action : int
            The action to take.
        reward : float
            The reward received from the action.
        '''
        self.action_counts[action] += 1
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_counts[action]

    def reset(self) -> None:
        '''
        Reset the Q-value estimates and action counts.
        '''
        self.q_estimates = self.q_estimates_func(self.n)
        self.action_counts = np.zeros(self.n)
        self.t = 0

    def __str__(self) -> str:
        return f'UCB(c={self.c})'


class GradientBandit(Algorithm): # Em - this is a great algorithm (also excels at non-stationary bandits) and is essentially the REINFORCE algorithm for bandits
    '''
    Implementation of the GradientBandit algorithm from Sutton and Barto's book.
    '''
    def __init__(self, n: int, alpha: float = None, baseline: bool = True, preference_func: Callable = lambda n: np.zeros(n)) -> None:
        '''
        Initialize the GradientBandit algorithm.

        Parameters
        ----------
        n : int
            Number of actions.
        alpha : float, optional
            The step size parameter. If None, the step size is 1/k, where k is count of picked action. Defaults to None.
        baseline : bool, optional
            Whether to use a baseline. Defaults to True. If True, the average reward is used as the baseline.
        preference_func : function, optional
            A function that takes n as input and returns a numpy array of 
            length n containing the initial preference values. Defaults to np.zeros(n).
        '''
        self.n = n
        self.preference_func = preference_func
        self.preferences = preference_func(n) # H_t(a) in the book
        self.action_probs = self.softmax(self.preferences) # π_t(a) in the book
        self.action_counts = np.zeros(n) # N_t(a) in the book
        self.t = 0
        self.average_reward = 0
        self.alpha = alpha
        self.baseline = baseline

    def select_action(self) -> int:
        '''
        Select an action using the GradientBandit algorithm.

        Returns
        -------
        action : int
            The selected action.
        '''
        return np.random.choice(self.n, p=self.action_probs)

    def train(self, action: int, reward: float):
        '''
        Train the GradientBandit algorithm.

        Parameters
        ----------
        action : int
            The action to take.
        reward : float
            The reward received from the action.
        '''
        self.t += 1
        if (self.alpha is None):
            self.action_counts[action] += 1
            alpha = 1 / self.action_counts[action]
        else:
            alpha = self.alpha

        if self.baseline:
            self.average_reward += (reward - self.average_reward) / self.t # Update the average reward incrementally
        else:
            self.average_reward = 0

        one_hot_action = np.eye(self.n)[action] # I_(a=A_t) in the book
        self.preferences += alpha * (reward - self.average_reward) * \
            (one_hot_action - self.action_probs) # H_t += α(R_t−R_avg_t)(I_(a=A_t) − π_t(a))
        
        self.action_probs = self.softmax(self.preferences)

    def reset(self) -> None:
        '''
        Reset the preference values and action counts.
        '''
        self.preferences = self.preference_func(self.n)
        self.action_probs = self.softmax(self.preferences)
        self.action_counts = np.zeros(self.n)
        self.t = 0
        self.average_reward = 0

    def __str__(self) -> str:
        return f'GradientBandit(alpha={self.alpha if self.alpha is not None else "1/k"}, baseline={self.baseline})'

    def softmax(self, x):
        '''
        Compute the softmax of an array.

        Parameters
        ----------
        x : array-like
            Input array.

        Returns
        -------
        softmax : array-like
            Softmax of the input array.
        '''
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

class EXP3(Algorithm):
    '''
    Implementation of the EXP3 algorithm from https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf for weak regret.
    Assumes rewards are in [0, 1]. Upper bound on G_max is T.
    '''
    def __init__(self, n: int, time_horizon: int, gamma: float = None):
        '''
        Initialize the EXP3 algorithm.

        Parameters
        ----------
        n : int
            Number of actions.
        time_horizon : int
            The time horizon.
        gamma : float
            The learning rate.
        '''
        self.n = n
        self.time_horizon = time_horizon # For g = T in gamma upper bound on G_max in the paper
        self.weights = np.ones(n) # w_i(t) in the paper
        self.action_probs = np.ones(n) # p_i(t) in the paper

        # Set gamma to the upper bound if not provided
        self.gamma = gamma if gamma is not None else \
            min(1, np.sqrt(n * np.log(n) / ((np.exp(1) - 1) * time_horizon)))

        self.t = 0

    def select_action(self) -> int:
        '''
        Select an action using the EXP3 algorithm.

        Returns
        -------
        action : int
            The selected action.
        '''
        self.action_probs = (1 - self.gamma) * self.weights / np.sum(self.weights) + self.gamma / self.n
        return np.random.choice(self.n, p=self.action_probs)

    def train(self, action: int, reward: float):
        '''
        Train the EXP3 algorithm.

        Parameters
        ----------
        action : int
            The action to take.
        reward : float
            The reward received from the action.
        '''
        estimated_reward = reward / self.action_probs[action]

        estimated_rewards_vector = np.eye(self.n)[action] * estimated_reward # x_hat_j(t) in the paper

        self.weights = self.weights * np.exp(self.gamma * estimated_rewards_vector / self.n)

        self.t += 1

    def reset(self) -> None:
        '''
        Reset the weights.
        '''
        self.weights = np.ones(self.n)
        self.action_probs = np.ones(self.n)
        self.t = 0

    def __str__(self) -> str:
        return f'EXP3(gamma={self.gamma})'

class EXP3IX(Algorithm):
    '''
    Implementation of the EXP3-IX algorithm
    '''
    def __init__(self, n: int, time_horizon: int, gamma: float = None):
        '''
        Initialize the EXP3-IX algorithm.

        Parameters
        ----------
        n : int
            Number of actions.
        time_horizon : int
            The time horizon.
        gamma : float
            Implicit eXploration parameter
        eta : float
            Learning rate term

        '''
        self.n = n
        self.time_horizon = time_horizon # For g = T in gamma upper bound on G_max in the paper
        self.weights = np.ones(n) # w_i(t) in the paper
        self.action_probs = np.ones(n) # p_i(t) in the paper
        self.visit_count = np.zeros(n) # N_i(t) in the paper

        self.gamma = np.sqrt((2 * np.log(n))/(n + self.time_horizon)) 
        self.eta = self.gamma * 2

        self.t = 0

    def select_action(self) -> int:
        '''
        Select an action using the EXP3 algorithm.

        Returns
        -------
        action : int
            The selected action.
        '''
        self.action_probs = self.weights / np.sum(self.weights)
        action  = np.random.choice(self.n, p=self.action_probs)
        self.visit_count[action] += 1
        return action

    def train(self, action: int, reward: float):
        '''
        Train the EXP3 algorithm.

        Parameters
        ----------
        action : int
            The action to take.
        reward : float
            The reward received from the action.
        '''
        estimated_reward = reward / (self.action_probs[action] + self.gamma) # r_t / (p_t(a) + γ)

        self.weights[action] *= np.exp(- self.eta * estimated_reward)
        
        self.t += 1

    def reset(self) -> None:
        '''
        Reset the weights.
        '''
        self.weights = np.ones(self.n)
        self.action_probs = np.ones(self.n)
        self.visit_count = np.zeros(self.n)
        self.t = 0

    def __str__(self) -> str:
        return f'EXP3(gamma={self.gamma})'