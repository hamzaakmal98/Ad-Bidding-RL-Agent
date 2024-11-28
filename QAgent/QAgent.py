import numpy as np
import random
class QAgent:
    def __init__(self, priority_keywords, num_actions, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.999999):
        """
        Initialize the agent with Q-table, priority keywords, and learning parameters.

        Parameters:
            priority_keywords (list): List of keywords the agent cares about, in order of priority.
            num_actions (int): Number of possible bid adjustment actions.
            alpha (float): Learning rate for Q-learning.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Initial exploration rate.
            epsilon_decay (float): Factor to decay epsilon after each episode.
            epsilon_min (float): Minimum exploration rate.
        """
        self.priority_keywords = priority_keywords
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Q-table: shape = (state space size, num_actions)
        self.q_table = {}

        self.bids = {keyword: 50.0 for keyword in priority_keywords}

        self.actions = [-40, -25, 0, 25, 40] 

    def choose_action(self, state):
        """
        Choose an action based on exploration or exploitation.

        Parameters:
            state (tuple): Current state.

        Returns:
            int: Chosen action index.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)

        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, current_state, action_idx, reward, next_state):
        """
        Update the Q-table using the Q-learning formula.

        Parameters:
            current_state (tuple): The current state.
            action_idx (int): Index of the action taken.
            reward (float): Reward received for the action.
            next_state (tuple): The next state.
        """
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)

        best_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[current_state][action_idx]

        self.q_table[current_state][action_idx] = current_q + self.alpha * (reward + self.gamma * (best_future_q - current_q))

    def decay_epsilon(self):
        """Decay the exploration rate."""
        self.epsilon = self.epsilon * self.epsilon_decay

    def calculate_bid(self, keyword, action_idx):
        """
        Calculate the bid amount based on the action taken.

        Parameters:
            keyword (str): The keyword being bid on.
            action_idx (int): Index of the action.

        Returns:
            float: Updated bid amount.
        """
        bid_adjustment = self.actions[action_idx]
        self.bids[keyword] = max(0, self.bids[keyword] + bid_adjustment)
        return self.bids[keyword]

