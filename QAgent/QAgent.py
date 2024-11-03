'''
    Contributor: Sarang
    READ ME:

    We should not generalize the q agent to be able to bid across all the industries.
    Would cause problems since Q tables overlap and data would mix from different fields.
    Also a state explosion would occur where too much memory and training data would be required. 

    Better alternative is to train each instance of the model for a specific ad campaign. The environment would provide ad impressions
    (specific for one type of ad i.e. tech, fashion, food etc.) and each ad impression would have 1 specific keyword associated with it
    and a pCTR (predicted click through rate) which would tell the agent how relevant and indemand this keyword is for this ad impression.
    Using this data the agent will then calculate a bidding amount.

    Why 1 keyword per impression?
    In real world scenerio, an impression is generated when a user searches something. Ex. "Best Smartphones under 1000$"
    The platforms like google Ads, recognize a keyword like Smartphones and create and send ad impressions for this keyword. 
    Users have to decide whether they bid for this keyword or not. Similarly, our agent will be provided with impression for 1 industry,
    and using pCTR it will decide whether it wants to bid for this keyword and if yes then how much. In this way our Q table gets filled
    and when we use it in real scenerios it already has data regarding which keywords it prefers more compared to others from the Q tables.

'''

import numpy as np
import random
"""  Contributor: Sarang  """

class QAgent:
    """  Contributor: Sarang  """
    def __init__(self, env, learning_rate, discount_factor, epsilon, decay_rate, value_per_click):
        """
        Initialize the QAgent with parameters for Q-learning.

        Parameters:
        - env: The environment the agent will interact with.
        - learning_rate : Initial learning rate.
        - discount_factor : Discount factor (gamma) for future rewards.
        - epsilon : Initial exploration rate.
        - decay_rate : Decay rate for epsilon after each episode.
        - value_per_click : Base value per click used in dynamic bid calculation.
        """
        self.env = env
        self.initial_learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.value_per_click = value_per_click
        self.q_table = {}  # Dictionary to store Q-values
        self.num_updates = {}  # Track updates for each (state, action) pair

    """  Contributor: Sarang  """

    def calculate_bid(self, pctr):
        """Calculate a dynamic bid amount based on pCTR and value per click.
            Right now it is a bit simple and straight forward but we can make it more efficient by incorporating num of steps remaining or budget etc.
        """
        return pctr * self.value_per_click

    """  Contributor: Sarang  """
    def select_action(self, state):
        """Select an action based on epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return self.calculate_bid(state[2]) * random.uniform(0.8, 1.2)  # Random exploration - range "reduces by 20% to increases by 20%"
        else:
            if state in self.q_table:
                return max(self.q_table[state], key=self.q_table[state].get)  # Exploit best action
            else:
                return self.calculate_bid(state[2])  # Default if state not in Q-table

    """  Contributor: Sarang  """
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using a variable learning rate based on update count."""
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        if state not in self.num_updates:
            self.num_updates[state] = {}
        if action not in self.num_updates[state]:
            self.num_updates[state][action] = 0

        eta = 1 / (1 + self.num_updates[state][action])

        if next_state in self.q_table:
            best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
            td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        else:
            td_target = reward

        # Update Q-value
        self.q_table[state][action] = (1 - eta) * self.q_table[state][action] + eta * td_target

        self.num_updates[state][action] += 1


    '''

    This train method will be in the main class for more modularization. here just for understanding.If managing in the main method gets harder
    we can have it here as well.

    '''
    """  Contributor: Sarang  """

    def train(self, num_episodes=10000):

        for episode in range(num_episodes):
            # Reset environment and get initial state
            state, _ = self.env.reset()
            done = False

            while not done:
                action = self.select_action(state)

                reward, next_state, done = self.env.step(action)

                self.update_q_value(state, action, reward, next_state)

                state = next_state

            self.epsilon *= self.decay_rate

        return self.q_table
