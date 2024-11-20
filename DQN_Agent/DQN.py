import torch
from torch import nn
from collections import deque
import itertools
import numpy as np
import random
from typing import Optional
from simulator.simul import AuctionSimulator

'''
NOTE by Weijia: 
The current implementation was adapted from the initial version I created that was tested successfully on another Gym game,
thus there is some assumption that may not be applicable to this ad bidding simulation. 
I will change the details after the simulator part is done. 
'''

class DQNAgent:
    ''' A Deep Q Network agent using the concept of Double Q-learning.

    Attributes:
        env (AuctionSimulator) : bidding simulating environment
        value_per_click (float) : Base value per click used in dynamic bid calculation
        gamma (float) : discount factor when considering future reward
        train_batch_size (int) : number of samples to be trained on during each training iteration
        replay_buffer_size (int) : size of the replay buffer that contains the history of transitions done
                                   which will be used to train the model
        min_replay_size (int) : number of samples resulted from random actions to be filled into the replay buffer
                                before the actual Q learning process begins
        reward_buffer_size (int) : size of the reward buffer that contains reward from all episodes done 
        epsilon_start (int) : epsilon value (for epsilon greedy exploration strategy) at the start 
        epsilon_end (int) : epsilon value at the end 
        epsilon_decay_period (int) : number of steps that the epsilon values will decay from the starting value to the ending value
        target_update_frequency (int) : frequency that the target net will be updatedd (by copying over online net's parameters)
        learning_rate (float) : learning rate used to optimized the online net
        logging_frequency (int or None) : frequency of the logging. If None, then no logging will be shown
    '''
    def __init__(self, env: AuctionSimulator, value_per_click: float, gamma: float = 0.99, train_batch_size: int = 32, 
                 replay_buffer_size: int = 50000, min_replay_size: int = 1000, reward_buffer_size: int = 100, 
                 epsilon_start: float = 1.0, epsilon_end: float = 0.02, epsilon_decay_period: int = 10000,
                 target_update_frequency: int = 1000, learning_rate: float = 5e-4, logging_frequency: Optional[int] = 1000):
        self.env = env
        self.value_per_click = value_per_click
        self.gamma = gamma
        self.train_batch_size = train_batch_size
        self.replay_buffer_size = replay_buffer_size
        self.min_replay_size = min_replay_size
        self.reward_buffer_size = reward_buffer_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start
        self.epsilon_decay_period = epsilon_decay_period
        self.target_update_frequency = target_update_frequency
        self.learning_rate = learning_rate
        self.logging_frequency = logging_frequency

        # Keeps track of (observation, action, reward, done, new_observation) transitions that have been played
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        # Keeps track of total rewards earned for each episode
        self.reward_buffer = deque([0], maxlen=reward_buffer_size)

        ### Use the Double Q-learning approach to mitigate over-optimism
        self.online_net = DQN(env)      # the main model to be optimized to predict expected reward more accurately
        self.target_net = DQN(env)      # the target model for predicting future reward

        # Ensures the 2 model have the same initialization
        self.target_net.load_state_dict(self.online_net.state_dict())
        # Only need an optimizer for the online net, as the target net will be updated by copying over online net's parameters
        self.optimizerDQN = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)

        ## Use another neural network to determine the bidding price
        self.price_net = BidPriceNet()  # the model determining a price to bid in order to win the auction
        self.optimizerPrice = torch.optim.Adam(self.price_net.parameters(), lr=learning_rate)

    def calculate_bid(self, pctr):
        """ Calculates a dynamic bid amount based on pCTR (predicted Click-Through Rate) and value per click."""
        return pctr * self.value_per_click
    
    def select_action(self, obs):
        """Select an action based on epsilon-greedy policy."""
        if random.random() <= self.epsilon:
            action = self.random_action(obs)         # Take a random action
        else:
            action = self.online_net.act(obs)        # Take an action maximizing the future reward
        return action
        
    def random_action(self, obs):
        # TODO: needs to change to discrete action space
        return self.calculate_bid(obs[2]) * random.uniform(0.8, 1.2)  
    
    def train(self, num_episodes=10000):
        episodes_done = 0
        # Initialize the replay buffer
        obs, _ = self.env.reset()    # Returned info (dict) is ignored
        # Partially fill the replay buffer with observations from completely random actions for exploration of the environment
        for _ in range(self.min_replay_size):
            action = self.random_action(obs)

            new_obs, reward, done, _, _ = self.env.step(action, bid, bid_price)  # truncated (bool) and info (dict) are ignored
            transition = (obs, action, reward, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs

            if done:
                obs, _ = self.env.reset()    # Returned info (dict) is ignored
                episodes_done += 1
        
        ### Main training loop
        obs, _ = self.env.reset()    # Returned info (dict) is ignored
        episode_reward = 0

        # for step in range(1):     # For debug purposes
        for step in itertools.count():  # Infinite loop, need to break using some logic
            current_state = self.env.get_metrics()
            # Uses epsilon greedy approach for facilitating exploration in the beginning
            self.epsilon = np.interp(step, [0, self.epsilon_decay_period], [self.epsilon_start, self.epsilon_end])
            action = self.select_action(obs)
            
            if action == 0:
                bid = False
                bid_price = 0
            else:
                bid = True
                bid_price = torch.clamp(self.price_net(), 0, current_state["Remaining Budget"]).item()

            # Take the action and record the transition into the replay buffer
            new_obs, reward, done, _, _ = self.env.step(action, bid, bid_price)  # truncated (bool) and info (dict) are ignored
            transition = (obs, action, reward, done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs
            episode_reward += reward

            # Record the total reward earned and reset for the next episode
            if done:
                obs, _ = self.env.reset()    # Returned info (dict) is ignored
                self.reward_buffer.append(episode_reward)
                episode_reward = 0
                episodes_done += 1

            ### After the number of episodes is reach, break the training loop
            if episodes_done >= num_episodes:
                break

            ### Starts gradient step, sample random minibatch of transitions from the replay buffer
            transitions = random.sample(self.replay_buffer, self.train_batch_size)

            # Separate the transitions into tensors of observations, actions, rewards, dones, and new_observations
            # * Note: First converts lists to np arrays then to torch tensors can be faster than directly from lists to tensors
            observations = torch.as_tensor(np.asarray([t[0] for t in transitions]), dtype=torch.float32)
            # the batch number is the number of randomly sampled transitions, add an dimension at the end to make each batch have its own sub tensor 
            actions = torch.as_tensor(np.asarray([t[1] for t in transitions]), dtype=torch.int64).unsqueeze(-1)
            rewards = torch.as_tensor(np.asarray([t[2] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
            dones = torch.as_tensor(np.asarray([t[3] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
            new_observations = torch.as_tensor(np.asarray([t[4] for t in transitions]), dtype=torch.float32)

            ### Compute target for loss function
            target_q_values = self.target_net(new_observations)
            max_target_q_value = target_q_values.max(dim=1, keepdim=True)[0]    # max() returns (max_values, indices), extract the max values
            # Estimate the total reward by summing of the current actual reward after taking an action on an observation
            # and future rewards predicted by target_net model.
            # If an episode terminates at the next step of a selected transition, 
            # then no need to add in rewards for the future
            targets = rewards + self.gamma * (1 - dones) * max_target_q_value

            ### Compute loss and apply gradients
            q_values = self.online_net(observations)
            # Get the predicted q-values of the actions of the radom sampled transition from the replay buffer
            action_q_values = torch.gather(input=q_values, dim=1, index=actions)
            # Calculate the loss between the online_net's prediction of the reward from taking the actions on the current observation states
            # and what the actual reward is (plus the predicted future reward by target_net)
            lossDQN = nn.functional.smooth_l1_loss(action_q_values, targets)

            ### Gradient Descent: update the online net to have more accurate estimation of the rewards 
            # that can be earned by each action on an observation state
            self.optimizerDQN.zero_grad()
            lossDQN.backward()
            self.optimizerDQN.step()

            # If the agent chose to bid for this round, then update the price net 
            if bid:
                lossPrice = nn.functional.mse_loss(bid_price, targets)
                self.optimizerPrice.zero_grad()
                lossPrice.backward()
                self.optimizerPrice.step()

            ### Update the target net model by copying over the online net's parameter by a frequency
            if step % self.target_update_frequency == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            ### Logging
            if step % self.logging_frequency == 0:
                print('Step ', step)
                print('Average reward : ', np.mean(self.reward_buffer))
        

class DQN(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))
        self.layers = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n),
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def act(self, obs):
        '''Determines an optimal action for the given observation'''
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_tensor.unsqueeze(0)) # add in batch dimension, then forward pass
        max_q_index = torch.argmax(q_values, dim = 1)
        action = max_q_index.detach().item()

        return action
    

class BidPriceNet(nn.Module):
    '''Predicts a bidding price needed to win an auction, regardless the current budget that the agent has. '''
    def __init__(self, env):
        super(BidPriceNet, self).__init__()

        in_features = int(np.prod(env.observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        bid_price = self.net(state)

        return bid_price


if __name__ == "__main__":
    agent = DQNAgent()
