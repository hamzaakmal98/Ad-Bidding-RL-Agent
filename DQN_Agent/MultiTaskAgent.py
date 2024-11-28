import torch
from torch import nn
from collections import deque
import itertools
import numpy as np
import random
from typing import Optional
import sys
sys.path.append(sys.path[0] + '/../')
from simulator.simul import AuctionSimulator

'''
Contributor: Weijia

This implementation uses a single multi-task multi-head model as the backbone of the DQN agent. 
'''

class DQNAgent:
    ''' A Deep Q Network agent using the concept of Double Q-learning and Multi-task Learning.

    Args:
        env (AuctionSimulator) : bidding simulating environment
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
        weight_DQN_loss (float) : weight for the DQN loss in the joint loss calculation
        weight_price_loss (float) : weight for the loss of the predicted price in the joint loss calculation
        target_update_frequency (int) : frequency that the target net will be updatedd (by copying over online net's parameters)
        learning_rate (float) : learning rate used to optimized the online net
        logging_frequency (int or None) : frequency of the logging. If None, then no logging will be shown
    '''
    def __init__(self, env: AuctionSimulator, gamma: float = 0.99, train_batch_size: int = 32, 
                 replay_buffer_size: int = 50000, min_replay_size: int = 1000, reward_buffer_size: int = 100, 
                 epsilon_start: float = 1.0, epsilon_end: float = 0.02, epsilon_decay_period: int = 10000,
                 weight_DQN_loss: float = 1.0, weight_price_loss: float = 1.0,
                 target_update_frequency: int = 1000, learning_rate: float = 5e-4, logging_frequency: Optional[int] = 1000):
        self.env = env
        self.gamma = gamma
        self.train_batch_size = train_batch_size
        self.min_replay_size = min_replay_size
        self.reward_buffer_size = reward_buffer_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start
        self.epsilon_decay_period = epsilon_decay_period
        self.weight_DQN_loss = weight_DQN_loss
        self.weight_price_loss = weight_price_loss
        self.target_update_frequency = target_update_frequency
        self.learning_rate = learning_rate
        self.logging_frequency = logging_frequency

        # Keeps track of (observation, action, reward, done, new_observation) transitions that have been played
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        # Keeps track of total rewards earned for each episode
        self.reward_buffer = deque([0], maxlen=reward_buffer_size)

        ### Use the Double Q-learning approach to mitigate over-optimism
        self.online_net = DeepQBidNet(env)      # the main model to be optimized to predict expected reward more accurately
        self.target_net = DeepQBidNet(env)      # the target model for predicting future reward

        # Ensures the 2 model have the same initialization
        self.target_net.load_state_dict(self.online_net.state_dict())
        # Only need an optimizer for the online net, as the target net will be updated by copying over online net's parameters
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)


    def select_action(self, obs):
        '''Select an action using the Online Net to maximize the future reward.'''
        action, bid_price = self.online_net.act(obs)
        bid = action != 0 and self.index_to_keyword(action) in self.env.get_current_available_keywords()
        return bid, action, bid_price.item()
        

    def random_action(self, budget):
        '''Select a random action in the action space.'''
        keyword_selection = random.randint(0, self.env.get_action_space_dim() - 1)
        # When the keyword index selected is the dummy 0, it means the agent doesn't participate in this round of auction
        bid = keyword_selection != 0 and self.index_to_keyword(keyword_selection) in self.env.get_current_available_keywords()
        bid_price = random.randint(1, int(budget))   
        return bid, keyword_selection, bid_price
    

    def index_to_keyword(self, index):
        ''' Translates an index output from the DQN to the corresponding keyword.
            If the index is 0, it represents no keyword is selected.
        '''
        return env.get_all_ad_keywords()[index - 1] if index > 0 else None


    def train(self, num_episodes=260):
        num_episodes_done = 0

        ### Initialize the replay buffer
        obs, info = self.env.reset()   
        # Partially fill the replay buffer with observations from completely random actions for exploration of the environment
        for _ in range(self.min_replay_size):
            bid, action, bid_price = self.random_action(info["remaining_budget"])
            keyword = self.index_to_keyword(action)
            new_obs, reward, done, info = self.env.run_auction_step(bid, keyword, bid_price) 
            transition = (obs, action, reward, info["highest_competitor_bid"], done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs

            if done:
                obs, info = self.env.reset() 
                num_episodes_done += 1
        
        ### Main training loop
        obs, info = self.env.reset()
        episode_reward = 0

        # for step in range(1):     # For debug 
        for step in itertools.count():  # Infinite loop, need to break using some logic
            # Uses epsilon greedy approach for facilitating exploration in the beginning
            self.epsilon = np.interp(step, [0, self.epsilon_decay_period], [self.epsilon_start, self.epsilon_end])
            if random.random() <= self.epsilon:
                bid, action, bid_price = self.random_action(info["remaining_budget"])   # Take a random action
            else:
                bid, action, bid_price = self.select_action(obs)    # Take a greedy action to maximize the future reward

            # print("Decides to", "bid" if bid else "not bid")
            # print("Current available keywords:", self.env.get_current_available_keywords(), 'Selected:', env.get_all_ad_keywords()[action - 1] if action != 0 else None)

            # Take the action and record the transition into the replay buffer
            keyword = self.index_to_keyword(action)
            new_obs, reward, done, info = self.env.run_auction_step(bid, keyword, bid_price) 
            transition = (obs, action, reward, info["highest_competitor_bid"], done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs
            episode_reward += reward
            # print(f"Bid for index {action} : {keyword} at price {bid_price} and", "won" if info["win"] else "lost")

            # Record the total reward earned and reset for the next episode
            if done:
                obs, info = self.env.reset()
                self.reward_buffer.append(episode_reward)
                episode_reward = 0
                num_episodes_done += 1

                ### After the number of episodes is reach, break the training loop
                if num_episodes_done >= num_episodes:
                    break

            ### Starts gradient step, sample random minibatch of transitions from the replay buffer
            transitions = random.sample(self.replay_buffer, self.train_batch_size)

            # Separate the transitions into tensors of observations, actions, rewards, dones, and new_observations
            # * Note: First converts lists to np arrays then to torch tensors can be faster than directly from lists to tensors
            observations = torch.as_tensor(np.asarray([t[0] for t in transitions]), dtype=torch.float32)
            # The batch number is the number of randomly sampled transitions, add an dimension at the end to make each batch have its own sub tensor 
            actions = torch.as_tensor(np.asarray([t[1] for t in transitions]), dtype=torch.int64).unsqueeze(-1)
            rewards = torch.as_tensor(np.asarray([t[2] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
            # The fourth element in each transition tuple is the highest competitor bid price for the keyword selected for that round
            # (if the agent decides not to bid on any keyword in a round, this value will be 0). A small value of 1 is added, 
            # so the agent will be trained to give a bid price slightly higher than the possible highest competitor price
            price_to_win = torch.as_tensor(np.asarray([t[3] for t in transitions]) + 1, dtype=torch.float32).unsqueeze(-1)
            dones = torch.as_tensor(np.asarray([t[4] for t in transitions]), dtype=torch.float32).unsqueeze(-1)
            new_observations = torch.as_tensor(np.asarray([t[5] for t in transitions]), dtype=torch.float32)

            ### Compute target for loss function
            target_q_values, _ = self.target_net(new_observations)
            max_target_q_value = target_q_values.max(dim=1, keepdim=True)[0]    # max() returns (max_values, indices), extract the max values
            # Estimate the total reward of an action by summing the current actual reward after taking the action
            # and the maximum future rewards predicted by target_net model with a factor of gamma.
            # If an episode terminates at the next step of a selected transition, then there is no future rewards
            targets = rewards + self.gamma * (1 - dones) * max_target_q_value

            ### Compute loss and apply gradients
            q_values, bid_prices = self.online_net(observations)
            # Get the predicted q-values of the actual actions from the radom sampled transitions from the replay buffer
            action_q_values = torch.gather(input=q_values, dim=1, index=actions)
            # Calculate the loss between the online_net's prediction of the reward from taking the actions 
            # and what the actual reward is (plus the predicted future reward by target_net)
            loss_DQN = nn.functional.smooth_l1_loss(action_q_values, targets)
            # Calculate the loss between the predicted price and the actual price needed to win the bid
            loss_price = nn.functional.mse_loss(bid_prices, price_to_win)

            loss_total = self.weight_DQN_loss * loss_DQN + self.weight_price_loss * loss_price

            ### Gradient Descent: update the online net to have more accurate estimation of the rewards 
            # that can be earned by each action on an observation state
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

            ### Update the target net model by copying over the online net's parameter by a frequency
            if step % self.target_update_frequency == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            ### Logging
            if step % self.logging_frequency == 0:
                print(f'Step {step} - Episode {num_episodes_done}')
                print(f'Average reward of past {len(self.reward_buffer)} episodes : {np.mean(self.reward_buffer)}')
        

class DeepQBidNet(nn.Module):
    '''A multi-task model that has 2 heads for predicting keyword selection and bid price respectively.'''
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.get_observation_space_dim()))
        self.main_trunk = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.keyword_head = nn.Linear(64, env.get_action_space_dim())
        self.price_head = nn.Linear(64, 1)
    
    def forward(self, x):
        shared_features = self.main_trunk(x)
        q_values = self.keyword_head(shared_features)
        bid_price = self.price_head(shared_features)
        return q_values, bid_price
    
    def act(self, obs):
        '''Determines an optimal bidding action (keyword and price) for the given observation'''
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        q_values, bid_price = self(obs_tensor.unsqueeze(0)) # add in batch dimension, then forward pass
        max_q_index = torch.argmax(q_values, dim=1)
        action = max_q_index.detach().item()
        # print('q_values', q_values)
        # print('action', action)
        return action, bid_price
    

if __name__ == "__main__":
    env = AuctionSimulator(initial_budget=1000, keyword_list=['A', 'B', 'C'])
    agent = DQNAgent(env)
    agent.train()
