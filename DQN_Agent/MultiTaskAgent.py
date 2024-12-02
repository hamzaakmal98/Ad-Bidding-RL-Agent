import torch
from torch import nn
from collections import deque
import itertools
import numpy as np
import random
from typing import Optional
import sys
import matplotlib.pyplot as plt
import os
sys.path.append(os.getcwd())
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
        reward_buffer_size (int) : the number of the most recent episodes whose episode reward will be recorded 
                                   in the reward buffer which will be used to calculate the average reward earned
        epsilon_start (int) : epsilon value (for epsilon greedy exploration strategy) at the start 
        epsilon_end (int) : epsilon value at the end 
        epsilon_decay_period (int) : number of steps that the epsilon values will decay from the starting value to the ending value
        weight_DQN_loss (float) : weight for the DQN loss in the joint loss calculation
        weight_price_loss (float) : weight for the loss of the predicted price in the joint loss calculation
        target_update_frequency (int) : frequency that the target net will be updatedd (by copying over online net's parameters)
        learning_rate (float) : learning rate used to optimized the online net
        logging_frequency (int or None) : frequency of the logging during training (by the number of steps). 
                                          If None, then no logging will be shown
    '''
    def __init__(self, env: AuctionSimulator, gamma: float = 0.5, train_batch_size: int = 32,
                 replay_buffer_size: int = 50000, min_replay_size: int = 1000, reward_buffer_size: int = 10, 
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay_period: int = 20000,
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


    def select_action(self, obs, budget):
        '''Select an action using the Online Net to maximize the future reward.'''
        action, bid_price = self.online_net.act(obs)
        bid = action != 0 and self.index_to_keyword(action) in self.env.get_current_available_keywords()
        bid_price = torch.clamp(bid_price, 0, budget).item()
        return bid, action, bid_price
        

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


    def train(self, num_episodes=260, model_save_path: str = None):
        self.online_net.train()

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
        
        ### Main training loop
        obs, info = self.env.reset()
        num_episodes_done = 0   

        # For tracking and visualization
        episode_reward = 0
        num_wins = 0
        num_bids_placed = 0
        num_episode_steps = 0
        episode_rewards = []
        episode_win_rates = []     # Number of wins / Number of auctions with a bid placed
        episode_steps = []
        cumulative_rewards = 0
        cumulative_wins = 0
        cumulative_bids_placed = 0
        cumulative_episode_rewards = []
        cumulative_episode_win_rates = []

        if self.logging_frequency:
            num_random_actions = 0      # for each logging period
            num_greedy_actions = 0      # for each logging period

        # for step in range(1):     # For debug 
        for step in itertools.count():  # Infinite loop, need to break using some logic

            # Uses epsilon greedy approach for facilitating exploration in the beginning
            self.epsilon = np.interp(step, [0, self.epsilon_decay_period], [self.epsilon_start, self.epsilon_end])
            if random.random() <= self.epsilon:
                bid, action, bid_price = self.random_action(info["remaining_budget"])   # Take a random action
                num_random_actions += 1
            else:
                bid, action, bid_price = self.select_action(obs, info["remaining_budget"])    # Take a greedy action to maximize the future reward
                num_greedy_actions += 1

            # if num_episodes_done in [0, num_episodes - 1] and self.index_to_keyword(action) in ['A', 'B', 'C']:
            #     print('=======================')
            #     print("Decides to", "bid" if bid else "not bid")
            #     print("Current available keywords:", self.env.get_current_available_keywords(), 'Selected:', env.get_all_ad_keywords()[action - 1] if action != 0 else None)

            # Take the action and record the transition into the replay buffer
            keyword = self.index_to_keyword(action)
            new_obs, reward, done, info = self.env.run_auction_step(bid, keyword, bid_price) 
            transition = (obs, action, reward, info["highest_competitor_bid"], done, new_obs)
            self.replay_buffer.append(transition)
            obs = new_obs
            episode_reward += reward
            cumulative_rewards += reward
            # if num_episodes_done in [0, num_episodes - 1] and keyword in ['A', 'B', 'C']:
            #     print(f"Bid for index {action} : {keyword} at price {bid_price} and", "won" if info["win"] else "lost")
            if bid:
                num_bids_placed += 1
                cumulative_bids_placed += 1
                if info["win"]:
                    num_wins += 1
                    cumulative_wins += 1
            num_episode_steps += 1

            # Record the total reward earned and reset for the next episode
            if done:
                self.reward_buffer.append(episode_reward)
                episode_rewards.append(episode_reward)
                episode_win_rates.append(num_wins / num_episode_steps)
                episode_steps.append(num_episode_steps)
                cumulative_episode_rewards.append(cumulative_rewards)
                cumulative_episode_win_rates.append(cumulative_wins / cumulative_bids_placed)

                obs, info = self.env.reset()
                episode_reward = 0
                num_wins = 0
                num_bids_placed = 0
                num_episode_steps = 0
                num_episodes_done += 1

                ### After the number of episodes is reach, break the training loop
                if num_episodes_done >= num_episodes:
                    if model_save_path:
                        torch.save(self.online_net.state_dict(), model_save_path)
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
            loss_price = nn.functional.smooth_l1_loss(bid_prices, price_to_win)
            

            # loss_total = self.weight_DQN_loss * loss_DQN + self.weight_price_loss * loss_price

            ### Gradient Descent: update the online net to have more accurate estimation of the rewards 
            # that can be earned by each action on an observation state
            # self.optimizer.zero_grad()
            # loss_total.backward()
            # self.optimizer.step()

            # Step 1: Update the keyword_head using loss_DQN (freezing price_head)
            self.optimizer.zero_grad()
            for param in self.online_net.price_head.parameters():
                param.requires_grad = False  # Freeze price_head, only updating keyword_head here
            loss_DQN.backward(retain_graph=True)  # Retain graph for the second backward pass

            # Step 2: Update the price_head using loss_price (freezing keyword_head)
            for param in self.online_net.price_head.parameters():
                param.requires_grad = True  # Unfreeze price_head
            for param in self.online_net.keyword_head.parameters():
                param.requires_grad = False  # Freeze keyword_head
            loss_price.backward()

            # Step 3: Backward pass (unfreezing keyword_head)
            for param in self.online_net.keyword_head.parameters():
                param.requires_grad = True  # Unfreeze keyword_head
            self.optimizer.step()

            ### Update the target net model by copying over the online net's parameter by a frequency
            if step % self.target_update_frequency == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            ### Logging
            if self.logging_frequency and step > 0 and step % self.logging_frequency == 0:
                print(f'Step {step} - Episode {num_episodes_done} ({num_random_actions} random actions, {num_greedy_actions} greedy actions)')
                print(f'Average reward of past {len(self.reward_buffer)} episodes : {np.mean(self.reward_buffer)}')
                # print(f'Step {step} - Episode {num_episodes_done} ({num_random_actions} random actions, {num_greedy_actions} greedy actions) - Last Done Episode Reward = {episode_rewards[-1]}')
                num_random_actions = 0
                num_greedy_actions = 0
        
        info = {
            'num_episodes' : num_episodes_done,
            'episode_rewards' : episode_rewards,
            'episode_win_rates' : episode_win_rates,
            'cumulative_episode_rewards' : cumulative_episode_rewards,
            'cumulative_win_rates' : cumulative_episode_win_rates,
            'episode_steps' : episode_steps
        }
        return info
    

    def evaluate(self, num_episodes=5, model_save_path: str = None):
        with torch.no_grad():
            if model_save_path:
                self.online_net.load_state_dict(torch.load(model_save_path))
            self.online_net.eval()
            obs, info = self.env.reset()

            # For tracking and visualization
            episode_reward = 0
            num_wins = 0
            num_bids_placed = 0
            num_episode_steps = 0
            episode_rewards = []
            episode_win_rates = []     # Number of wins / Number of auctions with a bid placed
            episode_steps = []
            cumulative_rewards = 0
            cumulative_wins = 0
            cumulative_bids_placed = 0
            cumulative_episode_rewards = []
            cumulative_episode_win_rates = []

            # for step in range(1):     # For debug 
            for episode in range(num_episodes):  # Infinite loop, need to break using some logic
                bid, action, bid_price = self.select_action(obs, info["remaining_budget"])    # Take a greedy action to maximize the future reward

                print("Decides to", "bid" if bid else "not bid")
                print("Current available keywords:", self.env.get_current_available_keywords(), 'Selected:', env.get_all_ad_keywords()[action - 1] if action != 0 else None)

                # Take the action and record the transition into the replay buffer
                keyword = self.index_to_keyword(action)
                obs, reward, done, info = self.env.run_auction_step(bid, keyword, bid_price) 
                episode_reward += reward
                cumulative_rewards += reward
                print(f"Bid for index {action} : {keyword} at price {bid_price} and", "won" if info["win"] else "lost")

                if bid:
                    num_bids_placed += 1
                    cumulative_bids_placed += 1
                    if info["win"]:
                        num_wins += 1
                        cumulative_wins += 1
                num_episode_steps += 1

                # Record the total reward earned and reset for the next episode
                if done:
                    self.reward_buffer.append(episode_reward)
                    episode_rewards.append(episode_reward)
                    episode_win_rates.append(num_wins / num_episode_steps)
                    episode_steps.append(num_episode_steps)
                    cumulative_episode_rewards.append(cumulative_rewards)
                    cumulative_episode_win_rates.append(cumulative_wins / cumulative_bids_placed)

                    obs, info = self.env.reset()
                    episode_reward = 0
                    num_wins = 0
                    num_bids_placed = 0
                    num_episode_steps = 0
            
            info = {
                'num_episodes' : num_episodes,
                'episode_rewards' : episode_rewards,
                'episode_win_rates' : episode_win_rates,
                'cumulative_episode_rewards' : cumulative_episode_rewards,
                'cumulative_win_rates' : cumulative_episode_win_rates,
                'episode_steps' : episode_steps
            }
        
            return info



class DeepQBidNet(nn.Module):
    '''A multi-task model that has 2 heads for predicting keyword selection and bid price respectively.'''
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.get_observation_space_dim()))
        self.main_trunk = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # self.keyword_head = nn.Linear(64, env.get_action_space_dim())
        # self.price_head = nn.Linear(64, 1)
        self.keyword_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, env.get_action_space_dim())
        )
        self.price_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
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
        return action, bid_price
    

if __name__ == "__main__":
    env = AuctionSimulator(initial_budget=10000, keyword_list=['A', 'B', 'C'])
    agent = DQNAgent(env)
    info = agent.train(num_episodes=2000, model_save_path='./Models_Saved/multi_task_DQN.pth')

    # print(info)

    # Visualization 1: reward graph
    plt.plot(info['episode_rewards'], color='crimson')
    plt.title('Model Learning (Rewards Over Time)')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.savefig('./Visualization/reward_graph.jpg')
    plt.close()

    # Visualization 2: win rate graph (denominator being the number of auctions the agent choose to place a bid)
    plt.plot(info['episode_win_rates'], color='crimson')
    plt.title('Win Rate Over Time')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate (Excluding Skipped Auctions)')
    plt.savefig('./Visualization/win_rate_graph.jpg')
    plt.close()

    # Visualization 3: trend graph of the total number of steps in an episode
    plt.plot(info['episode_steps'], color='crimson')
    plt.title('Budget Optimization Over Time')
    plt.xlabel('Episodes')
    plt.ylabel('Number of Auctions')
    plt.savefig('./Visualization/steps_count_graph.jpg')
    plt.close()

    # Visualization 4: cumulative reward graph
    plt.plot(info['cumulative_episode_rewards'], color='crimson')
    plt.title('Model Learning (Rewards Over Time)')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.savefig('./Visualization/cumulative_reward_graph.jpg')
    plt.close()

    # Visualization 5: cumulative win rate graph (denominator being the number of auctions the agent choose to place a bid)
    plt.plot(info['cumulative_win_rates'], color='crimson')
    plt.title('Win Rate Over Time')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate (Excluding Skipped Auctions)')
    plt.savefig('./Visualization/cumulative_win_rate_graph.jpg')
    plt.close()