# Ad Bidding Reinforcement Learning Agents

## Q-Learning Agent


## DQN Agent
To reproduce the result in the report (cumulative average win rate over time being over 50%), please clone the repository and run the following command in the terminal under the DQN_Agent folder: 

`python MultiTaskAgent.py --num_episodes 10000`

### Specific configuration details:

| Hyperparameters | Values |
| ----------- | ----------- |
| gamma | 0.75 |
| train_batch_size | 32 |
| replay_buffer_size | 50000 |
| min_replay_size | 1000 |
| reward_buffer_size | 10 |
| epsilon_start | 1.0 |
| epsilon_end | 0.01 |
| epsilon_decay_period | 20000 |
| weight_DQN_loss | 1.0 |
| weight_price_loss | 1.0 |
| target_update_frequency | 1000 |
| learning_rate | 0.0005 |
| initial_budget | 10000 |
| num_episodes | 10000 |
