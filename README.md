# Ad Bidding Reinforcement Learning Agents

## Q-Learning Agent


## DQN Agent
To reproduce the result in the [report](report.pdf) (cumulative average win rate being over 50% at the end of training), please clone the repository and run the following command in the terminal under the DQN_Agent folder: 

`python MultiTaskAgent.py --num_episodes 10000`

### Specific Configuration Details:

Agent hyperparameters:

| Hyperparameter | Value |
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


The training process took 3 hours to train on a cluster with the following settings:

| Setting | |
| ----------- | ----------- |
| GPU type | A100 | 
| GPU node count | 1 | 
| CPU count | 2 | 
| Memory | 16G | 
| python version | 3.8.8 | 
| anaconda version | 2021.11 | 