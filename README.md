# Ad Bidding Reinforcement Learning Agents
## 🏗️ Architecture Flow (Mermaid)

This flowchart illustrates how the simulator coordinates the interaction between the RL agents and the auction environment.

``` mermaid
graph TD
    subgraph Environment_Simulation [Auction Environment]
        E[State: Budget, Time, Competitor Bids] --> S[Simulator]
    end

    subgraph RL_Agents [Agent Strategies]
        S --> Q[Tabular Q-Agent]
        S --> D[Multi-Task DQN Agent]
    end

    subgraph Training_Loop [Feedback Loop]
        Q --> R{Reward Function}
        D --> R
        R -->|Update Q-Values| Q
        R -->|Backprop / Replay Buffer| D
    end

    subgraph Outcome [Market Results]
        R --> W[Win Rate Tracking]
        R --> B[Budget Consumption]
    end

    style D fill:#69f,stroke:#333,stroke-width:2px
    style R fill:#f96,stroke:#333,stroke-width:2px

```

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
| Python version | 3.8.8 | 
| Anaconda version | 2021.11 | 


## Q Learning Agent

To reproduce the result please use these hyperparameters and run the main.py file present in the QAgent directory associated with the Q learning agent.

### Specific configuration details:

| Hyperparameter  | Value                    |
| --------------- | ------------------------ |
| alpha           | 0.1                      |
| gamma           | 0.9                      |
| epsilon         | 1.0                      |
| epsilon_decay   | 0.999                    |
| actions         | [-6, -4, -2, 0, 2, 4, 6] |
| num_episodes    | 100000                   |
| initial_budget  | 10000                    |

## 📈 Results & Performance Comparison

The project evaluated two distinct reinforcement learning architectures over 100,000 training episodes. The primary goal was to maximize the Cumulative Win Rate while maintaining a stable Budget Decay.
**1. Tabular Q-Learning (Baseline)**

 - **Performance:** Achieved a steady win rate but struggled with high-dimensional state spaces.

 - **Behavior:** Effective for simple, discrete auction environments but showed limited adaptability to stochastic competitor bidding patterns.

 - **Limitation**: Memory constraints limited the granularity of the state-action pairs.

**2. Multi-Task DQN (Optimized) WINNER!**

 - **Performance:** Consistently achieved a >50% win rate.

 - **Behavior:** By utilizing a Neural Network for function approximation, the agent successfully identified non-linear patterns in competitor behavior.

 - **Advantage:** The Experience Replay Buffer allowed the agent to learn from past "expensive" mistakes without needing to repeat them, leading to superior budget preservation.
