from config_loader import ConfigLoader
from logger import Logger
from environment.env import setup
from simulator.simul import AuctionSimulator
from QAgent.QAgent import QAgent
from rewards_functions import calculate_reward, aggregate_rewards


class Main:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize main class with config and logger."""
        self.config = ConfigLoader(config_path)
        self.logger = Logger()

        # Initialize environment, agent, and simulator with config
        self.env = self.setup_environment()
        self.simulator = self.setup_simulator()
        self.agent = self.setup_agent()

    def setup_environment(self):
        """Initialize environment with config."""
        env_config = self.config.get_env_config()
        setup()  # Initialize global parameters
        return env_config

    def setup_simulator(self):
        """Initialize auction simulator with config."""
        sim_config = self.config.get_simulator_config()
        return AuctionSimulator(
            num_competitors=sim_config['num_competitors'],
            initial_budget=sim_config['initial_budget'],
            value_per_click=sim_config['value_per_click'],
            bid_distribution=sim_config['bid_distribution'],
            mean=sim_config['mean_bid'],
            stddev=sim_config['std_dev']
        )

    def setup_agent(self):
        """Initialize Q-learning agent with config."""
        agent_config = self.config.get_agent_config()
        return QAgent(
            env=self.env,
            learning_rate=agent_config['learning_rate'],
            discount_factor=agent_config['discount_factor'],
            epsilon=agent_config['epsilon'],
            decay_rate=agent_config['decay_rate'],
            value_per_click=agent_config['value_per_click']
        )

    def train(self, num_episodes: int):     # TODO: don't worry about this, the current version of DQN and Q-Learning both implemented a training method
        """Train the agent for specified number of episodes."""
        self.logger.log_info(f"Starting training for {num_episodes} episodes")

        for episode in range(num_episodes):
            self.simulator.remaining_budget = self.simulator.initial_budget
            episode_metrics = []
            episode_rewards = []

            while self.simulator.remaining_budget > 0:
                # Generate impression
                impression = self.simulator.generate_impression()   # TODO: the impression is from the env?
                state = (
                    impression['keyword'],
                    self.simulator.remaining_budget,
                    impression['pCTR']
                )

                # Get agent's action
                bid_amount = self.agent.select_action(state)

                # Run auction
                result = self.simulator.run_auction(impression['pCTR'])

                # Create state object for reward calculation
                reward_state = {
                    'bid_placed': bid_amount,
                    'bid_cost': result['cost'],
                    'budget_left': self.simulator.remaining_budget,
                    'keyword_importance': impression['pCTR'] * 100,  # Scale pCTR to importance
                    'priority_keyword': impression['keyword'] in self.simulator.desired_keywords
                }

                # Calculate reward using teammate's function
                reward = calculate_reward(
                    state=reward_state,
                    max_budget_consumption_per_auction=0.25,
                    stop_penalty_percent=0.1,
                    stop_penalty_decay=0.5
                )

                episode_rewards.append(reward)

                # Update agent
                next_state = (
                    impression['keyword'],
                    self.simulator.remaining_budget,
                    impression['pCTR']
                )
                self.agent.update_q_value(state, bid_amount, reward, next_state)

                episode_metrics.append({
                    'win_rate': self.simulator.get_metrics()['Win Rate'],
                    'remaining_budget': self.simulator.remaining_budget,
                    'wins': self.simulator.num_wins,
                    'bid_amount': bid_amount,
                    'reward': reward
                })

            # Log episode metrics
            avg_metrics = {
                'win_rate': sum(m['win_rate'] for m in episode_metrics) / len(episode_metrics),
                'remaining_budget': episode_metrics[-1]['remaining_budget'],
                'wins': episode_metrics[-1]['wins'],
                'avg_bid': sum(m['bid_amount'] for m in episode_metrics) / len(episode_metrics),
                'total_reward': aggregate_rewards(episode_rewards)
            }
            self.logger.log_metrics(episode, avg_metrics)

            if (episode + 1) % 100 == 0:
                self.logger.log_info(
                    f"Episode {episode + 1}: Win Rate = {avg_metrics['win_rate']:.4f}, "
                    f"Remaining Budget = {avg_metrics['remaining_budget']:.2f}, "
                    f"Total Reward = {avg_metrics['total_reward']:.2f}"
                )


if __name__ == "__main__":
    main = Main()
    main.train(num_episodes=1000)