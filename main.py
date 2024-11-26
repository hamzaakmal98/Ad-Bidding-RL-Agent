from config_loader import ConfigLoader
from logger import Logger
from environment.env import setup
from simulator.simul import AuctionSimulator
from QAgent.QAgent import QAgent

class Main:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize main class with config and logger."""
        self.config = ConfigLoader(config_path)
        self.logger = Logger()
        
        # Set up environment first
        setup()
        
        # Initialize simulator
        sim_config = self.config.get_simulator_config()
        self.simulator = AuctionSimulator(
            initial_budget=sim_config['initial_budget'],
            keyword_list=sim_config['priority_keywords']
        )
        
        # Get priority keywords from user
        self.simulator.prompt_keywords()
        
        # Initialize agent
        agent_config = self.config.get_agent_config()
        self.agent = QAgent(
            priority_keywords=self.simulator.desired_keywords,  # Use keywords from simulator
            num_actions=agent_config['num_actions'],
            alpha=agent_config['alpha'],
            gamma=agent_config['gamma'],
            epsilon=agent_config['epsilon'],
            epsilon_decay=agent_config['epsilon_decay'],
            epsilon_min=agent_config['epsilon_min']
        )
    
    def train(self, num_episodes: int):
        """Train the agent for specified number of episodes."""
        self.logger.log_info(f"Starting training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            self.simulator.reset()
            
            while not self.simulator.is_terminal():
                # Get available keywords for this round
                available_keywords = self.simulator.get_current_available_keywords()
                
                for keyword in available_keywords:
                    # Create state tuple
                    current_metrics = self.simulator.get_metrics()
                    state = (
                        keyword,
                        current_metrics["Remaining Budget"],
                        keyword in self.simulator.desired_keywords
                    )
                    
                    # Get action and bid
                    action_idx = self.agent.choose_action(state)
                    bid_amount = self.agent.calculate_bid(keyword, action_idx)
                    
                    # Make bid
                    sim_result, reward = self.simulator.run_auction_step(
                        bid_bool=True,
                        keyword=keyword,
                        bid_amount=bid_amount
                    )
                    
                    # Create next state
                    next_metrics = self.simulator.get_metrics()
                    next_state = (
                        keyword,
                        next_metrics["Remaining Budget"],
                        keyword in self.simulator.desired_keywords
                    )
                    
                    # Update agent
                    self.agent.update_q_table(state, action_idx, reward, next_state)
            
            # Decay exploration rate
            self.agent.decay_epsilon()
            
            # Log episode results
            final_metrics = self.simulator.get_metrics()
            self.logger.log_metrics(episode, final_metrics)
            
            if (episode + 1) % 100 == 0:
                self.logger.log_info(
                    f"Episode {episode + 1}: "
                    f"Win Rate = {final_metrics['Win Rate']:.4f}, "
                    f"Remaining Budget = {final_metrics['Remaining Budget']:.2f}, "
                    f"Total Wins = {final_metrics['Wins']}, "
                    f"Total Auctions = {final_metrics['Total Auctions']}, "
                    f"Total Rewards = {final_metrics['Cumulative Rewards so far']:.2f}"
                )

if __name__ == "__main__":
    main = Main()
    main.train(num_episodes=1000)