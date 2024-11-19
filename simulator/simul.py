"""
Contributor: Hamza

This module simulates a second-price auction environment for ad bidding.
The agent bids based on the predicted Click-Through Rate (pCTR) and a predefined value per click.
The simulation includes:
1. Generating impressions with attributes like keywords, page content, user profile, and pCTR.
2. Collecting bids from competitors and the agent.
3. Running a second-price auction to determine winners.
4. Tracking the agent's performance metrics like budget, number of wins, and win rate.
"""

import random
import numpy as np

class AuctionSimulator:
    """
    Contributor: Hamza

    This class implements the simulation of a second-price auction environment.
    It manages budget tracking, impression generation, bidding logic, and performance metrics.
    """

    def __init__(self, num_competitors, initial_budget, value_per_click, bid_distribution='normal', mean=50, stddev=10):
        """
        Initialize the auction simulator with competitors and tracking metrics.

        Parameters:
        - num_competitors (int): Number of competitors in the auction.
        - initial_budget (float): Starting budget for the agent.
        - value_per_click (float): Base value per click used for dynamic bid calculation.
        - bid_distribution (str): Type of distribution for generating competitor bids ('normal' or 'uniform').
        - mean (float): Mean value for the bid distribution.
        - stddev (float): Standard deviation for the bid distribution (if applicable).
        """
        self.num_competitors = num_competitors
        self.bid_distribution = bid_distribution
        self.mean = mean
        self.stddev = stddev
        self.initial_budget = initial_budget
        self.remaining_budget = initial_budget
        self.num_wins = 0
        self.total_auctions = 0
        self.value_per_click = value_per_click
        self.desired_keywords = []  # User-defined keywords for tracking.

    def prompt_keywords(self):
        """
        Contributor: Hamza

        Prompt the user to enter three desired keywords to track in the auction environment.
        These keywords are stored in a list and used for impression generation.
        """
        print("Enter 3 desired keywords for tracking:")
        for i in range(3):
            keyword = input(f"Keyword {i + 1}: ").strip()
            self.desired_keywords.append(keyword)
        print(f"Tracked Keywords: {self.desired_keywords}")

    def generate_impression(self):
        """
        Contributor: Hamza

        Generate a simulated ad impression with attributes like keyword, page content, user profile, and pCTR.
        The impression is randomly generated based on tracked keywords or defaults.
        
        Returns:
        - dict: Impression details including keyword, user profile, page content, and pCTR.
        """
        keyword = random.choice(self.desired_keywords if self.desired_keywords else ["Keyword1", "Keyword2", "Keyword3"])
        page_content = f"Content for {keyword}"
        user_profile = f"User interested in {keyword}"
        pctr = round(random.uniform(0.1, 0.9), 2)  # Simulated pCTR between 0.1 and 0.9
        return {"keyword": keyword, "page_content": page_content, "user_profile": user_profile, "pCTR": pctr}

    def simulate_competitor_bids(self):
        """
        Contributor: Hamza

        Generate bids for competitors based on the selected bid distribution.

        Returns:
        - list: A list of bids from all competitors.
        """
        if self.bid_distribution == 'normal':
            bids = np.random.normal(self.mean, self.stddev, self.num_competitors)
            bids = [max(0, bid) for bid in bids]  # Ensure no negative bids
        elif self.bid_distribution == 'uniform':
            bids = np.random.uniform(0, self.mean * 2, self.num_competitors)
        else:
            raise ValueError("Invalid bid distribution type. Choose 'normal' or 'uniform'.")
        return bids

    def run_auction(self, pctr):
        """
        Contributor: Hamza

        Execute a second-price auction for the given pCTR. The agent bids using the formula:
        Bid = pCTR × Value per Click.

        The agent wins if its bid is the highest and pays the second-highest bid.

        Parameters:
        - pctr (float): Predicted Click-Through Rate for the impression.

        Returns:
        - dict: Results of the auction including win status, cost, margin, and impression details.
        """
        agent_bid = pctr * self.value_per_click  # Calculate agent's bid.
        competitor_bids = self.simulate_competitor_bids()
        all_bids = competitor_bids + [agent_bid]
        all_bids.sort(reverse=True)

        impression = self.generate_impression()
        print(f"Your Bid: {agent_bid:.2f}, Winning Bid: {all_bids[0]:.2f}")

        if agent_bid == all_bids[0]:  # Agent wins
            cost = all_bids[1]
            margin = agent_bid - all_bids[1]  # How much more the agent bid than the second-highest bid
            if self.remaining_budget >= cost:
                self.remaining_budget -= cost
                self.num_wins += 1
                self.total_auctions += 1
                return {"win": True, "cost": cost, "margin": margin, "impression": impression}
            else:
                return {"win": False, "cost": 0.0, "margin": margin, "impression": impression}  # Budget exceeded
        else:
            margin = all_bids[0] - agent_bid  # How much less the agent bid than the highest bid
            self.total_auctions += 1
            return {"win": False, "cost": 0.0, "margin": margin, "impression": impression}



    def get_metrics(self):
        """
        Contributor: Hamza

        Compute and return performance metrics for the agent.

        Returns:
        - dict: Metrics including remaining budget, total wins, total auctions, and win rate.
        """
        win_rate = self.num_wins / self.total_auctions if self.total_auctions > 0 else 0
        return {
            "Remaining Budget": self.remaining_budget,
            "Wins": self.num_wins,
            "Total Auctions": self.total_auctions,
            "Win Rate": win_rate,
        }


# Example usage of the Auction Simulator
if __name__ == "__main__":
    # Initialize the simulator
    simulator = AuctionSimulator(num_competitors=5, initial_budget=500, value_per_click=10, bid_distribution='normal', mean=50, stddev=10)
    
    # Prompt user to define keywords
    simulator.prompt_keywords()

    # Simulate 10 auctions
    for i in range(10):
        impression = simulator.generate_impression()
        print(f"\nAuction {i + 1}: Impression Details: {impression}")

        # Run the auction for the generated impression
        result = simulator.run_auction(impression["pCTR"])
        if result["win"]:
            print(f"You won the auction! Cost: {result['cost']:.2f}, Won by: {result['margin']:.2f}")
        else:
            print(f"You lost the auction. Lost by: {result['margin']:.2f}")

        # Display metrics after each auction
        metrics = simulator.get_metrics()
        print("Metrics:", metrics)
