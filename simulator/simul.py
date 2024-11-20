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
import environment.env as env

env.setup()

class AuctionSimulator:
    """
    Contributor: Hamza

    This class implements the simulation of a second-price auction environment.
    It manages budget tracking, impression generation, bidding logic, and performance metrics.
    """

    def __init__(self, initial_budget, keyword_list, done):
        """
        Initialize the auction simulator with competitors and tracking metrics.

        Parameters:

        """
   
        self.initial_budget = initial_budget
        self.remaining_budget = initial_budget
        self.num_wins = 0
        self.total_auctions = 0
        self.desired_keywords = keyword_list  # User-defined keywords for tracking.
        self.done = False
        
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
    
    def win_update_budget(self, remaining_budget, bid):
        """
        Contributor: Hamza

        Update the agent's remaining budget after each auction.

        Parameters:
        - remaining_budget (float): The agent's remaining budget.
        - bid (float): The bid amount for the current auction.
        """
        self.remaining_budget = remaining_budget - bid
    
    def win_count_update(self):
        """
        Contributor: Hamza

        Update the agent's win count after each auction.

        Parameters:
        - win_count (int): The number of auctions won by the agent.
        """
        self.num_wins += 1

    def total_auctions_update(self):
        """
        Contributor: Hamza

        Update the total number of auctions after each auction.
        """
        self.total_auctions += 1

    def get_rank(self, keyword):
        """
        Contributor: Hamza

        Get the rank of a keyword in the list of desired keywords.

        Parameters:
        - keyword (str): The keyword to search for.

        Returns:
        - int: The rank of the keyword in the list of desired keywords.
        """
        if keyword:
            return self.desired_keywords.index(keyword) + 1   
        else:
            return 0
    
    def is_terminal(self,done):
        """
        Contributor: Hamza

        Check if the agent has exhausted its budget or completed all auctions.

        Returns:
        - bool: True if the agent has no budget or completed all auctions, False otherwise.
        """
        done = self.remaining_budget <= 50 or self.total_auctions == 10000 
        return done
    

    def run_auction_step(self, bid_bool, keyword, bid_amount, done):
        """
        Contributor: Hamza

        The agent wins if its bid is the highest and pays the second-highest bid.

        Returns:
        - dict: Results of the auction including win status, cost, margin, and impression details.
        """
        done = self.is_terminal(self, done)
        if not done:
            if bid_bool:
                bid_result, cost = env.step(bid_bool, keyword, bid_amount)
                if bid_result:
                    self.win_count_update()
                    self.win_update_budget(self.remaining_budget, bid_amount)
                    self.total_auctions_update()
                    return {"win": True, "cost": cost, "margin": bid_amount - cost, "remaining_budget": self.remaining_budget, "rank": self.get_rank(keyword) , "bid_amount": bid_amount, "done": done}
                else:
                    self.total_auctions_update()
                    return {"win": False, "cost": None, "margin": bid_amount - cost, "remaining_budget": self.remaining_budget, "rank": self.get_rank(keyword), "bid_amount": bid_amount, "done": done}
            else:
                env.step(bid_bool)
        

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
