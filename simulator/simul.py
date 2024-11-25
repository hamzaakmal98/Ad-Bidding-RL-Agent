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


import environment.env as env
import rewards.rewards_functions as rewards
from environment.env import KEYWORDS

env.setup()

class AuctionSimulator:
    """
    Contributor: Hamza

    This class implements the simulation of a second-price auction environment.
    It manages budget tracking, impression generation, bidding logic, and performance metrics.
    """

    def __init__(self, initial_budget, keyword_list):
        """
        Initialize the auction simulator episode with initial budget and 3 desired priority keywords.

        Parameters:
            initial_budget - constant defined by user,
            keyword_list - list of 3 user-defined priority keywords


        """
   
        self.initial_budget = initial_budget
        self.remaining_budget = initial_budget
        self.num_wins = 0
        self.total_auctions = 0
        self.desired_keywords = keyword_list  # User-defined keywords for tracking.
        self.done = False
        self.reward_list=[]
        
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
       Get the rank of a keyword in the list of desired keywords - used in reward calculation

        :param keyword: str
            keyword agent chose. note: keyword is None if agent doesn't bid

        :return: int - rank based on index of the keyword in the list of desired keywords.
                note: returns 0 if keyword not in desired keywords list, otherwise returns int>0
        """

        if keyword:
            return self.desired_keywords.index(keyword) + 1   
        else:
            return 0
    
    def is_terminal(self,max_rounds=1000, budget_lower_limit=50):
        """
        Checks if the agent has exhausted its budget or reached max number of allowed auction rounds.

        :params max_rounds: int
            max rounds at which we should end the episode even if budget limit not reached

        :params budget_lower_limit: int
            should end the episode when the budget has reached this value

        :return: boolean - True if the agent has no budget or completed all auctions, False otherwise.
        """
        done = self.remaining_budget <= budget_lower_limit or self.total_auctions >= max_rounds
        return done


    def get_all_possible_keywords_in_env(self):
        return env.KEYWORDS

    def get_current_available_keywords(self):
        return env.available_keywords()
    

    def run_auction_step(self, bid_bool, keyword, bid_amount):
        """
        Runs a single bid opportunity - acts as simulator step function.

        :param bid_bool: boolean
            boolean reflecting if the agent want to bid (True = yes, False = no)

        :param keyword: str
            which keyword the agent chooses to bid on

        :param bid_amount: float or int
            how much the agent bids on the chosen keyword

        :return: dictionary and float
        - (1) dictionary: {"win": boolean that is true when , "cost": cost of bid, "margin": bid_amount - cost,
                "remaining_budget": amount of budget left after bid was made,
                "rank": keyword ranking (lower rank = lower priority kw (ex: if 3 keywords, rank 3 = highest priority),
                 "bid_amount": how much the agent had bid, "done": boolean that is true if termination is reached,
                 "total_auctions": how many auctions have been run so far including this most recent one}
        - (2) reward value (float)

        """

        done = self.is_terminal(self)



        # check if we reached a terminal state
        if not done:

            # case where agent places a bid
            if bid_bool:
                bid_result, cost = env.step(bid_bool, keyword, bid_amount)

                # case where agent won after placing bid
                if bid_result == True:

                    self.win_count_update()
                    self.win_update_budget(self.remaining_budget, bid_amount)
                    self.total_auctions_update()

                    simulator_dict = {"win": True, "cost": cost, "margin": bid_amount - cost,
                                      "remaining_budget": self.remaining_budget, "rank": self.get_rank(keyword) ,
                                      "bid_amount": bid_amount, "done": done,"total_auctions":self.total_auctions }

                    return simulator_dict, rewards.calculate_reward(simulator_dict,self.initial_budget)

                # case where agent lost after placing bid
                else:

                    self.total_auctions_update()

                    simulator_dict =  {"win": False, "cost": None, "margin": bid_amount - cost,
                                       "remaining_budget": self.remaining_budget, "rank": self.get_rank(keyword),
                                       "bid_amount": bid_amount, "done": done,"total_auctions":self.total_auctions}

                    this_reward = rewards.calculate_reward(simulator_dict,self.initial_budget)
                    self.reward_list.append(this_reward)

                    return simulator_dict, this_reward

            # case where agent does NOT place a bid
            # note: this is still an action, so still need to update num auctions and return values
            else:

                bid_result, cost = env.step(bid_bool) # this will always be the following: bid_result = False, and cost  = 0.0
                self.total_auctions_update()

                simulator_dict = {"win": False, "cost": cost, "margin": 0 - cost,
                                  "remaining_budget": self.remaining_budget, "rank": self.get_rank(keyword),
                                  "bid_amount": 0, "done": done, "total_auctions": self.total_auctions}

                this_reward = rewards.calculate_reward(simulator_dict, self.initial_budget)
                self.reward_list.append(this_reward)

                return simulator_dict, this_reward

        

    def get_metrics(self):

        """
        Computes and returns dictionary regarding budget, wins, auction count, and rewards:
            keys in dictionary: "Remaining Budget", "Wins", "Total Auctions", "Win Rate", "Cumulative Rewards"
        """

        win_rate = self.num_wins / self.total_auctions if self.total_auctions > 0 else 0
        return {
            "Remaining Budget": self.remaining_budget,
            "Wins": self.num_wins,
            "Total Auctions": self.total_auctions,
            "Win Rate": win_rate,
            "Cumulative Rewards so far":rewards.aggregate_rewards(self.reward_list)
        }


    def reset(self):
        """
        Contributor: Weijia

        Resets the auction.
        """
        self.remaining_budget = self.initial_budget
        self.num_wins = 0
        self.total_auctions = 0
        self.done = False
        self.reward_list=[]


    def get_observation_space(self):
        """
        Contributor: Weijia

        Returns the observation space of the environment, including a list of 0s and 1s 
        to indicate which keywords are currently available for bidding, and
        the current budget.
        """

        return 