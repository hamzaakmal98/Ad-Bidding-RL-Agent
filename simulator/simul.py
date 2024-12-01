"""
Contributor: Hamza, Weijia

This module simulates a second-price auction environment for ad bidding.
The simulation includes:
1. Generating impressions with attributes like keywords, page content, user profile, and pCTR.
2. Collecting bids from competitors and the agent.
3. Running a second-price auction to determine winners.
4. Tracking the agent's performance metrics like budget, number of wins, and win rate.
"""


import environment.env as env
import rewards.rewards_functions as rewards

env.setup()

class AuctionSimulator:
    """
    This class implements the simulation of a second-price auction environment.
    It manages budget tracking, impression generation, bidding logic, and performance metrics.
    """

    def __init__(self, initial_budget, keyword_list, max_rounds=1000, budget_lower_limit=50):
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
        self.max_rounds = max_rounds
        self.budget_lower_limit = budget_lower_limit
        
    def prompt_keywords(self):
        """
        Prompt the user to enter three desired keywords to track in the auction environment.
        These keywords are stored in a list and used for impression generation.
        """
        print("Enter 3 desired keywords for tracking:")
        for i in range(3):
            keyword = input(f"Keyword {i + 1}: ").strip()
            self.desired_keywords.append(keyword)
        print(f"Tracked Keywords: {self.desired_keywords}")

    
    def win_update(self, bid):
        """
        Update the agent's win count and remaining budget after each auction won.

        Parameters:
        - bid (float): The bid amount for the current auction.
        """
        self.num_wins += 1
        self.remaining_budget -= bid


    def get_rank(self, keyword):
        """
        Get the rank of a keyword in the list of desired keywords - used in reward calculation

        :param keyword: str
            keyword agent chose. note: keyword is None if agent doesn't bid

        :return: int - rank based on index of the keyword in the list of desired keywords.
                note: returns 0 if keyword not in desired keywords list, otherwise returns int>0
        """

        if keyword in self.desired_keywords:
            return self.desired_keywords.index(keyword) + 1   
        else:
            return 0
    
    def is_terminal(self):
        """
        Checks if the agent has exhausted its budget or reached max number of allowed auction rounds.

        :params max_rounds: int
            max rounds at which we should end the episode even if budget limit not reached

        :params budget_lower_limit: int
            should end the episode when the budget has reached this value

        :return: boolean - True if the agent has no budget or completed all auctions, False otherwise.
        """
        return self.remaining_budget <= self.budget_lower_limit or self.total_auctions >= self.max_rounds


    def get_all_ad_keywords(self):
        return env.KEYWORDS

    def get_current_available_keywords(self):
        return env.available_keywords
    

    def run_auction_step(self, bid_bool, keyword, bid_amount, verbose=False):
        """
        Runs a single bid opportunity - acts as simulator step function.

        :param bid_bool: boolean
            boolean reflecting if the agent want to bid (True = yes, False = no)

        :param keyword: str
            which keyword the agent chooses to bid on

        :param bid_amount: float or int
            how much the agent bids on the chosen keyword

        :return: 
        - (1) observation: a list of numbers that includes informations of current state, including 
                           what keywords out of all keywords are currently available to bid for,
                           and the current budget left
        - (2) reward: the reward resulted from this step
        - (3) done: whether the budget ran out and the user can't bid anymore
        - (4) info_dict: {
                "win": whether the user have won the current auction, 
                "cost": cost that the user need to pay (if won, this would be the second highest bid price, if lost, then 0), 
                "margin": the bid price that the user placed - cost,
                "highest_competitor_bid": the highest competitor bid, will be 0 if the user choose to not to bid 
                "remaining_budget": the amount of budget left after the bid was made,
                "rank": keyword ranking, lower rank means the keyword has a lower priority (ex: if 3 keywords, rank 3 = highest priority),
                "bid_amount": the bid price that the user placed, 
                "total_auctions": how many auctions have been run so far including this most recent one}
        """
        # check if we reached a terminal state
        if not self.done:
            available_keywords = env.get_available_keywords()
            bid_result, highest_competitor_bid = env.step(bid_bool, keyword, bid_amount)

            # case where agent places a bid
            if bid_bool:
                
                # case where agent won the current bid
                if bid_result:
                    self.win_update(bid_amount)
                    amount_to_pay = highest_competitor_bid

                # case where agent lost the current bid
                else:
                    amount_to_pay = 0   # If a bid is lost, the user does not need to pay anything

                margin = bid_amount - highest_competitor_bid

            # case where agent does NOT place a bid
            # note: this is still an action, so still need to update num auctions and return values
            else:
                amount_to_pay = 0
                margin = 0
                bid_amount = 0

            observation = self.get_observation_space()
            info_dict = {
                "bid": bid_bool,
                "choosen_keyword_available": keyword in available_keywords,
                "win": bid_result, 
                "cost": amount_to_pay, 
                "margin": margin, 
                "highest_competitor_bid": highest_competitor_bid,
                "remaining_budget": self.remaining_budget, 
                "rank": self.get_rank(keyword),
                "bid_amount": bid_amount, 
                "total_auctions": self.total_auctions,
                # A dict recording if there are other high rank keywords currently available and 
                # isn't the current chosen keyword. Keys are those unselected high rank keywords and 
                # values are their corresponding rank
                "other_high_rank_keywords_available": {     
                    k:self.get_rank(k) for k in self.desired_keywords if k != keyword and k in available_keywords
                }
            }
            # print('keyword = ', keyword)
            # print('env.get_available_keywords() = ', available_keywords)
            # print([k for k in self.desired_keywords if k != keyword and k in available_keywords])
            # print(info_dict["other_high_rank_keywords_available"])
            reward = rewards.calculate_reward(info_dict, self.initial_budget, verbose=verbose)
            self.reward_list.append(reward)
            self.total_auctions += 1
            self.done = self.is_terminal()

            return observation, reward, self.done, info_dict


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
            "Cumulative Rewards so far": rewards.aggregate_rewards(self.reward_list)
        }


    def reset(self):
        """
        Resets the auction and returns the current simulator observation and a dictionary of additional informations.
        """
        self.remaining_budget = self.initial_budget
        self.num_wins = 0
        self.total_auctions = 0
        self.done = False
        self.reward_list=[]

        observation = self.get_observation_space()
        info_dict =  {"win": False, "cost": None, "margin": 0,
                      "remaining_budget": self.remaining_budget, 
                      "total_auctions": self.total_auctions}

        return observation, info_dict


    def get_observation_space(self):
        """
        Returns the observation space of the environment, including a list of 0s and 1s 
        to indicate which keywords are currently available for bidding, and
        the current budget.
        """
        available_keywords_set = set(env.available_keywords)
        available_keywords_binary = [1 if keyword in available_keywords_set else 0 for keyword in env.KEYWORDS]
        return available_keywords_binary + [self.remaining_budget]
    

    def get_observation_space_dim(self):
        """
        Returns the dimensionality of the observation space of the environment.
        Note: the return value must be consistent with the return value from the get_observation_space() method.
        """
        return len(env.KEYWORDS) + 1
    

    def get_action_space_dim(self):
        """
        Returns the dimensionality of the action space of the environment.
        """
        return len(env.KEYWORDS) + 1    # The agent can choose to either bid for any one of the keyword or not bid at all
    
