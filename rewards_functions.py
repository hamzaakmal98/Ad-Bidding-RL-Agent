'''

   Overall goal of rewards:

   Maximize number of impressions (intuitively, this goes hand in hand with winning high number auctions for important keywords)
   while penalizing expensive short-term spending to ensure responsible budget allocation

   (note: we don't know if, necessarily, we're optimizing the budget itself -
          so used the word responsible instead of optimal here)

   (note: a numerical reward is only given if we win the given auction. Otherwise 0)

   (don't want the agent to just maximize winning all opportunitiest with high bids


'''

# var should be imported from environment or simulation file - temporarily set here
global original_budget
original_budget = 1000  # replace with import statement


def calculate_reward(state,max_budget_consumption_per_auction = 0.25, stop_penalty_percent = 0.5, stop_penalty_decay = 0.0):
    '''
    :param state: object?
        State returned from simulation after auction complete - should include:
            bid amount, bid cost amount, budget left,
            keyword importance (associated number impressions expected)

    :param max_budget_consumption_per_auction: float in [0,1]
        If agent consumed greater than this percent of the budget, agent is penalized.

    :param stop_penalty_percent: float in [0,1]
        when this percent of the original budget is left, we no longer penalize the agent for high budget consumption.

    :param stop_penalty_decay: float in [0,1]
        when this >0, and the stop_penalty_percent is reached, we decay the penalization amount by this percent.
        when this is 0, the penalty is completely removed once the stop_penalty_percent is reached.
        when this is = 1, the penalty is never reduced even if the stop_penalty_percent is reached.


    :return: reward amount
    '''
    global original_budget

    # vars below will be pulled from state info -- from environment or simulation using step function?:
    bid_placed = float(input("bid placed: "))  # replace this with returned amount the agent just bid from current state
    bid_cost = float(input("bid_cost: "))  # replace this with returned bid cost from current state
    budget_left = float(input("budget_left: ") ) # replace this with returned updated budget from current state
    keyword_importance = float(input("keyword_importance: ") ) # from environment or simulation?

    priority_keyword = input("Is this a priority keyword (True / False): ")  # from environment or simulation?

    # did agent win or not
    won_boolean = bid_cost < bid_placed
    print('auction result:' ,won_boolean)

    # no impressions won since lost auction, so no reward unless it's an auction we don't mind losing
    if won_boolean == False:

        # case where it's an auction we don't mind losing
        if priority_keyword.lower() == "false": #if not a priorty keyword, then it's okay that we didn't win and we don't want to penalize
            return keyword_importance # just go ahead and return the reward (skip computation below)

        # case where we lost and we aren't happy about it
        return 0

    else:
        # agent won the auction of a keyword we care about
        # -- need to set reward based on proportion of budget used and the diff between cost and bid placed
        # -- aka, reward should reflect how did the agent won
        #           --> over-consumed budget in order to win/beat the cost?
        #           --> overbid when the cost was low?

        # get the maximum amount of budget allowed to have been spent based on budget for this simulation
        budget_consumption_max = max_budget_consumption_per_auction * (budget_left + bid_cost)


        #penalty will be the percent we reduce the reward by
        penalty = 0
        diff_bid = abs(bid_placed-bid_cost)

        # used too much of the budget
        if bid_placed > budget_consumption_max:
            print(f"used {bid_placed} which is > {budget_consumption_max} AKA too much of the budget!")
            penalty = penalty + 0.2

        # overbid by greater than a half of cost - how should/should I parameterize this?
        if bid_placed >= 1.5 * bid_cost:
            print("overbid by greater than a half of cost!")
            penalty = penalty + 0.2

        # reduce total penalty amount if little budget left
        percent_budget_left = budget_left / original_budget
        print("percent budget left:",percent_budget_left)

        # when there is "stop_penalty_percent" amount of budget left we want to stop (or decay) penalizing overbidding and budget
        # since we aren't as worried about overspending in the initial stages
        # --> for example: if stop_penalty_percent = 0.1, this means when there is <= 10% of the budget left
        #                  we reduce penalty
        if percent_budget_left <= stop_penalty_percent:
            print(f"penalty would have been: {penalty}, but since <= {stop_penalty_percent} budget left...")
            penalty = penalty * stop_penalty_decay
            print(f"applying stop penalty decay value results in {penalty} penalty ")


        # if no penalty, reward is just the bid impressions (value straight from keyword importance)
        print(f'total penalty: {penalty}')

        # note: +1 since diff_bid = 1 when cost and keyword importance are within 1 "dollar" of each other
        # --> we can consider the bid and cost being within margin of 1 an ideal situation so don't want to penalize
        reward = (keyword_importance - diff_bid + 1) - (keyword_importance * penalty)

        return reward

def aggregate_rewards(episode_rewards) :
    '''Aggregates rewards for an entire episode where param "episode_rewards" is a list '''
    return sum(episode_rewards)


def main():
    state = {} # to be replaced with some state

    #see screenshots in github for examples of each case (overbudget, overspend, non priority kw, priority kw, etc.)

    #when stop_penalty decay = 0, when there 10% budget left of the original budget,
    # --> stop penalizing overbidding/allow using a lot of budget
    print(calculate_reward(state, 0.25, 0.1, stop_penalty_decay=0))

    #when stop_penalty decay = 1, when there 10% budget left of the original budget,
    # --> reduce overall penalty by 50% for overbidding/allow using a lot of budget
    print(calculate_reward(state, 0.25, 0.1, stop_penalty_decay=0.5))

    #when stop_penalty decay = 1, when there 10% budget left of the original budget,
    # --> keep penalizing as normal for overbidding/allow using a lot of budget
    print(calculate_reward(state, 0.25, 0.1, stop_penalty_decay=1))




main()