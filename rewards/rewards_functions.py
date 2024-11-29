'''

   Overall goal of rewards:

   Maximize number of impressions (intuitively, this goes hand in hand with winning high number auctions for important keywords)
   while penalizing expensive short-term spending to ensure responsible budget allocation

   (note: we don't know if, necessarily, we're optimizing the budget itself -
          so used the word responsible instead of optimal here)

   (note: a numerical reward is only given if we win the given auction. Otherwise 0)

   (don't want the agent to just maximize winning all opportunitiest with high bids


'''

def calculate_reward(variables_dict,initial_budget,max_budget_consumption_per_auction = 0.5, stop_penalty_percent = 0.25, stop_penalty_decay = 0.0):
    '''
    Calculates reward based on auction variables and penalizes for over-consuming budget or overbidding

    :param variables_dict:
        dictionary passed from simulation after auction complete - should include:
            bid amount, bid cost amount, budget left,
            keyword rank, win boolean (did agent win the auction or not) and bid margin (signed difference b/w bid and cost)

    :param initial_budget:
        Initial budget passed for episode (ex: $1000)

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

    original_budget = initial_budget
    bid_placed = variables_dict["bid_amount"]
    bid_cost = variables_dict["cost"]
    budget_left = variables_dict["remaining_budget"]
    keyword_rank = variables_dict["rank"] # note: lower rank = lower priority kw (ex: if 3 keywords, rank 3 = highest priority)
    margin = variables_dict["margin"]
    won_boolean = variables_dict["win"]

    # - non priority keywords marked with 0, priority keywords marked with int>0
    priority_keyword = keyword_rank != 0 # True if keyword is one of the 3 priority keywords, False otherwise

    keyword_importance = 10 # set baseline non priority kw value
    if priority_keyword:
        keyword_importance *= (keyword_rank**2)

    # no impressions won since agent lost auction or didn't participate in the auction (no bid),
    # so no reward unless it's an auction we don't mind losing
    if not won_boolean:

        # case where it's an auction we don't mind losing
        if not priority_keyword: #if not a priorty keyword, then it's okay that we didn't win and we don't want to penalize
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
        diff_bid = abs(margin)

        # used too much of the budget
        if bid_placed > budget_consumption_max:
            # print(f"You used {bid_placed} which is > {budget_consumption_max} AKA too much of the budget!")
            penalty = penalty + 0.2

        # overbid by greater than a half of cost - how should/should I parameterize this?
        if bid_placed >= 1.5 * bid_cost:
            # print("You overbid by greater than a 1.5 times the cost!")
            penalty = penalty + 0.2

        # reduce total penalty amount if little budget left
        percent_budget_left = budget_left / original_budget

        # when there is "stop_penalty_percent" amount of budget left we want to stop (or decay) penalizing overbidding and budget
        # since we aren't as worried about overspending in the initial stages
        # --> for example: if stop_penalty_percent = 0.1, this means when there is <= 10% of the budget left
        #                  we reduce penalty
        if percent_budget_left <= stop_penalty_percent:
            # print(f"Overall penalty would have been: {penalty}, but since <= {stop_penalty_percent} budget left...")
            penalty = penalty * stop_penalty_decay
            # print(f"due tp applying the stop penalty decay, overall penalty is {penalty} instead.")


        # if no penalty, reward is just the bid impressions (value straight from keyword importance)
        # print(f'Therefore your total penalty percent is: {penalty}')

        # note: +1 since diff_bid = 1 when cost and keyword importance are within 1 "dollar" of each other
        # --> we can consider the bid and cost being within margin of 1 an ideal situation so don't want to penalize
        reward = (keyword_importance - diff_bid + 1) - (keyword_importance * penalty)

        # print(f'Your final reward is {reward}')

        return reward

def aggregate_rewards(episode_rewards) :
    '''Aggregates rewards for an entire episode where param "episode_rewards" is a list '''
    return sum(episode_rewards)

