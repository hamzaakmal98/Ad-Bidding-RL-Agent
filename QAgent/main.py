from simulator.simul import AuctionSimulator
from QAgent.QAgent import QAgent
import numpy as np
import pickle


totalWinsTillNowTraining = 0
totalAuctionsTillNowTraining = 0

totalWinsTillNowTesting = 0
totalAuctionsTillNowTesting = 0

def trainmetrics(simulMetrics):
    global totalWinsTillNowTraining, totalAuctionsTillNowTraining 
    totalWinsTillNowTraining = totalWinsTillNowTraining + simulMetrics.get('Wins')
    totalAuctionsTillNowTraining += simulMetrics.get('Total Auctions')
    return {'totalWinsTillNow': totalWinsTillNowTraining, 'totalAuctionsTillNow': totalAuctionsTillNowTraining, 'Win Rate': totalWinsTillNowTraining/totalAuctionsTillNowTraining}

def testmetrics(simulMetrics):
    global totalWinsTillNowTesting, totalAuctionsTillNowTesting 
    totalWinsTillNowTesting = totalWinsTillNowTesting + simulMetrics.get('Wins')
    totalAuctionsTillNowTesting += simulMetrics.get('Total Auctions')
    return {'totalWinsTillNow': totalWinsTillNowTesting, 'totalAuctionsTillNow': totalAuctionsTillNowTesting, 'Win Rate': totalWinsTillNowTesting/totalAuctionsTillNowTesting}

def hash_state(remaining_budget, available_keywords):
    """
    Hashes the state {remaining_budget, available_keywords} for efficient storage in the Q-table.
    The hash ensures uniqueness and minimizes collisions.
    """
    # keywords_hash = sum(hash(keyword) for keyword in available_keywords)
    keywords_hash = hash(available_keywords)
    return remaining_budget * (10**6) + keywords_hash

def save_q_table(q_table, filepath="Q_table.pickle"):
    """
    Save the Q-table to a file.
    """
    with open(filepath, 'wb') as handle:
        pickle.dump(q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_q_table(filepath="Q_table.pickle"):
    """
    Load the Q-table from a file.
    """
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)

def give_bidding_keyword(priority_keywords, available_keywords):
    for keyword in priority_keywords:
        if keyword in available_keywords:
            return keyword
    return None

def main_train(simulator, priority_keywords, num_episodes=1000):
    """
    Train the agent using Q-learning over multiple episodes.

    Parameters:
        simulator (AuctionSimulator): The auction simulator environment.
        priority_keywords (list): List of keywords the agent cares about.
        num_episodes (int): Number of training episodes.
    """
    agent = QAgent(priority_keywords=priority_keywords, num_actions=5)

    for episode in range(num_episodes):
        simulator.remaining_budget = simulator.initial_budget
        simulator.num_wins = 0
        simulator.total_auctions = 0
        simulator.reward_list = []
        
        available_keywords = simulator.get_current_available_keywords()
        bidding_keyword = give_bidding_keyword(priority_keywords, available_keywords)

        while not simulator.is_terminal():            

            # If no desired keyword is available, skip this round
            if bidding_keyword is None:
                simulator.run_auction_step(False, None, None)
                available_keywords = simulator.get_current_available_keywords()
                bidding_keyword = give_bidding_keyword(priority_keywords, available_keywords)
                continue
       
            # Define current state
            current_state = hash_state(remaining_budget = simulator.remaining_budget, available_keywords = bidding_keyword)

            # Choose an action (exploration or exploitation)
            action_idx = agent.choose_action(current_state)

            # Calculate the bid amount
            bid_amount = agent.calculate_bid(bidding_keyword, action_idx)

            # Perform the auction step in the simulator
            simulator_dict, reward = simulator.run_auction_step(True, bidding_keyword, bid_amount)

            available_keywords = simulator.get_current_available_keywords()
            bidding_keyword = give_bidding_keyword(priority_keywords, available_keywords)

            # Define next state
            next_state = hash_state(remaining_budget = simulator.remaining_budget, available_keywords = bidding_keyword)

            # Update Q-table
            agent.update_q_table(current_state, action_idx, reward, next_state)

        # Decay epsilon after each episode
        agent.decay_epsilon()
        agent.bids = {keyword: 50.0 for keyword in priority_keywords}

        # Print metrics periodically
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}: Metrics - {trainmetrics(simulator.get_metrics())}")
        else:
            trainmetrics(simulator.get_metrics())
    save_q_table(agent.q_table, 'Q_table.pickle')
    return agent

def test_agent(agent, simulator, num_episodes=1000):
    """
    Test the trained agent on the auction simulator.

    Parameters:
        agent (Agent): The trained agent.
        simulator (AuctionSimulator): The auction simulator environment.
        num_episodes (int): Number of testing episodes.
    """
    q_table = load_q_table('Q_table.pickle')
    for episode in range(num_episodes):
        simulator.remaining_budget = simulator.initial_budget
        simulator.num_wins = 0
        simulator.total_auctions = 0
        simulator.reward_list = []

        while not simulator.is_terminal():
            # Get available keywords
            available_keywords = simulator.get_current_available_keywords()
            # print(available_keywords)
            # Check if any priority keyword is available
            bidding_keyword = None
            for keyword in agent.priority_keywords:
                if keyword in available_keywords:
                    bidding_keyword = keyword
                    break

            # If no desired keyword is available, skip this round
            if bidding_keyword is None:
                simulator.run_auction_step(False, None, None)
                continue
            # print(bidding_keyword)
            # Define current state
            current_state = hash_state(simulator.remaining_budget, bidding_keyword)

            # Choose the best action (exploit)
            action_idx = np.argmax(q_table.get(current_state, np.zeros(agent.num_actions)))

            # Calculate the bid amount
            bid_amount = agent.calculate_bid(bidding_keyword, action_idx)
            # print('bidding amt: ', bid_amount)
            # Perform the auction step in the simulator
            simulator_dict, reward = simulator.run_auction_step(True, bidding_keyword, bid_amount)
            # print(simulator_dict.get('win'))
            # print(simulator_dict.get('cost'))

        # Print metrics periodically
        if (episode + 1) % 1000 == 0:
            print(f"Test Episode {episode + 1}: Metrics - {testmetrics(simulator.get_metrics())}")
        else:
            testmetrics(simulator.get_metrics())


initial_budget = 10000
priority_keywords = ["A", "B", "C"]
simulator = AuctionSimulator(initial_budget, priority_keywords)
trained_agent = main_train(simulator, priority_keywords, num_episodes=10000)
test_agent(trained_agent, simulator)