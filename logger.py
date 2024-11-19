import logging
import os
from datetime import datetime
from typing import Dict


class Logger:
    def __init__(self, log_dir: str = "logs"):
        """Initialize logger with timestamp-based files."""
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Main logger setup
        self.logger = logging.getLogger('AdBidding')
        self.logger.setLevel(logging.INFO)

        # File handler for all logs
        file_handler = logging.FileHandler(
            f"{log_dir}/ad_bidding_{timestamp}.log"
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(console_handler)

        # Metrics logger setup
        self.metrics_logger = logging.getLogger('Metrics')
        self.metrics_logger.setLevel(logging.INFO)
        metrics_handler = logging.FileHandler(
            f"{log_dir}/metrics_{timestamp}.csv"
        )
        metrics_handler.setFormatter(
            logging.Formatter('%(message)s')
        )
        self.metrics_logger.addHandler(metrics_handler)

        # Log headers for metrics CSV
        self.metrics_logger.info(
            "episode,win_rate,remaining_budget,total_wins,avg_bid,total_reward")

    def log_info(self, message: str):
        """Log general information."""
        self.logger.info(message)

    def log_metrics(self, episode: int, metrics: Dict[str, float]):
        """Log episode metrics to CSV."""
        self.metrics_logger.info(
            f"{episode},{metrics['win_rate']:.4f},{metrics['remaining_budget']:.2f},"
            f"{metrics['wins']},{metrics['avg_bid']:.2f},{metrics.get('total_reward', 0):.2f}"
        )