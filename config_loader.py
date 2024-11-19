import yaml
from typing import Dict, Any


class ConfigLoader:
    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent-specific configuration."""
        return self.config['agent']

    def get_env_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        return self.config['environment']

    def get_simulator_config(self) -> Dict[str, Any]:
        """Get simulator-specific configuration."""
        return self.config['simulator']

    def get_rewards_config(self) -> Dict[str, Any]:
        """Get rewards-specific configuration."""
        return self.config['rewards']