import yaml
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_broker_config(self, env_config_name: str) -> Dict[str, Any]:
        if env_config_name not in self.config['broker']:
            raise ValueError(f"Wrong env config name. Available: {self.config['broker']}")
        
        index = self.config['broker'].index(env_config_name)
        return {key: value[index] for key, value in self.config['broker_configs'].items()}

    def get_price_data_config(self, config_name: str) -> Dict[str, Any]:
        if config_name not in self.config['price']:
            raise ValueError(f"Wrong env config name. Available: {self.config['price']}")
        
        return self.config['price_configs'][config_name]