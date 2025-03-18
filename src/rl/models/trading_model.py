
from typing import Dict
from stable_baselines3.common.type_aliases import GymEnv
from .rl_algo import getRLAlgo, getRLType

class TradingModel:
    def __init__(self, 
        env: GymEnv,
        test_name: str,
        model_config: Dict,
    ):
        self.model_name = test_name
        self.model_type = getRLType(model_config["rl_algo"])
        self.model = getRLAlgo(
            env, 
            test_name, 
            model_config
        )

    def train(self, 
        total_timesteps: int,
    ):
        self.model.learn(
            total_timesteps=total_timesteps, 
            reset_num_timesteps=False,
            tb_log_name=self.model_name
        )

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str, env: GymEnv):
        self.model = self.model_type.load(path, env=env)