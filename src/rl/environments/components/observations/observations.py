import numpy as np
from gymnasium.spaces import Box
from src.rl.libs.utils import available_strategy
from .observation_01 import get_dual_ma_obs
from .observation_02 import get_donchian_obs

def validate_obs_name(observation_name: str):
    if observation_name in available_strategy:
        print(observation_name)
        return
    
    raise ValueError("Wrong obs config name")

class ObservationProvider():
    def __init__(self, observation_type: str) -> None:
        validate_obs_name(observation_type)
        self.type = observation_type
        
    def get_observation(self, params: any):
        if self.type == available_strategy[0]:
            return get_dual_ma_obs(**params)

        if self.type == available_strategy[1]:
            return get_donchian_obs(**params)

        raise ValueError("Cannot get observation type")
    
    def get_observation_space(self):
        if self.type == available_strategy[0]:
            return Box(low=0, high=1, shape=(5,), dtype=np.float32)
        
        if self.type == available_strategy[1]:
            return Box(low=0, high=1, shape=(9,), dtype=np.float32)
        
        raise ValueError("Cannot found observation space type")

    