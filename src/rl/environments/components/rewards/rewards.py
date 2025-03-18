from . import reward_01 as reward_01
from . import reward_02 as reward_02
from src.rl.libs.utils import available_strategy

def checkRewardsName(reward_name: str):
    if reward_name in available_strategy:
        return
    
    raise ValueError("Wrong reward config name")

class RewardCounter:
    def __init__(self, reward_type: str):
        self.type = reward_type
        self.total_reward = 0

    def reset(self):
        self.total_reward = 0

    def calculate_post_action(self, params: any ):
        reward = 0
        if self.type == available_strategy[0]:
            reward = reward_01.get_post_action_reward(**params)
            self.total_reward += reward
            return reward
            
        if self.type == available_strategy[1]:
            reward = reward_02.get_post_action_reward(**params)
            self.total_reward += reward
            return reward
        
        raise ValueError("Reward type unimplemented")

    def calculate_post_update(self, params: any):
        if self.type == available_strategy[0]:
            reward = reward_01.get_post_update_reward(**params)
            self.total_reward += reward
            return reward

        if self.type == available_strategy[1]:
            reward = reward_02.get_post_update_reward(**params)
            self.total_reward += reward
            return reward
    
        raise ValueError("Reward type unimplemented")