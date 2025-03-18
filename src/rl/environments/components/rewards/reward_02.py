import math

from typing import TypedDict
from src.rl.libs.utils import OrderInfo
from .reward_01 import get_reward_on_reentry, get_distance_given_mark
from ..observations import get_donchian_obs_dict, DonchianObsDict

def get_comission_cost(info: OrderInfo):
    return math.log10(1 - 0.003) * info['size']

def get_reward_entry(info: OrderInfo, pos: float):
    reward = 0

    if info['order'] == 'buy':
        reward = 0.3 - pos

    if info['order'] == 'sell':
        reward = pos - 0.5

    return reward * info['size'] * 0.01

def get_reward_from_delta(info: OrderInfo, delta: float):
    # multiplier = 1
    # reward = (delta * multiplier)
    # delta = 1 + reward 
    # deltaMa = math.log10(delta / 1)  * info['size']
    return delta * info['size']

def get_reward_on_reentry(info: OrderInfo, deltaSma: float):
    # the delta should be negative for buying
 
    return -deltaSma

class ParamsPostActionDonchian(TypedDict):
    action: int
    info: OrderInfo 
    inventory: int 
    obs: DonchianObsDict
    # pos_in_donchian: float
    # bottom_shift: bool
    # top_shift: bool
    # mark_price: float
    # avg_position: float
    # lowest_position: float
    # last_buy_position: float

class ParamsPostUpdateDonchian(TypedDict):
    ma_current: float 
    ma_before: float
    i_post_action: int


def get_post_update_reward( 
        ma_current: float, 
        ma_before: float,
        i_post_action: int,
        ):
    # Calculate log difference of moving average
    log_val = math.log10(ma_current/ma_before)

    multiplier = i_post_action 
    reward = log_val * multiplier * 2

    if i_post_action == 0:
        reward = int(0)

    return reward

def get_post_action_reward(
        action: int,
        info: OrderInfo, 
        inventory: int, 
        obs: DonchianObsDict,
        # pos_in_donchian: float,
        # bottom_shift: bool,
        # top_shift: bool, 
        # mark_price: float,
        # avg_position: float,
        # lowest_position: float,
        # last_buy_position: float,
    ):

    # 
    bottom_shift = obs['lower_shift_5'] == 1
    top_shift = obs['upper_shift_5'] == 1

    # reward config
    reward = 0.0 # base reward
    weight = 1
    penalty = 0.001

    # get distance from average position
    if info['order'] == 'buy' or info['order'] == 'sell':    
        reward += get_comission_cost(info)
        inventory  = abs(inventory)
        
        # BUYING
        ## if buying
        if inventory >= 1:
            # if entry
            if inventory == 1:
                # gets more reward as it entry near lower channel
                # if top_shift:
                #     reward += 0.0001
                reward += get_reward_entry(info, obs['position_inside_channel'])

            ## if averaging down
            else:
                # gets reward on lower channel + distance from last entry
                # reward += get_reward_entry(info, pos_in_donchian)
                reward += get_reward_on_reentry(info, obs['distance_from_last'] - 0.5 ) 

            if bottom_shift:
                reward -= penalty

        # SELLING
        if inventory == 0:   
            # get bonus as it sells near upper channel
            reward += get_reward_entry(info, obs['position_inside_channel'])

            # get bonus as exit higher than entry
            difference = get_reward_from_delta(info, obs['distance_from_avg_pos'] - 0.5) 
            reward += difference * 1
            
            # # get bonus as exit far from lowest point to encourage stop loss
            diffFromLowest = get_reward_from_delta(info, obs['distance_from_lowest'] - 0.5)
            reward += diffFromLowest * 1

            if top_shift or bottom_shift:
                reward -= penalty

            
        
        reward = reward * weight

    else:
        reward = int(0)

    # print(action, info['order'])
    # get penalty as choosing wrong move
    if (action==1 or action==0) and info['order'] == 'wait':
        # print("Get penalized")
        reward -= penalty    

    return reward

