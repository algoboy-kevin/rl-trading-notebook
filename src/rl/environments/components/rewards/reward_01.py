import math
from typing import TypedDict
from src.rl.libs.utils import OrderInfo, get_distance_given_mark

def get_reward_on_entry(info: OrderInfo, v3: float):
    reward = 0
    if info['order'] == 'buy':
        reward =  -v3
        reward *= info['size']

    elif info['order'] == 'sell':
        reward = v3
        reward *= info['size']

    reward -= 0.1

    delta = 100 + reward
    reward = math.log10(delta/100)
    return reward

def get_reward_from_delta(info: OrderInfo, deltaFromEntry: float):
    multiplier = 1
    # if info['order'] == 'sell':
    #     multiplier = -1
    
    reward = (deltaFromEntry * multiplier)
    delta = 1 + reward - 0.01
    deltaMa = math.log10(delta / 1)  * info['size']
    return deltaMa

def get_reward_on_reentry(info: OrderInfo, deltaSma: float):
    # the delta should be negative for buying
    if info['order'] == 'buy':
        reward = -deltaSma 

    elif info['order'] == 'sell':
        reward = deltaSma 

    delta = 1 + reward - 0.01

    reward = math.log10(delta)
    return reward

class ParamsPostActionDualMA(TypedDict):
    action: int
    info: OrderInfo
    v3: float
    inventory: int
    mark_price: float
    avg_position: float
    lowest_position: float
    last_buy_position: float

class ParamsPostUpdateDualMA(TypedDict):
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
    reward = log_val * multiplier * 1

    if i_post_action == 0:
        reward = int(0)

    return reward

def get_post_action_reward(
        action: int,
        info: OrderInfo, 
        inventory: int, 
        v3: float, 
        mark_price: float,
        avg_position: float,
        lowest_position: float,
        last_buy_position: float,
    ):

    # reward config
    reward = 0.0 # base reward
    weight = 1 
    comission = 0.03

    # get distance from average position
    distance_from_avg_pos = get_distance_given_mark(mark_price, avg_position, inventory)
    distance_from_lowest = get_distance_given_mark(mark_price, lowest_position, inventory)
    distance_from_last = get_distance_given_mark(mark_price, last_buy_position, inventory)

    if info['order'] == 'buy' or info['order'] == 'sell':    
        reward += math.log10((100 - comission)/100) * info['size'] 
        inventory  = abs(inventory)

        # if entry
        if inventory == 1:
            # if info['order'] == 'buy':
            reward += get_reward_on_entry(info, v3)

        # if selling
        if inventory == 0:                
            reward += get_reward_on_entry(info, v3) * 0.2

            # to stimulate profit taking
            difference = get_reward_from_delta(info, distance_from_avg_pos) 
            reward += difference * 1
            
            # to stimulate cutloss by closing from the lowest point
            diffFromLowest = get_reward_from_delta(info, distance_from_lowest)
            reward += diffFromLowest * 0.8
            
        # if averaging down
        if inventory > 1:
            reward += get_reward_on_reentry(info, distance_from_last) 

        reward = reward * weight

    else:
        reward = int(0)

    if (action==1 or action==0) and info['order'] == 'wait':
        reward += math.log10((100 - comission)/100)     

    return reward

