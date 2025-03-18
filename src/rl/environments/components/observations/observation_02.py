from typing import TypedDict
import numpy as np
from src.rl.libs.utils import (
    get_distance_from_avg_pos,
    get_distance_from_lowest,
    get_distance_from_last,
    )

class ParamsObsDonchian(TypedDict):
    mark_price: float
    avg_position: float
    lowest_position: float
    last_buy_position: float
    inventory: int
    max_inventory: int
    upper_channel: float
    lower_channel: float
    upper_channel_shift: float
    lower_channel_shift: float
    upper_shift_5: float
    lower_shift_5: float

def get_donchian_obs(
        mark_price: float,
        avg_position: float,
        lowest_position: float,
        last_buy_position: float,
        inventory: int,
        max_inventory: int,
        upper_channel: float,
        lower_channel: float,
        upper_channel_shift: float,
        lower_channel_shift: float,
        upper_shift_5: float,
        lower_shift_5: float
    ):

    # calculate distance
    channel_range = upper_channel - lower_channel
    if channel_range == 0:
        position_inside_channel = 0.5  # Default to middle if channel has no range
    else:
        position_inside_channel = (mark_price - lower_channel) / channel_range
        position_inside_channel = min(max(position_inside_channel, 0), 1)

    # calculate shifting
    # calculate distance
    log_multiplier = 1

    # get distance from average position
    distance_from_avg_pos = get_distance_from_avg_pos(mark_price, avg_position, inventory, log_multiplier)
    distance_from_lowest = get_distance_from_lowest(mark_price, lowest_position, inventory, log_multiplier)
    distance_from_last = get_distance_from_last(mark_price, last_buy_position, inventory, log_multiplier)
    
    # get inventory
    inventory = inventory / max_inventory

    result = np.array([
        position_inside_channel, 
        lower_channel_shift, 
        upper_channel_shift, 
        upper_shift_5,
        lower_shift_5,
        distance_from_avg_pos, 
        distance_from_lowest,
        distance_from_last,
        inventory,
        ], dtype=np.float32)

    return result

class DonchianObsDict(TypedDict):
    position_inside_channel: float 
    lower_channel_shift: float 
    upper_channel_shift: float 
    upper_shift_5: float
    lower_shift_5: float
    distance_from_avg_pos: float 
    distance_from_lowest: float
    distance_from_last: float
    inventory: float

def get_donchian_obs_dict(obs_array: np.ndarray) -> DonchianObsDict:
    return {
        'position_inside_channel': obs_array[0],
        'lower_channel_shift': obs_array[1], 
        'upper_channel_shift': obs_array[2],
        'upper_shift_5': obs_array[3],
        'lower_shift_5': obs_array[4],
        'distance_from_avg_pos': obs_array[5],
        'distance_from_lowest': obs_array[6],
        'distance_from_last': obs_array[7],
        'inventory': obs_array[8]
    }

    