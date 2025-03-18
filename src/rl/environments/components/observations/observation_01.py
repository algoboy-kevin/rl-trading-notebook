from typing import TypedDict
import numpy as np
from src.rl.libs.utils import fitPercentage, get_distance_from_avg_pos, get_distance_from_lowest, get_distance_given_mark, getLogValue

def get_dual_ma_obs(
    row_data: dict,
    mark_price: float,
    avg_position: float,
    lowest_position: float,
    n_long: int,
    n_short: int,
    max_order: float,
    inventory: int,
):
    
    log_multiplier = 2
        # get distance from average position
    distance_from_avg_pos = get_distance_from_avg_pos(mark_price, avg_position, inventory, log_multiplier)
    distance_from_lowest = get_distance_from_lowest(mark_price, lowest_position, inventory, log_multiplier)

    # shape v3 from pct into observation value
    v3 = fitPercentage(row_data['V3'])

    n_long = n_long / max_order
    n_short = n_short / max_order
    
    result = np.array([
        v3, 
        distance_from_avg_pos, 
        distance_from_lowest, 
        n_long, 
        n_short
        ], dtype=np.float32)
    
    return result

class ParamsObsDualMA(TypedDict):
    row_data: dict
    mark_price: float
    avg_position: float
    lowest_position: float
    n_long: int
    n_short: int
    max_order: float
    inventory: int