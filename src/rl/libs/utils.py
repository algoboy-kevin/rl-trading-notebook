import os
import configparser
import math
import random
import numpy as np
from typing import List, TypeVar, Generic, TypedDict

available_strategy = ['DUAL_MA', 'DONCHIAN_CHANNEL']

class OrderInfo(TypedDict):
    order: str
    size: int
    price: float
    reason: str
    profit: float

T = TypeVar('T')

class TrainingConfig(TypedDict):
    starting_balance: float
    leverage: int
    comission_trade: float
    comission_funding: float
    max_order: int
    is_random_inventory: bool
    is_record_history: bool
    is_long_only: bool

def getLastEpisode(folderPath):
    episodes = []
    for file in os.listdir(folderPath):
        if file.lower().startswith('checkpoint-'):
            episodes.append(file)

    if episodes:
        # Sort episodes based on the checkpoint number
        sortedEpisodes = sorted(episodes, key=lambda x: int(x.split('-')[-1].split('.')[0]))
        filename = sortedEpisodes[-1]
        filePath = os.path.join(folderPath, filename)
        return filePath
    else:
        raise FileNotFoundError("No checkpoint found in the folder.")
  
def doesCheckpointExist(folderPath):
    for file in os.listdir(folderPath):
        if 'checkpoint-' in file:
            return True
    return False

class FixedSizeList(Generic[T]):
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.items: List[T] = []

    def append(self, item: T):
        if len(self.items) >= self.maxlen:
            self.items.pop(0)
        self.items.append(item)

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)
    
def sigmoid(x: float) -> float:
    """
    Compute the sigmoid of x.

    Args:
    x (float): Input value

    Returns:
    float: Sigmoid of x
    """
    return 1 / (1 + math.exp(-x))

def exponential(x: float) -> float:
    """
    Compute a normalized exponential function for x between 0 and 1.

    Args:
    x (float): Input value (will be clipped between 0 and 1)

    Returns:
    float: Normalized exponential of x, ranging from 0 to 1
    """
    x = max(0, min(x, 1))  # Clip x between 0 and 1
    return (math.exp(x) - 1) / (math.e - 1)  # Normalizes the output to [0, 1]

def fitPercentage(x: float) -> float:
    return min(max(0.5 + x/50, 0), 1)

def fetchDataSlice(df, nSteps = 8640, randomize=False, startingIndex=0, endIndex=None):
    if randomize:
        max_start = len(df) - nSteps
        startingIndex = random.randint(0, max_start)
    
    endIndex = startingIndex + nSteps if endIndex is None else endIndex
    
    if endIndex > len(df):
        raise ValueError("End index exceeds data length")
    
    return df[startingIndex:endIndex]

def to_float_list(input_data):
    """
    Convert input to a list of floats.

    If input is a dictionary, extract all values and convert them to floats.
    If input is a list or any other iterable, convert all elements to floats.

    Args:
    input_data (dict or iterable): Input data to be converted

    Returns:
    list: A list of float values
    """
    if isinstance(input_data, dict):
         return [float(value) for values in input_data.values() for value in np.atleast_1d(values)]
    else:
        return list(input_data)
    
def int_to_one_hot(value: int, num_classes: int) -> list:
    """
    Convert an integer to a one-hot encoded list.

    Args:
    value (int): The integer to be converted.
    num_classes (int): The total number of classes (length of the resulting list).

    Returns:
    list: A one-hot encoded list where the 'value'-th element is 1 and others are 0.

    Raises:
    ValueError: If the value is negative or greater than or equal to num_classes.
    """

    if value < 0 or value >= num_classes:
        raise ValueError(f"Value must be between 0 and {num_classes - 1}")
    
    one_hot = [0] * num_classes
    one_hot[value] = 1
    return one_hot

def getConfigAtIndex(index: int, dict_obj: dict):
    first_key = list(dict(dict_obj).keys())[0]
    if index < 0 or index >= len(dict_obj[first_key]):
        raise ValueError("Invalid index")
    
    return {key: values[index] for key, values in dict_obj.items()}

class OrderInfo(TypedDict):
    order: str
    size: int
    price: float
    reason: str
    profit: float


class PositionStatus:
    def __init__(self, position: str, size: float, avgPrice: float):
        self.position = position
        self.size = size
        self.avgPrice = avgPrice

def getCrossover(streak: float):
    recentlyCrossoverLong = 0.0
    recentlyCrossoverShort = 0.0

    if abs(streak) <= 8:
        if streak > 0:
            recentlyCrossoverLong = max((9 - streak)/8, 0)

        elif streak < 0:
            recentlyCrossoverShort = max((9 + streak)/8, 0)

    return recentlyCrossoverLong, recentlyCrossoverShort

def calculateCrossoverReward(info: OrderInfo, streak: float):
    baseReward = 0.0021 * 1
    long, short = getCrossover(streak)

    if info['order'] == 'long':
        return baseReward * long

    else:
        return baseReward * short
    
def volBandsEntryReward(info: OrderInfo, rowData: dict):
    upper = rowData['VOL_BAND_UPPER']
    lower = rowData['VOL_BAND_LOWER']
    close = rowData['SMA5']

    upperToClose = (close - upper) / upper # close should be higher than upper
    lowerToClose = (lower - close) / lower # close should be lower than lower
    logUpper = math.log10(1 + upperToClose)
    logLower = math.log10(1 + lowerToClose)
    
        # the delta should be negative for buying
    reward = 0.0
    if info['order'] == 'buy':
        reward += logLower
        reward *= info['size']

    elif info['order'] == 'sell':
        reward += logUpper 
        reward *= info['size']

    return reward

def get_distance_given_mark(
        mark: float, 
        entry:float, 
        orders: int
    ):
    if orders == 0 or entry == 0:
        return 0.0
    
    difference = (mark/entry) - 1
    return difference

def getLogValue(distance: float, multiplier = 1):
    nominal = 1 + distance * multiplier
    return math.log10(nominal)

def get_distance_from_avg_pos(
        mark_price: float, 
        avg_position: float, 
        inventory: int, 
        log_multiplier: float
    ):
    distance_from_avg_pos = get_distance_given_mark(mark_price, avg_position, inventory)
    distance_from_avg_pos = min(max(0.5 + getLogValue(distance_from_avg_pos, log_multiplier), 0), 1)
    return distance_from_avg_pos

def get_distance_from_lowest(
    mark_price: float, 
    lowest_position: float, 
    inventory: int, 
    log_multiplier: float
):
    if lowest_position != 0:
        distance_from_lowest = get_distance_given_mark(mark_price, lowest_position, inventory)
        distance_from_lowest = min(max(0.5 + getLogValue(distance_from_lowest, log_multiplier), 0), 1)
    else:
        distance_from_lowest = 0.5

    return distance_from_lowest

def get_distance_from_last(
    mark_price: float, 
    last_buy_position: float, 
    inventory: int, 
    log_multiplier: float
):
    distance_from_last = get_distance_given_mark(mark_price, last_buy_position, inventory)
    distance_from_last = min(max(0.5 + getLogValue(distance_from_last, log_multiplier), 0), 1)
    return distance_from_last