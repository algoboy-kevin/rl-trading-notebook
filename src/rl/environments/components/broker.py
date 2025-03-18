import math
import numpy as np
from typing import Dict, List

from .account import Account
from .inventory_manager import InventoryManager, OrderInfo
from .records import Records

class Broker:
    def __init__(
        self,
        env_config: Dict,
        record_history: bool,
        ):

        # Setup configuration
        self.no_negative_reward = env_config['no_negative_reward']
        self.ending_bonus = env_config['ending_bonus_reward']
        self.is_long_only = env_config['is_long_only']
        self.is_scalping = env_config['is_scalping']

        # Initializing class
        self.ac = Account(env_config['starting_balance'])
        self.im = InventoryManager(
            env_config['max_order'], 
            env_config['leverage'], 
            10, 
            env_config['comission_trade'],
            env_config['comission_funding'], 
            env_config['is_random_inventory'], 
            env_config['is_long_only'],
            env_config['starting_balance']
        )
        self.re = Records(record_history)
        self.row_data: Dict[str, float] = {}
        self.n_step: int = 0
        self.n_step_idle: int = 0
        self.done: bool = False
        self.reward_cum = 0
        self.lowest_point = 0

    def reset(self, row: Dict[str, float]):
        self.n_step = 0
        self.n_step_idle = 0
        self.done = False
        self.row_data = row
        self.reward_cum = 0
        self.first_entry = 0
        self.lowest_point = 0

        self.ac.reset()
        self.im.reset(self.ac.balance, self.row_data)
        self.re.reset()

        self.updateState(row)
    
    def executeAction(self, action: int):
        info = self.takeAction(action)
        self.afterAction(info)
        
        return info

    def takeAction(self, action: int) -> OrderInfo:

        if action == 0:
            if self.im.inventory < 0:
                return self.closeAllOrders()

            return self.buy()

        if action == 1:
            if self.im.inventory > 0:
                return self.closeAllOrders()
            
            if not self.is_long_only:
                return self.sell()
        
        return OrderInfo(order="wait", reason="idle", size=0, price=0.0, profit=0.0)
    
    def buy(self) -> OrderInfo:
        close = self.row_data["close"]
        row_data = self.row_data
        info = self.im.buy(close, self.ac.balance, row_data)

        self.afterSales(close, info)
        return info

    def sell(self) -> OrderInfo:
        close = self.row_data["close"]
        mark = self.row_data
        info = self.im.sell(close, self.ac.balance, mark)

        self.afterSales(close, info)
        return info
    
    def closeAllOrders(self) -> OrderInfo:
        closePrice = self.row_data["close"]
        info = self.im.closeAllOrders(closePrice)

        self.afterSales(closePrice, info)
        return info
    
    def afterSales(self, close: float, info: OrderInfo):
        if info["profit"] != 0:
            self.ac.balance += info["profit"]

        self.ac.updateMargin(close, self.im.orders)
        self.ac.updateDrawdown()

    def afterAction(self, info):
        self.n_step += 1

        # Add idle count
        if info['order'] == 'wait':
            self.n_step_idle += 1
        else:
            self.n_step_idle = 0

        positionStatus = self.im.getCurrentPosition()

        self.re.recordStep(
            self.row_data["close"],
            self.ac.equity,
            self.im.inventory,
            positionStatus,
            info,
            self.row_data
        )

    def updateState(self, row: Dict[str, float]):
        self.row_data = row
        self.ac.updateMargin(self.row_data["close"], self.im.orders)
        self.ac.updateDrawdown()
        self.updateLowestPoint()

    def applyFundingFee(self):
        self.im.applyFundingFee()
        
    def getDrawdown(self) -> float:
        return self.ac.drawdown
    
    def getMaxDD(self) -> float:
        return self.ac.max_drawdown

    def getProfit(self) -> float:
        return (max(self.ac.equity, 0) / self.ac.balance) - 1
    
    def updateLowestPoint(self):
        if self.im.inventory != 0:
            # Initialize lowest_point if not set
            if self.lowest_point == 0:
                self.lowest_point = self.row_data['SMA5']
            # Update lowest point if current SMA5 is lower
            elif self.lowest_point > self.row_data['SMA5']:
                self.lowest_point = self.row_data['SMA5']
        else:
            # Reset lowest point when no position
            self.lowest_point = 0
    
    def getDiffFromAvgPos(self, mark: float):
        if len(self.im.orders) == 0:
            return 0.0
        
        avgPos = self.im.getAveragePrice('SMA5')
        currentDiff = (mark/avgPos) - 1
        return currentDiff
    
    def getDiffFromLatest(self, mark: float):
        if len(self.im.orders) == 0:
            return 0.0
        
        lastPos = self.im.getLastEntry('SMA5')
        currentDiff = (mark/lastPos) - 1
        return currentDiff

    