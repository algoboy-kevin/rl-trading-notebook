import math

from collections import deque
from typing import TypedDict
from src.rl.libs.utils import FixedSizeList, PositionStatus, OrderInfo

class Records:
    def __init__(self, recordHistory: bool):
        self.recordHistory = recordHistory
        self.positionStatus = PositionStatus("NEUTRAL", 0.0, 0.0)
        self.inventoryPrevious = deque(maxlen=6)
        self.historyInventory = deque(maxlen=100)
        self.historyMarker = deque(maxlen=100)
        self.historyPrice = deque(maxlen=100)
        self.historyEquity = deque(maxlen=100)
        self.historyOrder = FixedSizeList[OrderInfo](10)
        self.equityForCurve = deque(maxlen=86400)
            
    def reset(self):
        self.inventoryPrevious.clear()
        self.historyInventory.clear()
        self.historyMarker.clear()
        self.historyPrice.clear()
        self.historyEquity.clear()
        self.historyOrder = FixedSizeList[OrderInfo](10)
        self.equityForCurve.clear()
        self.positionStatus = PositionStatus("NEUTRAL", 0.0, 0.0)
            
    def recordStep(self, 
        close: float, 
        equity: float, 
        inventory: int, 
        positionStatus: PositionStatus, 
        info: OrderInfo,
        mark: dict,
        ):

        if self.recordHistory:
            self.historyEquity.append(equity)
            self.positionStatus = positionStatus

            if info["order"] != "wait":
                self.historyOrder.append(info)
            
            self.historyInventory.append(inventory)
            # self.historyMarker.append(mark)
            self.historyPrice.append(close)