import math
from typing import List
from .order import Order

class Account:
    def __init__(self, balance: float):
        self.balance: float = balance
        self.balance_initial: float = balance
        self.equity: float = balance
        self.equity_highest: float = balance
        self.margin_position: float = 0.0
        self.margin_available: float = balance
        self.drawdown: float = 0.0
        self.max_drawdown: float = 0.00
        self.local_high: float = balance
        self.local_low: float = balance

    def reset(self):
        self.balance = self.balance_initial
        self.equity = self.balance_initial
        self.equity_highest = self.balance_initial
        self.margin_position = 0.0
        self.margin_available = self.balance_initial
        self.drawdown = 0.0
        self.max_drawdown = 0.00
        self.local_high = self.balance_initial
        self.local_low = self.balance_initial

    def updateMargin(self, price: float, orders: List[Order]):
        marginUsed = 0.0
        profits = 0.0
        isAnyOrder = len(orders) != 0
        if isAnyOrder:
            for order in orders:
                order.updateProfit(price)
                marginUsed += order.marginUsed
                profits += order.profit

        self.margin_position = marginUsed
        self.margin_available = self.balance - marginUsed
        self.equity = self.margin_available + self.margin_position + profits
        self.recordLocalHighLow(isAnyOrder)
        

    def updateDrawdown(self):
        if self.equity <= 0:
            self.drawdown = 1.0
        else:
            self.equity_highest = max(self.equity_highest, self.equity)
            
            # Calculate current drawdown
            currentDrawdown = (self.equity_highest - self.equity) / self.equity_highest
            self.drawdown = currentDrawdown
            
            # Update max drawdown
            self.max_drawdown = max(self.max_drawdown, currentDrawdown)

    def getEquityMultiplier(self) -> float:
        return self.balance / self.balance_initial

    def recordLocalHighLow(self, isAnyOrder: bool):
        if isAnyOrder:
            self.local_high = max(self.local_high, self.equity)
            self.local_low = min(self.local_low, self.equity)
        else:
            self.local_high = self.balance
            self.local_low = self.balance

    def getLocalLowDistance(self) -> float:
        return max((self.equity - self.local_low) / self.equity, 0)


