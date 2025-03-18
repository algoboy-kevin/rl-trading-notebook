import random
import math
from .order import Order
from src.rl.libs.utils import OrderInfo, PositionStatus

AVG_DOWN_TRES = 0.05


class InventoryManager(): 
    def __init__(self, 
        maxOrders: int, 
        leverage: int, 
        minimumSize: float, 
        tradeComission: float,
        fundingFee: float, 
        isRandomInventory: bool, 
        isLongOnly: bool,
        balance: float
        ):
        self.isRandomizeInventory = isRandomInventory
        self.maxOrder = maxOrders
        self.leverage = leverage
        self.minimumSize = minimumSize
        self.tradeComission = tradeComission
        self.fundingFee = fundingFee
        self.isLongOnly = isLongOnly
        self.inventory: int = 0
        self.orders: list[Order] = []
        self.ordersProfitable = 0
        self.initialSize = self.getSize(balance)
        self.tradeTotal = 0
        self.shortTermThershold = 30
        self.profitability = 0
        self.totalProfit = 0
        self.nLongs = 0
        self.nShorts = 0
        self.shortTermProfits = 0
        self.sumHoldingTime = 0
        self.rowData = None
        
    def reset(self, balance: float, rowData: dict):
        self.orders = []
        self.ordersProfitable = 0
        self.tradeTotal = 0
        self.initialSize = self.getSize(balance)
        self.inventory = self.generateRandomInventory(balance, rowData)
        self.totalProfit = 0
        self.profitability = 0
        self.nLongs = 0
        self.nShorts = 0
        self.shortTermProfits = 0
        self.sumHoldingTime = 0

    def generateRandomInventory(self, balance: float, rowData: dict):
        if not self.isRandomizeInventory:
            return 0
        
        minimum = -self.maxOrder
        if self.isLongOnly:
            minimum = 0
        randomInventory = random.randint(minimum, self.maxOrder)

        if randomInventory == 0:
            return 0

        # Determine the position (buy or sell) based on the sign of randomInventory
        position = "buy" if randomInventory > 0 else "sell"
        numOrders = abs(randomInventory)

        for _ in range(numOrders):
            marginUsed = self.getSize(balance)
            self.createOrder(position, marginUsed, rowData)

        # Update the inventory
        return randomInventory

    def buy(self, price: float, balance: float, rowData: dict) -> OrderInfo:
        position = "buy"

        res = self.preOrderFunc(position, price)
        if res != None:
            return res
        
        profit = self.executeOrder(position, balance, rowData)
        return OrderInfo(
            order=position,
            size=1,
            price=price,
            profit=profit
        )
    
    def sell(self, price: float, balance: float, rowData: dict) -> OrderInfo:
        position = "sell"

        res = self.preOrderFunc(position, price)
        if res != None:
            return res
        
        profit = self.executeOrder(position, price, balance, rowData)
        return OrderInfo(
            order=position,
            size=1,
            price=price,
            profit=profit
        )
    
    def closeAllOrders(self, closePrice: float):
        totalProfit = 0.0
        countOrder = len(self.orders)

        if self.inventory > 0:
            label = "sell"
        else:
            label = "buy"
            
        for _ in range(countOrder):
            profit = self.closeOrder(closePrice)
            totalProfit += profit
            
        return OrderInfo(
            order=label, 
            size=countOrder,
            price=closePrice, 
            profit=totalProfit,
            reason=""
        )
    
    def applyFundingFee(self):
        for order in self.orders:
            order.addFundingFee(self.fundingFee)
    
    def preOrderFunc(self, position: str, price: float):
        if len(self.orders) >= self.maxOrder:
            return OrderInfo(
                order="wait",
                reason="max inventory reached",
                profit=0.0,
                price=price
            )
        
        return None
            
    def createOrder(self, position: str, margin_used: float, mark: dict):
        order = Order(position, margin_used, self.leverage, mark, self.tradeComission)
        self.orders.append(order)

        if order.position == "buy":
            self.inventory += 1
        else:
            self.inventory -= 1
        
    def executeOrder(self, position: float, balance: float, mark: dict ):
        profit = 0.0
        isBuyOrSell = self.inventory >= 0 if position == "buy" else self.inventory <= 0
        
        if isBuyOrSell:
            marginSize = self.getSize(balance)
            self.createOrder(position, marginSize, mark)
        else:
            profit = self.closeOrder(mark['close'])

        return profit
    
    def closeOrder(self, price: float) -> float:
        if len(self.orders) == 0:
            return 0.0
        
        order = self.orders.pop()
        profit, delta = order.closePosition(price)
        
        self.tradeTotal += 1
        self.totalProfit += profit
        self.profitability += delta
        
        if profit > 0:
            self.ordersProfitable += 1

        if order.position == "buy":
            self.inventory -= 1
            self.nLongs += 1
        else:
            self.inventory += 1
            self.nShorts += 1

        self.sumHoldingTime += order.holdingTime

        if order.holdingTime < self.shortTermThershold:
            self.shortTermProfits += 1

        return profit
    
    # View functions
    def getSize(self, balance: float):
        if self.inventory == 0:
            self.initialSize = round((balance / float(self.maxOrder)) * 100) / 100
        
        marginSize = self.initialSize

        if marginSize < self.minimumSize:
            marginSize = self.minimumSize
        
        return marginSize
    
    def getCurrentPosition(self) -> PositionStatus:
        if self.inventory == 0:
            return PositionStatus(position="NEUTRAL", size=0, avgPrice=0.0)
        
        position = "LONG" if self.inventory > 0 else "SHORT"
        
        return PositionStatus(
            position=position,
            size=abs(self.inventory),
            avgPrice=self.getAveragePrice('close')
        )
    
    def getProfitability(self):
        if self.tradeTotal == 0:
            return 0

        return self.ordersProfitable / self.tradeTotal
    
    def getInventoryCount(self) -> tuple[float, float]:
        if self.inventory == 0:
            return 0, 0
        elif self.inventory > 0:
            return self.inventory, 0
        else:
            return 0, abs(self.inventory)
    
    def getAveragePrice(self, mark_key: str):
        if len(self.orders) == 0:
            return 0.0

        totalPrice = 0.0
        totalSize = 0.0
        for order in self.orders:
            totalPrice += order.markPoint[mark_key] * order.size
            totalSize += order.size

        return totalPrice / totalSize
    
    def getLastEntry(self, mark_key: str):
        if not self.orders:
            return 0.0
        
        lastMark = 0
        for order in self.orders:
            lastMark = order.markPoint[mark_key]
        
        return lastMark
