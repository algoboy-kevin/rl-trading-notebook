class Order:
   def __init__(self, 
      position: str, 
      margin_used: float, 
      leverage: float, 
      mark_point: dict,
      comission_size: float = 0.03 / 100
   ):
      self.position = position
      self.size = margin_used * leverage
      self.marginUsed = margin_used
      self.entryPrice = mark_point['close']
      self.comissionSize = comission_size
      self.comissionTotal = self.getComission()
      self.profit = 0
      self.fundingFee = 0
      self.holdingTime = 0
      self.markPoint = mark_point 

   def getUpdatedSize(self):
      return self.size + self.getProfit()
   
   def getComission(self):
      return self.size * self.comissionSize

   def updateProfit(self, price: float):
      self.holdingTime += 1
      if self.position == "buy":
         self.profit = (price / self.entryPrice - 1) * self.size
      else:
         self.profit = (1 - price / self.entryPrice) * self.size

      self.profit -= self.comissionTotal + self.fundingFee

   def closePosition(self, price: float):
      self.updateProfit(price)

      # add comission to total comission when closing position
      profit = self.profit

      return profit, self.getDelta(profit)
   
   def addFundingFee(self, funding_fee: float):
      self.fundingFee += self.size * funding_fee

   def getDelta(self, profit: float):
     return (self.size + profit) / self.size
   
   
   


