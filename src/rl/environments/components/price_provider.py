import pandas as pd
import os
import random

from src.rl.libs.mocks import createPriceDataFromCSV


class PriceProvider:
    def __init__(self, 
        fileName='PEPEUSDT.csv',
        directory='',
        randomize=False,
        indicators=[],
        derivedIndicators=[],
        ):
        self.indicators = indicators
        self.derivedIndicators = derivedIndicators
        self.dir = directory
        self.fromRecords(fileName)
        self.randomize = randomize

    def getRow(self, index):
        if 0 <= index < len(self.df):
            return self.df.iloc[index]
        else:
            raise IndexError("Index out of bounds")
        
    def fromRecords(self, fileName):
        priceHistoryDir = self.dir
        csvFiles = [f for f in os.listdir(priceHistoryDir) if f.endswith('.csv')]
        
        if not csvFiles:
            raise FileNotFoundError("No CSV files found in ./price_history directory")
        
        csvFilePath = os.path.join(priceHistoryDir, fileName)
        data = createPriceDataFromCSV(
            csvFilePath, 
            self.indicators, 
            self.derivedIndicators
        )
        print(data.head())

        self.df = data.to_dict('records')

    def fetchDataSlice(self, 
        nSteps = 8640, 
        startingIndex=0, 
        endIndex=None
        ):

        if self.randomize:
            max_start = len(self.df) - nSteps
            print("Starting index:",startingIndex)
            startingIndex = random.randint(0, max_start)
        endIndex = startingIndex + nSteps if endIndex is None else endIndex
        
        if endIndex > len(self.df):
            raise ValueError("End index exceeds data length")
            
        data_slice = pd.DataFrame(self.df[startingIndex:endIndex])
        
        # first_log_sma = data_slice['LOG_SMA'].iloc[0]
        # data_slice['LOG_SMA'] = data_slice.apply(lambda row: row['LOG_SMA'] - first_log_sma, axis=1)
            
        return data_slice.to_dict('records')