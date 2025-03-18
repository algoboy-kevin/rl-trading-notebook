from typing import List
import numpy as np
import pandas as pd
import pandas_ta as ta

from src.rl.libs.indicators import calculate_indicators, calculate_derived_indicators


def createPriceDataFromCSV(
    csvFilePath: str, 
    indicators: list[str], 
    derivedIndicator=[], 
    startIndex = 0
    ):
    # Read the CSV file
    df = pd.read_csv(csvFilePath, sep=',')
    nSteps = len(df) - 1
    
    # Ensure we have at least nSteps rows
    if len(df) < nSteps:
        raise ValueError(f"CSV file must contain at least {nSteps} rows")
    
    length = len(df) - nSteps
    if startIndex > length:
        raise ValueError(f"The index must be below {length}")
    
    # Select nSteps consecutive rows
    selectedDf = df.iloc[startIndex:startIndex + nSteps].reset_index(drop=True)
    
    print(f"Before processing: {len(selectedDf)} rows")
    
    selectedDf = calculate_indicators(selectedDf, indicators)

    if len(derivedIndicator) != 0:
        selectedDf = calculate_derived_indicators(selectedDf, derivedIndicator)


    print(f"Price provider - After processing: {len(selectedDf)} rows")

    return selectedDf


