import numpy as np
import pandas as pd
from .indicator import *


def calculate_indicators(df: pd.DataFrame, indicators: list[str]):
    """
    Calculate specified indicators for a given DataFrame using pandas_ta.
    
    :param df: pandas DataFrame with 'time' and 'close' columns
    :param indicators: list of indicator names to calculate
    :return: pandas DataFrame with calculated indicators
    """
    print(f"Processing indicator: {indicators}")
    result_df = df.copy()
    
    for indicator in indicators:
        if indicator.startswith('EMA'):
            period = int(indicator[3:])
            result_df.ta.ema(length=period, append=True, col_names=(indicator,))
        elif indicator.startswith('SMA'):
            period = int(indicator[3:])
            result_df.ta.sma(length=period, append=True, col_names=(indicator,))
        elif indicator.startswith('RSI'):
            period = int(indicator[3:])
            result_df.ta.rsi(length=period, append=True, col_names=(indicator,))
        else:
            print(f"Warning: Unsupported indicator {indicator}")

    # Remove rows with NaN values
    result_df = result_df.dropna()
    return result_df

def calculate_derived_indicators(df: pd.DataFrame, derived_indicators: list[str]) -> pd.DataFrame:
    """
    Calculate specified derived indicators for a given DataFrame.
    
    :param df: pandas DataFrame with 'close', 'ema5', and 'ema60' columns
    :param derived_indicators: list of derived indicator names to calculate
    :return: pandas DataFrame with added derived indicator columns
    """
    print(f"Processing derived indicator: {derived_indicators}")
    result_df = df.copy()

    for indicator in derived_indicators:
        # Get all column names
        all_columns = result_df.columns.tolist()
            
        if indicator == 'LOG_SMA':
            getLogSMAIndicator(result_df, all_columns)
        elif indicator == 'V2':
            getV2Indicator(result_df, all_columns)
        elif indicator == 'V3':
            getV3Indicator(result_df, all_columns)
        elif indicator == 'PRICE_CHANGE':
            getPriceChangeIndicator(result_df, all_columns)
        elif indicator == 'RSI_SMA':
            getRSISMAIndicator(result_df, all_columns)
        elif indicator == 'V3_STREAK':
            getV3StreakIndicator(result_df, all_columns)
        elif indicator == 'V3_STREAK_SIGNAL':
            getV3StreakSignal(result_df, all_columns)
        elif indicator == 'VOLATILITY_BAND':
            getVolatilityBand(result_df, all_columns)
        elif indicator == 'DONCHIAN_CHANNEL':
            getDonchianChannels(result_df, all_columns)
        elif indicator == 'DONCHIAN_CHANNEL_SMA':
            getDonchianChannelsSMA(result_df, all_columns, 40)
    
        else:
            raise(f"Warning: Unsupported derived indicator {indicator}")
    
    # Remove rows with NaN values
    result_df = result_df.dropna()
    
    return result_df
