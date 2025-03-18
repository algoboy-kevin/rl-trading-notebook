import pandas as pd
import numpy as np

def check_prerequisites(prerequisites: list[str], all_columns: list[str]) -> bool:
    """
    Check if all prerequisites exist in all_columns
    
    :param prerequisites: List of required column names
    :param all_columns: List of available column names
    :return: True if all prerequisites are found, False otherwise
    """
    missing = [col for col in prerequisites if col not in all_columns]
    if missing:
        raise ValueError(f"Missing prerequisites: {missing}. Available columns: {all_columns}")
    return True

def getDeltaEmaClose(ema: float, close: float):
    return ((close - ema) / ema) * 100

def getLogSMAIndicator(result_df: pd.DataFrame, all_columns: list[str]):
    prerequisites = ['SMA5']
    check_prerequisites(prerequisites, all_columns)
    result_df['LOG_SMA'] = result_df.apply(
        lambda row: np.log10(row['SMA5'] / result_df['SMA5'].shift(1).loc[row.name]) if row.name > 0 else 0, 
        axis=1
    ).cumsum()

def getV2Indicator(result_df: pd.DataFrame, all_columns: list[str]):
    prerequisites = ['EMA5']
    check_prerequisites(prerequisites, all_columns)
    result_df['V2'] = result_df.apply(lambda row: getDeltaEmaClose(row['EMA5'], row['close']), axis=1)

def getV3Indicator(result_df: pd.DataFrame, all_columns: list[str]):
    prerequisites = ['SMA60', 'SMA5']
    check_prerequisites(prerequisites, all_columns)
    result_df['V3'] = result_df.apply(lambda row: getDeltaEmaClose(row['SMA60'], row['SMA5']), axis=1)

def getPriceChangeIndicator(result_df: pd.DataFrame, all_columns: list[str]):
    prerequisites = ['close']
    check_prerequisites(prerequisites, all_columns)
    result_df['PRICE_CHANGE'] = result_df.apply(lambda row: np.log10(row['close'] / result_df['close'].shift(1).loc[row.name]) if row.name > 0 else 0, axis=1)

def getRSISMAIndicator(result_df: pd.DataFrame, all_columns: list[str]):
    prerequisites = ['RSI14']
    check_prerequisites(prerequisites, all_columns)
    result_df['RSI_SMA'] = result_df['RSI14'].rolling(window=7).mean()

def getV3StreakIndicator(result_df: pd.DataFrame, all_columns: list[str]):
    prerequisites = ['V3']
    check_prerequisites(prerequisites, all_columns)
    result_df['V3_STREAK'] = 0
    # Calculate streaks
    for i in range(1, len(result_df)):
        curr_v3 = result_df['V3'].iloc[i]
        prev_v3 = result_df['V3'].iloc[i-1]
        prev_streak = result_df['V3_STREAK'].iloc[i-1]
        
        # If both current and previous V3 are positive
        if curr_v3 > 0 and prev_v3 > 0:
            result_df.iloc[i, result_df.columns.get_loc('V3_STREAK')] = prev_streak + 1
        # If both current and previous V3 are negative  
        elif curr_v3 < 0 and prev_v3 < 0:
            result_df.iloc[i, result_df.columns.get_loc('V3_STREAK')] = prev_streak - 1
        # If sign changes, reset streak to +1/-1 based on current V3
        else:
            result_df.iloc[i, result_df.columns.get_loc('V3_STREAK')] = 1 if curr_v3 > 0 else -1


def getV3StreakSignal(result_df: pd.DataFrame, all_columns: list[str]):
    prerequisites = ['V3', 'V3_STREAK']
    check_prerequisites(prerequisites, all_columns)
    result_df['V3_STREAK_SIGNAL'] = result_df['V3_STREAK'].apply(
        lambda x: 1 if x > 64 else (0 if x < -64 else 0.5)
    )

def getVolatilityBand(result_df: pd.DataFrame, all_columns: list[str]):
    prerequisites = ['SMA100', 'SMA5']
    multiplier = 3
    
    check_prerequisites(prerequisites, all_columns)

    # Calculate daily returns (close to close percentage change)
    result_df['returns'] = result_df['close'].pct_change()

    # Calculate CPV (standard deviation of returns) over 200 periods
    result_df['cpv'] = result_df['returns'].rolling(window=200).std() * 100

    # Smooth CPV with 50-period SMA
    result_df['smoothed_cpv'] = result_df['cpv'].rolling(window=5).mean()

    # Calculate bands using SMA60 as base
    result_df['VOL_BAND_UPPER'] = result_df['SMA100'] * (100 + (result_df['smoothed_cpv'] * multiplier)) / 100
    result_df['VOL_BAND_LOWER'] = result_df['SMA100'] * (100 - (result_df['smoothed_cpv'] * multiplier)) / 100


def getDonchianChannels(result_df: pd.DataFrame, all_columns: list[str], period: int = 20):
    prerequisites = ['close']
    
    check_prerequisites(prerequisites, all_columns)
    # Calculate upper and lower bands based on highest high and lowest low over period
    result_df['DC_UPPER'] = result_df['close'].rolling(window=period).max()
    result_df['DC_LOWER'] = result_df['close'].rolling(window=period).min()
    result_df['DC_MIDDLE'] = (result_df['DC_UPPER'] + result_df['DC_LOWER']) / 2
    # Get previous DC value
    result_df['PREVIOUS_DC_UPPER'] = result_df['DC_UPPER'].shift(1)
    result_df['PREVIOUS_DC_LOWER'] = result_df['DC_LOWER'].shift(1)

def getDonchianChannelsSMA(result_df: pd.DataFrame, all_columns: list[str], period: int = 20, window=5):
    prerequisites = ['SMA5']
    
    check_prerequisites(prerequisites, all_columns)
    # Calculate upper and lower bands based on highest high and lowest low over period
    result_df['DC_UPPER'] = result_df['SMA5'].rolling(window=period).max()
    result_df['DC_LOWER'] = result_df['SMA5'].rolling(window=period).min()
    result_df['DC_MIDDLE'] = (result_df['DC_UPPER'] + result_df['DC_LOWER']) / 2
    # Get previous DC value
    result_df['PREVIOUS_DC_UPPER'] = result_df['DC_UPPER'].shift(1)
    result_df['PREVIOUS_DC_LOWER'] = result_df['DC_LOWER'].shift(1)
    # Check if any previous DC previous
    result_df['DC_UPPER_CHANGES'] = (result_df['DC_UPPER'] - result_df['PREVIOUS_DC_UPPER']).apply(lambda x: 0 if x >= 0 else 1)
    result_df['DC_LOWER_CHANGES'] = (result_df['DC_LOWER'] - result_df['PREVIOUS_DC_LOWER']).apply(lambda x: 0 if x <= 0 else 1)
    # Check if each previous 5 row DC_UPPER_CHANGES and DC_UPPER_CHANGES has 1 value
    result_df['DC_UPPER_CHANGES_5_ROW'] = result_df['DC_UPPER_CHANGES'].rolling(window=window).sum().apply(lambda x: 1 if x > 0 else 0)
    result_df['DC_LOWER_CHANGES_5_ROW'] = result_df['DC_LOWER_CHANGES'].rolling(window=window).sum().apply(lambda x: 1 if x > 0 else 0)