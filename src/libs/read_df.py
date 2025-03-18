import os
from typing import List
import zipfile
import pandas as pd 

def read2df(symbols, freqs, market_type="spot", timestamp=""):
    # List to store individual DataFrames
    dfs = []
    if market_type != 'spot':
        market_type = f'futures/{market_type}'

    if symbols is None:
        klines_path = os.path.abspath(f'./binance-public-data/python/data/{market_type}/monthly/klines/')
        symbols = [folder for folder in os.listdir(klines_path)]

    # Loop through each freq
    for freq in freqs.keys():
        # Loop through each symbol
        rawdfs = []
        
        for symbol in symbols:
            if timestamp != "" :
                time = timestamp
                directory = os.path.abspath(f'./binance-public-data/python/data/{market_type}/monthly/klines/{symbol}/{freq}/{time}')
            else:
                directory = directory = os.path.abspath(f'./binance-public-data/python/data/{market_type}/monthly/klines/{symbol}/{freq}')
            

            # Loop through each zip file in the directory
            for file_name in os.listdir(directory):
                if file_name.endswith('.zip'):
                    zip_file_path = os.path.join(directory, file_name)
                    
                    if os.path.exists(zip_file_path):
                        with zipfile.ZipFile(os.path.join(directory, file_name), 'r') as zip_ref:
                            # only one CSV file in each zip archive
                            csv_file = zip_ref.namelist()[0]
                            with zip_ref.open(csv_file) as csv_fp:
                                # Read the CSV data into a DataFrame
                                temp_df = pd.read_csv(csv_fp, header=None)
                                temp_df.columns = [
                                    'open_time', 'open', 'high', 'low', 'close', 'volume', 
                                    'close_time', 'quote_asset_volume', 'number_of_trades', 
                                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                                ]
                                temp_df = temp_df.rename(columns={"close_time": "time"})
                                temp_df['tic'] = symbol
                                temp_df['itvl'] = freq

                                rawdfs.append(temp_df[['time', 'open', 'high', 'low', 'close', 'volume', 'tic', 'itvl']])


        # Concatenate all symbols into a single DataFrame
        rawdf = pd.concat(rawdfs, ignore_index=True)
        # Count the number of unique 'tic' values per date
        tic_counts = rawdf.groupby('time')['tic'].nunique()

        # Filter the DataFrame to keep only rows where all 'tic' values participate
        df = rawdf[rawdf['time'].isin(tic_counts[tic_counts == len(rawdf['tic'].unique())].index)]
        # Only wanted columns
        df = df[['time', 'open', 'high', 'low', 'close', 'volume', 'tic', 'itvl']]
        df = df[df['time']!='close_time']

        df['datetime'] = pd.to_datetime(df['time'], unit='ms',  errors = 'coerce')
        # Drop rows with NaT in datetime
        df = df.dropna(subset=['datetime'])
        numeric_columns = df.columns.difference(['datetime', 'tic', 'itvl'])
        
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        df = df.sort_values(['time', 'tic', 'itvl'],ignore_index=True,)
        df = df.drop_duplicates()
        df = df.dropna()
        
        dfs.append(df)
    
    return dfs

def split_into_train_test(dfs: List[pd.DataFrame], start_date: str, trade_date: str, end_date: str):
    trains, tests, combined = [], [], []
    for i in range(len(dfs)):
        trains.append(dfs[i][(dfs[i]['datetime'] > start_date) & (dfs[i]['datetime'] < trade_date)].reset_index(drop=True))
        tests.append(dfs[i][(dfs[i]['datetime'] >= trade_date) & (dfs[i]['datetime'] < end_date)].reset_index(drop=True))
        combined.append(dfs[i][(dfs[i]['datetime'] >= start_date) & (dfs[i]['datetime'] < end_date)].reset_index(drop=True))

    return trains, tests, combined
    