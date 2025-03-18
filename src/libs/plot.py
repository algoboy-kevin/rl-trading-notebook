import pandas as pd
from matplotlib import pyplot as plt 

def plot_close_price(symbol, data):
    # Convert 'datetime' column to datetime type for better plotting
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # Set 'datetime' as the index of the dataframe
    data.set_index('datetime', inplace=True)
    
    # Plotting the 'close' price
    plt.figure(figsize=(10, 5))
    plt.plot(data['close'], label='Close Price')
    plt.title(f'Close Price {symbol}')
    plt.xlabel('Time')
    plt.ylabel(f'Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()