from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def create_sideways_data(filename: str, reference_df: pd.DataFrame, price_starts: float, len_data: int):
    # Load the original data
    reference_df = reference_df[['datetime', 'close']]  # Select only time and close columns
    reference_df.set_index('datetime', inplace=True)

    # Select a random slice of 10000 rows
    start_idx = np.random.randint(0, len(reference_df) - len_data)
    df_slice = reference_df.iloc[start_idx:start_idx+len_data]

    # Calculate returns and rolling volatility
    returns = df_slice['close'].pct_change().dropna()
    rolling_volatility = returns.rolling(window=40).std().bfill()

    # Generate more stationary price movement with upper and lower bounds
    n = len(df_slice)
    np.random.seed(42)  # for reproducibility
    epsilon = np.random.normal(0, 1, n)
    sideways_series = np.zeros(n)
    sideways_series[0] = 1 * price_starts # Start at 1

    # Define upper and lower bounds
    lower_bound = 0.6 * price_starts
    upper_bound = 1.4 * price_starts

    # Parameters for mean reversion
    mean = 1 * price_starts
    reversion_strength = 0.05

    for t in range(1, n):
        # Use current volatility instead of average
        current_volatility = rolling_volatility.iloc[t-1]
        
        # Generate price movement
        sideways_series[t] = sideways_series[t-1] * (1 + current_volatility * epsilon[t])
        
        # Apply mean reversion
        sideways_series[t] += reversion_strength * (mean - sideways_series[t-1])
        
        # Enforce bounds
        sideways_series[t] = np.clip(sideways_series[t], lower_bound, upper_bound)

    # Create a new DataFrame with the stationary data
    sideways_df = pd.DataFrame(index=df_slice.index, columns=['close'])
    sideways_df['close'] = sideways_series

    mean_price = df_slice['close'].mean()
    min_pct = (df_slice['close'].min() - mean_price) / mean_price * price_starts
    max_pct = (df_slice['close'].max() - mean_price) / mean_price * price_starts

    # Set the feature range based on the calculated percentages, centered around 100
    scaler = MinMaxScaler(feature_range=(price_starts + min_pct, price_starts + max_pct))
    reference_df_scaled = pd.DataFrame(scaler.fit_transform(df_slice), columns=['close'], index=df_slice.index)
    
    # Save the stationary data
    sideways_df.to_csv(f'./data/generated/{filename}.csv')

    # Plot original (scaled) vs stationary data
    plt.figure(figsize=(12, 6))
    plt.plot(reference_df_scaled.index, reference_df_scaled['close'], label='Original (Scaled)', alpha=0.7)
    plt.plot(sideways_df.index, sideways_df['close'], label='Sideways', alpha=0.7)
    plt.title('Scaled Original vs Synthetic Sideways Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()

    # Add horizontal lines for upper and lower bounds
    plt.axhline(y=upper_bound, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=lower_bound, color='r', linestyle='--', alpha=0.5)

    # Show data
    plt.show()
