
import subprocess

def download_klines(symbols=None, freqs=None, start_date=None, end_date=None, market_type='spot', skip_daily=1):
    """
    Download kline data using subprocess instead of Jupyter's shell command
    
    Args:
        symbols (list): List of trading symbols. If None, downloads all available
        freqs (dict): Dictionary of frequencies
        start_date (str): Start date for data download
        end_date (str): End date for data download
        market_type (str): Market type (default: 'spot')
        skip_daily (int): Skip daily option (default: 1)
    """
    # data_path = os.path.abspath('./data')
    base_cmd = ["python", "./binance-public-data/python/download-kline.py"]
    
    if symbols:
        base_cmd.extend(["-s"] + symbols)
    
    if freqs:
        base_cmd.extend(["-i"] + list(freqs.keys()))
    
    if start_date:
        base_cmd.extend(["-startDate", start_date])
    
    if end_date:
        base_cmd.extend(["-endDate", end_date])

    base_cmd.extend(["-t", market_type])
    base_cmd.extend(["-skip-daily", str(skip_daily)])
    
    # Execute the command
    subprocess.run(base_cmd)