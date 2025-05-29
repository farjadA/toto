import yfinance as yf
import pandas as pd
import torch
from datetime import datetime, timedelta
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to the Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from toto.data.util.dataset import MaskedTimeseries

def fetch_sp500_data(lookback_days=700):
    """
    Fetch S&P 500 data and relevant features
    Returns hourly data for the last 700 days (Yahoo Finance limit is 730 days for hourly data)
    """
    # Get SPY (S&P 500 ETF) data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    try:
        spy = yf.download('^GSPC', start=start_date, end=end_date, interval='1h')
        
        if spy.empty:
            raise ValueError("No data received from Yahoo Finance")
            
        # Calculate additional features
        spy['Returns'] = spy['Close'].pct_change()
        spy['Volume_MA'] = spy['Volume'].rolling(window=24).mean()  # 24-hour moving average
        spy['Price_MA_short'] = spy['Close'].rolling(window=24).mean()
        spy['Price_MA_long'] = spy['Close'].rolling(window=72).mean()
        spy['Volatility'] = spy['Returns'].rolling(window=24).std()
        
        # Forward fill any NaN values (using newer pandas method)
        spy = spy.ffill()  # Changed from fillna(method='ffill') to ffill()
        
        # Get the relevant columns
        features = [
            'Open', 'High', 'Low', 'Close', 
            'Volume', 'Volume_MA',
            'Price_MA_short', 'Price_MA_long',
            'Volatility'
        ]
        
        if spy[features].empty:
            raise ValueError("No data available after processing")
            
        return spy[features]
        
    except Exception as e:
        print(f"Error fetching S&P 500 data: {str(e)}")
        raise

def prepare_for_toto(data, device='cpu'):
    """
    Prepare the data in the format Toto expects with robust normalization
    """
    # Normalize each feature
    normalized_data = {}
    for column in data.columns:
        series = data[column].values
        
        # Handle special cases for different features
        if column in ['Volume', 'Volume_MA']:
            # Log transform volume data to handle large numbers
            series = np.log1p(series)
        elif column == 'Volatility':
            # Ensure volatility is non-negative
            series = np.maximum(series, 0)
        
        # Robust normalization
        q1 = np.percentile(series[~np.isnan(series)], 25)
        q3 = np.percentile(series[~np.isnan(series)], 75)
        iqr = q3 - q1
        
        if iqr > 0:
            # Use robust statistics for normalization
            median = np.median(series[~np.isnan(series)])
            normalized_data[column] = (series - median) / (iqr + 1e-8)
        else:
            # Fallback to simple standardization with safeguards
            mean = np.mean(series[~np.isnan(series)])
            std = np.std(series[~np.isnan(series)])
            normalized_data[column] = (series - mean) / (std + 1e-8)
        
        # Clip extreme values
        normalized_data[column] = np.clip(normalized_data[column], -10, 10)
    
    # Convert to tensor format and move to device
    series_tensor = torch.tensor(
        np.array([normalized_data[col] for col in data.columns]),
        dtype=torch.float32
    ).to(device)
    
    # Validate tensor for any remaining issues
    if torch.isnan(series_tensor).any() or torch.isinf(series_tensor).any():
        raise ValueError("Invalid values detected in normalized data")
    
    # Create timestamp tensor (seconds since start)
    timestamps = pd.to_datetime(data.index)
    timestamp_seconds = torch.tensor(
        [(t - timestamps[0]).total_seconds() for t in timestamps],
        dtype=torch.float32
    ).to(device)
    timestamp_seconds = timestamp_seconds.unsqueeze(0).repeat(len(data.columns), 1)
    
    # Time interval is 1 hour = 3600 seconds
    time_interval_seconds = torch.full((len(data.columns),), 3600, dtype=torch.float32).to(device)
    
    # Create MaskedTimeseries object
    masked_series = MaskedTimeseries(
        series=series_tensor,
        padding_mask=torch.full_like(series_tensor, True, dtype=torch.bool),
        id_mask=torch.zeros_like(series_tensor),
        timestamp_seconds=timestamp_seconds,
        time_interval_seconds=time_interval_seconds,
    )
    
    return masked_series

if __name__ == "__main__":
    # Fetch data
    data = fetch_sp500_data()
    print("Data shape:", data.shape)
    
    # Prepare for Toto
    toto_input = prepare_for_toto(data)
    print("Prepared tensor shape:", toto_input.series.shape) 