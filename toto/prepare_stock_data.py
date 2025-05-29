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
            
        # Clean up the MultiIndex
        # First reset the columns to get rid of the MultiIndex
        spy.columns = [col[0] for col in spy.columns]
        
        # Calculate additional features
        spy['Returns'] = spy['Close'].pct_change()
        spy['Volume_MA'] = spy['Volume'].rolling(window=24).mean()  # 24-hour moving average
        spy['Price_MA_short'] = spy['Close'].rolling(window=24).mean()
        spy['Price_MA_long'] = spy['Close'].rolling(window=72).mean()
        spy['Volatility'] = spy['Returns'].rolling(window=24).std()
        
        # Forward fill any NaN values (using newer pandas method)
        spy = spy.ffill()  # Changed from fillna(method='ffill') to ffill()
        
        # Reset index to make datetime a column
        spy = spy.reset_index()
        spy = spy.rename(columns={'index': 'Datetime'})
        
        # Get the relevant columns
        features = [
            'Datetime',
            'Open', 
            'High', 
            'Low', 
            'Close', 
            'Volume', 
            'Volume_MA',
            'Price_MA_short', 
            'Price_MA_long',
            'Volatility'
        ]
        
        if spy[features].empty:
            raise ValueError("No data available after processing")
            
        # Set Datetime as index again but now it's clean
        spy = spy[features].set_index('Datetime')
            
        return spy
        
    except Exception as e:
        print(f"Error fetching S&P 500 data: {str(e)}")
        raise

def prepare_for_toto(data, device='cpu'):
    """
    Prepare the data in the format Toto expects with robust normalization
    """
    # First, check for any NaN or infinite values in the input data
    if data.isnull().any().any():
        # Forward fill, then backward fill any remaining NaNs
        data = data.ffill().bfill()
        
    # Store normalization parameters for later denormalization
    normalization_params = {}
    
    # Normalize each feature
    normalized_data = {}
    for column in data.columns:
        series = data[column].values
        
        # Handle special cases for different features
        if column in ['Volume', 'Volume_MA']:
            # Log transform volume data to handle large numbers
            # Add small constant to avoid log(0)
            series = np.log1p(series)
        elif column == 'Volatility':
            # Ensure volatility is non-negative
            series = np.maximum(series, 0)
        
        # Robust normalization with additional checks
        non_nan_mask = ~np.isnan(series)
        if non_nan_mask.any():  # Only proceed if we have valid data
            q1 = np.percentile(series[non_nan_mask], 25)
            q3 = np.percentile(series[non_nan_mask], 75)
            iqr = q3 - q1
            
            if iqr > 0:
                # Use robust statistics for normalization
                median = np.median(series[non_nan_mask])
                normalized_data[column] = (series - median) / (iqr + 1e-8)
                # Store parameters
                normalization_params[column] = {
                    'method': 'robust',
                    'median': median,
                    'iqr': iqr,
                    'transform': 'log1p' if column in ['Volume', 'Volume_MA'] else None
                }
            else:
                # Fallback to simple standardization with safeguards
                mean = np.mean(series[non_nan_mask])
                std = np.std(series[non_nan_mask])
                if std > 0:
                    normalized_data[column] = (series - mean) / (std + 1e-8)
                    # Store parameters
                    normalization_params[column] = {
                        'method': 'standard',
                        'mean': mean,
                        'std': std,
                        'transform': 'log1p' if column in ['Volume', 'Volume_MA'] else None
                    }
                else:
                    # If std is 0, just center the data
                    normalized_data[column] = series - mean
                    # Store parameters
                    normalization_params[column] = {
                        'method': 'center',
                        'mean': mean,
                        'transform': 'log1p' if column in ['Volume', 'Volume_MA'] else None
                    }
        else:
            # If all values are NaN, set to zeros
            normalized_data[column] = np.zeros_like(series)
            normalization_params[column] = {
                'method': 'zero',
                'transform': 'log1p' if column in ['Volume', 'Volume_MA'] else None
            }
        
        # Clip extreme values
        normalized_data[column] = np.clip(normalized_data[column], -10, 10)
    
    # Convert to tensor format and move to device
    series_tensor = torch.tensor(
        np.array([normalized_data[col] for col in data.columns]),
        dtype=torch.float32
    ).to(device)
    
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
    
    # Final validation
    if torch.isnan(series_tensor).any() or torch.isinf(series_tensor).any():
        raise ValueError("Invalid values detected in normalized data")
    
    return masked_series, normalization_params

def denormalize_predictions(predictions, normalization_params, column_names):
    """
    Convert normalized predictions back to original scale
    
    Args:
        predictions: Tensor of shape [batch, variables, prediction_length] or [batch, variables, prediction_length, samples]
        normalization_params: Dictionary of normalization parameters for each variable
        column_names: List of column names in the same order as variables dimension
    
    Returns:
        Denormalized predictions in the same shape as input
    """
    # Convert to numpy if tensor
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    # Handle both point predictions and sample predictions
    original_shape = predictions.shape
    if len(original_shape) == 4:
        # Shape: [batch, variables, prediction_length, samples]
        predictions = predictions.transpose(0, 3, 1, 2)  # -> [batch, samples, variables, prediction_length]
        predictions = predictions.reshape(-1, original_shape[1], original_shape[2])  # -> [batch*samples, variables, prediction_length]
    
    denormalized = np.zeros_like(predictions)
    
    for i, column in enumerate(column_names):
        params = normalization_params[column]
        values = predictions[:, i, :]
        
        if params['method'] == 'robust':
            denormalized[:, i, :] = values * (params['iqr'] + 1e-8) + params['median']
        elif params['method'] == 'standard':
            denormalized[:, i, :] = values * (params['std'] + 1e-8) + params['mean']
        elif params['method'] == 'center':
            denormalized[:, i, :] = values + params['mean']
        # else method is 'zero', no denormalization needed
        
        # Handle inverse transforms
        if params['transform'] == 'log1p':
            denormalized[:, i, :] = np.expm1(denormalized[:, i, :])
    
    # Restore original shape if needed
    if len(original_shape) == 4:
        denormalized = denormalized.reshape(-1, original_shape[3], original_shape[1], original_shape[2])  # -> [batch, samples, variables, prediction_length]
        denormalized = denormalized.transpose(0, 2, 3, 1)  # -> [batch, variables, prediction_length, samples]
    
    return denormalized

if __name__ == "__main__":
    # Fetch data
    data = fetch_sp500_data()
    print("Data shape:", data.shape)
    
    # Prepare for Toto
    toto_input, normalization_params = prepare_for_toto(data)
    print("Prepared tensor shape:", toto_input.series.shape) 