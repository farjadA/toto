import torch
import pandas as pd
from prepare_stock_data import fetch_sp500_data, prepare_for_toto
from toto.model.toto import Toto
from toto.inference.forecaster import TotoForecaster
import matplotlib.pyplot as plt
import numpy as np

def predict_sp500(prediction_hours=168):  # Default to 1 week prediction
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("Fetching S&P 500 data...")
    data = fetch_sp500_data()
    toto_input = prepare_for_toto(data, device=device)
    
    # Load pre-trained Toto model
    print("Loading Toto model...")
    toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0').to(device)
    toto.compile()  # Optional: for speed improvement
    
    # Create forecaster
    forecaster = TotoForecaster(toto.model)
    
    # Generate forecasts
    print(f"Generating {prediction_hours} hour forecast...")
    forecast = forecaster.forecast(
        toto_input,
        prediction_length=prediction_hours,
        num_samples=256,
        samples_per_batch=256,
    )
    
    # Get predictions
    median_prediction = forecast.median
    lower_quantile = forecast.quantile(0.1)
    upper_quantile = forecast.quantile(0.9)
    
    return {
        'data': data,
        'median_prediction': median_prediction,
        'lower_quantile': lower_quantile,
        'upper_quantile': upper_quantile
    }

def plot_predictions(results, feature_idx=3):  # Default to Close price (idx 3)
    """Plot the predictions with confidence intervals"""
    data = results['data']
    feature_name = data.columns[feature_idx]
    
    # Get the last timestamp from historical data
    last_timestamp = data.index[-1]
    
    # Create future timestamps
    future_timestamps = pd.date_range(
        start=last_timestamp,
        periods=results['median_prediction'].shape[-1] + 1,
        freq='H'
    )[1:]
    
    # Plot
    plt.figure(figsize=(15, 7))
    
    # Plot historical data
    plt.plot(data.index[-168:], data[feature_name].values[-168:], 
             label='Historical', color='blue')
    
    # Plot predictions
    median_pred = results['median_prediction'][0, feature_idx].cpu().numpy()
    lower_pred = results['lower_quantile'][0, feature_idx].cpu().numpy()
    upper_pred = results['upper_quantile'][0, feature_idx].cpu().numpy()
    
    plt.plot(future_timestamps, median_pred, 
             label='Prediction', color='red', linestyle='--')
    plt.fill_between(future_timestamps, lower_pred, upper_pred, 
                     color='red', alpha=0.2, label='90% Confidence Interval')
    
    plt.title(f'S&P 500 {feature_name} Prediction')
    plt.xlabel('Date')
    plt.ylabel(feature_name)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('sp500_prediction.png')
    plt.close()

if __name__ == "__main__":
    # Make predictions
    results = predict_sp500()
    
    # Plot results
    plot_predictions(results)
    print("Predictions have been generated and saved to 'sp500_prediction.png'") 