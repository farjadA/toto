import torch
from prepare_stock_data import fetch_sp500_data, prepare_for_toto, denormalize_predictions
from toto.model.toto import Toto
from toto.inference.forecaster import TotoForecaster
import pandas as pd
import matplotlib.pyplot as plt

def predict_sp500(prediction_hours=168):  # Default to 1 week prediction
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("Fetching S&P 500 data...")
    data = fetch_sp500_data()
    toto_input, normalization_params = prepare_for_toto(data, device=device)
    
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
    
    # Get predictions and denormalize them
    median_pred = forecast.median
    lower_pred = forecast.quantile(0.1)
    upper_pred = forecast.quantile(0.9)
    
    # Denormalize predictions to get actual values
    denorm_median = denormalize_predictions(median_pred, normalization_params, data.columns)
    denorm_lower = denormalize_predictions(lower_pred, normalization_params, data.columns)
    denorm_upper = denormalize_predictions(upper_pred, normalization_params, data.columns)
    
    # Create timestamps for predictions
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=prediction_hours + 1, freq='H')[1:]
    
    # Create a results dictionary
    results = {
        'dates': future_dates,
        'predictions': {
            'Close': {
                'median': denorm_median[0, data.columns.get_loc('Close'), :],
                'lower': denorm_lower[0, data.columns.get_loc('Close'), :],
                'upper': denorm_upper[0, data.columns.get_loc('Close'), :]
            },
            'Volume': {
                'median': denorm_median[0, data.columns.get_loc('Volume'), :],
                'lower': denorm_lower[0, data.columns.get_loc('Volume'), :],
                'upper': denorm_upper[0, data.columns.get_loc('Volume'), :]
            }
        },
        'historical': {
            'dates': data.index[-168:],  # Last week of historical data
            'Close': data['Close'][-168:],
            'Volume': data['Volume'][-168:]
        }
    }
    
    # Plot predictions
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title('S&P 500 Price Prediction')
    plt.plot(results['historical']['dates'], results['historical']['Close'], 
             label='Historical', color='blue')
    plt.plot(results['dates'], results['predictions']['Close']['median'], 
             label='Prediction', color='red', linestyle='--')
    plt.fill_between(results['dates'], 
                     results['predictions']['Close']['lower'],
                     results['predictions']['Close']['upper'],
                     color='red', alpha=0.2, label='90% Confidence')
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.title('S&P 500 Volume Prediction')
    plt.plot(results['historical']['dates'], results['historical']['Volume'], 
             label='Historical', color='blue')
    plt.plot(results['dates'], results['predictions']['Volume']['median'], 
             label='Prediction', color='red', linestyle='--')
    plt.fill_between(results['dates'], 
                     results['predictions']['Volume']['lower'],
                     results['predictions']['Volume']['upper'],
                     color='red', alpha=0.2, label='90% Confidence')
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print next day's predicted prices
    print("\nPredicted S&P 500 prices for next 24 hours:")
    for i in range(24):
        print(f"{results['dates'][i].strftime('%Y-%m-%d %H:%M')}: "
              f"${results['predictions']['Close']['median'][i]:.2f} "
              f"(90% CI: ${results['predictions']['Close']['lower'][i]:.2f} - "
              f"${results['predictions']['Close']['upper'][i]:.2f})")
    
    return results

if __name__ == "__main__":
    results = predict_sp500() 