# Databricks notebook source

# COMMAND ----------
# MAGIC %md
# MAGIC # S&P 500 Prediction Analysis using Toto
# MAGIC 
# MAGIC This notebook walks through the process of fetching S&P 500 data, preparing it for the Toto model, and generating predictions. We'll examine the data and results at each important step.

# COMMAND ----------

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prepare_stock_data import fetch_sp500_data, prepare_for_toto
from toto.model.toto import Toto
from toto.inference.forecaster import TotoForecaster

# Set plotting style
plt.style.use('seaborn')
sns.set_palette("husl")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Data Collection
# MAGIC 
# MAGIC First, let's fetch the S&P 500 data and examine its basic properties.

# COMMAND ----------

# Fetch data
data = fetch_sp500_data()

print("Data shape:", data.shape)
print("\nDate range:")
print(f"Start: {data.index[0]}")
print(f"End: {data.index[-1]}")

# Display first few rows
display(data.head())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Data Visualization
# MAGIC 
# MAGIC Let's visualize the key features of our dataset.

# COMMAND ----------

def plot_features(data, columns, rows=3):
    cols = (len(columns) + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        axes[i].plot(data.index, data[col])
        axes[i].set_title(col)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    display(plt.gcf())
    plt.close()

# Plot all features
plot_features(data, data.columns)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Data Normalization
# MAGIC 
# MAGIC Let's examine how our robust normalization affects the data distribution.

# COMMAND ----------

def plot_distributions(data, normalized_data):
    fig, axes = plt.subplots(2, len(data.columns), figsize=(20, 8))
    
    for i, col in enumerate(data.columns):
        # Original distribution
        sns.histplot(data[col], ax=axes[0, i], kde=True)
        axes[0, i].set_title(f'Original {col}')
        
        # Normalized distribution
        sns.histplot(normalized_data[col], ax=axes[1, i], kde=True)
        axes[1, i].set_title(f'Normalized {col}')
    
    plt.tight_layout()
    display(plt.gcf())
    plt.close()

# Get normalized data (before tensor conversion)
normalized_data = {}
for column in data.columns:
    series = data[column].values
    
    if column in ['Volume', 'Volume_MA']:
        series = np.log1p(series)
    elif column == 'Volatility':
        series = np.maximum(series, 0)
    
    q1 = np.percentile(series[~np.isnan(series)], 25)
    q3 = np.percentile(series[~np.isnan(series)], 75)
    iqr = q3 - q1
    
    if iqr > 0:
        median = np.median(series[~np.isnan(series)])
        normalized_data[column] = (series - median) / (iqr + 1e-8)
    else:
        mean = np.mean(series[~np.isnan(series)])
        std = np.std(series[~np.isnan(series)])
        normalized_data[column] = (series - mean) / (std + 1e-8)
    
    normalized_data[column] = np.clip(normalized_data[column], -10, 10)

# Plot distributions
plot_distributions(data, normalized_data)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Model Preparation and Prediction
# MAGIC 
# MAGIC Now let's prepare the data for Toto and generate predictions.

# COMMAND ----------

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Prepare data for Toto
toto_input = prepare_for_toto(data, device=device)

# Load model
toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0').to(device)
toto.compile()

# Create forecaster
forecaster = TotoForecaster(toto.model)

# Generate forecasts
prediction_hours = 168  # 1 week
forecast = forecaster.forecast(
    toto_input,
    prediction_length=prediction_hours,
    num_samples=256,
    samples_per_batch=256,
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Visualization of Predictions
# MAGIC 
# MAGIC Let's visualize the predictions with confidence intervals for each feature.

# COMMAND ----------

def plot_predictions(data, forecast, feature_idx, feature_name):
    # Get the last timestamp from historical data
    last_timestamp = data.index[-1]
    
    # Create future timestamps
    future_timestamps = pd.date_range(
        start=last_timestamp,
        periods=forecast.median.shape[-1] + 1,
        freq='H'
    )[1:]
    
    plt.figure(figsize=(15, 7))
    
    # Plot historical data
    plt.plot(data.index[-168:], data[feature_name].values[-168:], 
             label='Historical', color='blue')
    
    # Plot predictions
    median_pred = forecast.median[0, feature_idx].cpu().numpy()
    lower_pred = forecast.quantile(0.1)[0, feature_idx].cpu().numpy()
    upper_pred = forecast.quantile(0.9)[0, feature_idx].cpu().numpy()
    
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
    display(plt.gcf())
    plt.close()

# Plot predictions for each feature
for idx, feature in enumerate(data.columns):
    plot_predictions(data, forecast, idx, feature)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Analysis of Prediction Quality
# MAGIC 
# MAGIC Let's analyze the quality of our predictions by looking at the distribution of samples and uncertainty.

# COMMAND ----------

def analyze_prediction_uncertainty(forecast, feature_idx, feature_name):
    # Get all samples for the feature
    samples = forecast.samples[:, 0, feature_idx].cpu().numpy()
    
    # Calculate uncertainty metrics
    std_dev = np.std(samples, axis=0)
    mean = np.mean(samples, axis=0)
    
    plt.figure(figsize=(15, 5))
    
    # Plot standard deviation over time
    plt.plot(std_dev, label='Standard Deviation')
    plt.title(f'Prediction Uncertainty Over Time - {feature_name}')
    plt.xlabel('Hours into Future')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.grid(True)
    display(plt.gcf())
    plt.close()

# Analyze uncertainty for Close price
analyze_prediction_uncertainty(forecast, 3, 'Close')

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Save Results to Delta Table (Databricks-specific)
# MAGIC 
# MAGIC Let's save our predictions to a Delta table for future reference.

# COMMAND ----------

# Create a DataFrame with predictions
def create_predictions_df(data, forecast, feature_name, feature_idx):
    last_timestamp = data.index[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp,
        periods=forecast.median.shape[-1] + 1,
        freq='H'
    )[1:]
    
    predictions_df = pd.DataFrame({
        'timestamp': future_timestamps,
        'feature': feature_name,
        'median_prediction': forecast.median[0, feature_idx].cpu().numpy(),
        'lower_bound': forecast.quantile(0.1)[0, feature_idx].cpu().numpy(),
        'upper_bound': forecast.quantile(0.9)[0, feature_idx].cpu().numpy()
    })
    
    return predictions_df

# Combine predictions for all features
all_predictions = []
for idx, feature in enumerate(data.columns):
    feature_predictions = create_predictions_df(data, forecast, feature, idx)
    all_predictions.append(feature_predictions)

predictions_df = pd.concat(all_predictions, ignore_index=True)

# Convert to Spark DataFrame and save as Delta table
spark_predictions = spark.createDataFrame(predictions_df)
spark_predictions.write.format("delta").mode("overwrite").saveAsTable("sp500_predictions")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Query Saved Predictions (Databricks-specific)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Example query to view predictions
# MAGIC SELECT *
# MAGIC FROM sp500_predictions
# MAGIC WHERE feature = 'Close'
# MAGIC ORDER BY timestamp; 