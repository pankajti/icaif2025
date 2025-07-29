from icaif2025.time_series_model.times_fm_factory import get_timesfm_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from icaif2025.stock_market_predictor.stock_data_provider import get_stock_data
from datetime import datetime, timedelta

def main():
    # Assuming get_stock_data returns a pandas Series with a DatetimeIndex
    ticker = 'NVDA'
    historical_data_series = get_stock_data(ticker)

    # Ensure historical_data_series is a pandas Series with DatetimeIndex
    if not isinstance(historical_data_series, pd.Series) or not isinstance(historical_data_series.index, pd.DatetimeIndex):
        print("Warning: historical_data_series is not a pandas Series with a DatetimeIndex. Attempting conversion.")
        if isinstance(historical_data_series, dict) and ticker in historical_data_series:
            values = historical_data_series[ticker].values
        elif isinstance(historical_data_series, pd.DataFrame) and ticker in historical_data_series.columns:
            values = historical_data_series[ticker].values
        else:
            values = historical_data_series

        # Create a dummy DatetimeIndex for demonstration if original data has no dates
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(values), freq='D'))
        historical_data_series = pd.Series(values, index=dates)

    # --- NEW: Splitting data into training and validation ---
    # Let's use the last 30 days of historical data as the "actuals" for comparison
    forecast_horizon_days = 30
    if len(historical_data_series) <= forecast_horizon_days:
        raise ValueError(f"Historical data is too short to split for a {forecast_horizon_days}-day forecast horizon.")

    training_data = historical_data_series.iloc[:-forecast_horizon_days]
    actual_future_data = historical_data_series.iloc[-forecast_horizon_days:] # These are the "actuals" we want to compare against

    # Predict using only the training data
    predicted_future_values, forecast_start_date = get_predicted_data(training_data, forecast_horizon_days)

    # Define a time window for plotting
    # This example plots the period for which we have both forecasts and actuals
    # Plus some preceding historical data for context
    end_plot_date = actual_future_data.index[-1] # End of the actual forecast period
    start_plot_date = training_data.index[-1] - timedelta(days=60) # Start 60 days before forecast began for context

    plot_forecast(training_data, actual_future_data, predicted_future_values, forecast_start_date, start_plot_date, end_plot_date)


def get_predicted_data(training_data: pd.Series, forecast_horizon: int):
    """
    Predicts future data using TimesFM model.
    Returns predicted values and the start date of the forecast.
    """
    forecast_input = [training_data.values]
    print("\nPerforming forecast...")
    # Assuming the TimesFM model can take a forecast_horizon parameter.
    # If not, you might need to adjust the model's internal prediction length or loop its forecast method.
    frequency_input = np.zeros((len(forecast_input)))

    model = get_timesfm_model("v2")

    # Assuming model_v1.forecast can take an argument for the number of steps to forecast
    # If not, you'd need to adapt how TimesFM generates its output or iterate.
    # For simplicity, assuming it forecasts a default length, and we'll trim/pad if necessary.
    point_forecast, _ = model.forecast(
        forecast_input,
        freq=frequency_input,
    )
    predicted_full_length = point_forecast[0]

    # Ensure predicted_full_length matches forecast_horizon.
    # If TimesFM predicts a fixed number of steps, you might need to slice it.
    if len(predicted_full_length) < forecast_horizon:
        # Pad with NaNs or raise error if model doesn't predict enough
        print(f"Warning: TimesFM model predicted {len(predicted_full_length)} steps, but {forecast_horizon} were requested.")
        predicted_future = np.pad(predicted_full_length, (0, forecast_horizon - len(predicted_full_length)), 'constant', constant_values=np.nan)
    elif len(predicted_full_length) > forecast_horizon:
        # Trim if model predicted too many steps
        predicted_future = predicted_full_length[:forecast_horizon]
    else:
        predicted_future = predicted_full_length

    print("Forecast complete!")
    print(f"Shape of predicted future: {predicted_future.shape}")

    # Determine the start date of the forecast (the day *after* the last training data point)
    forecast_start_date = training_data.index[-1] + timedelta(days=1)

    return predicted_future, forecast_start_date


def plot_forecast(training_data_series: pd.Series, actual_future_data_series: pd.Series,
                  predicted_future_values: np.ndarray, forecast_start_date: datetime,
                  plot_start_date: datetime, plot_end_date: datetime):
    """
    Plots training data, actual future data, and forecasted values within a specified time window.

    Args:
        training_data_series (pd.Series): Historical data used for training with DatetimeIndex.
        actual_future_data_series (pd.Series): Actual observed values for the forecast period.
        predicted_future_values (np.ndarray): Array of predicted future values.
        forecast_start_date (datetime): The actual start date of the forecast.
        plot_start_date (datetime): The start date for the plotting window.
        plot_end_date (datetime): The end date for the plotting window.
    """
    plt.figure(figsize=(14, 7))

    # 1. Prepare forecasted data with dates
    predicted_dates = pd.date_range(start=forecast_start_date, periods=len(predicted_future_values), freq='D')
    predicted_series = pd.Series(predicted_future_values, index=predicted_dates)

    # 2. Filter historical data (training) to the plotting window
    plot_training = training_data_series[
        (training_data_series.index >= plot_start_date) &
        (training_data_series.index < forecast_start_date) # Plot training data up to the forecast start
    ]

    # 3. Filter actual future data to the plotting window
    plot_actual_future = actual_future_data_series[
        (actual_future_data_series.index >= plot_start_date) &
        (actual_future_data_series.index <= plot_end_date)
    ]

    # 4. Filter predicted data to the plotting window
    plot_predicted_future = predicted_series[
        (predicted_series.index >= plot_start_date) &
        (predicted_series.index <= plot_end_date)
    ]

    # Plotting training data
    if not plot_training.empty:
        plt.plot(plot_training.index, plot_training.values, label='Training Data', color='blue', linewidth=2)
    else:
        print(f"No training data found in the plotting window: {plot_start_date} to {forecast_start_date - timedelta(days=1)}")

    # Plotting actual future data (the ground truth for the forecast period)
    if not plot_actual_future.empty:
        plt.plot(plot_actual_future.index, plot_actual_future.values, label='Actual Future Data', color='green', linewidth=2, linestyle='-')
        # Add a marker for the transition point if relevant
        if not plot_training.empty and not plot_actual_future.empty:
             plt.axvline(x=forecast_start_date, color='grey', linestyle=':', label='Forecast Start')
    else:
        print(f"No actual future data found in the plotting window: {plot_start_date} to {plot_end_date}")

    # Plotting predicted future data
    if not plot_predicted_future.empty:
        # Connect the last training point to the first forecast point if available for visual continuity
        if not plot_training.empty:
            last_training_date = plot_training.index[-1]
            last_training_value = plot_training.iloc[-1]
            first_forecast_date = plot_predicted_future.index[0]
            first_forecast_value = plot_predicted_future.iloc[0]
            # Plot the line connecting the training and forecast
            plt.plot([last_training_date, first_forecast_date], [last_training_value, first_forecast_value],
                     color='red', linestyle='--', linewidth=2)

        plt.plot(plot_predicted_future.index, plot_predicted_future.values,
                 label='TimesFM Forecast', color='red', linestyle='--', linewidth=2)
    else:
        print(f"No forecasted data found in the plotting window: {plot_start_date} to {plot_end_date}")

    plt.title(f'TimesFM Forecasting: Actual vs. Predicted ({plot_start_date.strftime("%Y-%m-%d")} to {plot_end_date.strftime("%Y-%m-%d")})')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45) # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()