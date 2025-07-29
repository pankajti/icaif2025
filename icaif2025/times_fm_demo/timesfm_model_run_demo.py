from icaif2025.time_series_model.times_fm_factory import get_timesfm_model
import numpy as np # Added for data generation
import timesfm
import matplotlib.pyplot as plt # Added for plotting

# --- 1. Import necessary classes explicitly ---
# It's good practice to list all classes you use from the module.
from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint


def main():
    historical_data = get_data()
    forecast_input = [historical_data]
    print("\nPerforming forecast...")
    frequency_input = np.zeros((len(forecast_input)))
    # Get TimesFM v1 model
    model_v1 = get_timesfm_model("v1")
    # Get TimesFM v2 model
    #model_v2 = get_timesfm_model("v2")

    point_forecast, _ = model_v1.forecast(
        forecast_input,
        freq=frequency_input,
    )
    predicted_future = point_forecast[0]

    print("Forecast complete!")
    print(f"Shape of predicted future: {predicted_future.shape}")

    plot_forecast(historical_data, predicted_future)


def get_data():
    context_length = 256
    horizon_length = 64
    total_points = context_length + horizon_length * 2
    time = np.arange(total_points)
    data = 5 * np.sin(time / 10) + 2 * np.cos(time / 5) + np.random.normal(0, 0.5, total_points)
    historical_data = data[:context_length]
    return historical_data



def plot_forecast(historical_data, predicted_future):
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(historical_data)), historical_data, label='Historical Data', color='blue')
    plt.plot(np.arange(len(historical_data), len(historical_data) + len(predicted_future)), predicted_future,
             label='TimesFM Forecast', color='red', linestyle='--')
    plt.title('TimesFM Forecasting Example')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()


