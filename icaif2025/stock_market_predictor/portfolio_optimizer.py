import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timesfm # Requires 'pip install timesfm'
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import itertools
import collections
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- DUMMY STOCK DATA PROVIDER ---
# IMPORTANT: This is a DUMMY implementation for demonstration.
# In a real application, you MUST replace this with your actual
# stock data fetching logic (e.g., from Yahoo Finance API, Alpha Vantage, etc.).
# Ensure your actual get_stock_data returns a pandas Series with a DatetimeIndex.

def get_stock_data(ticker: str) -> pd.Series:
    """
    DUMMY FUNCTION: Simulates fetching historical stock data.
    In a real application, this would fetch data from a financial API.
    """
    # print(f"Simulating fetching data for {ticker}...") # Uncomment for verbose dummy data fetching
    end_date = datetime.now() + timedelta(days=30) # Extend a bit into future for hypothetical actuals
    start_date = end_date - timedelta(days=730) # Two years of data

    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    np.random.seed(hash(ticker) % (2**32 - 1)) # Seed based on ticker for some consistency
    base_price = 100 + (ord(ticker[0]) - ord('A')) * 5
    prices = base_price + np.cumsum(np.random.randn(len(dates)) * 0.5) + np.sin(np.arange(len(dates))/30) * 5
    prices = np.maximum(10, prices) # Ensure prices don't go below a floor

    series = pd.Series(prices, index=dates, name=ticker)
    return series

# --- TIMESFM FACTORY (Your original implementation) ---
# This part is kept as is, serving as the model creation module.

# Abstract Factory Base
class TimesFMFactory(ABC):
    @abstractmethod
    def create_model(self) -> timesfm.TimesFm:
        pass

# Concrete Factory for TimesFM v1
class TimesFMV1Factory(TimesFMFactory):
    def create_model(self) -> timesfm.TimesFm:
        context_length = 256
        horizon_length = 64

        hparams = timesfm.TimesFmHparams(
            backend="torch",
            context_len=context_length,
            horizon_len=horizon_length,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
        )
        checkpoint = timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        )
        print(f"[TimesFM-V1] Initializing with context_len={hparams.context_len}, horizon_len={hparams.horizon_len}")
        return timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

# Concrete Factory for TimesFM v2
class TimesFMV2Factory(TimesFMFactory):
    def create_model(self) -> timesfm.TimesFm:
        hparams = timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            horizon_len=128,
            num_layers=50,
            use_positional_embedding=False,
            context_len=2048,
        )
        checkpoint = timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        )
        print(f"[TimesFM-V2] Initializing with context_len={hparams.context_len}, horizon_len={hparams.horizon_len}")
        return timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

# Client Utility Function
def get_timesfm_model(version: str = "v1") -> timesfm.TimesFm:
    factories = {
        "v1": TimesFMV1Factory(),
        "v2": TimesFMV2Factory(),
    }
    factory = factories.get(version)
    if factory is None:
        raise ValueError(f"Unsupported TimesFM version: {version}")
    return factory.create_model()


# --- CORE COMPONENTS ---

class PortfolioDataProvider:
    def __init__(self):
        # Example: Fixed portfolio holdings. In real app, load this dynamically.
        self.portfolio_holdings = {
            'AAPL': 10,
            'MSFT': 5,
            'GOOG': 3
        }

    def get_historical_portfolio_value(self, start_date: str, end_date: str = None) -> pd.Series:
        """
        Fetches historical data for portfolio securities and calculates daily portfolio value.
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        all_security_data = {}
        for ticker in self.portfolio_holdings.keys():
            try:
                security_series = get_stock_data(ticker) # Using the dummy get_stock_data
                security_data_filtered = security_series[
                    (security_series.index >= pd.to_datetime(start_date)) &
                    (security_series.index <= pd.to_datetime(end_date))
                ]
                all_security_data[ticker] = security_data_filtered
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                continue

        if not all_security_data:
            raise ValueError("No historical data could be retrieved for any security in the portfolio.")

        combined_df = pd.DataFrame(all_security_data)
        combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')

        portfolio_value_series = pd.Series(0.0, index=combined_df.index, name="Portfolio Value")
        for ticker, shares in self.portfolio_holdings.items():
            if ticker in combined_df.columns:
                portfolio_value_series += combined_df[ticker] * shares

        portfolio_value_series = portfolio_value_series.sort_index()

        if not isinstance(portfolio_value_series, pd.Series) or not isinstance(portfolio_value_series.index, pd.DatetimeIndex):
            raise TypeError("Processed portfolio value is not a pandas Series with DatetimeIndex.")

        return portfolio_value_series

    def prepare_for_model(self, data: pd.Series, scale: bool = True) -> tuple[np.ndarray, callable, callable]:
        """
        Prepares data for model input (e.g., scaling).
        Returns scaled data array, and an inverse_transform callable.
        """
        values = data.values.astype(np.float32)

        if scale:
            min_val = values.min()
            max_val = values.max()
            if max_val == min_val:
                scaled_values = np.zeros_like(values)
            else:
                scaled_values = (values - min_val) / (max_val - min_val)

            def scaler_inverse_transform(scaled_array):
                return scaled_array * (max_val - min_val) + min_val

            return scaled_values, None, scaler_inverse_transform
        else:
            return values, None, lambda x: x # No-op inverse transform if not scaled


class TimesFMForecaster:
    def __init__(self, model: timesfm.TimesFm): # Accepts an initialized model
        self.model = model
        # Try to safely get context_len and horizon_len, as hparams might not always be directly available
        self.context_len = getattr(self.model, 'hparams', None) and self.model.hparams.context_len or 0
        self.horizon_len = getattr(self.model, 'hparams', None) and self.model.hparams.horizon_len or 0


    def forecast(self, data_array: np.ndarray, forecast_horizon: int) -> np.ndarray:
        """
        Performs a forecast using the initialized TimesFM model.
        """
        if self.context_len > 0 and len(data_array) < self.context_len:
            print(f"Warning: Input data length ({len(data_array)}) is less than model context length ({self.context_len}). "
                  "Model may not perform optimally or might fail if context is strictly required.")

        forecast_input = [data_array]
        frequency_input = np.zeros((len(forecast_input)))

        print(f"Performing forecast with TimesFM (context={self.context_len}, horizon={self.horizon_len}, target_forecast_horizon={forecast_horizon})...")
        point_forecast, _ = self.model.forecast(
            forecast_input,
            freq=frequency_input,
        )
        predicted_full_length = point_forecast[0]

        if len(predicted_full_length) < forecast_horizon:
            print(f"Warning: TimesFM model predicted {len(predicted_full_length)} steps, but {forecast_horizon} were requested. Padding with NaNs.")
            predicted_future = np.pad(predicted_full_length, (0, forecast_horizon - len(predicted_full_length)), 'constant', constant_values=np.nan)
        elif len(predicted_full_length) > forecast_horizon:
            predicted_future = predicted_full_length[:forecast_horizon]
            print(f"TimesFM model predicted {len(predicted_full_length)} steps, trimming to {forecast_horizon}.")
        else:
            predicted_future = predicted_full_length

        print("Forecast complete!")
        return predicted_future


class PerformanceEvaluator:
    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> dict:
        """
        Calculates common time series forecasting metrics.
        """
        if len(actual) != len(predicted):
            raise ValueError("Actual and predicted arrays must have the same length.")

        valid_indices = ~np.isnan(predicted)
        actual_filtered = actual[valid_indices]
        predicted_filtered = predicted[valid_indices]

        if len(actual_filtered) == 0:
            print("Warning: No valid predicted values for evaluation.")
            return {
                'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'directional_accuracy': np.nan
            }

        rmse = np.sqrt(mean_squared_error(actual_filtered, predicted_filtered))
        mae = mean_absolute_error(actual_filtered, predicted_filtered)

        non_zero_actual_indices = actual_filtered != 0
        if np.any(non_zero_actual_indices):
            mape = np.mean(np.abs((actual_filtered[non_zero_actual_indices] - predicted_filtered[non_zero_actual_indices]) / actual_filtered[non_zero_actual_indices])) * 100
        else:
            mape = np.nan

        if len(actual_filtered) > 1 and len(predicted_filtered) > 1:
            actual_diff = np.diff(actual_filtered)
            predicted_diff = np.diff(predicted_filtered)
            # Filter out zero diffs to avoid issues with np.sign, or cases where no change means no direction
            non_zero_diff_indices = (actual_diff != 0)
            if np.any(non_zero_diff_indices):
                correct_directions = np.sum(np.sign(actual_diff[non_zero_diff_indices]) == np.sign(predicted_diff[non_zero_diff_indices]))
                directional_accuracy = (correct_directions / np.sum(non_zero_diff_indices)) * 100
            else:
                directional_accuracy = np.nan # No change in actuals, so DA is undefined
        else:
            directional_accuracy = np.nan

        return {
            'rmse': rmse, 'mae': mae, 'mape': mape, 'directional_accuracy': directional_accuracy
        }


class PortfolioValueVisualizer:
    def plot_forecast_comparison(
        self,
        training_data_series: pd.Series,
        actual_future_data_series: pd.Series,
        predicted_future_values: np.ndarray,
        forecast_start_date: datetime,
        plot_start_date: datetime,
        plot_end_date: datetime,
        title: str = "Portfolio Value Forecast: Actual vs. Predicted"
    ):
        """
        Plots training data, actual future data, and forecasted values within a specified time window.
        """
        plt.figure(figsize=(14, 7))

        predicted_dates = pd.date_range(start=forecast_start_date, periods=len(predicted_future_values), freq='D')
        predicted_series = pd.Series(predicted_future_values, index=predicted_dates)

        plot_training = training_data_series[
            (training_data_series.index >= plot_start_date) &
            (training_data_series.index < forecast_start_date)
        ]

        plot_actual_future = actual_future_data_series[
            (actual_future_data_series.index >= plot_start_date) &
            (actual_future_data_series.index <= plot_end_date)
        ]

        plot_predicted_future = predicted_series[
            (predicted_series.index >= plot_start_date) &
            (predicted_series.index <= plot_end_date)
        ]

        if not plot_training.empty:
            plt.plot(plot_training.index, plot_training.values, label='Training Data', color='blue', linewidth=2)
        else:
            print(f"No training data found in the plotting window: {plot_start_date} to {forecast_start_date - pd.Timedelta(days=1)}")

        if not plot_actual_future.empty:
            plt.plot(plot_actual_future.index, plot_actual_future.values, label='Actual Future Data', color='green', linewidth=2, linestyle='-')
            if not plot_training.empty and not plot_actual_future.empty:
                 plt.axvline(x=forecast_start_date, color='grey', linestyle=':', label='Forecast Start')
        else:
            print(f"No actual future data found in the plotting window: {plot_start_date} to {plot_end_date}")

        if not plot_predicted_future.empty:
            if not plot_training.empty:
                last_training_date = plot_training.index[-1]
                last_training_value = plot_training.iloc[-1]
                first_forecast_date = plot_predicted_future.index[0]
                first_forecast_value = plot_predicted_future.iloc[0]
                plt.plot([last_training_date, first_forecast_date], [last_training_value, first_forecast_value],
                         color='red', linestyle='--', linewidth=2)

            plt.plot(plot_predicted_future.index, plot_predicted_future.values,
                     label='TimesFM Forecast', color='red', linestyle='--', linewidth=2)
        else:
            print(f"No forecasted data found in the plotting window: {plot_start_date} to {plot_end_date}")

        plt.title(f'{title} ({plot_start_date.strftime("%Y-%m-%d")} to {plot_end_date.strftime("%Y-%m-%d")})')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


class HyperparameterTuner:
    def __init__(self, data_provider: PortfolioDataProvider, evaluator: PerformanceEvaluator):
        self.data_provider = data_provider
        self.evaluator = evaluator
        self.results = []
        self.forecaster_instance = None # Will be set by the main runner

    def run_tuning(
        self,
        full_historical_data: pd.Series,
        param_grid: dict,
        forecast_horizon_days: int,
        validation_split_ratio: float = 0.2,
        model_version: str = "v1"
    ):
        """
        Runs hyperparameter tuning using a grid search approach.
        """
        print("\n--- Starting Hyperparameter Tuning ---")

        keys, values = zip(*param_grid.items())
        hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        num_validation_points = int(len(full_historical_data) * validation_split_ratio)
        if num_validation_points < forecast_horizon_days:
            num_validation_points = forecast_horizon_days
            print(f"Adjusted validation points to {num_validation_points} to accommodate forecast horizon.")

        if len(full_historical_data) <= num_validation_points:
            raise ValueError(f"Historical data ({len(full_historical_data)} days) is too short for validation split "
                             f"of {validation_split_ratio*100}% ({num_validation_points} days). "
                             "Please provide more historical data or reduce validation_split_ratio.")

        training_data_for_tuning = full_historical_data.iloc[:-num_validation_points]
        actual_validation_data = full_historical_data.iloc[-num_validation_points:]

        print(f"Tuning Training Data Range: {training_data_for_tuning.index.min().strftime('%Y-%m-%d')} to {training_data_for_tuning.index.max().strftime('%Y-%m-%d')}")
        print(f"Validation Data Range: {actual_validation_data.index.min().strftime('%Y-%m-%d')} to {actual_validation_data.index.max().strftime('%Y-%m-%d')}")

        scaled_training_values, _, inverse_scaler_func = self.data_provider.prepare_for_model(
            training_data_for_tuning, scale=True
        )

        best_score = float('inf')
        best_params = None
        best_prediction = None # Store the best unscaled prediction

        for i, hparams_config in enumerate(hyperparameter_combinations):
            print(f"\n--- Trial {i+1}/{len(hyperparameter_combinations)} ---")
            print(f"Trying HParams: {hparams_config}")

            current_hparams_dict = collections.defaultdict(lambda: None)
            if model_version == "v1":
                current_hparams_dict.update({
                    "backend": "torch", "context_len": 256, "horizon_len": 64,
                    "input_patch_len": 32, "output_patch_len": 128,
                    "num_layers": 20, "model_dims": 1280,
                })
            elif model_version == "v2":
                current_hparams_dict.update({
                    "backend": "gpu", "per_core_batch_size": 32, "horizon_len": 128,
                    "num_layers": 50, "use_positional_embedding": False, "context_len": 2048,
                })

            for k, v in hparams_config.items():
                if k in current_hparams_dict:
                    current_hparams_dict[k] = v
                else:
                    print(f"Warning: Hyperparameter '{k}' not a standard TimesFmHparams attribute for {model_version}.")

            try:
                trial_hparams = timesfm.TimesFmHparams(**current_hparams_dict)
                checkpoint_id = "google/timesfm-1.0-200m-pytorch" if model_version == "v1" else "google/timesfm-2.0-500m-pytorch"
                trial_checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=checkpoint_id)
                trial_model = timesfm.TimesFm(hparams=trial_hparams, checkpoint=trial_checkpoint)

                # Temporarily update the forecaster instance's model
                original_forecaster_model = self.forecaster_instance.model
                original_context_len = self.forecaster_instance.context_len
                original_horizon_len = self.forecaster_instance.horizon_len

                self.forecaster_instance.model = trial_model
                self.forecaster_instance.context_len = trial_hparams.context_len
                self.forecaster_instance.horizon_len = trial_hparams.horizon_len

                scaled_prediction = self.forecaster_instance.forecast(scaled_training_values, forecast_horizon_days)
                predicted_values = inverse_scaler_func(scaled_prediction)

                metrics = self.evaluator.calculate_metrics(
                    actual_validation_data.values[:len(predicted_values)],
                    predicted_values
                )

                print(f"Trial Metrics (RMSE): {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, DA: {metrics['directional_accuracy']:.2f}%")

                trial_result = {
                    'hparams': hparams_config,
                    'metrics': metrics,
                    'predicted_values': predicted_values.tolist(),
                    'model_version': model_version
                }
                self.results.append(trial_result)

                if not np.isnan(metrics['rmse']) and metrics['rmse'] < best_score:
                    best_score = metrics['rmse']
                    best_params = hparams_config
                    best_prediction = predicted_values

            except Exception as e:
                print(f"Error during trial with hparams {hparams_config}: {e}")
                trial_result = {
                    'hparams': hparams_config,
                    'metrics': {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'directional_accuracy': np.nan},
                    'predicted_values': [],
                    'model_version': model_version,
                    'error': str(e)
                }
                self.results.append(trial_result)
            finally:
                # Restore original forecaster model
                self.forecaster_instance.model = original_forecaster_model
                self.forecaster_instance.context_len = original_context_len
                self.forecaster_instance.horizon_len = original_horizon_len


        print("\n--- Hyperparameter Tuning Complete ---")
        print(f"Best RMSE: {best_score:.4f}")
        print(f"Best HParams: {best_params}")
        return best_params, best_score, best_prediction, self.results


# --- MAIN APPLICATION LOGIC ---

def run_portfolio_forecasting_app():
    # --- Configuration ---
    MODEL_VERSION = "v1" # Or "v2"
    FORECAST_HORIZON_DAYS = 30 # How many days into the future to forecast
    DATA_START_DATE = '2023-01-01' # Start fetching historical data from this date
    PLOT_LOOKBACK_DAYS = 90 # How many days before forecast start to show for context in plot in final plot

    # Hyperparameter Grid for Tuning
    # Define the ranges for hyperparameters you want to tune.
    # Be mindful of the capabilities of each TimesFM version (e.g., max context_len).
    param_grid = {
        'context_len': [256, 512],
        'horizon_len': [64, 128],
        # You can add more hparams to tune, e.g., 'input_patch_len': [16, 32]
    }
    # Adjust for v2 if selected
    if MODEL_VERSION == "v2":
        param_grid['context_len'] = [1024, 2048]
        param_grid['horizon_len'] = [128, 256] # Adjust based on your forecast horizon needs

    # --- 1. Initialize Core Components ---
    data_provider = PortfolioDataProvider()
    evaluator = PerformanceEvaluator()
    visualizer = PortfolioValueVisualizer()

    # Initial model setup for the forecaster (this model will be replaced during tuning)
    initial_timesfm_model = get_timesfm_model(MODEL_VERSION)
    forecaster_instance = TimesFMForecaster(model=initial_timesfm_model)

    # Initialize tuner and inject the forecaster instance
    tuner = HyperparameterTuner(data_provider, evaluator)
    tuner.forecaster_instance = forecaster_instance # This instance will be modified by tuner

    # --- 2. Get Historical Portfolio Data ---
    print(f"Fetching historical portfolio data from {DATA_START_DATE} to {datetime.now().strftime('%Y-%m-%d')}...")
    try:
        full_historical_data = data_provider.get_historical_portfolio_value(
            start_date=DATA_START_DATE,
            end_date=datetime.now().strftime('%Y-%m-%d') # Ensure we get data up to current actual date
        )
        print(f"Historical data fetched. Shape: {full_historical_data.shape}")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # --- 3. Run Hyperparameter Tuning ---
    print("\nStarting hyperparameter tuning...")
    best_params, best_score, best_prediction_on_validation, all_tuning_results = tuner.run_tuning(
        full_historical_data=full_historical_data,
        param_grid=param_grid,
        forecast_horizon_days=FORECAST_HORIZON_DAYS,
        validation_split_ratio=0.2, # Use 20% of data for validation during tuning
        model_version=MODEL_VERSION
    )

    print("\n--- Tuning Summary ---")
    print(f"Best Parameters: {best_params}")
    print(f"Best Validation RMSE: {best_score:.4f}")

    # --- 4. Prepare Final Forecast Model and Data (using *all* historical data) ---
    print("\nPreparing final forecast with best hyperparameters using all available historical data...")

    # Recreate Hparams with best_params and then the model
    final_hparams_dict = collections.defaultdict(lambda: None)
    if MODEL_VERSION == "v1":
        default_hparams_v1 = TimesFMV1Factory().create_model().hparams # Load defaults
        final_hparams_dict.update(default_hparams_v1.__dict__) # Convert to dict
    elif MODEL_VERSION == "v2":
        default_hparams_v2 = TimesFMV2Factory().create_model().hparams
        final_hparams_dict.update(default_hparams_v2.__dict__)
    else:
        raise ValueError(f"Unsupported model version: {MODEL_VERSION}")

    # Apply the best_params found during tuning
    for k, v in best_params.items():
        if k in final_hparams_dict:
            final_hparams_dict[k] = v
        else:
            print(f"Warning: Best parameter '{k}' not a standard TimesFmHparams attribute for final model. Ignoring.")

    # Create the final TimesFmHparams object
    final_model_hparams = timesfm.TimesFmHparams(**final_hparams_dict)

    # Re-initialize the final model with the chosen best parameters
    checkpoint_id = "google/timesfm-1.0-200m-pytorch" if MODEL_VERSION == "v1" else "google/timesfm-2.0-500m-pytorch"
    final_model_checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=checkpoint_id)
    final_tuned_model = timesfm.TimesFm(hparams=final_model_hparams, checkpoint=final_model_checkpoint)
    final_forecaster = TimesFMForecaster(model=final_tuned_model)

    # Prepare the *entire* historical dataset for the final forecast (training)
    scaled_full_data_for_final_forecast, _, inverse_scaler_func_final = data_provider.prepare_for_model(
        full_historical_data, scale=True
    )
    final_predicted_scaled = final_forecaster.forecast(scaled_full_data_for_final_forecast, FORECAST_HORIZON_DAYS)
    final_predicted_values = inverse_scaler_func_final(final_predicted_scaled)

    # --- 5. Define Plotting Window for Final Forecast ---
    last_historical_date_for_plot = full_historical_data.index[-1]
    forecast_start_date_for_plot = last_historical_date_for_plot + timedelta(days=1)
    # The end date for plotting will extend to the end of the final forecast horizon
    end_plot_date = forecast_start_date_for_plot + timedelta(days=FORECAST_HORIZON_DAYS - 1)
    # The start date for plotting will look back some days for context
    start_plot_date = last_historical_date_for_plot - timedelta(days=PLOT_LOOKBACK_DAYS)

    # Ensure plot start date is not before the earliest available data
    if start_plot_date < full_historical_data.index.min():
        start_plot_date = full_historical_data.index.min()
        print(f"Adjusted plot start date to {start_plot_date.strftime('%Y-%m-%d')} due to data availability.")

    # Create a dummy series for 'actual future data' in the final plot.
    # This will be empty, as we're forecasting into the truly unknown future.
    actual_future_for_final_plot = pd.Series(
        index=pd.date_range(start=forecast_start_date_for_plot, periods=FORECAST_HORIZON_DAYS, freq='D'),
        dtype=float
    )

    # --- 6. Visualize Final Forecast ---
    visualizer.plot_forecast_comparison(
        training_data_series=full_historical_data, # All historical data is now 'training'
        actual_future_data_series=actual_future_for_final_plot, # This is the placeholder for unknown future
        predicted_future_values=final_predicted_values,
        forecast_start_date=forecast_start_date_for_plot,
        plot_start_date=start_plot_date,
        plot_end_date=end_plot_date,
        title=f"Final Portfolio Value Forecast (TimesFM {MODEL_VERSION} - Tuned HParams)"
    )

    print("\nApplication run complete.")


if __name__ == '__main__':
    run_portfolio_forecasting_app()