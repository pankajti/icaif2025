import os

# Recreate the directory and file structure after kernel reset
package_name = "timesfm_forecasting"
os.makedirs(package_name, exist_ok=True)

file_contents = {
    "__init__.py": "# Init for timesfm_forecasting package",
    "data_provider.py": "# Portfolio data loader\n\n"
                        "import pandas as pd\n"
                        "import numpy as np\n"
                        "from datetime import datetime, timedelta\n\n"
                        "def get_stock_data(ticker: str) -> pd.Series:\n"
                        "    # Dummy data for demo; replace with real data fetching logic\n"
                        "    dates = pd.date_range(datetime.now() - timedelta(days=730), datetime.now(), freq='D')\n"
                        "    prices = 100 + np.cumsum(np.random.randn(len(dates)))\n"
                        "    return pd.Series(prices, index=dates, name=ticker)\n\n"
                        "class PortfolioDataProvider:\n"
                        "    def __init__(self):\n"
                        "        self.portfolio = {'AAPL': 10, 'MSFT': 5, 'GOOG': 3}\n\n"
                        "    def get_historical_portfolio_value(self) -> pd.Series:\n"
                        "        data = {t: get_stock_data(t) for t in self.portfolio}\n"
                        "        df = pd.DataFrame(data).fillna(method='ffill')\n"
                        "        return sum(df[t] * qty for t, qty in self.portfolio.items())",
    "model_factory.py": "# TimesFM model factory\n\n"
                        "from abc import ABC, abstractmethod\n"
                        "import timesfm\n\n"
                        "class TimesFMFactory(ABC):\n"
                        "    @abstractmethod\n"
                        "    def create_model(self) -> timesfm.TimesFm:\n"
                        "        pass\n\n"
                        "class TimesFMV1Factory(TimesFMFactory):\n"
                        "    def create_model(self):\n"
                        "        hparams = timesfm.TimesFmHparams(\n"
                        "            backend='torch', context_len=256, horizon_len=64,\n"
                        "            input_patch_len=32, output_patch_len=128,\n"
                        "            num_layers=20, model_dims=1280,\n"
                        "        )\n"
                        "        checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id='google/timesfm-1.0-200m-pytorch')\n"
                        "        return timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)\n\n"
                        "class TimesFMV2Factory(TimesFMFactory):\n"
                        "    def create_model(self):\n"
                        "        hparams = timesfm.TimesFmHparams(\n"
                        "            backend='gpu', context_len=2048, horizon_len=128,\n"
                        "            num_layers=50, use_positional_embedding=False,\n"
                        "        )\n"
                        "        checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id='google/timesfm-2.0-500m-pytorch')\n"
                        "        return timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)",
    "forecaster.py": "# TimesFM forecasting core\n\n"
                     "import numpy as np\n\n"
                     "class TimesFMForecaster:\n"
                     "    def __init__(self, model):\n"
                     "        self.model = model\n"
                     "        self.context_len = model.hparams.context_len\n"
                     "        self.horizon_len = model.hparams.horizon_len\n\n"
                     "    def forecast(self, input_array: np.ndarray, forecast_horizon: int):\n"
                     "        point_forecast, _ = self.model.forecast([input_array], freq=np.zeros(1))\n"
                     "        return point_forecast[0][:forecast_horizon]",
    "evaluator.py": "# Evaluation metrics\n\n"
                    "from sklearn.metrics import mean_squared_error, mean_absolute_error\n"
                    "import numpy as np\n\n"
                    "class PerformanceEvaluator:\n"
                    "    def calculate_metrics(self, actual, predicted):\n"
                    "        rmse = np.sqrt(mean_squared_error(actual, predicted))\n"
                    "        mae = mean_absolute_error(actual, predicted)\n"
                    "        da = np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))) * 100\n"
                    "        return {'rmse': rmse, 'mae': mae, 'directional_accuracy': da}",
    "plotter.py": "# Visualization\n\n"
                  "import matplotlib.pyplot as plt\n"
                  "import pandas as pd\n\n"
                  "class PortfolioValueVisualizer:\n"
                  "    def plot_forecast_comparison(self, history, forecast, start_date, title='Forecast'):\n"
                  "        plt.figure(figsize=(12,6))\n"
                  "        plt.plot(history.index, history.values, label='Historical')\n"
                  "        forecast_dates = pd.date_range(start=start_date, periods=len(forecast))\n"
                  "        plt.plot(forecast_dates, forecast, label='Forecast', linestyle='--')\n"
                  "        plt.legend(); plt.grid(); plt.title(title); plt.show()",
    "tuner.py": "# Hyperparameter tuning stub\n\n"
                "class HyperparameterTuner:\n"
                "    def __init__(self):\n"
                "        self.results = []\n\n"
                "    def tune(self):\n"
                "        print('Tuning not implemented in this stub.')",
    "app.py": "# Main entry point\n\n"
              "from timesfm_forecasting.data_provider import PortfolioDataProvider\n"
              "from timesfm_forecasting.model_factory import TimesFMV1Factory\n"
              "from timesfm_forecasting.forecaster import TimesFMForecaster\n"
              "from timesfm_forecasting.plotter import PortfolioValueVisualizer\n\n"
              "provider = PortfolioDataProvider()\n"
              "series = provider.get_historical_portfolio_value()\n\n"
              "model = TimesFMV1Factory().create_model()\n"
              "forecaster = TimesFMForecaster(model)\n"
              "values = series.values.astype('float32')\n"
              "forecast = forecaster.forecast(values, 30)\n\n"
              "visualizer = PortfolioValueVisualizer()\n"
              "visualizer.plot_forecast_comparison(series, forecast, series.index[-1], title='Demo Forecast')"
}

# Write files
for filename, content in file_contents.items():
    with open(os.path.join(package_name, filename), "w") as f:
        f.write(content)

"✅ All module files generated inside 'timesfm_forecasting' package."
