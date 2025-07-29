import os

# Define base path for new modules
base_path = "/Users/pankajti/dev/git/icaif2025/frdr/fed_narratives/agents"

# Create directories for new agents
os.makedirs(os.path.join(base_path, "theme_averager"), exist_ok=True)
os.makedirs(os.path.join(base_path, "surprise_detector"), exist_ok=True)

# Define initial content for ThemeAveragerAgent
theme_averager_code = '''\
from datetime import datetime, timedelta
import pandas as pd

class ThemeAveragerAgent:
    def __init__(self, history_df: pd.DataFrame, window_days: int = 365):
        self.history_df = history_df
        self.window_days = window_days

    def compute_baseline(self, theme: str, date: str) -> float:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        window_start = date_obj - timedelta(days=self.window_days)
        theme_scores = self.history_df[
            (self.history_df["theme"] == theme) &
            (self.history_df["date"] >= window_start.strftime("%Y-%m-%d")) &
            (self.history_df["date"] < date)
        ]["emphasis_score"]
        return theme_scores.mean() if not theme_scores.empty else 0.0
'''

# Define initial content for SurpriseDetectorAgent
surprise_detector_code = '''\
class SurpriseDetectorAgent:
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def detect(self, current_score: float, baseline_score: float) -> dict:
        surprise = current_score - baseline_score
        direction = "upward" if surprise > 0 else "downward"
        magnitude = abs(surprise)
        is_significant = magnitude > self.threshold
        return {
            "surprise_score": magnitude,
            "direction": direction,
            "significant": is_significant
        }
'''

# Write the content to files
with open(os.path.join(base_path, "theme_averager", "theme_averager_agent.py"), "w") as f:
    f.write(theme_averager_code)

with open(os.path.join(base_path, "surprise_detector", "surprise_detector_agent.py"), "w") as f:
    f.write(surprise_detector_code)

# Output success message
"ThemeAveragerAgent and SurpriseDetectorAgent scaffolding created under frdr/agents."
