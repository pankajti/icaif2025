import pandas as pd
import matplotlib.pyplot as plt

# Load narrative results
results = pd.read_csv("fed_speeches_emphasis_score.csv", parse_dates=["date"])

# Load FRED macro data
unrate = pd.read_csv("UNRATE.csv", parse_dates=["observation_date"]).rename(columns={"observation_date": "date", "UNRATE": "unrate"})
cpi = pd.read_csv("CPIAUCSL.csv", parse_dates=["observation_date"]).rename(columns={"observation_date": "date", "CPIAUCSL": "cpi"})

# If results_data.csv has narrative scores as columns (e.g., 'employment_score', 'inflation_score'), use them directly.
# If stored as lists/dicts per row, further processing may be needed. Here, we assume 'employment_emphasis' and 'inflation_emphasis' columns.

# Merge all datasets by date (monthly or nearest available)
merged = pd.merge_asof(
    results.sort_values("date"),
    unrate.sort_values("date"),
    on="date",
    direction="nearest"
)
merged = pd.merge_asof(
    merged.sort_values("date"),
    cpi.sort_values("date"),
    on="date",
    direction="nearest"
)

# Plot
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(merged["date"], merged["employment_score"], label="Narrative Employment Focus", color="tab:blue")
ax1.set_ylabel("Narrative Employment Emphasis", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(merged["date"], merged["unrate"], label="Unemployment Rate (FRED)", color="tab:orange", alpha=0.7)
ax2.set_ylabel("Unemployment Rate (%)", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

ax2.plot(merged["date"], merged["cpi"], label="CPI (FRED, not inflation %)", color="tab:green", alpha=0.7, linestyle="--")

fig.suptitle("Fed Employment Narrative Emphasis vs. Unemployment Rate & CPI")
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.87))
plt.tight_layout()
plt.show()

