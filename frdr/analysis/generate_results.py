"""
Script to replicate the analysis in the Fed speech emphasis paper.

This script reads the speech emphasis scores and macroeconomic series,
aligns them at monthly frequency, computes cross-correlation statistics
between speech emphasis and macro variables, and generates figures and
a CSV summarizing the correlations.  It requires pandas, numpy and
matplotlib.

To run:
    python generate_results.py

The script assumes the following files are present in the current
working directory:
    - fed_speeches_emphasis_score.csv
    - CPIAUCSL.csv
    - UNRATE.csv

Outputs:
    - inflation_cpi_crosscorr.png
    - employment_unrate_crosscorr.png
    - rolling_correlations.png
    - crosscorr_results.csv
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the emphasis scores
speeches = pd.read_csv('fed_speeches_emphasis_score.csv', parse_dates=['date'])

# Convert speech dates to monthly periods and compute average emphasis per month
speeches['month'] = speeches['date'].dt.to_period('M').dt.to_timestamp()
monthly_emphasis = speeches.groupby('month')[['inflation_score', 'employment_score']].mean().reset_index()

# Load CPI and unemployment series
cpi = pd.read_csv('CPIAUCSL.csv', parse_dates=['observation_date']).rename(columns={'observation_date': 'month'})
unrate = pd.read_csv('UNRATE.csv', parse_dates=['observation_date']).rename(columns={'observation_date': 'month'})

# Merge the series on the month
merged = monthly_emphasis.merge(cpi, on='month', how='inner').merge(unrate, on='month', how='inner')

# Compute month-over-month changes in macro variables
merged['cpi_change'] = merged['CPIAUCSL'].pct_change()
merged['unrate_change'] = merged['UNRATE'].diff()

# Drop missing values created by differencing
merged = merged.dropna(subset=['cpi_change', 'unrate_change'])

# Helper to compute cross-correlation for specified lags

def cross_corr(x, y, max_lag=6):
    """
    Compute cross-correlation coefficients for lags from -max_lag to +max_lag.

    Args:
        x (np.ndarray): First time series (e.g. emphasis).
        y (np.ndarray): Second time series (e.g. macro change).
        max_lag (int): Maximum number of months to lag/lead.

    Returns:
        lags (list): List of integer lags.
        corrs (list): Corresponding correlation coefficients.
    """
    lags = range(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        if lag < 0:
            # Negative lag: x is lagged behind y
            corr = np.corrcoef(x[:lag], y[-lag:])[0, 1]
        elif lag > 0:
            # Positive lag: x leads y
            corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
        else:
            # Zero lag
            corr = np.corrcoef(x, y)[0, 1]
        corrs.append(corr)
    return list(lags), corrs

# Compute cross-correlations
lags, corr_infl_cpi = cross_corr(merged['inflation_score'].values, merged['cpi_change'].values)
lags, corr_emp_unrate = cross_corr(merged['employment_score'].values, -merged['unrate_change'].values)

# Save correlation results to CSV
corr_df = pd.DataFrame({'LagMonths': lags,
                        'Infl_vs_CPI': corr_infl_cpi,
                        'Emp_vs_Unrate': corr_emp_unrate})
corr_df.to_csv('crosscorr_results.csv', index=False)

# Plot cross-correlation for inflation emphasis and CPI change
plt.figure(figsize=(8, 4))
plt.stem(lags, corr_infl_cpi, basefmt=' ')
plt.axhline(0, color='black', linewidth=0.8)
plt.xlabel('Lag (months)')
plt.ylabel('Correlation')
plt.title('Cross-correlation: Inflation emphasis vs. CPI change')
plt.tight_layout()
plt.savefig('inflation_cpi_crosscorr.png')

# Plot cross-correlation for employment emphasis and labour improvement
plt.figure(figsize=(8, 4))
plt.stem(lags, corr_emp_unrate, basefmt=' ')
plt.axhline(0, color='black', linewidth=0.8)
plt.xlabel('Lag (months)')
plt.ylabel('Correlation')
plt.title('Cross-correlation: Employment emphasis vs. labour improvement')
plt.tight_layout()
plt.savefig('employment_unrate_crosscorr.png')

# Compute rolling correlations with a 12-month window
roll_corr_infl = merged['inflation_score'].rolling(window=12).corr(merged['cpi_change'])
roll_corr_emp = merged['employment_score'].rolling(window=12).corr(-merged['unrate_change'])

# Plot rolling correlations
plt.figure(figsize=(8, 4))
plt.plot(merged['month'], roll_corr_infl, label='Inflation emphasis vs. CPI change')
plt.plot(merged['month'], roll_corr_emp, label='Employment emphasis vs. labour improvement')
plt.axhline(0, color='black', linewidth=0.8)
plt.xlabel('Date')
plt.ylabel('Rolling 12-month correlation')
plt.title('Rolling correlations between speech emphasis and macro indicators')
plt.legend()
plt.tight_layout()
plt.savefig('rolling_correlations.png')

print('Analysis complete. Outputs saved to current directory.')
