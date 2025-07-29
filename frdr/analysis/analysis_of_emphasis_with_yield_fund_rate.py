import pandas as pd
from pandas_datareader import data as pdr
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from frdr.database.connections import engine,session

# Set date range (Powell: 2018–2025)
start_date = '2018-01-01'
end_date = '2025-05-12'

# 1. Fetch External Data
# Override pandas-datareader to use FRED
import pandas_datareader.data as web
web.DataReader = pdr.DataReader

# 10-Year Treasury Yield (FRED: DGS10)
treasury_yield = web.DataReader('DGS10', 'fred', start_date, end_date)
treasury_yield = treasury_yield.rename(columns={'DGS10': 'Yield'})

# Federal Funds Rate (FRED: DFF)
ffr = web.DataReader('DFF', 'fred', start_date, end_date)
ffr = ffr.rename(columns={'DFF': 'FedFundsRate'})

speech_analysis_data = pd.read_sql("""
select chair,speaker, date, emphasis ,title 
from fed_speech_analysis fsa join
fed_speeches fs2 on fs2.id = fsa.speech_id
""",engine)

speech_data = speech_analysis_data[speech_analysis_data['chair'] == 'Jerome Powell']
speech_data['inflation'] = speech_analysis_data['emphasis'].apply(lambda x: x['inflation'])
speech_data['employment'] = speech_analysis_data['emphasis'].apply(lambda x: x['employment'])
speech_data['other'] = speech_analysis_data['emphasis'].apply(lambda x: x['other'])
speech_data['date']  =speech_analysis_data['date']
# 3. Merge Data
# Set speech dates as index
speech_data = speech_data.set_index('date')

# Merge with yield and federal funds rate
data = treasury_yield.join(ffr, how='outer')
data = data.join(speech_data[['inflation', 'employment', 'other']], how='left')

# Forward-fill market data for non-trading days
data[['Yield', 'FedFundsRate']] = data[['Yield', 'FedFundsRate']].ffill()

# 4. Event-Study Setup
# Filter speech days
events = data.dropna(subset=['inflation', 'employment', 'other'])


# Calculate abnormal yield changes
def calculate_ay(row, estimation_window=120, event_window=2):
    event_date = row.name
    est_start = event_date - pd.Timedelta(days=estimation_window + 10)
    est_end = event_date - pd.Timedelta(days=10)

    # Estimation window data
    est_data = data.loc[est_start:est_end, ['Yield']].dropna()
    if len(est_data) < 30:  # Ensure sufficient data
        return pd.Series({'AY': None, 'CAY': None})

    # Mean yield change
    est_data['Yield_Change'] = est_data['Yield'].diff()
    mu = est_data['Yield_Change'].mean()

    # Event window yield changes
    event_end = event_date + pd.Timedelta(days=event_window - 1)
    event_data = data.loc[event_date:event_end, ['Yield']].dropna()
    if len(event_data) < event_window:
        return pd.Series({'AY': None, 'CAY': None})

    event_data['Yield_Change'] = event_data['Yield'].diff()
    event_data['AY'] = event_data['Yield_Change'] - mu
    cay = event_data['AY'].sum()

    return pd.Series({'AY': event_data['AY'].iloc[0], 'CAY': cay})


events = events.join(events.apply(calculate_ay, axis=1))


# 5. Regression Analysis
# Regress CAY on emphasis scores and federal funds rate
X = events[['inflation', 'employment', 'other', 'FedFundsRate']].dropna()
y_cay = events['CAY'].dropna()
events.to_csv('processed_events.csv')

X_cay = sm.add_constant(X)
model_cay = sm.OLS(y_cay, X_cay).fit()

# Save regression summary to text file
with open('regression_summary.txt', 'w') as f:
    f.write(model_cay.summary().as_text())

# 6. Visualization
# Plot CAY distributions by high vs. low inflation emphasis
high_inflation = events[events['inflation'] > events['inflation'].quantile(0.75)]
low_inflation = events[events['inflation'] < events['inflation'].quantile(0.25)]

plt.figure(figsize=(8, 5))
sns.kdeplot(high_inflation['CAY'], label='High Inflation Emphasis', color='red')
sns.kdeplot(low_inflation['CAY'], label='Low Inflation Emphasis', color='blue')
plt.title('Cumulative Abnormal Yield Changes by Inflation Emphasis (Powell, 2018–2025)')
plt.xlabel('CAY (%)')
plt.ylabel('Density')
plt.legend()
plt.savefig('cay_distribution.png')
plt.close()

# 7. Save Results Table
results_table = pd.DataFrame({
    'Variable': X_cay.columns,
    'Coefficient': model_cay.params,
    'P-Value': model_cay.pvalues
})
results_table.to_csv('regression_results.csv', index=False)