import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import pandas_datareader.data as web
import json
from event_study_toolkit.eventstudy import eventstudy

# --- Setup database connection (adjust credentials if needed) ---
# Assuming 'engine' is configured in your connections file
from frdr.database.connections import engine,session

# -- Step 1: Load Financial Market Data from FRED --
start_date = '2018-01-01'
end_date = '2025-05-12'

# Treasury Yield (our "security")
treasury_yield = web.DataReader('DGS10', 'fred', start_date, end_date)

# Federal Funds Rate (a potential market factor)
ffr = web.DataReader('DFF', 'fred', start_date, end_date)

# --- Step 2: Load Fed Speech Data from PostgreSQL ---
event_query = """
    SELECT 
        1 AS permno, 
        fs.date, 
        fs.speaker, 
        fs.title, 
        fsa.emphasis 
    FROM fed_speech_analysis fsa 
    JOIN fed_speeches fs ON fs.id = fsa.speech_id
"""
speech_data = pd.read_sql(event_query, con=engine)
speech_data['date'] = pd.to_datetime(speech_data['date'])

# Filter for Jerome Powell
#speech_data = speech_data[speech_data['speaker'].str.contains('Jerome Powell')].copy()

# Correctly parse JSON and extract numerical values for grouping
# Assuming the data is a dictionary already, as per our last conversation
speech_data['inflation'] = speech_data['emphasis'].apply(lambda x: x.get('inflation', 0))
speech_data['employment'] = speech_data['emphasis'].apply(lambda x: x.get('employment', 0))

# --- Step 3: Construct Clean DataFrames for the Toolkit ---

# Create the 'data' DataFrame (continuous time series of market data)
market_data = treasury_yield.join(ffr, how='outer')
market_data = market_data.ffill()
market_data = market_data.reset_index().rename(columns={'index': 'date'})

# Calculate percentage returns for the yield (standard practice)
market_data['ret_dlst_adj'] = market_data['DGS10'].pct_change()
market_data['FedFundsRate'] = market_data['DFF'].pct_change()

# Add a unique ID for the single security (the Treasury Yield)
market_data['permno'] = 1
market_data = market_data.dropna().copy()

market_data =market_data.rename(columns={'DATE': 'date'})

# Create the 'events' DataFrame (discrete events with their metadata)
# Use a unique ID and the event date
events_data = speech_data.rename(columns={'date': 'EVT_DATE'})

# The 'inflation' and 'employment' columns will be used for grouping
# We can use a simple 'category' column to group all events together
events_data['event_group'] = 'FedSpeech'
events_data = events_data[['permno', 'EVT_DATE', 'inflation', 'employment', 'event_group']].copy()


# --- Step 4: Run Event Study Toolkit ---
ESTPERIOD = 120
GAP = 10
START = -1 # A 1-day event window before, on, and after the speech
END = 1

# Instantiate the event study
es_tool = eventstudy(
    estperiod=ESTPERIOD,
    gap=GAP,
    start=START,
    end=END,
    data=market_data, # Pass the clean, continuous market data
    events=events_data, # Pass the clean event data
    unique_id='permno',
    calType='NYSE',
    groups=['event_group'] # We group by our simple category first
)

# --- Step 5: Execute Models and Analyze Results ---

# A simple model:
# The abnormal return is the difference between the actual yield change and the expected change.
# We'll assume the expected change is just the mean over the estimation period.
# A more complex model would be 'ret_dlst_adj ~ FedFundsRate'
model_formula = 'ret_dlst_adj ~ 1'
model_formula='market'
model_formula = 'ret_dlst_adj ~ FedFundsRate'
model_formula='ret_dlst_adj ~ FedFundsRate + inflation + employment'


car_stats = es_tool.getFullSampleTestStatistic(model_formula)
print("Full Sample Test Statistics:")
print(car_stats)

# Now, use the groups to see if inflation-focused speeches have a different effect
# We can discretize your numerical scores into a simple categorical variable
events_data['inflation_level'] = pd.cut(events_data['inflation'], bins=[-0.1, 0.2, 0.6, 1.1], labels=['Low', 'Medium', 'High'])

# Rerun the event study with the new grouping variable
es_tool_groups = eventstudy(
    estperiod=ESTPERIOD,
    gap=GAP,
    start=START,
    end=END,
    data=market_data,
    events=events_data,
    unique_id='permno',
    calType='NYSE',
    groups=['inflation_level']
)

# Get group-level statistics
group_stats = es_tool_groups.getGroupLevelTestStatistics(model_formula, GRP='inflation_level')
print("\nGroup-Level Test Statistics by Inflation Emphasis:")
print(group_stats)

# -- Optional: Plot CARs for a better visual representation --
# Get the raw CARs from the model
car_df = es_tool_groups.getCARS(model_formula)

# Calculate the average CAR by event day and group
car_by_day = car_df.groupby([car_df['EVT_DATE'], 'inflation_level'])['car'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(
    x='EVT_DATE',
    y='car',
    hue='inflation_level',
    data=car_by_day,
    marker='o',
    linestyle='-',
    palette='viridis'
)
plt.title("Average CAR Around Speeches, by Inflation Emphasis")
plt.xlabel("Date")
plt.ylabel("Average CAR")
plt.axvline(x=0, color='r', linestyle='--', label='Event Day')
plt.legend(title="Inflation Emphasis")
plt.grid(True)
plt.show()