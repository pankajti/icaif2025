import pandas as pd
import re

# Read the text file
with open('/Users/pankajti/dev/git/icaif2025/wsda/data/downloaded_reports/2025/wasde0125_TXT.txt', 'r') as file:
    text = file.read()


# Function to clean and split lines
def clean_lines(lines):
    return [line.strip() for line in lines if
            line.strip() and not line.startswith('====') and not line.startswith('WASDE')]


# Function to parse a table into a DataFrame
def parse_table(lines, columns, index_col=None):
    data = []
    for line in lines:
        # Split by whitespace, handling multiple spaces
        values = re.split(r'\s+', line.strip())
        if len(values) >= len(columns):
            data.append(values[:len(columns)])
    return pd.DataFrame(data, columns=columns)


# Dictionary to store DataFrames for each commodity
dataframes = {}

# 1. World and U.S. Supply and Use for Grains
grain_section = re.search(r'World and U.S. Supply and Use for Grains.*?Foreign\s+3/', text, re.DOTALL)
if grain_section:
    grain_lines = grain_section.group(0).split('\n')
    grain_lines = clean_lines(grain_lines)

    # Define columns for grains
    grain_columns = ['Commodity', 'Year', 'Output', 'Supply', 'Trade', 'Use', 'Ending_Stocks']

    # Extract World, U.S., and Foreign data
    world_grains, us_grains, foreign_grains = [], [], []
    current_section = None

    for line in grain_lines:
        if 'World' in line:
            current_section = 'World'
            continue
        elif 'United States' in line:
            current_section = 'United States'
            continue
        elif 'Foreign' in line:
            current_section = 'Foreign'
            continue
        if current_section and re.match(r'^\s*(Total Grains|Wheat|Coarse Grains|Rice, milled)', line):
            values = re.split(r'\s+', line.strip())
            if len(values) >= 6:
                commodity = values[0]
                if len(values) > 6:
                    commodity = ' '.join(values[:2])
                    values = [commodity] + values[2:]
                if current_section == 'World':
                    world_grains.append(values)
                elif current_section == 'United States':
                    us_grains.append(values)
                elif current_section == 'Foreign':
                    foreign_grains.append(values)

    # Create DataFrames
    dataframes['World_Grains'] = pd.DataFrame(world_grains, columns=grain_columns)
    dataframes['US_Grains'] = pd.DataFrame(us_grains, columns=grain_columns)
    dataframes['Foreign_Grains'] = pd.DataFrame(foreign_grains, columns=grain_columns)

# 2. World and U.S. Supply and Use for Cotton
cotton_section = re.search(r'World and U.S. Supply and Use for Cotton.*?Oilseeds', text, re.DOTALL)
if cotton_section:
    cotton_lines = cotton_section.group(0).split('\n')
    cotton_lines = clean_lines(cotton_lines)

    # Define columns for cotton
    cotton_columns = ['Region', 'Year', 'Output', 'Supply', 'Trade', 'Use', 'Ending_Stocks']
    cotton_columns = ['Region',   'Output', 'Supply', 'Trade', 'Use', 'Ending_Stocks']


    # Extract World, U.S., and Foreign data
    world_cotton, us_cotton, foreign_cotton = [], [], []
    current_section = None

    for line in cotton_lines:
        if 'World' in line:
            current_section = 'World'
            continue
        elif 'United States' in line:
            current_section = 'United States'
            continue
        elif 'Foreign' in line:
            current_section = 'Foreign'
            continue
        if current_section and re.match(r'^\s*(2022/23|2023/24|2024/25|Dec|Jan)', line):
            values = re.split(r'\s+', line.strip())
            if len(values) >= 5:
                values = [current_section] + values
                if current_section == 'World':
                    world_cotton.append(values)
                elif current_section == 'United States':
                    us_cotton.append(values)
                elif current_section == 'Foreign':
                    foreign_cotton.append(values)

    # Create DataFrames
    dataframes['World_Cotton'] = pd.DataFrame(world_cotton, columns=cotton_columns)
    dataframes['US_Cotton'] = pd.DataFrame(us_cotton, columns=cotton_columns)
    dataframes['Foreign_Cotton'] = pd.DataFrame(foreign_cotton, columns=cotton_columns)

# 3. World and U.S. Supply and Use for Oilseeds
oilseed_section = re.search(r'World and U.S. Supply and Use for Oilseeds.*?U.S. Wheat Supply and Use', text, re.DOTALL)
if oilseed_section:
    oilseed_lines = oilseed_section.group(0).split('\n')
    oilseed_lines = clean_lines(oilseed_lines)

    # Define columns for oilseeds
    oilseed_columns = ['Commodity', 'Year', 'Output', 'Supply', 'Trade', 'Use', 'Ending_Stocks']

    # Extract World, U.S., and Foreign data
    world_oilseeds, us_oilseeds, foreign_oilseeds = [], [], []
    current_section = None

    for line in oilseed_lines:
        if 'World' in line:
            current_section = 'World'
            continue
        elif 'United States' in line:
            current_section = 'United States'
            continue
        elif 'Foreign' in line:
            current_section = 'Foreign'
            continue
        if current_section and re.match(r'^\s*(Oilseeds|Oilmeals|Vegetable Oils)', line):
            values = re.split(r'\s+', line.strip())
            if len(values) >= 6:
                commodity = values[0]
                if len(values) > 6:
                    commodity = ' '.join(values[:2])
                    values = [commodity] + values[2:]
                if current_section == 'World':
                    world_oilseeds.append(values)
                elif current_section == 'United States':
                    us_oilseeds.append(values)
                elif current_section == 'Foreign':
                    foreign_oilseeds.append(values)

    # Create DataFrames
    dataframes['World_Oilseeds'] = pd.DataFrame(world_oilseeds, columns=oilseed_columns)
    dataframes['US_Oilseeds'] = pd.DataFrame(us_oilseeds, columns=oilseed_columns)
    dataframes['Foreign_Oilseeds'] = pd.DataFrame(foreign_oilseeds, columns=oilseed_columns)

# 4. U.S. Wheat Supply and Use
wheat_section = re.search(r'U.S. Wheat Supply and Use.*?U.S. Wheat by Class', text, re.DOTALL)
if wheat_section:
    wheat_lines = wheat_section.group(0).split('\n')
    wheat_lines = clean_lines(wheat_lines)

    # Define columns for wheat
    wheat_columns = ['Item', '2022/23', '2023/24_Est', '2024/25_Proj_Dec', '2024/25_Proj_Jan']

    # Extract data
    wheat_data = []
    for line in wheat_lines:
        if re.match(
                r'^(Area Planted|Area Harvested|Yield per Harvested Acre|Beginning Stocks|Production|Imports|Supply, Total|Food|Seed|Feed and Residual|Domestic, Total|Exports|Use, Total|Ending Stocks|Avg.FarmPrice)',
                line):
            values = re.split(r'\s+', line.strip())
            if len(values) >= 5:
                item = ' '.join(values[:-4])
                values = [item] + values[-4:]
                wheat_data.append(values)

    dataframes['US_Wheat'] = pd.DataFrame(wheat_data, columns=wheat_columns)

# 5. U.S. Feed Grain and Corn Supply and Use
feed_grain_section = re.search(r'U.S. Feed Grain and Corn Supply and Use.*?U.S. Sorghum, Barley, and Oats', text,
                               re.DOTALL)
if feed_grain_section:
    feed_grain_lines = feed_grain_section.group(0).split('\n')
    feed_grain_lines = clean_lines(feed_grain_lines)

    # Define columns for feed grains and corn
    feed_grain_columns = ['Item', '2022/23', '2023/24_Est', '2024/25_Proj_Dec', '2024/25_Proj_Jan']

    # Extract Feed Grains and Corn separately
    feed_grain_data, corn_data = [], []
    current_section = None

    for line in feed_grain_lines:
        if 'FEED GRAINS' in line:
            current_section = 'Feed Grains'
            continue
        elif 'CORN' in line:
            current_section = 'Corn'
            continue
        if current_section and re.match(
                r'^(Area Planted|Area Harvested|Yield per Harvested Acre|Beginning Stocks|Production|Imports|Supply, Total|Feed and Residual|Food, Seed & Industrial|Ethanol & by-products|Domestic, Total|Exports|Use, Total|Ending Stocks|Avg.FarmPrice)',
                line):
            values = re.split(r'\s+', line.strip())
            if len(values) >= 5:
                item = ' '.join(values[:-4])
                values = [item] + values[-4:]
                if current_section == 'Feed Grains':
                    feed_grain_data.append(values)
                elif current_section == 'Corn':
                    corn_data.append(values)

    dataframes['US_Feed_Grains'] = pd.DataFrame(feed_grain_data, columns=feed_grain_columns)
    dataframes['US_Corn'] = pd.DataFrame(corn_data, columns=feed_grain_columns)

# 6. U.S. Soybeans and Products Supply and Use
soybean_section = re.search(r'U.S. Soybeans and Products Supply and Use.*?U.S. Sugar Supply and Use', text, re.DOTALL)
if soybean_section:
    soybean_lines = soybean_section.group(0).split('\n')
    soybean_lines = clean_lines(soybean_lines)

    # Define columns for soybeans
    soybean_columns = ['Item', '2022/23', '2023/24_Est', '2024/25_Proj_Dec', '2024/25_Proj_Jan']

    # Extract Soybeans, Soybean Oil, and Soybean Meal separately
    soybean_data, soybean_oil_data, soybean_meal_data = [], [], []
    current_section = None

    for line in soybean_lines:
        if 'SOYBEANS' in line:
            current_section = 'Soybeans'
            continue
        elif 'SOYBEAN OIL' in line:
            current_section = 'Soybean Oil'
            continue
        elif 'SOYBEAN MEAL' in line:
            current_section = 'Soybean Meal'
            continue
        if current_section and re.match(
                r'^(Area Planted|Area Harvested|Yield per Harvested Acre|Beginning Stocks|Production|Imports|Supply, Total|Crushings|Exports|Seed|Residual|Use, Total|Ending Stocks|Avg.FarmPrice|Domestic Disappearance|Biofuel|Food, Feed & other Industrial)',
                line):
            values = re.split(r'\s+', line.strip())
            if len(values) >= 5:
                item = ' '.join(values[:-4])
                values = [item] + values[-4:]
                if current_section == 'Soybeans':
                    soybean_data.append(values)
                elif current_section == 'Soybean Oil':
                    soybean_oil_data.append(values)
                elif current_section == 'Soybean Meal':
                    soybean_meal_data.append(values)

    dataframes['US_Soybeans'] = pd.DataFrame(soybean_data, columns=soybean_columns)
    dataframes['US_Soybean_Oil'] = pd.DataFrame(soybean_oil_data, columns=soybean_columns)
    dataframes['US_Soybean_Meal'] = pd.DataFrame(soybean_meal_data, columns=soybean_columns)

# 7. U.S. Rice Supply and Use
rice_section = re.search(r'U.S. Rice Supply and Use.*?U.S. Soybeans and Products', text, re.DOTALL)
if rice_section:
    rice_lines = rice_section.group(0).split('\n')
    rice_lines = clean_lines(rice_lines)

    # Define columns for rice
    rice_columns = ['Item', '2022/23', '2023/24_Est', '2024/25_Proj_Dec', '2024/25_Proj_Jan']

    # Extract Total Rice, Long Grain Rice, and Medium & Short Grain Rice
    total_rice_data, long_grain_data, medium_short_data = [], [], []
    current_section = None

    for line in rice_lines:
        if 'TOTAL RICE' in line:
            current_section = 'Total Rice'
            continue
        elif 'LONG GRAIN RICE' in line:
            current_section = 'Long Grain Rice'
            continue
        elif 'MEDIUM & SHORT GRAIN RICE' in line:
            current_section = 'Medium & Short Grain Rice'
            continue
        if current_section and re.match(
                r'^(Area Planted|Area Harvested|Yield per Harvested Acre|Beginning Stocks|Production|Imports|Supply, Total|Domestic & Residual|Exports|Use, Total|Ending Stocks|Avg.FarmPrice|Avg.MillingYield)',
                line):
            values = re.split(r'\s+', line.strip())
            if len(values) >= 5:
                item = ' '.join(values[:-4])
                values = [item] + values[-4:]
                if current_section == 'Total Rice':
                    total_rice_data.append(values)
                elif current_section == 'Long Grain Rice':
                    long_grain_data.append(values)
                elif current_section == 'Medium & Short Grain Rice':
                    medium_short_data.append(values)

    dataframes['US_Total_Rice'] = pd.DataFrame(total_rice_data, columns=rice_columns)
    dataframes['US_Long_Grain_Rice'] = pd.DataFrame(long_grain_data, columns=rice_columns)
    dataframes['US_Medium_Short_Grain_Rice'] = pd.DataFrame(medium_short_data, columns=rice_columns)

# 8. U.S. Sugar Supply and Use
sugar_section = re.search(r'U.S. Sugar Supply and Use.*?Mexico Sugar Supply', text, re.DOTALL)
if sugar_section:
    sugar_lines = sugar_section.group(0).split('\n')
    sugar_lines = clean_lines(sugar_lines)

    # Define columns for sugar
    sugar_columns = ['Item', '2022/23', '2023/24_Est', '2024/25_Proj_Dec', '2024/25_Proj_Jan']

    # Extract data
    sugar_data = []
    for line in sugar_lines:
        if re.match(
                r'^(Beginning Stocks|Production|Beet Sugar|Cane Sugar|Florida|Louisiana|Texas|Imports|TRQ|Other Program|Non-program|Mexico|High-tier tariff/other|Total Supply|Exports|Deliveries|Food|Other|Miscellaneous|Total Use|Ending Stocks|Stocks to Use Ratio)',
                line):
            values = re.split(r'\s+', line.strip())
            if len(values) >= 5:
                item = ' '.join(values[:-4])
                values = [item] + values[-4:]
                sugar_data.append(values)

    dataframes['US_Sugar'] = pd.DataFrame(sugar_data, columns=sugar_columns)

# 9. U.S. Cotton Supply and Use
cotton_us_section = re.search(r'U.S. Cotton Supply and Use.*?World Soybean Supply', text, re.DOTALL)
if cotton_us_section:
    cotton_us_lines = cotton_us_section.group(0).split('\n')
    cotton_us_lines = clean_lines(cotton_us_lines)

    # Define columns for cotton
    cotton_us_columns = ['Item', '2022/23', '2023/24_Est', '2024/25_Proj_Dec', '2024/25_Proj_Jan']

    # Extract data
    cotton_us_data = []
    for line in cotton_us_lines:
        if re.match(
                r'^(Planted|Harvested|Yield per Harvested Acre|Beginning Stocks|Production|Imports|Supply, Total|Domestic Use|Exports|Use, Total|Ending Stocks)',
                line):
            values = re.split(r'\s+', line.strip())
            if len(values) >= 5:
                item = ' '.join(values[:-4])
                values = [item] + values[-4:]
                cotton_us_data.append(values)

    dataframes['US_Cotton_Detailed'] = pd.DataFrame(cotton_us_data, columns=cotton_us_columns)

# 10. U.S. Quarterly Animal Product Production
animal_section = re.search(r'U.S. Quarterly Animal Product Production.*?U.S. Quarterly Prices', text, re.DOTALL)
if animal_section:
    animal_lines = animal_section.group(0).split('\n')
    animal_lines = clean_lines(animal_lines)

    # Define columns for animal products
    animal_columns = ['Year_Quarter', 'Beef', 'Pork', 'Total_Red_Meat', 'Broiler', 'Turkey', 'Total_Poultry',
                      'Red_Meat_Poultry', 'Egg', 'Milk']

    # Extract data
    animal_data = []
    for line in animal_lines:
        if re.match(r'^(2023|2024|2025|I|II|III|IV|DecProj|Jan Est|JanProj)', line):
            values = re.split(r'\s+', line.strip())
            if len(values) >= 10:
                animal_data.append(values[:10])

    dataframes['US_Animal_Products'] = pd.DataFrame(animal_data, columns=animal_columns)

# Convert numeric columns to appropriate types
for df_name, df in dataframes.items():
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Print the DataFrames
for df_name, df in dataframes.items():
    print(f"\n{df_name} DataFrame:")
    print(df)
    print("\n")

# Optionally, save DataFrames to CSV
for df_name, df in dataframes.items():
    df.to_csv(f'{df_name}.csv', index=False)