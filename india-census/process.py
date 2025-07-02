import pandas as pd
import glob
import os

pd.set_option('display.max_columns', None)  # Show all columns in the DataFrame
def process_population_data(file_path: str) -> pd.DataFrame:
    """
    Process the population data from the given Excel file.

    Args:
        file_path (str): Path to the Excel file containing population data.

    Returns:
        pd.DataFrame: Processed DataFrame with relevant columns.
    """
    df = pd.read_excel(file_path, header=1)
    df = df.dropna()
    df = df.rename(columns={
        'State': 'state_code',
        'Distt.': 'district_code',
        'Area Name': 'district_name',
        'Total/': 'area_type',
        'Age-group': "age_group",
        '    Total': "total_population",
        'Unnamed: 7': "total_males",
        'Unnamed: 8': "total_females",
        'Illiterate': "illiterate_population",
        'Unnamed: 10': "illiterate_males",
        'Unnamed: 11': "illiterate_females",
        'Literate': "literate_population",
        'Unnamed: 13': "literate_males",
        'Unnamed: 14': "literate_females",
        })
    df = df[(df['district_code'] != '000')]
    df['district_name'] = df['district_name'].str.strip().replace(r'^District\s*-\s*', '', regex=True).str.lower()
    df = df[[
        'state_code', 
        'district_code', 
        'district_name', 
        'area_type', 
        'age_group',
        'total_population', 
        'total_males', 
        'total_females', 
        'illiterate_population',
        'illiterate_males',
        'illiterate_females',
        'literate_population',
        'literate_males',
        'literate_females'
        ]]
    
    # print(df.head()) 
    # print(df.columns)
    return df

# Get all matching files and process them
file_pattern = "./india-census/resources/raw-data/population-state-2011/DDW-*.xlsx"
file_list = glob.glob(file_pattern)
all_dfs = [process_population_data(file) for file in file_list]
pd.concat(all_dfs, ignore_index=True).to_csv("./india-census/resources/processed-data/processed_population_data.csv", index=False)
