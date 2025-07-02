import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from rapidfuzz import process, fuzz

pd.set_option("display.max_columns", None)

# Load the india_distrcts_with_school_counts.geojson file
gdf_india_map = gpd.read_file("./census-schools-combined/resources/india_districts_with_school_counts.geojson") 

gdf_india_map = gdf_india_map.drop_duplicates(subset=['district_name'])

# Load the processed population data
df_population = pd.read_csv("./india-census/resources/processed-data/processed_population_data.csv")

# Consider only the rows which contain total population and not agewise population
df_population = df_population.loc[
    (df_population['age_group'] == 'All ages') & 
    (df_population['area_type'] == 'Total')
    ]

df_population = df_population.drop_duplicates(subset=['district_name'])

# Create a mapping from gdf_india_map district_name to best match in df_population
matches = df_population['district_name'].tolist()
def get_best_match(name):
    match, score, idx = process.extractOne(name, matches, scorer=fuzz.ratio)
    return match if score > 70 else None  # You can adjust the threshold

gdf_india_map['district_name_match'] = gdf_india_map['district_name'].apply(get_best_match)

# Now merge on the new matched column
gdf_india_map = gdf_india_map.merge(df_population, left_on='district_name_match', right_on='district_name', how='left')

# # Fill NaN values in population columns with 0
gdf_india_map['total_population'] = gdf_india_map['total_population'].fillna(0)
gdf_india_map['total_males'] = gdf_india_map['total_males'].fillna(0)
gdf_india_map['total_females'] = gdf_india_map['total_females'].fillna(0)

print(gdf_india_map.loc[gdf_india_map['total_population'] == 0].shape[0])
print(gdf_india_map.loc[gdf_india_map['total_population'] == 0, 'district_name_x'])   # Number of districts with total populatio

gdf_india_map['ratio_population_school'] = gdf_india_map['total_population'] / gdf_india_map['school_count']

gdf_india_map.to_file(
    "./census-schools-combined/resources/india_districts_with_population_and_school_counts.geojson", 
    driver='GeoJSON'
)

# # Plot the ratio of population to school count
fig, ax = plt.subplots(figsize=(15, 10))
gdf_india_map.plot(column='ratio_population_school', cmap='OrRd', legend=True, ax=ax, edgecolor='black')
plt.title("Ratio of Population to School Count in Each District")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
