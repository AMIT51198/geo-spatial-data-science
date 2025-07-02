import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import folium

pd.set_option("display.max_columns", None)

# Load the school data
gdf_schools = gpd.read_file("./india-schools/resources/all_india_schools.geojson")
gdf_schools['geometry'] = gpd.points_from_xy(gdf_schools['longitude'], gdf_schools['latitude'])
gdf_schools = gdf_schools[['longitude', 'latitude', 'schname', 'schtype', 'geometry']]

# Load the district polygons data
gdf_india_map = gpd.read_file('./India-Map/IND_adm2.shp')
gdf_india_map = gdf_india_map.rename(columns={
    'NAME_2': 'district_name',
    'ID_2': 'district_code',
    'NAME_1': 'state_name',
    'ID_1': 'state_code'
    })

gdf_india_map = gdf_india_map[['district_name', 'district_code', 'state_name', 'state_code', 'geometry']]

# Ensure the CRS is set correctly
gdf_india_map = gdf_india_map.set_crs(epsg=4326, allow_override=True)
gdf_schools = gdf_schools.set_crs(epsg=4326, allow_override=True)   

fig, ax = plt.subplots(figsize=(10, 10))

# Plot the district polygons
gdf_india_map.plot(ax=ax, color='lightgrey', edgecolor='black')

# Plot the school points on top
gdf_schools.plot(ax=ax, color='blue', markersize=1, label='Schools')

plt.title("Schools in India with District Polygons")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()

