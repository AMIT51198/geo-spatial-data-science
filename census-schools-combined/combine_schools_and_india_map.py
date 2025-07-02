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
gdf_india_map['district_name'] = gdf_india_map['district_name'].str.strip().str.lower()

# Ensure the CRS is set correctly
gdf_india_map = gdf_india_map.set_crs(epsg=4326, allow_override=True)
gdf_schools = gdf_schools.set_crs(epsg=4326, allow_override=True)   

# Perform spatial join to find schools within each district
schools_in_districts = gpd.sjoin(gdf_schools, gdf_india_map, how='left', predicate='within')
schools_in_districts = schools_in_districts.groupby('district_code').size().reset_index(name='school_count')

# Merge the school counts back to the district polygons
gdf_india_map = gdf_india_map.merge(schools_in_districts, on="district_code", how='left').fillna(0)
gdf_india_map.to_file("./census-schools-combined/resources/india_districts_with_school_counts.geojson", driver='GeoJSON')
gdf_india_map.plot(column='school_count', cmap='OrRd', legend=True, figsize=(15, 10), edgecolor='black')
plt.title("Number of Schools in Each District")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()