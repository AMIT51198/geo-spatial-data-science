import geopandas as gpd
from shapely.geometry import Point
from matplotlib import pyplot as plt
import folium
import pandas as pd

pd.set_option("display.max_columns", None)
school_gdf = gpd.read_file("/Users/amit.yadav/Desktop/geo/india-shools/all_india_schools_temp.geojson")
population_df = pd.read_excel("/Users/amit.yadav/Downloads/A-3_MDDS_Release.xls")

print([school_gdf["dtcode11"] == "629"])
# print(population_df)
# mp = folium.Map(location=[gdf["latitude"].mean(), gdf["longitude"].mean()], zoom_start=5)
# for _, series in gdf.iterrows():
#     folium.CircleMarker(
#         location=[series.latitude, series.longitude]
#     ).add_to(mp)

# mp.save("./interactive-maps/all_india_schools.html")
