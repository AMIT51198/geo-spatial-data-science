import geopandas as gpd
from matplotlib import pyplot as plt

school = gpd.read_file("/Users/amit.yadav/Desktop/geo/india-shools/all_india_schools.geojson")
print(len(school))
