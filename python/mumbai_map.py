"""
Mumbai map visualization script.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import folium

gdf = gpd.read_file("geo")

gdf.plot(edgecolor='black', color='lightblue')
plt.title("GeoJSON Polygons")
plt.show()

# Center map on the first polygon's centroid
center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
m = folium.Map(location=center, zoom_start=10)

# Add GeoJSON layer
folium.GeoJson(gdf).add_to(m)

# Save to HTML
m.save("interactive_geojson_map.html")

