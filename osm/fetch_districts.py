import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt 

goa_boundary = ox.geocode_to_gdf("Goa, India")

gdf = ox.features_from_place(
    "Goa, India",
    tags={"boundary": "administrative", "admin_level": "5"}
)

# Keep only polygons (district boundaries)
gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon']) & gdf['admin_level'].isin(['5'])]

gdf = gdf[gdf.geometry.centroid.within(goa_boundary.geometry.unary_union)]
print(gdf.crs)
print(gdf)
gdf.plot(
    figsize=(10, 10),
    edgecolor='black',
    column='name',
    legend=True,
    cmap='Set3',
)
plt.title("Goa Districts")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# gdf.to_file(
#     "./osm/resources/goa_districts.geojson", 
#     driver='GeoJSON'
# )
