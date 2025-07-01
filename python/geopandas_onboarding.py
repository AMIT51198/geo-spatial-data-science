import geopandas
from geodatasets import get_path
import matplotlib as mp
path = get_path('nybb')
gdf = geopandas.read_file(path)
gdf = gdf.set_index('BoroName')
gdf['area']= gdf.area
gdf['centroid']= gdf.centroid
gdf['boundary']= gdf.boundary
gdf['distance_from_staten'] = gdf['centroid'].distance(gdf['centroid'].iloc[0])

#plotting
gdf = gdf.set_geometry('centroid')
ax = gdf['centroid'].plot(color="yellow")
gdf['boundary'].plot(ax=ax, color="black")

m = gdf['geometry'].explore(legend=True)
m.save("/Users/amit.yadav/Desktop/geo/interactive-maps/nyc-area.html") 

#back to same geometry
gdf = gdf.set_geometry('geometry')
gdf["convex_hull"] = gdf.convex_hull
ax = gdf['convex_hull'].plot(alpha=0.5)
gdf['boundary'].plot(ax = ax, linewidth=0.5)


#buffer around a geometry
gdf['buffered'] = gdf.buffer(100000)
ax = gdf['buffered'].plot(alpha=0.3)
gdf['geometry'].plot(ax=ax, color="green", linewidth=2)


#buffered centroid
new_gdf = gdf.set_geometry("centroid")
new_gdf['buffered_centroid'] = new_gdf.buffer(10000)
new_gdf['within'] = new_gdf['buffered_centroid'].within(new_gdf['geometry'])
first_ax = new_gdf["geometry"].plot(alpha=0.25, linewidth=1.5, edgecolor="black")
new_gdf["buffered_centroid"].plot(ax=first_ax, color="red")

#crs
print(gdf.crs)
gdf = gdf.to_crs("EPSG:4326")
gdf.plot()
mp.pyplot.show()
