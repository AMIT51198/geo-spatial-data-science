import geopandas as gs
from matplotlib import pyplot as plt

indian_district_gdf = gs.read_file("/Users/amit.yadav/Desktop/geo/India-map/IND_adm2.shp")
indian_district_gdf = indian_district_gdf.to_crs('EPSG:3857')

#Test Plot - Uncomment it so that it can work
#indian_district_gdf["geometry"].plot()

#Interactive Map
interactive_indian_district = indian_district_gdf["geometry"].explore()
interactive_indian_district.save("/Users/amit.yadav/Desktop/geo/interactive-maps/india-district.html") 

#Boundary of the districts
indian_district_gdf['boundary'] = indian_district_gdf.boundary
indian_district_gdf['boundary'].plot(linewidth=1.5, edgecolor="black")

#Area of the district
indian_district_gdf['area'] = indian_district_gdf.area

interactive_indian_district = indian_district_gdf.explore('area')
interactive_indian_district.save("/Users/amit.yadav/Desktop/geo/interactive-maps/india-district.html") 

#Dissolve district into state
indian_state_gdf = indian_district_gdf.dissolve(by='NAME_1', aggfunc={'area': 'sum'})
indian_state_gdf.explore('area').save("/Users/amit.yadav/Desktop/geo/interactive-maps/india-state.html") 
print(indian_state_gdf.dtypes)

#Dissolve into one country
indian_gdf = indian_district_gdf.dissolve(aggfunc={'area': 'sum'})
indian_gdf.explore('area').save("/Users/amit.yadav/Desktop/geo/interactive-maps/india.html") 
print(indian_gdf.dtypes)
plt.show()