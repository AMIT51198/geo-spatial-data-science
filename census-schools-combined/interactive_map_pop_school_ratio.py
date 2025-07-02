import folium as fm
import geopandas as gpd

gdf = gpd.read_file("./census-schools-combined/resources/india_districts_with_population_and_school_counts.geojson")

m = fm.Map(location=[20.5937, 78.9629], zoom_start=5)

fm.Choropleth(
    geo_data=gdf,
    data=gdf,
    columns=['district_name_x', 'ratio_population_school'],
    key_on='feature.properties.district_name_x',
    fill_color='OrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    line_color='black',
    legend_name='Population to School Ratio'
).add_to(m)

fm.GeoJson(
    gdf,
    name="Districts",
    tooltip=fm.GeoJsonTooltip(
        fields=['district_name_x', 'ratio_population_school'],
        aliases=['District:', 'Population/School Ratio:'],
        localize=True
    )
).add_to(m)

m.save("./interactive-maps/population_school_ratio_map.html")
