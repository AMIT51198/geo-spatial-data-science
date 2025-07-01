import geodatasets
import geopandas
from matplotlib import pyplot as plt
import pandas as pd

chicago = geopandas.read_file(geodatasets.get_path('geoda.chicago_commpop'))
groceries = geopandas.read_file(geodatasets.get_path('geoda.groceries')).to_crs('EPSG:4326')

#Attribute Joins
chicago_shapes = chicago[['geometry', 'NID']]
chicago_name = chicago[['community', 'NID']]

#Appending Geoseries
joined = pd.concat([chicago.geometry, groceries.geometry])

#Appending Geodataframes
douglas = chicago[chicago.community == 'DOUGLAS']
oakland = chicago[chicago.community == 'OAKLAND']
appended = pd.concat([douglas, oakland])

#Attribute Join
chicago_shapes.merge(chicago_name, on='NID')

#Spatial Join
joinedDf = groceries.sjoin(chicago_shapes, how="inner", predicate="intersects")
joinedDf["geometry"].plot()
plt.show()