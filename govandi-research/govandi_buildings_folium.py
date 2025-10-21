from arcgis.gis import GIS
import requests

# Connect to ArcGIS Online anonymously
gis = GIS()

# Define bounding box
extent = {
    "xmin": 14.368186,
    "ymin": 50.083203,
    "xmax": 14.375578,
    "ymax": 50.086188,
    "spatialReference": {"wkid": 4326}
}

# World Imagery service
item = gis.content.search("World Imagery owner:esri", "Imagery Layer")[0]
imagery_layer = item.layers[0]

# Export image (use image_format instead of format)
img_info = imagery_layer.export_image(bbox=extent, size=[1024,1024], f='json')

# Get the image URL from the response
img_url = img_info['href']

# Download and save the image
response = requests.get(img_url)
with open("prague.jpg", "wb") as f:
    f.write(response.content)

print("Image saved as prague.jpg")
