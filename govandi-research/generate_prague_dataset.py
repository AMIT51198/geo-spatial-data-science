"""
Automated dataset creation for Prague building footprints.
"""

import os
import geopandas as gpd
import numpy as np
import random
import requests
from shapely.geometry import box

# Directory setup
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data-for-model')
images_dir = os.path.join(data_dir, 'images')
shapefiles_dir = os.path.join(data_dir, 'shapefiles')
os.makedirs(images_dir, exist_ok=True)
os.makedirs(shapefiles_dir, exist_ok=True)

# Prague bounds (approx)
# Use provided bounding boxes for min and max lat/lon
# Point 1: 50.108523, 14.379718
# Point 2: 50.098644, 14.408865
min_lat = min(50.108523, 50.056170)
max_lat = max(50.108523, 50.056170)
min_lon = min(14.509914, 14.408865)
max_lon = max(14.509914, 14.408865)
box_height = 0.00225  # ~250m latitude
box_width = 0.0035    # ~250m longitude

# Overpass Turbo API endpoint
OVERPASS_URL = "http://overpass-api.de/api/interpreter"

# Helper: Generate random bounding boxes in Prague
boxes = []
for i in range(50):
    lat0 = random.uniform(min_lat, max_lat - box_height)
    lon0 = random.uniform(min_lon, max_lon - box_width)
    lat1 = lat0 + box_height
    lon1 = lon0 + box_width
    boxes.append((lat0, lon0, lat1, lon1))

# Helper: Get next image index based on existing files
existing_images = [f for f in os.listdir(images_dir) if f.startswith('image') and f.endswith('.jpeg')]
existing_indices = []
for fname in existing_images:
    try:
        idx = int(fname.replace('image', '').replace('.jpeg', ''))
        existing_indices.append(idx)
    except ValueError:
        pass
next_idx = max(existing_indices) + 1 if existing_indices else 1

for i, (lat0, lon0, lat1, lon1) in enumerate(boxes):
    idx = next_idx + i
    print(f"\nProcessing bbox {idx}: lat0={lat0}, lon0={lon0}, lat1={lat1}, lon1={lon1}")
    # 1. Download satellite image using Satelite-Aerial-Image-Retrieval-with-Bing
    image_name = f"image{idx}.jpeg"
    image_path = os.path.join(images_dir, image_name)
    print(f"Retrieving satellite image: {image_name} ...")
    cmd = f"python /Users/amit.yadav/Desktop/geo/Satellite-Aerial-Image-Retrieval-with-Bing/aerialImageRetrieval.py {lat1} {lon0} {lat0} {lon1} {image_path}"
    result = os.system(cmd)
    if result != 0:
        print(f"Failed to retrieve satellite image for bbox {idx}")
        continue
    print(f"Satellite image saved: {image_path}")

    # 2. Download building footprints from Overpass Turbo API
    print(f"Querying Overpass Turbo API for buildings ...")
    query = f"""
    [out:json][timeout:90];
    (
      way[\"building\"]({lat0},{lon0},{lat1},{lon1});
      relation[\"building\"]({lat0},{lon0},{lat1},{lon1});
    );
    out geom;
    """
    response = requests.post(OVERPASS_URL, data={'data': query})
    if response.status_code != 200:
        print(f"Failed to fetch OSM data for bbox {idx}")
        continue
    print(f"Received OSM data for bbox {idx}")
    data = response.json()
    # Convert Overpass JSON to GeoDataFrame
    features = []
    for element in data['elements']:
        if 'geometry' in element:
            coords = [(pt['lon'], pt['lat']) for pt in element['geometry']]
            if element['type'] == 'way':
                features.append({'geometry': box(min([c[0] for c in coords]), min([c[1] for c in coords]), max([c[0] for c in coords]), max([c[1] for c in coords])), 'osm_id': element['id']})
    if features:
        gdf = gpd.GeoDataFrame(features, geometry=[f['geometry'] for f in features], crs='EPSG:4326')
        shp_name = f"image{idx}.shp"
        shp_path = os.path.join(shapefiles_dir, shp_name)
        gdf.to_file(shp_path)
        print(f"Shapefile saved: {shp_path}")
    else:
        print(f"No building features found for bbox {idx}, skipping shapefile creation.")
    print(f"Finished bbox {idx}")

print("\nDataset generation complete.")
