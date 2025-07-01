import requests
import geopandas as gpd
import pandas as pd
import os
import json

base_url = "https://geoportal.nic.in/nicgis/rest/services/SCHOOLGIS/Schooldata/MapServer/0/query"
save_path = "/Users/amit.yadav/Desktop/geo/india-shools/all_india_schools.geojson"

# Initialize
params = {
    "where": "1=1",
    "outFields": "*",
    "f": "geojson",
    "resultOffset": 647000,
    "resultRecordCount": 1000
}

# If file exists, load existing data and set offset accordingly
if os.path.exists(save_path):
    existing_gdf = gpd.read_file(save_path)
    features = json.loads(existing_gdf.to_json())['features']
    params['resultOffset'] = len(features)
    print(f"Resuming from offset: {params['resultOffset']} (already have {len(features)} features)")
else:
    features = []
    print("Starting fresh fetch...")

# Start fetching
while True:
    print(f"Fetching offset: {params['resultOffset']}")
    response = requests.get(base_url, params=params)
    try:
        data = response.json()
    except json.JSONDecodeError:
        print("Failed to decode JSON. Skipping.")
        break

    if 'features' not in data or not data['features']:
        print("No more features to fetch.")
        break

    print(f"Fetched {len(data['features'])} features")
    features.extend(data['features'])

    # Save to GeoJSON incrementally
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    gdf.to_file(save_path, driver="GeoJSON")
    print(f"Saved {len(gdf)} features to {save_path}")

    # Update offset
    params['resultOffset'] += 1000

print("Download complete. Total features:", len(features))
