"""
Mumbai places analysis script.
"""

import pandas as pd
import re
import requests
import time

API_KEY = "AIzaSyCxg0xA00Ayosg_9oA-QTMqF2soN6k5M0Y"

# Load Excel
df = pd.read_excel("/Users/amit.yadav/Downloads/Google maps scraping video and film production v11.xlsx").head(10)

# Extract lat/lon from Google Maps URLs
def extract_lat_lon(url):
    match = re.search(r'3d([-+]?\d*\.\d+|\d+)!4d([-+]?\d*\.\d+|\d+)', str(url))
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

df['latitude'], df['longitude'] = zip(*df.iloc[:, 0].map(extract_lat_lon))

# Reverse geocode function
def get_place_info(lat, lon):
    print(f"🔍 Processing coordinates: {lat}, {lon}")
    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={API_KEY}"
        response = requests.get(url)
        data = response.json()

        print(data)
        if data['results']:
            address = data['results'][0]['formatted_address']
            types = data['results'][0].get('types', [])
            return address, ", ".join(types)
        return "Not Found", "Unknown"
    except Exception as e:
        return "Error", str(e)

# Apply to DataFrame
results = df[['latitude', 'longitude']].dropna().apply(
    lambda row: get_place_info(row['latitude'], row['longitude']),
    axis=1
)

df[['place_address', 'place_type']] = pd.DataFrame(results.tolist(), index=results.index)

# Save results
df.to_json("Place_Info_Enriched.json", index=False)

import folium
import pandas as pd

# Load the enriched data
df = pd.read_json("Place_Info_Enriched.json")

df = df.dropna(subset=['latitude', 'longitude'])

# Create a folium map centered on the mean coordinates
center = [df['latitude'].mean(), df['longitude'].mean()]
m = folium.Map(location=center, zoom_start=12)

# Add markers for each place
for _, row in df.iterrows():
    popup_text = f"Business: {row['place_address']}<br>Type: {row['place_type']}"
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=popup_text,
        tooltip=row['place_address']
    ).add_to(m)

# Save the map
m.save("mumbai_places_map.html")
