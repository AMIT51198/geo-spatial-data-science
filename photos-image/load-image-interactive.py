import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import folium
import geopandas as gpd
from shapely.geometry import Point

def get_exif_data(img_path):
    """Extract EXIF data from image file"""
    image = Image.open(img_path)
    exif_data = image._getexif()
    if not exif_data:
        return {}
    exif = {}
    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)
        if tag == "GPSInfo":
            gps_data = {}
            for key in value:
                name = GPSTAGS.get(key, key)
                gps_data[name] = value[key]
            exif["GPSInfo"] = gps_data
        else:
            exif[tag] = value
    return exif

def convert_to_degrees(value):
    """Convert GPS coordinates to decimal degrees"""
    print(value)
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)

def get_lat_lon(gps_info):
    try:
        lat = convert_to_degrees(gps_info["GPSLatitude"])
        if gps_info["GPSLatitudeRef"] != "N":
            lat = -lat
        lon = convert_to_degrees(gps_info["GPSLongitude"])
        if gps_info["GPSLongitudeRef"] != "E":
            lon = -lon
        return lat, lon
    except KeyError:
        return None, None

# 📁 Folder with images
img_dir = "/Users/amit.yadav/Downloads/Test"
data = []

for fname in os.listdir(img_dir):
    print(fname)
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(img_dir, fname)
        exif = get_exif_data(path)
        gps_info = exif.get("GPSInfo", {})
        lat, lon = get_lat_lon(gps_info)
        if lat and lon:
            data.append({
                "file": fname,
                "path": path,
                "latitude": lat,
                "longitude": lon,
                "datetime": exif.get("DateTime", "Unknown"),
                "camera": exif.get("Model", "Unknown")
            })

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(data, geometry=[Point(x["longitude"], x["latitude"]) for x in data])
gdf.set_crs(epsg=4326, inplace=True)

print(gdf)
# 🌍 Create map
m = folium.Map(location=[gdf["latitude"].mean(), gdf["longitude"].mean()], zoom_start=10)

# Add markers
for _, row in gdf.iterrows():
    iframe = folium.IFrame(
        f'''
        <b>{row["file"]}</b><br>
        <img src="http://localhost:8000/{row["file"]}" width="300"><br>
        Date: {row["datetime"]}<br>
        Camera: {row["camera"]}
        ''',
        width=320,
        height=340
    )
    popup = folium.Popup(iframe, max_width=400)
    folium.Marker(
        location=[row.latitude, row.longitude],
        popup=popup,
        icon=folium.Icon(color="green")
    ).add_to(m)

# Save map
m.save("./photos-image/map_from_exif.html")
