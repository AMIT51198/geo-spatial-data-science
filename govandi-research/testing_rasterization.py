"""
Script for rasterization testing and visualization.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from glob import glob

sat_images_dir = "/Users/amit.yadav/Desktop/geo/govandi-research/data-for-model/images"
mask_shapefiles_dir = "/Users/amit.yadav/Desktop/geo/govandi-research/data-for-model/shapefiles"

image_paths = sorted(glob(os.path.join(sat_images_dir, "*.jpeg"))) # Assuming JPEG format
shapefile_paths = sorted(glob(os.path.join(mask_shapefiles_dir, "*.shp")))

for img_path, shp_path in zip(image_paths, shapefile_paths):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:,:,::-1]
    buildings_gdf = gpd.read_file(shp_path)
    buildings_gdf = buildings_gdf[buildings_gdf.geometry.type == 'Polygon']
    if buildings_gdf.empty:
        print(f"Warning: No valid polygons found in {os.path.basename(shp_path)}. Skipping overlay.")
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        plt.title(f"Satellite Image: {os.path.basename(img_path)} (No polygons)")
        plt.axis('off')
        plt.show()
        continue

    # Rasterize polygons to image pixel grid
    height, width = img.shape[0], img.shape[1]
    minx, miny, maxx, maxy = buildings_gdf.total_bounds
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    shapes = [(geom, 1) for geom in buildings_gdf.geometry if geom.is_valid]
    mask_img = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    # Overlay mask on image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.imshow(mask_img, cmap='Reds', alpha=0.5)
    plt.title(f"Rasterized Mask Overlay: {os.path.basename(img_path)}")
    plt.axis('off')
    plt.show()