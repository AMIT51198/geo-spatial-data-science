"""
Satellite Image Segmentation to Vector Polygons
Pipeline for detecting buildings, parks, and public areas in slum imagery
"""

import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, Polygon
import torch
import torch.nn as nn
from torchvision import transforms
from segmentation_models_pytorch import Unet, DeepLabV3Plus
import cv2
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PART 1: MODEL ARCHITECTURE
# ============================================

class BuildingSegmentationModel:
    """
    Wrapper for building/land-use segmentation models
    Uses U-Net or DeepLabV3+ for semantic segmentation
    """
    
    def __init__(self, model_type='unet', encoder='resnet34', num_classes=5):
        """
        Initialize segmentation model
        
        Args:
            model_type: 'unet' or 'deeplabv3plus'
            encoder: backbone encoder (resnet34, efficientnet-b4, etc.)
            num_classes: number of classes (e.g., background, buildings, parks, roads, public)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        if model_type == 'unet':
            self.model = Unet(
                encoder_name=encoder,
                encoder_weights='imagenet',
                classes=num_classes,
                activation=None
            )
        else:
            self.model = DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights='imagenet',
                classes=num_classes,
                activation=None
            )
        
        self.model = self.model.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_weights(self, weights_path):
        """Load pretrained weights"""
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
    
    def predict(self, image):
        """
        Predict segmentation mask for input image
        
        Args:
            image: numpy array (H, W, C) in RGB format
            
        Returns:
            mask: numpy array (H, W) with class predictions
        """
        # Preprocess
        if image.dtype == np.uint16:
            image = (image / 65535.0 * 255).astype(np.uint8)
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(img_tensor)
            mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
        
        return mask

# ============================================
# PART 2: INSTANCE SEGMENTATION (Separate Buildings)
# ============================================

def separate_instances(binary_mask, min_area=50):
    """
    Separate individual building instances using watershed algorithm
    
    Args:
        binary_mask: binary mask of buildings
        min_area: minimum area in pixels to consider as valid building
        
    Returns:
        instance_mask: labeled mask where each building has unique ID
    """
    # Distance transform
    dist_transform = ndimage.distance_transform_edt(binary_mask)
    
    # Find peaks (building centers)
    local_max = ndimage.maximum_filter(dist_transform, size=20) == dist_transform
    local_max = local_max & (dist_transform > 0)
    
    # Label markers
    markers, _ = ndimage.label(local_max)
    
    # Watershed segmentation
    labels = cv2.watershed(
        cv2.cvtColor((binary_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
        markers
    )
    
    # Clean up small regions
    instance_mask = np.zeros_like(labels)
    for i in range(1, labels.max() + 1):
        region = labels == i
        if region.sum() >= min_area:
            instance_mask[region] = i
    
    return instance_mask

# ============================================
# PART 3: VECTORIZATION
# ============================================

def mask_to_polygons(mask, transform, class_id, simplify_tolerance=1.0):
    """
    Convert raster mask to vector polygons
    
    Args:
        mask: numpy array with instance IDs
        transform: rasterio affine transform for georeferencing
        class_id: class identifier (e.g., 'building', 'park')
        simplify_tolerance: tolerance for polygon simplification
        
    Returns:
        GeoDataFrame with polygon geometries
    """
    # Extract shapes from raster
    mask_uint8 = mask.astype(np.uint8)
    polygon_generator = shapes(mask_uint8, mask=(mask_uint8 > 0), transform=transform)
    
    polygons = []
    for geom, value in polygon_generator:
        poly = shape(geom)
        
        # Simplify polygon to reduce vertices
        poly = poly.simplify(simplify_tolerance, preserve_topology=True)
        
        if poly.is_valid and poly.area > 0:
            polygons.append({
                'geometry': poly,
                'class': class_id,
                'instance_id': int(value),
                'area_m2': poly.area
            })
    
    return gpd.GeoDataFrame(polygons, crs='EPSG:4326')

# ============================================
# PART 4: MAIN PIPELINE
# ============================================

def satellite_to_vectors(
    image_path,
    model,
    output_shapefile,
    class_names={0: 'background', 1: 'building', 2: 'park', 3: 'road', 4: 'public'},
    separate_buildings=True,
    min_building_area=50
):
    """
    Complete pipeline: satellite image → segmentation → vector polygons
    
    Args:
        image_path: path to satellite image (GeoTIFF)
        model: trained segmentation model
        output_shapefile: output path for shapefile
        class_names: dictionary mapping class IDs to names
        separate_buildings: whether to separate individual buildings
        min_building_area: minimum building area in pixels
        
    Returns:
        GeoDataFrame with all polygons
    """
    print("Loading satellite image...")
    with rasterio.open(image_path) as src:
        image = src.read([1, 2, 3]).transpose(1, 2, 0)  # RGB
        transform = src.transform
        crs = src.crs
    
    print("Running segmentation model...")
    segmentation_mask = model.predict(image)
    
    all_polygons = []
    
    # Process each class
    for class_id, class_name in class_names.items():
        if class_id == 0:  # Skip background
            continue
        
        print(f"Processing class: {class_name}")
        class_mask = (segmentation_mask == class_id).astype(np.uint8)
        
        if class_mask.sum() == 0:
            continue
        
        # Separate building instances
        if class_name == 'building' and separate_buildings:
            instance_mask = separate_instances(class_mask, min_building_area)
            gdf = mask_to_polygons(instance_mask, transform, class_name)
        else:
            gdf = mask_to_polygons(class_mask, transform, class_name)
        
        all_polygons.append(gdf)
    
    # Combine all classes
    print("Combining polygons...")
    final_gdf = gpd.GeoDataFrame(pd.concat(all_polygons, ignore_index=True), crs=crs)
    
    # Save to shapefile
    print(f"Saving to {output_shapefile}...")
    final_gdf.to_file(output_shapefile)
    
    print(f"Complete! Generated {len(final_gdf)} polygons")
    return final_gdf

# ============================================
# PART 5: TRAINING UTILITIES (if you need to train)
# ============================================

def train_model(train_images, train_masks, val_images, val_masks, 
                num_classes=5, epochs=50, batch_size=8):
    """
    Train segmentation model on labeled data
    
    Args:
        train_images: list of training image paths
        train_masks: list of training mask paths (same order)
        val_images: validation images
        val_masks: validation masks
        num_classes: number of classes
        epochs: training epochs
        batch_size: batch size
    """
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import albumentations as A
    
    # Initialize model
    model = BuildingSegmentationModel(num_classes=num_classes)
    optimizer = optim.Adam(model.model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop would go here
    # This is a template - you'd need to implement DataLoader, training loop, etc.
    print("Training model...")
    print("Note: Implement full training loop based on your dataset structure")
    
    return model

# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    # Initialize model
    model = BuildingSegmentationModel(
        model_type='unet',
        encoder='resnet34',
        num_classes=5
    )
    
    # Load pretrained weights (you need to train or download these)
    # model.load_weights('path/to/weights.pth')
    
    # Process satellite image
    result_gdf = satellite_to_vectors(
        image_path='slum_satellite_image.tif',
        model=model,
        output_shapefile='slum_buildings.shp',
        class_names={
            0: 'background',
            1: 'building',
            2: 'park',
            3: 'road',
            4: 'public_area'
        },
        separate_buildings=True,
        min_building_area=50
    )
    
    print(f"Statistics:\n{result_gdf.groupby('class').size()}")