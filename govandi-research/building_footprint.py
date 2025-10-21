"""
Building footprint segmentation and visualization workflow.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import geopandas as gpd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from glob import glob

# ===============================
# 1. Load satellite images and shapefiles
# ===============================

sat_images_dir = "/Users/amit.yadav/Desktop/geo/govandi-research/data-for-model/images"
mask_shapefiles_dir = "/Users/amit.yadav/Desktop/geo/govandi-research/data-for-model/shapefiles"

image_paths = sorted(glob(os.path.join(sat_images_dir, "*.jpeg"))) # Assuming JPEG format
shapefile_paths = sorted(glob(os.path.join(mask_shapefiles_dir, "*.shp")))

# Ensure we have matching images and shapefiles
if len(image_paths) != len(shapefile_paths):
    print("Warning: Number of images and shapefiles do not match.")
    # You might want to add more robust matching based on filenames

print(f"Found {len(image_paths)} images and {len(shapefile_paths)} shapefiles.")

# Load all images and shapefiles (this might consume a lot of memory for large datasets)
images = []
masks = []
image_metadata = [] # Store metadata like bounds and shape

for img_path, shp_path in zip(image_paths, shapefile_paths):
    # Check if base names match (simple check)
    if os.path.splitext(os.path.basename(img_path))[0] != os.path.splitext(os.path.basename(shp_path))[0]:
        print(f"Skipping mismatching pair: {os.path.basename(img_path)} and {os.path.basename(shp_path)}")
        continue

    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:,:,::-1]  # BGR->RGB
    images.append(img)

    # Load shapefile and get bounds for rasterization
    buildings_gdf = gpd.read_file(shp_path)
    buildings_gdf = buildings_gdf[buildings_gdf.geometry.type == 'Polygon']
    if buildings_gdf.empty:
        print(f"Warning: No valid polygons found in {os.path.basename(shp_path)}. Creating empty mask.")
        mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        masks.append(mask_img)
        image_metadata.append({'shape': img.shape, 'bounds': None}) # Store shape, no bounds if no polygons
        continue


    # Store image shape and bounds for later georeferencing
    minx, miny, maxx, maxy = buildings_gdf.total_bounds
    image_metadata.append({'shape': img.shape, 'bounds': (minx, miny, maxx, maxy)})

    # Robust rasterization using rasterio.features.rasterize
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    height, width = img.shape[0], img.shape[1]
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    shapes = [(geom, 1) for geom in buildings_gdf.geometry if geom.is_valid]
    mask_img = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    masks.append(mask_img)

    # Visualization: show satellite image and mask overlay for quality check
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Satellite Image: {os.path.basename(img_path)}")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(mask_img, cmap='Reds', alpha=0.5)
    plt.title(f"Rasterized Mask Overlay: {os.path.basename(shp_path)}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

print(f"Loaded {len(images)} images and {len(masks)} masks.")

# Pad images and masks to a common tile size
TILE_SIZE = 256

def pad_to_tile(img, tile_size):
    H, W = img.shape[:2]
    pad_h = (tile_size - H % tile_size) % tile_size
    pad_w = (tile_size - W % tile_size) % tile_size
    if pad_h or pad_w:
        if img.ndim == 3:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0,0)), mode='constant')
        else:
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')
    return img

padded_images = [pad_to_tile(img, TILE_SIZE) for img in images]
padded_masks = [pad_to_tile(mask, TILE_SIZE) for mask in masks]

# ===============================
# 2. Dataset class and preprocessing (adapted for multiple images)
# ===============================
class BuildingDatasetMultiImage(Dataset):
    def __init__(self, images, masks, tile_size=256, transforms=None):
        self.images = images
        self.masks = masks
        self.tile_size = tile_size
        self.transforms = transforms

        # Generate tiles from all images
        self.tiles = []
        for img, mask in zip(self.images, self.masks):
            H, W, _ = img.shape
            for i in range(0, H, tile_size):
                for j in range(0, W, tile_size):
                    img_tile = img[i:i+tile_size, j:j+tile_size]
                    mask_tile = mask[i:i+tile_size, j:j+tile_size]
                    if img_tile.shape[0]==tile_size and img_tile.shape[1]==tile_size:
                        self.tiles.append((img_tile, mask_tile))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        img_tile, mask_tile = self.tiles[idx]
        if self.transforms:
            augmented = self.transforms(image=img_tile, mask=mask_tile)
            img_tile = augmented['image']
            mask_tile = augmented['mask']
        return img_tile.float(), mask_tile.long()

# Albumentations transforms (data augmentation + normalization)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(), # Normalize image pixels
    ToTensorV2() # Convert to PyTorch tensor
])

dataset_multi = BuildingDatasetMultiImage(padded_images, padded_masks, tile_size=TILE_SIZE, transforms=transform)
dataloader_multi = DataLoader(dataset_multi, batch_size=8, shuffle=True) # Increased batch size

print(f"Created dataset with {len(dataset_multi)} tiles.")

# ===============================
# Define the DeepLabV3+ model (moved from previous cell)
# ===============================
class DeepLabV3PlusWithResnet50(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(DeepLabV3PlusWithResnet50, self).__init__()

        # Use the smp.DeepLabV3Plus class directly and specify the encoder
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",  # Choose encoder, e.g. resnet34, resnet50, resnet101
            encoder_weights="imagenet", # Use pre-trained weights
            in_channels=in_channels,    # Model input channels (images have 3 channels)
            classes=out_channels,       # Model output channels (number of classes in your dataset)
        )

    def forward(self, x):
        return self.model(x)

# Instantiate the new model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_advanced = DeepLabV3PlusWithResnet50().to(device)

# Define loss function and optimizer
criterion_advanced = nn.CrossEntropyLoss()
optimizer_advanced = torch.optim.Adam(model_advanced.parameters(), lr=1e-4) # Start with a smaller learning rate

print("DeepLabV3+ model with pre-trained ResNet50 encoder instantiated.")

# ===============================
# Function to calculate evaluation metrics (added back)
# ===============================
def calculate_metrics(pred_mask, true_mask):
    """Calculates IoU and F1-score for binary segmentation masks."""
    # Ensure masks are binary (0 or 1)
    pred_mask = (pred_mask > 0).astype(np.uint8)
    true_mask = (true_mask > 0).astype(np.uint8)

    # Calculate True Positives, False Positives, False Negatives
    intersection = np.sum(pred_mask * true_mask)
    union = np.sum(pred_mask) + np.sum(true_mask) - intersection
    tp = intersection
    fp = np.sum(pred_mask) - tp
    fn = np.sum(true_mask) - tp

    # Avoid division by zero
    iou = tp / (union + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return iou, f1


# ===============================
# 3. Train the DeepLabV3+ model
# ===============================
# Assuming model_advanced, criterion_advanced, and optimizer_advanced are already defined from the previous cell

epochs_advanced = 1 # Increase epochs for training on more data
print(f"\nTraining the DeepLabV3+ model for {epochs_advanced} epochs...")

for epoch in range(epochs_advanced):
    model_advanced.train()
    running_loss_advanced = 0
    for imgs, masks in dataloader_multi:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer_advanced.zero_grad()
        outputs_advanced = model_advanced(imgs)
        loss_advanced = criterion_advanced(outputs_advanced, masks)
        loss_advanced.backward()
        optimizer_advanced.step()

        running_loss_advanced += loss_advanced.item()

    print(f"Epoch {epoch+1}/{epochs_advanced}, Loss={running_loss_advanced/len(dataloader_multi):.4f}")

print("\nTraining finished.")

# ===============================
# 4. Predict and Evaluate on full images
# ===============================
model_advanced.eval()

all_pred_masks = []
all_true_masks = [] # Store original masks for evaluation

# Apply the same transformations as the training data (excluding augmentations) for prediction
eval_transform = A.Compose([
    A.Normalize(),
    ToTensorV2()
])

with torch.no_grad():
    for i in range(len(padded_images)): # Iterate using index
        img = padded_images[i]
        mask = padded_masks[i]
        original_img_shape = images[i].shape # Get original shape using the same index

        H, W, _ = img.shape
        pred_mask_full = np.zeros((H,W), dtype=np.uint8)

        for tile_i in range(0,H,TILE_SIZE):
            for tile_j in range(0,W,TILE_SIZE):
                img_tile = img[tile_i:tile_i+TILE_SIZE, tile_j:tile_j+TILE_SIZE]
                # No need to check tile size here because images are padded

                augmented = eval_transform(image=img_tile, mask=np.zeros((TILE_SIZE, TILE_SIZE))) # Mask is a placeholder
                img_tile_norm = augmented['image']

                img_tensor = img_tile_norm.unsqueeze(0).to(device) # Add batch dimension

                output = model_advanced(img_tensor)
                pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
                pred_mask_full[tile_i:tile_i+TILE_SIZE, tile_j:tile_j+TILE_SIZE] = pred

        all_pred_masks.append(pred_mask_full)
        all_true_masks.append(mask)


# Evaluate metrics for each image and average
iou_scores = []
f1_scores = []

for i in range(len(all_pred_masks)): # Iterate using index
    pred_mask = all_pred_masks[i]
    true_mask = all_true_masks[i]
    original_img_shape = images[i].shape # Get original shape using the same index

     # Unpad the masks to their original size before evaluation
    pred_mask_unpadded = pred_mask[:original_img_shape[0], :original_img_shape[1]]
    true_mask_unpadded = true_mask[:original_img_shape[0], :original_img_shape[1]]


    iou, f1 = calculate_metrics(pred_mask_unpadded, true_mask_unpadded)
    iou_scores.append(iou)
    f1_scores.append(f1)

avg_iou = np.mean(iou_scores)
avg_f1 = np.mean(f1_scores)


print(f"\nAverage Evaluation Metrics (DeepLabV3+ Model):")
print(f"  Mean Intersection over Union (IoU): {avg_iou:.4f}")
print(f"  Mean F1-Score: {avg_f1:.4f}")


# ===============================
# 5. Visualize results for a few examples
# ===============================
print("\nVisualizing results for a few images:")
num_examples_to_show = min(3, len(images))

for i in range(num_examples_to_show):
    plt.figure(figsize=(15, 5))

    # Original satellite image (unpadded)
    plt.subplot(1, 3, 1)
    plt.imshow(images[i])
    plt.title(f"Original Image {i+1}")
    plt.axis('off')

    # Ground truth mask (unpadded)
    plt.subplot(1, 3, 2)
    # Ensure mask is on CPU and is a numpy array for plotting
    plt.imshow(all_true_masks[i][:images[i].shape[0], :images[i].shape[1]], cmap='gray')
    plt.title(f"Ground Truth Mask {i+1}")
    plt.axis('off')

    # Predicted mask (unpadded)
    plt.subplot(1, 3, 3)
    plt.imshow(all_pred_masks[i][:images[i].shape[0], :images[i].shape[1]], cmap='gray')
    plt.title(f"Predicted Mask {i+1} (DeepLabV3+)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

print("\nAnalysis (DeepLabV3+ Model):")
print(f"The average IoU ({avg_iou:.4f}) and F1-Score ({avg_f1:.4f}) indicate the model's performance across the dataset.")
print("Visually inspect the predicted masks and compare them to the ground truth to understand where the model performs well and where it struggles.")
print("Further improvements might involve more data, different architectures, hyperparameter tuning, or more advanced data augmentation.")