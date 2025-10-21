# Geospatial Data Science Repository

This repository contains multiple geospatial and data science projects. Below is a guide to each folder and its main Python scripts.

## Project Structure & Usage

### `caste-project/`
- **ScrapeCaste.py**: Scrapes caste-related place names from Google Maps using Selenium.
- **Usage**: Install Selenium and undetected_chromedriver, then run the script to generate CSV outputs.

### `govandi-research/`
- **building_footprint.py**: Deep learning workflow for building footprint segmentation (DeepLabV3+).
- **generate_prague_dataset.py**: Automates download of Prague satellite images and building shapefiles.
- **testing_rasterization.py**: Visualizes and checks rasterization quality.
- **govandi_buildings_folium.py**: Downloads and visualizes Govandi buildings using Folium and ArcGIS.
- **Usage**: Install dependencies (`geopandas`, `rasterio`, `segmentation_models_pytorch`, etc.), run scripts for dataset creation, training, and visualization.

### `census-schools-combined/`
- **combine_schools_and_india_map.py**: Spatial join of school and district data.
- **combine_census_school.py**: Merges census population and school data.
- **interactive_map_pop_school_ratio.py**: Creates interactive Folium map of population/school ratio.
- **plot_schools_on_india_map.py**: Plots schools and districts using matplotlib.
- **combine.py**: (empty or utility script)
- **Usage**: Requires `geopandas`, `folium`, `matplotlib`, and processed data files.

### `india-schools/`
- **school.py**: Loads and analyzes school GIS data.
- **villages.py**: Loads and prints school data.
- **Usage**: Requires `geopandas`, `pandas`, and access to GeoJSON files.

### `india-schools-analysis/`
- **transfer.py**: Uploads local files to S3 (uses environment variables for AWS credentials).
- **Usage**: Set AWS credentials, run to upload files.

### `osm/`
- **fetch_districts.py**: Fetches and visualizes OSM district boundaries for Goa.
- **Usage**: Requires `osmnx`, `geopandas`.

### `slums-vectorization/`
- **main.py**: Deep learning pipeline for segmenting and vectorizing slum imagery.
- **Usage**: Install `segmentation_models_pytorch`, `rasterio`, `geopandas`, and run for segmentation and vectorization.

### `python/`
- **GIS_pandas_test.py**: Geopandas and geodatasets demo for NYC boroughs.
- **Phase2Testing.py**, **Phase3Testing.py**: (content varies, see scripts for details)
- **school-data-load.py**: Downloads school GIS data from NIC portal.
- **Usage**: See individual scripts for details.

### Other folders
- **interactive-maps/**: Contains HTML outputs for interactive map visualizations.
- **India-map-shapefiles/**, **india-census/**, **photos-image/**, **data-for-model/**: Contain raw and processed data, images, shapefiles, and outputs. These are ignored in `.gitignore` for privacy and size reasons.

---

## Setup & Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```
2. Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   # Or manually:
   pip install geopandas rasterio segmentation_models_pytorch albumentations torch matplotlib folium osmnx pandas
   ```
3. Add your own data in the ignored folders if needed.

## Contributing
Pull requests and issues are welcome!

## License
MIT

## Credits
- Bing Maps, Overpass Turbo, segmentation_models_pytorch, geopandas, rasterio, and other open-source tools.
