# === USER CONFIGURATION ===
import os

# User selection: "landsat" or "sentinel"
SATELLITE = "landsat"  # <-- Change this to "landsat" if needed

# Common parameters
SENT_DIR = "Sentinel_data"
LANDSAT_DIR = "Landsat_data"
NDWI_THRESHOLD = 0.0
N_CLUSTERS = 3  # user-defined
k_max = 10  # for elbow method
DOWNSAMPLE = 4  # Downsample factor (1 = no downsampling)

# Set parameters based on satellite type
if SATELLITE.lower() == "landsat":
    PIXEL_SIZE = 30  # Landsat-8/9 resolution (m)
    OUTPUT_DIR = "Landsat_output"
    n_blue = 2
    n_green = 3
    n_red = 4
    n_nir = 5
    DATA_DIR = LANDSAT_DIR

elif SATELLITE.lower() == "sentinel":
    PIXEL_SIZE = 10  # Sentinel-2 resolution (m)
    OUTPUT_DIR = "Sentinel_output"
    n_blue = '02'
    n_green = '03'
    n_red = '04'
    n_nir = '08'
    DATA_DIR = SENT_DIR

else:
    raise ValueError("Invalid satellite type! Use 'landsat' or 'sentinel'.")

# Create output directory if it doesn’t exist
os.makedirs(OUTPUT_DIR, exist_ok=True)