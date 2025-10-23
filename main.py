import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio

from utils import find_band, read_band, stretch
from ndwi_module import compute_ndwi, masking
from ndwi_analysis import show_ndwi_maps, analyze_ndwi
from cluster_analysis import build_cluster_map, show_cluster_map, analyze_clusters
from kmeans_module import KMeans, elbow_method
from config import DATA_DIR, OUTPUT_DIR, NDWI_THRESHOLD, N_CLUSTERS, DOWNSAMPLE, PIXEL_SIZE, n_blue, n_green, n_red, n_nir, k_max

# === STEP 1: Locate Bands ===
blue = find_band(DATA_DIR, n_blue)  #blue
green = find_band(DATA_DIR, n_green)  #green
red = find_band(DATA_DIR, n_red)  #red
nir = find_band(DATA_DIR, n_nir)  #nir

print("Bands found:\n", blue, "\n", green, "\n", red, "\n", nir)

# === STEP 2: Read Bands ===
green, profile = read_band(green, DOWNSAMPLE)
nir, _ = read_band(nir, DOWNSAMPLE)

rgb = None
if blue and red:
    blue, _ = read_band(blue, DOWNSAMPLE)
    red, _  = read_band(red, DOWNSAMPLE)
    rbg_mask = (green==0) | (blue==0)| (red==0)
    rgb = np.dstack((stretch(red), stretch(green), stretch(blue)))
    rgb[rbg_mask] = np.nan

    
# === STEP 3: Compute NDWI ===
ndwi = compute_ndwi(green, nir)

# === STEP 4: Masking ===
non_mask, water_pixels, masked_nonwater = masking(ndwi, NDWI_THRESHOLD)
pixel_area = (PIXEL_SIZE * DOWNSAMPLE)**2 / 1e6

# === STEP 5: Visualization ===
show_ndwi_maps(rgb, ndwi, water_pixels, masked_nonwater, NDWI_THRESHOLD, OUTPUT_DIR)

# === STEP 6: NDWI Analysis ===
analyze_ndwi(ndwi, non_mask, NDWI_THRESHOLD, pixel_area, OUTPUT_DIR)

# === STEP 7: K-Means on Water Pixels ===
water_vals = ndwi[non_mask]
data = water_vals

model = KMeans(K=N_CLUSTERS)
model.fit(data)
labels = model.labels
centroids = model.centroids
print(f"\nResults for k={N_CLUSTERS}:")
print("Labels:", labels)
print("Centroids in std:", centroids)
print("WCSS:", model.wcss)
print(data.shape)

cluster_map = build_cluster_map(ndwi, non_mask, labels)
show_cluster_map(cluster_map, N_CLUSTERS, OUTPUT_DIR)

# === STEP 8: Cluster Analysis ===
analyze_clusters(ndwi, cluster_map, pixel_area, OUTPUT_DIR, K=N_CLUSTERS)


# === STEP 9: Elbow Analysis ===
elbow_k, wcss_values = elbow_method(water_vals, k_max=k_max, out_dir=OUTPUT_DIR)