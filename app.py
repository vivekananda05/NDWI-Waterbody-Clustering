import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import os
from utils import read_band, stretch
from ndwi_module import compute_ndwi, masking
from kmeans_module import KMeans
from cluster_analysis import build_cluster_map
from ndwi_analysis import otsu_threshold

# ---------------- APP HEADER ----------------
st.set_page_config(page_title="NDWI + KMeans Analyzer", layout="wide")
st.title(" NDWI & K-Means Water Body Clustering")

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("Satellite Source")

#  Satellite selection button
satellite_type = st.sidebar.radio(
    "Select Satellite Type:",
    options=["Landsat", "Sentinel-2"],
    index=0,
    help="Choose Landsat (30m) or Sentinel-2 (10m) data"
)

# Automatically set pixel size based on selection
if satellite_type == "Landsat":
    pixel_size = 30
else:
    pixel_size = 10



# ---------------- BAND UPLOADS ----------------
st.sidebar.header("Upload Bands")

if satellite_type == "Landsat":
    green_file = st.sidebar.file_uploader("Upload Green Band (B3.TIF)", type=["tif", "tiff"])
    nir_file   = st.sidebar.file_uploader("Upload NIR Band (B5.TIF)", type=["tif", "tiff"])
    red_file   = st.sidebar.file_uploader("Upload Red Band (B4.TIF, optional)", type=["tif", "tiff"])
    blue_file  = st.sidebar.file_uploader("Upload Blue Band (B2.TIF, optional)", type=["tif", "tiff"])
else:
    green_file = st.sidebar.file_uploader("Upload Green Band (B3.jp2)", type=["jp2"])
    nir_file   = st.sidebar.file_uploader("Upload NIR Band (B8.jp2)", type=["jp2"])
    red_file   = st.sidebar.file_uploader("Upload Red Band (B4.jp2, optional)", type=["jp2"])
    blue_file  = st.sidebar.file_uploader("Upload Blue Band (B2.jp2, optional)", type=["jp2"])

# ---------------- PARAMETERS ----------------
st.sidebar.header("Parameters")
threshold = st.sidebar.slider("NDWI Threshold", -1.0, 1.0, 0.0, 0.05)
k_value   = st.sidebar.slider("K-Means Clusters (K)", 2, 10, 3, 1)
downsample = st.sidebar.number_input("Downsample Factor", 1, 10, 2)

run_button = st.sidebar.button(" Run Analysis")


# ---------------- MAIN ANALYSIS ----------------
if run_button:
    if not (green_file and nir_file):
        st.error("Please upload both Green and NIR bands!")
        st.stop()
# Create temporary filenames based on uploaded extensions
    def save_temp(uploaded_file, name):
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        temp_name = f"{name}_tmp{ext}"
        with open(temp_name, "wb") as f:
            f.write(uploaded_file.read())
        return temp_name

    green_path = save_temp(green_file, "green")
    nir_path   = save_temp(nir_file, "nir")

    green, profile = read_band(green_path, downsample)
    nir, _ = read_band(nir_path, downsample)
    
    rgb = None
    if red_file and blue_file:
        red_path  = save_temp(red_file, "red")
        blue_path = save_temp(blue_file, "blue")    
        red, _  = read_band(red_path, downsample)
        blue, _ = read_band(blue_path, downsample)
        rbg_mask = (green==0) | (blue==0)| (red==0)
        rgb = np.dstack((stretch(red), stretch(green), stretch(blue)))
        rgb[rbg_mask] = np.nan

    # ---- Compute NDWI ----
    ndwi = compute_ndwi(green, nir)
    non_mask, water_pixels, masked_nonwater = masking(ndwi, threshold)
    pixel_area = (pixel_size * downsample)**2 / 1e6  # km² per pixel

    # ---------------- VISUALIZATION ----------------
    st.subheader(" Visualization")

    if rgb is not None:
        st.image(rgb, caption="Original RGB Image")

    # NDWI Map
    fig, ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(ndwi, cmap="BrBG", vmin=-1, vmax=1)
    ax.set_title("NDWI Map", fontsize=13)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.imshow(masked_nonwater, cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"Masked Non-Water (≤ {threshold})")
        ax.axis("off")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5,4))
        ax.imshow(water_pixels, cmap="Blues", vmin=-1, vmax=1)
        ax.set_title(f"Non-Masked Water (AOI)(>{threshold})")
        ax.axis("off")
        st.pyplot(fig)

    # ---------------- NDWI ANALYSIS ----------------
    st.subheader(" NDWI Analysis")

    valid_ndwi = ndwi[np.isfinite(ndwi)]
    mean_ndwi  = np.nanmean(valid_ndwi)
    std_ndwi   = np.nanstd(valid_ndwi)
    auto_thresh = otsu_threshold(valid_ndwi)
    water_area = np.sum(non_mask) * pixel_area
    non_water_pixels = np.sum(~non_mask & np.isfinite(ndwi))
    non_water_area = non_water_pixels * pixel_area

   
    # Histogram: water vs non-water
    water_vals = ndwi[non_mask]
    nonwater_vals = ndwi[~non_mask & np.isfinite(ndwi)]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(nonwater_vals, bins=100, color='orange', alpha=0.6, label='Non-Water')
    ax.hist(water_vals, bins=100, color='blue', alpha=0.6, label='Water')
    ax.axvline(threshold, color='r', linestyle='--', label=f'User Threshold = {threshold:.3f}')
    ax.axvline(auto_thresh, color='g', linestyle='--', label=f'Otsu Threshold = {auto_thresh:.3f}')
    ax.set_title("NDWI Distribution: Water vs Non-Water")
    ax.set_xlabel("NDWI Value")
    ax.set_ylabel("Pixel Count")
    ax.legend()
    st.pyplot(fig)

    st.subheader(" NDWI Statistical Summary")

    summary_dict = {
    "Mean NDWI": [np.nanmean(valid_ndwi)],
    "Median NDWI": [np.nanmedian(valid_ndwi)],
    "Std Dev": [np.nanstd(valid_ndwi)],
    "Min NDWI": [np.nanmin(valid_ndwi)],
    "Max NDWI": [np.nanmax(valid_ndwi)],
    "User Threshold": [threshold],
    "Otsu Threshold": [auto_thresh],
    "Water Pixels": [np.sum(non_mask)],
    "Water Area (km²)": [water_area],
    "Non Water Pixels":[non_water_pixels],
    "Non Water Area (km²)":[non_water_area]
  }

    df_summary = pd.DataFrame(summary_dict)
    st.dataframe(
    df_summary.style.format({
        "Mean NDWI": "{:.4f}", "Median NDWI": "{:.4f}", "Std Dev": "{:.4f}",
        "Min NDWI": "{:.4f}", "Max NDWI": "{:.4f}",
        "User Threshold": "{:.3f}", "Otsu Threshold": "{:.3f}",
        "Water Area (km²)": "{:.2f}"
    })
    .set_table_styles([
        # Header style (bold + blue background)
        {"selector": "th", "props": [
            ("font-weight", "bold"),
            ("text-align", "center"),
            ("color", "#E90C0C"),
            ("font-size", "14px"),
            ("border", "1px solid #dee2e6")
        ]},
        # Data cells style
        {"selector": "td", "props": [
            ("text-align", "center"),
            ("border", "1px solid #dee2e6"),
            ("font-size", "13px")
        ]},
        # Overall table border
        {"selector": "table", "props": [
            ("border-collapse", "collapse"),
            ("margin", "0 auto")
        ]}
    ]),
    #width="stretch",
    use_container_width=True,
    hide_index=True,      
   )

    # ---------------- K-MEANS CLUSTERING ----------------
    st.subheader(" K-Means Clustering")

    water_vals = ndwi[non_mask]
    model = KMeans(k_value)
    model.fit(water_vals)
    labels = model.labels
    centroids = model.centroids
    cluster_map = build_cluster_map(ndwi, non_mask, labels)

    fig, ax = plt.subplots(figsize=(7,5))
    #cmap = plt.cm.get_cmap('tab10', k_value)
    cmap = plt.get_cmap('tab10', k_value)
    im = plt.imshow(cluster_map, cmap=cmap)
    ax.set_title(f"K-Means Clusters (K={k_value})", fontsize=13)
    patches = [mpatches.Patch(color=cmap(i), label=f'Cluster {i}') for i in range(k_value)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    st.pyplot(fig)

    # Cluster statistics table
    cluster_ids = np.unique(cluster_map[np.isfinite(cluster_map)]).astype(int)
    stats = []
    for cid in cluster_ids:
        vals = ndwi[cluster_map == cid]
        stats.append({
            "Cluster": cid,
            "Mean NDWI": np.nanmean(vals),
            "Std NDWI": np.nanstd(vals),
            "Pixels": np.sum(cluster_map == cid),
            "Area (km²)": np.sum(cluster_map == cid) * pixel_area
        })
    df_stats = pd.DataFrame(stats)
    df_stats["Area (%)"] = df_stats["Area (km²)"]/df_stats["Area (km²)"].sum()*100
    st.dataframe(df_stats.style.format({"Mean NDWI": "{:.3f}", "Std NDWI": "{:.3f}", "Area (km²)": "{:.2f}", "Area (%)": "{:.1f}"}),
                # width="stretch",
                 use_container_width=True,
                 hide_index=True,  
                 )

    # Pie chart of area distribution
    colors = [cmap(i) for i in range(k_value)]
    fig, ax = plt.subplots(figsize=(5,5))
    ax.pie(df_stats["Area (km²)"], labels=[f"C{c} ({a:.1f}%)" for c,a in zip(df_stats["Cluster"], df_stats["Area (%)"])],
           colors=[colors[int(c) % len(colors)] for c in df_stats["Cluster"]],
           autopct='%1.1f%%', startangle=90)
    ax.set_title("Cluster Area Distribution (%)")
    st.pyplot(fig)

    st.success(" NDWI + K-Means Analysis Completed!")
