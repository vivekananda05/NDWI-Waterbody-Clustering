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
#from ndwi_analysis import otsu_threshold


# ============================================================
# Modified Elbow Method (Streamlit Version - NDWI 1D Only)
# ============================================================

def elbow_method_streamlit(data, k_max=10, max_iters=100, tol=1e-4, random_state=42):
    """
    Elbow method for 1D NDWI data (water pixels).
    """
    data = np.array(data).ravel()
    wcss_values = []
    k_values = list(range(1, k_max + 1))

    for k in k_values:
        model = KMeans(K=k, max_iters=max_iters, tol=tol, random_state=random_state)
        model.fit(data)
        wcss_values.append(model.wcss)

    elbow_k = find_elbow_point(k_values, wcss_values)
    fig = elbow_plot_streamlit(elbow_k, k_values, wcss_values)

    return elbow_k, wcss_values, fig


def find_elbow_point(k_values, wcss):
    x = np.array(k_values)
    y = np.array(wcss)

    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    distances = np.abs(np.cross(line_vec, np.vstack([x - x[0], y - y[0]]).T)) / line_len

    return x[np.argmax(distances)]


def elbow_plot_streamlit(elbow_k, k_values, wcss):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, wcss, 'bo-', markersize=6)
    ax.scatter(
        elbow_k,
        wcss[list(k_values).index(elbow_k)],
        s=200, facecolors='none', edgecolors='r',
        label=f'Optimal K = {elbow_k}'
    )
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Method")
    ax.legend()
    ax.grid(True)
    return fig


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
    green_file = st.sidebar.file_uploader("Upload Green Band (B3.TIF) ✔ Required", type=["tif", "tiff"])
    nir_file   = st.sidebar.file_uploader("Upload NIR Band (B5.TIF) ✔ Required", type=["tif", "tiff"])
    red_file   = st.sidebar.file_uploader("Upload Red Band (B4.TIF) ✔ Required", type=["tif", "tiff"])
    blue_file  = st.sidebar.file_uploader("Upload Blue Band (B2.TIF) ✔ Required", type=["tif", "tiff"])
else:
    green_file = st.sidebar.file_uploader("Upload Green Band (B3.jp2) ✔ Required", type=["jp2"])
    nir_file   = st.sidebar.file_uploader("Upload NIR Band (B8.jp2) ✔ Required", type=["jp2"])
    red_file   = st.sidebar.file_uploader("Upload Red Band (B4.jp2) ✔ Required", type=["jp2"])
    blue_file  = st.sidebar.file_uploader("Upload Blue Band (B2.jp2) ✔ Required", type=["jp2"])

# ---------------- PARAMETERS ----------------
st.sidebar.header("Parameters")
threshold   = st.sidebar.slider("NDWI Threshold", -1.0, 1.0, 0.0, 0.05)
k_value     = st.sidebar.slider("K-Means Clusters (K)", 2, 10, 3, 1)
downsample  = st.sidebar.number_input("Downsample Factor", 1, 10, 1)
k_range_max = st.sidebar.slider("Max K for Elbow Test", 3, 15, 5)

run_button = st.sidebar.button(" Run Analysis")


# ---------------- MAIN ANALYSIS ----------------
if run_button:
    # Check required bands
    missing = []
    if not green_file: missing.append("Green (B3)")
    if not nir_file:   missing.append("NIR (B5)")
    if not red_file:   missing.append("Red (B4)")
    if not blue_file:  missing.append("Blue (B2)")

    if missing:
        st.error(
            " Missing Required Bands:\n- " +
            "\n- ".join(missing) +
            "\n\nPlease upload all four bands to continue."
        )
        st.stop()

    # Helper to save uploaded file
    def save_temp(uploaded_file, name):
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        temp_name = f"{name}_tmp{ext}"
        with open(temp_name, "wb") as f:
            f.write(uploaded_file.read())
        return temp_name

    # Read mandatory bands
    green_path = save_temp(green_file, "green")
    nir_path   = save_temp(nir_file, "nir")
    red_path   = save_temp(red_file, "red")
    blue_path  = save_temp(blue_file, "blue")

    green, profile = read_band(green_path, downsample)
    nir, _        = read_band(nir_path, downsample)
    red, _        = read_band(red_path, downsample)
    blue, _       = read_band(blue_path, downsample)

    # Shape check: all bands must match
    if not (red.shape == green.shape == blue.shape == nir.shape):
        st.error(
            f"Band size mismatch detected:\n"
            f"Red: {red.shape}, Green: {green.shape}, Blue: {blue.shape}, NIR: {nir.shape}\n\n"
            "All four bands must have the same rows/columns (same scene + resolution)."
        )
        st.stop()

    # ---- FCC (False Color Composite: NIR-Red-Green) ----

    fcc_mask = (nir == 0) | (red == 0) | (green == 0)
    fcc = np.dstack((stretch(nir), stretch(red), stretch(green)))
    fcc[fcc_mask] = np.nan

    # ---- Compute NDWI ----
    ndwi = compute_ndwi(green, nir)
    non_mask, water_pixels, masked_nonwater = masking(ndwi, threshold)
    pixel_area = (pixel_size * downsample) ** 2 / 1e6  # km² per pixel

    # ---------------- VISUALIZATION ----------------
    st.subheader(" Visualization")

    # FCC image (caption above)
    st.markdown("### False Color Composite (FCC) Image (NIR-Red-Green)")
    st.image(fcc)

    # NDWI Map
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(ndwi, cmap="BrBG", vmin=-1, vmax=1)
    ax.set_title("NDWI Map", fontsize=13)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(masked_nonwater, cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"Masked Non-Water (≤ {threshold})")
        ax.axis("off")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(water_pixels, cmap="Blues", vmin=-1, vmax=1)
        ax.set_title(f"Non-Masked Water (AOI) (>{threshold})")
        ax.axis("off")
        st.pyplot(fig)

    # ---------------- NDWI ANALYSIS ----------------
    st.subheader(" NDWI Analysis")

    valid_ndwi  = ndwi[np.isfinite(ndwi)]
    mean_ndwi   = np.nanmean(valid_ndwi)
    std_ndwi    = np.nanstd(valid_ndwi)
    #auto_thresh = otsu_threshold(valid_ndwi)
    water_area  = np.sum(non_mask) * pixel_area
    non_water_pixels = np.sum(~non_mask & np.isfinite(ndwi))
    non_water_area   = non_water_pixels * pixel_area

    # Histogram: water vs non-water
    water_vals    = ndwi[non_mask]
    nonwater_vals = ndwi[~non_mask & np.isfinite(ndwi)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(nonwater_vals, bins=100, color='orange', alpha=0.6, label='Non-Water')
    ax.hist(water_vals, bins=100, color='blue', alpha=0.6, label='Water')
    ax.axvline(threshold,   color='r', linestyle='--', label=f'User Threshold = {threshold:.3f}')
    #ax.axvline(auto_thresh, color='g', linestyle='--', label=f'Otsu Threshold = {auto_thresh:.3f}')
    ax.set_title("NDWI Distribution: Water vs Non-Water")
    ax.set_xlabel("NDWI Value")
    ax.set_ylabel("Pixel Count")
    ax.legend()
    st.pyplot(fig)

    st.subheader(" NDWI Statistical Summary")

    summary_dict = {
        "Mean NDWI":           [np.nanmean(valid_ndwi)],
        "Median NDWI":         [np.nanmedian(valid_ndwi)],
        "Std Dev":             [np.nanstd(valid_ndwi)],
        "Min NDWI":            [np.nanmin(valid_ndwi)],
        "Max NDWI":            [np.nanmax(valid_ndwi)],
        "User Threshold":      [threshold],
        #"Otsu Threshold":      [auto_thresh],
        "Water Pixels":        [np.sum(non_mask)],
        "Water Area (km²)":    [water_area],
        "Non Water Pixels":    [non_water_pixels],
        "Non Water Area (km²)":[non_water_area],
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
            {"selector": "th", "props": [
                ("font-weight", "bold"),
                ("text-align", "center"),
                ("color", "#E90C0C"),
                ("font-size", "14px"),
                ("border", "1px solid #dee2e6")
            ]},
            {"selector": "td", "props": [
                ("text-align", "center"),
                ("border", "1px solid #dee2e6"),
                ("font-size", "13px")
            ]},
            {"selector": "table", "props": [
                ("border-collapse", "collapse"),
                ("margin", "0 auto")
            ]}
        ]),
        use_container_width=True,
        hide_index=True,
    )

    # ---------------- K-MEANS CLUSTERING (4 BANDS) ----------------
    st.subheader(" K-Means Clustering (4-Band: R, G, B, NIR)")

    # 4-band feature stack
    band_stack = np.dstack((red, green, blue, nir)).astype("float32")

    # Mask: NDWI > threshold and all bands finite
    water_mask = non_mask & np.all(np.isfinite(band_stack), axis=2)

    data = band_stack[water_mask]  # shape: (N_water, 4)
    if data.shape[0] == 0:
        st.error("No valid water pixels found for 4-band clustering (check NDWI threshold and band data).")
        st.stop()

    model = KMeans(k_value)
    model.fit(data)
    labels = model.labels
    centroids = model.centroids

    if water_mask.sum() != len(labels):
        st.error(
            f"Mask/labels mismatch in 4-band clustering: "
            f"mask True = {water_mask.sum()}, labels = {len(labels)}"
        )
        st.stop()

    cluster_map = build_cluster_map(ndwi, water_mask, labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap('tab10', k_value)
    im = ax.imshow(cluster_map, cmap=cmap)
    ax.set_title(f"K-Means Clusters (K={k_value})\nMode: 4-band (R,G,B,NIR) on NDWI > threshold", fontsize=13)
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
    df_stats["Area (%)"] = df_stats["Area (km²)"] / df_stats["Area (km²)"].sum() * 100
    st.dataframe(
        df_stats.style.format({
            "Mean NDWI": "{:.3f}",
            "Std NDWI": "{:.3f}",
            "Area (km²)": "{:.2f}",
            "Area (%)": "{:.1f}"
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Pie chart of area distribution
    colors = [cmap(i) for i in range(k_value)]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        df_stats["Area (km²)"],
        labels=[f"C{c} ({a:.1f}%)" for c, a in zip(df_stats["Cluster"], df_stats["Area (%)"])],
        colors=[colors[int(c) % len(colors)] for c in df_stats["Cluster"]],
        autopct='%1.1f%%',
        startangle=90
    )
    ax.set_title("Cluster Area Distribution (%)")
    st.pyplot(fig)

    # ---------------- ELBOW METHOD (NDWI-BASED) ----------------
    st.subheader("Elbow Method (Best K Suggestion from NDWI)")

    water_vals = ndwi[non_mask]
    water_vals = water_vals[np.isfinite(water_vals)]

    if len(water_vals) < 5:
        st.warning("Not enough valid pixels to run elbow method.")
    else:
        optimal_k, wcss_values, elbow_fig = elbow_method_streamlit(water_vals, k_max=int(k_range_max))
        st.pyplot(elbow_fig)
        st.success(f" Recommended K based on Elbow Method: **{optimal_k}**")

    st.success(" NDWI + 4-Band K-Means Analysis Completed!")