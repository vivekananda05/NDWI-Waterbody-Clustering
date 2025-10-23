import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import matplotlib.patches as mpatches

# =================================================================================================#
               ### Fit the KMeans model to the input 1D data using vectorized operations. ###
# =================================================================================================#

def build_cluster_map(ndwi, water_pixels, labels):
    cluster_map = np.full(ndwi.shape, np.nan, dtype='float32')# Create an empty map with same shape as NDWI, filled with NaN
    cluster_map[water_pixels] = labels # Assign cluster labels to pixels identified as water and water_pixels should be a Boolean mask (True for water pixels)
    return cluster_map# Return the cluster map where:
    # - water pixels have their cluster label
    # - non-water pixels remain NaN
def show_cluster_map(cluster_map, K, out_dir):
    plt.figure(figsize=(8,6))
    cmap = plt.cm.get_cmap('tab10', K)
    im = plt.imshow(cluster_map, cmap='tab10')
    plt.title(f"K-Means Clusters (K={K})")
    plt.colorbar(im, fraction=0.046, pad=0.02)
    patches = [mpatches.Patch(color=cmap(i), label=f'Cluster {i}') for i in range(K)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/kmeans_cluster_map.png", dpi=300)
# =================================================================================================#
               ### Analyze K-Means clusters: NDWI stats, area, density plots (consistent colors). ###
# =================================================================================================#
def analyze_clusters(ndwi, cluster_map, pixel_area, out_dir, K):

    # --- Define consistent color map (same as in show_cluster_map) ---
    cmap = plt.cm.get_cmap('tab10', K)  # or 'tab20' for larger K
    colors = [cmap(i) for i in range(K)]
    
    # --- Identify valid clusters ---
    valid_clusters = np.isfinite(cluster_map)
    cluster_ids = np.unique(cluster_map[valid_clusters]).astype(int)
    stats = []

    # --- Compute statistics ---
    for cid in cluster_ids:
        vals = ndwi[cluster_map == cid]
        stats.append((cid,
                      np.nanmean(vals),
                      np.nanstd(vals),
                      np.sum(cluster_map == cid),
                      np.sum(cluster_map == cid) * pixel_area))

    df = pd.DataFrame(stats, columns=["Cluster", "Mean_NDWI", "Std_NDWI", "Pixels", "Area_km2"])
    df["Area_%"] = df["Area_km2"] / df["Area_km2"].sum() * 100
    df.to_csv(os.path.join(out_dir, "cluster_summary.csv"), index=False)

    print("\n Cluster Summary:")
    print(df)

    # --- NDWI Density Plot by Cluster ---
    plt.figure(figsize=(9, 6))
    for cid in cluster_ids:
        sns.kdeplot(ndwi[cluster_map == cid],
                    color=colors[cid % len(colors)],
                    fill=True,
                    label=f"Cluster {cid}")
    plt.title("NDWI Distribution by Cluster")
    plt.xlabel("NDWI Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_ndwi_distribution.png"), dpi=300)


    # --- Cluster Area Pie Chart ---
    plt.figure(figsize=(6, 6))
    plt.pie(df["Area_km2"],
            labels=[f"Cluster {c} ({a:.1f}%)" for c, a in zip(df["Cluster"], df["Area_%"])],
            colors=[colors[int(c) % len(colors)] for c in df["Cluster"]],
            autopct='%1.1f%%',
            startangle=90)
    plt.title("Cluster Area Distribution (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cluster_area_distribution.png"), dpi=300)


    print(f"\n Cluster Analysis Saved: {os.path.join(out_dir, 'cluster_summary.csv')}")