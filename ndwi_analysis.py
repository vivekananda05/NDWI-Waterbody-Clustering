import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os



###======================================================================================###
                          #    FCC Image and NDWI Map    #
###======================================================================================###


def show_ndwi_maps(fcc, ndwi, water_pixels, masked_nonwater, threshold, out_dir):
    if fcc is not None:
        plt.figure(figsize=(8,6))
        plt.imshow(fcc)
        plt.title("FCC Image")
        #plt.axis('off')
        plt.savefig(f"{out_dir}/fcc_image.png", dpi=300)

    plt.figure(figsize=(8,6))
    im = plt.imshow(ndwi, cmap='BrBG', vmin=-1, vmax=1)
    plt.title("NDWI Map")
    plt.colorbar(im, fraction=0.046, pad=0.02)
    #plt.axis('off')
    plt.savefig(f"{out_dir}/ndwi_map.png", dpi=300)

    fig, axs = plt.subplots(1, 2, figsize=(10,6))
    axs[0].imshow(masked_nonwater, cmap='gray', vmin=-1, vmax=1)
    axs[0].set_title(f"Masked Non-Water (NDWI ≤ {threshold})")
    axs[0].axis('off')
    axs[1].imshow(water_pixels, cmap='Blues', vmin=-1, vmax=1)
    axs[1].set_title(f"Non-Masked Water (NDWI > {threshold})")
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/ndwi_water_nonwater.png", dpi=300)



###======================================================================================###
                          #    OTSU Threshold, Histogram and NDWI Stats #
###======================================================================================###
# def otsu_threshold(data):
#     """Compute Otsu threshold for NDWI."""
#     data = data[np.isfinite(data)]
#     hist, bin_edges = np.histogram(data, bins=256)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#     weight1 = np.cumsum(hist)
#     weight2 = np.cumsum(hist[::-1])[::-1]
#     mean1 = np.cumsum(hist * bin_centers) / (weight1 + 1e-10)
#     mean2 = (np.cumsum((hist * bin_centers)[::-1]) / (weight2[::-1] + 1e-10))[::-1]

#     variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
#     idx = np.argmax(variance)
#     return bin_centers[idx]


def analyze_ndwi(ndwi, non_mask, threshold, pixel_area, out_dir):
    """Perform detailed NDWI analysis including histogram, water/non-water separation, and auto-threshold."""
    print("\n NDWI Analysis Started...")
    valid_ndwi = ndwi[np.isfinite(ndwi)]

    # --- Basic Stats ---
    mean_ndwi = np.nanmean(valid_ndwi)
    median_ndwi = np.nanmedian(valid_ndwi)
    std_ndwi = np.nanstd(valid_ndwi)
    print(f"""
    NDWI Statistics:
      ▫ Mean NDWI   : {mean_ndwi:.4f}
      ▫ Median NDWI : {median_ndwi:.4f}
      ▫ Std NDWI    : {std_ndwi:.4f}
      ▫ Min NDWI    : {np.nanmin(valid_ndwi):.4f}
      ▫ Max NDWI    : {np.nanmax(valid_ndwi):.4f}
    """)

    # --- Otsu Auto Threshold ---
    # auto_thresh = otsu_threshold(valid_ndwi)
    # print(f" Otsu Auto Threshold Suggestion: {auto_thresh:.4f}")

    # --- Histogram (combined water vs non-water) ---
    water_vals = ndwi[non_mask]
    nonwater_vals = ndwi[~non_mask & np.isfinite(ndwi)]

    plt.figure(figsize=(10,6))
    plt.hist(nonwater_vals, bins=100, color='orange', alpha=0.6, label='Non-Water')
    plt.hist(water_vals, bins=100, color='blue', alpha=0.6, label='Water')
    plt.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'User Threshold = {threshold:.3f}')
    #plt.axvline(auto_thresh, color='g', linestyle='--', linewidth=2, label=f'Otsu Threshold = {auto_thresh:.3f}')
    plt.title("NDWI Distribution: Water vs Non-Water with Thresholds")
    plt.xlabel("NDWI Value")
    plt.ylabel("Pixel Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(out_dir, "ndwi_histogram.png"), dpi=300)


# --- Water Area Estimation ---
    water_pixels = np.sum(non_mask)
    non_water_pixels = np.sum(~non_mask & np.isfinite(ndwi))
    total_water_area = water_pixels * pixel_area
    total_nonwater_area = non_water_pixels * pixel_area
    # print(f" Estimated Water Area: {total_water_area:.2f} km²")
    # print(f" \n Estimated Non-Water Area: {total_nonwater_area:.2f} km²")

    # --- Save NDWI Analysis Summary ---
    summary = {
        "Mean_NDWI": mean_ndwi,
        "Median_NDWI": median_ndwi,
        "Std_NDWI": std_ndwi,
        "Min_NDWI": np.nanmin(valid_ndwi),
        "Max_NDWI": np.nanmax(valid_ndwi),
        "User_Threshold": threshold,
        #"Otsu_Threshold": auto_thresh,
        "Water_Pixels": water_pixels,
        "Non_Water_Pixels": non_water_pixels,
        "Water_Area_km2": total_water_area,
        "Non_Water_Area_km2": total_nonwater_area
    }

    df = pd.DataFrame([summary])
    df.to_csv(os.path.join(out_dir, "ndwi_analysis_summary.csv"), index=False)
    print(f"NDWI Analysis Summary Saved: {os.path.join(out_dir, 'ndwi_analysis_summary.csv')}")