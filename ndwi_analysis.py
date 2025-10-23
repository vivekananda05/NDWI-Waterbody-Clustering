import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os



###======================================================================================###
                          #    RBG (Orginal Image) and NDWI Map    #
###======================================================================================###


def show_ndwi_maps(rgb, ndwi, water_pixels, masked_nonwater, threshold, out_dir):
    if rgb is not None:
        plt.figure(figsize=(8,6))# Create a figure of size 8x6 inches
        plt.imshow(rgb) # Display the RGB composite image
        plt.title("Original RGB Image") # Add a title to the plot
        #plt.axis('off')
        plt.savefig(f"{out_dir}/rgb_image.png", dpi=300)# Save the RGB image to output directory at 300 dpi


    plt.figure(figsize=(8,6))  # Create a new figure
    im = plt.imshow(ndwi, cmap='BrBG', vmin=-1, vmax=1) # Display NDWI with color map 'BrBG' scaled between -1 and 1
    plt.title("NDWI Map")# Add title for NDWI visualization
    plt.colorbar(im, fraction=0.046, pad=0.02)# Add a colorbar beside the NDWI map for reference
    #plt.axis('off')
    plt.savefig(f"{out_dir}/ndwi_map.png", dpi=300)# Save the NDWI map image

    fig, axs = plt.subplots(1, 2, figsize=(10,6))# Show non-water pixels in grayscale

    axs[0].imshow(masked_nonwater, cmap='gray', vmin=-1, vmax=1)
    axs[0].set_title(f"Masked Non-Water (NDWI ≤ {threshold})")
    axs[0].axis('off')
    axs[1].imshow(water_pixels, cmap='Blues', vmin=-1, vmax=1)# Show water pixels using blue shades

    axs[1].set_title(f"Non-Masked Water (NDWI > {threshold})")
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/ndwi_water_nonwater.png", dpi=300)# Save combined figure showing both maps



###======================================================================================###
                          #    OTSU Threshold, Histogram and NDWI Stats #
###======================================================================================###
def otsu_threshold(data):
    """Compute Otsu threshold for NDWI."""
    data = data[np.isfinite(data)] #Remove NaN and infinite values from the data
    hist, bin_edges = np.histogram(data, bins=256)#Compute histogram of NDWI values with 256 bins
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 #Calculate the midpoint (center) of each bin
    # Compute cumulative sums (weights) for each side of a possible threshold
    weight1 = np.cumsum(hist)# cumulative weight for class 1 (below threshold)


    weight2 = np.cumsum(hist[::-1])[::-1]# cumulative weight for class 2 (above threshold)
    mean1 = np.cumsum(hist * bin_centers) / (weight1 + 1e-10)# mean intensity below threshold
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / (weight2[::-1] + 1e-10))[::-1]# mean intensity above threshold

    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance)
    return bin_centers[idx]


def analyze_ndwi(ndwi, non_mask, threshold, pixel_area, out_dir):
    """Perform detailed NDWI analysis including histogram, water/non-water separation, and auto-threshold."""
    print("\n NDWI Analysis Started...")
    valid_ndwi = ndwi[np.isfinite(ndwi)]#Remove NaN and infinite values from ndwi data

    # --- Basic Stats calculating Mean,Median,Mode ---
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
    auto_thresh = otsu_threshold(valid_ndwi)
    print(f" Otsu Auto Threshold Suggestion: {auto_thresh:.4f}")

    # --- Histogram (combined water vs non-water) ---
    water_vals = ndwi[non_mask]
    nonwater_vals = ndwi[~non_mask & np.isfinite(ndwi)]

    plt.figure(figsize=(10,6))
    plt.hist(nonwater_vals, bins=100, color='orange', alpha=0.6, label='Non-Water')
    plt.hist(water_vals, bins=100, color='blue', alpha=0.6, label='Water')
    plt.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'User Threshold = {threshold:.3f}')
    plt.axvline(auto_thresh, color='g', linestyle='--', linewidth=2, label=f'Otsu Threshold = {auto_thresh:.3f}')
    plt.title("NDWI Distribution: Water vs Non-Water with Thresholds")
    plt.xlabel("NDWI Value")
    plt.ylabel("Pixel Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(out_dir, "ndwi_histogram.png"), dpi=300)#save the histogram of water pixel distribution and non water pixel distribution in the output directory


# --- Water Area Estimation ---
    water_pixels = np.sum(non_mask)#gives the total number of water pixels
    non_water_pixels = np.sum(~non_mask & np.isfinite(ndwi))#gives the total number of non water pixels
    total_water_area = water_pixels * pixel_area#gives the total area occupied by water in the image
    total_nonwater_area = non_water_pixels * pixel_area#gives the total area occupied by land in the image
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
        "Otsu_Threshold": auto_thresh,
        "Water_Pixels": water_pixels,
        "Non_Water_Pixels": non_water_pixels,
        "Water_Area_km2": total_water_area,
        "Non_Water_Area_km2": total_nonwater_area
    }

    df = pd.DataFrame([summary])#it gives the total summery of the above calculation in dataframe format
    df.to_csv(os.path.join(out_dir, "ndwi_analysis_summary.csv"), index=False)#This is a Pandas function that writes your DataFrame to a CSV file and creates the full path to where the CSV file will be saved.
    print(f"NDWI Analysis Summary Saved: {os.path.join(out_dir, 'ndwi_analysis_summary.csv')}")