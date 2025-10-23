import numpy as np

# Function to compute NDWI (Normalized Difference Water Index)
def compute_ndwi(green, nir, nodata=0):
    eps = 1e-10  # Small constant to avoid division by zero
    mask = (green==nodata) | (nir==nodata)  # Create a mask for pixels that have 'nodata' values in either band
    ndwi = (green - nir) / (green + nir + eps)#NDWI calculation
    ndwi[mask]  = np.nan  # Assign NaN (Not a Number) to NDWI values where nodata mask is True
    return np.clip(ndwi, -1, 1)# Clip NDWI values to the valid range of -1 to 1

def masking(ndwi, threshold):
    non_mask = ndwi > threshold # Pixels greater than the threshold are considered water
    water_pixels = np.where(non_mask, ndwi, np.nan) # Keep NDWI values only for water pixels, others become NaN
    masked_nonwater = np.where(~non_mask, ndwi, np.nan)
    return non_mask, water_pixels, masked_nonwater