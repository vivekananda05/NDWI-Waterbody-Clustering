import os, re
import numpy as np
import rasterio
from rasterio.enums import Resampling
 
#=================================================================================================#
                            ### Finding outh the Bands with .TIF or .jp2 files ###
#=================================================================================================# 
def find_band(folder, band):
    patt = re.compile(rf'B{band}\.(?:TIF|jp2)$', re.I) #match filenames ending with .TIF or .jp2
    for root, _, files in os.walk(folder):  #Recursively walks through all subfolders and files under the given folder
        for f in files:
            if patt.search(f):
                return os.path.join(root, f) #Returns the full path to the first matching file
    return None

#=================================================================================================#
                            ### Read out bands ###
#=================================================================================================# 

def read_band(path, downsample=1):  
    with rasterio.open(path) as src:
        if downsample > 1:   #Reduce the resolution or dimensions
            h, w = src.height // downsample, src.width // downsample
            band = src.read(1, out_shape=(1, h, w),
                            resampling=Resampling.average).astype('float32') #read with pixel values float32 type
        else:
            band = src.read(1).astype('float32')
        profile = src.profile  #give the metadata
    return band, profile 

#=================================================================================================#
                            ### percentile-based stretching to preview RGB Image ###
#=================================================================================================# 

def stretch(arr, lo=2, hi=98): 
    vmin, vmax = np.nanpercentile(arr, [lo, hi]) #rank=p​×(n−1)/100
    return np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0, 1) 

