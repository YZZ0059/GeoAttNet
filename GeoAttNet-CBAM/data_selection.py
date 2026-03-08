
import numpy as np
import rasterio

import matplotlib.pyplot as plt

from pathlib import Path

import matplotlib as mpl
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

import os
import geopandas as gpd
from shapely.geometry import box
from rasterio.transform import from_bounds
import random

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    

def get_raster_info(file_path):

    with rasterio.open(file_path) as src:
        return {
            'resolution': src.res,
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height
        }

def get_common_bounds(file_paths):

    bounds = None
    for file_path in file_paths:
        with rasterio.open(file_path) as src:
            if bounds is None:
                bounds = src.bounds
            else:
                bounds = (
                    max(bounds[0], src.bounds[0]),  # left
                    max(bounds[1], src.bounds[1]),  # bottom
                    min(bounds[2], src.bounds[2]),  # right
                    min(bounds[3], src.bounds[3])   # top
                )
    return bounds

def resample_raster(src_path, target_res, target_bounds=None, target_shape=None):

    with rasterio.open(src_path) as src:
      
        if target_bounds is None:
            target_bounds = src.bounds
        
     
        if target_shape is None:
            width = int((target_bounds[2] - target_bounds[0]) / target_res[0])
            height = int((target_bounds[3] - target_bounds[1]) / target_res[1])
            target_shape = (height, width)
        
     
        transform = rasterio.transform.from_bounds(
            *target_bounds, target_shape[1], target_shape[0])
        
     
        data = src.read(
            1,
            out_shape=target_shape,
            resampling=Resampling.bilinear,
            window=rasterio.windows.from_bounds(
                *target_bounds, transform=src.transform)
        )
        
        return data, transform

def load_raster_data(file_paths, target_resolution=None):
   
    resolutions = []
    for file_path in file_paths:
        info = get_raster_info(file_path)
        resolutions.append(info['resolution'])
        print(f"file {Path(file_path).name} resolution: {info['resolution']} ")
    

    if target_resolution is None:
        target_res = (max(res[0] for res in resolutions), 
                     max(res[1] for res in resolutions))
    
    else:
        target_res = target_resolution

    

    common_bounds = get_common_bounds(file_paths)

    
 
    width = int((common_bounds[2] - common_bounds[0]) / target_res[0])
    height = int((common_bounds[3] - common_bounds[1]) / target_res[1])
    target_shape = (height, width)
 
    

    data_list = []
    feature_names = []
    
    for file_path in file_paths:
  
        info = get_raster_info(file_path)
        
  
        data, _ = resample_raster(
            file_path, 
            target_res,
            target_bounds=common_bounds,
            target_shape=target_shape
        )
        

        feature_name = Path(file_path).stem
        
 
        if isinstance(data, np.ma.MaskedArray):
            mask = data.mask
        else:
            with rasterio.open(file_path) as src:
                mask = data == src.nodata if src.nodata is not None else np.zeros_like(data, dtype=bool)
        
        data = np.ma.masked_array(data, mask)
        

        valid_data = data[~data.mask]
        if len(valid_data) > 0:
            mean = np.nanmean(valid_data)
            std = np.nanstd(valid_data)
            data = (data - mean) / (std + 1e-8)
        
        data_list.append(data)
        feature_names.append(feature_name)
    

    data_array = np.ma.stack(data_list)
    
    return data_array, feature_names


def get_union_bounds(file_paths, label_fn):
    bounds = None
    for file_path in file_paths:
        with rasterio.open(file_path) as src:
            b = src.bounds
            if bounds is None:
                bounds = list(b)
            else:
                bounds[0] = min(bounds[0], b.left)
                bounds[1] = min(bounds[1], b.bottom)
                bounds[2] = max(bounds[2], b.right)
                bounds[3] = max(bounds[3], b.top)
    gdf = gpd.read_file(label_fn)
    minx, miny, maxx, maxy = gdf.total_bounds
    bounds[0] = min(bounds[0], minx)
    bounds[1] = min(bounds[1], miny)
    bounds[2] = max(bounds[2], maxx)
    bounds[3] = max(bounds[3], maxy)
    return tuple(bounds)

def reproject_raster_to_target(src_path, target_crs, target_transform, target_shape):
    import warnings
    with rasterio.open(src_path) as src:
        dst_array = np.zeros(target_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )

        nodata = src.nodata
        if nodata is not None:
            dst_array[dst_array == nodata] = np.nan
     
        dst_array[(dst_array < -1e10) | (dst_array > 1e10)] = np.nan
 
        dst_array[~np.isfinite(dst_array)] = np.nan
        valid = ~np.isnan(dst_array)
        if np.any(valid):
    
            dst_array[~valid] = 0
        else:
            dst_array[:] = 0
        return dst_array

def stack_all_rasters(file_paths, target_crs, target_transform, target_shape):
    bands = []
    for path in file_paths:
        arr = reproject_raster_to_target(path, target_crs, target_transform, target_shape)

        bands.append(arr)
    return np.stack(bands)


BUFFER_M = 2500
POSITIVE_RATIO_THRESHOLD = 0.05

def rasterize_labels(label_fn, target_crs, target_transform, target_shape, buffer_m=BUFFER_M):
    gdf = gpd.read_file(label_fn)
 
    gdf_proj = gdf.to_crs(epsg=3857)
    gdf_proj['geometry'] = gdf_proj.geometry.buffer(buffer_m)
    gdf_buffer = gdf_proj.to_crs(target_crs)
    from rasterio.features import rasterize
    shapes = ((geom, 1) for geom in gdf_buffer.geometry)
    labels = rasterize(
        shapes=shapes,
        out_shape=target_shape,
        transform=target_transform,
        fill=0,
        dtype='float32',
        all_touched=True
    )
    return labels

def extract_patches(data_array, labels, patch_size=32, stride=32):
  
    n_features, height, width = data_array.shape
  
    actual_stride = patch_size  

    n_patches_h = (height - patch_size) // actual_stride + 1
    n_patches_w = (width - patch_size) // actual_stride + 1
    
    all_patches = []
    pos_patches = []
    neg_patches = []
    
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            start_h = i * actual_stride
            start_w = j * actual_stride
            end_h = start_h + patch_size
            end_w = start_w + patch_size
            

            patch_data = data_array[:, start_h:end_h, start_w:end_w]
            patch_label = labels[start_h:end_h, start_w:end_w]
            
       
            valid_ratio = np.sum(~np.isnan(patch_data[0])) / patch_data[0].size
            if valid_ratio < 0.5:  
                continue
            
    
            positive_ratio = np.sum(patch_label > 0) / patch_label.size
            has_positive = positive_ratio > POSITIVE_RATIO_THRESHOLD 
            
            patch_info = {
                'position': (start_h, start_w),
                'data': patch_data,
                'label': patch_label,
                'has_positive': has_positive,
                'positive_ratio': positive_ratio
            }
            
            all_patches.append(patch_info)
            
            if has_positive:
                pos_patches.append(patch_info)
            else:
                neg_patches.append(patch_info)
    
    return {
        'all_patches': all_patches,
        'pos_patches': pos_patches,
        'neg_patches': neg_patches,
        'patch_size': patch_size,
        'stride': actual_stride,
        'n_patches_h': n_patches_h,
        'n_patches_w': n_patches_w
    }

def prepare_blocks_for_training(data_files, label_fn, target_size=(2592, 2016)):

    target_crs = 'EPSG:4326'
    target_res = (0.001, 0.001)
    

    bounds = get_union_bounds(data_files, label_fn)

    height, width = target_size
    target_shape = (height, width)
    target_transform = from_bounds(*bounds, width, height)

    data_array = stack_all_rasters(data_files, target_crs, target_transform, target_shape)

    labels = rasterize_labels(label_fn, target_crs, target_transform, target_shape, buffer_m=BUFFER_M)

    
    print(f"\nUsing original data without PCA...")
    print(f"Original data shape: {data_array.shape}")
    pca_data = data_array
    
    patches_info = extract_patches(pca_data, labels, patch_size=32, stride=32)
    
    return patches_info

def main():

    data_files = [
  
        r'interpolated_results/1chemK_ppm.tif', 
        r'interpolated_results/1chemTh_ppm.tif', 
        r'interpolated_results/1chemU_ppm.tif',
 
        r'interpolated_results/1interpolated_Gravity_Resi.tif', 
        r'interpolated_results/2interpolated_Gravity(CSCBA)_1VD.tif',
        r'interpolated_results/2interpolated_Gravity(CSCBA).tif',
        
        r'interpolated_results/1interpolated_K_ppm_Resi.tif', 
        r'interpolated_results/1interpolated_K_ppm.tif',
        r'interpolated_results/1interpolated_Th_ppm_Resi.tif',
        r'interpolated_results/1interpolated_Th_ppm.tif',
        r'interpolated_results/1interpolated_U_ppm_Resi.tif', 
        r'interpolated_results/1interpolated_U_ppm.tif',
   
        r'interpolated_results/2interpolated_Magnetic.tif',
        r'interpolated_results/2interpolated_Megnetic_Resi.tif',
        r'interpolated_results/1interpolated_Magnetic_1VD.tif'
    ]
    label_fn = r'data_frome/uranium_occurrences.gpkg'
    
 
    patches_info = prepare_blocks_for_training(
        data_files=data_files,
        label_fn=label_fn,
        target_size=(2592, 2016)
    )
    

if __name__ == '__main__':
    main() 