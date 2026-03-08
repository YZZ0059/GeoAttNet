
import numpy as np
import rasterio
import os

def load_rasters(file_list, target_shape=None):
    data_list = []
    for fp in file_list:
        with rasterio.open(fp) as src:
            arr = src.read(1)
  
            nodata = src.nodata
            if nodata is not None:
                arr = np.where(arr == nodata, np.nan, arr)
            arr = np.where((arr < -1e10) | (arr > 1e10), np.nan, arr)
            if target_shape is not None and arr.shape != target_shape:
                from skimage.transform import resize
                arr = resize(arr, target_shape, order=1, preserve_range=True, anti_aliasing=False)
            data_list.append(arr)
    return np.stack(data_list, axis=0)  # (n, H, W)

def standardize_data(file_list, mode='fit', stats_path='train_stats.npy', target_shape=None):

    data = load_rasters(file_list, target_shape=target_shape)

    for i in range(data.shape[0]):
        d = data[i]
   
    if mode == 'fit':
        means = np.nanmean(data, axis=(1,2))
        stds = np.nanstd(data, axis=(1,2))
        np.save(stats_path, {'mean': means, 'std': stds})

        for i in range(data.shape[0]):
            if stds[i] < 1e-6:

                data[i] = 0
            else:
                data[i] = (data[i] - means[i]) / (stds[i] + 1e-8)
        data = np.nan_to_num(data, nan=0, posinf=3, neginf=-3)
        data = np.clip(data, -3, 3)
        return data
    elif mode == 'transform':
        stats = np.load(stats_path, allow_pickle=True).item()
        means = stats.get('mean', stats.get('means'))
        stds = stats.get('std', stats.get('stds'))
        if means is None or stds is None:
            raise KeyError("stats 需包含 'mean'/'means' 与 'std'/'stds'")
        for i in range(data.shape[0]):
            if stds[i] < 1e-6:
 
                data[i] = 0
            else:
                data[i] = (data[i] - means[i]) / (stds[i] + 1e-8)
        data = np.nan_to_num(data, nan=0, posinf=3, neginf=-3)
        data = np.clip(data, -3, 3)
        return data
    else:
        raise ValueError("mode must be 'fit' or 'transform'")

if __name__ == '__main__':
  
    train_files = [
 
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


    pred_files = [
        r'data_WM/interpolated_results/data_WM/interpolated_chemK_ppm.tif', 
        r'data_WM/interpolated_results/data_WM/interpolated_chemTh_ppm.tif', 
        r'data_WM/interpolated_results/data_WM/interpolated_chemU_ppm.tif',
        r'data_WM/interpolated_results/data_WM/interpolated_Gravity_res.tif', 
        r'data_WM/interpolated_results/data_WM/interpolated_Gravity(CSCBA).tif', 
        r'data_WM/interpolated_results/data_WM/interpolated_Gravity(CSCBA)1VD.tif',
        r'data_WM/interpolated_results/data_WM/interpolated_K_ppm.tif', 
        r'data_WM/interpolated_results/data_WM/interpolated_K_res.tif', 
        r'data_WM/interpolated_results/data_WM/interpolated_Th_ppm.tif',
        r'data_WM/interpolated_results/data_WM/interpolated_Th_res.tif',
        r'data_WM/interpolated_results/data_WM/interpolated_U_ppm.tif', 
        r'data_WM/interpolated_results/data_WM/interpolated_U_res.tif',
        r'data_WM/interpolated_results/data_WM/interpolated_Magnetic_res.tif',
        r'data_WM/interpolated_results/data_WM/interpolated_Magnetic1VD.tif', 
        r'data_WM/interpolated_results/data_WM/interpolated_Magnetic40m.tif'
    ]


   
    train_array = standardize_data(train_files, mode='fit', stats_path='train_stats_WM.npy', target_shape=(2592, 2016))

    pred_array = standardize_data(pred_files, mode='transform', stats_path='train_stats_WM.npy', target_shape=(2592, 2016)) 