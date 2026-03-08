
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from missing_value.geochemical_interpolation import GeochemicalInterpolator

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from GeoAttNet.GeoAttNet_model import DeepUNet

from standardize_for_prediction import standardize_data

CONFIG = {
    "model_path": os.path.join("GeoAttNet-Base", "result", "best_model.pth"),
    "data_files": [
   
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
    ],
    "mineral_points_file": r'data_frome/uranium_occurrences.gpkg',
    "stats_path": "train_stats_frome.npy",
    "output_dir": os.path.join("GeoAttNet-Base", "prediction_results_frome"),
    "target_shape": (2592, 2016),  
    "patch_size": 32,  
    "stride": 8  
}

class MineralPredictor:

    
    def __init__(self, model_path, device='auto', patch_size=32, stride=16):

        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else device)

        self.model = self._load_model(model_path)
        self.model.eval()
 
        self.patch_size = patch_size
        self.stride = stride  
        
    def _load_model(self, model_path):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = DeepUNet(in_channels=15, num_classes=1, dropout_rate=0.2, use_attention=False)
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)

        return model
    
    def preprocess_raster(self, file_path, target_crs='EPSG:4326', target_shape=None):

        with rasterio.open(file_path) as src:
   
            original_crs = src.crs
            original_bounds = src.bounds

            if target_shape is not None:
                height, width = target_shape
                transform = from_bounds(*original_bounds, width, height)
            else:
                transform = src.transform
                height, width = src.height, src.width
 
            data = np.zeros((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )

            data[~np.isfinite(data)] = np.nan
            valid = ~np.isnan(data)
            
            if np.any(valid):
   
                mean = np.nanmean(data)
                std = np.nanstd(data)
                if std > 0:
                    data[valid] = (data[valid] - mean) / (std + 1e-8)
                else:
                    data[valid] = 0
                data[~valid] = 0
            else:
                data[:] = 0
            
            return data, transform, target_crs
    
    def stack_rasters(self, file_paths, target_shape=(2592, 2016)):

        with rasterio.open(file_paths[0]) as src:
            bounds = src.bounds
            crs = src.crs

        data_list = []
        for i, file_path in enumerate(tqdm(file_paths, desc="deal file")):
            data, transform, _ = self.preprocess_raster(file_path, target_shape=target_shape)
            data_list.append(data)
            print(f"file {Path(file_path).name}: shape={data.shape}")

        stacked_data = np.stack(data_list, axis=0)
        
        return stacked_data, transform, crs
    
    def predict_patch(self, patch_data):

        if patch_data.shape[1:] != (self.patch_size, self.patch_size):
            raise ValueError(f"Patch is {self.patch_size}x{self.patch_size}")
        
        if np.isnan(patch_data).any():
            patch_data = np.nan_to_num(patch_data, nan=0.0)
        
  
        x = torch.from_numpy(patch_data).float().unsqueeze(0).to(self.device)
        
 
        with torch.no_grad():
            output = self.model(x)
            prediction = torch.sigmoid(output).cpu().numpy()[0, 0]  
        
        return prediction
    
    def predict_region(self, data_array, transform, crs, output_path=None):

        n_channels, height, width = data_array.shape
 
        prediction_map = np.zeros((height, width), dtype=np.float32)
        confidence_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)
        
   
        n_patches_h = (height - self.patch_size) // self.stride + 1
        n_patches_w = (width - self.patch_size) // self.stride + 1
        

        
        patch_print_count = 0  
    
        for i in tqdm(range(n_patches_h), desc="prediction"):
            for j in range(n_patches_w):
     
                start_h = i * self.stride
                start_w = j * self.stride
                end_h = start_h + self.patch_size
                end_w = start_w + self.patch_size
      
                patch_data = data_array[:, start_h:end_h, start_w:end_w]
                if patch_print_count < 5:
                    print(f'patch({i},{j}) NaN:', np.isnan(patch_data).sum() / patch_data.size)
                    patch_print_count += 1
                
           
                valid_ratio = np.sum(~np.isnan(patch_data[0])) / patch_data[0].size
                if valid_ratio < 0.5:  
                    continue
                
        
                try:
                    prediction = self.predict_patch(patch_data)
                    
            
                    prediction_map[start_h:end_h, start_w:end_w] += prediction
                    confidence_map[start_h:end_h, start_w:end_w] += 1
                    
                except Exception as e:
                    print(f"prediction patch ({i}, {j}) fail: {str(e)}")
                    continue
        
        valid_mask = confidence_map > 0
        prediction_map[valid_mask] /= confidence_map[valid_mask]
        
        if output_path:
            self.save_prediction(prediction_map, transform, crs, output_path)
        
        return prediction_map, confidence_map
    
    def save_prediction(self, prediction_map, transform, crs, output_path):

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=prediction_map.shape[0],
            width=prediction_map.shape[1],
            count=1,
            dtype=prediction_map.dtype,
            crs=crs,
            transform=transform,
            nodata=-9999
        ) as dst:
            dst.write(prediction_map, 1)

    
    def visualize_prediction(self, prediction_map, confidence_map, output_dir, title="prediction result"):

        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        im1 = axes[0].imshow(prediction_map, cmap='jet', vmin=0, vmax=1)
        axes[0].set_title('minerl prediction')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], label='prediction content')
        

        im2 = axes[1].imshow(confidence_map, cmap='viridis')
        axes[1].set_title('Prediction confidence')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], label='confidence')
        
   
        valid_predictions = prediction_map[confidence_map > 0]
        if len(valid_predictions) > 0:
            axes[2].hist(valid_predictions.flatten(), bins=50, alpha=0.7, color='red')
            axes[2].set_title('Predicted probability distribution')
            axes[2].set_xlabel('Predicted probability')
            axes[2].set_ylabel('Frequency')
            axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'prediction_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
 
        stats_path = os.path.join(output_dir, 'prediction_statistics.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("Mineral content prediction statistical information\n")
            f.write("=" * 50 + "\n")
            f.write(f"Prediction area size: {prediction_map.shape}\n")
            f.write(f"Effective prediction pixels: {np.sum(confidence_map > 0)}\n")
            f.write(f"Total number of pixels: {prediction_map.size}\n")
            f.write(f"Coverage: {np.sum(confidence_map > 0) / prediction_map.size * 100:.2f}%\n")
            
            if len(valid_predictions) > 0:
                f.write(f"Prediction probability statistics:\n")
                f.write(f"  Minimum: {np.min(valid_predictions):.4f}\n")
                f.write(f"  Maximum: {np.max(valid_predictions):.4f}\n")
                f.write(f"  average value: {np.mean(valid_predictions):.4f}\n")
                f.write(f"  Standard deviation: {np.std(valid_predictions):.4f}\n")
                f.write(f"  Median: {np.median(valid_predictions):.4f}\n")
                
                thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                f.write(f"\nThe number of pixels in different probability intervals:\n")
                for i, threshold in enumerate(thresholds):
                    if i == 0:
                        count = np.sum((valid_predictions >= 0) & (valid_predictions < threshold))
                    else:
                        count = np.sum((valid_predictions >= thresholds[i-1]) & (valid_predictions < threshold))
                    f.write(f"  {thresholds[i-1] if i > 0 else 0:.1f}-{threshold:.1f}: {count} Pixel\n")
                
                high_prob_count = np.sum(valid_predictions >= 0.7)
                f.write(f"\nHigh probability area  (≥0.7): {high_prob_count} Pixel ({high_prob_count/len(valid_predictions)*100:.2f}%)\n")


def save_separate_visualizations(prediction_map, confidence_map, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(prediction_map, cmap='jet', vmin=0, vmax=1)
    plt.title('Mineral content prediction')
    plt.axis('off')
    plt.colorbar(label='Prediction probability')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_map.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(confidence_map, cmap='viridis')
    plt.title('Prediction confidence')
    plt.axis('off')
    plt.colorbar(label='confidence')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_map.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 8))
    valid_predictions = prediction_map[confidence_map > 0]
    if len(valid_predictions) > 0:
        plt.hist(valid_predictions.flatten(), bins=50, alpha=0.7, color='red')
        plt.title('Predicted probability distribution')
        plt.xlabel('Prediction probability')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_hist.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_with_points(prediction_map, confidence_map, transform, label_fn, output_dir, crs=None):
 
    import geopandas as gpd
    import rasterio
    os.makedirs(output_dir, exist_ok=True)

    left, bottom, right, top = rasterio.transform.array_bounds(*prediction_map.shape, transform)

    gdf = gpd.read_file(label_fn)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(
        prediction_map, cmap='jet', vmin=0, vmax=1,
        extent=[left, right, bottom, top], origin='upper'
    )
    ax.axis('off')
 
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.015)

    if crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    if not gdf.empty:
 
        if gdf.geometry.iloc[0].geom_type == 'Point':
            xs = gdf.geometry.x
            ys = gdf.geometry.y
        else:

            xs = gdf.geometry.centroid.x
            ys = gdf.geometry.centroid.y

        plt.rcParams['font.family'] = 'Times New Roman'

        plt.scatter(xs, ys, c='#FFFF00', s=50, marker='o', edgecolors='black', label='Sandstone-hosted uranium deposit')

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fontsize=18, handletextpad=0.1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_with_points.png'), dpi=300, bbox_inches='tight')
    plt.close()

def interpolate_files_in_memory(file_list, mineral_points_file=None):

    import tempfile
    import shutil
    from pathlib import Path
 
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    for f in file_list:
        dst = os.path.join(temp_dir, os.path.basename(f))
        shutil.copy2(f, dst)
        temp_files.append(dst)

    if mineral_points_file and os.path.exists(mineral_points_file):
        shutil.copy2(mineral_points_file, os.path.join(temp_dir, os.path.basename(mineral_points_file)))
        mpf = os.path.join(temp_dir, os.path.basename(mineral_points_file))
    else:
        mpf = None

    interpolator = GeochemicalInterpolator(temp_dir, mpf)
    results = interpolator.interpolate_dataset(method='auto', preserve_mineral_areas=True)

    interpolated_arrays = []
    for f in file_list:
        name = os.path.basename(f)
        arr = results[name]['interpolated_data']
        interpolated_arrays.append(arr)

    shutil.rmtree(temp_dir, ignore_errors=True)
    return interpolated_arrays

def calc_buffer_high_prob_stats(prediction_map, confidence_map, transform, label_fn, output_dir, crs, prob_thresh=0.7, buffer_radius=2500):

    import geopandas as gpd
    from rasterio import features
    import os
    if label_fn is None or not os.path.exists(label_fn):
    
        return
    gdf = gpd.read_file(label_fn)
 
    if crs and gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    gdf['geometry'] = gdf.geometry.buffer(buffer_radius)
 
    buffer_union = gdf.unary_union
 
    mask = features.geometry_mask([buffer_union], out_shape=prediction_map.shape, transform=transform, invert=True)
 
    high_prob_mask = (prediction_map >= prob_thresh) & (confidence_map > 0)
    buffer_high_prob = high_prob_mask & mask
    buffer_high_prob_count = np.sum(buffer_high_prob)
    buffer_pixel_area = abs(transform.a * transform.e) 
    buffer_high_prob_area = buffer_high_prob_count * buffer_pixel_area

    total_high_prob_count = np.sum(high_prob_mask)
    total_high_prob_area = total_high_prob_count * buffer_pixel_area

    ratio = buffer_high_prob_area / total_high_prob_area if total_high_prob_area > 0 else 0

    stats_path = os.path.join(output_dir, 'prediction_statistics.txt')
    with open(stats_path, 'a', encoding='utf-8') as f:
        f.write(f"\nBuffer Statistics\n")
        f.write(f"The proportion of high-probability area in the buffer zone to the high-probability area in the entire region: {ratio*100:.2f}%\n")


def calc_mineral_points_in_high_prob(prediction_map, confidence_map, transform, label_fn, output_dir, crs, prob_thresh=0.7):

    import geopandas as gpd
    import numpy as np
    import os
    if label_fn is None or not os.path.exists(label_fn):

        return
    gdf = gpd.read_file(label_fn)
   
    if crs and gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    total_points = len(gdf)
    high_prob_points = 0
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        x, y = geom.x, geom.y
        col, row = ~transform * (x, y)
        col, row = int(round(col)), int(round(row))
        if 0 <= row < prediction_map.shape[0] and 0 <= col < prediction_map.shape[1]:
            if confidence_map[row, col] > 0 and prediction_map[row, col] >= prob_thresh:
                high_prob_points += 1
    ratio = high_prob_points / total_points if total_points > 0 else 0
    stats_path = os.path.join(output_dir, 'prediction_statistics.txt')
    with open(stats_path, 'a', encoding='utf-8') as f:
        f.write(f"Total mineral points: {total_points}\n")
        f.write(f"Proportion: {ratio*100:.2f}%\n")

def main():

    model_path = CONFIG["model_path"]
    data_files = CONFIG["data_files"]
    label_fn = CONFIG["mineral_points_file"] if os.path.exists(CONFIG["mineral_points_file"]) else None
    stats_path = CONFIG["stats_path"]
    output_dir = CONFIG["output_dir"]

    if not os.path.exists(model_path):

        alternative_paths = [
            os.path.join("DeepUnetPlus", '619train_result', 'best_model.pth'),
            os.path.join("DeepUnetPlus", 'result', 'best_model.pth'),
            os.path.join("DeepUnetPlus", '619train_result', 'best_model.pth')
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
 
                break
        else:
  
            for path in [model_path] + alternative_paths:
                print(f"  {path}")
            return

    missing_files = [f for f in data_files if not os.path.exists(f)]
    if missing_files:

        for f in missing_files:
            print(f"  {f}")
  
        return

    try:
        predictor = MineralPredictor(
            model_path, 
            patch_size=CONFIG["patch_size"], 
            stride=CONFIG["stride"]
        )
    except Exception as e:

        return

    try:
 
        from data_selection import get_union_bounds
        import rasterio
        bounds = get_union_bounds(data_files, label_fn)
    
        height, width = CONFIG["target_shape"]
        target_shape = (height, width)
        transform = rasterio.transform.from_bounds(*bounds, width, height)
  
        with rasterio.open(data_files[0]) as src:
            crs = src.crs
     
    
        data_array = standardize_data(data_files, mode='transform', stats_path=stats_path, target_shape=target_shape)
   
    except Exception as e:
  
        return
    print('data_array shape:', data_array.shape)
    print('data_array min:', np.nanmin(data_array), 'max:', np.nanmax(data_array))
    print('data_array NaN:', np.isnan(data_array).sum() / data_array.size)

    try:
        prediction_map, confidence_map = predictor.predict_region(
            data_array, transform, crs,
            output_path=os.path.join(output_dir, 'mineral_prediction.tif')
        )
    except Exception as e:
        print(f"prediction fail: {str(e)}")
        return

    try:
        if np.isnan(prediction_map).all():

            return
        predictor.visualize_prediction(
            prediction_map, confidence_map, output_dir,
            title="Uranium ore content prediction results"
        )

        save_separate_visualizations(prediction_map, confidence_map, output_dir)
    
        if label_fn:
            plot_prediction_with_points(prediction_map, confidence_map, transform, label_fn, output_dir, crs)
        calc_buffer_high_prob_stats(
            prediction_map, confidence_map, transform, label_fn, output_dir, crs,
            prob_thresh=0.7, buffer_radius=2500
        )
        calc_mineral_points_in_high_prob(
            prediction_map, confidence_map, transform, label_fn, output_dir, crs,
            prob_thresh=0.7
        )
    except Exception as e:
        print(f"pic fail: {str(e)}")
        return


if __name__ == '__main__':
    main() 