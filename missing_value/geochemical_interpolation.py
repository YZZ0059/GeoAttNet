import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RBFInterpolator
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class GeochemicalInterpolator:
    def __init__(self, data_dir, mineral_points_file=None):

        self.data_dir = data_dir
        self.mineral_points_file = mineral_points_file
        self.tif_files = list(Path(data_dir).glob("*.tif"))
        
    def load_data(self):
     
        datasets = {}
 
        total_files = len(self.tif_files)
        
        for i, tif_file in enumerate(self.tif_files):
            try:
                
                with rasterio.open(tif_file) as src:
                    data = src.read(1)
                    
    
                    data_size_mb = data.nbytes / (1024 * 1024)
                    if data_size_mb > 100:  
                        print(f"  warning: file {tif_file.name}  big ({data_size_mb:.1f}MB)")
                    
                    datasets[tif_file.name] = {
                        'data': data,
                        'profile': src.profile,
                        'nodata': src.nodata,
                        'transform': src.transform,
                        'crs': src.crs
                    }
                    print(f"  success {data.shape}, shape {data_size_mb:.1f}MB")
                    
                   
                    import gc
                    gc.collect()
                    
            except Exception as e:
                print(f"no load {tif_file.name}: {e}")
                
        print(f"load {len(datasets)} file")
        return datasets
    
    def analyze_missing_patterns(self, datasets):
       
        missing_patterns = {}
        
        for name, dataset in datasets.items():
            data = dataset['data']
            nodata = dataset['nodata']

            if nodata is not None:
                missing_mask = np.isnan(data) | (data == nodata)
            else:
                missing_mask = np.isnan(data)

            missing_percentage = np.sum(missing_mask) / missing_mask.size * 100
            missing_coords = np.where(missing_mask)
            valid_coords = np.where(~missing_mask)

            spatial_analysis = self._analyze_spatial_distribution(missing_mask)
            
            missing_patterns[name] = {
                'missing_mask': missing_mask,
                'missing_percentage': missing_percentage,
                'missing_coords': missing_coords,
                'valid_coords': valid_coords,
                'spatial_analysis': spatial_analysis
            }  
                
        return missing_patterns
    
    def _analyze_spatial_distribution(self, missing_mask):
        from scipy import ndimage
        
        missing_coords = np.where(missing_mask)
        missing_count = len(missing_coords[0])
        
        if missing_count == 0:
            return {
                'pattern_type': 'no miss value',
                'max_cluster_size': 0,
                'clustering_score': 0.0,
                'is_clustered': False,
                'is_scattered': False,
                'is_edge': False
            }

        labeled_mask, num_features = ndimage.label(missing_mask)
        if num_features > 0:
            cluster_sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
            max_cluster_size = max(cluster_sizes)
            avg_cluster_size = np.mean(cluster_sizes)
        else:
            max_cluster_size = 0
            avg_cluster_size = 0

        edge_missing = self._check_edge_missing(missing_mask)

        clustering_score = self._calculate_simplified_clustering(missing_mask, missing_count, max_cluster_size)
 
        pattern_type = self._classify_missing_pattern(
            missing_mask, clustering_score, max_cluster_size, edge_missing
        )
        
        return {
            'pattern_type': pattern_type,
            'max_cluster_size': max_cluster_size,
            'clustering_score': clustering_score,
            'is_clustered': clustering_score > 0.5,
            'is_scattered': clustering_score < 0.2,
            'is_edge': edge_missing,
            'num_clusters': num_features,
            'avg_cluster_size': avg_cluster_size
        }
    
    def _calculate_simplified_clustering(self, missing_mask, missing_count, max_cluster_size):
      
        total_pixels = missing_mask.size
        missing_percentage = missing_count / total_pixels

        if missing_count == 0:
            return 0.0

        if max_cluster_size / missing_count > 0.5:
            return 0.8
        elif max_cluster_size / missing_count > 0.2:
            return 0.6
        elif max_cluster_size / missing_count > 0.1:
            return 0.4
        else:
            return 0.2
    
    def _check_edge_missing(self, missing_mask):

        height, width = missing_mask.shape

        top_edge = np.sum(missing_mask[0, :])
        bottom_edge = np.sum(missing_mask[-1, :])
        left_edge = np.sum(missing_mask[:, 0])
        right_edge = np.sum(missing_mask[:, -1])
        
        total_edge_missing = top_edge + bottom_edge + left_edge + right_edge
        total_missing = np.sum(missing_mask)
   
        return total_edge_missing / total_missing > 0.5 if total_missing > 0 else False
    
    def _classify_missing_pattern(self, missing_mask, clustering_score, max_cluster_size, edge_missing):

        total_pixels = missing_mask.size
        missing_pixels = np.sum(missing_mask)
        
        if missing_pixels == 0:
            return "No missing values"
        elif edge_missing:
            return "Missing edges"
        elif clustering_score > 0.7:
            return "Highly Aggregated"
        elif clustering_score > 0.4:
            return "Moderate aggregation"
        elif max_cluster_size > missing_pixels * 0.1: 
            return "Large missing"
        elif clustering_score < 0.2:
            return "Scattered Missing"
        else:
            return "Missing at random"
    
    def _select_interpolation_method(self, missing_percentage, spatial_analysis):
   
        pattern_type = spatial_analysis['pattern_type']
        is_clustered = spatial_analysis['is_clustered']
        is_edge = spatial_analysis['is_edge']
        max_cluster_size = spatial_analysis['max_cluster_size']
        
        if is_edge:
            if missing_percentage < 10:
                return 'nearest'
            else:
                return 'linear'
        
        elif is_clustered and missing_percentage > 5:
            if missing_percentage < 30:
                return 'rbf'
            else:
                return 'ml'

        elif max_cluster_size > 1000:  
            if missing_percentage < 20:
                return 'rbf'
            else:
                return 'ml'

        elif spatial_analysis['is_scattered']:
            if missing_percentage < 5:
                return 'nearest'
            else:
                return 'linear'

        else:
            if missing_percentage < 2:
                return 'nearest'
            elif missing_percentage < 10:
                return 'linear'
            elif missing_percentage < 30:
                return 'cubic'
            elif missing_percentage < 50:
                return 'rbf'
            else:
                return 'ml'
    
    def _explain_method_selection(self, method, missing_percentage, spatial_analysis):
       
        pattern_type = spatial_analysis['pattern_type']
        
        explanations = {
            'none': f"No interpolation required - {pattern_type}",
            'nearest': f"Nearest neighbor interpolation - Suitable{pattern_type}And the missing ratio is low({missing_percentage:.1f}%)",
            'linear': f"Linear interpolation - Suitable{pattern_type}And the missing ratio is moderate({missing_percentage:.1f}%)",
            'cubic': f"Cubic interpolation - Suitable{pattern_type}And a smooth transition is required",
            'rbf': f"Radial Basis Function Interpolation - Suitable{pattern_type}And missing values are clustered",
            'ml': f"Machine Learning Interpolation - Suitable{pattern_type}And the missing ratio is high({missing_percentage:.1f}%)"
        }
        
        return explanations.get(method, f"Manual selection method: {method}")
    
    def nearest_neighbor_interpolation(self, data, missing_mask, valid_coords, missing_coords):
        
        valid_values = data[valid_coords]
        tree = cKDTree(np.column_stack(valid_coords))
        distances, indices = tree.query(np.column_stack(missing_coords))
        
        interpolated_values = valid_values[indices]
        return interpolated_values
    
    def linear_interpolation(self, data, missing_mask, valid_coords, missing_coords):
        
        valid_values = data[valid_coords]
        interpolated_values = griddata(
            np.column_stack(valid_coords),
            valid_values,
            np.column_stack(missing_coords),
            method='linear',
            fill_value=np.nan
        )
        
        nan_mask = np.isnan(interpolated_values)
        if np.any(nan_mask):
            tree = cKDTree(np.column_stack(valid_coords))
            distances, indices = tree.query(np.column_stack(missing_coords)[nan_mask])
            interpolated_values[nan_mask] = valid_values[indices]
            
        return interpolated_values
    
    def cubic_interpolation(self, data, missing_mask, valid_coords, missing_coords):
        
        valid_values = data[valid_coords]
        interpolated_values = griddata(
            np.column_stack(valid_coords),
            valid_values,
            np.column_stack(missing_coords),
            method='cubic',
            fill_value=np.nan
        )

        nan_mask = np.isnan(interpolated_values)
        if np.any(nan_mask):
            linear_values = griddata(
                np.column_stack(valid_coords),
                valid_values,
                np.column_stack(missing_coords)[nan_mask],
                method='linear'
            )
            interpolated_values[nan_mask] = linear_values
            
        return interpolated_values
    
    def rbf_interpolation(self, data, missing_mask, valid_coords, missing_coords):
        
        valid_values = data[valid_coords]

        max_points = 10000
        if len(valid_values) > max_points:
            indices = np.random.choice(len(valid_values), max_points, replace=False)
            sample_coords = (valid_coords[0][indices], valid_coords[1][indices])
            sample_values = valid_values[indices]
        else:
            sample_coords = valid_coords
            sample_values = valid_values
        
        try:
            rbf = RBFInterpolator(
                np.column_stack(sample_coords),
                sample_values,
                kernel='thin_plate_spline'
            )
            interpolated_values = rbf(np.column_stack(missing_coords))
        except Exception as e:
            return self.linear_interpolation(data, missing_mask, valid_coords, missing_coords)
            
        return interpolated_values
    
    def machine_learning_interpolation(self, datasets, target_name, missing_patterns):

        feature_names = [name for name in datasets.keys() if name != target_name]
        if len(feature_names) == 0:
            return None

        X_list = []
        for name in feature_names:
            data = datasets[name]['data'].flatten()
            X_list.append(data)
        
        X = np.column_stack(X_list)
        y = datasets[target_name]['data'].flatten()

        missing_mask = missing_patterns[target_name]['missing_mask'].flatten()
        valid_mask = ~missing_mask
        
        if np.sum(valid_mask) < 100:
            return None

        X_train = X[valid_mask]
        y_train = y[valid_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation R² score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        model.fit(X_train_scaled, y_train)

        X_missing = X[missing_mask]
        X_missing_scaled = scaler.transform(X_missing)
        interpolated_values = model.predict(X_missing_scaled)
        
        return interpolated_values
    
    def preserve_mineral_areas(self, interpolated_data, mineral_points_file, dataset_profile):
     
        if mineral_points_file is None:
            return interpolated_data
            
        try:
            mineral_points = gpd.read_file(mineral_points_file)
           
            buffer_distance = 2500  
            mineral_buffer = mineral_points.buffer(buffer_distance)

            from rasterio.features import rasterize
            
            buffer_mask = rasterize(
                mineral_buffer.geometry,
                out_shape=interpolated_data.shape,
                transform=dataset_profile['transform'],
                fill=0,
                dtype=np.uint8
            )

            buffer_mask = buffer_mask.astype(bool)
            if np.any(buffer_mask):
               
                from scipy.ndimage import uniform_filter
                smoothed_data = uniform_filter(interpolated_data, size=3)
                interpolated_data[buffer_mask] = smoothed_data[buffer_mask]
            
        except Exception as e:
            print(f"fail: {e}")
            
        return interpolated_data
    
    def quality_control(self, original_data, interpolated_data, missing_mask):

        original_valid = original_data[~missing_mask]
        interpolated_values = interpolated_data[missing_mask]
      
        if len(interpolated_values) == 0:
            print(f"no miss value")
            return 0.0
        
        original_mean = np.mean(original_valid)
        original_std = np.std(original_valid)
        
        outliers = np.abs(interpolated_values - original_mean) > 3 * original_std
        outlier_percentage = np.sum(outliers) / len(interpolated_values) * 100
        
        if outlier_percentage > 5:
            print("  warning: outlier percentage is high")
            
        return outlier_percentage
    
    def interpolate_dataset(self, method='auto', preserve_mineral_areas=True):
       
        datasets = self.load_data()
        if not datasets:
            print("no file")
            return

        missing_patterns = self.analyze_missing_patterns(datasets)

        results = {}
        
        for name, dataset in datasets.items():
            
            data = dataset['data']
            missing_mask = missing_patterns[name]['missing_mask']
            valid_coords = missing_patterns[name]['valid_coords']
            missing_coords = missing_patterns[name]['missing_coords']
            
            missing_percentage = missing_patterns[name]['missing_percentage']
            spatial_analysis = missing_patterns[name]['spatial_analysis']
            
            if missing_percentage == 0 or spatial_analysis['pattern_type'] == 'no miss value':

                results[name] = {
                    'original_data': data,
                    'interpolated_data': data, 
                    'missing_mask': missing_mask,
                    'method': 'none',  
                    'outlier_percentage': 0.0,
                    'profile': dataset['profile']
                }
                continue

            if method == 'auto':
                selected_method = self._select_interpolation_method(
                    missing_percentage, 
                    spatial_analysis
                )
            else:
                selected_method = method
            
            if selected_method == 'nearest':
                interpolated_values = self.nearest_neighbor_interpolation(
                    data, missing_mask, valid_coords, missing_coords
                )
            elif selected_method == 'linear':
                interpolated_values = self.linear_interpolation(
                    data, missing_mask, valid_coords, missing_coords
                )
            elif selected_method == 'cubic':
                interpolated_values = self.cubic_interpolation(
                    data, missing_mask, valid_coords, missing_coords
                )
            elif selected_method == 'rbf':
                interpolated_values = self.rbf_interpolation(
                    data, missing_mask, valid_coords, missing_coords
                )
            elif selected_method == 'ml':
                interpolated_values = self.machine_learning_interpolation(
                    datasets, name, missing_patterns
                )
                if interpolated_values is None:
                    interpolated_values = self.linear_interpolation(
                        data, missing_mask, valid_coords, missing_coords
                    )

            interpolated_data = data.copy()
            interpolated_data[missing_coords] = interpolated_values

            if preserve_mineral_areas:
                interpolated_data = self.preserve_mineral_areas(
                    interpolated_data, self.mineral_points_file, dataset['profile']
                )

            outlier_percentage = self.quality_control(data, interpolated_data, missing_mask)

            results[name] = {
                'original_data': data,
                'interpolated_data': interpolated_data,
                'missing_mask': missing_mask,
                'method': selected_method,
                'outlier_percentage': outlier_percentage,
                'profile': dataset['profile']
            }
            
        
        return results
    
    def save_results(self, results, output_dir):
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, result in results.items():
            output_file = output_path / f"interpolated_{name}"
            
            with rasterio.open(output_file, 'w', **result['profile']) as dst:
                dst.write(result['interpolated_data'], 1)

        report_data = []
        for name, result in results.items():
            report_data.append({
                'file_name': name,
                'interpolation_method': result['method'],
                'outlier_percentage': result['outlier_percentage'],
                'missing_pixels_original': np.sum(result['missing_mask']),
                'total_pixels': result['missing_mask'].size
            })
            
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(output_path / 'interpolation_report.csv', index=False)
    
    def visualize_results(self, results, output_dir):
       
        output_path = Path(output_dir)
        
        for name, result in results.items():
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            im1 = axes[0].imshow(result['original_data'], cmap='viridis')
            axes[0].set_title(f'original_data - {name}')
            plt.colorbar(im1, ax=axes[0])

            im2 = axes[1].imshow(result['interpolated_data'], cmap='viridis')
            axes[1].set_title(f'interpolated_data - {name}')
            plt.colorbar(im2, ax=axes[1])

            diff = result['interpolated_data'] - result['original_data']
            diff[result['missing_mask'] == False] = 0  
            im3 = axes[2].imshow(diff, cmap='RdBu_r')
            axes[2].set_title(f'interpolation_diff - {name}')
            plt.colorbar(im3, ax=axes[2])
            
            plt.tight_layout()
            plt.savefig(output_path / f'interpolation_visualization_{name}.png', dpi=300, bbox_inches='tight')
            plt.close()
 

def main():

    data_dir = "data_frome"  
    mineral_points_file = "data_frome/uranium_occurrences.gpkg"  

    interpolator = GeochemicalInterpolator(data_dir, mineral_points_file)

    results = interpolator.interpolate_dataset(method='auto', preserve_mineral_areas=True)

    output_dir = "interpolated_results"
    interpolator.save_results(results, output_dir)

    interpolator.visualize_results(results, output_dir)


if __name__ == "__main__":
    main() 