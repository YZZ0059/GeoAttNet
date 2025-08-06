import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MissingValueAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.tif_files = list(self.data_dir.glob("*.tif"))
        self.gpkg_files = list(self.data_dir.glob("*.gpkg"))
        
    def analyze_missing_values(self):
    
        missing_stats = []
        
        for tif_file in self.tif_files:
            try:
                with rasterio.open(tif_file) as src:
                    data = src.read(1)  
                    
                    total_pixels = data.size
                    missing_pixels = np.sum(np.isnan(data)) + np.sum(data == src.nodata) if src.nodata is not None else np.sum(np.isnan(data))
                    missing_percentage = (missing_pixels / total_pixels) * 100
                    
                    missing_stats.append({
                        'file_name': tif_file.name,
                        'total_pixels': total_pixels,
                        'missing_pixels': missing_pixels,
                        'missing_percentage': missing_percentage,
                        'data_shape': data.shape,
                        'nodata_value': src.nodata,
                        'data_type': data.dtype
                    })

                    
            except Exception as e:
                print(f"no file {tif_file.name}: {e}")
                
        return pd.DataFrame(missing_stats)
    
    def check_mineral_points_in_missing_areas(self):

        if self.gpkg_files:
            mineral_points = gpd.read_file(self.gpkg_files[0])

            for tif_file in self.tif_files:
                try:
                    with rasterio.open(tif_file) as src:
                     
                        mineral_coords = [(point.x, point.y) for point in mineral_points.geometry]
                        mineral_pixels = [src.index(x, y) for x, y in mineral_coords]
                        
                    
                        data = src.read(1)
                        missing_in_mineral_areas = 0
                        
                        for row, col in mineral_pixels:
                            if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
                                if np.isnan(data[row, col]) or (src.nodata is not None and data[row, col] == src.nodata):
                                    missing_in_mineral_areas += 1
                        
                        print(f"{tif_file.name}: {missing_in_mineral_areas}/{len(mineral_points)} Mineral points are in the missing value area")
                        
                except Exception as e:
                    print(f"anaylize {tif_file.name} error: {e}")
    
    def suggest_interpolation_methods(self, missing_stats_df):
        
        for _, row in missing_stats_df.iterrows():
            missing_pct = row['missing_percentage']
            file_name = row['file_name']
            
            if missing_pct < 5:
                print("  Recommended method: Simple interpolation")
                
            elif missing_pct < 20:
                print("  Recommended method: Statistical interpolation")
    
            elif missing_pct < 50:
                print("  Recommended method: Advanced interpolation")
   
            else:
                print(" Recommended method: Data reconstruction")
            print("-" * 50)


def main():

    data_dirs = ['data_frome', 'data_xiao', 'data_aisashan', 'dataset']
    
    for data_dir in data_dirs:
        if Path(data_dir).exists():
            
            analyzer = MissingValueAnalyzer(data_dir)

            missing_stats = analyzer.analyze_missing_values()

            analyzer.check_mineral_points_in_missing_areas()

            analyzer.suggest_interpolation_methods(missing_stats)

            analyzer.create_interpolation_pipeline()

if __name__ == "__main__":
    main() 