

import os
from pathlib import Path
from missing_value_analysis import MissingValueAnalyzer
from geochemical_interpolation import GeochemicalInterpolator

def main():

    config = {
        'data_directories': [
            'data_WM'
        ],
        'output_directory': 'data_WM/interpolated_results',
        'interpolation_method': 'auto',  
        'preserve_mineral_areas': True,
        'buffer_distance': 2500,  
        'only_analyze': False  
    }
    
    available_dirs = []
    for data_dir in config['data_directories']:
        if Path(data_dir).exists():
            tif_files = list(Path(data_dir).glob("*.tif"))
            if tif_files:
                available_dirs.append(data_dir)
            else:
                print(f"{data_dir}  no TIF files")
        else:
            print(f"dir {data_dir} not exists")
    
    if not available_dirs:
        print("no data dir")
        return

    for data_dir in available_dirs:

        mineral_points_file = None
        gpkg_files = list(Path(data_dir).glob("*.gpkg"))
        if gpkg_files:
            mineral_points_file = str(gpkg_files[0])
        analyzer = MissingValueAnalyzer(data_dir)
        missing_stats = analyzer.analyze_missing_values()
        
        if missing_stats.empty:
            print("no file")
            continue

        analyzer.check_mineral_points_in_missing_areas()

        analyzer.suggest_interpolation_methods(missing_stats)

        if config.get('only_analyze', False):
            print("\n" + "="*60)
            print("only analyze, skip interpolation")
            print("="*60)
            continue
        
        interpolator = GeochemicalInterpolator(data_dir, mineral_points_file)

        total_missing = missing_stats['missing_pixels'].sum()
        if total_missing == 0:
            print("no missing value")
            continue

        results = interpolator.interpolate_dataset(
            method=config['interpolation_method'],
            preserve_mineral_areas=config['preserve_mineral_areas']
        )
        
        if not results:
            print("interpolation failed")
            continue
        
        output_dir = Path(config['output_directory']) / data_dir
        interpolator.save_results(results, output_dir)

        interpolator.visualize_results(results, output_dir)

    
    print(f"\n{'='*60}")
    print("all done")
    if config.get('only_analyze', False):
        print("only analyze, no interpolation")
    else:
        print("interpolation done")
        print(f"results saved in: {config['output_directory']}")
    print(f"{'='*60}")

def quick_test():

    test_dir = "data_frome_chem4"
    if not Path(test_dir).exists():
        print(f" {test_dir} not exists")
        return

    tif_files = list(Path(test_dir).glob("*.tif"))
    if not tif_files:
        print("no file")
        return
    
    test_file = tif_files[0]
    print(f"file: {test_file}")

    test_output_dir = "test_interpolation"

    interpolator = GeochemicalInterpolator(test_dir)
    results = interpolator.interpolate_dataset(method='linear')
    
    if results:
        interpolator.save_results(results, test_output_dir)

def light_test():

    config = {
        'data_directories': ['data_MG'],
        'output_directory': 'light_test_results',
        'interpolation_method': 'auto',
        'preserve_mineral_areas': False,  
        'max_files': 3 
    }
    
    data_dir = config['data_directories'][0]
    if not Path(data_dir).exists():
        print(f" {data_dir} not exists")
        return

    tif_files = list(Path(data_dir).glob("*.tif"))[:config['max_files']]
    if not tif_files:
        print("no file")
        return
    
    for f in tif_files:
        print(f"  - {f.name}")

    temp_dir = Path("temp_light_test")
    temp_dir.mkdir(exist_ok=True)

    import shutil
    for tif_file in tif_files:
        shutil.copy2(tif_file, temp_dir / tif_file.name)

    gpkg_files = list(Path(data_dir).glob("*.gpkg"))
    if gpkg_files:
        shutil.copy2(gpkg_files[0], temp_dir / gpkg_files[0].name)
    
    try:

        interpolator = GeochemicalInterpolator(str(temp_dir))
        results = interpolator.interpolate_dataset(method='auto', preserve_mineral_areas=False)
        
        if results:
            output_dir = Path(config['output_directory'])
            interpolator.save_results(results, output_dir)
        
        else:
            print("interpolation failed")
            
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            quick_test()
        elif sys.argv[1] == 'light':
            light_test()
        else:
            main()
    else:
        main() 