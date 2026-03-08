import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
from sklearn.metrics import roc_curve
matplotlib.use('Agg')


plt.rcParams['font.family'] = 'Times New Roman'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

model_infos = [
    {"name": "GeoAttNet-Base", "history_path": "GeoAttNet-Base/result/history.npy",
     "roc_path": "GeoAttNet-Base/result/roc_data.npz"},
    {"name": "GeoAttNet-CBAM", "history_path": "GeoAttNet-CBAM/result/history.npy",
     "roc_path": "GeoAttNet-CBAM/result/roc_data.npz"},
    {"name": "GeoAttNet-Loss", "history_path": "GeoAttNet-Loss/result/history.npy",
     "roc_path": "GeoAttNet-Loss/result/roc_data.npz"},
    {"name": "GeoAttNet", "history_path": "GeoAttNet/result/history.npy",
     "roc_path": "GeoAttNet/result/roc_data.npz"},
]
OUTPUT_PATH_TEMPLATE = 'comparison_result/comodel_comparison_curves_{}.png' 
ROC_OUTPUT_PATH_TEMPLATE = 'comparison_result/roc_{}.png'  

def load_history_from_npy(history_path, model_name):
    if not os.path.exists(history_path):
        print(f"{model_name} history.npy not exist: {history_path}")
        return None
    try:
        history_data = np.load(history_path, allow_pickle=True).item()
        print(f"success load {model_name} history.npy: {history_path}")
        print(f"{model_name} history.npy include: {list(history_data.keys())}")
        return history_data
    except Exception as e:
        print(f"load{model_name} history.npy fail: {str(e)}")
        return None

def load_all_histories(model_infos):
    histories = []
    for info in model_infos:
        history = load_history_from_npy(info["history_path"], info["name"])
        histories.append({"name": info["name"], "history": history})
    return histories

def validate_all_paths(model_infos):

    for info in model_infos:
        his_exists = os.path.exists(info["history_path"])
        roc_exists = os.path.exists(info["roc_path"])
        print(f"{info['name']} history.npy: {info['history_path']} - {'✓' if his_exists else '✗'} | ROC: {info['roc_path']} - {'✓' if roc_exists else '✗'}")

def plot_comparison_curves_multi(histories, legend_order, legend_names=None):

    if legend_names is None:
        legend_names = legend_order  

    name_mapping = dict(zip(legend_order, legend_names))

    metrics = [
        ('val_loss', 'Validation Loss', 'Loss'),
        ('val_f1', 'Validation F1 Score', 'F1 Score'),
        ('val_auc', 'Validation AUC', 'AUC'),
        ('val_acc', 'Validation Accuracy', 'Accuracy'),
        ('val_precision', 'Validation Precision', 'Precision'),
        ('val_recall', 'Validation Recall', 'Recall')
    ]
    tab10_colors = list(plt.get_cmap('tab10').colors)
    set2_colors = list(plt.get_cmap('Set2').colors)
    color_list = tab10_colors + set2_colors
    color_list = color_list[:len(legend_order)]
    model_color_map = {}
    for i, name in enumerate(legend_order):
        model_color_map[name] = color_list[i]
    if 'GeoAttNet' in model_color_map and 'Swin-Unet' in model_color_map:
        model_color_map['GeoAttNet'], model_color_map['Swin-Unet'] = model_color_map['Swin-Unet'], model_color_map['GeoAttNet']
    if 'DeepUNet' in model_color_map and 'ConvNeXt' in model_color_map:
        model_color_map['DeepUNet'], model_color_map['ConvNeXt'] = model_color_map['ConvNeXt'], model_color_map['DeepUNet']
    for idx, (metric, title, ylabel) in enumerate(metrics):
        plt.figure(figsize=(8, 6))
        last_ys = []
        line_handles = {}
        for i, item in enumerate(histories):
            history = item["history"]
            name = item["name"]
            if history is not None and metric in history:
                epochs = range(1, len(history[metric]) + 1)
                color = model_color_map.get(name, color_list[i % len(color_list)])
                line, = plt.plot(epochs, history[metric], color=color, linewidth=2, label=name, marker='o', markersize=1)
                print(f"Plotting {name} {metric}: {len(history[metric])} points")
                last_ys.append(history[metric][-1])
                line_handles[name] = line
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        if metric == 'val_loss':
            plt.ylim(bottom=0)
            y_max = plt.gca().get_ylim()[1]
        elif metric in ['val_acc', 'val_precision', 'val_recall', 'val_f1', 'val_auc']:
            plt.ylim(0, 1)
            y_max = 1
        else:
            y_max = plt.gca().get_ylim()[1]
        legend_loc = 'upper right'
        if metric == 'val_recall':
            legend_loc = 'lower right'
        elif last_ys:
            if any(y > 0.8 * y_max for y in last_ys):
                legend_loc = 'lower right'
        handles = [line_handles[name] for name in legend_order if name in line_handles]
        labels = [name_mapping.get(name, name) for name in legend_order if name in line_handles]
        plt.legend(handles, labels, fontsize=11, loc=legend_loc)
        os.makedirs(os.path.dirname(OUTPUT_PATH_TEMPLATE), exist_ok=True)
        out_path = OUTPUT_PATH_TEMPLATE.format(metric)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison curve: {out_path}")
        plt.close()
    return model_color_map

def load_roc_data(roc_path, model_name):
    if not os.path.exists(roc_path):
        print(f"{model_name} ROCdata not exist: {roc_path}")
        return None
    try:
        roc_data = np.load(roc_path)
        keys = roc_data.files
        print(f"{model_name} ROC include: {keys}")
        if 'targets' in keys and 'preds' in keys:
            y_true = roc_data['targets']
            y_score = roc_data['preds']
        elif 'y_true' in keys and 'y_score' in keys:
            y_true = roc_data['y_true']
            y_score = roc_data['y_score']
        else:
            print(f"{model_name} ROC No available label or score key found in the file")
            return None
        fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
        print(f"success load {model_name} ROC curve: {roc_path}")
        return {"fpr": fpr, "tpr": tpr}
    except Exception as e:
        print(f"load{model_name} ROC data fail: {str(e)}")
        return None

def plot_roc_curves(model_infos, model_color_map, legend_order, legend_names=None):

    if legend_names is None:
        legend_names = legend_order
    name_mapping = dict(zip(legend_order, legend_names))

    ordered_infos = [info for name in legend_order for info in model_infos if info['name'] == name]
    plt.figure(figsize=(8, 8))
    handles = []
    labels = []
    for info in ordered_infos:
        roc = load_roc_data(info["roc_path"], info["name"])
        if roc is not None:
            color = model_color_map.get(info['name'], None)
            display_name = name_mapping.get(info["name"], info["name"])
            line, = plt.plot(roc["fpr"], roc["tpr"], label=display_name, linewidth=2, color=color)
            handles.append(line)
            labels.append(display_name)
    plt.plot([0, 1], [0, 1], 'k--')  
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve Comparison')
    plt.legend(handles, labels, fontsize=11, loc='lower right')
    os.makedirs(os.path.dirname('comparison_result/comodel_roc_comparison.png'), exist_ok=True)
    plt.savefig('comparison_result/comodel_roc_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved ROC curve comparison: comparison_result/comodel_roc_comparison.png")
    plt.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Comparing model training curves and ROC curves')
    parser.add_argument('--legend-names', nargs='+',
                       help='Customize legend names, corresponding to the models in model_infos in order')
    return parser.parse_args()

def main():

    args = parse_arguments()

    legend_order = [info["name"] for info in model_infos]

    legend_names = args.legend_names if args.legend_names else None

    if legend_names and len(legend_names) != len(legend_order):
        print(f"Warning: The number of provided legend names ({len(legend_names)}) does not match the number of models ({len(legend_order)}).")
        return

    if legend_names:
        print(f"Custom legend names：{legend_names}")

    validate_all_paths(model_infos)
    histories = load_all_histories(model_infos)
    if all(h["history"] is None for h in histories):
        print("No valid history data loaded, exit")
        return
    print("\nNumber of epochs for each model:")
    for item in histories:
        h = item["history"]
        print(f"{item['name']}: {len(h.get('val_loss', [])) if h else 0}")
    print("\n3. Plotting training curves...")
    model_color_map = plot_comparison_curves_multi(histories, legend_order, legend_names)
    print("\n4. Plotting ROC curves...")
    plot_roc_curves(model_infos, model_color_map, legend_order, legend_names)
    print("\n=== Comparison finished ===")

if __name__ == '__main__':
    main() 