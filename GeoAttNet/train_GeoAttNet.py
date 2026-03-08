
import os
import numpy as np
import torch
import torch.nn as nn
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sys
import cv2
from datetime import datetime
import random
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
import seaborn as sns
from tqdm import tqdm
from data_selection import prepare_blocks_for_training

def _is_validation_patch(position, stride, n_patches_h, n_patches_w):
    """4x4 分块时，右下角 4 块为验证集。块索引 (bi,bj) 满足 bi>=2 且 bj>=2 的为验证块。"""
    start_h, start_w = position
    i, j = start_h // stride, start_w // stride
    block_h, block_w = max(1, n_patches_h // 4), max(1, n_patches_w // 4)
    bi, bj = min(3, i // block_h), min(3, j // block_w)
    return (bi >= 2 and bj >= 2)

def _split_patches_by_spatial(patches_list, stride, n_patches_h, n_patches_w):
    """将 patch 列表按 4x4 空间划分为训练集与验证集。"""
    train_patches, val_patches = [], []
    for p in patches_list:
        if _is_validation_patch(p['position'], stride, n_patches_h, n_patches_w):
            val_patches.append(p)
        else:
            train_patches.append(p)
    return train_patches, val_patches
import json
import argparse
import multiprocessing


os.environ['PYTHONWARNINGS'] = 'ignore'
if hasattr(os, 'environ') and 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_optimal_num_workers():
    cpu_count = multiprocessing.cpu_count()

    if cpu_count >= 8:
        return 4
    elif cpu_count >= 4:
        return 2
    else:
        return 0


result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
os.makedirs(result_dir, exist_ok=True)

def safe_print(*args, **kwargs):
    try:
        if hasattr(os, 'getpid'):
            main_pid = os.getpid()
            current_pid = os.getpid()
            if current_pid == main_pid:
                print(*args, **kwargs)
    except:
        print(*args, **kwargs)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, dice_weight=0.6, focal_weight=0.4):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def dice_loss(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=10.0, threshold=0.5):
        super().__init__()
        self.pos_weight = pos_weight
        self.threshold = threshold

    def forward(self, logits, target):

        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight], device=logits.device))(logits, target)
        return bce

    def predict(self, logits):
        return torch.sigmoid(logits)


class PatchDataset(Dataset):
    def __init__(self, patches_info, is_train=True, normalization_params=None):
        """不使用数据增强。训练/验证按 4x4 空间划分：整图分为 4x4 块，右下角 4 块为验证集，其余为训练集。"""
        self.is_train = is_train
        self.normalization_params = normalization_params

        pos_patches = patches_info['pos_patches']
        neg_patches = patches_info['neg_patches']
        stride = patches_info['stride']
        n_patches_h = patches_info['n_patches_h']
        n_patches_w = patches_info['n_patches_w']

        safe_print(f"\nPositive samples patch: {len(pos_patches)}, Negative samples patch: {len(neg_patches)}")

        train_pos, val_pos = _split_patches_by_spatial(pos_patches, stride, n_patches_h, n_patches_w)
        train_neg, val_neg = _split_patches_by_spatial(neg_patches, stride, n_patches_h, n_patches_w)

        if is_train:
            selected_pos_patches = train_pos
            selected_neg_pool = train_neg
            safe_print(f"Number of positive samples in the training set: {len(selected_pos_patches)}")
        else:
            selected_pos_patches = val_pos
            selected_neg_pool = val_neg
            safe_print(f"Number of positive samples in the validation set: {len(selected_pos_patches)}")

        n_neg_needed = int(len(selected_pos_patches) * 2)
        if len(selected_neg_pool) >= n_neg_needed:
            selected_neg_patches = random.sample(selected_neg_pool, n_neg_needed)
        else:
            selected_neg_patches = random.choices(selected_neg_pool, k=n_neg_needed) if selected_neg_pool else []

        self.patches = selected_pos_patches + selected_neg_patches
        self.labels = [1] * len(selected_pos_patches) + [0] * len(selected_neg_patches)

        combined = list(zip(self.patches, self.labels))
        random.shuffle(combined)
        self.patches, self.labels = zip(*combined)

        safe_print(f" ({'training set' if is_train else 'Validation set'}): Total number of samples: {len(self.patches)}, Number of positive samples: {sum(self.labels)}, Number of negative samples: {len(self.labels) - sum(self.labels)}, Positive and negative ratio: 1:{(len(self.labels) - sum(self.labels)) / sum(self.labels) if sum(self.labels) > 0 else 'inf'}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        label = self.labels[idx]

        x = patch_info['data'].copy()
        y = patch_info['label'].copy()

        if self.normalization_params is not None:
            means = self.normalization_params['means']
            stds = self.normalization_params['stds']
            for i in range(x.shape[0]):
                x[i] = (x[i] - means[i]) / (stds[i] + 1e-8)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        y = y.unsqueeze(0)

        return x, y


def prepare_datasets(data_files, label_fn, target_size=(2592, 2016)):

    safe_print("\n=== Prepare the dataset ===")
    
    patches_info = prepare_blocks_for_training(
        data_files=data_files,
        label_fn=label_fn,
        target_size=target_size
    )
    
    safe_print("\n=== Calculate normalization parameters ===")
    stride = patches_info['stride']
    n_patches_h = patches_info['n_patches_h']
    n_patches_w = patches_info['n_patches_w']
    train_pos_patches, _ = _split_patches_by_spatial(
        patches_info['pos_patches'], stride, n_patches_h, n_patches_w
    )

    train_data_list = []
    for patch in train_pos_patches:
        train_data_list.append(patch['data'])

    train_data_array = np.stack(train_data_list, axis=0) 
    n_features = train_data_array.shape[1]
    
    means = []
    stds = []
    for i in range(n_features):
        feature_data = train_data_array[:, i, :, :].flatten()
        valid_data = feature_data[np.isfinite(feature_data)]
        if len(valid_data) > 0:
            mean = np.nanmean(valid_data)
            std = np.nanstd(valid_data)
        else:
            mean = 0
            std = 1
        means.append(mean)
        stds.append(std)
    
    normalization_params = {
        'means': np.array(means),
        'stds': np.array(stds)
    }
    
    train_dataset = PatchDataset(
        patches_info=patches_info,
        is_train=True,
        normalization_params=normalization_params
    )

    val_dataset = PatchDataset(
        patches_info=patches_info,
        is_train=False,
        normalization_params=normalization_params
    )
    
    return train_dataset, val_dataset


def find_optimal_threshold(y_true, y_pred):

    thresholds = np.arange(0.5, 0.85, 0.05)
    best_f1 = 0
    best_threshold = 0.6  
    
    for threshold in thresholds:
        y_pred_binary = (y_pred > threshold).astype(int)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def plot_history(history):

    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training loss')
    plt.plot(history['val_loss'], label='Validation loss')
    plt.title('Loss curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Training accuracy')
    plt.plot(history['val_acc'], label='Verification accuracy')
    plt.title('Accuracy curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(history['train_auc'], label='Training AUC')
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.title('AUC curve')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(history['train_f1'], label='Training F1')
    plt.plot(history['val_f1'], label='Verify F1')
    plt.title('F1 curve')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'training_history.png'))
    plt.close()


def calculate_metrics(all_preds, all_targets, threshold=0.5, verbose=True):

    optimal_threshold, optimal_f1 = find_optimal_threshold(all_targets, all_preds)
    if verbose:
        safe_print(f"Optimal threshold: {optimal_threshold:.3f}, F1: {optimal_f1:.4f}")

    precision, recall, thresholds = precision_recall_curve(all_targets, all_preds)
    ap = average_precision_score(all_targets, all_preds)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Accuracy')
    plt.title('PR curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, 'pr_curve.png'))
    plt.close()

    pred_labels = (all_preds > optimal_threshold).astype(int)
    cm = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            cm[i, j] = np.sum((all_targets == i) & (pred_labels == j))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix (Threshold={optimal_threshold:.3f})')
    plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'))
    plt.close()

    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    if verbose:
        safe_print(f"\n Use the optimal threshold {optimal_threshold:.3f} Evaluation results:")
        safe_print(f"Accuracy: {accuracy:.4f}")
        safe_print(f"Precision: {precision:.4f}")
        safe_print(f"Recall: {recall:.4f}")
        safe_print(f"F1 score: {f1:.4f}")
        safe_print(f"AP: {ap:.4f}")
        safe_print(f"Confusion Matrix:\n{cm}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ap': ap,
        'optimal_threshold': optimal_threshold
    }


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, scheduler=None, patience=15, start_time=None):
    best_model_path = os.path.join(result_dir, 'best_model.pth')
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [],
        'train_auc': [], 'val_auc': [],
        'train_f1': [], 'val_f1': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': []
    }
    best_val_f1 = 0
    patience_counter = 0  

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_targets = []
        
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} '):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.sigmoid(outputs)
            train_preds.extend(preds.cpu().detach().numpy().flatten())
            train_targets.extend(y.cpu().numpy().flatten())
            
            optimal_threshold, _ = find_optimal_threshold(
                np.array(train_targets), np.array(train_preds)
            )
            binary_preds = (preds > optimal_threshold).float()
            train_correct += (binary_preds == y).sum().item()
            train_total += y.numel()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        train_auc = roc_auc_score(train_targets, train_preds)
        
        optimal_threshold, _ = find_optimal_threshold(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds > optimal_threshold)
        train_precision = precision_score(train_targets, train_preds > optimal_threshold)
        train_recall = recall_score(train_targets, train_preds > optimal_threshold)
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                
                val_loss += loss.item()
                preds = torch.sigmoid(outputs)
                val_preds.extend(preds.cpu().numpy().flatten())
                val_targets.extend(y.cpu().numpy().flatten())
                
                optimal_threshold, _ = find_optimal_threshold(
                    np.array(val_targets), np.array(val_preds)
                )
                binary_preds = (preds > optimal_threshold).float()
                val_correct += (binary_preds == y).sum().item()
                val_total += y.numel()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)
        
        optimal_threshold, val_f1 = find_optimal_threshold(val_targets, val_preds)
        val_precision = precision_score(val_targets, val_preds > optimal_threshold)
        val_recall = recall_score(val_targets, val_preds > optimal_threshold)
        
        if scheduler is not None:
            scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0   
            torch.save(model.state_dict(), best_model_path)

        else:
            patience_counter += 1  

    if start_time:
        from datetime import datetime
        end_time = datetime.now()
        training_duration = end_time - start_time
        history['training_duration'] = str(training_duration)
        history['start_time'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
        history['end_time'] = end_time.strftime('%Y-%m-%d %H:%M:%S')
    
    plot_history(history)
    history_dict = {
        'val_loss': history['val_loss'],
        'val_f1': history['val_f1'],
        'val_auc': history['val_auc'],
        'val_acc': history['val_acc'],
        'val_precision': history['val_precision'],
        'val_recall': history['val_recall']
    }
    np.save(os.path.join(result_dir, 'history.npy'), history_dict)
    return history


def main():
    parser = argparse.ArgumentParser(description='GeoAttNet')
    parser.add_argument('--epochs', type=int, default=250,
                       help='Number of training rounds (default: 100)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='learning rate (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience value (default: 15)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers (default: autoselect)')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = get_optimal_num_workers()

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


    train_dataset, val_dataset = prepare_datasets(
        data_files=data_files,
        label_fn=label_fn,
        target_size=(2592, 2016)
    )

    if hasattr(train_dataset, 'normalization_params'):
        np.save('train_stats_frome.npy', train_dataset.normalization_params)
    elif hasattr(val_dataset, 'normalization_params'):
        np.save('train_stats_frome.npy', val_dataset.normalization_params)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    from GeoAttNet.GeoAttNet_model import DeepUNet
    model = DeepUNet(in_channels=15, num_classes=1, dropout_rate=0.2).to(device)

    criterion = CombinedLoss(alpha=1, gamma=2, dice_weight=0.7, focal_weight=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    start_time = datetime.now()
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        scheduler=scheduler,
        patience=args.patience,  
        start_time=start_time
    )

    best_model_path = os.path.join(result_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        safe_print("The best model file was not found, using the current model for evaluation")

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.sigmoid(outputs)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())

    metrics = calculate_metrics(np.array(all_preds), np.array(all_targets))
    for metric_name, value in metrics.items():
        safe_print(f"{metric_name}: {value:.4f}")
    
    save_final_metrics(metrics, history, result_dir)

    np.savez(
        os.path.join(result_dir, 'roc_data.npz'),
        y_true=np.array(all_targets),
        y_score=np.array(all_preds)
    )

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(np.array(all_targets), np.array(all_preds))
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True rate (TPR)')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, 'roc_curve.png'))
    plt.close()
    safe_print(f"The ROC curve has been saved to: {result_dir}/roc_curve.png")


def save_final_metrics(metrics, history, result_dir):

    import json
    from datetime import datetime
    
    final_metrics = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_info': {
            'model_name': 'DeepUNet',
            'input_channels': 15,
            'num_classes': 1,
            'dropout_rate': 0.2
        },
        'final_metrics': metrics,
        'training_summary': {
            'best_val_f1': max(history['val_f1']) if history['val_f1'] else 0,
            'best_val_auc': max(history['val_auc']) if history['val_auc'] else 0,
            'best_val_acc': max(history['val_acc']) if history['val_acc'] else 0,
            'final_epoch': len(history['val_loss']),
            'training_duration': history.get('training_duration', ''),
            'start_time': history.get('start_time', ''),
            'end_time': history.get('end_time', '')
        }
    }

    with open(os.path.join(result_dir, 'final_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)
    
    training_metrics = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_history': history,
        'data_info': {
            'input_files_count': 15,
            'patch_size': 32,
            'stride': 32,
            'train_val_split': '4x4_spatial_bottom_right_4_blocks_as_val',
            'pos_neg_ratio': '1:2'
        }
    }
    
    with open(os.path.join(result_dir, 'training_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(training_metrics, f, indent=2, ensure_ascii=False)
    
    create_text_report(final_metrics, history, result_dir)


def create_text_report(final_metrics, history, result_dir):

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("GeoAttNet Training Completion Report")
    report_lines.append("=" * 60)
    report_lines.append(f"Training completion time: {final_metrics['timestamp']}")
    report_lines.append("")

    report_lines.append("Model Information:")
    report_lines.append(f"  Model Name: {final_metrics['model_info']['model_name']}")
    report_lines.append(f"  Number of input channels: {final_metrics['model_info']['input_channels']}")
    report_lines.append(f"  Number of output categories: {final_metrics['model_info']['num_classes']}")
    report_lines.append(f"  Dropout: {final_metrics['model_info']['dropout_rate']}")
    report_lines.append("")
    
    report_lines.append("Final evaluation metrics:")
    metrics = final_metrics['final_metrics']
    report_lines.append(f"   (Accuracy): {metrics['accuracy']:.4f}")
    report_lines.append(f"   (Precision): {metrics['precision']:.4f}")
    report_lines.append(f"   (Recall): {metrics['recall']:.4f}")
    report_lines.append(f"  F1: {metrics['f1']:.4f}")
    report_lines.append(f"   (AP): {metrics['ap']:.4f}")
    report_lines.append(f"  Optimal threshold: {metrics['optimal_threshold']:.4f}")
    report_lines.append("")

    report_lines.append("Training Summary:")
    summary = final_metrics['training_summary']
    report_lines.append(f"  Best Verification F1: {summary['best_val_f1']:.4f}")
    report_lines.append(f"  Best validation AUC: {summary['best_val_auc']:.4f}")
    report_lines.append(f"  Best validation accuracy: {summary['best_val_acc']:.4f}")
    report_lines.append(f"  Number of training rounds: {summary['final_epoch']}")
    report_lines.append(f"  Training start time: {summary['start_time']}")
    report_lines.append(f"  Training end time: {summary['end_time']}")
    report_lines.append(f"  Total training time: {summary['training_duration']}")
    report_lines.append("")
    
    if history['val_loss']:
        report_lines.append("Training history statistics:")
        report_lines.append(f"  Final training loss: {history['train_loss'][-1]:.4f}")
        report_lines.append(f"  Final validation loss: {history['val_loss'][-1]:.4f}")
        report_lines.append(f"  Loss Improvement: {history['val_loss'][0] - history['val_loss'][-1]:.4f}")
        report_lines.append(f"  F1 Improvement: {history['val_f1'][-1] - history['val_f1'][0]:.4f}")
        report_lines.append("")

    with open(os.path.join(result_dir, 'training_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    safe_print('\n'.join(report_lines))

if __name__ == '__main__':
    main()
