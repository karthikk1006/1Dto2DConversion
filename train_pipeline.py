"""
train_pipeline.py
=================
Trains all 4 combinations of (model × method):
  - efficientnet + NCTD
  - efficientnet + ours
  - cnn          + NCTD
  - cnn          + ours
 
Output structure
----------------
results/
  efficientnet/
    nctd/
      metrics/      ← per-dataset JSON + confusion-matrix PNGs
      models/       ← best_*.pth  +  final_*.pth
      logs/         ← combo-level training log
    ours/  ...
  cnn/
    nctd/  ...
    ours/  ...
  logs/             ← master experiment log
 
Console output is intentionally minimal:
  - One banner line per combo
  - One header line per dataset
  - One tqdm bar per dataset (epochs, with live loss/acc postfix)
  - One result line per dataset
  - One summary table per combo
Everything verbose (GPU stats, splits, paths) goes to the log file only.
 
Usage
-----
  python train_pipeline.py --all
  python train_pipeline.py --method NCTD --model efficientnet
  python train_pipeline.py --method ours --model nctd_cnn
 
Colab bootstrap (uncomment when running on Google Colab)
---------------------------------------------------------
  from google.colab import drive
  drive.mount('/content/drive')
  import subprocess
  subprocess.run(["unzip", "/content/drive/MyDrive/2d_datasets.zip"], check=True)
  subprocess.run(["mv",  "/content/content/2d_datasets/", "/content/2d_datasets/"], check=True)
  subprocess.run(["rmdir", "/content/content/"], check=True)
"""
 
import os
import sys
import json
import time
import logging
import argparse
import datetime
 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
 
# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIR    = "2d_datasets"
RESULTS_DIR = "results"
 
# ═══════════════════════════════════════════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════════════════════════════════════════
DATASETS_LIST = [
    '13_rotated_rastrigin_50d.npz',
    'digen39_5578.npz', 'digen23_5191.npz', 'digen26_7270.npz',
    'digen15_5311.npz', 'digen20_5191.npz', 'digen2_6949.npz',
    'digen24_2433.npz', 'digen31_2433.npz', 'digen36_466.npz',
    'digen40_5390.npz', 'digen11_7270.npz', 'digen33_769.npz',
    '06_friedman1.npz', 'digen6_466.npz',   'digen13_769.npz',
    'digen37_769.npz',  '08_friedman3.npz', '07_friedman2.npz',
    'digen19_7270.npz', 'digen7_6949.npz',  'digen8_4426.npz',
    'digen29_8322.npz', 'digen10_8322.npz', 'digen35_4426.npz',
    'digen28_769.npz',  'digen12_8322.npz', 'digen32_5191.npz',
    'digen14_769.npz',  'digen5_6949.npz',  'digen22_2433.npz',
    'digen17_6949.npz', 'digen16_5390.npz', 'digen21_6265.npz',
    'digen30_4426.npz', 'digen38_4426.npz', 'digen27_860.npz',
    'digen34_769.npz',  'digen25_2433.npz', 'digen18_5578.npz',
    'digen1_6265.npz',  'digen4_860.npz',   'digen3_769.npz',
    'digen9_7270.npz',
]
 
# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# Console → INFO  (clean, minimal)
# File    → DEBUG (full trace: splits, GPU mem, checkpoint paths, …)
# ═══════════════════════════════════════════════════════════════════════════════
_LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_RUN_TS   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
 
 
def _make_logger(name: str, log_file: str) -> logging.Logger:
    """
    Returns a logger that:
      • writes INFO+ to stdout  (clean console view)
      • writes DEBUG+ to file   (full trace for diagnostics)
    Idempotent — safe to call multiple times with the same name.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False          # don't bubble up to root logger
 
    fmt = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
 
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
 
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
 
    return logger
 
 
def _master_logger() -> logging.Logger:
    log_file = os.path.join(RESULTS_DIR, "logs", f"experiment_{_RUN_TS}.log")
    return _make_logger("master", log_file)
 
 
def _combo_logger(model_key: str, method_key: str) -> logging.Logger:
    log_file = os.path.join(
        RESULTS_DIR, model_key, method_key, "logs", f"train_{_RUN_TS}.log"
    )
    return _make_logger(f"{model_key}.{method_key}", log_file)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE SETUP
#
# WHY TWO FUNCTIONS:
#   DataLoader(num_workers>0) on Windows uses the "spawn" start method, so each
#   worker process re-imports this entire module from scratch.  If device setup
#   also logs, the device line appears once per worker per combo start — that's
#   what caused the repeated prints.
#
#   _configure_device()  — silent; runs in main process AND every worker
#   _log_device_info()   — logging only; called once from __main__ guard
# ═══════════════════════════════════════════════════════════════════════════════
def _configure_device() -> tuple:
    """
    Sets cuDNN/TF32 flags and returns (device, use_amp).
    No logging — safe to run in DataLoader worker processes.
    """
    if torch.cuda.is_available():
        device  = torch.device("cuda")
        props   = torch.cuda.get_device_properties(0)
        torch.backends.cudnn.benchmark        = True
        torch.backends.cudnn.deterministic     = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True
        use_amp = props.major >= 7   # FP16 AMP on Volta / Turing / Ampere
    else:
        device  = torch.device("cpu")
        use_amp = False
    return device, use_amp
 
 
def _log_device_info() -> None:
    """
    Logs device info exactly once.  Call only from the __main__ block.
    """
    master = _master_logger()
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(DEVICE)
        master.debug("── Device Setup ──────────────────────────────────────")
        master.debug(f"PyTorch        : {torch.__version__}")
        master.debug(f"CUDA           : {torch.version.cuda}")
        master.debug(f"cuDNN          : {torch.backends.cudnn.version()}")
        master.debug(f"GPU name       : {props.name}")
        master.debug(f"Compute cap.   : {props.major}.{props.minor}")
        master.debug(f"Total VRAM     : {props.total_memory / 1024**2:.0f} MB")
        master.debug(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        master.debug(f"TF32 matmul    : {torch.backends.cuda.matmul.allow_tf32}")
        master.debug(f"AMP (FP16)     : {_USE_AMP}")
        master.debug("──────────────────────────────────────────────────────")
        master.info(
            f"Device: cuda — {props.name} "
            f"({props.total_memory // 1024**2} MB) | "
            f"AMP={'on' if _USE_AMP else 'off'} | "
            f"PyTorch {torch.__version__}"
        )
    else:
        master.debug(f"PyTorch : {torch.__version__} | no CUDA")
        master.info(f"Device: cpu | PyTorch {torch.__version__}")
 
 
# Silent at import time — logging happens only from __main__
DEVICE, _USE_AMP = _configure_device()
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# FOLDER HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _get_dirs(model_type: str, method_name: str) -> tuple:
    """
    Creates and returns (dirs_dict, model_key, method_key).
 
    results/
      efficientnet/          ← model_key
        nctd/                ← method_key
          metrics/
          models/
          logs/
        ours/ …
      cnn/
        nctd/ …
        ours/ …
    """
    model_key  = "efficientnet" if model_type == "efficientnet" else "cnn"
    method_key = method_name.lower()
    base       = os.path.join(RESULTS_DIR, model_key, method_key)
    dirs = {
        "metrics": os.path.join(base, "metrics"),
        "models":  os.path.join(base, "models"),
        "logs":    os.path.join(base, "logs"),
    }
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)
    return dirs, model_key, method_key
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
class LazyXDataset:
    def __init__(self, dataset_dir, cache_size=2):
        self.files = sorted([
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.endswith(".pt")
        ])
        self.index_map = []
        self.y_labels = []
        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size

        for file_idx, file in enumerate(self.files):
            data = torch.load(file, map_location="cpu", weights_only=True)
            size = data['X'].shape[0]
            for i in range(size):
                self.index_map.append((file_idx, i))
                self.y_labels.append(data['y'][i].item())
            del data
        self.y = torch.tensor(self.y_labels, dtype=torch.long)

    def __len__(self):
        return len(self.index_map)

    def _load_file(self, file_idx):
        if file_idx in self.cache:
            return self.cache[file_idx]

        if len(self.cache) >= self.cache_size:
            old = self.cache_order.pop(0)
            del self.cache[old]

        data = torch.load(self.files[file_idx], map_location="cpu", weights_only=True)
        self.cache[file_idx] = data
        self.cache_order.append(file_idx)
        return data

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]
        data = self._load_file(file_idx)
        return data['X'][sample_idx]

def load_2d_datasets(
    dataset_filename: str,
    method_name: str,
    main_dir: str = DATA_DIR,
    logger: logging.Logger = None,
    lazy_loading: bool = False,
    full_loading: bool = False,
):
    """
    Loads processed dataset.
    Returns X (Tensor or LazyXDataset) and y (Tensor).
    """
    log       = logger or _master_logger()
    base_name = os.path.splitext(dataset_filename)[0]
    
    # Chunked directory path vs singular file path
    dir_path = os.path.join(main_dir, method_name, base_name)
    file_path = os.path.join(main_dir, method_name, f"processed_{base_name}.pt")
    
    if os.path.isdir(dir_path) and (lazy_loading or full_loading):
        if lazy_loading:
            log.debug(f"Lazy loading chunked dataset: {dir_path}")
            lazy_ds = LazyXDataset(dir_path)
            return lazy_ds, lazy_ds.y
        else:
            log.debug(f"Full loading chunked dataset: {dir_path}")
            files = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".pt")])
            X_all, y_all = [], []
            for f in files:
                data = torch.load(f, map_location="cpu", weights_only=True)
                X_all.append(data['X'])
                y_all.append(data['y'])
            X = torch.cat(X_all, dim=0)
            y = torch.cat(y_all, dim=0)
            if torch.isnan(X).any():
                log.warning(f"NaN detected in input features (X) for {dataset_filename}. Replacing with zeros.")
                X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Check for extreme values and normalize if necessary
            x_min, x_max = X.min(), X.max()
            if x_max > 1e3 or x_min < -1e3:
                log.warning(f"Extreme values detected in {dataset_filename} (min={x_min:.2f}, max={x_max:.2f}). Scaling to [0, 1].")
                X = (X - x_min) / (x_max - x_min + 1e-8)
            log.debug(f"X={tuple(X.shape)}  y={tuple(y.shape)}")
            return X, y
    else:
        log.debug(f"Loading singular: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Processed file not found: {file_path}")
        data = torch.load(file_path, map_location="cpu", weights_only=True)
        X, y = data["X"], data["y"]
        if torch.isnan(X).any():
            log.warning(f"NaN detected in input features (X) for {dataset_filename}. Replacing with zeros.")
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check for extreme values and normalize if necessary
        x_min, x_max = X.min(), X.max()
        if x_max > 1e3 or x_min < -1e3:
            log.warning(f"Extreme values detected in {dataset_filename} (min={x_min:.2f}, max={x_max:.2f}). Scaling to [0, 1].")
            X = (X - x_min) / (x_max - x_min + 1e-8)
        log.debug(f"X={tuple(X.shape)}  y={tuple(y.shape)}  classes={y.unique().numel()}")
        return X, y

 
 
# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════
class Tabular2ImageDataset(Dataset):
    def __init__(self, X, y, model_type="efficientnet", transform=None, indices=None):
        self.X, self.y    = X, y.long()
        self.model_type   = model_type
        self.transform    = transform
        self.indices      = indices
        
        # Remap labels to be 0-indexed and contiguous to prevent CUDA device-side asserts (CrossEntropy out of bounds)
        unique_labels = torch.unique(self.y)
        if unique_labels.max() >= len(unique_labels) or unique_labels.min() < 0:
            mapping = {val.item(): i for i, val in enumerate(unique_labels.sort().values)}
            mapped_y = self.y.clone()
            for old_val, new_val in mapping.items():
                mapped_y[self.y == old_val] = new_val
            self.y = mapped_y
 
    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.y)
 
    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        img = self.X[real_idx]
        target = self.y[real_idx]

        if self.model_type == "nctd_cnn":
            img = img.mean(dim=0, keepdim=True)   # (3,H,W) → (1,H,W) grayscale
        if self.transform:
            img = self.transform(img)
        return img, target
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS
# EfficientNet → ImageNet norms (https://pytorch.org/vision/stable/models.html)
# NCTD paper   → raw, no aug    (https://doi.org/10.1038/s41598-025-01568-0)
# ═══════════════════════════════════════════════════════════════════════════════
efficientnet_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
efficientnet_val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
nctd_transform = transforms.Compose([])
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════
class NCTD_CNN(nn.Module):
    """Exact CNN from the NCTD 2025 paper (Fig. 6)."""
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),           nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),           nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),           nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier  = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )
 
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x.view(x.size(0), -1))
 
 
def get_model(model_type: str, num_classes: int) -> nn.Module:
    if model_type == "efficientnet":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    elif model_type == "nctd_cnn":
        return NCTD_CNN(num_classes=num_classes, in_channels=1)
    raise ValueError(f"Unknown model_type '{model_type}'.")
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_model(model: nn.Module, loader: DataLoader, logger: logging.Logger):
    """
    Full evaluation pass for binary classification.
 
    NOTE: autocast is intentionally disabled here.
    During training, GradScaler + autocast work together safely.
    During inference, autocast alone can let FP16 overflow to Inf/NaN on
    confident deep models (EfficientNet with Swish activations especially).
    We always cast outputs to float32 for evaluation.
 
    Metrics:
      accuracy, f1 (binary), f1_macro, f1_weighted,
      precision, recall, balanced_accuracy,
      roc_auc (NaN if test set is single-class),
      mcc, cohen_kappa
    """
    model.eval()
    all_logits, all_trues = [], []
 
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE, non_blocking=True)
            # No autocast here — inference always in float32 to avoid FP16 overflow
            out = model(Xb).float()
            all_logits.append(out.cpu().numpy())
            all_trues.append(yb.numpy())
 
    L  = np.concatenate(all_logits)    # (N, 2)  float32 logits
    T  = np.concatenate(all_trues)     # (N,)    ground truth
 
    # Guard: if training diverged, logits may contain Inf/NaN
    if not np.isfinite(L).all():
        n_bad = (~np.isfinite(L)).sum()
        logger.warning(
            f"Non-finite logits detected ({n_bad} values). "
            "Model may have diverged. Replacing with zeros for safe metric computation."
        )
        L = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
 
    P  = np.argmax(L, axis=1)          # (N,)    hard predictions
 
    # Numerically stable softmax → P(class=1) for ROC-AUC
    shifted = L - L.max(axis=1, keepdims=True)
    exp_L   = np.exp(shifted)
    probas  = exp_L / exp_L.sum(axis=1, keepdims=True)
    
    num_classes = L.shape[1]
    
    # roc_auc_score requires both classes in the test set
    if len(np.unique(T)) < 2:
        roc_auc = float("nan")
        logger.warning(
            f"roc_auc skipped — test set has only one class "
            f"({np.unique(T).tolist()}, n={len(T)})."
        )
    else:
        if num_classes == 2:
            roc_auc = float(roc_auc_score(T, probas[:, 1]))
        else:
            roc_auc = float(roc_auc_score(T, probas, multi_class='ovr'))

    # F1, Prec, Rec average setting
    avg_setting = "binary" if num_classes == 2 else "macro"

    metrics = {
        "accuracy":          float(accuracy_score(T, P)),
        "f1":                float(f1_score(T, P, average=avg_setting,   zero_division=0)),
        "f1_macro":          float(f1_score(T, P, average="macro",    zero_division=0)),
        "f1_weighted":       float(f1_score(T, P, average="weighted", zero_division=0)),
        "precision":         float(precision_score(T, P, average=avg_setting, zero_division=0)),
        "recall":            float(recall_score(T, P,    average=avg_setting, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(T, P)),
        "roc_auc":           roc_auc,
        "mcc":               float(matthews_corrcoef(T, P)),
        "cohen_kappa":       float(cohen_kappa_score(T, P)),
        "num_classes":       int(L.shape[1]),
        "num_samples":       int(len(T)),
        "per_class_report":  classification_report(T, P, output_dict=True, zero_division=0),
    }
 
    auc_str = f"{roc_auc:.4f}" if np.isfinite(roc_auc) else "N/A"
    logger.debug(
        f"eval acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  "
        f"prec={metrics['precision']:.4f}  rec={metrics['recall']:.4f}  "
        f"auc={auc_str}  mcc={metrics['mcc']:.4f}"
    )
    return metrics, P, T
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def train_uniform_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dataset_name: str,
    model_type: str,
    models_dir: str,
    logger: logging.Logger,
) -> str:
    """
    NCTD paper hyperparameters: Adam lr=0.0008, batch=64, 30 epochs.
    Uses torch.amp (replaces deprecated torch.cuda.amp) for mixed-precision.
 
    Console output: one tqdm bar per dataset showing epoch progress + live stats.
    File output:    per-epoch loss/acc detail at DEBUG level.
    """
    model     = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0008)
 
    # torch.amp.GradScaler — replaces deprecated torch.cuda.amp.GradScaler
    scaler = torch.amp.GradScaler(enabled=_USE_AMP)
 
    best_val_acc = -1.0   # ensures epoch 1 always saves; handles all-zero val_acc edge case
    base_name    = os.path.splitext(dataset_name)[0]
    best_path    = os.path.join(models_dir, f"best_{model_type}_{base_name}.pth")
 
    logger.debug(f"checkpoint path : {best_path}")
    logger.debug(f"AMP             : {_USE_AMP}")
 
    # One tqdm bar for 30 epochs; postfix updates each epoch
    epoch_bar = tqdm(
        range(30),
        desc=f"  training",
        unit="ep",
        leave=False,
        file=sys.stdout,
        dynamic_ncols=True,
    )
 
    for epoch in epoch_bar:
        t0 = time.time()
 
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss, n_batches = 0.0, 0
        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=_USE_AMP):
                loss = criterion(model(Xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            n_batches    += 1
        avg_train_loss = running_loss / max(n_batches, 1)
 
        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_preds, val_trues, val_loss_sum = [], [], 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb   = Xb.to(DEVICE, non_blocking=True)
                yb_d = yb.to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=_USE_AMP):
                    out   = model(Xb)
                    vloss = criterion(out, yb_d)
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_trues.extend(yb.numpy())
                val_loss_sum += vloss.item()
 
        val_acc  = accuracy_score(val_trues, val_preds)
        val_loss = val_loss_sum / max(len(val_loader), 1)
        elapsed  = time.time() - t0
 
        # ── Update the bar postfix (visible on console) ────────────────────
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_weights = model.state_dict().copy()
            torch.save(model.state_dict(), best_path)
            marker = "✓"
            logger.debug(f"  ↑ new best checkpoint saved (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            marker = f" ({patience_counter})"
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}/30 (patience={patience})")
                if best_weights is not None:
                    model.load_state_dict(best_weights)
                epoch_bar.close()
                break
        
        epoch_bar.set_postfix(
            train_loss=f"{avg_train_loss:.3f}",
            val_loss=f"{val_loss:.3f}",
            val_acc=f"{val_acc:.4f}",
            best=f"{best_val_acc:.4f}{marker}",
        )
        
        # ── Full per-epoch detail → log file only (DEBUG) ─────────────────
        logger.debug(
            f"ep {epoch+1:02d}/30 | "
            f"train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | patience={patience_counter}/{patience} | time={elapsed:.1f}s"
        )
    epoch_bar.close()
 
    # ── GPU cleanup ───────────────────────────────────────────────────────────
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
 
    return best_path, best_val_acc
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# GPU MEMORY HELPER  (debug-level only — never touches console)
# ═══════════════════════════════════════════════════════════════════════════════
def _log_gpu_mem(logger: logging.Logger, tag: str = ""):
    if DEVICE.type != "cuda":
        return
    alloc  = torch.cuda.memory_allocated() / 1024**2
    reserv = torch.cuda.memory_reserved()  / 1024**2
    free   = torch.cuda.mem_get_info()[0]  / 1024**2
    total  = torch.cuda.get_device_properties(0).total_memory / 1024**2
    logger.debug(
        f"GPU mem {tag}: alloc={alloc:.0f}MB  reserv={reserv:.0f}MB  "
        f"free={free:.0f}MB / {total:.0f}MB"
    )
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# MAIN COMBO PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def train_for_method_and_model(
    method_name: str,
    datasets_list: list,
    model_type: str = "efficientnet",
):
    """
    Trains + evaluates every dataset for one (method × model) combo.
    Outputs → results/<model_key>/<method_key>/
    """
    dirs, model_key, method_key = _get_dirs(model_type, method_name)
    logger = _combo_logger(model_key, method_key)
 
    combo_label = f"{model_type.upper()} + {method_name.upper()}"
 
    # ── Combo banner (console) ────────────────────────────────────────────────
    logger.info(f"{'─'*60}")
    logger.info(f"  Combo : {combo_label}  |  {len(datasets_list)} datasets")
    logger.info(f"{'─'*60}")
 
    # ── Full detail → log file ────────────────────────────────────────────────
    logger.debug(f"metrics  : {dirs['metrics']}")
    logger.debug(f"models   : {dirs['models']}")
    logger.debug(f"logs     : {dirs['logs']}")
    logger.debug(f"device   : {DEVICE}  AMP={_USE_AMP}")
    _log_gpu_mem(logger, "combo start")
 
    train_tf = efficientnet_train_transform if model_type == "efficientnet" else nctd_transform
    val_tf   = efficientnet_val_transform   if model_type == "efficientnet" else nctd_transform
 
    combo_summary = []
 
    for ds_idx, dataset_filename in enumerate(datasets_list, 1):
        ds_stem = os.path.splitext(dataset_filename)[0]
 
        # ── Dataset header (console) ──────────────────────────────────────────
        logger.info(
            f"  [{ds_idx:02d}/{len(datasets_list):02d}]  "
            f"{ds_stem:<35}  model={model_type}  method={method_name}"
        )
        t_ds = time.time()
 
        # ── Load ──────────────────────────────────────────────────────────────
        try:
            X, y = load_2d_datasets(dataset_filename, method_name, DATA_DIR, logger)
        except FileNotFoundError as exc:
            logger.error(f"         SKIPPED — {exc}")
            continue
 
        num_classes = int(y.unique().numel())
        num_samples = int(len(y))
        logger.debug(f"samples={num_samples}  classes={num_classes}")
 
        # ── Stratified splits: 70% train / 10% val / 20% test ────────────────
        train_idx, test_idx = train_test_split(
            range(num_samples), test_size=0.2,
            stratify=y.cpu().numpy(), random_state=42,
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.125,
            stratify=y[train_idx].cpu().numpy(), random_state=42,
        )
        logger.debug(
            f"split: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}"
        )
 
        train_ds = Tabular2ImageDataset(X, y, model_type, train_tf, indices=train_idx)
        val_ds   = Tabular2ImageDataset(X, y, model_type, val_tf, indices=val_idx)
        test_ds  = Tabular2ImageDataset(X, y, model_type, val_tf, indices=test_idx)
 
        # num_workers=0: data is already loaded into RAM as tensors — no disk
        # I/O means workers add zero throughput benefit.  More importantly,
        # num_workers>0 on Windows (spawn) re-imports this module in every
        # worker process, which caused the repeated device-info log lines.
        _pin = DEVICE.type == "cuda"   # pin_memory still helps GPU transfers
 
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                                  num_workers=0, pin_memory=_pin)
        val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                                  num_workers=0, pin_memory=_pin)
        test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False,
                                  num_workers=0, pin_memory=_pin)
 
        _log_gpu_mem(logger, "before train")
 
        # ── Train ─────────────────────────────────────────────────────────────
        model    = get_model(model_type, num_classes)
        n_params = sum(p.numel() for p in model.parameters())
        logger.debug(f"model params: {n_params:,}")
 
        best_path, best_val_acc = train_uniform_model(
            model, train_loader, val_loader,
            dataset_name=dataset_filename,
            model_type=model_type,
            models_dir=dirs["models"],
            logger=logger,
        )
 
        # ── Reload best checkpoint ────────────────────────────────────────────
        model = get_model(model_type, num_classes)
        model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE)
        logger.debug(f"best ckpt loaded: {best_path}")
 
        # ── Test evaluation ───────────────────────────────────────────────────
        metrics, preds, trues = evaluate_model(model, test_loader, logger)
 
        # ── Result line (console) — NaN-safe formatting ───────────────────────
        auc_str = f"{metrics['roc_auc']:.4f}" if np.isfinite(metrics['roc_auc']) else " N/A"
        logger.info(
            f"         → "
            f"acc={metrics['accuracy']:.4f}  "
            f"f1={metrics['f1']:.4f}  "
            f"prec={metrics['precision']:.4f}  "
            f"rec={metrics['recall']:.4f}  "
            f"auc={auc_str}  "
            f"mcc={metrics['mcc']:.4f}  "
            f"[{time.time()-t_ds:.0f}s]"
        )
 
        combo_summary.append({
            "dataset":   ds_stem,
            "accuracy":  metrics["accuracy"],
            "f1":        metrics["f1"],
            "precision": metrics["precision"],
            "recall":    metrics["recall"],
            "roc_auc":   metrics["roc_auc"],
            "mcc":       metrics["mcc"],
        })
 
        # ── Persist results ───────────────────────────────────────────────────
        save_stem  = f"{method_name}_{model_type}_{ds_stem}"
        final_path = os.path.join(dirs["models"], f"{save_stem}_final.pth")
        json_path  = os.path.join(dirs["metrics"], f"{save_stem}_metrics.json")
        cm_path    = os.path.join(dirs["metrics"], f"{save_stem}_cm.png")
 
        torch.save(model.state_dict(), final_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
 
        # Binary classification → always a 2×2 confusion matrix
        plt.figure(figsize=(4, 3))
        sns.heatmap(confusion_matrix(trues, preds),
                    annot=True, fmt="d", cmap="Blues",
                    linewidths=0.4, linecolor="white")
        plt.title(f"Confusion Matrix — {save_stem}", fontsize=9)
        plt.ylabel("True label");  plt.xlabel("Predicted label")
        plt.tight_layout();  plt.savefig(cm_path, dpi=120);  plt.close()
 
        logger.debug(f"saved: {final_path}")
        logger.debug(f"saved: {json_path}")
        logger.debug(f"saved: {cm_path}")
 
        # ── GPU cleanup per dataset ───────────────────────────────────────────
        del model
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        _log_gpu_mem(logger, "after cleanup")
 
    # ── End-of-combo summary (console) ───────────────────────────────────────
    if combo_summary:
        logger.info(f"{'─'*60}")
        logger.info(f"  Summary: {combo_label}")
        logger.info(
            f"  {'Dataset':<35} {'Acc':>6} {'F1':>6} "
            f"{'Prec':>6} {'Rec':>6} {'AUC':>6} {'MCC':>7}"
        )
        logger.info(f"  {'─'*76}")
        for r in combo_summary:
            auc_s = f"{r['roc_auc']:>6.4f}" if np.isfinite(r['roc_auc']) else "   N/A"
            logger.info(
                f"  {r['dataset']:<35} "
                f"{r['accuracy']:>6.4f} {r['f1']:>6.4f} "
                f"{r['precision']:>6.4f} {r['recall']:>6.4f} "
                f"{auc_s}  {r['mcc']:>7.4f}"
            )
        acc_vals = [r["accuracy"] for r in combo_summary]
        logger.info(f"  {'─'*76}")
        logger.info(
            f"  {'MEAN':<35} {np.mean(acc_vals):>6.4f}  "
            f"(std={np.std(acc_vals):.4f}  n={len(acc_vals)})"
        )
    logger.info(f"{'─'*60}")
    logger.info("")
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="1D→2D tabular image classification pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--method", choices=["NCTD", "ours"], help="Conversion method")
    parser.add_argument("--model",  choices=["efficientnet", "nctd_cnn"], help="Backbone")
    parser.add_argument("--all",    action="store_true", help="Run all 4 combos")
    parser.add_argument("--dataset-root", type=str, default=DATA_DIR, help="Set primary datasets folder name")
    parser.add_argument("--lazy-loading", action="store_true", help="Use lazy chunked generation logic")
    parser.add_argument("--full-loading", action="store_true", help="Use full chunked in-memory logic")
    args = parser.parse_args()
    
    # Overwrite global DATA_DIR safely for module execution
    DATA_DIR = args.dataset_root
 
    COMBOS = [
        ("NCTD", "nctd_cnn"),
        ("NCTD", "efficientnet"),
        ("ours", "nctd_cnn"),
        ("ours", "efficientnet"),
    ]
 
    master = _master_logger()
    _log_device_info()
    master.info(f"Run: {_RUN_TS}  |  results → {os.path.abspath(RESULTS_DIR)}")
 
    if args.all:
        for method, model in COMBOS:
            t0 = time.time()
            train_for_method_and_model(method, DATASETS_LIST, model_type=model)
            master.debug(f"combo {method}+{model} finished in {time.time()-t0:.1f}s")
    elif args.method and args.model:
        train_for_method_and_model(args.method, DATASETS_LIST, model_type=args.model)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python train_pipeline.py --all")
        print("  python train_pipeline.py --method NCTD --model efficientnet")
        print("  python train_pipeline.py --method ours --model nctd_cnn")
 
    master.info("Done.")