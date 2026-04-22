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
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import core utilities from the existing pipeline
from train_pipeline import (
    load_2d_datasets,
    Tabular2ImageDataset,
    evaluate_model,
    nctd_transform,
    DEVICE,
    _USE_AMP,
)

def log_gpu_usage(logger: logging.Logger, tag: str = ""):
    """Explicitly log GPU usage at INFO level."""
    if DEVICE.type == "cuda":
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserv = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"  [GPU] {tag} - Allocated: {alloc:.0f}MB, Reserved: {reserv:.0f}MB")

# ═══════════════════════════════════════════════════════════════════════════════
# STANDARD CNN ARCHITECTURE (WITH 0.3 DROPOUT)
# ═══════════════════════════════════════════════════════════════════════════════
class StandardCNN(nn.Module):
    """
    CNN architecture based on the NCTD 2025 paper, 
    enhanced with 0.3 Dropout as per the Improvement Analysis.
    """
    def __init__(self, num_classes: int, in_channels: int = 1, dropout_rate: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),           nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),           nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),           nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier  = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 128), 
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x.view(x.size(0), -1))

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING ENGINE (STANDARD PARAMS)
# ═══════════════════════════════════════════════════════════════════════════════
def train_standard_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dataset_name: str,
    models_dir: str,
    logger: logging.Logger,
):
    """
    Standardized training loop:
    - Optimizer: Adam (LR=0.001, WD=1e-4)
    - Scheduler: CosineAnnealing (T_max=40)
    - Epochs: 40
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    scaler = torch.amp.GradScaler(enabled=_USE_AMP)

    # Log initial GPU state
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        log_gpu_usage(logger, "Before Training")
    else:
        logger.warning("Training on CPU. Performance will be slow.")

    best_val_acc = -1.0
    base_name = os.path.splitext(dataset_name)[0]
    best_path = os.path.join(models_dir, f"standard_best_{base_name}.pth")
    
    epochs = 40
    patience = 10
    patience_counter = 0
    best_weights = None

    epoch_bar = tqdm(
        range(epochs),
        desc=f"  training",
        unit="ep",
        leave=False,
        file=sys.stdout,
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        t0 = time.time()
        
        # Train
        model.train()
        running_loss, n_batches = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=_USE_AMP):
                loss = criterion(model(Xb), yb)
            
            if torch.isnan(loss):
                logger.warning(f"  [DIVERGENCE] NaN loss detected at epoch {epoch+1}. Stopping training for {dataset_name}.")
                if best_weights is not None:
                    model.load_state_dict(best_weights)
                epoch_bar.close()
                return best_path, best_val_acc

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = running_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_preds, val_trues, val_loss_sum = [], [], 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb_d = Xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=_USE_AMP):
                    out = model(Xb)
                    vloss = criterion(out, yb_d)
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_trues.extend(yb.numpy())
                val_loss_sum += vloss.item()
        
        val_acc = accuracy_score(val_trues, val_preds)
        val_loss = val_loss_sum / max(len(val_loader), 1)
        
        # Scheduler Step
        scheduler.step()

        # Best model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_weights = model.state_dict().copy()
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                if best_weights is not None:
                    model.load_state_dict(best_weights)
                break

        # Log GPU usage every 10 epochs
        if (epoch + 1) % 10 == 0:
            log_gpu_usage(logger, f"Epoch {epoch+1}")

        epoch_bar.set_postfix(
            loss=f"{avg_train_loss:.3f}",
            val_acc=f"{val_acc:.4f}",
            best=f"{best_val_acc:.4f}"
        )
    
    epoch_bar.close()
    
    if DEVICE.type == "cuda":
        log_gpu_usage(logger, "After Training")
    
    return best_path, best_val_acc

from sklearn.metrics import accuracy_score

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DISCOVERY & EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Train Standardized CNN on all datasets")
    parser.add_argument("--dataset-root", type=str, required=True, help="Root folder containing 'ours' and 'NCTD' folders")
    parser.add_argument("--output-dir", type=str, default="standardized_results", help="Where to save results")
    parser.add_argument("--lazy-loading", action="store_true", help="Use lazy loading for directories")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup Loggers
    log_file = os.path.join(args.output_dir, f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s",
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger("standard_train")

    methods = ["ours", "NCTD"]
    
    for method in methods:
        method_path = os.path.join(args.dataset_root, method)
        if not os.path.exists(method_path):
            logger.warning(f"Method path {method_path} does not exist. Skipping.")
            continue
        
        # Discover datasets (.npz files or directories)
        items = os.listdir(method_path)
        datasets = []
        for item in items:
            full_item_path = os.path.join(method_path, item)
            if item.endswith(".npz"):
                datasets.append(item)
            elif os.path.isdir(full_item_path):
                # For directories, the code expects the name without extension usually
                datasets.append(item)

        logger.info(f"Starting standardized training for method: {method.upper()} ({len(datasets)} datasets)")
        
        # Results subdirs
        metrics_dir = os.path.join(args.output_dir, method, "metrics")
        models_dir = os.path.join(args.output_dir, method, "models")
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        for ds_name in datasets:
            logger.info(f"Processing: {ds_name}")
            try:
                # Load data
                is_dir = os.path.isdir(os.path.join(method_path, ds_name))
                
                # Determine loading strategy
                # For directories: use lazy_loading if requested, otherwise full_loading
                # For files: flags are ignored by load_2d_datasets logic
                lazy_flag = args.lazy_loading if is_dir else False
                full_flag = (not args.lazy_loading) if is_dir else False
                
                X, y = load_2d_datasets(
                    ds_name, method, args.dataset_root, logger,
                    lazy_loading=lazy_flag,
                    full_loading=full_flag
                )
                
                num_classes = int(y.unique().numel())
                num_samples = int(len(y))
                
                # Split 70/10/20
                train_idx, test_idx = train_test_split(range(num_samples), test_size=0.2, stratify=y.cpu().numpy(), random_state=42)
                train_idx, val_idx = train_test_split(train_idx, test_size=0.125, stratify=y[train_idx].cpu().numpy(), random_state=42)
                
                train_ds = Tabular2ImageDataset(X, y, "nctd_cnn", nctd_transform, indices=train_idx)
                val_ds = Tabular2ImageDataset(X, y, "nctd_cnn", nctd_transform, indices=val_idx)
                test_ds = Tabular2ImageDataset(X, y, "nctd_cnn", nctd_transform, indices=test_idx)
                
                train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=(DEVICE.type == "cuda"))
                val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
                test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
                
                # Model Initialization
                model = StandardCNN(num_classes=num_classes)
                
                # Train
                best_path, _ = train_standard_model(model, train_loader, val_loader, ds_name, models_dir, logger)
                
                # Evaluate
                model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
                metrics, _, _ = evaluate_model(model, test_loader, logger)
                
                # Save metrics
                ds_stem = os.path.splitext(ds_name)[0]
                with open(os.path.join(metrics_dir, f"{ds_stem}_metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=4)
                
                logger.info(f"Done: {ds_name} | Acc: {metrics['accuracy']:.4f} | MCC: {metrics['mcc']:.4f}")

            except Exception as e:
                logger.error(f"Failed to process {ds_name}: {str(e)}")
                continue

if __name__ == "__main__":
    main()
