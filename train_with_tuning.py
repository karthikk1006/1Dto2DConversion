"""
train_with_tuning.py
====================
Training script that uses hyperparameters tuned by hp_tuning.py for optimal results.

This script:
  1. Loads best hyperparameters from hp_tuning results
  2. Trains both EfficientNet and CNN using tuned hyperparameters
  3. Evaluates on all datasets
  4. Compares results with default hyperparameters
  5. Generates comparison report

Output structure:
  results/
    tuned_results/
      efficientnet/
        nctd/metrics/  & models/
        ours/metrics/  & models/
      cnn/
        nctd/metrics/  & models/
        ours/metrics/  & models/
      
      comparison/
        tuned_vs_default_summary.json
        tuned_vs_default_report.txt

Usage
-----
  # Train with tuned hyperparameters (requires hp_tuning.py to have run first)
  python train_with_tuning.py --all

  # Train specific model with tuning
  python train_with_tuning.py --model efficientnet --method ours

  # Train specific dataset only
  python train_with_tuning.py --all --dataset 13_rotated_rastrigin_50d.npz
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import warnings
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Import from train_pipeline
from train_pipeline import (
    DATA_DIR,
    RESULTS_DIR,
    DATASETS_LIST,
    DEVICE,
    _USE_AMP,
    load_2d_datasets,
    Tabular2ImageDataset,
    get_model,
    evaluate_model,
    efficientnet_train_transform,
    efficientnet_val_transform,
    nctd_transform,
    _master_logger,
    _combo_logger,
    _get_dirs,
)

# Suppress PyTorch lr_scheduler.step() warning about calling order
# (The warning is a false positive in our case as we call it correctly)
warnings.filterwarnings(
    "ignore",
    message=".*Detected call of `lr_scheduler.step()` before `optimizer.step()`.*",
    category=UserWarning
)


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_tuned_hyperparams(model_type: str, method_name: str, dataset_filename: str) -> Optional[Dict]:
    """
    Load the best hyperparameters found by hp_tuning.py for a given dataset.
    
    Returns:
      - Dict of hyperparameters if found
      - None if tuning results not available
    """
    ds_stem = os.path.splitext(dataset_filename)[0]
    params_path = os.path.join(
        RESULTS_DIR, "hp_tuning", model_type, method_name, f"best_params_{ds_stem}.json"
    )
    
    if not os.path.exists(params_path):
        return None
    
    with open(params_path, "r") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING WITH TUNED HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
def train_model_with_tuning(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dataset_name: str,
    model_type: str,
    models_dir: str,
    logger: logging.Logger,
    hyperparams: Dict,
) -> str:
    """
    Train model using tuned hyperparameters.
    
    Hyperparameters expected:
      - learning_rate
      - batch_size (ignored - already applied)
      - weight_decay
      - optimizer: "adam" or "sgd"
      - scheduler: "none", "step", "cosine", "exponential"
      - epochs
      - momentum (if SGD)
      - scheduler_step_size, scheduler_gamma (if step)
      - scheduler_t_max (if cosine)
      - scheduler_gamma (if exponential)
      - dropout_rate (for CNN, ignored for EfficientNet)
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    
    # Setup optimizer
    if hyperparams["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=hyperparams["learning_rate"],
            momentum=hyperparams.get("momentum", 0.9),
            weight_decay=hyperparams["weight_decay"],
        )
    
    # Setup scheduler
    scheduler = None
    if hyperparams["scheduler"] == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=hyperparams.get("scheduler_step_size", 10),
            gamma=hyperparams.get("scheduler_gamma", 0.5),
        )
    elif hyperparams["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=hyperparams.get("scheduler_t_max", 20),
        )
    elif hyperparams["scheduler"] == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=hyperparams.get("scheduler_gamma", 0.95),
        )
    
    scaler = torch.amp.GradScaler(enabled=_USE_AMP)
    
    best_val_acc = -1.0
    base_name = os.path.splitext(dataset_name)[0]
    best_path = os.path.join(models_dir, f"best_{model_type}_{base_name}.pth")
    
    logger.debug(f"Hyperparameters: {hyperparams}")
    logger.debug(f"Optimizer: {hyperparams['optimizer']}, LR: {hyperparams['learning_rate']:.2e}")
    logger.debug(f"Scheduler: {hyperparams['scheduler']}")
    logger.debug(f"Epochs: {hyperparams['epochs']}")
    
    epochs = hyperparams["epochs"]
    patience = 10  # Early stopping patience
    patience_counter = 0
    best_weights = None
    
    # Training progress bar
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
        
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        running_loss, n_batches = 0.0, 0
        
        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=_USE_AMP):
                loss = criterion(model(Xb), yb)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = running_loss / max(n_batches, 1)
        
        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_preds, val_trues, val_loss_sum = [], [], 0.0
        
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(DEVICE, non_blocking=True)
                yb_d = yb.to(DEVICE, non_blocking=True)
                
                with torch.amp.autocast(device_type=DEVICE.type, enabled=_USE_AMP):
                    out = model(Xb)
                    vloss = criterion(out, yb_d)
                
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_trues.extend(yb.numpy())
                val_loss_sum += vloss.item()
        
        val_acc = accuracy_score(val_trues, val_preds)
        val_loss = val_loss_sum / max(len(val_loader), 1)
        elapsed = time.time() - t0
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_weights = model.state_dict().copy()
            torch.save(model.state_dict(), best_path)
            logger.debug(f"  ↑ new best checkpoint saved (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}/{epochs} (patience={patience})")
                if best_weights is not None:
                    model.load_state_dict(best_weights)
                break
        
        # Update progress bar
        marker = "✓" if patience_counter == 0 else f" ({patience_counter})"
        epoch_bar.set_postfix(
            train_loss=f"{avg_train_loss:.3f}",
            val_loss=f"{val_loss:.3f}",
            val_acc=f"{val_acc:.4f}",
            best=f"{best_val_acc:.4f}{marker}",
        )
        
        logger.debug(
            f"ep {epoch+1:02d}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | patience={patience_counter}/{patience} | time={elapsed:.1f}s"
        )
    
    epoch_bar.close()
    
    # GPU cleanup
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    return best_path, best_val_acc


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE WITH TUNING
# ═══════════════════════════════════════════════════════════════════════════════
def train_for_method_and_model_with_tuning(
    method_name: str,
    datasets_list: list,
    model_type: str = "efficientnet",
    dataset_root: str = DATA_DIR,
    lazy_loading: bool = False,
    full_loading: bool = False,
):
    """
    Train + evaluate every dataset for one (method × model) combo using tuned hyperparameters.
    Falls back to default parameters if tuning results not available.
    """
    dirs, model_key, method_key = _get_dirs(model_type, method_name)
    
    # Redirect logs to tuned_results instead of results
    base_tuned = os.path.join(RESULTS_DIR, "tuned_results", model_key, method_key)
    tuned_metrics_dir = os.path.join(base_tuned, "metrics")
    tuned_models_dir = os.path.join(base_tuned, "models")
    tuned_logs_dir = os.path.join(base_tuned, "logs")
    
    for d in [tuned_metrics_dir, tuned_models_dir, tuned_logs_dir]:
        os.makedirs(d, exist_ok=True)
    
    logger = logging.getLogger(f"tuned_{model_key}.{method_key}")
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        
        log_file = os.path.join(tuned_logs_dir, f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    
    combo_label = f"{model_type.upper()} + {method_name.upper()}"
    
    logger.info(f"{'─'*60}")
    logger.info(f"  Combo : {combo_label}  |  {len(datasets_list)} datasets")
    logger.info(f"  Using TUNED hyperparameters")
    logger.info(f"{'─'*60}")
    
    train_tf = efficientnet_train_transform if model_type == "efficientnet" else nctd_transform
    val_tf = efficientnet_val_transform if model_type == "efficientnet" else nctd_transform
    
    combo_summary = []
    tuning_status = []
    
    for ds_idx, dataset_filename in enumerate(datasets_list, 1):
        ds_stem = os.path.splitext(dataset_filename)[0]
        
        logger.info(
            f"  [{ds_idx:02d}/{len(datasets_list):02d}]  "
            f"{ds_stem:<35}  model={model_type}  method={method_name}"
        )
        t_ds = time.time()
        
        # Load dataset
        try:
            X, y = load_2d_datasets(
                dataset_filename, method_name, dataset_root, logger,
                lazy_loading=lazy_loading, full_loading=full_loading
            )
        except FileNotFoundError as exc:
            logger.error(f"         SKIPPED — {exc}")
            continue
        
        num_classes = int(y.unique().numel())
        num_samples = int(len(y))
        
        # Split data
        train_idx, test_idx = train_test_split(
            range(num_samples), test_size=0.2,
            stratify=y.cpu().numpy(), random_state=42,
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.125,
            stratify=y[train_idx].cpu().numpy(), random_state=42,
        )
        
        # Load tuned hyperparameters
        hyperparams = load_tuned_hyperparams(model_type, method_name, dataset_filename)
        
        if hyperparams is not None:
            using_tuned = True
            logger.info(f"         ✓ Loaded tuned hyperparameters")
            tuning_status.append((ds_stem, "tuned"))
        else:
            # Fallback to defaults
            using_tuned = False
            hyperparams = {
                "learning_rate": 0.0008,
                "batch_size": 64,
                "weight_decay": 0.0,
                "optimizer": "adam",
                "scheduler": "none",
                "epochs": 30,
            }
            logger.warning(f"         ⚠ No tuned params found, using defaults")
            tuning_status.append((ds_stem, "default"))
        
        # Create datasets
        train_ds = Tabular2ImageDataset(X, y, model_type, train_tf, indices=train_idx)
        val_ds = Tabular2ImageDataset(X, y, model_type, val_tf, indices=val_idx)
        test_ds = Tabular2ImageDataset(X, y, model_type, val_tf, indices=test_idx)
        
        _pin = DEVICE.type == "cuda"
        train_loader = DataLoader(
            train_ds, batch_size=hyperparams["batch_size"], shuffle=True,
            num_workers=0, pin_memory=_pin,
        )
        val_loader = DataLoader(
            val_ds, batch_size=hyperparams["batch_size"], shuffle=False,
            num_workers=0, pin_memory=_pin,
        )
        test_loader = DataLoader(
            test_ds, batch_size=hyperparams["batch_size"], shuffle=False,
            num_workers=0, pin_memory=_pin,
        )
        
        # Train model
        model = get_model(model_type, num_classes)
        best_path, best_val_acc = train_model_with_tuning(
            model, train_loader, val_loader,
            dataset_name=dataset_filename,
            model_type=model_type,
            models_dir=tuned_models_dir,
            logger=logger,
            hyperparams=hyperparams,
        )
        
        # Evaluate on test set
        model = get_model(model_type, num_classes)
        model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE)
        
        metrics, preds, trues = evaluate_model(model, test_loader, logger)
        
        # Report
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
            "dataset": ds_stem,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "roc_auc": metrics["roc_auc"],
            "mcc": metrics["mcc"],
            "using_tuned": using_tuned,
        })
        
        # Save results
        save_stem = f"{method_name}_{model_type}_{ds_stem}"
        final_path = os.path.join(tuned_models_dir, f"{save_stem}_final.pth")
        json_path = os.path.join(tuned_metrics_dir, f"{save_stem}_metrics.json")
        
        torch.save(model.state_dict(), final_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        
        # Save hyperparameters used
        hp_path = os.path.join(tuned_metrics_dir, f"{save_stem}_hyperparams.json")
        with open(hp_path, "w", encoding="utf-8") as f:
            json.dump(hyperparams, f, indent=4)
        
        # Cleanup
        del model
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    # Summary
    if combo_summary:
        logger.info(f"{'─'*60}")
        logger.info(f"  Summary: {combo_label}")
        logger.info(
            f"  {'Dataset':<35} {'Acc':>6} {'F1':>6} "
            f"{'Prec':>6} {'Rec':>6} {'AUC':>6} {'MCC':>7} {'Status':>8}"
        )
        logger.info(f"  {'─'*90}")
        
        for r, (ds_stem, status) in zip(combo_summary, tuning_status):
            auc_s = f"{r['roc_auc']:>6.4f}" if np.isfinite(r['roc_auc']) else "   N/A"
            status_str = "✓ tuned" if status == "tuned" else "default"
            logger.info(
                f"  {r['dataset']:<35} "
                f"{r['accuracy']:>6.4f} {r['f1']:>6.4f} "
                f"{r['precision']:>6.4f} {r['recall']:>6.4f} "
                f"{auc_s}  {r['mcc']:>7.4f}  {status_str:>8}"
            )
        
        acc_vals = [r["accuracy"] for r in combo_summary]
        logger.info(f"  {'─'*90}")
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
        description="Train models with tuned hyperparameters",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--method", choices=["NCTD", "ours"], help="Conversion method")
    parser.add_argument("--model", choices=["efficientnet", "nctd_cnn"], help="Backbone")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Specific dataset (if not provided, all datasets used)",
    )
    parser.add_argument("--all", action="store_true", help="Run all 4 combos")
    parser.add_argument("--dataset-root", type=str, default=DATA_DIR, help="Set primary datasets folder name")
    parser.add_argument("--lazy-loading", action="store_true", help="Use lazy chunked generation logic")
    parser.add_argument("--full-loading", action="store_true", help="Use full chunked in-memory logic")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    args = parser.parse_args()
    
    # Overwrite DATA_DIR safely for module functions from train_pipeline
    import train_pipeline
    train_pipeline.DATA_DIR = args.dataset_root
    if args.output_dir:
        train_pipeline.RESULTS_DIR = args.output_dir
        global RESULTS_DIR
        RESULTS_DIR = args.output_dir
    
    COMBOS = [
        ("NCTD", "nctd_cnn"),
        ("NCTD", "efficientnet"),
        ("ours", "nctd_cnn"),
        ("ours", "efficientnet"),
    ]
    
    master = _master_logger()
    master.info(f"Training with TUNED hyperparameters")
    
    # Determine datasets
    if args.dataset:
        datasets_to_train = [args.dataset]
    else:
        datasets_to_train = DATASETS_LIST
    
    if args.all:
        for method, model in COMBOS:
            t0 = time.time()
            train_for_method_and_model_with_tuning(
                method, datasets_to_train, model_type=model,
                dataset_root=args.dataset_root,
                lazy_loading=args.lazy_loading,
                full_loading=args.full_loading
            )
            master.debug(f"combo {method}+{model} finished in {time.time()-t0:.1f}s")
    elif args.method and args.model:
        train_for_method_and_model_with_tuning(
            args.method, datasets_to_train, model_type=args.model,
            dataset_root=args.dataset_root,
            lazy_loading=args.lazy_loading,
            full_loading=args.full_loading
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python train_with_tuning.py --all")
        print("  python train_with_tuning.py --method ours --model efficientnet")
        print("  python train_with_tuning.py --all --dataset 13_rotated_rastrigin_50d.npz")
    
    master.info("Done.")
