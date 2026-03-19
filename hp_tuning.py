"""
hp_tuning.py
============
Comprehensive hyperparameter tuning using Optuna for both EfficientNet and CNN models.
Tunes across all datasets to find optimal hyperparameters for each (model, dataset) pair.

Features:
  • Optuna-based hyperparameter search
  • Tracks best hyperparameters for each model × dataset combination
  • Supports early stopping & pruning
  • Parallel trial execution
  • Saves tuning results & best hyperparameters to JSON
  • Generates tuning reports and visualizations

Output structure:
  results/
    hp_tuning/
      efficientnet/
        best_params_<dataset>.json      ← best hyperparameters
        tuning_history_<dataset>.json   ← full Optuna study data
      cnn/
        best_params_<dataset>.json
        tuning_history_<dataset>.json
      reports/
        hp_tuning_summary.json          ← summary across all
        hp_tuning_report.txt            ← human-readable results

Usage
-----
  # Tune both models on all datasets (parallel, 100 trials each)
  python hp_tuning.py --all --n-trials 100 --n-jobs 4

  # Tune only EfficientNet
  python hp_tuning.py --model efficientnet --n-trials 50

  # Tune only a specific dataset
  python hp_tuning.py --all --dataset 13_rotated_rastrigin_50d.npz

  # Quick test with few trials
  python hp_tuning.py --all --n-trials 10
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import optuna
from optuna.trial import Trial
from optuna.study import Study
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Import from train_pipeline
from train_pipeline import (
    DATA_DIR,
    RESULTS_DIR,
    DATASETS_LIST,
    DEVICE,
    _USE_AMP,
    load_2d_datasets,
    Tabular2ImageDataset,
    NCTD_CNN,
    get_model,
    evaluate_model,
    efficientnet_train_transform,
    efficientnet_val_transform,
    nctd_transform,
    _master_logger,
)

# Suppress PyTorch lr_scheduler.step() warning about calling order
# (The warning is a false positive in our case as we call it correctly)
warnings.filterwarnings(
    "ignore",
    message=".*Detected call of `lr_scheduler.step()` before `optimizer.step()`.*",
    category=UserWarning
)

# Enable CuDNN benchmark for faster training on fixed-size inputs
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════
_LOG_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_RUN_TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_tuning_logger(model_type: str, dataset_stem: str, method_name: str) -> logging.Logger:
    """Create logger for hyperparameter tuning."""
    log_file = os.path.join(
        RESULTS_DIR, "hp_tuning", model_type, method_name, "logs",
        f"tuning_{dataset_stem}_{_RUN_TS}.log"
    )
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(f"hp_tuning.{model_type}.{method_name}.{dataset_stem}")
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    fmt = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    return logger


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════
def suggest_hyperparams(trial: Trial, model_type: str) -> Dict:
    """
    Suggest hyperparameters for the given model type.
    
    EfficientNet tunes:
      - learning_rate
      - batch_size
      - weight_decay
      - optimizer (Adam / SGD)
      - scheduler (none / step / cosine)
      - epochs
      
    CNN tunes:
      - learning_rate
      - batch_size
      - weight_decay
      - optimizer
      - scheduler
      - epochs
      - Additional CNN-specific: dropout rate
    """
    
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
        "scheduler": trial.suggest_categorical(
            "scheduler", ["none", "step", "cosine", "exponential"]
        ),
        "epochs": trial.suggest_int("epochs", 20, 50),
    }
    
    # Optimizer-specific parameters
    if params["optimizer"] == "sgd":
        params["momentum"] = trial.suggest_float("momentum", 0.8, 0.99)
    
    # Scheduler-specific parameters
    if params["scheduler"] == "step":
        params["scheduler_step_size"] = trial.suggest_int("scheduler_step_size", 5, 15)
        params["scheduler_gamma"] = trial.suggest_float("scheduler_gamma", 0.1, 0.9)
    elif params["scheduler"] == "cosine":
        params["scheduler_t_max"] = trial.suggest_int("scheduler_t_max", 10, 40)
    elif params["scheduler"] == "exponential":
        params["scheduler_gamma"] = trial.suggest_float("scheduler_gamma", 0.95, 0.99)
    
    # Model-specific parameters
    if model_type == "nctd_cnn":
        params["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.5)
    
    return params


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING WITH CUSTOM HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
def train_with_hyperparams(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    params: Dict,
    logger: logging.Logger,
) -> Tuple[float, float]:
    """
    Train model with given hyperparameters.
    Returns (best_val_acc, final_val_acc).
    """
    try:
        # Check GPU memory before training
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        model = model.to(DEVICE)
        criterion = nn.CrossEntropyLoss().to(DEVICE)
        
        # Setup optimizer
        if params["optimizer"] == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
            )
        else:  # sgd
            optimizer = optim.SGD(
                model.parameters(),
                lr=params["learning_rate"],
                momentum=params.get("momentum", 0.9),
                weight_decay=params["weight_decay"],
            )
        
        # Setup scheduler
        scheduler = None
        if params["scheduler"] == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=params["scheduler_step_size"],
                gamma=params["scheduler_gamma"],
            )
        elif params["scheduler"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=params["scheduler_t_max"],
            )
        elif params["scheduler"] == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=params["scheduler_gamma"],
            )
        
        scaler = torch.amp.GradScaler(enabled=_USE_AMP)
        
        best_val_acc = -1.0
        epochs = params["epochs"]
        patience = 10  # Early stopping patience
        patience_counter = 0
        best_weights = None
        
        for epoch in range(epochs):
            # ── Train ─────────────────────────────────────────────────────────
            model.train()
            running_loss = 0.0
            n_batches = 0
            
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
            
            # ── Validate ──────────────────────────────────────────────────────
            model.eval()
            val_preds, val_trues = [], []
            
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb = Xb.to(DEVICE, non_blocking=True)
                    yb_d = yb.to(DEVICE, non_blocking=True)
                    
                    with torch.amp.autocast(device_type=DEVICE.type, enabled=_USE_AMP):
                        out = model(Xb)
                    
                    val_preds.extend(out.argmax(1).cpu().numpy())
                    val_trues.extend(yb.numpy())
            
            val_acc = accuracy_score(val_trues, val_preds)
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_weights = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}/{epochs} (patience={patience})")
                    if best_weights is not None:
                        model.load_state_dict(best_weights)
                    break
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
            
            logger.debug(
                f"ep {epoch+1:02d}/{epochs} | "
                f"train_loss={running_loss/n_batches:.4f} | "
                f"val_acc={val_acc:.4f} | best={best_val_acc:.4f} | patience={patience_counter}/{patience}"
            )
        
        # GPU cleanup
        if DEVICE.type == "cuda":
            del optimizer, scaler
            if scheduler is not None:
                del scheduler
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        return best_val_acc, val_acc
    
    except RuntimeError as e:
        # Handle CUDA out of memory or other runtime errors
        if "CUDA" in str(e) or "out of memory" in str(e).lower():
            logger.warning(f"CUDA error encountered: {str(e)}")
            # Force cleanup
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# OPTUNA OBJECTIVE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
def create_objective(
    X: torch.Tensor,
    y: torch.Tensor,
    model_type: str,
    num_classes: int,
    logger: logging.Logger,
):
    """Create objective function for Optuna."""
    
    def objective(trial: Trial) -> float:
        try:
            # Ensure data is on CPU to avoid GPU memory accumulation
            X_cpu = X.cpu() if X.is_cuda else X
            y_cpu = y.cpu() if y.is_cuda else y
            
            # Suggest hyperparameters
            params = suggest_hyperparams(trial, model_type)
            
            # Stratified splits: 70% train / 10% val / 20% test
            num_samples = len(y_cpu)
            train_idx, test_idx = train_test_split(
                range(num_samples), test_size=0.2,
                stratify=y_cpu.cpu().numpy(), random_state=42,
            )
            train_idx, val_idx = train_test_split(
                train_idx, test_size=0.125,
                stratify=y_cpu[train_idx].cpu().numpy(), random_state=42,
            )
            
            # Data transforms
            train_tf = (
                efficientnet_train_transform
                if model_type == "efficientnet"
                else nctd_transform
            )
            val_tf = (
                efficientnet_val_transform
                if model_type == "efficientnet"
                else nctd_transform
            )
            
            # Create datasets and dataloaders
            train_ds = Tabular2ImageDataset(
                X_cpu[train_idx], y_cpu[train_idx], model_type, train_tf
            )
            val_ds = Tabular2ImageDataset(X_cpu[val_idx], y_cpu[val_idx], model_type, val_tf)
            
            _pin = DEVICE.type == "cuda"
            train_loader = DataLoader(
                train_ds, batch_size=params["batch_size"], shuffle=True,
                num_workers=0, pin_memory=_pin,
            )
            val_loader = DataLoader(
                val_ds, batch_size=params["batch_size"], shuffle=False,
                num_workers=0, pin_memory=_pin,
            )
            
            # Create and train model
            model = get_model(model_type, num_classes)
            best_val_acc, final_val_acc = train_with_hyperparams(
                model, train_loader, val_loader, params, logger
            )
            
            # Explicit memory cleanup per trial
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            
            # Report intermediate values for pruning
            trial.report(best_val_acc, step=0)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            logger.info(
                f"Trial {trial.number}: lr={params['learning_rate']:.4e}, "
                f"batch_size={params['batch_size']}, "
                f"epochs={params['epochs']}, val_acc={best_val_acc:.4f}"
            )
            
            return best_val_acc
        
        except optuna.exceptions.TrialPruned:
            # Re-raise pruned trials
            raise
        except Exception as e:
            # Log error and cleanup memory
            error_msg = str(e)
            logger.error(f"Trial {trial.number} failed: {error_msg}")
            
            # Force aggressive memory cleanup on errors
            try:
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
            
            # Return 0.0 for failed trials
            return 0.0
    
    return objective


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TUNING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
def tune_hyperparameters(
    model_type: str,
    dataset_filename: str,
    method_name: str = "ours",
    n_trials: int = 100,
    n_jobs: int = 1,
    allow_gpu_parallel: bool = False,
):
    """
    Tune hyperparameters for a specific (model_type, dataset) pair using Optuna.
    """
    # Force n_jobs=1 for CUDA to avoid memory conflicts between parallel trials
    if DEVICE.type == "cuda" and n_jobs > 1 and not allow_gpu_parallel:
        logger_temp = _master_logger()
        logger_temp.warning(
            f"Forcing n_jobs=1 for CUDA device (was {n_jobs}). "
            "Parallel jobs on GPU can cause memory errors. "
            "Use --allow-gpu-parallel to override."
        )
        n_jobs = 1
    
    ds_stem = os.path.splitext(dataset_filename)[0]
    logger = _make_tuning_logger(model_type, ds_stem, method_name)
    
    logger.info(f"{'='*70}")
    logger.info(f"  HP Tuning: {model_type.upper()} on {ds_stem}")
    logger.info(f"  Trials: {n_trials}, Jobs: {n_jobs}")
    logger.info(f"{'='*70}")
    
    try:
        # Load dataset
        X, y = load_2d_datasets(dataset_filename, method_name, DATA_DIR, logger)
        num_classes = int(y.unique().numel())
        
        logger.info(f"Dataset: {num_classes} classes, {len(y)} samples")
        
        # Create Optuna study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner()
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"{model_type}_{ds_stem}",
        )
        
        # Create objective function
        objective = create_objective(X, y, model_type, num_classes, logger)
        
        # Optimize
        logger.info(f"Starting optimization with {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
        
        # Get best trial
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        
        logger.info(f"{'─'*70}")
        logger.info(f"Best trial #{best_trial.number}:")
        logger.info(f"  Validation Accuracy: {best_value:.4f}")
        logger.info(f"  Hyperparameters:")
        for key, val in best_params.items():
            logger.info(f"    {key}: {val}")
        logger.info(f"{'─'*70}")
        
        # Save results
        output_dir = os.path.join(RESULTS_DIR, "hp_tuning", model_type, method_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best parameters as JSON
        params_path = os.path.join(output_dir, f"best_params_{ds_stem}.json")
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Saved best params: {params_path}")
        
        # Save full study (including all trials)
        history_path = os.path.join(output_dir, f"tuning_history_{ds_stem}.json")
        history = {
            "best_trial": best_trial.number,
            "best_value": best_value,
            "trials": [
                {
                    "number": t.number,
                    "value": t.value,
                    "params": t.params,
                    "state": str(t.state),
                }
                for t in study.trials
            ],
        }
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
        logger.info(f"Saved tuning history: {history_path}")
        
        return best_params, best_value
    
    except Exception as e:
        logger.error(f"Tuning failed: {str(e)}", exc_info=True)
        return None, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH TUNING & REPORTING
# ═══════════════════════════════════════════════════════════════════════════════
def tune_all_combinations(
    models: List[str],
    datasets: List[str],
    n_trials: int = 100,
    n_jobs: int = 1,
    method_name: str = "ours",
    allow_gpu_parallel: bool = False,
):
    """
    Tune hyperparameters for all (model, dataset) combinations.
    """
    master = _master_logger()
    master.info(f"Starting HP tuning for {len(models)} models × {len(datasets)} datasets")
    
    results = {}
    
    for model_type in models:
        results[model_type] = {}
        for dataset_filename in datasets:
            ds_stem = os.path.splitext(dataset_filename)[0]
            
            start_time = time.time()
            best_params, best_value = tune_hyperparameters(
                model_type, dataset_filename, method_name, n_trials, n_jobs, allow_gpu_parallel
            )
            elapsed = time.time() - start_time
            
            results[model_type][ds_stem] = {
                "best_params": best_params,
                "best_value": best_value,
                "time_seconds": elapsed,
            }
            
            master.info(
                f"✓ {model_type} + {ds_stem}: "
                f"val_acc={best_value:.4f} ({elapsed:.1f}s)"
            )
    
    # Generate summary report
    generate_tuning_report(results)
    
    return results


def generate_tuning_report(results: Dict):
    """Generate a comprehensive tuning report."""
    report_dir = os.path.join(RESULTS_DIR, "hp_tuning", "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    # Save summary as JSON
    summary_path = os.path.join(report_dir, "hp_tuning_summary.json")
    summary_data = {}
    
    for model_type, datasets_dict in results.items():
        summary_data[model_type] = {}
        for ds_stem, data in datasets_dict.items():
            summary_data[model_type][ds_stem] = {
                "best_value": data["best_value"],
                "time_seconds": data["time_seconds"],
                "best_params": data["best_params"],
            }
    
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    
    # Generate human-readable report
    report_path = os.path.join(report_dir, "hp_tuning_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("HYPERPARAMETER TUNING SUMMARY\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for model_type, datasets_dict in results.items():
            f.write(f"\n{'model_type'.upper()}: {model_type}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Dataset':<40} {'Best Acc':>10} {'Time (s)':>10}\n")
            f.write("-" * 80 + "\n")
            
            total_time = 0
            accs = []
            for ds_stem, data in sorted(datasets_dict.items()):
                f.write(
                    f"{ds_stem:<40} {data['best_value']:>10.4f} "
                    f"{data['time_seconds']:>10.1f}\n"
                )
                total_time += data["time_seconds"]
                if data["best_value"] > 0:
                    accs.append(data["best_value"])
            
            f.write("-" * 80 + "\n")
            if accs:
                f.write(
                    f"{'MEAN':<40} {np.mean(accs):>10.4f} "
                    f"{total_time:>10.1f}\n"
                )
    
    print(f"\nReports saved to: {report_dir}")
    print(f"  • Summary JSON: {summary_path}")
    print(f"  • Report text: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning with Optuna",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["efficientnet", "nctd_cnn"],
        help="Model to tune (if --all not specified)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Specific dataset to tune (if not provided, all datasets used)",
    )
    parser.add_argument(
        "--method",
        choices=["ours", "NCTD"],
        default="ours",
        help="Conversion method (default: ours)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Tune both models on all datasets",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials per (model, dataset) pair (default: 100)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for Optuna (default: 1)",
    )
    parser.add_argument(
        "--allow-gpu-parallel",
        action="store_true",
        help="Allow parallel jobs on GPU (WARNING: may cause OOM errors)",
    )
    
    args = parser.parse_args()
    
    master = _master_logger()
    master.info(f"HP Tuning started | {_RUN_TS}")
    
    # Determine models and datasets to tune
    models_to_tune = ["efficientnet", "nctd_cnn"] if args.all else [args.model]
    if not args.model and not args.all:
        parser.print_help()
        print("\nExamples:")
        print("  python hp_tuning.py --all --n-trials 100 --n-jobs 4")
        print("  python hp_tuning.py --model efficientnet --n-trials 50")
        print("  python hp_tuning.py --all --dataset 13_rotated_rastrigin_50d.npz")
        sys.exit(1)
    
    if args.dataset:
        datasets_to_tune = [args.dataset]
    else:
        datasets_to_tune = DATASETS_LIST
    
    # Run tuning
    start_time = time.time()
    results = tune_all_combinations(
        models_to_tune,
        datasets_to_tune,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        method_name=args.method,
        allow_gpu_parallel=args.allow_gpu_parallel,
    )
    elapsed = time.time() - start_time
    
    master.info(f"HP tuning completed in {elapsed:.1f}s")
    master.info(f"Results saved to: {os.path.join(RESULTS_DIR, 'hp_tuning')}")
