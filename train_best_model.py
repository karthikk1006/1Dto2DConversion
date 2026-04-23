import os
import sys
import json
import time
import logging
import argparse
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import re

# Import core utilities from the existing pipeline
from train_pipeline import (
    load_2d_datasets,
    Tabular2ImageDataset,
    evaluate_model,
    nctd_transform,
    DEVICE,
    _USE_AMP,
)

# ═══════════════════════════════════════════════════════════════════════════════
# SEED SETTING
# ═══════════════════════════════════════════════════════════════════════════════
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed set to: {seed}")

# ═══════════════════════════════════════════════════════════════════════════════
# CNN ARCHITECTURE (WITH DYNAMIC DROPOUT)
# ═══════════════════════════════════════════════════════════════════════════════
class BestCNN(nn.Module):
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
# HYPERPARAMETER PARSER
# ═══════════════════════════════════════════════════════════════════════════════
def parse_best_params(dump_path, analysis_path):
    """
    Parses hyperparameters from both the tuned dump and the improvement analysis.
    Returns a dict: { dataset_name: { method: params_dict } }
    """
    best_params = {}

    # 1. Parse from all_tuned_results_full_dump.md (Markdown + JSON blocks)
    if os.path.exists(dump_path):
        with open(dump_path, "r") as f:
            content = f.read()
        
        dataset_blocks = content.split("## Dataset: `")
        for block in dataset_blocks[1:]:
            ds_name = block.split("`")[0]
            best_params[ds_name] = {}
            
            # Split by method
            model_blocks = block.split("### CNN + ")
            for m_block in model_blocks[1:]:
                method = "ours" if "Ours" in m_block.split("\n")[0] else "NCTD"
                # Find the first json block (Hyperparameters)
                hp_match = re.search(r"\*\*Hyperparameters:\*\*\s*```json\s*(\{.*?\})\s*```", m_block, re.DOTALL)
                if hp_match:
                    try:
                        best_params[ds_name][method] = json.loads(hp_match.group(1))
                    except: pass

    # 2. Parse from cnn_improvement_analysis.md (For missing DS datasets)
    if os.path.exists(analysis_path):
        with open(analysis_path, "r") as f:
            content = f.read()
            
        dataset_blocks = content.split("### Dataset: `")
        for block in dataset_blocks[1:]:
            ds_name = block.split("`")[0]
            if ds_name not in best_params:
                best_params[ds_name] = {}
            
            # Parse Ours params
            ours_hp_match = re.search(r"#### Hyperparameters \(Ours\):\s*```json\s*(\{.*?\})\s*```", block, re.DOTALL)
            if ours_hp_match:
                try:
                    best_params[ds_name]["ours"] = json.loads(ours_hp_match.group(1))
                except: pass
            
            # Parse NCTD params
            nctd_hp_match = re.search(r"#### Hyperparameters \(NCTD baseline\):\s*```json\s*(\{.*?\})\s*```", block, re.DOTALL)
            if nctd_hp_match:
                try:
                    best_params[ds_name]["NCTD"] = json.loads(nctd_hp_match.group(1))
                except: pass
                
    return best_params

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def train_best_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dataset_name: str,
    output_dir: str,
    logger: logging.Logger,
    hp: dict
):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    
    # Optimizer
    opt_name = hp.get("optimizer", "adam").lower()
    lr = hp.get("learning_rate", 0.001)
    wd = hp.get("weight_decay", 1e-4)
    
    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=hp.get("momentum", 0.9))
        
    # Scheduler
    sched_name = hp.get("scheduler", "none").lower()
    epochs = hp.get("epochs", 40)
    scheduler = None
    
    if sched_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.get("scheduler_t_max", epochs))
    elif sched_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hp.get("scheduler_step_size", 10), gamma=hp.get("scheduler_gamma", 0.1))
    elif sched_name == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=hp.get("scheduler_gamma", 0.95))
        
    scaler = torch.amp.GradScaler(enabled=_USE_AMP)
    
    best_val_acc = -1.0
    best_weights = None
    patience = 10
    patience_counter = 0
    
    best_path = os.path.join(output_dir, f"best_tuned_{dataset_name}.pth")
    
    epoch_bar = tqdm(range(epochs), desc=f"  Training {dataset_name}", leave=False)
    
    for epoch in epoch_bar:
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=_USE_AMP):
                loss = criterion(model(Xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        # Validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(DEVICE)
                out = model(Xb)
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_trues.extend(yb.numpy())
        
        val_acc = accuracy_score(val_trues, val_preds)
        if scheduler: scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.state_dict().copy()
            torch.save(best_weights, best_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
        epoch_bar.set_postfix(val_acc=f"{val_acc:.4f}", best=f"{best_val_acc:.4f}")
        
    return best_path, best_val_acc

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Train CNN with best found hyperparameters")
    parser.add_argument("--dataset-root", type=str, required=True, help="Root folder containing 'ours' and 'NCTD' datasets")
    parser.add_argument("--method", choices=["ours", "NCTD"], default="ours", help="Method to train")
    parser.add_argument("--output-dir", type=str, default="best_models_run", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Logging
    log_file = os.path.join(args.output_dir, "training.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger("best_train")

    # Load Params
    dump_path = "all_tuned_results_full_dump.md"
    analysis_path = "cnn_improvement_analysis.md"
    best_params_map = parse_best_params(dump_path, analysis_path)
    
    method_path = os.path.join(args.dataset_root, args.method)
    if not os.path.exists(method_path):
        logger.error(f"Path {method_path} not found.")
        return

    # Discover datasets
    datasets = []
    for item in os.listdir(method_path):
        name = item
        if name.startswith("processed_"): name = name[len("processed_"):]
        name = os.path.splitext(name)[0]
        if name not in datasets: datasets.append(name)
    
    logger.info(f"Starting training for {len(datasets)} datasets using {args.method.upper()} method.")

    for ds_name in datasets:
        if ds_name not in best_params_map or args.method not in best_params_map[ds_name]:
            logger.warning(f"No best params found for {ds_name} ({args.method}). Skipping.")
            continue
            
        hp = best_params_map[ds_name][args.method]
        logger.info(f"Training {ds_name} with params: {hp}")
        
        try:
            # Load Data
            X, y = load_2d_datasets(ds_name, args.method, args.dataset_root, logger)
            num_classes = int(y.unique().numel())
            num_samples = int(len(y))
            
            # Split
            train_idx, test_idx = train_test_split(range(num_samples), test_size=0.2, stratify=y.cpu().numpy(), random_state=args.seed)
            train_idx, val_idx = train_test_split(train_idx, test_size=0.125, stratify=y[train_idx].cpu().numpy(), random_state=args.seed)
            
            train_ds = Tabular2ImageDataset(X, y, "nctd_cnn", nctd_transform, indices=train_idx)
            val_ds = Tabular2ImageDataset(X, y, "nctd_cnn", nctd_transform, indices=val_idx)
            test_ds = Tabular2ImageDataset(X, y, "nctd_cnn", nctd_transform, indices=test_idx)
            
            batch_size = hp.get("batch_size", 32)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            
            # Model
            model = BestCNN(num_classes=num_classes, dropout_rate=hp.get("dropout_rate", 0.3))
            
            # Train
            best_path, best_val = train_best_model(model, train_loader, val_loader, ds_name, args.output_dir, logger, hp)
            
            # Evaluate
            model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
            metrics, _, _ = evaluate_model(model, test_loader, logger)
            
            logger.info(f"DONE: {ds_name} | Best Val: {best_val:.4f} | Test Acc: {metrics['accuracy']:.4f}")
            
            # Save results
            with open(os.path.join(args.output_dir, f"{ds_name}_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
                
        except Exception as e:
            logger.error(f"Failed {ds_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
