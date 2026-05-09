import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.trial import Trial

from model import get_model
from dataset import Tabular2ImageDataset, load_data, safe_stratified_split
from utils import set_seed

def objective(trial, args, X, y, num_classes, device):
    # Suggest hyperparameters
    lr = trial.suggest_categorical("lr", [1e-2, 1e-3, 5e-4, 1e-4])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-3, 1e-4, 1e-5])
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.5])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    scheduler_name = trial.suggest_categorical("scheduler", ["CosineAnnealingLR", "ReduceLROnPlateau"])
    label_smoothing = trial.suggest_categorical("label_smoothing", [0.0, 0.05, 0.1])

    # Split data
    num_samples = len(y)
    train_idx, test_idx = safe_stratified_split(range(num_samples), y, test_size=0.2, random_state=args.seed)
    train_idx, val_idx = safe_stratified_split(train_idx, y[train_idx], test_size=0.125, random_state=args.seed)

    train_ds = Tabular2ImageDataset(X, y, indices=train_idx, input_channels=args.input_channels)
    val_ds = Tabular2ImageDataset(X, y, indices=val_idx, input_channels=args.input_channels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = get_model(args.model, num_classes, in_channels=args.input_channels, dropout=dropout).to(device)
    
    # Optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Scheduler
    if scheduler_name == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    # Training loop (simplified for tuning)
    best_val_acc = 0.0
    for epoch in range(min(args.epochs, 20)): # Limit epochs for tuning
        model.train()
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        if scheduler_name == "CosineAnnealingLR":
            scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = val_correct / val_total
        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(val_acc)
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_acc

def tune_one_dataset(args, dataset_name, device):
    ds_stem = os.path.splitext(dataset_name)[0]
    output_dir = os.path.join(args.output_dir, ds_stem, args.model)
    os.makedirs(output_dir, exist_ok=True)

    # Try several potential paths (automatic search outside current root)
    potential_paths = [
        dataset_name,
        os.path.join("2d_datasets", "ours", dataset_name),
        os.path.join("2d_datasets", "ours", f"processed_{dataset_name}.pt"),
        os.path.join("2d_datasets", "ours", ds_stem),
        # Outside root
        os.path.join("..", "2d_datasets", "ours", dataset_name),
        os.path.join("..", "2d_datasets", "ours", f"processed_{dataset_name}.pt"),
        os.path.join("..", "2d_datasets", "ours", ds_stem),
        os.path.join("..", dataset_name),
    ]
    
    dataset_path = None
    for p in potential_paths:
        if os.path.exists(p):
            dataset_path = p
            break
            
    if dataset_path is None:
        print(f"Dataset not found: {dataset_name}")
        return

    X, y = load_data(dataset_path)
    num_classes = int(y.unique().numel())

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args, X, y, num_classes, device), n_trials=args.n_trials)

    print(f"Best hyperparameters for {dataset_name}: {study.best_params}")
    
    with open(os.path.join(output_dir, "best_hparams.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Tune CNN architectures")
    parser.add_argument("--datasets", nargs='+', required=True)
    parser.add_argument("--model", choices=['simpleresnet', 'seresnet', 'optimizedseresnet'], required=True)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_channels", type=int, default=1)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--output_dir", type=str, default=os.path.join(current_dir, "outputs"))

    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for ds in args.datasets:
        tune_one_dataset(args, ds, device)

if __name__ == "__main__":
    main()
