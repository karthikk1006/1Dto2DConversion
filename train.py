import os
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model import get_model
from dataset import Tabular2ImageDataset, load_data, safe_stratified_split
from utils import set_seed, setup_logger, save_checkpoint
from metrics import calculate_metrics, save_metrics, plot_confusion_matrix, plot_training_curves

def train_one_dataset(args, dataset_name, device):
    # Setup paths
    ds_stem = os.path.splitext(dataset_name)[0]
    output_dir = os.path.join(args.output_dir, ds_stem, args.model)
    os.makedirs(output_dir, exist_ok=True)
    
    logger = setup_logger(f"{ds_stem}_{args.model}", os.path.join(output_dir, "train.log"))
    logger.info(f"Starting training on {dataset_name} with {args.model}")

    # Try several potential paths
    potential_paths = [
        dataset_name,
        os.path.join("2d_datasets", "ours", dataset_name),
        os.path.join("2d_datasets", "ours", f"processed_{dataset_name}.pt"),
        os.path.join("2d_datasets", "ours", ds_stem),
    ]
    
    dataset_path = None
    for p in potential_paths:
        if os.path.exists(p):
            dataset_path = p
            break
            
    if dataset_path is None:
        logger.error(f"Dataset not found: {dataset_name}")
        return

    X, y = load_data(dataset_path, logger)
    num_samples = len(y)
    num_classes = int(y.unique().numel())
    logger.info(f"Loaded {num_samples} samples with {num_classes} classes from {dataset_path}")

    # Split data
    train_idx, test_idx = safe_stratified_split(range(num_samples), y, test_size=0.2, random_state=args.seed)
    train_idx, val_idx = safe_stratified_split(train_idx, y[train_idx], test_size=0.125, random_state=args.seed)

    train_ds = Tabular2ImageDataset(X, y, indices=train_idx, input_channels=args.input_channels)
    val_ds = Tabular2ImageDataset(X, y, indices=val_idx, input_channels=args.input_channels)
    test_ds = Tabular2ImageDataset(X, y, indices=test_idx, input_channels=args.input_channels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = get_model(args.model, num_classes, in_channels=args.input_channels).to(device)
    
    # Optimizer & Criterion
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision
    scaler = torch.amp.GradScaler(enabled=True)
    
    # Early stopping
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=True):
                outputs = model(imgs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*train_correct/train_total:.2f}%"})

        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=True):
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct / val_total
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        # Checkpoint
        is_best = avg_val_acc > best_val_acc
        if is_best:
            best_val_acc = avg_val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_val_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, output_dir)
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    best_model_path = os.path.join(output_dir, 'best_model.pt')
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    test_preds = []
    test_trues = []
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            test_preds.extend(predicted.cpu().numpy())
            test_trues.extend(targets.cpu().numpy())
    
    test_metrics = calculate_metrics(test_trues, test_preds, num_classes)
    logger.info(f"Test Metrics: {test_metrics}")
    save_metrics(test_metrics, os.path.join(output_dir, "metrics.json"))
    
    # Visualizations
    plot_training_curves(history['train_acc'], history['val_acc'], 'Accuracy', os.path.join(output_dir, 'training_curves.png'))
    plot_confusion_matrix(test_trues, test_preds, range(num_classes), os.path.join(output_dir, 'confusion_matrix.png'))
    
    logger.info(f"Finished training on {dataset_name}")

def main():
    parser = argparse.ArgumentParser(description="Train CNN architectures for tabular-image classification")
    parser.add_argument("--datasets", nargs='+', required=True, help="List of datasets to train on")
    parser.add_argument("--model", choices=['simpleresnet', 'seresnet', 'optimizedseresnet'], required=True, help="Model architecture")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--input_channels", type=int, default=1)
    parser.add_argument("--num_classes", type=int, help="Optional: override num classes")
    parser.add_argument("--image_size", type=int, help="Optional: input image size")

    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for ds in args.datasets:
        train_one_dataset(args, ds, device)

if __name__ == "__main__":
    main()
