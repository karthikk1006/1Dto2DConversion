import os
import json
import time
import logging
import datetime
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import optuna
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# ── GPU / Device Setup ───────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()

def print_gpu_banner():
    """Print a clear banner showing which device will be used for training."""
    sep = "=" * 60
    print(sep)
    if USE_GPU:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  🚀  GPU DETECTED — Training will use GPU")
        print(f"      Device : {gpu_name}")
        print(f"      VRAM   : {gpu_mem:.1f} GB")
        print(f"      Models benefiting: RNN (full GPU), XGBoost (GPU hist)")
        print(f"      Note   : LR / CART / ID3 / RF use scikit-learn (CPU)")
    else:
        print(f"  ⚠️   No GPU found — Training will use CPU")
        print(f"      Install CUDA + matching PyTorch build to enable GPU.")
    print(sep)

DATA_DIR = "datasets"
RESULTS_DIR = "results_ml"
_RUN_TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def get_logger(name, log_file):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    return logger

def load_data(filepath):
    # allow_pickle=True is needed for some datasets that store data as object arrays (e.g. digen)
    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        # Fallback for older numpy or specific cases
        data = np.load(filepath)
        
    X = data['X']
    y = data['y']
    
    # Handle datasets with headers in the first row (common in 'digen' files)
    if X.dtype == object or y.dtype == object:
        try:
            # Check if first element of y is a string (header)
            if isinstance(y[0], str):
                X = X[1:]
                y = y[1:]
        except:
            pass
        
        # Try to convert to proper numeric types
        try:
            X = X.astype(np.float64)
            y = y.astype(np.int64)
        except Exception as e:
            # If conversion fails, we might have mixed types or bad data
            pass
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    unique_labels = np.unique(y)
    mapping = {val: i for i, val in enumerate(np.sort(unique_labels))}
    mapped_y = np.copy(y)
    for old_val, new_val in mapping.items():
        mapped_y[y == old_val] = new_val
        
    return X, mapped_y

def compute_metrics(y_true, y_pred, y_prob, num_classes):
    avg_setting = "binary" if num_classes == 2 else "macro"
    try:
        if num_classes == 2:
            roc_auc = float(roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob))
        else:
            roc_auc = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=avg_setting, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=avg_setting, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg_setting, zero_division=0)),
        "roc_auc": roc_auc,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred))
    }
    return metrics

class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def train_eval_rnn(X_train, y_train, X_val, y_val, params, num_classes):
    device = DEVICE
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    batch_size = params.get('batch_size', 64)
    # pin_memory speeds up host→GPU transfers when a CUDA device is available
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=USE_GPU)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=USE_GPU)
    
    model = SimpleRNN(
        input_dim=X_train.shape[1],
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        num_classes=num_classes,
        dropout=params.get('dropout', 0.1)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    
    epochs = params.get('epochs', 30)
    best_val_acc = -1
    
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                out = model(Xb)
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_trues.extend(yb.numpy())
                
        val_acc = accuracy_score(val_trues, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
    return best_val_acc

def objective_ml(trial, X, y, model_name, num_classes):
    # Split the data into training and validation sets
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Ensure the training data contains at least two classes for classification
    if len(np.unique(y_train)) < 2:
        # Not enough classes; return a neutral score to allow Optuna to continue
        return 0.0
    
    if model_name == 'LR':
        C = trial.suggest_float('C', 1e-5, 1e2, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
        model = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        return accuracy_score(y_val, model.predict(X_val))
        
    elif model_name == 'CART':
        max_depth = trial.suggest_int('max_depth', 3, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        model = DecisionTreeClassifier(criterion='gini', max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        model.fit(X_train, y_train)
        return accuracy_score(y_val, model.predict(X_val))
        
    elif model_name == 'ID3':
        max_depth = trial.suggest_int('max_depth', 3, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        model.fit(X_train, y_train)
        return accuracy_score(y_val, model.predict(X_val))
        
    elif model_name == 'RF':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 30)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        return accuracy_score(y_val, model.predict(X_val))
        
    elif model_name == 'XGBoost':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.3, log=True)
        xgb_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric='logloss',
            random_state=42,
        )
        if USE_GPU:
            xgb_kwargs['device'] = 'cuda'
            xgb_kwargs['tree_method'] = 'hist'   # GPU-accelerated histogram method
        model = xgb.XGBClassifier(**xgb_kwargs)
        model.fit(X_train, y_train)
        return accuracy_score(y_val, model.predict(X_val))
        
    elif model_name == 'RNN':
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [16, 32, 64, 128]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'epochs': 30
        }
        if params['num_layers'] > 1:
            params['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)
        return train_eval_rnn(X_train, y_train, X_val, y_val, params, num_classes)

def train_final_model_and_evaluate(X_train, y_train, X_test, y_test, model_name, best_params, num_classes, model_save_path):
    if model_name == 'RNN':
        device = DEVICE
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        batch_size = best_params.get('batch_size', 64)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, pin_memory=USE_GPU)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=USE_GPU)
        
        model = SimpleRNN(
            input_dim=X_train.shape[1],
            hidden_dim=best_params['hidden_dim'],
            num_layers=best_params['num_layers'],
            num_classes=num_classes,
            dropout=best_params.get('dropout', 0.1)
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
        
        epochs = best_params.get('epochs', 30)
        # Re-create a shuffle train loader for training
        train_loader_shuffle = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=USE_GPU)
        for epoch in range(epochs):
            model.train()
            for Xb, yb in train_loader_shuffle:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()
                
        # Save model
        torch.save(model.state_dict(), model_save_path)
        
        # Evaluate
        model.eval()
        def evaluate_loader(loader):
            preds, trues, probs = [], [], []
            with torch.no_grad():
                for Xb, yb in loader:
                    Xb = Xb.to(device)
                    out = model(Xb)
                    prb = torch.softmax(out, dim=1).cpu().numpy()
                    probs.extend(prb)
                    preds.extend(out.argmax(1).cpu().numpy())
                    trues.extend(yb.numpy())
            return np.array(trues), np.array(preds), np.array(probs)
            
        train_true, train_pred, train_prob = evaluate_loader(train_loader)
        test_true, test_pred, test_prob = evaluate_loader(test_loader)
        
        return (train_true, train_pred, train_prob), (test_true, test_pred, test_prob)
        
    else:
        if model_name == 'LR':
            model = LogisticRegression(**best_params, max_iter=1000, random_state=42)
        elif model_name == 'CART':
            model = DecisionTreeClassifier(criterion='gini', **best_params, random_state=42)
        elif model_name == 'ID3':
            model = DecisionTreeClassifier(criterion='entropy', **best_params, random_state=42)
        elif model_name == 'RF':
            model = RandomForestClassifier(**best_params, random_state=42)
        elif model_name == 'XGBoost':
            xgb_final_kwargs = dict(**best_params, eval_metric='logloss', random_state=42)
            if USE_GPU:
                xgb_final_kwargs['device'] = 'cuda'
                xgb_final_kwargs['tree_method'] = 'hist'
            model = xgb.XGBClassifier(**xgb_final_kwargs)
            
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"Cannot fit model '{model_name}': Training data has only 1 class. Check data imbalance or split size.")
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, model_save_path)
        
        train_preds = model.predict(X_train)
        train_probs = model.predict_proba(X_train)
        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)
        
        return (y_train, train_preds, train_probs), (y_test, test_preds, test_probs)

def run_tuning_for_dataset(dataset_file, model_names, n_trials=100):
    ds_name = os.path.splitext(os.path.basename(dataset_file))[0]
    X, y = load_data(dataset_file)
    
    if len(X) != len(y):
        print(f"Skipping {ds_name}: Inconsistent sample counts (X: {len(X)}, y: {len(y)})")
        return

    num_classes = len(np.unique(y))
    if num_classes < 2:
        print(f"Skipping {ds_name}: Only {num_classes} class found. Classification requires at least 2.")
        return
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check if the resulting training set is valid for classification
    if len(np.unique(y_train)) < 2:
        print(f"Skipping {ds_name}: Training split contains only one class. (Imbalanced data?)")
        return
        
    for model_name in model_names:
        output_dir = os.path.join(RESULTS_DIR, model_name, ds_name)
        os.makedirs(output_dir, exist_ok=True)
        
        logger = get_logger(f"{model_name}_{ds_name}", os.path.join(output_dir, "tuning.log"))
        logger.info(f"Starting {model_name} on {ds_name} (Classes: {num_classes}, Samples: {len(y)})")
        
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective_ml(trial, X_train, y_train, model_name, num_classes), n_trials=n_trials, n_jobs=1)
        
        best_params = study.best_params
        if model_name == 'RNN':
            best_params['epochs'] = 30
            
        logger.info(f"Best params for {model_name}: {best_params}")
        
        model_ext = ".pt" if model_name == "RNN" else ".pkl"
        model_save_path = os.path.join(output_dir, f"best_model{model_ext}")
        
        train_results, test_results = train_final_model_and_evaluate(
            X_train, y_train, X_test, y_test, model_name, best_params, num_classes, model_save_path
        )
        
        train_metrics = compute_metrics(train_results[0], train_results[1], train_results[2], num_classes)
        test_metrics = compute_metrics(test_results[0], test_results[1], test_results[2], num_classes)
        
        with open(os.path.join(output_dir, "metrics_train.json"), "w") as f:
            json.dump(train_metrics, f, indent=4)
            
        with open(os.path.join(output_dir, "metrics_test.json"), "w") as f:
            json.dump(test_metrics, f, indent=4)
            
        with open(os.path.join(output_dir, "best_params.json"), "w") as f:
            json.dump(best_params, f, indent=4)
            
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    print_gpu_banner()   # Show GPU/CPU device info before anything else

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='Specific dataset to run')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of Optuna trials')
    args = parser.parse_args()
    
    models_to_run = ['LR', 'CART', 'ID3', 'RF', 'XGBoost', 'RNN']
    
    # Run dynamically for all files in DATA_DIR
    if args.dataset:
        files = [os.path.join(DATA_DIR, args.dataset)]
    else:
        files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.npz')]
        
    for f in sorted(files):
        run_tuning_for_dataset(f, models_to_run, n_trials=args.n_trials)
