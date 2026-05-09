import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class Tabular2ImageDataset(Dataset):
    def __init__(self, X, y, transform=None, indices=None, input_channels=1):
        self.X = X
        self.y = y.long()
        self.transform = transform
        self.indices = indices
        self.input_channels = input_channels
        
        # Remap labels to be 0-indexed and contiguous
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

        # Ensure image is tensor and float
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)
        
        # Ensure grayscale if needed (1, H, W)
        if img.dim() == 2:
            img = img.unsqueeze(0)
            
        # Handle channel mismatch
        if img.shape[0] != self.input_channels:
            if self.input_channels == 1:
                # Convert 3-channel to 1-channel (grayscale)
                img = img.mean(dim=0, keepdim=True)
            elif self.input_channels == 3:
                # Convert 1-channel to 3-channel (repeat)
                img = img.repeat(3, 1, 1)

        if self.transform:
            img = self.transform(img)
            
        return img, target

def safe_stratified_split(indices, y, test_size, random_state=42):
    y_numpy = y.cpu().numpy() if hasattr(y, 'cpu') else np.array(y)
    unique, counts = np.unique(y_numpy, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    singletons = [cls for cls, count in class_counts.items() if count < 2]
    
    if not singletons:
        return train_test_split(indices, test_size=test_size, stratify=y_numpy, random_state=random_state)
    
    singleton_mask = np.isin(y_numpy, singletons)
    singleton_indices = np.array(indices)[singleton_mask]
    
    multi_mask = ~singleton_mask
    multi_indices = np.array(indices)[multi_mask]
    multi_y = y_numpy[multi_mask]
    
    if len(multi_indices) < 2 or len(np.unique(multi_y)) < 2:
        return train_test_split(indices, test_size=test_size, random_state=random_state)

    try:
        train_idx, test_idx = train_test_split(
            multi_indices, test_size=test_size, stratify=multi_y, random_state=random_state
        )
        train_idx = np.concatenate([train_idx, singleton_indices])
        return train_idx.tolist(), test_idx.tolist()
    except Exception:
        return train_test_split(indices, test_size=test_size, random_state=random_state)

def load_data(dataset_path, logger=None):
    # Try local paths first
    if not os.path.exists(dataset_path):
        # Check specifically in 2d_datasets/ours/
        ours_path = os.path.join("2d_datasets", "ours", os.path.basename(dataset_path))
        if os.path.exists(ours_path):
            dataset_path = ours_path

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    if os.path.isdir(dataset_path):
        # Chunked dataset
        files = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".pt")])
        if not files:
            raise FileNotFoundError(f"No .pt files found in directory: {dataset_path}")
        
        X_all, y_all = [], []
        for f in files:
            data = torch.load(f, map_location="cpu", weights_only=True)
            X_all.append(data['X'])
            y_all.append(data['y'])
        
        X = torch.cat(X_all, dim=0)
        y = torch.cat(y_all, dim=0)
    else:
        # Singular file
        data = torch.load(dataset_path, map_location="cpu", weights_only=True)
        X, y = data["X"], data["y"]
    
    if torch.isnan(X).any():
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check for extreme values and normalize if necessary
    x_min, x_max = X.min(), X.max()
    if x_max > 1e3 or x_min < -1e3:
        X = (X - x_min) / (x_max - x_min + 1e-8)
        
    return X, y
