# ================= IMPORTS =================
import os
import math
import gc
import torch
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


# ================= CONFIG =================
@dataclass
class Config:
    input_dimension: int
    method: str = "NCTD"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= VECTORIZED CONVERT =================
def convert(x_gray, args):
    B, N = x_gray.shape
    device = x_gray.device

    if args.method == "NCTD":
        shifts = torch.arange(N, device=device)
        idx = (torch.arange(N, device=device).unsqueeze(0) - shifts.unsqueeze(1)) % N
        A = x_gray[:, idx]  # (B, N, N)

        G = torch.cat([A, A], dim=2)
        G = torch.cat([G, G], dim=1)

    elif args.method == "ours":
        rows = math.ceil(((N + 2) // 6) * 2 + 1)

        j = torch.arange(1, rows + 1, device=device)
        shift_vals = 3 * (j**2 - (j % 2)) // 4

        idx = (torch.arange(N, device=device).unsqueeze(0) + shift_vals.unsqueeze(1)) % N
        A = x_gray[:, idx]

        G = torch.cat([A[:, :, -1:], A, A[:, :, :1]], dim=2)

    return G.unsqueeze(1).repeat(1, 3, 1, 1)


# ================= STREAMING PROCESS =================
def process_dataset(method, filename, rootdir, out_dir, batch_size=64):

    print(f"\n🚀 Processing {filename} → {method}")

    data = np.load(os.path.join(rootdir, filename), mmap_mode='r')
    X_np, y_np = data['X'], data['y']

    N, D = X_np.shape
    args = Config(input_dimension=D, method=method)

    method_dir = os.path.join(out_dir, method)
    os.makedirs(method_dir, exist_ok=True)

    base_name = os.path.splitext(filename)[0]

    # ---- global normalization (no full torch load)
    x_min = torch.from_numpy(X_np.min(axis=0, keepdims=True)).to(args.device)
    x_max = torch.from_numpy(X_np.max(axis=0, keepdims=True)).to(args.device)

    part_idx = 0

    for i in range(0, N, batch_size):

        X_batch = torch.from_numpy(X_np[i:i+batch_size]).to(args.device)

        X_norm = (X_batch - x_min) / (x_max - x_min + 1e-8)
        x_gray = X_norm * 255.0

        with torch.no_grad():
            X_img = convert(x_gray, args).cpu()

        y_batch = torch.from_numpy(y_np[i:i+batch_size]).long()

        save_path = os.path.join(
            method_dir,
            f"{base_name}_part{part_idx}.pt"
        )

        torch.save({'X': X_img, 'y': y_batch}, save_path)

        print(f"  ✅ Saved part {part_idx}")

        part_idx += 1

        del X_batch, X_norm, x_gray, X_img, y_batch
        gc.collect()

    print("🎉 Done (no merge needed)")


# ================= LAZY DATASET =================
class LazyDataset(Dataset):
    def __init__(self, data_dir, cache_size=2):
        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".pt")
        ])

        self.index_map = []
        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size

        # Build index map
        for file_idx, file in enumerate(self.files):
            data = torch.load(file)
            size = data['X'].shape[0]

            for i in range(size):
                self.index_map.append((file_idx, i))

            del data

    def __len__(self):
        return len(self.index_map)

    def _load_file(self, file_idx):
        if file_idx in self.cache:
            return self.cache[file_idx]

        # LRU cache eviction
        if len(self.cache) >= self.cache_size:
            old = self.cache_order.pop(0)
            del self.cache[old]

        data = torch.load(self.files[file_idx])
        self.cache[file_idx] = data
        self.cache_order.append(file_idx)

        return data

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]
        data = self._load_file(file_idx)

        return data['X'][sample_idx], data['y'][sample_idx]


# ================= DATALOADER =================
def get_dataloader(data_dir, batch_size=32, num_workers=2):

    dataset = LazyDataset(data_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader


# ================= FULL LOAD (OPTIONAL) =================
def load_full_dataset(data_dir):
    files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".pt")
    ])

    X_all, y_all = [], []

    for f in files:
        data = torch.load(f)
        X_all.append(data['X'])
        y_all.append(data['y'])

    X = torch.cat(X_all, dim=0)
    y = torch.cat(y_all, dim=0)

    return X, y


# ================= MAIN =================
if __name__ == "__main__":

    rootdir = "nctd_datasets"
    out_dir = "2d_nctd_datasets"

    datasets = [f for f in os.listdir(rootdir) if f.endswith(".npz")]

    for filename in datasets:
        process_dataset("NCTD", filename, rootdir, out_dir)
        process_dataset("ours", filename, rootdir, out_dir)

    print("\n🎉 ALL DONE")
