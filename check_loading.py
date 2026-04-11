import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader


# ================= LAZY DATASET =================
class LazyDataset(Dataset):
    def __init__(self, dataset_dir, cache_size=2):
        self.files = sorted([
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.endswith(".pt")
        ])

        self.index_map = []
        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size

        print("🔍 Building index map...")

        for file_idx, file in enumerate(self.files):
            data = torch.load(file)
            size = data['X'].shape[0]

            for i in range(size):
                self.index_map.append((file_idx, i))

            del data

        print(f"✅ Total samples: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def _load_file(self, file_idx):
        if file_idx in self.cache:
            return self.cache[file_idx]

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


# ================= FULL LOAD =================
def load_full_dataset(dataset_dir):
    files = sorted([
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
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


# ================= TEST FUNCTIONS =================
def test_lazy(dataset_dir):
    try:
        dataset = LazyDataset(dataset_dir)
        loader = DataLoader(dataset, batch_size=8)

        for X, y in loader:
            print(f"Lazy batch: {X.shape}")
            break

        print("✅ Lazy loading works\n")
    except Exception as e:
        print(f"❌ Lazy loading failed: {e}\n")


def test_full(dataset_dir):
    try:
        X, y = load_full_dataset(dataset_dir)
        print(f"Full dataset: {X.shape}")
        print("✅ Full loading works\n")
    except Exception as e:
        print(f"❌ Full loading failed: {e}\n")


# ================= MAIN =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset (e.g. 2d_nctd_datasets/NCTD/dataset1)"
    )

    args = parser.parse_args()

    dataset_dir = args.dataset

    print(f"\n🚀 Testing dataset: {dataset_dir}\n")

    test_lazy(dataset_dir)
    test_full(dataset_dir)

    print("🎉 Done")