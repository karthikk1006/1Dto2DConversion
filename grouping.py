import os
import shutil


def group_pt_files(base_dir):
    """
    Groups .pt files inside NCTD and ours folders into dataset-wise subfolders.

    Example:
        dataset1_part0.pt → dataset1/
    """

    methods = ["NCTD", "ours"]

    for method in methods:
        method_dir = os.path.join(base_dir, method)

        if not os.path.exists(method_dir):
            print(f"⚠️ Skipping {method} (not found)")
            continue

        print(f"\n📂 Processing: {method_dir}")

        files = [f for f in os.listdir(method_dir) if f.endswith(".pt")]

        for file in files:
            try:
                # Extract dataset name
                dataset_name = file.split("_part")[0]

                target_dir = os.path.join(method_dir, dataset_name)
                os.makedirs(target_dir, exist_ok=True)

                src = os.path.join(method_dir, file)
                dst = os.path.join(target_dir, file)

                shutil.move(src, dst)

                print(f"  ✅ {file} → {dataset_name}/")

            except Exception as e:
                print(f"  ❌ Error moving {file}: {e}")

    print("\n🎉 Grouping complete!")


# ================= MAIN =================
if __name__ == "__main__":
    base_dir = "2d_nctd_datasets"   # change if needed
    group_pt_files(base_dir)