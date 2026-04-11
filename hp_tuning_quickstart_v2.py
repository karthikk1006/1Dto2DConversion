#!/usr/bin/env python3
"""
hp_tuning_quickstart_2d.py
=========================
One-command launcher for the complete hyperparameter tuning workflow (2D NCTD datasets).

This script provides an interactive menu to run the entire HP tuning pipeline
for the 2d_nctd_datasets directory, supporting both lazy and full loading strategies.

Usage
-----
  python hp_tuning_quickstart_2d.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def print_header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_section(text):
    print(f"\n{'─' * 80}")
    print(f"  {text}")
    print(f"{'─' * 80}")


def run_command(cmd, description, auto_confirm=False):
    print_section(description)
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    if not auto_confirm:
        confirm = input("Run command? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Skipped.")
            return False
    
    print("\n" + "=" * 80)
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print("=" * 80)
        print(f"✓ Completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print("=" * 80)
        print(f"✗ Command failed with exit code {e.returncode}")
        return False


def get_parallel_flags(default_jobs="2"):
    n_jobs = input(f"\nNumber of parallel jobs (1-4, default {default_jobs}): ").strip() or default_jobs
    flags = ["--n-jobs", n_jobs]
    if int(n_jobs) > 1:
        allow_gpu = input("Allow parallel GPU jobs? (WARNING: may cause OOM) (y/N): ").strip().lower()
        if allow_gpu == 'y':
            flags.append("--allow-gpu-parallel")
    return flags


def get_loading_strategy():
    print("\nChoose dataset loading strategy:")
    print("  [1] Lazy Loading (Recommended for large datasets)")
    print("  [2] Full Loading (For small datasets/debugging)")
    choice = input("Enter choice [1-2, default 1]: ").strip() or "1"
    if choice == "2":
        return "--full-loading"
    return "--lazy-loading"


def show_menu():
    print_header("HP TUNING QUICKSTART MENU (2D NCTD DATASETS)")
    print("""
Choose an option:

[1] QUICK TEST
    • 10 trials on 1 dataset (EfficientNet)
    • Best for: Testing setup
    • Est. time: 5-10 minutes

[2] SINGLE MODEL - QUICK TUNE
    • 50 trials on all datasets (EfficientNet)
    • Best for: Initial exploration
    • Est. time: 2-3 hours

[3] SINGLE MODEL - FULL TUNE
    • 100 trials on all datasets (EfficientNet)
    • Best for: Production quality
    • Est. time: 4-6 hours

[4] BOTH MODELS - QUICK TUNE
    • 50 trials per model on all datasets
    • Auto-optimized for your GPU (n_jobs auto-adjusted)
    • Best for: Fast comparison
    • Est. time: 2-3 hours (sequential on GPU)

[5] BOTH MODELS - FULL TUNE (RECOMMENDED)
    • 100 trials per model on all datasets
    • Auto-optimized for your GPU (n_jobs auto-adjusted)
    • Best for: Best possible results
    • Est. time: 4-8 hours (sequential on GPU)

[6] TRAIN WITH TUNED HYPERPARAMETERS
    • Train all models using tuned hyperparameters
    • Best for: After hp_tuning.py completes
    • Est. time: 4-6 hours

[7] TRAIN WITH TUNING (ALL-IN-ONE)
    • Combines tuning + training in one step
    • Best for: Complete workflow
    • Est. time: 8-14 hours

[8] COMPARE RESULTS
    • Generate comparison visualizations & tables
    • Best for: After training
    • Est. time: 5-10 minutes

[9] VIEW HP TUNING GUIDE
    • Display comprehensive documentation
    • Best for: Learning more details

[10] FULL TUNE CNN (BOTH METHODS, ALL DATASETS) + TRAIN
    • Tune CNN model for both 'ours' and 'NCTD' methods across datasets
    • Train CNN with tuned hyperparameters
    • Best for: Running the complete CNN pipeline

[11] FULL TUNE BOTH MODELS (BOTH METHODS, ALL DATASETS) + TRAIN
    • Tune both CNN and EfficientNet for both methods across datasets
    • Train all models with tuned hyperparameters
    • Best for: Comprehensive pipeline run (auto-discovers datasets)

[0] EXIT

""")
    choice = input("Enter choice [0-11]: ").strip()
    return choice


def main():
    os.chdir(Path(__file__).parent)
    print_header("HP TUNING WORKFLOW - INTERACTIVE LAUNCHER (2D NCTD DATASETS)")
    print("""
This tool provides an interactive interface for running the hyperparameter
tuning workflow on your 2D NCTD classification pipeline.

The workflow consists of:
  1. HP_TUNING.PY → Find optimal hyperparameters using Optuna
  2. TRAIN_WITH_TUNING.PY → Train models with tuned hyperparameters
  3. COMPARE_METRICS.PY → Compare results and generate visualizations

Start by running HP Tuning, then Training, then Comparison.

Press Enter to continue...
""")
    input()
    
    while True:
        choice = show_menu()
        loading_flag = get_loading_strategy()
        dataset_flag = ["--dataset-root", "2d_nctd_datasets"]
        
        if choice == "1":
            run_command(
                ["python", "hp_tuning.py", "--model", "efficientnet", "--n-trials", "10", "--dataset", "13_rotated_rastrigin_50d.npz"] + dataset_flag + [loading_flag],
                "QUICK TEST: EfficientNet - 10 trials on 1 dataset (2D NCTD)"
            )
        elif choice == "2":
            # Model selection for single model quick tune
            print("\nSelect model for tuning:")
            print("  [1] EfficientNet (default)")
            print("  [2] CNN (nctd_cnn)")
            model_choice = input("Enter choice [1-2]: ").strip() or "1"
            model = "efficientnet" if model_choice == "1" else "nctd_cnn"
            
            ds_input = input("\nEnter dataset name (e.g. 13_rotated_rastrigin_50d) mapping or press Enter for ALL: ").strip()
            ds_args = ["--dataset", ds_input] if ds_input else []

            run_command(
                ["python", "hp_tuning.py", "--model", model, "--method", "ours", "--n-trials", "50"] + ds_args + dataset_flag + [loading_flag],
                f"SINGLE MODEL - QUICK TUNE: {model.upper()} (ours) - 50 trials (2D NCTD)"
            )
            run_command(
                ["python", "hp_tuning.py", "--model", model, "--method", "NCTD", "--n-trials", "50"] + ds_args + dataset_flag + [loading_flag],
                f"SINGLE MODEL - QUICK TUNE: {model.upper()} (NCTD) - 50 trials (2D NCTD)"
            )
        elif choice == "3":
            # Model selection for single model full tune
            print("\nSelect model for tuning:")
            print("  [1] EfficientNet (default)")
            print("  [2] CNN (nctd_cnn)")
            model_choice = input("Enter choice [1-2]: ").strip() or "1"
            model = "efficientnet" if model_choice == "1" else "nctd_cnn"
            
            ds_input = input("\nEnter dataset name (e.g. 13_rotated_rastrigin_50d) mapping or press Enter for ALL: ").strip()
            ds_args = ["--dataset", ds_input] if ds_input else []

            run_command(
                ["python", "hp_tuning.py", "--model", model, "--method", "ours", "--n-trials", "100"] + ds_args + dataset_flag + [loading_flag],
                f"SINGLE MODEL - FULL TUNE: {model.upper()} (ours) - 100 trials (2D NCTD)"
            )
            run_command(
                ["python", "hp_tuning.py", "--model", model, "--method", "NCTD", "--n-trials", "100"] + ds_args + dataset_flag + [loading_flag],
                f"SINGLE MODEL - FULL TUNE: {model.upper()} (NCTD) - 100 trials (2D NCTD)"
            )
        elif choice == "4":
            parallel_flags = get_parallel_flags("2")
            run_command(
                ["python", "hp_tuning.py", "--all", "--method", "ours", "--n-trials", "50"] + parallel_flags + dataset_flag + [loading_flag],
                "BOTH MODELS - QUICK TUNE (Method: ours): 50 trials each (2D NCTD)"
            )
            run_command(
                ["python", "hp_tuning.py", "--all", "--method", "NCTD", "--n-trials", "50"] + parallel_flags + dataset_flag + [loading_flag],
                "BOTH MODELS - QUICK TUNE (Method: NCTD): 50 trials each (2D NCTD)"
            )
        elif choice == "5":
            print_section("BOTH MODELS - FULL TUNE (RECOMMENDED)")
            print("\nThis will tune all 4 combinations (2 models x 2 methods) and then train with best params.")
            parallel_flags = get_parallel_flags("4")
            print("\n[STEP 1] Tuning Method: ours...")
            if not run_command(
                ["python", "hp_tuning.py", "--all", "--method", "ours", "--n-trials", "100", "--output-dir", "results_nctd_dataset"] + parallel_flags + dataset_flag + [loading_flag],
                "STEP 1/3: HP TUNING - Method: ours (2D NCTD) -> results_nctd_dataset"
            ):
                print("HP Tuning ours failed. Aborting workflow.")
                continue
            print("\n[STEP 2] Tuning Method: NCTD...")
            if not run_command(
                ["python", "hp_tuning.py", "--all", "--method", "NCTD", "--n-trials", "100", "--output-dir", "results_nctd_dataset"] + parallel_flags + dataset_flag + [loading_flag],
                "STEP 2/3: HP TUNING - Method: NCTD (2D NCTD) -> results_nctd_dataset"
            ):
                print("HP Tuning NCTD failed. Aborting workflow.")
                continue
            print("\n[STEP 3] Starting Training and Evaluation with Tuned Hyperparameters...")
            if not run_command(
                ["python", "train_with_tuning.py", "--all", "--output-dir", "results_nctd_dataset"] + dataset_flag + [loading_flag],
                "STEP 3/3: TRAINING WITH TUNING (All Combinations, 2D NCTD) -> results_nctd_dataset"
            ):
                print("Training failed.")
                continue
            print("\nResults saved properly. Test results are stored in results_nctd_dataset/tuned_results/... (2D NCTD)")
        elif choice == "6":
            run_command(
                ["python", "train_with_tuning.py", "--all", "--output-dir", "results_nctd_dataset"] + dataset_flag + [loading_flag],
                "TRAINING: Use tuned hyperparameters for both models (2D NCTD), output to results_nctd_dataset"
            )
        elif choice == "10":
            # Full tune CNN for both methods and all 10 datasets, then train with tuned params, output to results_nctd_dataset
            print_section("FULL TUNE CNN (BOTH METHODS, ALL DATASETS) + TRAIN [NEW]")
            parallel_flags = get_parallel_flags("4")
            auto_yes = input("\nAuto-confirm all (y/n)? ").strip().lower() == 'y'
            methods = ["ours", "NCTD"]
            # You may want to specify your 10 datasets explicitly if needed
            datasets = [d for d in os.listdir(os.path.join("2d_nctd_datasets", "ours")) if os.path.isdir(os.path.join("2d_nctd_datasets", "ours", d))][:10]
            for method in methods:
                for dataset in datasets:
                    print(f"\nTuning: Method={method}, Dataset={dataset}, Model=nctd_cnn")
                    run_command(
                        ["python", "hp_tuning.py", "--model", "nctd_cnn", "--method", method, "--dataset", dataset, "--n-trials", "100", "--output-dir", "results_nctd_dataset"] + parallel_flags + dataset_flag + [loading_flag],
                        f"HP TUNING: CNN, Method={method}, Dataset={dataset}, 100 trials, output to results_nctd_dataset",
                        auto_confirm=auto_yes
                    )
            print("\nStarting training with tuned parameters for all combinations...")
            run_command(
                ["python", "train_with_tuning.py", "--all", "--output-dir", "results_nctd_dataset"] + dataset_flag + [loading_flag],
                "TRAINING: Use tuned hyperparameters for CNN, both methods, all datasets, output to results_nctd_dataset",
                auto_confirm=auto_yes
            )
        elif choice == "11":
            # Full tune BOTH models for both methods and all datasets, then train with tuned params, output to results_nctd_dataset
            print_section("FULL TUNE BOTH MODELS (BOTH METHODS, ALL DATASETS) + TRAIN [NEW]")
            parallel_flags = get_parallel_flags("4")
            auto_yes = input("\nAuto-confirm all (y/n)? ").strip().lower() == 'y'
            methods = ["ours", "NCTD"]
            models = ["nctd_cnn", "efficientnet"]
            
            ours_dir = os.path.join("2d_nctd_datasets", "ours")
            if not os.path.exists(ours_dir):
                print(f"Error: {ours_dir} not found!")
                continue
                
            datasets = [d for d in os.listdir(ours_dir) if os.path.isdir(os.path.join(ours_dir, d))]
            
            for model in models:
                for method in methods:
                    for dataset in datasets:
                        print(f"\nTuning: Model={model}, Method={method}, Dataset={dataset}")
                        run_command(
                            ["python", "hp_tuning.py", "--model", model, "--method", method, "--dataset", dataset, "--n-trials", "100", "--output-dir", "results_nctd_dataset"] + parallel_flags + dataset_flag + [loading_flag],
                            f"HP TUNING: {model.upper()}, Method={method}, Dataset={dataset}, 100 trials, output to results_nctd_dataset",
                            auto_confirm=auto_yes
                        )
            
            print("\nStarting training with tuned parameters for all combinations...")
            run_command(
                ["python", "train_with_tuning.py", "--all", "--output-dir", "results_nctd_dataset"] + dataset_flag + [loading_flag],
                "TRAINING: Use tuned hyperparameters for BOTH MODELS, both methods, all datasets, output to results_nctd_dataset",
                auto_confirm=auto_yes
            )
        elif choice == "7":
            print_section("ALL-IN-ONE WORKFLOW")
            print("""
This will run the complete workflow:
  1. HP Tuning for Method 'ours' (100 trials per model)
  2. HP Tuning for Method 'NCTD' (100 trials per model)
  3. Training with tuned hyperparameters (All Combinations)
  4. Comparison

Total estimated time: 8-14 hours
""")
            parallel_flags = get_parallel_flags("4")
            print("\n[STEP 1] Starting HP Tuning (Method: ours)...")
            if not run_command(
                ["python", "hp_tuning.py", "--all", "--method", "ours", "--n-trials", "100"] + parallel_flags + dataset_flag + [loading_flag],
                "STEP 1/4: HP TUNING (Method: ours, 2D NCTD)"
            ):
                print("HP Tuning failed. Aborting workflow.")
                continue
            print("\n[STEP 2] Starting HP Tuning (Method: NCTD)...")
            if not run_command(
                ["python", "hp_tuning.py", "--all", "--method", "NCTD", "--n-trials", "100"] + parallel_flags + dataset_flag + [loading_flag],
                "STEP 2/4: HP TUNING (Method: NCTD, 2D NCTD)"
            ):
                print("HP Tuning failed. Aborting workflow.")
                continue
            print("\n[STEP 3] Starting Training with Tuned Hyperparameters...")
            if not run_command(
                ["python", "train_with_tuning.py", "--all"] + dataset_flag + [loading_flag],
                "STEP 3/4: TRAINING WITH TUNING (2D NCTD)"
            ):
                print("Training failed. Aborting workflow.")
                continue
            print("\n[STEP 4] Generating Comparison Report...")
            if not run_command(
                ["python", "compare_metrics.py"] + dataset_flag + [loading_flag],
                "STEP 4/4: COMPARISON & VISUALIZATION (2D NCTD)"
            ):
                print("Comparison failed (non-critical).")
            print_header("ALL-IN-ONE WORKFLOW COMPLETED (2D NCTD)")
            print("""
Results saved to:
  • HP Tuning: results/hp_tuning/
  • Training: results/tuned_results/
  • Comparison: results/comparison/

Next steps:
  1. Review hp_tuning/reports/hp_tuning_report.txt
  2. Check tuned_results/ for trained models
  3. View comparison/ for visualizations
""")
        elif choice == "8":
            run_command(
                ["python", "compare_metrics.py"] + dataset_flag + [loading_flag],
                "COMPARISON: Generate visualizations & tables (2D NCTD)"
            )
        elif choice == "9":
            print_header("HP TUNING COMPREHENSIVE GUIDE")
            try:
                with open("HP_TUNING_GUIDE.txt", "r") as f:
                    print(f.read())
            except FileNotFoundError:
                print("HP_TUNING_GUIDE.txt not found!")
        elif choice == "0":
            print("\nGoodbye!")
            sys.exit(0)
        else:
            print(f"\nInvalid choice: {choice}")
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
