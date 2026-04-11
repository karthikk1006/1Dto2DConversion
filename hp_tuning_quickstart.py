#!/usr/bin/env python3
"""
hp_tuning_quickstart.py
========================
One-command launcher for the complete hyperparameter tuning workflow.

This script provides an interactive menu to run the entire HP tuning pipeline
without needing to remember command-line syntax.

Usage
-----
  python hp_tuning_quickstart.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_section(text):
    """Print a formatted section."""
    print(f"\n{'─' * 80}")
    print(f"  {text}")
    print(f"{'─' * 80}")


def run_command(cmd, description):
    """Run a shell command with user confirmation."""
    print_section(description)
    print(f"\nCommand: {' '.join(cmd)}\n")
    
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
    """Helper to get n_jobs and GPU parallel flags."""
    n_jobs = input(f"\nNumber of parallel jobs (1-4, default {default_jobs}): ").strip() or default_jobs
    flags = ["--n-jobs", n_jobs]
    if int(n_jobs) > 1:
        allow_gpu = input("Allow parallel GPU jobs? (WARNING: may cause OOM) (y/N): ").strip().lower()
        if allow_gpu == 'y':
            flags.append("--allow-gpu-parallel")
    return flags


def show_menu():
    """Show main menu and return user choice."""
    print_header("HP TUNING QUICKSTART MENU")
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

[0] EXIT

""")
    choice = input("Enter choice [0-9]: ").strip()
    return choice


def main():
    """Main interactive menu."""
    os.chdir(Path(__file__).parent)
    
    print_header("HP TUNING WORKFLOW - INTERACTIVE LAUNCHER")
    
    print("""
This tool provides an interactive interface for running the hyperparameter
tuning workflow on your 1D-to-2D classification pipeline.

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
        
        if choice == "1":
            # Quick test (Added a default dataset to run on)
            run_command(
                ["python", "hp_tuning.py", "--model", "efficientnet", "--n-trials", "10", 
                 "--dataset", "13_rotated_rastrigin_50d.npz"],
                "QUICK TEST: EfficientNet - 10 trials on 1 dataset"
            )
        
        elif choice == "2":
            # Single model quick tune
            run_command(
                ["python", "hp_tuning.py", "--model", "efficientnet", 
                 "--n-trials", "50"],
                "SINGLE MODEL - QUICK TUNE: EfficientNet - 50 trials"
            )
        
        elif choice == "3":
            # Single model full tune
            run_command(
                ["python", "hp_tuning.py", "--model", "efficientnet",
                 "--n-trials", "100"],
                "SINGLE MODEL - FULL TUNE: EfficientNet - 100 trials"
            )
        
        elif choice == "4":
            # Both models quick tune for both methods
            parallel_flags = get_parallel_flags("2")
            run_command(
                ["python", "hp_tuning.py", "--all", "--method", "ours",
                 "--n-trials", "50"] + parallel_flags,
                "BOTH MODELS - QUICK TUNE (Method: ours): 50 trials each"
            )
            run_command(
                ["python", "hp_tuning.py", "--all", "--method", "NCTD",
                 "--n-trials", "50"] + parallel_flags,
                "BOTH MODELS - QUICK TUNE (Method: NCTD): 50 trials each"
            )
        
        elif choice == "5":
            # Both models full tune + training (recommended)
            print_section("BOTH MODELS - FULL TUNE (RECOMMENDED)")
            print("\nThis will tune all 4 combinations (2 models x 2 methods) and then train with best params.")
            parallel_flags = get_parallel_flags("4")
            
            print("\n[STEP 1] Tuning Method: ours...")
            if not run_command(
                ["python", "hp_tuning.py", "--all", "--method", "ours",
                 "--n-trials", "100"] + parallel_flags,
                "STEP 1/3: HP TUNING - Method: ours"
            ):
                print("HP Tuning ours failed. Aborting workflow.")
                continue

            print("\n[STEP 2] Tuning Method: NCTD...")
            if not run_command(
                ["python", "hp_tuning.py", "--all", "--method", "NCTD",
                 "--n-trials", "100"] + parallel_flags,
                "STEP 2/3: HP TUNING - Method: NCTD"
            ):
                print("HP Tuning NCTD failed. Aborting workflow.")
                continue

            print("\n[STEP 3] Starting Training and Evaluation with Tuned Hyperparameters...")
            if not run_command(
                ["python", "train_with_tuning.py", "--all"],
                "STEP 3/3: TRAINING WITH TUNING (All Combinations)"
            ):
                print("Training failed.")
                continue
                
            print("\nResults saved properly. Test results are stored in results/tuned_results/...")
        
        elif choice == "6":
            # Train with tuned hyperparams
            run_command(
                ["python", "train_with_tuning.py", "--all"],
                "TRAINING: Use tuned hyperparameters for both models"
            )
        
        elif choice == "7":
            # All-in-one workflow
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
                ["python", "hp_tuning.py", "--all", "--method", "ours",
                 "--n-trials", "100"] + parallel_flags,
                "STEP 1/4: HP TUNING (Method: ours)"
            ):
                print("HP Tuning failed. Aborting workflow.")
                continue

            print("\n[STEP 2] Starting HP Tuning (Method: NCTD)...")
            if not run_command(
                ["python", "hp_tuning.py", "--all", "--method", "NCTD",
                 "--n-trials", "100"] + parallel_flags,
                "STEP 2/4: HP TUNING (Method: NCTD)"
            ):
                print("HP Tuning failed. Aborting workflow.")
                continue
            
            print("\n[STEP 3] Starting Training with Tuned Hyperparameters...")
            if not run_command(
                ["python", "train_with_tuning.py", "--all"],
                "STEP 3/4: TRAINING WITH TUNING"
            ):
                print("Training failed. Aborting workflow.")
                continue
            
            print("\n[STEP 4] Generating Comparison Report...")
            if not run_command(
                ["python", "compare_metrics.py"],
                "STEP 4/4: COMPARISON & VISUALIZATION"
            ):
                print("Comparison failed (non-critical).")
            
            print_header("ALL-IN-ONE WORKFLOW COMPLETED")
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
            # Compare results
            run_command(
                ["python", "compare_metrics.py"],
                "COMPARISON: Generate visualizations & tables"
            )
        
        elif choice == "9":
            # View guide
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
