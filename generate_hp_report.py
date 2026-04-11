import os
import glob
import json
import numpy as np

def generate():
    os.chdir(r"c:\Users\Karthik Krishna\Desktop\Python ENV\1Dto2DConversion")
    results_dir = os.path.join("results", "hp_tuning")
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found.")
        return
        
    for model in ["efficientnet", "nctd_cnn"]:
        for method in ["ours", "NCTD"]:
            path = os.path.join(results_dir, model, method)
            if not os.path.exists(path): continue
            
            history_files = glob.glob(os.path.join(path, "tuning_history_*.json"))
            if not history_files: continue
            
            accs = []
            print(f"\n{'='*70}")
            print(f"Model: {model.upper()} | Method: {method.upper()}")
            print(f"{'='*70}")
            
            for hf in history_files:
                try:
                    ds_stem = os.path.basename(hf).replace("tuning_history_", "").replace(".json", "")
                    with open(hf, 'r') as f:
                        data = json.load(f)
                    
                    best_val = data.get("best_value", 0.0)
                    accs.append(best_val)
                    
                    params_file = os.path.join(path, f"best_params_{ds_stem}.json")
                    
                    if os.path.exists(params_file):
                        with open(params_file, 'r') as pf:
                            params = json.load(pf)
                        lr = params.get("learning_rate", 0)
                        opt = params.get("optimizer", "")
                        epochs = params.get("epochs", 0)
                        print(f"  {ds_stem:<35} Acc: {best_val:>7.4f}  (LR: {lr:.2e}, Opt: {opt}, Ep: {epochs})")
                    else:
                        print(f"  {ds_stem:<35} Acc: {best_val:>7.4f}")
                        
                except Exception as e:
                    print(f"  Error reading {hf}: {e}")
            
            if accs:
                print(f"{'-'*70}")
                print(f"  MEAN ACCURACY: {np.mean(accs):.4f} over {len(accs)} datasets")
                print(f"{'-'*70}\n")

if __name__ == "__main__":
    generate()
