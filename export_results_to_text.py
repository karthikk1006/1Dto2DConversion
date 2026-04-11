import os
import glob
import json

def generate_report():
    output_file = "all_results_combined.txt"
    base_dirs = {
        "Default": "results",
        "Tuned": os.path.join("results", "tuned_results")
    }
    
    records = []
    
    # metrics to extract
    metrics_keys = ["accuracy", "f1", "precision", "recall", "roc_auc", "mcc"]
    
    for mode, base_path in base_dirs.items():
        if not os.path.exists(base_path):
            continue
            
        for model in ["efficientnet", "cnn", "nctd_cnn"]:
            for method in ["nctd", "ours", "NCTD"]:
                metrics_dir = os.path.join(base_path, model, method, "metrics")
                if not os.path.exists(metrics_dir):
                    continue
                    
                target = os.path.join(metrics_dir, "*_metrics.json")
                for fpath in glob.glob(target):
                    filename = os.path.basename(fpath)
                    
                    # Extract dataset name
                    ds_name = filename.replace("_metrics.json", "")
                    prefixes = [
                        f"NCTD_{model}_", f"ours_{model}_", 
                        f"NCTD_efficientnet_", f"ours_efficientnet_",
                        f"NCTD_nctd_cnn_", f"ours_nctd_cnn_"
                    ]
                    for p in prefixes:
                        if ds_name.startswith(p):
                            ds_name = ds_name[len(p):]
                            break
                    
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        record = {
                            "Mode": mode,
                            "Model": 'EfficientNet' if 'efficientnet' in model.lower() else 'CNN',
                            "Method": 'Ours' if 'our' in method.lower() else 'NCTD',
                            "Dataset": ds_name
                        }
                        for k in metrics_keys:
                            record[k] = data.get(k, float('nan'))
                        records.append(record)
                    except Exception as e:
                        print(f"Error reading {fpath}: {e}")
                        
    if not records:
        print("No results found in 'results' or 'results/tuned_results'. Are you running this script inside the project folder?")
        return
        
    records.sort(key=lambda x: (x["Dataset"], x["Mode"], x["Model"], x["Method"]))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        header = f"{'Dataset':<35} | {'Mode':<10} | {'Model':<15} | {'Method':<10} | {'Accuracy':>10} | {'F1':>10} | {'AUC':>10} | {'Precision':>10} | {'Recall':>10} | {'MCC':>10}"
        f.write("=" * len(header) + "\n")
        f.write(header + "\n")
        f.write("=" * len(header) + "\n")
        
        for r in records:
            acc = f"{r['accuracy']:.4f}" if isinstance(r['accuracy'], float) and r['accuracy']==r['accuracy'] else "NaN"
            f1 = f"{r['f1']:.4f}" if isinstance(r['f1'], float) and r['f1']==r['f1'] else "NaN"
            prec = f"{r['precision']:.4f}" if isinstance(r['precision'], float) and r['precision']==r['precision'] else "NaN"
            rec = f"{r['recall']:.4f}" if isinstance(r['recall'], float) and r['recall']==r['recall'] else "NaN"
            auc = f"{r['roc_auc']:.4f}" if isinstance(r['roc_auc'], float) and r['roc_auc']==r['roc_auc'] else "NaN"
            mcc = f"{r['mcc']:.4f}" if isinstance(r['mcc'], float) and r['mcc']==r['mcc'] else "NaN"
            
            row = f"{r['Dataset']:<35} | {r['Mode']:<10} | {r['Model']:<15} | {r['Method']:<10} | {acc:>10} | {f1:>10} | {auc:>10} | {prec:>10} | {rec:>10} | {mcc:>10}"
            f.write(row + "\n")
        
        f.write("=" * len(header) + "\n")
        
    print(f"Extraction complete! Successfully exported {len(records)} test results to '{output_file}'.")

if __name__ == "__main__":
    generate_report()
