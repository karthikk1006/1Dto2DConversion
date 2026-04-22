import json
import re
import os
from pathlib import Path

def parse_dump_file(file_path):
    """Parses all_tuned_results_full_dump.md for CNN metrics and hyperparams."""
    try:
        content = Path(file_path).read_text(encoding='utf-16')
        print(f"Read {file_path} with utf-16")
    except:
        content = Path(file_path).read_text(encoding='utf-8')
        print(f"Read {file_path} with utf-8. Snippet: {content[:200]!r}")
    
    # Use re.finditer to find all dataset sections
    pattern = r'##\s*Dataset:\s*[`]+(.*?)[`]+(.*?)(?=##\s*Dataset:|\Z)'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    data = {}
    for match in matches:
        ds_name = match.group(1).strip()
        ds_content = match.group(2)
        print(f"  Processing dataset: {ds_name} (Content length: {len(ds_content)})")
        
        def extract_method_data(method_name):
            # Very loose pattern to find the header and the next two json blocks
            escaped_method = method_name.replace('+', r'\+')
            p = rf'###\s*{escaped_method}.*?```json\s*(.*?)\s*```.*?```json\s*(.*?)\s*```'
            m = re.search(p, ds_content, re.DOTALL)
            if m:
                try:
                    hp = json.loads(m.group(1))
                    metrics = json.loads(m.group(2))
                    return hp, metrics
                except Exception as e:
                    print(f"Error parsing JSON for {ds_name} {method_name}: {e}")
            else:
                print(f"    Failed to find {method_name} in {ds_name}")
            return None, None

        nctd_hp, nctd_metrics = extract_method_data('CNN + NCTD')
        ours_hp, ours_metrics = extract_method_data('CNN + Ours')
        
        if nctd_metrics or ours_metrics:
            data[ds_name] = {
                'nctd_hp': nctd_hp,
                'nctd_acc': nctd_metrics.get('accuracy', 0.0) if nctd_metrics else 0.0,
                'ours_hp': ours_hp,
                'ours_acc': ours_metrics.get('accuracy', 0.0) if ours_metrics else 0.0
            }
            
    return data

def parse_ds_folders(base_path):
    """Parses DS folders for CNN metrics and hyperparams."""
    data = {}
    ds_names = [f"DS{str(i).zfill(2)}" for i in range(1, 11)]
    
    # We need to find the full folder names like DS01_breast_cancer
    # We can list the ours folder
    ours_dir = Path(base_path) / 'ours'
    nctd_dir = Path(base_path) / 'NCTD'
    
    all_files = os.listdir(ours_dir)
    for ds_prefix in ds_names:
        # Find matching best_params and tuning_history
        # Actually tuning_history has the best_value we want
        history_file = next((f for f in all_files if f.startswith(f'tuning_history_{ds_prefix}')), None)
        if not history_file:
            continue
            
        ds_full_name = history_file.replace('tuning_history_', '').replace('.json', '')
        
        # Load Ours history/params
        with open(ours_dir / history_file, 'r') as f:
            ours_hist = json.load(f)
        params_file = ours_dir / f'best_params_{ds_full_name}.json'
        ours_hp = None
        if params_file.exists():
            with open(params_file, 'r') as f:
                ours_hp = json.load(f)
        
        # Load NCTD history/params
        nctd_hp = None
        nctd_acc = 0.0
        nctd_hist_file = nctd_dir / history_file
        if nctd_hist_file.exists():
            with open(nctd_hist_file, 'r') as f:
                nctd_hist = json.load(f)
            nctd_acc = nctd_hist.get('best_value', 0.0)
            nctd_params_file = nctd_dir / f'best_params_{ds_full_name}.json'
            if nctd_params_file.exists():
                with open(nctd_params_file, 'r') as f:
                    nctd_hp = json.load(f)
        
        data[ds_full_name] = {
            'nctd_hp': nctd_hp,
            'nctd_acc': nctd_acc,
            'ours_hp': ours_hp,
            'ours_acc': ours_hist.get('best_value', 0.0)
        }
        
    return data

def generate_report(merged_data, output_path):
    ours_list = []
    nctd_list = []
    equal_list = []
    error_list = []
    
    # Known error cases confirmed in logs
    error_datasets = {
        'DS05_thyroid': 'Stratified split failed (too few samples in class [5]).',
        'DS08_isolet': 'Crashed with CUDA unknown error after trial 0.',
        'DS09_madelon': 'Failed entirely with CUDA unknown error.',
        'DS10_relathee': 'Failed entirely with CUDA Out of Memory (OOM).'
    }

    # Sort keys for consistent order: Friedman, Rastrigin, Digen, DS
    def sort_key(name):
        if 'friedman' in name: return (0, name)
        if 'rastrigin' in name: return (1, name)
        if 'digen' in name: 
            # Extract number from digenXX_...
            match = re.search(r'digen(\d+)', name)
            num = int(match.group(1)) if match else 999
            return (2, num, name)
        if name.startswith('DS'): return (3, name)
        return (4, name)

    sorted_names = sorted(merged_data.keys(), key=sort_key)
    
    # Categorize
    for name in sorted_names:
        stats = merged_data[name]
        o_acc = stats['ours_acc']
        n_acc = stats['nctd_acc']
        
        if name in error_datasets:
            error_list.append(name)
        elif o_acc > n_acc + 0.0001:
            ours_list.append(name)
        elif n_acc > o_acc + 0.0001:
            nctd_list.append(name)
        else:
            equal_list.append(name)

    # Building lines
    lines = []
    lines.append("# CNN Accuracy Comparison Analysis (Ours vs NCTD)\n")
    lines.append("## 📊 Executive Summary & Dataset Tally\n")
    lines.append(f"This report compares the **Ours** approach against the **NCTD** baseline across **{len(merged_data)} total datasets**.\n")
    
    lines.append("### 📈 Global Performance Breakdown")
    lines.append(f"- **Ours-List ({len(ours_list)} datasets)**: Ours > NCTD")
    lines.append(f"- **NCTD-List ({len(nctd_list)} datasets)**: NCTD > Ours")
    lines.append(f"- **Equal-List ({len(equal_list)} datasets)**: Ours == NCTD (or < 0.1% difference)")
    lines.append(f"- **Error-List ({len(error_list)} datasets)**: Excluded due to technical failures in the NCTD pipeline.\n")

    # Tally Table
    digen_count = len([n for n in merged_data if 'digen' in n])
    ds_count = len([n for n in merged_data if n.startswith('DS')])
    friedman_count = len([n for n in merged_data if 'friedman' in n])
    rastrigin_count = len([n for n in merged_data if 'rastrigin' in n])
    
    lines.append("### 🔢 Total Dataset Tally")
    lines.append("| Category | Datasets Count | Names |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(f"| **Digen** | {digen_count} | `digen1` to `digen40` |")
    lines.append(f"| **DS (Benchmark)** | {ds_count} | `DS01` to `DS10` |")
    lines.append(f"| **Friedman** | {friedman_count} | `06_friedman1` to `08_friedman3` |")
    lines.append(f"| **Rastrigin** | {rastrigin_count} | `13_rotated_rastrigin_50d` |")
    lines.append(f"| **TOTAL** | **{len(merged_data)}** | |\n")
    
    lines.append("---\n")
    lines.append("## 🏆 Grouped Dataset Lists\n")
    lines.append("### 🚀 Ours-List (Superior Performance)")
    lines.append(", ".join([f"`{n}`" for n in ours_list]) + "\n")
    lines.append("### 📉 NCTD-List (Baseline Superiority)")
    lines.append(", ".join([f"`{n}`" for n in nctd_list]) + "\n")
    lines.append("### ⚖️ Equal-List (Tied/Marginal)")
    lines.append(", ".join([f"`{n}`" for n in equal_list]) + "\n")
    lines.append("### ⚠️ Error-List (Excluded due to Baseline Failures)")
    for name in error_list:
        lines.append(f"- **{name}**: {error_datasets.get(name, '')}")
    
    lines.append("\n---\n")
    lines.append("## 🔍 Detailed Comparison Report\n")
    lines.append("This section provides the side-by-side accuracy and hyperparameter logs for each dataset.\n")

    # Significant Improvements Section (Ours > NCTD)
    lines.append("## 🚀 Significant Improvements (Ours > NCTD)\n")
    for name in ours_list:
        stats = merged_data[name]
        diff = stats['ours_acc'] - stats['nctd_acc']
        lines.append(f"### Dataset: `{name}`")
        lines.append(f"- **Improvement**: `+{diff:.4f}`")
        lines.append(f"- **Ours Accuracy**: `{stats['ours_acc']:.4f}`")
        lines.append(f"- **NCTD Accuracy**: `{stats['nctd_acc']:.4f}`\n")
        lines.append(f"#### Hyperparameters (Ours):\n```json\n{json.dumps(stats['ours_hp'], indent=4)}\n```\n")
        lines.append(f"#### Hyperparameters (NCTD baseline):\n```json\n{json.dumps(stats['nctd_hp'], indent=4)}\n```\n")
        lines.append("---\n")

    # Others Section
    lines.append("\n## ⚖️ NCTD Better/Equal or Error Cases\n")
    remaining_names = sorted(nctd_list + equal_list + error_list, key=sort_key)
    for name in remaining_names:
        stats = merged_data[name]
        diff = stats['ours_acc'] - stats['nctd_acc']
        lines.append(f"### Dataset: `{name}`")
        if name in error_list:
            lines.append(f"> [!WARNING]")
            lines.append(f"> {error_datasets[name]}\n")
        else:
            lines.append(f"> [!NOTE]")
            lines.append(f"> Ours is NOT the best performer in this case.\n")
            
        lines.append(f"- **Difference**: `{diff:.4f}`")
        lines.append(f"- **Ours Accuracy**: `{stats['ours_acc']:.4f}`")
        lines.append(f"- **NCTD Accuracy**: `{stats['nctd_acc']:.4f}`\n")
        lines.append(f"#### Hyperparameters (Ours):\n```json\n{json.dumps(stats['ours_hp'], indent=4)}\n```\n")
        lines.append(f"#### Hyperparameters (NCTD winner/equal):\n```json\n{json.dumps(stats['nctd_hp'], indent=4)}\n```\n")
        lines.append("---\n")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

if __name__ == '__main__':
    dump_data = parse_dump_file('all_tuned_results_full_dump.md')
    print(f"Parsed {len(dump_data)} datasets from dump.")
    
    ds_data = parse_ds_folders('nctd_cnn')
    print(f"Parsed {len(ds_data)} datasets from DS folder.")
    
    # Merge
    merged = {**dump_data, **ds_data}
    print(f"Total merged datasets: {len(merged)}")
    
    generate_report(merged, 'cnn_improvement_analysis.md')
    print("Report generated successfully.")
