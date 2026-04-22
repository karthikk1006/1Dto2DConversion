import re
from pathlib import Path

def categorize_results(file_path):
    content = Path(file_path).read_text(encoding='utf-8')
    
    # Split by separator
    sections = content.split('---')
    
    ours_list = []
    nctd_list = []
    equal_list = []
    error_list = []
    
    # Error datasets we already identified
    error_names = ['DS05_thyroid', 'DS08_isolet', 'DS09_madelon', 'DS10_relathee']
    
    dataset_pattern = re.compile(r'### Dataset: `(.*?)`')
    acc_pattern = re.compile(r'- \*\*Difference\*\*: `(.*?)`')
    ours_acc_pattern = re.compile(r'- \*\*Ours Accuracy\*\*: `(.*?)`')
    nctd_acc_pattern = re.compile(r'- \*\*NCTD Accuracy\*\*: `(.*?)`')
    
    all_seen = set()

    for section in sections:
        match = dataset_pattern.search(section)
        if not match:
            continue
        
        name = match.group(1)
        all_seen.add(name)
        
        diff_match = acc_pattern.search(section)
        diff = float(diff_match.group(1)) if diff_match else 0.0
        
        ours_match = ours_acc_pattern.search(section)
        nctd_match = nctd_acc_pattern.search(section)
        
        ours_acc = float(ours_match.group(1)) if ours_match else 0.0
        nctd_acc = float(nctd_match.group(1)) if nctd_match else 0.0
        
        if name in error_names:
            error_list.append(name)
        elif ours_acc > nctd_acc:
            ours_list.append(name)
        elif nctd_acc > ours_acc:
            nctd_list.append(name)
        else:
            equal_list.append(name)
            
    return sorted(ours_list), sorted(nctd_list), sorted(equal_list), sorted(error_list), all_seen

if __name__ == '__main__':
    analysis_file = 'cnn_improvement_analysis.md'
    ours, nctd, equal, errors, seen = categorize_results(analysis_file)
    
    print(f"Ours-List ({len(ours)}): {', '.join(ours)}")
    print(f"NCTD-List ({len(nctd)}): {', '.join(nctd)}")
    print(f"Equal-List ({len(equal)}): {', '.join(equal)}")
    print(f"Error-List ({len(errors)}): {', '.join(errors)}")
    print(f"Total Seen: {len(seen)}")
