import os
import csv
import re
from pathlib import Path

def extract_noise_levels(filename):
    match = re.search(r'add-noise(\d+\.\d+)_mult-noise(\d+\.\d+)', filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def process_csv_file(file_path, noise_type, technique):
    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                accuracy = float(row['accuracy'])
                f1 = float(row['f1_average'])
                precision_macro = float(row['precision_macro'])
                recall_macro = float(row['recall_macro'])
                add_noise, mult_noise = extract_noise_levels(file_path.name)
                
                if noise_type == 1:
                    return [add_noise, technique, accuracy, precision_macro, recall_macro, f1]
                elif noise_type == 2:
                    return [mult_noise, technique, accuracy, precision_macro, recall_macro, f1]
                else:
                    return [add_noise, mult_noise, technique, accuracy, precision_macro, recall_macro, f1]
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return None

def process_directory(root_dir):
    results = {1: [], 2: [], 3: []}
    all_techniques = set()
    processed_techniques = set()
    
    for technique_dir in root_dir.iterdir():
        if technique_dir.is_dir():
            technique = technique_dir.name
            all_techniques.add(technique)
            
            for path in technique_dir.rglob('experiment_*$'):
                if path.is_dir():
                    try:
                        experiment_num = int(path.name.split('_')[1][:-1])
                        bodmas_dir = path / 'BODMAS'
                        
                        if bodmas_dir.exists():
                            csv_files = list(bodmas_dir.rglob('*.csv'))
                            for csv_file in csv_files:
                                result = process_csv_file(csv_file, experiment_num, technique)
                                if result:
                                    results[experiment_num].append(result)
                                    processed_techniques.add(technique)
                    except ValueError as e:
                        print(f"Error processing directory {path}: {e}")
    
    print("All techniques found:", all_techniques)
    print("Techniques processed and saved:", processed_techniques)
    print("Missing techniques:", all_techniques - processed_techniques)
    
    return results

def write_results_to_csv(results, output_dir):
    for exp_num, data in results.items():
        if not data:
            print(f"No data to write for experiment {exp_num}")
            continue
        
        output_file = output_dir / f'experiment_{exp_num}_results.csv'
        
        try:
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                if exp_num == 1:
                    writer.writerow(['Additive Noise', 'Technique', 'Accuracy', 'Precision Macro', 'Recall Macro', 'F1'])
                elif exp_num == 2:
                    writer.writerow(['Multiplicative Noise', 'Technique', 'Accuracy', 'Precision Macro', 'Recall Macro', 'F1'])
                else:
                    writer.writerow(['Additive Noise', 'Multiplicative Noise', 'Technique', 'Accuracy', 'Precision Macro', 'Recall Macro', 'F1'])
                
                writer.writerows(data)
            
            print(f"Results for experiment {exp_num} written to {output_file}")
        except Exception as e:
            print(f"Error writing results to {output_file}: {e}")

def main():
    current_dir = Path.cwd()
    root_dir = current_dir / 'FeatureNoiseLearning'
    output_dir = current_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Starting processing of directory: {root_dir}")
    print(f"Output will be saved to: {output_dir}")
    
    results = process_directory(root_dir)
    write_results_to_csv(results, output_dir)
    
    print(f"Processing complete. All output files are in: {output_dir}")

if __name__ == "__main__":
    main()