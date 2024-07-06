import os
import csv
import re
from pathlib import Path
from collections import defaultdict

def extract_experiment_details(filename):
    details = {}
    parts = filename.split('_')
    for part in parts:
        if 'noise' in part:
            details['noise_level'] = part.split('noise')[-1]
        elif 'imbalance' in part:
            details['imbalance_ratio'] = part.split('imbalance')[-1]
        elif part in ['uniform', 'class', 'feature', 'MIMICRY']:
            details['noise_type'] = part
        elif part in ['undersampling', 'oversampling', 'smote', 'adasyn']:
            details['data_augmentation'] = part
        elif 'weight_resampling' in part:
            details['weight_resampling'] = part.split('weight_resampling')[-1]
    return details

def process_csv_file(file_path, technique, dataset, experiment_num):
    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                accuracy = float(row.get('accuracy', 0))
                precision = float(row.get('precision_macro', 0))
                recall = float(row.get('recall_macro', 0))
                f1 = float(row.get('f1_average', 0))
                
                details = extract_experiment_details(file_path.name)
                result = [technique, dataset, accuracy, precision, recall, f1]
                
                result.extend([
                    details.get('noise_level', 'N/A'),
                    details.get('imbalance_ratio', 'N/A'),
                    details.get('noise_type', 'N/A'),
                    details.get('data_augmentation', 'N/A'),
                    details.get('weight_resampling', 'N/A')
                ])
                
                print(f"Processed file: {file_path}")
                print(f"Result: {result}")
                return result
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return None

def process_directory(root_dir):
    results = defaultdict(lambda: defaultdict(list))
    all_techniques = set()
    processed_techniques = set()
    
    for technique_dir in root_dir.iterdir():
        if technique_dir.is_dir():
            technique = technique_dir.name
            if technique == 'E_ANRN':
                continue  # Skip E_ANRN
            all_techniques.add(technique)
            print(f"Processing technique: {technique}")
            
            results_dir = technique_dir / 'results'
            if not results_dir.exists():
                print(f"No results directory found for {technique}")
                continue
            
            # Group experiment folders
            exp_folders = defaultdict(list)
            for exp_dir in results_dir.iterdir():
                if exp_dir.name.startswith('experiment_'):
                    exp_num = exp_dir.name.split('_')[1].rstrip('$')
                    exp_folders[exp_num].append(exp_dir)
            
            for exp_num, folders in exp_folders.items():
                # Prefer '$' folder if it exists
                exp_dir = next((f for f in folders if f.name.endswith('$')), folders[0])
                
                try:
                    exp_num = int(exp_num)
                    if exp_num > 8:
                        continue  # Skip experiments beyond 8
                    
                    print(f"Processing experiment: {exp_dir.name}")
                    for dataset_dir in exp_dir.iterdir():
                        if dataset_dir.is_dir() and dataset_dir.name in ['BODMAS', 'windows_pe_real', 'CIC_IDS_2017']:
                            dataset = dataset_dir.name
                            print(f"Processing dataset: {dataset}")
                            
                            if technique == 'Bootstrapping' and dataset == 'BODMAS':
                                # Handle Bootstrap soft/hard for BODMAS
                                for bootstrap_type in ['hard', 'soft']:
                                    bootstrap_dir = dataset_dir / bootstrap_type
                                    if bootstrap_dir.exists():
                                        csv_files = list(bootstrap_dir.glob('*full_dataset*.csv'))
                                        for csv_file in csv_files:
                                            if 'validation' not in csv_file.name.lower():
                                                result = process_csv_file(csv_file, f'Bootstrapping_{bootstrap_type}', dataset, exp_num)
                                                if result:
                                                    key = (result[6], result[7])  # noise_level, imbalance_ratio
                                                    results[exp_num][key].append(result)
                                                    processed_techniques.add(f'Bootstrapping_{bootstrap_type}')
                            else:
                                # Search for CSV files in the dataset directory and its subdirectories
                                csv_files = list(dataset_dir.rglob('*full_dataset*.csv'))
                                print(f"Found {len(csv_files)} CSV files in {dataset_dir}")
                                
                                for csv_file in csv_files:
                                    if 'validation' not in csv_file.name.lower():
                                        print(f"Processing file: {csv_file}")
                                        
                                        result = process_csv_file(csv_file, technique, dataset, exp_num)
                                        if result:
                                            key = (result[6], result[7])  # noise_level, imbalance_ratio
                                            results[exp_num][key].append(result)
                                            processed_techniques.add(technique)
                except ValueError as e:
                    print(f"Error processing directory {exp_dir}: {e}")
    
    # For CoTeaching, keep only the first result for each noise/imbalance combination
    final_results = defaultdict(list)
    for exp_num, exp_results in results.items():
        for key, value_list in exp_results.items():
            if len(value_list) > 1 and value_list[0][0] == 'CoTeaching':
                final_results[exp_num].append(value_list[0])  # Keep only the first result
            else:
                final_results[exp_num].extend(value_list)
    
    print("All techniques found:", all_techniques)
    print("Techniques processed and saved:", processed_techniques)
    print("Missing techniques:", all_techniques - processed_techniques)
    
    return final_results

def write_results_to_csv(results, output_dir):
    for exp_num, data in results.items():
        if not data:
            print(f"No data to write for experiment {exp_num}")
            continue
        
        output_file = output_dir / f'experiment_{exp_num}_results.csv'
        
        try:
            # Sort the data based on Technique, Dataset, Noise Level, and Imbalance Ratio
            sorted_data = sorted(data, key=lambda x: (x[0], x[1], x[6], x[7]))
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                headers = ['Technique', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1',
                           'Noise Level', 'Imbalance Ratio', 'Noise Type', 'Data Augmentation', 'Weight Resampling']
                
                writer.writerow(headers)
                writer.writerows(sorted_data)
            
            print(f"Results for experiment {exp_num} written to {output_file}")
        except Exception as e:
            print(f"Error writing results to {output_file}: {e}")

def main():
    current_dir = Path.cwd()
    root_dir = current_dir / 'LabelNoiseLearning'
    output_dir = current_dir / 'output_label'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Starting processing of directory: {root_dir}")
    print(f"Output will be saved to: {output_dir}")
    
    results = process_directory(root_dir)
    write_results_to_csv(results, output_dir)
    
    print(f"Processing complete. All output files are in: {output_dir}")

if __name__ == "__main__":
    main()