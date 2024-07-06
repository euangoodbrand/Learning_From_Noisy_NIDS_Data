import os
import csv
import re
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_experiment_details(filename):
    details = {}
    parts = filename.split('_')
    
    # Initialize default values
    details['weight_resampling'] = 'None'
    details['data_augmentation'] = 'None'
    details['noise_type'] = 'N/A'
    details['noise_level'] = 'N/A'
    details['imbalance_ratio'] = 'N/A'
    details['add_noise'] = 'N/A'
    details['mult_noise'] = 'N/A'

    # Extract technique and dataset
    details['technique'] = parts[0]
    details['dataset'] = parts[1]

    # Extract data augmentation
    if 'no_augmentation' in parts:
        details['data_augmentation'] = 'None'
    else:
        for aug in ['undersampling', 'oversampling', 'smote', 'adasyn']:
            if aug in parts:
                details['data_augmentation'] = aug
                break

    # Extract noise type and level
    for part in parts:
        if '-noise' in part:
            noise_parts = part.split('-noise')
            details['noise_type'] = noise_parts[0]
            details['noise_level'] = noise_parts[1]

    # Extract imbalance ratio
    for part in parts:
        if 'imbalance' in part:
            details['imbalance_ratio'] = part.split('imbalance')[1]

    # Extract add and mult noise
    for part in parts:
        if 'addNoise' in part:
            details['add_noise'] = part.split('addNoise')[1]
        if 'multNoise' in part:
            details['mult_noise'] = part.split('multNoise')[1]

    # Extract weight resampling
    for part in parts:
        if part in ['Class-Balance', 'Focal', 'Naive']:
            details['weight_resampling'] = part
        elif part == 'no_weight_resampling':
            details['weight_resampling'] = 'None'

    return details

# The rest of the script remains the same

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
                
                logging.info(f"Processed file: {file_path}")
                logging.debug(f"Result: {result}")
                return result
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
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
            logging.info(f"Processing technique: {technique}")
            
            results_dir = technique_dir / 'results'
            if not results_dir.exists():
                logging.warning(f"No results directory found for {technique}")
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
                    
                    logging.info(f"Processing experiment: {exp_dir.name}")
                    for dataset_dir in exp_dir.iterdir():
                        if dataset_dir.is_dir() and dataset_dir.name in ['BODMAS', 'windows_pe_real', 'CIC_IDS_2017']:
                            dataset = dataset_dir.name
                            logging.info(f"Processing dataset: {dataset}")
                            
                            if technique == 'Bootstrapping':
                                # Handle Bootstrap soft/hard
                                for bootstrap_type in ['hard', 'soft']:
                                    bootstrap_dir = dataset_dir / bootstrap_type
                                    if bootstrap_dir.exists():
                                        csv_files = list(bootstrap_dir.glob('*full_dataset*.csv'))
                                        for csv_file in csv_files:
                                            if 'validation' not in csv_file.name.lower():
                                                result = process_csv_file(csv_file, f'Bootstrapping_{bootstrap_type}', dataset, exp_num)
                                                if result:
                                                    key = tuple(result[6:11])  # Use all experiment details as key
                                                    results[exp_num][key].append(result)
                                                    processed_techniques.add(f'Bootstrapping_{bootstrap_type}')
                            else:
                                # Search for CSV files in the dataset directory and its subdirectories
                                csv_files = list(dataset_dir.rglob('*full_dataset*.csv'))
                                logging.info(f"Found {len(csv_files)} CSV files in {dataset_dir}")
                                
                                for csv_file in csv_files:
                                    if 'validation' not in csv_file.name.lower():
                                        logging.debug(f"Processing file: {csv_file}")
                                        
                                        result = process_csv_file(csv_file, technique, dataset, exp_num)
                                        if result:
                                            key = tuple(result[6:11])  # Use all experiment details as key
                                            results[exp_num][key].append(result)
                                            processed_techniques.add(technique)
                except ValueError as e:
                    logging.error(f"Error processing directory {exp_dir}: {e}")
    
    # For CoTeaching, keep only the first result for each combination
    final_results = defaultdict(list)
    for exp_num, exp_results in results.items():
        for key, value_list in exp_results.items():
            if len(value_list) > 1 and value_list[0][0] == 'CoTeaching':
                final_results[exp_num].append(value_list[0])  # Keep only the first result
            else:
                final_results[exp_num].extend(value_list)
    
    logging.info("All techniques found: %s", all_techniques)
    logging.info("Techniques processed and saved: %s", processed_techniques)
    logging.info("Missing techniques: %s", all_techniques - processed_techniques)
    
    return final_results

def write_results_to_csv(results, output_dir):
    for exp_num, data in results.items():
        if not data:
            logging.warning(f"No data to write for experiment {exp_num}")
            continue
        
        output_file = output_dir / f'experiment_{exp_num}_results.csv'
        
        try:
            # Sort the data based on all columns
            sorted_data = sorted(data, key=lambda x: x)
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                headers = ['Technique', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1',
                           'Noise Level', 'Imbalance Ratio', 'Noise Type', 'Data Augmentation', 'Weight Resampling']
                
                writer.writerow(headers)
                writer.writerows(sorted_data)
            
            logging.info(f"Results for experiment {exp_num} written to {output_file}")
        except Exception as e:
            logging.error(f"Error writing results to {output_file}: {e}")

def main():
    current_dir = Path.cwd()
    root_dir = current_dir / 'LabelNoiseLearning'
    output_dir = current_dir / 'output_label'
    output_dir.mkdir(exist_ok=True)
    
    logging.info(f"Starting processing of directory: {root_dir}")
    logging.info(f"Output will be saved to: {output_dir}")
    
    results = process_directory(root_dir)
    write_results_to_csv(results, output_dir)
    
    logging.info(f"Processing complete. All output files are in: {output_dir}")

if __name__ == "__main__":
    main()