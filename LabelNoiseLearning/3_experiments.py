import os
import csv
import re
from collections import defaultdict
import pandas as pd

def extract_params(filename):
    params = {}
    match = re.search(r'(baseline|coTeaching_enesemble|mentorMix|mentorMix_ensemble|morse|generalisedCrossEntropy(?:_Ensemble)?(?:_Bootstrap)?(?:_Sampling)?)_(.+?)_no_augmentation_(?:uniform-noise([\d.]+)_imbalance([\d.]+)_addNoise([\d.]+)_multNoise([\d.]+)_)?(.+?)(?:_full_dataset)?(?:_model\d)?\.csv', filename)
    if match:
        params['technique'] = match.group(1)
        params['dataset'] = match.group(2)
        params['uniform_noise'] = float(match.group(3)) if match.group(3) else None
        params['imbalance'] = float(match.group(4)) if match.group(4) else None
        params['add_noise'] = float(match.group(5)) if match.group(5) else None
        params['mult_noise'] = float(match.group(6)) if match.group(6) else None
        params['extra_info'] = match.group(7)
    return params

def extract_metrics(filepath):
    accuracy = None
    f1_average = None
    try:
        with open(filepath, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if 'accuracy' in row:
                    accuracy = float(row['accuracy'])
                if 'f1_average' in row:
                    f1_average = float(row['f1_average'])
                if accuracy is not None and f1_average is not None:
                    break
    except Exception as e:
        print(f"Error reading file {filepath}: {str(e)}")
    return accuracy, f1_average

def process_csv_file(filepath, top_folder):
    filename = os.path.basename(filepath)
    params = extract_params(filename)
    accuracy, f1_average = extract_metrics(filepath)
    print(f"Processing file: {filename}")
    print(f"Extracted parameters: {params}")
    print(f"Extracted metrics - Accuracy: {accuracy}, F1 Average: {f1_average}")
    return {**params, 'top_folder': top_folder, 'accuracy': accuracy, 'f1_average': f1_average}

def process_dataset_folder(dataset_path, top_folder):
    results = []
    print(f"Processing dataset folder: {dataset_path}")
    for root, dirs, files in os.walk(dataset_path):
        if 'predictions' in dirs:
            dirs.remove('predictions')  # Don't process the 'predictions' folder
        for file in files:
            if file.endswith('.csv') and not file.endswith('validation.csv'):
                filepath = os.path.join(root, file)
                result = process_csv_file(filepath, top_folder)
                results.append(result)
    print(f"Found {len(results)} CSV files in {dataset_path}")
    return results


def write_results_to_csv(results, output_folder, experiment):
    print(f"Attempting to write results for {experiment}")
    print(f"Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder created/verified")
    
    output_file = os.path.join(output_folder, f'{experiment}_results.csv')
    print(f"Output file path: {output_file}")
    
    fieldnames = ['dataset', 'top_folder', 'technique', 'uniform_noise', 'imbalance', 'add_noise', 'mult_noise', 'extra_info', 'accuracy', 'f1_average']
    
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in results:
                writer.writerow(entry)
        
        print(f"Results for {experiment} have been written to {output_file}")
        print(f"Number of entries written: {len(results)}")
    except Exception as e:
        print(f"Error writing to file {output_file}: {str(e)}")

def remove_duplicates_and_sort(csv_file):
    print(f"Removing duplicates from and sorting {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Identify duplicate rows based on all columns except 'accuracy' and 'f1_average'
    duplicate_cols = [col for col in df.columns if col not in ['accuracy', 'f1_average']]
    
    # Sort by f1_average in descending order and keep the first occurrence (highest f1_average)
    df_sorted = df.sort_values('f1_average', ascending=False)
    df_deduped = df_sorted.drop_duplicates(subset=duplicate_cols, keep='first')
    
    # Sort the deduplicated dataframe
    df_sorted = df_deduped.sort_values(
        by=['dataset', 'technique', 'uniform_noise', 'imbalance', 'add_noise', 'mult_noise'],
        ascending=[True, True, True, True, True, True]
    )
    
    # Save the sorted and deduplicated dataframe back to CSV
    df_sorted.to_csv(csv_file, index=False)
    
    num_removed = len(df) - len(df_deduped)
    print(f"Removed {num_removed} duplicate rows from {csv_file}")
    print(f"Sorted the results and saved to {csv_file}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")
    
    output_folder = os.path.join(script_dir, 'experiment_results')
    print(f"Output folder path: {output_folder}")
    
    experiments = {
        'exp1_label_noise': [],
        'exp2_label_noise_imbalance_naive': [],
        'exp3_feature_label_noise_imbalance_naive': []
    }

    for top_folder in os.listdir(script_dir):
        top_folder_path = os.path.join(script_dir, top_folder)
        if os.path.isdir(top_folder_path):
            print(f"Processing top folder: {top_folder}")
            exp_path = os.path.join(top_folder_path, 'results', 'final_experiments')
            if not os.path.exists(exp_path):
                print(f"Path does not exist: {exp_path}")
                continue

            for exp in experiments.keys():
                exp_folder = os.path.join(exp_path, exp)
                if not os.path.exists(exp_folder):
                    print(f"Experiment folder does not exist: {exp_folder}")
                    continue

                print(f"Processing experiment: {exp}")
                for dataset in os.listdir(exp_folder):
                    dataset_path = os.path.join(exp_folder, dataset)
                    if os.path.isdir(dataset_path):
                        dataset_results = process_dataset_folder(dataset_path, top_folder)
                        experiments[exp].extend(dataset_results)

    # Write results to separate CSV files for each experiment
    for exp, results in experiments.items():
        write_results_to_csv(results, output_folder, exp)

# Remove duplicates and sort each experiment CSV file
    for exp in experiments.keys():
        csv_file = os.path.join(output_folder, f'{exp}_results.csv')
        remove_duplicates_and_sort(csv_file)

    print("Script execution completed.")

if __name__ == "__main__":
    main()