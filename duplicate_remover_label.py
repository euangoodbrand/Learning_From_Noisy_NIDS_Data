import csv
import os
from collections import defaultdict
from pathlib import Path

def process_csv(file_path):
    # Read the CSV file
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # Group rows by technique, dataset, and experiment conditions
    grouped_data = defaultdict(list)
    for row in data:
        key = (
            row['Technique'],
            row['Dataset'],
            row['Noise Level'],
            row['Noise Type'],
            row['Data Augmentation'],
            row['Imbalance Ratio']
        )
        grouped_data[key].append(row)

    # Keep only the best performing row for each group
    best_rows = []
    for group in grouped_data.values():
        best_row = max(group, key=lambda x: float(x['F1']))
        best_rows.append(best_row)

    # Sort the results
    best_rows.sort(key=lambda x: (x['Technique'], x['Dataset'], float(x['Noise Level']), float(x['Imbalance Ratio'])))

    # Write the results back to the same CSV file
    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(best_rows)

    print(f"Processed and updated: {file_path}")

def process_directory(directory):
    directory_path = Path(directory)
    csv_files = list(directory_path.glob('*.csv'))

    if not csv_files:
        print(f"No CSV files found in {directory}")
        return

    for csv_file in csv_files:
        process_csv(csv_file)

def main():
    output_label_dir = 'output_label'  # Folder containing the CSV files
    process_directory(output_label_dir)

if __name__ == "__main__":
    main()