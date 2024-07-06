import pandas as pd
import os
import numpy as np

# Read the CSV file
df = pd.read_csv('mnt/data/New Experiments - Combined (1).csv')

# Define the experiments and their corresponding noise levels
experiments = {
    '9': ['0.0', '0.3', '0.6', '1.0'],
    '10': ['0.0', '0.3', '0.6', '1.0'],
    '11': ['Additive 0.0, Multiplicative 0.0', 'Additive 0.0, Multiplicative 0.3', 'Additive 0.0, Multiplicative 0.6', 'Additive 0.0, Multiplicative 1.0'],
    '12': ['label 0.0, additive 0.0', 'label 0.0, additive 0.3', 'label 0.0, additive 0.6', 'label 0.0, add 1.0'],
    '13': ['label 0.0, Multiplicative 0.0', 'label 0.0, Multiplicative 0.3', 'label 0.0, Multiplicative 0.6', 'label 0.0, Multiplicative 1.0'],
    '14': ['0.0', '0.3', '0.6', '1.0'],
    '15': ['L2, Multiplicative 0.0'],
    '16': ['L2, Additive 0.0, Multiplicative 0.0'],
    '17': ['L2, label 0.0, add  0.0'],
    '18': ['L2, label 0.0, Multiplicative  0.0']
}

def safe_float_convert(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

# Function to clean and format experiment data
def clean_experiment_data(exp_data, noise_levels):
    cleaned_data = []
    for i, row in exp_data.iterrows():
        technique = row.iloc[0]
        if pd.notna(technique) and technique != 'Technique':
            for j, noise in enumerate(noise_levels):
                if j*2 + 2 < len(row):
                    accuracy = safe_float_convert(row.iloc[j*2 + 1])
                    f1 = safe_float_convert(row.iloc[j*2 + 2])
                    if not np.isnan(accuracy) and not np.isnan(f1):
                        cleaned_data.append({
                            'Technique': technique,
                            'Noise Level': noise,
                            'Accuracy': accuracy,
                            'F1': f1
                        })
                else:
                    print(f"Warning: Not enough columns for noise level {noise}")
    return pd.DataFrame(cleaned_data)

# Process each experiment
for exp_num, noise_levels in experiments.items():
    print(f"\nProcessing Experiment {exp_num}")
    exp_data = df.iloc[:, df.columns.get_loc(f'Experiment {exp_num}'):df.columns.get_loc(f'Experiment {int(exp_num)+1}') if int(exp_num) < 18 else len(df.columns)]
    
    print(f"Shape of exp_data: {exp_data.shape}")
    print(f"Columns of exp_data: {exp_data.columns}")
    print(f"First few rows of exp_data:")
    print(exp_data.head())
    
    print("\nUnique values in each column:")
    for col in exp_data.columns:
        print(f"{col}: {exp_data[col].unique()}")
    
    print("\nData types of columns:")
    print(exp_data.dtypes)
    
    cleaned_exp_data = clean_experiment_data(exp_data, noise_levels)
    
    if not cleaned_exp_data.empty:
        print("\nCleaned data:")
        print(cleaned_exp_data.head())
        print(f"\nShape of cleaned data: {cleaned_exp_data.shape}")
        
        # Save the cleaned data to a new CSV file
        output_file = f'mnt/data/experiment_{exp_num}_cleaned.csv'
        cleaned_exp_data.to_csv(output_file, index=False)
        print(f"Saved cleaned data to {output_file}")
    else:
        print(f"No data to save for Experiment {exp_num}")

print("\nAll experiments processed and saved.")