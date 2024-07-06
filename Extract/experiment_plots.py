import os
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

NOISE_COLUMNS = ["Label Noise", "Additive Noise", "Multiplicative Noise"]
TECHNIQUE_ORDER = ["Baseline", "Bootstrapping Hard", "Bootstrapping Soft", "CoTeaching", "ELR", "GCE", "LRT", "LIO", "MentorMix", "Morse", "NoiseAdaptation"]

NOISE_LEVELS_MAPPING = {
    0.0: "None",
    0.1: "Very Low",
    0.3: "Low",
    0.6: "Medium",
    1.0: "High"
}

def process_csv(file_content):
    df = pd.read_csv(io.StringIO(file_content))
    actual_noise_columns = [col for col in NOISE_COLUMNS if col in df.columns]
    
    # Remove duplicate rows based on 'Technique' and noise columns
    df = df.drop_duplicates(subset=['Technique'] + actual_noise_columns)
    
    # Convert Accuracy and F1 Score columns to numeric, coercing errors to NaN
    df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')
    df['F1 Score'] = pd.to_numeric(df['F1 Score'], errors='coerce')
    
    # Map additive and multiplicative noise levels to descriptive labels
    for noise_col in actual_noise_columns:
        if noise_col in df.columns and noise_col != "Label Noise":
            df[noise_col] = df[noise_col].map(NOISE_LEVELS_MAPPING).fillna(df[noise_col])
    
    # Remove rows with NaN values in Accuracy or F1 Score
    df = df.dropna(subset=['F1 Score'])
    
    # Ensure consistent technique order
    df['Technique'] = pd.Categorical(df['Technique'], categories=TECHNIQUE_ORDER, ordered=True)
    df = df.sort_values('Technique')
    
    return df, actual_noise_columns

def create_heatmap(df, noise_columns, output_dir, filename):
    pivot_table = df.pivot_table(values='F1 Score', index='Technique', columns=noise_columns, aggfunc='mean')
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='viridis', linewidths=.5, cbar_kws={'label': 'F1 Score'}, annot_kws={'fontsize': 12})
    
    noise_types = ", ".join(noise_columns)
    plt.xlabel(f'Noise Levels ({noise_types})', fontsize=16)
    plt.ylabel('Technique', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    
    # Adjust colorbar font size
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=14)
    cbar.set_ylabel('F1 Score', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}_heatmap.png'), bbox_inches='tight', dpi=300)
    plt.close()

def process_directory(directory, output_dir):
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {file_path}")
            with open(file_path, 'r') as file:
                file_content = file.read()
            df, noise_columns = process_csv(file_content)
            
            if noise_columns:  # Check if there are any noise columns present
                create_heatmap(df, noise_columns, output_dir, filename)

# Main execution
one_noise_dir = r'mnt\data\one_noise_type'
two_noise_dir = r'mnt\data\two_noise_type'
output_dir = r'mnt\data\output'

os.makedirs(output_dir, exist_ok=True)

process_directory(one_noise_dir, output_dir)
process_directory(two_noise_dir, output_dir)

print("Heatmaps have been generated and saved in the output directory.")
