import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV data
df = pd.read_csv('exp1_label_noise_results.csv')


# Convert noise and imbalance to strings for better labeling
df['noise_imbalance'] = df.apply(lambda row: f"Noise: {row['uniform_noise']}, Imb: {row['imbalance']}", axis=1)

# Create a new column that combines technique and top_folder
df['technique_folder'] = df['technique'] + ' (' + df['top_folder'] + ')'

# Remove rows with NaN or infinite values in f1_average
df = df[np.isfinite(df['f1_average'])]

# List of unique noise_imbalance combinations
noise_imbalance_combos = df['noise_imbalance'].unique()

# For each dataset
for dataset in df['dataset'].unique():
    plt.figure(figsize=(20, 15))
    
    # For each noise_imbalance combination
    for i, combo in enumerate(noise_imbalance_combos):
        plt.subplot(2, 3, i+1)
        
        # Filter data for this dataset and noise_imbalance combination
        data = df[(df['dataset'] == dataset) & (df['noise_imbalance'] == combo)]
        
        if not data.empty:
            # Sort by f1_average
            data = data.sort_values('f1_average', ascending=False)
            
            # Create bar plot
            ax = sns.barplot(x='technique_folder', y='f1_average', data=data, ci=None)
            
            plt.title(f'{dataset} - {combo}')
            plt.xticks(rotation=90)
            plt.ylim(0, 1)  # Assuming f1_average is between 0 and 1
            
            # Add value labels on the bars
            for j, v in enumerate(data['f1_average']):
                ax.text(j, v, f'{v:.2f}', ha='center', va='bottom', rotation=90)
            
            # Adjust x-axis labels
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        else:
            plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
            plt.title(f'{dataset} - {combo} (No Data)')
    
    plt.tight_layout()
    plt.savefig(f'{dataset}_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Graphs have been saved as PNG files.")