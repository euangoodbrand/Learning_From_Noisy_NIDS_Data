import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the CSV file
file_path = 'data/New Experiments - Combined (1).csv'
data = pd.read_csv(file_path)

# Create a directory to save the plots
output_dir = 'data/experiment_plots'
os.makedirs(output_dir, exist_ok=True)

# Clean the data and reformat it
experiments = {}
experiment_names = [col for col in data.columns if 'Experiment' in col]

# Experiment descriptions for better plot titles
experiment_titles = {
    '': 'Additive Noise Levels',
    '': 'Multiplicative Noise Levels',
    '': 'Additive and Multiplicative Noise',
    '': 'Label and Additive Noise',
    '': 'Label and Multiplicative Noise',
    '': 'Additive Noise with L2',
    '': 'Multiplicative Noise with L2',
    '': 'Additive and Multiplicative Noise with L2',
    '': 'Label and Additive Noise with L2',
    '': 'Label and Multiplicative Noise with L2'
}

# Loop through each experiment and extract data
for experiment_name in experiment_names:
    start_col = data.columns.get_loc(experiment_name)
    end_col = start_col + 4  # Assuming each experiment block spans 4 columns
    experiment_data = data.iloc[:, start_col:end_col]
    
    # Drop empty columns and rows
    experiment_data = experiment_data.dropna(axis=1, how='all').dropna(axis=0, how='all')
    
    # Check if the dataframe is empty after dropping NaNs
    if experiment_data.empty:
        continue
    
    # Set the first row as the header
    experiment_data.columns = experiment_data.iloc[0]
    experiment_data = experiment_data[1:]
    
    # Melt the dataframe for easier plotting
    experiment_data = experiment_data.melt(id_vars='Technique', var_name='Metric', value_name='Value')
    experiment_data['Value'] = pd.to_numeric(experiment_data['Value'], errors='coerce')
    
    experiments[experiment_name] = experiment_data.dropna()

# Function to calculate standard deviation
def calculate_std(df):
    std_df = df.groupby(['Technique', 'Metric'])['Value'].std().reset_index()
    std_df = std_df.rename(columns={'Value': 'Std'})
    return std_df

# Function to plot data
def plot_experiment_data(experiment_name, df):
    plt.figure(figsize=(10, 6))  # Standard figure size for all plots
    ax = sns.barplot(data=df, x='Technique', y='Value', hue='Metric', ci=None, palette='viridis')
    
    # Add error bars
    for i in range(len(ax.patches)):
        bar = ax.patches[i]
        width = bar.get_width()
        height = bar.get_height()
        x = bar.get_x()
        y = bar.get_y()
        technique = df['Technique'].unique()[i // 2]  # Assumes two metrics
        metric = df['Metric'].unique()[i % 2]
        std = df[(df['Technique'] == technique) & (df['Metric'] == metric)]['Std'].values[0]
        ax.errorbar(x + width / 2, height, yerr=std, fmt='none', c='black', capsize=5)
    
    plt.title(f'Performance Metrics for {experiment_titles.get(experiment_name, experiment_name)}', fontsize=16)
    plt.xlabel('Technique', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(rotation=45, fontsize=12, ha='right')  # Adjusted rotation and alignment for readability
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)  # Ensuring the y-axis starts at 0
    plt.legend(title='Metric', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{experiment_titles.get(experiment_name, experiment_name)}.png')
    plt.close()

# Process and plot each experiment
for experiment_name, df in experiments.items():
    try:
        if df.empty:
            print(f"No valid data for {experiment_name}. Skipping...")
            continue
        std_data = calculate_std(df)
        df = df.merge(std_data, on=['Technique', 'Metric'])
        plot_experiment_data(experiment_name, df)
    except Exception as e:
        print(f"Error processing {experiment_name}: {e}")

# Output the directory where plots are saved
output_dir
