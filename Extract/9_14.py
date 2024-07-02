import pandas as pd
import matplotlib.pyplot as plt
import os

# Reload the CSV file
file_path = 'mnt/data/New Experiments - Combined (1).csv'
data = pd.read_csv(file_path)

# Function to clean and reformat data
def clean_and_reformat_data(data):
    experiments = {}
    experiment_names = [col for col in data.columns if 'Experiment' in col]

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
        
        # Extract noise levels from the column names
        noise_levels = experiment_name.split(' ')[1:]
        noise_level_str = ' '.join(noise_levels)
        
        # Melt the dataframe for easier plotting
        experiment_data = experiment_data.melt(id_vars='Technique', var_name='Metric', value_name='Value')
        experiment_data['Value'] = pd.to_numeric(experiment_data['Value'], errors='coerce')
        
        # Create a new column for noise levels
        experiment_data['Noise Levels'] = noise_level_str
        
        experiments[experiment_name] = experiment_data.dropna()
    
    return experiments

experiments = clean_and_reformat_data(data)

# Create a directory to save the plots
output_dir = 'mnt/data/experiment_plots2'
os.makedirs(output_dir, exist_ok=True)

# Function to calculate standard deviation
def calculate_std(df):
    std_df = df.groupby(['Technique', 'Metric', 'Noise Levels'])['Value'].std().reset_index()
    std_df = std_df.rename(columns={'Value': 'Std'})
    return std_df

# Function to plot data
def plot_experiment_data(experiment_name, df):
    techniques = df['Technique'].unique()
    metrics = df['Metric'].unique()
    noise_levels = df['Noise Levels'].unique()

    fig, ax = plt.subplots(figsize=(14, 10))  # Standard figure size for all plots
    
    # Set positions for each technique
    positions = range(len(techniques))
    width = 0.2  # Width of each bar
    
    # Iterate over each combination of metric and noise level
    for i, metric in enumerate(metrics):
        for j, noise_level in enumerate(noise_levels):
            subset = df[(df['Metric'] == metric) & (df['Noise Levels'] == noise_level)]
            bar_positions = [p + width * (i * len(noise_levels) + j) for p in positions]
            if len(bar_positions) == len(subset['Value']):
                ax.bar(bar_positions, subset['Value'], width=width, label=f'{metric} - {noise_level}', alpha=0.7)
            
                # Add error bars
                for idx, row in subset.iterrows():
                    std = row['Std']
                    ax.errorbar(bar_positions[idx], row['Value'], yerr=std, fmt='none', c='black', capsize=5)
    
    ax.set_title(f'Performance Metrics for {experiment_name}', fontsize=16)
    ax.set_xlabel('Technique', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.set_xticks([p + width * (len(metrics) * len(noise_levels) - 1) / 2 for p in positions])
    ax.set_xticklabels(techniques, rotation=45, fontsize=12, ha='right')  # Adjusted rotation and alignment for readability
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylim(0, 1)  # Ensuring the y-axis starts at 0
    ax.legend(title='Metric - Noise Levels', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{experiment_name}.png')
    plt.close()

# Process and plot each experiment
for experiment_name, df in experiments.items():
    try:
        if df.empty:
            print(f"No valid data for {experiment_name}. Skipping...")
            continue
        std_data = calculate_std(df)
        df = df.merge(std_data, on=['Technique', 'Metric', 'Noise Levels'])
        plot_experiment_data(experiment_name, df)
    except Exception as e:
        print(f"Error processing {experiment_name}: {e}")

# Output the directory where plots are saved
output_dir
