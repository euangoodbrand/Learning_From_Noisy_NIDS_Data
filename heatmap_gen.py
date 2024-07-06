import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define consistent orders
NOISE_ORDER = [0.0, 0.1, 0.3, 0.6]
IMBALANCE_ORDER = ['1x', '20x', '100x']
NOISE_IMBALANCE_ORDER = [f"{noise}_{imb}" for noise in NOISE_ORDER for imb in IMBALANCE_ORDER]

TECHNIQUE_ORDER = ['Baseline', 'morse', 'Bootstrapping_hard', 'Bootstrapping_soft', 'LikelihoodRatioTest',
                   'EarlyLearningRegularisation', 'GeneralisedCrossEntropy', 'LIO', 'NoiseAdaptation',
                   'CoTeaching', 'MentorMix']

def create_heatmap(df, title, filename, x_label='Noise Level_Imbalance Ratio'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt='.2f', cmap='viridis', 
                cbar_kws={'label': 'F1 Score'}, annot_kws={'size': 6})
    plt.title(title, fontsize=10)
    plt.xlabel(x_label, fontsize=8)
    plt.ylabel('Technique', fontsize=8)
    plt.xticks(rotation=45, ha='right', fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def experiment_4():
    df = pd.read_csv('output_label/experiment_4_results.csv')
    imbalance_mapping = {0.0: '1x', 0.05: '20x', 0.01: '100x'}
    df['Imbalance Ratio'] = df['Imbalance Ratio'].map(imbalance_mapping)
    df['Noise_Imbalance'] = df['Noise Level'].astype(str) + '_' + df['Imbalance Ratio']
    pivot_df = df.pivot_table(values='F1', index='Technique', columns='Noise_Imbalance', aggfunc='first')
    pivot_df = pivot_df.reindex(index=TECHNIQUE_ORDER, columns=NOISE_IMBALANCE_ORDER)
    create_heatmap(pivot_df, 'Experiment 4: F1 Scores Across Noise Levels and Imbalance Ratios', 'experiment_4_heatmap.png')

def experiment_5():
    df = pd.read_csv('output_label/experiment_5_results.csv')
    df['Noise_Imbalance'] = df['Noise Level'].astype(str) + '_0.0'
    pivot_df = df.pivot_table(values='F1', index='Technique', columns='Noise_Imbalance', aggfunc='first')
    pivot_df = pivot_df.reindex(columns=[f"{noise}_0.0" for noise in NOISE_ORDER])
    create_heatmap(pivot_df, 'Experiment 5: F1 Scores Across Noise Levels', 'experiment_5_heatmap.png', x_label='Noise Level')

def experiment_6():
    df = pd.read_csv('output_label/experiment_6_results.csv')
    imbalance_mapping = {0.0: '1x', 0.05: '20x', 0.01: '100x'}
    df['Imbalance Ratio'] = df['Imbalance Ratio'].map(imbalance_mapping)
    df['Noise_Imbalance'] = df['Noise Level'].astype(str) + '_' + df['Imbalance Ratio']
    pivot_df = df.pivot_table(values='F1', index='Technique', columns='Noise_Imbalance', aggfunc='first')
    pivot_df = pivot_df.reindex(columns=NOISE_IMBALANCE_ORDER)
    create_heatmap(pivot_df, 'Experiment 6: F1 Scores Across Noise Levels and Imbalance Ratios', 'experiment_6_heatmap.png')

def experiment_7():
    df = pd.read_csv('output_label/experiment_7_results.csv')
    
    # Print unique values in each column to check data
    print("Unique values in each column:")
    for col in df.columns:
        print(f"{col}: {df[col].unique()}")
    
    # Handle missing values in Data Augmentation column
    df['Data Augmentation'] = df['Data Augmentation'].fillna('None')
    
    # Convert noise level to string and create combined column
    df['Noise_Augmentation'] = df['Noise Level'].astype(str) + '_' + df['Data Augmentation']
    
    # Print unique combinations to check
    print("\nUnique Noise_Augmentation combinations:")
    print(df['Noise_Augmentation'].unique())
    
    # Group by Technique and Noise_Augmentation, and take the mean of F1 scores
    grouped_df = df.groupby(['Technique', 'Noise_Augmentation'])['F1'].mean().reset_index()
    
    # Create pivot table
    pivot_df = grouped_df.pivot(index='Technique', columns='Noise_Augmentation', values='F1')
    
    # Print pivot table shape and columns
    print(f"\nPivot table shape: {pivot_df.shape}")
    print(f"Pivot table columns: {pivot_df.columns}")
    
    # Define order for noise levels and augmentation techniques
    noise_levels = ['0.0', '0.1', '0.3', '0.6']
    augmentation_order = ['None', 'undersampling', 'oversampling', 'smote', 'adasyn']
    noise_aug_order = [f"{noise}_{aug}" for noise in noise_levels for aug in augmentation_order]
    
    # Reindex pivot table
    pivot_df = pivot_df.reindex(columns=noise_aug_order)
    
    # Print final pivot table shape and columns
    print(f"\nFinal pivot table shape: {pivot_df.shape}")
    print(f"Final pivot table columns: {pivot_df.columns}")
    
    # Create heatmap
    create_heatmap(pivot_df, 'Experiment 7: F1 Scores Across Noise Levels and Data Augmentation', 'experiment_7_heatmap.png', x_label='Noise Level_Data Augmentation')
def experiment_8():
    df = pd.read_csv('output_label/experiment_8_results.csv')
    imbalance_mapping = {0.0: '1x', 0.05: '20x', 0.01: '100x'}
    df['Imbalance Ratio'] = df['Imbalance Ratio'].map(imbalance_mapping)
    df['Noise_Imbalance'] = df['Noise Level'].astype(str) + '_' + df['Imbalance Ratio']
    pivot_df = df.pivot_table(values='F1', index='Technique', columns='Noise_Imbalance', aggfunc='first')
    pivot_df = pivot_df.reindex(columns=NOISE_IMBALANCE_ORDER)
    create_heatmap(pivot_df, 'Experiment 8: F1 Scores Across Noise Levels and Imbalance Ratios', 'experiment_8_heatmap.png')

# Run all experiments
experiment_4()
experiment_5()
experiment_6()
experiment_7()
experiment_8()

print("All heatmaps have been generated and saved with consistent ordering.")