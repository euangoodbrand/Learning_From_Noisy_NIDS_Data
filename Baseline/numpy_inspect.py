import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

def inspect_and_compare(npz_file_path1, npz_file_path2):
    with np.load(npz_file_path1) as data1, np.load(npz_file_path2) as data2:
        y_train1 = data1['y_train']
        y_train2 = data2['y_train']

        # Calculate noise transition matrix
        num_classes = len(np.unique(y_train2))
        transition_matrix = np.zeros((num_classes, num_classes))

        for i in range(len(y_train1)):
            actual_label = y_train2[i]
            noisy_label = y_train1[i]
            transition_matrix[actual_label][noisy_label] += 1

        # Normalize the rows to create probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = transition_matrix / row_sums
        
        return transition_matrix

def save_and_visualize_matrix(transition_matrix, matrix_file_path, image_file_path):
    # Save the matrix to a file with high precision
    np.savetxt(matrix_file_path, transition_matrix, fmt='%.6f')

    # Define colors
    colors = ["#FFFFFF", "#B9F5F1", "#C8A8E2"]
    # Create a color map
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(transition_matrix, annot=True, fmt=".3f", cmap=cmap, annot_kws={"fontsize": 12}, linewidths=0)

    # Generate custom tick labels (arrow symbol) for the number of classes
    tick_labels = [chr(0x27A4) for _ in range(transition_matrix.shape[0])]
    
    ax.set_xticklabels(tick_labels, rotation=90)  # Set rotation for x-axis labels
    ax.set_yticklabels(tick_labels, rotation=0)  # Align y-axis labels
    ax.tick_params(left=False, bottom=False)

    # # Ensure the number of ticks match the number of classes
    # ax.xaxis.set_major_locator(plt.FixedLocator(np.arange(len(tick_labels))))
    # ax.yaxis.set_major_locator(plt.FixedLocator(np.arange(len(tick_labels))))

    plt.xlabel('Predicted', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=14, fontweight='bold')
    plt.title('Noise Transition Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save the figure
    if not os.path.exists(os.path.dirname(image_file_path)):
        os.makedirs(os.path.dirname(image_file_path))

    plt.savefig(image_file_path, bbox_inches='tight', dpi=300)
    plt.close()




# Usage
file1 = 'data/Windows_PE/real_world/malware.npz'
file2 = 'data/Windows_PE/real_world/malware_true.npz'
matrix = inspect_and_compare(file1, file2)
matrix_file_path = 'results/noise_transition_matrix.txt'
image_file_path = 'results/noise_transition_matrix.png'
save_and_visualize_matrix(matrix, matrix_file_path, image_file_path)
