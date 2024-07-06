import csv
import re
import os
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def read_csv(file_path):
    data = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            noise_key = 'Additive Noise' if 'Additive Noise' in row else 'Multiplicative Noise'
            noise_level = float(row[noise_key])
            technique = row['Technique'].lower()
            accuracy = float(row['Accuracy'])
            precision = float(row['Precision Macro'])
            recall = float(row['Recall Macro'])
            f1 = float(row['F1'])
            if technique not in data:
                data[technique] = {}
            data[technique][noise_level] = (accuracy, precision, recall, f1)
    logging.debug(f"Data read from {file_path}: {data}")
    return data

def format_value(value):
    return f"{value:.2f}"

def populate_latex_table(template, data, caption):
    lines = template.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith(r'\caption{'):
            lines[i] = r'\caption{' + caption + r'}'
        elif ' & ' in line and not line.startswith(r'\multirow'):
            parts = line.split('&')
            method = parts[0].strip().lower()
            if method in data:
                values = []
                for noise_level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                    if noise_level in data[method]:
                        acc, prec, rec, f1 = data[method][noise_level]
                        values.extend([format_value(acc), format_value(prec), format_value(rec), format_value(f1)])
                    else:
                        values.extend(['0.00', '0.00', '0.00', '0.00'])
                new_line = f"{parts[0].strip()} & " + " & ".join(values) + r" \\"
                lines[i] = new_line
            else:
                logging.warning(f"Method {parts[0].strip()} not found in data")

    return '\n'.join(lines)

def create_latex_template():
    template = r"""
\begin{table}[ht]
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|cccc|cccc|cccc|cccc|cccc|cccc}
\hline
\multirow{2}{*}{Method} & \multicolumn{4}{c|}{Noise Level 0} & \multicolumn{4}{c|}{Noise Level 0.2} & \multicolumn{4}{c|}{Noise Level 0.4} & \multicolumn{4}{c|}{Noise Level 0.6} & \multicolumn{4}{c|}{Noise Level 0.8} & \multicolumn{4}{c}{Noise Level 1.0} \\ \cline{2-25}
 & Acc & Prec & Rec & F1 & Acc & Prec & Rec & F1 & Acc & Prec & Rec & F1 & Acc & Prec & Rec & F1 & Acc & Prec & Rec & F1 & Acc & Prec & Rec & F1 \\ \hline
\multicolumn{25}{l}{\textbf{Base Models}} \\ \hline
FeatureBaseline & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 \\
Autoencoder & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 \\ \hline
\multicolumn{25}{l}{\textbf{Regularization Techniques}} \\ \hline
Dropout & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 \\
ElasticNetReg & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 \\
L1 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 \\
L2 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 \\
WeightDecay & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 \\
MaxNormReg & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 \\
NoiseInjection & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 \\ \hline
\multicolumn{25}{l}{\textbf{Data Augmentation Techniques}} \\ \hline
Mixup & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 \\
ManifoldMixup & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 \\
NoisyMix & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 \\
NFM & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 \\ \hline
\multicolumn{25}{l}{\textbf{Smoothing Techniques}} \\ \hline
LabelSmoothing & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 \\ \hline
\multicolumn{25}{l}{\textbf{Dimensionality Reduction Techniques}} \\ \hline
PCA & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 \\ \hline
\end{tabular}
}
\caption{Noise handling techniques on BODMAS dataset with varying noise levels}
\label{tab:noise-results}
\end{table}
    """
    return template

# Main execution
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('latex_output', exist_ok=True)

    # Create LaTeX template
    template = create_latex_template()

    # Read CSV data
    additive_data = read_csv(os.path.join('output', 'experiment_1_results.csv'))
    multiplicative_data = read_csv(os.path.join('output', 'experiment_2_results.csv'))

    # Populate tables
    white_noise_table = populate_latex_table(template, additive_data, 
                                             "White Noise (Additive) handling techniques on BODMAS dataset with varying noise levels")
    salt_pepper_noise_table = populate_latex_table(template, multiplicative_data, 
                                                   "Salt and Pepper Noise (Multiplicative) handling techniques on BODMAS dataset with varying noise levels")

    # Write results to files
    with open('latex_output/white_noise_table.tex', 'w') as f:
        f.write(white_noise_table)

    with open('latex_output/salt_pepper_noise_table.tex', 'w') as f:
        f.write(salt_pepper_noise_table)

    print("LaTeX tables have been populated and saved to 'latex_output' directory:")
    print("- white_noise_table.tex (Additive Noise)")
    print("- salt_pepper_noise_table.tex (Multiplicative Noise)")