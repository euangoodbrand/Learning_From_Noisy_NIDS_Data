import csv
from pathlib import Path
from collections import defaultdict

def read_csv(file_path):
    data = defaultdict(lambda: defaultdict(dict))
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Technique'] != 'E_ANRN':  # Exclude E_ANRN
                technique = row['Technique']
                noise_level = row['Noise Level']
                imbalance_ratio = row['Imbalance Ratio']
                key = noise_level if noise_level != '0.0' else imbalance_ratio
                data[technique][key] = row
    return data

def format_value(value):
    if value in ['N/A', '-']:
        return '-'
    try:
        return f"{float(value):.2f}"
    except ValueError:
        return value

def abbreviate_technique(technique):
    abbreviations = {
        'GeneralisedCrossEntropy': 'GCE',
        'EarlyLearningRegularisation': 'ELR',
        'LikelihoodRatioTest': 'LRT',
        'Bootstrapping_hard': 'Bootstrap (hard)',
        'Bootstrapping_soft': 'Bootstrap (soft)'
    }
    return abbreviations.get(technique, technique)

def categorize_technique(technique):
    if technique in ['Baseline', 'morse']:
        return "Baseline"
    elif technique in ['CoTeaching', 'MentorMix']:
        return "Sample Selection"
    elif technique.startswith('Bootstrapping') or technique == 'LikelihoodRatioTest':
        return "Label Sanitisation"
    elif technique in ['GeneralisedCrossEntropy', 'EarlyLearningRegularisation']:
        return "Loss Robustification"
    elif technique in ['NoiseAdaptation', 'LIO']:
        return "Noise Matrix Estimation"
    else:
        return "Other"

def generate_latex_table(data, experiment_num):
    if experiment_num == '2':
        columns = ['0.0', '0.1', '0.3', '0.6']
        column_type = 'Noise Level'
    elif experiment_num == '3':
        columns = ['0.0', '0.05', '0.01']
        column_type = 'Imbalance Ratio'
    else:
        raise ValueError(f"Unsupported experiment number: {experiment_num}")

    latex = r"\begin{table}[ht]" + "\n"
    latex += r"\centering" + "\n"
    latex += r"\caption{Results for Experiment " + str(experiment_num) + "}" + "\n"
    latex += r"\resizebox{\textwidth}{!}{%" + "\n"
    latex += r"\begin{tabular}{l" + "cccc" * len(columns) + "}" + "\n"
    latex += r"\hline" + "\n"
    
    # Header row
    latex += r"\textbf{Technique}"
    for col in columns:
        latex += f" & \multicolumn{{4}}{{c}}{{\\textbf{{{column_type} {col}}}}}"
    latex += r" \\" + "\n"
    
    latex += r"\cline{2-" + str(len(columns) * 4 + 1) + "}" + "\n"
    latex += r"\textbf{ } & " + " & ".join([r"\textbf{Acc}", r"\textbf{Prec}", r"\textbf{Rec}", r"\textbf{F1}"] * len(columns)) + r" \\" + "\n"
    latex += r"\hline" + "\n"

    # Custom sorting function
    def sort_key(x):
        if x in ['Baseline', 'morse']:
            return ('0', x)
        else:
            return (categorize_technique(x), x)

    techniques = sorted(data.keys(), key=sort_key)
    
    # Handle all techniques
    current_category = ""
    for technique in techniques:
        if technique not in ['Baseline', 'morse'] and categorize_technique(technique) != current_category:
            category = categorize_technique(technique)
            latex += r"\hline" + "\n"
            latex += r"\multicolumn{" + str(len(columns) * 4 + 1) + r"}{l}{\textbf{" + category + r"}} \\" + "\n"
            latex += r"\hline" + "\n"
            current_category = category

        latex += abbreviate_technique(technique)
        for col in columns:
            row_data = data[technique].get(col, {})
            latex += f" & {format_value(row_data.get('Accuracy', '-'))}"
            latex += f" & {format_value(row_data.get('Precision', '-'))}"
            latex += f" & {format_value(row_data.get('Recall', '-'))}"
            latex += f" & {format_value(row_data.get('F1', '-'))}"
        latex += r" \\" + "\n"

    latex += r"\hline" + "\n"
    latex += r"\end{tabular}" + "\n"
    latex += r"}" + "\n"
    latex += r"\end{table}" + "\n"

    return latex

def main():
    input_dir = Path('output_label')
    output_dir = Path('latex_output')
    output_dir.mkdir(exist_ok=True)

    for csv_file in input_dir.glob('experiment_*_results.csv'):
        experiment_num = csv_file.stem.split('_')[1]
        print(f"Processing experiment {experiment_num}")
        data = read_csv(csv_file)
        latex_table = generate_latex_table(data, experiment_num)

        output_file = output_dir / f'experiment_{experiment_num}_table.tex'
        with open(output_file, 'w') as f:
            f.write(latex_table)

        print(f"LaTeX table for Experiment {experiment_num} written to {output_file}")

if __name__ == "__main__":
    main()