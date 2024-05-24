# Applying Co-teaching on NIDS Dataset

This repository contains a PyTorch implementation of the co-teaching and co-teaching+ methods adapted for a Network Intrusion Detection System (NIDS) dataset, inspired by the ICML'19 paper [How does Disagreement Help Generalization against Label Corruption?](https://arxiv.org/abs/1901.04215).

## Introduction

Label noise is a common issue in real-world datasets, which can significantly degrade the performance of deep learning models. The co-teaching strategy involves training two neural networks simultaneously, where each network learns to teach the other network to select and learn from the most reliable samples. This project extends the application of co-teaching to the domain of network intrusion detection, aiming to improve the robustness and generalization of models against label noise in NIDS datasets.

## Requirements

- Python 3.6+
- PyTorch 1.7.0+
- scikit-learn
- imbalanced-learn
- pandas
- numpy
- tqdm

## Dataset

The dataset used in this project is derived from [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), a comprehensive dataset for network intrusion detection. The dataset contains various types of attacks simulated in a testbed to mirror real-world data, alongside benign traffic for a balanced representation.

## Usage

To run the co-teaching+ model on the NIDS dataset, adjust the parameters as needed and execute the following command:

```bash
python main.py --dataset cicids --model_type coteaching_plus --noise_type symmetric --noise_rate 0.2 --seed 1 --num_workers 4 --result_dir results/trial_1/
```

## Customization

- `--lr`: Learning rate for the optimizer.
- `--noise_rate`: The simulated rate of label noise in the dataset.
- `--num_gradual`: Specifies how many epochs for linear drop rate.
- `--num_workers`: The number of subprocesses to use for data loading.
- Additional arguments are available in `main.py` for further customization.

## Citation

If you find this implementation helpful for your research, please consider citing the original paper:

```bash
@inproceedings{yu2019does,
  title={How does Disagreement Help Generalization against Label Corruption?},
  author={Yu, Xingrui and Han, Bo and Yao, Jiangchao and Niu, Gang and Tsang, Ivor and Sugiyama, Masashi},
  booktitle={International Conference on Machine Learning},
  pages={7164--7173},
  year={2019}
}
```

Additionally, if you utilize this adaptation for your research, please reference this repository and the dataset accordingly.


## Acknowledgments

This project is inspired by the work of Xingrui Yu et al., on leveraging disagreement for improving generalization in the presence of label noise. Our adaptation focuses on the specific challenges posed by the NIDS domain.

Feel free to adjust the content to better suit your project or presentation needs.
