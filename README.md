# Learning_From_Noisy_NIDS_Data

![Image Noise](https://github.com/euangoodbrand/Learning_From_Noisy_NIDS_Data/raw/main/Assets/image_noise2_cleanup.png)


## Overview
This repository contains the code and datasets used in the research project focused on improving Network Intrusion Detection Systems (NIDS) through learning from noisy data. The project explores innovative techniques to address label noise, data imbalance, and concept drift in NIDS datasets. The objective is to develop robust models that are capable of performing accurately in adversarial environments typical of modern cybersecurity threats.



# Applying Co-teaching on NIDS Dataset

This repository contains a PyTorch implementation of all the techniques described and cited below.

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
python main.py --dataset cicids --model_type coteaching_plus --noise_type symmetric --noise_rate 0.2 data_augmentation none --seed 1 --num_workers 4 --result_dir results/trial_1/
```

## Customization

- `--lr`: Learning rate for the optimizer.
- `--noise_rate`: The simulated rate of label noise in the dataset.
- `--num_gradual`: Specifies how many epochs for linear drop rate.
- `--num_workers`: The number of subprocesses to use for data loading.
- Additional arguments are available in `main.py` for further customization.

## Citation

If you find this implementation helpful for your research, please consider citing the original papers:

```bash
@INPROCEEDINGS{10179453,
  author={Wu, Xian and Guo, Wenbo and Yan, Jia and Coskun, Baris and Xing, Xinyu},
  booktitle={2023 IEEE Symposium on Security and Privacy (SP)}, 
  title={From Grim Reality to Practical Solution: Malware Classification in Real-World Noise}, 
  year={2023},
  volume={},
  number={},
  pages={2602-2619},
  keywords={Training;Text mining;Privacy;Supervised learning;Training data;Semisupervised learning;Malware},
  doi={10.1109/SP46215.2023.10179453}}
```


### Co-Teaching

#### link https://arxiv.org/abs/1804.06872
```bash
@misc{han2018coteaching,
      title={Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels}, 
      author={Bo Han and Quanming Yao and Xingrui Yu and Gang Niu and Miao Xu and Weihua Hu and Ivor Tsang and Masashi Sugiyama},
      year={2018},
      eprint={1804.06872},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Co-Teaching + 

#### link: https://arxiv.org/abs/1901.04215
```bash

@misc{yu2019does,
      title={How does Disagreement Help Generalization against Label Corruption?}, 
      author={Xingrui Yu and Bo Han and Jiangchao Yao and Gang Niu and Ivor W. Tsang and Masashi Sugiyama},
      year={2019},
      eprint={1901.04215},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


### Mentor Mix

#### link : https://arxiv.org/pdf/1911.09781.pdfhttps://proceedings.mlr.press/v119/jiang20c/jiang20c.pdf
```bash

@inproceedings{jiang2020beyond,
  title={Beyond synthetic noise: Deep learning on controlled noisy labels},
  author={Jiang, L. and Huang, D. and Liu, M. and Yang, W.},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2020}
}
```

### Bootstrap

#### link: https://arxiv.org/abs/1412.6596
```

@misc{reed2015training,
      title={Training Deep Neural Networks on Noisy Labels with Bootstrapping}, 
      author={Scott Reed and Honglak Lee and Dragomir Anguelov and Christian Szegedy and Dumitru Erhan and Andrew Rabinovich},
      year={2015},
      eprint={1412.6596},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```

### LRT

#### Link : https://arxiv.org/pdf/2011.10077.pdf

```bash

@InProceedings{zheng2020error,
  title = 	 {Error-Bounded Correction of Noisy Labels},
  author =       {Zheng, Songzhu and Wu, Pengxiang and Goswami, Aman and Goswami, Mayank and Metaxas, Dimitris and Chen, Chao},
  booktitle = 	 {Proceedings of the 37th International Conference on Machine Learning},
  pages = 	 {11447--11457},
  year = 	 {2020},
  editor = 	 {III, Hal Daumé and Singh, Aarti},
  volume = 	 {119},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {13--18 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v119/zheng20c/zheng20c.pdf},
  url = 	 {https://proceedings.mlr.press/v119/zheng20c.html}
}

```


### GCE

#### link: https://arxiv.org/abs/1805.07836

```bash

@misc{zhang2018generalized,
      title={Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels}, 
      author={Zhilu Zhang and Mert R. Sabuncu},
      year={2018},
      eprint={1805.07836},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

```

### ELR

#### Link: https://arxiv.org/abs/2007.00151

```bash

@misc{liu2020earlylearning,
   title={Early-Learning Regularization Prevents Memorization of Noisy Labels}, 
      author={Sheng Liu and Jonathan Niles-Weed and Narges Razavian and Carlos Fernandez-Granda},
      year={2020},
      eprint={2007.00151},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


### Noise Adaption

#### Link: https://openreview.net/forum?id=H12GRgcxg

```bash
@inproceedings{
goldberger2017training,
title={Training deep neural-networks using a noise adaptation layer},
author={Jacob Goldberger and Ehud Ben-Reuven},
booktitle={International Conference on Learning Representations},
year={2017},
url={https://openreview.net/forum?id=H12GRgcxg}
}
```


### LIO 

#### Link: https://proceedings.mlr.press/v139/zhang21n.html

```bash
@InProceedings{pmlr-v139-zhang21n,
  title = 	 {Learning Noise Transition Matrix from Only Noisy Labels via Total Variation Regularization},
  author =       {Zhang, Yivan and Niu, Gang and Sugiyama, Masashi},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {12501--12512},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/zhang21n/zhang21n.pdf},
  url = 	 {https://proceedings.mlr.press/v139/zhang21n.html}
}


```

Additionally, if you utilize this adaptation for your research, please reference this repository and the dataset accordingly.


## Acknowledgments

This project is inspired by the work of Xiani Wu et al., on "From Grim Reality to Practical Solution: Malware Classification in Real-World Noise" Our adaptation focuses on the specific challenges posed by the NIDS dom
