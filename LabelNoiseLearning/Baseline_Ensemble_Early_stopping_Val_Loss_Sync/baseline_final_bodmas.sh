#!/bin/bash

num_workers=0
noise_rates=(0.1 0.3 0.6)
imbalance_ratios=(0.01 0.05)
feature_noise_levels=(0.1 0.3 0.6)
num_runs=5

for seed in {1..1}
do
  for model_type in baseline
  do
    # Experiment 1: Just Label Noise
    for noise_rate in "${noise_rates[@]}"; do
      CUDA_LAUNCH_BLOCKING=1 python baseline.py \
        --dataset BODMAS \
        --model_type ${model_type} \
        --weight_decay 0.01 \
        --data_augmentation none \
        --noise_rate ${noise_rate} \
        --noise_type uniform \
        --imbalance_ratio 0 \
        --seed ${seed} \
        --num_workers ${num_workers} \
        --result_dir results/final_experiments/exp1_label_noise
    done

    # # Experiment 2: Label Noise and Imbalance with Naive Resampling
    # for noise_rate in "${noise_rates[@]}"; do
    #   for imbalance_ratio in "${imbalance_ratios[@]}"; do
    #     CUDA_LAUNCH_BLOCKING=1 python baseline.py \
    #       --dataset BODMAS \
    #       --model_type ${model_type} \
    #       --weight_decay 0.01 \
    #       --data_augmentation none \
    #       --noise_rate ${noise_rate} \
    #       --noise_type uniform \
    #       --imbalance_ratio ${imbalance_ratio} \
    #       --seed ${seed} \
    #       --num_workers ${num_workers} \
    #       --weight_resampling Naive \
    #       --result_dir results/final_experiments/exp2_label_noise_imbalance_naive
    #   done
    # done

    # # Experiment 3: Feature Noise, Label Noise, and Imbalance with Naive Resampling
    # for noise_rate in "${noise_rates[@]}"; do
    #   for imbalance_ratio in "${imbalance_ratios[@]}"; do
    #     for feature_noise in "${feature_noise_levels[@]}"; do
    #       CUDA_LAUNCH_BLOCKING=1 python baseline.py \
    #         --dataset BODMAS \
    #         --model_type ${model_type} \
    #         --weight_decay 0.01 \
    #         --data_augmentation none \
    #         --noise_rate ${noise_rate} \
    #         --noise_type uniform \
    #         --imbalance_ratio ${imbalance_ratio} \
    #         --seed ${seed} \
    #         --num_workers ${num_workers} \
    #         --feature_add_noise_level ${feature_noise} \
    #         --feature_mult_noise_level ${feature_noise} \
    #         --weight_resampling Naive \
    #         --result_dir results/final_experiments/exp3_feature_label_noise_imbalance_naive
    #     done
    #   done
    # done

  done
done