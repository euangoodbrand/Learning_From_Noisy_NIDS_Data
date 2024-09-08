#!/bin/bash

num_workers=0
noise_rates=(0 0.1 0.3 0.6)
imbalance_ratios=(0.01 0.05)
feature_noise_levels=(0 0.1 0.3 0.6)
weight_decay=0.01  # L2 regularization

for run in $(seq 1 $num_runs)
do
  for model_type in morse
  do
    # # Experiment 1: Just Label Noise
    # for noise_rate in "${noise_rates[@]}"; do
    #   CUDA_LAUNCH_BLOCKING=1 python morse.py \
    #     --dataset BODMAS \
    #     --model_type ${model_type} \
    #     --weight_decay ${weight_decay} \
    #     --data_augmentation none \
    #     --noise_rate ${noise_rate} \
    #     --noise_type uniform \
    #     --imbalance_ratio 0 \
    #     --seed $((run * 100)) \
    #     --num_workers ${num_workers} \
    #     --result_dir results/final_experiments/exp1_label_noise \
    #     --num_runs 1

    # done

    # # Experiment 2: Label Noise and Imbalance with Naive Resampling
    # for noise_rate in "${noise_rates[@]}"; do
    #   for imbalance_ratio in "${imbalance_ratios[@]}"; do
    #     CUDA_LAUNCH_BLOCKING=1 python morse.py \
    #       --dataset BODMAS \
    #       --model_type ${model_type} \
    #       --weight_decay ${weight_decay} \
    #       --data_augmentation none \
    #       --noise_rate ${noise_rate} \
    #       --noise_type uniform \
    #       --imbalance_ratio ${imbalance_ratio} \
    #       --seed $((run * 100)) \
    #       --num_workers ${num_workers} \
    #       --weight_resampling Naive \
    #       --result_dir results/final_experiments/exp2_label_noise_imbalance_naive \
    #       --num_runs 1
    #   done
    # done

    # Experiment 3: Feature Noise, Label Noise, and Imbalance with Naive Resampling
    for noise_rate in "${noise_rates[@]}"; do
      for imbalance_ratio in "${imbalance_ratios[@]}"; do
        for feature_noise in "${feature_noise_levels[@]}"; do
          CUDA_LAUNCH_BLOCKING=1 python morse.py \
            --dataset BODMAS \
            --model_type ${model_type} \
            --data_augmentation none \
            --noise_rate ${noise_rate} \
            --noise_type uniform \
            --imbalance_ratio ${imbalance_ratio} \
            --seed $((run * 100)) \
            --num_workers ${num_workers} \
            --feature_add_noise_level ${feature_noise} \
            --feature_mult_noise_level ${feature_noise} \
            --result_dir results/final_experiments/exp4_feature_noise \
            --num_runs 1
        done
      done
    done

  done
done

echo "All experiments completed."