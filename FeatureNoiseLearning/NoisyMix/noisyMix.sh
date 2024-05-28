#!/bin/bash

num_workers=0
feature_add_noise_levels=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
feature_mult_noise_levels=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
imbalance_ratios=(0 0.05 0.01)
data_augmentations=("none" "undersampling" "oversampling" "smote" "adasyn")
weight_resamplings=("Class-Balance" "Focal" "Naive")

start_time=$(date +%s)

for seed in 1 #2 3 4 5 
do
  for model_type in noisyMix
  do
    # Experiment 1: Feature Noise (Additive)
    for add_noise_level in "${feature_add_noise_levels[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python noisyMix.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --feature_add_noise_level ${add_noise_level} --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_1$
    done

    # Experiment 2: Feature Noise (Multiplicative)
    for mult_noise_level in "${feature_mult_noise_levels[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python noisyMix.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --feature_mult_noise_level ${mult_noise_level} --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_2$
    done

    # Experiment 3: Feature Noise (Additive and Multiplicative)
    for add_noise_level in "${feature_add_noise_levels[@]}"; do
        for mult_noise_level in "${feature_mult_noise_levels[@]}"; do
            CUDA_LAUNCH_BLOCKING=1 python noisyMix.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --feature_add_noise_level ${add_noise_level} --feature_mult_noise_level ${mult_noise_level} --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_3$
        done
    done

  done
done

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Total time taken: $(($elapsed_time / 3600)) hours $(($elapsed_time % 3600 / 60)) minutes $(($elapsed_time % 60)) seconds"
