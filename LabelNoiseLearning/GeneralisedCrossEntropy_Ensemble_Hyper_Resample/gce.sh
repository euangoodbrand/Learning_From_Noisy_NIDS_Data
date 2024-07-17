#!/bin/bash

num_workers=0
noise_rates=(0 0.1 0.3 0.6)
imbalance_ratios=(0 0.05 0.01)
feature_add_noise_levels=(0.0 0.3 0.6 1.0)
feature_mult_noise_levels=(0.0 0.3 0.6 1.0)
noise_types=("uniform" "class" "feature" "MIMICRY")
data_augmentations=("none" "undersampling" "oversampling" "smote" "adasyn")
weight_resamplings=("Class-Balance" "Focal" "Naive")
num_runs=5

for seed in 1 #2 3 4 5 
do
  for model_type in generalisedCrossEntropy
  do

    # # Experiment 14: Additive Noise Only with L2 regularization
    # for add_noise_level in "${feature_add_noise_levels[@]}"; do
    #     CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_add_noise_level ${add_noise_level} --weight_decay 0.01 --result_dir results/experiment_14 --num_runs ${num_runs}
    # done

    # # Experiment 15: Multiplicative Noise Only with L2 regularization
    # for mult_noise_level in "${feature_mult_noise_levels[@]}"; do
    #     CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_mult_noise_level ${mult_noise_level} --weight_decay 0.01 --result_dir results/experiment_15 --num_runs ${num_runs}
    # done

    # # Experiment 16: Additive and Multiplicative Noise Combination with L2 regularization
    # for add_noise_level in "${feature_add_noise_levels[@]}"; do
    #     for mult_noise_level in "${feature_mult_noise_levels[@]}"; do
    #         CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_add_noise_level ${add_noise_level} --feature_mult_noise_level ${mult_noise_level} --weight_decay 0.01 --result_dir results/experiment_16 --num_runs ${num_runs}
    #     done
    # done

    # # Experiment 17: Label Noise with Additive Noise with L2 regularization
    # for noise_rate in "${noise_rates[@]}"; do
    #     for add_noise_level in "${feature_add_noise_levels[@]}"; do
    #         CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_add_noise_level ${add_noise_level} --weight_decay 0.01 --result_dir results/experiment_17 --num_runs ${num_runs}
    #     done
    # done

    # # Experiment 18: Label Noise with Multiplicative Noise with L2 regularization
    # for noise_rate in "${noise_rates[@]}"; do
    #     for mult_noise_level in "${feature_mult_noise_levels[@]}"; do
    #         CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_mult_noise_level ${mult_noise_level} --weight_decay 0.01 --result_dir results/experiment_18 --num_runs ${num_runs}
    #     done
    # done

    
    # Experiment 19: Additive Noise with Imbalance and Weight Resampling with L2 regularization
    for add_noise_level in "${feature_add_noise_levels[@]}"; do
        for imbalance_ratio in "${imbalance_ratios[@]}"; do
            for weight_resampling in "${weight_resamplings[@]}"; do
                CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --feature_add_noise_level ${add_noise_level} --weight_decay 0.01 --weight_resampling ${weight_resampling} --result_dir results/experiment_19 
            done
        done
    done

    # Experiment 20: Multiplicative Noise with Imbalance and Weight Resampling with L2 regularization
    for mult_noise_level in "${feature_mult_noise_levels[@]}"; do
        for imbalance_ratio in "${imbalance_ratios[@]}"; do
            for weight_resampling in "${weight_resamplings[@]}"; do
                CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --feature_mult_noise_level ${mult_noise_level} --weight_decay 0.01 --weight_resampling ${weight_resampling} --result_dir results/experiment_20 
            done
        done
    done

    # Experiment 21: Additive and Multiplicative Noise Combination with Imbalance and Weight Resampling with L2 regularization
    for add_noise_level in "${feature_add_noise_levels[@]}"; do
        for mult_noise_level in "${feature_mult_noise_levels[@]}"; do
            for imbalance_ratio in "${imbalance_ratios[@]}"; do
                for weight_resampling in "${weight_resamplings[@]}"; do
                    CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --feature_add_noise_level ${add_noise_level} --feature_mult_noise_level ${mult_noise_level} --weight_decay 0.01 --weight_resampling ${weight_resampling} --result_dir results/experiment_21 
                done
            done
        done
    done

    # Experiment 22: Label Noise with Additive Noise, Imbalance and Weight Resampling with L2 regularization
    for noise_rate in "${noise_rates[@]}"; do
        for add_noise_level in "${feature_add_noise_levels[@]}"; do
            for imbalance_ratio in "${imbalance_ratios[@]}"; do
                for weight_resampling in "${weight_resamplings[@]}"; do
                    CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type uniform --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --feature_add_noise_level ${add_noise_level} --weight_decay 0.01 --weight_resampling ${weight_resampling} --result_dir results/experiment_22 
                done
            done
        done
    done

    # Experiment 23: Label Noise with Multiplicative Noise, Imbalance and Weight Resampling with L2 regularization
    for noise_rate in "${noise_rates[@]}"; do
        for mult_noise_level in "${feature_mult_noise_levels[@]}"; do
            for imbalance_ratio in "${imbalance_ratios[@]}"; do
                for weight_resampling in "${weight_resamplings[@]}"; do
                    CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type uniform --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --feature_mult_noise_level ${mult_noise_level} --weight_decay 0.01 --weight_resampling ${weight_resampling} --result_dir results/experiment_23 
                done
            done
        done
    done


  done
done