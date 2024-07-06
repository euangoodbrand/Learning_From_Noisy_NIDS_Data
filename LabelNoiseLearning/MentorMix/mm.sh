#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error and exit immediately
set -o pipefail  # Return the exit status of the last command in the pipe that failed

num_workers=0
noise_rates=(0 0.1 0.3 0.6)
imbalance_ratios=(0 0.05 0.01)
noise_types=("uniform" "class" "feature" "MIMICRY")
data_augmentations=("none" "undersampling" "oversampling" "smote" "adasyn")
weight_resamplings=("Class-Balance" "Focal" "Naive")
feature_add_noise_levels=(0.0 0.3 0.6 1.0)
feature_mult_noise_levels=(0.0 0.3 0.6 1.0)

for seed in 1 #2 3 4 5 
do
  for model_type in mentorMix
  do
    for weight_resampling in "${weight_resamplings[@]}"
    do
      for noise_rate in "${noise_rates[@]}"
      do
        for noise_type in "${noise_types[@]}"
        do
          for imbalance_ratio in "${imbalance_ratios[@]}"
          do
            result_dir="results/experiment_8"
            cmd="CUDA_LAUNCH_BLOCKING=1 python mentorMix.py --dataset windows_pe_real --model_type ${model_type} --weight_resampling ${weight_resampling} --noise_rate ${noise_rate} --noise_type ${noise_type} --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --result_dir ${result_dir}"
            echo "Running command: ${cmd}"
            eval ${cmd}
            # Check the exit status of the command
            if [ $? -ne 0 ]; then
              echo "Command failed: ${cmd}"
              exit 1
            fi
          done
        done
      done
    done
  done
done
