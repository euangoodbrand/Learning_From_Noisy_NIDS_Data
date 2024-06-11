num_workers=0
noise_rates=(0 0.1 0.3 0.6)
imbalance_ratios=(0 0.05 0.01)
noise_types=("uniform" "class" "feature" "MIMICRY")
data_augmentations=("none" "undersampling" "oversampling" "smote" "adasyn")
weight_resamplings=("none" "Class-Balance" "Focal" "Naive")
feature_add_noise_levels=(0.0 0.3 0.6 1.0)
feature_mult_noise_levels=(0.0 0.3 0.6 1.0)

for seed in 1 #2 3 4 5 
do
  for model_type in soft
  do
    # Experiment 9, feature additive noise
    for feature_add_noise_level in "${feature_add_noise_levels[@]}"; do
      CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --weight_decay 0.0 --model_type ${model_type} --feature_add_noise_level ${feature_add_noise_level} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_9$
    done

    # Experiment 10, feature multiplicative noise
    for feature_mult_noise_level in "${feature_mult_noise_levels[@]}"; do
      CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --weight_decay 0.0  --model_type ${model_type} --feature_mult_noise_level ${feature_mult_noise_level} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_10$
    done

    # Experiment 11, feature additive and multiplicative noise combinations
    for feature_add_noise_level in "${feature_add_noise_levels[@]}"; do
      for feature_mult_noise_level in "${feature_mult_noise_levels[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --weight_decay 0.0 --weight_decay 0.0 --model_type ${model_type} --feature_add_noise_level ${feature_add_noise_level} --feature_mult_noise_level ${feature_mult_noise_level} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_11$
      done
    done

    # Experiment 12, label noise with feature additive noise
    for noise_rate in "${noise_rates[@]}"; do
      for feature_add_noise_level in "${feature_add_noise_levels[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --weight_decay 0.0 --model_type ${model_type} --noise_rate ${noise_rate} --feature_add_noise_level ${feature_add_noise_level} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_12$
      done
    done

    # Experiment 13, label noise with feature multiplicative noise
    for noise_rate in "${noise_rates[@]}"; do
      for feature_mult_noise_level in "${feature_mult_noise_levels[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --weight_decay 0.0 --model_type ${model_type} --noise_rate ${noise_rate} --feature_mult_noise_level ${feature_mult_noise_level} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_13$
      done
    done

    # Experiment 14, feature additive noise with L2 regularization
    for feature_add_noise_level in "${feature_add_noise_levels[@]}"; do
      CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --model_type ${model_type} --feature_add_noise_level ${feature_add_noise_level} --seed ${seed} --num_workers ${num_workers} --weight_decay 0.01 --result_dir results/experiment_14$
    done

    # Experiment 15, feature multiplicative noise with L2 regularization
    for feature_mult_noise_level in "${feature_mult_noise_levels[@]}"; do
      CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --model_type ${model_type} --feature_mult_noise_level ${feature_mult_noise_level} --seed ${seed} --num_workers ${num_workers} --weight_decay 0.01 --result_dir results/experiment_15$
    done

    # Experiment 16, feature additive and multiplicative noise combinations with L2 regularization
    for feature_add_noise_level in "${feature_add_noise_levels[@]}"; do
      for feature_mult_noise_level in "${feature_mult_noise_levels[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --model_type ${model_type} --feature_add_noise_level ${feature_add_noise_level} --feature_mult_noise_level ${feature_mult_noise_level} --seed ${seed} --num_workers ${num_workers} --weight_decay 0.01 --result_dir results/experiment_16$
      done
    done

    # Experiment 17, label noise with feature additive noise and L2 regularization
    for noise_rate in "${noise_rates[@]}"; do
      for feature_add_noise_level in "${feature_add_noise_levels[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --model_type ${model_type} --noise_rate ${noise_rate} --feature_add_noise_level ${feature_add_noise_level} --seed ${seed} --num_workers ${num_workers} --weight_decay 0.01 --result_dir results/experiment_17$
      done
    done

    # Experiment 18, label noise with feature multiplicative noise and L2 regularization
    for noise_rate in "${noise_rates[@]}"; do
      for feature_mult_noise_level in "${feature_mult_noise_levels[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --model_type ${model_type} --noise_rate ${noise_rate} --feature_mult_noise_level ${feature_mult_noise_level} --seed ${seed} --num_workers ${num_workers} --weight_decay 0.01 --result_dir results/experiment_18$
      done
    done

  done
done
