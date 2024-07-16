num_workers=0
noise_rates=(0 0.1 0.3 0.6)
imbalance_ratios=(0 0.05 0.01)
feature_add_noise_levels=(0.0 0.3 0.6 1.0)
feature_mult_noise_levels=(0.0 0.3 0.6 1.0)

for seed in 1 #2 3 4 5 
do
  for model_type in generalisedCrossEntropy
  do
    # Experiment 2: Label Noise Only
    for noise_rate in "${noise_rates[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset BODMAS --model_type ${model_type} --weight_decay 0.0 --data_augmentation none --noise_rate ${noise_rate} --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_2$
    done

    # Experiment 3: Imbalance Only
    for imbalance_ratio in "${imbalance_ratios[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset BODMAS --model_type ${model_type} --weight_decay 0.0 --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_3$
    done

    # Experiment 4: Label Noise and Imbalance
    for noise_rate in "${noise_rates[@]}"; do
        for imbalance_ratio in "${imbalance_ratios[@]}"; do
            CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset BODMAS --model_type ${model_type} --weight_decay 0.0 --data_augmentation none --noise_rate ${noise_rate} --noise_type uniform --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_4$
        done
    done

    # Experiment 9: Additive Feature Noise Only
    for add_noise_level in "${feature_add_noise_levels[@]}"; do
      CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --weight_decay 0.0 --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_add_noise_level ${add_noise_level} --result_dir results/experiment_9$
    done

    # Experiment 10: Multiplicative Feature Noise Only
    for mult_noise_level in "${feature_mult_noise_levels[@]}"; do
      CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --weight_decay 0.0 --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_mult_noise_level ${mult_noise_level} --result_dir results/experiment_10$
    done

    # Experiment 11: Additive and Multiplicative Feature Noise Combination
    for add_noise_level in "${feature_add_noise_levels[@]}"; do
      for mult_noise_level in "${feature_mult_noise_levels[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --weight_decay 0.0 --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_add_noise_level ${add_noise_level} --feature_mult_noise_level ${mult_noise_level} --result_dir results/experiment_11$
      done
    done

    # Experiment 12: Label Noise with Additive Feature Noise
    for noise_rate in "${noise_rates[@]}"; do
      for add_noise_level in "${feature_add_noise_levels[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --weight_decay 0.0 --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_add_noise_level ${add_noise_level} --result_dir results/experiment_12$
      done
    done

    # Experiment 13: Label Noise with Multiplicative Feature Noise
    for noise_rate in "${noise_rates[@]}"; do
      for mult_noise_level in "${feature_mult_noise_levels[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --weight_decay 0.0 --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_mult_noise_level ${mult_noise_level} --result_dir results/experiment_13$
      done
    done

    
    # Experiment 14: Additive Noise Only with L2 regularization
    for add_noise_level in "${feature_add_noise_levels[@]}"; do
      CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_add_noise_level ${add_noise_level} --weight_decay 0.01 --result_dir results/experiment_14$
    done

    # Experiment 15: Multiplicative Noise Only with L2 regularization
    for mult_noise_level in "${feature_mult_noise_levels[@]}"; do
      CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_mult_noise_level ${mult_noise_level} --weight_decay 0.01 --result_dir results/experiment_15$
    done

    # Experiment 16: Additive and Multiplicative Noise Combination with L2 regularization
    for add_noise_level in "${feature_add_noise_levels[@]}"; do
      for mult_noise_level in "${feature_mult_noise_levels[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_add_noise_level ${add_noise_level} --feature_mult_noise_level ${mult_noise_level} --weight_decay 0.01 --result_dir results/experiment_16$
      done
    done

    # Experiment 17: Label Noise with Additive Noise with L2 regularization
    for noise_rate in "${noise_rates[@]}"; do
      for add_noise_level in "${feature_add_noise_levels[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_add_noise_level ${add_noise_level} --weight_decay 0.01 --result_dir results/experiment_17$
      done
    done

    # Experiment 18: Label Noise with Multiplicative Noise with L2 regularization
    for noise_rate in "${noise_rates[@]}"; do
      for mult_noise_level in "${feature_mult_noise_levels[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python generalisedCrossEntropy.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_mult_noise_level ${mult_noise_level} --weight_decay 0.01 --result_dir results/experiment_18$
      done
    done

  done
done