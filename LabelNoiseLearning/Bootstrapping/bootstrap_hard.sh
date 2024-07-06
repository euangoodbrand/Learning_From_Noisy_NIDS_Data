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
  for model_type in hard
  do
    #Experiment 1
    # for noise_rate in "${noise_rates[@]}"; do
    #     CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset CIC_IDS_2017 --weight_decay 0.0 --model_type ${model_type} --data_augmentation ${data_augmentation} --noise_rate ${noise_rate} --noise_type ${noise_type} --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --result_dir ${result_dir}
    # done

    # Experiment 2
    # for noise_rate in "${noise_rates[@]}"; do
    #     CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset BODMAS --weight_decay 0.0 --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_2
    # done

    # # Experiment 3
    # for imbalance_ratio in "${imbalance_ratios[@]}"; do
    #     CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset BODMAS --weight_decay 0.0 --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_3
    # done

    # # Experiment 4
    # for noise_rate in "${noise_rates[@]}"; do
    #     for imbalance_ratio in "${imbalance_ratios[@]}"; do
    #         CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset BODMAS --weight_decay 0.0 --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type uniform --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_4
    #     done
    # done

    # # Experiment 5
    # for noise_rate in "${noise_rates[@]}"; do
    #     for noise_type in "${noise_types[@]}"; do
    #         CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset BODMAS --weight_decay 0.0 --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type ${noise_type} --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5
    #     done
    # done

    # # Experiment 6
    # for noise_rate in "${noise_rates[@]}"; do
    #     for noise_type in "${noise_types[@]}"; do
    #         for imbalance_ratio in "${imbalance_ratios[@]}"; do
    #             CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --weight_decay 0.0 --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type ${noise_type} --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6
    #         done
    #     done
    # done

    # # Experiment 7 bootstrapping, windows PE, all combinations of noise rate, noise type, and imbalance  ratios with data augmentation
    # for data_augmentation in "${data_augmentations[@]}"; do
    #     for noise_rate in "${noise_rates[@]}"; do
    #         for noise_type in "${noise_types[@]}"; do
    #             for imbalance_ratio in "${imbalance_ratios[@]}"; do
    #                 CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --weight_decay 0.0 --model_type ${model_type} --data_augmentation ${data_augmentation} --noise_rate ${noise_rate} --noise_type ${noise_type} --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7
    #             done
    #         done
    #     done
    # done

    # Experiment 8 bootstrapping, windows PE, all combinations of noise rate, noise type, and imbalance  ratios with sample re-weighting techniques
    for weight_resampling in "${weight_resamplings[@]}"; do
      for noise_rate in "${noise_rates[@]}"; do
        for noise_type in "${noise_types[@]}"; do
          for imbalance_ratio in "${imbalance_ratios[@]}"; do
            CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --weight_decay 0.0 --model_type ${model_type} --weight_resampling ${weight_resampling} --noise_rate ${noise_rate} --noise_type ${noise_type} --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8
          done
        done
      done
    done

    # # Experiment 9 bootstrapping, feature additive noise
    # for feature_add_noise_level in "${feature_add_noise_levels[@]}"; do
    #   CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --weight_decay 0.0 --model_type ${model_type} --feature_add_noise_level ${feature_add_noise_level} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_9$
    # done

    # # Experiment 10 bootstrapping, feature multiplicative noise
    # for feature_mult_noise_level in "${feature_mult_noise_levels[@]}"; do
    #   CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --weight_decay 0.0 --model_type ${model_type} --feature_mult_noise_level ${feature_mult_noise_level} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_10$
    # done

    # # Experiment 11 bootstrapping, feature additive and multiplicative noise combinations
    # for feature_add_noise_level in "${feature_add_noise_levels[@]}"; do
    #   for feature_mult_noise_level in "${feature_mult_noise_levels[@]}"; do
    #     CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --weight_decay 0.0 --model_type ${model_type} --feature_add_noise_level ${feature_add_noise_level} --feature_mult_noise_level ${feature_mult_noise_level} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_11$
    #   done
    # done

    # # Experiment 12 bootstrapping, label noise with feature additive noise
    # for noise_rate in "${noise_rates[@]}"; do
    #   for feature_add_noise_level in "${feature_add_noise_levels[@]}"; do
    #     CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --weight_decay 0.0 --model_type ${model_type} --noise_rate ${noise_rate} --feature_add_noise_level ${feature_add_noise_level} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_12$
    #   done
    # done

    # # Experiment 13 bootstrapping, label noise with feature multiplicative noise
    # for noise_rate in "${noise_rates[@]}"; do
    #   for feature_mult_noise_level in "${feature_mult_noise_levels[@]}"; do
    #     CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --weight_decay 0.0 --model_type ${model_type} --noise_rate ${noise_rate} --feature_mult_noise_level ${feature_mult_noise_level} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_13$
    #   done
    # done

    # # New Experiments with L2 Regularization

    # # Experiment 14 bootstrapping, feature additive noise with L2 regularization
    # for feature_add_noise_level in "${feature_add_noise_levels[@]}"; do
    #   CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --model_type ${model_type} --feature_add_noise_level ${feature_add_noise_level} --seed ${seed} --num_workers ${num_workers} --weight_decay 0.01 --result_dir results/experiment_14$
    # done

    # # Experiment 15 bootstrapping, feature multiplicative noise with L2 regularization
    # for feature_mult_noise_level in "${feature_mult_noise_levels[@]}"; do
    #   CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --model_type ${model_type} --feature_mult_noise_level ${feature_mult_noise_level} --seed ${seed} --num_workers ${num_workers} --weight_decay 0.01 --result_dir results/experiment_15$
    # done

    # # Experiment 16 bootstrapping, feature additive and multiplicative noise combinations with L2 regularization
    # for feature_add_noise_level in "${feature_add_noise_levels[@]}"; do
    #   for feature_mult_noise_level in "${feature_mult_noise_levels[@]}"; do
    #     CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --model_type ${model_type} --feature_add_noise_level ${feature_add_noise_level} --feature_mult_noise_level ${feature_mult_noise_level} --seed ${seed} --num_workers ${num_workers} --weight_decay 0.01 --result_dir results/experiment_16$
    #   done
    # done

    # # Experiment 17 bootstrapping, label noise with feature additive noise with L2 regularization
    # for noise_rate in "${noise_rates[@]}"; do
    #   for feature_add_noise_level in "${feature_add_noise_levels[@]}"; do
    #     CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --model_type ${model_type} --noise_rate ${noise_rate} --feature_add_noise_level ${feature_add_noise_level} --seed ${seed} --num_workers ${num_workers} --weight_decay 0.01 --result_dir results/experiment_17$
    #   done
    # done

    # # Experiment 18 bootstrapping, label noise with feature multiplicative noise with L2 regularization
    # for noise_rate in "${noise_rates[@]}"; do
    #   for feature_mult_noise_level in "${feature_mult_noise_levels[@]}"; do
    #     CUDA_LAUNCH_BLOCKING=1 python bootstrapping.py --dataset windows_pe_real --model_type ${model_type} --noise_rate ${noise_rate} --feature_mult_noise_level ${feature_mult_noise_level} --seed ${seed} --num_workers ${num_workers} --weight_decay 0.01 --result_dir results/experiment_18$
    #   done
    # done

  done
done
