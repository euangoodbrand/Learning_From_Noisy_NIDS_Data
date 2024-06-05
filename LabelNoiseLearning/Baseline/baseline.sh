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
  for model_type in baseline
  do
    #Experiment 1
    # for noise_rate in "${noise_rates[@]}"; do
    #     CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset CIC_IDS_2017 --model_type ${model_type} --data_augmentation ${data_augmentation} --noise_rate ${noise_rate} --noise_type ${noise_type} --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_1$
    # done

    # # Experiment 2
    # for noise_rate in "${noise_rates[@]}"; do
    #     CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_2$
    # done

    # # Experiment 3
    # for imbalance_ratio in "${imbalance_ratios[@]}"; do
    #     CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_3$
    # done

    # # Experiment 4
    # for noise_rate in "${noise_rates[@]}"; do
    #     for imbalance_ratio in "${imbalance_ratios[@]}"; do
    #         CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type uniform --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_4$
    #     done
    # done

    # # Experiment 5
    # for noise_rate in "${noise_rates[@]}"; do
    #     for noise_type in "${noise_types[@]}"; do
    #         CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type ${noise_type} --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5$
    #     done
    # done

    # # Experiment 6
    # for noise_rate in "${noise_rates[@]}"; do
    #     for noise_type in "${noise_types[@]}"; do
    #         for imbalance_ratio in "${imbalance_ratios[@]}"; do
    #             CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type ${noise_type} --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    #         done
    #     done
    # done

    # # Experiment 7 baseline, windows PE, all combinations of noise rate, noise type, and imbalance  ratios with data augmentation
    # for data_augmentation in "${data_augmentations[@]}"; do
    #     for noise_rate in "${noise_rates[@]}"; do
    #         for noise_type in "${noise_types[@]}"; do
    #             for imbalance_ratio in "${imbalance_ratios[@]}"; do
    #                 CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation ${data_augmentation} --noise_rate ${noise_rate} --noise_type ${noise_type} --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    #             done
    #         done
    #     done
    # done

    # Experiment 8 baseline, windows PE, all combinations of noise rate, noise type, and imbalance  ratios with sample re-weighting techniques
    # for weight_resampling in "${weight_resamplings[@]}"; do
    #   for noise_rate in "${noise_rates[@]}"; do
    #     for noise_type in "${noise_types[@]}"; do
    #       for imbalance_ratio in "${imbalance_ratios[@]}"; do
    #         CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset windows_pe_real --model_type ${model_type} --weight_resampling ${weight_resampling} --noise_rate ${noise_rate} --noise_type ${noise_type} --imbalance_ratio ${imbalance_ratio} --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    #       done
    #     done
    #   done
    # done

    # Experiment 9: Additive Noise Only
    for add_noise_level in "${feature_add_noise_levels[@]}"; do
      CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_add_noise_level ${add_noise_level} --result_dir results/experiment_9$
    done

    # Experiment 10: Multiplicative Noise Only
    for mult_noise_level in "${feature_mult_noise_levels[@]}"; do
      CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_mult_noise_level ${mult_noise_level} --result_dir results/experiment_10$
    done

    # Experiment 11: Additive and Multiplicative Noise Combination
    for add_noise_level in "${feature_add_noise_levels[@]}"; do
      for mult_noise_level in "${feature_mult_noise_levels[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_add_noise_level ${add_noise_level} --feature_mult_noise_level ${mult_noise_level} --result_dir results/experiment_11$
      done
    done

    # Experiment 12: Label Noise with Additive Noise
    for noise_rate in "${noise_rates[@]}"; do
      for add_noise_level in "${feature_add_noise_levels[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_add_noise_level ${add_noise_level} --result_dir results/experiment_12$
      done
    done

    # Experiment 13: Label Noise with Multiplicative Noise
    for noise_rate in "${noise_rates[@]}"; do
      for mult_noise_level in "${feature_mult_noise_levels[@]}"; do
        CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate ${noise_rate} --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --feature_mult_noise_level ${mult_noise_level} --result_dir results/experiment_13$
      done
    done

  done
done