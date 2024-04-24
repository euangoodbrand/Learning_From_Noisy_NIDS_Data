num_workers=0

for seed in 1 #2 3 4 5 
do
  for model_type in baseline
  do
    CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/
    # CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.5 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/
    # CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.5 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/
    # CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/

    # CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.0 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/
    # CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/
    # CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/
    # CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/

    # CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.0 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/
    # CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/
    # CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/
    # CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/
  done
done
