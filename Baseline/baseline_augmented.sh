num_workers=0

for seed in 1 #2 3 4 5 
do
  for model_type in baseline
  do
    CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset baseline --model_type ${model_type} --data_augmentation smote --noise_rate 0.0 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/
    CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset baseline --model_type ${model_type} --data_augmentation smote --noise_rate 0.1 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/
    CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset baseline --model_type ${model_type} --data_augmentation smote --noise_rate 0.3 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/
    CUDA_LAUNCH_BLOCKING=1 python baseline.py --dataset baseline --model_type ${model_type} --data_augmentation smote --noise_rate 0.6 --seed ${seed} --num_workers ${num_workers} --result_dir results/trial_${seed}/
  done
done