num_workers=0

for seed in 1 #2 3 4 5 
do
  for model_type in LIO
  do
    # CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset CIC_IDS_2017 --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_1$
    # CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset CIC_IDS_2017 --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_1$
    # CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset CIC_IDS_2017 --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_1$
    # CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset CIC_IDS_2017 --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_1$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_2$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_2$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_2$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_2$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_3$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_3$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_3$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_4$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_4$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_4$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_4$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_4$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_4$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_4$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_4$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_4$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_4$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_4$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_4$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset BODMAS --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_5$





  
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_6$


    # Experiment 7 LIO, windows PE, all combinations of noise rate, noise type, and imbalance  ratios with data augmentation

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$




    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.1 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.1 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.3 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.3 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.6 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.6 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.1 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.3 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.6 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.1 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.3 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.6 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation undersampling --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$



    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.1 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.1 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.3 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.3 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.6 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.6 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.1 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.3 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.6 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.1 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.3 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.6 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation oversampling --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.1 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.1 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.3 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.3 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.6 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.6 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.1 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.3 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.6 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.1 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.3 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.6 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation smote --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.1 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.3 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.6 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.1 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.3 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.6 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation adasyn --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_7$



    # Experiment 8 LIO, windows PE, all combinations of noise rate, noise type, and imbalance  ratios with sample re-weighting techniques


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.1 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.1 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.3 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.3 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.6 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.6 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.1 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.3 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.6 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.1 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.3 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.6 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling none --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.1 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.1 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.3 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.3 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.6 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.6 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.1 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.3 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.6 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.1 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.3 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.6 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Class-Balance --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$



    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.1 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.1 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.3 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.3 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.6 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.6 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.1 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.3 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.6 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.1 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.3 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.6 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Focal --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$



    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.1 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.1 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.3 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.3 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.6 --noise_type class --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.6 --noise_type feature --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.1 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.3 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.6 --noise_type class --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.05 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$


    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.1 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.1 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.1 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.1 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$

    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.3 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.3 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.3 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.3 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.6 --noise_type uniform --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.6 --noise_type class --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.6 --noise_type feature --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$
    CUDA_LAUNCH_BLOCKING=1 python LIO.py --dataset windows_pe_real --model_type ${model_type} --data_augmentation none --weight_resampling Naive --noise_rate 0.6 --noise_type MIMICRY --imbalance_ratio 0.01 --seed ${seed} --num_workers ${num_workers} --result_dir results/experiment_8$

  done
done
