cuda_device=1
perturb_list=("geometric")
model_name="monovit"
max_iter_list=(50)
optimize_method="pareto_direct"
seed=42
max_batch_num=16
reduced_bit_depth=5
kernel_size=7
min_data_idx=1
max_data_idx=2
min_depth=0.1
dataset_list=("kitti")
model_type_list=("640x192")

for perturb_type in ${perturb_list[@]}; do
    # get gamma (perturb range) for different perturbation type
    if [ "$perturb_type" == "color_shift" ]; then
        gamma=0.2
    elif [ "$perturb_type" == "geometric" ]; then
        gamma=0.1
    elif [ "$perturb_type" == "motion_blur" ]; then
        gamma=0.1
    elif [ "$perturb_type" == "banding" ]; then
        gamma=0.01
    else
        echo "Unknown perturbation type: $perturb_type"
        exit 1
    fi

    for dataset_name in ${dataset_list[@]}; do
        if [ "$dataset_name" == "nyud" ]; then
            max_depth=10.0
        elif [ "$dataset_name" == "kitti" ]; then
            max_depth=80.0
        elif [ "$dataset_name" == "hypersim" ]; then
            max_depth=20.0
        else
            echo "Unknown dataset name: $dataset_name"
            exit 1
        fi

        for model_type in ${model_type_list[@]}; do
            checkpoint_path="/path/to/checkpoints/monovit/${model_type}/"
            # run inference on different max_iter
            for max_iter in ${max_iter_list[@]}; do
                root_save_dir="/path/to/experiment_results/robustness_analysis/${model_name}/${model_type}/${max_iter}_iter/"
                CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.monovit_infer \
                --checkpoint_path $checkpoint_path --model_type $model_type \
                --dataset $dataset_name --min-depth $min_depth --max-depth $max_depth \
                --perturb_type $perturb_type --max_iter $max_iter --gamma $gamma --seed $seed --max_batch_num $max_batch_num --reduced_bit_depth $reduced_bit_depth --kernel_size $kernel_size \
                --optimize_method $optimize_method \
                --min_data_idx $min_data_idx --max_data_idx $max_data_idx \
                --root_save_dir $root_save_dir
            done
        done
    done
done