cuda_device=0
perturb_list=("color_shift" "geometric" "motion_blur" "banding")
model_name="depth_pro"
# max_iter_list=(10 20 30 40 50)
# max_iter_list=(10 30 50)
max_iter_list=(10 30)
optimize_method="pareto_direct"
seed=42
max_batch_num=10
reduced_bit_depth=6
kernel_size=7
checkpoint_path="/path/to/checkpoints/depth_pro/"
min_data_idx=0
max_data_idx=60
dataset_list=("nyud" "kitti" "hypersim")

for perturb_type in ${perturb_list[@]}; do
    # get gamma (perturb range) for different perturbation type
    if [ "$perturb_type" == "color_shift" ]; then
        gamma=0.2
    elif [ "$perturb_type" == "geometric" ]; then
        gamma=0.1
    elif [ "$perturb_type" == "motion_blur" ]; then
        gamma=0.1
    elif [ "$perturb_type" == "banding" ]; then
        gamma=0.15
    else
        echo "Unknown perturbation type: $perturb_type"
        exit 1
    fi

    for dataset_name in ${dataset_list[@]}; do
        # # set dataset specific config
        # if [ "$dataset_name" == "nyud" ]; then
        # elif [ "$dataset_name" == "kitti" ]; then
        # elif [ "$dataset_name" == "hypersim" ]; then
        # else
        #     echo "Unknown dataset: $dataset_name"
        #     exit 1
        # fi
        # run inference on different max_iter
        for max_iter in ${max_iter_list[@]}; do
            root_save_dir="/path/to/experiment_results/robustness_analysis/${model_name}/${max_iter}_iter/"
            CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.depth_pro_infer \
             --dataset $dataset_name \
             --perturb_type $perturb_type --max_iter $max_iter --gamma $gamma --seed $seed --max_batch_num $max_batch_num --reduced_bit_depth $reduced_bit_depth --kernel_size $kernel_size \
             --optimize_method $optimize_method \
             --min_data_idx $min_data_idx --max_data_idx $max_data_idx \
             --root_save_dir $root_save_dir \
             --skip_existing
        done
    done
done