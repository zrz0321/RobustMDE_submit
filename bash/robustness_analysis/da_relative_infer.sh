cuda_device=0
perturb_list=("color_shift" "geometric" "motion_blur" "banding")
model_name="depth_anything"
# max_iter_list=(10 20 30 40 50)
max_iter_list=(10 30 50)
optimize_method="pareto_direct"
seed=42
max_batch_num=16
reduced_bit_depth=5
kernel_size=7
#
# -------------------
# vitl  |   vkitti
# -------------------
encoder="vitl"
#
# -------------------
# vitb  |   vkitti
# -------------------
# encoder="vitb"
#
# -------------------
# vits  |   vkitti
# -------------------
# encoder="vits"

min_depth=0.1
img_size=518
min_data_idx=0
max_data_idx=60
dataset_list=("hypersim" "kitti" "nyud")
alignment="least_square"

pretrained_from="/path/to/checkpoints/depth_anything_v2/depth_anything_v2_${encoder}.pth"
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
        # set dataset specific config
        if [ "$dataset_name" == "hypersim" ]; then
            # check if this performs better
            max_depth=20.0
        elif [ "$dataset_name" == "kitti" ]; then
            max_depth=80.0
        elif [ "$dataset_name" == "nyud" ]; then
            max_depth=10.0
        else
            echo "Unknown dataset: $dataset_name"
            exit 1
        fi
        # run inference on different max_iter
        for max_iter in ${max_iter_list[@]}; do
            root_save_dir="/path/to/experiment_results/robustness_analysis/${model_name}_relative/${encoder}/${max_iter}_iter/"
            # root_save_dir="/path/to/experiment_results/robustness_analysis/DEBUG/"

            CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.da_infer \
            --encoder $encoder --dataset $dataset_name --img-size $img_size --min-depth $min_depth --max-depth $max_depth --pretrained-from $pretrained_from \
            --perturb_type $perturb_type --max_iter $max_iter --gamma $gamma --seed $seed --max_batch_num $max_batch_num --reduced_bit_depth $reduced_bit_depth --kernel_size $kernel_size \
            --optimize_method $optimize_method \
            --min_data_idx $min_data_idx --max_data_idx $max_data_idx \
            --root_save_dir $root_save_dir \
            --skip_existing \
            --alignment $alignment
        done
    done
done