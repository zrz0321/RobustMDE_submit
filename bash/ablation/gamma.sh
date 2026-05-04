# ablation study on gamma value for marigold
cuda_device=0
perturb_list=("color_shift" "geometric" "banding")
model_name="marigold"
max_iter=30
optimize_method="pareto_direct"
seed=42
max_batch_num=16
reduced_bit_depth=6
kernel_size=7
checkpoint_path="/path/to/checkpoints/marigold_depth/"
min_data_idx=0
max_data_idx=30
min_depth=0.1
max_depth=10.0
dataset="nyud"

# gamma ablation
for perturb_type in ${perturb_list[@]}; do
    # get gamma (perturb range) for different perturbation type
    if [ "$perturb_type" == "color_shift" ]; then
        gamma_list=(0.1 0.15 0.2 0.25 0.3)
    elif [ "$perturb_type" == "geometric" ]; then
        gamma_list=(0.08 0.09 0.1 0.11 0.12)
    elif [ "$perturb_type" == "banding" ]; then
        gamma_list=(0.005 0.0075 0.01 0.0125 0.015)
    fi

    for gamma in ${gamma_list[@]}; do
        root_save_dir="/path/to/experiment_results/ablation_study/gamma/gamma_${gamma}/${perturb_type}/"
        CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer \
            --checkpoint_path $checkpoint_path \
            --dataset $dataset --min-depth $min_depth --max-depth $max_depth --alignment "least_square"\
            --perturb_type $perturb_type --max_iter $max_iter --gamma $gamma --seed $seed --max_batch_num $max_batch_num --reduced_bit_depth $reduced_bit_depth --kernel_size $kernel_size\
            --optimize_method $optimize_method \
            --min_data_idx $min_data_idx --max_data_idx $max_data_idx \
            --root_save_dir $root_save_dir \
            --skip_existing
    done
done
    