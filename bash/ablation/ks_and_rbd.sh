# ablation study for kernel size and reduced bit depth
cuda_device=0
model_name="marigold"
max_iter=30
optimize_method="pareto_direct"
seed=42
max_batch_num=16
reduced_bit_depth_list=(4 5 6 7)
kernel_size_list=(3 5 7 9 11)
checkpoint_path="/path/to/checkpoints/marigold_depth/"
min_data_idx=0
max_data_idx=30
min_depth=0.1
max_depth=10.0
dataset="nyud"

# kernel size ablation
for kernel_size in ${kernel_size_list[@]}; do
    perturb_type="motion_blur"
    reduced_bit_depth=5
    gamma=0.1
    root_save_dir="/path/to/experiment_results/ablation_study/ks_and_rbd/kernel_size_${kernel_size}/"
    CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer \
    --checkpoint_path $checkpoint_path \
    --dataset $dataset --min-depth $min_depth --max-depth $max_depth --alignment "least_square" \
    --perturb_type $perturb_type --max_iter $max_iter --gamma $gamma --seed $seed --max_batch_num $max_batch_num --reduced_bit_depth $reduced_bit_depth --kernel_size $kernel_size\
    --optimize_method $optimize_method \
    --min_data_idx $min_data_idx --max_data_idx $max_data_idx \
    --root_save_dir $root_save_dir \
    --skip_existing
done

# reduced bit depth ablation
for reduced_bit_depth in ${reduced_bit_depth_list[@]}; do
    kernel_size=7
    perturb_type="banding"
    gamma=0.01
    root_save_dir="/path/to/experiment_results/ablation_study/ks_and_rbd/reduced_bit_depth_${reduced_bit_depth}/"
    CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer \
    --checkpoint_path $checkpoint_path \
    --dataset $dataset --min-depth $min_depth --max-depth $max_depth --alignment "least_square" \
    --perturb_type $perturb_type --max_iter $max_iter --gamma $gamma --seed $seed --max_batch_num $max_batch_num --reduced_bit_depth $reduced_bit_depth --kernel_size $kernel_size\
    --optimize_method $optimize_method \
    --min_data_idx $min_data_idx --max_data_idx $max_data_idx \
    --root_save_dir $root_save_dir \
    --skip_existing
done