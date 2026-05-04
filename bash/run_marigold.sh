# # run optimization on Marigold
cuda_device=1
# nyu-d
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 50 --gamma 0.2 --seed 42 --max_batch_num 8 \
#  --optimize_method original_direct \
#  --root_save_dir /path/to/experiment_results/marigold \

# # pareto_direct
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 50 --gamma 0.2 --seed 42 --max_batch_num 8 \
#  --optimize_method pareto_direct \
#  --root_save_dir /path/to/experiment_results/marigold \

# # scipy_direct
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 50 --gamma 0.2 --seed 42 --max_batch_num 1 \
#  --optimize_method direct \
#  --root_save_dir /path/to/experiment_results/marigold \

# # scipy_dual_annealing
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 50 --gamma 0.2 --seed 42 --max_batch_num 1 \
#  --optimize_method dual_annealing \
#  --root_save_dir /path/to/experiment_results/marigold \

# constrast experiment

# gamma
# # pareto_direct gamma 0.3
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 100 --gamma 0.3 --seed 42 --max_batch_num 1 \
#  --optimize_method pareto_direct \
#  --root_save_dir /path/to/experiment_results/marigold_ablation/gamma_0.3 \

#  # pareto_direct gamma 0.4
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 100 --gamma 0.4 --seed 42 --max_batch_num 1 \
#  --optimize_method pareto_direct \
#  --root_save_dir /path/to/experiment_results/marigold_ablation/gamma_0.4 \

# # pareto_direct gamma 0.25
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 100 --gamma 0.25 --seed 42 --max_batch_num 1 \
#  --optimize_method pareto_direct \
#  --root_save_dir /path/to/experiment_results/marigold_ablation/gamma_0.25 \

# # pareto_direct gamma 0.35
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 100 --gamma 0.35 --seed 42 --max_batch_num 1 \
#  --optimize_method pareto_direct \
#  --root_save_dir /path/to/experiment_results/marigold_ablation/gamma_0.35 \

# max_iter
# # pareto_direct max_iter 50
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 50 --gamma 0.2 --seed 42 --max_batch_num 1 \
#  --optimize_method pareto_direct \
#  --root_save_dir /path/to/experiment_results/marigold_ablation/max_iter_50 \

# # pareto_direct max_iter 20
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 20 --gamma 0.2 --seed 42 --max_batch_num 1 \
#  --optimize_method pareto_direct \
#  --root_save_dir /path/to/experiment_results/marigold_ablation/max_iter_20 \

# # pareto_direct max_iter 10
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 10 --gamma 0.2 --seed 42 --max_batch_num 1 \
#  --optimize_method pareto_direct \
#  --root_save_dir /path/to/experiment_results/marigold_ablation/max_iter_10 \

# objective function
# # abs_rel
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 50 --gamma 0.2 --seed 42 --max_batch_num 1 \
#  --optimize_method pareto_direct --target_function abs_rel \
#  --root_save_dir /path/to/experiment_results/marigold_ablation/target_abs_rel\

#  # rmse_l
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 50 --gamma 0.2 --seed 42 --max_batch_num 1 \
#  --optimize_method pareto_direct --target_function rmse_l \
#  --root_save_dir /path/to/experiment_results/marigold_ablation/target_rmse_l\

# # delta1
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 50 --gamma 0.2 --seed 42 --max_batch_num 1 \
#  --optimize_method pareto_direct --target_function delta1 \
#  --root_save_dir /path/to/experiment_results/marigold_ablation/target_delta1\

# # MSE_loss
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 50 --gamma 0.2 --seed 42 --max_batch_num 1 \
#  --optimize_method pareto_direct --target_function mse_loss \
#  --root_save_dir /path/to/experiment_results/marigold_ablation/target_mse_loss\

# kitti-eigen
# original_direct
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_kitti_eigen_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/kitti-eigen --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 50 --gamma 0.2 --seed 42 --max_batch_num 8 \
#  --optimize_method original_direct \
#  --root_save_dir /path/to/experiment_results/marigold

# # pareto_direct
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_kitti_eigen_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/kitti-eigen --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 50 --gamma 0.2 --seed 42 --max_batch_num 8 \
#  --optimize_method pareto_direct \
#  --root_save_dir /path/to/experiment_results/marigold

# # pareto_direct
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type motion_blur --max_iter 50 --kernel_size 7 --seed 42 --max_batch_num 8 \
#  --optimize_method pareto_direct \
#  --root_save_dir /path/to/experiment_results/marigold \

#  # pareto_direct
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_kitti_eigen_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/kitti-eigen --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type motion_blur --max_iter 50 --kernel_size 7 --seed 42 --max_batch_num 8 \
#  --optimize_method pareto_direct \
#  --root_save_dir /path/to/experiment_results/marigold

# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type motion_blur --max_iter 50 --kernel_size 7 --seed 42 --max_batch_num 8 \
#  --optimize_method original_direct \
#  --root_save_dir /path/to/experiment_results/marigold

# # scipy_direct
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_kitti_eigen_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/kitti-eigen --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 50 --gamma 0.2 --seed 42 --max_batch_num 1 \
#  --optimize_method direct \
#  --root_save_dir /path/to/experiment_results/marigold

# # scipy_dual_annealing
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_kitti_eigen_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/kitti-eigen --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 50 --gamma 0.2 --seed 42 --max_batch_num 1 \
#  --optimize_method dual_annealing \
#  --root_save_dir /path/to/experiment_results/marigold

# # motion_blur
# # scipy_direct
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type motion_blur --max_iter 50 --kernel_size 7 --seed 42 --max_batch_num 1 \
#  --optimize_method direct \
#  --root_save_dir /path/to/experiment_results/marigold

# # scipy_dual_annealing
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type motion_blur --max_iter 50 --kernel_size 7 --seed 42 --max_batch_num 1 \
#  --optimize_method dual_annealing \
#  --root_save_dir /path/to/experiment_results/marigold

# # kitti-eigen

# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_kitti_eigen_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/kitti-eigen --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type motion_blur --max_iter 50 --kernel_size 7 --seed 42 --max_batch_num 1 \
#  --optimize_method direct \
#  --root_save_dir /path/to/experiment_results/marigold

# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_kitti_eigen_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/kitti-eigen --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type motion_blur --max_iter 50 --kernel_size 7 --seed 42 --max_batch_num 1 \
#  --optimize_method dual_annealing \
#  --root_save_dir /path/to/experiment_results/marigold

# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_kitti_eigen_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/kitti-eigen --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type motion_blur --max_iter 50 --kernel_size 7 --seed 42 --max_batch_num 8 \
#  --optimize_method original_direct \
#  --root_save_dir /path/to/experiment_results/marigold
 
# # delta1
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type color_shift --max_iter 50 --gamma 0.2 --seed 42 --max_batch_num 1 \
#  --optimize_method pareto_direct --target_function delta1 \
#  --root_save_dir /path/to/experiment_results/marigold_ablation/target_delta1\


# perturb type geometric:
# nyu-d
# # original_direct
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type geometric --max_iter 50 --gamma 0.06 --seed 42 --max_batch_num 8 \
#  --optimize_method original_direct \
#  --root_save_dir /path/to/experiment_results/marigold \

# # pareto_direct
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type geometric --max_iter 50 --gamma 0.06 --seed 42 --max_batch_num 8 \
#  --optimize_method pareto_direct \
#  --root_save_dir /path/to/experiment_results/marigold \

# # scipy_direct
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type geometric --max_iter 50 --gamma 0.06 --seed 42 --max_batch_num 1 \
#  --optimize_method direct \
#  --root_save_dir /path/to/experiment_results/marigold \

# # scipy_dual_annealing
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type geometric --max_iter 50 --gamma 0.06 --seed 42 --max_batch_num 1 \
#  --optimize_method dual_annealing \
#  --root_save_dir /path/to/experiment_results/marigold \

# kitti-eigen
# # original_direct
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_kitti_eigen_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/kitti-eigen --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type geometric --max_iter 50 --gamma 0.06 --seed 42 --max_batch_num 8 \
#  --optimize_method original_direct \
#  --root_save_dir /path/to/experiment_results/marigold

# # pareto_direct
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_kitti_eigen_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/kitti-eigen --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type geometric --max_iter 50 --gamma 0.06 --seed 42 --max_batch_num 8 \
#  --optimize_method pareto_direct \
#  --root_save_dir /path/to/experiment_results/marigold

# # scipy_direct
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_kitti_eigen_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/kitti-eigen --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type geometric --max_iter 50 --gamma 0.06 --seed 42 --max_batch_num 1 \
#  --optimize_method direct \
#  --root_save_dir /path/to/experiment_results/marigold

# # dual_annealing
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_kitti_eigen_test.yaml \
#  --base_data_dir /path/to/datasets/marigold_dataset/kitti-eigen --checkpoint_path /path/to/checkpoints/marigold_depth/ \
#  --perturb_type geometric --max_iter 50 --gamma 0.06 --seed 42 --max_batch_num 1 \
#  --optimize_method dual_annealing \
#  --root_save_dir /path/to/experiment_results/marigold
