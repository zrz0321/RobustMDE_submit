# run banding perturbation on marigold
cuda_device=1
# nyu-d dataset
# original_direct
CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
 --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
 --perturb_type banding --max_iter 50 --gamma 0.15 --seed 42 --max_batch_num 8 --reduced_bit_depth 6\
 --optimize_method original_direct \
 --root_save_dir /path/to/experiment_results/marigold \

# pareto_direct
CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
 --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
 --perturb_type banding --max_iter 50 --gamma 0.15 --seed 42 --max_batch_num 8 --reduced_bit_depth 6\
 --optimize_method pareto_direct \
 --root_save_dir /path/to/experiment_results/marigold \

# scipy_direct
CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
 --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
 --perturb_type banding --max_iter 50 --gamma 0.15 --seed 42 --max_batch_num 1 --reduced_bit_depth 6\
 --optimize_method direct \
 --root_save_dir /path/to/experiment_results/marigold \

# scipy_dual_annealing
CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
 --base_data_dir /path/to/datasets/marigold_dataset/nyu-d --checkpoint_path /path/to/checkpoints/marigold_depth/ \
 --perturb_type banding --max_iter 50 --gamma 0.15 --seed 42 --max_batch_num 1 --reduced_bit_depth 6\
 --optimize_method dual_annealing \
 --root_save_dir /path/to/experiment_results/marigold \

# kitti-eigen
# original_direct
CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_kitti_eigen_test.yaml \
 --base_data_dir /path/to/datasets/marigold_dataset/kitti-eigen --checkpoint_path /path/to/checkpoints/marigold_depth/ \
 --perturb_type banding --max_iter 50 --gamma 0.15 --seed 42 --max_batch_num 8 --reduced_bit_depth 6\
 --optimize_method original_direct \
 --root_save_dir /path/to/experiment_results/marigold

# pareto_direct
CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_kitti_eigen_test.yaml \
 --base_data_dir /path/to/datasets/marigold_dataset/kitti-eigen --checkpoint_path /path/to/checkpoints/marigold_depth/ \
 --perturb_type banding --max_iter 50 --gamma 0.15 --seed 42 --max_batch_num 8 --reduced_bit_depth 6\
 --optimize_method pareto_direct \
 --root_save_dir /path/to/experiment_results/marigold

# scipy_direct
CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_kitti_eigen_test.yaml \
 --base_data_dir /path/to/datasets/marigold_dataset/kitti-eigen --checkpoint_path /path/to/checkpoints/marigold_depth/ \
 --perturb_type banding --max_iter 50 --gamma 0.15 --seed 42 --max_batch_num 1 --reduced_bit_depth 6\
 --optimize_method direct \
 --root_save_dir /path/to/experiment_results/marigold

# dual_annealing
CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.marigold_infer --dataset_config ./external/Marigold/config/dataset_depth/data_kitti_eigen_test.yaml \
 --base_data_dir /path/to/datasets/marigold_dataset/kitti-eigen --checkpoint_path /path/to/checkpoints/marigold_depth/ \
 --perturb_type banding --max_iter 50 --gamma 0.15 --seed 42 --max_batch_num 1 --reduced_bit_depth 6\
 --optimize_method dual_annealing \
 --root_save_dir /path/to/experiment_results/marigold
