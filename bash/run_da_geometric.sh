# run geometric perturbation on marigold
cuda_device=0
# nyu-d dataset
# CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.da_infer \
#  --encoder vitl  --dataset kitti  --img-size 518  --min-depth 0.1  --max-depth 80.0  --pretrained-from /path/to/checkpoints/depth_anything_v2/depth_anything_v2_metric_vkitti_vitl.pth \
#  --perturb_type geometric  --max_iter 10  --gamma 0.06  --seed 42  --max_batch_num 8 \
#  --optimize_method pareto_direct \
#  --root_save_dir /path/to/experiment_results/depth_anything

# test color shift
CUDA_VISIBLE_DEVICES=$cuda_device python3 -m src.scripts.da_infer \
 --encoder vitl  --dataset kitti  --img-size 518  --min-depth 0.1  --max-depth 80.0  --pretrained-from /path/to/checkpoints/depth_anything_v2/depth_anything_v2_metric_vkitti_vitl.pth \
 --perturb_type color_shift  --max_iter 10  --gamma 0.2  --seed 42  --max_batch_num 1 \
 --optimize_method direct \
 --root_save_dir /path/to/experiment_results/depth_anything