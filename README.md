# Robust Depth Estimation

This repository contains code for evaluating the robustness of monocular depth estimation models under several controlled image perturbations. Given an input image and a depth model, the code searches for perturbation parameters that degrade the prediction quality and saves the optimized perturbation, predictions, and evaluation records.

All local paths are anonymized. Before running experiments, replace `/path/to/datasets`, `/path/to/checkpoints`, and `/path/to/experiment_results` with paths on your machine.

## Repository Structure

```text
.
|-- bash/
|   |-- robustness_analysis/     # Main batch experiment scripts
|   `-- ablation/                # Ablation scripts
`-- src/
    |-- scripts/                 # Main Python entry points
    |-- inference_wrapper/       # Model wrappers
    |-- optimization/            # Optimization methods
    |-- statistic/               # Result aggregation/checking
    |-- examples/                # Visualization scripts
    |-- utils/                   # Utility functions
    `-- perturbation.py          # Perturbation definitions
```

## Requirements

Install the common Python dependencies:

```bash
pip install torch torchvision numpy scipy opencv-python pillow matplotlib tqdm omegaconf kornia
```

Some models require their official implementations and checkpoints, for example:

```text
external/Marigold/
external/Depth_Anything_V2/
```

These external repositories and checkpoints are not included in this submission.

## Important Note on Running

The scripts are not directly runnable without local configuration. Before running, edit the bash scripts or command-line arguments to set:

- checkpoint paths
- dataset paths or dataset split files
- output directory
- CUDA device id
- external model repository paths

After these paths are configured, run commands from the repository root so that `src.*` imports resolve correctly.

## Main Experiment Scripts

The main Python entry points are:

```text
src.scripts.marigold_infer
src.scripts.da_infer
src.scripts.zoe_infer
src.scripts.robustdepth_infer
src.scripts.monodepth_infer
src.scripts.monovit_infer
src.scripts.depth_pro_infer
```

Batch scripts are provided under `bash/robustness_analysis/`. For example:

```bash
bash bash/robustness_analysis/marigold_infer.sh
bash bash/robustness_analysis/da_infer.sh
bash bash/robustness_analysis/zoe_infer.sh
bash bash/robustness_analysis/monodepth_infer.sh
bash bash/robustness_analysis/monovit_infer.sh
bash bash/robustness_analysis/robustdepth_infer.sh
bash bash/robustness_analysis/depth_pro_infer.sh
```

The lists near the top of each script, such as `perturb_list`, `dataset_list`, `max_iter_list`, `encoder_list`, and `model_type_list`, control which experiments are executed.

## Example Commands

Marigold:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m src.scripts.marigold_infer \
  --checkpoint_path /path/to/checkpoints/marigold_depth/ \
  --dataset nyud \
  --min-depth 0.1 \
  --max-depth 10.0 \
  --alignment least_square \
  --perturb_type color_shift \
  --max_iter 50 \
  --gamma 0.2 \
  --seed 42 \
  --max_batch_num 16 \
  --optimize_method pareto_direct \
  --min_data_idx 0 \
  --max_data_idx 60 \
  --root_save_dir /path/to/experiment_results/robustness_analysis/marigold_modified/50_iter/ \
  --skip_existing
```

Depth Anything V2:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m src.scripts.da_infer \
  --encoder vitl \
  --dataset kitti \
  --img-size 518 \
  --min-depth 0.1 \
  --max-depth 80.0 \
  --pretrained-from /path/to/checkpoints/depth_anything_v2/depth_anything_v2_metric_vkitti_vitl.pth \
  --perturb_type geometric \
  --max_iter 50 \
  --gamma 0.1 \
  --seed 42 \
  --max_batch_num 16 \
  --optimize_method pareto_direct \
  --min_data_idx 0 \
  --max_data_idx 60 \
  --root_save_dir /path/to/experiment_results/robustness_analysis/depth_anything/vitl/50_iter/ \
  --skip_existing
```

## Key Parameters

| Parameter | Description |
| --- | --- |
| `--dataset` | Dataset name: `nyud`, `kitti`, or `hypersim`. |
| `--perturb_type` | Perturbation type: `geometric`, `color_shift`, `motion_blur`, or `banding`. |
| `--optimize_method` | Optimization method: `pareto_direct`, `original_direct`, `direct`, or `dual_annealing`. |
| `--max_iter` | Number of optimization iterations. |
| `--gamma` | Perturbation strength / search range scale. |
| `--max_batch_num` | Maximum number of candidates evaluated in parallel by the custom DIRECT variants. |
| `--checkpoint_path` | Checkpoint directory for Marigold, Monodepth, MonoViT, RobustDepth, etc. |
| `--pretrained-from` | Checkpoint path for Depth Anything V2 and ZoeDepth style scripts. |
| `--root_save_dir` | Output root directory. |
| `--min_data_idx`, `--max_data_idx` | Shuffled sample index range to process. |

Model-specific parameters:

- Depth Anything V2: `--encoder` is one of `vits`, `vitb`, `vitl`, `vitg`.
- RobustDepth: `--encoder` is `vit` or `resnet`.
- Monodepth: `--model_type` is `mono_640x192` or `mono_1024x320`.
- MonoViT: `--model_type` is `640x192` or `1024x320`.

## Output

Each run saves results under:

```text
<root_save_dir>/<dataset>/<perturb_type>/<sample_index>/
```

Typical files include:

```text
config_<optimize_method>.json
depth_pred_original.npy
depth_pred_<optimize_method>.npy
gt_depth.npy
valid_mask.npy
```

## Ablation and Result Aggregation

Ablation scripts are under `bash/ablation/`, for example:

```bash
bash bash/ablation/gamma.sh
bash bash/ablation/max_iter.sh
bash bash/ablation/max_batch_num.sh
bash bash/ablation/optimize_method.sh
```

After experiments finish, result checking and aggregation scripts are available under `src/statistic/`:

```bash
python -m src.statistic.results_check
python -m src.statistic.robustness_evaluation
python -m src.statistic.RR_50iter_results
python -m src.statistic.ablation
```

Edit the path and experiment-list variables near the top of these statistic scripts before running them.
