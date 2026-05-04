import PIL.Image as Image
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import torch
from matplotlib import pyplot as plt
import matplotlib

from src.inference_wrapper.marigold.marigold_pipeline import MarigoldDepthEstimator
from src.perturbation import Perturbation
from external.Marigold.src.dataset.base_depth_dataset import BaseDepthDataset, DatasetMode

from src.perturbation import get_parameter_range, Perturbation, geometric_perturb_for_depth
from external.Marigold.src.dataset import get_dataset
from external.Marigold.marigold.marigold_depth_pipeline import MarigoldDepthOutput

import json
import os

kernel_size = 9
reduced_bit_depth = 5

def get_dataloader(model_name: str, dataset: str) -> DataLoader:
    if model_name == "marigold":
        if dataset == "kitti":
            dataset_config="./external/Marigold/config/dataset_depth/data_kitti_eigen_test.yaml"
            base_data_dir="/path/to/datasets/marigold_dataset/kitti-eigen"
        elif dataset == "nyud":
            dataset_config="./external/Marigold/config/dataset_depth/data_nyu_test.yaml"
            base_data_dir="/path/to/datasets/marigold_dataset/nyu-d"
        elif dataset == "hypersim":
            dataset_config="./external/Marigold/config/dataset_depth/data_hypersim_val.yaml"
            base_data_dir="/path/to/datasets/marigold_dataset/hypersim"

        # LOAD DATASET
        cfg_data = OmegaConf.load(dataset_config)
        dataset : BaseDepthDataset = get_dataset(
            cfg_data,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        return dataloader
    elif model_name == "depth_anything":
        pass

def transform_func(img: torch.Tensor, mode: str="to"):
    """
    Transform the image to the normalized range [0, 1] when mode is "to", and transform back when mode is "back".
    """
    if mode == "to":
        img = (img + 1.0) / 2.0  # to [0, 1]
        img = torch.clamp(img, min=0.0, max=1.0)
    elif mode == "back":
        img = img * 2.0 - 1.0  # to [-1, 1]
        img = torch.clamp(img, min=-1.0, max=1.0)
    return img

def get_theta_dict(perturb_type, theta: list[float]):
    """
    Transform the list of theta to a dictionary for function api
    """
    if perturb_type == "geometric":
        s_hor, s_vrt, t_hor, t_vrt = theta
        theta = {
            "s_hor": s_hor,
            "s_vrt": s_vrt,
            "t_hor": t_hor,
            "t_vrt": t_vrt,
        }
    elif perturb_type == "color_shift":
        hue, sat, brt = theta
        theta = {
            "hue": hue,
            "sat": sat,
            "brt": brt,
        }
    elif perturb_type == "motion_blur":
        angle, direction = theta
        theta = {
            "kernel_size": kernel_size,  # fixed kernel size
            "angle": angle,
            "direction": direction,
        }
    elif perturb_type == "banding":
        luma, cb, cr = theta
        theta = {
            "luma": luma,
            "cb": cb,
            "cr": cr,
            "reduced_bit_depth": reduced_bit_depth,  # fixed reduced bit depth
        }
    else:
        theta = {}
    return theta


seed = 42

model_name = "marigold"
iter_num = 50
dataset = "nyud"

"""
Sample id is the index of the sample in the dataset, starting from 0.
You can change it to any valid index in the dataset.
"""
sample_id = 11


saved_dir = "./examples"
saved_dir = os.path.join(saved_dir, dataset, model_name, f"{sample_id}")
os.makedirs(saved_dir, exist_ok=True)

cm_type = "Spectral"

perturbed_type_list = ["geometric", "color_shift", "motion_blur", "banding", None]

dataloader = get_dataloader(model_name=model_name, dataset=dataset)

marigold_model = MarigoldDepthEstimator(
        checkpoint_path="/path/to/checkpoints/marigold_depth/",
        device=torch.device("cuda:0"),
    )
model_name = "marigold_modified"
id = 0
for batch in dataloader:
    if id != sample_id:
        id += 1
        continue
    if id > sample_id:
        break

    rgb_norm = batch["rgb_norm"].to(marigold_model.device)
    gt_depth = batch['depth_raw_linear'].cpu().squeeze().numpy()
    valid_mask = batch['valid_mask_raw'].cpu().squeeze().numpy().astype(bool)
    rgb_relatve_path = batch["rgb_relative_path"]
    print(f"Processing sample id {id}, path: {rgb_relatve_path}")

    for perturbed_type in perturbed_type_list:
        print(f"  Perturbation type: {perturbed_type}")
        perturb = Perturbation(perturb_type=perturbed_type, transform_func=transform_func)

        # get the best parameters for the perturbation type in the config file
        if perturbed_type is not None:
            config_path = f"/path/to/experiment_results/robustness_analysis/{model_name}/{iter_num}_iter/{dataset}/{perturbed_type}/{sample_id}/config_pareto_direct.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            best_params = config["best_params"]
            print(f"    |-->:Best parameters for |{perturbed_type}|: {best_params}")

        if perturbed_type is not None:
            img_perturbed = perturb.apply(rgb_norm.clone(), **get_theta_dict(perturbed_type, best_params))
        else:
            img_perturbed = rgb_norm.clone()

        # save perturbed image
        saved_path = os.path.join(saved_dir, f"sample_{sample_id}_{perturbed_type}.png")
        img_perturbed_to_save = (transform_func(img_perturbed, mode="to").squeeze().permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
        Image.fromarray(img_perturbed_to_save).save(saved_path)
        print(f"    |-->:Saved |{perturbed_type}| perturbed image to |{saved_path}|")

        with torch.no_grad():
            generator = torch.Generator(device=marigold_model.device).manual_seed(seed) if seed is not None else None
            pred_depth = marigold_model.batch_infer(img_perturbed,
                            num_inference_steps=1,
                            show_pbar=False,
                            generator=generator
                            )
            # draw the color map for pred_depth
            pred_depth = pred_depth.squeeze().cpu().numpy()
            pred_depth = (pred_depth - np.min(pred_depth)) / (np.max(pred_depth) - np.min(pred_depth) + 1e-8)
            pred_depth_colormap = matplotlib.colormaps[cm_type](pred_depth)[:, :, :3]  # H, W, 3
            pred_depth_colormap = (pred_depth_colormap * 255.0).astype("uint8")
            saved_path = os.path.join(saved_dir, f"sample_{sample_id}_{perturbed_type}_pred_depth.png")
            Image.fromarray(pred_depth_colormap).save(saved_path)
            print(f"    |-->:Saved |{perturbed_type}| perturbed depth map to |{saved_path}|")
    
        gt_depth = (gt_depth - np.min(gt_depth)) / (np.max(gt_depth) - np.min(gt_depth) + 1e-8)
        gt_depth_colormap = matplotlib.colormaps[cm_type](gt_depth)[:, :, :3]  # H, W
        gt_depth_colormap = (gt_depth_colormap * 255.0).astype("uint8")
        gt_depth_colormap = gt_depth_colormap * valid_mask[:, :, None]
        saved_path = os.path.join(saved_dir, f"sample_{sample_id}_gt_depth.png")
        Image.fromarray(gt_depth_colormap).save(saved_path)
        print(f"    |-->:Saved ground truth depth map to |{saved_path}|")
    id += 1

# params could be found in:
# /path/to/experiment_results/robustness_analysis/marigold/50_iter/kitti/color_shift/26/config_pareto_direct.json
# /path/to/experiment_results/robustness_analysis/{model_name}/{iter_num}_iter/{dataset}/{perturbed_type}/{sample_id}/config_pareto_direct.json
# json file
# "best_params": best_params shows the best parameters for each perturbation type

# run the following command:
# CUDA_VISIBLE_DEVICES=1 python3 -m src.examples.draw_examples