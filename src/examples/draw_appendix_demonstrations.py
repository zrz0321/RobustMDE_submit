import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src.perturbation import Perturbation
from src.inference_wrapper.marigold.marigold_dataset import KITTI, NYUD, HYPERSIM
import json
from torch.utils.data import DataLoader
import torch

def get_theta_dict(theta: list[float]):
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
            "kernel_size": 7,  # fixed kernel size
            "angle": angle,
            "direction": direction,
        }
    elif perturb_type == "banding":
        luma, cb, cr = theta
        theta = {
            "luma": luma,
            "cb": cb,
            "cr": cr,
            "reduced_bit_depth": 5,  # fixed reduced bit depth
        }
    return theta

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

fig, axes = plt.subplots(6, 4, figsize=(14, 18))
# plt.subplots_adjust(wspace=0.2, hspace=0.03)

dataset_list = ["kitti", "nyud", "hypersim"]
perturb_type_list = ["geometric", "color_shift", "motion_blur", "banding"]
idx_list = [(0, 2), (0, 2), (0, 2), (0, 2)]
seed = 42

for i, dataset in enumerate(dataset_list):
    if dataset == 'hypersim':
        valset = HYPERSIM('external/Marigold/data_split/hypersim_depth/filename_list_val_filtered.txt', 'val', '/path/to/datasets/marigold_dataset/hypersim/val')
    elif dataset == 'kitti':
        valset = KITTI('external/Depth_Anything_V2/metric_depth/dataset/splits/kitti/val.txt', 'val')
    elif dataset == 'nyud':
        valset = NYUD('external/Depth_Anything_V2/metric_depth/dataset/splits/nyud-v2/val.txt', 'val', '/path/to/datasets/nyu_depth_v2')
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    min_idx, max_idx = idx_list[i]
    for perturb_type in perturb_type_list:
        dataloader = DataLoader(valset, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(seed))
        perturbation = Perturbation(perturb_type=perturb_type, transform_func=transform_func)
        for j, data in enumerate(dataloader):
            if j < min_idx:
                continue
            if j >= max_idx:
                break
            img = data["image"]  # shape: (1, C, H, W)
            # load best params
            # /path/to/experiment_results/robustness_analysis/marigold/50_iter/nyud/banding/0/config_pareto_direct.json
            load_path = f"/path/to/experiment_results/robustness_analysis/marigold_modified/50_iter/{dataset}/{perturb_type}/{j}/config_pareto_direct.json"
            with open(load_path, "r") as f:
                json_data = json.load(f)
            theta = json_data["best_params"]
            theta_dict = get_theta_dict(theta)
            # apply perturbation
            perturbed_img = perturbation.apply(img, **theta_dict)
            # to [0, 1]
            perturbed_img = transform_func(perturbed_img, mode="to")
            perturbed_img = perturbed_img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            axes[i * 2 + j - min_idx, perturb_type_list.index(perturb_type)].imshow(perturbed_img)
            axes[i * 2 + j - min_idx, perturb_type_list.index(perturb_type)].axis("off")
            axes[i * 2 + j - min_idx, perturb_type_list.index(perturb_type)].set_title(f"{dataset.upper()} - {perturb_type.replace('_', ' ').title()}")
plt.tight_layout()
save_path = "./examples/appendix"
import os
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, "appendix_perturbations.pdf"))
plt.savefig(os.path.join(save_path, "appendix_perturbations_show.png"))
# run the following command
# python3 -m src.examples.draw_appendix_demonstrations