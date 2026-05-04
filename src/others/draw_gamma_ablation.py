import json
import matplotlib.pyplot as plt
import os
from src.perturbation import Perturbation
import torch
from src.inference_wrapper.marigold.marigold_dataset import NYUD
from torch.utils.data import DataLoader


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

idx = 5
save_root = "./src/others/tmp/gamma_ablation"
os.makedirs(save_root, exist_ok=True)

perturb_type_list = ["banding", "color_shift", "geometric"]
def get_gamma(perturb_type):
    if perturb_type == "geometric":
        return [0.08, 0.10, 0.12]
    elif perturb_type == "color_shift":
        return [0.10, 0.20, 0.30]
    elif perturb_type == "banding":
        return [0.005, 0.01, 0.015]

valset = NYUD('external/Depth_Anything_V2/metric_depth/dataset/splits/nyud-v2/val.txt', 'val', '/path/to/datasets/nyu_depth_v2')

for perturb_type in perturb_type_list:
    perturb = Perturbation(perturb_type, transform_func=transform_func)
    gamma_list = get_gamma(perturb_type)
    for gamma in gamma_list:
        record_path = f"/path/to/experiment_results/ablation_study/gamma/gamma_{gamma}/{perturb_type}/nyud/{perturb_type}/{idx}/config_pareto_direct.json"
        with open(record_path, 'r') as f:
            log_dict = json.load(f)
        params = log_dict['best_params']

        generator = torch.Generator(device=torch.device("cpu")).manual_seed(42)
        valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True,
                           shuffle=True, generator=generator)
        for i, data in enumerate(valloader):
            if i < idx:
                continue
            elif i > idx:
                break
            else:
                img = data['image'].float()
                params_dict = get_theta_dict(params)
                perturbed_img = perturb.apply(img, **params_dict)
                perturbed_img = transform_func(perturbed_img, mode="to")
                perturbed_img = perturbed_img.squeeze(0)
                save_dir = os.path.join(save_root, f"{gamma}")
                os.makedirs(save_dir, exist_ok=True)
                plt.imsave(os.path.join(save_dir, f"{idx}_{perturb_type}_perturbed.png"), perturbed_img.permute(1, 2, 0).cpu().numpy())