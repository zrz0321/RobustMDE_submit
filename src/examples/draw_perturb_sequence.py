from src.inference_wrapper.marigold.marigold_dataset import KITTI, NYUD, HYPERSIM
from torch.utils.data import DataLoader
import PIL.Image as Image
from src.perturbation import Perturbation, get_parameter_range
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as F1
import os
from src.inference_wrapper.marigold.marigold_pipeline import MarigoldDepthEstimator
from src.examples.draw_optimize_sequence import rmse_linear
from external.Marigold.src.util.alignment import (
    align_depth_least_square,
)

def align_pred_with_gt(depth_pred, gt_depth, valid_mask, alignment, align_min_depth, align_max_depth):
    """
    Align depth_pred with gt_depth, following marigold/eval.py
    """
    # Align with GT using least square
    if "least_square" == alignment:
        depth_pred, scale, shift = align_depth_least_square(
            gt_arr=gt_depth.cpu().numpy(),
            pred_arr=depth_pred.cpu().numpy(),
            valid_mask_arr=valid_mask.cpu().numpy(),
            return_scale_shift=True,
            max_resolution=None,
        )
        depth_pred = torch.from_numpy(depth_pred).to(marigold_model.device)  

    depth_pred = torch.clamp(depth_pred, min=align_min_depth, max=align_max_depth)
    depth_pred = torch.clamp(depth_pred, min=1e-6)
    return depth_pred

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
            "reduced_bit_depth": 6,  # fixed reduced bit depth
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

save_dir = "./examples/perturb_sequence"
os.makedirs(save_dir, exist_ok=True)

perturb_type = "color_shift"
dataset = "nyud"
target_id = 6
gamma = 0.2
total_sequence = [[0.0, 1.0, 0.0], [0.418879020478639, 1.0, 0.0], [0.418879020478639, 1.1333333333333333, 0.0], [0.418879020478639, 1.1333333333333333, -0.13333333333333333], [0.418879020478639, 1.0, -0.13333333333333333], [0.418879020478639, 1.0, -0.17777777777777776], [0.418879020478639, 0.9555555555555555, -0.17777777777777776], [0.4654211338651544, 1.0, -0.17777777777777776], [0.418879020478639, 0.9703703703703703, -0.17777777777777776], [0.418879020478639, 0.9703703703703703, -0.1925925925925926], [0.418879020478639, 0.9753086419753086, -0.1925925925925926], [0.34130883150111313, 1.1037037037037036, -0.1925925925925926], [0.34648017743294823, 1.1037037037037036, -0.1925925925925926], [0.34648017743294823, 1.105349794238683, -0.1925925925925926], [0.32579479370560793, 0.8864197530864198, -0.1925925925925926], [0.32579479370560793, 0.8880658436213993, -0.1925925925925926], [0.37233690709212364, 1.1086419753086418, -0.1925925925925926], [0.37750825302395863, 1.1086419753086418, -0.1925925925925926], [0.37750825302395863, 1.1102880658436214, -0.1925925925925926]]
total_value = [-0.12516136467456818, -0.13101442158222198, -0.13166318833827972, -0.17444652318954468, -0.17829982936382294, -0.18203406035900116, -0.18304233253002167, -0.18554867804050446, -0.1908624917268753, -0.19447384774684906, -0.20658093690872192, -0.209229975938797, -0.20937325060367584, -0.20937713980674744, -0.2095356434583664, -0.20962494611740112, -0.21033836901187897, -0.21047838032245636, -0.21049922704696655]
selected_id = [0, len(total_sequence) - 1]
params_sequence = [total_sequence[i] for i in selected_id]
values_sequence = [total_value[i] for i in selected_id]
valset = NYUD('external/Depth_Anything_V2/metric_depth/dataset/splits/nyud-v2/val.txt', 'val', '/path/to/datasets/nyu_depth_v2')

val_loader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
perturb = Perturbation(perturb_type=perturb_type, transform_func=transform_func)
for i, sample in enumerate(val_loader):
    if i < target_id:
        continue
    if i > target_id:
        break
    img = sample['image']
    for x, idx in enumerate(range(len(params_sequence))):
        theta_list = params_sequence[idx]
        theta_dict = get_theta_dict(theta_list)
        perturbed_img = perturb.apply(img, **theta_dict)
        perturbed_img_tmp = transform_func(perturbed_img, mode="to")
        perturbed_img_pil = F1.to_pil_image(perturbed_img_tmp.squeeze(0).cpu())
        save_path = os.path.join(save_dir, f"sample{target_id}_step{x}.png")
        perturbed_img_pil.save(save_path)

    params_range = get_parameter_range(perturb_type, gamma=gamma)
    random_theta = []
    for i in range(len(params_range)):
        param_min, param_max = params_range[i]
        random_value = torch.empty(1).uniform_(param_min, param_max).item()
        random_theta.append(random_value)

    random_theta_dict = get_theta_dict(random_theta)
    random_perturbed_img = perturb.apply(img, **random_theta_dict)
    random_perturbed_img_tmp = transform_func(random_perturbed_img, mode="to")
    random_perturbed_img_pil = F1.to_pil_image(random_perturbed_img_tmp.squeeze(0).cpu())
    save_path = os.path.join(save_dir, f"sample{target_id}_random.png")
    random_perturbed_img_pil.save(save_path)

    # evaluate random perturbation result
    checkpoint_path="/path/to/checkpoints/marigold_depth/"
    with torch.no_grad():
        marigold_model = MarigoldDepthEstimator(
            checkpoint_path=checkpoint_path,
            device=torch.device("cuda:0"),
        )
        depth_pred = marigold_model.batch_infer(random_perturbed_img.to(marigold_model.device)  ,
                                                num_inference_steps=1,
                                                show_pbar=False,
                                                generator=torch.Generator(marigold_model.device).manual_seed(42),
                                                )
        gt_depth = sample['depth'].to(marigold_model.device)
        valid_mask = sample['valid_mask'].to(marigold_model.device)
        
        if depth_pred.shape[-2:] != gt_depth.shape[-2:]:
            depth_pred = F.interpolate(depth_pred[:, None], size=gt_depth.shape[-2:], mode='bilinear', align_corners=True).squeeze(1)

        depth_pred = align_pred_with_gt(depth_pred, gt_depth, valid_mask, alignment="least_square", align_min_depth=0.1, align_max_depth=10)

        rmse_perturbed = rmse_linear(depth_pred, gt_depth, valid_mask)

    with open(os.path.join(save_dir, f"sample{target_id}_random_evaluation.txt"), "w") as f:
        f.write(f"Random theta: {random_theta}\n")
        f.write(f"RMSE on random perturbed image: {rmse_perturbed.item()}\n")
        f.write(f"selected theta sequence: {params_sequence}\n")
        f.write(f"corresponding value sequence: {values_sequence}\n")

