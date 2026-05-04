from src.inference_wrapper.marigold.marigold_dataset import KITTI, NYUD, HYPERSIM

from src.optimization.func_wrapped import func_wrapped_parallel
from src.inference_wrapper.marigold.marigold_pipeline import MarigoldDepthEstimator

from src.perturbation import get_parameter_range, Perturbation, geometric_perturb_for_depth

from external.Marigold.src.util.alignment import (
    align_depth_least_square,
    depth2disparity,
    disparity2depth,
)

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import time
import datetime
from src.utils.seed import set_seed

from src.optimization.direct import Interval, trisect_interval_pareto, projection, select_PLO
@torch.no_grad()
def reporting_pareto_direct(bounds, f_parallel, max_iter=1000, **kwargs):
    """
    Simple_direct optimization
    Input:
        bounds: list of tuples, [(min1, max1), (min2, max2), ...]
        f_parallel: function to minimize, which takes in a list of parameter lists and returns a list of function values(could be parallelized)
        max_iter: maximum number of iterations
        kwargs: additional arguments to pass to f_parallel

    Return:
        (best_value, best_params, value_list, inf_pos)
        where best_value is the minimum value found and best_params are the parameters that give this value
        and value_list is the list of best values found at each improvement step
        And inf_pos is the position of the best_value in batched evaluation, tuple: (pos, max_pos), for reproducibility only. If max_batch_num
        is 1 then just ignore this parameter.
    """
    # Initialize
    num_params = len(bounds)
    best_value = float('inf')
    best_params = None
    inf_pos = (0, 1)
    eval_count = 0
    value_list = []


    # Create the initial interval, covering the whole search hybercube [0, 1]^n
    initial_interval = Interval(
        size=1,
        center_value=f_parallel(
            [projection([0.5] * num_params, bounds)],
            parallel_num=1,
            **kwargs,
        )[0],
        bounds=[(0.0, 1.0)] * num_params,
        center_point=[0.5] * num_params,
    )
    eval_count += 1
    # update best value and params
    best_params_list = []
    best_value_list = []
    best_value = initial_interval.center_value
    best_params = projection(initial_interval.center_point, bounds)
    best_params_list.append(best_params)
    best_value_list.append(best_value)
    inf_pos = initial_interval.inf_pos
    value_list.append(best_value)

    interval_list = [initial_interval]
    PLO_list = [initial_interval]

    # iteration
    iter_time = 0
    while iter_time < max_iter:
        # free cuda memory
        # torch.cuda.empty_cache()
        # split each interval in PLO_list
        new_interval_list = []
        for interval in PLO_list:
            # for pareto DIRECT, use trisect_interval_pareto
            new_intervals = trisect_interval_pareto(interval, bounds, f_parallel, **kwargs)
            new_interval_list += new_intervals

        for interval in new_interval_list:
            eval_count += 1
            value = interval.center_value
            # update best value and params
            if value < best_value:
                best_value = value
                best_params = projection(interval.center_point, bounds)
                best_params_list.append(best_params)
                best_value_list.append(best_value)
                if interval.inf_pos[1] < interval.inf_pos[1] // kwargs["max_batch_num"] * kwargs["max_batch_num"]:
                    inf_pos = (interval.inf_pos[0] % kwargs["max_batch_num"], kwargs["max_batch_num"])
                else:
                    inf_pos = (interval.inf_pos[0] % kwargs["max_batch_num"], interval.inf_pos[1] % kwargs["max_batch_num"] + 1)

        interval_list += new_interval_list
        # early stopping if the interval containing the best_value is too small
        best_count, smaller_len_count, smaller_vol_count = 0, 0, 0
        for interval in interval_list:
            if interval.center_value == best_value:
                best_count += 1
                if interval.get_volume() < 1e-16:
                    return best_value, best_params, value_list, inf_pos, best_params_list, best_value_list
                    smaller_vol_count += 1
                if interval.get_length(interval.get_longest_side_index_single()[0]) < 1e-6:
                    smaller_len_count += 1
        if best_count > 0 and max(smaller_len_count / best_count, smaller_vol_count / best_count) >= 0.5:
            return best_value, best_params, value_list, inf_pos, best_params_list, best_value_list
        
        # select PLO_list from interval_list, pareto DIRECT
        PLO_list = select_PLO(interval_list)

        value_list.append(best_value)
        iter_time += 1
        # print(f"Iteration {iter_time}: Best value = {best_value}, Evaluations = {eval_count}, PLO count = {len(PLO_list)}")

    return best_value, best_params, value_list, inf_pos, best_params_list, best_value_list

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

def align_with_metric(pred, depth, valid_mask):
    ori_shape = pred.shape
    epsilon = 1e-6
    pred_masked = pred[valid_mask].reshape(-1, 1).cpu().numpy()
    depth_masked = depth[valid_mask].reshape(-1, 1).cpu().numpy()
    inversed_truth = 1.0 / (depth_masked + epsilon)
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    scale, shift = np.linalg.lstsq(A, inversed_truth, rcond=None)[0]
    scale, shift = torch.from_numpy(scale).type_as(pred), torch.from_numpy(shift).type_as(pred)
    # align the prediction
    aligned_pred = 1.0 / (scale * pred + shift + epsilon)
    # clip the aligned prediction to the range [min_depth, max_depth]
    aligned_pred = torch.clamp(aligned_pred, min=min_depth, max=max_depth).reshape(ori_shape)
    return aligned_pred

# define different loss functions
def rmse_linear(pred, target, valid_mask=None):
    diff = (pred - target) * valid_mask
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    return rmse
def abs_relative_difference(pred, target, valid_mask=None):
    diff = (pred - target) * valid_mask
    abs_rel = torch.mean(torch.abs(diff) / (target + 1e-8))
    return abs_rel
def delta1_acc_times_minus1(pred, target, valid_mask=None):
    pred, target = pred[valid_mask], target[valid_mask]
    thresh = torch.max((target / pred), (pred / target))
    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    return -d1  # we want to maximize the accuracy, so return negative value

def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)
    mae = torch.mean(torch.abs(diff))

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item(), 'mae':mae.item()}


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
    elif "least_square_disparity" == alignment:
        # convert GT depth -> GT disparity
        gt_disparity, gt_non_neg_mask = depth2disparity(
            depth=gt_depth, return_mask=True
        )
        # LS alignment in disparity space
        pred_non_neg_mask = depth_pred > 0
        valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

        disparity_pred, scale, shift = align_depth_least_square(
            gt_arr=gt_disparity.cpu().numpy(),
            pred_arr=depth_pred.cpu().numpy(),
            valid_mask_arr=valid_nonnegative_mask.cpu().numpy(),
            return_scale_shift=True,
            max_resolution=None,
        )
        # convert to depth
        disparity_pred = np.clip(
            disparity_pred, a_min=1e-3, a_max=None
        )  # avoid 0 disparity
        depth_pred = disparity2depth(disparity_pred)

    depth_pred = torch.clamp(depth_pred, min=align_min_depth, max=align_max_depth)
    depth_pred = torch.clamp(depth_pred, min=1e-6)
    return depth_pred


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

if __name__ == "__main__":
    max_iter = 50
    gamma = 0.2
    perturb_type = "color_shift"
    max_batch_num = 16
    seed = 42
    batch_size = 1
    num_workers = 4
    alignment = "least_square"
    checkpoint_path="/path/to/checkpoints/marigold_depth/"
    denoise_steps = 1
    dataset = "nyud"
    min_depth = 0.1
    max_depth = 10.0
    target_id = 6
    
    optimize_method = reporting_pareto_direct

    # set seed
    set_seed(seed)

    if dataset == 'hypersim':
        valset = HYPERSIM('external/Marigold/data_split/hypersim_depth/filename_list_val_filtered.txt', 'val', '/path/to/datasets/marigold_dataset/hypersim/val')
    elif dataset == 'kitti':
        valset = KITTI('external/Depth_Anything_V2/metric_depth/dataset/splits/kitti/val.txt', 'val')
    elif dataset == 'nyud':
        valset = NYUD('external/Depth_Anything_V2/metric_depth/dataset/splits/nyud-v2/val.txt', 'val', '/path/to/datasets/nyu_depth_v2')
    else:
        raise NotImplementedError
    
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, shuffle=True,
                           generator=torch.Generator().manual_seed(seed))

    info_data = next(iter(valloader))
    pic_size = (info_data["image"].shape[2], info_data["image"].shape[3])
    depth_size = (info_data["depth"].shape[-2], info_data["depth"].shape[-1])
    
    # LOAD MODEL
    marigold_model = MarigoldDepthEstimator(
        checkpoint_path=checkpoint_path,
        device=torch.device("cuda:0"),
    )

    data_iter = tqdm(enumerate(valloader), total=len(valloader))

    total = 0
    metric = 0
    loss_fn = rmse_linear
    # save hyper parameters
    parameters_dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "checkpoint_path": checkpoint_path,
        "alignment": alignment,
        "max_iter": max_iter,
        "gamma": gamma,
        "perturb_type": perturb_type,
        "pic_size": pic_size,
        "start_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "max_batch_num": max_batch_num,
        "kernel_size": 7,  # for motion blur perturbation only
        "reduced_bit_depth": 6,  # for color quantization only
    }

    generator = torch.Generator(device=marigold_model.device).manual_seed(seed) if seed is not None else None

    for i, batch in data_iter:
        if i < target_id:
            continue
        if i > target_id:
            break

        img, depth, valid_mask = batch['image'].cuda().float(), batch['depth'].cuda(), batch['valid_mask'].cuda()
        depth, valid_mask = depth.unsqueeze(1), valid_mask.unsqueeze(1)        

        # We get parameter range here, since different picture has different picture size in Depth Anything V2 evaluation
        parameter_range = get_parameter_range(perturb_type, gamma, pic_size=(img.shape[-2], img.shape[-1]))

        # depth = depth.squeeze()
        # valid_mask = valid_mask.squeeze()

        with torch.no_grad():
            # # debug test
            # abc = da_model.batch_infer(img)
            # print(rmse_linear(abc.squeeze(), depth.squeeze(), valid_mask.squeeze()))
            # exit()

            # optimize
            pareto_kwargs = {
                "perturb_type": perturb_type,
                "model": marigold_model,
                "loss": loss_fn,
                "img": img.float(),
                "gt_depth": depth,
                "valid_mask": valid_mask,
                "alignment": alignment,
                "min_depth": min_depth,
                "max_depth": max_depth,
                "depth_raw_linear": depth,
                "num_inference_steps": denoise_steps,
                "show_pbar": False,
                "generator": generator,
                "seed": seed,
                "max_batch_num": max_batch_num,
                "transform_func": transform_func,
            }
            start_time = time.time()

            best_value, best_params, value_list, inf_pos, best_params_list, best_value_list = optimize_method(
                bounds=parameter_range,
                f_parallel=func_wrapped_parallel,
                max_iter=max_iter,
                **pareto_kwargs,
            )
            end_time = time.time()
            depth_pred_original = marigold_model.batch_infer(img,
                    num_inference_steps=denoise_steps,
                    show_pbar=False,
                    generator=torch.Generator(marigold_model.device).manual_seed(seed),
            )  # depth_pred_original: 1, C, W

            # save the prediction and ground truth
            perturb = Perturbation(perturb_type=perturb_type, transform_func=transform_func)
            params_dict = get_theta_dict(best_params)
            input_image = perturb.apply(img, **params_dict)

            if max_batch_num > 1 and inf_pos[0] >= 0 and inf_pos[1] > 1:
                position, batch_num = inf_pos
                input_image = input_image.repeat(batch_num, 1, 1, 1)
                depth_pred = marigold_model.batch_infer(input_image,
                                                        num_inference_steps=denoise_steps,
                                                        show_pbar=False,
                                                        generator=torch.Generator(marigold_model.device).manual_seed(seed),
                                                        )[position:position+1, :]
            else:
                depth_pred = marigold_model.batch_infer(input_image,
                                                        num_inference_steps=denoise_steps,
                                                        show_pbar=False,
                                                        generator=torch.Generator(marigold_model.device).manual_seed(seed),
                                                        )
            
            if depth_pred.shape[-2:] != depth.shape[-2:]:
                depth_pred_original = F.interpolate(depth_pred_original[:, None], size=depth.shape[-2:], mode='bilinear', align_corners=True).squeeze(1)
                depth_pred = F.interpolate(depth_pred[:, None], size=depth.shape[-2:], mode='bilinear', align_corners=True).squeeze(1)

            if perturb_type == "geometric":
                gt_depth_perturbed = depth * valid_mask
                gt_depth_perturbed = geometric_perturb_for_depth(gt_depth_perturbed, **params_dict)
                mask_tmp = valid_mask.clone().float()
                mask_tmp = perturb.apply(mask_tmp, **params_dict)
                gt_depth_perturbed = gt_depth_perturbed / (mask_tmp + 1e-8)
                valid_mask_perturbed = (mask_tmp >= 0.5)
                gt_depth_perturbed = gt_depth_perturbed.squeeze(1)
                valid_mask_perturbed = valid_mask_perturbed.squeeze(1)

            if alignment == "least_square" or alignment == "least_square_disparity":
                if perturb_type == "geometric":
                    depth_pred = align_pred_with_gt(depth_pred, gt_depth_perturbed, valid_mask, alignment=alignment, align_min_depth=min_depth, align_max_depth=max_depth)
                else:
                    depth_pred = align_pred_with_gt(depth_pred, depth, valid_mask, alignment=alignment, align_min_depth=min_depth, align_max_depth=max_depth)
                depth_pred_original = align_pred_with_gt(depth_pred_original, depth, valid_mask, alignment=alignment, align_min_depth=min_depth, align_max_depth=max_depth)

            depth_pred = depth_pred.squeeze(1)
            depth = depth.squeeze(1)
            valid_mask = valid_mask.squeeze(1)
            depth_pred_original = depth_pred_original.squeeze(1)

            # eval perturbed loss
            if isinstance(loss_fn, torch.nn.modules.loss._Loss):
                if valid_mask is not None:
                    if perturb_type == "geometric":
                        current_loss = loss_fn(depth_pred[valid_mask_perturbed], gt_depth_perturbed[valid_mask_perturbed])
                    else:
                        current_loss = loss_fn(depth_pred[valid_mask], depth[valid_mask])

                    original_loss = loss_fn(depth_pred_original[valid_mask], depth[valid_mask])
                else:
                    if perturb_type == "geometric":
                        current_loss = loss_fn(depth_pred, gt_depth_perturbed)
                    else:
                        current_loss = loss_fn(depth_pred, depth)

                    original_loss = loss_fn(depth_pred_original, depth)
            else:
                if perturb_type == "geometric":
                    current_loss = loss_fn(depth_pred, gt_depth_perturbed, valid_mask_perturbed)
                else:
                    current_loss = loss_fn(depth_pred, depth, valid_mask)
                original_loss = loss_fn(depth_pred_original, depth, valid_mask)

            metric += best_value * -1
            total += 1
            print(f"iteration: {total-1}, best_value: {best_value}, best_params: {best_params}")
            # print(f"Time taken for optimization: {end_time - start_time} seconds")
            print(f"Original RMSE: {original_loss.item()}, Perturbed RMSE: {current_loss.item()}, Delta: {current_loss.item() - original_loss.item()}")

            save_path = "./experiment_results/draw_optimize_sequence.txt"
            with open(save_path, "a") as f:
                f.write(f"best_value: {best_value}, best_params: {best_params}\n")
                f.write(f"perturb type: {perturb_type}, dataset: {dataset}, sample_id: {target_id}\n")
                f.write(f"best value list: {best_value_list}\n")
                f.write(f"best params list: {best_params_list}\n")            
    print(f"Final RMSE linear: {metric/total}")
    print(f"Max Iterations: {max_iter}")
