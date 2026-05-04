from src.inference_wrapper.zoedepth.zoedepth_dataset import KITTI, NYUD, HYPERSIM

from src.optimization.func_wrapped import func_wrapped, _func_wrapped_for_scipy_direct, func_wrapped_parallel
from src.optimization.scipy_optimize import (
    direct_optimization,
    dual_annealing_optimization
)
from src.inference_wrapper.base import DepthEstimator
from src.inference_wrapper.zoedepth.zoedepth_pipeline import zoedepthEstimator

from src.optimization.direct import Interval, projection, select_PLO

from src.perturbation import get_parameter_range, Perturbation, geometric_perturb_for_depth

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
from matplotlib import pyplot as plt

import time
import datetime
import os
from src.utils.seed import set_seed

evaluate_num_list = []

def trisect_interval_pareto(interval:Interval, bounds, f, **kwargs):
    """
    Trisect an interval into child intervals.
    Following the pareto DIRECT algorithm
    Input:
        interval: Interval, the interval to be trisected
        bounds: list of tuples, [(min1, max1), (min2, max2), ...]
        f: function to minimize, which takes in a list of parameter lists and returns a list
        **kwargs: additional keyword arguments to pass to the function
    Return:
        child_interval_list: list[Interval], the child intervals after trisection. REMIND: the center_value is None and needs to be evaluated.
    """
    pass

@torch.no_grad()
def pareto_direct(bounds, f_parallel, max_iter=1000, **kwargs):
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
    best_value = initial_interval.center_value
    best_params = projection(initial_interval.center_point, bounds)
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
        to_evaluate_points = []
        father_interval = []
        theta_list = []

        for interval in PLO_list:
            # new_intervals = trisect_interval_pareto(interval, bounds, f_parallel, **kwargs)
            dim_to_split_list = interval.get_longest_side_index_single()

            for i in range(len(dim_to_split_list)):
                dim = dim_to_split_list[i]
                delta = interval.get_length(dim) / 3
                theta1 = interval.center_point.copy()
                theta1[dim] = interval.center_point[dim] - delta
                theta2 = interval.center_point.copy()
                theta2[dim] = interval.center_point[dim] + delta
                to_evaluate_points.append((theta1, theta2, dim))
                # (left_point, right_point, dim)
                father_interval.append(interval)

        for theta1, theta2, dim in to_evaluate_points:
            theta_list.append(projection(theta1, bounds))
            theta_list.append(projection(theta2, bounds))

        values = f_parallel(theta_list, parallel_num=len(theta_list), **kwargs)
        evaluate_num_list.append(len(theta_list))

        for i in range(len(to_evaluate_points)):
            to_evaluate_points[i] = (values[2 * i], values[2 * i + 1], to_evaluate_points[i][2])

        child_list = []
        for idx, interval in enumerate(father_interval):
            dim_to_split_list = interval.get_longest_side_index_single()
            for dim_to_split in dim_to_split_list:
                length = interval.get_length(dim_to_split)
                child_bounds = interval.bounds.copy()
                # get right child
                sign = 1
                child_bounds[dim_to_split] = (
                    interval.bounds[dim_to_split][0],
                    interval.bounds[dim_to_split][0] + length / 3 * sign,
                )
                child_center_point = interval.center_point.copy()
                child_center_point[dim_to_split] = interval.center_point[dim_to_split] + length / 3 * sign
                child = Interval(
                                    size=0,
                                    center_value=to_evaluate_points[idx][1],
                                    bounds=child_bounds,
                                    center_point=child_center_point,
                                )
                child.split_counter = [0] * len(child_bounds)
                child.update_size()
                child.inf_pos = (2 * idx + 1, 2 * len(to_evaluate_points))
                child_list.append(child)

                # get left child
                sign = -1
                child_bounds = interval.bounds.copy()
                child_bounds[dim_to_split] = (
                    interval.bounds[dim_to_split][1] + length / 3 * sign,
                    interval.bounds[dim_to_split][1],
                )

                child_center_point = interval.center_point.copy()
                child_center_point[dim_to_split] = interval.center_point[dim_to_split] + length / 3 * sign
                child = Interval(
                                    size=0,
                                    center_value=to_evaluate_points[idx][0],
                                    bounds=child_bounds,
                                    center_point=child_center_point,
                                )
                child.split_counter = [0] * len(child_bounds)
                child.update_size()
                child.inf_pos = (2 * idx, 2 * len(to_evaluate_points))
                child_list.append(child)

                # update the current interval
                interval.bounds[dim_to_split] = (
                    interval.bounds[dim_to_split][0] + length / 3,
                    interval.bounds[dim_to_split][1] - length / 3,
                )
                interval.split_counter[dim_to_split] += 1
                interval.update_size()

        new_interval_list += child_list

        for interval in new_interval_list:
            eval_count += 1
            value = interval.center_value
            # update best value and params
            if value < best_value:
                best_value = value
                best_params = projection(interval.center_point, bounds)
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
                    return best_value, best_params, value_list, inf_pos
                    smaller_vol_count += 1
                if interval.get_length(interval.get_longest_side_index_single()[0]) < 1e-6:
                    smaller_len_count += 1
        if best_count > 0 and max(smaller_len_count / best_count, smaller_vol_count / best_count) >= 0.5:
            return best_value, best_params, value_list, inf_pos
        
        # select PLO_list from interval_list, pareto DIRECT
        PLO_list = select_PLO(interval_list)

        value_list.append(best_value)
        iter_time += 1
        # print(f"Iteration {iter_time}: Best value = {best_value}, Evaluations = {eval_count}, PLO count = {len(PLO_list)}")

    return best_value, best_params, value_list, inf_pos


args = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

args.add_argument('--dataset', default='hypersim', choices=['hypersim', 'kitti', 'nyud'])
args.add_argument('--min-depth', default=0.1, type=float)
args.add_argument('--max-depth', default=20, type=float)
args.add_argument('--pretrained-from', type=str, required=True)
args.add_argument('--alignment', type=str, default="None", help='alignment method. Choices: None, "least_square", "least_square_disparity". Default None.')

args.add_argument("--max_iter", type=int, default=100,
                  help="maximum number of iterations for optimization, default 100.",
                  )
args.add_argument("--gamma", type=float, default=0.2,
                  help="parameter to control the range of perturbation",
                  )
args.add_argument("--perturb_type", type=str, default="color_shift",
                  required=True,
                  help="type of perturbation. Choices: 'geometric', 'color_shift', 'motion_blur'.",
                  )
args.add_argument("--root_save_dir", type=str,
                  required=True,
                  help="root directory to save the results.",
                  )
args.add_argument("--optimize_method", type=str, default="original_direct",
                    help="optimization method. Choices: 'direct', 'dual_annealing', 'pareto_direct', 'original_direct'. Default 'original_direct'.",
                  )
args.add_argument("--max_batch_num", type=int, default=8,
                    help="maximum number of instances in a batch for parallel optimization. Default 8.",
                  )
args.add_argument("--target_function", type=str, default="rmse_l",
                  help="target function for optimization. Choices: 'rmse_l', 'abs_rel', 'mse_loss', 'delta1'. Default 'rmse_linear'.",
                  )
args.add_argument("--kernel_size", type=int, default=9,
                  help="kernel size for motion blur perturbation, default 9. Only used when perturb_type is 'motion_blur'.",
                  )
args.add_argument("--reduced_bit_depth", type=int, default=6,
                  help="reduced_bit_depth for color quantization, default 6. Only used when perturb_type is 'banding'.",
                  )
args.add_argument("--seed", type=int, default=42,
                  help="random seed for reproducibility, default 42.",
                  )
args.add_argument("--batch_size", type=int, default=1,
                  help="batch size for dataloader, default 1. Usually set to 1 for Marigold model.",
                  )
args.add_argument("--num_workers", type=int, default=4,
                  help="number of workers for dataloader, default 4.",
                  )

args.add_argument("--min_data_idx", type=int, default=0,
                  help="minimum data index to start from, default 0.",
                  )
args.add_argument("--max_data_idx", type=int, default=60,
                    help="maximum data index to end at, default 60.",
                    )
args.add_argument("--skip_existing", action="store_true",
                    help="whether to skip the data that has already been processed, default False.",
                    )
args = args.parse_args()

if hasattr(args, "kernel_size"):
    kernel_size = args.kernel_size
else:
    kernel_size = 0

if hasattr(args, "reduced_bit_depth"):
    reduced_bit_depth = args.reduced_bit_depth  # for color quantization only
else:
    reduced_bit_depth = 0

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
    aligned_pred = torch.clamp(aligned_pred, min=args.min_depth, max=args.max_depth).reshape(ori_shape)
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

def transform_func(img_to_trans: torch.Tensor, mode: str="to"):
    """
    Transform the image to the normalized range [0, 1] when mode is "to", and transform back when mode is "back".
    This function should not change img_to_trans, which in other words, it should be out-of-place.
    """
    img = img_to_trans.clone()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    num_dim = len(img.shape)
    if num_dim == 4:
        if mode == "to":
            for dim in range(img.shape[1]):
                img[:, dim, :, :] = img[:, dim, :, :] * std[dim] + mean[dim]
            img = torch.clamp(img, 0, 1)
            return img
        elif mode == "back":
            for dim in range(img.shape[1]):
                img[:, dim, :, :] = (img[:, dim, :, :] - mean[dim]) / std[dim]
            return img
    elif num_dim == 3:
        if mode == "to":
            for dim in range(img.shape[0]):
                img[dim, :, :] = img[dim, :, :] * std[dim] + mean[dim]
            img = torch.clamp(img, 0, 1)
            return img
        elif mode == "back":
            for dim in range(img.shape[0]):
                img[dim, :, :] = (img[dim, :, :] - mean[dim]) / std[dim]
            return img
    else:
        raise ValueError("Invalid image shape, should be 3 or 4 dimensions.")


if __name__ == "__main__":
    max_iter = args.max_iter
    gamma = args.gamma
    perturb_type = args.perturb_type
    root_save_dir = args.root_save_dir
    optimize_method = args.optimize_method
    max_batch_num = args.max_batch_num
    target_function = args.target_function
    seed = args.seed
    batch_size = args.batch_size
    num_workers = args.num_workers
    alignment = args.alignment
    skip_existing = args.skip_existing
    
    if args.optimize_method not in ["direct", "dual_annealing", "pareto_direct", "original_direct"]:
        raise ValueError("Invalid optimize_method. Choices: 'direct', 'dual_annealing', 'pareto_direct', 'original_direct'.")
    if args.optimize_method == "original_direct":
        # optimize_method = original_direct
        pass
    elif args.optimize_method == "pareto_direct":
        optimize_method = pareto_direct

    # set seed
    set_seed(seed)

    if args.dataset == 'hypersim':
        valset = HYPERSIM('external/Marigold/data_split/hypersim_depth/filename_list_val_filtered.txt', 'val', '/path/to/datasets/marigold_dataset/hypersim/val')
    elif args.dataset == 'kitti':
        valset = KITTI('external/Depth_Anything_V2/metric_depth/dataset/splits/kitti/val.txt', 'val')
    elif args.dataset == 'nyud':
        valset = NYUD('external/Depth_Anything_V2/metric_depth/dataset/splits/nyud-v2/val.txt', 'val', '/path/to/datasets/nyu_depth_v2')
    else:
        raise NotImplementedError
    
    generator = torch.Generator(device=torch.device("cpu")).manual_seed(seed) if seed is not None else None
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True,
                           shuffle=True, generator=generator)

    info_data = next(iter(valloader))
    pic_size = (info_data["image"].shape[2], info_data["image"].shape[3])
    depth_size = (info_data["depth"].shape[-2], info_data["depth"].shape[-1])
    
    zoe_model = zoedepthEstimator(
        checkpoint_path=args.pretrained_from,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    data_iter = tqdm(enumerate(valloader), total=len(valloader))

    total = 0
    metric = 0
    # loss_fn = rmse_linear
    if target_function == "rmse_l":
        loss_fn = rmse_linear
    elif target_function == "abs_rel":
        loss_fn = abs_relative_difference
    elif target_function == "mse_loss":
        loss_fn = torch.nn.MSELoss()
    elif target_function == "delta1":
        loss_fn = delta1_acc_times_minus1
    else:
        raise ValueError("Invalid target_function. Choices: 'rmse_linear', 'abs_relative_difference', 'MSE_loss', 'delta1'.")
    
    if hasattr(loss_fn, "__name__"):
        metric_name = loss_fn.__name__
    else:
        metric_name = args.target_function

    base_saved_dir = os.path.join(root_save_dir, args.dataset)
    # save hyper parameters
    parameters_dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "checkpoint_path": args.pretrained_from,
        "alignment": alignment,
        "max_iter": max_iter,
        "gamma": gamma,
        "perturb_type": perturb_type,
        "pic_size": pic_size,
        "start_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "loss_function": metric_name,
        "seed": seed,
        "optimize_method": args.optimize_method,
        "max_batch_num": max_batch_num,
        "kernel_size": kernel_size,  # for motion blur perturbation only
        "reduced_bit_depth": reduced_bit_depth,  # for color quantization only
    }

    for i, batch in data_iter:

        if i < args.min_data_idx:
            total += 1
            continue

        if i >= args.max_data_idx:
            break

        if skip_existing:
            if os.path.exists(os.path.join(base_saved_dir, perturb_type, f"{total}", f"depth_pred_{args.optimize_method}.npy")):
                # print(f"Skipping existing data at index {total}")
                total += 1
                continue

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
            if args.optimize_method in ["original_direct", "pareto_direct"]:
                pareto_kwargs = {
                    "perturb_type": perturb_type,
                    "model": zoe_model,
                    "loss": loss_fn,
                    "img": img,
                    "gt_depth": depth,
                    "valid_mask": valid_mask,
                    "alignment": alignment,
                    "min_depth": args.min_depth,
                    "max_depth": args.max_depth,
                    "depth_raw_linear": depth,
                    "seed": seed,
                    "max_batch_num": max_batch_num,
                    "kernel_size": kernel_size,  # for motion blur perturbation only
                    "reduced_bit_depth": reduced_bit_depth,  # for color quantization only
                    "transform_func": transform_func,
                }
                start_time = time.time()
                # # pareto direct
                # best_value, best_params, value_list = pareto_direct(
                #     bounds=parameter_range,
                #     f_parallel=func_wrapped_parallel,
                #     max_iter=max_iter,
                #     **pareto_kwargs,
                # )

                best_value, best_params, value_list, inf_pos = optimize_method(
                    bounds=parameter_range,
                    f_parallel=func_wrapped_parallel,
                    max_iter=max_iter,
                    **pareto_kwargs,
                )
                end_time = time.time()
            elif args.optimize_method in ["direct", "dual_annealing"]:
                optimize_args = (
                    perturb_type,
                    zoe_model,
                    loss_fn,
                    img,
                    depth,
                    valid_mask,
                    alignment,
                    args.min_depth,
                    args.max_depth,
                    depth,
                    True,
                    seed,
                    kernel_size,
                    reduced_bit_depth,
                    transform_func,
                )

                start_time = time.time()
                if args.optimize_method == "direct":
                    # by scipy direct
                    best_value, best_params = direct_optimization(
                        parameter_range,
                        _func_wrapped_for_scipy_direct,
                        max_iter,
                        *optimize_args,
                    )
                    best_params = best_params.tolist()

                elif args.optimize_method == "dual_annealing":  
                    # by scipy dual_annealing
                    best_value, best_params = dual_annealing_optimization(
                        parameter_range,
                        _func_wrapped_for_scipy_direct,
                        max_iter,
                        *optimize_args,
                    )
                    best_params = best_params.tolist()

                end_time = time.time()

            depth_pred_original = zoe_model.infer(img)  # depth_pred_original: 1, C, W
            # valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
            
            # if valid_mask.sum() < 10:
            #     continue


            # save the prediction and ground truth
            perturb = Perturbation(perturb_type=perturb_type, transform_func=transform_func)
            params_dict = get_theta_dict(best_params)
            input_image = perturb.apply(img, **params_dict)

            if max_batch_num > 1 and args.optimize_method in ["original_direct", "pareto_direct"] and inf_pos[0] >= 0 and inf_pos[1] > 1:
                position, batch_num = inf_pos
                input_image = input_image.repeat(batch_num, 1, 1, 1)
                depth_pred = zoe_model.batch_infer(input_image)[position:position+1, :]
            else:
                depth_pred = zoe_model.batch_infer(input_image)
            
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

            if args.alignment == "alignment":
                if perturb_type == "geometric":
                    depth_pred = align_with_metric(depth_pred, gt_depth_perturbed, valid_mask)
                else:
                    depth_pred = align_with_metric(depth_pred, depth, valid_mask)
                depth_pred_original = align_with_metric(depth_pred_original, depth, valid_mask)

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
            print(f"original loss: {original_loss.item():.4f}, optimized loss: {current_loss.item():.4f}")
            print(f"delta: {original_loss.item() - current_loss.item():.4f}")

        os.makedirs("./src/others/tmp/parallel_num", exist_ok=True)
        with open(f"./src/others/tmp/parallel_num/pareto_evaluate_num_list_{i}.txt", "w") as f:
            for num in evaluate_num_list:
                f.write(f"{num}\t")
        