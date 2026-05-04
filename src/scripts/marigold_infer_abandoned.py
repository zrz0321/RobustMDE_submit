import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt

from src.inference_wrapper.base import DepthEstimator
from src.inference_wrapper.marigold.marigold_pipeline import MarigoldDepthEstimator

from src.optimization.func_wrapped import func_wrapped, _func_wrapped_for_scipy_direct, func_wrapped_parallel
from src.optimization.scipy_optimize import (
    direct_optimization,
    dual_annealing_optimization
)
from src.optimization.direct import pareto_direct, original_direct

from src.perturbation import get_parameter_range, Perturbation, geometric_perturb_for_depth

from external.Marigold.src.dataset.base_depth_dataset import BaseDepthDataset, DatasetMode
from external.Marigold.src.dataset import get_dataset
from external.Marigold.marigold.marigold_depth_pipeline import MarigoldDepthOutput
from external.Marigold.src.util.metric import rmse_linear, abs_relative_difference, delta1_acc, threshold_percentage
from external.Marigold.src.util.alignment import (
    align_depth_least_square,
    depth2disparity,
    disparity2depth,
)

import time
import datetime
from src.utils.save_tools import SaveTools, save_json_to_file
from src.utils.seed import set_seed

from src.evaluation.metrics import delta1_acc_times_minus1


import argparse
args = argparse.ArgumentParser()
args.add_argument("--dataset_config", type=str,
                  required=True,
                  help="path to the config file of the evaluation dataset of Marigold model.",
                  )
args.add_argument("--base_data_dir", type=str,
                  required=True,
                  help="base path to the datasets, following Marigold's setting.",
                  )
args.add_argument("--batch_size", type=int, default=1,
                  help="batch size for dataloader, default 1. Usually set to 1 for Marigold model.",
                  )
args.add_argument("--num_workers", type=int, default=4,
                  help="number of workers for dataloader, default 4.",
                  )
args.add_argument("--checkpoint_path", type=str,
                  required=True,
                  help="path to the checkpoint of Marigold model.",
                  )
args.add_argument("--seed", type=int, default=42,
                  help="random seed for reproducibility, default 42.",
                  )
args.add_argument("--denoise_steps", type=int, default=1,
                  help="number of denoising steps for Marigold model, default 1, following Marigold's setting.",
                  )
# args.add_argument("--processing_res", type=int, default=0)
# args.add_argument("--match_input_res", type=bool, default=False)
args.add_argument("--resample_method", type=str, default="bilinear")
args.add_argument("--alignment", type=str, default="least_square",
                  help="alignment method for depth prediction, following Marigold's setting. Default 'least_square'. Choices: None, 'least_square', 'least_square_disparity'.",
                  )
args.add_argument("--alignment_max_res", type=int, default=None)

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
            max_resolution=alignment_max_res,
        )
        # convert to depth
        disparity_pred = np.clip(
            disparity_pred, a_min=1e-3, a_max=None
        )  # avoid 0 disparity
        depth_pred = disparity2depth(disparity_pred)

    depth_pred = torch.clamp(depth_pred, min=align_min_depth, max=align_max_depth)
    depth_pred = torch.clamp(depth_pred, min=1e-6)
    return depth_pred

# CUDA_VISIBLE_DEVICES=0 python src/scripts/marigold_infer.py --dataset_config ./external/Marigold/config/dataset_depth/data_nyu_test.yaml \
# -- base_data_dir /path/to/datasets/nyu-d-marigold --checkpoint_path /path/to/checkpoints/marigold_depth/ --perturb_type color_shift --max_iter 100 --gamma 0.2 --seed 42

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
    # params
    # dataset_config = "./external/Marigold/config/dataset_depth/data_nyu_test.yaml"
    # base_data_dir = "/path/to/datasets/nyu-d-marigold"
    # batch_size = 1
    # num_workers = 0
    # checkpoint_path = "/path/to/checkpoints/marigold_depth/"
    # seed = 42

    # denoise_steps = 1
    # processing_res = 0
    # match_input_res = False
    # resample_method = "bilinear"
    # generator = None

    # alignment = "least_square"
    # alignment_max_res = None

    # max_iter = 100  # for optimization
    # gamma = 0.2 # for theta
    # # perturb_type = "motion_blur"
    # perturb_type = "color_shift"
    # root_save_dir = "/path/to/experiment_results/marigold"

    dataset_config = args.dataset_config
    base_data_dir = args.base_data_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    checkpoint_path = args.checkpoint_path
    seed = args.seed
    denoise_steps = args.denoise_steps
    processing_res = 0
    match_input_res = False
    resample_method = args.resample_method

    alignment = args.alignment
    alignment_max_res = args.alignment_max_res
    max_iter = args.max_iter
    gamma = args.gamma
    perturb_type = args.perturb_type
    root_save_dir = args.root_save_dir
    max_batch_num = args.max_batch_num

    skip_existing = args.skip_existing

    if args.optimize_method not in ["direct", "dual_annealing", "pareto_direct", "original_direct"]:
        raise ValueError("Invalid optimize_method. Choices: 'direct', 'dual_annealing', 'pareto_direct', 'original_direct'.")
    if args.optimize_method == "original_direct":
        optimize_method = original_direct
    elif args.optimize_method == "pareto_direct":
        optimize_method = pareto_direct

    # set seed
    set_seed(seed)

    # LOAD DATASET
    cfg_data = OmegaConf.load(dataset_config)
    dataset : BaseDepthDataset = get_dataset(
        cfg_data,
        base_data_dir=base_data_dir,
        mode=DatasetMode.EVAL,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    info_data = next(iter(dataloader))
    pic_size = (info_data["rgb_norm"].shape[2], info_data["rgb_norm"].shape[3])

    # LOAD MODEL
    marigold_model = MarigoldDepthEstimator(
        checkpoint_path=checkpoint_path,
        device=torch.device("cuda:0"),
    )

    total = 0
    metric = 0
    # loss_fn = rmse_linear
    if args.target_function == "rmse_l":
        loss_fn = rmse_linear
    elif args.target_function == "abs_rel":
        loss_fn = abs_relative_difference
    elif args.target_function == "mse_loss":
        loss_fn = torch.nn.MSELoss()
    elif args.target_function == "delta1":
        loss_fn = delta1_acc_times_minus1
    else:
        raise ValueError("Invalid target_function. Choices: 'rmse_linear', 'abs_relative_difference', 'MSE_loss'.")
    
    if hasattr(loss_fn, "__name__"):
        metric_name = loss_fn.__name__
    else:
        metric_name = args.target_function

    base_saved_dir = os.path.join(root_save_dir, args.dataset)
    # save hyper parameters
    parameters_dict = {
        "dataset_config": dataset_config,
        "base_data_dir": base_data_dir,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "checkpoint_path": checkpoint_path,
        "num_inference_steps": denoise_steps,
        "processing_res": processing_res,
        "match_input_res": match_input_res,
        "resample_method": resample_method,
        "alignment": alignment,
        "alignment_max_res": alignment_max_res,
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
    # save the parameters dict for this entire experiment
    if not os.path.exists(base_saved_dir):
        os.makedirs(base_saved_dir, exist_ok=True)
    save_json_to_file(parameters_dict, os.path.join(base_saved_dir, f"parameters_{perturb_type}_{args.optimize_method}.json"))

    # get parameter range
    parameter_range = get_parameter_range(perturb_type, gamma, pic_size=pic_size)

    generator = torch.Generator(device=marigold_model.device).manual_seed(seed) if seed is not None else None
    # INFERENCE
    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"Depth Inference on {dataset.disp_name}", leave=True
        ):
            if total < args.min_data_idx:
                total += 1
                continue

            if total >= args.max_data_idx:
                break

            if skip_existing:
                if os.path.exists(os.path.join(base_saved_dir, perturb_type, f"{total}", f"depth_pred_{args.optimize_method}.npy")):
                    # print(f"Skipping existing data at index {total}")
                    total += 1
                    continue

            # rgb_int = batch["rgb_int"].to(marigold_model.device)
            rgb_norm = batch["rgb_norm"].to(marigold_model.device)
            gt_depth = batch['depth_raw_linear'].to(marigold_model.device)
            valid_mask = batch['valid_mask_raw'].to(marigold_model.device)
            rgb_relatve_path = batch["rgb_relative_path"]

            # optimize
            if args.optimize_method in ["original_direct", "pareto_direct"]:
                pareto_kwargs = {
                    "perturb_type": perturb_type,
                    "model": marigold_model,
                    "loss": loss_fn,
                    "img": rgb_norm.float(),
                    "gt_depth": gt_depth,
                    "valid_mask": valid_mask,
                    "alignment": alignment,
                    "min_depth": dataset.min_depth,
                    "max_depth": dataset.max_depth,
                    "depth_raw_linear": gt_depth,
                    "num_inference_steps": denoise_steps,
                    "show_pbar": False,
                    "generator": generator,
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
                    marigold_model,
                    loss_fn,
                    rgb_norm.float(),
                    gt_depth,
                    valid_mask,
                    alignment,
                    dataset.min_depth,
                    dataset.max_depth,
                    gt_depth,
                    True,
                    seed,
                    kernel_size,
                    reduced_bit_depth,
                    transform_func,
                    {
                        "num_inference_steps": denoise_steps,
                        "show_pbar": False,
                        "generator": generator,
                    },
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

            # for those metric that larger is better
            if loss_fn in [delta1_acc_times_minus1]:
                best_value = -1 * best_value  # change back to positive value

            save_tool = SaveTools(base_dir=base_saved_dir, exp_name=os.path.join(perturb_type, f"{total}"))

            # save the prediction and ground truth
            perturb = Perturbation(perturb_type=perturb_type, transform_func=transform_func)
            params_dict = get_theta_dict(best_params)
            input_image = perturb.apply(rgb_norm.float(), **params_dict)

            # reproduce the result
            if max_batch_num > 1 and args.optimize_method in ["original_direct", "pareto_direct"] and inf_pos[0] >= 0 and inf_pos[1] > 1:
                position, batch_num = inf_pos[0], inf_pos[1]
                input_image = input_image.repeat(batch_num, 1, 1, 1)
                depth_pred = marigold_model.batch_infer(
                    input_image,
                    num_inference_steps=denoise_steps,
                    show_pbar=False,
                    generator=torch.Generator(marigold_model.device).manual_seed(seed),
                )
                depth_pred = depth_pred[position:position+1, :, :, :]
            else:
                # not parallel
                depth_pred = marigold_model.batch_infer(
                    input_image,
                    num_inference_steps=denoise_steps,
                    show_pbar=False,
                    generator=torch.Generator(marigold_model.device).manual_seed(seed),
                )

            depth_pred_original = marigold_model.batch_infer(
                rgb_norm.float(),
                num_inference_steps=denoise_steps,
                show_pbar=False,
                generator=torch.Generator(marigold_model.device).manual_seed(seed),
            )

            if perturb_type == "geometric":
                # valid_mask = perturb.apply(valid_mask.float(), **params_dict, geometric_mode='nearest').bool()
                # gt_depth = perturb.apply(gt_depth, **params_dict)
                gt_depth_perturbed = gt_depth * valid_mask
                # gt_depth_perturbed = perturb.apply(gt_depth_perturbed, **params_dict)
                gt_depth_perturbed = geometric_perturb_for_depth(gt_depth_perturbed, **params_dict)

                mask_tmp = valid_mask.clone().float()
                mask_tmp = perturb.apply(mask_tmp, **params_dict)
                gt_depth_perturbed = gt_depth_perturbed / (mask_tmp + 1e-8)
                valid_mask_perturbed = (mask_tmp >= 0.5)

                gt_depth_perturbed = gt_depth_perturbed.squeeze(1)  # (B, H, W)
                valid_mask_perturbed = valid_mask_perturbed.squeeze(1)

            depth_pred = depth_pred.squeeze(1)  # (B, H, W)
            depth_pred_original = depth_pred_original.squeeze(1)
            valid_mask = valid_mask.squeeze(1)
            gt_depth = gt_depth.squeeze(1)

            if perturb_type == "geometric":
                depth_pred = align_pred_with_gt(depth_pred, gt_depth_perturbed, valid_mask_perturbed, alignment, align_min_depth=dataset.min_depth, align_max_depth=dataset.max_depth)
            else:
                depth_pred = align_pred_with_gt(depth_pred, gt_depth, valid_mask, alignment, align_min_depth=dataset.min_depth, align_max_depth=dataset.max_depth)

            depth_pred_original = align_pred_with_gt(depth_pred_original, gt_depth, valid_mask, alignment, align_min_depth=dataset.min_depth, align_max_depth=dataset.max_depth)

            # eval perturbed loss
            if isinstance(loss_fn, torch.nn.modules.loss._Loss):
                if valid_mask is not None:
                    if perturb_type == "geometric":
                        current_loss = loss_fn(depth_pred[valid_mask_perturbed], gt_depth_perturbed[valid_mask_perturbed])
                    else:
                        current_loss = loss_fn(depth_pred[valid_mask], gt_depth[valid_mask])

                    original_loss = loss_fn(depth_pred_original[valid_mask], gt_depth[valid_mask])
                else:
                    if perturb_type == "geometric":
                        current_loss = loss_fn(depth_pred, gt_depth_perturbed)
                    else:
                        current_loss = loss_fn(depth_pred, gt_depth)

                    original_loss = loss_fn(depth_pred_original, gt_depth)
            else:
                if perturb_type == "geometric":
                    current_loss = loss_fn(depth_pred, gt_depth_perturbed, valid_mask_perturbed)
                else:
                    current_loss = loss_fn(depth_pred, gt_depth, valid_mask)
                original_loss = loss_fn(depth_pred_original, gt_depth, valid_mask)

            current_metric = {
                "metric_name": metric_name,
                "perturbation_type": perturb_type,
                "best_value": best_value * -1,
                "best_params": best_params,
                "time_taken_seconds": end_time - start_time,
                "rgb_image_path": rgb_relatve_path,
                "date_and_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "perturbed_metric": current_loss.item(),
                "original_metric": original_loss.item(),
            }


            save_tool.save_config(current_metric, additional_name=f"{args.optimize_method}")
            if not os.path.exists(os.path.join(save_tool.save_dir, "depth_pred_original.npy")):
                save_tool.save_as_npy(depth_pred_original, save_tool.save_dir, "depth_pred_original")
            if not os.path.exists(os.path.join(save_tool.save_dir, "gt_depth.npy")):
                save_tool.save_as_npy(gt_depth, save_tool.save_dir, "gt_depth")
            if not os.path.exists(os.path.join(save_tool.save_dir, "valid_mask.npy")):
                save_tool.save_as_npy(valid_mask.cpu().numpy(), save_tool.save_dir, "valid_mask")
                
            save_tool.save_as_npy(depth_pred, save_tool.save_dir, f"depth_pred_{args.optimize_method}")

            # draw the convergence curve
            if not args.optimize_method in ["direct", "dual_annealing"]:
                plt.figure()
                plt.plot([-v for v in value_list])
                plt.xlabel("Iteration")
                plt.ylabel("metric")
                plt.title(f"Convergence curve for {perturb_type} on {dataset.disp_name}, using {args.optimize_method}")
                save_tool.save_plot(plt, name=f"convergence_curve_{perturb_type}_{args.optimize_method}.png")
                plt.close()

            # draw difference map
            depth_diff = torch.abs(depth_pred - gt_depth).squeeze().cpu().numpy()
            depth_diff[~valid_mask.squeeze().cpu().numpy()] = 0
            save_tool.save_heat_map(depth_diff, name=f"depth_difference_{perturb_type}_{args.optimize_method}.png")

            metric += best_value * -1
            total += 1
            print(f"iteration: {total-1}, best_value: {best_value}, best_params: {best_params}")
            # print(f"Time taken for optimization: {end_time - start_time} seconds")
            print(f"Original {metric_name}: {original_loss.item()}, Perturbed {metric_name}: {current_loss.item()}, Delta: {current_loss.item() - original_loss.item()}")
            
    print(f"Final {args.target_function} linear: {metric/total}")
    print(f"Max Iterations: {max_iter}")
