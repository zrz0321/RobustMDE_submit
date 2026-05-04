import os
import numpy as np
import torch
from tqdm import tqdm

from external.Marigold.src.util.metric import *
import json
import argparse

args = argparse.ArgumentParser()
args.add_argument('--root_dir', type=str, required=False, default="/path/to/experiment_results/marigold/nyu_depth_test_full",
                  help='Root directory containing experiment results')
args.add_argument('--perturb_type', type=str, required=False, default="color_shift",
                    help='Type of perturbation applied (e.g., color_shift, motion_blur, geometric, banding)')
args.add_argument('--optimize_method', type=str, required=False, default="pareto_direct",
                  help='Type of optimization method appled (e.g., original_direct, pareto_direct, direct, dual_annealing)')
args.add_argument("--dataset_name", type=str, required=False, default="nyud",
                  help='Evaluation dataset')
args.add_argument("--model_name", type=str, required=False, default="marigold",
                  help="Model to evaluation")
args.add_argument("--start_idx", type=str, required=False, default=0,
                  help="The start index to begin evaluation, default is 0")
args.add_argument("--end_idx", type=str, required=False, default=1000000,
                  help="The end index to close evaluation, default is 1000000")
args.add_argument('--save_dir', type=str, required=False, default=None,
                  help='Directory to save the evaluation results, default is None, which will save to experiment_results/<model_name>/<dataset_name>/<perturb_type>/eval_<optimize_method>_result.json')
args = args.parse_args()

if __name__ == "__main__":
    # perturb_type = "color_shift"
    # perturb_type = "motion_blur"
    # perturb_type = "geometric"
    # optimize_method = "original_direct"
    # optimize_method = "pareto_direct"
    # optimize_method = "direct"
    # optimize_method = "dual_annealing"
    # dataset_name = "nyud"
    # dataset_name = "kitti-eigen"
    # model_name = "marigold"

    root_dir = args.root_dir
    perturb_type = args.perturb_type
    optimize_method = args.optimize_method
    dataset_name = args.dataset_name
    model_name = args.model_name
    start_idx = args.start_idx
    end_idx = args.end_idx

    if dataset_name == "nyud":
        root_dir = "/path/to/experiment_results/marigold/nyu_depth_test_full"
    elif dataset_name == "kitti-eigen":
        root_dir = "/path/to/experiment_results/marigold/kitti_depth_eigen_test_full"
    else:
        raise ValueError("Unknown dataset name")
    # target_function = "delta1"
    # root_dir = f"/path/to/experiment_results/marigold_ablation/target_{target_function}/nyu_depth_test_full"

    if args.save_dir is None:
        save_file_path = os.path.join(os.getcwd(), "experiment_results", model_name, dataset_name, perturb_type, f"eval_{optimize_method}_result.json")
    else:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir, exist_ok=True)
        save_file_path = os.path.join(args.save_dir, model_name, dataset_name, perturb_type, f"eval_{optimize_method}_result.json")


    # for target_function ablation study only
    # save_file_path = os.path.join(os.getcwd(), "experiment_results", model_name, dataset_name, perturb_type, "ablation", f"eval_{target_function}_result.json")



    metric_list = [abs_relative_difference, squared_relative_difference, rmse_linear, rmse_log, log10, delta1_acc, delta2_acc, delta3_acc, i_rmse, silog_rmse]
    root_path = os.path.join(root_dir, perturb_type)
    record = {}
    for metric_func in metric_list:
        metric_name = metric_func.__name__
        metric_values_perturbed = []
        metric_values_without_perturb = []
        delta = []
        delta_percentage = []
        inference_time = []

        total = 0
        for dir_path in tqdm(os.listdir(root_path)):
            if not os.path.isdir(os.path.join(root_path, dir_path)):
                continue
            
            idx = int(dir_path)
            if idx < start_idx or idx > end_idx:
                continue
            
            print(f"Processing {dir_path}...")
            # open config_{optimize_method}.json
            with open(os.path.join(root_path, dir_path, f"config_{optimize_method}.json"), "r") as f:
                config = json.load(f)
                inference_time.append(config["time_taken_seconds"])

            # depth_pred_original = np.load(os.path.join(root_path, dir_path, "depth_pred_original.npy"))
            gt_depth = np.load(os.path.join(root_path, dir_path, "gt_depth.npy"))
            depth_pred_perturbed = np.load(os.path.join(root_path, dir_path, f"depth_pred_{optimize_method}.npy"))
            valid_mask = np.load(os.path.join(root_path, dir_path, "valid_mask.npy"))

            if valid_mask.shape[1] == 1:
                valid_mask = valid_mask.squeeze(1)

            depth_pred_original = np.load(os.path.join(root_path, dir_path, "depth_pred_original.npy"))

            metric_perturbed = metric_func(
                torch.from_numpy(depth_pred_perturbed).float().cuda(), 
                torch.from_numpy(gt_depth).float().cuda(), 
                torch.from_numpy(valid_mask).bool().cuda()
            ).item()

            metric_without_perturb = metric_func(
                torch.from_numpy(depth_pred_original).float().cuda(),
                torch.from_numpy(gt_depth).float().cuda(),
                torch.from_numpy(valid_mask).bool().cuda()
            ).item()

            metric_values_perturbed.append(metric_perturbed)
            metric_values_without_perturb.append(metric_without_perturb)
            delta.append(metric_perturbed - metric_without_perturb)
            if metric_without_perturb != 0:
                delta_percentage.append((metric_perturbed - metric_without_perturb) / abs(metric_without_perturb))

            total += 1

        print(f"Metric: {metric_name}, Perturbed Mean: {np.mean(metric_values_perturbed)}, Perturbed Std: {np.std(metric_values_perturbed)}, Total: {total}, Without Perturb Mean: {np.mean(metric_values_without_perturb)}, Without Perturb Std: {np.std(metric_values_without_perturb)}")
        print(f"Delta Mean: {np.mean(delta)}, Delta percentage Mean: {np.mean(delta_percentage)}")
        # (mean, std)
        record[metric_name] = {
            "mean": float(np.mean(metric_values_perturbed)),
            "std": float(np.std(metric_values_perturbed)),
            "mean_without_perturb": float(np.mean(metric_values_without_perturb)),
            "std_without_perturb": float(np.std(metric_values_without_perturb)),
            "delta_mean": float(np.mean(delta)),
            "delta_percentage_mean": f"{round(float(np.mean(delta_percentage))*100, 2)}%",
            "Max delta": float(np.max(delta)),
            "Max delta percentage": f"{round(float(np.max(delta_percentage))*100, 2)}%",
            "Min delta": float(np.min(delta)),
            "Min delta percentage": f"{round(float(np.min(delta_percentage))*100, 2)}%",
            "average_inference_time_seconds": float(np.mean(inference_time)),
        }

    # save record to root_path
    # with open(os.path.join(root_path, f"eval_{optimize_method}_result.json"), "w") as f:
    #     json.dump(record, f, indent=4)
        
    # save to directory
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    with open(save_file_path, "w") as f:
        json.dump(record, f, indent=4)

    print("Evaluation completed.")
