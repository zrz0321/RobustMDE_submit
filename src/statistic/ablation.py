import os
import json
import torch
import numpy as np
import sys

from src.utils.logger import Logger
def rmse_linear(pred, target, valid_mask=None):
    diff = (pred - target) * valid_mask
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    return rmse
def abs_relative_difference(pred, target, valid_mask=None):
    diff = (pred - target) * valid_mask
    abs_rel = torch.mean(torch.abs(diff) / (target + 1e-8))
    return abs_rel
def delta1_acc(pred, target, valid_mask=None):
    pred, target = pred[valid_mask], target[valid_mask]
    thresh = torch.max((target / pred), (pred / target))
    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    return d1

def DEE(pred, target, valid_mask=None):
    absrel = abs_relative_difference(pred, target, valid_mask)
    delta1 = delta1_acc(pred, target, valid_mask)
    dee = ((absrel - delta1 + 1) / 2).item()
    return dee

def get_ablation_type(ablation_name):
    if ablation_name == "optimize_method":
        ablation_type = ["pareto_direct", "direct", "dual_annealing", "original_direct"]
    elif ablation_name == "max_iter":
        ablation_type = ["10", "30", "50", "70", "90"]
    elif ablation_name == "max_batch_num":
        ablation_type = ["1", "4", "8", "10", "12", "16"]
    elif ablation_name == "gamma":
        ablation_type = ["0.005", "0.0075", "0.01", "0.0125", "0.015", "0.08", "0.09", "0.1", "0.11", "0.12", "0.15", "0.2", "0.25", "0.3"]
    elif ablation_name == "target_function":
        ablation_type = ["abs_rel", "mse_loss", "delta1", "rmse_l"]
    elif ablation_name == "ks_and_rbd":
        ablation_type = [["3", "5", "7", "9", "11"], ["5", "6", "7"]]
    else:
        raise NotImplementedError
    
    return ablation_type

# ablation_name_list = ["optimize_method", "max_iter", "max_batch_num", "gamma", "ks_and_rbd", "target_function"]
# ablation_name_list = ["max_batch_num", "gamma", "ks_and_rbd", "target_function"]
# ablation_name_list = ["gamma", "target_function", "ks_and_rbd"]
ablation_name_list = ["max_batch_num"]


root_dir = "/path/to/experiment_results/ablation_study/"
save_dir = "./experiment_results/ablation_study/"
os.makedirs(save_dir, exist_ok=True)

min_idx = 0
# max_idx = 30
max_idx = 5
# perturb_type_list = ["geometric", "color_shift", "motion_blur", "banding"]
perturb_type_list = ["color_shift"]
dataset = "nyud"
result_in_percentage = True

for ablation_name in ablation_name_list:
    ablation_type = get_ablation_type(ablation_name)
    sys.stdout = Logger(os.path.join(save_dir, f"{ablation_name}_results.txt"))
    if ablation_name != "ks_and_rbd":
        print(f"-> Ablation Study on {ablation_name}:")
        for current_type in ablation_type:
            print(f"----->  {current_type}: ")
            tmp_path = os.path.join(root_dir, f"{ablation_name}", f"{ablation_name}_{current_type}")
            for perturb_type in perturb_type_list:
                if ablation_name == "gamma":
                    if not os.path.exists(os.path.join(tmp_path, f"{perturb_type}")):
                        continue
                print(f"---------> Perturbation Type: {perturb_type}")
                _path = os.path.join(tmp_path, f"{perturb_type}", f"{dataset}", f"{perturb_type}")
                RR_list = []
                time_list = []
                Delta_list = []
                for idx in range(min_idx, max_idx):
                    dir_path = os.path.join(_path, f"{idx}")
                    try:
                        original_result = np.load(os.path.join(dir_path, "depth_pred_original.npy"))
                    except:
                        print(f"File not found: {os.path.join(dir_path, 'depth_pred_original.npy')}")
                        continue
                    gt_result = np.load(os.path.join(dir_path, "gt_depth.npy"))
                    valid_mask = np.load(os.path.join(dir_path, "valid_mask.npy"))
                    if ablation_name == "optimize_method":
                        optimized_result = np.load(os.path.join(dir_path, f"depth_pred_{current_type}.npy"))
                    else:
                        optimized_result = np.load(os.path.join(dir_path, f"depth_pred_pareto_direct.npy"))
                    # mRR
                    #
                    RR_list.append((1 - DEE(torch.tensor(optimized_result), torch.tensor(gt_result), valid_mask=torch.tensor(valid_mask))) / (1 - DEE(torch.tensor(original_result), torch.tensor(gt_result), valid_mask=torch.tensor(valid_mask)) + 1e-8))
                    # time
                    #
                    if ablation_name == "optimize_method":
                        with open(os.path.join(dir_path, f"config_{current_type}.json"), "r") as f:
                            log_dict = json.load(f)
                    else:
                        with open(os.path.join(dir_path, f"config_pareto_direct.json"), "r") as f:
                            log_dict = json.load(f)
                    # print time for each perturbation type
                    time_list.append(log_dict["time_taken_seconds"])
                    # Delta
                    #
                    Delta_list.append((log_dict["perturbed_metric"] - log_dict["original_metric"]) / (log_dict["original_metric"] + 1e-8))
                if result_in_percentage:
                    RR_list = [rr * 100 for rr in RR_list]
                    Delta_list = [delta * 100 for delta in Delta_list]
                # calculate mean and std
                mRR = np.mean(RR_list)
                stdRR = np.std(RR_list)
                mtime = np.mean(time_list)
                stdtime = np.std(time_list)
                mDelta = np.mean(Delta_list)
                stdDelta = np.std(Delta_list)
                print(f"mRR: {mRR:.2f}", end=", ")
                print(f"stdRR: {stdRR:.2f}", end=", ")
                print(f"mtime: {mtime:.2f}", end=", ")
                print(f"stdtime: {stdtime:.2f}")
                print(f"mDelta: {mDelta:.2f}", end=", ")
                print(f"stdDelta: {stdDelta:.2f}")
        print("\n\n")
    else:
        for i in range(2):
            _ablation_type = ablation_type[i]
            if len(_ablation_type) == 5:
                # kernel_size
                ablation_name = "kernel_size"
                perturb_type = "motion_blur"
            elif len(_ablation_type) == 3:
                # rbd
                ablation_name = "reduced_bit_depth"
                perturb_type = "banding"
            else:
                raise NotImplementedError
            original_ablation_name = "ks_and_rbd"
            print(f"-> Ablation Study on {ablation_name}:")
            for current_type in _ablation_type:
                print(f"----->  {current_type}: ")
                tmp_path = os.path.join(root_dir, f"{original_ablation_name}", f"{ablation_name}_{current_type}")
                print(f"---------> Perturbation Type: {perturb_type}")
                _path = os.path.join(tmp_path, f"{dataset}", f"{perturb_type}")
                RR_list = []
                time_list = []
                Delta_list = []
                for idx in range(min_idx, max_idx):
                    dir_path = os.path.join(_path, f"{idx}")
                    original_result = np.load(os.path.join(dir_path, "depth_pred_original.npy"))
                    gt_result = np.load(os.path.join(dir_path, "gt_depth.npy"))
                    valid_mask = np.load(os.path.join(dir_path, "valid_mask.npy"))
                    if ablation_name == "optimize_method":
                        optimized_result = np.load(os.path.join(dir_path, f"depth_pred_{current_type}.npy"))
                    else:
                        optimized_result = np.load(os.path.join(dir_path, f"depth_pred_pareto_direct.npy"))
                    # mRR
                    #
                    RR_list.append((1 - DEE(torch.tensor(optimized_result), torch.tensor(gt_result), valid_mask=torch.tensor(valid_mask))) / (1 - DEE(torch.tensor(original_result), torch.tensor(gt_result), valid_mask=torch.tensor(valid_mask)) + 1e-8))
                    # time
                    #
                    if ablation_name == "optimize_method":
                        with open(os.path.join(dir_path, f"config_{current_type}.json"), "r") as f:
                            log_dict = json.load(f)
                    else:
                        with open(os.path.join(dir_path, f"config_pareto_direct.json"), "r") as f:
                            log_dict = json.load(f)
                    # print time for each perturbation type
                    time_list.append(log_dict["time_taken_seconds"])
                    # Delta
                    #
                    Delta_list.append((log_dict["perturbed_metric"] - log_dict["original_metric"]) / (log_dict["original_metric"] + 1e-8))
                if result_in_percentage:
                    RR_list = [rr * 100 for rr in RR_list]
                    Delta_list = [delta * 100 for delta in Delta_list]
                # calculate mean and std
                mRR = np.mean(RR_list)
                stdRR = np.std(RR_list)
                mtime = np.mean(time_list)
                stdtime = np.std(time_list)
                mDelta = np.mean(Delta_list)
                stdDelta = np.std(Delta_list)
                print(f"mRR: {mRR:.2f}", end=", ")
                print(f"stdRR: {stdRR:.2f}", end=", ")
                print(f"mtime: {mtime:.2f}", end=", ")
                print(f"stdtime: {stdtime:.2f}")
                print(f"mDelta: {mDelta:.2f}", end=", ")
                print(f"stdDelta: {stdDelta:.2f}")
        print("\n\n")

# run the following command:
# python3 -m src.statistic.ablation