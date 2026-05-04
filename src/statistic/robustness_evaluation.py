import os
import json
import torch
import numpy as np
from collections import defaultdict
import sys

from src.utils.logger import Logger

def tree():
    return defaultdict(tree)

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

opt_method = "pareto_direct"
save_dir = "./experiment_results"

# the index of samples to evaluate
min_idx = 0
max_idx = 60

intensity_list = ["10_iter", "30_iter", "50_iter"]
dataset_list = ["nyud", "kitti", "hypersim"]
perturbed_type_list = ["geometric", "color_shift", "motion_blur", "banding"]

base_dir = "/path/to/experiment_results/robustness_analysis/"
# model_list = ["depth_anything", "marigold", "monodepth", "zoe_depth", "robustdepth"]
model_list = ["depth_anything", "marigold_modified", "monodepth", "zoe_depth", "robustdepth", "depth_anything_relative", "monovit"]
# model_list = ["depth_anything_relative", "monovit"]
additional_type_list = None

baseline_model = "monodepth"
baseline_model_type = "mono_640x192"

result_dict = tree()

result_in_percentage = True
# if True, the results will be multiplied by 100
output_latex_line_format = True
# if True, the results will be printed in line format for LaTeX tables
# the result of each model will be printed in a single line, and with & as the separator

for model_name in model_list:
    if model_name == "depth_anything":
        additional_type_list = ["vitb", "vitl", "vits"]
    elif model_name == "depth_anything_relative":
        additional_type_list = ["vits", "vitb", "vitl"]
    elif model_name == "monodepth":
        additional_type_list = ["mono_1024x320", "mono_640x192"]
    elif model_name == "robustdepth":
        additional_type_list = ["resnet", "vit"]
    elif model_name == "monovit":
        additional_type_list = ["1024x320", "640x192"]
    else:
        additional_type_list = None

    print("Dealing with model:", model_name)

    for dataset_name in dataset_list:
        print("----| Dealing with dataset:", dataset_name)
        for perturbed_type in perturbed_type_list:
            print("--------| Dealing with perturbed type:", perturbed_type)
            if additional_type_list is not None:
                for additional_type in additional_type_list:
                    print("------------| Dealing with additional type:", additional_type)
                    for intensity in intensity_list:     
                        for idx in range(min_idx, max_idx):
                            result_path = os.path.join(base_dir, model_name, additional_type, intensity, dataset_name, perturbed_type, f"{idx}")

                            if not os.path.exists(result_path):
                                raise FileNotFoundError(f"Result path {result_path} does not exist.")
                            
                            try:
                                gt_depth = np.load(os.path.join(result_path, "gt_depth.npy"))
                            except:
                                print(f"Failed to load gt_depth from {result_path}")
                                exit()
                            
                            try:
                                valid_mask = np.load(os.path.join(result_path, "valid_mask.npy")).astype(bool)
                            except:
                                print(f"Failed to load valid_mask from {result_path}")
                                exit()

                            try:
                                pred_depth = np.load(os.path.join(result_path, f"depth_pred_{opt_method}.npy"))
                            except:
                                print(f"Failed to load pred_depth from {result_path}")
                                exit()

                            gt_depth, pred_depth, valid_mask = torch.from_numpy(gt_depth), torch.from_numpy(pred_depth), torch.from_numpy(valid_mask)
                            gt_depth, pred_depth, valid_mask = gt_depth.squeeze(), pred_depth.squeeze(), valid_mask.squeeze()
                            dee_value = DEE(pred_depth, gt_depth, valid_mask)
                            result_dict[model_name][additional_type][dataset_name][perturbed_type][intensity][f"{idx}"]["corrupted"] = dee_value

                            try:
                                depth_pred_original = np.load(os.path.join(result_path, "depth_pred_original.npy"))
                            except:
                                print(f"Failed to load depth_pred_original from {result_path}")
                                exit()

                            depth_pred_original = torch.from_numpy(depth_pred_original).squeeze()
                            dee_value_original = DEE(depth_pred_original, gt_depth, valid_mask)
                            result_dict[model_name][additional_type][dataset_name][perturbed_type][intensity][f"{idx}"]["original"] = dee_value_original

                            if model_name == baseline_model and additional_type == baseline_model_type:
                                result_dict["baseline"][dataset_name][perturbed_type][intensity][f"{idx}"] = dee_value
            else:
                for intensity in intensity_list:     
                    for idx in range(min_idx, max_idx):
                        result_path = os.path.join(base_dir, model_name, intensity, dataset_name, perturbed_type, f"{idx}")

                        if not os.path.exists(result_path):
                            raise FileNotFoundError(f"Result path {result_path} does not exist.")
                        
                        try:
                            gt_depth = np.load(os.path.join(result_path, "gt_depth.npy"))
                        except:
                            print(f"Failed to load gt_depth from {result_path}")
                            exit()

                        try:
                            valid_mask = np.load(os.path.join(result_path, "valid_mask.npy")).astype(bool)
                        except:
                            print(f"Failed to load valid_mask from {result_path}")
                            exit()
                        
                        try:
                            pred_depth = np.load(os.path.join(result_path, f"depth_pred_{opt_method}.npy"))
                        except:
                            print(f"Failed to load pred_depth from {result_path}")
                            exit()

                        gt_depth, pred_depth, valid_mask = torch.from_numpy(gt_depth), torch.from_numpy(pred_depth), torch.from_numpy(valid_mask)
                        gt_depth, pred_depth, valid_mask = gt_depth.squeeze(), pred_depth.squeeze(), valid_mask.squeeze()
                        dee_value = DEE(pred_depth, gt_depth, valid_mask)
                        result_dict[model_name][dataset_name][perturbed_type][intensity][f"{idx}"]["corrupted"] = dee_value

                        try:
                            depth_pred_original = np.load(os.path.join(result_path, "depth_pred_original.npy"))
                        except:
                            print(f"Failed to load depth_pred_original from {result_path}")
                            exit()
                        depth_pred_original = torch.from_numpy(depth_pred_original).squeeze()
                        dee_value_original = DEE(depth_pred_original, gt_depth, valid_mask)
                        result_dict[model_name][dataset_name][perturbed_type][intensity][f"{idx}"]["original"] = dee_value_original

                        if model_name == baseline_model and additional_type == baseline_model_type:
                            result_dict["baseline"][dataset_name][perturbed_type][intensity][f"{idx}"] = dee_value

print("-"*30+"\n"+"mean Corruption Error(mCE):"+"\n"+"-"*30+"\n"*2)
sys.stdout = Logger(os.path.join(save_dir, "robustness_evaluation_results.txt"))

# Print and save all results
for model_name in model_list:
    if model_name == "depth_anything":
        additional_type_list = ["vitb", "vitl", "vits"]
    elif model_name == "depth_anything_relative":
        additional_type_list = ["vits", "vitb", "vitl"]
    elif model_name == "monodepth":
        additional_type_list = ["mono_1024x320", "mono_640x192"]
    elif model_name == "robustdepth":
        additional_type_list = ["resnet", "vit"]
    elif model_name == "monovit":
        additional_type_list = ["1024x320", "640x192"]
    else:
        additional_type_list = None
    print(f"{model_name}:")

    if additional_type_list is not None:
        for additional_type in additional_type_list:
            print(f"----| {additional_type}:")
            for dataset_name in dataset_list:
                print(f"--------| {dataset_name}:")
                for perturbed_type in perturbed_type_list:
                    print(f"------------| {perturbed_type}:")
                    corruption_error = 0
                    for idx in range(min_idx, max_idx):
                        metric_current = 0
                        metric_basline = 0
                        for intensity in intensity_list:
                            metric_current += result_dict[model_name][additional_type][dataset_name][perturbed_type][intensity][f"{idx}"]["corrupted"]
                            metric_basline += result_dict["baseline"][dataset_name][perturbed_type][intensity][f"{idx}"]
                        corruption_error += metric_current / metric_basline
                    corruption_error /= (max_idx - min_idx)

                    if result_in_percentage:
                        corruption_error *= 100
                        print(f"------------| mCE: {corruption_error:.2f}")
                    else:
                        print(f"------------| mCE: {corruption_error:.4f}")
    else:
        for dataset_name in dataset_list:
            print(f"----| {dataset_name}:")
            for perturbed_type in perturbed_type_list:
                print(f"--------| {perturbed_type}:")
                corruption_error = 0
                for idx in range(min_idx, max_idx):
                    metric_current = 0
                    metric_basline = 0
                    for intensity in intensity_list:
                        metric_current += result_dict[model_name][dataset_name][perturbed_type][intensity][f"{idx}"]["corrupted"]
                        metric_basline += result_dict["baseline"][dataset_name][perturbed_type][intensity][f"{idx}"]
                    corruption_error += metric_current / metric_basline
                corruption_error /= (max_idx - min_idx)

                if result_in_percentage:
                    corruption_error *= 100
                    print(f"--------| mCE: {corruption_error:.2f}")
                else:
                    print(f"--------| mCE: {corruption_error:.4f}")


with open(os.path.join(save_dir, "robustness_evaluation_results.json"), "w") as f:
    json.dump(result_dict, f, indent=4)

if output_latex_line_format:
    print("\n\nRobustness Evaluation Results (DEE) in LaTeX line format:")
    for model_name in model_list:
        if model_name == "depth_anything":
            additional_type_list = ["vitb", "vitl", "vits"]
        elif model_name == "depth_anything_relative":
            additional_type_list = ["vits", "vitb", "vitl"]
        elif model_name == "monodepth":
            additional_type_list = ["mono_1024x320", "mono_640x192"]
        elif model_name == "robustdepth":
            additional_type_list = ["resnet", "vit"]
        elif model_name == "monovit":
            additional_type_list = ["1024x320", "640x192"]
        else:
            additional_type_list = None

        if additional_type_list is not None:
            for additional_type in additional_type_list:
                print(f"{model_name}_{additional_type} & ", end="")
                for dataset_name in dataset_list:
                    for perturbed_type in perturbed_type_list:
                        corruption_error = 0
                        for idx in range(min_idx, max_idx):
                            metric_current = 0
                            metric_basline = 0
                            for intensity in intensity_list:
                                metric_current += result_dict[model_name][additional_type][dataset_name][perturbed_type][intensity][f"{idx}"]["corrupted"]
                                metric_basline += result_dict["baseline"][dataset_name][perturbed_type][intensity][f"{idx}"]
                            corruption_error += metric_current / metric_basline
                        corruption_error /= (max_idx - min_idx)

                        if result_in_percentage:
                            corruption_error *= 100
                            print(f"{corruption_error:.2f} & ", end="")
                        else:
                            print(f"{corruption_error:.4f} & ", end="")
                print("\\\\")
        else:
            print(f"{model_name} & ", end="")
            for dataset_name in dataset_list:
                for perturbed_type in perturbed_type_list:
                    corruption_error = 0
                    for idx in range(min_idx, max_idx):
                        metric_current = 0
                        metric_basline = 0
                        for intensity in intensity_list:
                            metric_current += result_dict[model_name][dataset_name][perturbed_type][intensity][f"{idx}"]["corrupted"]
                            metric_basline += result_dict["baseline"][dataset_name][perturbed_type][intensity][f"{idx}"]
                        corruption_error += metric_current / metric_basline
                    corruption_error /= (max_idx - min_idx)

                    if result_in_percentage:
                        corruption_error *= 100
                        print(f"{corruption_error:.2f} & ", end="")
                    else:
                        print(f"{corruption_error:.4f} & ", end="")
            print("\\\\")

print("\n"*3+"-"*30+"\n"+"Resilience Rate(RR):"+"\n"+"-"*30+"\n"*2)
# RR = sum(1-DEE)/(L*(1-DEE_clean))
for model_name in model_list:
    if model_name == "depth_anything":
        additional_type_list = ["vitb", "vitl", "vits"]
    elif model_name == "depth_anything_relative":
        additional_type_list = ["vits", "vitb", "vitl"]
    elif model_name == "monodepth":
        additional_type_list = ["mono_1024x320", "mono_640x192"]
    elif model_name == "robustdepth":
        additional_type_list = ["resnet", "vit"]
    elif model_name == "monovit":
        additional_type_list = ["1024x320", "640x192"]
    else:
        additional_type_list = None

    print(f"{model_name}:")

    if additional_type_list is not None:
        for additional_type in additional_type_list:
            print(f"----| {additional_type}:")
            for dataset_name in dataset_list:
                print(f"--------| {dataset_name}:")
                for perturbed_type in perturbed_type_list:
                    print(f"------------| {perturbed_type}:", end="")
                    rr = 0
                    for idx in range(min_idx, max_idx):
                        metric_current = 0
                        for intensity in intensity_list:
                            metric_current += (1 - result_dict[model_name][additional_type][dataset_name][perturbed_type][intensity][f"{idx}"]["corrupted"])
                        rr += metric_current / (len(intensity_list) * (1 - result_dict[model_name][additional_type][dataset_name][perturbed_type][intensity][f"{idx}"]["original"]))
                    rr /= (max_idx - min_idx)

                    if result_in_percentage:
                        rr *= 100
                        print(f"{rr:.2f}")
                    else:
                        print(f"{rr:.4f}")
            print()
    else:
        for dataset_name in dataset_list:
            print(f"----| {dataset_name}:")
            for perturbed_type in perturbed_type_list:
                print(f"--------| {perturbed_type}:", end="")
                rr = 0
                for idx in range(min_idx, max_idx):
                    metric_current = 0
                    for intensity in intensity_list:
                        metric_current += (1 - result_dict[model_name][dataset_name][perturbed_type][intensity][f"{idx}"]["corrupted"])
                    rr += metric_current / (len(intensity_list) * (1 - result_dict[model_name][dataset_name][perturbed_type][intensity][f"{idx}"]["original"]))
                rr /= (max_idx - min_idx)
                if result_in_percentage:
                    rr *= 100
                    print(f"{rr:.2f}")
                else:
                    print(f"{rr:.4f}")
        print()

if output_latex_line_format:
    print("\n\nRobustness Evaluation Results (RR) in LaTeX line format:")
    for model_name in model_list:
        if model_name == "depth_anything":
            additional_type_list = ["vitb", "vitl", "vits"]
        elif model_name == "depth_anything_relative":
            additional_type_list = ["vits", "vitb", "vitl"]
        elif model_name == "monodepth":
            additional_type_list = ["mono_1024x320", "mono_640x192"]
        elif model_name == "robustdepth":
            additional_type_list = ["resnet", "vit"]
        elif model_name == "monovit":
            additional_type_list = ["1024x320", "640x192"]
        else:
            additional_type_list = None

        if additional_type_list is not None:
            for additional_type in additional_type_list:
                print(f"{model_name}_{additional_type} & ", end="")
                for dataset_name in dataset_list:
                    for perturbed_type in perturbed_type_list:
                        rr = 0
                        for idx in range(min_idx, max_idx):
                            metric_current = 0
                            for intensity in intensity_list:
                                metric_current += (1 - result_dict[model_name][additional_type][dataset_name][perturbed_type][intensity][f"{idx}"]["corrupted"])
                            rr += metric_current / (len(intensity_list) * (1 - result_dict[model_name][additional_type][dataset_name][perturbed_type][intensity][f"{idx}"]["original"]))
                        rr /= (max_idx - min_idx)

                        if result_in_percentage:
                            rr *= 100
                            print(f"{rr:.2f} & ", end="")
                        else:
                            print(f"{rr:.4f} & ", end="")
                print("\\\\")
        else:
            print(f"{model_name} & ", end="")
            for dataset_name in dataset_list:
                for perturbed_type in perturbed_type_list:
                    rr = 0
                    for idx in range(min_idx, max_idx):
                        metric_current = 0
                        for intensity in intensity_list:
                            metric_current += (1 - result_dict[model_name][dataset_name][perturbed_type][intensity][f"{idx}"]["corrupted"])
                        rr += metric_current / (len(intensity_list) * (1 - result_dict[model_name][dataset_name][perturbed_type][intensity][f"{idx}"]["original"]))
                    rr /= (max_idx - min_idx)

                    if result_in_percentage:
                        rr *= 100
                        print(f"{rr:.2f} & ", end="")
                    else:
                        print(f"{rr:.4f} & ", end="")
            print("\\\\")

print("\n"*3+"-"*30+"\n"+"mean DEE:"+"\n"+"-"*30+"\n"*2)
# mean DEE for each model
for model_name in model_list:
    if model_name == "depth_anything":
        additional_type_list = ["vitb", "vitl", "vits"]
    elif model_name == "depth_anything_relative":
        additional_type_list = ["vits", "vitb", "vitl"]
    elif model_name == "monodepth":
        additional_type_list = ["mono_1024x320", "mono_640x192"]
    elif model_name == "robustdepth":
        additional_type_list = ["resnet", "vit"]
    elif model_name == "monovit":
        additional_type_list = ["1024x320", "640x192"]
    else:
        additional_type_list = None

    print(f"{model_name}:")

    if additional_type_list is not None:
        for additional_type in additional_type_list:
            print(f"----| {additional_type}:")
            for dataset_name in dataset_list:
                print(f"--------| {dataset_name}:")
                for perturbed_type in perturbed_type_list:
                    print(f"------------| {perturbed_type}:", end="")
                    mDEE = 0
                    for idx in range(min_idx, max_idx):
                        metric_current = 0
                        for intensity in intensity_list:
                            metric_current += result_dict[model_name][additional_type][dataset_name][perturbed_type][intensity][f"{idx}"]["corrupted"]
                        mDEE += metric_current / len(intensity_list)

                    mDEE /= (max_idx - min_idx)

                    if result_in_percentage:
                        mDEE *= 100
                        print(f"{mDEE:.2f}")
                    else:
                        print(f"{mDEE:.4f}")
            print()
    else:
        for dataset_name in dataset_list:
            print(f"----| {dataset_name}:")
            for perturbed_type in perturbed_type_list:
                print(f"--------| {perturbed_type}:", end="")
                mDEE = 0
                for idx in range(min_idx, max_idx):
                    metric_current = 0
                    for intensity in intensity_list:
                        metric_current += result_dict[model_name][dataset_name][perturbed_type][intensity][f"{idx}"]["corrupted"]
                    mDEE += metric_current / len(intensity_list)
                mDEE /= (max_idx - min_idx)
                if result_in_percentage:
                    mDEE *= 100
                    print(f"{mDEE:.2f}")
                else:
                    print(f"{mDEE:.4f}")
        print()

if output_latex_line_format:
    print("\n\nmDEE in LaTeX line format:")
    for model_name in model_list:
        if model_name == "depth_anything":
            additional_type_list = ["vitb", "vitl", "vits"]
        elif model_name == "depth_anything_relative":
            additional_type_list = ["vits", "vitb", "vitl"]
        elif model_name == "monodepth":
            additional_type_list = ["mono_1024x320", "mono_640x192"]
        elif model_name == "robustdepth":
            additional_type_list = ["resnet", "vit"]
        elif model_name == "monovit":
            additional_type_list = ["1024x320", "640x192"]
        else:
            additional_type_list = None

        if additional_type_list is not None:
            for additional_type in additional_type_list:
                print(f"{model_name}_{additional_type} & ", end="")
                for dataset_name in dataset_list:
                    for perturbed_type in perturbed_type_list:
                        mDEE = 0
                        for idx in range(min_idx, max_idx):
                            metric_current = 0
                            for intensity in intensity_list:
                                metric_current += result_dict[model_name][additional_type][dataset_name][perturbed_type][intensity][f"{idx}"]["corrupted"]
                            mDEE += metric_current / len(intensity_list)
                        mDEE /= (max_idx - min_idx)

                        if result_in_percentage:
                            mDEE *= 100
                            print(f"{mDEE:.2f} & ", end="")
                        else:
                            print(f"{mDEE:.4f} & ", end="")
                print("\\\\")
        else:
            print(f"{model_name} & ", end="")
            for dataset_name in dataset_list:
                for perturbed_type in perturbed_type_list:
                    mDEE = 0
                    for idx in range(min_idx, max_idx):
                        metric_current = 0
                        for intensity in intensity_list:
                            metric_current += result_dict[model_name][dataset_name][perturbed_type][intensity][f"{idx}"]["corrupted"]
                        mDEE += metric_current / len(intensity_list)
                    mDEE /= (max_idx - min_idx)

                    if result_in_percentage:
                        mDEE *= 100
                        print(f"{mDEE:.2f} & ", end="")
                    else:
                        print(f"{mDEE:.4f} & ", end="")
            print("\\\\")

# run the following command:
# python3 -m src.statistic.robustness_evaluation