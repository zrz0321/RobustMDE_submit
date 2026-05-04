import os
import json
import torch
import numpy as np
from collections import defaultdict
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

opt_method = "pareto_direct"
save_dir = "./experiment_results"

# the index of samples to evaluate
min_idx = 0
max_idx = 60

intensity_list = ["50_iter"]
dataset_list = ["nyud", "kitti", "hypersim"]
perturbed_type_list = ["geometric", "color_shift", "motion_blur", "banding"]

base_dir = "/path/to/experiment_results/robustness_analysis/"
# model_list = ["depth_anything", "marigold", "monodepth", "zoe_depth", "robustdepth"]
model_list = ["depth_anything", "marigold_modified", "monodepth", "zoe_depth", "robustdepth", "depth_anything_relative", "monovit"]
additional_type_list = None

baseline_model = "monodepth"
baseline_model_type = "mono_640x192"

with open(os.path.join(save_dir, "robustness_evaluation_results.json"), "r") as f:
    result_dict = json.load(f)

sys.stdout = Logger(os.path.join(save_dir, "RR_50iter_results.txt"))


result_in_percentage = True
# if True, the results will be multiplied by 100
output_latex_line_format = True

print("\n"*3+"-"*30+"\n"+"Resilience Rate(RR) of 50iter:"+"\n"+"-"*30+"\n"*2)
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
    print("\n\nRobustness Evaluation Results (RR of 50iter) in LaTeX line format:")
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

# run the following command
# python3 -m src.statistic.RR_50iter_results