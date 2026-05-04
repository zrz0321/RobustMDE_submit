import os
import json
import torch
import numpy as np
from collections import defaultdict
import sys

from src.utils.logger import Logger

if not os.path.exists("./.log/"):
    os.makedirs("./.log/")
sys.stdout = Logger(sys.stdout, "./.log/results_check_log.txt")

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
additional_type_list = None

baseline_model = "monodepth"
baseline_model_type = "mono_640x192"



result_in_percentage = True
# if True, the results will be multiplied by 100
output_latex_line_format = True
# if True, the results will be printed in line format for LaTeX tables
# the result of each model will be printed in a single line, and with & as the separator

finished_model_list = []

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

    for dataset_name in dataset_list:
        for perturbed_type in perturbed_type_list:
            if additional_type_list is not None:
                for additional_type in additional_type_list:
                    for intensity in intensity_list:  
                        chk = True   
                        for idx in range(min_idx, max_idx):
                            result_path = os.path.join(base_dir, model_name, additional_type, intensity, dataset_name, perturbed_type, f"{idx}")
                            if not os.path.exists(result_path):
                                chk = False
                                print(f"Result path {result_path} does not exist.")
                                break
                        if chk == True:
                            print(f"Finished: Model: {model_name}, Type: {additional_type}, Dataset: {dataset_name}, Perturbation: {perturbed_type}, iterations: {intensity}")
                            finished_model_list.append((model_name, additional_type, dataset_name, perturbed_type, intensity,))
                        else:
                            print(f"ERROR:!!!! Model: {model_name}, Type: {additional_type}, Dataset: {dataset_name}, Perturbation: {perturbed_type}, iterations: {intensity}")
                            
            else:
                for intensity in intensity_list:
                    chk = True
                    for idx in range(min_idx, max_idx):
                        result_path = os.path.join(base_dir, model_name, intensity, dataset_name, perturbed_type, f"{idx}")

                        if not os.path.exists(result_path):
                            print(f"Result path {result_path} does not exist.")
                            chk = False
                            break
                    if chk == True:
                        print(f"Finished: Model: {model_name}, Dataset: {dataset_name}, Perturbation: {perturbed_type}, iterations: {intensity}")
                        finished_model_list.append((model_name, None, dataset_name, perturbed_type, intensity,))

                    else:
                        print(f"ERROR:!!!! Model: {model_name}, Dataset: {dataset_name}, Perturbation: {perturbed_type}, iterations: {intensity}")

print("Summary of finished experiments:")
for model_name, additional_type, dataset_name, perturbed_type, intensity in finished_model_list:
    if additional_type is not None:
        print(f"Model: {model_name}, Type: {additional_type}, Dataset: {dataset_name}, Perturbation: {perturbed_type}, iterations: {intensity}")
    else:
        print(f"Model: {model_name}, Dataset: {dataset_name}, Perturbation: {perturbed_type}, iterations: {intensity}")