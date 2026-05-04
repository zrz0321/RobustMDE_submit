from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from tqdm import tqdm
import torch
import pdb
from torchvision import transforms, datasets

from PIL import Image
import torch
from src.inference_wrapper.monodepth.monodepth_dataset import KITTI, NYUD, HYPERSIM
from src.inference_wrapper.monodepth.monodepth_pipeline import monodepthEstimator
from tqdm import tqdm


# _scale_or_not = False
_scale_or_not = True
min_depth = 0.1
max_depth = 80.0
# checkpoint_path="/path/to/checkpoints/monodepth/mono_640x192"
checkpoint_path="/path/to/checkpoints/monodepth/mono_1024x320"
img_size = (192, 640)

def rmse_linear(pred, target, valid_mask=None):
    diff = pred[valid_mask] - target[valid_mask]
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    return rmse

with torch.no_grad():
    # load dataset
    valset = KITTI('external/Depth_Anything_V2/metric_depth/dataset/splits/kitti/val.txt', 'val', img_size=img_size)
    # valset = NYUD('external/Depth_Anything_V2/metric_depth/dataset/splits/nyud-v2/val.txt', 'val', '/path/to/datasets/nyu_depth_v2', img_size=img_size)
    # valset = HYPERSIM('external/Marigold/data_split/hypersim_depth/filename_list_val_filtered.txt', 'val', '/path/to/datasets/marigold_dataset/hypersim/val', img_size=img_size)

    dataloader = torch.utils.data.DataLoader(
        valset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True
    )

    # Load model and preprocessing transform
    monodepth_model = monodepthEstimator(
        checkpoint_path=checkpoint_path,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        min_depth=min_depth,
        max_depth=max_depth,
        _scale_or_not=_scale_or_not,
    )

    # test
    def get_model_para_nums(model):
        num_params = sum(p.numel() for p in model.parameters())
        return num_params
    total_params = get_model_para_nums(monodepth_model.encoder) + get_model_para_nums(monodepth_model.depth_decoder)
    if total_params >= 1e9:
        num_str = f"{total_params/1e9:.2f}G"
    else:
        num_str = f"{total_params/1e6:.2f}M"
    print(f"Total params:{num_str}")
    exit()

    metric = 0
    total = 0
    print("-"*20+"\n"+"Starting evaluation..."+"\n"+"-"*20+"\n")
    for i, sample in tqdm(enumerate(dataloader)):
        image = sample['image'].to(torch.device("cuda:0"))
        depth = sample['depth'].to(torch.device("cuda:0"))

        # Forward pass
        depth_pred = monodepth_model.batch_infer(image)
        if depth_pred.shape[-2:] != depth.shape[-2:]:
            depth_pred = torch.nn.functional.interpolate(
                depth_pred.unsqueeze(1), size=depth.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)
        rmse_current = rmse_linear(depth_pred, depth, sample['valid_mask'])
        metric += rmse_current.item() 
        total += image.shape[0]

        # Save or visualize the depth prediction
        print(f"RMSE linear for batch {i}: {rmse_current.item() / image.shape[0]}")

        if i >= 20:
            break

    print(f"Average RMSE linear: {metric / total}")

# run the following command
# python3 -m src.inference_wrapper.monodepth.monodepth_eval