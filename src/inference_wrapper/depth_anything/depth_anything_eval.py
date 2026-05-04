from external.Depth_Anything_V2.metric_depth.dataset.hypersim import Hypersim
from external.Depth_Anything_V2.metric_depth.dataset.kitti import KITTI
from external.Depth_Anything_V2.metric_depth.dataset.nyud import NYUD

from src.inference_wrapper.depth_anything.depth_anything_pipeline import DADepthEstimator

from torch.utils.data import DataLoader
import torch
import numpy as np

from external.Marigold.src.util.alignment import (
    align_depth_least_square,
    depth2disparity,
    disparity2depth,
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
        depth_pred = torch.from_numpy(depth_pred).to(da_model.device)  
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

def rmse_linear(pred, target, valid_mask=None):
    diff = pred[valid_mask] - target[valid_mask]
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    return rmse

valset = KITTI('external/Depth_Anything_V2/metric_depth/dataset/splits/kitti/val.txt', 'val', size=(518, 518))
valloader = DataLoader(valset, batch_size=4, pin_memory=True, num_workers=4, drop_last=True)

encoder = "vitb"
pretrained_from = f"/path/to/checkpoints/depth_anything_v2/depth_anything_v2_{encoder}.pth"
max_depth = 80.0
min_depth = 1e-3
alignment = "least_square"
# alignment = None


da_model = DADepthEstimator(
    checkpoint_path=pretrained_from,
    encoder=encoder,
    max_depth=max_depth,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)

metric = 0
total = 0
print("-"*20+"\n"+"Starting evaluation..."+"\n"+"-"*20+"\n")
for i, data in enumerate(valloader):
    inputs, gt_depth = data['image'].to(da_model.device), data['depth'].to(da_model.device)

    with torch.no_grad():
        pred_depth = da_model.batch_infer(inputs)


    if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
        pred_depth = torch.nn.functional.interpolate(
            pred_depth.unsqueeze(1), size=gt_depth.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)

    if len(pred_depth.shape) == 3 and pred_depth.shape[0] > 1:
        pred_depth_aligned = []
        for b in range(pred_depth.shape[0]):
            pred_depth_b_aligned = align_pred_with_gt(
                pred_depth[b:b+1, :], gt_depth[b:b+1, :], data['valid_mask'][b:b+1], alignment, min_depth, max_depth
            )
            pred_depth_aligned.append(pred_depth_b_aligned)
        pred_depth_aligned = torch.cat(pred_depth_aligned, dim=0)
    else:
        pred_depth_aligned = align_pred_with_gt(
            pred_depth, gt_depth, data['valid_mask'], alignment, min_depth, max_depth
        )

    print(f"RMSE linear for batch {i}: {rmse_linear(pred_depth_aligned, gt_depth, data['valid_mask']).item() / inputs.shape[0]}")

    metric += rmse_linear(pred_depth_aligned, gt_depth)
    total += 1

print(f"Average RMSE: {metric / total if total > 0 else 0}")

# run the following command:
# CUDA_VISIBLE_DEVICES=1 python3 -m src.inference_wrapper.depth_anything.depth_anything_eval
