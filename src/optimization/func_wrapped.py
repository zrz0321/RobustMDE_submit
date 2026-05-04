import torch
import numpy as np
import torch.nn.functional as F

from src.inference_wrapper.base import DepthEstimator
from src.perturbation import Perturbation, geometric_perturb_for_depth

from external.Marigold.src.util.alignment import (
    align_depth_least_square,
    depth2disparity,
    disparity2depth,
)

def func_wrapped(theta, perturb_type: str, model: DepthEstimator, loss, img: torch.Tensor, gt_depth: torch.Tensor, valid_mask: torch.Tensor = None,
                 alignment: str = "least_square", min_depth: float = 0, max_depth: float = 100, depth_raw_linear: torch.Tensor = None, seed: int = 42, kernel_size: int = 0, reduced_bit_depth: int = 0, transform_func = None,
                 **kwargs):
    """
    A wrapper function to compute the loss given model parameters.

    theta: list, parameters of perturbation to be optimized in order
    perturb_type: str, type of perturbation to be applied
    model: DepthEstimator, depth estimation model
    loss: callable, loss/metric function to compute the error between prediction and ground truth
    img: torch.Tensor, shape (1, C, H, W), input RGB image, not normalized nor perturbed
    gt_depth: torch.Tensor, shape (1, 1, H, W), ground truth depth map
    valid_mask: torch.Tensor, shape (1, H, W), valid mask for depth map
    alignment: str, alignment method to align prediction with ground truth
    min_depth: float, minimum depth value of the dataset
    max_depth: float, maximum depth value of the dataset
    **kwargs: additional keyword arguments for the model

    RETURN
    current_loss: float, the computed loss value * -1 (for maximization)
    """
    if kernel_size == 0 and perturb_type == "motion_blur":
        raise ValueError("kernel_size must be specified for motion blur perturbation.")
    if reduced_bit_depth < 1 and perturb_type == "banding":
        raise ValueError("reduce_bit_depth must be at least 1 for color quantization perturbation.")
    
    theta = tuple(theta)
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

    perturb = Perturbation(perturb_type=perturb_type, transform_func=transform_func)

    img_perturbed = perturb.apply(img, **theta)
    if perturb_type == "geometric":
        gt_depth = gt_depth * valid_mask
        # gt_depth = perturb.apply(gt_depth, **theta)
        gt_depth = geometric_perturb_for_depth(gt_depth, **theta)

        mask_tmp = valid_mask.clone().float()
        mask_tmp = perturb.apply(mask_tmp, **theta)
        gt_depth = gt_depth / (mask_tmp + 1e-8)
        valid_mask = (mask_tmp >= 0.5)

    pred_depth = model.infer(img_perturbed, **kwargs)
    if pred_depth.shape != gt_depth.shape:
        pred_depth = F.interpolate(pred_depth[:, None], size=gt_depth.shape[-2:], mode='bilinear', align_corners=True).squeeze(1)

    gt_depth = gt_depth.squeeze()

    # Align with GT using least square
    if "least_square" == alignment:
        depth_pred, scale, shift = align_depth_least_square(
            gt_arr=gt_depth.cpu().numpy(),
            pred_arr=depth_pred.cpu().numpy(),
            valid_mask_arr=valid_mask.cpu().numpy(),
            return_scale_shift=True,
            max_resolution=None,
        )
        depth_pred = torch.from_numpy(depth_pred).to(model.device)  
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

    depth_pred = torch.clamp(depth_pred, min=min_depth, max=max_depth)
    depth_pred = torch.clamp(depth_pred, min=1e-6)

    # if valid_mask and loss is in torch.nn
    if isinstance(loss, torch.nn.modules.loss._Loss):
        if valid_mask is not None:
            valid_mask = valid_mask.squeeze()
            current_loss = loss(
                pred_depth[valid_mask], gt_depth[valid_mask]
            )
        else:
            current_loss = loss(
                pred_depth, gt_depth
            )
        return -1 * torch.mean(current_loss).to("cpu").item()
    else:
        current_loss = loss(
            pred_depth, gt_depth, valid_mask
        )
    return -1 * current_loss

def func_wrapped_parallel(theta_list, perturb_type: str, model: DepthEstimator, loss, img: torch.Tensor, gt_depth: torch.Tensor, valid_mask: torch.Tensor = None, parallel_num: int = 0,
                 alignment: str = "least_square", min_depth: float = 0, max_depth: float = 100, depth_raw_linear: torch.Tensor = None, max_batch_num = 8, seed: int = 42, kernel_size: int = 0, reduced_bit_depth: int = 0, transform_func = None,
                 **kwargs):
    """
    A wrapper function enable batch processing to compute the loss of a certain image under a batch of perturbation parameters.

    # theta_list: list, parameters of perturbation to be optimized in order
    theta: list[list[float]], parameters of perturbation to be optimized in order. Each element is a list of parameters for one instance.
    perturb_type: str, type of perturbation to be applied
    model: DepthEstimator, depth estimation model
    loss: callable, loss/metric function to compute the error between prediction and ground truth
    img: torch.Tensor, shape (1, C, H, W), input RGB image, normalized to [-1, 1] but not perturbed
    gt_depth: torch.Tensor, shape (1, 1, H, W), ground truth depth map
    valid_mask: torch.Tensor, shape (1, H, W), valid mask for depth map
    parallel_num: int, number of parallel instances to be processed in one batch. If 0, process all instances in one batch.
    alignment: str, alignment method to align prediction with ground truth
    min_depth: float, minimum depth value of the dataset
    max_depth: float, maximum depth value of the dataset
    max_batch_num: int, maximum number of batches to process at once to avoid OOM
    kernel_size: int, kernel size only required for motion blur perturbation, default 0
    reduced_bit_depth: int, reduced bit depth only required for color quantization perturbation, default 0
    seed: int, random seed for the model inference
    transform_func: callable, a function to transform the input image before applying perturbation, default None
    **kwargs: additional keyword arguments for the model

    RETURN
    loss_list: list[float], the computed loss value * -1 (for maximization). Each element is the loss value for one instance.
    """

    assert len(theta_list) == parallel_num, "The batch size should be the same as the number of instances."
    if kernel_size == 0 and perturb_type == "motion_blur":
        raise ValueError("kernel_size must be specified for motion blur perturbation.")
    if reduced_bit_depth < 1 and perturb_type == "banding":
        raise ValueError("reduce_bit_depth must be at least 1 for color quantization perturbation.")
    
    perturb = Perturbation(perturb_type=perturb_type, transform_func=transform_func)

    # get a list of dict for perturbation parameters for each instance
    input_theta_list = []
    for theta in theta_list:
        assert len(theta) == len(theta_list[0]), "All instances should have the same number of parameters."
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
        input_theta_list.append(theta)

    loss_list = []
    # process in smaller batches to avoid OOM
    for batch_start in range(0, parallel_num, max_batch_num):
        batch_end = min(batch_start + max_batch_num, parallel_num)

        # create a batch of perturbed images
        img_batch = []
        for i in range(batch_start, batch_end):
            theta = input_theta_list[i]
            img_perturbed = perturb.apply(img, **theta)
            img_batch.append(img_perturbed)
        
        # # for reproducibility
        # # make sure each batch has the same number of instances
        # if batch_end - batch_start < max_batch_num:
        #     for _ in range(max_batch_num - (batch_end - batch_start)):
        #         img_batch.append(img)  # append the original image to fill the batch

        img_batch = torch.cat(img_batch, dim=0)  # shape (B, C, H, W)

        if perturb_type == "geometric":
            valid_mask_batch = []
            gt_depth_batch = []
            for i in range(batch_start, batch_end):
                theta = input_theta_list[i]
                # valid_mask_perturbed = perturb.apply(valid_mask.float(), **theta, geometric_mode='nearest').bool()
                # gt_depth_perturbed = perturb.apply(gt_depth, **theta)
                gt_depth_perturbed = gt_depth * valid_mask
                # gt_depth_perturbed = perturb.apply(gt_depth_perturbed, **theta)
                gt_depth_perturbed = geometric_perturb_for_depth(gt_depth_perturbed, **theta)

                mask_tmp = valid_mask.clone().float()
                mask_tmp = perturb.apply(mask_tmp, **theta)
                gt_depth_perturbed = gt_depth_perturbed / (mask_tmp + 1e-8)
                valid_mask_perturbed = (mask_tmp >= 0.5)
                
                valid_mask_batch.append(valid_mask_perturbed)
                gt_depth_batch.append(gt_depth_perturbed)

            valid_mask_batch = torch.cat(valid_mask_batch, dim=0)  # shape (B, H, W)
            gt_depth_batch = torch.cat(gt_depth_batch, dim=0)  # shape (B, 1, H, W)
        else:
            if batch_end - batch_start > 1:
                valid_mask_batch = valid_mask.repeat(batch_end - batch_start, 1, 1, 1)
                gt_depth_batch = gt_depth.repeat(batch_end - batch_start, 1, 1, 1)
            else:
                valid_mask_batch = valid_mask
                gt_depth_batch = gt_depth

        # set seed of the generator for reproducibility
        if kwargs.get("generator") is not None:
            kwargs["generator"] = torch.Generator(model.device).manual_seed(seed)
        
        depth_pred_batch = model.batch_infer(img_batch, **kwargs)  # shape (B, 1, H, W)
        depth_pred_batch = depth_pred_batch.squeeze(1)  # shape (B, H, W)
        gt_depth_batch = gt_depth_batch.squeeze(1)  # shape (B, H, W)
        valid_mask_batch = valid_mask_batch.squeeze(1)  # shape (B, H, W)

        for i in range(batch_end - batch_start):
            _depth_pred = depth_pred_batch[i]
            _valid_mask = valid_mask_batch[i]
            _gt_depth = gt_depth_batch[i]
            if _depth_pred.shape != _gt_depth.shape:
                _depth_pred = F.interpolate(_depth_pred[None, None, :], size=_gt_depth.shape[-2:], mode='bilinear', align_corners=True).squeeze()
            if "least_square" == alignment:
                _depth_pred, scale, shift = align_depth_least_square(
                    gt_arr=_gt_depth.cpu().numpy(),
                    pred_arr=_depth_pred.cpu().numpy(),
                    valid_mask_arr=_valid_mask.cpu().numpy(),
                    return_scale_shift=True,
                    max_resolution=None,
                )
                _depth_pred = torch.from_numpy(_depth_pred).to(model.device)
            elif "least_square_disparity" == alignment:
                # convert GT depth -> GT disparity
                gt_disparity, gt_non_neg_mask = depth2disparity(
                    depth=_gt_depth, return_mask=True
                )
                # LS alignment in disparity space
                pred_non_neg_mask = _depth_pred > 0
                valid_nonnegative_mask = _valid_mask & gt_non_neg_mask & pred_non_neg_mask

                disparity_pred, scale, shift = align_depth_least_square(
                    gt_arr=gt_disparity.cpu().numpy(),
                    pred_arr=_depth_pred.cpu().numpy(),
                    valid_mask_arr=valid_nonnegative_mask.cpu().numpy(),
                    return_scale_shift=True,
                    max_resolution=None,
                )
                # convert to depth
                disparity_pred = np.clip(
                    disparity_pred, a_min=1e-3, a_max=None
                )  # avoid 0 disparity
                _depth_pred = disparity2depth(disparity_pred)
            _depth_pred = torch.clamp(_depth_pred, min=min_depth, max=max_depth)
            _depth_pred = torch.clamp(_depth_pred, min=1e-6)
            
            if isinstance(loss, torch.nn.modules.loss._Loss):
                if _valid_mask is not None:
                    _valid_mask = _valid_mask.squeeze()
                    current_loss = loss(
                        _depth_pred[_valid_mask], _gt_depth[_valid_mask]
                    )
                    # ??? why mean
                    loss_list.append(-1 * torch.mean(current_loss).to("cpu").item())
                else:
                    current_loss = loss(
                        _depth_pred, _gt_depth
                    )
                    loss_list.append(-1 * current_loss.cpu().item())
            else:
                current_loss = loss(
                    _depth_pred, _gt_depth, _valid_mask
                )
                loss_list.append(-1 * current_loss.cpu().item())
    assert len(loss_list) == parallel_num, "The number of output loss values should be the same as the number of instances."
    return loss_list
        


def _func_wrapped_for_scipy_direct(theta, perturb_type: str, model: DepthEstimator, loss, img: torch.Tensor, gt_depth: torch.Tensor, valid_mask: torch.Tensor = None,
                                   alignment: str = "least_square", min_depth: float = 0, max_depth: float = 100, depth_raw_linear: torch.Tensor = None, is_norm: bool = False, seed: int = 42, kernel_size: int = 0, reduced_bit_depth: int = 0, transform_func = None,
                                   kwargs: dict = None):
    """
    A wrapper function only for scipy direct to compute the loss given model parameters.

    theta: list, parameters of perturbation to be optimized in order
    perturb_type: str, type of perturbation to be applied
    model: DepthEstimator, depth estimation model
    loss: callable, loss/metric function to compute the error between prediction and ground truth
    img: torch.Tensor, shape (1, C, H, W), input RGB image, not normalized nor perturbed
    gt_depth: torch.Tensor, shape (1, 1, H, W), ground truth depth map
    valid_mask: torch.Tensor, shape (1, H, W), valid mask for depth map
    alignment: str, alignment method to align prediction with ground truth
    min_depth: float, minimum depth value of the dataset
    max_depth: float, maximum depth value of the dataset
    is_norm: bool, whether the input image is normalized to [-1, 1] or in [0, 255], default is False [0, 255]
    transform_func: callable, a function to transform the input image before applying perturbation, default None
    kwargs: dictionary containing additional keyword arguments for the model

    RETURN
    current_loss: torch.Tensor, the computed loss value * -1 (for maximization)
    """
    if kernel_size == 0 and perturb_type == "motion_blur":
        raise ValueError("kernel_size must be specified for motion blur perturbation.")
    if reduced_bit_depth < 1 and perturb_type == "banding":
        raise ValueError("reduce_bit_depth must be at least 1 for color quantization perturbation.")
    theta = tuple(theta)
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

    perturb = Perturbation(perturb_type=perturb_type, transform_func=transform_func)

    img_perturbed = perturb.apply(img, **theta)

    if perturb_type == "geometric":
        gt_depth = gt_depth * valid_mask
        # gt_depth = perturb.apply(gt_depth, **theta)
        gt_depth = geometric_perturb_for_depth(gt_depth, **theta)

        mask_tmp = valid_mask.clone().float()
        mask_tmp = perturb.apply(mask_tmp, **theta)
        gt_depth = gt_depth / (mask_tmp + 1e-8)
        valid_mask = (mask_tmp >= 0.5)
        
    # set seed
    if kwargs is not None and kwargs.get("generator") is not None:
        kwargs["generator"] = torch.Generator(model.device).manual_seed(seed)
        depth_pred = model.batch_infer(img_perturbed, **kwargs)
    else:
        depth_pred = model.infer(img_perturbed)

    if depth_pred.shape != gt_depth.shape:
        depth_pred = F.interpolate(depth_pred[:, None], size=gt_depth.shape[-2:], mode='bilinear', align_corners=True).squeeze(1)
        
    # depth_pred = model.infer(img_perturbed, **kwargs)
    depth_pred = depth_pred.squeeze()
    gt_depth = gt_depth.squeeze()
    valid_mask = valid_mask.squeeze()

    # Align with GT using least square
    if "least_square" == alignment:
        depth_pred, scale, shift = align_depth_least_square(
            gt_arr=gt_depth.cpu().numpy(),
            pred_arr=depth_pred.cpu().numpy(),
            valid_mask_arr=valid_mask.cpu().numpy(),
            return_scale_shift=True,
            max_resolution=None,
        )
        depth_pred = torch.from_numpy(depth_pred).to(model.device)  
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

    depth_pred = torch.clamp(depth_pred, min=min_depth, max=max_depth)
    depth_pred = torch.clamp(depth_pred, min=1e-6)

    # if valid_mask and loss is in torch.nn
    if isinstance(loss, torch.nn.modules.loss._Loss):
        if valid_mask is not None:
            valid_mask = valid_mask.squeeze()
            current_loss = loss(
                depth_pred[valid_mask], gt_depth[valid_mask]
            )
        else:
            current_loss = loss(
                depth_pred, gt_depth
            )
        return -1 * torch.mean(current_loss).to("cpu").item()
    ### ??? why mean
    else:
        current_loss = loss(
            depth_pred, gt_depth, valid_mask
        )
        return -1 * current_loss.cpu().item()