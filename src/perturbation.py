import torch
from math import pi as math_pi
from kornia.geometry.transform import warp_affine
from kornia.color import rgb_to_hsv, hsv_to_rgb, rgb_to_ycbcr, ycbcr_to_rgb
from kornia.filters import motion_blur
from math import pi as math_pi

class Perturbation():
    """
    A module to apply perturbations to input RGB image.
    """
    def __init__(self, perturb_type=None, transform_func=None):
        """
        perturb_type: str, "geometric", "color_shift", "motion_blur", "banding", or None
        1. geometric: apply geometric perturbation (scaling and translation)
            parameters: s_hor, s_vrt, t_hor, t_vrt
        2. color_shift: apply color shift perturbation (hue, saturation, brightness
            parameters: hue, sat, brt
        3. motion_blur: apply motion blur perturbation
            parameters: kernel_size(int), angle, direction
        4. banding: apply banding perturbation (not implemented yet)
            parameters: luma, cb, cr, reduced_bit_depth(int)
        5. None: no perturbation will be applied

        transform_func: function, a function to transform input to [0, 1] when passing kwargs mode=\"to\",
            and transform back to required range when passing kwargs mode=\"back\". It takes two arguments: img and mode and returns the transformed img.
        """
        super(Perturbation, self).__init__()
        self.perturb_type = perturb_type
        assert self.perturb_type in ["geometric", "color_shift", "motion_blur", "banding", None], "Invalid perturbation type"

        self.transform_func = transform_func
        # if self.perturb_type is None:
        #     print("No perturbation will be applied.")
        # else:
        #     print(f"Perturbation type: {self.perturb_type}")
    
    def apply(self, img, **kwargs):
        """
        Apply the specified perturbation to the input image.
        
        img: torch.Tensor, shape (B, C, H, W), input RGB image, normalized to [-1, 1]
        
        return: torch.Tensor, shape (B, C, H, W), perturbed image
        """
        if self.perturb_type is None:
            return img
        
        self._check_if_params_valid(**kwargs)
        
        if self.perturb_type == "geometric":
            s_hor, s_vrt, t_hor, t_vrt = kwargs["s_hor"], kwargs["s_vrt"], kwargs["t_hor"], kwargs["t_vrt"]
            if self.transform_func is not None:
                img = self.transform_func(img, mode="to")

            if kwargs.get("geometric_mode") is not None:
                geometric_mode = kwargs["geometric_mode"]
                B, C, H, W = img.shape
                M = torch.tensor([[s_hor, 0, t_hor],
                                [0, s_vrt, t_vrt]], dtype=torch.float).unsqueeze(0).repeat(B, 1, 1).to(img.device)
                img_out = warp_affine(img, M, dsize=(H, W), mode=geometric_mode)
            # Apply geometric perturbation (scaling and translation)
            else:
                B, C, H, W = img.shape
                M = torch.tensor([[s_hor, 0, t_hor],
                                [0, s_vrt, t_vrt]], dtype=torch.float).unsqueeze(0).repeat(B, 1, 1).to(img.device)
                img_out = warp_affine(img, M, dsize=(H, W))

            if self.transform_func is not None:
                img_out = self.transform_func(img_out, mode="back")

            return img_out

        elif self.perturb_type == "color_shift":
            hue, sat, brt = kwargs["hue"], kwargs["sat"], kwargs["brt"]
            # Apply color shift perturbation (hue, saturation, brightness)

            if self.transform_func is not None:
                img = self.transform_func(img, mode="to")
                
            # input to rgb_to_hsv() should be in range [0, 1]    
            img_hsv = rgb_to_hsv(img)
            img_hsv[:, 0] = torch.remainder(img_hsv[:, 0] + hue, 2 * math_pi)  # Hue
            img_hsv[:, 1] = torch.clamp(img_hsv[:, 1] * sat, min=0, max=1)  # Saturation
            img_hsv[:, 2] = torch.clamp(img_hsv[:, 2] + brt, min=0, max=1)  # Brightness
            img_out = hsv_to_rgb(img_hsv)

            if self.transform_func is not None:
                img_out = self.transform_func(img_out, mode="back")

            return img_out
        
        elif self.perturb_type == "motion_blur":
            if self.transform_func is not None:
                img = self.transform_func(img, mode="to")

            kernel_size, angle, direction = kwargs["kernel_size"], kwargs["angle"], kwargs["direction"]
            # Apply motion blur perturbation
            img_out = motion_blur(img, kernel_size=kernel_size, angle=angle, direction=direction)

            if self.transform_func is not None:
                img_out = self.transform_func(img_out, mode="back")
            return img_out
        elif self.perturb_type == "banding":
            if self.transform_func is not None:
                img = self.transform_func(img, mode="to")

            luma, cb, cr, b = kwargs["luma"], kwargs["cb"], kwargs["cr"], kwargs["reduced_bit_depth"]
            img_ycbcr = rgb_to_ycbcr(img)
            Delta = 1.0 / (2**b - 1)
            # quantization
            img_ycbcr[:, 0] = torch.clamp(torch.round((img_ycbcr[:, 0] + luma) / Delta) * Delta, min=0, max=1)
            img_ycbcr[:, 1] = torch.clamp(torch.round((img_ycbcr[:, 1] + cb) / Delta) * Delta, min=0, max=1)
            img_ycbcr[:, 2] = torch.clamp(torch.round((img_ycbcr[:, 2] + cr) / Delta) * Delta, min=0, max=1)

            # back to RGB
            img_out = ycbcr_to_rgb(img_ycbcr)

            if self.transform_func is not None:
                img_out = self.transform_func(img_out, mode="back")
            return img_out
            
        
    def _check_if_params_valid(self, **kwargs):
        """
        check if the parameters for perturbation are valid.
        """
        if self.perturb_type == "geometric":
            assert "s_hor" in kwargs and "s_vrt" in kwargs, "Missing parameters 's_hor' or 's_vrt' for geometric perturbation"
            assert "t_hor" in kwargs and "t_vrt" in kwargs, "Missing parameters 't_hor' or 't_vrt' for geometric perturbation"

        elif self.perturb_type == "color_shift":
            assert "hue" in kwargs, "Missing parameters 'hue' for color shift perturbation"
            assert "sat" in kwargs, "Missing parameters 'sat' for color shift perturbation"
            assert "brt" in kwargs, "Missing parameters 'brt' for color shift perturbation"

        elif self.perturb_type == "motion_blur":
            assert "kernel_size" in kwargs, "Missing parameters 'kernel_size' for motion blur perturbation"
            assert "angle" in kwargs, "Missing parameters 'angle' for motion blur perturbation"
            assert "direction" in kwargs, "Missing parameters 'direction' for motion blur perturbation"
        
        elif self.perturb_type == "banding":
            assert "luma" in kwargs, "Missing parameters 'luma' for banding perturbation"
            assert "cb" in kwargs, "Missing parameters 'cb' for banding perturbation"
            assert "cr" in kwargs, "Missing parameters 'cr' for banding perturbation"
            assert "reduced_bit_depth" in kwargs, "Missing parameters 'reduced_bit_depth' for banding perturbation"


def get_parameter_range(perturb_type:str, gamma: float, pic_size: tuple = None):
    """
    Get the range of parameters for the specified perturbation type.
    INPUT:
        perturb_type: str, "geometric", "color_shift", "motion_blur"
        gamma: float, the gamma value for perturbation
        pic_size: tuple, (H, W), the size of the input image, used for geometric perturbation
    """
    if perturb_type == "geometric":
        assert pic_size is not None, "pic_size must be provided for geometric perturbation"
        H, W = pic_size
        s_hor_range = (1 - gamma, 1 + gamma)
        s_vrt_range = (1 - gamma, 1 + gamma)
        t_hor_range = (-gamma * W, gamma * W)
        t_vrt_range = (-gamma * H, gamma * H)
        return [s_hor_range, s_vrt_range, t_hor_range, t_vrt_range]
    elif perturb_type == "color_shift":
        hue_range = (-1 * math_pi * gamma, math_pi * gamma)
        sat_range = (1 - gamma, 1 + gamma)
        brt_range = (-1 * gamma, gamma)
        return [hue_range, sat_range, brt_range]
    elif perturb_type == "motion_blur":
        angle_range = (-1 * math_pi, math_pi)
        direction_range = (-1, 1)
        return [angle_range, direction_range]
    elif perturb_type == "banding":
        luma_range = (-1 * gamma, 1 * gamma)
        cb_range = (-1 * gamma / 1.5, 1 * gamma / 1.5)
        cr_range = (-1 * gamma / 1.5, 1 * gamma / 1.5)
        return [luma_range, cb_range, cr_range]
    

def geometric_perturb_for_depth(depth, **kwargs):
    s_hor, s_vrt, t_hor, t_vrt = kwargs["s_hor"], kwargs["s_vrt"], kwargs["t_hor"], kwargs["t_vrt"]

    if kwargs.get("geometric_mode") is not None:
        geometric_mode = kwargs["geometric_mode"]
        B, C, H, W = depth.shape
        M = torch.tensor([[s_hor, 0, t_hor],
                        [0, s_vrt, t_vrt]], dtype=torch.float).unsqueeze(0).repeat(B, 1, 1).to(depth.device)
        perturbed_depth = warp_affine(depth, M, dsize=(H, W), mode=geometric_mode)
    # Apply geometric perturbation (scaling and translation)
    else:
        B, C, H, W = depth.shape
        M = torch.tensor([[s_hor, 0, t_hor],
                        [0, s_vrt, t_vrt]], dtype=torch.float).unsqueeze(0).repeat(B, 1, 1).to(depth.device)
        perturbed_depth = warp_affine(depth, M, dsize=(H, W))
    perturbed_depth = torch.clamp(perturbed_depth, min=0)
    return perturbed_depth
