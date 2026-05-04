from ..base import DepthEstimator
from external.Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
import torch
import torch.nn.functional as F

class DADepthEstimator(DepthEstimator):
    def __init__(self, checkpoint_path: str,
                 encoder: str = 'vitl',
                 max_depth: float = 20.0,
                 device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the Depth Anything depth estimator with the given checkpoint.

        Args:
            checkpoint_path: Path to the pre-trained Marigold model checkpoint.
            encoder: The type of encoder to use. Choices are 'vits', 'vitb', 'vitl', 'vitg'. Default is 'vitl'.
            max_depth: The maximum depth value to predict. Default is 20.0.
            device: The device to run the model on. Default is CPU.
            resample_size: The size to which input images will be resampled. Default is (518, 518).
        """
        self.device = device
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        ckpt = ckpt['model'] if 'model' in ckpt else ckpt
        ckpt = {".".join(k.split(".")[1:]) if k.startswith("module.") else k: v for k, v in ckpt.items()}
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def infer(self, image: torch.Tensor) -> torch.Tensor:
        """
        Perform depth estimation on the input image.
        Input should be normalized to [0, 1].

        Args:
            image: Input image for depth estimation. Type: torch.Tensor of shape (B, C, H, W).
            
        Returns:
            Depth map corresponding to the input image. Type: torch.Tensor of shape (B, H, W).
        """
        depth_pred: torch.Tensor = self.model(image.to(self.device)).squeeze(1)
        return depth_pred
    

    @torch.no_grad()
    def batch_infer(self, image: torch.Tensor) -> torch.Tensor:
        """
        Perform depth estimation on a batch of input images.
        Input should be normalized to [0, 1].

        Args:
            images: Batch of input images for depth estimation. Type: torch.Tensor of shape (B, C, H, W).

        Returns:
            Batch of depth maps corresponding to the input images. Type: torch.Tensor of shape (B, H, W).
        """

        depth_pred: torch.Tensor = self.model(image.to(self.device)).squeeze(1)
        return depth_pred