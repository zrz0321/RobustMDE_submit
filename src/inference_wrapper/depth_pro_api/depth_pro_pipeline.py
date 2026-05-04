import torch
from ..base import DepthEstimator
import depth_pro
from typing import Optional, Union

class DepthProEstimator(DepthEstimator):
    def __init__(self, precision: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")):
        """
        Initialize the Depth Pro API depth estimator with the given model.

        Args:
            precision: The precision to run the model in. Default is torch.float32.
            device: The device to run the model on. Default is CPU.
        """
        super().__init__()
        self.model, self.transform = depth_pro.create_model_and_transforms()
        self.model = self.model.to(device)
        self.model.eval()
        self.precision = precision

        if self.precision == torch.float16:
            self.model = self.model.half()
        
        self.device = device
    
    @torch.no_grad()
    def infer(self, image: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on the given image.

        Args:
            image: A tensor of shape (1, 3, H, W) representing the input image.
        Returns:
            A tensor of shape (1, H, W) representing the predicted depth map.
        """
        results = self.model.infer(image)["depth"]
        if len(results.shape) == 2:
            results = results.unsqueeze(0)
        return results
    
    @torch.no_grad()
    def batch_infer(self, images: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on the given image.

        Args:
            images: A tensor of shape (B, 3, H, W) representing the input image.
        Returns:
            A tensor of shape (B, H, W) representing the predicted depth map.
        """
        results = self.model.infer(images)["depth"]
        if len(results.shape) == 2:
            results = results.unsqueeze(0)
        return results