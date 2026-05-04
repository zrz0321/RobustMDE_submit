import torch
from ..base import DepthEstimator
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class zoedepthEstimator(DepthEstimator):
    def __init__(self, checkpoint_path, device: torch.device = torch.device("cpu")):
        """
        Initialize the ZoeDepth depth estimator with the given model.

        Args:
            precision: The precision to run the model in. Default is torch.float32.
            device: The device to run the model on. Default is CPU.
        """
        super().__init__()
        self.model = AutoModelForDepthEstimation.from_pretrained(checkpoint_path)
        self.model = self.model.to(device)
        self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained(checkpoint_path)
    
    @torch.no_grad()
    def infer(self, image: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on the given image.

        Args:
            image: A tensor of shape (1, 3, H, W) representing the input image.
        Returns:
            A tensor of shape (1, H, W) representing the predicted depth map.
        """
        results = self.model(image).predicted_depth
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
        results = self.model(images).predicted_depth
        if len(results.shape) == 2:
            results = results.unsqueeze(0)
        return results
