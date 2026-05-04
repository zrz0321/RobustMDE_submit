from abc import ABC, abstractmethod
import torch

class DepthEstimator(ABC):
    @abstractmethod
    def infer(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Perform depth estimation on the input image.

        Args:
            image: Input image for depth estimation. Type: torch.Tensor of shape (B, C, H, W). Not normalized, pixel values in [0, 255].

        Returns:
            Depth map corresponding to the input image. Type: torch.Tensor of shape (B, H, W).
        """
        pass