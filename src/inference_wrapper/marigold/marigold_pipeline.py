from ..base import DepthEstimator
from external.Marigold.marigold.marigold_depth_pipeline import MarigoldDepthPipeline, MarigoldDepthOutput
import torch

class MarigoldDepthEstimator(DepthEstimator):
    def __init__(self, checkpoint_path: str, 
                half_precision: bool = False, device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the Marigold depth estimator with the given checkpoint.

        Args:
            checkpoint_path: Path to the pre-trained Marigold model checkpoint.
            half_precision: Whether to use half precision for model weights. Default is False.
            device: The device to run the model on. Default is CPU.
        """
        self.device = device
        if half_precision:
            dtype = torch.float16
            variant = "fp16"
        else:
            dtype = torch.float32
            variant = None

        self.pipe: MarigoldDepthPipeline = MarigoldDepthPipeline.from_pretrained(
            checkpoint_path, variant=variant, torch_dtype=dtype
        )
        self.pipe = self.pipe.to(device)

    @torch.no_grad()
    def infer(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Perform depth estimation on the input image.
        !!!!!!
        Input should be in [0, 1] range and of type torch.Tensor. or PIT, so we didn't use this function in our evaluation.
        !!!!!!

        Args:
            image: Input image for depth estimation. Type: torch.Tensor of shape (B, C, H, W).
            **kwargs: Additional keyword arguments for the inference process. Including denoising_steps, ensemble_size, processing_res, match_input_res, resample_method, generator, color_map, show_progress_bar

        Returns:
            Depth map corresponding to the input image. Type: torch.Tensor of shape (B, H, W).
        """
        pipe_out: MarigoldDepthOutput = self.pipe(
            image,
            **kwargs
        )
        depth_pred: torch.Tensor = torch.from_numpy(pipe_out.depth_np).to(self.device)
        return depth_pred
    
    @torch.no_grad()
    def batch_infer(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Perform depth estimation on a batch of input images using MarigoldDepthPipeline.single_infer without ensembling and clipping.

        Args:
            images: Batch of input images for depth estimation. Type: torch.Tensor of shape (B, C, H, W).
            **kwargs: Additional keyword arguments for the inference process. Including denoising_steps, ensemble_size, processing_res, match_input_res, resample_method, generator, color_map, show_progress_bar

        Returns:
            Batch of depth maps corresponding to the input images. Type: torch.Tensor of shape (B, H, W).
        """
        assert image.min() >= -1.0 and image.max() <= 1.0, "Input image should be normalized to [-1, 1]"
        depth_preds: torch.Tensor = self.pipe.single_infer(
            image,
            **kwargs
        )
        # clip depth to [0, 1]
        depth_preds = torch.clamp(depth_preds, min=0.0, max=1.0)
        if len(depth_preds.shape) == 2:
            depth_preds = depth_preds.unsqueeze(0)
            
        depth_preds = depth_preds.squeeze(1)  # B, H, W

        return depth_preds
        