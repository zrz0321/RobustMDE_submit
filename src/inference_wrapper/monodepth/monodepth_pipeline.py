import os
import torch
from ..base import DepthEstimator

import external.monodepth2.networks as networks

class monodepthEstimator(DepthEstimator):
    def __init__(self, checkpoint_path: str, device: torch.device, min_depth: float = 0.1, max_depth: float = 80.0, _scale_or_not = False):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.encoder, self.depth_decoder, self.feed_height, self.feed_width = self.load_model(self.checkpoint_path)
        self.STEREO_SCALE_FACTOR = 5.4
        self._scale_or_not = _scale_or_not
    
    def load_model(self, checkpoint_path: str):
        encoder_path = os.path.join(checkpoint_path, "encoder.pth")
        depth_decoder_path = os.path.join(checkpoint_path, "depth.pth")

        encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location="cpu")
        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']

        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(self.device)
        encoder.eval()

        depth_decoder = networks.DepthDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))
        
        loaded_dict = torch.load(depth_decoder_path, map_location="cpu")
        depth_decoder.load_state_dict(loaded_dict)
        depth_decoder.to(self.device)
        depth_decoder.eval()

        return encoder, depth_decoder, feed_height, feed_width
    
    def disp_to_depth(self, disp):
        """Convert network's sigmoid output into depth prediction
        The formula for this conversion is given in the 'additional considerations'
        section of the paper.
        From monodepth2
        """
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return depth
    
    def infer(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        outputs = self.depth_decoder(features)
        disp = outputs[("disp", 0)]
        depth = self.disp_to_depth(disp)

        if self._scale_or_not:
            depth = depth * self.STEREO_SCALE_FACTOR

        if len(depth.shape) == 2:
            depth = depth.unsqueeze(0)  # Add batch dimension if missing
        depth = depth.squeeze(1)  # Remove channel dimension
        return depth
    
    def batch_infer(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        outputs = self.depth_decoder(features)
        disp = outputs[("disp", 0)]
        depth = self.disp_to_depth(disp)

        if self._scale_or_not:
            depth = depth * self.STEREO_SCALE_FACTOR
            
        if len(depth.shape) == 2:
            depth = depth.unsqueeze(0)  # Add batch dimension if missing
        depth = depth.squeeze(1)  # Remove channel dimension
        return depth