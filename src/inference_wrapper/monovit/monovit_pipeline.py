import os
import torch
from ..base import DepthEstimator

import external.MonoViT.networks as networks

class monovitEstimator(DepthEstimator):
    def __init__(self, checkpoint_path: str, device: torch.device, min_depth: float = 0.1, max_depth: float = 80.0, _scale_or_not = False, hr = False):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.hr = hr
        if not self.hr:
            self.encoder, self.depth_decoder = self.load_model(self.checkpoint_path)
        else:
            self.depth_decoder = self.load_model(self.checkpoint_path)
        self.STEREO_SCALE_FACTOR = 5.4
        self._scale_or_not = _scale_or_not
    
    def load_model(self, checkpoint_path: str):
        if not self.hr:
            encoder_path = os.path.join(checkpoint_path, "encoder.pth")
            depth_decoder_path = os.path.join(checkpoint_path, "depth.pth")

            encoder = networks.mpvit_small()
            loaded_dict_enc = torch.load(encoder_path, map_location="cpu")

            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
            encoder.load_state_dict(filtered_dict_enc)
            encoder.to(self.device)
            encoder.eval()

            depth_decoder = networks.DepthDecoder()
            
            loaded_dict = torch.load(depth_decoder_path, map_location="cpu")
            depth_decoder.load_state_dict(loaded_dict)
            depth_decoder.to(self.device)
            depth_decoder.eval()

            return encoder, depth_decoder
        else:
            depth_decoder_path = os.path.join(checkpoint_path, "depth.pth")

            depth_dict = torch.load(depth_decoder_path)
            new_dict = {}
            for k,v in depth_dict.items():
                name = k[7:]
                new_dict[name]=v
            
            depth_decoder = networks.DeepNet('mpvitnet')
            depth_decoder.load_state_dict({k: v for k, v in new_dict.items() if k in depth_decoder.state_dict()})
            depth_decoder.to(self.device)
            depth_decoder.eval()
            return depth_decoder
        
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
        if not self.hr:
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
        else:
            outputs = self.depth_decoder(image)
            disp = outputs[("disp", 0)]
            depth = self.disp_to_depth(disp)

            if self._scale_or_not:
                depth = depth * self.STEREO_SCALE_FACTOR

            if len(depth.shape) == 2:
                depth = depth.unsqueeze(0)  # Add batch dimension if missing
            depth = depth.squeeze(1)  # Remove channel dimension
            return depth
    
    def batch_infer(self, image: torch.Tensor) -> torch.Tensor:
        if not self.hr:
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
        else:
            outputs = self.depth_decoder(image)
            disp = outputs[("disp", 0)]
            depth = self.disp_to_depth(disp)

            if self._scale_or_not:
                depth = depth * self.STEREO_SCALE_FACTOR

            if len(depth.shape) == 2:
                depth = depth.unsqueeze(0)  # Add batch dimension if missing
            depth = depth.squeeze(1)  # Remove channel dimension
            return depth