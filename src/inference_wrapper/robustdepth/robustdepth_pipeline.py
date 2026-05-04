import os
import torch
from ..base import DepthEstimator
import external.robustdepth.Robust_Depth.networks as networks
import external.robustdepth.Robust_Depth.networksvit as networksvit

class robustdepthEstimator(DepthEstimator):
    def __init__(self, checkpoint_path: str, encoder: str, device: torch.device, min_depth: float = 0.1, max_depth: float = 80.0, _scale_or_not = False):
        assert encoder in ['vit', 'resnet'], "Only 'vit' or 'resnet' encoders are supported for RobustDepth."
        self.checkpoint_path = checkpoint_path
        self.encoder = encoder
        self.device = device
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.encoder, self.depth_decoder, self.feed_height, self.feed_width = self.load_model()

        self.STEREO_SCALE_FACTOR = 5.4
        self._scale_or_not = _scale_or_not

    def load_model(self):
        if self.encoder == 'vit':
            encoder_path = os.path.join(self.checkpoint_path, "encoder.pth")
            depth_decoder_path = os.path.join(self.checkpoint_path, "depth.pth")
            encoder_dict = torch.load(encoder_path, map_location='cpu')
            self.encoder = networksvit.mpvit_small() #networks.ResnetEncoder(opt.num_layers, False)
            self.encoder.num_ch_enc = [64,128,216,288,288]  # = networks.ResnetEncoder(opt.num_layers, False)
            self.depth_decoder = networksvit.DepthDecoder()

            model_dict = self.encoder.state_dict()
            self.encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
            self.depth_decoder.load_state_dict(torch.load(depth_decoder_path, map_location='cpu'))

            self.feed_height = encoder_dict['height']
            self.feed_width = encoder_dict['width']

            self.name = 'Robust-Depth-MonoVit'
        elif self.encoder == 'resnet':
            encoder_path = os.path.join(self.checkpoint_path, "encoder.pth")
            depth_decoder_path = os.path.join(self.checkpoint_path, "depth.pth")

            self.encoder = networks.ResnetEncoder(18, False)
            loaded_dict_enc = torch.load(encoder_path, map_location="cpu")

            # extract the height and width of image that this model was trained with
            self.feed_height = loaded_dict_enc['height']
            self.feed_width = loaded_dict_enc['width']
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(filtered_dict_enc)

            print("   Loading pretrained decoder")
            self.depth_decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

            loaded_dict = torch.load(depth_decoder_path, map_location="cpu")
            self.depth_decoder.load_state_dict(loaded_dict)
            self.name ='Robust-Depth-Resnet'
        
        self.encoder.to(self.device)
        self.encoder.eval()
        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()
        return self.encoder, self.depth_decoder, self.feed_height, self.feed_width
    
    def disp_to_depth(self, disp: torch.Tensor) -> torch.Tensor:
        """
        Convert disparity to depth.
        From Robust Depth:
        """
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        scaled_disp = disp * (max_disp - min_disp) + min_disp
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
