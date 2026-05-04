from external.Depth_Anything_V2.metric_depth.dataset.kitti import KITTI
from external.Depth_Anything_V2.metric_depth.dataset.hypersim import Hypersim
from external.Depth_Anything_V2.metric_depth.dataset.nyud import NYUD
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# img_size = (518, 518)
# valset = KITTI('external/Depth_Anything_V2/metric_depth/dataset/splits/kitti/val.txt', 'val', size=img_size)
valset = Hypersim('external/Depth_Anything_V2/metric_depth/dataset/splits/hypersim/val.txt', 'val')
# valset = NYUD('external/Depth_Anything_V2/metric_depth/dataset/splits/nyud-v2/val.txt', 'val', '/path/to/datasets/nyu_depth_v2', size=img_size)


valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True)


for i, data in tqdm(enumerate(valloader)):
    print(data.keys())
    print(f"Image type{type(data['image'])}, shape: {data['image'].shape}")
    print(f"Depth type{type(data['depth'])}, shape: {data['depth'].shape}")
    # print(f"Valid mask type{type(data['valid_mask'])}, shape: {data['valid_mask'].shape}")
    # print(f"Image path type{type(data['image_path'])}, shape: {data['image_path']}")
    # print(torch.mean(data['image']), torch.min(data['image']), torch.max(data['image']))
    # print(torch.mean(data['depth']), torch.min(data['depth']), torch.max(data['depth']))
    # print(f"Shape of image: {data['image'].shape}\nShape of depth: {data['depth'].shape}\nShape of valid_mask: {data['valid_mask'].shape}\nImage path: {data['image_path']}")
    # print(f"dtype of image: {data['image'].dtype}\ndtype of depth: {data['depth'].dtype}\ndtype of valid_mask: {data['valid_mask'].dtype}")
    exit()


# dict_keys(['image', 'depth', 'valid_mask', 'image_path'])
# shape: image: [B, 3, 518, 1722], depth: [B, 375, 1242], valid_mask: [B, 375, 1242], image_path: list of str

# run command:
# python3 -m src.inference_wrapper.depth_anything.depth_anything_dataset