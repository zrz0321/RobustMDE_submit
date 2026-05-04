import cv2
from PIL import Image
import torch
import os
from torch.utils.data import Dataset
from torchvision.transforms import(
    Compose,
    ConvertImageDtype,
    Lambda,
    Normalize,
    ToTensor,
    Resize,
)

# vit's feed height: 192, width: 640
# resnet's feed height: 192, width: 640

class KITTI(Dataset):
    def __init__(self, filelist_path, mode, img_size=(192, 640)):
        if mode != 'val':
            raise NotImplementedError
        
        self.mode = mode
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        self.transform = Compose(
            [
                Resize(img_size),
                ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        img_path = self.filelist[idx].split(' ')[0]
        depth_path = self.filelist[idx].split(' ')[1]

        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32')
        depth = torch.from_numpy(depth)
        valid_mask = depth > 0

        sample = {
            "image": image,
            "depth": depth / 256.0,  # convert in meters
            "valid_mask": valid_mask,
            "image_path": self.filelist[idx].split(' ')[0],
        }
        return sample
    
    def __len__(self):
        return len(self.filelist)
    

class NYUD(Dataset):
    def __init__(self, filelist_path, mode, base_path, img_size=(192, 640)):
        if mode != 'val':
            raise NotImplementedError
        
        self.mode = mode
        self.base_path = base_path
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        self.transform = Compose(
            [
                Resize(img_size),
                ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        img_path = self.filelist[idx].split(' ')[0]
        depth_path = self.filelist[idx].split(' ')[1]

        img_path = os.path.join(self.base_path, img_path)
        depth_path = os.path.join(self.base_path, depth_path)
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32')
        depth = torch.from_numpy(depth)
        valid_mask = depth > 0

        sample = {
            "image": image,
            "depth": depth / 1000.0,  # convert in meters
            "valid_mask": valid_mask,
            "image_path": os.path.join(self.base_path, self.filelist[idx].split(' ')[0]),
        }
        return sample
    
    def __len__(self):
        return len(self.filelist)


class HYPERSIM(Dataset):
    def __init__(self, filelist_path, mode, base_path, img_size=(192, 640)):
        if mode != 'val':
            raise NotImplementedError
        
        self.mode = mode
        self.base_path = base_path
        
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        self.transform = Compose(
            [
                Resize(img_size),
                ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        img_path = self.filelist[idx].split(' ')[0]
        depth_path = self.filelist[idx].split(' ')[1]

        img_path = os.path.join(self.base_path, img_path)
        depth_path = os.path.join(self.base_path, depth_path)
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32')
        depth = torch.from_numpy(depth)
        valid_mask = depth > 0

        sample = {
            "image": image,
            "depth": depth / 1000,  # convert in meters
            "valid_mask": valid_mask,
            "image_path": os.path.join(self.base_path, self.filelist[idx].split(' ')[0]),
        }
        return sample
    
    def __len__(self):
        return len(self.filelist)