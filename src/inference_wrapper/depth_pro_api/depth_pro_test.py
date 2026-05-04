import depth_pro
from PIL import Image
import torch
from src.inference_wrapper.depth_pro_api.depth_pro_dataset import KITTI, NYUD, HYPERSIM
from tqdm import tqdm

def rmse_linear(pred, target, valid_mask=None):
    diff = pred[valid_mask] - target[valid_mask]
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    return rmse

with torch.no_grad():
    # load dataset
    # valset = KITTI('external/Depth_Anything_V2/metric_depth/dataset/splits/kitti/val.txt', 'val')
    # valset = NYUD('external/Depth_Anything_V2/metric_depth/dataset/splits/nyud-v2/val.txt', 'val', '/path/to/datasets/nyu_depth_v2')
    valset = HYPERSIM('external/Marigold/data_split/hypersim_depth/filename_list_val_filtered.txt', 'val', '/path/to/datasets/marigold_dataset/hypersim/val')

    dataloader = torch.utils.data.DataLoader(
        valset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True
    )

    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms(
        device=torch.device("cuda:0"),
    )
    model.eval()
    metric = 0
    total = 0
    print("-"*20+"\n"+"Starting evaluation..."+"\n"+"-"*20+"\n")
    for i, sample in tqdm(enumerate(dataloader)):
        image = sample['image'].to(torch.device("cuda:0"))
        depth = sample['depth'].to(torch.device("cuda:0"))

        # print(image.shape, depth.shape)

        # Forward pass
        depth_pred = model.infer(image)["depth"]
        if depth_pred.shape[-2:] != depth.shape[-2:]:
            depth_pred = torch.nn.functional.interpolate(
                depth_pred.unsqueeze(1), size=depth.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)

        rmse_current = rmse_linear(depth_pred, depth, sample['valid_mask'])
        metric += rmse_current.item()
        total += image.shape[0]

        # Save or visualize the depth prediction
        print(f"RMSE linear for batch {i}: {rmse_current.item() / image.shape[0]}")

    print(f"Average RMSE linear: {metric / total}")

# run the following command
# conda activate depth-pro
# python3 -m src.inference_wrapper.depth_pro_api.depth_pro_test