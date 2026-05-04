from PIL import Image
import torch
from src.inference_wrapper.zoedepth.zoedepth_dataset import KITTI, NYUD, HYPERSIM
from src.inference_wrapper.zoedepth.zoedepth_pipeline import zoedepthEstimator
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
        valset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True
    )

    # Load model and preprocessing transform
    zoe_model = zoedepthEstimator(
        checkpoint_path="/path/to/checkpoints/zoe_depth/",
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    # test
    def get_model_para_nums(model):
        num_params = sum(p.numel() for p in model.parameters())
        return num_params
    total_params = get_model_para_nums(zoe_model.model)
    if total_params >= 1e9:
        num_str = f"{total_params/1e9:.2f}G"
    else:
        num_str = f"{total_params/1e6:.2f}M"
    print(f"Total params:{num_str}")
    exit()

    metric = 0
    total = 0
    print("-"*20+"\n"+"Starting evaluation..."+"\n"+"-"*20+"\n")
    for i, sample in tqdm(enumerate(dataloader)):
        image = sample['image'].to(torch.device("cuda:0"))
        depth = sample['depth'].to(torch.device("cuda:0"))

        # Forward pass
        depth_pred = zoe_model.batch_infer(image)
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
# python3 -m src.inference_wrapper.zoedepth.zoedepth_test