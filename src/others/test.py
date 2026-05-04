import numpy as np
import matplotlib.pyplot as plt
import os

valid_mask = np.load("/path/to/experiment_results/robustness_analysis/marigold_modified/10_iter/nyud/geometric/0/valid_mask.npy")

path1 = "/path/to/experiment_results/robustness_analysis/marigold_modified/10_iter/nyud/geometric/0/depth_pred_original.npy"
depth_pred_original = np.load(path1)

save_path = "./src/others/tmp/"
os.makedirs(save_path, exist_ok=True)
depth_pred_original = depth_pred_original.squeeze()
plt.imsave(os.path.join(save_path, "depth_pred_original.png"), depth_pred_original, cmap='hot')

path2 = "/path/to/experiment_results/robustness_analysis/marigold_modified/10_iter/nyud/geometric/0/depth_pred_pareto_direct.npy"
depth_pred = np.load(path2)
depth_pred = depth_pred.squeeze()
plt.imsave(os.path.join(save_path, "depth_pred_pareto_direct.png"),
              depth_pred, cmap='hot')

path3 = "/path/to/experiment_results/robustness_analysis/marigold_modified/10_iter/nyud/geometric/0/gt_depth.npy"
gt_depth = np.load(path3)
gt_depth = gt_depth.squeeze()
plt.imsave(os.path.join(save_path, "gt_depth.png"), gt_depth, cmap='hot')
