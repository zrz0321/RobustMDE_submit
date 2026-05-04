import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import torch

class SaveTools:
    def __init__(self, base_dir="experiment_results", exp_name=None):
        """
        Initialize SaveTools with a base directory and optional experiment name.
        Inputs:
            base_dir (str): The base directory to save experiments.
            exp_name (str): Optional name for the experiment. If None, an auto-generated name
        """
        self.base_dir = base_dir
        # if not exists, create the base directory recursively
        os.makedirs(base_dir, exist_ok=True)

        # Automatically generate experiment name by current date and time if not provided
        if exp_name is None:
            exp_name = self._get_auto_exp_name()

        self.save_dir = os.path.join(base_dir, exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "plots"), exist_ok=True)

        print(f"[SaveTools] Experiment directory: {self.save_dir}")

    def _get_auto_exp_name(self):
        """
        Get current date and time as a string for auto-generated experiment name.
        """
        now = datetime.now()
        return now.strftime("%Y%m%d_%H%M%S")

    def save_config(self, config, additional_name=None):
        """
        Save configuration to file:
        supported formats:
        - dict → JSON
        - OmegaConf.DictConfig → YAML
        """
        if isinstance(config, dict):
            if additional_name:
                path = os.path.join(self.save_dir, f"config_{additional_name}.json")
            else:
                path = os.path.join(self.save_dir, "config.json")
            with open(path, "w") as f:
                json.dump(config, f, indent=4)
            print(f"[SaveTools] Saved dict config -> {path}")

        elif isinstance(config, DictConfig):
            path = os.path.join(self.save_dir, "config.yaml")
            with open(path, "w") as f:
                OmegaConf.save(config, f)
            print(f"[SaveTools] Saved OmegaConf config -> {path}")

        else:
            raise TypeError("config must be dict or OmegaConf.DictConfig")

    def save_as_npy(self, file, path, name):
        """
        Save a numpy array or tensor to a .npy file.
        Inputs:
            file (np.ndarray or torch.Tensor): The data to save.
            path (str): The directory path to save the file.
            name (str): The name of the file (without extension).
        """
        if isinstance(file, torch.Tensor):
            file = file.cpu().numpy()

        if not isinstance(file, np.ndarray):
            raise TypeError("file must be a numpy array or torch tensor")
        
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            
        full_path = os.path.join(path, f"{name}.npy")
        np.save(full_path, file)
        print(f"[SaveTools] Saved numpy array -> {full_path}")

    def save_metrics(self, metrics_dict):
        """
        Save evaluation metrics.
        metrics_dict should be a dictionary with metric names as keys and their values.
        """
        path = os.path.join(self.save_dir, "metrics.json")
        with open(path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"[SaveTools] Saved metrics -> {path}")

    def save_plot(self, fig, name="plot.png"):
        """
        Save a matplotlib figure to the specified directory.
        """
        path = os.path.join(self.save_dir, "plots", name)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[SaveTools] Saved plot -> {path}")
    
    def save_heat_map(self, data, name="heatmap.png", cmap="hot"):
        """
        Save a heat map from a 2D numpy array.
        data should be a 2D numpy array or tensor.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("data must be a 2D numpy array or tensor")
        plt.figure(figsize=(8, 6))
        plt.title("Heat Map")
        path = os.path.join(self.save_dir, "plots", name)
        plt.imsave(path, data, cmap=cmap)
        plt.close()
        print(f"[SaveTools] Saved heat map -> {path}")


    def save_log(self, text, append=True):
        """
        Save a log message to a text file.
        If append is True, it appends to the existing log file, otherwise it overwrites
        """
        path = os.path.join(self.save_dir, "log.txt")
        mode = "a" if append else "w"
        with open(path, mode) as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")
        print(f"[SaveTools] Saved log -> {path}")


def save_json_to_file(data, file_path):
    """
    Save a dictionary to a JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved JSON data to {file_path}")