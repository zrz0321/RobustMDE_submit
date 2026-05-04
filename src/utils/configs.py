# src/utils/config.py
import omegaconf
import argparse

def get_config(default_path: str = None) -> omegaconf.DictConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=default_path,
        help="Path to the config file.",
    )
    args, unknown = parser.parse_known_args()
    if args.config is None:
        raise ValueError("Config file path must be provided.")
    config = omegaconf.OmegaConf.load(args.config)
    if unknown:
        print(f"Warning: Unknown arguments {unknown} will be ignored.")
    return config