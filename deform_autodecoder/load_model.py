from pathlib import Path
import sys

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deform_autodecoder.model import DeformAutoencoder


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_model_from_config(config_path):
    config = load_config(config_path)
    model_cfg = config["model"]

    model = DeformAutoencoder(
        input_channels=int(config["preprocess"]["num_channels"]),
        latent_channels=int(model_cfg["latent_channels"]),
        output_activation=model_cfg["output_activation"],
    )
    return model, config


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    model, config = create_model_from_config(root / "config.yaml")
    print("model:", model.__class__.__name__)
    print("input_channels:", config["preprocess"]["num_channels"])
    print("latent_channels:", config["model"]["latent_channels"])
    print("output_activation:", config["model"]["output_activation"])
