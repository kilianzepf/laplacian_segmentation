import sys
import os

# Determine the base directory of your project
# (__file__ refers to the path of the current script)
current_script_path = os.path.abspath(__file__)  # Absolute path of the current script
base_dir = os.path.join(
    os.path.dirname(current_script_path), "..", ".."
)  # Navigate up three levels to reach the base directory
base_dir = os.path.abspath(base_dir)  # Ensure the base directory path is absolute

# Add the base directory to sys.path
if base_dir not in sys.path:
    sys.path.append(base_dir)

import yaml
from yaml.loader import SafeLoader
import wandb
import torch
from pytorch_lightning.loggers import WandbLogger
from src.data_loaders.dataset_prostate import ProstateDataModule
from src.data_loaders.dataset_isic import ISICDataModule
from src.data_loaders.dataset_clinical_skin import ClinSkinDataModule
from src.data_loaders.dataset_derm_skin import DermSkinDataModule
from src.data_loaders.dataset_pad_ufes import PADUFESDataModule
from src.utils.utils import generate_visualization_pdf
from argparse import ArgumentParser
import uuid


def main(hparams):
    # Load configuration
    config_path = hparams.config
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    # Initialize DataModule and Model
    if config["dataset"] == "prostate":
        data_module_id = ProstateDataModule(
            config["data_path"],
            batch_size=config["batch_size"],
            resize_to=config["image_size"],
        )
        data_module_random_motion = ProstateDataModule(
            config["data_path"],
            batch_size=config["batch_size"],
            ood_transform="RandomMotion",
            ood_params=config["augmentation_params"],
            resize_to=config["image_size"],
        )
        data_module_random_noise = ProstateDataModule(
            config["data_path"],
            batch_size=config["batch_size"],
            ood_transform="RandomNoise",
            ood_params=config["augmentation_params"],
            resize_to=config["image_size"],
        )
        data_module_random_spike = ProstateDataModule(
            config["data_path"],
            batch_size=config["batch_size"],
            ood_transform="RandomSpike",
            ood_params=config["augmentation_params"],
            resize_to=config["image_size"],
        )
        data_module_random_ghosting = ProstateDataModule(
            config["data_path"],
            batch_size=config["batch_size"],
            ood_transform="RandomGhosting",
            ood_params=config["augmentation_params"],
            resize_to=config["image_size"],
        )
        data_module_dict = {
            "id": data_module_id,
            "random_motion": data_module_random_motion,
            "random_noise": data_module_random_noise,
            "random_spike": data_module_random_spike,
            "random_ghosting": data_module_random_ghosting,
        }
    elif "isic" in config["dataset"]:
        data_module_id = ISICDataModule(
            config["data_path"], batch_size=config["batch_size"]
        )
        data_module_derm = DermSkinDataModule(
            config["data_path_derm"], batch_size=config["batch_size"]
        )
        data_module_clin = ClinSkinDataModule(
            config["data_path_clin"], batch_size=config["batch_size"]
        )
        data_module_padufes = PADUFESDataModule(
            config["data_path_padufes"], batch_size=config["batch_size"]
        )
        data_module_dict = {
            "id": data_module_id,
            "derm": data_module_derm,
            "clin": data_module_clin,
            "padufes": data_module_padufes,
        }

    # Iterate over all the datamodules and create the PDF and the metrics
    for spec, data_module in data_module_dict.items():

        # Try to load the dictionary with the metrics
        try:
            precomputed_dict = torch.load(
                f"results/uq_metrics/{config['dataset']}/{config['model_folder_name']}/{spec}/{config['run_name']}",
                map_location=torch.device("cpu"),
            )
        except FileNotFoundError:
            print("Please run the evaluation script for this model and dataset first!")

        data_module.setup()

        generate_visualization_pdf(
            test_set=data_module.test_dataset,
            inference_tensor=precomputed_dict["sigmoid"],
            aleatoric_tensor=precomputed_dict["aleatoric"],
            epistemic_tensor=precomputed_dict["epistemic"],
            total_tensor=precomputed_dict["total"],
            epistemic_variance_tensor=precomputed_dict["epistemic_pixel_variance"],
            file_path=f"results/separation_plots/{config['dataset']}/{config['model_folder_name']}/{spec}/{config['run_name']}_{spec}_visual.svg",
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="None", type=str, required=True)
    args = parser.parse_args()

    main(args)
