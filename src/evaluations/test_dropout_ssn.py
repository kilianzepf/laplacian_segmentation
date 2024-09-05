import yaml
from yaml.loader import SafeLoader
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.data_loaders.dataset_prostate import ProstateDataModule
from src.data_loaders.dataset_brats import BRATS2DDataModule
from src.data_loaders.dataset_isic import ISICDataModule
from src.data_loaders.dataset_clinical_skin import ClinSkinDataModule
from src.data_loaders.dataset_derm_skin import DermSkinDataModule
from src.data_loaders.dataset_pad_ufes import PADUFESDataModule
from src.data_loaders.create_box_datasets import ISICBoxesDataModule, Boxes
from src.models.ssn_dropout_lightning_module import DropoutSSNLightning
from src.callbacks.save_uq_metrics_callback import MetricsSaveCallback
from argparse import ArgumentParser
import uuid


def main(hparams):
    # Load configuration
    config_path = hparams.config
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    # Separate Groups in Wandb for diagonal SSN
    if config["diagonal"]:
        group_name = "Diagonal Dropout SSN"
    else:
        group_name = "Dropout SSN"

    # Generate a unique identifier
    unique_id = uuid.uuid4()
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
    elif config["dataset"] == "brats":
        data_module_id = BRATS2DDataModule(
            config["data_path"],
            batch_size=config["batch_size"],
        )
        data_module_random_motion = BRATS2DDataModule(
            config["data_path"],
            batch_size=config["batch_size"],
            ood_transform="RandomMotion",
            ood_params=config["augmentation_params"],
        )
        data_module_random_noise = BRATS2DDataModule(
            config["data_path"],
            batch_size=config["batch_size"],
            ood_transform="RandomNoise",
            ood_params=config["augmentation_params"],
        )
        data_module_random_spike = BRATS2DDataModule(
            config["data_path"],
            batch_size=config["batch_size"],
            ood_transform="RandomSpike",
            ood_params=config["augmentation_params"],
        )
        data_module_random_ghosting = BRATS2DDataModule(
            config["data_path"],
            batch_size=config["batch_size"],
            ood_transform="RandomGhosting",
            ood_params=config["augmentation_params"],
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
        data_module_box_black_rand = ISICBoxesDataModule(
            batch_size=config["batch_size"], is_random=True, is_black_box=True
        )
        data_module_box_black_fixed = ISICBoxesDataModule(
            batch_size=config["batch_size"], is_random=False, is_black_box=True
        )
        data_module_box_white_rand = ISICBoxesDataModule(
            batch_size=config["batch_size"], is_random=True, is_black_box=False
        )
        data_module_box_white_fixed = ISICBoxesDataModule(
            batch_size=config["batch_size"], is_random=False, is_black_box=False
        )
        data_module_dict = {
            "id": data_module_id,
            "derm": data_module_derm,
            "clin": data_module_clin,
            "padufes": data_module_padufes,
            "box_black_rand": data_module_box_black_rand,
            "box_black_fixed": data_module_box_black_fixed,
            "box_white_rand": data_module_box_white_rand,
            "box_white_fixed": data_module_box_white_fixed,
        }

    # Initialize a new model and load the weights
    checkpoint = torch.load(
        f"saved_models/{config['dataset']}/{config['model_folder_name']}/{config['run_name']}/last.ckpt",
        map_location=torch.device("cpu"),
    )
    model = DropoutSSNLightning(config, training_run_name=config["run_name"])
    model.load_state_dict(checkpoint["state_dict"])

    # Iterate over all the datamodules and test the model
    for spec, data_module in data_module_dict.items():
        # Create a WandbLogger
        wandb_logger = WandbLogger(
            project="LSN",
            group=group_name,
            job_type="uq_metrics_calculation",
            config=config,
            save_dir=config["wandb_dir"],
            tags=[spec, str(unique_id)],
            name=f"{config['run_name']}-{spec}--{unique_id}",
        )

        # Initialize MetricSaving Callback
        metrics_save_callback = MetricsSaveCallback(
            config["dataset"],
            config["run_name"],
            config["model_folder_name"],
            spec,
            config["run_index"],
        )

        # Common setup for Trainer
        trainer_kwargs = {
            "logger": wandb_logger,
            "accelerator": "gpu",
            "devices": 1,
            "callbacks": [metrics_save_callback],
        }

        # Initialize Trainer with the appropriate configuration
        trainer = Trainer(**trainer_kwargs)

        # Test the Model
        trainer.test(model, datamodule=data_module)

        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="None", type=str, required=True)
    args = parser.parse_args()

    main(args)
