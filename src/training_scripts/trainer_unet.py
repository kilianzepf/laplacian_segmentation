# Import necessary modules
import yaml
import wandb
from yaml.loader import SafeLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.data_loaders.dataset_prostate import ProstateDataModule
from src.data_loaders.dataset_brats import BRATS2DDataModule
from src.data_loaders.dataset_isic import ISICDataModule
from src.models.unet_lightning_module import UnetLightning
from src.callbacks.log_imagegrid_callback import ImageLoggingCallback
from argparse import ArgumentParser


def main(hparams):
    # Load configuration
    config_path = hparams.config
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    # Create a WandbLogger
    wandb_logger = WandbLogger(
        project="LSN",
        group="Deterministic U-net",
        job_type="train",
        config=config,
        save_dir=config["wandb_dir"],
    )

    # Initialize Image Logging Callback
    image_logging_callback = ImageLoggingCallback()

    # Initialize DataModule and Model
    if config["dataset"] == "prostate":
        data_module = ProstateDataModule(
            config["data_path"],
            batch_size=config["batch_size"],
            resize_to=config["image_size"],
        )
    elif "isic" in config["dataset"]:
        data_module = ISICDataModule(
            config["data_path"], batch_size=config["batch_size"]
        )
    elif "brats" in config["dataset"]:
        data_module = BRATS2DDataModule(
            config["data_path"], batch_size=config["batch_size"]
        )

    model = UnetLightning(config, training_run_name=wandb_logger.name)

    # Common setup for Trainer
    trainer_kwargs = {
        "logger": wandb_logger,
        "max_epochs": config["epochs"],
        "accelerator": "gpu",
        "devices": 1,
        "callbacks": [image_logging_callback],
        "limit_train_batches": 0.1,
        "limit_val_batches": 0.2,
    }

    # Conditionally add checkpointing if not in sweep mode
    if not config["sweep"]:
        checkpoint_callback = ModelCheckpoint(
            f"saved_models/{config['dataset']}/{config['model_folder_name']}/{wandb_logger.experiment.name}",
            save_last=True,
        )
        trainer_kwargs["callbacks"].append(checkpoint_callback)

    # Initialize Trainer with the appropriate configuration
    trainer = Trainer(**trainer_kwargs)

    # Train the Model
    if config["continue_training"]:
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path=f"saved_models/{config['dataset']}/{config['model_folder_name']}/{config['run_name']}/last.ckpt",
        )
    else:
        trainer.fit(model, datamodule=data_module)
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="None", type=str, required=True)
    args = parser.parse_args()

    main(args)
