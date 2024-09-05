from pytorch_lightning.callbacks import Callback
import wandb
from src.utils.utils import make_image_grid
import torch


class ImageLoggingCallback(Callback):
    def __init__(self, every_n_epoch=50):
        self.every_n_epoch = every_n_epoch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        epoch = trainer.current_epoch
        # Log every `every_n_epoch` and for the first batch of the epoch
        if (epoch % self.every_n_epoch == 0) and (batch_idx == 0):
            (
                images,
                masks,
                _,
            ) = batch  # Assuming batch is a tuple of (images, masks, list of masks)

            logits = outputs[
                "logits"
            ]  # Assuming outputs are the logits from your model

            grid = make_image_grid(
                images, masks, torch.sigmoid(logits), required_padding=(0, 0, 0, 0)
            )

            # Log using Wandb
            wandb.log(
                {
                    "Images during Training": [
                        wandb.Image(grid, caption="Images, Targets, Predictions")
                    ]
                }
                # step=epoch,
            )
