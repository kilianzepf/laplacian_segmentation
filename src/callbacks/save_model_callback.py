from pytorch_lightning.callbacks import Callback
import os
import torch


class SaveModelCallback(Callback):
    def __init__(self, dataset, is_sweep, run_name, model_folder_name):
        self.model_folder_name = model_folder_name
        self.dataset = dataset
        self.is_sweep = is_sweep
        self.run_name = run_name

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == trainer.max_epochs - 1 and not self.is_sweep:
            os.makedirs(
                f"saved_models/{self.dataset}/{self.model_folder_name}", exist_ok=True
            )
            torch.save(
                pl_module.state_dict(),
                f"saved_models/{self.dataset}/{self.model_folder_name}/{self.run_name}",
            )
