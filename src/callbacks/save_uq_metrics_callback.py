from pytorch_lightning.callbacks import Callback
import os
import torch


class MetricsSaveCallback(Callback):
    def __init__(self, dataset, run_name, model_folder_name, spec, run_index=0):
        self.model_folder_name = model_folder_name
        self.dataset = dataset
        self.run_name = run_name
        self.spec = spec
        self.run_index = run_index

    def on_test_epoch_end(self, trainer, pl_module):
        sigmoid = pl_module.sigmoid
        total_uncertainty = pl_module.total
        aleatoric_uncertainty = pl_module.aleatoric
        epistemic_uncertainty = pl_module.epistemic
        epistemic_uncertainty_hochreiter = pl_module.epistemic_hochreiter
        epistemic_pixel_variance = pl_module.epistemic_pixel_variance

        metrics = {
            "sigmoid": torch.cat(sigmoid, dim=0),
            "total": torch.cat(total_uncertainty, dim=0),
            "aleatoric": torch.cat(aleatoric_uncertainty, dim=0),
            "epistemic": torch.cat(epistemic_uncertainty, dim=0),
            "epistemic_hochreiter": torch.cat(epistemic_uncertainty_hochreiter, dim=0),
            "epistemic_pixel_variance": torch.cat(epistemic_pixel_variance, dim=0),
        }

        os.makedirs(
            f"results/uq_metrics/{self.dataset}/{self.model_folder_name}/{self.spec}",
            exist_ok=True,
        )
        torch.save(
            metrics,
            f"results/uq_metrics/{self.dataset}/{self.model_folder_name}/{self.spec}/{self.run_name}_run_{self.run_index}",
        )

        print("Saved Preds and UQ Metrics to disk!")
