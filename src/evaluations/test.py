import argparse
import json
import yaml

import wandb
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchmetrics.functional import calibration_error
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
from stochman import BCEHessianCalculator as HessianCalculator
from stochman.laplace import DiagLaplace

from src.utils import iou_box_and_variance_map
from src.models.model_helper import create_model
from src.metrics.classification_metrics import IoU
from src.metrics.calibration_metrics import total_entropy
from src.data_loaders.data_helper import create_dataloaders
from src.data_loaders.create_box_datasets import Boxes


def get_epistmic_uncertainty_map(img, epistemic_model, model_name):
    """
    This function is used to calculate the hessian of the
    post-hoc laplace methods
    :param img:
    :param epistemic_model:
    :return: mean and variance maps
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = img.to(device)
    # returns the epistemic uncertainty map for a single image

    if "ensemble" in model_name:
        vals = [torch.sigmoid(logit) for logit in epistemic_model(img)]

        mean = torch.mean(torch.stack(vals), dim=0)
        variance = torch.var(torch.stack(vals), dim=[0])
        # normalize variance
        variance = (variance - torch.min(variance)) / (
            torch.max(variance) - torch.min(variance)
        )
        return mean, variance

    if "dropout" in model_name:
        epistemic_model.enable_dropout()
        vals = []
        # use 50 samples of dropout
        for i in range(50):
            vals.append(torch.sigmoid(epistemic_model(img)))

        mean = torch.mean(torch.stack(vals), dim=0)
        variance = torch.var(torch.stack(vals), dim=[0])

        # normalize variance (?)
        variance = (variance - torch.min(variance)) / (
            torch.max(variance) - torch.min(variance)
        )
        return mean, variance

    h = HessianCalculator(wrt="weight", shape="diagonal", speed="fast")
    h.to(device)
    hessian = h.compute_hessian(img, epistemic_model.combined)

    # mean parameter is the center of the Gaussian
    mean_parameter = parameters_to_vector(epistemic_model.combined.parameters())

    # Sample from the hessian for the ID Image
    laplace = DiagLaplace()
    standard_deviation = laplace.posterior_scale(hessian, prior_prec=1e6)
    samples = laplace.sample(mean_parameter, standard_deviation, n_samples=50)

    vals = []
    for sample in samples:
        vector_to_parameters(sample, epistemic_model.combined.parameters())
        val = epistemic_model(img)
        softmax_val = torch.sigmoid(val)
        vals.append(softmax_val)

    mean = torch.mean(torch.stack(vals), dim=0)
    variance = torch.var(torch.stack(vals), dim=[0])

    # normalize variance
    variance = (variance - torch.min(variance)) / (
        torch.max(variance) - torch.min(variance)
    )
    return mean, variance


class train_pl(pl.LightningModule):
    def __init__(self, config, model, loss_fn):
        super().__init__()
        self.config = config
        self.dataset_name = None
        self.model_name = self.config["architecture"].lower()
        self.lr = self.config["learning_rate"]
        self.model = model
        self.loss_fn = loss_fn

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def test_step(self, batch, batch_idx):
        # This function only considers the epistemic
        # components of these models.

        eval_outputs = {
            "test_iou": None,
            "ece": None,
            "ace": None,
            "box_ratio": None,
            "pixel_ratio": None,
            "testing_px_ratio": None,
        }

        if "isic" in self.dataset_name:
            # for id ISIC test set
            inputs, targets, _ = batch
        elif "derm" in self.dataset_name or "clin" in self.dataset_name:
            inputs = batch[0]
            targets_shape = (
                inputs.shape[0],
                1,
                inputs.shape[2],
                inputs.shape[3],
            )  # 1 for binary classification
            targets = torch.zeros(
                targets_shape, device=self.device
            )  # no lesions in these datasets
        elif "box" in self.dataset_name:
            isic_inputs, inputs, box_coords = batch
            isic_inputs, inputs = torch.squeeze(isic_inputs, 1), torch.squeeze(
                inputs, 1
            )
        elif "pad" in self.dataset_name:
            inputs, targets, _ = batch

        if "dropout" in self.model_name:
            self.model.eval()
            self.model.enable_dropout()
        mean, variance = get_epistmic_uncertainty_map(
            inputs, self.model, self.model_name
        )
        pred_entropy = total_entropy(mean, img_size=self.config["image_size"])
        if "box" in self.dataset_name:
            id_mean, id_variance = get_epistmic_uncertainty_map(
                isic_inputs, self.model, self.model_name
            )
            eval_outputs["pixel_ratio"] = torch.mean(
                (
                    torch.sum(torch.flatten(variance, start_dim=1), dim=1)
                    / torch.sum(torch.flatten(id_variance, start_dim=1), dim=1)
                )
            ).item()

            box_ratio_array = []
            for i in range(variance.shape[0]):
                a, b, c, d = (
                    box_coords[0][i].item(),
                    box_coords[1][i].item(),
                    box_coords[2][i].item(),
                    box_coords[3][i].item(),
                )
                box_ratio_array.append(
                    iou_box_and_variance_map(
                        variance[i, :, :, :].detach().cpu(),
                        box_coordinates=[a, b, c, d],
                    )
                )
            eval_outputs["box_ratio"] = torch.mean(torch.stack(box_ratio_array), dim=0)

        if "pad" not in self.dataset_name and "box" not in self.dataset_name:
            eval_outputs["test_ece"] = calibration_error(
                mean, targets.detach(), task="binary"
            )
            eval_outputs["epi_test_iou"] = float(IoU(targets.detach(), mean.ge(0.5)))

        field_name = "predictive_entropy" + "_" + self.dataset_name
        csv_dict = {
            "model": self.model_name,
            field_name: pred_entropy.cpu(),
        }
        df = pd.DataFrame(csv_dict)

        df.to_csv(
            self.config["architecture"] + "_" + self.dataset_name + "_pred_entropy.csv",
            mode="a",
            index=False,
            header=False,
        )

        return self.log_dict({k: v for k, v in eval_outputs.items() if v is not None})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


def run_experiment(config):
    # Check for GPU
    if torch.cuda.is_available():
        print("\nThe model will be run on GPU.")
    else:
        print("\nNo GPU available!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the random seed for reproducible experiments
    # torch.manual_seed(230)
    # if device == 'cuda':
    #     torch.cuda.manual_seed(230)

    # Create Model
    model, loss = create_model(config, is_test=True)
    model_pl = train_pl.load_from_checkpoint(
        config["chkpt"], config=config, model=model, loss_fn=loss
    )
    model_pl.eval()

    # Execute training in pytorch lightning
    logger = WandbLogger()
    trainer = pl.Trainer(logger=logger)

    """
    Begin Testing
    """
    # out of distribution test
    dataloader_dict = create_dataloaders(config, is_test=True)  # load the ood data

    if "ssn" in model_pl.model_name:
        model_pl.model.is_epistemic = True
    for dataset_name, dataloader in dataloader_dict.items():
        print("now beginning to test dataset: ", dataset_name)
        model_pl.set_dataset_name(dataset_name)
        trainer.test(model_pl, dataloaders=dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="../../configs/isic/evaluation/ssn_dropout.yaml",
        help="path to the desired experiment configs file",
    )
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.full_load(file)

    print("Running Experiment with the Following Configuration:")
    print(json.dumps(config, indent=4))

    # change path to desired pytorch lightning checkpoint save file
    run_experiment(config)
