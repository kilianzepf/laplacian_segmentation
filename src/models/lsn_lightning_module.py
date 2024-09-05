import pytorch_lightning as pl
import torch
from src.models.lsn import LSN
from src.utils.utils import StochasticSegmentationNetworkLossMCIntegral
from src.metrics.classification_metrics import (
    IoU as iou,
    generalized_energy_distance,
    normalized_cross_correlation,
    calculate_box_ratios,
)
import numpy as np
from pytorch_laplace import BCEHessianCalculator
from pytorch_laplace.laplace.diag import DiagLaplace
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class LSNLightning(pl.LightningModule):
    def __init__(self, config, training_run_name):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = LSN(
            name=training_run_name,
            num_channels=self.hparams.input_channels,
            num_classes=self.hparams.num_classes,
            diagonal=self.hparams.diagonal,
        )
        self.loss_function = StochasticSegmentationNetworkLossMCIntegral(
            num_mc_samples=20, pos_weight=self.hparams.pos_weight
        )
        self.hessian = None
        self.final_checkpoint_triggered = (
            False  # Flag to detect the last checkpoint save
        )

        self.iou_metric = iou
        self.name = training_run_name

        # Lists for tracking metrics during training
        self.train_loss_list = []
        self.train_iou_list = []
        self.val_loss_list = []
        self.val_iou_list = []

        # Lists for saving uncertainty metrics after testing
        self.sigmoid = []
        self.total = []
        self.aleatoric = []
        self.epistemic = []
        self.epistemic_hochreiter = []
        self.epistemic_pixel_variance = []
        self.ged = []
        self.ncc_total = []
        self.ncc_aleatoric = []
        self.ncc_epistemic = []
        self.ncc_logits = []
        self.ncc_pixel_variance = []
        self.box_ratio_total = []
        self.box_ratio_aleatoric = []
        self.box_ratio_epistemic = []
        self.box_ratio_logits = []
        self.box_ratio_pixel_variance = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        logits, output_dict = self(images)
        logit_distribution = output_dict["distribution"]
        loss = self.loss_function(logits, targets, logit_distribution)
        iou_score = self.iou_metric(
            targets.detach(), torch.sigmoid(logits.detach()).ge(0.5)
        )
        self.train_loss_list.append(float(loss.item()))
        self.train_iou_list.append(float(iou_score))
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        logits, output_dict = self(inputs)
        loss = self.loss_function(logits, targets, output_dict["distribution"])

        iou_score = self.iou_metric(
            targets.detach(), torch.sigmoid(logits.detach()).ge(0.5)
        )
        self.val_loss_list.append(loss.item())
        self.val_iou_list.append(iou_score)

    def test_step(self, batch, batch_idx):
        self.hessian.to()
        inputs, targets, targets_list = batch
        logits, output_dict = self(inputs)
        logit_distribution = output_dict["distribution"]
        total_uncertainty = self.model.total_uncertainty(
            inputs,
            self.hessian,
            posterior_samples=self.hparams.posterior_samples,
            logit_samples=self.hparams.logit_samples,
            prior_prec=float(self.hparams.prior_prec),
        )
        aleatoric_uncertainty = self.model.aleatoric_uncertainty(
            inputs,
            self.hessian,
            posterior_samples=self.hparams.posterior_samples,
            logit_samples=self.hparams.logit_samples,
            prior_prec=float(self.hparams.prior_prec),
        )
        epistemic_uncertainty = self.model.epistemic_uncertainty(inputs, self.hessian)
        epistemic_uncertainty_hochreiter = self.model.epistemic_uncertainty_hochreiter(
            inputs,
            self.hessian,
            posterior_samples=self.hparams.posterior_samples,
            prior_prec=float(self.hparams.prior_prec),
        )
        epistemic_pixel_variance = self.model.epistemic_pixel_variance(
            inputs,
            self.hessian,
            posterior_samples=self.hparams.posterior_samples,
            prior_prec=float(self.hparams.prior_prec),
        )
        prediction_list = []
        for _ in range(self.hparams.logit_samples):
            prediction_list.append(torch.sigmoid(logit_distribution.sample()).ge(0.5))
        ged = generalized_energy_distance(prediction_list, targets_list)
        ncc_total = normalized_cross_correlation(
            torch.squeeze(total_uncertainty, dim=1),
            torch.var(torch.squeeze(torch.stack(targets_list).float(), dim=2), dim=0),
        )
        ncc_aleatoric = normalized_cross_correlation(
            torch.squeeze(aleatoric_uncertainty, dim=1),
            torch.var(torch.squeeze(torch.stack(targets_list).float(), dim=2), dim=0),
        )
        ncc_epistemic = normalized_cross_correlation(
            torch.squeeze(epistemic_uncertainty, dim=1),
            torch.var(torch.squeeze(torch.stack(targets_list).float(), dim=2), dim=0),
        )
        ncc_logits = normalized_cross_correlation(
            torch.var(
                torch.squeeze(torch.stack(prediction_list).float(), dim=2), dim=0
            ),
            torch.var(torch.squeeze(torch.stack(targets_list).float(), dim=2), dim=0),
        )
        ncc_pixel_variance = normalized_cross_correlation(
            torch.squeeze(epistemic_pixel_variance, dim=1),
            torch.var(torch.squeeze(torch.stack(targets_list).float(), dim=2), dim=0),
        )
        box_ratio_total = calculate_box_ratios(
            torch.squeeze(total_uncertainty, dim=1), torch.squeeze(targets, dim=1)
        )
        box_ratio_aleatoric = calculate_box_ratios(
            torch.squeeze(aleatoric_uncertainty, dim=1), torch.squeeze(targets, dim=1)
        )
        box_ratio_epistemic = calculate_box_ratios(
            torch.squeeze(epistemic_uncertainty, dim=1), torch.squeeze(targets, dim=1)
        )
        box_ratio_logits = calculate_box_ratios(
            torch.var(
                torch.squeeze(torch.stack(prediction_list).float(), dim=2), dim=0
            ),
            torch.squeeze(targets, dim=1),
        )
        box_ratio_pixel_variance = calculate_box_ratios(
            torch.squeeze(epistemic_pixel_variance, dim=1),
            torch.squeeze(targets, dim=1),
        )
        self.sigmoid.append(torch.sigmoid(logits).detach().cpu())
        self.total.append(total_uncertainty.detach().cpu())
        self.aleatoric.append(aleatoric_uncertainty.detach().cpu())
        self.epistemic.append(epistemic_uncertainty.detach().cpu())
        self.epistemic_hochreiter.append(
            epistemic_uncertainty_hochreiter.detach().cpu()
        )
        self.epistemic_pixel_variance.append(epistemic_pixel_variance.detach().cpu())
        self.ged.append(ged)
        self.ncc_total.append(ncc_total)
        self.ncc_aleatoric.append(ncc_aleatoric)
        self.ncc_epistemic.append(ncc_epistemic)
        self.ncc_logits.append(ncc_logits)
        self.ncc_pixel_variance.append(ncc_pixel_variance)
        self.box_ratio_total.append(box_ratio_total)
        self.box_ratio_aleatoric.append(box_ratio_aleatoric)
        self.box_ratio_epistemic.append(box_ratio_epistemic)
        self.box_ratio_logits.append(box_ratio_logits)
        self.box_ratio_pixel_variance.append(box_ratio_pixel_variance)

    def on_train_epoch_end(self):
        avg_train_loss = np.mean(self.train_loss_list)
        avg_train_iou = np.mean(self.train_iou_list)
        self.train_loss_list = []
        self.train_iou_list = []
        self.log(
            "train_loss", avg_train_loss, on_step=False, on_epoch=True, logger=True
        )
        self.log("train_iou", avg_train_iou, on_step=False, on_epoch=True, logger=True)

    def on_validation_epoch_end(self):
        avg_val_loss = np.mean(self.val_loss_list)
        avg_val_iou = np.mean(self.val_iou_list)
        self.val_loss_list = []
        self.val_iou_list = []
        self.log("val_loss", avg_val_loss, on_step=False, on_epoch=True, logger=True)
        self.log("val_iou", avg_val_iou, on_step=False, on_epoch=True, logger=True)

    def on_test_epoch_end(self):
        # Log the mean of the metrics to wandb, aggregated over test set and pixels
        mean_total = torch.mean(torch.cat(self.total, dim=0))
        mean_aleatoric = torch.mean(torch.cat(self.aleatoric, dim=0))
        mean_epistemic = torch.mean(torch.cat(self.epistemic, dim=0))
        mean_epistemic_hochreiter = torch.mean(
            torch.cat(self.epistemic_hochreiter, dim=0)
        )
        mean_epistemic_pixel_variance = torch.mean(
            torch.cat(self.epistemic_pixel_variance, dim=0)
        )
        mean_ged = torch.mean(torch.tensor(self.ged))
        mean_ncc_total = torch.mean(torch.cat(self.ncc_total, dim=0))
        mean_ncc_aleatoric = torch.mean(torch.cat(self.ncc_aleatoric, dim=0))
        mean_ncc_epistemic = torch.mean(torch.cat(self.ncc_epistemic, dim=0))
        mean_ncc_logits = torch.mean(torch.cat(self.ncc_logits, dim=0))
        mean_ncc_pixel_variance = torch.mean(torch.cat(self.ncc_pixel_variance, dim=0))
        mean_box_ratio_total = torch.mean(torch.cat(self.box_ratio_total, dim=0))
        mean_box_ratio_aleatoric = torch.mean(
            torch.cat(self.box_ratio_aleatoric, dim=0)
        )
        mean_box_ratio_epistemic = torch.mean(
            torch.cat(self.box_ratio_epistemic, dim=0)
        )
        mean_box_ratio_logits = torch.mean(torch.cat(self.box_ratio_logits, dim=0))
        mean_box_ratio_pixel_variance = torch.mean(
            torch.cat(self.box_ratio_pixel_variance, dim=0)
        )
        self.log(
            "mean_total_uncertainty",
            mean_total,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "mean_aleatoric_uncertainty",
            mean_aleatoric,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "mean_epistemic_uncertainty",
            mean_epistemic,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "mean_epistemic_uncertainty_hochreiter",
            mean_epistemic_hochreiter,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "mean_epistemic_pixel_variance",
            mean_epistemic_pixel_variance,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "ged",
            mean_ged,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "ncc_total",
            mean_ncc_total,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "ncc_aleatoric",
            mean_ncc_aleatoric,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "ncc_epistemic",
            mean_ncc_epistemic,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "ncc_logits",
            mean_ncc_logits,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "ncc_pixel_variance",
            mean_ncc_pixel_variance,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "box_ratio_total",
            mean_box_ratio_total,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "box_ratio_aleatoric",
            mean_box_ratio_aleatoric,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "box_ratio_epistemic",
            mean_box_ratio_epistemic,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "box_ratio_logits",
            mean_box_ratio_logits,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "box_ratio_pixel_variance",
            mean_box_ratio_pixel_variance,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        # Reset the lists
        self.sigmoid = []
        self.total = []
        self.aleatoric = []
        self.epistemic = []
        self.epistemic_hochreiter = []
        self.epistemic_pixel_variance = []
        self.ged = []
        self.ncc_total = []
        self.ncc_aleatoric = []
        self.ncc_epistemic = []
        self.ncc_logits = []
        self.ncc_pixel_variance = []
        self.box_ratio_total = []
        self.box_ratio_aleatoric = []
        self.box_ratio_epistemic = []
        self.box_ratio_logits = []
        self.box_ratio_pixel_variance = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

    def calculate_hessian(self, dataloader):
        # Calculate the Hessian
        self.model.epistemic_component = True
        h = BCEHessianCalculator(wrt="weight", shape="diagonal", speed="fast")
        device = next(self.parameters()).device
        hessians = []
        for x, _, _ in dataloader:
            x = x.to(device)
            hessians.append(h.compute_hessian(x, self.model.combined))
        self.hessian = sum(hessians)

    def on_save_checkpoint(self, checkpoint):
        # Check if this is the final checkpoint save based on current epoch/step
        is_last_epoch = self.current_epoch == self.trainer.max_epochs - 1

        if is_last_epoch and not self.final_checkpoint_triggered:
            # Last time `on_save_checkpoint` is called, trigger Hessian calculation
            self.calculate_hessian(self.trainer.datamodule.train_dataloader())
            self.final_checkpoint_triggered = True  # Ensure this runs only once
            print("Triggered Hessian calculation for the final checkpoint.")

        # Save the Hessian matrix to the checkpoint
        if self.hessian is not None:
            checkpoint["hessian"] = self.hessian
            print("Hessian saved to checkpoint.")  # Debugging print
        else:
            print("Hessian is None; not saving to checkpoint.")  # Debugging print
