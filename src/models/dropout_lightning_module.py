import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.models.dropout import DropoutUnet
from src.metrics.classification_metrics import (
    IoU as iou,
    generalized_energy_distance,
    normalized_cross_correlation,
    calculate_box_ratios,
)
from src.utils.utils import pixelwise_entropy
import numpy as np


class LightningDropoutUnet(pl.LightningModule):
    def __init__(self, config, training_run_name):
        super(LightningDropoutUnet, self).__init__()
        self.save_hyperparameters(config)
        self.model = DropoutUnet(
            name=training_run_name,
            input_channels=self.hparams.input_channels,
            num_classes=self.hparams.num_classes,
            num_filters=self.hparams.num_filters,
        )
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.hparams.pos_weight)
        )
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
        inputs, targets, _ = batch
        logits = self(inputs)
        loss = self.criterion(logits, targets)
        iou_score = iou(targets.detach(), torch.sigmoid(logits.detach()).ge(0.5))
        self.train_iou_list.append(iou_score)
        self.train_loss_list.append(loss.item())
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        logits_list = []
        sigmoids_list = []
        dropout_samples = self.hparams.dropout_samples

        # Enable dropout and switch to evaluation mode
        self.model.enable_dropout()
        self.model.eval()

        # Perform multiple forward passes
        for _ in range(dropout_samples):
            logits = self(inputs)
            logits_list.append(logits)
            sigmoids_list.append(torch.sigmoid(logits))

        # Calculate mean of the logits and sigmoids
        mean_logits = torch.mean(torch.stack(logits_list), dim=0)
        mean_sigmoids = torch.mean(torch.stack(sigmoids_list), dim=0)

        # Compute loss and IoU
        loss = self.criterion(mean_logits, targets)
        iou_score = iou(targets.detach(), mean_sigmoids.ge(0.5))
        self.val_iou_list.append(iou_score)
        self.val_loss_list.append(loss.item())

    def test_step(self, batch, batch_idx):
        inputs, targets, targets_list = batch
        logits = self(inputs)
        total_uncertainty = self.model.total_uncertainty(
            inputs,
            posterior_samples=self.hparams.dropout_samples,
            logit_samples=self.hparams.logit_samples,
        )
        aleatoric_uncertainty = self.model.aleatoric_uncertainty(
            inputs,
            posterior_samples=self.hparams.dropout_samples,
            logit_samples=self.hparams.logit_samples,
        )
        epistemic_uncertainty = self.model.epistemic_uncertainty(inputs)
        epistemic_uncertainty_hochreiter = self.model.epistemic_uncertainty_hochreiter(
            inputs, posterior_samples=self.hparams.dropout_samples
        )
        epistemic_pixel_variance = self.model.epistemic_pixel_variance(
            inputs, posterior_samples=self.hparams.dropout_samples
        )
        prediction_list = [torch.sigmoid(logits).ge(0.5)]
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
            torch.squeeze(pixelwise_entropy((torch.sigmoid(logits.detach()))), dim=1),
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
            torch.squeeze(pixelwise_entropy(torch.sigmoid(logits.detach())), dim=1),
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
        avg_train_iou = np.stack(self.train_iou_list).mean()
        avg_train_loss = np.stack(self.train_loss_list).mean()
        self.train_iou_list.clear()
        self.train_loss_list.clear()
        self.log(
            "train_loss", avg_train_loss, on_step=False, on_epoch=True, logger=True
        )
        self.log("train_iou", avg_train_iou, on_step=False, on_epoch=True, logger=True)
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    def on_validation_epoch_end(self):
        avg_val_iou = np.stack(self.val_iou_list).mean()
        avg_val_loss = np.stack(self.val_loss_list).mean()
        self.val_iou_list.clear()
        self.val_loss_list.clear()
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
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.learning_rate
        )
        return optimizer
