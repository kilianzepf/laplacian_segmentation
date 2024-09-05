import torch.nn as nn
import torch
import torch.distributions as td
import pytorch_lightning as pl
from src.utils import *
from src.models.dropout import DropoutUnet
from src.utils.utils import ReshapedDistribution, pixelwise_entropy


class Dropout_SSN(pl.LightningModule):
    def __init__(
        self,
        name,
        num_channels=3,
        num_classes=1,
        num_filters=[32, 64, 128, 192],
        rank: int = 10,
        epsilon=1e-5,
        diagonal=False,
        deterministic=False,
        is_epistmic=False,
    ):
        super().__init__()
        self.name = name
        self.rank = rank
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.is_epistemic = is_epistmic
        self.epsilon = epsilon
        conv_fn = nn.Conv2d
        # whether to use only the diagonal (independent normals)
        self.diagonal = diagonal
        self.deterministic = deterministic

        self.mean_l = conv_fn(8, num_classes, kernel_size=1)
        self.log_cov_diag_l = conv_fn(8, num_classes, kernel_size=1)
        self.cov_factor_l = conv_fn(8, num_classes * rank, kernel_size=1)

        self.unet = DropoutUnet(
            self.name,
            input_channels=self.num_channels,
            num_classes=self.num_classes,
            num_filters=self.num_filters,
            apply_last_layer=False,
        )

        self.combined = nn.Sequential(self.unet, self.mean_l)

    def enable_dropout(self):
        """Call to enable the dropout layers during testtime"""
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def forward(self, image, enforce_lowrank=False):
        if self.is_epistemic:
            return self.combined(image)

        logits = self.unet.forward(image)
        batch_size = logits.shape[0]  # Get the batchsize

        # tensor size num_classesxHxW
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        # Flattens out each image in the batch, size is batchsize x (rest)
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))
        cov_factor = cov_factor.flatten(2, 3)
        cov_factor = cov_factor.transpose(1, 2)

        cov_factor = cov_factor
        cov_diag = cov_diag + self.epsilon

        if self.deterministic:
            # Set the covariance to zero
            base_distribution = td.Independent(
                td.Normal(loc=mean, scale=1e-20 + torch.zeros_like(cov_diag)), 1
            )
        elif self.diagonal:
            # Only allow pixel-wise variance
            base_distribution = td.Independent(
                td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1
            )
        else:
            if enforce_lowrank:
                try:
                    base_distribution = td.LowRankMultivariateNormal(
                        loc=mean, cov_factor=cov_factor, cov_diag=cov_diag
                    )
                except:
                    base_distribution = td.LowRankMultivariateNormal(
                        loc=mean,
                        cov_factor=torch.zeros_like(cov_factor),
                        cov_diag=cov_diag,
                    )
            else:
                # print('Using low-rank approximation of covariance matrix!')
                try:
                    base_distribution = td.LowRankMultivariateNormal(
                        loc=mean, cov_factor=cov_factor, cov_diag=cov_diag
                    )
                except:
                    # print(
                    #    'Covariance became not invertible using independent normals for this batch!')
                    base_distribution = td.Independent(
                        td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1
                    )

        distribution = ReshapedDistribution(
            base_distribution=base_distribution,
            new_event_shape=event_shape,
            validate_args=False,
        )

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        cov_diag_view = cov_diag.view(shape).detach()
        cov_factor_view = (
            cov_factor.transpose(2, 1)
            .view((batch_size, self.num_classes * self.rank) + event_shape[1:])
            .detach()
        )

        output_dict = {
            "logit_mean": logit_mean.detach(),
            "cov_diag": cov_diag_view,
            "cov_factor": cov_factor_view,
            "distribution": distribution,
        }

        return logit_mean, output_dict

    def total_uncertainty(self, image, posterior_samples=50, logit_samples=20):
        """
        Calculate the total uncertainty for the given image.
        Corresponds to the total uncertainty component of formula (1) and (2) in the
        paper "Introducing an Improved Information-Theoretic Measure of Predictive Uncertainty", Hochreiter et al. (2023).
        """
        self.enable_dropout()
        log_probs_list = []
        for _ in range(posterior_samples):
            # for the sample, predict the logit distribution
            _, output_dict = self.forward(image)
            logit_distribution = output_dict["distribution"]
            # sample from the logit distribution

            collect_logit_probs_sample_list = []
            for _ in range(logit_samples):
                collect_logit_probs_sample_list.append(
                    torch.sigmoid(logit_distribution.sample())
                )

            # Step 1. average over probability (logit) predictions
            # collect_logit_probs_sample_list after torch.stack() -> [n_logit_samples, bs, y_hat_probs, img_h, img_w]
            # after torch.mean() -> [bs, y_hat_probs, img_h, img_w]
            px_entropy_step_1 = torch.mean(
                torch.stack(collect_logit_probs_sample_list), dim=0
            )

            log_probs_list.append(px_entropy_step_1)

        # Step 2. Average all the [bs, y_hat_probs, img_h, img_w] along n_weight_samples
        average_log_probs_over_weights = torch.mean(
            torch.stack(log_probs_list, dim=0), dim=0
        )
        # Step 3. Calculate the entropy
        # no sum because we only measure across a single model?
        entropy = pixelwise_entropy(average_log_probs_over_weights)
        return entropy

    def aleatoric_uncertainty(self, image, posterior_samples=50, logit_samples=20):
        """
        Calculate the aleatoric uncertainty for the given image.
        Corresponds to the aleatoric component of formula (2) and (4) in the
        paper "Introducing an Improved Information-Theoretic Measure of Predictive Uncertainty", Hochreiter et al. (2023).
        """
        self.enable_dropout()
        entropy_list = []
        for _ in range(posterior_samples):
            # for the sample, predict the logit distribution
            _, output_dict = self.forward(image)
            logit_distribution = output_dict["distribution"]

            logit_sample_list = []
            for _ in range(logit_samples):
                logit_sample_list.append(torch.sigmoid(logit_distribution.sample()))
            mean_sigmoid = torch.mean(
                torch.stack(logit_sample_list), dim=0
            )  # [bs, class_dim, img_h, img_w]

            entropy_list.append(pixelwise_entropy(mean_sigmoid))

        # average the entropy maps over the sampled networks and return the average entropy map
        return torch.mean(torch.stack(entropy_list), dim=0)

    def epistemic_uncertainty(
        self,
        image,
    ):
        """
        Calculate the epistemic uncertainty for the given image.
        Corresponds to the epistemic component of formula (2) in the
        paper "Introducing an Improved Information-Theoretic Measure of Predictive Uncertainty", Hochreiter et al. (2023).
        """

        return self.total_uncertainty(image) - self.aleatoric_uncertainty(image)

    def epistemic_uncertainty_hochreiter(self, image, posterior_samples=50):
        """
        Calculate the epistemic uncertainty for the given image.
        Corresponds to the epistemic component of formula (4) in the
        paper "Introducing an Improved Information-Theoretic Measure of Predictive Uncertainty", Hochreiter et al. (2023).
        """

        self.enable_dropout()
        kl_sum = 0
        for _ in range(posterior_samples):
            for _ in range(posterior_samples):
                # predict the logit distribution for both NN samples
                _, output_dict_1 = self.forward(image, enforce_lowrank=True)
                _, output_dict_2 = self.forward(image, enforce_lowrank=True)
                logit_distribution_1 = output_dict_1["distribution"]
                logit_distribution_2 = output_dict_2["distribution"]

                # Check if parameters of both distributions are not identical
                if (
                    not self.diagonal
                ):  # The independent normals do not have a cov_factor
                    assert not torch.allclose(
                        logit_distribution_1.base_distribution.loc,
                        logit_distribution_2.base_distribution.loc,
                    ), "Means of the distributions are identical"
                    assert not torch.allclose(
                        logit_distribution_1.base_distribution.cov_diag,
                        logit_distribution_2.base_distribution.cov_diag,
                    ), "Diagonals of the distributions are identical"

                # Define the Transformed Distributions with f being the sigmoid function
                logit_distribution_1 = td.TransformedDistribution(
                    logit_distribution_1.base_distribution, td.SigmoidTransform()
                )

                logit_distribution_2 = td.TransformedDistribution(
                    logit_distribution_2.base_distribution, td.SigmoidTransform()
                )

                # Calculate the KL Divergence between the two distributions
                kl_div = td.kl_divergence(logit_distribution_1, logit_distribution_2)

                # Assert that all values in the KL divergence tensor are non-negative
                # assert torch.all(kl_div >= 0), "KL divergence contains negative values"
                # FIXME: The PyTorch implementation of the KL divergence is numerically unstable and can return negative values
                # Clamp negative values to zero
                kl_div_clamped = torch.clamp(kl_div, min=0)

                kl_sum += kl_div_clamped

        return kl_sum / (posterior_samples**2)

    def epistemic_pixel_variance(self, image, posterior_samples=50):
        """
        Calculate the epistemic uncertainty for the given image.
        This function returns the pixel wise entropy of the mean predictions given by NN samples from the Laplace Approximation.
        """
        self.enable_dropout()
        sigmoid_list = []
        for _ in range(posterior_samples):
            # for the sample, predict the logit mean
            logit_mean, _ = self.forward(image)
            # calculate the sigmoid of the logit mean and save the sigmoid map
            sigmoid_list.append(torch.sigmoid(logit_mean))

        # Calculate the pixel wise variance of the sigmoid maps
        sigmoid_map = torch.stack(sigmoid_list)
        sigmoid_variance = torch.var(sigmoid_map, dim=0)

        return sigmoid_variance
