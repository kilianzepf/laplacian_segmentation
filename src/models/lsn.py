import torch
import torch.distributions as td
from src.utils import *
import nnj
from src.utils.utils import ReshapedDistribution
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import pytorch_lightning as pl

from pytorch_laplace import BCEHessianCalculator
from pytorch_laplace.laplace.diag import DiagLaplace
from src.utils.utils import (
    pixelwise_entropy,
)


class LSN(pl.LightningModule):
    def __init__(
        self,
        name,
        num_channels=3,
        num_classes=1,
        rank: int = 10,
        epsilon=1e-5,
        diagonal=False,
        deterministic=False,
        epistemic_component=False,
    ):
        super().__init__()
        self.name = name
        self.rank = rank
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.diagonal = diagonal  # set true for independent normals
        self.deterministic = deterministic
        self.epistemic_component = epistemic_component
        conv_fn = nnj.Conv2d

        self.mean_l = conv_fn(8, num_classes, kernel_size=1)
        self.log_cov_diag_l = conv_fn(8, num_classes, kernel_size=1)
        self.cov_factor_l = conv_fn(8, num_classes * rank, kernel_size=1)

        self.unet = nnj.Sequential(
            nnj.Conv2d(num_channels, 8, 3, stride=1, padding=1),
            nnj.Tanh(),
            nnj.Conv2d(8, 8, 3, stride=1, padding=1),
            nnj.Tanh(),
            nnj.SkipConnection(
                nnj.MaxPool2d(2),
                nnj.Conv2d(8, 16, 3, stride=1, padding=1),
                nnj.Tanh(),
                nnj.Conv2d(16, 16, 3, stride=1, padding=1),
                nnj.Tanh(),
                nnj.SkipConnection(
                    nnj.MaxPool2d(2),
                    nnj.Conv2d(16, 32, 3, stride=1, padding=1),
                    nnj.Tanh(),
                    nnj.Conv2d(32, 32, 3, stride=1, padding=1),
                    nnj.Tanh(),
                    nnj.SkipConnection(
                        nnj.MaxPool2d(2),
                        nnj.Conv2d(32, 64, 3, stride=1, padding=1),
                        nnj.Tanh(),
                        nnj.Conv2d(64, 64, 3, stride=1, padding=1),
                        nnj.Tanh(),
                        nnj.SkipConnection(
                            nnj.MaxPool2d(2),
                            nnj.Conv2d(64, 128, 3, stride=1, padding=1),
                            nnj.Tanh(),
                            nnj.Conv2d(128, 128, 3, stride=1, padding=1),
                            nnj.Upsample(scale_factor=2),
                            nnj.Tanh(),
                        ),
                        nnj.Conv2d(128 + 64, 64, 3, stride=1, padding=1),
                        nnj.Tanh(),
                        nnj.Conv2d(64, 64, 3, stride=1, padding=1),
                        nnj.Upsample(scale_factor=2),
                        nnj.Tanh(),
                    ),
                    nnj.Conv2d(64 + 32, 32, 3, stride=1, padding=1),
                    nnj.Tanh(),
                    nnj.Conv2d(32, 32, 3, stride=1, padding=1),
                    nnj.Upsample(scale_factor=2),
                    nnj.Tanh(),
                ),
                nnj.Conv2d(32 + 16, 16, 3, stride=1, padding=1),
                nnj.Tanh(),
                nnj.Conv2d(16, 16, 3, stride=1, padding=1),
                nnj.Upsample(scale_factor=2),
                nnj.Tanh(),
            ),
            nnj.Conv2d(16 + 8, 8, 3, stride=1, padding=1),
            nnj.Tanh(),
            nnj.Conv2d(8, 8, 3, stride=1, padding=1),
            nnj.Tanh(),
        )

        self.combined = nnj.Sequential(self.unet, self.mean_l, add_hooks=True)

    def forward(self, image):
        if self.epistemic_component:
            return self.combined(image)

        self.combined.feature_maps = []  # Reset feature maps in nnj.Sequential
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

        # A dictionary that is handed over to the training loop for logging
        # infos_for_logging = {"mean": mean, "cov_factor": cov_factor, "cov_diag": cov_diag, "Max value of mean": torch.max(
        #    mean), "Min value of mean": torch.min(mean), "Max Value of Cov_diag": torch.max(cov_diag), "Max Value of Cov_factor": torch.max(cov_factor)}

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
            try:
                base_distribution = td.LowRankMultivariateNormal(
                    loc=mean, cov_factor=cov_factor, cov_diag=cov_diag
                )
            except:
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

        return logit_mean.detach(), output_dict

    def logit_distribution_from_laplace_samples(self, nn_sample, image):
        # nn_sample is a parameter vector for the mean network self.combined
        # use the sampled mean network and self.log_cov_diag_l and self.cov_factor_l to calculate the logit distribution

        vector_to_parameters(nn_sample, self.combined.parameters())

        # self.combined.feature_maps = []  # Reset feature maps in nnj.Sequential
        # logits = self.unet.forward(image)
        logits = self.combined[0].forward(image)
        batch_size = logits.shape[0]  # Get the batchsize

        # tensor size num_classesxHxW
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = self.combined[1].forward(logits)
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

        # A dictionary that is handed over to the training loop for logging
        # infos_for_logging = {"mean": mean, "cov_factor": cov_factor, "cov_diag": cov_diag, "Max value of mean": torch.max(
        #    mean), "Min value of mean": torch.min(mean), "Max Value of Cov_diag": torch.max(cov_diag), "Max Value of Cov_factor": torch.max(cov_factor)}

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
            try:
                base_distribution = td.LowRankMultivariateNormal(
                    loc=mean, cov_factor=cov_factor, cov_diag=cov_diag
                )
            except:
                base_distribution = td.LowRankMultivariateNormal(
                    loc=mean, cov_factor=torch.zeros_like(cov_factor), cov_diag=cov_diag
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

        return logit_mean.detach(), output_dict

    def total_uncertainty(
        self, image, hessian, posterior_samples=50, logit_samples=20, prior_prec=1e6
    ):
        """
        Calculate the total uncertainty for the given image.
        Corresponds to the total uncertainty component of formula (1) and (2) in the
        paper "Introducing an Improved Information-Theoretic Measure of Predictive Uncertainty", Hochreiter et al. (2023).
        """

        mean_parameter = parameters_to_vector(self.combined.parameters())
        laplace = DiagLaplace()

        # sample from the Laplace Approximation
        standard_deviation = laplace.posterior_scale(hessian, prior_prec=prior_prec)

        samples = laplace.sample_from_normal(
            mean_parameter,
            standard_deviation,
            n_samples=posterior_samples,
        )

        log_probs_list = []
        for sample in samples:
            # for the sample, predict the logit distribution
            _, output_dict = self.logit_distribution_from_laplace_samples(
                sample, image=image
            )
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

        # reset the epistemic component to false
        self.epistemic_component = False
        return entropy

    def aleatoric_uncertainty(
        self, image, hessian, posterior_samples=50, logit_samples=20, prior_prec=1e6
    ):
        """
        Calculate the aleatoric uncertainty for the given image.
        Corresponds to the aleatoric component of formula (2) and (4) in the
        paper "Introducing an Improved Information-Theoretic Measure of Predictive Uncertainty", Hochreiter et al. (2023).
        """

        mean_parameter = parameters_to_vector(self.combined.parameters())
        laplace = DiagLaplace()
        # sample from the Laplace Approximation
        standard_deviation = laplace.posterior_scale(hessian, prior_prec=prior_prec)

        samples = laplace.sample_from_normal(
            mean_parameter, standard_deviation, n_samples=posterior_samples
        )
        entropy_list = []
        for sample in samples:
            # for the sample, predict the logit distribution
            _, output_dict = self.logit_distribution_from_laplace_samples(
                sample, image=image
            )
            logit_distribution = output_dict["distribution"]

            logit_sample_list = []
            for _ in range(logit_samples):
                logit_sample_list.append(torch.sigmoid(logit_distribution.sample()))
            mean_sigmoid = torch.mean(
                torch.stack(logit_sample_list), dim=0
            )  # [bs, class_dim, img_h, img_w]

            entropy_list.append(pixelwise_entropy(mean_sigmoid))

        # reset the epistemic component to false
        self.epistemic_component = False

        # average the entropy maps over the sampled networks and return the average entropy map
        return torch.mean(torch.stack(entropy_list), dim=0)

    def epistemic_uncertainty(
        self,
        image,
        hessian,
    ):
        """
        Calculate the epistemic uncertainty for the given image.
        Corresponds to the epistemic component of formula (2) in the
        paper "Introducing an Improved Information-Theoretic Measure of Predictive Uncertainty", Hochreiter et al. (2023).
        """

        return self.total_uncertainty(image, hessian) - self.aleatoric_uncertainty(
            image, hessian
        )

    def epistemic_uncertainty_hochreiter(
        self, image, hessian, posterior_samples=200, prior_prec=1e6
    ):
        """
        Calculate the epistemic uncertainty for the given image.
        Corresponds to the epistemic component of formula (4) in the
        paper "Introducing an Improved Information-Theoretic Measure of Predictive Uncertainty", Hochreiter et al. (2023).
        """

        mean_parameter = parameters_to_vector(self.combined.parameters())
        laplace = DiagLaplace()

        # sample from the Laplace Approximation
        standard_deviation = laplace.posterior_scale(hessian, prior_prec=prior_prec)

        NN_samples_1 = laplace.sample_from_normal(
            mean_parameter,
            standard_deviation,
            n_samples=posterior_samples,
        )

        NN_samples_2 = laplace.sample_from_normal(
            mean_parameter,
            standard_deviation,
            n_samples=posterior_samples,
        )

        kl_sum = 0
        for sample_1 in NN_samples_1:
            for sample_2 in NN_samples_2:
                # predict the logit distribution for both NN samples
                _, output_dict_1 = self.logit_distribution_from_laplace_samples(
                    sample_1, image=image
                )
                _, output_dict_2 = self.logit_distribution_from_laplace_samples(
                    sample_2, image=image
                )
                logit_distribution_1 = output_dict_1["distribution"]
                logit_distribution_2 = output_dict_2["distribution"]

                # Define the Transformed Distributions with f being the sigmoid function
                logit_distribution_1 = td.TransformedDistribution(
                    logit_distribution_1.base_distribution, td.SigmoidTransform()
                )

                logit_distribution_2 = td.TransformedDistribution(
                    logit_distribution_2.base_distribution, td.SigmoidTransform()
                )

                # Calculate the KL Divergence between the two distributions
                kl_sum += td.kl_divergence(logit_distribution_1, logit_distribution_2)

        # reset the epistemic component to false
        self.epistemic_component = False

        return kl_sum / (posterior_samples**2)

    def epistemic_pixel_variance(
        self, image, hessian, posterior_samples=50, logit_samples=20, prior_prec=1e6
    ):
        """
        Calculate the epistemic uncertainty for the given image.
        This function returns the pixel wise entropy of the mean predictions given by NN samples from the Laplace Approximation.
        """
        mean_parameter = parameters_to_vector(self.combined.parameters())
        laplace = DiagLaplace()
        # sample from the Laplace Approximation
        standard_deviation = laplace.posterior_scale(hessian, prior_prec=prior_prec)

        samples = laplace.sample_from_normal(
            mean_parameter, standard_deviation, n_samples=posterior_samples
        )
        sigmoid_list = []
        for sample in samples:
            # for the sample, predict the logit mean
            logit_mean, _ = self.logit_distribution_from_laplace_samples(
                sample, image=image
            )
            # calculate the sigmoid of the logit mean and save the sigmoid map
            sigmoid_list.append(torch.sigmoid(logit_mean))

        # Calculate the pixel wise variance of the sigmoid maps
        sigmoid_map = torch.stack(sigmoid_list)
        sigmoid_variance = torch.var(sigmoid_map, dim=0)

        # reset the epistemic component to false
        self.epistemic_component = False

        return sigmoid_variance


"""
Using both metrics (variance and uncertainty (entropy)) together can provide a comprehensive understanding of your model's behavior:

High Variance & High Entropy: The model's predictions for certain pixels vary significantly and are generally uncertain. This might indicate areas of the image where the model struggles to make consistent and confident decisions.
High Variance & Low Entropy: The model is confident in its predictions, but those predictions vary significantly. This could occur in scenarios where the model is sensitive to small changes in input or in situations with ambiguous ground truth.
Low Variance & High Entropy: The model consistently predicts probabilities around 0.5 for certain pixels, indicating a consistent uncertainty. This might happen in areas of the image that are inherently ambiguous or challenging to segment.
Low Variance & Low Entropy: The model consistently makes confident predictions, which is the ideal scenario in many cases.
"""
