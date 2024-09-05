import torch
import torch.nn as nn
from src.models.unet import Unet
from src.utils.utils import pixelwise_entropy, logit_as_dirac_distribution
import torch.distributions as td


class Ensemble(nn.Module):
    def __init__(
        self, name, number_of_models, input_channels, num_classes, num_filters
    ):
        super().__init__()

        self.number_of_models = number_of_models
        self.name = name
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.models = nn.ModuleList(
            [
                Unet(
                    name=self.name,
                    input_channels=self.input_channels,
                    num_classes=self.num_classes,
                    num_filters=self.num_filters,
                )
                for _ in range(number_of_models)
            ]
        )

    def forward(self, image):
        logits = [model.forward(image) for model in self.models]
        return logits

    def total_uncertainty(self, image, logit_samples=20):
        """
        Calculate the total uncertainty for the given image.
        Corresponds to the total uncertainty component of formula (1) and (2) in the
        paper "Introducing an Improved Information-Theoretic Measure of Predictive Uncertainty", Hochreiter et al. (2023).
        """

        log_probs_list = []
        for model in self.models:
            # for the Ensemble member, predict the logit distribution
            logits = model.forward(image)
            logit_distribution = logit_as_dirac_distribution(logits)
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

    def aleatoric_uncertainty(self, image, logit_samples=20):
        """
        Calculate the aleatoric uncertainty for the given image.
        Corresponds to the aleatoric component of formula (2) and (4) in the
        paper "Introducing an Improved Information-Theoretic Measure of Predictive Uncertainty", Hochreiter et al. (2023).
        """

        entropy_list = []
        for model in self.models:
            # for the Ensemble member, predict the logit distribution
            logits = model.forward(image)

            logit_distribution = logit_as_dirac_distribution(logits)

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

    def epistemic_uncertainty_hochreiter(
        self,
        image,
    ):
        """
        Calculate the epistemic uncertainty for the given image.
        Corresponds to the epistemic component of formula (4) in the
        paper "Introducing an Improved Information-Theoretic Measure of Predictive Uncertainty", Hochreiter et al. (2023).
        """

        kl_sum = 0
        for model_1 in self.models:
            for model_2 in self.models:
                # predict the logit distribution for both NN samples
                logits_1 = model_1.forward(image)
                logits_2 = model_2.forward(image)

                logit_distribution_1 = logit_as_dirac_distribution(logits_1)
                logit_distribution_2 = logit_as_dirac_distribution(logits_2)

                # Define the Transformed Distributions with f being the sigmoid function
                logit_distribution_1 = td.TransformedDistribution(
                    logit_distribution_1.base_distribution, td.SigmoidTransform()
                )

                logit_distribution_2 = td.TransformedDistribution(
                    logit_distribution_2.base_distribution, td.SigmoidTransform()
                )

                # Calculate the KL Divergence between the two distributions
                kl_sum += td.kl_divergence(logit_distribution_1, logit_distribution_2)

        return kl_sum / (len(self.models) ** 2)

    def epistemic_pixel_variance(self, image):
        """
        Calculate the epistemic uncertainty for the given image.
        This function returns the pixel wise entropy of the mean predictions given by NN samples from the Laplace Approximation.
        """

        sigmoid_list = []
        for model in self.models:
            # for the sample, predict the logit mean
            logit_mean = model.forward(image)
            # calculate the sigmoid of the logit mean and save the sigmoid map
            sigmoid_list.append(torch.sigmoid(logit_mean))

        # Calculate the pixel wise variance of the sigmoid maps
        sigmoid_map = torch.stack(sigmoid_list)
        sigmoid_variance = torch.var(sigmoid_map, dim=0)

        return sigmoid_variance


if __name__ == "__main__":
    image = torch.rand(4, 3, 64, 64)

    unet = Ensemble(
        "test",
        number_of_models=1,
        input_channels=3,
        num_classes=1,
        num_filters=[8, 16, 32, 64, 128],
    )

    output = unet(image)

    print(f"Output is of type {type(output)} and has length {len(output)}")
    print(f"Output[0] is of type {type(output[0])} and has shape {output[0].shape}")

    # Count number of total parameters in the model and log
    pytorch_total_params = sum(p.numel() for p in unet.parameters())
    print("Total number of parameters: {}".format(pytorch_total_params))
    for p in unet.parameters():
        print(p.numel())