import nnj
import pytorch_lightning as pl
import torch
import torch.distributions as td
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from pytorch_laplace import BCEHessianCalculator
from pytorch_laplace.laplace.diag import DiagLaplace
from src.utils.utils import pixelwise_entropy, logit_as_dirac_distribution


class nnj_Unet(pl.LightningModule):
    def __init__(self, name, img_size=64, input_channels=3, num_classes=1):
        super().__init__()
        self.name = name
        self.stochastic_net = nnj.Sequential(
            nnj.Conv2d(input_channels, 8, 3, stride=1, padding=1),
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
            nnj.Conv2d(8, 8, 1),
            nnj.Tanh(),
            nnj.Conv2d(8, 8, 1),
            nnj.Tanh(),
            nnj.Conv2d(8, num_classes, 1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.stochastic_net(x)

    def total_uncertainty(
        self, image, posterior_samples=50, logit_samples=20, prior_prec=1e6
    ):
        """
        Calculate the total uncertainty for the given image.
        Corresponds to the total uncertainty component of formula (1) and (2) in the
        paper "Introducing an Improved Information-Theoretic Measure of Predictive Uncertainty", Hochreiter et al. (2023).
        """
        # Calculate the Laplace Approximation on the network
        h = BCEHessianCalculator(wrt="weight", shape="diagonal", speed="fast")
        hessian = h.compute_hessian(image, self.stochastic_net)
        mean_parameter = parameters_to_vector(self.stochastic_net.parameters())
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
            # Transfer the parameter vector into the model parameters
            vector_to_parameters(sample, self.stochastic_net.parameters())
            # for the sample, predict the logit distribution
            logits = self.stochastic_net(image)
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

    def aleatoric_uncertainty(
        self, image, posterior_samples=50, logit_samples=20, prior_prec=1e6
    ):
        """
        Calculate the aleatoric uncertainty for the given image.
        Corresponds to the aleatoric component of formula (2) and (4) in the
        paper "Introducing an Improved Information-Theoretic Measure of Predictive Uncertainty", Hochreiter et al. (2023).
        """

        # Calculate the Laplace Approximation on the network
        h = BCEHessianCalculator(wrt="weight", shape="diagonal", speed="fast")
        hessian = h.compute_hessian(image, self.stochastic_net)
        mean_parameter = parameters_to_vector(self.stochastic_net.parameters())
        laplace = DiagLaplace()

        # sample from the Laplace Approximation
        standard_deviation = laplace.posterior_scale(hessian, prior_prec=prior_prec)

        samples = laplace.sample_from_normal(
            mean_parameter, standard_deviation, n_samples=posterior_samples
        )
        entropy_list = []
        for sample in samples:
            # Transfer the parameter vector into the model parameters
            vector_to_parameters(sample, self.stochastic_net.parameters())
            # for the sample, predict the logit distribution
            logits = self.stochastic_net(image)
            logit_distribution = logit_as_dirac_distribution(logits)
            # sample from the logit distribution

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
        self, image, posterior_samples=50, prior_prec=1e6
    ):
        """
        Calculate the epistemic uncertainty for the given image.
        Corresponds to the epistemic component of formula (4) in the
        paper "Introducing an Improved Information-Theoretic Measure of Predictive Uncertainty", Hochreiter et al. (2023).
        """

        # Calculate the Laplace Approximation on the network

        h = BCEHessianCalculator(wrt="weight", shape="diagonal", speed="fast")
        hessian = h.compute_hessian(image, self.stochastic_net)
        mean_parameter = parameters_to_vector(self.stochastic_net.parameters())
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
                vector_to_parameters(sample_1, self.stochastic_net.parameters())
                # for the sample, predict the logit distribution
                logits_1 = self.stochastic_net(image)
                # Transfer the parameter vector into the model parameters
                vector_to_parameters(sample_2, self.stochastic_net.parameters())
                # for the sample, predict the logit distribution
                logits_2 = self.stochastic_net(image)
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

        return kl_sum / (posterior_samples**2)

    def epistemic_pixel_variance(
        self, image, posterior_samples=50, logit_samples=20, prior_prec=1e6
    ):
        """
        Calculate the epistemic uncertainty for the given image.
        This function returns the pixel wise entropy of the mean predictions given by NN samples from the Laplace Approximation.
        """
        # Calculate the Laplace Approximation on the network
        h = BCEHessianCalculator(wrt="weight", shape="diagonal", speed="fast")
        hessian = h.compute_hessian(image, self.stochastic_net)
        mean_parameter = parameters_to_vector(self.stochastic_net.parameters())
        laplace = DiagLaplace()
        # sample from the Laplace Approximation
        standard_deviation = laplace.posterior_scale(hessian, prior_prec=prior_prec)

        samples = laplace.sample_from_normal(
            mean_parameter, standard_deviation, n_samples=posterior_samples
        )
        sigmoid_list = []
        for sample in samples:
            # predict the logit distribution for both NN samples
            vector_to_parameters(sample, self.stochastic_net.parameters())
            # for the sample, predict the logit distribution
            logit_mean = self.stochastic_net(image)
            # calculate the sigmoid of the logit mean and save the sigmoid map
            sigmoid_list.append(torch.sigmoid(logit_mean))

        # Calculate the pixel wise variance of the sigmoid maps
        sigmoid_map = torch.stack(sigmoid_list)
        sigmoid_variance = torch.var(sigmoid_map, dim=0)

        return sigmoid_variance


if __name__ == "__main__":
    image = torch.rand(4, 3, 512, 512)

    unet = nnj_Unet("test", img_size=512)

    output = unet(image)

    print(output.size())

    # Count number of total parameters in the model and log
    pytorch_total_params = sum(p.numel() for p in unet.parameters())
    print("Total number of parameters: {}".format(pytorch_total_params))
    for p in unet.parameters():
        print(p.numel())
