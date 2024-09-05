import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributions as td
from src.utils.utils import (
    pixelwise_entropy,
    ReshapedDistribution,
    logit_as_dirac_distribution,
)


class DropoutUnet(pl.LightningModule):
    def __init__(
        self,
        name,
        input_channels,
        num_classes,
        num_filters,
        apply_last_layer=True,
        padding=True,
        p=0,
        batch_norm=False,
    ):
        super().__init__()
        self.name = name
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.apply_last_layer = apply_last_layer

        self.contracting_path = nn.ModuleList()
        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True
            if i != len(self.num_filters) - 2:
                self.contracting_path.append(
                    DownConvBlock(
                        input, output, padding, pool=pool, p=p, batch_norm=batch_norm
                    )
                )
            else:
                self.contracting_path.append(
                    DownConvBlock(
                        input,
                        output,
                        padding,
                        pool=pool,
                        p=p,
                        batch_norm=batch_norm,
                        mcdropout=True,
                    )
                )

        self.upsampling_path = nn.ModuleList()
        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]
            self.upsampling_path.append(
                UpConvBlock(input, output, padding, p=p, batch_norm=batch_norm)
            )

        if self.apply_last_layer:
            last_layer = []
            last_layer.append(nn.Conv2d(output, 8, kernel_size=1))
            last_layer.append(nn.Tanh())
            last_layer.append(nn.Conv2d(8, 8, kernel_size=1))
            last_layer.append(nn.Tanh())
            last_layer.append(nn.Conv2d(8, num_classes, kernel_size=1))
            self.last_layer = nn.Sequential(*last_layer)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path) - 1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i - 1])

        del blocks

        if self.apply_last_layer:
            x = self.last_layer(x)

        return x

    def enable_dropout(self):
        """Call to enable the dropout layers during testtime"""
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                # print(m)
                m.train()

    def total_uncertainty(self, image, posterior_samples=50, logit_samples=20):
        """
        Calculate the total uncertainty for the given image.
        Corresponds to the total uncertainty component of formula (1) and (2) in the
        paper "Introducing an Improved Information-Theoretic Measure of Predictive Uncertainty", Hochreiter et al. (2023).
        """
        self.enable_dropout()
        log_probs_list = []
        for _ in range(posterior_samples):
            # for the sample, predict the mean
            logits = self.forward(image)

            # Model the Dirac logit distribution as a limit case of a normal normal distribution
            dirac_logit_distribution = logit_as_dirac_distribution(logits)
            # sample from the logit distribution
            collect_logit_probs_sample_list = []
            for _ in range(logit_samples):
                collect_logit_probs_sample_list.append(
                    torch.sigmoid(dirac_logit_distribution.sample())
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
            # for the sample, predict the mean
            logits = self.forward(image)

            # Model the Dirac logit distribution as a limit case of a normal norma
            dirac_logit_distribution = logit_as_dirac_distribution(logits)

            logit_sample_list = []
            for _ in range(logit_samples):
                logit_sample_list.append(
                    torch.sigmoid(dirac_logit_distribution.sample())
                )
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
                logits_1 = self.forward(image)
                logits_2 = self.forward(image)

                # Model the Dirac logit distribution as a limit case of a normal normal distribution
                dirac_logit_distribution_1 = logit_as_dirac_distribution(logits_1)
                dirac_logit_distribution_2 = logit_as_dirac_distribution(logits_2)

                # Define the Transformed Distributions with f being the sigmoid function
                logit_distribution_1 = td.TransformedDistribution(
                    dirac_logit_distribution_1.base_distribution, td.SigmoidTransform()
                )

                logit_distribution_2 = td.TransformedDistribution(
                    dirac_logit_distribution_2.base_distribution, td.SigmoidTransform()
                )

                # Calculate the KL Divergence between the two distributions
                kl_sum += td.kl_divergence(logit_distribution_1, logit_distribution_2)

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
            logit_mean = self.forward(image)
            # calculate the sigmoid of the logit mean and save the sigmoid map
            sigmoid_list.append(torch.sigmoid(logit_mean))

        # Calculate the pixel wise variance of the sigmoid maps
        sigmoid_map = torch.stack(sigmoid_list)
        sigmoid_variance = torch.var(sigmoid_map, dim=0)

        return sigmoid_variance


"""
Building Blocks for the U-Net
"""


class DownConvBlock(nn.Module):
    """
    A block of three convolutional layers where each layer is followed by a non-linear activation function
    Between each block we add a pooling operation.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        padding,
        pool=True,
        p=0,
        batch_norm=False,
        mcdropout=False,
    ):
        super(DownConvBlock, self).__init__()
        layers = []

        if pool:
            layers.append(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
            )

        if batch_norm:
            layers.append(nn.BatchNorm2d(input_dim))

        if mcdropout:
            layers.append(nn.Dropout2d(p=0.5))
        layers.append(
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)
            )
        )
        layers.append(nn.Tanh())
        if mcdropout:
            layers.append(nn.Dropout2d(p=0.5))
        layers.append(
            nn.Conv2d(
                output_dim, output_dim, kernel_size=3, stride=1, padding=int(padding)
            )
        )
        layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

    def forward(self, patch):
        return self.layers(patch)


class UpConvBlock(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """

    def __init__(
        self, input_dim, output_dim, padding, bilinear=True, p=0, batch_norm=False
    ):
        super(UpConvBlock, self).__init__()
        self.bilinear = bilinear

        if not self.bilinear:
            upconv_layer = []
            upconv_layer.append(
                nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2)
            )
            self.upconv_layer = nn.Sequential(*upconv_layer)

        self.conv_block = DownConvBlock(
            input_dim, output_dim, padding, pool=False, p=p, batch_norm=batch_norm
        )

    def forward(self, x, bridge):
        if self.bilinear:
            up = nn.functional.interpolate(
                x, mode="bilinear", scale_factor=2, align_corners=True
            )
            # up = nn.functional.interpolate(
            #    x, mode='nearest', scale_factor=2)
        else:
            up = self.upconv_layer(x)

        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out


if __name__ == "__main__":
    image = torch.rand(4, 3, 64, 64)

    unet = DropoutUnet(
        "test",
        input_channels=3,
        num_classes=1,
        num_filters=[8, 16, 32, 64, 128],
        apply_last_layer=True,
        padding=True,
        p=0,
        batch_norm=False,
    )

    output = unet(image)

    print(output.size())

    # Test total uncertainty function
    uc = unet.total_uncertainty(image, posterior_samples=5)
    print(uc)  # Count number of total parameters in the model and log
    pytorch_total_params = sum(p.numel() for p in unet.parameters())
    print("Total number of parameters: {}".format(pytorch_total_params))
