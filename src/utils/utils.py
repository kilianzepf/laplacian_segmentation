import torchvision
import torch.nn.functional as F
import torch.distributions as td
import torch.nn as nn
import torch
import math
import numpy as np
import os
from typing import Tuple
import matplotlib.pyplot as plt
import torch


def logit_as_dirac_distribution(logits, std=1e-10):
    """
    Args:
        logits: (torch.tensor BxCxHxW) Tensor contains the logits
        std: (float) standard deviation of the normal distribution
    Returns:
        dirac_logit_distribution: (ReshapedDistribution) Dirac logit distribution
    """
    # Model the Dirac logit distribution as a limit case of a normal normal distribution
    batch_size = logits.shape[0]  # Get the batchsize
    event_shape = (logits.size(1),) + logits.shape[2:]
    logits = logits.view((batch_size, -1))
    dirac_logit_distribution = td.Independent(td.Normal(logits, 1e-10), 1)
    dirac_logit_distribution = ReshapedDistribution(
        base_distribution=dirac_logit_distribution,
        new_event_shape=event_shape,
        validate_args=False,
    )
    return dirac_logit_distribution


def pixelwise_entropy(normalized_tensor, batched=True):
    """
    Args:
        normalized_tensor: (torch.tensor CxHxW) Tensor contains the normalized/sigmoid tensor (or a mean of sigmoid tensors)
    """
    # Calculating pixel-wise entropy
    epsilon = 1e-9  # small value to avoid log(0)
    entropy = -(normalized_tensor * torch.log(normalized_tensor + epsilon))
    if batched:
        entropy = entropy.sum(dim=1, keepdim=True)  # Sum over the channel dimension
    else:
        entropy = entropy.sum(dim=0, keepdim=True)
    return entropy


def add_white_noise_box(bild, scale=2, pos=[10, 20]):
    copy_bild = bild.clone()
    img_size = bild.shape[2:]  # Assuming [channels, height, width]

    # Calculate box coordinates
    box_coordinates = [
        pos[0],
        pos[0] + int(np.floor(scale * 10)),
        pos[1],
        pos[1] + int(np.floor(scale * 10)),
    ]

    # Ensure box coordinates do not exceed image dimensions
    box_coordinates[1] = min(box_coordinates[1], img_size[0])
    box_coordinates[3] = min(box_coordinates[3], img_size[1])

    # Create a white noise box
    box = torch.rand_like(
        copy_bild[
            :,
            :,
            : box_coordinates[1] - box_coordinates[0],
            : box_coordinates[3] - box_coordinates[2],
        ]
    )
    box = (box - box.min()) / (box.max() - box.min())

    # Set the specified region to white noise in the image
    copy_bild[
        :,
        :,
        box_coordinates[0] : box_coordinates[1],
        box_coordinates[2] : box_coordinates[3],
    ] = box

    # Initialize a binary tensor with the same shape as the image but with 1 channel
    binary_tensor = torch.zeros((1, 1) + img_size, dtype=torch.float32)

    # Mark the region of the white noise box with 1s in the binary tensor
    binary_tensor[
        :,
        :,
        box_coordinates[0] : box_coordinates[1],
        box_coordinates[2] : box_coordinates[3],
    ] = 1

    return copy_bild, binary_tensor


# Example usage
# Assume `bild` is a tensor of shape [1, 3, 64, 64] representing your image
# bild = torch.rand(1, 3, 64, 64)  # Example image tensor

# modified_image, binary_mask = add_white_noise_box(bild, scale=2, pos=[10, 20])


def add_black_box(bild, scale=2, pos=[10, 20]):
    copy_bild = bild.clone()
    # Define the size of the image
    img_size = bild.shape[2:]  # Assuming [channels, height, width]

    # Calculate box coordinates
    box_coordinates = [
        pos[0],
        pos[0] + int(np.floor(scale * 10)),
        pos[1],
        pos[1] + int(np.floor(scale * 10)),
    ]

    # Ensure box coordinates do not exceed image dimensions
    box_coordinates[1] = min(box_coordinates[1], img_size[0])
    box_coordinates[3] = min(box_coordinates[3], img_size[1])

    # Create a black box
    box = torch.zeros_like(copy_bild)

    # Set the specified region to black in the image
    copy_bild[
        :,
        :,
        box_coordinates[0] : box_coordinates[1],
        box_coordinates[2] : box_coordinates[3],
    ] = box[
        :,
        :,
        box_coordinates[0] : box_coordinates[1],
        box_coordinates[2] : box_coordinates[3],
    ]

    # Initialize a binary tensor with the same shape as the image but with 1 channel
    binary_tensor = torch.zeros((1, 1) + img_size, dtype=torch.float32)

    # Mark the region of the black box with 1s in the binary tensor
    binary_tensor[
        :,
        :,
        box_coordinates[0] : box_coordinates[1],
        box_coordinates[2] : box_coordinates[3],
    ] = 1

    return copy_bild, binary_tensor


# Example usage
# Assume `bild` is a tensor of shape [1, 3, 64, 64] representing your image
# bild = torch.rand(1, 3, 64, 64)  # Example image tensor

# modified_image, binary_mask = add_black_box(bild, scale=2, pos=[10, 20])


def iou_box_and_variance_map(variance_map, box_coordinates=[10, 30, 20, 40], reach=0):
    """
    Args:
        variance_map: (torch.tensor CxHxW) Tensor contains the variance map
        box_coordinates: (list) list of coordinates of the box
    Returns:
        iou: (float) Intersection over union of the box and the variance map

    """
    variance_map_box = variance_map[
        :,
        box_coordinates[0] - reach : box_coordinates[1] + reach,
        box_coordinates[2] - reach : box_coordinates[3] + reach,
    ]
    iou = torch.sum(variance_map_box) / torch.sum(variance_map)
    return iou


def make_image_grid(images, masks, predictions, required_padding):
    """
    Args
        X_batch: (torch.tensor BxCxHxW) Tensor contains the input images
        target_batch: (torch.tensor BxCxHxW) Tensor contains the target segmentations
        pred_batch: (torch.tensor BxCxHxW) Tensor contains the predictions

    Returns:
        grid: grid object to be plotted in wandb
    """
    if images.shape[1] > 3:
        # BraTS specific settings
        images = images[:, :3, :, :]
        masks = masks * 255
        predictions = predictions * 255

    grid_img = torchvision.utils.make_grid(images, len(images))
    grid_target = torchvision.utils.make_grid(
        F.pad(masks, required_padding, "constant", 0), len(masks)
    )
    grid_pred = torchvision.utils.make_grid(
        F.pad(predictions, required_padding, "constant", 0), len(predictions)
    )

    grid = torch.stack([grid_img, grid_target, grid_pred])
    grid = torchvision.utils.make_grid(grid, 1)

    return grid


"""
SSN Implementation https://github.com/biomedia-mira/stochastic_segmentation_networks/blob/master/ssn/
"""


class SSNCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
    ):
        super().__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, logits: torch.tensor, target: torch.tensor, **kwargs):
        return super().forward(logits, target)


class StochasticSegmentationNetworkLossMCIntegral(nn.Module):
    def __init__(self, num_mc_samples: int = 1, pos_weight=1.0):
        super().__init__()
        self.num_mc_samples = num_mc_samples
        self.pos_weight = torch.tensor(pos_weight).detach()

    @staticmethod
    def fixed_re_parametrization_trick(dist, num_samples):
        assert num_samples % 2 == 0
        samples = dist.rsample((num_samples // 2,))
        mean = dist.mean.unsqueeze(0)
        samples = samples - mean
        return torch.cat([samples, -samples]) + mean

    def forward(self, logits, target, distribution, **kwargs):
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        logit_sample = self.fixed_re_parametrization_trick(
            distribution, self.num_mc_samples
        )

        target = target.expand((self.num_mc_samples,) + target.shape)
        flat_size = self.num_mc_samples * batch_size
        logit_sample = logit_sample.view((flat_size, num_classes, -1))
        target = target.reshape((flat_size, num_classes, -1))

        log_prob = -F.binary_cross_entropy_with_logits(
            logit_sample,
            target,
            reduction="none",
            pos_weight=self.pos_weight,
        ).view((self.num_mc_samples, batch_size, -1))

        loglikelihood = torch.mean(
            torch.logsumexp(torch.sum(log_prob, dim=-1), dim=0)
            - math.log(self.num_mc_samples)
        )
        loss = -loglikelihood

        return loss


class ReshapedDistribution(td.Distribution):
    def __init__(
        self,
        base_distribution: td.Distribution,
        new_event_shape: Tuple[int, ...],
        validate_args=None,
    ):
        super().__init__(
            batch_shape=base_distribution.batch_shape,
            event_shape=new_event_shape,
            validate_args=validate_args,
        )
        self.base_distribution = base_distribution
        self.new_shape = base_distribution.batch_shape + new_event_shape

    @property
    def support(self):
        return self.base_distribution.support

    @property
    def arg_constraints(self):
        return self.base_distribution.arg_constraints()

    @property
    def mean(self):
        return self.base_distribution.mean.view(self.new_shape)

    @property
    def variance(self):
        return self.base_distribution.variance.view(self.new_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_distribution.rsample(sample_shape).view(
            sample_shape + self.new_shape
        )

    def log_prob(self, value):
        return self.base_distribution.log_prob(value.view(self.batch_shape + (-1,)))

    def entropy(self):
        return self.base_distribution.entropy()


def generate_visualization_pdf(
    test_set,
    inference_tensor,
    aleatoric_tensor,
    epistemic_tensor,
    total_tensor,
    epistemic_variance_tensor,
    file_path,
):
    """
    Generate a PDF containing visualizations of test images, uncertainty maps, and entropy.

    Args:
        test_set: The test dataset containing the images and ground truths.
        inference_tensor: Tensor containing inference results.
        aleatoric_tensor: Tensor containing aleatoric uncertainty maps.
        epistemic_tensor: Tensor containing epistemic uncertainty maps.
        total_tensor: Tensor containing total uncertainty maps.
        epistemic_variance_tensor: Tensor containing epistemic variance maps.
        file_path: Path to save the PDF file.
    """

    # Create a figure with subplots
    fig, axes = plt.subplots(
        nrows=10, ncols=8, figsize=(40, 60)
    )  # Adjust figsize as needed

    for i in range(10):
        # Get the i-th image from the test set
        image, _, ground_truth_list = test_set[i]
        image = image.squeeze()

        # Plot the test image
        axes[i, 0].imshow(image.cpu().numpy(), cmap="gray")
        axes[i, 0].set_title(f"Test Image {i+1}")
        axes[i, 0].axis("off")

        # Calculate and plot the entropy from the mean of the 6 ground truth masks
        gt_stack = torch.stack(ground_truth_list, dim=0)
        mean_gt_image = torch.mean(gt_stack, dim=0).squeeze()
        entropy_image = pixelwise_entropy(mean_gt_image.unsqueeze(0), batched=False)
        entropy_im = axes[i, 1].imshow(
            entropy_image.squeeze().cpu().numpy(),
            cmap="hot",
            vmin=0,
            vmax=entropy_image.max().item(),
        )
        axes[i, 1].set_title(f"GTs Entropy {i+1}")
        axes[i, 1].axis("off")
        fig.colorbar(entropy_im, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Get and plot the aleatoric uncertainty map
        aleatoric_map = aleatoric_tensor[i].squeeze()
        aleatoric_im = axes[i, 2].imshow(
            aleatoric_map.cpu().numpy(),
            cmap="hot",
            vmin=0,
            vmax=entropy_image.max().item(),
        )
        axes[i, 2].set_title(f"Aleatoric {i+1}")
        axes[i, 2].axis("off")
        fig.colorbar(aleatoric_im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # Get and plot the epistemic uncertainty map
        epistemic_map = epistemic_tensor[i].squeeze()
        epistemic_im = axes[i, 3].imshow(epistemic_map.cpu().numpy(), cmap="hot")
        axes[i, 3].set_title(f"Epistemic (MI) {i+1}")
        axes[i, 3].axis("off")
        fig.colorbar(epistemic_im, ax=axes[i, 3], fraction=0.046, pad=0.04)

        # Get and plot the total uncertainty map
        total_map = total_tensor[i].squeeze()
        total_im = axes[i, 4].imshow(
            total_map.cpu().numpy(), cmap="hot", vmin=0, vmax=total_map.max().item()
        )
        axes[i, 4].set_title(f"Total {i+1}")
        axes[i, 4].axis("off")
        fig.colorbar(total_im, ax=axes[i, 4], fraction=0.046, pad=0.04)

        # Plot the epistemic variance map
        epistemic_variance_map = epistemic_variance_tensor[i].squeeze()
        epistemic_var_im = axes[i, 5].imshow(
            epistemic_variance_map.cpu().numpy(),
            cmap="hot",
            vmin=0,
            vmax=epistemic_variance_map.max().item(),
        )
        axes[i, 5].set_title(f"Epistemic Pixel Variance {i+1}")
        axes[i, 5].axis("off")
        fig.colorbar(epistemic_var_im, ax=axes[i, 5], fraction=0.046, pad=0.04)

        # Get the i-th inference result and plot it
        inference_result = inference_tensor[i].squeeze()
        inference_result_im = axes[i, 6].imshow(
            inference_result.cpu().numpy(), cmap="gray"
        )
        axes[i, 6].set_title(f"Mean Sigmoid Prediction {i+1}")
        axes[i, 6].axis("off")
        fig.colorbar(inference_result_im, ax=axes[i, 6], fraction=0.046, pad=0.04)

        # Calculate and plot the entropy of the i-th inference result
        entropy_inference = pixelwise_entropy(
            inference_result.unsqueeze(0), batched=False
        )
        entropy_inf_im = axes[i, 7].imshow(
            entropy_inference.squeeze().cpu().numpy(),
            cmap="hot",
            vmin=0,
            vmax=entropy_inference.max().item(),
        )
        axes[i, 7].set_title(f"Mean Sigmoid Entropy {i+1}")
        axes[i, 7].axis("off")
        fig.colorbar(entropy_inf_im, ax=axes[i, 7], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the figure as a PDF
    plt.savefig(file_path, format="svg")

    # Close the figure to free memory
    plt.close(fig)


if __name__ == "__main__":
    # unittest.main()
    pass
