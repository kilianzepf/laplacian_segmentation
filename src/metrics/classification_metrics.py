"""
evaluate.py provides common evaluation metrics for the semantic segmentation task and 
uncertainty quantification statistics.
Semantic Segmentation Metrics:
    + pixel accuracy: pixel_acc
    + mean intersection of union: mIoU
    + F1 Score
Uncertainty Metrics:
    + Expected Calibration Error
    + Maximum Calibration Error
    + Reliability Diagrams
    + PAVPU
"""

import torch
import numpy as np


def pixel_accuracy(outputs: torch.tensor, targets: torch.tensor) -> torch.float32:
    """
    This function takes as an input a batch of
        + predictions
        + labels

    returns: pixel-wise accuracy for entire batch as a single float value.

    To develop the average pixel accuracy for all batches in the validation or test set, place the
    following code in the prediction loop:

    pixel_accuracy_list = []
    for i, (image, label) in enumerate(validation_loader):
        output = model(image)
        pixel_accuracy_list.append(pixel_accuracy(prediction, label))
    pixel_acc_array = torch.FloatTensor(np.array(pixel_accuracy_list))
    average_pixel_accuracy = torch.mean(pixel_acc_array)
    """
    predictions = outputs.clone()
    labels = targets.clone()

    assert predictions.size() == labels.size(), f"prediction and label size mismatch."
    assert (
        len(predictions.size()) == 4
    ), f"dimension mismatch. Expects: ([b, num_classes, h, w])"

    predictions += 1
    labels += 1

    num_pixel_labeled = torch.sum(labels > 0)
    num_pixel_correct = torch.sum((predictions == labels) * (labels > 0))

    pixel_accuracy = num_pixel_correct / num_pixel_labeled
    return pixel_accuracy.cpu()


def IoU(target, predicted_mask):
    """
    Args:
        target: (torch.tensor (batchxCxHxW)) Binary Target Segmentation from training set
        predicted_mask: (torch.tensor (batchxCxHxW)) Predicted Segmentation Mask

    Returns:
        IoU: (Float) Average IoUs over Batch
    """

    target = target.detach()
    predicted_mask = predicted_mask.detach()
    smooth = 1e-8
    true_p = (torch.logical_and(target == 1, predicted_mask == 1)).sum()
    # true_n = (torch.logical_and(target == 0, predicted_mask == 0)).sum().item() #Currently not needed for IoU
    false_p = (torch.logical_and(target == 0, predicted_mask == 1)).sum()
    false_n = (torch.logical_and(target == 1, predicted_mask == 0)).sum()
    sample_IoU = (smooth + float(true_p)) / (
        float(true_p) + float(false_p) + float(false_n) + smooth
    )

    return sample_IoU


def calc_f1(
    outputs: torch.tensor, targets: torch.tensor, num_classes: int = 2
) -> torch.float32:
    """
    Computes the f1 score
    """
    predictions = outputs.clone()
    labels = targets.clone()

    assert predictions.size() == labels.size()

    # convert from torch to numpy because torch.histogram is broken.
    # link: https://github.com/pytorch/pytorch/issues/74236
    predictions = predictions.numpy()
    labels = labels.numpy()

    predictions += 1
    labels += 1

    # Compute area intersection:
    predictions = predictions * (labels > 0)
    intersection = predictions * (predictions == labels)

    # Compute area union:
    area_intersection, _ = np.histogram(
        intersection, bins=num_classes, range=(1, num_classes)
    )
    area_pred, _ = np.histogram(predictions, bins=num_classes, range=(1, num_classes))
    area_lab, _ = np.histogram(labels, bins=num_classes, range=(1, num_classes))

    mask_sum = area_pred + area_lab
    f1 = 2 * area_intersection / mask_sum

    return f1


def iou(mask1, mask2):
    """
    Calculate the IoU between two masks.

    Args:
        mask1, mask2: Tensors of shape [batch_size, 1, height, width].

    Returns:
        A tensor of IoU scores with shape [batch_size].
    """
    smooth = 1e-8
    intersection = (
        torch.logical_and(mask1, mask2).float().sum(dim=(1, 2, 3))
    )  # Calculate intersection
    union = torch.logical_or(mask1, mask2).float().sum(dim=(1, 2, 3))  # Calculate union
    iou_scores = (intersection + smooth) / (union + smooth)  # IoU calculation
    return iou_scores


def generalized_energy_distance(list_of_predictions, list_of_targets):
    """
    Calculate the Generalised Energy Distance for multiple sets of predictions against multiple sets of ground truth masks.

    Args:
        list_of_predictions: List of tensors, each with shape [batch_size, 1, height, width], representing different sets of predictions for each image in the batch.
        list_of_targets: List of 6 tensors, each with shape [batch_size, 1, height, width], representing the 6 ground truth masks for each image in the batch.

    Returns:
        The GED score as a scalar.
    """
    batch_size = list_of_predictions[0].size(0)
    num_preds = len(list_of_predictions)
    num_gts = len(list_of_targets)

    # Calculate IoU for each prediction against each ground truth mask
    iou_scores = torch.zeros((num_preds, batch_size, num_gts))
    for i, preds in enumerate(list_of_predictions):
        for j, gts in enumerate(list_of_targets):
            iou_scores[i, :, j] = iou(preds, gts)

    # Calculate distances (1 - IoU)
    distances = 1 - iou_scores

    # Average distances across all predictions and ground truths
    gt_pred_dist = distances.mean()

    # Handle the case when there's only one set of ground truths or predictions
    if num_gts > 1:
        gt_gt_dists = [
            1 - iou(list_of_targets[i], list_of_targets[j]).mean()
            for i in range(num_gts)
            for j in range(i + 1, num_gts)
        ]
        gt_gt_dist = (
            torch.tensor(gt_gt_dists).mean() if gt_gt_dists else torch.tensor(0.0)
        )
    else:
        gt_gt_dist = torch.tensor(0.0)

    if num_preds > 1:
        pred_pred_dists = [
            1 - iou(list_of_predictions[i], list_of_predictions[j]).mean()
            for i in range(num_preds)
            for j in range(i + 1, num_preds)
        ]
        pred_pred_dist = (
            torch.tensor(pred_pred_dists).mean()
            if pred_pred_dists
            else torch.tensor(0.0)
        )
    else:
        pred_pred_dist = torch.tensor(0.0)

    # Compute GED
    ged = 2 * gt_pred_dist - gt_gt_dist - pred_pred_dist

    return ged.item()


def normalized_cross_correlation(map1, map2):
    """
    Compute the Normalized Cross-Correlation (NCC) for each pair of maps within a single batch.

    Args:
        map1 (torch.Tensor): A batch of the first uncertainty maps with shape [B, H, W].
        map2 (torch.Tensor): A batch of the second uncertainty maps with shape [B, H, W].

    Returns:
        torch.Tensor: A tensor containing the NCC values for each pair in the batch with shape [B].
    """
    # Ensure the maps are floats
    map1, map2 = map1.float(), map2.float()
    # Ensure that the maps are of shape [B, H, W]
    assert map1.dim() == 3 and map2.dim() == 3, "Input maps must be [B, H, W]"

    # Calculate mean and std dev for each map in the batch
    mean1 = map1.mean(dim=[1, 2], keepdim=True)
    std1 = map1.std(dim=[1, 2], keepdim=True) + 1e-8
    mean2 = map2.mean(dim=[1, 2], keepdim=True)
    std2 = map2.std(dim=[1, 2], keepdim=True) + 1e-8

    # Normalize the maps
    map1_normalized = (map1 - mean1) / std1
    map2_normalized = (map2 - mean2) / std2

    # Compute the element-wise product, sum over H and W dimensions for each map in the batch
    ncc = (map1_normalized * map2_normalized).sum(dim=[1, 2])

    # Normalize by the number of elements in each map to get the average
    ncc /= map1[0].numel()

    return ncc


import torch


import torch


def calculate_box_ratios(uncertainty_maps, binary_masks):
    """
    Calculate the ratios of the sum of uncertainty values within specified rectangles to the total uncertainty values
    for each pair in a batch of uncertainty maps and corresponding binary masks.

    Parameters:
    - uncertainty_maps (torch.Tensor): A tensor of shape [B, H, W] containing batches of uncertainty maps,
      where B is the batch size, H is the height, and W is the width. Each uncertainty map contains pixel-wise
      uncertainty values.

    - binary_masks (torch.Tensor): A tensor of shape [B, H, W] containing batches of binary masks corresponding
      to the uncertainty maps. Each binary mask has a rectangle (or multiple rectangles) marked with ones (1s),
      indicating regions of interest, and zeros (0s) elsewhere.

    Returns:
    - torch.Tensor: A tensor of shape [B] containing the ratio of the sum of uncertainty values within the rectangles
      defined by the binary masks to the total sum of uncertainty values in each map, for each pair in the batch.
      Each element in the tensor corresponds to the ratio for each pair of uncertainty map and binary mask.

    """

    # Ensure the binary masks and uncertainty maps have matching shapes
    assert (
        uncertainty_maps.shape == binary_masks.shape
    ), "Uncertainty maps and binary masks must have the same shapes"

    # Calculate the sum of pixel values in the rectangles defined by the binary masks
    rectangle_sums = torch.sum(uncertainty_maps * binary_masks, dim=[1, 2])

    # Calculate the total sum of pixel values in each uncertainty map
    total_sums = torch.sum(uncertainty_maps, dim=[1, 2])

    # Calculate the ratios by dividing the rectangle sums by the total sums
    ratios = rectangle_sums / total_sums

    return ratios  # Returns a tensor of ratios for each pair in the batch
