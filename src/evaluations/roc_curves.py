import sys
import os
import yaml
import torch
from argparse import ArgumentParser
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from yaml.loader import SafeLoader

# Determine the base directory of your project
current_script_path = os.path.abspath(__file__)
base_dir = os.path.join(os.path.dirname(current_script_path), "..", "..")
base_dir = os.path.abspath(base_dir)

# Add the base directory to sys.path
if base_dir not in sys.path:
    sys.path.append(base_dir)


def patch_based_aggregation(tensor, patch_size=10):
    N, C, H, W = tensor.size()
    max_uncertainty = torch.zeros(N, device=tensor.device)

    for i in range(H - patch_size + 1):
        for j in range(W - patch_size + 1):
            patch = tensor[:, :, i : i + patch_size, j : j + patch_size]
            patch_sum = patch.sum(dim=(-2, -1))  # Sum over the patch
            max_uncertainty, _ = torch.max(
                patch_sum, dim=1, keepdim=True
            )  # Max over patches

    return max_uncertainty.squeeze()


def aggregate_scores(id_scores, ood_scores, ood_types, aggregation="mean"):
    scores_to_plot = {}

    for score_name, score_tensor in id_scores.items():
        if score_name == "sigmoid":  # Skip the "sigmoid" key
            continue

        if score_tensor.dim() == 4:  # For scores with dimensions [42, 1, 64, 64]
            if aggregation == "mean":
                scores_to_plot[score_name] = score_tensor.mean(dim=(-2, -1)).squeeze()
            else:  # Patch-based aggregation
                scores_to_plot[score_name] = patch_based_aggregation(score_tensor)
        else:  # For epistemic_hochreiter with dimension [42]
            scores_to_plot[score_name] = score_tensor  # No aggregation needed

        all_ood_scores = []
        for ood_type in ood_types:
            ood_score_tensor = ood_scores[ood_type][score_name]
            if ood_score_tensor.dim() == 4:
                if aggregation == "mean":
                    aggregated_ood_score = ood_score_tensor.mean(dim=(-2, -1)).squeeze()
                else:
                    aggregated_ood_score = patch_based_aggregation(ood_score_tensor)
            else:
                aggregated_ood_score = ood_score_tensor

            all_ood_scores.append(aggregated_ood_score)

        scores_to_plot[score_name] = torch.cat(
            [scores_to_plot[score_name], torch.cat(all_ood_scores)]
        )

    return scores_to_plot


def plot_roc_curves(scores_to_plot, id_scores, aggregation, subplot_index):
    plt.subplot(1, 2, subplot_index)
    auroc_scores = {}

    for score_name, scores in scores_to_plot.items():
        labels = torch.cat(
            [
                torch.zeros(id_scores[score_name].size(0)),
                torch.ones(scores.size(0) - id_scores[score_name].size(0)),
            ]
        )
        fpr, tpr, thresholds = roc_curve(labels.numpy(), scores.numpy())
        roc_auc = auc(fpr, tpr)
        auroc_scores[score_name] = roc_auc

    # Sort the scores in descending order
    sorted_scores = sorted(auroc_scores.items(), key=lambda x: x[1], reverse=True)

    for score_name, _ in sorted_scores:
        scores = scores_to_plot[score_name]
        labels = torch.cat(
            [
                torch.zeros(id_scores[score_name].size(0)),
                torch.ones(scores.size(0) - id_scores[score_name].size(0)),
            ]
        )
        fpr, tpr, _ = roc_curve(labels.numpy(), scores.numpy())
        plt.plot(
            fpr, tpr, lw=2, label=f"{score_name} (AUC = {auroc_scores[score_name]:.2f})"
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {aggregation.capitalize()}")
    plt.legend(
        loc="lower right", title="Metric (AUC)", title_fontsize="13", fontsize="11"
    )


def main(hparams):
    config_path = hparams.config
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)

    try:
        id_scores = torch.load(
            f"results/uq_metrics/{config['dataset']}/{config['model_folder_name']}/id/{config['run_name']}",
            map_location=torch.device("cpu"),
        )
        ood_types = (
            ["random_noise", "random_ghosting", "random_spike", "random_motion"]
            if config["dataset"] == "prostate"
            else ["derm", "clin", "padufes"]
        )
        ood_scores = {
            ood_type: torch.load(
                f"results/uq_metrics/{config['dataset']}/{config['model_folder_name']}/{ood_type}/{config['run_name']}",
                map_location=torch.device("cpu"),
            )
            for ood_type in ood_types
        }

        # Aggregation for Mean
        scores_to_plot_mean = aggregate_scores(
            id_scores, ood_scores, ood_types, aggregation="mean"
        )

        # Aggregation for Sum
        scores_to_plot_sum = aggregate_scores(
            id_scores, ood_scores, ood_types, aggregation="sum"
        )

    except FileNotFoundError:
        print("Please run the evaluation script for this model and dataset first!")
        return

    plt.figure(figsize=(20, 8))

    # Plot ROC curves for Mean aggregation
    plot_roc_curves(scores_to_plot_mean, id_scores, "mean", 1)

    # Plot ROC curves for Sum aggregation
    plot_roc_curves(scores_to_plot_sum, id_scores, "patch", 2)

    file_path = f"results/roc_curves/{config['dataset']}/{config['model_folder_name']}/{config['run_name']}_roc_curve_comparison_mean_sum.pdf"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, format="pdf")
    plt.close()
    print("PDF saved with both Mean and Sum aggregation strategies!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
