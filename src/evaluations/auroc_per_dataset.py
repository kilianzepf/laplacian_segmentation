import sys
import os
import yaml
import torch
import glob
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from yaml.loader import SafeLoader
from roc_curves import patch_based_aggregation, aggregate_scores, plot_roc_curves
from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd


def compute_auroc_values(id_scores, ood_scores, ood_types):
    # This function encapsulates the AUROC computation logic from the original script
    # Initialize an empty dictionary to store AUROC values
    auroc_values = {}

    # Perform aggregations for mean and patch-based strategies
    for aggregation in ["mean", "patch"]:
        scores_to_plot = aggregate_scores(id_scores, ood_scores, ood_types, aggregation)

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

        auroc_values[aggregation] = auroc_scores

    return auroc_values


def main(dataset_name):
    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "configs_kilian",
        dataset_name,
        "evaluation",
    )
    base_dir = os.path.abspath(base_dir)
    config_files = glob.glob(os.path.join(base_dir, "*.yaml"))

    # Initialize a list to store data for DataFrame
    data_for_df = []

    for config_file in tqdm(config_files, desc="Processing Configurations"):
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=SafeLoader)

        id_scores_path = f"results/uq_metrics/{config['dataset']}/{config['model_folder_name']}/id/{config['run_name']}"
        ood_types = (
            ["random_noise", "random_ghosting", "random_spike", "random_motion"]
            if config["dataset"] == "prostate"
            else ["derm", "clin", "padufes"]
        )

        try:
            id_scores = torch.load(id_scores_path, map_location=torch.device("cpu"))
            ood_scores = {
                ood_type: torch.load(
                    f"results/uq_metrics/{config['dataset']}/{config['model_folder_name']}/{ood_type}/{config['run_name']}",
                    map_location=torch.device("cpu"),
                )
                for ood_type in ood_types
            }

            auroc_values = compute_auroc_values(id_scores, ood_scores, ood_types)

            # Append AUROC values to the list for DataFrame
            for agg_type, scores in auroc_values.items():
                for score_name, auroc in scores.items():
                    data_for_df.append(
                        {
                            "Configuration": os.path.basename(config_file),
                            "Aggregation": agg_type,
                            "Score Name": score_name,
                            "AUROC": auroc,
                        }
                    )

        except FileNotFoundError as e:
            print(f"File not found for configuration: {config_file}")
            print(e)

    # Create a DataFrame
    df = pd.DataFrame(data_for_df)

    # Save the DataFrame to a CSV file
    output_file = f"results/auroc_values_{dataset_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved AUROC values to {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compute AUROC values for all models and configurations for a given dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'prostate', 'isic', 'brats')",
    )
    args = parser.parse_args()

    main(args.dataset)
