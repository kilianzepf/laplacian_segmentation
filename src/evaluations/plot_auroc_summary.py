import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_auroc_means(datasets):
    # Create a 1x4 grid of subplots
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # Adjust the figure size as needed

    base_x_position = 0  # Starting position for the first dataset
    space_between_groups = 4  # Space between groups for clarity
    model_colors = {}  # To store colors for each model
    measure_colors = {}  # To store colors for each uncertainty measure
    aggregation_colors = {}  # To store colors for each aggregation strategy
    dataset_ticks = []  # To store the x positions for dataset labels
    dataset_labels = []  # To store the dataset names for labels

    """
    First Plot - Mean AUROC for Epistemic Hochreiter Uncertainty
    """
    for i, dataset_name in enumerate(datasets):
        csv_file = os.path.join("results", f"auroc_values_{dataset_name}.csv")
        df = pd.read_csv(csv_file)

        filtered_df = df[
            (df["Score Name"] == "epistemic_hochreiter")
            & (df["Aggregation"].isin(["mean", "patch"]))
        ]

        mean_aurocs = filtered_df.groupby("Configuration")["AUROC"].mean().reset_index()

        # Ensure there are at least three models to swap
        if len(mean_aurocs) >= 3:
            # Store the last model temporarily
            last_model = mean_aurocs.iloc[-1].copy()

            # Swap the last model with the third last model
            mean_aurocs.iloc[-1] = mean_aurocs.iloc[-3]

            # Place the originally last model into the third last position
            mean_aurocs.iloc[-3] = last_model

        # Assign x positions for this dataset's models
        x_positions = np.arange(len(mean_aurocs)) + base_x_position

        # Scatter plot for each model with a unique color
        skip_colors = [3, 7, 11, 15, 19]
        color_index = 0
        for idx, row in mean_aurocs.iterrows():
            if color_index in skip_colors:
                color_index += 1

            model_name = row["Configuration"]
            if model_name not in model_colors:
                model_colors[model_name] = plt.cm.tab20c(
                    color_index % 20
                )  # Assign a color from a colormap
                color_index += 1
            axs[0].errorbar(
                x_positions[idx],
                row["AUROC"],
                yerr=0.1,
                fmt="o",  # Marker type, you can use 'o' for circle, '^' for triangle, etc.
                label=model_name if dataset_name == datasets[0] else "",
                color=model_colors[model_name],
                capsize=0,  # Size of the horizontal lines at the top and bottom of the error bar
                markersize=10,  # Marker size
            )

        # Update the dataset_ticks and dataset_labels lists
        dataset_ticks.append(np.mean(x_positions))
        dataset_labels.append(dataset_name.capitalize())

        # Update base_x_position for the next dataset, including space
        base_x_position = x_positions[-1] + space_between_groups + 1

    # Set the x-ticks to the center of each dataset group and label them with the dataset names
    axs[0].set_xticks(dataset_ticks)
    axs[0].set_xticklabels(dataset_labels)

    # Configure the plot
    axs[0].set_ylabel("Mean AUROC")
    axs[0].set_title("Epistemic Hochreiter Uncertainty across Models")
    axs[0].legend(
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        frameon=False,
        fancybox=False,
        shadow=False,
        ncol=3,
        fontsize="x-small",
    )
    axs[0].grid(True, axis="y", linestyle="--", alpha=0.7)
    axs[0].set_ylim(0.0, 1.0)

    # Reset the base_x_position for the next plot
    dataset_ticks = []  # To store the x positions for dataset labels
    dataset_labels = []  # To store the dataset names for labels
    """
    Second Plot - Mean AUROC for Epistemic Uncertainty
    """
    for i, dataset_name in enumerate(datasets):
        csv_file = os.path.join("results", f"auroc_values_{dataset_name}.csv")
        df = pd.read_csv(csv_file)

        filtered_df = df[
            (df["Score Name"] == "epistemic")
            & (df["Aggregation"].isin(["mean", "patch"]))
        ]

        mean_aurocs = filtered_df.groupby("Configuration")["AUROC"].mean().reset_index()

        # Ensure there are at least three models to swap
        if len(mean_aurocs) >= 3:
            # Store the last model temporarily
            last_model = mean_aurocs.iloc[-1].copy()

            # Swap the last model with the third last model
            mean_aurocs.iloc[-1] = mean_aurocs.iloc[-3]

            # Place the originally last model into the third last position
            mean_aurocs.iloc[-3] = last_model

        # Assign x positions for this dataset's models
        x_positions = np.arange(len(mean_aurocs)) + base_x_position

        # Scatter plot for each model with a unique color
        for idx, row in mean_aurocs.iterrows():
            model_name = row["Configuration"]
            if model_name not in model_colors:
                model_colors[model_name] = plt.cm.tab20(
                    idx % 20
                )  # Assign a color from a colormap

            axs[1].scatter(
                x_positions[idx],
                row["AUROC"],
                label=model_name if dataset_name == datasets[0] else "",
                color=model_colors[model_name],
                s=100,
            )

        # Update the dataset_ticks and dataset_labels lists
        dataset_ticks.append(np.mean(x_positions))
        dataset_labels.append(dataset_name.capitalize())

        # Update base_x_position for the next dataset, including space
        base_x_position = x_positions[-1] + space_between_groups + 1

    # Set the x-ticks to the center of each dataset group and label them with the dataset names
    axs[1].set_xticks(dataset_ticks)
    axs[1].set_xticklabels(dataset_labels)

    # Configure the plot
    axs[1].set_ylabel("Mean AUROC")
    axs[1].set_title("Epistemic Uncertainty across Models")
    axs[1].legend(
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        frameon=False,
        fancybox=False,
        shadow=False,
        ncol=3,
        fontsize="x-small",
    )
    axs[1].grid(True, axis="y", linestyle="--", alpha=0.7)
    axs[1].set_ylim(0.0, 1.0)

    # Reset the base_x_position for the next plot
    dataset_ticks = []  # To store the x positions for dataset labels
    dataset_labels = []  # To store the dataset names for labels
    """
    Third Plot - Mean AUROC for different uncertainty measures across models
    """
    for i, dataset_name in enumerate(datasets):
        csv_file = os.path.join("results", f"auroc_values_{dataset_name}.csv")
        df = pd.read_csv(csv_file)

        filtered_df = df

        mean_aurocs = filtered_df.groupby("Score Name")["AUROC"].mean().reset_index()

        # Assign x positions for this dataset's models
        x_positions = np.arange(len(mean_aurocs)) + base_x_position

        # Scatter plot for each uncertainty measure with a unique color
        for idx, row in mean_aurocs.iterrows():
            measure_name = row["Score Name"]
            if measure_name not in measure_colors:
                measure_colors[measure_name] = plt.cm.tab10(idx % 20)
            axs[2].scatter(
                x_positions[idx],
                row["AUROC"],
                label=measure_name if dataset_name == datasets[0] else "",
                color=measure_colors[measure_name],
                s=100,
            )

        # Update the dataset_ticks and dataset_labels lists
        dataset_ticks.append(np.mean(x_positions))
        dataset_labels.append(dataset_name.capitalize())

        # Update base_x_position for the next dataset, including space
        base_x_position = x_positions[-1] + space_between_groups + 1

    # Set the x-ticks to the center of each dataset group and label them with the dataset names
    axs[2].set_xticks(dataset_ticks)
    axs[2].set_xticklabels(dataset_labels)

    # Configure the plot
    axs[2].set_ylabel("Mean AUROC")
    axs[2].set_title("AUROC across Uncertainty Measures")
    axs[2].legend(
        title="Uncertainty Measure",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        frameon=False,
        fancybox=False,
        shadow=False,
        ncol=3,
        fontsize="x-small",
    )
    axs[2].grid(True, axis="y", linestyle="--", alpha=0.7)
    axs[2].set_ylim(0.0, 1.0)

    # Reset the base_x_position for the next plot
    dataset_ticks = []  # To store the x positions for dataset labels
    dataset_labels = []  # To store the dataset names for labels

    """
    Fourth Plot - Mean AUROC for different aggregation strategies 
    """
    for i, dataset_name in enumerate(datasets):
        csv_file = os.path.join("results", f"auroc_values_{dataset_name}.csv")
        df = pd.read_csv(csv_file)

        filtered_df = df

        mean_aurocs = filtered_df.groupby("Aggregation")["AUROC"].mean().reset_index()

        # Assign x positions for this dataset's models
        x_positions = np.arange(len(mean_aurocs)) + base_x_position

        # Scatter plot for each aggregation strategy with a unique color
        for idx, row in mean_aurocs.iterrows():
            aggregation_name = row["Aggregation"]
            if aggregation_name not in aggregation_colors:
                aggregation_colors[aggregation_name] = plt.cm.tab10(idx % 20)
            axs[3].scatter(
                x_positions[idx],
                row["AUROC"],
                label=aggregation_name if dataset_name == datasets[0] else "",
                color=aggregation_colors[aggregation_name],
                s=100,
            )

        # Update the dataset_ticks and dataset_labels lists
        dataset_ticks.append(np.mean(x_positions))
        dataset_labels.append(dataset_name.capitalize())

        # Update base_x_position for the next dataset, including space
        base_x_position = x_positions[-1] + space_between_groups + 1

    # Set the x-ticks to the center of each dataset group and label them with the dataset names
    axs[3].set_xticks(dataset_ticks)
    axs[3].set_xticklabels(dataset_labels)

    # Configure the plot
    axs[3].set_ylabel("Mean AUROC")
    axs[3].set_title("AUROC across Aggregation Strategies")
    axs[3].legend(
        title="Aggregation Strategy",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        frameon=False,
        fancybox=False,
        shadow=False,
        ncol=3,
        fontsize="x-small",
    )
    axs[3].grid(True, axis="y", linestyle="--", alpha=0.7)
    axs[3].set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig("results/combined_mean_auroc_plot_with_datasets_as_xlabels.png")
    plt.show()


if __name__ == "__main__":
    datasets = ["prostate", "isic"]  # Update this list with your datasets
    plot_auroc_means(datasets)
