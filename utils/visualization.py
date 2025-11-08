import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_with_target(
    data: pd.DataFrame, target_column: str, plot_size: tuple = (12, 8)
):
    # Compute correlations between all variables and target
    correlations = data.corr()[target_column].drop(target_column).sort_values()

    # Generate a color palette from red to green
    colors = sns.diverging_palette(10, 130, as_cmap=True)
    color_mapped = correlations.map(colors)

    # Set Seaborn style
    sns.set_style(
        "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
    )  # Light grey background and thicker grid lines

    # Create bar plot
    # fig = plt.figure(figsize=plot_size)
    plt.figure(figsize=plot_size)
    plt.barh(correlations.index, correlations.values, color=color_mapped)

    # Set labels and title with increased font size
    plt.title(f"Correlation with `{target_column}`", fontsize=18)
    plt.xlabel("Correlation Coefficient", fontsize=16)
    plt.ylabel("Variable", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="x")

    plt.tight_layout()
    plt.show()


def plot_heatmap(data, plot_size: tuple = (12, 8)):
    corr = data.corr()
    plt.figure(figsize=plot_size)
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()
