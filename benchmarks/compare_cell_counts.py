import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from myterial import (
    grey,
    grey_dark,
    indigo_dark,
    indigo_light,
    light_green_dark,
    light_green_light,
    purple_dark,
    purple_light,
    red_dark,
    red_light,
    teal_dark,
    teal_light,
)
from sklearn.linear_model import LinearRegression

POINT_SIZE = 50
ALPHA = 0.6
FIG_SIZE = (6, 3.7)
AXES_MAX = 1500
AXES_MIN = -60
NUM_DECIMAL_PLACES = 6

SAMPLE_NAMES = ["Brain 1", "Brain 2"]
STRUCT_NAMES_ABBREVIATIONS = {
    "Primary visual area, layer 2/3": "VISp2/3",
    "Primary visual area, layer 5": "VISp5",
    "Dorsal part of the lateral geniculate complex, core": "LGd-co",
    "Primary visual area, layer 4": "VISp4",
    "Lateral posterior nucleus of the thalamus": "LP",
}


def plot_scatter(df, ax, num_highlights=5):
    untrained_color = grey
    trained_color = grey_dark

    correlation, coeff, intercept = compare_counts_stats(
        df["total_cells"], df["total_cells_retrained"]
    )

    ax.scatter(
        df["total_cells"].iloc[num_highlights:],
        df["total_cells_default"].iloc[num_highlights:],
        s=POINT_SIZE,
        alpha=ALPHA,
        c=untrained_color,
        label="Pre-trained\nnetwork",
    )

    ax.scatter(
        df["total_cells"].iloc[num_highlights:],
        df["total_cells_retrained"].iloc[num_highlights:],
        s=POINT_SIZE,
        alpha=ALPHA,
        c=trained_color,
        label="Re-trained\nnetwork",
    )

    colors_light = [
        light_green_light,
        teal_light,
        indigo_light,
        purple_light,
        red_light,
    ]

    colors = [light_green_dark, teal_dark, indigo_dark, purple_dark, red_dark]

    for point in range(num_highlights):
        ax.scatter(
            df["total_cells"].iloc[point],
            df["total_cells_default"].iloc[point],
            s=POINT_SIZE,
            alpha=ALPHA,
            c=colors_light[point],
            label="Pre-trained\nnetwork",
        )
        ax.scatter(
            df["total_cells"].iloc[point],
            df["total_cells_retrained"].iloc[point],
            s=POINT_SIZE,
            alpha=ALPHA,
            c=colors[point],
            label="Re-trained\nnetwork",
        )

    line_top = AXES_MAX * 0.9

    ax.plot(
        [0, line_top], [0, line_top], color="0", ls="dashed", label="Slope = 1"
    )
    ax.plot(
        [0, line_top],
        [0, line_top * coeff],
        color=trained_color,
        ls="dashed",
        label="Best fit",
    )

    ax.text(
        0.95,
        0.1,
        f"{correlation=:.{NUM_DECIMAL_PLACES}f}\n{coeff=:.{NUM_DECIMAL_PLACES}f}",
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
    )


def compare_counts_stats(count_one: pd.Series, count_two: pd.Series):
    corr = np.corrcoef(count_one, count_two)[0][1]
    regression = LinearRegression(fit_intercept=False).fit(
        count_one.values.reshape(-1, 1),
        count_two.values.reshape(-1, 1),
    )

    coeff = regression.coef_[0][0]
    intercept = 0
    return corr, coeff, intercept


def compare_and_graph_counts(
    rater_one_paths,
    rater_two_paths,
    default_model_paths,
    retrained_model_paths,
    file_name,
):
    num_samples = len(rater_one_paths)
    dfs = []
    fig, axs = plt.subplots(
        1, num_samples, figsize=FIG_SIZE, tight_layout=True
    )

    for i in range(num_samples):
        count_one = pd.read_csv(rater_one_paths[i]).filter(
            ["structure_name", "total_cells"]
        )
        count_two = pd.read_csv(rater_two_paths[i]).filter(
            ["structure_name", "total_cells"]
        )

        df = pd.merge(
            count_one[["structure_name", "total_cells"]],
            count_two[["structure_name", "total_cells"]],
            on="structure_name",
            suffixes=("_rater1", "_rater2"),
        )

        df["total_cells"] = df[
            ["total_cells_rater1", "total_cells_rater2"]
        ].mean(axis=1)

        cellfinder_default = pd.read_csv(default_model_paths[i]).filter(
            ["structure_name", "total_cells"]
        )
        cellfinder_retrained = pd.read_csv(retrained_model_paths[i]).filter(
            ["structure_name", "total_cells"]
        )

        df = pd.merge(
            df,
            cellfinder_default,
            how="outer",
            on="structure_name",
            suffixes=(None, "_default"),
        )
        df = pd.merge(
            df,
            cellfinder_retrained,
            how="outer",
            on="structure_name",
            suffixes=(None, "_retrained"),
        )

        df.fillna(0, inplace=True)
        df.sort_values(by="total_cells", inplace=True, ascending=False)

        dfs.append(df)

        plot_scatter(df, axs[i])

    top_labels = dfs[0].head(5)["structure_name"].values

    for idx, label in enumerate(top_labels):
        top_labels[idx] = STRUCT_NAMES_ABBREVIATIONS[label]

    for idx, ax in enumerate(axs):
        handles, labels = ax.get_legend_handles_labels()
        handles = [
            handles[-2],
            handles[-1],
            *handles[3:-2:2],
        ]
        labels = [labels[-2], labels[-1], *top_labels]

        ax.legend(
            handles, labels, loc="upper left", frameon=False, fontsize=10
        )

        ax.set_title(SAMPLE_NAMES[idx])
        ax.set_yticks(np.arange(0, 1001, 500))
        ax.set_xticks(np.arange(0, 1001, 500))
        ax.set_xlim(AXES_MIN, AXES_MAX)
        ax.set_ylim(AXES_MIN, AXES_MAX)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    axs[0].set_ylabel("Algorithm cell count", fontsize=12)
    axs[-1].set_yticks([])

    axs[-1].spines["left"].set_visible(False)

    fig.supxlabel("Manual cell count", fontsize=12)
    fig.savefig(file_name, dpi=300)


if __name__ == "__main__":
    rater_one_paths = [
        "/home/igor/NIU-dev/cellfinder_data/cell_counts/manual_cell_counts/rater1/brain1.csv",
        "/home/igor/NIU-dev/cellfinder_data/cell_counts/manual_cell_counts/rater1/brain2.csv",
    ]

    rater_two_paths = [
        "/home/igor/NIU-dev/cellfinder_data/cell_counts/manual_cell_counts/rater2/brain1.csv",
        "/home/igor/NIU-dev/cellfinder_data/cell_counts/manual_cell_counts/rater2/brain2.csv",
    ]

    cellfinder_default_model_paths = [
        "/home/igor/NIU-dev/cellfinder_data/MS_CX_left_cellfinder_torch/analysis/summary.csv",
        "/home/igor/NIU-dev/cellfinder_data/MS_CX_right_cellfinder_torch/analysis/summary.csv",
    ]
    cellfinder_retrained_model_paths = [
        "/home/igor/NIU-dev/cellfinder_data/MS_CX_left_cellfinder_torch_paper/analysis/summary.csv",
        "/home/igor/NIU-dev/cellfinder_data/MS_CX_right_cellfinder_torch_paper/analysis/summary.csv",
    ]

    compare_and_graph_counts(
        rater_one_paths,
        rater_two_paths,
        cellfinder_default_model_paths,
        cellfinder_retrained_model_paths,
        "comparison_old_detection.png",
    )
