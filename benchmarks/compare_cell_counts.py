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


def compare_counts(path1, name1, path2, name2, plot=True):
    count_1 = pd.read_csv(path1)
    count_2 = pd.read_csv(path2)

    count_1 = count_1.filter(["structure_name", "total_cells"])
    count_2 = count_2.filter(["structure_name", "total_cells"])
    combined_df = pd.merge(
        left=count_1,
        right=count_2,
        how="outer",
        left_on="structure_name",
        right_on="structure_name",
        suffixes=["_" + name1, "_" + name2],
    )
    combined_df.fillna(value=0, inplace=True)
    combined_df.sort_values(
        by="total_cells" + "_" + name1, inplace=True, ascending=False
    )
    corr = combined_df.corr()

    X = combined_df["total_cells" + "_" + name1].values.reshape(-1, 1)
    Y = combined_df["total_cells" + "_" + name2].values.reshape(-1, 1)
    regression = LinearRegression(fit_intercept=False).fit(X, Y)

    if plot:
        # splot = sns.scatterplot(data=combined_df,
        #                         x='total_cells' + "_" + name1,
        #                         y='total_cells' + "_" + name2,
        #                         size=POINT_SIZE,
        #                         alpha=0.6)

        plt.scatter(
            combined_df["total_cells" + "_" + name1],
            combined_df["total_cells" + "_" + name2],
        )
        x = np.linspace(0, 1200, 1000)
        y = np.linspace(0, 1200, 1000)

        plt.plot(x, y, color="0.50", ls="dashed")
        plt.show()

    correlation = corr.iloc[0][1]
    coeff = regression.coef_[0][0]
    # intercept = regression.intercept_[0]
    intercept = 0
    return correlation, coeff, intercept


def compare_counts_to_average(
    path1, name1, path2, name2, path3, name3, plot=True
):
    count_1 = pd.read_csv(path1)
    count_2 = pd.read_csv(path2)
    count_3 = pd.read_csv(path3)

    count_1 = count_1.filter(["structure_name", "total_cells"])
    count_2 = count_2.filter(["structure_name", "total_cells"])
    count_3 = count_3.filter(["structure_name", "total_cells"])

    combined_df = pd.merge(
        left=count_1,
        right=count_2,
        how="outer",
        left_on="structure_name",
        right_on="structure_name",
        suffixes=["_" + name1, "_" + name2],
    )

    mean = pd.DataFrame()
    mean["structure_name"] = combined_df["structure_name"]
    mean["total_cells"] = combined_df.mean(numeric_only=True, axis=1)

    combined_df = pd.merge(
        left=mean,
        right=count_3,
        how="outer",
        left_on="structure_name",
        right_on="structure_name",
        suffixes=["_" + "consensus", "_" + name3],
    )

    combined_df.fillna(value=0, inplace=True)
    combined_df.sort_values(
        by="total_cells" + "_" + "consensus", inplace=True, ascending=False
    )
    corr = combined_df.corr(numeric_only=True)

    X = combined_df["total_cells" + "_" + "consensus"].values.reshape(-1, 1)
    Y = combined_df["total_cells" + "_" + name3].values.reshape(-1, 1)
    regression = LinearRegression(fit_intercept=False).fit(X, Y)

    correlation = corr.iloc[0][1]
    coeff = regression.coef_[0][0]

    intercept = 0
    x = combined_df["total_cells" + "_" + "consensus"]
    y = combined_df["total_cells" + "_" + name3]
    return x, y, correlation, coeff, intercept


left_rater_one = (
    "/home/igor/NIU-dev/cellfinder_data/"
    "cell_counts/manual_cell_counts/rater1/brain1.csv"
)
right_rater_one = (
    "/home/igor/NIU-dev/cellfinder_data/"
    "cell_counts/manual_cell_counts/rater1/brain2.csv"
)

left_rater_two = (
    "/home/igor/NIU-dev/cellfinder_data/cell_counts/"
    "manual_cell_counts/rater2/brain1.csv"
)
right_rater_two = (
    "/home/igor/NIU-dev/cellfinder_data/cell_counts/"
    "manual_cell_counts/rater2/brain2.csv"
)

left_cellfinder_untrained = (
    "/home/igor/NIU-dev/cellfinder_data/"
    "MS_CX_left_cellfinder_tensorflow/analysis/summary.csv"
)
right_cellfinder_untrained = (
    "/home/igor/NIU-dev/cellfinder_data/"
    "MS_CX_right_cellfinder_tensorflow/analysis/summary.csv"
)

left_cellfinder_retrained = (
    "/home/igor/NIU-dev/cellfinder_data/"
    "cell_counts/algorithm_cell_counts/brain1.csv"
)
right_cellfinder_retrained = (
    "/home/igor/NIU-dev/cellfinder_data/"
    "cell_counts/algorithm_cell_counts/brain2.csv"
)


fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, tight_layout=True)

x = np.linspace(0, 1400, 1000)
y = np.linspace(0, 1400, 1000)
untrained_color = grey
trained_color = grey_dark

for idx, brain in enumerate(["left", "right"]):
    if brain == "right":
        rater_one = right_rater_one
        rater_two = right_rater_two
        torch = right_cellfinder_untrained
        torch_retrained = right_cellfinder_retrained

    else:
        rater_one = left_rater_one
        rater_two = left_rater_two
        torch = left_cellfinder_untrained
        torch_retrained = left_cellfinder_retrained

    (
        x_untrained,
        y_untrained,
        correlation_untrained_,
        coeff_untrained,
        intercept,
    ) = compare_counts_to_average(
        rater_one, "1", rater_two, "2", torch, "cellfinder", plot=True
    )

    x_trained, y_trained, correlation, coeff, intercept = (
        compare_counts_to_average(
            rater_one,
            "1",
            rater_two,
            "2",
            torch_retrained,
            "cellfinder",
            plot=True,
        )
    )

    untrained_line = x * coeff_untrained
    trained_line = x * coeff

    splot = axs[idx].scatter(
        np.array(x_untrained)[5:],
        np.array(y_untrained)[5:],
        s=POINT_SIZE,
        alpha=ALPHA,
        c=untrained_color,
        label="Pre-trained\nnetwork",
    )
    splot = axs[idx].scatter(
        np.array(x_trained)[5:],
        np.array(y_trained)[5:],
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
    for point in range(0, 5):
        splot = axs[idx].scatter(
            np.array(x_untrained)[point],
            np.array(y_untrained)[point],
            s=POINT_SIZE,
            alpha=ALPHA,
            c=colors_light[point],
            label="Pre-trained\nnetwork",
        )
        splot = axs[idx].scatter(
            np.array(x_trained)[point],
            np.array(y_trained)[point],
            s=POINT_SIZE,
            alpha=ALPHA,
            c=colors[point],
            label="Re-trained\nnetwork",
        )

    axs[idx].plot(x, y, color="0", ls="dashed", label="Slope = 1")
    # axs[idx].plot(x, untrained_line, color=untrained_color, ls='dashed')
    axs[idx].plot(
        x, trained_line, color=trained_color, ls="dashed", label="Best fit"
    )

for axis in (0, 1):
    handles, labels = axs[axis].get_legend_handles_labels()
    handles = [
        handles[12],
        handles[13],
        handles[3],
        handles[7],
        handles[7],
        handles[9],
        handles[11],
    ]
    labels = [
        labels[-2],
        labels[-1],
        "VISp2/3",
        "VISp5",
        "LGd-co",
        "LP",
        "VISp4",
    ]
    axs[axis].legend(
        handles, labels, loc="upper left", frameon=False, fontsize=10
    )

axs[0].set_title("Brain 1")
axs[1].set_title("Brain 2")

axs[0].set_yticks(np.arange(0, 1001, 500))
axs[0].set_xticks(np.arange(0, 1001, 500))
axs[1].set_xticks(np.arange(0, 1001, 500))

x_min = -60
y_min = -60
x_max = 1400
y_max = 1400

axs[0].set_xlim(x_min, x_max)
axs[0].set_ylim(y_min, y_max)
axs[1].set_xlim(x_min, x_max)
axs[1].set_ylim(y_min, y_max)


axs[0].set_ylabel("Algorithm cell count", fontsize=12)
axs[1].set_yticks([])

axs[1].spines["left"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].spines["top"].set_visible(False)
fig.supxlabel("Manual cell count", fontsize=12)
fig.savefig("comparison_tensorflow.png", dpi=300)
plt.show()
