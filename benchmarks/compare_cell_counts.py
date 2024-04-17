import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    mean["total_cells"] = combined_df.mean(axis=1)

    combined_df = pd.merge(
        left=mean,
        right=count_3,
        how="outer",
        left_on="structure_name",
        right_on="structure_name",
        suffixes=["_" + "consensus", "_" + name3],
    )

    combined_df.fillna(value=0, inplace=True)
    corr = combined_df.corr()

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

left_torch_untrained = (
    "home/igor/NIU-dev/cellfinder_data/"
    "MS_CX_left_cellfinder_torch/analysis/summary.csv"
)

left_torch_retrained = (
    "/home/igor/NIU-dev/cellfinder_data/"
    "MS_CX_left_cellfinder_torch_paper/analysis/summary.csv"
)


fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, tight_layout=True)

x = np.linspace(0, 1200, 1000)
y = np.linspace(0, 1200, 1000)
for idx, brain in enumerate(["left", "right"]):
    if brain == "right":
        rater_one = right_rater_one
        rater_two = right_rater_two
        # torch = right_torch_untrained
        # torch_retrained = right_torch_retrained

    else:
        rater_one = left_rater_one
        rater_two = left_rater_two
        torch = left_torch_untrained
        torch_retrained = left_torch_retrained

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
        x_untrained,
        y_untrained,
        s=POINT_SIZE,
        alpha=ALPHA,
        c="cornflowerblue",
        label="Pre-trained\nnetwork",
    )
    splot = axs[idx].scatter(
        x_trained,
        y_trained,
        s=POINT_SIZE,
        alpha=ALPHA,
        c="coral",
        label="Re-trained\nnetwork",
    )

    axs[idx].plot(x, y, color="0.50", ls="dashed", label="Slope = 1")

    axs[idx].plot(x, untrained_line, color="cornflowerblue", ls="dashed")
    axs[idx].plot(x, trained_line, color="coral", ls="dashed")


legend = axs[0].legend(loc="upper left", frameon=False, fontsize=10)
axs[0].set_title("Brain 1")
axs[1].set_title("Brain 2")

axs[0].set_yticks(np.arange(0, 1001, 500))
axs[0].set_xticks(np.arange(0, 1001, 500))
axs[1].set_xticks(np.arange(0, 1001, 500))


axs[0].set_ylabel("Algorithm cell count", fontsize=12)
axs[1].set_yticks([])

axs[1].spines["left"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[1].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[0].spines["top"].set_visible(False)
fig.supxlabel("Manual cell count", fontsize=12)
plt.show()
