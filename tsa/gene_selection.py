import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def scale(data):
    """scale array between 0-1"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def score_normalization(scores: np.array, weight_expr=1, weight_r2=1) -> pd.DataFrame:
    df = scores
    df["scaled_delta"] = scale(df["delta"])
    df["score"] = df["scaled_delta"].pow(weight_expr) * df["r2"].pow(weight_r2)
    df = df.sort_values("score", ascending=False)
    return df


def plot_scores(normscores: pd.DataFrame, highlight_top_n: int = None):
    ax = sns.scatterplot(
        data=normscores, x="r2", y="delta", hue="score",
        edgecolor="none", alpha=0.5,  palette='viridis_r'
    )

    # show score cutoff
    if highlight_top_n:
        top_genes = normscores[:highlight_top_n]
        ax = sns.scatterplot(
            data=top_genes, x="r2", y="delta",
            edgecolor="none", alpha=0.05,
        )

    # create colobar
    # source: https://stackoverflow.com/questions/62884183/trying-to-add-a-colorbar-to-a-seaborn-scatterplot
    norm = plt.Normalize(normscores["score"].min(), normscores["score"].max())
    sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=norm)
    sm.set_array([])

    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    ax.figure.colorbar(sm)

    plt.show()


def best_n_genes(normscores, n=None, frac=None, to_file=None) -> list:
    """
    Return the top n or top fraction of genes. 
    Also saves to file if a path is given.
    """
    normscores = normscores.sort_values("score", ascending=False)
    total_genes = normscores.shape[0]
    if n and frac:
        raise ValueError("Please enter a value for `frac` OR `n`, not both")
    elif n:
        selected_genes = list(normscores.reset_index()[0:min(n, total_genes)]["gene"])
    elif frac:
        selected_genes = list(normscores.reset_index()[0:int(frac*total_genes)]["gene"])
    else:
        raise ValueError("Please specify `frac` OR `n`")
    if to_file:
        df = pd.DataFrame(selected_genes, columns=["gene"])
        df.to_csv(to_file, sep="\t", index=False)
    return selected_genes


# scores = np.array(
#     [[0.9352516, 3.0408133],
#      [0.94292263, 5.9805853],
#      [0.97562821, 7.79689539],
#      [0.94495644, 4.49907905],
#      [0.97739133, 7.61523243],
#      [0.94252969, 4.10630096],
#      [0.6160258, 3.3758958],
#      [0.97850004, 4.77149677],
#      [0.76169365, 4.40594625],
#      [0.88681373, 3.77448725]]
# )
# tpms = pd.DataFrame({"gene": [f"gene_{n}" for n in range(12)]}).set_index("gene")
# normscores = score_normalization(tpms)
# print(normscores)
# print(best_n_genes(normscores, 4).tolist())
# plot_scores(normscores, 4)
