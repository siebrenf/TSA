import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_alignments(template_time: list, extended_template_time: list, alignment_files: dict):
    # x axis
    df = pd.DataFrame({
        "template_time": extended_template_time,
        "x": list(range(len(extended_template_time))),
    })
    xticks = df[df["template_time"].isin(template_time)]["x"].to_list()
    xlabels = template_time
    for n, timeseries in enumerate(alignment_files):
        df2 = pd.read_csv(alignment_files[timeseries], sep="\t", index_col=0)
        df = df.merge(df2, left_on="template_time", right_on="inferred_time", how="left")
        cols = list(df.columns)[:-1] + [timeseries]
        df.columns = cols
    df.set_index("template_time", inplace=True)

    # start plot
    plt.rcParams['figure.figsize'] = [24, 6]
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)

    # add aligned timeseries (y axis)
    ylabels = []
    for n, timeseries in enumerate(alignment_files):
        x = df[~df[timeseries].isnull()]["x"].to_list()
        y = list(np.zeros_like(x) + n)
        ylabels.append(timeseries)
        plt.scatter(x=x, y=y)

    # plot shape
    n_series = len(alignment_files)
    x_range = max(xticks) - min(xticks)

    plt.yticks(list(range(n_series)), ylabels)
    plt.ylim(-0.5, n_series - 0.5)
    plt.ylabel("time series")

    plt.xlim(min(xticks) - x_range * 0.03, max(xticks) + x_range * 0.03)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels, rotation=45, ha="right")

    plt.show()

