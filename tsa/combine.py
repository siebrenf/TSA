import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from tsa.utils import list2floats, all_numeric, read_alignment


def plot_offset(template_timepoints: list, alignment_files: list, to_file=None):
    """
    Compare the annotated time to the inferred time.
    If time annotations of the template and queries would be perfect,
    as well as the time series alignment, then these would fall on the diagonal.
    
    Requires annotated times to be on the same (numeric) scale.
    """
    template_timepoints = list2floats(template_timepoints)
    if not all_numeric(template_timepoints):
        raise ValueError("time must be numeric")

    # limit the plot to the range of the query series (plus padding)
    start = template_timepoints[0]
    end = 0

    # plot each time series
    for alignment_file in alignment_files:
        alignment, query_label, template_label = read_alignment(alignment_file)
        x = list2floats(alignment.index)  # annotated_time
        y = list2floats(alignment.inferred_time)
        if not all_numeric(x):
            raise ValueError("time must be numeric")
        if query_label == template_label:
            # control data can be in the background
            plt.plot(x, y, alpha=0.5, label=query_label, zorder=-5)
            plt.scatter(x, y, alpha=0.5, s=10, color="black", zorder=-4)
        else:
            plt.plot(x, y, alpha=0.5, label=query_label)
            plt.scatter(x, y, alpha=0.5, s=10, color="black")
        
        # update the plot bounds
        end = max(end, y[-1]*1.05)
    start = start - (end - start)*0.05

    # add template samples at their annotated time on the y-axis
    ts_y = template_timepoints
    ts_x = start+end*0.003  # move the points into the plot a little (for visual clarity)
    plt.scatter([ts_x for i in ts_y], ts_y, alpha=1, s=25, label=f"{template_label} samples", marker='_')

    # add diagonal
    plt.plot([start, end], [start, end], label="if time annotations were perfect", alpha=0.9, ls='--', zorder=-10)

    plt.xlim(start, end)
    plt.ylim(start, end)
    plt.title(f"Time series alignment to {template_label}", fontsize=18)
    plt.ylabel("template time", fontsize=18)
    plt.xlabel("query time", fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
    
    if to_file:
        plt.savefig(to_file, bbox_inches='tight')
    plt.show()


# def nearest(myNumber, myList):
#     return min(myList, key=lambda x:abs(x-myNumber))
#
#
# # nearest(2.1, [1,2,3])


def plot_timeline(template_time: list, extended_template_time: list, alignment_files: list, labels=None, to_file=None):
    """
    
    """
    # parse input
    template_time = list2floats(template_time)
    extended_template_time = list2floats(extended_template_time)
    is_num = all_numeric(extended_template_time)
    if is_num is not all_numeric(template_time):
        raise ValueError("'template_time' and 'extended_template_time' must be of the same dtype "
                         "(You can add labels in any type).")
    
    # generate x-axis, xticks and xlabels
    if is_num:
        x = extended_template_time
        xticks = template_time
    else:
        x = list(range(len(extended_template_time)))
        xticks = [extended_template_time.index(i) for i in template_time]
    xlabels = labels if labels else template_time
    if len(template_time) != len(xlabels):
        raise ValueError("'labels' must have the same length as 'template_time'")
    
    ylabels = []
    yticks = []
    n = 0
    for alignment_file in alignment_files:
        query, query_label, template_label = read_alignment(alignment_file)
        annotated_time = list2floats(query.index)
        inferred_time = list2floats(query.inferred_time)
        
        # plot annotated time (both t and q(annotated) must be numeric)
        if is_num and all_numeric(annotated_time):
            x1 = annotated_time
            y = list(np.zeros_like(x1) + n)
            label = f"{query_label} annotated"
            plt.scatter(x=x1, y=y, label=label)
            ylabels.append(label)
            yticks.append(n)
            n += 1
        
        # plot inferred time
        x2 = inferred_time  # if t and q time are both numerics
        if not is_num:  # if t and q time are both strings
            x2 = [x[extended_template_time.index(it)] for it in inferred_time]
        y = list(np.zeros_like(x2) + n)
        label = f"{query_label} inferred"
        plt.scatter(x=x2, y=y, label=label)
        ylabels.append(label)
        yticks.append(n)
        n += 1
        
        if is_num and all_numeric(annotated_time):
            # plot connections between annotated and inferred times
            matches = zip(x1, x2)
            for i, j in matches:
                plt.plot((i, j), (n-2, n-1), color = 'black', alpha=0.1, linestyle='--')
        
        # add spacing between series
        n += 1
    
    plt.title(f"Time series alignment to {template_label}", fontsize=15)
    plt.xlabel("time", fontsize=15)
    plt.xticks(xticks, xlabels, rotation=45, ha="right")
    x_padding = (max(xticks)-min(xticks))*0.01
    plt.xlim(min(xticks) - x_padding, max(xticks) + x_padding)
    plt.ylabel("time series", fontsize=15)
    plt.yticks(yticks, ylabels, fontsize=15)
    
    if to_file:
        plt.savefig(to_file)
    
    plt.show()
    return
