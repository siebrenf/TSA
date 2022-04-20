import inspect
import multiprocessing as mp
import numpy as np
import os
import random
from typing import Union, List

from matplotlib import pyplot as plt
import networkx as nx
import pandas as pd
from scipy.spatial.distance import cdist
import sklearn

from tsa.clustering import top_cluster_genes
from tsa.utils import all_numeric, list2floats


def get_cost_matrix(template_tpms, query_tpms, metric='euclidean'):
    template = template_tpms.to_numpy(dtype=np.float64)
    query = query_tpms.to_numpy(dtype=np.float64)
    cost_matrix = cdist(template.T, query.T, metric=metric).T  # pairwise distance matrix
    return cost_matrix


def best_alignment_graph(cost_matrix: np.array) -> tuple:
    len_query, len_template = cost_matrix.shape
    G = nx.DiGraph()

    # add start edges
    all_edges = [("start", (0, i), cost_matrix[0, i]) for i in range(len_template)]

    # nodes are named as 2d tuples of coordinates ((q)uery point, (t)emplate point)
    # each node connects to the next number of q, and the same or higher number of t
    for q in range(len_query - 1):
        for t in range(len_template):
            node_i = (q, t)
            for t2 in range(t, len_template):
                node_j = (q + 1, t2)
                all_edges.append((node_i, node_j, cost_matrix[node_j]))

    # add final edges
    all_edges.extend([((len_query - 1, i), "end", 0) for i in range(len_template)])

    # now load them in networkx
    G.add_weighted_edges_from(all_edges)

    best_path = nx.shortest_path(G, source="start", target="end", weight='weight')[1:-1]
    best_path_template, best_path_query = zip(*best_path)
    best_score = cost_matrix[best_path_template, best_path_query].sum()
    best_path = best_path_query

    return best_path, best_score


def combine_alignments(paths: List[list], inference_time: list = None, is_time_numeric: bool = False,
                       force_chronological: bool = True, method: Union[str, int] = "mean"):
    """
    Combine {paths}, a list of integer lists, into a single list using the specified {method}.
    If {method} is an integer, use that percentile.
    If {force_chronological} is True, the combined path will be made chronological.
    
    This function also returns the standard deviation as a (2,n) numpy array.
    If time is numeric, the std will be returned in inferred time, instead of column indeces.
    """
    # combine paths
    if method == "mean":
        path = [int(i) for i in np.mean(paths, axis=0)]
    elif method == "median":
        path = [int(i) for i in np.median(paths, axis=0)]
    elif isinstance(method, int):
        path = [int(i) for i in np.percentile(paths, q=int(method), axis=0)]
    else:
        raise ValueError("`method` must be 'mean', 'median' or a integer precentile")

    # standard deviation from the combined path
    std_path = paths.std(axis=0).tolist()
    # if time is numeric, we convert the std to inferred time
    if is_time_numeric:
        if inference_time is None:
            raise ValueError("`inference_time` is required when time is numeric")
        end = len(inference_time) - 1
        for n, p in enumerate(path):
            inf_time = inference_time[p]
            std_time_min = abs(inference_time[max(0, p - int(std_path[n]))] - inf_time)
            std_time_max = abs(inference_time[min(end, p + int(std_path[n]))] - inf_time)
            std_path[n] = [std_time_min, std_time_max]
    std_path = np.array(std_path).T

    if force_chronological:
        # if an average timepoint is earlier than the previous timepoint, increase it to match
        for n in range(1, len(path)):
            if path[n - 1] > path[n]:
                path[n] = path[n - 1]

    return path, std_path


def _parse_sampling(frac=None, n_clust=None, n_total=None):
    """raise an error if not exactly one of the variables is specified"""
    vals = sum([1 for _ in [frac, n_clust, n_total] if _ is not None])
    if vals != 1:
        raise ValueError("need either frac, n_clust or n_total")


def _sample_genes(gene_cluster_df, frac=None, n=None, r=None):
    # frac != None: fraction of genes per cluster
    # n != None: fixed number of genes per cluster/divided over the number of clusters
    # uses replacement if n was specified
    genes = gene_cluster_df.groupby("cluster").sample(n=n, frac=frac, replace=bool(n), random_state=r).index.to_list()
    return genes


def subset_alignment(template_tpms_inf, query_tpms, gene_cluster_df, frac=None, n=None, metric='correlation', r=None):
    """
    Apply a local time series alignment of {query_tpms} to {template_tpms_inf}, using a subset of genes.
    The subset is drawn from {gene_cluster_df}, where a number of genes is used from each cluster.
    This can be a {frac}tion of genes, or an exact {n}umber.
    
    When using this function in parallel, a {r}andom number is required.
    
    Returns an alignment path
    """
    genes = _sample_genes(gene_cluster_df, frac, n, r)
    del gene_cluster_df
    del frac
    del n
    del r

    t = template_tpms_inf.loc[genes]
    del template_tpms_inf
    q = query_tpms.loc[genes]
    del query_tpms

    cost_matrix = get_cost_matrix(t, q, metric)
    path = best_alignment_graph(cost_matrix)[0]
    del cost_matrix

    return path


def multi_alignment(
        template_tpms_inf, query_tpms, gene_cluster_df,
        tries=10, frac=None, n_clust=None, n_total=None,
        metric="correlation", ncpu=4,
        showcase_gene=None, plot=True, verbose=True, return_std=False
):
    """
    Run a number of subset_alignments (in parallel), using combine_alignments on the results.
    
    tries: number of times the alignment is performed.
    frac: fraction of genes to use per cluster
    n_clust: number of genes to use per cluster
    n_total: total number of genes to use (rounding down if required)
    """
    _parse_sampling(frac, n_clust, n_total)

    # run tries in parallel
    p = max(1, min(ncpu, os.cpu_count() - 1, tries))
    pool = mp.Pool(processes=p)

    clusters = list(set(gene_cluster_df.cluster))
    n_clusters = len(clusters)
    if n_total:
        # fixed number of genes divided over the number of clusters
        n_clust = int(n_total / n_clusters)

    try:
        jobs = [
            pool.apply_async(
                func=subset_alignment,
                args=(template_tpms_inf, query_tpms, gene_cluster_df, frac, n_clust, metric, random.randint(0, 999))
            ) for _ in range(tries)
        ]
        paths = []
        for try_n, j in enumerate(jobs):
            if verbose:
                print(f"{int(100 * try_n / tries)}%", end="\r")
            paths.append(j.get())
        paths = np.asarray(paths)
        pool.close()
        pool.join()
    except:
        pool.terminate()
        raise RuntimeError("Exception occurred, multiprocessing halted")

    # combine tries
    inference_time = list2floats(template_tpms_inf.columns)
    query_time = list2floats(query_tpms.columns)
    is_time_numeric = all_numeric(inference_time) and all_numeric(query_time)
    path, std_path = combine_alignments(paths, inference_time, is_time_numeric, method="mean")

    if plot:
        if verbose:
            start_msg = f"TSA of {tries} alignments"
            clust_msg = f" with {n_clusters} clusters" if n_clusters > 1 else ""
            gene_no = int(len(gene_cluster_df) * frac) if frac is not None else n_clusters * n_clust
            print(f"\t{start_msg}{clust_msg}, {gene_no} genes.")
        plot_alignment(query_time, inference_time, path, std_path, is_time_numeric)

        if is_time_numeric:
            # gene mapping only makes sense if both query and template are in numeric time
            plot_gene(query_tpms, template_tpms_inf, path, showcase_gene, scale=True)

    if return_std:
        return path, std_path
    return path


# def avg_alignment(template_tpms_inf, query_tpms, gene_cluster_df, tries=10, frac=0.2,
#                   metric='correlation', showcase_gene=None, plot=True, verbose=True, return_std=False
# ):
#     paths = np.zeros((tries, query_tpms.shape[1]))
#     for n in range(tries):
#         if verbose:
#             print(f"{int(100*n/tries)}%", end="\r")
#
#         # get a fraction of genes per cluster
#         genes = gene_cluster_df.groupby("cluster").sample(frac=frac).index.to_list()
#
#         t = template_tpms_inf[template_tpms_inf.index.isin(genes)]
#         q = query_tpms[query_tpms.index.isin(genes)]
#
#         cost_matrix = get_cost_matrix(t, q, metric)
#         best_path, _ = best_alignment_graph(cost_matrix)
#
#         paths[n] = best_path
#
#     inference_time = list2floats(template_tpms_inf.columns)
#     query_time = list2floats(query_tpms.columns)
#     is_time_numeric = all_numeric(inference_time) and all_numeric(query_time)
#     avg_path, std_path = _avg_alignments(paths, inference_time, is_time_numeric)
#
#     if plot:
#         print(f"Average TSA of {tries} alignments with {int(frac*100)}% of genes per cluster \n"
#               f"({len(set(gene_cluster_df.cluster))} clusters, {len(q)} total genes)")
#         cm = pd.DataFrame(cost_matrix, index=q.columns, columns=t.columns)
#         plot_alignment(cm, avg_path, std_path)

#         if is_time_numeric:
#             # gene mapping only makes sense if both query and template are in numeric time
#             plot_gene(query_tpms, template_tpms_inf, avg_path, showcase_gene, scale=True)
#
#     if return_std:
#         return avg_path, std_path
#     return avg_path


def plot_alignment(query_time, template_time, path, std_path=None, is_time_numeric=False):
    if is_time_numeric:
        q = query_time
        t = [template_time[i] for i in path]

        # add diagonal
        start = min(t[0], q[0])
        end = max(t[-1], q[-1])
        plt.plot([start, end], [start, end], 'orange', alpha=0.2, ls='--')

        plt.ylim(t[0], t[-1])
        plt.xlim(q[0], q[-1])
    else:
        q = list(range(len(path)))
        t = path

        plt.ylim(0, len(template_time) - 1)
        plt.xlim(0, len(query_time) - 1)

    plt.plot(q, t, alpha=0.5)
    plt.scatter(q, t, s=10, color="black")
    if std_path is not None:
        plt.errorbar(q, t, color="grey", yerr=std_path, alpha=0.4, fmt='none')

    plt.title("local alignment", fontsize=15)
    plt.ylabel("template time", fontsize=18)
    plt.xlabel("query time", fontsize=15)
    plt.show()


def plot_gene(query_tpms, template_tpms_inf, path, gene=None, scale=False, cycle=True):
    """
    Visualize annotated time vs inferred time for a specified gene.
    Requires that annotated time is numeric for both template and query.
    """
    if gene is None:
        gene = query_tpms.sample(1).index[0]

    x1 = list2floats(template_tpms_inf.columns)
    x2 = list2floats(query_tpms.columns)
    x3 = [x1[i] for i in path]

    y1 = template_tpms_inf.loc[gene].to_list()
    y2 = query_tpms.loc[gene].to_list()
    if scale:
        y1 = sklearn.preprocessing.scale(y1)
        y2 = sklearn.preprocessing.scale(y2)
    y3 = [y1[i] for i in path]

    colors = plt.cm.plasma(np.linspace(0, 0.9, len(path)))
    # colors = cm.hot(np.linspace(0.4, 0.6, len(path)))
    # colors = cm.gist_heat(np.linspace(0.3, 0.9, len(path)))
    # plt.scatter(x=x1, y=y1, color="C0", s=30, label="template")
    plt.plot(x1, y1, color="C0", zorder=0)
    plt.scatter(x=x1, y=y1, color="C0", marker="|", zorder=0, label="template")
    plt.scatter(x=x2, y=y2, color=colors if cycle else "C8", alpha=0.5, marker="D", zorder=2, s=15,
                label="query annotated")
    plt.scatter(x=x3, y=y3, color=colors if cycle else "C1", zorder=1, s=30, label="query inferred")
    for n in range(len(x2)):
        plt.plot((x3[n], x2[n]), (y3[n], y2[n]), color="black", alpha=0.1, linestyle='--')

    plt.title(f"{gene} alignment", fontsize=15)
    plt.ylabel("normalized expression", fontsize=18)
    plt.xlabel("time", fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
    start = min(x1[path[0]], x2[0])
    end = max(x1[path[-1]], x2[-1])
    a_bit = (end - start) * 0.05
    plt.xlim(start - a_bit, end + a_bit)
    plt.show()


def time_series_alignment(template, query, gene_cluster_df=None, tries=10, frac=None, n_clust=None, n_total=None,
                          cycles=1, shrink_rate=0.9, verbose=True, **kwargs):
    """
    Apply a local time series alignment of {query} to {template}.
    Returns the path (and standard deviation if {return_std} is True) for each cycle.
    The path contains the indices of {template} columns best matching each {query} column.
    
    Uses a random fraction of genes (per cluster of genes if {gene_cluster_df} is provided).
    This is repeated several {tries}, and averaged to obtain an alignment path.
    
    If {cycles} > 1, the alignments will then be used to bootstrap a subsequent alignments:
    First by measuring the ({scale}d) Goodness of Fit for each gene,
    then selecting a top fraction of genes, based on the cumulative {shrink_rate}.
    This fraction is used for the next cycle's average alignment.
    
    Parameters
    ----------
    template: pd.DataFrame
        dataframe with gene names as index and expression values at a timepoint per column.
        Columns are assumed to be sorted chronologically, and the time range to contain the query time range.
        If column names can be converted to numeric in query and template, this is used to improve plots and standard deviations.
        It will be assumed that numerics are in the same time unit.
    query: pd.DataFrame
        dataframe with gene names as index and expression values at a timepoint per column.
        Columns are assumed to be sorted chronologically, and the time range to be contained by the template time range.
        If column names can be converted to numeric in query and template, this is used to improve plots and standard deviations.
        It will be assumed that numerics are in the same time unit.

    Alignment parameters
    --------------------
    gene_cluster_df: pd.DataFrame, optional
        optional dataframe with gene names as index and a column "cluster" with gene clusters. 
        Query and template are filtered for genes in this dataframe
    tries: int, optional
        numer of times the alignment is performed per average. Default is 10
    frac: float, optional
        fraction of genes to use (per cluster) for alignment. Default is 0.2
        You can only specify one: {frac}, {n_clust} or {n_total}.
    n_clust: int, optional
        number of genes to use per cluster.
        You can only specify one: {frac}, {n_clust} or {n_total}.
    n_total: int, optional
        total number of genes to use.
        You can only specify one: {frac}, {n_clust} or {n_total}.
    metric: string, optional
        distance metric to apply for the TSA. See scipy.spatial.distance.cdist for options. Default is 'correlation'
    ncpu: int, optional
        the number of tries to run in parallel. Default is 4
    
    Bootstrap parameters
    --------------------
    cycles: int, optional
        number of alignment cycles to perform. Bootstrapping takes effect with cycle > 1. Default is 1
    shrink_rate: float, optional
        cumulative fraction of genes to use for each cycle. Default is 0.9
    scale: bool, optional
        whether to scale gene expression values before determining Goodness of Fit. Default is True
    
    Other parameters
    ----------------
    verbose: bool, optional
        whether to return additional messages on progress and gene usage. Default is True
    return_std: bool, optional
        whether to return standard deviation of the average alignment. Default is False
        standard deviations are shown in template time if columns are numeric, or template timepoints if columns are not.
    plot: bool, optional
        whether to plot the alignment (and showcase_gene for each alignment). Default is True
    showcase_gene: string, optional
        specify a gene name to plot after each alignment. Uses a random gene if None (default).
        
    Returns
    -------
    paths: list
        if cycles == 1, containing the template columns indices best matching the query columns. Each column being a timepoint.
        if cycles >1, a list of lists with a path for each cycle.
        if return_std = True, returns tuple(s) containing the path and the standard deviation (both up and down).
    """
    if frac is None and n_clust is None and n_total is None:
        frac = 0.2  # default setting if none is provided
    if frac and (frac > 1 or frac <= 0):
        raise ValueError("`frac` must be within (0,1]")

    # filter kwargs for sub-functions
    alignment_kwargs = {}
    cluster_kwargs = {}
    if kwargs:
        keys = inspect.getfullargspec(multi_alignment).args
        alignment_kwargs = {k: kwargs[k] for k in keys if k in kwargs}
        keys = inspect.getfullargspec(top_cluster_genes).args
        cluster_kwargs = {k: kwargs[k] for k in keys if k in kwargs}

    # match query and template index order
    if gene_cluster_df is None:
        # use all overlapping genes if no cluster information is provided
        overlapping_genes = list(set(template.index).intersection(query.index))
        template = template.loc[overlapping_genes]
        query = query.loc[overlapping_genes]
        gene_cluster_df = pd.DataFrame({
            "gene": overlapping_genes,
            "cluster": [0 for _ in range(len(overlapping_genes))]
        }).set_index("gene")
    else:
        template = template.loc[gene_cluster_df.index]
        query = query.loc[gene_cluster_df.index]

    # initial TSA
    if verbose and cycles > 1:
        print(f"Cycle 1, using all {len(gene_cluster_df)} genes.\n")
    path = multi_alignment(template, query, gene_cluster_df, tries, frac, n_clust, n_total, verbose=verbose,
                           **alignment_kwargs)

    if cycles == 1:
        return path

    # bootstrapped TSA's
    # return a list of average paths per cycle
    paths = [path]
    current_frac = shrink_rate
    for n in range(cycles)[1:]:
        # score the Goodness of Fit for all genes, based on the current path.
        # use the {top_frac}tion of genes for the next alignment.
        # shrink this fraction by the {shrink_rate} every cycle.
        top_gene_clusters, _ = top_cluster_genes(
            template, query, path, gene_cluster_df,
            top_frac=current_frac, filter_frac=0, **cluster_kwargs
        )
        if verbose:
            print(f"Cycle {n + 1}, using the best {len(top_gene_clusters)} genes.\n")
        current_frac = current_frac * shrink_rate

        path = multi_alignment(template, query, top_gene_clusters, tries, frac, n_clust, n_total, verbose=verbose,
                               **alignment_kwargs)
        paths.append(path)
    return paths
