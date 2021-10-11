import inspect
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import networkx as nx
from matplotlib import pyplot as plt
import random
import sklearn

from tsa.clustering import top_cluster_genes
from tsa.utils import all_numeric, list2floats


# def get_cost_matrix(template_tpms, query_tpms, selected_genes, time2samples):
#     template = template_tpms.loc[selected_genes]
#     template = template.to_numpy(dtype=np.float64)

#     query = query_tpms.loc[selected_genes]
#     # average technical replicates
#     for timepoint in time2samples:
#         query[timepoint] = query[time2samples[timepoint]].mean(axis=1)
#     query = query.filter(items=time2samples.keys())
#     query = query.to_numpy()

#     cost_matrix = cdist(template.T, query.T, metric='euclidean').T  # pairwise distance matrix
#     return cost_matrix


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
    for q in range(len_query-1):
        for t in range(len_template):
            node_i = (q, t)
            for t2 in range(t, len_template):
                node_j = (q+1, t2)
                # all_edges.append((node_i, node_j, cost_matrix[node_j]))
                all_edges.append((node_i, node_j, cost_matrix[node_j]))

    # add final edges
    all_edges.extend([((len_query-1, i), "end", 0) for i in range(len_template)])

    # now load them in networkx
    G.add_weighted_edges_from(all_edges)

    best_path = nx.shortest_path(G, source="start", target="end", weight='weight')[1:-1]
    best_path_template, best_path_query = zip(*best_path)
    best_score = cost_matrix[best_path_template, best_path_query].sum()
    best_path = best_path_query

    return best_path, best_score


def _avg_alignments(paths: list, inference_time: list = None, is_time_numeric=False, force_chonological=True):
    # average path
    avg_path = [int(i) for i in paths.mean(axis=0)]

    # std of average path
    std_path = [i for i in paths.std(axis=0)]
    
    if is_time_numeric:
        if inference_time is None:
            raise ValueError("`inference_time` is required when time is numeric")
        
        # path contains column indices, here we convert those to time
        end = len(inference_time)-1
        for n, p in enumerate(avg_path):
            inf_time = inference_time[p]
            std_time_min = abs(inference_time[max(0,   p - int(std_path[n]))]-inf_time)
            std_time_max = abs(inference_time[min(end, p + int(std_path[n]))]-inf_time)
            std_path[n] = [std_time_min, std_time_max]
    std_path = np.array(std_path).T
    
    if force_chonological:
        # if the average timepoint is later that the previous timepoint, increase it to match
        for n in range(1, len(avg_path)):
            if avg_path[n-1] > avg_path[n]:
                avg_path[n] = avg_path[n-1]
                
    return avg_path, std_path


def avg_alignment(template_tpms_inf, query_tpms, gene_cluster_df, tries=10, frac=0.2, metric='correlation', showcase_gene=None, plot=True, verbose=True, return_std=False):
    paths = np.zeros((tries, query_tpms.shape[1]))
    for n in range(tries):
        if verbose:
            print(f"{int(100*n/tries)}%", end="\r")

        # get a fraction of genes per cluster
        genes = gene_cluster_df.groupby("cluster").sample(frac=frac).index.to_list()

        t = template_tpms_inf[template_tpms_inf.index.isin(genes)]
        q = query_tpms[query_tpms.index.isin(genes)]

        cost_matrix = get_cost_matrix(t, q, metric)
        best_path, _ = best_alignment_graph(cost_matrix)

        paths[n] = best_path

    inference_time = list2floats(template_tpms_inf.columns)
    query_time = list2floats(query_tpms.columns)
    is_time_numeric = all_numeric(inference_time) and all_numeric(query_time)
    avg_path, std_path = _avg_alignments(paths, inference_time, is_time_numeric)
    
    if plot:
        print(f"Average TSA of {tries} alignments with {int(frac*100)}% of genes per cluster \n"
              f"({len(set(gene_cluster_df.cluster))} clusters, {len(q)} total genes)")
        cm = pd.DataFrame(cost_matrix, index=q.columns, columns=t.columns)
        plot_alignment(cm, avg_path, std_path)
        
        if is_time_numeric:
            # gene mapping only makes sense if both query and template are in numeric time
            plot_gene(query_tpms, template_tpms_inf, avg_path, showcase_gene, scale=True)

    if return_std:
        return avg_path, std_path
    return avg_path


# def plot_alignment(cost_matrix, best_path):
#     len_query, len_template = cost_matrix.shape
#     q = list(range(len(best_path)))
#     t = best_path

#     plt.rcParams['figure.figsize'] = [8, 6]
#     plt.plot(q, t, alpha=0.5)
#     plt.scatter(q, t, s=10, color="black")
#     plt.title("local alignment")
#     plt.ylabel("template")
#     plt.ylim(0, len_template-1)
#     plt.xlabel("query")
#     plt.xlim(0, len_query-1)
#     plt.show()


def plot_alignment(cost_matrix, best_path, std_path=None):
    if all_numeric(cost_matrix.index) and all_numeric(cost_matrix.columns):
        q = list2floats(cost_matrix.index)
        template_time = list2floats(cost_matrix.columns)
        t = [template_time[i] for i in best_path]
        
        # add diagonal
        start = min(t[0], q[0])
        end = max(t[-1], q[-1])
        plt.plot([start, end], [start, end], 'orange', alpha=0.2, ls='--')
        
        plt.ylim(t[0], t[-1])
        plt.xlim(q[0], q[-1])
    else:
        q = list(range(len(best_path)))
        t = best_path

        len_query, len_template = cost_matrix.shape
        plt.ylim(0, len_template-1)
        plt.xlim(0, len_query-1)
    
    plt.plot(q, t, alpha=0.5)
    plt.scatter(q, t, s=10, color="black")
    if std_path is not None:
        plt.errorbar(q, t, color="grey", yerr=std_path, alpha=0.4, fmt='none')
    
    plt.title("local alignment", fontsize=15)
    plt.ylabel("template time", fontsize=18)
    plt.xlabel("query time", fontsize=15)
    plt.show()


def plot_gene(query_tpms, template_tpms_inf, path, gene=None, scale=False):
    """
    Visualize annotated time vs inferred time for a specified gene.
    Requires that annotated time is numeric for both template and query.
    """
    if gene is None:
        gene = random.sample(list(query_tpms.index), 1)[0]
    
    x1 = list2floats(template_tpms_inf.columns)
    x2 = list2floats(query_tpms.columns)
    
    y1 = template_tpms_inf.loc[gene].to_list()
    y2 = query_tpms.loc[gene].to_list() 
    if scale:
        y1 = sklearn.preprocessing.scale(y1)
        y2 = sklearn.preprocessing.scale(y2)
    
    plt.scatter(x=x1, y=y1)
    plt.scatter(x=x2, y=y2)
    for n in range(len(x2)):
        m = path[n]
        plt.plot((x1[m], x2[n]), (y1[m], y2[n]), color = 'black', alpha=0.1, linestyle='--')

    plt.title(f"{gene} alignment", fontsize=15)
    plt.ylabel("normalized expression", fontsize=18)
    plt.xlabel("time", fontsize=15)
    start = min(x1[path[0]], x2[0])
    end = max(x1[path[-1]], x2[-1])
    a_bit = (end - start) * 0.05
    plt.xlim(start - a_bit, end + a_bit)
    plt.show()

    
def time_series_alignment(template, query, gene_cluster_df=None, tries=10, frac=0.2, cycles=1, top_frac=0.8, filter_frac=0.2, method="skip_worst", verbose=True, **kwargs):
    """
    Apply a local time series alignment of {query} to {template}.
    Returns the path (and standard deviation if {return_std} is True) for each cycle.
    The path contains the indeces of {template} columns best matching each {query} column.
    
    Uses a random {frac}tion of genes (per cluster of genes if {gene_cluster_df} is provided).
    This is repeated several {tries}, and averaged to obtain an alignment path.
    
    If {cycles} > 1, the alignments will then be used to bootstrap a subsequent alignments:
    
    First by measuring the ({scale}d) Goodness of Fit for each gene, 
    and discarding the worst {filter_frac}tion genes (per cluster) between cycles.
    Second, if {method} is 'use_best', the {top_frac}tion of remaining genes is subset, 
    else if {method} is 'discard_worst', all remaining genes are subset.
    Finally, the subset genes are used for the next cycle's average alignment 
    (the subset is recreated from all remaining genes each cycle).
    
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
    metric: string, optional
        distance metric to apply for the TSA. See scipy's cdist for options. Default is 'correlation'
    
    Bootstrap parameters
    --------------------
    cycles: int, optional
        number of alignment cycles to perform. Bootstrapping takes effect with cycle > 1. Default is 1
    top_frac: float, optional
        fraction of best genes to keep between cycles. Required `method="use_best"`. Default is 0.8
    filter_frac: float, optional
        fraction of worst genes to discard between cycles. Default is 0.2
    method: string, optional
        bootstrap gene filter method. 
        With "use_best", the top_fraction of remaining genes is used for bootstrapped alignments.
        With "skip_worst", all remaining genes are used for bootstrapped alignments.
        This variable is automatically set if `filter_frac` = 0 or `top_frac` = 1. Default is "use_best"
    scale: bool, optional
        wether to scale gene expression values before determining Goodness of Fit. Default is True
    
    Other parameters
    ----------------
    verbose: bool, optional
        wether to return additional messages on progress and gene usage. Default is True
    return_std: bool, optional
        wether to return standard deviation of the average alignment. Default is False
        standard deviations are shown in template time if columns are numeric, or template timepoints if columns are not.
    plot: bool, optional
        wether to plot the alignment (and showcase_gene for each alignment). Default is True
    showcase_gene: string, optional
        specify a gene name to plot after each alignment. Uses a random gene if None (default).
        
    Returns
    -------
    paths: list
        if cycles == 1, containing the template columns indeces best matching the query columns. Each column being a timepoint.
        if cycles >1, a list of lists with a path for each cycle.
        if return_std = True, returns tuple(s) containing the path and the standard deviation (both up and down).
    """
    if method not in ["skip_worst", "use_best"]:
        raise ValueError("`method` can be either 'skip_worst' or 'use_best'")
    if frac > 1 or frac < 0:
        raise ValueError("`frac` must be within [0,1]")
    if top_frac > 1 or top_frac < 0:
        raise ValueError("`top_frac` must be within [0,1]")
    if filter_frac > 1 or filter_frac < 0:
        raise ValueError("`filter_frac` must be within [0,1]")
    if filter_frac == 0 and top_frac == 1:
        raise ValueError("You must filter genes with `top_frac` and/or `filter_frac` to bootstrap")
    if filter_frac == 0:
        method = "use_best"
    elif top_frac == 1:
        method = "skip_worst"

    # filter kwargs for subfunctions
    alignment_kwargs = {}
    cluster_kwargs = {}
    if kwargs:
        keys = inspect.getfullargspec(avg_alignment).args
        alignment_kwargs = {k: kwargs[k] for k in keys if k in kwargs}
        keys = inspect.getfullargspec(top_cluster_genes).args
        cluster_kwargs = {k: kwargs[k] for k in keys if k in kwargs} 
    
    # match query and template index order
    if gene_cluster_df is None:
        overlapping_genes = list(set(template.index).intersection(query.index))
        template = template.loc[overlapping_genes]
        query = query.loc[overlapping_genes]
        # use all overlapping genes if no cluster information is provided
        gene_cluster_df = pd.DataFrame({
            "gene1": overlapping_genes, 
            "cluster": [0 for _ in  range(len(overlapping_genes))]
        }).set_index("gene1")
    else:
        template = template.loc[gene_cluster_df.index]
        query = query.loc[gene_cluster_df.index]
    
    # initial TSA
    if verbose and cycles > 1:
        print(f"Cycle 1, using all {len(gene_cluster_df)} genes.\n")
    path = avg_alignment(template, query, gene_cluster_df, tries=tries, frac=frac, **alignment_kwargs)
    if cycles == 1:
        return path
    
    # bootstrapped TSA's
    # return a list of average paths per cycle
    paths = [path]
    filt_gene_clusters = gene_cluster_df
    for n in range(cycles)[1:]:
        top_gene_clusters, filt_gene_clusters = top_cluster_genes(template, query, path, gene_cluster_df=filt_gene_clusters, top_frac=top_frac, filter_frac=filter_frac, **cluster_kwargs)
        
        # use the best fraction of genes, or all genes minus the worst fraction.
        cluster_method = top_gene_clusters if method == "use_best" else filt_gene_clusters
        if verbose:
            if method == "use_best":
                print(f"Cycle {n+1}, using the best {len(top_gene_clusters)} of {len(filt_gene_clusters)} genes.\n")
            else:
                print(f"Cycle {n+1}, using the best {len(filt_gene_clusters)} genes.\n")
        
        path = avg_alignment(template, query, cluster_method, tries=tries, frac=frac, **alignment_kwargs)
        paths.append(path)
    return paths
