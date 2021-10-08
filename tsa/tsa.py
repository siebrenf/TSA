import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import networkx as nx
from matplotlib import pyplot as plt
import random
import sklearn

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


def combine_paths(paths: list, inference_time: list = None, is_time_numeric=False, force_chonological=True):
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
    avg_path, std_path = combine_paths(paths, inference_time, is_time_numeric)
    
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

