import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
from matplotlib import pyplot as plt


def get_cost_matrix(template_tpms, query_tpms, selected_genes, time2samples):
    template = template_tpms.loc[selected_genes]
    template = template.to_numpy(dtype=np.float64)

    query = query_tpms.loc[selected_genes]
    # average technical replicates
    for timepoint in time2samples:
        query[timepoint] = query[time2samples[timepoint]].mean(axis=1)
    query = query.filter(items=time2samples.keys())
    query = query.to_numpy()

    cost_matrix = cdist(template.T, query.T, metric='euclidean').T  # pairwise distance matrix
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


def plot_alignment(cost_matrix, best_path):
    len_query, len_template = cost_matrix.shape
    q = list(range(len(best_path)))
    t = best_path

    plt.rcParams['figure.figsize'] = [8, 6]
    plt.plot(q, t, alpha=0.5)
    plt.scatter(q, t, s=10, color="black")
    plt.title("local alignment")
    plt.ylabel("template")
    plt.ylim(0, len_template-1)
    plt.xlabel("query")
    plt.xlim(0, len_query-1)
    plt.show()

