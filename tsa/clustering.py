import pandas as pd
import sklearn
from sklearn.cluster import KMeans

from tsa.utils import subset_df


def cluster_genes(df, genes=None, n_clusters=None):
    """
    Clusters a gene expression dataframe with gene names as index by K-means clustering.
    Filters the dataframe for {genes} if provided.
    If no {n_clusters} is specified, genes/300 is used (minimum 10).
    Returns a dataframe with gene names as index and cluster id in column "cluster"
    """
    if genes is None:
        genes = list(df.index)
    else:
        genes = list(genes)
        df = df.loc[genes]

    if n_clusters is None:
        # select a number of clusters based on the number of genes
        n_clusters = max(10, int(len(genes)/300))

    # K-means clustering on TPMs
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
    kmeans.fit(df)
    k = kmeans.predict(df)

    gene_cluster_df = pd.DataFrame({"gene": genes, "cluster": k}).set_index("gene")
    return gene_cluster_df


def top_cluster_genes(template_tpms_inf, query_tpms, path, gene_cluster_df, top_frac=0.2, filter_frac=0, scale=True, verbose=False):
    # get 2 dataframe with desired genes and aligned timepoints
    genes = list(gene_cluster_df.index)
    template_Y = subset_df(template_tpms_inf, genes, path)
    query_Y = subset_df(query_tpms, genes)

    if scale:
        # scale y-axis per gene
        template_Y = pd.DataFrame(
            sklearn.preprocessing.scale(template_Y, axis=1),
            index=template_Y.index,
            columns=template_Y.columns
        )
        query_Y = pd.DataFrame(
            sklearn.preprocessing.scale(query_Y, axis=1),
            index=query_Y.index, 
            columns=query_Y.columns
        )

    # R^2: how close are the two time series? (argument order is arbitrary)
    scores = sklearn.metrics.r2_score(query_Y.T, template_Y.T, multioutput='raw_values')
    scores = pd.DataFrame({"gene": query_Y.index, "score": scores}).set_index("gene")

    # combine score and cluster info per gene
    subset = scores.merge(gene_cluster_df, left_index=True, right_index=True, how="right")
    subset.sort_values("score", ascending=False, inplace=True)
    if verbose:
        print("All genes per cluster")
        print("mean fit score:", round(subset.score.mean(), 3))
        s = subset.groupby("cluster").score.mean()
        g = subset.groupby("cluster").size()
        g.name = "n_genes"
        print(pd.concat([s, g], axis=1))
        print()

    filt_gene_cluster_df = gene_cluster_df
    if filter_frac:
        # drop the bottom fraction of each cluster
        keep = []
        for cluster in subset.cluster:
            subset_cluster = subset[subset.cluster==cluster]
            top_n = int(len(subset_cluster)*(1-filter_frac))
            keep.extend(list(subset_cluster.head(top_n).index))
        subset = subset[subset.index.isin(keep)]
        filt_gene_cluster_df = gene_cluster_df[gene_cluster_df.index.isin(keep)]
    
    if top_frac:
        # take the top fraction of each cluster
        subset = subset.groupby("cluster").apply(lambda x: x.head(int(len(x)*top_frac))).reset_index(drop=True)
    
    if verbose:
        print("Top genes per cluster")
        print("mean fit score:", round(subset.score.mean(), 3))
        s = subset.groupby("cluster").score.mean()
        g = subset.groupby("cluster").size()
        g.name = "n_genes"
        print(pd.concat([s, g], axis=1))
    
    subset = subset.index.to_list()
    top_gene_cluster_df = gene_cluster_df[gene_cluster_df.index.isin(subset)]
    
    return top_gene_cluster_df, filt_gene_cluster_df
