import pandas as pd
import sklearn

from tsa.tsa import avg_alignment

def subset_df(df, rows=None, columns=None, sort=True):
    """
    (efficiently) reconstruct a dataframe by row and/or column (name or number)
    
    rows: list of row/index names or numbers
    columns: list of column names or numbers
    """
    if rows:
        if set(rows).issubset(set(df.index)):
            # rows contains dataframe row names
            df = df[df.index.isin(rows)]
        else:
            # rows contains dataframe row numbers
            df = df.iloc[rows]
    if columns:
        if set(columns).issubset(set(df.columns)):
            # columns contains dataframe column names
            df = df[columns]
        else:
            # columns contains dataframe column numbers
            df = df[[df.columns[i] for i in columns]]
    if sort:
        df.sort_index(inplace=True)
    return df


def top_cluster_genes(template_tpms_inf, query_tpms, path, gene_cluster_df, frac=0.2, filter_frac=0, scale=True, verbose=False):
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
    scores = pd.DataFrame({"gene": query_Y.index, "score": scores})

    # combine score and cluster info per gene
    subset = scores.merge(gene_cluster_df, on="gene", how="right")
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
            keep.extend(list(subset_cluster.head(top_n).gene))
        subset = subset[subset.gene.isin(keep)]
        filt_gene_cluster_df = gene_cluster_df[gene_cluster_df.index.isin(keep)]
    
    if frac:
        # take the top fraction of each cluster
        subset = subset.groupby("cluster").apply(lambda x: x.head(int(len(x)*frac))).reset_index(drop=True)
    
    if verbose:
        print("Top genes per cluster")
        print("mean fit score:", round(subset.score.mean(), 3))
        s = subset.groupby("cluster").score.mean()
        g = subset.groupby("cluster").size()
        g.name = "n_genes"
        print(pd.concat([s, g], axis=1))
    
    subset = subset.gene.to_list()
    top_gene_cluster_df = gene_cluster_df[gene_cluster_df.index.isin(subset)]
    
    return top_gene_cluster_df, filt_gene_cluster_df


def bootstrap_alignment(template_tpms_inf, query_tpms, gene_cluster_df, cycles=3, tries=10, frac=0.2, filter_frac=0.2, method="skip_worst", verbose=True):
    """
    Repeat the avg_alignment {cycle} times, discarding {filter_frac} genes per cluster between cycles.
    
    cycle: number of TSAs. Bootstrapping takes effect with cycle > 1
    tries: tries per cycle
    frac: fraction of (best) genes to use per cluster
    filter_frac: fraction of (worst) genes to discard after each cycle
    """
    if method not in ["skip_worst", "use_best"]:
        raise ValueError("`method` can be either 'skip_worst' or 'use_best'")
    if frac > 1 or frac < 0:
        raise ValueError("`frac` must be within [0,1]")
    if filter_frac > 1 or filter_frac < 0:
        raise ValueError("`filter_frac` must be within [0,1]")
    if filter_frac == 0 and frac == 1:
        raise ValueError("You must filter genes with `frac` and/or `filter_frac` to bootstrap")
    if filter_frac == 0:
        method = "use_best"
    if frac == 1:
        method = "skip_worst"

    if verbose and cycles > 1:
        print(f"Cycle 1, using all {len(gene_cluster_df)} genes.\n")
    path = avg_alignment(template_tpms_inf, query_tpms, gene_cluster_df, tries=tries, frac=frac)
    if cycles == 1:
        return path

    # return a list of average paths per cycle
    paths = [path]
    filt_gene_clusters = gene_cluster_df
    for n in range(cycles)[1:]:
        top_gene_clusters, filt_gene_clusters = top_cluster_genes(template_tpms_inf, query_tpms, path, gene_cluster_df=filt_gene_clusters, frac=frac, filter_frac=filter_frac)
        
        # use the best fraction of genes, or all genes minus the worst fraction.
        cluster_method = top_gene_clusters if method == "use_best" else filt_gene_clusters
        if verbose:
            if method == "use_best":
                print(f"Cycle {n+1}, using the best {len(top_gene_clusters)} of {len(filt_gene_clusters)} genes.\n")
            else:
                print(f"Cycle {n+1}, using the best {len(filt_gene_clusters)} genes.\n")
        
        path = avg_alignment(template_tpms_inf, query_tpms, cluster_method, tries=tries, frac=frac)
        paths.append(path)
    return paths
