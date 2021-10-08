import numpy as np
import pandas as pd
import random
from qnorm import quantile_normalize


def tpm_normalization(
        tpms: pd.DataFrame,
        column_order: list = None,
        min_value: int = None,
        min_median_value: int = None,
        qnorm_target_dist: list = None,
        verbose=True,
) -> pd.DataFrame:
    """filter, (quantile) normalize and (log) transform a dataframe"""
    # filter samples
    if column_order is not None:
        bc = tpms[column_order]
    
    # filter genes
    if min_value is not None:
        b4 = bc.shape[0]
        bc = bc[bc.max(axis=1) >= min_value]
        aft = bc.shape[0]
        if b4 != aft and verbose:
            print(f"{b4-aft} genes with max below {min_value} TPM ({int(100*(b4-aft)/b4)}%)")
    if min_median_value is not None:
        b4 = bc.shape[0]
        bc = bc[bc.median(axis=1) > min_median_value]
        aft = bc.shape[0]
        if b4 != aft and verbose:
            print(f"{b4-aft} genes with median below or equal to {min_median_value} TPM ({int(100*(b4-aft)/b4)}%)")
    if verbose:
        print(f"{bc.shape[0]} genes, {bc.shape[1]} samples left after filtering")
    
    # normalize & transform
    if qnorm_target_dist:
        if len(bc) < len(qnorm_target_dist):
            # subsample distribution to number of genes
            qnorm_target_dist = random.sample(qnorm_target_dist, len(bc))
        elif len(bc) > len(qnorm_target_dist):
            raise NotImplementedError("Query cannot have more genes than target distribution")
    bc = quantile_normalize(bc, axis=1, target=qnorm_target_dist)  # normalize
    bc = np.log2(bc+1)                                             # transform
    return bc


def get_sample_info(samples: pd.DataFrame):
    # check its what we expect
    assert "time" in samples
    assert samples.index.name == "sample"
    assert samples.shape[1] == 1

    # chronological order of samples
    sample_order = samples.index.to_list()

    # samples per timepoint
    time2samples = dict()
    keys = samples.time.unique()
    for k in keys:
        v = samples[samples.time == k].index.to_list()
        time2samples[k] = v

    return sample_order, time2samples


def merge_replicates(df: pd.DataFrame, time2samples: dict, how="mean") -> pd.DataFrame:
    """merge replicate columns by averaging the values"""
    # TODO: merge with GPR?
    if how == "mean":
        for timepoint in time2samples:
            df[timepoint] = df[time2samples[timepoint]].mean(axis=1)
    elif how == "median":
        for timepoint in time2samples:
            df[timepoint] = df[time2samples[timepoint]].median(axis=1)
    else:
        raise ValueError("`how` can be 'mean' or 'median'")
    df = df.filter(items=time2samples.keys())
    return df
