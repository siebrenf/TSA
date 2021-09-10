import pandas as pd
import numpy as np
from qnorm import quantile_normalize


def tpm_normalization(
        tpms: pd.DataFrame,
        column_order: list,
        minimum_value: int = None,
) -> pd.DataFrame:
    """filter and order a tpm table, then quantile normalize and log transform"""
    bc = tpms[column_order]                       # filter & order samples
    if minimum_value:
        b4 = bc.shape[0]
        bc = bc[bc.max(axis=1) >= minimum_value]  # filter genes
        aft = b4 - bc.shape[0]
        print(f"Genes with TPM below {minimum_value}: {aft} of {b4} ({round(100*aft/b4,0)}%)")
    bc = quantile_normalize(bc, axis=1)           # normalize
    bc = np.log2(bc+1)                            # transform
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
