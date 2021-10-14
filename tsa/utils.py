from typing import Iterable


def isfloat(query):
    try:
        float(query)
    except ValueError:
        return False
    return True


def all_numeric(query: Iterable) -> bool:
    if not isinstance(query, Iterable):
        query = list(query)

    return all(isfloat(n) for n in query)


def list2floats(lst):
    if all_numeric(lst):
        lst = [float(t) for t in lst]
    return lst


def _str_inf_ts(timepoints, n):
    t_extended = []
    for t in timepoints:
        for dec in range(n):
            if dec == 0:
                t_extended.append(f"{t}")
            else:
                t_extended.append(f"{t}+{dec}/{n}")
    return t_extended


def _flt_inf_ts(timepoints, n):
    t_extended = []
    timepoints = [round(float(t), 2) for t in timepoints]
    for t in range(len(timepoints)):
        curr_timepoint = timepoints[t]
        diff_time = 0
        if t < len(timepoints)-1:
            next_timepoint = timepoints[t+1]
            diff_time = next_timepoint-curr_timepoint
        for dec in range(n):
            t_extended.append(round(curr_timepoint + diff_time*(dec/n), 2))
    return t_extended


def inference_timeseries(timepoints: list, n: int = 10) -> list:
    """
    Extends the given list of timepoints to n points between for each original point, up to the final point.

    Returns the extended list of named timepoints with added time.

    e.g. func([1, 2, 3], n=3) -> [1, 1.33, 1.66, 2, 2.33, 2.66, 3]
    e.g. func(["a","b","c"], n=3) -> ["a", "a+1/3", "a+2/3", "b", "b+1/3", "b+2/3", "c"]
    """
    if all_numeric(timepoints):
        t_extended = _flt_inf_ts(timepoints, n)
    else:
        t_extended = _str_inf_ts(timepoints, n)

    # remove the points after the last real timepoint
    total_timepoints = len(timepoints) * n - n + 1
    t_extended = t_extended[:total_timepoints]

    return t_extended

# print(inference_timeseries([1, 2, 3], n=3))
# print(inference_timeseries(["a", "b", "c"], n=3))


def subset_df(df, rows=None, columns=None, sort=True):
    """
    (efficiently) reconstruct a dataframe by row and/or column (name or number)
    
    rows: list of row/index names or numbers
    columns: list of column names or numbers
    """
    if rows:
        if set(rows).issubset(set(df.index)):
            # rows contains dataframe row names
            df = df.loc[rows]
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
