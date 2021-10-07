import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from matplotlib import pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
from tsa.utils import all_numeric

def gpr(
        time2samples: dict,
        template_tpms: pd.DataFrame,
        extended_timepoints: list,
        plot=False,
        verbose=True,
        run_n: int = None,
):
    """
    Infers GPRs for each gene in the template_tpms dataframe (combining replicates).

    Returns 2 numpy arrays with gene names as first column.
    Y_pred contains len(extended_timepoints) inferred timepoints between the first and last timepoint.
    scores contains the r^2 and sum of absolute differences for each GPR.
    """
    timepoints = list(time2samples)
    n_replicates = len(time2samples[timepoints[0]])
    n_genes = len(template_tpms)
    output_index = template_tpms.index
    if run_n:
        n_genes = min(run_n, n_genes)
        output_index = template_tpms.index[:n_genes]

    # A GP kernel can be specified as the sum of additive components using the sum operator,
    # so we can include a Matèrn component (Matern), an amplitude factor (ConstantKernel),
    # as well as an observation noise (WhiteKernel)
    # source: https://blog.dominodatalab.com/fitting-gaussian-process-models-python
    kernel = (
            Matern(length_scale=1, nu=5/2)
            + WhiteKernel(noise_level=1)
    )
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)

    # X:
    # n_features: 1 (time)
    # n_samples: number of timepoints
    X = np.repeat(range(0, len(timepoints)), n_replicates).reshape(-1, 1).astype(float)
    # # map the predictions to linear space (so the gpr works nicely)
    # X_pred = np.atleast_2d(np.linspace(np.min(X), np.max(X), len(extended_timepoints))).T
    
    # # map the predictions to the same space as the extended_timepoints (method 1)
    # divisor = max(timepoints)/X.max()
    # X_pred = np.atleast_2d([t/divisor for t in extended_timepoints]).T
    
    # # map the predictions dynamically, depending if the input it numeric
    # if all_numeric(timepoints) and all_numeric(extended_timepoints):
    #     # map the predictions to the same space as the extended_timepoints (method 2)
    #     xp = []
    #     for n in range(len(timepoints))[1:]:
    #         points = [t for t in extended_timepoints if t >= timepoints[n - 1] and t < timepoints[n]]
    #         xp.extend([n-1 + t/timepoints[n] for t in points])
    #     xp.append(float(n))  # add last point
    #     X_pred = np.atleast_2d(xp).T
    # else:
    #     # map the predictions to the same space as the GPR (linear space)
    #     X_pred = np.atleast_2d(np.linspace(np.min(X), np.max(X), len(extended_timepoints))).T

    # map the predictions dynamically, depending if the input it numeric
    # map the predictions to the same space as the GPR (linear space)
    X_pred = np.atleast_2d(np.linspace(np.min(X), np.max(X), len(extended_timepoints))).T
    if all_numeric(timepoints) and all_numeric(extended_timepoints):
        if min(extended_timepoints) < min(timepoints) or max(extended_timepoints) > max(timepoints):
            raise ValueError("inference only, no outference!")
        # map the predictions to the same space as the extended_timepoints (method 3)
        xp = []
        for n, tp in enumerate(extended_timepoints):
            if tp in timepoints:
                x = timepoints.index(tp)
            else:
                prev_tp = max([t for t in timepoints if t < tp])
                next_tp = min([t for t in timepoints if t > tp])
                dist_tp = (tp - prev_tp)/(next_tp - prev_tp)
                x = timepoints.index(prev_tp) + dist_tp
            xp.append(x)
        X_pred = np.atleast_2d(xp).T

    # y:
    # target values: expression scores per timepoint
    Y = template_tpms.to_numpy()
    Y_pred = np.zeros((n_genes, X_pred.shape[0]))

    # array with 2 columns: RMSE and the absolute sum of the expression differences
    scores = np.zeros((n_genes, 2))

    for n in range(n_genes):
        # progress
        if verbose and (n+1) % 100 == 0:
            print(f"{n+1}/{n_genes}", end="\r")

        # expression for 1 gene
        y = Y[n]

        # attempt to model the gene's expression pattern.
        # skip on a ConvergenceWarning
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=ConvergenceWarning)
            try:
                gpr.fit(X, y)
            except ConvergenceWarning:
                scores[n] = [np.NaN, np.NaN]
                continue
        
        if plot:
            y_pred, sigma = gpr.predict(X_pred, return_std=True)
        else:
            y_pred = gpr.predict(X_pred, return_std=False)
        Y_pred[n] = y_pred

        r2 = gpr.score(X, y)
        delta = np.abs(np.diff(y_pred)).sum()
        scores[n] = [r2, delta]

        if plot:
            plt.plot(X_pred, y_pred, 'b-', label='Prediction')
            plt.fill(np.concatenate([X_pred, X_pred[::-1]]),
                     np.concatenate([y_pred - 1.9600 * sigma,
                                    (y_pred + 1.9600 * sigma)[::-1]]),
                     alpha=.2, fc='b', ec='None', label='95% confidence interval')
            if n_replicates > 1:
                for r in range(n_replicates):
                    x_replicate = X[np.array(range(0, X.shape[0], n_replicates)) + r]
                    y_replicate = y[np.array(range(0, y.shape[0], n_replicates)) + r]
                    plt.plot(x_replicate, y_replicate, '.', alpha=.7, label=f'Observations replicate {r+1}')
            else:
                plt.plot(X, y, 'r.', markersize=10, label='Observations')
            
            # pretty stuff 
            plt.title(f"{template_tpms.index[n]}    "
                      f"R\u00b2: {round(r2, 2)}    "
                      f"|\u0394|: {round(delta, 2)}", 
                      fontsize=15)
            plt.xlabel("time", fontsize=15)
            plt.ylabel("normalized gene expression", fontsize=15)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
            plt.show()

    inference = pd.DataFrame(Y_pred, index=output_index, columns=extended_timepoints)
    scores = pd.DataFrame(scores, index=output_index, columns=["r2", "delta"])
    return inference, scores

# generate testdata
cols = ["tp1-1",  "tp1-2",  "tp2-1",  "tp2-2",  "tp3-1", "tp3-2"]
idx = ["gene 1", "gene 2"]
g1 = [1.798, 1.850, 2.919, 2.731, 1.983, 1.829]
g2 = [3.300, 3.493, 2.263, 2.754, 1.533, 1.623]
test_tpms = pd.DataFrame([g1, g2], index=idx, columns = cols)
test_time2samples = {0.0: ["tp1-1",  "tp1-2"], 45.0: ["tp2-1",  "tp2-2"], 135.0: ["tp3-1", "tp3-2"]}
test_timepoints = [0.0, 15.0, 30.0, 45.0, 75.0, 105.0, 135.0]
# test_timepoints = [0.0, 45/3, 2*45/3, 45.0, 45+(135-45)/3, 45+2*(135-45)/3, 135.0]
# test_timepoints = ["a", "a+1/3", "a+2/3", "b", "b+1/3","b+2/3", "c"]

# test function
inf, sco = gpr(test_time2samples, test_tpms, test_timepoints, plot=True, verbose=True, run_n=1)
inf