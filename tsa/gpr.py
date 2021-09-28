import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from matplotlib import pyplot as plt


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
    n_genes = template_tpms.shape[0]

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
    # map the timeseries to linear space (so the gpr works nicely)
    X = np.repeat(range(0, len(timepoints)), n_replicates).reshape(-1, 1).astype(float)
    X_pred = np.atleast_2d(np.linspace(np.min(X), np.max(X), len(extended_timepoints))).T
    
    # X = np.repeat(timepoints, n_replicates).reshape(-1, 1)
    # X_pred = np.atleast_2d(extended_timepoints).T

    # X = np.repeat(range(0, len(timepoints)), n_replicates).reshape(-1, 1).astype(float)
    # divisor = max(timepoints)/X.max()
    # X_pred = np.atleast_2d([t/divisor for t in extended_timepoints]).T
    
    # X = np.repeat(range(0, len(timepoints)), n_replicates).reshape(-1, 1).astype(float)
    # xp = []
    # for n in range(len(timepoints))[1:]:
    #     npoints = len([t for t in extended_timepoints if t >= timepoints[n-1] and t < timepoints[n]])
    #     xp.extend(list(np.linspace(n-1, n, npoints, endpoint=False)))
    # xp.append(float(n))  # add last point
    # X_pred = np.atleast_2d(xp).T
    
    # X = np.repeat(range(0, len(timepoints)), n_replicates).reshape(-1, 1).astype(float)
    # xp = []
    # for n in range(len(timepoints))[1:]:
    #     points = [t for t in extended_timepoints if t >= timepoints[n - 1] and t < timepoints[n]]
    #     xp.extend([n-1 + t/timepoints[n] for t in points])
    # xp.append(float(n))  # add last point
    # X_pred = np.atleast_2d(xp).T
    
    # X = np.repeat(range(0, len(timepoints)), n_replicates).reshape(-1, 1).astype(float)
    # if all_numeric(timepoints) and all_numeric(extended_timepoints):
    #     # map the predictions to the same space as the extended_timepoints
    #     xp = []
    #     for n in range(len(timepoints))[1:]:
    #         points = [t for t in extended_timepoints if t >= timepoints[n - 1] and t < timepoints[n]]
    #         xp.extend([n-1 + t/timepoints[n] for t in points])
    #     xp.append(float(n))  # add last point
    #     X_pred = np.atleast_2d(xp).T
    # else:
    #     # map the predictions to the same space as the GPR (linear space)
    #     X_pred = np.atleast_2d(np.linspace(np.min(X), np.max(X), len(extended_timepoints))).T

    # y:
    # target values: expression scores per timepoint
    Y = template_tpms.to_numpy()
    Y_pred = np.zeros((n_genes, X_pred.shape[0]))

    # array with 2 columns: RMSE and the absolute sum of the expression differences
    scores = np.zeros((n_genes, 2))

    for n in range(n_genes):
        # progress
        if verbose and n+1 % 100 == 0:
            print(f"{n+1}/{n_genes}", end="\r")

        # expression for 1 gene
        y = Y[n]

        gpr.fit(X, y)
        y_pred = gpr.predict(X_pred)
        Y_pred[n] = y_pred

        r2 = gpr.score(X, y)
        delta = np.abs(np.diff(y_pred)).sum()
        scores[n] = [r2, delta]

        if plot:
            _, sigma = gpr.predict(X_pred, return_std=True)

            print(f"R^2: {r2}")
            print(f"delta: {delta}")

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

            gene = template_tpms.index[n]
            plt.title(gene)
            plt.xlabel("time")
            plt.ylabel("normalized gene expression")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()

        if run_n and n >= run_n-1:
            break

    inference = pd.DataFrame(Y_pred, index=template_tpms.index, columns=extended_timepoints)
    scores = pd.DataFrame(scores, index=template_tpms.index, columns=["r2", "delta"])
    return inference, scores

# _ = gpr(time2samples, template_tpms, extended_timepoints, plot=True, verbose=False, run_n=1)
