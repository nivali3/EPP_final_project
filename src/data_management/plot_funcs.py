"""Auxiliary functions for generating plots corresponding to
figures 3, 4(a), 4(b), 4(c) of the paper.

"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


def data_plot_fig_3(dt):
    """Create dataset suitable for replicating Figure 3 of the
    original paper.

    Args:
        dt (dataset): initial data containing actual efforts and
        treatment identifiers

    Returns:
        (pd.DataFrame): dataset of shape m x 3, where m is the
            number of treatments. The columns includes the treatment
            name, the sample mean effort, the size of the upper bound
            of the 95% confidence interval for the sample mean.

    """

    frame = pd.DataFrame(data=dt["treatmentname"].unique(), columns=["treatmentname"])
    intervals = []
    means = []
    for i in frame["treatmentname"]:
        sample_mean = dt[dt["treatmentname"] == i]["buttonpresses"].mean()
        means.append(sample_mean)
        n_size = len(dt[dt["treatmentname"] == i])
        t_critical = stats.t.ppf(q=0.975, df=(n_size - 1))
        sample_std = dt[dt["treatmentname"] == i]["buttonpresses"].std(ddof=1)
        upper_bound = t_critical * (sample_std / math.sqrt(n_size))
        intervals.append(upper_bound)
    frame["means"], frame["upper bound ci"] = means, intervals
    return frame.sort_values("means")


def plot_CDF(dt, treat_names):  # treat_names is a list from treatmentname column
    """Plot Cumulative Distribution Functions to replicate Figures
    4 a) b) c) of the original paper.

    Args:
        dt (dataset): initial data containing treatment names,
            actual efforts, and experts forecasted efforts.
        treat_names (list): list of treatment names for which we want
            to plot the CDF of effort

    Returns:
        (plot): plot of the CDF of participants' effort task in each
            specified treatment.

    """

    line_styles = ["-", "--", "-.", ":", "-."]
    colors = ["blue", "red", "green", "orange", "cyan"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, j in enumerate(treat_names):
        treat_outcome = dt[dt["treatmentname"] == j]["buttonpresses"]
        x = np.sort(treat_outcome)  # sort the data in ascending order
        y = np.arange(len(treat_outcome)) / len(
            treat_outcome
        )  # get the cdf values of y

        ax.plot(x, y, color=colors[i], ls=line_styles[i])
    ax.set_xlim([0, 3100])
    ax.set_xlabel("Points in Task")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("CDF of MTurk Workers' effort")
    ax.legend(treat_names, loc="best", bbox_to_anchor=(0.8, -0.2), ncol=2)
