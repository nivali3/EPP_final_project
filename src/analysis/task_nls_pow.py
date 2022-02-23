import numpy as np
import pandas as pd
import pytask

import scipy.optimize as opt

from src.config import BLD
from src.analysis.benchmark_with_power import benchmark_power


# fix depends on 
@pytask.mark.depends_on('../original_data/our_data.csv')
@pytask.mark.produces('../analysis/least_squares_optimization_power.csv')
def task_least_squares_power(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""

    gamma_init_power, k_init_power, s_init_power =  19.8117987, 1.66306e-10, 7.74996
    k_scaler_power, s_scaler_power = 1e+57,1e+6

    k_init_power = k_init_power/k_scaler_power
    s_init_power = s_init_power/s_scaler_power
    
    st_values_power = [gamma_init_power, k_init_power, s_init_power]


    dt = pd.read_csv(depends_on)
    sol = opt.curve_fit(benchmark_power,
                    dt.loc[dt['dummy1']==1].payoff_per_100,
                    dt.loc[dt['dummy1']==1].logbuttonpresses_nearest_100,
                    st_values_power)
    bp52 = sol[0]                       # sol[0] is the array containing our estimates
    sp52 = np.sqrt(np.diagonal(sol[1])) # sol[1] is a 3x3 variance-covariance matrix of our estimates

    final = {'estimates' : bp52,
        'variances' : sp52}

    final = pd.DataFrame(final)

    with open(produces, "w") as f:
        final.to_csv(f, index=False)




# Find the solution to the problem by non-linear least squares 