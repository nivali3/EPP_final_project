import numpy as np
import pandas as pd
import pytask
import yaml

import scipy.optimize as opt

from src.config import BLD
from src.analysis.benchmark_with_exponential import benchmark_exp


# fix depends on 
@pytask.mark.depends_on('../original_data/our_data.csv')
@pytask.mark.produces('../analysis/least_squares_optimization.csv')
def task_least_squares(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""

    gamma_init_exp, k_init_exp, s_init_exp =  0.015645717, 1.69443, 3.69198
    k_scaler_exp, s_scaler_exp = 1e+16, 1e+6

    k_init_exp = k_init_exp/k_scaler_exp
    s_init_exp = s_init_exp/s_scaler_exp
    
    st_values_exp = [gamma_init_exp, k_init_exp, s_init_exp]
    


    dt = pd.read_csv(depends_on)
    sol = opt.curve_fit(benchmark_exp,
                    dt.loc[dt['dummy1']==1].payoff_per_100,
                    dt.loc[dt['dummy1']==1].buttonpresses_nearest_100,
                    st_values_exp)

    

    be54 = sol[0]                      # sol[0] is the array containing our estimates
    se54 = np.sqrt(np.diagonal(sol[1]))  # sol[1] is a 3x3 variance-covariance matrix of our estimates

    final = {'estimates' : be54,
        'variances' : se54}

    final = pd.DataFrame(final)

    with open(produces, "w") as f:
        final.to_csv(f, index=False)


# Find the solution to the problem by non-linear least squares 
