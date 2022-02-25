import functools
import numpy as np
import pandas as pd
import pytask
import yaml

import scipy.optimize as opt

from functools import partial

from src.config import BLD
from src.analysis.benchmark_with_exponential import benchmark_exp
from src.analysis.weight_exp import no_weight_exp, prob_weight_6_exp


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


@pytask.mark.depends_on('../original_data/our_data.csv')
@pytask.mark.produces('../analysis/no_weight_optimization.csv')
def task_no_weight(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""

    dt = pd.read_csv(depends_on)
    alpha_init, a_init, beta_init, delta_init, gift_init = 0.003, 0.13, 1.16, 0.75, 5e-6
    stvale_spec = [alpha_init, a_init, gift_init, beta_init, delta_init]
    
    gamma_init_exp, k_init_exp, s_init_exp =  0.015645717, 1.69443, 3.69198
    k_scaler_exp, s_scaler_exp = 1e+16, 1e+6

    k_init_exp = k_init_exp/k_scaler_exp
    s_init_exp = s_init_exp/s_scaler_exp
    
    st_values_exp = [gamma_init_exp, k_init_exp, s_init_exp]


    st_valuesnoweight_exp = np.concatenate((st_values_exp,stvale_spec)) # starting values

    args = [dt.loc[dt['samplenw']==1].payoff_per_100, dt.loc[dt['samplenw']==1].gift_dummy, dt.loc[dt['samplenw']==1].delay_dummy,
        dt.loc[dt['samplenw']==1].delay_wks, dt.loc[dt['samplenw']==1].payoff_charity_per_100, dt.loc[dt['samplenw']==1].charity_dummy]

    sol = opt.curve_fit(no_weight_exp, 
                        args,
                        dt.loc[dt['samplenw']==1].buttonpresses_nearest_100,
                        st_valuesnoweight_exp)
    be56 = sol[0]
    se56 = np.sqrt(np.diagonal(sol[1]))

    final = {'estimates' : be56,
        'variances' : se56}

    final = pd.DataFrame(final)

    with open(produces, "w") as f:
        final.to_csv(f, index=False)


@pytask.mark.depends_on('../original_data/our_data.csv')
@pytask.mark.produces('../analysis/prob_weight_4_exp.csv')
def task_prob_weight_4_exp(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""
    # inputs
    dt = pd.read_csv(depends_on)
   
    gamma_init_exp, k_init_exp, s_init_exp =  0.015645717, 1.69443, 3.69198
    k_scaler_exp, s_scaler_exp = 1e+16, 1e+6

    k_init_exp /= k_scaler_exp
    s_init_exp /= s_scaler_exp
    
    st_values_exp = [gamma_init_exp, k_init_exp, s_init_exp]

    prob_weight_init = [0.2]
    st_valuesprobweight_exp = np.concatenate((st_values_exp,prob_weight_init))
    args = [dt.loc[dt['samplepr']==1].payoff_per_100, dt.loc[dt['samplepr']==1].weight_dummy, dt.loc[dt['samplepr']==1].prob]

    # funtion
    prob_weight_4_exp_partial = partial(prob_weight_6_exp, curv=1)
    # optimization
    sol = opt.curve_fit(prob_weight_4_exp_partial,
                    args,
                    dt.loc[dt['samplepr']==1].buttonpresses_nearest_100,
                    st_valuesprobweight_exp)
    be64 = sol[0] 
    se64 = np.sqrt(np.diagonal(sol[1])) 

    final = {'estimates' : be64,
        'variances' : se64}

    final = pd.DataFrame(final)

    with open(produces, "w") as f:
        final.to_csv(f, index=False)


@pytask.mark.depends_on('../original_data/our_data.csv')
@pytask.mark.produces('../analysis/prob_weight_5_exp.csv')
def task_prob_weight_5_exp(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""
    # inputs
    dt = pd.read_csv(depends_on)
   
    gamma_init_exp, k_init_exp, s_init_exp =  0.015645717, 1.69443, 3.69198
    k_scaler_exp, s_scaler_exp = 1e+16, 1e+6

    k_init_exp /= k_scaler_exp
    s_init_exp /= s_scaler_exp
    
    st_values_exp = [gamma_init_exp, k_init_exp, s_init_exp]

    prob_weight_init = [0.2]
    st_valuesprobweight_exp = np.concatenate((st_values_exp,prob_weight_init))
    args = [dt.loc[dt['samplepr']==1].payoff_per_100, dt.loc[dt['samplepr']==1].weight_dummy, dt.loc[dt['samplepr']==1].prob]

    # funtion
    prob_weight_5_exp_partial = partial(prob_weight_6_exp, curv=0.88)
    # optimization
    sol = opt.curve_fit(prob_weight_5_exp_partial,
                        args,
                        dt.loc[dt['samplepr']==1].buttonpresses_nearest_100,
                        st_valuesprobweight_exp)
    be65 = sol[0]
    se65 = np.sqrt(np.diagonal(sol[1])) 

    final = {'estimates' : be65,
        'variances' : se65}

    final = pd.DataFrame(final)

    with open(produces, "w") as f:
        final.to_csv(f, index=False)


@pytask.mark.depends_on('../original_data/our_data.csv')
@pytask.mark.produces('../analysis/prob_weight_6_exp.csv')
def task_prob_weight_6_exp(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""
    # inputs
    dt = pd.read_csv(depends_on)
   
    gamma_init_exp, k_init_exp, s_init_exp =  0.015645717, 1.69443, 3.69198
    k_scaler_exp, s_scaler_exp = 1e+16, 1e+6

    k_init_exp /= k_scaler_exp
    s_init_exp /= s_scaler_exp
    
    st_values_exp = [gamma_init_exp, k_init_exp, s_init_exp]

    prob_weight_init = [0.2]
    curv_init = [0.5]
    st_valuesprobweight6_exp = np.concatenate((st_values_exp,prob_weight_init,curv_init))

    args = [dt.loc[dt['samplepr']==1].payoff_per_100, dt.loc[dt['samplepr']==1].weight_dummy, dt.loc[dt['samplepr']==1].prob]

    # optimization
    sol = opt.curve_fit(prob_weight_6_exp,
                    args,
                    dt.loc[dt['samplepr']==1].buttonpresses_nearest_100,
                    st_valuesprobweight6_exp)
    be66 = sol[0]
    se66 = np.sqrt(np.diagonal(sol[1])) 

    final = {'estimates' : be66,
        'variances' : se66}

    final = pd.DataFrame(final)

    with open(produces, "w") as f:
        final.to_csv(f, index=False)
