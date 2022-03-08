"""GENERAL DOCSTRING

"""
import numpy as np
import pandas as pd
import scipy.optimize as opt
import pytask
import yaml

from functools import partial

from src.config import SRC
from src.config import BLD
from src.analysis.utils import create_inputs
from src.analysis.funcs_with_exp_cost import benchmark_exp, no_weight_exp, prob_weight_exp

gamma_init_exp, k_init_exp, s_init_exp =  0.015645717, 1.69443, 3.69198
k_scaler_exp, s_scaler_exp = 1e+16, 1e+6
k_init_exp = k_init_exp/k_scaler_exp
s_init_exp = s_init_exp/s_scaler_exp

st_values_exp = [gamma_init_exp, k_init_exp, s_init_exp]

alpha_init, a_init, beta_init, delta_init, gift_init = 0.003, 0.13, 1.16, 0.75, 5e-6

stvale_spec = [alpha_init, a_init, gift_init, beta_init, delta_init]

st_valuesnoweight_exp = np.concatenate((st_values_exp,stvale_spec)) # starting values

prob_weight_init = [0.2]
st_valuesprobweight_exp = np.concatenate((st_values_exp,prob_weight_init))

curv_init = [0.5]
st_valuesprobweight6_exp = np.concatenate((st_valuesprobweight_exp,curv_init))




 
@pytask.mark.depends_on(BLD / "data" / "nls_data.csv")
@pytask.mark.produces(BLD / "analysis" / "est_benchmark_exp.yaml")
def task_opt_benchmark_exp(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result.
    
    """
    dt = pd.read_csv(depends_on)

    sol = opt.curve_fit(benchmark_exp,
                    dt.loc[dt['dummy1']==1].payoff_per_100,
                    dt.loc[dt['dummy1']==1].buttonpresses_nearest_100,
                    st_values_exp)
    be54 = sol[0]                      # sol[0] is the array containing our estimates
    se54 = np.sqrt(np.diagonal(sol[1]))  # sol[1] is a 3x3 variance-covariance matrix of our estimates

    final = {'estimates' : be54.tolist(),
        'std dev' : se54.tolist()}

    #final = pd.DataFrame(final)

    #with open(produces, "w") as f:
    #    final.to_csv(f, index=False)
    with open(produces, "w") as y:
        yaml.dump(final, y)


@pytask.mark.depends_on(BLD / "data" / "nls_data.csv")
@pytask.mark.produces(BLD / "analysis" / "est_noweight_exp.yaml")
def task_opt_noweight_exp(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result.
    
    """
    dt = pd.read_csv(depends_on)

    input = create_inputs(dt)
    args = input['samplenw']

    sol = opt.curve_fit(no_weight_exp, 
                        args,
                        dt.loc[dt['samplenw']==1].buttonpresses_nearest_100,
                        st_valuesnoweight_exp)
    be56 = sol[0]
    se56 = np.sqrt(np.diagonal(sol[1]))

    final = {'estimates' : be56.tolist(),
        'std dev' : se56.tolist()}

    #final = pd.DataFrame(final)

    #with open(produces, "w") as f:
    #   final.to_csv(f, index=False)
    with open(produces, "w") as y:
        yaml.dump(final, y)


@pytask.mark.depends_on(BLD / "data" / "nls_data.csv")
@pytask.mark.produces(BLD / "analysis" / "est_weight_exp_lin_curv.yaml")
def task_opt_weight_exp_lin_curv(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result.
    
    """
    dt = pd.read_csv(depends_on)

    input = create_inputs(dt)
    args = input['samplepr']

    prob_weight_4_exp_partial = partial(prob_weight_exp, curv=1)
    
    sol = opt.curve_fit(prob_weight_4_exp_partial,
                    args,
                    dt.loc[dt['samplepr']==1].buttonpresses_nearest_100,
                    st_valuesprobweight_exp)
    be64 = sol[0] 
    se64 = np.sqrt(np.diagonal(sol[1])) 

    final = {'estimates' : be64.tolist(),
        'std dev' : se64.tolist()}

    #final = pd.DataFrame(final)

    #with open(produces, "w") as f:
    #    final.to_csv(f, index=False)
    with open(produces, "w") as y:
        yaml.dump(final, y)


@pytask.mark.depends_on(BLD / "data" / "nls_data.csv")
@pytask.mark.produces(BLD / "analysis" / "est_weight_exp_conc_curv.yaml")
def task_opt_weight_exp_conc_curv(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result.
    
    """
    dt = pd.read_csv(depends_on)

    input = create_inputs(dt)
    args = input['samplepr']

    prob_weight_5_exp_partial = partial(prob_weight_exp, curv=0.88)
    
    sol = opt.curve_fit(prob_weight_5_exp_partial,
                        args,
                        dt.loc[dt['samplepr']==1].buttonpresses_nearest_100,
                        st_valuesprobweight_exp)
    be65 = sol[0]
    se65 = np.sqrt(np.diagonal(sol[1])) 

    final = {'estimates' : be65.tolist(),
        'std dev' : se65.tolist()}

    #final = pd.DataFrame(final)

    #with open(produces, "w") as f:
    #    final.to_csv(f, index=False)
    with open(produces, "w") as y:
        yaml.dump(final, y)


@pytask.mark.depends_on(BLD / "data" / "nls_data.csv")
@pytask.mark.produces(BLD / "analysis" / "est_weight_exp_est_curv.yaml")
def task_opt_weight_exp_est_curv(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result.
    
    """
    dt = pd.read_csv(depends_on)

    input = create_inputs(dt)
    args = input['samplepr']

    sol = opt.curve_fit(prob_weight_exp,
                    args,
                    dt.loc[dt['samplepr']==1].buttonpresses_nearest_100,
                    st_valuesprobweight6_exp)
    be66 = sol[0]
    se66 = np.sqrt(np.diagonal(sol[1])) 

    final = {'estimates' : be66.tolist(),
        'std dev' : se66.tolist()}

    #final = pd.DataFrame(final)

    #with open(produces, "w") as f:
    #    final.to_csv(f, index=False)
    with open(produces, "w") as y:
        yaml.dump(final, y)
