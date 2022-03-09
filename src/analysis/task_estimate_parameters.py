"""Estimate the parameters of the model as well as other
relevant statistics (e.g. standard errors) from the data,
`nls_data.csv` under five scenarios for two types of cost
function assumptions using the same optimizers used by
Massimiliano and Nunnari (2021, :cite:`PozziNunnari`).

The results for each scenario and cost function are stored in
`.yaml` files named as `[scenario_name]_est_['cost_function].yaml'
in "estimated_parameters" folder.

"""
import numpy as np
import pandas as pd
import scipy.optimize as opt
import pytask
import yaml

from functools import partial

from src.config import BLD
from src.model_code.model_function import estimated_effort, sqrd_residuals_benchmark


def _create_inputs_exp(data):
    """Create inputs for `estimated_effort()` and `opt.curve_fit()`
    functions under exponential cost assumption.

    Args:
        data (pd.DataFrame): raw data for non-linear least squares estimation

    Returns:
        dict: keys are scenario names and values are dictionaries storing
            inputs under each scenario

    """

    gamma_init, k_init, s_init =  0.015645717, 1.69443, 3.69198
    k_scaler, s_scaler = 1e+16, 1e+6

    alpha_init, a_init, beta_init, delta_init, gift_init = 0.003, 0.13, 1.16, 0.75, 5e-6
    prob_weight_init = [0.2]
    curv_init = [0.5]
    
    piece_rates = data.loc[data['dummy1']==1].payoff_per_100
    no_weight_x_data = {'payoff_per_100': data.loc[data['samplenw']==1].payoff_per_100,
        'gift_dummy': data.loc[data['samplenw']==1].gift_dummy,
        'delay_dummy': data.loc[data['samplenw']==1].delay_dummy,
        'delay_wks': data.loc[data['samplenw']==1].delay_wks,
        'payoff_charity_per_100': data.loc[data['samplenw']==1].payoff_charity_per_100,
        'charity_dummy': data.loc[data['samplenw']==1].charity_dummy
    }
    prob_weight_x_data = {'payoff_per_100': data.loc[data['samplepr']==1].payoff_per_100,
        'weight_dummy': data.loc[data['samplepr']==1].weight_dummy,
        'prob': data.loc[data['samplepr']==1].prob
    }

    benchmark = {
        "params_init": [gamma_init, k_init/k_scaler, s_init/s_scaler],
        "x_data": piece_rates,
        "buttonpresses": data.loc[data['dummy1']==1].buttonpresses_nearest_100
        
    }
    no_weight = {
        "params_init": np.concatenate((
            [gamma_init, k_init/k_scaler, s_init/s_scaler], 
            [alpha_init, a_init, gift_init, beta_init, delta_init]
            )),
        "x_data": no_weight_x_data,
        "buttonpresses": data.loc[data['samplenw']==1].buttonpresses_nearest_100
        
    }
    prob_weight_lin_curv = {
        "params_init": np.concatenate((
            [gamma_init, k_init/k_scaler, s_init/s_scaler], 
            prob_weight_init
            )),
        "x_data": prob_weight_x_data,
        "buttonpresses": data.loc[data['samplepr']==1].buttonpresses_nearest_100
    }
    prob_weight_conc_curv = {
        "params_init": np.concatenate((
            [gamma_init, k_init/k_scaler, s_init/s_scaler], 
            prob_weight_init
            )),
        "x_data": prob_weight_x_data,
        "buttonpresses": data.loc[data['samplepr']==1].buttonpresses_nearest_100
    }
    prob_weight_est_curv = {
        "params_init": np.concatenate((
            [gamma_init, k_init/k_scaler, s_init/s_scaler], 
            prob_weight_init, curv_init
            )),
        "x_data": prob_weight_x_data,
        "buttonpresses": data.loc[data['samplepr']==1].buttonpresses_nearest_100
    }

    out = {
        "benchmark": benchmark,
        "no_weight": no_weight,
        "prob_weight_lin_curv": prob_weight_lin_curv,
        "prob_weight_conc_curv": prob_weight_conc_curv,
        "prob_weight_est_curv": prob_weight_est_curv
    }

    return out


def _create_inputs_pow(data):
    """Create inputs for `estimated_effort()` and `opt.curve_fit()`
    functions under power cost assumption.

    Args:
        data (pd.DataFrame): raw data for non-linear least squares estimation

    Returns:
        dict: keys are the input names and values are the inputs

    """

    gamma_init, k_init, s_init =  19.8117987, 1.66306e-10, 7.74996
    k_scaler, s_scaler = 1e+57, 1e+6

    alpha_init, a_init, beta_init, delta_init, gift_init = 0.003, 0.13, 1.16, 0.75, 5e-6
    prob_weight_init = [0.2]
    curv_init = [0.5]

    piece_rates = data.loc[data['dummy1']==1].payoff_per_100
    no_weight_x_data = {'payoff_per_100': data.loc[data['samplenw']==1].payoff_per_100,
        'gift_dummy': data.loc[data['samplenw']==1].gift_dummy,
        'delay_dummy': data.loc[data['samplenw']==1].delay_dummy,
        'delay_wks': data.loc[data['samplenw']==1].delay_wks,
        'payoff_charity_per_100': data.loc[data['samplenw']==1].payoff_charity_per_100,
        'charity_dummy': data.loc[data['samplenw']==1].charity_dummy
    }
    prob_weight_x_data = {'payoff_per_100': data.loc[data['samplepr']==1].payoff_per_100,
        'weight_dummy': data.loc[data['samplepr']==1].weight_dummy,
        'prob': data.loc[data['samplepr']==1].prob
    }

    benchmark = {
        "params_init": [gamma_init, k_init/k_scaler, s_init/s_scaler],
        "x_data": piece_rates,
        "log_buttonpresses": data.loc[data['dummy1']==1].logbuttonpresses_nearest_100
        
    }
    no_weight = {
        "params_init": np.concatenate((
            [gamma_init, k_init/k_scaler, s_init/s_scaler], 
            [alpha_init, a_init, gift_init, beta_init, delta_init]
            )),
        "x_data": no_weight_x_data,
        "log_buttonpresses": data.loc[data['samplenw']==1].logbuttonpresses_nearest_100
        
    }
    prob_weight_lin_curv = {
        "params_init": np.concatenate((
            [gamma_init, k_init/k_scaler, s_init/s_scaler], 
            prob_weight_init
            )),
        "x_data": prob_weight_x_data,
        "log_buttonpresses": data.loc[data['samplepr']==1].logbuttonpresses_nearest_100
    }
    prob_weight_conc_curv = {
        "params_init": np.concatenate((
            [gamma_init, k_init/k_scaler, s_init/s_scaler], 
            prob_weight_init
            )),
        "x_data": prob_weight_x_data,
        "log_buttonpresses": data.loc[data['samplepr']==1].logbuttonpresses_nearest_100
    }
    prob_weight_est_curv = {
        "params_init": np.concatenate((
            [gamma_init, k_init/k_scaler, s_init/s_scaler], 
            prob_weight_init, curv_init
            )),
        "x_data": prob_weight_x_data,
        "log_buttonpresses": data.loc[data['samplepr']==1].logbuttonpresses_nearest_100
    }

    out = {
        "benchmark": benchmark,
        "no_weight": no_weight,
        "prob_weight_lin_curv": prob_weight_lin_curv,
        "prob_weight_conc_curv": prob_weight_conc_curv,
        "prob_weight_est_curv": prob_weight_est_curv
    }

    return out


@pytask.mark.depends_on(BLD / "data" / "nls_data.csv")
@pytask.mark.parametrize(
    "produces, scenario",
    [(BLD / "analysis" / "estimated_parameters"/ "benchmark_est_exp.yaml", "benchmark"), 
    (BLD / "analysis" / "estimated_parameters"/ "no_weight_est_exp.yaml", "no_weight"),
    (BLD / "analysis" / "estimated_parameters"/ "weight_lin_curv_est_exp.yaml", "prob_weight_lin_curv"), 
    (BLD / "analysis" / "estimated_parameters"/ "weight_conc_curv_est_exp.yaml", "prob_weight_conc_curv"),
    (BLD / "analysis" / "estimated_parameters"/ "weight_est_curv_est_exp.yaml", "prob_weight_est_curv")],
)
def task_estimate_params_exp(depends_on, produces, scenario):
    """Estimate and store the parameters of the model under five scenarios
    for exponential cost function.

    """

    dt = pd.read_csv(depends_on)

    inputs = _create_inputs_exp(dt)

    x_data = inputs[scenario]["x_data"]        
    buttonpresses = inputs[scenario]["buttonpresses"]
    params_init = inputs[scenario]["params_init"]

    _partial = partial(estimated_effort, scenario = scenario)

    sol = opt.curve_fit(_partial,
        x_data,
        buttonpresses,
        params_init)
    est = sol[0]
    std_errors = np.sqrt(np.diagonal(sol[1]))

    final = {
        'estimates' : est.tolist(),
        'std dev' : std_errors.tolist()
    }
    with open(produces, "w") as y:
        yaml.dump(final, y)


@pytask.mark.depends_on(BLD / "data" / "nls_data.csv")
@pytask.mark.parametrize(
    "produces, scenario",
    [(BLD / "analysis" / "estimated_parameters"/ "benchmark_est_pow.yaml", "benchmark"), 
    (BLD / "analysis" / "estimated_parameters"/ "no_weight_est_pow.yaml", "no_weight"),
    (BLD / "analysis" / "estimated_parameters"/ "weight_lin_curv_est_pow.yaml", "prob_weight_lin_curv"), 
    (BLD / "analysis" / "estimated_parameters"/ "weight_conc_curv_est_pow.yaml", "prob_weight_conc_curv"),
    (BLD / "analysis" / "estimated_parameters"/ "weight_est_curv_est_pow.yaml", "prob_weight_est_curv")],
)
def task_estimate_params_pow(depends_on, produces, scenario):
    """Estimate and store the parameters of the model under five scenarios
    for power cost function.

    """

    dt = pd.read_csv(depends_on)

    inputs = _create_inputs_pow(dt)

    x_data = inputs[scenario]["x_data"]        
    log_buttonpresses = inputs[scenario]["log_buttonpresses"]
    params_init = inputs[scenario]["params_init"]

    _partial = partial(estimated_effort, scenario = scenario)

    sol = opt.curve_fit(_partial,
        x_data,
        log_buttonpresses,
        params_init)
    est = sol[0]
    std_errors = np.sqrt(np.diagonal(sol[1]))

    final = {'estimates' : est.tolist(),
        'std dev' : std_errors.tolist(),
        'min obj func' : (np.sum((_partial(x_data, *est)-log_buttonpresses)**2)).tolist()}
    with open(produces, "w") as y:
        yaml.dump(final, y)


@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'benchmark_est_pow_alt1.yaml')
def task_estimate_params_benchmark_pow_alt1(depends_on, produces):
    """Estimate the parameters of the model using an alternative
    optimizer, opt.least_squares() for 'benchmark' scenario under
    power cost assumption.
    
    """

    dt = pd.read_csv(depends_on)

    inputs = _create_inputs_pow(dt)

    x_data = np.array(inputs["benchmark"]["x_data"])        
    log_buttonpresses = np.array(inputs["benchmark"]["log_buttonpresses"])
    params_init = inputs["benchmark"]["params_init"]

    sol = opt.least_squares(
        lambda params: sqrd_residuals_benchmark(params, xdata=x_data, logbuttonpresses=log_buttonpresses, optimizer='opt.least_squares'),
        params_init,
        xtol=1e-15,
        ftol=1e-15,
        gtol=1e-15,
        method='lm')
    est = sol.x              
    
    final = {
        'estimates' : est.tolist(),
        'min obj func' : (sqrd_residuals_benchmark(est, xdata=x_data, logbuttonpresses=log_buttonpresses, optimizer='opt.minimze')).tolist()
    }
    with open(produces, "w") as y:
        yaml.dump(final, y)


@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'benchmark_est_pow_alt2.yaml')
def task_estimate_params_benchmark_pow_alt2(depends_on, produces):
    """Estimate the parameters of the model using an alternative
    optimizer, opt.minimize() for 'benchmark' scenario under
    power cost assumption.
    
    """
    dt = pd.read_csv(depends_on)

    inputs = _create_inputs_pow(dt)

    x_data = np.array(inputs["benchmark"]["x_data"])        
    log_buttonpresses = np.array(inputs["benchmark"]["log_buttonpresses"])
    params_init = inputs["benchmark"]["params_init"]

    sol = opt.minimize(
        lambda params: sqrd_residuals_benchmark(params, xdata=x_data, logbuttonpresses=log_buttonpresses, optimizer='opt.minimize'),
        params_init,
        method='Nelder-Mead',
        options={'maxiter': 2500})
    est = sol.x                  
    
    final = {'estimates' : est.tolist(),
        'min obj func' : (sqrd_residuals_benchmark(est, xdata=x_data, logbuttonpresses=log_buttonpresses, optimizer='opt.minimze')).tolist()
    }
    with open(produces, "w") as y:
        yaml.dump(final, y)
