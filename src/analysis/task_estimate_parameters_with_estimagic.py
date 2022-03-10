"""Estimate the parameters of the model as well as standart errors
using `minimize()` and `bootstrap()` functions from estimagic package
(2021, :cite:`Gabler2021`) with three different algorithms:
- 'scipy_neldermead';
- 'scipy_ls_trf';
- 'scipy_ls_dogbox'.

The results for each scenario, cost function, and algorithm are stored
in `.yaml` files named as
`[scenario_name]_est_['cost_function]_estimagic_[algorithm_name].yaml'
in "estimated_parameters_estimagic" folder.

"""
import numpy as np
import pandas as pd
import pytask
import yaml

from functools import partial
from estimagic import minimize
from estimagic.inference import bootstrap

from src.config import BLD
from src.model_code.model_function_estimagic import sqrd_residuals_estimagic


def _create_inputs_exp(data):
    """Create inputs for `sqrd_residuals_estimagic()`, `minimize()`, and
    `bootstrap()` functions under exponential cost assumption.

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

    benckmark_params = pd.DataFrame([gamma_init, k_init/k_scaler, s_init/s_scaler], columns=["value"])
    benckmark_params["soft_lower_bound"] = 1e-123
    benckmark_params["soft_upper_bound"] = 40

    no_weight_params = pd.DataFrame(np.concatenate((
        [gamma_init, k_init/k_scaler, s_init/s_scaler], 
        [alpha_init, a_init, gift_init, beta_init, delta_init]
        )), 
        columns=["value"])
    no_weight_params["soft_lower_bound"] = 1e-123
    no_weight_params["soft_upper_bound"] = 40

    prob_weight_lin_curv_params = pd.DataFrame(np.concatenate((
        [gamma_init, k_init/k_scaler, s_init/s_scaler], 
        prob_weight_init
        )), 
        columns=["value"])
    prob_weight_lin_curv_params["soft_lower_bound"] = 1e-123
    prob_weight_lin_curv_params["soft_upper_bound"] = 40
    
    prob_weight_conc_curv_params = pd.DataFrame(np.concatenate((
        [gamma_init, k_init/k_scaler, s_init/s_scaler], 
        prob_weight_init
        )), 
        columns=["value"])
    prob_weight_conc_curv_params["soft_lower_bound"] = 1e-123
    prob_weight_conc_curv_params["soft_upper_bound"] = 40

    prob_weight_est_curv_params = pd.DataFrame(np.concatenate((
        [gamma_init, k_init/k_scaler, s_init/s_scaler], 
        prob_weight_init,
        curv_init
        )), 
        columns=["value"])
    prob_weight_est_curv_params["soft_lower_bound"] = 1e-123
    prob_weight_est_curv_params["soft_upper_bound"] = 40
    
    piece_rates = np.array(data.loc[data['dummy1']==1].payoff_per_100)
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
        "params_init": benckmark_params,
        "x_data": piece_rates,
        "buttonpresses": np.array(data.loc[data['dummy1']==1].buttonpresses_nearest_100)
        
    }
    no_weight = {
        "params_init": no_weight_params,
        "x_data": no_weight_x_data,
        "buttonpresses": np.array(data.loc[data['samplenw']==1].buttonpresses_nearest_100)
        
    }
    prob_weight_lin_curv = {
        "params_init": prob_weight_lin_curv_params,
        "x_data": prob_weight_x_data,
        "buttonpresses": np.array(data.loc[data['samplepr']==1].buttonpresses_nearest_100)
    }
    prob_weight_conc_curv = {
        "params_init": prob_weight_conc_curv_params,
        "x_data": prob_weight_x_data,
        "buttonpresses": np.array(data.loc[data['samplepr']==1].buttonpresses_nearest_100)
    }
    prob_weight_est_curv = {
        "params_init": prob_weight_est_curv_params,
        "x_data": prob_weight_x_data,
        "buttonpresses": np.array(data.loc[data['samplepr']==1].buttonpresses_nearest_100)
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
    """Create inputs for `sqrd_residuals_estimagic()`, `minimize()`, and
    `bootstrap()` functions under power cost assumption.

    Args:
        data (pd.DataFrame): raw data for non-linear least squares estimation

    Returns:
        dict: keys are scenario names and values are dictionaries storing
            inputs under each scenario

    """

    gamma_init, k_init, s_init =  19.8117987, 1.66306e-10, 7.74996
    k_scaler, s_scaler = 1e+57, 1e+6

    alpha_init, a_init, beta_init, delta_init, gift_init = 0.003, 0.13, 1.16, 0.75, 5e-6
    prob_weight_init = [0.2]
    curv_init = [0.5]

    benckmark_params = pd.DataFrame([gamma_init, k_init/k_scaler, s_init/s_scaler], columns=["value"])
    benckmark_params["soft_lower_bound"] = 1e-123
    benckmark_params["soft_upper_bound"] = 40

    no_weight_params = pd.DataFrame(np.concatenate((
        [gamma_init, k_init/k_scaler, s_init/s_scaler], 
        [alpha_init, a_init, gift_init, beta_init, delta_init]
        )), 
        columns=["value"])
    no_weight_params["soft_lower_bound"] = 1e-123
    no_weight_params["soft_upper_bound"] = 40

    prob_weight_lin_curv_params = pd.DataFrame(np.concatenate((
        [gamma_init, k_init/k_scaler, s_init/s_scaler], 
        prob_weight_init
        )), 
        columns=["value"])
    prob_weight_lin_curv_params["soft_lower_bound"] = 1e-123
    prob_weight_lin_curv_params["soft_upper_bound"] = 40
    
    prob_weight_conc_curv_params = pd.DataFrame(np.concatenate((
        [gamma_init, k_init/k_scaler, s_init/s_scaler], 
        prob_weight_init
        )), 
        columns=["value"])
    prob_weight_conc_curv_params["soft_lower_bound"] = 1e-123
    prob_weight_conc_curv_params["soft_upper_bound"] = 40

    prob_weight_est_curv_params = pd.DataFrame(np.concatenate((
        [gamma_init, k_init/k_scaler, s_init/s_scaler], 
        prob_weight_init,
        curv_init
        )), 
        columns=["value"])
    prob_weight_est_curv_params["soft_lower_bound"] = 1e-123
    prob_weight_est_curv_params["soft_upper_bound"] = 40

    piece_rates = np.array(data.loc[data['dummy1']==1].payoff_per_100)
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
        "params_init": benckmark_params,
        "x_data": piece_rates,
        "log_buttonpresses": np.array(data.loc[data['dummy1']==1].logbuttonpresses_nearest_100)
        
    }
    no_weight = {
        "params_init": no_weight_params,
        "x_data": no_weight_x_data,
        "log_buttonpresses": np.array(data.loc[data['samplenw']==1].logbuttonpresses_nearest_100)
        
    }
    prob_weight_lin_curv = {
        "params_init": prob_weight_lin_curv_params,
        "x_data": prob_weight_x_data,
        "log_buttonpresses": np.array(data.loc[data['samplepr']==1].logbuttonpresses_nearest_100)
    }
    prob_weight_conc_curv = {
        "params_init": prob_weight_conc_curv_params,
        "x_data": prob_weight_x_data,
        "log_buttonpresses": np.array(data.loc[data['samplepr']==1].logbuttonpresses_nearest_100)
    }
    prob_weight_est_curv = {
        "params_init": prob_weight_est_curv_params,
        "x_data": prob_weight_x_data,
        "log_buttonpresses": np.array(data.loc[data['samplepr']==1].logbuttonpresses_nearest_100)
    }

    out = {
        "benchmark": benchmark,
        "no_weight": no_weight,
        "prob_weight_lin_curv": prob_weight_lin_curv,
        "prob_weight_conc_curv": prob_weight_conc_curv,
        "prob_weight_est_curv": prob_weight_est_curv
    }

    return out


dt = pd.read_csv(BLD / "data" / "nls_data.csv")

inputs_exp = _create_inputs_exp(dt)
inputs_pow = _create_inputs_pow(dt)


def exp_bootstrap_estimagic(data, scenario, algorithm):
    """Calculate the bootstrap outcomes, the parameters of
    the model under exponential cost assumption estimated
    by `minimize()`.

    Args:
        data (pd.DataFrame): data for estimation
        scenario (string): 'benchmark'; 'no_weight' for behavioral; 
            'prob_weight_lin_curv', 'prob_weight_conc_curv', or 
            'prob_weight_est_curv' for probability weighting with linear,
            concave, or estimated curvature parameter, respectively
        algorithm (string): the optimization algorithm: 'scipy_neldermead',
            'scipy_ls_trf', or 'scipy_ls_dogbox'.

    Returns:
        pd.Series: parameter estimates

    """

    inputs = _create_inputs_exp(data)

    x_data = inputs[scenario]["x_data"]        
    buttonpresses = inputs[scenario]["buttonpresses"]
    params_init = inputs[scenario]["params_init"]

    _estimagic = partial(
        sqrd_residuals_estimagic,
        xdata=x_data, buttonpresses=buttonpresses, scenario = scenario)

    sol_estimagic = minimize(
        criterion=_estimagic,
        params=params_init,
        algorithm=algorithm,
        #scaling=True,
        #scaling_options={"method": "scipy_neldermead"},
        #multistart=True
    )

    return pd.Series(sol_estimagic['solution_x'])


def pow_bootstrap_estimagic(data, scenario, algorithm):
    """Calculate the bootstrap outcomes, the parameters of
    the model under power cost assumption estimated
    by `minimize()`.

    Args:
        data (pd.DataFrame): data for estimation
        scenario (string): 'benchmark'; 'no_weight' for behavioral; 
            'prob_weight_lin_curv', 'prob_weight_conc_curv', or 
            'prob_weight_est_curv' for probability weighting with linear,
            concave, or estimated curvature parameter, respectively
        algorithm (string): the optimization algorithm: 'scipy_neldermead',
            'scipy_ls_trf', or 'scipy_ls_dogbox'.

    Returns:
        pd.Series: parameter estimates

    """
    inputs = _create_inputs_pow(data)

    x_data = inputs[scenario]["x_data"]        
    log_buttonpresses = inputs[scenario]["log_buttonpresses"]
    params_init = inputs[scenario]["params_init"]

    _estimagic = partial(
        sqrd_residuals_estimagic,
        xdata=x_data, buttonpresses=log_buttonpresses, scenario = scenario)

    sol_estimagic = minimize(
        criterion=_estimagic,
        params=params_init,
        algorithm=algorithm,
        #scaling=True,
        #scaling_options={"method": "scipy_neldermead"},
        #multistart=True
    )

    return pd.Series(sol_estimagic['solution_x'])


specifications_exp = (
    (BLD / "analysis" / "estimated_parameters_estimagic" / f"{scenario}_est_exp_estimagic_{algorithm}.yaml", scenario, algorithm)
    for scenario in ['benchmark', 'no_weight', 'prob_weight_lin_curv', 'prob_weight_conc_curv','prob_weight_est_curv'] 
    for algorithm in  ['scipy_neldermead','scipy_ls_trf','scipy_ls_dogbox']
)
specifications_pow = (
    (BLD / "analysis" / "estimated_parameters_estimagic" / f"{scenario}_est_pow_estimagic_{algorithm}.yaml", scenario, algorithm)
    for scenario in ['benchmark', 'no_weight', 'prob_weight_lin_curv', 'prob_weight_conc_curv','prob_weight_est_curv'] 
    for algorithm in  ['scipy_neldermead','scipy_ls_trf','scipy_ls_dogbox']
)


@pytask.mark.parametrize("produces, scenario, algorithm", specifications_exp)
def task_estimate_params_exp_estimagic(produces, scenario, algorithm):
    """Estimate and store the parameters of the model together with
    bootstrapped standard errors under five scenarios for exponential
    cost function using estimagic with three alternative algorithms.

    """
    
    x_data = inputs_exp[scenario]["x_data"]        
    buttonpresses = inputs_exp[scenario]["buttonpresses"]
    params_init = inputs_exp[scenario]["params_init"]

    _estimagic = partial(
        sqrd_residuals_estimagic,
        xdata=x_data, buttonpresses=buttonpresses, scenario = scenario)
    _bootstrap = partial(
        exp_bootstrap_estimagic,
        scenario=scenario, algorithm=algorithm)

    sol_estimagic = minimize(
        criterion=_estimagic,
        params=params_init,
        algorithm=algorithm,
        #scaling=True,
        #scaling_options={"method": "scipy_neldermead"},
        #multistart=True
    )

    sol_bootstrap = bootstrap(data=dt, outcome=_bootstrap, n_cores=2)  
    
    final = {
        'estimates' : sol_estimagic['solution_x'].tolist(),
        # 'min obj func' : (np.sum(sol_estimagic["solution_criterion"]).tolist()),
        'bootstrap_se' : (sol_bootstrap["summary"]["std"]).tolist(),
        
    }
    with open(produces, "w") as y:
        yaml.dump(final, y)


@pytask.mark.parametrize("produces, scenario, algorithm", specifications_pow)
def task_estimate_params_pow_estimagic(produces, scenario, algorithm):
    """Estimate and store the parameters of the model together with
    bootstrapped standard errors under five scenarios for power
    cost function using estimagic with three alternative algorithms.

    """

    x_data = inputs_pow[scenario]["x_data"]        
    log_buttonpresses = inputs_pow[scenario]["log_buttonpresses"]
    params_init = inputs_pow[scenario]["params_init"]

    _estimagic = partial(
        sqrd_residuals_estimagic,
        xdata=x_data, buttonpresses=log_buttonpresses, scenario = scenario)
    _bootstrap = partial(
        pow_bootstrap_estimagic,
        scenario=scenario, algorithm=algorithm)

    sol_estimagic = minimize(
        criterion=_estimagic,
        params=params_init,
        algorithm=algorithm,
        #scaling=True,
        #scaling_options={"method": "scipy_neldermead"},
        #multistart=True
    )

    sol_bootstrap = bootstrap(data=dt, outcome=_bootstrap, n_cores=2)  
    
    final = {
        'estimates' : sol_estimagic['solution_x'].tolist(),
        # 'min obj func' : (np.sum(sol_estimagic["solution_criterion"]).tolist()),
        'bootstrap_se' : (sol_bootstrap["summary"]["std"]).tolist(),
        
    }
    with open(produces, "w") as y:
        yaml.dump(final, y)
