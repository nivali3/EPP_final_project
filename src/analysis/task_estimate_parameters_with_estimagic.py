"""GENERAL DOCSTRING

"""
import numpy as np
import pandas as pd
import pytask
import yaml

from functools import partial
from estimagic import minimize
from estimagic.inference import bootstrap

from src.config import BLD
from src.analysis.estimagic_functions import sqrd_residuals_estimagic


specifications_exp = (
    (BLD / "analysis" / f"{scenario}_est_exp_estimagic_{algorithm}.yaml", scenario, algorithm)
    for scenario in ['benchmark', 'no_weight', 'prob_weight_lin_curv', 'prob_weight_conc_curv','prob_weight_est_curv'] 
    for algorithm in  ['scipy_neldermead','scipy_ls_trf','scipy_ls_dogbox']
)


@pytask.mark.depends_on(BLD / "data" / "nls_data.csv")
@pytask.mark.parametrize("produces, scenario, algorithm", specifications_exp)
def task_estimate_params_exp_estimagic(depends_on, produces, scenario, algorithm):
    """Measure the runtime of pandas_batch_update and save the result.
    
    """
    dt = pd.read_csv(depends_on)

    inputs = _create_inputs_exp(dt)

    x_data = inputs[scenario]["x_data"]        
    buttonpresses = inputs[scenario]["buttonpresses"]
    params_init = inputs[scenario]["params_init"]

    _estimagic = partial(sqrd_residuals_estimagic, xdata=x_data, buttonpresses=buttonpresses, scenario = scenario)

    sol_estimagic = minimize(
        criterion=_estimagic,
        params=params_init,
        algorithm=algorithm,
        #scaling=True,
        #scaling_options={"method": "scipy_neldermead"},
        #multistart=True
    )

    # results = bootstrap(data=dt, outcome=sol_nls_exp, n_cores=2)  
    
    final = {
        'estimates' : sol_estimagic['solution_x'].tolist(),
        'min obj func' : (np.sum(sol_estimagic["solution_criterion"]).tolist()),
        # 'bootstrap_se' : (results["summary"]["std"]).tolist(),
        # 'bootstrap_ci' : (results["summary"]["lower_ci"], results["summary"]["upper_ci"]).tolist(),
    }
    with open(produces, "w") as y:
        yaml.dump(final, y)


specifications_pow = (
    (BLD / "analysis" / f"{scenario}_est_pow_estimagic_{algorithm}.yaml", scenario, algorithm)
    for scenario in ['benchmark', 'no_weight', 'prob_weight_lin_curv', 'prob_weight_conc_curv','prob_weight_est_curv'] 
    for algorithm in  ['scipy_neldermead','scipy_ls_trf','scipy_ls_dogbox']
)


@pytask.mark.depends_on(BLD / "data" / "nls_data.csv")
@pytask.mark.parametrize("produces, scenario, algorithm", specifications_pow)
def task_estimate_params_pow_estimagic(depends_on, produces, scenario, algorithm):
    """Measure the runtime of pandas_batch_update and save the result.
    
    """
    dt = pd.read_csv(depends_on)

    inputs = _create_inputs_pow(dt)

    x_data = inputs[scenario]["x_data"]        
    log_buttonpresses = inputs[scenario]["log_buttonpresses"]
    params_init = inputs[scenario]["params_init"]

    _estimagic = partial(sqrd_residuals_estimagic, xdata=x_data, buttonpresses=log_buttonpresses, scenario = scenario)

    sol_estimagic = minimize(
        criterion=_estimagic,
        params=params_init,
        algorithm=algorithm,
        #scaling=True,
        #scaling_options={"method": "scipy_neldermead"},
        #multistart=True
    )

    # results = bootstrap(data=dt, outcome=sol_nls_exp, n_cores=2)  
    
    final = {
        'estimates' : sol_estimagic['solution_x'].tolist(),
        'min obj func' : (np.sum(sol_estimagic["solution_criterion"]).tolist()),
        # 'bootstrap_se' : (results["summary"]["std"]).tolist(),
        # 'bootstrap_ci' : (results["summary"]["lower_ci"], results["summary"]["upper_ci"]).tolist(),
    }
    with open(produces, "w") as y:
        yaml.dump(final, y)







# @pytask.mark.depends_on(BLD / "data" / "nls_data.csv")
# @pytask.mark.parametrize(
#     "produces, scenario",
#     [(BLD / "analysis" / "benchmark_est_pow.yaml", "benchmark"), 
#     (BLD / "analysis" / "no_weight_est_pow.yaml", "no_weight"),
#     (BLD / "analysis" / "weight_lin_curv_est_pow.yaml", "prob_weight_lin_curv"), 
#     (BLD / "analysis" / "weight_conc_curv_est_pow.yaml", "prob_weight_conc_curv"),
#     (BLD / "analysis" / "weight_est_curv_est_pow.yaml", "prob_weight_est_curv")],
# )
# def task_estimate_params_pow(depends_on, produces, scenario):
#     """Measure the runtime of pandas_batch_update and save the result.
    
#     """
#     dt = pd.read_csv(depends_on)

#     inputs = _create_inputs_pow(dt)

#     x_data = inputs[scenario]["x_data"]        
#     log_buttonpresses = inputs[scenario]["log_buttonpresses"]
#     params_init = inputs[scenario]["params_init"]

#     _partial = partial(estimated_effort, scenario = scenario)

#     sol = opt.curve_fit(_partial,
#         x_data,
#         log_buttonpresses,
#         params_init)
#     est = sol[0]
#     std_errors = np.sqrt(np.diagonal(sol[1]))

#     final = {'estimates' : est.tolist(),
#         'std dev' : std_errors.tolist()}
#     with open(produces, "w") as y:
#         yaml.dump(final, y)


# @pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
# @pytask.mark.produces(BLD/'analysis'/'benchmark_est_pow_alt1.yaml')
# def task_estimate_params_benchmark_pow_alt1(depends_on, produces):
#     """Measure the runtime of pandas_batch_update and save the result."""

#     dt = pd.read_csv(depends_on)

#     inputs = _create_inputs_pow(dt)

#     x_data = np.array(inputs["benchmark"]["x_data"])        
#     log_buttonpresses = np.array(inputs["benchmark"]["log_buttonpresses"])
#     params_init = inputs["benchmark"]["params_init"]

#     sol = opt.least_squares(
#         lambda params: sqrd_residuals_benchmark(params, xdata=x_data, logbuttonpresses=log_buttonpresses, optimizer='opt.least_squares'),
#         params_init,
#         xtol=1e-15,
#         ftol=1e-15,
#         gtol=1e-15,
#         method='lm')
#     est = sol.x              
    
#     final = {
#         'estimates' : est.tolist()
#     }
#     with open(produces, "w") as y:
#         yaml.dump(final, y)


# @pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
# @pytask.mark.produces(BLD/'analysis'/'benchmark_est_pow_alt2.yaml')
# def task_estimate_params_benchmark_pow_alt2(depends_on, produces):
#     """Measure the runtime of pandas_batch_update and save the result."""

#     dt = pd.read_csv(depends_on)

#     inputs = _create_inputs_pow(dt)

#     x_data = np.array(inputs["benchmark"]["x_data"])        
#     log_buttonpresses = np.array(inputs["benchmark"]["log_buttonpresses"])
#     params_init = inputs["benchmark"]["params_init"]

#     sol = opt.minimize(
#         lambda params: sqrd_residuals_benchmark(params, xdata=x_data, logbuttonpresses=log_buttonpresses, optimizer='opt.minimize'),
#         params_init,
#         method='Nelder-Mead',
#         options={'maxiter': 2500})
#     est = sol.x                  
    
#     final = {'estimates' : est.tolist()
#     }
#     with open(produces, "w") as y:
#         yaml.dump(final, y)


def _create_inputs_exp(data):

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

