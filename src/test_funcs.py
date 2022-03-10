'''testing the functions in src/model_code.
'''

import pandas as pd
import numpy as np
from pandas.testing import assert_series_equal

from src.model_code.model_function import estimated_effort


def test_estimated_effort_benchmark():
    inputs = _create_test_inputs()

    params = inputs['benchmark']['params']
    x_data = inputs['benchmark']['x_data']

    expected_est_effort = pd.Series(data=[39.02018128, 39.42286802, 40.21864078])
    actual_est_effort = estimated_effort(x_data, *params, scenario='benchmark')

    assert_series_equal(expected_est_effort, actual_est_effort, check_names=False)


def test_estimated_effort_no_weight():
    inputs = _create_test_inputs()

    params = inputs['no_weight']['params']
    x_data = inputs['no_weight']['x_data']

    expected_est_effort = pd.Series(data=[39.00306244, 39.42447227, 40.06841533])
    actual_est_effort = estimated_effort(x_data, *params, scenario='no_weight')

    assert_series_equal(expected_est_effort, actual_est_effort, check_names=False)


def test_estimated_effort_prob_weight_lin_curv():
    inputs = _create_test_inputs()

    params = inputs['prob_weight_lin_curv']['params']
    x_data = inputs['prob_weight_lin_curv']['x_data']

    expected_est_effort = pd.Series(data=[38.89874011709243, 38.93925328608813, 38.89874011709243])
    actual_est_effort = estimated_effort(x_data, *params, scenario='prob_weight_lin_curv')

    assert_series_equal(expected_est_effort, actual_est_effort, check_names=False)


def test_estimated_effort_prob_weight_conc_curv():
    inputs = _create_test_inputs()

    params = inputs['prob_weight_conc_curv']['params']
    x_data = inputs['prob_weight_conc_curv']['x_data']

    expected_est_effort = pd.Series(data=[38.908706711223445,38.964704404887044,38.90303953691895])
    actual_est_effort = estimated_effort(x_data, *params, scenario='prob_weight_conc_curv')

    assert_series_equal(expected_est_effort, actual_est_effort, check_names=False)


def test_estimated_effort_prob_weight_est_curv():
    inputs = _create_test_inputs()

    params = inputs['prob_weight_est_curv']['params']
    x_data = inputs['prob_weight_est_curv']['x_data']

    expected_est_effort = pd.Series(data=[39.02018127785867,39.15477086114424,38.927943660188205])
    actual_est_effort = estimated_effort(x_data, *params, scenario='prob_weight_est_curv')

    assert_series_equal(expected_est_effort, actual_est_effort, check_names=False)


def _create_test_inputs():

    gamma, k, s =  0.02, 1.7, 3.7

    alpha, a, beta, delta, gift = 0.003, 0.13, 1.16, 0.75, 0.0
    prob_weight = [0.2]
    curv = [0.5]
    
    piece_rates = pd.Series(data=[0.01, 0.04, 0.1], name='payoff_per_100')
    no_weight_x_data = {
        'payoff_per_100': pd.Series(data=[0.01, 0.04, 0.1]),
        'gift_dummy': pd.Series(data=[0, 1, 0]),
        'delay_dummy': pd.Series(data=[1, 0, 1]),
        'delay_wks': pd.Series(data=[1, 0, 1]),
        'payoff_charity_per_100': pd.Series(data=[0.01, 0.04, 0.1]),
        'charity_dummy': pd.Series(data=[0, 0, 1])
    }
    prob_weight_x_data = {
        'payoff_per_100': pd.Series(data=[0.01, 0.04, 0.1]),
        'weight_dummy': pd.Series(data=[1, 1, 0]),
        'prob': pd.Series(data=[0.5, 0.5, 0.01])
    }

    benchmark = {
        "params": [gamma, k, s],
        "x_data": piece_rates,
        "buttonpresses": pd.Series(data=[25, 100, 200])
        
    }
    no_weight = {
        "params": np.concatenate((
            [gamma, k, s], 
            [alpha, a, gift, beta, delta]
            )),
        "x_data": no_weight_x_data,
        "buttonpresses": pd.Series(data=[25, 100, 200])
        
    }
    prob_weight_lin_curv = {
        "params": np.concatenate((
            [gamma, k, s], 
            prob_weight
            )),
        "x_data": prob_weight_x_data,
        "buttonpresses": pd.Series(data=[25, 100, 200])
    }
    prob_weight_conc_curv = {
        "params": np.concatenate((
            [gamma, k, s], 
            prob_weight
            )),
        "x_data": prob_weight_x_data,
        "buttonpresses": pd.Series(data=[25, 100, 200])
    }
    prob_weight_est_curv = {
        "params": np.concatenate((
            [gamma, k, s], 
            prob_weight, curv
            )),
        "x_data": prob_weight_x_data,
        "buttonpresses": pd.Series(data=[25, 100, 200])
    }

    out = {
        "benchmark": benchmark,
        "no_weight": no_weight,
        "prob_weight_lin_curv": prob_weight_lin_curv,
        "prob_weight_conc_curv": prob_weight_conc_curv,
        "prob_weight_est_curv": prob_weight_est_curv
    }

    return out