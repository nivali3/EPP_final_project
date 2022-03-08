"""Test to check whether forward moving average is calculated correctly.

"""
import pandas as pd
import numpy as np

from src.config import BLD
from src.config import SRC

# check
def test_nls_data():

    expected_values = replication_data()
    actual_values = pd.read_csv(BLD / "data" / "nls_data.csv")

    assert actual_values[sorted(actual_values.columns)].equals(expected_values[sorted(expected_values.columns)])


def replication_data():

    # import the dataset

    dt = pd.read_stata(SRC / "original_data" / "mturk_clean_data_short.dta")

    # Create new variables needed for estimation:

    # Create piece-rate payoffs per 100 button presses (p)
    
    dt['payoff_per_100'] = 0
    dt.loc[dt.treatment == '1.1', 'payoff_per_100'] = 0.01
    dt.loc[dt.treatment == '1.2', 'payoff_per_100'] = 0.1
    dt.loc[dt.treatment == '1.3', 'payoff_per_100'] = 0.0
    dt.loc[dt.treatment == '2'  , 'payoff_per_100'] = 0.001
    dt.loc[dt.treatment == '1.4', 'payoff_per_100'] = 0.04
    dt.loc[dt.treatment == '4.1', 'payoff_per_100'] = 0.01
    dt.loc[dt.treatment == '4.2', 'payoff_per_100'] = 0.01
    dt.loc[dt.treatment == '6.2', 'payoff_per_100'] = 0.02
    dt.loc[dt.treatment == '6.1', 'payoff_per_100'] = 1

    # (alpha/a) create payoff per 100 to charity and dummy charity

    dt['payoff_charity_per_100'] = 0
    dt.loc[dt.treatment == '3.1', 'payoff_charity_per_100'] = 0.01
    dt.loc[dt.treatment == '3.2', 'payoff_charity_per_100'] = 0.1
    dt['charity_dummy'] = 0
    dt.loc[dt.treatment == '3.1', 'charity_dummy'] = 1
    dt.loc[dt.treatment == '3.2', 'charity_dummy'] = 1

    # (beta/delta) create payoff per 100 delayed by 2 weeks and dummy delay

    dt['delay_wks'] = 0
    dt.loc[dt.treatment == '4.1', 'delay_wks'] = 2
    dt.loc[dt.treatment == '4.2', 'delay_wks'] = 4
    dt['delay_dummy'] = 0
    dt.loc[dt.treatment == '4.1', 'delay_dummy'] = 1
    dt.loc[dt.treatment == '4.2', 'delay_dummy'] = 1

    # probability weights to back out curvature and dummy

    dt['prob'] = 1
    dt.loc[dt.treatment == '6.2', 'prob'] = 0.5
    dt.loc[dt.treatment == '6.1', 'prob'] = 0.01
    dt['weight_dummy'] = 0
    dt.loc[dt.treatment == '6.1', 'weight_dummy'] = 1

    # dummy for gift exchange

    dt['gift_dummy'] = 0
    dt.loc[dt.treatment == '10', 'gift_dummy'] = 1

    # generating effort and log effort. authors round buttonpressed to nearest 100 value. If 0 set it to 25.

    dt['buttonpresses'] = dt['buttonpresses'] + 0.1 # python rounds 50 to 0, while stata to 100. by adding a small value we avoid this mismatch
    dt['buttonpresses_nearest_100'] = round(dt['buttonpresses'],-2)
    dt.loc[dt['buttonpresses_nearest_100'] == 0, 'buttonpresses_nearest_100'] = 25
    dt['logbuttonpresses_nearest_100']  = np.log(dt['buttonpresses_nearest_100'])

    # Define the benchmark sample by creating dummies equal to one if in treatment 1.1, 1.2, 1.3 

    dt['t1.1']= (dt['treatment']=='1.1').astype(int)
    dt['t1.2']= (dt['treatment']=='1.2').astype(int)
    dt['t1.3']= (dt['treatment']=='1.3').astype(int)
    dt['dummy1']= dt['t1.1']+dt['t1.2']+dt['t1.3']

    # Allnoweight Exp. Create dummies for this specification

    dt['t3.1']= (dt['treatment']=='3.1').astype(int)
    dt['t3.2']= (dt['treatment']=='3.2').astype(int)
    dt['t4.1']= (dt['treatment']=='4.1').astype(int)
    dt['t4.2']= (dt['treatment']=='4.2').astype(int)
    dt['t10'] = (dt['treatment']=='10').astype(int)
    dt['samplenw']= dt['dummy1']+dt['t3.1']+dt['t3.2']+dt['t4.1']+dt['t4.2']+dt['t10']

    # Create the sample used for Table 6 panel A
        
    dt['t6.1']= (dt['treatment']=='6.1').astype(int)
    dt['t6.2']= (dt['treatment']=='6.2').astype(int)
    dt['samplepr']= dt['dummy1']+dt['t6.1']+dt['t6.2']

    return dt



# testing functions
import pandas as pd
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal

from src.analysis.costly_effort_model_functions import estimated_effort


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
    pass


def test_estimated_effort_prob_weight_est_curv():
    pass


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


# FACTORS = list("cni")


# def test_square_root_unscented_predict_mean():
#     inputs = create_inputs()
#     expected_mean = pd.Series(data=[13.916099, 54.692016, 9.600000], index=FACTORS)
#     calc_mean, calc_root_cov = square_root_unscented_predict(**inputs)
#     assert_series_equal(calc_mean, expected_mean)


# def test_square_root_unscented_predict_cov_values():
#     inputs = create_inputs()
#     expected_cov = pd.DataFrame(
#         data=[
#             [146.342189, 566.426833, 54.003827],
#             [566.426833, 3164.463379, 137.814019],
#             [54.003827, 137.814019, 39.240000],
#         ],
#         columns=FACTORS,
#         index=FACTORS,
#     )
#     calc_mean, calc_root_cov = square_root_unscented_predict(**inputs)
#     calc_cov = calc_root_cov.dot(calc_root_cov.T)
#     assert_frame_equal(calc_cov, expected_cov)


# def create_inputs():
#     out = {}
#     out["state"] = pd.Series(data=[12, 10, 8], index=FACTORS)
#     out["root_cov"] = pd.DataFrame(
#         data=[[6, 0, 0], [3, 5, 0], [2, 1, 4.0]],
#         columns=FACTORS,
#         index=FACTORS,
#     )
#     params = {
#         "c": {"gammas": pd.Series(data=[0.5] * 3, index=FACTORS), "a": 0.5},
#         "n": {"gammas": pd.Series(data=[1.5, 1, 0], index=FACTORS), "a": 0.1},
#         "i": {"gammas": pd.Series(data=[0, 0, 1.0], index=FACTORS), "a": 1.2},
#     }
#     out["params"] = params
#     out["shock_sds"] = pd.Series(data=[1, 2, 3.0], index=FACTORS)
#     out["kappa"] = 1

#     return out
