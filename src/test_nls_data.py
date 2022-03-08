"""Test to check whether forward moving average is calculated correctly.

"""
from matplotlib.pyplot import axis
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


def test_estimated_effort_benchmark_exp():
    inputs = create_inputs()
    expected_est_effort = pd.Series(data=[13.916099, 54.692016, 9.600000], index=FACTORS)
    actual_est_effort = estimated_effort(xdata, *params, scenario='benchmark')
    assert_series_equal(calc_mean, expected_mean)


def test_estimated_effort_benchmark_pow():
    pass


def test_estimated_effort_no_weight_exp():
    pass


def test_estimated_effort_no_weight_pow():
    pass


def test_estimated_effort_prob_weight_lin_curv_exp():
    pass


def test_estimated_effort_prob_weight_lin_curv_pow():
    pass


def test_estimated_effort_prob_weight_conc_curv_exp():
    pass


def test_estimated_effort_prob_weight_conc_curv_pow():
    pass


def test_estimated_effort_prob_weight_est_curv_exp():
    pass


def test_estimated_effort_prob_weight_est_curv_pow():
    pass



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
