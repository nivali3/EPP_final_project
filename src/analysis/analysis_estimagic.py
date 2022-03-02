import numpy as np
import pandas as pd
import dfols

from estimagic import minimize
from estimagic import estimate_msm

from functools import partial
from src.config import BLD
import scipy.optimize as opt

from src.analysis.funcs_with_pow_cost import benchmark_power, benchmark_power_least_squares, benchmark_power_opt, noweight_power, probweight_power


gamma_init_power, k_init_power, s_init_power =  19.8117987, 1.66306e-10, 7.74996
k_scaler_power, s_scaler_power = 1e+57,1e+6
k_init_power /= k_scaler_power
s_init_power /= s_scaler_power
st_values_power = [gamma_init_power, k_init_power, s_init_power]
curv_init = [0.5]
prob_weight_init = [0.2]
alpha_init, a_init, beta_init, delta_init, gift_init = 0.003, 0.13, 1.16, 0.75, 5e-6
stvale_spec = [alpha_init, a_init, gift_init, beta_init, delta_init]


def create_inputs(dt):
    out={}
    out['samplenw'] = {'payoff_per_100': dt.loc[dt['samplenw']==1].payoff_per_100,
        'gift_dummy': dt.loc[dt['samplenw']==1].gift_dummy,
        'delay_dummy': dt.loc[dt['samplenw']==1].delay_dummy,
        'delay_wks': dt.loc[dt['samplenw']==1].delay_wks,
        'payoff_charity_per_100': dt.loc[dt['samplenw']==1].payoff_charity_per_100,
        'charity_dummy': dt.loc[dt['samplenw']==1].charity_dummy
    }
    out['samplepr'] = {'payoff_per_100': dt.loc[dt['samplepr']==1].payoff_per_100,
        'weight_dummy': dt.loc[dt['samplepr']==1].weight_dummy,
        'prob': dt.loc[dt['samplepr']==1].prob
    }

    return out
# ==========================================
# ESTIMAGIC
st_values_power_estimagic = pd.DataFrame(st_values_power, columns=["value"])
st_values_power_estimagic["soft_lower_bound"] = -5
st_values_power_estimagic["soft_upper_bound"] = 40


def benchmark_power_estimagic(params, dt):
    
    g = params['value'][0]
    k = params['value'][1]
    s = params['value'][2]

    pay100 = np.array(dt.loc[dt['dummy1']==1].payoff_per_100)

    # Write them with assert!!!
    check1= max(k, 1e-115)                  # since check1 will enter log it must be greater than zero
    check2= np.maximum(s + pay100, 1e-10)   # np.maximum computes the max element wise. We do not want a negative value inside log
    
    out = {
        "value": (-1/g * np.log(check1) +1/g * np.log(check2)),
        "root_contributions": params["value"]
    }

    return out

dt = pd.read_csv(BLD / 'data' / 'nls_data.csv')
criterion_fnc = partial(benchmark_power_estimagic, dt=dt)

# ===========
st_values_power_estimagic = pd.DataFrame(st_values_power, columns=["value"])
st_values_power_estimagic


def benchmarkPower_opt_estimagic(params):
    
    pay100 = np.array(dt.loc[dt['dummy1']==1].payoff_per_100)
    logbuttonpresses = np.array(dt.loc[dt['dummy1']==1].logbuttonpresses_nearest_100)
    g = params['value'][0]
    k = params['value'][1]
    s = params['value'][2]
    
    
    check1= max(k/k_scaler_power, 1e-115)
    check2= np.maximum(s/s_scaler_power + pay100, 1e-10)   
    
    f_x = np.sum(0.5*((-1/g * np.log(check1) +1/g * np.log(check2))-logbuttonpresses)**2)
    
    return f_x


res_variation_snd = minimize(
        criterion=benchmarkPower_opt_estimagic,
        params=st_values_power_estimagic,
        algorithm="scipy_newton_cg",
        logging="benchmarkPower_opt_estimagic.db",
    )
print(res_variation_snd)

# ============
st_values_power_estimagic = pd.DataFrame(st_values_power, columns=["value"])
st_values_power_estimagic["soft_lower_bound"] = 1e-13
st_values_power_estimagic["soft_upper_bound"] = 40
st_values_power_estimagic


def benchmarkPower_opt(params, dt):
    
    pay100 = np.array(dt.loc[dt['dummy1']==1].payoff_per_100)
    logbuttonpresses = np.array(dt.loc[dt['dummy1']==1].logbuttonpresses_nearest_100)
    g = params['value'][0]
    k = params['value'][1]
    s = params['value'][2]
    
    check1= max(k/k_scaler_power, 1e-115)
    check2= np.maximum(s/s_scaler_power + pay100, 1e-10)   
    real = np.array(dt.loc[dt['dummy1']==1].logbuttonpresses_nearest_100)
    predicted = np.array((-1/g * np.log(check1) +1/g * np.log(check2))-logbuttonpresses)
    
    out = {
        "value": (np.sum(0.5*((-1/g * np.log(check1) +1/g * np.log(check2))-logbuttonpresses)**2)),
        "root_contributions": (real - predicted)**2
    }
    return out


# ============
# MSM
# def simulate_data(params, n_draws):
#     x = np.array(dt.loc[dt['dummy1']==1].payoff_per_100)
#     y = np.array(dt.loc[dt['dummy1']==1].logbuttonpresses_nearest_100)
#     y = params.loc["intercept", "value"] + params.loc["slope", "value"] * x + e
#     return pd.DataFrame({"y": y, "x": x})

x = np.array(dt.loc[dt['dummy1']==1].payoff_per_100)
y = np.array(dt.loc[dt['dummy1']==1].logbuttonpresses_nearest_100)

data = pd.DataFrame({"y": y, "x": x})

def calculate_moments(sample):
    moments = {
        "y_mean": sample["y"].mean(),
        "x_mean": sample["x"].mean(),
        "yx_mean": (sample["y"] * sample["x"]).mean(),
        "y_sqrd_mean": (sample["y"] ** 2).mean(),
        "x_sqrd_mean": (sample["x"] ** 2).mean(),
    }
    return pd.Series(moments)

empirical_moments = calculate_moments(data)
empirical_moments

from estimagic import get_moments_cov

moments_cov = get_moments_cov(
    data, calculate_moments, bootstrap_kwargs={"n_draws": 5_000, "seed": 0}
)

moments_cov


def benchmark_power_estimagic(params, dt):
    
    g = params['value'][0]
    k = params['value'][1]
    s = params['value'][2]

    pay100 = np.array(dt.loc[dt['dummy1']==1].payoff_per_100)

    # Write them with assert!!!
    check1= max(k, 1e-115)                  # since check1 will enter log it must be greater than zero
    check2= np.maximum(s + pay100, 1e-10)   # np.maximum computes the max element wise. We do not want a negative value inside log
    
    out = {
        "y": (-1/g * np.log(check1) +1/g * np.log(check2)),
        "x": x
    }

    return out

def simulate_moments(params, dt=dt):
    sim_data = benchmark_power_estimagic(params, dt)
    sim_moments = calculate_moments(sim_data)
    return sim_moments

simulate_moments(st_values_power_estimagic, dt)



start_params = st_values_power_estimagic

res = estimate_msm(
    simulate_moments,
    empirical_moments,
    moments_cov,
    start_params,
    optimize_options={"algorithm": "scipy_lbfgsb"}
)

res["summary"]

# ================
def benchmark_power_estimagic(params, dt):
    
    g = params['value'][0]
    k = params['value'][1]
    s = params['value'][2]

    pay100 = np.array(dt.loc[dt['dummy1']==1].payoff_per_100)

    # Write them with assert!!!
    check1= max(k, 1e-115)                  # since check1 will enter log it must be greater than zero
    check2= np.maximum(s + pay100, 1e-10)   # np.maximum computes the max element wise. We do not want a negative value inside log
    
    out = {
        "value": (-1/g * np.log(check1) +1/g * np.log(check2)),
        "root_contributions": params["value"]
    }

    return out


dt = pd.read_csv(BLD / 'data' / 'nls_data.csv')
criterion_fnc = partial(benchmarkPower_opt, dt=dt)

sol = minimize(
    criterion=criterion_fnc,
    params=st_values_power_estimagic,
    algorithm="nag_dfols",
    #scaling=True,
    #scaling_options={"method": "start_values", "clipping_value": 0.1},
    #multistart=True
)


sol
