import numpy as np
import pandas as pd
import pytask
import yaml

from functools import partial
from estimagic.inference import bootstrap
from src.config import BLD
from estimagic import minimize


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


st_values_power_estimagic = pd.DataFrame(st_values_power, columns=["value"])
st_values_power_estimagic["soft_lower_bound"] = -5
st_values_power_estimagic["soft_upper_bound"] = 40



def benchmark_power_opt_estimagic(params, pay100, logbuttonpresses):

    g = params['value'][0]
    k = params['value'][1]
    s = params['value'][2]

    check1= max(k, 1e-115)
    check2= np.maximum(s + pay100, 1e-10)

    out = {
        "value": (np.sum((-1/g * np.log(check1) +1/g * np.log(check2))-logbuttonpresses)**2),
        "root_contributions": ((-1/g * np.log(check1) +1/g * np.log(check2))-logbuttonpresses)**2
    }

    return out

def sol_nls(data):
    pay100 = np.array(data.loc[data['dummy1']==1].payoff_per_100)
    logbuttonpresses = np.array(data.loc[data['dummy1']==1].logbuttonpresses_nearest_100)

    _estimagic = partial(benchmark_power_opt_estimagic, pay100=pay100, logbuttonpresses=logbuttonpresses)

    sol_opt_estimagic = minimize(
        criterion=_estimagic,
        params=st_values_power_estimagic,
        algorithm="scipy_ls_trf",

    )

    return pd.Series(sol_opt_estimagic['solution_x'])


@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'est_benchmark_opt_estimagic.yaml')
def task_opt_benchmark_estimagic(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""

    dt = pd.read_csv(depends_on)
    pay100 = np.array(dt.loc[dt['dummy1']==1].payoff_per_100)
    logbuttonpresses = np.array(dt.loc[dt['dummy1']==1].logbuttonpresses_nearest_100)

    _estimagic = partial(benchmark_power_opt_estimagic, pay100=pay100, logbuttonpresses=logbuttonpresses)

    sol_opt_estimagic = minimize(
    criterion=_estimagic,
    params=st_values_power_estimagic,
    algorithm="scipy_ls_trf",
    #scaling=True,
    #scaling_options={"method": "scipy_neldermead"},
    #multistart=True

    )

    results = bootstrap(data=dt, outcome=sol_nls, n_cores=2)  
                  
    
    final = {'estimates' : (sol_nls(dt)).tolist(),
        'min obj func' : (np.sum(sol_opt_estimagic["solution_criterion"])).tolist(),
        'bootstrap_se' : (results["summary"]["std"]).tolist()}

    with open(produces, "w") as y:
        yaml.dump(final, y)



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

st_values_exp_estimagic = pd.DataFrame(st_values_exp, columns=["value"])
st_values_exp_estimagic["soft_lower_bound"] = -5
st_values_exp_estimagic["soft_upper_bound"] = 40



def benchmark_exp_estimagic(params, pay100, logbuttonpresses):
    """Estimates the optimal effort level using the benchmark model with exponential
    cost function.
    
    Args:
        pay100 (pd.Series): the piece rates for benchmark treatments
        g (float): parameter for curvature of the cost function
        k (float): parameter indicating level of the cost function
        s (float): parameter for intrinsic motivation

    Returns:
        estimated_effort (float)

    """
    g = params['value'][0]
    k = params['value'][1]
    s = params['value'][2]

    check1= max(k, 1e-115)
    check2= np.maximum(s + pay100, 1e-10)

    out = {
        "value": (np.sum((-1/g * np.log(check1) +1/g * np.log(check2))-logbuttonpresses)**2),
        "root_contributions": ((-1/g * np.log(check1) +1/g * np.log(check2))-logbuttonpresses)**2
    }

    return out


def sol_nls_exp(data):
    pay100 = np.array(data.loc[data['dummy1']==1].payoff_per_100)
    logbuttonpresses = np.array(data.loc[data['dummy1']==1].logbuttonpresses_nearest_100)

    _estimagic = partial(benchmark_exp_estimagic, pay100=pay100, logbuttonpresses=logbuttonpresses)

    sol_opt_estimagic = minimize(
        criterion=_estimagic,
        params=st_values_exp_estimagic,
        algorithm="scipy_ls_trf",

    )

    return pd.Series(sol_opt_estimagic['solution_x'])



@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'est_benchmark_opt_estimagic_exp.yaml')
def task_opt_benchmark_estimagic_exp(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""

    dt = pd.read_csv(depends_on)
    pay100 = np.array(dt.loc[dt['dummy1']==1].payoff_per_100)
    logbuttonpresses = np.array(dt.loc[dt['dummy1']==1].logbuttonpresses_nearest_100)

    _estimagic = partial(benchmark_exp_estimagic, pay100=pay100, logbuttonpresses=logbuttonpresses)

    sol_opt_estimagic = minimize(
    criterion=_estimagic,
    params=st_values_exp_estimagic,
    algorithm="scipy_ls_trf",
    #scaling=True,
    #scaling_options={"method": "scipy_neldermead"},
    #multistart=True

    )

    results = bootstrap(data=dt, outcome=sol_nls_exp, n_cores=2)  
                  
    
    final = {'estimates' : (sol_nls_exp(dt)).tolist(),
        'min obj func' : (np.sum(sol_opt_estimagic["solution_criterion"]).tolist()),
        'bootstrap_se' : (results["summary"]["std"]).tolist()}

    with open(produces, "w") as y:
        yaml.dump(final, y)



# =========
# 
def no_weight_exp_estimagic(params, args, buttonpresses):
    """Estimates the optimal effort level using the behavioral model without probability weighting 
    and with exponential cost function.
    
    Args:
        args (dict): keys includes treatment dummies, the piece rates for all treatments except probability weighting ones,
        and the piece rates devolved to charity
        g (float): parameter for curvature of the cost function
        k (float): parameter indicating level of the cost function
        s (float): parameter for intrinsic motivation
        alpha (float): pure altruism coefficient
        a (float): warm glow coefficient
        gift (float): gift exchange coefficient
        beta (float): parametre for present bias
        delta (float): (weekly) discount factor

    Returns:
        estimated_effort (float)

    """

    g, k, s, alpha, a, gift, beta, delta = params['value']

    pay100 = args['payoff_per_100']
    gd = args['gift_dummy']
    dd = args['delay_dummy']
    dw = args['delay_wks']
    paychar = args['payoff_charity_per_100']
    dc = args['charity_dummy']
    
    check1 = k/k_scaler_exp
    check2 = s/s_scaler_exp + gift*0.4*gd + (beta**dd)*(delta**dw)*pay100 + alpha*paychar +a*0.01*dc
    
    out = {
        "value": (np.sum((-1/g * np.log(check1) +1/g * np.log(check2))-buttonpresses)**2),
        "root_contributions": ((-1/g * np.log(check1) +1/g * np.log(check2))-buttonpresses)**2
    }

    return out

st_valuesnoweight_exp_estimagic = pd.DataFrame(st_valuesnoweight_exp, columns=["value"])
st_valuesnoweight_exp_estimagic["soft_lower_bound"] = -5
st_valuesnoweight_exp_estimagic["soft_upper_bound"] = 40

def sol_nls_exp_noweight(dt):
    input = create_inputs(dt)
    args = input['samplenw']
    buttonpresses = np.array(dt.loc[dt['samplenw']==1].buttonpresses_nearest_100)

    _estimagic = partial(no_weight_exp_estimagic, args=args, buttonpresses=buttonpresses)

    sol_opt_estimagic = minimize(
        criterion=_estimagic,
        params=st_valuesnoweight_exp_estimagic,
        algorithm="scipy_ls_trf",
        #scaling=True,
        #scaling_options={"method": "scipy_neldermead"},
        #multistart=True
    )

    return pd.Series(sol_opt_estimagic['solution_x'])

@pytask.mark.depends_on(BLD / "data" / "nls_data.csv")
@pytask.mark.produces(BLD / "analysis" / "est_noweight_exp_estimagic.yaml")
def task_opt_noweight_exp_estimagic(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result.
    
    """

    dt = pd.read_csv(depends_on)
    input = create_inputs(dt)
    args = input['samplenw']
    buttonpresses = np.array(dt.loc[dt['samplenw']==1].buttonpresses_nearest_100)

    _estimagic = partial(no_weight_exp_estimagic, args=args, buttonpresses=buttonpresses)

    sol_opt_estimagic = minimize(
        criterion=_estimagic,
        params=st_valuesnoweight_exp_estimagic,
        algorithm="scipy_ls_trf",
        #scaling=True,
        #scaling_options={"method": "scipy_neldermead"},
        #multistart=True
    )

    results = bootstrap(data=dt, outcome=sol_nls_exp_noweight, n_cores=2)  
                  
    
    final = {'estimates' : (sol_nls_exp_noweight(dt)).tolist(),
        'min obj func' : (np.sum(sol_opt_estimagic["solution_criterion"]).tolist()),
        'bootstrap_se' : (results["summary"]["std"]).tolist(),
        'lower_ci' : (results["summary"]["lower_ci"]).tolist(),
        'upper_ci' : (results["summary"]["upper_ci"]).tolist()}

    with open(produces, "w") as y:
        yaml.dump(final, y)



# ======
def prob_weight_exp_estimagic_wthcurv(params, curv, args, buttonpresses):
    """Estimates the optimal effort level using the probability weighting model and exponential cost function.
    
    Args:
        args (dict): keys includes treatment dummies, the piece rates for all treatments except probability weighting ones,
        and the piece rates devolved to charity
        g (float): parameter for curvature of the cost function
        k (float): parameter indicating level of the cost function
        s (float): parameter for intrinsic motivation
        p_weight (float): the probability weighting coefficient
        curv (float): curvature of the value function

    Returns:
        estimated_effort (float)

    """

    g, k, s, p_weight = params['value']
    
    pay100 = args['payoff_per_100']
    wd = args['weight_dummy']
    prob = args['prob']
    
    check1=k/k_scaler_exp
    check2=s/s_scaler_exp + p_weight**wd*prob*pay100**curv

    out = {
        "value": (np.sum((-1/g * np.log(check1) +1/g * np.log(check2))-buttonpresses)**2),
        "root_contributions": ((-1/g * np.log(check1) +1/g * np.log(check2))-buttonpresses)**2
    }

    return out


def prob_weight_exp_estimagic(params, args, buttonpresses):
    """Estimates the optimal effort level using the probability weighting model and exponential cost function.
    
    Args:
        args (dict): keys includes treatment dummies, the piece rates for all treatments except probability weighting ones,
        and the piece rates devolved to charity
        g (float): parameter for curvature of the cost function
        k (float): parameter indicating level of the cost function
        s (float): parameter for intrinsic motivation
        p_weight (float): the probability weighting coefficient
        curv (float): curvature of the value function

    Returns:
        estimated_effort (float)

    """

    g, k, s, p_weight, curv = params['value']
    
    pay100 = args['payoff_per_100']
    wd = args['weight_dummy']
    prob = args['prob']
    
    check1=k/k_scaler_exp
    check2=s/s_scaler_exp + p_weight**wd*prob*pay100**curv

    out = {
        "value": (np.sum((-1/g * np.log(check1) +1/g * np.log(check2))-buttonpresses)**2),
        "root_contributions": ((-1/g * np.log(check1) +1/g * np.log(check2))-buttonpresses)**2
    }

    return out


st_valuesprobweight_exp_estimagic = pd.DataFrame(st_valuesprobweight_exp, columns=["value"])
st_valuesprobweight_exp_estimagic["soft_lower_bound"] = -5
st_valuesprobweight_exp_estimagic["soft_upper_bound"] = 40

def sol_4(dt):

    input = create_inputs(dt)
    args = input['samplepr']
    buttonpresses = np.array(dt.loc[dt['samplepr']==1].buttonpresses_nearest_100)

    _estimagic = partial(prob_weight_exp_estimagic_wthcurv, curv=1, args=args, buttonpresses=buttonpresses)
    
    sol_opt_estimagic = minimize(
        criterion=_estimagic,
        params=st_valuesprobweight_exp_estimagic,
        algorithm="scipy_ls_trf",
        #scaling=True,
        #scaling_options={"method": "scipy_neldermead"},
        #multistart=True
    )

    return pd.Series(sol_opt_estimagic['solution_x'])

st_valuesprobweight6_exp_estimagic = pd.DataFrame(st_valuesprobweight6_exp, columns=["value"])
st_valuesprobweight6_exp_estimagic["soft_lower_bound"] = -5
st_valuesprobweight6_exp_estimagic["soft_upper_bound"] = 40

def sol_5(dt):

    input = create_inputs(dt)
    args = input['samplepr']
    buttonpresses = np.array(dt.loc[dt['samplepr']==1].buttonpresses_nearest_100)

    _estimagic = partial(prob_weight_exp_estimagic_wthcurv, curv=10.88, args=args, buttonpresses=buttonpresses)
    
    sol_opt_estimagic = minimize(
        criterion=_estimagic,
        params=st_valuesprobweight_exp_estimagic,
        algorithm="scipy_ls_trf",
        #scaling=True,
        #scaling_options={"method": "scipy_neldermead"},
        #multistart=True
    )

    return pd.Series(sol_opt_estimagic['solution_x'])


def sol_6(dt):

    input = create_inputs(dt)
    args = input['samplepr']
    buttonpresses = np.array(dt.loc[dt['samplepr']==1].buttonpresses_nearest_100)

    _estimagic = partial(prob_weight_exp_estimagic, args=args, buttonpresses=buttonpresses)
    
    sol_opt_estimagic = minimize(
        criterion=_estimagic,
        params=st_valuesprobweight6_exp_estimagic,
        algorithm="scipy_ls_trf",
        #scaling=True,
        #scaling_options={"method": "scipy_neldermead"},
        #multistart=True
    )

    return pd.Series(sol_opt_estimagic['solution_x'])

@pytask.mark.depends_on(BLD / "data" / "nls_data.csv")
@pytask.mark.produces(BLD / "analysis" / "est_weight_exp_lin_curv_estimagic.yaml")
def task_opt_weight_exp_lin_curv_estimagic(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result.
    
    """
    dt = pd.read_csv(depends_on)

    input = create_inputs(dt)
    args = input['samplepr']
    buttonpresses = np.array(dt.loc[dt['samplepr']==1].buttonpresses_nearest_100)

    _estimagic = partial(prob_weight_exp_estimagic_wthcurv, curv=1, args=args, buttonpresses=buttonpresses)
    
    sol_opt_estimagic = minimize(
        criterion=_estimagic,
        params=st_valuesprobweight_exp_estimagic,
        algorithm="scipy_ls_trf",
        #scaling=True,
        #scaling_options={"method": "scipy_neldermead"},
        #multistart=True
    )

    results = bootstrap(data=dt, outcome=sol_4, n_cores=2)  
   

    final = {'estimates' : (sol_4(dt)).tolist(),
        'min obj func' : (np.sum(sol_opt_estimagic["solution_criterion"]).tolist()),
        'bootstrap_se' : (results["summary"]["std"]).tolist()}

    with open(produces, "w") as y:
        yaml.dump(final, y)


@pytask.mark.depends_on(BLD / "data" / "nls_data.csv")
@pytask.mark.produces(BLD / "analysis" / "est_weight_exp_conc_curv_estimagic.yaml")
def task_opt_weight_exp_conc_curv_estimagic(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result.
    
    """
    dt = pd.read_csv(depends_on)

    input = create_inputs(dt)
    args = input['samplepr']
    buttonpresses = np.array(dt.loc[dt['samplepr']==1].buttonpresses_nearest_100)

    _estimagic = partial(prob_weight_exp_estimagic_wthcurv, curv=0.88, args=args, buttonpresses=buttonpresses)
    
    sol_opt_estimagic = minimize(
        criterion=_estimagic,
        params=st_valuesprobweight_exp_estimagic,
        algorithm="scipy_ls_trf",
        #scaling=True,
        #scaling_options={"method": "scipy_neldermead"},
        #multistart=True
    )

    results = bootstrap(data=dt, outcome=sol_5, n_cores=2)  
   
    final = {'estimates' : (sol_5(dt)).tolist(),
        'min obj func' : (np.sum(sol_opt_estimagic["solution_criterion"]).tolist()),
        'bootstrap_se' : (results["summary"]["std"]).tolist()}

    with open(produces, "w") as y:
        yaml.dump(final, y)


@pytask.mark.depends_on(BLD / "data" / "nls_data.csv")
@pytask.mark.produces(BLD / "analysis" / "est_weight_exp_est_curv_estimagic.yaml")
def task_opt_weight_exp_est_curv_estimagic(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result.
    
    """
    dt = pd.read_csv(depends_on)

    input = create_inputs(dt)
    args = input['samplepr']
    buttonpresses = np.array(dt.loc[dt['samplepr']==1].buttonpresses_nearest_100)

    _estimagic = partial(prob_weight_exp_estimagic, args=args, buttonpresses=buttonpresses)
    
    sol_opt_estimagic = minimize(
        criterion=_estimagic,
        params=st_valuesprobweight6_exp_estimagic,
        algorithm="scipy_ls_trf",
        #scaling=True,
        #scaling_options={"method": "scipy_neldermead"},
        #multistart=True
    )

    results = bootstrap(data=dt, outcome=sol_6, n_cores=2)  
   
    final = {'estimates' : (sol_6(dt)).tolist(),
        'min obj func' : (np.sum(sol_opt_estimagic["solution_criterion"]).tolist()),
        'bootstrap_se' : (results["summary"]["std"]).tolist()}

    with open(produces, "w") as y:
        yaml.dump(final, y)
        

gamma_init_power, k_init_power, s_init_power =  19.8117987, 1.66306e-10, 7.74996
k_scaler_power, s_scaler_power = 1e+57,1e+6
k_init_power /= k_scaler_power
s_init_power /= s_scaler_power
st_values_power = [gamma_init_power, k_init_power, s_init_power]
curv_init = [0.5]
prob_weight_init = [0.2]
alpha_init, a_init, beta_init, delta_init, gift_init = 0.003, 0.13, 1.16, 0.75, 5e-6
stvale_spec = [alpha_init, a_init, gift_init, beta_init, delta_init]

st_valuesnoweight_power = np.concatenate((st_values_power,stvale_spec))

def noweight_power_estimagic(params, args, logbuttonpresses):
    """Estimates the optimal effort level using the behavioral model without probability weighting
    and with power cost function.

    Args:
        args (dict): keys includes treatment dummies, the piece rates for all treatments except probability weighting ones,
        and the piece rates devolved to charity
        g (float): parameter for curvature of the cost function
        k (float): parameter indicating level of the cost function
        s (float): parameter for intrinsic motivation
        alpha (float): pure altruism coefficient
        a (float): warm glow coefficient
        gift (float): gift exchange coefficient
        beta (float): parametre for present bias
        delta (float): (weekly) discount factor

    Returns:
        estimated_effort (float)

    """
    g, k, s, alpha, a, gift, beta, delta = params['value']

    pay100 = args['payoff_per_100']
    gd = args['gift_dummy']
    dd = args['delay_dummy']
    dw = args['delay_wks']
    paychar = args['payoff_charity_per_100']
    dc = args['charity_dummy']

    #assert k>0,"Only positive quantities are allowed to enter log."
    #assert (s + gift*0.4*gd + (beta**dd)*(delta**dw)*pay100 + alpha*paychar + a*0.01*dc, 1e-10)>0, "Only positive quantities are allowed to enter log."
    #f_x = (-1/g * np.log(k) + 1/g*np.log(s + gift*0.4*gd + (beta**dd)*(delta**dw)*pay100 + alpha*paychar + a*0.01*dc))


    out = {
        "value": (np.sum((-1/g * np.log(k) +1/g * np.log(s + gift*0.4*gd + (beta**dd)*(delta**dw)*pay100 + alpha*paychar + a*0.01*dc))-logbuttonpresses)**2),
        "root_contributions": ((-1/g * np.log(k) +1/g * np.log(s + gift*0.4*gd + (beta**dd)*(delta**dw)*pay100 + alpha*paychar + a*0.01*dc))-logbuttonpresses)**2
    }

    return out

st_valuesnoweight_pow_estimagic = pd.DataFrame(st_valuesnoweight_power, columns=["value"])
st_valuesnoweight_pow_estimagic["soft_lower_bound"] = -5
st_valuesnoweight_pow_estimagic["soft_upper_bound"] = 40

def sol_nls_pow_noweight(dt):
    input = create_inputs(dt)
    args = input['samplenw']
    buttonpresses = np.array(dt.loc[dt['samplenw']==1].logbuttonpresses_nearest_100)

    _estimagic = partial(noweight_power_estimagic, args=args, logbuttonpresses=buttonpresses)

    sol_opt_estimagic = minimize(
        criterion=_estimagic,
        params=st_valuesnoweight_pow_estimagic,
        algorithm="scipy_ls_trf",
        #scaling=True,
        #scaling_options={"method": "scipy_neldermead"},
        #multistart=True
    )

    return pd.Series(sol_opt_estimagic['solution_x'])

@pytask.mark.depends_on(BLD / "data" / "nls_data.csv")
@pytask.mark.produces(BLD / "analysis" / "est_noweight_pow_estimagic.yaml")
def task_opt_noweight_pow_estimagic(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result.

    """

    dt = pd.read_csv(depends_on)
    input = create_inputs(dt)
    args = input['samplenw']
    buttonpresses = np.array(dt.loc[dt['samplenw']==1].logbuttonpresses_nearest_100)

    _estimagic = partial(noweight_power_estimagic, args=args, logbuttonpresses=buttonpresses)

    sol_opt_estimagic = minimize(
        criterion=_estimagic,
        params=st_valuesnoweight_pow_estimagic,
        algorithm="scipy_ls_trf",
        #scaling=True,
        #scaling_options={"method": "scipy_neldermead"},
        #multistart=True
    )

    results = bootstrap(data=dt, outcome=sol_nls_pow_noweight, n_cores=2)


    final = {'estimates' : (sol_nls_pow_noweight(dt)).tolist(),
        'min obj func' : (np.sum(sol_opt_estimagic["solution_criterion"]).tolist()),
        'bootstrap_se' : (results["summary"]["std"]).tolist(),
        'lower_ci' : (results["summary"]["lower_ci"]).tolist(),
        'upper_ci' : (results["summary"]["upper_ci"]).tolist()
    }

    with open(produces, "w") as y:
        yaml.dump(final, y)
