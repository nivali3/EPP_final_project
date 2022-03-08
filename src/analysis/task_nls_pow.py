import numpy as np
import pandas as pd
import pytask
import yaml

from src.config import BLD
import scipy.optimize as opt
from functools import partial
from src.analysis.utils import create_inputs
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



# fix depends on 
@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'est_benchmark_pow.yaml')
def task_opt_benchmark_pow(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""

    dt = pd.read_csv(depends_on)

    sol = opt.curve_fit(benchmark_power,
                    dt.loc[dt['dummy1']==1].payoff_per_100,
                    dt.loc[dt['dummy1']==1].logbuttonpresses_nearest_100,
                    st_values_power)
    bp52 = sol[0]                       # sol[0] is the array containing our estimates
    sp52 = np.sqrt(np.diagonal(sol[1])) # sol[1] is a 3x3 variance-covariance matrix of our estimates

    final = {'estimates' : bp52.tolist(),
        'std dev' : sp52.tolist(),
        'min obj func' : (2*benchmark_power_opt(bp52, dt)).tolist()}

    with open(produces, "w") as y:
        yaml.dump(final, y)


@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'est_benchmark_pow_alt1.yaml')
def task_opt_benchmark_pow_alt1(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""

    dt = pd.read_csv(depends_on)
    pay100 = np.array(dt.loc[dt['dummy1']==1].payoff_per_100)
    logbuttonpresses = np.array(dt.loc[dt['dummy1']==1].logbuttonpresses_nearest_100)    

    _partial = partial(benchmark_power_least_squares, pay100=pay100, logbuttonpresses=logbuttonpresses)
    sol = opt.least_squares(_partial,
                        st_values_power,
                        xtol=1e-15,
                        ftol=1e-15,
                        gtol=1e-15,
                        method='lm')
    bp52 = sol.x # sol.x is the array containing estimates 
                 # opt.least_squares does have any attribute that return var-cov matrix                     
    
    final = {'estimates' : bp52.tolist(),
        'min obj func' : (2*benchmark_power_opt(bp52, dt)).tolist()}

    with open(produces, "w") as y:
        yaml.dump(final, y)




# Find the solution to the problem by non-linear least squares 

@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'est_benchmark_pow_alt2.yaml')
def task_opt_benchmark_pow_alt2(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""

    dt = pd.read_csv(depends_on)
    pay100 = np.array(dt.loc[dt['dummy1']==1].payoff_per_100)
    logbuttonpresses = np.array(dt.loc[dt['dummy1']==1].logbuttonpresses_nearest_100)

    _partial = partial(benchmark_power_opt, pay100=pay100, logbuttonpresses=logbuttonpresses)
    sol_opt = opt.minimize(_partial,
                       st_values_power,
                       method='Nelder-Mead',
                       options={'maxiter': 2500})
    bp52_opt = sol_opt.x                  
    
    final = {'estimates' : bp52_opt.tolist(),
        'min obj func' : (2*benchmark_power_opt(bp52_opt, dt)).tolist()}

    with open(produces, "w") as y:
        yaml.dump(final, y)

 
@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'est_noweight_pow.yaml')
def task_opt_noweight_pow(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""

    st_valuesnoweight_power = np.concatenate((st_values_power,stvale_spec)) # starting values

    dt = pd.read_csv(depends_on)
    input = create_inputs(dt)
    args = input['samplenw']
    sol = opt.curve_fit(noweight_power,
                    args,
                    dt.loc[dt['samplenw']==1].logbuttonpresses_nearest_100,
                    st_valuesnoweight_power)
    bp53 = sol[0]
    sp53 = np.sqrt(np.diagonal(sol[1]))

    final = {'estimates' : bp53.tolist(),
            'std dev': sp53.tolist()}

    #final = pd.DataFrame(final)

    #with open(produces, "w") as f:
    #    final.to_csv(f, index=False)
    with open(produces, "w") as y:
        yaml.dump(final, y)

@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'est_weight_pow_lin_curv.yaml')
def task_opt_weight_pow_lin_curv(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""


    st_valuesprobweight_power = np.concatenate((st_values_power,prob_weight_init)) # starting values

    dt = pd.read_csv(depends_on)
    input = create_inputs(dt)
    args = input['samplepr']
    _partial= partial(probweight_power, curv=1)
    sol = opt.curve_fit(_partial,
                    args,
                    dt.loc[dt['samplepr']==1].logbuttonpresses_nearest_100,
                    st_valuesprobweight_power)
    bp61 = sol[0]
    sp61 = np.sqrt(np.diagonal(sol[1]))

    final = {'estimates' : bp61.tolist(),
            'std dev': sp61.tolist()}

    #final = pd.DataFrame(final)

    #with open(produces, "w") as f:
    #   final.to_csv(f, index=False)
    with open(produces, "w") as y:
        yaml.dump(final, y)


@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'est_weight_pow_conc_curv.yaml')
def task_opt_weight_pow_conc_curv(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""


    st_valuesprobweight_power = np.concatenate((st_values_power,prob_weight_init)) # starting values

    dt = pd.read_csv(depends_on)
    input = create_inputs(dt)
    args = input['samplepr']

    _partial= partial(probweight_power, curv=0.88)
    sol = opt.curve_fit(_partial,
                    args,
                    dt.loc[dt['samplepr']==1].logbuttonpresses_nearest_100,
                    st_valuesprobweight_power)
    bp62 = sol[0]
    sp62 = np.sqrt(np.diagonal(sol[1]))

    final = {'estimates' : bp62.tolist(),
            'std dev': sp62.tolist()}

    #final = pd.DataFrame(final)

    #with open(produces, "w") as f:
    #    final.to_csv(f, index=False)
    with open(produces, "w") as y:
        yaml.dump(final, y)


@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'est_weight_pow_est_curv.yaml')
def task_opt_weight_pow_est_curv(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""


    st_valuesprobweight_power = np.concatenate((st_values_power,prob_weight_init,curv_init)) # starting values

    dt = pd.read_csv(depends_on)
    input = create_inputs(dt)
    args = input['samplepr']

    sol = opt.curve_fit(probweight_power,
                    args,
                    dt.loc[dt['samplepr']==1].logbuttonpresses_nearest_100,
                    st_valuesprobweight_power)
    bp63 = sol[0]
    sp63 = np.sqrt(np.diagonal(sol[1]))

    final = {'estimates' : bp63.tolist(),
            'std dev': sp63.tolist()}

    #final = pd.DataFrame(final)

    #with open(produces, "w") as f:
    #    final.to_csv(f, index=False)
    with open(produces, "w") as y:
        yaml.dump(final, y)



def noweight_power_trial(args, alpha, a, gift, beta, delta, g, k, s):
    
    pay100 = args['payoff_per_100']
    gd = args['gift_dummy']
    dd = args['delay_dummy']
    dw = args['delay_wks']
    paychar = args['payoff_charity_per_100']
    dc = args['charity_dummy']
    
    check1= max(k, 1e-115)
    check2= np.maximum(s + gift*0.4*gd + (beta**dd)*(delta**dw)*pay100 + alpha*paychar + a*0.01*dc, 1e-10)  
    f_x = (-1/g * np.log(check1) + 1/g*np.log(check2))
    
    return f_x

@pytask.mark.depends_on({
    "main" : BLD/'data'/'nls_data.csv',
    "bench" : BLD/'analysis'/'est_benchmark_pow.yaml'})
@pytask.mark.produces(BLD/'analysis'/'est_noweight_pow_trial.yaml')
def task_opt_noweight_pow_trial(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""

    dt = pd.read_csv(depends_on["main"])
    
    with open(depends_on["bench"], "r") as stream:
        temp = yaml.safe_load(stream)

    input = create_inputs(dt)
    args = input['samplenw']
    
    _partial_aut = partial(noweight_power_trial, g=20.546, k=5.12e-70, s=3.17e-06)
    _partial = partial(noweight_power_trial, g=temp['estimates'][0], k=temp['estimates'][1], s=temp['estimates'][2]) 
    sol = opt.curve_fit(_partial,
                    args,
                    dt.loc[dt['samplenw']==1].logbuttonpresses_nearest_100,
                    stvale_spec)
    bp53 = sol[0]
    sp53 = np.sqrt(np.diagonal(sol[1]))

    sol_aut = opt.curve_fit(_partial_aut,
                    args,
                    dt.loc[dt['samplenw']==1].logbuttonpresses_nearest_100,
                    stvale_spec)
    bp53_aut = sol[0]
    sp53_aut = np.sqrt(np.diagonal(sol[1]))

    final = {'estimates' : bp53.tolist(),
        'authors estimates' : bp53_aut.tolist(),
        'std dev': sp53.tolist(),
        'authors std dev': sp53_aut.tolist()}

    #final = pd.DataFrame(final)

    #with open(produces, "w") as f:
    #    final.to_csv(f, index=False)
    with open(produces, "w") as y:
        yaml.dump(final, y)


