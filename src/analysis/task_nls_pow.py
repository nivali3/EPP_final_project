import numpy as np
import pandas as pd
import pytask
from src.config import BLD
import scipy.optimize as opt
from functools import partial

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


# fix depends on 
@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'est_benchmark_pow.csv')
def task_opt_benchmark_pow(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""

    dt = pd.read_csv(depends_on)

    sol = opt.curve_fit(benchmark_power,
                    dt.loc[dt['dummy1']==1].payoff_per_100,
                    dt.loc[dt['dummy1']==1].logbuttonpresses_nearest_100,
                    st_values_power)
    bp52 = sol[0]                       # sol[0] is the array containing our estimates
    sp52 = np.sqrt(np.diagonal(sol[1])) # sol[1] is a 3x3 variance-covariance matrix of our estimates

    final = {'estimates' : bp52,
        'variances' : sp52}

    final = pd.DataFrame(final)

    with open(produces, "w") as f:
        final.to_csv(f, index=False)


@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'est_benchmark_pow_alt1.csv')
def task_opt_benchmark_pow_alt1(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""

    dt = pd.read_csv(depends_on)
    _partial = partial(benchmark_power_least_squares, dt=dt)
    sol = opt.least_squares(_partial,
                        st_values_power,
                        xtol=1e-15,
                        ftol=1e-15,
                        gtol=1e-15,
                        method='lm')
    bp52 = sol.x # sol.x is the array containing estimates 
                 # opt.least_squares does have any attribute that return var-cov matrix                     
    

    final = {'estimates' : bp52}

    final = pd.DataFrame(final)

    with open(produces, "w") as f:
        final.to_csv(f, index=False)




# Find the solution to the problem by non-linear least squares 

@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'est_benchmark_pow_alt2.csv')
def task_opt_benchmark_pow_alt2(depends_on, produces):
    """Measure the runtime of pandas_batch_update and save the result."""

    dt = pd.read_csv(depends_on)
    _partial = partial(benchmark_power_opt, dt=dt)

    sol_opt = opt.minimize(_partial,
                       st_values_power,
                       method='Nelder-Mead',
                       options={'maxiter': 2500})
    bp52_opt = sol_opt.x                  
    

    final = {'estimates' : bp52_opt}

    final = pd.DataFrame(final)

    with open(produces, "w") as f:
        final.to_csv(f, index=False)

 
@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'est_noweight_pow.csv')
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

    final = {'estimates' : bp53,
            'variances': sp53}

    final = pd.DataFrame(final)

    with open(produces, "w") as f:
        final.to_csv(f, index=False)

@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'est_weight_pow_lin_curv.csv')
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

    final = {'estimates' : bp61,
            'variances': sp61}

    final = pd.DataFrame(final)

    with open(produces, "w") as f:
        final.to_csv(f, index=False)


@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'est_weight_pow_conc_curv.csv')
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

    final = {'estimates' : bp62,
            'variances': sp62}

    final = pd.DataFrame(final)

    with open(produces, "w") as f:
        final.to_csv(f, index=False)


@pytask.mark.depends_on(BLD/'data'/'nls_data.csv')
@pytask.mark.produces(BLD/'analysis'/'est_weight_pow_est_curv.csv')
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

    final = {'estimates' : bp63,
            'variances': sp63}

    final = pd.DataFrame(final)

    with open(produces, "w") as f:
        final.to_csv(f, index=False)


#args = pd.DataFrame(args)



