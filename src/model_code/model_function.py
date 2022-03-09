"""Functions from the costly effort model with exponential and power cost
functions for five different scenarios, namely:

benchmark (i.e. monetary incentive treatments), which includes
treatments 1.1 ('1c PieceRate'), 1.2 ('10c PieceRate'), 1.3 ('No Payment');

behavioral (i.e. extension from the benchmark by adding behavioral 
parameters), which additionally includes treatments 3.1 ('1c RedCross'),
3.2 ('10c RedCross'), 4.1 ('1c 2Wks'), 4.2 ('1c 4Wks'), 10 ('Gift Exchange');

probability weighting with linear, concave, and estimated curvature parameter 
of the value function (i.e. extension from the benchmark by adding probability
weighting parameters), which additionally includes treatments 6.1
('Prob.01 $1'), 6.2 ('Prob.5 2c').

"""

import numpy as np


def estimated_effort(xdata, *params, scenario):
    """Calculates optimal effort under exponential cost assumption and
    log of optimal effort under power cost assumption for corresponding scenario.

    Args:
        xdata (dict): keys are the names of the corresponding columns of the
            data and values (pd.Series) are column values for each scenario
        *params (float): parameters of the model under
            corresponding scenario (e.g. g, k, s under 'benchmark')
        scenario (string): 'benchmark'; 'no_weight' for behavioral; 
            'prob_weight_lin_curv', 'prob_weight_conc_curv', or 
            'prob_weight_est_curv' for probability weighting with linear,
            concave, or estimated curvature parameter, respectively

    Returns:
        estimated_effort (float)

    """

    if scenario == 'benchmark':

        g, k, s = params

        piece_rates = xdata

        comp_1= max(k, 1e-115)
        comp_2= np.maximum(s + piece_rates, 1e-10)
        
    elif scenario == 'no_weight':

        g, k, s, alpha, a, gift, beta, delta = params

        piece_rates = xdata['payoff_per_100']
        gift_dmmy = xdata['gift_dummy']
        delay_dmmy = xdata['delay_dummy']
        delay_wks = xdata['delay_wks']
        charity_rates = xdata['payoff_charity_per_100']
        charity_dmmy = xdata['charity_dummy']

        comp_1= max(k, 1e-115)
        comp_2= np.maximum(s + gift*0.4*gift_dmmy + (beta**delay_dmmy)*(delta**delay_wks)*piece_rates + alpha*charity_rates + a*0.01*charity_dmmy, 1e-10)  

    elif scenario == 'prob_weight_lin_curv':

        curv = 1

        g, k, s, p_weight = params

        piece_rates = xdata['payoff_per_100']
        weight_dmmy = xdata['weight_dummy']
        prob_dmmy = xdata['prob']

        comp_1 = max(k, 1e-115)
        comp_2 = np.maximum(s+p_weight**weight_dmmy*prob_dmmy*piece_rates**curv, 1e-10)
    
    elif scenario == 'prob_weight_conc_curv':

        curv = 0.88

        g, k, s, p_weight = params

        piece_rates = xdata['payoff_per_100']
        weight_dmmy = xdata['weight_dummy']
        prob_dmmy = xdata['prob']

        comp_1 = max(k, 1e-115)
        comp_2 = np.maximum(s+p_weight**weight_dmmy*prob_dmmy*piece_rates**curv, 1e-10)

    else:

        g, k, s, p_weight, curv = params

        piece_rates = xdata['payoff_per_100']
        weight_dmmy = xdata['weight_dummy']
        prob_dmmy = xdata['prob']

        comp_1 = max(k, 1e-115)
        comp_2 = np.maximum(s+p_weight**weight_dmmy*prob_dmmy*piece_rates**curv, 1e-10)

    return (-1/g * np.log(comp_1) + 1/g*np.log(comp_2))


def sqrd_residuals_benchmark(params, xdata, logbuttonpresses, optimizer):
    """Calculates squared residuals for opt.least_squares optimizer and
    sum of squared residuals for opt.minimize optimizer under benchmark
    scenario with power cost.

    Args:
        params (list): benchmark parameters
        xdata (pd.Series): the piece rates for benchmark treatments
        logbuttonpresses (array): log of actual efforts
        optimizer (string): the name of the optimizer using this function:
            in this case, opt.least_squares or opt.minimize

    Returns:
        sqrd_resid (array): if optimizer is opt.least_squares
        sum_sqrd_resid (float): otherwise

    """
    
    g, k, s = params
    
    comp_1 = max(k, 1e-115)
    comp_2 = np.maximum(s + xdata, 1e-10)  
    
    if optimizer == 'opt.least_squares':
        sqrd_resid = ((-1/g * np.log(comp_1) +1/g * np.log(comp_2))-logbuttonpresses)**2
        return sqrd_resid

    else:
        sum_sqrd_resid = np.sum(((-1/g * np.log(comp_1) +1/g * np.log(comp_2))-logbuttonpresses)**2)
        return sum_sqrd_resid
