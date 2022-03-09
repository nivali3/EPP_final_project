import numpy as np


# def sqrd_residuals_estimagic(xdata, *params, scenario):
def sqrd_residuals_estimagic(params, xdata, buttonpresses, scenario):
    """
    scenario : 'benchmark' or 'no_weight' or 'prob_weight'

    """
    if scenario == 'benchmark':

        g, k, s = params['value']

        piece_rates = xdata

        comp_1= max(k, 1e-115)
        comp_2= np.maximum(s + piece_rates, 1e-10)
        
    elif scenario == 'no_weight':

        g, k, s, alpha, a, gift, beta, delta = params['value']

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

        g, k, s, p_weight = params['value']

        piece_rates = xdata['payoff_per_100']
        weight_dmmy = xdata['weight_dummy']
        prob_dmmy = xdata['prob']

        comp_1 = max(k, 1e-115)
        comp_2 = np.maximum(s+p_weight**weight_dmmy*prob_dmmy*piece_rates**curv, 1e-10)
    
    elif scenario == 'prob_weight_conc_curv':

        curv = 0.88

        g, k, s, p_weight = params['value']

        piece_rates = xdata['payoff_per_100']
        weight_dmmy = xdata['weight_dummy']
        prob_dmmy = xdata['prob']

        comp_1 = max(k, 1e-115)
        comp_2 = np.maximum(s+p_weight**weight_dmmy*prob_dmmy*piece_rates**curv, 1e-10)

    else:

        g, k, s, p_weight, curv = params['value']

        piece_rates = xdata['payoff_per_100']
        weight_dmmy = xdata['weight_dummy']
        prob_dmmy = xdata['prob']

        comp_1 = max(k, 1e-115)
        comp_2 = np.maximum(s+p_weight**weight_dmmy*prob_dmmy*piece_rates**curv, 1e-10)

    out = {
        "value": (np.sum((-1/g * np.log(comp_1) +1/g * np.log(comp_2))-buttonpresses)**2),
        "root_contributions": ((-1/g * np.log(comp_1) +1/g * np.log(comp_2))-buttonpresses)
    }

    return out
