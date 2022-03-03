"""Functions to estimate the optimal effort level with power cost and specified models, namely:

- benchmark (i.e. monetary incentive treatments), which includes treatments 1.1 ('1c PieceRate'), 1.2 ('10c PieceRate'), 1.3 ('No Payment');

- behavioral (i.e. extension from the benchmark by adding behavioral parameters), which additionally includes treatments
3.1 ('1c RedCross'), 3.2 ('10c RedCross'), 4.1 ('1c 2Wks'), 4.2 ('1c 4Wks'), 10 ('Gift Exchange');

- probability weighting (i.e. extension from the benchmark by adding probability weighting parameters), which additionally includes treatments
6.1 ('Prob.01 $1'), 6.2 ('Prob.5 2c').
"""
import numpy as np

def benchmark_power(pay100, g, k, s):
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
    assert k>0, "Only positive quantities are allowed to enter log."
    assert (s + pay100)>0, "Only positive quantities are allowed to enter log."
    
    estimated_effort = (-1/g * np.log(k) +1/g * np.log(s + pay100))
    
    return estimated_effort


def benchmark_power_least_squares(params, pay100, logbuttonpresses):
    """Estimates the optimal effort level using an alternative benchmark model with power
    cost function.
    
    Args:
        params (list): benchmark parameters
        pay100 (pd.Series): the piece rates for benchmark treatments
        logbuttonpresses (array): log of actual efforts

    Returns:
        square_res (array)

    """
    g, k, s = params
    
    assert k>0, "Only positive quantities are allowed to enter log."
    assert (s + pay100), "Only positive quantities are allowed to enter log." 
    
    square_res = 0.5*((-1/g * np.log(k) +1/g * np.log(s + pay100))-logbuttonpresses)**2
    
    return square_res


def benchmark_power_opt(params, pay100, logbuttonpresses):
    """Estimates the optimal effort level using an alternative benchmark model with power
    cost function.
    
    Args:
        params (list): benchmark parameters
        pay100 (pd.Series): the piece rates for benchmark treatments
        logbuttonpresses (array): log of actual efforts

    Returns:
        sum_square_res (float)

    """
    g, k, s = params
    
    assert k>0, "Only positive quantities are allowed to enter log."
    assert (s + pay100), "Only positive quantities are allowed to enter log."   
    
    sum_square_res = np.sum(0.5*((-1/g * np.log(k) +1/g * np.log(s + pay100))-logbuttonpresses)**2)
    
    return sum_square_res


def noweight_power(args, g, k, s, alpha, a, gift, beta, delta):
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
    pay100 = args['payoff_per_100']
    gd = args['gift_dummy']
    dd = args['delay_dummy']
    dw = args['delay_wks']
    paychar = args['payoff_charity_per_100']
    dc = args['charity_dummy']
    
    assert k>0,"Only positive quantities are allowed to enter log."
    assert (s + gift*0.4*gd + (beta**dd)*(delta**dw)*pay100 + alpha*paychar + a*0.01*dc, 1e-10)>0, "Only positive quantities are allowed to enter log."
    f_x = (-1/g * np.log(k) + 1/g*np.log(s + gift*0.4*gd + (beta**dd)*(delta**dw)*pay100 + alpha*paychar + a*0.01*dc))
    
    return f_x


def probweight_power(args, g, k, s, p_weight, curv):
    """Estimates the optimal effort level using the probability weighting model and power cost function.
    
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
    pay100 = args['payoff_per_100']
    wd = args['weight_dummy']
    prob = args['prob']
    
    assert k>0,"Only positive quantities are allowed to enter log."
    assert (s+p_weight**wd*prob*pay100**curv)>0,"Only positive quantities are allowed to enter log."
    estimated_effort = (-1/g * np.log(k) + 1/g*np.log(s+p_weight**wd*prob*pay100**curv))
    
    return estimated_effort



