"""TO BE FIXED Functions for the predict step of a square root unscented Kalman filter.

TO BE FIXED The functions use pandas for most of the calculations. This means that for
most operations the order of columns or the index is irrelevant. Nevertheless,
the order might be relevant whenever matrix factorizations are involved!

the benchmark model - the model for treatments 1. 1, 1.2, and 1.3
using the costly effort model with exponential cost function

"""

import numpy as np


def benchmark_exp(pay100, g, k, s):
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
    
    assert k > 0, "Only positive quantities allowed for *k*."
    assert (s + pay100) > 0, "Only positive quantities allowed for the sum of *s* and *pay100*."

    estimated_effort = (-1/g * np.log(k) +1/g * np.log(s + pay100))
    
    return estimated_effort


def no_weight_exp(args, g, k, s, alpha, a, gift, beta, delta):
    """Estimates the optimal effort level using the general model without probability weighting 
    and with exponential cost function.
    
    Args:
        pay100 (pd.Series): the piece rates for all treatments except probability weighting ones
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
    
    check1 = k
    check2 = s + gift*0.4*gd + (beta**dd)*(delta**dw)*pay100 + alpha*paychar +a*0.01*dc
    estimated_effort = (-1/g * np.log(check1) + 1/g*np.log(check2))
    
    return estimated_effort


def prob_weight_exp(args, g, k, s, p_weight, curv):
    """Estimates the optimal effort level using the general model with probability weighting 
    and exponential cost function.
    
    Args:
        pay100 (pd.Series): the piece rates for all treatments except probability weighting ones
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
    
    check1=k
    check2=s + p_weight**wd*prob*pay100**curv
    
    f_x = (-1/g * np.log(check1) + 1/g*np.log(check2))
    
    return f_x
