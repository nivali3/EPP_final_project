import numpy as np


def benchmark_exp(pay100, g, k, s):
    
    check1 = k            # 'first'  component to compute f(x,θ). We call it check1 since it will enter a log, so we need to be careful with its value being > 0
    check2 = s + pay100   # 'second' component to compute f(x,θ)
    
    f_x = (-1/g * np.log(check1) +1/g * np.log(check2))   # f(x,θ) written above
    
    return f_x


def no_weight_exp(args, g, k, s, alpha, a, gift, beta, delta):

    pay100 = args['payoff_per_100']
    gd = args['gift_dummy']
    dd = args['delay_dummy']
    dw = args['delay_wks']
    paychar = args['payoff_charity_per_100']
    dc = args['charity_dummy']
    
    check1 = k
    check2 = s + gift*0.4*gd + (beta**dd)*(delta**dw)*pay100 + alpha*paychar +a*0.01*dc
    f_x = (-1/g * np.log(check1) + 1/g*np.log(check2))
    
    return f_x


def prob_weight_exp(args, g, k, s, p_weight, curv):
    '''DOCSTRING
    '''
    pay100 = args['payoff_per_100']
    wd = args['weight_dummy']
    prob = args['prob']
    
    check1=k
    check2=s + p_weight**wd*prob*pay100**curv
    
    f_x = (-1/g * np.log(check1) + 1/g*np.log(check2))
    
    return f_x
