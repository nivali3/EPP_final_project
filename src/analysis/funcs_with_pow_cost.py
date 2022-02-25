import numpy as np
import pandas as pd

# Define the function that computes the optimal effort, what we called f(x,θ) above
# pay100 is the column we created containing the piece rate for different treatments
# g, k, s are the parameters to estimate (our θ vector). g stands for gamma.

def benchmark_power(pay100, g, k, s):
    
    check1= max(k, 1e-115)                  # since check1 will enter log it must be greater than zero
    check2= np.maximum(s + pay100, 1e-10)   # np.maximum computes the max element wise. We do not want a negative value inside log
    
    f_x = (-1/g * np.log(check1) +1/g * np.log(check2))
    
    return f_x


def benchmark_power_least_squares(params, dt):
    
    pay100 = np.array(dt.loc[dt['dummy1']==1].payoff_per_100)
    logbuttonpresses = np.array(dt.loc[dt['dummy1']==1].logbuttonpresses_nearest_100)
    g, k, s = params
    
    check1= max(k, 1e-115)
    check2= np.maximum(s + pay100, 1e-10)   
    
    f_x = 0.5*((-1/g * np.log(check1) +1/g * np.log(check2))-logbuttonpresses)**2
    
    return f_x


def benchmark_power_opt(params, dt):
    
    pay100 = np.array(dt.loc[dt['dummy1']==1].payoff_per_100)
    logbuttonpresses = np.array(dt.loc[dt['dummy1']==1].logbuttonpresses_nearest_100)
    g, k, s = params
    
    check1= max(k, 1e-115)
    check2= np.maximum(s + pay100, 1e-10)   
    
    f_x = np.sum(0.5*((-1/g * np.log(check1) +1/g * np.log(check2))-logbuttonpresses)**2)
    
    return f_x


# Define the f(x,θ) to estimate all parameters but the probability weight in the power case

def noweight_power(args, g, k, s, alpha, a, gift, beta, delta):
    
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

#for columns 4-5-6
def probweight_power(args, g, k, s, p_weight, curv):
    
    pay100 = args['payoff_per_100']
    wd = args['weight_dummy']
    prob = args['prob']
    
    check1 = max(k, 1e-115)
    check2 = np.maximum(s+p_weight**wd*prob*pay100**curv, 1e-10)
    f_x = (-1/g * np.log(check1) + 1/g*np.log(check2))
    
    return f_x



