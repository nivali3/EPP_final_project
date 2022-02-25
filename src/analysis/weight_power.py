import numpy as np
import pandas as pd



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



