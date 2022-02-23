import numpy as np

# Define the function that computes the optimal effort, what we called f(x,θ) above
# pay100 is the column we created containing the piece rate for different treatments
# g, k, s are the parameters to estimate (our θ vector). g stands for gamma.

def no_weight_exp(xdata, g, k, s, alpha, a, gift, beta, delta):
    
    pay100 = xdata[0]
    gd = xdata[1]
    dd = xdata[2]
    dw = xdata[3]
    paychar = xdata[4]
    dc = xdata[5]
    
    check1 = k
    check2 = s + gift*0.4*gd + (beta**dd)*(delta**dw)*pay100 + alpha*paychar +a*0.01*dc
    f_x = (-1/g * np.log(check1) + 1/g*np.log(check2))
    
    return f_x


def probweight4Exp(xdata, g, k, s, p_weight, k_scaler_exp, s_scaler_exp):
    
    pay100 = xdata[0]
    wd = xdata[1]
    prob = xdata[2]
    
    check1 = k/k_scaler_exp
    check2 = s/s_scaler_exp + p_weight**wd*prob*pay100
    
    f_x = (-1/g * np.log(check1) + 1/g*np.log(check2))
    
    return f_x


def probweight5Exp(xdata, g, k, s, p_weight, k_scaler_exp, s_scaler_exp):
    
    pay100 = xdata[0]
    wd = xdata[1]
    prob = xdata[2]
    
    check1=k/k_scaler_exp
    check2=s/s_scaler_exp + p_weight**wd*prob*pay100**0.88
    f_x = (-1/g * np.log(check1) + 1/g*np.log(check2))
    
    return f_x


def probweight6Exp(xdata, g, k, s, p_weight, curv, k_scaler_exp, s_scaler_exp):
    
    pay100 = xdata[0]
    wd = xdata[1]
    prob = xdata[2]
    
    check1=k/k_scaler_exp
    check2=s/s_scaler_exp + p_weight**wd*prob*pay100**curv
    
    f_x = (-1/g * np.log(check1) + 1/g*np.log(check2))
    
    return f_x



