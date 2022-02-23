import numpy as np

# Define the function that computes the optimal effort, what we called f(x,θ) above
# pay100 is the column we created containing the piece rate for different treatments
# g, k, s are the parameters to estimate (our θ vector). g stands for gamma.

def benchmark_power(pay100, g, k, s):
    
    check1= max(k, 1e-115)                  # since check1 will enter log it must be greater than zero
    check2= np.maximum(s + pay100, 1e-10)   # np.maximum computes the max element wise. We do not want a negative value inside log
    
    f_x = (-1/g * np.log(check1) +1/g * np.log(check2))
    
    return f_x


def benchmark_power_least_squares(dt, params, k_scaler_power, s_scaler_power):
    
    pay100 = np.array(dt.loc[dt['dummy1']==1].payoff_per_100)
    logbuttonpresses = np.array(dt.loc[dt['dummy1']==1].logbuttonpresses_nearest_100)
    g, k, s = params
    
    check1= max(k/k_scaler_power, 1e-115)
    check2= np.maximum(s/s_scaler_power + pay100, 1e-10)   
    
    f_x = 0.5*((-1/g * np.log(check1) +1/g * np.log(check2))-logbuttonpresses)**2
    
    return f_x


def benchmark_power_opt(dt, params, k_scaler_power, s_scaler_power):
    
    pay100 = np.array(dt.loc[dt['dummy1']==1].payoff_per_100)
    logbuttonpresses = np.array(dt.loc[dt['dummy1']==1].logbuttonpresses_nearest_100)
    g, k, s = params
    
    check1= max(k/k_scaler_power, 1e-115)
    check2= np.maximum(s/s_scaler_power + pay100, 1e-10)   
    
    f_x = np.sum(0.5*((-1/g * np.log(check1) +1/g * np.log(check2))-logbuttonpresses)**2)
    
    return f_x


