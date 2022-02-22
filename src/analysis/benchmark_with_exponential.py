import numpy as np

# Define the function that computes the optimal effort, what we called f(x,θ) above
# pay100 is the column we created containing the piece rate for different treatments
# g, k, s are the parameters to estimate (our θ vector). g stands for gamma.

def benchmark_exp(pay100, g, k, s, k_scaler_exp, s_scaler_exp):
    
    check1 = k/k_scaler_exp            # 'first'  component to compute f(x,θ). We call it check1 since it will enter a log, so we need to be careful with its value being > 0
    check2 = s/s_scaler_exp + pay100   # 'second' component to compute f(x,θ)
    
    f_x = (-1/g * np.log(check1) +1/g * np.log(check2))   # f(x,θ) written above
    
    return f_x

# Find the solution to the problem by non-linear least squares 


# sol = opt.curve_fit(benchmark_exp,
#                     dt.loc[dt['dummy1']==1].payoff_per_100,
#                     dt.loc[dt['dummy1']==1].buttonpresses_nearest_100,
#                     st_values_exp)

# be54 = sol[0]                        # sol[0] is the array containing our estimates
# se54 = np.sqrt(np.diagonal(sol[1]))  # sol[1] is a 3x3 variance-covariance matrix of our estimates