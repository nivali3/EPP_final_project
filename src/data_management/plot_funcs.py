'''Auxiliary functions for generating plots corresponding to figures 3, 4(a), 4(b), 4(c) of the papaer.
'''
import pandas as pd
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt

def data_plot_fig_3(dt):
    frame = pd.DataFrame(data = dt['treatmentname'].unique(), columns=['treatmentname'])
    intervals = []
    means = []
    for i in (frame['treatmentname']):
        treat_mean = dt[dt['treatmentname']==i]['buttonpresses'].mean()
        means.append(treat_mean)

        z_critical = stats.norm.ppf(q=0.975)
        n_size = len(dt[dt['treatmentname']==i])
        treat_std = dt[dt['treatmentname']==i]['buttonpresses'].std()
        upper_bound = z_critical*(treat_std/math.sqrt(n_size))
        intervals.append(upper_bound)
    frame['means'], frame['upper bound ci'] = means, intervals
    return frame.sort_values('means')


def plot_CDF(dt, treat_names): #treat_names is a list from treatmentname column
    line_styles = ['-','--','-.',':','-.']
    colors = ['blue','red','green','orange','cyan']
    for i,j in enumerate(treat_names):
        treat_outcome = dt[dt['treatmentname']==j]['buttonpresses']
        x = np.sort(treat_outcome) #sort the data in ascending order
        y = np.arange(len(treat_outcome)) / len(treat_outcome) #get the cdf values of y
        plt.plot(x, y, color=colors[i], ls=line_styles[i], label=j)
    plt.xlim([0, 3100])
    plt.xlabel('Points in Task')
    plt.ylabel('Cumulative Fraction')
    plt.title("CDF of MTurk Workers' effort")
    plt.legend(loc='best', bbox_to_anchor=(0.8,-0.2),ncol=2)
    