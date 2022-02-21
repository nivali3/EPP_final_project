import numpy as np
import pandas as pd


def create_new_payoffs(data, treatment_classes, payoff_classes):
    """DOCSTRING
    """

    for k1 in treatment_classes.keys():
        data[k1] = 1 if k1 == 'prob' else 0
        for i,j in list(zip(treatment_classes[k1], payoff_classes[k1])):
            data.loc[data.treatment == i, k1] = j

    return data


def create_efforts(data, buttonpresses):
    """DOCSTRING
    """
    data.buttonpresses = data.buttonpresses + 0.1 # python rounds 50 to 0, while stata to 100. by adding a small value we avoid this mismatch
    data['buttonpresses_nearest_100'] = round(data.buttonpresses,-2)
    data.loc[data.buttonpresses_nearest_100 == 0, 'buttonpresses_nearest_100'] = 25
    data['logbuttonpresses_nearest_100']  = np.log(data['buttonpresses_nearest_100'])
    
    return data


treatment_classes = {
    'payoff_per_100' : ['1.1', '1.2', '1.3', '2', '1.4', '4.1', '4.2', '6.2', '6.1'],
    'payoff_charity_per_100' : ['3.1', '3.2'],
    'dummy_charity' : ['3.1', '3.2'],
    'delay_wks' : ['4.1', '4.2'],
    'delay_dummy' : ['4.1', '4.2'],
    'prob' : ['6.1', '6.2'],
    'weight_dummy' : ['6.1'],
    'gift_dummy' : ['10']
}


payoff_classes = {
    'payoff_per_100' : [0.01, 0.1, 0.0, 0.001, 0.04, 0.01, 0.01, 0.02, 1],
    'payoff_charity_per_100' : [0.01, 0.1],
    'dummy_charity' : [1, 1],
    'delay_wks' : [2, 4],
    'delay_dummy' : [1, 1],
    'prob' : [0.01, 0.5],
    'weight_dummy' : [1],
    'gift_dummy' : [1]
}


data = pd.read_stata('../original_data/mturk_clean_data_short.dta')

data = create_new_payoffs(data, treatment_classes, payoff_classes)
data = create_efforts(data, 'buttonpresses')

# # For testing
# data.equals(dt)

# a = np.where(data != dt)
# a
