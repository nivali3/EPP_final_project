import numpy as np
import pandas as pd


def create_nls_data(data, treatment_classes, payoff_classes):
    """DOCSTRING
    data : initial data-mturk_clean_data_short
    """

    data_with_payoffs = _create_new_payoffs(data, treatment_classes, payoff_classes)
    data_with_efforts = _create_efforts(data)
    data.drop('buttonpresses', axis=1, inplace=True)

    nls_data = pd.concat(objs=[data, data_with_payoffs, data_with_efforts], axis=1)

    return nls_data


def _create_new_payoffs(data, treatment_classes, payoff_classes):
    """DOCSTRING
    data : initial data-mturk_clean_data_short
    """
    data_with_payoffs = data[['treatment']]
    
    for key in treatment_classes.keys():
        data_with_payoffs[key] = 1 if key == 'prob' else 0
        for i,j in list(zip(treatment_classes[key], payoff_classes[key])):
            data_with_payoffs.loc[data_with_payoffs.treatment == i, key] = j
    
    data_with_payoffs.drop('treatment', axis=1, inplace=True)

    return data_with_payoffs


def _create_efforts(data):
    """DOCSTRING
    """
    data_with_efforts = data[['buttonpresses']]
    
    data_with_efforts += 0.1 # python rounds 50 to 0, while stata to 100. by adding a small value we avoid this mismatch
    data_with_efforts['buttonpresses_nearest_100'] = round(data_with_efforts,-2)
    data_with_efforts.loc[data_with_efforts['buttonpresses_nearest_100'] == 0, 'buttonpresses_nearest_100'] = 25
    data_with_efforts['logbuttonpresses_nearest_100']  = np.log(data_with_efforts['buttonpresses_nearest_100'])
    
    return data_with_efforts


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


# # For testing
# sorted(new_data.columns)
# sorted(dt.columns)

# a = np.where(new_data[sorted(new_data.columns)] != dt[sorted(dt.columns)])
# a
