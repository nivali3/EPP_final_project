import numpy as np
import pandas as pd


def create_nls_data(data, treatment_classes, payoff_classes):
    """Generate the dataset suitable for NLS by adding columns for payoffs and efforts to the original data.
    Args:
        data (dataset): initial data-mturk_clean_data_short
        treatment_classes (dict): dictionary containing treatment identifiers
        payoff_classes (dict): dictionary containing payoff. It must have the same keys and the same number of
        values for each key as treatment_classes
 
    Returns:
        (pd.DataFrame): dataset ready to be used for NLS analysis
    """

    data_with_payoffs = _create_new_payoffs(data, treatment_classes, payoff_classes)
    data_with_efforts = _create_efforts(data)
    data.drop('buttonpresses', axis=1, inplace=True) 

    return pd.concat(objs=[data, data_with_payoffs, data_with_efforts], axis=1)


def _create_new_payoffs(data, treatment_classes, payoff_classes):
    """Generate the dataset whose columns containt the treatments' payoffs 
    Args:
        data (dataset): initial data-mturk_clean_data_short
        treatment_classes (dict): dictionary containing treatment identifiers
        payoff_classes (dict): dictionary containing payoff. It must have the same keys and the same number of
        values for each key as treatment_classes
 
    Returns:
        data_with_payoffs (pd.DataFrame): dataset with payoffs
    """
    data_with_payoffs = data[['treatment']]
    
    for key in treatment_classes.keys():
        data_with_payoffs[key] = 1 if key == 'prob' else 0
        for i,j in list(zip(treatment_classes[key], payoff_classes[key])):
            data_with_payoffs.loc[data_with_payoffs.treatment == i, key] = j
    
    data_with_payoffs.drop('treatment', axis=1, inplace=True)

    return data_with_payoffs


def _create_efforts(data):
    """Generate the dataset whose columns containt the treatments' efforts 
    Args:
        data (dataset): initial data-mturk_clean_data_short
 
    Returns:
        data_with_efforts (pd.DataFrame): dataset with efforts
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
    'charity_dummy' : ['3.1', '3.2'],
    'delay_wks' : ['4.1', '4.2'],
    'delay_dummy' : ['4.1', '4.2'],
    'prob' : ['6.1', '6.2'],
    'weight_dummy' : ['6.1'],
    'gift_dummy' : ['10']
}


payoff_classes = {
    'payoff_per_100' : [0.01, 0.1, 0.0, 0.001, 0.04, 0.01, 0.01, 0.02, 1],
    'payoff_charity_per_100' : [0.01, 0.1],
    'charity_dummy' : [1, 1],
    'delay_wks' : [2, 4],
    'delay_dummy' : [1, 1],
    'prob' : [0.01, 0.5],
    'weight_dummy' : [1],
    'gift_dummy' : [1]
}

