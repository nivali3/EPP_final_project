import numpy as np
import pandas as pd


def create_nls_data(data, treatment_classes, payoff_classes, treat_id_dummies):
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
    data_with_dummies =_create_dummy(data, treat_id_dummies)


    return pd.concat(objs=[data, data_with_payoffs, data_with_efforts, data_with_dummies], axis=1)


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


def _create_dummy(data, treat):
    data_with_dummies = data[['treatment']]
    for i in treat:
        data_with_dummies[i] = 0
        if i == 'dummy1':
            for j in treat[i]:
                data_with_dummies[i] += (data_with_dummies['treatment']==j).astype(int)
        else:
            data_with_dummies[i] = data_with_dummies['dummy1']
            for j in treat[i]:
                data_with_dummies[i] += (data_with_dummies['treatment']==j).astype(int)
                
    data_with_dummies.drop('treatment', axis=1, inplace=True)            
    return data_with_dummies
