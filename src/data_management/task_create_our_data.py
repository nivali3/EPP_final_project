"""General DOCSTRING
"""
import pandas as pd

from src.data_management.data_cleaning import create_nls_data

import pytask

from src.config import BLD
from src.config import SRC


@pytask.mark.depends_on(SRC/'original_data'/'mturk_clean_data_short.dta')
@pytask.mark.produces(BLD/'our_nls_data.csv')
def task_store_our_data(depends_on, produces):
    """Obtain the nls dataset used by Massimiliano and Nunnary 
    in the Jupyter notebook and save this as `replicated_nls-dataset.csv` 
    in `bld`.
    """
    # import the dataset

    dt = pd.read_stata(depends_on)

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


    treat_id_dummies = {'dummy1':['1.1','1.2','1.3'],
                    'samplenw':['3.1','3.2','4.1','4.2','10'],
                    'samplepr':['6.1','6.2']}


    our_data = create_nls_data(dt, treatment_classes, payoff_classes, treat_id_dummies)

    # Create new variables needed for estimation:

    # Create piece-rate payoffs per 100 button presses (p)


    
    with open(produces, "w") as f:
        our_data.to_csv(f, index=False)
