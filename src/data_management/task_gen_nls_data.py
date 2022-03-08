"""Obtain the nls dataset used by Massimiliano and Nunnary 
in the Jupyter notebook and save this as `replicated_nls-dataset.csv` 
in `bld`.

"""
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.data_management.data_cleaning_funcs import create_nls_data


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

scenario_dummies = {'dummy1':['1.1','1.2','1.3'],
    'samplenw':['3.1','3.2','4.1','4.2','10'],
    'samplepr':['6.1','6.2']
}


@pytask.mark.depends_on(SRC / "original_data" / "mturk_clean_data_short.dta")
@pytask.mark.produces(BLD / "data" / "nls_data.csv")
def task_generate_nls_data(depends_on, produces):

    raw_data = pd.read_stata(depends_on)

    nls_data = create_nls_data(raw_data, treatment_classes, payoff_classes, scenario_dummies)
    
    with open(produces, "w") as f:
        nls_data.to_csv(f, index=False)
