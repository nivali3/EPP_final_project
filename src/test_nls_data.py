"""Test to check whether forward moving average is calculated correctly.

"""
import pandas as pd
import numpy as np

from src.config import BLD
from src.config import SRC


def test_nls_data():
    expected_values = replication_data()
    actual_values = pd.read_csv(BLD / "data" / "nls_data.csv")
    assert actual_values[sorted(actual_values.columns)].equals(expected_values[sorted(expected_values.columns)])


def replication_data():

    dt = pd.read_stata(SRC / "original_data" / "mturk_clean_data_short.dta")

    # Create new variables needed for estimation:

    # Create piece-rate payoffs per 100 button presses (p)
    
    dt['payoff_per_100'] = 0
    dt.loc[dt.treatment == '1.1', 'payoff_per_100'] = 0.01
    dt.loc[dt.treatment == '1.2', 'payoff_per_100'] = 0.1
    dt.loc[dt.treatment == '1.3', 'payoff_per_100'] = 0.0
    dt.loc[dt.treatment == '2'  , 'payoff_per_100'] = 0.001
    dt.loc[dt.treatment == '1.4', 'payoff_per_100'] = 0.04
    dt.loc[dt.treatment == '4.1', 'payoff_per_100'] = 0.01
    dt.loc[dt.treatment == '4.2', 'payoff_per_100'] = 0.01
    dt.loc[dt.treatment == '6.2', 'payoff_per_100'] = 0.02
    dt.loc[dt.treatment == '6.1', 'payoff_per_100'] = 1

    # (alpha/a) create payoff per 100 to charity and dummy charity

    dt['payoff_charity_per_100'] = 0
    dt.loc[dt.treatment == '3.1', 'payoff_charity_per_100'] = 0.01
    dt.loc[dt.treatment == '3.2', 'payoff_charity_per_100'] = 0.1
    dt['charity_dummy'] = 0
    dt.loc[dt.treatment == '3.1', 'charity_dummy'] = 1
    dt.loc[dt.treatment == '3.2', 'charity_dummy'] = 1

    # (beta/delta) create payoff per 100 delayed by 2 weeks and dummy delay

    dt['delay_wks'] = 0
    dt.loc[dt.treatment == '4.1', 'delay_wks'] = 2
    dt.loc[dt.treatment == '4.2', 'delay_wks'] = 4
    dt['delay_dummy'] = 0
    dt.loc[dt.treatment == '4.1', 'delay_dummy'] = 1
    dt.loc[dt.treatment == '4.2', 'delay_dummy'] = 1

    # probability weights to back out curvature and dummy

    dt['prob'] = 1
    dt.loc[dt.treatment == '6.2', 'prob'] = 0.5
    dt.loc[dt.treatment == '6.1', 'prob'] = 0.01
    dt['weight_dummy'] = 0
    dt.loc[dt.treatment == '6.1', 'weight_dummy'] = 1

    # dummy for gift exchange

    dt['gift_dummy'] = 0
    dt.loc[dt.treatment == '10', 'gift_dummy'] = 1

    # generating effort and log effort. authors round buttonpressed to nearest 100 value. If 0 set it to 25.

    dt['buttonpresses'] = dt['buttonpresses'] + 0.1 # python rounds 50 to 0, while stata to 100. by adding a small value we avoid this mismatch
    dt['buttonpresses_nearest_100'] = round(dt['buttonpresses'],-2)
    dt.loc[dt['buttonpresses_nearest_100'] == 0, 'buttonpresses_nearest_100'] = 25
    dt['logbuttonpresses_nearest_100']  = np.log(dt['buttonpresses_nearest_100'])

    return dt
