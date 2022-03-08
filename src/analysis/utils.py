def create_inputs(dt):
    out={}
    out['samplenw'] = {'payoff_per_100': dt.loc[dt['samplenw']==1].payoff_per_100,
        'gift_dummy': dt.loc[dt['samplenw']==1].gift_dummy,
        'delay_dummy': dt.loc[dt['samplenw']==1].delay_dummy,
        'delay_wks': dt.loc[dt['samplenw']==1].delay_wks,
        'payoff_charity_per_100': dt.loc[dt['samplenw']==1].payoff_charity_per_100,
        'charity_dummy': dt.loc[dt['samplenw']==1].charity_dummy
    }
    out['samplepr'] = {'payoff_per_100': dt.loc[dt['samplepr']==1].payoff_per_100,
        'weight_dummy': dt.loc[dt['samplepr']==1].weight_dummy,
        'prob': dt.loc[dt['samplepr']==1].prob
    }

    return out