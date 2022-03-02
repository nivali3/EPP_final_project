'''Generate tables which are stored in `bld/figures`.
'''
import numpy as np
import pandas as pd
import pytask
import yaml

from src.config import BLD


@pytask.mark.depends_on(
    {
        "curve fit": BLD / "analysis" / "est_benchmark_pow.yaml", 
        "least square": BLD / "analysis" / "est_benchmark_pow_alt1.yaml",
        "minimize": BLD / "analysis" / "est_benchmark_pow_alt2.yaml"
    }
)
@pytask.mark.produces(BLD / "tables" / "table_pow_comparison.csv")
def task_compare_opt_pow(depends_on, produces):
    """Load the benchmark parameter estimates (power cost function scenario) 
    computed using three different optimization functions from Scipy, namely: 
    curve_fit, least_square, minimize. Save this comparison table in a csv file in `bld/tables`.
    """
    est_dictionary = {'parameters': ["Curvature \u03B3 of cost function","Level k of cost of effort", "Intrinsic motivation s","Min obj. function"]}
    for name, values in depends_on.items():
        with open(values, "r") as stream:
            temp = yaml.safe_load(stream)
            est_dictionary[name] = list(np.round(temp['estimates'], 3)) + list(np.round([temp['min obj func']], 3))
    est_dictionary['authors'] = np.round([20.546,5.12e-13,3.17, 672.387], 3)

    est_DF = pd.DataFrame.from_dict(est_dictionary)
    est_DF.set_index('parameters', inplace=True)

    with open(produces, "w", encoding="utf-8") as y:
        est_DF.to_csv(y)

@pytask.mark.depends_on(
    {
        "bench pow": BLD / "analysis" / "est_benchmark_pow.yaml", 
        "noweight pow": BLD / "analysis" / "est_noweight_pow.yaml", 
        "bench exp": BLD / "analysis" / "est_benchmark_exp.yaml", 
        "noweight exp": BLD / "analysis" / "est_noweight_exp.yaml" 
    }
)
@pytask.mark.produces(BLD / "tables" / "table_nls_noweight_behavioral.csv")
def task_nls_noweight_behavioral(depends_on, produces):
    """Load non-linear-least-squares estimates of behavioural parameters without the weight on 
    probability (power and exponential cost function scenarios). Create a table that replicate Panel B of Table 5 of the paper.
    Save this in a csv file in `bld/tables`. 
    """
    est_dictionary = {
        'parameters': ["Curvature \u03B3 of cost function", "Level k of cost of effort", "Intrinsic motivation s","Social preferences \u03B1",
        "Warm glow coefficient a","Gift exchange \u0394s ", "Present bias \u03B2","(Weekly) discount factor \u03B4"]
    }

    for i in ['pow','exp']:
        with open(depends_on[f'bench {i}'], "r") as stream:
            temp = yaml.safe_load(stream)
            est_dictionary[f'{i}_est'] = temp['estimates'] 
            est_dictionary[f'{i}_se'] = temp['std dev']
        with open(depends_on[f'noweight {i}'], "r") as stream:
            temp = yaml.safe_load(stream)
            est_dictionary[f'{i}_est'] += temp['estimates'][3:] 
            est_dictionary[f'{i}_se'] += temp['std dev'][3:]

    est_DF = pd.DataFrame.from_dict(est_dictionary)
    with open(produces, "w", encoding="utf-8") as y:
        est_DF.to_csv(y)


@pytask.mark.depends_on(
    {
        "lin curv pow": BLD / "analysis" / "est_weight_pow_lin_curv.yaml", 
        "conc curv pow": BLD / "analysis" / "est_weight_pow_conc_curv.yaml", 
        "estimated curv pow": BLD / "analysis" / "est_weight_pow_est_curv.yaml", 
        "lin curv exp": BLD / "analysis" / "est_weight_exp_lin_curv.yaml",
        "conc curv exp": BLD / "analysis" / "est_weight_exp_conc_curv.yaml",
        "estimated curv exp": BLD / "analysis" / "est_weight_exp_est_curv.yaml" 

    }
)
@pytask.mark.produces(BLD / "tables" / "table_nls_probweight_behavioral.csv")
def task_nls_probweight_behavioral(depends_on, produces):
    """Load non-linear-least-squares estimates of model on effort in three benchmark treatments and
    two probability treatments (power and exponential cost function scenarios).
    Create a table that replicate Panel A of Table 6 of the paper.
    Save this in a csv file in `bld/tables`.
    """
    est_dictionary = {'parameters': ["Curvature \u03B3 of cost function", "Level k of cost of effort", "Intrinsic motivation s","Probability weighting \u03C0 (1%) (in %)", "Curvature of utility over piece rate"]}

    for i in ['pow','exp']:
        with open(depends_on[f'lin curv {i}'], "r") as stream:
            temp = yaml.safe_load(stream)
            est_dictionary[f'{i}_est1'] = temp['estimates'] + [1]
            est_dictionary[f'{i}_se1'] = temp['std dev'] + [0]
        with open(depends_on[f'conc curv {i}'], "r") as stream:
            temp = yaml.safe_load(stream)
            est_dictionary[f'{i}_est2'] = temp['estimates'] + [0.88]
            est_dictionary[f'{i}_se2'] = temp['std dev'] + [0]
        with open(depends_on[f'estimated curv {i}'], "r") as stream:
            temp = yaml.safe_load(stream)
            est_dictionary[f'{i}_est3'] = temp['estimates'] 
            est_dictionary[f'{i}_se3'] = temp['std dev']

    est_DF = pd.DataFrame.from_dict(est_dictionary)
    with open(produces, "w", encoding="utf-8") as y:
        est_DF.to_csv(y)
