import numpy as np
import pandas as pd
import pytask
import yaml

from src.config import BLD


@pytask.mark.depends_on(
    {
        "curve_fit": BLD / "analysis" / "est_benchmark_pow.yaml", 
        "least_square": BLD / "analysis" / "est_benchmark_pow_alt1.yaml",
        "minimize_nd": BLD / "analysis" / "est_benchmark_pow_alt2.yaml"
    }
)
@pytask.mark.produces(BLD / "tables" / "table_pow_comparison.csv")
def task_compare_opt_pow(depends_on, produces):
    """Load all the runtime information, combine it into a DataFrame with
    one column per function and also calculate the relative speedup of
    each version compared to `pandas_batch_update`using the median
    runtime to calculate relative improvement. Save this as `summary.csv`
    in `bld`.
    """


    # final = {'estimates' : bp52.tolist(),
    #     'min_obj_func' : (2*benchmark_power_opt(bp52, dt)).tolist()}

    # with open(produces, "w") as y:
    #     yaml.dump(final, y)


    est_dictionary = {'parameters': ["Curvature \u03B3 of cost function","Level k of cost of effort", "Intrinsic motivation s","Min obj. function"]}


    # with open(depends_on["pandas"], "r") as stream:
    #     runtimes_dictionary = yaml.safe_load(stream)

    # depends_on.pop("pandas", None)
    


    #for name, values in list(zip(col, depends_on.values())):
    for name, values in depends_on.items():

        with open(values, "r") as stream:
            temp = yaml.safe_load(stream)
            est_dictionary[name] = temp['estimates'] + [temp['min obj func']]
    est_dictionary['authors'] = [20.546,5.12e-13,3.17, 672.387]

    est_DF = pd.DataFrame.from_dict(est_dictionary)

    #with open(produces, "w") as y:
    #    yaml.dump(runtimes_dictionary, y)
    with open(produces, "w", encoding="utf-8") as y:
        est_DF.to_csv(y)


# !!! this task needs to be adjusted with anoter version using partial instead of computing again the 3 benchmark parameters
@pytask.mark.depends_on(
    {
        "bench pow": BLD / "analysis" / "est_benchmark_pow.yaml", #bp52 sp52
        "noweight pow": BLD / "analysis" / "est_noweight_pow.yaml", #bp53 sp53
        "bench exp": BLD / "analysis" / "est_benchmark_exp.yaml", #be54 se54
        "noweight exp": BLD / "analysis" / "est_noweight_exp.yaml" #be56 se56
    }
)
@pytask.mark.produces(BLD / "tables" / "table_nls_noweight_behavioral.csv")
def task_nls_est_behavioral(depends_on, produces):
    """Load all the runtime information, combine it into a DataFrame with
    one column per function and also calculate the relative speedup of
    each version compared to `pandas_batch_update`using the median
    runtime to calculate relative improvement. Save this as `summary.csv`
    in `bld`.
    """
    est_dictionary = {'parameters': ["Curvature \u03B3 of cost function", "Level k of cost of effort", "Intrinsic motivation s","Social preferences \u03B1",
        "Warm glow coefficient a","Gift exchange \u0394s ", "Present bias \u03B2","(Weekly) discount factor \u03B4"]}

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
        "lin curv pow": BLD / "analysis" / "est_weight_pow_lin_curv.yaml", #bp61 sp61
        "conc curv pow": BLD / "analysis" / "est_weight_pow_conc_curv.yaml", #bp62 sp62
        "estimated curv pow": BLD / "analysis" / "est_weight_pow_est_curv.yaml", #bp63 sp63
        "lin curv exp": BLD / "analysis" / "est_weight_exp_lin_curv.yaml",#be64 se64
        "conc curv exp": BLD / "analysis" / "est_weight_exp_conc_curv.yaml",#be65 se65
        "estimated curv exp": BLD / "analysis" / "est_weight_exp_est_curv.yaml" #be66 se66

    }
)
@pytask.mark.produces(BLD / "tables" / "table_nls_probweight_behavioral.csv")
def task_nls_est_behavioral(depends_on, produces):
    """Load all the runtime information, combine it into a DataFrame with
    one column per function and also calculate the relative speedup of
    each version compared to `pandas_batch_update`using the median
    runtime to calculate relative improvement. Save this as `summary.csv`
    in `bld`.
    """
    est_dictionary = {'parameters': ["Curvature \u03B3 of cost function", "Level k of cost of effort", "Intrinsic motivation s","Probability weighting \u03C0 (1%) (in %)", "Curvature of utility over piece rate"]}

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
