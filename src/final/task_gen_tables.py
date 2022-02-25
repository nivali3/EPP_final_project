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
@pytask.mark.produces(BLD / "tables" / "table_pow_comparison.yaml")
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


    runtimes_dictionary = {'parameters': ["Curvature γ of cost function","Level k of cost of effort", "Intrinsic motivation s","Min obj. function"]}


    # with open(depends_on["pandas"], "r") as stream:
    #     runtimes_dictionary = yaml.safe_load(stream)

    # depends_on.pop("pandas", None)
    


    #for name, values in list(zip(col, depends_on.values())):
    for name, values in depends_on.items():

        with open(values, "r") as stream:
            temp = yaml.safe_load(stream)
            runtimes_dictionary[name] = temp['estimates'] + [temp['min_obj_func']]
    runtimes_dictionary['authors'] = [20.546,5.12e-13,3.17, 672.387]

    with open(produces, "w") as y:
        yaml.dump(runtimes_dictionary, y)

@pytask.mark.depends_on(
    {
        "power": BLD / "analysis" / "est_benchmark_exp.yaml",
        "exponential": BLD / "analysis" / "est_noweight_exp.csv",
        "exponentil": BLD / "analysis" / "est_benchmark_pow.yaml",
        "exponial": BLD / "analysis" / "est_noweight_pow.csv"
    }
)
@pytask.mark.produces(BLD / "tables" / "table_nls_est_behavioral.yaml")
def task_nls_est_behavioral(depends_on, produces):
    """Load all the runtime information, combine it into a DataFrame with
    one column per function and also calculate the relative speedup of
    each version compared to `pandas_batch_update`using the median
    runtime to calculate relative improvement. Save this as `summary.csv`
    in `bld`.
    """
    runtimes_dictionary = {'parameters': ["Curvature γ of cost function", "Level k of cost of effort", "Intrinsic motivation s","Social preferences α",
        "Warm glow coefficient a","Gift exchange Δs", "Present bias β","(Weekly) discount factor δ"]}

    #col = ['power_est', 'power_se', 'exp_est','exp_se']

    with open(depends_on['power'], "r") as stream:
        temp = yaml.safe_load(stream)
    runtimes_dictionary['power_est'] = temp['estimates'] 
    runtimes_dictionary['power_se'] = temp['variances']

    with open(depends_on['exponential'], "r") as stream:
        temp = yaml.safe_load(stream)
    runtimes_dictionary['exp_est'] = temp['estimates'] 
    runtimes_dictionary['exp_se'] = temp['variances']

    with open(produces, "w") as y:
        yaml.dump(runtimes_dictionary, y)


