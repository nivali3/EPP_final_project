import numpy as np
import pandas as pd
import pytask
import yaml

from src.config import BLD


@pytask.mark.depends_on(
    {
        "pandas": BLD / "analysis" / "est_benchmark_pow.yaml",
        "fast": BLD / "analysis" / "est_benchmark_pow_alt1.yaml",
        "faster": BLD / "analysis" / "est_benchmark_pow_alt2.yaml"
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


    runtimes_dictionary = {}


    # with open(depends_on["pandas"], "r") as stream:
    #     runtimes_dictionary = yaml.safe_load(stream)

    # depends_on.pop("pandas", None)
    a = ['curve_fit', 'least_square', 'minimize_nd']


    for name, values in list(zip(a, depends_on.values())):

        with open(values, "r") as stream:
            temp = yaml.safe_load(stream)
            runtimes_dictionary[name] = temp['estimates'] + [temp['min_obj_func']]

    with open(produces, "w") as y:
        yaml.dump(runtimes_dictionary, y)

