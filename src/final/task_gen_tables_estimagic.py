"""Generate tables which are stored in `bld/tables`.

"""
import numpy as np
import pandas as pd
import pytask
import yaml

from src.config import BLD


@pytask.mark.depends_on({
    "nel_pow": BLD / "analysis" / "estimated_parameters_estimagic" / "benchmark_est_pow_estimagic_scipy_neldermead.yaml",
    "trf_pow": BLD / "analysis" / "estimated_parameters_estimagic" / "benchmark_est_pow_estimagic_scipy_ls_trf.yaml",
    "dog_pow": BLD / "analysis" / "estimated_parameters_estimagic" / "benchmark_est_pow_estimagic_scipy_ls_dogbox.yaml",
    "nel_exp": BLD / "analysis" / "estimated_parameters_estimagic" / "benchmark_est_exp_estimagic_scipy_neldermead.yaml",
    "trf_exp": BLD / "analysis" / "estimated_parameters_estimagic" / "benchmark_est_exp_estimagic_scipy_ls_trf.yaml",
    "dog_exp": BLD / "analysis" / "estimated_parameters_estimagic" / "benchmark_est_exp_estimagic_scipy_ls_dogbox.yaml"
})
@pytask.mark.produces(BLD / "tables" / "table_estimagic_nls_benchmark.csv")
def task_estimagic_nls_benchmark(depends_on, produces):
    """Load non-linear-least-squares estimates of behavioural parameters
    without the weight on probability (power and exponential cost function
    scenarios). Create a table that replicate Panel B of Table 5 of the
    paper. Save this in a csv file in `bld/tables`.

    """

    est_dictionary = {
        'parameters': ["Curvature \u03B3 of cost function", "Level k of cost of effort", "Intrinsic motivation s"]
    }
    
    comb = [f'{a}{b}' for a in ['nel_','trf_','dog_'] for b in ['pow','exp']]
    for i in comb:


        with open(depends_on[i], "r") as stream:
            temp = yaml.safe_load(stream)
            
            est_dictionary[f'{i}_est'] = temp['estimates'] 
            est_dictionary[f'{i}_se'] = temp['bootstrap_se']

    est_DF = pd.DataFrame.from_dict(est_dictionary)
    est_DF.set_index('parameters', inplace=True)
    est_DF.columns = pd.MultiIndex.from_product([['neldermead','ls_trf','ls_dogbox'], ['pow_est','pow_se','exp_est','exp_se']])

    with open(produces, "w", encoding="utf-8") as y:
        est_DF.to_csv(y)


@pytask.mark.depends_on({
    "nel_pow": BLD / "analysis" / "estimated_parameters_estimagic" / "no_weight_est_pow_estimagic_scipy_neldermead.yaml",
    "trf_pow": BLD / "analysis" / "estimated_parameters_estimagic" / "no_weight_est_pow_estimagic_scipy_ls_trf.yaml",
    "dog_pow": BLD / "analysis" / "estimated_parameters_estimagic" / "no_weight_est_pow_estimagic_scipy_ls_dogbox.yaml",
    "nel_exp": BLD / "analysis" / "estimated_parameters_estimagic" / "no_weight_est_exp_estimagic_scipy_neldermead.yaml",
    "trf_exp": BLD / "analysis" / "estimated_parameters_estimagic" / "no_weight_est_exp_estimagic_scipy_ls_trf.yaml",
    "dog_exp": BLD / "analysis" / "estimated_parameters_estimagic" / "no_weight_est_exp_estimagic_scipy_ls_dogbox.yaml"
})
@pytask.mark.produces(BLD / "tables" / "table_estimagic_nls_noweight_behavioral.csv")
def task_estimagic_nls_behavioral(depends_on, produces):
    """Load non-linear-least-squares estimates of behavioural
    parameters without the weight on probability (power and
    exponential cost function scenarios). Create a table that
    replicate Panel B of Table 5 of the paper. Save this in a
    csv file in `bld/tables`.

    """
    est_dictionary = {
        'parameters': [
            "Curvature \u03B3 of cost function", "Level k of cost of effort",
            "Intrinsic motivation s","Social preferences \u03B1",
            "Warm glow coefficient a","Gift exchange \u0394s ",
            "Present bias \u03B2","(Weekly) discount factor \u03B4"]
    }
    
    comb = [f'{a}{b}' for a in ['nel_','trf_','dog_'] for b in ['pow','exp']]
    for i in comb:

        with open(depends_on[i], "r") as stream:
            temp = yaml.safe_load(stream)
            
            est_dictionary[f'{i}_est'] = temp['estimates'] 
            est_dictionary[f'{i}_se'] = temp['bootstrap_se']

    est_DF = pd.DataFrame.from_dict(est_dictionary)
    est_DF.set_index('parameters', inplace=True)
    est_DF.columns = pd.MultiIndex.from_product([
        ['neldermead','ls_trf','ls_dogbox'],
        ['pow_est','pow_se','exp_est','exp_se']])

    with open(produces, "w", encoding="utf-8") as y:
        est_DF.to_csv(y)


@pytask.mark.depends_on({
    "nel_pow": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_lin_curv_est_pow_estimagic_scipy_neldermead.yaml",
    "trf_pow": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_lin_curv_est_pow_estimagic_scipy_ls_trf.yaml",
    "dog_pow": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_lin_curv_est_pow_estimagic_scipy_ls_dogbox.yaml",
    "nel_exp": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_lin_curv_est_exp_estimagic_scipy_neldermead.yaml",
    "trf_exp": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_lin_curv_est_exp_estimagic_scipy_ls_trf.yaml",
    "dog_exp": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_lin_curv_est_exp_estimagic_scipy_ls_dogbox.yaml"
    })
@pytask.mark.produces(BLD / "tables" / "table_estimagic_nls_probweight_lin_curv.csv")
def task_estimagic_nls_probweight_lin_curv(depends_on, produces):
    """Load non-linear-least-squares estimates of behavioural parameters
    without the weight on probability (power and exponential cost
    function scenarios). Create a table that replicate Panel B of
    Table 5 of the paper. Save this in a csv file in `bld/tables`.

    """

    est_dictionary = {
        'parameters': [
            "Curvature \u03B3 of cost function", "Level k of cost of effort", 
            "Intrinsic motivation s","Probability weighting \u03C0 (1%) (in %)",
            "Curvature of utility over piece rate"]
    }
    
    comb = [f'{a}{b}' for a in ['nel_','trf_','dog_'] for b in ['pow','exp']]
    for i in comb:

        with open(depends_on[i], "r") as stream:
            temp = yaml.safe_load(stream)
            
            est_dictionary[f'{i}_est'] = temp['estimates'] + [1]
            est_dictionary[f'{i}_se'] = temp['bootstrap_se'] + [0]

    est_DF = pd.DataFrame.from_dict(est_dictionary)
    est_DF.set_index('parameters', inplace=True)
    est_DF.columns = pd.MultiIndex.from_product(
        [['neldermead','ls_trf','ls_dogbox'], ['pow_est','pow_se','exp_est','exp_se']],
        names=['algorithm','linear curvature'])

    with open(produces, "w", encoding="utf-8") as y:
        est_DF.to_csv(y)


@pytask.mark.depends_on({
    "nel_pow": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_conc_curv_est_pow_estimagic_scipy_neldermead.yaml",
    "trf_pow": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_conc_curv_est_pow_estimagic_scipy_ls_trf.yaml",
    "dog_pow": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_conc_curv_est_pow_estimagic_scipy_ls_dogbox.yaml",
    "nel_exp": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_conc_curv_est_exp_estimagic_scipy_neldermead.yaml",
    "trf_exp": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_conc_curv_est_exp_estimagic_scipy_ls_trf.yaml",
    "dog_exp": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_conc_curv_est_exp_estimagic_scipy_ls_dogbox.yaml"
    })
@pytask.mark.produces(BLD / "tables" / "table_estimagic_nls_probweight_conc_curv.csv")
def task_estimagic_nls_probweight_conc_curv(depends_on, produces):
    """Load non-linear-least-squares estimates of behavioural parameters
    without the weight on probability (power and exponential cost
    function scenarios). Create a table that replicate Panel B of
    Table 5 of the paper. Save this in a csv file in `bld/tables`.

    """

    est_dictionary = {
        'parameters': [
            "Curvature \u03B3 of cost function", "Level k of cost of effort",
            "Intrinsic motivation s","Probability weighting \u03C0 (1%) (in %)",
            "Curvature of utility over piece rate"]
    }
    
    comb = [f'{a}{b}' for a in ['nel_','trf_','dog_'] for b in ['pow','exp']]
    for i in comb:

        with open(depends_on[i], "r") as stream:
            temp = yaml.safe_load(stream)
            
            est_dictionary[f'{i}_est'] = temp['estimates'] + [0.88]
            est_dictionary[f'{i}_se'] = temp['bootstrap_se'] + [0]

    est_DF = pd.DataFrame.from_dict(est_dictionary)
    est_DF.set_index('parameters', inplace=True)
    est_DF.columns = pd.MultiIndex.from_product(
        [['neldermead','ls_trf','ls_dogbox'], ['pow_est','pow_se','exp_est','exp_se']],
        names=['algorithm','concave curvature'])

    with open(produces, "w", encoding="utf-8") as y:
        est_DF.to_csv(y)


@pytask.mark.depends_on({
    "nel_pow": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_est_curv_est_pow_estimagic_scipy_neldermead.yaml",
    "trf_pow": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_est_curv_est_pow_estimagic_scipy_ls_trf.yaml",
    "dog_pow": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_est_curv_est_pow_estimagic_scipy_ls_dogbox.yaml",
    "nel_exp": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_est_curv_est_exp_estimagic_scipy_neldermead.yaml",
    "trf_exp": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_est_curv_est_exp_estimagic_scipy_ls_trf.yaml",
    "dog_exp": BLD / "analysis" / "estimated_parameters_estimagic" / "prob_weight_est_curv_est_exp_estimagic_scipy_ls_dogbox.yaml"
    })
@pytask.mark.produces(BLD / "tables" / "table_estimagic_nls_probweight_est_curv.csv")
def task_estimagic_nls_probweight_est_curv(depends_on, produces):
    """Load non-linear-least-squares estimates of behavioural parameters
    without the weight on probability (power and exponential cost function
    scenarios). Create a table that replicate Panel B of Table 5 of the paper.
    Save this in a csv file in `bld/tables`.

    """

    est_dictionary = {
        'parameters': [
            "Curvature \u03B3 of cost function", "Level k of cost of effort",
            "Intrinsic motivation s","Probability weighting \u03C0 (1%) (in %)",
            "Curvature of utility over piece rate"]
    }
    
    comb = [f'{a}{b}' for a in ['nel_','trf_','dog_'] for b in ['pow','exp']]
    for i in comb:

        with open(depends_on[i], "r") as stream:
            temp = yaml.safe_load(stream)
            
            est_dictionary[f'{i}_est'] = temp['estimates'] 
            est_dictionary[f'{i}_se'] = temp['bootstrap_se']

    est_DF = pd.DataFrame.from_dict(est_dictionary)
    est_DF.set_index('parameters', inplace=True)
    est_DF.columns = pd.MultiIndex.from_product(
        [['neldermead','ls_trf','ls_dogbox'], ['pow_est','pow_se','exp_est','exp_se']],
        names=['algorithm','estimated curvature'])

    with open(produces, "w", encoding="utf-8") as y:
        est_DF.to_csv(y)
