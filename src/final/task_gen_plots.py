"""Generate plots corresponding to Figures 3, 4(a), 4(b), 4(c),
5 of the paper and save them in `bld/figures`.

"""
import matplotlib.pyplot as plt
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC
from src.data_management.plot_funcs import data_plot_fig_3
from src.data_management.plot_funcs import plot_CDF


@pytask.mark.depends_on(SRC / "original_data" / "mturk_clean_data_short.dta")
@pytask.mark.produces(BLD / "figures" / "figure_3.png")
def task_gen_fig_3(depends_on, produces):
    dt = pd.read_stata(depends_on)
    prepare_errorbar = data_plot_fig_3(dt)
    plt.gcf().clear()
    plt.errorbar(
        "means",
        "treatmentname",
        xerr="upper bound ci",
        data=prepare_errorbar,
        fmt="o",
        color="Black",
        elinewidth=3,
        capthick=3,
        errorevery=1,
        alpha=1,
        ms=4,
        capsize=5,
    )
    plt.gca().invert_yaxis()
    plt.title(
        "Button Presses by Treatment (from least to most effective) and Confidence Intervals"
    )
    plt.xlabel("Button Presses")
    plt.savefig(produces, bbox_inches="tight")


@pytask.mark.depends_on(SRC / "original_data" / "mturk_clean_data_short.dta")
@pytask.mark.produces(
    {
        "fig_4_a": BLD / "figures" / "figure_4_a.png",
        "fig_4_b": BLD / "figures" / "figure_4_b.png",
        "fig_4_c": BLD / "figures" / "figure_4_c.png",
    }
)
def task_gen_figs_4(depends_on, produces):
    dt = pd.read_stata(depends_on)
    plt.gcf().clear()
    fig_4_a = plot_CDF(
        dt, ["No Payment", "Very Low Pay", "1c PieceRate", "10c PieceRate"]
    )
    plt.savefig(produces["fig_4_a"], bbox_inches="tight")
    plt.gcf().clear()
    fig_4_b = plot_CDF(
        dt, ["No Payment", "Gift Exchange", "Social Comp", "Ranking", "Task Signif"]
    )
    plt.savefig(produces["fig_4_b"], bbox_inches="tight")
    plt.gcf().clear()
    fig_4_c = plot_CDF(dt, ["Gain 40c", "Loss 40c", "Gain 80c"])
    plt.savefig(produces["fig_4_c"], bbox_inches="tight")


@pytask.mark.depends_on(SRC / "original_data" / "ExpertForecastCleanLong.dta")
@pytask.mark.produces(BLD / "figures" / "figure_5.png")
def task_gen_fig_5(depends_on, produces):
    dt_expert = pd.read_stata(depends_on)
    dt_expert = dt_expert[["treatmentname", "actual", "WoC_forecast"]]
    dt_expert = dt_expert.drop_duplicates(subset="treatmentname", keep="first")
    dt_expert = dt_expert.sort_values("actual")
    plt.gcf().clear()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(
        "actual",
        "treatmentname",
        data=dt_expert,
        fmt="o",
        color="Black",
        elinewidth=3,
        capthick=3,
        errorevery=1,
        alpha=1,
        ms=4,
        capsize=5,
    )
    ax.errorbar(
        "WoC_forecast",
        "treatmentname",
        data=dt_expert,
        fmt="s",
        color="Orange",
        elinewidth=3,
        capthick=3,
        errorevery=1,
        alpha=1,
        ms=4,
        capsize=5,
    )
    ax.invert_yaxis()
    ax.set_title("Actual and Forecasted Button Presses by Treatment")
    ax.set_xlabel("Button Presses")
    ax.legend(
        ["Actual Effort", "Forecasts"], loc="best", bbox_to_anchor=(0.8, -0.2), ncol=2
    )
    fig.savefig(produces, bbox_inches="tight")
