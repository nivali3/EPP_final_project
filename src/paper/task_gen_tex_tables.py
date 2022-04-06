"""
Converts <file>.csv into latex tables to be used as input in the final paper.

"""
from os import listdir
from pathlib import Path

import pandas as pd
import pytask

from src.config import BLD


@pytask.mark.parametrize(
    "depends_on, produces, table",
    [
        (BLD / "tables" / f"{table}.csv", BLD / "paper" / f"{table}.tex", table)
        for table in [Path(file).stem for file in listdir(BLD / "tables")]
    ],
)
def task_gen_tex_tables(depends_on, produces, table):

    if "estimagic" in table:
        tab = pd.read_csv(depends_on, header=[0, 1], index_col=0)
    else:
        tab = pd.read_csv(depends_on, index_col=0)

    with open(produces, "w", encoding="utf-8") as tf:
        tf.write(tab.to_latex(na_rep="", bold_rows=True))
