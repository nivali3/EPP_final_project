"""
Converts <file>.csv into latex tables to be used as input in the final paper.
"""

import pandas as pd
from os import listdir
import pytask

from pathlib import Path

from src.config import BLD
from src.config import SRC

@pytask.mark.parametrize(
    "depends_on, produces",
    [
    (BLD / "tables" / f"{table}.csv", BLD / "paper" / f"{table}.tex")
    for table in [Path(file).stem for file in listdir(BLD / "tables")]
    ],
)
def task_gen_tex_tables(depends_on, produces):

    table = pd.read_csv(depends_on)
    with open(produces, "w", encoding="utf-8") as tf:
        tf.write(table.to_latex(na_rep="-"))