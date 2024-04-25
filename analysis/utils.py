from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Union
import plotly.graph_objects as go
import plotly.express as px

def write_figure(
    filename: str, fig: go.Figure, path: Union[Path, str] = "figures"
):
    """Write a Figure to a file.

    Parameters
    ----------
    filename : str
        The name of the file to write to.
    fig : go.Figure
        The plotly figure object.
    path : Union[Path, str], optional
        The path where the file will be stored, by default figures_path
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    fname = path / filename
    fig.write_image(fname)

    print(f"Figure written to: {fname}")
    print(f"Filename   :", filename)
    print(f"Latex label:", filename.replace(".pdf", ""))


def write_latex_table(
    filename: str, table: str, path: Union[Path, str] = "tables"
):
    """Write a latex table to a file.

    Parameters
    ----------
    filename : str
        The name of the file to write to.
    table : str
        The table, as a string.
    path : Union[Path, str], optional
        The path where the file will be stored, by default latex_tables_path
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    fname = path / filename
    with fname.open("w") as f:
        f.write(
            "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        )
        f.write(
            "%% WARNING: DO NOT CHANGE THIS FILE. IT IS GENERATED AUTOMATICALLY %\n"
        )
        f.write(
            "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        )
        f.write(table)
    print(f"Table written to: {fname}")


def aggregate_mean_std(
    df: pd.DataFrame,
    group_by: List[str],
    keys_to_aggregate: List[str],
) -> pd.DataFrame:
    """Group and aggregate columns of a dataframe, using mean and std.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to aggregate.
    group_by : List[str]
        The columns used to group the dataframe.
    keys_to_aggregate : List[str]
        The column names that will be aggregated.

    Returns
    -------
    pd.DataFrame
        A dataframe with the aggregated values.

    Example
    -------
    >>> df = pd.DataFrame({
    ...     "a": [1, 1, 2, 2],
    ...     "b": [1, 2, 3, 4],
    ...     "c": [5, 6, 7, 8],
    ... })
    >>> aggregate_mean_std(df, ["a"], ["b", "c"])
       a    b  b_std    c  c_std
    0  1 1.500  0.707 5.500  0.707
    1  2 3.500  0.707 7.500  0.707

    """
    x = (
        df.groupby(group_by)[keys_to_aggregate]
        .agg("mean")
        .join(
            df.groupby(group_by)[keys_to_aggregate].agg("std"), rsuffix="_std"
        )
    )
    return x.reset_index()