import pandas as pd
import numpy as np

import cellfinder.tools.df as df_tools


def test_sanitise_df():
    columns = ["name", "number"]

    data_with_nan = [["one", np.nan], ["two", 15], ["three", np.nan]]
    df_with_nan = pd.DataFrame(data_with_nan, columns=columns)

    data_with_inf = [["one", np.inf], ["two", 15], ["three", np.inf]]
    df_with_inf = pd.DataFrame(data_with_inf, columns=columns)
    sanitised_df = df_tools.sanitise_df(df_with_inf)

    assert sanitised_df.equals(df_with_nan)
