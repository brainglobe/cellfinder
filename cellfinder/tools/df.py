import numpy as np


def sanitise_df(df):
    """
    Replaces infinite values in a dataframe with NaN
    :param df: Any dataframe
    :return: Dataframe with Inf replaced with NaN
    """
    df = df.replace(np.inf, np.nan)
    return df
