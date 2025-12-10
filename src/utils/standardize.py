import pandas as pd

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize all values other than CustomerID in the df"""
    temp_df = df.drop(columns=["CustomerID"], axis=1)
    norm_df = (temp_df - temp_df.min()) / (temp_df.max() - temp_df.min())

    return pd.concat([df["CustomerID"], norm_df], axis=1)
