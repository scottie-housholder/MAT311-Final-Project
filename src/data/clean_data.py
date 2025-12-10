import pandas as pd
from ..utils.one_hot import one_hot_encode
from ..utils.standardize import standardize

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataframe by one hot encoding non-numeric values and filling null values"""

    df["Support Calls"] = df["Support Calls"].replace({"none": 0}).astype("Float64")
    df = one_hot_encode(df)
    df["Tenure"] = df["Tenure"].fillna(df["Tenure"].mean())
    df["Support Calls"] = df["Support Calls"].fillna(df["Support Calls"].mean())
    df["Last Interaction"] = df["Last Interaction"].fillna(df["Last Interaction"].mean())

    df = df.drop(["Last Payment Date", "Last Due Date"], axis=1)

    df = standardize(df)
    return df



# if __name__ == "__main__":
#     train = pd.read_csv("data/raw/train.csv")
#     print(train.isna().sum())
#     train = clean_data(train)
#     print(train.isna().sum())