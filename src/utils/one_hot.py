import pandas as pd

def one_hot_encode(df: pd.DataFrame, features: list = ["Gender", "Subscription Type", "Contract Length", "Customer Status"]) -> pd.DataFrame:
    
    """One hot encodes features in the dataframe and replaces boolean values with binary values"""
    
    new_df = pd.get_dummies(df, columns=features)
    new_df = new_df.replace({True: 1, False: 0})
    new_df = new_df.drop(columns=["Gender_Male", "Customer Status_active"], axis=1)
    return new_df