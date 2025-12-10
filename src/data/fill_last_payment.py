import pandas as pd

def fill_last_payment(train_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fills the 'Last Payment Date' col in the train data df
    Must be run before clean_data.py
    """
    train_data["Last Payment Date"] = pd.to_datetime(train_data["Last Payment Date"], format='%m-%d')
    train_data["Last Due Date"] = pd.to_datetime(train_data["Last Due Date"], format='%m-%d')
    train_data["Payment Delay"] = (train_data["Last Payment Date"] - train_data["Last Due Date"]).dt.days.astype("int")
    return train_data