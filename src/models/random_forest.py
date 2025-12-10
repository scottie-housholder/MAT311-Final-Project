import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_forest_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train and return a KNN classifier."""
    model = RandomForestClassifier(n_jobs=-1)
    model.fit(X_train, y_train)
    return model