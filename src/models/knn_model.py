import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from ..utils.find_k import find_k


def train_knn_model(X_train: pd.DataFrame, y_train: pd.Series) -> KNeighborsClassifier:
    """Train and return a KNN classifier."""
    model = KNeighborsClassifier(n_neighbors=find_k(len(X_train)))
    model.fit(X_train, y_train)
    return model
