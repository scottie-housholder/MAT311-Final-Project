import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

def importance_visuals(df: pd.DataFrame) -> None:
    X = df.drop(["Churn", "Customer Status_inactive"], axis=1).values
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=754, stratify=y)

    cols = df.drop(["Churn", "Customer Status_inactive"], axis=1).columns

    find_features = Lasso(alpha=0.00001)
    find_features.fit(X_train, y_train)

    importance = np.abs(find_features.coef_)

    sns.barplot(x=cols, y=importance)
    plt.xticks(rotation=90)
    plt.title('Feature Importance')
    plt.show()