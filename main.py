#Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from src.data.load_data import load_dataset
from src.data.clean_data import clean_data
from src.data.fill_last_payment import fill_last_payment
from src.visualization.important_features import importance_visuals
from src.models.knn_model import train_knn_model
from src.models.random_forest import train_forest_model
from src.visualization.performance import plot_confusion_matrices, plot_performance_comparison
from src.models.train_model import plot_roc_curve

def main() -> None:
    print("---Loading data...")
    train = load_dataset("data/raw/train.csv")
    test = load_dataset("data/raw/test.csv")

    print("---Cleaning data...")
    train = fill_last_payment(train)
    train_clean = clean_data(train)
    # test_clean = clean_data(test)

    print(f"Cleaned dataset shape: {train_clean.shape}")

    print("---Visualizing data...")
    importance_visuals(train_clean)

    print("---Splitting data...")
    X = train_clean[['Support Calls', "Contract Length_Monthly", "Payment Delay", "Total Spend", "Age"]].values
    y = train_clean["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=754, stratify=y)

    print("---Training Models...")
    knn_model = train_knn_model(X_train, y_train)
    forest_model = train_forest_model(X_train, y_train)

    print("---Evaluating on validation set...")
    y_test_pred_knn = knn_model.predict(X_test)
    y_test_pred_forest = forest_model.predict(X_test)

    val_prob_knn = knn_model.predict_proba(X_test)[:, 1]
    val_prob_forest = forest_model.predict_proba(X_test)[:, 1]

    plot_confusion_matrices(y_test, y_test_pred_forest, y_test_pred_knn)
    plot_performance_comparison(y_test, y_test_pred_forest, y_test_pred_knn)

    auc_dumb = plot_roc_curve(y_test, val_prob_forest, "Random Forest")
    auc_knn = plot_roc_curve(y_test, val_prob_knn, "KNN")

    print("---Finding best model...")
    best_model = knn_model if auc_knn >= auc_dumb else forest_model
    best_label = "KNN" if best_model is knn_model else "Random Forest"

    print(f"---Testing best model ({best_label})...")
    y_test_pred = best_model.predict(X_test)
    test_prob = best_model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, test_prob, f"Test {best_label}")

    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Best Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    print("Done.")

if __name__ == "__main__":
    main()
