# MAT311-Final-Project

## Project layout

```
.
├── main.py                 # Entry point that runs the entire pipeline
├── requirements.txt        # Python dependencies
├── data/
│   ├── processed/          # Created after running the pipeline
│   │   └── test_cleaned_temp.csv
│   │   └── train_cleaned_temp.csv
│   └── raw/
│       └── test.csv
│       └── train.csv
├── notebooks/
│   └── clean_datasets.ipynb
│   └── feature_selection.ipynb
│   └── ML.ipynb
└── src/
    ├── data/
    │   ├── clean_data.py
    │   ├── fill_last_payment.py
    │   └── load_data.py
    ├── models/
    │   ├── train_model.py
    │   ├── random_forest.py
    │   └── knn_model.py
    ├── utils/
    │   └── find_k.py
    │   └── one_hot.py
    │   └── standardize.py
    └── visualization/
        ├── important_features.py
        └── performance.py
```

`main.py` imports the modules inside `src/` and executes them to reproduce the analysis and results. Jupyter notebooks contain prototyping used for the Kaggle leaderboards.

## Running the code

Install the dependencies and run the pipeline. You should use the versions of the dependencies as specified by the requirements file:

```bash
conda create -n churn --file requirements.txt
conda activate churn
python3 main.py
```

This will load the dataset, display the best features to use, train KNN and Random Forest models using those features, and produce visualizations to test their performace.
All plots will be displayed interactively.
The csv files in data/processed are created from the notebook, but they are the exact same data used in main.py
