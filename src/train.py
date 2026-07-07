import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from evaluate import evaluate_model
SEED = 42



DATASETS = {
    "v2": "data/v2/customer_churn_clean.csv",
    "v3": "data/v3/customer_churn_features.csv"
}


def load_dataset(version):

    path = Path(DATASETS[version])

    df = pd.read_csv(path)

    return df

def split_data(df):

    X = df.drop(
        "Churn Value",
        axis=1
    )

    y = df["Churn Value"]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )


    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=SEED,
        stratify=y_train
    )


    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )



def get_models():


    models = {

        "LogisticRegression":
            LogisticRegression(
                max_iter=1000,
                random_state=SEED
            ),


        "RandomForest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=SEED
            ),


        "XGBoost":
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=SEED
            ),


        "CatBoost":
            CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                random_seed=SEED,
                verbose=0
            )

    }


    return models
from pathlib import Path
import pandas as pd

def train_models(df, dataset_version):

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test

    ) = split_data(df)



    models = get_models()

    results = {}

    best_model = None
    best_score = 0



    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )


    for name, model in models.items():


        print("="*50)
        print(f"Training {name}")


        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc"
        )


        # final training

        model.fit(
            X_train,
            y_train
        )


        # validation

        val_score = model.score(
            X_val,
            y_val
        )


        if val_score > best_score:

            best_score = val_score
            best_model = name



        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )


        results[name] = {

            "model": model,

            "model_name": name,

            "dataset_version": dataset_version,

            "cv_score": scores.mean(),

            "val_score": val_score,

            "metrics": metrics,

            "X_test": X_test,

            "y_test": y_test

        }


    print(
        "Best Model:",
        best_model
    )

    print(
        "Best Validation Score:",
        best_score
    )


    return results