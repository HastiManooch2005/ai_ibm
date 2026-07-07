import os

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.catboost

import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


EXPERIMENT_NAME = "Customer Churn Prediction"


def log_to_mlflow(
    model,
    model_name,
    dataset_version,
    metrics,
    seed,
):

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name=f"{model_name}_{dataset_version}"
    ):
    

        # ======================
        # Parameters
        # ======================

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("seed", seed)

        # ======================
        # Hyperparameters
        # ======================

        mlflow.log_params(model.get_params())

        # ======================
        # Metrics
        # ======================

        for key, value in metrics.items():

            if key != "confusion_matrix":
                mlflow.log_metric(key, value)

        # ======================
        # Confusion Matrix
        # ======================

        cm = metrics["confusion_matrix"]

        plt.figure(figsize=(5, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"{model_name} - {dataset_version}")

        plt.tight_layout()

        file_name = (
            f"{model_name}_{dataset_version}_confusion_matrix.png"
        )

        plt.savefig(file_name)
        plt.close()

        mlflow.log_artifact(file_name)
        os.remove(file_name)

        # ======================
        # Save Model
        # ======================

        if isinstance(model, XGBClassifier):

            mlflow.xgboost.log_model(
                xgb_model=model,
                name="model"
            )

        elif isinstance(model, CatBoostClassifier):

            mlflow.catboost.log_model(
                cb_model=model,
                name="model"
            )

        else:

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model"
            )