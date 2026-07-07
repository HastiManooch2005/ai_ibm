import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
import seaborn as sns

import os


def log_to_mlflow(
        model,
        model_name,
        dataset_version,
        metrics,
        seed
):

    with mlflow.start_run():



        mlflow.log_param(
            "model_name",
            model_name
        )


        mlflow.log_param(
            "dataset_version",
            dataset_version
        )


        mlflow.log_param(
            "seed",
            seed
        )


        # ======================
        # Hyperparameters
        # ======================

        params = model.get_params()


        mlflow.log_params(
            params
        )



        # ======================
        # Metrics
        # ======================

        for key, value in metrics.items():

            if key != "confusion_matrix":

                mlflow.log_metric(
                    key,
                    value
                )



        # ======================
        #Confusion Matrix
        # ======================

        cm = metrics["confusion_matrix"]


        plt.figure(
            figsize=(5,4)
        )


        sns.heatmap(
            cm,
            annot=True,
            fmt="d"
        )


        plt.xlabel(
            "Prediction"
        )

        plt.ylabel(
            "Actual"
        )


        plt.title(
            f"{model_name}-{dataset_version}"
        )


        file_name = "confusion_matrix.png"


        plt.savefig(
            file_name
        )


        plt.close()



        mlflow.log_artifact(
            file_name
        )



        # ======================
        # model save
        # ======================

        mlflow.sklearn.log_model(
            model,
            "model"
        )