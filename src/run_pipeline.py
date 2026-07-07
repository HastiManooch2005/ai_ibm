from data_loader import *
from preprocessing import preprocessing_pipeline
from features import feature_pipeline
from train import train_models
from evaluate import evaluate_model
from mlflow_utils import log_to_mlflow


SEED = 42

DATASETS = {
    "v2": "data/v2/customer_churn_clean.csv",
    "v3": "data/v3/customer_churn_features.csv",
}


def main():

    # =====================
    # v1 -> v2
    # =====================
    preprocessing_pipeline()

    # =====================
    # v2 -> v3
    # =====================
    feature_pipeline()

    # =====================
    # Train each version
    # =====================
    for version, file_path in DATASETS.items():

       print("\n")
       print("=" * 70)
       print(f"Running Pipeline Dataset {version}")
       print("=" * 70)

       df = load_data(file_path)

       result = train_models(
    df,
    version
)

       metrics = evaluate_model(
       model=result["model"],
      X_test=result["X_test"],
      y_test=result["y_test"]
       )

    print(metrics)

    log_to_mlflow(
     model=result["model"],
     model_name=result["model_name"],
     dataset_version=result["dataset_version"],
     metrics=metrics,
     seed=SEED
 )

if __name__ == "__main__":
    main()