from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

SEED = 42


def split_data(df):

    X = df.drop("Churn Value", axis=1)
    y = df["Churn Value"]

    # ---------- Test ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED
    )

    # ---------- Validation ----------
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=SEED
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

    return {

        "LogisticRegression": LogisticRegression(
            max_iter=5000,
            random_state=SEED
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=SEED
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=SEED,
            eval_metric="logloss"
        ),

        "CatBoost": CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            random_seed=SEED,
            verbose=0
        )

    }


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

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    best_model = None
    best_model_name = None
    best_cv_score = 0
    best_val_score = 0

    print("\n")
    print("=" * 70)
    print(f"Dataset : {dataset_version}")
    print("=" * 70)

    for name, model in models.items():

        print("\n")
        print("=" * 50)
        print(f"Training {name}")
        print("=" * 50)

        # Cross Validation
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc"
        )

        # Train
        model.fit(
            X_train,
            y_train
        )

        # Validation
        val_score = model.score(
            X_val,
            y_val
        )

        print(f"CV ROC-AUC : {cv_scores.mean():.4f}")
        print(f"Validation Accuracy : {val_score:.4f}")

        if val_score > best_val_score:

            best_val_score = val_score
            best_cv_score = cv_scores.mean()

            best_model = model
            best_model_name = name

    print("\n")
    print("=" * 70)
    print("Best Model Selected")
    print("=" * 70)
    print(best_model_name)
    print(f"Validation Accuracy : {best_val_score:.4f}")

    return {

        "model": best_model,

        "model_name": best_model_name,

        "dataset_version": dataset_version,

        "cv_score": best_cv_score,

        "X_test": X_test,

        "y_test": y_test

    }