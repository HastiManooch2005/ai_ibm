from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(
        model,
        X_test,
        y_test
):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]


    metrics = {

        "accuracy":
            accuracy_score(
                y_test,
                predictions
            ),


        "precision":
            precision_score(
                y_test,
                predictions
            ),


        "recall":
            recall_score(
                y_test,
                predictions
            ),


        "f1_score":
            f1_score(
                y_test,
                predictions
            ),


        "roc_auc":
            roc_auc_score(
                y_test,
                probabilities
            ),


        "confusion_matrix":
            confusion_matrix(
                y_test,
                predictions
            ).tolist()
    }


    return metrics