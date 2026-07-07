from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


INPUT_PATH = Path("data/v2/customer_churn_clean.csv")
OUTPUT_PATH = Path("data/v3/customer_churn_features.csv")


def load_clean_data():
    """Load cleaned dataset (v2)."""
    return pd.read_csv(INPUT_PATH)


def create_features(df):
    """
    Create new features.
    """

    # جلوگیری از تقسیم بر صفر
    tenure = df["Tenure Months"].replace(0, np.nan)

    df["Average Monthly Spending"] = (
        df["Total Charges"] / tenure
    )

    df["Average Monthly Spending"] = (
        df["Average Monthly Spending"]
        .fillna(0)
    )

    # مشتری جدید
    df["Is_New_Customer"] = (
        df["Tenure Months"] <= 12
    ).astype(int)

    # مشتری بلند مدت
    df["Is_Long_Term"] = (
        df["Tenure Months"] >= 24
    ).astype(int)

    # هزینه ماهانه بالا
    median_charge = df["Monthly Charges"].median()

    df["High_Monthly_Charges"] = (
        df["Monthly Charges"] > median_charge
    ).astype(int)

    return df


def normalize_numeric_columns(df):
    """
    Normalize numerical columns.
    """

    scaler = MinMaxScaler()

    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # هدف مدل نباید نرمال شود
    if "Churn Value" in numeric_columns:
        numeric_columns.remove("Churn Value")

    df[numeric_columns] = scaler.fit_transform(
        df[numeric_columns]
    )

    return df


def save_dataset(df):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )


def feature_pipeline():

    print("=" * 60)
    print("Loading Clean Dataset")
    print("=" * 60)

    df = load_clean_data()

    print(df.shape)

    print()

    print("=" * 60)
    print("Creating New Features")
    print("=" * 60)

    df = create_features(df)

    print(df.shape)

    print()

    print("=" * 60)
    print("Normalizing Numeric Features")
    print("=" * 60)

    df = normalize_numeric_columns(df)

    print()

    print("=" * 60)
    print("Saving Version 3")
    print("=" * 60)

    save_dataset(df)

    print("Done!")

    return df


if __name__ == "__main__":

    feature_pipeline()