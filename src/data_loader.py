from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/v1/Telco_customer_churn.xlsx")


def load_data():
    """
    Read raw dataset (Version 1).

    Returns
    -------
    pandas.DataFrame
    """

    df = pd.read_excel(DATA_PATH)

    return df


def show_dataset_info(df):
    """Display basic information about dataset."""

    print("=" * 60)
    print("First 5 Rows")
    print("=" * 60)
    print(df.head())

    print()

    print("=" * 60)
    print("Dataset Shape")
    print("=" * 60)
    print(df.shape)

    print()

    print("=" * 60)
    print("Columns")
    print("=" * 60)
    print(df.columns.tolist())

    print()

    print("=" * 60)
    print("Dataset Information")
    print("=" * 60)
    df.info()

    print()

    print("=" * 60)
    print("Missing Values")
    print("=" * 60)
    print(df.isnull().sum())


if __name__ == "__main__":
    dataframe = load_data()
    show_dataset_info(dataframe)