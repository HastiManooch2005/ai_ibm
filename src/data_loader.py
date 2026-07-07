from pathlib import Path

import pandas as pd


def load_data(file_path):
    """
    Read raw dataset.

    Parameters
    ----------
    file_path : str | Path

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_excel(file_path)


def show_dataset_info(df):
    """Display basic information about dataset."""

    print("=" * 60)
    print("First 5 Rows")
    print("=" * 60)
    print(df.head())

    print("\n" + "=" * 60)
    print("Dataset Shape")
    print("=" * 60)
    print(df.shape)

    print("\n" + "=" * 60)
    print("Columns")
    print("=" * 60)
    print(df.columns.tolist())

    print("\n" + "=" * 60)
    print("Dataset Information")
    print("=" * 60)
    df.info()

    print("\n" + "=" * 60)
    print("Missing Values")
    print("=" * 60)
    print(df.isnull().sum())


if __name__ == "__main__":
    file_path = Path(input("Enter dataset path: ").strip())

    dataframe = load_data(file_path)
    show_dataset_info(dataframe)