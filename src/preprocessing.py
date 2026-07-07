from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


RAW_DATA_PATH = Path("data/v1/Telco_customer_churn.xlsx")
OUTPUT_PATH = Path("data/v2/customer_churn_clean.csv")


def load_raw_data():
    """
    Load raw dataset (Version 1)
    """

    return pd.read_excel(RAW_DATA_PATH)


def remove_unnecessary_columns(df):
    """
    Remove columns that are not useful for prediction.
    """

    columns_to_drop = [

        "CustomerID",
        "Count",
        "Country",
        "State",
        "Lat Long",
         "CLTV",
        "Churn Label",
        "Churn Score",
        "Churn Reason"

    ]

    existing_columns = [
        column
        for column in columns_to_drop
        if column in df.columns
    ]

    df = df.drop(columns=existing_columns)

    return df


def fix_data_types(df):
    """
    Convert columns to correct data types.
    """

    if "Total Charges" in df.columns:

        df["Total Charges"] = pd.to_numeric(
            df["Total Charges"],
            errors="coerce"
        )

    return df


def handle_missing_values(df):
    """
    Fill missing values.
    """

    numeric_columns = df.select_dtypes(include=["number"]).columns

    for column in numeric_columns:

        df[column] = df[column].fillna(df[column].median())

    categorical_columns = df.select_dtypes(include=["object"]).columns

    for column in categorical_columns:

        df[column] = df[column].fillna(df[column].mode()[0])

    return df


def encode_categorical_columns(df):
    """
    Label Encoding for categorical columns.
    """

    encoder = LabelEncoder()

    categorical_columns = df.select_dtypes(include=["object"]).columns

    for column in categorical_columns:

        df[column] = encoder.fit_transform(df[column])

    return df


def save_clean_dataset(df):
    """
    Save cleaned dataset.
    """

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )


def preprocessing_pipeline():

    print("=" * 60)
    print("Loading Raw Dataset")
    print("=" * 60)

    df = load_raw_data()

    print(df.shape)

    print()

    print("=" * 60)
    print("Removing unnecessary columns")
    print("=" * 60)

    df = remove_unnecessary_columns(df)

    print(df.shape)

    print()

    print("=" * 60)
    print("Fixing data types")
    print("=" * 60)

    df = fix_data_types(df)

    print()

    print("=" * 60)
    print("Handling Missing Values")
    print("=" * 60)

    df = handle_missing_values(df)

    print(df.isnull().sum().sum(), "Missing Values Remaining")

    print()

    print("=" * 60)
    print("Encoding Categorical Features")
    print("=" * 60)

    df = encode_categorical_columns(df)

    print()

    print("=" * 60)
    print("Saving Clean Dataset")
    print("=" * 60)

    save_clean_dataset(df)

    print("Saved Successfully!")

    return df


if __name__ == "__main__":

    preprocessing_pipeline()