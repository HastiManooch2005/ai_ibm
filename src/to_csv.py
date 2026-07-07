from pathlib import Path

import pandas as pd

excel_path = Path("data/v1/Telco_customer_churn.xlsx")
csv_path = Path("data/v1/Telco_customer_churn.csv")

df = pd.read_excel(excel_path)

df.to_csv(csv_path, index=False, encoding="utf-8")

print("Excel file converted to CSV successfully.")