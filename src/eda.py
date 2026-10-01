"""
eda.py
------
Exploratory data analysis on dataset/errors.csv.
Run: python src/eda.py
"""
import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "errors.csv")


def run_eda(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Number of records        : {len(df)}")
    print(f"Number of error categories: {df['error_type'].nunique()}")
    print(f"Missing values (any col) : {int(df.isnull().sum().sum())}")
    print(f"Duplicate rows           : {int(df.duplicated().sum())}")
    print(f"Duplicate error_messages : {int(df['error_message'].duplicated().sum())}")

    print("\n" + "=" * 50)
    print("CATEGORY DISTRIBUTION")
    print("=" * 50)
    dist = df["error_type"].value_counts().sort_index()
    print(dist.to_string())

    max_count, min_count = dist.max(), dist.min()
    ratio = max_count / min_count if min_count else float("inf")
    print(f"\nMax/Min category ratio: {ratio:.2f}")
    if ratio > 3:
        print("⚠️  Categories are notably imbalanced — consider adding more "
              "examples to the smaller categories.")
    else:
        print("✅ Category balance looks reasonable.")

    return df


if __name__ == "__main__":
    run_eda()
