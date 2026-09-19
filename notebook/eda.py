import pandas as pd
import numpy as np

# 1. Load the data 
train_df = pd.read_csv('data/train-test.csv')

# 2. Basic Information
print("=== Dataset Shape ===")
print(f"Rows: {train_df.shape[0]}, Columns: {train_df.shape[1]}\n")

print("=== Data Types and Missing Values ===")
train_info = pd.DataFrame({
    'Data Type': train_df.dtypes,
    'Missing Values': train_df.isnull().sum(),
    '% Missing': (train_df.isnull().sum() / len(train_df)) * 100
})
print(train_info.sort_values(by='% Missing', ascending=False))
print("\n" + "="*50 + "\n")

# 3. Quick look at the first few rows
print("=== First 5 Rows ===")
print(train_df.head())
print("\n" + "="*50 + "\n")

# 4. Summary statistics for numerical columns
print("=== Summary Statistics ===")
print(train_df.describe())