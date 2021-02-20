# Praktekkan metode imputing missing value dengan menggunakan nilai median.
# dataset = https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/online_raw.csv

import pandas as pd
dataset1 = pd.read_csv(
    'online_raw.csv')

print("Before imputation:")
# Checking missing value for each feature
print(dataset1.isnull().sum())
# Counting total missing value
print(dataset1.isnull().sum().sum())

print("\nAfter imputation:")
# Fill missing value with median of feature value
dataset1.fillna(dataset1.median(), inplace=True)
# Checking missing value for each feature
print(dataset1.isnull().sum())
# Counting total missing value
print(dataset1.isnull().sum().sum())
