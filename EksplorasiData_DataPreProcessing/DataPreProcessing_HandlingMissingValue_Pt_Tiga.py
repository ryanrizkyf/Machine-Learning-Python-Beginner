# Kalau tidak dihapus, ada metode lain yang bisa dipakai?
# Kita bisa menggunakan metode impute missing value, yaitu mengisi record yang hilang ini dengan suatu nilai.
# Ada berbagai teknik dalam metode imputing, mulai dari yang paling sederhana yaitu mengisi missing value
# dengan nilai mean, median, modus, atau nilai konstan, sampai teknik paling advance yaitu dengan menggunakan
# nilai yang diestimasi oleh suatu predictive model. Untuk kasus ini, kita akan menggunakan imputing sederhana
# yaitu menggunakan nilai rataan atau mean.

# Dataset ini adalah data pembeli online yang mengunjungi website dari suatu e-commerce selama setahun,
# yaitu 'https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/online_raw.csv'

import pandas as pd
dataset = pd.read_csv(
    'online_raw.csv')

# Imputing missing value sangat mudah dilakukan di Python,
# cukup memanfaatkan fungsi .fillna() dan .mean() dari Pandas

print("Before imputation:")
# Checking missing value for each feature
print(dataset.isnull().sum())
# Counting total missing value
print(dataset.isnull().sum().sum())

print("\nAfter imputation:")
# Fill missing value with mean of feature value
dataset.fillna(dataset.mean(), inplace=True)
# Checking missing value for each feature
print(dataset.isnull().sum())
# Counting total missing value
print(dataset.isnull().sum().sum())
