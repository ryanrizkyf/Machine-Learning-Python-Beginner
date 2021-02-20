# Setelah kita melakukan eksplorasi data, kita akan melanjutkan ke tahap data pre-processing.
# Seperti yang saya jelaskan sebelumnya, raw data kita belum tentu bisa langsung digunakan untuk pemodelan.
# Jika kita memiliki banyak missing value, maka akan mengurangi performansi model dan juga beberapa
# algorithm machine learning tidak dapat memproses data dengan missing value. Oleh karena itu,
# kita perlu mengecek apakah terdapat missing value dalam data atau tidak.
# Jika tidak, maka kita tidak perlu melakukan apa-apa dan bisa melanjutkan ke tahap berikutnya.
# Jika ada, maka kita perlu melakukan treatment khusus untuk missing value ini.

# Dataset ini adalah data pembeli online yang mengunjungi website dari suatu e-commerce selama setahun,
# yaitu 'https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/online_raw.csv'

import pandas as pd
dataset = pd.read_csv(
    'online_raw.csv')

# Pengecekan missing value dapat dilakukan dengan menggunakan metod .isnull pada dataset
# dan kemudian men-chaining-nya dengan method sum. Untuk jumlah keseluruhan missing value
# digunakan chaining method sum sekali lagi.

# checking missing value for each feature
print('Checking missing value for each feature:')
print(dataset.isnull().sum())
# Counting total missing value
print('\nCounting total missing value:')
print(dataset.isnull().sum().sum())
