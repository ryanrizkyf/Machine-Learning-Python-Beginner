# Praktikkan code untuk data scaling

# Langkah - langkah untuk proses scaling dengan dataset yang memiliki feature dengan tipe data yang berbeda:
# Import MinMaxScaler dari sklearn.preprocessing
# Deklarasikan fungsi MinMaxScaler() ke dalam variabel scaler
# List semua feature yang akan di-scaling dan beri nama scaling_column yaitu :
# ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated',
# 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
# Berdasarkan contoh code yang dipraktekkan oleh Aksara, ganti dataset.columns dengan scaling_column.

# dataset = https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/online_raw.csv

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
dataset = pd.read_csv(
    'online_raw.csv')

# Define MinMaxScaler as scaler
scaler = MinMaxScaler()
# list all the feature that need to be scaled
scaling_column = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                  'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
# Apply fit_transfrom to scale selected feature
dataset[scaling_column] = scaler.fit_transform(dataset[scaling_column])
# Cheking min and max value of the scaling_column
print(dataset[scaling_column].describe().T[['min', 'max']])

# Jika dijalankan dengan kode program seperti berikut, hasilnya akan error
# dataset[dataset.columns] = scaler.fit_transform(dataset[dataset.columns])
# Karena kode program diatas merupakan basic code untuk proses scaling dengan asumsi bahwa semua feature
# adalah numerik. Tetapi, ketika menjalankan code tersebut untuk dataset online_raw, pasti akan terjadi error.
# Proses scaling hanya bisa dilakukan untuk feature dengan tipe numerik, sedangkan dalam dataset online_raw,
# terdapat feature dengan tipe string atau karakter dan categorical, seperti Month, VisitorType, Region.
# Oleh karena itu, kita tidak dapat langsung menggunakan code di atas, tetapi kita perlu terlebih dahulu
# menyeleksi feature - feature dari dataset yang bertipe numerik.
