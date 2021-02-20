# Ternyata ada missing value di dataset kita. Apakah data point-nya bisa dihapus saja?
# Ada beberapa metode yang dapat kita lakukan untuk menangani missing value.
# Pilihan tepat, menghapus data adalah salah satunya. Tetapi, metode ini tidak dapat serta merta
# diimplementasikan. Kita juga perlu menganalisis penyebaran missing value, dan
# berapa persen jumlah missing value dalam data kita

# Metode ini dapat diterapkan jika tidak banyak missing value dalam data, sehingga walaupun
# data point ini dihapus, kita masih memiliki sejumlah data yang cukup untuk melatih model Machine Learning.
# Tetapi jika kita memiliki banyak missing value dan tersebar di setiap variabel, maka metode
# menghapus missing value tidak dapat digunakan. Kita akan kehilangan sejumlah data yang tentunya
# mempengaruhi performansi model. Kita bisa menghapus data point yang memiliki missing value
# dengan fungsi .dropna( ) dari pandas library. Fungsi dropna( ) akan menghapus data point atau
# baris yang memiliki missing value.

# Dataset ini adalah data pembeli online yang mengunjungi website dari suatu e-commerce selama setahun,
# yaitu 'https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/online_raw.csv'

import pandas as pd
dataset = pd.read_csv(
    'online_raw.csv')

# Kita bisa menghapus data point yang memiliki missing value dengan fungsi .dropna( ) dari pandas library.
# Fungsi dropna( ) akan menghapus data point atau baris yang memiliki missing value.

# Drop rows with missing value
dataset_clean = dataset.dropna()
print('Ukuran dataset_clean:', dataset_clean.shape)
