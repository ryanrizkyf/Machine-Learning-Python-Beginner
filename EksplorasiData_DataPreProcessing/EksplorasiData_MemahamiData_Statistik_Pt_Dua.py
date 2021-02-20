# Data eksplorasi tidaklah cukup dengan mengetahui dimensi data dan statistical properties saja,
# tetapi kita juga perlu sedikit menggali tentang hubungan atau korelasi dari setiap feature,
# karena beberapa algorithm seperti linear regression dan logistic regression akan menghasilkan model
# dengan performansi yang buruk jika kita menggunakan feature/variabel saling dependensi atau
# berkorelasi kuat (multicollinearity). Jadi, jika kita sudah tahu bahwa data kita berkorelasi kuat,
# kita bisa menggunakan algorithm lain yang tidak sensitif terhadap hubungan korelasi dari feature/variabel
# seperti decision tree.

# Dataset ini adalah data pembeli online yang mengunjungi website dari suatu e-commerce selama setahun,
# yaitu 'https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/online_raw.csv'

import pandas as pd
dataset = pd.read_csv(
    'online_raw.csv')
print('Shape dataset:', dataset.shape)
print('\nLima data teratas:\n', dataset.head())
print('\nInformasi dataset:')
print(dataset.info())
print('\nStatistik deskriptif:\n', dataset.describe())

# Lanjutkan eksplorasi data untuk melihat korelasi dan distribusi dataset.
dataset_corr = dataset.corr()
print('Korelasi dataset:\n', dataset.corr())
print('Distribusi Label (Revenue):\n', dataset['Revenue'].value_counts())

# Kenapa mengetahui distribusi LABEL dari dataset itu penting?
# Mengetahui distribusi label sangat penting untuk permasalahan klasifikasi,
# karena jika distribusi label sangat tidak seimbang (imbalanced class),
# maka akan sulit bagi model untuk mempelajari pola dari LABEL yang sedikit dan hasilnya bisa misleading.
# Contohnya, kita memiliki 100 row data, 90 row adalah non fraud dan 10 row adalah fraud.
# Jika kita menggunakan data ini tanpa melakukan treatment khusus (handling imbalanced class),
# maka kemungkinan besar model kita akan cenderung mengenali observasi baru sebagai non-fraud,
# dan hal ini tentunya tidak diinginkan

# Tugas praktek
print('\nKorelasi BounceRates-ExitRates:',
      dataset_corr.loc['BounceRates', 'ExitRates'])
print('\nKorelasi Revenue-PageValues:',
      dataset_corr.loc['Revenue', 'PageValues'])
print('\nKorelasi TrafficType-Weekend:',
      dataset_corr.loc['TrafficType', 'Weekend'])
