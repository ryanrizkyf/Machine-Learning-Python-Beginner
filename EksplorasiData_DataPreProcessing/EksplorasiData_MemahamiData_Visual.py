# Begini, selain dengan statistik, kita juga bisa melakukan eksplorasi data dalam bentuk visual.
# Dengan visualisasi kita dapat dengan mudah dan cepat dalam memahami data,
# bahkan dapat memberikan pemahaman yang lebih baik terkait hubungan setiap variabel/ features.

# Misalnya kita ingin melihat distribusi label dalam bentuk visual, dan jumlah pembelian saat weekend.
# Kita dapat memanfaatkan matplotlib library untuk membuat chart yang menampilkan perbandingan jumlah
# yang membeli (1) dan tidak membeli (0), serta perbandingan jumlah pembelian saat weekend

# Dataset ini adalah data pembeli online yang mengunjungi website dari suatu e-commerce selama setahun,
# yaitu 'https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/online_raw.csv'

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv(
    'online_raw.csv')

# Pertama- tama,  kita check dimensi data kita terlebih dahulu.
# Silakan load datanya dan gunakan .shape, .head(), .info(), dan .describe()
# untuk mengeksplorasi dataset secara berurut.
print('Shape dataset:', dataset.shape)
print('\nLima data teratas:\n', dataset.head())
print('\nInformasi dataset:')
print(dataset.info())
print('\nStatistik deskriptif:\n', dataset.describe())

# Lanjutkan eksplorasi data untuk melihat korelasi dan distribusi dataset.
dataset_corr = dataset.corr()
print('Korelasi dataset:\n', dataset.corr())
print('Distribusi Label (Revenue):\n', dataset['Revenue'].value_counts())

# Tugas praktek
print('\nKorelasi BounceRates-ExitRates:',
      dataset_corr.loc['BounceRates', 'ExitRates'])
print('\nKorelasi Revenue-PageValues:',
      dataset_corr.loc['Revenue', 'PageValues'])
print('\nKorelasi TrafficType-Weekend:',
      dataset_corr.loc['TrafficType', 'Weekend'])

# Eksplorasi data dalam bentuk visual
# checking the Distribution of customers on Revenue
plt.rcParams['figure.figsize'] = (12, 5)
plt.subplot(1, 2, 1)
sns.countplot(dataset['Revenue'], palette='pastel')
plt.title('Buy or Not', fontsize=20)
plt.xlabel('Revenue or not', fontsize=14)
plt.ylabel('count', fontsize=14)
# checking the Distribution of customers on Weekend
plt.subplot(1, 2, 2)
sns.countplot(dataset['Weekend'], palette='inferno')
plt.title('Purchase on Weekends', fontsize=20)
plt.xlabel('Weekend or not', fontsize=14)
plt.ylabel('count', fontsize=14)
plt.show()
