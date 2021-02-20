# Tahapan-tahapan dalam pembuatan model machine learning.
# Membuat model machine learning tidak serta-merta langsung modelling,
# ada tahapan sebelumnya yang penting untuk dilakukan sehingga kita menghasilkan model yang baik.
# Untuk penjelasan ini, kita akan mempraktekkan langsung ya. Kita akan memanfaatkan Pandas library.
# Pandas cukup powerful untuk digunakan dalam menganalisa, memanipulasi dan membersihkan data.

# Pertama- tama,  kita check dimensi data kita terlebih dahulu.
# Silakan load datanya dan gunakan .shape, .head(), .info(), dan .describe()
# untuk mengeksplorasi dataset secara berurut.

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

# dengan mengetahui dimensi data yaitu jumlah baris dan kolom,
# kita bisa mengetahui apakah data kita terlalu banyak atau justru sangat sedikit.
# Jika data terlalu banyak, waktu melatih model akan lebih lama,
# sedangkan jika data terlalu sedikit, performansi model yang kita hasilkan mungkin tidak cukup bagus,
# karena tidak mampu mengenali pola dengan baik.
