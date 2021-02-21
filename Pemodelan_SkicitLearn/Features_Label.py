# Dalam dataset user online purchase, label target sudah diketahui,
# yaitu kolom Revenue yang bernilai 1 untuk user yang membeli dan 0 untuk yang tidak membeli,
# sehingga pemodelan yang dilakukan ini adalah klasifikasi.
# Nah, untuk melatih dataset menggunakan Scikit-Learn library,
# dataset perlu dipisahkan ke dalam Features dan Label/Target.


# dataset = https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/online_raw.csv

import pandas as pd
dataset = pd.read_csv(
    'online_raw.csv')

# Variabel Feature akan terdiri dari variabel yang dideklarasikan sebagai
# X dan [Revenue] adalah variabel Target yang dideklarasikan sebagai y.
# Gunakan fungsi drop() untuk menghapus kolom [Revenue] dari dataset.
# removing the target column Revenue from dataset and assigning to X
X = dataset.drop(['Revenue'], axis=1)
# assigning the target column Revenue to y
y = dataset['Revenue']
# checking the shapes
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
