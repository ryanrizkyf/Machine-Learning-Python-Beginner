# Nah, sebelum kita melatih model dengan suatu algorithm machine , seperti yang saya jelaskan sebelumnya.
# Dataset perlu kita bagi ke dalam training dataset dan test dataset dengan perbandingan 80:20.
# 80% digunakan untuk training dan 20% untuk proses testing.

# Perbandingan lain yang biasanya digunakan adalah 75:25.
# Hal penting yang perlu diketahui adalah scikit-learn tidak dapat memproses dataframe dan
# hanya mengakomodasi format data tipe Array. Tetapi kalian tidak perlu khawatir,
# fungsi train_test_split( ) dari Scikit-Learn, otomatis mengubah dataset dari dataframe ke dalam format array.

# Kenapa perlu ada Training dan Testing?
# Fungsi Training adalah melatih model untuk mengenali pola dalam data,
# sedangkan testing berfungsi untuk memastikan bahwa model yang telah dilatih tersebut mampu
# dengan baik memprediksi label dari new observation dan belum dipelajari oleh model sebelumnya.

# dataset = https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/online_raw.csv

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
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

# Pengecekan missing value dapat dilakukan dengan menggunakan metod .isnull pada dataset
# dan kemudian men-chaining-nya dengan method sum. Untuk jumlah keseluruhan missing value
# digunakan chaining method sum sekali lagi.
# checking missing value for each feature
print('Checking missing value for each feature:')
print(dataset.isnull().sum())
# Counting total missing value
print('\nCounting total missing value:')
print(dataset.isnull().sum().sum())

# Kita bisa menghapus data point yang memiliki missing value dengan fungsi .dropna( ) dari pandas library.
# Fungsi dropna( ) akan menghapus data point atau baris yang memiliki missing value.
# Drop rows with missing value
dataset_clean = dataset.dropna()
print('Ukuran dataset_clean:', dataset_clean.shape)

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

# Define MinMaxScaler as scaler
scaler = MinMaxScaler()
# list all the feature that need to be scaled
scaling_column = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                  'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
# Apply fit_transfrom to scale selected feature
dataset[scaling_column] = scaler.fit_transform(dataset[scaling_column])
# Cheking min and max value of the scaling_column
print(dataset[scaling_column].describe().T[['min', 'max']])

# Cara mengubah tipe pandas object ini ke numerik (int, float).
# LabelEncoder akan mengurutkan label secara otomatis secara alfabetik,
# posisi/indeks dari setiap label ini digunakan sebagai nilai numeris konversi pandas objek ke numeris
# (dalam hal ini tipe data int). Dengan demikian kita telah membuat dataset kita menjadi dataset bernilai
# numeris seluruhnya yang siap digunakan untuk pemodelan dengan algoritma machine learning tertentu
# Convert feature/column 'Month'
LE = LabelEncoder()
dataset['Month'] = LE.fit_transform(dataset['Month'])
print(LE.classes_)
print(np.sort(dataset['Month'].unique()))
print('')

# Convert feature/column 'VisitorType'
LE = LabelEncoder()
dataset['VisitorType'] = LE.fit_transform(dataset['VisitorType'])
print(LE.classes_)
print(np.sort(dataset['VisitorType'].unique()))

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

# Gunakan test_size = 0.2 dan tambahkan argumen random_state = 0,  pada fungsi train_test_split( ).
# splitting the X, and y
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
# checking the shapes
print("Shape of X_train :", X_train.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of X_test :", X_test.shape)
print("Shape of y_test :", y_test.shape)
