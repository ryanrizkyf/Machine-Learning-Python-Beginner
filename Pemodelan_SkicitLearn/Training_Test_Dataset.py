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

from sklearn.model_selection import train_test_split
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

# Gunakan test_size = 0.2 dan tambahkan argumen random_state = 0,  pada fungsi train_test_split( ).
# splitting the X, and y
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
# checking the shapes
print("Shape of X_train :", X_train.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of X_test :", X_test.shape)
print("Shape of y_test :", y_test.shape)
