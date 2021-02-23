# Pemodelan Logistic Regression dengan memanfaatkan Scikit-Learn sangatlah mudah.
# Dengan menggunakan dataset yang sama yaitu online_raw, dan setelah dataset dibagi ke dalam
# Training Set dan Test Set, cukup menggunakan modul linear_model dari Scikit-learn,
# dan memanggil fungsi LogisticRegression() yang diberi nama logreg.

# Kemudian, model yang sudah ditraining ini bisa digunakan untuk memprediksi output/label
# dari test dataset sekaligus mengevaluasi model performance dengan fungsi score(),
# confusion_matrix() dan classification_report().

# dataset = https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/online_raw.csv

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
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

# Kita akan menggunakan Decision Tree.
# Kita hanya perlu memanggil fungsi DecisionTreeClassifier() yang kita namakan “model”.
# Kemudian menggunakan fungsi .fit() dan X_train, y_train untuk
# melatih classifier tersebut dengan training dataset.
# Call the classifier
model = DecisionTreeClassifier()
# Fit the classifier to the training data
model = model.fit(X_train, y_train)

# Untuk memprediksi LABEL dari testing dataset (X_test), menggunakan fungsi .predict().
# Fungsi ini akan mengembalikan hasil prediksi untuk setiap data point dari X_test dalam bentuk array.
# Apply the classifier/model to the test data
y_pred = model.predict(X_test)
print(y_pred.shape)

# Evaluasi Model Performance
# evaluating the model
print('Training Accuracy :', model.score(X_train, y_train))
print('Testing Accuracy :', model.score(X_test, y_test))

# Untuk menampilkan confusion matrix cukup menggunakan fungsi confusion_matrix() dari Scikit-Learn
# confusion matrix
print('\nConfusion matrix:')
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Berdasarkan confusion matrix, dapat mengukur metrik - metrik berikut :
# 1. Accuracy = (TP + TN ) / (TP+FP+FN+TN)
# 2. Precision = (TP) / (TP+FP)
# 3. Recall = (TP) / (TP + FN)
# 4. F1 Score = 2 * (Recall*Precission) / (Recall + Precission)

# Tidak perlu menghitung nilai ini secara manual.
# Cukup gunakan fungsi classification_report() untuk memunculkan hasil perhitungan metrik - metrik tersebut.
# classification report
print('\nClassification report:')
cr = classification_report(y_test, y_pred)
print(cr)

# Setelah dataset dibagi ke dalam Training Set dan Test Set,
# cukup menggunakan modul linear_model dari Scikit-learn,
# dan memanggil fungsi LogisticRegression() yang diberi nama logreg.
# Kemudian, model yang sudah ditraining ini bisa digunakan untuk memprediksi output/label
# dari test dataset sekaligus mengevaluasi model performance dengan fungsi score(),
# confusion_matrix() dan classification_report().
# Call the classifier
logreg = LogisticRegression()
# Fit the classifier to the training data
logreg = logreg.fit(X_train, y_train)
# Training Model: Predict
y_pred = logreg.predict(X_test)

# Evaluate Model Performance
print('Training Accuracy :', model.score(X_train, y_train))
print('Testing Accuracy :', model.score(X_test, y_test))

# confusion matrix
print('\nConfusion matrix')
cm = confusion_matrix(y_test, y_pred)
print(cm)

# classification report
print('\nClassification report')
cr = classification_report(y_test, y_pred)
print(cr)
