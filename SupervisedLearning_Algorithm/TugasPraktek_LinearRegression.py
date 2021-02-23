# Tugas Praktek

# dataset = https://storage.googleapis.com/dqlab-dataset/pythonTutorial/housing_boston.csv

# Detail Instruksi
# 1. Pisahkan dataset ke dalam Feature dan Label, gunakan fungsi .drop().
# Pada dataset ini, label/target adalah variabel MEDV.
# 2. Checking dan print jumlah data setelah Dataset pisahkan ke dalam Feature dan Label, gunakan .shape().
# 3. Bagi dataset ke dalam Training dan test dataset, 70% data digunakan untuk training dan 30% untuk testing,
# gunakan fungsi train_test_split() , dengan random_state = 0.
# 4. Checking dan print kembali jumlah data dengan fungsi .shape().
# 5. Import LinearRegression dari sklearn.linear_model.
# 6. Deklarasikan  LinearRegression regressor dengan nama reg.
# 7. Fit regressor ke training dataset dengan .fit(), dan gunakan .predict() untuk memprediksi nilai dari
# testing dataset.

# load dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
housing = pd.read_csv(
    'housing_boston.csv')
# Data rescaling
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
housing[['RM', 'LSTAT', 'PTRATIO', 'MEDV']] = data_scaler.fit_transform(
    housing[['RM', 'LSTAT', 'PTRATIO', 'MEDV']])
# getting dependent and independent variables
X = housing.drop(['MEDV'], axis=1)
y = housing['MEDV']
# checking the shapes
print('Shape of X:', X.shape)
print('Shape of y:', y.shape)

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
# checking the shapes
print('Shape of X_train :', X_train.shape)
print('Shape of y_train :', y_train.shape)
print('Shape of X_test :', X_test.shape)
print('Shape of y_test :', y_test.shape)

# import regressor from Scikit-Learn
# Call the regressor
reg = LinearRegression()
# Fit the regressor to the training data
reg = reg.fit(X_train, y_train)
# Apply the regressor/model to the test data
y_pred = reg.predict(X_test)
