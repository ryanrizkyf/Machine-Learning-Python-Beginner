# Tugas Praktek

# dataset = https://storage.googleapis.com/dqlab-dataset/pythonTutorial/housing_boston.csv

# Detail Instruksi
# 1. Import library yang digunakan: mean_squared_error, mean_absolute_error dari sklearn.metrics
# dan numpy sebagai aliasnya yaitu np. Serta, import juga matplotlib.pyplot sebagai aliasnya, plt.
# 2. Hitung dan print nilai MSE dan RMSE dengan menggunakan argumen y_test dan y_pred,
# untuk rmse gunakan np.sqrt()
# 3. Buat scatter plot yang menggambarkan hasil prediksi (y_pred) dan harga actual (y_test)

# load dataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
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


# Calculating MSE, lower the value better it is. 0 means perfect prediction
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error of testing set:', mse)
# Calculating MAE
mae = mean_absolute_error(y_test, y_pred)
print('Mean absolute error of testing set:', mae)
# Calculating RMSE
rmse = np.sqrt(mse)
print('Root Mean Squared Error of testing set:', rmse)

# Plotting y_test dan y_pred
plt.scatter(y_test, y_pred, c='green')
plt.xlabel('Price Actual')
plt.ylabel('Predicted value')
plt.title('True value vs predicted value : Linear Regression')
plt.show()
