# Sekarang saatnya kita melatih model atau training.
# Dengan Scikit-Learn, proses ini menjadi sangat sederhana.
# Kita cukup memanggil nama algorithm yang akan kita gunakan,
# biasanya disebut classifier untuk problem klasifikasi, dan regressor untuk problem regresi.

# dataset = https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/online_raw.csv

from sklearn.tree import DecisionTreeClassifier
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

# Kita akan menggunakan Decision Tree.
# Kita hanya perlu memanggil fungsi DecisionTreeClassifier() yang kita namakan “model”.
# Kemudian menggunakan fungsi .fit() dan X_train, y_train untuk
# melatih classifier tersebut dengan training dataset.
# Call the classifier
model = DecisionTreeClassifier()
# Fit the classifier to the training data
model = model.fit(X_train, y_train)
