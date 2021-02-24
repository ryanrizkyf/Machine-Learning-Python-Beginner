# Baru dapat kiriman data dari e-commerce kita.
# Divisi e-commerce kita ingin memprediksi apakah user- user yang sedang mengunjungi halaman website
# yang baru akan mengklik banner promo (ads) di halaman tersebut atau tidak berdasarkan feature yang ada.
# Tolong buatkan machine learning model untuk menyelesaikan permasalahan dari e-commerce kita ini ya.

# Kita akan membuat machine learning model untuk menyelesaikan permasalahan dari e-commerce divisi kantor.

# Adapun feature - feature dalam dataset ini adalah :
# 'Daily Time Spent on Site' : lama waktu user mengunjungi site (menit)
# 'Age' : usia user (tahun)
# 'Area Income' : rata - rata pendapatan di daerah sekitar user
# 'Daily Internet Usage' : rata - rata waktu yang dihabiskan user di internet dalam sehari (menit)
# 'Ad Topic Line' : topik/konten dari promo banner
# 'City' : kota dimana user mengakses website
# 'Male' : apakah user adalah Pria atau bukan
# 'Country' : negara dimana user mengakses website
# 'Timestamp' : waktu saat user mengklik promo banner atau keluar dari halaman website tanpa mengklik banner
# 'Clicked on Ad' : mengindikasikan user mengklik promo banner atau tidak (0 = tidak; 1 = klik).

# Di proyek ini, diharapkan untuk membuat machine learning model sesuai dengan prosedur machine learning
# yang sudah disharing sebelumnya. Jadi, tahap-tahap yang perlu dilakukan adalah (langkah ke-1) terlebih dahulu

# Import library
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Baca data 'ecommerce_banner_promo.csv'
data = pd.read_csv(
    'ecommerce_banner_promo.csv')

# Langkah 1
# 1.Data eksplorasi dengan head(), info(), describe(), shape

# 1. Data eksplorasi dengan head(), info(), describe(), shape
print("\n[1] Data eksplorasi dengan head(), info(), describe(), shape")
print("Lima data teratas:")
print(data.head())
print("Informasi dataset:")
print(data.info())
print("Statistik deskriptif dataset:")
print(data.describe())
print("Ukuran dataset:")
print(data.shape)

# Sekarang mari melanjutkan dengan ekplorasi data untuk langkah ke-2 dan ke-3:
# 2. Data eksplorasi dengan dengan mengecek korelasi dari setiap feature menggunakan fungsi corr()
# 3. Data eksplorasi dengan mengecek distribusi label menggunakan fungsi groupby() dan size()

# 2. Data eksplorasi dengan dengan mengecek korelasi dari setiap feature menggunakan fungsi corr()
print("\n[2] Data eksplorasi dengan dengan mengecek korelasi dari setiap feature menggunakan fungsi corr()")
print(data.corr())

# 3. Data eksplorasi dengan mengecek distribusi label menggunakan fungsi groupby() dan size()
print("\n[3] Data eksplorasi dengan mengecek distribusi label menggunakan fungsi groupby() dan size()")
# print(data.groupby('Clicked on Ad').size())

# Di proyek ini, aku akan melanjutkan mengeksplorasi data dengan visualisasi
# dengan tahap - tahap yang perlu dilakukan adalah (langkah ke-4):
# 4. Data eksplorasi dengan visualisasi:
# Jumlah user dibagi ke dalam rentang usia menggunakan histogram (hist()),
# gunakan bins = data.Age.nunique() sebagai argumen. nunique() adalah fungsi untuk menghitung jumlah data
# untuk setiap usia (Age).
# Gunakan pairplot() dari seaborn modul untuk menggambarkan hubungan setiap feature.

# Seting: matplotlib and seaborn
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

# 4. Data eksplorasi dengan visualisasi
# 4a. Visualisasi Jumlah user dibagi ke dalam rentang usia (Age) menggunakan histogram (hist()) plot
plt.figure(figsize=(10, 5))
plt.hist(data['Age'], bins=data.Age.nunique())
plt.xlabel('Age')
plt.tight_layout()
plt.show()

# 4b. Gunakan pairplot() dari seaborn (sns) modul untuk menggambarkan hubungan setiap feature.
plt.figure()
sns.pairplot(data)
plt.show()

# Di bagian proyek (langkah ke-5) ini aku akan mengecek apakah terdapat missing value dari data,
# jika terdapat missing value dapat dilakukan treatment seperti didrop atau diimputasi
# dan jika tidak maka dapat melanjutkan ke langkah berikutnya.
# Cek missing value

# 5. Cek missing value
print("\n[5] Cek missing value")
print(data.isnull().sum().sum())

# Pada langkah ke-6 ini aku akan melakukan pemodelan dengan Logistic Regression dengan cara seperti berikut:
# 1. Lakukan pemodelan dengan Logistic Regression, gunakan perbandingan 80:20 untuk training vs testing :
# 2. Deklarasikan data ke dalam X dengan mendrop feature/variabel yang bukan numerik, (type = object)
# dari data (Logistic Regression hanya dapat memproses numerik variabel).
# Assign Target/Label feature dan assign sebagai y
# 3. Split X dan y ke dalam training dan testing dataset, gunakan perbandingan 80:20 dan random_state = 42
# 4. Assign classifier sebagai logreg, kemudian fit classifier ke X_train dan predict dengan X_test.
# Print evaluation score.

# 6.Lakukan pemodelan dengan Logistic Regression, gunakan perbandingan 80:20 untuk training vs testing
print("\n[6] Lakukan pemodelan dengan Logistic Regression, gunakan perbandingan 80:20 untuk training vs testing")
# 6a.Drop Non-Numerical (object type) feature from X, as Logistic Regression can only take numbers, and also drop Target/label, assign Target Variable to y.
X = data.drop(['Ad Topic Line', 'City', 'Country',
               'Timestamp', 'Clicked on Ad'], axis=1)
y = data['Clicked on Ad']

# 6b. splitting the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

# 6c. Modelling
# Call the classifier
logreg = LogisticRegression()
# Fit the classifier to the training data
logreg = logreg.fit(X_train, y_train)
# Prediksi model
y_pred = logreg.predict(X_test)

# 6d. Evaluasi Model Performance
print("Evaluasi Model Performance:")
print("Training Accuracy :", logreg.score(X_train, y_train))
print("Testing Accuracy :", logreg.score(X_test, y_test))

# Di langkah terakhir ini atau langkah ke-7 aku akan melihat performansi model
# dengan menggunakan confusion matrix dan classification report.
# Print Confusion matrix dan classification report

# 7. Print Confusion matrix dan classification report
print("\n[7] Print Confusion matrix dan classification report")

# apply confusion_matrix function to y_test and y_pred
print("Confusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# apply classification_report function to y_test and y_pred
print("Classification report:")
cr = classification_report(y_test, y_pred)
print(cr)
