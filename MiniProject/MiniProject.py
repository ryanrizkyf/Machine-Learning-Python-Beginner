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
print(data.groupby('Clicked on Ad').size())

# Di proyek ini, aku akan melanjutkan mengeksplorasi data dengan visualisasi
# dengan tahap - tahap yang perlu dilakukan adalah (langkah ke-4):
# 4. Data eksplorasi dengan visualisasi:
# Jumlah user dibagi ke dalam rentang usia menggunakan histogram (hist()),
# gunakan bins = data.Age.nunique() sebagai argumen. nunique() adalah fungsi untuk menghitung jumlah data
# untuk setiap usia (Age).
# Gunakan pairplot() dari seaborn modul untuk menggambarkan hubungan setiap feature.
