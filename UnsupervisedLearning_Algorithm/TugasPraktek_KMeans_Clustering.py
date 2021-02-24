# Tugas Praktek

# dataset = https://storage.googleapis.com/dqlab-dataset/pythonTutorial/mall_customers.csv

# Dataset ini merupakan data customer suatu mall dan berisi basic informasi customer berupa :
# CustomerID, age, gender, annual income, dan spending score.
# Adapun tujuan dari clustering adalah untuk memahami customer - customer mana saja yang sering
# melakukan transaksi sehingga informasi ini dapat diberikan kepada marketing team
# untuk membuat strategi promosi yang sesuai dengan karakteristik customer.

# Instruksi
# 1. Import pandas sebagai aliasnya dan KMeans dari sklearn.cluster.
# 2. Load dataset 'https://storage.googleapis.com/dqlab-dataset/pythonTutorial/mall_customers.csv'
# dan beri nama dataset
# 3. Diasumsikan EDA dan preprocessing sudah dilakukan, selanjutnya kita memilih feature
# yang akan digunakan untuk membuat model yaitu annual_income dan spending_score.
# Assign dataset dengan feature yang sudah dipilih ke dalam 'X'.
# Pada dasarnya terdapat teknik khusus yang dilakukan untuk menyeleksi feature - feature (Feature Selection)
# mana saja yang dapat digunakan untuk machine learning modelling, karena tidak semua feature itu berguna.
# Beberapa feature justru bisa menyebabkan performansi model menurun. Tetapi untuk problem ini,
# secara default kita akan menggunakan annual_income dan spending_score.
# 4. Deklarasikan  KMeans( ) dengan nama cluster_model dan gunakan n_cluster = 5. n_cluster adalah
# argumen dari fungsi KMeans( ) yang merupakan jumlah cluster/centroid (K). random_state = 24.
# 5. Gunakan fungsi .fit_predict( ) dari cluster_model pada 'X' untuk proses clustering.

# Import library
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# load dataset
dataset = pd.read_csv(
    'mall_customers.csv')

# selecting features
X = dataset[['annual_income', 'spending_score']]

# Define KMeans as cluster_model
cluster_model = KMeans(n_clusters=5, random_state=24)
labels = cluster_model.fit_predict(X)

# Kalau sudah membuat cluster,
# tolong visualisasikan hasil dari clustering yang telah kita lakukan sebelumnya ya.

# Instruksi
# 1. Pertama - tama, import matplotlib.pyplot dan beri inisial plt.
# 2. Gunakan fungsi .values untuk mengubah tipe ‘X’ dari dataframe menjadi array.
# 3. Pisahkan X kedalam xs dan ys, di mana xs adalah Kolom index [0] dan ys adalah kolom index [1]
# 4. Buatlah scatter plot plt.scatter() dari xs dan ys, kemudian tambahkan c = labels untuk
# secara otomatis memberikan warna yang berbeda pada setiap cluster, dan alpha = 0.5
# ke dalam scatter plot argumen.
# 5. Hitunglah koordinat dari centroid menggunakan .cluster_centers_ dari cluster_model,
# deklarasikan ke dalam variabel centroids.
# 6. Pisahkan centroids kedalam centroids_x dan centroids_y, di mana centroids_x adalah kolom index [0]
# dan centroids_y adalah kolom index [1]
# 7. Buatlah scatter plot dari centroids_x dan centroids_y , gunakan ‘D’ (diamond) sebagai marker parameter,
# dengan ukuran 50, s = 50

# convert dataframe to array
X = X.values
# Separate X to xs and ys --> use for chart axis
xs = X[:, 0]
ys = X[:, 1]
# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = cluster_model.cluster_centers_
# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]
# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.title('K Means Clustering', fontsize=20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
