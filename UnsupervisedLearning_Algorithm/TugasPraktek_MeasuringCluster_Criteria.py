# Tugas Praktek
# Coba kita membuat inertia plot untuk melihat apakah K = 5 merupakan jumlah cluster yang optimal.

# dataset = https://storage.googleapis.com/dqlab-dataset/pythonTutorial/mall_customers.csv

# Instruksi
# Untuk membuat inertia plot, silakan memanfaatkan fungsi looping (for):
# 1. Pertama - tama, buatlah sebuah list kosong yang dinamakan 'inertia'.
# List ini akan kita gunakan untuk menyimpan nilai inertia dari setiap nilai K.
# 2. Gunakan for untuk membuat looping dengan range 1-10. Sebagai index looping gunakan k
# 3. Di dalam fungsi looping, deklarasikan  KMeans()  dengan nama cluster_model dan gunakan n_cluster = k,
# dan random_state = 24
# 4. Gunakan fungsi .fit() dari cluster_model pada 'X'
# 5. Dari cluster_model yang sudah di-fit ke dataset, dapatkan nilai inertia menggunakan inertia_
# dan deklarasikan sebagai inertia_value
# 6. Append inertia_value ke dalam list 'inertia'
# 7. Setelah iterasi/looping selesai plotlah list 'inertia' tadi sebagai ordinat-nya dan absica-nya
# adalah range(1, 10).

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


# Elbow Method - Inertia plot
inertia = []
# looping the inertia calculation for each k
for k in range(1, 10):
    # Assign KMeans as cluster_model
    cluster_model = KMeans(n_clusters=k, random_state=24)
    # Fit cluster_model to X
    cluster_model.fit(X)
    # Get the inertia value
    inertia_value = cluster_model.inertia_
    # Append the inertia_value to inertia list
    inertia.append(inertia_value)

# Inertia plot
plt.plot(range(1, 10), inertia)
plt.title('The Elbow Method - Inertia plot', fontsize=20)
plt.xlabel('No. of Clusters')
plt.ylabel('Inertia')
plt.show()
