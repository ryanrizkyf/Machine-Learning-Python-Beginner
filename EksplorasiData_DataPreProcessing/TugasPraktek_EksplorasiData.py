# Membuat visualisasi berupa histogram yang menggambarkan jumlah customer untuk setiap Region.

# Dataset ini adalah data pembeli online yang mengunjungi website dari suatu e-commerce selama setahun,
# yaitu 'https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/online_raw.csv'

import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv(
    'online_raw.csv')

# Dalam membuat visualisasi ini aku akan menggunakan dataset['region'] untuk membuat histogram, dan
# berikan judul 'Distribution of Customers' pada title, 'Region Codes' sebagai label axis-x
# dan 'Count Users' sebagai label axis-y.

# visualizing the distribution of customers around the Region
plt.hist(dataset['Region'], color='lightblue')
plt.title('Distribution of Customers', fontsize=20)
plt.xlabel('Region Codes', fontsize=14)
plt.ylabel('Count Users', fontsize=14)
plt.show()
