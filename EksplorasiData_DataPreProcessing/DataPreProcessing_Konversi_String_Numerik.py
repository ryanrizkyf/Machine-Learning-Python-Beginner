# Kita memiliki dua kolom yang bertipe object yang dinyatakan dalam tipe data str,
# yaitu kolom 'Month' dan 'VisitorType'. Karena setiap algoritma machine learning bekerja
# dengan menggunakan nilai numeris, maka kita perlu mengubah kolom dengan tipe pandas object atau str
# ini ke bertipe numeris. Untuk itu, kita list terlebih dahulu apa saja label unik di kedua kolom ini.

# Label unik kolom 'Month':
# ['Feb' 'Mar' 'May' 'Oct' 'June' 'Jul' 'Aug' 'Nov' 'Sep' 'Dec']

# dan label unik kolom 'VisitorType':
# ['Returning_Visitor' 'New_Visitor' 'Other']

# dataset = https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/online_raw.csv

# Cara mengubah tipe pandas object ini ke numerik (int, float).
# Kita dapat menggunakan LabelEncoder dari sklearn.preprocessing untuk merubah kedua kolom ini.
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
dataset = pd.read_csv(
    'online_raw.csv')

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

# LabelEncoder akan mengurutkan label secara otomatis secara alfabetik,
# posisi/indeks dari setiap label ini digunakan sebagai nilai numeris konversi pandas objek ke numeris
# (dalam hal ini tipe data int). Dengan demikian kita telah membuat dataset kita menjadi dataset bernilai
# numeris seluruhnya yang siap digunakan untuk pemodelan dengan algoritma machine learning tertentu
