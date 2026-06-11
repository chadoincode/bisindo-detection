# import pickle

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np


# data_dict = pickle.load(open('./data.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# model = RandomForestClassifier()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly !'.format(score * 100))

# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter

data_dict = pickle.load(open('./data.pickle', 'rb'))

X = np.asarray(data_dict['data'])
y = np.asarray(data_dict['labels'])

class_counts = Counter(y)
min_membership = min(class_counts.values())

# Jika ada kelas yang datanya kurang dari 2, matikan fitur stratify agar tidak eror
if min_membership < 2:
    print(f"⚠️ Peringatan: Ada kelas dengan sampel terlalu sedikit (< 2). Fitur stratify dinonaktifkan.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
else:
    # Berjalan normal dengan stratify jika semua kelas aman
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )

print(f"Jumlah data training: {len(X_train)}")
print(f"Jumlah data testing: {len(X_test)}")

print("Memulai pelatihan model Random Forest...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
score = accuracy_score(y_test, y_predict)
print(f"Pelatihan Selesai! Akurasi Model Akhir: {score * 100:.2f}%")

with open('model.p', 'wb') as f:
    pickle.dump({
        'model': model,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'all_labels': y
    }, f)

print("Model dan data pengujian sukses disimpan ke './model.p'.")