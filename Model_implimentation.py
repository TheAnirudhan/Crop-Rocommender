import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

df = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/crop_recommendation/train_set_label.csv")
X = df.drop('crop',axis=1)
y = df.crop
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print("Checkpoint:1")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y = df.crop
print("Checkpoint:2")

knn = KNeighborsClassifier(5)
knn.fit(X_train,y_train)
print("Checkpoint:3a")
pred = knn.predict(X_test)
print("Checkpoint:3b")
print("Done")

filename = 'crop-recommendation-knn-model.pkl'
pickle.dump(knn, open(filename, 'wb'))
