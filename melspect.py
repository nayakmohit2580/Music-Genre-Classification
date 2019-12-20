import numpy as np
import random
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

train_data = np.load('F:/ML/project/melspects/x_tr.npy')
train_label = np.load('F:/ML/project/melspects/y_tr.npy')
#train_data=np.delete(train_data, -1, 1)
test_data = np.load('F:/ML/project/melspects/x_te.npy')
test_label = np.load('F:/ML/project/melspects/y_te.npy')
#test_data=np.delete(test_data, -1, 1)
train_data = train_data.reshape(train_data.shape[0], train_data.shape[1]*train_data.shape[2])
test_data = test_data.reshape(test_data.shape[0], test_data.shape[1]*test_data.shape[2])
#Scaling
scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

#PCA
pca = PCA(n_components = 40, whiten = True)
pca.fit(train_data)
train_data = pca.transform(train_data)
test_data = pca.transform(test_data)


neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(train_data, train_label)
print(neigh.score(train_data,train_label))
print(neigh.score(test_data,test_label))