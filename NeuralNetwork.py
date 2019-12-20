from keras import models
from keras import layers
import pandas as pd
import numpy as np

data = pd.read_csv('F:/ML/project/music_data_set_30.csv')
df=data.sample(frac=1).reset_index(drop=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
y=df['genre']
encoder = LabelEncoder()
y = encoder.fit_transform(y)
df.drop(['genre'],axis=1,inplace=True)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(df, dtype = float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train,y_train,epochs=50,batch_size=10)
nn_predictions = model.predict_classes(X_test)
print(classification_report(y_test, nn_predictions))
train_loss, train_acc = model.evaluate(X_train,y_train)
test_loss, test_acc = model.evaluate(X_test,y_test)
print('Training Accuracy: ',train_acc)
print('Test Accuracy: ',test_acc)