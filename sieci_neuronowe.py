import numpy as np
from keras import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

df = pd.read_csv('vectors.csv', sep=',')
data = np.array(df)
X = data[:, 1:].astype('float64')
Y = data[:, 0]


label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Y)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

X_train, X_test, Y_train, Y_test = train_test_split(X, onehot_encoded, test_size=0.3)

model = Sequential()
model.add(Dense(10, input_dim=72, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()

model.fit(X_train, Y_train, epochs=100, batch_size=10, shuffle=True)

Y_pred = model.predict(X_test)
Y_pred_int = np.argmax(Y_pred, axis=1)
Y_test_int = np.argmax(Y_test, axis=1)
cm = confusion_matrix(Y_test_int, Y_pred_int)
print(cm)