# -*- coding: utf-8 -*-

# Car Purchase Amount Predictions Using ANNs

## Import libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense

"""## Import dataset"""

cars = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')

cars.head(5)

"""## Visualize data"""

sns.pairplot(cars)

X = cars.drop(['Customer Name', 'Customer e-mail', 'Car Purchase Amount'], axis = 1)
X.head(5)
X= X.iloc[:,:].values
labelencoder_X = LabelEncoder()
X[:,0]= labelencoder_X.fit_transform(X[:,0])
X= pd.DataFrame(X)
X

y = cars['Car Purchase Amount']
y.shape

"""## Scaling data"""

from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)
y = y.values.reshape(-1,1)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

"""### Splitting data into training and testing"""

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)

"""### Training the model"""

model = Sequential()
model.add(Dense(25, input_dim=6, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

epochs_hist = model.fit(X_train, y_train, epochs=20, batch_size=25,  verbose=1, validation_split=0.2)
print(epochs_hist.history.keys())

plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])

plt.title('Model Loss Progression During Training/Validation')
plt.ylabel('Training and Validation Losses')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])

# Gender, Age, Annual Salary, Credit Card Debt, Net Worth 

# ***(Note that input data must be normalized)***

X_test_sample = np.array([[5,0, 0.4370344,  0.53515116, 0.57836085, 0.22342985]])
#X_test_sample = np.array([[1, 0.53462305, 0.51713347, 0.46690159, 0.45198622]])

y_predict_sample = model.predict(X_test_sample)

print('Expected Purchase Amount=', y_predict_sample)
y_predict_sample_orig = scaler_y.inverse_transform(y_predict_sample)
print('Expected Purchase Amount=', y_predict_sample_orig)
