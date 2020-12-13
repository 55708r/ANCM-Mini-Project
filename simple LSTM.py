#Import the libraries
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

features = (np.random.randint(10, size=(100, 1)))
print(features.shape)

training_dataset_length = math.ceil(len(features) * .75)
print(training_dataset_length)

#Scale the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(features)

train_data = scaled_data[0:training_dataset_length  , : ]

#Splitting the data
x_train=[]
y_train = []

for i in range(10, len(train_data)):
    x_train.append(train_data[i-10:i,0])
    y_train.append(train_data[i,0])

 #Convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data into 3-D array
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

from keras.layers import Dropout

# Initialising the RNN
model = Sequential()

drop = 0
drops = []
for l in range (0,9):
 np.insert(drops,l,drop)
 drop = drop +0.1
 
 model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
 model.add(Dropout(drop))
 
 # Adding a second LSTM layer and Dropout layer
 model.add(LSTM(units = 50, return_sequences = True))
 model.add(Dropout(drop))

 # Adding a third LSTM layer and Dropout layer
 model.add(LSTM(units = 50, return_sequences = True))
 model.add(Dropout(drop))

 # Adding a fourth LSTM layer and and Dropout layer
 model.add(LSTM(units = 50, return_sequences= True))
 model.add(Dropout(drop))

 ## Adding the output layer
 # For Full connection layer we use dense
 # As the output is 1D so we use unit=1
 model.add(Dense(units = 1))

 ##Adding linear layer with dropout
 #model.add(units = )


 #check model properties
 model.summary()
 #compile and fit the model on 30 epochs
 model.compile(optimizer = 'adam', loss = 'mean_squared_error')
 model.fit(x_train, y_train, epochs = 30, batch_size = 50)

 #Test data set
 test_data = scaled_data[training_dataset_length - 10: , : ]

 #splitting the x_test and y_test data sets
 x_test = []
 y_test =  features[training_dataset_length : , : ] 

 for i in range(10,len(test_data)):
     x_test.append(test_data[i-10:i,0])
    
 #Convert x_test to a numpy array 
 x_test = np.array(x_test)

 #Reshape the data into 3-D array
 x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

 #check predicted values
 predictions = model.predict(x_test) 
 #Undo scaling
 predictions = scaler.inverse_transform(predictions)
 
 #Calculate RMSE score
 rmse = np.sqrt(np.mean(((predictions- y_test)**2)))
 rmses = []
 rmses = np.insert(rmses,l,rmse)
 
## PLOTTING 

#Import plotting package 
import matplotlib.pyplot as plt

#plot rmse to dropout value 
plt.plot(drops,rmses)
plt.xlabel('dropout values')
plt.ylabel('Root mean square error (rsme)')

