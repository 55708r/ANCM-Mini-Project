import math

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

drop = 0.0 #set dropout value
rmses = []
rmsesT = []
varrmses = []
varrmsesT = []

for j in range(10):
	rmse =[]
	rmseT =[]
	print(drop)
	for i in range(10):
		print(i)
		look_back = 1
		trainX, trainY = create_dataset(train, look_back)
		testX, testY = create_dataset(test, look_back)
		# reshape input to be [samples, time steps, features]
		trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
		testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
		#create and fit the LSTM network
		model = Sequential()
		model.add(LSTM(4, input_shape=(1, look_back), return_sequences=True))
		model.add(Dropout(drop))
		model.add(LSTM(4, input_shape=(1, look_back)))
		model.add(Dropout(drop))
		model.add(Dense(1))
		## Adding the output layer
		model.compile(loss='mean_squared_error', optimizer='adam')
		model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
		# make predictions
		trainPredict = model.predict(trainX)
		testPredict = model.predict(testX)
		# invert predictions
		trainPredict = scaler.inverse_transform(trainPredict)
		trainY = scaler.inverse_transform([trainY])
		testPredict = scaler.inverse_transform(testPredict)
		testY = scaler.inverse_transform([testY])
		# calculate root mean squared error
		trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
		print('Train Score: %.2f RMSE' % (trainScore))
		testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
		print('Test Score: %.2f RMSE' % (testScore))
		rmse.append(trainScore)
		rmseT.append(testScore)
		i = i+1

	varrmses.append(rmse) #add rmse scores (10) for a given dropout value 
	varrmsesT.append(rmseT) #add rmse scores for a given dropout value
	meanrmse = np.mean(rmse)
	meanrmseT =np.mean(rmseT)
	rmses.append(meanrmse)
	rmsesT.append(meanrmseT)
	drop = drop+0.1
	j = j+1

print(varrmses)
print(varrmsesT)
print(rmses)
print(rmsesT)

#shift train predictions for plotting
#trainPredictPlot = np.empty_like(dataset)
#trainPredictPlot[:, :] = np.nan
#trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
#shift test predictions for plotting
#testPredictPlot = np.empty_like(dataset)
#testPredictPlot[:, :] = np.nan
#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	
# plot baseline and predictions
#plt.plot(scaler.inverse_transform(dataset))
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
#plt.xlabel('Number of airflight passengers in this month (t) in thousands', fontsize = 11)
#plt.ylabel('Number of airflight passengers in next month (t+1) in thousands', fontsize = 11)
#plt.title('Prediction of airflight passenger numbers by a single-layer LSTM', fontsize=14)
#plt.show()
#plt.savefig('Airflight predictions for dropout value 0.7.png')

np.save('Train RMSE scores per dropout value', varrmses)
np.save('Test RMSE scores per dropout value', varrmsesT)
np.save('Train RMSE scores per dropout value averaged over 10 runs', rmses)
np.save('Test RMSE scores per dropout value averaged over 10 runs', rmsesT)


