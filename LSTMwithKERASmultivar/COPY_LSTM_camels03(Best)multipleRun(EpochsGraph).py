from datetime import datetime
from math import sqrt
import numpy as np
from keras.regularizers import l2
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import DataFrame
from pandas import read_csv,read_table
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.layers import LSTM
from keras.metrics import accuracy
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


metObs = read_table(r"E:\MLProject\CAMELS_dataset\Hydrometeorological_Time_Series\basin_dataset_public_v1p2\basin_mean_forcing\daymet\08\07375000_lump_cida_forcing_leap.txt",header=3,sep="\s+|\t+",parse_dates=[['Year','Mnth','Day','Hr']],date_parser=parse,index_col=0)
flowObs = read_table(r"E:\MLProject\CAMELS_dataset\Hydrometeorological_Time_Series\basin_dataset_public_v1p2\usgs_streamflow\08\07375000_streamflow_qc.txt",sep="\s+|\t+",names=['GAGEID','Year','Mnth','Day','streamflow(cfs)','QC_flag'])
metObs.index.name = "date"
metObs["streamflow(cfs)"] = flowObs["streamflow(cfs)"].values
metObs = metObs[metObs["streamflow(cfs)"]!=-999.0]
metObs = metObs[['streamflow(cfs)', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)','tmax(C)', 'tmin(C)', 'vp(Pa)']]
metObs.to_csv('allData.csv')
cleaned = read_csv('allData.csv', index_col=0)
cleaned.drop(cleaned.columns[[1, 3, 4, 5, 6, 7]], axis=1, inplace=True)
values = cleaned.values
# ensure all data is float
values = values.astype('float32')

reframed = series_to_supervised(values, 1, 1)
reframed.drop(reframed.columns[[3]], axis=1, inplace=True)
values = reframed.values
scaler = MinMaxScaler(feature_range=(0,1))

NNtype = 'LSTM'
div = 5
batch_size = 75
epochsList = [10,20,30,40,50,60,70,80,90,100]
input_neurons = 100
output_activation = 'linear'
loss = 'mae'
optimizer = 'adam'
num_runs = 10
runsList = [i+1 for i in range(num_runs)]
ticksList = [runsList[i-1] for i in [div+(div*i) for i in range(int(num_runs/div))]]
ticksList.insert(0,1)
titletxt = "{} Root Mean Squared Error (RMSE)".format(NNtype)
txt = 'batch_size: {}, input neurons: {}, output activation: {}, loss: {}, optimizer: {}, number of runs: {}'.format(
	batch_size, input_neurons, output_activation, loss, optimizer, num_runs)
fig = pyplot.figure(figsize=(6,5))
axes = fig.add_axes([0.2,0.2,0.7,0.7])
pyplot.ylim((200,350))
pyplot.xticks(runsList)
pyplot.xlabel("Run Number")
pyplot.ylabel("RMSE")
fig.suptitle(titletxt,fontsize=16)
colorList = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
list_of_all_rmse_lists = []
for i in epochsList:
	epochs = i
	rmseList = []
	for i in range(num_runs):
		scaled = scaler.fit_transform(values)
		scaled = pd.DataFrame(scaled)
		values = scaled.values
		n_train_days = 15*365
		train = values[:n_train_days, :]
		test = values[n_train_days:, :]
		train_X, train_y = train[:,:-1], train[:, -1]
		test_X, test_y = test[:,:-1], test[:,-1]
		train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
		test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
		#CREATE LSTM
		# design network
		model = Sequential()
		model.add(LSTM(input_neurons, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
		model.add(LSTM(input_neurons, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=False))
		model.add(Dense(1,activation=output_activation))
		model.compile(loss=loss, optimizer=optimizer)
		# fit network
		pyplot.figure(1)
		history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
		yhat = model.predict(test_X)
		test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
		# invert scaling for forecast
		tempTest_X = test_X
		print("first concat shape (yhat,tempTest_X)", yhat.shape,tempTest_X.shape)
		inv_yhat = concatenate((yhat, tempTest_X), axis=1)
		print("inv_yhat shape after first concat: ",inv_yhat.shape)
		inv_yhat = scaler.inverse_transform(inv_yhat)
		inv_yhat = inv_yhat[:, 0]
		# invert scaling for actual
		test_y = test_y.reshape((len(test_y), 1))
		print("second concat shape (test_y,test_X): ", test_y.shape, test_X.shape)
		inv_y = concatenate((test_y, tempTest_X), axis=1)
		print("inv_y shape after second concat: ",inv_y.shape)
		inv_y = scaler.inverse_transform(inv_y)
		inv_y = inv_y[:, 0]
		# calculate RMSE
		rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
		print(rmse)
		rmseList.append(rmse)
	list_of_all_rmse_lists.append(rmseList)


pyplot.figtext(0.5,0.01,txt,wrap=True,horizontalalignment='center',fontsize=12)
for l in list_of_all_rmse_lists:
	for epochs in epochsList:
		color = colorList[int((i/10)-1)]
		axes.plot(runsList, l, color=color, marker='o', label=str(epochs))
axes.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
pyplot.savefig("MultipleRunsRMSE(EpochsGraph)_scaleAfter.png")

