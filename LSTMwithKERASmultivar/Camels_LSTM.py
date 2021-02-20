from datetime import datetime
from math import sqrt
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
from keras.layers import Dense
from keras.layers import LSTM

"""
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


datasetInput = read_csv(r"C:\MLProject\CSV07375000_lump_cida_forcing_leap (1).csv")

datasetOutput = read_csv(r"C:\MLProject\CSV07375000_streamflow_qc.csv",header=None)

datasetInput.drop(['dayl(s)','srad(W/m2)','swe(mm)','tmax(C)','tmin(C)','vp(Pa)'],axis=1,inplace=True)

datasetOutput.drop([0,1,2,3],axis='columns',inplace=True)
print(datasetInput.index,datasetOutput.index)
datasetInput.index = datasetOutput.index

datasetCleaned = pd.concat([datasetOutput,datasetInput],axis=1,ignore_index=True)

datasetInput.to_csv("datasetCleaned.csv")
datasetCleaned = read_csv("datasetCleaned.csv")
print(datasetOutput.head())
print(datasetInput.head())
print(datasetCleaned.head())
"""

data = [{'a': 1, 'b': 2, 'c': 3}, {'a': 10, 'b': 20, 'c': 30}]
df = pd.DataFrame(data)
print(df)