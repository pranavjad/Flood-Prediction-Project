import datetime
from math import sqrt
import numpy as np
import pandas as pd
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import glob

df1 = pd.DataFrame(np.arange(12),columns=['A'])
df2 = pd.DataFrame(np.arange(12),columns=['B'])
print(df2.index,df1.index)
df2.index = df1.index
print(pd.concat([df1,df2],axis=1))
