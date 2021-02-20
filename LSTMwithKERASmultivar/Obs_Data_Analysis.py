from datetime import datetime
from math import sqrt
from numpy import concatenate
import numpy as np
from matplotlib import pyplot as plt
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
Index(['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)',
       'tmin(C)', 'vp(Pa)', 'streamflow(cfs)'],
      dtype='object')
"""
cleaned = read_csv("allData.csv",index_col=0)
print(cleaned.head())
qObs = cleaned["streamflow(cfs)"]
precip = cleaned["prcp(mm/day)"]
x = cleaned.index[:365]
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(qObs[:365], 'g-')
ax2.plot(precip[:365], 'b-')
plt.xticks(np.arange(0, 365, 30))
ax1.set_xlabel('time')
ax1.set_ylabel('Streamflow (cfs)', color='g')
ax2.set_ylabel('Precipitation (mm/day)', color='b')

plt.show()
"""
streamflow_combined = read_csv("streamflow_combined.csv")
cleaned = read_csv("allData.csv",index_col=0)
qObs = cleaned["streamflow(cfs)"]
precip = cleaned["prcp(mm/day)"]
pyplot.figure(2)
pyplot.plot(qObs, label='Q')
pyplot.plot(precip, label='precip')
pyplot.legend()
pyplot.show()
#pyplot.savefig("Time_series.png")
"""