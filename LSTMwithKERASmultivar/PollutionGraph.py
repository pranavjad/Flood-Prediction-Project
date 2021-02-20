from pandas import read_csv
from matplotlib import pyplot

# load dataset. Header is what row is your header, index_col is what column is your index
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# plot each column
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    # the line below plots all the rows of values and a single column at position group
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i+=1
pyplot.show()