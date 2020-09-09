import sys
import csv
import collections
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.optimize import nnls
import numpy as np
from numpy import matmul
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
from datetime import datetime


def do_nnls_with_given_stocks(X, Y, stock_list, TARGET_INDEX):
	# split test and training
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)

	# do NNLS
	Coef_nnls, rnorm =nnls(X_train.to_numpy(), Y_train.to_numpy(), maxiter=1000)
	Train_Data=X_train.copy()
	Train_Data[TARGET_INDEX]=Y_train

	Y_nnls = []
	for i in range(len(stock_list)):
		index = stock_list[i]

		C = np.array(pd.DataFrame(Coef_nnls).iloc[i])
		Y_nnls.append(C*Train_Data[index])

	start = Y_nnls[0]
	for i in range(1, len(Y_nnls)):
		start += Y_nnls[i]
	start.name = "Y_nnls"
	Train_Data[start.name] = start

	# Measure error
	rsq = r2_score(Train_Data[TARGET_INDEX],Train_Data['Y_nnls']) #R-Squared from nnls

	X_test_numpy = X_test.to_numpy()
	Y_pred = matmul(X_test_numpy, Coef_nnls)
	return Coef_nnls, rsq


if __name__ == "__main__":
	# python3 approximate_index.py [n] [symbol of index to approximate] [historical prices csv] > [results csv]

	_, N, TARGET_INDEX, HISTORICAL_PRICES_CSV = sys.argv
	N = int(N)
	# set number of tries for n indices
	SHUFFLE_TIMES = 10

	indices = collections.OrderedDict()
	indices_list = []
	time = collections.OrderedDict()
	with open('/Users/mason/Downloads/201911jrtrader/dow_jones_historical_prices.csv') as csvfile:
		spamreader = csv.reader(csvfile, delimiter='\n')
		for row in spamreader:
			newrow = row[0].split(",")
			if newrow[0] in indices:
				try:
					indices[newrow[0]].append(float(newrow[2]))
				except:
					# print("error maybe a str")
					continue
			else:
				try:
					indices[newrow[0]] = [float(newrow[2])]
				except:
					# print("error maybe a str")
					continue
			if newrow[0] in time:
				try:
					time[newrow[0]].append(newrow[1])
				except:
					# print("error maybe a str")
					continue
			else:
				try:
					time[newrow[0]] = [newrow[1]]
				except:
					# print("error maybe a str")
					continue
	# assuming all the stocks are over the same time interval
	time = time[TARGET_INDEX]

	for thing in indices:
		indices_list.append(thing)

	data = pd.DataFrame(indices)
	Xall = data.drop(TARGET_INDEX, axis = 1)
	Y = data[TARGET_INDEX]

	Xs = []
	Ys = []
	cs = [] # coeffs
	rs = [] # rsqs 
	stock_lists = []

	# print("break into possible combinations")
	items = range(1, data.shape[1])
	for i in range(SHUFFLE_TIMES):
		new_X = []
		for j in range(N): 
			choice = random.choice(items)
			while choice in new_X:
				choice = random.choice(items)
			new_X.append(choice)

		Xdict = {}
		stock_list = []
		for i in range(len(new_X)):
			Xdict[indices_list[new_X[i]]] = Xall[indices_list[new_X[i]]]
			stock_list.append(indices_list[new_X[i]])

		X = pd.DataFrame(Xdict)
		Xs.append(X)

		c, r = do_nnls_with_given_stocks(X, Y, stock_list, TARGET_INDEX)
		cs.append(c)
		rs.append(r)
		stock_lists.append(stock_list)

	# get the coefs with best r
	besti, bestr = 0, -sys.maxsize
	for i in range(len(rs)):
		if rs[i] > bestr:
			bestr = rs[i]
			besti = i

	coefdic = {}
	print("Symbol, Weight")
	for i in range(len(cs[besti])):
		coefdic[stock_lists[besti][i]] = cs[besti][i]
		print("%s, %.3f" % (stock_lists[besti][i], cs[besti][i]))
	print("%s, %.3f" % ("r", bestr))
	print("%s, %d" % ("SHUFFLE_TIMES", SHUFFLE_TIMES))


	#===optionally, plot the replicated index with actual index===
	# X_best = Xs[besti].to_numpy()
	# Y_pred = matmul(X_best, cs[besti])

	# formatter = matplotlib.dates.DateFormatter("%Y-%m-%d")
	# fig, ax = plt.subplots()
	# ax.grid(True, which='both')


	# plt.scatter(Y,Y_pred)
	# plt.xlabel("Actual Index: $Y_i$")
	# plt.ylabel("Replicated Index: $\hat{Y}_i$")
	# plt.title("Replicated Index vs Actual Index: $Y_i$ vs $\hat{Y}_i$")
	# plt.show()

	# plt.plot_date(time,Y_pred, c = "r", label= "Y_pred")

	# plt.plot_date(time,Y, c = "b", label = "Y")
	# plt.xticks(time)
	# plt.legend()
	# ax.xaxis.set_tick_params(rotation=30, labelsize=10)
	# [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 30 != 0]
	# plt.ylabel("Index: $\hat{Y}_i$")
	# plt.xlabel("Dates")

	# plt.title("Replicated Index and Actual Index vs Time: \n$Y_i$ and $Y$ vs $\hat{X}_i$, R is %.2f" % bestr)
	# plt.show()