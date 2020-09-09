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
	# print(X_train.shape)
	# print(X_train.head())
	# print(X_test.shape)
	# print(Y_train.shape)
	# print(Y_test.shape)


	# do NNLS
	Coef_nnls, rnorm =nnls(X_train.to_numpy(), Y_train.to_numpy(), maxiter=1000)

	#Merge stuff
	Train_Data=X_train.copy()
	Train_Data[TARGET_INDEX]=Y_train
	# print("Train_Data.head()")
	# print(Train_Data.head())

	# add nnl result to matrix
	# print(Train_Data.shape)
	Y_nnls = []
	# print("coef_nnls")
	# print(pd.DataFrame(Coef_nnls))
	for i in range(len(stock_list)):
		index = stock_list[i]
		# print("index: %s" % index)
		# print("i: %s" % i)
		C = np.array(pd.DataFrame(Coef_nnls).iloc[i])
		# print(type(C))
		# print("index: %s, C: %s" % (index, C))
		# print(Train_Data[index].shape)	
		# print(Train_Data[index][0])
		# print(type(Train_Data[index][0]))
		# print("this is appended")
		# print(Train_Data[index][:-1])
		# print(C*Train_Data[index])
		# print("end of appeneded")
		Y_nnls.append(C*Train_Data[index])
		# print(Y_nnls)
		# print("len")
		# print(len(Y_nnls))

	start = Y_nnls[0]
	for i in range(1, len(Y_nnls)):
		start += Y_nnls[i]
		# print("start now added %s" % Y_nnls[i].name)
		# print(start)
	start.name = "Y_nnls"
	# print(start)
	Train_Data[start.name] = start

	# print("Result with linear regression and nnls")
	# print(Train_Data)	

	# print("Measure error")
	rsq = r2_score(Train_Data[TARGET_INDEX],Train_Data['Y_nnls']) #R-Squared from nnls
	# print('R-square, NNLS: ',rsq)

	# print("Coef_nnls")
	# print(Coef_nnls)
	# print(type(Coef_nnls))

	# print("testing")
	# print(X_test.shape)
	# print(type(X_test))
	X_test_numpy = X_test.to_numpy()
	# print("convert ")
	# print(X_test_numpy.shape)
	Y_pred = matmul(X_test_numpy, Coef_nnls)
	# print(type(Y_pred))
	# print(Y_pred.shape)
	# plt.scatter(Y_test,Y_pred)
	# plt.xlabel("Actual Index: $Y_i$")
	# plt.ylabel("Replicated Index: $\hat{Y}_i$")
	# plt.title("Replicated Index vs Actual Index: $Y_i$ vs $\hat{Y}_i$")
	# plt.show()
	# print("Return res for %s" % stock_list)
	return Coef_nnls, rsq


if __name__ == "__main__":
# python3 approximate_index.py [n] [symbol of index to approximate] [historical prices csv] > [results csv]
	# print('Number of arguments:', len(sys.argv), 'arguments.')
	# print('Argument List:', str(sys.argv))

	_, N, TARGET_INDEX, HISTORICAL_PRICES_CSV = sys.argv
	N = int(N)
	SHUFFLE_TIMES = 1

	indices = collections.OrderedDict()
	indices_list = []
	time = collections.OrderedDict()
	with open('/Users/mason/Downloads/201911jrtrader/dow_jones_historical_prices.csv') as csvfile:
		spamreader = csv.reader(csvfile, delimiter='\n')
		for row in spamreader:
			newrow = row[0].split(",")
			# print("newrow:%s", newrow)
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
				# print("%s not in time" % newrow[0])
				# print(newrow[1])
				try:
					time[newrow[0]] = [newrow[1]]

				except:
					# print("error maybe a str")
					continue
	# assuming all the stocks are over the same time interval
	# print(time)
	# print("ttttime")

	time = time[TARGET_INDEX]

	# print("done making dict")
	for thing in indices:
		# print(len(indices[thing]))
		indices_list.append(thing)

	data = pd.DataFrame(indices)
	# print("done converting")
	# print(data.head())
	# print("done printing head")
	# print(data.shape)
	Xall = data.drop(TARGET_INDEX, axis = 1)
	Y = data[TARGET_INDEX]

	Xs = []
	Ys = []
	cs = [] # coeffs
	rs = [] # rsqs 
	stock_lists = []

	# print("break into possible combinations")
	# print("lendata")
	# print(data.shape[1])
	items = range(1, data.shape[1])
	for i in range(SHUFFLE_TIMES):
		# print("shuffling for the %sth time" % i)
		new_X = []
		for j in range(N): 
			choice = random.choice(items)
			while choice in new_X:
				choice = random.choice(items)
			new_X.append(choice)

		# print(new_X)	
		Xdict = {}
		stock_list = []
		for i in range(len(new_X)):
			# print("getting %s" % indices_list[new_X[i]])
			Xdict[indices_list[new_X[i]]] = Xall[indices_list[new_X[i]]]
			stock_list.append(indices_list[new_X[i]])
		# print("stock_list")
		# print(stock_list)

		X = pd.DataFrame(Xdict)
		Xs.append(X)
		# print(X.head())

		c, r = do_nnls_with_given_stocks(X, Y, stock_list, TARGET_INDEX)
		cs.append(c)
		rs.append(r)
		stock_lists.append(stock_list)


	# print("top of result =======")
	besti, bestr = 0, -sys.maxsize
	for i in range(len(rs)):
		# print(rs[i])
		if rs[i] > bestr:
			bestr = rs[i]
			besti = i
	# print("bestr: %s" % bestr)
	# print("bestc:\n %s" % cs[besti])
	coefdic = {}
	print("Symbol, Weight")
	for i in range(len(cs[besti])):
		coefdic[stock_lists[besti][i]] = cs[besti][i]
		print("%s, %.3f" % (stock_lists[besti][i], cs[besti][i]))

	# print("show the plot of replicate index and original")
	X_best = Xs[besti].to_numpy()
	Y_pred = matmul(X_best, cs[besti])
	# print("tttt")
	# print(len(time))
	# print(Y_pred.shape)


	formatter = matplotlib.dates.DateFormatter("%Y-%m-%d")
	fig, ax = plt.subplots()
	ax.grid(True, which='both')


	# plt.scatter(Y,Y_pred)
	# plt.xlabel("Actual Index: $Y_i$")
	# plt.ylabel("Replicated Index: $\hat{Y}_i$")
	# plt.title("Replicated Index vs Actual Index: $Y_i$ vs $\hat{Y}_i$")
	# plt.show()

	plt.plot_date(time,Y_pred, c = "r", label= "Y_pred")

	plt.plot_date(time,Y, c = "b", label = "Y")
	plt.xticks(time)
	plt.legend()
	ax.xaxis.set_tick_params(rotation=30, labelsize=10)
	[l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 30 != 0]
	plt.ylabel("Index: $\hat{Y}_i$")
	plt.xlabel("Dates")

	plt.title("Replicated Index and Actual Index vs Time: \n$Y_i$ and $Y$ vs $\hat{X}_i$, R is %.2f" % bestr)
	# plt.show()
	# print("bottom of main ======")



