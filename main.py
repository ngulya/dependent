#!/usr/bin/env python3
	
import random as r
import pandas as pd
import matplotlib.pyplot as plt
import math as m
import numpy as np
import os
import warnings

import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers.advanced_activations import LeakyReLU

warnings.simplefilter("ignore")
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)

def get_model(num = -1):
	model = Sequential()
	model.add(Dense(24, input_dim = 2, activation=LeakyReLU(0.2)))	
	model.add(Dense(96, activation=LeakyReLU(0.4)))
	model.add(Dense(24, activation=LeakyReLU(0.4)))
	model.add(Dense(12, activation=LeakyReLU(0.1)))
	model.add(Dense(1, activation="linear"))
	model.compile(loss="mean_squared_error", optimizer="adam")
	if num != -1:
		model.load_weights(f'weights_{num}.h5')
	return model

def predict_all_class(data):
	model0 = get_model(0)
	model1 = get_model(1)
	
	data_n = prepare_data_for_nn(data.copy(), num_class = -1)
	for_pred = data_n[['x1', 'x3']].as_matrix()
	res1 = model1.predict(for_pred)
	res0 = model0.predict(for_pred)
	res1 = [i[0] for i in res1]
	res0 = [i[0] for i in res0]
	return res0, res1


def prepare_data_for_nn(data, num_class = -1):
	global TRAIN_SIZE, PI_s
	if num_class != -1:
		data = data[data['y'] == num_class].reset_index(drop = True)
		filt = data['x1'] < PI_s * TRAIN_SIZE

	if normalization_minmax:
		MAX_X, MIN_X = max(data['x1']), min(data['x1'])
		MAX_nums, MIN_nums = max(data['x3']), min(data['x3'])
		if diap_zero_one:
			data['x3'] = (data['x3'] - MIN_nums)/(MAX_nums - MIN_nums)
			data['x1'] = (data['x1'] - MIN_X)/(MAX_X - MIN_X)
			if num_class == -1:
				return data
		else:
			data['x3'] = 2 * ((data['x3'] - MIN_nums)/(MAX_nums - MIN_nums)) - 1
			data['x1'] = 2 * ((data['x1'] - MIN_X)/(MAX_X - MIN_X)) - 1
	else:
		std_x = np.std(data['x1'])
		mean_x = np.mean(data['x1'])
		std_y = np.std(data['x2'])
		mean_y = np.mean(data['x2'])
		data['x3'] = (data['x3'] - np.std(data['x3']))/np.mean(data['x3'])
		data['x1'] =(data['x1'] - std_x)/mean_x

	train_data = data[filt]
	test_data = data[~filt]
	
	# plt.plot(test_data['x1'], test_data['x2'], 'bo')
	# plt.plot(train_data['x1'], train_data['x2'], 'go')
	# plt.show()
	# exit()
	return  train_data[['x1', 'x3']].as_matrix(), list(train_data['x2']),  test_data[['x1', 'x3']].as_matrix(), list(test_data['x2'])

def trainig_model(x_train, y_train, x_test, y_test, num_class):
	model = get_model()
	model.summary()
	if not FIT:
		model.load_weights(f'weights_{num_class}.h5')
	else:
		
		model.fit(x_train, y_train, batch_size = 5, epochs=200, verbose=1, validation_data=(x_test,y_test))
		model.save_weights(f'weights_{num_class}.h5')
	# model.fit(x_train, y_train, batch_size = 5, epochs=1, verbose=1, validation_data=(x_test,y_test))
	y_pred_train = model.predict(x_train)	
	y_pred_test = model.predict(x_test)


	from matplotlib.pyplot import figure
	figure(num=None, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')


	plt.plot(x_train[:,0], y_train,  marker = '.', color = 'grey', markersize=8)
	plt.plot(x_test[:,0], y_test,  marker = '.', color = 'grey', markersize=8)
	plt.plot(x_train[:,0], y_pred_train,  marker = '.', color = 'black', markersize=8)
	plt.plot(x_test[:,0], y_pred_test,  marker = '.', color = 'black', markersize=8)
	
	yy = np.linspace(-1.1, 1.1, 100)
	xx = [x_test[:,0][0] for i in range(100)]
	plt.plot(xx, yy, marker = '.', color = 'black', markersize=16)
	plt.show()
	return y_train, y_pred_train, y_test, y_pred_test

def generate_data(NUM_HORIZ, PERIODS_pi, PI_s):
	l = np.linspace(0, PI_s, NUMS_SPLIT)
	sin = [m.sin(i) for i in l]

	x1 = []
	x2 = []
	y = []
	for _x1 in l:
		for i in range(NUM_HORIZ):
			rr = 2*r.random() - 0.5
			_x2 = m.sin(_x1 + rr)
			x1.append(_x1)
			x2.append(_x2)
			y.append(0)
		x1.append(_x1)
		x2.append(m.sin(_x1))
		y.append(1)

	#'x3':[0 for i in y] for order it's need
	data = pd.DataFrame({'x1':x1, 'x2':x2,'x3':[0 for i in y] ,'y':y})
	data['x3'] = 0
	for nums in range(1, int(PERIODS_pi/2) + 1):
		data['tmp'] = 0
		data['tmp'] = data['x1'] - 2 * nums * m.pi
		data['x3'][(-2*m.pi <= data['tmp']) & (data['tmp'] <= 0)] = nums
	return data

def train_and_get_score(model, x_train, x_test, y_train, y_test):
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)
	y_pred_train = model.predict(x_train)
	return roc_auc_score(y_test, y_pred), roc_auc_score(y_train, y_pred_train)

if __name__ == '__main__':

	PERIODS_pi = 20
	PI_s = PERIODS_pi*m.pi
	NUMS_SPLIT = 50 * PI_s / m.pi
	NUM_HORIZ = 5
	TRAIN_SIZE = 0.7

	if not os.path.exists('data.csv'):
		data = generate_data(NUM_HORIZ, PERIODS_pi, PI_s)
		data.to_csv('data.csv', index = False)
	else:
		data = pd.read_csv('data.csv')

	ix_filter = data['y'] == 1
	fig, ax = plt.subplots()

	gr = TRAIN_SIZE * PI_s
	xx = [gr for i in range(100)]
	yy = np.linspace(-1.1, 1.1, 100)
	
	# # plt.figure(figsize=(15,6))
	# ax.plot(data['x1'][~ix_filter], data['x2'][~ix_filter], marker = '.', color = 'gray',linestyle='dashed')
	# ax.plot(data['x1'][ix_filter], data['x2'][ix_filter], marker = '.', color = 'black',linestyle='dashed')
	# ax.plot(xx, yy, marker = '.', color = 'black', markersize=16)
	# # ax.set_xlabel('x2')
	# # ax.set_ylabel('x1')

	# ax.set_xlabel('x2')
	# ax.set_ylabel('x1')
	
	# plt.show()
	# exit()


	# FIT = True
	FIT = False
	normalization_minmax = True
	diap_zero_one = True

	num_class = 1
	x_train, y_train, x_test, y_test = prepare_data_for_nn(data, num_class)
	y_train, y_pred_train, y_test, y_pred_test = trainig_model(x_train, y_train, x_test, y_test, num_class)
	
	num_class = 0
	x_train, y_train, x_test, y_test = prepare_data_for_nn(data, num_class)
	y_train, y_pred_train, y_test, y_pred_test = trainig_model(x_train, y_train, x_test, y_test, num_class)
	
	
	# predict_without_nn = True
	predict_without_nn = False
	ix_filter = data['x1'] < PI_s * TRAIN_SIZE
	if predict_without_nn:
		feature = ['x1', 'x2', 'x3']
	else:
		res0, res1 = predict_all_class(data)
		data['x0nn'] = res0
		data['x1nn'] = res1
		feature = ['x1', 'x2', 'x0nn', 'x1nn']

	x_train = data[feature][ix_filter]
	x_test = data[feature][~ix_filter]
	y_train = data[['y']][ix_filter]
	y_test = data[['y']][~ix_filter]

	score, score_train = train_and_get_score(DecisionTreeClassifier(), x_train, x_test, y_train, y_test)
	print('DecisionTreeClassifier roc_auc_score:', score, score_train)

