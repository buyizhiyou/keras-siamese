#-*-coding:utf8-*-

__author__ = "buyizhiyou"
__date__ = "2017-10-31"

import keras
from keras.layers import Input, Convolution2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
import numpy.random as rng
import numpy as np
import os
import dill as pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pdb

def Siamese():
	
	#pdb.set_trace()
	input_shape = (105, 105, 1)
	left_input = Input(input_shape)# shape=(?, 105, 105, 1)
	right_input = Input(input_shape)# shape=(?, 105, 105, 1)

	sequential = Sequential()
	sequential.add(Convolution2D(64,10,10,activation='relu',input_shape=input_shape,W_regularizer=l2(2e-4)))
	sequential.add(MaxPooling2D())
	sequential.add(Convolution2D(128,7,7,activation='relu'))
	sequential.add(MaxPooling2D())
	sequential.add(Convolution2D(128,4,4,activation='relu'))
	sequential.add(MaxPooling2D())
	sequential.add(Convolution2D(256,4,4,activation='relu'))
	sequential.add(Flatten())
	sequential.add(Dense(4096,activation="relu",))
	#encode each of the two inputs into a vector with the sequential
	encoded_l = sequential(left_input)#shape=(?, 4096)
	encoded_r = sequential(right_input)#shape=(?, 4096)
	#merge two encoded inputs with the l1 distance between them
	L1_distance = lambda x: K.abs(x[0]-x[1])
	both = keras.layers.merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])#shape=(?, 4096)
	dense1 = Dense(1024,activation='relu')(both)#shape=(?, 1024)
	dense2 = Dense(128,activation='relu')(dense1)#shape=(?,128)
	prediction = Dense(1,activation='relu')(dense2)
	siamese_net = Model(input=[left_input,right_input],output=prediction)
	optimizer = SGD(0.04,momentum=0.6,nesterov=True,decay=0.0003)
	#optimizer = Adam(0.006)
	siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
	sequential.summary()
	siamese_net.summary()

	return siamese_net