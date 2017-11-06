#-*-coding:utf8-*-

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
from data_process import Siamese_Loader
from siamese import  Siamese

loader = Siamese_Loader("./images/")
#Training loop
evaluate_every = 500 # interval for evaluating on one-shot tasks
loss_every=10 # interval for printing loss (iterations)
batch_size = 128
n_iter = 900000 
N_way = 20 # how many classes for testing one-shot tasks>
n_val = 250 #how mahy one-shot tasks to validate on?
best = 9999
siamese_net = Siamese()
for i in range(1, n_iter):
    (inputs,targets)=loader.get_batch(batch_size)
    loss=siamese_net.train_on_batch(inputs,targets)
    if i % evaluate_every == 0:
        val_acc = loader.test_oneshot(siamese_net,N_way,n_val,verbose=True)
        if val_acc >= best:
            print("saving")
            siamese_net.save('./models/')
            best=val_acc
    if i % loss_every == 0:
        print("iteration {}, training loss: {:.2f},".format(i,loss))


