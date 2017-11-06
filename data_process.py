#-*-coding:utf8-*-

__author__ = "buyizhiyou"
__date__ = "2017-10-31"


import numpy.random as random
import numpy as np
import os
import dill as pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self, path):
        self.data = {}
        self.categories = {}
        self.info = {}
        with open(os.path.join(path,"train.pickle"),"rb") as f:
            (X,c) = pickle.load(f)
            self.data["train"] = X
            self.categories["train"] = c
        with open(os.path.join(path,"val.pickle"),"rb") as f:
            (X,c) = pickle.load(f)
            self.data["val"] = X
            self.categories["val"] = c
        self.n_classes,self.n_examples,self.w,self.h = self.data["train"].shape
        self.n_val,self.n_ex_val,_,_ = self.data['val'].shape

    def get_batch(self,n,s="train"):
        """Create batch of n pairs, half same class, half different class"""
        X=self.data[s]
        categories = random.choice(self.n_classes,size=(n,),replace=False)
        pairs=[np.zeros((n, self.h, self.w,1)) for i in range(2)]
        targets=np.zeros((n,))
        targets[n//2:] = 1
        for i in range(n):
            category = categories[i]
            idx_1 = random.randint(0,self.n_examples)
            pairs[0][i,:,:,:] = X[category,idx_1].reshape(self.w,self.h,1)
            idx_2 = random.randint(0,self.n_examples)
            #pick images of same class for 1st half, different for 2nd
            category_2 = category if i >= n//2 else (category + random.randint(1,self.n_classes)) % self.n_classes
            pairs[1][i,:,:,:] = X[category_2,idx_2].reshape(self.w,self.h,1)
        return pairs, targets

    def make_oneshot_task(self,N,s="val",language=None):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        X=self.data[s]
        n_classes, n_examples = X.shape[0],X.shape[1]
        if language is not None:
            low, high = self.categories[s][language]
            if N > high - low:
                raise ValueError("This language ({}) has less than {} letters".format(language, N))
            categories = random.choice(range(low,high),size=(N,),replace=False)
        else:#if no language specified just pick a bunch of random letters
            categories = random.choice(range(n_classes),size=(N,),replace=False)
        indices = random.randint(0,self.n_examples,size=(N,))
        true_category = categories[0]
        ex1, ex2 = random.choice(n_examples,replace=False,size=(2,))
        test_image = np.asarray([X[true_category,ex1,:,:]]*N).reshape(N,self.w,self.h,1)
        support_set = X[categories,indices,:,:]
        support_set[0,:,:] = X[true_category,ex2]
        support_set = support_set.reshape(N,self.w,self.h,1)
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image,support_set]

        return pairs, targets

    def test_oneshot(self,model,N,k,s="val",verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} unique {} way one-shot learning tasks ...".format(k,N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N,s)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct+=1
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct
