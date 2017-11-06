#-*-coding:utf8-*-

import sys
import numpy as np
from scipy.misc import imread
import dill as pickle
import os
import matplotlib.pyplot as plt
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--path",help="Path where omniglot folder resides")
parser.add_argument("--save", help = "Path to pickle data to.", default=os.getcwd())
args = parser.parse_args()
data_path = os.path.join(args.path,"omniglot")
train_folder = os.path.join(data_path,'images_background')
valpath = os.path.join(data_path,'images_evaluation')
save_path = args.save
lang_dict = {}

def loadimgs(path,n=0):
    #if data not already unzipped, unzip it.
    if not os.path.exists(path):
        print("unzipping")
        os.chdir(data_path)
        os.system("unzip {}".format(path+".zip" ))
    X = []
    lang_dict = {}
    curr_y = n
    #we load every alphabet seperately so we can isolate them later
    for alphabet in os.listdir(path):#一种语言
        print("loading alphabet: " + alphabet)
        lang_dict[alphabet] = [curr_y,None]
        alphabet_path = os.path.join(path,alphabet)
        #every letter/category has it's own column in the array, so  load seperately
        for letter in os.listdir(alphabet_path):#一种语言的一个字母
            category_images=[]
            letter_path = os.path.join(alphabet_path, letter)
            for filename in os.listdir(letter_path):#一种语言的一个字母的一个样本
                image_path = os.path.join(letter_path, filename)
                image = imread(image_path)
                category_images.append(image)
            try:
                X.append(np.stack(category_images))
            #edge case  - last one
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
            curr_y += 1
            lang_dict[alphabet][1] = curr_y - 1
    X = np.stack(X)
    return X,lang_dict

X,c=loadimgs(train_folder)
with open(os.path.join(save_path,"train.pickle"), "wb") as f:
	pickle.dump((X,c),f)


X,c=loadimgs(valpath)
with open(os.path.join(save_path,"val.pickle"), "wb") as f:
	pickle.dump((X,c),f)
