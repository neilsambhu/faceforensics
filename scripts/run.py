# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:41:58 2019
@author: neils
"""

import os
import numpy as np
import cv2
import keras
from keras.utils import multi_gpu_model
import sklearn
import datetime
from tqdm import tqdm
import tensorflow as tf
from keras.applications import Xception

def convertPngToNpy():
    sourceDirs = ['../data/FaceForensics_selfreenactment_images/test/altered/', 
                  '../data/FaceForensics_selfreenactment_images/test/original/']
    destinationDirs = list()
    for sourcePath in sourceDirs:
        destinationPath = sourcePath.replace('FaceForensics_selfreenactment_images', 'FaceForensics_selfreenactment_images_npy')
        destinationDirs.append(destinationPath)
        if not os.path.exists(destinationPath):
               os.makedirs(destinationPath)
    
    if len(sourceDirs) is not len(destinationDirs):
        print('ERROR: length of sourceDirs and destinationDirs is not equal')
    
    for i, sourceDir in enumerate(sourceDirs):
        print('PNG to NPY conversion of {} started at {}'.format(sourceDir, str(datetime.datetime.now())))
        for file in tqdm(os.listdir(sourceDir)):
            if file.endswith('.png'):
                path = os.path.join(sourceDir, file)
                img = cv2.resize(cv2.imread(path), (256,256))
                np.save(os.path.join(destinationDirs[i], file), img)
        print('PNG to NPY conversion of {} ended at {}'.format(sourceDir, str(datetime.datetime.now())))

def directorySearch(directory, label):
    x, y = [], []
    if label is 0:
        fileBadImages = open('../data/FaceForensics_selfreenactment_images/test/BadImagesOriginal.txt', 'w+')
    elif label is 1:
        fileBadImages = open('../data/FaceForensics_selfreenactment_images/test/BadImagesAltered.txt', 'w+')
    else:
        print('label should be 0 or 1')
    #countBadImages = 0
    for file in tqdm(os.listdir(directory)):
        if file.endswith('.png'):
            path = os.path.join(directory, file)
            img = cv2.imread(path)
            if img is None:
                fileBadImages.write(file)
                #countBadImages += 1
                pass
            else:
                x.append(cv2.resize(img,(256,256)))
                y.append(label)
    #print('Bad images count: {}'.format(countBadImages))
    return x, y

# preprocessing
def readImages(pathData):
    # get test data
    pathTestOriginal = '{}test/original/'.format(pathData)
    x_TestOriginal, y_TestOriginal = directorySearch(pathTestOriginal, 0)
    pathTestAltered = '{}test/altered/'.format(pathData)
    x_TestAltered, y_TestAltered = directorySearch(pathTestAltered, 1)

    x_TestOriginal, y_TestOriginal = sklearn.utils.shuffle(x_TestOriginal, y_TestOriginal)
    x_TestAltered, y_TestAltered = sklearn.utils.shuffle(x_TestAltered, y_TestAltered)
    
    print('Length of x_TestOriginal: {}'.format(len(x_TestOriginal)))
    print('Length of y_TestOriginal: {}'.format(len(y_TestOriginal)))
    print('Length of x_TestAltered: {}'.format(len(x_TestAltered)))
    print('Length of y_TestAltered: {}'.format(len(y_TestAltered)))
    
    # setup training data
    train_x = np.asarray(x_TestOriginal[0:9*len(x_TestOriginal)//10] + x_TestAltered[0:9*len(x_TestAltered)//10])
    train_y = np.asarray(y_TestOriginal[0:9*len(y_TestOriginal)//10] + y_TestAltered[0:9*len(y_TestAltered)//10])
    train_x, train_y = sklearn.utils.shuffle(train_x, train_y)
    
    # setup testing data
    test_x = np.asarray(x_TestOriginal[9*len(x_TestOriginal)//10:-1] + x_TestAltered[9*len(x_TestAltered)//10:-1])
    test_y = np.asarray(y_TestOriginal[9*len(y_TestOriginal)//10:-1] + y_TestAltered[9*len(y_TestAltered)//10:-1])
    test_x, test_y = sklearn.utils.shuffle(test_x, test_y)
    
    print('Length of train_x: {}'.format(len(train_x)))
    print('Length of train_y: {}'.format(len(train_y)))
    print('Length of test_x: {}'.format(len(test_x)))
    print('Length of test_y: {}'.format(len(test_y)))
    
    return train_x, train_y, test_x, test_y 

def buildModel():
    # create model
    model = keras.models.Sequential()
#    with tf.device('/cpu:0'):
#        model = Xception(weights=None, input_shape=(256, 256, 3), classes=2)
    
    # 2 layers of convolution
    model.add(keras.layers.Conv2D(128, 3, activation='relu', input_shape=(256,256,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(128, 3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    
    # max pooling
    model.add(keras.layers.MaxPooling2D())
    
    # flatten
    model.add(keras.layers.Flatten())
    
    # fully connected layer
    model.add(keras.layers.Dense(100, activation='relu'))
    
    # dropout
    model.add(keras.layers.Dropout(0.5))
    
    # final dense layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    # multiple GPUs
    model = multi_gpu_model(model, gpus=16)
    
    # compile
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=keras.losses.binary_crossentropy, metrics=['acc'])    
    
    return model

def main():
    print('Image reading started at {}'.format(str(datetime.datetime.now())))
    train_x, train_y, test_x, test_y = readImages('../data/FaceForensics_selfreenactment_images/')
    print('Image reading finished at {}'.format(str(datetime.datetime.now())))
    
    print('Model building started at {}'.format(str(datetime.datetime.now())))
    model = buildModel()
    print('Model building finished at {}'.format(str(datetime.datetime.now())))
    
    print('Model evaluation started at {}'.format(str(datetime.datetime.now())))
    # fit model to data
    model.fit(x=train_x, y=train_y, batch_size=16, epochs=50, verbose=2, callbacks=[keras.callbacks.EarlyStopping('loss',0.0001,2)])
    print(model.evaluate(test_x, test_y))
    print('Model evaluation finished at {}'.format(str(datetime.datetime.now())))

if __name__ == "__main__":
    main()
    print('Finished at {}'.format(str(datetime.datetime.now())))