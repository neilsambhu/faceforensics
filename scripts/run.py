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
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import sklearn
import datetime
from tqdm import tqdm
import fnmatch
from time import gmtime, strftime
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Process
import itertools

#def convertPngToNpy():
#    sourceDirs = ['../data/FaceForensics_selfreenactment_images/test/altered/', 
#                  '../data/FaceForensics_selfreenactment_images/test/original/']
#    destinationDirs = list()
#    for sourcePath in sourceDirs:
#        destinationPath = sourcePath.replace('FaceForensics_selfreenactment_images', 'FaceForensics_selfreenactment_images_npy')
#        destinationDirs.append(destinationPath)
#        if not os.path.exists(destinationPath):
#               os.makedirs(destinationPath)
#    
#    if len(sourceDirs) is not len(destinationDirs):
#        print('ERROR: length of sourceDirs and destinationDirs is not equal')
#    
#    for i, sourceDir in enumerate(sourceDirs):
#        print('PNG to NPY conversion of {} started at {}'.format(sourceDir, str(datetime.datetime.now())))
#        for file in tqdm(os.listdir(sourceDir)):
#            if file.endswith('.png'):
#                path = os.path.join(sourceDir, file)
#                img = cv2.resize(cv2.imread(path), (256,256))
#                np.save(os.path.join(destinationDirs[i], file), img)
#        print('PNG to NPY conversion of {} ended at {}'.format(sourceDir, str(datetime.datetime.now())))

def directorySearch(directory, label, dataName):
    print('Started directory search of {} at {}'.format(dataName, str(datetime.datetime.now())))
    x, y = [], []
#    time = strftime("%Y-%m-%d--%H-%M-%S", gmtime())
    if label is 0:
#        fileBadImages = open('../data/FaceForensics_selfreenactment_images/{0}-BadImagesOriginal.txt'.format(time), 'w+')
        pass
    elif label is 1:
#        fileBadImages = open('../data/FaceForensics_selfreenactment_images/{0}-BadImagesAltered.txt'.format(time), 'w+')
        pass
    else:
        print('Error: label should be 0 or 1')
        return
    countBadImages = 0
    for file in tqdm(os.listdir(directory)[0:1000]):
        if file.endswith('.png'):
            path = os.path.join(directory, file)
            img = cv2.imread(path)
            if img is None:
#                fileBadImages.write(file + '\n')
                countBadImages += 1
                pass
            else:
                x.append(cv2.resize(img,(128,128)))
                y.append(label)
    if countBadImages > 0:
        print('Bad images count: {}'.format(countBadImages))
    print('Ended directory search of {} at {}'.format(dataName, str(datetime.datetime.now())))
    return x, y

def directorySearchParallelHelper(directory, file, label):
    if file.endswith('.png'):
        path = os.path.join(directory, file)
        img = cv2.imread(path)
        if img is not None:
            return cv2.resize(img,(128,128)), label
        else:
            print('Error: image {} is None'.format(path))
            return None, None

def directorySearchParallel(directory, label, dataName):
    print('Started directory search of {} at {}'.format(dataName, str(datetime.datetime.now())))
    x, y = [], []
    pool = ThreadPool()
#    time = strftime("%Y-%m-%d--%H-%M-%S", gmtime())
    if label is 0:
#        fileBadImages = open('../data/FaceForensics_selfreenactment_images/{0}-BadImagesOriginal.txt'.format(time), 'w+')
        pass
    elif label is 1:
#        fileBadImages = open('../data/FaceForensics_selfreenactment_images/{0}-BadImagesAltered.txt'.format(time), 'w+')
        pass
    else:
        print('Error: label should be 0 or 1')
        return

    xy = pool.starmap(directorySearchParallelHelper, 
                        zip(itertools.repeat(directory),
                            os.listdir(directory),
                            itertools.repeat(label)))
    pool.close()
    pool.join()
    x, y = zip(*xy)
    print('Ended directory search of {} at {}'.format(dataName, str(datetime.datetime.now())))
    return x, y

def verifyLength(list1, list2, list1Name, list2Name):
    if len(list1) != len(list2):
        print('Error: {0} length does not equal {1} length'.format(list1Name, list2Name))

# preprocessing
def readImages(pathData):
    # get test data
    pathTestOriginal = '{}test/original/'.format(pathData)
    x_TestOriginal, y_TestOriginal = directorySearch(pathTestOriginal, 0, 'Test Original')
    verifyLength(x_TestOriginal, y_TestOriginal, 'x_TestOriginal', 'y_TestOriginal')
    pathTestAltered = '{}test/altered/'.format(pathData)
    x_TestAltered, y_TestAltered = directorySearch(pathTestAltered, 1, 'Test Altered')
    verifyLength(x_TestAltered, y_TestAltered, 'x_TestAltered', 'y_TestAltered')

    # get train data
    pathTrainOriginal = '{}train/original/'.format(pathData)
    x_TrainOriginal, y_TrainOriginal = directorySearch(pathTrainOriginal, 0, 'Train Original')
    verifyLength(x_TrainOriginal, y_TrainOriginal, 'x_TrainOriginal', 'y_TrainOriginal')
    pathTrainAltered = '{}train/altered/'.format(pathData)
    x_TrainAltered, y_TrainAltered = directorySearch(pathTrainAltered, 1, 'Train Altered')
    verifyLength(x_TrainAltered, y_TrainAltered, 'x_TrainAltered', 'y_TrainAltered')

    # get val data
    pathValOriginal = '{}val/original/'.format(pathData)
    x_ValOriginal, y_ValOriginal = directorySearch(pathValOriginal, 0, 'Val Original')
    verifyLength(x_ValOriginal, y_ValOriginal, 'x_ValOriginal', 'y_ValOriginal')
    pathValAltered = '{}val/altered/'.format(pathData)
    x_ValAltered, y_ValAltered = directorySearch(pathValAltered, 1, 'Val Altered')
    verifyLength(x_ValAltered, y_ValAltered, 'x_ValAltered', 'y_ValAltered')

    # setup testing data
    test_x = np.asarray(x_TestOriginal + x_TestAltered)
    test_y = np.asarray(y_TestOriginal + y_TestAltered)
    test_x, test_y = sklearn.utils.shuffle(test_x, test_y)
    
    # setup training data
    train_x = np.asarray(x_TrainOriginal + x_TrainAltered)
    train_y = np.asarray(y_TrainOriginal + y_TrainAltered)
    train_x, train_y = sklearn.utils.shuffle(train_x, train_y)

    # setup validation data
    val_x = np.asarray(x_ValOriginal + x_ValAltered)
    val_y = np.asarray(y_ValOriginal + y_ValAltered)
    val_x, val_y = sklearn.utils.shuffle(val_x, val_y)
    
    return test_x, test_y, train_x, train_y, val_x, val_y

def find_files(base, pattern):
    '''Return list of files matching pattern in base folder.'''
    return [n for n in fnmatch.filter(os.listdir(base), pattern) if
        os.path.isfile(os.path.join(base, n))]

def buildModel(pathBase):
    # create model
    model = keras.models.Sequential()
#    with tf.device('/cpu:0'):
#        model = Xception(weights=None, input_shape=(256, 256, 3), classes=2)
    
    # 2 layers of convolution
    model.add(keras.layers.Conv2D(8, 3, activation='relu', input_shape=(128,128,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(8, 3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    
    # max pooling
    model.add(keras.layers.MaxPooling2D())
    
    # 2 layers of convolution
    model.add(keras.layers.Conv2D(8, 3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(8, 3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    
    # max pooling
    model.add(keras.layers.MaxPooling2D())
    
    # flatten
    model.add(keras.layers.Flatten())
    
    # fully connected layer
    model.add(keras.layers.Dense(100, activation='relu'))
    
    # dropout
    model.add(keras.layers.Dropout(0.1))
    
    # final dense layer
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    # multiple GPUs
    model = multi_gpu_model(model, gpus=16)
    
    # resume from checkpoint
    savedModelFiles = find_files(pathBase, '2019-02-07--*.hdf5')
    if len(savedModelFiles) > 0:
        if len(savedModelFiles) > 1:
            print('Error: There are multiple saved model files.')
            return
        print("Resumed model's weights from {}".format(savedModelFiles[-1]))
        # load weights
        model.load_weights(os.path.join(pathBase, savedModelFiles[-1]))

    # compile
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=keras.losses.binary_crossentropy, metrics=['acc'])
    
    return model

def main():
    pathBase = '../data/FaceForensics_selfreenactment_images/'
    
    print('Image reading started at {}'.format(str(datetime.datetime.now())))
    test_x, test_y, train_x, train_y, val_x, val_y = readImages(pathBase)
    print('Image reading finished at {}'.format(str(datetime.datetime.now())))
    
    print('Model building started at {}'.format(str(datetime.datetime.now())))
    model = buildModel(pathBase)
    print('Model building finished at {}'.format(str(datetime.datetime.now())))
    
    print('Model evaluation started at {}'.format(str(datetime.datetime.now())))
    # fit model to data
    time = strftime("%Y-%m-%d--%H-%M-%S", gmtime())
    checkpoint = ModelCheckpoint('{0}{1}_{{epoch:02d}}-{{val_acc:.2f}}.hdf5'.format(pathBase, time), 
								 monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    earlyStop = EarlyStopping('loss',0.0001,2)
    callbacks_list = [checkpoint, earlyStop]
    model.fit(x=train_x, y=train_y, batch_size=512, epochs=50, verbose=2, 
              callbacks=callbacks_list,
              validation_data=(val_x, val_y),
              initial_epoch=0)    
    print(model.evaluate(test_x, test_y))
    print('Model evaluation finished at {}'.format(str(datetime.datetime.now())))

if __name__ == "__main__":
    main()
    print('Finished at {}'.format(str(datetime.datetime.now())))