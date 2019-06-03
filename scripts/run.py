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
from keras import regularizers
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Reshape, Conv2D, LSTM, Input, Lambda, MaxPooling2D
from keras.models import Sequential, Model, load_model
import sklearn
from sklearn.metrics import confusion_matrix
import datetime
from tqdm import tqdm
import fnmatch
from time import gmtime, strftime
from multiprocessing.dummy import Pool as ThreadPool
import itertools
import sys

imgSize = 128
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

def directorySearch(directory, label, dataName, numVideos=1):
    print('Started directory search of {} at {}'.format(dataName, str(datetime.datetime.now())))
    x, y = [], []
#    x, y = np.empty([128,128,3]), np.empty([128,128,3])
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
#    for file in tqdm(sklearn.utils.shuffle(os.listdir(directory))[0:10*numVideos]):
#    for file in tqdm(sklearn.utils.shuffle(os.listdir(directory))[0:10]):
    for file in tqdm(sklearn.utils.shuffle(os.listdir(directory))):
        if file.endswith('.png'):
            path = os.path.join(directory, file)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
#                fileBadImages.write(file + '\n')
                countBadImages += 1
                pass
            else:
                x.append(cv2.resize(img,(imgSize,imgSize)))
                y.append(label)
#                x = np.concatenate(x, cv2.resize(img,(imgSize,imgSize)))
#                y = np.concatenate(y, label)
#                x += cv2.resize(img,(imgSize,imgSize))
#                y += label

    if countBadImages > 0:
        print('Bad images count: {}'.format(countBadImages))
    print('Ended directory search of {} at {}'.format(dataName, str(datetime.datetime.now())))
    return x, y

def directorySearchParallelHelper(directory, file, label):
    if file.endswith('.png'):
        path = os.path.join(directory, file)
        img = cv2.imread(path)
        if img is not None:
            return cv2.resize(img,(imgSize,imgSize)), label
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
    x_TestOriginal, y_TestOriginal = directorySearch(pathTestOriginal, 0, 'Test Original', 150)
    verifyLength(x_TestOriginal, y_TestOriginal, 'x_TestOriginal', 'y_TestOriginal')
    pathTestAltered = '{}test/altered/'.format(pathData)
    x_TestAltered, y_TestAltered = directorySearch(pathTestAltered, 1, 'Test Altered', 150)
    verifyLength(x_TestAltered, y_TestAltered, 'x_TestAltered', 'y_TestAltered')

    # get train data
    pathTrainOriginal = '{}train/original/'.format(pathData)
    x_TrainOriginal, y_TrainOriginal = directorySearch(pathTrainOriginal, 0, 'Train Original', 704)
    verifyLength(x_TrainOriginal, y_TrainOriginal, 'x_TrainOriginal', 'y_TrainOriginal')
    pathTrainAltered = '{}train/altered/'.format(pathData)
    x_TrainAltered, y_TrainAltered = directorySearch(pathTrainAltered, 1, 'Train Altered', 704)
    verifyLength(x_TrainAltered, y_TrainAltered, 'x_TrainAltered', 'y_TrainAltered')

    # get val data
    pathValOriginal = '{}val/original/'.format(pathData)
    x_ValOriginal, y_ValOriginal = directorySearch(pathValOriginal, 0, 'Val Original', 150)
    verifyLength(x_ValOriginal, y_ValOriginal, 'x_ValOriginal', 'y_ValOriginal')
    pathValAltered = '{}val/altered/'.format(pathData)
    x_ValAltered, y_ValAltered = directorySearch(pathValAltered, 1, 'Val Altered', 150)
    verifyLength(x_ValAltered, y_ValAltered, 'x_ValAltered', 'y_ValAltered')

    # setup testing data
    test_x = np.asarray(x_TestOriginal + x_TestAltered)
    test_y = np.asarray(y_TestOriginal + y_TestAltered)
#    test_x = np.concatenate(x_TestOriginal, x_TestAltered)
#    test_y = np.concatenate(y_TestOriginal, y_TestAltered)
#    test_x = x_TestOriginal + x_TestAltered
#    test_y = y_TestOriginal + y_TestAltered
#    test_x, test_y = sklearn.utils.shuffle(test_x, test_y)
    
    # setup training data
    train_x = np.asarray(x_TrainOriginal + x_TrainAltered)
    train_y = np.asarray(y_TrainOriginal + y_TrainAltered)
#    train_x = np.concatenate(x_TrainOriginal, x_TrainAltered)
#    train_y = np.concatenate(y_TrainOriginal, y_TrainAltered)
#    train_x = x_TrainOriginal + x_TrainAltered
#    train_y = y_TrainOriginal + y_TrainAltered

#    train_x, train_y = sklearn.utils.shuffle(train_x, train_y)

    # setup validation data
    val_x = np.asarray(x_ValOriginal + x_ValAltered)
    val_y = np.asarray(y_ValOriginal + y_ValAltered)
#    val_x = np.concatenate(x_ValOriginal, x_ValAltered)
#    val_y = np.concatenate(y_ValOriginal, y_ValAltered)
#    val_x = x_ValOriginal + x_ValAltered
#    val_y = y_ValOriginal + y_ValAltered
#    val_x, val_y = sklearn.utils.shuffle(val_x, val_y)
    
    # normalize x data
#    test_x = test_x.astype('float32')/255.0
#    train_x = train_x.astype('float32')/255.0
#    val_x = val_x.astype('float32')/255.0
#    test_x /= 255.0
#    train_x /= 255.0
#    val_x /= 255.0
    
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
    model.add(keras.layers.Conv2D(128, 3, activation='relu', input_shape=(imgSize,imgSize,3)))
    model.add(keras.layers.BatchNormalization())
#    # dropout
##    model.add(keras.layers.Dropout(0.50))
    model.add(keras.layers.Conv2D(128, 3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
#    # dropout
##    model.add(keras.layers.Dropout(0.25))
#    
#    # max pooling
#    model.add(keras.layers.MaxPooling2D())
#    
#    # 2 layers of convolution
    model.add(keras.layers.Conv2D(128, 3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(128, 3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(128, 3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
#    
#    # max pooling
#    model.add(keras.layers.MaxPooling2D())
#    
#    # 3 layers of convolution
#    model.add(keras.layers.Conv2D(256, 3, activation='relu'))
#    model.add(keras.layers.BatchNormalization())
#    model.add(keras.layers.Conv2D(256, 3, activation='relu'))
#    model.add(keras.layers.BatchNormalization())
#    model.add(keras.layers.Conv2D(256, 3, activation='relu'))
#    model.add(keras.layers.BatchNormalization())

    # max pooling
#    model.add(keras.layers.MaxPooling2D())
    
    # flatten
    model.add(keras.layers.Flatten())
    
    # fully connected layer
#    model.add(keras.layers.Dense(1024, activation='relu'))
#    model.add(keras.layers.Dense(1024, activation='relu'))
    
    # dropout
#    model.add(keras.layers.Dropout(0.9))
    
    # final dense layer
    model.add(keras.layers.Dense(1, activation='sigmoid', 
#                                 kernel_regularizer=regularizers.l2(0.025), 
#                                 activity_regularizer=regularizers.l1(0.025)
                                 ))
    
#    model = keras.applications.Xception(weights = "imagenet", include_top=False, input_shape=(imgSize, imgSize, 3))
#    for layer in model.layers[:36]:
#        layer.trainable=False
#    x = model.output
#    x = Flatten()(x)
#    predictions = Dense(2, activation='softmax')(x)
##   predictions = Dense(1, activation='sigmoid')(x)
#    model = Model(inputs = model.input, outputs = predictions)
#    model.summary()
    # multiple GPUs
#    model = multi_gpu_model(model, gpus=16)
    
#    # resume from checkpoint
#    savedModelFiles = find_files(pathBase, '2019-02-07--*.hdf5')
#    if len(savedModelFiles) > 0:
#        if len(savedModelFiles) > 1:
#            print('Error: There are multiple saved model files.')
#            return
#        print("Resumed model's weights from {}".format(savedModelFiles[-1]))
#        # load weights
#        model.load_weights(os.path.join(pathBase, savedModelFiles[-1]))

    # compile
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-8), 
                  loss=keras.losses.binary_crossentropy, 
#                  loss=keras.losses.sparse_categorical_crossentropy, 
                  metrics=['acc'])
    
    return model

#def sendEmail():
#    import yagmail
#    
#    receiver = "nsambhu@mail.usf.edu"
#    
#    yag = yagmail.SMTP("neilmsambhu@gmail.com")
#    yag.send(
#        to=receiver,
#        subject="AWS CNN Finished"
#    )
#    yagmail.SMTP('neilmsambhu@gmail.com', 'NeilSambhu123').send('nsambhu@mail.usf.edu', 'test')
    
if __name__ == "__main__":
    pathBase = '../data/FaceForensics_selfreenactment_images/'
    initialFileRead = True
    print('Image reading started at {}'.format(str(datetime.datetime.now())))
    test_x = None
    test_y = None
    train_x = None
    train_y = None
    val_x = None
    val_y = None
    if initialFileRead:
        test_x, test_y, train_x, train_y, val_x, val_y = readImages(pathBase)
        np.save('{}test_x_{}'.format(pathBase, imgSize), test_x)
        np.save('{}test_y_{}'.format(pathBase, imgSize), np.array(test_y, dtype='uint8'))
        np.save('{}train_x_{}'.format(pathBase, imgSize), train_x)
        np.save('{}train_y_{}'.format(pathBase, imgSize), np.array(train_y, dtype='uint8'))
        np.save('{}val_x_{}'.format(pathBase, imgSize), val_x)
        np.save('{}val_y_{}'.format(pathBase, imgSize), np.array(val_y, dtype='uint8'))
    else:
        test_x = np.load('{}test_x_{}.npy'.format(pathBase, imgSize))
        test_y = np.load('{}test_y_{}.npy'.format(pathBase, imgSize))
        train_x = np.load('{}train_x_{}.npy'.format(pathBase, imgSize))
        train_y = np.load('{}train_y_{}.npy'.format(pathBase, imgSize))
        val_x = np.load('{}val_x_{}.npy'.format(pathBase, imgSize))
        val_y = np.load('{}val_y_{}.npy'.format(pathBase, imgSize))
    print('Image reading finished at {}'.format(str(datetime.datetime.now())))
#    os.system('shutdown -s')
    
    print('Class balance started at {}'.format(str(datetime.datetime.now())))
    unique, counts = np.unique(test_y, return_counts=True)
    print('test_y: {}'.format(dict(zip(unique, counts))))
    unique, counts = np.unique(train_y, return_counts=True)
    print('train_y: {}'.format(dict(zip(unique, counts))))
    unique, counts = np.unique(val_y, return_counts=True)
    print('val_y: {}'.format(dict(zip(unique, counts))))
    print('Class balance finished at {}'.format(str(datetime.datetime.now())))

    print('Model building started at {}'.format(str(datetime.datetime.now())))
    keras.backend.clear_session()
    model = buildModel(pathBase)
    print('Model building finished at {}'.format(str(datetime.datetime.now())))
    
    print('Model evaluation started at {}'.format(str(datetime.datetime.now())))
    # fit model to data
    time = strftime("%Y-%m-%d--%H-%M-%S", gmtime())
    checkpoint = ModelCheckpoint('{0}{1}_{{epoch:02d}}-{{val_acc:.2f}}.hdf5'.format(pathBase, time), 
								 monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    earlyStop = EarlyStopping('val_acc',0.01,1)
    callbacks_list = [checkpoint, earlyStop]
    model.fit(x=train_x, y=train_y, batch_size=64, epochs=10, verbose=2, 
              callbacks=callbacks_list,
              validation_data=(val_x, val_y),
              initial_epoch=0)    
    print(model.evaluate(test_x, test_y))
    test_y_prob = model.predict(test_x)
    test_y_pred = np.round(test_y_prob)
#    test_y_pred = np.argmax(test_y_prob, axis=-1)
    print('Confusion matrix:\n{}'.format(confusion_matrix(test_y, test_y_pred)))
    print('Model evaluation finished at {}'.format(str(datetime.datetime.now())))