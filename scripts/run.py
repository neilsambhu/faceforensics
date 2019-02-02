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
from keras.applications import Xception
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import sklearn
import datetime
from tqdm import tqdm
import tensorflow as tf
import glob
import fnmatch

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
		fileBadImages = open('../data/FaceForensics_selfreenactment_images/test/{0}-BadImagesOriginal.txt'.format(str(datetime.datetime.now())), 'w+')
	elif label is 1:
		fileBadImages = open('../data/FaceForensics_selfreenactment_images/test/{0}-BadImagesAltered.txt'.format(str(datetime.datetime.now())), 'w+')
	else:
		print('Error: label should be 0 or 1')
		return
	countBadImages = 0
	for file in tqdm(os.listdir(directory)):
		if file.endswith('.png'):
			path = os.path.join(directory, file)
			img = cv2.imread(path)
			if img is None:
				#fileBadImages.write(file + '\n')
				countBadImages += 1
				pass
			else:
				x.append(cv2.resize(img,(256,256)))
				y.append(label)
	print('Bad images count: {}'.format(countBadImages))
	return x, y

# preprocessing
def readImages(pathData):
	# get test data
	pathTestOriginal = '{}test/original/'.format(pathData)
	x_TestOriginal, y_TestOriginal = directorySearch(pathTestOriginal, 0)
	if len(x_TestOriginal) != len(y_TestOriginal):
		print('Error: x_TestOriginal length does not equal y_TestOriginal length')
	pathTestAltered = '{}test/altered/'.format(pathData)
	x_TestAltered, y_TestAltered = directorySearch(pathTestAltered, 1)
	if len(x_TestAltered) != len(y_TestAltered):
		print('Error: x_TestAltered length does not equal y_TestAltered length')

	# get train data
	pathTrainOriginal = '{}train/original/'.format(pathData)
	x_TrainOriginal, y_TrainOriginal = directorySearch(pathTrainOriginal, 0)
	if len(x_TrainOriginal) != len(y_TrainOriginal):
		print('Error: x_TrainOriginal length does not equal y_TrainOriginal length')
	pathTrainAltered = '{}train/altered/'.format(pathData)
	x_TrainAltered, y_TrainAltered = directorySearch(pathTrainAltered, 1)
	if len(x_TrainAltered) != len(y_TrainAltered):
		print('Error: x_TrainAltered length does not equal y_TrainAltered length')

	# get val data
	pathValOriginal = '{}val/original/'.format(pathData)
	x_ValOriginal, y_ValOriginal = directorySearch(pathValOriginal, 0)
	if len(x_ValOriginal) != len(y_ValOriginal):
		print('Error: x_ValOriginal length does not equal y_ValOriginal length')
	pathValAltered = '{}val/altered/'.format(pathData)
	x_ValAltered, y_ValAltered = directorySearch(pathValAltered, 1)
	if len(x_ValAltered) != len(y_ValAltered):
		print('Error: x_ValAltered length does not equal y_ValAltered length')

	#x_TestOriginal, y_TestOriginal = sklearn.utils.shuffle(x_TestOriginal, y_TestOriginal)
	#x_TestAltered, y_TestAltered = sklearn.utils.shuffle(x_TestAltered, y_TestAltered)
	
	## setup training data
	#train_x = np.asarray(x_TestOriginal[0:9*len(x_TestOriginal)//10] + x_TestAltered[0:9*len(x_TestAltered)//10])
	#train_y = np.asarray(y_TestOriginal[0:9*len(y_TestOriginal)//10] + y_TestAltered[0:9*len(y_TestAltered)//10])
	#train_x, train_y = sklearn.utils.shuffle(train_x, train_y)
	
	## setup testing data
	#test_x = np.asarray(x_TestOriginal[9*len(x_TestOriginal)//10:-1] + x_TestAltered[9*len(x_TestAltered)//10:-1])
	#test_y = np.asarray(y_TestOriginal[9*len(y_TestOriginal)//10:-1] + y_TestAltered[9*len(y_TestAltered)//10:-1])
	#test_x, test_y = sklearn.utils.shuffle(test_x, test_y)

	# shuffle data
	#x_TestOriginal, y_TestOriginal = sklearn.utils.shuffle(x_TestOriginal, y_TestOriginal)
	#x_TestAltered, y_TestAltered = sklearn.utils.shuffle(x_TestAltered, y_TestAltered)
	#x_TrainOriginal, y_TrainOriginal = sklearn.utils.shuffle(x_TrainOriginal, y_TrainOriginal)
	#x_TrainAltered, y_TrainAltered = sklearn.utils.shuffle(x_TrainAltered, y_TrainAltered)
	#x_ValOriginal, y_ValOriginal = sklearn.utils.shuffle(x_ValOriginal, y_ValOriginal)
	#x_ValAltered, y_ValAltered = sklearn.utils.shuffle(x_ValAltered, y_ValAltered)

#    print('Length of train_x: {}'.format(len(train_x)))
#    print('Length of train_y: {}'.format(len(train_y)))
#    print('Length of test_x: {}'.format(len(test_x)))
#    print('Length of test_y: {}'.format(len(test_y)))
	
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
	model.add(keras.layers.Conv2D(4, 3, activation='relu', input_shape=(256,256,3)))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Conv2D(4, 3, activation='relu'))
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
	#model = multi_gpu_model(model, gpus=8)
	
	# resume from checkpoint
	savedModelFiles = find_files(pathBase, '*.hdf5')
	if len(savedModelFiles) > 0:
		if len(savedModelFiles) > 1:
			print('Error: There are multiple saved model files.')
			return
		print("Resumed model's weights from {}".format(savedModelFiles[0]))
		# load weights
		model.load_weights(savedModelFiles[0])

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
	checkpoint = ModelCheckpoint('{}{epoch:02d}-{val_acc:.2f}.hdf5'.format(pathBase), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	earlyStop = EarlyStopping('loss',0.0001,2)
	callbacks_list = [checkpoint, earlyStop]
	model.fit(x=train_x, y=train_y, batch_size=16, epochs=50, verbose=2, 
			  callbacks=callbacks_list,
			  validation_data=(val_x, val_y),
			  initial_epoch=0)	
	print(model.evaluate(test_x, test_y))
	print('Model evaluation finished at {}'.format(str(datetime.datetime.now())))

if __name__ == "__main__":
	main()
	print('Finished at {}'.format(str(datetime.datetime.now())))