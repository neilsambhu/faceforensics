import json
from tqdm import tqdm
import datetime
import os
import sys
import glob
import keras
from keras.models import load_model
import cv2
import numpy as np
from statistics import mode
from sklearn.metrics import confusion_matrix

pathPNGs = open(r'pathPNGs.txt', 'w')
errorMessages = open(r'errorMessages.txt', 'w')
outputFile = open(r'outputFile.txt', 'w')
model = load_model('../data/FaceForensics_selfreenactment_images/2019-06-29--04-48-44_03-0.99.hdf5')
imgSize = 128
test_y_all_groundTruth = []
test_y_all_pred = []

def VerifyDir(dir):
    if not os.path.exists(dir):
        sys.exit('Directory \'{}\' does not exist'.format(dir))

def JSON_ParserVideoSequence(pathJSON, dirVideoName, JSON_VideoSequenceNumber):
    fullPathJSON = os.path.join(os.getcwd(), pathJSON)
    # VerifyDir(fullPathJSON)
    with open(fullPathJSON) as JSON_VideoSequenceFile:
        data = json.load(JSON_VideoSequenceFile)
        firstFrame = data['first frame']
        lastFrame = data['last frame']
        framesCount = lastFrame - firstFrame

        test_x_altered = []
        test_x_original = []
        test_y_altered_pred = []
        test_y_original_pred = []

        for frameNumber in range(0, framesCount):
            fileNamePNG = '{}_{}_{}_{}_{}.png'.format(dirVideoName, 
                           JSON_VideoSequenceNumber, dirVideoName, 
                           JSON_VideoSequenceNumber, frameNumber)
            # pathPNGs.write(fileNamePNG+'\n')
            fullFileNamePNGs = glob.glob(
                    os.path.join('..', 'data', 
                                 'FaceForensics_selfreenactment_images', 
                                 'test', 
                                 '*', fileNamePNG))
            if len(fullFileNamePNGs) is 2:
               # ready to check model against 2 complementary images
               # process altered
               pathImgAltered = fullFileNamePNGs[0]
               test_x_altered.append(cv2.resize(cv2.imread(pathImgAltered, cv2.IMREAD_COLOR),(imgSize,imgSize)))
               # process original
               pathImgOriginal = fullFileNamePNGs[1]
               test_x_original.append(cv2.resize(cv2.imread(pathImgOriginal, cv2.IMREAD_COLOR),(imgSize,imgSize)))
        if len(test_x_altered) > 0:
            # establish ground truth
            test_y_all_groundTruth.append(1)
            test_y_all_groundTruth.append(0)
            # get predictions from model
            test_y_altered_pred.append(np.round(model.predict(test_x_altered)))
            test_y_original_pred.append(np.round(model.predict(test_x_original)))
            # majority voting on video
            test_y_all_pred.append(mode(test_y_altered_pred))
            test_y_all_pred.append(mode(test_y_original_pred))

def JSON_ParserVideo(dirBase, dirVideoName):
    dirJSON = os.path.join(dirBase, dirVideoName, 'faces')
    # VerifyDir(dirJSON)
    JSON_Files = glob.glob(os.path.join(dirJSON, '*.json'))
    for JSON_VideoSequence in JSON_Files:
        JSON_VideoSequence = os.path.basename(JSON_VideoSequence)
        pathJSON = os.path.join(dirJSON, JSON_VideoSequence)
        # print('dirJSON {}, JSON_VideoSequence {}'.format(dirJSON, JSON_VideoSequence))
        # VerifyDir(pathJSON)
        JSON_ParserVideoSequence(pathJSON, dirVideoName, 
                                 os.path.splitext(JSON_VideoSequence)[0])

def JSON_Parser(dirBase='Face2Face_video_information'):
    # VerifyDir(dirBase)
    for dirVideoName in tqdm(os.listdir(dirBase)):
        JSON_ParserVideo(dirBase, dirVideoName)
    print('Confusion matrix:\n{}'.format(confusion_matrix(test_y_all_groundTruth, test_y_all_pred)))

if __name__ == "__main__":
    JSON_Parser()
    # os.system('shutdown -s')