import json
from tqdm import tqdm
import datetime
import os
import sys
import glob

pathPNGs = open(r'pathPNGs.txt', 'w')
errorMessages = open(r'errorMessages.txt', 'w')
outputFile = open(r'outputFile.txt', 'w')
model = load_model('../data/FaceForensics_selfreenactment_images/2019-06-29--04-48-44_03-0.99.hdf5')
test_y_all_groundTruth = []
test_y_all_pred = []

def LoadModel(pathModel):
    model = Sequential()

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

        test_y_original_groundTruth = []
        test_y_original_pred = []
        test_y_altered_groundTruth = []
        test_y_altered_pred = []

        for frameNumber in range(0, framesCount):
            fileNamePNG = '{}_{}_{}_{}_{}.png'.format(dirVideoName, 
                           JSON_VideoSequenceNumber, dirVideoName, 
                           JSON_VideoSequenceNumber, frameNumber)
#            pathPNGs.write(fileNamePNG+'\n')
            fullFileNamePNGs = glob.glob(
                    os.path.join('..', 'data', 
                                 'FaceForensics_selfreenactment_images', 
                                 'test', 
                                 '*', fileNamePNG))
           if len(fullFileNamePNGs) is 2:
               # ready to check model against 2 complementary images
               print(fullFileNamePNGs)

            # if len(fullFileNamePNGs) is not 2:
            #     errorMessages.write('Length of {} files is {}. Contents: {}\n'.format(fileNamePNG, 
            #           len(fullFileNamePNGs), fullFileNamePNGs))
            # if frameNumber is framesCount and len(fullFileNamePNGs) is 2:
            #      errorMessages.write('{} goes beyond framesCount ({}) bounds\n'.format(fileNamePNG, 
            #           framesCount))

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

if __name__ == "__main__":
    JSON_Parser()
    # os.system('shutdown -s')