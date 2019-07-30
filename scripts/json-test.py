import json
from tqdm import tqdm
import datetime
import os
import sys
import glob

pathPNGs = open(r'pathPNGs.txt', 'w')

def VerifyDir(dir):
	if not os.path.exists(dir):
		sys.exit('Directory \'{}\' does not exist'.format(dir))

def JSON_ParserVideoSequence(pathJSON, dirVideoName, JSON_VideoSequenceNumber):
	fullPathJSON = os.path.join(os.getcwd(), pathJSON)
	# print('cwd {}, pathJSON {}'.format(os.getcwd(), pathJSON))
	# VerifyDir(fullPathJSON)
	with open(fullPathJSON) as JSON_VideoSequenceFile:
		data = json.load(JSON_VideoSequenceFile)
		# print('first {}, last {}'.format(data['first frame'], data['last frame']))
		firstFrame = data['first frame']
		lastFrame = data['last frame']
		framesCount = lastFrame - firstFrame
		for frameNumber in range(0, framesCount):
			fileNamePNG = '{}_{}_{}_{}_{}.png'.format(dirVideoName, JSON_VideoSequenceNumber, dirVideoName, JSON_VideoSequenceNumber, frameNumber)
			pathPNGs.write(fileNamePNG+'\n')

def JSON_ParserVideo(dirBase, dirVideoName):
	dirJSON = os.path.join(dirBase, dirVideoName, 'faces')
	# VerifyDir(dirJSON)
	JSON_Files = glob.glob(os.path.join(dirJSON, '*.json'))
	for JSON_VideoSequence in JSON_Files:
		JSON_VideoSequence = os.path.basename(JSON_VideoSequence)
		pathJSON = os.path.join(dirJSON, JSON_VideoSequence)
		# print('dirJSON {}, JSON_VideoSequence {}'.format(dirJSON, JSON_VideoSequence))
		# VerifyDir(pathJSON)
		JSON_ParserVideoSequence(pathJSON, dirVideoName, os.path.splitext(JSON_VideoSequence)[0])

def JSON_Parser(dirBase='Face2Face_video_information'):
	# VerifyDir(dirBase)
	for dirVideoName in os.listdir(dirBase):
		JSON_ParserVideo(dirBase, dirVideoName)

if __name__ == "__main__":
	JSON_Parser()
