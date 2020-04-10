import cv2
import json
import numpy as np
import random
import glog as log
import math
import time
from sklearn.metrics.pairwise import cosine_similarity
import copy

class Point:
	def __init__(self,_x, _y, _class, _color):
		self.x_ = _x
		self.y_ = _y
		self.class_ = _class
		self.color_ = _color

def loadJson(path):
	file = open(path, 'r')
	jsonData = file.read()
	file.close()
	data = json.loads(jsonData)
	return data
def L2Distance(featureList):
    featureListLength = len(featureList)
    featureList = featureList.reshape(featureListLength, -1)
    featureLength = featureList.shape[1]
    print(featureList.shape)
    featureListTile = np.tile(featureList, featureListLength)
    print(featureListTile.shape)

    featureListRepeat = featureList.reshape(1,-1)
    featureListRepeat = np.repeat(featureListRepeat, featureListLength, axis=0)
    subArray = featureListRepeat - featureListTile
    subArray = subArray.reshape(featureListLength, featureLength,-1)
    print(subArray.shape)
    distanceList = np.linalg.norm(subArray,axis=1)
    print(distanceList.shape)
    return distanceList


def cosSimilarity(featureList):
	# featureList = np.array(featureList)
	similarityList = np.zeros(shape=(len(featureList), len(featureList )), dtype='float32')
	for i in range(len(featureList)):
		feature1 = np.array(featureList[i])
		similarityListPart = {}
		log.info('getting similarity %d'%i)
		for j in range(len(featureList)):
			feature2 = np.array(featureList[j])
			# similarity = np.linalg.norm(feature1 - feature2, axis = 0)
			similarity = cosine_similarity([feature1, feature2])[0][1]
			similarityList[i][j] = similarity
	return similarityList

def drawCircle(frame, num, pointList):
	frame[int(frame.shape[0]/2),int(frame.shape[1]/2)]=[254,125,8]
	for i in range(num):
		cv2.circle(frame, (pointList[i].x_,pointList[i].y_),10,tuple(pointList[i].color_),-1)
	return frame

def generatePoints(frame, num):
	frame[int(frame.shape[0]/2),int(frame.shape[1]/2)]=[254,125,8]
	radius = min(frame.shape[0], frame.shape[1])/3
	print('radius: %f'%radius)
	theta = 2*math.pi / num
	pointList = []
	for i in range(num):
		x = int(frame.shape[1] / 2 + int(math.cos(i * theta) * radius))
		y = int(frame.shape[0] / 2 + int(math.sin(i * theta) * radius))
		r = colorTabel[i][2]
		g = colorTabel[i][1]
		b = colorTabel[i][0]
		point = Point(_x=x,_y=y,_color=[b,g,r],_class= i)
		pointList.append(point)
	return pointList

def generateColor(num):
	colorTabel = []
	for i in range(num):
		r = random.randint(0,255)
		time.sleep(0.001)
		g = random.randint(0,255)
		time.sleep(0.001)
		b = random.randint(0,255)
		time.sleep(0.001)
		colorTabel.append([b,g,r])
	return colorTabel

def update(frame, similarityList, pointList, thresh, colorTabel, num):
	frameCopy = frame
	pointListCopy = copy.copy(pointList)
	for i in range(len(similarityList)):
		classSimilarity = {} 
		for j in range(len(similarityList[i])):
			if j == i:
				continue
			if similarityList[i][j] > thresh:
				cv2.line(frameCopy, (pointList[i].x_, pointList[i].y_),(pointList[j].x_, pointList[j].y_),(233,7,7),1)
				# count all the neighbor's class similarity
				if pointList[j].class_ in classSimilarity.keys():
					classSimilarity[pointList[j].class_]+=similarityList[i][j]
				else:
					classSimilarity[pointList[j].class_]=similarityList[i][j]

			id = pointListCopy[i].class_
			if len(classSimilarity.keys()) > 0:
				id = max(classSimilarity, key=classSimilarity.get)
			pointListCopy[i].class_ = id
			pointListCopy[i].color_ = colorTabel[id]
	frameCopy = drawCircle(frameCopy, num, pointListCopy)
	cv2.imshow('frame', frameCopy)
	cv2.waitKey(0)
	return pointListCopy

jsonData = loadJson('feature.json')

width = 1920
height = 1080
frame = np.zeros(shape=(height,width,3),dtype=np.uint8)

featureList = []
for name in jsonData:
	for i in range(len(jsonData[name])):
		featureList.append(jsonData[name][i])

num = len(featureList)
similarityList = cosSimilarity(featureList)

colorTabel = generateColor(num)

pointList = generatePoints(frame, num)
frameCopy = drawCircle(frame, num, pointList)

cv2.imshow('frame', frameCopy)
cv2.waitKey(0)

thresh = 0.8


for i in range(5):
	pointList = update(frame, similarityList, pointList, thresh, colorTabel, num)
