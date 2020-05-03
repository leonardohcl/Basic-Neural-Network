# encoding: utf-8
import math

def getWineDataFromFile(filename):
	try:
		f = open(filename)
	except:
		return []
	data = []
	x = f.readline()

	while x != '':		
		x = x.rstrip()
		arr = x.split(',')
		for i in range(len(arr)):
			arr[i] = float(arr[i])

		data.append(arr)
		x = f.readline()
	f.close()

	data = applyZscore(data)
	for i in range(len(data)):
		data[i][0] = int(data[i][0])

	return data

def getClassification(results):
	resultList = results.flatten().tolist()
	return resultList.index(max(resultList))

def applyZscore(entrys):
	means = getColMeans(entrys)
	devs = getColDevs(entrys, means)
	lines = len(entrys)
	cols = len(entrys[0])
	for i in range(lines):
		for j in range(1,cols,1):
			entrys[i][j] = (entrys[i][j] - means[j])/devs[j]

	return entrys

def getColMeans(matrix):
	cols = len(matrix[0])
	lines = len(matrix)
	colsSummed = cols* [0]
	means= cols * [0]
	for i in range(lines):
		for j in range(cols):
			colsSummed[j] += matrix[i][j]
	for i in range(cols):
		means[i] = colsSummed[i] / lines

	return means

def getColDevs(matrix, means):
	cols = len(matrix[0])
	lines = len(matrix)
	devs = cols * [0]
	difSquaredSum = cols * [0]
	for i in range(lines):
		for j in range(cols):
			difSquaredSum[j] += (matrix[i][j] - means[j])**2
	
	for i in range(cols):
		devs[i] = difSquaredSum[i] / lines 

	return devs

def getEntryIndex(entry):
	return int(entry[0] - 1)

def getEntryCode(entry):
	return entry[0]

def getEntryWithoutClass(entry):	
	return entry[1:]	

def countItemsInSet(dataset):
	counter = [0,0,0]
	for i in range(len(dataset)):
		counter[getEntryIndex(dataset[i])] += 1	
	return counter