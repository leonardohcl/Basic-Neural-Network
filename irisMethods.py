# encoding: utf-8

def getIrisDataFromFile(filename):
	try:
		f = open(filename)
	except:
		return []
	data = []
	x = f.readline()

	while x != '':		
		x = x.rstrip()
		arr = x.split(',')
		arr[0] = float(arr[0])
		arr[1] = float(arr[1])
		arr[2] = float(arr[2])
		arr[3] = float(arr[3])
		arr[4] = irisNameToCode(arr[4])
		arr = applyZscore(arr)
		data.append(arr)
		x = f.readline()

	f.close()
	return data

def irisNameToCode(name):
	if name == 'Iris-setosa':
		return 0
	elif name == 'Iris-versicolor':
		return 1
	elif name == 'Iris-virginica':
		return 2

def irisCodeToName(code):
	if code == 0:
		return 'Iris-setosa'
	elif code == 1:
		return 'Iris-versicolor'
	elif code == 2:
		return 'Iris-virginica'

def getClassification(results):
	resultList = results.flatten().tolist()
	return resultList.index(max(resultList))

def applyZscore(entry):	
	entry[0] = (entry[0] - 5.84)/0.83
	entry[1] = (entry[1] - 3.05)/0.43
	entry[2] = (entry[2] - 3.76)/1.76
	entry[3] = (entry[3] - 1.20)/0.76

	return entry

def getEntryIndex(entry):
	return int(entry[4])

def getEntryCode(entry):
	return entry[4]

def getEntryWithoutClass(entry):
	return entry[:4]	

def countItemsInSet(dataset):
	counter = [0,0,0]
	for i in range(len(dataset)):
		counter[getEntryIndex(dataset[i])] += 1
	
	return counter