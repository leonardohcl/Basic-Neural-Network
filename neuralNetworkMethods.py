# encoding: utf-8

import math
import numpy as np

def softmax(results):
	#Transforma os resultados de uma matriz de uma coluna para uma lista com uma linha
	resultList = results.flatten().tolist()
	#Encontra o exponencial de cada resultado
	expResults = []
	for i in range(len(resultList)):
		try:
			expResults.append(math.exp(resultList[i]))
		except OverflowError:
    			expResults.append(float('inf'))
	#Faz o somatório do exponencial de cada resultado
	expResultsSum = sum(expResults)
	softmaxResults = []
	for i in range(len(expResults)):
		if math.isinf(expResults[i]):
			division = 1.0
		elif expResultsSum == 0:
			division = 0.0
		else:
			division = expResults[i] / expResultsSum
		softmaxResults.append(division)
	
	return makeOneColumnMatrix(softmaxResults)

def makeOneColumnMatrix(itemList):
	m = []
	for i in range(len(itemList)):
		m.append([itemList[i]])
	
	return np.array(m)

def square(x):
	return x**2

def learn(result, expected, entry, W, learningRate):
	#Matriz com os erros desta entrada
	errors = getOutputErrors(expected, result)
	#Correção da matriz de pesos
	adjustment = getWeightAdjustment(learningRate, errors, entry)
	
	#Calculo do erro quadrático médio da entrada
	squaredError = getMeanSquaredError(errors)

	#Aplica a correção em W
	newW = np.add(W, adjustment)

	return [newW, squaredError]

def getWeightAdjustment(learningRate, errors, entry):
	return learningRate * (np.matmul(errors, entry.transpose()))

def getOutputErrors(expected, results):
	return np.add(expected, -results)

def getMeanSquaredError(errors):
	#Faz o somatório
	squareSum = sum(map(square, errors.flatten().tolist()))
	#Divide pelo número de entradas
	return squareSum/errors.shape[0]