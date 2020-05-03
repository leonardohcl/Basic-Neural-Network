# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import random
import wineMethods as wm
import irisMethods as im
import neuralNetworkMethods as nnm

dataset = 0 #0 - Iris | 1 - Wine
outs = 3
maxIt = 1000
maxError = 0
trainingAmount = 0.7
validationAmount = 0.15
learningRate = 0.01

#Lê dados do arquivo
if dataset == 0:
	print("Lendo dados... Iris")
	data = im.getIrisDataFromFile('iris.data')
else:
	print("Lendo dados... Wine")
	data = wm.getWineDataFromFile('wine.data')

#Conta o número de entradas (-1 devido a classificação ser um dos dados)
entries = len(data[0]) - 1

#Embaralha ordem do arquivo
print("Embaralhando dados...")
np.random.shuffle(data)

print("Preparando matrizes...")
#Separa os amontoados de dados
size = len(data)
endTraining = int(size * trainingAmount)
endValidation = endTraining + int(size * validationAmount)
trainingSet = data[:endTraining]
validationSet = data[endTraining:endValidation]
testSet = data[endValidation:]
print("Divisao de dados:")

if dataset == 0:
	counter = im.countItemsInSet(trainingSet)
else: 
	counter = wm.countItemsInSet(trainingSet)
print("Treinamento: "+ str(len(trainingSet))+' | 0 - '+str(counter[0])+' | 1 - '+str(counter[1])+' | 2 - '+str(counter[2]))
if dataset == 0:
	counter = im.countItemsInSet(validationSet)
else: 
	counter = wm.countItemsInSet(validationSet)
print("Validacao: "+ str(len(validationSet))+' | 0 - '+str(counter[0])+' | 1 - '+str(counter[1])+' | 2 - '+str(counter[2]))
if dataset == 0:
	counter = im.countItemsInSet(testSet)
else: 
	counter = wm.countItemsInSet(testSet)
print("Testes: "+ str(len(testSet))+' | 0 - '+str(counter[0])+' | 1 - '+str(counter[1])+' | 2 - '+str(counter[2]))

# Cria matriz de pesos
W = np.zeros((outs, entries + 1))

#Vetor para o erro de cada geração
errors = []
trainingErrors = []

#Controle para exibição do feedback da porcentagem de execução
feedback = 0.0 

#Marca o menor erro como infinito e inicia o melhor W como o zerado
lowestError = float('inf')
bestW = W
bestIt = 0

for i in range(maxIt):

	#Controle de feedback de porcentagem de execução
	prct = float(i)/maxIt
	if(prct >= feedback):
		print('Treinando: '+str(prct*100)+'%')
		feedback += 0.1

	#Inicia erro como 0
	itError = 0.0
	#Passa por todo o conjunto de treinamento
	for j in range(len(trainingSet)):
		#Inicia matriz de uma coluna com as entradas com o primeiro elemento como 1
		if dataset == 0:
			entryList = im.getEntryWithoutClass(trainingSet[j])
		else:
			entryList = wm.getEntryWithoutClass(trainingSet[j])
		entry = np.array([1]+entryList)
		entry = entry.reshape(entry.shape[0],1)

		#Calcula o resultado para a entrada
		results = np.matmul(W,entry)
		#Aplica o softmax para o valor ficar entre 0 e 1
		results = nnm.softmax(results)

		#O esperado para a saída final
		expected = np.zeros((3,1))
		if dataset == 0:
			expected[im.getEntryIndex(trainingSet[j])][0] = 1.0
		else:
			expected[wm.getEntryIndex(trainingSet[j])][0] = 1.0

		#Aprende com os resultados		
		[W, error] = nnm.learn(results, expected, entry, W, learningRate)
		itError += error
	
	#Calcula e armazena a média de erro quadrático médio para a iteração
	itError = itError/len(trainingSet)
	trainingErrors.append(itError)

	#Inicia erro como 0
	itError = 0.0
	#Passa pelo conjunto de validação
	for j in range(len(validationSet)):
		#Inicia matriz de uma coluna com as entradas com o primeiro elemento como 1
		if dataset == 0:
			entryList = im.getEntryWithoutClass(validationSet[j])
		else:
			entryList = wm.getEntryWithoutClass(validationSet[j])

		entry = np.array([1]+entryList)
		entry = entry.reshape(entry.shape[0],1)

		#Calcula o resultado para a entrada
		results = np.matmul(W,entry)

		#Aplica o softmax para o valor ficar entre 0 e 1
		results = nnm.softmax(results)

		#O esperado para a saída final
		expected = np.zeros((3,1))
		if dataset == 0:
			expected[im.getEntryIndex(validationSet[j])][0] = 1.0
		else:
			expected[wm.getEntryIndex(validationSet[j])][0] = 1.0

		#Calcula o erro e o soma ao total da iteração
		entryErrors = nnm.getOutputErrors(expected, results)
		entryError = nnm.getMeanSquaredError(entryErrors)
		itError += entryError

	#Calcula e armazena a média de erro quadrático médio para a iteração
	itError = itError/len(validationSet)
	errors.append(itError)

	#Verifica se é o menor erro encontrado para salvar W
	if itError < lowestError:
		lowestError = itError
		bestW = W 
		bestIt = i
	
	#Embaralha o conjunto de treinamento
	np.random.shuffle(trainingSet)	

	#Se o erro médio for menor ou igual ao erro limite para as iterações
	if(itError <= maxError):
		break

#Inicia matriz de confusão
confusionMatrix = np.zeros((outs, outs))
bestConfusionMatrix = np.zeros((outs, outs))

#Variável auxiliar
size = len(testSet)

#Passa pelo conjunto de validação
for i in range(size):

	#Inicia matriz de uma coluna com as entradas com o primeiro elemento como 1
	if dataset == 0:
		entryList = im.getEntryWithoutClass(testSet[i])
	else:
		entryList = wm.getEntryWithoutClass(testSet[i])

	entry = np.array([1]+entryList)
	entry = entry.reshape(entry.shape[0],1)

	#Calcula o resultado para a entrada
	results = np.matmul(W,entry)

	#Aplica o softmax para o valor ficar entre 0 e 1
	results = nnm.softmax(results)

	#Calcula o resultado para a entrada
	bestResults = np.matmul(bestW,entry)

	#Aplica o softmax para o valor ficar entre 0 e 1
	bestResults = nnm.softmax(results)

	if dataset == 0:
		classification = im.getClassification(results)
		bestClassification = im.getClassification(bestResults)
		correct = im.getEntryIndex(testSet[i])
	else:
		classification = wm.getClassification(results)
		bestClassification = wm.getClassification(bestResults)
		correct = wm.getEntryIndex(testSet[i])
		
	confusionMatrix[correct][classification] += 1
	bestConfusionMatrix[correct][bestClassification] += 1


right = 0
bestRight = 0
wrong = 0
bestWrong = 0
for i in range(outs):
	for j in range(outs):
		if i == j:
			right += confusionMatrix[i][j]
			bestRight += bestConfusionMatrix[i][j]
		else:
			wrong += confusionMatrix[i][j]
			bestWrong += bestConfusionMatrix[i][j]

print("Melhor Matriz de pesos (it:"+str(bestIt)+" | erro: "+str(lowestError)+"):")
print(bestW)

print("Matriz de confusao para a melhor matriz:")
print(bestConfusionMatrix)

bestAccr = float(right)/(right+wrong)
print("Acertos: "+ str(bestRight))
print("Erros: "+ str(bestWrong))
print("Acuracia: "+ str(bestAccr * 100)+'%')
print('--------------------')

print("Matriz de pesos final:")
print(W)

print("Matriz de confusao para a matriz final:")
print(confusionMatrix)

accr = float(right)/(right+wrong)
print("Acertos: "+ str(right))
print("Erros: "+ str(wrong))
print("Acuracia: "+ str(accr * 100)+'%')

x = range(len(errors))
plt.plot(x,errors,'-b', label="Conjunto de validacao")
plt.plot(x,trainingErrors,'-r', label="Conjunto de treinamento")
plt.ylabel('Erro Quadratico Medio')
plt.xlabel('Iteracao')
plt.legend()
plt.show()
plt.close()
