import csv
import random 
import math

# HANDLE DATA
def loadCsv(filename):
    lines = csv.reader(open(filename,'r'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

# #test read file
# filename = 'pima-indians-diabetes.data.csv'
# dataset = loadCsv(filename)
# print(('Loaded data file {0} with {1} rows').format(filename,len(dataset)))

# Split the dataset to train and test
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset)*splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]
# #test a mock dataset with 5 instances
# dataset = [[1],[2],[3],[4],[5]]
# splitRatio = 0.6666666666666666666667
# train, test = splitDataset(dataset, splitRatio)
# print(('Split {0} rows into train with {1} and test with {2}').format(len(dataset), train, test))

#SUMMARIZE DATA
# 1. Separate Data by Class
# 2. Calculate Mean
# 3. Calculate Standard Deviation
# 4. Summarize Dataset 
# 5. Summarize Attributes by Class 

# Separate Data by Class
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated
# #test separate data
# dataset = [[1,20,1],[2,21,0],[3,22,1]]
# separated = separateByClass(dataset)
# print(('Separated instances: {0}').format(separated))

#Calculate Mean
def mean(numbers):
    return sum(numbers)/float(len(numbers))
def standardDeviation(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)
# #test by taking mean from 1 to 5
# numbers = [1,2,3,4,5]
# print(('Summary of {0}: mean = {1}, standard deviation = {2}').format(numbers,mean(numbers),standardDeviation(numbers)))

#Summarize Dataset
def summarize(dataset):
    summaries = [(mean(attribute), standardDeviation(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries
# #test with the first and second data attributes
# dataset = [[1,20,1],[2,21,0],[3,22,0]]
# summary = summarize(dataset)
# print(('Attribute summaries: {0}').format(summary))

#Summarize Attributes by Class
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries
# #test with small test dataset
# dataset = [[1,20,1],[2,21,0],[3,22,1],[4,22,0]]
# summary = summarizeByClass(dataset)
# print(('Summary by class value: {0}').format(summary))

#MAKE PREDICTION
#1. Calculate Gaussian Probability Density Function
#2. Calculate Class Probabilities
#3. Make a Prediction
#4. Estimate Accuracy

#Calculate Gaussian Probability Density Function
def calculateProbability(x, mean, standardDeviation):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(standardDeviation,2))))
    return (1/(math.sqrt(2*math.pi)*standardDeviation))*exponent
# #test with sample data
# x= 71.5
# mean = 73
# standardDeviation = 6.2
# probability = calculateProbability(x, mean, standardDeviation)
# print(('Probability of belonging to this class: {0}').format(probability))

#Calculate Class Probabilities
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] =1
        for i in range(len(classSummaries)):
            mean, standardDeviation = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x,mean,standardDeviation)
    return probabilities
# #test
# summaries = {0:[(1,0.5)],1:[(20,0.5)]}
# inputVector = [1.1,'?']
# probabilities = calculateClassProbabilities(summaries, inputVector)
# print(('Probabilities for each class: {0}').format(probabilities))

#Make a prediction
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries,inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel
# #test
# summaries = {'A':[(1,0.5)],'B':[(20,5.0)]}
# inputVector = [1.1,'?']
# result = predict(summaries, inputVector)
# print(('Prediction: {0}').format(result))

#MAKE PREDICTIONS
def getPrediction(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions
# #test
# summaries = {'A':[(1,0.5)],'B':[(20,0.5)]}
# testSet = [[1.1,'?'],[19.1,'?']]
# predictions = getPrediction(summaries, testSet)
# print(('Predictions: {}').format(predictions))

#GET ACCURACY
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0
# #test
# testSet = [[1,1,1,'a'],[2,2,2,'a'],[3,3,3,'b']]
# predictions = ['a','a','a']
# accuracy = getAccuracy(testSet, predictions)
# print(('Accuracy: {0}').format(accuracy))

#MAIN
def main():
    filename = 'pima-indians-diabetes.data.csv'
    splitRatio = 0.7
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset,splitRatio)
    print(('Split {0} rows into train = {1} and test = {2} rows').format(len(dataset),len(trainingSet), len(testSet)))
    #prepare model
    summaries = summarizeByClass(trainingSet)
    #test model
    predictions = getPrediction(summaries,testSet)
    accuracy = getAccuracy(testSet, predictions)
    print(('Accuracy: {0}%').format(accuracy))
main()