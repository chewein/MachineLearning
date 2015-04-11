# -*- coding: utf-8 -*-

from sklearn.feature_extraction import  DictVectorizer 
import csv
import math
import operator

# Implementation the ID3 algorithm 

def createdataSet():
    '''
	create Data :1.read from the csv file;2 transform into digit value from string
	'''
    reader=csv.reader(file("./data/AllElectronics.csv","rb"))
    headers=reader.next()
    
    dataSet=[]

    for row in reader:
        rowDict={}
        for i in range(1,len(row)):
            rowDict[headers[i]]=row[i]
        dataSet.append(rowDict)
	
    vec = DictVectorizer()
    dataSet=vec.fit_transform(dataSet)
    
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]  
    features = ['no surfacing','flippers']
    
    return dataSet,features


def treeGrowth(dataSet,headers):
    '''
    the main recursion 
    '''
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList): #all example are the same
        return classList[0]
    if len(dataSet[0])==1: #examples only have one attribute  
        return classify(classList)


    bestAttriIndex = findBestAttri(dataSet)
    bestHeader=headers[bestAttriIndex]
    myTree={bestHeader:{}}
    
    bestAttriValue=[example[bestAttriIndex] for example in dataSet]
    uiquebestAttriValue=set(bestAttriValue)  
    del(headers[bestAttriIndex])
    
    # build sub tree 
    for values in uiquebestAttriValue:
        subDataSet=splitDataSet(dataSet,bestAttriIndex,values)
        myTree[bestHeader][values]=treeGrowth(subDataSet,headers)
    headers.insert(bestAttriIndex,bestHeader)
    return myTree
  
# examples may more than one but only have one attri
def classify(classList):  
    '''
    find the most in the set 
    '''  
    classCount = {}  
    for vote in classList:  
        if vote not in classCount.keys():  
            classCount[vote] = 0  
        classCount[vote] += 1  
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True)  
    return sortedClassCount[0][0] 


def findBestAttri(dataset):  
    numFeatures = len(dataset[0])-1  
    baseEntropy = calclShannonEnt(dataset)  
    bestInfoGain = 0.0  
    bestFeat = -1  
    for attriIndex in range(numFeatures):  
        featValues = [example[attriIndex] for example in dataset]  
        uniqueFeatValues = set(featValues)  
        newEntropy = 0.0  
        for val in uniqueFeatValues:  
            subDataSet = splitDataSet(dataset,attriIndex,val)  
            prob = len(subDataSet)/float(len(dataset))  
            newEntropy += prob*calclShannonEnt(subDataSet)  
        if(baseEntropy - newEntropy)>bestInfoGain:  
            bestInfoGain = baseEntropy - newEntropy  
            bestFeat = attriIndex  
    return bestFeat 
    

# 
def splitDataSet(dataset,attriIndex,values):  
    retDataSet = []  
    for featVec in dataset: 
        if featVec[attriIndex] == values:  
            reducedFeatVec = featVec[:attriIndex]  
            reducedFeatVec.extend(featVec[attriIndex+1:])  
            retDataSet.append(reducedFeatVec)  
    return retDataSet    


# calculation the Shannon Entropy
def calclShannonEnt(dataSet):
    numEntry = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
            labelCounts[currentLabel] +=1
	
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntry
        if prob != 0:
            shannonEnt -=prob*math.log(prob,2)
    
    return shannonEnt
	
def predict(tree,newObject):  
    while isinstance(tree,dict):  
        key = tree.keys()[0]  
        tree = tree[key][newObject[key]]  
    return tree  

def run():
   dataSet,headers= createdataSet()
   mytree=treeGrowth(dataSet,headers)
   return mytree

def test(mytree):
   print predict(mytree,{'no surfacing':1,'flippers':1})  
   print predict(mytree,{'no surfacing':1,'flippers':0})  
   print predict(mytree,{'no surfacing':0,'flippers':1})  
   print predict(mytree,{'no surfacing':0,'flippers':0}) 
	
	
	
if __name__ == '__main__':
    mytree = run()
    print mytree
    test(mytree)


