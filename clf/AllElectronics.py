# -*- coding: utf-8 -*-

from sklearn.feature_extraction import  DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree

# Read the csv file and put features in a list of dict and list of class label
# read by line 
# the header of the data in the csv:RID,age,income...

def CreateData():
    reader=csv.reader(file("./data/AllElectronics.csv","rb"))
    headers = reader.next()
    
    featureList=[]
    labelList=[]
    
    for row in reader:
        labelList.append(row[-1])
        rowDict={}
        for i in range(1,len(row)-1):
            rowDict[headers[i]]=row[i]
            
        featureList.append(rowDict)
        
    # Vectorize features 
    vec=DictVectorizer()
    dummyX=vec.fit_transform(featureList).toarray()
    
    #Vectorize class labels 
    lb=preprocessing.LabelBinarizer()
    dummyY=lb.fit_transform(labelList)
    
    return dummyX,dummyY,vec


# Using decison tree for classification 

def clfDecision(features,labelList):
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(features,labelList)
    return clf 
    
# write out the decison tree 
def writeDot(clf,vec):
    with open("./data/allElectronicInformaationGainOri.dot",'w') as f:
        f=tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)

def run():
    features,labelList,vec=CreateData()
    clf = clfDecision(features,labelList)
    writeDot(clf,vec)
		
if __name__=='__main__':
    run()
    





    
    